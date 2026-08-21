/*
 * X11 WSI that presents through anlabwc's present.sock (AHB2).
 * Host allocates the AHB; Turnip imports the dmabuf. Keep the on-wire
 * structs in sync with ardesk android/app/src/main/cpp/present_share.h.
 */

#include "wsi_common_private.h"

#include "drm-uapi/drm_fourcc.h"
#include "util/macros.h"
#include "util/os_file.h"
#include "util/timespec.h"
#include "vk_util.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>
#include <xcb/xcb.h>
#ifdef VK_USE_PLATFORM_XLIB_KHR
#include <X11/Xlib-xcb.h>
#endif

#define ARDESK_MAGIC 0x32424841u /* 'AHB2' */
#define ARDESK_ALLOC 1
#define ARDESK_PRESENT 2
#define ARDESK_RELEASE 3
#define ARDESK_KIND_X11 1
#define ARDESK_FOURCC(a, b, c, d) \
   ((uint32_t)(a) | ((uint32_t)(b) << 8) | ((uint32_t)(c) << 16) | \
    ((uint32_t)(d) << 24))

struct ardesk_req {
   uint32_t magic;
   uint32_t cmd;
   uint32_t kind;
   uint32_t id;
   uint32_t w;
   uint32_t h;
   uint32_t token;
   uint32_t fourcc;
};

struct ardesk_alloc_reply {
   uint32_t status;
   uint32_t token;
   uint32_t width;
   uint32_t height;
   uint32_t stride;
   uint32_t pitch;
   uint32_t fourcc;
   uint32_t offset;
};

struct ardesk_ack {
   uint32_t status;
};

struct wsi_ardesk {
   struct wsi_interface base;
};

struct wsi_ardesk_image {
   struct wsi_image base;
   uint32_t token;
   int fd;
   bool busy_on_host;
   bool busy_on_device;
};

struct wsi_ardesk_swapchain {
   struct wsi_swapchain base;
   mtx_t sock_mutex;
   int sock;
   uint32_t window;
   uint32_t width;
   uint32_t height;
   VkFormat format;
   VkImageUsageFlags usage;
   struct wsi_ardesk_image images[0];
};

static const VkPresentModeKHR present_modes[] = {
   VK_PRESENT_MODE_IMMEDIATE_KHR,
   VK_PRESENT_MODE_MAILBOX_KHR,
   VK_PRESENT_MODE_FIFO_KHR,
};

bool
wsi_ardesk_available(void)
{
   const char *dir = getenv("XDG_RUNTIME_DIR");
   char path[108];

   if (!dir || !dir[0])
      return false;
   if (snprintf(path, sizeof(path), "%s/present.sock", dir) >= (int)sizeof(path))
      return false;
   return access(path, F_OK) == 0;
}

static int
write_all(int fd, const void *buf, size_t n)
{
   const uint8_t *p = buf;

   while (n) {
      ssize_t w = write(fd, p, n);
      if (w < 0) {
         if (errno == EINTR)
            continue;
         return -1;
      }
      p += (size_t)w;
      n -= (size_t)w;
   }
   return 0;
}

static int
read_all(int fd, void *buf, size_t n)
{
   uint8_t *p = buf;

   while (n) {
      ssize_t r = read(fd, p, n);
      if (r == 0)
         return -1;
      if (r < 0) {
         if (errno == EINTR)
            continue;
         return -1;
      }
      p += (size_t)r;
      n -= (size_t)r;
   }
   return 0;
}

static int
recv_reply_fd(int sock, void *buf, size_t n, int *out_fd)
{
   struct iovec iov = { .iov_base = buf, .iov_len = n };
   union {
      char buf[CMSG_SPACE(sizeof(int))];
      struct cmsghdr align;
   } cmsgbuf;
   struct msghdr msg = {
      .msg_iov = &iov,
      .msg_iovlen = 1,
      .msg_control = &cmsgbuf,
      .msg_controllen = sizeof(cmsgbuf),
   };
   ssize_t r;
   struct cmsghdr *cmsg;

   *out_fd = -1;
   do {
      r = recvmsg(sock, &msg, 0);
   } while (r < 0 && errno == EINTR);
   if (r != (ssize_t)n)
      return -1;

   for (cmsg = CMSG_FIRSTHDR(&msg); cmsg; cmsg = CMSG_NXTHDR(&msg, cmsg)) {
      if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS &&
          cmsg->cmsg_len >= CMSG_LEN(sizeof(int))) {
         memcpy(out_fd, CMSG_DATA(cmsg), sizeof(int));
         break;
      }
   }
   return 0;
}

static int
ardesk_connect(void)
{
   const char *dir = getenv("XDG_RUNTIME_DIR");
   struct sockaddr_un addr;
   int fd;

   if (!dir || !dir[0])
      return -1;
   fd = socket(AF_UNIX, SOCK_STREAM, 0);
   if (fd < 0)
      return -1;
   memset(&addr, 0, sizeof(addr));
   addr.sun_family = AF_UNIX;
   if (snprintf(addr.sun_path, sizeof(addr.sun_path), "%s/present.sock", dir) >=
       (int)sizeof(addr.sun_path)) {
      close(fd);
      return -1;
   }
   if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
      close(fd);
      return -1;
   }
   return fd;
}

static uint32_t
fourcc_from_vk(VkFormat format)
{
   switch (format) {
   case VK_FORMAT_R8G8B8A8_UNORM:
   case VK_FORMAT_R8G8B8A8_SRGB:
      return ARDESK_FOURCC('A', 'B', '2', '4');
   case VK_FORMAT_B8G8R8A8_UNORM:
   case VK_FORMAT_B8G8R8A8_SRGB:
      return ARDESK_FOURCC('A', 'R', '2', '4');
   default:
      return 0;
   }
}

static xcb_connection_t *
ardesk_connection(VkIcdSurfaceBase *surface)
{
#ifdef VK_USE_PLATFORM_XLIB_KHR
   if (surface->platform == VK_ICD_WSI_PLATFORM_XLIB)
      return XGetXCBConnection(((VkIcdSurfaceXlib *)surface)->dpy);
#endif
   return ((VkIcdSurfaceXcb *)surface)->connection;
}

static xcb_window_t
ardesk_window(VkIcdSurfaceBase *surface)
{
#ifdef VK_USE_PLATFORM_XLIB_KHR
   if (surface->platform == VK_ICD_WSI_PLATFORM_XLIB)
      return ((VkIcdSurfaceXlib *)surface)->window;
#endif
   return ((VkIcdSurfaceXcb *)surface)->window;
}

static bool
ardesk_window_size(VkIcdSurfaceBase *surface, uint32_t *w, uint32_t *h)
{
   xcb_connection_t *conn = ardesk_connection(surface);
   xcb_window_t window = ardesk_window(surface);
   xcb_get_geometry_cookie_t cookie;
   xcb_get_geometry_reply_t *reply;

   if (!conn || !window)
      return false;
   cookie = xcb_get_geometry(conn, window);
   reply = xcb_get_geometry_reply(conn, cookie, NULL);
   if (!reply)
      return false;
   *w = reply->width;
   *h = reply->height;
   free(reply);
   return *w > 0 && *h > 0;
}

static VkResult
ardesk_get_support(VkIcdSurfaceBase *surface, struct wsi_device *wsi_device,
                   uint32_t queueFamilyIndex, VkBool32 *pSupported)
{
   (void)surface;
   (void)wsi_device;
   (void)queueFamilyIndex;
   *pSupported = true;
   return VK_SUCCESS;
}

static VkResult
ardesk_get_capabilities(VkIcdSurfaceBase *surface, struct wsi_device *wsi_device,
                        VkSurfaceCapabilities2KHR *caps)
{
   uint32_t w = 0, h = 0;

   ardesk_window_size(surface, &w, &h);
   caps->surfaceCapabilities.minImageCount = 2;
   caps->surfaceCapabilities.maxImageCount = 0;
   if (w && h) {
      caps->surfaceCapabilities.currentExtent = (VkExtent2D){ w, h };
      caps->surfaceCapabilities.minImageExtent = (VkExtent2D){ w, h };
      caps->surfaceCapabilities.maxImageExtent = (VkExtent2D){ w, h };
   } else {
      caps->surfaceCapabilities.currentExtent = (VkExtent2D){ UINT32_MAX, UINT32_MAX };
      caps->surfaceCapabilities.minImageExtent = (VkExtent2D){ 1, 1 };
      caps->surfaceCapabilities.maxImageExtent = (VkExtent2D){
         wsi_device->maxImageDimension2D,
         wsi_device->maxImageDimension2D,
      };
   }
   caps->surfaceCapabilities.supportedTransforms = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
   caps->surfaceCapabilities.currentTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
   caps->surfaceCapabilities.maxImageArrayLayers = 1;
   caps->surfaceCapabilities.supportedCompositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
   caps->surfaceCapabilities.supportedUsageFlags = wsi_caps_get_image_usage();
   return VK_SUCCESS;
}

static VkResult
ardesk_get_capabilities2(VkIcdSurfaceBase *surface, struct wsi_device *wsi_device,
                         const void *info_next, VkSurfaceCapabilities2KHR *caps)
{
   (void)info_next;
   return ardesk_get_capabilities(surface, wsi_device, caps);
}

static VkResult
ardesk_get_formats(VkIcdSurfaceBase *surface, struct wsi_device *wsi_device,
                   uint32_t *pSurfaceFormatCount, VkSurfaceFormatKHR *pSurfaceFormats)
{
   VK_OUTARRAY_MAKE_TYPED(VkSurfaceFormatKHR, out, pSurfaceFormats, pSurfaceFormatCount);
   (void)surface;
   (void)wsi_device;

   vk_outarray_append_typed(VkSurfaceFormatKHR, &out, fmt) {
      fmt->format = VK_FORMAT_B8G8R8A8_UNORM;
      fmt->colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
   }
   vk_outarray_append_typed(VkSurfaceFormatKHR, &out, fmt) {
      fmt->format = VK_FORMAT_R8G8B8A8_UNORM;
      fmt->colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
   }
   return vk_outarray_status(&out);
}

static VkResult
ardesk_get_formats2(VkIcdSurfaceBase *surface, struct wsi_device *wsi_device,
                    const void *info_next, uint32_t *pSurfaceFormatCount,
                    VkSurfaceFormat2KHR *pSurfaceFormats)
{
   VK_OUTARRAY_MAKE_TYPED(VkSurfaceFormat2KHR, out, pSurfaceFormats, pSurfaceFormatCount);
   (void)surface;
   (void)wsi_device;
   (void)info_next;

   vk_outarray_append_typed(VkSurfaceFormat2KHR, &out, fmt) {
      fmt->surfaceFormat.format = VK_FORMAT_B8G8R8A8_UNORM;
      fmt->surfaceFormat.colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
   }
   vk_outarray_append_typed(VkSurfaceFormat2KHR, &out, fmt) {
      fmt->surfaceFormat.format = VK_FORMAT_R8G8B8A8_UNORM;
      fmt->surfaceFormat.colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
   }
   return vk_outarray_status(&out);
}

static VkResult
ardesk_get_present_modes(VkIcdSurfaceBase *surface, struct wsi_device *wsi_device,
                         uint32_t *pPresentModeCount, VkPresentModeKHR *pPresentModes)
{
   (void)surface;
   (void)wsi_device;
   if (!pPresentModes) {
      *pPresentModeCount = ARRAY_SIZE(present_modes);
      return VK_SUCCESS;
   }
   *pPresentModeCount = MIN2(*pPresentModeCount, ARRAY_SIZE(present_modes));
   typed_memcpy(pPresentModes, present_modes, *pPresentModeCount);
   return *pPresentModeCount < ARRAY_SIZE(present_modes) ? VK_INCOMPLETE : VK_SUCCESS;
}

static VkResult
ardesk_get_present_rectangles(VkIcdSurfaceBase *surface, struct wsi_device *wsi_device,
                              uint32_t *pRectCount, VkRect2D *pRects)
{
   uint32_t w = 0, h = 0;
   VK_OUTARRAY_MAKE_TYPED(VkRect2D, out, pRects, pRectCount);
   (void)wsi_device;
   ardesk_window_size(surface, &w, &h);
   vk_outarray_append_typed(VkRect2D, &out, rect) {
      *rect = (VkRect2D){
         .offset = { 0, 0 },
         .extent = { w ? w : UINT32_MAX, h ? h : UINT32_MAX },
      };
   }
   return vk_outarray_status(&out);
}

static VkResult
ardesk_alloc_image(struct wsi_ardesk_swapchain *chain, struct wsi_ardesk_image *image)
{
   struct ardesk_req req = {
      .magic = ARDESK_MAGIC,
      .cmd = ARDESK_ALLOC,
      .kind = ARDESK_KIND_X11,
      .id = chain->window,
      .w = chain->width,
      .h = chain->height,
      .fourcc = fourcc_from_vk(chain->format),
   };
   struct ardesk_alloc_reply reply;
   int fd = -1;

   if (!req.fourcc)
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
   mtx_lock(&chain->sock_mutex);
   if (write_all(chain->sock, &req, sizeof(req)) != 0 ||
       recv_reply_fd(chain->sock, &reply, sizeof(reply), &fd) != 0) {
      mtx_unlock(&chain->sock_mutex);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }
   mtx_unlock(&chain->sock_mutex);
   if (reply.status != 0 || fd < 0 || reply.token == 0 ||
       reply.fourcc != req.fourcc) {
      if (fd >= 0)
         close(fd);
      return VK_ERROR_OUT_OF_DATE_KHR;
   }
   image->token = reply.token;
   image->fd = fd;
   image->base.dma_buf_fd = fd;
   image->base.drm_modifier = DRM_FORMAT_MOD_LINEAR;
   image->base.num_planes = 1;
   image->base.row_pitches[0] = reply.pitch;
   image->base.offsets[0] = reply.offset;
   image->base.sizes[0] = (uint64_t)reply.pitch * reply.height;
   return VK_SUCCESS;
}

static void
ardesk_release_image(struct wsi_ardesk_swapchain *chain, struct wsi_ardesk_image *image)
{
   if (image->token && chain->sock >= 0) {
      struct ardesk_req req = {
         .magic = ARDESK_MAGIC,
         .cmd = ARDESK_RELEASE,
         .token = image->token,
      };
      struct ardesk_ack ack;
      mtx_lock(&chain->sock_mutex);
      write_all(chain->sock, &req, sizeof(req));
      read_all(chain->sock, &ack, sizeof(ack));
      mtx_unlock(&chain->sock_mutex);
      image->token = 0;
   }
   if (image->fd >= 0) {
      close(image->fd);
      image->fd = -1;
      image->base.dma_buf_fd = -1;
   }
}

static VkResult
ardesk_bind_image(struct wsi_ardesk_swapchain *chain, struct wsi_ardesk_image *image)
{
   const struct wsi_device *wsi = chain->base.wsi;
   VkSubresourceLayout layout = {
      .offset = image->base.offsets[0],
      .size = image->base.sizes[0],
      .rowPitch = image->base.row_pitches[0],
   };
   VkImageDrmFormatModifierExplicitCreateInfoEXT expl = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT,
      .drmFormatModifier = DRM_FORMAT_MOD_LINEAR,
      .drmFormatModifierPlaneCount = 1,
      .pPlaneLayouts = &layout,
   };
   VkExternalMemoryImageCreateInfo ext = {
      .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
      .pNext = &expl,
      .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
   };
   VkImageCreateInfo create = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .pNext = &ext,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = chain->format,
      .extent = { chain->width, chain->height, 1 },
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
      .usage = chain->usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkMemoryRequirements reqs;
   VkMemoryFdPropertiesKHR fd_props = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR,
   };
   VkResult result;
   int import_fd;
   uint32_t type_bits;

   result = wsi->CreateImage(chain->base.device, &create, &chain->base.alloc,
                             &image->base.image);
   if (result != VK_SUCCESS)
      return result;

   wsi->GetImageMemoryRequirements(chain->base.device, image->base.image, &reqs);
   type_bits = reqs.memoryTypeBits;
   if (wsi->GetMemoryFdPropertiesKHR &&
       wsi->GetMemoryFdPropertiesKHR(chain->base.device,
                                     VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
                                     image->fd, &fd_props) == VK_SUCCESS)
      type_bits &= fd_props.memoryTypeBits;
   if (!type_bits)
      return VK_ERROR_INVALID_EXTERNAL_HANDLE;

   import_fd = os_dupfd_cloexec(image->fd);
   if (import_fd < 0)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   {
      VkImportMemoryFdInfoKHR import = {
         .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
         .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
         .fd = import_fd,
      };
      VkMemoryDedicatedAllocateInfo dedicated = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
         .pNext = &import,
         .image = image->base.image,
      };
      VkMemoryAllocateInfo alloc_info = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .pNext = &dedicated,
         .allocationSize = MAX2(reqs.size, image->base.sizes[0]),
         .memoryTypeIndex = wsi_select_device_memory_type(wsi, type_bits),
      };
      result = wsi->AllocateMemory(chain->base.device, &alloc_info,
                                   &chain->base.alloc, &image->base.memory);
      if (result != VK_SUCCESS) {
         close(import_fd);
         return result;
      }
   }

   return wsi->BindImageMemory(chain->base.device, image->base.image,
                               image->base.memory, 0);
}

static struct wsi_image *
ardesk_get_wsi_image(struct wsi_swapchain *wsi_chain, uint32_t image_index)
{
   struct wsi_ardesk_swapchain *chain = (struct wsi_ardesk_swapchain *)wsi_chain;
   return &chain->images[image_index].base;
}

static VkResult
ardesk_release_images(struct wsi_swapchain *wsi_chain, uint32_t count,
                      const uint32_t *indices)
{
   struct wsi_ardesk_swapchain *chain = (struct wsi_ardesk_swapchain *)wsi_chain;

   for (uint32_t i = 0; i < count; i++) {
      chain->images[indices[i]].busy_on_device = false;
      chain->images[indices[i]].busy_on_host = false;
   }
   return VK_SUCCESS;
}

static VkResult
ardesk_acquire_next_image(struct wsi_swapchain *wsi_chain,
                          const VkAcquireNextImageInfoKHR *info,
                          uint32_t *image_index)
{
   struct wsi_ardesk_swapchain *chain = (struct wsi_ardesk_swapchain *)wsi_chain;
   struct timespec start_time, end_time, rel_timeout;

   timespec_from_nsec(&rel_timeout, info->timeout);
   clock_gettime(CLOCK_MONOTONIC, &start_time);
   timespec_add(&end_time, &rel_timeout, &start_time);

   while (1) {
      for (uint32_t i = 0; i < chain->base.image_count; i++) {
         if (!chain->images[i].busy_on_host) {
            if (chain->images[i].busy_on_device) {
               chain->images[i].busy_on_device = false;
               continue;
            }
            *image_index = i;
            chain->images[i].busy_on_host = true;
            chain->images[i].busy_on_device = true;
            return VK_SUCCESS;
         }
      }
      struct timespec current_time;
      clock_gettime(CLOCK_MONOTONIC, &current_time);
      if (timespec_after(&current_time, &end_time))
         return VK_NOT_READY;
   }
}

static VkResult
ardesk_queue_present(struct wsi_swapchain *wsi_chain, uint32_t image_index,
                     uint64_t present_id, const VkPresentRegionKHR *damage)
{
   struct wsi_ardesk_swapchain *chain = (struct wsi_ardesk_swapchain *)wsi_chain;
   struct ardesk_req req = {
      .magic = ARDESK_MAGIC,
      .cmd = ARDESK_PRESENT,
      .kind = ARDESK_KIND_X11,
      .id = chain->window,
      .w = chain->width,
      .h = chain->height,
      .token = chain->images[image_index].token,
   };
   struct ardesk_ack ack;
   int rc;

   (void)present_id;
   (void)damage;
   mtx_lock(&chain->sock_mutex);
   rc = write_all(chain->sock, &req, sizeof(req));
   if (rc == 0)
      rc = read_all(chain->sock, &ack, sizeof(ack));
   mtx_unlock(&chain->sock_mutex);
   chain->images[image_index].busy_on_host = false;
   if (rc != 0 || ack.status != 0)
      return VK_ERROR_OUT_OF_DATE_KHR;
   return VK_SUCCESS;
}

static VkResult
ardesk_wait_for_present(struct wsi_swapchain *wsi_chain, uint64_t waitValue,
                        uint64_t timeout)
{
   return wsi_swapchain_wait_for_present_semaphore(wsi_chain, waitValue, timeout);
}

static VkResult
ardesk_swapchain_destroy(struct wsi_swapchain *wsi_chain,
                         const VkAllocationCallbacks *pAllocator)
{
   struct wsi_ardesk_swapchain *chain = (struct wsi_ardesk_swapchain *)wsi_chain;
   const struct wsi_device *wsi = chain->base.wsi;

   for (uint32_t i = 0; i < chain->base.image_count; i++) {
      if (chain->images[i].base.image != VK_NULL_HANDLE)
         wsi->DestroyImage(chain->base.device, chain->images[i].base.image,
                           &chain->base.alloc);
      if (chain->images[i].base.memory != VK_NULL_HANDLE)
         wsi->FreeMemory(chain->base.device, chain->images[i].base.memory,
                         &chain->base.alloc);
      ardesk_release_image(chain, &chain->images[i]);
   }
   if (chain->sock >= 0)
      close(chain->sock);
   mtx_destroy(&chain->sock_mutex);
   wsi_swapchain_finish(&chain->base);
   vk_free(pAllocator, chain);
   return VK_SUCCESS;
}

static VkResult
ardesk_create_swapchain(VkIcdSurfaceBase *surface, VkDevice device,
                        struct wsi_device *wsi_device,
                        const VkSwapchainCreateInfoKHR *pCreateInfo,
                        const VkAllocationCallbacks *pAllocator,
                        struct wsi_swapchain **swapchain_out)
{
   struct wsi_ardesk_swapchain *chain;
   struct wsi_drm_image_params drm_params = {
      .base.image_type = WSI_IMAGE_TYPE_DRM,
      .same_gpu = true,
   };
   uint32_t num_images = MAX2(pCreateInfo->minImageCount, 2);
   VkResult result;
   uint32_t image;

   chain = vk_zalloc(pAllocator,
                     sizeof(*chain) + num_images * sizeof(chain->images[0]), 8,
                     VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!chain)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   result = wsi_swapchain_init(wsi_device, &chain->base, device, pCreateInfo,
                               &drm_params.base, pAllocator);
   if (result != VK_SUCCESS) {
      vk_free(pAllocator, chain);
      return result;
   }

   chain->sock = ardesk_connect();
   if (chain->sock < 0) {
      wsi_swapchain_finish(&chain->base);
      vk_free(pAllocator, chain);
      return VK_ERROR_INITIALIZATION_FAILED;
   }
   if (mtx_init(&chain->sock_mutex, mtx_plain) != thrd_success) {
      close(chain->sock);
      wsi_swapchain_finish(&chain->base);
      vk_free(pAllocator, chain);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }

   chain->window = ardesk_window(surface);
   chain->width = pCreateInfo->imageExtent.width;
   chain->height = pCreateInfo->imageExtent.height;
   chain->format = pCreateInfo->imageFormat;
   chain->usage = pCreateInfo->imageUsage;
   chain->base.destroy = ardesk_swapchain_destroy;
   chain->base.get_wsi_image = ardesk_get_wsi_image;
   chain->base.acquire_next_image = ardesk_acquire_next_image;
   chain->base.release_images = ardesk_release_images;
   chain->base.queue_present = ardesk_queue_present;
   chain->base.wait_for_present = ardesk_wait_for_present;
   chain->base.present_mode = wsi_swapchain_get_present_mode(wsi_device, pCreateInfo);
   chain->base.image_count = num_images;

   for (image = 0; image < num_images; image++) {
      chain->images[image].fd = -1;
      chain->images[image].base.dma_buf_fd = -1;
      result = ardesk_alloc_image(chain, &chain->images[image]);
      if (result == VK_SUCCESS)
         result = ardesk_bind_image(chain, &chain->images[image]);
      if (result != VK_SUCCESS)
         goto fail;
   }

   *swapchain_out = &chain->base;
   return VK_SUCCESS;

fail:
   ardesk_swapchain_destroy(&chain->base, pAllocator);
   return result;
}

VkResult
wsi_ardesk_init_wsi(struct wsi_device *wsi_device,
                    const VkAllocationCallbacks *alloc)
{
   struct wsi_ardesk *wsi = vk_alloc(alloc, sizeof(*wsi), 8,
                                     VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE);
   if (!wsi) {
      wsi_device->wsi[VK_ICD_WSI_PLATFORM_XCB] = NULL;
      wsi_device->wsi[VK_ICD_WSI_PLATFORM_XLIB] = NULL;
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }

   wsi->base.get_support = ardesk_get_support;
   wsi->base.get_capabilities2 = ardesk_get_capabilities2;
   wsi->base.get_formats = ardesk_get_formats;
   wsi->base.get_formats2 = ardesk_get_formats2;
   wsi->base.get_present_modes = ardesk_get_present_modes;
   wsi->base.get_present_rectangles = ardesk_get_present_rectangles;
   wsi->base.create_swapchain = ardesk_create_swapchain;
   wsi_device->wsi[VK_ICD_WSI_PLATFORM_XCB] = &wsi->base;
   wsi_device->wsi[VK_ICD_WSI_PLATFORM_XLIB] = &wsi->base;
   return VK_SUCCESS;
}

void
wsi_ardesk_finish_wsi(struct wsi_device *wsi_device,
                      const VkAllocationCallbacks *alloc)
{
   struct wsi_ardesk *wsi =
      (struct wsi_ardesk *)wsi_device->wsi[VK_ICD_WSI_PLATFORM_XCB];
   if (wsi)
      vk_free(alloc, wsi);
}
