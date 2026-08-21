/*
 * GLX/DRI present through anlabwc's present.sock (AHB2). Keep the on-wire
 * structs in sync with ardesk android/app/src/main/cpp/present_share.h and
 * src/vulkan/wsi/wsi_common_ardesk.c.
 */

#include "loader_ardesk_present.h"

#include "util/simple_mtx.h"

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#define ARDESK_MAGIC 0x32424841u /* 'AHB2' */
#define ARDESK_ALLOC 1
#define ARDESK_PRESENT 2
#define ARDESK_RELEASE 3
#define ARDESK_KIND_X11 1

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

static simple_mtx_t sock_mtx = SIMPLE_MTX_INITIALIZER;
static int sock_fd = -1;

bool
loader_ardesk_available(void)
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

int
loader_ardesk_connect(void)
{
   const char *dir;
   struct sockaddr_un addr;
   int fd;

   simple_mtx_lock(&sock_mtx);
   if (sock_fd >= 0) {
      fd = sock_fd;
      simple_mtx_unlock(&sock_mtx);
      return fd;
   }

   dir = getenv("XDG_RUNTIME_DIR");
   if (!dir || !dir[0]) {
      simple_mtx_unlock(&sock_mtx);
      return -1;
   }
   fd = socket(AF_UNIX, SOCK_STREAM, 0);
   if (fd < 0) {
      simple_mtx_unlock(&sock_mtx);
      return -1;
   }
   memset(&addr, 0, sizeof(addr));
   addr.sun_family = AF_UNIX;
   if (snprintf(addr.sun_path, sizeof(addr.sun_path), "%s/present.sock", dir) >=
       (int)sizeof(addr.sun_path)) {
      close(fd);
      simple_mtx_unlock(&sock_mtx);
      return -1;
   }
   if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
      close(fd);
      simple_mtx_unlock(&sock_mtx);
      return -1;
   }
   fcntl(fd, F_SETFD, fcntl(fd, F_GETFD) | FD_CLOEXEC);
   sock_fd = fd;
   simple_mtx_unlock(&sock_mtx);
   return fd;
}

int
loader_ardesk_alloc(int sock, uint32_t window, uint32_t w, uint32_t h,
                    uint32_t fourcc, uint32_t *token, int *fd,
                    uint32_t *pitch, uint32_t *offset, uint32_t *got_fourcc)
{
   struct ardesk_req req = {
      .magic = ARDESK_MAGIC,
      .cmd = ARDESK_ALLOC,
      .kind = ARDESK_KIND_X11,
      .id = window,
      .w = w,
      .h = h,
      .fourcc = fourcc,
   };
   struct ardesk_alloc_reply reply;
   int got = -1;

   simple_mtx_lock(&sock_mtx);
   if (write_all(sock, &req, sizeof(req)) != 0 ||
       recv_reply_fd(sock, &reply, sizeof(reply), &got) != 0) {
      simple_mtx_unlock(&sock_mtx);
      return -1;
   }
   simple_mtx_unlock(&sock_mtx);
   if (reply.status != 0 || got < 0 || reply.token == 0) {
      if (got >= 0)
         close(got);
      return -1;
   }
   *token = reply.token;
   *fd = got;
   *pitch = reply.pitch;
   *offset = reply.offset;
   *got_fourcc = reply.fourcc;
   return 0;
}

int
loader_ardesk_present(int sock, uint32_t window, uint32_t w, uint32_t h,
                      uint32_t token)
{
   struct ardesk_req req = {
      .magic = ARDESK_MAGIC,
      .cmd = ARDESK_PRESENT,
      .kind = ARDESK_KIND_X11,
      .id = window,
      .w = w,
      .h = h,
      .token = token,
   };
   struct ardesk_ack ack;
   int rc;

   simple_mtx_lock(&sock_mtx);
   rc = write_all(sock, &req, sizeof(req));
   if (rc == 0)
      rc = read_all(sock, &ack, sizeof(ack));
   simple_mtx_unlock(&sock_mtx);
   return (rc == 0 && ack.status == 0) ? 0 : -1;
}

int
loader_ardesk_release(int sock, uint32_t token)
{
   struct ardesk_req req = {
      .magic = ARDESK_MAGIC,
      .cmd = ARDESK_RELEASE,
      .token = token,
   };
   struct ardesk_ack ack;

   if (sock < 0 || !token)
      return 0;
   simple_mtx_lock(&sock_mtx);
   write_all(sock, &req, sizeof(req));
   read_all(sock, &ack, sizeof(ack));
   simple_mtx_unlock(&sock_mtx);
   return 0;
}
