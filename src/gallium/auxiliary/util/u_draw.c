/**************************************************************************
 *
 * Copyright 2011 VMware, Inc.
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sub license, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
 * IN NO EVENT SHALL VMWARE AND/OR ITS SUPPLIERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 **************************************************************************/


#include "util/u_debug.h"
#include "util/u_inlines.h"
#include "util/u_math.h"
#include "util/format/u_format.h"
#include "util/u_draw.h"


/**
 * Returns the largest legal index value plus one for the current set
 * of bound vertex buffers.  Regardless of any other consideration,
 * all vertex lookups need to be clamped to 0..max_index-1 to prevent
 * an out-of-bound access.
 *
 * Note that if zero is returned it means that one or more buffers is
 * too small to contain any valid vertex data.
 */
unsigned
util_draw_max_index(
      const struct pipe_vertex_buffer *vertex_buffers,
      const struct pipe_vertex_element *vertex_elements,
      unsigned nr_vertex_elements,
      const struct pipe_draw_info *info)
{
   unsigned max_index;
   unsigned i;

   max_index = ~0U - 1;
   for (i = 0; i < nr_vertex_elements; i++) {
      const struct pipe_vertex_element *element =
         &vertex_elements[i];
      const struct pipe_vertex_buffer *buffer =
         &vertex_buffers[element->vertex_buffer_index];
      unsigned buffer_size;
      const struct util_format_description *format_desc;
      unsigned format_size;

      if (buffer->is_user_buffer || !buffer->buffer.resource) {
         continue;
      }

      assert(buffer->buffer.resource->height0 == 1);
      assert(buffer->buffer.resource->depth0 == 1);
      buffer_size = buffer->buffer.resource->width0;

      format_desc = util_format_description(element->src_format);
      assert(format_desc->block.width == 1);
      assert(format_desc->block.height == 1);
      assert(format_desc->block.bits % 8 == 0);
      format_size = format_desc->block.bits/8;

      if (buffer->buffer_offset >= buffer_size) {
         /* buffer is too small */
         return 0;
      }

      buffer_size -= buffer->buffer_offset;

      if (element->src_offset >= buffer_size) {
         /* buffer is too small */
         return 0;
      }

      buffer_size -= element->src_offset;

      if (format_size > buffer_size) {
         /* buffer is too small */
         return 0;
      }

      buffer_size -= format_size;

      if (element->src_stride != 0) {
         unsigned buffer_max_index;

         buffer_max_index = buffer_size / element->src_stride;

         if (element->instance_divisor == 0) {
            /* Per-vertex data */
            max_index = MIN2(max_index, buffer_max_index);
         }
         else {
            /* Per-instance data. Simply make sure gallium frontends didn't
             * request more instances than those that fit in the buffer */
            if ((info->start_instance + info->instance_count)/element->instance_divisor
                > (buffer_max_index + 1)) {
               /* FIXME: We really should stop thinking in terms of maximum
                * indices/instances and simply start clamping against buffer
                * size. */
               debug_printf("%s: too many instances for vertex buffer\n",
                            __func__);
               return 0;
            }
         }
      }
   }

   return max_index + 1;
}

struct u_indirect_params *
util_draw_indirect_read(struct pipe_context *pipe,
                        const struct pipe_draw_info *info_in,
                        const struct pipe_draw_indirect_info *indirect,
                        unsigned *num_draws)
{
   unsigned num_params = info_in->index_size ? 5 : 4;
   unsigned command_size = num_params * sizeof(uint32_t);
   unsigned stride = indirect->stride ? indirect->stride : command_size;
   *num_draws = 0;
   assert(!indirect->count_from_stream_output);

   uint32_t draw_count = indirect->draw_count;
   if (indirect->indirect_draw_count) {
      struct pipe_transfer *transfer;
      if ((uint64_t)indirect->indirect_draw_count_offset + 4 >
          indirect->indirect_draw_count->width0)
         return NULL;
      uint32_t *count = pipe_buffer_map_range(pipe, indirect->indirect_draw_count,
                                             indirect->indirect_draw_count_offset,
                                             4, PIPE_MAP_READ, &transfer);
      if (!count)
         return NULL;
      draw_count = MIN2(draw_count, *count);
      pipe_buffer_unmap(pipe, transfer);
   }
   if (!draw_count)
      return NULL;

   uint64_t map_size = (uint64_t)(draw_count - 1) * stride + command_size;
   if (!indirect->buffer || (draw_count > 1 && (stride % 4 || stride < command_size)) ||
       (uint64_t)indirect->offset + map_size > indirect->buffer->width0)
      return NULL;
   struct u_indirect_params *draws = calloc(draw_count, sizeof(*draws));
   if (!draws)
      return NULL;
   struct pipe_transfer *transfer;
   const uint32_t *params = pipe_buffer_map_range(pipe, indirect->buffer,
                                                 indirect->offset, map_size,
                                                 PIPE_MAP_READ, &transfer);
   if (!params) {
      free(draws);
      return NULL;
   }
   for (unsigned i = 0; i < draw_count; i++) {
      draws[i].info = *info_in;
      draws[i].draw.count = params[0];
      draws[i].info.instance_count = params[1];
      draws[i].draw.start = params[2];
      draws[i].draw.index_bias = info_in->index_size ? params[3] : 0;
      draws[i].info.start_instance = info_in->index_size ? params[4] : params[3];
      params += stride / 4;
   }
   pipe_buffer_unmap(pipe, transfer);
   *num_draws = draw_count;
   return draws;
}

void
util_draw_indirect(struct pipe_context *pipe,
                   const struct pipe_draw_info *info,
                   unsigned drawid_offset,
                   const struct pipe_draw_indirect_info *indirect)
{
   unsigned num_draws;
   struct u_indirect_params *draws = util_draw_indirect_read(pipe, info, indirect, &num_draws);
   for (unsigned i = 0; i < num_draws; i++)
      pipe->draw_vbo(pipe, &draws[i].info, drawid_offset + i, NULL, &draws[i].draw, 1);
   free(draws);
}

void
util_draw_multi(struct pipe_context *pctx, const struct pipe_draw_info *info,
                unsigned drawid_offset,
                const struct pipe_draw_indirect_info *indirect,
                const struct pipe_draw_start_count_bias *draws,
                unsigned num_draws)
{
   unsigned drawid = drawid_offset;

   /* If you call this with num_draws==1, that is probably going to be
    * an infinite loop
    */
   assert(num_draws > 1);

   for (unsigned i = 0; i < num_draws; i++) {
      if (indirect || (draws[i].count && info->instance_count))
         pctx->draw_vbo(pctx, info, drawid, indirect, &draws[i], 1);
      if (info->increment_draw_id)
         drawid++;
   }
}
