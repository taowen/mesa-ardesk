/* SPDX-License-Identifier: MIT */
#include "zink_instance_rebase.h"
#include "zink_context.h"
#include "zink_screen.h"
#include "util/u_draw.h"

/* KHR/core allow devices to reject nonzero firstInstance with divisors != 1.
 * Keep native vertex processing: shift instanced buffers by the application
 * base, draw with firstInstance zero, and supply BaseInstance via push data. */
bool
zink_instance_rebase_draw(struct pipe_context *pctx,
                          const struct pipe_draw_info *info,
                          unsigned drawid_offset,
                          const struct pipe_draw_indirect_info *indirect,
                          const struct pipe_draw_start_count_bias *draws,
                          unsigned num_draws)
{
   struct zink_context *ctx = zink_context(pctx);
   const struct zink_vertex_elements_state *elems = ctx->element_state;
   if (zink_screen(pctx->screen)->info.vdiv_supports_nonzero_first_instance ||
       ctx->base_instance_offset || !elems)
      return false;
   bool needs_rebase = false;
   for (unsigned i = 0; i < elems->num_elements; i++)
      needs_rebase |= elems->elements[i].instance_divisor > 1;
   if (!needs_rebase)
      return false;

   if (indirect && indirect->buffer) {
      /* The base may be GPU-generated. The existing Gallium fallback maps
       * just the argument/count buffers and re-enters with direct draws. */
      util_draw_indirect(pctx, info, drawid_offset, indirect);
      return true;
   }
   if (!info->start_instance)
      return false;

   unsigned mask = 0;
   unsigned saved_offsets[PIPE_MAX_ATTRIBS];
   unsigned offsets[PIPE_MAX_ATTRIBS];
   for (unsigned i = 0; i < elems->num_elements; i++) {
      const struct pipe_vertex_element *elem = &elems->elements[i];
      if (!elem->instance_divisor)
         continue;
      unsigned index = elem->vertex_buffer_index;
      if (mask & BITFIELD_BIT(index))
         continue;
      const struct pipe_vertex_buffer *vb = &ctx->vertex_buffers[index];
      if (!vb->buffer.resource)
         continue;
      uint64_t offset = (uint64_t)vb->buffer_offset +
                        (uint64_t)info->start_instance * elem->src_stride;
      if (offset >= vb->buffer.resource->width0) {
         mesa_loge("ZINK: instanced vertex buffer base is out of bounds");
         return true;
      }
      mask |= BITFIELD_BIT(index);
      saved_offsets[index] = vb->buffer_offset;
      offsets[index] = offset;
   }
   struct pipe_draw_info rebased = *info;
   rebased.start_instance = 0;
   u_foreach_bit(i, mask)
      ctx->vertex_buffers[i].buffer_offset = offsets[i];
   ctx->base_instance_offset = info->start_instance;
   ctx->vertex_buffers_dirty = true;
   pctx->draw_vbo(pctx, &rebased, drawid_offset, indirect, draws, num_draws);
   u_foreach_bit(i, mask)
      ctx->vertex_buffers[i].buffer_offset = saved_offsets[i];
   ctx->base_instance_offset = 0;
   ctx->vertex_buffers_dirty = true;
   return true;
}
