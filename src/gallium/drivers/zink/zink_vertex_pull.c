/* SPDX-License-Identifier: MIT */
#include "zink_vertex_pull.h"
#include "zink_context.h"
#include "zink_screen.h"
#include "compiler/nir/nir_format_convert.h"
#include "util/format/u_format.h"
#include "util/u_inlines.h"

/* Original VBOs stay on the GPU. Only their final partial words need padding. */
bool
zink_vertex_inputs_prepare(struct zink_context *ctx, const nir_shader *nir,
               const struct pipe_draw_info *info,
               const struct pipe_draw_start_count_bias *draw,
               struct zink_vertex_inputs *inputs)
{
   inputs->first_binding = 0;
   inputs->index_size = info->index_size;
   inputs->primitive_restart = info->primitive_restart;
   inputs->restart_index = info->primitive_restart ? info->restart_index : 0;
   unsigned used = 0;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_load_input)
               continue;
            unsigned index = nir_intrinsic_base(intr);
            if (!ctx->element_state || index >= ctx->element_state->num_elements || index >= 32)
               return false;
            if (used & BITFIELD_BIT(index))
               continue;
            used |= BITFIELD_BIT(index);
            const struct pipe_vertex_element *elem = &ctx->element_state->elements[index];
            const struct pipe_vertex_buffer *vb = &ctx->vertex_buffers[elem->vertex_buffer_index];
            const struct util_format_description *desc = util_format_description(elem->src_format);
            if (elem->dual_slot || vb->is_user_buffer || !vb->buffer.resource ||
                desc->layout != UTIL_FORMAT_LAYOUT_PLAIN || !desc->block.bits ||
                desc->block.bits > 128 || desc->block.bits % 8)
               return false;
            bool packed_float = elem->src_format == PIPE_FORMAT_R11G11B10_FLOAT ||
                                elem->src_format == PIPE_FORMAT_R9G9B9E5_FLOAT;
            for (unsigned c = 0; c < desc->nr_channels && !packed_float; c++) {
               const struct util_format_channel_description *ch = &desc->channel[c];
               if (ch->size > 32 || ch->type == UTIL_FORMAT_TYPE_FIXED ||
                   (ch->type == UTIL_FORMAT_TYPE_FLOAT && ch->size != 16 && ch->size != 32) ||
                   (desc->block.bits > 32 && ch->size != desc->channel[0].size))
                  return false;
            }
            uint64_t last = elem->instance_divisor ?
               (uint64_t)info->start_instance + (info->instance_count - 1) / elem->instance_divisor :
               (uint64_t)draw->start + draw->count - 1;
            uint64_t offset = (uint64_t)vb->buffer_offset + elem->src_offset;
            uint64_t width = vb->buffer.resource->width0;
            unsigned bytes = desc->block.bits / 8;
            if (offset > width || bytes > width - offset ||
                ((!info->index_size || elem->instance_divisor) && elem->src_stride &&
                 last > (width - offset - bytes) / elem->src_stride) ||
                width > zink_screen(ctx->base.screen)->info.props.limits.maxStorageBufferRange)
               return false;
            unsigned buffer = 0;
            while (buffer < inputs->count &&
                   inputs->buffers[buffer].buffer != vb->buffer.resource)
               buffer++;
            unsigned slot = buffer;
            if (slot >= ZINK_VERTEX_INPUT_BUFFERS)
               return false;
            inputs->elements[index] = *elem;
            inputs->slots[index] = slot;
            inputs->offsets[index] = offset;
            inputs->max_records[index] = elem->src_stride ?
               (width - offset - bytes) / elem->src_stride : UINT32_MAX;
            if (buffer == inputs->count)
               inputs->buffers[inputs->count++] = (struct pipe_shader_buffer) {
                  .buffer = vb->buffer.resource, .buffer_size = vb->buffer.resource->width0,
               };
         }
      }
   }
   if (info->index_size) {
      if (info->has_user_indices || !info->index.resource ||
          (info->index_size != 1 && info->index_size != 2 && info->index_size != 4))
         return false;
      uint64_t end = ((uint64_t)draw->start + draw->count) * info->index_size;
      struct pipe_resource *index = info->index.resource;
      if (end > index->width0 || index->width0 >
          zink_screen(ctx->base.screen)->info.props.limits.maxStorageBufferRange)
         return false;
      unsigned buffer = 0;
      while (buffer < inputs->count && inputs->buffers[buffer].buffer != index)
         buffer++;
      if (buffer >= ZINK_VERTEX_INPUT_BUFFERS)
         return false;
      inputs->index_slot = buffer;
      if (buffer == inputs->count)
         inputs->buffers[inputs->count++] = (struct pipe_shader_buffer) {
            .buffer = index, .buffer_size = index->width0,
         };
   }
   inputs->source_count = inputs->count;
   unsigned tail_size = 0;
   for (unsigned i = 0; i < inputs->source_count; i++) {
      inputs->tail_offsets[i] = UINT32_MAX;
      if (inputs->buffers[i].buffer_size % 4) {
         inputs->tail_offsets[i] = tail_size;
         tail_size += 4;
      }
   }
   unsigned total = inputs->count + (tail_size != 0);
   if (total > ZINK_VERTEX_INPUT_BUFFERS)
      return false;
   struct pipe_screen *pscreen = ctx->base.screen;
   struct zink_screen *screen = zink_screen(pscreen);
   inputs->texture_binding = MAX2(nir->info.num_textures, ctx->di.num_sampler_views[MESA_SHADER_VERTEX]);
   inputs->first_binding = nir->info.num_ssbos;
   unsigned texture_limit = pscreen->shader_caps[MESA_SHADER_COMPUTE].max_sampler_views;
   bool texel_format = pscreen->is_format_supported(pscreen, PIPE_FORMAT_R32_UINT,
                                                   PIPE_BUFFER, 0, 0, PIPE_BIND_SAMPLER_VIEW);
   for (unsigned i = 0; i < total; i++) {
      unsigned bytes = i < inputs->source_count ? inputs->buffers[i].buffer_size : tail_size;
      if (texel_format && inputs->texture_binding + inputs->texel_count < texture_limit &&
          bytes / 4 <= screen->info.props.limits.maxTexelBufferElements) {
         inputs->texel_mask |= BITFIELD_BIT(i);
         inputs->bindings[i] = inputs->texture_binding + inputs->texel_count++;
      } else {
         inputs->bindings[i] = inputs->first_binding + inputs->ssbo_count++;
      }
   }
   if (inputs->first_binding + inputs->ssbo_count > ZINK_VERTEX_OUTPUT_SLOT)
      return false;
   for (unsigned i = 0; i < PIPE_MAX_ATTRIBS; i++)
      inputs->slots[i] += inputs->first_binding;
   inputs->index_slot += inputs->first_binding;
   if (tail_size) {
      struct pipe_context *pctx = &ctx->base;
      inputs->tail = pipe_buffer_create(pctx->screen, PIPE_BIND_SHADER_BUFFER,
                                        PIPE_USAGE_DEFAULT, tail_size);
      if (!inputs->tail)
         return false;
      uint32_t zero = 0;
      pctx->clear_buffer(pctx, inputs->tail, 0, tail_size, &zero, sizeof(zero));
      for (unsigned i = 0; i < inputs->source_count; i++) {
         if (inputs->tail_offsets[i] == UINT32_MAX)
            continue;
         struct pipe_shader_buffer *buffer = &inputs->buffers[i];
         struct pipe_box box = {.x = buffer->buffer_size & ~3u,
                                .width = buffer->buffer_size % 4, .height = 1, .depth = 1};
         pctx->resource_copy_region(pctx, inputs->tail, 0, inputs->tail_offsets[i], 0, 0,
                                    buffer->buffer, 0, &box);
      }
      inputs->buffers[inputs->count++] = (struct pipe_shader_buffer) {
         .buffer = inputs->tail, .buffer_size = tail_size,
      };
   }
   for (unsigned i = 0; i < inputs->count; i++) {
      if (inputs->texel_mask & BITFIELD_BIT(i)) {
         struct pipe_resource *resource = inputs->buffers[i].buffer;
         unsigned bytes = inputs->buffers[i].buffer_size & ~3u;
         /* A sub-word source is fetched only through its padded tail. Give its
          * unused descriptor a valid view as well. */
         if (!bytes) {
            resource = inputs->tail;
            bytes = resource->width0;
         }
         struct pipe_sampler_view view = {
            .format = PIPE_FORMAT_R32_UINT, .target = PIPE_BUFFER,
            .swizzle_r = PIPE_SWIZZLE_X, .swizzle_g = PIPE_SWIZZLE_0,
            .swizzle_b = PIPE_SWIZZLE_0, .swizzle_a = PIPE_SWIZZLE_1,
            .u.buf = {.offset = 0, .size = bytes},
         };
         unsigned slot = inputs->bindings[i] - inputs->texture_binding;
         inputs->views[slot] = ctx->base.create_sampler_view(&ctx->base, resource, &view);
         if (!inputs->views[slot]) {
            zink_vertex_inputs_finish(&ctx->base, inputs);
            return false;
         }
      } else {
         inputs->ssbo_buffers[inputs->bindings[i] - inputs->first_binding] = inputs->buffers[i];
      }
   }
   return true;
}

static nir_def *
fetch_word(nir_builder *b, const struct zink_vertex_inputs *inputs, unsigned binding, nir_def *offset)
{
   unsigned buffer = binding - inputs->first_binding;
   unsigned slot = inputs->bindings[buffer];
   if (!(inputs->texel_mask & BITFIELD_BIT(buffer)))
      return nir_load_ssbo(b, 1, 32, nir_imm_int(b, slot), offset,
                           .align_mul = 4, .access = ACCESS_NON_WRITEABLE);
   nir_tex_instr *tex = nir_tex_instr_create(b->shader, 2);
   tex->op = nir_texop_txf;
   tex->sampler_dim = GLSL_SAMPLER_DIM_BUF;
   tex->coord_components = 1;
   tex->dest_type = nir_type_uint32;
   tex->texture_index = slot;
   tex->sampler_index = slot;
   tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord, nir_ushr_imm(b, offset, 2));
   tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_texture_deref,
      &nir_build_deref_var(b, inputs->variables[buffer])->def);
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(b, &tex->instr);
   return nir_channel(b, &tex->def, 0);
}

static nir_def *
load_word(nir_builder *b, const struct zink_vertex_inputs *inputs,
          unsigned buffer, unsigned first_binding, nir_def *offset)
{
   nir_def *tail_word = NULL;
   if (inputs->tail_offsets[buffer] != UINT32_MAX) {
      nir_push_if(b, nir_ieq_imm(b, offset, inputs->buffers[buffer].buffer_size & ~3u));
      tail_word = fetch_word(b, inputs, first_binding + inputs->source_count,
                             nir_imm_int(b, inputs->tail_offsets[buffer]));
      nir_push_else(b, NULL);
   }
   nir_def *word = fetch_word(b, inputs, first_binding + buffer, offset);
   if (tail_word) {
      nir_pop_if(b, NULL);
      word = nir_if_phi(b, tail_word, word);
   }
   return word;
}

nir_def *
zink_vertex_input_load(nir_builder *b, const struct zink_vertex_inputs *inputs,
                       unsigned index, nir_def *record)
{
   unsigned first_binding = inputs->first_binding;
   const struct pipe_vertex_element *elem = &inputs->elements[index];
   unsigned buffer = inputs->slots[index] - first_binding;
   unsigned bits = util_format_get_blocksizebits(elem->src_format);
   bool guarded = inputs->index_size && !elem->instance_divisor;
   if (guarded)
      nir_push_if(b, nir_ule_imm(b, record, inputs->max_records[index]));
   nir_def *offset = nir_iadd_imm(b, nir_imul_imm(b, record, elem->src_stride), inputs->offsets[index]);
   nir_def *aligned = nir_iand_imm(b, offset, ~3u);
   nir_def *shift = nir_imul_imm(b, nir_iand_imm(b, offset, 3), 8);
   nir_def *words[4];
   for (unsigned i = 0; i < DIV_ROUND_UP(bits, 32); i++) {
      nir_def *address = nir_iadd_imm(b, aligned, 4 * i);
      nir_def *low = nir_ushr(b, load_word(b, inputs, buffer, first_binding, address), shift);
      /* Never fetch another word unless this format actually needs its bytes. */
      nir_push_if(b, nir_ugt_imm(b, shift, 32 - MIN2(bits - 32 * i, 32)));
      nir_def *high = load_word(b, inputs, buffer, first_binding, nir_iadd_imm(b, address, 4));
      nir_def *merged = nir_ior(b, low, nir_ishl(b, high, nir_isub(b, nir_imm_int(b, 32), shift)));
      nir_pop_if(b, NULL);
      words[i] = nir_if_phi(b, merged, low);
   }
   nir_def *rgba = nir_format_unpack_rgba(b, nir_vec(b, words, DIV_ROUND_UP(bits, 32)), elem->src_format);
   if (guarded) {
      nir_push_else(b, NULL);
      nir_def *zero = nir_imm_zero(b, 4, 32);
      nir_pop_if(b, NULL);
      rgba = nir_if_phi(b, rgba, zero);
   }
   return rgba;
}

nir_def *
zink_vertex_index_load(nir_builder *b, const struct zink_vertex_inputs *inputs, nir_def *offset)
{
   unsigned buffer = inputs->index_slot - inputs->first_binding;
   nir_def *word = load_word(b, inputs, buffer, inputs->first_binding, nir_iand_imm(b, offset, ~3u));
   nir_def *value = nir_ushr(b, word, nir_imul_imm(b, nir_iand_imm(b, offset, 3), 8));
   return nir_iand_imm(b, value, inputs->index_size == 4 ? UINT32_MAX :
                       (1u << (8 * inputs->index_size)) - 1);
}

void
zink_vertex_inputs_finish(struct pipe_context *pctx, struct zink_vertex_inputs *inputs)
{
   for (unsigned i = 0; i < inputs->texel_count; i++)
      pipe_sampler_view_reference(&inputs->views[i], NULL);
   if (inputs->tail)
      pipe_resource_release(pctx, inputs->tail);
}
