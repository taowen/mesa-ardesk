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
   inputs->first_binding = nir->info.num_ssbos;
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
                (elem->src_stride && last > (width - offset - bytes) / elem->src_stride) ||
                width > zink_screen(ctx->base.screen)->info.props.limits.maxStorageBufferRange)
               return false;
            unsigned buffer = 0;
            while (buffer < inputs->count &&
                   inputs->buffers[buffer].buffer != vb->buffer.resource)
               buffer++;
            unsigned slot = nir->info.num_ssbos + buffer;
            if (slot >= ZINK_VERTEX_INPUT_BUFFERS)
               return false;
            inputs->elements[index] = *elem;
            inputs->slots[index] = slot;
            inputs->offsets[index] = offset;
            if (buffer == inputs->count)
               inputs->buffers[inputs->count++] = (struct pipe_shader_buffer) {
                  .buffer = vb->buffer.resource, .buffer_size = vb->buffer.resource->width0,
               };
         }
      }
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
   if (tail_size) {
      if (nir->info.num_ssbos + inputs->count >= ZINK_VERTEX_INPUT_BUFFERS)
         return false;
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
   return true;
}

static nir_def *
load_word(nir_builder *b, const struct zink_vertex_inputs *inputs,
          unsigned buffer, unsigned first_binding, nir_def *offset)
{
   nir_def *tail_word = NULL;
   if (inputs->tail_offsets[buffer] != UINT32_MAX) {
      nir_push_if(b, nir_ieq_imm(b, offset, inputs->buffers[buffer].buffer_size & ~3u));
      tail_word = nir_load_ssbo(b, 1, 32,
                               nir_imm_int(b, first_binding + inputs->source_count),
                               nir_imm_int(b, inputs->tail_offsets[buffer]),
                               .align_mul = 4, .access = ACCESS_NON_WRITEABLE);
      nir_push_else(b, NULL);
   }
   nir_def *word = nir_load_ssbo(b, 1, 32, nir_imm_int(b, first_binding + buffer), offset,
                                .align_mul = 4, .access = ACCESS_NON_WRITEABLE);
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
   return nir_format_unpack_rgba(b, nir_vec(b, words, DIV_ROUND_UP(bits, 32)), elem->src_format);
}

void
zink_vertex_inputs_finish(struct pipe_context *pctx, struct zink_vertex_inputs *inputs)
{
   if (inputs->tail)
      pipe_resource_release(pctx, inputs->tail);
}
