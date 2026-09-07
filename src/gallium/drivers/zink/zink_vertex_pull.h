/* SPDX-License-Identifier: MIT */
#ifndef ZINK_VERTEX_PULL_H
#define ZINK_VERTEX_PULL_H
#include "zink_types.h"
#include "compiler/nir/nir_builder.h"
#define ZINK_VERTEX_INPUT_BUFFERS 15
struct zink_vertex_inputs {
   struct pipe_vertex_element elements[PIPE_MAX_ATTRIBS];
   unsigned slots[PIPE_MAX_ATTRIBS];
   unsigned offsets[PIPE_MAX_ATTRIBS];
   struct pipe_shader_buffer buffers[ZINK_VERTEX_INPUT_BUFFERS];
   unsigned first_binding;
   unsigned count;
   unsigned source_count;
   unsigned tail_offsets[ZINK_VERTEX_INPUT_BUFFERS];
   struct pipe_resource *tail;
};

bool zink_vertex_inputs_prepare(struct zink_context *ctx, const nir_shader *nir,
                               const struct pipe_draw_info *info,
                               const struct pipe_draw_start_count_bias *draw,
                               struct zink_vertex_inputs *inputs);
nir_def *zink_vertex_input_load(nir_builder *b, const struct zink_vertex_inputs *inputs,
                                unsigned index, nir_def *record);
void zink_vertex_inputs_finish(struct pipe_context *pctx, struct zink_vertex_inputs *inputs);
#endif
