/* SPDX-License-Identifier: MIT */
#ifndef ZINK_VERTEX_PREPASS_H
#define ZINK_VERTEX_PREPASS_H
#include "zink_types.h"
#ifdef __cplusplus
extern "C" {
#endif
nir_shader *zink_vertex_prepass_prepare(const nir_shader *nir);
bool zink_vertex_prepass_draw(struct pipe_context *pctx,
                             const struct pipe_draw_info *info,
                             unsigned drawid_offset,
                             const struct pipe_draw_indirect_info *indirect,
                             const struct pipe_draw_start_count_bias *draws,
                             unsigned num_draws);
#ifdef __cplusplus
}
#endif
#endif
