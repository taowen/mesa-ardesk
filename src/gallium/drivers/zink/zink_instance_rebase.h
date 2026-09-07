/* SPDX-License-Identifier: MIT */
#ifndef ZINK_INSTANCE_REBASE_H
#define ZINK_INSTANCE_REBASE_H
#include "zink_types.h"
#ifdef __cplusplus
extern "C" {
#endif
bool zink_instance_rebase_draw(struct pipe_context *pctx,
                               const struct pipe_draw_info *info,
                               unsigned drawid_offset,
                               const struct pipe_draw_indirect_info *indirect,
                               const struct pipe_draw_start_count_bias *draws,
                               unsigned num_draws);
#ifdef __cplusplus
}
#endif
#endif
