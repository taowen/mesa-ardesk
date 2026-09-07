/* SPDX-License-Identifier: MIT */
#include "zink_vertex_prepass.h"
#include "zink_context.h"
#include "zink_screen.h"
#include "compiler/nir/nir_builder.h"
#include "util/u_inlines.h"

/* Development path, deliberately not used to raise GL/Vulkan capabilities.
 * Reserved bindings are checked before any application shader is converted. */
#define PREPASS_UBO 15
#define PREPASS_SSBO 15
#define PREPASS_STRIDE (VARYING_SLOT_MAX * 16)

static bool
supported_system_value(unsigned value)
{
   switch (value) {
   case SYSTEM_VALUE_VERTEX_ID:
   case SYSTEM_VALUE_INSTANCE_ID:
   case SYSTEM_VALUE_BASE_VERTEX:
   case SYSTEM_VALUE_FIRST_VERTEX:
   case SYSTEM_VALUE_BASE_INSTANCE:
   case SYSTEM_VALUE_DRAW_ID:
      return true;
   default:
      return false;
   }
}

nir_shader *
zink_vertex_prepass_prepare(const nir_shader *nir)
{
   if (nir->info.stage != MESA_SHADER_VERTEX || !nir->info.io_lowered ||
       nir->info.inputs_read || nir->info.outputs_read || nir->info.num_textures ||
       nir->info.num_images || nir->info.num_abos || nir->info.uses_bindless ||
       nir->info.clip_distance_array_size || nir->info.cull_distance_array_size ||
       nir->info.vs.window_space_position || nir->xfb_info ||
       nir->info.has_transform_feedback_varyings || nir->info.view_mask ||
       nir->info.num_ubos > PREPASS_UBO || nir->info.num_ssbos > PREPASS_SSBO)
      return NULL;
   for (unsigned i = 0; i < SYSTEM_VALUE_MAX; i++)
      if (BITSET_TEST(nir->info.system_values_read, i) && !supported_system_value(i))
         return NULL;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_tex || instr->type == nir_instr_type_call)
               return NULL;
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (nir_intrinsic_has_image_dim(intr) ||
                (nir_intrinsic_infos[intr->intrinsic].flags & NIR_INTRINSIC_SUBGROUP))
               return NULL;
            if (intr->intrinsic == nir_intrinsic_store_output &&
                (intr->src[0].ssa->bit_size != 32 || !nir_src_is_const(intr->src[1]) ||
                 nir_src_as_uint(intr->src[1]) != 0))
               return NULL;
         }
      }
   }
   return nir_shader_clone(NULL, nir);
}

static nir_def *
parameter(nir_builder *b, unsigned index)
{
   return nir_load_ubo(b, 1, 32, nir_imm_int(b, PREPASS_UBO),
                       nir_imm_int(b, index * 4), .align_mul = 4,
                       .range_base = index * 4, .range = 4);
}

static nir_def *
record_offset(nir_builder *b, bool compute)
{
   nir_def *vertex, *instance;
   if (compute) {
      nir_def *id = nir_load_global_invocation_id(b, 32);
      vertex = nir_channel(b, id, 0);
      instance = nir_channel(b, id, 1);
   } else {
      vertex = nir_isub(b, nir_load_vertex_id(b), parameter(b, 1));
      instance = nir_load_instance_id(b);
   }
   return nir_imul_imm(b, nir_iadd(b, vertex, nir_imul(b, instance, parameter(b, 0))),
                       PREPASS_STRIDE);
}

static unsigned
output_offset(nir_intrinsic_instr *intr)
{
   return nir_intrinsic_io_semantics(intr).location * 16 + nir_intrinsic_component(intr) * 4;
}

static void
add_buffer(nir_shader *nir, nir_variable_mode mode, unsigned binding, bool readonly)
{
   struct glsl_struct_field field = {
      .name = "words",
      .type = glsl_array_type(glsl_uint_type(), mode == nir_var_mem_ubo ? 8 : 0, 4),
   };
   const struct glsl_type *type = glsl_interface_type(&field, 1,
                           GLSL_INTERFACE_PACKING_STD430, false, "prepass_buffer");
   nir_variable *var = nir_variable_create(nir, mode, type, "prepass_buffer");
   var->interface_type = type;
   var->data.driver_location = var->data.binding = binding;
   if (readonly)
      var->data.access = ACCESS_NON_WRITEABLE;
}

static bool
lower_to_compute(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *value = NULL;
   switch (intr->intrinsic) {
   case nir_intrinsic_load_vertex_id:
      value = nir_iadd(b, nir_channel(b, nir_load_global_invocation_id(b, 32), 0), parameter(b, 1));
      break;
   case nir_intrinsic_load_instance_id:
      value = nir_channel(b, nir_load_global_invocation_id(b, 32), 1);
      break;
   case nir_intrinsic_load_base_vertex:
      value = nir_imm_int(b, 0); /* Only non-indexed draws enter this path. */
      break;
   case nir_intrinsic_load_first_vertex:
      value = parameter(b, 1);
      break;
   case nir_intrinsic_load_base_instance:
      value = parameter(b, 3);
      break;
   case nir_intrinsic_load_draw_id:
      value = parameter(b, 4);
      break;
   case nir_intrinsic_store_output:
      nir_store_ssbo(b, intr->src[0].ssa, nir_imm_int(b, PREPASS_SSBO),
                     nir_iadd_imm(b, record_offset(b, true), output_offset(intr)),
                     .write_mask = nir_intrinsic_write_mask(intr), .align_mul = 4);
      nir_instr_remove(&intr->instr);
      return true;
   default:
      return false;
   }
   nir_def_replace(&intr->def, value);
   return true;
}

static void
make_shaders(const nir_shader *source, nir_shader **compute, nir_shader **replay)
{
   nir_builder rb = nir_builder_init_simple_shader(MESA_SHADER_VERTEX, source->options,
                                                  "zink vertex prepass replay");
   rb.shader->info = source->info;
   rb.shader->info.name = ralloc_strdup(rb.shader, "zink vertex prepass replay");
   rb.shader->info.internal = true;
   rb.shader->info.num_ubos = PREPASS_UBO + 1;
   rb.shader->info.first_ubo_is_default_ubo = true;
   rb.shader->info.num_ssbos = PREPASS_SSBO + 1;
   rb.shader->info.num_inlinable_uniforms = 0;
   nir_foreach_function_impl(impl, source) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_store_output)
               continue;
            nir_def *value = nir_load_ssbo(&rb, intr->num_components, 32,
                                          nir_imm_int(&rb, PREPASS_SSBO),
                                          nir_iadd_imm(&rb, record_offset(&rb, false), output_offset(intr)),
                                          .align_mul = 4, .access = ACCESS_NON_WRITEABLE);
            nir_intrinsic_instr *out = nir_intrinsic_instr_create(rb.shader, nir_intrinsic_store_output);
            out->num_components = intr->num_components;
            out->src[0] = nir_src_for_ssa(value);
            out->src[1] = nir_src_for_ssa(nir_imm_int(&rb, 0));
            nir_intrinsic_copy_const_indices(out, intr);
            nir_builder_instr_insert(&rb, &out->instr);
         }
      }
   }
   add_buffer(rb.shader, nir_var_mem_ubo, PREPASS_UBO, true);
   add_buffer(rb.shader, nir_var_mem_ssbo, PREPASS_SSBO, true);
   nir_shader_gather_info(rb.shader, nir_shader_get_entrypoint(rb.shader));
   *replay = rb.shader;

   nir_shader *cs = nir_shader_clone(NULL, source);
   cs->info.stage = MESA_SHADER_COMPUTE;
   cs->info.internal = true;
   cs->info.prev_stage = cs->info.next_stage = MESA_SHADER_NONE;
   memset(&cs->info.cs, 0, sizeof(cs->info.cs));
   cs->info.workgroup_size[0] = cs->info.workgroup_size[1] = cs->info.workgroup_size[2] = 1;
   cs->info.num_inlinable_uniforms = 0;
   cs->info.num_ubos = PREPASS_UBO + 1;
   cs->info.first_ubo_is_default_ubo = true;
   cs->info.num_ssbos = PREPASS_SSBO + 1;
   NIR_PASS(_, cs, nir_shader_intrinsics_pass, lower_to_compute, nir_metadata_control_flow, NULL);
   nir_foreach_variable_with_modes_safe(var, cs, nir_var_shader_out)
      exec_node_remove(&var->node);
   cs->info.outputs_written = cs->info.outputs_read = 0;
   cs->num_outputs = 0;
   add_buffer(cs, nir_var_mem_ubo, PREPASS_UBO, true);
   add_buffer(cs, nir_var_mem_ssbo, PREPASS_SSBO, false);
   nir_shader_gather_info(cs, nir_shader_get_entrypoint(cs));
   *compute = cs;
}

static void
bind_ubo(struct pipe_context *pctx, mesa_shader_stage stage, unsigned slot,
         const struct pipe_constant_buffer *buffer)
{
   /* An empty struct is not the driver's unbind operation. */
   pctx->set_constant_buffer(pctx, stage, slot,
                            buffer->buffer || buffer->user_buffer ? buffer : NULL);
}

bool
zink_vertex_prepass_draw(struct pipe_context *pctx, const struct pipe_draw_info *info,
                         unsigned drawid_offset, const struct pipe_draw_indirect_info *indirect,
                         const struct pipe_draw_start_count_bias *draws, unsigned num_draws)
{
   struct zink_context *ctx = zink_context(pctx);
   struct zink_screen *screen = zink_screen(pctx->screen);
   struct zink_shader *vs = ctx->gfx_stages[MESA_SHADER_VERTEX];
   if (ctx->vertex_prepass_active || !vs || !vs->vertex_prepass_nir ||
       indirect || info->index_size || num_draws != 1 || !draws[0].count || !info->instance_count ||
       ctx->gfx_stages[MESA_SHADER_TESS_CTRL] || ctx->gfx_stages[MESA_SHADER_TESS_EVAL] ||
       ctx->gfx_stages[MESA_SHADER_GEOMETRY] || ctx->num_so_targets || ctx->render_condition_active ||
       (ctx->bs && ctx->bs->active_queries.entries) || !list_is_empty(&ctx->suspended_queries) ||
       screen->info.props.limits.maxPerStageDescriptorUniformBuffers < PREPASS_UBO + 1 ||
       screen->info.props.limits.maxPerStageDescriptorStorageBuffers < PREPASS_SSBO + 1 ||
       draws[0].count > screen->info.props.limits.maxComputeWorkGroupCount[0] ||
       info->instance_count > screen->info.props.limits.maxComputeWorkGroupCount[1])
      return false;
   uint64_t size = (uint64_t)draws[0].count * info->instance_count * PREPASS_STRIDE;
   if (size > UINT32_MAX || size > screen->info.props.limits.maxStorageBufferRange)
      return false;
   struct pipe_resource *output = pipe_buffer_create(pctx->screen, PIPE_BIND_SHADER_BUFFER,
                                                     PIPE_USAGE_DEFAULT, size);
   if (!output)
      return false;
   ctx->vertex_prepass_active = true;
   nir_shader *cs, *replay;
   make_shaders(vs->vertex_prepass_nir, &cs, &replay);
   struct pipe_compute_state cs_state = {.ir_type = PIPE_SHADER_IR_NIR, .prog = cs};
   struct pipe_shader_state vs_state = {.type = PIPE_SHADER_IR_NIR, .ir.nir = replay};
   void *compute_cso = pctx->create_compute_state(pctx, &cs_state);
   void *replay_cso = pctx->create_vs_state(pctx, &vs_state);
   if (!compute_cso || !replay_cso) {
      if (compute_cso) pctx->delete_compute_state(pctx, compute_cso);
      if (replay_cso) pctx->delete_vs_state(pctx, replay_cso);
      pipe_resource_release(pctx, output);
      ctx->vertex_prepass_active = false;
      return false;
   }
   struct zink_compute_program *saved_compute = ctx->curr_compute;
   struct pipe_constant_buffer saved_ubos[PREPASS_UBO + 1];
   struct pipe_shader_buffer saved_ssbos[PREPASS_SSBO + 1];
   memcpy(saved_ubos, ctx->ubos[MESA_SHADER_COMPUTE], sizeof(saved_ubos));
   memcpy(saved_ssbos, ctx->ssbos[MESA_SHADER_COMPUTE], sizeof(saved_ssbos));
   unsigned saved_writable = ctx->writable_ssbos[MESA_SHADER_COMPUTE];
   struct pipe_constant_buffer saved_vs_ubo = ctx->ubos[MESA_SHADER_VERTEX][PREPASS_UBO];
   struct pipe_shader_buffer saved_vs_ssbo = ctx->ssbos[MESA_SHADER_VERTEX][PREPASS_SSBO];
   unsigned saved_vs_writable = (ctx->writable_ssbos[MESA_SHADER_VERTEX] >> PREPASS_SSBO) & 1;
   uint32_t params[8] = {draws[0].count, draws[0].start, info->instance_count, info->start_instance, drawid_offset};
   struct pipe_constant_buffer parameters = {.user_buffer = params, .buffer_size = sizeof(params)};
   struct pipe_shader_buffer generated = {.buffer = output, .buffer_size = size};
   for (unsigned i = 0; i < PREPASS_UBO; i++)
      bind_ubo(pctx, MESA_SHADER_COMPUTE, i, &ctx->ubos[MESA_SHADER_VERTEX][i]);
   pctx->set_constant_buffer(pctx, MESA_SHADER_COMPUTE, PREPASS_UBO, &parameters);
   pctx->set_shader_buffers(pctx, MESA_SHADER_COMPUTE, 0, PREPASS_SSBO,
                           ctx->ssbos[MESA_SHADER_VERTEX], ctx->writable_ssbos[MESA_SHADER_VERTEX]);
   pctx->set_shader_buffers(pctx, MESA_SHADER_COMPUTE, PREPASS_SSBO, 1, &generated, 1);
   pctx->bind_compute_state(pctx, compute_cso);
   struct pipe_grid_info grid = {.work_dim = 2, .block = {1, 1, 1},
                                .grid = {draws[0].count, info->instance_count, 1}};
   pctx->launch_grid(pctx, &grid);
   pctx->memory_barrier(pctx, PIPE_BARRIER_SHADER_BUFFER);
   pctx->set_constant_buffer(pctx, MESA_SHADER_VERTEX, PREPASS_UBO, &parameters);
   pctx->set_shader_buffers(pctx, MESA_SHADER_VERTEX, PREPASS_SSBO, 1, &generated, 0);
   pctx->bind_vs_state(pctx, replay_cso);
   pctx->draw_vbo(pctx, info, drawid_offset, NULL, draws, 1);
   pctx->bind_vs_state(pctx, vs);
   bind_ubo(pctx, MESA_SHADER_VERTEX, PREPASS_UBO, &saved_vs_ubo);
   pctx->set_shader_buffers(pctx, MESA_SHADER_VERTEX, PREPASS_SSBO, 1, &saved_vs_ssbo, saved_vs_writable);
   pctx->bind_compute_state(pctx, saved_compute);
   for (unsigned i = 0; i <= PREPASS_UBO; i++)
      bind_ubo(pctx, MESA_SHADER_COMPUTE, i, &saved_ubos[i]);
   pctx->set_shader_buffers(pctx, MESA_SHADER_COMPUTE, 0, PREPASS_SSBO + 1, saved_ssbos, saved_writable);
   pctx->delete_compute_state(pctx, compute_cso);
   pctx->delete_vs_state(pctx, replay_cso);
   pipe_resource_release(pctx, output);
   ctx->vertex_prepass_active = false;
   mesa_logi("ZINK_VERTEX_PREPASS draw vertices=%u instances=%u", draws[0].count, info->instance_count);
   return true;
}
