/* SPDX-License-Identifier: MIT */
#include "zink_vertex_prepass.h"
#include "zink_context.h"
#include "zink_screen.h"
#include "compiler/nir/nir_builder.h"
#include "zink_vertex_pull.h"
#include "util/u_inlines.h"

/* Development path, deliberately not used to raise GL/Vulkan capabilities.
 * Reserved bindings are checked before any application shader is converted. */
#define PREPASS_UBO 15
#define PREPASS_SSBO ZINK_VERTEX_OUTPUT_SLOT
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
       nir->info.outputs_read ||
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
            if (instr->type == nir_instr_type_call)
               return NULL;
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (nir_intrinsic_has_image_dim(intr) ||
                (nir_intrinsic_infos[intr->intrinsic].flags & NIR_INTRINSIC_SUBGROUP))
               return NULL;
            if (intr->intrinsic == nir_intrinsic_load_input &&
                (intr->def.bit_size != 32 || !nir_src_is_const(intr->src[0]) ||
                 nir_src_as_uint(intr->src[0]) != 0 ||
                 nir_intrinsic_component(intr) + intr->num_components > 4))
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
source_vertex(nir_builder *b, const struct zink_vertex_inputs *inputs)
{
   nir_def *id = nir_channel(b, nir_load_global_invocation_id(b, 32), 0);
   if (inputs->index_size) {
      nir_def *offset = nir_iadd(b, parameter(b, 7), nir_imul_imm(b, id, inputs->index_size));
      return nir_iadd(b, zink_vertex_index_load(b, inputs, offset), parameter(b, 5));
   }
   return nir_iadd(b, id, parameter(b, 1));
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
   const struct zink_vertex_inputs *inputs = data;
   switch (intr->intrinsic) {
   case nir_intrinsic_load_input: {
      unsigned index = nir_intrinsic_base(intr);
      const struct pipe_vertex_element *elem = &inputs->elements[index];
      nir_def *id = nir_load_global_invocation_id(b, 32);
      nir_def *record = elem->instance_divisor ?
         nir_iadd(b, parameter(b, 3), nir_udiv_imm(b, nir_channel(b, id, 1), elem->instance_divisor)) :
         source_vertex(b, inputs);
      nir_def *rgba = zink_vertex_input_load(b, inputs, index, record);
      unsigned channels[4];
      for (unsigned c = 0; c < intr->num_components; c++)
         channels[c] = nir_intrinsic_component(intr) + c;
      value = nir_swizzle(b, rgba, channels, intr->num_components);
      break;
   }
   case nir_intrinsic_load_vertex_id:
      value = source_vertex(b, inputs);
      break;
   case nir_intrinsic_load_instance_id:
      value = nir_channel(b, nir_load_global_invocation_id(b, 32), 1);
      break;
   case nir_intrinsic_load_base_vertex:
      value = parameter(b, 5);
      break;
   case nir_intrinsic_load_first_vertex:
      value = parameter(b, inputs->index_size ? 5 : 1);
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
make_shaders(const nir_shader *source, struct zink_vertex_inputs *inputs,
             nir_shader **compute, nir_shader **replay)
{
   nir_builder rb = nir_builder_init_simple_shader(MESA_SHADER_VERTEX, source->options,
                                                  "zink vertex prepass replay");
   rb.shader->info = source->info;
   rb.shader->info.name = ralloc_strdup(rb.shader, "zink vertex prepass replay");
   rb.shader->info.internal = true;
   rb.shader->info.inputs_read = 0;
   rb.shader->info.num_textures = 0;
   BITSET_ZERO(rb.shader->info.textures_used);
   BITSET_ZERO(rb.shader->info.textures_used_by_txf);
   BITSET_ZERO(rb.shader->info.texture_buffers);
   BITSET_ZERO(rb.shader->info.samplers_used);
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
   /* Resolve implicit vertex LOD before changing stages. */
   nir_lower_tex_options tex_options = {.lower_invalid_implicit_lod = true};
   NIR_PASS(_, cs, nir_lower_tex, &tex_options);
   cs->info.stage = MESA_SHADER_COMPUTE;
   cs->info.internal = true;
   cs->info.prev_stage = cs->info.next_stage = MESA_SHADER_NONE;
   memset(&cs->info.cs, 0, sizeof(cs->info.cs));
   cs->info.workgroup_size[0] = cs->info.workgroup_size[1] = cs->info.workgroup_size[2] = 1;
   cs->info.num_inlinable_uniforms = 0;
   cs->info.num_ubos = PREPASS_UBO + 1;
   cs->info.first_ubo_is_default_ubo = true;
   cs->info.num_ssbos = PREPASS_SSBO + 1;
   for (unsigned i = 0; i < inputs->count; i++) {
      unsigned binding = inputs->bindings[i];
      if (inputs->texel_mask & BITFIELD_BIT(i)) {
         const struct glsl_type *type = glsl_sampler_type(GLSL_SAMPLER_DIM_BUF, false, false, GLSL_TYPE_UINT);
         nir_variable *var = nir_variable_create(cs, nir_var_uniform, type, "vertex_input_words");
         inputs->variables[i] = var;
         var->data.binding = var->data.driver_location = binding;
         BITSET_SET(cs->info.texture_buffers, binding);
      } else {
         add_buffer(cs, nir_var_mem_ssbo, binding, true);
      }
   }
   if (inputs->texel_count)
      cs->info.num_textures = inputs->texture_binding + inputs->texel_count;
   NIR_PASS(_, cs, nir_shader_intrinsics_pass, lower_to_compute, nir_metadata_control_flow, inputs);
   if (inputs->index_size) {
      nir_builder b = nir_builder_create(nir_shader_get_entrypoint(cs));
      b.cursor = nir_before_impl(b.impl);
      nir_def *id = nir_load_global_invocation_id(&b, 32);
      nir_def *vertex = nir_channel(&b, id, 0);
      nir_def *offset = nir_iadd(&b, parameter(&b, 7), nir_imul_imm(&b, vertex, inputs->index_size));
      nir_def *index = zink_vertex_index_load(&b, inputs, offset);
      nir_def *restart = inputs->primitive_restart ?
         nir_ieq_imm(&b, index, inputs->restart_index) : nir_imm_false(&b);
      nir_push_if(&b, nir_ieq_imm(&b, nir_channel(&b, id, 1), 0));
      nir_store_ssbo(&b, nir_bcsel(&b, restart, nir_imm_int(&b, UINT32_MAX), vertex),
                     nir_imm_int(&b, PREPASS_SSBO),
                     nir_iadd(&b, parameter(&b, 6), nir_imul_imm(&b, vertex, 4)),
                     .write_mask = 1, .align_mul = 4);
      nir_pop_if(&b, NULL);
      nir_push_if(&b, restart);
      nir_jump(&b, nir_jump_return);
      nir_pop_if(&b, NULL);
      nir_progress(true, b.impl, nir_metadata_none);
      NIR_PASS(_, cs, nir_lower_returns);
   }
   nir_foreach_variable_with_modes_safe(var, cs, nir_var_shader_out | nir_var_shader_in)
      exec_node_remove(&var->node);
   cs->info.outputs_written = cs->info.outputs_read = 0;
   cs->num_outputs = cs->num_inputs = 0;
   cs->info.inputs_read = 0;
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
       indirect || num_draws != 1 || !draws[0].count || !info->instance_count ||
       ctx->gfx_stages[MESA_SHADER_TESS_CTRL] || ctx->gfx_stages[MESA_SHADER_TESS_EVAL] ||
       ctx->gfx_stages[MESA_SHADER_GEOMETRY] || ctx->num_so_targets || ctx->render_condition_active ||
       (ctx->bs && ctx->bs->active_queries.entries) || !list_is_empty(&ctx->suspended_queries) ||
       screen->info.props.limits.maxPerStageDescriptorUniformBuffers < PREPASS_UBO + 1 ||
       screen->info.props.limits.maxPerStageDescriptorStorageBuffers < PREPASS_SSBO + 1 ||
       draws[0].count > screen->info.props.limits.maxComputeWorkGroupCount[0] ||
       info->instance_count > screen->info.props.limits.maxComputeWorkGroupCount[1])
      return false;
   struct zink_vertex_inputs inputs = {0};
   if (!zink_vertex_inputs_prepare(ctx, vs->vertex_prepass_nir, info, draws, &inputs))
      return false;
   uint64_t vertex_size = (uint64_t)draws[0].count * info->instance_count * PREPASS_STRIDE;
   uint64_t size = vertex_size + (info->index_size ? (uint64_t)draws[0].count * 4 : 0);
   if (size > UINT32_MAX || size > screen->info.props.limits.maxStorageBufferRange) {
      zink_vertex_inputs_finish(pctx, &inputs);
      return false;
   }
   struct pipe_resource *output = pipe_buffer_create(pctx->screen, PIPE_BIND_SHADER_BUFFER | PIPE_BIND_INDEX_BUFFER,
                                                     PIPE_USAGE_DEFAULT, size);
   if (!output) {
      zink_vertex_inputs_finish(pctx, &inputs);
      return false;
   }
   ctx->vertex_prepass_active = true;
   nir_shader *cs, *replay;
   make_shaders(vs->vertex_prepass_nir, &inputs, &cs, &replay);
   struct pipe_compute_state cs_state = {.ir_type = PIPE_SHADER_IR_NIR, .prog = cs};
   struct pipe_shader_state vs_state = {.type = PIPE_SHADER_IR_NIR, .ir.nir = replay};
   void *compute_cso = pctx->create_compute_state(pctx, &cs_state);
   void *replay_cso = pctx->create_vs_state(pctx, &vs_state);
   if (!compute_cso || !replay_cso) {
      if (compute_cso) pctx->delete_compute_state(pctx, compute_cso);
      if (replay_cso) pctx->delete_vs_state(pctx, replay_cso);
      zink_vertex_inputs_finish(pctx, &inputs);
      pipe_resource_release(pctx, output);
      ctx->vertex_prepass_active = false;
      return false;
   }
   struct zink_compute_program *saved_compute = ctx->curr_compute;
   struct pipe_sampler_view *saved_views[PIPE_MAX_SAMPLERS] = {0};
   void *saved_samplers[PIPE_MAX_SAMPLERS];
   unsigned saved_view_count = ctx->di.num_sampler_views[MESA_SHADER_COMPUTE];
   unsigned saved_sampler_count = ctx->di.num_samplers[MESA_SHADER_COMPUTE];
   unsigned vertex_view_count = ctx->di.num_sampler_views[MESA_SHADER_VERTEX];
   unsigned vertex_sampler_count = ctx->di.num_samplers[MESA_SHADER_VERTEX];
   unsigned sampler_count = MAX2(saved_sampler_count, vertex_sampler_count);
   for (unsigned i = 0; i < PIPE_MAX_SAMPLERS; i++) {
      /* A deleted GL texture can remain alive only through this binding. */
      pipe_sampler_view_reference(&saved_views[i], ctx->sampler_views[MESA_SHADER_COMPUTE][i]);
      saved_samplers[i] = ctx->sampler_states[MESA_SHADER_COMPUTE][i];
   }
   struct pipe_sampler_view *compute_views[PIPE_MAX_SAMPLERS] = {0};
   memcpy(compute_views, ctx->sampler_views[MESA_SHADER_VERTEX], vertex_view_count * sizeof(compute_views[0]));
   unsigned compute_view_count = vertex_view_count;
   if (inputs.texel_count) {
      memcpy(compute_views + inputs.texture_binding, inputs.views, inputs.texel_count * sizeof(compute_views[0]));
      compute_view_count = inputs.texture_binding + inputs.texel_count;
   }
   pctx->set_sampler_views(pctx, MESA_SHADER_COMPUTE, 0, compute_view_count,
                          saved_view_count > compute_view_count ? saved_view_count - compute_view_count : 0,
                          compute_views);
   void *vertex_samplers[PIPE_MAX_SAMPLERS] = {0};
   for (unsigned i = 0; i < vertex_sampler_count; i++)
      vertex_samplers[i] = ctx->sampler_states[MESA_SHADER_VERTEX][i];
   pctx->bind_sampler_states(pctx, MESA_SHADER_COMPUTE, 0, sampler_count, vertex_samplers);
   ctx->di.num_samplers[MESA_SHADER_COMPUTE] = vertex_sampler_count;
   struct pipe_constant_buffer saved_ubos[PREPASS_UBO + 1];
   struct pipe_shader_buffer saved_ssbos[PREPASS_SSBO + 1];
   memcpy(saved_ubos, ctx->ubos[MESA_SHADER_COMPUTE], sizeof(saved_ubos));
   memcpy(saved_ssbos, ctx->ssbos[MESA_SHADER_COMPUTE], sizeof(saved_ssbos));
   unsigned saved_writable = ctx->writable_ssbos[MESA_SHADER_COMPUTE];
   struct pipe_constant_buffer saved_vs_ubo = ctx->ubos[MESA_SHADER_VERTEX][PREPASS_UBO];
   struct pipe_shader_buffer saved_vs_ssbo = ctx->ssbos[MESA_SHADER_VERTEX][PREPASS_SSBO];
   unsigned saved_vs_writable = (ctx->writable_ssbos[MESA_SHADER_VERTEX] >> PREPASS_SSBO) & 1;
   uint32_t params[8] = {draws[0].count, info->index_size ? 0 : draws[0].start,
                         info->instance_count, info->start_instance, drawid_offset,
                         info->index_size ? draws[0].index_bias : 0, vertex_size,
                         draws[0].start * info->index_size};
   struct pipe_constant_buffer parameters = {.user_buffer = params, .buffer_size = sizeof(params)};
   struct pipe_shader_buffer generated = {.buffer = output, .buffer_size = size};
   for (unsigned i = 0; i < PREPASS_UBO; i++)
      bind_ubo(pctx, MESA_SHADER_COMPUTE, i, &ctx->ubos[MESA_SHADER_VERTEX][i]);
   pctx->set_constant_buffer(pctx, MESA_SHADER_COMPUTE, PREPASS_UBO, &parameters);
   pctx->set_shader_buffers(pctx, MESA_SHADER_COMPUTE, 0, PREPASS_SSBO,
                           ctx->ssbos[MESA_SHADER_VERTEX], ctx->writable_ssbos[MESA_SHADER_VERTEX]);
   if (inputs.ssbo_count)
      pctx->set_shader_buffers(pctx, MESA_SHADER_COMPUTE, inputs.first_binding,
                              inputs.ssbo_count, inputs.ssbo_buffers, 0);
   pctx->set_shader_buffers(pctx, MESA_SHADER_COMPUTE, PREPASS_SSBO, 1, &generated, 1);
   pctx->bind_compute_state(pctx, compute_cso);
   struct pipe_grid_info grid = {.work_dim = 2, .block = {1, 1, 1},
                                .grid = {draws[0].count, info->instance_count, 1}};
   pctx->launch_grid(pctx, &grid);
   pctx->memory_barrier(pctx, PIPE_BARRIER_SHADER_BUFFER | PIPE_BARRIER_INDEX_BUFFER);
   pctx->set_constant_buffer(pctx, MESA_SHADER_VERTEX, PREPASS_UBO, &parameters);
   pctx->set_shader_buffers(pctx, MESA_SHADER_VERTEX, PREPASS_SSBO, 1, &generated, 0);
   pctx->bind_vs_state(pctx, replay_cso);
   struct pipe_draw_info replay_info = *info;
   struct pipe_draw_start_count_bias replay_draw = *draws;
   if (info->index_size) {
      replay_info.index_size = 4;
      replay_info.has_user_indices = false;
      replay_info.index.resource = output;
      replay_info.restart_index = UINT32_MAX;
      replay_info.index_bounds_valid = true;
      replay_info.min_index = 0;
      replay_info.max_index = draws[0].count - 1;
      replay_info.index_bias_varies = false;
      replay_draw.start = vertex_size / 4;
      replay_draw.index_bias = 0;
   }
   pctx->draw_vbo(pctx, &replay_info, drawid_offset, NULL, &replay_draw, 1);
   pctx->bind_vs_state(pctx, vs);
   bind_ubo(pctx, MESA_SHADER_VERTEX, PREPASS_UBO, &saved_vs_ubo);
   pctx->set_shader_buffers(pctx, MESA_SHADER_VERTEX, PREPASS_SSBO, 1, &saved_vs_ssbo, saved_vs_writable);
   pctx->bind_compute_state(pctx, saved_compute);
   for (unsigned i = 0; i <= PREPASS_UBO; i++)
      bind_ubo(pctx, MESA_SHADER_COMPUTE, i, &saved_ubos[i]);
   pctx->set_shader_buffers(pctx, MESA_SHADER_COMPUTE, 0, PREPASS_SSBO + 1, saved_ssbos, saved_writable);
   pctx->set_sampler_views(pctx, MESA_SHADER_COMPUTE, 0, saved_view_count,
                          compute_view_count > saved_view_count ? compute_view_count - saved_view_count : 0,
                          saved_views);
   pctx->bind_sampler_states(pctx, MESA_SHADER_COMPUTE, 0, sampler_count, saved_samplers);
   ctx->di.num_samplers[MESA_SHADER_COMPUTE] = saved_sampler_count;
   for (unsigned i = 0; i < PIPE_MAX_SAMPLERS; i++)
      pipe_sampler_view_reference(&saved_views[i], NULL);
   pctx->delete_compute_state(pctx, compute_cso);
   pctx->delete_vs_state(pctx, replay_cso);
   zink_vertex_inputs_finish(pctx, &inputs);
   pipe_resource_release(pctx, output);
   ctx->vertex_prepass_active = false;
   mesa_logi("ZINK_VERTEX_PREPASS draw vertices=%u instances=%u inputs=%u index_size=%u restart=%u restart_index=%u input_binding=%s input_texels=%u input_ssbos=%u",
             draws[0].count, info->instance_count, inputs.count, (unsigned)info->index_size,
             (unsigned)info->primitive_restart, info->restart_index, inputs.texel_count ? (inputs.ssbo_count ? "mixed" : "texel") : "ssbo",
             inputs.texel_count, inputs.ssbo_count);
   return true;
}
