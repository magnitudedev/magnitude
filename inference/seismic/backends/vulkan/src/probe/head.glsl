// What every probe kernel declares: the floor's extensions it uses and the
// backend's argument block (buffer addresses, the scalar-result address,
// then the words), as the generated prefix declares them.
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_control_flow_attributes : require
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
layout(push_constant, scalar) uniform probe_push_t { uint64_t probe_arguments; };
layout(buffer_reference, scalar, buffer_reference_align = 8) readonly buffer probe_block_t { uint64_t w[]; };
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer probe_u32 { uint v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer probe_f32 { float v[]; };
#define PROBE_ARGUMENT(index) (probe_block_t(probe_arguments).w[index])
