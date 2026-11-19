// What every probe kernel declares: the floor's extensions it uses and the
// launch's arguments as the generated prefix declares them
// (`seismic_runtime::native::abi::vulkan`), so formation reads them as it
// reads every launch's: the push constants, then the argument block (buffer
// addresses, the scalar-result address, then the words) through its two
// block types, the addresses through the argument block and the words, past
// the addresses, through the input block.
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_control_flow_attributes : require
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
layout(push_constant, scalar) uniform seismic_push_t { uint64_t seismic_arguments; uint64_t seismic_tables; };
layout(buffer_reference, scalar, buffer_reference_align = 8) readonly buffer seismic_argument_block_t { uint64_t w[]; };
layout(buffer_reference, scalar, buffer_reference_align = 8) readonly buffer seismic_input_block_t { uint64_t w[]; };
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer probe_u32 { uint v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer probe_f32 { float v[]; };
// Buffer `index`'s address.
#define PROBE_ARGUMENT(index) (seismic_argument_block_t(seismic_arguments).w[index])
// Word `index` of a launch of `buffers` buffers (a uint64_t literal).
#define PROBE_WORD(buffers, index) (seismic_input_block_t(seismic_arguments + ((buffers) + 1ul) * 8ul).w[index])
