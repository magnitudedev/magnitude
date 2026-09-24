// qwen_features_rows: the final RMS normalization of the `out_rows` hidden
// rows, published in the activation type. One threadgroup per output row.
#include "common/projection.h"

#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_BF16)
typedef packets::bf16 activation;
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_F16)
typedef packets::f16 activation;
#else
#error "qwen_features_rows requires a bf16 or f16 activation"
#endif
#if defined(SEISMIC_NORM_REPRESENTATION_F32)
typedef packets::f32 norm_element;
#elif defined(SEISMIC_NORM_REPRESENTATION_F16)
typedef packets::f16 norm_element;
#elif defined(SEISMIC_NORM_REPRESENTATION_BF16)
typedef packets::bf16 norm_element;
#else
#error "qwen_features_rows requires a dense norm"
#endif

kernel void qwen_features_rows(
    device const float *hidden [[buffer(SEISMIC_BUFFER_HIDDEN)]],
    device const uchar *norm [[buffer(SEISMIC_BUFFER_NORM)]],
    device const int *out_rows [[buffer(SEISMIC_BUFFER_OUT_ROWS)]],
    device uchar *features [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partials[8];
    threadgroup float inverse_value;
    const uint width = uint(SEISMIC_DIM_D);
    projection::input_rms<activation, norm_element> in{hidden, SEISMIC_HIDDEN_STRIDE_0,
        SEISMIC_HIDDEN_STRIDE_1, norm, SEISMIC_NORM_STRIDE_0,
        as_type<float>(uint(SEISMIC_PARAM_EPSILON)), width, {out_rows}};
    float squares = 0.0f;
    for (uint i = thread_index; i < width; i += 256u) {
        float v = in.norm_input(row, 0, i);
        squares = metal::fma(v, v, squares);
    }
    squares = simd_sum(squares);
    if (lane == 0)
        partials[sg] = squares;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index == 0) {
        float total = 0.0f;
        for (uint j = 0; j < 8; ++j)
            total += partials[j];
        inverse_value = metal::rsqrt(total / float(width) + in.epsilon());
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inverse = inverse_value;
    device typename activation::storage *out = reinterpret_cast<device typename activation::storage *>(features);
    for (uint i = thread_index; i < width; i += 256u)
        out[ulong(row) * SEISMIC_RESULT_0_STRIDE_0 + ulong(i) * SEISMIC_RESULT_0_STRIDE_1] =
            activation::store(in.value(row, i, inverse));
}
