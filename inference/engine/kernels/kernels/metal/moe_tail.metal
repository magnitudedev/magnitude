// moe_tail: the tail of a feed-forward with a dense branch beside the routed
// experts, all F32:
//   f = rms(dense) * dense_norm + rms(routed) * routed_norm
//   result = (residual + rms(f) * norm) * scale
// One threadgroup per row. The combined row is formed again (the same
// operations, so the same bits) by the pass that reduces it and by the pass
// that publishes, instead of being held in threadgroup memory.
#include "lib/core/activation.h"
#include "lib/core/reduce.h"

typedef ELEMENT_OF(SEISMIC_NORM) Norm;

// f[i] of one row: both branches normalized with their inverses and norms.
struct Combined {
    device const float *dense;
    ulong dense_stride;
    device const float *routed;
    ulong routed_stride;
    device const uchar *dense_norm;
    ulong dense_norm_stride;
    device const uchar *routed_norm;
    ulong routed_norm_stride;
    float dense_inverse, routed_inverse;
    float at(ulong i) const {
        const float from_dense = dense[i * dense_stride] * dense_inverse
            * element::at<Norm>(dense_norm, i * dense_norm_stride);
        const float from_routed = routed[i * routed_stride] * routed_inverse
            * element::at<Norm>(routed_norm, i * routed_norm_stride);
        return from_dense + from_routed;
    }
};

kernel void moe_tail(
    device const float *residual [[buffer(SEISMIC_BUFFER_RESIDUAL)]],
    device const float *dense [[buffer(SEISMIC_BUFFER_DENSE)]],
    device const float *routed [[buffer(SEISMIC_BUFFER_ROUTED)]],
    device const uchar *dense_norm [[buffer(SEISMIC_BUFFER_DENSE_NORM)]],
    device const uchar *routed_norm [[buffer(SEISMIC_BUFFER_ROUTED_NORM)]],
    device const uchar *norm [[buffer(SEISMIC_BUFFER_NORM)]],
    device float *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partials[8];
    const ulong width = SEISMIC_DIM_D;
    const float epsilon = as_type<float>(uint(SEISMIC_PARAM_EPSILON));
    const float scale = as_type<float>(uint(SEISMIC_PARAM_SCALE));
    device const float *d = dense + ulong(row) * SEISMIC_DENSE_STRIDE_0;
    device const float *r = routed + ulong(row) * SEISMIC_ROUTED_STRIDE_0;
    float dense_squares = 0.0f, routed_squares = 0.0f;
    for (ulong i = thread_index; i < width; i += 256u) {
        const float dv = d[i * SEISMIC_DENSE_STRIDE_1], rv = r[i * SEISMIC_ROUTED_STRIDE_1];
        dense_squares = metal::fma(dv, dv, dense_squares);
        routed_squares = metal::fma(rv, rv, routed_squares);
    }
    const float dense_total = reduce::group_sum<8>(dense_squares, partials, sg, lane);
    const float routed_total = reduce::group_sum<8>(routed_squares, partials, sg, lane);
    const Combined combined{d, SEISMIC_DENSE_STRIDE_1, r, SEISMIC_ROUTED_STRIDE_1, dense_norm,
        SEISMIC_DENSE_NORM_STRIDE_0, routed_norm, SEISMIC_ROUTED_NORM_STRIDE_0,
        metal::rsqrt(dense_total / float(width) + epsilon), metal::rsqrt(routed_total / float(width) + epsilon)};
    float squares = 0.0f;
    for (ulong i = thread_index; i < width; i += 256u) {
        const float f = combined.at(i);
        squares = metal::fma(f, f, squares);
    }
    const float inverse = metal::rsqrt(reduce::group_sum<8>(squares, partials, sg, lane) / float(width) + epsilon);
    device const float *base = residual + ulong(row) * SEISMIC_RESIDUAL_STRIDE_0;
    for (ulong i = thread_index; i < width; i += 256u) {
        const float normalized = combined.at(i) * inverse * element::at<Norm>(norm, i * SEISMIC_NORM_STRIDE_0);
        result[ulong(row) * SEISMIC_RESULT_0_STRIDE_0 + i * SEISMIC_RESULT_0_STRIDE_1] =
            (base[i * SEISMIC_RESIDUAL_STRIDE_1] + normalized) * scale;
    }
}
