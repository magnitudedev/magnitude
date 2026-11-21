// moe_tail: the tail of a feed-forward with a dense branch beside the routed
// experts, all F32:
//   f = rms(dense) * dense_norm + rms(routed) * routed_norm
//   result = (residual + rms(f) * norm) * scale
// One threadgroup per row. The combined row is formed again (the same
// operations, so the same bits) by the pass that reduces it and by the pass
// that publishes, instead of being held in threadgroup memory.
#include "lib/core/activation.h"
#include "lib/core/norm.h"
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
    float at(uint i) const {
        const float from_dense = row_norm::apply<true, false>(
            dense[ulong(i) * dense_stride], dense_inverse,
            element::at<Norm>(dense_norm, ulong(i) * dense_norm_stride), 0.0f);
        const float from_routed = row_norm::apply<true, false>(
            routed[ulong(i) * routed_stride], routed_inverse,
            element::at<Norm>(routed_norm, ulong(i) * routed_norm_stride), 0.0f);
        return from_dense + from_routed;
    }
    float4 at4(uint i) const {
        const float4 dv = float4(*reinterpret_cast<device const packed_float4 *>(dense + i));
        const float4 rv = float4(*reinterpret_cast<device const packed_float4 *>(routed + i));
        const float4 dn = Norm::load4(reinterpret_cast<device const typename Norm::storage *>(dense_norm) + i);
        const float4 rn = Norm::load4(reinterpret_cast<device const typename Norm::storage *>(routed_norm) + i);
        const float4 from_dense = row_norm::apply4<true, false>(dv, dense_inverse, dn, float4(0.0f));
        const float4 from_routed = row_norm::apply4<true, false>(rv, routed_inverse, rn, float4(0.0f));
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
    SEISMIC_RESULT_0_PARAM(result_cells),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partials[8];
    const element::Out<ELEMENT_OF(SEISMIC_RESULT_0)> result =
        element::Out<ELEMENT_OF(SEISMIC_RESULT_0)>(result_cells).offset(ulong(row) * SEISMIC_RESULT_0_STRIDE_0);
    const uint width = uint(SEISMIC_DIM_D);
    const uint vec_width = width & ~3u;
    const float epsilon = as_type<float>(uint(SEISMIC_PARAM_EPSILON));
    const float scale = as_type<float>(uint(SEISMIC_PARAM_SCALE));
    device const float *d = dense + ulong(row) * SEISMIC_DENSE_STRIDE_0;
    device const float *r = routed + ulong(row) * SEISMIC_ROUTED_STRIDE_0;
    const bool unit_dr = SEISMIC_DENSE_STRIDE_1 == 1 && SEISMIC_ROUTED_STRIDE_1 == 1;
    float dense_squares = 0.0f, routed_squares = 0.0f;
    if (unit_dr) {
        float4 dsq4 = float4(0.0f), rsq4 = float4(0.0f);
        for (uint i = 4u * thread_index; i < vec_width; i += 4u * 256u) {
            const float4 dv = float4(*reinterpret_cast<device const packed_float4 *>(d + i));
            const float4 rv = float4(*reinterpret_cast<device const packed_float4 *>(r + i));
            dsq4 = row_norm::square4(dv, dsq4);
            rsq4 = row_norm::square4(rv, rsq4);
        }
        dense_squares = row_norm::sum4(dsq4);
        routed_squares = row_norm::sum4(rsq4);
        for (uint i = vec_width + thread_index; i < width; i += 256u) {
            dense_squares = row_norm::square(d[i], dense_squares);
            routed_squares = row_norm::square(r[i], routed_squares);
        }
    } else {
        for (uint i = thread_index; i < width; i += 256u) {
            const float dv = d[ulong(i) * SEISMIC_DENSE_STRIDE_1];
            const float rv = r[ulong(i) * SEISMIC_ROUTED_STRIDE_1];
            dense_squares = row_norm::square(dv, dense_squares);
            routed_squares = row_norm::square(rv, routed_squares);
        }
    }
    const float dense_total = reduce::group_sum<8>(dense_squares, partials, sg, lane);
    const float routed_total = reduce::group_sum<8>(routed_squares, partials, sg, lane);
    const Combined combined{d, SEISMIC_DENSE_STRIDE_1, r, SEISMIC_ROUTED_STRIDE_1, dense_norm,
        SEISMIC_DENSE_NORM_STRIDE_0, routed_norm, SEISMIC_ROUTED_NORM_STRIDE_0,
        row_norm::inverse(dense_total, width, epsilon), row_norm::inverse(routed_total, width, epsilon)};
    const bool vec_combined = unit_dr && SEISMIC_DENSE_NORM_STRIDE_0 == 1 && SEISMIC_ROUTED_NORM_STRIDE_0 == 1;
    float squares = 0.0f;
    if (vec_combined) {
        float4 sq4 = float4(0.0f);
        for (uint i = 4u * thread_index; i < vec_width; i += 4u * 256u)
            sq4 = row_norm::square4(combined.at4(i), sq4);
        squares = row_norm::sum4(sq4);
        for (uint i = vec_width + thread_index; i < width; i += 256u)
            squares = row_norm::square(combined.at(i), squares);
    } else {
        for (uint i = thread_index; i < width; i += 256u)
            squares = row_norm::square(combined.at(i), squares);
    }
    const float inverse = row_norm::inverse(reduce::group_sum<8>(squares, partials, sg, lane), width, epsilon);
    device const float *base = residual + ulong(row) * SEISMIC_RESIDUAL_STRIDE_0;
    const bool vec_publish = vec_combined && SEISMIC_NORM_STRIDE_0 == 1 && SEISMIC_RESIDUAL_STRIDE_1 == 1
        && SEISMIC_RESULT_0_STRIDE_1 == 1;
    if (vec_publish) {
        device const typename Norm::storage *norm_vals = reinterpret_cast<device const typename Norm::storage *>(norm);
        for (uint i = 4u * thread_index; i < vec_width; i += 4u * 256u) {
            const float4 nv = Norm::load4(norm_vals + i);
            const float4 normalized = row_norm::apply4<true, false>(combined.at4(i), inverse, nv, float4(0.0f));
            const float4 bv = float4(*reinterpret_cast<device const packed_float4 *>(base + i));
            result.put4(i, (bv + normalized) * scale);
        }
        for (uint i = vec_width + thread_index; i < width; i += 256u) {
            const float normalized = row_norm::apply<true, false>(combined.at(i), inverse, element::at<Norm>(norm, i), 0.0f);
            result.put(i, (base[i] + normalized) * scale);
        }
    } else {
        for (uint i = thread_index; i < width; i += 256u) {
            const float normalized = row_norm::apply<true, false>(
                combined.at(i), inverse, element::at<Norm>(norm, ulong(i) * SEISMIC_NORM_STRIDE_0), 0.0f);
            result.put(ulong(i) * SEISMIC_RESULT_0_STRIDE_1,
                (base[ulong(i) * SEISMIC_RESIDUAL_STRIDE_1] + normalized) * scale);
        }
    }
}
