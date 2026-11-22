// moe_tail: the tail of a feed-forward with a dense branch beside the routed
// experts, all F32:
//   f = rms(dense) * dense_norm + rms(routed) * routed_norm
//   result = (residual + rms(f) * norm) * scale
// One threadgroup per row. Vectorized rows keep up to MAX_CACHE float4 chunks
// per thread of `dense` and `routed` in registers across Pass 1 -> Pass 2
// (covering D <= 4096 in a single DRAM read), then overwrite `d_cache` (and
// reuse `r_cache` for 4096 < D <= 8192) with `combined` across Pass 2 ->
// Pass 3; columns beyond 2 * cache_span recompute `combined` directly.
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
    float combine(float dv, float rv, uint i) const {
        const float from_dense =
            row_norm::apply<true, false>(dv, dense_inverse, element::at<Norm>(dense_norm, i), 0.0f);
        const float from_routed =
            row_norm::apply<true, false>(rv, routed_inverse, element::at<Norm>(routed_norm, i), 0.0f);
        return from_dense + from_routed;
    }
    float4 combine4(float4 dv, float4 rv, uint i) const {
        const float4 dn = Norm::load4(reinterpret_cast<device const typename Norm::storage *>(dense_norm) + i);
        const float4 rn = Norm::load4(reinterpret_cast<device const typename Norm::storage *>(routed_norm) + i);
        const float4 from_dense = row_norm::apply4<true, false>(dv, dense_inverse, dn, float4(0.0f));
        const float4 from_routed = row_norm::apply4<true, false>(rv, routed_inverse, rn, float4(0.0f));
        return from_dense + from_routed;
    }
    float4 at4(uint i) const {
        return combine4(float4(*reinterpret_cast<device const packed_float4 *>(dense + i)),
            float4(*reinterpret_cast<device const packed_float4 *>(routed + i)), i);
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
    threadgroup float partials[16];
    constexpr uint MAX_CACHE = 4u;
    constexpr uint cache_span = 4u * MAX_CACHE * 256u;
    const element::Out<ELEMENT_OF(SEISMIC_RESULT_0)> result =
        element::Out<ELEMENT_OF(SEISMIC_RESULT_0)>(result_cells).offset(ulong(row) * SEISMIC_RESULT_0_STRIDE_0);
    const uint width = uint(SEISMIC_DIM_D);
    const uint vec_width = width & ~3u;
    const float epsilon = as_type<float>(uint(SEISMIC_PARAM_EPSILON));
    const float scale = as_type<float>(uint(SEISMIC_PARAM_SCALE));
    device const float *d = dense + ulong(row) * SEISMIC_DENSE_STRIDE_0;
    device const float *r = routed + ulong(row) * SEISMIC_ROUTED_STRIDE_0;
    const bool unit_dr = SEISMIC_DENSE_STRIDE_1 == 1 && SEISMIC_ROUTED_STRIDE_1 == 1;
    float4 d_cache[MAX_CACHE];
    float4 r_cache[MAX_CACHE];
    float d_tail = 0.0f, r_tail = 0.0f;
    float dense_squares = 0.0f, routed_squares = 0.0f;
    if (unit_dr) {
        float4 dsq4 = float4(0.0f), rsq4 = float4(0.0f);
        _Pragma("clang loop unroll(full)")
        for (uint k = 0; k < MAX_CACHE; ++k) {
            const uint i = 4u * (thread_index + k * 256u);
            if (i < vec_width) {
                const float4 dv = float4(*reinterpret_cast<device const packed_float4 *>(d + i));
                const float4 rv = float4(*reinterpret_cast<device const packed_float4 *>(r + i));
                d_cache[k] = dv;
                r_cache[k] = rv;
                dsq4 = row_norm::square4(dv, dsq4);
                rsq4 = row_norm::square4(rv, rsq4);
            }
        }
        for (uint i = cache_span + 4u * thread_index; i < vec_width; i += 4u * 256u) {
            const float4 dv = float4(*reinterpret_cast<device const packed_float4 *>(d + i));
            const float4 rv = float4(*reinterpret_cast<device const packed_float4 *>(r + i));
            dsq4 = row_norm::square4(dv, dsq4);
            rsq4 = row_norm::square4(rv, rsq4);
        }
        dense_squares = row_norm::sum4(dsq4);
        routed_squares = row_norm::sum4(rsq4);
        if (vec_width + thread_index < width) {
            d_tail = d[vec_width + thread_index];
            r_tail = r[vec_width + thread_index];
            dense_squares = row_norm::square(d_tail, dense_squares);
            routed_squares = row_norm::square(r_tail, routed_squares);
        }
    } else {
        for (uint i = thread_index; i < width; i += 256u) {
            const float dv = d[ulong(i) * SEISMIC_DENSE_STRIDE_1];
            const float rv = r[ulong(i) * SEISMIC_ROUTED_STRIDE_1];
            dense_squares = row_norm::square(dv, dense_squares);
            routed_squares = row_norm::square(rv, routed_squares);
        }
    }
    float branch_squares[2] = {dense_squares, routed_squares};
    reduce::group_sum_n<2, 8>(branch_squares, partials, sg, lane);
    const Combined combined{d, SEISMIC_DENSE_STRIDE_1, r, SEISMIC_ROUTED_STRIDE_1, dense_norm,
        SEISMIC_DENSE_NORM_STRIDE_0, routed_norm, SEISMIC_ROUTED_NORM_STRIDE_0,
        row_norm::inverse(branch_squares[0], width, epsilon), row_norm::inverse(branch_squares[1], width, epsilon)};
    const bool vec_combined = unit_dr && SEISMIC_DENSE_NORM_STRIDE_0 == 1 && SEISMIC_ROUTED_NORM_STRIDE_0 == 1;
    float squares = 0.0f;
    if (vec_combined) {
        float4 sq4 = float4(0.0f);
        _Pragma("clang loop unroll(full)")
        for (uint k = 0; k < MAX_CACHE; ++k) {
            const uint i = 4u * (thread_index + k * 256u);
            if (i < vec_width) {
                const float4 f = combined.combine4(d_cache[k], r_cache[k], i);
                d_cache[k] = f;
                sq4 = row_norm::square4(f, sq4);
            }
        }
        if (vec_width > cache_span) {
            _Pragma("clang loop unroll(full)")
            for (uint k = 0; k < MAX_CACHE; ++k) {
                const uint i = cache_span + 4u * (thread_index + k * 256u);
                if (i < vec_width) {
                    const float4 f = combined.at4(i);
                    r_cache[k] = f;
                    sq4 = row_norm::square4(f, sq4);
                }
            }
            for (uint i = 2u * cache_span + 4u * thread_index; i < vec_width; i += 4u * 256u)
                sq4 = row_norm::square4(combined.at4(i), sq4);
        }
        squares = row_norm::sum4(sq4);
        if (vec_width + thread_index < width) {
            d_tail = combined.combine(d_tail, r_tail, vec_width + thread_index);
            squares = row_norm::square(d_tail, squares);
        }
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
        auto publish4 = [&](uint i, float4 f) {
            const float4 nv = Norm::load4(norm_vals + i);
            const float4 normalized = row_norm::apply4<true, false>(f, inverse, nv, float4(0.0f));
            const float4 bv = float4(*reinterpret_cast<device const packed_float4 *>(base + i));
            result.put4(i, (bv + normalized) * scale);
        };
        _Pragma("clang loop unroll(full)")
        for (uint k = 0; k < MAX_CACHE; ++k) {
            const uint i = 4u * (thread_index + k * 256u);
            if (i < vec_width)
                publish4(i, d_cache[k]);
        }
        if (vec_width > cache_span) {
            _Pragma("clang loop unroll(full)")
            for (uint k = 0; k < MAX_CACHE; ++k) {
                const uint i = cache_span + 4u * (thread_index + k * 256u);
                if (i < vec_width)
                    publish4(i, r_cache[k]);
            }
            for (uint i = 2u * cache_span + 4u * thread_index; i < vec_width; i += 4u * 256u)
                publish4(i, combined.at4(i));
        }
        if (vec_width + thread_index < width) {
            const float normalized =
                row_norm::apply<true, false>(d_tail, inverse, element::at<Norm>(norm, vec_width + thread_index), 0.0f);
            result.put(vec_width + thread_index, (base[vec_width + thread_index] + normalized) * scale);
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
