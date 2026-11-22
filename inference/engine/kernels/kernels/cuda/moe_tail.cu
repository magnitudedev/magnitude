// moe_tail: the tail of a feed-forward with a dense branch beside the routed
// experts, all F32:
//   f = rms(dense) * dense_norm + rms(routed) * routed_norm
//   result = (residual + rms(f) * norm) * scale
// One block per row. Vectorized rows keep up to MAX_CACHE float4 chunks per
// thread of `dense` and `routed` in registers across Pass 1 -> Pass 2 (covering
// D <= 4096 in a single DRAM read), then overwrite `d_cache` (and reuse
// `r_cache` for 4096 < D <= 8192) with `combined` across Pass 2 -> Pass 3;
// columns beyond 2 * cache_span recompute `combined` directly.
#include "lib/core/activation.cuh"
#include "lib/core/norm.cuh"
#include "lib/core/reduce.cuh"

using element::u32;
using element::u64;
using element::u8;
using Norm = ELEMENT_OF(SEISMIC_NORM);

// f[i] of one row: both branches normalized with their inverses and norms.
struct Combined {
    const float *dense;
    u64 dense_stride;
    const float *routed;
    u64 routed_stride;
    const u8 *dense_norm;
    u64 dense_norm_stride;
    const u8 *routed_norm;
    u64 routed_norm_stride;
    float dense_inverse, routed_inverse;
    __device__ __forceinline__ float at(u32 i) const {
        const float from_dense = row_norm::apply<true, false>(
            dense[static_cast<u64>(i) * dense_stride], dense_inverse,
            element::at<Norm>(dense_norm, static_cast<u64>(i) * dense_norm_stride), 0.0f);
        const float from_routed = row_norm::apply<true, false>(
            routed[static_cast<u64>(i) * routed_stride], routed_inverse,
            element::at<Norm>(routed_norm, static_cast<u64>(i) * routed_norm_stride), 0.0f);
        return from_dense + from_routed;
    }
    __device__ __forceinline__ float combine(float dv, float rv, u32 i) const {
        const float from_dense =
            row_norm::apply<true, false>(dv, dense_inverse, element::at<Norm>(dense_norm, i), 0.0f);
        const float from_routed =
            row_norm::apply<true, false>(rv, routed_inverse, element::at<Norm>(routed_norm, i), 0.0f);
        return from_dense + from_routed;
    }
    __device__ __forceinline__ float4 combine4(float4 dv, float4 rv, u32 i) const {
        float dn[4], rn[4];
        element::span<Norm, 4, false>(dense_norm, i, dn);
        element::span<Norm, 4, false>(routed_norm, i, rn);
        const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        const float4 from_dense =
            row_norm::apply4<true, false>(dv, dense_inverse, make_float4(dn[0], dn[1], dn[2], dn[3]), zero);
        const float4 from_routed =
            row_norm::apply4<true, false>(rv, routed_inverse, make_float4(rn[0], rn[1], rn[2], rn[3]), zero);
        return make_float4(from_dense.x + from_routed.x, from_dense.y + from_routed.y,
                           from_dense.z + from_routed.z, from_dense.w + from_routed.w);
    }
    __device__ __forceinline__ float4 at4(u32 i) const {
        return combine4(*reinterpret_cast<const float4 *>(dense + i),
                        *reinterpret_cast<const float4 *>(routed + i), i);
    }
};

extern "C" __global__ void moe_tail(SEISMIC_KERNEL_PARAMS) {
    __shared__ float partials[64];
    constexpr u32 MAX_CACHE = 4u;
    const u64 row = blockIdx.x;
    const u32 width = static_cast<u32>(SEISMIC_DIM_D);
    const u32 vec_width = width & ~3u;
    const u32 tid = threadIdx.x;
    const u32 threads = blockDim.x;
    const u32 cache_span = 4u * MAX_CACHE * threads;
    const float epsilon = __uint_as_float(static_cast<u32>(SEISMIC_PARAM_EPSILON));
    const float scale = __uint_as_float(static_cast<u32>(SEISMIC_PARAM_SCALE));
    const float *d = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_DENSE)) + row * SEISMIC_DENSE_STRIDE_0;
    const float *r = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTED)) + row * SEISMIC_ROUTED_STRIDE_0;
    const bool unit_dr = SEISMIC_DENSE_STRIDE_1 == 1 && SEISMIC_ROUTED_STRIDE_1 == 1
                         && ((reinterpret_cast<u64>(d) | reinterpret_cast<u64>(r)) & 15u) == 0u;
    float4 d_cache[MAX_CACHE];
    float4 r_cache[MAX_CACHE];
    float d_tail = 0.0f, r_tail = 0.0f;
    float dense_squares = 0.0f, routed_squares = 0.0f;
    if (unit_dr) {
        float4 dsq4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 rsq4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
        for (u32 k = 0; k < MAX_CACHE; ++k) {
            const u32 i = 4u * (tid + k * threads);
            if (i < vec_width) {
                const float4 dv = *reinterpret_cast<const float4 *>(d + i);
                const float4 rv = *reinterpret_cast<const float4 *>(r + i);
                d_cache[k] = dv;
                r_cache[k] = rv;
                dsq4 = row_norm::square4(dv, dsq4);
                rsq4 = row_norm::square4(rv, rsq4);
            }
        }
        for (u32 i = cache_span + 4u * tid; i < vec_width; i += 4u * threads) {
            const float4 dv = *reinterpret_cast<const float4 *>(d + i);
            const float4 rv = *reinterpret_cast<const float4 *>(r + i);
            dsq4 = row_norm::square4(dv, dsq4);
            rsq4 = row_norm::square4(rv, rsq4);
        }
        dense_squares = row_norm::sum4(dsq4);
        routed_squares = row_norm::sum4(rsq4);
        if (vec_width + tid < width) {
            d_tail = d[vec_width + tid];
            r_tail = r[vec_width + tid];
            dense_squares = row_norm::square(d_tail, dense_squares);
            routed_squares = row_norm::square(r_tail, routed_squares);
        }
    } else {
        for (u32 i = tid; i < width; i += threads) {
            const float dv = d[static_cast<u64>(i) * SEISMIC_DENSE_STRIDE_1];
            const float rv = r[static_cast<u64>(i) * SEISMIC_ROUTED_STRIDE_1];
            dense_squares = row_norm::square(dv, dense_squares);
            routed_squares = row_norm::square(rv, routed_squares);
        }
    }
    float branch_squares[2] = {dense_squares, routed_squares};
    reduce::group_sum_n<2>(branch_squares, partials);
    const u8 *dense_norm = SEISMIC_PTR(SEISMIC_BUFFER_DENSE_NORM);
    const u8 *routed_norm = SEISMIC_PTR(SEISMIC_BUFFER_ROUTED_NORM);
    const Combined combined{d, SEISMIC_DENSE_STRIDE_1, r, SEISMIC_ROUTED_STRIDE_1, dense_norm,
                            SEISMIC_DENSE_NORM_STRIDE_0, routed_norm, SEISMIC_ROUTED_NORM_STRIDE_0,
                            row_norm::inverse(branch_squares[0], width, epsilon),
                            row_norm::inverse(branch_squares[1], width, epsilon)};
    constexpr u32 norm_align = 4u * Norm::bytes - 1u;
    const bool vec_combined = unit_dr && SEISMIC_DENSE_NORM_STRIDE_0 == 1 && SEISMIC_ROUTED_NORM_STRIDE_0 == 1
                              && ((reinterpret_cast<u64>(dense_norm) | reinterpret_cast<u64>(routed_norm)) & norm_align) == 0u;
    float squares = 0.0f;
    if (vec_combined) {
        float4 sq4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
        for (u32 k = 0; k < MAX_CACHE; ++k) {
            const u32 i = 4u * (tid + k * threads);
            if (i < vec_width) {
                const float4 f = combined.combine4(d_cache[k], r_cache[k], i);
                d_cache[k] = f;
                sq4 = row_norm::square4(f, sq4);
            }
        }
        if (vec_width > cache_span) {
#pragma unroll
            for (u32 k = 0; k < MAX_CACHE; ++k) {
                const u32 i = cache_span + 4u * (tid + k * threads);
                if (i < vec_width) {
                    const float4 f = combined.at4(i);
                    r_cache[k] = f;
                    sq4 = row_norm::square4(f, sq4);
                }
            }
            for (u32 i = 2u * cache_span + 4u * tid; i < vec_width; i += 4u * threads)
                sq4 = row_norm::square4(combined.at4(i), sq4);
        }
        squares = row_norm::sum4(sq4);
        if (vec_width + tid < width) {
            d_tail = combined.combine(d_tail, r_tail, vec_width + tid);
            squares = row_norm::square(d_tail, squares);
        }
    } else {
        for (u32 i = tid; i < width; i += threads)
            squares = row_norm::square(combined.at(i), squares);
    }
    const float inverse = row_norm::inverse(reduce::group_sum(squares, partials), width, epsilon);
    const u8 *norm = SEISMIC_PTR(SEISMIC_BUFFER_NORM);
    const float *base =
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL)) + row * SEISMIC_RESIDUAL_STRIDE_0;
    const element::Out<element::F32> result = SEISMIC_RESULT_0_OUT.offset(row * SEISMIC_RESULT_0_STRIDE_0);
    const bool vec_publish = vec_combined && SEISMIC_NORM_STRIDE_0 == 1 && SEISMIC_RESIDUAL_STRIDE_1 == 1
                             && SEISMIC_RESULT_0_STRIDE_1 == 1
                             && ((reinterpret_cast<u64>(norm) & norm_align) == 0u)
                             && (((reinterpret_cast<u64>(base) | reinterpret_cast<u64>(result.stored(0))) & 15u) == 0u);
    if (vec_publish) {
        const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        auto publish4 = [&](u32 i, float4 f) {
            float nv[4];
            element::span<Norm, 4, false>(norm, i, nv);
            const float4 normalized =
                row_norm::apply4<true, false>(f, inverse, make_float4(nv[0], nv[1], nv[2], nv[3]), zero);
            const float4 bv = *reinterpret_cast<const float4 *>(base + i);
            const float out[4] = {(bv.x + normalized.x) * scale, (bv.y + normalized.y) * scale,
                                  (bv.z + normalized.z) * scale, (bv.w + normalized.w) * scale};
            result.put_span<4>(i, out);
        };
#pragma unroll
        for (u32 k = 0; k < MAX_CACHE; ++k) {
            const u32 i = 4u * (tid + k * threads);
            if (i < vec_width)
                publish4(i, d_cache[k]);
        }
        if (vec_width > cache_span) {
#pragma unroll
            for (u32 k = 0; k < MAX_CACHE; ++k) {
                const u32 i = cache_span + 4u * (tid + k * threads);
                if (i < vec_width)
                    publish4(i, r_cache[k]);
            }
            for (u32 i = 2u * cache_span + 4u * tid; i < vec_width; i += 4u * threads)
                publish4(i, combined.at4(i));
        }
        if (vec_width + tid < width) {
            const float normalized =
                row_norm::apply<true, false>(d_tail, inverse, element::at<Norm>(norm, vec_width + tid), 0.0f);
            result.put(vec_width + tid, (base[vec_width + tid] + normalized) * scale);
        }
    } else {
        for (u32 i = tid; i < width; i += threads) {
            const float normalized = row_norm::apply<true, false>(
                combined.at(i), inverse, element::at<Norm>(norm, static_cast<u64>(i) * SEISMIC_NORM_STRIDE_0), 0.0f);
            result.put(static_cast<u64>(i) * SEISMIC_RESULT_0_STRIDE_1,
                       (base[static_cast<u64>(i) * SEISMIC_RESIDUAL_STRIDE_1] + normalized) * scale);
        }
    }
}
