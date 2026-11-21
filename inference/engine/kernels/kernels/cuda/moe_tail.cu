// moe_tail: the tail of a feed-forward with a dense branch beside the routed
// experts, all F32:
//   f = rms(dense) * dense_norm + rms(routed) * routed_norm
//   result = (residual + rms(f) * norm) * scale
// One block per row. The combined row is formed again (the same operations,
// so the same bits) by the pass that reduces it and by the pass that
// publishes, instead of being held in shared memory.
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
    __device__ __forceinline__ float4 at4(u32 i) const {
        const float4 dv = *reinterpret_cast<const float4 *>(dense + i);
        const float4 rv = *reinterpret_cast<const float4 *>(routed + i);
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
};

extern "C" __global__ void moe_tail(SEISMIC_KERNEL_PARAMS) {
    __shared__ float partials[32];
    const u64 row = blockIdx.x;
    const u32 width = static_cast<u32>(SEISMIC_DIM_D);
    const u32 vec_width = width & ~3u;
    const u32 tid = threadIdx.x;
    const u32 threads = blockDim.x;
    const float epsilon = __uint_as_float(static_cast<u32>(SEISMIC_PARAM_EPSILON));
    const float scale = __uint_as_float(static_cast<u32>(SEISMIC_PARAM_SCALE));
    const float *d = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_DENSE)) + row * SEISMIC_DENSE_STRIDE_0;
    const float *r = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTED)) + row * SEISMIC_ROUTED_STRIDE_0;
    const bool unit_dr = SEISMIC_DENSE_STRIDE_1 == 1 && SEISMIC_ROUTED_STRIDE_1 == 1
                         && ((reinterpret_cast<u64>(d) | reinterpret_cast<u64>(r)) & 15u) == 0u;
    float dense_squares = 0.0f, routed_squares = 0.0f;
    if (unit_dr) {
        float4 dsq4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 rsq4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (u32 i = 4u * tid; i < vec_width; i += 4u * threads) {
            const float4 dv = *reinterpret_cast<const float4 *>(d + i);
            const float4 rv = *reinterpret_cast<const float4 *>(r + i);
            dsq4 = row_norm::square4(dv, dsq4);
            rsq4 = row_norm::square4(rv, rsq4);
        }
        dense_squares = row_norm::sum4(dsq4);
        routed_squares = row_norm::sum4(rsq4);
        for (u32 i = vec_width + tid; i < width; i += threads) {
            dense_squares = row_norm::square(d[i], dense_squares);
            routed_squares = row_norm::square(r[i], routed_squares);
        }
    } else {
        for (u32 i = tid; i < width; i += threads) {
            const float dv = d[static_cast<u64>(i) * SEISMIC_DENSE_STRIDE_1];
            const float rv = r[static_cast<u64>(i) * SEISMIC_ROUTED_STRIDE_1];
            dense_squares = row_norm::square(dv, dense_squares);
            routed_squares = row_norm::square(rv, routed_squares);
        }
    }
    const float dense_total = reduce::group_sum(dense_squares, partials);
    const float routed_total = reduce::group_sum(routed_squares, partials);
    const u8 *dense_norm = SEISMIC_PTR(SEISMIC_BUFFER_DENSE_NORM);
    const u8 *routed_norm = SEISMIC_PTR(SEISMIC_BUFFER_ROUTED_NORM);
    const Combined combined{d, SEISMIC_DENSE_STRIDE_1, r, SEISMIC_ROUTED_STRIDE_1, dense_norm,
                            SEISMIC_DENSE_NORM_STRIDE_0, routed_norm, SEISMIC_ROUTED_NORM_STRIDE_0,
                            row_norm::inverse(dense_total, width, epsilon),
                            row_norm::inverse(routed_total, width, epsilon)};
    constexpr u32 norm_align = 4u * Norm::bytes - 1u;
    const bool vec_combined = unit_dr && SEISMIC_DENSE_NORM_STRIDE_0 == 1 && SEISMIC_ROUTED_NORM_STRIDE_0 == 1
                              && ((reinterpret_cast<u64>(dense_norm) | reinterpret_cast<u64>(routed_norm)) & norm_align) == 0u;
    float squares = 0.0f;
    if (vec_combined) {
        float4 sq4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (u32 i = 4u * tid; i < vec_width; i += 4u * threads)
            sq4 = row_norm::square4(combined.at4(i), sq4);
        squares = row_norm::sum4(sq4);
        for (u32 i = vec_width + tid; i < width; i += threads)
            squares = row_norm::square(combined.at(i), squares);
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
        for (u32 i = 4u * tid; i < vec_width; i += 4u * threads) {
            float nv[4];
            element::span<Norm, 4, false>(norm, i, nv);
            const float4 normalized =
                row_norm::apply4<true, false>(combined.at4(i), inverse, make_float4(nv[0], nv[1], nv[2], nv[3]), zero);
            const float4 bv = *reinterpret_cast<const float4 *>(base + i);
            const float out[4] = {(bv.x + normalized.x) * scale, (bv.y + normalized.y) * scale,
                                  (bv.z + normalized.z) * scale, (bv.w + normalized.w) * scale};
            result.put_span<4>(i, out);
        }
        for (u32 i = vec_width + tid; i < width; i += threads) {
            const float normalized = row_norm::apply<true, false>(combined.at(i), inverse, element::at<Norm>(norm, i), 0.0f);
            result.put(i, (base[i] + normalized) * scale);
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
