// post_norm_residual: a sandwich-norm tail over the `out_rows` rows,
// (residual + rms(projected) * norm) * scale, all F32. One block per output
// row: the square sum of the projected row, then the update. For aligned
// unit-stride rows up to 8 float4 slots/thread (width <= 8192 at 256 threads),
// Pass 1 caches `projected` in registers so Pass 2 avoids re-reading it.
#include "lib/core/activation.cuh"
#include "lib/core/reduce.cuh"

using element::u64;
using element::u8;

namespace {
constexpr int CACHE_SLOTS = 8;

template <typename Norm, typename ResultOut>
__device__ __forceinline__ void store4(
    const float4 pv, const float *r, const u8 *norm, const ResultOut &result,
    const u64 i, const float inverse, const float scale) {
    const float4 rv = *reinterpret_cast<const float4 *>(r + i);
    float nv[4];
    element::span<Norm, 4, false>(norm, i, nv);
    float out[4];
    out[0] = (rv.x + pv.x * inverse * nv[0]) * scale;
    out[1] = (rv.y + pv.y * inverse * nv[1]) * scale;
    out[2] = (rv.z + pv.z * inverse * nv[2]) * scale;
    out[3] = (rv.w + pv.w * inverse * nv[3]) * scale;
    result.template put_span<4>(i, out);
}
} // namespace

extern "C" __global__ void post_norm_residual(SEISMIC_KERNEL_PARAMS) {
    using Norm = ELEMENT_OF(SEISMIC_NORM);
    using ResultElem = ELEMENT_OF(SEISMIC_RESULT_0);
    __shared__ float partials[32];
    const u64 row = blockIdx.x;
    const u64 width = SEISMIC_DIM_D;
    const u64 width4 = width & ~3ull;
    const u64 step4 = static_cast<u64>(blockDim.x) * 4ull;
    const u64 base4 = static_cast<u64>(threadIdx.x) * 4ull;
    const float epsilon = __uint_as_float((unsigned)SEISMIC_PARAM_EPSILON);
    const float scale = __uint_as_float((unsigned)SEISMIC_PARAM_SCALE);
    const float *p = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_PROJECTED)) + row * SEISMIC_PROJECTED_STRIDE_0;
    const bool p_vec = SEISMIC_PROJECTED_STRIDE_1 == 1ull && (reinterpret_cast<u64>(p) & 15ull) == 0ull;
    const bool p_cache = p_vec && width <= step4 * static_cast<u64>(CACHE_SLOTS);
    float4 p_buf[CACHE_SLOTS];
    float p_tail = 0.0f;
    float squares = 0.0f;
    if (p_cache) {
        #pragma unroll
        for (int k = 0; k < CACHE_SLOTS; ++k) {
            const u64 i = base4 + static_cast<u64>(k) * step4;
            if (i < width4) {
                const float4 v = *reinterpret_cast<const float4 *>(p + i);
                p_buf[k] = v;
                squares = seismic_fma_rn(v.x, v.x, squares);
                squares = seismic_fma_rn(v.y, v.y, squares);
                squares = seismic_fma_rn(v.z, v.z, squares);
                squares = seismic_fma_rn(v.w, v.w, squares);
            }
        }
        const u64 tail_i = width4 + threadIdx.x;
        if (tail_i < width) {
            p_tail = p[tail_i];
            squares = seismic_fma_rn(p_tail, p_tail, squares);
        }
    } else if (p_vec) {
        for (u64 i = base4; i < width4; i += step4) {
            const float4 v = *reinterpret_cast<const float4 *>(p + i);
            squares = seismic_fma_rn(v.x, v.x, squares);
            squares = seismic_fma_rn(v.y, v.y, squares);
            squares = seismic_fma_rn(v.z, v.z, squares);
            squares = seismic_fma_rn(v.w, v.w, squares);
        }
        for (u64 i = width4 + threadIdx.x; i < width; i += blockDim.x) {
            const float v = p[i];
            squares = seismic_fma_rn(v, v, squares);
        }
    } else {
        for (u64 i = threadIdx.x; i < width; i += blockDim.x) {
            const float v = p[i * SEISMIC_PROJECTED_STRIDE_1];
            squares = seismic_fma_rn(v, v, squares);
        }
    }
    const float total = reduce::group_sum(squares, partials);
    const float inverse = rsqrtf(total / (float)width + epsilon);
    const int *out_rows = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_OUT_ROWS));
    const float *r = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL))
        + (u64)out_rows[row * SEISMIC_OUT_ROWS_STRIDE_0] * SEISMIC_RESIDUAL_STRIDE_0;
    const u8 *norm = SEISMIC_PTR(SEISMIC_BUFFER_NORM);
    const auto result = SEISMIC_RESULT_0_OUT.offset(row * SEISMIC_RESULT_0_STRIDE_0);
    const bool update_vec = p_vec &&
                            SEISMIC_RESIDUAL_STRIDE_1 == 1ull &&
                            (reinterpret_cast<u64>(r) & 15ull) == 0ull &&
                            SEISMIC_NORM_STRIDE_0 == 1ull &&
                            (reinterpret_cast<u64>(norm) & (4ull * Norm::bytes - 1ull)) == 0ull &&
                            SEISMIC_RESULT_0_STRIDE_1 == 1ull &&
                            (reinterpret_cast<u64>(result.stored(0)) & (4ull * ResultElem::bytes - 1ull)) == 0ull;
    if (update_vec) {
        if (p_cache) {
            #pragma unroll
            for (int k = 0; k < CACHE_SLOTS; ++k) {
                const u64 i = base4 + static_cast<u64>(k) * step4;
                if (i < width4) {
                    store4<Norm>(p_buf[k], r, norm, result, i, inverse, scale);
                }
            }
            const u64 tail_i = width4 + threadIdx.x;
            if (tail_i < width) {
                const float normalized = p_tail * inverse * element::at<Norm>(norm, tail_i);
                result.put(tail_i, (r[tail_i] + normalized) * scale);
            }
        } else {
            for (u64 i = base4; i < width4; i += step4) {
                const float4 pv = *reinterpret_cast<const float4 *>(p + i);
                store4<Norm>(pv, r, norm, result, i, inverse, scale);
            }
            for (u64 i = width4 + threadIdx.x; i < width; i += blockDim.x) {
                const float normalized = p[i] * inverse * element::at<Norm>(norm, i);
                result.put(i, (r[i] + normalized) * scale);
            }
        }
    } else {
        for (u64 i = threadIdx.x; i < width; i += blockDim.x) {
            const float normalized = p[i * SEISMIC_PROJECTED_STRIDE_1] * inverse
                * element::at<Norm>(norm, i * SEISMIC_NORM_STRIDE_0);
            result.put(i * SEISMIC_RESULT_0_STRIDE_1, (r[i * SEISMIC_RESIDUAL_STRIDE_1] + normalized) * scale);
        }
    }
}
