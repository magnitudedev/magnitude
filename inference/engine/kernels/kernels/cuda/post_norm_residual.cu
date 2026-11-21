// post_norm_residual: a sandwich-norm tail over the `out_rows` rows,
// (residual + rms(projected) * norm) * scale, all F32. One block per output
// row: the square sum of the projected row, then the update.
#include "lib/core/activation.cuh"
#include "lib/core/reduce.cuh"

using element::u64;
using element::u8;

extern "C" __global__ void post_norm_residual(SEISMIC_KERNEL_PARAMS) {
    using Norm = ELEMENT_OF(SEISMIC_NORM);
    using ResultElem = ELEMENT_OF(SEISMIC_RESULT_0);
    __shared__ float partials[32];
    const u64 row = blockIdx.x;
    const u64 width = SEISMIC_DIM_D;
    const u64 width4 = width & ~3ull;
    const float epsilon = __uint_as_float((unsigned)SEISMIC_PARAM_EPSILON);
    const float scale = __uint_as_float((unsigned)SEISMIC_PARAM_SCALE);
    const float *p = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_PROJECTED)) + row * SEISMIC_PROJECTED_STRIDE_0;
    const bool p_vec = SEISMIC_PROJECTED_STRIDE_1 == 1ull && (reinterpret_cast<u64>(p) & 15ull) == 0ull;
    float squares = 0.0f;
    if (p_vec) {
        const u64 step4 = static_cast<u64>(blockDim.x) * 4ull;
        for (u64 i = static_cast<u64>(threadIdx.x) * 4ull; i < width4; i += step4) {
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
        const u64 step4 = static_cast<u64>(blockDim.x) * 4ull;
        for (u64 i = static_cast<u64>(threadIdx.x) * 4ull; i < width4; i += step4) {
            const float4 pv = *reinterpret_cast<const float4 *>(p + i);
            const float4 rv = *reinterpret_cast<const float4 *>(r + i);
            float nv[4];
            element::span<Norm, 4, false>(norm, i, nv);
            float out[4];
            out[0] = (rv.x + pv.x * inverse * nv[0]) * scale;
            out[1] = (rv.y + pv.y * inverse * nv[1]) * scale;
            out[2] = (rv.z + pv.z * inverse * nv[2]) * scale;
            out[3] = (rv.w + pv.w * inverse * nv[3]) * scale;
            result.put_span<4>(i, out);
        }
        for (u64 i = width4 + threadIdx.x; i < width; i += blockDim.x) {
            const float normalized = p[i] * inverse * element::at<Norm>(norm, i);
            result.put(i, (r[i] + normalized) * scale);
        }
    } else {
        for (u64 i = threadIdx.x; i < width; i += blockDim.x) {
            const float normalized = p[i * SEISMIC_PROJECTED_STRIDE_1] * inverse
                * element::at<Norm>(norm, i * SEISMIC_NORM_STRIDE_0);
            result.put(i * SEISMIC_RESULT_0_STRIDE_1, (r[i * SEISMIC_RESIDUAL_STRIDE_1] + normalized) * scale);
        }
    }
}
