// rel_bias_project: `[M, H, D] A @ [E, D]^T W -> [M, H, E] f32`, with optional
// fused `head_l2_clip` when `threshold > 0.0` (rounding clipped R to `A` before
// the projection to preserve the exact intermediate dtype boundary).
// One 32-thread warp per (head, row, 32-extent tile).
#include "lib/core/activation.cuh"

using element::u64;
using element::u8;

extern "C" __global__ void rel_bias_project(SEISMIC_KERNEL_PARAMS) {
    using RElem = ELEMENT_OF(SEISMIC_R);
    using WElem = ELEMENT_OF(SEISMIC_WEIGHT);
    __shared__ float r_sh[64];

    const u64 head = blockIdx.x;
    const u64 ext = (u64)blockIdx.z * 32ull + (u64)threadIdx.x;
    const u64 D = SEISMIC_DIM_D;
    const u64 E = SEISMIC_DIM_E;
    const bool has_scale = SEISMIC_DIM_SC > 0;
    const u64 scale_base = (SEISMIC_DIM_SC > 1 ? head : 0ull) * SEISMIC_SCALE_STRIDE_0;
    const float threshold = __uint_as_float((unsigned)SEISMIC_PARAM_THRESHOLD);
    const float eps = __uint_as_float((unsigned)SEISMIC_PARAM_EPSILON);
    const float log_floor = __uint_as_float((unsigned)SEISMIC_PARAM_LOG_FLOOR);
    const float log_alpha = __uint_as_float((unsigned)SEISMIC_PARAM_LOG_ALPHA);
    const bool do_clip = threshold > 0.0f;

    const u8 *r = SEISMIC_PTR(SEISMIC_BUFFER_R);
    const u8 *weight = SEISMIC_PTR(SEISMIC_BUFFER_WEIGHT);
    const float *scale = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCALE));
    const int *coordinates = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_COORDINATES));
    const u64 w_base = ext * SEISMIC_WEIGHT_STRIDE_0;

    for (u64 row = blockIdx.y; row < SEISMIC_DIM_M; row += gridDim.y) {
        const u64 r_base = row * SEISMIC_R_STRIDE_0 + head * SEISMIC_R_STRIDE_1;
        float factor = 1.0f;
        if (do_clip) {
            float tau = 1.0f;
            if (log_alpha != 0.0f) {
                const float pos = (float)coordinates[row * SEISMIC_COORDINATES_STRIDE_0];
                const float ratio = (pos + 1.0f) / log_floor;
                tau = 1.0f + log_alpha * logf(fmaxf(1.0f, ratio));
            }
            float sq = 0.0f;
            for (u64 d = threadIdx.x; d < D; d += 32ull) {
                float v = element::at<RElem>(r, r_base + d * SEISMIC_R_STRIDE_2);
                if (has_scale)
                    v *= scale[scale_base + d * SEISMIC_SCALE_STRIDE_1];
                sq = seismic_fma_rn(v, v, sq);
            }
            const float total_sq = seismic_warp_sum_f32(sq);
            const float clip = fminf(1.0f, threshold / (sqrtf(total_sq) + eps));
            factor = clip * tau;
        }

        if (D <= 64ull) {
            for (u64 d = threadIdx.x; d < D; d += 32ull) {
                float v = element::at<RElem>(r, r_base + d * SEISMIC_R_STRIDE_2);
                if (do_clip) {
                    if (has_scale)
                        v *= scale[scale_base + d * SEISMIC_SCALE_STRIDE_1];
                    v = RElem::round(v * factor);
                }
                r_sh[d] = v;
            }
            __syncthreads();
            if (ext < E) {
                float acc = 0.0f;
                for (u64 d = 0; d < D; ++d) {
                    const float wv = element::at<WElem>(weight, w_base + d * SEISMIC_WEIGHT_STRIDE_1);
                    acc = seismic_fma_rn(r_sh[d], wv, acc);
                }
                const auto result = SEISMIC_RESULT_0_OUT.offset(
                    row * SEISMIC_RESULT_0_STRIDE_0 + head * SEISMIC_RESULT_0_STRIDE_1);
                result.put(ext * SEISMIC_RESULT_0_STRIDE_2, acc);
            }
            __syncthreads();
        } else if (ext < E) {
            float acc = 0.0f;
            for (u64 d = 0; d < D; ++d) {
                float rv = element::at<RElem>(r, r_base + d * SEISMIC_R_STRIDE_2);
                if (do_clip) {
                    if (has_scale)
                        rv *= scale[scale_base + d * SEISMIC_SCALE_STRIDE_1];
                    rv = RElem::round(rv * factor);
                }
                const float wv = element::at<WElem>(weight, w_base + d * SEISMIC_WEIGHT_STRIDE_1);
                acc = seismic_fma_rn(rv, wv, acc);
            }
            const auto result = SEISMIC_RESULT_0_OUT.offset(
                row * SEISMIC_RESULT_0_STRIDE_0 + head * SEISMIC_RESULT_0_STRIDE_1);
            result.put(ext * SEISMIC_RESULT_0_STRIDE_2, acc);
        }
    }
}
