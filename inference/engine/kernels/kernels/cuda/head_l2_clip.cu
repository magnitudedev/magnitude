// head_l2_clip: per-head optional channel scale, L2 norm clip, and optional
// position-dependent log-scaling `tau`. One 64-thread block per (head, row).
#include "lib/core/activation.cuh"
#include "lib/core/reduce.cuh"

using element::u64;
using element::u8;

extern "C" __global__ void head_l2_clip(SEISMIC_KERNEL_PARAMS) {
    using InElem = ELEMENT_OF(SEISMIC_X);
    __shared__ float partials[32];
    const u64 head = blockIdx.x;
    const u64 W = SEISMIC_DIM_W;
    const bool has_scale = SEISMIC_DIM_SC > 0;
    const u64 scale_base = (SEISMIC_DIM_SC > 1 ? head : 0ull) * SEISMIC_SCALE_STRIDE_0;
    const float threshold = __uint_as_float((unsigned)SEISMIC_PARAM_THRESHOLD);
    const float eps = __uint_as_float((unsigned)SEISMIC_PARAM_EPSILON);
    const float log_floor = __uint_as_float((unsigned)SEISMIC_PARAM_LOG_FLOOR);
    const float log_alpha = __uint_as_float((unsigned)SEISMIC_PARAM_LOG_ALPHA);

    const u8 *x = SEISMIC_PTR(SEISMIC_BUFFER_X);
    const float *scale = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCALE));
    const int *coordinates = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_COORDINATES));

    for (u64 row = blockIdx.y; row < SEISMIC_DIM_M; row += gridDim.y) {
        float tau = 1.0f;
        if (log_alpha != 0.0f) {
            const float pos = (float)coordinates[row * SEISMIC_COORDINATES_STRIDE_0];
            const float ratio = (pos + 1.0f) / log_floor;
            tau = 1.0f + log_alpha * logf(fmaxf(1.0f, ratio));
        }

        const u64 x_base = row * SEISMIC_X_STRIDE_0 + head * SEISMIC_X_STRIDE_1;
        float sq = 0.0f;
        for (u64 c = threadIdx.x; c < W; c += blockDim.x) {
            float v = element::at<InElem>(x, x_base + c * SEISMIC_X_STRIDE_2);
            if (has_scale)
                v *= scale[scale_base + c * SEISMIC_SCALE_STRIDE_1];
            sq = seismic_fma_rn(v, v, sq);
        }
        const float total_sq = reduce::group_sum(sq, partials);
        const float clip = fminf(1.0f, threshold / (sqrtf(total_sq) + eps));
        const float factor = clip * tau;

        const auto result = SEISMIC_RESULT_0_OUT.offset(
            row * SEISMIC_RESULT_0_STRIDE_0 + head * SEISMIC_RESULT_0_STRIDE_1);
        for (u64 c = threadIdx.x; c < W; c += blockDim.x) {
            float v = element::at<InElem>(x, x_base + c * SEISMIC_X_STRIDE_2);
            if (has_scale)
                v *= scale[scale_base + c * SEISMIC_SCALE_STRIDE_1];
            result.put(c * SEISMIC_RESULT_0_STRIDE_2, v * factor);
        }
    }
}
