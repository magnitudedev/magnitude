// Prefill output of the general routed feed-forward; the CUDA form of
// `metal/routed_scatter.metal`: block (x, m) owns columns of row m, each
// thread one column. The grouped expert outputs of the row are unpermuted
// and weighted in slot order, then added to `base` and published in R.
#include "lib/core/activation.cuh"

using element::Act;
using element::u64;

extern "C" __global__ void routed_scatter(SEISMIC_KERNEL_PARAMS) {
    const u64 m = blockIdx.y;
    const u64 n = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= SEISMIC_DIM_H)
        return;
    const int *inverse = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_INVERSE));
    const float *weights = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_WEIGHTS));
    const element::u8 *expert_output = SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_OUTPUT);
    float selected = 0.0f;
    for (u64 k = 0; k < SEISMIC_DIM_K; ++k) {
        const u64 position = (u64)inverse[m * SEISMIC_INVERSE_STRIDE_0 + k * SEISMIC_INVERSE_STRIDE_1];
        const float projected = element::at<Act>(expert_output,
            (position / SEISMIC_DIM_T) * SEISMIC_EXPERT_OUTPUT_STRIDE_0
                + (position % SEISMIC_DIM_T) * SEISMIC_EXPERT_OUTPUT_STRIDE_1 + n * SEISMIC_EXPERT_OUTPUT_STRIDE_2);
        selected = __fmaf_rn(weights[m * SEISMIC_WEIGHTS_STRIDE_0 + k * SEISMIC_WEIGHTS_STRIDE_1], projected, selected);
    }
    const float *base = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_BASE));
    element::put<ELEMENT_OF(SEISMIC_ELEMENT_R)>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER),
                                                m * SEISMIC_RESULT_0_STRIDE_0 + n * SEISMIC_RESULT_0_STRIDE_1,
                                                base[m * SEISMIC_BASE_STRIDE_0 + n * SEISMIC_BASE_STRIDE_1] + selected);
}
