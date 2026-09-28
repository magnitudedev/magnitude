// draft_convolve_input: DFlash2's half-0 grouped dynamic causal convolution of
// a sublayer's normed input rows within each draft block; one thread per
// output element, summing its taps in offset order.
#include "lib/core/activation.cuh"

extern "C" __global__ void draft_convolve_input(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    const element::u8 *input = SEISMIC_PTR(SEISMIC_BUFFER_INPUT);
    const float *dynamic = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_DYNAMIC));
    const float *base = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_BASE));
    element::u8 *result = SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER);
    const u64 index = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    const u64 channels = SEISMIC_DIM_C, width = SEISMIC_DIM_G * channels;
    if (index >= (u64)SEISMIC_DIM_M * width)
        return;
    const u64 row = index / width, column = index % width, group = column / channels;
    const u64 position = row % (u64)SEISMIC_PARAM_BLOCK;
    float sum = 0.0f;
    for (u64 offset = 0; offset < SEISMIC_DIM_K && offset <= position; ++offset) {
        const float coefficient = base[offset * SEISMIC_BASE_STRIDE_1 + column * SEISMIC_BASE_STRIDE_2] +
            dynamic[row * SEISMIC_DYNAMIC_STRIDE_0 + offset * SEISMIC_DYNAMIC_STRIDE_2 + group * SEISMIC_DYNAMIC_STRIDE_3];
        sum = seismic_fma_rn(coefficient,
            element::at<element::Act>(input, (row - offset) * SEISMIC_INPUT_STRIDE_0 + column * SEISMIC_INPUT_STRIDE_1), sum);
    }
    element::put<element::Act>(result, row * SEISMIC_RESULT_0_STRIDE_0 + column * SEISMIC_RESULT_0_STRIDE_1, sum);
}
