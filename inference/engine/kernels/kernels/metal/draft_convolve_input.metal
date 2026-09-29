// draft_convolve_input: DFlash2's half-0 grouped dynamic causal convolution of
// a sublayer's normed input rows within each draft block; one thread per
// output element, summing its taps in offset order.
#include "lib/core/activation.h"

typedef element::Act activation;

kernel void draft_convolve_input(
    device const uchar *input [[buffer(SEISMIC_BUFFER_INPUT)]],
    device const float *dynamic [[buffer(SEISMIC_BUFFER_DYNAMIC)]],
    device const float *base [[buffer(SEISMIC_BUFFER_BASE)]],
    device uchar *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint raw_index [[thread_position_in_grid]]) {
    const ulong index = ulong(raw_index);
    const ulong channels = SEISMIC_DIM_C, width = SEISMIC_DIM_G * channels;
    if (index >= SEISMIC_DIM_M * width) return;
    const ulong row = index / width, column = index % width, group = column / channels;
    const ulong position = row % ulong(SEISMIC_PARAM_BLOCK);
    float sum = 0.0f;
    for (ulong offset = 0; offset < SEISMIC_DIM_K && offset <= position; ++offset) {
        const float coefficient = base[offset * SEISMIC_BASE_STRIDE_1 + column * SEISMIC_BASE_STRIDE_2] +
            dynamic[row * SEISMIC_DYNAMIC_STRIDE_0 + offset * SEISMIC_DYNAMIC_STRIDE_2 + group * SEISMIC_DYNAMIC_STRIDE_3];
        sum = metal::fma(coefficient,
            element::at<activation>(input, (row - offset) * SEISMIC_INPUT_STRIDE_0 + column * SEISMIC_INPUT_STRIDE_1), sum);
    }
    element::put<activation>(result, row * SEISMIC_RESULT_0_STRIDE_0 + column * SEISMIC_RESULT_0_STRIDE_1, sum);
}
