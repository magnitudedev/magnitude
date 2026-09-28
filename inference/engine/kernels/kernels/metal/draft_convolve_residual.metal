// draft_convolve_residual: DFlash2's half-1 grouped dynamic causal convolution
// of a sublayer's F32 output rows within each draft block, plus the residual;
// one thread per output element, summing its taps in offset order.

kernel void draft_convolve_residual(
    device const float *residual [[buffer(SEISMIC_BUFFER_RESIDUAL)]],
    device const float *output [[buffer(SEISMIC_BUFFER_OUTPUT)]],
    device const float *dynamic [[buffer(SEISMIC_BUFFER_DYNAMIC)]],
    device const float *base [[buffer(SEISMIC_BUFFER_BASE)]],
    device float *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint raw_index [[thread_position_in_grid]]) {
    const ulong index = ulong(raw_index);
    const ulong channels = SEISMIC_DIM_C, width = SEISMIC_DIM_G * channels;
    if (index >= SEISMIC_DIM_M * width) return;
    const ulong row = index / width, column = index % width, group = column / channels;
    const ulong position = row % ulong(SEISMIC_PARAM_BLOCK);
    float sum = 0.0f;
    for (ulong offset = 0; offset < SEISMIC_DIM_K && offset <= position; ++offset) {
        const float coefficient =
            base[SEISMIC_BASE_STRIDE_0 + offset * SEISMIC_BASE_STRIDE_1 + column * SEISMIC_BASE_STRIDE_2] +
            dynamic[row * SEISMIC_DYNAMIC_STRIDE_0 + SEISMIC_DYNAMIC_STRIDE_1 + offset * SEISMIC_DYNAMIC_STRIDE_2 +
                group * SEISMIC_DYNAMIC_STRIDE_3];
        sum = metal::fma(coefficient,
            output[(row - offset) * SEISMIC_OUTPUT_STRIDE_0 + column * SEISMIC_OUTPUT_STRIDE_1], sum);
    }
    result[row * SEISMIC_RESULT_0_STRIDE_0 + column * SEISMIC_RESULT_0_STRIDE_1] =
        residual[row * SEISMIC_RESIDUAL_STRIDE_0 + column * SEISMIC_RESIDUAL_STRIDE_1] + sum;
}
