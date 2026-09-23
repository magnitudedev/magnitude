kernel void qwen_conditioning_overlay(
    device const float *input [[buffer(SEISMIC_BUFFER_INPUT)]],
    device float *out [[buffer(SEISMIC_BUFFER_OUT)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint raw_index [[thread_position_in_grid]]) {
    ulong index = ulong(raw_index);
    if (index >= SEISMIC_DIM_M * SEISMIC_DIM_D) return;
    ulong row = index / SEISMIC_DIM_D;
    ulong column = index % SEISMIC_DIM_D;
    out[row * SEISMIC_OUT_STRIDE_0 + column * SEISMIC_OUT_STRIDE_1] =
        input[row * SEISMIC_INPUT_STRIDE_0 + column * SEISMIC_INPUT_STRIDE_1];
}
