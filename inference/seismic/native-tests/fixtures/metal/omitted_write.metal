kernel void omitted_write(
    device const float *x [[buffer(SEISMIC_BUFFER_X)]],
    device float *output [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint index [[thread_position_in_grid]]) {
    if (SEISMIC_TUNE_WRITE && index < SEISMIC_DIM_N) output[index] = x[index * SEISMIC_X_STRIDE_0];
}
