kernel void stray_write(
    device float *state [[buffer(SEISMIC_BUFFER_STATE)]],
    device const float *x [[buffer(SEISMIC_BUFFER_X)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint index [[thread_position_in_grid]]) {
    if (index < SEISMIC_DIM_N) {
        const bool stray = SEISMIC_TUNE_STRAY && index == SEISMIC_DIM_N - 1;
        state[index * SEISMIC_STATE_STRIDE_0] = stray ? 42.0f : x[index * SEISMIC_X_STRIDE_0];
    }
}
