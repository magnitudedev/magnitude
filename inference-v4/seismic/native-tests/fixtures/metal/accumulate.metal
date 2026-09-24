kernel void accumulate(
    device float *state [[buffer(SEISMIC_BUFFER_STATE)]],
    device const float *x [[buffer(SEISMIC_BUFFER_X)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint index [[thread_position_in_grid]]) {
    if (index < SEISMIC_DIM_N) {
        const ulong at = index * SEISMIC_STATE_STRIDE_0;
        state[at] = state[at] + (x[index * SEISMIC_X_STRIDE_0] + float(SEISMIC_TUNE_BIAS) * 0.5f);
    }
}
