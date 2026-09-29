kernel void scale_rows(
    device const float *x [[buffer(SEISMIC_BUFFER_X)]],
    device float *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    const float factor = as_type<float>(uint(SEISMIC_PARAM_FACTOR));
    for (uint local = 0; local < SEISMIC_TUNE_ROWS; ++local) {
        const ulong row = ulong(group) * SEISMIC_TUNE_ROWS + local;
        if (row >= SEISMIC_DIM_M) {
            return;
        }
        for (ulong column = lane; column < SEISMIC_DIM_N; column += lanes) {
            result[row * SEISMIC_RESULT_0_STRIDE_0 + column * SEISMIC_RESULT_0_STRIDE_1] =
                x[row * SEISMIC_X_STRIDE_0 + column * SEISMIC_X_STRIDE_1] * factor;
        }
    }
}
