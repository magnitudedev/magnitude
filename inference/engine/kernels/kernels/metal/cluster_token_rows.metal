kernel void cluster_token_rows(device const int *selected [[buffer(SEISMIC_BUFFER_SELECTED)]], device const int *token_order [[buffer(SEISMIC_BUFFER_TOKEN_ORDER)]], device int *rows [[buffer(SEISMIC_RESULT_0_BUFFER)]], constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]], uint raw_index [[thread_position_in_grid]]) {
    const ulong index = ulong(raw_index);
    if (index >= SEISMIC_DIM_M * SEISMIC_DIM_K * SEISMIC_DIM_B) return;
    const ulong row = index / (SEISMIC_DIM_K * SEISMIC_DIM_B);
    const ulong col = index % (SEISMIC_DIM_K * SEISMIC_DIM_B);
    const int cluster = selected[(row * SEISMIC_SELECTED_STRIDE_0 + (col / SEISMIC_DIM_B) * SEISMIC_SELECTED_STRIDE_1)];
    const int token = cluster >= 0 && ulong(cluster) < SEISMIC_DIM_C
        ? token_order[(ulong(cluster) * SEISMIC_TOKEN_ORDER_STRIDE_0 + (col % SEISMIC_DIM_B) * SEISMIC_TOKEN_ORDER_STRIDE_1)] : -1;
    rows[row * SEISMIC_RESULT_0_STRIDE_0 + col * SEISMIC_RESULT_0_STRIDE_1] = token;
}
