kernel void cluster_token_rows(
    device const int *selected [[buffer(SEISMIC_BUFFER_SELECTED)]],
    device const int *token_order [[buffer(SEISMIC_BUFFER_TOKEN_ORDER)]],
    device int *rows [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 position [[thread_position_in_grid]]) {
    const ulong offset = ulong(position.x);
    const ulong rank = ulong(position.y);
    const ulong row = ulong(position.z);
    if (offset >= SEISMIC_DIM_B || rank >= SEISMIC_DIM_K || row >= SEISMIC_DIM_M) return;
    const int cluster = selected[row * SEISMIC_SELECTED_STRIDE_0 + rank * SEISMIC_SELECTED_STRIDE_1];
    const int token = cluster >= 0 && ulong(cluster) < SEISMIC_DIM_C
        ? token_order[ulong(cluster) * SEISMIC_TOKEN_ORDER_STRIDE_0 + offset * SEISMIC_TOKEN_ORDER_STRIDE_1]
        : -1;
    const ulong col = rank * SEISMIC_DIM_B + offset;
    rows[row * SEISMIC_RESULT_0_STRIDE_0 + col * SEISMIC_RESULT_0_STRIDE_1] = token;
}
