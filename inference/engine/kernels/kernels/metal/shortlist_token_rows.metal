kernel void shortlist_token_rows(
    device const int *selected [[buffer(SEISMIC_BUFFER_SELECTED)]],
    device int *rows [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint2 position [[thread_position_in_grid]]) {
    const ulong rank = ulong(position.x);
    const ulong row = ulong(position.y);
    if (rank >= SEISMIC_DIM_K || row >= SEISMIC_DIM_M) return;
    const int token = selected[row * SEISMIC_SELECTED_STRIDE_0 + rank * SEISMIC_SELECTED_STRIDE_1];
    rows[row * SEISMIC_RESULT_0_STRIDE_0 + rank * SEISMIC_RESULT_0_STRIDE_1] = token;
}
