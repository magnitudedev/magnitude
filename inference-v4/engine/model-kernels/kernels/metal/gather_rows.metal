kernel void gather_rows(
    device const float *source [[buffer(SEISMIC_BUFFER_SOURCE)]],
    device const int *rows [[buffer(SEISMIC_BUFFER_ROWS)]],
    device float *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint raw_index [[thread_position_in_grid]]) {
    ulong index = ulong(raw_index);
    if (index >= SEISMIC_DIM_O * SEISMIC_DIM_D) return;
    ulong row = index / SEISMIC_DIM_D;
    ulong col = index % SEISMIC_DIM_D;
    ulong source_row = ulong(rows[row * SEISMIC_ROWS_STRIDE_0]);
    result[row * SEISMIC_RESULT_0_STRIDE_0 + col * SEISMIC_RESULT_0_STRIDE_1] =
        source[source_row * SEISMIC_SOURCE_STRIDE_0 + col * SEISMIC_SOURCE_STRIDE_1];
}
