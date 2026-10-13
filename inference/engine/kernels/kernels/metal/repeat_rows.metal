// repeat_rows: S scaled copies of each row, one thread per value.

kernel void repeat_rows(device const float *rows [[buffer(SEISMIC_BUFFER_ROWS)]],
    device const float *scales [[buffer(SEISMIC_BUFFER_SCALES)]],
    device float *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint index [[thread_position_in_grid]]) {
    const uint h = uint(SEISMIC_DIM_H), width = uint(SEISMIC_DIM_S) * h;
    if (index >= uint(SEISMIC_DIM_M) * width)
        return;
    const uint row = index / width, stream = index % width / h, column = index % h;
    result[ulong(row) * SEISMIC_RESULT_0_STRIDE_0 + ulong(stream * h + column) * SEISMIC_RESULT_0_STRIDE_1] =
        rows[ulong(row) * SEISMIC_ROWS_STRIDE_0 + ulong(column) * SEISMIC_ROWS_STRIDE_1]
        * scales[ulong(stream) * SEISMIC_SCALES_STRIDE_0];
}
