// selection_notice: notice[row] = selection[row], a (token, status) pair per
// row; a thread per row.
kernel void selection_notice(
    device const int *selection [[buffer(SEISMIC_BUFFER_SELECTION)]],
    device int *notice [[buffer(SEISMIC_BUFFER_NOTICE)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint raw_row [[thread_position_in_grid]]) {
    const ulong row = ulong(raw_row);
    if (row >= SEISMIC_DIM_M)
        return;
    for (ulong field = 0; field < 2; ++field)
        notice[row * SEISMIC_NOTICE_STRIDE_0 + field * SEISMIC_NOTICE_STRIDE_1] =
            selection[row * SEISMIC_SELECTION_STRIDE_0 + field * SEISMIC_SELECTION_STRIDE_1];
}
