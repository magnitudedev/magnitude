// `written_marks` (conditional.seismic): the grid written, every mark
// cleared, then each group of the written grid marks its own.

kernel void written_marks_grid(
    device const int *count [[buffer(SEISMIC_BUFFER_COUNT)]],
    device int *grid [[buffer(SEISMIC_BUFFER_SCRATCH_GRID)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]) {
    grid[0] = count[0];
    grid[1] = 2;
    grid[2] = 1;
}

kernel void written_marks_clear(
    device int *marks [[buffer(SEISMIC_BUFFER_MARKS)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint index [[thread_position_in_grid]]) {
    if (index < SEISMIC_DIM_N * 2)
        marks[(index / SEISMIC_DIM_N) * SEISMIC_MARKS_STRIDE_0 + (index % SEISMIC_DIM_N) * SEISMIC_MARKS_STRIDE_1] = 0;
}

kernel void written_marks(
    device int *marks [[buffer(SEISMIC_BUFFER_MARKS)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    if (lane == 0)
        marks[group.y * SEISMIC_MARKS_STRIDE_0 + group.x * SEISMIC_MARKS_STRIDE_1] = 1;
}
