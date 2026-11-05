// `marked_groups` (conditional.seismic): every mark cleared, then each group
// of the grid the launch reads marks its own.

kernel void marked_groups_clear(
    device int *marks [[buffer(SEISMIC_BUFFER_MARKS)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint index [[thread_position_in_grid]]) {
    if (index < SEISMIC_DIM_N)
        marks[index * SEISMIC_MARKS_STRIDE_0] = 0;
}

kernel void marked_groups(
    device int *marks [[buffer(SEISMIC_BUFFER_MARKS)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint group [[threadgroup_position_in_grid]]) {
    marks[group * SEISMIC_MARKS_STRIDE_0] = 1;
}
