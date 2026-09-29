kernel void copy_rows(
    device const uchar *src [[buffer(SEISMIC_BUFFER_SRC)]],
    device uchar *dst [[buffer(SEISMIC_BUFFER_DST)]],
    device const uchar *from [[buffer(SEISMIC_BUFFER_FROM)]],
    device const uchar *to [[buffer(SEISMIC_BUFFER_TO)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint raw_index [[thread_position_in_grid]]) {
#if !defined(SEISMIC_SRC_KIND_DENSE) || !defined(SEISMIC_DST_KIND_DENSE)
#error "copy_rows supports dense state planes only"
#endif
#if SEISMIC_SRC_PACKET_SIZE != SEISMIC_DST_PACKET_SIZE
#error "copy_rows requires identical source and destination plane dtypes"
#endif
    ulong index = ulong(raw_index);
    ulong count = SEISMIC_DIM_N * SEISMIC_DIM_KV * SEISMIC_DIM_W;
    if (index >= count) return;
    ulong item = index / (SEISMIC_DIM_KV * SEISMIC_DIM_W);
    ulong rem = index % (SEISMIC_DIM_KV * SEISMIC_DIM_W);
    ulong head = rem / SEISMIC_DIM_W;
    ulong column = rem % SEISMIC_DIM_W;
    int source_row = *reinterpret_cast<device const int *>(
        from + item * SEISMIC_FROM_STRIDE_0 * sizeof(int));
    int destination_row = *reinterpret_cast<device const int *>(
        to + item * SEISMIC_TO_STRIDE_0 * sizeof(int));
    if (source_row < 0 || destination_row < 0
        || ulong(source_row) >= SEISMIC_DIM_TS || ulong(destination_row) >= SEISMIC_DIM_TD) return;
    ulong source_offset = (ulong(source_row) * SEISMIC_SRC_STRIDE_0
        + head * SEISMIC_SRC_STRIDE_1 + column * SEISMIC_SRC_STRIDE_2)
        * SEISMIC_SRC_PACKET_SIZE;
    ulong destination_offset = (ulong(destination_row) * SEISMIC_DST_STRIDE_0
        + head * SEISMIC_DST_STRIDE_1 + column * SEISMIC_DST_STRIDE_2)
        * SEISMIC_DST_PACKET_SIZE;
    for (ulong byte = 0; byte < SEISMIC_SRC_PACKET_SIZE; ++byte)
        dst[destination_offset + byte] = src[source_offset + byte];
}
