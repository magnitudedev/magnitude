#include <seismic/slab.h>

kernel void copy_rows(
    device const ulong *rows [[buffer(SEISMIC_BUFFER_ROWS)]],
    device const uchar *from [[buffer(SEISMIC_BUFFER_FROM)]],
    device const uchar *to [[buffer(SEISMIC_BUFFER_TO)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint raw_index [[thread_position_in_grid]]) {
#if !defined(SEISMIC_ROWS_KIND_DENSE)
#error "copy_rows supports dense state planes only"
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
        || ulong(source_row) >= SEISMIC_DIM_T || ulong(destination_row) >= SEISMIC_DIM_T) return;
    const ulong slab_rows = ulong(SEISMIC_PARAM_SLAB_ROWS);
    device const uchar *source = slab::region<uchar>(rows, ulong(source_row) / slab_rows);
    device uchar *target = slab::region<uchar>(rows, ulong(destination_row) / slab_rows);
    ulong source_offset = ((ulong(source_row) % slab_rows) * SEISMIC_ROWS_STRIDE_0
        + head * SEISMIC_ROWS_STRIDE_1 + column * SEISMIC_ROWS_STRIDE_2)
        * SEISMIC_ROWS_PACKET_SIZE;
    ulong destination_offset = ((ulong(destination_row) % slab_rows) * SEISMIC_ROWS_STRIDE_0
        + head * SEISMIC_ROWS_STRIDE_1 + column * SEISMIC_ROWS_STRIDE_2)
        * SEISMIC_ROWS_PACKET_SIZE;
    for (ulong byte = 0; byte < SEISMIC_ROWS_PACKET_SIZE; ++byte)
        target[destination_offset + byte] = source[source_offset + byte];
}
