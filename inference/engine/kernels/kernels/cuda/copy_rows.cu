// copy_rows: rows[to[item], head, :] = rows[from[item], head, :] for dense state
// planes of one element type; one thread per copied element. Rows outside
// plane (including -1) are skipped.
#if !defined(SEISMIC_ROWS_KIND_DENSE)
#error "copy_rows supports dense state planes only"
#endif
#include <seismic/slab.cuh>

extern "C" __global__ void copy_rows(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    const unsigned int *table = reinterpret_cast<const unsigned int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROWS));
    const int *from = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_FROM));
    const int *to = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_TO));
    const u64 index = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    const u64 row_elements = (u64)SEISMIC_DIM_KV * SEISMIC_DIM_W;
    if (index >= (u64)SEISMIC_DIM_N * row_elements)
        return;
    const u64 item = index / row_elements;
    const u64 head = (index % row_elements) / SEISMIC_DIM_W;
    const u64 column = index % SEISMIC_DIM_W;
    const int source_row = from[item * SEISMIC_FROM_STRIDE_0];
    const int destination_row = to[item * SEISMIC_TO_STRIDE_0];
    if (source_row < 0 || destination_row < 0 || (u64)source_row >= SEISMIC_DIM_T
        || (u64)destination_row >= SEISMIC_DIM_T)
        return;
    const unsigned char *src = slab::row(table, (u64)source_row, SEISMIC_PARAM_SLAB_ROWS,
        SEISMIC_ROWS_STRIDE_0 * SEISMIC_ROWS_PACKET_SIZE);
    unsigned char *dst = slab::row(table, (u64)destination_row, SEISMIC_PARAM_SLAB_ROWS,
        SEISMIC_ROWS_STRIDE_0 * SEISMIC_ROWS_PACKET_SIZE);
    const u64 source = (head * SEISMIC_ROWS_STRIDE_1 + column * SEISMIC_ROWS_STRIDE_2) * SEISMIC_ROWS_PACKET_SIZE;
    const u64 destination = (head * SEISMIC_ROWS_STRIDE_1 + column * SEISMIC_ROWS_STRIDE_2) * SEISMIC_ROWS_PACKET_SIZE;
    for (u64 byte = 0; byte < SEISMIC_ROWS_PACKET_SIZE; ++byte)
        dst[destination + byte] = src[source + byte];
}
