// copy_rows: dst[to[item], head, :] = src[from[item], head, :] for dense state
// planes of one element type; one thread per copied element. Rows outside
// either plane (including -1) are skipped.
#if !defined(SEISMIC_SRC_KIND_DENSE) || !defined(SEISMIC_DST_KIND_DENSE)
#error "copy_rows supports dense state planes only"
#endif
#if SEISMIC_SRC_PACKET_SIZE != SEISMIC_DST_PACKET_SIZE
#error "copy_rows requires identical source and destination plane dtypes"
#endif

extern "C" __global__ void copy_rows(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    const unsigned char *src = reinterpret_cast<const unsigned char *>(SEISMIC_PTR(SEISMIC_BUFFER_SRC));
    unsigned char *dst = reinterpret_cast<unsigned char *>(SEISMIC_PTR(SEISMIC_BUFFER_DST));
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
    if (source_row < 0 || destination_row < 0 || (u64)source_row >= SEISMIC_DIM_TS
        || (u64)destination_row >= SEISMIC_DIM_TD)
        return;
    const u64 source = ((u64)source_row * SEISMIC_SRC_STRIDE_0 + head * SEISMIC_SRC_STRIDE_1
                        + column * SEISMIC_SRC_STRIDE_2) * SEISMIC_SRC_PACKET_SIZE;
    const u64 destination = ((u64)destination_row * SEISMIC_DST_STRIDE_0 + head * SEISMIC_DST_STRIDE_1
                             + column * SEISMIC_DST_STRIDE_2) * SEISMIC_DST_PACKET_SIZE;
    for (u64 byte = 0; byte < SEISMIC_SRC_PACKET_SIZE; ++byte)
        dst[destination + byte] = src[source + byte];
}
