#pragma METAL internals : enable

// Expert streaming, a scratch fill's copy (`routed_copy` in routed.seismic):
// the listed bank slots into their scratch slots, byte for byte. Bank and
// scratch hold one resident form, so a matrix is the same bytes in both.
// Threadgroup rows stride over the list entries; the 64 x 256 threads of a
// row stride over each matrix in 16-byte vectors, then its tail bytes.

// A matrix's byte stride: a packet layout's matrix stride counts packets, a
// row layout's counts stored rows.
#if defined(SEISMIC_ELEMENT_U_LAYOUT_PACKET)
#define COPY_MATRIX_BYTES (SEISMIC_BANK_STRIDE_0 * SEISMIC_ELEMENT_U_PACKET_SIZE)
#else
#define COPY_MATRIX_BYTES (SEISMIC_BANK_STRIDE_0 * SEISMIC_BANK_ROW_STRIDE_BYTES)
#endif

kernel void routed_copy(
    device const uchar *bank [[buffer(SEISMIC_BUFFER_BANK)]],
    volatile coherent(system) device const int *list [[buffer(SEISMIC_BUFFER_LIST)]],
    device const int *step [[buffer(SEISMIC_BUFFER_STEP)]],
    device uchar *scratch [[buffer(SEISMIC_BUFFER_SCRATCH)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 groups [[threadgroups_per_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    if (list[0] != step[0]) return;
    const ulong bytes = ulong(COPY_MATRIX_BYTES);
    const ulong lane = ulong(group.x) * 256 + tid, lanes = ulong(groups.x) * 256;
    const ulong vectors = bytes / 16;
    const int count = list[1];
    for (int entry = int(group.y); entry < count; entry += int(groups.y)) {
        device const uchar *from = bank + ulong(list[2 + 2 * entry]) * bytes;
        device uchar *to = scratch + ulong(list[3 + 2 * entry]) * bytes;
        for (ulong v = lane; v < vectors; v += lanes)
            ((device uint4 *)to)[v] = ((device const uint4 *)from)[v];
        for (ulong b = vectors * 16 + lane; b < bytes; b += lanes)
            to[b] = from[b];
    }
}
