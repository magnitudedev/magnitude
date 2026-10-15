#pragma METAL internals : enable

// Expert streaming, a scratch fill's copy (`routed_copy` in routed.seismic):
// the listed records of the source into the destination, byte for byte. Bank
// and scratch hold one resident form, so a matrix is the same bytes in both.
// Threadgroup rows stride over the list entries; the 64 x 256 threads of a
// row stride over each matrix in 16-byte vectors, then its tail bytes. One
// thread reads the list's system-coherent words into threadgroup memory.

// A matrix's byte stride: a packet layout's matrix stride counts packets, a
// row layout's counts stored rows.
#if defined(SEISMIC_ELEMENT_U_LAYOUT_PACKET)
#define COPY_MATRIX_BYTES (SEISMIC_SOURCE_STRIDE_0 * SEISMIC_ELEMENT_U_PACKET_SIZE)
#else
#define COPY_MATRIX_BYTES (SEISMIC_SOURCE_STRIDE_0 * SEISMIC_SOURCE_ROW_STRIDE_BYTES)
#endif

kernel void routed_copy(
    device const uchar *source [[buffer(SEISMIC_BUFFER_SOURCE)]],
    volatile coherent(system) device const int *list [[buffer(SEISMIC_BUFFER_LIST)]],
    device const int *step [[buffer(SEISMIC_BUFFER_STEP)]],
    device uchar *destination [[buffer(SEISMIC_BUFFER_DESTINATION)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 groups [[threadgroups_per_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup int header[2];
    threadgroup int entry_records[2];
    if (tid == 0) {
        header[0] = list[0];
        header[1] = list[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (header[0] != step[0]) return;
    const ulong bytes = ulong(COPY_MATRIX_BYTES);
    const ulong lane = ulong(group.x) * 256 + tid, lanes = ulong(groups.x) * 256;
    const ulong vectors = bytes / 16;
    const int count = header[1];
    for (int entry = int(group.y); entry < count; entry += int(groups.y)) {
        if (tid == 0) {
            entry_records[0] = list[2 + 2 * entry];
            entry_records[1] = list[3 + 2 * entry];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        device const uchar *from = source + ulong(entry_records[0]) * bytes;
        device uchar *to = destination + ulong(entry_records[1]) * bytes;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (ulong v = lane; v < vectors; v += lanes)
            ((device uint4 *)to)[v] = ((device const uint4 *)from)[v];
        for (ulong b = vectors * 16 + lane; b < bytes; b += lanes)
            to[b] = from[b];
    }
}
