// unit_reads on Metal (contract and portable body in indexer.seismic).
//
// A threadgroup of 256 threads per row. Its picks are ascending and -1
// padded (`rows_topk`): thread 0 finds the count of valid picks by binary
// search and broadcasts it through threadgroup memory; all 256 threads write
// `units` (the picks, then the tail unit, then -1) in parallel, and each
// thread sets the bitmap words it owns by binary search over the valid picks.
//
// Dynamic extents read `seismic_words`.

kernel void unit_reads(
    device const int *picks [[buffer(SEISMIC_BUFFER_PICKS)]],
    device const int *complete [[buffer(SEISMIC_BUFFER_COMPLETE)]],
    device uint *bitmap [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    device int *units [[buffer(SEISMIC_RESULT_1_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup uint shared_count;
    const ulong row = group.x;
    const uint k = uint(SEISMIC_DIM_K);
    const uint b = uint(SEISMIC_DIM_B);
    device const int *row_picks = picks + row * k;
    const int tail = complete[row];
    // The valid picks: the first `count`, found once by thread 0 and broadcast.
    if (thread_index == 0) {
        uint lo = 0, hi = k;
        while (lo < hi) {
            const uint mid = (lo + hi) / 2;
            if (row_picks[mid] >= 0)
                lo = mid + 1;
            else
                hi = mid;
        }
        shared_count = lo;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint count = shared_count;
    device int *row_units = units + row * (k + 1);
    for (uint slot = thread_index; slot <= k; slot += 256)
        row_units[slot] = slot < count ? row_picks[slot] : (slot == count ? tail : -1);
    for (uint word = thread_index; word < b; word += 256) {
        const int first = int(word * 32);
        // The first pick at or past the word's first unit.
        uint at = 0, end = count;
        while (at < end) {
            const uint mid = (at + end) / 2;
            if (row_picks[mid] < first)
                at = mid + 1;
            else
                end = mid;
        }
        uint bits = 0;
        for (; at < count && row_picks[at] < first + 32; ++at)
            bits |= 1u << uint(row_picks[at] - first);
        // Every unit from the tail's on.
        if (tail <= first)
            bits = 0xffffffffu;
        else if (tail < first + 32)
            bits |= 0xffffffffu << uint(tail - first);
        bitmap[row * b + word] = bits;
    }
}
