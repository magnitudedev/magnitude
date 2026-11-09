// The span lists' and indexers' Metal helpers (contracts in
// indexer.seismic and pooled.seismic): the one owner of a span's rows
// (`span_length`, `span_row`; the counterpart of `cpu/lib/core/spans.rs`),
// and the threadgroup scan the selections place their picks with. A head
// row's norm and rotary are `head::rms_norm` and `rotary::layout`. This file
// names no entry ABI symbol.

namespace indexer {

// The exclusive prefix of `value` over the 256 threads of a threadgroup in
// thread order, and in `total` their sum. Every thread calls it; `totals`
// holds the 8 simdgroups' sums.
inline uint group_exclusive(uint value, threadgroup uint *totals, uint simd, uint lane, thread uint &total) {
    const uint inclusive = simd_prefix_inclusive_sum(value);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 31)
        totals[simd] = inclusive;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint before = 0;
    total = 0;
    for (uint other = 0; other < 8; ++other) {
        const uint sum = totals[other];
        before += other < simd ? sum : 0u;
        total += sum;
    }
    return before + inclusive - value;
}

// The rows of span [lo, hi): it holds rows iff lo >= 0 and hi > lo. A span
// with lo < 0 and hi > lo would hold rows below row 0, an invariant
// violation every producer excludes (the CPU asserts it); Metal cannot trap,
// so such a span counts as empty, decided here alone.
inline uint span_length(int lo, int hi) {
    return lo >= 0 && hi > lo ? uint(hi - lo) : 0u;
}

// The row of ordinal `ordinal` of a row's `count` spans, -1 past them.
inline int span_row(device const int *spans, uint count, uint ordinal) {
    uint before = 0;
    for (uint span = 0; span < count; ++span) {
        const int lo = spans[span * 2];
        const uint length = span_length(lo, spans[span * 2 + 1]);
        if (ordinal < before + length)
            return lo + int(ordinal - before);
        before += length;
    }
    return -1;
}

} // namespace indexer
