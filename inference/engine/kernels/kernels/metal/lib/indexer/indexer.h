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

// The listed rows of a row's `count` spans, returned to each lane of a
// simdgroup, and in `row` the key-plane row of the lane's `ordinal` (-1
// past them). The lanes take the spans 32 at a time: a scan of their
// lengths gives each span's first ordinal, and a lane whose ordinal the
// chunk holds finds its span as the last whose first ordinal is at most
// it, by bisection over the lanes (an empty span's first ordinal is the
// next span's, so it is never the last). A lane does O(count / 32) work,
// where walking the spans to its ordinal (`span_row`) is O(count). Every
// lane calls it with the same spans.
inline uint simd_span_row(device const int *spans, uint count, uint ordinal, uint lane, thread int &row) {
    uint listed = 0;
    row = -1;
    for (uint base = 0; base < count; base += 32) {
        const uint span = base + lane;
        const int lo = span < count ? spans[span * 2] : 0;
        const uint length = span < count ? span_length(lo, spans[span * 2 + 1]) : 0u;
        const uint first = listed + simd_prefix_exclusive_sum(length);
        const uint chunk = simd_sum(length);
        ushort holder = 0;
        for (ushort step = 16; step > 0; step >>= 1) {
            if (simd_shuffle(first, ushort(holder + step)) <= ordinal)
                holder += step;
        }
        const int holder_lo = simd_shuffle(lo, holder);
        const uint holder_first = simd_shuffle(first, holder);
        if (ordinal >= listed && ordinal < listed + chunk)
            row = holder_lo + int(ordinal - holder_first);
        listed += chunk;
    }
    return listed;
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
