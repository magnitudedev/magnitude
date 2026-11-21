// The general indexer's CUDA helpers (`indexer_keys`, `indexer_score`,
// `rows_topk`, `candidate_rows`, and `pool_rows`' window spans; contracts in
// indexer.seismic and pooled.seismic): a span's length and the walk of a
// row's spans to an ordinal's plane row. A head row's
// norm is `head::rms_norm`, its rotation `rotary::layout`'s, its rounding
// `rounding::round_lanes`, and the selection's block scans
// `reduce::group_exclusive_add`. The counterpart of
// `metal/lib/indexer/indexer.h`.
//
// This file is independent of any entry ABI.

#pragma once

namespace indexer {

// The length of span [lo, hi): its rows, or none when it is empty. A
// nonempty span below row 0 violates the spans' invariant and traps.
__device__ __forceinline__ unsigned span_length(int lo, int hi) {
    if (lo < 0 && hi > lo) __trap();
    return hi > lo ? static_cast<unsigned>(hi - lo) : 0u;
}

// The plane row of ordinal `ordinal` of a row's `count` spans, `stride` apart
// with their bounds `bound` apart: its span's first row plus its place. An
// ordinal past the spans' rows violates the caller's invariant and traps.
__device__ __forceinline__ int span_row(const int *spans, unsigned long long count, unsigned long long stride,
                                        unsigned long long bound, unsigned long long ordinal) {
    unsigned long long before = 0;
    for (unsigned long long span = 0; span < count; ++span) {
        const int lo = spans[span * stride];
        const unsigned length = span_length(lo, spans[span * stride + bound]);
        if (ordinal < before + length) return lo + static_cast<int>(ordinal - before);
        before += length;
    }
    __trap();
    __builtin_unreachable();
}

// The listed rows of a row's `count` spans (`stride` apart, bounds `bound`
// apart), returned to each lane of a warp, and in `row` the key-plane row of
// the lane's `ordinal` (-1 past them). The lanes take the spans 32 at a time:
// a warp scan of their lengths gives each span's first ordinal, and a lane
// whose ordinal the chunk holds finds its span as the last whose first ordinal
// is at most it, by bisection over the lanes (an empty span's first ordinal is
// the next span's, so it is never the last). A lane does O(count / 32) work,
// where walking the spans to its ordinal (`span_row`) is O(count). Every lane
// calls it with the same spans, convergent.
__device__ __forceinline__ unsigned long long warp_span_row(const int *spans, unsigned long long count,
                                                            unsigned long long stride, unsigned long long bound,
                                                            unsigned long long ordinal, int lane, int &row) {
    unsigned long long listed = 0;
    row = -1;
    for (unsigned long long base = 0; base < count; base += 32) {
        const unsigned long long span = base + static_cast<unsigned long long>(lane);
        const int lo = span < count ? spans[span * stride] : 0;
        const unsigned long long length =
            span < count ? static_cast<unsigned long long>(span_length(lo, spans[span * stride + bound])) : 0ull;
        unsigned long long inclusive = length;
#pragma unroll
        for (int offset = 1; offset < 32; offset *= 2) {
            const unsigned long long below = __shfl_up_sync(0xffffffffu, inclusive, offset);
            if (lane >= offset) inclusive += below;
        }
        const unsigned long long first = listed + (inclusive - length);
        const unsigned long long chunk = __shfl_sync(0xffffffffu, inclusive, 31);
        int holder = 0;
#pragma unroll
        for (int step = 16; step > 0; step >>= 1) {
            if (__shfl_sync(0xffffffffu, first, holder + step) <= ordinal) holder += step;
        }
        const int holder_lo = __shfl_sync(0xffffffffu, lo, holder);
        const unsigned long long holder_first = __shfl_sync(0xffffffffu, first, holder);
        if (ordinal >= listed && ordinal < listed + chunk) row = holder_lo + static_cast<int>(ordinal - holder_first);
        listed += chunk;
    }
    return listed;
}

} // namespace indexer
