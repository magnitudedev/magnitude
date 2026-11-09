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

} // namespace indexer
