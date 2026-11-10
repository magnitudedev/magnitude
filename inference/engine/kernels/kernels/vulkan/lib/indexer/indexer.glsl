// The general indexer's Vulkan helpers (`indexer_keys`, `indexer_score`,
// `rows_topk`, `candidate_rows`, and `pool_rows`' window spans; contracts in
// indexer.seismic and pooled.seismic): a span's length and the walk of a
// row's spans to an ordinal's plane row. A head row's norm is
// `head_rms_norm`, its rotation `rotary_lane_rotate`'s, its rounding
// `rounding_lanes`, and the selection's scans `reduce_group_exclusive_add`.
// The counterpart of `metal/lib/indexer/indexer.h` and
// `cuda/lib/indexer/indexer.cuh`.
//
// This file is independent of any entry ABI. A span is two i32 at `spans`,
// its bounds `bound` elements apart; a row's spans step by `stride` elements.

// The length of span [lo, hi): its rows, or none when it is empty. (Vulkan
// has no trap: a nonempty span below row 0, which the spans' invariant
// excludes, counts as its rows like any other.)
uint indexer_span_length(int lo, int hi) {
    return hi > lo ? uint(hi - lo) : 0u;
}

uint indexer_span_length_at(uint64_t spans, uint64_t stride, uint64_t bound, uint64_t span) {
    const uint64_t at = spans + span * stride * 4ul;
    return indexer_span_length(element_i32_at(at), element_i32_at(at + bound * 4ul));
}

// The rows of a row's `count` spans.
uint indexer_span_rows(uint64_t spans, uint64_t count, uint64_t stride, uint64_t bound) {
    uint rows = 0u;
    for (uint64_t span = 0ul; span < count; ++span)
        rows += indexer_span_length_at(spans, stride, bound, span);
    return rows;
}

// The plane row of ordinal `ordinal` of a row's `count` spans: its span's
// first row plus its place. The caller holds the ordinal within the spans'
// rows (-1 past them).
int indexer_span_row(uint64_t spans, uint64_t count, uint64_t stride, uint64_t bound, uint64_t ordinal) {
    uint64_t before = 0ul;
    for (uint64_t span = 0ul; span < count; ++span) {
        const uint64_t at = spans + span * stride * 4ul;
        const int lo = element_i32_at(at);
        const uint length = indexer_span_length(lo, element_i32_at(at + bound * 4ul));
        if (ordinal < before + length)
            return lo + int(ordinal - before);
        before += length;
    }
    return -1;
}
