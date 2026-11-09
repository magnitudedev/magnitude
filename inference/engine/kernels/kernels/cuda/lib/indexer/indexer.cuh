// The general indexer's CUDA helpers (`indexer_keys`, `indexer_score`,
// `rows_topk`; contracts in indexer.seismic): a span's length. A head row's
// norm is `head::rms_norm`, its rotation `rotary::layout`'s, its rounding
// `rounding::round_lanes`, and the selection's block scans
// `reduce::group_exclusive_add`. The counterpart of
// `metal/lib/indexer/indexer.h`.
//
// This file is independent of any entry ABI.

#pragma once

namespace indexer {

// The length of span [lo, hi): its rows, or none when it is empty.
__device__ __forceinline__ unsigned span_length(int lo, int hi) {
    return hi > lo ? static_cast<unsigned>(hi - lo) : 0u;
}

} // namespace indexer
