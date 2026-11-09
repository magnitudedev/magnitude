// rows_topk on CUDA (contract and portable body in indexer.seismic), as
// Metal's (`metal/rows_topk.metal`):
//
// A block of 256 threads per row. The row's span offsets (the listed rows
// before each span, then their count) are scanned into scratch; its scored
// ordinals are the first min(N, listed). A row that reads a prefix (dense,
// complete <= K, or with at most K scored ordinals) picks the first
// min(K, scored). A selecting row finds its K-th best score T by a radix
// select over the scores' order keys (`order::key`: a < b exactly when
// their keys are, -0.0 and 0.0 one key), a byte a pass from the top: each
// pass's histogram takes one atomic per distinct digit of a warp, and the
// digit is found by a scan of the bins. It picks every ordinal above T and
// the first equal ones in ordinal order (ranked by a scan of each thread's
// count of equal ones): the K best, ties to the lower ordinal. Thread t owns
// a contiguous share of the ordinals; a scan of the shares' pick counts
// places its picks, which it emits in ordinal order as their key-plane rows
// (its first span found by bisection of the offsets, then advanced span by
// span). The rest of the row's K picks are -1.
//
// The block scans are `reduce::group_exclusive_add`. Shared memory: the bins
// [256], the scans' 8 warp sums and the select state [4]. Dynamic extents and
// strides read `seismic_words`.

#include "lib/core/order.cuh"
#include "lib/core/reduce.cuh"
#include "lib/indexer/indexer.cuh"

namespace topk {

typedef unsigned int u32;
typedef unsigned long long u64;

constexpr u32 THREADS = 256;

} // namespace topk

extern "C" __global__ void rows_topk(SEISMIC_KERNEL_PARAMS) {
    using namespace topk;
    extern __shared__ u32 shared[];
    u32 *bins = shared;          // [256]
    u32 *totals = shared + 256;  // [8]: the scans' warp sums
    u32 *state = totals + 8;     // prefix, remaining
    const u64 row = blockIdx.x;
    const u32 thread = threadIdx.x;
    const int lane = static_cast<int>(thread % 32);
    const u32 n = static_cast<u32>(SEISMIC_DIM_N);
    const u32 k = static_cast<u32>(SEISMIC_DIM_K);
    const u32 r = static_cast<u32>(SEISMIC_DIM_R);
    if (k == 0) return;
    const int *row_spans = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_SPANS)) + row * SEISMIC_SPANS_STRIDE_0;
    const u64 span_stride = SEISMIC_SPANS_STRIDE_1;
    const u64 bound_stride = SEISMIC_SPANS_STRIDE_2;
    const float *row_scores = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCORES)) + row * SEISMIC_SCORES_STRIDE_0;
    const u64 score_stride = SEISMIC_SCORES_STRIDE_1;
    int *out = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER)) + row * SEISMIC_RESULT_0_STRIDE_0;
    const u64 out_stride = SEISMIC_RESULT_0_STRIDE_1;
    auto key = [&](u32 ordinal) { return order::key(row_scores[ordinal * score_stride]); };
    u32 total = 0;

    // The span offsets [R + 1]: the listed rows before each span, then the
    // row's.
    u32 *span_first = reinterpret_cast<u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_SPANS)) + row * (static_cast<u64>(r) + 1);
    u32 listed = 0;
    for (u32 base = 0; base < r; base += THREADS) {
        const u32 span = base + thread;
        const u32 length =
            span < r ? indexer::span_length(row_spans[span * span_stride], row_spans[span * span_stride + bound_stride])
                     : 0u;
        u32 chunk = 0;
        const u32 before = reduce::group_exclusive_add(length, totals, chunk);
        if (span < r) span_first[span] = listed + before;
        listed += chunk;
    }
    if (thread == 0) span_first[r] = listed;
    __syncthreads();
    const u32 scored = min(listed, n);

    // A prefix row picks its first `limit` ordinals; a selecting row those
    // above `threshold` and the first `take` equal to it.
    const int complete =
        reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_COMPLETE))[row * SEISMIC_COMPLETE_STRIDE_0];
    const bool prefix = complete <= static_cast<int>(k) || scored <= k;
    const u32 limit = min(scored, k);
    u32 threshold = 0;
    u32 take = 0;
    if (!prefix) {
        if (thread == 0) {
            state[0] = 0;
            state[1] = k;
        }
        u32 mask = 0;
        for (int digit = 3; digit >= 0; --digit) {
            bins[thread] = 0;
            __syncthreads();
            const u32 prefix_bits = state[0];
            const u32 remaining = state[1];
            const u32 shift = static_cast<u32>(digit) * 8u;
            // One atomic per distinct digit of a warp: its lowest lane adds
            // the count of the warp's lanes holding it.
            for (u32 base = 0; base < scored; base += THREADS) {
                const u32 ordinal = base + thread;
                const u32 u = ordinal < scored ? key(ordinal) : 0u;
                const bool counted = ordinal < scored && (u & mask) == prefix_bits;
                const u32 value = counted ? (u >> shift) & 255u : 256u;
                const u32 peers = __match_any_sync(0xffffffffu, value);
                if (counted && lane == __ffs(peers) - 1) atomicAdd(&bins[value], static_cast<u32>(__popc(peers)));
            }
            __syncthreads();
            // Thread t holds bin 255 - t: the candidates above it are the
            // scan of the bins before it. The bin holding the `remaining`-th
            // best takes the digit.
            const u32 bin = 255u - thread;
            const u32 count = bins[bin];
            const u32 above = reduce::group_exclusive_add(count, totals, total);
            const bool pick = above < remaining && remaining <= above + count;
            __syncthreads();
            if (pick) {
                state[0] = prefix_bits | (bin << shift);
                state[1] = remaining - above;
            }
            mask |= 255u << shift;
            __syncthreads();
        }
        threshold = state[0];
        take = state[1];
    }

    // Thread t's share of the scored ordinals [lo, hi).
    const u32 share = (scored + THREADS - 1) / THREADS;
    const u32 lo = min(thread * share, scored);
    const u32 hi = min(lo + share, scored);
    // The rank among the ordinals equal to T of the share's first.
    u32 equal = 0;
    if (!prefix)
        for (u32 ordinal = lo; ordinal < hi; ++ordinal) equal += key(ordinal) == threshold ? 1u : 0u;
    const u32 first_rank = reduce::group_exclusive_add(equal, totals, total);
    // Whether `ordinal` is picked, its equal ones ranked from `rank`.
    auto picked = [&](u32 ordinal, u32 &rank) {
        if (prefix) return ordinal < limit;
        const u32 u = key(ordinal);
        if (u > threshold) return true;
        if (u != threshold) return false;
        return rank++ < take;
    };
    u32 picks = 0;
    u32 rank = first_rank;
    for (u32 ordinal = lo; ordinal < hi; ++ordinal) picks += picked(ordinal, rank) ? 1u : 0u;
    u32 placed = 0;
    u32 position = reduce::group_exclusive_add(picks, totals, placed);
    if (lo < hi) {
        // The span holding `lo`: the last whose offset is at most it.
        u32 span = 0;
        for (u32 high = r - 1; span < high;) {
            const u32 middle = (span + high + 1) / 2;
            if (span_first[middle] <= lo)
                span = middle;
            else
                high = middle - 1;
        }
        rank = first_rank;
        for (u32 ordinal = lo; ordinal < hi; ++ordinal) {
            while (ordinal >= span_first[span + 1]) ++span;
            if (picked(ordinal, rank) && position < k)
                out[position++ * out_stride] = row_spans[span * span_stride] + static_cast<int>(ordinal - span_first[span]);
        }
    }
    for (u32 pick = placed + thread; pick < k; pick += THREADS) out[pick * out_stride] = -1;
}
