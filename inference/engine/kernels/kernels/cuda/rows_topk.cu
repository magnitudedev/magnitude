// rows_topk on CUDA (contract and portable body in indexer.seismic):
//
// A block of THREADS threads per row. The row's span offsets (the listed
// rows before each span, then their count) are scanned into scratch; its
// scored ordinals are the first min(N, listed). A row that reads a prefix
// (dense, complete <= K, or with at most K scored ordinals) picks the first
// min(K, scored). A selecting row finds its K-th best score T by a radix
// select over the scores' order keys (`order::key`: a < b exactly when
// their keys are, -0.0 and 0.0 one key), a byte a pass from the top: each
// pass's histogram takes one shared atomic per counted ordinal, and the
// digit is found by a scan of the bins. It picks every ordinal above T and
// the first equal ones in ordinal order (ranked by a scan of each thread's
// count of equal ones): the K best, ties to the lower ordinal. The rest of
// the row's K picks are -1.
//
// One row is all a decode step selects for, so the block alone sets the
// select's latency: its walks over the row go a chunk of THREADS * ITEMS
// ordinals at a time, each thread issuing its ITEMS loads before using any,
// so a pass over V4.1's 8192 pooled rows is one round of loads rather than
// 32 dependent ones. The histograms read a chunk coalesced (item i of thread
// t is ordinal t + i * THREADS); the picks read it in contiguous shares
// (thread t's ITEMS ordinals from t * ITEMS), which a scan of the shares'
// pick counts places, carried from chunk to chunk, and which each thread
// emits in ordinal order as their key-plane rows (its share's span found by
// bisection of the offsets, then advanced span by span).
//
// The block scans are `reduce::group_exclusive_add`. Shared memory: the bins
// [256], the scans' 32 warp sums and the select state [4]. Dynamic extents
// and strides read `seismic_words`.

#include "lib/core/order.cuh"
#include "lib/core/reduce.cuh"
#include "lib/indexer/indexer.cuh"

namespace topk {

typedef unsigned int u32;
typedef unsigned long long u64;

constexpr u32 THREADS = 1024;
constexpr u32 ITEMS = 8;
constexpr u32 CHUNK = THREADS * ITEMS;

} // namespace topk

extern "C" __global__ void __launch_bounds__(topk::THREADS) rows_topk(SEISMIC_KERNEL_PARAMS) {
    using namespace topk;
    extern __shared__ u32 shared[];
    u32 *bins = shared;          // [256]
    u32 *totals = shared + 256;  // [32]: the scans' warp sums
    u32 *state = totals + 32;    // prefix, remaining
    const u64 row = blockIdx.x;
    const u32 thread = threadIdx.x;
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
            if (thread < 256) bins[thread] = 0;
            __syncthreads();
            const u32 prefix_bits = state[0];
            const u32 remaining = state[1];
            const u32 shift = static_cast<u32>(digit) * 8u;
            // One shared atomic per counted ordinal.
            for (u32 base = 0; base < scored; base += CHUNK) {
                u32 keys[ITEMS];
#pragma unroll
                for (u32 i = 0; i < ITEMS; ++i) {
                    const u32 ordinal = base + i * THREADS + thread;
                    keys[i] = ordinal < scored ? key(ordinal) : 0u;
                }
#pragma unroll
                for (u32 i = 0; i < ITEMS; ++i) {
                    const u32 ordinal = base + i * THREADS + thread;
                    if (ordinal < scored && (keys[i] & mask) == prefix_bits) atomicAdd(&bins[(keys[i] >> shift) & 255u], 1u);
                }
            }
            __syncthreads();
            // Thread t < 256 holds bin 255 - t: the candidates above it are
            // the scan of the bins before it. The bin holding the
            // `remaining`-th best takes the digit.
            const u32 bin = 255u - thread;
            const u32 count = thread < 256 ? bins[bin] : 0u;
            const u32 above = reduce::group_exclusive_add(count, totals, total);
            const bool pick = thread < 256 && above < remaining && remaining <= above + count;
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

    // The picks, a chunk at a time: thread t's share of the chunk [lo, hi);
    // `ranked` equal ones and `placed` picks before the chunk.
    u32 ranked = 0;
    u32 placed = 0;
    for (u32 base = 0; base < scored; base += CHUNK) {
        const u32 lo = min(base + thread * ITEMS, scored);
        const u32 hi = min(lo + ITEMS, scored);
        u32 keys[ITEMS];
#pragma unroll
        for (u32 i = 0; i < ITEMS; ++i) keys[i] = !prefix && lo + i < hi ? key(lo + i) : 0u;
        // The rank among the ordinals equal to T of the share's first.
        u32 equal = 0;
#pragma unroll
        for (u32 i = 0; i < ITEMS; ++i) equal += !prefix && lo + i < hi && keys[i] == threshold ? 1u : 0u;
        u32 chunk_equal = 0;
        u32 rank = ranked + reduce::group_exclusive_add(equal, totals, chunk_equal);
        ranked += chunk_equal;
        bool picked[ITEMS];
        u32 picks = 0;
#pragma unroll
        for (u32 i = 0; i < ITEMS; ++i) {
            const u32 ordinal = lo + i;
            picked[i] = ordinal < hi &&
                        (prefix ? ordinal < limit
                                : keys[i] > threshold || (keys[i] == threshold && rank++ < take));
            picks += picked[i] ? 1u : 0u;
        }
        u32 chunk_picks = 0;
        u32 position = placed + reduce::group_exclusive_add(picks, totals, chunk_picks);
        placed += chunk_picks;
        if (picks > 0) {
            // The span holding `lo`: the last whose offset is at most it.
            u32 span = 0;
            for (u32 high = r - 1; span < high;) {
                const u32 middle = (span + high + 1) / 2;
                if (span_first[middle] <= lo)
                    span = middle;
                else
                    high = middle - 1;
            }
#pragma unroll
            for (u32 i = 0; i < ITEMS; ++i) {
                const u32 ordinal = lo + i;
                if (ordinal >= hi) continue;
                while (ordinal >= span_first[span + 1]) ++span;
                if (picked[i] && position < k)
                    out[position++ * out_stride] = row_spans[span * span_stride] + static_cast<int>(ordinal - span_first[span]);
            }
        }
    }
    for (u32 pick = placed + thread; pick < k; pick += THREADS) out[pick * out_stride] = -1;
}
