// rows_topk on CUDA (contract and portable body in indexer.seismic):
//
// A block of THREADS (1024) threads per row. The row's span offsets (the listed
// rows before each span, then their count) are scanned into scratch; its
// scored ordinals are the first min(N, listed). A row that reads a prefix
// (dense, complete <= K, or with at most K scored ordinals) picks the first
// min(K, scored). A selecting row finds its K-th best score T by a radix
// select over the scores' order keys (`order::key`: a < b exactly when
// their keys are, -0.0 and 0.0 one key), a byte a pass from the top: each
// pass's histogram aggregates counts per distinct digit within each warp via
// `__match_any_sync` (one shared atomic per distinct digit per warp), and the
// digit is found by a scan of the bins. It picks every ordinal above T and
// the first equal ones in ordinal order (ranked by a scan of each thread's
// count of equal ones): the K best, ties to the lower ordinal. The rest of
// the row's K picks are -1.
//
// When `scored <= CHUNK` (8192: covering DeepSeek V4.1's 8192 pooled rows and
// all shorter contexts), each thread loads its contiguous share of ITEMS (8)
// keys ONCE (vectorized `float4` when unit-stride and 16-byte aligned) into
// registers and reuses them across all 4 radix passes and the final emit pass
// (1 global pass instead of 5). When `scored > CHUNK`, the radix passes walk
// coalesced strided chunks of CHUNK ordinals and the emit pass walks
// contiguous chunks with vectorized loads, carrying `ranked` and `placed`
// across chunks.
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

__device__ __forceinline__ void warp_histogram_add(u32 *bins, u32 key, u32 shift, bool match, int lane) {
    const u32 active = __ballot_sync(0xffffffffu, match);
    if (match) {
        const u32 bin = (key >> shift) & 255u;
        const u32 peers = __match_any_sync(active, bin);
        if (__ffs(peers) - 1 == lane) atomicAdd(&bins[bin], static_cast<u32>(__popc(peers)));
    }
}

__device__ __forceinline__ void load_contiguous_keys(const float *row_scores, u64 score_stride, bool vec4, u32 lo,
                                                     u32 scored, u32 *keys) {
    if (vec4 && lo < scored && scored - lo >= ITEMS) {
        const float4 v0 = *reinterpret_cast<const float4 *>(row_scores + lo);
        const float4 v1 = *reinterpret_cast<const float4 *>(row_scores + lo + 4);
        keys[0] = order::key(v0.x);
        keys[1] = order::key(v0.y);
        keys[2] = order::key(v0.z);
        keys[3] = order::key(v0.w);
        keys[4] = order::key(v1.x);
        keys[5] = order::key(v1.y);
        keys[6] = order::key(v1.z);
        keys[7] = order::key(v1.w);
    } else {
#pragma unroll
        for (u32 i = 0; i < ITEMS; ++i)
            keys[i] = lo + i < scored ? order::key(row_scores[(lo + i) * score_stride]) : 0u;
    }
}

} // namespace topk

extern "C" __global__ void __launch_bounds__(topk::THREADS) rows_topk(SEISMIC_KERNEL_PARAMS) {
    using namespace topk;
    extern __shared__ u32 shared[];
    u32 *bins = shared;          // [256]
    u32 *totals = shared + 256;  // [32]: the scans' warp sums
    u32 *state = totals + 32;    // prefix, remaining
    const u64 row = blockIdx.x;
    const u32 thread = threadIdx.x;
    const int lane = static_cast<int>(thread & 31u);
    const u32 n = static_cast<u32>(SEISMIC_DIM_N);
    const u32 k = static_cast<u32>(SEISMIC_DIM_K);
    const u32 r = static_cast<u32>(SEISMIC_DIM_R);
    if (k == 0) return;
    const int *row_spans = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_SPANS)) + row * SEISMIC_SPANS_STRIDE_0;
    const u64 span_stride = SEISMIC_SPANS_STRIDE_1;
    const u64 bound_stride = SEISMIC_SPANS_STRIDE_2;
    const float *row_scores = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCORES)) + row * SEISMIC_SCORES_STRIDE_0;
    const u64 score_stride = SEISMIC_SCORES_STRIDE_1;
    const bool vec4 = score_stride == 1 && (reinterpret_cast<u64>(row_scores) & 15ull) == 0ull;
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
    const bool single_chunk = scored <= CHUNK;
    u32 keys[ITEMS];
    if (single_chunk && !prefix) load_contiguous_keys(row_scores, score_stride, vec4, thread * ITEMS, scored, keys);
    u32 threshold = 0;
    u32 take = 0;
    if (!prefix) {
        if (thread == 0) {
            state[0] = 0;
            state[1] = k;
        }
        if (thread < 256) bins[thread] = 0;
        __syncthreads();
        u32 mask = 0;
        for (int digit = 3; digit >= 0; --digit) {
            const u32 prefix_bits = state[0];
            const u32 remaining = state[1];
            const u32 shift = static_cast<u32>(digit) * 8u;
            if (single_chunk) {
                const u32 lo = thread * ITEMS;
#pragma unroll
                for (u32 i = 0; i < ITEMS; ++i) {
                    const bool match = lo + i < scored && (keys[i] & mask) == prefix_bits;
                    warp_histogram_add(bins, keys[i], shift, match, lane);
                }
            } else {
                for (u32 base = 0; base < scored; base += CHUNK) {
#pragma unroll
                    for (u32 i = 0; i < ITEMS; ++i) {
                        const u32 ordinal = base + i * THREADS + thread;
                        keys[i] = ordinal < scored ? key(ordinal) : 0u;
                    }
#pragma unroll
                    for (u32 i = 0; i < ITEMS; ++i) {
                        const u32 ordinal = base + i * THREADS + thread;
                        const bool match = ordinal < scored && (keys[i] & mask) == prefix_bits;
                        warp_histogram_add(bins, keys[i], shift, match, lane);
                    }
                }
            }
            __syncthreads();
            // Thread t < 256 holds bin 255 - t: the candidates above it are
            // the scan of the bins before it. The bin holding the
            // `remaining`-th best takes the digit.
            const u32 bin = thread < 256 ? 255u - thread : 0u;
            const u32 count = thread < 256 ? bins[bin] : 0u;
            const u32 above = reduce::group_exclusive_add(count, totals, total);
            if (thread < 256 && above < remaining && remaining <= above + count) {
                state[0] = prefix_bits | (bin << shift);
                state[1] = remaining - above;
            }
            if (digit > 0 && thread < 256) bins[thread] = 0;
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
        if (!prefix && !single_chunk)
            load_contiguous_keys(row_scores, score_stride, vec4, base + thread * ITEMS, scored, keys);
        // The rank among the ordinals equal to T of the share's first.
        u32 rank = 0;
        if (!prefix) {
            u32 equal = 0;
#pragma unroll
            for (u32 i = 0; i < ITEMS; ++i) equal += lo + i < hi && keys[i] == threshold ? 1u : 0u;
            u32 chunk_equal = 0;
            rank = ranked + reduce::group_exclusive_add(equal, totals, chunk_equal);
            ranked += chunk_equal;
        }
        u32 picked_mask = 0;
        u32 picks = 0;
#pragma unroll
        for (u32 i = 0; i < ITEMS; ++i) {
            const u32 ordinal = lo + i;
            const bool picked = ordinal < hi &&
                                (prefix ? ordinal < limit
                                        : keys[i] > threshold || (keys[i] == threshold && rank++ < take));
            if (picked) {
                picked_mask |= 1u << i;
                ++picks;
            }
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
                if (ordinal >= hi) break;
                while (ordinal >= span_first[span + 1]) ++span;
                if (((picked_mask >> i) & 1u) != 0u && position < k)
                    out[position++ * out_stride] = row_spans[span * span_stride] + static_cast<int>(ordinal - span_first[span]);
            }
        }
    }
    for (u32 pick = placed + thread; pick < k; pick += THREADS) out[pick * out_stride] = -1;
}
