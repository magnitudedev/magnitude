// rows_topk on Metal (contract and portable body in indexer.seismic).
//
// A threadgroup of THREADS (1024) threads per row (matching CUDA: ITEMS = 8,
// CHUNK = 8192). The row's span offsets (the listed rows before each span,
// then their count) are scanned into scratch; its scored ordinals are the
// first min(N, listed). A row that reads a prefix (dense, complete <= K, or
// with at most K scored ordinals) picks the first min(K, scored). A selecting
// row finds its K-th best score T by a radix select over the scores' order
// keys (`order::key`: a < b exactly when their keys are, -0.0 and 0.0 one
// key), a byte a pass from the top: each pass's histogram takes one atomic per
// distinct digit of a simdgroup, and the digit is found by a scan of the bins.
// It picks every ordinal above T and the first equal ones in ordinal order
// (ranked by a scan of each thread's count of equal ones): the K best, ties to
// the lower ordinal. The rest of the row's K picks are -1.
//
// When `scored <= CHUNK` (8192: covering DeepSeek V4.1's 8192 pooled rows and
// all shorter contexts), each thread loads its contiguous share of ITEMS (8)
// keys ONCE (vectorized `packed_float4`) into registers and reuses them across
// all 4 radix passes and the final emit pass (1 global pass instead of 7).
// When `scored > CHUNK`, the radix passes walk coalesced strided chunks of
// CHUNK ordinals and the emit pass walks contiguous chunks with vectorized
// loads, carrying `ranked` and `placed` across chunks.
//
// Threadgroup memory: the bins [256], the scans' 32 simdgroup sums and the
// select state [4]. Dynamic extents read `seismic_words`.

#include "lib/core/order.h"
#include "lib/indexer/indexer.h"

namespace topk {

constant constexpr uint THREADS = 1024;
constant constexpr uint ITEMS = 8;
constant constexpr uint CHUNK = THREADS * ITEMS;

// Exclusive prefix sum of `value` over the 1024 threads (32 simdgroups of 32
// lanes) of a threadgroup in thread order, and in `total` their sum. Every
// thread calls it; `totals` holds the 32 simdgroups' sums.
inline uint group_exclusive_1024(uint value, threadgroup uint *totals, uint simd, uint lane, thread uint &total) {
    const uint inclusive = simd_prefix_inclusive_sum(value);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 31)
        totals[simd] = inclusive;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint sg_sum = totals[lane];
    const uint sg_before = simd_prefix_exclusive_sum(sg_sum);
    total = simd_sum(sg_sum);
    const uint before = simd_shuffle(sg_before, ushort(simd));
    return before + inclusive - value;
}

// One threadgroup atomic per distinct digit of a simdgroup.
inline void simd_histogram_add(threadgroup uint *bins, uint key, uint shift, bool pending, uint lane) {
    const uint value = (key >> shift) & 255u;
    uint waiting = uint(uint64_t(simd_ballot(pending)));
    while (waiting != 0u) {
        const uint leader = metal::ctz(waiting);
        const uint leading = simd_shuffle(value, ushort(leader));
        const uint same = uint(uint64_t(simd_ballot(pending && value == leading)));
        if (lane == leader)
            atomic_fetch_add_explicit(reinterpret_cast<threadgroup atomic_uint *>(&bins[leading]),
                metal::popcount(same), memory_order_relaxed);
        pending = pending && value != leading;
        waiting &= ~same;
    }
}

inline void load_contiguous_keys(device const float *row_scores, uint lo, uint scored, thread uint *keys) {
    if (lo < scored && scored - lo >= ITEMS) {
        const float4 v0 = float4(*reinterpret_cast<device const packed_float4 *>(row_scores + lo));
        const float4 v1 = float4(*reinterpret_cast<device const packed_float4 *>(row_scores + lo + 4));
        keys[0] = order::key(v0.x);
        keys[1] = order::key(v0.y);
        keys[2] = order::key(v0.z);
        keys[3] = order::key(v0.w);
        keys[4] = order::key(v1.x);
        keys[5] = order::key(v1.y);
        keys[6] = order::key(v1.z);
        keys[7] = order::key(v1.w);
    } else {
        for (uint i = 0; i < ITEMS; ++i)
            keys[i] = lo + i < scored ? order::key(row_scores[lo + i]) : 0u;
    }
}

} // namespace topk

kernel void rows_topk(
    device const float *scores [[buffer(SEISMIC_BUFFER_SCORES)]],
    device const int *complete [[buffer(SEISMIC_BUFFER_COMPLETE)]],
    device const int *spans [[buffer(SEISMIC_BUFFER_SPANS)]],
    device int *selected [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    device uint *offsets [[buffer(SEISMIC_BUFFER_SCRATCH_SPANS)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    threadgroup uint *shared [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    using namespace topk;
    const ulong row = group.x;
    const uint n = uint(SEISMIC_DIM_N);
    const uint k = uint(SEISMIC_DIM_K);
    const uint r = uint(SEISMIC_DIM_R);
    if (k == 0)
        return;
    device const int *row_spans = spans + row * r * 2;
    device const float *row_scores = scores + row * ulong(n);
    device int *out = selected + row * ulong(k);
    threadgroup uint *bins = shared;          // [256]
    threadgroup uint *totals = shared + 256;  // [32]: the scans' simdgroup sums
    threadgroup uint *state = totals + 32;    // prefix, remaining
    uint total = 0;

    // The span offsets [R + 1]: the listed rows before each span, then the
    // row's.
    device uint *span_first = offsets + row * ulong(r + 1);
    uint listed = 0;
    for (uint base = 0; base < r; base += THREADS) {
        const uint span = base + thread_index;
        const uint length = span < r ? indexer::span_length(row_spans[span * 2], row_spans[span * 2 + 1]) : 0u;
        uint chunk = 0;
        const uint before = group_exclusive_1024(length, totals, simd, lane, chunk);
        if (span < r)
            span_first[span] = listed + before;
        listed += chunk;
    }
    if (thread_index == 0)
        span_first[r] = listed;
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    const uint scored = metal::min(listed, n);

    // A prefix row picks its first `limit` ordinals; a selecting row those
    // above `threshold` and the first `take` equal to it.
    const bool prefix = complete[row] <= int(k) || scored <= k;
    const uint limit = metal::min(scored, k);
    const bool single_chunk = scored <= CHUNK;
    uint keys[ITEMS];
    if (single_chunk && !prefix)
        load_contiguous_keys(row_scores, thread_index * ITEMS, scored, keys);
    uint threshold = 0;
    uint take = 0;
    if (!prefix) {
        if (thread_index == 0) {
            state[0] = 0;
            state[1] = k;
        }
        if (thread_index < 256)
            bins[thread_index] = 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint mask = 0;
        for (int digit = 3; digit >= 0; --digit) {
            const uint prefix_bits = state[0];
            const uint remaining = state[1];
            const uint shift = uint(digit) * 8u;
            if (single_chunk) {
                const uint lo = thread_index * ITEMS;
                for (uint i = 0; i < ITEMS; ++i) {
                    const bool pending = lo + i < scored && (keys[i] & mask) == prefix_bits;
                    simd_histogram_add(bins, keys[i], shift, pending, lane);
                }
            } else {
                for (uint base = 0; base < scored; base += CHUNK) {
                    for (uint i = 0; i < ITEMS; ++i) {
                        const uint ordinal = base + i * THREADS + thread_index;
                        keys[i] = ordinal < scored ? order::key(row_scores[ordinal]) : 0u;
                    }
                    for (uint i = 0; i < ITEMS; ++i) {
                        const uint ordinal = base + i * THREADS + thread_index;
                        const bool pending = ordinal < scored && (keys[i] & mask) == prefix_bits;
                        simd_histogram_add(bins, keys[i], shift, pending, lane);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // Thread t < 256 holds bin 255 - t: the candidates above it are
            // the scan of the bins before it. The bin holding the
            // `remaining`-th best takes the digit.
            const uint bin = thread_index < 256 ? 255u - thread_index : 0u;
            const uint count = thread_index < 256 ? bins[bin] : 0u;
            const uint above = group_exclusive_1024(count, totals, simd, lane, total);
            if (thread_index < 256 && above < remaining && remaining <= above + count) {
                state[0] = prefix_bits | (bin << shift);
                state[1] = remaining - above;
            }
            if (digit > 0 && thread_index < 256)
                bins[thread_index] = 0;
            mask |= 255u << shift;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        threshold = state[0];
        take = state[1];
    }

    // The picks, a chunk at a time: thread t's share of the chunk [lo, hi);
    // `ranked` equal ones and `placed` picks before the chunk.
    uint ranked = 0;
    uint placed = 0;
    for (uint base = 0; base < scored; base += CHUNK) {
        const uint lo = metal::min(base + thread_index * ITEMS, scored);
        const uint hi = metal::min(lo + ITEMS, scored);
        if (!prefix && !single_chunk)
            load_contiguous_keys(row_scores, base + thread_index * ITEMS, scored, keys);
        // The rank among the ordinals equal to T of the share's first.
        uint rank = 0;
        if (!prefix) {
            uint equal = 0;
            for (uint i = 0; i < ITEMS; ++i)
                equal += lo + i < hi && keys[i] == threshold ? 1u : 0u;
            uint chunk_equal = 0;
            rank = ranked + group_exclusive_1024(equal, totals, simd, lane, chunk_equal);
            ranked += chunk_equal;
        }
        uint picked_mask = 0;
        uint picks = 0;
        for (uint i = 0; i < ITEMS; ++i) {
            const uint ordinal = lo + i;
            const bool picked = ordinal < hi &&
                                (prefix ? ordinal < limit
                                        : keys[i] > threshold || (keys[i] == threshold && rank++ < take));
            if (picked) {
                picked_mask |= 1u << i;
                ++picks;
            }
        }
        uint chunk_picks = 0;
        uint position = placed + group_exclusive_1024(picks, totals, simd, lane, chunk_picks);
        placed += chunk_picks;
        if (picks > 0) {
            // The span holding `lo`: the last whose offset is at most it.
            uint span = 0;
            for (uint high = r - 1; span < high;) {
                const uint middle = (span + high + 1) / 2;
                if (span_first[middle] <= lo)
                    span = middle;
                else
                    high = middle - 1;
            }
            for (uint i = 0; i < ITEMS; ++i) {
                const uint ordinal = lo + i;
                if (ordinal >= hi)
                    break;
                while (ordinal >= span_first[span + 1])
                    ++span;
                if (((picked_mask >> i) & 1u) != 0u && position < k)
                    out[position++] = row_spans[span * 2] + int(ordinal - span_first[span]);
            }
        }
    }
    for (uint pick = placed + thread_index; pick < k; pick += THREADS)
        out[pick] = -1;
}
