// rows_topk on Metal (contract and portable body in indexer.seismic).
//
// A threadgroup of 256 threads per row. The row's span offsets (the listed
// rows before each span, then their count) are scanned into scratch; its
// scored ordinals are the first min(N, listed). A row that reads a prefix
// (dense, complete <= K, or with at most K scored ordinals) picks the first
// min(K, scored). A selecting row finds its K-th best score T by a radix
// select over the scores' order keys (`ordered_key`: a < b exactly when
// their keys are, -0.0 and 0.0 one key), a byte a pass from the top: each
// pass's histogram takes one atomic per distinct digit of a simdgroup, and
// the digit is found by a scan of the bins. It picks every ordinal above T
// and the first equal ones in ordinal order (ranked by a scan of each
// thread's count of equal ones): the K best, ties to the lower ordinal.
// Thread t owns a contiguous share of the ordinals; a scan of the shares'
// pick counts places its picks, which it emits in ordinal order as their
// key-plane rows (its first span found by bisection of the offsets, then
// advanced span by span). The rest of the row's K picks are -1.
//
// Threadgroup memory: the bins [256], the scans' 8 simdgroup sums and the
// select state [4]. Dynamic extents read `seismic_words`.

#include "lib/core/order.h"
#include "lib/indexer/indexer.h"

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
    constexpr uint THREADS = 256;
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
    threadgroup uint *totals = shared + 256;  // [8]: the scans' simdgroup sums
    threadgroup uint *state = totals + 8;     // prefix, remaining
    uint total = 0;

    // The span offsets [R + 1]: the listed rows before each span, then the
    // row's.
    device uint *span_first = offsets + row * ulong(r + 1);
    uint listed = 0;
    for (uint base = 0; base < r; base += THREADS) {
        const uint span = base + thread_index;
        const uint length = span < r ? indexer::span_length(row_spans[span * 2], row_spans[span * 2 + 1]) : 0u;
        uint chunk = 0;
        const uint before = indexer::group_exclusive(length, totals, simd, lane, chunk);
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
    uint threshold = 0;
    uint take = 0;
    if (!prefix) {
        if (thread_index == 0) {
            state[0] = 0;
            state[1] = k;
        }
        uint mask = 0;
        for (int digit = 3; digit >= 0; --digit) {
            bins[thread_index] = 0;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint prefix_bits = state[0];
            const uint remaining = state[1];
            const uint shift = uint(digit) * 8;
            // One atomic per distinct digit of a simdgroup: its lanes holding
            // the first pending lane's digit add their count together.
            for (uint base = 0; base < scored; base += THREADS) {
                const uint ordinal = base + thread_index;
                const uint u = ordinal < scored ? order::key(row_scores[ordinal]) : 0u;
                const uint value = (u >> shift) & 255u;
                bool pending = ordinal < scored && (u & mask) == prefix_bits;
                for (uint waiting = uint(uint64_t(simd_ballot(pending))); waiting != 0u;
                     waiting = uint(uint64_t(simd_ballot(pending)))) {
                    const uint leader = metal::ctz(waiting);
                    const uint leading = simd_shuffle(value, ushort(leader));
                    const bool same = pending && value == leading;
                    const uint count = metal::popcount(uint(uint64_t(simd_ballot(same))));
                    if (lane == leader)
                        atomic_fetch_add_explicit(reinterpret_cast<threadgroup atomic_uint *>(&bins[leading]), count,
                            memory_order_relaxed);
                    pending = pending && !same;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // Thread t holds bin 255 - t: the candidates above it are the
            // scan of the bins before it. The bin holding the `remaining`-th
            // best takes the digit.
            const uint bin = 255 - thread_index;
            const uint count = bins[bin];
            const uint above = indexer::group_exclusive(count, totals, simd, lane, total);
            const bool pick = above < remaining && remaining <= above + count;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (pick) {
                state[0] = prefix_bits | (bin << shift);
                state[1] = remaining - above;
            }
            mask |= 255u << shift;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        threshold = state[0];
        take = state[1];
    }

    // Thread t's share of the scored ordinals [lo, hi).
    const uint share = (scored + THREADS - 1) / THREADS;
    const uint lo = metal::min(thread_index * share, scored);
    const uint hi = metal::min(lo + share, scored);
    // The rank among the ordinals equal to T of the share's first.
    uint equal = 0;
    if (!prefix)
        for (uint ordinal = lo; ordinal < hi; ++ordinal)
            equal += order::key(row_scores[ordinal]) == threshold ? 1u : 0u;
    const uint first_rank = indexer::group_exclusive(equal, totals, simd, lane, total);
    // Whether `ordinal` is picked, its equal ones ranked from `rank`.
    auto picked = [&](uint ordinal, thread uint &rank) {
        if (prefix)
            return ordinal < limit;
        const uint u = order::key(row_scores[ordinal]);
        if (u > threshold)
            return true;
        if (u != threshold)
            return false;
        return rank++ < take;
    };
    uint picks = 0;
    uint rank = first_rank;
    for (uint ordinal = lo; ordinal < hi; ++ordinal)
        picks += picked(ordinal, rank) ? 1u : 0u;
    uint placed = 0;
    uint position = indexer::group_exclusive(picks, totals, simd, lane, placed);
    if (lo < hi) {
        // The span holding `lo`: the last whose offset is at most it.
        uint span = 0;
        for (uint high = r - 1; span < high;) {
            const uint middle = (span + high + 1) / 2;
            if (span_first[middle] <= lo)
                span = middle;
            else
                high = middle - 1;
        }
        rank = first_rank;
        for (uint ordinal = lo; ordinal < hi; ++ordinal) {
            while (ordinal >= span_first[span + 1])
                ++span;
            if (picked(ordinal, rank) && position < k)
                out[position++] = row_spans[span * 2] + int(ordinal - span_first[span]);
        }
    }
    for (uint pick = placed + thread_index; pick < k; pick += THREADS)
        out[pick] = -1;
}
