// candidate_rows on CUDA (contract and portable body in indexer.seismic), as
// Metal's (`metal/candidate_rows.metal`):
//
// A block of 256 threads per row; no thread walks the row's picks or spans
// alone. The picks are `rows_topk`'s: ascending, then -1, so the first
// `valid` are the picked blocks. With the fill F = min(K, c) - pooled (0 when
// the pool holds min(K, c) units), the fill takes the F lowest unpicked
// units, so the taken units are exactly [0, X) and the picked blocks, X the
// unit after the F-th unpicked one (0 without a fill). Then:
// - the span offsets are scanned into scratch (as `rows_topk`), and the
//   picks' count and pooled units summed;
// - X is placed by a scan of the gaps before each pick and after the last
//   (the thread whose gap holds the F-th unpicked unit);
// - the taken units are owned as ranges in unit order: [0, X) by thread 0,
//   then each picked block's units at or past X by its pick's thread (a
//   contiguous share of the picks). A range is preceded by a taken unit when
//   it starts at X > 0 or right after the previous pick's block, and
//   followed by one when the next pick's block starts at its end (for
//   [0, X): the pick whose block starts at X);
// - each range is cut where a list span starts; a piece opens an entry at a
//   span's start or a range start not preceded, and closes it at a span's
//   end or a range end not followed. A scan of each thread's opened entries
//   places them, so each entry's start and end are each stored once, by the
//   piece that opens or closes it. The rest of the row's B + R spans are
//   [-1, -1].
//
// The block scans are `reduce::group_exclusive_add`. Dynamic extents and
// strides read `seismic_words`.

#include "lib/core/reduce.cuh"
#include "lib/indexer/indexer.cuh"

extern "C" __global__ void candidate_rows(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned int u32;
    typedef unsigned long long u64;
    constexpr u32 THREADS = 256;
    __shared__ u32 totals[THREADS / 32];
    // X, and whether the unit X is taken.
    __shared__ u32 fill_end[2];
    const u64 row = blockIdx.x;
    const u32 thread = threadIdx.x;
    const u32 r = static_cast<u32>(SEISMIC_DIM_R);
    const u32 b = static_cast<u32>(SEISMIC_DIM_B);
    const u32 bu = static_cast<u32>(SEISMIC_DIM_BU);
    const u32 k = static_cast<u32>(SEISMIC_DIM_K);
    const u32 capacity = b + r;
    const int *row_spans = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_SPANS)) + row * SEISMIC_SPANS_STRIDE_0;
    const u64 span_stride = SEISMIC_SPANS_STRIDE_1;
    const u64 bound_stride = SEISMIC_SPANS_STRIDE_2;
    const int *picks = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_BLOCKS)) + row * SEISMIC_BLOCKS_STRIDE_0;
    const u64 pick_stride = SEISMIC_BLOCKS_STRIDE_1;
    int *out = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER)) + row * SEISMIC_RESULT_0_STRIDE_0;
    const u64 out_stride = SEISMIC_RESULT_0_STRIDE_1;
    const u64 out_bound = SEISMIC_RESULT_0_STRIDE_2;
    u32 total = 0;

    // The span offsets [R + 1]: the units before each span, then the row's.
    u32 *span_first = reinterpret_cast<u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_OFFSETS)) + row * (static_cast<u64>(r) + 1);
    u32 units = 0;
    for (u32 base = 0; base < r; base += THREADS) {
        const u32 span = base + thread;
        const u32 length =
            span < r ? indexer::span_length(row_spans[span * span_stride], row_spans[span * span_stride + bound_stride])
                     : 0u;
        u32 chunk = 0;
        const u32 before = reduce::group_exclusive_add(length, totals, chunk);
        if (span < r)
            span_first[span] = units + before;
        units += chunk;
    }
    if (thread == 0) {
        span_first[r] = units;
        fill_end[0] = 0;
        fill_end[1] = 0;
    }

    // Pick s's block, and its units [first_unit(s), end_unit(s)), clipped to
    // the row's.
    auto pick = [&](u32 s) { return picks[s * pick_stride]; };
    auto first_unit = [&](u32 s) { return min(units, static_cast<u32>(pick(s)) * bu); };
    auto end_unit = [&](u32 s) { return min(units, (static_cast<u32>(pick(s)) + 1) * bu); };
    u32 valid = 0;
    u32 pooled = 0;
    {
        u32 count = 0, sum = 0;
        for (u32 s = thread; s < b; s += THREADS)
            if (pick(s) >= 0) {
                ++count;
                sum += end_unit(s) - first_unit(s);
            }
        reduce::group_exclusive_add(count, totals, valid);
        reduce::group_exclusive_add(sum, totals, pooled);
    }
    const u32 wanted = min(k, units);
    const u32 fill = wanted > pooled ? wanted - pooled : 0u;

    // X: the gap of slot s (s <= valid) is the units between the previous
    // pick's block (or 0) and pick s's block (or, for s == valid, the row's
    // end); the F-th unpicked unit lies in the gap whose scan holds F.
    if (fill > 0) {
        const u32 slots = valid + 1;
        const u32 share = (slots + THREADS - 1) / THREADS;
        const u32 lo = min(thread * share, slots);
        const u32 hi = min(lo + share, slots);
        auto gap_first = [&](u32 s) { return s == 0 ? 0u : end_unit(s - 1); };
        auto gap_end = [&](u32 s) { return s == valid ? units : first_unit(s); };
        u32 gaps = 0;
        for (u32 s = lo; s < hi; ++s)
            gaps += gap_end(s) - gap_first(s);
        u32 before = reduce::group_exclusive_add(gaps, totals, total);
        for (u32 s = lo; s < hi; ++s) {
            const u32 first = gap_first(s), end = gap_end(s);
            if (before < fill && fill <= before + end - first) {
                const u32 x = first + (fill - before);
                fill_end[0] = x;
                fill_end[1] = x == end && s < valid && x < units ? 1u : 0u;
            }
            before += end - first;
        }
    }
    __syncthreads();
    const u32 x = fill_end[0];
    const bool x_followed = fill_end[1] != 0;

    // This thread's picks [lo, hi) and their ranges; thread 0 first owns
    // [0, X).
    const u32 share = (valid + THREADS - 1) / THREADS;
    const u32 lo = min(thread * share, valid);
    const u32 hi = min(lo + share, valid);
    // Each piece of the range [a, e) in unit order, cut where a list span
    // starts: it opens an entry at a span's start or a range start not
    // preceded, and closes it at a span's end or a range end not followed.
    // Counting, it adds the entries it opens to `entry`; writing, it stores
    // each opened entry's start at `entry` and each closed entry's end at
    // `entry - 1`, advancing `entry` past the ones it opens.
    auto pieces = [&](u32 a, u32 e, bool preceded, bool followed, bool write, u32 &entry) {
        // The span holding `a` (a < units): the last whose offset is at most
        // it, which holds units.
        u32 span = 0;
        for (u32 high = r - 1; span < high;) {
            const u32 middle = (span + high + 1) / 2;
            if (span_first[middle] <= a)
                span = middle;
            else
                high = middle - 1;
        }
        for (u32 at = a; at < e;) {
            while (at >= span_first[span + 1])
                ++span;
            const u32 offset = span_first[span];
            const u32 span_end = span_first[span + 1];
            const u32 stop = min(e, span_end);
            const bool opens = at == offset || (at == a && !preceded);
            const bool closes = stop == span_end || !followed;
            if (!write) {
                entry += opens ? 1u : 0u;
            } else {
                const int row_lo = row_spans[span * span_stride];
                if (opens) {
                    if (entry < capacity)
                        out[entry * out_stride] = row_lo + static_cast<int>(at - offset);
                    ++entry;
                }
                if (closes && entry - 1 < capacity)
                    out[(entry - 1) * out_stride + out_bound] = row_lo + static_cast<int>(stop - offset);
            }
            at = stop;
        }
    };
    // Every range of this thread in unit order, through `pieces`.
    auto ranges = [&](bool write, u32 &entry) {
        if (thread == 0 && x > 0)
            pieces(0, x, false, x_followed, write, entry);
        for (u32 s = lo; s < hi; ++s) {
            const u32 first = first_unit(s);
            const u32 a = max(first, x);
            const u32 e = end_unit(s);
            if (a >= e)
                continue;
            const bool preceded = (a == x && x > 0) || (s > 0 && pick(s - 1) + 1 == pick(s));
            const bool followed = e < units && s + 1 < valid && pick(s + 1) == pick(s) + 1;
            pieces(a, e, preceded, followed, write, entry);
        }
    };

    u32 opened = 0;
    ranges(false, opened);
    u32 count = 0;
    u32 entry = reduce::group_exclusive_add(opened, totals, count);
    ranges(true, entry);
    for (u32 pad = count + thread; pad < capacity; pad += THREADS) {
        out[pad * out_stride] = -1;
        out[pad * out_stride + out_bound] = -1;
    }
}
