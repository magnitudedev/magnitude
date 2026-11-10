// candidate_rows on Metal (contract and portable body in indexer.seismic).
//
// A threadgroup of 256 threads per row; no thread walks the row's picks or
// spans alone. The picks are `rows_topk`'s: ascending, then -1, so the
// first `valid` are the picked blocks. With the fill F = min(K, c) - pooled
// (0 when the pool holds min(K, c) units), the fill takes the F lowest
// unpicked units, so the taken units are exactly [0, X) and the picked
// blocks, X the unit after the F-th unpicked one (0 without a fill). Then:
// - the span offsets are scanned into scratch (as `rows_topk`), and the
//   picks' count and pooled units summed;
// - X is placed by a scan of the gaps before each pick and after the last
//   (the thread whose gap holds the F-th unpicked unit);
// - the taken units are owned as ranges in unit order: [0, X) by thread 0,
//   then each picked block's units at or past X by its pick's thread (a
//   contiguous share of the picks). A range is preceded by a taken unit
//   when it starts at X > 0 or right after the previous pick's block, and
//   followed by one when the next pick's block starts at its end (for
//   [0, X): the pick whose block starts at X);
// - each range is cut where a list span starts; a piece opens an entry at
//   a span's start or a range start not preceded, and closes it at a span's
//   end or a range end not followed. A scan of each thread's opened entries
//   places them, so each entry's start and end are each stored once, by the
//   piece that opens or closes it. The rest of the row's B + R spans are
//   [-1, -1].
//
// Dynamic extents read `seismic_words`.

#include "lib/indexer/indexer.h"

kernel void candidate_rows(
    device const int *blocks [[buffer(SEISMIC_BUFFER_BLOCKS)]],
    device const int *spans [[buffer(SEISMIC_BUFFER_SPANS)]],
    device int *pool [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    device uint *offsets [[buffer(SEISMIC_BUFFER_SCRATCH_OFFSETS)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint THREADS = 256;
    threadgroup uint totals[8];
    // X, and whether the unit X is taken.
    threadgroup uint fill_end[2];
    const ulong row = group.x;
    const uint r = uint(SEISMIC_DIM_R);
    const uint b = uint(SEISMIC_DIM_B);
    const uint bu = uint(SEISMIC_DIM_BU);
    const uint k = uint(SEISMIC_DIM_K);
    const uint capacity = b + r;
    device const int *row_spans = spans + row * r * 2;
    device const int *picks = blocks + row * b;
    device int *out = pool + row * ulong(capacity) * 2;
    uint total = 0;

    // The span offsets [R + 1]: the units before each span, then the row's.
    device uint *span_first = offsets + row * ulong(r + 1);
    uint units = 0;
    for (uint base = 0; base < r; base += THREADS) {
        const uint span = base + thread_index;
        const uint length = span < r ? indexer::span_length(row_spans[span * 2], row_spans[span * 2 + 1]) : 0u;
        uint chunk = 0;
        const uint before = indexer::group_exclusive(length, totals, simd, lane, chunk);
        if (span < r)
            span_first[span] = units + before;
        units += chunk;
    }
    if (thread_index == 0) {
        span_first[r] = units;
        fill_end[0] = 0;
        fill_end[1] = 0;
    }

    // Pick s's block's units [first_unit(s), end_unit(s)), clipped to the row's.
    auto first_unit = [&](uint s) { return metal::min(units, uint(picks[s]) * bu); };
    auto end_unit = [&](uint s) { return metal::min(units, (uint(picks[s]) + 1) * bu); };
    uint valid = 0;
    uint pooled = 0;
    {
        uint count = 0, sum = 0;
        for (uint s = thread_index; s < b; s += THREADS)
            if (picks[s] >= 0) {
                ++count;
                sum += end_unit(s) - first_unit(s);
            }
        indexer::group_exclusive(count, totals, simd, lane, valid);
        indexer::group_exclusive(sum, totals, simd, lane, pooled);
    }
    const uint wanted = metal::min(k, units);
    const uint fill = wanted > pooled ? wanted - pooled : 0u;

    // X: the gap of slot s (s <= valid) is the units between the previous
    // pick's block (or 0) and pick s's block (or, for s == valid, the row's
    // end); the F-th unpicked unit lies in the gap whose scan holds F.
    if (fill > 0) {
        const uint slots = valid + 1;
        const uint share = (slots + THREADS - 1) / THREADS;
        const uint lo = metal::min(thread_index * share, slots);
        const uint hi = metal::min(lo + share, slots);
        auto gap_first = [&](uint s) { return s == 0 ? 0u : end_unit(s - 1); };
        auto gap_end = [&](uint s) { return s == valid ? units : first_unit(s); };
        uint gaps = 0;
        for (uint s = lo; s < hi; ++s)
            gaps += gap_end(s) - gap_first(s);
        uint before = indexer::group_exclusive(gaps, totals, simd, lane, total);
        for (uint s = lo; s < hi; ++s) {
            const uint first = gap_first(s), end = gap_end(s);
            if (before < fill && fill <= before + end - first) {
                const uint x = first + (fill - before);
                fill_end[0] = x;
                fill_end[1] = x == end && s < valid && x < units ? 1u : 0u;
            }
            before += end - first;
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    const uint x = fill_end[0];
    const bool x_followed = fill_end[1] != 0;

    // This thread's picks [lo, hi) and their ranges; thread 0 first owns
    // [0, X).
    const uint share = (valid + THREADS - 1) / THREADS;
    const uint lo = metal::min(thread_index * share, valid);
    const uint hi = metal::min(lo + share, valid);
    // Visits each piece of the range [a, e) in unit order: its units [at,
    // stop), its span, and whether it opens and closes an entry.
    auto pieces = [&](uint a, uint e, bool preceded, bool followed, auto visit) {
        // The span holding `a` (a < units): the last whose offset is at most
        // it, which holds units.
        uint span = 0;
        for (uint high = r - 1; span < high;) {
            const uint middle = (span + high + 1) / 2;
            if (span_first[middle] <= a)
                span = middle;
            else
                high = middle - 1;
        }
        for (uint at = a; at < e;) {
            while (at >= span_first[span + 1])
                ++span;
            const uint span_end = span_first[span + 1];
            const uint stop = metal::min(e, span_end);
            const bool opens = at == span_first[span] || (at == a && !preceded);
            const bool closes = stop == span_end || !followed;
            visit(at, stop, span, opens, closes);
            at = stop;
        }
    };
    // Visits every range of this thread in unit order.
    auto ranges = [&](auto visit) {
        if (thread_index == 0 && x > 0)
            pieces(0, x, false, x_followed, visit);
        for (uint s = lo; s < hi; ++s) {
            const uint first = first_unit(s);
            const uint a = metal::max(first, x);
            const uint e = end_unit(s);
            if (a >= e)
                continue;
            const bool preceded = (a == x && x > 0) || (s > 0 && picks[s - 1] + 1 == picks[s]);
            const bool followed = e < units && s + 1 < valid && picks[s + 1] == picks[s] + 1;
            pieces(a, e, preceded, followed, visit);
        }
    };

    uint opened = 0;
    ranges([&](uint, uint, uint, bool opens, bool) { opened += opens ? 1u : 0u; });
    uint count = 0;
    uint entry = indexer::group_exclusive(opened, totals, simd, lane, count);
    ranges([&](uint at, uint stop, uint span, bool opens, bool closes) {
        const int row_lo = row_spans[span * 2];
        const uint offset = span_first[span];
        if (opens) {
            if (entry < capacity)
                out[entry * 2] = row_lo + int(at - offset);
            ++entry;
        }
        if (closes && entry - 1 < capacity)
            out[(entry - 1) * 2 + 1] = row_lo + int(stop - offset);
    });
    for (uint pad = count + thread_index; pad < capacity; pad += THREADS) {
        out[pad * 2] = -1;
        out[pad * 2 + 1] = -1;
    }
}
