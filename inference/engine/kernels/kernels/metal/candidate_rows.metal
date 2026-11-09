// candidate_rows on Metal (contract and portable body in indexer.seismic).
//
// A thread per row, as the CPU native: its taken units form ascending runs
// (before each picked block, ascending and clipped to the row's units, the
// fill takes the gap's lowest units while it lasts, then the block's own;
// after the last, the rest of the fill). Each run is mapped through the
// row's spans, split where a span ends, and adjacent runs within one span
// join. The rest of the row's B + R spans are [-1, -1].
//
// Dynamic extents read `seismic_words`.

namespace candidate {

// A row's taken units as plane-row spans: the pending run (joined with an
// adjacent next), mapped through the row's list spans on flush.
struct Pool {
    device const int *list;
    uint spans;
    device int *out;
    uint capacity;
    uint count;
    // The list span the next run starts in, and the units before it.
    uint span;
    uint offset;
    uint start;
    uint end;

    static uint length(int lo, int hi) {
        return lo >= 0 && hi > lo ? uint(hi - lo) : 0u;
    }

    void flush() {
        uint at = start;
        while (at < end && span < spans) {
            const int lo = list[span * 2];
            const uint size = length(lo, list[span * 2 + 1]);
            if (offset + size <= at) {
                offset += size;
                ++span;
                continue;
            }
            const uint stop = metal::min(end, offset + size);
            if (count < capacity) {
                out[count * 2] = lo + int(at - offset);
                out[count * 2 + 1] = lo + int(stop - offset);
            }
            ++count;
            at = stop;
        }
        start = end = 0;
    }

    // Takes the units [first, last), ascending after every earlier run.
    void take(uint first, uint last) {
        if (first == last)
            return;
        if (end > start && end == first) {
            end = last;
            return;
        }
        flush();
        start = first;
        end = last;
    }
};

} // namespace candidate

kernel void candidate_rows(
    device const int *blocks [[buffer(SEISMIC_BUFFER_BLOCKS)]],
    device const int *spans [[buffer(SEISMIC_BUFFER_SPANS)]],
    device int *pool [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    const ulong row = ulong(group.x) * 64 + thread_index;
    if (row >= ulong(SEISMIC_DIM_M))
        return;
    const uint r = uint(SEISMIC_DIM_R);
    const uint b = uint(SEISMIC_DIM_B);
    const uint bu = uint(SEISMIC_DIM_BU);
    const uint k = uint(SEISMIC_DIM_K);
    device const int *row_spans = spans + row * r * 2;
    device const int *picks = blocks + row * b;
    uint units = 0;
    for (uint span = 0; span < r; ++span)
        units += candidate::Pool::length(row_spans[span * 2], row_spans[span * 2 + 1]);
    uint pooled = 0;
    for (uint pick = 0; pick < b; ++pick)
        if (picks[pick] >= 0)
            pooled += metal::min(units, (uint(picks[pick]) + 1) * bu) - metal::min(units, uint(picks[pick]) * bu);
    const uint wanted = metal::min(k, units);
    uint fill = wanted > pooled ? wanted - pooled : 0u;
    candidate::Pool out{row_spans, r, pool + row * (b + r) * 2, b + r, 0, 0, 0, 0, 0};
    uint cursor = 0;
    for (uint pick = 0; pick <= b; ++pick) {
        const bool last = pick == b;
        if (!last && picks[pick] < 0)
            continue;
        const uint start = last ? units : metal::min(units, uint(picks[pick]) * bu);
        const uint end = last ? units : metal::min(units, (uint(picks[pick]) + 1) * bu);
        const uint take = metal::min(fill, start - cursor);
        out.take(cursor, cursor + take);
        fill -= take;
        out.take(start, end);
        cursor = end;
    }
    out.flush();
    for (uint entry = out.count; entry < b + r; ++entry) {
        out.out[entry * 2] = -1;
        out.out[entry * 2 + 1] = -1;
    }
}
