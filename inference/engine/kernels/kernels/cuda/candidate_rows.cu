// candidate_rows on CUDA (contract and portable body in indexer.seismic), as
// Metal's (`metal/candidate_rows.metal`):
//
// A thread per row, as the CPU native: its taken units form ascending runs
// (before each picked block, ascending and clipped to the row's units, the
// fill takes the gap's lowest units while it lasts, then the block's own;
// after the last, the rest of the fill). Each run is mapped through the
// row's spans, split where a span ends, and adjacent runs within one span
// join. The rest of the row's B + R spans are [-1, -1].
//
// Dynamic extents and strides read `seismic_words`.

namespace candidate {

typedef unsigned int u32;
typedef unsigned long long u64;

// A row's taken units as plane-row spans: the pending run (joined with an
// adjacent next), mapped through the row's list spans on flush.
struct Pool {
    const int *list;
    u64 list_stride, list_bound;
    u32 spans;
    int *out;
    u64 out_stride, out_bound;
    u32 capacity;
    u32 count;
    // The list span the next run starts in, and the units before it.
    u32 span;
    u32 offset;
    u32 start;
    u32 end;

    __device__ static __forceinline__ u32 length(int lo, int hi) {
        return lo >= 0 && hi > lo ? static_cast<u32>(hi - lo) : 0u;
    }

    __device__ __forceinline__ void flush() {
        u32 at = start;
        while (at < end && span < spans) {
            const int lo = list[span * list_stride];
            const u32 size = length(lo, list[span * list_stride + list_bound]);
            if (offset + size <= at) {
                offset += size;
                ++span;
                continue;
            }
            const u32 stop = min(end, offset + size);
            if (count < capacity) {
                out[count * out_stride] = lo + static_cast<int>(at - offset);
                out[count * out_stride + out_bound] = lo + static_cast<int>(stop - offset);
            }
            ++count;
            at = stop;
        }
        start = end = 0;
    }

    // Takes the units [first, last), ascending after every earlier run.
    __device__ __forceinline__ void take(u32 first, u32 last) {
        if (first == last) return;
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

extern "C" __global__ void candidate_rows(SEISMIC_KERNEL_PARAMS) {
    using namespace candidate;
    const u64 row = static_cast<u64>(blockIdx.x) * 64 + threadIdx.x;
    if (row >= SEISMIC_DIM_M) return;
    const u32 r = static_cast<u32>(SEISMIC_DIM_R);
    const u32 b = static_cast<u32>(SEISMIC_DIM_B);
    const u32 bu = static_cast<u32>(SEISMIC_DIM_BU);
    const u32 k = static_cast<u32>(SEISMIC_DIM_K);
    const int *row_spans = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_SPANS)) + row * SEISMIC_SPANS_STRIDE_0;
    const u64 span_stride = SEISMIC_SPANS_STRIDE_1;
    const u64 bound_stride = SEISMIC_SPANS_STRIDE_2;
    const int *picks = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_BLOCKS)) + row * SEISMIC_BLOCKS_STRIDE_0;
    const u64 pick_stride = SEISMIC_BLOCKS_STRIDE_1;
    u32 units = 0;
    for (u32 span = 0; span < r; ++span)
        units += Pool::length(row_spans[span * span_stride], row_spans[span * span_stride + bound_stride]);
    u32 pooled = 0;
    for (u32 pick = 0; pick < b; ++pick) {
        const int block = picks[pick * pick_stride];
        if (block >= 0)
            pooled += min(units, (static_cast<u32>(block) + 1) * bu) - min(units, static_cast<u32>(block) * bu);
    }
    const u32 wanted = min(k, units);
    u32 fill = wanted > pooled ? wanted - pooled : 0u;
    Pool out{row_spans, span_stride, bound_stride, r,
             reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER)) + row * SEISMIC_RESULT_0_STRIDE_0,
             SEISMIC_RESULT_0_STRIDE_1, SEISMIC_RESULT_0_STRIDE_2, b + r, 0, 0, 0, 0, 0};
    u32 cursor = 0;
    for (u32 pick = 0; pick <= b; ++pick) {
        const bool last = pick == b;
        const int block = last ? 0 : picks[pick * pick_stride];
        if (!last && block < 0) continue;
        const u32 start = last ? units : min(units, static_cast<u32>(block) * bu);
        const u32 end = last ? units : min(units, (static_cast<u32>(block) + 1) * bu);
        const u32 take = min(fill, start - cursor);
        out.take(cursor, cursor + take);
        fill -= take;
        out.take(start, end);
        cursor = end;
    }
    out.flush();
    for (u32 entry = out.count; entry < b + r; ++entry) {
        out.out[entry * out.out_stride] = -1;
        out.out[entry * out.out_stride + out.out_bound] = -1;
    }
}
