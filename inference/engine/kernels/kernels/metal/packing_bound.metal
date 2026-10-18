// packing_bound: a resident weight's packing bound, read through its
// format's packed codes (`projection::packing_codes`): per run of the
// 32 / folds columns that share a scale, the largest sum over one row's run
// of (code - centre)^2. One threadgroup per 32-column block: its threads take
// the rows in turn, each row's block as the four lanes' words of the packed
// forms (`at`, `load`, `step`), and join their largest per run.
//
// The vector-coded formats (iq2xs, iq3xxs) have integer codes but no packed
// codes: their signed grid codes are centred at zero, and a row's block is
// its packet's four steps of decoded codes, `groups` runs of 32 / groups
// columns.
#define KERNEL_W0 SEISMIC_WEIGHT
#include "lib/projection/projection.h"
#include "lib/projection/packing.h"

// A row's block of squared centred codes per run, and the runs a block has.
template <typename W>
struct packing_bound_rows {
    typedef projection::packing_codes<W> coded;
    static_assert(coded::folds == 1 || coded::folds == 2, "a packing bound is of a format's packed codes");
    static constant constexpr uint runs = coded::folds;
    static uint2 squares(thread const projection::Weights<W> &w, uint n, uint block) {
        uint2 squares = uint2(0u);
        PROJECTION_UNROLL
        for (uint x = 0; x < 8u; x += 2u) {
            const typename coded::words words = coded::load(w, coded::at(w, n, x), block);
            PROJECTION_UNROLL
            for (uint s = 0; s < 4u; ++s) {
                const int2 centred = int2(as_type<ushort2>(coded::step(words, s))) - int(coded::centre);
                const uint2 square = uint2(centred * centred);
                squares[s * runs / 4u] += square.x + square.y;
            }
        }
        return squares;
    }
};
template <typename G>
struct packing_bound_rows<packets::Grid<G>> {
    typedef packets::Grid<G> W;
    static constant constexpr uint runs = W::groups;
    static uint2 squares(thread const projection::Weights<W> &w, uint n, uint block) {
        uint2 squares = uint2(0u);
        const typename W::packet packet = w.packet(n, block);
        PROJECTION_UNROLL
        for (uint s = 0; s < 4u; ++s) {
            float4 even, odd;
            W::codes(packet, s, even, odd);
            const int4 e = int4(even), o = int4(odd);
            const uint4 square = uint4(e * e + o * o);
            squares[s * runs / 4u] += square.x + square.y + square.z + square.w;
        }
        return squares;
    }
};

kernel void packing_bound(device const uchar *weight [[buffer(SEISMIC_BUFFER_WEIGHT)]],
    device float *bound [[buffer(SEISMIC_BUFFER_BOUND)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint block [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    typedef packing_bound_rows<packets::W0> coded;
    const uint k = uint(SEISMIC_DIM_K), rows = uint(SEISMIC_DIM_N);
    const projection::Weights<packets::W0> w{weight, KERNEL_W0_LAYOUT(k), k};
    // Each run's largest sum of squares over this thread's rows: a run's
    // sum is at most 32 x 128^2, so it is exact in 32 bits and in F32.
    uint2 largest = uint2(0u);
    for (uint n = thread_index; n < rows; n += 256u)
        largest = metal::max(largest, coded::squares(w, n, block));
    largest = uint2(simd_max(largest.x), simd_max(largest.y));
    threadgroup uint2 peaks[8];
    if (lane == 0)
        peaks[sg] = largest;
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    if (thread_index == 0) {
        for (uint i = 1; i < 8u; ++i)
            largest = metal::max(largest, peaks[i]);
        for (uint run = 0; run < coded::runs; ++run)
            bound[block * coded::runs + run] = float(largest[run]);
    }
}
