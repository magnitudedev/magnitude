// packing_bound: a resident weight's packing bound, read through its
// format's packed codes (`projection::packing_codes`): per run of the
// 32 / folds columns that share a scale, the largest sum over one row's run
// of (code - centre)^2. One threadgroup per 32-column block: its threads take
// the rows in turn, each row's block as the four lanes' words of the packed
// forms (`at`, `load`, `step`), and join their largest per run.
#define KERNEL_W0 SEISMIC_WEIGHT
#include "lib/projection/projection.h"
#include "lib/projection/packing.h"

kernel void packing_bound(device const uchar *weight [[buffer(SEISMIC_BUFFER_WEIGHT)]],
    device float *bound [[buffer(SEISMIC_BUFFER_BOUND)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint block [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    typedef projection::packing_codes<packets::W0> coded;
    static_assert(coded::folds == 1 || coded::folds == 2, "a packing bound is of a format's packed codes");
    const uint k = uint(SEISMIC_DIM_K), rows = uint(SEISMIC_DIM_N);
    const projection::Weights<packets::W0> w{weight, KERNEL_W0_LAYOUT(k), k};
    // Each run's largest sum of squares over this thread's rows: a run's
    // sum is at most 32 x 128^2, so it is exact in 32 bits and in F32.
    uint2 largest = uint2(0u);
    for (uint n = thread_index; n < rows; n += 256u) {
        uint2 squares = uint2(0u);
        PROJECTION_UNROLL
        for (uint x = 0; x < 8u; x += 2u) {
            const typename coded::words words = coded::load(w, coded::at(w, n, x), block);
            PROJECTION_UNROLL
            for (uint s = 0; s < 4u; ++s) {
                const int2 centred = int2(as_type<ushort2>(coded::step(words, s))) - int(coded::centre);
                const uint2 square = uint2(centred * centred);
                squares[s * coded::folds / 4u] += square.x + square.y;
            }
        }
        largest = metal::max(largest, squares);
    }
    largest = uint2(simd_max(largest.x), simd_max(largest.y));
    threadgroup uint2 peaks[8];
    if (lane == 0)
        peaks[sg] = largest;
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    if (thread_index == 0) {
        for (uint i = 1; i < 8u; ++i)
            largest = metal::max(largest, peaks[i]);
        for (uint run = 0; run < coded::folds; ++run)
            bound[block * coded::folds + run] = float(largest[run]);
    }
}
