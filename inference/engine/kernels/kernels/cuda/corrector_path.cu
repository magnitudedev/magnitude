// corrector_path: one block of 256 threads per slot walks its lattice
// greedily: for each of the first P mask rows, every candidate's score
// (incoming · anchor for row 0, incoming · the previous selection's outgoing
// row after it) is reduced over the block in candidate order, and the first
// highest-scoring candidate is selected.
#include "lib/core/reduce.cuh"

extern "C" __global__ void corrector_path(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    __shared__ float partials[32];
    const int *candidates = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_CANDIDATES));
    const float *incoming = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_INCOMING));
    const float *outgoing = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_OUTGOING));
    const float *anchor = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_ANCHOR));
    int *selection = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_SELECTION));
    const u64 slot = blockIdx.x;
    const u64 count = SEISMIC_DIM_K, factor = SEISMIC_DIM_F;
    u64 previous = 0;
    for (u64 step = 0; step < SEISMIC_DIM_P; ++step) {
        // Row 0 scores against the anchor, later rows against the previous
        // selection's outgoing row.
        const float *other = step == 0 ? anchor + slot * SEISMIC_ANCHOR_STRIDE_0
                                       : outgoing + slot * SEISMIC_OUTGOING_STRIDE_0 +
                                             (step - 1) * SEISMIC_OUTGOING_STRIDE_1 +
                                             previous * SEISMIC_OUTGOING_STRIDE_2;
        const u64 other_stride = step == 0 ? SEISMIC_ANCHOR_STRIDE_1 : SEISMIC_OUTGOING_STRIDE_3;
        float best = -__int_as_float(0x7f800000);
        u64 chosen = 0;
        for (u64 candidate = 0; candidate < count; ++candidate) {
            const float *row = incoming + slot * SEISMIC_INCOMING_STRIDE_0 + step * SEISMIC_INCOMING_STRIDE_1 +
                               candidate * SEISMIC_INCOMING_STRIDE_2;
            float partial = 0.0f;
            for (u64 f = threadIdx.x; f < factor; f += blockDim.x)
                partial = seismic_fma_rn(row[f * SEISMIC_INCOMING_STRIDE_3], other[f * other_stride], partial);
            const float score = reduce::group_sum(partial, partials);
            if (score > best) {
                best = score;
                chosen = candidate;
            }
        }
        previous = chosen;
        if (threadIdx.x == 0) {
            const u64 at = step * SEISMIC_SELECTION_STRIDE_0 + slot * SEISMIC_SELECTION_STRIDE_1;
            selection[at] = candidates[slot * SEISMIC_CANDIDATES_STRIDE_0 + step * SEISMIC_CANDIDATES_STRIDE_1 +
                                       chosen * SEISMIC_CANDIDATES_STRIDE_2];
            selection[at + SEISMIC_SELECTION_STRIDE_2] = 0;
        }
    }
}
