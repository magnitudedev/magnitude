// corrector_path: one threadgroup of 256 threads per slot walks its lattice
// greedily: for each of the first P mask rows it reduces every candidate's
// score incoming[c] · (anchor, or the previous selection's outgoing row) in
// candidate order and selects the first highest-scoring one. Every thread
// receives each reduced score, so the walk stays uniform.
#include "lib/core/reduce.h"

#define PATH_THREADS 256u

kernel void corrector_path(
    device const int *candidates [[buffer(SEISMIC_BUFFER_CANDIDATES)]],
    device const float *incoming [[buffer(SEISMIC_BUFFER_INCOMING)]],
    device const float *outgoing [[buffer(SEISMIC_BUFFER_OUTGOING)]],
    device const float *anchor [[buffer(SEISMIC_BUFFER_ANCHOR)]],
    device int *selection [[buffer(SEISMIC_BUFFER_SELECTION)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint slot [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partials[PATH_THREADS / 32u];
    const ulong count = SEISMIC_DIM_K, width = SEISMIC_DIM_F, steps = SEISMIC_DIM_P;
    ulong previous = 0;
    for (ulong step = 0; step < steps; ++step) {
        device const float *other;
        ulong other_stride;
        if (step == 0) {
            other = anchor + ulong(slot) * SEISMIC_ANCHOR_STRIDE_0;
            other_stride = SEISMIC_ANCHOR_STRIDE_1;
        } else {
            other = outgoing + ulong(slot) * SEISMIC_OUTGOING_STRIDE_0 + (step - 1) * SEISMIC_OUTGOING_STRIDE_1 +
                previous * SEISMIC_OUTGOING_STRIDE_2;
            other_stride = SEISMIC_OUTGOING_STRIDE_3;
        }
        float best = -INFINITY;
        ulong chosen = 0;
        for (ulong candidate = 0; candidate < count; ++candidate) {
            device const float *row = incoming + ulong(slot) * SEISMIC_INCOMING_STRIDE_0 +
                step * SEISMIC_INCOMING_STRIDE_1 + candidate * SEISMIC_INCOMING_STRIDE_2;
            float partial = 0.0f;
            for (ulong f = thread_index; f < width; f += PATH_THREADS)
                partial = metal::fma(row[f * SEISMIC_INCOMING_STRIDE_3], other[f * other_stride], partial);
            const float score = reduce::group_sum<PATH_THREADS / 32u>(partial, partials, sg, lane);
            if (score > best) {
                best = score;
                chosen = candidate;
            }
        }
        previous = chosen;
        if (thread_index == 0) {
            const ulong at = step * SEISMIC_SELECTION_STRIDE_0 + ulong(slot) * SEISMIC_SELECTION_STRIDE_1;
            selection[at] = candidates[ulong(slot) * SEISMIC_CANDIDATES_STRIDE_0 + step * SEISMIC_CANDIDATES_STRIDE_1 +
                chosen * SEISMIC_CANDIDATES_STRIDE_2];
            selection[at + SEISMIC_SELECTION_STRIDE_2] = 0;
        }
    }
}
