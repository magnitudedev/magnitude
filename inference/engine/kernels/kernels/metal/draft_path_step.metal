// draft_path_step: one threadgroup of 256 threads per row reduces each
// candidate's score unary + (predecessor * hidden) . successor, then selects
// the first highest-scoring candidate.
#include "lib/core/activation.h"
#include "lib/core/reduce.h"

typedef element::Act activation;

kernel void draft_path_step(
    device const int *candidates [[buffer(SEISMIC_BUFFER_CANDIDATES)]],
    device const float *unary [[buffer(SEISMIC_BUFFER_UNARY)]],
    device const uchar *hidden [[buffer(SEISMIC_BUFFER_HIDDEN)]],
    device const uchar *predecessor [[buffer(SEISMIC_BUFFER_PREDECESSOR)]],
    device const uchar *successor [[buffer(SEISMIC_BUFFER_SUCCESSOR)]],
    device int *selection [[buffer(SEISMIC_BUFFER_SELECTION)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partials[8];
    const ulong rank = SEISMIC_DIM_R, count = SEISMIC_DIM_K;
    float best = -INFINITY;
    ulong chosen = 0;
    for (ulong candidate = 0; candidate < count; ++candidate) {
        const ulong at = ulong(row) * count + candidate;
        float partial = 0.0f;
        for (ulong r = thread_index; r < rank; r += 256u) {
            const float joint =
                element::at<activation>(predecessor, ulong(row) * SEISMIC_PREDECESSOR_STRIDE_0 + r * SEISMIC_PREDECESSOR_STRIDE_1) *
                element::at<activation>(hidden, ulong(row) * SEISMIC_HIDDEN_STRIDE_0 + r * SEISMIC_HIDDEN_STRIDE_1);
            partial = metal::fma(joint,
                element::at<activation>(successor, at * SEISMIC_SUCCESSOR_STRIDE_0 + r * SEISMIC_SUCCESSOR_STRIDE_1), partial);
        }
        const float score = reduce::group_sum<8>(partial, partials, sg, lane) +
            unary[ulong(row) * SEISMIC_UNARY_STRIDE_0 + candidate * SEISMIC_UNARY_STRIDE_1];
        if (score > best) {
            best = score;
            chosen = at;
        }
    }
    if (thread_index == 0) {
        selection[ulong(row) * SEISMIC_SELECTION_STRIDE_0] = candidates[chosen * SEISMIC_CANDIDATES_STRIDE_0];
        selection[ulong(row) * SEISMIC_SELECTION_STRIDE_0 + SEISMIC_SELECTION_STRIDE_1] = 0;
    }
}
