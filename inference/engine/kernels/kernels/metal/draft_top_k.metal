// draft_top_k: one threadgroup of 256 threads per row selects the row's K
// highest logits in (value descending, token ascending) order. Round r scans
// the tokens ordered strictly after round r - 1's choice, so no chosen set is
// kept.
#include "lib/core/reduce.h"

kernel void draft_top_k(
    device const float *logits [[buffer(SEISMIC_BUFFER_LOGITS)]],
    device int *candidates [[buffer(SEISMIC_BUFFER_CANDIDATES)]],
    device float *unary [[buffer(SEISMIC_BUFFER_UNARY)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partials[8];
    const ulong vocabulary = SEISMIC_DIM_V;
    device const float *line = logits + ulong(row) * SEISMIC_LOGITS_STRIDE_0;
    float previous = INFINITY;
    int previous_token = -1;
    for (ulong rank = 0; rank < SEISMIC_DIM_K; ++rank) {
        float best = -INFINITY;
        int best_token = 0x7fffffff;
        for (ulong token = thread_index; token < vocabulary; token += 256u) {
            const float value = line[token * SEISMIC_LOGITS_STRIDE_1];
            const bool after = value < previous || (value == previous && int(token) > previous_token);
            if (after && (value > best || (value == best && int(token) < best_token))) {
                best = value;
                best_token = int(token);
            }
        }
        const float chosen = reduce::group_max<8>(best, partials, sg, lane);
        const float lowest = -reduce::group_max<8>(best == chosen ? -float(best_token) : -INFINITY, partials, sg, lane);
        previous = chosen;
        previous_token = int(lowest);
        if (thread_index == 0) {
            const ulong at = ulong(row) * SEISMIC_DIM_K + rank;
            candidates[at * SEISMIC_CANDIDATES_STRIDE_0] = previous_token;
            candidates[at * SEISMIC_CANDIDATES_STRIDE_0 + SEISMIC_CANDIDATES_STRIDE_1] = 0;
            unary[ulong(row) * SEISMIC_UNARY_STRIDE_0 + rank * SEISMIC_UNARY_STRIDE_1] = chosen;
        }
    }
}
