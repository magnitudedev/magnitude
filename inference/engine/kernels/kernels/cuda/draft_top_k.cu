// draft_top_k: one block of 256 threads per row selects the row's K highest
// logits in (value descending, token ascending) order. Round r scans the
// tokens ordered strictly after round r - 1's choice, so no chosen set is
// kept.
#include "lib/core/reduce.cuh"

extern "C" __global__ void draft_top_k(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    __shared__ float partials[32];
    const float *logits = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_LOGITS));
    int *candidates = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_CANDIDATES));
    float *unary = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_UNARY));
    const float infinity = __int_as_float(0x7f800000);
    const u64 row = blockIdx.x;
    const float *line = logits + row * SEISMIC_LOGITS_STRIDE_0;
    float previous = infinity;
    int previous_token = -1;
    for (u64 rank = 0; rank < SEISMIC_DIM_K; ++rank) {
        float best = -infinity;
        int best_token = 0x7fffffff;
        for (u64 token = threadIdx.x; token < SEISMIC_DIM_V; token += blockDim.x) {
            const float value = line[token * SEISMIC_LOGITS_STRIDE_1];
            const bool after = value < previous || (value == previous && (int)token > previous_token);
            if (after && (value > best || (value == best && (int)token < best_token))) {
                best = value;
                best_token = (int)token;
            }
        }
        const float chosen = reduce::group_max(best, partials);
        const float lowest = -reduce::group_max(best == chosen ? -(float)best_token : -infinity, partials);
        previous = chosen;
        previous_token = (int)lowest;
        if (threadIdx.x == 0) {
            const u64 at = row * SEISMIC_DIM_K + rank;
            candidates[at * SEISMIC_CANDIDATES_STRIDE_0] = previous_token;
            candidates[at * SEISMIC_CANDIDATES_STRIDE_0 + SEISMIC_CANDIDATES_STRIDE_1] = 0;
            unary[row * SEISMIC_UNARY_STRIDE_0 + rank * SEISMIC_UNARY_STRIDE_1] = chosen;
        }
    }
}
