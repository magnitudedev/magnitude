// draft_path_step: one block of 256 threads per row reduces each candidate's
// score unary + (predecessor * hidden) . successor, then selects the first
// highest-scoring candidate.
#include "lib/core/activation.cuh"
#include "lib/core/reduce.cuh"

extern "C" __global__ void draft_path_step(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    __shared__ float partials[32];
    const int *candidates = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_CANDIDATES));
    const float *unary = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_UNARY));
    const element::u8 *hidden = SEISMIC_PTR(SEISMIC_BUFFER_HIDDEN);
    const element::u8 *predecessor = SEISMIC_PTR(SEISMIC_BUFFER_PREDECESSOR);
    const element::u8 *successor = SEISMIC_PTR(SEISMIC_BUFFER_SUCCESSOR);
    int *selection = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_SELECTION));
    const u64 row = blockIdx.x;
    float best = -__int_as_float(0x7f800000);
    u64 chosen = 0;
    for (u64 candidate = 0; candidate < SEISMIC_DIM_K; ++candidate) {
        const u64 at = row * SEISMIC_DIM_K + candidate;
        float partial = 0.0f;
        for (u64 r = threadIdx.x; r < SEISMIC_DIM_R; r += blockDim.x) {
            const float joint =
                element::at<element::Act>(predecessor, row * SEISMIC_PREDECESSOR_STRIDE_0 + r * SEISMIC_PREDECESSOR_STRIDE_1) *
                element::at<element::Act>(hidden, row * SEISMIC_HIDDEN_STRIDE_0 + r * SEISMIC_HIDDEN_STRIDE_1);
            partial = seismic_fma_rn(joint,
                element::at<element::Act>(successor, at * SEISMIC_SUCCESSOR_STRIDE_0 + r * SEISMIC_SUCCESSOR_STRIDE_1), partial);
        }
        const float score = reduce::group_sum(partial, partials) +
                            unary[row * SEISMIC_UNARY_STRIDE_0 + candidate * SEISMIC_UNARY_STRIDE_1];
        if (score > best) {
            best = score;
            chosen = at;
        }
    }
    if (threadIdx.x == 0) {
        selection[row * SEISMIC_SELECTION_STRIDE_0] = candidates[chosen * SEISMIC_CANDIDATES_STRIDE_0];
        selection[row * SEISMIC_SELECTION_STRIDE_0 + SEISMIC_SELECTION_STRIDE_1] = 0;
    }
}
