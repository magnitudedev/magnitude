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

// Exact vocabulary partitions, followed by their top-K union.
extern "C" __global__ void draft_top_k_parts(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long ulong;
    const float infinity = __int_as_float(0x7f800000);
    const float *logits = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_LOGITS));
    int *candidates = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIAL_TOKENS));
    float *unary = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIAL_VALUES));
    __shared__ float partials[32];
    const ulong row = blockIdx.x;
    const ulong part = blockIdx.y;
    const ulong parts = (SEISMIC_DIM_V + 4095ul) / 4096ul;
    const ulong begin = part * 4096ul;
    const ulong vocabulary = SEISMIC_DIM_V < begin + 4096ul ? SEISMIC_DIM_V : begin + 4096ul;
    const ulong count = SEISMIC_DIM_K < vocabulary - begin ? SEISMIC_DIM_K : vocabulary - begin;
    const float *line = logits + ulong(row) * SEISMIC_LOGITS_STRIDE_0;
    float previous = infinity;
    int previous_token = -1;
    for (ulong rank = 0; rank < count; ++rank) {
        float best = -infinity;
        int best_token = 0x7fffffff;
        for (ulong token = begin + threadIdx.x; token < vocabulary; token += 256u) {
            const float value = line[token * SEISMIC_LOGITS_STRIDE_1];
            const bool after = value < previous || (value == previous && int(token) > previous_token);
            if (after && (value > best || (value == best && int(token) < best_token))) {
                best = value;
                best_token = int(token);
            }
        }
        const float chosen = reduce::group_max(best, partials);
        const float lowest = -reduce::group_max(best == chosen ? -float(best_token) : -infinity, partials);
        previous = chosen;
        previous_token = int(lowest);
        if (threadIdx.x == 0) {
            const ulong at = (row * parts + part) * SEISMIC_DIM_K + rank;
            candidates[at * 2ul] = previous_token;
            candidates[at * 2ul + 1ul] = 0;
            unary[at] = chosen;
        }
    }
    // A short final partition contributes only its actual tokens.
    if (threadIdx.x == 0) {
        for (ulong rank = count; rank < SEISMIC_DIM_K; ++rank) {
            const ulong at = (row * parts + part) * SEISMIC_DIM_K + rank;
            candidates[at * 2ul] = -1;
            candidates[at * 2ul + 1ul] = 0;
            unary[at] = -infinity;
        }
    }
}


extern "C" __global__ void draft_top_k_merge(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long ulong;
    const float infinity = __int_as_float(0x7f800000);
    const int *partial_ids = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIAL_TOKENS));
    const float *partial_values = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIAL_VALUES));
    int *candidates = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_CANDIDATES));
    float *unary = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_UNARY));
    const ulong row = blockIdx.x;
    __shared__ float partials[32];
    const ulong parts = (SEISMIC_DIM_V + 4095ul) / 4096ul;
    const ulong count = parts * SEISMIC_DIM_K;
    float previous = infinity;
    int previous_token = -1;
    for (ulong rank = 0; rank < SEISMIC_DIM_K; ++rank) {
        float best = -infinity;
        int best_token = 0x7fffffff;
        for (ulong i = threadIdx.x; i < count; i += 256ul) {
            const ulong at = ulong(row) * count + i;
            const float value = partial_values[at];
            const int token = partial_ids[at * 2ul];
            if (token < 0) continue;
            const bool after = value < previous || (value == previous && token > previous_token);
            if (after && (value > best || (value == best && token < best_token))) {
                best = value;
                best_token = token;
            }
        }
        const float chosen = reduce::group_max(best, partials);
        const float lowest = -reduce::group_max(best == chosen ? -float(best_token) : -infinity, partials);
        previous = chosen;
        previous_token = int(lowest);
        if (threadIdx.x == 0) {
            const ulong at = ulong(row) * SEISMIC_DIM_K + rank;
            candidates[at * SEISMIC_CANDIDATES_STRIDE_0] = previous_token;
            candidates[at * SEISMIC_CANDIDATES_STRIDE_0 + SEISMIC_CANDIDATES_STRIDE_1] = 0;
            unary[ulong(row) * SEISMIC_UNARY_STRIDE_0 + rank * SEISMIC_UNARY_STRIDE_1] = chosen;
        }
    }
}
