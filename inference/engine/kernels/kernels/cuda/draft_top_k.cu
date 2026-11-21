// draft_top_k: one block of 256 threads per row selects the row's K highest
// logits in (value descending, token ascending) order. Round r scans the
// tokens ordered strictly after round r - 1's choice, so no chosen set is
// kept.
#include <seismic/element.cuh>
#include "lib/core/reduce.cuh"

extern "C" __global__ void draft_top_k(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    __shared__ reduce::Candidate partials[32];
    const float *logits = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_LOGITS));
    int *candidates = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_CANDIDATES));
    const auto unary = SEISMIC_OUT_UNARY;
    const float infinity = __int_as_float(0x7f800000);
    const u64 row = blockIdx.x;
    const float *line = logits + row * SEISMIC_LOGITS_STRIDE_0;
    float previous = infinity;
    int previous_token = -1;
    for (u64 rank = 0; rank < SEISMIC_DIM_K; ++rank) {
        float best = -infinity;
        int best_token = reduce::LowerIndex::none;
        for (u64 token = threadIdx.x; token < SEISMIC_DIM_V; token += blockDim.x) {
            const float value = line[token * SEISMIC_LOGITS_STRIDE_1];
            const bool after = value < previous || (value == previous && (int)token > previous_token);
            if (after && (value > best || (value == best && (int)token < best_token))) {
                best = value;
                best_token = (int)token;
            }
        }
        const reduce::Candidate winner = reduce::group_argmax<reduce::LowerIndex>(
            reduce::Candidate{best, best_token}, partials);
        previous = winner.score;
        previous_token = winner.index;
        if (threadIdx.x == 0) {
            const u64 at = row * SEISMIC_DIM_K + rank;
            candidates[at * SEISMIC_CANDIDATES_STRIDE_0] = previous_token;
            candidates[at * SEISMIC_CANDIDATES_STRIDE_0 + SEISMIC_CANDIDATES_STRIDE_1] = 0;
            unary.put(row * SEISMIC_UNARY_STRIDE_0 + rank * SEISMIC_UNARY_STRIDE_1, previous);
        }
    }
}

// Exact vocabulary partitions, followed by their top-K union. Each of the 256
// threads loads its 16 partition logits once into registers (coalesced float4
// loads when unit-strided and 16-byte aligned) and masks chosen winners in
// registers via a 16-bit active bitmask.
extern "C" __global__ void draft_top_k_parts(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long ulong;
    const float infinity = __int_as_float(0x7f800000);
    const float *logits = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_LOGITS));
    int *candidates = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIAL_TOKENS));
    float *unary = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIAL_VALUES));
    __shared__ reduce::Candidate partials[32];
    const ulong row = blockIdx.x;
    const ulong part = blockIdx.y;
    const ulong parts = (SEISMIC_DIM_V + 4095ul) / 4096ul;
    const ulong begin = part * 4096ul;
    const ulong vocabulary = SEISMIC_DIM_V < begin + 4096ul ? SEISMIC_DIM_V : begin + 4096ul;
    const ulong valid = vocabulary - begin;
    const ulong count = SEISMIC_DIM_K < valid ? SEISMIC_DIM_K : valid;
    const float *line = logits + ulong(row) * SEISMIC_LOGITS_STRIDE_0;
    const float *part_line = line + begin * SEISMIC_LOGITS_STRIDE_1;
    const bool vec_ok = SEISMIC_LOGITS_STRIDE_1 == 1ul && (reinterpret_cast<ulong>(part_line) & 15ul) == 0ul;
    float vals[16];
    unsigned short active = 0u;
#pragma unroll
    for (unsigned c = 0; c < 4; ++c) {
        const ulong local_base = ulong(c) * 1024ul + ulong(threadIdx.x) * 4ul;
        if (vec_ok && local_base + 4ul <= valid) {
            const float4 v = *reinterpret_cast<const float4 *>(part_line + local_base);
            vals[c * 4u + 0u] = v.x;
            vals[c * 4u + 1u] = v.y;
            vals[c * 4u + 2u] = v.z;
            vals[c * 4u + 3u] = v.w;
            active |= static_cast<unsigned short>(0xfu << (c * 4u));
        } else {
#pragma unroll
            for (unsigned j = 0; j < 4; ++j) {
                const ulong offset = local_base + ulong(j);
                if (offset < valid) {
                    vals[c * 4u + j] = part_line[offset * SEISMIC_LOGITS_STRIDE_1];
                    active |= static_cast<unsigned short>(1u << (c * 4u + j));
                }
            }
        }
    }
    for (ulong rank = 0; rank < count; ++rank) {
        float best = -infinity;
        int best_slot = -1;
        if (active != 0u) {
#pragma unroll
            for (int idx = 0; idx < 16; ++idx) {
                if ((active & (1u << idx)) == 0u)
                    continue;
                const float value = vals[idx];
                if (best_slot < 0 ? (value == value) : (value > best)) {
                    best = value;
                    best_slot = idx;
                }
            }
        }
        const int best_token = best_slot >= 0
            ? int(begin) + ((best_slot >> 2) * 1024 + int(threadIdx.x) * 4 + (best_slot & 3))
            : reduce::LowerIndex::none;
        const reduce::Candidate winner = reduce::group_argmax<reduce::LowerIndex>(
            reduce::Candidate{best, best_token}, partials);
        if (best_slot >= 0 && best_token == winner.index)
            active &= static_cast<unsigned short>(~(1u << best_slot));
        if (threadIdx.x == 0) {
            const ulong at = (row * parts + part) * SEISMIC_DIM_K + rank;
            candidates[at * 2ul] = winner.index;
            candidates[at * 2ul + 1ul] = 0;
            unary[at] = winner.score;
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
    const auto unary = SEISMIC_OUT_UNARY;
    const ulong row = blockIdx.x;
    __shared__ reduce::Candidate partials[32];
    const ulong parts = (SEISMIC_DIM_V + 4095ul) / 4096ul;
    const ulong count = parts * SEISMIC_DIM_K;
    float previous = infinity;
    int previous_token = -1;
    for (ulong rank = 0; rank < SEISMIC_DIM_K; ++rank) {
        float best = -infinity;
        int best_token = reduce::LowerIndex::none;
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
        const reduce::Candidate winner = reduce::group_argmax<reduce::LowerIndex>(
            reduce::Candidate{best, best_token}, partials);
        previous = winner.score;
        previous_token = winner.index;
        if (threadIdx.x == 0) {
            const ulong at = ulong(row) * SEISMIC_DIM_K + rank;
            candidates[at * SEISMIC_CANDIDATES_STRIDE_0] = previous_token;
            candidates[at * SEISMIC_CANDIDATES_STRIDE_0 + SEISMIC_CANDIDATES_STRIDE_1] = 0;
            unary.put(ulong(row) * SEISMIC_UNARY_STRIDE_0 + rank * SEISMIC_UNARY_STRIDE_1, previous);
        }
    }
}
