// draft_top_k: one block of 256 threads per row selects the row's K highest
// logits in (value descending, token ascending) order. Tiles of <= 4096
// elements are loaded once into registers (coalesced float4 loads when
// unit-strided and 16-byte aligned) and masked via a 16-bit active bitmask;
// larger inputs stream in rank order with vectorized float4 loads where aligned.
#include <seismic/element.cuh>
#include "lib/core/reduce.cuh"

namespace {

using u64 = unsigned long long;

__device__ __forceinline__ unsigned short load_logit_tile(
    const float *part_line, u64 valid, u64 stride, float (&vals)[16]) {
    const bool vec_ok = stride == 1ull && (reinterpret_cast<u64>(part_line) & 15ull) == 0ull;
    unsigned short active = 0u;
#pragma unroll
    for (unsigned c = 0; c < 4; ++c) {
        const u64 local_base = static_cast<u64>(c) * 1024ull + static_cast<u64>(threadIdx.x) * 4ull;
        if (vec_ok && local_base + 4ull <= valid) {
            const float4 v = *reinterpret_cast<const float4 *>(part_line + local_base);
            vals[c * 4u + 0u] = v.x;
            vals[c * 4u + 1u] = v.y;
            vals[c * 4u + 2u] = v.z;
            vals[c * 4u + 3u] = v.w;
            active |= static_cast<unsigned short>(0xfu << (c * 4u));
        } else {
#pragma unroll
            for (unsigned j = 0; j < 4; ++j) {
                const u64 offset = local_base + static_cast<u64>(j);
                if (offset < valid) {
                    vals[c * 4u + j] = part_line[offset * stride];
                    active |= static_cast<unsigned short>(1u << (c * 4u + j));
                }
            }
        }
    }
    return active;
}

__device__ __forceinline__ reduce::Candidate select_tile_winner(
    const float (&vals)[16], unsigned short &active, int begin, float infinity, reduce::Candidate *partials) {
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
        ? begin + ((best_slot >> 2) * 1024 + static_cast<int>(threadIdx.x) * 4 + (best_slot & 3))
        : reduce::LowerIndex::none;
    const reduce::Candidate winner = reduce::group_argmax<reduce::LowerIndex>(
        reduce::Candidate{best, best_token}, partials);
    if (best_slot >= 0 && best_token == winner.index)
        active &= static_cast<unsigned short>(~(1u << best_slot));
    return winner;
}

__device__ __forceinline__ void consider_streaming(
    float value, int token, float previous, int previous_token, float &best, int &best_token) {
    const bool after = value < previous || (value == previous && token > previous_token);
    if (after && (value > best || (value == best && token < best_token))) {
        best = value;
        best_token = token;
    }
}

} // namespace

extern "C" __global__ void draft_top_k(SEISMIC_KERNEL_PARAMS) {
    __shared__ reduce::Candidate partials[32];
    const float *logits = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_LOGITS));
    int *candidates = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_CANDIDATES));
    const auto unary = SEISMIC_OUT_UNARY;
    const float infinity = __int_as_float(0x7f800000);
    const u64 row = blockIdx.x;
    const float *line = logits + row * SEISMIC_LOGITS_STRIDE_0;
    if (SEISMIC_DIM_V <= 4096ull) {
        float vals[16];
        unsigned short active = load_logit_tile(line, SEISMIC_DIM_V, SEISMIC_LOGITS_STRIDE_1, vals);
        for (u64 rank = 0; rank < SEISMIC_DIM_K; ++rank) {
            const reduce::Candidate winner = select_tile_winner(vals, active, 0, infinity, partials);
            if (threadIdx.x == 0) {
                const u64 at = row * SEISMIC_DIM_K + rank;
                candidates[at * SEISMIC_CANDIDATES_STRIDE_0] = winner.index;
                candidates[at * SEISMIC_CANDIDATES_STRIDE_0 + SEISMIC_CANDIDATES_STRIDE_1] = 0;
                unary.put(row * SEISMIC_UNARY_STRIDE_0 + rank * SEISMIC_UNARY_STRIDE_1, winner.score);
            }
        }
        return;
    }
    const bool vec_ok = SEISMIC_LOGITS_STRIDE_1 == 1ull && (reinterpret_cast<u64>(line) & 15ull) == 0ull;
    const u64 vec_end = SEISMIC_DIM_V & ~3ull;
    float previous = infinity;
    int previous_token = -1;
    for (u64 rank = 0; rank < SEISMIC_DIM_K; ++rank) {
        float best = -infinity;
        int best_token = reduce::LowerIndex::none;
        if (vec_ok) {
            for (u64 base = static_cast<u64>(threadIdx.x) * 4ull; base < vec_end; base += 1024ull) {
                const float4 v = *reinterpret_cast<const float4 *>(line + base);
                const int tok = static_cast<int>(base);
                consider_streaming(v.x, tok + 0, previous, previous_token, best, best_token);
                consider_streaming(v.y, tok + 1, previous, previous_token, best, best_token);
                consider_streaming(v.z, tok + 2, previous, previous_token, best, best_token);
                consider_streaming(v.w, tok + 3, previous, previous_token, best, best_token);
            }
            const u64 tail = vec_end + threadIdx.x;
            if (tail < SEISMIC_DIM_V) {
                consider_streaming(line[tail], static_cast<int>(tail), previous, previous_token, best, best_token);
            }
        } else {
            for (u64 token = threadIdx.x; token < SEISMIC_DIM_V; token += blockDim.x) {
                consider_streaming(
                    line[token * SEISMIC_LOGITS_STRIDE_1], static_cast<int>(token),
                    previous, previous_token, best, best_token);
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
    const float infinity = __int_as_float(0x7f800000);
    const float *logits = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_LOGITS));
    int *candidates = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIAL_TOKENS));
    float *unary = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIAL_VALUES));
    __shared__ reduce::Candidate partials[32];
    const u64 row = blockIdx.x;
    const u64 part = blockIdx.y;
    const u64 parts = (SEISMIC_DIM_V + 4095ull) / 4096ull;
    const u64 begin = part * 4096ull;
    const u64 vocabulary = SEISMIC_DIM_V < begin + 4096ull ? SEISMIC_DIM_V : begin + 4096ull;
    const u64 valid = vocabulary - begin;
    const u64 count = SEISMIC_DIM_K < valid ? SEISMIC_DIM_K : valid;
    const float *line = logits + row * SEISMIC_LOGITS_STRIDE_0;
    const float *part_line = line + begin * SEISMIC_LOGITS_STRIDE_1;
    float vals[16];
    unsigned short active = load_logit_tile(part_line, valid, SEISMIC_LOGITS_STRIDE_1, vals);
    for (u64 rank = 0; rank < count; ++rank) {
        const reduce::Candidate winner = select_tile_winner(
            vals, active, static_cast<int>(begin), infinity, partials);
        if (threadIdx.x == 0) {
            const u64 at = (row * parts + part) * SEISMIC_DIM_K + rank;
            candidates[at * 2ull] = winner.index;
            candidates[at * 2ull + 1ull] = 0;
            unary[at] = winner.score;
        }
    }
    // A short final partition contributes only its actual tokens.
    if (threadIdx.x == 0) {
        for (u64 rank = count; rank < SEISMIC_DIM_K; ++rank) {
            const u64 at = (row * parts + part) * SEISMIC_DIM_K + rank;
            candidates[at * 2ull] = -1;
            candidates[at * 2ull + 1ull] = 0;
            unary[at] = -infinity;
        }
    }
}

extern "C" __global__ void draft_top_k_merge(SEISMIC_KERNEL_PARAMS) {
    const float infinity = __int_as_float(0x7f800000);
    const int *partial_ids = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIAL_TOKENS));
    const float *partial_values = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIAL_VALUES));
    int *candidates = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_CANDIDATES));
    const auto unary = SEISMIC_OUT_UNARY;
    const u64 row = blockIdx.x;
    __shared__ reduce::Candidate partials[32];
    const u64 parts = (SEISMIC_DIM_V + 4095ull) / 4096ull;
    const u64 count = parts * SEISMIC_DIM_K;
    if (count <= 4096ull) {
        float vals[16];
        int toks[16];
        unsigned short active = 0u;
        const u64 base = row * count;
#pragma unroll
        for (int slot = 0; slot < 16; ++slot) {
            const u64 i = static_cast<u64>(slot) * 256ull + threadIdx.x;
            if (i < count) {
                const u64 at = base + i;
                const int token = partial_ids[at * 2ull];
                if (token >= 0) {
                    const float value = partial_values[at];
                    if (value == value) {
                        vals[slot] = value;
                        toks[slot] = token;
                        active |= static_cast<unsigned short>(1u << slot);
                    }
                }
            }
        }
        for (u64 rank = 0; rank < SEISMIC_DIM_K; ++rank) {
            float best = -infinity;
            int best_token = reduce::LowerIndex::none;
            int best_slot = -1;
            if (active != 0u) {
#pragma unroll
                for (int idx = 0; idx < 16; ++idx) {
                    if ((active & (1u << idx)) == 0u)
                        continue;
                    const float value = vals[idx];
                    const int token = toks[idx];
                    if (value > best || (value == best && token < best_token)) {
                        best = value;
                        best_token = token;
                        best_slot = idx;
                    }
                }
            }
            const reduce::Candidate winner = reduce::group_argmax<reduce::LowerIndex>(
                reduce::Candidate{best, best_token}, partials);
            if (best_slot >= 0 && best_token == winner.index)
                active &= static_cast<unsigned short>(~(1u << best_slot));
            if (threadIdx.x == 0) {
                const u64 at = row * SEISMIC_DIM_K + rank;
                candidates[at * SEISMIC_CANDIDATES_STRIDE_0] = winner.index;
                candidates[at * SEISMIC_CANDIDATES_STRIDE_0 + SEISMIC_CANDIDATES_STRIDE_1] = 0;
                unary.put(row * SEISMIC_UNARY_STRIDE_0 + rank * SEISMIC_UNARY_STRIDE_1, winner.score);
            }
        }
        return;
    }
    float previous = infinity;
    int previous_token = -1;
    for (u64 rank = 0; rank < SEISMIC_DIM_K; ++rank) {
        float best = -infinity;
        int best_token = reduce::LowerIndex::none;
        for (u64 i = threadIdx.x; i < count; i += 256ull) {
            const u64 at = row * count + i;
            const float value = partial_values[at];
            const int token = partial_ids[at * 2ull];
            if (token < 0) continue;
            consider_streaming(value, token, previous, previous_token, best, best_token);
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
