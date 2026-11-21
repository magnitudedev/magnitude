// draft_top_k: one threadgroup of 256 threads per row selects the row's K
// highest logits in (value descending, token ascending) order. Round r scans
// the tokens ordered strictly after round r - 1's choice, so no chosen set is
// kept.
#include <seismic/element.h>
#include "lib/core/reduce.h"

kernel void draft_top_k(
    device const float *logits [[buffer(SEISMIC_BUFFER_LOGITS)]],
    device int *candidates [[buffer(SEISMIC_BUFFER_CANDIDATES)]],
    SEISMIC_OUT_UNARY_PARAM(unary_cells),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup reduce::Candidate partials[8];
    const element::Out<element::F32> unary(unary_cells);
    const ulong vocabulary = SEISMIC_DIM_V;
    device const float *line = logits + ulong(row) * SEISMIC_LOGITS_STRIDE_0;
    float previous = INFINITY;
    int previous_token = -1;
    for (ulong rank = 0; rank < SEISMIC_DIM_K; ++rank) {
        float best = -INFINITY;
        int best_token = reduce::LowerIndex::none;
        for (ulong token = thread_index; token < vocabulary; token += 256u) {
            const float value = line[token * SEISMIC_LOGITS_STRIDE_1];
            const bool after = value < previous || (value == previous && int(token) > previous_token);
            if (after && (value > best || (value == best && int(token) < best_token))) {
                best = value;
                best_token = int(token);
            }
        }
        const reduce::Candidate winner = reduce::group_argmax<reduce::LowerIndex, 8>(
            reduce::Candidate{best, best_token}, partials, sg, lane);
        previous = winner.score;
        previous_token = winner.index;
        if (thread_index == 0) {
            const ulong at = ulong(row) * SEISMIC_DIM_K + rank;
            candidates[at * SEISMIC_CANDIDATES_STRIDE_0] = previous_token;
            candidates[at * SEISMIC_CANDIDATES_STRIDE_0 + SEISMIC_CANDIDATES_STRIDE_1] = 0;
            unary.put(ulong(row) * SEISMIC_UNARY_STRIDE_0 + rank * SEISMIC_UNARY_STRIDE_1, previous);
        }
    }
}

// Partitioning exposes vocabulary parallelism without dropping any candidate.
// Each of the 256 threads loads its 16 partition logits once into registers
// (coalesced packed_float4 loads when unit-strided) and masks chosen winners
// in registers via a 16-bit active bitmask.
kernel void draft_top_k_parts(
    device const float *logits [[buffer(SEISMIC_BUFFER_LOGITS)]],
    device int *candidates [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIAL_TOKENS)]],
    device float *unary [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIAL_VALUES)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup reduce::Candidate partials[8];
    const ulong row = group.x;
    const ulong part = group.y;
    const ulong parts = (SEISMIC_DIM_V + 4095ul) / 4096ul;
    const ulong begin = part * 4096ul;
    const ulong vocabulary = min(SEISMIC_DIM_V, begin + 4096ul);
    const ulong valid = vocabulary - begin;
    const ulong count = min(SEISMIC_DIM_K, valid);
    device const float *line = logits + ulong(row) * SEISMIC_LOGITS_STRIDE_0;
    device const float *part_line = line + begin * SEISMIC_LOGITS_STRIDE_1;
    const bool vec_ok = SEISMIC_LOGITS_STRIDE_1 == 1ul;
    float vals[16];
    ushort active = 0u;
    for (uint c = 0; c < 4u; ++c) {
        const ulong local_base = ulong(c) * 1024ul + ulong(thread_index) * 4ul;
        if (vec_ok && local_base + 4ul <= valid) {
            const float4 v = float4(*reinterpret_cast<device const packed_float4 *>(part_line + local_base));
            vals[c * 4u + 0u] = v.x;
            vals[c * 4u + 1u] = v.y;
            vals[c * 4u + 2u] = v.z;
            vals[c * 4u + 3u] = v.w;
            active |= ushort(0xfu << (c * 4u));
        } else {
            for (uint j = 0; j < 4u; ++j) {
                const ulong offset = local_base + ulong(j);
                if (offset < valid) {
                    vals[c * 4u + j] = part_line[offset * SEISMIC_LOGITS_STRIDE_1];
                    active |= ushort(1u << (c * 4u + j));
                }
            }
        }
    }
    for (ulong rank = 0; rank < count; ++rank) {
        float best = -INFINITY;
        int best_slot = -1;
        if (active != 0u) {
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
            ? int(begin) + ((best_slot >> 2) * 1024 + int(thread_index) * 4 + (best_slot & 3))
            : reduce::LowerIndex::none;
        const reduce::Candidate winner = reduce::group_argmax<reduce::LowerIndex, 8>(
            reduce::Candidate{best, best_token}, partials, sg, lane);
        if (best_slot >= 0 && best_token == winner.index)
            active &= ushort(~(1u << best_slot));
        if (thread_index == 0) {
            const ulong at = (row * parts + part) * SEISMIC_DIM_K + rank;
            candidates[at * 2ul] = winner.index;
            candidates[at * 2ul + 1ul] = 0;
            unary[at] = winner.score;
        }
    }
    // A short final partition contributes only its actual tokens.
    if (thread_index == 0) {
        for (ulong rank = count; rank < SEISMIC_DIM_K; ++rank) {
            const ulong at = (row * parts + part) * SEISMIC_DIM_K + rank;
            candidates[at * 2ul] = -1;
            candidates[at * 2ul + 1ul] = 0;
            unary[at] = -INFINITY;
        }
    }
}

// The global top K is contained in the union of the partitions' top K.
kernel void draft_top_k_merge(
    device const int *partial_ids [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIAL_TOKENS)]],
    device const float *partial_values [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIAL_VALUES)]],
    device int *candidates [[buffer(SEISMIC_BUFFER_CANDIDATES)]],
    SEISMIC_OUT_UNARY_PARAM(unary_cells),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup reduce::Candidate partials[8];
    const element::Out<element::F32> unary(unary_cells);
    const ulong parts = (SEISMIC_DIM_V + 4095ul) / 4096ul;
    const ulong count = parts * SEISMIC_DIM_K;
    float previous = INFINITY;
    int previous_token = -1;
    for (ulong rank = 0; rank < SEISMIC_DIM_K; ++rank) {
        float best = -INFINITY;
        int best_token = reduce::LowerIndex::none;
        for (ulong i = thread_index; i < count; i += 256ul) {
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
        const reduce::Candidate winner = reduce::group_argmax<reduce::LowerIndex, 8>(
            reduce::Candidate{best, best_token}, partials, sg, lane);
        previous = winner.score;
        previous_token = winner.index;
        if (thread_index == 0) {
            const ulong at = ulong(row) * SEISMIC_DIM_K + rank;
            candidates[at * SEISMIC_CANDIDATES_STRIDE_0] = previous_token;
            candidates[at * SEISMIC_CANDIDATES_STRIDE_0 + SEISMIC_CANDIDATES_STRIDE_1] = 0;
            unary.put(ulong(row) * SEISMIC_UNARY_STRIDE_0 + rank * SEISMIC_UNARY_STRIDE_1, previous);
        }
    }
}
