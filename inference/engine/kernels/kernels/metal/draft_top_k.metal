// draft_top_k: one threadgroup of 256 threads per row selects the row's K
// highest logits in (value descending, token ascending) order. Tiles of up to
// 4096 items are loaded once into registers (16 slots/thread with a 16-bit
// active bitmask); larger scans fall back to rank-ordered passes.
#include <seismic/element.h>
#include "lib/core/reduce.h"

// Loads up to 4096 logits (16 per thread across 256 threads) once into
// registers using coalesced packed_float4 loads when unit-strided, returning
// the 16-bit active slot bitmask.
inline ushort load_logit_tile(
    device const float *base,
    ulong valid,
    ulong stride,
    uint thread_index,
    thread float (&vals)[16]) {
    const bool vec_ok = stride == 1ul;
    ushort active = 0u;
    for (uint c = 0; c < 4u; ++c) {
        const ulong local_base = ulong(c) * 1024ul + ulong(thread_index) * 4ul;
        if (vec_ok && local_base + 4ul <= valid) {
            const float4 v = float4(*reinterpret_cast<device const packed_float4 *>(base + local_base));
            vals[c * 4u + 0u] = v.x;
            vals[c * 4u + 1u] = v.y;
            vals[c * 4u + 2u] = v.z;
            vals[c * 4u + 3u] = v.w;
            active |= ushort(0xfu << (c * 4u));
        } else {
            for (uint j = 0; j < 4u; ++j) {
                const ulong offset = local_base + ulong(j);
                if (offset < valid) {
                    vals[c * 4u + j] = base[offset * stride];
                    active |= ushort(1u << (c * 4u + j));
                }
            }
        }
    }
    return active;
}

// Local argmax over a thread's active slots in a 4096-element logit tile.
// Slots are in strictly ascending token order, so strict `value > best`
// preserves lower-token tie-breaking while excluding NaNs.
inline reduce::Candidate tile_argmax(
    thread const float (&vals)[16],
    ushort active,
    int begin,
    uint thread_index,
    thread int &best_slot) {
    float best = -INFINITY;
    best_slot = -1;
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
        ? begin + ((best_slot >> 2) * 1024 + int(thread_index) * 4 + (best_slot & 3))
        : reduce::LowerIndex::none;
    return reduce::Candidate{best, best_token};
}

// Updates (best, best_token) with (value, token) if it orders strictly after
// (previous, previous_token) and beats the current best. NaNs are excluded.
inline void consider_after(
    float value,
    int token,
    float previous,
    int previous_token,
    thread float &best,
    thread int &best_token) {
    const bool after = value < previous || (value == previous && token > previous_token);
    if (after && (value > best || (value == best && token < best_token))) {
        best = value;
        best_token = token;
    }
}

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
    if (vocabulary <= 4096ul) {
        float vals[16];
        ushort active = load_logit_tile(line, vocabulary, SEISMIC_LOGITS_STRIDE_1, thread_index, vals);
        for (ulong rank = 0; rank < SEISMIC_DIM_K; ++rank) {
            int best_slot;
            const reduce::Candidate local = tile_argmax(vals, active, 0, thread_index, best_slot);
            const reduce::Candidate winner = reduce::group_argmax<reduce::LowerIndex, 8>(
                local, partials, sg, lane);
            if (best_slot >= 0 && local.index == winner.index)
                active &= ushort(~(1u << best_slot));
            if (thread_index == 0) {
                const ulong at = ulong(row) * SEISMIC_DIM_K + rank;
                candidates[at * SEISMIC_CANDIDATES_STRIDE_0] = winner.index;
                candidates[at * SEISMIC_CANDIDATES_STRIDE_0 + SEISMIC_CANDIDATES_STRIDE_1] = 0;
                unary.put(ulong(row) * SEISMIC_UNARY_STRIDE_0 + rank * SEISMIC_UNARY_STRIDE_1, winner.score);
            }
        }
        return;
    }
    const bool vec_ok = SEISMIC_LOGITS_STRIDE_1 == 1ul;
    const ulong vec_end = vocabulary & ~3ul;
    float previous = INFINITY;
    int previous_token = -1;
    for (ulong rank = 0; rank < SEISMIC_DIM_K; ++rank) {
        float best = -INFINITY;
        int best_token = reduce::LowerIndex::none;
        if (vec_ok) {
            for (ulong base = ulong(thread_index) * 4ul; base < vec_end; base += 1024ul) {
                const float4 v = float4(*reinterpret_cast<device const packed_float4 *>(line + base));
                consider_after(v.x, int(base + 0ul), previous, previous_token, best, best_token);
                consider_after(v.y, int(base + 1ul), previous, previous_token, best, best_token);
                consider_after(v.z, int(base + 2ul), previous, previous_token, best, best_token);
                consider_after(v.w, int(base + 3ul), previous, previous_token, best, best_token);
            }
            const ulong tail_token = vec_end + ulong(thread_index);
            if (tail_token < vocabulary) {
                consider_after(line[tail_token], int(tail_token), previous, previous_token, best, best_token);
            }
        } else {
            for (ulong token = thread_index; token < vocabulary; token += 256ul) {
                consider_after(
                    line[token * SEISMIC_LOGITS_STRIDE_1], int(token), previous, previous_token, best, best_token);
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
    float vals[16];
    ushort active = load_logit_tile(part_line, valid, SEISMIC_LOGITS_STRIDE_1, thread_index, vals);
    for (ulong rank = 0; rank < count; ++rank) {
        int best_slot;
        const reduce::Candidate local = tile_argmax(vals, active, int(begin), thread_index, best_slot);
        const reduce::Candidate winner = reduce::group_argmax<reduce::LowerIndex, 8>(
            local, partials, sg, lane);
        if (best_slot >= 0 && local.index == winner.index)
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
// When count <= 4096 (up to 16 partials per thread), load all partials once
// into registers and mask chosen winners via a 16-bit active bitmask; fall
// back to streaming rank passes only for larger counts.
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
    const ulong row_base = ulong(row) * count;
    if (count <= 4096ul) {
        float vals[16];
        int toks[16];
        ushort active = 0u;
        for (uint slot = 0; slot < 16u; ++slot) {
            const ulong i = ulong(slot) * 256ul + ulong(thread_index);
            if (i < count) {
                const ulong at = row_base + i;
                const int token = partial_ids[at * 2ul];
                if (token >= 0) {
                    vals[slot] = partial_values[at];
                    toks[slot] = token;
                    active |= ushort(1u << slot);
                }
            }
        }
        for (ulong rank = 0; rank < SEISMIC_DIM_K; ++rank) {
            float best = -INFINITY;
            int best_token = reduce::LowerIndex::none;
            int best_slot = -1;
            if (active != 0u) {
                for (int slot = 0; slot < 16; ++slot) {
                    if ((active & (1u << slot)) == 0u)
                        continue;
                    const float value = vals[slot];
                    const int token = toks[slot];
                    if (value > best || (value == best && token < best_token)) {
                        best = value;
                        best_token = token;
                        best_slot = slot;
                    }
                }
            }
            const reduce::Candidate winner = reduce::group_argmax<reduce::LowerIndex, 8>(
                reduce::Candidate{best, best_token}, partials, sg, lane);
            if (best_slot >= 0 && best_token == winner.index)
                active &= ushort(~(1u << best_slot));
            if (thread_index == 0) {
                const ulong at = ulong(row) * SEISMIC_DIM_K + rank;
                candidates[at * SEISMIC_CANDIDATES_STRIDE_0] = winner.index;
                candidates[at * SEISMIC_CANDIDATES_STRIDE_0 + SEISMIC_CANDIDATES_STRIDE_1] = 0;
                unary.put(ulong(row) * SEISMIC_UNARY_STRIDE_0 + rank * SEISMIC_UNARY_STRIDE_1, winner.score);
            }
        }
        return;
    }
    float previous = INFINITY;
    int previous_token = -1;
    for (ulong rank = 0; rank < SEISMIC_DIM_K; ++rank) {
        float best = -INFINITY;
        int best_token = reduce::LowerIndex::none;
        for (ulong i = thread_index; i < count; i += 256ul) {
            const ulong at = row_base + i;
            const int token = partial_ids[at * 2ul];
            if (token < 0) continue;
            consider_after(partial_values[at], token, previous, previous_token, best, best_token);
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
