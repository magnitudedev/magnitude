// How every expert selection holds and orders its experts (`rank_experts`,
// functions.seismic): `route_rank_at` (also the hints) and `routed_select`
// (select.glsl). The counterpart of `metal/lib/routed/ranked.h`,
// `cuda/lib/routed/ranked.cuh` and `cpu/lib/routed/ranked.rs`.
//   selection_candidates: lane `lane` of a subgroup holds experts
//     lane + 32 i, ceil(E / 32) registers; when E is not a multiple of 32,
//     lanes at or past E % 32 hold one fewer, and their last register is an
//     excluded entry (ranked -inf, expert -1, as a taken one).
//   selection_ranked: a NaN as -inf. A bit test, which no math mode folds.
//   selection_precedes: descending ranked value, ties to the higher expert
//     index. Over ranked values it is a total order, so every rank names an
//     expert whatever a row holds, and an excluded entry precedes nothing.
#include "../core/reduce.glsl"

// The experts lane `lane` holds of `experts`: ceil(E / 32) on the first
// E % 32 lanes, E / 32 on the others (a constant, E / 32, when 32 divides a
// constant E).
uint selection_candidates(const uint experts, const uint lane) {
    return experts / 32u + ((experts % 32u) != 0u && lane < experts % 32u ? 1u : 0u);
}

float selection_ranked(float value) {
    return (floatBitsToUint(value) & 0x7fffffffu) > 0x7f800000u ? -uintBitsToFloat(0x7f800000u) : value;
}

bool selection_precedes(float value, int expert, float best, int best_expert) {
    return value > best || (value == best && expert > best_expert);
}

#define ROUTE_DEFINE_RANK_AT \
void route_rank_at(uint64_t logits_base, uint64_t row, out float chosen[ROUTE_K], out int winners[ROUTE_K]) { \
    const uint lane = SEISMIC_LANE; \
    const uint held = selection_candidates(ROUTE_E, lane); \
    float probability[ROUTE_PER_LANE]; \
    int expert[ROUTE_PER_LANE]; \
    float maximum = -uintBitsToFloat(0x7f800000u); \
    [[unroll]] for (uint i = 0u; i < ROUTE_PER_LANE; ++i) { \
        expert[i] = i < held ? int(lane + 32u * i) : -1; \
        probability[i] = i < held \
            ? element_f32_at(logits_base + (row * uint64_t(ROUTE_STRIDE) + uint64_t(lane + 32u * i)) * 4ul) \
            : -uintBitsToFloat(0x7f800000u); \
        maximum = max(maximum, probability[i]); \
    } \
    maximum = seismic_subgroup_max(maximum); \
    float total = 0.0; \
    [[unroll]] for (uint i = 0u; i < ROUTE_PER_LANE; ++i) { \
        if (i < held) { \
            probability[i] = exp(probability[i] - maximum); \
            total += probability[i]; \
        } \
    } \
    total = seismic_subgroup_sum_f32(total); \
    [[unroll]] for (uint i = 0u; i < ROUTE_PER_LANE; ++i) \
        if (i < held) \
            probability[i] = selection_ranked(seismic_div_rn(probability[i], total)); \
    for (uint rank = 0u; rank < ROUTE_K; ++rank) { \
        float best = -uintBitsToFloat(0x7f800000u); \
        int best_expert = -1; \
        [[unroll]] for (uint i = 0u; i < ROUTE_PER_LANE; ++i) { \
            if (selection_precedes(probability[i], expert[i], best, best_expert)) { \
                best = probability[i]; \
                best_expert = expert[i]; \
            } \
        } \
        float winner_probability; \
        const int winner = reduce_argmax(best, best_expert, REDUCE_HIGHER_INDEX, winner_probability); \
        [[unroll]] for (uint i = 0u; i < ROUTE_PER_LANE; ++i) { \
            if (winner == expert[i]) { \
                probability[i] = -uintBitsToFloat(0x7f800000u); \
                expert[i] = -1; \
            } \
        } \
        chosen[rank] = winner_probability; \
        winners[rank] = winner; \
    } \
}
