// How every expert selection holds and orders its experts (`rank_experts`,
// functions.seismic): the routes' `rank_row` (also the hints) and
// `routed_select` (select.h). The counterpart of
// `cuda/lib/routed/ranked.cuh`, `vulkan/lib/routed/ranked.glsl` and
// `cpu/lib/routed/ranked.rs`.
//   candidates: lane `lane` of a simdgroup holds experts lane + 32 i,
//     ceil(E / 32) registers; when E is not a multiple of 32, lanes at or
//     past E % 32 hold one fewer, and their last register is an excluded
//     entry, held as a taken one (`routed_select`: ranked -inf, expert -1;
//     the routes: `out`).
//   ranked: a NaN as -inf. A bit test, which no math mode folds.
//   precedes: descending ranked value, ties to the higher expert index.
//     Over ranked values it is a total order, so every rank names an
//     expert whatever a row holds, and an excluded entry precedes nothing.
//   out: an entry out of the ranking, for a selection that keeps no expert
//     array: a NaN, which no ranked value is, so `precedes` neither chooses
//     it nor lets it hold a best. Every library compiles with
//     MTLMathMode::Safe (seismic's Metal profile), whose comparisons of a NaN
//     are false.
#include "../core/reduce.h"

namespace selection {

constant constexpr float out = NAN;

// The experts lane `lane` holds of E: ceil(E / 32) on the first E % 32
// lanes, E / 32 on the others (a constant, E / 32, when 32 divides E).
template <uint E>
inline uint candidates(uint lane) {
    return E / 32 + (E % 32 != 0 && lane < E % 32 ? 1u : 0u);
}

inline float ranked(float value) {
    return (as_type<uint>(value) & 0x7fffffffu) > 0x7f800000u ? -INFINITY : value;
}

inline bool precedes(float value, int expert, float best, int best_expert) {
    return value > best || (value == best && expert > best_expert);
}

template <uint K>
struct Ranked {
    float chosen[K];
    int winners[K];
};

// The ranking of one row by one simdgroup, reading the row's logit of expert
// `e` as `logit_of(e)`: softmax over `EXPERTS` experts, then K rounds of
// simdgroup argmax (`precedes`).
template <uint EXPERTS, uint K, typename LogitOf>
inline Ranked<K> rank_row(LogitOf logit_of, uint lane) {
    constexpr uint per_lane = (EXPERTS + 31u) / 32u;
    const uint held = candidates<EXPERTS>(lane);
    float probability[per_lane];
    float maximum = -INFINITY;
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < per_lane; ++i) {
        probability[i] = i < held ? logit_of(lane + 32u * i) : -INFINITY;
        maximum = metal::max(maximum, probability[i]);
    }
    maximum = simd_max(maximum);
    float total = 0.0f;
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < per_lane; ++i) {
        if (i < held) {
            probability[i] = metal::exp(probability[i] - maximum);
            total += probability[i];
        }
    }
    total = simd_sum(total);
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < per_lane; ++i)
        probability[i] = i < held ? ranked(probability[i] / total) : out;

    Ranked<K> result;
    _Pragma("clang loop unroll(full)")
    for (uint rank = 0; rank < K; ++rank) {
        float best = -INFINITY;
        int best_expert = -1;
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < per_lane; ++i) {
            if (probability[i] >= best) {
                best = probability[i];
                best_expert = int(lane + 32u * i);
            }
        }
        float winner_probability;
        const int winner = reduce::argmax<reduce::HigherIndex>(best, best_expert, winner_probability);
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < per_lane; ++i)
            if (winner == int(lane + 32u * i))
                probability[i] = out;
        result.chosen[rank] = winner_probability;
        result.winners[rank] = winner;
    }
    return result;
}

} // namespace selection
