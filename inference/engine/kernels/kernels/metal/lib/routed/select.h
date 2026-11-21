// Expert selection of the general routed feed-forward (`routed_select`, and
// its hints through `rank`), one simdgroup per row: scores, the
// selection-only bias, top-K and the combine weights. The counterpart of
// `cuda/lib/routed/select.cuh`, `vulkan/lib/routed/select.glsl` and
// `cpu/lib/routed/select.rs`.
//
// Lane `lane` holds experts lane + 32 i (i < P = ceil(E / 32)); when E is
// not a multiple of 32, lanes at or past E % 32 hold P - 1
// (`selection::candidates`, ranked.h) and their last register is
// `select::excluded()` (NaN), which `selection::precedes` never picks. Top-K
// keeps each lane's best remaining candidate `(head_r, head_s, head_e)`
// seeded by one linear scan of its P slots, then runs K simdgroup argmax
// rounds over the 32 lane heads (ties to the higher expert index); on each
// round before the last, only the winning lane marks its winning slot
// `select::excluded()` and re-scans its P slots for the next head. Because
// each slot stays at `lane + 32 * i`, no `expert[P]` array is stored. The
// winners are the portable body's K argmax rounds in the same order.
//
// This file is independent of any entry ABI.

#include "../core/reduce.h"
#include "ranked.h"

namespace select {

constant constexpr int softmax = 0;
constant constexpr int sigmoid = 1;
constant constexpr int sqrt_softplus = 2;

constant constexpr int no_normalization = 0;
constant constexpr int sum = 1;
constant constexpr int sum_plus_epsilon = 2;
constant constexpr int clamped_sum = 3;

// The scores of this lane's experts from their logits (in place); the
// softmax runs over the `held` leading values only (the others are excluded
// entries, whose scores are unused). Every loop unrolls fully, so every
// register index is static.
template <uint P>
inline void scores(int function, thread float (&values)[P], uint held) {
    if (function == softmax) {
        float maximum = -INFINITY;
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < P; ++i)
            if (i < held)
                maximum = metal::max(maximum, values[i]);
        maximum = simd_max(maximum);
        float total = 0.0f;
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < P; ++i) {
            if (i < held) {
                values[i] = metal::exp(values[i] - maximum);
                total += values[i];
            }
        }
        total = simd_sum(total);
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < P; ++i)
            values[i] = values[i] / total;
    } else if (function == sigmoid) {
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < P; ++i)
            values[i] = 1.0f / (1.0f + metal::exp(-values[i]));
    } else {
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < P; ++i)
            values[i] = metal::sqrt(metal::max(values[i], 0.0f) + metal::log(1.0f + metal::exp(-metal::abs(values[i]))));
    }
}

// Sentinel for an excluded or already-chosen slot: a quiet NaN, which no
// `selection::ranked` value is, so `selection::precedes` never picks it
// (every library compiles with MTLMathMode::Safe).
inline float excluded() {
    return as_type<float>(0x7fc00000u);
}

// Finds this lane's best remaining candidate by `selection::precedes`,
// optionally marking expert `erase` (`>= 0`) as `excluded()` (NaN) in the
// same pass. Excluded and taken slots hold `excluded()`, which precedes
// nothing, so an exhausted lane yields (-inf, 0.0f, -1). The loop unrolls
// fully so every register index is static.
template <uint P>
inline void lane_max(thread float (&ranked)[P], thread const float (&score)[P], uint lane, int erase,
                     thread float &head_r, thread float &head_s, thread int &head_e) {
    head_r = -INFINITY;
    head_s = 0.0f;
    head_e = -1;
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < P; ++i) {
        const int e = int(lane + 32u * i);
        if (erase >= 0 && e == erase)
            ranked[i] = excluded();
        else if (selection::precedes(ranked[i], e, head_r, head_e)) {
            head_r = ranked[i];
            head_s = score[i];
            head_e = e;
        }
    }
}

// A row's K choices, rank r at slot K - 1 - r (as `rank_experts` stores
// them), as every lane of its simdgroup holds them: each winner and its
// unbiased score.
template <uint K>
struct Selected {
    int winners[K];
    float chosen[K];
};

// The selection order of the general routed form (`select_experts`,
// functions.seismic), shared by `routed_select` and its hints: the scores of
// this lane's candidates from their logits (`logit(expert)`) by `function`,
// ranked on score + `bias(expert)` (each held expert's ranked value given to
// `keys`), then K argmax rounds over the 32 lane heads with lazy winning-lane
// refill. Every lane of the simdgroup calls it.
template <uint E, uint K, typename Logit, typename Bias, typename Keys = selection::NoKeys>
inline Selected<K> rank(int function, Logit logit, Bias bias, uint lane, Keys keys = Keys{}) {
    constexpr uint P = (E + 31) / 32;
    // P when 32 divides E (a constant: no excluded entries).
    const uint held = selection::candidates<E>(lane);
    float score[P], ranked[P];
    // Every loop over a register array unrolls: a dynamically indexed array
    // lives in stack memory.
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < P; ++i)
        score[i] = i < held ? logit(lane + 32u * i) : 0.0f;
    scores<P>(function, score, held);
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < P; ++i)
        ranked[i] = i < held ? selection::ranked(score[i] + bias(int(lane + 32u * i))) : excluded();
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < P; ++i)
        if (i < held)
            keys(lane + 32u * i, ranked[i]);

    float head_r, head_s;
    int head_e;
    lane_max<P>(ranked, score, lane, -1, head_r, head_s, head_e);

    Selected<K> selected;
    _Pragma("clang loop unroll(full)")
    for (uint rank = 0; rank < K; ++rank) {
        float best;
        const int winner = reduce::argmax<reduce::HigherIndex>(head_r, head_e, best);
        // The winner's unbiased score, from its lane.
        const float winner_score = simd_broadcast(head_s, ushort(uint(winner) % 32u));
        if (rank + 1u < K && head_e == winner)
            lane_max<P>(ranked, score, lane, winner, head_r, head_s, head_e);
        selected.chosen[K - 1 - rank] = winner_score;
        selected.winners[K - 1 - rank] = winner;
    }
    return selected;
}

// The weight of a selected score after normalization over the slot-order
// sum `denominator`.
inline float normalized(int normalization, float weight, float denominator, float epsilon) {
    if (normalization == sum)
        return weight / denominator;
    if (normalization == sum_plus_epsilon)
        return weight / (denominator + epsilon);
    if (normalization == clamped_sum)
        return weight / metal::max(denominator, epsilon);
    return weight;
}

} // namespace select
