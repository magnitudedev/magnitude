// Expert selection of the general routed feed-forward (`routed_select`, and
// its hints through `rank`), one simdgroup per row: scores, the
// selection-only bias, top-K and the combine weights. The counterpart of
// `cuda/lib/routed/select.cuh`, `vulkan/lib/routed/select.glsl` and
// `cpu/lib/routed/select.rs`.
//
// Lane `lane` holds experts lane + 32 i (i < P = ceil(E / 32)); when E is
// not a multiple of 32, lanes at or past E % 32 hold P - 1
// (`selection::candidates`, ranked.h) and their last register is an
// excluded entry (ranked -inf, expert -1, as a popped one). Top-K is one
// pass over a lane-local order: every lane sorts its P candidates once (an
// odd-even transposition network over registers, by `selection::precedes`
// over ranked values), then each of the K rounds is a single simdgroup
// argmax over the 32 lane heads (ties to the higher expert index) and one
// shift of the winning lane's candidates. The winners are the portable
// body's K argmax rounds in the same order.
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

// Sorts this lane's (ranked, score, expert) triples by
// `selection::precedes`. Both loops unroll fully, so every register index is
// static.
template <uint P>
inline void sort(thread float (&ranked)[P], thread float (&score)[P], thread int (&expert)[P]) {
    _Pragma("clang loop unroll(full)")
    for (uint pass = 0; pass < P; ++pass) {
        _Pragma("clang loop unroll(full)")
        for (uint i = pass & 1u; i + 1u < P; i += 2u) {
            if (selection::precedes(ranked[i + 1u], expert[i + 1u], ranked[i], expert[i])) {
                const float r = ranked[i], s = score[i];
                const int e = expert[i];
                ranked[i] = ranked[i + 1u];
                score[i] = score[i + 1u];
                expert[i] = expert[i + 1u];
                ranked[i + 1u] = r;
                score[i + 1u] = s;
                expert[i + 1u] = e;
            }
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
// ranked on score + `bias(expert)`, sorted, then K argmax rounds over the
// lane heads. Every lane of the simdgroup calls it.
template <uint E, uint K, typename Logit, typename Bias>
inline Selected<K> rank(int function, Logit logit, Bias bias, uint lane) {
    constexpr uint P = (E + 31) / 32;
    // P when 32 divides E (a constant: no excluded entries).
    const uint held = selection::candidates<E>(lane);
    float score[P], ranked[P];
    int expert[P];
    // Every loop over a register array unrolls: a dynamically indexed array
    // lives in stack memory.
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < P; ++i) {
        expert[i] = i < held ? int(lane + 32 * i) : -1;
        score[i] = i < held ? logit(lane + 32 * i) : 0.0f;
    }
    scores<P>(function, score, held);
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < P; ++i)
        ranked[i] = i < held ? selection::ranked(score[i] + bias(expert[i])) : -INFINITY;
    sort<P>(ranked, score, expert);

    Selected<K> selected;
    _Pragma("clang loop unroll(full)")
    for (uint rank = 0; rank < K; ++rank) {
        float best;
        const int winner = reduce::argmax<reduce::HigherIndex>(ranked[0], expert[0], best);
        // The winner's unbiased score, from its lane.
        const float winner_score = simd_broadcast(score[0], ushort(uint(winner) % 32u));
        if (expert[0] == winner) {
            _Pragma("clang loop unroll(full)")
            for (uint i = 0; i + 1u < P; ++i) {
                ranked[i] = ranked[i + 1u];
                score[i] = score[i + 1u];
                expert[i] = expert[i + 1u];
            }
            ranked[P - 1] = -INFINITY;
            expert[P - 1] = -1;
        }
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
