// Expert selection of the general routed feed-forward (`routed_select`, and
// its hints through `rank_row`), one warp per row: scores, the selection-only bias, top-K and the combine
// weights. The counterpart of `metal/lib/routed/select.h`,
// `vulkan/lib/routed/select.glsl` and `cpu/lib/routed/select.rs`.
//
// Lane `lane` holds experts lane + 32 i (i < P = ceil(E / 32)); when E is
// not a multiple of 32, lanes at or past E % 32 hold P - 1
// (`selection::candidates`, ranked.cuh) and their last register is an
// excluded entry (ranked -inf, expert -1, as a popped one). Top-K is one
// pass over a lane-local order: every lane sorts its P candidates once (an
// odd-even transposition network over registers, by `selection::precedes`
// over ranked values), then each of the K rounds is a single warp argmax
// over the 32 lane heads (ties to the higher expert index) and one shift of
// the winning lane's candidates. The winners are the portable body's K
// argmax rounds in the same order.
//
// This file is independent of any entry ABI.

#include "../core/reduce.cuh"
#include "ranked.cuh"

namespace select {

constexpr int softmax = 0;
constexpr int sigmoid = 1;
constexpr int sqrt_softplus = 2;

constexpr int no_normalization = 0;
constexpr int sum = 1;
constexpr int sum_plus_epsilon = 2;
constexpr int clamped_sum = 3;

// The scores of this lane's experts from their logits (in place); the
// softmax runs over the `held` leading values only (the others are excluded
// entries, whose scores are unused).
template <unsigned P> __device__ __forceinline__ void scores(int function, float (&values)[P], unsigned held) {
    if (function == softmax) {
        float maximum = -__int_as_float(0x7f800000);
#pragma unroll
        for (unsigned i = 0; i < P; ++i)
            if (i < held)
                maximum = fmaxf(maximum, values[i]);
        maximum = seismic_warp_max_f32(maximum);
        float total = 0.0f;
#pragma unroll
        for (unsigned i = 0; i < P; ++i) {
            if (i < held) {
                values[i] = expf(values[i] - maximum);
                total = seismic_add_rn(total, values[i]);
            }
        }
        total = seismic_warp_sum_f32(total);
#pragma unroll
        for (unsigned i = 0; i < P; ++i)
            values[i] = values[i] / total;
    } else if (function == sigmoid) {
#pragma unroll
        for (unsigned i = 0; i < P; ++i)
            values[i] = 1.0f / (1.0f + expf(-values[i]));
    } else {
#pragma unroll
        for (unsigned i = 0; i < P; ++i)
            values[i] = sqrtf(fmaxf(values[i], 0.0f) + logf(1.0f + expf(-fabsf(values[i]))));
    }
}

// Sorts this lane's (ranked, score, expert) triples by
// `selection::precedes`.
template <unsigned P>
__device__ __forceinline__ void sort(float (&ranked)[P], float (&score)[P], int (&expert)[P]) {
#pragma unroll
    for (unsigned pass = 0; pass < P; ++pass) {
#pragma unroll
        for (unsigned i = pass & 1u; i + 1u < P; i += 2u) {
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

// Pops the lane's head (the winning lane only).
template <unsigned P> __device__ __forceinline__ void pop(float (&ranked)[P], float (&score)[P], int (&expert)[P]) {
#pragma unroll
    for (unsigned i = 0; i + 1u < P; ++i) {
        ranked[i] = ranked[i + 1u];
        score[i] = score[i + 1u];
        expert[i] = expert[i + 1u];
    }
    ranked[P - 1] = -__int_as_float(0x7f800000);
    expert[P - 1] = -1;
}

// A row's K choices, rank r at slot K - 1 - r (as `rank_experts` stores
// them), as every lane of its warp holds them: each winner and its unbiased
// score.
template <unsigned K> struct Selected {
    int winners[K];
    float chosen[K];
};

// The selection order of the general routed form (`select_experts`,
// functions.seismic), shared by `routed_select` and its hints: the scores of
// this lane's candidates from their logits (`logit(expert)`) by `function`,
// ranked on score + `bias(expert)`, sorted, then K argmax rounds over the
// lane heads. Every lane of the warp calls it.
template <unsigned E, unsigned K, class Logit, class Bias>
__device__ __forceinline__ Selected<K> rank_row(int function, Logit logit, Bias bias, unsigned lane) {
    constexpr unsigned P = (E + 31) / 32;
    // P when 32 divides E (a constant: no excluded entries).
    const unsigned held = selection::candidates<E>(lane);
    float score[P], ranked[P];
    int expert[P];
#pragma unroll
    for (unsigned i = 0; i < P; ++i) {
        expert[i] = i < held ? static_cast<int>(lane + 32 * i) : -1;
        score[i] = i < held ? logit(lane + 32 * i) : 0.0f;
    }
    scores<P>(function, score, held);
#pragma unroll
    for (unsigned i = 0; i < P; ++i)
        ranked[i] = i < held ? selection::ranked(seismic_add_rn(score[i], bias(expert[i]))) : -__int_as_float(0x7f800000);
    sort<P>(ranked, score, expert);

    Selected<K> selected;
    for (unsigned rank = 0; rank < K; ++rank) {
        float best;
        const int winner = reduce::argmax<reduce::HigherIndex>(ranked[0], expert[0], best);
        // The winner's unbiased score, from its lane.
        const float winner_score = __shfl_sync(0xffffffffu, score[0], static_cast<unsigned>(winner) % 32u);
        if (expert[0] == winner)
            pop<P>(ranked, score, expert);
        selected.chosen[K - 1 - rank] = winner_score;
        selected.winners[K - 1 - rank] = winner;
    }
    return selected;
}

// The weight of a selected score after normalization over the slot-order
// sum `denominator`.
__device__ __forceinline__ float normalized(int normalization, float weight, float denominator, float epsilon) {
    if (normalization == sum)
        return weight / denominator;
    if (normalization == sum_plus_epsilon)
        return weight / seismic_add_rn(denominator, epsilon);
    if (normalization == clamped_sum)
        return weight / fmaxf(denominator, epsilon);
    return weight;
}

} // namespace select
