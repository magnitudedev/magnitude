// Expert selection of the general routed feed-forward (`routed_select`, and
// its hints through `rank_row`), one warp per row: scores, the selection-only bias, top-K and the combine
// weights. The counterpart of `metal/lib/routed/select.h`,
// `vulkan/lib/routed/select.glsl` and `cpu/lib/routed/select.rs`.
//
// Lane `lane` holds experts lane + 32 i (i < P = ceil(E / 32)); when E is
// not a multiple of 32, lanes at or past E % 32 hold P - 1
// (`selection::candidates`, ranked.cuh) and their last register is
// `select::excluded()` (NaN), which `selection::precedes` never picks. Top-K
// keeps each lane's best remaining candidate `(head_r, head_s, head_e)`
// seeded by one linear scan of its P slots, then runs K warp argmax rounds
// over the 32 lane heads (ties to the higher expert index); on each round
// before the last, only the winning lane marks its winning slot
// `select::excluded()` and re-scans its P slots for the next head. Because
// each slot stays at `lane + 32 * i`, no `expert[P]` array is stored. The
// winners are the portable body's K argmax rounds in the same order.
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

// Sentinel for an excluded or already-chosen slot: a quiet NaN, which no
// `selection::ranked` value is, so `selection::precedes` never picks it
// (NVRTC compiles without fast math).
__device__ __forceinline__ float excluded() {
    return __int_as_float(0x7fc00000);
}

// Finds this lane's best remaining candidate by `selection::precedes`,
// optionally marking expert `erase` (`>= 0`) as `excluded()` (NaN) in the
// same pass. Excluded and taken slots hold `excluded()`, which precedes
// nothing, so an exhausted lane yields (-inf, 0.0f, -1).
template <unsigned P>
__device__ __forceinline__ void lane_max(float (&ranked)[P], const float (&score)[P], unsigned lane, int erase,
                                         float &head_r, float &head_s, int &head_e) {
    head_r = -__int_as_float(0x7f800000);
    head_s = 0.0f;
    head_e = -1;
#pragma unroll
    for (unsigned i = 0; i < P; ++i) {
        const int e = static_cast<int>(lane + 32 * i);
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
// them), as every lane of its warp holds them: each winner and its unbiased
// score.
template <unsigned K> struct Selected {
    int winners[K];
    float chosen[K];
};

// The selection order of the general routed form (`select_experts`,
// functions.seismic), shared by `routed_select` and its hints: the scores of
// this lane's candidates from their logits (`logit(expert)`) by `function`,
// ranked on score + `bias(expert)` (each held expert's ranked value given to
// `keys`), then K argmax rounds over the 32 lane heads with lazy winning-lane
// refill. Every lane of the warp calls it.
template <unsigned E, unsigned K, class Logit, class Bias, class Keys = selection::NoKeys>
__device__ __forceinline__ Selected<K> rank_row(int function, Logit logit, Bias bias, unsigned lane,
                                                Keys keys = Keys{}) {
    constexpr unsigned P = (E + 31) / 32;
    // P when 32 divides E (a constant: no excluded entries).
    const unsigned held = selection::candidates<E>(lane);
    float score[P], ranked[P];
#pragma unroll
    for (unsigned i = 0; i < P; ++i)
        score[i] = i < held ? logit(lane + 32 * i) : 0.0f;
    scores<P>(function, score, held);
#pragma unroll
    for (unsigned i = 0; i < P; ++i)
        ranked[i] = i < held ? selection::ranked(seismic_add_rn(score[i], bias(static_cast<int>(lane + 32 * i))))
                             : excluded();
#pragma unroll
    for (unsigned i = 0; i < P; ++i)
        if (i < held)
            keys(lane + 32 * i, ranked[i]);

    float head_r, head_s;
    int head_e;
    lane_max<P>(ranked, score, lane, -1, head_r, head_s, head_e);

    Selected<K> selected;
#pragma unroll
    for (unsigned rank = 0; rank < K; ++rank) {
        float best;
        const int winner = reduce::argmax<reduce::HigherIndex>(head_r, head_e, best);
        // The winner's unbiased score, from its lane.
        const float winner_score = __shfl_sync(0xffffffffu, head_s, static_cast<unsigned>(winner) % 32u);
        if (rank + 1u < K && head_e == winner)
            lane_max<P>(ranked, score, lane, winner, head_r, head_s, head_e);
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
