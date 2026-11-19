// How every expert selection holds and orders its experts (`rank_experts`,
// functions.seismic): `routed_route`'s `rank_row` (also the hints) and
// `routed_select` (select.cuh). The counterpart of
// `metal/lib/routed/ranked.h`, `vulkan/lib/routed/ranked.glsl` and
// `cpu/lib/routed/ranked.rs`.
//   candidates: lane `lane` of a warp holds experts lane + 32 i,
//     ceil(E / 32) registers; when E is not a multiple of 32, lanes at or
//     past E % 32 hold one fewer, and their last register is an excluded
//     entry, held as a taken one (`routed_select`: ranked -inf, expert -1;
//     the route: `out`).
//   ranked: a NaN as -inf. A bit test, which no math mode folds.
//   precedes: descending ranked value, ties to the higher expert index.
//     Over ranked values it is a total order, so every rank names an
//     expert whatever a row holds, and an excluded entry precedes nothing.
//   out: an entry out of the ranking, for a selection that keeps no expert
//     array: a NaN, which no ranked value is, so `precedes` neither chooses
//     it nor lets it hold a best. NVRTC compiles without fast math, and PTX
//     comparisons of a NaN are false.
#include "../core/reduce.cuh"

namespace selection {

__device__ __forceinline__ float out() {
    return __int_as_float(0x7fc00000);
}

// The experts lane `lane` holds of E: ceil(E / 32) on the first E % 32
// lanes, E / 32 on the others (a constant, E / 32, when 32 divides E).
template <unsigned E> __device__ __forceinline__ unsigned candidates(unsigned lane) {
    return E / 32 + (E % 32 != 0 && lane < E % 32 ? 1u : 0u);
}

__device__ __forceinline__ float ranked(float value) {
    return (__float_as_uint(value) & 0x7fffffffu) > 0x7f800000u ? -__int_as_float(0x7f800000) : value;
}

__device__ __forceinline__ bool precedes(float value, int expert, float best, int best_expert) {
    return value > best || (value == best && expert > best_expert);
}

template <unsigned K>
struct Ranked {
    float chosen[K];
    int winners[K];
};

// Where a selection's keys go (`keys(expert, key)`, each held expert's
// ranked value, before its rounds): nowhere, but in a recording load's
// streamed route (the routing record's keys, streaming/resolve.hpp).
struct NoKeys {
    __device__ __forceinline__ void operator()(unsigned, float) const {}
};

template <unsigned EXPERTS, unsigned K, class Logit, class Keys = NoKeys>
__device__ __forceinline__ Ranked<K> rank_row(Logit logit, unsigned lane, Keys keys = Keys{}) {
    constexpr unsigned per_lane = (EXPERTS + 31) / 32;
    const unsigned held = candidates<EXPERTS>(lane);
    float probability[per_lane];
    float maximum = -__int_as_float(0x7f800000);
#pragma unroll
    for (unsigned i = 0; i < per_lane; ++i) {
        probability[i] = i < held ? logit(lane + 32 * i) : -__int_as_float(0x7f800000);
        maximum = fmaxf(maximum, probability[i]);
    }
    maximum = seismic_warp_max_f32(maximum);
    float total = 0.0f;
#pragma unroll
    for (unsigned i = 0; i < per_lane; ++i) {
        if (i < held) {
            probability[i] = expf(probability[i] - maximum);
            total = seismic_add_rn(total, probability[i]);
        }
    }
    total = seismic_warp_sum_f32(total);
#pragma unroll
    for (unsigned i = 0; i < per_lane; ++i)
        probability[i] = i < held ? ranked(probability[i] / total) : out();
#pragma unroll
    for (unsigned i = 0; i < per_lane; ++i)
        if (i < held)
            keys(lane + 32 * i, probability[i]);

    Ranked<K> result;
#pragma unroll
    for (unsigned rank = 0; rank < K; ++rank) {
        float best = -__int_as_float(0x7f800000);
        int best_expert = -1;
#pragma unroll
        for (unsigned i = 0; i < per_lane; ++i) {
            if (precedes(probability[i], static_cast<int>(lane + 32 * i), best, best_expert)) {
                best = probability[i];
                best_expert = static_cast<int>(lane + 32 * i);
            }
        }
        float winner_probability;
        const int winner = reduce::argmax<reduce::HigherIndex>(best, best_expert, winner_probability);
#pragma unroll
        for (unsigned i = 0; i < per_lane; ++i)
            if (winner == static_cast<int>(lane + 32 * i))
                probability[i] = out();
        result.chosen[rank] = winner_probability;
        result.winners[rank] = winner;
    }
    return result;
}

} // namespace selection
