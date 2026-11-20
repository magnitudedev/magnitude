// Output selection of the `routed_route` contract, shared by `routed_route`,
// `routed_route_shared` and `routed_route_streamed`. Projection of the router
// and shared-expert gate rows is the `dense` projection operator's
// (`lib/projection/operator.h`).
#include "../core/activation.h"
#include "../core/arrive.h"
#include "ranked.h"

#ifndef PROJECTION_UNROLL
#define PROJECTION_UNROLL _Pragma("clang loop unroll(full)")
#endif

// Logit `expert` of row `row` in `[M, STRIDE]` logits.
template <uint STRIDE = uint(SEISMIC_DIM_E) + 1u>
inline ulong logit_index(ulong row, ulong expert) {
    return row * STRIDE + expert;
}

using Ranked = selection::Ranked<uint(SEISMIC_DIM_K)>;

template <uint EXPERTS = uint(SEISMIC_DIM_E), typename LogitOf, typename Keys = selection::NoKeys>
inline Ranked rank_row(LogitOf logit_of, uint lane, Keys keys = Keys{}) {
    return selection::rank_row<EXPERTS, uint(SEISMIC_DIM_K)>(logit_of, lane, keys);
}

// One row's selection by one simdgroup (`rank_row`, its keys to `keys`),
// optional renormalization, `expert_scale` where present, and the
// shared-expert coefficient from column E.
template <typename LogitOf, typename Keys = selection::NoKeys>
inline void select_row(ulong row, LogitOf logit_of, device const float *expert_scale, device int *routes,
    element::Out<element::F32> scores, element::Out<element::F32> coefficient, constant ulong *seismic_words,
    uint lane, Keys keys = Keys{}) {
    constexpr uint K = SEISMIC_DIM_K;
    const Ranked ranked = rank_row<uint(SEISMIC_DIM_E)>(logit_of, lane, keys);
    float denominator = 0.0f;
    PROJECTION_UNROLL
    for (uint slot = 0; slot < K; ++slot)
        denominator += ranked.chosen[K - 1u - slot];
    if (lane == 0u) {
        PROJECTION_UNROLL
        for (uint slot = 0; slot < K; ++slot) {
            const float probability = ranked.chosen[K - 1u - slot];
            const int winner = ranked.winners[K - 1u - slot];
            routes[row * SEISMIC_ROUTES_STRIDE_0 + slot * SEISMIC_ROUTES_STRIDE_1] = winner;
            float score = SEISMIC_PARAM_NORMALIZE != 0 ? probability / denominator : probability;
            if (SEISMIC_DIM_ES != 0)
                score *= expert_scale[ulong(winner) * SEISMIC_EXPERT_SCALE_STRIDE_0];
            scores.put(row * SEISMIC_SCORES_STRIDE_0 + slot * SEISMIC_SCORES_STRIDE_1, score);
        }
        coefficient.put(row * SEISMIC_RESULT_1_STRIDE_0,
            1.0f / (1.0f + metal::exp(-logit_of(uint(SEISMIC_DIM_E)))));
    }
}
