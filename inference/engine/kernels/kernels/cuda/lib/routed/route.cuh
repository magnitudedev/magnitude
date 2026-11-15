// Output selection of the `routed_route` contract, shared by `routed_route`,
// `routed_route_shared` and `routed_route_streamed`; the CUDA form of
// `metal/lib/routed/route.h`. Projection of the router and shared-expert gate
// rows is the `dense` projection operator's (`lib/projection/operator.cuh`).
#include "../core/activation.cuh"
#include "../core/arrive.cuh"
#include "ranked.cuh"

namespace route {

using Ranked = selection::Ranked<SEISMIC_DIM_K>;

template <unsigned EXPERTS = SEISMIC_DIM_E, class Logit>
__device__ __forceinline__ Ranked rank_row(Logit logit, unsigned lane) {
    return selection::rank_row<EXPERTS, SEISMIC_DIM_K>(logit, lane);
}

struct Selection {
    int *routes;
    unsigned long long route_row, route_slot;
    element::Out<element::F32> scores;
    unsigned long long score_row, score_slot;
    element::Out<element::F32> coefficient;
    unsigned long long coefficient_row;
    bool normalize;
    const float *expert_scale;
    unsigned long long expert_scale_stride;
};

#define ROUTE_SELECTION                                                                                   \
    route::Selection {                                                                                    \
        reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES)), SEISMIC_ROUTES_STRIDE_0,             \
            SEISMIC_ROUTES_STRIDE_1, SEISMIC_OUT_SCORES, SEISMIC_SCORES_STRIDE_0, SEISMIC_SCORES_STRIDE_1, \
            SEISMIC_RESULT_1_OUT, SEISMIC_RESULT_1_STRIDE_0,                                              \
            SEISMIC_PARAM_NORMALIZE != 0,                                                                 \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_SCALE)),                    \
            SEISMIC_EXPERT_SCALE_STRIDE_0                                                                 \
    }

template <class Logit>
__device__ __forceinline__ void select_row(Logit logit, const Selection &out, unsigned long long row, unsigned lane,
                                           const seismic_words_t &seismic_words_value) {
    constexpr unsigned K = SEISMIC_DIM_K;
    const Ranked ranked = rank_row(logit, lane);
    float denominator = 0.0f;
    if (out.normalize)
        for (unsigned slot = 0; slot < K; ++slot)
            denominator = seismic_add_rn(denominator, ranked.chosen[K - 1 - slot]);
    if (lane != 0)
        return;
    for (unsigned slot = 0; slot < K; ++slot) {
        const unsigned rank = K - 1 - slot;
        out.routes[row * out.route_row + slot * out.route_slot] = ranked.winners[rank];
        float score = out.normalize ? ranked.chosen[rank] / denominator : ranked.chosen[rank];
        if (SEISMIC_DIM_ES != 0)
            score = seismic_mul_rn(score,
                out.expert_scale[static_cast<unsigned long long>(ranked.winners[rank]) * out.expert_scale_stride]);
        out.scores.put(row * out.score_row + slot * out.score_slot, score);
    }
    out.coefficient.put(row * out.coefficient_row, 1.0f / (1.0f + expf(-logit(SEISMIC_DIM_E))));
}

} // namespace route
