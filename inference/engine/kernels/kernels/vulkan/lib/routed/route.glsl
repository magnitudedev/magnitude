// Output selection of the `routed_route` contract, shared by `routed_route`,
// `routed_route_shared` and `routed_route_streamed`. Projection of the router
// and shared-expert gate rows is the `dense` projection operator's
// (`lib/projection/ordered.glsl`). Before including, the entry defines ROUTE_E
// (expert count) and ROUTE_STRIDE (logits row stride: E + 1).
#include "ranked.glsl"

#define ROUTE_K uint(SEISMIC_DIM_K)
#define ROUTE_PER_LANE ((ROUTE_E + 31u) / 32u)
ROUTE_DEFINE_RANK_AT

void route_select_row(uint64_t logits_base, uint64_t row) {
    const uint lane = SEISMIC_LANE;
    const uint64_t routes = SEISMIC_PTR(SEISMIC_BUFFER_ROUTES);
    const Out scores = SEISMIC_OUT_SCORES;
    float chosen[ROUTE_K];
    int winners[ROUTE_K];
    route_rank_at(logits_base, row, chosen, winners);
    if (lane == 0u)
        for (uint slot = 0u; slot < ROUTE_K; ++slot)
            element_i32_put(routes + (row * SEISMIC_ROUTES_STRIDE_0 + uint64_t(slot) * SEISMIC_ROUTES_STRIDE_1) * 4ul,
                winners[ROUTE_K - 1u - slot]);
    float denominator = 0.0;
    if (SEISMIC_PARAM_NORMALIZE != 0ul)
        for (uint slot = 0u; slot < ROUTE_K; ++slot)
            denominator += chosen[ROUTE_K - 1u - slot];
    if (lane == 0u) {
        for (uint slot = 0u; slot < ROUTE_K; ++slot) {
            const uint rank = ROUTE_K - 1u - slot;
            float score = SEISMIC_PARAM_NORMALIZE != 0ul ? seismic_div_rn(chosen[rank], denominator) : chosen[rank];
            if (SEISMIC_DIM_ES != 0ul)
                score = seismic_mul_rn(score, element_f32_at(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_SCALE)
                    + uint64_t(winners[rank]) * SEISMIC_EXPERT_SCALE_STRIDE_0 * 4ul));
            out_put(scores, row * SEISMIC_SCORES_STRIDE_0 + uint64_t(slot) * SEISMIC_SCORES_STRIDE_1, score);
        }
        const float gate_logit = element_f32_at(logits_base + (row * uint64_t(ROUTE_STRIDE) + uint64_t(ROUTE_E)) * 4ul);
        out_put(SEISMIC_RESULT_1_OUT, row * SEISMIC_RESULT_1_STRIDE_0,
            seismic_div_rn(1.0, 1.0 + exp(-gate_logit)));
    }
}
