// Output selection of the `routed_select` contract, shared by `routed_select`
// and `routed_select_streamed`. Projection of the router rows is the `dense`
// projection operator's (`lib/projection/ordered.glsl`). Before including, the
// entry defines SELECT_E (expert count).
#include "../core/precise.glsl"
#include "ranked.glsl"

#define SELECT_K uint(SEISMIC_DIM_K)
#define SELECT_P ((SELECT_E + 31u) / 32u)
#define SELECT_FN(name) name
#include "select.glsl"
SELECT_RANKING

void select_select_row(uint64_t logits_base, uint64_t row) {
    const uint lane = SEISMIC_LANE;
    float chosen[SELECT_K];
    int winners[SELECT_K];
    select_rank_at(logits_base, SEISMIC_PTR(SEISMIC_BUFFER_BIAS), SEISMIC_BIAS_STRIDE_0,
        int(SEISMIC_PARAM_SCORE), row, chosen, winners);
    float denominator = 0.0;
    for (uint slot = 0u; slot < SELECT_K; ++slot)
        denominator += chosen[slot];
    if (lane == 0u) {
        const int normalization = int(SEISMIC_PARAM_NORMALIZATION);
        const float scale = element_word_f32(SEISMIC_PARAM_SCALE);
        const float epsilon = element_word_f32(SEISMIC_PARAM_NORMALIZATION_EPSILON);
        for (uint slot = 0u; slot < SELECT_K; ++slot) {
            float weight = select_normalized(normalization, chosen[slot], denominator, epsilon) * scale;
            if (SEISMIC_DIM_ES != 0ul)
                weight = weight * element_f32_at(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_SCALE)
                    + uint64_t(winners[slot]) * SEISMIC_EXPERT_SCALE_STRIDE_0 * 4ul);
            element_i32_put(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES)
                + (row * SEISMIC_ROUTES_STRIDE_0 + uint64_t(slot) * SEISMIC_ROUTES_STRIDE_1) * 4ul, winners[slot]);
            out_put(SEISMIC_OUT_WEIGHTS,
                row * SEISMIC_WEIGHTS_STRIDE_0 + uint64_t(slot) * SEISMIC_WEIGHTS_STRIDE_1, weight);
        }
    }
}
