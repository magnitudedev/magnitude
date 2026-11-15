// Output selection of the `routed_select` contract, shared by `routed_select`,
// `routed_select_streamed` and `routed_hint_select`. Projection of the router
// rows is the `dense` projection operator's (`lib/projection/operator.h`).
#include "../core/activation.h"
#include "../core/arrive.h"
#include "select.h"

#ifndef PROJECTION_UNROLL
#define PROJECTION_UNROLL _Pragma("clang loop unroll(full)")
#endif

// One row's selection by one simdgroup (`select::rank`), stored by lane 0.
template <typename Logits>
inline void select_row(ulong row, Logits logits, device const float *bias,
    device const float *expert_scale, device int *routes, element::Out<element::F32> weights,
    constant ulong *seismic_words, uint lane) {
    constexpr uint K = SEISMIC_DIM_K;
    const int normalization = int(SEISMIC_PARAM_NORMALIZATION);
    const select::Selected<K> selected = select::rank<uint(SEISMIC_DIM_E), K>(int(SEISMIC_PARAM_SCORE),
        [&](uint expert) { return logits[row * SEISMIC_DIM_E + expert]; },
        [&](int expert) { return bias[ulong(expert) * SEISMIC_BIAS_STRIDE_0]; }, lane);
    float denominator = 0.0f;
    PROJECTION_UNROLL
    for (uint slot = 0; slot < K; ++slot)
        denominator += selected.chosen[slot];
    if (lane == 0u) {
        const float scale = as_type<float>(uint(SEISMIC_PARAM_SCALE));
        const float epsilon = as_type<float>(uint(SEISMIC_PARAM_NORMALIZATION_EPSILON));
        PROJECTION_UNROLL
        for (uint slot = 0; slot < K; ++slot) {
            float weight = select::normalized(normalization, selected.chosen[slot], denominator, epsilon) * scale;
            if (SEISMIC_DIM_ES != 0)
                weight *= expert_scale[ulong(selected.winners[slot]) * SEISMIC_EXPERT_SCALE_STRIDE_0];
            routes[row * SEISMIC_ROUTES_STRIDE_0 + slot * SEISMIC_ROUTES_STRIDE_1] = selected.winners[slot];
            weights.put(row * SEISMIC_WEIGHTS_STRIDE_0 + slot * SEISMIC_WEIGHTS_STRIDE_1, weight);
        }
    }
}
