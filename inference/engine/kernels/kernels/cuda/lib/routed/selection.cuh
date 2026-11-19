// Output selection of the `routed_select` contract, shared by `routed_select`,
// `routed_select_streamed` and `routed_hint_select`; the CUDA form of
// `metal/lib/routed/selection.h`. Projection of the router rows is the `dense`
// projection operator's (`lib/projection/operator.cuh`).
#include "../core/activation.cuh"
#include "../core/arrive.cuh"
#include "select.cuh"

// One row's selection by one warp (`select::rank_row`, its keys to `keys`).
template <class Logit, class Keys = selection::NoKeys>
__device__ __forceinline__ void select_row(unsigned long long row, Logit logit, const float *bias,
                                           const float *expert_scale, int *routes,
                                           const element::Out<element::F32> &weights,
                                           const seismic_words_t &seismic_words_value, Keys keys = Keys{}) {
    constexpr unsigned K = SEISMIC_DIM_K;
    const unsigned lane = threadIdx.x % 32;
    const select::Selected<K> selected = select::rank_row<static_cast<unsigned>(SEISMIC_DIM_E), K>(
        static_cast<int>(SEISMIC_PARAM_SCORE),
        [&](unsigned expert) { return logit(expert); },
        [&](int expert) { return bias[static_cast<unsigned long long>(expert) * SEISMIC_BIAS_STRIDE_0]; }, lane,
        keys);
    float denominator = 0.0f;
    for (unsigned slot = 0; slot < K; ++slot)
        denominator = seismic_add_rn(denominator, selected.chosen[slot]);
    if (lane == 0) {
        const int normalization = static_cast<int>(SEISMIC_PARAM_NORMALIZATION);
        const float scale = __uint_as_float(static_cast<unsigned>(SEISMIC_PARAM_SCALE));
        const float epsilon = __uint_as_float(static_cast<unsigned>(SEISMIC_PARAM_NORMALIZATION_EPSILON));
        for (unsigned slot = 0; slot < K; ++slot) {
            float weight = seismic_mul_rn(select::normalized(normalization, selected.chosen[slot], denominator, epsilon), scale);
            if (SEISMIC_DIM_ES != 0)
                weight = seismic_mul_rn(weight,
                    expert_scale[static_cast<unsigned long long>(selected.winners[slot]) * SEISMIC_EXPERT_SCALE_STRIDE_0]);
            routes[row * SEISMIC_ROUTES_STRIDE_0 + slot * SEISMIC_ROUTES_STRIDE_1] = selected.winners[slot];
            weights.put(row * SEISMIC_WEIGHTS_STRIDE_0 + slot * SEISMIC_WEIGHTS_STRIDE_1, weight);
        }
    }
}
