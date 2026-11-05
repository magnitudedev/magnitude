// The decode selection of the `routed_select` contract, shared by the
// `routed_select` and `routed_select_streamed` entries: both bind its
// parameters (`residual`, `norm`, `router_norm`, `router`, `bias`,
// `expert_scale` of static extent ES, `routes`, `weights`, `epsilon`,
// `score`, `normalization`, `normalization_epsilon`, `scale`), its result
// (`normalized`) and SIMDGROUPS under the same names. The includer defines
// KERNEL_W0 as the router.
//   decode_inverses: a GEMV threadgroup's RMS inverses of the decode rows
//     (M <= DECODE_ROWS), in the stage launch's order.
//   publish_normalized: the normalized rows (`norm`), by one threadgroup.
//   select_columns: SIMDGROUPS logit columns of the decode rows over a
//     router norm and router (`routing::Gemv`).
//   select_row: one row's choices by one simdgroup (`select::rank`), their
//     normalization and scales into `routes` and `weights`.

#include "router.h"
#include "select.h"

typedef element::Act Act;
typedef ELEMENT_OF(SEISMIC_NORM) Norm;
typedef ELEMENT_OF(SEISMIC_ROUTER_NORM) RouterNorm;
typedef routing::Packets<packets::W0>::type Router;

constant constexpr uint select_threads = routing::stage_threads;
constant constexpr uint select_simdgroups = routing::stage_simdgroups;

inline void decode_inverses(device const float *residual, threadgroup float *parts, threadgroup float *inverses,
    constant ulong *seismic_words, uint simd, uint lane, uint tid) {
    routing::inverses(residual, SEISMIC_RESIDUAL_STRIDE_0, SEISMIC_RESIDUAL_STRIDE_1, uint(SEISMIC_DIM_H),
        as_type<float>(uint(SEISMIC_PARAM_EPSILON)), uint(SEISMIC_DIM_M), parts, inverses, SEISMIC_TUNE_SIMDGROUPS,
        simd, lane, tid);
}

inline void publish_normalized(device const float *residual, device const uchar *norm, device uchar *normalized,
    threadgroup const float *inverses, constant ulong *seismic_words, uint tid) {
    for (uint item = tid; item < uint(SEISMIC_DIM_M) * uint(SEISMIC_DIM_H); item += 32 * SEISMIC_TUNE_SIMDGROUPS) {
        const ulong row = item / SEISMIC_DIM_H, source = item % SEISMIC_DIM_H;
        const float value = residual[row * SEISMIC_RESIDUAL_STRIDE_0 + source * SEISMIC_RESIDUAL_STRIDE_1];
        element::put<Act>(normalized, row * SEISMIC_RESULT_0_STRIDE_0 + source * SEISMIC_RESULT_0_STRIDE_1,
            Act::round(value * inverses[row] * element::at<Norm>(norm, source * SEISMIC_NORM_STRIDE_0)));
    }
}

// The logits of group `group`'s SIMDGROUPS router rows over the router input
// round_A(r * inverse * router_norm), into `logits` ([M, EXPERTS], a router
// of EXPERTS experts: the entry's own by default).
template <uint EXPERTS = uint(SEISMIC_DIM_E)>
inline void select_columns(device const float *residual, device const uchar *router_norm, ulong router_norm0,
    device const uchar *router, device float *logits, threadgroup const float *inverses, uint group,
    threadgroup uchar *shared, constant ulong *seismic_words, uint simd, uint lane) {
    const uint k = uint(SEISMIC_DIM_H);
    const routing::Rows<RouterNorm> in{residual, SEISMIC_RESIDUAL_STRIDE_0, SEISMIC_RESIDUAL_STRIDE_1, router_norm,
        router_norm0, inverses, k};
    const projection::Weights<Router> weights{router, KERNEL_W0_LAYOUT(k), k, nullptr};
    const routing::Logits out{logits, EXPERTS, 0};
    routing::Gemv<Act::bytes>::columns(in, weights, out, uint(SEISMIC_DIM_M), EXPERTS, group, shared,
        SEISMIC_TUNE_SIMDGROUPS, simd, lane);
}

// One row's selection by one simdgroup, stored by lane 0.
inline void select_row(ulong row, device const float *logits, device const float *bias,
    device const float *expert_scale, device int *routes, device float *weights, constant ulong *seismic_words,
    uint lane) {
    constexpr uint K = SEISMIC_DIM_K;
    const int normalization = int(SEISMIC_PARAM_NORMALIZATION);
    const select::Selected<K> selected = select::rank<uint(SEISMIC_DIM_E), K>(int(SEISMIC_PARAM_SCORE),
        [&](uint expert) { return logits[row * SEISMIC_DIM_E + expert]; },
        [&](int expert) { return bias[ulong(expert) * SEISMIC_BIAS_STRIDE_0]; }, lane);
    // The slot-order sum.
    float denominator = 0.0f;
    PROJECTION_UNROLL
    for (uint slot = 0; slot < K; ++slot)
        denominator += selected.chosen[slot];
    if (lane == 0) {
        const float scale = as_type<float>(uint(SEISMIC_PARAM_SCALE));
        const float epsilon = as_type<float>(uint(SEISMIC_PARAM_NORMALIZATION_EPSILON));
        PROJECTION_UNROLL
        for (uint slot = 0; slot < K; ++slot) {
            float weight = select::normalized(normalization, selected.chosen[slot], denominator, epsilon) * scale;
            if (SEISMIC_DIM_ES != 0)
                weight *= expert_scale[ulong(selected.winners[slot]) * SEISMIC_EXPERT_SCALE_STRIDE_0];
            routes[row * SEISMIC_ROUTES_STRIDE_0 + slot * SEISMIC_ROUTES_STRIDE_1] = selected.winners[slot];
            weights[row * SEISMIC_WEIGHTS_STRIDE_0 + slot * SEISMIC_WEIGHTS_STRIDE_1] = weight;
        }
    }
}
