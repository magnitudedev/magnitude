// rnn_markov_gate: fused DSpark MarkovHead::RnnHidden recurrent gate step.
// Publishes (h_next [S, R] f32, h_next_a [S, R] A, latent [S, R] A).
#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/core/activation.h"

typedef activation::Act activation_element;

kernel void rnn_markov_gate(
    device const float *h_prev [[buffer(SEISMIC_BUFFER_H_PREV)]],
    device const float *base_frnn [[buffer(SEISMIC_BUFFER_BASE_FRNN)]],
    device const float *base_x [[buffer(SEISMIC_BUFFER_BASE_X)]],
    device const float *he_h [[buffer(SEISMIC_BUFFER_HE_H)]],
    device const float *he_e [[buffer(SEISMIC_BUFFER_HE_E)]],
    SEISMIC_RESULT_0_PARAM(h_next_cells),
    SEISMIC_RESULT_1_PARAM(h_next_a_cells),
    SEISMIC_RESULT_2_PARAM(latent_cells),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint index [[thread_position_in_grid]]) {
    const ulong s_dim = SEISMIC_DIM_S;
    const ulong r_dim = SEISMIC_DIM_R;
    if (ulong(index) >= s_dim * r_dim)
        return;
    const ulong s = ulong(index) / r_dim;
    const ulong r = ulong(index) % r_dim;
    const ulong r1 = r + r_dim;
    const ulong r2 = r + 2ul * r_dim;

    const ulong bf_row = s * SEISMIC_BASE_FRNN_STRIDE_0;
    const ulong bx_row = s * SEISMIC_BASE_X_STRIDE_0;
    const ulong hh_row = s * SEISMIC_HE_H_STRIDE_0;
    const ulong he_row = s * SEISMIC_HE_E_STRIDE_0;

    const float g0 = base_frnn[bf_row + r * SEISMIC_BASE_FRNN_STRIDE_1]
                   + base_x[bx_row + r * SEISMIC_BASE_X_STRIDE_1]
                   + he_h[hh_row + r * SEISMIC_HE_H_STRIDE_1]
                   + he_e[he_row + r * SEISMIC_HE_E_STRIDE_1];
    const float g1 = base_frnn[bf_row + r1 * SEISMIC_BASE_FRNN_STRIDE_1]
                   + base_x[bx_row + r1 * SEISMIC_BASE_X_STRIDE_1]
                   + he_h[hh_row + r1 * SEISMIC_HE_H_STRIDE_1]
                   + he_e[he_row + r1 * SEISMIC_HE_E_STRIDE_1];
    const float g2 = base_frnn[bf_row + r2 * SEISMIC_BASE_FRNN_STRIDE_1]
                   + base_x[bx_row + r2 * SEISMIC_BASE_X_STRIDE_1]
                   + he_h[hh_row + r2 * SEISMIC_HE_H_STRIDE_1]
                   + he_e[he_row + r2 * SEISMIC_HE_E_STRIDE_1];

    const float hp = h_prev[s * SEISMIC_H_PREV_STRIDE_0 + r * SEISMIC_H_PREV_STRIDE_1];
    const float gate = 1.0f / (1.0f + metal::exp(-g0));
    const float cand = 2.0f / (1.0f + metal::exp(-2.0f * g1)) - 1.0f;
    const float next = gate * hp + (1.0f - gate) * cand;
    const float lat = 2.0f / (1.0f + metal::exp(-2.0f * g2)) - 1.0f;

    const element::Out<element::F32> h_next(h_next_cells);
    const element::Out<activation_element> h_next_a(h_next_a_cells);
    const element::Out<activation_element> latent(latent_cells);
    h_next.put(s * SEISMIC_RESULT_0_STRIDE_0 + r * SEISMIC_RESULT_0_STRIDE_1, next);
    h_next_a.put(s * SEISMIC_RESULT_1_STRIDE_0 + r * SEISMIC_RESULT_1_STRIDE_1, next);
    latent.put(s * SEISMIC_RESULT_2_STRIDE_0 + r * SEISMIC_RESULT_2_STRIDE_1, lat);
}
