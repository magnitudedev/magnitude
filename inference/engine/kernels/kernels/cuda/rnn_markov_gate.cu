// rnn_markov_gate: fused DSpark MarkovHead::RnnHidden recurrent gate step.
// Publishes (h_next [S, R] f32, h_next_a [S, R] A, latent [S, R] A).
#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/core/activation.cuh"

using element::u64;

extern "C" __global__ void rnn_markov_gate(SEISMIC_KERNEL_PARAMS) {
    const u64 index = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
    const u64 s_dim = SEISMIC_DIM_S;
    const u64 r_dim = SEISMIC_DIM_R;
    if (index >= s_dim * r_dim)
        return;
    const u64 s = index / r_dim;
    const u64 r = index % r_dim;
    const u64 r1 = r + r_dim;
    const u64 r2 = r + 2ull * r_dim;

    const float *h_prev = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_H_PREV));
    const float *base_frnn = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_BASE_FRNN));
    const float *base_x = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_BASE_X));
    const float *he_h = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_HE_H));
    const float *he_e = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_HE_E));

    const u64 bf_row = s * SEISMIC_BASE_FRNN_STRIDE_0;
    const u64 bx_row = s * SEISMIC_BASE_X_STRIDE_0;
    const u64 hh_row = s * SEISMIC_HE_H_STRIDE_0;
    const u64 he_row = s * SEISMIC_HE_E_STRIDE_0;

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
    const float gate = 1.0f / (1.0f + expf(-g0));
    const float cand = 2.0f / (1.0f + expf(-2.0f * g1)) - 1.0f;
    const float next = gate * hp + (1.0f - gate) * cand;
    const float lat = 2.0f / (1.0f + expf(-2.0f * g2)) - 1.0f;

    SEISMIC_RESULT_0_OUT.put(s * SEISMIC_RESULT_0_STRIDE_0 + r * SEISMIC_RESULT_0_STRIDE_1, next);
    SEISMIC_RESULT_1_OUT.put(s * SEISMIC_RESULT_1_STRIDE_0 + r * SEISMIC_RESULT_1_STRIDE_1, next);
    SEISMIC_RESULT_2_OUT.put(s * SEISMIC_RESULT_2_STRIDE_0 + r * SEISMIC_RESULT_2_STRIDE_1, lat);
}
