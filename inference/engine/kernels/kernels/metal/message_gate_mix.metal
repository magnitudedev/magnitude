// message_gate_mix: `(1 - tanh(forget)) * dest + tanh(gain) * msg`, F32.
#include "lib/core/activation.h"

kernel void message_gate_mix(
    device const float *dest [[buffer(SEISMIC_BUFFER_DEST)]],
    device const float *msg [[buffer(SEISMIC_BUFFER_MSG)]],
    device const float *gain [[buffer(SEISMIC_BUFFER_GAIN)]],
    device const float *forget [[buffer(SEISMIC_BUFFER_FORGET)]],
    SEISMIC_RESULT_0_PARAM(result_cells),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint index [[thread_position_in_grid]]) {
    const ulong total = SEISMIC_DIM_M * SEISMIC_DIM_H;
    if (ulong(index) >= total)
        return;
    const ulong row = ulong(index) / SEISMIC_DIM_H;
    const ulong col = ulong(index) % SEISMIC_DIM_H;
    const float d = dest[row * SEISMIC_DEST_STRIDE_0 + col * SEISMIC_DEST_STRIDE_1];
    const float m = msg[row * SEISMIC_MSG_STRIDE_0 + col * SEISMIC_MSG_STRIDE_1];
    const float g = gain[row * SEISMIC_GAIN_STRIDE_0 + col * SEISMIC_GAIN_STRIDE_1];
    const float f = forget[row * SEISMIC_FORGET_STRIDE_0 + col * SEISMIC_FORGET_STRIDE_1];
    const float tf = 2.0f / (1.0f + metal::exp(-2.0f * f)) - 1.0f;
    const float tg = 2.0f / (1.0f + metal::exp(-2.0f * g)) - 1.0f;
    const element::Out<element::F32> result(result_cells);
    result.put(row * SEISMIC_RESULT_0_STRIDE_0 + col * SEISMIC_RESULT_0_STRIDE_1,
        (1.0f - tf) * d + tg * m);
}
