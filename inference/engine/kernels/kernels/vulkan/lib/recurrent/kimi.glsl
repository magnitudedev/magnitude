// Kimi Delta Attention's rule for the Vulkan delta-rule library
// (`recurrent.glsl`; entries `kimi_delta_step` and `kimi_delta_chunk`,
// contracts in recurrent.seismic): beta's input, the output gate's input z
// and the decay inputs are their own operands, and each key column of a value
// head has its own decay factor, so a tape row records W per value head.
#define RECURRENT_DECAYS RECURRENT_W
#define RECURRENT_GATE_FIRST 0u
#include "recurrent.glsl"

// The decay factor exp(log d) of key channel `channel` of value head `head`
// at `row`, log d = decay_lower * sigmoid(-rate * (a + time_bias)), and for
// channel 0 beta = sigmoid(b).
void recurrent_row_gates(recurrent_inputs in_, uint64_t row, uint head, uint channel, out float beta,
    out float decay) {
    beta = channel == 0u ? seismic_div_rn(1.0, 1.0 + exp(-element_at(ELEMENT_ACT, SEISMIC_PTR(SEISMIC_BUFFER_BETA),
        row * SEISMIC_BETA_STRIDE_0 + uint64_t(head) * SEISMIC_BETA_STRIDE_1))) : 0.0;
    const uint index = head * RECURRENT_W + channel;
    const float a = element_at(ELEMENT_ACT, SEISMIC_PTR(SEISMIC_BUFFER_DECAY),
        row * SEISMIC_DECAY_STRIDE_0 + uint64_t(index) * SEISMIC_DECAY_STRIDE_1);
    const float shifted = a + element_f32_at(in_.time_bias + uint64_t(index) * SEISMIC_TIME_BIAS_STRIDE_0 * 4ul);
    const float rate = element_f32_at(in_.rate + uint64_t(head) * SEISMIC_RATE_STRIDE_0 * 4ul);
    decay = exp(seismic_div_rn(element_word_f32(SEISMIC_PARAM_DECAY_LOWER), 1.0 + exp(rate * shifted)));
}

// z column `column` of the gate row `row`.
float recurrent_gate_input(recurrent_inputs in_, uint64_t row, uint64_t column) {
    return element_at(ELEMENT_ACT, SEISMIC_PTR(SEISMIC_BUFFER_GATE), row * SEISMIC_GATE_STRIDE_0 + column);
}
