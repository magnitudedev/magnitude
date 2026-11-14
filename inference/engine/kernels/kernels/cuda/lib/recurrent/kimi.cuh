// Kimi Delta Attention's rule for the CUDA delta-rule library
// (`recurrent.cuh`; entries `kimi_delta_step` and `kimi_delta_chunk`,
// contracts in recurrent.seismic): beta, z and the decay inputs as their own
// operands and one bounded decay per key channel, the gates the library's
// entry bodies take (`ChannelGates`).

#include "recurrent.cuh"

namespace kimi {

using recurrent::Act;
using recurrent::u8;
using recurrent::u64;
using recurrent::W;

struct Inputs {
    // The operands the gated-delta library reads; its projection is the
    // q | k | v row.
    recurrent::Inputs base;
    const u8 *beta;
    const u8 *gate;
    const u8 *decay;
    // The bounded sigmoid's lower bound (< 0).
    float decay_lower;
};

#define KIMI_INPUTS()                                                                                \
    kimi::Inputs {                                                                                   \
        RECURRENT_INPUTS(), SEISMIC_PTR(SEISMIC_BUFFER_BETA), SEISMIC_PTR(SEISMIC_BUFFER_GATE),      \
            SEISMIC_PTR(SEISMIC_BUFFER_DECAY), element::word_f32(SEISMIC_PARAM_DECAY_LOWER)          \
    }

// beta = sigmoid(b) of value head `head` at `row`.
__device__ __forceinline__ float beta_of(const Inputs &in, int row, int head) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.base.words;
    const float input = element::at<Act>(in.beta, static_cast<u64>(row) * SEISMIC_BETA_STRIDE_0 + head);
    return 1.0f / (1.0f + expf(-input));
}

// The log decay of key channel `channel` of value head `head` at `row`:
// lower * sigmoid(-rate * (a + bias)). Its factor is exp of it.
__device__ __forceinline__ float log_decay(const Inputs &in, int row, int head, int channel) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.base.words;
    const int index = head * W + channel;
    const float input = element::at<Act>(in.decay, static_cast<u64>(row) * SEISMIC_DECAY_STRIDE_0 + index);
    const float shifted = input + in.base.time_bias[index * SEISMIC_TIME_BIAS_STRIDE_0];
    const float rate = in.base.rate[head * SEISMIC_RATE_STRIDE_0];
    return in.decay_lower / (1.0f + expf(rate * shifted));
}

// Kimi's gates (`recurrent::HeadGates`' interface): beta from its own input,
// one decay factor per key channel, z the gate operand's rows; `beta` and
// `log_decay` for the chunked form.
struct ChannelGates {
    static constexpr int DECAYS = W;
    using Inputs = kimi::Inputs;
    const Inputs &in;
    __device__ static const recurrent::Inputs &base(const Inputs &in) { return in.base; }
    __device__ static float beta(const Inputs &in, int row, int head) { return beta_of(in, row, head); }
    __device__ static float log_decay(const Inputs &in, int row, int head, int channel) {
        return kimi::log_decay(in, row, head, channel);
    }
    __device__ __forceinline__ recurrent::GateInput gate_input() const {
        [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.base.words;
        return recurrent::GateInput{in.gate, SEISMIC_GATE_STRIDE_0, 0};
    }
    __device__ __forceinline__ float at(int row, int head, int lane, float (&factor)[recurrent::CPL]) const {
#pragma unroll
        for (int c = 0; c < recurrent::CPL; ++c)
            factor[c] = expf(kimi::log_decay(in, row, head, lane * recurrent::CPL + c));
        return beta_of(in, row, head);
    }
};

}  // namespace kimi
