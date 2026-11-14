// Kimi Delta Attention's rule for the Metal delta-rule libraries
// (`recurrent.h`, `convolution.h`, `chunk.h`; entries `kimi_delta_step` and
// `kimi_delta_chunk`, contracts in recurrent.seismic): its gates and its
// output gate's input, which the libraries' step, chunk and gating take. A
// KDA row has W decay factors per value head (one per key column), beta's
// input is its own tensor, and the output gate's input z is its own tensor.
// Channels of the projection, gate and decay rows and tape rows are
// contiguous (unit stride).

#include "convolution.h"

namespace kimi {

using recurrent::Storage;

// The decay inputs and parameters: the log decay of key channel c of value
// head h at a row is lower * sigmoid(-rate[h] * (a + time_bias[h * W + c])),
// lower = `decay_lower` < 0.
struct Decay {
    device const Storage *input;
    device const float *rate;
    device const float *time_bias;
    float lower;

    float log_at(ulong row, ulong head, ulong channel, constant ulong *seismic_words) const {
        const ulong index = head * SEISMIC_DIM_W + channel;
        const float shifted = element::Act::load(input[row * SEISMIC_DECAY_STRIDE_0 + index])
            + time_bias[index * SEISMIC_TIME_BIAS_STRIDE_0];
        return lower / (1.0f + metal::exp(rate[head * SEISMIC_RATE_STRIDE_0] * shifted));
    }
};

inline Decay decay_of(device const Storage *input, device const float *rate, device const float *time_bias,
    constant ulong *seismic_words) {
    return Decay{input, rate, time_bias, as_type<float>(uint(SEISMIC_PARAM_DECAY_LOWER))};
}

// beta = sigmoid(b) of value head `head` at `row`.
inline float beta_at(device const Storage *beta, ulong row, ulong head, constant ulong *seismic_words) {
    const float input = element::Act::load(beta[row * SEISMIC_BETA_STRIDE_0 + head * SEISMIC_BETA_STRIDE_1]);
    return 1.0f / (1.0f + metal::exp(-input));
}

// Kimi's gates of value head `head` for `recurrent::advance_rows`: beta from
// its own input, one decay factor per key channel.
struct ChannelGates {
    enum : uint { DECAYS = SEISMIC_DIM_W };
    device const Storage *beta_input;
    Decay decay;
    ulong head;

    float at(ulong row, ulong channel, thread float &beta, constant ulong *seismic_words) const {
        if (channel == 0) {
            beta = beta_at(beta_input, row, head, seismic_words);
        }
        return decay.log_at(row, head, channel, seismic_words);
    }
};

// Kimi's rule (`recurrent::HeadRule`'s counterpart): its beta input and decays.
struct ChannelRule {
    enum : uint { DECAYS = ChannelGates::DECAYS };
    device const Storage *beta;
    Decay decay;

    ChannelGates gates(ulong head, constant ulong *) const { return ChannelGates{beta, decay, head}; }
};

inline ChannelRule channel_rule(device const Storage *beta, device const Storage *decay, device const float *rate,
    device const float *time_bias, constant ulong *seismic_words) {
    return ChannelRule{beta, decay_of(decay, rate, time_bias, seismic_words)};
}

// Kimi's z: the gate operand's rows.
inline recurrent::GateInput gate_input(device const Storage *gate, constant ulong *seismic_words) {
    return recurrent::GateInput{gate, SEISMIC_GATE_STRIDE_0};
}

} // namespace kimi
