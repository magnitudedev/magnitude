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

// Loads `N` contiguous F32 values from `src`, using the widest aligned vector
// load (`float4`, `float2`), falling back to scalar loads when unaligned.
template <int N>
__device__ __forceinline__ void load_f32_span(const float *src, float (&out)[N]) {
    if constexpr (N % 4 == 0) {
        if ((reinterpret_cast<u64>(src) & 15u) == 0) {
            element::f32_span<N>(src, out);
            return;
        }
    }
    if constexpr (N % 2 == 0) {
        if ((reinterpret_cast<u64>(src) & 7u) == 0) {
            const float2 *vec = reinterpret_cast<const float2 *>(src);
#pragma unroll
            for (int k = 0; k < N / 2; ++k) {
                const float2 word = vec[k];
                out[2 * k] = word.x;
                out[2 * k + 1] = word.y;
            }
            return;
        }
    }
#pragma unroll
    for (int i = 0; i < N; ++i) out[i] = src[i];
}

// Loads `N` contiguous activation elements from `in.decay` at `(row, head, col0)`
// using `element::span<Act, N, true>` when aligned, falling back to scalar loads.
template <int N>
__device__ __forceinline__ void load_decay_span(const Inputs &in, int row, int head, int col0, float (&out)[N]) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.base.words;
    const u64 offset = static_cast<u64>(row) * SEISMIC_DECAY_STRIDE_0 + static_cast<u64>(head) * W + col0;
    constexpr u64 ALIGN = (N * Act::bytes) % 16 == 0  ? 16u
                          : (N * Act::bytes) % 8 == 0 ? 8u
                          : (N * Act::bytes) % 4 == 0 ? 4u
                                                      : 1u;
    if ((reinterpret_cast<u64>(in.decay + offset * Act::bytes) & (ALIGN - 1)) == 0) {
        element::span<Act, N, true>(in.decay, offset, out);
    } else {
#pragma unroll
        for (int c = 0; c < N; ++c) out[c] = element::at<Act>(in.decay, offset + c);
    }
}

// Loads `N` time_bias entries for value head `head` starting at channel `col0`,
// vectorized when contiguous and aligned.
template <int N>
__device__ __forceinline__ void load_time_bias_span(const Inputs &in, int head, int col0, float (&out)[N]) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.base.words;
    const int index0 = head * W + col0;
    if (SEISMIC_TIME_BIAS_STRIDE_0 == 1) {
        load_f32_span<N>(in.base.time_bias + index0, out);
    } else {
#pragma unroll
        for (int c = 0; c < N; ++c)
            out[c] = in.base.time_bias[static_cast<u64>(index0 + c) * SEISMIC_TIME_BIAS_STRIDE_0];
    }
}

__device__ __forceinline__ float rate_of(const Inputs &in, int head) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.base.words;
    return in.base.rate[head * SEISMIC_RATE_STRIDE_0];
}

__device__ __forceinline__ float log_decay_from(float decay_lower, float rate, float input, float bias) {
    return decay_lower / (1.0f + expf(rate * (input + bias)));
}

// The log decay of key channel `channel` of value head `head` at `row`:
// lower * sigmoid(-rate * (a + bias)). Its factor is exp of it.
__device__ __forceinline__ float log_decay(const Inputs &in, int row, int head, int channel) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.base.words;
    const int index = head * W + channel;
    const float input = element::at<Act>(in.decay, static_cast<u64>(row) * SEISMIC_DECAY_STRIDE_0 + index);
    const float bias = in.base.time_bias[index * SEISMIC_TIME_BIAS_STRIDE_0];
    return log_decay_from(in.decay_lower, rate_of(in, head), input, bias);
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
    __device__ static float rate(const Inputs &in, int head) { return rate_of(in, head); }
    __device__ static float log_decay(const Inputs &in, int row, int head, int channel) {
        return kimi::log_decay(in, row, head, channel);
    }
    __device__ static __forceinline__ void load_time_bias(const Inputs &in, int head, int col0,
                                                          float (&out)[recurrent::CPL]) {
        load_time_bias_span<recurrent::CPL>(in, head, col0, out);
    }
    __device__ static __forceinline__ void log_decay_span(const Inputs &in, int row, int head, int col0, float rate,
                                                          const float (&bias)[recurrent::CPL],
                                                          float (&out)[recurrent::CPL]) {
        float decay_in[recurrent::CPL];
        load_decay_span<recurrent::CPL>(in, row, head, col0, decay_in);
#pragma unroll
        for (int c = 0; c < recurrent::CPL; ++c) out[c] = log_decay_from(in.decay_lower, rate, decay_in[c], bias[c]);
    }
    __device__ __forceinline__ recurrent::GateInput gate_input() const {
        [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.base.words;
        return recurrent::GateInput{in.gate, SEISMIC_GATE_STRIDE_0, 0};
    }
    __device__ __forceinline__ float at(int row, int head, int lane, float (&factor)[recurrent::CPL]) const {
        constexpr int CPL = recurrent::CPL;
        const int col0 = lane * CPL;
        const float rate = rate_of(in, head);
        float decay_in[CPL], bias[CPL];
        load_decay_span<CPL>(in, row, head, col0, decay_in);
        load_time_bias_span<CPL>(in, head, col0, bias);
#pragma unroll
        for (int c = 0; c < CPL; ++c)
            factor[c] = expf(log_decay_from(in.decay_lower, rate, decay_in[c], bias[c]));
        const float b = lane == 0 ? beta_of(in, row, head) : 0.0f;
        return __shfl_sync(0xffffffffu, b, 0);
    }
};

}  // namespace kimi
