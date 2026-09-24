// Shared device code of the CUDA gated-delta entries (contracts in
// recurrent_stages.seismic): tensor addressing, the per-channel causal
// convolution + SiLU of one row, the per-head gates, and the state-arena
// window publication. Activation tensors are canonical in their last axis.

#include "common/mixer.cuh"

namespace rec {

using mx::u8;
using mx::u32;
using mx::u64;

constexpr int NK = static_cast<int>(SEISMIC_DIM_NK);
constexpr int NV = static_cast<int>(SEISMIC_DIM_NV);
constexpr int W = static_cast<int>(SEISMIC_DIM_W);
constexpr int C = static_cast<int>(SEISMIC_DIM_C);
constexpr int CH = (2 * NK + NV) * W;
// Columns of the projection's gate segments.
constexpr int ALPHA = CH + NV * W;
constexpr int BETA = ALPHA + NV;

struct Inputs {
    // The launch's argument words, read by the ABI stride macros.
    const seismic_words_t *words;
    const u8 *projection;
    const float *convolution;
    const float *rate;
    const float *time_bias;
    const int *segments;
    const int *stop;
    const int *previous_bank;
    const int *following_bank;
    u8 *window;
    float *delta;
    float epsilon;
    bool grouped;
};

#define REC_INPUTS()                                                                          \
    rec::Inputs {                                                                             \
        &seismic_words_value, SEISMIC_PTR(SEISMIC_BUFFER_PROJECTION),                         \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_CONVOLUTION)),         \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RATE)),                \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_TIME_BIAS)),           \
            reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_SEGMENTS)),              \
            reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_STOP)),                  \
            reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_PREVIOUS_BANK)),         \
            reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_FOLLOWING_BANK)),        \
            SEISMIC_PTR(SEISMIC_BUFFER_WINDOW),                                               \
            reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_DELTA)),                     \
            mx::word_f32(SEISMIC_PARAM_NORM_EPSILON), SEISMIC_PARAM_GROUPED != 0              \
    }

// One slot's rows [lo, hi), its banks and publication row count.
struct Slot {
    int lo;
    int hi;
    int stop;
    int source;
    int target;
};
__device__ __forceinline__ Slot slot_of(const Inputs &in, u64 slot) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    return Slot{in.segments[slot * SEISMIC_SEGMENTS_STRIDE_0],
                in.segments[slot * SEISMIC_SEGMENTS_STRIDE_0 + SEISMIC_SEGMENTS_STRIDE_1],
                in.stop[slot * SEISMIC_STOP_STRIDE_0],
                in.previous_bank[slot * SEISMIC_PREVIOUS_BANK_STRIDE_0],
                in.following_bank[slot * SEISMIC_FOLLOWING_BANK_STRIDE_0]};
}

// Rows [first, last) that no slot covers: slots partition a prefix of rows.
__device__ __forceinline__ int covered_end(const Inputs &in) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    const u64 slots = SEISMIC_DIM_B;
    return slots == 0 ? 0 : in.segments[(slots - 1) * SEISMIC_SEGMENTS_STRIDE_0 +
                                        SEISMIC_SEGMENTS_STRIDE_1];
}

__device__ __forceinline__ float projection(const Inputs &in, u64 row, u64 column) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    return mx::act_load(in.projection, row * SEISMIC_PROJECTION_STRIDE_0 + column);
}

__device__ __forceinline__ u64 window_at(const Inputs &in, int bank, int tap, u64 channel) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    return static_cast<u64>(bank) * SEISMIC_WINDOW_STRIDE_0 +
           static_cast<u64>(tap) * SEISMIC_WINDOW_STRIDE_1 + channel;
}

// State row `state_row` of value head `head` in `bank`: W contiguous floats.
__device__ __forceinline__ float *state_row(const Inputs &in, int bank, int head, int state_row) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    return in.delta + static_cast<u64>(bank) * SEISMIC_DELTA_STRIDE_0 +
           static_cast<u64>(head) * SEISMIC_DELTA_STRIDE_1 +
           static_cast<u64>(state_row) * SEISMIC_DELTA_STRIDE_2;
}

// Causal depthwise convolution of `channel` at slot-local row `local` (row
// `row` of the batch), then SiLU. Taps before the slot's first row come from
// the source window.
__device__ __forceinline__ float convolved(const Inputs &in, const Slot &slot, int row,
                                           int channel) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    const float *weights =
        in.convolution + static_cast<u64>(channel) * SEISMIC_CONVOLUTION_STRIDE_0;
    const int local = row - slot.lo;
    float sum = weights[(C - 1) * SEISMIC_CONVOLUTION_STRIDE_1] * projection(in, row, channel);
#pragma unroll
    for (int tap = 0; tap < C - 1; ++tap) {
        const float previous =
            local + tap < C - 1
                ? mx::act_load(in.window, window_at(in, slot.source, local + tap, channel))
                : projection(in, row + tap - (C - 1), channel);
        sum = __fmaf_rn(weights[tap * SEISMIC_CONVOLUTION_STRIDE_1], previous, sum);
    }
    return sum / (1.0f + expf(-sum));
}

// The key head of value head `head`.
__device__ __forceinline__ int key_head(const Inputs &in, int head) {
    return in.grouped ? head * NK / NV : head % NK;
}

// beta = sigmoid(b) and the decay exp(log_decay), log_decay =
// rate * softplus(alpha + time_bias).
struct Gates {
    float beta;
    float log_decay;
    float decay;
};
__device__ __forceinline__ Gates gates(const Inputs &in, int row, int head) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    const float alpha = projection(in, row, ALPHA + head);
    const float beta_input = projection(in, row, BETA + head);
    const float shifted = alpha + in.time_bias[head * SEISMIC_TIME_BIAS_STRIDE_0];
    const float softplus = fmaxf(shifted, 0.0f) + logf(1.0f + expf(-fabsf(shifted)));
    const float log_decay = in.rate[head * SEISMIC_RATE_STRIDE_0] * softplus;
    return Gates{1.0f / (1.0f + expf(-beta_input)), log_decay, expf(log_decay)};
}

// A slot's pieces: its rows split into runs of at most `CHUNK` rows, with a
// run boundary at the publication row lo + stop.
template <int CHUNK>
__device__ __forceinline__ int piece_count(const Slot &slot) {
    const int length = slot.hi - slot.lo;
    return (slot.stop + CHUNK - 1) / CHUNK + (length - slot.stop + CHUNK - 1) / CHUNK;
}
// Rows [first, first + count) of piece `index` of the slot.
struct Piece {
    int first;
    int count;
};
template <int CHUNK>
__device__ __forceinline__ Piece piece_of(const Slot &slot, int index) {
    const int before = (slot.stop + CHUNK - 1) / CHUNK;
    if (index < before) {
        const int first = index * CHUNK;
        return Piece{slot.lo + first, min(CHUNK, slot.stop - first)};
    }
    const int first = slot.stop + (index - before) * CHUNK;
    return Piece{slot.lo + first, min(CHUNK, slot.hi - slot.lo - first)};
}

// Publish the slot's successor window: the C - 1 raw rows before the
// publication row. `part` of `parts` cooperating blocks writes an even share.
__device__ __forceinline__ void publish_window(const Inputs &in, const Slot &slot, int part,
                                               int parts) {
    constexpr int total = (C - 1) * CH;
    const int per = (total + parts - 1) / parts;
    const int first = part * per;
    const int last = min(total, first + per);
    const int publish = slot.lo + slot.stop;
    for (int index = first + threadIdx.x; index < last; index += blockDim.x) {
        const int tap = index / CH;
        const int channel = index % CH;
        const float value =
            slot.stop + tap < C - 1
                ? mx::act_load(in.window, window_at(in, slot.source, slot.stop + tap, channel))
                : projection(in, publish + tap - (C - 1), channel);
        mx::act_store(in.window, window_at(in, slot.target, tap, channel), value);
    }
}

}  // namespace rec
