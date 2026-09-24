// Shared device code of the CUDA gated-delta entries (`qwen_recurrent_step`,
// `qwen_recurrent_chunk`; contracts in recurrent.seismic): slot and
// bank lookup, tensor addressing, the per-head gates, piece splitting, the
// state-arena window publication, and the row-sequential advance both entries
// run for slots of at most SEQUENTIAL_ROWS rows.
// Activation tensors are canonical in their last axis.

#include "common/element.cuh"

namespace recurrent {

using element::u8;
using element::u16;
using element::u32;
using element::u64;
typedef element::Act Act;

constexpr int NK = static_cast<int>(SEISMIC_DIM_NK);
constexpr int NV = static_cast<int>(SEISMIC_DIM_NV);
constexpr int W = static_cast<int>(SEISMIC_DIM_W);
constexpr int C = static_cast<int>(SEISMIC_DIM_C);
constexpr int CH = (2 * NK + NV) * W;
// Columns of the projection's gate segments.
constexpr int ALPHA = CH + NV * W;
constexpr int BETA = ALPHA + NV;
// A tape row: the innovations u [NV, W], the normalized keys k [NK, W], the
// decays d [NV].
constexpr int TAPE_U = 0;
constexpr int TAPE_K = NV * W;
constexpr int TAPE_D = NV * W + NK * W;

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
    const int *previous_tape;
    const int *following_bank;
    u8 *window;
    float *delta;
    float *tape;
    float epsilon;
    bool grouped;
};

#define RECURRENT_INPUTS()                                                                    \
    recurrent::Inputs {                                                                       \
        &seismic_words_value, SEISMIC_PTR(SEISMIC_BUFFER_PROJECTION),                         \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_CONVOLUTION)),         \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RATE)),                \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_TIME_BIAS)),           \
            reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_SEGMENTS)),              \
            reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_STOP)),                  \
            reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_PREVIOUS_BANK)),         \
            reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_PREVIOUS_TAPE)),         \
            reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_FOLLOWING_BANK)),        \
            SEISMIC_PTR(SEISMIC_BUFFER_WINDOW),                                               \
            reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_DELTA)),                     \
            reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_TAPE)),                      \
            element::word_f32(SEISMIC_PARAM_NORM_EPSILON), SEISMIC_PARAM_GROUPED != 0         \
    }

// One slot's rows [lo, hi), its publication row count, the version it reads
// (bank `source` advanced by its first `taped` tape rows) and its successor.
struct Slot {
    int lo;
    int hi;
    int stop;
    int source;
    int taped;
    int target;
};
__device__ __forceinline__ Slot slot_of(const Inputs &in, u64 slot) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    return Slot{in.segments[slot * SEISMIC_SEGMENTS_STRIDE_0],
                in.segments[slot * SEISMIC_SEGMENTS_STRIDE_0 + SEISMIC_SEGMENTS_STRIDE_1],
                in.stop[slot * SEISMIC_STOP_STRIDE_0],
                in.previous_bank[slot * SEISMIC_PREVIOUS_BANK_STRIDE_0],
                in.previous_tape[slot * SEISMIC_PREVIOUS_TAPE_STRIDE_0],
                in.following_bank[slot * SEISMIC_FOLLOWING_BANK_STRIDE_0]};
}

// Rows the slot records in its successor's tape: those after the stop row, at
// most T.
__device__ __forceinline__ int tape_rows(const Inputs &in, const Slot &slot) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    return min(static_cast<int>(SEISMIC_DIM_T), slot.hi - slot.lo - slot.stop);
}

// Tape row `entry` of `bank`.
__device__ __forceinline__ float *tape_row(const Inputs &in, int bank, int entry) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    return in.tape + static_cast<u64>(bank) * SEISMIC_TAPE_STRIDE_0 +
           static_cast<u64>(entry) * SEISMIC_TAPE_STRIDE_1;
}

// The first row no slot covers: slots partition a prefix of rows.
__device__ __forceinline__ int covered_end(const Inputs &in) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    const u64 slots = SEISMIC_DIM_B;
    return slots == 0 ? 0 : in.segments[(slots - 1) * SEISMIC_SEGMENTS_STRIDE_0 +
                                        SEISMIC_SEGMENTS_STRIDE_1];
}

__device__ __forceinline__ float projection(const Inputs &in, u64 row, u64 column) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    return element::at<Act>(in.projection, row * SEISMIC_PROJECTION_STRIDE_0 + column);
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

// The chunked rows of a slot, the ones before its publication row lo + stop,
// split into pieces of at most `CHUNK` rows:
template <int CHUNK>
__device__ __forceinline__ int pieces_before(const Slot &slot) {
    return (slot.stop + CHUNK - 1) / CHUNK;
}
// Rows [first, first + count) of piece `index` (< pieces_before) of the slot.
struct Piece {
    int first;
    int count;
};
template <int CHUNK>
__device__ __forceinline__ Piece piece_of(const Slot &slot, int index) {
    const int first = index * CHUNK;
    return Piece{slot.lo + first, min(CHUNK, slot.stop - first)};
}

// Raw input row `position` (slot-local) of a channel: the source version's
// window rows before the slot, the projection after.
__device__ __forceinline__ float raw_input(const Inputs &in, const Slot &slot, int position, int channel) {
    return position < 0 ? element::at<Act>(in.window, window_at(in, slot.source, slot.taped + C - 1 + position, channel))
                        : projection(in, slot.lo + position, channel);
}

// Publish the slot's successor window: the C - 1 raw rows before the
// publication row, then the raw rows of its tape. `part` of `parts`
// cooperating blocks writes an even share.
__device__ __forceinline__ void publish_window(const Inputs &in, const Slot &slot, int part,
                                               int parts) {
    const int total = (C - 1 + tape_rows(in, slot)) * CH;
    const int per = (total + parts - 1) / parts;
    const int first = part * per;
    const int last = min(total, first + per);
    for (int index = first + threadIdx.x; index < last; index += blockDim.x) {
        const int row = index / CH;
        const int channel = index % CH;
        element::put<Act>(in.window, window_at(in, slot.target, row, channel),
                          raw_input(in, slot, slot.stop + row - (C - 1), channel));
    }
}

// The first member value head of `head`'s key head: the one head of each key
// head that records the key in a tape row.
__device__ __forceinline__ bool records_key(const Inputs &in, int head) {
    return in.grouped ? head % (NV / NK) == 0 : head < NK;
}

// Tape row `entry` of the slot's source bank and its decay of value head
// `head`: S <- decay S + u k^T replays it.
struct TapeEntry {
    const float *row;
    float decay;
};
__device__ __forceinline__ TapeEntry tape_entry(const Inputs &in, const Slot &slot, int head, int entry) {
    const float *row = tape_row(in, slot.source, entry);
    return TapeEntry{row, row[TAPE_D + head]};
}
static_assert((NV * W + NK * W + NV) % 4 == 0, "16-byte tape rows");

// Slots of at most this many rows advance row-sequentially in either entry,
// so a request's recurrent bits never depend on its row class or its peers
// (an MTP verify slot gets the bits of one-row decode).
constexpr int SEQUENTIAL_ROWS = 16;

// Shared memory of the sequential advance of a block owning BLOCK_ROWS state
// rows of one value head: a row's convolved q | k of the key head and the
// block's v channels, double-buffered by row parity.
template <int BLOCK_ROWS>
struct SequentialShared {
    float prepared[2][2 * W + BLOCK_ROWS];
};

// SiLU of the causal depthwise convolution of `channel` at slot-local row
// `local` (the source version's window before the slot, the projection after).
__device__ __forceinline__ float convolved(const Inputs &in, const Slot &slot, int local, int channel) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    const float *weights = in.convolution + static_cast<u64>(channel) * SEISMIC_CONVOLUTION_STRIDE_0;
    float sum = weights[(C - 1) * SEISMIC_CONVOLUTION_STRIDE_1] * projection(in, slot.lo + local, channel);
#pragma unroll
    for (int tap = 0; tap < C - 1; ++tap)
        sum = __fmaf_rn(weights[tap * SEISMIC_CONVOLUTION_STRIDE_1],
                        raw_input(in, slot, local + tap - (C - 1), channel), sum);
    return sum / (1.0f + expf(-sum));
}

// A warp's ROWS state rows in the sequential layout: lane l holds columns
// [l W / 32, (l + 1) W / 32) of each.
template <int ROWS>
using WarpRows = float[ROWS][W / 32];

template <int ROWS>
__device__ __forceinline__ void load_rows(const Inputs &in, int bank, int head, int first_row,
                                          WarpRows<ROWS> &state) {
    const int lane = threadIdx.x % 32;
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
        element::f32_span(state_row(in, bank, head, first_row + r) + lane * (W / 32), state[r]);
}

template <int ROWS>
__device__ __forceinline__ void store_rows(const Inputs &in, int bank, int head, int first_row,
                                           const WarpRows<ROWS> &state) {
    const int lane = threadIdx.x % 32;
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
        element::f32_span_store(state_row(in, bank, head, first_row + r) + lane * (W / 32), state[r]);
}

// A warp's ROWS rows of the slot's source version: the bank's state advanced
// by its first `taped` tape rows with the step's update, so the bits equal a
// run that published after those rows.
template <int ROWS>
__device__ __forceinline__ void load_version(const Inputs &in, const Slot &slot, int head, int first_row,
                                             WarpRows<ROWS> &state) {
    constexpr int CPL = W / 32;
    load_rows<ROWS>(in, slot.source, head, first_row, state);
    const int lane = threadIdx.x % 32;
    const int key = key_head(in, head);
    for (int entry = 0; entry < slot.taped; ++entry) {
        const TapeEntry tape = tape_entry(in, slot, head, entry);
        float k[CPL];
        element::f32_span(tape.row + TAPE_K + key * W + lane * CPL, k);
#pragma unroll
        for (int s = 0; s < ROWS; ++s) {
            const float u = tape.row[TAPE_U + head * W + first_row + s];
#pragma unroll
            for (int c = 0; c < CPL; ++c) state[s][c] = __fmaf_rn(u, k[c], state[s][c] * tape.decay);
        }
    }
}

// The row-sequential gated delta rule over the slot's rows [begin, hi), the
// arithmetic of `qwen_recurrent_step` (a block's shape never changes bits).
// Every thread of the block calls it. The block owns state rows [block_row,
// block_row + BLOCK_ROWS) of value head `head`; a warp with `owns` holds its
// ROWS rows from `first_row` in `state` (the state before row `begin`),
// advances them, writes their mixed outputs, publishes them after the slot's
// first `stop` rows (before any row when `begin` = lo and stop = 0) and
// records the rows after the stop row in the successor's tape. Per row the
// block convolves (causal depthwise conv + SiLU) the key head's q and k
// channels and the block's v channels into a buffer double-buffered by row
// parity; each warp then L2-normalizes q (scaled by W^-1/2) and k and
// advances its rows: S <- decay S + beta (v - decay S k) k^T, output S q.
template <int ROWS, int BLOCK_ROWS>
__device__ __forceinline__ void advance_rows(const Inputs &in, const Slot &slot, int begin, int head,
                                             int block_row, bool owns, int first_row,
                                             WarpRows<ROWS> &state, u8 *mixed,
                                             SequentialShared<BLOCK_ROWS> &shared) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    constexpr int CPL = W / 32;
    constexpr int PREPARED = 2 * W + BLOCK_ROWS;
    const int lane = threadIdx.x % 32;
    const int key_row = key_head(in, head);
    auto mixed_at = [&](int row, int state_row) {
        return static_cast<u64>(row) * SEISMIC_RESULT_0_STRIDE_0 +
               static_cast<u64>(head) * SEISMIC_RESULT_0_STRIDE_1 +
               static_cast<u64>(state_row) * SEISMIC_RESULT_0_STRIDE_2;
    };
    const int publish = slot.lo + slot.stop;
    const int taped = tape_rows(in, slot);
    // The warp that owns state row 0 records the head's decay, and the key
    // when the head records its key head's key.
    const bool records = first_row == 0;
    const bool records_keys = records && records_key(in, head);
    if (owns && begin == slot.lo && slot.stop == 0) store_rows<ROWS>(in, slot.target, head, first_row, state);

    const float root = rsqrtf(static_cast<float>(W));
    for (int row = begin; row < slot.hi; ++row) {
        float *buffer = shared.prepared[row & 1];
        for (int index = threadIdx.x; index < PREPARED; index += blockDim.x) {
            const int channel = index < W       ? key_row * W + index
                                : index < 2 * W ? (NK + key_row) * W + index - W
                                                : (2 * NK + head) * W + block_row + index - 2 * W;
            buffer[index] = convolved(in, slot, row - slot.lo, channel);
        }
        __syncthreads();
        if (!owns) continue;

        float q[CPL], k[CPL];
        element::f32_span(buffer + lane * CPL, q);
        element::f32_span(buffer + W + lane * CPL, k);
        float q_squares = 0.0f, k_squares = 0.0f;
#pragma unroll
        for (int c = 0; c < CPL; ++c) {
            q_squares = __fmaf_rn(q[c], q[c], q_squares);
            k_squares = __fmaf_rn(k[c], k[c], k_squares);
        }
        const float q_inverse = rsqrtf(seismic_warp_sum_f32(q_squares) + in.epsilon) * root;
        const float k_inverse = rsqrtf(seismic_warp_sum_f32(k_squares) + in.epsilon);
#pragma unroll
        for (int c = 0; c < CPL; ++c) {
            q[c] *= q_inverse;
            k[c] *= k_inverse;
        }
        const Gates row_gates = gates(in, row, head);

        float remembered[ROWS];
#pragma unroll
        for (int s = 0; s < ROWS; ++s) {
            float sum = 0.0f;
#pragma unroll
            for (int c = 0; c < CPL; ++c) sum = __fmaf_rn(state[s][c] * row_gates.decay, k[c], sum);
            remembered[s] = sum;
        }
#pragma unroll
        for (int s = 0; s < ROWS; ++s) remembered[s] = seismic_warp_sum_f32(remembered[s]);
        // Rows after the stop row are recorded in the successor's tape.
        float *entry = row >= publish && row - publish < taped ? tape_row(in, slot.target, row - publish)
                                                              : nullptr;
        if (entry != nullptr && records_keys) {
#pragma unroll
            for (int c = 0; c < CPL; ++c) entry[TAPE_K + key_row * W + lane * CPL + c] = k[c];
        }
        if (entry != nullptr && records && lane == 0) entry[TAPE_D + head] = row_gates.decay;
        float output[ROWS];
#pragma unroll
        for (int s = 0; s < ROWS; ++s) {
            const float v = buffer[2 * W + first_row - block_row + s];
            const float residual = (v - remembered[s]) * row_gates.beta;
            if (entry != nullptr && lane == s) entry[TAPE_U + head * W + first_row + s] = residual;
            float sum = 0.0f;
#pragma unroll
            for (int c = 0; c < CPL; ++c) {
                state[s][c] = __fmaf_rn(residual, k[c], state[s][c] * row_gates.decay);
                sum = __fmaf_rn(state[s][c], q[c], sum);
            }
            output[s] = sum;
        }
#pragma unroll
        for (int s = 0; s < ROWS; ++s) output[s] = seismic_warp_sum_f32(output[s]);
        if (lane < ROWS) {
            float value = output[0];
#pragma unroll
            for (int s = 1; s < ROWS; ++s)
                if (lane == s) value = output[s];
            element::put<Act>(mixed, mixed_at(row, first_row + lane), value);
        }
        if (row + 1 == publish) store_rows<ROWS>(in, slot.target, head, first_row, state);
    }
}

}  // namespace recurrent
