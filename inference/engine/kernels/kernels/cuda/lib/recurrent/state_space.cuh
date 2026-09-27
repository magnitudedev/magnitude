// Shared device code of the CUDA state-space (Mamba-2) entries
// (`state_space_step`, `state_space_chunk`; contracts and the portable
// `state_space_rows` in state_space.seismic): operand addressing, the causal
// convolution with its bias and SiLU, the step size and decay, the successor
// window publication, the version load (tape replay), the chunk's pieces, and
// the row-sequential advance both entries run: all of the step, and in the
// chunk the slots of at most STATE_SPACE_SEQUENTIAL_ROWS rows and the rows
// after a stop row. Channels of the projection, window, state and tape rows
// are contiguous (unit stride). The counterpart of
// `metal/lib/recurrent/state_space.h`, whose arithmetic it follows.
#pragma once

#include "../core/activation.cuh"
#include "versions.cuh"

namespace state_space {

using versions::u32;
using versions::u64;
using versions::u8;
typedef element::Act Act;

constexpr int NH = static_cast<int>(SEISMIC_DIM_NH);
constexpr int P = static_cast<int>(SEISMIC_DIM_P);
constexpr int G = static_cast<int>(SEISMIC_DIM_G);
constexpr int N = static_cast<int>(SEISMIC_DIM_N);
constexpr int TAPS = static_cast<int>(SEISMIC_DIM_C);
// Convolved channels x | B | C, and their first projection column.
constexpr int CH = NH * P + 2 * G * N;
constexpr int X_COLUMN = NH * P;
constexpr int DT_COLUMN = 2 * NH * P + 2 * G * N;
// Offsets of the group's B and C among the convolved channels.
constexpr int B_CHANNEL = NH * P;
constexpr int C_CHANNEL = NH * P + G * N;
// A tape row: the inputs u [NH, P], the convolved B [G, N], the decays d [NH].
constexpr int TAPE_U = 0;
constexpr int TAPE_B = NH * P;
constexpr int TAPE_D = NH * P + G * N;
// State columns per lane.
constexpr int COLUMNS = N / 32;
// Floats of one row of the chunk's `inputs` scratch: the convolved B and C
// channels, then each head's step size, then its log decay.
constexpr int INPUT_B = 0;
constexpr int INPUT_C = G * N;
constexpr int INPUT_DELTA = 2 * G * N;
constexpr int INPUT_LOG_DECAY = 2 * G * N + NH;
constexpr int INPUT_WIDTH = 2 * G * N + 2 * NH;

// Slots of at most this many rows advance row-sequentially in either entry,
// so a request's state bits never depend on its row class or its peers.
constexpr int SEQUENTIAL_ROWS = 16;
// The chunk's pieces: a slot of more than SEQUENTIAL_ROWS rows splits its
// rows before the stop row into pieces of at most PIECE rows from its first
// row; other slots have none. Pieces are numbered over the slots in order.
constexpr int PIECE = 32;

struct Operands {
    // The launch's argument words, read by the ABI stride macros.
    const seismic_words_t *words;
    const u8 *projection;
    const float *convolution;
    const float *convolution_bias;
    const float *rate;
    const float *time_bias;
    const float *skip;
    const u32 *window;
    const u32 *state;
    const u32 *tape;
    u8 *mixed;
    versions::Slots slots;
};

#define STATE_SPACE_OPERANDS()                                                                     \
    state_space::Operands {                                                                        \
        &seismic_words_value, SEISMIC_PTR(SEISMIC_BUFFER_PROJECTION),                              \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_CONVOLUTION)),              \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_CONVOLUTION_BIAS)),         \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RATE)),                     \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_TIME_BIAS)),                \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SKIP)),                     \
            reinterpret_cast<const state_space::u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_WINDOW)),       \
            reinterpret_cast<const state_space::u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_STATE)),        \
            reinterpret_cast<const state_space::u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_TAPE)),         \
            SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER), VERSIONS_SLOTS()                                 \
    }

// The group whose B and C head `head` reads.
__device__ __forceinline__ int group_of(int head) { return head * G / NH; }

// Whether `head` is its group's first head: the one that publishes and
// records the group's B and C.
__device__ __forceinline__ bool leads_group(int head) { return head == 0 || group_of(head - 1) != group_of(head); }

// Tape row `entry` of `bank`.
__device__ __forceinline__ float *tape_row(const Operands &in, int bank, int entry) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    return reinterpret_cast<float *>(versions::bank(in.tape, bank, SEISMIC_TAPE_STRIDE_0 * 4, seismic_words_value)) +
           static_cast<u64>(entry) * SEISMIC_TAPE_STRIDE_1;
}

// Row `row` of head `head`'s state in `bank`: N contiguous floats.
__device__ __forceinline__ float *state_row(const Operands &in, int bank, int head, int row) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    return reinterpret_cast<float *>(versions::bank(in.state, bank, SEISMIC_STATE_STRIDE_0 * 4, seismic_words_value)) +
           static_cast<u64>(head) * SEISMIC_STATE_STRIDE_1 + static_cast<u64>(row) * SEISMIC_STATE_STRIDE_2;
}

__device__ __forceinline__ u8 *window_row(const Operands &in, int bank, int tap) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    return versions::bank(in.window, bank, SEISMIC_WINDOW_STRIDE_0 * Act::bytes, seismic_words_value) +
           static_cast<u64>(tap) * SEISMIC_WINDOW_STRIDE_1 * Act::bytes;
}

// The raw convolution input of `channel` at slot-local `position`: the source
// version's window rows before the slot, the projection after.
__device__ __forceinline__ float raw(const Operands &in, const versions::Slot &slot, int position, int channel) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    if (position < 0)
        return element::at<Act>(window_row(in, slot.source, slot.taped + TAPS - 1 + position), channel);
    return element::at<Act>(in.projection,
                            static_cast<u64>(slot.lo + position) * SEISMIC_PROJECTION_STRIDE_0 + X_COLUMN + channel);
}

// SiLU of the causal depthwise convolution of `channel` at slot-local row
// `local` plus the channel's bias, in the body's order: the current row's
// tap, then taps 0..C - 1 fused in turn, then the bias.
__device__ __forceinline__ float convolve(const Operands &in, const versions::Slot &slot, int local, int channel) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    const float *weights = in.convolution + static_cast<u64>(channel) * SEISMIC_CONVOLUTION_STRIDE_0;
    float sum = seismic_mul_rn(weights[(TAPS - 1) * SEISMIC_CONVOLUTION_STRIDE_1], raw(in, slot, local, channel));
#pragma unroll
    for (int tap = 0; tap + 1 < TAPS; ++tap)
        sum = seismic_fma_rn(weights[tap * SEISMIC_CONVOLUTION_STRIDE_1], raw(in, slot, local + tap - (TAPS - 1), channel),
                             sum);
    sum = seismic_add_rn(sum, in.convolution_bias[static_cast<u64>(channel) * SEISMIC_CONVOLUTION_BIAS_STRIDE_0]);
    return sum / (1.0f + expf(-sum));
}

// The step size delta = softplus(dt + time_bias) and the log decay
// rate * delta of head `head` at `row`.
struct Step {
    float delta;
    float log_decay;
};

__device__ __forceinline__ Step step_of(const Operands &in, int row, int head) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    const float shifted = element::at<Act>(in.projection, static_cast<u64>(row) * SEISMIC_PROJECTION_STRIDE_0 +
                                                              DT_COLUMN + head) +
                          in.time_bias[head * SEISMIC_TIME_BIAS_STRIDE_0];
    const float delta = fmaxf(shifted, 0.0f) + logf(1.0f + expf(-fabsf(shifted)));
    return Step{delta, seismic_mul_rn(in.rate[head * SEISMIC_RATE_STRIDE_0], delta)};
}

// Publishes head `head`'s share of the slot's successor window (the C - 1 raw
// rows before the publication row, then the raw rows of its tape): its x
// channels, and its group's B and C channels when it leads the group. The
// `part`-th of `parts` cooperating blocks copies an even share.
__device__ __forceinline__ void publish_window(const Operands &in, const versions::Slot &slot, int head, u64 part,
                                               u64 parts) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    const int taps = TAPS - 1;
    const int rows = taps + versions::tape_rows(slot, seismic_words_value);
    const int group = group_of(head);
    const u64 per_tap = P + (leads_group(head) ? 2 * N : 0);
    const u64 total = static_cast<u64>(rows) * per_tap;
    const u64 per = (total + parts - 1) / parts;
    const u64 last = min(total, (part + 1) * per);
    for (u64 item = part * per + threadIdx.x; item < last; item += blockDim.x) {
        const int tap = static_cast<int>(item / per_tap);
        const int offset = static_cast<int>(item % per_tap);
        const int channel = offset < P       ? head * P + offset
                            : offset < P + N ? B_CHANNEL + group * N + offset - P
                                             : C_CHANNEL + group * N + offset - P - N;
        element::put<Act>(window_row(in, slot.target, tap), channel, raw(in, slot, slot.stop + tap - taps, channel));
    }
}

// A warp's LANE_ROWS state rows from `row0`, lane `lane` holding columns
// [lane * COLUMNS, (lane + 1) * COLUMNS) of each.
template <int LANE_ROWS> struct Rows {
    float values[LANE_ROWS][COLUMNS];
};

template <int LANE_ROWS>
__device__ __forceinline__ void store_rows(const Operands &in, int bank, int head, int row0, int lane,
                                           const Rows<LANE_ROWS> &rows) {
#pragma unroll
    for (int r = 0; r < LANE_ROWS; ++r) {
        float *to = state_row(in, bank, head, row0 + r) + lane * COLUMNS;
#pragma unroll
        for (int c = 0; c < COLUMNS; ++c) to[c] = rows.values[r][c];
    }
}

// The rows of the slot's source version: the bank's state advanced by its
// first `taped` tape rows with the step's update, so the bits equal a run that
// published after those rows.
template <int LANE_ROWS>
__device__ __forceinline__ void load_version(const Operands &in, const versions::Slot &slot, int head, int row0,
                                             int lane, Rows<LANE_ROWS> &rows) {
#pragma unroll
    for (int r = 0; r < LANE_ROWS; ++r) {
        const float *from = state_row(in, slot.source, head, row0 + r) + lane * COLUMNS;
#pragma unroll
        for (int c = 0; c < COLUMNS; ++c) rows.values[r][c] = from[c];
    }
    const int group = group_of(head);
    for (int entry = 0; entry < slot.taped; ++entry) {
        const float *tape = tape_row(in, slot.source, entry);
        const float decay = tape[TAPE_D + head];
        float b[COLUMNS];
#pragma unroll
        for (int c = 0; c < COLUMNS; ++c) b[c] = tape[TAPE_B + group * N + lane * COLUMNS + c];
#pragma unroll
        for (int r = 0; r < LANE_ROWS; ++r) {
            const float input = tape[TAPE_U + head * P + row0 + r];
#pragma unroll
            for (int c = 0; c < COLUMNS; ++c)
                rows.values[r][c] = seismic_fma_rn(input, b[c], seismic_mul_rn(rows.values[r][c], decay));
        }
    }
}

// Shared memory of the sequential advance: a span's B and C rows, x rows of
// the block's BLOCK_ROWS state rows, step sizes and decays.
template <int BLOCK_ROWS, int SPAN> struct SequentialShared {
    float b[SPAN][N];
    float c[SPAN][N];
    float x[SPAN][BLOCK_ROWS];
    float delta[SPAN];
    float decay[SPAN];
};

// The row-sequential state-space rule over the slot's rows [begin, hi), the
// arithmetic of `state_space_step` (a block's shape never changes bits). Every
// thread of the block calls it. The block owns state rows [block_row0,
// block_row0 + BLOCK_ROWS) of head `head`; each warp holds LANE_ROWS of them
// from `row0` in `rows` (the state before row `begin`), advances them, writes
// their mixed outputs and publishes them after the slot's first `stop` rows
// (at the start when `begin` is the publication row). Rows after the stop row
// are recorded in the successor's tape. Per span of up to SPAN rows the block
// convolves the group's B and C and its x channels into shared memory with
// each row's step size and decay; after one barrier the rows advance in
// order: S <- decay S + (delta x) B^T, output S C + D x.
template <int LANE_ROWS, int BLOCK_ROWS, int SPAN>
__device__ __forceinline__ void advance_rows(const Operands &in, const versions::Slot &slot, int begin, int head,
                                             int block_row0, int row0, Rows<LANE_ROWS> &rows,
                                             SequentialShared<BLOCK_ROWS, SPAN> &shared) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    const int lane = threadIdx.x % 32;
    const int group = group_of(head);
    const int publish = slot.lo + slot.stop;
    const int recorded = versions::tape_rows(slot, seismic_words_value);
    const float skip = in.skip[head * SEISMIC_SKIP_STRIDE_0];
    // The warp that owns state row 0 records the decay, and B when the head
    // leads its group.
    const bool records = row0 == 0;
    const bool records_b = records && leads_group(head);
    const int local_row = row0 - block_row0;
    if (begin == publish)
        store_rows<LANE_ROWS>(in, slot.target, head, row0, lane, rows);
    for (int first = begin; first < slot.hi; first += SPAN) {
        const int count = min(SPAN, slot.hi - first);
        for (int item = threadIdx.x; item < count * 2 * N; item += blockDim.x) {
            const int i = item / (2 * N);
            const int column = item % (2 * N);
            const int channel = column < N ? B_CHANNEL + group * N + column : C_CHANNEL + group * N + column - N;
            const float value = convolve(in, slot, first + i - slot.lo, channel);
            if (column < N)
                shared.b[i][column] = value;
            else
                shared.c[i][column - N] = value;
        }
        // The x channels go to the last threads, off the B/C work.
        for (int item = blockDim.x - 1 - threadIdx.x; item < count * BLOCK_ROWS; item += blockDim.x) {
            const int i = item / BLOCK_ROWS;
            shared.x[i][item % BLOCK_ROWS] =
                convolve(in, slot, first + i - slot.lo, head * P + block_row0 + item % BLOCK_ROWS);
        }
        for (int i = threadIdx.x; i < count; i += blockDim.x) {
            const Step step = step_of(in, first + i, head);
            shared.delta[i] = step.delta;
            shared.decay[i] = expf(step.log_decay);
        }
        __syncthreads();
        for (int i = 0; i < count; ++i) {
            const int row = first + i;
            const float decay = shared.decay[i];
            const float delta = shared.delta[i];
            float b[COLUMNS];
            float c[COLUMNS];
#pragma unroll
            for (int j = 0; j < COLUMNS; ++j) {
                b[j] = shared.b[i][lane * COLUMNS + j];
                c[j] = shared.c[i][lane * COLUMNS + j];
            }
            float *entry = row >= publish && row - publish < recorded ? tape_row(in, slot.target, row - publish)
                                                                      : nullptr;
            if (entry != nullptr && records_b) {
#pragma unroll
                for (int j = 0; j < COLUMNS; ++j) entry[TAPE_B + group * N + lane * COLUMNS + j] = b[j];
            }
            if (entry != nullptr && records && lane == 0)
                entry[TAPE_D + head] = decay;
            float mine = 0.0f;
#pragma unroll
            for (int r = 0; r < LANE_ROWS; ++r) {
                const float value = shared.x[i][local_row + r];
                const float input = seismic_mul_rn(delta, value);
                if (entry != nullptr && lane == r)
                    entry[TAPE_U + head * P + row0 + r] = input;
                float sum = 0.0f;
#pragma unroll
                for (int j = 0; j < COLUMNS; ++j) {
                    rows.values[r][j] = seismic_fma_rn(input, b[j], seismic_mul_rn(rows.values[r][j], decay));
                    sum = seismic_fma_rn(rows.values[r][j], c[j], sum);
                }
                const float output = seismic_fma_rn(skip, value, seismic_warp_sum_f32(sum));
                mine = lane == r ? output : mine;
            }
            if (lane < LANE_ROWS)
                element::put<Act>(in.mixed,
                                  static_cast<u64>(row) * SEISMIC_RESULT_0_STRIDE_0 + head * SEISMIC_RESULT_0_STRIDE_1 +
                                      (row0 + lane) * SEISMIC_RESULT_0_STRIDE_2,
                                  mine);
            if (row + 1 == publish)
                store_rows<LANE_ROWS>(in, slot.target, head, row0, lane, rows);
        }
        __syncthreads();
    }
}

// Zeroes this warp's state rows of head `head` in the mixed rows no slot
// covers.
template <int LANE_ROWS>
__device__ __forceinline__ void zero_uncovered(const Operands &in, int head, int row0) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    const int lane = threadIdx.x % 32;
    const int covered = versions::covered_end(in.slots, seismic_words_value);
    for (u64 row = covered; row < SEISMIC_DIM_M; ++row)
        if (lane < LANE_ROWS)
            element::put<Act>(in.mixed,
                              row * SEISMIC_RESULT_0_STRIDE_0 + head * SEISMIC_RESULT_0_STRIDE_1 +
                                  (row0 + lane) * SEISMIC_RESULT_0_STRIDE_2,
                              0.0f);
}

// The pieces of a slot (see PIECE).
__device__ __forceinline__ int pieces_of(const versions::Slot &slot) {
    return slot.hi - slot.lo > SEQUENTIAL_ROWS ? (slot.stop + PIECE - 1) / PIECE : 0;
}

// The number of the first piece of slot `index`.
__device__ __forceinline__ int first_piece(const Operands &in, u64 index) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    int total = 0;
    for (u64 slot = 0; slot < index; ++slot) total += pieces_of(versions::slot_of(in.slots, slot, seismic_words_value));
    return total;
}

// Piece `piece`'s slot and its piece within the slot; false past the last
// piece.
__device__ __forceinline__ bool piece_at(const Operands &in, int piece, versions::Slot &slot, int &local) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    for (u64 index = 0; index < SEISMIC_DIM_B; ++index) {
        slot = versions::slot_of(in.slots, index, seismic_words_value);
        const int count = pieces_of(slot);
        if (piece < count) {
            local = piece;
            return true;
        }
        piece -= count;
    }
    return false;
}

} // namespace state_space
