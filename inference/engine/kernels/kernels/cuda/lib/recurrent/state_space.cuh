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

// Whether `head` is its group's first head: the one that records the group's
// B in the tape.
__device__ __forceinline__ bool leads_group(int head) { return head == 0 || group_of(head - 1) != group_of(head); }

// A slot with its banks resolved: the source version's window, state and
// tape banks and the successor's. Each arena's slab lookup happens once per
// thread, so the addressing of a row never waits on a lookup.
struct Slot : versions::Slot {
    const u8 *window;
    const float *state;
    const float *tape;
    u8 *next_window;
    float *next_state;
    float *next_tape;
};

__device__ __forceinline__ Slot bind(const Operands &in, const versions::Slot &rows) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    Slot slot;
    static_cast<versions::Slot &>(slot) = rows;
    const u64 window_bytes = SEISMIC_WINDOW_STRIDE_0 * Act::bytes;
    const u64 state_bytes = SEISMIC_STATE_STRIDE_0 * 4;
    slot.window = versions::bank(in.window, rows.source, window_bytes, seismic_words_value);
    slot.next_window = versions::bank(in.window, rows.target, window_bytes, seismic_words_value);
    slot.state = reinterpret_cast<const float *>(versions::bank(in.state, rows.source, state_bytes, seismic_words_value));
    slot.next_state = reinterpret_cast<float *>(versions::bank(in.state, rows.target, state_bytes, seismic_words_value));
    // A planned speculative width of zero has no tape rows to address.
    const bool taped = SEISMIC_DIM_T > 0;
    const u64 tape_bytes = SEISMIC_TAPE_STRIDE_0 * 4;
    slot.tape = taped ? reinterpret_cast<const float *>(versions::bank(in.tape, rows.source, tape_bytes, seismic_words_value))
                      : nullptr;
    slot.next_tape = taped ? reinterpret_cast<float *>(versions::bank(in.tape, rows.target, tape_bytes, seismic_words_value))
                           : nullptr;
    return slot;
}

// Tape row `entry` of a tape bank.
template <class F> __device__ __forceinline__ F *tape_row(const Operands &in, F *bank, int entry) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    return bank + static_cast<u64>(entry) * SEISMIC_TAPE_STRIDE_1;
}

// Row `row` of head `head`'s state in a state bank: N contiguous floats.
template <class F> __device__ __forceinline__ F *state_row(const Operands &in, F *bank, int head, int row) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    return bank + static_cast<u64>(head) * SEISMIC_STATE_STRIDE_1 + static_cast<u64>(row) * SEISMIC_STATE_STRIDE_2;
}

// Row `tap` of a window bank.
template <class B> __device__ __forceinline__ B *window_row(const Operands &in, B *bank, int tap) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    return bank + static_cast<u64>(tap) * SEISMIC_WINDOW_STRIDE_1 * Act::bytes;
}

// The raw convolution input of `channel` at slot-local `position`: the source
// version's window rows before the slot, the projection after.
__device__ __forceinline__ float raw(const Operands &in, const Slot &slot, int position, int channel) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    if (position < 0)
        return element::at<Act>(window_row(in, slot.window, slot.taped + TAPS - 1 + position), channel);
    return element::at<Act>(in.projection,
                            static_cast<u64>(slot.lo + position) * SEISMIC_PROJECTION_STRIDE_0 + X_COLUMN + channel);
}

// The inputs of one convolved channel at one row: its raw values at the
// taps (the oldest first, the row's own last), the tap weights and the bias.
// Gathering them apart from the arithmetic lets a thread's reads of several
// channels overlap.
struct ConvolutionInputs {
    float raw[TAPS];
    float weight[TAPS];
    float bias;
};

__device__ __forceinline__ ConvolutionInputs convolution_inputs(const Operands &in, const Slot &slot, int local,
                                                                int channel) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    const float *weights = in.convolution + static_cast<u64>(channel) * SEISMIC_CONVOLUTION_STRIDE_0;
    ConvolutionInputs inputs;
#pragma unroll
    for (int tap = 0; tap < TAPS; ++tap) {
        inputs.raw[tap] = raw(in, slot, local + tap - (TAPS - 1), channel);
        inputs.weight[tap] = weights[tap * SEISMIC_CONVOLUTION_STRIDE_1];
    }
    inputs.bias = in.convolution_bias[static_cast<u64>(channel) * SEISMIC_CONVOLUTION_BIAS_STRIDE_0];
    return inputs;
}

// SiLU of the causal depthwise convolution plus the channel's bias, in the
// body's order: the row's own tap, then taps 0..C - 2 fused in turn, then the
// bias.
__device__ __forceinline__ float convolved(const ConvolutionInputs &inputs) {
    float sum = seismic_mul_rn(inputs.weight[TAPS - 1], inputs.raw[TAPS - 1]);
#pragma unroll
    for (int tap = 0; tap + 1 < TAPS; ++tap) sum = seismic_fma_rn(inputs.weight[tap], inputs.raw[tap], sum);
    sum = seismic_add_rn(sum, inputs.bias);
    return sum / (1.0f + expf(-sum));
}

// The convolved `channel` at slot-local row `local`.
__device__ __forceinline__ float convolve(const Operands &in, const Slot &slot, int local, int channel) {
    return convolved(convolution_inputs(in, slot, local, channel));
}

// The step size delta = softplus(dt + time_bias) and the log decay
// rate * delta of head `head` at `row`, from the inputs gathered apart.
struct Step {
    float delta;
    float log_decay;
};

struct StepInputs {
    float dt;
    float time_bias;
    float rate;
};

__device__ __forceinline__ StepInputs step_inputs(const Operands &in, int row, int head) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    return StepInputs{
        element::at<Act>(in.projection, static_cast<u64>(row) * SEISMIC_PROJECTION_STRIDE_0 + DT_COLUMN + head),
        in.time_bias[head * SEISMIC_TIME_BIAS_STRIDE_0], in.rate[head * SEISMIC_RATE_STRIDE_0]};
}

__device__ __forceinline__ Step stepped(const StepInputs &inputs) {
    const float shifted = inputs.dt + inputs.time_bias;
    const float delta = fmaxf(shifted, 0.0f) + logf(1.0f + expf(-fabsf(shifted)));
    return Step{delta, seismic_mul_rn(inputs.rate, delta)};
}

__device__ __forceinline__ Step step_of(const Operands &in, int row, int head) {
    return stepped(step_inputs(in, row, head));
}

// The heads of group `group`: [first_head(group), first_head(group + 1)).
__device__ __forceinline__ int first_head(int group) { return (group * NH + G - 1) / G; }

// Publishes head `head`'s share of the slot's successor window (the C - 1 raw
// rows before the publication row, then the raw rows of its tape): its x
// channels and an even share of its group's B and C channels among the
// group's heads. The `part`-th of `parts` cooperating blocks copies an even
// share. A thread reads its next BATCH items before writing any, so the reads
// overlap.
__device__ __forceinline__ void publish_window(const Operands &in, const Slot &slot, int head, int part, int parts) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    constexpr int BATCH = 4;
    const int group = group_of(head);
    const int lead = first_head(group);
    const int heads = first_head(group + 1) - lead;
    // This head's columns [shared_lo, shared_lo + per_tap - P) of the group's
    // 2N B | C columns.
    const int shared_lo = (head - lead) * 2 * N / heads;
    const int per_tap = P + (head - lead + 1) * 2 * N / heads - shared_lo;
    const int total = (TAPS - 1 + versions::tape_rows(slot, static_cast<int>(SEISMIC_DIM_T))) * per_tap;
    const int per = (total + parts - 1) / parts;
    const int last = min(total, (part + 1) * per);
    const int stride = static_cast<int>(blockDim.x);
    for (int first = part * per + static_cast<int>(threadIdx.x); first < last; first += BATCH * stride) {
        int taps_of[BATCH];
        int channels[BATCH];
        float values[BATCH];
#pragma unroll
        for (int k = 0; k < BATCH; ++k) {
            const int item = first + k * stride;
            const int offset = item % per_tap;
            const int column = shared_lo + offset - P;
            taps_of[k] = item / per_tap;
            channels[k] = offset < P    ? head * P + offset
                          : column < N ? B_CHANNEL + group * N + column
                                       : C_CHANNEL + group * N + column - N;
            values[k] = item < last ? raw(in, slot, slot.stop + taps_of[k] - (TAPS - 1), channels[k]) : 0.0f;
        }
#pragma unroll
        for (int k = 0; k < BATCH; ++k)
            if (first + k * stride < last)
                element::put<Act>(window_row(in, slot.next_window, taps_of[k]), channels[k], values[k]);
    }
}

// A warp's LANE_ROWS state rows from `row0`, lane `lane` holding columns
// [lane * COLUMNS, (lane + 1) * COLUMNS) of each.
template <int LANE_ROWS> struct Rows {
    float values[LANE_ROWS][COLUMNS];
};

// The widest access that moves a lane's COLUMNS columns in equal pieces. A
// lane's columns start at a multiple of COLUMNS floats of a state row, and
// state rows and banks are 16-byte aligned (N % 32 == 0, 256-byte slabs).
constexpr int PIECE_FLOATS = COLUMNS % 4 == 0 ? 4 : COLUMNS % 2 == 0 ? 2 : 1;

__device__ __forceinline__ void load_columns(const float *from, float (&to)[COLUMNS]) {
#pragma unroll
    for (int c = 0; c < COLUMNS; c += PIECE_FLOATS) {
        if constexpr (PIECE_FLOATS == 4) {
            const float4 piece = *reinterpret_cast<const float4 *>(from + c);
            to[c] = piece.x;
            to[c + 1] = piece.y;
            to[c + 2] = piece.z;
            to[c + 3] = piece.w;
        } else if constexpr (PIECE_FLOATS == 2) {
            const float2 piece = *reinterpret_cast<const float2 *>(from + c);
            to[c] = piece.x;
            to[c + 1] = piece.y;
        } else {
            to[c] = from[c];
        }
    }
}

__device__ __forceinline__ void store_columns(float *to, const float (&from)[COLUMNS]) {
#pragma unroll
    for (int c = 0; c < COLUMNS; c += PIECE_FLOATS) {
        if constexpr (PIECE_FLOATS == 4)
            *reinterpret_cast<float4 *>(to + c) = make_float4(from[c], from[c + 1], from[c + 2], from[c + 3]);
        else if constexpr (PIECE_FLOATS == 2)
            *reinterpret_cast<float2 *>(to + c) = make_float2(from[c], from[c + 1]);
        else
            to[c] = from[c];
    }
}

template <int LANE_ROWS>
__device__ __forceinline__ void store_rows(const Operands &in, float *bank, int head, int row0, int lane,
                                           const Rows<LANE_ROWS> &rows) {
#pragma unroll
    for (int r = 0; r < LANE_ROWS; ++r)
        store_columns(state_row(in, bank, head, row0 + r) + lane * COLUMNS, rows.values[r]);
}

// The rows of the slot's source version: the bank's state advanced by its
// first `taped` tape rows with the step's update, so the bits equal a run that
// published after those rows.
template <int LANE_ROWS>
__device__ __forceinline__ void load_version(const Operands &in, const Slot &slot, int head, int row0,
                                             int lane, Rows<LANE_ROWS> &rows) {
#pragma unroll
    for (int r = 0; r < LANE_ROWS; ++r)
        load_columns(state_row(in, slot.state, head, row0 + r) + lane * COLUMNS, rows.values[r]);
    const int group = group_of(head);
    for (int entry = 0; entry < slot.taped; ++entry) {
        const float *tape = tape_row(in, slot.tape, entry);
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
__device__ __forceinline__ void advance_rows(const Operands &in, const Slot &slot, int begin, int head,
                                             int block_row0, int row0, Rows<LANE_ROWS> &rows,
                                             SequentialShared<BLOCK_ROWS, SPAN> &shared) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    const int lane = threadIdx.x % 32;
    const int group = group_of(head);
    const int publish = slot.lo + slot.stop;
    const int recorded = versions::tape_rows(slot, static_cast<int>(SEISMIC_DIM_T));
    const float skip = in.skip[head * SEISMIC_SKIP_STRIDE_0];
    // The warp that owns state row 0 records the decay, and B when the head
    // leads its group.
    const bool records = row0 == 0;
    const bool records_b = records && leads_group(head);
    const int local_row = row0 - block_row0;
    // A row's staged items, and a thread's share of one row's items in a
    // block of four warps.
    constexpr int STAGED_WIDTH = 2 * N + BLOCK_ROWS;
    constexpr int STAGED = (STAGED_WIDTH + 127) / 128;
    if (begin == publish)
        store_rows<LANE_ROWS>(in, slot.next_state, head, row0, lane, rows);
    for (int first = begin; first < slot.hi; first += SPAN) {
        const int count = min(SPAN, slot.hi - first);
        // Items of a row: the group's B and C channels, then the block's x
        // channels. A thread gathers STAGED items' inputs (and the step inputs
        // of row first + threadIdx.x) before any arithmetic, so the reads
        // overlap.
        const int items = count * STAGED_WIDTH;
        for (int base = 0; base < items; base += STAGED * static_cast<int>(blockDim.x)) {
            ConvolutionInputs inputs[STAGED];
#pragma unroll
            for (int k = 0; k < STAGED; ++k) {
                const int item = base + k * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
                const int column = item % STAGED_WIDTH;
                const int channel = column < N       ? B_CHANNEL + group * N + column
                                    : column < 2 * N ? C_CHANNEL + group * N + column - N
                                                     : head * P + block_row0 + column - 2 * N;
                if (item < items)
                    inputs[k] = convolution_inputs(in, slot, first + item / STAGED_WIDTH - slot.lo, channel);
            }
            const bool steps = base == 0 && static_cast<int>(threadIdx.x) < count;
            StepInputs step;
            if (steps)
                step = step_inputs(in, first + threadIdx.x, head);
#pragma unroll
            for (int k = 0; k < STAGED; ++k) {
                const int item = base + k * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
                const int i = item / STAGED_WIDTH;
                const int column = item % STAGED_WIDTH;
                if (item >= items)
                    continue;
                const float value = convolved(inputs[k]);
                if (column < N)
                    shared.b[i][column] = value;
                else if (column < 2 * N)
                    shared.c[i][column - N] = value;
                else
                    shared.x[i][column - 2 * N] = value;
            }
            if (steps) {
                const Step stepped_row = stepped(step);
                shared.delta[threadIdx.x] = stepped_row.delta;
                shared.decay[threadIdx.x] = expf(stepped_row.log_decay);
            }
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
            float *entry = row >= publish && row - publish < recorded ? tape_row(in, slot.next_tape, row - publish)
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
                store_rows<LANE_ROWS>(in, slot.next_state, head, row0, lane, rows);
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
