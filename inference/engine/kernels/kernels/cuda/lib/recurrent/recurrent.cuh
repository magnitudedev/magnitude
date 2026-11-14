// Shared device code of the CUDA delta-rule entries (`gated_delta_step`,
// `gated_delta_chunk`, and through `kimi.cuh` `kimi_delta_step` and
// `kimi_delta_chunk`; contracts in recurrent.seismic): slot and bank lookup,
// tensor addressing, Gated DeltaNet's per-head gates, piece splitting, the
// state-arena window publication, the row-sequential advance every entry
// runs for slots of at most SEQUENTIAL_ROWS rows, the gating of the raw
// outputs into the entries' result, and the step entries' body (`step`).
// They take the rule's gates (`HeadGates`, Kimi's `kimi::ChannelGates`) and
// the output gate's input (`GateInput`) as arguments.
// Activation tensors are canonical in their last axis.

#include "../core/activation.cuh"
#include "../core/arrive.cuh"
#include "../core/functions.cuh"
#include <seismic/slab.cuh>

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
// decay factors d [NV, DECAYS] (`HeadGates`).
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
    const u32 *window;
    const u32 *delta;
    const u32 *tape;
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
            reinterpret_cast<const recurrent::u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_WINDOW)),    \
            reinterpret_cast<const recurrent::u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_DELTA)),     \
            reinterpret_cast<const recurrent::u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_TAPE)),      \
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
    return reinterpret_cast<float *>(slab::row(in.tape, bank, SEISMIC_PARAM_SLAB_BANKS,
               SEISMIC_TAPE_STRIDE_0 * 4)) + static_cast<u64>(entry) * SEISMIC_TAPE_STRIDE_1;
}

// The first row no slot covers: slots partition a prefix of rows.
__device__ __forceinline__ int covered_end(const Inputs &in) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    const u64 slots = SEISMIC_DIM_B;
    return slots == 0 ? 0 : in.segments[(slots - 1) * SEISMIC_SEGMENTS_STRIDE_0 +
                                        SEISMIC_SEGMENTS_STRIDE_1];
}

// Element (row, value head, state row) of the raw outputs, an [M, NV, W]
// scratch in A.
__device__ __forceinline__ u64 raw_index(int row, int head, int state_row) {
    return (static_cast<u64>(row) * NV + head) * W + state_row;
}

__device__ __forceinline__ float projection(const Inputs &in, u64 row, u64 column) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    return element::at<Act>(in.projection, row * SEISMIC_PROJECTION_STRIDE_0 + column);
}

__device__ __forceinline__ u8 *window_at(const Inputs &in, int bank, int tap, u64 channel) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    return slab::row(in.window, bank, SEISMIC_PARAM_SLAB_BANKS,
               SEISMIC_WINDOW_STRIDE_0 * Act::bytes)
           + (static_cast<u64>(tap) * SEISMIC_WINDOW_STRIDE_1 + channel) * Act::bytes;
}

// State row `state_row` of value head `head` in `bank`: W contiguous floats.
__device__ __forceinline__ float *state_row(const Inputs &in, int bank, int head, int state_row) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    return reinterpret_cast<float *>(slab::row(in.delta, bank, SEISMIC_PARAM_SLAB_BANKS,
               SEISMIC_DELTA_STRIDE_0 * 4)) + static_cast<u64>(head) * SEISMIC_DELTA_STRIDE_1 +
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

// A lane's W / 32 state columns in the sequential layout.
constexpr int CPL = W / 32;

// The output gate's input z: column `column` of row `row` is element
// row * stride + first + column of `rows`, a value-head-major [NV, W] row.
struct GateInput {
    const u8 *rows;
    u64 stride;
    u64 first;
    __device__ __forceinline__ float at(u64 row, u64 column) const {
        return element::at<Act>(rows, row * stride + first + column);
    }
};

// A rule's gates, which the entry bodies (`step`, `chunk::prepare`,
// `chunk::scan`), `load_version` and `advance_rows` take: the rule's `Inputs`
// and their gated-delta operands (`base`); `at(row, head, lane, factor)`,
// value head `head`'s beta at `row`, setting `factor` to the decay factors of
// lane `lane`'s CPL key columns; and `gate_input()`, z. DECAYS is 1 (the
// head's one decay scales every key column) or W (one per key column); a tape
// row records DECAYS factors per value head. Gated DeltaNet's are columns of
// the projection row (`gates`), computed once per row, and z its columns from
// CH.
struct HeadGates {
    static constexpr int DECAYS = 1;
    using Inputs = recurrent::Inputs;
    const Inputs &in;
    __device__ static const Inputs &base(const Inputs &in) { return in; }
    // Row `row`'s gates of value head `head`, its one decay for every column.
    __device__ static Gates row_gates(const Inputs &in, int row, int head) { return gates(in, row, head); }
    __device__ __forceinline__ GateInput gate_input() const {
        [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
        return GateInput{in.projection, SEISMIC_PROJECTION_STRIDE_0, CH};
    }
    __device__ __forceinline__ float at(int row, int head, int, float (&factor)[CPL]) const {
        const Gates g = row_gates(in, row, head);
#pragma unroll
        for (int c = 0; c < CPL; ++c) factor[c] = g.decay;
        return g.beta;
    }
};

// The offset in a tape row of value head `head`'s decay factor of key
// column `column`, DECAYS factors per value head.
template <int DECAYS>
__device__ __forceinline__ int tape_decay(int head, int column) {
    static_assert(DECAYS == 1 || DECAYS == W, "one decay per value head or per key column");
    return TAPE_D + head * DECAYS + (DECAYS == 1 ? 0 : column);
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
    return position < 0 ? element::at<Act>(window_at(in, slot.source, slot.taped + C - 1 + position, channel), 0)
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
        element::put<Act>(window_at(in, slot.target, row, channel), 0,
                          raw_input(in, slot, slot.stop + row - (C - 1), channel));
    }
}

// The first member value head of `head`'s key head: the one head of each key
// head that records the key in a tape row.
__device__ __forceinline__ bool records_key(const Inputs &in, int head) {
    return in.grouped ? head % (NV / NK) == 0 : head < NK;
}

// Slots of at most this many rows advance row-sequentially in either entry,
// so a request's recurrent bits never depend on its row class or its peers
// (an MTP verify slot gets the bits of one-row decode).
constexpr int SEQUENTIAL_ROWS = SEISMIC_CONST_SEQUENTIAL_ROWS;

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
using WarpRows = float[ROWS][CPL];

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
// by its first `taped` tape rows with the step's update (S <- S diag(d) +
// u k^T, the tape's DECAYS factors per value head of `gates`' rule), so the
// bits equal a run that published after those rows.
template <int ROWS, class RowGates>
__device__ __forceinline__ void load_version(const Inputs &in, const RowGates &, const Slot &slot, int head,
                                             int first_row, WarpRows<ROWS> &state) {
    constexpr int DECAYS = RowGates::DECAYS;
    static_assert((NV * W + NK * W + NV * DECAYS) % 4 == 0, "16-byte tape rows");
    load_rows<ROWS>(in, slot.source, head, first_row, state);
    const int lane = threadIdx.x % 32;
    const int key = key_head(in, head);
    for (int entry = 0; entry < slot.taped; ++entry) {
        const float *row = tape_row(in, slot.source, entry);
        float k[CPL], factor[CPL];
        element::f32_span(row + TAPE_K + key * W + lane * CPL, k);
#pragma unroll
        for (int c = 0; c < CPL; ++c)
            factor[c] = DECAYS == 1 && c > 0 ? factor[0] : row[tape_decay<DECAYS>(head, lane * CPL + c)];
#pragma unroll
        for (int s = 0; s < ROWS; ++s) {
            const float u = row[TAPE_U + head * W + first_row + s];
#pragma unroll
            for (int c = 0; c < CPL; ++c) state[s][c] = __fmaf_rn(u, k[c], state[s][c] * factor[c]);
        }
    }
}

// The row-sequential delta rule over the slot's rows [begin, hi) with the
// rule's gates `gates` (`HeadGates`), the arithmetic of the step entries (a
// block's shape never changes bits).
// Every thread of the block calls it. The block owns state rows [block_row,
// block_row + BLOCK_ROWS) of value head `head`; a warp with `owns` holds its
// ROWS rows from `first_row` in `state` (the state before row `begin`),
// advances them, writes their raw outputs to `mixed` (`raw_index`), publishes
// them after the slot's first `stop` rows (before any row when `begin` = lo
// and stop = 0) and records the rows after the stop row in the successor's
// tape. Per row the
// block convolves (causal depthwise conv + SiLU) the key head's q and k
// channels and the block's v channels into a buffer double-buffered by row
// parity; each warp then L2-normalizes q (scaled by W^-1/2) and k and
// advances its rows: S <- S diag(d) + beta (v - S diag(d) k) k^T, output
// S q, with d the row's decay factors of each lane's columns.
template <int ROWS, int BLOCK_ROWS, class RowGates>
__device__ __forceinline__ void advance_rows(const Inputs &in, const RowGates &gates, const Slot &slot, int begin,
                                             int head, int block_row, bool owns, int first_row,
                                             WarpRows<ROWS> &state, u8 *mixed,
                                             SequentialShared<BLOCK_ROWS> &shared) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    constexpr int DECAYS = RowGates::DECAYS;
    // A lane's distinct decay factors.
    constexpr int LANE_DECAYS = DECAYS == 1 ? 1 : CPL;
    constexpr int PREPARED = 2 * W + BLOCK_ROWS;
    const int lane = threadIdx.x % 32;
    const int key_row = key_head(in, head);
    const int publish = slot.lo + slot.stop;
    const int taped = tape_rows(in, slot);
    // The warp that owns state row 0 records the head's decay factors, and
    // the key when the head records its key head's key.
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
        float factor[CPL];
        const float beta = gates.at(row, head, lane, factor);

        float remembered[ROWS];
#pragma unroll
        for (int s = 0; s < ROWS; ++s) {
            float sum = 0.0f;
#pragma unroll
            for (int c = 0; c < CPL; ++c) sum = __fmaf_rn(state[s][c] * factor[c], k[c], sum);
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
        if (entry != nullptr && records && (DECAYS != 1 || lane == 0)) {
#pragma unroll
            for (int c = 0; c < LANE_DECAYS; ++c) entry[tape_decay<DECAYS>(head, lane * CPL + c)] = factor[c];
        }
        float output[ROWS];
#pragma unroll
        for (int s = 0; s < ROWS; ++s) {
            const float v = buffer[2 * W + first_row - block_row + s];
            const float residual = (v - remembered[s]) * beta;
            if (entry != nullptr && lane == s) entry[TAPE_U + head * W + first_row + s] = residual;
            float sum = 0.0f;
#pragma unroll
            for (int c = 0; c < CPL; ++c) {
                state[s][c] = __fmaf_rn(residual, k[c], state[s][c] * factor[c]);
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
            element::put<Act>(mixed, raw_index(row, head, first_row + lane), value);
        }
        if (row + 1 == publish) store_rows<ROWS>(in, slot.target, head, first_row, state);
    }
}

// The `gate` launch of every entry: block `row` forms the row's gated result
// from its raw outputs `mixed`. Per value head (one warp each, lanes striding
// the head's columns, then a warp sum),
//   gated = round_A(round_A(raw * rsqrt(sum_head raw^2 / W + eps) * norm[i])
//                   * round_A(gate(z)))
// with gate the output gate of `gate_function` (`functions::output_gate`)
// with z column head * W + i of the row's `input`.
template <class NORM>
__device__ __forceinline__ void gate_row(const Inputs &in, const GateInput &input, const u8 *mixed, const u8 *norm,
                                         float eps, u8 *gated, int row) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    __shared__ float inverses[NV];
    const int warps = blockDim.x / 32;
    for (int head = threadIdx.x / 32; head < NV; head += warps) {
        float total = 0.0f;
        for (int i = threadIdx.x % 32; i < W; i += 32) {
            const float v = element::at<Act>(mixed, raw_index(row, head, i));
            total = __fmaf_rn(v, v, total);
        }
        total = seismic_warp_sum_f32(total);
        const float inverse = rsqrtf(total / static_cast<float>(W) + eps);
        if (threadIdx.x % 32 == 0) inverses[head] = inverse;
    }
    __syncthreads();
    for (int k = threadIdx.x; k < NV * W; k += blockDim.x) {
        const int head = k / W, i = k % W;
        const float v = element::at<Act>(mixed, raw_index(row, head, i));
        const float normalized = Act::round(v * inverses[head] * element::at<NORM>(norm, i));
        const float z = input.at(row, k);
        const float activated = Act::round(functions::output_gate((int)SEISMIC_PARAM_GATE_FUNCTION, z));
        element::put<Act>(gated,
                          static_cast<u64>(row) * SEISMIC_RESULT_0_STRIDE_0 +
                              static_cast<u64>(head) * SEISMIC_RESULT_0_STRIDE_1 +
                              static_cast<u64>(i) * SEISMIC_RESULT_0_STRIDE_2,
                          Act::round(normalized * activated));
    }
}

// `gate_row`'s arithmetic for one value head over every row, one warp per
// row, by the last of the head's step blocks to arrive: the raw outputs the
// other blocks published are read from L2.
template <class NORM>
__device__ __forceinline__ void gate_head(const Inputs &in, const GateInput &input, const u8 *mixed, const u8 *norm,
                                          float eps, u8 *gated, int head) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    const typename Act::storage *raw = reinterpret_cast<const typename Act::storage *>(mixed);
    const int lane = threadIdx.x % 32;
    for (int row = threadIdx.x / 32; row < static_cast<int>(SEISMIC_DIM_M); row += blockDim.x / 32) {
        float total = 0.0f;
        for (int i = lane; i < W; i += 32) {
            const float v = Act::load(__ldcg(raw + raw_index(row, head, i)));
            total = __fmaf_rn(v, v, total);
        }
        // The butterfly leaves every lane the same sum.
        total = seismic_warp_sum_f32(total);
        const float inverse = rsqrtf(total / static_cast<float>(W) + eps);
        for (int i = lane; i < W; i += 32) {
            const float v = Act::load(__ldcg(raw + raw_index(row, head, i)));
            const float normalized = Act::round(v * inverse * element::at<NORM>(norm, i));
            const float z = input.at(row, head * W + i);
            const float activated = Act::round(functions::output_gate((int)SEISMIC_PARAM_GATE_FUNCTION, z));
            element::put<Act>(gated,
                              static_cast<u64>(row) * SEISMIC_RESULT_0_STRIDE_0 +
                                  static_cast<u64>(head) * SEISMIC_RESULT_0_STRIDE_1 +
                                  static_cast<u64>(i) * SEISMIC_RESULT_0_STRIDE_2,
                              Act::round(normalized * activated));
        }
    }
}

// The row-sequential step entries' body (`gated_delta_step`,
// `kimi_delta_step`) over the rule's gates: one block per (value head,
// state-row block, slot); slots are a grid axis. Each warp owns ROWS state
// rows (value coordinates), a lane W / 32 columns (key coordinates) of each,
// held in registers for the whole slot. The state is read from version
// (previous_bank, previous_tape)[slot] and published to following_bank[slot]
// after the slot's first stop[slot] rows, with the window and the tape of the
// rows after it. The raw outputs go to the `mixed` scratch; grid z = B zeroes
// the raw rows no slot covers. The last of a value head's blocks to arrive
// (`arrivals`, one counter per head) gates the head's rows (`gate_head`:
// `gate_row`'s arithmetic). ROWS and WARPS never change result bits, and the
// chunked entries give their short slots the same bits.
template <unsigned ROWS, unsigned WARPS, class NORM, class RowGates>
__device__ __forceinline__ void step(const typename RowGates::Inputs &rule, u8 *mixed, unsigned *arrivals,
                                     const u8 *norm, float eps, u8 *gated) {
    constexpr int BLOCK_ROWS = ROWS * WARPS;
    static_assert(W % 32 == 0 && W % BLOCK_ROWS == 0, "state rows split evenly");
    const Inputs &in = RowGates::base(rule);
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    const int head = blockIdx.x;
    const int block_row = blockIdx.y * BLOCK_ROWS;
    const u64 slot_index = blockIdx.z;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int first_row = block_row + warp * ROWS;  // this warp's state rows
    // Let a programmatic dependent start (its first weight loads overlap this
    // launch; it still waits for this launch's completion).
    seismic_dependents_launch();

    if (slot_index == SEISMIC_DIM_B) {
        const int covered = covered_end(in);
        for (u64 row = covered; row < SEISMIC_DIM_M; ++row)
            if (lane < ROWS) element::put<Act>(mixed, raw_index(row, head, first_row + lane), 0.0f);
    } else {
        const Slot slot = slot_of(in, slot_index);
        WarpRows<ROWS> state;
        load_version<ROWS>(in, RowGates{rule}, slot, head, first_row, state);
        publish_window(in, slot, head * gridDim.y + blockIdx.y, NV * gridDim.y);
        __shared__ __align__(16) SequentialShared<BLOCK_ROWS> shared;
        advance_rows<ROWS, BLOCK_ROWS>(in, RowGates{rule}, slot, slot.lo, head, block_row, true, first_row, state,
                                       mixed, shared);
    }

    __shared__ unsigned last;
    if (!arrive::last(arrivals + head, gridDim.y * gridDim.z, &last)) return;
    gate_head<NORM>(in, RowGates{rule}.gate_input(), mixed, norm, eps, gated, head);
}


}  // namespace recurrent
