// Shared pieces of the Metal state-space (Mamba-2) entries
// (`state_space_step`, `state_space_chunk`; contracts and the portable
// `state_space_rows` in state_space.seismic): operand addressing, the causal
// convolution with its bias and SiLU, the step size and decay, the successor
// window publication, the version load (tape replay), and the row-sequential
// advance both entries run: all of the step, and in the chunk the slots of at
// most STATE_SPACE_SEQUENTIAL_ROWS rows and the rows after a stop row.
// Channels of the projection, window, state and tape rows are contiguous
// (unit stride). The counterpart of `cuda/lib/recurrent/state_space.cuh`,
// `vulkan/lib/recurrent/state_space.glsl` and `cpu/lib/recurrent/state_space.rs`.

#include "../core/activation.h"
#include "versions.h"

namespace state_space {

typedef element::Act::storage Storage;

#define STATE_SPACE_UNROLL _Pragma("clang loop unroll(full)")

constant constexpr uint NH = SEISMIC_DIM_NH;
constant constexpr uint P = SEISMIC_DIM_P;
constant constexpr uint G = SEISMIC_DIM_G;
constant constexpr uint N = SEISMIC_DIM_N;
constant constexpr uint TAPS = SEISMIC_DIM_C;
// Convolved channels x | B | C, and their first projection column.
constant constexpr uint CH = NH * P + 2 * G * N;
constant constexpr uint X_COLUMN = NH * P;
constant constexpr uint DT_COLUMN = 2 * NH * P + 2 * G * N;
// Offsets of the group's B and C among the convolved channels.
constant constexpr uint B_CHANNEL = NH * P;
constant constexpr uint C_CHANNEL = NH * P + G * N;
// A tape row: the inputs u [NH, P], the convolved B [G, N], the decays d [NH].
constant constexpr uint TAPE_U = 0;
constant constexpr uint TAPE_B = NH * P;
constant constexpr uint TAPE_D = NH * P + G * N;
// State columns per lane.
constant constexpr uint COLUMNS = N / 32;

// Slots of at most this many rows advance row-sequentially in either entry,
// so a request's state bits never depend on its row class or its peers (a
// speculative verify slot gets the bits of one-row decode).
#define STATE_SPACE_SEQUENTIAL_ROWS 16

// The operands of one call.
struct Operands {
    device const Storage *projection;
    device const float *convolution;
    device const float *convolution_bias;
    device const float *rate;
    device const float *time_bias;
    device const float *skip;
    device const ulong *window;
    device const ulong *state;
    device const ulong *tape;
    device Storage *mixed;
    constant ulong *seismic_words;
};

// The group whose B and C head `head` reads.
inline uint group_of(uint head) {
    return head * G / NH;
}

// Whether `head` is its group's first head: the one that publishes and
// records the group's B and C.
inline bool leads_group(uint head) {
    return head == 0 || group_of(head - 1) != group_of(head);
}

// Tape row `entry` of `bank`.
inline device float *tape_row(Operands in, ulong bank, long entry) {
    constant ulong *seismic_words = in.seismic_words;
    return versions::bank<float>(in.tape, bank, SEISMIC_TAPE_STRIDE_0, seismic_words)
        + ulong(entry) * SEISMIC_TAPE_STRIDE_1;
}

// Row `row` of head `head`'s state in `bank`: N contiguous floats.
inline device float *state_row(Operands in, ulong bank, uint head, ulong row) {
    constant ulong *seismic_words = in.seismic_words;
    return versions::bank<float>(in.state, bank, SEISMIC_STATE_STRIDE_0, seismic_words)
        + head * SEISMIC_STATE_STRIDE_1 + row * SEISMIC_STATE_STRIDE_2;
}

// The raw convolution input of `channel` at slot-local `position`: the
// source version's window rows before the slot, the projection after.
inline float raw(Operands in, versions::Slot slot, long position, ulong channel) {
    constant ulong *seismic_words = in.seismic_words;
    if (position < 0) {
        device const Storage *window = versions::bank<Storage>(in.window, slot.source, SEISMIC_WINDOW_STRIDE_0,
            seismic_words);
        return element::Act::load(window[ulong(slot.taped + long(TAPS) - 1 + position) * SEISMIC_WINDOW_STRIDE_1
            + channel]);
    }
    return element::Act::load(in.projection[ulong(slot.lo + position) * SEISMIC_PROJECTION_STRIDE_0 + X_COLUMN
        + channel]);
}

// SiLU of the causal depthwise convolution of `channel` at slot-local row
// `local` plus the channel's bias, in the body's order: the current row's tap,
// then taps 0..C - 1 fused in turn, then the bias.
inline float convolve(Operands in, versions::Slot slot, long local, ulong channel) {
    constant ulong *seismic_words = in.seismic_words;
    device const float *weights = in.convolution + channel * SEISMIC_CONVOLUTION_STRIDE_0;
    float sum = weights[(TAPS - 1) * SEISMIC_CONVOLUTION_STRIDE_1] * raw(in, slot, local, channel);
    STATE_SPACE_UNROLL for (uint tap = 0; tap + 1 < TAPS; ++tap) {
        sum = metal::fma(weights[tap * SEISMIC_CONVOLUTION_STRIDE_1],
            raw(in, slot, local + long(tap) - long(TAPS - 1), channel), sum);
    }
    sum = sum + in.convolution_bias[channel * SEISMIC_CONVOLUTION_BIAS_STRIDE_0];
    return sum / (1.0f + metal::exp(-sum));
}

// The step size delta = softplus(dt + time_bias) and the log decay
// rate * delta of head `head` at `row`.
struct Step {
    float delta;
    float log_decay;
};

inline Step step_of(Operands in, ulong row, uint head) {
    constant ulong *seismic_words = in.seismic_words;
    const float shifted = element::Act::load(in.projection[row * SEISMIC_PROJECTION_STRIDE_0 + DT_COLUMN + head])
        + in.time_bias[head * SEISMIC_TIME_BIAS_STRIDE_0];
    Step result;
    result.delta = metal::max(shifted, 0.0f) + metal::log(1.0f + metal::exp(-metal::abs(shifted)));
    result.log_decay = in.rate[head * SEISMIC_RATE_STRIDE_0] * result.delta;
    return result;
}

// Publishes head `head`'s share of the slot's successor window (the C - 1
// raw rows before the publication row, then the raw rows of its tape): its
// x channels, and its group's B and C channels when it leads the group.
// Thread `thread_index` of `threads` copies an even share.
inline void publish_window(Operands in, versions::Slot slot, uint head, uint thread_index, uint threads) {
    constant ulong *seismic_words = in.seismic_words;
    const long taps = long(TAPS) - 1;
    const long rows = taps + versions::tape_rows(slot, seismic_words);
    device Storage *target = versions::bank<Storage>(in.window, slot.target, SEISMIC_WINDOW_STRIDE_0, seismic_words);
    const uint group = group_of(head);
    const ulong per_tap = P + (leads_group(head) ? 2 * N : 0);
    for (ulong item = thread_index; item < ulong(rows) * per_tap; item += threads) {
        const long tap = long(item / per_tap);
        const ulong offset = item % per_tap;
        const ulong channel = offset < P ? head * P + offset
            : offset < P + N           ? B_CHANNEL + group * N + offset - P
                                       : C_CHANNEL + group * N + offset - P - N;
        target[ulong(tap) * SEISMIC_WINDOW_STRIDE_1 + channel]
            = element::Act::store(raw(in, slot, slot.stop + tap - taps, channel));
    }
}

// A simdgroup's LANE_ROWS state rows from `row0`, lane `lane` holding columns
// [lane * COLUMNS, (lane + 1) * COLUMNS) of each.
template <uint LANE_ROWS>
struct Rows {
    float values[LANE_ROWS][COLUMNS];
};

template <uint LANE_ROWS>
inline void load_rows(Operands in, ulong bank, uint head, ulong row0, uint lane, thread Rows<LANE_ROWS> &rows) {
    STATE_SPACE_UNROLL for (uint r = 0; r < LANE_ROWS; ++r) {
        device const float *from = state_row(in, bank, head, row0 + r) + lane * COLUMNS;
        STATE_SPACE_UNROLL for (uint c = 0; c < COLUMNS; ++c) {
            rows.values[r][c] = from[c];
        }
    }
}

template <uint LANE_ROWS>
inline void store_rows(Operands in, ulong bank, uint head, ulong row0, uint lane, thread const Rows<LANE_ROWS> &rows) {
    STATE_SPACE_UNROLL for (uint r = 0; r < LANE_ROWS; ++r) {
        device float *to = state_row(in, bank, head, row0 + r) + lane * COLUMNS;
        STATE_SPACE_UNROLL for (uint c = 0; c < COLUMNS; ++c) {
            to[c] = rows.values[r][c];
        }
    }
}

// The rows of the slot's source version: the bank's state advanced by its
// first `taped` tape rows with the step's update, so the bits equal a run that
// published after those rows.
template <uint LANE_ROWS>
inline void load_version(Operands in, versions::Slot slot, uint head, ulong row0, uint lane,
    thread Rows<LANE_ROWS> &rows) {
    load_rows<LANE_ROWS>(in, slot.source, head, row0, lane, rows);
    const uint group = group_of(head);
    for (long entry = 0; entry < slot.taped; ++entry) {
        device const float *tape = tape_row(in, slot.source, entry);
        const float decay = tape[TAPE_D + head];
        float b[COLUMNS];
        STATE_SPACE_UNROLL for (uint c = 0; c < COLUMNS; ++c) {
            b[c] = tape[TAPE_B + group * N + lane * COLUMNS + c];
        }
        STATE_SPACE_UNROLL for (uint r = 0; r < LANE_ROWS; ++r) {
            const float input = tape[TAPE_U + head * P + row0 + r];
            STATE_SPACE_UNROLL for (uint c = 0; c < COLUMNS; ++c) {
                rows.values[r][c] = metal::fma(input, b[c], rows.values[r][c] * decay);
            }
        }
    }
}

// The row-sequential state-space rule over the slot's rows [begin, hi), the
// arithmetic of `state_space_step` (a threadgroup's shape never changes
// bits). Every thread of the threadgroup calls it. The threadgroup owns state
// rows [block_row0, block_row0 + BLOCK_ROWS) of head `head`; simdgroup
// `simdgroup` holds LANE_ROWS of them from `row0` in `rows` (the state before
// row `begin`), advances them, writes their mixed outputs and publishes them
// after the slot's first `stop` rows. Rows after the stop row are recorded in
// the successor's tape; when `begin` is the publication row the rows are
// published first. For each span of up to SPAN rows the threadgroup
// convolves the group's B and C and its x channels into threadgroup memory
// (`b_block`, `c_block` rows of N floats, `x_block` rows of BLOCK_ROWS) with
// each row's step size and decay; after one barrier the rows advance in
// order: S <- decay S + (delta x) B^T, output S C + D x.
template <uint LANE_ROWS, uint BLOCK_ROWS, uint SPAN>
inline void advance_rows(Operands in, versions::Slot slot, long begin, uint head, ulong block_row0, ulong row0,
    thread Rows<LANE_ROWS> &rows, threadgroup float *b_block, threadgroup float *c_block,
    threadgroup float *x_block, threadgroup float *delta_block, threadgroup float *decay_block,
    uint thread_index, uint threads, uint lane) {
    constant ulong *seismic_words = in.seismic_words;
    const uint group = group_of(head);
    const long publish = slot.lo + slot.stop;
    const long recorded = versions::tape_rows(slot, seismic_words);
    const float skip = in.skip[head * SEISMIC_SKIP_STRIDE_0];
    // The simdgroup that owns state row 0 records the decay, and B when the
    // head leads its group.
    const bool records = row0 == 0;
    const bool records_b = records && leads_group(head);
    const ulong local_row = row0 - block_row0;
    if (begin == publish) {
        store_rows<LANE_ROWS>(in, slot.target, head, row0, lane, rows);
    }
    for (long first = begin; first < slot.hi; first += SPAN) {
        const ulong count = ulong(metal::min(long(SPAN), slot.hi - first));
        for (ulong item = thread_index; item < count * 2 * N; item += threads) {
            const ulong i = item / (2 * N);
            const ulong column = item % (2 * N);
            const ulong channel = column < N ? B_CHANNEL + group * N + column : C_CHANNEL + group * N + column - N;
            const float value = convolve(in, slot, first + long(i) - slot.lo, channel);
            if (column < N)
                b_block[i * N + column] = value;
            else
                c_block[i * N + column - N] = value;
        }
        // The x channels go to the last threads, off the B/C work.
        for (ulong item = threads - 1 - thread_index; item < count * BLOCK_ROWS; item += threads) {
            const ulong i = item / BLOCK_ROWS;
            x_block[i * BLOCK_ROWS + item % BLOCK_ROWS] = convolve(in, slot, first + long(i) - slot.lo,
                head * P + block_row0 + item % BLOCK_ROWS);
        }
        for (ulong i = thread_index; i < count; i += threads) {
            const Step step = step_of(in, ulong(first) + i, head);
            delta_block[i] = step.delta;
            decay_block[i] = metal::exp(step.log_decay);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (ulong i = 0; i < count; ++i) {
            const long row = first + long(i);
            const float decay = decay_block[i];
            const float delta = delta_block[i];
            float b[COLUMNS];
            float c[COLUMNS];
            STATE_SPACE_UNROLL for (uint j = 0; j < COLUMNS; ++j) {
                b[j] = b_block[i * N + lane * COLUMNS + j];
                c[j] = c_block[i * N + lane * COLUMNS + j];
            }
            device float *entry = row >= publish && row - publish < recorded
                ? tape_row(in, slot.target, row - publish) : nullptr;
            if (entry != nullptr && records_b) {
                STATE_SPACE_UNROLL for (uint j = 0; j < COLUMNS; ++j) {
                    entry[TAPE_B + group * N + lane * COLUMNS + j] = b[j];
                }
            }
            if (entry != nullptr && records && lane == 0) {
                entry[TAPE_D + head] = decay;
            }
            float mine = 0.0f;
            STATE_SPACE_UNROLL for (uint r = 0; r < LANE_ROWS; ++r) {
                const float value = x_block[i * BLOCK_ROWS + local_row + r];
                const float input = delta * value;
                if (entry != nullptr && lane == r) {
                    entry[TAPE_U + head * P + row0 + r] = input;
                }
                float sum = 0.0f;
                STATE_SPACE_UNROLL for (uint j = 0; j < COLUMNS; ++j) {
                    rows.values[r][j] = metal::fma(input, b[j], rows.values[r][j] * decay);
                    sum = metal::fma(rows.values[r][j], c[j], sum);
                }
                const float output = metal::fma(skip, value, simd_sum(sum));
                mine = lane == r ? output : mine;
            }
            if (lane < LANE_ROWS) {
                in.mixed[ulong(row) * SEISMIC_RESULT_0_STRIDE_0 + head * SEISMIC_RESULT_0_STRIDE_1
                    + (row0 + lane) * SEISMIC_RESULT_0_STRIDE_2] = element::Act::store(mine);
            }
            if (row + 1 == publish) {
                store_rows<LANE_ROWS>(in, slot.target, head, row0, lane, rows);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// The chunk's pieces: a slot of more than STATE_SPACE_SEQUENTIAL_ROWS rows
// splits its rows before the stop row into pieces of at most PIECE rows from
// its first row; other slots have none. Pieces are numbered over the slots
// in order.
#define STATE_SPACE_PIECE 32

inline ulong pieces_of(versions::Slot slot) {
    return slot.hi - slot.lo > STATE_SPACE_SEQUENTIAL_ROWS
        ? ulong(slot.stop + STATE_SPACE_PIECE - 1) / STATE_SPACE_PIECE : 0;
}

// The number of the first piece of slot `index`.
inline ulong first_piece(versions::Slots slots, ulong index, constant ulong *seismic_words) {
    ulong total = 0;
    for (ulong slot = 0; slot < index; ++slot) {
        total += pieces_of(versions::slot_of(slots, slot, seismic_words));
    }
    return total;
}

// Piece `piece`'s slot and its piece within the slot; false past the last
// piece.
inline bool piece_at(versions::Slots slots, ulong piece, thread versions::Slot &slot, thread ulong &local,
    constant ulong *seismic_words) {
    for (ulong index = 0; index < SEISMIC_DIM_B; ++index) {
        slot = versions::slot_of(slots, index, seismic_words);
        const ulong count = pieces_of(slot);
        if (piece < count) {
            local = piece;
            return true;
        }
        piece -= count;
    }
    return false;
}

// Floats of one row of the chunk's `inputs` scratch: the convolved B and C
// channels, then each head's step size, then its log decay.
constant constexpr uint INPUT_B = 0;
constant constexpr uint INPUT_C = G * N;
constant constexpr uint INPUT_DELTA = 2 * G * N;
constant constexpr uint INPUT_LOG_DECAY = 2 * G * N + NH;
constant constexpr uint INPUT_WIDTH = 2 * G * N + 2 * NH;

// Zeroes this simdgroup's state rows of head `head` in the mixed rows no
// slot covers.
template <uint LANE_ROWS>
inline void zero_uncovered(Operands in, versions::Slots slots, uint head, ulong row0, uint lane) {
    constant ulong *seismic_words = in.seismic_words;
    const ulong count = SEISMIC_DIM_B;
    const long covered = count == 0 ? 0 : versions::slot_end(slots, count - 1, seismic_words);
    for (ulong row = ulong(covered); row < SEISMIC_DIM_M; ++row) {
        if (lane < LANE_ROWS) {
            in.mixed[row * SEISMIC_RESULT_0_STRIDE_0 + head * SEISMIC_RESULT_0_STRIDE_1
                + (row0 + lane) * SEISMIC_RESULT_0_STRIDE_2] = element::Act::store(0.0f);
        }
    }
}

} // namespace state_space
