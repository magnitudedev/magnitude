// Chunked state-space advance (contract and portable body `state_space_rows`
// in state_space.seismic) in the state-space dual form. `inputs` convolves
// the B and C channels of every row once into scratch with each head's step
// size and log decay. `products` forms, once per piece and group, C_i . B_j
// (j <= i) of the piece's rows from B and C staged in threadgroup memory. In
// `scan` a threadgroup owns (head, block of ROWS state rows, slot) with the
// step's geometry: four simdgroups of ROWS / 4 state rows in registers, N / 32
// columns per lane. Per piece of at most STATE_SPACE_PIECE rows before the
// stop row, with gamma_i = exp(sum of the piece's log decays to row i) and
// u_j = delta_j x_j (x convolved in the scan):
//   y_i = gamma_i (S C_i) + sum_{j <= i} (gamma_i / gamma_j) (C_i . B_j) u_j + D x_i
//   S  <- gamma_last S + sum_j (gamma_last / gamma_j) u_j B_j^T
// The S C_i sums of 32 / LANE_ROWS rows at a time are reduced over the
// simdgroup together, so each lane finishes one (row, state row) output. Slots of
// at most STATE_SPACE_SEQUENTIAL_ROWS rows are not chunked and the rows after
// a stop row advance row-sequentially from the published state, both with
// `state_space_step`'s bits. ROWS never changes result bits.
#define SSC_LANE_ROWS (SEISMIC_TUNE_ROWS / 4)
// The sequential advance's span (as the step's).
#define SSC_SPAN (2048 / SEISMIC_DIM_N)

#include "lib/recurrent/state_space.h"

#define STATE_SPACE_CHUNK_ARGUMENTS                                                          \
    device const state_space::Storage *projection [[buffer(SEISMIC_BUFFER_PROJECTION)]],     \
    device const float *convolution [[buffer(SEISMIC_BUFFER_CONVOLUTION)]],                   \
    device const float *convolution_bias [[buffer(SEISMIC_BUFFER_CONVOLUTION_BIAS)]],         \
    device const float *rate [[buffer(SEISMIC_BUFFER_RATE)]],                                 \
    device const float *time_bias [[buffer(SEISMIC_BUFFER_TIME_BIAS)]],                       \
    device const float *skip [[buffer(SEISMIC_BUFFER_SKIP)]],                                 \
    device const int *segments [[buffer(SEISMIC_BUFFER_SEGMENTS)]],                           \
    device const int *stop [[buffer(SEISMIC_BUFFER_STOP)]],                                   \
    device const int *previous_bank [[buffer(SEISMIC_BUFFER_PREVIOUS_BANK)]],                 \
    device const int *previous_tape [[buffer(SEISMIC_BUFFER_PREVIOUS_TAPE)]],                 \
    device const int *following_bank [[buffer(SEISMIC_BUFFER_FOLLOWING_BANK)]],               \
    device const ulong *window [[buffer(SEISMIC_BUFFER_WINDOW)]],                             \
    device const ulong *state [[buffer(SEISMIC_BUFFER_STATE)]],                               \
    device const ulong *tape [[buffer(SEISMIC_BUFFER_TAPE)]],                                 \
    device state_space::Storage *mixed [[buffer(SEISMIC_RESULT_0_BUFFER)]],                   \
    device float *inputs [[buffer(SEISMIC_BUFFER_SCRATCH_INPUTS)]],                           \
    device float *products [[buffer(SEISMIC_BUFFER_SCRATCH_PRODUCTS)]],                       \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

#define STATE_SPACE_CHUNK_OPERANDS                                                           \
    const state_space::Operands in{projection, convolution, convolution_bias, rate, time_bias, skip, window, \
        state, tape, mixed, seismic_words};                                                  \
    const versions::Slots slots{segments, stop, previous_bank, previous_tape, following_bank}

using state_space::COLUMNS;
using state_space::INPUT_WIDTH;
using state_space::N;
using state_space::P;

// The staged inputs of row `row`.
inline device const float *staged(device const float *inputs, long row) {
    return inputs + ulong(row) * INPUT_WIDTH;
}

// One row per threadgroup x, 256 of its B and C channels and step sizes per
// threadgroup y. Rows no slot covers are no slot's input.
kernel void state_space_chunk_inputs(STATE_SPACE_CHUNK_ARGUMENTS,
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    STATE_SPACE_CHUNK_OPERANDS;
    const long row = long(group.x);
    const ulong slot_index = versions::slot_index_of_row(slots, row, seismic_words);
    if (slot_index == SEISMIC_DIM_B)
        return;
    const versions::Slot slot = versions::slot_of(slots, slot_index, seismic_words);
    const ulong item = ulong(group.y) * 256 + thread_index;
    device float *out = inputs + ulong(row) * INPUT_WIDTH;
    if (item < 2 * state_space::G * N) {
        out[state_space::INPUT_B + item] = state_space::convolve(in, slot, row - slot.lo, state_space::B_CHANNEL + item);
    } else if (item < 2 * state_space::G * N + state_space::NH) {
        const uint head = uint(item - 2 * state_space::G * N);
        const state_space::Step step = state_space::step_of(in, ulong(row), head);
        out[state_space::INPUT_DELTA + head] = step.delta;
        out[state_space::INPUT_LOG_DECAY + head] = step.log_decay;
    }
}

// Piece x (numbered over the slots), group y: C_i . B_j for j <= i, an F32
// FMA chain over the state coordinate, B and C staged in threadgroup memory 32
// state columns at a time. A thread owns pairs t, t + 256, t + 512 of the
// piece's lower triangle.
#define SSC_PAIRS (STATE_SPACE_PIECE * (STATE_SPACE_PIECE + 1) / 2)
#define SSC_OWNED ((SSC_PAIRS + 255) / 256)
kernel void state_space_chunk_products(STATE_SPACE_CHUNK_ARGUMENTS,
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_shape [[threads_per_threadgroup]]) {
    threadgroup float c_rows[STATE_SPACE_PIECE][33];
    threadgroup float b_rows[STATE_SPACE_PIECE][33];
    STATE_SPACE_CHUNK_OPERANDS;
    versions::Slot slot;
    ulong local;
    if (!state_space::piece_at(slots, group.x, slot, local, seismic_words))
        return;
    const long first = slot.lo + long(local) * STATE_SPACE_PIECE;
    const uint count = uint(metal::min(long(STATE_SPACE_PIECE), slot.stop - long(local) * STATE_SPACE_PIECE));
    const uint g = group.y;
    uint rows[SSC_OWNED];
    uint columns[SSC_OWNED];
    float sums[SSC_OWNED];
    STATE_SPACE_UNROLL for (uint k = 0; k < SSC_OWNED; ++k) {
        // Pair p = i (i + 1) / 2 + j of the lower triangle.
        const uint pair = thread_index + 256 * k;
        uint i = 0;
        while ((i + 1) * (i + 2) / 2 <= pair)
            ++i;
        rows[k] = pair < SSC_PAIRS ? i : STATE_SPACE_PIECE;
        columns[k] = pair - i * (i + 1) / 2;
        sums[k] = 0.0f;
    }
    for (uint base = 0; base < N; base += 32) {
        for (uint item = thread_index; item < count * 32; item += threadgroup_shape.x) {
            const uint i = item / 32;
            c_rows[i][item % 32] = staged(inputs, first + long(i))[state_space::INPUT_C + g * N + base + item % 32];
            b_rows[i][item % 32] = staged(inputs, first + long(i))[state_space::INPUT_B + g * N + base + item % 32];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        STATE_SPACE_UNROLL for (uint k = 0; k < SSC_OWNED; ++k) {
            if (rows[k] < count) {
                for (uint n = 0; n < 32; ++n)
                    sums[k] = metal::fma(c_rows[rows[k]][n], b_rows[columns[k]][n], sums[k]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device float *out = products + (ulong(group.x) * state_space::G + g) * STATE_SPACE_PIECE * STATE_SPACE_PIECE;
    STATE_SPACE_UNROLL for (uint k = 0; k < SSC_OWNED; ++k) {
        if (rows[k] < count)
            out[rows[k] * STATE_SPACE_PIECE + columns[k]] = sums[k];
    }
}

// Threadgroup memory of the sequential advance: B and C rows, x rows, step
// sizes and decays of a span.
#define SSC_SEQUENTIAL_FLOATS (SSC_SPAN * (2 * SEISMIC_DIM_N + SEISMIC_TUNE_ROWS) + 2 * SSC_SPAN)
// Of a piece: x rows, the weights (gamma_i / gamma_j) (C_i . B_j), the
// cumulative log decays, gamma, the tail factors gamma_last / gamma_j, the
// step sizes. Weight rows are padded so the lanes reading different rows at
// one column hit different banks.
#define SSC_WEIGHT_STRIDE (STATE_SPACE_PIECE + 1)
#define SSC_PIECE_FLOATS (STATE_SPACE_PIECE * SEISMIC_TUNE_ROWS + STATE_SPACE_PIECE * SSC_WEIGHT_STRIDE \
    + 4 * STATE_SPACE_PIECE)
static_assert(SSC_PIECE_FLOATS <= SSC_SEQUENTIAL_FLOATS, "a piece's staging fits the sequential staging");

// Sums each of `values`' 32 elements over the simdgroup and leaves lane l
// with element l's total. On Apple GPUs 32 hardware `simd_sum`s are about
// twice as fast here as a recursive-halving shuffle scatter.
inline float reduce_scatter(thread const float (&values)[32], uint lane) {
    float mine = 0.0f;
    STATE_SPACE_UNROLL for (uint m = 0; m < 32; ++m) {
        const float total = simd_sum(values[m]);
        mine = lane == m ? total : mine;
    }
    return mine;
}

// Advances this simdgroup's `rows` over piece `piece` (the slot's `local`-th)
// in the state-space dual form and writes the piece's mixed outputs.
inline void advance_piece(state_space::Operands in, device const float *inputs, device const float *products,
    versions::Slot slot, ulong piece, ulong local, uint head, ulong block_row0, ulong row0,
    thread state_space::Rows<SSC_LANE_ROWS> &rows, threadgroup float *shared, uint thread_index, uint threads,
    uint lane) {
    constant ulong *seismic_words = in.seismic_words;
    // Rows of the piece whose S C sums one reduce-scatter finishes.
    constexpr uint GROUP = 32 / SSC_LANE_ROWS;
    const uint g = state_space::group_of(head);
    const long first = slot.lo + long(local) * STATE_SPACE_PIECE;
    const uint count = uint(metal::min(long(STATE_SPACE_PIECE), slot.stop - long(local) * STATE_SPACE_PIECE));
    threadgroup float *x_piece = shared;
    threadgroup float *weights = x_piece + STATE_SPACE_PIECE * SEISMIC_TUNE_ROWS;
    threadgroup float *log_gamma = weights + STATE_SPACE_PIECE * SSC_WEIGHT_STRIDE;
    threadgroup float *gamma = log_gamma + STATE_SPACE_PIECE;
    threadgroup float *tail = gamma + STATE_SPACE_PIECE;
    threadgroup float *delta = tail + STATE_SPACE_PIECE;
    for (uint item = thread_index; item < count * SEISMIC_TUNE_ROWS; item += threads) {
        const uint i = item / SEISMIC_TUNE_ROWS;
        x_piece[item] = state_space::convolve(in, slot, first + long(i) - slot.lo,
            head * P + block_row0 + item % SEISMIC_TUNE_ROWS);
    }
    for (uint i = thread_index; i < count; i += threads) {
        delta[i] = staged(inputs, first + long(i))[state_space::INPUT_DELTA + head];
    }
    if (thread_index == 0) {
        float sum = 0.0f;
        for (uint i = 0; i < count; ++i) {
            sum += staged(inputs, first + long(i))[state_space::INPUT_LOG_DECAY + head];
            log_gamma[i] = sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const float *piece_products = products
        + (piece * state_space::G + g) * STATE_SPACE_PIECE * STATE_SPACE_PIECE;
    for (uint item = thread_index; item < count * count; item += threads) {
        const uint i = item / count;
        const uint j = item % count;
        weights[i * SSC_WEIGHT_STRIDE + j] = j <= i
            ? metal::exp(log_gamma[i] - log_gamma[j]) * piece_products[i * STATE_SPACE_PIECE + j] : 0.0f;
    }
    for (uint i = thread_index; i < count; i += threads) {
        gamma[i] = metal::exp(log_gamma[i]);
        tail[i] = metal::exp(log_gamma[count - 1] - log_gamma[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const ulong local_row = row0 - block_row0;
    const float skip = in.skip[head * SEISMIC_SKIP_STRIDE_0];
    // Lane l finishes row first + i0 + l / LANE_ROWS, state row l % LANE_ROWS.
    const uint r = lane % SSC_LANE_ROWS;
    for (uint i0 = 0; i0 < count; i0 += GROUP) {
        float sums[32];
        STATE_SPACE_UNROLL for (uint k = 0; k < GROUP; ++k) {
            const uint i = metal::min(i0 + k, count - 1);
            device const float *c = staged(inputs, first + long(i)) + state_space::INPUT_C + g * N + lane * COLUMNS;
            float cs[COLUMNS];
            STATE_SPACE_UNROLL for (uint j = 0; j < COLUMNS; ++j) {
                cs[j] = c[j];
            }
            STATE_SPACE_UNROLL for (uint s = 0; s < SSC_LANE_ROWS; ++s) {
                float sum = 0.0f;
                STATE_SPACE_UNROLL for (uint j = 0; j < COLUMNS; ++j) {
                    sum = metal::fma(rows.values[s][j], cs[j], sum);
                }
                sums[k * SSC_LANE_ROWS + s] = sum;
            }
        }
        const float inter = reduce_scatter(sums, lane);
        const uint i = i0 + lane / SSC_LANE_ROWS;
        if (i < count) {
            float sum = 0.0f;
            for (uint j = 0; j <= i; ++j) {
                sum = metal::fma(weights[i * SSC_WEIGHT_STRIDE + j],
                    delta[j] * x_piece[j * SEISMIC_TUNE_ROWS + local_row + r], sum);
            }
            const float value = x_piece[i * SEISMIC_TUNE_ROWS + local_row + r];
            const float output = metal::fma(skip, value, metal::fma(gamma[i], inter, sum));
            in.mixed[ulong(first + long(i)) * SEISMIC_RESULT_0_STRIDE_0 + head * SEISMIC_RESULT_0_STRIDE_1
                + (row0 + r) * SEISMIC_RESULT_0_STRIDE_2] = element::Act::store(output);
        }
    }
    // S <- gamma_last S + sum_j (gamma_last / gamma_j) u_j B_j^T.
    const float last = gamma[count - 1];
    STATE_SPACE_UNROLL for (uint s = 0; s < SSC_LANE_ROWS; ++s) {
        STATE_SPACE_UNROLL for (uint j = 0; j < COLUMNS; ++j) {
            rows.values[s][j] *= last;
        }
    }
    for (uint j = 0; j < count; ++j) {
        device const float *b = staged(inputs, first + long(j)) + state_space::INPUT_B + g * N + lane * COLUMNS;
        float bs[COLUMNS];
        STATE_SPACE_UNROLL for (uint c = 0; c < COLUMNS; ++c) {
            bs[c] = b[c];
        }
        STATE_SPACE_UNROLL for (uint s = 0; s < SSC_LANE_ROWS; ++s) {
            const float coefficient = tail[j] * (delta[j] * x_piece[j * SEISMIC_TUNE_ROWS + local_row + s]);
            STATE_SPACE_UNROLL for (uint c = 0; c < COLUMNS; ++c) {
                rows.values[s][c] = metal::fma(coefficient, bs[c], rows.values[s][c]);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void state_space_chunk_scan(STATE_SPACE_CHUNK_ARGUMENTS,
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_shape [[threads_per_threadgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float shared[SSC_SEQUENTIAL_FLOATS];
    STATE_SPACE_CHUNK_OPERANDS;
    const uint head = group.x;
    const ulong block_row0 = ulong(group.y) * SEISMIC_TUNE_ROWS;
    const ulong row0 = block_row0 + ulong(simdgroup) * SSC_LANE_ROWS;
    if (group.z == SEISMIC_DIM_B) {
        state_space::zero_uncovered<SSC_LANE_ROWS>(in, slots, head, row0, lane);
        return;
    }
    const versions::Slot slot = versions::slot_of(slots, group.z, seismic_words);
    if (group.y == 0)
        state_space::publish_window(in, slot, head, thread_index, threadgroup_shape.x);
    state_space::Rows<SSC_LANE_ROWS> rows;
    state_space::load_version<SSC_LANE_ROWS>(in, slot, head, row0, lane, rows);
    const ulong pieces = state_space::pieces_of(slot);
    const ulong base = state_space::first_piece(slots, group.z, seismic_words);
    for (ulong piece = 0; piece < pieces; ++piece) {
        advance_piece(in, inputs, products, slot, base + piece, piece, head, block_row0, row0, rows, shared,
            thread_index, threadgroup_shape.x, lane);
    }
    threadgroup float *b_block = shared;
    threadgroup float *c_block = b_block + SSC_SPAN * SEISMIC_DIM_N;
    threadgroup float *x_block = c_block + SSC_SPAN * SEISMIC_DIM_N;
    threadgroup float *delta_block = x_block + SSC_SPAN * SEISMIC_TUNE_ROWS;
    threadgroup float *decay_block = delta_block + SSC_SPAN;
    const long begin = pieces == 0 ? slot.lo : slot.lo + slot.stop;
    state_space::advance_rows<SSC_LANE_ROWS, SEISMIC_TUNE_ROWS, SSC_SPAN>(in, slot, begin, head, block_row0, row0,
        rows, b_block, c_block, x_block, delta_block, decay_block, thread_index, threadgroup_shape.x, lane);
}
