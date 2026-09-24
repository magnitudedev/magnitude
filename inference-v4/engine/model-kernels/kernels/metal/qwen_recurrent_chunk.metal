// Chunked gated delta rule (WY form) over pieces of at most 8 rows, in two
// launches that share the `inputs` scratch (f32, rows of every slot only):
//   rows    [NK, M, 2W]   per key head: L2-normalized q (scaled by W^-1/2), k
//   values  [NV, M, W]    convolved v heads
//   gates   [NV, 2, M]    log decay, beta
//
// `qwen_recurrent_chunk_inputs`: one simdgroup per (row, channel head)
// convolves the row once, W / 4 channel quads over the lanes.
//
// `qwen_recurrent_chunk_scan`: a threadgroup owns (ROWS state rows, value
// head, slot); each simdgroup keeps the transposed state of 16 rows, two
// S^T [W, 8] halves, in registers as 8x8 matrices (each K / Q / K^T operand
// tile it loads serves both halves), and the slot's pieces advance in
// order. Per piece (rows past the piece are zero):
//   1. the piece's q/k rows, the threadgroup's value channels and the gates
//      are staged into threadgroup memory;
//   2. every simdgroup computes X = K S^T and Y = Q S^T for its rows, and
//      shares of the key columns of K K^T and Q K^T;
//   3. every simdgroup sums the shares and, with cumulative log decay G_i
//      (gamma_i = exp G_i), forms
//        A_ij = beta_i exp(G_i - G_j) k_i . k_j   (j < i),
//        D_ij = exp(G_i - G_j) q_i . k_j          (j <= i),
//        T = (I + A)^-1 = (I - A)(I + A^2)(I + A^4)   (A strictly lower, A^8 = 0),
//        U = T diag(beta) (V - diag(gamma) X),  out = diag(gamma) Y + D U,
//        S^T <- gamma_last S^T + K^T diag(gamma_last / gamma) U.
// Threadgroup row strides are 8 or 24 floats modulo 32 banks, so 8x8 loads
// are conflict-free. Decay products are exponentials of differences of
// cumulative log decays, never ratios. Channels of the projection and window
// rows and the columns of the delta arena's rows must be contiguous (unit
// stride).
#define RCH_PIECE 8
#define RCH_ROWS SEISMIC_TUNE_ROWS
#define RCH_THREADS (2 * RCH_ROWS)
#define RCH_SIMDGROUPS (RCH_ROWS / 16)
#define RCH_WIDTH SEISMIC_DIM_W
#define RCH_TILES (RCH_WIDTH / 8)
#define RCH_LANE_QUADS ((RCH_WIDTH / 4 + 31) / 32)
#define RCH_KEY_HEADS SEISMIC_DIM_NK
#define RCH_VALUE_HEADS SEISMIC_DIM_NV
#define RCH_ROW_STRIDE (2 * RCH_WIDTH)
// K K^T and Q K^T are summed from 4 shares of the key columns, whatever the
// number of simdgroups, so ROWS never changes their bits.
#define RCH_SHARES 4
#define RCH_SHARE_TILES (RCH_TILES / RCH_SHARES)
#define RCH_PADDED(n) ((n) % 32 == 8 || (n) % 32 == 24 ? (n) : (n) + 8)
#define RCH_INPUT_STRIDE RCH_PADDED(RCH_WIDTH)
#define RCH_VALUE_STRIDE RCH_PADDED(RCH_ROWS)

#include "common/recurrent.h"

// The regions of the `inputs` scratch.
struct RchScratch {
    device float *rows;
    device float *values;
    device float *gates;
    ulong height;
};

inline RchScratch rch_scratch(device float *inputs, constant ulong *seismic_words) {
    RchScratch scratch;
    scratch.height = SEISMIC_DIM_M;
    scratch.rows = inputs;
    scratch.values = scratch.rows + RCH_KEY_HEADS * scratch.height * RCH_ROW_STRIDE;
    scratch.gates = scratch.values + RCH_VALUE_HEADS * scratch.height * RCH_WIDTH;
    return scratch;
}

// The slot containing `row` and its first row, or false outside every slot.
inline bool rch_row_slot(device const int *segments, ulong row, thread ulong &slot,
    thread long &lo, constant ulong *seismic_words) {
    for (ulong candidate = 0; candidate < SEISMIC_DIM_B; ++candidate) {
        const long first = segments[candidate * SEISMIC_SEGMENTS_STRIDE_0];
        const long last = segments[candidate * SEISMIC_SEGMENTS_STRIDE_0
            + SEISMIC_SEGMENTS_STRIDE_1];
        if (long(row) >= first && long(row) < last) {
            slot = candidate;
            lo = first;
            return true;
        }
    }
    return false;
}

kernel void qwen_recurrent_chunk_inputs(
    device const recurrent::Storage *projection [[buffer(SEISMIC_BUFFER_PROJECTION)]],
    device const float *convolution [[buffer(SEISMIC_BUFFER_CONVOLUTION)]],
    device const float *rate [[buffer(SEISMIC_BUFFER_RATE)]],
    device const float *time_bias [[buffer(SEISMIC_BUFFER_TIME_BIAS)]],
    device const int *segments [[buffer(SEISMIC_BUFFER_SEGMENTS)]],
    device const int *previous_bank [[buffer(SEISMIC_BUFFER_PREVIOUS_BANK)]],
    device const recurrent::Storage *window [[buffer(SEISMIC_BUFFER_WINDOW)]],
    device float *inputs [[buffer(SEISMIC_BUFFER_SCRATCH_INPUTS)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint width = RCH_WIDTH;
    const uint head = group.y * 4 + simdgroup;
    const ulong row = group.x;
    ulong slot;
    long lo;
    if (head >= 2 * RCH_KEY_HEADS + RCH_VALUE_HEADS
        || !rch_row_slot(segments, row, slot, lo, seismic_words)) {
        return;
    }
    const RchScratch scratch = rch_scratch(inputs, seismic_words);
    const bool is_value = head >= 2 * RCH_KEY_HEADS;
    const bool is_key = head >= RCH_KEY_HEADS && !is_value;
    const uint value_head = head - 2 * RCH_KEY_HEADS;
    device float *destination = is_value
        ? scratch.values + (ulong(value_head) * scratch.height + row) * width
        : scratch.rows + (ulong(head % RCH_KEY_HEADS) * scratch.height + row) * RCH_ROW_STRIDE
            + (is_key ? width : 0);

    // The tap rows: the window of the slot's accepted bank before the slot,
    // the projection after.
    const ulong source = ulong(previous_bank[slot * SEISMIC_PREVIOUS_BANK_STRIDE_0]);
    device const recurrent::Storage *taps[RECURRENT_TAPS];
    recurrent::taps(projection, window, source, long(row), long(row) - lo, taps, seismic_words);
    // This lane's channel quads.
    float4 values[RCH_LANE_QUADS];
    float squares = 0.0f;
    RECURRENT_UNROLL for (uint j = 0; j < RCH_LANE_QUADS; ++j) {
        const uint channel = 4 * (lane + 32 * j);
        values[j] = channel < width
            ? recurrent::convolve4(convolution, taps, head * width + channel, seismic_words) : float4(0.0f);
        squares += metal::dot(values[j], values[j]);
    }
    if (!is_value) {
        const float epsilon = as_type<float>(uint(SEISMIC_PARAM_NORM_EPSILON));
        const float inverse = metal::rsqrt(simd_sum(squares) + epsilon)
            * (is_key ? 1.0f : metal::rsqrt(float(width)));
        RECURRENT_UNROLL for (uint j = 0; j < RCH_LANE_QUADS; ++j) {
            values[j] *= inverse;
        }
    }
    RECURRENT_UNROLL for (uint j = 0; j < RCH_LANE_QUADS; ++j) {
        const uint channel = 4 * (lane + 32 * j);
        if (channel < width) {
            *reinterpret_cast<device float4 *>(destination + channel) = values[j];
        }
    }
    if (is_value && lane == 0) {
        const ulong alpha_column = (2 * RCH_KEY_HEADS + RCH_VALUE_HEADS) * width
            + RCH_VALUE_HEADS * width + value_head;
        device const recurrent::Storage *item = projection + row * SEISMIC_PROJECTION_STRIDE_0;
        const float alpha = element::Act::load(item[alpha_column * SEISMIC_PROJECTION_STRIDE_1]);
        const float beta_input = element::Act::load(item[(alpha_column + RCH_VALUE_HEADS)
            * SEISMIC_PROJECTION_STRIDE_1]);
        const recurrent::Gates gates = recurrent::gates(alpha, beta_input,
            rate[value_head * SEISMIC_RATE_STRIDE_0], time_bias[value_head * SEISMIC_TIME_BIAS_STRIDE_0]);
        scratch.gates[value_head * 2 * scratch.height + row] = gates.log_decay;
        scratch.gates[(value_head * 2 + 1) * scratch.height + row] = gates.beta;
    }
}

kernel void qwen_recurrent_chunk_scan(
    device const recurrent::Storage *projection [[buffer(SEISMIC_BUFFER_PROJECTION)]],
    device const int *segments [[buffer(SEISMIC_BUFFER_SEGMENTS)]],
    device const int *stop [[buffer(SEISMIC_BUFFER_STOP)]],
    device const int *previous_bank [[buffer(SEISMIC_BUFFER_PREVIOUS_BANK)]],
    device const int *following_bank [[buffer(SEISMIC_BUFFER_FOLLOWING_BANK)]],
    device recurrent::Storage *window [[buffer(SEISMIC_BUFFER_WINDOW)]],
    device float *delta [[buffer(SEISMIC_BUFFER_DELTA)]],
    device float *inputs [[buffer(SEISMIC_BUFFER_SCRATCH_INPUTS)]],
    device recurrent::Storage *mixed [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane_index [[thread_index_in_simdgroup]]) {
    // The piece's inputs: Q and K [8, W], value channels [8, ROWS], per row
    // the log decay and beta, and the shares of K K^T and Q K^T.
    threadgroup float queries[RCH_PIECE * RCH_INPUT_STRIDE];
    threadgroup float keys[RCH_PIECE * RCH_INPUT_STRIDE];
    threadgroup float values[RCH_PIECE * RCH_VALUE_STRIDE];
    threadgroup float gates[2 * RCH_PIECE];
    threadgroup float shares[RCH_SHARES][2][RCH_PIECE * RCH_PIECE];

    const ushort lane = ushort(lane_index);
    const ulong width = RCH_WIDTH;
    const ulong head = group.y;
    const ulong slot = group.z;
    const ulong block_row0 = ulong(group.x) * RCH_ROWS;
    const ulong row0 = block_row0 + ulong(simdgroup) * 16;
    const ulong key_head = recurrent::key_head(head, seismic_words);
    const RchScratch scratch = rch_scratch(inputs, seismic_words);
    const recurrent::Slot geometry = recurrent::slot_of(segments, stop, previous_bank, following_bank, slot,
        seismic_words);
    const ulong source = geometry.source;
    const ulong target = geometry.target;
    // The first row block publishes this head's window channels (for the
    // state after `stop` rows).
    if (group.x == 0)
        recurrent::publish_window(projection, window, geometry, head, thread_index, RCH_THREADS, seismic_words);

    // This lane's elements of an 8x8 simdgroup matrix: (row, column + e).
    const ushort quad = lane / 4;
    const ushort row = (quad & 4) + ((lane / 2) % 4);
    const ushort column = (quad & 2) * 2 + (lane % 2) * 2;

    // S^T tiles of the two 8-row halves: rows are key coordinates.
    const ulong state_stride = SEISMIC_DELTA_STRIDE_2;
    device float *published = delta + target * SEISMIC_DELTA_STRIDE_0
        + head * SEISMIC_DELTA_STRIDE_1 + row0 * state_stride;
    simdgroup_float8x8 state[2][RCH_TILES];
    {
        device const float *initial = delta + source * SEISMIC_DELTA_STRIDE_0
            + head * SEISMIC_DELTA_STRIDE_1 + row0 * state_stride;
        RECURRENT_UNROLL for (uint h = 0; h < 2; ++h) {
            RECURRENT_UNROLL for (uint c = 0; c < RCH_TILES; ++c) {
                simdgroup_load(state[h][c], initial + 8 * h * state_stride + 8 * c, state_stride,
                    ulong2(0), true);
            }
        }
    }
    if (geometry.stop == 0) {
        RECURRENT_UNROLL for (uint h = 0; h < 2; ++h) {
            RECURRENT_UNROLL for (uint c = 0; c < RCH_TILES; ++c) {
                simdgroup_store(state[h][c], published + 8 * h * state_stride + 8 * c,
                    state_stride, ulong2(0), true);
            }
        }
    }

    device const float *head_rows = scratch.rows + key_head * scratch.height * RCH_ROW_STRIDE;
    device const float *head_values = scratch.values + head * scratch.height * width + block_row0;
    device const float *head_gates = scratch.gates + head * 2 * scratch.height;
    const ulong pieces = recurrent::piece_count<RCH_PIECE>(geometry);
    const ulong pieces_before = recurrent::pieces_before<RCH_PIECE>(geometry);
    for (ulong piece = 0; piece < pieces; ++piece) {
        long first;
        long length;
        recurrent::piece_of<RCH_PIECE>(geometry, piece, first, length);

        // 1. Stage the piece's q/k rows, value channels and gates.
        for (uint item = thread_index; item < RCH_PIECE * RCH_ROW_STRIDE / 4; item += RCH_THREADS) {
            const uint i = item / (RCH_ROW_STRIDE / 4);
            const uint channel = 4 * (item % (RCH_ROW_STRIDE / 4));
            const float4 value = long(i) < length
                ? *reinterpret_cast<device const float4 *>(head_rows
                    + (ulong(first) + i) * RCH_ROW_STRIDE + channel) : 0.0f;
            *reinterpret_cast<threadgroup float4 *>((channel < width ? queries : keys)
                + i * RCH_INPUT_STRIDE + channel % width) = value;
        }
        for (uint item = thread_index; item < RCH_PIECE * RCH_ROWS / 4; item += RCH_THREADS) {
            const uint i = item / (RCH_ROWS / 4);
            const uint channel = 4 * (item % (RCH_ROWS / 4));
            *reinterpret_cast<threadgroup float4 *>(values + i * RCH_VALUE_STRIDE + channel)
                = long(i) < length ? *reinterpret_cast<device const float4 *>(head_values
                    + (ulong(first) + i) * width + channel) : 0.0f;
        }
        if (thread_index < 2 * RCH_PIECE) {
            const uint i = thread_index % RCH_PIECE;
            gates[thread_index] = long(i) < length
                ? head_gates[(thread_index / RCH_PIECE) * scratch.height + ulong(first) + i] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. Shares of K K^T and Q K^T, then X = K S^T and Y = Q S^T.
        for (uint share = simdgroup; share < RCH_SHARES; share += RCH_SIMDGROUPS) {
            simdgroup_float8x8 keys_share = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            simdgroup_float8x8 queries_share = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            RECURRENT_UNROLL for (uint s = 0; s < RCH_SHARE_TILES; ++s) {
                const uint c = share * RCH_SHARE_TILES + s;
                simdgroup_float8x8 key;
                simdgroup_float8x8 query;
                simdgroup_float8x8 transposed;
                simdgroup_load(key, keys + 8 * c, RCH_INPUT_STRIDE);
                simdgroup_load(query, queries + 8 * c, RCH_INPUT_STRIDE);
                simdgroup_load(transposed, keys + 8 * c, RCH_INPUT_STRIDE, ulong2(0), true);
                simdgroup_multiply_accumulate(keys_share, key, transposed, keys_share);
                simdgroup_multiply_accumulate(queries_share, query, transposed, queries_share);
            }
            simdgroup_store(keys_share, shares[share][0], RCH_PIECE);
            simdgroup_store(queries_share, shares[share][1], RCH_PIECE);
        }
        simdgroup_float8x8 removed[2];
        simdgroup_float8x8 output[2];
        RECURRENT_UNROLL for (uint h = 0; h < 2; ++h) {
            removed[h] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            output[h] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
        RECURRENT_UNROLL for (uint c = 0; c < RCH_TILES; ++c) {
            simdgroup_float8x8 key;
            simdgroup_float8x8 query;
            simdgroup_load(key, keys + 8 * c, RCH_INPUT_STRIDE);
            simdgroup_load(query, queries + 8 * c, RCH_INPUT_STRIDE);
            RECURRENT_UNROLL for (uint h = 0; h < 2; ++h) {
                simdgroup_multiply_accumulate(removed[h], key, state[h][c], removed[h]);
                simdgroup_multiply_accumulate(output[h], query, state[h][c], output[h]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 3. Lane i < 8 holds row i's cumulative log decay and beta; padding
        // rows have zero decay and beta.
        const float beta = lane < RCH_PIECE ? gates[RCH_PIECE + lane] : 0.0f;
        const float cumulative = simd_prefix_inclusive_sum(lane < RCH_PIECE ? gates[lane] : 0.0f);
        const float last = simd_shuffle(cumulative, ushort(RCH_PIECE - 1));
        const float row_cumulative = simd_shuffle(cumulative, row);
        const float row_beta = simd_shuffle(beta, row);
        const float row_gamma = metal::exp(row_cumulative);
        const float row_tail = metal::exp(last - row_cumulative);

        // A, D and I - A from the summed shares; T = (I - A)(I + A^2)(I + A^4).
        simdgroup_float8x8 system;
        simdgroup_float8x8 attention;
        simdgroup_float8x8 complement;
        RECURRENT_UNROLL for (ushort e = 0; e < 2; ++e) {
            const ushort j = ushort(column + e);
            float key_dot = 0.0f;
            float query_dot = 0.0f;
            RECURRENT_UNROLL for (uint s = 0; s < RCH_SHARES; ++s) {
                key_dot += shares[s][0][row * RCH_PIECE + j];
                query_dot += shares[s][1][row * RCH_PIECE + j];
            }
            const float decay = metal::exp(j <= row ? row_cumulative
                - simd_shuffle(cumulative, j) : 0.0f);
            const float a = j < row ? row_beta * decay * key_dot : 0.0f;
            system.thread_elements()[e] = a;
            complement.thread_elements()[e] = (j == row ? 1.0f : 0.0f) - a;
            attention.thread_elements()[e] = j <= row ? decay * query_dot : 0.0f;
        }
        simdgroup_float8x8 square;
        simdgroup_float8x8 fourth;
        simdgroup_multiply(square, system, system);
        simdgroup_multiply(fourth, square, square);
        RECURRENT_UNROLL for (ushort e = 0; e < 2; ++e) {
            const float identity = row == column + e ? 1.0f : 0.0f;
            square.thread_elements()[e] += identity;
            fourth.thread_elements()[e] += identity;
        }
        simdgroup_float8x8 partial;
        simdgroup_float8x8 inverse;
        simdgroup_multiply(partial, complement, square);
        simdgroup_multiply(inverse, partial, fourth);

        // Per half: Z = diag(beta) (V - diag(gamma) X), U = T Z,
        // O = diag(gamma) Y + D U, then U <- diag(gamma_last / gamma) U.
        simdgroup_float8x8 update[2];
        RECURRENT_UNROLL for (uint h = 0; h < 2; ++h) {
            simdgroup_float8x8 value;
            simdgroup_load(value, values + simdgroup * 16 + 8 * h, RCH_VALUE_STRIDE);
            RECURRENT_UNROLL for (ushort e = 0; e < 2; ++e) {
                removed[h].thread_elements()[e] = (value.thread_elements()[e]
                    - removed[h].thread_elements()[e] * row_gamma) * row_beta;
                output[h].thread_elements()[e] *= row_gamma;
            }
            simdgroup_multiply(update[h], inverse, removed[h]);
            simdgroup_multiply_accumulate(output[h], attention, update[h], output[h]);
            if (long(row) < length) {
                device recurrent::Storage *destination = mixed + ulong(first + long(row))
                    * SEISMIC_RESULT_0_STRIDE_0 + head * SEISMIC_RESULT_0_STRIDE_1;
                RECURRENT_UNROLL for (ushort e = 0; e < 2; ++e) {
                    destination[(row0 + 8 * h + column + e) * SEISMIC_RESULT_0_STRIDE_2]
                        = element::Act::store(output[h].thread_elements()[e]);
                }
            }
            update[h].thread_elements()[0] *= row_tail;
            update[h].thread_elements()[1] *= row_tail;
        }
        // S^T <- gamma_last S^T + K^T U.
        const float decay = metal::exp(last);
        RECURRENT_UNROLL for (uint c = 0; c < RCH_TILES; ++c) {
            simdgroup_float8x8 key;
            simdgroup_load(key, keys + 8 * c, RCH_INPUT_STRIDE, ulong2(0), true);
            RECURRENT_UNROLL for (uint h = 0; h < 2; ++h) {
                state[h][c].thread_elements()[0] *= decay;
                state[h][c].thread_elements()[1] *= decay;
                simdgroup_multiply_accumulate(state[h][c], key, update[h], state[h][c]);
            }
        }
        if (piece + 1 == pieces_before) {
            RECURRENT_UNROLL for (uint h = 0; h < 2; ++h) {
                RECURRENT_UNROLL for (uint c = 0; c < RCH_TILES; ++c) {
                    simdgroup_store(state[h][c], published + 8 * h * state_stride + 8 * c,
                        state_stride, ulong2(0), true);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Rows after the last slot belong to no sequence; their output is zero.
    if (slot + 1 == SEISMIC_DIM_B) {
        const ulong rows = SEISMIC_DIM_M - ulong(geometry.hi);
        for (ulong index = lane; index < rows * 16; index += 32) {
            mixed[(ulong(geometry.hi) + index / 16) * SEISMIC_RESULT_0_STRIDE_0
                + head * SEISMIC_RESULT_0_STRIDE_1
                + (row0 + index % 16) * SEISMIC_RESULT_0_STRIDE_2] = element::Act::store(0.0f);
        }
    }
}
