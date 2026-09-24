// Row-sequential gated delta step. A threadgroup owns (value head, block of
// ROWS state rows, slot); each simdgroup keeps four state rows in registers,
// W / 32 contiguous key columns per lane. For each block of the slot's rows
// the threadgroup computes the prologue into threadgroup memory: a
// simdgroup convolves (causal convolution over the window, SiLU) and
// L2-normalizes a whole q or k row with one simd_sum, threads convolve the
// value channels of its state rows, and the gates; after one barrier the
// rows advance in order. State is read from bank previous_bank[slot] and
// written only to bank following_bank[slot] after the slot's stop row.
// Channels of the projection and window rows and the columns of the delta
// arena's rows must be contiguous (unit stride).
#define RST_COLUMNS (SEISMIC_DIM_W / 32)
// State rows per simdgroup.
#define RST_LANE_ROWS 4
// Rows whose prologue is staged at once: 8 KiB each of q and k.
#define RST_BLOCK (2048 / SEISMIC_DIM_W)

#include "common/recurrent.h"

kernel void qwen_recurrent_step(
    device const recurrent::Storage *projection [[buffer(SEISMIC_BUFFER_PROJECTION)]],
    device const float *convolution [[buffer(SEISMIC_BUFFER_CONVOLUTION)]],
    device const float *rate [[buffer(SEISMIC_BUFFER_RATE)]],
    device const float *time_bias [[buffer(SEISMIC_BUFFER_TIME_BIAS)]],
    device const int *segments [[buffer(SEISMIC_BUFFER_SEGMENTS)]],
    device const int *stop [[buffer(SEISMIC_BUFFER_STOP)]],
    device const int *previous_bank [[buffer(SEISMIC_BUFFER_PREVIOUS_BANK)]],
    device const int *following_bank [[buffer(SEISMIC_BUFFER_FOLLOWING_BANK)]],
    device recurrent::Storage *window [[buffer(SEISMIC_BUFFER_WINDOW)]],
    device float *delta [[buffer(SEISMIC_BUFFER_DELTA)]],
    device recurrent::Storage *mixed [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_shape [[threads_per_threadgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float query_block[RST_BLOCK * SEISMIC_DIM_W];
    threadgroup float key_block[RST_BLOCK * SEISMIC_DIM_W];
    threadgroup float value_block[RST_BLOCK * SEISMIC_TUNE_ROWS];
    threadgroup float beta_block[RST_BLOCK];
    threadgroup float decay_block[RST_BLOCK];
    const uint threads = threadgroup_shape.x;
    const uint simdgroups = threads / 32;
    const ulong width = SEISMIC_DIM_W;
    const ulong key_heads = SEISMIC_DIM_NK;
    const ulong value_heads = SEISMIC_DIM_NV;
    const ulong head = group.x;
    const ulong block_row0 = ulong(group.y) * SEISMIC_TUNE_ROWS;
    const ulong row0 = block_row0 + ulong(simdgroup) * RST_LANE_ROWS;
    const ulong key_head = recurrent::key_head(head, seismic_words);
    const recurrent::Slot slot = recurrent::slot_of(segments, stop, previous_bank, following_bank, group.z,
        seismic_words);
    const long lo = slot.lo;
    const long hi = slot.hi;
    const long publish = lo + slot.stop;
    const float epsilon = as_type<float>(uint(SEISMIC_PARAM_NORM_EPSILON));
    const float query_scale = metal::rsqrt(float(width));
    const float head_rate = rate[head * SEISMIC_RATE_STRIDE_0];
    const float head_bias = time_bias[head * SEISMIC_TIME_BIAS_STRIDE_0];
    const ulong first_column = ulong(lane) * RST_COLUMNS;

    // This simdgroup's state rows, loaded first so their traffic overlaps the
    // prologue.
    device const float *initial = delta + slot.source * SEISMIC_DELTA_STRIDE_0
        + head * SEISMIC_DELTA_STRIDE_1 + row0 * SEISMIC_DELTA_STRIDE_2 + first_column;
    device float *published = delta + slot.target * SEISMIC_DELTA_STRIDE_0
        + head * SEISMIC_DELTA_STRIDE_1 + row0 * SEISMIC_DELTA_STRIDE_2 + first_column;
    float state[RST_LANE_ROWS][RST_COLUMNS];
    RECURRENT_UNROLL for (uint r = 0; r < RST_LANE_ROWS; ++r) {
        RECURRENT_UNROLL for (uint j = 0; j < RST_COLUMNS; ++j) {
            state[r][j] = initial[r * SEISMIC_DELTA_STRIDE_2 + j];
        }
    }
    if (publish == lo) {
        RECURRENT_UNROLL for (uint r = 0; r < RST_LANE_ROWS; ++r) {
            RECURRENT_UNROLL for (uint j = 0; j < RST_COLUMNS; ++j) {
                published[r * SEISMIC_DELTA_STRIDE_2 + j] = state[r][j];
            }
        }
    }
    const ulong alpha_column = (2 * key_heads + 2 * value_heads) * width + head;
    const ulong value_channel = (2 * key_heads + head) * width + block_row0;
    for (long first = lo; first < hi; first += RST_BLOCK) {
        const ulong rows = ulong(metal::min(long(RST_BLOCK), hi - first));
        // A simdgroup convolves and L2-normalizes a whole q or k row.
        for (ulong task = simdgroup; task < rows * 2; task += simdgroups) {
            const ulong i = task / 2;
            const bool is_key = task % 2 != 0;
            const long row = first + long(i);
            device const recurrent::Storage *taps[RECURRENT_TAPS];
            recurrent::taps(projection, window, slot.source, row, row - lo, taps, seismic_words);
            const ulong channel0 = (is_key ? key_heads + key_head : key_head) * width + first_column;
            float values[RST_COLUMNS];
            float squares = 0.0f;
            RECURRENT_UNROLL for (uint j = 0; j < RST_COLUMNS; ++j) {
                values[j] = recurrent::convolve(convolution, taps, channel0 + j, seismic_words);
                squares = metal::fma(values[j], values[j], squares);
            }
            const float inverse = metal::rsqrt(simd_sum(squares) + epsilon)
                * (is_key ? 1.0f : query_scale);
            threadgroup float *destination = (is_key ? key_block : query_block) + i * width
                + first_column;
            RECURRENT_UNROLL for (uint j = 0; j < RST_COLUMNS; ++j) {
                destination[j] = values[j] * inverse;
            }
        }
        for (ulong item = thread_index; item < rows * SEISMIC_TUNE_ROWS; item += threads) {
            const ulong i = item / SEISMIC_TUNE_ROWS;
            const long row = first + long(i);
            device const recurrent::Storage *taps[RECURRENT_TAPS];
            recurrent::taps(projection, window, slot.source, row, row - lo, taps, seismic_words);
            value_block[item] = recurrent::convolve(convolution, taps, value_channel
                + item % SEISMIC_TUNE_ROWS, seismic_words);
        }
        for (ulong i = thread_index; i < rows; i += threads) {
            device const recurrent::Storage *item = projection + (ulong(first) + i)
                * SEISMIC_PROJECTION_STRIDE_0;
            const recurrent::Gates gates = recurrent::gates(element::Act::load(item[alpha_column]),
                element::Act::load(item[alpha_column + value_heads]), head_rate, head_bias);
            beta_block[i] = gates.beta;
            decay_block[i] = metal::exp(gates.log_decay);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (ulong i = 0; i < rows; ++i) {
            const long row = first + long(i);
            const float factor = decay_block[i];
            const float beta = beta_block[i];
            float query[RST_COLUMNS];
            float key[RST_COLUMNS];
            float remembered[RST_LANE_ROWS];
            RECURRENT_UNROLL for (uint j = 0; j < RST_COLUMNS; ++j) {
                query[j] = query_block[i * width + first_column + j];
                key[j] = key_block[i * width + first_column + j];
            }
            RECURRENT_UNROLL for (uint r = 0; r < RST_LANE_ROWS; ++r) {
                remembered[r] = 0.0f;
                RECURRENT_UNROLL for (uint j = 0; j < RST_COLUMNS; ++j) {
                    state[r][j] *= factor;
                    remembered[r] = metal::fma(state[r][j], key[j], remembered[r]);
                }
            }
            const ulong local_row = ulong(simdgroup) * RST_LANE_ROWS;
            float output[RST_LANE_ROWS];
            RECURRENT_UNROLL for (uint r = 0; r < RST_LANE_ROWS; ++r) {
                const float residual = (value_block[i * SEISMIC_TUNE_ROWS + local_row + r]
                    - simd_sum(remembered[r])) * beta;
                output[r] = 0.0f;
                RECURRENT_UNROLL for (uint j = 0; j < RST_COLUMNS; ++j) {
                    state[r][j] = metal::fma(residual, key[j], state[r][j]);
                    output[r] = metal::fma(state[r][j], query[j], output[r]);
                }
                output[r] = simd_sum(output[r]);
            }
            if (lane < RST_LANE_ROWS) {
                float mine = output[0];
                RECURRENT_UNROLL for (uint r = 1; r < RST_LANE_ROWS; ++r) {
                    mine = lane == r ? output[r] : mine;
                }
                mixed[ulong(row) * SEISMIC_RESULT_0_STRIDE_0 + head * SEISMIC_RESULT_0_STRIDE_1
                    + (row0 + lane) * SEISMIC_RESULT_0_STRIDE_2] = element::Act::store(mine);
            }
            if (row + 1 == publish) {
                RECURRENT_UNROLL for (uint r = 0; r < RST_LANE_ROWS; ++r) {
                    RECURRENT_UNROLL for (uint j = 0; j < RST_COLUMNS; ++j) {
                        published[r * SEISMIC_DELTA_STRIDE_2 + j] = state[r][j];
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Rows after the last slot belong to no sequence; their output is zero.
    if (group.z + 1 == SEISMIC_DIM_B && lane < RST_LANE_ROWS) {
        for (ulong row = ulong(hi); row < SEISMIC_DIM_M; ++row) {
            mixed[row * SEISMIC_RESULT_0_STRIDE_0 + head * SEISMIC_RESULT_0_STRIDE_1
                + (row0 + lane) * SEISMIC_RESULT_0_STRIDE_2] = element::Act::store(0.0f);
        }
    }
    // The first row block publishes this head's window channels.
    if (group.y == 0)
        recurrent::publish_window(projection, window, slot, head, thread_index, threads, seismic_words);
}
