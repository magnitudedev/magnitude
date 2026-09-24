// Row-sequential gated delta step (`recurrent::advance_rows`). A threadgroup
// of four simdgroups owns (value head, block of ROWS state rows, slot); each
// simdgroup keeps ROWS / 4 state rows in registers, W / 32 contiguous key
// columns per lane. State is read from version (previous_bank,
// previous_tape)[slot] and written only to bank following_bank[slot] after the
// slot's stop row, with the tape of the rows after it. The first row block of
// a head publishes its window channels before the state work, so that copy
// overlaps the state traffic. ROWS never changes bits, and
// `qwen_recurrent_chunk` gives its short slots the same bits. Channels of the
// projection and window rows and the columns of the delta arena's rows must be
// contiguous (unit stride).
// State rows per simdgroup.
#define RST_LANE_ROWS (SEISMIC_TUNE_ROWS / 4)
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
    device const int *previous_tape [[buffer(SEISMIC_BUFFER_PREVIOUS_TAPE)]],
    device const int *following_bank [[buffer(SEISMIC_BUFFER_FOLLOWING_BANK)]],
    device recurrent::Storage *window [[buffer(SEISMIC_BUFFER_WINDOW)]],
    device float *delta [[buffer(SEISMIC_BUFFER_DELTA)]],
    device float *tape [[buffer(SEISMIC_BUFFER_TAPE)]],
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
    const ulong head = group.x;
    const ulong block_row0 = ulong(group.y) * SEISMIC_TUNE_ROWS;
    const ulong row0 = block_row0 + ulong(simdgroup) * RST_LANE_ROWS;
    const recurrent::Slot slot = recurrent::slot_of(segments, stop, previous_bank, previous_tape, following_bank,
        group.z, seismic_words);
    // The successor bank's window is no slot's source, so it is written first.
    if (group.y == 0)
        recurrent::publish_window(projection, window, slot, head, thread_index, threads, seismic_words);
    recurrent::advance_rows<RST_LANE_ROWS, SEISMIC_TUNE_ROWS, RST_BLOCK, SEISMIC_DIM_W, SEISMIC_TUNE_ROWS>(
        projection, convolution, rate, time_bias, window, delta, tape, mixed, slot, slot.lo, head, block_row0,
        row0, query_block, key_block, value_block, beta_block, decay_block, thread_index, threads, simdgroup,
        lane, seismic_words);
    // Rows after the last slot belong to no sequence; their output is zero.
    if (group.z + 1 == SEISMIC_DIM_B && lane < RST_LANE_ROWS) {
        for (ulong row = ulong(slot.hi); row < SEISMIC_DIM_M; ++row) {
            mixed[row * SEISMIC_RESULT_0_STRIDE_0 + head * SEISMIC_RESULT_0_STRIDE_1
                + (row0 + lane) * SEISMIC_RESULT_0_STRIDE_2] = element::Act::store(0.0f);
        }
    }
}
