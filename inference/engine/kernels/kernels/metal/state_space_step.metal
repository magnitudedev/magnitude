// Row-sequential state-space step (`state_space::advance_rows`). A
// threadgroup of four simdgroups owns (head, block of ROWS state rows, slot);
// each simdgroup keeps ROWS / 4 state rows in registers, N / 32 contiguous
// state columns per lane. State is read from version (previous_bank,
// previous_tape)[slot] and written only to bank following_bank[slot] after
// the slot's stop row, with the tape of the rows after it. The first row
// block of each head publishes its share of the window first, so that copy
// overlaps the state traffic. Grid z = B zeroes the mixed rows no slot
// covers. ROWS never changes bits, and `state_space_chunk` gives its short
// slots the same bits.
#define SSS_LANE_ROWS (SEISMIC_TUNE_ROWS / 4)
// Rows whose inputs are staged at once: 8 KiB each of B and C.
#define SSS_SPAN (2048 / SEISMIC_DIM_N)

#include "lib/recurrent/state_space.h"

kernel void state_space_step(
    device const state_space::Storage *projection [[buffer(SEISMIC_BUFFER_PROJECTION)]],
    device const float *convolution [[buffer(SEISMIC_BUFFER_CONVOLUTION)]],
    device const float *convolution_bias [[buffer(SEISMIC_BUFFER_CONVOLUTION_BIAS)]],
    device const float *rate [[buffer(SEISMIC_BUFFER_RATE)]],
    device const float *time_bias [[buffer(SEISMIC_BUFFER_TIME_BIAS)]],
    device const float *skip [[buffer(SEISMIC_BUFFER_SKIP)]],
    device const int *segments [[buffer(SEISMIC_BUFFER_SEGMENTS)]],
    device const int *stop [[buffer(SEISMIC_BUFFER_STOP)]],
    device const int *previous_bank [[buffer(SEISMIC_BUFFER_PREVIOUS_BANK)]],
    device const int *previous_tape [[buffer(SEISMIC_BUFFER_PREVIOUS_TAPE)]],
    device const int *following_bank [[buffer(SEISMIC_BUFFER_FOLLOWING_BANK)]],
    device const ulong *window [[buffer(SEISMIC_BUFFER_WINDOW)]],
    device const ulong *state [[buffer(SEISMIC_BUFFER_STATE)]],
    device const ulong *tape [[buffer(SEISMIC_BUFFER_TAPE)]],
    device state_space::Storage *mixed [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_shape [[threads_per_threadgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float b_block[SSS_SPAN * SEISMIC_DIM_N];
    threadgroup float c_block[SSS_SPAN * SEISMIC_DIM_N];
    threadgroup float x_block[SSS_SPAN * SEISMIC_TUNE_ROWS];
    threadgroup float delta_block[SSS_SPAN];
    threadgroup float decay_block[SSS_SPAN];
    const state_space::Operands in{projection, convolution, convolution_bias, rate, time_bias, skip, window, state,
        tape, mixed, seismic_words};
    const versions::Slots slots{segments, stop, previous_bank, previous_tape, following_bank};
    const uint head = group.x;
    const ulong block_row0 = ulong(group.y) * SEISMIC_TUNE_ROWS;
    const ulong row0 = block_row0 + ulong(simdgroup) * SSS_LANE_ROWS;
    if (group.z == SEISMIC_DIM_B) {
        state_space::zero_uncovered<SSS_LANE_ROWS>(in, slots, head, row0, lane);
        return;
    }
    const versions::Slot slot = versions::slot_of(slots, group.z, seismic_words);
    // The successor bank's window is no slot's source, so it is written first.
    if (group.y == 0)
        state_space::publish_window(in, slot, head, thread_index, threadgroup_shape.x);
    state_space::Rows<SSS_LANE_ROWS> rows;
    state_space::load_version<SSS_LANE_ROWS>(in, slot, head, row0, lane, rows);
    state_space::advance_rows<SSS_LANE_ROWS, SEISMIC_TUNE_ROWS, SSS_SPAN>(in, slot, slot.lo, head, block_row0, row0,
        rows, b_block, c_block, x_block, delta_block, decay_block, thread_index, threadgroup_shape.x, lane);
}
