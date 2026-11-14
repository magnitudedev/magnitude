// Row-sequential gated delta step (`recurrent::step`). A threadgroup
// of four simdgroups owns (value head, block of ROWS state rows, slot); each
// simdgroup keeps ROWS / 4 state rows in registers, W / 32 contiguous key
// columns per lane. State is read from version (previous_bank,
// previous_tape)[slot] and written only to bank following_bank[slot] after the
// slot's stop row, with the tape of the rows after it. The first row block of
// a head publishes its window channels before the state work, so that copy
// overlaps the state traffic. The raw outputs go to the `mixed` scratch; the
// last of a head's threadgroups to arrive (`arrivals[head]`, over every row
// block and slot) gates the head's rows into the result
// (`recurrent::gate_head`). ROWS never changes bits, and `gated_delta_chunk`
// gives its short slots the same raw outputs. Channels of the projection and
// window rows and the columns of the delta arena's rows must be contiguous
// (unit stride).
// State rows per simdgroup.
#define RST_LANE_ROWS (SEISMIC_TUNE_ROWS / 4)
// Rows whose prologue is staged at once: 8 KiB each of q and k.
#define RST_BLOCK (2048 / SEISMIC_DIM_W)

#include "lib/recurrent/convolution.h"

kernel void gated_delta_step(
    device const recurrent::Storage *projection [[buffer(SEISMIC_BUFFER_PROJECTION)]],
    device const float *convolution [[buffer(SEISMIC_BUFFER_CONVOLUTION)]],
    device const float *rate [[buffer(SEISMIC_BUFFER_RATE)]],
    device const float *time_bias [[buffer(SEISMIC_BUFFER_TIME_BIAS)]],
    device const uchar *recurrent_norm [[buffer(SEISMIC_BUFFER_RECURRENT_NORM)]],
    device const int *segments [[buffer(SEISMIC_BUFFER_SEGMENTS)]],
    device const int *stop [[buffer(SEISMIC_BUFFER_STOP)]],
    device const int *previous_bank [[buffer(SEISMIC_BUFFER_PREVIOUS_BANK)]],
    device const int *previous_tape [[buffer(SEISMIC_BUFFER_PREVIOUS_TAPE)]],
    device const int *following_bank [[buffer(SEISMIC_BUFFER_FOLLOWING_BANK)]],
    device const ulong *window [[buffer(SEISMIC_BUFFER_WINDOW)]],
    device const ulong *delta [[buffer(SEISMIC_BUFFER_DELTA)]],
    device const ulong *tape [[buffer(SEISMIC_BUFFER_TAPE)]],
    device recurrent::Storage *gated [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    arrive::coherent_device<recurrent::Storage> mixed [[buffer(SEISMIC_BUFFER_SCRATCH_MIXED)]],
    device atomic_uint *arrivals [[buffer(SEISMIC_BUFFER_SCRATCH_ARRIVALS)]],
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
    threadgroup uint last;
    const recurrent::Slot slot = recurrent::slot_of(segments, stop, previous_bank, previous_tape, following_bank,
        group.z, seismic_words);
    // The successor bank's window is no slot's source, so it is written first.
    if (group.y == 0)
        recurrent::publish_window(projection, window, slot, group.x, thread_index, threadgroup_shape.x,
            seismic_words);
    recurrent::step<RST_LANE_ROWS, RST_BLOCK>(recurrent::Convolving{projection, window, convolution, slot},
        recurrent::HeadRule{projection, rate, time_bias}, recurrent::projected_gate(projection, seismic_words),
        recurrent_norm, gated, mixed, arrivals, delta, tape, slot, group, query_block, key_block, value_block,
        beta_block, decay_block, &last, thread_index, threadgroup_shape.x, simdgroup, lane, seismic_words);
}
