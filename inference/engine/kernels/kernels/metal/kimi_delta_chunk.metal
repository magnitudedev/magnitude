// kimi_delta_chunk: `lib/recurrent/chunk.h`'s launches over Kimi Delta
// Attention's gates (one log decay per key channel).
#include "lib/recurrent/chunk.h"
#include "lib/recurrent/kimi.h"

kernel void kimi_delta_chunk_inputs(
    device const recurrent::Storage *projection [[buffer(SEISMIC_BUFFER_PROJECTION)]],
    device const recurrent::Storage *beta [[buffer(SEISMIC_BUFFER_BETA)]],
    device const recurrent::Storage *decay [[buffer(SEISMIC_BUFFER_DECAY)]],
    device const float *convolution [[buffer(SEISMIC_BUFFER_CONVOLUTION)]],
    device const float *rate [[buffer(SEISMIC_BUFFER_RATE)]],
    device const float *time_bias [[buffer(SEISMIC_BUFFER_TIME_BIAS)]],
    device const int *segments [[buffer(SEISMIC_BUFFER_SEGMENTS)]],
    device const int *stop [[buffer(SEISMIC_BUFFER_STOP)]],
    device const int *previous_bank [[buffer(SEISMIC_BUFFER_PREVIOUS_BANK)]],
    device const int *previous_tape [[buffer(SEISMIC_BUFFER_PREVIOUS_TAPE)]],
    device const int *following_bank [[buffer(SEISMIC_BUFFER_FOLLOWING_BANK)]],
    device const ulong *window [[buffer(SEISMIC_BUFFER_WINDOW)]],
    device float *inputs [[buffer(SEISMIC_BUFFER_SCRATCH_INPUTS)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    rch_inputs(kimi::channel_rule(beta, decay, rate, time_bias, seismic_words), projection, convolution, segments,
        stop, previous_bank, previous_tape, following_bank, window, inputs, group, simdgroup, lane, seismic_words);
}

kernel void kimi_delta_chunk_scan(
    device const recurrent::Storage *projection [[buffer(SEISMIC_BUFFER_PROJECTION)]],
    device const recurrent::Storage *beta [[buffer(SEISMIC_BUFFER_BETA)]],
    device const recurrent::Storage *decay [[buffer(SEISMIC_BUFFER_DECAY)]],
    device const float *convolution [[buffer(SEISMIC_BUFFER_CONVOLUTION)]],
    device const float *rate [[buffer(SEISMIC_BUFFER_RATE)]],
    device const float *time_bias [[buffer(SEISMIC_BUFFER_TIME_BIAS)]],
    device const int *segments [[buffer(SEISMIC_BUFFER_SEGMENTS)]],
    device const int *stop [[buffer(SEISMIC_BUFFER_STOP)]],
    device const int *previous_bank [[buffer(SEISMIC_BUFFER_PREVIOUS_BANK)]],
    device const int *previous_tape [[buffer(SEISMIC_BUFFER_PREVIOUS_TAPE)]],
    device const int *following_bank [[buffer(SEISMIC_BUFFER_FOLLOWING_BANK)]],
    device const ulong *window [[buffer(SEISMIC_BUFFER_WINDOW)]],
    device const ulong *delta [[buffer(SEISMIC_BUFFER_DELTA)]],
    device const ulong *tape [[buffer(SEISMIC_BUFFER_TAPE)]],
    device float *inputs [[buffer(SEISMIC_BUFFER_SCRATCH_INPUTS)]],
    device recurrent::Storage *mixed [[buffer(SEISMIC_BUFFER_SCRATCH_MIXED)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane_index [[thread_index_in_simdgroup]]) {
    threadgroup float queries[RCH_PIECE * RCH_INPUT_STRIDE];
    threadgroup float keys[RCH_PIECE * RCH_INPUT_STRIDE];
    threadgroup float values[RCH_PIECE * RCH_VALUE_STRIDE];
    threadgroup float gates[RCH_PIECE * RCH_INPUT_STRIDE + RCH_PIECE];
    threadgroup float decayed[RCH_PIECE * RCH_INPUT_STRIDE];
    threadgroup float shares[RCH_SHARES][2][RCH_PIECE * RCH_PIECE];
    rch_scan(kimi::channel_rule(beta, decay, rate, time_bias, seismic_words), RchDecays<SEISMIC_DIM_W>{decayed},
        projection, convolution, segments, stop, previous_bank, previous_tape, following_bank, window, delta, tape,
        inputs, mixed, queries, keys, values, gates, shares, group, thread_index, simdgroup, lane_index,
        seismic_words);
}

kernel void kimi_delta_chunk_gate(
    device const recurrent::Storage *gate [[buffer(SEISMIC_BUFFER_GATE)]],
    device const uchar *recurrent_norm [[buffer(SEISMIC_BUFFER_RECURRENT_NORM)]],
    device recurrent::Storage *gated [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    device const recurrent::Storage *mixed [[buffer(SEISMIC_BUFFER_SCRATCH_MIXED)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint item [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    recurrent::gate_row(mixed, kimi::gate_input(gate, seismic_words), recurrent_norm, gated,
        item / RCH_VALUE_HEADS, item % RCH_VALUE_HEADS, lane, seismic_words);
}
