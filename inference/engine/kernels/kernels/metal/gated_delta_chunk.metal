// gated_delta_chunk: `lib/recurrent/chunk.h`'s launches over Gated DeltaNet's
// gates (one decay per value head).
#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/recurrent/chunk.h"

kernel void gated_delta_chunk_prepare(
    device const recurrent::Storage *projection [[buffer(SEISMIC_BUFFER_PROJECTION)]],
    device const float *convolution [[buffer(SEISMIC_BUFFER_CONVOLUTION)]],
    device const float *rate [[buffer(SEISMIC_BUFFER_RATE)]],
    device const float *time_bias [[buffer(SEISMIC_BUFFER_TIME_BIAS)]],
    device const int *segments [[buffer(SEISMIC_BUFFER_SEGMENTS)]],
    device const int *stop [[buffer(SEISMIC_BUFFER_STOP)]],
    device const int *previous_bank [[buffer(SEISMIC_BUFFER_PREVIOUS_BANK)]],
    device const int *previous_tape [[buffer(SEISMIC_BUFFER_PREVIOUS_TAPE)]],
    device const int *following_bank [[buffer(SEISMIC_BUFFER_FOLLOWING_BANK)]],
    SEISMIC_OUT_WINDOW_PARAM(window),
    device uchar *pieces [[buffer(SEISMIC_BUFFER_SCRATCH_PIECES)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup rch::PrepareShared<1> sh;
    rch_prepare(recurrent::HeadRule{projection, rate, time_bias}, projection, convolution, segments, stop,
        previous_bank, previous_tape, following_bank, window, pieces, sh, group, thread_index, simdgroup, lane,
        seismic_words);
}

kernel void gated_delta_chunk_scan(
    device const recurrent::Storage *projection [[buffer(SEISMIC_BUFFER_PROJECTION)]],
    device const float *convolution [[buffer(SEISMIC_BUFFER_CONVOLUTION)]],
    device const float *rate [[buffer(SEISMIC_BUFFER_RATE)]],
    device const float *time_bias [[buffer(SEISMIC_BUFFER_TIME_BIAS)]],
    device const int *segments [[buffer(SEISMIC_BUFFER_SEGMENTS)]],
    device const int *stop [[buffer(SEISMIC_BUFFER_STOP)]],
    device const int *previous_bank [[buffer(SEISMIC_BUFFER_PREVIOUS_BANK)]],
    device const int *previous_tape [[buffer(SEISMIC_BUFFER_PREVIOUS_TAPE)]],
    device const int *following_bank [[buffer(SEISMIC_BUFFER_FOLLOWING_BANK)]],
    SEISMIC_OUT_WINDOW_PARAM(window),
    SEISMIC_OUT_DELTA_PARAM(delta),
    SEISMIC_OUT_TAPE_PARAM(tape),
    device uchar *pieces [[buffer(SEISMIC_BUFFER_SCRATCH_PIECES)]],
    device recurrent::Storage *mixed [[buffer(SEISMIC_BUFFER_SCRATCH_MIXED)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane_index [[thread_index_in_simdgroup]]) {
    threadgroup rch::ScanShared<1> shared;
    rch_scan(recurrent::HeadRule{projection, rate, time_bias}, projection, convolution, segments, stop,
        previous_bank, previous_tape, following_bank, window, delta, tape, pieces, mixed, shared, group,
        thread_index, simdgroup, lane_index, seismic_words);
}

kernel void gated_delta_chunk_gate(
    device const recurrent::Storage *projection [[buffer(SEISMIC_BUFFER_PROJECTION)]],
    device const uchar *recurrent_norm [[buffer(SEISMIC_BUFFER_RECURRENT_NORM)]],
    device const int *segments [[buffer(SEISMIC_BUFFER_SEGMENTS)]],
    SEISMIC_RESULT_0_PARAM(gated),
    device const recurrent::Storage *mixed [[buffer(SEISMIC_BUFFER_SCRATCH_MIXED)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint item [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    recurrent::gate_row(mixed, versions::covered_end(segments, seismic_words),
        recurrent::projected_gate(projection, seismic_words), recurrent_norm, element::Out<activation::Act>(gated),
        item / RCH_VALUE_HEADS, item % RCH_VALUE_HEADS, lane, seismic_words);
}
