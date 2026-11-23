// `gated_delta_step` over the convolved channels `gated_delta_project_convolved`
// publishes: the same threadgroups (value head, block of ROWS state rows,
// slot), arithmetic and gating (`recurrent::step`), with a prologue that
// loads each row's convolved q, k and v channels instead of convolving them.
// The projection launch published the successor windows, so no threadgroup
// touches a window. ROWS never changes bits. Channels of the projection and convolved rows and the
// columns of the delta arena's rows must be contiguous (unit stride).
// State rows per simdgroup.
#define KERNEL_A SEISMIC_ELEMENT_A
#define RST_LANE_ROWS (SEISMIC_TUNE_ROWS / 4)
// Rows whose prologue is staged at once: 8 KiB each of q and k.
#define RST_BLOCK (2048 / SEISMIC_DIM_W)

#include "lib/recurrent/recurrent.h"

kernel void gated_delta_step_convolved(
    device const recurrent::Storage *projection [[buffer(SEISMIC_BUFFER_PROJECTION)]],
    device const float *convolved [[buffer(SEISMIC_BUFFER_CONVOLVED)]],
    device const float *rate [[buffer(SEISMIC_BUFFER_RATE)]],
    device const float *time_bias [[buffer(SEISMIC_BUFFER_TIME_BIAS)]],
    device const uchar *recurrent_norm [[buffer(SEISMIC_BUFFER_RECURRENT_NORM)]],
    device const int *segments [[buffer(SEISMIC_BUFFER_SEGMENTS)]],
    device const int *stop [[buffer(SEISMIC_BUFFER_STOP)]],
    device const int *previous_bank [[buffer(SEISMIC_BUFFER_PREVIOUS_BANK)]],
    device const int *previous_tape [[buffer(SEISMIC_BUFFER_PREVIOUS_TAPE)]],
    device const int *following_bank [[buffer(SEISMIC_BUFFER_FOLLOWING_BANK)]],
    SEISMIC_OUT_DELTA_PARAM(delta),
    SEISMIC_OUT_TAPE_PARAM(tape),
    SEISMIC_RESULT_0_PARAM(gated),
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
    const recurrent::Slot slot = versions::slot_of({segments, stop, previous_bank, previous_tape, following_bank},
        group.z, seismic_words);
    recurrent::step<RST_LANE_ROWS, RST_BLOCK>(recurrent::Convolved{convolved, SEISMIC_CONVOLVED_STRIDE_0},
        recurrent::HeadRule{projection, rate, time_bias}, recurrent::projected_gate(projection, seismic_words),
        recurrent_norm, element::Out<activation::Act>(gated), mixed, arrivals, delta, tape, slot,
        versions::covered_end(segments, seismic_words), group, query_block,
        key_block, value_block, beta_block, decay_block, &last, thread_index, threadgroup_shape.x, simdgroup, lane,
        seismic_words);
}

// M >= 3 staging pass: L2-normalizes q and k once per (row, key_head) into
// `qk_norm` and computes (beta, decay) once per (row, value_head) into `gates`.
kernel void gated_delta_step_convolved_prepare(
    device const recurrent::Storage *projection [[buffer(SEISMIC_BUFFER_PROJECTION)]],
    device const float *convolved [[buffer(SEISMIC_BUFFER_CONVOLVED)]],
    device const float *rate [[buffer(SEISMIC_BUFFER_RATE)]],
    device const float *time_bias [[buffer(SEISMIC_BUFFER_TIME_BIAS)]],
    device float *qk_norm [[buffer(SEISMIC_BUFFER_SCRATCH_QK_NORM)]],
    device float2 *gates [[buffer(SEISMIC_BUFFER_SCRATCH_GATES)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    recurrent::prepare_convolved(recurrent::Convolved{convolved, SEISMIC_CONVOLVED_STRIDE_0},
        recurrent::HeadRule{projection, rate, time_bias}, qk_norm, gates, group, thread_index, simdgroup, lane,
        seismic_words);
}

// M >= 3 step pass: consumes `qk_norm` and `gates` directly from L2-resident
// scratch with no q/k/v/gate threadgroup staging (4 bytes TG memory total).
kernel void gated_delta_step_convolved_prepared(
    device const recurrent::Storage *projection [[buffer(SEISMIC_BUFFER_PROJECTION)]],
    device const float *convolved [[buffer(SEISMIC_BUFFER_CONVOLVED)]],
    device const uchar *recurrent_norm [[buffer(SEISMIC_BUFFER_RECURRENT_NORM)]],
    device const int *segments [[buffer(SEISMIC_BUFFER_SEGMENTS)]],
    device const int *stop [[buffer(SEISMIC_BUFFER_STOP)]],
    device const int *previous_bank [[buffer(SEISMIC_BUFFER_PREVIOUS_BANK)]],
    device const int *previous_tape [[buffer(SEISMIC_BUFFER_PREVIOUS_TAPE)]],
    device const int *following_bank [[buffer(SEISMIC_BUFFER_FOLLOWING_BANK)]],
    SEISMIC_OUT_DELTA_PARAM(delta),
    SEISMIC_OUT_TAPE_PARAM(tape),
    SEISMIC_RESULT_0_PARAM(gated),
    arrive::coherent_device<recurrent::Storage> mixed [[buffer(SEISMIC_BUFFER_SCRATCH_MIXED)]],
    device atomic_uint *arrivals [[buffer(SEISMIC_BUFFER_SCRATCH_ARRIVALS)]],
    device const float *qk_norm [[buffer(SEISMIC_BUFFER_SCRATCH_QK_NORM)]],
    device const float2 *gates [[buffer(SEISMIC_BUFFER_SCRATCH_GATES)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_shape [[threads_per_threadgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint last;
    const recurrent::Slot slot = versions::slot_of({segments, stop, previous_bank, previous_tape, following_bank},
        group.z, seismic_words);
    recurrent::step_prepared<RST_LANE_ROWS>(
        recurrent::PreparedConvolved{qk_norm, gates, convolved, SEISMIC_CONVOLVED_STRIDE_0},
        recurrent::projected_gate(projection, seismic_words), recurrent_norm, element::Out<activation::Act>(gated),
        mixed, arrivals, delta, tape, slot, versions::covered_end(segments, seismic_words), group, &last,
        thread_index, threadgroup_shape.x, simdgroup, lane, seismic_words);
}
