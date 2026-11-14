// `gated_delta_step` over the convolved channels `gated_delta_project_convolved`
// publishes: the same threadgroups (value head, block of ROWS state rows,
// slot), arithmetic and gating (`recurrent::step`), with a prologue that
// loads each row's convolved q, k and v channels instead of convolving them.
// The projection launch published the successor windows, so no threadgroup
// touches a window. ROWS never changes bits. Channels of the projection and convolved rows and the
// columns of the delta arena's rows must be contiguous (unit stride).
// State rows per simdgroup.
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
    recurrent::step<RST_LANE_ROWS, RST_BLOCK>(recurrent::Convolved{convolved, SEISMIC_CONVOLVED_STRIDE_0},
        recurrent::HeadRule{projection, rate, time_bias}, recurrent::projected_gate(projection, seismic_words),
        recurrent_norm, gated, mixed, arrivals, delta, tape, slot, group, query_block, key_block, value_block,
        beta_block, decay_block, &last, thread_index, threadgroup_shape.x, simdgroup, lane, seismic_words);
}
