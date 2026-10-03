// `gated_delta_step` over the convolved channels `gated_delta_project_convolved`
// publishes: the same threadgroups (value head, block of ROWS state rows,
// slot), arithmetic and gating (`recurrent::advance_rows`,
// `recurrent::gate_head`), with a prologue that loads each row's convolved
// q, k and v channels instead of convolving them. The projection launch
// published the successor windows, so no threadgroup touches a window. ROWS
// never changes bits. Channels of the projection and convolved rows and the
// columns of the delta arena's rows must be contiguous (unit stride).
// State rows per simdgroup.
#define RST_LANE_ROWS (SEISMIC_TUNE_ROWS / 4)
// Rows whose prologue is staged at once: 8 KiB each of q and k.
#define RST_BLOCK (2048 / SEISMIC_DIM_W)

#include "lib/recurrent/recurrent.h"
#include "lib/core/arrive.h"

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
    device recurrent::Storage *mixed [[buffer(SEISMIC_BUFFER_SCRATCH_MIXED)]],
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
    const uint threads = threadgroup_shape.x;
    const ulong head = group.x;
    const ulong block_row0 = ulong(group.y) * SEISMIC_TUNE_ROWS;
    const ulong row0 = block_row0 + ulong(simdgroup) * RST_LANE_ROWS;
    const recurrent::Slot slot = recurrent::slot_of(segments, stop, previous_bank, previous_tape, following_bank,
        group.z, seismic_words);
    recurrent::advance_rows<RST_LANE_ROWS, SEISMIC_TUNE_ROWS, RST_BLOCK, SEISMIC_DIM_W, SEISMIC_TUNE_ROWS>(
        projection, recurrent::Convolved{convolved, SEISMIC_CONVOLVED_STRIDE_0}, rate, time_bias, delta, tape, mixed, slot, slot.lo, head,
        block_row0, row0, query_block, key_block, value_block, beta_block, decay_block, thread_index, threads,
        simdgroup, lane, seismic_words);
    // Rows after the last slot belong to no sequence; their raw output is zero.
    if (group.z + 1 == SEISMIC_DIM_B && lane < RST_LANE_ROWS) {
        for (ulong row = ulong(slot.hi); row < SEISMIC_DIM_M; ++row)
            mixed[recurrent::raw_index(row, head, row0 + lane, seismic_words)] = element::Act::store(0.0f);
    }
    const uint expected = uint(SEISMIC_DIM_W / SEISMIC_TUNE_ROWS * SEISMIC_DIM_B);
    if (!arrive::last(arrivals + head, expected, &last, thread_index))
        return;
    recurrent::gate_head(mixed, projection, recurrent_norm, gated, head, thread_index, threads, seismic_words);
}
