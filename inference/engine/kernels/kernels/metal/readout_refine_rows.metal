// readout_refine_rows: level two of a certified selection (`readout.seismic`).
// One simdgroup per 32 vocabulary rows keeps the rows level one leaves
// (`coarse`, `floor`), adds bit 3 to their logits (the 5-bit view) and
// publishes the raised threshold (`lib/readout/progressive.h`).
#include "lib/readout/progressive.h"

#ifdef SEISMIC_FORMING_READOUT_REFINE_ROWS_GATHER
kernel void readout_refine_rows_gather(
    device const uchar *features [[buffer(SEISMIC_BUFFER_FEATURES)]],
    device const uint *bit3 [[buffer(SEISMIC_BUFFER_BIT3)]],
    device const half *scales [[buffer(SEISMIC_BUFFER_SCALES)]],
    device const float *radius [[buffer(SEISMIC_BUFFER_RADIUS)]],
    device const float *coarse [[buffer(SEISMIC_BUFFER_COARSE)]],
    device const float *floor [[buffer(SEISMIC_BUFFER_FLOOR)]],
    device const float *lengths [[buffer(SEISMIC_BUFFER_LENGTH)]],
    device const uint *draws [[buffer(SEISMIC_BUFFER_DRAWS)]],
    device const float *temperature [[buffer(SEISMIC_BUFFER_TEMPERATURE)]],
    device const uint *mask [[buffer(SEISMIC_BUFFER_MASK)]],
    device const int *constrained [[buffer(SEISMIC_BUFFER_CONSTRAINED)]],
    device float *logits [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    device float *threshold [[buffer(SEISMIC_RESULT_1_BUFFER)]],
    device atomic_uint *bounds [[buffer(SEISMIC_BUFFER_SCRATCH_BOUNDS)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint tile [[threadgroup_position_in_grid]],
    uint tiles [[threadgroups_per_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup atomic_uint lowest[PROGRESSIVE_ROWS];
    threadgroup uint last;
    threadgroup uint4 stage[PROGRESSIVE_STAGE_WORDS];
    const progressive::Planes planes{nullptr, 0, bit3, SEISMIC_BIT3_STRIDE_0, nullptr, 0, 0, 0, scales,
        SEISMIC_SCALES_STRIDE_0};
    const progressive::Selection selection{draws, SEISMIC_DRAWS_STRIDE_0, SEISMIC_DRAWS_STRIDE_1, temperature,
        SEISMIC_TEMPERATURE_STRIDE_0, mask, SEISMIC_MASK_STRIDE_0, SEISMIC_MASK_STRIDE_1, constrained,
        SEISMIC_CONSTRAINED_STRIDE_0};
    // Radius columns 0 and 1: the 4- and 5-bit views'.
    progressive::gather<true>(features, SEISMIC_FEATURES_STRIDE_0, planes, radius, radius + SEISMIC_RADIUS_STRIDE_1,
        SEISMIC_RADIUS_STRIDE_0, coarse, SEISMIC_COARSE_STRIDE_0, SEISMIC_COARSE_STRIDE_1, floor,
        SEISMIC_FLOOR_STRIDE_0, lengths, SEISMIC_LENGTH_STRIDE_0, selection, logits, SEISMIC_RESULT_0_STRIDE_0,
        SEISMIC_RESULT_0_STRIDE_1, threshold, SEISMIC_RESULT_1_STRIDE_0, bounds, uint(SEISMIC_DIM_O),
        uint(SEISMIC_DIM_V), uint(SEISMIC_DIM_D), lowest, &last, stage, tile, tiles, tid, threads, simdgroups, sg, lane);
}
#endif
