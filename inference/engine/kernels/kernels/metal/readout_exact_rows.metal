// readout_exact_rows: level three of a certified selection (`readout.seismic`).
// One simdgroup per 32 vocabulary rows keeps the rows level two leaves
// (`fine`, `floor`) and projects them exactly over every plane, with the full
// pass's own routine (`lib/readout/progressive.h`); −inf elsewhere.
#include "lib/readout/progressive.h"

#ifdef SEISMIC_FORMING_READOUT_EXACT_ROWS_GATHER
kernel void readout_exact_rows_gather(
    device const uchar *features [[buffer(SEISMIC_BUFFER_FEATURES)]],
    device const uint *top [[buffer(SEISMIC_BUFFER_TOP)]],
    device const uint *bit3 [[buffer(SEISMIC_BUFFER_BIT3)]],
    device const uint *rest [[buffer(SEISMIC_BUFFER_REST)]],
    device const half *scales [[buffer(SEISMIC_BUFFER_SCALES)]],
    device const float *radius [[buffer(SEISMIC_BUFFER_RADIUS)]],
    device const float *fine [[buffer(SEISMIC_BUFFER_FINE)]],
    device const float *floor [[buffer(SEISMIC_BUFFER_FLOOR)]],
    device const float *lengths [[buffer(SEISMIC_BUFFER_LENGTH)]],
    device const uint *draws [[buffer(SEISMIC_BUFFER_DRAWS)]],
    device const float *temperature [[buffer(SEISMIC_BUFFER_TEMPERATURE)]],
    device const uint *mask [[buffer(SEISMIC_BUFFER_MASK)]],
    device const int *constrained [[buffer(SEISMIC_BUFFER_CONSTRAINED)]],
    device float *logits [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint tile [[threadgroup_position_in_grid]],
    uint tiles [[threadgroups_per_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint4 stage[PROGRESSIVE_STAGE_WORDS];
    const progressive::Planes planes{top, SEISMIC_TOP_STRIDE_0, bit3, SEISMIC_BIT3_STRIDE_0, rest,
        SEISMIC_REST_STRIDE_0, SEISMIC_REST_STRIDE_1, SEISMIC_REST_STRIDE_2, scales, SEISMIC_SCALES_STRIDE_0};
    const progressive::Selection selection{draws, SEISMIC_DRAWS_STRIDE_0, SEISMIC_DRAWS_STRIDE_1, temperature,
        SEISMIC_TEMPERATURE_STRIDE_0, mask, SEISMIC_MASK_STRIDE_0, SEISMIC_MASK_STRIDE_1, constrained,
        SEISMIC_CONSTRAINED_STRIDE_0};
    // Radius column 1: the 5-bit view's.
    progressive::gather<false>(features, SEISMIC_FEATURES_STRIDE_0, planes, radius + SEISMIC_RADIUS_STRIDE_1,
        nullptr, SEISMIC_RADIUS_STRIDE_0, fine, SEISMIC_FINE_STRIDE_0, SEISMIC_FINE_STRIDE_1, floor,
        SEISMIC_FLOOR_STRIDE_0, lengths, SEISMIC_LENGTH_STRIDE_0, selection, logits, SEISMIC_RESULT_0_STRIDE_0,
        SEISMIC_RESULT_0_STRIDE_1, nullptr, 0, nullptr, uint(SEISMIC_DIM_O), uint(SEISMIC_DIM_V),
        uint(SEISMIC_DIM_D), nullptr, nullptr, stage, tile, tiles, tid, threads, simdgroups, sg, lane);
}
#endif
