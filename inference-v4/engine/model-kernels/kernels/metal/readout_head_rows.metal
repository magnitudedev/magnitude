// readout_head_rows: final RMS prologue over the `out_rows` hidden rows and the
// vocabulary projection into F32 logits.
#define KERNEL_W0 SEISMIC_WEIGHT
#include "common/projection.h"

typedef element::Act activation;
typedef ELEMENT_OF(SEISMIC_NORM) norm_element;

#define HEAD_ROWS_ARGUMENTS                                                             \
    device const float *hidden [[buffer(SEISMIC_BUFFER_HIDDEN)]],                       \
    device const uchar *norm [[buffer(SEISMIC_BUFFER_NORM)]],                           \
    device const uchar *weight [[buffer(SEISMIC_BUFFER_WEIGHT)]],                       \
    device const int *out_rows [[buffer(SEISMIC_BUFFER_OUT_ROWS)]],                     \
    device uchar *logits [[buffer(SEISMIC_RESULT_0_BUFFER)]],                           \
    device uchar *normalized [[buffer(SEISMIC_BUFFER_SCRATCH_NORMALIZED)]],             \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

#define HEAD_ROWS_OPERANDS                                                              \
    const uint k = uint(SEISMIC_DIM_D);                                                 \
    projection::Rms<activation, norm_element, projection::SelectedRows> in{hidden, SEISMIC_HIDDEN_STRIDE_0, \
        SEISMIC_HIDDEN_STRIDE_1, norm, SEISMIC_NORM_STRIDE_0,                           \
        as_type<float>(uint(SEISMIC_PARAM_EPSILON)), k, {out_rows}};                    \
    projection::Store<element::F32> out{logits, SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1, 0}; \
    projection::Weights<packets::W0> w{weight, KERNEL_W0_LAYOUT(k), k}

kernel void readout_head_rows_gemv(HEAD_ROWS_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    HEAD_ROWS_OPERANDS;
    uint rows = uint(SEISMIC_DIM_O);
    PROJECTION_SQUARES_SHARED(squares, decltype(in)::parts);
    projection::threadgroup_squares<SEISMIC_TUNE_SIMDGROUPS>(in, rows, squares, sg, lane);
    projection::SharedNorm<decltype(in)> x{in, squares};
    PROJECTION_FOR_ROWS(rows,
        projection::gemv<packets::W0, SEISMIC_TUNE_SIMDGROUPS, SEISMIC_TUNE_ROWS, MAXM, SEISMIC_TUNE_LANES>(
            x, out, w, rows, uint(SEISMIC_DIM_V), k, tile, shared, sg, lane));
}

kernel void readout_head_rows_batch(HEAD_ROWS_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    HEAD_ROWS_OPERANDS;
    PROJECTION_SQUARES_SHARED(squares, decltype(in)::parts);
    projection::threadgroup_squares<SEISMIC_TUNE_BATCH_SIMDGROUPS>(in, uint(SEISMIC_DIM_O), squares, sg, lane);
    projection::SharedNorm<decltype(in)> x{in, squares};
    projection::gemv_batch<packets::W0, SEISMIC_TUNE_BATCH_SIMDGROUPS, SEISMIC_TUNE_BATCH_ROWS>(x, out, w,
        uint(SEISMIC_DIM_O), uint(SEISMIC_DIM_V), k, tile, shared, sg, lane);
}

kernel void readout_head_rows_stage(HEAD_ROWS_ARGUMENTS,
    uint item [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    PROJECTION_NORMALIZE_SHARED(norms);
    HEAD_ROWS_OPERANDS;
    projection::device_normalize<256>(in, item, normalized, k, norms, thread_index);
}

kernel void readout_head_rows_gemm(HEAD_ROWS_ARGUMENTS,
    uint2 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    PROJECTION_GEMM_SHARED(shared, SEISMIC_TUNE_TILE_M, SEISMIC_TUNE_TILE_N);
    HEAD_ROWS_OPERANDS;
    projection::Plain<activation, projection::AllRows> x{normalized, k, 1, k, {}};
    projection::gemm<packets::W0, SEISMIC_TUNE_TILE_M, SEISMIC_TUNE_TILE_N>(x, out, w, uint(SEISMIC_DIM_O),
        uint(SEISMIC_DIM_V), k, tile.y, tile.x, shared, sg, lane);
}
