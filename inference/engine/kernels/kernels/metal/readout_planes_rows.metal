// readout_planes_rows: final RMS prologue over the `out_rows` hidden rows and
// the exact vocabulary projection of a progressive head into F32 logits:
// `readout_head_rows`' launches over the projection library's packets of the
// head's exact view (`lib/readout/progressive.h`).
#include "lib/readout/progressive.h"

typedef progressive::activation activation;
typedef ELEMENT_OF(SEISMIC_NORM) norm_element;

#define PLANES_ROWS_ARGUMENTS                                                             \
    device const float *hidden [[buffer(SEISMIC_BUFFER_HIDDEN)]],                       \
    device const uchar *norm [[buffer(SEISMIC_BUFFER_NORM)]],                           \
    device const uint *top [[buffer(SEISMIC_BUFFER_TOP)]],                              \
    device const uint *bit3 [[buffer(SEISMIC_BUFFER_BIT3)]],                            \
    device const uint *rest [[buffer(SEISMIC_BUFFER_REST)]],                            \
    device const half *scales [[buffer(SEISMIC_BUFFER_SCALES)]],                        \
    device const int *out_rows [[buffer(SEISMIC_BUFFER_OUT_ROWS)]],                     \
    device uchar *logits [[buffer(SEISMIC_RESULT_0_BUFFER)]],                           \
    device uchar *normalized [[buffer(SEISMIC_BUFFER_SCRATCH_NORMALIZED)]],             \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

#define PLANES_ROWS_OPERANDS                                                              \
    const uint k = uint(SEISMIC_DIM_D);                                                 \
    projection::Rms<activation, norm_element, projection::SelectedRows> in{hidden, SEISMIC_HIDDEN_STRIDE_0, \
        SEISMIC_HIDDEN_STRIDE_1, norm, SEISMIC_NORM_STRIDE_0,                           \
        as_type<float>(uint(SEISMIC_PARAM_EPSILON)), k, {out_rows}};                    \
    const projection::Logits out{logits, SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1, 0.0f}; \
    const projection::Weights<packets::ProgressiveExact> w{top, SEISMIC_TOP_STRIDE_0, bit3, \
        SEISMIC_BIT3_STRIDE_0, rest, SEISMIC_REST_STRIDE_0, SEISMIC_REST_STRIDE_1, SEISMIC_REST_STRIDE_2, \
        scales, SEISMIC_SCALES_STRIDE_0}

#ifdef SEISMIC_FORMING_READOUT_PLANES_ROWS_GEMV
template <uint ROWS, uint LANES>
kernel void readout_planes_rows_gemv(PLANES_ROWS_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    PLANES_ROWS_OPERANDS;
    uint rows = uint(SEISMIC_DIM_O);
    PROJECTION_SQUARES_SHARED(squares, decltype(in)::parts);
    projection::threadgroup_squares_runtime(in, rows, squares, simdgroups, sg, lane);
    const auto x = projection::shared_norm(in, squares);
    PROJECTION_FOR_ROWS(rows,
        projection::gemv_runtime<packets::ProgressiveExact, ROWS, MAXM, LANES>(
            x, out, w, rows, uint(SEISMIC_DIM_V), k, tile, shared, simdgroups, sg, lane));
}
#endif

#ifdef SEISMIC_FORMING_READOUT_PLANES_ROWS_BATCH
template <uint BATCH_ROWS>
kernel void readout_planes_rows_batch(PLANES_ROWS_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    PLANES_ROWS_OPERANDS;
    PROJECTION_SQUARES_SHARED(squares, decltype(in)::parts);
    projection::threadgroup_squares_runtime(in, uint(SEISMIC_DIM_O), squares, simdgroups, sg, lane);
    const auto x = projection::shared_norm(in, squares);
    projection::gemv_batch_runtime<packets::ProgressiveExact, BATCH_ROWS>(x, out, w,
        uint(SEISMIC_DIM_O), uint(SEISMIC_DIM_V), k, tile, shared, simdgroups, sg, lane);
}
#endif

#ifdef SEISMIC_FORMING_READOUT_PLANES_ROWS_STAGE
kernel void readout_planes_rows_stage(PLANES_ROWS_ARGUMENTS,
    uint item [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    PROJECTION_NORMALIZE_SHARED(norms);
    PLANES_ROWS_OPERANDS;
    projection::device_normalize<256>(in, item, normalized, k, norms, thread_index);
}
#endif

#ifdef SEISMIC_FORMING_READOUT_PLANES_ROWS_GEMM
template <uint TILE_M, uint TILE_N>
kernel void readout_planes_rows_gemm(PLANES_ROWS_ARGUMENTS,
    uint2 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    PROJECTION_GEMM_SHARED(shared, TILE_M, TILE_N);
    PLANES_ROWS_OPERANDS;
    projection::Plain<activation, projection::AllRows> x{normalized, k, 1, k, {}};
    projection::gemm<packets::ProgressiveExact, TILE_M, TILE_N>(x, out, w, uint(SEISMIC_DIM_O),
        uint(SEISMIC_DIM_V), k, tile.y, tile.x, shared, sg, lane);
}
#endif
