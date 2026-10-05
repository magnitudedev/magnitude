// dense_up: RMS prologue over the `out_rows` residual rows, the up
// projection, and A(act(A(up))) (`activation`: ReLU²). `dense_expand` with
// one weight stream.
#define KERNEL_W0 SEISMIC_UP_WEIGHT
#include "lib/projection/projection.h"

typedef element::Act activation;
typedef ELEMENT_OF(SEISMIC_NORM) norm_element;

#define DENSE_UP_ARGUMENTS                                                              \
    device const float *residual [[buffer(SEISMIC_BUFFER_RESIDUAL)]],                   \
    device const uchar *norm [[buffer(SEISMIC_BUFFER_NORM)]],                           \
    device const uchar *up_weight [[buffer(SEISMIC_BUFFER_UP_WEIGHT)]],                 \
    device const int *out_rows [[buffer(SEISMIC_BUFFER_OUT_ROWS)]],                     \
    device uchar *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],                           \
    device uchar *normalized [[buffer(SEISMIC_BUFFER_SCRATCH_NORMALIZED)]],             \
    device const float *up_scale [[buffer(SEISMIC_BUFFER_UP_SCALE)]],                   \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

#define DENSE_UP_OPERANDS                                                               \
    projection::Rms<activation, norm_element, projection::SelectedRows> in{residual,    \
        SEISMIC_RESIDUAL_STRIDE_0, SEISMIC_RESIDUAL_STRIDE_1, norm, SEISMIC_NORM_STRIDE_0, \
        as_type<float>(uint(SEISMIC_PARAM_EPS)), uint(SEISMIC_DIM_H), {out_rows}};      \
    const auto out = projection::scaling<(SEISMIC_DIM_US != 0)>::wrap(                  \
        projection::Activated<activation>{result, SEISMIC_RESULT_0_STRIDE_0,            \
            SEISMIC_RESULT_0_STRIDE_1, int(SEISMIC_PARAM_ACTIVATION)},                  \
        projection::scale_factor(up_scale, SEISMIC_DIM_US, 0, 0), 1.0f);                \
    projection::Weights<packets::W0> up{up_weight, KERNEL_W0_LAYOUT(SEISMIC_DIM_H), uint(SEISMIC_DIM_H)}

#ifdef SEISMIC_FORMING_DENSE_UP_GEMV
template <uint ROWS, uint LANES>
kernel void dense_up_gemv(DENSE_UP_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    DENSE_UP_OPERANDS;
    uint rows = uint(SEISMIC_DIM_O);
    PROJECTION_SQUARES_SHARED(squares, decltype(in)::parts);
    projection::threadgroup_squares_runtime(in, rows, squares, simdgroups, sg, lane);
    const auto x = projection::shared_norm(in, squares);
    PROJECTION_FOR_ROWS(rows,
        projection::gemv_runtime<packets::W0, ROWS, MAXM, LANES>(
            x, out, up, rows, uint(SEISMIC_DIM_F), uint(SEISMIC_DIM_H), tile, shared, simdgroups, sg, lane));
}
#endif

#ifdef SEISMIC_FORMING_DENSE_UP_BATCH
template <uint BATCH_ROWS>
kernel void dense_up_batch(DENSE_UP_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    DENSE_UP_OPERANDS;
    PROJECTION_SQUARES_SHARED(squares, decltype(in)::parts);
    projection::threadgroup_squares_runtime(in, uint(SEISMIC_DIM_O), squares, simdgroups, sg, lane);
    const auto x = projection::shared_norm(in, squares);
    projection::gemv_batch_runtime<packets::W0, BATCH_ROWS>(x, out, up, uint(SEISMIC_DIM_O),
        uint(SEISMIC_DIM_F), uint(SEISMIC_DIM_H), tile, shared, simdgroups, sg, lane);
}
#endif

#ifdef SEISMIC_FORMING_DENSE_UP_NORMALIZE
kernel void dense_up_normalize(DENSE_UP_ARGUMENTS,
    uint item [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    PROJECTION_NORMALIZE_SHARED(norms);
    DENSE_UP_OPERANDS;
    projection::device_normalize<256>(in, item, normalized, uint(SEISMIC_DIM_H), norms, thread_index);
}
#endif

#define DENSE_UP_GEMM(TM, TN)                                                           \
    PROJECTION_GEMM_SHARED(shared, TM, TN);                                             \
    DENSE_UP_OPERANDS;                                                                  \
    projection::Plain<activation, projection::AllRows> x{normalized, SEISMIC_DIM_H, 1, uint(SEISMIC_DIM_H), {}}; \
    projection::gemm<packets::W0, TM, TN>(x, out, up, uint(SEISMIC_DIM_O), uint(SEISMIC_DIM_F), \
        uint(SEISMIC_DIM_H), tile.y, tile.x, shared, sg, lane)

// 17..64 rows: the fixed small-row tile.
#ifdef SEISMIC_FORMING_DENSE_UP_GEMM_SMALL
kernel void dense_up_gemm_small(DENSE_UP_ARGUMENTS,
    uint2 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    DENSE_UP_GEMM(projection::small_tile_m, projection::small_tile_n);
}
#endif

#ifdef SEISMIC_FORMING_DENSE_UP_GEMM
template <uint TILE_M, uint TILE_N>
kernel void dense_up_gemm(DENSE_UP_ARGUMENTS,
    uint2 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    DENSE_UP_GEMM(TILE_M, TILE_N);
}
#endif
