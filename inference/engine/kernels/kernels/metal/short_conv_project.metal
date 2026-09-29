// short_conv_project: RMS prologue over every residual row, then one
// segmented projection: threadgroups [0, S) along x pair the B and X
// streams into u = B * X (F32, columns 0..CH), the rest project C (F32,
// columns CH..2CH). S is the u segment's tile count at the launch's tiling.
#define KERNEL_W0 SEISMIC_B_WEIGHT
#define KERNEL_W1 SEISMIC_X_WEIGHT
#define KERNEL_W2 SEISMIC_C_WEIGHT
#include "lib/projection/projection.h"

typedef element::Act activation;
typedef ELEMENT_OF(SEISMIC_NORM) norm_element;

#define SHORT_CONV_PROJECT_ARGUMENTS                                                    \
    device const float *residual [[buffer(SEISMIC_BUFFER_RESIDUAL)]],                   \
    device const uchar *norm [[buffer(SEISMIC_BUFFER_NORM)]],                           \
    device const uchar *b_weight [[buffer(SEISMIC_BUFFER_B_WEIGHT)]],                   \
    device const uchar *c_weight [[buffer(SEISMIC_BUFFER_C_WEIGHT)]],                   \
    device const uchar *x_weight [[buffer(SEISMIC_BUFFER_X_WEIGHT)]],                   \
    device uchar *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],                           \
    device uchar *normalized [[buffer(SEISMIC_BUFFER_SCRATCH_NORMALIZED)]],             \
    device const float *b_scale [[buffer(SEISMIC_BUFFER_B_SCALE)]],                     \
    device const float *c_scale [[buffer(SEISMIC_BUFFER_C_SCALE)]],                     \
    device const float *x_scale [[buffer(SEISMIC_BUFFER_X_SCALE)]],                     \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

#define SHORT_CONV_PROJECT_OPERANDS                                                     \
    const uint channels = uint(SEISMIC_DIM_CH), hidden = uint(SEISMIC_DIM_H);           \
    projection::Rms<activation, norm_element, projection::AllRows> in{residual,         \
        SEISMIC_RESIDUAL_STRIDE_0, SEISMIC_RESIDUAL_STRIDE_1, norm, SEISMIC_NORM_STRIDE_0, \
        as_type<float>(uint(SEISMIC_PARAM_EPS)), hidden, {}};                           \
    const auto u = projection::scaling<(SEISMIC_DIM_BS != 0 || SEISMIC_DIM_XS != 0)>::wrap( \
        projection::Mul{reinterpret_cast<device float *>(result), SEISMIC_RESULT_0_STRIDE_0, \
            SEISMIC_RESULT_0_STRIDE_1},                                                 \
        projection::scale_factor(b_scale, SEISMIC_DIM_BS, 0, 0),                        \
        projection::scale_factor(x_scale, SEISMIC_DIM_XS, 0, 0));                       \
    const auto c = projection::scaling<(SEISMIC_DIM_CS != 0)>::wrap(                    \
        projection::Store<element::F32>{result, SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1, channels}, \
        projection::scale_factor(c_scale, SEISMIC_DIM_CS, 0, 0), 1.0f);                 \
    projection::Weights<packets::W0> b_rows{b_weight, KERNEL_W0_LAYOUT(SEISMIC_DIM_H), hidden}; \
    projection::Weights<packets::W1> x_rows{x_weight, KERNEL_W1_LAYOUT(SEISMIC_DIM_H), hidden}; \
    projection::Weights<packets::W2> c_rows{c_weight, KERNEL_W2_LAYOUT(SEISMIC_DIM_H), hidden}

#ifdef SEISMIC_FORMING_SHORT_CONV_PROJECT_GEMV
template <uint ROWS, uint LANES>
kernel void short_conv_project_gemv(SHORT_CONV_PROJECT_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    SHORT_CONV_PROJECT_OPERANDS;
    uint rows = uint(SEISMIC_DIM_M);
    const uint per_tile = simdgroups * ROWS * (32u / LANES);
    const uint segment = (channels + per_tile - 1u) / per_tile;
    PROJECTION_SQUARES_SHARED(squares, decltype(in)::parts);
    projection::threadgroup_squares_runtime(in, rows, squares, simdgroups, sg, lane);
    projection::SharedNorm<decltype(in)> x{in, squares};
    if (tile < segment) {
        PROJECTION_FOR_ROWS(rows,
            projection::gemv_paired_runtime<packets::W0, packets::W1, ROWS, MAXM, LANES>(
                x, u, b_rows, x_rows, rows, channels, hidden, tile, shared, simdgroups, sg, lane));
    } else {
        PROJECTION_FOR_ROWS(rows,
            projection::gemv_runtime<packets::W2, ROWS, MAXM, LANES>(
                x, c, c_rows, rows, channels, hidden, tile - segment, shared, simdgroups, sg, lane));
    }
}
#endif

#ifdef SEISMIC_FORMING_SHORT_CONV_PROJECT_BATCH
template <uint BATCH_ROWS>
kernel void short_conv_project_batch(SHORT_CONV_PROJECT_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    SHORT_CONV_PROJECT_OPERANDS;
    const uint rows = uint(SEISMIC_DIM_M);
    const uint per_tile = simdgroups * BATCH_ROWS * 8u;
    const uint segment = (channels + per_tile - 1u) / per_tile;
    PROJECTION_SQUARES_SHARED(squares, decltype(in)::parts);
    projection::threadgroup_squares_runtime(in, rows, squares, simdgroups, sg, lane);
    projection::SharedNorm<decltype(in)> x{in, squares};
    if (tile < segment)
        projection::gemv_batch_paired_runtime<packets::W0, packets::W1, BATCH_ROWS>(x, u, b_rows, x_rows, rows,
            channels, hidden, tile, shared, simdgroups, sg, lane);
    else
        projection::gemv_batch_runtime<packets::W2, BATCH_ROWS>(x, c, c_rows, rows, channels, hidden,
            tile - segment, shared, simdgroups, sg, lane);
}
#endif

#ifdef SEISMIC_FORMING_SHORT_CONV_PROJECT_NORMALIZE
kernel void short_conv_project_normalize(SHORT_CONV_PROJECT_ARGUMENTS,
    uint item [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    PROJECTION_NORMALIZE_SHARED(norms);
    SHORT_CONV_PROJECT_OPERANDS;
    projection::device_normalize<256>(in, item, normalized, hidden, norms, thread_index);
}
#endif

#define SHORT_CONV_PROJECT_GEMM(TM, TN)                                                 \
    PROJECTION_GEMM_SHARED(shared, TM, TN);                                             \
    SHORT_CONV_PROJECT_OPERANDS;                                                        \
    projection::Plain<activation, projection::AllRows> x{normalized, SEISMIC_DIM_H, 1, hidden, {}}; \
    const uint segment = (channels + TN / 2u - 1u) / (TN / 2u);                         \
    if (tile.x < segment)                                                               \
        projection::gemm_paired<packets::W0, packets::W1, TM, TN>(x, u, b_rows, x_rows, uint(SEISMIC_DIM_M), \
            channels, hidden, tile.y, tile.x, shared, sg, lane);                        \
    else                                                                                \
        projection::gemm<packets::W2, TM, TN>(x, c, c_rows, uint(SEISMIC_DIM_M), channels, hidden, tile.y, \
            tile.x - segment, shared, sg, lane)

// 17..64 rows: the fixed small-row tile.
#ifdef SEISMIC_FORMING_SHORT_CONV_PROJECT_GEMM_SMALL
kernel void short_conv_project_gemm_small(SHORT_CONV_PROJECT_ARGUMENTS,
    uint2 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    SHORT_CONV_PROJECT_GEMM(projection::small_tile_m, projection::small_tile_n);
}
#endif

#ifdef SEISMIC_FORMING_SHORT_CONV_PROJECT_GEMM
template <uint TILE_M, uint TILE_N>
kernel void short_conv_project_gemm(SHORT_CONV_PROJECT_ARGUMENTS,
    uint2 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    SHORT_CONV_PROJECT_GEMM(TILE_M, TILE_N);
}
#endif
