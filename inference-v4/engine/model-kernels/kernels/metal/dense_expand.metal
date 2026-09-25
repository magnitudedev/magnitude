// dense_expand: RMS prologue over the `out_rows` residual rows, the
// paired gate/up projection, and SiLU(gate)·up.
#define KERNEL_W0 SEISMIC_GATE_WEIGHT
#define KERNEL_W1 SEISMIC_UP_WEIGHT
#include "lib/projection/projection.h"

typedef element::Act activation;
typedef ELEMENT_OF(SEISMIC_NORM) norm_element;

#define DENSE_EXPAND_ARGUMENTS                                                          \
    device const float *residual [[buffer(SEISMIC_BUFFER_RESIDUAL)]],                   \
    device const uchar *norm [[buffer(SEISMIC_BUFFER_NORM)]],                           \
    device const uchar *gate_weight [[buffer(SEISMIC_BUFFER_GATE_WEIGHT)]],             \
    device const uchar *up_weight [[buffer(SEISMIC_BUFFER_UP_WEIGHT)]],                 \
    device const int *out_rows [[buffer(SEISMIC_BUFFER_OUT_ROWS)]],                     \
    device uchar *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],                           \
    device uchar *normalized [[buffer(SEISMIC_BUFFER_SCRATCH_NORMALIZED)]],             \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

#define DENSE_EXPAND_OPERANDS                                                           \
    projection::Rms<activation, norm_element, projection::SelectedRows> in{residual,    \
        SEISMIC_RESIDUAL_STRIDE_0, SEISMIC_RESIDUAL_STRIDE_1, norm, SEISMIC_NORM_STRIDE_0, \
        as_type<float>(uint(SEISMIC_PARAM_EPS)), uint(SEISMIC_DIM_H), {out_rows}};      \
    projection::SiluMul<activation> out{result, SEISMIC_RESULT_0_STRIDE_0,              \
        SEISMIC_RESULT_0_STRIDE_1};                                                     \
    projection::Weights<packets::W0> gate{gate_weight, KERNEL_W0_LAYOUT(SEISMIC_DIM_H), \
        uint(SEISMIC_DIM_H)};                                                           \
    projection::Weights<packets::W1> up{up_weight, KERNEL_W1_LAYOUT(SEISMIC_DIM_H), uint(SEISMIC_DIM_H)}

kernel void dense_expand_gemv(DENSE_EXPAND_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    DENSE_EXPAND_OPERANDS;
    uint rows = uint(SEISMIC_DIM_O);
    PROJECTION_SQUARES_SHARED(squares, decltype(in)::parts);
    projection::threadgroup_squares<SEISMIC_TUNE_SIMDGROUPS>(in, rows, squares, sg, lane);
    projection::SharedNorm<decltype(in)> x{in, squares};
    PROJECTION_FOR_ROWS(rows,
        projection::gemv_paired<packets::W0, packets::W1, SEISMIC_TUNE_SIMDGROUPS, SEISMIC_TUNE_ROWS, MAXM,
            SEISMIC_TUNE_LANES>(
            x, out, gate, up, rows, uint(SEISMIC_DIM_F), uint(SEISMIC_DIM_H), tile, shared, sg, lane));
}

kernel void dense_expand_batch(DENSE_EXPAND_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    DENSE_EXPAND_OPERANDS;
    PROJECTION_SQUARES_SHARED(squares, decltype(in)::parts);
    projection::threadgroup_squares<SEISMIC_TUNE_BATCH_SIMDGROUPS>(in, uint(SEISMIC_DIM_O), squares, sg, lane);
    projection::SharedNorm<decltype(in)> x{in, squares};
    projection::gemv_batch_paired<packets::W0, packets::W1, SEISMIC_TUNE_BATCH_SIMDGROUPS,
        SEISMIC_TUNE_BATCH_ROWS>(x, out,
        gate, up, uint(SEISMIC_DIM_O), uint(SEISMIC_DIM_F), uint(SEISMIC_DIM_H), tile, shared, sg, lane);
}

kernel void dense_expand_normalize(DENSE_EXPAND_ARGUMENTS,
    uint item [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    PROJECTION_NORMALIZE_SHARED(norms);
    DENSE_EXPAND_OPERANDS;
    projection::device_normalize<256>(in, item, normalized, uint(SEISMIC_DIM_H), norms, thread_index);
}

#define DENSE_EXPAND_GEMM(TM, TN)                                                       \
    PROJECTION_GEMM_SHARED(shared, TM, TN);                                             \
    DENSE_EXPAND_OPERANDS;                                                              \
    projection::Plain<activation, projection::AllRows> x{normalized, SEISMIC_DIM_H, 1, uint(SEISMIC_DIM_H), {}}; \
    projection::gemm_paired<packets::W0, packets::W1, TM, TN>(x, out, gate, up, uint(SEISMIC_DIM_O), \
        uint(SEISMIC_DIM_F), uint(SEISMIC_DIM_H), tile.y, tile.x, shared, sg, lane)

// 17..64 rows: the fixed small-row tile.
kernel void dense_expand_gemm_small(DENSE_EXPAND_ARGUMENTS,
    uint2 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    DENSE_EXPAND_GEMM(projection::small_tile_m, projection::small_tile_n);
}

kernel void dense_expand_gemm(DENSE_EXPAND_ARGUMENTS,
    uint2 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    DENSE_EXPAND_GEMM(SEISMIC_TUNE_TILE_M, SEISMIC_TUNE_TILE_N);
}
