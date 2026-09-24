// qwen_dense_output: the down projection of the product rows plus the
// residual rows they were gathered from (`out_rows`).
#define KERNEL_W0 SEISMIC_DOWN_WEIGHT
#include "common/projection.h"

typedef element::Act activation;

#define DENSE_OUTPUT_ARGUMENTS                                                          \
    device const float *residual [[buffer(SEISMIC_BUFFER_RESIDUAL)]],                   \
    device const uchar *product [[buffer(SEISMIC_BUFFER_PRODUCT)]],                     \
    device const uchar *down_weight [[buffer(SEISMIC_BUFFER_DOWN_WEIGHT)]],             \
    device const int *out_rows [[buffer(SEISMIC_BUFFER_OUT_ROWS)]],                     \
    device float *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],                           \
    device float *partials [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIALS)]],                 \
    device float *small_partials [[buffer(SEISMIC_BUFFER_SCRATCH_SMALL_PARTIALS)]],     \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

#define DENSE_OUTPUT_OPERANDS                                                         \
    projection::Plain<activation, projection::AllRows> in{product, SEISMIC_PRODUCT_STRIDE_0, \
        SEISMIC_PRODUCT_STRIDE_1, uint(SEISMIC_DIM_F), {}};                             \
    projection::Residual<activation, projection::SelectedRows> out{result,              \
        SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1, residual, SEISMIC_RESIDUAL_STRIDE_0, \
        SEISMIC_RESIDUAL_STRIDE_1, {out_rows}};                                         \
    projection::Weights<packets::W0> w{down_weight, KERNEL_W0_LAYOUT(SEISMIC_DIM_F), uint(SEISMIC_DIM_F)}

kernel void qwen_dense_output_gemv(DENSE_OUTPUT_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    DENSE_OUTPUT_OPERANDS;
    uint rows = uint(SEISMIC_DIM_O);
    PROJECTION_FOR_ROWS(rows,
        projection::gemv<packets::W0, SEISMIC_TUNE_SIMDGROUPS, SEISMIC_TUNE_ROWS, MAXM, SEISMIC_TUNE_LANES>(
            in, out, w, rows, uint(SEISMIC_DIM_H), uint(SEISMIC_DIM_F), tile, shared, sg, lane));
}

kernel void qwen_dense_output_batch(DENSE_OUTPUT_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    DENSE_OUTPUT_OPERANDS;
    projection::gemv_batch<packets::W0, SEISMIC_TUNE_BATCH_SIMDGROUPS, SEISMIC_TUNE_BATCH_ROWS>(in, out, w,
        uint(SEISMIC_DIM_O), uint(SEISMIC_DIM_H), uint(SEISMIC_DIM_F), tile, shared, sg, lane);
}

#define DENSE_OUTPUT_GEMM(TM, TN, SPLIT, PARTIALS)                                     \
    PROJECTION_GEMM_SHARED(shared, TM, TN);                                             \
    DENSE_OUTPUT_OPERANDS;                                                              \
    const uint m = uint(SEISMIC_DIM_O), n = uint(SEISMIC_DIM_H), k = uint(SEISMIC_DIM_F); \
    if (SPLIT == 1)                                                                     \
        projection::gemm<packets::W0, TM, TN>(in, out, w, m, n, k, tile.y, tile.x, shared, sg, lane); \
    else                                                                                \
        projection::gemm_part<packets::W0, TM, TN>(in, PARTIALS, w, m, n, k, SPLIT, tile.z, tile.y, tile.x, \
            shared, sg, lane)

// 17..64 rows: the fixed small-row tile and split.
kernel void qwen_dense_output_gemm_small(DENSE_OUTPUT_ARGUMENTS,
    uint3 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    DENSE_OUTPUT_GEMM(projection::small_tile_m, projection::small_tile_n, projection::small_split, small_partials);
}

kernel void qwen_dense_output_reduce_small(DENSE_OUTPUT_ARGUMENTS,
    uint index [[thread_position_in_grid]]) {
    DENSE_OUTPUT_OPERANDS;
    projection::gemm_reduce(out, small_partials, uint(SEISMIC_DIM_O), uint(SEISMIC_DIM_H), projection::small_split,
        index);
}

kernel void qwen_dense_output_gemm(DENSE_OUTPUT_ARGUMENTS,
    uint3 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    DENSE_OUTPUT_GEMM(SEISMIC_TUNE_TILE_M, SEISMIC_TUNE_TILE_N, SEISMIC_TUNE_SPLIT, partials);
}

kernel void qwen_dense_output_reduce(DENSE_OUTPUT_ARGUMENTS,
    uint index [[thread_position_in_grid]]) {
    DENSE_OUTPUT_OPERANDS;
    projection::gemm_reduce(out, partials, uint(SEISMIC_DIM_O), uint(SEISMIC_DIM_H), SEISMIC_TUNE_SPLIT, index);
}
