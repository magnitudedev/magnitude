// dense_output: the down projection of the product rows plus the
// residual rows they were gathered from (`out_rows`).
#define KERNEL_W0 SEISMIC_DOWN_WEIGHT
#include "lib/projection/projection.h"

typedef element::Act activation;

#define DENSE_OUTPUT_ARGUMENTS                                                          \
    device const float *residual [[buffer(SEISMIC_BUFFER_RESIDUAL)]],                   \
    device const uchar *product [[buffer(SEISMIC_BUFFER_PRODUCT)]],                     \
    device const uchar *down_weight [[buffer(SEISMIC_BUFFER_DOWN_WEIGHT)]],             \
    device const int *out_rows [[buffer(SEISMIC_BUFFER_OUT_ROWS)]],                     \
    device float *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],                           \
    device float *partials [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIALS)]],                 \
    device float *small_partials [[buffer(SEISMIC_BUFFER_SCRATCH_SMALL_PARTIALS)]],     \
    device uchar *fragments [[buffer(SEISMIC_BUFFER_SCRATCH_FRAGMENTS)]],               \
    device const float *down_scale [[buffer(SEISMIC_BUFFER_DOWN_SCALE)]],               \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

#define DENSE_OUTPUT_OPERANDS                                                         \
    projection::Plain<activation, projection::AllRows> in{product, SEISMIC_PRODUCT_STRIDE_0, \
        SEISMIC_PRODUCT_STRIDE_1, uint(SEISMIC_DIM_F), {}};                             \
    const auto out = projection::scaling<(SEISMIC_DIM_DS != 0)>::wrap(                  \
        projection::Residual<activation, projection::SelectedRows>{result,              \
            SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1, residual, SEISMIC_RESIDUAL_STRIDE_0, \
            SEISMIC_RESIDUAL_STRIDE_1, {out_rows}},                                     \
        projection::scale_factor(down_scale, SEISMIC_DIM_DS, 0, 0), 1.0f);              \
    projection::Weights<packets::W0> w{down_weight, KERNEL_W0_LAYOUT(SEISMIC_DIM_F), uint(SEISMIC_DIM_F)}

#ifdef SEISMIC_FORMING_DENSE_OUTPUT_GEMV
template <uint ROWS, uint LANES>
kernel void dense_output_gemv(DENSE_OUTPUT_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    DENSE_OUTPUT_OPERANDS;
    uint rows = uint(SEISMIC_DIM_O);
    PROJECTION_FOR_ROWS(rows,
        projection::gemv_runtime<packets::W0, ROWS, MAXM, LANES>(
            in, out, w, rows, uint(SEISMIC_DIM_H), uint(SEISMIC_DIM_F), tile, shared, simdgroups, sg, lane));
}
#endif

#ifdef SEISMIC_FORMING_DENSE_OUTPUT_BATCH
template <uint BATCH_ROWS>
kernel void dense_output_batch(DENSE_OUTPUT_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    DENSE_OUTPUT_OPERANDS;
    projection::gemv_batch_runtime<packets::W0, BATCH_ROWS>(in, out, w,
        uint(SEISMIC_DIM_O), uint(SEISMIC_DIM_H), uint(SEISMIC_DIM_F), tile, shared, simdgroups, sg, lane);
}
#endif

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
#ifdef SEISMIC_FORMING_DENSE_OUTPUT_GEMM_SMALL
kernel void dense_output_gemm_small(DENSE_OUTPUT_ARGUMENTS,
    uint3 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    DENSE_OUTPUT_GEMM(projection::small_tile_m, projection::small_tile_n, projection::small_split, small_partials);
}
#endif

#ifdef SEISMIC_FORMING_DENSE_OUTPUT_REDUCE_SMALL
kernel void dense_output_reduce_small(DENSE_OUTPUT_ARGUMENTS,
    uint index [[thread_position_in_grid]]) {
    DENSE_OUTPUT_OPERANDS;
    projection::gemm_reduce(out, small_partials, uint(SEISMIC_DIM_O), uint(SEISMIC_DIM_H), projection::small_split,
        index);
}
#endif

#ifdef SEISMIC_FORMING_DENSE_OUTPUT_GEMM
template <uint TILE_M, uint TILE_N>
kernel void dense_output_gemm(DENSE_OUTPUT_ARGUMENTS,
    uint3 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    DENSE_OUTPUT_GEMM(TILE_M, TILE_N, uint(SEISMIC_RUNTIME_SPLIT), partials);
}
#endif

#ifdef SEISMIC_FORMING_DENSE_OUTPUT_REDUCE
kernel void dense_output_reduce(DENSE_OUTPUT_ARGUMENTS,
    uint index [[thread_position_in_grid]]) {
    DENSE_OUTPUT_OPERANDS;
    projection::gemm_reduce(out, partials, uint(SEISMIC_DIM_O), uint(SEISMIC_DIM_H),
        uint(SEISMIC_RUNTIME_SPLIT), index);
}
#endif

// The TALL form past 64 rows: the product rows in the tall GEMM's order, then
// its tiles.
#ifdef SEISMIC_FORMING_DENSE_OUTPUT_RELAYOUT
kernel void dense_output_relayout(DENSE_OUTPUT_ARGUMENTS,
    uint item [[thread_position_in_grid]]) {
    DENSE_OUTPUT_OPERANDS;
    projection::tall_relayout(in, fragments, uint(SEISMIC_DIM_O), uint(SEISMIC_DIM_F), item);
}
#endif

#ifdef SEISMIC_FORMING_DENSE_OUTPUT_TALL
template <uint TALL_M, uint TALL_K, uint STAGERS>
kernel void dense_output_tall(DENSE_OUTPUT_ARGUMENTS,
    uint3 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    PROJECTION_GEMM_TALL_SHARED(shared, TALL_K);
    DENSE_OUTPUT_OPERANDS;
    projection::gemm_tall<packets::W0, TALL_M, TALL_K, STAGERS>(projection::tall_operand(in, fragments), out, w,
        uint(SEISMIC_DIM_O), uint(SEISMIC_DIM_H), uint(SEISMIC_DIM_F), tile.y, tile.x, shared, sg, lane);
}
#endif
