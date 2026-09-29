// project_rows: the plain projection of the A rows `source`, published in Y
// (the `attention_output` launch structure with a store epilogue).
#define KERNEL_W0 SEISMIC_WEIGHT
#include "lib/projection/projection.h"

typedef element::Act activation;
typedef ELEMENT_OF(SEISMIC_ELEMENT_Y) published;

#define PROJECT_ROWS_ARGUMENTS                                                          \
    device const uchar *source [[buffer(SEISMIC_BUFFER_SOURCE)]],                       \
    device const uchar *weight [[buffer(SEISMIC_BUFFER_WEIGHT)]],                       \
    device uchar *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],                           \
    device float *partials [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIALS)]],                 \
    device float *small_partials [[buffer(SEISMIC_BUFFER_SCRATCH_SMALL_PARTIALS)]],     \
    device const float *weight_scale [[buffer(SEISMIC_BUFFER_WEIGHT_SCALE)]],           \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

#define PROJECT_ROWS_OPERANDS                                                           \
    const uint k = uint(SEISMIC_DIM_K);                                                 \
    projection::Plain<activation, projection::AllRows> in{source, SEISMIC_SOURCE_STRIDE_0, \
        SEISMIC_SOURCE_STRIDE_1, k, {}};                                                \
    const auto out = projection::scaling<(SEISMIC_DIM_WS != 0)>::wrap(                  \
        projection::Store<published>{result, SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1, 0}, \
        projection::scale_factor(weight_scale, SEISMIC_DIM_WS, 0, 0), 1.0f);            \
    projection::Weights<packets::W0> w{weight, KERNEL_W0_LAYOUT(k), k}

#ifdef SEISMIC_FORMING_PROJECT_ROWS_GEMV
template <uint ROWS, uint LANES>
kernel void project_rows_gemv(PROJECT_ROWS_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    PROJECT_ROWS_OPERANDS;
    uint rows = uint(SEISMIC_DIM_M);
    PROJECTION_FOR_ROWS(rows,
        projection::gemv_runtime<packets::W0, ROWS, MAXM, LANES>(
            in, out, w, rows, uint(SEISMIC_DIM_N), k, tile, shared, simdgroups, sg, lane));
}
#endif

#ifdef SEISMIC_FORMING_PROJECT_ROWS_BATCH
template <uint BATCH_ROWS>
kernel void project_rows_batch(PROJECT_ROWS_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    PROJECT_ROWS_OPERANDS;
    projection::gemv_batch_runtime<packets::W0, BATCH_ROWS>(in, out, w,
        uint(SEISMIC_DIM_M), uint(SEISMIC_DIM_N), k, tile, shared, simdgroups, sg, lane);
}
#endif

#define PROJECT_ROWS_GEMM(TM, TN, SPLIT, PARTIALS)                                      \
    PROJECTION_GEMM_SHARED(shared, TM, TN);                                             \
    PROJECT_ROWS_OPERANDS;                                                              \
    const uint m = uint(SEISMIC_DIM_M), n = uint(SEISMIC_DIM_N);                        \
    if (SPLIT == 1)                                                                     \
        projection::gemm<packets::W0, TM, TN>(in, out, w, m, n, k, tile.y, tile.x, shared, sg, lane); \
    else                                                                                \
        projection::gemm_part<packets::W0, TM, TN>(in, PARTIALS, w, m, n, k, SPLIT, tile.z, tile.y, tile.x, \
            shared, sg, lane)

// 17..64 rows: the fixed small-row tile and split.
#ifdef SEISMIC_FORMING_PROJECT_ROWS_GEMM_SMALL
kernel void project_rows_gemm_small(PROJECT_ROWS_ARGUMENTS,
    uint3 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    PROJECT_ROWS_GEMM(projection::small_tile_m, projection::small_tile_n, projection::small_split, small_partials);
}
#endif

#ifdef SEISMIC_FORMING_PROJECT_ROWS_FINALIZE_SMALL
kernel void project_rows_finalize_small(PROJECT_ROWS_ARGUMENTS,
    uint index [[thread_position_in_grid]]) {
    PROJECT_ROWS_OPERANDS;
    projection::gemm_reduce(out, small_partials, uint(SEISMIC_DIM_M), uint(SEISMIC_DIM_N), projection::small_split,
        index);
}
#endif

#ifdef SEISMIC_FORMING_PROJECT_ROWS_GEMM
template <uint TILE_M, uint TILE_N>
kernel void project_rows_gemm(PROJECT_ROWS_ARGUMENTS,
    uint3 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    PROJECT_ROWS_GEMM(TILE_M, TILE_N, uint(SEISMIC_RUNTIME_SPLIT), partials);
}
#endif

#ifdef SEISMIC_FORMING_PROJECT_ROWS_FINALIZE
kernel void project_rows_finalize(PROJECT_ROWS_ARGUMENTS,
    uint index [[thread_position_in_grid]]) {
    PROJECT_ROWS_OPERANDS;
    projection::gemm_reduce(out, partials, uint(SEISMIC_DIM_M), uint(SEISMIC_DIM_N),
        uint(SEISMIC_RUNTIME_SPLIT), index);
}
#endif
