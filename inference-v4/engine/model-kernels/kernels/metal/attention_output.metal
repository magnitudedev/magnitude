// attention_output: the output projection of the gated attention rows
// plus the F32 residual.
#define KERNEL_W0 SEISMIC_OUTPUT_WEIGHT
#include "lib/projection/projection.h"

typedef element::Act activation;

#define ATTENTION_OUTPUT_ARGUMENTS                                                      \
    device const float *hidden [[buffer(SEISMIC_BUFFER_HIDDEN)]],                       \
    device const uchar *gated [[buffer(SEISMIC_BUFFER_GATED)]],                         \
    device const uchar *output_weight [[buffer(SEISMIC_BUFFER_OUTPUT_WEIGHT)]],         \
    device float *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],                           \
    device float *partials [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIALS)]],                 \
    device float *small_partials [[buffer(SEISMIC_BUFFER_SCRATCH_SMALL_PARTIALS)]],     \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

// `gated` is [M, Q, W] and canonical, so a row is Q*W contiguous values.
#define ATTENTION_OUTPUT_OPERANDS                                                       \
    const uint k = uint(SEISMIC_DIM_Q * SEISMIC_DIM_W);                                 \
    projection::Plain<activation, projection::AllRows> in{gated, SEISMIC_GATED_STRIDE_0, \
        SEISMIC_GATED_STRIDE_2, k, {}};                                                 \
    projection::Residual<activation, projection::AllRows> out{result, SEISMIC_RESULT_0_STRIDE_0, \
        SEISMIC_RESULT_0_STRIDE_1, hidden, SEISMIC_HIDDEN_STRIDE_0,                     \
        SEISMIC_HIDDEN_STRIDE_1, {}};                                                   \
    projection::Weights<packets::W0> w{output_weight, KERNEL_W0_LAYOUT(k), k}

#ifdef SEISMIC_FORMING_ATTENTION_OUTPUT_GEMV
template <uint ROWS, uint LANES>
kernel void attention_output_gemv(ATTENTION_OUTPUT_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    ATTENTION_OUTPUT_OPERANDS;
    uint rows = uint(SEISMIC_DIM_M);
    PROJECTION_FOR_ROWS(rows,
        projection::gemv_runtime<packets::W0, ROWS, MAXM, LANES>(
            in, out, w, rows, uint(SEISMIC_DIM_D), k, tile, shared, simdgroups, sg, lane));
}
#endif

#ifdef SEISMIC_FORMING_ATTENTION_OUTPUT_BATCH
template <uint BATCH_ROWS>
kernel void attention_output_batch(ATTENTION_OUTPUT_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    ATTENTION_OUTPUT_OPERANDS;
    projection::gemv_batch_runtime<packets::W0, BATCH_ROWS>(in, out, w,
        uint(SEISMIC_DIM_M), uint(SEISMIC_DIM_D), k, tile, shared, simdgroups, sg, lane);
}
#endif

#define ATTENTION_OUTPUT_GEMM(TM, TN, SPLIT, PARTIALS)                                  \
    PROJECTION_GEMM_SHARED(shared, TM, TN);                                             \
    ATTENTION_OUTPUT_OPERANDS;                                                          \
    const uint m = uint(SEISMIC_DIM_M), n = uint(SEISMIC_DIM_D);                        \
    if (SPLIT == 1)                                                                     \
        projection::gemm<packets::W0, TM, TN>(in, out, w, m, n, k, tile.y, tile.x, shared, sg, lane); \
    else                                                                                \
        projection::gemm_part<packets::W0, TM, TN>(in, PARTIALS, w, m, n, k, SPLIT, tile.z, tile.y, tile.x, \
            shared, sg, lane)

// 17..64 rows: the fixed small-row tile and split.
#ifdef SEISMIC_FORMING_ATTENTION_OUTPUT_GEMM_SMALL
kernel void attention_output_gemm_small(ATTENTION_OUTPUT_ARGUMENTS,
    uint3 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    ATTENTION_OUTPUT_GEMM(projection::small_tile_m, projection::small_tile_n, projection::small_split,
        small_partials);
}
#endif

#ifdef SEISMIC_FORMING_ATTENTION_OUTPUT_FINALIZE_SMALL
kernel void attention_output_finalize_small(ATTENTION_OUTPUT_ARGUMENTS,
    uint index [[thread_position_in_grid]]) {
    ATTENTION_OUTPUT_OPERANDS;
    projection::gemm_reduce(out, small_partials, uint(SEISMIC_DIM_M), uint(SEISMIC_DIM_D), projection::small_split,
        index);
}
#endif

#ifdef SEISMIC_FORMING_ATTENTION_OUTPUT_GEMM
template <uint TILE_M, uint TILE_N>
kernel void attention_output_gemm(ATTENTION_OUTPUT_ARGUMENTS,
    uint3 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    ATTENTION_OUTPUT_GEMM(TILE_M, TILE_N, uint(SEISMIC_RUNTIME_SPLIT), partials);
}
#endif

#ifdef SEISMIC_FORMING_ATTENTION_OUTPUT_FINALIZE
kernel void attention_output_finalize(ATTENTION_OUTPUT_ARGUMENTS,
    uint index [[thread_position_in_grid]]) {
    ATTENTION_OUTPUT_OPERANDS;
    projection::gemm_reduce(out, partials, uint(SEISMIC_DIM_M), uint(SEISMIC_DIM_D),
        uint(SEISMIC_RUNTIME_SPLIT), index);
}
#endif
