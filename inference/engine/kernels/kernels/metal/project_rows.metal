// project_rows: the plain projection of the A rows `source`, published in Y
// (the `attention_output` launch structure with a store epilogue).
#define KERNEL_W0 SEISMIC_WEIGHT
#include "lib/projection/projection.h"
#include "lib/projection/packing.h"

typedef element::Act activation;
typedef ELEMENT_OF(SEISMIC_ELEMENT_Y) published;

#define PROJECT_ROWS_ARGUMENTS                                                          \
    device const uchar *source [[buffer(SEISMIC_BUFFER_SOURCE)]],                       \
    device const uchar *weight [[buffer(SEISMIC_BUFFER_WEIGHT)]],                       \
    device uchar *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],                           \
    device float *partials [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIALS)]],                 \
    device float *small_partials [[buffer(SEISMIC_BUFFER_SCRATCH_SMALL_PARTIALS)]],     \
    device uchar *fragments [[buffer(SEISMIC_BUFFER_SCRATCH_FRAGMENTS)]],               \
    device float *packed [[buffer(SEISMIC_BUFFER_SCRATCH_PACKED)]],                     \
    device half *token_factors [[buffer(SEISMIC_BUFFER_SCRATCH_TOKEN_FACTORS)]],        \
    device half *weight_factors [[buffer(SEISMIC_BUFFER_SCRATCH_WEIGHT_FACTORS)]],      \
    device float *token_scales [[buffer(SEISMIC_BUFFER_SCRATCH_TOKEN_SCALES)]],         \
    device float *weight_scales [[buffer(SEISMIC_BUFFER_SCRATCH_WEIGHT_SCALES)]],       \
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

// The TALL form past 64 rows: the source rows in the tall GEMM's order, then
// its tiles.
#ifdef SEISMIC_FORMING_PROJECT_ROWS_RELAYOUT
kernel void project_rows_relayout(PROJECT_ROWS_ARGUMENTS,
    uint item [[thread_position_in_grid]]) {
    PROJECT_ROWS_OPERANDS;
    projection::tall_relayout(in, fragments, uint(SEISMIC_DIM_M), k, item);
}
#endif

#ifdef SEISMIC_FORMING_PROJECT_ROWS_TALL
template <uint TALL_M, uint TALL_K, uint STAGERS>
kernel void project_rows_tall(PROJECT_ROWS_ARGUMENTS,
    uint3 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    PROJECTION_GEMM_TALL_SHARED(shared, TALL_K);
    PROJECT_ROWS_OPERANDS;
    projection::gemm_tall<packets::W0, TALL_M, TALL_K, STAGERS>(projection::tall_operand(in, fragments), out, w,
        uint(SEISMIC_DIM_M), uint(SEISMIC_DIM_N), k, tile.y, tile.x, shared, sg, lane);
}
#endif

// The PACK form past 64 rows: the source rows as integer codes, two rows
// packed per operand element, the weights' block scales and biases, then the
// packed tiles.
#define PROJECT_ROWS_PACKING_SCRATCH \
    const projection::packing_scratch scratch{packed, token_factors, token_scales, weight_factors, weight_scales}

#ifdef SEISMIC_FORMING_PROJECT_ROWS_PACK
kernel void project_rows_pack(PROJECT_ROWS_ARGUMENTS,
    uint pairs [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    if constexpr (!projection::packing_codes<packets::W0>::available)
        return;
    threadgroup float4 peaks[8];
    PROJECT_ROWS_OPERANDS;
    PROJECT_ROWS_PACKING_SCRATCH;
    projection::packing_operand<projection::packing_codes<packets::W0>::centre,
        projection::packing_codes<packets::W0>::folds>(in, scratch, uint(SEISMIC_DIM_M), k, pairs, peaks,
        thread_index, sg, lane);
}
#endif

#ifdef SEISMIC_FORMING_PROJECT_ROWS_PACK_COEFFICIENTS
kernel void project_rows_pack_coefficients(PROJECT_ROWS_ARGUMENTS,
    uint tile [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    if constexpr (!projection::packing_codes<packets::W0>::available)
        return;
    threadgroup float peaks[8];
    PROJECT_ROWS_OPERANDS;
    PROJECT_ROWS_PACKING_SCRATCH;
    projection::packing_tile_coefficients(w, scratch, k, tile, peaks, thread_index, sg, lane);
}
#endif

#ifdef SEISMIC_FORMING_PROJECT_ROWS_PACKED
template <uint PACK_TOKENS, uint WEIGHTS_AHEAD>
kernel void project_rows_packed(PROJECT_ROWS_ARGUMENTS,
    uint tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    if constexpr (!projection::packing_codes<packets::W0>::available)
        return;
    PROJECT_ROWS_OPERANDS;
    PROJECT_ROWS_PACKING_SCRATCH;
    const uint m = uint(SEISMIC_DIM_M);
    projection::gemm_packed<packets::W0, PACK_TOKENS, WEIGHTS_AHEAD>(out, w, scratch, m,
        (m + 127u) / 128u * 128u, uint(SEISMIC_DIM_N), k, tile * projection::packing_simdgroups + sg, lane);
}
#endif

// PACK with weights that have no packed path: the staged tiles of the
// default form, with its results.
#ifdef SEISMIC_FORMING_PROJECT_ROWS_UNPACKED
kernel void project_rows_unpacked(PROJECT_ROWS_ARGUMENTS,
    uint3 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    if constexpr (projection::packing_codes<packets::W0>::available)
        return;
    PROJECT_ROWS_GEMM(64, 64, 1, partials);
}
#endif
