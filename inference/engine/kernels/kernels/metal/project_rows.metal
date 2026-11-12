// project_rows: the plain projection of the A rows `source`, published in Y
// (the `attention_output` launch structure with a store epilogue). Group g
// of a grouped weight (grid z, `dense.seismic`) is the projection of
// the source columns g * K .., through weight rows g * N .., into output
// columns g * N ..: the same launches over its views (`PROJECT_ROWS_OPERANDS`).
#define KERNEL_W0 SEISMIC_WEIGHT
#include "lib/projection/projection.h"
#include "lib/projection/packing.h"
#ifdef SEISMIC_FORMING_PROJECT_ROWS_GEMV
#include "lib/projection/gemv.h"
#endif

typedef element::Act activation;
typedef ELEMENT_OF(SEISMIC_ELEMENT_Y) published;
// The weight's operand rows (the source rows, or NVFP4 operands): the GEMV
// row classes form them while staging, the stage launch into the scratch
// every other form reads.
constant constexpr bool NVFP4 = projection::OperandRecipe<packets::W0_OPERANDS_NVFP4>::NVFP4;
typedef projection::operands<NVFP4> Operands;

#define PROJECT_ROWS_ARGUMENTS                                                          \
    device const uchar *source [[buffer(SEISMIC_BUFFER_SOURCE)]],                       \
    device uchar *staged [[buffer(SEISMIC_BUFFER_SCRATCH_STAGED)]],                     \
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
    device const float *weight_bound [[buffer(SEISMIC_BUFFER_WEIGHT_BOUND)]],           \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

// The group a launch's grid z names: literally 0 at G = 1 (G is static), so
// every ungrouped address, split part and tile test compiles to its
// ungrouped form.
#define PROJECT_ROWS_GROUP(z) (SEISMIC_DIM_G == 1 ? 0u : uint(z))

// Group `group`'s operands: `raw` is its source columns; `in` the rows every
// form past the GEMV reads (staged NVFP4 operands only at G = 1); `out` its
// output columns and `w` its weight rows.
#define PROJECT_ROWS_OPERANDS(group)                                                    \
    const uint k = uint(SEISMIC_DIM_K);                                                 \
    const projection::Plain<activation, projection::AllRows> raw{source                 \
        + ulong(group) * k * SEISMIC_SOURCE_STRIDE_1 * sizeof(typename activation::storage), \
        SEISMIC_SOURCE_STRIDE_0, SEISMIC_SOURCE_STRIDE_1, k, {}};                       \
    const projection::Plain<activation, projection::AllRows> in = NVFP4                 \
        ? projection::Plain<activation, projection::AllRows>{staged, k, 1, k, {}}       \
        : raw;                                                                          \
    const auto out = projection::scaling<(SEISMIC_DIM_WS != 0)>::wrap(                  \
        projection::Store<published>{result, SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1, \
            ulong(group) * uint(SEISMIC_DIM_N)},                                        \
        projection::scale_factor(weight_scale, SEISMIC_DIM_WS, 0, 0), 1.0f);            \
    projection::Weights<packets::W0> w{weight, KERNEL_W0_LAYOUT(k), k, nullptr,          \
        ulong(group) * uint(SEISMIC_DIM_N)}

// Group `group`'s share of a split's partials: `split` parts of m x n each.
inline device float *group_partials(device float *partials, uint group, uint split, uint m, uint n) {
    return partials + ulong(group) * split * m * n;
}

#ifdef SEISMIC_FORMING_PROJECT_ROWS_GEMV
template <uint ROWS, uint LANES>
kernel void project_rows_gemv(PROJECT_ROWS_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint3 position [[threadgroup_position_in_grid]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    PROJECT_ROWS_OPERANDS(PROJECT_ROWS_GROUP(position.z));
    const uint tile = position.x;
    uint rows = uint(SEISMIC_DIM_M);
    PROJECTION_OPERANDS_SHARED(globals, NVFP4);
    const auto x = Operands::rows(raw, rows, k, globals, simdgroups, sg, lane);
    PROJECTION_FOR_ROWS(rows,
        projection::gemv_runtime<packets::W0, ROWS, MAXM, LANES>(
            x, out, w, rows, uint(SEISMIC_DIM_N), k, tile, shared, simdgroups, sg, lane));
}
#endif

// The source rows as NVFP4 operands into `staged`, one row a threadgroup.
#ifdef SEISMIC_FORMING_PROJECT_ROWS_STAGE
kernel void project_rows_stage(PROJECT_ROWS_ARGUMENTS,
    uint item [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    PROJECTION_NORMALIZE_SHARED(maxima);
    PROJECT_ROWS_OPERANDS(0u);
    projection::operands<true>::form<256>(raw, item, staged, k, maxima, thread_index);
}
#endif

#ifdef SEISMIC_FORMING_PROJECT_ROWS_BATCH
template <uint BATCH_ROWS, uint BATCH_PARTS, uint MATRIX>
kernel void project_rows_batch(PROJECT_ROWS_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint3 position [[threadgroup_position_in_grid]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    PROJECT_ROWS_OPERANDS(PROJECT_ROWS_GROUP(position.z));
    const uint tile = position.x;
    if constexpr (MATRIX == 1) {
        projection::gemv_matrix<packets::W0, BATCH_PARTS>(in, out, w,
            uint(SEISMIC_DIM_M), uint(SEISMIC_DIM_N), k, tile, shared, simdgroups, sg, lane);
        return;
    }
    if (tile * simdgroups * BATCH_ROWS * 8u >= uint(SEISMIC_DIM_N)) return;
    projection::gemv_batch_runtime<packets::W0, BATCH_ROWS>(in, out, w,
        uint(SEISMIC_DIM_M), uint(SEISMIC_DIM_N), k, tile, shared, simdgroups, sg, lane);
}
#endif

// Grid z is group * SPLIT + part (the part alone at G = 1).
#define PROJECT_ROWS_GEMM(TM, TN, SPLIT, PARTIALS)                                      \
    PROJECTION_GEMM_SHARED(shared, TM, TN);                                             \
    const uint group = PROJECT_ROWS_GROUP(tile.z / (SPLIT));                            \
    PROJECT_ROWS_OPERANDS(group);                                                       \
    const uint m = uint(SEISMIC_DIM_M), n = uint(SEISMIC_DIM_N);                        \
    if (SPLIT == 1)                                                                     \
        projection::gemm<packets::W0, TM, TN>(in, out, w, m, n, k, tile.y, tile.x, shared, sg, lane); \
    else                                                                                \
        projection::gemm_part<packets::W0, TM, TN>(in, group_partials(PARTIALS, group, SPLIT, m, n), \
            w, m, n, k, SPLIT, SEISMIC_DIM_G == 1 ? tile.z : tile.z % (SPLIT), tile.y, tile.x, shared, sg, lane)

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
    uint3 position [[thread_position_in_grid]]) {
    const uint group = PROJECT_ROWS_GROUP(position.z);
    PROJECT_ROWS_OPERANDS(group);
    const uint m = uint(SEISMIC_DIM_M), n = uint(SEISMIC_DIM_N);
    projection::gemm_reduce(out, group_partials(small_partials, group, projection::small_split, m, n), m, n,
        projection::small_split, position.x);
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
    uint3 position [[thread_position_in_grid]]) {
    const uint group = PROJECT_ROWS_GROUP(position.z);
    PROJECT_ROWS_OPERANDS(group);
    const uint m = uint(SEISMIC_DIM_M), n = uint(SEISMIC_DIM_N), split = uint(SEISMIC_RUNTIME_SPLIT);
    projection::gemm_reduce(out, group_partials(partials, group, split, m, n), m, n, split, position.x);
}
#endif

// The TALL form past 64 rows: the source rows in the tall GEMM's order, then
// its tiles. Group g's source rows lie in the g-th of the fragments, one
// group's ceil(M / 64) * 64 rows of K halves each.
inline device uchar *group_fragments(device uchar *fragments, uint group, uint m, uint k) {
    return fragments + ulong(group) * ((m + 63u) / 64u) * 64u * k * 2u;
}

#ifdef SEISMIC_FORMING_PROJECT_ROWS_RELAYOUT
kernel void project_rows_relayout(PROJECT_ROWS_ARGUMENTS,
    uint3 position [[thread_position_in_grid]]) {
    const uint group = PROJECT_ROWS_GROUP(position.z);
    PROJECT_ROWS_OPERANDS(group);
    const uint m = uint(SEISMIC_DIM_M);
    projection::tall_relayout(in, group_fragments(fragments, group, m, k), m, k, position.x);
}
#endif

#ifdef SEISMIC_FORMING_PROJECT_ROWS_TALL
template <uint TALL_M, uint TALL_K, uint STAGERS>
kernel void project_rows_tall(PROJECT_ROWS_ARGUMENTS,
    uint3 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    PROJECTION_GEMM_TALL_SHARED(shared, TALL_K);
    const uint group = PROJECT_ROWS_GROUP(tile.z);
    PROJECT_ROWS_OPERANDS(group);
    const uint m = uint(SEISMIC_DIM_M);
    projection::gemm_tall<packets::W0, TALL_M, TALL_K, STAGERS>(
        projection::tall_operand(in, group_fragments(fragments, group, m, k)), out, w, m, uint(SEISMIC_DIM_N), k,
        tile.y, tile.x, shared, sg, lane);
}
#endif

// The PACK form past 64 rows: the source rows as integer codes, two rows
// packed per operand element, the weights' block scales and biases, then the
// packed tiles.
#define PROJECT_ROWS_PACKING_SCRATCH \
    const projection::packing_scratch scratch{packed, token_factors, token_scales, weight_factors, weight_scales}

#ifdef SEISMIC_FORMING_PROJECT_ROWS_PACK
// RESIDUE: the operand's residue width for weights with residue bits.
template <uint RESIDUE>
kernel void project_rows_pack(PROJECT_ROWS_ARGUMENTS,
    uint pairs [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    if constexpr (!projection::packing_path<packets::W0>())
        return;
    threadgroup float4 peaks[8];
    PROJECT_ROWS_OPERANDS(0u);
    PROJECT_ROWS_PACKING_SCRATCH;
    typedef projection::packing_codes<packets::W0> coded;
    constexpr uint BOUNDS = SEISMIC_DIM_WB != 0 ? 1 : 0;
    const projection::packing_bounds<BOUNDS> bounds{{weight_bound}};
    projection::packing_operand<coded::centre, coded::folds, (coded::residue != 0 ? RESIDUE : 0), BOUNDS>(in,
        scratch, bounds, uint(SEISMIC_DIM_M), k, pairs, peaks, thread_index, sg, lane);
}
#endif

#ifdef SEISMIC_FORMING_PROJECT_ROWS_PACK_COEFFICIENTS
kernel void project_rows_pack_coefficients(PROJECT_ROWS_ARGUMENTS,
    uint tile [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    if constexpr (!projection::packing_path<packets::W0>())
        return;
    threadgroup float peaks[8];
    PROJECT_ROWS_OPERANDS(0u);
    PROJECT_ROWS_PACKING_SCRATCH;
    projection::packing_tile_coefficients(w, scratch, k, tile, peaks, thread_index, sg, lane);
}
#endif

#ifdef SEISMIC_FORMING_PROJECT_ROWS_PACKED
template <uint RESIDUE, uint PACK_TOKENS, uint WEIGHTS_AHEAD>
kernel void project_rows_packed(PROJECT_ROWS_ARGUMENTS,
    uint tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    if constexpr (!projection::packing_path<packets::W0>())
        return;
    PROJECT_ROWS_OPERANDS(0u);
    PROJECT_ROWS_PACKING_SCRATCH;
    const uint m = uint(SEISMIC_DIM_M);
    projection::gemm_packed<packets::W0, RESIDUE, PACK_TOKENS, WEIGHTS_AHEAD>(out, w, scratch, m,
        (m + 127u) / 128u * 128u, uint(SEISMIC_DIM_N), k, tile * projection::packing_simdgroups + sg, lane);
}
#endif

// PACK on a device without the packed path: the staged tiles of the default
// form, with its results.
#ifdef SEISMIC_FORMING_PROJECT_ROWS_UNPACKED
kernel void project_rows_unpacked(PROJECT_ROWS_ARGUMENTS,
    uint3 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    if constexpr (projection::packing_path<packets::W0>())
        return;
    PROJECT_ROWS_GEMM(64, 64, 1, partials);
}
#endif
