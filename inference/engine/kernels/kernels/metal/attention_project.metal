// attention_project: RMS prologue over the F32 residual, then the `dense`
// projection (segmented) query | gate | key | value into four results (a
// segment of zero rows has no threadgroups). The GEMV row classes, and the
// batched rows, normalize while staging; `stage` normalizes once into
// `normalized` for the matrix GEMV and every form past 16 rows, and
// `stage_tall` in the tall GEMM's order. Each segment's result is scaled by
// its present scale port; in mode 1 the query and gate segments are zeros.
#define KERNEL_A SEISMIC_ELEMENT_A
#define KERNEL_W0 SEISMIC_QUERY_WEIGHT
#define KERNEL_W1 SEISMIC_GATE_WEIGHT
#define KERNEL_W2 SEISMIC_KEY_WEIGHT
#define KERNEL_W3 SEISMIC_VALUE_WEIGHT
#include "lib/projection/projection.h"
#include "lib/projection/packing.h"
#include "lib/projection/operator.h"
#if defined(SEISMIC_FORMING_ATTENTION_PROJECT_GEMV) || defined(SEISMIC_FORMING_ATTENTION_PROJECT_GEMV_ROWS) || defined(SEISMIC_FORMING_ATTENTION_PROJECT_GEMV_PAIR)
#include "lib/projection/gemv.h"
#endif

typedef activation::Act activation_element;
// The results' element Y (A, or F32 for a pooled source's projections).
typedef ELEMENT_OF(SEISMIC_ELEMENT_Y) published;
typedef ELEMENT_OF(SEISMIC_INPUT_NORM) norm_element;
// The rows every weight takes (the normalized rows, or NVFP4 operands).
constant constexpr bool NVFP4 = projection::OperandRecipe<packets::W0_OPERANDS_NVFP4, packets::W1_OPERANDS_NVFP4,
    packets::W2_OPERANDS_NVFP4, packets::W3_OPERANDS_NVFP4>::NVFP4;
typedef projection::operands<NVFP4> Operands;

#define PROJECTION_ARGUMENTS                                                            \
    device const float *hidden [[buffer(SEISMIC_BUFFER_HIDDEN)]],                       \
    device const uchar *input_norm [[buffer(SEISMIC_BUFFER_INPUT_NORM)]],               \
    device const uchar *query_weight [[buffer(SEISMIC_BUFFER_QUERY_WEIGHT)]],           \
    device const uchar *gate_weight [[buffer(SEISMIC_BUFFER_GATE_WEIGHT)]],             \
    device const uchar *key_weight [[buffer(SEISMIC_BUFFER_KEY_WEIGHT)]],               \
    device const uchar *value_weight [[buffer(SEISMIC_BUFFER_VALUE_WEIGHT)]],           \
    SEISMIC_RESULT_0_PARAM(query_cells),                                                \
    SEISMIC_RESULT_1_PARAM(gate_cells),                                                 \
    SEISMIC_RESULT_2_PARAM(key_cells),                                                  \
    SEISMIC_RESULT_3_PARAM(value_cells),                                                \
    device uchar *normalized [[buffer(SEISMIC_BUFFER_SCRATCH_NORMALIZED)]],             \
    device float *packed [[buffer(SEISMIC_BUFFER_SCRATCH_PACKED)]],                     \
    device half *token_factors [[buffer(SEISMIC_BUFFER_SCRATCH_TOKEN_FACTORS)]],        \
    device half *factors [[buffer(SEISMIC_BUFFER_SCRATCH_FACTORS)]],                    \
    device float *token_scales [[buffer(SEISMIC_BUFFER_SCRATCH_TOKEN_SCALES)]],         \
    device float *scales [[buffer(SEISMIC_BUFFER_SCRATCH_SCALES)]],                     \
    device const float *query_scale [[buffer(SEISMIC_BUFFER_QUERY_SCALE)]],             \
    device const float *gate_scale [[buffer(SEISMIC_BUFFER_GATE_SCALE)]],               \
    device const float *key_scale [[buffer(SEISMIC_BUFFER_KEY_SCALE)]],                 \
    device const float *value_scale [[buffer(SEISMIC_BUFFER_VALUE_SCALE)]],             \
    device const float *query_bound [[buffer(SEISMIC_BUFFER_QUERY_BOUND)]],             \
    device const float *gate_bound [[buffer(SEISMIC_BUFFER_GATE_BOUND)]],               \
    device const float *key_bound [[buffer(SEISMIC_BUFFER_KEY_BOUND)]],                 \
    device const float *value_bound [[buffer(SEISMIC_BUFFER_VALUE_BOUND)]],             \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

// A segment's result after its weight's second-level scale, where the
// segment's scale port (static extent EXTENT) is present.
#define ATTENTION_PROJECT_SCALED(EXTENT, SCALE, RESULT, STRIDE_0, STRIDE_1)             \
    projection::scaling<(EXTENT != 0)>::wrap(                                           \
        projection::Store<published>{RESULT, STRIDE_0, STRIDE_1, 0},                    \
        projection::scale_factor(SCALE, EXTENT, 0, 0), 1.0f)

#define PROJECTION_N0 SEISMIC_DIM_Q
#define PROJECTION_N1 SEISMIC_DIM_GR
#define PROJECTION_N2 SEISMIC_DIM_K
#define PROJECTION_N3 SEISMIC_DIM_V

// `raw` is the residual rows under the norm prologue; `in` the normalized
// rows every form past the GEMV reads.
#define PROJECTION_OPERANDS                                                             \
    const uint m = uint(SEISMIC_DIM_M), k = uint(SEISMIC_DIM_D);                        \
    projection::InputOf<activation_element, norm_element, projection::AllRows, SEISMIC_DIM_NM, SEISMIC_DIM_PT>::type \
        raw{hidden, SEISMIC_HIDDEN_STRIDE_0, SEISMIC_HIDDEN_STRIDE_1, input_norm,        \
        SEISMIC_INPUT_NORM_STRIDE_0, as_type<float>(uint(SEISMIC_PARAM_EPSILON)), k, {}}; \
    const projection::Plain<activation_element, projection::AllRows> in{normalized, k, 1, k, {}}; \
    projection::Weights<packets::W0> w0{{query_weight, KERNEL_W0_LAYOUT(k), k}};        \
    projection::Weights<packets::W1> w1{{gate_weight, KERNEL_W1_LAYOUT(k), k}};         \
    projection::Weights<packets::W2> w2{{key_weight, KERNEL_W2_LAYOUT(k), k}};          \
    projection::Weights<packets::W3> w3{{value_weight, KERNEL_W3_LAYOUT(k), k}};        \
    const auto out0 = ATTENTION_PROJECT_SCALED(SEISMIC_DIM_QS, query_scale, element::Out<published>(query_cells),      \
        SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1);                          \
    const auto out1 = ATTENTION_PROJECT_SCALED(SEISMIC_DIM_GS, gate_scale, element::Out<published>(gate_cells),        \
        SEISMIC_RESULT_1_STRIDE_0, SEISMIC_RESULT_1_STRIDE_1);                          \
    const auto out2 = ATTENTION_PROJECT_SCALED(SEISMIC_DIM_KS, key_scale, element::Out<published>(key_cells),          \
        SEISMIC_RESULT_2_STRIDE_0, SEISMIC_RESULT_2_STRIDE_1);                          \
    const auto out3 = ATTENTION_PROJECT_SCALED(SEISMIC_DIM_VS, value_scale, element::Out<published>(value_cells),      \
        SEISMIC_RESULT_3_STRIDE_0, SEISMIC_RESULT_3_STRIDE_1)

// The GEMV's rows: the norm applied as they stage, from their sums of
// squares over the threadgroup.
#define PROJECTION_GEMV_INPUT(x)                                                        \
    PROJECTION_SQUARES_SHARED(squares, decltype(raw)::parts);                           \
    projection::threadgroup_squares_runtime(raw, m, squares, simdgroups, sg, lane);     \
    const auto x = projection::shared_norm(raw, squares)
#define PROJECTION_FRAGMENTS normalized
// Mode 1 has no query or gate projection.
#define PROJECTION_ZEROED(s) (SEISMIC_PARAM_PROJECT_MODE != 0 && s < 2)
#define PROJECTION_BOUNDED                                                              \
    ((SEISMIC_DIM_QB != 0 ? 1u : 0u) | (SEISMIC_DIM_GB != 0 ? 2u : 0u) | (SEISMIC_DIM_KB != 0 ? 4u : 0u) \
        | (SEISMIC_DIM_VB != 0 ? 8u : 0u))
#define PROJECTION_BOUND_TABLES {query_bound, gate_bound, key_bound, value_bound}
#define PROJECTION_COLUMNS(s) false
#define PROJECTION_GEMV_PROLOGUE(s)
#define PROJECTION_BATCH_PROLOGUE(s)
#define PROJECTION_GEMV_EPILOGUE(s, first, count)

// The normalized rows into `normalized`, one row a threadgroup: in row
// order, and in the tall GEMM's.
#ifdef SEISMIC_FORMING_ATTENTION_PROJECT_STAGE
template <uint MATRIX_FROM>
kernel void attention_project_stage(PROJECTION_ARGUMENTS,
    uint item [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    PROJECTION_NORMALIZE_SHARED(norms);
    PROJECTION_OPERANDS;
    Operands::normalize<256>(raw, item, normalized, k, norms, thread_index);
}
#endif
#ifdef SEISMIC_FORMING_ATTENTION_PROJECT_STAGE_TALL
kernel void attention_project_stage_tall(PROJECTION_ARGUMENTS,
    uint item [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    PROJECTION_NORMALIZE_SHARED(norms);
    PROJECTION_OPERANDS;
    Operands::normalize<256, projection::TallOrder<activation_element>>(raw, item, normalized, k, norms, thread_index);
}
#endif

// The `dense` projection's launches (`lib/projection/operator.h`).
#ifdef SEISMIC_FORMING_ATTENTION_PROJECT_GEMV
template <uint MATRIX_FROM, uint LANES, uint TILED, uint STAGE, uint ROWS>
kernel void attention_project_gemv(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_SEGMENTED_GEMV(ONE, TILED);
}
#endif
#ifdef SEISMIC_FORMING_ATTENTION_PROJECT_BATCH
template <uint MATRIX_FROM, uint BATCH_ROWS, uint BATCH_PARTS>
kernel void attention_project_batch(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_SEGMENTED_BATCH;
}
#endif
#ifdef SEISMIC_FORMING_ATTENTION_PROJECT_GEMM_SMALL
kernel void attention_project_gemm_small(PROJECTION_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_SEGMENTED_GEMM_SMALL;
}
#endif
#ifdef SEISMIC_FORMING_ATTENTION_PROJECT_GEMM
template <uint TILE_M, uint TILE_N>
kernel void attention_project_gemm(PROJECTION_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_SEGMENTED_GEMM;
}
#endif
#ifdef SEISMIC_FORMING_ATTENTION_PROJECT_TALL
template <uint TALL_M, uint TALL_K, uint STAGERS>
kernel void attention_project_tall(PROJECTION_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_SEGMENTED_TALL;
}
#endif
#ifdef SEISMIC_FORMING_ATTENTION_PROJECT_GEMV_ROWS
template <uint MATRIX_FROM, uint LANES, uint TILED, uint STAGE, uint ROWS>
kernel void attention_project_gemv_rows(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_SEGMENTED_GEMV(SEVERAL, TILED);
}
#endif
#ifdef SEISMIC_FORMING_ATTENTION_PROJECT_GEMV_PAIR
template <uint MATRIX_FROM, uint LANES, uint TILED, uint STAGE, uint ROWS>
kernel void attention_project_gemv_pair(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_SEGMENTED_GEMV(PAIR, TILED);
}
#endif
#ifdef SEISMIC_FORMING_ATTENTION_PROJECT_PACK
template <uint RESIDUE>
kernel void attention_project_pack(PROJECTION_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    PROJECTION_SEGMENTED_PACK;
}
#endif
#ifdef SEISMIC_FORMING_ATTENTION_PROJECT_PACK_COEFFICIENTS
kernel void attention_project_pack_coefficients(PROJECTION_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    PROJECTION_SEGMENTED_PACK_COEFFICIENTS;
}
#endif
#ifdef SEISMIC_FORMING_ATTENTION_PROJECT_PACKED
template <uint RESIDUE, uint PACK_TOKENS, uint WEIGHTS_AHEAD>
kernel void attention_project_packed(PROJECTION_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    PROJECTION_SEGMENTED_PACKED;
}
#endif
#ifdef SEISMIC_FORMING_ATTENTION_PROJECT_UNPACKED
kernel void attention_project_unpacked(PROJECTION_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_SEGMENTED_UNPACKED;
}
#endif
