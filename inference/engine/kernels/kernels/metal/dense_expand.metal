// dense_expand: RMS prologue over the `out_rows` residual rows, the
// paired gate/up projection, and act(gate)·up (`activation`: SiLU or
// GELU-tanh).
#define KERNEL_W0 SEISMIC_GATE_WEIGHT
#define KERNEL_W1 SEISMIC_UP_WEIGHT
// The expansion's limit as its F32 bits LB (0: none), which the Glu epilogue
// clamps at.
#if SEISMIC_STATIC_LB != 0
#define PROJECTION_GLU_LIMIT SEISMIC_STATIC_LB
#endif
#include "lib/projection/projection.h"
#include "lib/projection/packing.h"
#include "lib/projection/operator.h"
#if defined(SEISMIC_FORMING_DENSE_EXPAND_GEMV) || defined(SEISMIC_FORMING_DENSE_EXPAND_GEMV_ROWS) || defined(SEISMIC_FORMING_DENSE_EXPAND_GEMV_PAIR)
#include "lib/projection/gemv.h"
#endif

typedef element::Act activation;
typedef ELEMENT_OF(SEISMIC_NORM) norm_element;
// The rows both weights take (the normalized rows, or NVFP4 operands): the
// GEMV row classes form them while staging, the normalize launches into
// the scratch every other form reads.
constant constexpr bool NVFP4 =
    projection::OperandRecipe<packets::W0_OPERANDS_NVFP4, packets::W1_OPERANDS_NVFP4>::NVFP4;
typedef projection::operands<NVFP4> Operands;

#define PROJECTION_ARGUMENTS                                                            \
    device const float *residual [[buffer(SEISMIC_BUFFER_RESIDUAL)]],                   \
    device const uchar *norm [[buffer(SEISMIC_BUFFER_NORM)]],                           \
    device const uchar *gate_weight [[buffer(SEISMIC_BUFFER_GATE_WEIGHT)]],             \
    device const uchar *up_weight [[buffer(SEISMIC_BUFFER_UP_WEIGHT)]],                 \
    device const int *out_rows [[buffer(SEISMIC_BUFFER_OUT_ROWS)]],                     \
    SEISMIC_RESULT_0_PARAM(result_cells),                                               \
    device uchar *normalized [[buffer(SEISMIC_BUFFER_SCRATCH_NORMALIZED)]],             \
    device uchar *quantized [[buffer(SEISMIC_BUFFER_SCRATCH_QUANTIZED)]],               \
    device half *row_scales [[buffer(SEISMIC_BUFFER_SCRATCH_ROW_SCALES)]],              \
    device half *block_sums [[buffer(SEISMIC_BUFFER_SCRATCH_BLOCK_SUMS)]],              \
    device float *gate_coefficients [[buffer(SEISMIC_BUFFER_SCRATCH_GATE_COEFFICIENTS)]], \
    device half *gate_biases [[buffer(SEISMIC_BUFFER_SCRATCH_GATE_BIASES)]],            \
    device float *up_coefficients [[buffer(SEISMIC_BUFFER_SCRATCH_UP_COEFFICIENTS)]],   \
    device half *up_biases [[buffer(SEISMIC_BUFFER_SCRATCH_UP_BIASES)]],                \
    device float *packed [[buffer(SEISMIC_BUFFER_SCRATCH_PACKED)]],                     \
    device half *token_factors [[buffer(SEISMIC_BUFFER_SCRATCH_TOKEN_FACTORS)]],        \
    device half *gate_factors [[buffer(SEISMIC_BUFFER_SCRATCH_GATE_FACTORS)]],          \
    device half *up_factors [[buffer(SEISMIC_BUFFER_SCRATCH_UP_FACTORS)]],              \
    device float *token_scales [[buffer(SEISMIC_BUFFER_SCRATCH_TOKEN_SCALES)]],         \
    device float *gate_scales [[buffer(SEISMIC_BUFFER_SCRATCH_GATE_SCALES)]],           \
    device float *up_scales [[buffer(SEISMIC_BUFFER_SCRATCH_UP_SCALES)]],               \
    device const float *gate_scale [[buffer(SEISMIC_BUFFER_GATE_SCALE)]],               \
    device const float *up_scale [[buffer(SEISMIC_BUFFER_UP_SCALE)]],                   \
    device const float *gate_bound [[buffer(SEISMIC_BUFFER_GATE_BOUND)]],               \
    device const float *up_bound [[buffer(SEISMIC_BUFFER_UP_BOUND)]],                   \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

// `raw` is the residual rows under the norm; `in` the normalized rows every
// form past the GEMV reads.
#define PROJECTION_OPERANDS                                                             \
    const uint m = uint(SEISMIC_DIM_O), n = uint(SEISMIC_DIM_F), k = uint(SEISMIC_DIM_H); \
    projection::Rms<activation, norm_element, projection::SelectedRows> raw{residual,   \
        SEISMIC_RESIDUAL_STRIDE_0, SEISMIC_RESIDUAL_STRIDE_1, norm, SEISMIC_NORM_STRIDE_0, \
        as_type<float>(uint(SEISMIC_PARAM_EPS)), k, {out_rows}};                        \
    const projection::Plain<activation, projection::AllRows> in{normalized, k, 1, k, {}}; \
    const auto out = projection::scaling<(SEISMIC_DIM_GS != 0 || SEISMIC_DIM_US != 0)>::wrap( \
        projection::Glu<activation>{element::Out<activation>(result_cells), SEISMIC_RESULT_0_STRIDE_0, \
            SEISMIC_RESULT_0_STRIDE_1, int(SEISMIC_PARAM_ACTIVATION)},                  \
        projection::scale_factor(gate_scale, SEISMIC_DIM_GS, 0, 0),                     \
        projection::scale_factor(up_scale, SEISMIC_DIM_US, 0, 0));                      \
    projection::Weights<packets::W0> w0{gate_weight, KERNEL_W0_LAYOUT(k), k};           \
    projection::Weights<packets::W1> w1{up_weight, KERNEL_W1_LAYOUT(k), k}

// The norm prologue: the GEMV normalizes its rows while staging them, and
// the normalize launches into `normalized`, in the tall GEMM's order where
// the TALL form runs.
#define PROJECTION_GEMV_INPUT(x)                                                        \
    PROJECTION_SQUARES_SHARED(squares, decltype(raw)::parts);                           \
    projection::threadgroup_squares_runtime(raw, m, squares, simdgroups, sg, lane);     \
    const auto x = projection::shared_norm(raw, squares)
#define PROJECTION_FRAGMENTS normalized
#define PROJECTION_INT8_SCRATCH                                                         \
    const projection::int8_scratch scratch0{quantized, row_scales, block_sums, gate_coefficients, gate_biases}; \
    const projection::int8_scratch scratch1{quantized, row_scales, block_sums, up_coefficients, up_biases}
#define PROJECTION_PACKING_SCRATCH                                                      \
    const projection::packing_scratch scratch0{packed, token_factors, token_scales, gate_factors, gate_scales}; \
    const projection::packing_scratch scratch1{packed, token_factors, token_scales, up_factors, up_scales}
#define PROJECTION_BOUNDED ((SEISMIC_DIM_GB != 0 ? 1u : 0u) | (SEISMIC_DIM_UB != 0 ? 2u : 0u))
#define PROJECTION_BOUND_TABLES {gate_bound, up_bound}

#ifdef SEISMIC_FORMING_DENSE_EXPAND_NORMALIZE
kernel void dense_expand_normalize(PROJECTION_ARGUMENTS,
    uint item [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    PROJECTION_NORMALIZE_SHARED(norms);
    PROJECTION_OPERANDS;
    Operands::normalize<256>(raw, item, normalized, k, norms, thread_index);
}
#endif

#ifdef SEISMIC_FORMING_DENSE_EXPAND_NORMALIZE_TALL
kernel void dense_expand_normalize_tall(PROJECTION_ARGUMENTS,
    uint item [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    PROJECTION_NORMALIZE_SHARED(norms);
    PROJECTION_OPERANDS;
    Operands::normalize<256, projection::TallOrder<activation>>(raw, item, normalized, k, norms, thread_index);
}
#endif

// The `dense` projection's launches (`lib/projection/operator.h`).
#ifdef SEISMIC_FORMING_DENSE_EXPAND_GEMV
template <uint LANES, uint TILED, uint ROWS>
kernel void dense_expand_gemv(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_PAIRED_GEMV(ONE, TILED);
}
#endif
#ifdef SEISMIC_FORMING_DENSE_EXPAND_GEMV_PAIR
template <uint LANES, uint TILED, uint ROWS>
kernel void dense_expand_gemv_pair(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_PAIRED_GEMV(PAIR, TILED);
}
#endif
#ifdef SEISMIC_FORMING_DENSE_EXPAND_GEMV_ROWS
template <uint LANES, uint TILED, uint ROWS>
kernel void dense_expand_gemv_rows(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_PAIRED_GEMV(SEVERAL, TILED);
}
#endif
#ifdef SEISMIC_FORMING_DENSE_EXPAND_BATCH
template <uint MATRIX, uint BATCH_ROWS, uint BATCH_PARTS>
kernel void dense_expand_batch(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_PAIRED_BATCH;
}
#endif
#ifdef SEISMIC_FORMING_DENSE_EXPAND_GEMM_SMALL
kernel void dense_expand_gemm_small(PROJECTION_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_PAIRED_GEMM_SMALL;
}
#endif
#ifdef SEISMIC_FORMING_DENSE_EXPAND_GEMM
template <uint TILE_M, uint TILE_N>
kernel void dense_expand_gemm(PROJECTION_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_PAIRED_GEMM;
}
#endif
#ifdef SEISMIC_FORMING_DENSE_EXPAND_TALL
template <uint TALL_M, uint TALL_K, uint STAGERS>
kernel void dense_expand_tall(PROJECTION_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_PAIRED_TALL;
}
#endif
#ifdef SEISMIC_FORMING_DENSE_EXPAND_QUANTIZE
kernel void dense_expand_quantize(PROJECTION_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    PROJECTION_PAIRED_QUANTIZE;
}
#endif
#ifdef SEISMIC_FORMING_DENSE_EXPAND_COEFFICIENTS
kernel void dense_expand_coefficients(PROJECTION_ARGUMENTS, PROJECTION_ITEM_ARGUMENTS) {
    PROJECTION_PAIRED_COEFFICIENTS;
}
#endif
#ifdef SEISMIC_FORMING_DENSE_EXPAND_INT8
kernel void dense_expand_int8(PROJECTION_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_PAIRED_INT8;
}
#endif
#ifdef SEISMIC_FORMING_DENSE_EXPAND_PACK
template <uint RESIDUE>
kernel void dense_expand_pack(PROJECTION_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    PROJECTION_PAIRED_PACK;
}
#endif
#ifdef SEISMIC_FORMING_DENSE_EXPAND_PACK_COEFFICIENTS
kernel void dense_expand_pack_coefficients(PROJECTION_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    PROJECTION_PAIRED_PACK_COEFFICIENTS;
}
#endif
#ifdef SEISMIC_FORMING_DENSE_EXPAND_PACKED
template <uint RESIDUE, uint PACK_TOKENS, uint WEIGHTS_AHEAD>
kernel void dense_expand_packed(PROJECTION_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    PROJECTION_PAIRED_PACKED;
}
#endif
#ifdef SEISMIC_FORMING_DENSE_EXPAND_UNPACKED
kernel void dense_expand_unpacked(PROJECTION_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_PAIRED_UNPACKED;
}
#endif
