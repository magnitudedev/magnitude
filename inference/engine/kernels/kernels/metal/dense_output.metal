// dense_output: the down projection of the product rows plus the
// residual rows they were gathered from (`out_rows`).
#define KERNEL_A SEISMIC_ELEMENT_A
#define KERNEL_W0 SEISMIC_DOWN_WEIGHT
#include "lib/projection/projection.h"
#include "lib/projection/packing.h"
#include "lib/projection/operator.h"
#if defined(SEISMIC_FORMING_DENSE_OUTPUT_GEMV) || defined(SEISMIC_FORMING_DENSE_OUTPUT_GEMV_ROWS) || defined(SEISMIC_FORMING_DENSE_OUTPUT_GEMV_PAIR)
#include "lib/projection/gemv.h"
#endif

typedef activation::Act activation_element;
// The down weight's operand rows (the product rows, or NVFP4 operands): the
// GEMV row classes form them while staging, the stage launch into the
// scratch every other form reads.
constant constexpr bool NVFP4 = projection::OperandRecipe<packets::W0_OPERANDS_NVFP4>::NVFP4;
typedef projection::operands<NVFP4> Operands;

#define PROJECTION_ARGUMENTS                                                            \
    device const float *residual [[buffer(SEISMIC_BUFFER_RESIDUAL)]],                   \
    device const uchar *product [[buffer(SEISMIC_BUFFER_PRODUCT)]],                     \
    device uchar *staged [[buffer(SEISMIC_BUFFER_SCRATCH_STAGED)]],                     \
    device const uchar *down_weight [[buffer(SEISMIC_BUFFER_DOWN_WEIGHT)]],             \
    device const int *out_rows [[buffer(SEISMIC_BUFFER_OUT_ROWS)]],                     \
    SEISMIC_RESULT_0_PARAM(result_cells),                                               \
    device float *partials [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIALS)]],                 \
    device float *small_partials [[buffer(SEISMIC_BUFFER_SCRATCH_SMALL_PARTIALS)]],     \
    device uchar *fragments [[buffer(SEISMIC_BUFFER_SCRATCH_FRAGMENTS)]],               \
    device uchar *quantized [[buffer(SEISMIC_BUFFER_SCRATCH_QUANTIZED)]],               \
    device half *row_scales [[buffer(SEISMIC_BUFFER_SCRATCH_ROW_SCALES)]],              \
    device half *block_sums [[buffer(SEISMIC_BUFFER_SCRATCH_BLOCK_SUMS)]],              \
    device float *down_coefficients [[buffer(SEISMIC_BUFFER_SCRATCH_DOWN_COEFFICIENTS)]], \
    device half *down_biases [[buffer(SEISMIC_BUFFER_SCRATCH_DOWN_BIASES)]],            \
    device float *packed [[buffer(SEISMIC_BUFFER_SCRATCH_PACKED)]],                     \
    device half *token_factors [[buffer(SEISMIC_BUFFER_SCRATCH_TOKEN_FACTORS)]],        \
    device half *down_factors [[buffer(SEISMIC_BUFFER_SCRATCH_DOWN_FACTORS)]],          \
    device float *token_scales [[buffer(SEISMIC_BUFFER_SCRATCH_TOKEN_SCALES)]],         \
    device float *down_scales [[buffer(SEISMIC_BUFFER_SCRATCH_DOWN_SCALES)]],           \
    device const float *down_scale [[buffer(SEISMIC_BUFFER_DOWN_SCALE)]],               \
    device const float *down_bound [[buffer(SEISMIC_BUFFER_DOWN_BOUND)]],               \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

// `raw` is the product rows; `in` the rows every form past the GEMV reads.
#define PROJECTION_OPERANDS(group)                                                      \
    const uint m = uint(SEISMIC_DIM_O), n = uint(SEISMIC_DIM_H), k = uint(SEISMIC_DIM_F); \
    const projection::Plain<activation_element, projection::AllRows> raw{product, SEISMIC_PRODUCT_STRIDE_0, \
        SEISMIC_PRODUCT_STRIDE_1, uint(SEISMIC_DIM_F), {}};                             \
    const projection::Plain<activation_element, projection::AllRows> in = NVFP4         \
        ? projection::Plain<activation_element, projection::AllRows>{staged, SEISMIC_DIM_F, 1, uint(SEISMIC_DIM_F), {}} \
        : raw;                                                                          \
    const auto out = projection::scaling<(SEISMIC_DIM_DS != 0)>::wrap(                  \
        projection::Residual<activation_element, projection::SelectedRows>{element::Out<element::F32>(result_cells), \
            SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1, residual, SEISMIC_RESIDUAL_STRIDE_0, \
            SEISMIC_RESIDUAL_STRIDE_1, {out_rows}},                                     \
        projection::scale_factor(down_scale, SEISMIC_DIM_DS, 0, 0), 1.0f);              \
    projection::Weights<packets::W0> w{{down_weight, KERNEL_W0_LAYOUT(SEISMIC_DIM_F), uint(SEISMIC_DIM_F)}}

// The product rows as NVFP4 operands into `staged`, one row a threadgroup.
#ifdef SEISMIC_FORMING_DENSE_OUTPUT_STAGE
kernel void dense_output_stage(PROJECTION_ARGUMENTS,
    uint item [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    PROJECTION_NORMALIZE_SHARED(maxima);
    PROJECTION_OPERANDS(0u);
    projection::operands<true>::form<256>(raw, item, staged, uint(SEISMIC_DIM_F), maxima, thread_index);
}
#endif

#define PROJECTION_GROUPS 1
#define PROJECTION_GEMV_INPUT(x) const auto x = raw
#define PROJECTION_INT8_SCRATCH \
    const projection::int8_scratch scratch{quantized, row_scales, block_sums, down_coefficients, down_biases}
#define PROJECTION_PACKING_SCRATCH \
    const projection::packing_scratch scratch{packed, token_factors, token_scales, down_factors, down_scales}
#define PROJECTION_BOUNDS                                                               \
    constexpr uint BOUNDS = SEISMIC_DIM_DB != 0 ? 1 : 0;                                \
    const projection::packing_bounds<BOUNDS> bounds{{down_bound}}

// The `dense` projection's launches (`lib/projection/operator.h`).
#ifdef SEISMIC_FORMING_DENSE_OUTPUT_GEMV
template <uint LANES, uint TILED, uint STAGE, uint ROWS>
kernel void dense_output_gemv(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_SINGLE_GEMV(ONE, TILED);
}
#endif
#ifdef SEISMIC_FORMING_DENSE_OUTPUT_GEMV_PAIR
template <uint LANES, uint TILED, uint STAGE, uint ROWS>
kernel void dense_output_gemv_pair(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_SINGLE_GEMV(PAIR, TILED);
}
#endif
#ifdef SEISMIC_FORMING_DENSE_OUTPUT_GEMV_ROWS
template <uint LANES, uint TILED, uint STAGE, uint ROWS>
kernel void dense_output_gemv_rows(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_SINGLE_GEMV(SEVERAL, TILED);
}
#endif
#ifdef SEISMIC_FORMING_DENSE_OUTPUT_BATCH
template <uint MATRIX, uint BATCH_ROWS, uint BATCH_PARTS>
kernel void dense_output_batch(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_SINGLE_BATCH;
}
#endif
#ifdef SEISMIC_FORMING_DENSE_OUTPUT_GEMM_SMALL
kernel void dense_output_gemm_small(PROJECTION_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_SINGLE_GEMM_SMALL;
}
#endif
#ifdef SEISMIC_FORMING_DENSE_OUTPUT_REDUCE_SMALL
kernel void dense_output_reduce_small(PROJECTION_ARGUMENTS, PROJECTION_ITEM_ARGUMENTS) {
    PROJECTION_SINGLE_REDUCE_SMALL;
}
#endif
#ifdef SEISMIC_FORMING_DENSE_OUTPUT_GEMM
template <uint TILE_M, uint TILE_N>
kernel void dense_output_gemm(PROJECTION_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_SINGLE_GEMM(uint(SEISMIC_RUNTIME_SPLIT));
}
#endif
#ifdef SEISMIC_FORMING_DENSE_OUTPUT_REDUCE
kernel void dense_output_reduce(PROJECTION_ARGUMENTS, PROJECTION_ITEM_ARGUMENTS) {
    PROJECTION_SINGLE_REDUCE(uint(SEISMIC_RUNTIME_SPLIT));
}
#endif
#ifdef SEISMIC_FORMING_DENSE_OUTPUT_RELAYOUT
kernel void dense_output_relayout(PROJECTION_ARGUMENTS, PROJECTION_ITEM_ARGUMENTS) {
    PROJECTION_SINGLE_RELAYOUT;
}
#endif
#ifdef SEISMIC_FORMING_DENSE_OUTPUT_TALL
template <uint TALL_M, uint TALL_K, uint STAGERS>
kernel void dense_output_tall(PROJECTION_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_SINGLE_TALL;
}
#endif
#ifdef SEISMIC_FORMING_DENSE_OUTPUT_QUANTIZE
kernel void dense_output_quantize(PROJECTION_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    PROJECTION_SINGLE_QUANTIZE;
}
#endif
#ifdef SEISMIC_FORMING_DENSE_OUTPUT_COEFFICIENTS
kernel void dense_output_coefficients(PROJECTION_ARGUMENTS, PROJECTION_ITEM_ARGUMENTS) {
    PROJECTION_SINGLE_COEFFICIENTS;
}
#endif
#ifdef SEISMIC_FORMING_DENSE_OUTPUT_INT8
template <uint INT8_TILE>
kernel void dense_output_int8(PROJECTION_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_SINGLE_INT8;
}
#endif
#ifdef SEISMIC_FORMING_DENSE_OUTPUT_PACK
template <uint RESIDUE>
kernel void dense_output_pack(PROJECTION_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    PROJECTION_SINGLE_PACK;
}
#endif
#ifdef SEISMIC_FORMING_DENSE_OUTPUT_PACK_COEFFICIENTS
kernel void dense_output_pack_coefficients(PROJECTION_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    PROJECTION_SINGLE_PACK_COEFFICIENTS;
}
#endif
#ifdef SEISMIC_FORMING_DENSE_OUTPUT_PACKED
template <uint RESIDUE, uint PACK_TOKENS, uint WEIGHTS_AHEAD>
kernel void dense_output_packed(PROJECTION_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    PROJECTION_SINGLE_PACKED;
}
#endif
#ifdef SEISMIC_FORMING_DENSE_OUTPUT_UNPACKED
kernel void dense_output_unpacked(PROJECTION_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_SINGLE_UNPACKED;
}
#endif
