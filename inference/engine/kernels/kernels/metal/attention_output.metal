// attention_output: the output projection of the gated attention rows
// plus the F32 residual.
#define KERNEL_W0 SEISMIC_OUTPUT_WEIGHT
#include "lib/projection/projection.h"
#include "lib/projection/packing.h"
#include "lib/projection/operator.h"
#if defined(SEISMIC_FORMING_ATTENTION_OUTPUT_GEMV) || defined(SEISMIC_FORMING_ATTENTION_OUTPUT_GEMV_ROWS) || defined(SEISMIC_FORMING_ATTENTION_OUTPUT_GEMV_PAIR)
#include "lib/projection/gemv.h"
#endif

typedef element::Act activation;
// The output weight's operand rows (the gated rows, or NVFP4 operands): the
// GEMV row classes form them while staging, the stage launch into the
// scratch every other form reads.
constant constexpr bool NVFP4 = projection::OperandRecipe<packets::W0_OPERANDS_NVFP4>::NVFP4;
typedef projection::operands<NVFP4> Operands;

#define PROJECTION_ARGUMENTS                                                            \
    device const float *hidden [[buffer(SEISMIC_BUFFER_HIDDEN)]],                       \
    device const float *inject [[buffer(SEISMIC_BUFFER_INJECT)]],                       \
    device const uchar *gated [[buffer(SEISMIC_BUFFER_GATED)]],                         \
    device uchar *staged [[buffer(SEISMIC_BUFFER_SCRATCH_STAGED)]],                     \
    device const uchar *output_weight [[buffer(SEISMIC_BUFFER_OUTPUT_WEIGHT)]],         \
    SEISMIC_RESULT_0_PARAM(result_cells),                                               \
    device float *partials [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIALS)]],                 \
    device float *small_partials [[buffer(SEISMIC_BUFFER_SCRATCH_SMALL_PARTIALS)]],     \
    device uchar *fragments [[buffer(SEISMIC_BUFFER_SCRATCH_FRAGMENTS)]],               \
    device uchar *quantized [[buffer(SEISMIC_BUFFER_SCRATCH_QUANTIZED)]],               \
    device half *row_scales [[buffer(SEISMIC_BUFFER_SCRATCH_ROW_SCALES)]],              \
    device half *block_sums [[buffer(SEISMIC_BUFFER_SCRATCH_BLOCK_SUMS)]],              \
    device float *output_coefficients [[buffer(SEISMIC_BUFFER_SCRATCH_OUTPUT_COEFFICIENTS)]], \
    device half *output_biases [[buffer(SEISMIC_BUFFER_SCRATCH_OUTPUT_BIASES)]],        \
    device float *packed [[buffer(SEISMIC_BUFFER_SCRATCH_PACKED)]],                     \
    device half *token_factors [[buffer(SEISMIC_BUFFER_SCRATCH_TOKEN_FACTORS)]],        \
    device half *output_factors [[buffer(SEISMIC_BUFFER_SCRATCH_OUTPUT_FACTORS)]],      \
    device float *token_scales [[buffer(SEISMIC_BUFFER_SCRATCH_TOKEN_SCALES)]],         \
    device float *output_scales [[buffer(SEISMIC_BUFFER_SCRATCH_OUTPUT_SCALES)]],       \
    device const float *output_scale [[buffer(SEISMIC_BUFFER_OUTPUT_SCALE)]],           \
    device const float *output_bound [[buffer(SEISMIC_BUFFER_OUTPUT_BOUND)]],           \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

// `gated` is [M, Q, W] and canonical, so a row is Q*W contiguous values.
// `raw` is the gated rows; `in` the rows every form past the GEMV reads.
#define PROJECTION_OPERANDS(group)                                                      \
    const uint m = uint(SEISMIC_DIM_M), n = uint(SEISMIC_DIM_D), k = uint(SEISMIC_DIM_Q * SEISMIC_DIM_W); \
    const projection::Plain<activation, projection::AllRows> raw{gated, SEISMIC_GATED_STRIDE_0, \
        SEISMIC_GATED_STRIDE_2, k, {}};                                                 \
    const projection::Plain<activation, projection::AllRows> in = NVFP4                 \
        ? projection::Plain<activation, projection::AllRows>{staged, k, 1, k, {}}       \
        : raw;                                                                          \
    ATTENTION_OUTPUT_EPILOGUE;                                                          \
    projection::Weights<packets::W0> w{output_weight, KERNEL_W0_LAYOUT(k), k}

// The residual join: a plain residual, or the scatter into each
// hyper-connection stream (`scatter`, with its comb when the inject row
// carries one), after the weight's second-level scale where it has one.
typedef projection::OutputOf<activation, projection::AllRows, SEISMIC_DIM_SN, SEISMIC_DIM_IN != 0,
    (SEISMIC_DIM_IN > SEISMIC_DIM_SN)> output_join;
#define ATTENTION_OUTPUT_EPILOGUE                                                       \
    const auto out = projection::scaling<(SEISMIC_DIM_OS != 0)>::wrap(                  \
        output_join::make(element::Out<element::F32>(result_cells), SEISMIC_RESULT_0_STRIDE_0, \
            SEISMIC_RESULT_0_STRIDE_1, hidden, SEISMIC_HIDDEN_STRIDE_0, SEISMIC_HIDDEN_STRIDE_1, {}, \
            inject, SEISMIC_INJECT_STRIDE_0, uint(SEISMIC_DIM_D)),                      \
        projection::scale_factor(output_scale, SEISMIC_DIM_OS, 0, 0), 1.0f)


// The gated rows as NVFP4 operands into `staged`, one row a threadgroup.
#ifdef SEISMIC_FORMING_ATTENTION_OUTPUT_STAGE
kernel void attention_output_stage(PROJECTION_ARGUMENTS,
    uint item [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    PROJECTION_NORMALIZE_SHARED(maxima);
    PROJECTION_OPERANDS(0u);
    projection::operands<true>::form<256>(raw, item, staged, k, maxima, thread_index);
}
#endif

#define PROJECTION_GROUPS 1
#define PROJECTION_GEMV_INPUT(x) const auto x = raw
#define PROJECTION_INT8_SCRATCH \
    const projection::int8_scratch scratch{quantized, row_scales, block_sums, output_coefficients, output_biases}
#define PROJECTION_PACKING_SCRATCH \
    const projection::packing_scratch scratch{packed, token_factors, token_scales, output_factors, output_scales}
#define PROJECTION_BOUNDS                                                               \
    constexpr uint BOUNDS = SEISMIC_DIM_OB != 0 ? 1 : 0;                                \
    const projection::packing_bounds<BOUNDS> bounds{{output_bound}}

// The `dense` projection's launches (`lib/projection/operator.h`).
#ifdef SEISMIC_FORMING_ATTENTION_OUTPUT_GEMV
template <uint LANES, uint TILED, uint STAGE, uint ROWS>
kernel void attention_output_gemv(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_SINGLE_GEMV(ONE, TILED);
}
#endif
#ifdef SEISMIC_FORMING_ATTENTION_OUTPUT_GEMV_PAIR
template <uint LANES, uint TILED, uint STAGE, uint ROWS>
kernel void attention_output_gemv_pair(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_SINGLE_GEMV(PAIR, TILED);
}
#endif
#ifdef SEISMIC_FORMING_ATTENTION_OUTPUT_GEMV_ROWS
template <uint LANES, uint TILED, uint STAGE, uint ROWS>
kernel void attention_output_gemv_rows(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_SINGLE_GEMV(SEVERAL, TILED);
}
#endif
#ifdef SEISMIC_FORMING_ATTENTION_OUTPUT_BATCH
template <uint MATRIX, uint BATCH_ROWS, uint BATCH_PARTS>
kernel void attention_output_batch(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_SINGLE_BATCH;
}
#endif
#ifdef SEISMIC_FORMING_ATTENTION_OUTPUT_GEMM_SMALL
kernel void attention_output_gemm_small(PROJECTION_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_SINGLE_GEMM_SMALL;
}
#endif
#ifdef SEISMIC_FORMING_ATTENTION_OUTPUT_REDUCE_SMALL
kernel void attention_output_reduce_small(PROJECTION_ARGUMENTS, PROJECTION_ITEM_ARGUMENTS) {
    PROJECTION_SINGLE_REDUCE_SMALL;
}
#endif
#ifdef SEISMIC_FORMING_ATTENTION_OUTPUT_GEMM
template <uint TILE_M, uint TILE_N>
kernel void attention_output_gemm(PROJECTION_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_SINGLE_GEMM(uint(SEISMIC_RUNTIME_SPLIT));
}
#endif
#ifdef SEISMIC_FORMING_ATTENTION_OUTPUT_REDUCE
kernel void attention_output_reduce(PROJECTION_ARGUMENTS, PROJECTION_ITEM_ARGUMENTS) {
    PROJECTION_SINGLE_REDUCE(uint(SEISMIC_RUNTIME_SPLIT));
}
#endif
#ifdef SEISMIC_FORMING_ATTENTION_OUTPUT_RELAYOUT
kernel void attention_output_relayout(PROJECTION_ARGUMENTS, PROJECTION_ITEM_ARGUMENTS) {
    PROJECTION_SINGLE_RELAYOUT;
}
#endif
#ifdef SEISMIC_FORMING_ATTENTION_OUTPUT_TALL
template <uint TALL_M, uint TALL_K, uint STAGERS>
kernel void attention_output_tall(PROJECTION_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_SINGLE_TALL;
}
#endif
#ifdef SEISMIC_FORMING_ATTENTION_OUTPUT_QUANTIZE
kernel void attention_output_quantize(PROJECTION_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    PROJECTION_SINGLE_QUANTIZE;
}
#endif
#ifdef SEISMIC_FORMING_ATTENTION_OUTPUT_COEFFICIENTS
kernel void attention_output_coefficients(PROJECTION_ARGUMENTS, PROJECTION_ITEM_ARGUMENTS) {
    PROJECTION_SINGLE_COEFFICIENTS;
}
#endif
#ifdef SEISMIC_FORMING_ATTENTION_OUTPUT_INT8
template <uint INT8_TILE>
kernel void attention_output_int8(PROJECTION_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_SINGLE_INT8;
}
#endif
#ifdef SEISMIC_FORMING_ATTENTION_OUTPUT_PACK
template <uint RESIDUE>
kernel void attention_output_pack(PROJECTION_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    PROJECTION_SINGLE_PACK;
}
#endif
#ifdef SEISMIC_FORMING_ATTENTION_OUTPUT_PACK_COEFFICIENTS
kernel void attention_output_pack_coefficients(PROJECTION_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    PROJECTION_SINGLE_PACK_COEFFICIENTS;
}
#endif
#ifdef SEISMIC_FORMING_ATTENTION_OUTPUT_PACKED
template <uint RESIDUE, uint PACK_TOKENS, uint WEIGHTS_AHEAD>
kernel void attention_output_packed(PROJECTION_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    PROJECTION_SINGLE_PACKED;
}
#endif
#ifdef SEISMIC_FORMING_ATTENTION_OUTPUT_UNPACKED
kernel void attention_output_unpacked(PROJECTION_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_SINGLE_UNPACKED;
}
#endif
