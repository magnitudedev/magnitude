// readout_selected_rows: final RMS prologue over the `out_rows` hidden rows and
// the projection onto the `selected` vocabulary rows into F32 logits, through
// the `dense` projection (single, the weight rows gathered). The GEMV row
// classes normalize while staging; from BATCH_FROM rows `stage` normalizes
// once into `normalized`, which every other form reads. Logits are softcapped
// from the accumulator when `softcap` > 0.
#define KERNEL_W0 SEISMIC_WEIGHT
#include "lib/projection/projection.h"
#include "lib/projection/packing.h"
#include "lib/projection/operator.h"
#if defined(SEISMIC_FORMING_READOUT_SELECTED_ROWS_GEMV) || defined(SEISMIC_FORMING_READOUT_SELECTED_ROWS_GEMV_ROWS) || defined(SEISMIC_FORMING_READOUT_SELECTED_ROWS_GEMV_PAIR)
#include "lib/projection/gemv.h"
#endif

typedef element::Act activation;
typedef ELEMENT_OF(SEISMIC_NORM) norm_element;
// The weight's operand rows (the normalized rows, or NVFP4 operands).
constant constexpr bool NVFP4 = projection::OperandRecipe<packets::W0_OPERANDS_NVFP4>::NVFP4;
typedef projection::operands<NVFP4> Operands;

#define PROJECTION_ARGUMENTS                                                            \
    device const float *hidden [[buffer(SEISMIC_BUFFER_HIDDEN)]],                       \
    device const uchar *norm [[buffer(SEISMIC_BUFFER_NORM)]],                           \
    device const uchar *weight [[buffer(SEISMIC_BUFFER_WEIGHT)]],                       \
    device const int *out_rows [[buffer(SEISMIC_BUFFER_OUT_ROWS)]],                     \
    device const int *selected [[buffer(SEISMIC_BUFFER_SELECTED)]],                     \
    SEISMIC_RESULT_0_PARAM(logits),                                                     \
    device uchar *normalized [[buffer(SEISMIC_BUFFER_SCRATCH_NORMALIZED)]],             \
    device float *partials [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIALS)]],                 \
    device float *small_partials [[buffer(SEISMIC_BUFFER_SCRATCH_SMALL_PARTIALS)]],     \
    device uchar *fragments [[buffer(SEISMIC_BUFFER_SCRATCH_FRAGMENTS)]],               \
    device uchar *quantized [[buffer(SEISMIC_BUFFER_SCRATCH_QUANTIZED)]],               \
    device half *row_scales [[buffer(SEISMIC_BUFFER_SCRATCH_ROW_SCALES)]],              \
    device half *block_sums [[buffer(SEISMIC_BUFFER_SCRATCH_BLOCK_SUMS)]],              \
    device float *coefficients [[buffer(SEISMIC_BUFFER_SCRATCH_COEFFICIENTS)]],         \
    device half *biases [[buffer(SEISMIC_BUFFER_SCRATCH_BIASES)]],                      \
    device float *packed [[buffer(SEISMIC_BUFFER_SCRATCH_PACKED)]],                     \
    device half *token_factors [[buffer(SEISMIC_BUFFER_SCRATCH_TOKEN_FACTORS)]],        \
    device half *factors [[buffer(SEISMIC_BUFFER_SCRATCH_FACTORS)]],                    \
    device float *token_scales [[buffer(SEISMIC_BUFFER_SCRATCH_TOKEN_SCALES)]],         \
    device float *scales [[buffer(SEISMIC_BUFFER_SCRATCH_SCALES)]],                     \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

// `raw` is the selected hidden rows under the norm prologue; `in` the
// normalized rows every form past the GEMV reads.
#define PROJECTION_OPERANDS(group)                                                      \
    const uint m = uint(SEISMIC_DIM_O), n = uint(SEISMIC_DIM_SV), k = uint(SEISMIC_DIM_D); \
    projection::InputOf<activation, norm_element, projection::SelectedRows, SEISMIC_DIM_NM, SEISMIC_DIM_PT>::type \
        raw{hidden, SEISMIC_HIDDEN_STRIDE_0, SEISMIC_HIDDEN_STRIDE_1, norm, SEISMIC_NORM_STRIDE_0, \
        as_type<float>(uint(SEISMIC_PARAM_EPSILON)), k, {out_rows}};                    \
    const projection::Plain<activation, projection::AllRows> in{normalized, k, 1, k, {}}; \
    const projection::Logits out{element::Out<element::F32>(logits), SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1, \
        as_type<float>(uint(SEISMIC_PARAM_SOFTCAP))};                                   \
    projection::Weights<packets::W0> w{weight, KERNEL_W0_LAYOUT(k), k, selected}

// The GEMV's rows: the norm applied as they stage, from their sums of
// squares over the threadgroup.
#define PROJECTION_GROUPS 1
#define PROJECTION_GEMV_INPUT(x)                                                        \
    PROJECTION_SQUARES_SHARED(squares, decltype(raw)::parts);                           \
    projection::threadgroup_squares_runtime(raw, m, squares, simdgroups, sg, lane);     \
    const auto x = projection::shared_norm(raw, squares)
#define PROJECTION_INT8_SCRATCH \
    const projection::int8_scratch scratch{quantized, row_scales, block_sums, coefficients, biases}
#define PROJECTION_PACKING_SCRATCH \
    const projection::packing_scratch scratch{packed, token_factors, token_scales, factors, scales}
#define PROJECTION_BOUNDS                                                               \
    constexpr uint BOUNDS = 0;                                                          \
    const projection::packing_bounds<BOUNDS> bounds{{}}

// The normalized rows into `normalized`, one row a threadgroup.
#ifdef SEISMIC_FORMING_READOUT_SELECTED_ROWS_STAGE
kernel void readout_selected_rows_stage(PROJECTION_ARGUMENTS,
    uint item [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    PROJECTION_NORMALIZE_SHARED(norms);
    PROJECTION_OPERANDS(0u);
    Operands::normalize<256>(raw, item, normalized, k, norms, thread_index);
}
#endif

// The `dense` projection's launches (`lib/projection/operator.h`).
#ifdef SEISMIC_FORMING_READOUT_SELECTED_ROWS_GEMV
template <uint LANES, uint TILED, uint ROWS>
kernel void readout_selected_rows_gemv(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_SINGLE_GEMV(ONE, TILED);
}
#endif
#ifdef SEISMIC_FORMING_READOUT_SELECTED_ROWS_GEMV_PAIR
template <uint LANES, uint TILED, uint ROWS>
kernel void readout_selected_rows_gemv_pair(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_SINGLE_GEMV(PAIR, TILED);
}
#endif
#ifdef SEISMIC_FORMING_READOUT_SELECTED_ROWS_GEMV_ROWS
template <uint LANES, uint TILED, uint ROWS>
kernel void readout_selected_rows_gemv_rows(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_SINGLE_GEMV(SEVERAL, TILED);
}
#endif
#ifdef SEISMIC_FORMING_READOUT_SELECTED_ROWS_BATCH
template <uint MATRIX, uint BATCH_ROWS, uint BATCH_PARTS>
kernel void readout_selected_rows_batch(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_SINGLE_BATCH;
}
#endif
#ifdef SEISMIC_FORMING_READOUT_SELECTED_ROWS_GEMM_SMALL
kernel void readout_selected_rows_gemm_small(PROJECTION_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_SINGLE_GEMM_SMALL;
}
#endif
#ifdef SEISMIC_FORMING_READOUT_SELECTED_ROWS_REDUCE_SMALL
kernel void readout_selected_rows_reduce_small(PROJECTION_ARGUMENTS, PROJECTION_ITEM_ARGUMENTS) {
    PROJECTION_SINGLE_REDUCE_SMALL;
}
#endif
#ifdef SEISMIC_FORMING_READOUT_SELECTED_ROWS_GEMM
template <uint TILE_M, uint TILE_N>
kernel void readout_selected_rows_gemm(PROJECTION_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_SINGLE_GEMM(uint(SEISMIC_RUNTIME_SPLIT));
}
#endif
#ifdef SEISMIC_FORMING_READOUT_SELECTED_ROWS_REDUCE
kernel void readout_selected_rows_reduce(PROJECTION_ARGUMENTS, PROJECTION_ITEM_ARGUMENTS) {
    PROJECTION_SINGLE_REDUCE(uint(SEISMIC_RUNTIME_SPLIT));
}
#endif
#ifdef SEISMIC_FORMING_READOUT_SELECTED_ROWS_RELAYOUT
kernel void readout_selected_rows_relayout(PROJECTION_ARGUMENTS, PROJECTION_ITEM_ARGUMENTS) {
    PROJECTION_SINGLE_RELAYOUT;
}
#endif
#ifdef SEISMIC_FORMING_READOUT_SELECTED_ROWS_TALL
template <uint TALL_M, uint TALL_K, uint STAGERS>
kernel void readout_selected_rows_tall(PROJECTION_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_SINGLE_TALL;
}
#endif
#ifdef SEISMIC_FORMING_READOUT_SELECTED_ROWS_QUANTIZE
kernel void readout_selected_rows_quantize(PROJECTION_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    PROJECTION_SINGLE_QUANTIZE;
}
#endif
#ifdef SEISMIC_FORMING_READOUT_SELECTED_ROWS_COEFFICIENTS
kernel void readout_selected_rows_coefficients(PROJECTION_ARGUMENTS, PROJECTION_ITEM_ARGUMENTS) {
    PROJECTION_SINGLE_COEFFICIENTS;
}
#endif
#ifdef SEISMIC_FORMING_READOUT_SELECTED_ROWS_INT8
template <uint INT8_TILE>
kernel void readout_selected_rows_int8(PROJECTION_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_SINGLE_INT8;
}
#endif
#ifdef SEISMIC_FORMING_READOUT_SELECTED_ROWS_PACK
template <uint RESIDUE>
kernel void readout_selected_rows_pack(PROJECTION_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    PROJECTION_SINGLE_PACK;
}
#endif
#ifdef SEISMIC_FORMING_READOUT_SELECTED_ROWS_PACK_COEFFICIENTS
kernel void readout_selected_rows_pack_coefficients(PROJECTION_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    PROJECTION_SINGLE_PACK_COEFFICIENTS;
}
#endif
#ifdef SEISMIC_FORMING_READOUT_SELECTED_ROWS_PACKED
template <uint RESIDUE, uint PACK_TOKENS, uint WEIGHTS_AHEAD>
kernel void readout_selected_rows_packed(PROJECTION_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    PROJECTION_SINGLE_PACKED;
}
#endif
#ifdef SEISMIC_FORMING_READOUT_SELECTED_ROWS_UNPACKED
kernel void readout_selected_rows_unpacked(PROJECTION_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_SINGLE_UNPACKED;
}
#endif
