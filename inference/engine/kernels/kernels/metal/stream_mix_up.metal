// stream_mix_up: the low-rank rows through `up`, each output column the gate
// of its stream's column, publishing the per-stream partials (`Mix`): the
// `dense` projection (single) over the rows as given.
#define KERNEL_W0 SEISMIC_UP_WEIGHT
#include "lib/projection/projection.h"
#include "lib/projection/packing.h"
#include "lib/projection/operator.h"
#if defined(SEISMIC_FORMING_STREAM_MIX_UP_GEMV) || defined(SEISMIC_FORMING_STREAM_MIX_UP_GEMV_ROWS) || defined(SEISMIC_FORMING_STREAM_MIX_UP_GEMV_PAIR)
#include "lib/projection/gemv.h"
#endif

typedef element::Act activation;
typedef ELEMENT_OF(SEISMIC_NORM) norm_element;
// The up weight takes no NVFP4 operands (the entry's contract): every form
// reads the rows as given.
constant constexpr bool NVFP4 = projection::OperandRecipe<packets::W0_OPERANDS_NVFP4>::NVFP4;
typedef projection::operands<NVFP4> Operands;

#define PROJECTION_ARGUMENTS                                                            \
    device const uchar *low [[buffer(SEISMIC_BUFFER_LOW)]],                             \
    device const uchar *up_weight [[buffer(SEISMIC_BUFFER_UP_WEIGHT)]],                 \
    device const float *hidden [[buffer(SEISMIC_BUFFER_HIDDEN)]],                       \
    device const float *inverse [[buffer(SEISMIC_BUFFER_INVERSE)]],                     \
    device const uchar *norm [[buffer(SEISMIC_BUFFER_NORM)]],                           \
    SEISMIC_RESULT_0_PARAM(result_cells),                                               \
    device float *partials [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIALS)]],                 \
    device float *small_partials [[buffer(SEISMIC_BUFFER_SCRATCH_SMALL_PARTIALS)]],     \
    device uchar *fragments [[buffer(SEISMIC_BUFFER_SCRATCH_FRAGMENTS)]],               \
    device uchar *quantized [[buffer(SEISMIC_BUFFER_SCRATCH_QUANTIZED)]],               \
    device half *row_scales [[buffer(SEISMIC_BUFFER_SCRATCH_ROW_SCALES)]],              \
    device half *block_sums [[buffer(SEISMIC_BUFFER_SCRATCH_BLOCK_SUMS)]],              \
    device float *up_coefficients [[buffer(SEISMIC_BUFFER_SCRATCH_UP_COEFFICIENTS)]],   \
    device half *up_biases [[buffer(SEISMIC_BUFFER_SCRATCH_UP_BIASES)]],                \
    device float *packed [[buffer(SEISMIC_BUFFER_SCRATCH_PACKED)]],                     \
    device half *token_factors [[buffer(SEISMIC_BUFFER_SCRATCH_TOKEN_FACTORS)]],        \
    device half *up_factors [[buffer(SEISMIC_BUFFER_SCRATCH_UP_FACTORS)]],              \
    device float *token_scales [[buffer(SEISMIC_BUFFER_SCRATCH_TOKEN_SCALES)]],         \
    device float *up_scales [[buffer(SEISMIC_BUFFER_SCRATCH_UP_SCALES)]],               \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

// `raw` and `in` are the low-rank rows, read in place by every form.
#define PROJECTION_OPERANDS(group)                                                      \
    const uint m = uint(SEISMIC_DIM_M), n = uint(SEISMIC_DIM_S * SEISMIC_DIM_H), k = uint(SEISMIC_DIM_LR); \
    const projection::Plain<activation, projection::AllRows> raw{low, SEISMIC_LOW_STRIDE_0, \
        SEISMIC_LOW_STRIDE_1, k, {}};                                                   \
    const auto in = raw;                                                                \
    const projection::Mix<norm_element> out{element::Out<element::F32>(result_cells), SEISMIC_RESULT_0_STRIDE_0, \
        SEISMIC_RESULT_0_STRIDE_1, hidden, SEISMIC_HIDDEN_STRIDE_0, SEISMIC_HIDDEN_STRIDE_2, \
        inverse, SEISMIC_INVERSE_STRIDE_0, SEISMIC_INVERSE_STRIDE_1, norm,              \
        SEISMIC_NORM_STRIDE_1, uint(SEISMIC_DIM_H), float(SEISMIC_DIM_S)};              \
    projection::Weights<packets::W0> w{up_weight, KERNEL_W0_LAYOUT(k), k}

#define PROJECTION_GROUPS 1
#define PROJECTION_GEMV_INPUT(x) const auto x = raw
#define PROJECTION_INT8_SCRATCH \
    const projection::int8_scratch scratch{quantized, row_scales, block_sums, up_coefficients, up_biases}
#define PROJECTION_PACKING_SCRATCH \
    const projection::packing_scratch scratch{packed, token_factors, token_scales, up_factors, up_scales}
#define PROJECTION_BOUNDS                                                               \
    constexpr uint BOUNDS = 0;                                                          \
    const projection::packing_bounds<BOUNDS> bounds{{}}

// The `dense` projection's launches (`lib/projection/operator.h`).
#ifdef SEISMIC_FORMING_STREAM_MIX_UP_GEMV
template <uint LANES, uint TILED, uint STAGE, uint ROWS>
kernel void stream_mix_up_gemv(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_SINGLE_GEMV(ONE, TILED);
}
#endif
#ifdef SEISMIC_FORMING_STREAM_MIX_UP_GEMV_PAIR
template <uint LANES, uint TILED, uint STAGE, uint ROWS>
kernel void stream_mix_up_gemv_pair(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_SINGLE_GEMV(PAIR, TILED);
}
#endif
#ifdef SEISMIC_FORMING_STREAM_MIX_UP_GEMV_ROWS
template <uint LANES, uint TILED, uint STAGE, uint ROWS>
kernel void stream_mix_up_gemv_rows(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_SINGLE_GEMV(SEVERAL, TILED);
}
#endif
#ifdef SEISMIC_FORMING_STREAM_MIX_UP_BATCH
template <uint MATRIX, uint BATCH_ROWS, uint BATCH_PARTS>
kernel void stream_mix_up_batch(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_SINGLE_BATCH;
}
#endif
#ifdef SEISMIC_FORMING_STREAM_MIX_UP_GEMM_SMALL
kernel void stream_mix_up_gemm_small(PROJECTION_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_SINGLE_GEMM_SMALL;
}
#endif
#ifdef SEISMIC_FORMING_STREAM_MIX_UP_REDUCE_SMALL
kernel void stream_mix_up_reduce_small(PROJECTION_ARGUMENTS, PROJECTION_ITEM_ARGUMENTS) {
    PROJECTION_SINGLE_REDUCE_SMALL;
}
#endif
#ifdef SEISMIC_FORMING_STREAM_MIX_UP_GEMM
template <uint TILE_M, uint TILE_N>
kernel void stream_mix_up_gemm(PROJECTION_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_SINGLE_GEMM(uint(SEISMIC_RUNTIME_SPLIT));
}
#endif
#ifdef SEISMIC_FORMING_STREAM_MIX_UP_REDUCE
kernel void stream_mix_up_reduce(PROJECTION_ARGUMENTS, PROJECTION_ITEM_ARGUMENTS) {
    PROJECTION_SINGLE_REDUCE(uint(SEISMIC_RUNTIME_SPLIT));
}
#endif
#ifdef SEISMIC_FORMING_STREAM_MIX_UP_RELAYOUT
kernel void stream_mix_up_relayout(PROJECTION_ARGUMENTS, PROJECTION_ITEM_ARGUMENTS) {
    PROJECTION_SINGLE_RELAYOUT;
}
#endif
#ifdef SEISMIC_FORMING_STREAM_MIX_UP_TALL
template <uint TALL_M, uint TALL_K, uint STAGERS>
kernel void stream_mix_up_tall(PROJECTION_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_SINGLE_TALL;
}
#endif
#ifdef SEISMIC_FORMING_STREAM_MIX_UP_QUANTIZE
kernel void stream_mix_up_quantize(PROJECTION_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    PROJECTION_SINGLE_QUANTIZE;
}
#endif
#ifdef SEISMIC_FORMING_STREAM_MIX_UP_COEFFICIENTS
kernel void stream_mix_up_coefficients(PROJECTION_ARGUMENTS, PROJECTION_ITEM_ARGUMENTS) {
    PROJECTION_SINGLE_COEFFICIENTS;
}
#endif
#ifdef SEISMIC_FORMING_STREAM_MIX_UP_INT8
template <uint INT8_TILE>
kernel void stream_mix_up_int8(PROJECTION_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_SINGLE_INT8;
}
#endif
#ifdef SEISMIC_FORMING_STREAM_MIX_UP_PACK
template <uint RESIDUE>
kernel void stream_mix_up_pack(PROJECTION_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    PROJECTION_SINGLE_PACK;
}
#endif
#ifdef SEISMIC_FORMING_STREAM_MIX_UP_PACK_COEFFICIENTS
kernel void stream_mix_up_pack_coefficients(PROJECTION_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    PROJECTION_SINGLE_PACK_COEFFICIENTS;
}
#endif
#ifdef SEISMIC_FORMING_STREAM_MIX_UP_PACKED
template <uint RESIDUE, uint PACK_TOKENS, uint WEIGHTS_AHEAD>
kernel void stream_mix_up_packed(PROJECTION_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    PROJECTION_SINGLE_PACKED;
}
#endif
#ifdef SEISMIC_FORMING_STREAM_MIX_UP_UNPACKED
kernel void stream_mix_up_unpacked(PROJECTION_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_SINGLE_UNPACKED;
}
#endif
