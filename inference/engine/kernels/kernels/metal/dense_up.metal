// dense_up: RMS prologue over the `out_rows` residual rows, the up
// projection, and A(act(A(up))) (`activation`: ReLU²), through the `dense`
// projection (single). The GEMV row classes normalize while staging; from
// BATCH_FROM rows `normalize` normalizes once into `normalized`, which every
// other form reads.
#define KERNEL_W0 SEISMIC_UP_WEIGHT
#include "lib/projection/projection.h"
#include "lib/projection/packing.h"
#include "lib/projection/operator.h"
#if defined(SEISMIC_FORMING_DENSE_UP_GEMV) || defined(SEISMIC_FORMING_DENSE_UP_GEMV_ROWS) || defined(SEISMIC_FORMING_DENSE_UP_GEMV_PAIR) || defined(SEISMIC_FORMING_DENSE_UP_GEMV_PACKED)
#include "lib/projection/gemv.h"
#endif
#ifdef SEISMIC_FORMING_DENSE_UP_GEMV_PACKED
#include "lib/projection/gemv_packed.h"
#endif

typedef element::Act activation;
typedef ELEMENT_OF(SEISMIC_NORM) norm_element;
// The weight's operand rows (the normalized rows, or NVFP4 operands).
constant constexpr bool NVFP4 = projection::OperandRecipe<packets::W0_OPERANDS_NVFP4>::NVFP4;
typedef projection::operands<NVFP4> Operands;

#define PROJECTION_ARGUMENTS                                                            \
    device const float *residual [[buffer(SEISMIC_BUFFER_RESIDUAL)]],                   \
    device const uchar *norm [[buffer(SEISMIC_BUFFER_NORM)]],                           \
    device const uchar *up_weight [[buffer(SEISMIC_BUFFER_UP_WEIGHT)]],                 \
    device const int *out_rows [[buffer(SEISMIC_BUFFER_OUT_ROWS)]],                     \
    SEISMIC_RESULT_0_PARAM(result_cells),                                               \
    device uchar *normalized [[buffer(SEISMIC_BUFFER_SCRATCH_NORMALIZED)]],             \
    device float *partials [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIALS)]],                 \
    device float *small_partials [[buffer(SEISMIC_BUFFER_SCRATCH_SMALL_PARTIALS)]],     \
    device uchar *fragments [[buffer(SEISMIC_BUFFER_SCRATCH_FRAGMENTS)]],               \
    device uchar *quantized [[buffer(SEISMIC_BUFFER_SCRATCH_QUANTIZED)]],               \
    device half *row_scales [[buffer(SEISMIC_BUFFER_SCRATCH_ROW_SCALES)]],              \
    device half *block_sums [[buffer(SEISMIC_BUFFER_SCRATCH_BLOCK_SUMS)]],              \
    device float *coefficients [[buffer(SEISMIC_BUFFER_SCRATCH_UP_COEFFICIENTS)]],         \
    device half *biases [[buffer(SEISMIC_BUFFER_SCRATCH_UP_BIASES)]],                      \
    device float *packed [[buffer(SEISMIC_BUFFER_SCRATCH_PACKED)]],                     \
    device half *token_factors [[buffer(SEISMIC_BUFFER_SCRATCH_TOKEN_FACTORS)]],        \
    device half *factors [[buffer(SEISMIC_BUFFER_SCRATCH_UP_FACTORS)]],                    \
    device float *token_scales [[buffer(SEISMIC_BUFFER_SCRATCH_TOKEN_SCALES)]],         \
    device float *scales [[buffer(SEISMIC_BUFFER_SCRATCH_UP_SCALES)]],                     \
    device const float *up_scale [[buffer(SEISMIC_BUFFER_UP_SCALE)]],                   \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

// `raw` is the selected residual rows under the norm prologue; `in` the
// normalized rows every form past the GEMV reads.
#define PROJECTION_OPERANDS(group)                                                      \
    const uint m = uint(SEISMIC_DIM_O), n = uint(SEISMIC_DIM_F), k = uint(SEISMIC_DIM_H); \
    projection::Rms<activation, norm_element, projection::SelectedRows> raw{residual,   \
        SEISMIC_RESIDUAL_STRIDE_0, SEISMIC_RESIDUAL_STRIDE_1, norm, SEISMIC_NORM_STRIDE_0, \
        as_type<float>(uint(SEISMIC_PARAM_EPS)), k, {out_rows}};                        \
    const projection::Plain<activation, projection::AllRows> in{normalized, k, 1, k, {}}; \
    const auto out = projection::scaling<(SEISMIC_DIM_US != 0)>::wrap(                  \
        projection::Activated<activation>{element::Out<activation>(result_cells), SEISMIC_RESULT_0_STRIDE_0, \
            SEISMIC_RESULT_0_STRIDE_1, int(SEISMIC_PARAM_ACTIVATION)},                  \
        projection::scale_factor(up_scale, SEISMIC_DIM_US, 0, 0), 1.0f);                \
    projection::Weights<packets::W0> w{up_weight, KERNEL_W0_LAYOUT(k), k}

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
#ifdef SEISMIC_FORMING_DENSE_UP_NORMALIZE
kernel void dense_up_normalize(PROJECTION_ARGUMENTS,
    uint item [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    PROJECTION_NORMALIZE_SHARED(norms);
    PROJECTION_OPERANDS(0u);
    Operands::normalize<256>(raw, item, normalized, k, norms, thread_index);
}
#endif

// The `dense` projection's launches (`lib/projection/operator.h`).
#ifdef SEISMIC_FORMING_DENSE_UP_GEMV
template <uint LANES, uint TILED, uint STAGE, uint ROWS>
kernel void dense_up_gemv(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_SINGLE_GEMV(ONE, TILED);
}
#endif
#ifdef SEISMIC_FORMING_DENSE_UP_GEMV_PAIR
template <uint LANES, uint TILED, uint STAGE, uint PACKED_ROWS, uint ROWS>
kernel void dense_up_gemv_pair(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_SINGLE_GEMV(PAIR, TILED);
}
#endif
#ifdef SEISMIC_FORMING_DENSE_UP_GEMV_ROWS
template <uint LANES, uint TILED, uint STAGE, uint PACKED_ROWS, uint ROWS>
kernel void dense_up_gemv_rows(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_SINGLE_GEMV(SEVERAL, TILED);
}
#endif
#ifdef SEISMIC_FORMING_DENSE_UP_GEMV_PACKED
template <uint LANES, uint PACKED_ROWS, uint ROWS>
kernel void dense_up_gemv_packed(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_SINGLE_GEMV_PACKED;
}
#endif
#ifdef SEISMIC_FORMING_DENSE_UP_BATCH
template <uint MATRIX, uint BATCH_ROWS, uint BATCH_PARTS>
kernel void dense_up_batch(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_SINGLE_BATCH;
}
#endif
#ifdef SEISMIC_FORMING_DENSE_UP_GEMM_SMALL
kernel void dense_up_gemm_small(PROJECTION_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_SINGLE_GEMM_SMALL;
}
#endif
#ifdef SEISMIC_FORMING_DENSE_UP_REDUCE_SMALL
kernel void dense_up_reduce_small(PROJECTION_ARGUMENTS, PROJECTION_ITEM_ARGUMENTS) {
    PROJECTION_SINGLE_REDUCE_SMALL;
}
#endif
#ifdef SEISMIC_FORMING_DENSE_UP_GEMM
template <uint TILE_M, uint TILE_N>
kernel void dense_up_gemm(PROJECTION_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_SINGLE_GEMM(uint(SEISMIC_RUNTIME_SPLIT));
}
#endif
#ifdef SEISMIC_FORMING_DENSE_UP_REDUCE
kernel void dense_up_reduce(PROJECTION_ARGUMENTS, PROJECTION_ITEM_ARGUMENTS) {
    PROJECTION_SINGLE_REDUCE(uint(SEISMIC_RUNTIME_SPLIT));
}
#endif
#ifdef SEISMIC_FORMING_DENSE_UP_RELAYOUT
kernel void dense_up_relayout(PROJECTION_ARGUMENTS, PROJECTION_ITEM_ARGUMENTS) {
    PROJECTION_SINGLE_RELAYOUT;
}
#endif
#ifdef SEISMIC_FORMING_DENSE_UP_TALL
template <uint TALL_M, uint TALL_K, uint STAGERS>
kernel void dense_up_tall(PROJECTION_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_SINGLE_TALL;
}
#endif
#ifdef SEISMIC_FORMING_DENSE_UP_QUANTIZE
kernel void dense_up_quantize(PROJECTION_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    PROJECTION_SINGLE_QUANTIZE;
}
#endif
#ifdef SEISMIC_FORMING_DENSE_UP_COEFFICIENTS
kernel void dense_up_coefficients(PROJECTION_ARGUMENTS, PROJECTION_ITEM_ARGUMENTS) {
    PROJECTION_SINGLE_COEFFICIENTS;
}
#endif
#ifdef SEISMIC_FORMING_DENSE_UP_INT8
template <uint INT8_TILE>
kernel void dense_up_int8(PROJECTION_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_SINGLE_INT8;
}
#endif
#ifdef SEISMIC_FORMING_DENSE_UP_PACK
template <uint RESIDUE>
kernel void dense_up_pack(PROJECTION_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    PROJECTION_SINGLE_PACK;
}
#endif
#ifdef SEISMIC_FORMING_DENSE_UP_PACK_COEFFICIENTS
kernel void dense_up_pack_coefficients(PROJECTION_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    PROJECTION_SINGLE_PACK_COEFFICIENTS;
}
#endif
#ifdef SEISMIC_FORMING_DENSE_UP_PACKED
template <uint RESIDUE, uint PACK_TOKENS, uint WEIGHTS_AHEAD>
kernel void dense_up_packed(PROJECTION_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    PROJECTION_SINGLE_PACKED;
}
#endif
#ifdef SEISMIC_FORMING_DENSE_UP_UNPACKED
kernel void dense_up_unpacked(PROJECTION_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_SINGLE_UNPACKED;
}
#endif
