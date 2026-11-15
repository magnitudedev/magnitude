// per_layer_gate: `stage` rounds the F32 hidden rows to A in scratch; the
// `dense` projection (single) projects them through the gate rows with the
// activated-product epilogue A(A(act(A(gate))) * inputs[m, layer, p]).
#define KERNEL_W0 SEISMIC_GATE_WEIGHT
#include "lib/projection/projection.h"
#include "lib/projection/packing.h"
#include "lib/projection/operator.h"
#if defined(SEISMIC_FORMING_PER_LAYER_GATE_GEMV) || defined(SEISMIC_FORMING_PER_LAYER_GATE_GEMV_ROWS) || defined(SEISMIC_FORMING_PER_LAYER_GATE_GEMV_PAIR)
#include "lib/projection/gemv.h"
#endif

typedef element::Act activation;
// The gate weight's operand rows (the rounded rows, or NVFP4 operands),
// which `stage` writes and every other launch reads.
constant constexpr bool NVFP4 = projection::OperandRecipe<packets::W0_OPERANDS_NVFP4>::NVFP4;
typedef projection::operands<NVFP4> Operands;

#define PROJECTION_ARGUMENTS                                                            \
    device const float *hidden [[buffer(SEISMIC_BUFFER_HIDDEN)]],                       \
    device const uchar *gate_weight [[buffer(SEISMIC_BUFFER_GATE_WEIGHT)]],             \
    device const float *inputs [[buffer(SEISMIC_BUFFER_INPUTS)]],                       \
    SEISMIC_RESULT_0_PARAM(result_cells),                                               \
    device uchar *rounded [[buffer(SEISMIC_BUFFER_SCRATCH_ROUNDED)]],                   \
    device float *partials [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIALS)]],                 \
    device float *small_partials [[buffer(SEISMIC_BUFFER_SCRATCH_SMALL_PARTIALS)]],     \
    device uchar *fragments [[buffer(SEISMIC_BUFFER_SCRATCH_FRAGMENTS)]],               \
    device uchar *quantized [[buffer(SEISMIC_BUFFER_SCRATCH_QUANTIZED)]],               \
    device half *row_scales [[buffer(SEISMIC_BUFFER_SCRATCH_ROW_SCALES)]],              \
    device half *block_sums [[buffer(SEISMIC_BUFFER_SCRATCH_BLOCK_SUMS)]],              \
    device float *coefficients [[buffer(SEISMIC_BUFFER_SCRATCH_GATE_COEFFICIENTS)]],         \
    device half *biases [[buffer(SEISMIC_BUFFER_SCRATCH_GATE_BIASES)]],                      \
    device float *packed [[buffer(SEISMIC_BUFFER_SCRATCH_PACKED)]],                     \
    device half *token_factors [[buffer(SEISMIC_BUFFER_SCRATCH_TOKEN_FACTORS)]],        \
    device half *factors [[buffer(SEISMIC_BUFFER_SCRATCH_GATE_FACTORS)]],                    \
    device float *token_scales [[buffer(SEISMIC_BUFFER_SCRATCH_TOKEN_SCALES)]],         \
    device float *scales [[buffer(SEISMIC_BUFFER_SCRATCH_GATE_SCALES)]],                     \
    device const float *gate_scale [[buffer(SEISMIC_BUFFER_GATE_SCALE)]],               \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

// `raw` and `in` are the rounded rows every form reads.
#define PROJECTION_OPERANDS(group)                                                      \
    const uint m = uint(SEISMIC_DIM_M), n = uint(SEISMIC_DIM_P), k = uint(SEISMIC_DIM_D); \
    const projection::Plain<activation, projection::AllRows> raw{rounded, SEISMIC_DIM_D, 1, k, {}}; \
    const auto in = raw;                                                                \
    const auto out = projection::scaling<(SEISMIC_DIM_GS != 0)>::wrap(                  \
        projection::ActivatedMul<activation>{element::Out<activation>(result_cells), SEISMIC_RESULT_0_STRIDE_0, \
            SEISMIC_RESULT_0_STRIDE_1, inputs + ulong(int(SEISMIC_PARAM_LAYER)) * SEISMIC_INPUTS_STRIDE_1, \
            SEISMIC_INPUTS_STRIDE_0, \
            SEISMIC_INPUTS_STRIDE_2, int(SEISMIC_PARAM_ACTIVATION)},                    \
        projection::scale_factor(gate_scale, SEISMIC_DIM_GS, 0, 0), 1.0f);              \
    projection::Weights<packets::W0> w{gate_weight, KERNEL_W0_LAYOUT(k), k}

#define PROJECTION_GROUPS 1
#define PROJECTION_GEMV_INPUT(x) const auto x = raw
#define PROJECTION_INT8_SCRATCH \
    const projection::int8_scratch scratch{quantized, row_scales, block_sums, coefficients, biases}
#define PROJECTION_PACKING_SCRATCH \
    const projection::packing_scratch scratch{packed, token_factors, token_scales, factors, scales}
#define PROJECTION_BOUNDS                                                               \
    constexpr uint BOUNDS = 0;                                                          \
    const projection::packing_bounds<BOUNDS> bounds{{}}

#ifdef SEISMIC_FORMING_PER_LAYER_GATE_STAGE
kernel void per_layer_gate_stage(PROJECTION_ARGUMENTS,
    uint item [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    if constexpr (NVFP4) {
        PROJECTION_NORMALIZE_SHARED(maxima);
        // The hidden rows rounded to A (one part, no norm).
        const projection::Summed<activation, activation, projection::AllRows, 1> source{hidden,
            SEISMIC_HIDDEN_STRIDE_0, SEISMIC_HIDDEN_STRIDE_1, nullptr, 0, 0.0f, uint(SEISMIC_DIM_D), {}};
        projection::operands<true>::form<256>(source, item, rounded, uint(SEISMIC_DIM_D), maxima, thread_index);
        return;
    }
    device typename activation::storage *row =
        reinterpret_cast<device typename activation::storage *>(rounded) + ulong(item) * SEISMIC_DIM_D;
    for (uint i = thread_index; i < uint(SEISMIC_DIM_D); i += 256u)
        row[i] = activation::store(hidden[ulong(item) * SEISMIC_HIDDEN_STRIDE_0 + ulong(i) * SEISMIC_HIDDEN_STRIDE_1]);
}
#endif

// The `dense` projection's launches (`lib/projection/operator.h`).
#ifdef SEISMIC_FORMING_PER_LAYER_GATE_GEMV
template <uint LANES, uint TILED, uint ROWS>
kernel void per_layer_gate_gemv(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_SINGLE_GEMV(ONE, TILED);
}
#endif
#ifdef SEISMIC_FORMING_PER_LAYER_GATE_GEMV_PAIR
template <uint LANES, uint TILED, uint ROWS>
kernel void per_layer_gate_gemv_pair(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_SINGLE_GEMV(PAIR, TILED);
}
#endif
#ifdef SEISMIC_FORMING_PER_LAYER_GATE_GEMV_ROWS
template <uint LANES, uint TILED, uint ROWS>
kernel void per_layer_gate_gemv_rows(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_SINGLE_GEMV(SEVERAL, TILED);
}
#endif
#ifdef SEISMIC_FORMING_PER_LAYER_GATE_BATCH
template <uint MATRIX, uint BATCH_ROWS, uint BATCH_PARTS>
kernel void per_layer_gate_batch(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_SINGLE_BATCH;
}
#endif
#ifdef SEISMIC_FORMING_PER_LAYER_GATE_GEMM_SMALL
kernel void per_layer_gate_gemm_small(PROJECTION_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_SINGLE_GEMM_SMALL;
}
#endif
#ifdef SEISMIC_FORMING_PER_LAYER_GATE_REDUCE_SMALL
kernel void per_layer_gate_reduce_small(PROJECTION_ARGUMENTS, PROJECTION_ITEM_ARGUMENTS) {
    PROJECTION_SINGLE_REDUCE_SMALL;
}
#endif
#ifdef SEISMIC_FORMING_PER_LAYER_GATE_GEMM
template <uint TILE_M, uint TILE_N>
kernel void per_layer_gate_gemm(PROJECTION_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_SINGLE_GEMM(uint(SEISMIC_RUNTIME_SPLIT));
}
#endif
#ifdef SEISMIC_FORMING_PER_LAYER_GATE_REDUCE
kernel void per_layer_gate_reduce(PROJECTION_ARGUMENTS, PROJECTION_ITEM_ARGUMENTS) {
    PROJECTION_SINGLE_REDUCE(uint(SEISMIC_RUNTIME_SPLIT));
}
#endif
#ifdef SEISMIC_FORMING_PER_LAYER_GATE_RELAYOUT
kernel void per_layer_gate_relayout(PROJECTION_ARGUMENTS, PROJECTION_ITEM_ARGUMENTS) {
    PROJECTION_SINGLE_RELAYOUT;
}
#endif
#ifdef SEISMIC_FORMING_PER_LAYER_GATE_TALL
template <uint TALL_M, uint TALL_K, uint STAGERS>
kernel void per_layer_gate_tall(PROJECTION_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_SINGLE_TALL;
}
#endif
#ifdef SEISMIC_FORMING_PER_LAYER_GATE_QUANTIZE
kernel void per_layer_gate_quantize(PROJECTION_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    PROJECTION_SINGLE_QUANTIZE;
}
#endif
#ifdef SEISMIC_FORMING_PER_LAYER_GATE_COEFFICIENTS
kernel void per_layer_gate_coefficients(PROJECTION_ARGUMENTS, PROJECTION_ITEM_ARGUMENTS) {
    PROJECTION_SINGLE_COEFFICIENTS;
}
#endif
#ifdef SEISMIC_FORMING_PER_LAYER_GATE_INT8
template <uint INT8_TILE>
kernel void per_layer_gate_int8(PROJECTION_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_SINGLE_INT8;
}
#endif
#ifdef SEISMIC_FORMING_PER_LAYER_GATE_PACK
template <uint RESIDUE>
kernel void per_layer_gate_pack(PROJECTION_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    PROJECTION_SINGLE_PACK;
}
#endif
#ifdef SEISMIC_FORMING_PER_LAYER_GATE_PACK_COEFFICIENTS
kernel void per_layer_gate_pack_coefficients(PROJECTION_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    PROJECTION_SINGLE_PACK_COEFFICIENTS;
}
#endif
#ifdef SEISMIC_FORMING_PER_LAYER_GATE_PACKED
template <uint RESIDUE, uint PACK_TOKENS, uint WEIGHTS_AHEAD>
kernel void per_layer_gate_packed(PROJECTION_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    PROJECTION_SINGLE_PACKED;
}
#endif
#ifdef SEISMIC_FORMING_PER_LAYER_GATE_UNPACKED
kernel void per_layer_gate_unpacked(PROJECTION_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_SINGLE_UNPACKED;
}
#endif
