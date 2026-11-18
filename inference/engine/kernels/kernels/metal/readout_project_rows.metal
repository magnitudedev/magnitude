// readout_project_rows: the vocabulary projection of already-normalized
// feature rows (the draft head's readout) into F32 logits, scaled by a
// present `weight_scale` (static WS = 1) and softcapped from the accumulator
// when `softcap` > 0. Where I is 0 a row's logits are the first S vocabulary
// rows', through the `dense` projection (single); where I is 1 they are the
// S rows `token_rows` selects for it, through the routed projection's
// `choice` (`gather_choice`): each feature row an expert of one choice, the
// rows it selects that expert's matrix, a selection past the vocabulary
// -inf.
#define KERNEL_W0 SEISMIC_WEIGHT
// A selection past the vocabulary reads row 0 (`projection::Weights`).
#define PROJECTION_TABLE_ROWS SEISMIC_DIM_V
#include "lib/projection/projection.h"
#include "lib/projection/packing.h"
#include "lib/projection/operator.h"
#if defined(SEISMIC_FORMING_READOUT_PROJECT_ROWS_GEMV) || defined(SEISMIC_FORMING_READOUT_PROJECT_ROWS_GEMV_ROWS) || defined(SEISMIC_FORMING_READOUT_PROJECT_ROWS_GEMV_PAIR) || defined(SEISMIC_FORMING_READOUT_PROJECT_ROWS_GATHER_CHOICE) || \
    defined(SEISMIC_FORMING_READOUT_PROJECT_ROWS_GEMV_PACKED)
#include "lib/projection/gemv.h"
#endif
#ifdef SEISMIC_FORMING_READOUT_PROJECT_ROWS_GEMV_PACKED
#include "lib/projection/gemv_packed.h"
#endif

typedef element::Act activation;
// The weight's operand rows (the feature rows, or NVFP4 operands): the GEMV
// row classes form them while staging, the stage launch into the scratch
// every other form reads.
constant constexpr bool NVFP4 = projection::OperandRecipe<packets::W0_OPERANDS_NVFP4>::NVFP4;
typedef projection::operands<NVFP4> Operands;

#define PROJECTION_ARGUMENTS                                                            \
    device const uchar *features [[buffer(SEISMIC_BUFFER_FEATURES)]],                   \
    device const uchar *weight [[buffer(SEISMIC_BUFFER_WEIGHT)]],                       \
    device const int *token_rows [[buffer(SEISMIC_BUFFER_TOKEN_ROWS)]],                 \
    device const float *weight_scale [[buffer(SEISMIC_BUFFER_WEIGHT_SCALE)]],           \
    SEISMIC_RESULT_0_PARAM(logits_cells),                                               \
    device uchar *staged [[buffer(SEISMIC_BUFFER_SCRATCH_STAGED)]],                     \
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

// The logits epilogue of row 0 at `y`: scaled, then softcapped.
#define READOUT_LOGITS(y)                                                               \
    projection::scaling<(SEISMIC_DIM_WS != 0)>::wrap(                                   \
        projection::Logits{y, SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1,     \
            as_type<float>(uint(SEISMIC_PARAM_SOFTCAP))},                               \
        projection::scale_factor(weight_scale, SEISMIC_DIM_WS, 0, 0), 1.0f)

// `raw` is the feature rows; `in` the rows every form past the GEMV reads;
// the weight's first S rows.
#define PROJECTION_OPERANDS(group)                                                      \
    const uint m = uint(SEISMIC_DIM_O), n = uint(SEISMIC_DIM_S), k = uint(SEISMIC_DIM_D); \
    const projection::Plain<activation, projection::AllRows> raw{features, SEISMIC_FEATURES_STRIDE_0, \
        SEISMIC_FEATURES_STRIDE_1, k, {}};                                              \
    const projection::Plain<activation, projection::AllRows> in = NVFP4                 \
        ? projection::Plain<activation, projection::AllRows>{staged, k, 1, k, {}}       \
        : raw;                                                                          \
    const auto out = READOUT_LOGITS(element::Out<element::F32>(logits_cells));         \
    projection::Weights<packets::W0> w{weight, KERNEL_W0_LAYOUT(k), k}

#define PROJECTION_GROUPS 1
#define PROJECTION_GEMV_INPUT(x) const auto x = raw
#define PROJECTION_INT8_SCRATCH \
    const projection::int8_scratch scratch{quantized, row_scales, block_sums, coefficients, biases}
#define PROJECTION_PACKING_SCRATCH \
    const projection::packing_scratch scratch{packed, token_factors, token_scales, factors, scales}
#define PROJECTION_BOUNDS                                                               \
    constexpr uint BOUNDS = 0;                                                          \
    const projection::packing_bounds<BOUNDS> bounds{{}}

// The feature rows as NVFP4 operands into `staged`, one row a threadgroup.
#ifdef SEISMIC_FORMING_READOUT_PROJECT_ROWS_STAGE
kernel void readout_project_rows_stage(PROJECTION_ARGUMENTS,
    uint item [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    PROJECTION_NORMALIZE_SHARED(maxima);
    PROJECTION_OPERANDS(0u);
    projection::operands<true>::form<256>(raw, item, staged, k, maxima, thread_index);
}
#endif

// The routed projection's `choice` over row m's selected rows (choice k of
// I): threadgroup x a tile of SIMDGROUPS x 32 / LANES lane groups of ROWS
// selected rows, the K1 GEMV of the feature row against them. A selection
// past the vocabulary reads row 0 (`PROJECTION_TABLE_ROWS`), and once the tile's
// logits are stored its logit is -inf.
#ifdef SEISMIC_FORMING_READOUT_PROJECT_ROWS_GATHER_CHOICE
template <uint GATHER_ROWS, uint GATHER_LANES>
kernel void readout_project_rows_gather_choice(PROJECTION_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint s = uint(SEISMIC_DIM_S), d = uint(SEISMIC_DIM_D);
    const uint tile = group.x, span = simdgroups * GATHER_ROWS * (32u / GATHER_LANES);
    if (tile * span >= s)
        return;
    const ulong m = group.y / SEISMIC_DIM_I, choice = group.y % SEISMIC_DIM_I;
    device const int *selected = token_rows + m * SEISMIC_TOKEN_ROWS_STRIDE_0 + choice * SEISMIC_TOKEN_ROWS_STRIDE_1;
    const projection::Plain<activation, projection::AllRows> in{
        features + m * SEISMIC_FEATURES_STRIDE_0 * activation::bytes,
        SEISMIC_FEATURES_STRIDE_0, SEISMIC_FEATURES_STRIDE_1, d, {}};
    projection::Weights<packets::W0> w{weight, KERNEL_W0_LAYOUT(d), d, selected, 0,
        SEISMIC_TOKEN_ROWS_STRIDE_2};
    const element::Out<element::F32> row = element::Out<element::F32>(logits_cells).offset(m * SEISMIC_RESULT_0_STRIDE_0);
    const auto out = READOUT_LOGITS(row);
    PROJECTION_OPERANDS_SHARED(globals, NVFP4);
    const auto x = Operands::rows(in, 1u, d, globals, simdgroups, sg, lane);
    projection::gemv_runtime<packets::W0, GATHER_ROWS, 1, GATHER_LANES>(
        x, out, w, 1u, s, d, tile, shared, simdgroups, sg, lane);
    threadgroup_barrier(mem_flags::mem_device);
    for (uint n = tile * span + sg * 32u + lane; n < min(s, (tile + 1u) * span); n += simdgroups * 32u)
        if (ulong(selected[ulong(n) * SEISMIC_TOKEN_ROWS_STRIDE_2]) >= SEISMIC_DIM_V)
            row.put(ulong(n) * SEISMIC_RESULT_0_STRIDE_1, -INFINITY);
}
#endif

// The `dense` projection's launches (`lib/projection/operator.h`).
#ifdef SEISMIC_FORMING_READOUT_PROJECT_ROWS_GEMV
template <uint LANES, uint TILED, uint STAGE, uint ROWS>
kernel void readout_project_rows_gemv(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_SINGLE_GEMV(ONE, TILED);
}
#endif
#ifdef SEISMIC_FORMING_READOUT_PROJECT_ROWS_GEMV_PAIR
template <uint LANES, uint TILED, uint STAGE, uint PACKED_ROWS, uint ROWS>
kernel void readout_project_rows_gemv_pair(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_SINGLE_GEMV(PAIR, TILED);
}
#endif
#ifdef SEISMIC_FORMING_READOUT_PROJECT_ROWS_GEMV_ROWS
template <uint LANES, uint TILED, uint STAGE, uint PACKED_ROWS, uint ROWS>
kernel void readout_project_rows_gemv_rows(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_SINGLE_GEMV(SEVERAL, TILED);
}
#endif
#ifdef SEISMIC_FORMING_READOUT_PROJECT_ROWS_GEMV_PACKED
template <uint LANES, uint PACKED_ROWS, uint ROWS>
kernel void readout_project_rows_gemv_packed(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_SINGLE_GEMV_PACKED;
}
#endif
#ifdef SEISMIC_FORMING_READOUT_PROJECT_ROWS_BATCH
template <uint MATRIX, uint BATCH_ROWS, uint BATCH_PARTS>
kernel void readout_project_rows_batch(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_SINGLE_BATCH;
}
#endif
#ifdef SEISMIC_FORMING_READOUT_PROJECT_ROWS_GEMM_SMALL
kernel void readout_project_rows_gemm_small(PROJECTION_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_SINGLE_GEMM_SMALL;
}
#endif
#ifdef SEISMIC_FORMING_READOUT_PROJECT_ROWS_REDUCE_SMALL
kernel void readout_project_rows_reduce_small(PROJECTION_ARGUMENTS, PROJECTION_ITEM_ARGUMENTS) {
    PROJECTION_SINGLE_REDUCE_SMALL;
}
#endif
#ifdef SEISMIC_FORMING_READOUT_PROJECT_ROWS_GEMM
template <uint TILE_M, uint TILE_N>
kernel void readout_project_rows_gemm(PROJECTION_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_SINGLE_GEMM(uint(SEISMIC_RUNTIME_SPLIT));
}
#endif
#ifdef SEISMIC_FORMING_READOUT_PROJECT_ROWS_REDUCE
kernel void readout_project_rows_reduce(PROJECTION_ARGUMENTS, PROJECTION_ITEM_ARGUMENTS) {
    PROJECTION_SINGLE_REDUCE(uint(SEISMIC_RUNTIME_SPLIT));
}
#endif
#ifdef SEISMIC_FORMING_READOUT_PROJECT_ROWS_RELAYOUT
kernel void readout_project_rows_relayout(PROJECTION_ARGUMENTS, PROJECTION_ITEM_ARGUMENTS) {
    PROJECTION_SINGLE_RELAYOUT;
}
#endif
#ifdef SEISMIC_FORMING_READOUT_PROJECT_ROWS_TALL
template <uint TALL_M, uint TALL_K, uint STAGERS>
kernel void readout_project_rows_tall(PROJECTION_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_SINGLE_TALL;
}
#endif
#ifdef SEISMIC_FORMING_READOUT_PROJECT_ROWS_QUANTIZE
kernel void readout_project_rows_quantize(PROJECTION_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    PROJECTION_SINGLE_QUANTIZE;
}
#endif
#ifdef SEISMIC_FORMING_READOUT_PROJECT_ROWS_COEFFICIENTS
kernel void readout_project_rows_coefficients(PROJECTION_ARGUMENTS, PROJECTION_ITEM_ARGUMENTS) {
    PROJECTION_SINGLE_COEFFICIENTS;
}
#endif
#ifdef SEISMIC_FORMING_READOUT_PROJECT_ROWS_INT8
template <uint INT8_TILE>
kernel void readout_project_rows_int8(PROJECTION_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_SINGLE_INT8;
}
#endif
#ifdef SEISMIC_FORMING_READOUT_PROJECT_ROWS_PACK
template <uint RESIDUE>
kernel void readout_project_rows_pack(PROJECTION_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    PROJECTION_SINGLE_PACK;
}
#endif
#ifdef SEISMIC_FORMING_READOUT_PROJECT_ROWS_PACK_COEFFICIENTS
kernel void readout_project_rows_pack_coefficients(PROJECTION_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    PROJECTION_SINGLE_PACK_COEFFICIENTS;
}
#endif
#ifdef SEISMIC_FORMING_READOUT_PROJECT_ROWS_PACKED
template <uint RESIDUE, uint PACK_TOKENS, uint WEIGHTS_AHEAD>
kernel void readout_project_rows_packed(PROJECTION_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    PROJECTION_SINGLE_PACKED;
}
#endif
#ifdef SEISMIC_FORMING_READOUT_PROJECT_ROWS_UNPACKED
kernel void readout_project_rows_unpacked(PROJECTION_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_SINGLE_UNPACKED;
}
#endif
