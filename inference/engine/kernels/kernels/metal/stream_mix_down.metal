// stream_mix_down: the per-stream RMS prologue over the S hyper-connection
// streams, then the `dense` projection (segmented, K in S parts, the
// streams) down | inject (an inject segment of zero rows, the exit site's,
// has no tiles), and the streams' inverses. Below BATCH_FROM rows the parts
// GEMV normalizes only its stream (its first row tile publishing the
// stream's inverses); the batched rows reduce every stream's square sums and
// normalize while staging (the first threadgroup publishing the inverses);
// past 16 rows `stage` normalizes each row's streams into `normalized` with
// their inverses, which the GEMM reads.
#define KERNEL_A SEISMIC_ELEMENT_A
#define KERNEL_W0 SEISMIC_DOWN_WEIGHT
#define KERNEL_W1 SEISMIC_INJECT_WEIGHT
#include "lib/core/arrive.h"
#include "lib/projection/projection.h"
#include "lib/projection/packing.h"
#if defined(SEISMIC_FORMING_STREAM_MIX_DOWN_GEMV) || defined(SEISMIC_FORMING_STREAM_MIX_DOWN_GEMV_ROWS) || \
    defined(SEISMIC_FORMING_STREAM_MIX_DOWN_GEMV_PAIR)
#include "lib/projection/gemv.h"
#endif

typedef activation::Act activation_element;
typedef ELEMENT_OF(SEISMIC_NORM) norm_element;
// The rows every weight takes: the normalized streams (no NVFP4 operands,
// `stream_mix_down`'s contract).
constant constexpr bool NVFP4 = projection::OperandRecipe<packets::W0_OPERANDS_NVFP4,
    packets::W1_OPERANDS_NVFP4>::NVFP4;
typedef projection::operands<NVFP4> Operands;

#define PROJECTION_ARGUMENTS                                                            \
    device const float *hidden [[buffer(SEISMIC_BUFFER_HIDDEN)]],                       \
    device const uchar *norm [[buffer(SEISMIC_BUFFER_NORM)]],                           \
    device const uchar *down_weight [[buffer(SEISMIC_BUFFER_DOWN_WEIGHT)]],             \
    device const uchar *inject_weight [[buffer(SEISMIC_BUFFER_INJECT_WEIGHT)]],         \
    SEISMIC_RESULT_0_PARAM(low_cells),                                                  \
    SEISMIC_RESULT_1_PARAM(inject_cells),                                               \
    SEISMIC_RESULT_2_PARAM(inverse_cells),                                              \
    device uchar *normalized [[buffer(SEISMIC_BUFFER_SCRATCH_NORMALIZED)]],             \
    arrive::coherent_device<float> partials [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIALS)]], \
    device atomic_uint *arrivals [[buffer(SEISMIC_BUFFER_SCRATCH_ARRIVALS)]],           \
    device float *packed [[buffer(SEISMIC_BUFFER_SCRATCH_PACKED)]],                     \
    device half *token_factors [[buffer(SEISMIC_BUFFER_SCRATCH_TOKEN_FACTORS)]],        \
    device half *factors [[buffer(SEISMIC_BUFFER_SCRATCH_FACTORS)]],                    \
    device float *token_scales [[buffer(SEISMIC_BUFFER_SCRATCH_TOKEN_SCALES)]],         \
    device float *scales [[buffer(SEISMIC_BUFFER_SCRATCH_SCALES)]],                     \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

// The segments down | inject; K is S parts, the streams.
#define PROJECTION_N0 SEISMIC_DIM_LR
#define PROJECTION_N1 (SEISMIC_DIM_C * SEISMIC_DIM_S)
#define PROJECTION_PARTS SEISMIC_DIM_S

// `raw` is the streams under their per-stream norm; `in` the normalized
// streams the GEMMs read.
#define PROJECTION_OPERANDS                                                             \
    const uint m = uint(SEISMIC_DIM_M), k = uint(SEISMIC_DIM_S * SEISMIC_DIM_H);        \
    projection::GroupRms<activation_element, norm_element, projection::AllRows> raw{hidden,     \
        SEISMIC_HIDDEN_STRIDE_0, SEISMIC_HIDDEN_STRIDE_2, norm, SEISMIC_NORM_STRIDE_1,  \
        as_type<float>(uint(SEISMIC_PARAM_EPSILON)), k, {}, uint(SEISMIC_DIM_H)};       \
    const projection::Plain<activation_element, projection::AllRows> in{normalized, k, 1, k, {}}; \
    projection::Weights<packets::W0> w0{{down_weight, KERNEL_W0_LAYOUT(k), k}};         \
    projection::Weights<packets::W1> w1{{inject_weight, KERNEL_W1_LAYOUT(k), k}};       \
    projection::MixLow<activation_element> out0{element::Out<activation_element>(low_cells), SEISMIC_RESULT_0_STRIDE_0, \
        SEISMIC_RESULT_0_STRIDE_1, float(SEISMIC_DIM_S)};                               \
    projection::MixInject out1{element::Out<element::F32>(inject_cells), SEISMIC_RESULT_1_STRIDE_0, \
        SEISMIC_RESULT_1_STRIDE_1, float(SEISMIC_DIM_S)};                               \
    const element::Out<element::F32> inverse(inverse_cells)

// Stream `part`'s rows, normalized from its own square sums over the
// threadgroup; its first row tile publishes its inverses.
#define PROJECTION_PART_INPUT(rows_var, part, first)                                    \
    const ulong part_first = ulong(part) * uint(SEISMIC_DIM_H);                         \
    const projection::GroupRms<activation_element, norm_element, projection::AllRows> stream{   \
        hidden + part_first * SEISMIC_HIDDEN_STRIDE_2, SEISMIC_HIDDEN_STRIDE_0,         \
        SEISMIC_HIDDEN_STRIDE_2, norm + part_first * SEISMIC_NORM_STRIDE_1 * norm_element::bytes, \
        SEISMIC_NORM_STRIDE_1, as_type<float>(uint(SEISMIC_PARAM_EPSILON)), width, {}, width}; \
    PROJECTION_SQUARES_SHARED(squares, decltype(stream)::parts);                        \
    projection::threadgroup_squares_runtime(stream, m, squares, simdgroups, sg, lane);  \
    if (first) {                                                                        \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                \
        projection::publish_inverses(stream, m, squares, inverse.offset(ulong(part) * SEISMIC_RESULT_2_STRIDE_1), \
            SEISMIC_RESULT_2_STRIDE_0, SEISMIC_RESULT_2_STRIDE_1, sg * 32 + lane, simdgroups * 32); \
    }                                                                                   \
    const auto rows_var = projection::shared_norm(stream, squares)
#define PROJECTION_GEMV_INPUT(rows_var)
#define PROJECTION_FRAGMENTS normalized
#define PROJECTION_ZEROED(s) false
#define PROJECTION_BOUNDED 0u
#define PROJECTION_BOUND_TABLES {nullptr, nullptr}
#define PROJECTION_COLUMNS(s) false
#define PROJECTION_GEMV_PROLOGUE(s)
#define PROJECTION_BATCH_PROLOGUE(s)
#define PROJECTION_GEMV_EPILOGUE(s, first, count)

#include "lib/projection/operator.h"

// Past 16 rows: one threadgroup per row and stream normalizes it into
// `normalized` (in row order, or in the tall GEMM's order past 64 rows)
// and publishes its inverse.
#ifdef SEISMIC_FORMING_STREAM_MIX_DOWN_STAGE
kernel void stream_mix_down_stage(PROJECTION_ARGUMENTS,
    uint item [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    PROJECTION_NORMALIZE_SHARED(norms);
    PROJECTION_OPERANDS;
    const float value = projection::device_normalize<256>(raw, item, normalized, k, norms, thread_index);
    if (thread_index == 0)
        inverse.put(ulong(item / uint(SEISMIC_DIM_S)) * SEISMIC_RESULT_2_STRIDE_0
            + ulong(item % uint(SEISMIC_DIM_S)) * SEISMIC_RESULT_2_STRIDE_1, value);
}
#endif
#ifdef SEISMIC_FORMING_STREAM_MIX_DOWN_STAGE_TALL
kernel void stream_mix_down_stage_tall(PROJECTION_ARGUMENTS,
    uint item [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    PROJECTION_NORMALIZE_SHARED(norms);
    PROJECTION_OPERANDS;
    const float value = projection::device_normalize<256, projection::TallOrder<activation_element>>(
        raw, item, normalized, k, norms, thread_index);
    if (thread_index == 0)
        inverse.put(ulong(item / uint(SEISMIC_DIM_S)) * SEISMIC_RESULT_2_STRIDE_0
            + ulong(item % uint(SEISMIC_DIM_S)) * SEISMIC_RESULT_2_STRIDE_1, value);
}
#endif

// The `dense` projection's launches (`lib/projection/operator.h`); every
// GEMV row class is the parts GEMV.
#ifdef SEISMIC_FORMING_STREAM_MIX_DOWN_GEMV
template <uint MATRIX_FROM, uint LANES, uint TILED, uint STAGE, uint ROWS>
kernel void stream_mix_down_gemv(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_SEGMENTED_PARTS_GEMV(ONE);
}
#endif
#ifdef SEISMIC_FORMING_STREAM_MIX_DOWN_BATCH
template <uint MATRIX_FROM, uint BATCH_ROWS, uint BATCH_PARTS>
kernel void stream_mix_down_batch(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_SEGMENTED_PARTS_BATCH;
}
#endif
#ifdef SEISMIC_FORMING_STREAM_MIX_DOWN_GEMM_SMALL
kernel void stream_mix_down_gemm_small(PROJECTION_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_SEGMENTED_GEMM_SMALL;
}
#endif
#ifdef SEISMIC_FORMING_STREAM_MIX_DOWN_GEMM
template <uint TILE_M, uint TILE_N>
kernel void stream_mix_down_gemm(PROJECTION_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_SEGMENTED_GEMM;
}
#endif
#ifdef SEISMIC_FORMING_STREAM_MIX_DOWN_TALL
template <uint TALL_M, uint TALL_K, uint STAGERS>
kernel void stream_mix_down_tall(PROJECTION_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_SEGMENTED_TALL;
}
#endif
#ifdef SEISMIC_FORMING_STREAM_MIX_DOWN_GEMV_ROWS
template <uint MATRIX_FROM, uint LANES, uint TILED, uint STAGE, uint ROWS>
kernel void stream_mix_down_gemv_rows(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_SEGMENTED_PARTS_GEMV(SEVERAL);
}
#endif
#ifdef SEISMIC_FORMING_STREAM_MIX_DOWN_GEMV_PAIR
template <uint MATRIX_FROM, uint LANES, uint TILED, uint STAGE, uint ROWS>
kernel void stream_mix_down_gemv_pair(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_SEGMENTED_PARTS_GEMV(PAIR);
}
#endif
#ifdef SEISMIC_FORMING_STREAM_MIX_DOWN_PACK
template <uint RESIDUE>
kernel void stream_mix_down_pack(PROJECTION_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    PROJECTION_SEGMENTED_PACK;
}
#endif
#ifdef SEISMIC_FORMING_STREAM_MIX_DOWN_PACK_COEFFICIENTS
kernel void stream_mix_down_pack_coefficients(PROJECTION_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    PROJECTION_SEGMENTED_PACK_COEFFICIENTS;
}
#endif
#ifdef SEISMIC_FORMING_STREAM_MIX_DOWN_PACKED
template <uint RESIDUE, uint PACK_TOKENS, uint WEIGHTS_AHEAD>
kernel void stream_mix_down_packed(PROJECTION_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    PROJECTION_SEGMENTED_PACKED;
}
#endif
#ifdef SEISMIC_FORMING_STREAM_MIX_DOWN_UNPACKED
kernel void stream_mix_down_unpacked(PROJECTION_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_SEGMENTED_UNPACKED;
}
#endif
