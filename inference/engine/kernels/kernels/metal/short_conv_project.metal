// short_conv_project: RMS prologue over every residual row, then the `dense`
// projection (segmented): the u segment (columns 0..CH) pairs the B and X
// streams into u = B * X (F32), the C segment (columns CH..2CH) projects
// one stream (F32). The GEMV row classes, and the batched rows, normalize
// while staging; `normalize` normalizes once into `normalized` for the matrix
// GEMV and every form past 16 rows, and `normalize_tall` in the tall GEMM's
// order.
#define KERNEL_W0 SEISMIC_B_WEIGHT
#define KERNEL_W1 SEISMIC_X_WEIGHT
#define KERNEL_W2 SEISMIC_C_WEIGHT
// Segment 0 is the pair (B, X), segment 1 the C stream (weight W2): stated
// before the operator, which takes them as the segments' weights.
#define PROJECTION_N0 SEISMIC_DIM_CH
#define PROJECTION_N1 SEISMIC_DIM_CH
#define PROJECTION_PARTNER_0 x_rows
#define PROJECTION_PARTNER_PACKET_0 packets::W1
#define PROJECTION_PACKET_1 packets::W2
#include "lib/projection/projection.h"
#include "lib/projection/packing.h"
#include "lib/projection/operator.h"
#if defined(SEISMIC_FORMING_SHORT_CONV_PROJECT_GEMV) || defined(SEISMIC_FORMING_SHORT_CONV_PROJECT_GEMV_ROWS) || defined(SEISMIC_FORMING_SHORT_CONV_PROJECT_GEMV_PAIR)
#include "lib/projection/gemv.h"
#endif

typedef element::Act activation;
typedef ELEMENT_OF(SEISMIC_NORM) norm_element;
// The rows the three weights take (the normalized rows, or NVFP4 operands).
constant constexpr bool NVFP4 = projection::OperandRecipe<packets::W0_OPERANDS_NVFP4, packets::W1_OPERANDS_NVFP4,
    packets::W2_OPERANDS_NVFP4>::NVFP4;
typedef projection::operands<NVFP4> Operands;

#define PROJECTION_ARGUMENTS                                                            \
    device const float *residual [[buffer(SEISMIC_BUFFER_RESIDUAL)]],                   \
    device const uchar *norm [[buffer(SEISMIC_BUFFER_NORM)]],                           \
    device const uchar *b_weight [[buffer(SEISMIC_BUFFER_B_WEIGHT)]],                   \
    device const uchar *c_weight [[buffer(SEISMIC_BUFFER_C_WEIGHT)]],                   \
    device const uchar *x_weight [[buffer(SEISMIC_BUFFER_X_WEIGHT)]],                   \
    SEISMIC_RESULT_0_PARAM(result_cells),                                               \
    device uchar *normalized [[buffer(SEISMIC_BUFFER_SCRATCH_NORMALIZED)]],             \
    device float *packed [[buffer(SEISMIC_BUFFER_SCRATCH_PACKED)]],                     \
    device half *token_factors [[buffer(SEISMIC_BUFFER_SCRATCH_TOKEN_FACTORS)]],        \
    device half *factors [[buffer(SEISMIC_BUFFER_SCRATCH_FACTORS)]],                    \
    device float *token_scales [[buffer(SEISMIC_BUFFER_SCRATCH_TOKEN_SCALES)]],         \
    device float *scales [[buffer(SEISMIC_BUFFER_SCRATCH_SCALES)]],                     \
    device const float *b_scale [[buffer(SEISMIC_BUFFER_B_SCALE)]],                     \
    device const float *c_scale [[buffer(SEISMIC_BUFFER_C_SCALE)]],                     \
    device const float *x_scale [[buffer(SEISMIC_BUFFER_X_SCALE)]],                     \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

// `raw` is the residual rows under the norm prologue; `in` the normalized
// rows every form past the GEMV reads.
#define PROJECTION_OPERANDS                                                             \
    const uint m = uint(SEISMIC_DIM_M), k = uint(SEISMIC_DIM_H), channels = uint(SEISMIC_DIM_CH); \
    projection::Rms<activation, norm_element, projection::AllRows> raw{residual,        \
        SEISMIC_RESIDUAL_STRIDE_0, SEISMIC_RESIDUAL_STRIDE_1, norm, SEISMIC_NORM_STRIDE_0, \
        as_type<float>(uint(SEISMIC_PARAM_EPS)), k, {}};                                \
    const projection::Plain<activation, projection::AllRows> in{normalized, k, 1, k, {}}; \
    const auto out0 = projection::scaling<(SEISMIC_DIM_BS != 0 || SEISMIC_DIM_XS != 0)>::wrap( \
        projection::Mul{element::Out<element::F32>(result_cells), SEISMIC_RESULT_0_STRIDE_0, \
            SEISMIC_RESULT_0_STRIDE_1},                                                 \
        projection::scale_factor(b_scale, SEISMIC_DIM_BS, 0, 0),                        \
        projection::scale_factor(x_scale, SEISMIC_DIM_XS, 0, 0));                       \
    const auto out1 = projection::scaling<(SEISMIC_DIM_CS != 0)>::wrap(                 \
        projection::Store<element::F32>{element::Out<element::F32>(result_cells), SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1, channels}, \
        projection::scale_factor(c_scale, SEISMIC_DIM_CS, 0, 0), 1.0f);                 \
    projection::Weights<packets::W0> w0{b_weight, KERNEL_W0_LAYOUT(k), k};              \
    projection::Weights<packets::W1> x_rows{x_weight, KERNEL_W1_LAYOUT(k), k};          \
    projection::Weights<packets::W2> w1{c_weight, KERNEL_W2_LAYOUT(k), k}

#define PROJECTION_GEMV_INPUT(x)                                                        \
    PROJECTION_SQUARES_SHARED(squares, decltype(raw)::parts);                           \
    projection::threadgroup_squares_runtime(raw, m, squares, simdgroups, sg, lane);     \
    const auto x = projection::shared_norm(raw, squares)
#define PROJECTION_FRAGMENTS normalized
#define PROJECTION_ZEROED(s) false
#define PROJECTION_BOUNDED 0u
#define PROJECTION_BOUND_TABLES {}
#define PROJECTION_COLUMNS(s) false
#define PROJECTION_GEMV_PROLOGUE(s)
#define PROJECTION_BATCH_PROLOGUE(s)
#define PROJECTION_GEMV_EPILOGUE(s, first, count)

#ifdef SEISMIC_FORMING_SHORT_CONV_PROJECT_NORMALIZE
template <uint MATRIX>
kernel void short_conv_project_normalize(PROJECTION_ARGUMENTS,
    uint item [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    PROJECTION_NORMALIZE_SHARED(norms);
    PROJECTION_OPERANDS;
    Operands::normalize<256>(raw, item, normalized, k, norms, thread_index);
}
#endif
#ifdef SEISMIC_FORMING_SHORT_CONV_PROJECT_NORMALIZE_TALL
kernel void short_conv_project_normalize_tall(PROJECTION_ARGUMENTS,
    uint item [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    PROJECTION_NORMALIZE_SHARED(norms);
    PROJECTION_OPERANDS;
    Operands::normalize<256, projection::TallOrder<activation>>(raw, item, normalized, k, norms, thread_index);
}
#endif

// The `dense` projection's launches (`lib/projection/operator.h`).
#ifdef SEISMIC_FORMING_SHORT_CONV_PROJECT_GEMV
template <uint LANES, uint TILED, uint STAGE, uint ROWS>
kernel void short_conv_project_gemv(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_SEGMENTED_GEMV(ONE, TILED);
}
#endif
#ifdef SEISMIC_FORMING_SHORT_CONV_PROJECT_BATCH
template <uint MATRIX, uint BATCH_ROWS, uint BATCH_PARTS>
kernel void short_conv_project_batch(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_SEGMENTED_BATCH;
}
#endif
#ifdef SEISMIC_FORMING_SHORT_CONV_PROJECT_GEMM_SMALL
kernel void short_conv_project_gemm_small(PROJECTION_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_SEGMENTED_GEMM_SMALL;
}
#endif
#ifdef SEISMIC_FORMING_SHORT_CONV_PROJECT_GEMM
template <uint TILE_M, uint TILE_N>
kernel void short_conv_project_gemm(PROJECTION_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_SEGMENTED_GEMM;
}
#endif
#ifdef SEISMIC_FORMING_SHORT_CONV_PROJECT_TALL
template <uint TALL_M, uint TALL_K, uint STAGERS>
kernel void short_conv_project_tall(PROJECTION_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_SEGMENTED_TALL;
}
#endif
#ifdef SEISMIC_FORMING_SHORT_CONV_PROJECT_GEMV_ROWS
template <uint LANES, uint TILED, uint STAGE, uint ROWS>
kernel void short_conv_project_gemv_rows(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_SEGMENTED_GEMV(SEVERAL, TILED);
}
#endif
#ifdef SEISMIC_FORMING_SHORT_CONV_PROJECT_GEMV_PAIR
template <uint LANES, uint TILED, uint STAGE, uint ROWS>
kernel void short_conv_project_gemv_pair(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_SEGMENTED_GEMV(PAIR, TILED);
}
#endif
#ifdef SEISMIC_FORMING_SHORT_CONV_PROJECT_PACK
template <uint RESIDUE>
kernel void short_conv_project_pack(PROJECTION_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    PROJECTION_SEGMENTED_PACK;
}
#endif
#ifdef SEISMIC_FORMING_SHORT_CONV_PROJECT_PACK_COEFFICIENTS
kernel void short_conv_project_pack_coefficients(PROJECTION_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    PROJECTION_SEGMENTED_PACK_COEFFICIENTS;
}
#endif
#ifdef SEISMIC_FORMING_SHORT_CONV_PROJECT_PACKED
template <uint RESIDUE, uint PACK_TOKENS, uint WEIGHTS_AHEAD>
kernel void short_conv_project_packed(PROJECTION_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    PROJECTION_SEGMENTED_PACKED;
}
#endif
#ifdef SEISMIC_FORMING_SHORT_CONV_PROJECT_UNPACKED
kernel void short_conv_project_unpacked(PROJECTION_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_SEGMENTED_UNPACKED;
}
#endif
