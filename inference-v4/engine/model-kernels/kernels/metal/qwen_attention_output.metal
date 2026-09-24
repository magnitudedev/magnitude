// qwen_attention_output: the output projection of the gated attention rows
// plus the F32 residual.
#include "common/projection.h"

#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_BF16)
typedef packets::bf16 activation;
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_F16)
typedef packets::f16 activation;
#else
#error "qwen_attention_output requires a bf16 or f16 activation"
#endif
#if defined(SEISMIC_OUTPUT_WEIGHT_REPRESENTATION_Q4K)
typedef packets::q4k output_packet;
#define OUTPUT_LAYOUT {SEISMIC_OUTPUT_WEIGHT_ROW_STRIDE_BYTES, SEISMIC_OUTPUT_WEIGHT_PLANE_CODES_LO_ROW_OFFSET, 0, SEISMIC_OUTPUT_WEIGHT_PLANE_SCALES_ROW_OFFSET, SEISMIC_OUTPUT_WEIGHT_PLANE_SUPERS_ROW_OFFSET}
#elif defined(SEISMIC_OUTPUT_WEIGHT_REPRESENTATION_Q5K)
typedef packets::q5k output_packet;
#define OUTPUT_LAYOUT {SEISMIC_OUTPUT_WEIGHT_ROW_STRIDE_BYTES, SEISMIC_OUTPUT_WEIGHT_PLANE_CODES_LO_ROW_OFFSET, SEISMIC_OUTPUT_WEIGHT_PLANE_CODES_HI_ROW_OFFSET, SEISMIC_OUTPUT_WEIGHT_PLANE_SCALES_ROW_OFFSET, SEISMIC_OUTPUT_WEIGHT_PLANE_SUPERS_ROW_OFFSET}
#elif defined(SEISMIC_OUTPUT_WEIGHT_REPRESENTATION_Q6K)
typedef packets::q6k output_packet;
#define OUTPUT_LAYOUT {SEISMIC_OUTPUT_WEIGHT_ROW_STRIDE_BYTES, SEISMIC_OUTPUT_WEIGHT_PLANE_CODES_LO_ROW_OFFSET, SEISMIC_OUTPUT_WEIGHT_PLANE_CODES_HI_ROW_OFFSET, SEISMIC_OUTPUT_WEIGHT_PLANE_SCALES_ROW_OFFSET, SEISMIC_OUTPUT_WEIGHT_PLANE_SUPERS_ROW_OFFSET}
#elif defined(SEISMIC_OUTPUT_WEIGHT_REPRESENTATION_Q8G32S)
typedef packets::q8 output_packet;
#define OUTPUT_LAYOUT {SEISMIC_OUTPUT_WEIGHT_ROW_STRIDE_BYTES, SEISMIC_OUTPUT_WEIGHT_PLANE_CODES_ROW_OFFSET, 0, 0, SEISMIC_OUTPUT_WEIGHT_PLANE_SUPERS_ROW_OFFSET}
#elif defined(SEISMIC_OUTPUT_WEIGHT_REPRESENTATION_BF16)
typedef packets::dense<packets::bf16> output_packet;
#define OUTPUT_LAYOUT {SEISMIC_OUTPUT_WEIGHT_STRIDE_0 * 2, 0, 0, 0, 0}
#elif defined(SEISMIC_OUTPUT_WEIGHT_REPRESENTATION_F16)
typedef packets::dense<packets::f16> output_packet;
#define OUTPUT_LAYOUT {SEISMIC_OUTPUT_WEIGHT_STRIDE_0 * 2, 0, 0, 0, 0}
#elif defined(SEISMIC_OUTPUT_WEIGHT_REPRESENTATION_F32)
typedef packets::dense<packets::f32> output_packet;
#define OUTPUT_LAYOUT {SEISMIC_OUTPUT_WEIGHT_STRIDE_0 * 4, 0, 0, 0, 0}
#else
#error "unsupported qwen_attention_output output_weight representation"
#endif
#if defined(SEISMIC_OUTPUT_WEIGHT_KIND_PACKED) && !defined(SEISMIC_OUTPUT_WEIGHT_LAYOUT_ROWS16)
#error "qwen_attention_output requires the rows16 layout for output_weight"
#endif

#define ATTENTION_OUTPUT_ARGUMENTS                                                      \
    device const float *hidden [[buffer(SEISMIC_BUFFER_HIDDEN)]],                       \
    device const uchar *gated [[buffer(SEISMIC_BUFFER_GATED)]],                         \
    device const uchar *output_weight [[buffer(SEISMIC_BUFFER_OUTPUT_WEIGHT)]],         \
    device float *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],                           \
    device float *partials [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIALS)]],                 \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

// `gated` is [M, Q, W] and canonical, so a row is Q*W contiguous values.
#define ATTENTION_OUTPUT_OPERANDS                                                       \
    const uint k = uint(SEISMIC_DIM_Q * SEISMIC_DIM_W);                                                \
    projection::input_plain<activation> in{gated, SEISMIC_GATED_STRIDE_0,              \
        SEISMIC_GATED_STRIDE_2, k, {nullptr}};                                          \
    projection::output_residual<activation> out{result, SEISMIC_RESULT_0_STRIDE_0,      \
        SEISMIC_RESULT_0_STRIDE_1, hidden, SEISMIC_HIDDEN_STRIDE_0,                     \
        SEISMIC_HIDDEN_STRIDE_1, {nullptr}};                                            \
    projection::weight_rows<output_packet> w{output_weight, OUTPUT_LAYOUT, k, nullptr}

kernel void qwen_attention_output_gemv(ATTENTION_OUTPUT_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    ATTENTION_OUTPUT_OPERANDS;
    uint rows = uint(SEISMIC_DIM_M);
    PROJECTION_FOR_ROWS(rows,
        projection::gemv<output_packet, SEISMIC_TUNE_SIMDGROUPS, SEISMIC_TUNE_ROWS, MAXM, SEISMIC_TUNE_LANES>(
            in, out, w, rows, uint(SEISMIC_DIM_D), k, tile, shared, sg, lane));
}

kernel void qwen_attention_output_batch(ATTENTION_OUTPUT_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    ATTENTION_OUTPUT_OPERANDS;
    projection::gemv_batch<output_packet, SEISMIC_TUNE_BATCH_SIMDGROUPS, SEISMIC_TUNE_BATCH_ROWS>(in, out, w,
        uint(SEISMIC_DIM_M), uint(SEISMIC_DIM_D), k, tile, shared, sg, lane);
}

kernel void qwen_attention_output_gemm(ATTENTION_OUTPUT_ARGUMENTS,
    uint3 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint TM = SEISMIC_TUNE_TILE_M, TN = SEISMIC_TUNE_TILE_N;
    PROJECTION_GEMM_SHARED(shared, TM, TN);
    ATTENTION_OUTPUT_OPERANDS;
    const uint m = uint(SEISMIC_DIM_M), n = uint(SEISMIC_DIM_D);
    if (SEISMIC_TUNE_SPLIT == 1)
        projection::gemm<output_packet, TM, TN>(in, out, w, m, n, k, tile.y, tile.x, shared, sg, lane);
    else
        projection::gemm_part<output_packet, TM, TN>(in, partials, w, m, n, k, SEISMIC_TUNE_SPLIT, tile.z,
            tile.y, tile.x, shared, sg, lane);
}

kernel void qwen_attention_output_reduce(ATTENTION_OUTPUT_ARGUMENTS,
    uint index [[thread_position_in_grid]]) {
    ATTENTION_OUTPUT_OPERANDS;
    projection::gemm_reduce(out, partials, uint(SEISMIC_DIM_M), uint(SEISMIC_DIM_D), SEISMIC_TUNE_SPLIT, index);
}
