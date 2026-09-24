// qwen_recurrent_output: gated per-head RMS·SiLU(z) prologue over the mixed
// recurrence output, the output projection, and the F32 residual.
#include "common/projection.h"

#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_BF16)
typedef packets::bf16 activation;
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_F16)
typedef packets::f16 activation;
#else
#error "qwen_recurrent_output requires a bf16 or f16 activation"
#endif
#if defined(SEISMIC_RECURRENT_NORM_REPRESENTATION_F32)
typedef packets::f32 norm_element;
#elif defined(SEISMIC_RECURRENT_NORM_REPRESENTATION_F16)
typedef packets::f16 norm_element;
#elif defined(SEISMIC_RECURRENT_NORM_REPRESENTATION_BF16)
typedef packets::bf16 norm_element;
#else
#error "qwen_recurrent_output requires a dense recurrent_norm"
#endif
// The GEMV and batched launches reduce each head's norm across the staging
// lanes that load it (projection::LaneNorm): W / 8 lanes, a power of two.
static_assert(SEISMIC_DIM_W % 8 == 0 && SEISMIC_DIM_W <= 256 && ((SEISMIC_DIM_W / 8) & (SEISMIC_DIM_W / 8 - 1)) == 0,
    "qwen_recurrent_output reduces a head over W / 8 lanes, a power of two up to 32");
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
#error "unsupported qwen_recurrent_output output_weight representation"
#endif
#if defined(SEISMIC_OUTPUT_WEIGHT_KIND_PACKED) && !defined(SEISMIC_OUTPUT_WEIGHT_LAYOUT_ROWS16)
#error "qwen_recurrent_output requires the rows16 layout for output_weight"
#endif

#define RECURRENT_OUTPUT_ARGUMENTS                                                      \
    device const float *hidden [[buffer(SEISMIC_BUFFER_HIDDEN)]],                       \
    device const uchar *projected [[buffer(SEISMIC_BUFFER_PROJECTION)]],                \
    device const uchar *mixed [[buffer(SEISMIC_BUFFER_MIXED)]],                         \
    device const uchar *recurrent_norm [[buffer(SEISMIC_BUFFER_RECURRENT_NORM)]],       \
    device const uchar *output_weight [[buffer(SEISMIC_BUFFER_OUTPUT_WEIGHT)]],         \
    device float *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],                           \
    device uchar *normalized [[buffer(SEISMIC_BUFFER_SCRATCH_NORMALIZED)]],             \
    device float *partials [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIALS)]],                 \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

#define RECURRENT_OUTPUT_OPERANDS                                                       \
    const uint k = uint(SEISMIC_DIM_NV * SEISMIC_DIM_W);                                \
    projection::input_gated<activation, norm_element> in{mixed, SEISMIC_MIXED_STRIDE_0, \
        SEISMIC_MIXED_STRIDE_1, SEISMIC_MIXED_STRIDE_2, projected,                      \
        SEISMIC_PROJECTION_STRIDE_0, SEISMIC_PROJECTION_STRIDE_1,                       \
        ulong((2 * SEISMIC_DIM_NK + SEISMIC_DIM_NV) * SEISMIC_DIM_W), recurrent_norm,   \
        SEISMIC_RECURRENT_NORM_STRIDE_0, as_type<float>(uint(SEISMIC_PARAM_EPSILON)),   \
        uint(SEISMIC_DIM_NV), uint(SEISMIC_DIM_W), {nullptr}};                          \
    projection::output_residual<activation> out{result, SEISMIC_RESULT_0_STRIDE_0,      \
        SEISMIC_RESULT_0_STRIDE_1, hidden, SEISMIC_HIDDEN_STRIDE_0,                     \
        SEISMIC_HIDDEN_STRIDE_1, {nullptr}};                                            \
    projection::weight_rows<output_packet> w{output_weight, OUTPUT_LAYOUT, k, nullptr}

kernel void qwen_recurrent_output_gemv(RECURRENT_OUTPUT_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    RECURRENT_OUTPUT_OPERANDS;
    uint rows = uint(SEISMIC_DIM_M);
    projection::LaneNorm<decltype(in)> x{in};
    PROJECTION_FOR_ROWS(rows,
        projection::gemv<output_packet, SEISMIC_TUNE_SIMDGROUPS, SEISMIC_TUNE_ROWS, MAXM, SEISMIC_TUNE_LANES>(
            x, out, w, rows, uint(SEISMIC_DIM_H), k, tile, shared, sg, lane));
}

kernel void qwen_recurrent_output_batch(RECURRENT_OUTPUT_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    RECURRENT_OUTPUT_OPERANDS;
    projection::LaneNorm<decltype(in)> x{in};
    projection::gemv_batch<output_packet, SEISMIC_TUNE_BATCH_SIMDGROUPS, SEISMIC_TUNE_BATCH_ROWS>(x, out, w,
        uint(SEISMIC_DIM_M), uint(SEISMIC_DIM_H), k, tile, shared, sg, lane);
}

kernel void qwen_recurrent_output_normalize(RECURRENT_OUTPUT_ARGUMENTS,
    uint item [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    PROJECTION_NORMALIZE_SHARED(norms);
    RECURRENT_OUTPUT_OPERANDS;
    projection::device_normalize<32>(in, item, normalized, k, norms, thread_index);
}

kernel void qwen_recurrent_output_gemm(RECURRENT_OUTPUT_ARGUMENTS,
    uint3 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint TM = SEISMIC_TUNE_TILE_M, TN = SEISMIC_TUNE_TILE_N;
    PROJECTION_GEMM_SHARED(shared, TM, TN);
    RECURRENT_OUTPUT_OPERANDS;
    projection::input_plain<activation> x{normalized, k, 1, k, {nullptr}};
    const uint m = uint(SEISMIC_DIM_M), n = uint(SEISMIC_DIM_H);
    if (SEISMIC_TUNE_SPLIT == 1)
        projection::gemm<output_packet, TM, TN>(x, out, w, m, n, k, tile.y, tile.x, shared, sg, lane);
    else
        projection::gemm_part<output_packet, TM, TN>(x, partials, w, m, n, k, SEISMIC_TUNE_SPLIT, tile.z,
            tile.y, tile.x, shared, sg, lane);
}

kernel void qwen_recurrent_output_reduce(RECURRENT_OUTPUT_ARGUMENTS,
    uint index [[thread_position_in_grid]]) {
    RECURRENT_OUTPUT_OPERANDS;
    projection::gemm_reduce(out, partials, uint(SEISMIC_DIM_M), uint(SEISMIC_DIM_H), SEISMIC_TUNE_SPLIT, index);
}
