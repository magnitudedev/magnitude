// qwen_selected_rows: final RMS prologue over the `out_rows` hidden rows and the
// projection onto the `selected` vocabulary rows, into F32 logits.
#include "common/projection.h"

#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_BF16)
typedef packets::bf16 activation;
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_F16)
typedef packets::f16 activation;
#else
#error "qwen_selected_rows requires a bf16 or f16 activation"
#endif
#if defined(SEISMIC_NORM_REPRESENTATION_F32)
typedef packets::f32 norm_element;
#elif defined(SEISMIC_NORM_REPRESENTATION_F16)
typedef packets::f16 norm_element;
#elif defined(SEISMIC_NORM_REPRESENTATION_BF16)
typedef packets::bf16 norm_element;
#else
#error "qwen_selected_rows requires a dense norm"
#endif
#if defined(SEISMIC_WEIGHT_REPRESENTATION_Q4K)
typedef packets::q4k head_packet;
#define HEAD_LAYOUT {SEISMIC_WEIGHT_ROW_STRIDE_BYTES, SEISMIC_WEIGHT_PLANE_CODES_LO_ROW_OFFSET, 0, SEISMIC_WEIGHT_PLANE_SCALES_ROW_OFFSET, SEISMIC_WEIGHT_PLANE_SUPERS_ROW_OFFSET}
#elif defined(SEISMIC_WEIGHT_REPRESENTATION_Q5K)
typedef packets::q5k head_packet;
#define HEAD_LAYOUT {SEISMIC_WEIGHT_ROW_STRIDE_BYTES, SEISMIC_WEIGHT_PLANE_CODES_LO_ROW_OFFSET, SEISMIC_WEIGHT_PLANE_CODES_HI_ROW_OFFSET, SEISMIC_WEIGHT_PLANE_SCALES_ROW_OFFSET, SEISMIC_WEIGHT_PLANE_SUPERS_ROW_OFFSET}
#elif defined(SEISMIC_WEIGHT_REPRESENTATION_Q6K)
typedef packets::q6k head_packet;
#define HEAD_LAYOUT {SEISMIC_WEIGHT_ROW_STRIDE_BYTES, SEISMIC_WEIGHT_PLANE_CODES_LO_ROW_OFFSET, SEISMIC_WEIGHT_PLANE_CODES_HI_ROW_OFFSET, SEISMIC_WEIGHT_PLANE_SCALES_ROW_OFFSET, SEISMIC_WEIGHT_PLANE_SUPERS_ROW_OFFSET}
#elif defined(SEISMIC_WEIGHT_REPRESENTATION_Q8G32S)
typedef packets::q8 head_packet;
#define HEAD_LAYOUT {SEISMIC_WEIGHT_ROW_STRIDE_BYTES, SEISMIC_WEIGHT_PLANE_CODES_ROW_OFFSET, 0, 0, SEISMIC_WEIGHT_PLANE_SUPERS_ROW_OFFSET}
#elif defined(SEISMIC_WEIGHT_REPRESENTATION_BF16)
typedef packets::dense<packets::bf16> head_packet;
#define HEAD_LAYOUT {SEISMIC_WEIGHT_STRIDE_0 * 2, 0, 0, 0, 0}
#elif defined(SEISMIC_WEIGHT_REPRESENTATION_F16)
typedef packets::dense<packets::f16> head_packet;
#define HEAD_LAYOUT {SEISMIC_WEIGHT_STRIDE_0 * 2, 0, 0, 0, 0}
#elif defined(SEISMIC_WEIGHT_REPRESENTATION_F32)
typedef packets::dense<packets::f32> head_packet;
#define HEAD_LAYOUT {SEISMIC_WEIGHT_STRIDE_0 * 4, 0, 0, 0, 0}
#else
#error "unsupported qwen_selected_rows weight representation"
#endif
#if defined(SEISMIC_WEIGHT_KIND_PACKED) && !defined(SEISMIC_WEIGHT_LAYOUT_ROWS16)
#error "qwen_selected_rows requires the rows16 layout for weight"
#endif

#define SELECTED_ROWS_ARGUMENTS                                                             \
    device const float *hidden [[buffer(SEISMIC_BUFFER_HIDDEN)]],                       \
    device const uchar *norm [[buffer(SEISMIC_BUFFER_NORM)]],                           \
    device const uchar *weight [[buffer(SEISMIC_BUFFER_WEIGHT)]],                       \
    device const int *out_rows [[buffer(SEISMIC_BUFFER_OUT_ROWS)]],                     \
    device const int *selected [[buffer(SEISMIC_BUFFER_SELECTED)]],                     \
    device float *logits [[buffer(SEISMIC_RESULT_0_BUFFER)]],                           \
    device uchar *normalized [[buffer(SEISMIC_BUFFER_SCRATCH_NORMALIZED)]],             \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

#define SELECTED_ROWS_OPERANDS                                                              \
    const uint k = uint(SEISMIC_DIM_D);                                                 \
    projection::input_rms<activation, norm_element> in{hidden, SEISMIC_HIDDEN_STRIDE_0, \
        SEISMIC_HIDDEN_STRIDE_1, norm, SEISMIC_NORM_STRIDE_0,                           \
        as_type<float>(uint(SEISMIC_PARAM_EPSILON)), k, {out_rows}};                    \
    projection::output_logits out{logits, SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1}; \
    projection::weight_rows<head_packet> w{weight, HEAD_LAYOUT, k, selected}

kernel void qwen_selected_rows_gemv(SELECTED_ROWS_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    SELECTED_ROWS_OPERANDS;
    uint rows = uint(SEISMIC_DIM_O);
    PROJECTION_SQUARES_SHARED(squares, decltype(in)::parts);
    projection::threadgroup_squares<SEISMIC_TUNE_SIMDGROUPS>(in, rows, squares, sg, lane);
    projection::SharedNorm<decltype(in)> x{in, squares};
    PROJECTION_FOR_ROWS(rows,
        projection::gemv<head_packet, SEISMIC_TUNE_SIMDGROUPS, SEISMIC_TUNE_ROWS, MAXM, SEISMIC_TUNE_LANES>(
            x, out, w, rows, uint(SEISMIC_DIM_SV), k, tile, shared, sg, lane));
}

kernel void qwen_selected_rows_batch(SELECTED_ROWS_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    SELECTED_ROWS_OPERANDS;
    PROJECTION_SQUARES_SHARED(squares, decltype(in)::parts);
    projection::threadgroup_squares<SEISMIC_TUNE_BATCH_SIMDGROUPS>(in, uint(SEISMIC_DIM_O), squares, sg, lane);
    projection::SharedNorm<decltype(in)> x{in, squares};
    projection::gemv_batch<head_packet, SEISMIC_TUNE_BATCH_SIMDGROUPS, SEISMIC_TUNE_BATCH_ROWS>(x, out, w,
        uint(SEISMIC_DIM_O), uint(SEISMIC_DIM_SV), k, tile, shared, sg, lane);
}

kernel void qwen_selected_rows_normalize(SELECTED_ROWS_ARGUMENTS,
    uint item [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    PROJECTION_NORMALIZE_SHARED(norms);
    SELECTED_ROWS_OPERANDS;
    projection::device_normalize<256>(in, item, normalized, k, norms, thread_index);
}

kernel void qwen_selected_rows_gemm(SELECTED_ROWS_ARGUMENTS,
    uint2 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    PROJECTION_GEMM_SHARED(shared, SEISMIC_TUNE_TILE_M, SEISMIC_TUNE_TILE_N);
    SELECTED_ROWS_OPERANDS;
    projection::input_plain<activation> x{normalized, k, 1, k, {nullptr}};
    projection::gemm<head_packet, SEISMIC_TUNE_TILE_M, SEISMIC_TUNE_TILE_N>(x, out, w, uint(SEISMIC_DIM_O),
        uint(SEISMIC_DIM_SV), k, tile.y, tile.x, shared, sg, lane);
}
