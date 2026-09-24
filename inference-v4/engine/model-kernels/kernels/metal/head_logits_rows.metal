// head_logits_rows: the vocabulary projection of already-normalized feature
// rows (the draft head's readout) into F32 logits.
#include "common/projection.h"

#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_BF16)
typedef packets::bf16 activation;
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_F16)
typedef packets::f16 activation;
#else
#error "head_logits_rows requires a bf16 or f16 activation"
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
#error "unsupported head_logits_rows weight representation"
#endif
#if defined(SEISMIC_WEIGHT_KIND_PACKED) && !defined(SEISMIC_WEIGHT_LAYOUT_ROWS16)
#error "head_logits_rows requires the rows16 layout for weight"
#endif

#define HEAD_LOGITS_ARGUMENTS                                                           \
    device const uchar *features [[buffer(SEISMIC_BUFFER_FEATURES)]],                   \
    device const uchar *weight [[buffer(SEISMIC_BUFFER_WEIGHT)]],                       \
    device float *logits [[buffer(SEISMIC_RESULT_0_BUFFER)]],                           \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

#define HEAD_LOGITS_OPERANDS                                                            \
    const uint k = uint(SEISMIC_DIM_D);                                                 \
    projection::input_plain<activation> in{features, SEISMIC_FEATURES_STRIDE_0,        \
        SEISMIC_FEATURES_STRIDE_1, k, {nullptr}};                                       \
    projection::output_logits out{logits, SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1}; \
    projection::weight_rows<head_packet> w{weight, HEAD_LAYOUT, k, nullptr}

kernel void head_logits_rows_gemv(HEAD_LOGITS_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    HEAD_LOGITS_OPERANDS;
    uint rows = uint(SEISMIC_DIM_O);
    PROJECTION_FOR_ROWS(rows,
        projection::gemv<head_packet, SEISMIC_TUNE_SIMDGROUPS, SEISMIC_TUNE_ROWS, MAXM, SEISMIC_TUNE_LANES>(
            in, out, w, rows, uint(SEISMIC_DIM_V), k, tile, shared, sg, lane));
}

kernel void head_logits_rows_batch(HEAD_LOGITS_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    HEAD_LOGITS_OPERANDS;
    projection::gemv_batch<head_packet, SEISMIC_TUNE_BATCH_SIMDGROUPS, SEISMIC_TUNE_BATCH_ROWS>(in, out, w,
        uint(SEISMIC_DIM_O), uint(SEISMIC_DIM_V), k, tile, shared, sg, lane);
}

kernel void head_logits_rows_gemm(HEAD_LOGITS_ARGUMENTS,
    uint2 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    PROJECTION_GEMM_SHARED(shared, SEISMIC_TUNE_TILE_M, SEISMIC_TUNE_TILE_N);
    HEAD_LOGITS_OPERANDS;
    projection::gemm<head_packet, SEISMIC_TUNE_TILE_M, SEISMIC_TUNE_TILE_N>(in, out, w, uint(SEISMIC_DIM_O),
        uint(SEISMIC_DIM_V), k, tile.y, tile.x, shared, sg, lane);
}
