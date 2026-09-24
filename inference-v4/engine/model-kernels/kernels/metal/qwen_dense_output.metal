// qwen_dense_output: the down projection of the product rows plus the
// residual rows they were gathered from (`out_rows`).
#include "common/projection.h"

#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_BF16)
typedef packets::bf16 activation;
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_F16)
typedef packets::f16 activation;
#else
#error "qwen_dense_output requires a bf16 or f16 activation"
#endif
#if defined(SEISMIC_DOWN_WEIGHT_REPRESENTATION_Q4K)
typedef packets::q4k down_packet;
#define DOWN_LAYOUT {SEISMIC_DOWN_WEIGHT_ROW_STRIDE_BYTES, SEISMIC_DOWN_WEIGHT_PLANE_CODES_LO_ROW_OFFSET, 0, SEISMIC_DOWN_WEIGHT_PLANE_SCALES_ROW_OFFSET, SEISMIC_DOWN_WEIGHT_PLANE_SUPERS_ROW_OFFSET}
#elif defined(SEISMIC_DOWN_WEIGHT_REPRESENTATION_Q5K)
typedef packets::q5k down_packet;
#define DOWN_LAYOUT {SEISMIC_DOWN_WEIGHT_ROW_STRIDE_BYTES, SEISMIC_DOWN_WEIGHT_PLANE_CODES_LO_ROW_OFFSET, SEISMIC_DOWN_WEIGHT_PLANE_CODES_HI_ROW_OFFSET, SEISMIC_DOWN_WEIGHT_PLANE_SCALES_ROW_OFFSET, SEISMIC_DOWN_WEIGHT_PLANE_SUPERS_ROW_OFFSET}
#elif defined(SEISMIC_DOWN_WEIGHT_REPRESENTATION_Q6K)
typedef packets::q6k down_packet;
#define DOWN_LAYOUT {SEISMIC_DOWN_WEIGHT_ROW_STRIDE_BYTES, SEISMIC_DOWN_WEIGHT_PLANE_CODES_LO_ROW_OFFSET, SEISMIC_DOWN_WEIGHT_PLANE_CODES_HI_ROW_OFFSET, SEISMIC_DOWN_WEIGHT_PLANE_SCALES_ROW_OFFSET, SEISMIC_DOWN_WEIGHT_PLANE_SUPERS_ROW_OFFSET}
#elif defined(SEISMIC_DOWN_WEIGHT_REPRESENTATION_Q8G32S)
typedef packets::q8 down_packet;
#define DOWN_LAYOUT {SEISMIC_DOWN_WEIGHT_ROW_STRIDE_BYTES, SEISMIC_DOWN_WEIGHT_PLANE_CODES_ROW_OFFSET, 0, 0, SEISMIC_DOWN_WEIGHT_PLANE_SUPERS_ROW_OFFSET}
#elif defined(SEISMIC_DOWN_WEIGHT_REPRESENTATION_BF16)
typedef packets::dense<packets::bf16> down_packet;
#define DOWN_LAYOUT {SEISMIC_DOWN_WEIGHT_STRIDE_0 * 2, 0, 0, 0, 0}
#elif defined(SEISMIC_DOWN_WEIGHT_REPRESENTATION_F16)
typedef packets::dense<packets::f16> down_packet;
#define DOWN_LAYOUT {SEISMIC_DOWN_WEIGHT_STRIDE_0 * 2, 0, 0, 0, 0}
#elif defined(SEISMIC_DOWN_WEIGHT_REPRESENTATION_F32)
typedef packets::dense<packets::f32> down_packet;
#define DOWN_LAYOUT {SEISMIC_DOWN_WEIGHT_STRIDE_0 * 4, 0, 0, 0, 0}
#else
#error "unsupported qwen_dense_output down_weight representation"
#endif
#if defined(SEISMIC_DOWN_WEIGHT_KIND_PACKED) && !defined(SEISMIC_DOWN_WEIGHT_LAYOUT_ROWS16)
#error "qwen_dense_output requires the rows16 layout for down_weight"
#endif

#define DENSE_OUTPUT_ARGUMENTS                                                          \
    device const float *residual [[buffer(SEISMIC_BUFFER_RESIDUAL)]],                   \
    device const uchar *product [[buffer(SEISMIC_BUFFER_PRODUCT)]],                     \
    device const uchar *down_weight [[buffer(SEISMIC_BUFFER_DOWN_WEIGHT)]],             \
    device const int *out_rows [[buffer(SEISMIC_BUFFER_OUT_ROWS)]],                     \
    device float *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],                           \
    device float *partials [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIALS)]],                 \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

#define DENSE_OUTPUT_OPERANDS                                                         \
    projection::input_plain<activation> in{product, SEISMIC_PRODUCT_STRIDE_0,          \
        SEISMIC_PRODUCT_STRIDE_1, uint(SEISMIC_DIM_F), {nullptr}};                      \
    projection::output_residual<activation> out{result, SEISMIC_RESULT_0_STRIDE_0,      \
        SEISMIC_RESULT_0_STRIDE_1, residual, SEISMIC_RESIDUAL_STRIDE_0,                 \
        SEISMIC_RESIDUAL_STRIDE_1, {out_rows}};                                         \
    projection::weight_rows<down_packet> w{down_weight, DOWN_LAYOUT,                    \
        uint(SEISMIC_DIM_F), nullptr}

kernel void qwen_dense_output_gemv(DENSE_OUTPUT_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    DENSE_OUTPUT_OPERANDS;
    uint rows = uint(SEISMIC_DIM_O);
    PROJECTION_FOR_ROWS(rows,
        projection::gemv<down_packet, SEISMIC_TUNE_SIMDGROUPS, SEISMIC_TUNE_ROWS, MAXM, SEISMIC_TUNE_LANES>(
            in, out, w, rows, uint(SEISMIC_DIM_H), uint(SEISMIC_DIM_F), tile, shared, sg, lane));
}

kernel void qwen_dense_output_batch(DENSE_OUTPUT_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    DENSE_OUTPUT_OPERANDS;
    projection::gemv_batch<down_packet, SEISMIC_TUNE_BATCH_SIMDGROUPS, SEISMIC_TUNE_BATCH_ROWS>(in, out, w,
        uint(SEISMIC_DIM_O), uint(SEISMIC_DIM_H), uint(SEISMIC_DIM_F), tile, shared, sg, lane);
}

kernel void qwen_dense_output_gemm(DENSE_OUTPUT_ARGUMENTS,
    uint3 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint TM = SEISMIC_TUNE_TILE_M, TN = SEISMIC_TUNE_TILE_N;
    PROJECTION_GEMM_SHARED(shared, TM, TN);
    DENSE_OUTPUT_OPERANDS;
    const uint m = uint(SEISMIC_DIM_O), n = uint(SEISMIC_DIM_H), k = uint(SEISMIC_DIM_F);
    if (SEISMIC_TUNE_SPLIT == 1)
        projection::gemm<down_packet, TM, TN>(in, out, w, m, n, k, tile.y, tile.x, shared, sg, lane);
    else
        projection::gemm_part<down_packet, TM, TN>(in, partials, w, m, n, k, SEISMIC_TUNE_SPLIT, tile.z,
            tile.y, tile.x, shared, sg, lane);
}

kernel void qwen_dense_output_reduce(DENSE_OUTPUT_ARGUMENTS,
    uint index [[thread_position_in_grid]]) {
    DENSE_OUTPUT_OPERANDS;
    projection::gemm_reduce(out, partials, uint(SEISMIC_DIM_O), uint(SEISMIC_DIM_H), SEISMIC_TUNE_SPLIT, index);
}
