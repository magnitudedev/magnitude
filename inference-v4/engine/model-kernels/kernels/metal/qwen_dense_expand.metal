// qwen_dense_expand: RMS prologue over the `out_rows` residual rows, the
// paired gate/up projection, and SiLU(gate)·up.
#include "common/projection.h"

#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_BF16)
typedef packets::bf16 activation;
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_F16)
typedef packets::f16 activation;
#else
#error "qwen_dense_expand requires a bf16 or f16 activation"
#endif
#if defined(SEISMIC_NORM_REPRESENTATION_F32)
typedef packets::f32 norm_element;
#elif defined(SEISMIC_NORM_REPRESENTATION_F16)
typedef packets::f16 norm_element;
#elif defined(SEISMIC_NORM_REPRESENTATION_BF16)
typedef packets::bf16 norm_element;
#else
#error "qwen_dense_expand requires a dense norm"
#endif
#if defined(SEISMIC_GATE_WEIGHT_REPRESENTATION_Q4K)
typedef packets::q4k gate_packet;
#define GATE_LAYOUT {SEISMIC_GATE_WEIGHT_ROW_STRIDE_BYTES, SEISMIC_GATE_WEIGHT_PLANE_CODES_LO_ROW_OFFSET, 0, SEISMIC_GATE_WEIGHT_PLANE_SCALES_ROW_OFFSET, SEISMIC_GATE_WEIGHT_PLANE_SUPERS_ROW_OFFSET}
#elif defined(SEISMIC_GATE_WEIGHT_REPRESENTATION_Q5K)
typedef packets::q5k gate_packet;
#define GATE_LAYOUT {SEISMIC_GATE_WEIGHT_ROW_STRIDE_BYTES, SEISMIC_GATE_WEIGHT_PLANE_CODES_LO_ROW_OFFSET, SEISMIC_GATE_WEIGHT_PLANE_CODES_HI_ROW_OFFSET, SEISMIC_GATE_WEIGHT_PLANE_SCALES_ROW_OFFSET, SEISMIC_GATE_WEIGHT_PLANE_SUPERS_ROW_OFFSET}
#elif defined(SEISMIC_GATE_WEIGHT_REPRESENTATION_Q6K)
typedef packets::q6k gate_packet;
#define GATE_LAYOUT {SEISMIC_GATE_WEIGHT_ROW_STRIDE_BYTES, SEISMIC_GATE_WEIGHT_PLANE_CODES_LO_ROW_OFFSET, SEISMIC_GATE_WEIGHT_PLANE_CODES_HI_ROW_OFFSET, SEISMIC_GATE_WEIGHT_PLANE_SCALES_ROW_OFFSET, SEISMIC_GATE_WEIGHT_PLANE_SUPERS_ROW_OFFSET}
#elif defined(SEISMIC_GATE_WEIGHT_REPRESENTATION_Q8G32S)
typedef packets::q8 gate_packet;
#define GATE_LAYOUT {SEISMIC_GATE_WEIGHT_ROW_STRIDE_BYTES, SEISMIC_GATE_WEIGHT_PLANE_CODES_ROW_OFFSET, 0, 0, SEISMIC_GATE_WEIGHT_PLANE_SUPERS_ROW_OFFSET}
#elif defined(SEISMIC_GATE_WEIGHT_REPRESENTATION_BF16)
typedef packets::dense<packets::bf16> gate_packet;
#define GATE_LAYOUT {SEISMIC_GATE_WEIGHT_STRIDE_0 * 2, 0, 0, 0, 0}
#elif defined(SEISMIC_GATE_WEIGHT_REPRESENTATION_F16)
typedef packets::dense<packets::f16> gate_packet;
#define GATE_LAYOUT {SEISMIC_GATE_WEIGHT_STRIDE_0 * 2, 0, 0, 0, 0}
#elif defined(SEISMIC_GATE_WEIGHT_REPRESENTATION_F32)
typedef packets::dense<packets::f32> gate_packet;
#define GATE_LAYOUT {SEISMIC_GATE_WEIGHT_STRIDE_0 * 4, 0, 0, 0, 0}
#else
#error "unsupported qwen_dense_expand gate_weight representation"
#endif
#if defined(SEISMIC_GATE_WEIGHT_KIND_PACKED) && !defined(SEISMIC_GATE_WEIGHT_LAYOUT_ROWS16)
#error "qwen_dense_expand requires the rows16 layout for gate_weight"
#endif
#if defined(SEISMIC_UP_WEIGHT_REPRESENTATION_Q4K)
typedef packets::q4k up_packet;
#define UP_LAYOUT {SEISMIC_UP_WEIGHT_ROW_STRIDE_BYTES, SEISMIC_UP_WEIGHT_PLANE_CODES_LO_ROW_OFFSET, 0, SEISMIC_UP_WEIGHT_PLANE_SCALES_ROW_OFFSET, SEISMIC_UP_WEIGHT_PLANE_SUPERS_ROW_OFFSET}
#elif defined(SEISMIC_UP_WEIGHT_REPRESENTATION_Q5K)
typedef packets::q5k up_packet;
#define UP_LAYOUT {SEISMIC_UP_WEIGHT_ROW_STRIDE_BYTES, SEISMIC_UP_WEIGHT_PLANE_CODES_LO_ROW_OFFSET, SEISMIC_UP_WEIGHT_PLANE_CODES_HI_ROW_OFFSET, SEISMIC_UP_WEIGHT_PLANE_SCALES_ROW_OFFSET, SEISMIC_UP_WEIGHT_PLANE_SUPERS_ROW_OFFSET}
#elif defined(SEISMIC_UP_WEIGHT_REPRESENTATION_Q6K)
typedef packets::q6k up_packet;
#define UP_LAYOUT {SEISMIC_UP_WEIGHT_ROW_STRIDE_BYTES, SEISMIC_UP_WEIGHT_PLANE_CODES_LO_ROW_OFFSET, SEISMIC_UP_WEIGHT_PLANE_CODES_HI_ROW_OFFSET, SEISMIC_UP_WEIGHT_PLANE_SCALES_ROW_OFFSET, SEISMIC_UP_WEIGHT_PLANE_SUPERS_ROW_OFFSET}
#elif defined(SEISMIC_UP_WEIGHT_REPRESENTATION_Q8G32S)
typedef packets::q8 up_packet;
#define UP_LAYOUT {SEISMIC_UP_WEIGHT_ROW_STRIDE_BYTES, SEISMIC_UP_WEIGHT_PLANE_CODES_ROW_OFFSET, 0, 0, SEISMIC_UP_WEIGHT_PLANE_SUPERS_ROW_OFFSET}
#elif defined(SEISMIC_UP_WEIGHT_REPRESENTATION_BF16)
typedef packets::dense<packets::bf16> up_packet;
#define UP_LAYOUT {SEISMIC_UP_WEIGHT_STRIDE_0 * 2, 0, 0, 0, 0}
#elif defined(SEISMIC_UP_WEIGHT_REPRESENTATION_F16)
typedef packets::dense<packets::f16> up_packet;
#define UP_LAYOUT {SEISMIC_UP_WEIGHT_STRIDE_0 * 2, 0, 0, 0, 0}
#elif defined(SEISMIC_UP_WEIGHT_REPRESENTATION_F32)
typedef packets::dense<packets::f32> up_packet;
#define UP_LAYOUT {SEISMIC_UP_WEIGHT_STRIDE_0 * 4, 0, 0, 0, 0}
#else
#error "unsupported qwen_dense_expand up_weight representation"
#endif
#if defined(SEISMIC_UP_WEIGHT_KIND_PACKED) && !defined(SEISMIC_UP_WEIGHT_LAYOUT_ROWS16)
#error "qwen_dense_expand requires the rows16 layout for up_weight"
#endif

#define DENSE_EXPAND_ARGUMENTS                                                          \
    device const float *residual [[buffer(SEISMIC_BUFFER_RESIDUAL)]],                   \
    device const uchar *norm [[buffer(SEISMIC_BUFFER_NORM)]],                           \
    device const uchar *gate_weight [[buffer(SEISMIC_BUFFER_GATE_WEIGHT)]],             \
    device const uchar *up_weight [[buffer(SEISMIC_BUFFER_UP_WEIGHT)]],                 \
    device const int *out_rows [[buffer(SEISMIC_BUFFER_OUT_ROWS)]],                     \
    device uchar *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],                           \
    device uchar *normalized [[buffer(SEISMIC_BUFFER_SCRATCH_NORMALIZED)]],             \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

#define DENSE_EXPAND_OPERANDS                                                           \
    projection::input_rms<activation, norm_element> in{residual,                        \
        SEISMIC_RESIDUAL_STRIDE_0, SEISMIC_RESIDUAL_STRIDE_1, norm, SEISMIC_NORM_STRIDE_0, \
        as_type<float>(uint(SEISMIC_PARAM_EPS)), uint(SEISMIC_DIM_H), {out_rows}};      \
    projection::output_paired<activation> out{result, SEISMIC_RESULT_0_STRIDE_0,        \
        SEISMIC_RESULT_0_STRIDE_1};                                                     \
    projection::weight_rows<gate_packet> gate{gate_weight, GATE_LAYOUT,                 \
        uint(SEISMIC_DIM_H), nullptr};                                                  \
    projection::weight_rows<up_packet> up{up_weight, UP_LAYOUT, uint(SEISMIC_DIM_H), nullptr}

kernel void qwen_dense_expand_gemv(DENSE_EXPAND_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    DENSE_EXPAND_OPERANDS;
    uint rows = uint(SEISMIC_DIM_O);
    PROJECTION_SQUARES_SHARED(squares, decltype(in)::parts);
    projection::threadgroup_squares<SEISMIC_TUNE_SIMDGROUPS>(in, rows, squares, sg, lane);
    projection::SharedNorm<decltype(in)> x{in, squares};
    PROJECTION_FOR_ROWS(rows,
        projection::gemv_paired<gate_packet, up_packet, SEISMIC_TUNE_SIMDGROUPS, SEISMIC_TUNE_ROWS, MAXM,
            SEISMIC_TUNE_LANES>(
            x, out, gate, up, rows, uint(SEISMIC_DIM_F), uint(SEISMIC_DIM_H), tile, shared, sg, lane));
}

kernel void qwen_dense_expand_batch(DENSE_EXPAND_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    DENSE_EXPAND_OPERANDS;
    PROJECTION_SQUARES_SHARED(squares, decltype(in)::parts);
    projection::threadgroup_squares<SEISMIC_TUNE_BATCH_SIMDGROUPS>(in, uint(SEISMIC_DIM_O), squares, sg, lane);
    projection::SharedNorm<decltype(in)> x{in, squares};
    projection::gemv_batch_paired<gate_packet, up_packet, SEISMIC_TUNE_BATCH_SIMDGROUPS,
        SEISMIC_TUNE_BATCH_ROWS>(x, out,
        gate, up, uint(SEISMIC_DIM_O), uint(SEISMIC_DIM_F), uint(SEISMIC_DIM_H), tile, shared, sg, lane);
}

kernel void qwen_dense_expand_normalize(DENSE_EXPAND_ARGUMENTS,
    uint item [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    PROJECTION_NORMALIZE_SHARED(norms);
    DENSE_EXPAND_OPERANDS;
    projection::device_normalize<256>(in, item, normalized, uint(SEISMIC_DIM_H), norms, thread_index);
}

kernel void qwen_dense_expand_gemm(DENSE_EXPAND_ARGUMENTS,
    uint2 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    PROJECTION_GEMM_SHARED(shared, SEISMIC_TUNE_TILE_M, SEISMIC_TUNE_TILE_N);
    DENSE_EXPAND_OPERANDS;
    projection::input_plain<activation> x{normalized, SEISMIC_DIM_H, 1, uint(SEISMIC_DIM_H), {nullptr}};
    projection::gemm_paired<gate_packet, up_packet, SEISMIC_TUNE_TILE_M, SEISMIC_TUNE_TILE_N>(x, out, gate, up,
        uint(SEISMIC_DIM_O), uint(SEISMIC_DIM_F), uint(SEISMIC_DIM_H), tile.y, tile.x, shared, sg, lane);
}
