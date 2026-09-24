// qwen_recurrent_project: RMS prologue over the F32 residual, then one
// segmented projection qkv | z | alpha | beta, each segment with its own
// representation, into one activation row per input row.
#include "common/projection.h"

#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_BF16)
typedef packets::bf16 activation;
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_F16)
typedef packets::f16 activation;
#else
#error "qwen_recurrent_project requires a bf16 or f16 activation"
#endif
#if defined(SEISMIC_INPUT_NORM_REPRESENTATION_F32)
typedef packets::f32 norm_element;
#elif defined(SEISMIC_INPUT_NORM_REPRESENTATION_F16)
typedef packets::f16 norm_element;
#elif defined(SEISMIC_INPUT_NORM_REPRESENTATION_BF16)
typedef packets::bf16 norm_element;
#else
#error "qwen_recurrent_project requires a dense input_norm"
#endif
#if defined(SEISMIC_QKV_WEIGHT_REPRESENTATION_Q4K)
typedef packets::q4k qkv_packet;
#define QKV_LAYOUT {SEISMIC_QKV_WEIGHT_ROW_STRIDE_BYTES, SEISMIC_QKV_WEIGHT_PLANE_CODES_LO_ROW_OFFSET, 0, SEISMIC_QKV_WEIGHT_PLANE_SCALES_ROW_OFFSET, SEISMIC_QKV_WEIGHT_PLANE_SUPERS_ROW_OFFSET}
#elif defined(SEISMIC_QKV_WEIGHT_REPRESENTATION_Q5K)
typedef packets::q5k qkv_packet;
#define QKV_LAYOUT {SEISMIC_QKV_WEIGHT_ROW_STRIDE_BYTES, SEISMIC_QKV_WEIGHT_PLANE_CODES_LO_ROW_OFFSET, SEISMIC_QKV_WEIGHT_PLANE_CODES_HI_ROW_OFFSET, SEISMIC_QKV_WEIGHT_PLANE_SCALES_ROW_OFFSET, SEISMIC_QKV_WEIGHT_PLANE_SUPERS_ROW_OFFSET}
#elif defined(SEISMIC_QKV_WEIGHT_REPRESENTATION_Q6K)
typedef packets::q6k qkv_packet;
#define QKV_LAYOUT {SEISMIC_QKV_WEIGHT_ROW_STRIDE_BYTES, SEISMIC_QKV_WEIGHT_PLANE_CODES_LO_ROW_OFFSET, SEISMIC_QKV_WEIGHT_PLANE_CODES_HI_ROW_OFFSET, SEISMIC_QKV_WEIGHT_PLANE_SCALES_ROW_OFFSET, SEISMIC_QKV_WEIGHT_PLANE_SUPERS_ROW_OFFSET}
#elif defined(SEISMIC_QKV_WEIGHT_REPRESENTATION_Q8G32S)
typedef packets::q8 qkv_packet;
#define QKV_LAYOUT {SEISMIC_QKV_WEIGHT_ROW_STRIDE_BYTES, SEISMIC_QKV_WEIGHT_PLANE_CODES_ROW_OFFSET, 0, 0, SEISMIC_QKV_WEIGHT_PLANE_SUPERS_ROW_OFFSET}
#elif defined(SEISMIC_QKV_WEIGHT_REPRESENTATION_BF16)
typedef packets::dense<packets::bf16> qkv_packet;
#define QKV_LAYOUT {SEISMIC_QKV_WEIGHT_STRIDE_0 * 2, 0, 0, 0, 0}
#elif defined(SEISMIC_QKV_WEIGHT_REPRESENTATION_F16)
typedef packets::dense<packets::f16> qkv_packet;
#define QKV_LAYOUT {SEISMIC_QKV_WEIGHT_STRIDE_0 * 2, 0, 0, 0, 0}
#elif defined(SEISMIC_QKV_WEIGHT_REPRESENTATION_F32)
typedef packets::dense<packets::f32> qkv_packet;
#define QKV_LAYOUT {SEISMIC_QKV_WEIGHT_STRIDE_0 * 4, 0, 0, 0, 0}
#else
#error "unsupported qwen_recurrent_project qkv_weight representation"
#endif
#if defined(SEISMIC_QKV_WEIGHT_KIND_PACKED) && !defined(SEISMIC_QKV_WEIGHT_LAYOUT_ROWS16)
#error "qwen_recurrent_project requires the rows16 layout for qkv_weight"
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
#error "unsupported qwen_recurrent_project gate_weight representation"
#endif
#if defined(SEISMIC_GATE_WEIGHT_KIND_PACKED) && !defined(SEISMIC_GATE_WEIGHT_LAYOUT_ROWS16)
#error "qwen_recurrent_project requires the rows16 layout for gate_weight"
#endif
#if defined(SEISMIC_ALPHA_WEIGHT_REPRESENTATION_Q4K)
typedef packets::q4k alpha_packet;
#define ALPHA_LAYOUT {SEISMIC_ALPHA_WEIGHT_ROW_STRIDE_BYTES, SEISMIC_ALPHA_WEIGHT_PLANE_CODES_LO_ROW_OFFSET, 0, SEISMIC_ALPHA_WEIGHT_PLANE_SCALES_ROW_OFFSET, SEISMIC_ALPHA_WEIGHT_PLANE_SUPERS_ROW_OFFSET}
#elif defined(SEISMIC_ALPHA_WEIGHT_REPRESENTATION_Q5K)
typedef packets::q5k alpha_packet;
#define ALPHA_LAYOUT {SEISMIC_ALPHA_WEIGHT_ROW_STRIDE_BYTES, SEISMIC_ALPHA_WEIGHT_PLANE_CODES_LO_ROW_OFFSET, SEISMIC_ALPHA_WEIGHT_PLANE_CODES_HI_ROW_OFFSET, SEISMIC_ALPHA_WEIGHT_PLANE_SCALES_ROW_OFFSET, SEISMIC_ALPHA_WEIGHT_PLANE_SUPERS_ROW_OFFSET}
#elif defined(SEISMIC_ALPHA_WEIGHT_REPRESENTATION_Q6K)
typedef packets::q6k alpha_packet;
#define ALPHA_LAYOUT {SEISMIC_ALPHA_WEIGHT_ROW_STRIDE_BYTES, SEISMIC_ALPHA_WEIGHT_PLANE_CODES_LO_ROW_OFFSET, SEISMIC_ALPHA_WEIGHT_PLANE_CODES_HI_ROW_OFFSET, SEISMIC_ALPHA_WEIGHT_PLANE_SCALES_ROW_OFFSET, SEISMIC_ALPHA_WEIGHT_PLANE_SUPERS_ROW_OFFSET}
#elif defined(SEISMIC_ALPHA_WEIGHT_REPRESENTATION_Q8G32S)
typedef packets::q8 alpha_packet;
#define ALPHA_LAYOUT {SEISMIC_ALPHA_WEIGHT_ROW_STRIDE_BYTES, SEISMIC_ALPHA_WEIGHT_PLANE_CODES_ROW_OFFSET, 0, 0, SEISMIC_ALPHA_WEIGHT_PLANE_SUPERS_ROW_OFFSET}
#elif defined(SEISMIC_ALPHA_WEIGHT_REPRESENTATION_BF16)
typedef packets::dense<packets::bf16> alpha_packet;
#define ALPHA_LAYOUT {SEISMIC_ALPHA_WEIGHT_STRIDE_0 * 2, 0, 0, 0, 0}
#elif defined(SEISMIC_ALPHA_WEIGHT_REPRESENTATION_F16)
typedef packets::dense<packets::f16> alpha_packet;
#define ALPHA_LAYOUT {SEISMIC_ALPHA_WEIGHT_STRIDE_0 * 2, 0, 0, 0, 0}
#elif defined(SEISMIC_ALPHA_WEIGHT_REPRESENTATION_F32)
typedef packets::dense<packets::f32> alpha_packet;
#define ALPHA_LAYOUT {SEISMIC_ALPHA_WEIGHT_STRIDE_0 * 4, 0, 0, 0, 0}
#else
#error "unsupported qwen_recurrent_project alpha_weight representation"
#endif
#if defined(SEISMIC_ALPHA_WEIGHT_KIND_PACKED) && !defined(SEISMIC_ALPHA_WEIGHT_LAYOUT_ROWS16)
#error "qwen_recurrent_project requires the rows16 layout for alpha_weight"
#endif
#if defined(SEISMIC_BETA_WEIGHT_REPRESENTATION_Q4K)
typedef packets::q4k beta_packet;
#define BETA_LAYOUT {SEISMIC_BETA_WEIGHT_ROW_STRIDE_BYTES, SEISMIC_BETA_WEIGHT_PLANE_CODES_LO_ROW_OFFSET, 0, SEISMIC_BETA_WEIGHT_PLANE_SCALES_ROW_OFFSET, SEISMIC_BETA_WEIGHT_PLANE_SUPERS_ROW_OFFSET}
#elif defined(SEISMIC_BETA_WEIGHT_REPRESENTATION_Q5K)
typedef packets::q5k beta_packet;
#define BETA_LAYOUT {SEISMIC_BETA_WEIGHT_ROW_STRIDE_BYTES, SEISMIC_BETA_WEIGHT_PLANE_CODES_LO_ROW_OFFSET, SEISMIC_BETA_WEIGHT_PLANE_CODES_HI_ROW_OFFSET, SEISMIC_BETA_WEIGHT_PLANE_SCALES_ROW_OFFSET, SEISMIC_BETA_WEIGHT_PLANE_SUPERS_ROW_OFFSET}
#elif defined(SEISMIC_BETA_WEIGHT_REPRESENTATION_Q6K)
typedef packets::q6k beta_packet;
#define BETA_LAYOUT {SEISMIC_BETA_WEIGHT_ROW_STRIDE_BYTES, SEISMIC_BETA_WEIGHT_PLANE_CODES_LO_ROW_OFFSET, SEISMIC_BETA_WEIGHT_PLANE_CODES_HI_ROW_OFFSET, SEISMIC_BETA_WEIGHT_PLANE_SCALES_ROW_OFFSET, SEISMIC_BETA_WEIGHT_PLANE_SUPERS_ROW_OFFSET}
#elif defined(SEISMIC_BETA_WEIGHT_REPRESENTATION_Q8G32S)
typedef packets::q8 beta_packet;
#define BETA_LAYOUT {SEISMIC_BETA_WEIGHT_ROW_STRIDE_BYTES, SEISMIC_BETA_WEIGHT_PLANE_CODES_ROW_OFFSET, 0, 0, SEISMIC_BETA_WEIGHT_PLANE_SUPERS_ROW_OFFSET}
#elif defined(SEISMIC_BETA_WEIGHT_REPRESENTATION_BF16)
typedef packets::dense<packets::bf16> beta_packet;
#define BETA_LAYOUT {SEISMIC_BETA_WEIGHT_STRIDE_0 * 2, 0, 0, 0, 0}
#elif defined(SEISMIC_BETA_WEIGHT_REPRESENTATION_F16)
typedef packets::dense<packets::f16> beta_packet;
#define BETA_LAYOUT {SEISMIC_BETA_WEIGHT_STRIDE_0 * 2, 0, 0, 0, 0}
#elif defined(SEISMIC_BETA_WEIGHT_REPRESENTATION_F32)
typedef packets::dense<packets::f32> beta_packet;
#define BETA_LAYOUT {SEISMIC_BETA_WEIGHT_STRIDE_0 * 4, 0, 0, 0, 0}
#else
#error "unsupported qwen_recurrent_project beta_weight representation"
#endif
#if defined(SEISMIC_BETA_WEIGHT_KIND_PACKED) && !defined(SEISMIC_BETA_WEIGHT_LAYOUT_ROWS16)
#error "qwen_recurrent_project requires the rows16 layout for beta_weight"
#endif

#define RECURRENT_PROJECT_ARGUMENTS                                                     \
    device const float *hidden [[buffer(SEISMIC_BUFFER_HIDDEN)]],                       \
    device const uchar *input_norm [[buffer(SEISMIC_BUFFER_INPUT_NORM)]],               \
    device const uchar *qkv_weight [[buffer(SEISMIC_BUFFER_QKV_WEIGHT)]],               \
    device const uchar *gate_weight [[buffer(SEISMIC_BUFFER_GATE_WEIGHT)]],             \
    device const uchar *alpha_weight [[buffer(SEISMIC_BUFFER_ALPHA_WEIGHT)]],           \
    device const uchar *beta_weight [[buffer(SEISMIC_BUFFER_BETA_WEIGHT)]],             \
    device uchar *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],                           \
    device uchar *normalized [[buffer(SEISMIC_BUFFER_SCRATCH_NORMALIZED)]],             \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

#define RECURRENT_PROJECT_OPERANDS                                                      \
    const uint k = uint(SEISMIC_DIM_H);                                                 \
    const uint qkv_rows = uint((2 * SEISMIC_DIM_NK + SEISMIC_DIM_NV) * SEISMIC_DIM_W);  \
    const uint gate_rows = uint(SEISMIC_DIM_NV * SEISMIC_DIM_W);                        \
    const uint head_rows = uint(SEISMIC_DIM_NV);                                        \
    projection::input_rms<activation, norm_element> in{hidden, SEISMIC_HIDDEN_STRIDE_0, \
        SEISMIC_HIDDEN_STRIDE_1, input_norm, SEISMIC_INPUT_NORM_STRIDE_0,               \
        as_type<float>(uint(SEISMIC_PARAM_EPSILON)), k, {nullptr}};                     \
    projection::weight_rows<qkv_packet> qkv{qkv_weight, QKV_LAYOUT, k, nullptr};        \
    projection::weight_rows<gate_packet> gate{gate_weight, GATE_LAYOUT, k, nullptr};    \
    projection::weight_rows<alpha_packet> alpha{alpha_weight, ALPHA_LAYOUT, k, nullptr}; \
    projection::weight_rows<beta_packet> beta{beta_weight, BETA_LAYOUT, k, nullptr};    \
    projection::output_plain<activation> qkv_out{result, SEISMIC_RESULT_0_STRIDE_0,     \
        SEISMIC_RESULT_0_STRIDE_1, 0};                                                  \
    projection::output_plain<activation> gate_out{result, SEISMIC_RESULT_0_STRIDE_0,    \
        SEISMIC_RESULT_0_STRIDE_1, qkv_rows};                                           \
    projection::output_plain<activation> alpha_out{result, SEISMIC_RESULT_0_STRIDE_0,   \
        SEISMIC_RESULT_0_STRIDE_1, qkv_rows + gate_rows};                               \
    projection::output_plain<activation> beta_out{result, SEISMIC_RESULT_0_STRIDE_0,    \
        SEISMIC_RESULT_0_STRIDE_1, qkv_rows + gate_rows + head_rows}

kernel void qwen_recurrent_project_gemv(RECURRENT_PROJECT_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    RECURRENT_PROJECT_OPERANDS;
    constexpr uint SG = SEISMIC_TUNE_SIMDGROUPS, R = SEISMIC_TUNE_ROWS, L = SEISMIC_TUNE_LANES;
    constexpr uint per = projection::gemv_threadgroup_rows<SG, R, L>();
    uint rows = uint(SEISMIC_DIM_M);
    PROJECTION_SQUARES_SHARED(squares, decltype(in)::parts);
    projection::threadgroup_squares<SG>(in, rows, squares, sg, lane);
    projection::SharedNorm<decltype(in)> x{in, squares};
    uint t0 = (qkv_rows + per - 1) / per, t1 = (gate_rows + per - 1) / per;
    uint t2 = (head_rows + per - 1) / per;
    if (tile < t0) {
        PROJECTION_FOR_ROWS(rows, projection::gemv<qkv_packet, SG, R, MAXM, L>(
            x, qkv_out, qkv, rows, qkv_rows, k, tile, shared, sg, lane));
    } else if (tile < t0 + t1) {
        PROJECTION_FOR_ROWS(rows, projection::gemv<gate_packet, SG, R, MAXM, L>(
            x, gate_out, gate, rows, gate_rows, k, tile - t0, shared, sg, lane));
    } else if (tile < t0 + t1 + t2) {
        PROJECTION_FOR_ROWS(rows, projection::gemv<alpha_packet, SG, R, MAXM, L>(
            x, alpha_out, alpha, rows, head_rows, k, tile - t0 - t1, shared, sg, lane));
    } else {
        PROJECTION_FOR_ROWS(rows, projection::gemv<beta_packet, SG, R, MAXM, L>(
            x, beta_out, beta, rows, head_rows, k, tile - t0 - t1 - t2, shared, sg, lane));
    }
}

kernel void qwen_recurrent_project_batch(RECURRENT_PROJECT_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    RECURRENT_PROJECT_OPERANDS;
    constexpr uint SG = SEISMIC_TUNE_BATCH_SIMDGROUPS, R = SEISMIC_TUNE_BATCH_ROWS;
    constexpr uint per = projection::gemv_batch_threadgroup_rows<SG, R>();
    uint rows = uint(SEISMIC_DIM_M);
    PROJECTION_SQUARES_SHARED(squares, decltype(in)::parts);
    projection::threadgroup_squares<SG>(in, rows, squares, sg, lane);
    projection::SharedNorm<decltype(in)> x{in, squares};
    uint t0 = (qkv_rows + per - 1) / per, t1 = (gate_rows + per - 1) / per;
    uint t2 = (head_rows + per - 1) / per;
    if (tile < t0)
        projection::gemv_batch<qkv_packet, SG, R>(x, qkv_out, qkv, rows, qkv_rows, k, tile, shared, sg, lane);
    else if (tile < t0 + t1)
        projection::gemv_batch<gate_packet, SG, R>(x, gate_out, gate, rows, gate_rows, k, tile - t0, shared, sg,
            lane);
    else if (tile < t0 + t1 + t2)
        projection::gemv_batch<alpha_packet, SG, R>(x, alpha_out, alpha, rows, head_rows, k, tile - t0 - t1,
            shared, sg, lane);
    else
        projection::gemv_batch<beta_packet, SG, R>(x, beta_out, beta, rows, head_rows, k, tile - t0 - t1 - t2,
            shared, sg, lane);
}

kernel void qwen_recurrent_project_normalize(RECURRENT_PROJECT_ARGUMENTS,
    uint item [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    PROJECTION_NORMALIZE_SHARED(norms);
    RECURRENT_PROJECT_OPERANDS;
    projection::device_normalize<256>(in, item, normalized, k, norms, thread_index);
}

kernel void qwen_recurrent_project_gemm(RECURRENT_PROJECT_ARGUMENTS,
    uint2 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint TM = SEISMIC_TUNE_TILE_M, TN = SEISMIC_TUNE_TILE_N;
    PROJECTION_GEMM_SHARED(shared, TM, TN);
    RECURRENT_PROJECT_OPERANDS;
    projection::input_plain<activation> x{normalized, k, 1, k, {nullptr}};
    uint m = uint(SEISMIC_DIM_M);
    uint t0 = (qkv_rows + TN - 1) / TN, t1 = (gate_rows + TN - 1) / TN, t2 = (head_rows + TN - 1) / TN;
    uint n = tile.x;
    if (n < t0)
        projection::gemm<qkv_packet, TM, TN>(x, qkv_out, qkv, m, qkv_rows, k, tile.y, n, shared, sg, lane);
    else if (n < t0 + t1)
        projection::gemm<gate_packet, TM, TN>(x, gate_out, gate, m, gate_rows, k, tile.y, n - t0, shared, sg, lane);
    else if (n < t0 + t1 + t2)
        projection::gemm<alpha_packet, TM, TN>(x, alpha_out, alpha, m, head_rows, k, tile.y, n - t0 - t1, shared, sg, lane);
    else
        projection::gemm<beta_packet, TM, TN>(x, beta_out, beta, m, head_rows, k, tile.y, n - t0 - t1 - t2, shared, sg, lane);
}
