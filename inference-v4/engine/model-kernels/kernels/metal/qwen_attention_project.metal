// qwen_attention_project: RMS prologue over the F32 residual, then one
// segmented projection query+gate | key | value into three results.
#include "common/projection.h"

#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_BF16)
typedef packets::bf16 activation;
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_F16)
typedef packets::f16 activation;
#else
#error "qwen_attention_project requires a bf16 or f16 activation"
#endif
#if defined(SEISMIC_INPUT_NORM_REPRESENTATION_F32)
typedef packets::f32 norm_element;
#elif defined(SEISMIC_INPUT_NORM_REPRESENTATION_F16)
typedef packets::f16 norm_element;
#elif defined(SEISMIC_INPUT_NORM_REPRESENTATION_BF16)
typedef packets::bf16 norm_element;
#else
#error "qwen_attention_project requires a dense input_norm"
#endif
#if defined(SEISMIC_QUERY_GATE_WEIGHT_REPRESENTATION_Q4K)
typedef packets::q4k query_packet;
#define QUERY_LAYOUT {SEISMIC_QUERY_GATE_WEIGHT_ROW_STRIDE_BYTES, SEISMIC_QUERY_GATE_WEIGHT_PLANE_CODES_LO_ROW_OFFSET, 0, SEISMIC_QUERY_GATE_WEIGHT_PLANE_SCALES_ROW_OFFSET, SEISMIC_QUERY_GATE_WEIGHT_PLANE_SUPERS_ROW_OFFSET}
#elif defined(SEISMIC_QUERY_GATE_WEIGHT_REPRESENTATION_Q5K)
typedef packets::q5k query_packet;
#define QUERY_LAYOUT {SEISMIC_QUERY_GATE_WEIGHT_ROW_STRIDE_BYTES, SEISMIC_QUERY_GATE_WEIGHT_PLANE_CODES_LO_ROW_OFFSET, SEISMIC_QUERY_GATE_WEIGHT_PLANE_CODES_HI_ROW_OFFSET, SEISMIC_QUERY_GATE_WEIGHT_PLANE_SCALES_ROW_OFFSET, SEISMIC_QUERY_GATE_WEIGHT_PLANE_SUPERS_ROW_OFFSET}
#elif defined(SEISMIC_QUERY_GATE_WEIGHT_REPRESENTATION_Q6K)
typedef packets::q6k query_packet;
#define QUERY_LAYOUT {SEISMIC_QUERY_GATE_WEIGHT_ROW_STRIDE_BYTES, SEISMIC_QUERY_GATE_WEIGHT_PLANE_CODES_LO_ROW_OFFSET, SEISMIC_QUERY_GATE_WEIGHT_PLANE_CODES_HI_ROW_OFFSET, SEISMIC_QUERY_GATE_WEIGHT_PLANE_SCALES_ROW_OFFSET, SEISMIC_QUERY_GATE_WEIGHT_PLANE_SUPERS_ROW_OFFSET}
#elif defined(SEISMIC_QUERY_GATE_WEIGHT_REPRESENTATION_Q8G32S)
typedef packets::q8 query_packet;
#define QUERY_LAYOUT {SEISMIC_QUERY_GATE_WEIGHT_ROW_STRIDE_BYTES, SEISMIC_QUERY_GATE_WEIGHT_PLANE_CODES_ROW_OFFSET, 0, 0, SEISMIC_QUERY_GATE_WEIGHT_PLANE_SUPERS_ROW_OFFSET}
#elif defined(SEISMIC_QUERY_GATE_WEIGHT_REPRESENTATION_BF16)
typedef packets::dense<packets::bf16> query_packet;
#define QUERY_LAYOUT {SEISMIC_QUERY_GATE_WEIGHT_STRIDE_0 * 2, 0, 0, 0, 0}
#elif defined(SEISMIC_QUERY_GATE_WEIGHT_REPRESENTATION_F16)
typedef packets::dense<packets::f16> query_packet;
#define QUERY_LAYOUT {SEISMIC_QUERY_GATE_WEIGHT_STRIDE_0 * 2, 0, 0, 0, 0}
#elif defined(SEISMIC_QUERY_GATE_WEIGHT_REPRESENTATION_F32)
typedef packets::dense<packets::f32> query_packet;
#define QUERY_LAYOUT {SEISMIC_QUERY_GATE_WEIGHT_STRIDE_0 * 4, 0, 0, 0, 0}
#else
#error "unsupported qwen_attention_project query_gate_weight representation"
#endif
#if defined(SEISMIC_QUERY_GATE_WEIGHT_KIND_PACKED) && !defined(SEISMIC_QUERY_GATE_WEIGHT_LAYOUT_ROWS16)
#error "qwen_attention_project requires the rows16 layout for query_gate_weight"
#endif
#if defined(SEISMIC_KEY_WEIGHT_REPRESENTATION_Q4K)
typedef packets::q4k key_packet;
#define KEY_LAYOUT {SEISMIC_KEY_WEIGHT_ROW_STRIDE_BYTES, SEISMIC_KEY_WEIGHT_PLANE_CODES_LO_ROW_OFFSET, 0, SEISMIC_KEY_WEIGHT_PLANE_SCALES_ROW_OFFSET, SEISMIC_KEY_WEIGHT_PLANE_SUPERS_ROW_OFFSET}
#elif defined(SEISMIC_KEY_WEIGHT_REPRESENTATION_Q5K)
typedef packets::q5k key_packet;
#define KEY_LAYOUT {SEISMIC_KEY_WEIGHT_ROW_STRIDE_BYTES, SEISMIC_KEY_WEIGHT_PLANE_CODES_LO_ROW_OFFSET, SEISMIC_KEY_WEIGHT_PLANE_CODES_HI_ROW_OFFSET, SEISMIC_KEY_WEIGHT_PLANE_SCALES_ROW_OFFSET, SEISMIC_KEY_WEIGHT_PLANE_SUPERS_ROW_OFFSET}
#elif defined(SEISMIC_KEY_WEIGHT_REPRESENTATION_Q6K)
typedef packets::q6k key_packet;
#define KEY_LAYOUT {SEISMIC_KEY_WEIGHT_ROW_STRIDE_BYTES, SEISMIC_KEY_WEIGHT_PLANE_CODES_LO_ROW_OFFSET, SEISMIC_KEY_WEIGHT_PLANE_CODES_HI_ROW_OFFSET, SEISMIC_KEY_WEIGHT_PLANE_SCALES_ROW_OFFSET, SEISMIC_KEY_WEIGHT_PLANE_SUPERS_ROW_OFFSET}
#elif defined(SEISMIC_KEY_WEIGHT_REPRESENTATION_Q8G32S)
typedef packets::q8 key_packet;
#define KEY_LAYOUT {SEISMIC_KEY_WEIGHT_ROW_STRIDE_BYTES, SEISMIC_KEY_WEIGHT_PLANE_CODES_ROW_OFFSET, 0, 0, SEISMIC_KEY_WEIGHT_PLANE_SUPERS_ROW_OFFSET}
#elif defined(SEISMIC_KEY_WEIGHT_REPRESENTATION_BF16)
typedef packets::dense<packets::bf16> key_packet;
#define KEY_LAYOUT {SEISMIC_KEY_WEIGHT_STRIDE_0 * 2, 0, 0, 0, 0}
#elif defined(SEISMIC_KEY_WEIGHT_REPRESENTATION_F16)
typedef packets::dense<packets::f16> key_packet;
#define KEY_LAYOUT {SEISMIC_KEY_WEIGHT_STRIDE_0 * 2, 0, 0, 0, 0}
#elif defined(SEISMIC_KEY_WEIGHT_REPRESENTATION_F32)
typedef packets::dense<packets::f32> key_packet;
#define KEY_LAYOUT {SEISMIC_KEY_WEIGHT_STRIDE_0 * 4, 0, 0, 0, 0}
#else
#error "unsupported qwen_attention_project key_weight representation"
#endif
#if defined(SEISMIC_KEY_WEIGHT_KIND_PACKED) && !defined(SEISMIC_KEY_WEIGHT_LAYOUT_ROWS16)
#error "qwen_attention_project requires the rows16 layout for key_weight"
#endif
#if defined(SEISMIC_VALUE_WEIGHT_REPRESENTATION_Q4K)
typedef packets::q4k value_packet;
#define VALUE_LAYOUT {SEISMIC_VALUE_WEIGHT_ROW_STRIDE_BYTES, SEISMIC_VALUE_WEIGHT_PLANE_CODES_LO_ROW_OFFSET, 0, SEISMIC_VALUE_WEIGHT_PLANE_SCALES_ROW_OFFSET, SEISMIC_VALUE_WEIGHT_PLANE_SUPERS_ROW_OFFSET}
#elif defined(SEISMIC_VALUE_WEIGHT_REPRESENTATION_Q5K)
typedef packets::q5k value_packet;
#define VALUE_LAYOUT {SEISMIC_VALUE_WEIGHT_ROW_STRIDE_BYTES, SEISMIC_VALUE_WEIGHT_PLANE_CODES_LO_ROW_OFFSET, SEISMIC_VALUE_WEIGHT_PLANE_CODES_HI_ROW_OFFSET, SEISMIC_VALUE_WEIGHT_PLANE_SCALES_ROW_OFFSET, SEISMIC_VALUE_WEIGHT_PLANE_SUPERS_ROW_OFFSET}
#elif defined(SEISMIC_VALUE_WEIGHT_REPRESENTATION_Q6K)
typedef packets::q6k value_packet;
#define VALUE_LAYOUT {SEISMIC_VALUE_WEIGHT_ROW_STRIDE_BYTES, SEISMIC_VALUE_WEIGHT_PLANE_CODES_LO_ROW_OFFSET, SEISMIC_VALUE_WEIGHT_PLANE_CODES_HI_ROW_OFFSET, SEISMIC_VALUE_WEIGHT_PLANE_SCALES_ROW_OFFSET, SEISMIC_VALUE_WEIGHT_PLANE_SUPERS_ROW_OFFSET}
#elif defined(SEISMIC_VALUE_WEIGHT_REPRESENTATION_Q8G32S)
typedef packets::q8 value_packet;
#define VALUE_LAYOUT {SEISMIC_VALUE_WEIGHT_ROW_STRIDE_BYTES, SEISMIC_VALUE_WEIGHT_PLANE_CODES_ROW_OFFSET, 0, 0, SEISMIC_VALUE_WEIGHT_PLANE_SUPERS_ROW_OFFSET}
#elif defined(SEISMIC_VALUE_WEIGHT_REPRESENTATION_BF16)
typedef packets::dense<packets::bf16> value_packet;
#define VALUE_LAYOUT {SEISMIC_VALUE_WEIGHT_STRIDE_0 * 2, 0, 0, 0, 0}
#elif defined(SEISMIC_VALUE_WEIGHT_REPRESENTATION_F16)
typedef packets::dense<packets::f16> value_packet;
#define VALUE_LAYOUT {SEISMIC_VALUE_WEIGHT_STRIDE_0 * 2, 0, 0, 0, 0}
#elif defined(SEISMIC_VALUE_WEIGHT_REPRESENTATION_F32)
typedef packets::dense<packets::f32> value_packet;
#define VALUE_LAYOUT {SEISMIC_VALUE_WEIGHT_STRIDE_0 * 4, 0, 0, 0, 0}
#else
#error "unsupported qwen_attention_project value_weight representation"
#endif
#if defined(SEISMIC_VALUE_WEIGHT_KIND_PACKED) && !defined(SEISMIC_VALUE_WEIGHT_LAYOUT_ROWS16)
#error "qwen_attention_project requires the rows16 layout for value_weight"
#endif

#define ATTENTION_PROJECT_ARGUMENTS                                                     \
    device const float *hidden [[buffer(SEISMIC_BUFFER_HIDDEN)]],                       \
    device const uchar *input_norm [[buffer(SEISMIC_BUFFER_INPUT_NORM)]],               \
    device const uchar *query_gate_weight [[buffer(SEISMIC_BUFFER_QUERY_GATE_WEIGHT)]], \
    device const uchar *key_weight [[buffer(SEISMIC_BUFFER_KEY_WEIGHT)]],               \
    device const uchar *value_weight [[buffer(SEISMIC_BUFFER_VALUE_WEIGHT)]],           \
    device uchar *query_gate [[buffer(SEISMIC_RESULT_0_BUFFER)]],                       \
    device uchar *key [[buffer(SEISMIC_RESULT_1_BUFFER)]],                              \
    device uchar *value [[buffer(SEISMIC_RESULT_2_BUFFER)]],                            \
    device uchar *normalized [[buffer(SEISMIC_BUFFER_SCRATCH_NORMALIZED)]],             \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

#define ATTENTION_PROJECT_OPERANDS                                                      \
    const uint k = uint(SEISMIC_DIM_D);                                                 \
    const uint query_rows = uint(SEISMIC_DIM_KV * SEISMIC_DIM_G * 2 * SEISMIC_DIM_W);   \
    const uint kv_rows = uint(SEISMIC_DIM_KV * SEISMIC_DIM_W);                          \
    projection::input_rms<activation, norm_element> in{hidden, SEISMIC_HIDDEN_STRIDE_0, \
        SEISMIC_HIDDEN_STRIDE_1, input_norm, SEISMIC_INPUT_NORM_STRIDE_0,               \
        as_type<float>(uint(SEISMIC_PARAM_EPSILON)), k, {nullptr}};                     \
    projection::weight_rows<query_packet> query_w{query_gate_weight, QUERY_LAYOUT, k, nullptr}; \
    projection::weight_rows<key_packet> key_w{key_weight, KEY_LAYOUT, k, nullptr};      \
    projection::weight_rows<value_packet> value_w{value_weight, VALUE_LAYOUT, k, nullptr}; \
    projection::output_plain<activation> query_out{query_gate, SEISMIC_RESULT_0_STRIDE_0, \
        SEISMIC_RESULT_0_STRIDE_1, 0};                                                  \
    projection::output_plain<activation> key_out{key, SEISMIC_RESULT_1_STRIDE_0,        \
        SEISMIC_RESULT_1_STRIDE_1, 0};                                                  \
    projection::output_plain<activation> value_out{value, SEISMIC_RESULT_2_STRIDE_0,    \
        SEISMIC_RESULT_2_STRIDE_1, 0}

kernel void qwen_attention_project_gemv(ATTENTION_PROJECT_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    ATTENTION_PROJECT_OPERANDS;
    constexpr uint SG = SEISMIC_TUNE_SIMDGROUPS, R = SEISMIC_TUNE_ROWS, L = SEISMIC_TUNE_LANES;
    constexpr uint per = projection::gemv_threadgroup_rows<SG, R, L>();
    uint rows = uint(SEISMIC_DIM_M);
    PROJECTION_SQUARES_SHARED(squares, decltype(in)::parts);
    projection::threadgroup_squares<SG>(in, rows, squares, sg, lane);
    projection::SharedNorm<decltype(in)> x{in, squares};
    uint t0 = (query_rows + per - 1) / per, t1 = (kv_rows + per - 1) / per;
    if (tile < t0) {
        PROJECTION_FOR_ROWS(rows, projection::gemv<query_packet, SG, R, MAXM, L>(
            x, query_out, query_w, rows, query_rows, k, tile, shared, sg, lane));
    } else if (tile < t0 + t1) {
        PROJECTION_FOR_ROWS(rows, projection::gemv<key_packet, SG, R, MAXM, L>(
            x, key_out, key_w, rows, kv_rows, k, tile - t0, shared, sg, lane));
    } else {
        PROJECTION_FOR_ROWS(rows, projection::gemv<value_packet, SG, R, MAXM, L>(
            x, value_out, value_w, rows, kv_rows, k, tile - t0 - t1, shared, sg, lane));
    }
}

kernel void qwen_attention_project_batch(ATTENTION_PROJECT_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    ATTENTION_PROJECT_OPERANDS;
    constexpr uint SG = SEISMIC_TUNE_BATCH_SIMDGROUPS, R = SEISMIC_TUNE_BATCH_ROWS;
    constexpr uint per = projection::gemv_batch_threadgroup_rows<SG, R>();
    uint rows = uint(SEISMIC_DIM_M);
    PROJECTION_SQUARES_SHARED(squares, decltype(in)::parts);
    projection::threadgroup_squares<SG>(in, rows, squares, sg, lane);
    projection::SharedNorm<decltype(in)> x{in, squares};
    uint t0 = (query_rows + per - 1) / per, t1 = (kv_rows + per - 1) / per;
    if (tile < t0)
        projection::gemv_batch<query_packet, SG, R>(x, query_out, query_w, rows, query_rows, k, tile, shared, sg,
            lane);
    else if (tile < t0 + t1)
        projection::gemv_batch<key_packet, SG, R>(x, key_out, key_w, rows, kv_rows, k, tile - t0, shared, sg,
            lane);
    else
        projection::gemv_batch<value_packet, SG, R>(x, value_out, value_w, rows, kv_rows, k, tile - t0 - t1,
            shared, sg, lane);
}

kernel void qwen_attention_project_normalize(ATTENTION_PROJECT_ARGUMENTS,
    uint item [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    PROJECTION_NORMALIZE_SHARED(norms);
    ATTENTION_PROJECT_OPERANDS;
    projection::device_normalize<256>(in, item, normalized, k, norms, thread_index);
}

kernel void qwen_attention_project_gemm(ATTENTION_PROJECT_ARGUMENTS,
    uint2 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint TM = SEISMIC_TUNE_TILE_M, TN = SEISMIC_TUNE_TILE_N;
    PROJECTION_GEMM_SHARED(shared, TM, TN);
    ATTENTION_PROJECT_OPERANDS;
    projection::input_plain<activation> x{normalized, k, 1, k, {nullptr}};
    uint m = uint(SEISMIC_DIM_M);
    uint t0 = (query_rows + TN - 1) / TN, t1 = (kv_rows + TN - 1) / TN;
    uint n = tile.x;
    if (n < t0)
        projection::gemm<query_packet, TM, TN>(x, query_out, query_w, m, query_rows, k, tile.y, n, shared, sg, lane);
    else if (n < t0 + t1)
        projection::gemm<key_packet, TM, TN>(x, key_out, key_w, m, kv_rows, k, tile.y, n - t0, shared, sg, lane);
    else
        projection::gemm<value_packet, TM, TN>(x, value_out, value_w, m, kv_rows, k, tile.y, n - t0 - t1, shared, sg, lane);
}
