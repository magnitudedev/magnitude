// gated_attention_project: RMS prologue over the F32 residual, then one
// segmented projection query+gate | key | value into three results.
#define KERNEL_W0 SEISMIC_QUERY_GATE_WEIGHT
#define KERNEL_W1 SEISMIC_KEY_WEIGHT
#define KERNEL_W2 SEISMIC_VALUE_WEIGHT
#include "lib/projection/projection.h"

typedef element::Act activation;
typedef ELEMENT_OF(SEISMIC_INPUT_NORM) norm_element;

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
    projection::Rms<activation, norm_element, projection::AllRows> in{hidden, SEISMIC_HIDDEN_STRIDE_0, \
        SEISMIC_HIDDEN_STRIDE_1, input_norm, SEISMIC_INPUT_NORM_STRIDE_0,               \
        as_type<float>(uint(SEISMIC_PARAM_EPSILON)), k, {}};                            \
    projection::Weights<packets::W0> query_w{query_gate_weight, KERNEL_W0_LAYOUT(k), k}; \
    projection::Weights<packets::W1> key_w{key_weight, KERNEL_W1_LAYOUT(k), k};         \
    projection::Weights<packets::W2> value_w{value_weight, KERNEL_W2_LAYOUT(k), k};     \
    projection::Store<activation> query_out{query_gate, SEISMIC_RESULT_0_STRIDE_0,      \
        SEISMIC_RESULT_0_STRIDE_1, 0};                                                  \
    projection::Store<activation> key_out{key, SEISMIC_RESULT_1_STRIDE_0,               \
        SEISMIC_RESULT_1_STRIDE_1, 0};                                                  \
    projection::Store<activation> value_out{value, SEISMIC_RESULT_2_STRIDE_0,           \
        SEISMIC_RESULT_2_STRIDE_1, 0}

kernel void gated_attention_project_gemv(ATTENTION_PROJECT_ARGUMENTS,
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
        PROJECTION_FOR_ROWS(rows, projection::gemv<packets::W0, SG, R, MAXM, L>(
            x, query_out, query_w, rows, query_rows, k, tile, shared, sg, lane));
    } else if (tile < t0 + t1) {
        PROJECTION_FOR_ROWS(rows, projection::gemv<packets::W1, SG, R, MAXM, L>(
            x, key_out, key_w, rows, kv_rows, k, tile - t0, shared, sg, lane));
    } else {
        PROJECTION_FOR_ROWS(rows, projection::gemv<packets::W2, SG, R, MAXM, L>(
            x, value_out, value_w, rows, kv_rows, k, tile - t0 - t1, shared, sg, lane));
    }
}

kernel void gated_attention_project_batch(ATTENTION_PROJECT_ARGUMENTS,
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
        projection::gemv_batch<packets::W0, SG, R>(x, query_out, query_w, rows, query_rows, k, tile, shared, sg,
            lane);
    else if (tile < t0 + t1)
        projection::gemv_batch<packets::W1, SG, R>(x, key_out, key_w, rows, kv_rows, k, tile - t0, shared, sg,
            lane);
    else
        projection::gemv_batch<packets::W2, SG, R>(x, value_out, value_w, rows, kv_rows, k, tile - t0 - t1,
            shared, sg, lane);
}

kernel void gated_attention_project_stage(ATTENTION_PROJECT_ARGUMENTS,
    uint item [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    PROJECTION_NORMALIZE_SHARED(norms);
    ATTENTION_PROJECT_OPERANDS;
    projection::device_normalize<256>(in, item, normalized, k, norms, thread_index);
}

#define ATTENTION_PROJECT_GEMM(TM, TN)                                                  \
    PROJECTION_GEMM_SHARED(shared, TM, TN);                                             \
    ATTENTION_PROJECT_OPERANDS;                                                         \
    projection::Plain<activation, projection::AllRows> x{normalized, k, 1, k, {}};      \
    uint m = uint(SEISMIC_DIM_M);                                                       \
    uint t0 = (query_rows + TN - 1) / TN, t1 = (kv_rows + TN - 1) / TN;                 \
    uint n = tile.x;                                                                    \
    if (n < t0)                                                                         \
        projection::gemm<packets::W0, TM, TN>(x, query_out, query_w, m, query_rows, k, tile.y, n, shared, sg, lane); \
    else if (n < t0 + t1)                                                               \
        projection::gemm<packets::W1, TM, TN>(x, key_out, key_w, m, kv_rows, k, tile.y, n - t0, shared, sg, lane); \
    else                                                                                \
        projection::gemm<packets::W2, TM, TN>(x, value_out, value_w, m, kv_rows, k, tile.y, n - t0 - t1, shared, \
            sg, lane)

// 17..64 rows: the fixed small-row tile.
kernel void gated_attention_project_gemm_small(ATTENTION_PROJECT_ARGUMENTS,
    uint2 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    ATTENTION_PROJECT_GEMM(projection::small_tile_m, projection::small_tile_n);
}

kernel void gated_attention_project_gemm(ATTENTION_PROJECT_ARGUMENTS,
    uint2 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    ATTENTION_PROJECT_GEMM(SEISMIC_TUNE_TILE_M, SEISMIC_TUNE_TILE_N);
}
