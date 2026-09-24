// qwen_recurrent_project: RMS prologue over the F32 residual, then one
// segmented projection qkv | z | alpha | beta, each segment with its own
// representation, into one activation row per input row.
#define KERNEL_W0 SEISMIC_QKV_WEIGHT
#define KERNEL_W1 SEISMIC_GATE_WEIGHT
#define KERNEL_W2 SEISMIC_ALPHA_WEIGHT
#define KERNEL_W3 SEISMIC_BETA_WEIGHT
#include "common/projection.h"

typedef element::Act activation;
typedef ELEMENT_OF(SEISMIC_INPUT_NORM) norm_element;

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
    projection::Rms<activation, norm_element, projection::AllRows> in{hidden, SEISMIC_HIDDEN_STRIDE_0, \
        SEISMIC_HIDDEN_STRIDE_1, input_norm, SEISMIC_INPUT_NORM_STRIDE_0,               \
        as_type<float>(uint(SEISMIC_PARAM_EPSILON)), k, {}};                            \
    projection::Weights<packets::W0> qkv{qkv_weight, KERNEL_W0_LAYOUT(k), k};           \
    projection::Weights<packets::W1> gate{gate_weight, KERNEL_W1_LAYOUT(k), k};         \
    projection::Weights<packets::W2> alpha{alpha_weight, KERNEL_W2_LAYOUT(k), k};       \
    projection::Weights<packets::W3> beta{beta_weight, KERNEL_W3_LAYOUT(k), k};         \
    projection::Store<activation> qkv_out{result, SEISMIC_RESULT_0_STRIDE_0,            \
        SEISMIC_RESULT_0_STRIDE_1, 0};                                                  \
    projection::Store<activation> gate_out{result, SEISMIC_RESULT_0_STRIDE_0,           \
        SEISMIC_RESULT_0_STRIDE_1, qkv_rows};                                           \
    projection::Store<activation> alpha_out{result, SEISMIC_RESULT_0_STRIDE_0,          \
        SEISMIC_RESULT_0_STRIDE_1, qkv_rows + gate_rows};                               \
    projection::Store<activation> beta_out{result, SEISMIC_RESULT_0_STRIDE_0,           \
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
        PROJECTION_FOR_ROWS(rows, projection::gemv<packets::W0, SG, R, MAXM, L>(
            x, qkv_out, qkv, rows, qkv_rows, k, tile, shared, sg, lane));
    } else if (tile < t0 + t1) {
        PROJECTION_FOR_ROWS(rows, projection::gemv<packets::W1, SG, R, MAXM, L>(
            x, gate_out, gate, rows, gate_rows, k, tile - t0, shared, sg, lane));
    } else if (tile < t0 + t1 + t2) {
        PROJECTION_FOR_ROWS(rows, projection::gemv<packets::W2, SG, R, MAXM, L>(
            x, alpha_out, alpha, rows, head_rows, k, tile - t0 - t1, shared, sg, lane));
    } else {
        PROJECTION_FOR_ROWS(rows, projection::gemv<packets::W3, SG, R, MAXM, L>(
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
        projection::gemv_batch<packets::W0, SG, R>(x, qkv_out, qkv, rows, qkv_rows, k, tile, shared, sg, lane);
    else if (tile < t0 + t1)
        projection::gemv_batch<packets::W1, SG, R>(x, gate_out, gate, rows, gate_rows, k, tile - t0, shared, sg,
            lane);
    else if (tile < t0 + t1 + t2)
        projection::gemv_batch<packets::W2, SG, R>(x, alpha_out, alpha, rows, head_rows, k, tile - t0 - t1,
            shared, sg, lane);
    else
        projection::gemv_batch<packets::W3, SG, R>(x, beta_out, beta, rows, head_rows, k, tile - t0 - t1 - t2,
            shared, sg, lane);
}

kernel void qwen_recurrent_project_stage(RECURRENT_PROJECT_ARGUMENTS,
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
    projection::Plain<activation, projection::AllRows> x{normalized, k, 1, k, {}};
    uint m = uint(SEISMIC_DIM_M);
    uint t0 = (qkv_rows + TN - 1) / TN, t1 = (gate_rows + TN - 1) / TN, t2 = (head_rows + TN - 1) / TN;
    uint n = tile.x;
    if (n < t0)
        projection::gemm<packets::W0, TM, TN>(x, qkv_out, qkv, m, qkv_rows, k, tile.y, n, shared, sg, lane);
    else if (n < t0 + t1)
        projection::gemm<packets::W1, TM, TN>(x, gate_out, gate, m, gate_rows, k, tile.y, n - t0, shared, sg, lane);
    else if (n < t0 + t1 + t2)
        projection::gemm<packets::W2, TM, TN>(x, alpha_out, alpha, m, head_rows, k, tile.y, n - t0 - t1, shared, sg, lane);
    else
        projection::gemm<packets::W3, TM, TN>(x, beta_out, beta, m, head_rows, k, tile.y, n - t0 - t1 - t2, shared, sg, lane);
}
