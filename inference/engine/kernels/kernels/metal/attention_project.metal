// attention_project: RMS prologue over the F32 residual, then one
// segmented projection query | gate | key | value into four results (a
// segment of zero rows has no tiles).
#define KERNEL_W0 SEISMIC_QUERY_WEIGHT
#define KERNEL_W1 SEISMIC_GATE_WEIGHT
#define KERNEL_W2 SEISMIC_KEY_WEIGHT
#define KERNEL_W3 SEISMIC_VALUE_WEIGHT
#include "lib/projection/projection.h"

typedef element::Act activation;
typedef ELEMENT_OF(SEISMIC_INPUT_NORM) norm_element;

#define ATTENTION_PROJECT_ARGUMENTS                                                     \
    device const float *hidden [[buffer(SEISMIC_BUFFER_HIDDEN)]],                       \
    device const uchar *input_norm [[buffer(SEISMIC_BUFFER_INPUT_NORM)]],               \
    device const uchar *query_weight [[buffer(SEISMIC_BUFFER_QUERY_WEIGHT)]],           \
    device const uchar *gate_weight [[buffer(SEISMIC_BUFFER_GATE_WEIGHT)]],             \
    device const uchar *key_weight [[buffer(SEISMIC_BUFFER_KEY_WEIGHT)]],               \
    device const uchar *value_weight [[buffer(SEISMIC_BUFFER_VALUE_WEIGHT)]],           \
    device uchar *query [[buffer(SEISMIC_RESULT_0_BUFFER)]],                            \
    device uchar *gate [[buffer(SEISMIC_RESULT_1_BUFFER)]],                             \
    device uchar *key [[buffer(SEISMIC_RESULT_2_BUFFER)]],                              \
    device uchar *value [[buffer(SEISMIC_RESULT_3_BUFFER)]],                            \
    device uchar *normalized [[buffer(SEISMIC_BUFFER_SCRATCH_NORMALIZED)]],             \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

#define ATTENTION_PROJECT_OPERANDS                                                      \
    const uint k = uint(SEISMIC_DIM_D);                                                 \
    const uint query_rows = uint(SEISMIC_DIM_Q);                                        \
    const uint gate_rows = uint(SEISMIC_DIM_GR);                                        \
    const uint key_rows = uint(SEISMIC_DIM_K);                                          \
    const uint value_rows = uint(SEISMIC_DIM_V);                                        \
    projection::Rms<activation, norm_element, projection::AllRows> in{hidden, SEISMIC_HIDDEN_STRIDE_0, \
        SEISMIC_HIDDEN_STRIDE_1, input_norm, SEISMIC_INPUT_NORM_STRIDE_0,               \
        as_type<float>(uint(SEISMIC_PARAM_EPSILON)), k, {}};                            \
    projection::Weights<packets::W0> query_w{query_weight, KERNEL_W0_LAYOUT(k), k};     \
    projection::Weights<packets::W1> gate_w{gate_weight, KERNEL_W1_LAYOUT(k), k};       \
    projection::Weights<packets::W2> key_w{key_weight, KERNEL_W2_LAYOUT(k), k};         \
    projection::Weights<packets::W3> value_w{value_weight, KERNEL_W3_LAYOUT(k), k};     \
    projection::Store<activation> query_out{query, SEISMIC_RESULT_0_STRIDE_0,           \
        SEISMIC_RESULT_0_STRIDE_1, 0};                                                  \
    projection::Store<activation> gate_out{gate, SEISMIC_RESULT_1_STRIDE_0,             \
        SEISMIC_RESULT_1_STRIDE_1, 0};                                                  \
    projection::Store<activation> key_out{key, SEISMIC_RESULT_2_STRIDE_0,               \
        SEISMIC_RESULT_2_STRIDE_1, 0};                                                  \
    projection::Store<activation> value_out{value, SEISMIC_RESULT_3_STRIDE_0,           \
        SEISMIC_RESULT_3_STRIDE_1, 0}

template <typename Output>
inline void attention_project_zero_tile(Output out, uint rows, uint columns,
    uint row_start, uint row_count, uint column_start, uint column_count,
    uint thread_index, uint threads) {
    for (uint item = thread_index; item < row_count * column_count; item += threads) {
        uint row = row_start + item / column_count;
        uint column = column_start + item % column_count;
        if (row < rows && column < columns)
            out.store(row, column, 0.0f);
    }
}

#ifdef SEISMIC_FORMING_ATTENTION_PROJECT_GEMV
template <uint ROWS, uint LANES>
kernel void attention_project_gemv(ATTENTION_PROJECT_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    ATTENTION_PROJECT_OPERANDS;
    uint per = simdgroups * ROWS * (32u / LANES);
    uint rows = uint(SEISMIC_DIM_M);
    uint t0 = (query_rows + per - 1) / per, t1 = (gate_rows + per - 1) / per;
    uint t2 = (key_rows + per - 1) / per;
    if (SEISMIC_PARAM_PROJECT_MODE != 0 && tile < t0 + t1) {
        if (tile < t0)
            attention_project_zero_tile(query_out, rows, query_rows, 0, rows,
                tile * per, per, sg * 32 + lane, simdgroups * 32);
        else
            attention_project_zero_tile(gate_out, rows, gate_rows, 0, rows,
                (tile - t0) * per, per, sg * 32 + lane, simdgroups * 32);
        return;
    }
    PROJECTION_SQUARES_SHARED(squares, decltype(in)::parts);
    projection::threadgroup_squares_runtime(in, rows, squares, simdgroups, sg, lane);
    projection::SharedNorm<decltype(in)> x{in, squares};
    if (tile < t0) {
        PROJECTION_FOR_ROWS(rows, projection::gemv_runtime<packets::W0, ROWS, MAXM, LANES>(
            x, query_out, query_w, rows, query_rows, k, tile, shared, simdgroups, sg, lane));
    } else if (tile < t0 + t1) {
        PROJECTION_FOR_ROWS(rows, projection::gemv_runtime<packets::W1, ROWS, MAXM, LANES>(
            x, gate_out, gate_w, rows, gate_rows, k, tile - t0, shared, simdgroups, sg, lane));
    } else if (tile < t0 + t1 + t2) {
        PROJECTION_FOR_ROWS(rows, projection::gemv_runtime<packets::W2, ROWS, MAXM, LANES>(
            x, key_out, key_w, rows, key_rows, k, tile - t0 - t1, shared, simdgroups, sg, lane));
    } else {
        PROJECTION_FOR_ROWS(rows, projection::gemv_runtime<packets::W3, ROWS, MAXM, LANES>(
            x, value_out, value_w, rows, value_rows, k, tile - t0 - t1 - t2, shared, simdgroups, sg, lane));
    }
}
#endif

#ifdef SEISMIC_FORMING_ATTENTION_PROJECT_BATCH
template <uint BATCH_ROWS>
kernel void attention_project_batch(ATTENTION_PROJECT_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    ATTENTION_PROJECT_OPERANDS;
    uint per = simdgroups * BATCH_ROWS * 8u;
    uint rows = uint(SEISMIC_DIM_M);
    uint t0 = (query_rows + per - 1) / per, t1 = (gate_rows + per - 1) / per;
    uint t2 = (key_rows + per - 1) / per;
    if (SEISMIC_PARAM_PROJECT_MODE != 0 && tile < t0 + t1) {
        if (tile < t0)
            attention_project_zero_tile(query_out, rows, query_rows, 0, rows,
                tile * per, per, sg * 32 + lane, simdgroups * 32);
        else
            attention_project_zero_tile(gate_out, rows, gate_rows, 0, rows,
                (tile - t0) * per, per, sg * 32 + lane, simdgroups * 32);
        return;
    }
    PROJECTION_SQUARES_SHARED(squares, decltype(in)::parts);
    projection::threadgroup_squares_runtime(in, rows, squares, simdgroups, sg, lane);
    projection::SharedNorm<decltype(in)> x{in, squares};
    if (tile < t0)
        projection::gemv_batch_runtime<packets::W0, BATCH_ROWS>(x, query_out, query_w, rows, query_rows, k,
            tile, shared, simdgroups, sg, lane);
    else if (tile < t0 + t1)
        projection::gemv_batch_runtime<packets::W1, BATCH_ROWS>(x, gate_out, gate_w, rows, gate_rows, k,
            tile - t0, shared, simdgroups, sg, lane);
    else if (tile < t0 + t1 + t2)
        projection::gemv_batch_runtime<packets::W2, BATCH_ROWS>(x, key_out, key_w, rows, key_rows, k,
            tile - t0 - t1, shared, simdgroups, sg, lane);
    else
        projection::gemv_batch_runtime<packets::W3, BATCH_ROWS>(x, value_out, value_w, rows, value_rows, k,
            tile - t0 - t1 - t2, shared, simdgroups, sg, lane);
}
#endif

#ifdef SEISMIC_FORMING_ATTENTION_PROJECT_STAGE
kernel void attention_project_stage(ATTENTION_PROJECT_ARGUMENTS,
    uint item [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    PROJECTION_NORMALIZE_SHARED(norms);
    ATTENTION_PROJECT_OPERANDS;
    projection::device_normalize<256>(in, item, normalized, k, norms, thread_index);
}
#endif

#define ATTENTION_PROJECT_GEMM(TM, TN)                                                  \
    PROJECTION_GEMM_SHARED(shared, TM, TN);                                             \
    ATTENTION_PROJECT_OPERANDS;                                                         \
    projection::Plain<activation, projection::AllRows> x{normalized, k, 1, k, {}};      \
    uint m = uint(SEISMIC_DIM_M);                                                       \
    uint t0 = (query_rows + TN - 1) / TN, t1 = (gate_rows + TN - 1) / TN;               \
    uint t2 = (key_rows + TN - 1) / TN;                                                 \
    uint n = tile.x;                                                                    \
    if (SEISMIC_PARAM_PROJECT_MODE != 0 && n < t0 + t1) {                               \
        if (n < t0)                                                                     \
            attention_project_zero_tile(query_out, m, query_rows, tile.y * TM, TM,    \
                n * TN, TN, sg * 32 + lane, max(TM, 64u) * TN / 32u);                  \
        else                                                                            \
            attention_project_zero_tile(gate_out, m, gate_rows, tile.y * TM, TM,      \
                (n - t0) * TN, TN, sg * 32 + lane, max(TM, 64u) * TN / 32u);          \
        return;                                                                         \
    }                                                                                   \
    if (n < t0)                                                                         \
        projection::gemm<packets::W0, TM, TN>(x, query_out, query_w, m, query_rows, k, tile.y, n, shared, sg, lane); \
    else if (n < t0 + t1)                                                               \
        projection::gemm<packets::W1, TM, TN>(x, gate_out, gate_w, m, gate_rows, k, tile.y, n - t0, shared, sg, \
            lane);                                                                      \
    else if (n < t0 + t1 + t2)                                                          \
        projection::gemm<packets::W2, TM, TN>(x, key_out, key_w, m, key_rows, k, tile.y, n - t0 - t1, shared, sg, \
            lane);                                                                      \
    else                                                                                \
        projection::gemm<packets::W3, TM, TN>(x, value_out, value_w, m, value_rows, k, tile.y, n - t0 - t1 - t2, \
            shared, sg, lane)

// 17..64 rows: the fixed small-row tile.
#ifdef SEISMIC_FORMING_ATTENTION_PROJECT_GEMM_SMALL
kernel void attention_project_gemm_small(ATTENTION_PROJECT_ARGUMENTS,
    uint2 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    ATTENTION_PROJECT_GEMM(projection::small_tile_m, projection::small_tile_n);
}
#endif

#ifdef SEISMIC_FORMING_ATTENTION_PROJECT_GEMM
template <uint TILE_M, uint TILE_N>
kernel void attention_project_gemm(ATTENTION_PROJECT_ARGUMENTS,
    uint2 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    ATTENTION_PROJECT_GEMM(TILE_M, TILE_N);
}
#endif

// The TALL form past 64 rows: the normalized rows in the tall GEMM's order,
// then its tiles, 32 rows of one segment each.
#ifdef SEISMIC_FORMING_ATTENTION_PROJECT_STAGE_TALL
kernel void attention_project_stage_tall(ATTENTION_PROJECT_ARGUMENTS,
    uint item [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    PROJECTION_NORMALIZE_SHARED(norms);
    ATTENTION_PROJECT_OPERANDS;
    projection::device_normalize<256, projection::TallOrder<activation>>(in, item, normalized, k, norms,
        thread_index);
}
#endif

#ifdef SEISMIC_FORMING_ATTENTION_PROJECT_TALL
template <uint TALL_M, uint TALL_K, uint STAGERS>
kernel void attention_project_tall(ATTENTION_PROJECT_ARGUMENTS,
    uint2 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    PROJECTION_GEMM_TALL_SHARED(shared, TALL_K);
    ATTENTION_PROJECT_OPERANDS;
    const uint TN = projection::tall_n;
    const auto x = projection::tall_operand(
        projection::Plain<activation, projection::AllRows>{normalized, k, 1, k, {}}, normalized);
    uint m = uint(SEISMIC_DIM_M);
    uint t0 = (query_rows + TN - 1) / TN, t1 = (gate_rows + TN - 1) / TN;
    uint t2 = (key_rows + TN - 1) / TN;
    uint n = tile.x;
    if (SEISMIC_PARAM_PROJECT_MODE != 0 && n < t0 + t1) {
        if (n < t0)
            attention_project_zero_tile(query_out, m, query_rows, tile.y * TALL_M, TALL_M,
                n * TN, TN, sg * 32 + lane, (TALL_M / 32u + STAGERS) * 32u);
        else
            attention_project_zero_tile(gate_out, m, gate_rows, tile.y * TALL_M, TALL_M,
                (n - t0) * TN, TN, sg * 32 + lane, (TALL_M / 32u + STAGERS) * 32u);
        return;
    }
    if (n < t0)
        projection::gemm_tall<packets::W0, TALL_M, TALL_K, STAGERS>(x, query_out, query_w, m, query_rows, k,
            tile.y, n, shared, sg, lane);
    else if (n < t0 + t1)
        projection::gemm_tall<packets::W1, TALL_M, TALL_K, STAGERS>(x, gate_out, gate_w, m, gate_rows, k,
            tile.y, n - t0, shared, sg, lane);
    else if (n < t0 + t1 + t2)
        projection::gemm_tall<packets::W2, TALL_M, TALL_K, STAGERS>(x, key_out, key_w, m, key_rows, k,
            tile.y, n - t0 - t1, shared, sg, lane);
    else
        projection::gemm_tall<packets::W3, TALL_M, TALL_K, STAGERS>(x, value_out, value_w, m, value_rows, k,
            tile.y, n - t0 - t1 - t2, shared, sg, lane);
}
#endif
