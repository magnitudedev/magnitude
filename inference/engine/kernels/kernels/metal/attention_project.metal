// attention_project: RMS prologue over the F32 residual, then one
// segmented projection query | gate | key | value into four results (a
// segment of zero rows has no tiles).
#define KERNEL_W0 SEISMIC_QUERY_WEIGHT
#define KERNEL_W1 SEISMIC_GATE_WEIGHT
#define KERNEL_W2 SEISMIC_KEY_WEIGHT
#define KERNEL_W3 SEISMIC_VALUE_WEIGHT
#include "lib/projection/projection.h"
#include "lib/projection/packing.h"

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
    device float *packed [[buffer(SEISMIC_BUFFER_SCRATCH_PACKED)]],                     \
    device half *token_factors [[buffer(SEISMIC_BUFFER_SCRATCH_TOKEN_FACTORS)]],        \
    device half *weight_factors [[buffer(SEISMIC_BUFFER_SCRATCH_WEIGHT_FACTORS)]],      \
    device float *token_scales [[buffer(SEISMIC_BUFFER_SCRATCH_TOKEN_SCALES)]],         \
    device float *weight_scales [[buffer(SEISMIC_BUFFER_SCRATCH_WEIGHT_SCALES)]],       \
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

// The GEMV of a launch that serves COUNT (ONE, SEVERAL) rows.
#define ATTENTION_PROJECT_GEMV(ROWS, LANES, COUNT)                                      \
    ATTENTION_PROJECT_OPERANDS;                                                         \
    uint per = simdgroups * ROWS * (32u / LANES);                                       \
    uint rows = uint(SEISMIC_DIM_M);                                                    \
    uint t0 = (query_rows + per - 1) / per, t1 = (gate_rows + per - 1) / per;           \
    uint t2 = (key_rows + per - 1) / per;                                               \
    if (SEISMIC_PARAM_PROJECT_MODE != 0 && tile < t0 + t1) {                            \
        if (tile < t0)                                                                  \
            attention_project_zero_tile(query_out, rows, query_rows, 0, rows,           \
                tile * per, per, sg * 32 + lane, simdgroups * 32);                      \
        else                                                                            \
            attention_project_zero_tile(gate_out, rows, gate_rows, 0, rows,             \
                (tile - t0) * per, per, sg * 32 + lane, simdgroups * 32);               \
        return;                                                                         \
    }                                                                                   \
    PROJECTION_SQUARES_SHARED(squares, decltype(in)::parts);                            \
    projection::threadgroup_squares_runtime(in, rows, squares, simdgroups, sg, lane);   \
    const auto x = projection::shared_norm(in, squares);                                \
    if (tile < t0) {                                                                    \
        PROJECTION_FOR_##COUNT##_ROWS(rows, projection::gemv_runtime<packets::W0, ROWS, MAXM, LANES>( \
            x, query_out, query_w, rows, query_rows, k, tile, shared, simdgroups, sg, lane)); \
    } else if (tile < t0 + t1) {                                                        \
        PROJECTION_FOR_##COUNT##_ROWS(rows, projection::gemv_runtime<packets::W1, ROWS, MAXM, LANES>( \
            x, gate_out, gate_w, rows, gate_rows, k, tile - t0, shared, simdgroups, sg, lane)); \
    } else if (tile < t0 + t1 + t2) {                                                   \
        PROJECTION_FOR_##COUNT##_ROWS(rows, projection::gemv_runtime<packets::W2, ROWS, MAXM, LANES>( \
            x, key_out, key_w, rows, key_rows, k, tile - t0 - t1, shared, simdgroups, sg, lane)); \
    } else {                                                                            \
        PROJECTION_FOR_##COUNT##_ROWS(rows, projection::gemv_runtime<packets::W3, ROWS, MAXM, LANES>( \
            x, value_out, value_w, rows, value_rows, k, tile - t0 - t1 - t2, shared, simdgroups, sg, lane)); \
    }

// One row.
#ifdef SEISMIC_FORMING_ATTENTION_PROJECT_GEMV
template <uint ROWS, uint LANES>
kernel void attention_project_gemv(ATTENTION_PROJECT_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    ATTENTION_PROJECT_GEMV(ROWS, LANES, ONE)
}
#endif

// Three rows up to BATCH_FROM: the same GEMV under this launch's mapping.
#ifdef SEISMIC_FORMING_ATTENTION_PROJECT_GEMV_ROWS
template <uint ROWS, uint LANES>
kernel void attention_project_gemv_rows(ATTENTION_PROJECT_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    ATTENTION_PROJECT_GEMV(ROWS, LANES, SEVERAL)
}
#endif

// Two rows: the same GEMV under this launch's mapping.
#ifdef SEISMIC_FORMING_ATTENTION_PROJECT_GEMV_PAIR
template <uint ROWS, uint LANES>
kernel void attention_project_gemv_pair(ATTENTION_PROJECT_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    ATTENTION_PROJECT_GEMV(ROWS, LANES, PAIR)
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
    const auto x = projection::shared_norm(in, squares);
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

// The staged tiles of the segments; QUERY, GATE, KEY and VALUE say whether
// these tiles run that segment (the PACK form's own tiles run the ones it
// takes; the zero tiles of mode 1 are always these).
#define ATTENTION_PROJECT_GEMM_SEGMENTS(TM, TN, QUERY, GATE, KEY, VALUE)                \
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
    if (!(n < t0 ? QUERY : n < t0 + t1 ? GATE : n < t0 + t1 + t2 ? KEY : VALUE))        \
        return;                                                                         \
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

#define ATTENTION_PROJECT_GEMM(TM, TN) ATTENTION_PROJECT_GEMM_SEGMENTS(TM, TN, true, true, true, true)

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

// The PACK form past 64 rows: the normalized rows (`attention_project_stage`)
// as integer codes under the gain limits of the four weights together, two
// rows packed per operand element, the weights' block scales and biases (in
// segment order in each table), then the packed tiles of the segments. The
// operand is laid out for the first weight with a packed path; a segment
// whose weights it does not serve runs the staged tiles (`unpacked`), with
// their results, as do the zero tiles of mode 1.
#define ATTENTION_PROJECT_PACKING_SCRATCH(first)                                        \
    projection::packing_scratch{packed, token_factors, token_scales,                    \
        weight_factors + ulong(first) * (k / 32u) * 2u, weight_scales + (first)}
#define ATTENTION_PROJECT_PACKING                                                       \
    projection::packing_shared<packets::W0, packets::W1, packets::W2, packets::W3>
#define ATTENTION_PROJECT_PACKED(SLOT)                                                  \
    projection::packing_serves<packets::SLOT, ATTENTION_PROJECT_PACKING::folds>::value

#ifdef SEISMIC_FORMING_ATTENTION_PROJECT_PACK
kernel void attention_project_pack(ATTENTION_PROJECT_ARGUMENTS,
    uint pairs [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    if constexpr (ATTENTION_PROJECT_PACKING::folds == 0)
        return;
    threadgroup float4 peaks[8];
    ATTENTION_PROJECT_OPERANDS;
    projection::Plain<activation, projection::AllRows> x{normalized, k, 1, k, {}};
    projection::packing_operand<ATTENTION_PROJECT_PACKING::centre, ATTENTION_PROJECT_PACKING::folds>(x,
        ATTENTION_PROJECT_PACKING_SCRATCH(0u), uint(SEISMIC_DIM_M), k, pairs, peaks, thread_index, sg, lane);
}
#endif

#ifdef SEISMIC_FORMING_ATTENTION_PROJECT_PACK_COEFFICIENTS
kernel void attention_project_pack_coefficients(ATTENTION_PROJECT_ARGUMENTS,
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    ATTENTION_PROJECT_OPERANDS;
    const uint r1 = query_rows, r2 = r1 + gate_rows, r3 = r2 + key_rows;
    if (row < r1) {
        if constexpr (ATTENTION_PROJECT_PACKED(W0))
            projection::packing_coefficients(query_w, ATTENTION_PROJECT_PACKING_SCRATCH(0u), query_rows, k, row,
                lane);
    } else if (row < r2) {
        if constexpr (ATTENTION_PROJECT_PACKED(W1))
            projection::packing_coefficients(gate_w, ATTENTION_PROJECT_PACKING_SCRATCH(r1), gate_rows, k, row - r1,
                lane);
    } else if (row < r3) {
        if constexpr (ATTENTION_PROJECT_PACKED(W2))
            projection::packing_coefficients(key_w, ATTENTION_PROJECT_PACKING_SCRATCH(r2), key_rows, k, row - r2,
                lane);
    } else {
        if constexpr (ATTENTION_PROJECT_PACKED(W3))
            projection::packing_coefficients(value_w, ATTENTION_PROJECT_PACKING_SCRATCH(r3), value_rows, k, row - r3,
                lane);
    }
}
#endif

#ifdef SEISMIC_FORMING_ATTENTION_PROJECT_PACKED
template <uint PACK_TOKENS, uint WEIGHTS_AHEAD>
kernel void attention_project_packed(ATTENTION_PROJECT_ARGUMENTS,
    uint tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    ATTENTION_PROJECT_OPERANDS;
    const uint m = uint(SEISMIC_DIM_M), padded = (m + 127u) / 128u * 128u;
    const uint r1 = query_rows, r2 = r1 + gate_rows, r3 = r2 + key_rows;
    // Mode 1 has no query or gate tiles (`unpacked` zeroes those results).
    const bool projected = SEISMIC_PARAM_PROJECT_MODE == 0;
    uint group = tile;
    constexpr uint FOLDS = ATTENTION_PROJECT_PACKING::folds;
    if (projection::gemm_packed_segment<packets::W0, FOLDS, PACK_TOKENS, WEIGHTS_AHEAD>(query_out,
            query_w, ATTENTION_PROJECT_PACKING_SCRATCH(0u), projected ? m : 0u, padded, query_rows, k, group,
            projection::packing_simdgroups, sg, lane))
        return;
    if (projection::gemm_packed_segment<packets::W1, FOLDS, PACK_TOKENS, WEIGHTS_AHEAD>(gate_out,
            gate_w, ATTENTION_PROJECT_PACKING_SCRATCH(r1), projected ? m : 0u, padded, gate_rows, k, group,
            projection::packing_simdgroups, sg, lane))
        return;
    if (projection::gemm_packed_segment<packets::W2, FOLDS, PACK_TOKENS, WEIGHTS_AHEAD>(key_out,
            key_w, ATTENTION_PROJECT_PACKING_SCRATCH(r2), m, padded, key_rows, k, group,
            projection::packing_simdgroups, sg, lane))
        return;
    projection::gemm_packed_segment<packets::W3, FOLDS, PACK_TOKENS, WEIGHTS_AHEAD>(value_out,
        value_w, ATTENTION_PROJECT_PACKING_SCRATCH(r3), m, padded, value_rows, k, group,
        projection::packing_simdgroups, sg, lane);
}
#endif

#ifdef SEISMIC_FORMING_ATTENTION_PROJECT_UNPACKED
kernel void attention_project_unpacked(ATTENTION_PROJECT_ARGUMENTS,
    uint2 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    ATTENTION_PROJECT_GEMM_SEGMENTS(64u, 64u, !ATTENTION_PROJECT_PACKED(W0), !ATTENTION_PROJECT_PACKED(W1),
        !ATTENTION_PROJECT_PACKED(W2), !ATTENTION_PROJECT_PACKED(W3));
}
#endif
