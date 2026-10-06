// gated_delta_project: RMS prologue over the F32 residual, then one
// segmented projection qkv | z | alpha | beta, each segment with its own
// representation, into one activation row per input row.
#define KERNEL_W0 SEISMIC_QKV_WEIGHT
#define KERNEL_W1 SEISMIC_GATE_WEIGHT
#define KERNEL_W2 SEISMIC_ALPHA_WEIGHT
#define KERNEL_W3 SEISMIC_BETA_WEIGHT
#include "lib/projection/projection.h"
#include "lib/projection/packing.h"

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
    device float *packed [[buffer(SEISMIC_BUFFER_SCRATCH_PACKED)]],                     \
    device half *token_factors [[buffer(SEISMIC_BUFFER_SCRATCH_TOKEN_FACTORS)]],        \
    device half *weight_factors [[buffer(SEISMIC_BUFFER_SCRATCH_WEIGHT_FACTORS)]],      \
    device float *token_scales [[buffer(SEISMIC_BUFFER_SCRATCH_TOKEN_SCALES)]],         \
    device float *weight_scales [[buffer(SEISMIC_BUFFER_SCRATCH_WEIGHT_SCALES)]],       \
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

#ifdef SEISMIC_FORMING_GATED_DELTA_PROJECT_GEMV
template <uint ROWS, uint LANES>
kernel void gated_delta_project_gemv(RECURRENT_PROJECT_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    RECURRENT_PROJECT_OPERANDS;
    uint per = simdgroups * ROWS * (32u / LANES);
    uint rows = uint(SEISMIC_DIM_M);
    PROJECTION_SQUARES_SHARED(squares, decltype(in)::parts);
    projection::threadgroup_squares_runtime(in, rows, squares, simdgroups, sg, lane);
    const auto x = projection::shared_norm(in, squares);
    uint t0 = (qkv_rows + per - 1) / per, t1 = (gate_rows + per - 1) / per;
    uint t2 = (head_rows + per - 1) / per;
    if (tile < t0) {
        PROJECTION_FOR_ROWS(rows, projection::gemv_runtime<packets::W0, ROWS, MAXM, LANES>(
            x, qkv_out, qkv, rows, qkv_rows, k, tile, shared, simdgroups, sg, lane));
    } else if (tile < t0 + t1) {
        PROJECTION_FOR_ROWS(rows, projection::gemv_runtime<packets::W1, ROWS, MAXM, LANES>(
            x, gate_out, gate, rows, gate_rows, k, tile - t0, shared, simdgroups, sg, lane));
    } else if (tile < t0 + t1 + t2) {
        PROJECTION_FOR_ROWS(rows, projection::gemv_runtime<packets::W2, ROWS, MAXM, LANES>(
            x, alpha_out, alpha, rows, head_rows, k, tile - t0 - t1, shared, simdgroups, sg, lane));
    } else {
        PROJECTION_FOR_ROWS(rows, projection::gemv_runtime<packets::W3, ROWS, MAXM, LANES>(
            x, beta_out, beta, rows, head_rows, k, tile - t0 - t1 - t2, shared, simdgroups, sg, lane));
    }
}
#endif

#ifdef SEISMIC_FORMING_GATED_DELTA_PROJECT_BATCH
template <uint BATCH_ROWS>
kernel void gated_delta_project_batch(RECURRENT_PROJECT_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    RECURRENT_PROJECT_OPERANDS;
    uint per = simdgroups * BATCH_ROWS * 8u;
    uint rows = uint(SEISMIC_DIM_M);
    PROJECTION_SQUARES_SHARED(squares, decltype(in)::parts);
    projection::threadgroup_squares_runtime(in, rows, squares, simdgroups, sg, lane);
    const auto x = projection::shared_norm(in, squares);
    uint t0 = (qkv_rows + per - 1) / per, t1 = (gate_rows + per - 1) / per;
    uint t2 = (head_rows + per - 1) / per;
    if (tile < t0)
        projection::gemv_batch_runtime<packets::W0, BATCH_ROWS>(x, qkv_out, qkv, rows, qkv_rows, k,
            tile, shared, simdgroups, sg, lane);
    else if (tile < t0 + t1)
        projection::gemv_batch_runtime<packets::W1, BATCH_ROWS>(x, gate_out, gate, rows, gate_rows, k,
            tile - t0, shared, simdgroups, sg, lane);
    else if (tile < t0 + t1 + t2)
        projection::gemv_batch_runtime<packets::W2, BATCH_ROWS>(x, alpha_out, alpha, rows, head_rows, k,
            tile - t0 - t1, shared, simdgroups, sg, lane);
    else
        projection::gemv_batch_runtime<packets::W3, BATCH_ROWS>(x, beta_out, beta, rows, head_rows, k,
            tile - t0 - t1 - t2, shared, simdgroups, sg, lane);
}
#endif

#ifdef SEISMIC_FORMING_GATED_DELTA_PROJECT_STAGE
kernel void gated_delta_project_stage(RECURRENT_PROJECT_ARGUMENTS,
    uint item [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    PROJECTION_NORMALIZE_SHARED(norms);
    RECURRENT_PROJECT_OPERANDS;
    projection::device_normalize<256>(in, item, normalized, k, norms, thread_index);
}
#endif

// The staged tiles of the segments; QKV and GATE say whether these tiles run
// the qkv and the z segment (the PACK form's own tiles run the ones it
// takes).
#define RECURRENT_PROJECT_GEMM_SEGMENTS(TM, TN, QKV, GATE)                              \
    PROJECTION_GEMM_SHARED(shared, TM, TN);                                             \
    RECURRENT_PROJECT_OPERANDS;                                                         \
    projection::Plain<activation, projection::AllRows> x{normalized, k, 1, k, {}};      \
    uint m = uint(SEISMIC_DIM_M);                                                       \
    uint t0 = (qkv_rows + TN - 1) / TN, t1 = (gate_rows + TN - 1) / TN, t2 = (head_rows + TN - 1) / TN; \
    uint n = tile.x;                                                                    \
    if (n < t0 + t1 && !(n < t0 ? QKV : GATE))                                          \
        return;                                                                         \
    if (n < t0)                                                                         \
        projection::gemm<packets::W0, TM, TN>(x, qkv_out, qkv, m, qkv_rows, k, tile.y, n, shared, sg, lane); \
    else if (n < t0 + t1)                                                               \
        projection::gemm<packets::W1, TM, TN>(x, gate_out, gate, m, gate_rows, k, tile.y, n - t0, shared, sg, lane); \
    else if (n < t0 + t1 + t2)                                                          \
        projection::gemm<packets::W2, TM, TN>(x, alpha_out, alpha, m, head_rows, k, tile.y, n - t0 - t1, shared, \
            sg, lane);                                                                  \
    else                                                                                \
        projection::gemm<packets::W3, TM, TN>(x, beta_out, beta, m, head_rows, k, tile.y, n - t0 - t1 - t2, \
            shared, sg, lane)

#define RECURRENT_PROJECT_GEMM(TM, TN) RECURRENT_PROJECT_GEMM_SEGMENTS(TM, TN, true, true)

// 17..64 rows: the fixed small-row tile.
#ifdef SEISMIC_FORMING_GATED_DELTA_PROJECT_GEMM_SMALL
kernel void gated_delta_project_gemm_small(RECURRENT_PROJECT_ARGUMENTS,
    uint2 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    RECURRENT_PROJECT_GEMM(projection::small_tile_m, projection::small_tile_n);
}
#endif

#ifdef SEISMIC_FORMING_GATED_DELTA_PROJECT_GEMM
template <uint TILE_M, uint TILE_N>
kernel void gated_delta_project_gemm(RECURRENT_PROJECT_ARGUMENTS,
    uint2 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    RECURRENT_PROJECT_GEMM(TILE_M, TILE_N);
}
#endif

// The TALL form past 64 rows: the normalized rows in the tall GEMM's order,
// then its tiles, 32 rows of one segment each.
#ifdef SEISMIC_FORMING_GATED_DELTA_PROJECT_STAGE_TALL
kernel void gated_delta_project_stage_tall(RECURRENT_PROJECT_ARGUMENTS,
    uint item [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    PROJECTION_NORMALIZE_SHARED(norms);
    RECURRENT_PROJECT_OPERANDS;
    projection::device_normalize<256, projection::TallOrder<activation>>(in, item, normalized, k, norms,
        thread_index);
}
#endif

#ifdef SEISMIC_FORMING_GATED_DELTA_PROJECT_TALL
template <uint TALL_M, uint TALL_K, uint STAGERS>
kernel void gated_delta_project_tall(RECURRENT_PROJECT_ARGUMENTS,
    uint2 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    PROJECTION_GEMM_TALL_SHARED(shared, TALL_K);
    RECURRENT_PROJECT_OPERANDS;
    const uint TN = projection::tall_n;
    const auto x = projection::tall_operand(
        projection::Plain<activation, projection::AllRows>{normalized, k, 1, k, {}}, normalized);
    uint m = uint(SEISMIC_DIM_M);
    uint t0 = (qkv_rows + TN - 1) / TN, t1 = (gate_rows + TN - 1) / TN, t2 = (head_rows + TN - 1) / TN;
    uint n = tile.x;
    if (n < t0)
        projection::gemm_tall<packets::W0, TALL_M, TALL_K, STAGERS>(x, qkv_out, qkv, m, qkv_rows, k, tile.y, n,
            shared, sg, lane);
    else if (n < t0 + t1)
        projection::gemm_tall<packets::W1, TALL_M, TALL_K, STAGERS>(x, gate_out, gate, m, gate_rows, k, tile.y,
            n - t0, shared, sg, lane);
    else if (n < t0 + t1 + t2)
        projection::gemm_tall<packets::W2, TALL_M, TALL_K, STAGERS>(x, alpha_out, alpha, m, head_rows, k, tile.y,
            n - t0 - t1, shared, sg, lane);
    else
        projection::gemm_tall<packets::W3, TALL_M, TALL_K, STAGERS>(x, beta_out, beta, m, head_rows, k, tile.y,
            n - t0 - t1 - t2, shared, sg, lane);
}
#endif

// The PACK form past 64 rows: the normalized rows (`gated_delta_project_stage`)
// as integer codes under the gain limits of the qkv and z weights together,
// two rows packed per operand element, those weights' block scales and biases
// (the z weights' after the qkv weights' in each table), then the packed
// tiles of those two segments. The operand is laid out for the first of the
// two weights with a packed path; a segment whose weights it does not serve
// and the alpha and beta segments run the staged tiles (`unpacked`), with
// their results.
#define RECURRENT_PROJECT_PACKING_SCRATCH                                               \
    const projection::packing_scratch qkv_scratch{packed, token_factors, token_scales, weight_factors, \
        weight_scales};                                                                 \
    const projection::packing_scratch gate_scratch{packed, token_factors, token_scales, \
        weight_factors + ulong(qkv_rows) * (k / 32u) * 2u, weight_scales + qkv_rows}
#define RECURRENT_PROJECT_PACKING projection::packing_shared<packets::W0, packets::W1>
#define RECURRENT_PROJECT_PACKED(SLOT)                                                  \
    projection::packing_serves<packets::SLOT, RECURRENT_PROJECT_PACKING::folds>::value

#ifdef SEISMIC_FORMING_GATED_DELTA_PROJECT_PACK
kernel void gated_delta_project_pack(RECURRENT_PROJECT_ARGUMENTS,
    uint pairs [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    if constexpr (RECURRENT_PROJECT_PACKING::folds == 0)
        return;
    threadgroup float4 peaks[8];
    RECURRENT_PROJECT_OPERANDS;
    RECURRENT_PROJECT_PACKING_SCRATCH;
    projection::Plain<activation, projection::AllRows> x{normalized, k, 1, k, {}};
    projection::packing_operand<RECURRENT_PROJECT_PACKING::centre, RECURRENT_PROJECT_PACKING::folds>(x,
        qkv_scratch, uint(SEISMIC_DIM_M), k, pairs, peaks, thread_index, sg, lane);
}
#endif

#ifdef SEISMIC_FORMING_GATED_DELTA_PROJECT_PACK_COEFFICIENTS
kernel void gated_delta_project_pack_coefficients(RECURRENT_PROJECT_ARGUMENTS,
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    RECURRENT_PROJECT_OPERANDS;
    RECURRENT_PROJECT_PACKING_SCRATCH;
    if (row < qkv_rows) {
        if constexpr (RECURRENT_PROJECT_PACKED(W0))
            projection::packing_coefficients(qkv, qkv_scratch, qkv_rows, k, row, lane);
    } else {
        if constexpr (RECURRENT_PROJECT_PACKED(W1))
            projection::packing_coefficients(gate, gate_scratch, gate_rows, k, row - qkv_rows, lane);
    }
}
#endif

#ifdef SEISMIC_FORMING_GATED_DELTA_PROJECT_PACKED
template <uint PACK_TOKENS, uint WEIGHTS_AHEAD>
kernel void gated_delta_project_packed(RECURRENT_PROJECT_ARGUMENTS,
    uint tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    RECURRENT_PROJECT_OPERANDS;
    RECURRENT_PROJECT_PACKING_SCRATCH;
    const uint m = uint(SEISMIC_DIM_M), padded = (m + 127u) / 128u * 128u;
    uint group = tile;
    constexpr uint FOLDS = RECURRENT_PROJECT_PACKING::folds;
    if (projection::gemm_packed_segment<packets::W0, FOLDS, PACK_TOKENS, WEIGHTS_AHEAD>(qkv_out, qkv,
            qkv_scratch, m, padded, qkv_rows, k, group, projection::packing_simdgroups, sg, lane))
        return;
    projection::gemm_packed_segment<packets::W1, FOLDS, PACK_TOKENS, WEIGHTS_AHEAD>(gate_out, gate,
        gate_scratch, m, padded, gate_rows, k, group, projection::packing_simdgroups, sg, lane);
}
#endif

#ifdef SEISMIC_FORMING_GATED_DELTA_PROJECT_UNPACKED
kernel void gated_delta_project_unpacked(RECURRENT_PROJECT_ARGUMENTS,
    uint2 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    RECURRENT_PROJECT_GEMM_SEGMENTS(64, 64, !RECURRENT_PROJECT_PACKED(W0), !RECURRENT_PROJECT_PACKED(W1));
}
#endif
