#define ATTENTION_QUERY_GROUP SEISMIC_DIM_G
#define ATTENTION_I SEISMIC_DIM_I
#define ATTENTION_U SEISMIC_DIM_U
#define ATTENTION_FRESH (SEISMIC_DIM_F != 0)
#define ATTENTION_NORM (SEISMIC_DIM_N != 0)
#define ATTENTION_VALUE_NORM (SEISMIC_DIM_NV != 0)
#define PREFILL_HEADS_PER_GROUP SEISMIC_TUNE_HEADS
#define PREFILL_DIRECT SEISMIC_TUNE_DIRECT
#define PREFILL_COISSUE SEISMIC_TUNE_COISSUE
#include "lib/attention/attention.h"

// The launches over affine K8/V4 history (bodies in lib/attention/attention.h):
// the prepare launch appends encoded rows, and every history K/V tile is
// decoded to F16 as it is staged; every product takes F16 operands
// (attention::affine_history). The scratch rows are F16. Under DIRECT the
// call runs in rounds: the `decode` launch decodes the round's part of the
// history row tiles the batch sees (`history_tiles`) to F16 in the cells of
// `partials` the key partitions leave (attention::history_window), the
// attend launch reads its tiles from there (attention::decoded_history), and
// when the call takes several rounds the `fold` launch folds each round's
// split records into the state the window keeps. Under COISSUE on simdgroup
// matrices the attend launch pairs its simdgroups, one forming Q K^T on the
// matrix pipe and one P V as scalar F16 products (attention::prefill_coissue),
// over the same decoded history; a 512-column head takes a pair per
// 256-column window.

#if SEISMIC_TUNE_DIRECT
// The window of a call that lists `listed` tiles, placed by the terms of
// the entry's Metal declaration in attention.seismic, which these must equal:
// `taken` is TAKEN (the attend grid's key partitions) and `charged` is
// CHARGED; attention::history_window derives SPARE, STATE, WINDOW and ROUND
// from them.
static inline attention::history_window prefill_window(device float *partials, uint M, uint listed) {
    const uint tiles = (M + SEISMIC_TUNE_QT - 1) / SEISMIC_TUNE_QT * SEISMIC_DIM_KV * PREFILL_HEAD_GROUPS;
    const uint split = metal::max(1u, (SEISMIC_TUNE_SPLIT_GROUPS + tiles - 1) / tiles);
    const uint most = ((M + 31) / 32) * SEISMIC_DIM_KV * ((SEISMIC_DIM_G + 15) / 16);
    const uint charged = (512 + most - 1) / most;
    const uint taken = metal::min(split, metal::max(1u, charged / 2));
    return attention::history_window::in(partials, M, taken, charged, listed);
}
#endif

kernel void attention_prefill_k8v4_prepare(
    device const attention::Scalar *query [[buffer(SEISMIC_BUFFER_QUERY)]],
    device const attention::Scalar *key [[buffer(SEISMIC_BUFFER_KEY)]],
    device const attention::Scalar *value [[buffer(SEISMIC_BUFFER_VALUE)]],
    device const float *query_norm [[buffer(SEISMIC_BUFFER_QUERY_NORM)]],
    device const float *key_norm [[buffer(SEISMIC_BUFFER_KEY_NORM)]],
    device const float *value_norm [[buffer(SEISMIC_BUFFER_VALUE_NORM)]],
    device const int *rotary_components [[buffer(SEISMIC_BUFFER_ROTARY_COMPONENTS)]],
    device const float *rotary_frequencies [[buffer(SEISMIC_BUFFER_ROTARY_FREQUENCIES)]],
    device const float *rotary_amplitudes [[buffer(SEISMIC_BUFFER_ROTARY_AMPLITUDES)]],
    device const int *coordinates [[buffer(SEISMIC_BUFFER_COORDINATES)]],
    device const int *destinations [[buffer(SEISMIC_BUFFER_DESTINATIONS)]],
    device const ulong *key_codes [[buffer(SEISMIC_BUFFER_HISTORY_KEY_CODES)]],
    device const ulong *key_coefficients [[buffer(SEISMIC_BUFFER_HISTORY_KEY_COEFFICIENTS)]],
    device const ulong *value_codes [[buffer(SEISMIC_BUFFER_HISTORY_VALUE_CODES)]],
    device const ulong *value_coefficients [[buffer(SEISMIC_BUFFER_HISTORY_VALUE_COEFFICIENTS)]],
    device half *queries [[buffer(SEISMIC_BUFFER_SCRATCH_QUERIES)]],
    device half *keys [[buffer(SEISMIC_BUFFER_SCRATCH_KEYS)]],
    device half *values [[buffer(SEISMIC_BUFFER_SCRATCH_VALUES)]],
#if SEISMIC_TUNE_DIRECT
    device const int *tiles [[buffer(SEISMIC_BUFFER_HISTORY_TILES)]],
    device float *partials [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIALS)]],
#endif
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
#if SEISMIC_TUNE_DIRECT
    // The call's first round, and the tiles it lists.
    if (group == 0 && simd == 0 && lane == 0) {
        const attention::history_window window = prefill_window(partials, uint(SEISMIC_DIM_M), uint(SEISMIC_DIM_HT));
        window.words[attention::history_window::DECODE] = 0;
        window.words[attention::history_window::TILES] =
            attention::held_tiles::listed(tiles, uint(SEISMIC_DIM_HT)).count;
    }
#endif
    attention::prefill_prepare<SEISMIC_TUNE_QT>(
        attention::affine_history{key_codes, key_coefficients, value_codes, value_coefficients,
            ulong(SEISMIC_PARAM_SLAB_ROWS)},
        query, key, value, query_norm, key_norm, value_norm, rotary_components, rotary_frequencies,
        rotary_amplitudes, coordinates, destinations, queries, keys, values, SEISMIC_DIM_M,
        as_type<float>(uint(SEISMIC_PARAM_EPSILON)), int(uint(SEISMIC_PARAM_GATE_FUNCTION)) == -1,
        group, simd, lane);
}

kernel void attention_prefill_k8v4_decode(
    device const ulong *key_codes [[buffer(SEISMIC_BUFFER_HISTORY_KEY_CODES)]],
    device const ulong *key_coefficients [[buffer(SEISMIC_BUFFER_HISTORY_KEY_COEFFICIENTS)]],
    device const ulong *value_codes [[buffer(SEISMIC_BUFFER_HISTORY_VALUE_CODES)]],
    device const ulong *value_coefficients [[buffer(SEISMIC_BUFFER_HISTORY_VALUE_COEFFICIENTS)]],
    device const int *tiles [[buffer(SEISMIC_BUFFER_HISTORY_TILES)]],
    device float *partials [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIALS)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
#if SEISMIC_TUNE_DIRECT
    const attention::history_window window = prefill_window(partials, uint(SEISMIC_DIM_M), uint(SEISMIC_DIM_HT));
    const uint round = window.words[attention::history_window::DECODE];
    if (group == 0 && simd == 0 && lane == 0)
        window.words[attention::history_window::ATTEND] = round;
    attention::prefill_decode(
        attention::affine_history{key_codes, key_coefficients, value_codes, value_coefficients,
            ulong(SEISMIC_PARAM_SLAB_ROWS)},
        window,
        attention::decoded_history::of(window,
            attention::held_tiles{tiles, window.words[attention::history_window::TILES]}, round,
            uint(SEISMIC_PARAM_SLAB_ROWS)),
        SEISMIC_DIM_T, group * 32 + simd, lane);
#endif
}

kernel void attention_prefill_k8v4_attend(
    device const attention::Scalar *query [[buffer(SEISMIC_BUFFER_QUERY)]],
    device const attention::Scalar *gate [[buffer(SEISMIC_BUFFER_GATE)]],
    device const int *visible [[buffer(SEISMIC_BUFFER_VISIBLE)]],
    device const int *fresh [[buffer(SEISMIC_BUFFER_FRESH)]],
    device const int *tiles [[buffer(SEISMIC_BUFFER_HISTORY_TILES)]],
    device const ulong *key_codes [[buffer(SEISMIC_BUFFER_HISTORY_KEY_CODES)]],
    device const ulong *key_coefficients [[buffer(SEISMIC_BUFFER_HISTORY_KEY_COEFFICIENTS)]],
    device const ulong *value_codes [[buffer(SEISMIC_BUFFER_HISTORY_VALUE_CODES)]],
    device const ulong *value_coefficients [[buffer(SEISMIC_BUFFER_HISTORY_VALUE_COEFFICIENTS)]],
    device attention::Scalar *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    device const half *queries [[buffer(SEISMIC_BUFFER_SCRATCH_QUERIES)]],
    device const half *keys [[buffer(SEISMIC_BUFFER_SCRATCH_KEYS)]],
    device const half *values [[buffer(SEISMIC_BUFFER_SCRATCH_VALUES)]],
    device float *partials [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIALS)]],
    device float *statistics [[buffer(SEISMIC_BUFFER_SCRATCH_STATISTICS)]],
    device uint *counts [[buffer(SEISMIC_BUFFER_SCRATCH_COUNTS)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    threadgroup uchar *shared [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 groups [[threadgroups_per_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
#if SEISMIC_TUNE_DIRECT
    // The round the decode launch decoded; the next one's follows it.
    const attention::history_window window = prefill_window(partials, uint(SEISMIC_DIM_M), uint(SEISMIC_DIM_HT));
    const uint round = window.words[attention::history_window::ATTEND];
    if (group.x == 0 && group.y == 0 && group.z == 0 && thread_index == 0)
        window.words[attention::history_window::DECODE] = round + 1;
#endif
    if (int(uint(SEISMIC_PARAM_GATE_FUNCTION)) == -1)
        return;
    PREFILL_EXCHANGE(exchange, SEISMIC_TUNE_QT);
#if SEISMIC_TUNE_DIRECT
    const attention::decoded_history history = attention::decoded_history::of(window,
        attention::held_tiles{tiles, window.words[attention::history_window::TILES]}, round,
        uint(SEISMIC_PARAM_SLAB_ROWS));
#else
    const attention::affine_history history{key_codes, key_coefficients, value_codes, value_coefficients,
        ulong(SEISMIC_PARAM_SLAB_ROWS)};
#endif
    attention::prefill_attend<SEISMIC_TUNE_QT>(history,
        query, gate, visible, fresh, result, queries, keys, values, partials, statistics, counts,
        SEISMIC_DIM_M, SEISMIC_DIM_R, as_type<float>(uint(SEISMIC_PARAM_SCALE)) * ATTENTION_LOG2E,
        SEISMIC_PARAM_GATE_FUNCTION != 0, shared, exchange, group, groups, thread_index, simd, lane);
}

kernel void attention_prefill_k8v4_fold(
    device const attention::Scalar *query [[buffer(SEISMIC_BUFFER_QUERY)]],
    device const attention::Scalar *gate [[buffer(SEISMIC_BUFFER_GATE)]],
    device const int *tiles [[buffer(SEISMIC_BUFFER_HISTORY_TILES)]],
    device attention::Scalar *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    device float *partials [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIALS)]],
    device const float *statistics [[buffer(SEISMIC_BUFFER_SCRATCH_STATISTICS)]],
    device const uint *counts [[buffer(SEISMIC_BUFFER_SCRATCH_COUNTS)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint column [[thread_index_in_threadgroup]]) {
#if SEISMIC_TUNE_DIRECT
    if (int(uint(SEISMIC_PARAM_GATE_FUNCTION)) == -1)
        return;
    // The round the attend launch before it took.
    const attention::history_window window = prefill_window(partials, uint(SEISMIC_DIM_M), uint(SEISMIC_DIM_HT));
    const uint round = window.words[attention::history_window::ATTEND];
    attention::prefill_fold<SEISMIC_TUNE_QT>(window,
        attention::decoded_history::of(window,
            attention::held_tiles{tiles, window.words[attention::history_window::TILES]}, round,
            uint(SEISMIC_PARAM_SLAB_ROWS)).held(),
        round, query, gate, result, partials, statistics, counts, SEISMIC_DIM_M, group.x, group.y, column,
        SEISMIC_PARAM_GATE_FUNCTION != 0);
#endif
}

kernel void attention_prefill_k8v4_merge(
    device const attention::Scalar *query [[buffer(SEISMIC_BUFFER_QUERY)]],
    device const attention::Scalar *gate [[buffer(SEISMIC_BUFFER_GATE)]],
    device attention::Scalar *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    device float *partials [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIALS)]],
    device const float *statistics [[buffer(SEISMIC_BUFFER_SCRATCH_STATISTICS)]],
    device const uint *counts [[buffer(SEISMIC_BUFFER_SCRATCH_COUNTS)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint column [[thread_index_in_threadgroup]]) {
    if (int(uint(SEISMIC_PARAM_GATE_FUNCTION)) == -1)
        return;
#if SEISMIC_TUNE_DIRECT
    // The fold launch stored the results of a call of several rounds.
    const attention::history_window window = prefill_window(partials, uint(SEISMIC_DIM_M), uint(SEISMIC_DIM_HT));
    if (window.rounds(window.words[attention::history_window::TILES]) > 1)
        return;
#endif
    attention::prefill_merge<SEISMIC_TUNE_QT>(query, gate, result, partials, statistics, counts,
        SEISMIC_DIM_M, group.x, group.y, column, SEISMIC_PARAM_GATE_FUNCTION != 0);
}
