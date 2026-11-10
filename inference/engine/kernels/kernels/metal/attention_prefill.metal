#define ATTENTION_QUERY_GROUP SEISMIC_DIM_G
#define ATTENTION_I SEISMIC_DIM_I
#define ATTENTION_U SEISMIC_DIM_U
#define ATTENTION_FRESH (SEISMIC_DIM_F != 0)
#define ATTENTION_NORM (SEISMIC_DIM_N != 0)
#define ATTENTION_QUERY_NORM (SEISMIC_DIM_NQ != 0)
#define ATTENTION_VALUE_NORM (SEISMIC_DIM_NV != 0)
#define ATTENTION_SINKS (SEISMIC_DIM_SK != 0)
#define ATTENTION_STORED_KEY (SEISMIC_DIM_VK != 0)
#define PREFILL_HEADS_PER_GROUP SEISMIC_TUNE_HEADS
#define PREFILL_DIRECT SEISMIC_TUNE_DIRECT
#define PREFILL_SELECTED (SEISMIC_DIM_B != 0)
#include "lib/attention/attention.h"

// Listed key tiles a query tile can hold: each of its rows reads at most the
// bitmap's 32 * B * BR keys, in at most R + 1 runs (the entry's `lists`
// scratch, attention.seismic).
#define PREFILL_LIST_CAPACITY uint(SEISMIC_TUNE_QT * ((32 * SEISMIC_DIM_B * SEISMIC_DIM_BR + PREFILL_KEYS - 1) / PREFILL_KEYS + SEISMIC_DIM_R + 1))
// The worklist scratch: the query tiles' (span, first) lists, their counts,
// then the rows' span ordinals.
#define PREFILL_LIST_TILES uint((SEISMIC_DIM_M + SEISMIC_TUNE_QT - 1) / SEISMIC_TUNE_QT)
// The attend grid's key partitions (its z extent, attention.seismic).
#define PREFILL_PARTS_DIVISOR                                                                              \
    (PREFILL_LIST_TILES * uint(SEISMIC_DIM_KV) * ((ATTENTION_QUERY_GROUP + SEISMIC_TUNE_HEADS - 1) / SEISMIC_TUNE_HEADS))
#define PREFILL_PARTS                                                                                      \
    metal::max(1u, uint((uint(SEISMIC_TUNE_SPLIT_GROUPS) + PREFILL_PARTS_DIVISOR - 1) / PREFILL_PARTS_DIVISOR))
#define PREFILL_WORKLIST(selection, worklist)                                                              \
    attention::prefill_selection{selection,                                                                \
        (worklist) + ulong(PREFILL_LIST_TILES) * PREFILL_LIST_CAPACITY * 2 + PREFILL_LIST_TILES,          \
        reinterpret_cast<device const int2 *>(worklist),                                                   \
        reinterpret_cast<device const uint *>((worklist) + ulong(PREFILL_LIST_TILES) * PREFILL_LIST_CAPACITY * 2), \
        uint(SEISMIC_DIM_B), int(SEISMIC_DIM_BR), PREFILL_LIST_CAPACITY}


#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_F32)
#error "prefill attention stages 2-byte activations in threadgroup memory"
#endif

// The launches over dense history (bodies in lib/attention/attention.h). A
// layer with picks (KS > 0) runs prepare, which only appends the rows, then
// the shared-tile launch; the others prepare, attend, merge and rotate.

kernel void attention_prefill_prepare(
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
    device const ulong *history_key [[buffer(SEISMIC_BUFFER_HISTORY_KEY)]],
    device const ulong *history_value [[buffer(SEISMIC_BUFFER_HISTORY_VALUE)]],
    device attention::Scalar *queries [[buffer(SEISMIC_BUFFER_SCRATCH_QUERIES)]],
    device attention::Scalar *keys [[buffer(SEISMIC_BUFFER_SCRATCH_KEYS)]],
    device attention::Scalar *values [[buffer(SEISMIC_BUFFER_SCRATCH_VALUES)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    attention::prefill_prepare<SEISMIC_TUNE_QT>(attention::dense_history{history_key,
        attention::values_plane(history_key, history_value), ulong(SEISMIC_PARAM_SLAB_ROWS)},
        query, key, value, query_norm, key_norm, value_norm, rotary_components, rotary_frequencies,
        rotary_amplitudes, coordinates, destinations, queries, keys, values, SEISMIC_DIM_M,
        as_type<float>(uint(SEISMIC_PARAM_EPSILON)),
        int(uint(SEISMIC_PARAM_GATE_FUNCTION)) == -1 || SEISMIC_DIM_KS != 0,
        group, simd, lane, attention::key_rounding(SEISMIC_DIM_KR != 0, uint(SEISMIC_PARAM_KEY_BLOCK), uint(SEISMIC_PARAM_KEY_ELEMENT),
        uint(SEISMIC_PARAM_KEY_SCALE), uint(SEISMIC_PARAM_KEY_AMAX_FLOOR)));
}

kernel void attention_prefill_attend(
    device const attention::Scalar *query [[buffer(SEISMIC_BUFFER_QUERY)]],
    device const attention::Scalar *gate [[buffer(SEISMIC_BUFFER_GATE)]],
    device const float *sinks [[buffer(SEISMIC_BUFFER_SINKS)]],
    device const int *visible [[buffer(SEISMIC_BUFFER_VISIBLE)]],
    device const int *fresh [[buffer(SEISMIC_BUFFER_FRESH)]],
    device const ulong *history_key [[buffer(SEISMIC_BUFFER_HISTORY_KEY)]],
    device const ulong *history_value [[buffer(SEISMIC_BUFFER_HISTORY_VALUE)]],
    device attention::Scalar *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    device const attention::Scalar *queries [[buffer(SEISMIC_BUFFER_SCRATCH_QUERIES)]],
    device const attention::Scalar *keys [[buffer(SEISMIC_BUFFER_SCRATCH_KEYS)]],
    device const attention::Scalar *values [[buffer(SEISMIC_BUFFER_SCRATCH_VALUES)]],
    device float *partials [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIALS)]],
    device float *statistics [[buffer(SEISMIC_BUFFER_SCRATCH_STATISTICS)]],
    device uint *counts [[buffer(SEISMIC_BUFFER_SCRATCH_COUNTS)]],
    device const uint *selection [[buffer(SEISMIC_BUFFER_SELECTION)]],
    device int *worklist [[buffer(SEISMIC_BUFFER_SCRATCH_WORKLIST)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    threadgroup uchar *shared [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 groups [[threadgroups_per_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (int(uint(SEISMIC_PARAM_GATE_FUNCTION)) == -1)
        return;
    PREFILL_EXCHANGE(exchange, SEISMIC_TUNE_QT);
    attention::prefill_attend<SEISMIC_TUNE_QT>(attention::dense_history{history_key,
        attention::values_plane(history_key, history_value), ulong(SEISMIC_PARAM_SLAB_ROWS)},
        query, gate, visible, fresh, result, queries, keys, values, partials, statistics, counts,
        SEISMIC_DIM_M, SEISMIC_DIM_R, as_type<float>(uint(SEISMIC_PARAM_SCALE)) * ATTENTION_LOG2E,
        SEISMIC_PARAM_GATE_FUNCTION != 0, shared, exchange, group, groups, thread_index, simd, lane,
        PREFILL_WORKLIST(selection, worklist), sinks);
}

kernel void attention_prefill_merge(
    device const attention::Scalar *query [[buffer(SEISMIC_BUFFER_QUERY)]],
    device const attention::Scalar *gate [[buffer(SEISMIC_BUFFER_GATE)]],
    device const float *sinks [[buffer(SEISMIC_BUFFER_SINKS)]],
    device attention::Scalar *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    device const float *partials [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIALS)]],
    device const float *statistics [[buffer(SEISMIC_BUFFER_SCRATCH_STATISTICS)]],
    device const uint *counts [[buffer(SEISMIC_BUFFER_SCRATCH_COUNTS)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint column [[thread_index_in_threadgroup]]) {
    if (int(uint(SEISMIC_PARAM_GATE_FUNCTION)) == -1)
        return;
    attention::prefill_merge<SEISMIC_TUNE_QT>(query, gate, result, partials, statistics, counts,
        SEISMIC_DIM_M, group.x, group.y, column, SEISMIC_PARAM_GATE_FUNCTION != 0, sinks);
}

// A layer with picks (KS > 0): threadgroup (kv head, row), the shared-tile
// form of `attention_decode` (`attention::decode_shared`) over the row's
// whole key sequence (its visible keys, its fresh keys prepared in place,
// then its KS picks gathered into KEYS-key tiles), all G query heads of the
// kv head as the rows of its products, publishing the finished row itself:
// the output rotary (RO = 1) and the gate, after prepare appended every row.
kernel void attention_prefill_shared(
    device const attention::Scalar *query [[buffer(SEISMIC_BUFFER_QUERY)]],
    device const attention::Scalar *gate [[buffer(SEISMIC_BUFFER_GATE)]],
    device const attention::Scalar *key [[buffer(SEISMIC_BUFFER_KEY)]],
    device const attention::Scalar *value [[buffer(SEISMIC_BUFFER_VALUE)]],
    device const float *query_norm [[buffer(SEISMIC_BUFFER_QUERY_NORM)]],
    device const float *key_norm [[buffer(SEISMIC_BUFFER_KEY_NORM)]],
    device const float *value_norm [[buffer(SEISMIC_BUFFER_VALUE_NORM)]],
    device const float *sinks [[buffer(SEISMIC_BUFFER_SINKS)]],
    device const int *rotary_components [[buffer(SEISMIC_BUFFER_ROTARY_COMPONENTS)]],
    device const float *rotary_frequencies [[buffer(SEISMIC_BUFFER_ROTARY_FREQUENCIES)]],
    device const float *rotary_amplitudes [[buffer(SEISMIC_BUFFER_ROTARY_AMPLITUDES)]],
    device const int *coordinates [[buffer(SEISMIC_BUFFER_COORDINATES)]],
    device const int *visible [[buffer(SEISMIC_BUFFER_VISIBLE)]],
    device const int *fresh [[buffer(SEISMIC_BUFFER_FRESH)]],
    device const ulong *history_key [[buffer(SEISMIC_BUFFER_HISTORY_KEY)]],
    device const ulong *history_value [[buffer(SEISMIC_BUFFER_HISTORY_VALUE)]],
    device const int *selected_rows [[buffer(SEISMIC_BUFFER_SELECTED_ROWS)]],
    device const ulong *selected [[buffer(SEISMIC_BUFFER_SELECTED)]],
    device attention::Scalar *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    threadgroup uchar *shared [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
#if SEISMIC_STATIC_KS != 0
    if (int(uint(SEISMIC_PARAM_GATE_FUNCTION)) == -1)
        return;
    attention::decode_shared<SEISMIC_TUNE_KEYS, 1, SEISMIC_DIM_KS, true, SEISMIC_DIM_RO != 0>(
        attention::dense_history{history_key, attention::values_plane(history_key, history_value),
            ulong(SEISMIC_PARAM_SLAB_ROWS)},
        query, key, value, query_norm, key_norm, value_norm, rotary_components, rotary_frequencies,
        rotary_amplitudes, coordinates, visible, fresh, nullptr, nullptr, shared, SEISMIC_DIM_R,
        as_type<float>(uint(SEISMIC_PARAM_EPSILON)), as_type<float>(uint(SEISMIC_PARAM_SCALE)) * ATTENTION_LOG2E, 1,
        group.x, 0, group.y, thread_index, simd, lane,
        attention::decode_picks{selected_rows, selected, ulong(SEISMIC_PARAM_SELECTED_SLAB_ROWS)},
        attention::key_rounding(SEISMIC_DIM_KR != 0, uint(SEISMIC_PARAM_KEY_BLOCK), uint(SEISMIC_PARAM_KEY_ELEMENT),
            uint(SEISMIC_PARAM_KEY_SCALE), uint(SEISMIC_PARAM_KEY_AMAX_FLOOR)),
        attention::shared_result{query, gate, sinks, result, SEISMIC_PARAM_GATE_FUNCTION != 0});
#endif
}

// The WorkList's list launch (B > 0): each query tile's read key tiles
// (`attention::prefill_list`).
kernel void attention_prefill_list(
    device const int *visible [[buffer(SEISMIC_BUFFER_VISIBLE)]],
    device const int *fresh [[buffer(SEISMIC_BUFFER_FRESH)]],
    device const uint *selection [[buffer(SEISMIC_BUFFER_SELECTION)]],
    device int *worklist [[buffer(SEISMIC_BUFFER_SCRATCH_WORKLIST)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    threadgroup uchar *shared [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (int(uint(SEISMIC_PARAM_GATE_FUNCTION)) == -1)
        return;
    const attention::prefill_selection list = PREFILL_WORKLIST(selection, worklist);
    attention::prefill_list<SEISMIC_TUNE_QT>(visible, fresh, selection, const_cast<device int *>(list.bases),
        const_cast<device int2 *>(list.lists), const_cast<device uint *>(list.listed), SEISMIC_DIM_M,
        SEISMIC_DIM_R, SEISMIC_DIM_B, int(SEISMIC_DIM_BR), PREFILL_LIST_CAPACITY, shared, group.x,
        thread_index, simd, lane);
}

// The output rotary (RO = 1, `attention::output_rotary`): threadgroup (row,
// query head), one thread per rotary pair, after every output is stored.
kernel void attention_prefill_rotate(
    device const int *rotary_components [[buffer(SEISMIC_BUFFER_ROTARY_COMPONENTS)]],
    device const float *rotary_frequencies [[buffer(SEISMIC_BUFFER_ROTARY_FREQUENCIES)]],
    device const float *rotary_amplitudes [[buffer(SEISMIC_BUFFER_ROTARY_AMPLITUDES)]],
    device const int *coordinates [[buffer(SEISMIC_BUFFER_COORDINATES)]],
    device attention::Scalar *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint pair [[thread_index_in_threadgroup]]) {
    if (int(uint(SEISMIC_PARAM_GATE_FUNCTION)) == -1)
        return;
    attention::output_rotary(result, coordinates, rotary_components, rotary_frequencies, rotary_amplitudes,
        group.x, group.y, pair);
}
