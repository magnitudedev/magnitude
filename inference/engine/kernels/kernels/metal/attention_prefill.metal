#define ATTENTION_I SEISMIC_DIM_I
#define ATTENTION_U SEISMIC_DIM_U
#define ATTENTION_FRESH (SEISMIC_DIM_F != 0)
#define ATTENTION_NORM (SEISMIC_DIM_N != 0)
#define ATTENTION_VALUE_NORM (SEISMIC_DIM_NV != 0)
#define PREFILL_HEAD_GROUPS SEISMIC_TUNE_HEAD_GROUPS
#include "lib/attention/attention.h"

#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_F32)
#error "prefill attention stages 2-byte activations in threadgroup memory"
#endif

// The three launches over dense history (bodies in lib/attention/attention.h).

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
    attention::prefill_prepare<SEISMIC_TUNE_QT>(attention::dense_history{history_key, history_value,
        ulong(SEISMIC_PARAM_SLAB_ROWS)},
        query, key, value, query_norm, key_norm, value_norm, rotary_components, rotary_frequencies,
        rotary_amplitudes, coordinates, destinations, queries, keys, values, SEISMIC_DIM_M,
        as_type<float>(uint(SEISMIC_PARAM_EPSILON)), int(uint(SEISMIC_PARAM_GATE_FUNCTION)) == -1,
        group, simd, lane);
}

kernel void attention_prefill_attend(
    device const attention::Scalar *query [[buffer(SEISMIC_BUFFER_QUERY)]],
    device const attention::Scalar *gate [[buffer(SEISMIC_BUFFER_GATE)]],
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
    attention::prefill_attend<SEISMIC_TUNE_QT>(attention::dense_history{history_key, history_value,
        ulong(SEISMIC_PARAM_SLAB_ROWS)},
        query, gate, visible, fresh, result, queries, keys, values, partials, statistics, counts,
        SEISMIC_DIM_M, SEISMIC_DIM_R, as_type<float>(uint(SEISMIC_PARAM_SCALE)) * ATTENTION_LOG2E,
        SEISMIC_PARAM_GATE_FUNCTION != 0, shared, exchange, group, groups, thread_index, simd, lane);
}

kernel void attention_prefill_merge(
    device const attention::Scalar *query [[buffer(SEISMIC_BUFFER_QUERY)]],
    device const attention::Scalar *gate [[buffer(SEISMIC_BUFFER_GATE)]],
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
        SEISMIC_DIM_M, group.x, group.y, column, SEISMIC_PARAM_GATE_FUNCTION != 0);
}
