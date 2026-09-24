#include "common/attention.h"

#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_F32)
#error "prefill attention stages 2-byte activations in threadgroup memory"
#endif

// The three launches over affine K8/V4 history (bodies in common/attention.h):
// the prepare launch appends encoded rows, and every history K/V tile is
// decoded into the activation dtype as it is staged.

kernel void qwen_attention_prefill_k8v4_prepare(
    device const attention::Scalar *query_gate [[buffer(SEISMIC_BUFFER_QUERY_GATE)]],
    device const attention::Scalar *key [[buffer(SEISMIC_BUFFER_KEY)]],
    device const attention::Scalar *value [[buffer(SEISMIC_BUFFER_VALUE)]],
    device const float *query_norm [[buffer(SEISMIC_BUFFER_QUERY_NORM)]],
    device const float *key_norm [[buffer(SEISMIC_BUFFER_KEY_NORM)]],
    device const int *rotary_components [[buffer(SEISMIC_BUFFER_ROTARY_COMPONENTS)]],
    device const float *rotary_frequencies [[buffer(SEISMIC_BUFFER_ROTARY_FREQUENCIES)]],
    device const int *coordinates [[buffer(SEISMIC_BUFFER_COORDINATES)]],
    device const int *destinations [[buffer(SEISMIC_BUFFER_DESTINATIONS)]],
    device uint *key_codes [[buffer(SEISMIC_BUFFER_HISTORY_KEY_CODES)]],
    device half *key_coefficients [[buffer(SEISMIC_BUFFER_HISTORY_KEY_COEFFICIENTS)]],
    device uint *value_codes [[buffer(SEISMIC_BUFFER_HISTORY_VALUE_CODES)]],
    device half *value_coefficients [[buffer(SEISMIC_BUFFER_HISTORY_VALUE_COEFFICIENTS)]],
    device attention::Scalar *queries [[buffer(SEISMIC_BUFFER_SCRATCH_QUERIES)]],
    device attention::Scalar *keys [[buffer(SEISMIC_BUFFER_SCRATCH_KEYS)]],
    device attention::Scalar *values [[buffer(SEISMIC_BUFFER_SCRATCH_VALUES)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    attention::prefill_prepare<SEISMIC_TUNE_QT>(
        attention::affine_history{key_codes, key_coefficients, value_codes, value_coefficients},
        query_gate, key, value, query_norm, key_norm, rotary_components, rotary_frequencies,
        coordinates, destinations, queries, keys, values, SEISMIC_DIM_M,
        as_type<float>(uint(SEISMIC_PARAM_EPSILON)), group, simd, lane);
}

kernel void qwen_attention_prefill_k8v4_attend(
    device const attention::Scalar *query_gate [[buffer(SEISMIC_BUFFER_QUERY_GATE)]],
    device const int *visible [[buffer(SEISMIC_BUFFER_VISIBLE)]],
    device const int *fresh [[buffer(SEISMIC_BUFFER_FRESH)]],
    device uint *key_codes [[buffer(SEISMIC_BUFFER_HISTORY_KEY_CODES)]],
    device half *key_coefficients [[buffer(SEISMIC_BUFFER_HISTORY_KEY_COEFFICIENTS)]],
    device uint *value_codes [[buffer(SEISMIC_BUFFER_HISTORY_VALUE_CODES)]],
    device half *value_coefficients [[buffer(SEISMIC_BUFFER_HISTORY_VALUE_COEFFICIENTS)]],
    device attention::Scalar *gated [[buffer(SEISMIC_RESULT_0_BUFFER)]],
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
    attention::prefill_attend<SEISMIC_TUNE_QT>(
        attention::affine_history{key_codes, key_coefficients, value_codes, value_coefficients},
        query_gate, visible, fresh, gated, queries, keys, values, partials, statistics, counts,
        SEISMIC_DIM_M, SEISMIC_DIM_R, as_type<float>(uint(SEISMIC_PARAM_SCALE)) * ATTENTION_LOG2E,
        shared, group, groups, thread_index, simd, lane);
}

kernel void qwen_attention_prefill_k8v4_merge(
    device const attention::Scalar *query_gate [[buffer(SEISMIC_BUFFER_QUERY_GATE)]],
    device attention::Scalar *gated [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    device const float *partials [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIALS)]],
    device const float *statistics [[buffer(SEISMIC_BUFFER_SCRATCH_STATISTICS)]],
    device const uint *counts [[buffer(SEISMIC_BUFFER_SCRATCH_COUNTS)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint column [[thread_index_in_threadgroup]]) {
    attention::prefill_merge<SEISMIC_TUNE_QT>(query_gate, gated, partials, statistics, counts,
        SEISMIC_DIM_M, group.x, group.y, column);
}
