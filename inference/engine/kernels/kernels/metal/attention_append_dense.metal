#define ATTENTION_QUERY_GROUP 1
// State-only geometry: common helpers have no query or gate rows.
#define ATTENTION_I 0
#define ATTENTION_U 0
#define ATTENTION_FRESH (SEISMIC_DIM_F != 0)
#define ATTENTION_NORM (SEISMIC_DIM_N != 0)
#define ATTENTION_VALUE_NORM (SEISMIC_DIM_NV != 0)
#include "lib/attention/attention.h"


// Ordered cache publication before causal target attention.
kernel void attention_append_dense(
    device const attention::Scalar *key [[buffer(SEISMIC_BUFFER_KEY)]],
    device const attention::Scalar *value [[buffer(SEISMIC_BUFFER_VALUE)]],
    device const float *key_norm [[buffer(SEISMIC_BUFFER_KEY_NORM)]],
    device const float *value_norm [[buffer(SEISMIC_BUFFER_VALUE_NORM)]],
    device const int *rotary_components [[buffer(SEISMIC_BUFFER_ROTARY_COMPONENTS)]],
    device const float *rotary_frequencies [[buffer(SEISMIC_BUFFER_ROTARY_FREQUENCIES)]],
    device const float *rotary_amplitudes [[buffer(SEISMIC_BUFFER_ROTARY_AMPLITUDES)]],
    device const int *coordinates [[buffer(SEISMIC_BUFFER_COORDINATES)]],
    device const int *destinations [[buffer(SEISMIC_BUFFER_DESTINATIONS)]],
    device const ulong *history_key [[buffer(SEISMIC_BUFFER_HISTORY_KEY)]],
    device const ulong *history_value [[buffer(SEISMIC_BUFFER_HISTORY_VALUE)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint KV = SEISMIC_DIM_KV;
    constexpr uint W = ATTENTION_W;
    constexpr uint E = ATTENTION_E;
    const ulong item = ulong(group.x) * 8 + simd;
    const ulong row = item / KV;
    const uint kv_head = uint(item % KV);
    if (!ATTENTION_FRESH || row >= SEISMIC_DIM_M || destinations[row] < 0)
        return;
    const ulong source = (row * KV + kv_head) * W;
    const ulong destination = ulong(destinations[row]);
    const ulong slab_rows = ulong(SEISMIC_PARAM_SLAB_ROWS);
    const float epsilon = as_type<float>(uint(SEISMIC_PARAM_EPSILON));
    float k[E], v[E];
    attention::head_rotary<ATTENTION_NORM>(key + source, key_norm, coordinates + row * 4,
        rotary_components, rotary_frequencies, rotary_amplitudes, epsilon, lane, k);
    attention::head_norm<ATTENTION_VALUE_NORM>(value + source, value_norm, epsilon, lane, v);
    attention::append(slab::row<attention::Scalar>(history_key, destination, slab_rows, KV * W),
        0, kv_head, lane, k);
    attention::append(slab::row<attention::Scalar>(history_value, destination, slab_rows, KV * W),
        0, kv_head, lane, v);
}
