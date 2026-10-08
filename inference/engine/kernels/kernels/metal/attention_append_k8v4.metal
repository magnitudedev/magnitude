#define ATTENTION_QUERY_GROUP 1
// State-only geometry: common helpers have no query or gate rows.
#define ATTENTION_I 0
#define ATTENTION_U 0
#define ATTENTION_FRESH (SEISMIC_DIM_F != 0)
#define ATTENTION_NORM (SEISMIC_DIM_N != 0)
#define ATTENTION_VALUE_NORM (SEISMIC_DIM_NV != 0)
#include "lib/attention/attention.h"


// Ordered cache publication before causal target attention.
kernel void attention_append_k8v4(
    device const attention::Scalar *key [[buffer(SEISMIC_BUFFER_KEY)]],
    device const attention::Scalar *value [[buffer(SEISMIC_BUFFER_VALUE)]],
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
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint KV = SEISMIC_DIM_KV;
    constexpr uint W = ATTENTION_W;
    constexpr uint E = ATTENTION_E;
    typedef attention::lane_codes<ATTENTION_KEY_BITS> key_lane;
    typedef attention::lane_codes<ATTENTION_VALUE_BITS> value_lane;
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
    ATTENTION_UNROLL
    for (uint i = 0; i < E; ++i) {
        k[i] = element::Act::round(k[i]);
        v[i] = element::Act::round(v[i]);
    }
    attention::encode<ATTENTION_KEY_BITS>(k,
        slab::row<uint>(key_codes, destination, slab_rows, KV * key_lane::row_words) + kv_head * key_lane::row_words,
        slab::row<half>(key_coefficients, destination, slab_rows, KV * key_lane::pairs * 2) + kv_head * key_lane::pairs * 2, lane);
    attention::encode<ATTENTION_VALUE_BITS>(v,
        slab::row<uint>(value_codes, destination, slab_rows, KV * value_lane::row_words) + kv_head * value_lane::row_words,
        slab::row<half>(value_coefficients, destination, slab_rows, KV * value_lane::pairs * 2) + kv_head * value_lane::pairs * 2, lane);
}
