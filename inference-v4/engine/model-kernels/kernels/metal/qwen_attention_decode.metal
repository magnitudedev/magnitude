#include "common/attention.h"

// Keys each simdgroup loads before scoring them, so their loads are in flight
// together.
#define DECODE_BATCH 4

// L1: threadgroup (kv head, partition, row). A row's visible keys, in order
// (spans, then its fresh span), split into equal partitions of
// attention::partition_span keys; each simdgroup scans a contiguous
// sub-range of its partition for all G query heads, so each key/value row is
// loaded once per kv head. Scores are F32 in the exp2 domain (scale * log2e
// folded into the query). The simdgroup states merge in fixed order into one
// partial per (row, query head, partition): the unnormalized output relative
// to the partition maximum, and (maximum, denominator). Partition 0 also
// appends the row's key and value at its destination.
kernel void qwen_attention_decode_partial(
    device const attention::Scalar *query_gate [[buffer(SEISMIC_BUFFER_QUERY_GATE)]],
    device const attention::Scalar *key [[buffer(SEISMIC_BUFFER_KEY)]],
    device const attention::Scalar *value [[buffer(SEISMIC_BUFFER_VALUE)]],
    device const float *query_norm [[buffer(SEISMIC_BUFFER_QUERY_NORM)]],
    device const float *key_norm [[buffer(SEISMIC_BUFFER_KEY_NORM)]],
    device const int *rotary_components [[buffer(SEISMIC_BUFFER_ROTARY_COMPONENTS)]],
    device const float *rotary_frequencies [[buffer(SEISMIC_BUFFER_ROTARY_FREQUENCIES)]],
    device const int *coordinates [[buffer(SEISMIC_BUFFER_COORDINATES)]],
    device const int *visible [[buffer(SEISMIC_BUFFER_VISIBLE)]],
    device const int *fresh [[buffer(SEISMIC_BUFFER_FRESH)]],
    device const int *destinations [[buffer(SEISMIC_BUFFER_DESTINATIONS)]],
    device attention::Scalar *history_key [[buffer(SEISMIC_BUFFER_HISTORY_KEY)]],
    device attention::Scalar *history_value [[buffer(SEISMIC_BUFFER_HISTORY_VALUE)]],
    device float *partials [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIALS)]],
    device float *statistics [[buffer(SEISMIC_BUFFER_SCRATCH_STATISTICS)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    threadgroup float *shared [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint W = ATTENTION_W;
    constexpr uint E = ATTENTION_E;
    constexpr uint KV = SEISMIC_DIM_KV;
    constexpr uint G = SEISMIC_DIM_G;
    constexpr uint SIMDS = SEISMIC_TUNE_SIMDS;
    constexpr uint PARTS = SEISMIC_TUNE_PARTS;
    const ulong R = SEISMIC_DIM_R;
    const uint kv_head = group.x;
    const uint partition = group.y;
    const ulong row = group.z;
    const float epsilon = as_type<float>(uint(SEISMIC_PARAM_EPSILON));
    const float scale = as_type<float>(uint(SEISMIC_PARAM_SCALE));
    device const int *row_coordinates = coordinates + row * 4;

    if (partition == 0 && simd < 2) {
        const int destination = destinations[row];
        if (destination >= 0) {
            const ulong source = (row * KV + kv_head) * W;
            if (simd == 0) {
                float k[E];
                attention::prepare(key + source, key_norm, row_coordinates,
                    rotary_components, rotary_frequencies, epsilon, lane, k);
                attention::append(history_key, destination, kv_head, lane, k);
            } else {
                attention::Scalar v[E];
                ATTENTION_UNROLL
                for (uint i = 0; i < E; ++i)
                    v[i] = value[source + lane * E + i];
                attention::append(history_value, destination, kv_head, lane, v);
            }
        }
    }

    const uint total = attention::visible_total(visible, fresh, row, R);
    const uint span_keys = attention::partition_span(total, SEISMIC_TUNE_SPAN, PARTS);
    const uint partition_lo = partition * span_keys;
    if (partition_lo >= total)
        return;
    const uint partition_hi = metal::min(partition_lo + span_keys, total);

    // Threadgroup memory: prepared queries [G][W], then the simdgroup states
    // [SIMDS][G][2] and merge columns [SIMDS][W].
    threadgroup float *prepared = shared;
    threadgroup float *states = shared + G * W;
    threadgroup float *columns = states + SIMDS * G * 2;
    for (uint g = simd; g < G; g += SIMDS) {
        float x[E];
        attention::prepare(query_gate + (row * KV * G + kv_head * G + g) * 2 * W, query_norm,
            row_coordinates, rotary_components, rotary_frequencies, epsilon, lane, x);
        ATTENTION_UNROLL
        for (uint i = 0; i < E; ++i)
            prepared[g * W + lane * E + i] = float(attention::Scalar(x[i])) * (scale * ATTENTION_LOG2E);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float q[G][E];
    float maximum[G];
    float denominator[G];
    float output[G][E];
    ATTENTION_UNROLL
    for (uint g = 0; g < G; ++g) {
        ATTENTION_UNROLL
        for (uint i = 0; i < E; ++i) {
            q[g][i] = prepared[g * W + lane * E + i];
            output[g][i] = 0.0f;
        }
        maximum[g] = -INFINITY;
        denominator[g] = 0.0f;
    }

    const uint sub = (partition_hi - partition_lo + SIMDS - 1) / SIMDS;
    const uint first = partition_lo + simd * sub;
    const uint last = metal::min(first + sub, partition_hi);
    uint offset = 0;
    for (ulong span = 0; span <= R && offset < last; ++span) {
        const bool historical = span < R;
        int lo, hi;
        attention::span(visible, fresh, row, R, span, lo, hi);
        const uint length = uint(metal::max(hi - lo, 0));
        const uint begin = metal::max(first, offset);
        const uint end = metal::min(last, offset + length);
        uint position = begin;
        if (historical) {
            for (; position + DECODE_BATCH <= end; position += DECODE_BATCH) {
                float k[DECODE_BATCH][E];
                float v[DECODE_BATCH][E];
                ATTENTION_UNROLL
                for (uint j = 0; j < DECODE_BATCH; ++j) {
                    const ulong token = ulong(lo) + (position + j - offset);
                    const ulong at = (token * KV + kv_head) * W + lane * E;
                    element::span<element::Act>(history_key + at, k[j]);
                    element::span<element::Act>(history_value + at, v[j]);
                }
                attention::absorb<DECODE_BATCH>(q, k, v, maximum, denominator, output);
            }
        }
        for (; position < end; ++position) {
            const ulong token = ulong(lo) + (position - offset);
            float k[1][E];
            float v[1][E];
            if (historical) {
                const ulong at = (token * KV + kv_head) * W + lane * E;
                element::span<element::Act>(history_key + at, k[0]);
                element::span<element::Act>(history_value + at, v[0]);
            } else {
                const ulong at = (token * KV + kv_head) * W;
                attention::prepare(key + at, key_norm, coordinates + token * 4,
                    rotary_components, rotary_frequencies, epsilon, lane, k[0]);
                ATTENTION_UNROLL
                for (uint i = 0; i < E; ++i) {
                    k[0][i] = float(attention::Scalar(k[0][i]));
                    v[0][i] = float(value[at + lane * E + i]);
                }
            }
            attention::absorb<1>(q, k, v, maximum, denominator, output);
        }
        offset += length;
    }

    attention::publish<SIMDS, PARTS>(maximum, denominator, output, states, columns, partials,
        statistics, (row * KV + kv_head) * G * PARTS + partition, simd, lane, thread_index);
}

// L2: threadgroup (query head, row), one thread per column: the fixed-order
// merge of the row's partitions and the sigmoid gate.
kernel void qwen_attention_decode_merge(
    device const attention::Scalar *query_gate [[buffer(SEISMIC_BUFFER_QUERY_GATE)]],
    device const int *visible [[buffer(SEISMIC_BUFFER_VISIBLE)]],
    device const int *fresh [[buffer(SEISMIC_BUFFER_FRESH)]],
    device attention::Scalar *gated [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    device const float *partials [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIALS)]],
    device const float *statistics [[buffer(SEISMIC_BUFFER_SCRATCH_STATISTICS)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint column [[thread_index_in_threadgroup]]) {
    attention::decode_gate<SEISMIC_TUNE_SPAN, SEISMIC_TUNE_PARTS>(query_gate, visible, fresh, gated,
        partials, statistics, SEISMIC_DIM_R, group.x, group.y, column);
}
