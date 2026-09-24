#include "common/attention.h"

// History keys each simdgroup loads before scoring them. Affine rows are 2.6x
// smaller than dense ones, so twice the dense batch keeps as many bytes in
// flight.
#define DECODE_BATCH 8

// L1: threadgroup (kv head, partition, row), as `qwen_attention_decode`, over
// affine K8/V4 history. History keys score as scale * (q . code) + zero *
// sum(q) and values accumulate (p * scale) * code plus the carried per-head
// bias sum(p * zero) (attention::absorb_affine), so no history element is
// decoded. The fresh span stays dense; the bias is folded into the output
// before it. Partition 0 appends the row's key (prepared, rounded to the
// activation dtype) and value encoded (attention::encode).
kernel void qwen_attention_decode_k8v4_partial(
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
    device uint *key_codes [[buffer(SEISMIC_BUFFER_HISTORY_KEY_CODES)]],
    device half *key_coefficients [[buffer(SEISMIC_BUFFER_HISTORY_KEY_COEFFICIENTS)]],
    device uint *value_codes [[buffer(SEISMIC_BUFFER_HISTORY_VALUE_CODES)]],
    device half *value_coefficients [[buffer(SEISMIC_BUFFER_HISTORY_VALUE_COEFFICIENTS)]],
    device float *partials [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIALS)]],
    device float *statistics [[buffer(SEISMIC_BUFFER_SCRATCH_STATISTICS)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    threadgroup float *shared [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    typedef attention::lane_codes<ATTENTION_KEY_BITS> key_lane;
    typedef attention::lane_codes<ATTENTION_VALUE_BITS> value_lane;
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
            const ulong vector = ulong(destination) * KV + kv_head;
            float x[E];
            if (simd == 0) {
                attention::prepare(key + source, key_norm, row_coordinates,
                    rotary_components, rotary_frequencies, epsilon, lane, x);
                ATTENTION_UNROLL
                for (uint i = 0; i < E; ++i)
                    x[i] = float(attention::Scalar(x[i]));
                attention::encode<ATTENTION_KEY_BITS>(x, key_codes + vector * key_lane::row_words,
                    key_coefficients + vector * 2, lane);
            } else {
                ATTENTION_UNROLL
                for (uint i = 0; i < E; ++i)
                    x[i] = float(value[source + lane * E + i]);
                attention::encode<ATTENTION_VALUE_BITS>(x, value_codes + vector * value_lane::row_words,
                    value_coefficients + vector * 2, lane);
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
    float qsum[G];
    float maximum[G];
    float denominator[G];
    float output[G][E];
    float bias[G];
    ATTENTION_UNROLL
    for (uint g = 0; g < G; ++g) {
        float sum = 0.0f;
        ATTENTION_UNROLL
        for (uint i = 0; i < E; ++i) {
            q[g][i] = prepared[g * W + lane * E + i];
            sum += q[g][i];
            output[g][i] = 0.0f;
        }
        qsum[g] = simd_sum(sum);
        maximum[g] = -INFINITY;
        denominator[g] = 0.0f;
        bias[g] = 0.0f;
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
                uint k[DECODE_BATCH][key_lane::words];
                uint v[DECODE_BATCH][value_lane::words];
                float2 kc[DECODE_BATCH];
                float2 vc[DECODE_BATCH];
                ATTENTION_UNROLL
                for (uint j = 0; j < DECODE_BATCH; ++j) {
                    const ulong vector = (ulong(lo) + (position + j - offset)) * KV + kv_head;
                    key_lane::load(key_codes + vector * key_lane::row_words, lane, k[j]);
                    value_lane::load(value_codes + vector * value_lane::row_words, lane, v[j]);
                    kc[j] = float2(*reinterpret_cast<device const half2 *>(key_coefficients + vector * 2));
                    vc[j] = float2(*reinterpret_cast<device const half2 *>(value_coefficients + vector * 2));
                }
                attention::absorb_affine<DECODE_BATCH>(q, qsum, k, kc, v, vc, maximum, denominator,
                    output, bias);
            }
            for (; position < end; ++position) {
                uint k[1][key_lane::words];
                uint v[1][value_lane::words];
                float2 kc[1];
                float2 vc[1];
                const ulong vector = (ulong(lo) + (position - offset)) * KV + kv_head;
                key_lane::load(key_codes + vector * key_lane::row_words, lane, k[0]);
                value_lane::load(value_codes + vector * value_lane::row_words, lane, v[0]);
                kc[0] = float2(*reinterpret_cast<device const half2 *>(key_coefficients + vector * 2));
                vc[0] = float2(*reinterpret_cast<device const half2 *>(value_coefficients + vector * 2));
                attention::absorb_affine<1>(q, qsum, k, kc, v, vc, maximum, denominator, output, bias);
            }
        } else {
            // The dense absorb carries only the output: fold the bias in first.
            ATTENTION_UNROLL
            for (uint g = 0; g < G; ++g) {
                ATTENTION_UNROLL
                for (uint i = 0; i < E; ++i)
                    output[g][i] += bias[g];
                bias[g] = 0.0f;
            }
            for (; position < end; ++position) {
                const ulong at = (ulong(lo) + (position - offset)) * KV * W + kv_head * W;
                float k[1][E];
                float v[1][E];
                attention::prepare(key + at, key_norm, coordinates + (ulong(lo) + (position - offset)) * 4,
                    rotary_components, rotary_frequencies, epsilon, lane, k[0]);
                ATTENTION_UNROLL
                for (uint i = 0; i < E; ++i) {
                    k[0][i] = float(attention::Scalar(k[0][i]));
                    v[0][i] = float(value[at + lane * E + i]);
                }
                attention::absorb<1>(q, k, v, maximum, denominator, output);
            }
        }
        offset += length;
    }
    ATTENTION_UNROLL
    for (uint g = 0; g < G; ++g) {
        ATTENTION_UNROLL
        for (uint i = 0; i < E; ++i)
            output[g][i] += bias[g];
    }

    attention::publish<SIMDS, PARTS>(maximum, denominator, output, states, columns, partials,
        statistics, (row * KV + kv_head) * G * PARTS + partition, simd, lane, thread_index);
}

// L2: threadgroup (query head, row), one thread per column: the fixed-order
// merge of the row's partitions and the sigmoid gate.
kernel void qwen_attention_decode_k8v4_merge(
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
