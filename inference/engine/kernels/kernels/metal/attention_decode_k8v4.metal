#define ATTENTION_I SEISMIC_DIM_I
#define ATTENTION_U SEISMIC_DIM_U
#define ATTENTION_FRESH (SEISMIC_DIM_F != 0)
#define ATTENTION_NORM (SEISMIC_DIM_N != 0)
#define ATTENTION_VALUE_NORM (SEISMIC_DIM_NV != 0)
#include "lib/attention/attention.h"

// History keys each simdgroup loads before scoring them: the dense batch.
// The affine absorb is bound by per-key work, not bytes in flight; larger
// batches only raise register pressure (measured 8 and 16 slower on M4).
#define DECODE_BATCH 4

// L1: threadgroup (kv head, partition, row), as `attention_decode`, over
// affine K8/V4 history. History keys score as the sum over groups of
// scale * (q . code) + zero * sum(q) and values accumulate (p * scale) * code
// plus the carried bias sum(p * zero) of each lane's group
// (attention::absorb_affine), so no history element is decoded. The fresh
// span stays dense; the bias is folded into the output before it. Partition 0
// of a layer with fresh rows appends the row's key (prepared, rounded to the
// activation dtype) and value encoded (attention::encode).
kernel void attention_decode_k8v4_partial(
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
    device const int *visible [[buffer(SEISMIC_BUFFER_VISIBLE)]],
    device const int *fresh [[buffer(SEISMIC_BUFFER_FRESH)]],
    device const int *destinations [[buffer(SEISMIC_BUFFER_DESTINATIONS)]],
    device const ulong *key_codes [[buffer(SEISMIC_BUFFER_HISTORY_KEY_CODES)]],
    device const ulong *key_coefficients [[buffer(SEISMIC_BUFFER_HISTORY_KEY_COEFFICIENTS)]],
    device const ulong *value_codes [[buffer(SEISMIC_BUFFER_HISTORY_VALUE_CODES)]],
    device const ulong *value_coefficients [[buffer(SEISMIC_BUFFER_HISTORY_VALUE_COEFFICIENTS)]],
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
    constexpr uint SLICES = SEISMIC_TUNE_SLICES;
    constexpr uint H = G / SLICES;
    const ulong R = SEISMIC_DIM_R;
    const uint kv_head = group.x;
    const uint partition = group.y;
    const ulong row = group.z;
    const float epsilon = as_type<float>(uint(SEISMIC_PARAM_EPSILON));
    const float scale = as_type<float>(uint(SEISMIC_PARAM_SCALE));
    device const int *row_coordinates = coordinates + row * 4;

    if (ATTENTION_FRESH && partition == 0 && simd < 2) {
        const int destination = destinations[row];
        if (destination >= 0) {
            const ulong source = (row * KV + kv_head) * W;
            const ulong vector = kv_head;
            const ulong slab_rows = ulong(SEISMIC_PARAM_SLAB_ROWS);
            float x[E];
            if (simd == 0) {
                attention::head_rotary<ATTENTION_NORM>(key + source, key_norm, row_coordinates,
                    rotary_components, rotary_frequencies, rotary_amplitudes, epsilon, lane, x);
                ATTENTION_UNROLL
                for (uint i = 0; i < E; ++i)
                    x[i] = float(attention::Scalar(x[i]));
                attention::encode<ATTENTION_KEY_BITS>(x,
                    slab::row<uint>(key_codes, ulong(destination), slab_rows, KV * key_lane::row_words)
                        + vector * key_lane::row_words,
                    slab::row<half>(key_coefficients, ulong(destination), slab_rows, KV * key_lane::pairs * 2)
                        + vector * key_lane::pairs * 2, lane);
            } else {
                attention::head_norm<ATTENTION_VALUE_NORM>(value + source, value_norm, epsilon, lane, x);
                ATTENTION_UNROLL
                for (uint i = 0; i < E; ++i)
                    x[i] = float(attention::Scalar(x[i]));
                attention::encode<ATTENTION_VALUE_BITS>(x,
                    slab::row<uint>(value_codes, ulong(destination), slab_rows, KV * value_lane::row_words)
                        + vector * value_lane::row_words,
                    slab::row<half>(value_coefficients, ulong(destination), slab_rows, KV * value_lane::pairs * 2)
                        + vector * value_lane::pairs * 2, lane);
            }
        }
    }

    const uint total = attention::form_total(visible, fresh, row, R);
    const uint span_keys = attention::partition_span(total, SEISMIC_TUNE_SPAN, PARTS);
    const uint partition_lo = partition * span_keys;
    if (partition_lo >= total)
        return;
    const uint partition_hi = metal::min(partition_lo + span_keys, total);

    // Threadgroup memory: the simdgroup states [SIMDS][H][2] (floats), then
    // the prepared queries [G][W] (activation dtype, as the contract
    // publishes them), which the merge columns [SIMDS][W] (floats) alias:
    // every simdgroup has read its queries before the publication's first
    // barrier. Simdgroup s holds the H query heads of slice s % SLICES over
    // key group s / SLICES.
    threadgroup float *states = shared;
    threadgroup float *columns = states + SIMDS * H * 2;
    threadgroup attention::Scalar *prepared = reinterpret_cast<threadgroup attention::Scalar *>(columns);
    for (uint g = simd; g < G; g += SIMDS) {
        float x[E];
        attention::head_rotary<ATTENTION_NORM>(query + (row * KV * G + kv_head * G + g) * ATTENTION_QUERY_STRIDE,
            query_norm, row_coordinates, rotary_components, rotary_frequencies, rotary_amplitudes, epsilon,
            lane, x);
        ATTENTION_UNROLL
        for (uint i = 0; i < E; ++i)
            prepared[g * W + lane * E + i] = attention::Scalar(x[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint head0 = (simd % SLICES) * H;
    float q[H][E];
    float qsum[H];
    float maximum[H];
    float denominator[H];
    float output[H][E];
    float bias[H];
    ATTENTION_UNROLL
    for (uint h = 0; h < H; ++h) {
        float sum = 0.0f;
        ATTENTION_UNROLL
        for (uint i = 0; i < E; ++i) {
            q[h][i] = float(prepared[(head0 + h) * W + lane * E + i]) * (scale * ATTENTION_LOG2E);
            sum += q[h][i];
            output[h][i] = 0.0f;
        }
        qsum[h] = sum;
        maximum[h] = -INFINITY;
        denominator[h] = 0.0f;
        bias[h] = 0.0f;
    }

    constexpr uint GROUPS = SIMDS / SLICES;
    const uint sub = (partition_hi - partition_lo + GROUPS - 1) / GROUPS;
    const uint first = partition_lo + (simd / SLICES) * sub;
    const uint last = metal::min(first + sub, partition_hi);
    uint offset = 0;
    for (ulong span = 0; span <= R && offset < last; ++span) {
        const bool historical = span < R;
        int lo, hi;
        attention::form_span(visible, fresh, row, R, span, lo, hi);
        const uint length = uint(metal::max(hi - lo, 0));
        const uint begin = metal::max(first, offset);
        const uint end = metal::min(last, offset + length);
        uint position = begin;
        if (historical) {
            // Each slab's part of the span: its region is resolved once, and
            // this kv head's planes of the part's first token advance by
            // 32-bit steps of one history row (per-token 64-bit vector
            // arithmetic cost a quarter of the loop on M4).
            const ulong slab_rows = ulong(SEISMIC_PARAM_SLAB_ROWS);
            constexpr uint key_step = KV * key_lane::row_words;
            constexpr uint value_step = KV * value_lane::row_words;
            constexpr uint key_pair_step = KV * key_lane::pairs;
            constexpr uint value_pair_step = KV * value_lane::pairs;
            while (position < end) {
                const ulong token = ulong(lo) + (position - offset);
                const ulong slab_index = token / slab_rows;
                const uint part_end =
                    uint(metal::min(ulong(end), (slab_index + 1) * slab_rows - ulong(lo) + ulong(offset)));
                const ulong vector = (token - slab_index * slab_rows) * KV + kv_head;
                device const uint *key_rows =
                    slab::region<uint>(key_codes, slab_index) + vector * key_lane::row_words;
                device const uint *value_rows =
                    slab::region<uint>(value_codes, slab_index) + vector * value_lane::row_words;
                device const half2 *key_pairs = reinterpret_cast<device const half2 *>(
                    slab::region<half>(key_coefficients, slab_index)) + vector * key_lane::pairs
                    + lane / key_lane::pair_lanes;
                device const half2 *value_pairs = reinterpret_cast<device const half2 *>(
                    slab::region<half>(value_coefficients, slab_index)) + vector * value_lane::pairs
                    + lane / value_lane::pair_lanes;
                uint t = 0;
                for (; position + DECODE_BATCH <= part_end; position += DECODE_BATCH, t += DECODE_BATCH) {
                    uint k[DECODE_BATCH][key_lane::words];
                    uint v[DECODE_BATCH][value_lane::words];
                    float2 kc[DECODE_BATCH];
                    float2 vc[DECODE_BATCH];
                    ATTENTION_UNROLL
                    for (uint j = 0; j < DECODE_BATCH; ++j) {
                        key_lane::load(key_rows + (t + j) * key_step, lane, k[j]);
                        value_lane::load(value_rows + (t + j) * value_step, lane, v[j]);
                        kc[j] = float2(key_pairs[(t + j) * key_pair_step]);
                        vc[j] = float2(value_pairs[(t + j) * value_pair_step]);
                    }
                    attention::absorb_affine<DECODE_BATCH>(q, qsum, k, kc, v, vc, maximum, denominator,
                        output, bias);
                }
                for (; position < part_end; ++position, ++t) {
                    uint k[1][key_lane::words];
                    uint v[1][value_lane::words];
                    float2 kc[1];
                    float2 vc[1];
                    key_lane::load(key_rows + t * key_step, lane, k[0]);
                    value_lane::load(value_rows + t * value_step, lane, v[0]);
                    kc[0] = float2(key_pairs[t * key_pair_step]);
                    vc[0] = float2(value_pairs[t * value_pair_step]);
                    attention::absorb_affine<1>(q, qsum, k, kc, v, vc, maximum, denominator, output, bias);
                }
            }
        } else {
            // The dense absorb carries only the output: fold the bias in first.
            ATTENTION_UNROLL
            for (uint h = 0; h < H; ++h) {
                ATTENTION_UNROLL
                for (uint i = 0; i < E; ++i)
                    output[h][i] += bias[h];
                bias[h] = 0.0f;
            }
            for (; position < end; ++position) {
                const ulong at = (ulong(lo) + (position - offset)) * KV * W + kv_head * W;
                float k[1][E];
                float v[1][E];
                attention::head_rotary<ATTENTION_NORM>(key + at, key_norm,
                    coordinates + (ulong(lo) + (position - offset)) * 4, rotary_components, rotary_frequencies,
                    rotary_amplitudes, epsilon, lane, k[0]);
                attention::head_norm<ATTENTION_VALUE_NORM>(value + at, value_norm, epsilon, lane, v[0]);
                ATTENTION_UNROLL
                for (uint i = 0; i < E; ++i) {
                    k[0][i] = float(attention::Scalar(k[0][i]));
                    v[0][i] = float(attention::Scalar(v[0][i]));
                }
                attention::absorb_heads<H, 1>(q, k, v, maximum, denominator, output);
            }
        }
        offset += length;
    }
    ATTENTION_UNROLL
    for (uint h = 0; h < H; ++h) {
        ATTENTION_UNROLL
        for (uint i = 0; i < E; ++i)
            output[h][i] += bias[h];
    }

    attention::publish_slices<SIMDS, PARTS, H, SLICES>(maximum, denominator, output, states, columns,
        partials, statistics, (row * KV + kv_head) * G * PARTS + partition, simd, lane, thread_index);
}

// L2: threadgroup (query head, row), one thread per column: the fixed-order
// merge of the row's partitions and the output gate.
kernel void attention_decode_k8v4_merge(
    device const attention::Scalar *query [[buffer(SEISMIC_BUFFER_QUERY)]],
    device const attention::Scalar *gate [[buffer(SEISMIC_BUFFER_GATE)]],
    device const int *visible [[buffer(SEISMIC_BUFFER_VISIBLE)]],
    device const int *fresh [[buffer(SEISMIC_BUFFER_FRESH)]],
    device attention::Scalar *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    device const float *partials [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIALS)]],
    device const float *statistics [[buffer(SEISMIC_BUFFER_SCRATCH_STATISTICS)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint column [[thread_index_in_threadgroup]]) {
    attention::decode_output<SEISMIC_TUNE_SPAN, SEISMIC_TUNE_PARTS>(query, gate, visible, fresh, result,
        partials, statistics, SEISMIC_DIM_R, group.x, group.y, column, SEISMIC_PARAM_GATE_FUNCTION != 0);
}
