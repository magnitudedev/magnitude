#define ATTENTION_QUERY_GROUP SEISMIC_DIM_G
#define ATTENTION_I SEISMIC_DIM_I
#define ATTENTION_U SEISMIC_DIM_U
#define ATTENTION_FRESH (SEISMIC_DIM_F != 0)
#define ATTENTION_NORM (SEISMIC_DIM_N != 0)
#define ATTENTION_VALUE_NORM (SEISMIC_DIM_NV != 0)
#include "lib/attention/attention.h"

// Keys each simdgroup loads before scoring them, so their loads are in flight
// together (fewer for wide heads, whose key and value columns fill the
// registers next to the queries and outputs).
#define DECODE_BATCH (ATTENTION_E >= 16 ? 2 : 4)

// L1: threadgroup (kv head, partition, row). With MATRIX, the grouped-query
// matrix form (attention::decode_matrix). Otherwise: a row's visible keys,
// in order (spans, then its fresh span), split into equal partitions of
// attention::partition_span keys. The G query heads split into SLICES slices
// of H = G / SLICES heads (so wide heads keep their queries and outputs in
// registers); each simdgroup scans a contiguous sub-range (its key group) of
// the partition for the heads of its slice, so each key/value row is loaded
// once per slice. Scores are F32 in the exp2 domain (scale * log2e folded
// into the query). The key groups' states merge in fixed order into one
// partial per (row, query head, partition): the unnormalized output relative
// to the partition maximum, and (maximum, denominator). Partition 0 of a
// layer with fresh rows also appends the row's key and value at its
// destination.
kernel void attention_decode_partial(
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
    device const ulong *history_key [[buffer(SEISMIC_BUFFER_HISTORY_KEY)]],
    device const ulong *history_value [[buffer(SEISMIC_BUFFER_HISTORY_VALUE)]],
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
    constexpr uint SLICES = SEISMIC_TUNE_SLICES;
    constexpr uint H = G / SLICES;
    const ulong R = SEISMIC_DIM_R;
    // Rows per threadgroup: TOKENS decode rows packed into the matrix form's
    // rows; one row otherwise.
    constexpr uint TOKENS = SEISMIC_TUNE_TOKENS;
    const uint kv_head = group.x;
    const uint partition = group.y;
    const ulong row0 = group.z * TOKENS;
    const ulong row = row0;
    const float epsilon = as_type<float>(uint(SEISMIC_PARAM_EPSILON));
    const float scale = as_type<float>(uint(SEISMIC_PARAM_SCALE));
    device const int *row_coordinates = coordinates + row * 4;

    // Partition 0 appends the rows' keys and values, a simdgroup per (row,
    // key or value).
    if (ATTENTION_FRESH && partition == 0) {
        for (uint item = simd; item < 2 * TOKENS; item += SIMDS) {
            const ulong appended = row0 + item / 2;
            if (appended >= SEISMIC_DIM_M)
                break;
            const int destination = destinations[appended];
            if (destination < 0)
                continue;
            const ulong source = (appended * KV + kv_head) * W;
            float x[E];
            if (item % 2 == 0)
                attention::head_rotary<ATTENTION_NORM>(key + source, key_norm, coordinates + appended * 4,
                    rotary_components, rotary_frequencies, rotary_amplitudes, epsilon, lane, x);
            else
                attention::head_norm<ATTENTION_VALUE_NORM>(value + source, value_norm, epsilon, lane, x);
            attention::append(slab::row<attention::Scalar>(item % 2 == 0 ? history_key : history_value,
                ulong(destination), ulong(SEISMIC_PARAM_SLAB_ROWS), KV * W), 0, kv_head, lane, x);
        }
    }

    if (int(uint(SEISMIC_PARAM_GATE_FUNCTION)) == -1)
        return;
#if SEISMIC_TUNE_MATRIX
    attention::decode_matrix<SIMDS, SEISMIC_TUNE_KEYS, PARTS, TOKENS>(attention::dense_history{history_key, history_value,
        ulong(SEISMIC_PARAM_SLAB_ROWS)}, query, key, value, query_norm, key_norm, value_norm, rotary_components,
        rotary_frequencies, rotary_amplitudes, coordinates, visible, fresh, partials, statistics,
        reinterpret_cast<threadgroup uchar *>(shared), R, SEISMIC_DIM_M, epsilon, scale * ATTENTION_LOG2E, SEISMIC_TUNE_SPAN,
        kv_head, partition, group.z, thread_index, simd, lane);
#else
    const uint total = attention::form_total(visible, fresh, row, R);
    const uint span_keys = attention::partition_span(total, SEISMIC_TUNE_SPAN, PARTS);
    const uint partition_lo = partition * span_keys;
    if (partition_lo >= total)
        return;
    const uint partition_hi = metal::min(partition_lo + span_keys, total);

    // Threadgroup memory: the simdgroup states [SIMDS][H][2] (floats), then
    // the prepared queries [G][W] (activation dtype, as the contract publishes
    // them), which the merge columns [SIMDS][W] (floats) alias: every
    // simdgroup has read its queries before the publication's first barrier.
    // Simdgroup s holds the H query heads of slice s % SLICES over key group
    // s / SLICES.
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
    float maximum[H];
    float denominator[H];
    float output[H][E];
    ATTENTION_UNROLL
    for (uint h = 0; h < H; ++h) {
        ATTENTION_UNROLL
        for (uint i = 0; i < E; ++i) {
            q[h][i] = float(prepared[(head0 + h) * W + lane * E + i]) * (scale * ATTENTION_LOG2E);
            output[h][i] = 0.0f;
        }
        maximum[h] = -INFINITY;
        denominator[h] = 0.0f;
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
            // this lane's columns of the part's first token advance by 32-bit
            // steps of one history row.
            const ulong slab_rows = ulong(SEISMIC_PARAM_SLAB_ROWS);
            constexpr uint step = KV * W;
            while (position < end) {
                const ulong token = ulong(lo) + (position - offset);
                const ulong slab_index = token / slab_rows;
                const uint part_end =
                    uint(metal::min(ulong(end), (slab_index + 1) * slab_rows - ulong(lo) + ulong(offset)));
                const ulong at = ((token - slab_index * slab_rows) * KV + kv_head) * W + lane * E;
                device const attention::Scalar *key_rows = slab::region<attention::Scalar>(history_key, slab_index) + at;
                device const attention::Scalar *value_rows =
                    slab::region<attention::Scalar>(history_value, slab_index) + at;
                uint t = 0;
                for (; position + DECODE_BATCH <= part_end; position += DECODE_BATCH, t += DECODE_BATCH) {
                    float k[DECODE_BATCH][E];
                    float v[DECODE_BATCH][E];
                    ATTENTION_UNROLL
                    for (uint j = 0; j < DECODE_BATCH; ++j) {
                        element::span<element::Act>(key_rows + (t + j) * step, k[j]);
                        element::span<element::Act>(value_rows + (t + j) * step, v[j]);
                    }
                    attention::absorb_heads<H, DECODE_BATCH>(q, k, v, maximum, denominator, output, lane);
                }
                for (; position < part_end; ++position, ++t) {
                    float k[1][E];
                    float v[1][E];
                    element::span<element::Act>(key_rows + t * step, k[0]);
                    element::span<element::Act>(value_rows + t * step, v[0]);
                    attention::absorb_heads<H, 1>(q, k, v, maximum, denominator, output, lane);
                }
            }
        }
        for (; position < end; ++position) {
            const ulong token = ulong(lo) + (position - offset);
            float k[1][E];
            float v[1][E];
            const ulong at = (token * KV + kv_head) * W;
            attention::head_rotary<ATTENTION_NORM>(key + at, key_norm, coordinates + token * 4,
                rotary_components, rotary_frequencies, rotary_amplitudes, epsilon, lane, k[0]);
            attention::head_norm<ATTENTION_VALUE_NORM>(value + at, value_norm, epsilon, lane, v[0]);
            ATTENTION_UNROLL
            for (uint i = 0; i < E; ++i) {
                k[0][i] = float(attention::Scalar(k[0][i]));
                v[0][i] = float(attention::Scalar(v[0][i]));
            }
            attention::absorb_heads<H, 1>(q, k, v, maximum, denominator, output, lane);
        }
        offset += length;
    }

    attention::publish_slices<SIMDS, PARTS, H, SLICES>(maximum, denominator, output, states, columns,
        partials, statistics, (row * KV + kv_head) * G * PARTS + partition, simd, lane, thread_index);
#endif
}

// L2: threadgroup (query head, row), one thread per column: the fixed-order
// merge of the row's partitions and the output gate.
kernel void attention_decode_merge(
    device const attention::Scalar *query [[buffer(SEISMIC_BUFFER_QUERY)]],
    device const attention::Scalar *gate [[buffer(SEISMIC_BUFFER_GATE)]],
    device const int *visible [[buffer(SEISMIC_BUFFER_VISIBLE)]],
    device const int *fresh [[buffer(SEISMIC_BUFFER_FRESH)]],
    device attention::Scalar *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    device const float *partials [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIALS)]],
    device const float *statistics [[buffer(SEISMIC_BUFFER_SCRATCH_STATISTICS)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    if (int(uint(SEISMIC_PARAM_GATE_FUNCTION)) == -1)
        return;
    threadgroup float weights[2 * SEISMIC_TUNE_PARTS];
    attention::decode_output<SEISMIC_TUNE_SPAN, SEISMIC_TUNE_PARTS, SEISMIC_TUNE_TOKENS>(query, gate, visible, fresh, result,
        partials, statistics, weights, SEISMIC_DIM_R, SEISMIC_DIM_M, group.x, group.y, group.z * 32 + lane, lane,
        SEISMIC_PARAM_GATE_FUNCTION != 0);
}
