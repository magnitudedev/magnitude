#define ATTENTION_QUERY_GROUP SEISMIC_DIM_G
#define ATTENTION_I SEISMIC_DIM_I
#define ATTENTION_U SEISMIC_DIM_U
#define ATTENTION_FRESH (SEISMIC_DIM_F != 0)
#define ATTENTION_NORM (SEISMIC_DIM_N != 0)
#define ATTENTION_QUERY_NORM (SEISMIC_DIM_NQ != 0)
#define ATTENTION_VALUE_NORM (SEISMIC_DIM_NV != 0)
#define ATTENTION_SINKS (SEISMIC_DIM_SK != 0)
#define ATTENTION_STORED_KEY (SEISMIC_DIM_VK != 0)
#include "lib/attention/attention.h"

// Keys each simdgroup loads before scoring them, so their loads are in flight
// together (fewer for wide heads, whose key and value columns fill the
// registers next to the queries and outputs).
#define DECODE_BATCH (ATTENTION_E >= 16 ? 2 : 4)

// The launch rows [row0, row0 + TOKENS)' keys and values (a layer with fresh
// rows) appended at their destinations, a simdgroup of SIMDS per (row, key
// or value), the key rounded by the trained key rounding (KR = 1);
// stored-key values are the key plane.
template <uint SIMDS, uint TOKENS>
inline void append_rows(device const attention::Scalar *key, device const attention::Scalar *value,
    device const float *key_norm, device const float *value_norm, device const int *rotary_components,
    device const float *rotary_frequencies, device const float *rotary_amplitudes, device const int *coordinates,
    device const int *destinations, device const ulong *history_key, device const ulong *history_value,
    constant ulong *seismic_words, rounding::Rounding key_rounding, ulong row0, uint kv_head, uint simd,
    uint lane) {
    constexpr uint W = ATTENTION_W;
    constexpr uint E = ATTENTION_E;
    constexpr uint KV = SEISMIC_DIM_KV;
    const float epsilon = as_type<float>(uint(SEISMIC_PARAM_EPSILON));
    for (uint item = simd; item < 2 * TOKENS; item += SIMDS) {
        const ulong appended = row0 + item / 2;
        if (appended >= SEISMIC_DIM_M)
            break;
        const int destination = destinations[appended];
        if (destination < 0 || (ATTENTION_STORED_KEY && item % 2 != 0))
            continue;
        float x[E];
        if (item % 2 == 0)
            attention::fresh_row<true>(key, value, key_norm, value_norm, coordinates, rotary_components,
                rotary_frequencies, rotary_amplitudes, appended, kv_head, epsilon, key_rounding, lane, x);
        else
            attention::fresh_row<false>(key, value, key_norm, value_norm, coordinates, rotary_components,
                rotary_frequencies, rotary_amplitudes, appended, kv_head, epsilon, key_rounding, lane, x);
        attention::append(slab::row<attention::Scalar>(item % 2 == 0 ? history_key : history_value,
            ulong(destination), ulong(SEISMIC_PARAM_SLAB_ROWS), KV * W), 0, kv_head, lane, x);
    }
}

// L1 of the vector and matrix forms (MATRIX 0 and 1; the shared-tile form is
// `attention_decode_shared`): threadgroup (kv head and head group, partition,
// row). With MATRIX 1, the grouped-query matrix form (attention::decode_matrix) over a head group of
// min(HEADS, G) query heads, whose key sequence ends in the row's KS picks.
// Otherwise (one head group, no picks): a row's visible keys,
// in order (spans, then its fresh span), split into equal partitions of
// attention::partition_span keys. The G query heads split into SLICES slices
// of H = G / SLICES heads (so wide heads keep their queries and outputs in
// registers); each simdgroup scans a contiguous sub-range (its key group) of
// the partition for the heads of its slice, so each key/value row is loaded
// once per slice. Scores are F32 in the exp2 domain (scale * log2e folded
// into the query). The key groups' states merge in fixed order into one
// partial per (row, query head, partition): the unnormalized output relative
// to the partition maximum, and (maximum, denominator). Partition 0 of a
// layer with fresh rows (its first head group's) also appends the row's key,
// rounded by the trained key rounding (KR = 1), and value at its
// destination; a fresh key is attended as it is appended.
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
    device const int *selected_rows [[buffer(SEISMIC_BUFFER_SELECTED_ROWS)]],
    device const ulong *selected [[buffer(SEISMIC_BUFFER_SELECTED)]],
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
    // The kv head's head groups of GROUP query heads (one unless the matrix
    // form splits G).
    constexpr uint HEADS = SEISMIC_TUNE_HEADS;
    constexpr uint GROUP = HEADS < G ? HEADS : G;
    constexpr uint HEAD_GROUPS = G / GROUP;
    const uint kv_head = group.x / HEAD_GROUPS;
    const uint head_base = group.x % HEAD_GROUPS * GROUP;
    const uint partition = group.y;
    const ulong row0 = group.z * TOKENS;
    const ulong row = row0;
    const float epsilon = as_type<float>(uint(SEISMIC_PARAM_EPSILON));
    const float scale = as_type<float>(uint(SEISMIC_PARAM_SCALE));
    device const int *row_coordinates = coordinates + row * 4;
    const rounding::Rounding key_rounding = attention::key_rounding(SEISMIC_DIM_KR != 0,
        uint(SEISMIC_PARAM_KEY_BLOCK), uint(SEISMIC_PARAM_KEY_ELEMENT), uint(SEISMIC_PARAM_KEY_SCALE),
        uint(SEISMIC_PARAM_KEY_AMAX_FLOOR));

    // Partition 0 of the first head group appends the rows' keys and values.
    if (ATTENTION_FRESH && partition == 0 && head_base == 0)
        append_rows<SIMDS, TOKENS>(key, value, key_norm, value_norm, rotary_components, rotary_frequencies,
            rotary_amplitudes, coordinates, destinations, history_key, history_value, seismic_words, key_rounding,
            row0, kv_head, simd, lane);

    if (int(uint(SEISMIC_PARAM_GATE_FUNCTION)) == -1)
        return;
#if SEISMIC_TUNE_MATRIX == 1
    attention::decode_matrix<SIMDS, SEISMIC_TUNE_KEYS, PARTS, TOKENS, 0, HEADS, SEISMIC_DIM_KS>(
        attention::dense_history{history_key, attention::values_plane(history_key, history_value),
            ulong(SEISMIC_PARAM_SLAB_ROWS)},
        query, key, value, query_norm, key_norm, value_norm, rotary_components, rotary_frequencies,
        rotary_amplitudes, coordinates, visible, fresh, partials, statistics,
        reinterpret_cast<threadgroup uchar *>(shared), R, SEISMIC_DIM_M, epsilon, scale * ATTENTION_LOG2E,
        SEISMIC_TUNE_SPAN, kv_head, partition, group.z, thread_index, simd, lane, head_base,
        attention::decode_picks{selected_rows, selected, ulong(SEISMIC_PARAM_SELECTED_SLAB_ROWS)}, key_rounding);
#elif SEISMIC_TUNE_MATRIX == 0
    static_assert(GROUP == G && SEISMIC_DIM_KS == 0, "the vector form holds a whole query group and no picks");
    const uint total = attention::form_total(visible, fresh, row, R, lane);
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
        attention::head_rotary<ATTENTION_QUERY_NORM>(query + (row * KV * G + kv_head * G + g) * ATTENTION_QUERY_STRIDE,
            query_norm, row_coordinates, rotary_components, rotary_frequencies, rotary_amplitudes, epsilon,
            lane, x);
        ATTENTION_UNROLL
        for (uint i = 0; i < E; ++i)
            reinterpret_cast<threadgroup element::Act::storage *>(prepared)[g * W + lane * E + i] =
                element::Act::store(x[i]);
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
    // The key group's keys absorb in batches of DECODE_BATCH at fixed
    // positions first + i * DECODE_BATCH, then one at a time, read through
    // its cursor (attention::key_cursor): a key's batch, and so the row's
    // reduction order, depends on its position alone.
    attention::key_cursor cursor =
        attention::form_cursor(visible, fresh, row, R, ulong(SEISMIC_PARAM_SLAB_ROWS), first, lane);
    // The cursor's slab's planes.
    device const attention::Scalar *key_slab = nullptr;
    device const attention::Scalar *value_slab = nullptr;
    // Key `position` of the row's sequence (at or past the cursor's) into
    // `k` and `v`: a history row from its slab, a fresh row as it is
    // appended (prepared, as history holds it); a stored-key value is the
    // key.
    auto read = [&](uint position, thread float (&k)[E], thread float (&v)[E]) {
        bool moved;
        const ulong token = cursor.seek(position, moved);
        if (cursor.historical()) {
            if (moved) {
                key_slab = slab::region<attention::Scalar>(history_key, cursor.slab);
                value_slab = slab::region<attention::Scalar>(attention::values_plane(history_key, history_value),
                    cursor.slab);
            }
            const ulong at = ((token - cursor.slab_lo) * KV + kv_head) * W + lane * E;
            element::span<element::Act>(key_slab + at, k);
            element::span<element::Act>(value_slab + at, v);
        } else {
            attention::fresh_row<true>(key, value, key_norm, value_norm, coordinates, rotary_components,
                rotary_frequencies, rotary_amplitudes, token, kv_head, epsilon, key_rounding, lane, k);
            attention::dense_history::held<true>(k, lane);
            if (ATTENTION_STORED_KEY) {
                ATTENTION_UNROLL
                for (uint i = 0; i < E; ++i)
                    v[i] = k[i];
            } else {
                attention::fresh_row<false>(key, value, key_norm, value_norm, coordinates, rotary_components,
                    rotary_frequencies, rotary_amplitudes, token, kv_head, epsilon, key_rounding, lane, v);
                attention::dense_history::held<false>(v, lane);
            }
        }
    };
    uint position = first;
    for (; position + DECODE_BATCH <= last; position += DECODE_BATCH) {
        float k[DECODE_BATCH][E];
        float v[DECODE_BATCH][E];
        ATTENTION_UNROLL
        for (uint j = 0; j < DECODE_BATCH; ++j)
            read(position + j, k[j], v[j]);
        attention::absorb_heads<H, DECODE_BATCH>(q, k, v, maximum, denominator, output, lane);
    }
    for (; position < last; ++position) {
        float k[1][E];
        float v[1][E];
        read(position, k[0], v[0]);
        attention::absorb_heads<H, 1>(q, k, v, maximum, denominator, output, lane);
    }

    attention::publish_slices<SIMDS, PARTS, H, SLICES>(maximum, denominator, output, states, columns,
        partials, statistics, (row * KV + kv_head) * G * PARTS + partition, simd, lane, thread_index);
#endif
}

// L1 of the shared-tile matrix form (MATRIX = 2, attention::decode_shared):
// threadgroup (kv head, partition, row), its ceil(G / 8) row blocks by
// column slices of simdgroups holding all G query heads of its kv head.
// Partition 0 also appends the row's key and value, as the partial launch's.
kernel void attention_decode_shared(
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
    device const int *selected_rows [[buffer(SEISMIC_BUFFER_SELECTED_ROWS)]],
    device const ulong *selected [[buffer(SEISMIC_BUFFER_SELECTED)]],
    device float *partials [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIALS)]],
    device float *statistics [[buffer(SEISMIC_BUFFER_SCRATCH_STATISTICS)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    threadgroup float *shared [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
#if SEISMIC_TUNE_MATRIX == 2
    const uint kv_head = group.x;
    const uint partition = group.y;
    const ulong row = group.z;
    const rounding::Rounding key_rounding = attention::key_rounding(SEISMIC_DIM_KR != 0,
        uint(SEISMIC_PARAM_KEY_BLOCK), uint(SEISMIC_PARAM_KEY_ELEMENT), uint(SEISMIC_PARAM_KEY_SCALE),
        uint(SEISMIC_PARAM_KEY_AMAX_FLOOR));
    if (ATTENTION_FRESH && partition == 0)
        append_rows<attention::shared_simdgroups(), 1>(key, value, key_norm, value_norm, rotary_components, rotary_frequencies,
            rotary_amplitudes, coordinates, destinations, history_key, history_value, seismic_words, key_rounding,
            row, kv_head, simd, lane);
    if (int(uint(SEISMIC_PARAM_GATE_FUNCTION)) == -1)
        return;
    attention::decode_shared<SEISMIC_TUNE_KEYS, SEISMIC_TUNE_PARTS, SEISMIC_DIM_KS>(
        attention::dense_history{history_key, attention::values_plane(history_key, history_value),
            ulong(SEISMIC_PARAM_SLAB_ROWS)},
        query, key, value, query_norm, key_norm, value_norm, rotary_components, rotary_frequencies,
        rotary_amplitudes, coordinates, visible, fresh, partials, statistics,
        reinterpret_cast<threadgroup uchar *>(shared), SEISMIC_DIM_R, as_type<float>(uint(SEISMIC_PARAM_EPSILON)),
        as_type<float>(uint(SEISMIC_PARAM_SCALE)) * ATTENTION_LOG2E, SEISMIC_TUNE_SPAN, kv_head, partition, row,
        thread_index, simd, lane,
        attention::decode_picks{selected_rows, selected, ulong(SEISMIC_PARAM_SELECTED_SLAB_ROWS)}, key_rounding);
#endif
}

// L2: threadgroup (query head, row, 32-column group), one thread per column:
// the fixed-order merge of the row's partitions (of its visible keys and KS
// picks; with the head's sink when SK = 1) and the output gate. With the
// output rotary (RO = 1), one threadgroup holds the head's whole row, which
// it turns back in F32 before publishing it.
kernel void attention_decode_merge(
    device const attention::Scalar *query [[buffer(SEISMIC_BUFFER_QUERY)]],
    device const attention::Scalar *gate [[buffer(SEISMIC_BUFFER_GATE)]],
    device const float *sinks [[buffer(SEISMIC_BUFFER_SINKS)]],
    device const int *rotary_components [[buffer(SEISMIC_BUFFER_ROTARY_COMPONENTS)]],
    device const float *rotary_frequencies [[buffer(SEISMIC_BUFFER_ROTARY_FREQUENCIES)]],
    device const float *rotary_amplitudes [[buffer(SEISMIC_BUFFER_ROTARY_AMPLITUDES)]],
    device const int *coordinates [[buffer(SEISMIC_BUFFER_COORDINATES)]],
    device const int *visible [[buffer(SEISMIC_BUFFER_VISIBLE)]],
    device const int *fresh [[buffer(SEISMIC_BUFFER_FRESH)]],
    device attention::Scalar *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    device const float *partials [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIALS)]],
    device const float *statistics [[buffer(SEISMIC_BUFFER_SCRATCH_STATISTICS)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (int(uint(SEISMIC_PARAM_GATE_FUNCTION)) == -1)
        return;
    constexpr bool ROTATED = SEISMIC_DIM_RO != 0;
    threadgroup float weights[2 * SEISMIC_TUNE_PARTS];
    threadgroup float row_values[ROTATED ? ATTENTION_W : 1];
    attention::decode_output<SEISMIC_TUNE_SPAN, SEISMIC_TUNE_PARTS, SEISMIC_TUNE_TOKENS, SEISMIC_DIM_KS, ROTATED>(
        query, gate, visible, fresh, result, partials, statistics, weights, SEISMIC_DIM_R, SEISMIC_DIM_M, group.x,
        group.y, ROTATED ? thread_index : group.z * 32 + lane, lane, SEISMIC_PARAM_GATE_FUNCTION != 0, sinks,
        thread_index < 32, row_values, coordinates, rotary_components, rotary_frequencies, rotary_amplitudes);
}
