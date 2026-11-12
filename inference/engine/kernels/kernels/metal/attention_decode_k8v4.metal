#define ATTENTION_AFFINE_HISTORY 1
#define ATTENTION_KEY_CODE_WORDS SEISMIC_DIM_KC
#define ATTENTION_VALUE_CODE_WORDS SEISMIC_DIM_VC
#define ATTENTION_KEY_COEFFICIENTS SEISMIC_DIM_KP
#define ATTENTION_QUERY_GROUP SEISMIC_DIM_G
#define ATTENTION_I SEISMIC_DIM_I
#define ATTENTION_U SEISMIC_DIM_U
#define ATTENTION_FRESH (SEISMIC_DIM_F != 0)
#define ATTENTION_NORM (SEISMIC_DIM_N != 0)
#define ATTENTION_QUERY_NORM (SEISMIC_DIM_NQ != 0)
#define ATTENTION_VALUE_NORM (SEISMIC_DIM_NV != 0)
#define ATTENTION_SINKS (SEISMIC_DIM_SK != 0)
#include "lib/attention/attention.h"

// Keys each simdgroup reads before scoring them: the dense batch.
// The affine absorb is bound by per-key work, not bytes in flight; larger
// batches only raise register pressure (measured 8 and 16 slower on M4).
#define DECODE_BATCH 4

// L1: threadgroup (kv head, partition, row), as `attention_decode`, over
// affine K8/V4 history. With MATRIX, the grouped-query matrix form
// (attention::decode_matrix): tiles are decoded to F16 as they are staged
// and every product takes F16 operands. Otherwise the vector form: keys
// score as the sum over groups of
// scale * (q . code) + zero * sum(q) and values accumulate (p * scale) * code
// plus the carried bias sum(p * zero) of each lane's group
// (attention::absorb_affine), so no element is decoded. A fresh key and value
// are read as their encoding (attention::held_encoding), the one history
// holds, in both forms. Partition 0 of a layer with fresh rows appends the
// row's key (prepared, rounded to the activation dtype) and value as that
// encoding (attention::store).
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
    // Rows per threadgroup: TOKENS decode rows packed into the matrix form's
    // rows; one row otherwise.
    constexpr uint TOKENS = SEISMIC_TUNE_TOKENS;
    constexpr uint HEAD_BAND = SEISMIC_TUNE_HEAD_BAND;
    constexpr uint HEAD_BANDS = HEAD_BAND == 0 ? 1 : (G + HEAD_BAND - 1) / HEAD_BAND;
    const uint kv_head = group.x / HEAD_BANDS;
    const uint head_band = group.x % HEAD_BANDS;
    const uint head_base = head_band * HEAD_BAND;
    const uint partition = group.y;
    const ulong row0 = group.z * TOKENS;
    const ulong row = row0;
    const float epsilon = as_type<float>(uint(SEISMIC_PARAM_EPSILON));
    const float scale = as_type<float>(uint(SEISMIC_PARAM_SCALE));
    device const int *row_coordinates = coordinates + row * 4;

    // Fresh row `token`'s key and value as history holds them
    // (attention::held_encoding): what partition 0 appends and what every
    // read of the row absorbs. The whole simdgroup calls them.
    auto fresh_key = [&](ulong token) {
        float x[E];
        attention::fresh_row<true>(key, value, key_norm, value_norm, coordinates, rotary_components,
            rotary_frequencies, rotary_amplitudes, token, kv_head, epsilon, rounding::none(), lane, x);
        return attention::held_encoding<true>(x);
    };
    auto fresh_value = [&](ulong token) {
        float x[E];
        attention::fresh_row<false>(key, value, key_norm, value_norm, coordinates, rotary_components,
            rotary_frequencies, rotary_amplitudes, token, kv_head, epsilon, rounding::none(), lane, x);
        return attention::held_encoding<false>(x);
    };

    // Partition 0 appends the rows' keys and values encoded, a simdgroup per
    // (row, key or value).
    for (uint item = simd; ATTENTION_FRESH && partition == 0 && head_band == 0 && item < 2 * TOKENS; item += SIMDS) {
        const ulong appended = row0 + item / 2;
        if (appended >= SEISMIC_DIM_M)
            break;
        const int destination = destinations[appended];
        if (destination >= 0) {
            const ulong vector = kv_head;
            const ulong slab_rows = ulong(SEISMIC_PARAM_SLAB_ROWS);
            if (item % 2 == 0)
                attention::store<ATTENTION_KEY_BITS>(fresh_key(appended),
                    slab::row<uint>(key_codes, ulong(destination), slab_rows, KV * key_lane::row_words)
                        + vector * key_lane::row_words,
                    slab::row<half>(key_coefficients, ulong(destination), slab_rows, KV * key_lane::pairs * 2)
                        + vector * key_lane::pairs * 2, lane);
            else
                attention::store<ATTENTION_VALUE_BITS>(fresh_value(appended),
                    slab::row<uint>(value_codes, ulong(destination), slab_rows, KV * value_lane::row_words)
                        + vector * value_lane::row_words,
                    slab::row<half>(value_coefficients, ulong(destination), slab_rows, KV * value_lane::pairs * 2)
                        + vector * value_lane::pairs * 2, lane);
        }
    }

    if (int(uint(SEISMIC_PARAM_GATE_FUNCTION)) == -1)
        return;
#if SEISMIC_TUNE_MATRIX
    attention::decode_matrix<SIMDS, SEISMIC_TUNE_KEYS, PARTS, TOKENS, HEAD_BAND>(attention::affine_history{key_codes,
        key_coefficients, value_codes, value_coefficients, ulong(SEISMIC_PARAM_SLAB_ROWS)}, query, key, value,
        query_norm, key_norm, value_norm, rotary_components, rotary_frequencies, rotary_amplitudes, coordinates,
        visible, fresh, partials, statistics, reinterpret_cast<threadgroup uchar *>(shared), R, SEISMIC_DIM_M, epsilon,
        scale * ATTENTION_LOG2E, SEISMIC_TUNE_SPAN, kv_head, partition, group.z, thread_index, simd, lane, head_base);
#else
    const uint total = attention::form_total(visible, fresh, row, R, lane);
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
    // The key group's keys absorb in batches of DECODE_BATCH at fixed
    // positions first + i * DECODE_BATCH up to `singles`, then one at a time:
    // a key's batch, and so the row's reduction order, depends on its
    // position alone, wherever the row's spans, slabs and fresh rows fall.
    // The spans are walked as main walks them, each history span by slab
    // parts whose planes are resolved once. A batch wholly inside a part
    // loads its rows directly; the keys of a batch that a part or span edge
    // cuts (and every fresh row, as partition 0 appends it) are gathered one
    // at a time into the pending batch, which absorbs once full. The pending
    // count is (position - first) % DECODE_BATCH below `singles`.
    const uint singles = first + (last > first ? (last - first) / DECODE_BATCH * DECODE_BATCH : 0);
    uint pk[DECODE_BATCH][key_lane::words];
    uint pv[DECODE_BATCH][value_lane::words];
    float2 pkc[DECODE_BATCH];
    float2 pvc[DECODE_BATCH];
    uint pending = 0;
    // Key `position` (its codes and pairs read): gathered into the pending
    // batch below `singles`, absorbed alone from there.
    auto take = [&](uint position, thread const uint (&k)[key_lane::words], float2 kc,
                    thread const uint (&v)[value_lane::words], float2 vc) {
        if (position >= singles) {
            uint k1[1][key_lane::words];
            uint v1[1][value_lane::words];
            float2 kc1[1] = {kc};
            float2 vc1[1] = {vc};
            ATTENTION_UNROLL
            for (uint w = 0; w < key_lane::words; ++w)
                k1[0][w] = k[w];
            ATTENTION_UNROLL
            for (uint w = 0; w < value_lane::words; ++w)
                v1[0][w] = v[w];
            attention::absorb_affine<1>(q, qsum, k1, kc1, v1, vc1, maximum, denominator, output, bias, lane);
            return;
        }
        ATTENTION_UNROLL
        for (uint j = 0; j < DECODE_BATCH; ++j) {
            if (j == pending) {
                ATTENTION_UNROLL
                for (uint w = 0; w < key_lane::words; ++w)
                    pk[j][w] = k[w];
                ATTENTION_UNROLL
                for (uint w = 0; w < value_lane::words; ++w)
                    pv[j][w] = v[w];
                pkc[j] = kc;
                pvc[j] = vc;
            }
        }
        if (++pending == DECODE_BATCH) {
            attention::absorb_affine<DECODE_BATCH>(q, qsum, pk, pkc, pv, pvc, maximum, denominator, output, bias,
                lane);
            pending = 0;
        }
    };
    const attention::span_start start = attention::form_locate(visible, fresh, row, R, first, lane);
    uint offset = start.offset;
    uint position = first;
    for (ulong span = start.span; span <= R && offset < last; ++span) {
        int lo, hi;
        attention::form_span(visible, fresh, row, R, span, lo, hi);
        const uint length = uint(metal::max(hi - lo, 0));
        const uint end = metal::min(last, offset + length);
        if (span < R) {
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
                // The part's keys that complete a pending batch.
                for (; pending != 0 && position < part_end; ++position, ++t) {
                    uint k[key_lane::words];
                    uint v[value_lane::words];
                    key_lane::load(key_rows + t * key_step, lane, k);
                    value_lane::load(value_rows + t * value_step, lane, v);
                    take(position, k, float2(key_pairs[t * key_pair_step]), v,
                        float2(value_pairs[t * value_pair_step]));
                }
                // Its whole batches.
                const uint batches_end = metal::min(part_end, singles);
                for (; position + DECODE_BATCH <= batches_end; position += DECODE_BATCH, t += DECODE_BATCH) {
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
                        output, bias, lane);
                }
                // The keys an edge cuts from their batch, and the last keys.
                for (; position < part_end; ++position, ++t) {
                    uint k[key_lane::words];
                    uint v[value_lane::words];
                    key_lane::load(key_rows + t * key_step, lane, k);
                    value_lane::load(value_rows + t * value_step, lane, v);
                    take(position, k, float2(key_pairs[t * key_pair_step]), v,
                        float2(value_pairs[t * value_pair_step]));
                }
            }
        } else {
            for (; position < end; ++position) {
                const ulong token = ulong(lo) + (position - offset);
                const attention::lane_encoding<ATTENTION_KEY_BITS> key_encoding = fresh_key(token);
                const attention::lane_encoding<ATTENTION_VALUE_BITS> value_encoding = fresh_value(token);
                take(position, key_encoding.codes, float2(key_encoding.pair), value_encoding.codes,
                    float2(value_encoding.pair));
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
#endif
}

// L2: threadgroup (query head, row, 32-column group), one thread per column:
// the fixed-order merge of the row's partitions (with the head's sink when
// SK = 1) and the output gate. With the output rotary (RO = 1), one
// threadgroup holds the head's whole row, which it turns back in F32 before
// publishing it.
kernel void attention_decode_k8v4_merge(
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
    attention::decode_output<SEISMIC_TUNE_SPAN, SEISMIC_TUNE_PARTS, SEISMIC_TUNE_TOKENS, 0, ROTATED>(query, gate,
        visible, fresh, result, partials, statistics, weights, SEISMIC_DIM_R, SEISMIC_DIM_M, group.x, group.y,
        ROTATED ? thread_index : group.z * 32 + lane, lane, SEISMIC_PARAM_GATE_FUNCTION != 0, sinks,
        thread_index < 32, row_values, coordinates, rotary_components, rotary_frequencies, rotary_amplitudes);
}
