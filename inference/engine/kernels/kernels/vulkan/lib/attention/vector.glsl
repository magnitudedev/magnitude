// The vector form of the dense attention entries: `attention_decode`'s and
// `attention_prefill`'s MATRIX = 0 (the Vulkan form of
// `metal/attention_decode.metal`'s vector body). Every operand is read in the
// activation element A and every product is F32: nothing is staged in a
// narrower element, so this form is the prefill entry's canonical default
// (precision.md, "Absorbed latent attention").
//
// Workgroup (kv head and head group, partition, row). A workgroup holds
// ATTENTION_HEAD_GROUP of its kv head's G query heads, group `group` of them
// (the entry defines it before `attention.glsl`: the decode entry's GROUPS
// groups, the prefill entry's SLICES). A row's key sequence (its visible
// spans, its fresh span, then its `picks` index-list picks) splits into equal
// partitions of attention_partition_span keys. The group splits into slices
// of ATTENTION_HEADS heads; each subgroup scans a contiguous sub-range (its
// key group) of its partition for the heads of its slice (`attention_slice`),
// so each key/value row is loaded once per slice. Scores are F32 in the exp2
// domain (scale * log2e folded into the query). The subgroup states merge in
// fixed order into one partial per (row, query head, partition): the
// unnormalized output relative to the partition maximum, and (maximum,
// denominator). Partition 0 of the first head group of a layer with fresh
// rows also appends the row's key and value (its key alone for stored-key
// values, VK) at its destination.
//
// A key group's keys absorb in batches of VECTOR_BATCH at fixed positions
// first + i * VECTOR_BATCH of its row's sequence, then one at a time: a key's
// batch, and so the row's reduction order, depends on its position alone,
// never on which keys are still fresh (how many rows share the launch) or
// where its spans, slabs and picks fall; and a fresh key and value are
// attended exactly as they are appended. So under one specialization a row's
// result is bit for bit the same whatever rows share its launch.
//
// Shared words: the subgroup states [SIMDS][ATTENTION_HEADS][2] (floats),
// then the group's prepared queries [ceil(HEAD_GROUP / 2)][W] (two
// activation-element heads per word), which the merge columns [SIMDS][W]
// (floats) alias: every subgroup has read its queries before the
// publication's first barrier.
#include <seismic/slab.glsl>

// The prepared queries pack two activation elements per shared word.
#if ELEMENT_KIND(SEISMIC_ELEMENT_A) != ELEMENT_BF16 && ELEMENT_KIND(SEISMIC_ELEMENT_A) != ELEMENT_F16
#error "the vector attention form requires a bf16 or f16 activation"
#endif

// Keys each subgroup loads before scoring them, so their loads are in flight
// together (fewer for wide heads, whose key and value columns fill the
// registers next to the queries and outputs).
#define VECTOR_BATCH (ATTENTION_E >= 16u ? 2u : 4u)

// A WorkList prefill (the entry's PREFILL_SELECTED) attends a key of a row
// only where the row's block bitmap has its block's bit (`prefill_reads`, the
// matrix form's reading of the same bitmap); the key's ordinal is its
// position in the row's visible order, spans then fresh. Every other entry
// reads every key, and the bitmap code is not compiled. The decode entry's
// vector form includes this file without `prefill.glsl`, so it defines no
// PREFILL_SELECTED and has no `prefill_reads`.
//
// `vector_next` is the next position a key group can have a read at: the
// next set bit of the bitmap among the `visible` keys (`prefill_next_read`),
// or `position` itself where a key is read there, in the picks past them,
// and for the entries that read every key. A batch holding no read leaves the
// state as it is (`attention_absorb`), so the walk skips to the batch of the
// next read and its cost is the selection's, not the context's.
#ifdef PREFILL_SELECTED
#define VECTOR_SELECTED PREFILL_SELECTED
bool vector_reads(uint64_t row, uint ordinal) { return prefill_reads(row, 0, int(ordinal), 0); }
uint vector_next(uint64_t row, uint position, uint visible) { return prefill_next_read(row, position, visible); }
#else
#define VECTOR_SELECTED false
bool vector_reads(uint64_t row, uint ordinal) { return true; }
uint vector_next(uint64_t row, uint position, uint visible) { return position; }
#endif

// Key `position` of `row`'s sequence (at or past the cursor's,
// `attention_cursor`) for `kv_head`, lane `lane`'s E columns, into `k` and
// `v`: a history row of `h` from the cursor's slab, whose kv head's key and
// value rows start at `keys` and `values` (resolved again only when the
// cursor's slab moves), a fresh row exactly as it is appended
// (`history_fresh_row`), or one of the row's `picks` from the selected plane
// read as key and value. A stored-key value is the key. Returns whether the
// key is attended: a pick of -1 is not, nor a visible key a WorkList prefill
// does not select, nor any key where the read is not `live` (their columns
// are zero; a cursor not live stays where it is). The whole hardware
// subgroup calls it. On 32 lanes the subgroup leaves at the key's kind; on
// 64 lanes its logical subgroups may read keys of different kinds, so it
// takes every path, and a fresh row is prepared where either reads one.
bool vector_read(attention_history h, inout attention_cursor c, inout Stored keys, inout Stored values,
    uint64_t row, uint kv_head, uint picks, uint position, const bool live, out float k[ATTENTION_E],
    out float v[ATTENTION_E]) {
    bool moved;
    const uint64_t index = attention_cursor_seek(c, SEISMIC_PTR(SEISMIC_BUFFER_VISIBLE),
        SEISMIC_PTR(SEISMIC_BUFFER_FRESH), row, SEISMIC_DIM_R, picks, uint64_t(SEISMIC_PARAM_SLAB_ROWS), position,
        live, moved);
    const uint lane = SEISMIC_LANE;
    const uint64_t row_elements = uint64_t(ATTENTION_KV) * ATTENTION_W;
    // The slab's rows follow the cursor whether or not this key is read.
    if (moved) {
        keys = out_stored(slab_out_region(h.key, c.slab), uint64_t(kv_head) * ATTENTION_W);
        values = out_stored(slab_out_region(h.value, c.slab), uint64_t(kv_head) * ATTENTION_W);
    }
#if SEISMIC_SUBGROUP_LANES == 32
    if (VECTOR_SELECTED && c.span <= SEISMIC_DIM_R && !vector_reads(row, position)) {
        [[unroll]] for (uint i = 0u; i < ATTENTION_E; ++i) {
            k[i] = 0.0;
            v[i] = 0.0;
        }
        return false;
    }
    if (c.span < SEISMIC_DIM_R) {
        const uint64_t at = (index - c.slab_lo) * row_elements + uint64_t(lane) * ATTENTION_E;
        attention_load_row(keys, at, k);
        if (ATTENTION_STORED_KEY)
            v = k;
        else
            attention_load_row(values, at, v);
        return true;
    }
    if (c.span == SEISMIC_DIM_R) {
        history_fresh_row(true, index, kv_head, lane, k);
        if (ATTENTION_STORED_KEY)
            v = k;
        else
            history_fresh_row(false, index, kv_head, lane, v);
        return true;
    }
    // The picks path exists only where there are picks: without them the
    // selected plane is absent (address 0), and its read must not be one the
    // module can reach.
    int pick = -1;
    if (picks != 0u)
        pick = element_i32_at(SEISMIC_PTR(SEISMIC_BUFFER_SELECTED_ROWS) + (row * picks + index) * 4ul);
    if (pick < 0) {
        [[unroll]] for (uint i = 0u; i < ATTENTION_E; ++i) {
            k[i] = 0.0;
            v[i] = 0.0;
        }
        return false;
    }
    const uint64_t head_bytes = uint64_t(ATTENTION_W) * uint64_t(ELEMENT_BYTES(ACTIVATION_A));
    attention_load_row(history_address(SEISMIC_PTR(SEISMIC_BUFFER_SELECTED), uint64_t(pick),
        uint64_t(SEISMIC_PARAM_SELECTED_SLAB_ROWS), uint64_t(ATTENTION_KV) * head_bytes, head_bytes, kv_head),
        uint64_t(lane) * ATTENTION_E, k);
    v = k;
    return true;
#else
    [[unroll]] for (uint i = 0u; i < ATTENTION_E; ++i) {
        k[i] = 0.0;
        v[i] = 0.0;
    }
    bool read = live;
    if (VECTOR_SELECTED && read && c.span <= SEISMIC_DIM_R && !vector_reads(row, position))
        read = false;
    // The fresh row first: a logical subgroup reading no fresh key prepares
    // its own batch row (which exists) and zeroes it again.
    bool fresh_key;
    if (seismic_steer(fresh_key, read && c.span == SEISMIC_DIM_R)) {
        const uint64_t token = fresh_key ? index : row;
        history_fresh_row(true, token, kv_head, lane, k);
        if (ATTENTION_STORED_KEY)
            v = k;
        else
            history_fresh_row(false, token, kv_head, lane, v);
        if (!fresh_key)
            [[unroll]] for (uint i = 0u; i < ATTENTION_E; ++i) {
                k[i] = 0.0;
                v[i] = 0.0;
            }
    }
    if (read && c.span < SEISMIC_DIM_R) {
        const uint64_t at = (index - c.slab_lo) * row_elements + uint64_t(lane) * ATTENTION_E;
        attention_load_row(keys, at, k);
        if (ATTENTION_STORED_KEY)
            v = k;
        else
            attention_load_row(values, at, v);
    }
    // The picks path exists only where there are picks (above).
    if (picks != 0u && read && c.span > SEISMIC_DIM_R) {
        const int pick = element_i32_at(SEISMIC_PTR(SEISMIC_BUFFER_SELECTED_ROWS) + (row * picks + index) * 4ul);
        if (pick < 0) {
            read = false;
        } else {
            const uint64_t head_bytes = uint64_t(ATTENTION_W) * uint64_t(ELEMENT_BYTES(ACTIVATION_A));
            attention_load_row(history_address(SEISMIC_PTR(SEISMIC_BUFFER_SELECTED), uint64_t(pick),
                uint64_t(SEISMIC_PARAM_SELECTED_SLAB_ROWS), uint64_t(ATTENTION_KV) * head_bytes, head_bytes, kv_head),
                uint64_t(lane) * ATTENTION_E, k);
            v = k;
        }
    }
    return read;
#endif
}

// Where a key group's walk still takes batches (`batches`), moves `position`
// to the batch holding its next read, at a fixed position first + i *
// VECTOR_BATCH, and returns whether that batch ends by `last`; elsewhere
// `position` stays and it returns false. The whole hardware subgroup calls
// it (`vector_next`).
bool vector_batch(uint64_t row, uint first, uint last, uint visible, const bool batches, inout uint position) {
    const uint next = vector_next(row, position, batches ? visible : 0u);
    position = batches ? first + (next - first) / VECTOR_BATCH * VECTOR_BATCH : position;
    return batches && position + VECTOR_BATCH <= last;
}

// The partial of workgroup (`kv_head`, head group `group`, partition `part`,
// `row`) over history `h`, its rows' key sequences split by `span` and
// `parts` (the entry's SPAN and PARTS).
void vector_partial(attention_history h, uint kv_head, uint group, uint part, uint64_t row, uint picks, uint span,
    uint parts) {
    const uint64_t query_norm = SEISMIC_PTR(SEISMIC_BUFFER_QUERY_NORM);
    const uint64_t components = SEISMIC_PTR(SEISMIC_BUFFER_ROTARY_COMPONENTS);
    const uint64_t frequencies = SEISMIC_PTR(SEISMIC_BUFFER_ROTARY_FREQUENCIES);
    const uint64_t amplitudes = ATTENTION_AMPLITUDE_BUFFER;
    const uint64_t visible = SEISMIC_PTR(SEISMIC_BUFFER_VISIBLE);
    const uint64_t fresh = SEISMIC_PTR(SEISMIC_BUFFER_FRESH);
    const uint W = ATTENTION_W, E = ATTENTION_E, KV = ATTENTION_KV, G = ATTENTION_G;
    const uint GROUP = ATTENTION_HEAD_GROUP;
    const uint simds = SEISMIC_SUBGROUPS, sg = SEISMIC_SUBGROUP, lane = SEISMIC_LANE;
    const uint64_t spans = SEISMIC_DIM_R;
    const uint head_base = group * GROUP;
    const float epsilon = element_word_f32(SEISMIC_PARAM_EPSILON);
    const float scale = element_word_f32(SEISMIC_PARAM_SCALE);
    const uint64_t row_coordinates = SEISMIC_PTR(SEISMIC_BUFFER_COORDINATES) + row * 16ul;
    const uint64_t act_bytes = uint64_t(ELEMENT_BYTES(ACTIVATION_A));

    // Subgroup 0 appends the key, subgroup 1 the value (none for a
    // stored-key history); a hardware subgroup holding either runs the
    // preparation in both its logical subgroups, the other storing nothing.
    bool appending;
    if (seismic_steer(appending,
            ATTENTION_FRESH && part == 0u && group == 0u && sg < (ATTENTION_STORED_KEY ? 1u : 2u))) {
        const int destination = element_i32_at(SEISMIC_PTR(SEISMIC_BUFFER_DESTINATIONS) + row * 4ul);
        bool appends;
        if (seismic_steer(appends, appending && destination >= 0)) {
            float x[ATTENTION_E];
            history_fresh_row(sg == 0u, row, kv_head, lane, x);
            history_hold(h, sg == 0u, appends ? destination : -1, kv_head, lane, false, x);
        }
    }

    const uint total = attention_visible_total(visible, fresh, row, spans) + picks;
    const uint span_keys = attention_partition_span(total, span, parts);
    const uint partition_lo = part * span_keys;
    if (partition_lo >= total)
        return;
    const uint partition_hi = min(partition_lo + span_keys, total);

    const uint states = 0u;
    const uint columns = states + simds * ATTENTION_HEADS * 2u;
    const uint prepared = columns;
    const float query_scale = scale * ATTENTION_LOG2E;
    // Head pair p's prepared queries share words: head 2p in the low half,
    // head 2p + 1 in the high half.
    // The hardware subgroup runs while either of its logical subgroups holds
    // a pair; one past the pairs prepares pair 0 and stores nothing.
    bool held;
    for (uint pair = sg; seismic_steer(held, pair < (GROUP + 1u) / 2u); pair += simds) {
        const uint64_t head = row * KV * G + kv_head * G + head_base + 2u * (held ? pair : 0u);
        float low[ATTENTION_E];
        float high[ATTENTION_E];
        attention_prepare(ATTENTION_QUERY_BUFFER + head * ATTENTION_QUERY_STRIDE * act_bytes, ATTENTION_QUERY_NORM,
            query_norm, row_coordinates, components, frequencies, amplitudes, epsilon, lane, rounding_none(), low);
        bool paired;
        if (seismic_steer(paired, held && 2u * pair + 1u < GROUP)) {
            attention_prepare(ATTENTION_QUERY_BUFFER + (paired ? head + 1ul : head) * ATTENTION_QUERY_STRIDE * act_bytes,
                ATTENTION_QUERY_NORM, query_norm, row_coordinates, components, frequencies, amplitudes, epsilon,
                lane, rounding_none(), high);
            if (!paired)
                [[unroll]] for (uint i = 0u; i < ATTENTION_E; ++i)
                    high[i] = 0.0;
        } else {
            [[unroll]] for (uint i = 0u; i < ATTENTION_E; ++i)
                high[i] = 0.0;
        }
        if (held)
            [[unroll]] for (uint i = 0u; i < ATTENTION_E; ++i)
                seismic_shared_u32[prepared + pair * W + lane * E + i] =
                    element_store(ACTIVATION_A, low[i]) | (element_store(ACTIVATION_A, high[i]) << 16u);
    }
    barrier();
    const uint head0 = attention_slice(sg) * ATTENTION_HEADS;
    float q[ATTENTION_HEADS][ATTENTION_E];
    float maximum[ATTENTION_HEADS];
    float denominator[ATTENTION_HEADS];
    float result[ATTENTION_HEADS][ATTENTION_E];
    [[unroll]] for (uint g = 0u; g < ATTENTION_HEADS; ++g) {
        const uint head = head0 + g;
        [[unroll]] for (uint i = 0u; i < ATTENTION_E; ++i) {
            const uint word = seismic_shared_u32[prepared + (head / 2u) * W + lane * E + i];
            q[g][i] = element_load(ACTIVATION_A, (word >> (16u * (head % 2u))) & 0xffffu) * query_scale;
            result[g][i] = 0.0;
        }
        maximum[g] = -ATTENTION_INF;
        denominator[g] = 0.0;
    }

    const uvec2 range = attention_key_range(partition_lo, partition_hi, attention_key_group(sg));
    const uint first = range.x, last = range.y;
    attention_cursor c = attention_cursor_at(visible, fresh, row, spans, picks, first);
    // The cursor's slab's key and value rows.
    Stored keys = stored_at_buffer(ACTIVATION_A, 0ul), values = keys;
    // The row's visible keys, which the bitmap selects among; its picks follow.
    const uint visible_keys = total - picks;
    // The batches at fixed positions first + i * VECTOR_BATCH, from the one
    // holding the next read; then the keys that fill no batch, one at a
    // time. The hardware subgroup walks while any of its logical subgroups
    // has a key left; one that has none reads nothing and absorbs nothing.
    uint position = first;
    // `batching` is also the last round's: a key group out of batches stays.
    bool batching = true;
    while (seismic_steer(batching, vector_batch(row, first, last, visible_keys, batching, position))) {
        float k[4][ATTENTION_E];
        float v[4][ATTENTION_E];
        uint live = 0u;
        [[unroll]] for (uint j = 0u; j < VECTOR_BATCH; ++j)
            live |= uint(vector_read(h, c, keys, values, row, kv_head, picks, position + j, batching, k[j], v[j])) << j;
        attention_absorb(VECTOR_BATCH, live, q, k, v, maximum, denominator, result);
        position += batching ? VECTOR_BATCH : 0u;
    }
    // A key group past its keys stays (`vector_next` of no visible keys).
    bool single;
    for (position = vector_next(row, position, visible_keys); seismic_steer(single, position < last);
         position = vector_next(row, single ? position + 1u : position, single ? visible_keys : 0u)) {
        float k[4][ATTENTION_E];
        float v[4][ATTENTION_E];
        const uint live = uint(vector_read(h, c, keys, values, row, kv_head, picks, position, single, k[0], v[0]));
        attention_absorb(1u, live, q, k, v, maximum, denominator, result);
    }

    attention_publish(maximum, denominator, result, states, columns, SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS),
        SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STATISTICS), ((row * KV + kv_head) * G + head_base) * parts + part, parts);
}
