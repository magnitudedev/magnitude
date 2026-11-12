// Shared pieces of the attention family entries (`attention_decode`,
// `attention_prefill` and their K8/V4 forms): the per-head preparation
// (optional RMS norm and amplitude-scaled partial M-RoPE) held in one
// subgroup's registers, a row's span total and key location (subgroup
// cooperative), decode partition bounds, the online-softmax absorb, the
// fixed-order merge of partial states, the decode partition publication and
// gated merge, and the K/V append. The counterpart of
// `metal/lib/attention/attention.h`.
//
// Every dense operand is bound canonically (row-major, unit innermost
// stride), so rows are addressed from their logical offsets. Activations are
// the entry's element A (ELEMENT_ACT); the head width W = 2P + S is a
// multiple of 32 and each of a subgroup's 32 lanes owns E = W / 32
// contiguous columns.
#define ATTENTION_W uint(2ul * SEISMIC_DIM_P + SEISMIC_DIM_S)
#define ATTENTION_E (ATTENTION_W / 32u)
#define ATTENTION_P uint(SEISMIC_DIM_P)
// A head row held by a subgroup is the lane row of the core head, rotary and
// rounding libraries.
#define LANE_E ATTENTION_E
#include "../core/activation.glsl"
#include "../core/reduce.glsl"
#include "../core/head.glsl"
#include "../core/rotary.glsl"
#include "../core/rounding.glsl"

// The rotated pairs: P of them over the leading columns, or the trailing ones
// (RT = 1, from S); half-split, or adjacent (RA = 1).
#define ATTENTION_ROTARY rotary_layout(ATTENTION_P, SEISMIC_DIM_RT != 0 ? uint(SEISMIC_DIM_S) : 0u, SEISMIC_DIM_RA != 0)
#define ATTENTION_KV uint(SEISMIC_DIM_KV)
#define ATTENTION_G uint(ATTENTION_QUERY_GROUP)
// Query heads one decode subgroup holds in registers. The decode entries split
// the G heads of a kv head into ATTENTION_G / ATTENTION_HEADS slices (their
// SLICES parameter) and define it before including this file; a subgroup
// holding all G heads of a large group (G = 16, E = 8: 256 query and output
// floats per lane) no longer fits in registers.
#ifndef ATTENTION_HEADS
#define ATTENTION_HEADS ATTENTION_G
#endif
// Query heads one decode workgroup holds: a group of the kv head's G (the
// entry's head-group split, its GROUPS parameter), or all G.
#ifndef ATTENTION_HEAD_GROUP
#define ATTENTION_HEAD_GROUP ATTENTION_G
#endif
#define ATTENTION_LOG2E 1.4426950408889634

#define ATTENTION_INF uintBitsToFloat(0x7f800000u)

// The entry's form (contract in attention.seismic), defined by every entry
// before including this file: ATTENTION_I interleaved gate columns after each
// query head's W columns (0 or W), ATTENTION_U separate gate values per query
// head (0, 1 or W), ATTENTION_FRESH (the layer has fresh rows),
// ATTENTION_QUERY_NORM (query RMS norm, NQ), ATTENTION_NORM (key RMS norm,
// N), ATTENTION_VALUE_NORM (value RMS norm), ATTENTION_STORED_KEY (VK: a
// row's value is its prepared key, and history holds the key plane alone),
// ATTENTION_KEY_ROUNDING (the trained rounding of a prepared key, a
// `rounding_t`: the entry's KR scalars, or `rounding_none()`) and
// ATTENTION_SOFTPLUS (the gate function is softplus rather than sigmoid).
#define ATTENTION_QUERY_STRIDE (ATTENTION_W + ATTENTION_I)

// Attention sinks (`sinks [SK, KV * G]`): an entry with the input defines
// ATTENTION_SINKS (its SK = 1) and ATTENTION_SINK_BUFFER (its address) before
// including this file; without them every sink site compiles away.
#ifndef ATTENTION_SINKS
#define ATTENTION_SINKS false
#define ATTENTION_SINK_BUFFER 0ul
#endif

// E consecutive A elements from element `index` (aligned to E) as F32.
void attention_load_row(uint64_t base, uint64_t index, out float x[ATTENTION_E]) {
    if (ELEMENT_ACT != ELEMENT_F32 && ATTENTION_E % 8u == 0u) {
        [[unroll]] for (uint c = 0u; c < ATTENTION_E; c += 8u) {
            const uvec4 words = element_uvec4_at(base + (index + c) * 2ul);
            [[unroll]] for (uint j = 0u; j < 4u; ++j) {
                const vec2 pair = element_unpack2(ELEMENT_ACT, words[j]);
                x[c + 2u * j] = pair.x;
                x[c + 2u * j + 1u] = pair.y;
            }
        }
    } else {
        [[unroll]] for (uint i = 0u; i < ATTENTION_E; ++i)
            x[i] = element_at(ELEMENT_ACT, base, index + i);
    }
}

// One value head row at `raw` (A elements) into lane `lane`'s E columns,
// RMS-normalized with `norm` and rounded to A when ATTENTION_VALUE_NORM.
void attention_value(uint64_t raw, uint64_t norm, float epsilon, uint lane, out float x[ATTENTION_E]) {
    attention_load_row(raw, uint64_t(lane) * ATTENTION_E, x);
    if (ATTENTION_VALUE_NORM) {
        head_rms_norm(x, norm, 1ul, epsilon, lane);
        [[unroll]] for (uint i = 0u; i < ATTENTION_E; ++i)
            x[i] = element_round(ELEMENT_ACT, x[i]);
    }
}

// One head row at `raw` (A elements) prepared as the portable
// `rounded_head_rotary` does, into lane `lane`'s E columns: RMS-normalized
// with `norm` (F32) when `normed`, its pairs rotated (`ATTENTION_ROTARY`;
// pair p by coordinate axis components[p] at frequencies[p], cosine and sine
// scaled by amplitudes[p]), held in A after each where `trained` rounds the
// row, then rounded by it. The caller rounds the result to A. The whole
// subgroup calls it.
void attention_prepare(uint64_t raw, const bool normed, uint64_t norm, uint64_t coordinates, uint64_t components,
    uint64_t frequencies, uint64_t amplitudes, float epsilon, uint lane, rounding_t trained, out float x[ATTENTION_E]) {
    attention_load_row(raw, uint64_t(lane) * ATTENTION_E, x);
    if (normed)
        head_rms_norm(x, norm, 1ul, epsilon, lane);
    rounding_hold_lanes(x, trained);
    const rotary_turns turns =
        rotary_lane_turns(coordinates, 1ul, components, 1ul, frequencies, 1ul, amplitudes, 1ul, ATTENTION_ROTARY, lane);
    rotary_lane_rotate(x, turns, ATTENTION_ROTARY, lane);
    rounding_hold_lanes(x, trained);
    rounding_lanes(x, trained);
}

// Appends lane `lane`'s columns of one kv head row at history row
// `destination` (the caller skips rows without a destination), rounded to A.
void attention_append(uint64_t history, int destination, uint kv_head, uint lane, float x[ATTENTION_E]) {
    const uint64_t target = (uint64_t(destination) * ATTENTION_KV + kv_head) * ATTENTION_W + lane * ATTENTION_E;
    if (ELEMENT_ACT != ELEMENT_F32 && ATTENTION_E % 8u == 0u) {
        [[unroll]] for (uint c = 0u; c < ATTENTION_E; c += 8u) {
            uvec4 words;
            [[unroll]] for (uint j = 0u; j < 4u; ++j)
                words[j] = element_pack2(ELEMENT_ACT, x[c + 2u * j], x[c + 2u * j + 1u]);
            element_uvec4_put(history + (target + c) * 2ul, words);
        }
    } else {
        [[unroll]] for (uint i = 0u; i < ATTENTION_E; ++i)
            element_put(ELEMENT_ACT, history, target + i, x[i]);
    }
}

// Span `span` of `row`'s keys: visible history spans 0..R-1, then the fresh
// span R (rows of this batch; empty for a layer without fresh rows).
void attention_span(uint64_t visible, uint64_t fresh, uint64_t row, uint64_t spans, uint64_t span, out int lo, out int hi) {
    if (span == spans && !ATTENTION_FRESH) {
        lo = 0;
        hi = 0;
        return;
    }
    const uint64_t at = span < spans ? visible + ((row * spans + span) * 2ul) * 4ul : fresh + row * 8ul;
    lo = element_i32_at(at);
    hi = element_i32_at(at + 4ul);
}

// The number of keys of span `index` of `row` (none past the fresh span).
uint attention_span_length(uint64_t visible, uint64_t fresh, uint64_t row, uint64_t spans, uint64_t index) {
    if (index > spans)
        return 0u;
    int lo, hi;
    attention_span(visible, fresh, row, spans, index, lo, hi);
    return uint(max(hi - lo, 0));
}

// The total number of keys a row sees: its visible spans, then its fresh span.
// Lane l sums spans l, l + 32, ...; the whole subgroup calls it.
uint attention_visible_total(uint64_t visible, uint64_t fresh, uint64_t row, uint64_t spans) {
    uint total = 0u;
    for (uint64_t index = SEISMIC_LANE; index <= spans; index += 32ul)
        total += attention_span_length(visible, fresh, row, spans, index);
    return seismic_subgroup_add(total);
}

// The span of `row` holding its key `key` and the span's first key: the first
// span s in 0..=spans whose keys end past `key`, or (spans + 1, total) when
// the row has no such key. Spans before it hold no key at or after `key`. The
// whole subgroup calls it, 32 spans at a time.
uvec2 attention_locate(uint64_t visible, uint64_t fresh, uint64_t row, uint64_t spans, uint key) {
    uint carried = 0u;
    for (uint64_t base = 0ul; base <= spans; base += 32ul) {
        uint found;
        if (reduce_lanes_locate(attention_span_length(visible, fresh, row, spans, base + SEISMIC_LANE), key, carried,
                found))
            return uvec2(uint(base) + found, carried);
    }
    return uvec2(uint(spans) + 1u, carried);
}

// A key group's cursor over its row's key sequence: its `spans` visible
// spans, its fresh span (span `spans`), then its `picks` picks (span
// `spans` + 1; a decode entry's KS). It is on span `span`, `count` keys from
// row `lo` at key offset `offset`, and holds the slab `slab` of the last
// history row it reached, rows [slab_lo, slab_hi) (none before one). The
// vector forms read a key group's keys through it in position order, so
// they take them in batches at fixed positions of the row's sequence,
// whichever of its keys are still fresh and wherever its spans, slabs and
// picks fall.
struct attention_cursor {
    uint64_t span;
    uint offset;
    uint count;
    int lo;
    uint64_t slab;
    uint64_t slab_lo;
    uint64_t slab_hi;
};

// Loads the cursor's span of `row`.
void attention_cursor_span(inout attention_cursor c, uint64_t visible, uint64_t fresh, uint64_t row,
    uint64_t spans, uint picks) {
    if (c.span > spans) {
        c.lo = 0;
        c.count = picks;
        return;
    }
    int lo, hi;
    attention_span(visible, fresh, row, spans, c.span, lo, hi);
    c.lo = lo;
    c.count = uint(max(hi - lo, 0));
}

// The cursor on key `position` of `row`'s sequence (`attention_locate`;
// past the visible keys, on the picks). The whole subgroup calls it.
attention_cursor attention_cursor_at(uint64_t visible, uint64_t fresh, uint64_t row, uint64_t spans, uint picks,
    uint position) {
    const uvec2 located = attention_locate(visible, fresh, row, spans, position);
    attention_cursor c = attention_cursor(uint64_t(located.x), located.y, 0u, 0, 0ul, 1ul, 0ul);
    attention_cursor_span(c, visible, fresh, row, spans, picks);
    return c;
}

// Moves the cursor to key `position` (at or past its own) and returns the
// key's index in its span: a history or fresh row, or the ordinal of a
// pick. On a history row (span < spans) the cursor holds the row's slab of
// `slab_rows` rows; `moved` is whether that changed it.
uint64_t attention_cursor_seek(inout attention_cursor c, uint64_t visible, uint64_t fresh, uint64_t row,
    uint64_t spans, uint picks, uint64_t slab_rows, uint position, out bool moved) {
    while (position >= c.offset + c.count) {
        c.offset += c.count;
        ++c.span;
        attention_cursor_span(c, visible, fresh, row, spans, picks);
    }
    const uint64_t index = uint64_t(c.lo) + uint64_t(position - c.offset);
    moved = c.span < spans && (index < c.slab_lo || index >= c.slab_hi);
    if (moved) {
        c.slab = index / slab_rows;
        c.slab_lo = c.slab * slab_rows;
        c.slab_hi = c.slab_lo + slab_rows;
    }
    return index;
}

// Keys per decode partition for a row seeing `total` keys: at least `span`,
// and few enough that `parts` partitions cover the row. Never zero.
uint attention_partition_span(uint total, uint span, uint parts) {
    return max(span, (total + parts - 1u) / parts);
}

// The online-softmax state of a subgroup's ATTENTION_HEADS query heads over
// its keys, in the exp2 domain, absorbing `n` (1 to 4) keys at a time. Key j
// is attended where bit j of `live` is set; another (a pick of none) scores
// -inf, so it adds nothing and a batch of none leaves the state as it is.
void attention_absorb(const uint n, uint live, float q[ATTENTION_HEADS][ATTENTION_E], float k[4][ATTENTION_E],
    float v[4][ATTENTION_E], inout float maximum[ATTENTION_HEADS], inout float denominator[ATTENTION_HEADS],
    inout float result[ATTENTION_HEADS][ATTENTION_E]) {
    [[unroll]] for (uint g = 0u; g < ATTENTION_HEADS; ++g) {
        float score[4];
        [[unroll]] for (uint j = 0u; j < 4u; ++j) {
            if (j < n) {
                float partial = 0.0;
                [[unroll]] for (uint i = 0u; i < ATTENTION_E; ++i)
                    partial = seismic_fma_rn(q[g][i], k[j][i], partial);
                score[j] = ((live >> j) & 1u) != 0u ? seismic_subgroup_sum_f32(partial) : -ATTENTION_INF;
            }
        }
        float next = maximum[g];
        [[unroll]] for (uint j = 0u; j < 4u; ++j)
            if (j < n)
                next = max(next, score[j]);
        const float carry = next == maximum[g] ? 1.0 : exp2(maximum[g] - next);
        float probability[4];
        float sum = 0.0;
        [[unroll]] for (uint j = 0u; j < 4u; ++j) {
            if (j < n) {
                probability[j] = score[j] == -ATTENTION_INF ? 0.0 : exp2(score[j] - next);
                sum += probability[j];
            }
        }
        denominator[g] = seismic_fma_rn(denominator[g], carry, sum);
        [[unroll]] for (uint i = 0u; i < ATTENTION_E; ++i) {
            float o = result[g][i] * carry;
            [[unroll]] for (uint j = 0u; j < 4u; ++j)
                if (j < n)
                    o = seismic_fma_rn(probability[j], v[j][i], o);
            result[g][i] = o;
        }
        maximum[g] = next;
    }
}

// ---------------------------------------------------------------------------
// The key-parallel decode form (`attention_decode_k8v4`'s KEYWISE parameter;
// its history part is `history_keywise_absorb`): a subgroup absorbs up to
// ATTENTION_KEYS keys at a time. Lane t scores key t for all ATTENTION_HEADS
// heads of its slice over the whole width, reading the prepared queries from
// shared memory as broadcasts, so a score needs no subgroup reduction; the
// batch takes one subgroup maximum and sum per head. Its probabilities pass
// through the subgroup's shared weights [ATTENTION_KEYS][ATTENTION_HEADS]
// (floats) to the value product, which keeps `attention_absorb`'s
// lane-owns-columns layout, so the state is the vector form's.
//
// Keywise prepared queries: [G][W] F32 (rounded to A, times scale * log2e)
// from shared float `queries`, 16-byte aligned.
#define ATTENTION_KEYS 32u

// Query head `head`'s prepared columns [column, column + 4).
vec4 attention_keywise_query(uint queries, uint head, uint column) {
    return uintBitsToFloat(seismic_shared_uvec4[(queries + head * ATTENTION_W + column) / 4u]);
}

// Folds a batch of `n` keys into the online state from each lane's scores of
// its key (lanes at or past `n` hold none): per head one maximum over the
// batch; the outputs rescale, and the probabilities go to the subgroup's
// weights at shared float `weights`.
void attention_keywise_softmax(const uint n, float score[ATTENTION_HEADS], uint weights,
    inout float maximum[ATTENTION_HEADS], inout float denominator[ATTENTION_HEADS],
    inout float result[ATTENTION_HEADS][ATTENTION_E]) {
    const uint lane = SEISMIC_LANE;
    // The previous batch's weights are read.
    subgroupBarrier();
    [[unroll]] for (uint g = 0u; g < ATTENTION_HEADS; ++g) {
        const float next = max(maximum[g], seismic_subgroup_max_f32(lane < n ? score[g] : -ATTENTION_INF));
        // An empty batch leaves the state as it is, an empty state too.
        const float carry = next == maximum[g] ? 1.0 : exp2(maximum[g] - next);
        const float probability = lane < n ? exp2(score[g] - next) : 0.0;
        denominator[g] = seismic_fma_rn(denominator[g], carry, seismic_subgroup_sum_f32(probability));
        maximum[g] = next;
        [[unroll]] for (uint i = 0u; i < ATTENTION_E; ++i)
            result[g][i] *= carry;
        seismic_shared_f32[weights + lane * ATTENTION_HEADS + g] = probability;
    }
    subgroupMemoryBarrierShared();
    subgroupBarrier();
}

// Accumulates key `key` of the batch (this lane's E value columns `v`) with
// its weights into the outputs.
void attention_keywise_accumulate(uint weights, uint key, float v[ATTENTION_E],
    inout float result[ATTENTION_HEADS][ATTENTION_E]) {
    float p[ATTENTION_HEADS];
    if (ATTENTION_HEADS % 4u == 0u) {
        [[unroll]] for (uint g = 0u; g < ATTENTION_HEADS; g += 4u) {
            const vec4 four = uintBitsToFloat(seismic_shared_uvec4[(weights + key * ATTENTION_HEADS + g) / 4u]);
            [[unroll]] for (uint j = 0u; j < 4u; ++j)
                p[g + j] = four[j];
        }
    } else {
        [[unroll]] for (uint g = 0u; g < ATTENTION_HEADS; ++g)
            p[g] = seismic_shared_f32[weights + key * ATTENTION_HEADS + g];
    }
    [[unroll]] for (uint g = 0u; g < ATTENTION_HEADS; ++g)
        [[unroll]] for (uint i = 0u; i < ATTENTION_E; ++i)
            result[g][i] = seismic_fma_rn(p[g], v[i], result[g][i]);
}

// One attended column of query head `head` (a global query head, kv head *
// G + local): its unnormalized output `accumulated` over its softmax state's
// `denominator`, both relative to the state's `maximum` in the exp2 domain of
// the scores (scale * log2 e folded into the query). With sinks, the head's
// sink logit s (natural-log domain of the scaled scores) adds exp(s - m) =
// exp2(s log2 e - maximum) to the denominator; a head that read no key
// (maximum -inf) attends to zero.
float attention_attended(float accumulated, float maximum, float denominator, uint64_t head) {
    if (ATTENTION_SINKS) {
        if (maximum == -ATTENTION_INF)
            return 0.0;
        denominator += exp2(element_f32_at(ATTENTION_SINK_BUFFER + head * 4ul) * ATTENTION_LOG2E - maximum);
    }
    return seismic_div_rn(accumulated, max(denominator, 1e-30));
}

// The fixed-order merge of `count` partial attention states for one column:
// state p is slot first + p * stride, with its unnormalized output at
// partials[slot * W + column] (relative to its maximum) and (maximum,
// denominator) at statistics[slot * 2]. Empty states (denominator 0) are
// skipped. Returns (maximum, denominator, accumulated), the sums relative to
// the maximum (0 with no non-empty state).
vec3 attention_merge_state(uint64_t partials, uint64_t statistics, uint64_t first, uint64_t stride, uint count,
    uint column) {
    float maximum = -ATTENTION_INF;
    for (uint p = 0u; p < count; ++p) {
        const uint64_t slot = first + p * stride;
        if (element_f32_at(statistics + (slot * 2ul + 1ul) * 4ul) > 0.0)
            maximum = max(maximum, element_f32_at(statistics + slot * 8ul));
    }
    float denominator = 0.0;
    float accumulated = 0.0;
    for (uint p = 0u; p < count; ++p) {
        const uint64_t slot = first + p * stride;
        const float d = element_f32_at(statistics + (slot * 2ul + 1ul) * 4ul);
        if (d > 0.0) {
            const float weight = exp2(element_f32_at(statistics + slot * 8ul) - maximum);
            denominator = seismic_fma_rn(d, weight, denominator);
            accumulated = seismic_fma_rn(element_f32_at(partials + (slot * ATTENTION_W + column) * 4ul), weight, accumulated);
        }
    }
    return vec3(maximum, denominator, accumulated);
}

// `attention_merge_state` attended: a column with no state attends to zero;
// with sinks, query head `head`'s sink joins the denominator
// (`attention_attended`).
float attention_merge(uint64_t partials, uint64_t statistics, uint64_t first, uint64_t stride, uint count, uint column,
    uint64_t head) {
    const vec3 merged = attention_merge_state(partials, statistics, first, stride, count, column);
    return attention_attended(merged.z, merged.x, merged.y, head);
}

// Decode subgroups: subgroup sg holds the ATTENTION_HEADS query heads of
// slice sg % slices (heads slice * ATTENTION_HEADS ..) over key group
// sg / slices, one of subgroups / slices contiguous sub-ranges of the
// partition (`attention_key_range`). Every key is read by one subgroup per
// slice.
uint attention_slices() { return ATTENTION_HEAD_GROUP / ATTENTION_HEADS; }
uint attention_slice(uint sg) { return sg % attention_slices(); }
uint attention_key_group(uint sg) { return sg / attention_slices(); }
uint attention_key_groups() { return SEISMIC_SUBGROUPS / attention_slices(); }

// Key group `group`'s sub-range of the partition [lo, hi): its first and end
// key, of ceil((hi - lo) / groups) keys, the last shorter or empty.
uvec2 attention_key_range(uint lo, uint hi, uint group) {
    const uint sub = (hi - lo + attention_key_groups() - 1u) / attention_key_groups();
    const uint first = min(lo + group * sub, hi);
    return uvec2(first, min(first + sub, hi));
}

// The maximum over the key groups of one slice of its local head h's states.
float attention_slice_maximum(uint states, uint slice, uint h) {
    float maximum = -ATTENTION_INF;
    for (uint group = 0u; group < attention_key_groups(); ++group) {
        const uint at = states + ((group * attention_slices() + slice) * ATTENTION_HEADS + h) * 2u;
        if (seismic_shared_f32[at + 1u] > 0.0)
            maximum = max(maximum, seismic_shared_f32[at]);
    }
    return maximum;
}

// Publishes one decode partition: the states of each query head (outputs
// relative to their own maxima) merge in key-group order into one partial per
// query head g at slot first + g * parts: the unnormalized output at
// partials[slot * W] relative to the partition maximum, and (maximum,
// denominator) at statistics[slot * 2]. Shared floats: `states` holds
// [subgroups][ATTENTION_HEADS][2], `columns` [subgroups][W].
void attention_publish(float maximum[ATTENTION_HEADS], float denominator[ATTENTION_HEADS],
    float result[ATTENTION_HEADS][ATTENTION_E], uint states, uint columns, uint64_t partials, uint64_t statistics,
    uint64_t first, uint parts) {
    const uint sg = SEISMIC_SUBGROUP, lane = SEISMIC_LANE;
    const uint thread = gl_LocalInvocationIndex;
    const uint slices = attention_slices();
    if (lane == 0u) {
        [[unroll]] for (uint h = 0u; h < ATTENTION_HEADS; ++h) {
            seismic_shared_f32[states + (sg * ATTENTION_HEADS + h) * 2u] = maximum[h];
            seismic_shared_f32[states + (sg * ATTENTION_HEADS + h) * 2u + 1u] = denominator[h];
        }
    }
    barrier();
    [[unroll]] for (uint h = 0u; h < ATTENTION_HEADS; ++h) {
        const float partition_maximum = attention_slice_maximum(states, attention_slice(sg), h);
        const float weight = denominator[h] > 0.0 ? exp2(maximum[h] - partition_maximum) : 0.0;
        [[unroll]] for (uint i = 0u; i < ATTENTION_E; ++i)
            seismic_shared_f32[columns + sg * ATTENTION_W + lane * ATTENTION_E + i] = result[h][i] * weight;
        barrier();
        for (uint item = thread; item < slices * ATTENTION_W; item += gl_WorkGroupSize.x) {
            const uint slice = item / ATTENTION_W, column = item % ATTENTION_W;
            float sum = 0.0;
            for (uint group = 0u; group < attention_key_groups(); ++group)
                sum += seismic_shared_f32[columns + (group * slices + slice) * ATTENTION_W + column];
            const uint64_t slot = first + uint64_t(slice * ATTENTION_HEADS + h) * parts;
            element_f32_put(partials + (slot * ATTENTION_W + column) * 4ul, sum);
        }
        if (thread < slices) {
            const uint slice = thread;
            const float slice_maximum = attention_slice_maximum(states, slice, h);
            float total_denominator = 0.0;
            for (uint group = 0u; group < attention_key_groups(); ++group) {
                const uint at = states + ((group * slices + slice) * ATTENTION_HEADS + h) * 2u;
                const float d = seismic_shared_f32[at + 1u];
                if (d > 0.0)
                    total_denominator = seismic_fma_rn(d, exp2(seismic_shared_f32[at] - slice_maximum), total_denominator);
            }
            const uint64_t slot = first + uint64_t(slice * ATTENTION_HEADS + h) * parts;
            element_f32_put(statistics + slot * 8ul, slice_maximum);
            element_f32_put(statistics + slot * 8ul + 4ul, total_denominator);
        }
        barrier();
    }
}

// The store of one attended column of query head `head` of `row` times its
// output gate: value column % count of the head's ATTENTION_I interleaved
// gate columns (after its W query columns in `query`) or its ATTENTION_U
// separate ones (in `gate`); sigmoid, or softplus under ATTENTION_SOFTPLUS.
void attention_store_gated(uint64_t query, uint64_t gate, uint64_t result, uint64_t row, uint64_t head, uint column,
    float attended) {
    const uint64_t heads = uint64_t(ATTENTION_KV * ATTENTION_G);
    float value = attended;
    if (ATTENTION_I > 0u || ATTENTION_U > 0u) {
        const float g = ATTENTION_I > 0u
            ? element_at(ELEMENT_ACT, query, (row * heads + head) * ATTENTION_QUERY_STRIDE + ATTENTION_W + column)
            : element_at(ELEMENT_ACT, gate, (row * heads + head) * ATTENTION_U + column % max(ATTENTION_U, 1u));
        value = ATTENTION_SOFTPLUS ? attended * (max(g, 0.0) + log(1.0 + exp(-abs(g))))
                                   : seismic_div_rn(attended, 1.0 + exp(-g));
    }
    element_put(ELEMENT_ACT, result, (row * heads + head) * ATTENTION_W + column, value);
}

// The decode merge of one (query head, row) column over the row's key
// sequence, its `spans` visible spans then its `picks` picks: the row's
// non-empty partitions in partition order, then the output gate. A row that
// sees no key attends to zero. Whole subgroups call it (each finds the row's
// total keys).
void attention_decode_gate(uint64_t query, uint64_t gate, uint64_t visible, uint64_t fresh, uint64_t result,
    uint64_t partials, uint64_t statistics, uint64_t spans, uint picks, uint64_t head, uint64_t row, uint column,
    uint span, uint parts) {
    const uint total = attention_visible_total(visible, fresh, row, spans) + picks;
    const uint span_keys = attention_partition_span(total, span, parts);
    const uint used = (total + span_keys - 1u) / span_keys;
    const float attended = attention_merge(partials, statistics,
        (row * ATTENTION_KV * ATTENTION_G + head) * parts, 1ul, used, column, head);
    attention_store_gated(query, gate, result, row, head, column, attended);
}

// The output rotary (RO = 1), after every query head's output is stored in
// `result` ([rows][KV * G][W] activations): workgroup (row, query head),
// invocation `pair` turns its pair of the published row back by the row's
// negated angles (the portable `output_rotary`), in F32, and publishes it
// again. Every attention entry with the output rotary launches it last.
void attention_rotate_output(uint64_t result, uint64_t coordinates, uint64_t components, uint64_t frequencies,
    uint64_t amplitudes) {
    const uint64_t row = gl_WorkGroupID.x;
    const uint64_t head = gl_WorkGroupID.y;
    const uint pair = gl_LocalInvocationIndex;
    if (pair >= ATTENTION_P)
        return;
    float cosine, sine;
    rotary_turn(coordinates + row * 16ul, 1ul, components, 1ul, frequencies, 1ul, amplitudes, 1ul, pair, cosine, sine);
    const rotary_layout pairs = ATTENTION_ROTARY;
    const uint lo = rotary_leading(pair, pairs);
    const uint hi = rotary_partner(lo, pairs.p, pairs.first, pairs.adjacent);
    const uint64_t row_at = (row * uint64_t(ATTENTION_KV * ATTENTION_G) + head) * ATTENTION_W;
    const float low = element_at(ELEMENT_ACT, result, row_at + lo);
    const float high = element_at(ELEMENT_ACT, result, row_at + hi);
    element_put(ELEMENT_ACT, result, row_at + lo, low * cosine + high * sine);
    element_put(ELEMENT_ACT, result, row_at + hi, high * cosine - low * sine);
}
