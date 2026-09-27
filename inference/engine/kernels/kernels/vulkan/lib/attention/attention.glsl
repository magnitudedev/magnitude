// Shared pieces of the attention family entries (`attention_decode`,
// `attention_prefill` and their K8/V4 forms): the per-head preparation
// (optional RMS norm and amplitude-scaled partial M-RoPE) held in one
// subgroup's registers, a row's span walk, decode partition bounds, the
// online-softmax absorb, the fixed-order merge of partial states, the decode
// partition publication and gated merge, and the
// K/V append. The counterpart of `metal/lib/attention/attention.h`.
//
// Every dense operand is bound canonically (row-major, unit innermost
// stride), so rows are addressed from their logical offsets. Activations are
// the entry's element A (ELEMENT_ACT); the head width W = 2P + S is a
// multiple of 32 and each of a subgroup's 32 lanes owns E = W / 32
// contiguous columns.
#include "../core/activation.glsl"
#include "../core/rotary.glsl"

#define ATTENTION_W uint(2ul * SEISMIC_DIM_P + SEISMIC_DIM_S)
#define ATTENTION_E (ATTENTION_W / 32u)
#define ATTENTION_P uint(SEISMIC_DIM_P)
#define ATTENTION_KV uint(SEISMIC_DIM_KV)
#define ATTENTION_G uint(SEISMIC_DIM_G)
// Query heads one decode subgroup holds in registers. The decode entries split
// the G heads of a kv head into ATTENTION_G / ATTENTION_HEADS slices (their
// SLICES parameter) and define it before including this file; a subgroup
// holding all G heads of a large group (G = 16, E = 8: 256 query and output
// floats per lane) no longer fits in registers.
#ifndef ATTENTION_HEADS
#define ATTENTION_HEADS ATTENTION_G
#endif
#define ATTENTION_LOG2E 1.4426950408889634

#define ATTENTION_INF uintBitsToFloat(0x7f800000u)

// The entry's form (contract in attention.seismic), defined by every entry
// before including this file: ATTENTION_I interleaved gate columns after each
// query head's W columns (0 or W), ATTENTION_U separate gate values per query
// head (0, 1 or W), ATTENTION_FRESH (the layer has fresh rows),
// ATTENTION_NORM (q/k RMS norm), ATTENTION_VALUE_NORM (value RMS norm) and
// ATTENTION_SOFTPLUS (the gate function is softplus rather than sigmoid).
#define ATTENTION_QUERY_STRIDE (ATTENTION_W + ATTENTION_I)

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

// Lane `lane`'s E columns of `x` RMS-normalized with `norm` (F32 weights). The
// whole subgroup calls it: the square sum is a subgroup reduction.
void attention_normalize(uint64_t norm, float epsilon, uint lane, inout float x[ATTENTION_E]) {
    float squares = 0.0;
    [[unroll]] for (uint i = 0u; i < ATTENTION_E; ++i)
        squares = seismic_fma_rn(x[i], x[i], squares);
    squares = seismic_subgroup_sum_f32(squares);
    const float inverse = inversesqrt(seismic_div_rn(squares, float(ATTENTION_W)) + epsilon);
    [[unroll]] for (uint i = 0u; i < ATTENTION_E; ++i)
        x[i] = x[i] * inverse * element_f32_at(norm + uint64_t(lane * ATTENTION_E + i) * 4ul);
}

// One value head row at `raw` (A elements) into lane `lane`'s E columns,
// RMS-normalized with `norm` and rounded to A when ATTENTION_VALUE_NORM.
void attention_value(uint64_t raw, uint64_t norm, float epsilon, uint lane, out float x[ATTENTION_E]) {
    attention_load_row(raw, uint64_t(lane) * ATTENTION_E, x);
    if (ATTENTION_VALUE_NORM) {
        attention_normalize(norm, epsilon, lane, x);
        [[unroll]] for (uint i = 0u; i < ATTENTION_E; ++i)
            x[i] = element_round(ELEMENT_ACT, x[i]);
    }
}

// One head row at `raw` (A elements), RMS-normalized with `norm` (F32) when
// ATTENTION_NORM and rotated on its first 2P columns (pair p by coordinate
// axis components[p] at frequencies[p], cosine and sine scaled by
// amplitudes[p]), into lane `lane`'s E columns. The
// whole subgroup calls it: each rotated column's pair partner lives P / E
// lanes away.
void attention_prepare(uint64_t raw, uint64_t norm, uint64_t coordinates, uint64_t components, uint64_t frequencies,
    uint64_t amplitudes, float epsilon, uint lane, out float x[ATTENTION_E]) {
    attention_load_row(raw, uint64_t(lane) * ATTENTION_E, x);
    if (ATTENTION_NORM)
        attention_normalize(norm, epsilon, lane, x);
    if (ATTENTION_P == 0u)
        return;
    const uint half_lanes = ATTENTION_P / ATTENTION_E;
    const uint partner_lane = lane < half_lanes ? lane + half_lanes : (lane < 2u * half_lanes ? lane - half_lanes : lane);
    float partner[ATTENTION_E];
    [[unroll]] for (uint i = 0u; i < ATTENTION_E; ++i)
        partner[i] = seismic_shuffle(x[i], partner_lane);
    if (lane >= 2u * half_lanes)
        return;
    [[unroll]] for (uint i = 0u; i < ATTENTION_E; ++i) {
        const uint column = lane * ATTENTION_E + i;
        const uint pair = column % max(ATTENTION_P, 1u);
        float c;
        const int axis = element_i32_at(components + uint64_t(pair) * 4ul);
        const float angle = float(element_i32_at(coordinates + uint64_t(axis) * 4ul))
            * element_f32_at(frequencies + uint64_t(pair) * 4ul);
        float s = rotary_sincos(angle, c);
        const float amplitude = element_f32_at(amplitudes + uint64_t(pair) * 4ul);
        c *= amplitude;
        s *= amplitude;
        x[i] = column < ATTENTION_P ? x[i] * c - partner[i] * s : x[i] * c + partner[i] * s;
    }
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

// The total number of keys a row sees: its visible spans, then its fresh span.
uint attention_visible_total(uint64_t visible, uint64_t fresh, uint64_t row, uint64_t spans) {
    uint total = 0u;
    for (uint64_t index = 0ul; index <= spans; ++index) {
        int lo, hi;
        attention_span(visible, fresh, row, spans, index, lo, hi);
        total += uint(max(hi - lo, 0));
    }
    return total;
}

// Keys per decode partition for a row seeing `total` keys: at least `span`,
// and few enough that `parts` partitions cover the row. Never zero.
uint attention_partition_span(uint total, uint span, uint parts) {
    return max(span, (total + parts - 1u) / parts);
}

// The online-softmax state of a subgroup's ATTENTION_HEADS query heads over
// its keys, in the exp2 domain, absorbing `n` (1 or 4) keys at a time.
void attention_absorb(const uint n, float q[ATTENTION_HEADS][ATTENTION_E], float k[4][ATTENTION_E],
    float v[4][ATTENTION_E], inout float maximum[ATTENTION_HEADS], inout float denominator[ATTENTION_HEADS],
    inout float result[ATTENTION_HEADS][ATTENTION_E]) {
    [[unroll]] for (uint g = 0u; g < ATTENTION_HEADS; ++g) {
        float score[4];
        [[unroll]] for (uint j = 0u; j < 4u; ++j) {
            if (j < n) {
                float partial = 0.0;
                [[unroll]] for (uint i = 0u; i < ATTENTION_E; ++i)
                    partial = seismic_fma_rn(q[g][i], k[j][i], partial);
                score[j] = seismic_subgroup_sum_f32(partial);
            }
        }
        float next = maximum[g];
        [[unroll]] for (uint j = 0u; j < 4u; ++j)
            if (j < n)
                next = max(next, score[j]);
        const float carry = exp2(maximum[g] - next);
        float probability[4];
        float sum = 0.0;
        [[unroll]] for (uint j = 0u; j < 4u; ++j) {
            if (j < n) {
                probability[j] = exp2(score[j] - next);
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

// The fixed-order merge of `count` partial attention states for one column:
// state p is slot first + p * stride, with its unnormalized output at
// partials[slot * W + column] (relative to its maximum) and (maximum,
// denominator) at statistics[slot * 2]. Empty states (denominator 0) are
// skipped; a column with no state attends to zero.
float attention_merge(uint64_t partials, uint64_t statistics, uint64_t first, uint64_t stride, uint count, uint column) {
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
    return seismic_div_rn(accumulated, max(denominator, 1e-30));
}

// Decode subgroups: subgroup sg holds the ATTENTION_HEADS query heads of
// slice sg % slices (heads slice * ATTENTION_HEADS ..) over key group
// sg / slices, one of subgroups / slices contiguous sub-ranges of the
// partition. Every key is read by one subgroup per slice.
uint attention_slices() { return ATTENTION_G / ATTENTION_HEADS; }
uint attention_slice(uint sg) { return sg % attention_slices(); }
uint attention_key_group(uint sg) { return sg / attention_slices(); }
uint attention_key_groups() { return SEISMIC_SUBGROUPS / attention_slices(); }

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

// The decode merge of one (query head, row) column over the row's `spans`
// visible spans: the row's non-empty partitions in partition order, then the
// output gate. A row that sees no key attends to zero.
void attention_decode_gate(uint64_t query, uint64_t gate, uint64_t visible, uint64_t fresh, uint64_t result,
    uint64_t partials, uint64_t statistics, uint64_t spans, uint64_t head, uint64_t row, uint column, uint span,
    uint parts) {
    const uint total = attention_visible_total(visible, fresh, row, spans);
    const uint span_keys = attention_partition_span(total, span, parts);
    const uint used = (total + span_keys - 1u) / span_keys;
    const float attended = attention_merge(partials, statistics,
        (row * ATTENTION_KV * ATTENTION_G + head) * parts, 1ul, used, column);
    attention_store_gated(query, gate, result, row, head, column, attended);
}
