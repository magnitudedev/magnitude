// Shared pieces of the gated attention entries (`gated_attention_decode`,
// `gated_attention_prefill`): the per-head preparation (RMS norm and partial
// M-RoPE) held in one subgroup's registers, a row's span walk, decode
// partition bounds, the online-softmax absorb, the fixed-order merge of
// partial states, the decode partition publication and gated merge, and the
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
#define ATTENTION_LOG2E 1.4426950408889634

#define ATTENTION_INF uintBitsToFloat(0x7f800000u)

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

// One head row at `raw` (A elements), RMS-normalized with `norm` (F32) and
// rotated on its first 2P columns (pair p by coordinate axis components[p] at
// frequencies[p]), into lane `lane`'s E columns. The whole subgroup calls it:
// the square sum is a subgroup reduction and each rotated column's pair
// partner lives P / E lanes away.
void attention_prepare(uint64_t raw, uint64_t norm, uint64_t coordinates, uint64_t components, uint64_t frequencies,
    float epsilon, uint lane, out float x[ATTENTION_E]) {
    attention_load_row(raw, uint64_t(lane) * ATTENTION_E, x);
    float squares = 0.0;
    [[unroll]] for (uint i = 0u; i < ATTENTION_E; ++i)
        squares = seismic_fma_rn(x[i], x[i], squares);
    squares = seismic_subgroup_sum_f32(squares);
    const float inverse = inversesqrt(seismic_div_rn(squares, float(ATTENTION_W)) + epsilon);
    [[unroll]] for (uint i = 0u; i < ATTENTION_E; ++i)
        x[i] = x[i] * inverse * element_f32_at(norm + uint64_t(lane * ATTENTION_E + i) * 4ul);
    const uint half_lanes = ATTENTION_P / ATTENTION_E;
    const uint partner_lane = lane < half_lanes ? lane + half_lanes : (lane < 2u * half_lanes ? lane - half_lanes : lane);
    float partner[ATTENTION_E];
    [[unroll]] for (uint i = 0u; i < ATTENTION_E; ++i)
        partner[i] = subgroupShuffle(x[i], partner_lane);
    if (lane >= 2u * half_lanes)
        return;
    [[unroll]] for (uint i = 0u; i < ATTENTION_E; ++i) {
        const uint column = lane * ATTENTION_E + i;
        const uint pair = column % ATTENTION_P;
        float c;
        const int axis = element_i32_at(components + uint64_t(pair) * 4ul);
        const float angle = float(element_i32_at(coordinates + uint64_t(axis) * 4ul))
            * element_f32_at(frequencies + uint64_t(pair) * 4ul);
        const float s = rotary_sincos(angle, c);
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
// span R (rows of this batch).
void attention_span(uint64_t visible, uint64_t fresh, uint64_t row, uint64_t spans, uint64_t span, out int lo, out int hi) {
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

// The online-softmax state of G query heads over one subgroup's keys, in the
// exp2 domain, absorbing `n` (1 or 4) keys at a time.
void attention_absorb(const uint n, float q[ATTENTION_G][ATTENTION_E], float k[4][ATTENTION_E],
    float v[4][ATTENTION_E], inout float maximum[ATTENTION_G], inout float denominator[ATTENTION_G],
    inout float result[ATTENTION_G][ATTENTION_E]) {
    [[unroll]] for (uint g = 0u; g < ATTENTION_G; ++g) {
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

// Publishes one decode partition: the workgroup's subgroup states of G query
// heads (outputs relative to their own maxima) merge in subgroup order into
// one partial per query head g at slot first + g * parts: the unnormalized
// output at partials[slot * W] relative to the partition maximum, and
// (maximum, denominator) at statistics[slot * 2]. Shared floats: `states`
// holds [subgroups][G][2], `columns` [subgroups][W].
void attention_publish(float maximum[ATTENTION_G], float denominator[ATTENTION_G],
    float result[ATTENTION_G][ATTENTION_E], uint states, uint columns, uint64_t partials, uint64_t statistics,
    uint64_t first, uint parts) {
    const uint sg = gl_SubgroupID, lane = gl_SubgroupInvocationID, subgroups = gl_NumSubgroups;
    const uint thread = gl_LocalInvocationIndex;
    if (lane == 0u) {
        [[unroll]] for (uint g = 0u; g < ATTENTION_G; ++g) {
            seismic_shared_f32[states + (sg * ATTENTION_G + g) * 2u] = maximum[g];
            seismic_shared_f32[states + (sg * ATTENTION_G + g) * 2u + 1u] = denominator[g];
        }
    }
    barrier();
    [[unroll]] for (uint g = 0u; g < ATTENTION_G; ++g) {
        float partition_maximum = -ATTENTION_INF;
        for (uint s = 0u; s < subgroups; ++s)
            if (seismic_shared_f32[states + (s * ATTENTION_G + g) * 2u + 1u] > 0.0)
                partition_maximum = max(partition_maximum, seismic_shared_f32[states + (s * ATTENTION_G + g) * 2u]);
        const float weight = denominator[g] > 0.0 ? exp2(maximum[g] - partition_maximum) : 0.0;
        [[unroll]] for (uint i = 0u; i < ATTENTION_E; ++i)
            seismic_shared_f32[columns + sg * ATTENTION_W + lane * ATTENTION_E + i] = result[g][i] * weight;
        barrier();
        const uint64_t slot = first + uint64_t(g) * parts;
        for (uint column = thread; column < ATTENTION_W; column += gl_WorkGroupSize.x) {
            float sum = 0.0;
            for (uint s = 0u; s < subgroups; ++s)
                sum += seismic_shared_f32[columns + s * ATTENTION_W + column];
            element_f32_put(partials + (slot * ATTENTION_W + column) * 4ul, sum);
        }
        if (thread == 0u) {
            float total_denominator = 0.0;
            for (uint s = 0u; s < subgroups; ++s) {
                const float d = seismic_shared_f32[states + (s * ATTENTION_G + g) * 2u + 1u];
                if (d > 0.0)
                    total_denominator = seismic_fma_rn(d,
                        exp2(seismic_shared_f32[states + (s * ATTENTION_G + g) * 2u] - partition_maximum), total_denominator);
            }
            element_f32_put(statistics + slot * 8ul, partition_maximum);
            element_f32_put(statistics + slot * 8ul + 4ul, total_denominator);
        }
        barrier();
    }
}

// The sigmoid-gated store of one attended column of query head `head` of
// `row`: the gate is the second W of the head's query_gate slice.
void attention_store_gated(uint64_t query_gate, uint64_t gated, uint64_t row, uint64_t head, uint column, float attended) {
    const uint64_t heads = uint64_t(ATTENTION_KV * ATTENTION_G);
    const float gate = element_at(ELEMENT_ACT, query_gate, (row * heads + head) * 2ul * ATTENTION_W + ATTENTION_W + column);
    element_put(ELEMENT_ACT, gated, (row * heads + head) * ATTENTION_W + column, seismic_div_rn(attended, 1.0 + exp(-gate)));
}

// The decode merge of one (query head, row) column over the row's `spans`
// visible spans: the row's non-empty partitions in partition order, then the
// sigmoid gate. A row that sees no key attends to zero.
void attention_decode_gate(uint64_t query_gate, uint64_t visible, uint64_t fresh, uint64_t gated, uint64_t partials,
    uint64_t statistics, uint64_t spans, uint64_t head, uint64_t row, uint column, uint span, uint parts) {
    const uint total = attention_visible_total(visible, fresh, row, spans);
    const uint span_keys = attention_partition_span(total, span, parts);
    const uint used = (total + span_keys - 1u) / span_keys;
    const float attended = attention_merge(partials, statistics,
        (row * ATTENTION_KV * ATTENTION_G + head) * parts, 1ul, used, column);
    attention_store_gated(query_gate, gated, row, head, column, attended);
}
