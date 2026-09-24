// The two forms of attention history: dense (rows of A) and affine K8/V4
// (`qwen_attention_*_k8v4`). The counterpart of the affine part of
// `metal/common/attention.h`.
//
// An affine (history row, kv head) vector is a code row of W * B / 32 u32
// words (code i at bits B * (i % (32 / B)) of word i / (32 / B)) plus one f16
// (scale, zero) pair per group of HISTORY_GROUP consecutive columns, pairs in
// group order; its decoded value is code * scale + zero with its group's
// pair. Keys use 8-bit codes, values 4-bit codes. A lane holding E = W / 32
// columns of a vector owns E * B code bits: whole words when E * B >= 32,
// else a power-of-two part of a word shared with its neighbours (W is a power
// of two); its columns lie in one group, shared by HISTORY_GROUP / E lanes.
//
// `attention_history` names an entry's history buffers; its `affine` field is
// a compile-time constant at every construction, so the driver folds the form
// switches (the projection library's convention).
//
// This file is independent of any entry ABI.
#include "common/attention.glsl"
#include "common/flash.glsl"

#define HISTORY_KEY_BITS 8u
#define HISTORY_VALUE_BITS 4u
// Columns per (scale, zero) pair: the codec's group (model-state
// `AFFINE_GROUP`); the (scale, zero) pairs per vector; the lanes sharing one.
#define HISTORY_GROUP 32u
#define HISTORY_PAIRS (ATTENTION_W / HISTORY_GROUP)
#define HISTORY_PAIR_LANES (HISTORY_GROUP / ATTENTION_E)

struct attention_history {
    bool affine;
    uint64_t key;                 // dense: [T, KV, W] A; affine: key codes
    uint64_t value;               // dense: [T, KV, W] A; affine: value codes
    uint64_t key_coefficients;    // affine: [T, KV, 2 * HISTORY_PAIRS] f16
    uint64_t value_coefficients;
};

attention_history attention_dense_history(uint64_t key, uint64_t value) {
    return attention_history(false, key, value, 0ul, 0ul);
}

attention_history attention_affine_history(uint64_t key_codes, uint64_t key_coefficients, uint64_t value_codes,
    uint64_t value_coefficients) {
    return attention_history(true, key_codes, value_codes, key_coefficients, value_coefficients);
}

// ---------------------------------------------------------------------------
// The affine codec.

uint history_levels(const uint b) { return (1u << b) - 1u; }
uint history_row_words(const uint b) { return ATTENTION_W * b / 32u; }

// This lane's codes of one vector's code row at byte address `row`, shifted so
// its code i sits at bits B * i of word (i * B) / 32.
void history_lane_load(const uint b, uint64_t row, uint lane, out uint w[2]) {
    const uint bits = ATTENTION_E * b;
    w[0] = 0u;
    w[1] = 0u;
    if (bits == 64u) {
        const uvec2 pair = element_uvec2_at(row + uint64_t(lane) * 8ul);
        w[0] = pair.x;
        w[1] = pair.y;
    } else if (bits == 32u) {
        w[0] = element_u32_at(row + uint64_t(lane) * 4ul);
    } else {
        w[0] = element_u32_at(row + uint64_t(lane * bits / 32u) * 4ul) >> ((lane * bits) % 32u);
    }
}

// Code i of this lane's columns, as F32 (exact).
float history_code(const uint b, uint w[2], uint i) {
    return float((w[(i * b) / 32u] >> ((i * b) % 32u)) & history_levels(b));
}

// The (scale, zero) pair of group `group` of one vector.
vec2 history_coefficients(uint64_t coefficients, uint64_t vector, uint group) {
    return unpackHalf2x16(element_u32_at(coefficients + (vector * HISTORY_PAIRS + group) * 4ul));
}

// Encodes one vector held by a subgroup (lane `lane` owns columns [lane * E,
// lane * E + E)) with B-bit codes at code row `row` and its group coefficient
// pairs at `coefficients`: per group, zero = f16(min), scale =
// f16((max - min) / L), code = min(L, u32(fma(x - zero, 1 / scale, 0.5))), 0
// when scale is 0.
void history_encode(const uint b, float x[ATTENTION_E], uint64_t row, uint64_t coefficients, uint lane) {
    const uint levels = history_levels(b);
    const uint bits = ATTENTION_E * b;
    float low = x[0], high = x[0];
    [[unroll]] for (uint i = 1u; i < ATTENTION_E; ++i) {
        low = min(low, x[i]);
        high = max(high, x[i]);
    }
    // The group's range, over its HISTORY_PAIR_LANES neighbouring lanes.
    [[unroll]] for (uint offset = 1u; offset < 32u; offset *= 2u) {
        if (offset < HISTORY_PAIR_LANES) {
            low = min(low, subgroupShuffleXor(low, offset));
            high = max(high, subgroupShuffleXor(high, offset));
        }
    }
    const float zero = element_round(ELEMENT_F16, low);
    const float scale = element_round(ELEMENT_F16, seismic_div_rn(high - low, float(levels)));
    const float inverse = scale > 0.0 ? seismic_div_rn(1.0, scale) : 0.0;
    uint w[2];
    w[0] = 0u;
    w[1] = 0u;
    [[unroll]] for (uint i = 0u; i < ATTENTION_E; ++i) {
        const float t = seismic_fma_rn(x[i] - zero, inverse, 0.5);
        const uint c = min(uint(max(t, 0.0)), levels);
        w[(i * b) / 32u] |= c << ((i * b) % 32u);
    }
    if (bits >= 32u) {
        const uint words = bits / 32u;
        [[unroll]] for (uint j = 0u; j < 2u; ++j)
            if (j < words)
                element_u32_put(row + uint64_t(lane * words + j) * 4ul, w[j]);
    } else {
        // 32 / bits neighbouring lanes share a word: join their parts.
        const uint sharing = 32u / bits;
        uint joined = w[0] << ((lane % sharing) * bits);
        [[unroll]] for (uint offset = 1u; offset < 32u; offset *= 2u)
            if (offset < sharing)
                joined |= subgroupShuffleXor(joined, offset);
        if (lane % sharing == 0u)
            element_u32_put(row + uint64_t(lane / sharing) * 4ul, joined);
    }
    if (lane % HISTORY_PAIR_LANES == 0u)
        element_u32_put(coefficients + uint64_t(lane / HISTORY_PAIR_LANES) * 4ul,
            element_pack2(ELEMENT_F16, scale, zero));
}

// ---------------------------------------------------------------------------
// Appending one row (the subgroup's E columns per lane) at history row
// `destination`: the key (prepared, already rounded to A) or the value.
void history_append(attention_history h, const bool is_key, int destination, uint kv_head, uint lane,
    float x[ATTENTION_E]) {
    if (!h.affine) {
        attention_append(is_key ? h.key : h.value, destination, kv_head, lane, x);
        return;
    }
    const uint b = is_key ? HISTORY_KEY_BITS : HISTORY_VALUE_BITS;
    const uint64_t vector = uint64_t(destination) * ATTENTION_KV + kv_head;
    history_encode(b, x, (is_key ? h.key : h.value) + vector * history_row_words(b) * 4ul,
        (is_key ? h.key_coefficients : h.value_coefficients) + vector * HISTORY_PAIRS * 4ul, lane);
}

// ---------------------------------------------------------------------------
// Staging history rows [first, first + FLASH_KEYS) of one kv head as the f16
// tile at shared half `base` (row pitch W + 8); rows at or past `end` are
// zero. Affine rows decode as code * scale + zero rounded to f16 directly (a
// rounding to a BF16 A first would cost the 8-bit keys up to a code step).
// Every invocation of the workgroup takes part.
void history_stage(attention_history h, const bool is_key, int first, int end, uint kv_head, uint base) {
    if (!h.affine) {
        flash_stage(ELEMENT_ACT, is_key ? h.key : h.value, uint64_t(ATTENTION_KV * ATTENTION_W),
            uint64_t(kv_head * ATTENTION_W), first, end, ATTENTION_W, base);
        return;
    }
    const uint b = is_key ? HISTORY_KEY_BITS : HISTORY_VALUE_BITS;
    const uint64_t codes = is_key ? h.key : h.value;
    const uint64_t coefficients = is_key ? h.key_coefficients : h.value_coefficients;
    const uint pieces = ATTENTION_W / 8u;
    for (uint item = gl_LocalInvocationIndex; item < FLASH_KEYS * pieces; item += gl_WorkGroupSize.x) {
        const uint k = item / pieces;
        const uint c = (item % pieces) * 8u;
        const int t = first + int(k);
        uvec4 bits = uvec4(0u);
        if (t < end) {
            const uint64_t vector = uint64_t(t) * ATTENTION_KV + kv_head;
            const vec2 sz = history_coefficients(coefficients, vector, c / HISTORY_GROUP);
            const uint64_t word = codes + (vector * history_row_words(b) + (c * b) / 32u) * 4ul;
            uint w[2];
            w[0] = element_u32_at(word);
            w[1] = b == 8u ? element_u32_at(word + 4ul) : 0u;
            float v[8];
            [[unroll]] for (uint i = 0u; i < 8u; ++i)
                v[i] = seismic_fma_rn(history_code(b, w, i), sz.x, sz.y);
            bits = element_pack8(ELEMENT_F16, vec4(v[0], v[2], v[4], v[6]), vec4(v[1], v[3], v[5], v[7]));
        }
        seismic_shared_uvec4[(base + k * flash_pitch(ATTENTION_W) + c) / 8u] = bits;
    }
}

// ---------------------------------------------------------------------------
// The online-softmax state of G query heads absorbing `n` (at most
// HISTORY_BATCH) affine-coded keys and values, with this lane's group pairs.
// Scores are corrected, not decoded: the sum over groups of
// scale * (q . code) + zero * sum(q), each lane adding its columns' share
// (`qsum` is the sum of this lane's query columns). The value product
// accumulates (p * scale) * code into `result` and sum(p * zero) into this
// lane's per-head `bias`, both carried by the same factor, so result + bias is
// the attended sum.
#define HISTORY_BATCH 8u

void history_absorb_affine(const uint n, float q[ATTENTION_G][ATTENTION_E], float qsum[ATTENTION_G],
    uint key[HISTORY_BATCH][2], vec2 key_coefficients[HISTORY_BATCH], uint value[HISTORY_BATCH][2],
    vec2 value_coefficients[HISTORY_BATCH], inout float maximum[ATTENTION_G], inout float denominator[ATTENTION_G],
    inout float result[ATTENTION_G][ATTENTION_E], inout float bias[ATTENTION_G]) {
    float score[ATTENTION_G][HISTORY_BATCH];
    [[unroll]] for (uint j = 0u; j < HISTORY_BATCH; ++j) {
        if (j < n) {
            float k[ATTENTION_E];
            [[unroll]] for (uint i = 0u; i < ATTENTION_E; ++i)
                k[i] = history_code(HISTORY_KEY_BITS, key[j], i);
            [[unroll]] for (uint g = 0u; g < ATTENTION_G; ++g) {
                float partial = 0.0;
                [[unroll]] for (uint i = 0u; i < ATTENTION_E; ++i)
                    partial = seismic_fma_rn(q[g][i], k[i], partial);
                score[g][j] = seismic_subgroup_sum_f32(seismic_fma_rn(key_coefficients[j].x, partial,
                    key_coefficients[j].y * qsum[g]));
            }
        }
    }
    // Each value code's weight: its probability times the value scale.
    float weight[ATTENTION_G][HISTORY_BATCH];
    [[unroll]] for (uint g = 0u; g < ATTENTION_G; ++g) {
        float next = maximum[g];
        [[unroll]] for (uint j = 0u; j < HISTORY_BATCH; ++j)
            if (j < n)
                next = max(next, score[g][j]);
        const float carry = exp2(maximum[g] - next);
        float sum = 0.0;
        float offset = bias[g] * carry;
        [[unroll]] for (uint j = 0u; j < HISTORY_BATCH; ++j) {
            if (j < n) {
                const float probability = exp2(score[g][j] - next);
                sum += probability;
                offset = seismic_fma_rn(probability, value_coefficients[j].y, offset);
                weight[g][j] = probability * value_coefficients[j].x;
            }
        }
        denominator[g] = seismic_fma_rn(denominator[g], carry, sum);
        bias[g] = offset;
        maximum[g] = next;
        [[unroll]] for (uint i = 0u; i < ATTENTION_E; ++i)
            result[g][i] *= carry;
    }
    [[unroll]] for (uint j = 0u; j < HISTORY_BATCH; ++j) {
        if (j < n) {
            float v[ATTENTION_E];
            [[unroll]] for (uint i = 0u; i < ATTENTION_E; ++i)
                v[i] = history_code(HISTORY_VALUE_BITS, value[j], i);
            [[unroll]] for (uint g = 0u; g < ATTENTION_G; ++g)
                [[unroll]] for (uint i = 0u; i < ATTENTION_E; ++i)
                    result[g][i] = seismic_fma_rn(weight[g][j], v[i], result[g][i]);
        }
    }
}
