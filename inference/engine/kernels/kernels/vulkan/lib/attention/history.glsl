// The two forms of attention history: dense (rows of A) and affine K8/V4
// (`attention_*_k8v4`). The counterpart of the affine part of
// `metal/lib/attention/attention.h`.
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
#include "attention.glsl"
#include "flash.glsl"

#define HISTORY_KEY_BITS 8u
#define HISTORY_VALUE_BITS 4u
// Columns per (scale, zero) pair: the codec's group (model-state
// `AFFINE_GROUP`); the (scale, zero) pairs per vector; the lanes sharing one.
#define HISTORY_GROUP 32u
#define HISTORY_PAIRS (ATTENTION_W / HISTORY_GROUP)
#define HISTORY_PAIR_LANES (HISTORY_GROUP / ATTENTION_E)
// The words a lane's codes of one vector span (keys, the wider codes): two
// up to W = 256, four at W = 512.
#define HISTORY_LANE_WORDS (ATTENTION_E > 8u ? ATTENTION_E / 4u : 2u)

struct attention_history {
    bool affine;
    uint64_t key;                 // dense: [T, KV, W] A; affine: key codes
    uint64_t value;               // dense: [T, KV, W] A; affine: value codes
    uint64_t key_coefficients;    // affine: [T, KV, 2 * HISTORY_PAIRS] f16
    uint64_t value_coefficients;
    uint64_t slab_rows;
};

attention_history attention_dense_history(uint64_t key, uint64_t value) {
    return attention_history(false, key, value, 0ul, 0ul, SEISMIC_PARAM_SLAB_ROWS);
}

attention_history attention_affine_history(uint64_t key_codes, uint64_t key_coefficients, uint64_t value_codes,
    uint64_t value_coefficients) {
    return attention_history(true, key_codes, value_codes, key_coefficients, value_coefficients, SEISMIC_PARAM_SLAB_ROWS);
}

uint64_t history_address(uint64_t table, uint64_t row, uint64_t slab_rows, uint64_t row_bytes,
    uint64_t head_bytes, uint kv_head) {
    return slab_row(table, uint(row), uint(slab_rows), row_bytes) + uint64_t(kv_head) * head_bytes;
}

// ---------------------------------------------------------------------------
// The affine codec.

uint history_levels(const uint b) { return (1u << b) - 1u; }
uint history_row_words(const uint b) { return ATTENTION_W * b / 32u; }

// This lane's codes of one vector's code row at byte address `row`, shifted so
// its code i sits at bits B * i of word (i * B) / 32.
void history_lane_load(const uint b, uint64_t row, uint lane, out uint w[HISTORY_LANE_WORDS]) {
    const uint bits = ATTENTION_E * b;
    [[unroll]] for (uint j = 0u; j < HISTORY_LANE_WORDS; ++j)
        w[j] = 0u;
    if (bits == 128u) {
        const uvec4 words = element_uvec4_at(row + uint64_t(lane) * 16ul);
        [[unroll]] for (uint j = 0u; j < 4u; ++j)
            w[j] = words[j];
    } else if (bits == 64u) {
        const uvec2 pair = element_uvec2_at(row + uint64_t(lane) * 8ul);
        w[0] = pair.x;
        w[1] = pair.y;
    } else if (bits == 32u) {
        w[0] = element_u32_at(row + uint64_t(lane) * 4ul);
    } else {
        w[0] = element_u32_at(row + uint64_t(lane * bits / 32u) * 4ul) >> ((lane * bits) % 32u);
    }
}

// The B-bit code at bit `shift` of `word`, as F32 (exact). The code becomes
// the top mantissa bits of 1 + code / 2^B, which one exact FMA rescales:
// shifts and a logic op instead of an integer conversion (a quarter-rate
// instruction on NVIDIA hardware).
float history_code_at(const uint b, uint word, const uint shift) {
    const uint top = 23u - b;
    const uint placed = shift <= top ? word << (top - shift) : word >> (shift - top);
    const float mantissa = uintBitsToFloat((placed & (history_levels(b) << top)) | 0x3f800000u);
    return seismic_fma_rn(mantissa, float(1u << b), -float(1u << b));
}

// Code i of this lane's columns, as F32 (exact).
float history_code(const uint b, uint w[HISTORY_LANE_WORDS], uint i) {
    return history_code_at(b, w[(i * b) / 32u], (i * b) % 32u);
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
    uint w[HISTORY_LANE_WORDS];
    [[unroll]] for (uint j = 0u; j < HISTORY_LANE_WORDS; ++j)
        w[j] = 0u;
    [[unroll]] for (uint i = 0u; i < ATTENTION_E; ++i) {
        const float t = seismic_fma_rn(x[i] - zero, inverse, 0.5);
        const uint c = min(uint(max(t, 0.0)), levels);
        w[(i * b) / 32u] |= c << ((i * b) % 32u);
    }
    if (bits >= 32u) {
        const uint words = bits / 32u;
        [[unroll]] for (uint j = 0u; j < HISTORY_LANE_WORDS; ++j)
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
        const uint64_t head_bytes = uint64_t(ATTENTION_W) * ELEMENT_BYTES(ELEMENT_ACT);
        const uint64_t address = history_address(is_key ? h.key : h.value, uint64_t(destination),
            h.slab_rows, uint64_t(ATTENTION_KV) * head_bytes, head_bytes, kv_head);
        attention_append(address, 0, 0u, lane, x);
        return;
    }
    const uint b = is_key ? HISTORY_KEY_BITS : HISTORY_VALUE_BITS;
    const uint64_t code_bytes = uint64_t(history_row_words(b)) * 4ul;
    const uint64_t pair_bytes = uint64_t(HISTORY_PAIRS) * 4ul;
    history_encode(b, x, history_address(is_key ? h.key : h.value, uint64_t(destination), h.slab_rows,
            uint64_t(ATTENTION_KV) * code_bytes, code_bytes, kv_head),
        history_address(is_key ? h.key_coefficients : h.value_coefficients, uint64_t(destination), h.slab_rows,
            uint64_t(ATTENTION_KV) * pair_bytes, pair_bytes, kv_head), lane);
}

// ---------------------------------------------------------------------------
// Staging columns [column, column + w) of history rows [first, first +
// FLASH_KEYS) of one kv head as the f16 tile at shared half `base` (row pitch
// w + 8); rows at or past `end` are zero. Affine rows decode as code * scale
// + zero rounded to f16 directly (a rounding to a BF16 A first would cost the
// 8-bit keys up to a code step). Every invocation of the workgroup takes part.
void history_stage(attention_history h, const bool is_key, int first, int end, uint kv_head, uint column,
    const uint w, uint base) {
    if (!h.affine) {
        flash_stage_slab(ELEMENT_ACT, is_key ? h.key : h.value, h.slab_rows, ATTENTION_KV * ATTENTION_W,
            kv_head * ATTENTION_W + column, w, first, end, base);
        return;
    }
    const uint b = is_key ? HISTORY_KEY_BITS : HISTORY_VALUE_BITS;
    const uint64_t codes = is_key ? h.key : h.value;
    const uint64_t coefficients = is_key ? h.key_coefficients : h.value_coefficients;
    const uint pieces = w / 8u;
    for (uint item = gl_LocalInvocationIndex; item < FLASH_KEYS * pieces; item += gl_WorkGroupSize.x) {
        const uint k = item / pieces;
        const uint staged = (item % pieces) * 8u;
        const uint c = column + staged;
        const int t = first + int(k);
        uvec4 bits = uvec4(0u);
        if (t < end) {
            const uint64_t code_bytes = uint64_t(history_row_words(b)) * 4ul;
            const uint64_t pair_bytes = uint64_t(HISTORY_PAIRS) * 4ul;
            const uint64_t pair_row = history_address(coefficients, uint64_t(t), h.slab_rows,
                uint64_t(ATTENTION_KV) * pair_bytes, pair_bytes, kv_head);
            const vec2 sz = unpackHalf2x16(element_u32_at(pair_row + uint64_t(c / HISTORY_GROUP) * 4ul));
            const uint64_t word = history_address(codes, uint64_t(t), h.slab_rows,
                uint64_t(ATTENTION_KV) * code_bytes, code_bytes, kv_head) + uint64_t((c * b) / 32u) * 4ul;
            // Eight codes: two key words or one value word.
            uint w[HISTORY_LANE_WORDS];
            [[unroll]] for (uint j = 0u; j < HISTORY_LANE_WORDS; ++j)
                w[j] = 0u;
            w[0] = element_u32_at(word);
            w[1] = b == 8u ? element_u32_at(word + 4ul) : 0u;
            float v[8];
            [[unroll]] for (uint i = 0u; i < 8u; ++i)
                v[i] = seismic_fma_rn(history_code(b, w, i), sz.x, sz.y);
            bits = element_pack8(ELEMENT_F16, vec4(v[0], v[2], v[4], v[6]), vec4(v[1], v[3], v[5], v[7]));
        }
        seismic_shared_uvec4[(base + k * flash_pitch(w) + staged) / 8u] = bits;
    }
}

// ---------------------------------------------------------------------------
// The keywise form (`attention_keywise_*`) over affine history, a batch of up
// to ATTENTION_KEYS keys at a time. Lane t's score of key t is, per group,
// scale * (q . code) + zero * sum(q) over the whole width; each lane decodes
// its E value columns (code * scale + zero) as it accumulates them. Every
// history row is read coalesced (consecutive lanes read consecutive 16 bytes
// of a row) through the subgroup's staging tile: lane t reading key t's row
// directly leaves each load instruction one 16-byte piece of 32 rows, which
// streams DRAM at about half the rate. The tile is [ATTENTION_KEYS]
// [HISTORY_STAGE_PITCH] uvec4: a row holds HISTORY_STAGE_CHUNK key code
// bytes, or a value's codes (HISTORY_VALUE_PIECES uvec4) then its coefficient
// pairs, then a pad keeping a quarter subgroup's row reads on distinct banks.
#define HISTORY_STAGE_CHUNK (ATTENTION_W < 128u ? ATTENTION_W : 128u)
#define HISTORY_VALUE_PIECES (ATTENTION_W / 32u)
#define HISTORY_STAGE_PITCH (max(HISTORY_STAGE_CHUNK / 16u, HISTORY_VALUE_PIECES + (HISTORY_PAIRS + 3u) / 4u) + 1u)
// The words a lane's value codes of one vector span, when whole.
#define HISTORY_VALUE_WORDS (ATTENTION_E > 8u ? ATTENTION_E / 8u : 1u)

// One group's share of a keywise score: its 32 codes (`low`, then `high`,
// four a word) against the heads head0.. of the keywise queries, corrected
// with the group's pair `sz` and the query group sums at shared float `sums`
// ([G][HISTORY_PAIRS]).
void history_keywise_group(uint group, uvec4 low, uvec4 high, vec2 sz, uint queries, uint sums, uint head0,
    inout float score[ATTENTION_HEADS]) {
    float dot[ATTENTION_HEADS];
    [[unroll]] for (uint g = 0u; g < ATTENTION_HEADS; ++g)
        dot[g] = 0.0;
    [[unroll]] for (uint piece = 0u; piece < 2u; ++piece) {
        const uvec4 words = piece == 0u ? low : high;
        [[unroll]] for (uint j = 0u; j < 4u; ++j) {
            const uint column = group * HISTORY_GROUP + piece * 16u + j * 4u;
            vec4 k;
            [[unroll]] for (uint i = 0u; i < 4u; ++i)
                k[i] = history_code_at(HISTORY_KEY_BITS, words[j], 8u * i);
            [[unroll]] for (uint g = 0u; g < ATTENTION_HEADS; ++g) {
                const vec4 q = attention_keywise_query(queries, head0 + g, column);
                float d = dot[g];
                [[unroll]] for (uint i = 0u; i < 4u; ++i)
                    d = seismic_fma_rn(q[i], k[i], d);
                dot[g] = d;
            }
        }
    }
    [[unroll]] for (uint g = 0u; g < ATTENTION_HEADS; ++g)
        score[g] = seismic_fma_rn(sz.x, dot[g],
            seismic_fma_rn(sz.y, seismic_shared_f32[sums + (head0 + g) * HISTORY_PAIRS + group], score[g]));
}

// Lane t's scores of key t of a batch of `n` keys (key j's code row at `rows`
// + j * `row_stride` bytes; `pairs` is lane t's key's coefficient row), the
// rows passing through the tile at shared uvec4 `tile` HISTORY_STAGE_CHUNK
// columns at a time. Rows past the batch repeat its last key.
void history_keywise_score(const uint n, uint64_t rows, uint64_t row_stride, uint64_t pairs, uint tile,
    uint queries, uint sums, uint head0, out float score[ATTENTION_HEADS]) {
    const uint lane = SEISMIC_LANE;
    const uint pieces = HISTORY_STAGE_CHUNK / 16u;
    const uint pitch = HISTORY_STAGE_PITCH;
    [[unroll]] for (uint g = 0u; g < ATTENTION_HEADS; ++g)
        score[g] = 0.0;
    [[unroll]] for (uint first = 0u; first < ATTENTION_W; first += HISTORY_STAGE_CHUNK) {
        uvec4 staged[HISTORY_STAGE_CHUNK / 16u];
        [[unroll]] for (uint i = 0u; i < pieces; ++i) {
            const uint item = i * 32u + lane;
            staged[i] = element_uvec4_at(rows + uint64_t(min(item / pieces, n - 1u)) * row_stride
                + uint64_t(first + (item % pieces) * 16u));
        }
        // The tile's previous rows are read.
        subgroupBarrier();
        [[unroll]] for (uint i = 0u; i < pieces; ++i) {
            const uint item = i * 32u + lane;
            seismic_shared_uvec4[tile + (item / pieces) * pitch + item % pieces] = staged[i];
        }
        subgroupMemoryBarrierShared();
        subgroupBarrier();
        [[unroll]] for (uint group = first / HISTORY_GROUP; group < (first + HISTORY_STAGE_CHUNK) / HISTORY_GROUP;
            ++group) {
            const uint piece = (group * HISTORY_GROUP - first) / 16u;
            history_keywise_group(group, seismic_shared_uvec4[tile + lane * pitch + piece],
                seismic_shared_uvec4[tile + lane * pitch + piece + 1u],
                unpackHalf2x16(element_u32_at(pairs + uint64_t(group) * 4ul)), queries, sums, head0, score);
        }
    }
}

// Stage the value rows after scoring. Reading them directly into the tile
// avoids keeping W/16 code vectors and W/32 coefficient words live across the
// key score, which otherwise limits occupancy for wide heads.
void history_keywise_value_stage(const uint n, uint64_t rows, uint64_t row_stride, uint64_t pairs,
    uint64_t pair_stride, uint tile) {
    const uint lane = SEISMIC_LANE;
    const uint pitch = HISTORY_STAGE_PITCH;
    [[unroll]] for (uint i = 0u; i < HISTORY_VALUE_PIECES; ++i) {
        const uint item = i * 32u + lane;
        seismic_shared_uvec4[tile + (item / HISTORY_VALUE_PIECES) * pitch + item % HISTORY_VALUE_PIECES] =
            element_uvec4_at(rows + uint64_t(min(item / HISTORY_VALUE_PIECES, n - 1u)) * row_stride
            + uint64_t(item % HISTORY_VALUE_PIECES) * 16ul);
    }
    [[unroll]] for (uint i = 0u; i < HISTORY_PAIRS; ++i) {
        const uint item = i * 32u + lane;
        seismic_shared_u32[(tile + (item / HISTORY_PAIRS) * pitch + HISTORY_VALUE_PIECES) * 4u + item % HISTORY_PAIRS] =
            element_u32_at(pairs + uint64_t(min(item / HISTORY_PAIRS, n - 1u)) * pair_stride
                + uint64_t(item % HISTORY_PAIRS) * 4ul);
    }
    subgroupMemoryBarrierShared();
    subgroupBarrier();
}

// The narrower-head path prefetches values while keys are scored. It retains
// that overlap where storing the batch in registers does not cost occupancy.
void history_keywise_value_load(const uint n, uint64_t rows, uint64_t row_stride, uint64_t pairs,
    uint64_t pair_stride, out uvec4 codes[HISTORY_VALUE_PIECES], out uint coefficients[HISTORY_PAIRS]) {
    const uint lane = SEISMIC_LANE;
    [[unroll]] for (uint i = 0u; i < HISTORY_VALUE_PIECES; ++i) {
        const uint item = i * 32u + lane;
        codes[i] = element_uvec4_at(rows + uint64_t(min(item / HISTORY_VALUE_PIECES, n - 1u)) * row_stride
            + uint64_t(item % HISTORY_VALUE_PIECES) * 16ul);
    }
    [[unroll]] for (uint i = 0u; i < HISTORY_PAIRS; ++i) {
        const uint item = i * 32u + lane;
        coefficients[i] = element_u32_at(pairs + uint64_t(min(item / HISTORY_PAIRS, n - 1u)) * pair_stride
            + uint64_t(item % HISTORY_PAIRS) * 4ul);
    }
}

void history_keywise_value_store(uvec4 codes[HISTORY_VALUE_PIECES], uint coefficients[HISTORY_PAIRS], uint tile) {
    const uint lane = SEISMIC_LANE;
    const uint pitch = HISTORY_STAGE_PITCH;
    subgroupBarrier();
    [[unroll]] for (uint i = 0u; i < HISTORY_VALUE_PIECES; ++i) {
        const uint item = i * 32u + lane;
        seismic_shared_uvec4[tile + (item / HISTORY_VALUE_PIECES) * pitch + item % HISTORY_VALUE_PIECES] = codes[i];
    }
    [[unroll]] for (uint i = 0u; i < HISTORY_PAIRS; ++i) {
        const uint item = i * 32u + lane;
        seismic_shared_u32[(tile + (item / HISTORY_PAIRS) * pitch + HISTORY_VALUE_PIECES) * 4u + item % HISTORY_PAIRS] =
            coefficients[i];
    }
    subgroupMemoryBarrierShared();
    subgroupBarrier();
}

// The value product of a batch of `n` keys from the tile: each lane decodes
// its E columns of key j (code * scale + zero) and accumulates them with the
// key's weights.
void history_keywise_values(const uint n, uint tile, uint weights, inout float result[ATTENTION_HEADS][ATTENTION_E]) {
    const uint lane = SEISMIC_LANE;
    const uint bits = ATTENTION_E * HISTORY_VALUE_BITS;
    for (uint j = 0u; j < n; ++j) {
        const uint row = (tile + j * HISTORY_STAGE_PITCH) * 4u;
        uint w[HISTORY_LANE_WORDS];
        [[unroll]] for (uint k = 0u; k < HISTORY_LANE_WORDS; ++k)
            w[k] = 0u;
        if (bits >= 32u) {
            [[unroll]] for (uint k = 0u; k < HISTORY_VALUE_WORDS; ++k)
                w[k] = seismic_shared_u32[row + lane * HISTORY_VALUE_WORDS + k];
        } else {
            w[0] = seismic_shared_u32[row + lane * bits / 32u] >> ((lane * bits) % 32u);
        }
        const vec2 sz = unpackHalf2x16(seismic_shared_u32[row + HISTORY_VALUE_PIECES * 4u + lane / HISTORY_PAIR_LANES]);
        float v[ATTENTION_E];
        [[unroll]] for (uint i = 0u; i < ATTENTION_E; ++i)
            v[i] = seismic_fma_rn(history_code(HISTORY_VALUE_BITS, w, i), sz.x, sz.y);
        attention_keywise_accumulate(weights, j, v, result);
    }
}

// Absorbs a batch of `n` (at most ATTENTION_KEYS) affine keys and values into
// the online state of the heads head0..: key j's code row at `keys` + j *
// `key_stride` bytes and coefficient pairs at `key_pairs` + j * `pair_stride`,
// its value's likewise. `tile` is the subgroup's staging tile (shared uvec4),
// `weights` its shared weights (`attention_keywise_softmax`).
void history_keywise_absorb(const uint n, uint64_t keys, uint64_t key_pairs, uint64_t key_stride, uint64_t values,
    uint64_t value_pairs, uint64_t value_stride, uint64_t pair_stride, uint tile, uint queries, uint sums, uint head0,
    uint weights, inout float maximum[ATTENTION_HEADS], inout float denominator[ATTENTION_HEADS],
    inout float result[ATTENTION_HEADS][ATTENTION_E]) {
    // At 512 columns the prefetched values occupy at least 144 registers;
    // shorter heads retain their overlap with scoring.
    if (ATTENTION_W < 512u) {
        uvec4 codes[HISTORY_VALUE_PIECES];
        uint coefficients[HISTORY_PAIRS];
        history_keywise_value_load(n, values, value_stride, value_pairs, pair_stride, codes, coefficients);
        float score[ATTENTION_HEADS];
        history_keywise_score(n, keys, key_stride, key_pairs + uint64_t(min(SEISMIC_LANE, n - 1u)) * pair_stride, tile,
            queries, sums, head0, score);
        attention_keywise_softmax(n, score, weights, maximum, denominator, result);
        history_keywise_value_store(codes, coefficients, tile);
    } else {
        float score[ATTENTION_HEADS];
        history_keywise_score(n, keys, key_stride, key_pairs + uint64_t(min(SEISMIC_LANE, n - 1u)) * pair_stride, tile,
            queries, sums, head0, score);
        attention_keywise_softmax(n, score, weights, maximum, denominator, result);
        history_keywise_value_stage(n, values, value_stride, value_pairs, pair_stride, tile);
    }
    history_keywise_values(n, tile, weights, result);
}

// ---------------------------------------------------------------------------
// The online-softmax state of a subgroup's ATTENTION_HEADS query heads absorbing `n` (at most
// HISTORY_BATCH) affine-coded keys and values, with this lane's group pairs.
// Scores are corrected, not decoded: the sum over groups of
// scale * (q . code) + zero * sum(q), each lane adding its columns' share
// (`qsum` is the sum of this lane's query columns). The value product
// accumulates (p * scale) * code into `result` and sum(p * zero) into this
// lane's per-head `bias`, both carried by the same factor, so result + bias is
// the attended sum.
#define HISTORY_BATCH 8u

void history_absorb_affine(const uint n, float q[ATTENTION_HEADS][ATTENTION_E], float qsum[ATTENTION_HEADS],
    uint key[HISTORY_BATCH][HISTORY_LANE_WORDS], vec2 key_coefficients[HISTORY_BATCH],
    uint value[HISTORY_BATCH][HISTORY_LANE_WORDS],
    vec2 value_coefficients[HISTORY_BATCH], inout float maximum[ATTENTION_HEADS],
    inout float denominator[ATTENTION_HEADS], inout float result[ATTENTION_HEADS][ATTENTION_E],
    inout float bias[ATTENTION_HEADS]) {
    float score[ATTENTION_HEADS][HISTORY_BATCH];
    [[unroll]] for (uint j = 0u; j < HISTORY_BATCH; ++j) {
        if (j < n) {
            float k[ATTENTION_E];
            [[unroll]] for (uint i = 0u; i < ATTENTION_E; ++i)
                k[i] = history_code(HISTORY_KEY_BITS, key[j], i);
            [[unroll]] for (uint g = 0u; g < ATTENTION_HEADS; ++g) {
                float partial = 0.0;
                [[unroll]] for (uint i = 0u; i < ATTENTION_E; ++i)
                    partial = seismic_fma_rn(q[g][i], k[i], partial);
                score[g][j] = seismic_subgroup_sum_f32(seismic_fma_rn(key_coefficients[j].x, partial,
                    key_coefficients[j].y * qsum[g]));
            }
        }
    }
    // Each value code's weight: its probability times the value scale.
    float weight[ATTENTION_HEADS][HISTORY_BATCH];
    [[unroll]] for (uint g = 0u; g < ATTENTION_HEADS; ++g) {
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
            [[unroll]] for (uint g = 0u; g < ATTENTION_HEADS; ++g)
                [[unroll]] for (uint i = 0u; i < ATTENTION_E; ++i)
                    result[g][i] = seismic_fma_rn(weight[g][j], v[i], result[g][i]);
        }
    }
}
