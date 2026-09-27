// history shared attention mechanisms; included at the original declaration point.
// ---------------------------------------------------------------------------
// History planes. Each policy appends a row's prepared key and raw value at
// its destination (one warp, the caller skips rows without one) and locates a
// (history row, kv head) vector in its planes. An entry source builds its
// policy from its own ABI (`HISTORY()`): a header may name only macros every
// including entry generates.

// Dense history: key and value planes [T, KV, W] of the activation element.
struct DenseHistory {
    static constexpr bool CODED = false;
    u8 *key;
    u8 *value;
    u64 key_row, key_head, value_row, value_head;

    // Element offsets of a (history row, kv head) vector.
    __device__ __forceinline__ u64 key_at(int token, int kv_head) const {
        return static_cast<u64>(token) * key_row + static_cast<u64>(kv_head) * key_head;
    }
    __device__ __forceinline__ u64 value_at(int token, int kv_head) const {
        return static_cast<u64>(token) * value_row + static_cast<u64>(kv_head) * value_head;
    }

    __device__ __forceinline__ void append(int destination, int kv_head, const float (&k)[DPL],
                                           const float (&v)[DPL], int lane) const {
        const u64 k_at = key_at(destination, kv_head) + lane * DPL;
        const u64 v_at = value_at(destination, kv_head) + lane * DPL;
#pragma unroll
        for (int d = 0; d < DPL; ++d) {
            element::put<Act>(key, k_at + d, k[d]);
            element::put<Act>(value, v_at + d, v[d]);
        }
    }
};

// Affine K8/V4 history (the `affine_k8_uniform_v4` codec,
// `gated_attention_*_k8v4`). A (history row, kv head) vector is a code row of
// W * B / 32 u32 words (code i at bits B * (i % (32 / B)) of word
// i / (32 / B)) plus one F16 (scale, zero) pair per group of GROUP
// consecutive dimensions, pairs in group order; decoded value =
// code * scale + zero with its group's pair. Keys use B = 8, values B = 4. A
// lane's DPL dimensions lie in one group, shared by PAIR_LANES lanes.
constexpr int KEY_BITS = 8;
constexpr int VALUE_BITS = 4;
// Dimensions per (scale, zero) pair: the codec's group (model-state
// `AFFINE_GROUP`), and the pairs per vector.
constexpr int GROUP = 32;
constexpr int PAIRS = W / GROUP;
constexpr int PAIR_LANES = GROUP / DPL;
static_assert(W % GROUP == 0 && GROUP % DPL == 0, "a lane's dimensions lie in one group");

// A lane's share of one code row: its DPL codes are DPL * B bits, whole
// words or a power-of-two part of one word shared with its neighbours.
template <int B> struct LaneCodes {
    static constexpr int bits = DPL * B;
    static constexpr int words = (bits + 31) / 32;
    static constexpr int row_words = W * B / 32;
    static constexpr u32 levels = (1u << B) - 1u;
    static_assert(bits % 32 == 0 || 32 % bits == 0,
                  "a lane's codes are whole words or a power-of-two part of one");

    // This lane's codes of a code row, shifted so its code i sits at bits
    // B * i of word i * B / 32.
    __device__ static __forceinline__ void load(const u32 *row, int lane, u32 (&w)[words]) {
        if constexpr (bits == 64) {
            const uint2 pair = *reinterpret_cast<const uint2 *>(row + lane * 2);
            w[0] = pair.x;
            w[1] = pair.y;
        } else if constexpr (bits >= 32) {
#pragma unroll
            for (int j = 0; j < words; ++j) w[j] = row[lane * words + j];
        } else {
            w[0] = row[lane * bits / 32] >> ((lane * bits) % 32);
        }
    }

    // Code i of this lane's dimensions, as F32 (exact): the code in the
    // mantissa of 2^23, minus 2^23 (no conversion instruction).
    __device__ static __forceinline__ float code(const u32 (&w)[words], int i) {
        return __uint_as_float(0x4B000000u | ((w[i * B / 32] >> ((i * B) % 32)) & levels)) -
               8388608.0f;
    }
};

// Encodes one vector held by a warp (lane `lane` owns dimensions
// [lane * DPL, (lane + 1) * DPL)) as B-bit codes at `row` (its code row) and
// its group (scale, zero) pairs at `pairs`: per group, zero = f16(min),
// scale = f16((max - min) / L), code = min(L, u32(fma(x - zero, 1 / scale,
// 0.5))), 0 when scale is 0.
template <int B>
__device__ __forceinline__ void encode(const float (&x)[DPL], u32 *row, u32 *pairs, int lane) {
    typedef LaneCodes<B> Codes;
    float low = x[0];
    float high = x[0];
#pragma unroll
    for (int d = 1; d < DPL; ++d) {
        low = fminf(low, x[d]);
        high = fmaxf(high, x[d]);
    }
    // The group's range, over its PAIR_LANES neighbouring lanes.
#pragma unroll
    for (int offset = 1; offset < PAIR_LANES; offset *= 2) {
        low = fminf(low, seismic_shfl_xor_f32(low, offset));
        high = fmaxf(high, seismic_shfl_xor_f32(high, offset));
    }
    const u16 zero_bits = seismic_f32_to_f16(low);
    const u16 scale_bits = seismic_f32_to_f16((high - low) / static_cast<float>(Codes::levels));
    const float zero = seismic_f16_to_f32(zero_bits);
    const float scale = seismic_f16_to_f32(scale_bits);
    const float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
    u32 w[Codes::words];
#pragma unroll
    for (int j = 0; j < Codes::words; ++j) w[j] = 0u;
#pragma unroll
    for (int d = 0; d < DPL; ++d) {
        const float t = __fmaf_rn(x[d] - zero, inverse, 0.5f);
        const u32 c = min(static_cast<u32>(fmaxf(t, 0.0f)), Codes::levels);
        w[d * B / 32] |= c << ((d * B) % 32);
    }
    if constexpr (Codes::bits >= 32) {
#pragma unroll
        for (int j = 0; j < Codes::words; ++j) row[lane * Codes::words + j] = w[j];
    } else {
        // 32 / bits neighbouring lanes share a word: join their parts.
        constexpr int sharing = 32 / Codes::bits;
        u32 joined = w[0] << ((lane % sharing) * Codes::bits);
#pragma unroll
        for (int offset = 1; offset < sharing; offset *= 2)
            joined |= seismic_shfl_xor_u32(joined, offset);
        if (lane % sharing == 0) row[lane / sharing] = joined;
    }
    if (lane % PAIR_LANES == 0)
        pairs[lane / PAIR_LANES] = static_cast<u32>(scale_bits) | (static_cast<u32>(zero_bits) << 16);
}

// A (scale, zero) pair, as F32.
__device__ __forceinline__ float2 coefficients(const u32 *pair) {
    return seismic_unpack_f16x2(*pair);
}

// Affine history: code planes [T, KV, W * B / 32] u32 and (scale, zero)
// planes [T, KV, 2 * PAIRS] f16 per vector kind. Every plane is canonical: a
// vector's code row and its pairs are contiguous.
struct AffineHistory {
    static constexpr bool CODED = true;
    u32 *key_codes;
    u32 *key_pairs;
    u32 *value_codes;
    u32 *value_pairs;
    u64 key_codes_row, key_codes_head, key_pairs_row, key_pairs_head;
    u64 value_codes_row, value_codes_head, value_pairs_row, value_pairs_head;

    // A vector's code row and (scale, zero) pairs (one u32 each).
    __device__ __forceinline__ u32 *key_row(int token, int kv_head) const {
        return key_codes + static_cast<u64>(token) * key_codes_row +
               static_cast<u64>(kv_head) * key_codes_head;
    }
    __device__ __forceinline__ u32 *key_pair(int token, int kv_head) const {
        return key_pairs + (static_cast<u64>(token) * key_pairs_row +
                            static_cast<u64>(kv_head) * key_pairs_head) / 2;
    }
    __device__ __forceinline__ u32 *value_row(int token, int kv_head) const {
        return value_codes + static_cast<u64>(token) * value_codes_row +
               static_cast<u64>(kv_head) * value_codes_head;
    }
    __device__ __forceinline__ u32 *value_pair(int token, int kv_head) const {
        return value_pairs + (static_cast<u64>(token) * value_pairs_row +
                              static_cast<u64>(kv_head) * value_pairs_head) / 2;
    }

    // The key is the prepared key rounded to the activation element (as `k`
    // holds it), the value the projected value.
    __device__ __forceinline__ void append(int destination, int kv_head, const float (&k)[DPL],
                                           const float (&v)[DPL], int lane) const {
        encode<KEY_BITS>(k, key_row(destination, kv_head), key_pair(destination, kv_head), lane);
        encode<VALUE_BITS>(v, value_row(destination, kv_head), value_pair(destination, kv_head),
                           lane);
    }
};

// Append row `row`'s prepared key `k` and raw value for `kv_head` at its
// destination through `history`, when it has one. One warp.
template <class History>
__device__ __forceinline__ void append(const Inputs &in, const History &history, u64 row,
                                       int kv_head, const float (&k)[DPL], int lane) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    const int destination = ATTENTION_DESTINATION(in, row);
    if (destination < 0) return;
    float v[DPL];
    fresh_value(in, row, kv_head, v, lane);
    history.append(destination, kv_head, k, v, lane);
}

