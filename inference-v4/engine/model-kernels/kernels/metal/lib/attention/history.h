// history shared attention mechanisms; included at the original declaration point.
// ---------------------------------------------------------------------------
// Affine K8/V4 history (`gated_attention_*_k8v4`). A (history row, kv head)
// vector is a code row of W * B / 32 u32 words (code i at bits B * (i % (32 /
// B)) of word i / (32 / B)) plus one F16 (scale, zero) pair per group of
// GROUP consecutive columns, pairs in group order; decoded value = code *
// scale + zero with its group's pair. A lane holding E columns of a vector
// owns E * B code bits: whole words when E * B >= 32, else a part of a word
// shared with its neighbours; its columns lie in one group, shared by
// GROUP / E neighbouring lanes.
// ---------------------------------------------------------------------------

#define ATTENTION_KEY_BITS 8
#define ATTENTION_VALUE_BITS 4
// Columns per (scale, zero) pair: the codec's group (model-state
// `AFFINE_GROUP`).
#define ATTENTION_GROUP 32

template <uint B>
struct lane_codes {
    // Compile-time constants (MSL admits no non-constant-space static data).
    enum : uint {
        bits = ATTENTION_E * B,
        words = (bits + 31) / 32,
        row_words = ATTENTION_W * B / 32,
        levels = (1u << B) - 1u,
        // (scale, zero) pairs per vector, and the lanes sharing one.
        pairs = ATTENTION_W / ATTENTION_GROUP,
        pair_lanes = ATTENTION_GROUP / ATTENTION_E,
    };
    static_assert(bits % 32 == 0 || 32 % bits == 0,
        "a lane's codes are whole words or a power-of-two part of one");
    static_assert(ATTENTION_W % ATTENTION_GROUP == 0 && ATTENTION_GROUP % ATTENTION_E == 0,
        "a lane's columns lie in one group");

    // The (scale, zero) pair of this lane's group of vector `vector`, whose
    // pairs start at `coefficients + vector * pairs * 2`.
    static inline float2 pair(device const half *coefficients, ulong vector, uint lane) {
        return float2(*reinterpret_cast<device const half2 *>(
            coefficients + (vector * pairs + lane / pair_lanes) * 2));
    }

    // This lane's codes of one vector's code row, shifted so its code i sits
    // at bits B * i of the (i * B / 32)-th word.
    static inline void load(device const uint *row, uint lane, thread uint (&w)[words]) {
        if (bits == 64) {
            const uint2 pair = *reinterpret_cast<device const uint2 *>(row + lane * 2);
            w[0] = pair.x;
            w[words - 1] = pair.y;
        } else if (bits >= 32) {
            ATTENTION_UNROLL
            for (uint j = 0; j < words; ++j)
                w[j] = row[lane * words + j];
        } else {
            w[0] = row[lane * bits / 32] >> ((lane * bits) % 32);
        }
    }

    // Code i of this lane's columns, as F32 (exact).
    static inline float code(thread const uint (&w)[words], uint i) {
        return float((w[i * B / 32] >> ((i * B) % 32)) & levels);
    }

    // Every code of this lane's columns, as F32 (exact). Whole words unpack
    // two codes at a time without a conversion instruction: a code c in the
    // mantissa of the half 1024 (0x6400 | c) is 1024 + c exactly, and one
    // half2 subtraction leaves c.
    static inline void unpack(thread const uint (&w)[words], thread float (&x)[ATTENTION_E]) {
        constexpr uint pair_mask = B == 8 ? 0x00FF00FFu : 0x000F000Fu;
        constexpr uint per_word = 32 / B;
        if (bits >= 32) {
            ATTENTION_UNROLL
            for (uint j = 0; j < words; ++j) {
                // Shift s of the word holds codes s and s + per_word / 2 in
                // its two halves.
                ATTENTION_UNROLL
                for (uint s = 0; s < per_word / 2; ++s) {
                    const half2 pair = as_type<half2>(((w[j] >> (s * B)) & pair_mask) | 0x64006400u)
                        - half2(1024.0h);
                    x[j * per_word + s] = float(pair.x);
                    x[j * per_word + s + per_word / 2] = float(pair.y);
                }
            }
        } else {
            ATTENTION_UNROLL
            for (uint i = 0; i < ATTENTION_E; ++i)
                x[i] = code(w, i);
        }
    }
};

// Encodes one vector held by a simdgroup (lane `lane` owns columns [lane * E,
// lane * E + E)) with B-bit codes at `row` (its code row) and its group
// coefficient pairs at `coefficients`: per group, zero = f16(min), scale =
// f16((max - min) / L), code = min(L, u32(fma(x - zero, 1 / scale, 0.5))), 0
// when scale is 0.
template <uint B>
inline void encode(thread const float (&x)[ATTENTION_E], device uint *row,
    device half *coefficients, uint lane) {
    typedef lane_codes<B> codes;
    float low = x[0];
    float high = x[0];
    ATTENTION_UNROLL
    for (uint i = 1; i < ATTENTION_E; ++i) {
        low = metal::min(low, x[i]);
        high = metal::max(high, x[i]);
    }
    // The group's range, over its pair_lanes neighbouring lanes.
    ATTENTION_UNROLL
    for (uint offset = 1; offset < codes::pair_lanes; offset *= 2) {
        low = metal::min(low, simd_shuffle_xor(low, ushort(offset)));
        high = metal::max(high, simd_shuffle_xor(high, ushort(offset)));
    }
    const half zero = half(low);
    const half scale = half((high - low) / float(codes::levels));
    const float inverse = float(scale) > 0.0f ? 1.0f / float(scale) : 0.0f;
    uint w[codes::words];
    ATTENTION_UNROLL
    for (uint j = 0; j < codes::words; ++j)
        w[j] = 0u;
    ATTENTION_UNROLL
    for (uint i = 0; i < ATTENTION_E; ++i) {
        const float t = metal::fma(x[i] - float(zero), inverse, 0.5f);
        const uint c = metal::min(uint(metal::max(t, 0.0f)), uint(codes::levels));
        w[i * B / 32] |= c << ((i * B) % 32);
    }
    if (codes::bits >= 32) {
        ATTENTION_UNROLL
        for (uint j = 0; j < codes::words; ++j)
            row[lane * codes::words + j] = w[j];
    } else {
        // 32 / bits neighbouring lanes share a word: join their parts.
        constexpr uint sharing = codes::bits >= 32 ? 1 : 32 / codes::bits;
        uint joined = w[0] << ((lane % sharing) * codes::bits);
        ATTENTION_UNROLL
        for (uint offset = 1; offset < sharing; offset *= 2)
            joined |= simd_shuffle_xor(joined, ushort(offset));
        if (lane % sharing == 0)
            row[lane / sharing] = joined;
    }
    if (lane % codes::pair_lanes == 0) {
        coefficients[lane / codes::pair_lanes * 2] = scale;
        coefficients[lane / codes::pair_lanes * 2 + 1] = zero;
    }
}

// The online-softmax state of G query heads absorbing N affine-coded keys and
// values, with this lane's group pairs. Scores are corrected, not decoded:
// the sum over groups of scale * (q . code) + zero * sum(q), each lane adding
// its columns' share (`qsum` is the sum of this lane's query columns). The
// value product accumulates (p * scale) * code into `output` and
// sum(p * zero) into this lane's per-head `bias`, both carried by the same
// factor, so output + bias is the attended sum.
template <uint N>
inline void absorb_affine(thread const float (&q)[SEISMIC_DIM_G][ATTENTION_E],
    thread const float (&qsum)[SEISMIC_DIM_G],
    thread const uint (&key)[N][lane_codes<ATTENTION_KEY_BITS>::words],
    thread const float2 (&key_coefficients)[N],
    thread const uint (&value)[N][lane_codes<ATTENTION_VALUE_BITS>::words],
    thread const float2 (&value_coefficients)[N],
    thread float (&maximum)[SEISMIC_DIM_G], thread float (&denominator)[SEISMIC_DIM_G],
    thread float (&output)[SEISMIC_DIM_G][ATTENTION_E], thread float (&bias)[SEISMIC_DIM_G]) {
    typedef lane_codes<ATTENTION_KEY_BITS> key_codes;
    typedef lane_codes<ATTENTION_VALUE_BITS> value_codes;
    constexpr uint G = SEISMIC_DIM_G;
    // Codes unpack one token at a time, so only one token's columns are live
    // as F32 next to the scores and the output.
    float score[G][N];
    ATTENTION_UNROLL
    for (uint j = 0; j < N; ++j) {
        float k[ATTENTION_E];
        key_codes::unpack(key[j], k);
        ATTENTION_UNROLL
        for (uint g = 0; g < G; ++g) {
            float partial = 0.0f;
            ATTENTION_UNROLL
            for (uint i = 0; i < ATTENTION_E; ++i)
                partial = metal::fma(q[g][i], k[i], partial);
            score[g][j] = simd_sum(metal::fma(key_coefficients[j].x, partial,
                key_coefficients[j].y * qsum[g]));
        }
    }
    // Each value code's weight: its probability times the value scale.
    float weight[G][N];
    ATTENTION_UNROLL
    for (uint g = 0; g < G; ++g) {
        float next = maximum[g];
        ATTENTION_UNROLL
        for (uint j = 0; j < N; ++j)
            next = metal::max(next, score[g][j]);
        const float carry = metal::fast::exp2(maximum[g] - next);
        float sum = 0.0f;
        float offset = bias[g] * carry;
        ATTENTION_UNROLL
        for (uint j = 0; j < N; ++j) {
            const float probability = metal::fast::exp2(score[g][j] - next);
            sum += probability;
            offset = metal::fma(probability, value_coefficients[j].y, offset);
            weight[g][j] = probability * value_coefficients[j].x;
        }
        denominator[g] = metal::fma(denominator[g], carry, sum);
        bias[g] = offset;
        maximum[g] = next;
        ATTENTION_UNROLL
        for (uint i = 0; i < ATTENTION_E; ++i)
            output[g][i] *= carry;
    }
    ATTENTION_UNROLL
    for (uint j = 0; j < N; ++j) {
        float v[ATTENTION_E];
        value_codes::unpack(value[j], v);
        ATTENTION_UNROLL
        for (uint g = 0; g < G; ++g)
            ATTENTION_UNROLL
            for (uint i = 0; i < ATTENTION_E; ++i)
                output[g][i] = metal::fma(weight[g][j], v[i], output[g][i]);
    }
}

