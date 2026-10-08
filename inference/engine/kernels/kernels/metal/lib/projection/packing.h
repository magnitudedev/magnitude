// The token-packing GEMM (an entry's PACK form, M > 64, on simdgroup
// matrices): two activation rows (tokens) share one F32 matrix operand, so one
// multiply-accumulate carries two products. Included after `projection.h`.
//
// Per (token, 32 columns) the activations are rounded to integer codes under
// a gain, and the operand element of a token pair is 2^16 top + low. The
// weights enter as their stored codes minus their centre, so a 32-column block of
// one weight row accumulates A = 2^16 S_top + S_low in an F32 accumulator;
// both sums are split off as 16-bit integers and fold into the output under
// (token scale) x (weight block scale), with the weights' min term as the
// product of the token's code sum and the block's bias.
//
// The fold runs in F16 on normalized factors, so it cannot overflow: a
// token's block scales are held over a power of two at or above its largest,
// a weight row's block factors over a power of two above every
// |scale| + 128 |bias| of the row (a block's code sum / 64 is below 128), so
// a block's term is at most 1 in magnitude and a result at most K / 32. The
// two powers of two multiply the F32 result once, at the store.
//
// The gain is the largest one that bounds every block sum for any row of the
// format (codes minus their centre c, 8 for Q4_K and q4g32s and 16 for Q5_K,
// have at most l1 32 c and l2 c sqrt(32) per block), by Hoelder and Cauchy-Schwarz
// with the rounding of the codes inside:
//   min((g |x|_2 + sqrt(32) / 2) c sqrt(32), (g |x|_inf + 1 / 2) 32 c) <= limit
// with limit 32767 for the top token and 32767 - 32 x 64 for the low one
// (the accumulator stays below 2^31, so each of a block's 32 roundings is at
// most 64). Q6_K scales every 16 columns: a block is two runs of 16, each
// with its own accumulation, sum pair and fold (c = 32, the bounds over 16
// columns), and no min term. The top sum is then exact for every input, and the low sum
// carries the accumulator's rounding: a token's result depends on the token
// it is packed with, so the form is an error class of its entry. A token is
// the top one in the first half of the blocks and the low one in the second.
//
// Two pre-passes write the form's scratch in the order the lanes read it:
// `packing_operand` the packed operand and the tokens' scales and code sums,
// `packing_coefficients` the weights' block scales and biases. Weight codes
// are read from the resident rows, one 4-byte word per weight row and run
// (and the lane's share of the plane of high bits, where the format has one).
//
// The weights are the left operand (inference/docs/kernels/lane-order.md): a
// lane at fragment coordinate (x, y) holds weight row y of a weight fragment
// and token pairs x, x + 1 of an operand fragment. A block is four 8-column
// steps, whose columns are the format's (`packing_codes`): each lane's two
// are a nibble of each half of one word of its row, and the operand is laid
// out to match.

namespace projection {

// The form's scratch, for `groups` = K / 32 blocks, tokens in tiles of 64
// (32 pairs: pair r is tokens 2 r, 2 r + 1; operand fragment r / 8):
//   operand  8 F32 per (token tile, 8-column step, lane): fragments 0 .. 3,
//            two pairs each; element (pair, c) = (2^16 top + low) 2^24 at the
//            lane of (x = pair % 8, y = c). One step past the last tile is
//            read ahead and never used.
//   tokens   4 F16 per (token tile, block, pair): 64 / gain of the pair's two
//            tokens over their token scales, then their code sums / 64; for
//            weights of two runs a block (which have no min) the gains are
//            a run's, the first run's pair then the second's
//   token_scales   1 F32 per token: the power of two its block scales are
//            held over
//   weights  2 F16 per (32-row weight tile, block, row % 8, row / 8):
//            d * scale6 * 32767 / 64 and centre d * scale6 - dmin * min6, over the
//            row's weight scale
//   weight_scales  1 F32 per weight row: the power of two its block factors
//            are held over
struct packing_scratch {
    device float *operand;
    device half *tokens;
    device float *token_scales;
    device half *weights;
    device float *weight_scales;
};

// The power of two above a non-negative `value` (the smallest normal one
// above zero).
inline float4 packing_scale(float4 value) {
    return as_type<float4>((as_type<uint4>(value) & 0x7f800000u) + 0x00800000u);
}

// Weights whose stored codes enter the packed product: the formats whose
// codes' low four bits are one plane (Q4_K and q4g32s; Q5_K and Q6_K with
// their plane of high bits), on a device without tensor operations (their int8
// products carry more than a packed matrix operand does). The codes enter
// minus `centre`, half their range. A block has `folds` runs of 32 / folds
// columns under one weight scale each (Q6_K scales 16 columns), a sum pair
// per run; `biased` formats have a min per block. A lane's `words` are the
// codes of its columns of one weight row and block, loaded from the lane's
// place `at` in that row's planes (the lane holds the two columns x, x + 1 of
// an 8-column step); `step` yields its two codes of step s as F16
// subnormals. `factors` gives a row's block factors of a 256-column block
// for `packing_coefficients`: `reach` a bound on what the fold multiplies,
// `start` and `take` each block's two factors in turn.
template <typename W>
struct packing_codes {
    static constant constexpr bool available = false;
    static constant constexpr uint centre = 0, folds = 0;
};
#if !SEISMIC_HAS_TENSOR_OPS
// q4k and q5k: a block's factors are d * scale6 * 32767 / 64 and
// centre d * scale6 - dmin * min6, and the fold takes a sum of at most 1 of
// the first and a code sum / 64 below 128 of the second, bounded from the
// super factors: |d * scale6| <= 63 |d| and |dmin * min6| <= 63 |dmin|.
template <uint CENTRE>
struct packing_k_factors {
    typedef packets::KBlock run;
    static float reach(device const uchar *row, Rows16 layout, uint block) {
        const float2 supers = metal::abs(float2(*reinterpret_cast<device const half2 *>(
            layout.super_at(row, 4ul * block))));
        return 63.0f * ((512.0f + 128.0f * float(CENTRE)) * supers.x + 128.0f * supers.y);
    }
    static run start(device const uchar *row, Rows16 layout, uint block) {
        return packets::KBlock::load(row, layout, block * 8u);
    }
    static float2 take(thread run &state) {
        // (d * scale6, -(dmin * min6)); the codes enter minus the centre.
        const float2 coefficient = packets::KBlock::take(state);
        return float2(coefficient.x * (32767.0f / 64.0f), float(CENTRE) * coefficient.x + coefficient.y);
    }
};
// The 4-bit codes of a plane of low nibbles alone (q4k, q4g32s): step s
// multiplies the block's columns 8 (c / 2) + s + 4 (c % 2) for c = 0 .. 7: a
// row's word c / 2 holds them as nibble s of each half.
template <typename W>
struct packing_nibbles {
    typedef uint place;
    typedef uint words;
    static place at(thread const Weights<W> &w, uint n, uint x) {
        return w.codes_at(n, 16u, 0u) + (x >> 1) * 4u;
    }
    // A row tile's rows of one block are adjacent: 16 bytes a row.
    static words load(thread const Weights<W> &w, place at, uint g) {
        return *reinterpret_cast<device const uint *>(w.base + at + g * 16u * w.layout.tile);
    }
    static half2 step(words codes, uint s) { return as_type<half2>((codes >> (4u * s)) & 0x000f000fu); }
};
template <>
struct packing_codes<packets::Q4K> {
    static constant constexpr bool available = true;
    static constant constexpr uint centre = 8, folds = 1;
    static constant constexpr bool biased = true;
    typedef packing_k_factors<8> factors;
    typedef packing_nibbles<packets::Q4K> nibbles;
    typedef nibbles::place place;
    typedef nibbles::words words;
    static place at(thread const Weights<packets::Q4K> &w, uint n, uint x) { return nibbles::at(w, n, x); }
    static words load(thread const Weights<packets::Q4K> &w, place at, uint g) { return nibbles::load(w, at, g); }
    static half2 step(words codes, uint s) { return nibbles::step(codes, s); }
};
// q4g32s: value = d * (code - 8) with one F16 d per 32 columns, so a block
// is one run and, the codes entering minus 8, has no min. Its factor is
// d * 32767 / 64 (the second is unused), and the fold takes a sum of at most
// 1 of it: 512 |d| bounds it.
struct packing_g32s_factors {
    // The scales of a 256-column block's packets from the next taken.
    typedef device const half *run;
    static run start(device const uchar *row, Rows16 layout, uint block) {
        return reinterpret_cast<device const half *>(layout.super_at(row, 16ul * block));
    }
    static float reach(device const uchar *row, Rows16 layout, uint block) {
        const run scales = start(row, layout, block);
        float largest = 0.0f;
        PROJECTION_UNROLL
        for (uint i = 0; i < 8; ++i)
            largest = metal::max(largest, metal::abs(float(scales[i])));
        return 512.0f * largest;
    }
    static float2 take(thread run &state) { return float2(float(*state++) * (32767.0f / 64.0f), 0.0f); }
};
template <>
struct packing_codes<packets::Q4G32S> {
    static constant constexpr bool available = true;
    static constant constexpr uint centre = 8, folds = 1;
    static constant constexpr bool biased = false;
    typedef packing_g32s_factors factors;
    typedef packing_nibbles<packets::Q4G32S> nibbles;
    typedef nibbles::place place;
    typedef nibbles::words words;
    static place at(thread const Weights<packets::Q4G32S> &w, uint n, uint x) { return nibbles::at(w, n, x); }
    static words load(thread const Weights<packets::Q4G32S> &w, place at, uint g) { return nibbles::load(w, at, g); }
    static half2 step(words codes, uint s) { return nibbles::step(codes, s); }
};
// q5k: q4k's columns; the lane's byte of the block's high bits beside its
// word of low nibbles, a nibble of the byte under each half of the word, so
// that bit s of each half is the high bit of step s's code there.
template <>
struct packing_codes<packets::Q5K> {
    static constant constexpr bool available = true;
    static constant constexpr uint centre = 16, folds = 1;
    static constant constexpr bool biased = true;
    typedef packing_k_factors<16> factors;
    typedef uint2 place;
    typedef uint2 words;
    static place at(thread const Weights<packets::Q5K> &w, uint n, uint x) {
        return uint2(w.codes_at(n, 16u, 0u) + (x >> 1) * 4u, w.high_at(n, 4u, 0u) + (x >> 1));
    }
    static words load(thread const Weights<packets::Q5K> &w, place at, uint g) {
        const uint high = w.base[at.y + g * 4u * w.layout.tile];
        return uint2(*reinterpret_cast<device const uint *>(w.base + at.x + g * 16u * w.layout.tile),
            (high | (high << 12)) & 0x000f000fu);
    }
    static half2 step(words codes, uint s) {
        return as_type<half2>(((codes.x >> (4u * s)) & 0x000f000fu) | (((codes.y >> s) & 0x00010001u) << 4));
    }
};
// q6k: value = d * scale8 * (code - 32) with a scale per 16 columns, so a
// block is two runs of two steps and has no min. Step 2 h + t multiplies the
// columns 16 h + 8 (c / 4) + 2 ((c / 2) % 2) + t + 4 (c % 2) of run h: a
// row's word 2 h + c / 4 holds them as nibble 2 ((c / 2) % 2) + t of each
// half, and a 16-bit field of the high plane their two high bits alike. A
// lane holds its word and field of each run, shifted to its first nibble.
// Its factors are d * scale8 * 32767 / 64 of the two runs.
struct packing_q6k_factors {
    typedef packets::Block<packets::Q6K>::state run;
    static float reach(device const uchar *row, Rows16 layout, uint block) {
        return 127.0f * 512.0f * metal::abs(float(*reinterpret_cast<device const half *>(
            layout.super_at(row, 2ul * block))));
    }
    static run start(device const uchar *row, Rows16 layout, uint block) {
        return packets::Block<packets::Q6K>::load(row, layout, block * 8u);
    }
    static float2 take(thread run &state) {
        const char2 scales = as_type<char2>(ushort(state.scales.x));
        packets::Block<packets::Q6K>::advance(state);
        return state.d * float2(scales) * (32767.0f / 64.0f);
    }
};
template <>
struct packing_codes<packets::Q6K> {
    static constant constexpr bool available = true;
    static constant constexpr uint centre = 32, folds = 2;
    static constant constexpr bool biased = false;
    typedef packing_q6k_factors factors;
    typedef uint3 place;
    typedef uint4 words;
    static place at(thread const Weights<packets::Q6K> &w, uint n, uint x) {
        return uint3(w.codes_at(n, 16u, 0u) + (x >> 2) * 4u, w.high_at(n, 8u, 0u) + (x >> 2) * 2u, (x & 2u) * 4u);
    }
    static words load(thread const Weights<packets::Q6K> &w, place at, uint g) {
        device const uchar *low = w.base + at.x + g * 16u * w.layout.tile;
        device const uchar *high = w.base + at.y + g * 8u * w.layout.tile;
        const uint2 nibbles = uint2(*reinterpret_cast<device const uint *>(low),
            *reinterpret_cast<device const uint *>(low + 8));
        const uint2 fields = uint2(*reinterpret_cast<device const ushort *>(high),
            *reinterpret_cast<device const ushort *>(high + 4));
        // A byte of each field (four columns' bits) under each half.
        return uint4(nibbles >> at.z, ((fields | (fields << 8)) & 0x00ff00ffu) >> (at.z / 2u));
    }
    static half2 step(words codes, uint s) {
        const uint nibbles = (s & 2u) ? codes.y : codes.x, fields = (s & 2u) ? codes.w : codes.z;
        const uint t = s & 1u;
        return as_type<half2>(((nibbles >> (4u * t)) & 0x000f000fu) | (((fields >> (2u * t)) & 0x00030003u) << 4));
    }
};
#endif

// Whether an operand laid out for blocks of FOLDS runs serves the weights W.
template <typename W, uint FOLDS>
struct packing_serves {
    static constant constexpr bool value = packing_codes<W>::available && packing_codes<W>::folds == FOLDS;
};

// The operand shared by the weights A .. D: laid out for the runs of the
// first of them with a packed path (`folds`; 0: none has one), and coded for
// the largest centre among those it serves, whose gain limits are the
// tightest.
template <typename A, typename B = A, typename C = A, typename D = A>
struct packing_shared {
    static constant constexpr uint folds = packing_codes<A>::folds != 0 ? packing_codes<A>::folds
        : packing_codes<B>::folds != 0 ? packing_codes<B>::folds
        : packing_codes<C>::folds != 0 ? packing_codes<C>::folds : packing_codes<D>::folds;
    static constant constexpr uint a = packing_serves<A, folds>::value ? packing_codes<A>::centre : 0;
    static constant constexpr uint b = packing_serves<B, folds>::value ? packing_codes<B>::centre : 0;
    static constant constexpr uint c = packing_serves<C, folds>::value ? packing_codes<C>::centre : 0;
    static constant constexpr uint d = packing_serves<D, folds>::value ? packing_codes<D>::centre : 0;
    static constant constexpr uint ab = a > b ? a : b, cd = c > d ? c : d;
    static constant constexpr uint centre = ab > cd ? ab : cd;
};

// The gain limits of a run of n = 32 / folds columns against codes minus
// `centre` (at most l1 n centre and l2 centre sqrt(n) per run): (l2,
// l-infinity) of the top token and of the low token,
// limit / (centre sqrt(n)) - sqrt(n) / 2 and limit / (n centre) - 1 / 2, with
// limit 32767 for the top token and 32767 - 64 n for the low one. Returned
// as what a run's scale (64 / gain) is per unit of each norm, 64 / limit
// over 1 - 2^-10, which covers the F32 rounding of the norms (at most 2^-19
// of a sum of 32 squares, in any order) and of the products with them.
inline float4 packing_limits(uint centre, uint folds) {
    const float n = 32.0f / float(folds), root = metal::sqrt(n);
    const float l2 = float(centre) * root, l1 = float(centre) * n, low = 32767.0f - 64.0f * n;
    return (64.0f / 0.9990234375f)
        / float4(32767.0f / l2 - 0.5f * root, 32767.0f / l1 - 0.5f, low / l2 - 0.5f * root, low / l1 - 0.5f);
}
// The largest top code: 2^16 code + low stays below 2^24.
constant constexpr float packing_top_codes = 255.0f;

// The scale of a run (32 / folds columns of one token) from its norms, the
// sum of its squares and its largest magnitude, under `limits`: 64 / gain,
// what the fold multiplies, with no division: each norm times its limit is
// the scale that norm admits.
inline float packing_run_scale(float squares, float peak, bool top, float4 limits) {
    const float2 limit = top ? limits.xy : limits.zw;
    // The largest gain either norm admits is the smaller scale.
    float scale = metal::min(metal::sqrt(squares) * limit.x, peak * limit.y);
    if (top)
        scale = metal::max(scale, peak * (64.0f / packing_top_codes));
    // The fold's scale is held in 16 bits (`packing_scale_bits`): its
    // fraction is rounded up to 8 bits, and the gain is the reciprocal of
    // what is held.
    return as_type<float>((as_type<uint>(scale) + 0x7fffu) & 0xffff8000u);
}
// The gain of a run under its `packing_run_scale`.
inline float packing_gain(float scale) { return scale > 0.0f ? 64.0f / scale : 0.0f; }

// One token's eight columns of a run as codes: its gain from the run's norms
// (32 / folds columns: four neighbouring lanes hold a block, two a run of
// 16) under `limits`, the codes, and to `factors` 64 / gain and the run's
// code sum / 64.
inline void packing_codes_of(float4 even, float4 odd, bool top, float4 limits, uint folds,
    thread float4 &even_codes, thread float4 &odd_codes, thread float2 &factors) {
    const float4 pairs = even * even + odd * odd;
    float squares = (pairs.x + pairs.y) + (pairs.z + pairs.w);
    const float4 peaks = metal::max(metal::abs(even), metal::abs(odd));
    float peak = metal::max(metal::max(peaks.x, peaks.y), metal::max(peaks.z, peaks.w));
    squares += simd_shuffle_xor(squares, ushort(1));
    peak = metal::max(peak, simd_shuffle_xor(peak, ushort(1)));
    if (folds == 1) {
        squares += simd_shuffle_xor(squares, ushort(2));
        peak = metal::max(peak, simd_shuffle_xor(peak, ushort(2)));
    }
    const float scale = packing_run_scale(squares, peak, top, limits);
    const float gain = packing_gain(scale);
    even_codes = metal::rint(even * gain);
    odd_codes = metal::rint(odd * gain);
    float sum = (even_codes.x + even_codes.y) + (even_codes.z + even_codes.w)
        + (odd_codes.x + odd_codes.y) + (odd_codes.z + odd_codes.w);
    sum += simd_shuffle_xor(sum, ushort(1));
    if (folds == 1)
        sum += simd_shuffle_xor(sum, ushort(2));
    factors = float2(scale, sum * 0x1p-6f);
}

// A block scale of `packing_codes_of` (non-negative, 8 fraction bits) as 16
// bits, and back.
inline ushort packing_scale_bits(float scale) { return ushort(as_type<uint>(scale) >> 15); }
inline float packing_scale_of(ushort bits) { return as_type<float>(uint(bits) << 15); }
// Two tokens' block scales, held as their bits, over the tokens' scales: a
// division by a power of two, exact but for a block 2^14 below the token's
// largest.
inline ushort2 packing_scales_over(ushort2 held, float2 scale) {
    return as_type<ushort2>(half2(float2(packing_scale_of(held.x), packing_scale_of(held.y)) / scale));
}

// The operand pass: one threadgroup of 256 threads per two token pairs (the
// two elements of an operand lane); a thread takes 8 columns of the four
// tokens at a time, four neighbouring lanes one 32-column block, and stores
// each column's two pairs together. Tokens at or past `m_rows` are zero.
// The block scales are first stored as their bits; once the threadgroup has
// the four tokens' largest (through `peaks`, one entry per simdgroup) their
// token scales are stored and the block scales rewritten over them. CENTRE
// and FOLDS are the centre and the runs a block of the weights the operand
// multiplies (`packing_shared`), whose steps' columns it is laid out in
// (`packing_codes`).
template <uint CENTRE, uint FOLDS, typename In>
inline void packing_operand(thread const In &in, thread const packing_scratch &s, uint m_rows, uint columns,
    uint pairs, threadgroup float4 *peaks, uint thread_index, uint sg, uint lane) {
    const uint groups = columns / 32u, steps = columns / 8u;
    const float4 limits = packing_limits(CENTRE, FOLDS);
    const uint tile = pairs / 16u, within = (2u * pairs) & 31u;
    device metal::vec<ushort, 4> *entries = reinterpret_cast<device metal::vec<ushort, 4> *>(s.tokens)
        + ulong(tile) * groups * 32u + within;
    float4 largest = float4(0.0f);
    for (uint chunk = thread_index; chunk < steps; chunk += 256u) {
        const uint g = chunk / 4u, word = chunk % 4u;
        // A pair's first token is the top one in the first half of the blocks.
        const bool first_top = 2u * g < groups;
        float4 even[4], odd[4];
        float2 factors[4];
        PROJECTION_UNROLL
        for (uint t = 0; t < 4; ++t) {
            float4 e = float4(0.0f), o = float4(0.0f);
            if (4u * pairs + t < m_rows)
                in.load8(4u * pairs + t, chunk * 8u, 1.0f, e, o);
            packing_codes_of(e, o, ((t & 1u) == 0) == first_top, limits, FOLDS, even[t], odd[t], factors[t]);
            largest[t] = metal::max(largest[t], factors[t].x);
        }
        float4 packed_even[2], packed_odd[2];
        PROJECTION_UNROLL
        for (uint p = 0; p < 2; ++p) {
            const uint top = 2u * p + (first_top ? 0u : 1u), low = 2u * p + (first_top ? 1u : 0u);
            packed_even[p] = (even[top] * 65536.0f + even[low]) * 0x1p24f;
            packed_odd[p] = (odd[top] * 65536.0f + odd[low]) * 0x1p24f;
            // A run's first lane stores its entry: the block's scales and
            // code sums, or with two runs a block each run's scales.
            if constexpr (FOLDS == 2) {
                if ((lane & 1u) == 0)
                    reinterpret_cast<device ushort2 *>(entries + g * 32u + p)[word >> 1] = ushort2(
                        packing_scale_bits(factors[2u * p].x), packing_scale_bits(factors[2u * p + 1u].x));
            } else {
                if ((lane & 3u) == 0)
                    entries[g * 32u + p] = metal::vec<ushort, 4>(packing_scale_bits(factors[2u * p].x),
                        packing_scale_bits(factors[2u * p + 1u].x), as_type<ushort>(half(factors[2u * p].y)),
                        as_type<ushort>(half(factors[2u * p + 1u].y)));
            }
        }
        // Column n + 4 h of the chunk is step n of the block, at lane row
        // 2 word + h; with two runs a block, step n % 2 of run word / 2, at
        // lane row 4 (word % 2) + 2 (n / 2) + h.
        device float2 *to = reinterpret_cast<device float2 *>(s.operand)
            + (ulong(tile) * steps + g * 4u) * 128u + (within >> 3);
        PROJECTION_UNROLL
        for (uint near = 0; near < 4; ++near) {
            const uint far = near + 4u;
            const uint step = FOLDS == 2 ? (word & 2u) + (near & 1u) : near;
            const uint row = FOLDS == 2 ? 4u * (word & 1u) + (near & 2u) : 2u * word;
            to[(step * 32u + fragment_lane(within & 6u, row)) * 4u] = float2(
                (near & 1u) ? packed_odd[0][near / 2u] : packed_even[0][near / 2u],
                (near & 1u) ? packed_odd[1][near / 2u] : packed_even[1][near / 2u]);
            to[(step * 32u + fragment_lane(within & 6u, row + 1u)) * 4u] = float2(
                (far & 1u) ? packed_odd[0][far / 2u] : packed_even[0][far / 2u],
                (far & 1u) ? packed_odd[1][far / 2u] : packed_even[1][far / 2u]);
        }
    }
    largest = simd_max(largest);
    if (lane == 0)
        peaks[sg] = largest;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    PROJECTION_UNROLL
    for (uint other = 0; other < 8; ++other)
        largest = metal::max(largest, peaks[other]);
    const float4 scale = packing_scale(largest);
    if (thread_index == 0)
        *reinterpret_cast<device float4 *>(s.token_scales + 4u * pairs) = scale;
    if ((lane & (FOLDS == 2 ? 1u : 3u)) != 0)
        return;
    for (uint chunk = thread_index; chunk < steps; chunk += 256u) {
        PROJECTION_UNROLL
        for (uint p = 0; p < 2; ++p) {
            // The scales this lane stored: the entry's first pair, or with
            // two runs a block its run's pair.
            device ushort2 *entry = reinterpret_cast<device ushort2 *>(entries + (chunk / 4u) * 32u + p)
                + (FOLDS == 2 ? (chunk % 4u) >> 1 : 0u);
            *entry = packing_scales_over(*entry, float2(scale[2u * p], scale[2u * p + 1u]));
        }
    }
}

// The coefficient pass: one simdgroup per weight row, a lane per 256-column
// block. The row's weight scale is above what the fold multiplies of every
// block (the format's `reach`). Weights without a packed path have none.
template <typename W>
inline void packing_coefficients(thread const Weights<W> &w, thread const packing_scratch &s, uint rows,
    uint columns, uint n, uint lane) {
    if constexpr (packing_codes<W>::available) {
        if (n >= rows)
            return;
        typedef typename packing_codes<W>::factors coded;
        const uint groups = columns / 32u, blocks = columns / 256u;
        const uint tile = n / 32u, within = n % 32u;
        device const uchar *row = w.row(n);
        const Rows16 layout = w.geometry(n);
        float largest = 0.0f;
        for (uint block = lane; block < blocks; block += 32u)
            largest = metal::max(largest, coded::reach(row, layout, block));
        const float scale = packing_scale(float4(simd_max(largest))).x;
        if (lane == 0)
            s.weight_scales[n] = scale;
        const float inverse = 1.0f / scale;
        for (uint block = lane; block < blocks; block += 32u) {
            typename coded::run run = coded::start(row, layout, block);
            PROJECTION_UNROLL
            for (uint i = 0; i < 8; ++i)
                reinterpret_cast<device half2 *>(s.weights)[((ulong(tile) * groups + block * 8u + i) * 8u
                    + (within & 7u)) * 4u + (within >> 3)] = half2(coded::take(run) * inverse);
        }
    }
}

// The coefficient pass over whole 32-row weight tiles (`rows` a multiple of
// 32): one threadgroup of 256 threads per tile, a thread per (row, 256-column
// block) at a time, and one weight scale for the tile's rows (through
// `peaks`, one entry per simdgroup). Every thread decodes, where a simdgroup
// per row leaves most lanes idle; a row keeps the fold's precision down to
// factors 2^-14 of its tile's largest.
template <typename W>
inline void packing_tile_coefficients(thread const Weights<W> &w, thread const packing_scratch &s,
    uint columns, uint tile, threadgroup float *peaks, uint thread_index, uint sg, uint lane) {
    if constexpr (packing_codes<W>::available) {
        typedef typename packing_codes<W>::factors coded;
        const uint groups = columns / 32u, items = 32u * (columns / 256u);
        float largest = 0.0f;
        for (uint item = thread_index; item < items; item += 256u) {
            const uint n = tile * 32u + (item & 31u);
            largest = metal::max(largest, coded::reach(w.row(n), w.geometry(n), item >> 5));
        }
        largest = simd_max(largest);
        if (lane == 0)
            peaks[sg] = largest;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        PROJECTION_UNROLL
        for (uint other = 0; other < 8; ++other)
            largest = metal::max(largest, peaks[other]);
        const float scale = packing_scale(float4(largest)).x;
        if (thread_index < 32u)
            s.weight_scales[tile * 32u + thread_index] = scale;
        const float inverse = 1.0f / scale;
        for (uint item = thread_index; item < items; item += 256u) {
            const uint within = item & 31u, block = item >> 5, n = tile * 32u + within;
            typename coded::run run = coded::start(w.row(n), w.geometry(n), block);
            PROJECTION_UNROLL
            for (uint i = 0; i < 8; ++i)
                reinterpret_cast<device half2 *>(s.weights)[((ulong(tile) * groups + block * 8u + i) * 8u
                    + (within & 7u)) * 4u + (within >> 3)] = half2(coded::take(run) * inverse);
        }
    }
}

// Simdgroups of a PACK tile launch's threadgroup, each an independent tile:
// the best count on every device measured (1 to 4 on the M4 Pro and M1).
constant constexpr uint packing_simdgroups = 4;

// A simdgroup's fragments: TOKENS operand fragments (16 tokens each) against
// 8 / TOKENS weight fragments (8 weight rows each), 4 x 2 or 2 x 4. The best
// shape differs by device, so it is the launch's.
template <uint TOKENS>
struct packing_tile {
    static_assert(TOKENS == 4 || TOKENS == 2, "a packed tile is 4 x 2 or 2 x 4 fragments");
    static constant constexpr uint ti = TOKENS, tj = 8 / TOKENS;
    static constant constexpr uint tokens = 16 * ti, rows = 8 * tj;
    typedef float4 sums[ti][tj];
};

// One simdgroup's tile of `packing_tile<TOKENS>`: tokens from m0 (a multiple
// of the tile's tokens) by weight rows from n0 (a multiple of its rows).
// `y[i][j]` receives the sums of weight row n0 + 8 j + y and tokens
// m0 + 2 (8 i + x) .. + 3 (K below 2^20: a normalized sum of at most two
// terms a block stays in F16).
// WEIGHTS_AHEAD loads a block's weight words while the block before it
// multiplies (whether that helps differs by device and shape); a step's
// operand is always loaded while the step before it multiplies.
template <typename W, uint TOKENS, uint WEIGHTS_AHEAD>
inline void gemm_packed_sums(thread const Weights<W> &w, thread const packing_scratch &s, uint k, uint m0,
    uint n0, uint lane, thread typename packing_tile<TOKENS>::sums &y) {
    constexpr uint TI = packing_tile<TOKENS>::ti, TJ = packing_tile<TOKENS>::tj;
    const uint groups = k / 32u;
    const ushort2 at = fragment_coordinate(lane);
    // The lane's positions as 32-bit offsets from the bound buffers: a
    // per-lane pointer is a 64-bit register held through the loop.
    typedef packing_codes<W> coded;
    typename coded::place codes[TJ];
    PROJECTION_UNROLL
    for (uint j = 0; j < TJ; ++j)
        codes[j] = coded::at(w, n0 + 8u * j + at.y, at.x);
    // Steps a run: a run's first restarts the accumulators and its last
    // folds them.
    constexpr uint RUN = 4 / coded::folds;
    device const float4 *operand = reinterpret_cast<device const float4 *>(s.operand);
    device const uint4 *tokens = reinterpret_cast<device const uint4 *>(s.tokens);
    device const uint2 *weights = reinterpret_cast<device const uint2 *>(s.weights);
    // The tile's first operand fragment of its 64-token tile, and its first
    // 8-row group of its 32-row weight tile.
    const uint tile_m = m0 / 64u, fragment0 = (m0 % 64u) / 16u, rows0 = (n0 % 32u) / 8u;
    const uint operand_at = (tile_m * (k / 8u) * 32u + lane) * 2u + fragment0 / 2u;
    const uint tokens_at = tile_m * groups * 16u + 4u * fragment0 + (at.x >> 1);
    const uint weights_at = ((n0 / 32u) * groups * 8u + at.y) * 2u + rows0 / 2u;
    // The centre as a code: the stored codes are F16 subnormals of the same
    // exponent (the operand carries the 2^24), which the matrix unit
    // multiplies exactly.
    const half centre = as_type<half>(ushort(coded::centre));
    simdgroup_float8x8 acc[TI][TJ];
    half4 folded[TI][TJ];
    PROJECTION_UNROLL
    for (uint i = 0; i < TI; ++i) {
        PROJECTION_UNROLL
        for (uint j = 0; j < TJ; ++j)
            folded[i][j] = half4(0.0h);
    }
    float4 ahead[TI / 2];
    PROJECTION_UNROLL
    for (uint h = 0; h < TI / 2; ++h)
        ahead[h] = operand[operand_at + h];
    typename coded::words next[TJ];
    if constexpr (WEIGHTS_AHEAD != 0) {
        PROJECTION_UNROLL
        for (uint j = 0; j < TJ; ++j)
            next[j] = coded::load(w, codes[j], 0u);
    }
    // One copy of the block loop per half: which token of a pair is the top
    // one is then a constant of the copy.
    PROJECTION_UNROLL
    for (uint phase = 0; phase < 2; ++phase) {
        const bool swapped = phase == 1;
        for (uint local = 0; local < groups / 2u; ++local) {
            const uint g = phase * (groups / 2u) + local;
            typename coded::words words[TJ];
            PROJECTION_UNROLL
            for (uint j = 0; j < TJ; ++j) {
                if constexpr (WEIGHTS_AHEAD != 0) {
                    words[j] = next[j];
                    next[j] = coded::load(w, codes[j], metal::min(g + 1u, groups - 1u));
                } else {
                    words[j] = coded::load(w, codes[j], g);
                }
            }
            uint4 token[TI];
            PROJECTION_UNROLL
            for (uint i = 0; i < TI; ++i)
                token[i] = tokens[tokens_at + g * 16u + 4u * i];
            uint2 weight[TJ / 2];
            PROJECTION_UNROLL
            for (uint q = 0; q < TJ / 2; ++q)
                weight[q] = weights[weights_at + g * 16u + q];
            PROJECTION_UNROLL
            for (uint step = 0; step < 4; ++step) {
                simdgroup_float8x8 a[TI];
                simdgroup_half8x8 b[TJ];
                PROJECTION_UNROLL
                for (uint h = 0; h < TI / 2; ++h) {
                    const float4 current = ahead[h];
                    ahead[h] = operand[operand_at + (g * 4u + step + 1u) * 64u + h];
                    reinterpret_cast<thread float2 &>(a[2u * h].thread_elements()) = current.xy;
                    reinterpret_cast<thread float2 &>(a[2u * h + 1u].thread_elements()) = current.zw;
                }
                PROJECTION_UNROLL
                for (uint j = 0; j < TJ; ++j)
                    reinterpret_cast<thread half2 &>(b[j].thread_elements()) =
                        coded::step(words[j], step) - centre;
                PROJECTION_UNROLL
                for (uint i = 0; i < TI; ++i) {
                    PROJECTION_UNROLL
                    for (uint jj = 0; jj < TJ; ++jj) {
                        const uint j = (i & 1u) ? TJ - 1u - jj : jj;
                        if (step % RUN == 0)
                            simdgroup_multiply(acc[i][j], b[j], a[i]);
                        else
                            simdgroup_multiply_accumulate(acc[i][j], b[j], a[i], acc[i][j]);
                    }
                }
                if ((step + 1u) % RUN != 0)
                    continue;
                PROJECTION_UNROLL
                for (uint i = 0; i < TI; ++i) {
                    const half4 first = as_type<half4>(token[i].xy), second = as_type<half4>(token[i].zw);
                    // The tokens' scales of the block, then their code sums
                    // or their scales of its second run.
                    const half4 sums = half4(first.zw, second.zw);
                    const half4 scales = step / RUN == 0 ? half4(first.xy, second.xy) : sums;
                    PROJECTION_UNROLL
                    for (uint j = 0; j < TJ; ++j) {
                        const half2 factors = as_type<half2>(weight[j / 2u][j & 1u]);
                        const float2 v = reinterpret_cast<thread float2 &>(acc[i][j].thread_elements());
                        // Each element's (low, top) sums as two's complement
                        // 16-bit halves of one word.
                        const uint2 split = (uint2(int2(v)) + uint2(0x8000u)) ^ uint2(0x8000u);
                        const half2 e0 = unpack_snorm2x16_to_half(split.x), e1 = unpack_snorm2x16_to_half(split.y);
                        const half4 sum = swapped ? half4(e0.x, e0.y, e1.x, e1.y) : half4(e0.y, e0.x, e1.y, e1.x);
                        // A term of at most 1 (normalized factors): the
                        // block's scale and bias, or the run's scale.
                        half4 term;
                        if constexpr (coded::biased)
                            term = metal::fma(sum, half4(factors.x), sums * factors.y);
                        else
                            term = sum * half4(factors[step / RUN]);
                        folded[i][j] = metal::fma(scales, term, folded[i][j]);
                    }
                }
            }
        }
    }
    // The tokens' and the weight rows' scales, once.
    PROJECTION_UNROLL
    for (uint i = 0; i < TI; ++i) {
        const float4 token_scale = *reinterpret_cast<device const float4 *>(
            s.token_scales + m0 + 2u * (8u * i + at.x));
        PROJECTION_UNROLL
        for (uint j = 0; j < TJ; ++j)
            y[i][j] = float4(folded[i][j]) * token_scale * s.weight_scales[n0 + 8u * j + at.y];
    }
}

// Tile `index` of the `packing_tile<TOKENS>` tiles over `padded` tokens (a
// multiple of 64) by `rows` weight rows: tokens first, so neighbouring tiles
// share their weight rows. (m0, n0); n0 is at or past `rows` beyond the last
// tile.
template <uint TOKENS>
inline uint2 packing_tile_origin(uint index, uint padded) {
    const uint across = padded / packing_tile<TOKENS>::tokens;
    return uint2((index % across) * packing_tile<TOKENS>::tokens, (index / across) * packing_tile<TOKENS>::rows);
}

// The PACK launch: every simdgroup one `packing_tile<TOKENS>`, tile
// threadgroup x simdgroups + sg of the tiles over `padded` tokens and `rows`
// weight rows (a multiple of 32). The simdgroups of a threadgroup are
// independent: their number sets only how many tiles a core runs together,
// whose best differs by device.
template <typename W, uint TOKENS, uint WEIGHTS_AHEAD, typename Out>
inline void gemm_packed(thread const Out &out, thread const Weights<W> &w, thread const packing_scratch &s,
    uint m_rows, uint padded, uint rows, uint k, uint index, uint lane) {
    typedef packing_tile<TOKENS> tile;
    const uint2 origin = packing_tile_origin<TOKENS>(index, padded);
    const uint m0 = origin.x, n0 = origin.y;
    if (m0 >= m_rows || n0 >= rows)
        return;
    const ushort2 at = fragment_coordinate(lane);
    typename tile::sums y;
    // Weights without a packed path have no tile (their launch returns
    // before it).
    if constexpr (packing_codes<W>::available)
        gemm_packed_sums<W, TOKENS, WEIGHTS_AHEAD>(w, s, k, m0, n0, lane, y);
    // A lane holds four tokens of each operand fragment, a weight row of
    // each weight fragment: a token's result and residual rows are resolved
    // once.
    PROJECTION_UNROLL
    for (uint i = 0; i < tile::ti; ++i) {
        PROJECTION_UNROLL
        for (uint t = 0; t < 4; ++t) {
            const uint m = m0 + 2u * (8u * i + at.x) + t;
            if (m < m_rows) {
                const auto row = out.row(m);
                PROJECTION_UNROLL
                for (uint j = 0; j < tile::tj; ++j)
                    row.store(n0 + 8u * j + at.y, y[i][j][t]);
            }
        }
    }
}

// One segment of a segmented entry's PACK launch: the segment's `rows` weight
// rows (a multiple of 32) take ceil(padded rows / 1024 / simdgroups)
// threadgroups of the launch, in segment order. A threadgroup `group` at or
// past them belongs to a later segment, whose first is then `group` 0;
// weights the operand (laid out for FOLDS runs a block) does not serve leave
// theirs empty (the entry's exact tiles run that segment).
template <typename W, uint FOLDS, uint TOKENS, uint WEIGHTS_AHEAD, typename Out>
inline bool gemm_packed_segment(thread const Out &out, thread const Weights<W> &w, thread const packing_scratch &s,
    uint m_rows, uint padded, uint rows, uint k, thread uint &group, uint simdgroups, uint sg, uint lane) {
    const uint groups = (padded / 32u * (rows / 32u) + simdgroups - 1u) / simdgroups;
    if (group >= groups) {
        group -= groups;
        return false;
    }
    if constexpr (packing_serves<W, FOLDS>::value)
        gemm_packed<W, TOKENS, WEIGHTS_AHEAD>(out, w, s, m_rows, padded, rows, k,
            group * simdgroups + sg, lane);
    return true;
}

// F32 quadruples of `exchange` in the paired PACK launch per gate simdgroup:
// eight per lane.
constant constexpr uint packing_exchange = 32 * 8;

// The paired PACK launch (gate and up weights over one operand): simdgroups
// 2 p and 2 p + 1 share tile threadgroup x (simdgroups / 2) + p; the even one
// sums the gate weights' products, the odd one the up weights' and stores
// the pair with the gate sums, exchanged through `exchange`
// (`packing_exchange` quadruples per pair). `gate_scratch` and `up_scratch`
// share the operand and differ in the weights' factors.
template <typename G, typename U, uint TOKENS, uint WEIGHTS_AHEAD, typename Out>
inline void gemm_packed_paired(thread const Out &out, thread const Weights<G> &gate, thread const Weights<U> &up,
    thread const packing_scratch &gate_scratch, thread const packing_scratch &up_scratch, uint m_rows,
    uint padded, uint rows, uint k, uint threadgroup_index, uint simdgroups, threadgroup float4 *exchange, uint sg,
    uint lane) {
    typedef packing_tile<TOKENS> tile;
    const uint pair = sg / 2u;
    const uint2 origin = packing_tile_origin<TOKENS>(threadgroup_index * (simdgroups / 2u) + pair, padded);
    const uint m0 = origin.x, n0 = origin.y;
    const bool live = m0 < m_rows && n0 < rows;
    const ushort2 at = fragment_coordinate(lane);
    const bool second = (sg % 2u) != 0;
    typename tile::sums y;
    if constexpr (packing_codes<G>::available && packing_serves<U, packing_codes<G>::folds>::value) {
        if (live) {
            if (second)
                gemm_packed_sums<U, TOKENS, WEIGHTS_AHEAD>(up, up_scratch, k, m0, n0, lane, y);
            else
                gemm_packed_sums<G, TOKENS, WEIGHTS_AHEAD>(gate, gate_scratch, k, m0, n0, lane, y);
        }
    }
    threadgroup float4 *slot = exchange + (pair * 32u + lane) * 8u;
    if (live && !second) {
        PROJECTION_UNROLL
        for (uint i = 0; i < tile::ti; ++i) {
            PROJECTION_UNROLL
            for (uint j = 0; j < tile::tj; ++j)
                slot[i * tile::tj + j] = y[i][j];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!live || !second)
        return;
    PROJECTION_UNROLL
    for (uint i = 0; i < tile::ti; ++i) {
        PROJECTION_UNROLL
        for (uint t = 0; t < 4; ++t) {
            const uint m = m0 + 2u * (8u * i + at.x) + t;
            if (m < m_rows) {
                PROJECTION_UNROLL
                for (uint j = 0; j < tile::tj; ++j)
                    out.store_pair(m, n0 + 8u * j + at.y, slot[i * tile::tj + j][t], y[i][j][t]);
            }
        }
    }
}

} // namespace projection
