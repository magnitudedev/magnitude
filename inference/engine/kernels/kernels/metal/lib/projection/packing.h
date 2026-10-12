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
// format. Codes minus their centre c (8 for Q4_K and q4g32s, 16 for Q5_K, 32
// for Q6_K, 128 for 8-bit values) are at most c in magnitude, so a sum is at
// most c |q|_1 (Hoelder; the exact worst case over the format's rows, which
// Cauchy-Schwarz and the l-infinity bound only exceed), and codes q rounded
// from x under gain g to multiples of s have |q|_1 <= g |x|_1 + 32 s / 2:
//   (g |x|_1 + 16 s) c <= limit
// with limit 32767 for the top token and 32767 - 32 x 64 for the low one
// (the accumulator stays below 2^31, so each of a block's 32 roundings is at
// most 64). The weights carry their packing bound (`packing_bounds`: per run
// of n columns B^2, the largest squared l2 norm of a row's codes minus their
// centre over the tensor's rows, made on the device when the weight becomes
// resident), and
// Cauchy-Schwarz against it certifies a gain of its own,
// (g |x|_2 + sqrt(n) s / 2) B <= limit; the larger of the two gains holds.
// A tensor's rows lie well inside the format's worst case (for 8-bit values
// B is about 0.6 of 128 sqrt(32)), so the bound raises the gain of every
// format. Q6_K scales every 16 columns: a block is two runs of 16, each
// with its own accumulation, sum pair, fold and bound, and no min term. The
// top sum is then exact for every input, and the low sum
// carries the accumulator's rounding: a token's result depends on the token
// it is packed with, so the form is an error class of its entry. A token is
// the top one in the first half of the blocks and the low one in the second.
// An operand element must be exact in F32: the top token's codes stay within
// `top_codes`, or within twice that where the pair's low codes are even
// (s = 2), which the operand pass chooses per pair and run where it lowers the
// pair's error.
//
// Weights whose values are 8 bits (Q8_0 and iq4g32) have sums 16 times those
// of 4-bit codes, which 16 bits a token do not hold at a useful gain. Their
// accumulator is A = 2^(16 - R) T + L / 8 with R residue bits, the form's
// width (`RESIDUE`, 1 or 2): the top sum T takes 16 + R bits and overlaps
// the low field's upper R bits, and the low sum L enters over 8, so its field
// holds 2^18. The overlap is resolved by T mod 2^R from the products of the
// top token's codes' and the weight row's values' low bit planes (AND and
// popcount: T mod 4 = sum t0 w0 + 2 (t0 w1 + t1 w0)): with it subtracted at
// bit 16 - R, the word's low half is the low field and its high half is
// floor(T / 2^R), less one where the low field is negative. The fold reads
// both as it does elsewhere, the tokens' scales carrying 2^R and 8, and adds
// the top half's mean deficit (1 / 2 for the borrow and (2^R - 1) / 2^(R+1)
// for the floor) with the weight factor. A top sum of 17 bits leaves an 8-bit
// weight row's top token about 6.5-bit codes (Q8_0 rows' l2 norms are near
// the format's), an error near the admitted one; R = 2 halves its step for
// two more popcounts and twice the token entries, so it is the second width,
// taken where R = 1 is not admitted.
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
//            a run's, the first run's pair then the second's; for weights
//            with residue bits the two scales, then the low bit plane of
//            the top token's codes (bit c: the block's column c), and with
//            two residue bits its second plane (8 F16 a pair)
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

// The residue bits of the weights whose values are 8 bits are the form's
// width (the entries' PACK, 1 or 2; `packing_shared`): each bit holds the
// top sum to one more bit beside a 16-bit low field, 17 or 18. One is the
// faster; two halves the top token's step where one is not precise enough.
// Codes that need residue bits declare `residue` 1.

// The packed codes of a weight format, the one place its packed forms and
// its packing bound (`packing_bound`) read them from. Every format with
// integer codes has them: the formats whose codes' low four bits are one
// plane (Q4_K and q4g32s; Q5_K, Q6_K, q5g32s and q5g32 with their plane of
// high bits) and those whose values are 8 bits (Q8_0 and iq4g32). The codes
// enter minus `centre`, half their range. A block has `folds` runs of
// 32 / folds columns under one weight scale each (Q6_K scales 16 columns), a
// sum pair and a bound per run (`folds` 0: no packed codes). A lane's `words`
// are the codes of its columns of one weight row and block, loaded from the
// lane's place `at` in that row's planes (the lane holds the two columns x,
// x + 1 of an 8-column step; lanes 0, 2, 4 and 6 hold a block); `step` yields
// its two codes of step s, of run s * folds / 4, as F16 subnormals.
//
// A format the packed product serves is `available`, on a device without
// tensor operations (their int8 products carry more than a packed matrix
// operand does): `biased` formats have a min per block; `factors` gives a
// row's block factors of a 256-column block for `packing_coefficients`:
// `reach` a bound on what the fold multiplies, `start` and `take` each
// block's two factors in turn. `residue` is the bits of the top sum the code
// planes resolve (0: the sums fit 16 bits each); weights with a residue bit
// give a row's low bit plane of a block from its lanes' words (`plane`), and
// their steps' columns are bytes in column order.
template <typename W>
struct packing_codes {
    static constant constexpr bool available = false;
    static constant constexpr uint centre = 0, folds = 0, residue = 0;
};
// The packed product's forms, on a device without tensor operations.
constant constexpr bool packing_forms = !SEISMIC_HAS_TENSOR_OPS;
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
    static constant constexpr bool available = packing_forms;
    static constant constexpr uint centre = 8, folds = 1, residue = 0;
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
    static constant constexpr bool available = packing_forms;
    static constant constexpr uint centre = 8, folds = 1, residue = 0;
    static constant constexpr bool biased = false;
    typedef packing_g32s_factors factors;
    typedef packing_nibbles<packets::Q4G32S> nibbles;
    typedef nibbles::place place;
    typedef nibbles::words words;
    static place at(thread const Weights<packets::Q4G32S> &w, uint n, uint x) { return nibbles::at(w, n, x); }
    static words load(thread const Weights<packets::Q4G32S> &w, place at, uint g) { return nibbles::load(w, at, g); }
    static half2 step(words codes, uint s) { return nibbles::step(codes, s); }
};
// The 5-bit codes of a plane of low nibbles and a plane of high bits (q5k,
// q5g32s, q5g32): q4k's columns; the lane's byte of the block's high bits
// beside its word of low nibbles, a nibble of the byte under each half of the
// word, so that bit s of each half is the high bit of step s's code there.
template <typename W>
struct packing_five_bits {
    typedef uint2 place;
    typedef uint2 words;
    static place at(thread const Weights<W> &w, uint n, uint x) {
        return uint2(w.codes_at(n, 16u, 0u) + (x >> 1) * 4u, w.high_at(n, 4u, 0u) + (x >> 1));
    }
    static words load(thread const Weights<W> &w, place at, uint g) {
        const uint high = w.base[at.y + g * 4u * w.layout.tile];
        return uint2(*reinterpret_cast<device const uint *>(w.base + at.x + g * 16u * w.layout.tile),
            (high | (high << 12)) & 0x000f000fu);
    }
    static half2 step(words codes, uint s) {
        return as_type<half2>(((codes.x >> (4u * s)) & 0x000f000fu) | (((codes.y >> s) & 0x00010001u) << 4));
    }
};
template <>
struct packing_codes<packets::Q5K> {
    static constant constexpr bool available = packing_forms;
    static constant constexpr uint centre = 16, folds = 1, residue = 0;
    static constant constexpr bool biased = true;
    typedef packing_k_factors<16> factors;
    typedef packing_five_bits<packets::Q5K> bits;
    typedef bits::place place;
    typedef bits::words words;
    static place at(thread const Weights<packets::Q5K> &w, uint n, uint x) { return bits::at(w, n, x); }
    static words load(thread const Weights<packets::Q5K> &w, place at, uint g) { return bits::load(w, at, g); }
    static half2 step(words codes, uint s) { return bits::step(codes, s); }
};
// q5g32s (value = d * (code - 16)) and q5g32 (value = d * code + m), one
// scale per 32 columns: their codes, which no packed form multiplies.
template <bool MINIMUM>
struct packing_codes<packets::Q5G32<MINIMUM>> {
    static constant constexpr bool available = false;
    static constant constexpr uint centre = 16, folds = 1, residue = 0;
    typedef packets::Q5G32<MINIMUM> W;
    typedef packing_five_bits<W> bits;
    typedef typename bits::place place;
    typedef typename bits::words words;
    static place at(thread const Weights<W> &w, uint n, uint x) { return bits::at(w, n, x); }
    static words load(thread const Weights<W> &w, place at, uint g) { return bits::load(w, at, g); }
    static half2 step(words codes, uint s) { return bits::step(codes, s); }
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
    static constant constexpr bool available = packing_forms;
    static constant constexpr uint centre = 32, folds = 2, residue = 0;
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
// Codes of 8-bit values (q8g32s, iq4g32): the lane at fragment column x
// holds its row's eight values 4 x .. 4 x + 7 of the block as two words of
// bytes, each value + 128 (read as an F16 subnormal, so the centre is 128).
// Step s multiplies the block's columns
// 8 (c / 2) + 4 (s / 2) + s % 2 + 2 (c % 2) for c = 0 .. 7: bytes s % 2 and
// s % 2 + 2 of word s / 2.
struct packing_bytes {
    typedef uint2 words;
    static half2 step(words codes, uint s) {
        return as_type<half2>((codes[s >> 1] >> (8u * (s & 1u))) & 0x00ff00ffu);
    }
    // Bit planes 0 and, with two residue bits, 1 of the row's 32 values of
    // the block (bit c: column c): a word's four bits of a plane are
    // gathered by one multiply (byte i's bit to bit 24 + i), and the row's
    // four lanes (fragment columns 0, 2, 4, 6: lane bits 0 and 3) hold its
    // 32 bytes between them. Adding 128 leaves both planes the value's.
    template <uint RESIDUE>
    static uint2 planes(words codes, uint x) {
        const uint2 low = ((codes & 0x01010101u) * 0x01020408u) >> 24;
        uint2 row = uint2(low.x | (low.y << 4), 0u);
        if (RESIDUE > 1) {
            const uint2 high = (((codes >> 1) & 0x01010101u) * 0x01020408u) >> 24;
            row.y = high.x | (high.y << 4);
        }
        row <<= 4u * x;
        row |= simd_shuffle_xor(row, ushort(1));
        row |= simd_shuffle_xor(row, ushort(8));
        return row;
    }
};
// q8 (q8g32s): value = d * code with int8 codes and one F16 d per 32
// columns, so a block is one run and has no min; its factor is q4g32s's. A
// stored byte with its sign bit flipped is code + 128.
template <>
struct packing_codes<packets::Q8> {
    static constant constexpr bool available = packing_forms;
    static constant constexpr uint centre = 128, folds = 1, residue = 1;
    static constant constexpr bool biased = false;
    typedef packing_g32s_factors factors;
    typedef uint place;
    typedef packing_bytes::words words;
    static place at(thread const Weights<packets::Q8> &w, uint n, uint x) {
        return w.codes_at(n, 32u, 0u) + (x >> 1) * 8u;
    }
    // A row tile's rows of one block are adjacent: 32 bytes a row.
    static words load(thread const Weights<packets::Q8> &w, place at, uint g) {
        return *reinterpret_cast<device const uint2 *>(w.base + at + g * 32u * w.layout.tile) ^ 0x80808080u;
    }
    static half2 step(words codes, uint s) { return packing_bytes::step(codes, s); }
    template <uint RESIDUE>
    static uint2 planes(words codes, uint x) { return packing_bytes::planes<RESIDUE>(codes, x); }
};
// iq4g32: value = d * table(code) with 4-bit codes into the IQ4 table (values
// -127 .. 113) and one F32 d per 32 columns, so a block is one run and has
// no min. A block's factor is d * 32767 / 64 (the second is unused), and the
// fold takes a sum of at most 1 of it: 512 |d| bounds it.
struct packing_iq4_factors {
    // The scales of a 256-column block's packets from the next taken.
    typedef device const float *run;
    static run start(device const uchar *row, Rows16 layout, uint block) {
        return reinterpret_cast<device const float *>(layout.super_at(row, 32ul * block));
    }
    static float reach(device const uchar *row, Rows16 layout, uint block) {
        const run scales = start(row, layout, block);
        float largest = 0.0f;
        PROJECTION_UNROLL
        for (uint i = 0; i < 8; ++i)
            largest = metal::max(largest, metal::abs(scales[i]));
        return 512.0f * largest;
    }
    static float2 take(thread run &state) { return float2(*state++ * (32767.0f / 64.0f), 0.0f); }
};
// A byte of two IQ4 codes (the low nibble's column first) as its two values
// + 128, the low nibble's in the low byte: (table(b % 16) + 128) +
// 256 (table(b / 16) + 128).
constant ushort packing_iq4_pairs[256] = {
    0x0101, 0x0118, 0x012d, 0x013f, 0x014f, 0x015d, 0x016a, 0x0176, 0x0181, 0x018d, 0x0199, 0x01a6,
    0x01b5, 0x01c5, 0x01d9, 0x01f1, 0x1801, 0x1818, 0x182d, 0x183f, 0x184f, 0x185d, 0x186a, 0x1876,
    0x1881, 0x188d, 0x1899, 0x18a6, 0x18b5, 0x18c5, 0x18d9, 0x18f1, 0x2d01, 0x2d18, 0x2d2d, 0x2d3f,
    0x2d4f, 0x2d5d, 0x2d6a, 0x2d76, 0x2d81, 0x2d8d, 0x2d99, 0x2da6, 0x2db5, 0x2dc5, 0x2dd9, 0x2df1,
    0x3f01, 0x3f18, 0x3f2d, 0x3f3f, 0x3f4f, 0x3f5d, 0x3f6a, 0x3f76, 0x3f81, 0x3f8d, 0x3f99, 0x3fa6,
    0x3fb5, 0x3fc5, 0x3fd9, 0x3ff1, 0x4f01, 0x4f18, 0x4f2d, 0x4f3f, 0x4f4f, 0x4f5d, 0x4f6a, 0x4f76,
    0x4f81, 0x4f8d, 0x4f99, 0x4fa6, 0x4fb5, 0x4fc5, 0x4fd9, 0x4ff1, 0x5d01, 0x5d18, 0x5d2d, 0x5d3f,
    0x5d4f, 0x5d5d, 0x5d6a, 0x5d76, 0x5d81, 0x5d8d, 0x5d99, 0x5da6, 0x5db5, 0x5dc5, 0x5dd9, 0x5df1,
    0x6a01, 0x6a18, 0x6a2d, 0x6a3f, 0x6a4f, 0x6a5d, 0x6a6a, 0x6a76, 0x6a81, 0x6a8d, 0x6a99, 0x6aa6,
    0x6ab5, 0x6ac5, 0x6ad9, 0x6af1, 0x7601, 0x7618, 0x762d, 0x763f, 0x764f, 0x765d, 0x766a, 0x7676,
    0x7681, 0x768d, 0x7699, 0x76a6, 0x76b5, 0x76c5, 0x76d9, 0x76f1, 0x8101, 0x8118, 0x812d, 0x813f,
    0x814f, 0x815d, 0x816a, 0x8176, 0x8181, 0x818d, 0x8199, 0x81a6, 0x81b5, 0x81c5, 0x81d9, 0x81f1,
    0x8d01, 0x8d18, 0x8d2d, 0x8d3f, 0x8d4f, 0x8d5d, 0x8d6a, 0x8d76, 0x8d81, 0x8d8d, 0x8d99, 0x8da6,
    0x8db5, 0x8dc5, 0x8dd9, 0x8df1, 0x9901, 0x9918, 0x992d, 0x993f, 0x994f, 0x995d, 0x996a, 0x9976,
    0x9981, 0x998d, 0x9999, 0x99a6, 0x99b5, 0x99c5, 0x99d9, 0x99f1, 0xa601, 0xa618, 0xa62d, 0xa63f,
    0xa64f, 0xa65d, 0xa66a, 0xa676, 0xa681, 0xa68d, 0xa699, 0xa6a6, 0xa6b5, 0xa6c5, 0xa6d9, 0xa6f1,
    0xb501, 0xb518, 0xb52d, 0xb53f, 0xb54f, 0xb55d, 0xb56a, 0xb576, 0xb581, 0xb58d, 0xb599, 0xb5a6,
    0xb5b5, 0xb5c5, 0xb5d9, 0xb5f1, 0xc501, 0xc518, 0xc52d, 0xc53f, 0xc54f, 0xc55d, 0xc56a, 0xc576,
    0xc581, 0xc58d, 0xc599, 0xc5a6, 0xc5b5, 0xc5c5, 0xc5d9, 0xc5f1, 0xd901, 0xd918, 0xd92d, 0xd93f,
    0xd94f, 0xd95d, 0xd96a, 0xd976, 0xd981, 0xd98d, 0xd999, 0xd9a6, 0xd9b5, 0xd9c5, 0xd9d9, 0xd9f1,
    0xf101, 0xf118, 0xf12d, 0xf13f, 0xf14f, 0xf15d, 0xf16a, 0xf176, 0xf181, 0xf18d, 0xf199, 0xf1a6,
    0xf1b5, 0xf1c5, 0xf1d9, 0xf1f1,
};
template <>
struct packing_codes<packets::IQ4> {
    static constant constexpr bool available = packing_forms;
    static constant constexpr uint centre = 128, folds = 1, residue = 1;
    static constant constexpr bool biased = false;
    typedef packing_iq4_factors factors;
    typedef uint place;
    typedef packing_bytes::words words;
    static place at(thread const Weights<packets::IQ4> &w, uint n, uint x) {
        return w.codes_at(n, 16u, 0u) + (x >> 1) * 4u;
    }
    // A row tile's rows of one block are adjacent: 16 bytes a row. The
    // lane's four bytes of codes become its eight values.
    static words load(thread const Weights<packets::IQ4> &w, place at, uint g) {
        const uint codes = *reinterpret_cast<device const uint *>(w.base + at + g * 16u * w.layout.tile);
        return uint2(uint(packing_iq4_pairs[codes & 0xffu]) | (uint(packing_iq4_pairs[(codes >> 8) & 0xffu]) << 16),
            uint(packing_iq4_pairs[(codes >> 16) & 0xffu]) | (uint(packing_iq4_pairs[codes >> 24]) << 16));
    }
    static half2 step(words codes, uint s) { return packing_bytes::step(codes, s); }
    template <uint RESIDUE>
    static uint2 planes(words codes, uint x) { return packing_bytes::planes<RESIDUE>(codes, x); }
};

// The fields of a packed element and of a run's accumulator for weights
// with RESIDUE residue bits. An element is (2^top_shift top + low) x
// `element`; a run accumulates 2^top_bit T + L / 2^low_shift. The top sum is
// read over 2^RESIDUE and the low sum over 2^low_shift, which the tokens'
// scales carry (`top_role`, `low_role`). `top_codes` is the largest top
// code beside any low code (the element stays below 2^24; twice that and one
// more beside even low codes), `top_limit` the largest top sum (the
// accumulator stays below 2^31, with the low field beside it), and
// `low_limit` the largest low sum of a run of n columns: its field's 32767,
// less the accumulator's rounding (each of n roundings at most 64) and, where
// the low sum is divided, the truncation of the read.
template <uint RESIDUE>
struct packing_fields {
    static_assert(RESIDUE <= 2, "the top sum's residue is one or two bit planes");
    static constant constexpr uint low_shift = RESIDUE != 0 ? 3 : 0;
    static constant constexpr uint top_bit = 16 - RESIDUE;
    static constant constexpr uint top_shift = top_bit + low_shift;
    static constant constexpr float element = float(1u << (24 - low_shift));
    static constant constexpr float top_codes = float((1u << (24 - top_shift)) - 1u);
    // The largest low code: below the top code's lowest bit.
    static constant constexpr float low_codes = float((1u << top_shift) - 1u);
    static constant constexpr float top_limit = RESIDUE != 0 ? float((1u << (31 - top_bit)) - 4u) : 32767.0f;
    static constant constexpr float top_role = float(1u << RESIDUE), low_role = float(1u << low_shift);
    static float low_limit(float n) {
        return (32767.0f - 64.0f * n - (RESIDUE != 0 ? 2.0f : 0.0f)) * low_role;
    }
    // The top half's mean deficit against floor(T / 2^RESIDUE), in the
    // fold's units (a sum over 32767): the borrow of a negative low field
    // and the floor's dropped bits.
    static constant constexpr float deficit =
        (0.5f + float((1u << RESIDUE) - 1u) / float(2u << RESIDUE)) / 32767.0f;
};

// Whether an operand laid out for blocks of FOLDS runs and coded for RESIDUE
// residue bits (any width where nonzero) serves the weights W.
template <typename W, uint FOLDS, uint RESIDUE>
struct packing_serves {
    static constant constexpr bool value = packing_codes<W>::available && packing_codes<W>::folds == FOLDS
        && (packing_codes<W>::residue != 0) == (RESIDUE != 0);
};

// The operand shared by the weights A .. D of RA .. RD weight rows: one
// operand serves weights of one layout (one run a block, two runs a block,
// or one run with residue bits), and the shared one is laid out for the
// layout that serves the most weight rows (`folds`; 0: no weight has a
// packed path), a layout without residue bits before one with them where
// they serve as many. It is coded for the largest centre among the weights it
// serves, whose gain limits are the tightest, and with residue bits for
// WIDTH of them (the form's PACK).
template <uint WIDTH, typename A, uint RA, typename B = A, uint RB = 0, typename C = A, uint RC = 0,
    typename D = A, uint RD = 0>
struct packing_shared {
    static_assert(WIDTH == 1 || WIDTH == 2, "an operand's residue is one or two bits");
    // The weight rows an operand of FOLDS runs a block and RESIDUE residue
    // bits serves.
    template <uint FOLDS, uint RESIDUE>
    struct rows {
        static constant constexpr uint value = (packing_serves<A, FOLDS, RESIDUE>::value ? RA : 0)
            + (packing_serves<B, FOLDS, RESIDUE>::value ? RB : 0)
            + (packing_serves<C, FOLDS, RESIDUE>::value ? RC : 0)
            + (packing_serves<D, FOLDS, RESIDUE>::value ? RD : 0);
    };
    static constant constexpr uint one = rows<1, 0>::value, two = rows<2, 0>::value,
        bytes = rows<1, 1>::value;
    static constant constexpr uint folds = one + two + bytes == 0 ? 0 : (two > one && two >= bytes) ? 2 : 1;
    static constant constexpr uint residue = folds == 1 && bytes > one ? WIDTH : 0;
    static constant constexpr uint a = packing_serves<A, folds, residue>::value ? packing_codes<A>::centre : 0;
    static constant constexpr uint b = packing_serves<B, folds, residue>::value ? packing_codes<B>::centre : 0;
    static constant constexpr uint c = packing_serves<C, folds, residue>::value ? packing_codes<C>::centre : 0;
    static constant constexpr uint d = packing_serves<D, folds, residue>::value ? packing_codes<D>::centre : 0;
    static constant constexpr uint ab = a > b ? a : b, cd = c > d ? c : d;
    static constant constexpr uint centre = ab > cd ? ab : cd;
};

// The gain limits of a run of n = 32 / folds columns against codes minus
// `centre` (each at most `centre` in magnitude): a run's sum is at most
// centre |q|_1, and codes rounded to multiples of s have
// |q|_1 <= g |x|_1 + n s / 2, so the largest gain under a sum limit is
// (limit / centre - n s / 2) / |x|_1, with the limits of `packing_fields`
// (32767 for the top token and 32767 - 64 n for the low one where the sums
// fit 16 bits each). Returned as what a run's scale (64 / gain) is per unit
// of |x|_1, of the top token, of the low token and of the low token with even
// codes: 64 / limit over 1 - 2^-10, which covers the F32 rounding of the norm
// (at most 2^-19 of a sum of 32 magnitudes, in any order) and of the
// products with it.
template <uint RESIDUE>
inline float3 packing_limits(uint centre, uint folds) {
    const float n = 32.0f / float(folds), c = float(centre);
    const float top = packing_fields<RESIDUE>::top_limit, low = packing_fields<RESIDUE>::low_limit(n);
    return (64.0f / 0.9990234375f) / float3(top / c - 0.5f * n, low / c - 0.5f * n, low / c - n);
}

// A scale (64 / gain, what the fold multiplies) as the fold holds it, in 16
// bits (`packing_scale_bits`): its fraction rounded up to 8 bits, so that
// the gain, the reciprocal of what is held, is at most the one admitted.
inline float packing_held(float scale) { return as_type<float>((as_type<uint>(scale) + 0x7fffu) & 0xffff8000u); }
// The gain of a held scale.
inline float packing_gain(float scale) { return scale > 0.0f ? 64.0f / scale : 0.0f; }

// A token's norms of a run from its eight columns of a chunk, joined over
// the run's lanes (four neighbouring lanes hold a block, two a run of 16):
// the sum of the magnitudes, the largest, and the sum of the squares.
inline float3 packing_norms(float4 even, float4 odd, uint folds) {
    const float4 magnitudes = metal::abs(even) + metal::abs(odd);
    const float4 peaks = metal::max(metal::abs(even), metal::abs(odd));
    const float4 squares = even * even + odd * odd;
    float3 norms = float3((magnitudes.x + magnitudes.y) + (magnitudes.z + magnitudes.w),
        metal::max(metal::max(peaks.x, peaks.y), metal::max(peaks.z, peaks.w)),
        (squares.x + squares.y) + (squares.z + squares.w));
    float3 other = simd_shuffle_xor(norms, ushort(1));
    norms = float3(norms.x + other.x, metal::max(norms.y, other.y), norms.z + other.z);
    if (folds == 1) {
        other = simd_shuffle_xor(norms, ushort(2));
        norms = float3(norms.x + other.x, metal::max(norms.y, other.y), norms.z + other.z);
    }
    return norms;
}

// The packing bounds of the weights an operand serves (`<weight>_bound`
// ports, `[K / n] f32`, `packing_bound`): per run of n columns the largest
// squared l2 norm of a row's codes minus their centre, over each weight's
// rows. A run's bound is the largest of the served weights', `tables[i]` for
// each bit i of SERVED. SERVED 0: no bound (a served weight has none), and
// the format's bound alone certifies the gains.
template <uint SERVED>
struct packing_bounds {
    static_assert(SERVED < 16, "an operand serves at most four weights");
    device const float *tables[4];
    float at(uint run) const {
        float bound = 0.0f;
        PROJECTION_UNROLL
        for (uint i = 0; i < 4; ++i) {
            if ((SERVED >> i) & 1u)
                bound = metal::max(bound, tables[i][run]);
        }
        return bound;
    }
};
// The `packing_bounds` mask of an operand serving the weights of the set
// bits of SERVED, of which those of BOUNDED have a bound: SERVED where every
// served weight has one, else none.
template <uint SERVED, uint BOUNDED>
struct packing_bounded {
    static constant constexpr uint value = (SERVED & ~BOUNDED) == 0 ? SERVED : 0;
};

// The held scales of a pair's top and low token of one run, from their
// `packing_norms`, and the step of the low token's codes. The format's
// bound certifies scales `limits` per unit of |x|_1; with the tensors' bound
// B^2 of the run (`squares`, BOUNDED), Cauchy-Schwarz with the rounding of
// the codes inside, (g |x|_2 + sqrt(n) s / 2) B <= limit for runs of
// n = 32 / FOLDS columns, certifies its own, and the smaller of the two
// scales (the larger gain) holds. The top token's codes stay within
// `top_codes`, or within 2 top_codes + 1 beside even low codes, and the low
// token's within `low_codes`; the pair takes the wider top where it lowers
// the sum of the two tokens' squared relative steps (64 / gain over |x|_1,
// the low token's doubled by its step), which only a top capped by its
// codes can.
template <uint RESIDUE, uint FOLDS, bool BOUNDED>
inline float3 packing_pair_scales(float3 top, float3 low, float3 limits, float squares) {
    typedef packing_fields<RESIDUE> fields;
    float3 certified = float3(top.x * limits.x, low.x * limits.y, low.x * limits.z);
    if (BOUNDED) {
        // sqrt(n) / 2 of rounding, twice that for even codes, against each
        // field's limit over the bound (none in a run whose codes are all
        // the centre: the inverse is then infinite, and so is the room).
        const float n = 32.0f / float(FOLDS), root = FOLDS == 1 ? 5.65685424949238f : 4.0f;
        const float inverse = metal::rsqrt(squares);
        const float low_limit = fields::low_limit(n);
        const float3 room = float3(fields::top_limit * inverse - 0.5f * root, low_limit * inverse - 0.5f * root,
            low_limit * inverse - root);
        certified = metal::min(certified,
            (64.0f / 0.9990234375f) * metal::sqrt(float3(top.z, low.z, low.z)) / room);
    }
    const float narrow = packing_held(metal::max(certified.x, top.y * (64.0f / fields::top_codes)));
    const float wide = packing_held(metal::max(certified.x, top.y * (64.0f / (2.0f * fields::top_codes + 1.0f))));
    const float fine = packing_held(metal::max(certified.y, low.y * (64.0f / fields::low_codes)));
    const float even = packing_held(metal::max(certified.z, low.y * (64.0f / fields::low_codes)));
    const bool widen = narrow * narrow * low.x * low.x + fine * fine * top.x * top.x
        > wide * wide * low.x * low.x + 4.0f * even * even * top.x * top.x;
    return widen ? float3(wide, even, 2.0f) : float3(narrow, fine, 1.0f);
}

// One token's eight columns of a run as codes under a held `scale`, in
// multiples of `step`, and the run's code sum (over its lanes).
inline float packing_code(float4 even, float4 odd, float scale, float step, uint folds, thread float4 &even_codes,
    thread float4 &odd_codes) {
    const float gain = packing_gain(scale) / step;
    even_codes = metal::rint(even * gain) * step;
    odd_codes = metal::rint(odd * gain) * step;
    float sum = (even_codes.x + even_codes.y) + (even_codes.z + even_codes.w)
        + (odd_codes.x + odd_codes.y) + (odd_codes.z + odd_codes.w);
    sum += simd_shuffle_xor(sum, ushort(1));
    if (folds == 1)
        sum += simd_shuffle_xor(sum, ushort(2));
    return sum;
}

// Bit plane `bit` of a token's eight codes of chunk `word` of a block (bit
// 8 word + c: the chunk's column c), joined over the block's four lanes.
inline uint packing_plane(float4 even_codes, float4 odd_codes, uint bit, uint word) {
    const uint4 even = uint4((int4(even_codes) >> int(bit)) & 1), odd = uint4((int4(odd_codes) >> int(bit)) & 1);
    uint plane = (even.x | (odd.x << 1) | (even.y << 2) | (odd.y << 3) | (even.z << 4) | (odd.z << 5)
        | (even.w << 6) | (odd.w << 7)) << (8u * word);
    plane |= simd_shuffle_xor(plane, ushort(1));
    plane |= simd_shuffle_xor(plane, ushort(2));
    return plane;
}

// A held block scale (`packing_held`: non-negative, 8 fraction bits) as 16
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
// token scales are stored and the block scales rewritten over them. CENTRE,
// FOLDS and RESIDUE are the centre, the runs a block and the residue bits of
// the weights the operand multiplies (`packing_shared`), whose steps' columns
// it is laid out in (`packing_codes`); `bounds` the packing bounds of the
// weights it serves.
template <uint CENTRE, uint FOLDS, uint RESIDUE, uint BOUNDS, typename In>
inline void packing_operand(thread const In &in, thread const packing_scratch &s,
    thread const packing_bounds<BOUNDS> &bounds, uint m_rows, uint columns, uint pairs, threadgroup float4 *peaks,
    uint thread_index, uint sg, uint lane) {
    static_assert(RESIDUE == 0 || FOLDS == 1, "weights with a residue bit scale every 32 columns");
    typedef packing_fields<RESIDUE> fields;
    const uint groups = columns / 32u, steps = columns / 8u;
    const float3 limits = packing_limits<RESIDUE>(CENTRE, FOLDS);
    const uint tile = pairs / 16u, within = (2u * pairs) & 31u;
    // A pair's entry of a block is 8 bytes, 16 with two residue bits.
    constexpr uint ENTRY = RESIDUE > 1 ? 16u : 8u;
    device uchar *entries = reinterpret_cast<device uchar *>(s.tokens) + (ulong(tile) * groups * 32u + within) * ENTRY;
    float4 largest = float4(0.0f);
    for (uint chunk = thread_index; chunk < steps; chunk += 256u) {
        const uint g = chunk / 4u, word = chunk % 4u;
        // A pair's first token is the top one in the first half of the blocks.
        const bool first_top = 2u * g < groups;
        // The run of this chunk's 8 columns: runs are of 32 / FOLDS columns.
        const float bound = BOUNDS != 0 ? bounds.at(chunk * FOLDS / 4u) : 0.0f;
        float4 even[4], odd[4];
        float3 norms[4];
        PROJECTION_UNROLL
        for (uint t = 0; t < 4; ++t) {
            even[t] = float4(0.0f);
            odd[t] = float4(0.0f);
            if (4u * pairs + t < m_rows)
                in.load8(4u * pairs + t, chunk * 8u, 1.0f, even[t], odd[t]);
            norms[t] = packing_norms(even[t], odd[t], FOLDS);
        }
        float4 packed_even[2], packed_odd[2];
        PROJECTION_UNROLL
        for (uint p = 0; p < 2; ++p) {
            const uint top = 2u * p + (first_top ? 0u : 1u), low = 2u * p + (first_top ? 1u : 0u);
            const float3 chosen = packing_pair_scales<RESIDUE, FOLDS, (BOUNDS != 0)>(norms[top], norms[low], limits, bound);
            float4 top_even, top_odd, low_even, low_odd;
            const float top_sum = packing_code(even[top], odd[top], chosen.x, 1.0f, FOLDS, top_even, top_odd);
            const float low_sum = packing_code(even[low], odd[low], chosen.y, chosen.z, FOLDS, low_even, low_odd);
            packed_even[p] = (top_even * float(1u << fields::top_shift) + low_even) * fields::element;
            packed_odd[p] = (top_odd * float(1u << fields::top_shift) + low_odd) * fields::element;
            // The pair's scales over what their sums are read over, and code
            // sums / 64, in token order.
            const float2 top_factors = float2(chosen.x * fields::top_role, top_sum * 0x1p-6f);
            const float2 low_factors = float2(chosen.y * fields::low_role, low_sum * 0x1p-6f);
            const float2 first = first_top ? top_factors : low_factors, second = first_top ? low_factors : top_factors;
            largest[2u * p] = metal::max(largest[2u * p], first.x);
            largest[2u * p + 1u] = metal::max(largest[2u * p + 1u], second.x);
            device uchar *entry = entries + (g * 32u + p) * ENTRY;
            // A run's first lane stores its entry: the block's scales and
            // code sums, with two runs a block each run's scales, or with
            // residue bits the block's scales and the top token's low plane
            // (and with two its second).
            if constexpr (RESIDUE > 1) {
                const uint2 planes = uint2(packing_plane(top_even, top_odd, 0u, word),
                    packing_plane(top_even, top_odd, 1u, word));
                if ((lane & 3u) == 0)
                    *reinterpret_cast<device uint4 *>(entry) = uint4(uint(packing_scale_bits(first.x))
                        | (uint(packing_scale_bits(second.x)) << 16), planes.x, planes.y, 0u);
            } else if constexpr (RESIDUE != 0) {
                const uint plane = packing_plane(top_even, top_odd, 0u, word);
                if ((lane & 3u) == 0)
                    *reinterpret_cast<device uint2 *>(entry) = uint2(uint(packing_scale_bits(first.x))
                        | (uint(packing_scale_bits(second.x)) << 16), plane);
            } else if constexpr (FOLDS == 2) {
                if ((lane & 1u) == 0)
                    reinterpret_cast<device ushort2 *>(entry)[word >> 1] = ushort2(
                        packing_scale_bits(first.x), packing_scale_bits(second.x));
            } else {
                if ((lane & 3u) == 0)
                    *reinterpret_cast<device metal::vec<ushort, 4> *>(entry) = metal::vec<ushort, 4>(
                        packing_scale_bits(first.x), packing_scale_bits(second.x),
                        as_type<ushort>(half(first.y)), as_type<ushort>(half(second.y)));
            }
        }
        // Column c of the chunk is step c % 4 of the block, at lane row
        // 2 word + c / 4; with two runs a block, step c % 2 of run word / 2,
        // at lane row 4 (word % 2) + 2 ((c / 2) % 2) + c / 4; with residue
        // bits (bytes in column order), step 2 (c / 4) + c % 2, at lane row
        // 2 word + (c / 2) % 2.
        device float2 *to = reinterpret_cast<device float2 *>(s.operand)
            + (ulong(tile) * steps + g * 4u) * 128u + (within >> 3);
        PROJECTION_UNROLL
        for (uint c = 0; c < 8; ++c) {
            const uint step = RESIDUE != 0 ? 2u * (c >> 2) + (c & 1u) : FOLDS == 2 ? (word & 2u) + (c & 1u) : c & 3u;
            const uint row = RESIDUE != 0 ? 2u * word + ((c >> 1) & 1u)
                : FOLDS == 2 ? 4u * (word & 1u) + (c & 2u) + (c >> 2) : 2u * word + (c >> 2);
            to[(step * 32u + fragment_lane(within & 6u, row)) * 4u] = float2(
                (c & 1u) ? packed_odd[0][c / 2u] : packed_even[0][c / 2u],
                (c & 1u) ? packed_odd[1][c / 2u] : packed_even[1][c / 2u]);
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
            device ushort2 *entry = reinterpret_cast<device ushort2 *>(entries + ((chunk / 4u) * 32u + p) * ENTRY)
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
// WIDTH is the operand's residue width for weights with residue bits (the
// form's PACK, `packing_shared`).
template <typename W, uint WIDTH, uint TOKENS, uint WEIGHTS_AHEAD>
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
    constexpr uint RESIDUE = coded::residue != 0 ? WIDTH : 0;
    typedef packing_fields<RESIDUE> fields;
    device const float4 *operand = reinterpret_cast<device const float4 *>(s.operand);
    device const uint4 *tokens = reinterpret_cast<device const uint4 *>(s.tokens);
    device const uint2 *weights = reinterpret_cast<device const uint2 *>(s.weights);
    // The tile's first operand fragment of its 64-token tile, and its first
    // 8-row group of its 32-row weight tile.
    const uint tile_m = m0 / 64u, fragment0 = (m0 % 64u) / 16u, rows0 = (n0 % 32u) / 8u;
    const uint operand_at = (tile_m * (k / 8u) * 32u + lane) * 2u + fragment0 / 2u;
    // The lane's two pairs' entries of a block: one quadruple, or with two
    // residue bits one each.
    const uint tokens_at = RESIDUE > 1 ? tile_m * groups * 32u + 8u * fragment0 + at.x
                                       : tile_m * groups * 16u + 4u * fragment0 + (at.x >> 1);
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
            uint4 token[TI], second_token[TI];
            PROJECTION_UNROLL
            for (uint i = 0; i < TI; ++i) {
                if constexpr (RESIDUE > 1) {
                    token[i] = tokens[tokens_at + g * 32u + 8u * i];
                    second_token[i] = tokens[tokens_at + g * 32u + 8u * i + 1u];
                } else {
                    token[i] = tokens[tokens_at + g * 16u + 4u * i];
                }
            }
            // The rows' bit planes of the block, which do not depend on its
            // products.
            uint2 planes[TJ];
            if constexpr (RESIDUE != 0) {
                PROJECTION_UNROLL
                for (uint j = 0; j < TJ; ++j)
                    planes[j] = coded::template planes<RESIDUE>(words[j], at.x);
            }
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
                    if constexpr (RESIDUE != 0) {
                        // The two pairs' scales of the block and the planes
                        // of their top tokens: the low ones, and with two
                        // residue bits the second.
                        half4 scales;
                        uint2 low, high = uint2(0u);
                        if constexpr (RESIDUE > 1) {
                            scales = half4(as_type<half2>(token[i].x), as_type<half2>(second_token[i].x));
                            low = uint2(token[i].y, second_token[i].y);
                            high = uint2(token[i].z, second_token[i].z);
                        } else {
                            scales = half4(as_type<half2>(token[i].x), as_type<half2>(token[i].z));
                            low = uint2(token[i].y, token[i].w);
                        }
                        PROJECTION_UNROLL
                        for (uint j = 0; j < TJ; ++j) {
                            const half factor = as_type<half2>(weight[j / 2u][j & 1u]).x;
                            const float2 v = reinterpret_cast<thread float2 &>(acc[i][j].thread_elements());
                            // The top sums modulo 2^RESIDUE: T = sum t w, and
                            // mod 4 that is sum t0 w0 + 2 (t0 w1 + t1 w0)
                            // over the codes' bits (two's complement), whose
                            // second term needs only its parity.
                            uint2 residue = popcount(low & planes[j].x);
                            if constexpr (RESIDUE > 1)
                                residue += popcount((low & planes[j].y) ^ (high & planes[j].x)) << 1;
                            // Without them the word is the low field under
                            // floor(T / 2^RESIDUE), less the low field's
                            // borrow.
                            const uint2 split = uint2(int2(v))
                                - ((residue & ((1u << RESIDUE) - 1u)) << fields::top_bit);
                            const half2 e0 = unpack_snorm2x16_to_half(split.x), e1 = unpack_snorm2x16_to_half(split.y);
                            const half4 sum = swapped ? half4(e0.x, e0.y, e1.x, e1.y) : half4(e0.y, e0.x, e1.y, e1.x);
                            // The top half's mean deficit rides in the
                            // product with the block's scale.
                            const half deficit = factor * half(fields::deficit);
                            const half4 term = metal::fma(sum, half4(factor),
                                swapped ? half4(0.0h, deficit, 0.0h, deficit) : half4(deficit, 0.0h, deficit, 0.0h));
                            folded[i][j] = metal::fma(scales, term, folded[i][j]);
                        }
                        continue;
                    }
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
// WIDTH is the operand's residue width for weights with residue bits.
template <typename W, uint WIDTH, uint TOKENS, uint WEIGHTS_AHEAD, typename Out>
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
        gemm_packed_sums<W, WIDTH, TOKENS, WEIGHTS_AHEAD>(w, s, k, m0, n0, lane, y);
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
// weights the operand (laid out for FOLDS runs a block and coded for RESIDUE
// residue bits) does not serve leave theirs empty (the entry's exact tiles
// run that segment).
template <typename W, uint FOLDS, uint RESIDUE, uint TOKENS, uint WEIGHTS_AHEAD, typename Out>
inline bool gemm_packed_segment(thread const Out &out, thread const Weights<W> &w, thread const packing_scratch &s,
    uint m_rows, uint padded, uint rows, uint k, thread uint &group, uint simdgroups, uint sg, uint lane) {
    const uint groups = (padded / 32u * (rows / 32u) + simdgroups - 1u) / simdgroups;
    if (group >= groups) {
        group -= groups;
        return false;
    }
    if constexpr (packing_serves<W, FOLDS, RESIDUE>::value)
        gemm_packed<W, (RESIDUE != 0 ? RESIDUE : 1), TOKENS, WEIGHTS_AHEAD>(out, w, s, m_rows, padded, rows, k,
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
template <typename G, typename U, uint WIDTH, uint TOKENS, uint WEIGHTS_AHEAD, typename Out>
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
    if constexpr (packing_codes<G>::available
        && packing_serves<U, packing_codes<G>::folds, packing_codes<G>::residue>::value) {
        if (live) {
            if (second)
                gemm_packed_sums<U, WIDTH, TOKENS, WEIGHTS_AHEAD>(up, up_scratch, k, m0, n0, lane, y);
            else
                gemm_packed_sums<G, WIDTH, TOKENS, WEIGHTS_AHEAD>(gate, gate_scratch, k, m0, n0, lane, y);
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
