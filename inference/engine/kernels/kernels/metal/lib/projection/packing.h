// The token-packing GEMM (an entry's PACK form, M > 64, on simdgroup
// matrices): two activation rows (tokens; `packing_gemm_lanes`) share one F32
// matrix operand, so one multiply-accumulate carries two products. Included
// after `projection.h`.
//
// Per (token, 32 columns) the activations are rounded to integer codes under
// a gain, and the operand element of two tokens is 2^16 top + low
// (`packing_lanes`). The weights enter as their stored codes minus their
// centre, so a 32-column block of one weight row accumulates
// A = 2^16 T + L in an F32 accumulator; each lane's sum is split off, over
// 2^15 (`packing_split`), and folds into the output under (token scale) x
// (weight block scale), with the weights' min term as the product of the
// token's code sum and the block's bias.
//
// The fold runs in F16 on normalized factors, so it cannot overflow: a
// token's block scales are held over a power of two at or above its largest,
// a weight row's block factors over a power of two above every
// |scale| + 128 |bias| of the row (a block's code sum / 64 is below 128), so
// a block's term is at most 1 in magnitude and a result at most K / 32. The
// two powers of two multiply the F32 result once, at the store.
//
// The gain is the largest one that bounds every block sum for any row of the
// format. Codes minus their centre c (half their entries' range: 8 for 4-bit
// entries, 4 for 3-bit, 128 for 8-bit values) are at most c in magnitude, so a sum is at
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
// format. Weights with a scale every 16 columns have blocks of two runs of
// 16, each with its own accumulation, sum pair, fold and bound, and where
// they have a min term (`sums`), each run its own, the product of the
// tokens' code sums of the run and the run's bias. The
// top sum is then exact for every input, and the low sum
// carries the accumulator's rounding: a token's result depends on the tokens
// it is packed with, so the form is an error class of its entry. The tokens'
// lanes rotate over two phases of the blocks (`packing_role`): a token is
// the top one in the first half of the blocks and the low one in the
// second. An operand element must be exact in F32: the top token's codes
// stay within its lane's codes, or within twice that where the bottom lane's
// codes are even (s = 2), which the operand pass chooses per element and run
// where it lowers the element's error.
//
// Weights whose values are 8 bits (8-bit codes, and codebook codes of 8-bit
// values) have sums 16 times those
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
// weight row's top token about 6.5-bit codes (8-bit rows' l2 norms are near
// the format's), an error near the admitted one; R = 2 halves its step for
// two more popcounts and twice the token entries, so it is the second width,
// taken where R = 1 is not admitted.
//
// Two pre-passes write the form's scratch in the order the lanes read it:
// `packing_operand` the packed operand and the tokens' scales and code sums,
// `packing_coefficients` the weights' block scales and biases. Weight codes
// are read from the resident rows, a lane's chunk of eight codes per weight
// row and block (`packets::lane`: its share of each code plane, its codebook
// codes or its grid entry), whose pairs the lanes sharing them exchange where
// a block has two runs (`packing_layout`).
//
// The weights are the left operand (inference/docs/kernels/lane-order.md): a
// lane at fragment coordinate (x, y) holds weight row y of a weight fragment
// and token elements x, x + 1 of an operand fragment (two tokens each). A
// block is four 8-column steps, whose columns are the format's
// (`packing_codes`): each lane's two are a nibble of each half of one word of
// its row, and the operand is laid out to match.

namespace projection {

// The form's scratch, for `groups` = K / 32 blocks, tokens in tiles of
// 64 (32 elements: element r is tokens 2 r and 2 r + 1; operand fragment
// r / 8):
//   operand  8 F32 per (token tile, 8-column step, lane): fragments 0 .. 3,
//            two elements each; element (r, c) = (2^16 top + low) 2^24
//            (`packing_element`) at the lane of (x = r % 8, y = c). One step
//            past the last tile is read ahead and never used.
//   tokens   4 F16 per (token tile, block, element)
//            (`packing_entry`): 64 / gain of the element's tokens over
//            their token scales, then their code sums / 64; for weights of
//            two runs a block without a min term the gains are a run's,
//            the first run's then the second's; with a min term a run
//            (`sums`) each run's four as one run's are, the first run's
//            then the second's (8 F16 an element); for weights with residue
//            bits the two scales, then the low bit plane of the top token's
//            codes (bit c: the block's column c), and with two residue bits
//            its second plane (8 F16 an element)
//   token_scales   1 F32 per token: the power of two its block scales are
//            held over
//   weights  2 F16 per (32-row weight tile, block, row % 8, row / 8): the
//            block's two factors (`packets::block_factors`), over the row's weight
//            scale; with a min term a run (`sums`) 4, each run's two
//            factors as one run's are
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
inline float packing_scale(float value) {
    return as_type<float>((as_type<uint>(value) & 0x7f800000u) + 0x00800000u);
}

// The residue bits of the weights whose values are 8 bits are the form's
// width (the entries' PACK, 1 or 2; `packing_shared`): each bit holds the
// top sum to one more bit beside a 16-bit low field, 17 or 18. One is the
// faster; two halves the top token's step where one is not precise enough.
// The runtime admits a weight's codes with residue bits (`_FORM_PACK_RESIDUE`
// 1).

// The packed codes of weights W (`packing_codes<W>`), the one place its
// packed forms and its packing bound (`packing_bound`) read them from: a
// layout of its lane codes (`packets::lane`, entering at the admitted
// `offset`; `packing_layout`) and its coefficients' factors
// (`packets::block_factors`), for weights the runtime admits to the form
// (`packets::admits<W>::pack`, `_FORM_PACK`, with `packets::pack<W>` giving
// their codes' `centre`, `offset` and `residue`). Codes enter as F16 subnormals
// minus `centre`, half their range; a code's value is
// scale (code - offset) + bias, so a block has the min term
// (centre - offset) scale + bias where that is not zero (`biased`). A block
// has `folds` runs of 32 / folds columns under one weight scale each (the
// coefficient groups of a packet), a sum pair and a bound per run (`folds`
// 0: no packed codes), and with two runs and a min term (`sums`) a min term
// per run. A lane's `words` are the codes of its columns of one
// weight row and block, loaded from the lane's place `at` in that row's
// planes (the lane holds the two columns x, x + 1 of an 8-column step; lanes
// 0, 2, 4 and 6 hold a block); `step` yields its two codes of step s, of run
// s * folds / 4, as F16 subnormals.
//
// An admitted weight's packed product is `available` on a device without
// tensor operations (their int8 products carry more than a packed matrix
// operand does). A PACK = 1 configuration names the form only of weights
// admitted to it (a declaration's `form(W, PACK)`); on a device with tensor
// operations its `unpacked` launch runs the staged tiles of the default form
// instead, with their results. `factors` gives a row's block factors of a 256-column
// block for `packing_coefficients`: `reach` a bound on what the fold
// multiplies, `start` and `take` each block's factors (`width`) in turn. `residue`
// is the bits of the top sum the code planes resolve (0: the sums fit 16
// bits each); weights with a residue bit give a row's low bit plane of a
// block from its lanes' words (`planes`), and their steps' columns are bytes
// in column order.

// The packed product's forms, on a device without tensor operations.
constant constexpr bool packing_forms = !SEISMIC_HAS_TENSOR_OPS;

// How PACK carries a lane's codes L (`packets::lane`) ahead: a block's loads
// alone (`fetch`, its `fields`: code planes' loads, or the words of a lane
// without them) issue as it starts and become words as it ends (`spread`;
// `gemm_packed_sums`).
template <typename L, typename = void>
struct packing_fetch {
    typedef typename L::words fields;
    template <typename W>
    static fields fetch(thread const Weights<W> &w, typename L::place at, uint g) {
        return L::template load<false>(w.matrix, at, g);
    }
    static typename L::words spread(fields codes) { return codes; }
};
template <typename L>
struct packing_fetch<L, decltype((void)sizeof(typename L::fields))> {
    typedef typename L::fields fields;
    template <typename W>
    static fields fetch(thread const Weights<W> &w, typename L::place at, uint g) {
        return L::template fetch<false>(w.matrix, at, g);
    }
    static typename L::words spread(fields codes) { return L::spread(codes); }
};

// The layout of a weight's lane codes L (`packets::lane`) for FOLDS runs a
// block, in bytes of column order where BYTES (entries of 8 bits, the codes
// with residue bits; `planes`): `at`, `fetch` (a block's loads, `fetched`),
// `take` (their words, which the simdgroup forms together) and `step`. A lane
// alone has its own chunk's four code pairs (`own`), all of run `run`, which
// the four lanes of a row hold between them (the packing bound's).
// None by default.
template <typename L, uint FOLDS, bool BYTES>
struct packing_layout {
    static constexpr bool implemented() { return false; }
};

// One run a block: step s multiplies the block's columns
// 8 (c / 2) + s + 4 (c % 2) for c = 0 .. 7: a lane's chunk x / 2, its pair s.
template <typename L>
struct packing_layout<L, 1, false> {
    static constexpr bool implemented() { return L::implemented(); }
    typedef typename L::place place;
    typedef typename packing_fetch<L>::fields fetched;
    typedef typename L::words words;
    template <typename W>
    static place at(thread const Weights<W> &w, uint n, uint x) {
        return L::at(w.matrix, w.locate(n), x >> 1);
    }
    template <typename W>
    static fetched fetch(thread const Weights<W> &w, place at, uint g) { return packing_fetch<L>::fetch(w, at, g); }
    static words take(place, fetched codes) { return packing_fetch<L>::spread(codes); }
    static half2 step(thread const words &codes, uint s) { return as_type<half2>(L::pair(codes, s)); }
    template <typename W>
    static uint4 own(thread const Weights<W> &w, place at, uint g) {
        const words codes = take(at, fetch(w, at, g));
        return uint4(L::pair(codes, 0u), L::pair(codes, 1u), L::pair(codes, 2u), L::pair(codes, 3u));
    }
    static uint run(place) { return 0u; }
};

// Two runs a block: step 2 h + t multiplies the columns
// 16 h + 8 (c / 4) + 2 ((c / 2) % 2) + t + 4 (c % 2) of run h: chunks x / 4
// and x / 4 + 2 at the lane's pairs x % 4 + t (x even). Each of the two
// lanes that share those chunks (x and x ^ 2, lanes l and l ^ 1) loads one
// whole, x / 4 + x % 4 (run x % 4 / 2), keeps its own pairs of it and gives
// the other its pairs, so a chunk is loaded and decoded once.
template <typename L>
struct packing_layout<L, 2, false> {
    static constexpr bool implemented() { return L::implemented(); }
    struct place {
        typename L::place chunk;
        bool second;
    };
    typedef typename packing_fetch<L>::fields fetched;
    // The pairs of steps 0..3.
    typedef uint4 words;
    template <typename W>
    static place at(thread const Weights<W> &w, uint n, uint x) {
        return place{L::at(w.matrix, w.locate(n), (x >> 2) + (x & 2u)), (x & 2u) != 0u};
    }
    template <typename W>
    static fetched fetch(thread const Weights<W> &w, place at, uint g) {
        return packing_fetch<L>::fetch(w, at.chunk, g);
    }
    static words take(place at, fetched codes) {
        const typename L::words chunk = packing_fetch<L>::spread(codes);
        const uint2 front = uint2(L::pair(chunk, 0u), L::pair(chunk, 1u));
        const uint2 back = uint2(L::pair(chunk, 2u), L::pair(chunk, 3u));
        // The first run's lane keeps its chunk's front pairs and gives the
        // back ones; the second's keeps the back.
        const uint2 given = simd_shuffle_xor(at.second ? front : back, ushort(1));
        return at.second ? uint4(given, back) : uint4(front, given);
    }
    static half2 step(words codes, uint s) { return as_type<half2>(codes[s]); }
    template <typename W>
    static uint4 own(thread const Weights<W> &w, place at, uint g) {
        const typename L::words chunk = packing_fetch<L>::spread(fetch(w, at, g));
        return uint4(L::pair(chunk, 0u), L::pair(chunk, 1u), L::pair(chunk, 2u), L::pair(chunk, 3u));
    }
    static uint run(place at) { return at.second ? 1u : 0u; }
};

// Codes of 8-bit values: the lane at fragment column x holds its row's eight
// values 4 x .. 4 x + 7 of the block as two words of bytes (read as F16
// subnormals, so the centre is 128). Step s multiplies the block's columns
// 8 (c / 2) + 4 (s / 2) + s % 2 + 2 (c % 2) for c = 0 .. 7: bytes s % 2 and
// s % 2 + 2 of word s / 2. `planes` are bit planes 0 and, with two residue
// bits, 1 of the row's 32 values of the block (bit c: column c): a word's
// four bits of a plane are gathered by one multiply (byte i's bit to bit
// 24 + i), and the row's four lanes (fragment columns 0, 2, 4, 6: lane bits
// 0 and 3) hold its 32 bytes between them. Adding 128 leaves both planes the
// value's.
inline half2 packing_byte_step(uint2 codes, uint s) {
    return as_type<half2>((codes[s >> 1] >> (8u * (s & 1u))) & 0x00ff00ffu);
}
template <uint RESIDUE>
inline uint2 packing_byte_planes(uint2 codes, uint x) {
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

// Values of 8 bits, one run a block: the lane's chunk x / 2 as bytes
// (`packets::lane::bytes`).
template <typename L>
struct packing_layout<L, 1, true> {
    static constexpr bool implemented() { return L::implemented() && L::bytes(); }
    typedef typename L::place place;
    typedef typename packing_fetch<L>::fields fetched;
    typedef typename L::words words;
    template <typename W>
    static place at(thread const Weights<W> &w, uint n, uint x) {
        return L::at(w.matrix, w.locate(n), x >> 1);
    }
    template <typename W>
    static fetched fetch(thread const Weights<W> &w, place at, uint g) { return packing_fetch<L>::fetch(w, at, g); }
    static words take(place, fetched codes) { return packing_fetch<L>::spread(codes); }
    static half2 step(words codes, uint s) { return packing_byte_step(codes, s); }
    template <typename W>
    static uint4 own(thread const Weights<W> &w, place at, uint g) {
        const words codes = take(at, fetch(w, at, g));
        return uint4(as_type<uint>(step(codes, 0u)), as_type<uint>(step(codes, 1u)), as_type<uint>(step(codes, 2u)),
            as_type<uint>(step(codes, 3u)));
    }
    static uint run(place) { return 0u; }
    template <uint RESIDUE>
    static uint2 planes(words codes, uint x) { return packing_byte_planes<RESIDUE>(codes, x); }
};

// The packed codes of weights W the runtime admits to the form (`ADMITTED`,
// `packets::admits<W>::pack`), its runs a block the coefficient groups of a
// packet (`packets::pack`), with the layout of its lane codes
// (`packets::lane`, entering at the admitted offset) and its block factors
// (`packets::block_factors`). None otherwise.
template <typename W, bool ADMITTED = packets::admits<W>::pack>
struct packing_codes_of {
    static constant constexpr bool available = false, biased = false, sums = false;
    static constant constexpr uint centre = 0, folds = 0, residue = 0;
};
template <typename W>
struct packing_codes_of<W, true> {
    typedef packets::pack<W> admitted;
    static constant constexpr uint centre = admitted::centre, folds = admitted::folds, residue = admitted::residue;
    typedef packets::lane<W, int(admitted::offset)> lanes;
    static_assert(lanes::zero() == int(admitted::offset), "an admitted weight's codes enter at their offset");
    typedef packing_layout<lanes, folds, (residue != 0)> layout;
    static_assert(layout::implemented(), "an admitted weight's codes have a packing layout");
    typedef packets::block_factors<W, centre, admitted::offset> factors;
    static_assert(factors::implemented, "an admitted weight's coefficients have packing factors");
    static constant constexpr bool available = packing_forms;
    static constant constexpr bool biased = factors::biased, sums = factors::sums;
    // The runtime sizes the operand's and the factors' scratch by its
    // admission's PACK_SUMS: it is this layout's.
    static_assert(admitted::sums == sums, "PACK_SUMS is the packed codes' layout");
    typedef typename layout::place place;
    typedef typename layout::fetched fetched;
    typedef typename layout::words words;
    static place at(thread const Weights<W> &w, uint n, uint x) { return layout::at(w, n, x); }
    static fetched fetch(thread const Weights<W> &w, place at, uint g) { return layout::fetch(w, at, g); }
    static words take(place at, fetched codes) { return layout::take(at, codes); }
    static words load(thread const Weights<W> &w, place at, uint g) { return take(at, fetch(w, at, g)); }
    static half2 step(words codes, uint s) { return layout::step(codes, s); }
    static uint4 own(thread const Weights<W> &w, place at, uint g) { return layout::own(w, at, g); }
    static uint run(place at) { return layout::run(at); }
    template <uint RESIDUE>
    static uint2 planes(words codes, uint x) { return layout::template planes<RESIDUE>(codes, x); }
};

template <typename W>
using packing_codes = packing_codes_of<W>;

// Whether a PACK launch over the weights W... runs the packed products: on a
// device without tensor operations (`packing_forms`); elsewhere the form's
// launches return and its `unpacked` launch takes the staged tiles of the
// default form. PACK = 1 names the packed GEMM only of weights admitted to it
// (`form(W, PACK)`), which the declaration's `where` requires; only a
// configuration that runs a launch forms it.
// A function, not a class's constant: a class's static data member is a
// constant of the program, which moves what follows it.
template <typename... W> constexpr bool packing_path() {
    static_assert((packets::admits<W>::pack && ...), "PACK = 1 names the packed GEMM of weights admitted to it");
    return packing_forms;
}

// The tokens one operand element of the packed GEMM carries, two: narrow
// codes would leave room for three, but three tokens' bottom lane of about
// 5-bit codes errs above admission at every format that admits it (tuner
// ratio 10-40 BF16 steps against alpha 8 at 4096 x 4096, 512 rows).
constant constexpr uint packing_gemm_lanes = 2;

// The fields of the two lanes of tokens in one packed element, for weights
// without the top sum's residue (`packing_fields` with RESIDUE 0 is this
// case). Lane 0 is the bottom one and lane 1 the top. An element is
// (2^16 a_1 + a_0) x `element` for lane codes a_l, so a run of n columns
// accumulates A = 2^16 S_1 + S_0 for the lanes' sums S_l against the
// weights' codes minus their centre. The bottom lane has a field of 16 bits,
// holding |S_0| up to 2^15; the top lane's field is free, and its sum stays
// within `limit` so that |A| < 2^31. Each of a run's n roundings is then at
// most 2^6, which the bottom lane's limit takes off its field. Lane codes:
// the top lane's are within `codes(1)` (the element stays below 2^24; twice
// that and one more beside even bottom codes), the bottom lane's within its
// field, `codes(0)`.
struct packing_lanes {
    static constexpr uint shift(uint l) { return 16u * l; }
    static constant constexpr uint top_shift = 16;
    static constant constexpr float element = float(1u << 24);
    static constexpr float codes(uint l) {
        return l == 1u ? float((1u << (24u - top_shift)) - 1u) : float((1u << 16) - 1u);
    }
    // The largest |S_l| of a run of n columns.
    static constexpr float limit(uint l, float n) {
        return l == 1u ? float((1u << (31u - top_shift)) - 1u) : 32767.0f - n * 64.0f;
    }
};

// The lane token `position` of an element takes in `phase`
// (`packing_phase`), and the position whose token takes lane `position`
// (the map is its own inverse): a token is the top one in the first half of
// the blocks and the low one in the second.
constexpr uint packing_role(uint position, uint phase) {
    return (1u - position + phase) % 2u;
}

// The first block of phase `phase` (0 or 1) of `groups` blocks.
constexpr uint packing_phase_start(uint phase, uint groups) {
    return phase * groups / 2u;
}

// The phase of block g: 1 from the second half of the blocks.
constexpr uint packing_phase(uint g, uint groups) {
    return (2u * g + 1u) / groups;
}

// The bytes of an element's entry of a block in the tokens' scratch
// (`packing_scratch::tokens`): two scales and two code sums in F16 (or with
// two runs a block and no min term each run's two scales), each run's two
// scales and two code sums with SUMS, or 16 with two residue bits.
template <uint RESIDUE, bool SUMS>
constexpr uint packing_entry() {
    return RESIDUE > 1 || SUMS ? 16u : 8u;
}

// `m` tokens padded to whole units of `tiles` operand tiles of 64 tokens: a
// launch's tiles cover the padded tokens.
inline uint packing_padded(uint m, uint tiles) {
    const uint unit = tiles * 64u;
    return (m + unit - 1u) / unit * unit;
}

// The F32 of the operand pass's `peaks`: one per token of its two elements,
// per simdgroup.
constexpr uint packing_peaks() {
    return 8u * 4u;
}

// The fields of a packed element and of a run's accumulator for weights
// with RESIDUE residue bits. An element is (2^top_shift top + low) x
// `element`; a run accumulates 2^top_bit T + L / 2^low_shift. The top sum is
// read over 2^RESIDUE and the low sum over 2^low_shift, which the tokens'
// scales carry (`top_role`, `low_role`). As `packing_lanes` gives them, lane
// 1 is the top one and lane 0 the low: `codes(1)` is the largest top code
// beside any low code (the element stays below 2^24; twice that and one more
// beside even low codes), `codes(0)` the largest low code, below the top
// code's lowest bit, `limit(1, n)` the largest top sum (the accumulator stays
// below 2^31, with the low field beside it), and `limit(0, n)` the largest
// low sum of a run of n columns: its field's 32767, less the accumulator's
// rounding (each of n roundings at most 64) and, where the low sum is
// divided, the truncation of the read. Without residue bits these are two
// lanes' (`packing_lanes`).
template <uint RESIDUE>
struct packing_fields {
    static_assert(RESIDUE <= 2, "the top sum's residue is one or two bit planes");
    typedef packing_lanes lanes;
    static constant constexpr uint low_shift = RESIDUE != 0 ? 3 : 0;
    static constant constexpr uint top_bit = lanes::top_shift - RESIDUE;
    static constant constexpr uint top_shift = top_bit + low_shift;
    static constant constexpr float element = lanes::element / float(1u << low_shift);
    static constexpr float codes(uint l) {
        return l == 1u ? float((1u << (24 - top_shift)) - 1u) : float((1u << top_shift) - 1u);
    }
    static constant constexpr float top_role = float(1u << RESIDUE), low_role = float(1u << low_shift);
    static float limit(uint l, float n) {
        if (l == 1u)
            return RESIDUE != 0 ? float((1u << (31 - top_bit)) - 4u) : lanes::limit(1, 0.0f);
        return RESIDUE != 0 ? (lanes::limit(0, n) - 2.0f) * low_role : lanes::limit(0, n);
    }
    // The top half's mean deficit against floor(T / 2^RESIDUE), in the
    // fold's units (a sum over 2^15): the borrow of a negative low field
    // and the floor's dropped bits.
    static constant constexpr float deficit =
        (0.5f + float((1u << RESIDUE) - 1u) / float(2u << RESIDUE)) / 32768.0f;
};

// A run's accumulator A (an integer, `packing_lanes`) as its lanes' sums
// over 2^15, bottom lane first, by exact F32 operations of full rate: from
// the top down, a lane's sum over 2^15 is the remainder over 2^shift rounded
// at 384 (whose unit is 2^-15), and the remainder loses it:
// (fma(A, 2^-31, 384) - 384, fma(top, -2^16, A 2^-15)), A 2^-15 being exact.
// T is float or a float vector (several accumulator elements at once).
template <typename T>
inline void packing_split(T v, thread T (&sums)[packing_gemm_lanes]) {
    typedef packing_lanes lanes;
    T rest = v * 0x1p-15f;
    PROJECTION_UNROLL
    for (uint l = packing_gemm_lanes - 1u; l > 0u; --l) {
        sums[l] = metal::fma(rest, T(1.0f / float(1u << lanes::shift(l))), T(384.0f)) - 384.0f;
        rest = metal::fma(sums[l], T(-float(1u << lanes::shift(l))), rest);
    }
    sums[0] = rest;
}

// The packed element of the lanes' codes, bottom lane first.
template <typename T>
inline T packing_element(thread const T (&codes)[packing_gemm_lanes]) {
    typedef packing_lanes lanes;
    T element = codes[0];
    PROJECTION_UNROLL
    for (uint l = 1; l < packing_gemm_lanes; ++l)
        element = metal::fma(codes[l], T(float(1u << lanes::shift(l))), element);
    return element * lanes::element;
}

// Whether an operand laid out for blocks of FOLDS runs, with each run's code
// sums where SUMS, and coded for RESIDUE residue bits (any width where
// nonzero) serves the weights W.
template <typename W, uint FOLDS, uint RESIDUE, bool SUMS>
struct packing_serves {
    static constant constexpr bool value = packing_codes<W>::available && packing_codes<W>::folds == FOLDS
        && (packing_codes<W>::residue != 0) == (RESIDUE != 0) && packing_codes<W>::sums == SUMS;
};

// The operand shared by the weights A .. D of RA .. RD weight rows: one
// operand serves weights of one layout (one run a block, two runs a block,
// two runs a block each with its code sums, or one run with residue bits),
// and the shared one is laid out for the layout that serves the most weight
// rows (`folds`, `sums`; folds 0: no weight has a packed path), the first of
// those in that order where several serve as many. It is coded for the
// largest centre among the weights it serves, whose gain limits are the
// tightest, and with residue bits for WIDTH of them (the form's PACK).
template <uint WIDTH, typename A, uint RA, typename B = A, uint RB = 0, typename C = A, uint RC = 0,
    typename D = A, uint RD = 0>
struct packing_shared {
    static_assert(WIDTH == 1 || WIDTH == 2, "an operand's residue is one or two bits");
    // The weight rows an operand of FOLDS runs a block, RESIDUE residue bits
    // and SUMS serves.
    template <uint FOLDS, uint RESIDUE, bool SUMS>
    struct rows {
        static constant constexpr uint value = (packing_serves<A, FOLDS, RESIDUE, SUMS>::value ? RA : 0)
            + (packing_serves<B, FOLDS, RESIDUE, SUMS>::value ? RB : 0)
            + (packing_serves<C, FOLDS, RESIDUE, SUMS>::value ? RC : 0)
            + (packing_serves<D, FOLDS, RESIDUE, SUMS>::value ? RD : 0);
    };
    static constant constexpr uint one = rows<1, 0, false>::value, two = rows<2, 0, false>::value,
        summed = rows<2, 0, true>::value, bytes = rows<1, 1, false>::value;
    static constant constexpr uint unsummed = one > two ? one : two, other = summed > bytes ? summed : bytes;
    static constant constexpr uint most = unsummed > other ? unsummed : other;
    static constant constexpr uint folds = most == 0 ? 0 : (one == most || (two != most && summed != most)) ? 1 : 2;
    static constant constexpr bool sums = folds == 2 && two != most;
    static constant constexpr uint residue = folds == 1 && one != most ? WIDTH : 0;
    static constant constexpr uint a = packing_serves<A, folds, residue, sums>::value ? packing_codes<A>::centre : 0;
    static constant constexpr uint b = packing_serves<B, folds, residue, sums>::value ? packing_codes<B>::centre : 0;
    static constant constexpr uint c = packing_serves<C, folds, residue, sums>::value ? packing_codes<C>::centre : 0;
    static constant constexpr uint d = packing_serves<D, folds, residue, sums>::value ? packing_codes<D>::centre : 0;
    static constant constexpr uint ab = a > b ? a : b, cd = c > d ? c : d;
    static constant constexpr uint centre = ab > cd ? ab : cd;
};

// All distinct `(folds, residue, sums)` packing classes present among the
// segments A .. D of RA .. RD weight rows (in canonical order: one, two,
// summed, bytes). Each present class gets a contiguous operand slot in
// `0 .. count - 1`, its own maximum `centre`, and its own `mask` of served
// segments so every `N % 32 == 0` packable segment of a segmented projection
// runs `gemm_packed_segment` with no `unpacked` fallback.
template <uint WIDTH, typename A, uint RA, typename B = A, uint RB = 0, typename C = A, uint RC = 0,
    typename D = A, uint RD = 0, typename E = A, uint RE = 0, typename F = A, uint RF = 0,
    typename G = A, uint RG = 0, typename H = A, uint RH = 0>
struct packing_classes {
    static_assert(WIDTH == 1 || WIDTH == 2, "an operand's residue is one or two bits");
    template <uint FOLDS, uint RESIDUE, bool SUMS>
    struct info {
        static constant constexpr bool s0 = RA != 0 && packing_serves<A, FOLDS, RESIDUE, SUMS>::value;
        static constant constexpr bool s1 = RB != 0 && packing_serves<B, FOLDS, RESIDUE, SUMS>::value;
        static constant constexpr bool s2 = RC != 0 && packing_serves<C, FOLDS, RESIDUE, SUMS>::value;
        static constant constexpr bool s3 = RD != 0 && packing_serves<D, FOLDS, RESIDUE, SUMS>::value;
        static constant constexpr bool s4 = RE != 0 && packing_serves<E, FOLDS, RESIDUE, SUMS>::value;
        static constant constexpr bool s5 = RF != 0 && packing_serves<F, FOLDS, RESIDUE, SUMS>::value;
        static constant constexpr bool s6 = RG != 0 && packing_serves<G, FOLDS, RESIDUE, SUMS>::value;
        static constant constexpr bool s7 = RH != 0 && packing_serves<H, FOLDS, RESIDUE, SUMS>::value;
        static constant constexpr bool active = s0 || s1 || s2 || s3 || s4 || s5 || s6 || s7;
        static constant constexpr uint mask =
            (s0 ? 1u : 0u) | (s1 ? 2u : 0u) | (s2 ? 4u : 0u) | (s3 ? 8u : 0u)
            | (s4 ? 16u : 0u) | (s5 ? 32u : 0u) | (s6 ? 64u : 0u) | (s7 ? 128u : 0u);
        static constant constexpr uint c0 = s0 ? packing_codes<A>::centre : 0u;
        static constant constexpr uint c1 = s1 ? packing_codes<B>::centre : 0u;
        static constant constexpr uint c2 = s2 ? packing_codes<C>::centre : 0u;
        static constant constexpr uint c3 = s3 ? packing_codes<D>::centre : 0u;
        static constant constexpr uint c4 = s4 ? packing_codes<E>::centre : 0u;
        static constant constexpr uint c5 = s5 ? packing_codes<F>::centre : 0u;
        static constant constexpr uint c6 = s6 ? packing_codes<G>::centre : 0u;
        static constant constexpr uint c7 = s7 ? packing_codes<H>::centre : 0u;
        static constant constexpr uint c01 = c0 > c1 ? c0 : c1;
        static constant constexpr uint c23 = c2 > c3 ? c2 : c3;
        static constant constexpr uint c45 = c4 > c5 ? c4 : c5;
        static constant constexpr uint c67 = c6 > c7 ? c6 : c7;
        static constant constexpr uint c03 = c01 > c23 ? c01 : c23;
        static constant constexpr uint c47 = c45 > c67 ? c45 : c67;
        static constant constexpr uint centre = c03 > c47 ? c03 : c47;
        static constant constexpr uint folds = FOLDS;
        static constant constexpr uint residue = RESIDUE;
        static constant constexpr bool sums = SUMS;
    };
    typedef info<1, 0, false> c_one;
    typedef info<2, 0, false> c_two;
    typedef info<2, 0, true> c_summed;
    typedef info<1, WIDTH, false> c_bytes;
    static constant constexpr uint slot_one = 0u;
    static constant constexpr uint slot_two = c_one::active ? 1u : 0u;
    static constant constexpr uint slot_summed = slot_two + (c_two::active ? 1u : 0u);
    static constant constexpr uint slot_bytes = slot_summed + (c_summed::active ? 1u : 0u);
    static constant constexpr uint count = slot_bytes + (c_bytes::active ? 1u : 0u);
    // Matches `most(W, form(W, PACK_SUMS))` in `dense.seismic`.
    static constant constexpr bool any_sums =
        (packing_codes<A>::available && packing_codes<A>::sums)
        || (packing_codes<B>::available && packing_codes<B>::sums)
        || (packing_codes<C>::available && packing_codes<C>::sums)
        || (packing_codes<D>::available && packing_codes<D>::sums)
        || (packing_codes<E>::available && packing_codes<E>::sums)
        || (packing_codes<F>::available && packing_codes<F>::sums)
        || (packing_codes<G>::available && packing_codes<G>::sums)
        || (packing_codes<H>::available && packing_codes<H>::sums);
    template <typename W>
    struct of {
        static constant constexpr bool served = packing_codes<W>::available;
        static constant constexpr uint slot =
            packing_serves<W, 1, 0, false>::value ? slot_one :
            packing_serves<W, 2, 0, false>::value ? slot_two :
            packing_serves<W, 2, 0, true>::value ? slot_summed : slot_bytes;
        static constant constexpr uint residue =
            (served && packing_codes<W>::residue != 0) ? WIDTH : 0u;
    };
};

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
    static_assert(SERVED < 256, "an operand serves at most eight weights");
    device const float *tables[8];
    float at(uint run) const {
        float bound = 0.0f;
        PROJECTION_UNROLL
        for (uint i = 0; i < 8; ++i) {
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

// One token's eight columns of a run as codes under a held `scale`, in
// multiples of `step`, and the run's code sum (over its lanes).
inline float packing_code(float4 even, float4 odd, float scale, float step, uint folds, thread float4 &even_codes,
    thread float4 &odd_codes) {
    const float gain = packets::packing_gain(scale) / step;
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

// A held block scale (`packets::packing_held`: non-negative, 8 fraction bits) as 16
// bits, and back.
inline ushort packing_scale_bits(float scale) { return ushort(as_type<uint>(scale) >> 15); }
inline float packing_scale_of(ushort bits) { return as_type<float>(uint(bits) << 15); }
// A token's block scale, held as its bits, over the token's scale: a
// division by a power of two, exact but for a block 2^14 below the token's
// largest.
inline ushort packing_scale_over(ushort held, float scale) {
    return as_type<ushort>(half(packing_scale_of(held) / scale));
}

// The operand pass: one threadgroup of 256 threads per two elements (an
// operand lane's two, 4 tokens: threadgroup `index` holds tokens
// 4 index ..); a thread takes 8 columns of the tokens at a time, four
// neighbouring lanes one 32-column block, and stores each column's two
// elements together. Tokens at or past `m_rows` are zero. In each block's
// phase (`packing_phase`) a token takes its lane (`packing_role`). The block
// scales are first stored as their bits; once the threadgroup has the
// tokens' largest (through `peaks`, `packing_peaks`) their token scales are
// stored and the block scales rewritten over them. CENTRE, FOLDS, SUMS and
// RESIDUE are the centre, the runs a block, whether each run carries its
// code sums and the residue bits of the weights the operand multiplies
// (`packing_shared`), whose steps' columns it is laid out in
// (`packing_codes`); `bounds` the packing bounds of the weights it serves.
template <uint CENTRE, uint FOLDS, bool SUMS, uint RESIDUE, uint BOUNDS, typename In>
inline void packing_operand(thread const In &in, thread const packing_scratch &s,
    thread const packing_bounds<BOUNDS> &bounds, uint m_rows, uint columns, uint index, threadgroup float *peaks,
    uint thread_index, uint sg, uint lane) {
    static_assert(RESIDUE == 0 || FOLDS == 1, "weights with a residue bit scale every 32 columns");
    static_assert(!SUMS || FOLDS == 2, "a run's own code sums are those of two runs a block");
    // The threadgroup's tokens.
    constexpr uint TOKENS = 2u * packing_gemm_lanes;
    typedef metal::conditional_t<(RESIDUE != 0), packing_fields<RESIDUE>, packing_lanes> fields;
    const uint groups = columns / 32u, steps = columns / 8u;
    const uint tile = index / 16u, within = (2u * index) & 31u;
    constexpr uint ENTRY = packing_entry<RESIDUE, SUMS>();
    device uchar *entries = reinterpret_cast<device uchar *>(s.tokens) + (ulong(tile) * groups * 32u + within) * ENTRY;
    float largest[TOKENS];
    PROJECTION_UNROLL
    for (uint t = 0; t < TOKENS; ++t)
        largest[t] = 0.0f;
    for (uint chunk = thread_index; chunk < steps; chunk += 256u) {
        const uint g = chunk / 4u, word = chunk % 4u;
        const uint phase = packing_phase(g, groups);
        // The run of this chunk's 8 columns: runs are of 32 / FOLDS columns.
        const float bound = BOUNDS != 0 ? bounds.at(chunk * FOLDS / 4u) : 0.0f;
        float4 even[TOKENS], odd[TOKENS];
        float3 norms[TOKENS];
        bool finite[TOKENS];
        bool all_finite = true;
        PROJECTION_UNROLL
        for (uint t = 0; t < TOKENS; ++t) {
            even[t] = float4(0.0f);
            odd[t] = float4(0.0f);
            if (TOKENS * index + t < m_rows)
                in.load8(TOKENS * index + t, chunk * 8u, 1.0f, even[t], odd[t]);
            norms[t] = packing_norms(even[t], odd[t], FOLDS);
            finite[t] = norms[t].x < INFINITY;
            all_finite = all_finite && finite[t];
        }
        // A token's run holding a NaN or ±Inf (its sum of magnitudes then
        // NaN or +Inf, as past F32's range, where its scale would be
        // infinite; the same on every lane of the run) is coded as a zero
        // run: its codes and code sum are 0, and the tokens it shares
        // elements with take the codes they take beside a zero run. Its
        // scale is then NaN (below), so its products are NaN, as the dense
        // product's would be. Only a chunk holding such a run takes this
        // branch: a finite one pays a comparison a token and a select a
        // lane's scale, nothing a value.
        if (!all_finite) {
            PROJECTION_UNROLL
            for (uint t = 0; t < TOKENS; ++t) {
                if (!finite[t]) {
                    even[t] = float4(0.0f);
                    odd[t] = float4(0.0f);
                    norms[t] = float3(0.0f);
                }
            }
        }
        float4 packed_even[2], packed_odd[2];
        PROJECTION_UNROLL
        for (uint p = 0; p < 2; ++p) {
            // The element's tokens by lane, their scales and codes.
            float3 lane_norms[packing_gemm_lanes];
            PROJECTION_UNROLL
            for (uint l = 0; l < packing_gemm_lanes; ++l)
                lane_norms[l] = norms[packing_gemm_lanes * p + packing_role(l, phase)];
            float scales[packing_gemm_lanes];
            const float step = packets::packing_lane_scales<fields, FOLDS, (BOUNDS != 0)>(lane_norms, bound, CENTRE, scales);
            float4 codes_even[packing_gemm_lanes], codes_odd[packing_gemm_lanes];
            float sums[packing_gemm_lanes];
            PROJECTION_UNROLL
            for (uint l = 0; l < packing_gemm_lanes; ++l) {
                const uint t = packing_gemm_lanes * p + packing_role(l, phase);
                // A non-finite run's scale: NaN, whose gain is 0
                // (`packets::packing_gain`), coding its zero run 0.
                scales[l] = finite[t] ? scales[l] : NAN;
                sums[l] = packing_code(even[t], odd[t], scales[l], l == 0 ? step : 1.0f, FOLDS, codes_even[l],
                    codes_odd[l]);
            }
            if constexpr (RESIDUE != 0) {
                packed_even[p] = (codes_even[1] * float(1u << fields::top_shift) + codes_even[0]) * fields::element;
                packed_odd[p] = (codes_odd[1] * float(1u << fields::top_shift) + codes_odd[0]) * fields::element;
            } else {
                packed_even[p] = packing_element(codes_even);
                packed_odd[p] = packing_element(codes_odd);
            }
            // The element's scales over what their sums are read over (with
            // residue bits the top's over 2^RESIDUE and the low's over
            // 2^low_shift), and code sums / 64, in token order.
            float factor[packing_gemm_lanes], code_sum[packing_gemm_lanes];
            PROJECTION_UNROLL
            for (uint t = 0; t < packing_gemm_lanes; ++t) {
                const uint l = packing_role(t, phase);
                factor[t] = scales[l];
                if constexpr (RESIDUE != 0)
                    factor[t] *= l == 1u ? fields::top_role : fields::low_role;
                code_sum[t] = sums[l] * 0x1p-6f;
                largest[packing_gemm_lanes * p + t] = metal::max(largest[packing_gemm_lanes * p + t], factor[t]);
            }
            device uchar *entry = entries + (g * 32u + p) * ENTRY;
            // A run's first lane stores its entry: the block's scales and
            // code sums, with two runs a block each run's scales (and with
            // SUMS its code sums), or with residue bits the block's scales
            // and the top token's low plane (and with two its second).
            if constexpr (RESIDUE > 1) {
                const uint2 planes = uint2(packing_plane(codes_even[1], codes_odd[1], 0u, word),
                    packing_plane(codes_even[1], codes_odd[1], 1u, word));
                if ((lane & 3u) == 0)
                    *reinterpret_cast<device uint4 *>(entry) = uint4(uint(packing_scale_bits(factor[0]))
                        | (uint(packing_scale_bits(factor[1])) << 16), planes.x, planes.y, 0u);
            } else if constexpr (RESIDUE != 0) {
                const uint plane = packing_plane(codes_even[1], codes_odd[1], 0u, word);
                if ((lane & 3u) == 0)
                    *reinterpret_cast<device uint2 *>(entry) = uint2(uint(packing_scale_bits(factor[0]))
                        | (uint(packing_scale_bits(factor[1])) << 16), plane);
            } else if constexpr (FOLDS == 2) {
                // The run's two scales, then with SUMS its two code
                // sums.
                if ((lane & 1u) == 0) {
                    device ushort *held = reinterpret_cast<device ushort *>(entry)
                        + (word >> 1) * (SUMS ? 2u : 1u) * packing_gemm_lanes;
                    PROJECTION_UNROLL
                    for (uint t = 0; t < packing_gemm_lanes; ++t) {
                        held[t] = packing_scale_bits(factor[t]);
                        if constexpr (SUMS)
                            held[packing_gemm_lanes + t] = as_type<ushort>(half(code_sum[t]));
                    }
                }
            } else if ((lane & 3u) == 0) {
                // The two scales, then the two code sums, two a word.
                ushort held[2 * packing_gemm_lanes];
                PROJECTION_UNROLL
                for (uint t = 0; t < packing_gemm_lanes; ++t) {
                    held[t] = packing_scale_bits(factor[t]);
                    held[packing_gemm_lanes + t] = as_type<ushort>(half(code_sum[t]));
                }
                PROJECTION_UNROLL
                for (uint w = 0; w < packing_gemm_lanes; ++w)
                    reinterpret_cast<device uint *>(entry)[w] = uint(held[2u * w]) | (uint(held[2u * w + 1u]) << 16);
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
    PROJECTION_UNROLL
    for (uint t = 0; t < TOKENS; ++t)
        largest[t] = simd_max(largest[t]);
    if (lane == 0) {
        PROJECTION_UNROLL
        for (uint t = 0; t < TOKENS; ++t)
            peaks[sg * TOKENS + t] = largest[t];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale[TOKENS];
    PROJECTION_UNROLL
    for (uint t = 0; t < TOKENS; ++t) {
        PROJECTION_UNROLL
        for (uint other = 0; other < 8; ++other)
            largest[t] = metal::max(largest[t], peaks[other * TOKENS + t]);
        scale[t] = packing_scale(largest[t]);
    }
    if (thread_index == 0) {
        PROJECTION_UNROLL
        for (uint t = 0; t < TOKENS; ++t)
            s.token_scales[TOKENS * index + t] = scale[t];
    }
    if ((lane & (FOLDS == 2 ? 1u : 3u)) != 0)
        return;
    for (uint chunk = thread_index; chunk < steps; chunk += 256u) {
        PROJECTION_UNROLL
        for (uint p = 0; p < 2; ++p) {
            // The scales this lane stored: the entry's first two, or with
            // two runs a block its run's pair (with SUMS, before its sums).
            device ushort *held = reinterpret_cast<device ushort *>(entries + ((chunk / 4u) * 32u + p) * ENTRY)
                + (FOLDS == 2 ? ((chunk % 4u) >> 1) * (SUMS ? 2u : 1u) * packing_gemm_lanes : 0u);
            PROJECTION_UNROLL
            for (uint t = 0; t < packing_gemm_lanes; ++t)
                held[t] = packing_scale_over(held[t], scale[packing_gemm_lanes * p + t]);
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
        const uint groups = columns / 32u, full_blocks = groups / 8u, tail = groups & 7u;
        const uint tile = n / 32u, within = n % 32u;
        const typename Weights<W>::located row = w.locate(n);
        float largest = 0.0f;
        for (uint block = lane; block < full_blocks; block += 32u)
            largest = metal::max(largest, coded::reach(row, block));
        if (tail != 0u && lane == 0u)
            largest = metal::max(largest, coded::reach(row, full_blocks, tail));
        const float scale = packing_scale(simd_max(largest));
        if (lane == 0)
            s.weight_scales[n] = scale;
        const float inverse = 1.0f / scale;
        for (uint block = lane; block < full_blocks; block += 32u) {
            typename coded::run run = coded::start(row, block);
            PROJECTION_UNROLL
            for (uint i = 0; i < 8; ++i)
                reinterpret_cast<device metal::vec<half, coded::width> *>(s.weights)[((ulong(tile) * groups
                    + block * 8u + i) * 8u + (within & 7u)) * 4u + (within >> 3)] =
                    metal::vec<half, coded::width>(coded::take(run) * inverse);
        }
        if (tail != 0u && lane == 0u) {
            typename coded::run run = coded::start(row, full_blocks);
            for (uint i = 0; i < tail; ++i)
                reinterpret_cast<device metal::vec<half, coded::width> *>(s.weights)[((ulong(tile) * groups
                    + full_blocks * 8u + i) * 8u + (within & 7u)) * 4u + (within >> 3)] =
                    metal::vec<half, coded::width>(coded::take(run) * inverse);
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
        const uint groups = columns / 32u, full_blocks = groups / 8u, tail = groups & 7u, items = 32u * full_blocks;
        float largest = 0.0f;
        for (uint item = thread_index; item < items; item += 256u) {
            const uint n = tile * 32u + (item & 31u);
            largest = metal::max(largest, coded::reach(w.locate(n), item >> 5));
        }
        if (tail != 0u && thread_index < 32u)
            largest = metal::max(largest, coded::reach(w.locate(tile * 32u + thread_index), full_blocks, tail));
        largest = simd_max(largest);
        if (lane == 0)
            peaks[sg] = largest;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        PROJECTION_UNROLL
        for (uint other = 0; other < 8; ++other)
            largest = metal::max(largest, peaks[other]);
        const float scale = packing_scale(largest);
        if (thread_index < 32u)
            s.weight_scales[tile * 32u + thread_index] = scale;
        const float inverse = 1.0f / scale;
        for (uint item = thread_index; item < items; item += 256u) {
            const uint within = item & 31u, block = item >> 5, n = tile * 32u + within;
            typename coded::run run = coded::start(w.locate(n), block);
            PROJECTION_UNROLL
            for (uint i = 0; i < 8; ++i)
                reinterpret_cast<device metal::vec<half, coded::width> *>(s.weights)[((ulong(tile) * groups
                    + block * 8u + i) * 8u + (within & 7u)) * 4u + (within >> 3)] =
                    metal::vec<half, coded::width>(coded::take(run) * inverse);
        }
        if (tail != 0u && thread_index < 32u) {
            const uint within = thread_index, n = tile * 32u + within;
            typename coded::run run = coded::start(w.locate(n), full_blocks);
            for (uint i = 0; i < tail; ++i)
                reinterpret_cast<device metal::vec<half, coded::width> *>(s.weights)[((ulong(tile) * groups
                    + full_blocks * 8u + i) * 8u + (within & 7u)) * 4u + (within >> 3)] =
                    metal::vec<half, coded::width>(coded::take(run) * inverse);
        }
    }
}

// Simdgroups of a PACK tile launch's threadgroup, each an independent tile:
// the best count on every device measured (1 to 4 on the M4 Pro and M1).
constant constexpr uint packing_simdgroups = 4;

// A simdgroup's fragments: TOKENS operand fragments (16 tokens each)
// against 8 / TOKENS weight fragments (8 weight rows each), 4 x 2 or 2 x 4.
// The best shape differs by device, so it is the launch's. A lane's two
// elements of an operand fragment hold `held` = 4 tokens, each with
// its sum of every weight row it holds (`sums`).
template <uint TOKENS>
struct packing_tile {
    static_assert(TOKENS == 4 || TOKENS == 2, "a packed tile is 4 x 2 or 2 x 4 fragments");
    static constant constexpr uint ti = TOKENS, tj = 8 / TOKENS, held = 2 * packing_gemm_lanes;
    static constant constexpr uint tokens = 8 * packing_gemm_lanes * ti, rows = 8 * tj;
    typedef float sums[ti][tj][held];
};

// F16 `at` of a lane's words of its elements' entries (`packing_entry`),
// two a word.
template <uint WORDS>
inline half packing_entry_half(thread const uint (&words)[WORDS], uint at) {
    const half2 pair = as_type<half2>(words[at / 2u]);
    return pair[at % 2u];
}

// One simdgroup's tile of `packing_tile<TOKENS>`: tokens from m0 (a
// multiple of the tile's tokens) by weight rows from n0 (a multiple of its
// rows). `y[i][j]` receives the sums of weight row n0 + 8 j + y and tokens
// m0 + 2 (8 i + x) .. + 3 (K below 2^20: a normalized sum of
// at most two terms a block stays in F16).
// WEIGHTS_AHEAD loads a block's weight words while the block before it
// multiplies (whether that helps differs by device and shape): the loads
// (`fetch`) issue as the block starts and their words are formed (`take`)
// as it ends, so no arithmetic waits on a load before the block's products;
// a step's operand is always loaded while the step before it multiplies.
// WIDTH is the operand's residue width for weights with residue bits (the
// form's PACK, `packing_shared`).
template <typename W, uint WIDTH, uint TOKENS, uint WEIGHTS_AHEAD>
inline void gemm_packed_sums(thread const Weights<W> &w, thread const packing_scratch &s, uint k, uint m0,
    uint n0, uint lane, thread typename packing_tile<TOKENS>::sums &y) {
    typedef packing_tile<TOKENS> tile;
    constexpr uint TI = tile::ti, TJ = tile::tj, HELD = tile::held;
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
    // An element's entry of a block (`packing_operand`): ENTRY bytes, HALVES
    // F16 and those of a run from `run * RUN_HALVES`. The lane's two
    // elements' entries are read whole, in units of 16 bytes where they fill
    // them, as WORDS words.
    constexpr uint ENTRY = packing_entry<RESIDUE, coded::sums>();
    constexpr uint HALVES = ENTRY / 2u, RUN_HALVES = HALVES / coded::folds, WORDS = 2u * ENTRY / 4u;
    typedef metal::conditional_t<(2u * ENTRY) % 16u == 0u, uint4, uint2> unit;
    constexpr uint UNIT_WORDS = sizeof(unit) / 4u, UNITS = WORDS / UNIT_WORDS;
    device const float4 *operand = reinterpret_cast<device const float4 *>(s.operand);
    device const unit *tokens = reinterpret_cast<device const unit *>(s.tokens);
    // Two weight rows' factors (`packets::block_factors::width` F16 each).
    typedef metal::conditional_t<coded::sums, uint4, uint2> factor_pair;
    device const factor_pair *weights = reinterpret_cast<device const factor_pair *>(s.weights);
    // The tile's first operand fragment of its operand tile (64
    // tokens), and its first 8-row group of its 32-row weight tile.
    const uint tile_m = m0 / (32u * packing_gemm_lanes);
    const uint fragment0 = (m0 % (32u * packing_gemm_lanes)) / (8u * packing_gemm_lanes);
    const uint rows0 = (n0 % 32u) / 8u;
    const uint operand_at = (tile_m * (k / 8u) * 32u + lane) * 2u + fragment0 / 2u;
    // The lane's two elements' entries of a block (elements x and x + 1 of
    // fragment i: x even), in units: 32 elements a block, 8 a fragment.
    const uint tokens_at = (tile_m * groups * 32u + 8u * fragment0 + at.x) / 2u * UNITS;
    const uint weights_at = ((n0 / 32u) * groups * 8u + at.y) * 2u + rows0 / 2u;
    // The centre as a code: the stored codes are F16 subnormals of the same
    // exponent (the operand carries the 2^24), which the matrix unit
    // multiplies exactly.
    const half centre = as_type<half>(ushort(coded::centre));
    simdgroup_float8x8 acc[TI][TJ];
    half folded[TI][TJ][HELD];
    PROJECTION_UNROLL
    for (uint i = 0; i < TI; ++i) {
        PROJECTION_UNROLL
        for (uint j = 0; j < TJ; ++j) {
            PROJECTION_UNROLL
            for (uint t = 0; t < HELD; ++t)
                folded[i][j][t] = 0.0h;
        }
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
    // One copy of the block loop per phase: which lane each token of an
    // element takes (`packing_role`) is then a constant of the copy.
    PROJECTION_UNROLL
    for (uint phase = 0; phase < packing_gemm_lanes; ++phase) {
        const uint first = packing_phase_start(phase, groups);
        const uint last = packing_phase_start(phase + 1u, groups);
        for (uint g = first; g < last; ++g) {
            typename coded::words words[TJ];
            typename coded::fetched fetched[TJ];
            PROJECTION_UNROLL
            for (uint j = 0; j < TJ; ++j) {
                if constexpr (WEIGHTS_AHEAD != 0) {
                    words[j] = next[j];
                    fetched[j] = coded::fetch(w, codes[j], metal::min(g + 1u, groups - 1u));
                } else {
                    words[j] = coded::take(codes[j], coded::fetch(w, codes[j], g));
                }
            }
            uint token[TI][WORDS];
            PROJECTION_UNROLL
            for (uint i = 0; i < TI; ++i) {
                PROJECTION_UNROLL
                for (uint u = 0; u < UNITS; ++u) {
                    const unit loaded = tokens[tokens_at + g * (16u * UNITS) + i * (4u * UNITS) + u];
                    PROJECTION_UNROLL
                    for (uint c = 0; c < UNIT_WORDS; ++c)
                        token[i][u * UNIT_WORDS + c] = loaded[c];
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
            factor_pair weight[TJ / 2];
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
                const uint run = step / RUN;
                PROJECTION_UNROLL
                for (uint i = 0; i < TI; ++i) {
                    // The two elements' tokens' scales of the run, and their
                    // code sums where the run has a min term, in token
                    // order: an element's entry holds a run's from its
                    // `run * RUN_HALVES`; with two runs a block and no min
                    // term the second run's scales are where one run's code
                    // sums are.
                    half scales[HELD], sums[HELD];
                    PROJECTION_UNROLL
                    for (uint e = 0; e < 2; ++e) {
                        PROJECTION_UNROLL
                        for (uint t = 0; t < packing_gemm_lanes; ++t) {
                            const uint from = e * HALVES + run * RUN_HALVES;
                            scales[packing_gemm_lanes * e + t] = packing_entry_half(token[i], from + t);
                            if constexpr (coded::biased && RESIDUE == 0)
                                sums[packing_gemm_lanes * e + t] =
                                    packing_entry_half(token[i], from + packing_gemm_lanes + t);
                        }
                    }
                    if constexpr (RESIDUE != 0) {
                        // The planes of the two elements' top tokens: the
                        // low ones, and with two residue bits the second.
                        constexpr uint ELEMENT = WORDS / 2u;
                        const uint2 low = uint2(token[i][1], token[i][ELEMENT + 1u]);
                        uint2 high = uint2(0u);
                        if constexpr (RESIDUE > 1)
                            high = uint2(token[i][2], token[i][ELEMENT + 2u]);
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
                            // borrow: each element's (low, top) over 2^15.
                            const uint2 split = uint2(int2(v))
                                - ((residue & ((1u << RESIDUE) - 1u)) << fields::top_bit);
                            const half2 parts[2] = {half2(as_type<short2>(split.x)) * 0x1p-15h,
                                half2(as_type<short2>(split.y)) * 0x1p-15h};
                            // The top half's mean deficit rides in the
                            // product with the block's scale.
                            const half deficit = factor * half(fields::deficit);
                            PROJECTION_UNROLL
                            for (uint e = 0; e < 2; ++e) {
                                PROJECTION_UNROLL
                                for (uint t = 0; t < packing_gemm_lanes; ++t) {
                                    const uint l = packing_role(t, phase), held = packing_gemm_lanes * e + t;
                                    const half term = metal::fma(parts[e][l], factor, l == 1u ? deficit : 0.0h);
                                    folded[i][j][held] = metal::fma(scales[held], term, folded[i][j][held]);
                                }
                            }
                        }
                        continue;
                    }
                    PROJECTION_UNROLL
                    for (uint j = 0; j < TJ; ++j) {
                        // The row's factors of the run: its scale and min
                        // term, or without sums of each run the block's.
                        half2 factors;
                        if constexpr (coded::sums) {
                            const half4 both = as_type<half4>((j & 1u) ? weight[j / 2u].zw : weight[j / 2u].xy);
                            factors = run == 0 ? both.xy : both.zw;
                        } else {
                            factors = as_type<half2>(weight[j / 2u][j & 1u]);
                        }
                        const float2 v = reinterpret_cast<thread float2 &>(acc[i][j].thread_elements());
                        // Each element's lane sums over 2^15, bottom lane
                        // first.
                        float2 parts[packing_gemm_lanes];
                        packing_split(v, parts);
                        PROJECTION_UNROLL
                        for (uint e = 0; e < 2; ++e) {
                            PROJECTION_UNROLL
                            for (uint t = 0; t < packing_gemm_lanes; ++t) {
                                const uint held = packing_gemm_lanes * e + t;
                                const half sum = half(parts[packing_role(t, phase)][e]);
                                // A term of at most 1 (normalized factors):
                                // the run's scale and min term, or the run's
                                // scale.
                                half term;
                                if constexpr (coded::biased)
                                    term = metal::fma(sum, factors.x, sums[held] * factors.y);
                                else
                                    term = sum * factors[run];
                                folded[i][j][held] = metal::fma(scales[held], term, folded[i][j][held]);
                            }
                        }
                    }
                }
            }
            if constexpr (WEIGHTS_AHEAD != 0) {
                PROJECTION_UNROLL
                for (uint j = 0; j < TJ; ++j)
                    next[j] = coded::take(codes[j], fetched[j]);
            }
        }
    }
    // The tokens' and the weight rows' scales, once.
    PROJECTION_UNROLL
    for (uint i = 0; i < TI; ++i) {
        float token_scale[HELD];
        PROJECTION_UNROLL
        for (uint t = 0; t < HELD; ++t)
            token_scale[t] = s.token_scales[m0 + packing_gemm_lanes * (8u * i + at.x) + t];
        PROJECTION_UNROLL
        for (uint j = 0; j < TJ; ++j) {
            const float weight_scale = s.weight_scales[n0 + 8u * j + at.y];
            PROJECTION_UNROLL
            for (uint t = 0; t < HELD; ++t)
                y[i][j][t] = float(folded[i][j][t]) * token_scale[t] * weight_scale;
        }
    }
}

// Tile `index` of the `packing_tile<TOKENS>` tiles over `padded`
// tokens (`packing_padded`) by `rows` weight rows: tokens first, so
// neighbouring tiles share their weight rows. (m0, n0); n0 is at or past
// `rows` beyond the last tile.
template <uint TOKENS>
inline uint2 packing_tile_origin(uint index, uint padded) {
    typedef packing_tile<TOKENS> tile;
    const uint across = padded / tile::tokens;
    return uint2((index % across) * tile::tokens, (index / across) * tile::rows);
}

// One simdgroup's `packing_tile<TOKENS>` at tokens m0 by weight rows
// n0 (a tile at or past `m_rows` or `rows` has nothing to do):
// `gemm_packed`'s tile, and that of a launch placing its tiles itself (a
// grouped expert block's tokens, which share their weights only within the
// block). WIDTH is the operand's residue width for weights with residue
// bits.
template <typename W, uint WIDTH, uint TOKENS, uint WEIGHTS_AHEAD, typename Out>
inline void gemm_packed_tile(thread const Out &out, thread const Weights<W> &w, thread const packing_scratch &s,
    uint m_rows, uint m0, uint n0, uint rows, uint k, uint lane) {
    typedef packing_tile<TOKENS> tile;
    if (m0 >= m_rows || n0 >= rows)
        return;
    const ushort2 at = fragment_coordinate(lane);
    typename tile::sums y;
    // Weights without a packed path have no tile (their launch returns
    // before it).
    if constexpr (packing_codes<W>::available)
        gemm_packed_sums<W, WIDTH, TOKENS, WEIGHTS_AHEAD>(w, s, k, m0, n0, lane, y);
    // A lane holds 4 tokens of each operand fragment, a weight row of
    // each weight fragment: a token's result and residual rows are resolved
    // once.
    PROJECTION_UNROLL
    for (uint i = 0; i < tile::ti; ++i) {
        PROJECTION_UNROLL
        for (uint t = 0; t < tile::held; ++t) {
            const uint m = m0 + packing_gemm_lanes * (8u * i + at.x) + t;
            if (m < m_rows) {
                const auto row = row_of(out, m);
                PROJECTION_UNROLL
                for (uint j = 0; j < tile::tj; ++j)
                    row.store(n0 + 8u * j + at.y, y[i][j][t]);
            }
        }
    }
}

// The PACK launch: every simdgroup one `packing_tile<TOKENS>`, tile
// threadgroup x simdgroups + sg of the tiles over `padded` tokens and `rows`
// weight rows (a multiple of 32). The simdgroups of a threadgroup are
// independent: their number sets only how many tiles a core runs together,
// whose best differs by device.
template <typename W, uint WIDTH, uint TOKENS, uint WEIGHTS_AHEAD, typename Out>
inline void gemm_packed(thread const Out &out, thread const Weights<W> &w, thread const packing_scratch &s,
    uint m_rows, uint padded, uint rows, uint k, uint index, uint lane) {
    const uint2 origin = packing_tile_origin<TOKENS>(index, padded);
    gemm_packed_tile<W, WIDTH, TOKENS, WEIGHTS_AHEAD>(out, w, s, m_rows, origin.x, origin.y, rows, k, lane);
}

// One segment of a segmented projection's PACK launch: the segment's `rows`
// weight rows take ceil(padded rows / 1024 / simdgroups) threadgroups
// of the launch (a tile is 1024 token-rows), in segment order. A
// threadgroup `group` at or past them belongs to a later segment, whose first
// is then `group` 0; a segment the operand (coded for RESIDUE residue bits)
// does not serve (SERVES, its rows then any) leaves its threadgroups empty
// (the staged tiles run it).
template <typename W, bool SERVES, uint RESIDUE, uint TOKENS, uint WEIGHTS_AHEAD, typename Out>
inline bool gemm_packed_segment(thread const Out &out, thread const Weights<W> &w, thread const packing_scratch &s,
    uint m_rows, uint padded, uint rows, uint k, thread uint &group, uint simdgroups, uint sg, uint lane) {
    const uint groups = (padded / (16u * packing_gemm_lanes) * (rows / 32u) + simdgroups - 1u) / simdgroups;
    if (group >= groups) {
        group -= groups;
        return false;
    }
    if constexpr (SERVES)
        gemm_packed<W, (RESIDUE != 0 ? RESIDUE : 1), TOKENS, WEIGHTS_AHEAD>(out, w, s, m_rows, padded, rows,
            k, group * simdgroups + sg, lane);
    return true;
}

// F32 of `exchange` in the paired PACK launch per gate simdgroup: a lane's
// sums of its tile.
constexpr uint packing_exchange() {
    return 32u * 8u * 2u * packing_gemm_lanes;
}

// A paired tile of tokens m0 by features n0 (`packing_tile<TOKENS>`
// of each weight) for simdgroups 2 pair and 2 pair + 1: the even one sums
// the gate weights' products, the odd one the up weights' and stores the
// pair with the gate sums, exchanged through `exchange`
// (`packing_exchange` F32 per pair). Every simdgroup of the threadgroup
// reaches its barrier, a tile at or past `m_rows` or `rows` with nothing to
// do. `gate_scratch` and `up_scratch` share the operand and differ in the
// weights' factors.
template <typename G, typename U, uint WIDTH, uint TOKENS, uint WEIGHTS_AHEAD, typename Out>
inline void gemm_packed_paired_tile(thread const Out &out, thread const Weights<G> &gate,
    thread const Weights<U> &up, thread const packing_scratch &gate_scratch, thread const packing_scratch &up_scratch,
    uint m_rows, uint m0, uint n0, uint rows, uint k, uint pair, threadgroup float *exchange, uint sg, uint lane) {
    typedef packing_tile<TOKENS> tile;
    const bool live = m0 < m_rows && n0 < rows;
    const ushort2 at = fragment_coordinate(lane);
    const bool second = (sg % 2u) != 0;
    typename tile::sums y;
    if constexpr (packing_codes<G>::available
        && packing_serves<U, packing_codes<G>::folds, packing_codes<G>::residue, packing_codes<G>::sums>::value) {
        if (live) {
            if (second)
                gemm_packed_sums<U, WIDTH, TOKENS, WEIGHTS_AHEAD>(up, up_scratch, k, m0, n0, lane, y);
            else
                gemm_packed_sums<G, WIDTH, TOKENS, WEIGHTS_AHEAD>(gate, gate_scratch, k, m0, n0, lane, y);
        }
    }
    threadgroup float *slot = exchange + (pair * 32u + lane) * (8u * tile::held);
    if (live && !second) {
        PROJECTION_UNROLL
        for (uint i = 0; i < tile::ti; ++i) {
            PROJECTION_UNROLL
            for (uint j = 0; j < tile::tj; ++j) {
                PROJECTION_UNROLL
                for (uint t = 0; t < tile::held; ++t)
                    slot[(i * tile::tj + j) * tile::held + t] = y[i][j][t];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!live || !second)
        return;
    PROJECTION_UNROLL
    for (uint i = 0; i < tile::ti; ++i) {
        PROJECTION_UNROLL
        for (uint t = 0; t < tile::held; ++t) {
            const uint m = m0 + packing_gemm_lanes * (8u * i + at.x) + t;
            if (m < m_rows) {
                PROJECTION_UNROLL
                for (uint j = 0; j < tile::tj; ++j)
                    out.store_pair(m, n0 + 8u * j + at.y, slot[(i * tile::tj + j) * tile::held + t], y[i][j][t]);
            }
        }
    }
}

// The paired PACK launch (gate and up weights over one operand): simdgroups
// 2 p and 2 p + 1 share tile threadgroup x (simdgroups / 2) + p of the tiles
// over `padded` tokens and `rows` features.
template <typename G, typename U, uint WIDTH, uint TOKENS, uint WEIGHTS_AHEAD, typename Out>
inline void gemm_packed_paired(thread const Out &out, thread const Weights<G> &gate, thread const Weights<U> &up,
    thread const packing_scratch &gate_scratch, thread const packing_scratch &up_scratch, uint m_rows,
    uint padded, uint rows, uint k, uint threadgroup_index, uint simdgroups, threadgroup float *exchange, uint sg,
    uint lane) {
    const uint pair = sg / 2u;
    const uint2 origin = packing_tile_origin<TOKENS>(threadgroup_index * (simdgroups / 2u) + pair, padded);
    gemm_packed_paired_tile<G, U, WIDTH, TOKENS, WEIGHTS_AHEAD>(out, gate, up, gate_scratch, up_scratch,
        m_rows, origin.x, origin.y, rows, k, pair, exchange, sg, lane);
}

} // namespace projection
