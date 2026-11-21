// Seismic native library (Metal): the per-form readers of a bound weight,
// `#include <seismic/forms.h>`. A kernel's product takes a weight in the
// forms the runtime admits it to (`admits`, the element's `_FORM_*`), each
// read here, over the weight's matrix (`Matrix`), in the register shape the
// product consumes:
// - lane codes (`lane`): a chunk's codes two at a time as unsigned integers
//   in the halves of a word, the operand of the matrix GEMV and the
//   token-packing GEMM (PACK);
// - the matrix GEMV's coefficients: a block's local scale codes folded into
//   the codes (`matrix_codes`), a block's runs (`matrix_runs`) or symmetric
//   32 columns (`matrix_symmetric`);
// - the int8 product's codes and coefficients (`int8`);
// - PACK's codes (`pack`, its lane codes `lane<W, pack<W>::offset>`) and a
//   block's factors (`block_factors`).
// Every reader is written against the one description of where a code
// class's bits are (`detail::Fields`, `detail::Entries`, `detail::Grid`,
// <seismic/packets.h>), so a technique one reader takes reaches the others'
// formats alike. Each keeps its own technique: the lane reader its chunk's
// bytes alone, its fetch apart from its spread and the INDEXED choice of its
// addressing; the S8 slot a load and masks or a reinterpret.

#include <seismic/packets.h>

// Register-array loops are fully unrolled (`clang loop unroll(full)`): an
// array indexed by a rolled loop lives in stack memory.

namespace packets {

// Whether the runtime admits weights W to each consumer form
// (`detail::Forms`, `_FORM_*`): the token-packing GEMM (`pack`), the int8
// product (`int8`) and the matrix GEMV folding a block's scale codes, over a
// block's runs or of symmetric 32 columns (`matrix_*`, at most one). None
// for dense weights or an engine's own decoder.
template <typename W>
struct admits {
    static constant constexpr bool pack = false, int8 = false, matrix_codes = false, matrix_runs = false,
                                   matrix_symmetric = false;
};
template <typename P, typename C, typename K, typename F>
struct admits<Packet<P, C, K, F>> {
    static constant constexpr bool pack = F::pack, int8 = F::int8, matrix_codes = F::matrix_codes,
                                   matrix_runs = F::matrix_runs, matrix_symmetric = F::matrix_symmetric;
};

// The packed form of weights W the runtime admits to it (`admits<W>::pack`):
// a code enters as its value + `offset`, centred at `centre`, a block has
// `folds` runs (its packets' coefficient groups), its sums take `residue`
// residue bits, and `sums` where each of a block's two runs folds a min term.
template <typename W>
struct pack;
template <typename P, typename C, typename K, typename F>
struct pack<Packet<P, C, K, F>> {
    static constant constexpr uint centre = F::pack_centre, offset = F::pack_offset, residue = F::pack_residue,
                                   folds = K::groups;
    static constant constexpr bool sums = F::pack_sums;
};

namespace detail {

// ---------------------------------------------------------------------------
// Lane codes: chunk k of a packet is its columns 8 k .. 8 k + 7; a lane that
// holds a chunk takes its codes two at a time, p and p + 4 (`pair`, position
// p < 4), as unsigned integers in the two halves of a word whose values are
// the codes less `zero()`, each below 1024 (a half of the same bits is the
// code times 2^-24, exactly), or as the exact halves of the codes less their
// own zero (`operand`). `at` is a lane's place of chunk k in a located row's
// planes (32-bit offsets from the tensor) and `load<INDEXED>` its words of
// packet g; code planes split a load into its loads alone (`fetch<INDEXED>`,
// its `fields`) and their words (`spread`), which PACK carries ahead apart.
// One reader per code source: linear codes of code planes of every split
// the registry makes (`registry::code_planes`), entering as their stored
// codes or above them (8-bit codes as bytes at their zero, or as pairs at any
// ZERO); a codebook's values, as bytes at 128 (`codebooks::pairs`) or as
// pairs at any ZERO; and grid entries' values, entering at any ZERO. Where
// `bytes()`, a chunk's words are its eight codes as bytes in column order.

// A chunk's eight codes as bytes (columns 0..3 in x, 4..7 in y): pair p is
// byte p of each word.
inline uint chunk_byte_pair(uint2 codes, uint p) {
    return ((codes.x >> (8u * p)) & 0xffu) | (((codes.y >> (8u * p)) & 0xffu) << 16);
}

// A byte lane's pair p (`chunk_byte_pair`) of codes stored at zero OWN,
// entering at ZERO: one add moves both halves, which the form's admission
// keeps non-negative (`producers::Lanes::enter_at`).
template <int OWN, int ZERO>
inline uint byte_lane_pair(uint2 codes, uint p) {
    const uint pair = chunk_byte_pair(codes, p);
    if constexpr (ZERO != OWN)
        return pair + uint(ZERO - OWN) * 0x00010001u;
    else
        return pair;
}

// A pair of codes entering at ZERO less ZERO, as exact halves: one half
// subtraction.
template <int ZERO>
inline half2 pair_operand(uint pair) {
    return as_type<half2>(pair | 0x64006400u) - half2(half(1024 + ZERO));
}

// The zero code a producer's codes enter at by themselves: a linear code's
// stored zero, a codebook's values' declared one (`producers::lane_zero`),
// else 128 (a grid's signed byte values as bytes).
template <typename P, typename C>
struct lane_zero {
    static constexpr int value() { return 128; }
};
template <typename P, uint MASK, int Z>
struct lane_zero<P, Linear<MASK, Z>> {
    static constexpr int value() { return Z; }
};
template <uint LOW, uint HIGH, int K>
struct lane_zero<Planes<LOW, HIGH>, OddCodes<K>> {
    static constexpr int value() { return 2 * K - 1; }
};
template <uint LOW, uint HIGH, uint ID>
struct lane_zero<Planes<LOW, HIGH>, Table<ID>> {
    static constexpr int value() { return producers::lane_zero(producers::TABLE, LOW, HIGH, 0); }
};

// The producer declaration's id (`producers::<KIND>`) of codes interpreted by
// C, which `producers::lane` (the producer declaration) is asked about.
template <typename C>
struct code_of {
    static constexpr uint value() { return producers::ODD; }
};
template <uint MASK, int Z>
struct code_of<Linear<MASK, Z>> {
    static constexpr uint value() { return MASK != 0u ? producers::TWOS : (Z == 0 ? producers::UNSIGNED : producers::OFFSET); }
};
template <uint ID>
struct code_of<Table<ID>> {
    static constexpr uint value() { return producers::TABLE; }
};
template <uint GRID, uint INDEX_BITS, uint LANES, uint SIGNS, bool DELTA, uint DELTA_EXPONENT, uint SIGN_BIT,
    uint SIGN_BITS, uint DELTA_BIT>
struct code_of<Grid<GRID, INDEX_BITS, LANES, SIGNS, DELTA, DELTA_EXPONENT, SIGN_BIT, SIGN_BITS, DELTA_BIT>> {
    static constexpr uint value() { return producers::GRID; }
};
template <>
struct code_of<E4M3Codes> {
    static constexpr uint value() { return producers::E4M3; }
};

// The lane codes of decoders of codes in planes P interpreted by C, entering
// at ZERO (`implemented()`); none by default. A specialization is the decode
// function of the axis values it names; it is dispatched only where the
// producer declaration has them (`producers::lane`), and enters them as the
// declaration states (`producers::lane_bytes`, `producers::lane_zero`), so it
// states none of its own reach or entry, only its arithmetic.
template <typename P, typename C, int ZERO = lane_zero<P, C>::value()>
struct lane_codes {
    static constexpr bool implemented() { return false; }
};

// Linear codes of code planes, whose value zero is code Z: a low plane of
// LOW bits (1, 2 or 4) and, HIGH, a high plane of HIGH bits above them
// (`Fields`), entering at any ZERO from Z on (one add moves both of a pair).
// A single plane of 1- or 2-bit codes takes its operands in place
// (`Fields::lane_operand`).
template <uint LOW, uint HIGH, uint MASK, int Z, int ZERO>
struct lane_codes<Planes<LOW, HIGH>, Linear<MASK, Z>, ZERO> {
    typedef Fields<LOW> low_plane;
    typedef Fields<HIGH> high_plane;
    typedef metal::conditional_t<HIGH == 0u, uint, uint2> place;
    typedef place words;
    typedef place fields;
    static constexpr bool implemented() {
        return producers::lane(code_of<Linear<MASK, Z>>::value(), LOW, HIGH)
            && !producers::lane_bytes(code_of<Linear<MASK, Z>>::value(), LOW, HIGH)
            && ZERO >= producers::lane_zero(code_of<Linear<MASK, Z>>::value(), LOW, HIGH, Z);
    }
    static constexpr int zero() { return ZERO; }
    static constexpr bool bytes() { return false; }
    static constexpr bool byte_words() { return false; }
    template <typename W>
    static place at(thread const Matrix<W> &m, thread const Located &row, uint k) {
        const uint low = low_plane::at(m.codes_at(row, low_plane::unit_bytes, 0u), k);
        if constexpr (HIGH == 0u)
            return low;
        else
            return uint2(low, high_plane::at(m.high_at(row, high_plane::unit_bytes, 0u), k));
    }
    template <bool INDEXED, typename W>
    static fields fetch(thread const Matrix<W> &m, place at, uint g) {
        if constexpr (HIGH == 0u)
            return low_plane::template fetch<INDEXED>(m.base, m.layout.tile, at, g);
        else
            return uint2(low_plane::template fetch<INDEXED>(m.base, m.layout.tile, at.x, g),
                high_plane::template fetch<INDEXED>(m.base, m.layout.tile, at.y, g));
    }
    static words spread(fields codes) {
        if constexpr (HIGH == 0u)
            return low_plane::spread(codes);
        else
            return uint2(low_plane::spread(codes.x), high_plane::spread(codes.y));
    }
    template <bool INDEXED, typename W>
    static words load(thread const Matrix<W> &m, place at, uint g) {
        return spread(fetch<INDEXED>(m, at, g));
    }
    static uint pair(words codes, uint p) {
        uint stored;
        if constexpr (HIGH == 0u)
            stored = low_plane::lane_pair(codes, p);
        else
            stored = low_plane::lane_pair(codes.x, p) | (high_plane::lane_pair(codes.y, p) << LOW);
        if constexpr (ZERO != Z)
            return stored + uint(ZERO - Z) * 0x00010001u;
        else
            return stored;
    }
    static half2 operand(words codes, uint p) {
        if constexpr (HIGH == 0u && LOW <= 2u)
            return low_plane::template lane_operand<Z>(codes, p);
        else
            return pair_operand<ZERO>(pair(codes, p));
    }
};

// One-bit odd codes 2 (c - K) + 1 (-1 and +1 under K = 1), entering at any
// ZERO from their own, 2K - 1: the plane's stored codes (`Unsigned`'s
// lanes), each pair doubled and moved by ZERO + 1 - 2K.
template <int K, int ZERO>
struct lane_codes<Planes<1, 0>, OddCodes<K>, ZERO> : lane_codes<Planes<1, 0>, Unsigned, 0> {
    typedef lane_codes<Planes<1, 0>, Unsigned, 0> stored;
    static constexpr bool implemented() {
        return producers::lane(producers::ODD, 1u, 0u) && !producers::lane_bytes(producers::ODD, 1u, 0u)
            && ZERO >= producers::lane_zero(producers::ODD, 1u, 0u, 2 * K - 1);
    }
    static constexpr int zero() { return ZERO; }
    static uint pair(typename stored::words codes, uint p) {
        return (stored::pair(codes, p) << 1) + uint(ZERO + 1 - 2 * K) * 0x00010001u;
    }
    static half2 operand(typename stored::words codes, uint p) { return pair_operand<ZERO>(pair(codes, p)); }
};

// Linear codes of a plane of bytes (`Fields<8>`), MASK flipped (two's
// complement bytes become offset binary): bytes entering at their zero, or
// pairs at any ZERO (`byte_lane_pair`).
template <uint MASK, int Z, int ZERO>
struct lane_codes<Planes<8, 0>, Linear<MASK, Z>, ZERO> {
    typedef Fields<8> plane;
    typedef uint place;
    typedef uint2 words;
    static constexpr bool implemented() {
        return producers::lane(code_of<Linear<MASK, Z>>::value(), 8u, 0u)
            && producers::lane_bytes(code_of<Linear<MASK, Z>>::value(), 8u, 0u)
            && producers::lane_zero(code_of<Linear<MASK, Z>>::value(), 8u, 0u, Z) == Z;
    }
    static constexpr int zero() { return ZERO; }
    static constexpr bool bytes() { return ZERO == Z; }
    static constexpr bool byte_words() { return bytes(); }
    static uint2 byte_codes(words codes) { return codes; }
    template <typename W>
    static place at(thread const Matrix<W> &m, thread const Located &row, uint k) {
        return plane::at(m.codes_at(row, plane::unit_bytes, 0u), k);
    }
    template <bool INDEXED, typename W>
    static words load(thread const Matrix<W> &m, place at, uint g) {
        return plane::template fetch<INDEXED>(m.base, m.layout.tile, at, g) ^ MASK;
    }
    static uint pair(words codes, uint p) { return byte_lane_pair<Z, ZERO>(codes, p); }
    static half2 operand(words codes, uint p) { return pair_operand<ZERO>(pair(codes, p)); }
};

// A codebook's 4-bit codes (`Fields<4>`): the lane's four bytes of codes
// become its eight values + 128 (`codebooks::pairs`, each code byte's two
// values), a byte each: bytes entering at 128, or pairs at any ZERO
// (`byte_lane_pair`).
template <uint ID, int ZERO>
struct lane_codes<Planes<4, 0>, Table<ID>, ZERO> {
    typedef Fields<4> plane;
    typedef uint place;
    typedef uint2 words;
    static constexpr int own() { return producers::lane_zero(producers::TABLE, 4u, 0u, 0); }
    static constexpr bool implemented() {
        return producers::lane(producers::TABLE, 4u, 0u) && producers::lane_bytes(producers::TABLE, 4u, 0u);
    }
    static constexpr int zero() { return ZERO; }
    static constexpr bool bytes() { return ZERO == own(); }
    static constexpr bool byte_words() { return bytes(); }
    static uint2 byte_codes(words codes) { return codes; }
    template <typename W>
    static place at(thread const Matrix<W> &m, thread const Located &row, uint k) {
        return plane::at(m.codes_at(row, plane::unit_bytes, 0u), k);
    }
    template <bool INDEXED, typename W>
    static words load(thread const Matrix<W> &m, place at, uint g) {
        constant ushort *pairs = codebooks::pairs<ID>::values();
        const uint codes = plane::spread(plane::template fetch<INDEXED>(m.base, m.layout.tile, at, g));
        return uint2(uint(pairs[codes & 0xffu]) | (uint(pairs[(codes >> 8) & 0xffu]) << 16),
            uint(pairs[(codes >> 16) & 0xffu]) | (uint(pairs[codes >> 24]) << 16));
    }
    static uint pair(words codes, uint p) { return byte_lane_pair<own(), ZERO>(codes, p); }
    static half2 operand(words codes, uint p) { return pair_operand<ZERO>(pair(codes, p)); }
};

// Grid codes: a chunk is one entry (`Entries::chunk_entry`), whose eight
// values (`Grid`: its points, signed by its sign bits, then moved by its
// delta) enter at ZERO. A lane's place is its entry's in the packet units of
// the grids plane and of the grid high plane, and its shifts there
// (`EntryChunk`). Without a delta an entry's values + 128 are its words'
// bytes, formed four at a time when it loads: a two's complement point's
// bytes with their sign bits flipped, or its magnitudes signed as bytes
// (`signed_bytes`). With a delta a pair is formed at each step,
// 2^DELTA_EXPONENT (value + 128) moved by the delta, in halves that stay
// below 2^16; a grid whose points are trits (`grids::trits`, each lane's
// value + 1 at bit 2j, a step of 2-bit codes, `Fields<2>`) holds them spread
// at bit DELTA_EXPONENT when it loads, so a pair is a shift, a mask and the
// move, and its table is a quarter of the points' (resident in the constant
// cache).
template <uint BYTES, uint HIGH, uint GRID, uint INDEX_BITS, uint LANES, uint SIGNS, bool DELTA,
    uint DELTA_EXPONENT, uint SIGN_BIT, uint SIGN_BITS, uint DELTA_BIT, int ZERO>
struct lane_codes<Entries<BYTES, HIGH>,
    Grid<GRID, INDEX_BITS, LANES, SIGNS, DELTA, DELTA_EXPONENT, SIGN_BIT, SIGN_BITS, DELTA_BIT>, ZERO> {
    typedef Entries<BYTES, HIGH> E;
    typedef Grid<GRID, INDEX_BITS, LANES, SIGNS, DELTA, DELTA_EXPONENT, SIGN_BIT, SIGN_BITS, DELTA_BIT> G;
    typedef uint3 place;
    static constant constexpr bool trits = DELTA && grids::trits<GRID>::any;
    // The entry's bytes (its values + 128 without a delta; its points with
    // one), its sign bits with a delta (columns 0..3 at bits 0.., 4..7 at
    // bits 16..), and what moves a pair to the values + ZERO, in both halves
    // of a word.
    struct point_words {
        uint2 points;
        uint signs;
        uint offset;
    };
    // Trits: the entry's lanes' values + 1 at bit DELTA_EXPONENT of each
    // lane's 2-bit field (lanes p and p + 4 two halves apart), and the move.
    struct trit_words {
        uint codes;
        uint offset;
    };
    typedef metal::conditional_t<trits, trit_words, point_words> words;
    static constexpr bool implemented() {
        return producers::lane(producers::GRID, 8u * BYTES, HIGH) && !producers::lane_bytes(producers::GRID, 8u * BYTES, HIGH)
            && ZERO >= producers::lane_zero(producers::GRID, 8u * BYTES, HIGH, 0);
    }
    static constexpr int zero() { return ZERO; }
    static constexpr bool bytes() { return false; }
    // Its values + 128 as bytes, without a delta.
    static constexpr bool byte_words() { return !DELTA && ZERO == 128; }
    static uint2 byte_codes(thread const point_words &codes) {
        static_assert(byte_words(), "only a grid without a delta has its values + 128 as bytes: a trit or "
                                    "delta grid's words are no bytes");
        return codes.points;
    }
    template <typename W>
    static place at(thread const Matrix<W> &m, thread const Located &row, uint k) {
        const EntryChunk chunk = E::chunk_of(k);
        return uint3(m.codes_at(row, E::unit_bytes(), 0u) + chunk.offset,
            m.high_at(row, E::high_bytes(), 0u),
            chunk.shift | (chunk.high_shift << 8));
    }
    // Addressed by sums either way (INDEXED: the consumer's choice for
    // planes).
    template <bool INDEXED, typename W>
    static words load(thread const Matrix<W> &m, place at, uint g) {
        const uint entry = E::chunk_entry(m.base + at.x + g * E::unit_bytes() * m.layout.tile,
            m.base + at.y + g * E::high_bytes() * m.layout.tile, EntryChunk{0u, at.z & 0xffu, at.z >> 8});
        words codes;
        if constexpr (trits) {
            const int delta = ((entry >> DELTA_BIT) & 1u) != 0u ? -1 : 1;
            codes.codes = Fields<2>::spread(grids::trits<GRID>::codes()[entry & ((1u << INDEX_BITS) - 1u)])
                << DELTA_EXPONENT;
            codes.offset = uint(ZERO + delta - (1 << DELTA_EXPONENT)) * 0x00010001u;
        } else {
            const uint2 points = G::points::point(entry);
            codes.signs = 0u;
            if constexpr (DELTA) {
                codes.points = points;
                if constexpr (!G::signed_points()) {
                    const uint signs = G::signs(entry);
                    codes.signs = (signs & 15u) | ((signs & 0xf0u) << 12);
                }
                const int delta = ((entry >> DELTA_BIT) & 1u) != 0u ? -1 : 1;
                codes.offset = uint(ZERO + delta - (128 << DELTA_EXPONENT)) * 0x00010001u;
            } else {
                if constexpr (G::signed_points()) {
                    codes.points = points ^ 0x80808080u;
                } else {
                    const uint signs = G::signs(entry);
                    codes.points = uint2(signed_bytes(points.x, signs & 15u), signed_bytes(points.y, signs >> 4));
                }
                codes.offset = uint(ZERO - 128) * 0x00010001u;
            }
        }
        return codes;
    }
    static uint pair(words codes, uint p) {
        if constexpr (trits) {
            return ((codes.codes >> (2u * p)) & ((3u << DELTA_EXPONENT) * 0x00010001u)) + codes.offset;
        } else {
            const uint pair = chunk_byte_pair(codes.points, p);
            if constexpr (DELTA) {
                uint shifted;
                if constexpr (G::signed_points())
                    shifted = pair ^ 0x00800080u;
                else {
                    const uint negative = (codes.signs >> p) & 0x00010001u;
                    shifted = pair + 0x00800080u - ((pair << 1) & ((negative << 16) - negative));
                }
                return (shifted << DELTA_EXPONENT) + codes.offset;
            } else if constexpr (ZERO != 128) {
                return pair + codes.offset;
            } else {
                return pair;
            }
        }
    }
    static half2 operand(words codes, uint p) { return pair_operand<ZERO>(pair(codes, p)); }
};

// The lane codes a declared lane producer (`producers.decl`, rendered as
// `PRODUCERS_LANE`) names: each must have its decode function, so a
// declaration without one fails the build.
template <uint CODE, uint LOW, uint HIGH>
struct declared_lane;
template <uint LOW, uint HIGH>
struct declared_lane<producers::UNSIGNED, LOW, HIGH> {
    typedef lane_codes<Planes<LOW, HIGH>, Unsigned> type;
};
template <uint LOW, uint HIGH>
struct declared_lane<producers::OFFSET, LOW, HIGH> {
    typedef lane_codes<Planes<LOW, HIGH>, Offset<8>> type;
};
template <uint LOW, uint HIGH>
struct declared_lane<producers::TWOS, LOW, HIGH> {
    typedef lane_codes<Planes<LOW, HIGH>, Twos> type;
};
template <uint LOW, uint HIGH>
struct declared_lane<producers::ODD, LOW, HIGH> {
    typedef lane_codes<Planes<LOW, HIGH>, OddCodes<1>> type;
};
template <uint LOW, uint HIGH>
struct declared_lane<producers::TABLE, LOW, HIGH> {
    typedef lane_codes<Planes<LOW, HIGH>, Table<1>> type;
};
#define PACKETS_DETAIL_DECLARED_LANE_HAS_DECODE(CODE, LOW, HIGH) \
    static_assert(declared_lane<CODE, LOW, HIGH>::type::implemented(), "a declared lane producer has a decode function");
PRODUCERS_LANE(PACKETS_DETAIL_DECLARED_LANE_HAS_DECODE)
#undef PACKETS_DETAIL_DECLARED_LANE_HAS_DECODE

// The code planes and interpretation of a decoder's lane codes.
template <typename W>
struct lane_parts;
template <typename P, typename C, typename K, typename F>
struct lane_parts<Packet<P, C, K, F>> {
    typedef P planes;
    typedef C codes;
};

} // namespace detail

// The lane codes of weights W entering at ZERO (by default their codes' own
// zero): `implemented()`, `zero()`, `bytes()`, `byte_words()`, a lane's
// `place` of chunk k in a located row (`at`), its `words` of packet g
// (`load<INDEXED>`; code planes also `fetch<INDEXED>` and `spread`, their
// `fields`), a pair p of them (`pair`, codes p and p + 4 in the halves of a
// word, entering at ZERO), the same as exact halves less the codes' own zero
// (`operand`) and, where `byte_words()`, the chunk's eight as bytes
// (`byte_codes`).
template <typename W, int ZERO = detail::lane_zero<typename detail::lane_parts<W>::planes,
                          typename detail::lane_parts<W>::codes>::value()>
using lane = detail::lane_codes<typename detail::lane_parts<W>::planes, typename detail::lane_parts<W>::codes, ZERO>;

// A chunk's four pairs of lane codes `codes` (lane reader L, `byte_words()`)
// less its zero as exact halves, taken at once: a word of bytes moves four
// codes by two masks and two half subtractions (`pair_operand`), then the
// halves pair up as `operand`'s, codes p and p + 4 (`pairs[p]`).
template <typename L>
inline void byte_operands(thread const typename L::words &codes, thread half2 (&pairs)[4]) {
    const uint2 bytes = L::byte_codes(codes);
    const half2 low_even = detail::pair_operand<L::zero()>(bytes.x & 0x00ff00ffu);
    const half2 low_odd = detail::pair_operand<L::zero()>((bytes.x >> 8) & 0x00ff00ffu);
    const half2 high_even = detail::pair_operand<L::zero()>(bytes.y & 0x00ff00ffu);
    const half2 high_odd = detail::pair_operand<L::zero()>((bytes.y >> 8) & 0x00ff00ffu);
    pairs[0] = half2(low_even.x, high_even.x);
    pairs[1] = half2(low_odd.x, high_odd.x);
    pairs[2] = half2(low_even.y, high_even.y);
    pairs[3] = half2(low_odd.y, high_odd.y);
}

// ---------------------------------------------------------------------------
// The matrix GEMV's coefficients.

// Weights with a matrix GEMV whose products fold a block's local scale codes
// into the left operand: unsigned codes (`lane`) under hierarchical
// coefficients with a minimum, an unsigned local scale code per 32 columns
// and factors per 256, which the runtime admits where a code times a scale
// code stays below 2048 (an exact F16 integer of the operand's 2^-24). `at`
// is a lane's place in its row's planes (fragment column x holds chunk
// x / 2), `load` its words of block g, `pair` its two codes of step s in the
// halves of one word. A block's local codes (`fields`, `block_fields`) give
// packet b's scale and minimum codes (`scale_code`, `min_code`) under the
// block's factors (`block_factors`: scale, then minimum).
template <typename W>
struct matrix_codes {
    static constexpr bool implemented() { return false; }
};
template <typename P, typename FS, int EXPONENT, uint LOCAL_BITS, int BIAS_SIGN, typename F>
struct matrix_codes<Packet<P, detail::Unsigned,
    detail::Hier<FS, 256u, EXPONENT, 0, LOCAL_BITS, detail::LocalUnsigned<LOCAL_BITS>, 32u, true, BIAS_SIGN, 0u>, F>> {
private:
    typedef detail::Hier<FS, 256u, EXPONENT, 0, LOCAL_BITS, detail::LocalUnsigned<LOCAL_BITS>, 32u, true, BIAS_SIGN, 0u> K;

public:
    typedef detail::lane_codes<P, detail::Unsigned> lanes;
    static constexpr bool implemented() { return lanes::implemented(); }
    typedef typename lanes::place place;
    typedef typename lanes::words words;
    typedef typename K::fields::words fields;
    template <typename W>
    static place at(thread const Matrix<W> &m, thread const Located &row, uint x) {
        return lanes::at(m, row, x / 2u);
    }
    template <typename W>
    static words load(thread const Matrix<W> &m, place at, uint g) { return lanes::template load<true>(m, at, g); }
    static uint pair(words codes, uint s) { return lanes::pair(codes, s); }
    static fields block_fields(thread const Located &row, uint block) {
        return K::group_fields(row.row, row.geometry, block);
    }
    static float2 block_factors(thread const Located &row, uint block) {
        const typename K::group_factors factors = K::factors(row.row, row.geometry, block);
        return float2(factors.scale(0), factors.bias(0));
    }
    static uint scale_code(fields f, uint b) { return K::scale_code(K::field_of(f, b), 0u); }
    static uint min_code(fields f, uint b) { return K::min_code(K::field_of(f, b), 0u); }
};

// The weights the matrix GEMV over a block's runs serves: integer codes
// (`lane`) under hierarchical coefficients without a bias, a local scale code
// per GROUP (16 or 32) columns and factors per 256: a lane's chunk lies in
// one group (`group_of`). A block's local codes (`fields`, `block_fields`)
// give run `run`'s scale code of packet b as an exact float (`run_scale`),
// under the block's factor (`block_factor`); `halves` where a code times its
// scale code is an exact F16 operand.
template <typename W>
struct matrix_runs {
    static constexpr bool implemented() { return false; }
};
template <typename P, typename C, typename FS, int EXPONENT, int CODE_EXPONENT, uint LOCAL_BITS, typename L, uint GROUP,
    int BIAS_SIGN, typename F>
struct matrix_runs<Packet<P, C,
    detail::Hier<FS, 256u, EXPONENT, CODE_EXPONENT, LOCAL_BITS, L, GROUP, false, BIAS_SIGN, 0u>, F>> {
private:
    typedef detail::Hier<FS, 256u, EXPONENT, CODE_EXPONENT, LOCAL_BITS, L, GROUP, false, BIAS_SIGN, 0u> K;

public:
    typedef detail::lane_codes<P, C> lanes;
    static constexpr bool implemented() { return lanes::implemented(); }
    // The group of a packet's chunk k.
    static uint group_of(uint k) { return k / (GROUP / 8u); }
    static constexpr bool halves() { return F::matrix_runs_halves(); }
    typedef typename K::fields::words fields;
    static fields block_fields(thread const Located &row, uint block) {
        return K::group_fields(row.row, row.geometry, block);
    }
    static float block_factor(thread const Located &row, uint block) {
        return K::factors(row.row, row.geometry, block).scale(0);
    }
    static float run_scale(fields f, uint b, uint run) { return L::value(K::scale_code(K::field_of(f, b), run)); }
};

// The weights the 32-column matrix GEMV serves: integer codes (`lane`) under
// a direct scale per 32 columns or per a multiple of them, which every
// 32-column block of a group repeats, and, `biased`, a bias per 32 columns
// after it. A row's coefficients are at `supers(row)`; block g's scale and
// bias are `block_scale`, `block_bias`.
template <typename W>
struct matrix_symmetric {
    static constexpr bool implemented() { return false; }
};
template <typename P, typename C, typename S, uint GROUP, bool BIAS, uint STORAGE, int EXPONENT, typename F>
struct matrix_symmetric<Packet<P, C, detail::Direct<S, GROUP, BIAS, STORAGE, EXPONENT>, F>> {
private:
    typedef detail::Direct<S, GROUP, BIAS, STORAGE, EXPONENT> K;

public:
    typedef detail::lane_codes<P, C> lanes;
    static constant constexpr bool biased = BIAS;
    static constexpr bool implemented() { return lanes::implemented() && GROUP % 32u == 0u; }
    static weight_bytes supers(thread const Located &row) { return row.geometry.super_at(row.row, 0ul); }
    static float block_scale(weight_bytes supers, uint g) { return K::at(supers, g).scale(0); }
    static float block_bias(weight_bytes supers, uint g) { return K::at(supers, g).bias(0); }
};

// ---------------------------------------------------------------------------
// The int8 product (on tensor operations): how a lane reads its codes of one
// row and block (`slot`, of the `slot_bytes` of a lane's eight codes at its
// place in a block of `block_bytes` a row, and of `high_slot_bytes` of
// `high_block_bytes` in the plane of high code bits), whether the weights
// have a min term (`biased`), and a row's block scales and biases of the
// eight blocks of one 256-column block (`coefficients`): block i's scale is
// `factor * scale[i]`. `factored`: the scales are integers below 256 and the
// factor is the 256 columns'. The product takes the stored codes
// themselves, so its codes are unsigned 4-bit codes under hierarchical
// coefficients (the local codes integers under the factor) or two's
// complement 8-bit codes under direct ones, a scale per block; none
// otherwise. `interleaved`: a slot's first four positions take the even
// columns, the last four the odd ones (the operand is quantized to match).
template <typename W>
struct int8 {
    static_assert(!SEISMIC_HAS_TENSOR_OPS || !admits<W>::int8, "an admitted weight's codes enter the int8 product");
    static constant constexpr bool biased = false;
    static constant constexpr bool factored = false;
    static constant constexpr bool interleaved = false;
    static void coefficients(thread const Located &, uint, uint, thread float &, thread float (&)[8],
        thread float (&)[8]) {}
};
#if SEISMIC_HAS_TENSOR_OPS
// Unsigned 4-bit codes (`Fields<4>`): a lane's eight codes of one row and
// block are one chunk, its byte x columns 2x (low nibble) and 2x + 1;
// value = factor * scale code * code + bias factor * minimum code.
template <typename FS, int EXPONENT, uint LOCAL_BITS, bool MIN, int BIAS_SIGN, typename F>
struct int8<Packet<detail::Planes<4, 0>, detail::Unsigned,
    detail::Hier<FS, 256u, EXPONENT, 0, LOCAL_BITS, detail::LocalUnsigned<LOCAL_BITS>, 32u, MIN, BIAS_SIGN, 0u>, F>> {
private:
    typedef detail::Hier<FS, 256u, EXPONENT, 0, LOCAL_BITS, detail::LocalUnsigned<LOCAL_BITS>, 32u, MIN, BIAS_SIGN, 0u> K;
    typedef detail::Fields<4> plane;

public:
    // Admitted where a factor of at most 16 bits times a byte code is exact
    // in F32.
    static constant constexpr bool biased = MIN;
    static constant constexpr uint block_bytes = plane::unit_bytes;
    static constant constexpr bool interleaved = true;
    static constant constexpr uint slot_bytes = plane::unit_bytes / 4u;
    static constant constexpr uint high_block_bytes = 0, high_slot_bytes = 0;
    static void slot(weight_bytes at, weight_bytes, thread char4 &first, thread char4 &second) {
        uint even, odd;
        split_nibbles(*slab::as<uint>(at), even, odd);
        first = as_type<char4>(even);
        second = as_type<char4>(odd);
    }
    // A block's scale is factor * scale code, exact in F32 (a 16-bit
    // factor's significand times a code of at most 8 bits): the factor is
    // the 256 columns' and the scales are the integer codes.
    static constant constexpr bool factored = true;
    static void coefficients(thread const Located &row, uint block, uint count, thread float &factor,
        thread float (&scale)[8], thread float (&bias)[8]) {
        typename K::run run = K::start(row.row, row.geometry, block * K::packets);
        factor = run.factor.scale(0);
        // A block's bias, bias factor * minimum code, is exact in F32.
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < 8; ++i) {
            const uint field = K::next(run);
            scale[i] = i < count ? float(K::scale_code(field, 0u)) : 0.0f;
            bias[i] = (MIN && i < count) ? run.factor.bias(0) * float(K::min_code(field, 0u)) : 0.0f;
        }
    }
};
// Two's complement 8-bit codes (`Fields<8>`), a scale per block: a lane's
// eight codes of one row and block are one chunk, in column order.
template <typename S, uint STORAGE, typename F>
struct int8<Packet<detail::Planes<8, 0>, detail::Twos, detail::Direct<S, 32u, false, STORAGE, 0>, F>> {
private:
    typedef detail::Direct<S, 32u, false, STORAGE, 0> K;
    typedef detail::Fields<8> plane;

public:
    static constant constexpr bool biased = false;
    static constant constexpr uint block_bytes = plane::unit_bytes;
    static constant constexpr bool interleaved = false;
    static constant constexpr uint slot_bytes = plane::unit_bytes / 4u;
    static constant constexpr uint high_block_bytes = 0, high_slot_bytes = 0;
    static void slot(weight_bytes at, weight_bytes, thread char4 &first, thread char4 &second) {
        const uint2 words = *slab::as<uint2>(at);
        first = as_type<char4>(words.x);
        second = as_type<char4>(words.y);
    }
    static constant constexpr bool factored = false;
    static void coefficients(thread const Located &row, uint block, uint count, thread float &factor,
        thread float (&scale)[8], thread float (&bias)[8]) {
        weight_bytes supers = row.geometry.super_at(row.row, 0ul);
        factor = 1.0f;
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < 8; ++i) {
            scale[i] = i < count ? K::at(supers, block * 8u + i).scale(0) : 0.0f;
            bias[i] = 0.0f;
        }
    }
};
#endif

// ---------------------------------------------------------------------------
// PACK's block factors.

namespace detail {

// A block's factors of coefficients K of codes centred at CENTRE whose value
// is zero at OFFSET, its runs K's groups of a packet: a run's factors are its
// scale 2^15 / 64 and, where the block has a min term (`biased`), its min
// term (CENTRE - OFFSET) scale + bias. A block of one run has its two (or its
// scale and 0); a block of two runs has its runs' scales, or with a min term
// (`sums`) each run's two in turn (`width` 4, `entry`).
template <typename K, uint CENTRE, uint OFFSET>
struct packing_terms {
    static constant constexpr bool biased = K::biased || CENTRE != OFFSET;
    static constant constexpr bool sums = K::groups == 2 && biased;
    static constant constexpr uint width = sums ? 4u : 2u;
    typedef metal::vec<float, width> entry;
    // The difference CENTRE - OFFSET.
    static constexpr float shift() { return float(int(CENTRE) - int(OFFSET)); }
    // The min term of group `g` from the coefficients `k` of its packet;
    // none where the block has none.
    static float min_term(thread const typename K::coefficients &k, uint g) {
        if constexpr (biased)
            return shift() * k.scale(g) + k.bias(g);
        else
            return 0.0f;
    }
    // A block's factors from the coefficients `k` of its packet.
    static entry of(thread const typename K::coefficients &k) {
        constexpr float SCALE = 32768.0f / 64.0f;
        if constexpr (sums)
            return float4(k.scale(0) * SCALE, min_term(k, 0), k.scale(1) * SCALE, min_term(k, 1));
        else if constexpr (K::groups == 2)
            return float2(k.scale(0), k.scale(1)) * SCALE;
        else
            return float2(k.scale(0) * SCALE, min_term(k, 0));
    }
};

// The run of a 256-column block's factors (`start`, `take` each packet's in
// turn) and a bound on what the fold multiplies (`reach`: it takes a sum of
// at most 1 of the first factor and a code sum / 64 below 128 of the second,
// so 512 |scale| + 128 |min term| of each run bounds it). Implemented for
// the run one factor group's or eight packets'; a power of two scaling the
// codes is in their scales (`Direct`, `Hier`). None by default.
template <typename K, uint CENTRE, uint OFFSET, typename P>
struct packing_factors : packing_terms<K, CENTRE, OFFSET> {
    static constant constexpr bool implemented = false;
};

// Hierarchical coefficients: a block's factors from the factor group's run
// (`Hier::run`), each packet's field from the scales plane or, where its
// grid entries (in planes P) carry it, from its entries; bounded by the
// local codes' largest magnitude under the group's factors, with a signed
// bias's offset beside the min term's shift, alike for each run.
template <typename FS, int EXPONENT, int CODE_EXPONENT, uint LOCAL_BITS, typename L, uint GROUP, bool MIN,
    int BIAS_SIGN, uint CARRIED_BIT, typename SB, uint CENTRE, uint OFFSET, typename P>
struct packing_factors<Hier<FS, 256u, EXPONENT, CODE_EXPONENT, LOCAL_BITS, L, GROUP, MIN, BIAS_SIGN, CARRIED_BIT, SB>,
    CENTRE, OFFSET, P>
    : packing_terms<Hier<FS, 256u, EXPONENT, CODE_EXPONENT, LOCAL_BITS, L, GROUP, MIN, BIAS_SIGN, CARRIED_BIT, SB>,
          CENTRE, OFFSET> {
    typedef Hier<FS, 256u, EXPONENT, CODE_EXPONENT, LOCAL_BITS, L, GROUP, MIN, BIAS_SIGN, CARRIED_BIT, SB> K;
    typedef packing_terms<K, CENTRE, OFFSET> terms;
    static constant constexpr bool implemented = true;
    struct run {
        typename K::run fields;
        Located at;
        uint p;
    };
    static run start(thread const Located &row, uint block) {
        return run{K::start(row.row, row.geometry, block * K::packets), row, block * K::packets};
    }
    static typename terms::entry take(thread run &state) {
        uint field;
        if constexpr (K::carries)
            field = K::template carried<P>(P::load(state.at.row, state.at.geometry, state.p));
        else
            field = K::next(state.fields);
        ++state.p;
        return terms::of(K::form(field, state.fields.factor));
    }
    static float reach(thread const Located &row, uint block) {
        const typename K::group_factors factors = K::factors(row.row, row.geometry, block);
        const float offset = K::signed_bias() ? power2<SB::exponent() - CODE_EXPONENT>() : 0.0f;
        const float scale = (512.0f + 128.0f * (metal::abs(terms::shift()) + offset)) * metal::abs(factors.scale(0));
        return float(L::largest) * (MIN ? metal::fma(128.0f, metal::abs(factors.bias(0)), scale) : scale);
    }
    static float reach(thread const Located &row, uint block, uint) { return reach(row, block); }
};

// Direct coefficients: a block's factors from its eight packets' own,
// bounded by the largest of them.
template <typename S, uint GROUP, bool BIAS, uint STORAGE, int EXPONENT, uint CENTRE, uint OFFSET, typename P>
struct packing_factors<Direct<S, GROUP, BIAS, STORAGE, EXPONENT>, CENTRE, OFFSET, P>
    : packing_terms<Direct<S, GROUP, BIAS, STORAGE, EXPONENT>, CENTRE, OFFSET> {
    typedef Direct<S, GROUP, BIAS, STORAGE, EXPONENT> K;
    typedef packing_terms<K, CENTRE, OFFSET> terms;
    static_assert(256u % GROUP == 0u, "a 256-column block holds whole groups");
    static constant constexpr bool implemented = true;
    // A 256-column block's share of the supers plane, from its first
    // group's scale: a packet's place in it is its index in the block.
    static weight_bytes block_supers(thread const Located &row, uint block) {
        return row.geometry.super_at(row.row, 0ul) + ulong(K::scale_offset(block * (256u / GROUP)));
    }
    // The block's supers and the next packet taken.
    struct run {
        weight_bytes supers;
        uint p;
    };
    static run start(thread const Located &row, uint block) { return run{block_supers(row, block), 0u}; }
    static typename terms::entry take(thread run &state) { return terms::of(K::at(state.supers, state.p++)); }
    static float packet_reach(weight_bytes supers, uint i) {
        const typename K::coefficients k = K::at(supers, i);
        float largest = 0.0f;
        _Pragma("clang loop unroll(full)")
        for (uint g = 0; g < K::groups; ++g) {
            const float scale = metal::abs(k.scale(g));
            largest = metal::max(largest, terms::biased
                ? metal::fma(128.0f, metal::abs(terms::min_term(k, g)), 512.0f * scale)
                : scale);
        }
        return largest;
    }
    static float reach(thread const Located &row, uint block) {
        weight_bytes supers = block_supers(row, block);
        // Without a min term the bound is 512 times the largest scale,
        // exactly: a power of two scales every term alike.
        float largest = 0.0f;
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < 8; ++i)
            largest = metal::max(largest, packet_reach(supers, i));
        return terms::biased ? largest : 512.0f * largest;
    }
    static float reach(thread const Located &row, uint block, uint count) {
        weight_bytes supers = block_supers(row, block);
        float largest = 0.0f;
        for (uint i = 0; i < count; ++i)
            largest = metal::max(largest, packet_reach(supers, i));
        return terms::biased ? largest : 512.0f * largest;
    }
};

} // namespace detail

// The factors of a block of weights W whose codes are centred at CENTRE and
// zero at OFFSET (`pack<W>`), its runs its packets' coefficient groups:
// `biased` (a min term), `sums` (two runs, each with its min term), `width`
// factors an `entry`; a packet's (`of`); `implemented` for the run of a
// 256-column block a row (`run`, `start(row, block)`, `take(run)`) and its
// bound (`reach(row, block)`).
template <typename W, uint CENTRE, uint OFFSET>
struct block_factors {
    static constant constexpr bool implemented = false;
};
template <typename P, typename C, typename K, typename F, uint CENTRE, uint OFFSET>
struct block_factors<Packet<P, C, K, F>, CENTRE, OFFSET> : detail::packing_factors<K, CENTRE, OFFSET, P> {
    static typename detail::packing_terms<K, CENTRE, OFFSET>::entry of(
        thread const typename Packet<P, C, K, F>::packet &k) {
        return detail::packing_terms<K, CENTRE, OFFSET>::of(k.coefficients);
    }
};

} // namespace packets
