// Seismic native library (Metal): packet decoders for weights stored in the
// `rows16` layout, and weight slot bindings. `#include <seismic/packets.h>`.
// Dense element types are `element` (<seismic/element.h>).
//
// A packet is 32 consecutive logical elements of one weight row: its slice of
// each row plane (its codes in the low code plane, their high bits in the
// high plane, or its four grid entries; its local coefficient codes and its
// factors). A packet is consumed in four 8-element sub-steps. Each sub-step
// decodes its codes as exact small integers in two float4s, `even` (elements
// 0,2,4,6) and `odd` (1,3,5,7), by splitting code bytes on nibble boundaries:
// no per-element shifts, no activation prescale, and every decoded value is an
// exact integer, so no rounding or subnormal case exists in the decode. Code
// bytes become floats without integer conversions (`byte_floats`): the M = 1
// projections are bound by the decode's instruction issue, and an integer
// conversion costs several times a half subtraction.
//
// A packet's logical value is `scale(g) * (code - zero code)`, plus a
// stored bias(g), for its coefficient group g, dequantized with one F32
// rounding: as product operands (`values`, where a zero is +0) or exactly
// (`exact_values`, `pair_values`).
//
// Weight tensors are bound to slots before this file is included:
//     #define KERNEL_W0 SEISMIC_GATE_WEIGHT
// which defines `packets::W0` (the decoder of the bound representation) and
// `KERNEL_W0_LAYOUT(k)` (its `Rows16` geometry for rows of `k` logical
// values). Slots W0..W3 exist. A bound weight is dense (f32, bf16, f16) or
// packed in the rows16 or rows32 layout.
//
// The bound weights are read in one address space (<seismic/slab.h>): a
// host-written port's where any slot is bound to one, so a record the host
// landed is read as the host last wrote it. `weight_bytes` addresses a
// weight's bytes; `slab::as` reads them as another type.

#include <seismic/element.h>
#include <seismic/slab.h>
#include <seismic/tables.h>

namespace packets {

// A launch that runs only while nothing lands in its records defines
// PACKETS_PLAIN_RECORDS before this file: plain loads read them.
#ifdef PACKETS_PLAIN_RECORDS
typedef slab::space<false> space;
#else
typedef slab::space<ELEMENT_HAS(KERNEL_W0, _HOST_WRITTEN) || ELEMENT_HAS(KERNEL_W1, _HOST_WRITTEN)
    || ELEMENT_HAS(KERNEL_W2, _HOST_WRITTEN) || ELEMENT_HAS(KERNEL_W3, _HOST_WRITTEN)> space;
#endif
typedef space::pointer<const uchar> weight_bytes;

// ---------------------------------------------------------------------------
// Weight row geometry. `stride` is the byte distance between rows; plane
// offsets are within a row. Planes a representation lacks are unused.
//
// `tile` is the rows of one row tile: 1 for `rows16` and dense rows, 32 for
// `rows32`, whose code planes are interleaved across the tile (unit `u` of
// the tile's row `r` follows unit `u` of row `r - 1`) and whose coefficient
// planes hold the tile's rows one after another, each row's bytes contiguous
// (`scales_row`, `supers_row` bytes a row). A row is then addressed by its
// tile's base and `r`, which the geometry of that row carries (`of`): a code
// plane address is `at(row, plane, unit bytes, unit)`, a coefficient address
// `scale_at(row, offset)` or `super_at(row, offset)`.

struct Rows16 {
    ulong stride;
    ulong codes;    // the low code plane, the grid entries or the values (dense)
    ulong high;     // the high code plane
    ulong scales;   // the local coefficient codes
    ulong supers;   // the direct coefficients or the factors
    uint tile;
    uint r;
    ulong scales_row;
    ulong supers_row;

    // Row `n`'s base (its tile's first row) from the tensor's `base`.
    weight_bytes base_of(weight_bytes base, ulong n) const {
        return base + (n - n % tile) * stride;
    }
    // This geometry for row `n`.
    Rows16 of(ulong n) const {
        Rows16 row = *this;
        row.r = uint(n % tile);
        return row;
    }
    // Unit `unit` (of `bytes` bytes) of the code plane at row offset `plane`.
    // A tile's planes are below 4 GB, so the offset inside one is 32-bit
    // arithmetic, and the row's share (everything but the unit's step) is
    // one loop-invariant address: 64-bit index arithmetic on every load
    // costs a decode lane several instructions a packet.
    weight_bytes at(weight_bytes row, ulong plane, ulong bytes, ulong unit) const {
        return (row + plane * tile + ulong(r * uint(bytes))) + ulong(uint(unit) * (tile * uint(bytes)));
    }
    // Byte `offset` of this row's share of the scales and of the supers
    // plane.
    weight_bytes scale_at(weight_bytes row, ulong offset) const {
        return (row + scales * tile + ulong(r * uint(scales_row))) + ulong(uint(offset));
    }
    weight_bytes super_at(weight_bytes row, ulong offset) const {
        return (row + supers * tile + ulong(r * uint(supers_row))) + ulong(uint(offset));
    }
};

// The eight WIDTH-bit codes (WIDTH 1 or 2) of an 8 WIDTH-bit field, codes
// 0,2,4,6 in bytes 0..3 of `even` and 1,3,5,7 in those of `odd`, at bit AT
// of each byte: the field's halves move to bits 0 and 16, each half's two
// quarters to two bytes, and a byte's two codes apart.
template <uint WIDTH, uint AT>
inline void split_fields(uint field, thread uint &even, thread uint &odd) {
    static_assert(WIDTH == 1u || WIDTH == 2u, "a byte holds two codes of a quarter field");
    constexpr uint code = (1u << WIDTH) - 1u;
    uint t = (field | (field << (16u - 4u * WIDTH))) & (((1u << (4u * WIDTH)) - 1u) * 0x00010001u);
    t = (t | (t << (8u - 2u * WIDTH))) & (((1u << (2u * WIDTH)) - 1u) * 0x01010101u);
    even = (t & (code * 0x01010101u)) << AT;
    if constexpr (AT >= WIDTH)
        odd = (t & ((code << WIDTH) * 0x01010101u)) << (AT - WIDTH);
    else
        odd = (t & ((code << WIDTH) * 0x01010101u)) >> (WIDTH - AT);
}

inline void split_nibbles(uint word, thread uint &even, thread uint &odd) {
    even = word & 0x0f0f0f0fu;
    odd = (word >> 4) & 0x0f0f0f0fu;
}

// The four bytes of `word` as exact floats, less `offset - 1024`: alternate
// bytes become the low mantissa bits of a half2 whose exponent makes each
// lane 1024 + byte, and one half subtraction removes `offset`.
inline float4 byte_floats(uint word, half offset) {
    half2 even = as_type<half2>((word & 0x00ff00ffu) | 0x64006400u) - half2(offset);
    half2 odd = as_type<half2>(((word >> 8) & 0x00ff00ffu) | 0x64006400u) - half2(offset);
    return float4(even.x, odd.x, even.y, odd.y);
}
inline float4 unsigned_bytes(uint word) { return byte_floats(word, 1024.0h); }

// Two codes (each below 1024; the second in bits 16..) as exact floats, the
// same way.
inline float2 code_pair(uint two, half offset) {
    return float2(as_type<half2>(two | 0x64006400u) - half2(offset));
}

// 2^N, exactly.
template <int N>
constexpr float power2() {
    return N >= 0 ? float(1u << uint(N)) : 1.0f / float(1u << uint(-N));
}

// ---------------------------------------------------------------------------
// Decoders are composed from the registry's decode axes (`registry::decode_axes`,
// emitted as each bound element's `_CODE*` and `_COEFF*` macros): the code
// planes a packet's codes are stored in (`P`), the codes' interpretation
// (`C`), and the coefficients that scale them (`K`), with the consumer forms
// it is admitted to (`F`, `_FORM_*`). `Packet<P, C, K, F>` is a
// decoder of the uniform interface every reader takes: `load` a packet, and
// its step's eight values (`values`, `even` lanes 0,2,4,6 and `odd` 1,3,5,7)
// or two of them (`pair_values`), each with one F32 rounding.

// A plane of WIDTH-bit codes (WIDTH 1 or 2): a packet's codes fill WIDTH words
// (`words`, `load`ed from unit p of the code plane at row offset `plane`),
// step s's eight are their bits [8 WIDTH s, 8 WIDTH (s + 1)) (`step`), and
// its codes 2j and 2j + 1 the 2 WIDTH bits from bit 2 WIDTH j of those
// (`pair`).
template <uint WIDTH>
struct Fields {
    static_assert(WIDTH == 1u || WIDTH == 2u, "a packet's codes fill one or two words");
    typedef metal::conditional_t<WIDTH == 1u, uint, uint2> words;
    static words load(weight_bytes row, Rows16 layout, ulong plane, uint p) {
        return *slab::as<words>(layout.at(row, plane, 4ul * WIDTH, p));
    }
    static uint step(words w, uint s) {
        if constexpr (WIDTH == 1u)
            return (w >> (8u * s)) & 0xffu;
        else
            return (w[s >> 1] >> (16u * (s & 1u))) & 0xffffu;
    }
    // Always inlined: as a call, the batched GEMV of 4 + 1-bit codes
    // compiles to other code (M4 Pro, q5g32s `dense_output_batch`, +160 B).
    __attribute__((always_inline)) static uint pair(words w, uint s, uint j) {
        if constexpr (WIDTH == 1u)
            return (w >> (8u * s + 2u * j)) & 3u;
        else
            return (w[s >> 1] >> (16u * (s & 1u) + 4u * j)) & 0xfu;
    }
};

// Code planes (`registry::code_planes`): a low plane of LOW bits a code and a
// high plane of HIGH. `floats` yields a step's codes, their bytes with `mask`
// flipped, as exact floats less `offset - 1024`; `pair` the step's codes 2j
// and 2j + 1, the second in bits 16.., with `mask` flipped alike; `nibbles`
// the step's eight 4-bit codes as one word and `nibble_pair` its byte j (two
// codes), the indices a codebook reads.

// The words of a packet's codes below a nibble: its low plane's
// (`Fields<LOW>`) and, HIGH, its high plane's (`Fields<HIGH>`).
template <uint LOW, uint HIGH>
struct FieldWords {
    typename Fields<LOW>::words low;
    typename Fields<HIGH>::words high;
};
template <uint LOW>
struct FieldWords<LOW, 0u> {
    typename Fields<LOW>::words low;
};

// Codes below a nibble: a plane of LOW-bit codes (`Fields<LOW>`) and, HIGH,
// a plane of their HIGH high bits (`Fields<HIGH>`), at bit LOW of a code.
template <uint LOW, uint HIGH>
struct Planes {
    static_assert(LOW < 4u, "codes below a nibble");
    typedef Fields<LOW> low_plane;
    typedef Fields<HIGH> high_plane;
    typedef FieldWords<LOW, HIGH> words;
    static words load(weight_bytes row, Rows16 layout, uint p) {
        words w;
        w.low = low_plane::load(row, layout, layout.codes, p);
        if constexpr (HIGH != 0u)
            w.high = high_plane::load(row, layout, layout.high, p);
        return w;
    }
    template <uint MASK>
    static void floats(thread const words &w, uint step, half offset, thread float4 &even, thread float4 &odd) {
        uint e, o;
        split_fields<LOW, 0>(low_plane::step(w.low, step), e, o);
        if constexpr (HIGH != 0u) {
            uint he, ho;
            split_fields<HIGH, LOW>(high_plane::step(w.high, step), he, ho);
            e |= he;
            o |= ho;
        }
        even = byte_floats(e ^ MASK, offset);
        odd = byte_floats(o ^ MASK, offset);
    }
    template <uint MASK>
    static uint pair(thread const words &w, uint step, uint j) {
        uint b = low_plane::pair(w.low, step, j);
        if constexpr (HIGH != 0u) {
            uint h = high_plane::pair(w.high, step, j);
            return ((b & ((1u << LOW) - 1u)) | ((h & ((1u << HIGH) - 1u)) << LOW)
                       | (((b >> LOW) | ((h >> HIGH) << LOW)) << 16))
                ^ (MASK & 0x00ff00ffu);
        } else {
            return ((b & ((1u << LOW) - 1u)) | ((b >> LOW) << 16)) ^ (MASK & 0x00ff00ffu);
        }
    }
};

template <>
struct Planes<4, 0> {
    struct words {
        uint4 low;
    };
    static words load(weight_bytes row, Rows16 layout, uint p) {
        words w;
        w.low = *slab::as<uint4>(layout.at(row, layout.codes, 16ul, p));
        return w;
    }
    template <uint MASK>
    static void floats(thread const words &w, uint step, half offset, thread float4 &even, thread float4 &odd) {
        uint e, o;
        split_nibbles(w.low[step], e, o);
        even = byte_floats(e ^ MASK, offset);
        odd = byte_floats(o ^ MASK, offset);
    }
    template <uint MASK>
    static uint pair(thread const words &w, uint step, uint j) {
        uint b = (w.low[step] >> (8u * j)) & 0xffu;
        return ((b & 15u) | ((b >> 4) << 16)) ^ (MASK & 0x00ff00ffu);
    }
    static uint nibbles(thread const words &w, uint step) { return w.low[step]; }
    static uint nibble_pair(thread const words &w, uint step, uint j) { return (w.low[step] >> (8u * j)) & 0xffu; }
};

// Nibbles under a plane of their HIGH high bits (`Fields<HIGH>`), at bit 4
// of a code.
template <uint HIGH>
struct Planes<4, HIGH> {
    typedef Fields<HIGH> high_plane;
    struct words {
        uint4 low;
        typename high_plane::words high;
    };
    static words load(weight_bytes row, Rows16 layout, uint p) {
        words w;
        w.low = *slab::as<uint4>(layout.at(row, layout.codes, 16ul, p));
        w.high = high_plane::load(row, layout, layout.high, p);
        return w;
    }
    template <uint MASK>
    static void floats(thread const words &w, uint step, half offset, thread float4 &even, thread float4 &odd) {
        uint e, o, he, ho;
        split_nibbles(w.low[step], e, o);
        split_fields<HIGH, 4>(high_plane::step(w.high, step), he, ho);
        even = byte_floats((e | he) ^ MASK, offset);
        odd = byte_floats((o | ho) ^ MASK, offset);
    }
    template <uint MASK>
    static uint pair(thread const words &w, uint step, uint j) {
        uint b = (w.low[step] >> (8u * j)) & 0xffu;
        uint h = high_plane::pair(w.high, step, j);
        return ((b & 15u) | ((h & ((1u << HIGH) - 1u)) << 4) | (((b >> 4) | ((h >> HIGH) << 4)) << 16))
            ^ (MASK & 0x00ff00ffu);
    }
};

template <>
struct Planes<8, 0> {
    struct words {
        uint4 first;
        uint4 second;
    };
    static words load(weight_bytes row, Rows16 layout, uint p) {
        words w;
        const auto codes = slab::as<uint4>(layout.at(row, layout.codes, 32ul, p));
        w.first = codes[0];
        w.second = codes[1];
        return w;
    }
    template <uint MASK>
    static void floats(thread const words &w, uint step, half offset, thread float4 &even, thread float4 &odd) {
        uint4 half_words = step < 2 ? w.first : w.second;
        uint base = (step & 1u) * 2u;
        float4 a = byte_floats(half_words[base] ^ MASK, offset);
        float4 b = byte_floats(half_words[base + 1] ^ MASK, offset);
        even = float4(a.x, a.z, b.x, b.z);
        odd = float4(a.y, a.w, b.y, b.w);
    }
    template <uint MASK>
    static uint pair(thread const words &w, uint step, uint j) {
        uint4 half_words = step < 2 ? w.first : w.second;
        uint word = half_words[(step & 1u) * 2u + (j >> 1)];
        uint flipped = (word ^ MASK) >> ((j & 1u) * 16u);
        return (flipped & 0xffu) | ((flipped & 0xff00u) << 8);
    }
};

// The bits `[bit, bit + WIDTH)` (WIDTH < 32, `bit` a multiple of WIDTH)
// of consecutive little-endian words `w` held in registers: one word where
// WIDTH divides 32, so that no field spans two, else the two words it may
// span.
template <uint WIDTH, typename V>
inline uint word_bits(V w, uint bit) {
    const uint word = bit >> 5, shift = bit & 31u;
    if constexpr (32u % WIDTH == 0u)
        return (w[word] >> shift) & ((1u << WIDTH) - 1u);
    else
        return ((w[word] >> shift) | (shift + WIDTH > 32u ? w[word + 1u] << (32u - shift) : 0u))
            & ((1u << WIDTH) - 1u);
}

// The bits `[bit, bit + WIDTH)` (WIDTH < 32, `bit` a multiple of WIDTH) of
// the little-endian byte plane at `plane`, read from memory as the fewest
// bytes a field at such a bit can span: a field starts at a multiple of
// gcd(WIDTH, 8) bits into its byte. A whole aligned field of 8 or 16 bits is
// one load; any other is that many byte loads. A field read alone costs a
// decode lane its address arithmetic and loads on every packet, so it is
// addressed from `bit` directly.
template <uint WIDTH>
inline uint plane_bits(weight_bytes plane, uint bit) {
    constexpr uint lowest = WIDTH & (0u - WIDTH), step = lowest < 8u ? lowest : 8u;
    static_assert(8u - step + WIDTH <= 32u, "a field read alone spans at most four bytes");
    constexpr uint bytes = (8u - step + WIDTH + 7u) / 8u;
    weight_bytes at = plane + (bit >> 3);
    if constexpr (WIDTH == 8u)
        return at[0];
    else if constexpr (WIDTH == 16u)
        return *slab::as<ushort>(at);
    else {
        uint word = 0u;
        for (uint i = 0; i < bytes; ++i)
            word |= uint(at[i]) << (8u * i);
        return (word >> (bit & 7u)) & ((1u << WIDTH) - 1u);
    }
}

// The reader of one entry of every packet (`Entries::chunk_of`): its byte
// `offset` in a packet's unit of the grids plane, and the bit `shift` of its
// bytes there and `high_shift` of its bits in the unit of the grid high
// plane (`Entries::chunk_entry`, from the units at `offset`).
struct EntryChunk {
    uint offset;
    uint shift;
    uint high_shift;
};

// Grid entries: a packet's four entries, the code plane of a grid code
// (`Grid`), entry `step` the step's: their BYTES whole bytes in the grids
// plane and, HIGH, their HIGH further bits in the grid high plane, entry e's
// at bit HIGH e of the packet's (`registry::RowPlaneContent::GridHigh`).
template <uint BYTES, uint HIGH = 0u>
struct Entries {
    static_assert(HIGH == 2u || HIGH == 4u, "a packet's high entry bits are one or two bytes");
    typedef Entries<BYTES> low_plane;
    typedef metal::conditional_t<HIGH == 2u, uchar, ushort> high_bits;
    struct words {
        typename low_plane::words low;
        uint high;
    };
    static words load(weight_bytes row, Rows16 layout, uint p) {
        words w;
        w.low = low_plane::load(row, layout, p);
        w.high = *slab::as<high_bits>(layout.at(row, layout.high, HIGH / 2u, p));
        return w;
    }
    static uint entry(thread const words &w, uint step) {
        return low_plane::entry(w.low, step) | (((w.high >> (HIGH * step)) & ((1u << HIGH) - 1u)) << (8u * BYTES));
    }
    // A reader of one entry of every packet (`EntryChunk`): its units are
    // `unit_bytes` of the grids plane and `high_bytes` of the grid high
    // plane a packet.
    static constexpr uint unit_bytes() { return 4u * BYTES; }
    static constexpr uint high_bytes() { return HIGH / 2u; }
    static EntryChunk chunk_of(uint e) {
        EntryChunk chunk = low_plane::chunk_of(e);
        chunk.high_shift = HIGH * e;
        return chunk;
    }
    static uint chunk_entry(weight_bytes low, weight_bytes high, EntryChunk chunk) {
        return low_plane::chunk_entry(low, high, chunk)
            | ((uint(*slab::as<high_bits>(high)) >> chunk.high_shift) & ((1u << HIGH) - 1u))
            << (8u * BYTES);
    }
};

template <>
struct Entries<1> {
    typedef uint words;
    static words load(weight_bytes row, Rows16 layout, uint p) {
        return *slab::as<uint>(layout.at(row, layout.codes, 4ul, p));
    }
    static uint entry(words e, uint step) { return (e >> (8u * step)) & 0xffu; }
    static constexpr uint unit_bytes() { return 4u; }
    static constexpr uint high_bytes() { return 0u; }
    static EntryChunk chunk_of(uint e) { return EntryChunk{e, 0u, 0u}; }
    static uint chunk_entry(weight_bytes low, weight_bytes,EntryChunk) { return low[0]; }
};

template <>
struct Entries<2> {
    typedef uint2 words;
    static words load(weight_bytes row, Rows16 layout, uint p) {
        return *slab::as<uint2>(layout.at(row, layout.codes, 8ul, p));
    }
    static uint entry(words e, uint step) { return (e[step >> 1] >> (16u * (step & 1u))) & 0xffffu; }
    static constexpr uint unit_bytes() { return 8u; }
    static constexpr uint high_bytes() { return 0u; }
    static EntryChunk chunk_of(uint e) { return EntryChunk{2u * e, 0u, 0u}; }
    static uint chunk_entry(weight_bytes low, weight_bytes,EntryChunk) {
        return *slab::as<ushort>(low);
    }
};

template <>
struct Entries<3> {
    typedef uint3 words;
    static words load(weight_bytes row, Rows16 layout, uint p) {
        return uint3(*slab::as<packed_uint3>(layout.at(row, layout.codes, 12ul, p)));
    }
    // Entry `step` at bit 24 * step of the packet's three words.
    static uint entry(words e, uint step) { return word_bits<24>(e, 24u * step); }
    static constexpr uint unit_bytes() { return 12u; }
    static constexpr uint high_bytes() { return 0u; }
    // Entry e's bytes 3 e .. 3 e + 2 of the unit's three words, by one load
    // of the two words it spans (or ends in).
    static EntryChunk chunk_of(uint e) {
        const uint word = min(3u * e / 4u, 1u);
        return EntryChunk{4u * word, 24u * e - 32u * word, 0u};
    }
    static uint chunk_entry(weight_bytes low, weight_bytes,EntryChunk chunk) {
        const ulong words = as_type<ulong>(uint2(*slab::as<packed_uint2>(low)));
        return uint(words >> chunk.shift) & 0xffffffu;
    }
};

// Interpretations of codes: a step's codes (`codes`) and the exact values of
// two (`pair`), with the code of value zero (`zero`, 0 where codes are their
// values). A linear code is its bytes with MASK flipped (two's complement
// bytes become offset binary), whose code of value zero is ZERO. `codes` are the flipped bytes themselves: 1024 + byte
// less 1024, a half subtraction the compiler issues as a conversion, where
// any other constant costs a subtraction more a step (M4 Pro, prefill
// `attention_project` staging q6k: 9.74 against 9.54 ms); a product
// operand takes the zero off as the exact bias -(zero scale)
// (`Packet::values`). `pair` takes it off in its half subtraction, so a value
// of zero keeps its sign.
template <uint MASK, int ZERO>
struct Linear {
    static_assert(ZERO >= 0 && (ZERO & (ZERO - 1)) == 0, "a zero code is a power of two: -(zero scale) is exact");
    static constant constexpr int zero = ZERO;
    template <typename P>
    static void codes(thread const typename P::words &w, uint step, thread float4 &even, thread float4 &odd) {
        P::template floats<MASK>(w, step, 1024.0h, even, odd);
    }
    template <typename P>
    static float2 pair(thread const typename P::words &w, uint step, uint j) {
        return code_pair(P::template pair<MASK>(w, step, j), half(1024 + ZERO));
    }
};

// The linear interpretations (`registry::CodeInterpretation`): codes whose
// value zero is code Z (`Offset<Z>`, of which `Unsigned` is Z = 0), and
// two's complement bytes (`Twos`), whose flipped sign bits map -128..127
// onto 0..255.
template <int ZERO>
using Offset = Linear<0u, ZERO>;
typedef Offset<0> Unsigned;
typedef Linear<0x80808080u, 128> Twos;

// Odd codes `2 (c - K) + 1` (`registry::CodeInterpretation::Odd`): the codes'
// bytes as exact floats, doubled and moved by 1 - 2K by one exact fma.
template <int K>
struct OddCodes {
    static constant constexpr int zero = 0;
    template <typename P>
    static void codes(thread const typename P::words &w, uint step, thread float4 &even, thread float4 &odd) {
        P::template floats<0u>(w, step, 1024.0h, even, odd);
        even = metal::fma(even, float4(2.0f), float4(float(1 - 2 * K)));
        odd = metal::fma(odd, float4(2.0f), float4(float(1 - 2 * K)));
    }
    template <typename P>
    static float2 pair(thread const typename P::words &w, uint step, uint j) {
        return metal::fma(code_pair(P::template pair<0u>(w, step, j), 1024.0h), float2(2.0f), float2(float(1 - 2 * K)));
    }
};

inline void table_codes(constant float *table, uint word, thread float4 &even, thread float4 &odd) {
    even = float4(table[word & 15u], table[(word >> 8) & 15u], table[(word >> 16) & 15u],
        table[(word >> 24) & 15u]);
    odd = float4(table[(word >> 4) & 15u], table[(word >> 12) & 15u], table[(word >> 20) & 15u],
        table[word >> 28]);
}

// Codebook ID's 4-bit codes (`codebooks::table<ID>`, exact integers).
template <uint ID>
struct Table {
    static constant constexpr int zero = 0;
    template <typename P>
    static void codes(thread const typename P::words &w, uint step, thread float4 &even, thread float4 &odd) {
        table_codes(codebooks::table<ID>::values(), P::nibbles(w, step), even, odd);
    }
    template <typename P>
    static float2 pair(thread const typename P::words &w, uint step, uint j) {
        uint b = P::nibble_pair(w, step, j);
        return float2(codebooks::table<ID>::values()[b & 15u], codebooks::table<ID>::values()[b >> 4]);
    }
};

// E4M3 bytes (`registry::ValueCodes::E4m3`) in a plane of bytes
// (`Planes<8, 0>`): each code 2^9 times its value, an exact float built
// from its bits (a normal code's (1 + m/8) 2^(e + 2), a subnormal one's m),
// its sign kept on a zero, a NaN code (S.1111.111) NaN.
struct E4M3Codes {
    static constant constexpr int zero = 0;
    // The two codes of `two` (bits 0.. and 16.., `Planes<8, 0>::pair`).
    static float2 pair_of(uint two) {
        const uint2 raw = uint2(two & 0xffu, (two >> 16) & 0xffu);
        const uint2 e = (raw >> 3) & 15u, m = raw & 7u;
        float2 magnitude = metal::select(as_type<float2>(((e + 129u) << 23) | (m << 20)), float2(m), e == 0u);
        magnitude = metal::select(magnitude, float2(as_type<float>(0x7fc00000u)), (raw & 0x7fu) == 0x7fu);
        return as_type<float2>(as_type<uint2>(magnitude) | ((raw & 0x80u) << 24));
    }
    template <typename P>
    static void codes(thread const typename P::words &w, uint step, thread float4 &even, thread float4 &odd) {
        static_assert(metal::is_same_v<P, Planes<8, 0>>, "E4M3 codes are a plane of bytes");
        for (uint j = 0; j < 4u; ++j) {
            const float2 two = pair_of(P::template pair<0u>(w, step, j));
            even[j] = two.x;
            odd[j] = two.y;
        }
    }
    template <typename P>
    static float2 pair(thread const typename P::words &w, uint step, uint j) {
        return pair_of(P::template pair<0u>(w, step, j));
    }
};

// Grid codes (`registry::GridCodes`): an entry's LANES-value grid points
// (`grids::table<GRID>`, its first INDEX_BITS-bit index lowest) and, past its
// indices, its signs (SIGNS, `registry::GridSigns`, from entry bit
// SIGN_BIT, SIGN_BITS of them): seven sign bits completed by an eighth of
// even parity, eight sign bits, or none, the points' bytes then two's
// complement values; then, DELTA, a delta bit (entry bit DELTA_BIT): the
// code is 2^DELTA_EXPONENT v + 1, or - 1 where the bit is set. The bits are
// the registry's (`_CODES_GRID_SIGN_BIT`, `_CODES_GRID_SIGN_BITS`,
// `_CODES_GRID_DELTA_BIT`).
template <uint GRID, uint INDEX_BITS, uint LANES>
struct Points;

template <uint GRID, uint INDEX_BITS>
struct Points<GRID, INDEX_BITS, 8> {
    static uint2 point(uint entry) { return grids::table<GRID>::points()[entry & ((1u << INDEX_BITS) - 1u)]; }
};

template <uint GRID, uint INDEX_BITS>
struct Points<GRID, INDEX_BITS, 4> {
    static uint2 point(uint entry) {
        constant uint *points = grids::table<GRID>::points();
        return uint2(points[entry & ((1u << INDEX_BITS) - 1u)],
            points[(entry >> INDEX_BITS) & ((1u << INDEX_BITS) - 1u)]);
    }
};

// Four sign bits as byte masks (bit i: byte i all ones), which sign four
// magnitudes as bytes of their values + 128: 128 + m - (2 m & mask), with
// no byte carry or borrow for magnitudes below 128 (every grid's). A table:
// spreading the bits by integer multiplies costs a decode lane more than
// the lookup.
constant uint sign_bytes[16] = {0x00000000u, 0x000000ffu, 0x0000ff00u, 0x0000ffffu, 0x00ff0000u, 0x00ff00ffu, 0x00ffff00u, 0x00ffffffu, 0xff000000u, 0xff0000ffu, 0xff00ff00u, 0xff00ffffu, 0xffff0000u, 0xffff00ffu, 0xffffff00u, 0xffffffffu};

// Four magnitudes (bytes of `word`) under sign bits `signs` (bit i: byte i),
// as bytes of their values + 128.
inline uint signed_bytes(uint word, uint signs) {
    return 0x80808080u + word - ((word << 1) & sign_bytes[signs]);
}

// The eight lane sign bits of an entry's seven: the eighth makes the count of
// set bits even.
inline uint grid_signs(uint signs) { return signs | ((popcount(signs) & 1u) << 7); }

template <uint GRID, uint INDEX_BITS, uint LANES, uint SIGNS, bool DELTA, uint DELTA_EXPONENT, uint SIGN_BIT,
    uint SIGN_BITS, uint DELTA_BIT>
struct Grid {
    static_assert(SIGNS == SEISMIC_GRID_SIGNS_PARITY_PLUS7 || SIGNS == SEISMIC_GRID_SIGNS_EXPLICIT8 ||
        SIGNS == SEISMIC_GRID_SIGNS_NONE, "a grid entry's signs are parity, explicit or none");
    static_assert(DELTA || DELTA_EXPONENT == 0u, "a delta exponent is a delta's");
    static constant constexpr int zero = 0;
    typedef Points<GRID, INDEX_BITS, LANES> points;
    static constexpr bool signed_points() { return SIGNS == SEISMIC_GRID_SIGNS_NONE; }
    static uint signs(uint entry) {
        const uint bits = (entry >> SIGN_BIT) & ((1u << SIGN_BITS) - 1u);
        if constexpr (SIGNS == SEISMIC_GRID_SIGNS_PARITY_PLUS7)
            return grid_signs(bits);
        else
            return bits;
    }
    // The delta of `entry`'s codes: -1 where its delta bit is set, else 1.
    static float delta(uint entry) { return ((entry >> DELTA_BIT) & 1u) != 0u ? -1.0f : 1.0f; }
    // The eight signed codes of `entry` (lanes 0..3 in x, 4..7 in y).
    template <typename P>
    static void codes(thread const typename P::words &w, uint step, thread float4 &even, thread float4 &odd) {
        uint entry = P::entry(w, step);
        if constexpr (signed_points()) {
            // Two's complement bytes, their sign bits flipped to offset binary.
            uint2 point = points::point(entry);
            float4 low = byte_floats(point.x ^ 0x80808080u, 1152.0h), high = byte_floats(point.y ^ 0x80808080u, 1152.0h);
            even = float4(low.x, low.z, high.x, high.z);
            odd = float4(low.y, low.w, high.y, high.w);
        } else {
            // The signed values + 128 as bytes, four at a time.
            uint s = signs(entry);
            uint2 point = points::point(entry);
            float4 low = byte_floats(signed_bytes(point.x, s & 15u), 1152.0h);
            float4 high = byte_floats(signed_bytes(point.y, s >> 4), 1152.0h);
            even = float4(low.x, low.z, high.x, high.z);
            odd = float4(low.y, low.w, high.y, high.w);
        }
        if constexpr (DELTA) {
            const float d = delta(entry);
            even = metal::fma(even, float4(power2<int(DELTA_EXPONENT)>()), float4(d));
            odd = metal::fma(odd, float4(power2<int(DELTA_EXPONENT)>()), float4(d));
        }
    }
    // Lanes 2j and 2j + 1 of the same.
    template <typename P>
    static float2 pair(thread const typename P::words &w, uint step, uint j) {
        uint entry = P::entry(w, step);
        uint2 point = points::point(entry);
        uint word = (j < 2u ? point.x : point.y) >> (16u * (j & 1u));
        float2 values;
        if constexpr (signed_points()) {
            values = code_pair(((word & 0xffu) | ((word & 0xff00u) << 8)) ^ 0x00800080u, 1152.0h);
        } else {
            uint s = signs(entry) >> (2u * j);
            float2 magnitudes = code_pair((word & 0xffu) | ((word & 0xff00u) << 8), 1024.0h);
            values = as_type<float2>(as_type<uint2>(magnitudes) ^ ((uint2(s, s >> 1) & 1u) << 31));
        }
        if constexpr (DELTA)
            values = metal::fma(values, float2(power2<int(DELTA_EXPONENT)>()), float2(delta(entry)));
        return values;
    }
};

// Stored scales: `bytes` a scale; `read` one scale's value, `read2` two
// consecutive ones.
struct F16Scale {
    static constant constexpr uint bytes = 2;
    static float read(weight_bytes at) { return float(*slab::as<half>(at)); }
    static float2 read2(weight_bytes at) { return float2(*slab::as<half2>(at)); }
};
struct BF16Scale {
    static constant constexpr uint bytes = 2;
    static float read(weight_bytes at) {
        return as_type<float>(uint(*slab::as<ushort>(at)) << 16);
    }
    static float2 read2(weight_bytes at) {
        uint two = *slab::as<uint>(at);
        return float2(as_type<float>(two << 16), as_type<float>(two & 0xffff0000u));
    }
};
struct F32Scale {
    static constant constexpr uint bytes = 4;
    static float read(weight_bytes at) { return *slab::as<float>(at); }
    static float2 read2(weight_bytes at) { return *slab::as<float2>(at); }
};
// E8M0: 2^(e - 127) (e = 0 is the F32 subnormal 2^-127); 0xff is NaN.
struct E8M0Scale {
    static constant constexpr uint bytes = 1;
    static float value(uint e) { return as_type<float>(e == 0u ? 0x00400000u : e == 255u ? 0x7fc00000u : e << 23); }
    static float read(weight_bytes at) { return value(at[0]); }
    static float2 read2(weight_bytes at) {
        uint two = *slab::as<ushort>(at);
        return float2(value(two & 0xffu), value(two >> 8));
    }
};
// UE4M3 (sign bit ignored; 0x7f is NaN): (1 + m/8) 2^(e - 7), or m 2^-9 when
// e = 0.
struct UE4M3Scale {
    static constant constexpr uint bytes = 1;
    // Times 2^X, exactly: X in the exponent the value's bits are built
    // with, and in the subnormal's power of two.
    template <int X = 0>
    static float value(uint raw) {
        raw &= 0x7fu;
        uint e = raw >> 3, m = raw & 7u;
        float normal = as_type<float>((uint(int(e) + 120 + X) << 23) | (m << 20));
        return raw == 0x7fu ? as_type<float>(0x7fc00000u) : e == 0u ? float(m) * power2<X - 9>() : normal;
    }
    template <int X = 0>
    static float read(weight_bytes at) { return value<X>(at[0]); }
    template <int X = 0>
    static float2 read2(weight_bytes at) {
        uint two = *slab::as<ushort>(at);
        return float2(value<X>(two & 0xffu), value<X>(two >> 8));
    }
};

// Scale encoding S's stored scales times 2^X (an exact power of two): one
// multiply, or none where S builds a value's bits (`UE4M3Scale`).
template <typename S, int X>
struct ScalesTimes {
    static float read(weight_bytes at) { return X == 0 ? S::read(at) : S::read(at) * power2<X>(); }
    static float2 read2(weight_bytes at) { return X == 0 ? S::read2(at) : S::read2(at) * power2<X>(); }
};
template <int X>
struct ScalesTimes<UE4M3Scale, X> {
    static float read(weight_bytes at) { return UE4M3Scale::read<X>(at); }
    static float2 read2(weight_bytes at) { return UE4M3Scale::read2<X>(at); }
};

// The BITS-bit codes at bits 0 and BITS of `field` as a code pair: the
// second in bits 16.. (`code_pair`).
template <uint BITS>
inline uint adjacent_pair(uint field) {
    const uint mask = (1u << BITS) - 1u;
    return (field & mask) | (((field >> BITS) & mask) << 16);
}

// Interpretations of a BITS-bit local coefficient code (`value`, an exact
// float; `adjacent`, those of the two codes at bits 0 and BITS of a field:
// byte codes by one vector conversion, narrower ones as a code pair with one
// half subtraction; `largest`, the largest magnitude of one).
template <uint BITS>
struct LocalUnsigned {
    static_assert(BITS <= 10u, "a code pair's codes are below 1024");
    static constant constexpr uint largest = (1u << BITS) - 1u;
    static float value(uint code) { return float(code); }
    static float2 adjacent(uint field) {
        if constexpr (BITS == 8u)
            return float2(as_type<uchar2>(ushort(field)));
        else
            return code_pair(adjacent_pair<BITS>(field), 1024.0h);
    }
};
template <uint BITS>
struct LocalTwos {
    static_assert(BITS <= 10u, "a code pair's codes are below 1024");
    static constant constexpr uint largest = 1u << (BITS - 1u);
    static float value(uint code) { return float(int(code << (32u - BITS)) >> (32u - BITS)); }
    // Flipped sign bits make narrower codes offset binary, less
    // 2^(BITS - 1).
    static float2 adjacent(uint field) {
        if constexpr (BITS == 8u)
            return float2(as_type<char2>(ushort(field)));
        else
            return code_pair(adjacent_pair<BITS>(field) ^ (largest * 0x00010001u), half(1024u + largest));
    }
};
template <uint BITS, int OFFSET>
struct LocalOffset {
    static_assert(BITS <= 10u, "a code pair's codes are below 1024");
    static constant constexpr uint largest = uint(OFFSET) > (1u << BITS) - 1u - uint(OFFSET)
        ? uint(OFFSET) : (1u << BITS) - 1u - uint(OFFSET);
    static float value(uint code) { return float(int(code) - OFFSET); }
    static float2 adjacent(uint field) { return code_pair(adjacent_pair<BITS>(field), half(1024 + OFFSET)); }
};
// Odd codes 2 (c - K) + 1: 2c + 1 less 2K.
template <uint BITS, int K>
struct LocalOdd {
    static_assert(BITS <= 9u, "a code pair's values are below 1024");
    static_assert(-512 < K && K < 512, "the code pair's offset, 1024 + 2K, is an exact half");
    static constant constexpr int least = 1 - 2 * K, greatest = int(2u << BITS) - 1 - 2 * K;
    static constant constexpr uint largest = uint(-least > greatest ? -least : greatest);
    static float value(uint code) { return float(2u * code + 1u) - float(2 * K); }
    static float2 adjacent(uint field) {
        return code_pair((adjacent_pair<BITS>(field) << 1) | 0x00010001u, half(1024 + 2 * K));
    }
};
// A group's signed bias (`registry::LocalBias::Signed`): ±2^EXPONENT times
// its scale, signed by the top bit of its carried local code past its
// SCALE_BITS-bit scale code; or none.
struct NoSignedBias {
    static constexpr bool any() { return false; }
    static constexpr int exponent() { return 0; }
};
template <int EXPONENT, uint SCALE_BITS>
struct SignedBias {
    static constexpr bool any() { return true; }
    static constexpr int exponent() { return EXPONENT; }
    static constexpr uint scale_bits() { return SCALE_BITS; }
};

// One float per coefficient group of a packet (one or two): `of` takes the
// groups' values from a float2, `at` reads group g's.
template <uint GROUPS>
struct GroupFloats;
template <>
struct GroupFloats<1> {
    typedef float type;
    static float of(float2 v) { return v.x; }
    static float at(float v, uint) { return v; }
};
template <>
struct GroupFloats<2> {
    typedef float2 type;
    static float2 of(float2 v) { return v; }
    static float at(float2 v, uint g) { return g == 0 ? v.x : v.y; }
};

// The coefficients of a packet's GROUPS coefficient groups: the scale of
// each and, BIASED, the bias of each. A walk holds packets in registers (two
// a range), so they hold nothing a component lacks. `make` takes the groups'
// scales and biases from float2s; `scale(g)` and `bias(g)` are group g's.
template <uint GROUPS, bool BIASED>
struct Coefficients {
    typedef GroupFloats<GROUPS> F;
    typename F::type scales;
    typename F::type biases;
    static Coefficients make(float2 scale, float2 bias) { return Coefficients{F::of(scale), F::of(bias)}; }
    float scale(uint g) const { return F::at(scales, g); }
    float bias(uint g) const { return F::at(biases, g); }
};
template <uint GROUPS>
struct Coefficients<GROUPS, false> {
    typedef GroupFloats<GROUPS> F;
    typename F::type scales;
    static Coefficients make(float2 scale, float2) { return Coefficients{F::of(scale)}; }
    float scale(uint g) const { return F::at(scales, g); }
    float bias(uint) const { return 0.0f; }
};

// Direct coefficients: a scale `S` (with BIAS a bias after it) per GROUP
// values; the scale times 2^EXPONENT. A biased storage group is one
// coefficient group. A packet has `groups` of them (one per 32
// values or more, two per 16).
template <typename S, uint GROUP, bool BIAS, uint STORAGE, int EXPONENT>
struct Direct {
    static_assert(GROUP == 16u || GROUP % 32u == 0u, "a packet's direct coefficients are two or one");
    static_assert(!BIAS || STORAGE == GROUP, "a group's bias follows its scale");
    static constant constexpr uint groups = GROUP < 32u ? 32u / GROUP : 1u;
    static constant constexpr bool biased = BIAS;
    typedef Coefficients<groups, biased> coefficients;
    // Byte offset of group G's scale in the row's share of the supers
    // plane; its bias follows it.
    static uint scale_offset(uint g) { return g * ((BIAS ? 2u : 1u) * S::bytes); }
    template <typename P>
    static coefficients load(weight_bytes row, Rows16 layout, uint p, thread const typename P::words &) {
        return at(layout.super_at(row, 0ul), p);
    }
    // Packet p's coefficients, of the row whose share of the supers plane
    // is at `supers` (`Rows16::super_at(row, 0)`).
    static coefficients at(weight_bytes supers, uint p) {
        // Packet p's first group, without a product that could wrap (which
        // keeps a compiler from folding it).
        const uint first = GROUP < 32u ? p * (32u / GROUP) : p / (GROUP / 32u);
        float2 scale = float2(0.0f), bias = float2(0.0f);
        if (groups == 2 && !BIAS) {
            scale = ScalesTimes<S, EXPONENT>::read2(supers + scale_offset(first));
        } else {
            for (uint g = 0; g < groups; ++g)
                scale[g] = ScalesTimes<S, EXPONENT>::read(supers + scale_offset(first + g));
        }
        if (BIAS) {
            for (uint g = 0; g < groups; ++g)
                bias[g] = S::read(supers + (scale_offset(first + g) + S::bytes));
        }
        return coefficients::make(scale, bias);
    }
};

// WORDS consecutive little-endian words, shifted right as one value
// (`shift`), with their lowest word (`low`) and their bits
// [BITS index, BITS (index + 1)) (`field`, BITS < 32). No words (`Run<0>`)
// are a run of nothing.
template <uint WORDS>
struct Run;
template <>
struct Run<0> {
    struct words {};
};
template <>
struct Run<1> {
    typedef uint words;
    static words load(weight_bytes at) { return *slab::as<uint>(at); }
    template <uint BITS>
    static words shift(words w) { return w >> BITS; }
    static uint low(words w) { return w; }
    template <uint BITS>
    static uint field(words w, uint index) { return (w >> (BITS * index)) & ((1u << BITS) - 1u); }
};
template <>
struct Run<2> {
    typedef uint2 words;
    static words load(weight_bytes at) { return uint2(*slab::as<packed_uint2>(at)); }
    template <uint BITS>
    static words shift(words w) { return uint2((w.x >> BITS) | (w.y << (32u - BITS)), w.y >> BITS); }
    static uint low(words w) { return w.x; }
    template <uint BITS>
    static uint field(words w, uint index) {
        return word_bits<BITS>(w, BITS * index);
    }
};
template <>
struct Run<3> {
    typedef uint3 words;
    static words load(weight_bytes at) { return uint3(*slab::as<packed_uint3>(at)); }
    template <uint BITS>
    static words shift(words w) {
        return uint3((w.x >> BITS) | (w.y << (32u - BITS)), (w.y >> BITS) | (w.z << (32u - BITS)), w.z >> BITS);
    }
    static uint low(words w) { return w.x; }
    template <uint BITS>
    static uint field(words w, uint index) {
        return word_bits<BITS>(w, BITS * index);
    }
};
template <>
struct Run<4> {
    typedef uint4 words;
    static words load(weight_bytes at) { return uint4(*slab::as<packed_uint4>(at)); }
    template <uint BITS>
    static words shift(words w) {
        return uint4((w.x >> BITS) | (w.y << (32u - BITS)), (w.y >> BITS) | (w.z << (32u - BITS)),
            (w.z >> BITS) | (w.w << (32u - BITS)), w.w >> BITS);
    }
    static uint low(words w) { return w.x; }
    template <uint BITS>
    static uint field(words w, uint index) {
        return word_bits<BITS>(w, BITS * index);
    }
};

// WORDS words (`Run<WORDS>`) of the BYTES little-endian bytes at `at`, read
// in the widest loads their alignment keeps (a factor group's fields of
// whole bytes start at a multiple of BYTES), the bytes past them zero.
template <uint WORDS, uint BYTES>
inline typename Run<WORDS>::words byte_run(weight_bytes at) {
    static_assert(WORDS >= 1u && WORDS <= 4u && (BYTES + 3u) / 4u == WORDS, "a run of bytes is its words");
    constexpr uint unit = BYTES % 4u == 0u ? 4u : BYTES % 2u == 0u ? 2u : 1u;
    uint w[4] = {0u, 0u, 0u, 0u};
    for (uint i = 0; i < BYTES; i += unit) {
        const uint value = unit == 4u   ? *slab::as<uint>(at + i)
                           : unit == 2u ? uint(*slab::as<ushort>(at + i))
                                        : uint(at[i]);
        w[i / 4u] |= value << (8u * (i % 4u));
    }
    if constexpr (WORDS == 1u)
        return w[0];
    else if constexpr (WORDS == 2u)
        return uint2(w[0], w[1]);
    else if constexpr (WORDS == 3u)
        return uint3(w[0], w[1], w[2]);
    else
        return uint4(w[0], w[1], w[2], w[3]);
}

// Hierarchical coefficients: per GROUP values a LOCAL_BITS-bit scale code
// (and with MIN a minimum code after it), interpreted by `L`, under a factor
// `FS` (and a minimum factor) per FACTOR_GROUP values: scale = factor
// 2^(EXPONENT + CODE_EXPONENT) L(scale code) and bias = BIAS_SIGN minimum
// factor 2^EXPONENT L(minimum code); or, with a signed bias SB (carried
// codes only), scale = factor 2^(EXPONENT + CODE_EXPONENT) L(the code's low
// SB::scale_bits() bits) and bias = ±factor 2^(EXPONENT + SB::exponent()) L(the
// same bits), - where the code's top bit is set. A packet's codes are one `field` of `field_bits`,
// in the scales plane, where each factor group's fields fill whole bytes,
// or, where CARRIED_BIT is not 0, bit j of a group's code at bit
// CARRIED_BIT of its entry j.
template <typename FS, uint FACTOR_GROUP, int EXPONENT, int CODE_EXPONENT, uint LOCAL_BITS, typename L,
    uint GROUP, bool MIN, int BIAS_SIGN, uint CARRIED_BIT, typename SB = NoSignedBias>
struct Hier {
    static_assert(GROUP == 16u || GROUP == 32u, "a packet's hierarchical coefficients are two or one");
    // Whether its codes sign a bias of their scale.
    static constexpr bool signed_bias() { return SB::any(); }
    // Whether its entries carry its local codes (at bit CARRIED_BIT).
    static constant constexpr bool carries = CARRIED_BIT != 0u;
    static_assert(!carries || (!MIN && LOCAL_BITS <= GROUP / 8u), "a group's entries carry its scale code");
    static constant constexpr uint groups = 32u / GROUP;
    static constant constexpr bool biased = MIN || signed_bias();
    typedef Coefficients<groups, biased> coefficients;
    static constant constexpr uint width = LOCAL_BITS * (MIN ? 2u : 1u);
    static constant constexpr uint field_bits = width * groups;
    static_assert(field_bits < 32u, "a packet's field is below a word: a run shifts by it");
    // Packets a factor group, and the words of its fields in the scales
    // plane (none when its entries carry them).
    static constant constexpr uint packets = FACTOR_GROUP / 32u;
    static constant constexpr uint words = carries ? 0u : (packets * field_bits + 31u) / 32u;
    static_assert(carries || packets * field_bits % 8u == 0, "a factor group's fields fill whole bytes");
    // A factor group's factors, as the coefficients of one group: the scale
    // factor 2^(EXPONENT + CODE_EXPONENT) (`scale(0)`) and, MIN, the BIAS_SIGN
    // minimum factor 2^EXPONENT (`bias(0)`).
    typedef Coefficients<1, MIN> group_factors;
    // Factor group f's.
    static group_factors factors(weight_bytes row, Rows16 layout, uint f) {
        weight_bytes at =layout.super_at(row, ulong(f) * ((MIN ? 2u : 1u) * FS::bytes));
        float2 factor = MIN ? FS::read2(at) : float2(FS::read(at), 0.0f);
        if (EXPONENT + CODE_EXPONENT != 0)
            factor.x *= power2<EXPONENT + CODE_EXPONENT>();
        if (EXPONENT != 0)
            factor.y *= power2<EXPONENT>();
        if (BIAS_SIGN < 0)
            factor.y = -factor.y;
        return group_factors::make(float2(factor.x, 0.0f), float2(factor.y, 0.0f));
    }
    // Group g's scale code and minimum code in a packet's field.
    static uint scale_code(uint field, uint g) { return (field >> (g * width)) & ((1u << LOCAL_BITS) - 1u); }
    static uint min_code(uint field, uint g) {
        return (field >> (g * width + LOCAL_BITS)) & ((1u << LOCAL_BITS) - 1u);
    }
    // A packet's coefficients from its field: a group's scale code with its
    // minimum code, or the two groups' scale codes, as one adjacent pair
    // (`L::adjacent`).
    static coefficients form(uint field, thread const group_factors &factor) {
        float2 scale = float2(0.0f), bias = float2(0.0f);
        if constexpr (signed_bias()) {
            for (uint g = 0; g < groups; ++g) {
                const uint code = scale_code(field, g);
                scale[g] = factor.scale(0) * L::value(code & ((1u << SB::scale_bits()) - 1u));
                bias[g] = (((code >> SB::scale_bits()) & 1u) != 0u ? -1.0f : 1.0f)
                    * (scale[g] * power2<SB::exponent() - CODE_EXPONENT>());
            }
        } else if (MIN) {
            for (uint g = 0; g < groups; ++g) {
                const float2 codes = L::adjacent(field >> (g * width));
                scale[g] = factor.scale(0) * codes.x;
                bias[g] = factor.bias(0) * codes.y;
            }
        } else if (groups == 2) {
            scale = factor.scale(0) * L::adjacent(field);
        } else {
            scale.x = factor.scale(0) * L::value(scale_code(field, 0));
        }
        return coefficients::make(scale, bias);
    }
    // A packet's field carried by its entries `w` (`P`).
    template <typename P>
    static uint carried(thread const typename P::words &w) {
        uint field = 0u;
        for (uint g = 0; g < groups; ++g)
            for (uint j = 0; j < LOCAL_BITS; ++j)
                field |= ((P::entry(w, g * (GROUP / 8u) + j) >> CARRIED_BIT) & 1u) << (g * width + j);
        return field;
    }
    // A packet's field: carried by its entries `w` (`P`) or in the scales
    // plane, where factor groups' fields fill whole words one after
    // another, so packet p's is at bit `field_bits` p of the row's share.
    template <typename P>
    static uint field(weight_bytes row, Rows16 layout, uint p, thread const typename P::words &w) {
        if constexpr (carries)
            return carried<P>(w);
        else
            return plane_bits<field_bits>(layout.scale_at(row, 0ul), p * field_bits);
    }
    template <typename P>
    static coefficients load(weight_bytes row, Rows16 layout, uint p, thread const typename P::words &w) {
        return form(field<P>(row, layout, p, w), factors(row, layout, p / packets));
    }

    // A run over a factor group's packets from packet p on: its fields from
    // packet p's lowest (no words when its entries carry them), and its factors. `next` takes
    // the next packet's field. `group_fields` are factor group f's fields
    // whole, packet b's `field_of` them.
    typedef Run<words> fields;
    static typename fields::words group_fields(weight_bytes row, Rows16 layout, uint f) {
        if constexpr (packets * field_bits % 32u == 0u)
            return fields::load(layout.scale_at(row, ulong(f) * (4u * words)));
        else
            return byte_run<words, packets * field_bits / 8u>(layout.scale_at(row, ulong(f) * (packets * field_bits / 8u)));
    }
    static uint field_of(typename fields::words w, uint b) { return fields::template field<field_bits>(w, b); }
    struct run {
        typename fields::words remaining;
        group_factors factor;
    };
    static run start(weight_bytes row, Rows16 layout, uint p) {
        run state;
        if constexpr (!carries) {
            state.remaining = group_fields(row, layout, p / packets);
            for (uint skipped = 0; skipped < p % packets; ++skipped)
                state.remaining = fields::template shift<field_bits>(state.remaining);
        }
        state.factor = factors(row, layout, p / packets);
        return state;
    }
    static uint next(thread run &state) {
        uint field = fields::low(state.remaining) & ((1u << field_bits) - 1u);
        state.remaining = fields::template shift<field_bits>(state.remaining);
        return field;
    }
    template <typename P>
    static coefficients take(thread run &state, thread const typename P::words &w) {
        if constexpr (carries)
            return form(carried<P>(w), state.factor);
        else
            return form(next(state), state.factor);
    }
};

// The consumer forms the runtime admits a weight to (`<prefix>_FORM_<NAME>`,
// its one form admission, `abi/forms.rs`), which projection libraries select
// on: the token-packing GEMM (`PACK`, with its codes' entry: a code enters as
// its value + PACK_OFFSET, centred at PACK_CENTRE, and its sums take
// PACK_RESIDUE residue bits), the int8 GEMM (`INT8`) and the matrix GEMV
// folding a block's scale codes, over a block's runs or of symmetric 32
// columns (`MATRIX_CODES`, `MATRIX_RUNS`, `MATRIX_SYMMETRIC`, at most one;
// `MATRIX_RUNS` 2 where a run's products are exact F16 operands,
// `matrix_runs_halves`).
template <bool PACK, uint PACK_CENTRE, uint PACK_OFFSET, uint PACK_RESIDUE, bool INT8, bool MATRIX_CODES,
    uint MATRIX_RUNS, bool MATRIX_SYMMETRIC>
struct Forms {
    static constant constexpr bool pack = PACK, int8 = INT8;
    static constant constexpr uint pack_centre = PACK_CENTRE, pack_offset = PACK_OFFSET, pack_residue = PACK_RESIDUE;
    static constant constexpr bool matrix_codes = MATRIX_CODES, matrix_runs = MATRIX_RUNS != 0u,
        matrix_symmetric = MATRIX_SYMMETRIC;
    static constexpr bool matrix_runs_halves() { return MATRIX_RUNS == 2u; }
};

// A decoder of codes in planes `P` interpreted by `C` under coefficients `K`,
// admitted to the consumer forms `F` (`Forms`).
template <typename P, typename C, typename K, typename F>
struct Packet {
    static constant constexpr uint groups = K::groups;
    struct packet {
        typename P::words codes;
        typename K::coefficients coefficients;
    };
    static packet load(weight_bytes row, Rows16 layout, uint p) {
        packet k;
        k.codes = P::load(row, layout, p);
        k.coefficients = K::template load<P>(row, layout, p, k.codes);
        return k;
    }
    // The coefficient group of step `step` (two steps a group of 16).
    static uint group(uint step) { return groups == 1 ? 0u : step / 2u; }
    static float scale(thread const packet &k, uint step) { return k.coefficients.scale(group(step)); }
    // Step `step`'s eight values (even, odd) as product operands: scale *
    // code with one rounding, plus by the same fma a stored bias, the exact
    // -(zero code scale) (`Linear`) or +0, so a value of zero is +0. Where
    // nothing is added the fma is still the form: a multiply rounded to a
    // staged half issues as `fmul` where the fma is `ffma` with 0.0, and the
    // staged GEMM is slower for it (M4 Pro, prefill NVFP4
    // `dense_output_gemm`, 17.6 µs a launch: +0.9-1.9% a round).
    static void values(thread const packet &k, uint step, thread float4 &even, thread float4 &odd) {
        static_assert(C::zero == 0 || !K::biased, "a stored bias and a zero code are not one exact bias");
        C::template codes<P>(k.codes, step, even, odd);
        const float s = scale(k, step);
        const float b = C::zero != 0 ? -(float(C::zero) * s) : K::biased ? k.coefficients.bias(group(step)) : 0.0f;
        even = metal::fma(float4(s), even, float4(b));
        odd = metal::fma(float4(s), odd, float4(b));
    }
    // Step `step`'s eight logical values exactly, as a gather publishes
    // them: scale * (code - zero code), the difference exact, plus a stored
    // bias by one fma, so a zero keeps the sign of scale * 0.
    static void exact_values(thread const packet &k, uint step, thread float4 &even, thread float4 &odd) {
        C::template codes<P>(k.codes, step, even, odd);
        if (C::zero != 0) {
            even -= float(C::zero);
            odd -= float(C::zero);
        }
        const float s = scale(k, step);
        if (K::biased) {
            const float b = k.coefficients.bias(group(step));
            even = metal::fma(float4(s), even, float4(b));
            odd = metal::fma(float4(s), odd, float4(b));
        } else {
            even *= s;
            odd *= s;
        }
    }
    // The values of step `step`'s elements 2j and 2j + 1, each its logical
    // value exactly (a zero keeps the sign of scale * 0).
    static float2 pair_values(thread const packet &k, uint step, uint j) {
        const float2 codes = C::template pair<P>(k.codes, step, j);
        return K::biased ? metal::fma(float2(scale(k, step)), codes, float2(k.coefficients.bias(group(step))))
                         : scale(k, step) * codes;
    }
};

// Dense weights of element type E (element::Bf16, F16 or F32). The row's
// final packet may be partial when K is not a multiple of 32; its missing
// elements decode as zero.
template <typename E>
struct Dense {
    static constant constexpr uint groups = 1;
    struct packet {
        weight_bytes values;
        uint valid;   // elements of this packet inside the row
    };
    static packet load(weight_bytes row, Rows16 layout, uint p, uint k) {
        packet out;
        out.values = row + layout.codes + ulong(p) * 32ul * E::bytes;
        out.valid = min(32u, k - 32u * p);
        return out;
    }
    static float element(thread const packet &k, uint i) {
        return i < k.valid
            ? E::load(slab::as<typename E::storage>(k.values)[i])
            : 0.0f;
    }
    static void values(thread const packet &k, uint step, thread float4 &even, thread float4 &odd) {
        uint first = 8u * step;
        if (first + 8u <= k.valid) {
            const auto v = slab::as<typename E::storage>(k.values) + first;
            even = float4(E::load(v[0]), E::load(v[2]), E::load(v[4]), E::load(v[6]));
            odd = float4(E::load(v[1]), E::load(v[3]), E::load(v[5]), E::load(v[7]));
        } else {
            even = float4(element(k, first), element(k, first + 2), element(k, first + 4),
                element(k, first + 6));
            odd = float4(element(k, first + 1), element(k, first + 3), element(k, first + 5),
                element(k, first + 7));
        }
    }
    static float2 pair_values(thread const packet &k, uint step, uint j) {
        return float2(element(k, 8u * step + 2u * j), element(k, 8u * step + 2u * j + 1u));
    }
    static void exact_values(thread const packet &k, uint step, thread float4 &even, thread float4 &odd) {
        values(k, step, even, odd);
    }
};

// Uniform packet loading: quantized packets ignore the row length.
template <typename W>
struct Loader {
    static typename W::packet load(weight_bytes row, Rows16 layout, uint p, uint) {
        return W::load(row, layout, p);
    }
};
template <typename E>
struct Loader<Dense<E>> {
    static typename Dense<E>::packet load(weight_bytes row, Rows16 layout, uint p, uint k) {
        return Dense<E>::load(row, layout, p, k);
    }
};

// A row's coefficients over a run of `packets` consecutive packets, held in
// registers by a kernel that walks one row along K. `load` reads the run's
// coefficient storage once, positioned at packet p; `packet` then forms
// packet p from its codes and that state, with the values `Loader<W>::load`
// yields, and moves the state on to packet p + 1. Hierarchical coefficients
// keep their local codes and factors per factor group in planes apart from
// the codes, so a walk that loads them per packet touches two more places of
// every row on every packet; with a run it touches them once per factor
// group. Other decoders have no run: `packets` is 1, the state is empty and
// `packet` is the plain load.
template <typename W>
struct Block {
    static constant constexpr uint packets = 1;
    struct state {};
    static state load(weight_bytes, Rows16, uint) { return state{}; }
    static typename W::packet packet(weight_bytes row, Rows16 layout, uint p, uint k,
        thread state &) {
        return Loader<W>::load(row, layout, p, k);
    }
};

template <typename P, typename C, typename FS, uint FACTOR_GROUP, int EXPONENT, int CODE_EXPONENT, uint LOCAL_BITS,
    typename L, uint GROUP, bool MIN, int BIAS_SIGN, uint CARRIED_BIT, typename SB, typename F>
struct Block<Packet<P, C,
    Hier<FS, FACTOR_GROUP, EXPONENT, CODE_EXPONENT, LOCAL_BITS, L, GROUP, MIN, BIAS_SIGN, CARRIED_BIT, SB>,
    F>> {
    typedef Hier<FS, FACTOR_GROUP, EXPONENT, CODE_EXPONENT, LOCAL_BITS, L, GROUP, MIN, BIAS_SIGN, CARRIED_BIT, SB> K;
    typedef Packet<P, C, K, F> W;
    static constant constexpr uint packets = K::packets;
    typedef typename K::run state;
    static state load(weight_bytes row, Rows16 layout, uint p) { return K::start(row, layout, p); }
    static typename W::packet packet(weight_bytes row, Rows16 layout, uint p, uint,
        thread state &run) {
        typename W::packet k;
        k.codes = P::load(row, layout, p);
        k.coefficients = K::template take<P>(run, k.codes);
        return k;
    }
};

// One decoded logical weight value (used by gathers such as the embedding).
template <typename W>
inline float value_at(thread const typename W::packet &k, uint i) {
    return W::pair_values(k, i >> 3, (i & 7u) >> 1)[i & 1u];
}

// ---------------------------------------------------------------------------
// The decoder of a packed element's emitted decode axes (`_CODE*`,
// `_COEFF*`): each axis value's component, chosen in one place.

// The producer of codes CODES in planes of LOW and HIGH bits.
template <uint CODES, uint LOW, uint HIGH>
struct PlanesOf {
    typedef Planes<LOW, HIGH> type;
};
template <uint LOW, uint HIGH>
struct PlanesOf<SEISMIC_CODES_GRID, LOW, HIGH> {
    typedef Entries<LOW / 8u, HIGH> type;
};

// The interpretation of BITS-bit codes (a grid's with its signs and delta).
template <uint CODES, uint BITS, int OFFSET, uint TABLE, uint GRID, uint INDEX_BITS, uint LANES, uint SIGNS,
    bool DELTA, uint DELTA_EXPONENT, uint SIGN_BIT, uint SIGN_BITS, uint DELTA_BIT>
struct InterpretationOf;
template <uint BITS, int OFFSET, uint TABLE, uint GRID, uint INDEX_BITS, uint LANES, uint SIGNS, bool DELTA,
    uint DELTA_EXPONENT, uint SIGN_BIT, uint SIGN_BITS, uint DELTA_BIT>
struct InterpretationOf<SEISMIC_CODES_UNSIGNED, BITS, OFFSET, TABLE, GRID, INDEX_BITS, LANES, SIGNS, DELTA,
    DELTA_EXPONENT, SIGN_BIT, SIGN_BITS, DELTA_BIT> {
    typedef Unsigned type;
};
template <uint BITS, int OFFSET, uint TABLE, uint GRID, uint INDEX_BITS, uint LANES, uint SIGNS, bool DELTA,
    uint DELTA_EXPONENT, uint SIGN_BIT, uint SIGN_BITS, uint DELTA_BIT>
struct InterpretationOf<SEISMIC_CODES_OFFSET, BITS, OFFSET, TABLE, GRID, INDEX_BITS, LANES, SIGNS, DELTA,
    DELTA_EXPONENT, SIGN_BIT, SIGN_BITS, DELTA_BIT> {
    typedef Offset<OFFSET> type;
};
template <uint BITS, int OFFSET, uint TABLE, uint GRID, uint INDEX_BITS, uint LANES, uint SIGNS, bool DELTA,
    uint DELTA_EXPONENT, uint SIGN_BIT, uint SIGN_BITS, uint DELTA_BIT>
struct InterpretationOf<SEISMIC_CODES_ODD, BITS, OFFSET, TABLE, GRID, INDEX_BITS, LANES, SIGNS, DELTA,
    DELTA_EXPONENT, SIGN_BIT, SIGN_BITS, DELTA_BIT> {
    typedef OddCodes<OFFSET> type;
};
template <uint BITS, int OFFSET, uint TABLE, uint GRID, uint INDEX_BITS, uint LANES, uint SIGNS, bool DELTA,
    uint DELTA_EXPONENT, uint SIGN_BIT, uint SIGN_BITS, uint DELTA_BIT>
struct InterpretationOf<SEISMIC_CODES_TWOS, BITS, OFFSET, TABLE, GRID, INDEX_BITS, LANES, SIGNS, DELTA,
    DELTA_EXPONENT, SIGN_BIT, SIGN_BITS, DELTA_BIT> {
    static_assert(BITS == 8, "two's complement codes are bytes");
    typedef Twos type;
};
template <uint BITS, int OFFSET, uint TABLE, uint GRID, uint INDEX_BITS, uint LANES, uint SIGNS, bool DELTA,
    uint DELTA_EXPONENT, uint SIGN_BIT, uint SIGN_BITS, uint DELTA_BIT>
struct InterpretationOf<SEISMIC_CODES_TABLE, BITS, OFFSET, TABLE, GRID, INDEX_BITS, LANES, SIGNS, DELTA,
    DELTA_EXPONENT, SIGN_BIT, SIGN_BITS, DELTA_BIT> {
    static_assert(BITS == 4, "codebook codes are nibbles");
    typedef Table<TABLE> type;
};
template <uint BITS, int OFFSET, uint TABLE, uint GRID, uint INDEX_BITS, uint LANES, uint SIGNS, bool DELTA,
    uint DELTA_EXPONENT, uint SIGN_BIT, uint SIGN_BITS, uint DELTA_BIT>
struct InterpretationOf<SEISMIC_CODES_E4M3, BITS, OFFSET, TABLE, GRID, INDEX_BITS, LANES, SIGNS, DELTA,
    DELTA_EXPONENT, SIGN_BIT, SIGN_BITS, DELTA_BIT> {
    static_assert(BITS == 8, "E4M3 codes are bytes");
    typedef E4M3Codes type;
};
template <uint BITS, int OFFSET, uint TABLE, uint GRID, uint INDEX_BITS, uint LANES, uint SIGNS, bool DELTA,
    uint DELTA_EXPONENT, uint SIGN_BIT, uint SIGN_BITS, uint DELTA_BIT>
struct InterpretationOf<SEISMIC_CODES_GRID, BITS, OFFSET, TABLE, GRID, INDEX_BITS, LANES, SIGNS, DELTA,
    DELTA_EXPONENT, SIGN_BIT, SIGN_BITS, DELTA_BIT> {
    typedef Grid<GRID, INDEX_BITS, LANES, SIGNS, DELTA, DELTA_EXPONENT, SIGN_BIT, SIGN_BITS, DELTA_BIT> type;
};

template <uint SCALE>
struct ScaleOf;
template <> struct ScaleOf<SEISMIC_SCALE_F16> { typedef F16Scale type; };
template <> struct ScaleOf<SEISMIC_SCALE_BF16> { typedef BF16Scale type; };
template <> struct ScaleOf<SEISMIC_SCALE_F32> { typedef F32Scale type; };
template <> struct ScaleOf<SEISMIC_SCALE_E8M0> { typedef E8M0Scale type; };
template <> struct ScaleOf<SEISMIC_SCALE_UE4M3> { typedef UE4M3Scale type; };

// The interpretation of BITS-bit local codes LOCAL.
template <uint LOCAL, uint BITS, int OFFSET>
struct LocalOf;
template <uint BITS, int OFFSET> struct LocalOf<SEISMIC_CODES_UNSIGNED, BITS, OFFSET> { typedef LocalUnsigned<BITS> type; };
template <uint BITS, int OFFSET> struct LocalOf<SEISMIC_CODES_TWOS, BITS, OFFSET> { typedef LocalTwos<BITS> type; };
template <uint BITS, int OFFSET> struct LocalOf<SEISMIC_CODES_OFFSET, BITS, OFFSET> { typedef LocalOffset<BITS, OFFSET> type; };
template <uint BITS, int OFFSET> struct LocalOf<SEISMIC_CODES_ODD, BITS, OFFSET> { typedef LocalOdd<BITS, OFFSET> type; };

// The coefficients of codes CODES (whose entries carry the local codes at
// bit CARRIED_BIT, where it is not 0) under a bias of kind BIAS
// (`SEISMIC_BIAS_KIND_*`, 0 for none): a minimum of sign BIAS_SIGN, or a
// signed bias of exponent SIGNED_EXPONENT; a scale code is LOCAL_SCALE_BITS
// of a local code's LOCAL_BITS (under a signed bias, all but its sign bit).
template <uint COEFF, uint CODES, uint SCALE, uint BIAS, int BIAS_SIGN, uint GROUP, uint STORAGE, uint FACTOR_GROUP,
    int EXPONENT, int CODE_EXPONENT, uint LOCAL_BITS, uint LOCAL_SCALE_BITS, uint LOCAL, int LOCAL_OFFSET,
    uint CARRIED_BIT, int SIGNED_EXPONENT>
struct CoefficientsOf;
template <uint CODES, uint SCALE, uint BIAS, int BIAS_SIGN, uint GROUP, uint STORAGE, uint FACTOR_GROUP, int EXPONENT,
    int CODE_EXPONENT, uint LOCAL_BITS, uint LOCAL_SCALE_BITS, uint LOCAL, int LOCAL_OFFSET, uint CARRIED_BIT,
    int SIGNED_EXPONENT>
struct CoefficientsOf<SEISMIC_COEFF_DIRECT, CODES, SCALE, BIAS, BIAS_SIGN, GROUP, STORAGE, FACTOR_GROUP, EXPONENT,
    CODE_EXPONENT, LOCAL_BITS, LOCAL_SCALE_BITS, LOCAL, LOCAL_OFFSET, CARRIED_BIT, SIGNED_EXPONENT> {
    typedef Direct<typename ScaleOf<SCALE>::type, GROUP, BIAS == SEISMIC_BIAS_KIND_MINIMUM, STORAGE, CODE_EXPONENT> type;
};
template <uint CODES, uint SCALE, uint BIAS, int BIAS_SIGN, uint GROUP, uint STORAGE, uint FACTOR_GROUP, int EXPONENT,
    int CODE_EXPONENT, uint LOCAL_BITS, uint LOCAL_SCALE_BITS, uint LOCAL, int LOCAL_OFFSET, uint CARRIED_BIT,
    int SIGNED_EXPONENT>
struct CoefficientsOf<SEISMIC_COEFF_HIER, CODES, SCALE, BIAS, BIAS_SIGN, GROUP, STORAGE, FACTOR_GROUP, EXPONENT,
    CODE_EXPONENT, LOCAL_BITS, LOCAL_SCALE_BITS, LOCAL, LOCAL_OFFSET, CARRIED_BIT, SIGNED_EXPONENT> {
    static_assert(CARRIED_BIT == 0 || CODES == SEISMIC_CODES_GRID, "grid entries carry scale codes");
    typedef Hier<typename ScaleOf<SCALE>::type, FACTOR_GROUP, EXPONENT, CODE_EXPONENT, LOCAL_BITS,
        typename LocalOf<LOCAL, LOCAL_SCALE_BITS, LOCAL_OFFSET>::type,
        GROUP, BIAS == SEISMIC_BIAS_KIND_MINIMUM, BIAS_SIGN, CARRIED_BIT,
        metal::conditional_t<BIAS == SEISMIC_BIAS_KIND_SIGNED, SignedBias<SIGNED_EXPONENT, LOCAL_SCALE_BITS>,
            NoSignedBias>> type;
};

} // namespace packets

// ---------------------------------------------------------------------------
// Weight slots. Names are formed by token pasting from the bound prefix; a
// dense weight row is `k` contiguous elements.

#define PACKETS_DECODER(P)                                                                                      \
    packets::Packet<                                                                                            \
        packets::PlanesOf<ELEMENT_CAT(P, _CODES), ELEMENT_CAT(P, _CODE_LOW_BITS), ELEMENT_CAT(P, _CODE_HIGH_BITS)>::type, \
        packets::InterpretationOf<ELEMENT_CAT(P, _CODES), ELEMENT_CAT(P, _CODE_LOW_BITS) + ELEMENT_CAT(P, _CODE_HIGH_BITS), \
            ELEMENT_CAT(P, _CODES_OFFSET), ELEMENT_CAT(P, _CODES_TABLE), ELEMENT_CAT(P, _CODES_GRID), ELEMENT_CAT(P, _CODES_GRID_INDEX_BITS), ELEMENT_CAT(P, _CODES_GRID_LANES), \
            ELEMENT_CAT(P, _CODES_GRID_SIGNS), ELEMENT_CAT(P, _CODES_GRID_DELTA) != 0,                          \
            ELEMENT_CAT(P, _CODES_GRID_DELTA_EXPONENT), ELEMENT_CAT(P, _CODES_GRID_SIGN_BIT),                   \
            ELEMENT_CAT(P, _CODES_GRID_SIGN_BITS), ELEMENT_CAT(P, _CODES_GRID_DELTA_BIT)>::type,                                                       \
        packets::CoefficientsOf<ELEMENT_CAT(P, _COEFF), ELEMENT_CAT(P, _CODES), ELEMENT_CAT(P, _COEFF_SCALE),         \
            ELEMENT_CAT(P, _COEFF_BIAS), ELEMENT_CAT(P, _COEFF_BIAS_SIGN), ELEMENT_CAT(P, _COEFF_GROUP),            \
            ELEMENT_CAT(P, _LOGICAL_GROUP),                                                                     \
            ELEMENT_CAT(P, _COEFF_FACTOR_GROUP), ELEMENT_CAT(P, _COEFF_EXPONENT), ELEMENT_CAT(P, _CODES_EXPONENT),     \
            ELEMENT_CAT(P, _COEFF_LOCAL_BITS), ELEMENT_CAT(P, _COEFF_LOCAL_SCALE_BITS), ELEMENT_CAT(P, _COEFF_LOCAL), \
            ELEMENT_CAT(P, _COEFF_LOCAL_OFFSET),                                                                \
            ELEMENT_CAT(P, _COEFF_CARRIED), ELEMENT_CAT(P, _COEFF_SIGNED_BIAS_EXPONENT)>::type,                 \
        packets::Forms<ELEMENT_CAT(P, _FORM_PACK) != 0, ELEMENT_CAT(P, _FORM_PACK_CENTRE),                       \
            ELEMENT_CAT(P, _FORM_PACK_OFFSET), ELEMENT_CAT(P, _FORM_PACK_RESIDUE), ELEMENT_CAT(P, _FORM_INT8) != 0,  \
            ELEMENT_CAT(P, _FORM_MATRIX_CODES) != 0, ELEMENT_CAT(P, _FORM_MATRIX_RUNS),                          \
            ELEMENT_CAT(P, _FORM_MATRIX_SYMMETRIC) != 0>>
#define PACKETS_ROWS(P)                                                                                \
    packets::Rows16 { ELEMENT_CAT(P, _ROW_STRIDE_BYTES), ELEMENT_CAT(P, _ROW_CODES), ELEMENT_CAT(P, _ROW_HIGH), \
        ELEMENT_CAT(P, _ROW_SCALES), ELEMENT_CAT(P, _ROW_SUPERS), ELEMENT_CAT(P, _TILE_ROWS), 0,            \
        ELEMENT_CAT(P, _ROW_SCALES_BYTES), ELEMENT_CAT(P, _ROW_SUPERS_BYTES) }
#define PACKETS_DENSE(P) packets::Dense<element::Of<ELEMENT_KIND(P)>::type>
#define PACKETS_ROWS16_DENSE(E, k) packets::Rows16 { ulong(k) * E::bytes, 0, 0, 0, 0, 1, 0 }

// Binds slot `SLOT` to the decoder type `TYPE`.
#define PACKETS_BIND(SLOT, TYPE) namespace packets { typedef TYPE SLOT; }

#if defined(KERNEL_W0)
#if ELEMENT_HAS(KERNEL_W0, _KIND_DENSE)
PACKETS_BIND(W0, PACKETS_DENSE(KERNEL_W0))
#define KERNEL_W0_LAYOUT(k) PACKETS_ROWS16_DENSE(element::Of<ELEMENT_KIND(KERNEL_W0)>::type, k)
#elif !ELEMENT_HAS(KERNEL_W0, _LAYOUT_ROWS16) && !ELEMENT_HAS(KERNEL_W0, _LAYOUT_ROWS32)
#error "KERNEL_W0 must be dense or in the rows16 or rows32 layout"
#else
PACKETS_BIND(W0, PACKETS_DECODER(KERNEL_W0))
#define KERNEL_W0_LAYOUT(k) PACKETS_ROWS(KERNEL_W0)
#endif
// The slot's operand recipe (`operands(W)`): whether products over it take
// NVFP4 operand rows.
namespace packets { constant constexpr bool W0_OPERANDS_NVFP4 = ELEMENT_HAS(KERNEL_W0, _OPERANDS_NVFP4); }
#endif

#if defined(KERNEL_W1)
#if ELEMENT_HAS(KERNEL_W1, _KIND_DENSE)
PACKETS_BIND(W1, PACKETS_DENSE(KERNEL_W1))
#define KERNEL_W1_LAYOUT(k) PACKETS_ROWS16_DENSE(element::Of<ELEMENT_KIND(KERNEL_W1)>::type, k)
#elif !ELEMENT_HAS(KERNEL_W1, _LAYOUT_ROWS16) && !ELEMENT_HAS(KERNEL_W1, _LAYOUT_ROWS32)
#error "KERNEL_W1 must be dense or in the rows16 or rows32 layout"
#else
PACKETS_BIND(W1, PACKETS_DECODER(KERNEL_W1))
#define KERNEL_W1_LAYOUT(k) PACKETS_ROWS(KERNEL_W1)
#endif
namespace packets { constant constexpr bool W1_OPERANDS_NVFP4 = ELEMENT_HAS(KERNEL_W1, _OPERANDS_NVFP4); }
#endif

#if defined(KERNEL_W2)
#if ELEMENT_HAS(KERNEL_W2, _KIND_DENSE)
PACKETS_BIND(W2, PACKETS_DENSE(KERNEL_W2))
#define KERNEL_W2_LAYOUT(k) PACKETS_ROWS16_DENSE(element::Of<ELEMENT_KIND(KERNEL_W2)>::type, k)
#elif !ELEMENT_HAS(KERNEL_W2, _LAYOUT_ROWS16) && !ELEMENT_HAS(KERNEL_W2, _LAYOUT_ROWS32)
#error "KERNEL_W2 must be dense or in the rows16 or rows32 layout"
#else
PACKETS_BIND(W2, PACKETS_DECODER(KERNEL_W2))
#define KERNEL_W2_LAYOUT(k) PACKETS_ROWS(KERNEL_W2)
#endif
namespace packets { constant constexpr bool W2_OPERANDS_NVFP4 = ELEMENT_HAS(KERNEL_W2, _OPERANDS_NVFP4); }
#endif

#if defined(KERNEL_W3)
#if ELEMENT_HAS(KERNEL_W3, _KIND_DENSE)
PACKETS_BIND(W3, PACKETS_DENSE(KERNEL_W3))
#define KERNEL_W3_LAYOUT(k) PACKETS_ROWS16_DENSE(element::Of<ELEMENT_KIND(KERNEL_W3)>::type, k)
#elif !ELEMENT_HAS(KERNEL_W3, _LAYOUT_ROWS16) && !ELEMENT_HAS(KERNEL_W3, _LAYOUT_ROWS32)
#error "KERNEL_W3 must be dense or in the rows16 or rows32 layout"
#else
PACKETS_BIND(W3, PACKETS_DECODER(KERNEL_W3))
#define KERNEL_W3_LAYOUT(k) PACKETS_ROWS(KERNEL_W3)
#endif
namespace packets { constant constexpr bool W3_OPERANDS_NVFP4 = ELEMENT_HAS(KERNEL_W3, _OPERANDS_NVFP4); }
#endif
