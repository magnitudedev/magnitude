// Seismic native library (CUDA): row gathers of packed weights stored in the
// `rows16` layout, the layout of a streamed gathered table's slots on CUDA
// (one row to a tile, so a slot holds the one row a step names).
// `#include <seismic/rows16.cuh>`. A gather decodes one row a value at a
// time.
//
// In `rows16` every row holds its planes one after another at fixed offsets
// (the emitted `_ROW_*` roles). Element i's code is bits of the code planes
// (`registry::code_planes`: bits LOW i of a sub-byte LOW-bit low plane, such
// as nibble i & 1 of byte i / 2 of a 4-bit one, byte i of an 8-bit one, bit i
// or bits 2i, 2i + 1 of a 1- or 2-bit high plane) or of the vector-code entry
// holding it (one per eight elements);
// its coefficients keep the registry's per-row placement, as in `mma16`,
// and are `<seismic/packets.cuh>`'s components.
//
// The table tensor is bound to the slot before this file is included:
//     #define ROWS16_T SEISMIC_TABLE
// which defines `rows16::T` (the decoder type of the bound representation)
// and `ROWS16_T_AT(pointer)` (its value for the table at `pointer`), whose
// `value(row, i)` is element i of row `row`.

#include <seismic/element.cuh>
#include <seismic/packets.cuh>
#include <seismic/tables.cuh>

namespace rows16 {

using element::u32;
using element::u64;
using element::u8;

__device__ __forceinline__ u32 u8_at(const u8 *address) { return *address; }

// A row's planes: its base, stride and the offsets of its code, high,
// scales and supers planes (0 for those it lacks).
struct Planes {
    const u8 *base;
    u64 stride;
    u64 codes;
    u64 high;
    u64 scales;
    u64 supers;
};

// Every address below is written over the element index `i` itself (byte
// `i >> 1` of a nibble plane, not packet `i / 32`'s 16 bytes plus `i % 32`
// halved): the NVIDIA aarch64 compiler traps folding a value's own split
// bit fields added back together.

// Element i's unsigned code of a LOW-bit low plane and HIGH-bit high plane.
template <u32 LOW, u32 HIGH> __device__ __forceinline__ u32 code(const Planes &w, const u8 *row, u64 i) {
    static_assert(LOW == 1 || LOW == 2 || LOW == 4 || LOW == 8, "a low plane of 1, 2, 4 or 8 bits");
    static_assert(LOW < 8 || HIGH == 0, "8-bit codes have no high plane");
    if constexpr (LOW == 8)
        return u8_at(row + w.codes + i);
    // A byte's PER codes, the first lowest.
    constexpr u32 PER = 8 / LOW, SHIFT = PER == 2 ? 1 : PER == 4 ? 2 : 3;
    u32 code = (u8_at(row + w.codes + (i >> SHIFT)) >> (LOW * (u32)(i & (PER - 1)))) & ((1u << LOW) - 1u);
    if constexpr (HIGH == 1)
        code |= ((u8_at(row + w.high + (i >> 3)) >> (u32)(i & 7u)) & 1u) << LOW;
    else if constexpr (HIGH == 2)
        code |= ((u8_at(row + w.high + (i >> 2)) >> (2 * (u32)(i & 3u))) & 3u) << LOW;
    return code;
}

// A gather of codes in planes LOW/HIGH read through interpretation C
// (`packets::CodesOf`), coefficients K (`packets::Direct`, `packets::Hier`):
// value = scale * code - bias (`packets::coded_value`).
template <u32 LOW, u32 HIGH, class C, class K> struct Packed : Planes {
    static_assert(C::template READS<LOW + HIGH>, "the interpretation reads codes of the planes' width");
    // Element i of row `index`.
    __device__ __forceinline__ float value(u64 index, u64 i) const {
        const u8 *row = base + index * stride;
        float scale, bias;
        K::at(row + scales, row + supers, i / K::GROUP, scale, bias);
        return packets::coded_value<C, K::BIAS>(rows16::code<LOW, HIGH>(*this, row, i), scale, bias);
    }
};

// A gather of digit codes D (`packets::DigitPlanes`: a row's groups,
// each `D::group_bytes()` bytes in the codes plane) read through
// interpretation C (`packets::CodesOf`, as `Packed` reads codes) under
// coefficients K.
template <class D, class C, class K> struct Digits : Planes {
    __device__ __forceinline__ float value(u64 index, u64 i) const {
        const u8 *row = base + index * stride;
        u32 byte, digit;
        D::place((u32)(i % D::group_values()), byte, digit);
        const u32 b = u8_at(row + codes + i / D::group_values() * D::group_bytes() + byte);
        float scale, bias;
        K::at(row + scales, row + supers, i / K::GROUP, scale, bias);
        return packets::coded_value<C, K::BIAS>(D::digit(b, digit), scale, bias);
    }
};

// A gather of grid codes G (`packets::GridCodes`: one entry per eight
// elements) under hierarchical coefficients K whose local codes are in the
// scales plane (`packets::Hier`) or carried (`packets::CarriedHier`: bit j
// of a group's code at entry bit K::CARRIED of its entry j). An entry's bits
// past its whole bytes are in the high plane (`G::high_bits()` an entry, in
// entry order), and so are carried codes past them.
template <class G, class K> struct Grid : Planes {
    typedef typename G::entries E;
    static constexpr u32 HIGH = (u32)G::high_bits();
    static_assert(HIGH == 0 || HIGH == 2 || HIGH == 4, "an entry's high bits lie in one byte");
    static constexpr bool CARRIED_LOCALS = packets::IsCarried<K>::value;
    // Entry e's high bits.
    __device__ __forceinline__ u32 high_of(const u8 *row, u64 e) const {
        const u64 bit = e * HIGH;
        return (u8_at(row + high + (bit >> 3)) >> (u32)(bit & 7u)) & ((1u << HIGH) - 1u);
    }
    // Entry e whole: its bytes, then its high bits.
    __device__ __forceinline__ u32 entry(const u8 *row, u64 e) const {
        const u32 bytes = E::read(row + codes + E::BYTES * e);
        if constexpr (HIGH == 0)
            return bytes;
        else
            return bytes | (high_of(row, e) << (8 * E::BYTES));
    }
    // Entry e's carried bit (entry bit K::CARRIED): of its bytes, or of its
    // high bits past them.
    __device__ __forceinline__ u32 carried(const u8 *row, u64 e) const {
        constexpr u32 BYTE_BITS = 8 * E::BYTES;
        if constexpr (K::CARRIED < BYTE_BITS)
            return (E::read(row + codes + E::BYTES * e) >> K::CARRIED) & 1u;
        else
            return (high_of(row, e) >> (K::CARRIED - BYTE_BITS)) & 1u;
    }
    // Element i of row `index`: its lane's value of its entry
    // (`GridCodes::entry_values`) under its group's coefficients, a carried
    // code from the group's entries.
    __device__ __forceinline__ float value(u64 index, u64 i) const {
        const u8 *row = base + index * stride;
        const u32 lane = (u32)(i & 7u);
        const u64 group = i / K::GROUP;
        float scale, bias;
        if constexpr (CARRIED_LOCALS) {
            u32 field = 0u;
#pragma unroll
            for (u32 j = 0; j < K::WIDTH; ++j)
                field |= carried(row, group * (K::GROUP / 8) + j) << j;
            const u32 fields[1] = {field};
            K::form(fields, 0, K::factors(K::factor_word(row + supers, i / K::factor_group())), scale, bias);
        } else {
            K::at(row + scales, row + supers, group, scale, bias);
        }
        const u32 code = (G::entry_values(entry(row, i >> 3), (int)(lane / 4)) >> (8 * (lane % 4))) & 0xFFu;
        return packets::coded_value<packets::Twos, K::BIAS>(code, scale, bias);
    }
};

// The gather decoder of a packed element's emitted decode axes.
template <u32 CODES, u32 LOW, u32 HIGH, int OFFSET, u32 TABLE, u32 GRID, u32 INDEX_BITS, u32 LANES, u32 SIGNS,
          u32 DELTA, u32 DELTA_EXPONENT, u32 SIGN_BIT, u32 SIGN_BITS, u32 DELTA_BIT, u32 DIGIT_RUNS, u32 RUN_BYTES,
          u32 RUN_DIGITS, u32 DIGIT_BITS, class K>
struct Decoder {
    typedef typename packets::Select<
        DIGIT_RUNS != 0,
        Digits<packets::DigitPlanes<DIGIT_RUNS, RUN_BYTES, RUN_DIGITS, DIGIT_BITS>,
               typename packets::CodesOf<CODES, OFFSET, TABLE>::type, K>,
        Packed<LOW, HIGH, typename packets::CodesOf<CODES, OFFSET, TABLE>::type, K>>::type type;
};
template <u32 LOW, u32 HIGH, int OFFSET, u32 TABLE, u32 GRID, u32 INDEX_BITS, u32 LANES, u32 SIGNS, u32 DELTA,
          u32 DELTA_EXPONENT, u32 SIGN_BIT, u32 SIGN_BITS, u32 DELTA_BIT, u32 DIGIT_RUNS, u32 RUN_BYTES,
          u32 RUN_DIGITS, u32 DIGIT_BITS, class K>
struct Decoder<SEISMIC_CODES_GRID, LOW, HIGH, OFFSET, TABLE, GRID, INDEX_BITS, LANES, SIGNS, DELTA, DELTA_EXPONENT,
               SIGN_BIT, SIGN_BITS, DELTA_BIT, DIGIT_RUNS, RUN_BYTES, RUN_DIGITS, DIGIT_BITS, K> {
    typedef Grid<typename packets::GridCodesOf<LOW, HIGH, GRID, INDEX_BITS, LANES, SIGNS, DELTA, DELTA_EXPONENT,
                                               SIGN_BIT, SIGN_BITS, DELTA_BIT>::type,
                 K>
        type;
};

} // namespace rows16

#if defined(ROWS16_T)
#if !ELEMENT_HAS(ROWS16_T, _LAYOUT_ROWS16)
#error "ROWS16_T must be bound in the rows16 layout"
#endif
namespace rows16 {
typedef Decoder<ELEMENT_CAT(ROWS16_T, _CODES), ELEMENT_CAT(ROWS16_T, _CODE_LOW_BITS),
                ELEMENT_CAT(ROWS16_T, _CODE_HIGH_BITS), ELEMENT_CAT(ROWS16_T, _CODES_OFFSET),
                ELEMENT_CAT(ROWS16_T, _CODES_TABLE), ELEMENT_CAT(ROWS16_T, _CODES_GRID),
                ELEMENT_CAT(ROWS16_T, _CODES_GRID_INDEX_BITS), ELEMENT_CAT(ROWS16_T, _CODES_GRID_LANES),
                ELEMENT_CAT(ROWS16_T, _CODES_GRID_SIGNS), ELEMENT_CAT(ROWS16_T, _CODES_GRID_DELTA),
                ELEMENT_CAT(ROWS16_T, _CODES_GRID_DELTA_EXPONENT), ELEMENT_CAT(ROWS16_T, _CODES_GRID_SIGN_BIT),
                ELEMENT_CAT(ROWS16_T, _CODES_GRID_SIGN_BITS), ELEMENT_CAT(ROWS16_T, _CODES_GRID_DELTA_BIT),
                ELEMENT_CAT(ROWS16_T, _CODES_DIGIT_RUNS),
                ELEMENT_CAT(ROWS16_T, _CODES_DIGIT_RUN_BYTES), ELEMENT_CAT(ROWS16_T, _CODES_DIGIT_RUN_DIGITS),
                ELEMENT_CAT(ROWS16_T, _CODES_DIGIT_BITS),
                PACKETS_COEFFICIENTS(ROWS16_T)>::type T;
} // namespace rows16
#define ROWS16_T_AT(pointer)                                                                                   \
    rows16::T {                                                                                               \
        rows16::Planes {                                                                                      \
            (const rows16::u8 *)(pointer), (rows16::u64)ELEMENT_CAT(ROWS16_T, _ROW_STRIDE_BYTES),             \
                (rows16::u64)ELEMENT_CAT(ROWS16_T, _ROW_CODES), (rows16::u64)ELEMENT_CAT(ROWS16_T, _ROW_HIGH), \
                (rows16::u64)ELEMENT_CAT(ROWS16_T, _ROW_SCALES), (rows16::u64)ELEMENT_CAT(ROWS16_T, _ROW_SUPERS) \
        }                                                                                                     \
    }
#endif
