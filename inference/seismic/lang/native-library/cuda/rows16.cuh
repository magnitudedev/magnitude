// Seismic native library (CUDA): row gathers of packed weights stored in the
// `rows16` layout, the layout of a streamed gathered table's slots on CUDA
// (one row to a tile, so a slot holds the one row a step names).
// `#include <seismic/rows16.cuh>`. A gather decodes one row a value at a
// time.
//
// In `rows16` every row holds its planes one after another at fixed offsets
// (the emitted `_ROW_*` roles). Element i's code is bits of the code planes
// (`registry::code_planes`: crumb i % 4 of byte i / 4 of a 2-bit low plane,
// nibble i & 1 of byte i / 2 of a 4-bit one, byte i of an 8-bit one, bit i or bits 2i, 2i + 1 of a 1- or 2-bit high
// plane) or of the vector-code entry holding it (one per eight elements);
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

// Element i's unsigned code of a LOW-bit low plane and HIGH-bit high plane:
// a sub-byte low plane holds 8 / LOW codes a byte, element i's at bit
// LOW (i % (8 / LOW)) of byte i / (8 / LOW).
template <u32 LOW, u32 HIGH> __device__ __forceinline__ u32 code(const Planes &w, const u8 *row, u64 i) {
    static_assert(LOW == 2 || LOW == 4 || LOW == 8, "a low plane of 2, 4 or 8 bits");
    static_assert(LOW == 8 ? HIGH == 0 : HIGH <= LOW / 2, "a high plane of up to LOW / 2 bits joins a sub-byte low plane");
    if constexpr (LOW == 8)
        return u8_at(row + w.codes + i);
    // log2 of the codes a byte holds.
    constexpr u32 PER = LOW == 4 ? 1 : 2;
    u32 code = (u8_at(row + w.codes + (i >> PER)) >> (LOW * (u32)(i & ((1u << PER) - 1u)))) & ((1u << LOW) - 1u);
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

// A gather of grid codes G (`packets::GridCodes`: one entry per eight
// elements, its whole bytes in the codes plane and E::HIGH bits more in the
// high plane, a row's entries' high bits one after another) under
// hierarchical coefficients K whose local codes are in the scales plane
// (`packets::Hier`) or carried (`packets::CarriedHier`: bit j of a group's
// code at entry bit K::CARRIED of its entry j).
template <class G, class K> struct Grid : Planes {
    typedef typename G::entries E;
    typedef typename G::points Pts;
    static constexpr bool CARRIED_LOCALS = packets::IsCarried<K>::value;
    // Entry k of the row at `row`.
    __device__ __forceinline__ u32 entry_at(const u8 *row, u64 k) const {
        u32 entry = E::read(row + codes + E::BYTES * k);
        if constexpr (E::HIGH != 0) {
            const u64 bit = k * E::HIGH;
            entry |= ((u8_at(row + high + bit / 8) >> (u32)(bit % 8)) & ((1u << E::HIGH) - 1u)) << (8 * E::BYTES);
        }
        return entry;
    }
    // A group's entries, and the words they fill.
    static constexpr u32 ENTRIES = K::GROUP / 8;
    static constexpr int WORDS = (int)(ENTRIES * E::BYTES / 4);
    static_assert(!CARRIED_LOCALS || (K::WIDTH <= ENTRIES && ENTRIES * E::BYTES % 4 == 0),
                  "a group's entries carry its local code and fill whole words");
    __device__ __forceinline__ float value(u64 index, u64 i) const {
        const u8 *row = base + index * stride;
        const u32 lane = (u32)(i & 7u);
        const u64 group = i / K::GROUP;
        float scale, bias;
        u32 entry;
        if constexpr (CARRIED_LOCALS && E::HIGH != 0) {
            u32 field = 0u;
#pragma unroll
            for (u32 j = 0; j < K::WIDTH; ++j)
                field |= ((entry_at(row, group * ENTRIES + j) >> K::CARRIED) & 1u) << j;
            const u32 fields[1] = {field};
            K::form(fields, 0, K::factors(K::factor_word(row + supers, i >> 8)), scale, bias);
            entry = entry_at(row, i >> 3);
        } else if constexpr (CARRIED_LOCALS) {
            // The group's entries, loaded as words once: their carried bits
            // and the element's own entry.
            const u8 *first = row + codes + group * (ENTRIES * E::BYTES);
            u32 w[WORDS];
#pragma unroll
            for (int k = 0; k < WORDS; ++k)
                w[k] = *reinterpret_cast<const u32 *>(first + 4 * k);
            u32 field = 0u;
#pragma unroll
            for (u32 j = 0; j < K::WIDTH; ++j)
                field |= ((packets::bits_of<8 * E::BYTES>(w, 8 * E::BYTES * j) >> K::CARRIED) & 1u) << j;
            const u32 fields[1] = {field};
            K::form(fields, 0, K::factors(K::factor_word(row + supers, i >> 8)), scale, bias);
            entry = packets::bits_of<8 * E::BYTES>(w, 8 * E::BYTES * (u32)((i >> 3) % ENTRIES));
        } else {
            entry = entry_at(row, i >> 3);
            K::at(row + scales, row + supers, group, scale, bias);
        }
        if constexpr (E::HIGH != 0) {
            // (Its bits span a second byte only where HIGH does not divide 8.)
            const u64 bit = E::HIGH * (i >> 3);
            u32 high = u8_at(row + this->high + (bit >> 3));
            if constexpr (8 % E::HIGH != 0)
                high |= u8_at(row + this->high + (bit >> 3) + 1) << 8;
            entry |= ((high >> (bit & 7u)) & ((1u << E::HIGH) - 1u)) << (8 * E::BYTES);
        }
        const uint2 point = Pts::point(entry);
        const u32 word = lane < 4 ? point.x : point.y;
        if constexpr (G::SIGNED) {
            const float code = float((int)(signed char)((G::codes(word, entry) >> (8 * (lane % 4))) & 0xffu));
            return K::BIAS ? __fmaf_rn(scale, code, -bias) : scale * code;
        } else {
            static_assert(!K::BIAS, "magnitude grid codes take no bias");
            const u32 signs = G::signs(entry);
            const float magnitude = float((word >> (8 * (lane % 4))) & 0xffu);
            return scale * ((signs >> lane) & 1u ? -magnitude : magnitude);
        }
    }
};

// The gather decoder of a packed element's emitted decode axes.
template <u32 CODES, u32 LOW, u32 HIGH, int OFFSET, u32 TABLE, u32 GRID, u32 INDEX_BITS, u32 LANES, u32 SIGNS,
          u32 DELTA_BIT, u32 DELTA, class K>
struct Decoder {
    typedef Packed<LOW, HIGH, typename packets::CodesOf<CODES, OFFSET, TABLE>::type, K> type;
};
template <u32 LOW, u32 HIGH, int OFFSET, u32 TABLE, u32 GRID, u32 INDEX_BITS, u32 LANES, u32 SIGNS, u32 DELTA_BIT,
          u32 DELTA, class K>
struct Decoder<SEISMIC_CODES_GRID, LOW, HIGH, OFFSET, TABLE, GRID, INDEX_BITS, LANES, SIGNS, DELTA_BIT, DELTA, K> {
    typedef Grid<typename packets::GridCodesOf<LOW, HIGH, GRID, INDEX_BITS, LANES, SIGNS, DELTA_BIT, DELTA>::type, K>
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
                ELEMENT_CAT(ROWS16_T, _CODES_GRID_SIGNS), ELEMENT_CAT(ROWS16_T, _CODES_GRID_DELTA_BIT),
                ELEMENT_CAT(ROWS16_T, _CODES_GRID_DELTA), PACKETS_COEFFICIENTS(ROWS16_T)>::type T;
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
