// Seismic native library (CUDA): row gathers of packed weights stored in the
// `rows16` layout, the layout of a streamed gathered table's slots on CUDA
// (one row to a tile, so a slot holds the one row a step names).
// `#include <seismic/rows16.cuh>`. The counterpart of the Vulkan decoders in
// <seismic/packets.glsl>, for gathers, which decode one row a value at a time.
//
// In `rows16` every row holds its planes one after another at fixed offsets.
// A packet is 32 consecutive elements: element j of packet p of a nibble kind
// (q4k, q4g32s, iq4g32, iq4g32s, mxfp4g32, nvfp4g16) is nibble j & 1 of byte
// j / 2 of the packet's 16 bytes of codes; q5k and q5g32 add bit j of the
// packet's 32-bit high plane as the code's fifth bit, q6k bits 2j, 2j + 1 of
// its 64-bit high plane as its fifth and sixth; q8g32s codes are the packet's
// 32 signed bytes; a vector-coded (iq2xs, iq3xxs, iq2xxs) packet is four grid
// entries, one per eight elements. Coefficients keep the registry's per-row
// placement, as in `mma16`.
//
// The table tensor is bound to the slot before this file is included:
//     #define ROWS16_T SEISMIC_TABLE
// which defines `rows16::T` (the decoder type of the bound representation)
// and `ROWS16_T_AT(pointer)` (its value for the table at `pointer`), whose
// `value(row, i)` is element i of row `row`.

#include <seismic/element.cuh>
#include <seismic/grids.cuh>

namespace rows16 {

using element::u16;
using element::u32;
using element::u64;
using element::u8;

enum Kind { Q4K, Q5K, Q6K, Q8, IQ4, IQ4S, Q4G32S, MXFP4, NVFP4, Q5G32S, Q5G32, IQ2XS, IQ3XXS, IQ2XXS };

__device__ __forceinline__ u32 u8_at(const u8 *address) { return *address; }
__device__ __forceinline__ u32 u16_at(const u8 *address) { return *reinterpret_cast<const u16 *>(address); }
__device__ __forceinline__ u32 u32_at(const u8 *address) { return *reinterpret_cast<const u32 *>(address); }
__device__ __forceinline__ float f16_at(const u8 *address) { return seismic_f16_to_f32((u16)u16_at(address)); }

// Registry iq4g32 code table, shared by GGUF IQ4_XS and IQ4_NL imports.
__device__ __forceinline__ float iq4_code(u32 code) {
    constexpr int values[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    return float(values[code]);
}

// E2M1 doubled (GGUF kvalues_mxfp4), shared by mxfp4g32 and nvfp4g16; their
// scales carry the compensating factor 1/2.
__device__ __forceinline__ float e2m1_code(u32 code) {
    constexpr int values[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    return float(values[code]);
}

// E8M0 2^(e - 127) halved: 2^(e - 128) (e < 2 are F32 subnormals); 0xff is NaN.
__device__ __forceinline__ float e8m0_half(u32 e) {
    return __uint_as_float(e < 2u ? 0x00200000u << e : e == 255u ? 0x7fc00000u : (e - 1u) << 23);
}

// UE4M3 (sign bit ignored; 0x7f is NaN) halved: (1 + m/8) 2^(e - 8), or
// m 2^-10 when e = 0.
__device__ __forceinline__ float ue4m3_half(u32 raw) {
    raw &= 0x7fu;
    const u32 e = raw >> 3, m = raw & 7u;
    if (raw == 0x7fu)
        return __uint_as_float(0x7fc00000u);
    return e == 0u ? float(m) * 0.0009765625f : __uint_as_float(((e + 119u) << 23) | (m << 20));
}

// The signed grid code `lane` (0..7) of `point` under the seven sign bits
// `signs`, the eighth making the count of set bits even.
__device__ __forceinline__ float grid_code(uint2 point, u32 signs, u32 lane) {
    signs |= (u32(__popc(signs)) & 1u) << 7;
    const u32 word = lane < 4 ? point.x : point.y;
    const float magnitude = float((word >> (8 * (lane % 4))) & 0xffu);
    return (signs >> lane) & 1u ? -magnitude : magnitude;
}

// Every address below is written over the element index `i` itself (byte
// `i >> 1` of a nibble plane, not packet `i / 32`'s 16 bytes plus `i % 32`
// halved): the NVIDIA aarch64 compiler traps folding a value's own split
// bit fields added back together (seismic-compilation.md).
template <int KIND> struct Table {
    const u8 *base;
    u64 stride;
    u64 codes;  // codes_lo (nibble kinds), codes (q8), grids
    u64 high;   // codes_hi (q5k, q6k, q5g32)
    u64 scales; // local coefficients (k-quants, iq2xs, iq3xxs)
    u64 supers; // super factors or group scales

    // Element i's nibble: nibble i & 1 of byte i >> 1 of the code plane.
    __device__ __forceinline__ u32 nibble(const u8 *row, u64 i) const {
        return (u8_at(row + codes + (i >> 1)) >> (4 * (u32)(i & 1u))) & 15u;
    }

    // Element i's high code bit (bit i of the high plane) as bit 4.
    __device__ __forceinline__ u32 high1(const u8 *row, u64 i) const {
        return ((u8_at(row + high + (i >> 3)) >> (u32)(i & 7u)) & 1u) << 4;
    }

    // The (scale, bias) of k-quant packet p: its (scale6, min6) pair at bit
    // 12 p of the scales plane and its superblock's (d, dmin).
    __device__ __forceinline__ void kquant(const u8 *row, u64 p, float &scale, float &bias) const {
        const u8 *fields = row + scales + ((3 * p) >> 1);
        const u32 pair = (u8_at(fields) | (u8_at(fields + 1) << 8)) >> ((u32)(p & 1u) * 4u);
        const u8 *factors = row + supers + 4 * (p >> 3);
        scale = f16_at(factors) * float(pair & 63u);
        bias = -(f16_at(factors + 2) * float((pair >> 6) & 63u));
    }

    // Element i of row `index`.
    __device__ __forceinline__ float value(u64 index, u64 i) const {
        const u8 *row = base + index * stride;
        if constexpr (KIND == Q4K || KIND == Q5K) {
            float scale, bias;
            kquant(row, i >> 5, scale, bias);
            u32 code = nibble(row, i);
            if constexpr (KIND == Q5K)
                code |= high1(row, i);
            return seismic_fma_rn(scale, float(code), bias);
        } else if constexpr (KIND == Q6K) {
            // Bits 2i, 2i + 1 of the high plane as bits 4, 5; one int8 scale
            // per 16 under the superblock's d.
            const u32 code = nibble(row, i) | (((u8_at(row + high + (i >> 2)) >> (2 * (u32)(i & 3u))) & 3u) << 4);
            const int field = (int)(signed char)u8_at(row + scales + (i >> 4));
            const float d = f16_at(row + supers + 2 * (i >> 8));
            return d * float(field) * (float(code) - 32.0f);
        } else if constexpr (KIND == Q8) {
            const int code = (int)(signed char)u8_at(row + codes + i);
            return f16_at(row + supers + 2 * (i >> 5)) * float(code);
        } else if constexpr (KIND == IQ4) {
            return __uint_as_float(u32_at(row + supers + 4 * (i >> 5))) * iq4_code(nibble(row, i));
        } else if constexpr (KIND == IQ4S) {
            return f16_at(row + supers + 2 * (i >> 5)) * iq4_code(nibble(row, i));
        } else if constexpr (KIND == Q4G32S) {
            // Codes 0..15 with the exact bias -8 d.
            const float d = f16_at(row + supers + 2 * (i >> 5));
            return seismic_fma_rn(d, float(nibble(row, i)), -8.0f * d);
        } else if constexpr (KIND == MXFP4) {
            return e8m0_half(u8_at(row + supers + (i >> 5))) * e2m1_code(nibble(row, i));
        } else if constexpr (KIND == NVFP4) {
            // One UE4M3 scale per 16.
            return ue4m3_half(u8_at(row + supers + (i >> 4))) * e2m1_code(nibble(row, i));
        } else if constexpr (KIND == Q5G32S || KIND == Q5G32) {
            const u32 code = nibble(row, i) | high1(row, i);
            if constexpr (KIND == Q5G32S) {
                // The exact bias -16 d.
                const float d = f16_at(row + supers + 2 * (i >> 5));
                return seismic_fma_rn(d, float(code), -16.0f * d);
            } else {
                // The f16 minimum m.
                const u8 *fields = row + supers + 4 * (i >> 5);
                return seismic_fma_rn(f16_at(fields), float(code), f16_at(fields + 2));
            }
        } else {
            static_assert(KIND == IQ2XS || KIND == IQ3XXS || KIND == IQ2XXS, "a kind with a row decode");
            // Element i's entry (one per eight elements), its grid point and
            // sign bits, and its group's scale: factor * (2 s + 1), s the
            // scale code (iq2xs: one per 16, the others one per 32).
            const u32 lane = (u32)(i & 7u);
            uint2 point;
            u32 signs, field;
            float factor;
            if constexpr (KIND == IQ2XS) {
                const u32 entry = u16_at(row + codes + 2 * (i >> 3));
                point = grids::iq2xs[entry & 511u];
                signs = entry >> 9;
                field = (u8_at(row + scales + (i >> 5)) >> (4 * (u32)((i >> 4) & 1u))) & 15u;
                factor = __uint_as_float(u32_at(row + supers + 4 * (i >> 8)));
            } else if constexpr (KIND == IQ2XXS) {
                // The scale code's bit k is bit 15 of its packet's entry k,
                // under the block's f16 d, the factor d / 8.
                const u32 entry = u16_at(row + codes + 2 * (i >> 3));
                point = grids::iq2xxs[entry & 255u];
                signs = (entry >> 8) & 127u;
                const u8 *entries = row + codes + 8 * (i >> 5);
                const u32 x = u32_at(entries), y = u32_at(entries + 4);
                field = ((x >> 15) & 1u) | ((x >> 30) & 2u) | ((y >> 13) & 4u) | ((y >> 28) & 8u);
                factor = 0.125f * f16_at(row + supers + 2 * (i >> 8));
            } else {
                // Three-byte entries, one per eight elements.
                const u8 *at = row + codes + 3 * (i >> 3);
                const u32 entry = u8_at(at) | (u8_at(at + 1) << 8) | (u8_at(at + 2) << 16);
                point = make_uint2(grids::iq3xxs[entry & 255u], grids::iq3xxs[(entry >> 8) & 255u]);
                signs = (entry >> 16) & 127u;
                field = (u8_at(row + scales + (i >> 6)) >> (4 * (u32)((i >> 5) & 1u))) & 15u;
                factor = __uint_as_float(u32_at(row + supers + 4 * (i >> 8)));
            }
            return factor * float(2u * field + 1u) * grid_code(point, signs, lane);
        }
    }
};

} // namespace rows16

// The decoder of a table of prefix P at `pointer`: its kind, row stride and
// the offsets of the planes its representation has (0 for the others).
#define ROWS16_PLANE(P, NAME) ELEMENT_CAT(P, _PLANE_##NAME##_ROW_OFFSET)
#define ROWS16_MAKE(KIND, P, pointer, CODES, HIGH, SCALES, SUPERS)                                             \
    rows16::Table<rows16::KIND> {                                                                             \
        (const rows16::u8 *)(pointer), (rows16::u64)ELEMENT_CAT(P, _ROW_STRIDE_BYTES), (rows16::u64)(CODES),   \
            (rows16::u64)(HIGH), (rows16::u64)(SCALES), (rows16::u64)(SUPERS)                                 \
    }

// The bound table's kind (`ROWS16_T_KIND`, a compile-time constant, so
// `rows16::T` is its decoder type) and its plane offsets (`ROWS16_T_PLANES`:
// codes, high, scales, supers; 0 for the planes it lacks).
#if defined(ROWS16_T)
#if !ELEMENT_HAS(ROWS16_T, _LAYOUT_ROWS16)
#error "ROWS16_T must be bound in the rows16 layout"
#elif ELEMENT_HAS(ROWS16_T, _REPRESENTATION_Q4K)
#define ROWS16_T_KIND Q4K
#define ROWS16_T_PLANES ROWS16_PLANE(ROWS16_T, CODES_LO), 0, ROWS16_PLANE(ROWS16_T, SCALES), ROWS16_PLANE(ROWS16_T, SUPERS)
#elif ELEMENT_HAS(ROWS16_T, _REPRESENTATION_Q5K) || ELEMENT_HAS(ROWS16_T, _REPRESENTATION_Q6K)
#if ELEMENT_HAS(ROWS16_T, _REPRESENTATION_Q5K)
#define ROWS16_T_KIND Q5K
#else
#define ROWS16_T_KIND Q6K
#endif
#define ROWS16_T_PLANES                                                                                        \
    ROWS16_PLANE(ROWS16_T, CODES_LO), ROWS16_PLANE(ROWS16_T, CODES_HI), ROWS16_PLANE(ROWS16_T, SCALES),         \
        ROWS16_PLANE(ROWS16_T, SUPERS)
#elif ELEMENT_HAS(ROWS16_T, _REPRESENTATION_Q8G32S)
#define ROWS16_T_KIND Q8
#define ROWS16_T_PLANES ROWS16_PLANE(ROWS16_T, CODES), 0, 0, ROWS16_PLANE(ROWS16_T, SUPERS)
#elif ELEMENT_HAS(ROWS16_T, _REPRESENTATION_IQ4G32) || ELEMENT_HAS(ROWS16_T, _REPRESENTATION_IQ4G32S) ||        \
    ELEMENT_HAS(ROWS16_T, _REPRESENTATION_Q4G32S) || ELEMENT_HAS(ROWS16_T, _REPRESENTATION_MXFP4G32) ||        \
    ELEMENT_HAS(ROWS16_T, _REPRESENTATION_NVFP4G16)
#if ELEMENT_HAS(ROWS16_T, _REPRESENTATION_IQ4G32)
#define ROWS16_T_KIND IQ4
#elif ELEMENT_HAS(ROWS16_T, _REPRESENTATION_IQ4G32S)
#define ROWS16_T_KIND IQ4S
#elif ELEMENT_HAS(ROWS16_T, _REPRESENTATION_Q4G32S)
#define ROWS16_T_KIND Q4G32S
#elif ELEMENT_HAS(ROWS16_T, _REPRESENTATION_MXFP4G32)
#define ROWS16_T_KIND MXFP4
#else
#define ROWS16_T_KIND NVFP4
#endif
#define ROWS16_T_PLANES ROWS16_PLANE(ROWS16_T, CODES_LO), 0, 0, ROWS16_PLANE(ROWS16_T, SUPERS)
#elif ELEMENT_HAS(ROWS16_T, _REPRESENTATION_Q5G32S) || ELEMENT_HAS(ROWS16_T, _REPRESENTATION_Q5G32)
#if ELEMENT_HAS(ROWS16_T, _REPRESENTATION_Q5G32S)
#define ROWS16_T_KIND Q5G32S
#else
#define ROWS16_T_KIND Q5G32
#endif
#define ROWS16_T_PLANES ROWS16_PLANE(ROWS16_T, CODES_LO), ROWS16_PLANE(ROWS16_T, CODES_HI), 0, ROWS16_PLANE(ROWS16_T, SUPERS)
#elif ELEMENT_HAS(ROWS16_T, _REPRESENTATION_IQ2XS) || ELEMENT_HAS(ROWS16_T, _REPRESENTATION_IQ3XXS)
#if ELEMENT_HAS(ROWS16_T, _REPRESENTATION_IQ2XS)
#define ROWS16_T_KIND IQ2XS
#else
#define ROWS16_T_KIND IQ3XXS
#endif
#define ROWS16_T_PLANES ROWS16_PLANE(ROWS16_T, GRIDS), 0, ROWS16_PLANE(ROWS16_T, SCALES), ROWS16_PLANE(ROWS16_T, SUPERS)
#elif ELEMENT_HAS(ROWS16_T, _REPRESENTATION_IQ2XXS)
// iq2xxs: its entries carry the scale codes; no scales plane.
#define ROWS16_T_KIND IQ2XXS
#define ROWS16_T_PLANES ROWS16_PLANE(ROWS16_T, GRIDS), 0, 0, ROWS16_PLANE(ROWS16_T, SUPERS)
#else
#error "ROWS16_T binds a representation without a rows16 row decode"
#endif
// The planes expand into separate arguments before `ROWS16_MAKE` takes them.
#define ROWS16_CALL(MACRO, ...) MACRO(__VA_ARGS__)
#define ROWS16_T_AT(pointer) ROWS16_CALL(ROWS16_MAKE, ROWS16_T_KIND, ROWS16_T, pointer, ROWS16_T_PLANES)
namespace rows16 {
typedef Table<ROWS16_T_KIND> T;
} // namespace rows16
#endif
