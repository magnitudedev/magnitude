// Seismic native library (CUDA): packed-weight decoders and weight slot
// bindings, over the CUDA prelude (`seismic_*` PTX helpers).
// `#include <seismic/packets.cuh>`.
//
// Resident CUDA weights use the `mma16` layout (registry
// `PackedRowLayout::code_bit`): the `rows16` row structure (every plane at a
// fixed offset inside a 16 B-aligned row, rows padded to whole 16-row tiles)
// with the code planes of each 16-row tile permuted into m16n8k16 A-fragment
// order. For a tile, the concatenation of one code plane over its 16 rows (b
// bits per code) is laid out as [K/64 k-blocks][32 lanes][4*b bytes]; lane
// l = 4g+t holds, for the four k16 steps s of the k-block, the fragment
// elements in the bits [8b*s, 8b*(s+1)) with slot order
// [a0,a2,a4,a6,a1,a3,a5,a7], each slot b bits wide. So `(word >> 4i) &
// 0x000f000f` isolates the two codes of A register i. Scales and super factors
// keep the rows16 per-row placement.
//
// Weight tensors are bound to slots before this file is included:
//     #define KERNEL_W0 SEISMIC_GATE_WEIGHT
// which defines `packets::W0` (the decoder type of the bound representation)
// and `KERNEL_W0_AT(pointer)` (its value for a tensor base pointer) and
// `KERNEL_W0_MATRIX(pointer, index)` (its value for one matrix of a stacked
// [B, N, K] tensor, such as one expert). Slots W0..W3 exist. Dense element types are `element` (<seismic/element.cuh>).

#pragma once

#include <seismic/element.cuh>
#include <seismic/tables.cuh>

namespace packets {

using element::u8;
using element::u16;
using element::u32;
using element::u64;

template <class A, class B> struct Same {
    static constexpr bool value = false;
};
template <class A> struct Same<A, A> {
    static constexpr bool value = true;
};
template <bool C, class T, class F> struct Select { typedef T type; };
template <class T, class F> struct Select<false, T, F> { typedef F type; };
// A member a configuration has no use for.
struct None {};

__device__ __forceinline__ u32 prmt(u32 a, u32 b, u32 selector) {
    u32 value;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(value) : "r"(a), "r"(b), "r"(selector));
    return value;
}

__device__ __forceinline__ u32 word_of(const uint4 &value, int index) {
    return index == 0 ? value.x : index == 1 ? value.y : index == 2 ? value.z : value.w;
}

// int8 fragments. Within a 32-code group (k16 steps 2h and 2h+1) the s8 MMA's
// virtual k 4t..4t+3 is real k {2t, 2t+1, 2t+8, 2t+9} of step 2h and virtual
// 16 + 4t.. the same of step 2h+1, so one mma16 step word gives the s8 bytes
// of rows g and g+8 of lane (g, t) directly: slots [0, 4, 2, 6] and [1, 5, 3, 7].
struct S8Pair {
    u32 row_g;
    u32 row_g8;
};
// Four 4-bit slot values -> s8 bytes in virtual order.
__device__ __forceinline__ S8Pair s8_nibbles(u32 word) {
    return S8Pair{prmt(word & 0x0F0F0F0Fu, 0u, 0x3120u), prmt((word >> 4) & 0x0F0F0F0Fu, 0u, 0x3120u)};
}
// Codes below 128 per byte less OFFSET (at most 128), as s8: each byte
// with its top bit set is at least OFFSET, so one subtraction borrows
// across no byte, and clearing that bit again leaves code - OFFSET modulo
// 256 (three instructions for any OFFSET).
template <u32 OFFSET> __device__ __forceinline__ u32 s8_less(u32 bytes) {
    static_assert(OFFSET <= 128, "an offset within s8");
    return ((bytes | 0x80808080u) - OFFSET * 0x01010101u) ^ 0x80808080u;
}
// Two's complement 8-bit codes in a step's slots 0..3 (`low`) and 4..7
// (`high`), the form every byte code reaches: the four A registers (exact
// 16-bit operands, `O::bytes`), the s8 fragments of rows g and g+8, and
// one slot's code.
template <class O> __device__ __forceinline__ void slot_operands(u32 low, u32 high, u32 (&a)[4]) {
#pragma unroll
    for (int i = 0; i < 4; ++i)
        a[i] = O::bytes(low, high, i);
}
__device__ __forceinline__ S8Pair slot_s8(u32 low, u32 high) {
    return S8Pair{prmt(low, high, 0x6240u), prmt(low, high, 0x7351u)};
}
__device__ __forceinline__ u32 slot_code(u32 low, u32 high, u32 slot) {
    return ((slot < 4 ? low : high) >> (8 * (slot % 4))) & 0xFFu;
}

// ---------------------------------------------------------------------------
// MMA operand types. Codes are small integers, exact in both 16-bit formats;
// they are formed by OR-ing the code into the mantissa of a power of two and
// subtracting that power (plus the representation's code offset) exactly.

struct OpBF16 {
    __device__ static __forceinline__ u32 pack(float low, float high) { return seismic_pack_bf16x2(low, high); }
    // Two codes (bits 0..7 and 16..23, each < 128) minus `OFFSET`.
    template <u32 OFFSET> __device__ static __forceinline__ u32 codes(u32 pair) {
        const u32 biased = pair | 0x43004300u;     // 128 + code
        constexpr u32 negative = 0xC300u + OFFSET; // -(128 + OFFSET), OFFSET < 128
        u32 value;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;"
            : "=r"(value)
            : "r"(biased), "r"(0x3F803F80u), "r"(negative | (negative << 16)));
        return value;
    }
    // Signed bytes `i` of `slots_low` and `slots_high`.
    __device__ static __forceinline__ u32 bytes(u32 slots_low, u32 slots_high, int i) {
        const int low = ((int)(slots_low << (24 - 8 * i))) >> 24;
        const int high = ((int)(slots_high << (24 - 8 * i))) >> 24;
        return pack((float)low, (float)high);
    }
    __device__ static __forceinline__ void mma(float (&c)[4], const u32 (&a)[4], const u32 (&b)[2]) {
        seismic_mma_m16n8k16_bf16(c, a, b);
    }
    __device__ static __forceinline__ float2 unpack(u32 pair) { return seismic_unpack_bf16x2(pair); }
    static constexpr u32 ONES = 0x3F803F80u;
    // Explicit mantissa bits, and 2^bias (the factor from an integer
    // exponent field's value to its real value).
    static constexpr int MANTISSA = 7;
    static constexpr u32 BIAS_POWER = 0x7F00u;
    // pair * factor (both halves; exact for powers of two), +0 for -0.
    __device__ static __forceinline__ u32 scale(u32 pair, u32 factor) {
        u32 value;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(value) : "r"(pair), "r"(factor), "r"(0u));
        return value;
    }
};

struct OpF16 {
    __device__ static __forceinline__ u32 pack(float low, float high) { return seismic_pack_f16x2(low, high); }
    template <u32 OFFSET> __device__ static __forceinline__ u32 codes(u32 pair) {
        const u32 biased = pair | 0x64006400u;     // 1024 + code
        constexpr u32 negative = 0xE400u + OFFSET; // -(1024 + OFFSET), OFFSET < 1024
        u32 value;
        asm("fma.rn.f16x2 %0, %1, %2, %3;"
            : "=r"(value)
            : "r"(biased), "r"(0x3C003C00u), "r"(negative | (negative << 16)));
        return value;
    }
    __device__ static __forceinline__ u32 bytes(u32 slots_low, u32 slots_high, int i) {
        // 1024 + (code + 128) in each half, then subtract 1152.
        const u32 pair =
            prmt(slots_low ^ 0x80808080u, slots_high ^ 0x80808080u, (u32)i | ((u32)(4 + i) << 4));
        const u32 biased = prmt(pair, 0x64646464u, 0x4140u);
        u32 value;
        asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(value) : "r"(biased), "r"(0x3C003C00u), "r"(0xE480E480u));
        return value;
    }
    __device__ static __forceinline__ void mma(float (&c)[4], const u32 (&a)[4], const u32 (&b)[2]) {
        seismic_mma_m16n8k16_f16(c, a, b);
    }
    __device__ static __forceinline__ float2 unpack(u32 pair) { return seismic_unpack_f16x2(pair); }
    static constexpr u32 ONES = 0x3C003C00u;
    static constexpr int MANTISSA = 10;
    static constexpr u32 BIAS_POWER = 0x7800u;
    __device__ static __forceinline__ u32 scale(u32 pair, u32 factor) {
        u32 value;
        asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(value) : "r"(pair), "r"(factor), "r"(0u));
        return value;
    }
};

// The MMA operand type of a 16-bit dense element. An F32 element has no exact
// 16-bit operand, so it has none.
template <class E> struct OperandOf;
template <> struct OperandOf<element::Bf16> {
    using type = OpBF16;
};
template <> struct OperandOf<element::F16> {
    using type = OpF16;
};
// Whether O is element E's own operand type (E's values enter it exactly).
template <class O, class E> struct NativeOperand {
    static constexpr bool value = false;
};
template <> struct NativeOperand<OpBF16, element::Bf16> {
    static constexpr bool value = true;
};
template <> struct NativeOperand<OpF16, element::F16> {
    static constexpr bool value = true;
};

// ---------------------------------------------------------------------------
// Packed weights in the mma16 layout. Every decoder provides
//   GEMV:  Super fetch_superblock(tile, superblock, kblocks, lane) (the lane
//          chunks of 4 k-blocks), Raw raw(super, q, lane) (k-block q; warp-
//          collective), decode<Op>(raw, step, a), Block block(tile,
//          superblock, lane) (packed coefficients, one vector load per row),
//          coefficients(block, q, c)
//   GEMM:  CHUNKS (16-byte chunks, or CHUNK_BYTES), chunk_source(tile, kb,
//          chunk), from_shared(staged, lane), SUPER_WORDS, super_word(row, superblock, word) (the 4-byte words
//          of a row's coefficient record for one superblock, staged once per
//          superblock), word_kblock(word) (the first k-block of the
//          superblock that reads the word: a partial last superblock stages
//          only the words of its k-blocks), staged_coefficient(record,
//          codes, r, kb, group, scale, bias) (k-block kb of the record's
//          superblock for row r of its tile, whose staged chunks of kb are
//          `codes`)
//   rows:  Raw fetch(tile, kb, lane), code(raw, step, slot), apply(code,
//          scale, bias), coefficient(row, kb, group, scale, bias)
// with value = scale * code - bias per group of GROUP codes (GROUPS per
// 64-code k-block).

// One code plane: rows of a tile are contiguous in the virtual concatenation,
// and a lane chunk never straddles a row (plane bytes per row are whole
// k-blocks).
struct CodePlane {
    u64 offset;
    u64 bytes_per_row;
    __device__ __forceinline__ const u8 *chunk(const u8 *base, u64 stride, u64 tile, u64 virtual_byte) const {
        const u64 row = tile * 16 + virtual_byte / bytes_per_row;
        return base + row * stride + offset + virtual_byte % bytes_per_row;
    }
};

// A tensor's rows in the mma16 row layout (the emitted `_ROW_*` roles):
// its base, row stride and planes (offset 0 and no bytes for those its
// representation lacks).
struct Rows {
    const u8 *base;
    u64 stride;
    CodePlane low;
    CodePlane high;
    u64 scales;
    u64 supers;
    u64 supers_bytes; // payload bytes of a row's supers plane
};

// Coefficients of one k-block for the fragment rows g and g+8, per group.
template <int GROUPS> struct Coefficients {
    float scale[2][GROUPS];
    float bias[2][GROUPS];
};

__device__ __forceinline__ float f16_at(const u8 *address) {
    return seismic_f16_to_f32(*reinterpret_cast<const u16 *>(address));
}

// ---------------------------------------------------------------------------
// Decoders composed from the registry's decode axes (`registry::decode_axes`,
// emitted as each bound element's `_CODE*` and `_COEFF*` macros):
// `Packed<P, C, K, SCALED>` is code planes P (fetch, staging, a step's codes
// as 16-bit operand pairs, s8 bytes or one slot), their interpretation C
// (operands, s8 values and the code term of `apply`) and coefficients K
// (scales and biases, with their staging). SCALED selects the scaled GEMM.

// Code planes `Planes<LOW, HIGH, SUPERBLOCKS>` (`registry::code_planes`): a
// lane chunk's codes as a LOW-bit and a HIGH-bit plane; SUPERBLOCKS when rows
// hold whole 256-code superblocks (the coefficients' storage group is one),
// so a superblock's four k-blocks always exist and its high plane is fetched
// warp-coalesced. `W` is the decoder value (its base,
// stride and planes).
// LOW-bit codes of a 2- or 4-bit low plane, with a 1-bit high plane (bit LOW
// of a code) or none. A lane chunk of a k-block is 4 LOW bytes of the low
// plane (and 4 bytes of a 1-bit high plane); a step's word holds its eight
// slots, LOW bits each (`step_word`). A 2-bit plane's `step_word` spreads its
// step to the 4-bit slot form, each slot's code in the low bits of its
// nibble, so every read below is the nibble form's.
template <u32 LOW> struct LowPlane;
template <> struct LowPlane<4> {
    typedef uint4 Words;
    __device__ static __forceinline__ Words load(const u8 *at) { return seismic_ld_nc_na_v4(at); }
    __device__ static __forceinline__ u32 step_word(const Words &low, int step) { return word_of(low, step); }
};
template <> struct LowPlane<2> {
    typedef uint2 Words;
    __device__ static __forceinline__ Words load(const u8 *at) { return seismic_ld_nc_na_v2(at); }
    // Step s's 16 bits, slot j's code moved from bit 2j to bit 4j.
    __device__ static __forceinline__ u32 step_word(const Words &low, int step) {
        u32 x = ((step < 2 ? low.x : low.y) >> (16 * (step % 2))) & 0xFFFFu;
        x = (x | (x << 8)) & 0x00FF00FFu;
        x = (x | (x << 4)) & 0x0F0F0F0Fu;
        return (x | (x << 2)) & 0x33333333u;
    }
};
template <int LOW, int HIGH, bool SUPERBLOCKS> struct Planes {
    static_assert(LOW == 2 || LOW == 4, "a sub-byte low plane of 2 or 4 bits");
    static_assert(HIGH <= 1, "a 2-bit high plane needs rows of whole superblocks (Planes<4, 2, true>)");
    static constexpr int BITS = LOW + HIGH;
    typedef LowPlane<(u32)LOW> L;
    // Low-plane bytes of a lane chunk and of a tile's k-block, and the 16 B
    // staging chunks of the latter (enumerators: no storage a program keeps).
    enum : u32 { LANE_BYTES = 4 * LOW, KBLOCK_BYTES = 32 * LANE_BYTES, LOW_CHUNKS = KBLOCK_BYTES / 16 };
    struct Raw {
        typename L::Words low;
        u32 high;
    };
    template <class W> __device__ static __forceinline__ Raw fetch(const W &w, u64 tile, u64 kblock, u32 lane) {
        Raw raw;
        raw.low = L::load(w.low.chunk(w.base, w.stride, tile, kblock * KBLOCK_BYTES + lane * LANE_BYTES));
        if constexpr (HIGH == 1)
            raw.high = seismic_ld_nc_u32(w.high.chunk(w.base, w.stride, tile, kblock * 128 + lane * 4));
        else
            raw.high = 0;
        return raw;
    }
    // The lane chunks of one superblock (4 k-blocks). With SUPERBLOCKS the
    // warp's high bits of the superblock are 512 contiguous bytes, fetched as one
    // 16 B load per lane and redistributed by `raw` (warp-collective);
    // otherwise a row's last superblock may be partial and its missing
    // k-blocks are not fetched.
    struct Super {
        Raw raw[4];
        typename Select<SUPERBLOCKS && HIGH == 1, uint4, None>::type high;
    };
    template <class W>
    __device__ static __forceinline__ Super fetch_superblock(const W &w, u64 tile, u64 superblock, u64 kblocks,
                                                             u32 lane) {
        Super super;
#pragma unroll
        for (int q = 0; q < 4; ++q)
            if (SUPERBLOCKS || 4 * superblock + q < kblocks) {
                if constexpr (SUPERBLOCKS) {
                    super.raw[q].low = L::load(
                        w.low.chunk(w.base, w.stride, tile, (4 * superblock + q) * KBLOCK_BYTES + lane * LANE_BYTES));
                    super.raw[q].high = 0;
                } else {
                    super.raw[q] = fetch(w, tile, 4 * superblock + q, lane);
                }
            }
        if constexpr (SUPERBLOCKS && HIGH == 1)
            super.high = seismic_ld_nc_na_v4(w.high.chunk(w.base, w.stride, tile, superblock * 512 + lane * 16));
        return super;
    }
    __device__ static __forceinline__ Raw raw(const Super &super, int q, u32 lane) {
        Raw raw = super.raw[q];
        if constexpr (SUPERBLOCKS && HIGH == 1) {
            // Word 32q + lane of the superblock sits in lane 8q + lane/4, component lane % 4.
            const u32 source = 8 * (u32)q + lane / 4;
            const u32 x = seismic_shfl_idx_u32(super.high.x, source);
            const u32 y = seismic_shfl_idx_u32(super.high.y, source);
            const u32 z = seismic_shfl_idx_u32(super.high.z, source);
            const u32 w = seismic_shfl_idx_u32(super.high.w, source);
            const u32 component = lane % 4;
            raw.high = component == 0 ? x : component == 1 ? y : component == 2 ? z : w;
        }
        return raw;
    }
    // The step's eight 4-bit low codes, one nibble a slot (a codebook's
    // indices).
    __device__ static __forceinline__ u32 nibbles(const Raw &raw, int step) {
        static_assert(LOW == 4, "codebook indices are nibbles");
        return L::step_word(raw.low, step);
    }
    // Register i's two codes of step `step` (slots 2i, 2i + 1, in bits 0..
    // and 16..).
    __device__ static __forceinline__ u32 pair(const Raw &raw, int step, int i) {
        u32 pair = (L::step_word(raw.low, step) >> (4 * i)) & 0x000F000Fu;
        if constexpr (HIGH == 1) {
            const u32 bits = raw.high >> (8 * step + i);
            pair |= ((bits << LOW) & (1u << LOW)) | ((bits << (12 + LOW)) & (1u << (16 + LOW)));
        }
        return pair;
    }
    // Unsigned s8 bytes of rows g and g+8 of step `step` (virtual order).
    __device__ static __forceinline__ S8Pair s8(const Raw &raw, int step) {
        S8Pair pair = s8_nibbles(L::step_word(raw.low, step));
        if constexpr (HIGH == 1) {
            const u32 bits = raw.high >> (8 * step);
            pair.row_g |= ((bits & 1u) << LOW) | (((bits >> 4) & 1u) << (8 + LOW)) |
                          (((bits >> 2) & 1u) << (16 + LOW)) | (((bits >> 6) & 1u) << (24 + LOW));
            pair.row_g8 |= (((bits >> 1) & 1u) << LOW) | (((bits >> 5) & 1u) << (8 + LOW)) |
                           (((bits >> 3) & 1u) << (16 + LOW)) | (((bits >> 7) & 1u) << (24 + LOW));
        }
        return pair;
    }
    // Unsigned code of fragment slot `slot` in step `step`.
    __device__ static __forceinline__ u32 code(const Raw &raw, int step, u32 slot) {
        u32 value = (L::step_word(raw.low, step) >> (4 * slot)) & 15u;
        if constexpr (HIGH == 1)
            value |= ((raw.high >> (8 * step + slot)) & 1u) << LOW;
        return value;
    }
    // GEMM staging: the 16 B chunks of one tile's k-block, low plane first.
    static constexpr int CHUNKS = LOW_CHUNKS + (HIGH == 1 ? 8 : 0);
    template <class W> __device__ static __forceinline__ const u8 *chunk_source(const W &w, u64 tile, u64 kblock, u32 chunk) {
        if constexpr (HIGH != 0)
            if (chunk >= LOW_CHUNKS)
                return w.high.chunk(w.base, w.stride, tile, kblock * 128 + (chunk - LOW_CHUNKS) * 16);
        return w.low.chunk(w.base, w.stride, tile, kblock * KBLOCK_BYTES + chunk * 16);
    }
    __device__ static __forceinline__ Raw from_shared(const u8 *staged, u32 lane) {
        Raw raw;
        raw.low = *reinterpret_cast<const typename L::Words *>(staged + lane * LANE_BYTES);
        if constexpr (HIGH == 1)
            raw.high = *reinterpret_cast<const u32 *>(staged + KBLOCK_BYTES + lane * 4);
        else
            raw.high = 0;
        return raw;
    }
};

// 4 + 2 bit codes in rows of whole superblocks: the warp's high bits of
// a superblock are 1024 contiguous bytes, fetched as two 16 B loads per lane
// and redistributed by `raw` (warp-collective).
template <> struct Planes<4, 2, true> {
    static constexpr int BITS = 6;
    struct Raw {
        uint4 low;
        uint2 high;
    };
    template <class W> __device__ static __forceinline__ Raw fetch(const W &w, u64 tile, u64 kblock, u32 lane) {
        Raw raw;
        raw.low = seismic_ld_nc_na_v4(w.low.chunk(w.base, w.stride, tile, kblock * 512 + lane * 16));
        raw.high = seismic_ld_nc_v2(w.high.chunk(w.base, w.stride, tile, kblock * 256 + lane * 8));
        return raw;
    }
    struct Super {
        uint4 low[4];
        uint4 high[2];
    };
    template <class W>
    __device__ static __forceinline__ Super fetch_superblock(const W &w, u64 tile, u64 superblock, u64, u32 lane) {
        Super super;
#pragma unroll
        for (int q = 0; q < 4; ++q)
            super.low[q] =
                seismic_ld_nc_na_v4(w.low.chunk(w.base, w.stride, tile, (4 * superblock + q) * 512 + lane * 16));
        const u8 *high_chunk = w.high.chunk(w.base, w.stride, tile, superblock * 1024 + lane * 32);
        super.high[0] = seismic_ld_nc_na_v4(high_chunk);
        super.high[1] = seismic_ld_nc_na_v4(high_chunk + 16);
        return super;
    }
    __device__ static __forceinline__ Raw raw(const Super &super, int q, u32 lane) {
        // 8-byte word 32q + lane sits in lane 8q + lane/4, word lane % 4.
        const u32 source = 8 * (u32)q + lane / 4;
        const u32 component = lane % 4;
        const uint4 &half0 = super.high[0];
        const uint4 &half1 = super.high[1];
        const u32 first[4] = {half0.x, half0.z, half1.x, half1.z};
        const u32 second[4] = {half0.y, half0.w, half1.y, half1.w};
        u32 shuffled_first[4], shuffled_second[4];
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            shuffled_first[w] = seismic_shfl_idx_u32(first[w], source);
            shuffled_second[w] = seismic_shfl_idx_u32(second[w], source);
        }
        Raw raw;
        raw.low = super.low[q];
        raw.high.x = component == 0 ? shuffled_first[0] : component == 1 ? shuffled_first[1] : component == 2 ? shuffled_first[2] : shuffled_first[3];
        raw.high.y = component == 0 ? shuffled_second[0] : component == 1 ? shuffled_second[1] : component == 2 ? shuffled_second[2] : shuffled_second[3];
        return raw;
    }
    __device__ static __forceinline__ u32 high_bits(const Raw &raw, int step) {
        return ((step < 2 ? raw.high.x : raw.high.y) >> (16 * (step % 2))) & 0xFFFFu;
    }
    __device__ static __forceinline__ u32 pair(const Raw &raw, int step, int i) {
        const u32 high = high_bits(raw, step);
        u32 pair = (word_of(raw.low, step) >> (4 * i)) & 0x000F000Fu;
        pair |= (((high >> (2 * i)) & 3u) << 4) | (((high >> (2 * i + 8)) & 3u) << 20);
        return pair;
    }
    __device__ static __forceinline__ S8Pair s8(const Raw &raw, int step) {
        S8Pair pair = s8_nibbles(word_of(raw.low, step));
        const u32 high = high_bits(raw, step);
        pair.row_g |= ((high & 3u) << 4) | (((high >> 8) & 3u) << 12) | (((high >> 4) & 3u) << 20) |
                      (((high >> 12) & 3u) << 28);
        pair.row_g8 |= (((high >> 2) & 3u) << 4) | (((high >> 10) & 3u) << 12) | (((high >> 6) & 3u) << 20) |
                       (((high >> 14) & 3u) << 28);
        return pair;
    }
    __device__ static __forceinline__ u32 code(const Raw &raw, int step, u32 slot) {
        return ((word_of(raw.low, step) >> (4 * slot)) & 15u) | (((high_bits(raw, step) >> (2 * slot)) & 3u) << 4);
    }
    static constexpr int CHUNKS = 32 + 16;
    template <class W> __device__ static __forceinline__ const u8 *chunk_source(const W &w, u64 tile, u64 kblock, u32 chunk) {
        if (chunk < 32)
            return w.low.chunk(w.base, w.stride, tile, kblock * 512 + chunk * 16);
        return w.high.chunk(w.base, w.stride, tile, kblock * 256 + (chunk - 32) * 16);
    }
    __device__ static __forceinline__ Raw from_shared(const u8 *staged, u32 lane) {
        Raw raw;
        raw.low = *reinterpret_cast<const uint4 *>(staged + lane * 16);
        raw.high = *reinterpret_cast<const uint2 *>(staged + 512 + lane * 8);
        return raw;
    }
};

// 8-bit codes: rows may end in a partial superblock.
template <> struct Planes<8, 0, false> {
    static constexpr int BITS = 8;
    struct Raw {
        uint4 first;
        uint4 second;
    };
    template <class W> __device__ static __forceinline__ Raw fetch(const W &w, u64 tile, u64 kblock, u32 lane) {
        const u8 *chunk = w.low.chunk(w.base, w.stride, tile, kblock * 1024 + lane * 32);
        Raw raw;
        raw.first = seismic_ld_nc_na_v4(chunk);
        raw.second = seismic_ld_nc_na_v4(chunk + 16);
        return raw;
    }
    struct Super {
        Raw raw[4];
    };
    template <class W>
    __device__ static __forceinline__ Super fetch_superblock(const W &w, u64 tile, u64 superblock, u64 kblocks,
                                                             u32 lane) {
        Super super;
#pragma unroll
        for (int q = 0; q < 4; ++q)
            if (4 * superblock + q < kblocks)
                super.raw[q] = fetch(w, tile, 4 * superblock + q, lane);
        return super;
    }
    __device__ static __forceinline__ Raw raw(const Super &super, int q, u32) { return super.raw[q]; }
    // Slots 0..3 (low) and 4..7 (high) of step `step`.
    __device__ static __forceinline__ void slots(const Raw &raw, int step, u32 &low, u32 &high) {
        const uint4 &half = step < 2 ? raw.first : raw.second;
        low = step % 2 == 0 ? half.x : half.z;
        high = step % 2 == 0 ? half.y : half.w;
    }
    __device__ static __forceinline__ u32 code(const Raw &raw, int step, u32 slot) {
        u32 low, high;
        slots(raw, step, low, high);
        return slot_code(low, high, slot);
    }
    static constexpr int CHUNKS = 64;
    template <class W> __device__ static __forceinline__ const u8 *chunk_source(const W &w, u64 tile, u64 kblock, u32 chunk) {
        return w.low.chunk(w.base, w.stride, tile, kblock * 1024 + chunk * 16);
    }
    __device__ static __forceinline__ Raw from_shared(const u8 *staged, u32 lane) {
        Raw raw;
        raw.first = *reinterpret_cast<const uint4 *>(staged + lane * 32);
        raw.second = *reinterpret_cast<const uint4 *>(staged + lane * 32 + 16);
        return raw;
    }
};

// ---------------------------------------------------------------------------
// Codebook codes: a 4-bit code names one of sixteen exact integer values of a
// codebook (all within s8). A codebook decodes one mma16 step word (the eight
// 4-bit slots of a lane): `decode<O>` into the four A registers and `s8` into
// the s8 fragments of rows g and g+8; after `values` every path is two's
// complement 8-bit codes' (`slot_operands`, `slot_s8`).

// A codebook as little-endian s8 bytes: word w holds the values of codes
// 4w..4w+3.
template <u32 W0, u32 W1, u32 W2, u32 W3> struct Codebook {
    // The s8 table values of the four 4-bit codes in bits 0..15 of `codes`,
    // as bytes 0..3. Two byte permutes look each code's low three bits up in
    // both table halves; a third keeps the half its high bit names (selector
    // +4). Raw `prmt` reads bit 3 of a selector nibble as sign replication,
    // so the lookup selectors carry only the low three bits.
    __device__ static __forceinline__ u32 values(u32 codes) {
        const u32 low = prmt(W0, W1, codes & 0x7777u);
        const u32 high = prmt(W2, W3, codes & 0x7777u);
        return prmt(low, high, 0x3210u | ((codes & 0x8888u) >> 1));
    }
    template <class O> __device__ static __forceinline__ void decode(u32 word, u32 (&a)[4]) {
        slot_operands<O>(values(word & 0xFFFFu), values(word >> 16), a);
    }
    __device__ static __forceinline__ S8Pair s8(u32 word) { return slot_s8(values(word & 0xFFFFu), values(word >> 16)); }
};
// Codebook ID's codes (`codebooks::words<ID>`); an E2M1 codebook's
// (`codebooks::e2m1<ID>`) also decode without the table.
template <u32 ID, bool E2M1 = codebooks::e2m1<ID>::value>
struct CodebookOf {
    typedef Codebook<codebooks::words<ID>::W0, codebooks::words<ID>::W1, codebooks::words<ID>::W2,
                     codebooks::words<ID>::W3>
        type;
};
// An E2M1 codebook, E2M1 doubled. `decode` forms the operands without
// the table: a code's magnitude bits (e1 e0 m) placed at the operand's two
// low exponent bits and top mantissa bit read 2^(e - bias) (1 + m/2) for
// e > 0 and m 2^-bias for e = 0, in both cases the doubled value times
// 2^-bias, so one exact scaling by 2^bias forms it; the code's sign bit is
// the operand's (-0 becomes +0, as the codebook's). The operands are the
// codebook's values exactly, without its conversions.
template <u32 ID>
struct CodebookOf<ID, true> {
    struct type : Codebook<codebooks::words<ID>::W0, codebooks::words<ID>::W1, codebooks::words<ID>::W2,
                           codebooks::words<ID>::W3> {
        template <class O> __device__ static __forceinline__ void decode(u32 word, u32 (&a)[4]) {
            constexpr int P = O::MANTISSA - 1;
            constexpr u32 MAGNITUDE = (7u << P) * 0x00010001u;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                // Register i holds the codes of nibbles i and i + 4 (slots 2i,
                // 2i + 1): its magnitude bits at 4i, sign bit at 4i + 3.
                const u32 magnitude = (P >= 4 * i ? word << (P - 4 * i) : word >> (4 * i - P)) & MAGNITUDE;
                const u32 sign = (word << (12 - 4 * i)) & 0x80008000u;
                a[i] = O::scale(magnitude | sign, O::BIAS_POWER * 0x00010001u);
            }
        }
    };
};

// 2^N, exactly.
template <int N> __host__ __device__ constexpr float power2() {
    return N >= 0 ? (float)(1u << N) : 1.0f / (float)(1u << -N);
}
// x 2^N (exact for the scales it takes).
template <int N> __device__ __forceinline__ float times_power2(float x) {
    if constexpr (N != 0)
        return x * power2<N>();
    else
        return x;
}

// Stored scales: `BYTES` a scale, `decode` its value from its bits and `read`
// it at an address.
struct F32Scale {
    static constexpr int BYTES = 4;
    __device__ static __forceinline__ float decode(u32 bits) { return __uint_as_float(bits); }
    __device__ static __forceinline__ float read(const u8 *at) { return *reinterpret_cast<const float *>(at); }
};
struct F16Scale {
    static constexpr int BYTES = 2;
    __device__ static __forceinline__ float decode(u32 bits) { return seismic_f16_to_f32((u16)bits); }
    __device__ static __forceinline__ float read(const u8 *at) { return f16_at(at); }
};
struct BF16Scale {
    static constexpr int BYTES = 2;
    __device__ static __forceinline__ float decode(u32 bits) { return __uint_as_float(bits << 16); }
    __device__ static __forceinline__ float read(const u8 *at) {
        return decode(*reinterpret_cast<const u16 *>(at));
    }
};
// E8M0: 2^(e - 127) (e = 0 is the F32 subnormal 2^-127); 0xff is NaN.
struct E8M0Scale {
    static constexpr int BYTES = 1;
    __device__ static __forceinline__ float decode(u32 e) {
        e &= 0xFFu;
        return __uint_as_float(e == 0u ? 0x00400000u : e == 255u ? 0x7FC00000u : e << 23);
    }
    __device__ static __forceinline__ float read(const u8 *at) { return decode(*at); }
};
// UE4M3 (sign bit ignored; 0x7f is NaN): (1 + m/8) 2^(e - 7), or m 2^-9 when
// e = 0. The GEMV stages a tile's scales in shared memory (`Direct::stage`).
struct UE4M3Scale {
    static constexpr int BYTES = 1;
    static constexpr bool GEMV_STAGED = true;
    __device__ static __forceinline__ float decode(u32 raw) {
        raw &= 0x7Fu;
        const u32 e = raw >> 3, m = raw & 7u;
        const float normal = __uint_as_float(((e + 120u) << 23) | (m << 20));
        return raw == 0x7Fu ? __uint_as_float(0x7FC00000u) : e == 0u ? (float)m * 0.001953125f : normal;
    }
    __device__ static __forceinline__ float read(const u8 *at) { return decode(*at); }
    // Four consecutive scales of a word, `decode`'s values times 2^N: the
    // fields without their sign bits are E4M3 (0x7f NaN), converted two at a
    // time to f16 and scaled there, a pair a multiply (both exact: the
    // values lie in [2^-9, 448]), where the device converts E4M3 (sm_89 on),
    // one `decode` each before.
    template <int N> __device__ static __forceinline__ void decode4(u32 word, float (&scale)[4]) {
        static_assert(N >= -14 && N <= 7, "scaled UE4M3 values are exact f16 normals or subnormals");
        const u32 fields = word & 0x7F7F7F7Fu;
#if __CUDA_ARCH__ >= 890
        constexpr u32 FACTOR = (u32)(15 + N) << 10; // 2^N in f16
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            u32 pair;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(pair) : "h"((u16)(fields >> (16 * h))));
            if constexpr (N != 0)
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(pair) : "r"(pair), "r"(FACTOR * 0x00010001u));
            const float2 values = seismic_unpack_f16x2(pair);
            scale[2 * h] = values.x;
            scale[2 * h + 1] = values.y;
        }
#else
#pragma unroll
        for (int i = 0; i < 4; ++i)
            scale[i] = times_power2<N>(decode((fields >> (8 * i)) & 0xFFu));
#endif
    }
};

// Whether a scale's GEMV stages a tile's scales (`S::GEMV_STAGED`), and the
// four consecutive scales of a word times 2^N (`S::decode4` where it has
// one).
template <class S, class = void> struct StagedScales {
    static constexpr bool value = false;
};
template <class S> struct StagedScales<S, decltype((void)S::GEMV_STAGED)> {
    static constexpr bool value = S::GEMV_STAGED;
};
template <class S, class = void> struct Decode4 {
    template <int N> __device__ static __forceinline__ void get(u32 word, float (&scale)[4]) {
#pragma unroll
        for (int i = 0; i < 4; ++i)
            scale[i] = times_power2<N>(S::decode(word >> (8 * i)));
    }
};
template <class S> struct Decode4<S, decltype((void)&S::template decode4<0>)> {
    template <int N> __device__ static __forceinline__ void get(u32 word, float (&scale)[4]) {
        S::template decode4<N>(word, scale);
    }
};

// Bits [bit, bit + WIDTH) of WORDS little-endian words (WIDTH <= 32), the
// words they lie in chosen by selects (a runtime index would place the
// words in local memory): where WIDTH divides 32 bit is a multiple of it, so
// the field lies in one word; else in two.
template <u32 WIDTH, int WORDS> __device__ __forceinline__ u32 bits_of(const u32 (&w)[WORDS], u32 bit) {
    constexpr u32 MASK = WIDTH == 32 ? ~0u : (1u << WIDTH) - 1u;
    const u32 word = bit / 32;
    u32 low = w[0], high = 0u;
#pragma unroll
    for (int i = 1; i < WORDS; ++i) {
        low = (u32)i == word ? w[i] : low;
        if constexpr (32 % WIDTH != 0)
            high = (u32)i == word + 1 ? w[i] : high;
    }
    if constexpr (32 % WIDTH == 0)
        return (low >> (bit % 32)) & MASK;
    else
        return (u32)((((u64)high << 32) | low) >> (bit % 32)) & MASK;
}

// Scalar interpretations of a code (`value`, exact): the value of one
// code, for local coefficient codes (`Hier`) and for the fragment
// interpretations below, which build on them.
// Unsigned codes less OFFSET (`Offset(c)`, Unsigned at 0).
template <int OFFSET> struct OffsetCode {
    __device__ static __forceinline__ float value(u32 code) { return (float)((int)code - OFFSET); }
};
// BITS-bit two's complement codes.
template <u32 BITS> struct TwosCode {
    __device__ static __forceinline__ float value(u32 code) { return (float)((int)(code << (32 - BITS)) >> (32 - BITS)); }
};
// 2c + 1 as F32: 2^23 + (2c + 1) less 2^23.
struct OddCode {
    __device__ static __forceinline__ float value(u32 code) { return __uint_as_float(2u * code + 0x4B000001u) - 8388608.0f; }
};

// Interpretations of a step's codes: their scalar `value` (of a code as
// `P::code` reads it), `decode<O, P>` the four A registers, `s8<P>` the s8
// fragments of rows g and g+8, and `READS<BITS>` whether it reads codes of
// BITS bits.

// Linear codes (`OffsetCode`): the offset subtracted exactly in the
// operand (`O::codes<OFFSET>`), in the s8 bytes (`s8_less`) and in the
// integer value; codes below 128.
template <u32 OFFSET> struct Linear : OffsetCode<(int)OFFSET> {
    template <int BITS> static constexpr bool READS = BITS < 8;
    template <class O, class P> __device__ static __forceinline__ void decode(const typename P::Raw &raw, int step,
                                                                           u32 (&a)[4]) {
#pragma unroll
        for (int i = 0; i < 4; ++i)
            a[i] = O::template codes<OFFSET>(P::pair(raw, step, i));
    }
    template <class P> __device__ static __forceinline__ S8Pair s8(const typename P::Raw &raw, int step) {
        const S8Pair pair = P::s8(raw, step);
        if constexpr (OFFSET != 0)
            return S8Pair{s8_less<OFFSET>(pair.row_g), s8_less<OFFSET>(pair.row_g8)};
        else
            return pair;
    }
};

// Two's complement 8-bit codes, the operand bytes themselves, read from a
// step's slots (`P::slots`).
struct Twos : TwosCode<8> {
    template <int BITS> static constexpr bool READS = BITS == 8;
    template <class O, class P> __device__ static __forceinline__ void decode(const typename P::Raw &raw, int step,
                                                                           u32 (&a)[4]) {
        u32 low, high;
        P::slots(raw, step, low, high);
        slot_operands<O>(low, high, a);
    }
    template <class P> __device__ static __forceinline__ S8Pair s8(const typename P::Raw &raw, int step) {
        u32 low, high;
        P::slots(raw, step, low, high);
        return slot_s8(low, high);
    }
};

// Codebook ID's codes, read from a step's 4-bit nibbles: s8 table values.
template <u32 ID> struct Table {
    typedef typename CodebookOf<ID>::type B;
    template <int BITS> static constexpr bool READS = BITS == 4;
    __device__ static __forceinline__ float value(u32 code) { return (float)(int)(signed char)B::values(code); }
    template <class O, class P> __device__ static __forceinline__ void decode(const typename P::Raw &raw, int step,
                                                                           u32 (&a)[4]) {
        B::template decode<O>(P::nibbles(raw, step), a);
    }
    template <class P> __device__ static __forceinline__ S8Pair s8(const typename P::Raw &raw, int step) {
        return B::s8(P::nibbles(raw, step));
    }
};

// value = scale * C's value of `code` - bias (the registry's decode
// recipe): an offset code's value is its exact integer difference
// (`OffsetCode`), times the scale; a stored bias is subtracted exactly.
template <class C, bool BIAS> __device__ __forceinline__ float coded_value(u32 code, float scale, float bias) {
    if constexpr (BIAS)
        return seismic_fma_rn(scale, C::value(code), -bias);
    else
        return scale * C::value(code);
}

// Coefficients. A component reads, for a decoder value `w` (its base,
// stride, `scales` and `supers` offsets and `supers_bytes`), a row's
// coefficient of one group (`coefficient`), a superblock's packed fields of
// rows g and g+8 (`block`) and one k-block's coefficients from them
// (`coefficients`), and the GEMM's staged words of a superblock
// (`SUPER_WORDS`, `super_word`, `word_kblock`, `staged_coefficient`). Scales
// carry the codes' exponent CODE_EXPONENT; a bias is the value subtracted
// (value = scale * code - bias).

// Direct coefficients: a scale S (with BIAS a bias after it) per GROUP
// codes, rows holding whole storage groups of STORAGE codes; GROUPS a
// 64-code k-block.
template <class S, int GROUP_, bool BIAS_, int STORAGE_, int CODE_EXPONENT> struct Direct {
    static constexpr int GROUP = GROUP_;
    static_assert(64 % GROUP == 0, "a k-block holds whole groups");
    static constexpr int GROUPS = 64 / GROUP;
    static constexpr bool BIAS = BIAS_;
    static constexpr int STORAGE = STORAGE_;
    // Bytes of a group's coefficients, and of a superblock's (256 codes),
    // loaded 8 B or 16 B at a time (`load`, and `staged_block`) and staged
    // 4 B a word (`super_word`).
    static_assert(!BIAS || STORAGE == GROUP, "a group's bias follows its scale");
    static constexpr int GROUP_BYTES = S::BYTES * (BIAS ? 2 : 1);
    static constexpr int BLOCK_BYTES = 256 / GROUP * GROUP_BYTES;
    // Staged where its scales stage (`StagedScales`) and its words copy in
    // 16 B chunks.
    static constexpr bool GEMV_STAGED = StagedScales<S>::value && BLOCK_BYTES % 16 == 0;
    static_assert(BLOCK_BYTES == 8 || BLOCK_BYTES % 16 == 0, "a superblock's coefficients are 8 B or whole 16 B");
    static constexpr int BLOCK_WORDS = BLOCK_BYTES / 4;
    // Byte offset of group g's scale; its bias follows it.
    __host__ __device__ static constexpr u64 scale_offset(u64 g) { return g * GROUP_BYTES; }
    __device__ static __forceinline__ float scaled(float scale) { return times_power2<CODE_EXPONENT>(scale); }
    // Group g (of the row's) of the row's planes at `scales` and `supers`
    // (direct coefficients are all in the supers plane).
    __device__ static __forceinline__ void at(const u8 *, const u8 *supers, u64 g, float &scale, float &bias) {
        scale = scaled(S::read(supers + scale_offset(g)));
        bias = BIAS ? -S::read(supers + scale_offset(g) + S::BYTES) : 0.0f;
    }
    template <class W>
    __device__ static __forceinline__ void coefficient(const W &w, u64 row, u64 kblock, int group, float &scale,
                                                       float &bias) {
        at(w.base + row * w.stride + w.scales, w.base + row * w.stride + w.supers,
           kblock * GROUPS + (u64)group, scale, bias);
    }
    // Group g of a superblock's coefficient words (its first group's at
    // byte 0).
    __device__ static __forceinline__ void decode(const u32 *words, int g, float &scale, float &bias) {
        const u64 at = scale_offset((u64)g);
        scale = scaled(S::decode(words[at / 4] >> (8 * (at % 4))));
        if constexpr (BIAS) {
            const u64 b = at + S::BYTES;
            bias = -S::decode(words[b / 4] >> (8 * (b % 4)));
        } else {
            bias = 0.0f;
        }
    }
    // The coefficient words of one superblock of rows g, g+8, 16 B a load
    // (8 B for 8), past the first 16 B where the row's supers plane holds
    // them (a row's last superblock may end inside its final 16 B).
    struct Block {
        u32 words[2][BLOCK_WORDS];
    };
    __device__ static __forceinline__ void put(u32 (&words)[BLOCK_WORDS], int c, uint4 v) {
        words[4 * c] = v.x, words[4 * c + 1] = v.y, words[4 * c + 2] = v.z, words[4 * c + 3] = v.w;
    }
    __device__ static __forceinline__ void load(const u8 *fields, u64 superblock, u64 supers_bytes, u32 (&words)[BLOCK_WORDS]) {
        if constexpr (BLOCK_BYTES == 8) {
            const uint2 v = seismic_ld_nc_v2(fields);
            words[0] = v.x;
            words[1] = v.y;
        } else {
#pragma unroll
            for (int c = 0; c < BLOCK_BYTES / 16; ++c)
                put(words, c,
                    c == 0 || superblock * BLOCK_BYTES + 16 * c < supers_bytes
                        ? seismic_ld_nc_v4(fields + 16 * c)
                        : make_uint4(0, 0, 0, 0));
        }
    }
    template <class W> __device__ static __forceinline__ Block block(const W &w, u64 tile, u64 superblock, u32 lane) {
        Block block;
#pragma unroll
        for (int r = 0; r < 2; ++r)
            load(w.base + (tile * 16 + lane / 4 + 8 * r) * w.stride + w.supers + superblock * BLOCK_BYTES, superblock,
                 w.supers_bytes, block.words[r]);
        return block;
    }
    template <class W>
    __device__ static __forceinline__ void coefficients(const W &, const Block &block, int q, Coefficients<GROUPS> &c) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            if constexpr (GROUPS == 4 && !BIAS && S::BYTES == 1) {
                // A k-block's four one-byte scales are one word.
                Decode4<S>::template get<CODE_EXPONENT>(block.words[r][q], c.scale[r]);
#pragma unroll
                for (int group = 0; group < 4; ++group)
                    c.bias[r][group] = 0.0f;
            } else {
#pragma unroll
                for (int group = 0; group < GROUPS; ++group)
                    decode(block.words[r], q * GROUPS + group, c.scale[r][group], c.bias[r][group]);
            }
        }
    }
    // GEMV staging of a tile's coefficient words where the scales ask for it
    // (`S::GEMV_STAGED`): a row's words are one contiguous run, while a
    // superblock's are 16 B pieces of 16 rows, so the tile's group copies
    // the 16 runs whole into shared memory (`stage`, its threads `thread`
    // of `threads`) and each k-block reads its rows' words there
    // (`staged_block`). Measured on GB10 (NVFP4 4096 -> 18560 and
    // 8192 -> 4096, weights not in L2): 1-7% faster than per-superblock
    // loads. A staged row is its words and 16 B more, so the eight rows of a
    // fragment read fall in distinct banks.
    __host__ __device__ static constexpr u64 staged_pitch(u64 kblocks) {
        return (kblocks + 3) / 4 * BLOCK_BYTES + 16;
    }
    __host__ __device__ static constexpr u64 staged_bytes(u64 kblocks) { return 16 * staged_pitch(kblocks); }
    template <class W>
    __device__ static __forceinline__ void stage(const W &w, u64 tile, u64 kblocks, u8 *shared, u32 thread,
                                                 u32 threads) {
        const u64 pitch = staged_pitch(kblocks);
        const u64 chunks = (pitch - 16) / 16;
        for (u64 c = thread; c < 16 * chunks; c += threads) {
            const u64 row = c / chunks, chunk = c % chunks;
            seismic_cp_async_16(shared + row * pitch + chunk * 16,
                                w.base + (tile * 16 + row) * w.stride + w.supers + chunk * 16);
        }
        seismic_cp_async_commit();
    }
    __device__ static __forceinline__ Block staged_block(const u8 *shared, u64 kblocks, u64 superblock, u32 lane) {
        const u64 pitch = staged_pitch(kblocks);
        Block block;
#pragma unroll
        for (int r = 0; r < 2; ++r)
#pragma unroll
            for (int c = 0; c < BLOCK_BYTES / 16; ++c)
                put(block.words[r], c,
                    reinterpret_cast<const uint4 *>(shared + (lane / 4 + 8 * r) * pitch + superblock * BLOCK_BYTES)[c]);
        return block;
    }
    // GEMM staging: the superblock's coefficient words. A word's first
    // k-block is that of the first group whose scale or bias it holds.
    static constexpr int SUPER_WORDS = BLOCK_WORDS;
    template <class W> __device__ static __forceinline__ const u8 *super_word(const W &w, u64 row, u64 superblock, int word) {
        return w.base + row * w.stride + w.supers + superblock * BLOCK_BYTES + 4 * word;
    }
    __host__ __device__ static constexpr int word_kblock(int word) {
        return (int)((u64)(4 * word) / GROUP_BYTES * GROUP / 64);
    }
    __device__ static __forceinline__ void staged_coefficient(const u32 *words, u64 kblock, int group, float &scale,
                                                              float &bias) {
        decode(words, (int)(kblock % 4) * GROUPS + group, scale, bias);
    }
};

// Hierarchical coefficients: per GROUP codes a LOCAL_BITS-bit scale code
// (and with MIN a minimum code after it), interpreted by L, under a factor FS
// (and a minimum factor) per 256-code superblock: scale = factor
// 2^(EXPONENT + CODE_EXPONENT) L(scale code) and value = scale code - bias
// with bias = -(BIAS_SIGN minimum factor 2^EXPONENT L(minimum code)). A
// superblock's factors are a half word or a word of the supers plane, and
// it is the storage group (`Repr::storage_group`), so rows hold whole ones.
// Where the codes are is the component: the scales plane (`Hier`) or
// carried in grid entries (`CarriedHier`); `HierFactors` is what both share.
template <class FS, int EXPONENT, int CODE_EXPONENT, u32 LOCAL_BITS, class L, int GROUP_, bool MIN, int BIAS_SIGN>
struct HierFactors {
    typedef L local;
    static constexpr int GROUP = GROUP_;
    static constexpr int GROUPS = 64 / GROUP;
    static constexpr bool BIAS = MIN;
    // The storage group: a row holds whole superblocks.
    static constexpr int STORAGE = 256;
    static constexpr bool GEMV_STAGED = false;
    // Bits of a group's codes.
    static constexpr u32 WIDTH = LOCAL_BITS * (MIN ? 2 : 1);
    // Bytes of a superblock's factors, and of the aligned word holding them.
    static constexpr int FACTOR_BYTES = FS::BYTES * (MIN ? 2 : 1);
    static_assert(FACTOR_BYTES == 2 || FACTOR_BYTES == 4, "a superblock's factors are a half word or a word");
    // The superblock's factors from the bits of their word: (scale factor
    // 2^(EXPONENT + CODE_EXPONENT), minimum factor 2^EXPONENT -BIAS_SIGN),
    // the second the factor of a bias as subtracted.
    __device__ static __forceinline__ float2 factors(u32 bits) {
        const float factor = FS::decode(bits) * power2<EXPONENT + CODE_EXPONENT>();
        const float minimum = MIN ? FS::decode(bits >> (8 * FS::BYTES)) * power2<EXPONENT>() : 0.0f;
        return make_float2(factor, BIAS_SIGN < 0 ? minimum : -minimum);
    }
    // The word holding superblock sb's factors in the supers plane at
    // `supers`, its factors shifted to bit 0.
    __device__ static __forceinline__ u32 factor_word(const u8 *supers, u64 sb) {
        const u64 at = sb * FACTOR_BYTES;
        if constexpr (FACTOR_BYTES == 4)
            return seismic_ld_nc_u32(supers + at);
        else
            return *reinterpret_cast<const u16 *>(supers + at);
    }
    // Group j (of a superblock's) from its codes `fields` (group j's at bit
    // `WIDTH` j) under `factor`.
    template <int N>
    __device__ static __forceinline__ void form(const u32 (&fields)[N], int j, float2 factor, float &scale, float &bias) {
        const u32 bit = (u32)j * WIDTH;
        scale = factor.x * L::value(bits_of<LOCAL_BITS>(fields, bit));
        bias = MIN ? factor.y * L::value(bits_of<LOCAL_BITS>(fields, bit + LOCAL_BITS)) : 0.0f;
    }
};

// Hierarchical coefficients whose codes are carried in grid entries: bit j
// of a group's code at entry bit CARRIED of its entry j (no minimum).
template <class FS, int EXPONENT, int CODE_EXPONENT, u32 LOCAL_BITS, class L, int GROUP, u32 CARRIED_>
struct CarriedHier : HierFactors<FS, EXPONENT, CODE_EXPONENT, LOCAL_BITS, L, GROUP, false, 1> {
    static constexpr u32 CARRIED = CARRIED_;
};
template <class K> struct IsCarried {
    static constexpr bool value = false;
};
template <class FS, int EXPONENT, int CODE_EXPONENT, u32 LOCAL_BITS, class L, int GROUP, u32 CARRIED>
struct IsCarried<CarriedHier<FS, EXPONENT, CODE_EXPONENT, LOCAL_BITS, L, GROUP, CARRIED>> {
    static constexpr bool value = true;
};

// Hierarchical coefficients whose codes are in the scales plane: a
// superblock's are SPAN bits after the previous superblock's, read as the
// WORDS words from the word they start in. They start at a multiple of
// SPAN's lowest set bit (at most 32) within that word, so they reach 32 less
// that bit past SPAN; where SPAN fills whole words they are its words.
template <class FS, int EXPONENT, int CODE_EXPONENT, u32 LOCAL_BITS, class L, int GROUP_, bool MIN, int BIAS_SIGN>
struct Hier : HierFactors<FS, EXPONENT, CODE_EXPONENT, LOCAL_BITS, L, GROUP_, MIN, BIAS_SIGN> {
    typedef HierFactors<FS, EXPONENT, CODE_EXPONENT, LOCAL_BITS, L, GROUP_, MIN, BIAS_SIGN> F;
    using F::FACTOR_BYTES;
    using F::GROUP;
    using F::GROUPS;
    using F::WIDTH;
    using F::factor_word;
    using F::factors;
    static constexpr u32 SPAN = 256 / GROUP * WIDTH;
    static constexpr u32 LEAD = (SPAN & (0u - SPAN)) < 32 ? (SPAN & (0u - SPAN)) : 32;
    static constexpr int WORDS = (int)((SPAN + 32 - LEAD + 31) / 32);
    // Byte offset of the word superblock sb's fields start in, and their
    // first bit in it.
    __host__ __device__ static constexpr u64 fields_offset(u64 sb) {
        return SPAN % 32 == 0 ? sb * (4 * WORDS) : sb * SPAN / 32 * 4;
    }
    __host__ __device__ static constexpr u32 fields_shift(u64 sb) { return (u32)(sb * SPAN % 32); }
    // Group g (of a row's) of the scales plane at `scales` and the supers
    // plane at `supers`. Superblocks' fields follow one another, so group
    // g's are at bit `WIDTH` g, addressed from g alone.
    __device__ static __forceinline__ void at(const u8 *scales, const u8 *supers, u64 g, float &scale, float &bias) {
        const u64 bit = g * WIDTH;
        const u32 shift = (u32)(bit % 8);
        const u8 *field = scales + bit / 8;
        const u32 window = (u32)field[0] | (shift + WIDTH > 8 ? (u32)field[1] << 8 : 0u)
                           | (shift + WIDTH > 16 ? (u32)field[2] << 16 : 0u)
                           | (shift + WIDTH > 24 ? (u32)field[3] << 24 : 0u);
        const u32 codes[1] = {window >> shift};
        F::template form<1>(codes, 0, factors(factor_word(supers, g / (256 / GROUP))), scale, bias);
    }
    template <class W>
    __device__ static __forceinline__ void coefficient(const W &w, u64 row, u64 kblock, int group, float &scale,
                                                       float &bias) {
        const u8 *line = w.base + row * w.stride;
        at(line + w.scales, line + w.supers, kblock * GROUPS + (u64)group, scale, bias);
    }
    // A superblock's factors as a Block holds them: a single factor decoded
    // once (in the one register its word takes), two as their word, decoded
    // per k-block (decoded, they would take two).
    struct FactorWord {
        u32 word;
        __device__ __forceinline__ float2 get() const { return factors(word); }
    };
    struct Factor {
        float2 value;
        __device__ __forceinline__ float2 get() const { return value; }
    };
    typedef typename Select<MIN, FactorWord, Factor>::type Held;
    __device__ static __forceinline__ Held hold(u32 word) {
        if constexpr (MIN)
            return Held{word};
        else
            return Held{factors(word)};
    }
    // A superblock's codes and factors of rows g and g+8.
    struct Block {
        u32 fields[2][WORDS];
        Held factors[2];
    };
    template <class W> __device__ static __forceinline__ Block block(const W &w, u64 tile, u64 superblock, u32 lane) {
        Block block;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const u8 *line = w.base + (tile * 16 + lane / 4 + 8 * r) * w.stride;
            const u8 *fields = line + w.scales + fields_offset(superblock);
            if constexpr (WORDS == 4 && SPAN % 32 == 0) {
                const uint4 v = seismic_ld_nc_v4(fields);
                block.fields[r][0] = v.x, block.fields[r][1] = v.y, block.fields[r][2] = v.z, block.fields[r][3] = v.w;
            } else {
#pragma unroll
                for (int i = 0; i < WORDS; ++i)
                    block.fields[r][i] = seismic_ld_nc_u32(fields + 4 * i);
            }
            if constexpr (SPAN % 32 != 0) {
                // The superblock's fields from bit 0.
                const u32 shift = fields_shift(superblock);
#pragma unroll
                for (int i = 0; i < WORDS; ++i)
                    block.fields[r][i] =
                        __funnelshift_r(block.fields[r][i], i + 1 < WORDS ? block.fields[r][i + 1] : 0u, shift);
            }
            block.factors[r] = hold(factor_word(line + w.supers, superblock));
        }
        return block;
    }
    template <class W>
    __device__ static __forceinline__ void coefficients(const W &, const Block &block, int q, Coefficients<GROUPS> &c) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const float2 factor = block.factors[r].get();
#pragma unroll
            for (int group = 0; group < GROUPS; ++group)
                F::template form<WORDS>(block.fields[r], q * GROUPS + group, factor, c.scale[r][group], c.bias[r][group]);
        }
    }
    // GEMM staging: the words the superblock's codes lie in and the aligned
    // word holding its factors (the supers plane is padded to 16 B). Where
    // its codes do not fill whole words, a word's first k-block is taken as
    // the superblock's first (a row holds whole superblocks).
    static constexpr int SUPER_WORDS = WORDS + 1;
    template <class W> __device__ static __forceinline__ const u8 *super_word(const W &w, u64 row, u64 superblock, int word) {
        const u8 *line = w.base + row * w.stride;
        return word < WORDS ? line + w.scales + fields_offset(superblock) + 4 * word
                            : line + w.supers + (superblock * FACTOR_BYTES & ~3ull);
    }
    __host__ __device__ static constexpr int word_kblock(int word) {
        return word < WORDS && SPAN % 32 == 0 ? (int)((u32)(32 * word) / (GROUPS * WIDTH)) : 0;
    }
    __device__ static __forceinline__ void staged_coefficient(const u32 *words, u64 kblock, int group, float &scale,
                                                              float &bias) {
        const u64 sb = kblock / 4;
        const u32 factor = words[WORDS] >> (8 * (sb * FACTOR_BYTES % 4));
        // The record's words are read in place (in shared memory): a register
        // copy of them all would be indexed dynamically, through local
        // memory. The field's word, and the next only where a field can
        // cross into it (`bits_of`).
        const u32 bit = fields_shift(sb) + (u32)((kblock % 4) * GROUPS + group) * WIDTH;
        const u32 *at = words + bit / 32;
        u32 two[2] = {at[0], 0u};
        if constexpr (32 % WIDTH != 0)
            two[1] = bit / 32 + 1 < (u32)WORDS ? at[1] : 0u;
        const u32 codes[1] = {bits_of<WIDTH>(two, bit % 32)};
        F::template form<1>(codes, 0, factors(factor), scale, bias);
    }
};

// The GEMV order of code planes P under coefficient groups of GROUP values
// (`P::Order<GROUP>`, such as grid entries' entry order), where P has one.
template <class P, int GROUP, class = void> struct OrderOf {
    typedef None type;
};
template <class P, int GROUP> struct OrderOf<P, GROUP, decltype((void)sizeof(typename P::template Order<GROUP>))> {
    typedef typename P::template Order<GROUP> type;
};

// A packed decoder of code planes P, interpretation C and coefficients K (see
// the interface above); SCALED selects the scaled GEMM (projection.cuh
// `gemm_tile_half`), OPERANDS that products take NVFP4 operand rows (the
// registry's operand recipe, `_OPERANDS_NVFP4`: an element of any form of
// E2M1 codes in 16-value groups under UE4M3 scales).
template <class P, class C, class K, bool SCALED, bool OPERANDS = false>
struct Packed : Rows, OrderOf<P, K::GROUP>::type {
    static_assert(C::template READS<P::BITS>, "the interpretation reads codes of the planes' width (Linear: below "
                                              "8 bits, Twos: 8, Table: 4)");
    static constexpr int GROUP = K::GROUP;
    static constexpr int GROUPS = K::GROUPS;
    static constexpr bool BIAS = K::BIAS;
    static constexpr bool DENSE = false;
    static constexpr bool SCALED_GEMM = SCALED;
    static constexpr bool NVFP4_OPERANDS = OPERANDS;
    static constexpr bool GEMV_STAGED = K::GEMV_STAGED;
    typedef typename P::Raw Raw;
    typedef typename P::Super Super;
    typedef typename K::Block Block;

    __device__ __forceinline__ Raw fetch(u64 tile, u64 kblock, u32 lane) const { return P::fetch(*this, tile, kblock, lane); }
    __device__ __forceinline__ Super fetch_superblock(u64 tile, u64 superblock, u64 kblocks, u32 lane) const {
        return P::fetch_superblock(*this, tile, superblock, kblocks, lane);
    }
    __device__ __forceinline__ Raw raw(const Super &super, int q, u32 lane) const { return P::raw(super, q, lane); }
    template <class O> __device__ __forceinline__ void decode(const Raw &raw, int step, u32 (&a)[4]) const {
        C::template decode<O, P>(raw, step, a);
    }
    __device__ __forceinline__ S8Pair s8(const Raw &raw, int step) const { return C::template s8<P>(raw, step); }
    __device__ __forceinline__ void coefficient(u64 row, u64 kblock, int group, float &scale, float &bias) const {
        K::coefficient(*this, row, kblock, group, scale, bias);
    }
    __device__ __forceinline__ Block block(u64 tile, u64 superblock, u32 lane) const {
        return K::block(*this, tile, superblock, lane);
    }
    __host__ __device__ static constexpr u64 staged_bytes(u64 kblocks) { return K::staged_bytes(kblocks); }
    __device__ __forceinline__ void stage(u64 tile, u64 kblocks, u8 *shared, u32 thread, u32 threads) const {
        K::stage(*this, tile, kblocks, shared, thread, threads);
    }
    __device__ __forceinline__ Block staged_block(const u8 *shared, u64 kblocks, u64 superblock, u32 lane) const {
        return K::staged_block(shared, kblocks, superblock, lane);
    }
    static constexpr int SUPER_WORDS = K::SUPER_WORDS;
    __device__ __forceinline__ const u8 *super_word(u64 row, u64 superblock, int word) const {
        return K::super_word(*this, row, superblock, word);
    }
    __device__ static __forceinline__ int word_kblock(int word) { return K::word_kblock(word); }
    __device__ __forceinline__ void staged_coefficient(const u32 *words, const u8 *, u32, u64 kblock, int group,
                                                       float &scale, float &bias) const {
        K::staged_coefficient(words, kblock, group, scale, bias);
    }
    __device__ __forceinline__ void coefficients(const Block &block, int q, Coefficients<GROUPS> &c) const {
        K::coefficients(*this, block, q, c);
    }
    __device__ __forceinline__ u32 code(const Raw &raw, int step, u32 slot) const { return P::code(raw, step, slot); }
    __device__ static __forceinline__ float apply(u32 code, float scale, float bias) {
        return coded_value<C, BIAS>(code, scale, bias);
    }
    static constexpr int CHUNKS = P::CHUNKS;
    __device__ __forceinline__ const u8 *chunk_source(u64 tile, u64 kblock, u32 chunk) const {
        return P::chunk_source(*this, tile, kblock, chunk);
    }
    __device__ __forceinline__ Raw from_shared(const u8 *staged, u32 lane) const { return P::from_shared(staged, lane); }
};

// Whether products over weights W take NVFP4 operand rows.
template <class W, class = void> struct nvfp4_operands {
    static constexpr bool value = false;
};
template <class W> struct nvfp4_operands<W, decltype(void(W::NVFP4_OPERANDS))> {
    static constexpr bool value = W::NVFP4_OPERANDS;
};

// ---------------------------------------------------------------------------
// Vector-coded weights (the registry's grid codes: iq2xs, iq3xxs, iq2xxs). An
// entry of the grids plane codes eight values: grid indices name their
// magnitudes (<seismic/tables.cuh>), and seven sign bits, completed by an
// eighth of even parity, their signs. Entries cannot be split across fragment lanes, so a
// tile's grids plane (iq2xs, iq3xxs; iq2xxs's `mma16` storage is records,
// below) is [k-block][row][k-block bytes] (registry
// `PackedRowLayout::code_bit`): lane (g, t) reads the k-block bytes of rows
// g and g + 8 whole (the four lanes of a row share each load) and forms its
// slots of step s from lanes 2t, 2t + 1 of entries 2s (k 2t, 2t + 1) and
// 2s + 1 (k 2t + 8, 2t + 9). From the slots on every path is two's
// complement 8-bit codes': the codes are s8 (magnitudes up to 62), exact
// 16-bit operands and s8 fragments. Their coefficients are hierarchical
// (`Hier`). Rows hold whole 256-value groups.

// The eight lane sign bits of an entry's seven: the eighth makes the count
// of set bits even.
__device__ __forceinline__ u32 grid_signs(u32 signs) { return signs | ((__popc(signs) & 1u) << 7); }

// Four sign bits as four 0x00 / 0xff bytes (bit i to byte i).
__device__ __forceinline__ u32 sign_bytes(u32 bits) { return ((bits * 0x00204081u) & 0x01010101u) * 0xFFu; }

// A grids plane's entries `Entries<BYTES>`: a row's k-block holds eight
// BYTES-byte entries, WORDS words; entry e at bit 8 BYTES e
// (`GridCodes::entry`). `read` is one entry at an address.
template <int BYTES> struct Entries;
template <> struct Entries<2> {
    static constexpr int BYTES = 2;
    static constexpr int WORDS = 4;
    __device__ static __forceinline__ u32 read(const u8 *at) { return *reinterpret_cast<const u16 *>(at); }
    __device__ static __forceinline__ void load(const u8 *global, u32 (&w)[WORDS]) {
        const uint4 v = seismic_ld_nc_v4(global);
        w[0] = v.x, w[1] = v.y, w[2] = v.z, w[3] = v.w;
    }
    __device__ static __forceinline__ void load_shared(const u8 *shared, u32 (&w)[WORDS]) {
        const uint4 v = *reinterpret_cast<const uint4 *>(shared);
        w[0] = v.x, w[1] = v.y, w[2] = v.z, w[3] = v.w;
    }
};
template <> struct Entries<3> {
    static constexpr int BYTES = 3;
    static constexpr int WORDS = 6;
    __device__ static __forceinline__ u32 read(const u8 *at) {
        return (u32)at[0] | ((u32)at[1] << 8) | ((u32)at[2] << 16);
    }
    __device__ static __forceinline__ void load(const u8 *global, u32 (&w)[WORDS]) {
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            const uint2 v = seismic_ld_nc_v2(global + 8 * i);
            w[2 * i] = v.x, w[2 * i + 1] = v.y;
        }
    }
    __device__ static __forceinline__ void load_shared(const u8 *shared, u32 (&w)[WORDS]) {
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            const uint2 v = reinterpret_cast<const uint2 *>(shared)[i];
            w[2 * i] = v.x, w[2 * i + 1] = v.y;
        }
    }
};

// An entry's grid points (`grids::table<GRID>`, LANES values a point, its
// first INDEX_BITS-bit index lowest) and its seven explicit signs, which
// follow the indices.
template <u32 GRID, int INDEX_BITS, int LANES> struct Points;
template <u32 GRID_, int INDEX_BITS> struct Points<GRID_, INDEX_BITS, 8> {
    static constexpr int LANES = 8;
    static constexpr u32 GRID = GRID_;
    static constexpr u32 MASK = (1u << INDEX_BITS) - 1u;
    static constexpr int SIGNS = INDEX_BITS;
    // Lanes 2t, 2t + 1's magnitudes (bytes 0, 1).
    __device__ static __forceinline__ u32 lanes(u32 entry, u32 t) {
        return seismic_ld_nc_u32(reinterpret_cast<const u32 *>(grids::table<GRID>::points()) + 2 * (entry & MASK) +
                                 t / 2) >>
               (16 * (t % 2));
    }
    __device__ static __forceinline__ uint2 point(u32 entry) {
        return seismic_ld_nc_v2(grids::table<GRID>::points() + (entry & MASK));
    }
};
template <u32 GRID_, int INDEX_BITS> struct Points<GRID_, INDEX_BITS, 4> {
    static constexpr int LANES = 4;
    static constexpr u32 GRID = GRID_;
    static constexpr u32 MASK = (1u << INDEX_BITS) - 1u;
    static constexpr int SIGNS = 2 * INDEX_BITS;
    __device__ static __forceinline__ u32 lanes(u32 entry, u32 t) {
        return seismic_ld_nc_u32(grids::table<GRID>::points() + ((entry >> (INDEX_BITS * (t / 2))) & MASK)) >>
               (16 * (t % 2));
    }
    __device__ static __forceinline__ uint2 point(u32 entry) {
        return make_uint2(seismic_ld_nc_u32(grids::table<GRID>::points() + (entry & MASK)),
                          seismic_ld_nc_u32(grids::table<GRID>::points() + ((entry >> INDEX_BITS) & MASK)));
    }
};

// A grid code of entries E and points Pts.
template <class E, class Pts> struct GridCodes {
    typedef E entries;
    typedef Pts points;
    static constexpr int WORDS = E::WORDS;       // words of a row's k-block entries
    __device__ static __forceinline__ void load(const u8 *global, u32 (&w)[WORDS]) { E::load(global, w); }
    __device__ static __forceinline__ void load_shared(const u8 *shared, u32 (&w)[WORDS]) {
        E::load_shared(shared, w);
    }
    __device__ static __forceinline__ u32 signs(u32 entry) { return grid_signs((entry >> Pts::SIGNS) & 127u); }
    // Entry e of a row's k-block words.
    __device__ static __forceinline__ u32 entry(const u32 (&w)[WORDS], int e) {
        return bits_of<8 * E::BYTES>(w, 8 * E::BYTES * (u32)e);
    }
    // Lanes 2t, 2t + 1 of entry e: their magnitudes (bytes 0, 1) and sign
    // bits (bits 0, 1).
    __device__ static __forceinline__ u32 lanes(const u32 (&w)[WORDS], int e, u32 t, u32 &signs_) {
        const u32 entry = GridCodes::entry(w, e);
        signs_ = signs(entry) >> (2 * t);
        return Pts::lanes(entry, t);
    }
    // Entry e whole: its eight magnitudes (lanes 0..3 in x) and eight signs.
    __device__ static __forceinline__ uint2 point(const u32 (&w)[WORDS], int e, u32 &signs_) {
        const u32 entry = GridCodes::entry(w, e);
        signs_ = signs(entry);
        return Pts::point(entry);
    }
};

// The grid code of a packed element's grid axes: entries of LOW / 8 bytes
// (no high plane) and points of INDEX_BITS-bit indices, LANES values each.
template <u32 LOW, u32 HIGH, u32 GRID, u32 INDEX_BITS, u32 LANES> struct GridCodesOf {
    static_assert(HIGH == 0, "grid entries have no high plane");
    typedef GridCodes<Entries<(int)LOW / 8>, Points<GRID, (int)INDEX_BITS, (int)LANES>> type;
};

// Grid code planes (`Packed<GridPlanes<G>, Twos, K>`): the grids plane of
// grid codes G is the low code plane, and a step's slots are two's
// complement 8-bit codes (`Twos`), so they read as 8-bit codes.
template <class G> struct GridPlanes {
    static constexpr int BITS = 8;
    // Bytes of a row's k-block of entries.
    static constexpr int KBLOCK_BYTES = 4 * G::WORDS;
    // A lane's entries of one k-block: rows g and g + 8, and its t.
    struct Raw {
        u32 g[G::WORDS];
        u32 g8[G::WORDS];
        u32 t;
    };
    template <class W> __device__ static __forceinline__ Raw fetch(const W &w, u64 tile, u64 kblock, u32 lane) {
        Raw raw;
        const u64 row = (kblock * 16 + lane / 4) * KBLOCK_BYTES;
        G::load(w.low.chunk(w.base, w.stride, tile, row), raw.g);
        G::load(w.low.chunk(w.base, w.stride, tile, row + 8 * KBLOCK_BYTES), raw.g8);
        raw.t = lane % 4;
        return raw;
    }
    // Rows hold whole 256-value groups, so a superblock's four k-blocks exist.
    struct Super {
        Raw raw[4];
    };
    template <class W>
    __device__ static __forceinline__ Super fetch_superblock(const W &w, u64 tile, u64 superblock, u64, u32 lane) {
        Super super;
#pragma unroll
        for (int q = 0; q < 4; ++q)
            super.raw[q] = fetch(w, tile, 4 * superblock + q, lane);
        return super;
    }
    __device__ static __forceinline__ Raw raw(const Super &super, int q, u32) { return super.raw[q]; }
    // Slots 0..3 (low) and 4..7 (high) of step `step`, as q8's: the codes at
    // k 2t of (row g, entry 2s), (row g + 8, entry 2s), (row g, entry 2s + 1)
    // and (row g + 8, entry 2s + 1), and at k 2t + 1. A byte negates where
    // its sign bit is set (mask 0xff: (m ^ mask) - mask).
    __device__ static __forceinline__ void slots(const Raw &raw, int step, u32 &low, u32 &high) {
        u32 s0, s1, s2, s3;
        const u32 m0 = G::lanes(raw.g, 2 * step, raw.t, s0);
        const u32 m1 = G::lanes(raw.g8, 2 * step, raw.t, s1);
        const u32 m2 = G::lanes(raw.g, 2 * step + 1, raw.t, s2);
        const u32 m3 = G::lanes(raw.g8, 2 * step + 1, raw.t, s3);
        const u32 x = (m0 & 0xFFFFu) | (m1 << 16), y = (m2 & 0xFFFFu) | (m3 << 16);
        const u32 mx = sign_bytes((s0 & 3u) | ((s1 & 3u) << 2)), my = sign_bytes((s2 & 3u) | ((s3 & 3u) << 2));
        const u32 sx = __vsub4(x ^ mx, mx), sy = __vsub4(y ^ my, my);
        low = prmt(sx, sy, 0x6420u);
        high = prmt(sx, sy, 0x7531u);
    }
    __device__ static __forceinline__ u32 code(const Raw &raw, int step, u32 slot) {
        u32 low, high;
        slots(raw, step, low, high);
        return slot_code(low, high, slot);
    }
    // A tile's k-block is its 16 rows' k-block bytes, row by row.
    static constexpr int CHUNKS = KBLOCK_BYTES;
    template <class W> __device__ static __forceinline__ const u8 *chunk_source(const W &w, u64 tile, u64 kblock, u32 chunk) {
        return w.low.chunk(w.base, w.stride, tile, kblock * 16 * KBLOCK_BYTES + chunk * 16);
    }
    __device__ static __forceinline__ Raw from_shared(const u8 *staged, u32 lane) {
        Raw raw;
        G::load_shared(staged + (lane / 4) * KBLOCK_BYTES, raw.g);
        G::load_shared(staged + (lane / 4 + 8) * KBLOCK_BYTES, raw.g8);
        raw.t = lane % 4;
        return raw;
    }
    // GEMV in entry order (`projection::gemv_entries_group`) for coefficient
    // groups of GROUP values: the k of a step is permuted within its group
    // so that lane t's four A values of a row are half of one entry, and an
    // entry is looked up once per row. Step j of group G: lane t holds values
    // 4 h .. 4 h + 3 of entry `gemv_entry`, h = `gemv_half`; the activations
    // read the same places.
    template <int GROUP> struct Order {
        static constexpr bool GEMV_ENTRIES = true;
        static constexpr int GROUP_ENTRIES = GROUP / 8; // entries of a group
        static constexpr int GROUP_STEPS = GROUP / 16;  // MMA steps of a group
        __device__ static __forceinline__ int gemv_entry(int group, u32 t) {
            return GROUP_ENTRIES * group + (int)t / (4 / GROUP_ENTRIES);
        }
        __device__ static __forceinline__ int gemv_half(int step, u32 t) {
            return GROUP_STEPS == 2 ? step : (int)(t % 2);
        }
        // The A pairs of values (4 h, 4 h + 1) and (4 h + 2, 4 h + 3) of entry
        // e of one row's k-block words, signed, as O operands.
        template <class O>
        __device__ static __forceinline__ void entry_pairs(const u32 (&w)[G::WORDS], int e, int h, u32 &low,
                                                           u32 &high) {
            u32 signs;
            const uint2 magnitudes = G::point(w, e, signs);
            const u32 bytes = h ? magnitudes.y : magnitudes.x;
            // sign bit j of the half to bit 8 j + 7, then to the pairs' bits 15 and 31
            const u32 negate = (((signs >> (4 * h)) & 15u) * 0x10204080u) & 0x80808080u;
            low = O::template codes<0>(prmt(bytes, 0u, 0x4140u)) ^ prmt(negate, 0u, 0x1404u);
            high = O::template codes<0>(prmt(bytes, 0u, 0x4342u)) ^ prmt(negate, 0u, 0x3424u);
        }
    };
};

// A carried grid code (iq2xxs), its `mma16` storage records (the registry's
// `PackedRowLayout::record_byte`): a tile's rows hold one record per 256
// columns over the 16 rows' bytes, every record's entries first, 1024 bytes
// each, then every record's factors, 32 bytes each. A record's entries are
// [half h][lane l = 4g + t][16 bytes]; its factors are the 16 rows' f16 d as
// eight words (word g: row g low, row g + 8 high). Word w of a lane's 16 bytes is
// row g + 8 (w % 2) of k-block 2h + w / 2: its entries t (low half) and 4 + t
// (high half), the two entries of each group the lane reads in the GEMV's
// entry order (`projection::gemv_records_group`), so a lane loads exactly its
// bytes of a record, 16 at a time, and one word of factors. An entry carries
// at bit CARRIED bit (entry % 4) of its group's scale code s, under the
// carried hierarchical coefficients K (`CarriedHier`): value = factor L(s) code, every product exact
// in F32.
template <class G, class K> struct Records {
    typedef typename G::points Pts;
    typedef typename K::local L;
    static_assert(IsCarried<K>::value, "records carry their coefficient codes (`CarriedHier`)");
    static constexpr u32 CARRIED = K::CARRIED;
    static_assert(G::WORDS == 4 && Pts::LANES == 8 && K::GROUP == 32 && K::WIDTH == 4 && !K::BIAS &&
                      K::FACTOR_BYTES == 2,
                  "records hold sixteen-bit entries of 8-value points and a 16-bit factor a row, a 4-bit scale code "
                  "per 32 values");
    const u8 *base;
    u64 stride;
    // Offset within a tile of its factors, past every group's entries.
    u64 factors;

    static constexpr int GROUP = K::GROUP;
    static constexpr int GROUPS = K::GROUPS;
    static constexpr bool BIAS = K::BIAS;
    static constexpr bool DENSE = false;
    static constexpr bool SCALED_GEMM = true;
    // Bytes of one group's entries (16 rows of 64) and of its factors.
    static constexpr u64 ENTRIES = 1024;
    static constexpr u64 FACTORS = 32;

    __device__ __forceinline__ const u8 *entries_at(u64 tile, u64 superblock) const {
        return base + tile * 16 * stride + superblock * ENTRIES;
    }
    __device__ __forceinline__ const u8 *factors_at(u64 tile, u64 superblock) const {
        return base + tile * 16 * stride + factors + superblock * FACTORS;
    }

    // ---- GEMV in records (`gemv_records_group`).
    static constexpr bool GEMV_RECORDS = true;
    // A lane's bytes of one record: w[2q + r] row half r of k-block q.
    struct Record {
        u32 w[8];
        u32 factors;
    };
    __device__ __forceinline__ Record record(u64 tile, u64 superblock, u32 lane) const {
        const u8 *at = entries_at(tile, superblock);
        const uint4 low = seismic_ld_nc_na_v4(at + 16 * lane), high = seismic_ld_nc_na_v4(at + 512 + 16 * lane);
        Record record;
        record.w[0] = low.x, record.w[1] = low.y, record.w[2] = low.z, record.w[3] = low.w;
        record.w[4] = high.x, record.w[5] = high.y, record.w[6] = high.z, record.w[7] = high.w;
        record.factors = seismic_ld_nc_u32(factors_at(tile, superblock) + 4 * (lane / 4));
        return record;
    }
    // The scales of the record's groups of the lane's rows, held as the
    // groups' codes and the rows' factors and formed where used: `(r, n)` is
    // the scale of row half r, group n (k-block n / 2), exact. Lane t holds
    // bit t of every scale code; its four lanes' OR is the code
    // (warp-collective). Four registers rather than sixteen keep a records
    // GEMV warp within the registers its SM residency allows
    // (`projection::GemvResidency`).
    struct Scales {
        u32 codes[2];
        float d[2];
        __device__ __forceinline__ float operator()(int r, int n) const {
            // The code of group 2q + h at nibble q + 4h.
            const u32 code = (codes[r] >> (4 * (n / 2 + 4 * (n % 2)))) & 15u;
            return d[r] * L::value(code);
        }
    };
    __device__ static __forceinline__ Scales scales(const Record &record, u32 t) {
        Scales scales;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            u32 codes = 0u;
#pragma unroll
            for (int q = 0; q < 4; ++q)
                codes |= ((record.w[2 * q + r] >> CARRIED) & 0x00010001u) << (4 * q);
            codes <<= t;
            codes |= seismic_shfl_xor_u32(codes, 1);
            codes |= seismic_shfl_xor_u32(codes, 2);
            scales.codes[r] = codes;
            scales.d[r] = K::factors(record.factors >> (16 * r)).x;
        }
        return scales;
    }
    // Each point's eight magnitude codes (`grids::magnitudes`: a magnitude
    // shifted right by `SHIFT`), value i at nibble i. The GEMV's KSPLIT warps of a tile group form a copy in
    // shared memory.
    struct Table {
        u32 points[Pts::MASK + 1];
    };
    // The magnitude codes of a point word's four lanes, lane i at nibble i.
    __device__ static __forceinline__ u32 magnitude_codes(u32 magnitudes) {
        const u32 c = (magnitudes >> grids::magnitudes<Pts::GRID>::SHIFT) & 0x03030303u;
        return (c & 0xFu) | ((c >> 4) & 0xF0u) | ((c >> 8) & 0xF00u) | ((c >> 12) & 0xF000u);
    }
    __device__ static __forceinline__ void form_table(Table &table, u32 thread, u32 threads) {
        for (u32 i = thread; i <= Pts::MASK; i += threads) {
            const uint2 point = Pts::point(i);
            table.points[i] = magnitude_codes(point.x) | (magnitude_codes(point.y) << 16);
        }
    }
    // The operand pairs (values 2j, 2j + 1 in a[j]) of `entry` (its low 16
    // bits), signed: each half is formed by one PRMT from the operand bytes
    // of the grid's magnitudes (`GridOperands`), its selector bytes looked
    // up by magnitude code.
    template <class O> __device__ static __forceinline__ void operands(const Table &table, u32 entry, u32 (&a)[4]);

    // ---- GEMV in fragment order (`gemv_mma_group`, a pair with another
    // representation): the k-block's entries of rows g and g + 8 gathered
    // from the quad, and the record's scales.
    typedef typename GridPlanes<G>::Raw Raw;
    typedef Record Super;
    __device__ __forceinline__ Super fetch_superblock(u64 tile, u64 superblock, u64, u32 lane) const {
        return record(tile, superblock, lane);
    }
    // Rows g, g + 8 of k-block q: entries 2i, 2i + 1 in word i (warp-collective).
    __device__ __forceinline__ Raw raw(const Super &super, int q, u32 lane) const {
        Raw raw;
        u32 e[2][4];
#pragma unroll
        for (int r = 0; r < 2; ++r)
#pragma unroll
            for (int j = 0; j < 4; ++j)
                e[r][j] = seismic_shfl_idx_u32(super.w[2 * q + r], (lane & ~3u) + (u32)j);
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            u32 *row = r == 0 ? raw.g : raw.g8;
            row[0] = prmt(e[r][0], e[r][1], 0x5410u);
            row[1] = prmt(e[r][2], e[r][3], 0x5410u);
            row[2] = prmt(e[r][0], e[r][1], 0x7632u);
            row[3] = prmt(e[r][2], e[r][3], 0x7632u);
        }
        raw.t = lane % 4;
        return raw;
    }
    struct Block {
        float scale[2][8];
    };
    __device__ __forceinline__ Block block(u64 tile, u64 superblock, u32 lane) const {
        const Scales scale = scales(record(tile, superblock, lane), lane % 4);
        Block block;
#pragma unroll
        for (int r = 0; r < 2; ++r)
#pragma unroll
            for (int n = 0; n < 8; ++n)
                block.scale[r][n] = scale(r, n);
        return block;
    }
    __device__ __forceinline__ void coefficients(const Block &block, int q, Coefficients<GROUPS> &c) const {
#pragma unroll
        for (int r = 0; r < 2; ++r)
#pragma unroll
            for (int group = 0; group < GROUPS; ++group)
                c.scale[r][group] = block.scale[r][2 * q + group];
    }
    template <class O> __device__ __forceinline__ void decode(const Raw &raw, int step, u32 (&a)[4]) const {
        Twos::template decode<O, GridPlanes<G>>(raw, step, a);
    }
    __device__ __forceinline__ S8Pair s8(const Raw &raw, int step) const {
        return Twos::template s8<GridPlanes<G>>(raw, step);
    }

    // ---- GEMM: a k-block's chunks are its eight bytes of each lane of the
    // record, staged [lane][8 bytes].
    static constexpr int CHUNKS = 32;
    static constexpr int CHUNK_BYTES = 8;
    __device__ __forceinline__ const u8 *chunk_source(u64 tile, u64 kblock, u32 chunk) const {
        return entries_at(tile, kblock / 4) + kblock % 4 / 2 * 512 + chunk * 16 + kblock % 2 * 8;
    }
    __device__ __forceinline__ Raw from_shared(const u8 *staged, u32 lane) const {
        Raw raw;
        // Lanes 4g .. 4g + 3's staged bytes: rows g, g + 8 of entries j, 4 + j.
        const uint4 low = *reinterpret_cast<const uint4 *>(staged + 32 * (lane / 4));
        const uint4 high = *reinterpret_cast<const uint4 *>(staged + 32 * (lane / 4) + 16);
        const uint2 e[4] = {make_uint2(low.x, low.y), make_uint2(low.z, low.w), make_uint2(high.x, high.y),
                            make_uint2(high.z, high.w)};
        raw.g[0] = prmt(e[0].x, e[1].x, 0x5410u), raw.g[1] = prmt(e[2].x, e[3].x, 0x5410u);
        raw.g[2] = prmt(e[0].x, e[1].x, 0x7632u), raw.g[3] = prmt(e[2].x, e[3].x, 0x7632u);
        raw.g8[0] = prmt(e[0].y, e[1].y, 0x5410u), raw.g8[1] = prmt(e[2].y, e[3].y, 0x5410u);
        raw.g8[2] = prmt(e[0].y, e[1].y, 0x7632u), raw.g8[3] = prmt(e[2].y, e[3].y, 0x7632u);
        raw.t = lane % 4;
        return raw;
    }
    // A row's record word: its factors word (rows g, g + 8).
    static constexpr int SUPER_WORDS = 1;
    __device__ __forceinline__ const u8 *super_word(u64 row, u64 superblock, int) const {
        return factors_at(row / 16, superblock) + 4 * (row % 8);
    }
    __device__ static __forceinline__ int word_kblock(int) { return 0; }
    // Row r's scale code of `group` from its four entries' carried bits in
    // the staged chunks of the k-block.
    __device__ __forceinline__ void staged_coefficient(const u32 *words, const u8 *codes, u32 r, u64, int group,
                                                       float &scale, float &bias) const {
        // Lanes 4 (r % 8) + j's chunks: word 2j + r / 8 is row r's entries j, 4 + j.
        const uint4 low = *reinterpret_cast<const uint4 *>(codes + 32 * (r % 8));
        const uint4 high = *reinterpret_cast<const uint4 *>(codes + 32 * (r % 8) + 16);
        const u32 half = r / 8, shift = CARRIED + 16 * group;
        const u32 code = (((half ? low.y : low.x) >> shift) & 1u) | ((((half ? low.w : low.z) >> shift) & 1u) << 1) |
                         ((((half ? high.y : high.x) >> shift) & 1u) << 2) |
                         ((((half ? high.w : high.z) >> shift) & 1u) << 3);
        scale = K::factors(words[0] >> (16 * (r / 8))).x * L::value(code);
        bias = 0.0f;
    }

    // ---- One row's values (`row_values16`), read by a lane alone: word r
    // of the 16 bytes of lane `l` holding row g + 8 r's entries of k-block
    // `kblock` (entries l % 4, low half, and 4 + l % 4, high half).
    __device__ __forceinline__ u32 entry_word(u64 tile, u64 kblock, u32 lane, u32 r) const {
        return seismic_ld_nc_u32(entries_at(tile, kblock / 4) + kblock % 4 / 2 * 512 + 16 * lane +
                                 4 * (kblock % 2 * 2 + r));
    }
    // Rows g, g + 8 of k-block `kblock`: entries 2i, 2i + 1 in word i, as
    // `raw` gathers them from the quad.
    __device__ __forceinline__ Raw fetch(u64 tile, u64 kblock, u32 lane) const {
        Raw raw;
        const u32 quad = lane & ~3u;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            u32 e[4];
#pragma unroll
            for (int j = 0; j < 4; ++j)
                e[j] = entry_word(tile, kblock, quad + (u32)j, (u32)r);
            u32 *row = r == 0 ? raw.g : raw.g8;
            row[0] = prmt(e[0], e[1], 0x5410u);
            row[1] = prmt(e[2], e[3], 0x5410u);
            row[2] = prmt(e[0], e[1], 0x7632u);
            row[3] = prmt(e[2], e[3], 0x7632u);
        }
        raw.t = lane % 4;
        return raw;
    }
    // Group `group` of k-block `kblock` of stored row `row`: factor L(s), s's
    // bit j at bit CARRIED of the group's entry j.
    __device__ __forceinline__ void coefficient(u64 row, u64 kblock, int group, float &scale, float &bias) const {
        const u64 tile = row / 16;
        const u32 g = (u32)(row % 8), r = (u32)(row % 16 / 8);
        u32 s = 0;
#pragma unroll
        for (u32 j = 0; j < 4; ++j)
            s |= ((entry_word(tile, kblock, 4 * g + j, r) >> (CARRIED + 16 * group)) & 1u) << j;
        const u32 d = seismic_ld_nc_u32(factors_at(tile, kblock / 4) + 4 * g);
        scale = K::factors(d >> (16 * r)).x * L::value(s);
        bias = 0.0f;
    }
    __device__ __forceinline__ u32 code(const Raw &raw, int step, u32 slot) const {
        return GridPlanes<G>::code(raw, step, slot);
    }
    __device__ static __forceinline__ float apply(u32 code, float scale, float bias) {
        return coded_value<Twos, BIAS>(code, scale, bias);
    }
};

// A records grid's magnitude operands per operand type O
// (`grids::magnitudes`): its operand bytes and the PRMT selector byte of each
// magnitude code's positive operand; bf16 negates by selector (sign bit i to
// bit 6 of the selector), f16 by the operand's sign bit.
template <u32 GRID, class O> struct GridOperands;
template <u32 GRID> struct GridOperands<GRID, OpBF16> : grids::magnitudes<GRID>::bf16 {
    static constexpr bool SIGNED_SELECTORS = true;
};
template <u32 GRID> struct GridOperands<GRID, OpF16> : grids::magnitudes<GRID>::f16 {
    static constexpr bool SIGNED_SELECTORS = false;
};

template <class G, class K>
template <class O>
__device__ __forceinline__ void Records<G, K>::operands(const Table &table, u32 entry, u32 (&a)[4]) {
    using P = GridOperands<Pts::GRID, O>;
    const u32 codes = table.points[entry & Pts::MASK];
    const u32 signs = G::signs(entry);
    // Selector bytes of values 0..3 and 4..7.
    u32 low = prmt(P::SELECTORS, 0u, codes), high = prmt(P::SELECTORS, 0u, codes >> 16);
    if constexpr (P::SIGNED_SELECTORS) {
        // Sign bit i to bit 6 of selector byte i % 4.
        low |= ((signs & 15u) * (0x00204081u << 6)) & 0x40404040u;
        high |= ((signs >> 4) * (0x00204081u << 6)) & 0x40404040u;
    }
    a[0] = prmt(P::LOW, P::HIGH, low);
    a[1] = prmt(P::LOW, P::HIGH, low >> 16);
    a[2] = prmt(P::LOW, P::HIGH, high);
    a[3] = prmt(P::LOW, P::HIGH, high >> 16);
    if constexpr (!P::SIGNED_SELECTORS) {
        // Sign bits 2j, 2j + 1 to bits 15, 31 of pair j.
#pragma unroll
        for (int j = 0; j < 4; ++j)
            a[j] ^= (((signs >> (2 * j)) & 1u) << 15) | (((signs >> (2 * j + 1)) & 1u) << 31);
    }
}


// Dense weights of element E (row-major [rows, K], unpadded, row stride
// `stride` elements) in the decoder interface: the lane's fragment elements are
// loaded directly (rows past `rows` read as zero), and there is no scale or
// bias (one group of 32 codes with scale 1). An operand of E's own 16-bit
// type is exact; any other E is rounded to the operand type (as the GEMM
// rounds dequantized packed weights). The GEMM reads fragments from global
// memory rather than staging them (`DENSE`), and there is no INT8 path.
template <class E> struct Dense {
    const u8 *base;
    u64 stride;
    u64 rows;

    static constexpr int GROUP = 32;
    static constexpr int GROUPS = 2;
    static constexpr bool BIAS = false;
    static constexpr bool DENSE = true;
    static constexpr bool SCALED_GEMM = false;
    static constexpr int WORDS = E::bytes / 2; // 32-bit words per fragment register
    // Matrix `index` of a stacked tensor whose matrices are `matrix_stride`
    // elements apart.
    __device__ static __forceinline__ Dense matrix(const u8 *base, u64 index, u64 matrix_stride, u64 stride,
                                                   u64 rows) {
        return Dense{base + index * matrix_stride * E::bytes, stride, rows};
    }
    // Fragment register i of k16 step s: elements (2 per register) in words
    // w[s][i][0..WORDS).
    struct Raw {
        u32 w[4][4][WORDS];
    };
    __device__ __forceinline__ Raw fetch(u64 tile, u64 kblock, u32 lane) const {
        const u32 g = lane / 4, t = lane % 4;
        Raw raw;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const u64 row = tile * 16 + g + 8 * half;
            const bool valid = row < rows;
            const u8 *line = base + (valid ? row : 0) * stride * E::bytes;
#pragma unroll
            for (int s = 0; s < 4; ++s)
#pragma unroll
                for (int upper = 0; upper < 2; ++upper) {
                    // Register half + 2 * upper: k = 16 s + 2 t (+ 8 when upper).
                    const u8 *at = line + (kblock * 64 + 16 * s + 2 * t + 8 * upper) * E::bytes;
#pragma unroll
                    for (int word = 0; word < WORDS; ++word)
                        raw.w[s][half + 2 * upper][word] =
                            valid ? seismic_ld_nc_u32(at + 4 * word) : 0u;
                }
        }
        return raw;
    }
    // GEMV: a superblock's k-blocks, fetched when the superblock is staged so
    // that its loads are in flight while the previous one computes.
    struct Super {
        Raw raw[4];
    };
    __device__ __forceinline__ Super fetch_superblock(u64 tile, u64 superblock, u64 kblocks, u32 lane) const {
        Super super;
#pragma unroll
        for (int q = 0; q < 4; ++q)
            if (4 * superblock + q < kblocks)
                super.raw[q] = fetch(tile, 4 * superblock + q, lane);
        return super;
    }
    __device__ __forceinline__ Raw raw(const Super &super, int q, u32) const { return super.raw[q]; }
    // GEMV in the permuted column order (`gemv_column`): lane (g, t) holds
    // k-block columns 16 t .. 16 t + 15 of rows g and g + 8, register
    // (s, half + 2 upper) the pair at 16 t + 4 s + 2 upper, so each row's 16
    // columns load as whole 16-byte vectors. A GEMV sums over k, so the
    // activations read the same columns and the product is unchanged.
    static constexpr bool GEMV_PERMUTED = true;
    __device__ __forceinline__ Super fetch_superblock_permuted(u64 tile, u64 superblock, u64 kblocks,
                                                               u32 lane) const {
        constexpr int VECTORS = E::bytes; // 16-byte vectors of a lane's 16 columns
        const u32 g = lane / 4, t = lane % 4;
        Super super;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const u64 kblock = 4 * superblock + q;
            if (kblock >= kblocks)
                continue;
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const u64 row = tile * 16 + g + 8 * half;
                const bool valid = row < rows;
                const u8 *at = base + ((valid ? row : 0) * stride + kblock * 64 + 16 * t) * E::bytes;
                u32 words[4 * VECTORS];
#pragma unroll
                for (int v = 0; v < VECTORS; ++v) {
                    const uint4 x = valid ? seismic_ld_nc_v4(at + 16 * v) : make_uint4(0u, 0u, 0u, 0u);
                    words[4 * v] = x.x, words[4 * v + 1] = x.y, words[4 * v + 2] = x.z, words[4 * v + 3] = x.w;
                }
#pragma unroll
                for (int s = 0; s < 4; ++s)
#pragma unroll
                    for (int upper = 0; upper < 2; ++upper)
#pragma unroll
                        for (int word = 0; word < WORDS; ++word)
                            super.raw[q].w[s][half + 2 * upper][word] =
                                words[(4 * s + 2 * upper) * E::bytes / 4 + word];
            }
        }
        return super;
    }
    // The two elements of register i, as F32.
    __device__ static __forceinline__ float2 pair(const Raw &raw, int step, int i) {
        if constexpr (E::bytes == 4)
            return make_float2(__uint_as_float(raw.w[step][i][0]), __uint_as_float(raw.w[step][i][1]));
        else
            return E::unpack2(raw.w[step][i][0]);
    }
    template <class O> __device__ __forceinline__ void decode(const Raw &raw, int step, u32 (&a)[4]) const {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            if constexpr (NativeOperand<O, E>::value) {
                a[i] = raw.w[step][i][0];
            } else {
                const float2 values = pair(raw, step, i);
                a[i] = O::pack(values.x, values.y);
            }
        }
    }
    struct Block {};
    __device__ __forceinline__ Block block(u64, u64, u32) const { return Block{}; }
    __device__ __forceinline__ void coefficients(const Block &, int, Coefficients<GROUPS> &c) const {
#pragma unroll
        for (int r = 0; r < 2; ++r)
#pragma unroll
            for (int group = 0; group < GROUPS; ++group) {
                c.scale[r][group] = 1.0f;
                c.bias[r][group] = 0.0f;
            }
    }
    // GEMM: nothing is staged; fragments come from `fetch`.
    static constexpr int CHUNKS = 0;
    static constexpr int SUPER_WORDS = 0;
    __device__ __forceinline__ const u8 *chunk_source(u64, u64, u32) const { return base; }
    __device__ __forceinline__ const u8 *super_word(u64, u64, int) const { return base; }
    __device__ static __forceinline__ int word_kblock(int) { return 0; }
    __device__ __forceinline__ void staged_coefficient(const u32 *, const u8 *, u32, u64, int, float &scale,
                                                       float &bias) const {
        scale = 1.0f;
        bias = 0.0f;
    }
    // Rows: the element of fragment slot `slot` (register slot % 4, low half
    // for slot < 4) of step `step`, as bits; `apply` converts it.
    __device__ __forceinline__ u32 code(const Raw &raw, int step, u32 slot) const {
        if constexpr (E::bytes == 4)
            return raw.w[step][slot % 4][slot / 4];
        else
            return (raw.w[step][slot % 4][0] >> (16 * (slot / 4))) & 0xFFFFu;
    }
    __device__ static __forceinline__ float apply(u32 code, float, float) {
        if constexpr (E::bytes == 4)
            return __uint_as_float(code);
        else
            return E::load((typename E::storage)code);
    }
    __device__ __forceinline__ void coefficient(u64, u64, int, float &scale, float &bias) const {
        scale = 1.0f;
        bias = 0.0f;
    }
};

// The decoder of a packed element's emitted decode axes (`_CODE*`,
// `_COEFF*`): each axis value's component, chosen in one place.

template <u32 CODES, int OFFSET, u32 TABLE> struct CodesOf;
template <int OFFSET, u32 TABLE> struct CodesOf<SEISMIC_CODES_UNSIGNED, OFFSET, TABLE> { typedef Linear<0> type; };
template <int OFFSET, u32 TABLE> struct CodesOf<SEISMIC_CODES_OFFSET, OFFSET, TABLE> { typedef Linear<(u32)OFFSET> type; };
template <int OFFSET, u32 TABLE> struct CodesOf<SEISMIC_CODES_TWOS, OFFSET, TABLE> { typedef Twos type; };
template <int OFFSET, u32 TABLE> struct CodesOf<SEISMIC_CODES_TABLE, OFFSET, TABLE> { typedef Table<TABLE> type; };

template <u32 SCALE> struct ScaleOf;
template <> struct ScaleOf<SEISMIC_SCALE_F16> { typedef F16Scale type; };
template <> struct ScaleOf<SEISMIC_SCALE_BF16> { typedef BF16Scale type; };
template <> struct ScaleOf<SEISMIC_SCALE_F32> { typedef F32Scale type; };
template <> struct ScaleOf<SEISMIC_SCALE_E8M0> { typedef E8M0Scale type; };
template <> struct ScaleOf<SEISMIC_SCALE_UE4M3> { typedef UE4M3Scale type; };

template <u32 LOCAL, u32 BITS, int OFFSET> struct LocalOf;
template <u32 BITS, int OFFSET> struct LocalOf<SEISMIC_CODES_UNSIGNED, BITS, OFFSET> { typedef OffsetCode<0> type; };
template <u32 BITS, int OFFSET> struct LocalOf<SEISMIC_CODES_TWOS, BITS, OFFSET> { typedef TwosCode<BITS> type; };
template <u32 BITS, int OFFSET> struct LocalOf<SEISMIC_CODES_OFFSET, BITS, OFFSET> { typedef OffsetCode<OFFSET> type; };
template <u32 BITS, int OFFSET> struct LocalOf<SEISMIC_CODES_ODD, BITS, OFFSET> { typedef OddCode type; };

template <u32 COEFF, u32 SCALE, u32 BIAS, int BIAS_SIGN, u32 GROUP, u32 STORAGE, u32 FACTOR_GROUP, int EXPONENT,
          int CODE_EXPONENT, u32 LOCAL_BITS, u32 LOCAL, int LOCAL_OFFSET, u32 CARRIED>
struct CoefficientsOf;
template <u32 SCALE, u32 BIAS, int BIAS_SIGN, u32 GROUP, u32 STORAGE, u32 FACTOR_GROUP, int EXPONENT,
          int CODE_EXPONENT, u32 LOCAL_BITS, u32 LOCAL, int LOCAL_OFFSET, u32 CARRIED>
struct CoefficientsOf<SEISMIC_COEFF_DIRECT, SCALE, BIAS, BIAS_SIGN, GROUP, STORAGE, FACTOR_GROUP, EXPONENT,
                      CODE_EXPONENT, LOCAL_BITS, LOCAL, LOCAL_OFFSET, CARRIED> {
    typedef Direct<typename ScaleOf<SCALE>::type, (int)GROUP, BIAS != 0, (int)STORAGE, CODE_EXPONENT> type;
};
// Hierarchical coefficients' factors are per 256-code superblock; their
// local codes are in the scales plane (`Hier`) or, where CARRIED is not 0,
// at entry bit CARRIED of grid entries (`CarriedHier`).
template <u32 SCALE, u32 BIAS, int BIAS_SIGN, u32 GROUP, u32 STORAGE, u32 FACTOR_GROUP, int EXPONENT,
          int CODE_EXPONENT, u32 LOCAL_BITS, u32 LOCAL, int LOCAL_OFFSET, u32 CARRIED>
struct CoefficientsOf<SEISMIC_COEFF_HIER, SCALE, BIAS, BIAS_SIGN, GROUP, STORAGE, FACTOR_GROUP, EXPONENT,
                      CODE_EXPONENT, LOCAL_BITS, LOCAL, LOCAL_OFFSET, CARRIED> {
    static_assert(FACTOR_GROUP == 256, "hierarchical factors are per 256-code superblock");
    static_assert(CARRIED == 0 || BIAS == 0, "grid entries carry scale codes without a minimum");
    typedef typename ScaleOf<SCALE>::type FS;
    typedef typename LocalOf<LOCAL, LOCAL_BITS, LOCAL_OFFSET>::type L;
    typedef typename Select<CARRIED != 0, CarriedHier<FS, EXPONENT, CODE_EXPONENT, LOCAL_BITS, L, (int)GROUP, CARRIED>,
                            Hier<FS, EXPONENT, CODE_EXPONENT, LOCAL_BITS, L, (int)GROUP, BIAS != 0, BIAS_SIGN>>::type type;
};

// Whether a form multiplies through the scaled GEMM: direct coefficients of
// codes below 8 bits. Today's forms' choice, which tuning owns once both
// GEMMs are qualified for each.
template <class K> struct IsDirect {
    static constexpr bool value = false;
};
template <class S, int GROUP, bool BIAS, int STORAGE, int CODE_EXPONENT>
struct IsDirect<Direct<S, GROUP, BIAS, STORAGE, CODE_EXPONENT>> {
    static constexpr bool value = true;
};
template <class K, u32 LOW> struct ScaledGemm {
    static constexpr bool value = IsDirect<K>::value && LOW != 8;
};

// The decoder of code axes CODES .. LANES under coefficients K
// (`PACKETS_COEFFICIENTS`), RECORDS when the layout holds the codes in
// records (`_PLANE_RECORDS`).
template <u32 CODES, u32 LOW, u32 HIGH, int OFFSET, u32 TABLE, u32 GRID, u32 INDEX_BITS, u32 LANES, class K, bool RECORDS,
          bool OPERANDS>
struct Decoder {
    static_assert(!RECORDS, "only grid codes are held in records");
    typedef Packed<Planes<(int)LOW, (int)HIGH, K::STORAGE % 256 == 0>, typename CodesOf<CODES, OFFSET, TABLE>::type, K,
                   ScaledGemm<K, LOW>::value, OPERANDS>
        type;
};
// A grid code under its coefficients: with carried coefficient codes, the
// `mma16` records, else the grids plane (multiplied through the scaled GEMM).
template <u32 LOW, u32 HIGH, int OFFSET, u32 TABLE, u32 GRID, u32 INDEX_BITS, u32 LANES, class K, bool RECORDS,
          bool OPERANDS>
struct Decoder<SEISMIC_CODES_GRID, LOW, HIGH, OFFSET, TABLE, GRID, INDEX_BITS, LANES, K, RECORDS, OPERANDS> {
    static_assert(!OPERANDS, "grid codes have no NVFP4 operands");
    static_assert(RECORDS == IsCarried<K>::value, "mma16 holds carried coefficient codes in records, and only those");
    typedef typename GridCodesOf<LOW, HIGH, GRID, INDEX_BITS, LANES>::type Codes;
    typedef typename Select<RECORDS, Records<Codes, K>, Packed<GridPlanes<Codes>, Twos, K, true>>::type type;
};

// One row's values at k = 64*kb + 16*s + {2t, 2t+1, 2t+8, 2t+9} (s = 0..3)
// into v[4*s + j]: the part of the row held by the lane chunk of lane
// 4*(row % 8) + t, decoded with the row's coefficients.
template <class W> __device__ __forceinline__ void row_values16(const W &w, u64 row, u64 kb, u32 t, float (&v)[16]) {
    const u32 local = (u32)(row % 16);
    const u32 upper = local / 8;
    const typename W::Raw raw = w.fetch(row / 16, kb, 4 * (local % 8) + t);
    const u32 slots[4] = {upper, upper + 4, upper + 2, upper + 6};
    const u32 offsets[4] = {2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9};
#pragma unroll
    for (int s = 0; s < 4; ++s) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float scale;
            float bias;
            w.coefficient(row, kb, (int)((16 * s + offsets[j]) / W::GROUP), scale, bias);
            v[4 * s + j] = W::apply(w.code(raw, s, slots[j]), scale, bias);
        }
    }
}

} // namespace packets

// A packed element's coefficients component (`packets::Direct`,
// `packets::Hier`) for a tensor prefix `P`, in any row layout.
#define PACKETS_COEFFICIENTS(P)                                                                          \
    packets::CoefficientsOf<ELEMENT_CAT(P, _COEFF), ELEMENT_CAT(P, _COEFF_SCALE), ELEMENT_CAT(P, _COEFF_BIAS),  \
                            ELEMENT_CAT(P, _COEFF_BIAS_SIGN), ELEMENT_CAT(P, _COEFF_GROUP),                    \
                            ELEMENT_CAT(P, _LOGICAL_GROUP), ELEMENT_CAT(P, _COEFF_FACTOR_GROUP),               \
                            ELEMENT_CAT(P, _COEFF_EXPONENT), ELEMENT_CAT(P, _CODES_EXPONENT),                  \
                            ELEMENT_CAT(P, _COEFF_LOCAL_BITS), ELEMENT_CAT(P, _COEFF_LOCAL),                   \
                            ELEMENT_CAT(P, _COEFF_LOCAL_OFFSET), ELEMENT_CAT(P, _COEFF_CARRIED)>::type

// A packed element's decoder type, and its value at `pointer`.
#define PACKETS_DECODER(P)                                                                               \
    packets::Decoder<ELEMENT_CAT(P, _CODES), ELEMENT_CAT(P, _CODE_LOW_BITS), ELEMENT_CAT(P, _CODE_HIGH_BITS), \
                     ELEMENT_CAT(P, _CODES_OFFSET), ELEMENT_CAT(P, _CODES_TABLE), ELEMENT_CAT(P, _CODES_GRID),     \
                     ELEMENT_CAT(P, _CODES_GRID_INDEX_BITS), ELEMENT_CAT(P, _CODES_GRID_LANES),                  \
                     PACKETS_COEFFICIENTS(P), ELEMENT_HAS(P, _PLANE_RECORDS) != 0,                               \
                     ELEMENT_HAS(P, _OPERANDS_NVFP4) != 0>::type
#define PACKETS_MAKE(TYPE, P, pointer)                                                                   \
    TYPE {                                                                                               \
        packets::Rows {                                                                                  \
            (const packets::u8 *)(pointer), ELEMENT_CAT(P, _ROW_STRIDE_BYTES),                            \
                packets::CodePlane{ELEMENT_CAT(P, _ROW_CODES), ELEMENT_CAT(P, _ROW_CODES_BYTES)},          \
                packets::CodePlane{ELEMENT_CAT(P, _ROW_HIGH), ELEMENT_CAT(P, _ROW_HIGH_BYTES)},            \
                ELEMENT_CAT(P, _ROW_SCALES), ELEMENT_CAT(P, _ROW_SUPERS), ELEMENT_CAT(P, _ROW_SUPERS_BYTES) \
        }                                                                                                \
    }

// A records layout's value (its factors past a tile's entries).
#define PACKETS_MAKE_RECORDS(TYPE, P, pointer)                                                           \
    TYPE {                                                                                               \
        (const packets::u8 *)(pointer), ELEMENT_CAT(P, _ROW_STRIDE_BYTES), ELEMENT_CAT(P, _PLANE_RECORDS_FACTORS_OFFSET) \
    }
// Dense weights: the tensor's rows (extent 0) at its row stride (stride 0,
// elements). A rank-2 [rows, K] tensor is required.
#define PACKETS_MAKE_DENSE(P, pointer)                                                                   \
    packets::Dense<ELEMENT_OF(P)> {                                                                      \
        (const packets::u8 *)(pointer), (packets::u64)ELEMENT_CAT(P, _STRIDE_0),                         \
            (packets::u64)ELEMENT_CAT(P, _EXTENT_0)                                                      \
    }

// Matrix `index` of a stacked rank-3 [B, N, K] tensor (such as expert
// weights [E, N, K]). Dense: N rows at row stride stride 1 from element
// index * stride 0. Packed: every matrix's N rows are padded to whole row
// tiles, so matrix `index` starts at stored row index * N (tile-padded).
#define PACKETS_DENSE_MATRIX(P, pointer, index)                                                          \
    packets::Dense<ELEMENT_OF(P)>::matrix((const packets::u8 *)(pointer), (packets::u64)(index),         \
                                          (packets::u64)ELEMENT_CAT(P, _STRIDE_0),                       \
                                          (packets::u64)ELEMENT_CAT(P, _STRIDE_1),                       \
                                          (packets::u64)ELEMENT_CAT(P, _EXTENT_1))
#define PACKETS_PACKED_MATRIX(P, pointer, index)                                                         \
    ((const packets::u8 *)(pointer) +                                                                    \
     (packets::u64)(index) *                                                                             \
         (((packets::u64)ELEMENT_CAT(P, _EXTENT_1) + ELEMENT_CAT(P, _TILE_ROWS) - 1) /                   \
          ELEMENT_CAT(P, _TILE_ROWS) * ELEMENT_CAT(P, _TILE_ROWS)) *                                     \
         (packets::u64)ELEMENT_CAT(P, _ROW_STRIDE_BYTES))

// Slot bindings. Each bound slot must name a dense tensor or a tensor in the
// mma16 layout. `KERNEL_Wn_MATRIX(pointer, index)` is the decoder value of
// matrix `index` of a stacked [B, N, K] tensor bound to slot n.
#if defined(KERNEL_W0)
#if ELEMENT_HAS(KERNEL_W0, _KIND_DENSE)
namespace packets { typedef Dense<ELEMENT_OF(KERNEL_W0)> W0; }
#define KERNEL_W0_AT(pointer) PACKETS_MAKE_DENSE(KERNEL_W0, pointer)
#elif !ELEMENT_HAS(KERNEL_W0, _LAYOUT_MMA16)
#error "KERNEL_W0 must be dense or bound in the mma16 layout"
#else
namespace packets { typedef PACKETS_DECODER(KERNEL_W0) W0; }
#if ELEMENT_HAS(KERNEL_W0, _PLANE_RECORDS)
#define KERNEL_W0_AT(pointer) PACKETS_MAKE_RECORDS(packets::W0, KERNEL_W0, pointer)
#else
#define KERNEL_W0_AT(pointer) PACKETS_MAKE(packets::W0, KERNEL_W0, pointer)
#endif
#endif
#if ELEMENT_HAS(KERNEL_W0, _KIND_DENSE)
#define KERNEL_W0_MATRIX(pointer, index) PACKETS_DENSE_MATRIX(KERNEL_W0, pointer, index)
#else
#define KERNEL_W0_MATRIX(pointer, index) KERNEL_W0_AT(PACKETS_PACKED_MATRIX(KERNEL_W0, pointer, index))
#endif
// The slot's operand recipe (`operands(W)`): whether products over it take
// NVFP4 operand rows.
namespace packets { constexpr bool W0_OPERANDS_NVFP4 = ELEMENT_HAS(KERNEL_W0, _OPERANDS_NVFP4); }
#endif

#if defined(KERNEL_W1)
#if ELEMENT_HAS(KERNEL_W1, _KIND_DENSE)
namespace packets { typedef Dense<ELEMENT_OF(KERNEL_W1)> W1; }
#define KERNEL_W1_AT(pointer) PACKETS_MAKE_DENSE(KERNEL_W1, pointer)
#elif !ELEMENT_HAS(KERNEL_W1, _LAYOUT_MMA16)
#error "KERNEL_W1 must be dense or bound in the mma16 layout"
#else
namespace packets { typedef PACKETS_DECODER(KERNEL_W1) W1; }
#if ELEMENT_HAS(KERNEL_W1, _PLANE_RECORDS)
#define KERNEL_W1_AT(pointer) PACKETS_MAKE_RECORDS(packets::W1, KERNEL_W1, pointer)
#else
#define KERNEL_W1_AT(pointer) PACKETS_MAKE(packets::W1, KERNEL_W1, pointer)
#endif
#endif
#if ELEMENT_HAS(KERNEL_W1, _KIND_DENSE)
#define KERNEL_W1_MATRIX(pointer, index) PACKETS_DENSE_MATRIX(KERNEL_W1, pointer, index)
#else
#define KERNEL_W1_MATRIX(pointer, index) KERNEL_W1_AT(PACKETS_PACKED_MATRIX(KERNEL_W1, pointer, index))
#endif
namespace packets { constexpr bool W1_OPERANDS_NVFP4 = ELEMENT_HAS(KERNEL_W1, _OPERANDS_NVFP4); }
#endif

#if defined(KERNEL_W2)
#if ELEMENT_HAS(KERNEL_W2, _KIND_DENSE)
namespace packets { typedef Dense<ELEMENT_OF(KERNEL_W2)> W2; }
#define KERNEL_W2_AT(pointer) PACKETS_MAKE_DENSE(KERNEL_W2, pointer)
#elif !ELEMENT_HAS(KERNEL_W2, _LAYOUT_MMA16)
#error "KERNEL_W2 must be dense or bound in the mma16 layout"
#else
namespace packets { typedef PACKETS_DECODER(KERNEL_W2) W2; }
#if ELEMENT_HAS(KERNEL_W2, _PLANE_RECORDS)
#define KERNEL_W2_AT(pointer) PACKETS_MAKE_RECORDS(packets::W2, KERNEL_W2, pointer)
#else
#define KERNEL_W2_AT(pointer) PACKETS_MAKE(packets::W2, KERNEL_W2, pointer)
#endif
#endif
#if ELEMENT_HAS(KERNEL_W2, _KIND_DENSE)
#define KERNEL_W2_MATRIX(pointer, index) PACKETS_DENSE_MATRIX(KERNEL_W2, pointer, index)
#else
#define KERNEL_W2_MATRIX(pointer, index) KERNEL_W2_AT(PACKETS_PACKED_MATRIX(KERNEL_W2, pointer, index))
#endif
namespace packets { constexpr bool W2_OPERANDS_NVFP4 = ELEMENT_HAS(KERNEL_W2, _OPERANDS_NVFP4); }
#endif

#if defined(KERNEL_W3)
#if ELEMENT_HAS(KERNEL_W3, _KIND_DENSE)
namespace packets { typedef Dense<ELEMENT_OF(KERNEL_W3)> W3; }
#define KERNEL_W3_AT(pointer) PACKETS_MAKE_DENSE(KERNEL_W3, pointer)
#elif !ELEMENT_HAS(KERNEL_W3, _LAYOUT_MMA16)
#error "KERNEL_W3 must be dense or bound in the mma16 layout"
#else
namespace packets { typedef PACKETS_DECODER(KERNEL_W3) W3; }
#if ELEMENT_HAS(KERNEL_W3, _PLANE_RECORDS)
#define KERNEL_W3_AT(pointer) PACKETS_MAKE_RECORDS(packets::W3, KERNEL_W3, pointer)
#else
#define KERNEL_W3_AT(pointer) PACKETS_MAKE(packets::W3, KERNEL_W3, pointer)
#endif
#endif
#if ELEMENT_HAS(KERNEL_W3, _KIND_DENSE)
#define KERNEL_W3_MATRIX(pointer, index) PACKETS_DENSE_MATRIX(KERNEL_W3, pointer, index)
#else
#define KERNEL_W3_MATRIX(pointer, index) KERNEL_W3_AT(PACKETS_PACKED_MATRIX(KERNEL_W3, pointer, index))
#endif
namespace packets { constexpr bool W3_OPERANDS_NVFP4 = ELEMENT_HAS(KERNEL_W3, _OPERANDS_NVFP4); }
#endif
