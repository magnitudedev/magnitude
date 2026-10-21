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

#include <seismic/element.cuh>
#include <seismic/grids.cuh>

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
// Six-bit unsigned codes per byte minus 32, sign-extended to s8.
__device__ __forceinline__ u32 s8_offset32(u32 bytes) {
    const u32 flipped = bytes ^ 0x20202020u;
    const u32 sign = flipped & 0x20202020u;
    return flipped | (sign << 1) | (sign << 2);
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

// Coefficients of one k-block for the fragment rows g and g+8, per group.
template <int GROUPS> struct Coefficients {
    float scale[2][GROUPS];
    float bias[2][GROUPS];
};

__device__ __forceinline__ u32 bits6(const u8 *bytes, u32 field) {
    const u32 bit = field * 6;
    const u32 pair = (u32)bytes[bit / 8] | ((u32)bytes[bit / 8 + 1] << 8);
    return (pair >> (bit % 8)) & 63u;
}

__device__ __forceinline__ float f16_at(const u8 *address) {
    return seismic_f16_to_f32(*reinterpret_cast<const u16 *>(address));
}

// q4k / q5k: 4 (+1 high) bit codes, (scale6, min6) per 32, (d, dmin) per 256.
template <int HIGH_BITS> struct KQuant45 {
    const u8 *base;
    u64 stride;
    CodePlane low;
    CodePlane high;
    u64 scales;
    u64 supers;

    static constexpr int GROUP = 32;
    static constexpr int GROUPS = 2;
    static constexpr bool BIAS = true;
    static constexpr bool DENSE = false;
    static constexpr bool SCALED_GEMM = false;
    struct Raw {
        uint4 low;
        u32 high;
    };

    __device__ __forceinline__ Raw fetch(u64 tile, u64 kblock, u32 lane) const {
        Raw raw;
        raw.low = seismic_ld_nc_na_v4(low.chunk(base, stride, tile, kblock * 512 + lane * 16));
        if constexpr (HIGH_BITS == 1)
            raw.high = seismic_ld_nc_u32(high.chunk(base, stride, tile, kblock * 128 + lane * 4));
        else
            raw.high = 0;
        return raw;
    }
    // The lane chunks of one superblock (4 k-blocks). The warp's high bits of
    // the superblock are 512 contiguous bytes, fetched as one 16 B load per
    // lane and redistributed by `raw` (warp-collective).
    struct Super {
        uint4 low[4];
        uint4 high;
    };
    // Rows hold whole 256-code superblocks, so all four k-blocks exist.
    __device__ __forceinline__ Super fetch_superblock(u64 tile, u64 superblock, u64, u32 lane) const {
        Super super;
#pragma unroll
        for (int q = 0; q < 4; ++q)
            super.low[q] = seismic_ld_nc_na_v4(low.chunk(base, stride, tile, (4 * superblock + q) * 512 + lane * 16));
        if constexpr (HIGH_BITS == 1)
            super.high = seismic_ld_nc_na_v4(high.chunk(base, stride, tile, superblock * 512 + lane * 16));
        return super;
    }
    __device__ __forceinline__ Raw raw(const Super &super, int q, u32 lane) const {
        Raw raw;
        raw.low = super.low[q];
        if constexpr (HIGH_BITS == 1) {
            // Word 32q + lane of the superblock sits in lane 8q + lane/4, component lane % 4.
            const u32 source = 8 * (u32)q + lane / 4;
            const u32 x = seismic_shfl_idx_u32(super.high.x, source);
            const u32 y = seismic_shfl_idx_u32(super.high.y, source);
            const u32 z = seismic_shfl_idx_u32(super.high.z, source);
            const u32 w = seismic_shfl_idx_u32(super.high.w, source);
            const u32 component = lane % 4;
            raw.high = component == 0 ? x : component == 1 ? y : component == 2 ? z : w;
        } else {
            raw.high = 0;
        }
        return raw;
    }
    // A registers of k16 step `step`.
    template <class O> __device__ __forceinline__ void decode(const Raw &raw, int step, u32 (&a)[4]) const {
        const u32 word = word_of(raw.low, step);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            u32 pair = (word >> (4 * i)) & 0x000F000Fu;
            if constexpr (HIGH_BITS == 1) {
                const u32 bits = raw.high >> (8 * step + i);
                pair |= ((bits << 4) & 0x10u) | ((bits << 16) & 0x100000u);
            }
            a[i] = O::template codes<0>(pair);
        }
    }
    // s8 bytes of rows g and g+8 of step `step` (virtual order).
    __device__ __forceinline__ S8Pair s8(const Raw &raw, int step) const {
        S8Pair pair = s8_nibbles(word_of(raw.low, step));
        if constexpr (HIGH_BITS == 1) {
            const u32 bits = raw.high >> (8 * step);
            pair.row_g |= ((bits & 1u) << 4) | (((bits >> 4) & 1u) << 12) | (((bits >> 2) & 1u) << 20) |
                          (((bits >> 6) & 1u) << 28);
            pair.row_g8 |= (((bits >> 1) & 1u) << 4) | (((bits >> 5) & 1u) << 12) | (((bits >> 3) & 1u) << 20) |
                           (((bits >> 7) & 1u) << 28);
        }
        return pair;
    }
    __device__ __forceinline__ void coefficient(u64 row, u64 kblock, int group, float &scale, float &bias) const {
        const u8 *line = base + row * stride;
        const u8 *packed = line + scales + (kblock / 4) * 12;
        const u32 field = 2 * ((u32)(kblock % 4) * 2 + (u32)group);
        scale = f16_at(line + supers + (kblock / 4) * 4) * (float)bits6(packed, field);
        bias = f16_at(line + supers + (kblock / 4) * 4 + 2) * (float)bits6(packed, field + 1);
    }
    // Packed coefficients of one superblock (4 k-blocks) of rows g and g+8.
    struct Block {
        u32 packed[2][3];
        u32 factors[2];
    };
    __device__ __forceinline__ Block block(u64 tile, u64 superblock, u32 lane) const {
        Block block;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const u8 *line = base + (tile * 16 + lane / 4 + 8 * r) * stride;
            const u8 *packed = line + scales + superblock * 12;
#pragma unroll
            for (int w = 0; w < 3; ++w)
                block.packed[r][w] = seismic_ld_nc_u32(packed + 4 * w);
            block.factors[r] = seismic_ld_nc_u32(line + supers + superblock * 4);
        }
        return block;
    }
    __device__ static __forceinline__ u32 field(const u32 *packed, u32 index) {
        const u32 bit = 6 * index;
        const u32 word = bit / 32;
        const u64 wide = (u64)packed[word] | (word < 2 ? (u64)packed[word + 1] << 32 : 0ull);
        return (u32)(wide >> (bit % 32)) & 63u;
    }
    // GEMM staging: the superblock's 12 packed scale bytes and (d, dmin) word.
    static constexpr int SUPER_WORDS = 4;
    __device__ __forceinline__ const u8 *super_word(u64 row, u64 superblock, int word) const {
        const u8 *line = base + row * stride;
        return word < 3 ? line + scales + superblock * 12 + 4 * word : line + supers + superblock * 4;
    }
    __device__ static __forceinline__ int word_kblock(int) { return 0; }
    __device__ __forceinline__ void staged_coefficient(const u32 *words, const u8 *, u32, u64 kblock, int group,
                                                       float &scale, float &bias) const {
        // The record's packed fields are read in place (in shared memory): a
        // register copy would be indexed dynamically, through local memory.
        const u32 index = 2 * ((u32)(kblock % 4) * 2 + (u32)group);
        scale = seismic_f16_to_f32((u16)(words[3] & 0xFFFFu)) * (float)field(words, index);
        bias = seismic_f16_to_f32((u16)(words[3] >> 16)) * (float)field(words, index + 1);
    }
    // Coefficients of k-block `q` of the superblock.
    __device__ __forceinline__ void coefficients(const Block &block, int q, Coefficients<2> &c) const {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const float d = seismic_f16_to_f32((u16)(block.factors[r] & 0xFFFFu));
            const float dmin = seismic_f16_to_f32((u16)(block.factors[r] >> 16));
#pragma unroll
            for (int group = 0; group < 2; ++group) {
                const u32 index = 2 * (2 * (u32)q + (u32)group);
                c.scale[r][group] = d * (float)field(block.packed[r], index);
                c.bias[r][group] = dmin * (float)field(block.packed[r], index + 1);
            }
        }
    }
    // Code of fragment slot `slot` in step `step` of a lane chunk.
    __device__ __forceinline__ u32 code(const Raw &raw, int step, u32 slot) const {
        u32 value = (word_of(raw.low, step) >> (4 * slot)) & 15u;
        if constexpr (HIGH_BITS == 1)
            value |= ((raw.high >> (8 * step + slot)) & 1u) << 4;
        return value;
    }
    __device__ static __forceinline__ float apply(u32 code, float scale, float bias) {
        return seismic_fma_rn(scale, (float)code, -bias);
    }
    // GEMM staging: the 16 B chunks of one tile's k-block, low plane first.
    static constexpr int CHUNKS = 32 + (HIGH_BITS == 1 ? 8 : 0);
    __device__ __forceinline__ const u8 *chunk_source(u64 tile, u64 kblock, u32 chunk) const {
        if (chunk < 32)
            return low.chunk(base, stride, tile, kblock * 512 + chunk * 16);
        return high.chunk(base, stride, tile, kblock * 128 + (chunk - 32) * 16);
    }
    __device__ __forceinline__ Raw from_shared(const u8 *staged, u32 lane) const {
        Raw raw;
        raw.low = *reinterpret_cast<const uint4 *>(staged + lane * 16);
        if constexpr (HIGH_BITS == 1)
            raw.high = *reinterpret_cast<const u32 *>(staged + 512 + lane * 4);
        else
            raw.high = 0;
        return raw;
    }
};

// q6k: 4 + 2 bit codes offset by 32, int8 scale per 16, f16 d per 256.
struct KQuant6 {
    const u8 *base;
    u64 stride;
    CodePlane low;
    CodePlane high;
    u64 scales;
    u64 supers;

    static constexpr int GROUP = 16;
    static constexpr int GROUPS = 4;
    static constexpr bool BIAS = false;
    static constexpr bool DENSE = false;
    static constexpr bool SCALED_GEMM = false;
    struct Raw {
        uint4 low;
        uint2 high;
    };

    __device__ __forceinline__ Raw fetch(u64 tile, u64 kblock, u32 lane) const {
        Raw raw;
        raw.low = seismic_ld_nc_na_v4(low.chunk(base, stride, tile, kblock * 512 + lane * 16));
        raw.high = seismic_ld_nc_v2(high.chunk(base, stride, tile, kblock * 256 + lane * 8));
        return raw;
    }
    // The lane chunks of one superblock. The warp's high bits of the
    // superblock are 1024 contiguous bytes, fetched as two 16 B loads per lane
    // and redistributed by `raw` (warp-collective).
    struct Super {
        uint4 low[4];
        uint4 high[2];
    };
    // Rows hold whole 256-code superblocks, so all four k-blocks exist.
    __device__ __forceinline__ Super fetch_superblock(u64 tile, u64 superblock, u64, u32 lane) const {
        Super super;
#pragma unroll
        for (int q = 0; q < 4; ++q)
            super.low[q] = seismic_ld_nc_na_v4(low.chunk(base, stride, tile, (4 * superblock + q) * 512 + lane * 16));
        const u8 *high_chunk = high.chunk(base, stride, tile, superblock * 1024 + lane * 32);
        super.high[0] = seismic_ld_nc_na_v4(high_chunk);
        super.high[1] = seismic_ld_nc_na_v4(high_chunk + 16);
        return super;
    }
    __device__ __forceinline__ Raw raw(const Super &super, int q, u32 lane) const {
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
    template <class O> __device__ __forceinline__ void decode(const Raw &raw, int step, u32 (&a)[4]) const {
        const u32 word = word_of(raw.low, step);
        const u32 high = high_bits(raw, step);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            u32 pair = (word >> (4 * i)) & 0x000F000Fu;
            pair |= (((high >> (2 * i)) & 3u) << 4) | (((high >> (2 * i + 8)) & 3u) << 20);
            a[i] = O::template codes<32>(pair);
        }
    }
    __device__ __forceinline__ S8Pair s8(const Raw &raw, int step) const {
        S8Pair pair = s8_nibbles(word_of(raw.low, step));
        const u32 high = high_bits(raw, step);
        pair.row_g |= ((high & 3u) << 4) | (((high >> 8) & 3u) << 12) | (((high >> 4) & 3u) << 20) |
                      (((high >> 12) & 3u) << 28);
        pair.row_g8 |= (((high >> 2) & 3u) << 4) | (((high >> 10) & 3u) << 12) | (((high >> 6) & 3u) << 20) |
                       (((high >> 14) & 3u) << 28);
        return S8Pair{s8_offset32(pair.row_g), s8_offset32(pair.row_g8)};
    }
    __device__ __forceinline__ void coefficient(u64 row, u64 kblock, int group, float &scale, float &bias) const {
        const u8 *line = base + row * stride;
        scale = f16_at(line + supers + (kblock / 4) * 2) * (float)(signed char)line[scales + kblock * 4 + group];
        bias = 0.0f;
    }
    // The 16 int8 scales and the super factor of one superblock of rows g, g+8.
    struct Block {
        uint4 packed[2];
        float d[2];
    };
    __device__ __forceinline__ Block block(u64 tile, u64 superblock, u32 lane) const {
        Block block;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const u8 *line = base + (tile * 16 + lane / 4 + 8 * r) * stride;
            block.packed[r] = seismic_ld_nc_v4(line + scales + superblock * 16);
            block.d[r] = f16_at(line + supers + superblock * 2);
        }
        return block;
    }
    // GEMM staging: the superblock's sixteen int8 scales (four per k-block)
    // and the aligned word holding its f16 d (the supers plane is padded to
    // 16 B).
    static constexpr int SUPER_WORDS = 5;
    __device__ __forceinline__ const u8 *super_word(u64 row, u64 superblock, int word) const {
        const u8 *line = base + row * stride;
        return word < 4 ? line + scales + superblock * 16 + 4 * word : line + supers + (superblock * 2 & ~3ull);
    }
    __device__ static __forceinline__ int word_kblock(int word) { return word < 4 ? word : 0; }
    __device__ __forceinline__ void staged_coefficient(const u32 *words, const u8 *, u32, u64 kblock, int group,
                                                       float &scale, float &bias) const {
        const u16 d = (u16)(words[4] >> (16 * ((kblock / 4) % 2)));
        scale = seismic_f16_to_f32(d) * (float)(((int)(words[kblock % 4] << (24 - 8 * group))) >> 24);
        bias = 0.0f;
    }
    __device__ __forceinline__ void coefficients(const Block &block, int q, Coefficients<4> &c) const {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const u32 packed = word_of(block.packed[r], q);
#pragma unroll
            for (int group = 0; group < 4; ++group)
                c.scale[r][group] = block.d[r] * (float)(((int)(packed << (24 - 8 * group))) >> 24);
        }
    }
    __device__ __forceinline__ u32 code(const Raw &raw, int step, u32 slot) const {
        return ((word_of(raw.low, step) >> (4 * slot)) & 15u) | (((high_bits(raw, step) >> (2 * slot)) & 3u) << 4);
    }
    __device__ static __forceinline__ float apply(u32 code, float scale, float) {
        return scale * (float)((int)code - 32);
    }
    static constexpr int CHUNKS = 32 + 16;
    __device__ __forceinline__ const u8 *chunk_source(u64 tile, u64 kblock, u32 chunk) const {
        if (chunk < 32)
            return low.chunk(base, stride, tile, kblock * 512 + chunk * 16);
        return high.chunk(base, stride, tile, kblock * 256 + (chunk - 32) * 16);
    }
    __device__ __forceinline__ Raw from_shared(const u8 *staged, u32 lane) const {
        Raw raw;
        raw.low = *reinterpret_cast<const uint4 *>(staged + lane * 16);
        raw.high = *reinterpret_cast<const uint2 *>(staged + 512 + lane * 8);
        return raw;
    }
};

// q8g32s (GGUF q8_0): int8 codes, f16 scale per 32.
struct Q8 {
    const u8 *base;
    u64 stride;
    CodePlane codes_plane;
    u64 supers;

    static constexpr int GROUP = 32;
    static constexpr int GROUPS = 2;
    static constexpr bool BIAS = false;
    static constexpr bool DENSE = false;
    static constexpr bool SCALED_GEMM = false;
    struct Raw {
        uint4 first;
        uint4 second;
    };

    __device__ __forceinline__ Raw fetch(u64 tile, u64 kblock, u32 lane) const {
        const u8 *chunk = codes_plane.chunk(base, stride, tile, kblock * 1024 + lane * 32);
        Raw raw;
        raw.first = seismic_ld_nc_na_v4(chunk);
        raw.second = seismic_ld_nc_na_v4(chunk + 16);
        return raw;
    }
    // The lane chunks of one superblock (4 k-blocks, 8 groups).
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
    // Slots 0..3 (low) and 4..7 (high) of step `step`.
    __device__ static __forceinline__ void slots(const Raw &raw, int step, u32 &low, u32 &high) {
        const uint4 &half = step < 2 ? raw.first : raw.second;
        low = step % 2 == 0 ? half.x : half.z;
        high = step % 2 == 0 ? half.y : half.w;
    }
    template <class O> __device__ __forceinline__ void decode(const Raw &raw, int step, u32 (&a)[4]) const {
        u32 low, high;
        slots(raw, step, low, high);
#pragma unroll
        for (int i = 0; i < 4; ++i)
            a[i] = O::bytes(low, high, i);
    }
    __device__ __forceinline__ S8Pair s8(const Raw &raw, int step) const {
        u32 low, high;
        slots(raw, step, low, high);
        return S8Pair{prmt(low, high, 0x6240u), prmt(low, high, 0x7351u)};
    }
    __device__ __forceinline__ void coefficient(u64 row, u64 kblock, int group, float &scale, float &bias) const {
        scale = f16_at(base + row * stride + supers + (kblock * 2 + group) * 2);
        bias = 0.0f;
    }
    // The 8 f16 group scales of one superblock (256 codes) of rows g, g+8.
    // The supers plane ends 16 B-aligned inside the row, so the last
    // (possibly partial) superblock is readable whole.
    struct Block {
        uint4 packed[2];
    };
    __device__ __forceinline__ Block block(u64 tile, u64 superblock, u32 lane) const {
        Block block;
#pragma unroll
        for (int r = 0; r < 2; ++r)
            block.packed[r] =
                seismic_ld_nc_v4(base + (tile * 16 + lane / 4 + 8 * r) * stride + supers + superblock * 16);
        return block;
    }
    // GEMM staging: the superblock's eight f16 group scales (a word per
    // k-block).
    static constexpr int SUPER_WORDS = 4;
    __device__ __forceinline__ const u8 *super_word(u64 row, u64 superblock, int word) const {
        return base + row * stride + supers + superblock * 16 + 4 * word;
    }
    __device__ static __forceinline__ int word_kblock(int word) { return word; }
    __device__ __forceinline__ void staged_coefficient(const u32 *words, const u8 *, u32, u64 kblock, int group,
                                                       float &scale, float &bias) const {
        scale = seismic_f16_to_f32((u16)(words[kblock % 4] >> (16 * group)));
        bias = 0.0f;
    }
    __device__ __forceinline__ void coefficients(const Block &block, int q, Coefficients<2> &c) const {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const u32 packed = word_of(block.packed[r], q);
            c.scale[r][0] = seismic_f16_to_f32((u16)(packed & 0xFFFFu));
            c.scale[r][1] = seismic_f16_to_f32((u16)(packed >> 16));
        }
    }
    __device__ __forceinline__ u32 code(const Raw &raw, int step, u32 slot) const {
        u32 low, high;
        slots(raw, step, low, high);
        return ((slot < 4 ? low : high) >> (8 * (slot % 4))) & 0xFFu;
    }
    __device__ static __forceinline__ float apply(u32 code, float scale, float) {
        return (float)(int)(signed char)code * scale;
    }
    static constexpr int CHUNKS = 64;
    __device__ __forceinline__ const u8 *chunk_source(u64 tile, u64 kblock, u32 chunk) const {
        return codes_plane.chunk(base, stride, tile, kblock * 1024 + chunk * 16);
    }
    __device__ __forceinline__ Raw from_shared(const u8 *staged, u32 lane) const {
        Raw raw;
        raw.first = *reinterpret_cast<const uint4 *>(staged + lane * 32);
        raw.second = *reinterpret_cast<const uint4 *>(staged + lane * 32 + 16);
        return raw;
    }
};

// ---------------------------------------------------------------------------
// 4-bit coded weights (the registry's c4 family: iq4g32, q4g32s, mxfp4g32,
// nvfp4g16): a nibble code names one of sixteen exact integer values of a
// codebook (all within s8), and every group of 32 or 16 codes has one scale:
// value = scale * table[code]. The code plane is q4k's low-nibble plane. A
// table codebook's step word becomes s8 table values by `values`, after
// which every path is q8's: exact 16-bit operands (`O::bytes`), s8
// fragments, and value = scale * value (one F32 rounding); a linear codebook
// decodes its nibbles directly, as q4k does. E2M1 tables hold the values
// doubled, and their scales carry the exact factor 1/2.

// A codebook as little-endian s8 bytes: word w holds the values of codes
// 4w..4w+3.
//
// A codebook decodes one mma16 step word (the eight 4-bit slots of a lane):
// `decode<O>` into the four A registers, `s8` into the s8 fragments of rows g
// and g+8, and `code` one slot as an s8 byte.
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
        const u32 low = values(word & 0xFFFFu), high = values(word >> 16);
#pragma unroll
        for (int i = 0; i < 4; ++i)
            a[i] = O::bytes(low, high, i);
    }
    __device__ static __forceinline__ S8Pair s8(u32 word) {
        const u32 low = values(word & 0xFFFFu), high = values(word >> 16);
        return S8Pair{prmt(low, high, 0x6240u), prmt(low, high, 0x7351u)};
    }
    __device__ static __forceinline__ u32 code(u32 word, u32 slot) {
        return (values((word >> (16 * (slot / 4))) & 0xFFFFu) >> (8 * (slot % 4))) & 0xFFu;
    }
};
// GGUF kvalues_iq4nl (iq4g32).
typedef Codebook<0xBFAD9881u,  // -127 -104 -83 -65
                 0xF6EADDCFu,  //  -49  -35 -22 -10
                 0x26190D01u,  //    1   13  25  38
                 0x71594535u>  //   53   69  89 113
    Iq4Codes;
// Codes offset by OFFSET (q4g32s: 8): decoded directly, as q4k's nibbles are,
// with the offset subtracted exactly (`O::codes<OFFSET>`, a bytewise s8
// subtraction) instead of a table lookup.
template <u32 OFFSET> struct LinearCodebook {
    template <class O> __device__ static __forceinline__ void decode(u32 word, u32 (&a)[4]) {
#pragma unroll
        for (int i = 0; i < 4; ++i)
            a[i] = O::template codes<OFFSET>((word >> (4 * i)) & 0x000F000Fu);
    }
    __device__ static __forceinline__ S8Pair s8(u32 word) {
        const S8Pair pair = s8_nibbles(word);
        return S8Pair{__vsub4(pair.row_g, OFFSET * 0x01010101u), __vsub4(pair.row_g8, OFFSET * 0x01010101u)};
    }
    __device__ static __forceinline__ u32 code(u32 word, u32 slot) {
        return (((word >> (4 * slot)) & 15u) - OFFSET) & 0xFFu;
    }
};
typedef LinearCodebook<8> LinearCodes;
// E2M1 doubled, GGUF kvalues_mxfp4 (mxfp4g32, nvfp4g16). `decode` forms
// the operands without the table: a code's magnitude bits (e1 e0 m) placed
// at the operand's two low exponent bits and top mantissa bit read
// 2^(e - bias) (1 + m/2) for e > 0 and m 2^-bias for e = 0, in both cases
// the doubled value times 2^-bias, so one exact scaling by 2^bias forms it;
// the code's sign bit is the operand's (-0 becomes +0). The operands are
// the table's values exactly, without its conversions.
struct E2M1Codes : Codebook<0x03020100u,  //  0  1  2   3
                            0x0C080604u,  //  4  6  8  12
                            0xFDFEFF00u,  //  0 -1 -2  -3
                            0xF4F8FAFCu>  // -4 -6 -8 -12
{
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

// Scale fields in the supers plane. `GROUP` codes share one scale; a
// superblock (256 codes) of one row holds `BLOCK_BYTES` bytes of fields. The
// supers plane ends 16 B-aligned inside the row, so a row's last (possibly
// partial) superblock is readable whole.
struct F32Scale {
    static constexpr int GROUP = 32;
    static constexpr int BLOCK_BYTES = 32;
    static constexpr bool GEMV_STAGED = false;
    struct Block {
        uint4 packed[2][2];
    };
    __device__ static __forceinline__ void load(const u8 *fields, Block &block, int r) {
        block.packed[r][0] = seismic_ld_nc_v4(fields);
        block.packed[r][1] = seismic_ld_nc_v4(fields + 16);
    }
    // Scale `group` (of 8 per superblock) of row half `r`.
    __device__ static __forceinline__ float at(const Block &block, int r, int group) {
        const uint4 &half = block.packed[r][group / 4];
        return __uint_as_float(word_of(half, group % 4));
    }
    __device__ static __forceinline__ float read(const u8 *supers, u64 group) {
        return *reinterpret_cast<const float *>(supers + group * 4);
    }
    // GEMM staging: the superblock's eight scales (two words per k-block).
    static constexpr int SUPER_WORDS = 8;
    __device__ static __forceinline__ int word_kblock(int word) { return word / 2; }
    __device__ static __forceinline__ float staged(const u32 *words, u64 kblock, int group) {
        return __uint_as_float(words[(kblock % 4) * 2 + group]);
    }
};
struct F16Scale {
    static constexpr int GROUP = 32;
    static constexpr int BLOCK_BYTES = 16;
    static constexpr bool GEMV_STAGED = false;
    struct Block {
        uint4 packed[2];
    };
    __device__ static __forceinline__ void load(const u8 *fields, Block &block, int r) {
        block.packed[r] = seismic_ld_nc_v4(fields);
    }
    __device__ static __forceinline__ float at(const Block &block, int r, int group) {
        return seismic_f16_to_f32((u16)(word_of(block.packed[r], group / 2) >> (16 * (group % 2))));
    }
    __device__ static __forceinline__ float read(const u8 *supers, u64 group) { return f16_at(supers + group * 2); }
    static constexpr int SUPER_WORDS = 4;
    __device__ static __forceinline__ int word_kblock(int word) { return word; }
    __device__ static __forceinline__ float staged(const u32 *words, u64 kblock, int group) {
        return seismic_f16_to_f32((u16)(words[kblock % 4] >> (16 * group)));
    }
};
// E8M0 2^(e - 127), halved for the doubled codebook: 2^(e - 128) (e < 2 are
// F32 subnormals); 0xff is NaN.
struct E8M0Scale {
    static constexpr int GROUP = 32;
    static constexpr int BLOCK_BYTES = 8;
    static constexpr bool GEMV_STAGED = false;
    struct Block {
        uint2 packed[2];
    };
    __device__ static __forceinline__ float decode(u32 e) {
        return __uint_as_float(e < 2u ? 0x00200000u << e : e == 255u ? 0x7FC00000u : (e - 1u) << 23);
    }
    __device__ static __forceinline__ void load(const u8 *fields, Block &block, int r) {
        block.packed[r] = seismic_ld_nc_v2(fields);
    }
    __device__ static __forceinline__ float at(const Block &block, int r, int group) {
        const u32 word = group < 4 ? block.packed[r].x : block.packed[r].y;
        return decode((word >> (8 * (group % 4))) & 0xFFu);
    }
    __device__ static __forceinline__ float read(const u8 *supers, u64 group) { return decode(supers[group]); }
    // GEMM staging: the superblock's eight fields (a word per two k-blocks).
    static constexpr int SUPER_WORDS = 2;
    __device__ static __forceinline__ int word_kblock(int word) { return 2 * word; }
    __device__ static __forceinline__ float staged(const u32 *words, u64 kblock, int group) {
        return decode((words[(kblock % 4) / 2] >> (16 * (kblock % 2) + 8 * group)) & 0xFFu);
    }
};
// UE4M3 (sign bit ignored; 0x7f is NaN), halved for the doubled codebook:
// (1 + m/8) 2^(e - 8), or m 2^-10 when e = 0. One per 16 codes.
struct UE4M3Scale {
    static constexpr int GROUP = 16;
    static constexpr int BLOCK_BYTES = 16;
    // The GEMV stages a tile's fields in shared memory (`Coded4::stage`).
    static constexpr bool GEMV_STAGED = true;
    struct Block {
        uint4 packed[2];
    };
    __device__ static __forceinline__ float decode(u32 raw) {
        raw &= 0x7Fu;
        const u32 e = raw >> 3, m = raw & 7u;
        const float normal = __uint_as_float(((e + 119u) << 23) | (m << 20));
        return raw == 0x7Fu ? __uint_as_float(0x7FC00000u) : e == 0u ? (float)m * 0.0009765625f : normal;
    }
    __device__ static __forceinline__ void load(const u8 *fields, Block &block, int r) {
        block.packed[r] = seismic_ld_nc_v4(fields);
    }
    __device__ static __forceinline__ void load_staged(const u8 *fields, Block &block, int r) {
        block.packed[r] = *reinterpret_cast<const uint4 *>(fields);
    }
    __device__ static __forceinline__ float at(const Block &block, int r, int group) {
        return decode((word_of(block.packed[r], group / 4) >> (8 * (group % 4))) & 0xFFu);
    }
    // The four scales of k-block q of row half r, `decode`'s values: the
    // fields without their sign bits are E4M3 (0x7f NaN), converted two at a
    // time to f16 (exact) and halved there (exact: the least is 2^-9).
    __device__ static __forceinline__ void kblock(const Block &block, int r, int q, float (&scale)[4]) {
        const u32 fields = word_of(block.packed[r], q) & 0x7F7F7F7Fu;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            u32 pair;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(pair) : "h"((u16)(fields >> (16 * h))));
            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(pair) : "r"(pair), "r"(0x38003800u));
            const float2 values = seismic_unpack_f16x2(pair);
            scale[2 * h] = values.x;
            scale[2 * h + 1] = values.y;
        }
    }
    __device__ static __forceinline__ float read(const u8 *supers, u64 group) { return decode(supers[group]); }
    static constexpr int SUPER_WORDS = 4;
    __device__ static __forceinline__ int word_kblock(int word) { return word; }
    __device__ static __forceinline__ float staged(const u32 *words, u64 kblock, int group) {
        return decode((words[kblock % 4] >> (8 * group)) & 0xFFu);
    }
};

// The scales of k-block q of row half r: the scale type's `kblock` where it
// decodes a k-block's fields together, otherwise one `at` per group.
template <class S, class = void> struct KblockScales {
    __device__ static __forceinline__ void get(const typename S::Block &block, int r, int q,
                                               float (&scale)[64 / S::GROUP]) {
#pragma unroll
        for (int group = 0; group < 64 / S::GROUP; ++group)
            scale[group] = S::at(block, r, q * (64 / S::GROUP) + group);
    }
};
template <class S> struct KblockScales<S, decltype((void)&S::kblock)> {
    __device__ static __forceinline__ void get(const typename S::Block &block, int r, int q,
                                               float (&scale)[64 / S::GROUP]) {
        S::kblock(block, r, q, scale);
    }
};

// A coded representation of codebook C and scale fields S. `WHOLE` states
// that rows hold whole 256-code storage groups (iq4g32), so a superblock's
// four k-blocks always exist; otherwise a row's last superblock may be
// partial and its missing k-blocks are not fetched. `SCALED` selects the
// scaled GEMM (projection.cuh `gemm_tile_half`): the table values are exact
// operands and group scales multiply F32 partial products. iq4g32 keeps the
// dequantized GEMM its Qwen results were qualified with.
template <class C, class S, bool WHOLE, bool SCALED> struct Coded4 {
    const u8 *base;
    u64 stride;
    CodePlane low;
    u64 supers;

    static constexpr int GROUP = S::GROUP;
    static constexpr int GROUPS = 64 / GROUP;
    static constexpr bool BIAS = false;
    static constexpr bool DENSE = false;
    static constexpr bool SCALED_GEMM = SCALED;
    struct Raw {
        uint4 low;
    };

    __device__ __forceinline__ Raw fetch(u64 tile, u64 kblock, u32 lane) const {
        return Raw{seismic_ld_nc_na_v4(low.chunk(base, stride, tile, kblock * 512 + lane * 16))};
    }
    struct Super {
        uint4 low[4];
    };
    __device__ __forceinline__ Super fetch_superblock(u64 tile, u64 superblock, u64 kblocks, u32 lane) const {
        Super super;
#pragma unroll
        for (int q = 0; q < 4; ++q)
            if (WHOLE || 4 * superblock + q < kblocks)
                super.low[q] =
                    seismic_ld_nc_na_v4(low.chunk(base, stride, tile, (4 * superblock + q) * 512 + lane * 16));
        return super;
    }
    __device__ __forceinline__ Raw raw(const Super &super, int q, u32) const { return Raw{super.low[q]}; }
    template <class O> __device__ __forceinline__ void decode(const Raw &raw, int step, u32 (&a)[4]) const {
        C::template decode<O>(word_of(raw.low, step), a);
    }
    __device__ __forceinline__ S8Pair s8(const Raw &raw, int step) const { return C::s8(word_of(raw.low, step)); }
    __device__ __forceinline__ void coefficient(u64 row, u64 kblock, int group, float &scale, float &bias) const {
        scale = S::read(base + row * stride + supers, kblock * GROUPS + group);
        bias = 0.0f;
    }
    // The scale fields of one superblock (256 codes) of rows g, g+8.
    typedef typename S::Block Block;
    __device__ __forceinline__ Block block(u64 tile, u64 superblock, u32 lane) const {
        Block block;
#pragma unroll
        for (int r = 0; r < 2; ++r)
            S::load(base + (tile * 16 + lane / 4 + 8 * r) * stride + supers + superblock * S::BLOCK_BYTES, block, r);
        return block;
    }
    // GEMV staging of a tile's scale fields when the scale type asks for it
    // (`S::GEMV_STAGED`): a row's fields are one contiguous run, while a
    // superblock's are 16 B pieces of 16 rows, so the tile's group copies
    // the 16 runs whole into shared memory (`stage`, its threads `thread`
    // of `threads`) and each k-block reads its rows' fields there
    // (`staged_block`). Measured on GB10 (NVFP4 4096 -> 18560 and
    // 8192 -> 4096, weights not in L2): 1-7% faster than per-superblock
    // loads. A staged row is its fields and 16 B more, so the eight rows of a
    // fragment read fall in distinct banks.
    static constexpr bool GEMV_STAGED = S::GEMV_STAGED;
    __host__ __device__ static constexpr u64 staged_pitch(u64 kblocks) {
        return (kblocks + 3) / 4 * S::BLOCK_BYTES + 16;
    }
    __host__ __device__ static constexpr u64 staged_bytes(u64 kblocks) { return 16 * staged_pitch(kblocks); }
    __device__ __forceinline__ void stage(u64 tile, u64 kblocks, u8 *shared, u32 thread, u32 threads) const {
        static_assert(S::BLOCK_BYTES % 16 == 0, "staged fields copy in 16 B chunks");
        const u64 pitch = staged_pitch(kblocks);
        const u64 chunks = (pitch - 16) / 16;
        for (u64 c = thread; c < 16 * chunks; c += threads) {
            const u64 row = c / chunks, chunk = c % chunks;
            seismic_cp_async_16(shared + row * pitch + chunk * 16,
                                base + (tile * 16 + row) * stride + supers + chunk * 16);
        }
        seismic_cp_async_commit();
    }
    __device__ __forceinline__ Block staged_block(const u8 *shared, u64 kblocks, u64 superblock, u32 lane) const {
        const u64 pitch = staged_pitch(kblocks);
        Block block;
#pragma unroll
        for (int r = 0; r < 2; ++r)
            S::load_staged(shared + (lane / 4 + 8 * r) * pitch + superblock * S::BLOCK_BYTES, block, r);
        return block;
    }
    // GEMM staging: the superblock's scale fields.
    static constexpr int SUPER_WORDS = S::SUPER_WORDS;
    __device__ __forceinline__ const u8 *super_word(u64 row, u64 superblock, int word) const {
        return base + row * stride + supers + superblock * S::BLOCK_BYTES + 4 * word;
    }
    __device__ static __forceinline__ int word_kblock(int word) { return S::word_kblock(word); }
    __device__ __forceinline__ void staged_coefficient(const u32 *words, const u8 *, u32, u64 kblock, int group,
                                                       float &scale, float &bias) const {
        scale = S::staged(words, kblock, group);
        bias = 0.0f;
    }
    __device__ __forceinline__ void coefficients(const Block &block, int q, Coefficients<GROUPS> &c) const {
#pragma unroll
        for (int r = 0; r < 2; ++r)
            KblockScales<S>::get(block, r, q, c.scale[r]);
    }
    // The s8 value of fragment slot `slot` in step `step`.
    __device__ __forceinline__ u32 code(const Raw &raw, int step, u32 slot) const {
        return C::code(word_of(raw.low, step), slot);
    }
    __device__ static __forceinline__ float apply(u32 code, float scale, float) {
        return (float)(int)(signed char)code * scale;
    }
    static constexpr int CHUNKS = 32;
    __device__ __forceinline__ const u8 *chunk_source(u64 tile, u64 kblock, u32 chunk) const {
        return low.chunk(base, stride, tile, kblock * 512 + chunk * 16);
    }
    __device__ __forceinline__ Raw from_shared(const u8 *staged, u32 lane) const {
        return Raw{*reinterpret_cast<const uint4 *>(staged + lane * 16)};
    }
};

typedef Coded4<Iq4Codes, F32Scale, true, false> IQ4;
typedef Coded4<LinearCodes, F16Scale, false, true> Q4G32S;
// IQ4_NL blocks of a row whose width is not a multiple of 256 (iq4g32s).
typedef Coded4<Iq4Codes, F16Scale, false, true> IQ4G32S;
typedef Coded4<E2M1Codes, E8M0Scale, false, true> MXFP4;
typedef Coded4<E2M1Codes, UE4M3Scale, false, true> NVFP4;

// ---------------------------------------------------------------------------
// Vector-coded weights (the registry's grid codes: iq2xs, iq3xxs, iq2xxs). An
// entry of the grids plane codes eight values: grid indices name their
// magnitudes (<seismic/grids.cuh>), and seven sign bits, completed by an
// eighth of even parity, their signs. Entries cannot be split across fragment lanes, so a
// tile's grids plane (iq2xs, iq3xxs; iq2xxs's `mma16` storage is records,
// below) is [k-block][row][k-block bytes] (registry
// `PackedRowLayout::code_bit`): lane (g, t) reads the k-block bytes of rows
// g and g + 8 whole (the four lanes of a row share each load) and forms its
// slots of step s from lanes 2t, 2t + 1 of entries 2s (k 2t, 2t + 1) and
// 2s + 1 (k 2t + 8, 2t + 9). From the slots on every path is q8's: the
// codes are s8 (magnitudes up to 62), exact 16-bit operands and s8
// fragments. Every 16 (iq2xs) or 32 (iq3xxs, iq2xxs) values have an odd
// scale code 2s + 1 (a nibble of the scales plane) and every 256 one f32
// factor: value = factor * (2s + 1) * code, exact in F32. Rows hold whole
// 256-value groups.

// The eight lane sign bits of an entry's seven: the eighth makes the count
// of set bits even.
__device__ __forceinline__ u32 grid_signs(u32 signs) { return signs | ((__popc(signs) & 1u) << 7); }

// Word i of an entries array for an i known only at run time, by selects (a
// runtime index would place the array in local memory).
template <int N> __device__ __forceinline__ u32 word_at(const u32 (&w)[N], int i) {
    u32 value = w[0];
#pragma unroll
    for (int k = 1; k < N; ++k)
        value = i == k ? w[k] : value;
    return value;
}

// Four sign bits as four 0x00 / 0xff bytes (bit i to byte i).
__device__ __forceinline__ u32 sign_bytes(u32 bits) { return ((bits * 0x00204081u) & 0x01010101u) * 0xFFu; }

// iq2xs: sixteen-bit entries (a 9-bit index into the 512-point grid, then
// the signs); a scale per 16 values.
struct Iq2XsGrid {
    static constexpr int WORDS = 4;       // words of a row's k-block entries
    static constexpr int GROUP = 16;
    static constexpr int SCALE_WORDS = 2; // scales plane words per 256 values
    __device__ static __forceinline__ void load(const u8 *global, u32 (&w)[WORDS]) {
        const uint4 v = seismic_ld_nc_v4(global);
        w[0] = v.x, w[1] = v.y, w[2] = v.z, w[3] = v.w;
    }
    __device__ static __forceinline__ void load_shared(const u8 *shared, u32 (&w)[WORDS]) {
        const uint4 v = *reinterpret_cast<const uint4 *>(shared);
        w[0] = v.x, w[1] = v.y, w[2] = v.z, w[3] = v.w;
    }
    // Lanes 2t, 2t + 1 of entry e: their magnitudes (bytes 0, 1) and sign
    // bits (bits 0, 1).
    __device__ static __forceinline__ u32 lanes(const u32 (&w)[WORDS], int e, u32 t, u32 &signs) {
        const u32 entry = (w[e / 2] >> (16 * (e % 2))) & 0xFFFFu;
        signs = grid_signs(entry >> 9) >> (2 * t);
        return seismic_ld_nc_u32(reinterpret_cast<const u32 *>(grids::iq2xs) + 2 * (entry & 511u) + t / 2) >>
               (16 * (t % 2));
    }
    // Entry e whole: its eight magnitudes (lanes 0..3 in x) and eight signs.
    __device__ static __forceinline__ uint2 point(const u32 (&w)[WORDS], int e, u32 &signs) {
        const u32 entry = (word_at(w, e / 2) >> (16 * (e % 2))) & 0xFFFFu;
        signs = grid_signs(entry >> 9);
        return seismic_ld_nc_v2(grids::iq2xs + (entry & 511u));
    }
};

// iq3xxs: three-byte entries (two 8-bit indices into the 256-point grid of
// four values, then the signs); a scale per 32 values.
struct Iq3XxsGrid {
    static constexpr int WORDS = 6;
    static constexpr int GROUP = 32;
    static constexpr int SCALE_WORDS = 1;
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
    __device__ static __forceinline__ u32 lanes(const u32 (&w)[WORDS], int e, u32 t, u32 &signs) {
        const int word = 3 * e / 4, shift = 8 * (3 * e % 4);
        const u32 spill = shift > 8 ? w[word + 1] << (32 - shift) : 0u;
        const u32 entry = ((w[word] >> shift) | spill) & 0xFFFFFFu;
        signs = grid_signs((entry >> 16) & 127u) >> (2 * t);
        return seismic_ld_nc_u32(grids::iq3xxs + ((entry >> (8 * (t / 2))) & 255u)) >> (16 * (t % 2));
    }
    __device__ static __forceinline__ uint2 point(const u32 (&w)[WORDS], int e, u32 &signs) {
        const int word = 3 * e / 4, shift = 8 * (3 * e % 4);
        const u32 spill = shift > 8 ? word_at(w, word + 1) << (32 - shift) : 0u;
        const u32 entry = ((word_at(w, word) >> shift) | spill) & 0xFFFFFFu;
        signs = grid_signs((entry >> 16) & 127u);
        return make_uint2(seismic_ld_nc_u32(grids::iq3xxs + (entry & 255u)),
                          seismic_ld_nc_u32(grids::iq3xxs + ((entry >> 8) & 255u)));
    }
};

// iq2xxs: sixteen-bit entries (an 8-bit index into the 256-point grid, then
// the signs); a scale per 32 values. The entries of a k-block's rows, as
// `IQ2XXS` gathers them for its GEMM and its GEMV in fragment order.
struct Iq2XxsGrid {
    static constexpr int WORDS = 4;
    static constexpr int GROUP = 32;
    static constexpr int SCALE_WORDS = 1;
    __device__ static __forceinline__ void load(const u8 *global, u32 (&w)[WORDS]) {
        Iq2XsGrid::load(global, w);
    }
    __device__ static __forceinline__ void load_shared(const u8 *shared, u32 (&w)[WORDS]) {
        Iq2XsGrid::load_shared(shared, w);
    }
    __device__ static __forceinline__ u32 lanes(const u32 (&w)[WORDS], int e, u32 t, u32 &signs) {
        const u32 entry = (w[e / 2] >> (16 * (e % 2))) & 0xFFFFu;
        signs = grid_signs((entry >> 8) & 127u) >> (2 * t);
        return seismic_ld_nc_u32(reinterpret_cast<const u32 *>(grids::iq2xxs) + 2 * (entry & 255u) + t / 2) >>
               (16 * (t % 2));
    }
    __device__ static __forceinline__ uint2 point(const u32 (&w)[WORDS], int e, u32 &signs) {
        const u32 entry = (word_at(w, e / 2) >> (16 * (e % 2))) & 0xFFFFu;
        signs = grid_signs((entry >> 8) & 127u);
        return seismic_ld_nc_v2(grids::iq2xxs + (entry & 255u));
    }
};

template <class G> struct Grid {
    const u8 *base;
    u64 stride;
    CodePlane grids;
    u64 scales;
    u64 supers;

    static constexpr int GROUP = G::GROUP;
    static constexpr int GROUPS = 64 / GROUP;
    static constexpr bool BIAS = false;
    static constexpr bool DENSE = false;
    static constexpr bool SCALED_GEMM = true;
    // Bytes of a row's k-block of entries.
    static constexpr int KBLOCK_BYTES = 4 * G::WORDS;
    // A lane's entries of one k-block: rows g and g + 8, and its t.
    struct Raw {
        u32 g[G::WORDS];
        u32 g8[G::WORDS];
        u32 t;
    };

    __device__ __forceinline__ Raw fetch(u64 tile, u64 kblock, u32 lane) const {
        Raw raw;
        const u64 row = (kblock * 16 + lane / 4) * KBLOCK_BYTES;
        G::load(grids.chunk(base, stride, tile, row), raw.g);
        G::load(grids.chunk(base, stride, tile, row + 8 * KBLOCK_BYTES), raw.g8);
        raw.t = lane % 4;
        return raw;
    }
    struct Super {
        Raw raw[4];
    };
    __device__ __forceinline__ Super fetch_superblock(u64 tile, u64 superblock, u64, u32 lane) const {
        Super super;
#pragma unroll
        for (int q = 0; q < 4; ++q)
            super.raw[q] = fetch(tile, 4 * superblock + q, lane);
        return super;
    }
    __device__ __forceinline__ Raw raw(const Super &super, int q, u32) const { return super.raw[q]; }
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
    template <class O> __device__ __forceinline__ void decode(const Raw &raw, int step, u32 (&a)[4]) const {
        u32 low, high;
        slots(raw, step, low, high);
#pragma unroll
        for (int i = 0; i < 4; ++i)
            a[i] = O::bytes(low, high, i);
    }
    __device__ __forceinline__ S8Pair s8(const Raw &raw, int step) const {
        u32 low, high;
        slots(raw, step, low, high);
        return S8Pair{prmt(low, high, 0x6240u), prmt(low, high, 0x7351u)};
    }
    // The scale of a group of scale code `field` under `factor`.
    __device__ static __forceinline__ float scaled(float factor, u32 field) {
        return factor * (float)(2u * field + 1u);
    }
    __device__ __forceinline__ void coefficient(u64 row, u64 kblock, int group, float &scale, float &bias) const {
        const u8 *at = base + row * stride;
        const u64 index = kblock * GROUPS + group;
        const u32 field = (at[scales + index / 2] >> (4 * (index % 2))) & 15u;
        scale = scaled(*reinterpret_cast<const float *>(at + supers + 4 * (kblock / 4)), field);
        bias = 0.0f;
    }
    // A superblock's scale fields and factor of rows g, g + 8.
    struct Block {
        u32 fields[2][G::SCALE_WORDS];
        float factor[2];
    };
    __device__ __forceinline__ Block block(u64 tile, u64 superblock, u32 lane) const {
        Block block;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const u8 *at = base + (tile * 16 + lane / 4 + 8 * r) * stride;
#pragma unroll
            for (int w = 0; w < G::SCALE_WORDS; ++w)
                block.fields[r][w] = seismic_ld_nc_u32(at + scales + 4 * (G::SCALE_WORDS * superblock + w));
            block.factor[r] = __uint_as_float(seismic_ld_nc_u32(at + supers + 4 * superblock));
        }
        return block;
    }
    __device__ __forceinline__ void coefficients(const Block &block, int q, Coefficients<GROUPS> &c) const {
#pragma unroll
        for (int r = 0; r < 2; ++r)
#pragma unroll
            for (int group = 0; group < GROUPS; ++group) {
                const int n = q * GROUPS + group;
                c.scale[r][group] = scaled(block.factor[r], (block.fields[r][n / 8] >> (4 * (n % 8))) & 15u);
            }
    }
    // GEMM staging: the superblock's scale words, then its factor.
    static constexpr int SUPER_WORDS = G::SCALE_WORDS + 1;
    __device__ __forceinline__ const u8 *super_word(u64 row, u64 superblock, int word) const {
        const u8 *at = base + row * stride;
        return word < G::SCALE_WORDS ? at + scales + 4 * (G::SCALE_WORDS * superblock + word)
                                     : at + supers + 4 * superblock;
    }
    __device__ static __forceinline__ int word_kblock(int word) {
        return word < G::SCALE_WORDS ? word * (4 / G::SCALE_WORDS) : 0;
    }
    __device__ __forceinline__ void staged_coefficient(const u32 *words, const u8 *, u32, u64 kblock, int group,
                                                       float &scale, float &bias) const {
        const int n = (int)(kblock % 4) * GROUPS + group;
        scale = scaled(__uint_as_float(words[G::SCALE_WORDS]), (words[n / 8] >> (4 * (n % 8))) & 15u);
        bias = 0.0f;
    }
    __device__ __forceinline__ u32 code(const Raw &raw, int step, u32 slot) const {
        u32 low, high;
        slots(raw, step, low, high);
        return ((slot < 4 ? low : high) >> (8 * (slot % 4))) & 0xFFu;
    }
    __device__ static __forceinline__ float apply(u32 code, float scale, float) {
        return (float)(int)(signed char)code * scale;
    }
    // A tile's k-block is its 16 rows' k-block bytes, row by row.
    static constexpr int CHUNKS = KBLOCK_BYTES;
    __device__ __forceinline__ const u8 *chunk_source(u64 tile, u64 kblock, u32 chunk) const {
        return grids.chunk(base, stride, tile, kblock * 16 * KBLOCK_BYTES + chunk * 16);
    }
    __device__ __forceinline__ Raw from_shared(const u8 *staged, u32 lane) const {
        Raw raw;
        G::load_shared(staged + (lane / 4) * KBLOCK_BYTES, raw.g);
        G::load_shared(staged + (lane / 4 + 8) * KBLOCK_BYTES, raw.g8);
        raw.t = lane % 4;
        return raw;
    }
    // GEMV in entry order (`projection::gemv_entries_group`): the k of a
    // step is permuted within its scale group so that lane t's four A values
    // of a row are half of one entry, and an entry is looked up once per
    // row. Step j of group G: lane t holds values 4 h .. 4 h + 3 of entry
    // `gemv_entry`, h = `gemv_half`; the activations read the same places.
    static constexpr bool GEMV_ENTRIES = true;
    static constexpr int GROUP_ENTRIES = GROUP / 8; // entries of a group
    static constexpr int GROUP_STEPS = GROUP / 16;  // MMA steps of a group
    __device__ static __forceinline__ int gemv_entry(int group, u32 t) {
        return GROUP_ENTRIES * group + (int)t / (4 / GROUP_ENTRIES);
    }
    __device__ static __forceinline__ int gemv_half(int step, u32 t) {
        return GROUP_STEPS == 2 ? step : (int)(t % 2);
    }
    // The A pairs of values (4 h, 4 h + 1) and (4 h + 2, 4 h + 3) of entry e
    // of one row's k-block words, signed, as O operands.
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

typedef Grid<Iq2XsGrid> IQ2XS;
typedef Grid<Iq3XxsGrid> IQ3XXS;

// iq2xxs, its `mma16` storage records (the registry's
// `PackedRowLayout::record_byte`): a tile's rows hold one record per 256
// columns over the 16 rows' bytes, every record's entries first, 1024 bytes
// each, then every record's factors, 32 bytes each. A record's entries are
// [half h][lane l = 4g + t][16 bytes]; its factors are the 16 rows' f16 d as
// eight words (word g: row g low, row g + 8 high). Word w of a lane's 16 bytes is
// row g + 8 (w % 2) of k-block 2h + w / 2: its entries t (low half) and 4 + t
// (high half), the two entries of each group the lane reads in the GEMV's
// entry order (`projection::gemv_records_group`), so a lane loads exactly its
// bytes of a record, 16 at a time, and one word of factors. An entry carries
// at bit 15 bit (entry % 4) of its group's scale code s:
// value = d (2s + 1) code / 8, every product exact in F32.
struct IQ2XXS {
    const u8 *base;
    u64 stride;
    // Offset within a tile of its factors, past every group's entries.
    u64 factors;

    static constexpr int GROUP = 32;
    static constexpr int GROUPS = 2;
    static constexpr bool BIAS = false;
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
    // the scale of row half r, group n (k-block n / 2), (d / 8) (2s + 1),
    // exact. Lane t holds bit t of every scale code; its four lanes' OR is
    // the code (warp-collective). Four registers rather than sixteen keep a
    // records GEMV warp within the registers its SM residency allows
    // (`projection::GemvResidency`).
    struct Scales {
        u32 codes[2];
        float d[2];
        __device__ __forceinline__ float operator()(int r, int n) const {
            // The code of group 2q + h at nibble q + 4h; 2s + 1 as F32:
            // 2^23 + (2s + 1) less 2^23.
            const u32 code = (codes[r] >> (4 * (n / 2 + 4 * (n % 2)))) & 15u;
            return d[r] * (__uint_as_float(code * 2u + 0x4B000001u) - 8388608.0f);
        }
    };
    __device__ static __forceinline__ Scales scales(const Record &record, u32 t) {
        Scales scales;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            u32 codes = 0u;
#pragma unroll
            for (int q = 0; q < 4; ++q)
                codes |= ((record.w[2 * q + r] >> 15) & 0x00010001u) << (4 * q);
            codes <<= t;
            codes |= seismic_shfl_xor_u32(codes, 1);
            codes |= seismic_shfl_xor_u32(codes, 2);
            scales.codes[r] = codes;
            scales.d[r] = seismic_f16_to_f32((u16)(record.factors >> (16 * r))) * 0.125f;
        }
        return scales;
    }
    // Each point's eight magnitude codes c (magnitudes 8, 25, 43 -> c 0, 1,
    // 2: the magnitude's high nibble), value i at nibble i. The GEMV's
    // KSPLIT warps of a tile group form a copy in shared memory.
    struct Table {
        u32 points[256];
    };
    __device__ static __forceinline__ u32 magnitude_codes(u32 magnitudes) {
        const u32 c = (magnitudes >> 4) & 0x0F0F0F0Fu;
        return (c & 0xFu) | ((c >> 4) & 0xF0u) | ((c >> 8) & 0xF00u) | ((c >> 12) & 0xF000u);
    }
    __device__ static __forceinline__ void form_table(Table &table, u32 thread, u32 threads) {
        for (u32 i = thread; i < 256; i += threads) {
            const uint2 point = seismic_ld_nc_v2(grids::iq2xxs + i);
            table.points[i] = magnitude_codes(point.x) | (magnitude_codes(point.y) << 16);
        }
    }
    // The operand pairs (values 2j, 2j + 1 in a[j]) of `entry` (its low 16
    // bits), signed: each half is formed by one PRMT from the operand bytes
    // of the three magnitudes (`GridOperands`), its selector bytes looked up
    // by magnitude code.
    template <class O> __device__ static __forceinline__ void operands(const Table &table, u32 entry, u32 (&a)[4]);

    // ---- GEMV in fragment order (`gemv_mma_group`, a pair with another
    // representation): the k-block's entries of rows g and g + 8 gathered
    // from the quad, and the record's scales.
    typedef Grid<Iq2XxsGrid>::Raw Raw;
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
        u32 low, high;
        Grid<Iq2XxsGrid>::slots(raw, step, low, high);
#pragma unroll
        for (int i = 0; i < 4; ++i)
            a[i] = O::bytes(low, high, i);
    }
    __device__ __forceinline__ S8Pair s8(const Raw &raw, int step) const {
        u32 low, high;
        Grid<Iq2XxsGrid>::slots(raw, step, low, high);
        return S8Pair{prmt(low, high, 0x6240u), prmt(low, high, 0x7351u)};
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
        const u32 half = r / 8, shift = 15 + 16 * group;
        const u32 code = (((half ? low.y : low.x) >> shift) & 1u) | ((((half ? low.w : low.z) >> shift) & 1u) << 1) |
                         ((((half ? high.y : high.x) >> shift) & 1u) << 2) |
                         ((((half ? high.w : high.z) >> shift) & 1u) << 3);
        const float d = seismic_f16_to_f32((u16)(words[0] >> (16 * (r / 8)))) * 0.125f;
        scale = d * (float)(2u * code + 1u);
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
    // Group `group` of k-block `kblock` of stored row `row`: (d / 8) (2s + 1),
    // s's bit j at bit 15 of the group's entry j.
    __device__ __forceinline__ void coefficient(u64 row, u64 kblock, int group, float &scale, float &bias) const {
        const u64 tile = row / 16;
        const u32 g = (u32)(row % 8), r = (u32)(row % 16 / 8);
        u32 s = 0;
#pragma unroll
        for (u32 j = 0; j < 4; ++j)
            s |= ((entry_word(tile, kblock, 4 * g + j, r) >> (15 + 16 * group)) & 1u) << j;
        const u32 d = seismic_ld_nc_u32(factors_at(tile, kblock / 4) + 4 * g);
        scale = seismic_f16_to_f32((u16)(d >> (16 * r))) * 0.125f * (float)(2u * s + 1u);
        bias = 0.0f;
    }
    __device__ __forceinline__ u32 code(const Raw &raw, int step, u32 slot) const {
        u32 low, high;
        Grid<Iq2XxsGrid>::slots(raw, step, low, high);
        return ((slot < 4 ? low : high) >> (8 * (slot % 4))) & 0xFFu;
    }
    __device__ static __forceinline__ float apply(u32 code, float scale, float) {
        return (float)(int)(signed char)code * scale;
    }
};

// The operand bytes of the iq2xxs magnitudes 8, 25, 43 and their negations,
// and per magnitude code c the PRMT selector byte (low nibble: the operand's
// low byte, high nibble: its high byte) of the positive value.
template <class O> struct GridOperands;
// bf16 8, 25, 43 = 0x4100, 0x41c8, 0x422c: bytes [41 42 00 c8 | c1 c2 2c],
// a negative value's high byte the positive's selector | 4.
template <> struct GridOperands<OpBF16> {
    static constexpr u32 LOW = 0xC8004241u, HIGH = 0x002CC2C1u, SELECTORS = 0x00160302u;
    static constexpr bool SIGNED_SELECTORS = true;
};
// f16 8, 25, 43 = 0x4800, 0x4e40, 0x5160: bytes [00 40 60 48 | 4e 51]; a
// negative value flips its sign bit.
template <> struct GridOperands<OpF16> {
    static constexpr u32 LOW = 0x48604000u, HIGH = 0x0000514Eu, SELECTORS = 0x00524130u;
    static constexpr bool SIGNED_SELECTORS = false;
};

template <class O>
__device__ __forceinline__ void IQ2XXS::operands(const Table &table, u32 entry, u32 (&a)[4]) {
    using P = GridOperands<O>;
    const u32 codes = table.points[entry & 255u];
    const u32 signs = grid_signs((entry >> 8) & 127u);
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

// q5g32s (GGUF Q5_0: value = d * (code - 16)) and q5g32 (GGUF Q5_1: value =
// d * code + m, the bias -m): q5k's code planes, one f16 d (and m) per 32.
// Rows need not hold whole superblocks: the high bits are fetched per k-block
// and a missing k-block is not fetched.
template <bool MINIMUM> struct Q5G32 {
    const u8 *base;
    u64 stride;
    CodePlane low;
    CodePlane high;
    u64 supers;
    u64 supers_bytes; // payload bytes of a row's supers plane

    static constexpr int GROUP = 32;
    static constexpr int GROUPS = 2;
    static constexpr bool BIAS = MINIMUM;
    static constexpr bool DENSE = false;
    // Codes (Q5_0: less 16) are exact operands; Q5_1's minimum is folded by
    // the scaled GEMM as m * sum(x) per group.
    static constexpr bool SCALED_GEMM = true;
    static constexpr u32 OFFSET = MINIMUM ? 0u : 16u;
    static constexpr u64 FIELD = MINIMUM ? 4 : 2; // supers bytes per group
    struct Raw {
        uint4 low;
        u32 high;
    };

    __device__ __forceinline__ Raw fetch(u64 tile, u64 kblock, u32 lane) const {
        Raw raw;
        raw.low = seismic_ld_nc_na_v4(low.chunk(base, stride, tile, kblock * 512 + lane * 16));
        raw.high = seismic_ld_nc_u32(high.chunk(base, stride, tile, kblock * 128 + lane * 4));
        return raw;
    }
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
    template <class O> __device__ __forceinline__ void decode(const Raw &raw, int step, u32 (&a)[4]) const {
        const u32 word = word_of(raw.low, step);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            u32 pair = (word >> (4 * i)) & 0x000F000Fu;
            const u32 bits = raw.high >> (8 * step + i);
            pair |= ((bits << 4) & 0x10u) | ((bits << 16) & 0x100000u);
            a[i] = O::template codes<OFFSET>(pair);
        }
    }
    __device__ __forceinline__ S8Pair s8(const Raw &raw, int step) const {
        S8Pair pair = s8_nibbles(word_of(raw.low, step));
        const u32 bits = raw.high >> (8 * step);
        pair.row_g |= ((bits & 1u) << 4) | (((bits >> 4) & 1u) << 12) | (((bits >> 2) & 1u) << 20) |
                      (((bits >> 6) & 1u) << 28);
        pair.row_g8 |= (((bits >> 1) & 1u) << 4) | (((bits >> 5) & 1u) << 12) | (((bits >> 3) & 1u) << 20) |
                       (((bits >> 7) & 1u) << 28);
        if constexpr (!MINIMUM) {
            // Codes 0..31 less 16, per byte: -16..15 as s8.
            pair.row_g = __vsub4(pair.row_g, 0x10101010u);
            pair.row_g8 = __vsub4(pair.row_g8, 0x10101010u);
        }
        return pair;
    }
    __device__ __forceinline__ void coefficient(u64 row, u64 kblock, int group, float &scale, float &bias) const {
        const u8 *fields = base + row * stride + supers + (kblock * 2 + (u64)group) * FIELD;
        scale = f16_at(fields);
        bias = MINIMUM ? -f16_at(fields + 2) : 0.0f;
    }
    // The fields of one superblock (8 groups) of rows g, g+8. A row's last
    // superblock may end inside its supers plane's final 16 B: the second
    // half of the (d, m) fields is loaded only where it exists.
    struct Block {
        uint4 packed[2][MINIMUM ? 2 : 1];
    };
    __device__ __forceinline__ Block block(u64 tile, u64 superblock, u32 lane) const {
        Block block;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const u8 *fields = base + (tile * 16 + lane / 4 + 8 * r) * stride + supers + superblock * 8 * FIELD;
            block.packed[r][0] = seismic_ld_nc_v4(fields);
            if constexpr (MINIMUM)
                block.packed[r][1] =
                    superblock * 32 + 16 < supers_bytes ? seismic_ld_nc_v4(fields + 16) : make_uint4(0, 0, 0, 0);
        }
        return block;
    }
    __device__ __forceinline__ void coefficients(const Block &block, int q, Coefficients<2> &c) const {
#pragma unroll
        for (int r = 0; r < 2; ++r)
#pragma unroll
            for (int group = 0; group < 2; ++group) {
                const int index = 2 * q + group;
                if constexpr (MINIMUM) {
                    const u32 word = word_of(block.packed[r][index / 4], index % 4);
                    c.scale[r][group] = seismic_f16_to_f32((u16)(word & 0xFFFFu));
                    c.bias[r][group] = -seismic_f16_to_f32((u16)(word >> 16));
                } else {
                    c.scale[r][group] = seismic_f16_to_f32((u16)(word_of(block.packed[r][0], q) >> (16 * group)));
                }
            }
    }
    // GEMM staging: the superblock's fields (a word per group).
    static constexpr int SUPER_WORDS = MINIMUM ? 8 : 4;
    __device__ __forceinline__ const u8 *super_word(u64 row, u64 superblock, int word) const {
        return base + row * stride + supers + superblock * 8 * FIELD + 4 * word;
    }
    __device__ static __forceinline__ int word_kblock(int word) { return MINIMUM ? word / 2 : word; }
    __device__ __forceinline__ void staged_coefficient(const u32 *words, const u8 *, u32, u64 kblock, int group,
                                                       float &scale, float &bias) const {
        if constexpr (MINIMUM) {
            const u32 word = words[(kblock % 4) * 2 + group];
            scale = seismic_f16_to_f32((u16)(word & 0xFFFFu));
            bias = -seismic_f16_to_f32((u16)(word >> 16));
        } else {
            scale = seismic_f16_to_f32((u16)(words[kblock % 4] >> (16 * group)));
            bias = 0.0f;
        }
    }
    __device__ __forceinline__ u32 code(const Raw &raw, int step, u32 slot) const {
        return ((word_of(raw.low, step) >> (4 * slot)) & 15u) | (((raw.high >> (8 * step + slot)) & 1u) << 4);
    }
    __device__ static __forceinline__ float apply(u32 code, float scale, float bias) {
        if constexpr (MINIMUM)
            return seismic_fma_rn(scale, (float)code, -bias);
        else
            return scale * (float)((int)code - 16);
    }
    static constexpr int CHUNKS = 32 + 8;
    __device__ __forceinline__ const u8 *chunk_source(u64 tile, u64 kblock, u32 chunk) const {
        if (chunk < 32)
            return low.chunk(base, stride, tile, kblock * 512 + chunk * 16);
        return high.chunk(base, stride, tile, kblock * 128 + (chunk - 32) * 16);
    }
    __device__ __forceinline__ Raw from_shared(const u8 *staged, u32 lane) const {
        Raw raw;
        raw.low = *reinterpret_cast<const uint4 *>(staged + lane * 16);
        raw.high = *reinterpret_cast<const u32 *>(staged + 512 + lane * 4);
        return raw;
    }
};

typedef KQuant45<0> Q4K;
typedef KQuant45<1> Q5K;
typedef KQuant6 Q6K;

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

// Values of the decoder types for a tensor prefix `P` (mma16 layout).
#define PACKETS_PLANE(P, NAME)                                                                           \
    packets::CodePlane { ELEMENT_CAT(P, _PLANE_##NAME##_ROW_OFFSET), ELEMENT_CAT(P, _PLANE_##NAME##_BYTES_PER_ROW) }
#define PACKETS_MAKE_Q4K(P, pointer)                                                                     \
    packets::Q4K {                                                                                       \
        (const packets::u8 *)(pointer), ELEMENT_CAT(P, _ROW_STRIDE_BYTES), PACKETS_PLANE(P, CODES_LO), packets::CodePlane{0, 1}, \
            ELEMENT_CAT(P, _PLANE_SCALES_ROW_OFFSET), ELEMENT_CAT(P, _PLANE_SUPERS_ROW_OFFSET)                \
    }
#define PACKETS_MAKE_Q5K(P, pointer)                                                                     \
    packets::Q5K {                                                                                       \
        (const packets::u8 *)(pointer), ELEMENT_CAT(P, _ROW_STRIDE_BYTES), PACKETS_PLANE(P, CODES_LO), PACKETS_PLANE(P, CODES_HI), \
            ELEMENT_CAT(P, _PLANE_SCALES_ROW_OFFSET), ELEMENT_CAT(P, _PLANE_SUPERS_ROW_OFFSET)                \
    }
#define PACKETS_MAKE_Q6K(P, pointer)                                                                     \
    packets::Q6K {                                                                                       \
        (const packets::u8 *)(pointer), ELEMENT_CAT(P, _ROW_STRIDE_BYTES), PACKETS_PLANE(P, CODES_LO), PACKETS_PLANE(P, CODES_HI), \
            ELEMENT_CAT(P, _PLANE_SCALES_ROW_OFFSET), ELEMENT_CAT(P, _PLANE_SUPERS_ROW_OFFSET)                \
    }
#define PACKETS_MAKE_Q8(P, pointer)                                                                      \
    packets::Q8 {                                                                                   \
        (const packets::u8 *)(pointer), ELEMENT_CAT(P, _ROW_STRIDE_BYTES), PACKETS_PLANE(P, CODES),                \
            ELEMENT_CAT(P, _PLANE_SUPERS_ROW_OFFSET)                                                     \
    }
#define PACKETS_MAKE_C4(TYPE, P, pointer)                                                                \
    packets::TYPE {                                                                                      \
        (const packets::u8 *)(pointer), ELEMENT_CAT(P, _ROW_STRIDE_BYTES), PACKETS_PLANE(P, CODES_LO),             \
            ELEMENT_CAT(P, _PLANE_SUPERS_ROW_OFFSET)                                                     \
    }
#define PACKETS_MAKE_Q5(MINIMUM, P, pointer)                                                             \
    packets::Q5G32<MINIMUM> {                                                                            \
        (const packets::u8 *)(pointer), ELEMENT_CAT(P, _ROW_STRIDE_BYTES), PACKETS_PLANE(P, CODES_LO),             \
            PACKETS_PLANE(P, CODES_HI), ELEMENT_CAT(P, _PLANE_SUPERS_ROW_OFFSET),                        \
            ELEMENT_CAT(P, _PLANE_SUPERS_BYTES_PER_ROW)                                                  \
    }

#define PACKETS_MAKE_RECORDS(TYPE, P, pointer)                                                           \
    packets::TYPE {                                                                                      \
        (const packets::u8 *)(pointer), ELEMENT_CAT(P, _ROW_STRIDE_BYTES), ELEMENT_CAT(P, _PLANE_RECORDS_FACTORS_OFFSET) \
    }
#define PACKETS_MAKE_GRID(TYPE, P, pointer)                                                              \
    packets::TYPE {                                                                                      \
        (const packets::u8 *)(pointer), ELEMENT_CAT(P, _ROW_STRIDE_BYTES), PACKETS_PLANE(P, GRIDS),                \
            ELEMENT_CAT(P, _PLANE_SCALES_ROW_OFFSET), ELEMENT_CAT(P, _PLANE_SUPERS_ROW_OFFSET)                \
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
#elif ELEMENT_HAS(KERNEL_W0, _REPRESENTATION_Q4K)
namespace packets { typedef Q4K W0; }
#define KERNEL_W0_AT(pointer) PACKETS_MAKE_Q4K(KERNEL_W0, pointer)
#elif ELEMENT_HAS(KERNEL_W0, _REPRESENTATION_Q5K)
namespace packets { typedef Q5K W0; }
#define KERNEL_W0_AT(pointer) PACKETS_MAKE_Q5K(KERNEL_W0, pointer)
#elif ELEMENT_HAS(KERNEL_W0, _REPRESENTATION_Q6K)
namespace packets { typedef Q6K W0; }
#define KERNEL_W0_AT(pointer) PACKETS_MAKE_Q6K(KERNEL_W0, pointer)
#elif ELEMENT_HAS(KERNEL_W0, _REPRESENTATION_Q8G32S)
namespace packets { typedef Q8 W0; }
#define KERNEL_W0_AT(pointer) PACKETS_MAKE_Q8(KERNEL_W0, pointer)
#elif ELEMENT_HAS(KERNEL_W0, _REPRESENTATION_IQ4G32)
namespace packets { typedef IQ4 W0; }
#define KERNEL_W0_AT(pointer) PACKETS_MAKE_C4(IQ4, KERNEL_W0, pointer)
#elif ELEMENT_HAS(KERNEL_W0, _REPRESENTATION_Q4G32S)
namespace packets { typedef Q4G32S W0; }
#define KERNEL_W0_AT(pointer) PACKETS_MAKE_C4(Q4G32S, KERNEL_W0, pointer)
#elif ELEMENT_HAS(KERNEL_W0, _REPRESENTATION_IQ4G32S)
namespace packets { typedef IQ4G32S W0; }
#define KERNEL_W0_AT(pointer) PACKETS_MAKE_C4(IQ4G32S, KERNEL_W0, pointer)
#elif ELEMENT_HAS(KERNEL_W0, _REPRESENTATION_MXFP4G32)
namespace packets { typedef MXFP4 W0; }
#define KERNEL_W0_AT(pointer) PACKETS_MAKE_C4(MXFP4, KERNEL_W0, pointer)
#elif ELEMENT_HAS(KERNEL_W0, _REPRESENTATION_NVFP4G16)
namespace packets { typedef NVFP4 W0; }
#define KERNEL_W0_AT(pointer) PACKETS_MAKE_C4(NVFP4, KERNEL_W0, pointer)
#elif ELEMENT_HAS(KERNEL_W0, _REPRESENTATION_Q5G32S)
namespace packets { typedef Q5G32<false> W0; }
#define KERNEL_W0_AT(pointer) PACKETS_MAKE_Q5(false, KERNEL_W0, pointer)
#elif ELEMENT_HAS(KERNEL_W0, _REPRESENTATION_Q5G32)
namespace packets { typedef Q5G32<true> W0; }
#define KERNEL_W0_AT(pointer) PACKETS_MAKE_Q5(true, KERNEL_W0, pointer)
#elif ELEMENT_HAS(KERNEL_W0, _REPRESENTATION_IQ2XS)
namespace packets { typedef IQ2XS W0; }
#define KERNEL_W0_AT(pointer) PACKETS_MAKE_GRID(IQ2XS, KERNEL_W0, pointer)
#elif ELEMENT_HAS(KERNEL_W0, _REPRESENTATION_IQ3XXS)
namespace packets { typedef IQ3XXS W0; }
#define KERNEL_W0_AT(pointer) PACKETS_MAKE_GRID(IQ3XXS, KERNEL_W0, pointer)
#elif ELEMENT_HAS(KERNEL_W0, _REPRESENTATION_IQ2XXS)
namespace packets { typedef IQ2XXS W0; }
#define KERNEL_W0_AT(pointer) PACKETS_MAKE_RECORDS(IQ2XXS, KERNEL_W0, pointer)
#else
#error "KERNEL_W0: unsupported weight representation"
#endif
#if ELEMENT_HAS(KERNEL_W0, _KIND_DENSE)
#define KERNEL_W0_MATRIX(pointer, index) PACKETS_DENSE_MATRIX(KERNEL_W0, pointer, index)
#else
#define KERNEL_W0_MATRIX(pointer, index) KERNEL_W0_AT(PACKETS_PACKED_MATRIX(KERNEL_W0, pointer, index))
#endif
#endif

#if defined(KERNEL_W1)
#if ELEMENT_HAS(KERNEL_W1, _KIND_DENSE)
namespace packets { typedef Dense<ELEMENT_OF(KERNEL_W1)> W1; }
#define KERNEL_W1_AT(pointer) PACKETS_MAKE_DENSE(KERNEL_W1, pointer)
#elif !ELEMENT_HAS(KERNEL_W1, _LAYOUT_MMA16)
#error "KERNEL_W1 must be dense or bound in the mma16 layout"
#elif ELEMENT_HAS(KERNEL_W1, _REPRESENTATION_Q4K)
namespace packets { typedef Q4K W1; }
#define KERNEL_W1_AT(pointer) PACKETS_MAKE_Q4K(KERNEL_W1, pointer)
#elif ELEMENT_HAS(KERNEL_W1, _REPRESENTATION_Q5K)
namespace packets { typedef Q5K W1; }
#define KERNEL_W1_AT(pointer) PACKETS_MAKE_Q5K(KERNEL_W1, pointer)
#elif ELEMENT_HAS(KERNEL_W1, _REPRESENTATION_Q6K)
namespace packets { typedef Q6K W1; }
#define KERNEL_W1_AT(pointer) PACKETS_MAKE_Q6K(KERNEL_W1, pointer)
#elif ELEMENT_HAS(KERNEL_W1, _REPRESENTATION_Q8G32S)
namespace packets { typedef Q8 W1; }
#define KERNEL_W1_AT(pointer) PACKETS_MAKE_Q8(KERNEL_W1, pointer)
#elif ELEMENT_HAS(KERNEL_W1, _REPRESENTATION_IQ4G32)
namespace packets { typedef IQ4 W1; }
#define KERNEL_W1_AT(pointer) PACKETS_MAKE_C4(IQ4, KERNEL_W1, pointer)
#elif ELEMENT_HAS(KERNEL_W1, _REPRESENTATION_Q4G32S)
namespace packets { typedef Q4G32S W1; }
#define KERNEL_W1_AT(pointer) PACKETS_MAKE_C4(Q4G32S, KERNEL_W1, pointer)
#elif ELEMENT_HAS(KERNEL_W1, _REPRESENTATION_IQ4G32S)
namespace packets { typedef IQ4G32S W1; }
#define KERNEL_W1_AT(pointer) PACKETS_MAKE_C4(IQ4G32S, KERNEL_W1, pointer)
#elif ELEMENT_HAS(KERNEL_W1, _REPRESENTATION_MXFP4G32)
namespace packets { typedef MXFP4 W1; }
#define KERNEL_W1_AT(pointer) PACKETS_MAKE_C4(MXFP4, KERNEL_W1, pointer)
#elif ELEMENT_HAS(KERNEL_W1, _REPRESENTATION_NVFP4G16)
namespace packets { typedef NVFP4 W1; }
#define KERNEL_W1_AT(pointer) PACKETS_MAKE_C4(NVFP4, KERNEL_W1, pointer)
#elif ELEMENT_HAS(KERNEL_W1, _REPRESENTATION_Q5G32S)
namespace packets { typedef Q5G32<false> W1; }
#define KERNEL_W1_AT(pointer) PACKETS_MAKE_Q5(false, KERNEL_W1, pointer)
#elif ELEMENT_HAS(KERNEL_W1, _REPRESENTATION_Q5G32)
namespace packets { typedef Q5G32<true> W1; }
#define KERNEL_W1_AT(pointer) PACKETS_MAKE_Q5(true, KERNEL_W1, pointer)
#elif ELEMENT_HAS(KERNEL_W1, _REPRESENTATION_IQ2XS)
namespace packets { typedef IQ2XS W1; }
#define KERNEL_W1_AT(pointer) PACKETS_MAKE_GRID(IQ2XS, KERNEL_W1, pointer)
#elif ELEMENT_HAS(KERNEL_W1, _REPRESENTATION_IQ3XXS)
namespace packets { typedef IQ3XXS W1; }
#define KERNEL_W1_AT(pointer) PACKETS_MAKE_GRID(IQ3XXS, KERNEL_W1, pointer)
#elif ELEMENT_HAS(KERNEL_W1, _REPRESENTATION_IQ2XXS)
namespace packets { typedef IQ2XXS W1; }
#define KERNEL_W1_AT(pointer) PACKETS_MAKE_RECORDS(IQ2XXS, KERNEL_W1, pointer)
#else
#error "KERNEL_W1: unsupported weight representation"
#endif
#if ELEMENT_HAS(KERNEL_W1, _KIND_DENSE)
#define KERNEL_W1_MATRIX(pointer, index) PACKETS_DENSE_MATRIX(KERNEL_W1, pointer, index)
#else
#define KERNEL_W1_MATRIX(pointer, index) KERNEL_W1_AT(PACKETS_PACKED_MATRIX(KERNEL_W1, pointer, index))
#endif
#endif

#if defined(KERNEL_W2)
#if ELEMENT_HAS(KERNEL_W2, _KIND_DENSE)
namespace packets { typedef Dense<ELEMENT_OF(KERNEL_W2)> W2; }
#define KERNEL_W2_AT(pointer) PACKETS_MAKE_DENSE(KERNEL_W2, pointer)
#elif !ELEMENT_HAS(KERNEL_W2, _LAYOUT_MMA16)
#error "KERNEL_W2 must be dense or bound in the mma16 layout"
#elif ELEMENT_HAS(KERNEL_W2, _REPRESENTATION_Q4K)
namespace packets { typedef Q4K W2; }
#define KERNEL_W2_AT(pointer) PACKETS_MAKE_Q4K(KERNEL_W2, pointer)
#elif ELEMENT_HAS(KERNEL_W2, _REPRESENTATION_Q5K)
namespace packets { typedef Q5K W2; }
#define KERNEL_W2_AT(pointer) PACKETS_MAKE_Q5K(KERNEL_W2, pointer)
#elif ELEMENT_HAS(KERNEL_W2, _REPRESENTATION_Q6K)
namespace packets { typedef Q6K W2; }
#define KERNEL_W2_AT(pointer) PACKETS_MAKE_Q6K(KERNEL_W2, pointer)
#elif ELEMENT_HAS(KERNEL_W2, _REPRESENTATION_Q8G32S)
namespace packets { typedef Q8 W2; }
#define KERNEL_W2_AT(pointer) PACKETS_MAKE_Q8(KERNEL_W2, pointer)
#elif ELEMENT_HAS(KERNEL_W2, _REPRESENTATION_IQ4G32)
namespace packets { typedef IQ4 W2; }
#define KERNEL_W2_AT(pointer) PACKETS_MAKE_C4(IQ4, KERNEL_W2, pointer)
#elif ELEMENT_HAS(KERNEL_W2, _REPRESENTATION_Q4G32S)
namespace packets { typedef Q4G32S W2; }
#define KERNEL_W2_AT(pointer) PACKETS_MAKE_C4(Q4G32S, KERNEL_W2, pointer)
#elif ELEMENT_HAS(KERNEL_W2, _REPRESENTATION_IQ4G32S)
namespace packets { typedef IQ4G32S W2; }
#define KERNEL_W2_AT(pointer) PACKETS_MAKE_C4(IQ4G32S, KERNEL_W2, pointer)
#elif ELEMENT_HAS(KERNEL_W2, _REPRESENTATION_MXFP4G32)
namespace packets { typedef MXFP4 W2; }
#define KERNEL_W2_AT(pointer) PACKETS_MAKE_C4(MXFP4, KERNEL_W2, pointer)
#elif ELEMENT_HAS(KERNEL_W2, _REPRESENTATION_NVFP4G16)
namespace packets { typedef NVFP4 W2; }
#define KERNEL_W2_AT(pointer) PACKETS_MAKE_C4(NVFP4, KERNEL_W2, pointer)
#elif ELEMENT_HAS(KERNEL_W2, _REPRESENTATION_Q5G32S)
namespace packets { typedef Q5G32<false> W2; }
#define KERNEL_W2_AT(pointer) PACKETS_MAKE_Q5(false, KERNEL_W2, pointer)
#elif ELEMENT_HAS(KERNEL_W2, _REPRESENTATION_Q5G32)
namespace packets { typedef Q5G32<true> W2; }
#define KERNEL_W2_AT(pointer) PACKETS_MAKE_Q5(true, KERNEL_W2, pointer)
#elif ELEMENT_HAS(KERNEL_W2, _REPRESENTATION_IQ2XS)
namespace packets { typedef IQ2XS W2; }
#define KERNEL_W2_AT(pointer) PACKETS_MAKE_GRID(IQ2XS, KERNEL_W2, pointer)
#elif ELEMENT_HAS(KERNEL_W2, _REPRESENTATION_IQ3XXS)
namespace packets { typedef IQ3XXS W2; }
#define KERNEL_W2_AT(pointer) PACKETS_MAKE_GRID(IQ3XXS, KERNEL_W2, pointer)
#elif ELEMENT_HAS(KERNEL_W2, _REPRESENTATION_IQ2XXS)
namespace packets { typedef IQ2XXS W2; }
#define KERNEL_W2_AT(pointer) PACKETS_MAKE_RECORDS(IQ2XXS, KERNEL_W2, pointer)
#else
#error "KERNEL_W2: unsupported weight representation"
#endif
#if ELEMENT_HAS(KERNEL_W2, _KIND_DENSE)
#define KERNEL_W2_MATRIX(pointer, index) PACKETS_DENSE_MATRIX(KERNEL_W2, pointer, index)
#else
#define KERNEL_W2_MATRIX(pointer, index) KERNEL_W2_AT(PACKETS_PACKED_MATRIX(KERNEL_W2, pointer, index))
#endif
#endif

#if defined(KERNEL_W3)
#if ELEMENT_HAS(KERNEL_W3, _KIND_DENSE)
namespace packets { typedef Dense<ELEMENT_OF(KERNEL_W3)> W3; }
#define KERNEL_W3_AT(pointer) PACKETS_MAKE_DENSE(KERNEL_W3, pointer)
#elif !ELEMENT_HAS(KERNEL_W3, _LAYOUT_MMA16)
#error "KERNEL_W3 must be dense or bound in the mma16 layout"
#elif ELEMENT_HAS(KERNEL_W3, _REPRESENTATION_Q4K)
namespace packets { typedef Q4K W3; }
#define KERNEL_W3_AT(pointer) PACKETS_MAKE_Q4K(KERNEL_W3, pointer)
#elif ELEMENT_HAS(KERNEL_W3, _REPRESENTATION_Q5K)
namespace packets { typedef Q5K W3; }
#define KERNEL_W3_AT(pointer) PACKETS_MAKE_Q5K(KERNEL_W3, pointer)
#elif ELEMENT_HAS(KERNEL_W3, _REPRESENTATION_Q6K)
namespace packets { typedef Q6K W3; }
#define KERNEL_W3_AT(pointer) PACKETS_MAKE_Q6K(KERNEL_W3, pointer)
#elif ELEMENT_HAS(KERNEL_W3, _REPRESENTATION_Q8G32S)
namespace packets { typedef Q8 W3; }
#define KERNEL_W3_AT(pointer) PACKETS_MAKE_Q8(KERNEL_W3, pointer)
#elif ELEMENT_HAS(KERNEL_W3, _REPRESENTATION_IQ4G32)
namespace packets { typedef IQ4 W3; }
#define KERNEL_W3_AT(pointer) PACKETS_MAKE_C4(IQ4, KERNEL_W3, pointer)
#elif ELEMENT_HAS(KERNEL_W3, _REPRESENTATION_Q4G32S)
namespace packets { typedef Q4G32S W3; }
#define KERNEL_W3_AT(pointer) PACKETS_MAKE_C4(Q4G32S, KERNEL_W3, pointer)
#elif ELEMENT_HAS(KERNEL_W3, _REPRESENTATION_IQ4G32S)
namespace packets { typedef IQ4G32S W3; }
#define KERNEL_W3_AT(pointer) PACKETS_MAKE_C4(IQ4G32S, KERNEL_W3, pointer)
#elif ELEMENT_HAS(KERNEL_W3, _REPRESENTATION_MXFP4G32)
namespace packets { typedef MXFP4 W3; }
#define KERNEL_W3_AT(pointer) PACKETS_MAKE_C4(MXFP4, KERNEL_W3, pointer)
#elif ELEMENT_HAS(KERNEL_W3, _REPRESENTATION_NVFP4G16)
namespace packets { typedef NVFP4 W3; }
#define KERNEL_W3_AT(pointer) PACKETS_MAKE_C4(NVFP4, KERNEL_W3, pointer)
#elif ELEMENT_HAS(KERNEL_W3, _REPRESENTATION_Q5G32S)
namespace packets { typedef Q5G32<false> W3; }
#define KERNEL_W3_AT(pointer) PACKETS_MAKE_Q5(false, KERNEL_W3, pointer)
#elif ELEMENT_HAS(KERNEL_W3, _REPRESENTATION_Q5G32)
namespace packets { typedef Q5G32<true> W3; }
#define KERNEL_W3_AT(pointer) PACKETS_MAKE_Q5(true, KERNEL_W3, pointer)
#elif ELEMENT_HAS(KERNEL_W3, _REPRESENTATION_IQ2XS)
namespace packets { typedef IQ2XS W3; }
#define KERNEL_W3_AT(pointer) PACKETS_MAKE_GRID(IQ2XS, KERNEL_W3, pointer)
#elif ELEMENT_HAS(KERNEL_W3, _REPRESENTATION_IQ3XXS)
namespace packets { typedef IQ3XXS W3; }
#define KERNEL_W3_AT(pointer) PACKETS_MAKE_GRID(IQ3XXS, KERNEL_W3, pointer)
#elif ELEMENT_HAS(KERNEL_W3, _REPRESENTATION_IQ2XXS)
namespace packets { typedef IQ2XXS W3; }
#define KERNEL_W3_AT(pointer) PACKETS_MAKE_RECORDS(IQ2XXS, KERNEL_W3, pointer)
#else
#error "KERNEL_W3: unsupported weight representation"
#endif
#if ELEMENT_HAS(KERNEL_W3, _KIND_DENSE)
#define KERNEL_W3_MATRIX(pointer, index) PACKETS_DENSE_MATRIX(KERNEL_W3, pointer, index)
#else
#define KERNEL_W3_MATRIX(pointer, index) KERNEL_W3_AT(PACKETS_PACKED_MATRIX(KERNEL_W3, pointer, index))
#endif
#endif
