// Packed-weight decoders and element access for CUDA native kernels, over the
// Seismic CUDA prelude (`seismic_*` PTX helpers).
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
//     #define SJ_W0 SEISMIC_GATE_WEIGHT
// which defines `sj::W0` (the decoder type of the bound representation) and
// `SJ_W0_AT(pointer)` (its value for a tensor base pointer). Slots W0..W3
// exist. The activation element is `SJ_ACT` (default `SEISMIC_ELEMENT_A`).

#define SJ_CAT_(a, b) a##b
#define SJ_CAT(a, b) SJ_CAT_(a, b)
// 1 when the macro `<prefix><suffix>` is defined as 1, else 0; usable in
// both `#if` and constant expressions.
#define SJ_SECOND_(a, b, ...) b
#define SJ_SECOND(...) SJ_SECOND_(__VA_ARGS__, 0, 0)
#define SJ_PROBE_1 ~, 1
#define SJ_IS_ONE(value) SJ_SECOND(SJ_CAT(SJ_PROBE_, value))
#define SJ_HAS(prefix, suffix) SJ_IS_ONE(SJ_CAT(prefix, suffix))

#ifndef SJ_ACT
#define SJ_ACT SEISMIC_ELEMENT_A
#endif

namespace sj {

typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

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
            prmt(slots_low ^ 0x80808080u, slots_high ^ 0x80808080u, (u32)i | ((u32)(4 + i) << 8));
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
};

// ---------------------------------------------------------------------------
// Dense elements. `KIND` is 0 = f32, 1 = bf16, 2 = f16.

#define SJ_DENSE_KIND(prefix)                                                                      \
    (SJ_HAS(prefix, _REPRESENTATION_F32)    ? 0                                                    \
     : SJ_HAS(prefix, _REPRESENTATION_BF16) ? 1                                                    \
     : SJ_HAS(prefix, _REPRESENTATION_F16)  ? 2                                                    \
                                            : -1)

template <int KIND> struct Dense {
    static_assert(KIND >= 0, "a dense f32, bf16 or f16 representation is required");
    __device__ static __forceinline__ float load(const u8 *base, u64 index) {
        if constexpr (KIND == 0)
            return reinterpret_cast<const float *>(base)[index];
        else if constexpr (KIND == 1)
            return seismic_bf16_to_f32(reinterpret_cast<const u16 *>(base)[index]);
        else
            return seismic_f16_to_f32(reinterpret_cast<const u16 *>(base)[index]);
    }
    __device__ static __forceinline__ float round(float value) {
        if constexpr (KIND == 0)
            return value;
        else if constexpr (KIND == 1)
            return seismic_bf16_to_f32(seismic_f32_to_bf16(value));
        else
            return seismic_f16_to_f32(seismic_f32_to_f16(value));
    }
    __device__ static __forceinline__ void store(u8 *base, u64 index, float value) {
        if constexpr (KIND == 0)
            reinterpret_cast<float *>(base)[index] = value;
        else if constexpr (KIND == 1)
            reinterpret_cast<u16 *>(base)[index] = seismic_f32_to_bf16(value);
        else
            reinterpret_cast<u16 *>(base)[index] = seismic_f32_to_f16(value);
    }
    // Elements index and index + 1: one vector store when index is even.
    __device__ static __forceinline__ void store2(u8 *base, u64 index, float first, float second) {
        if (index % 2 != 0) {
            store(base, index, first);
            store(base, index + 1, second);
        } else if constexpr (KIND == 0) {
            reinterpret_cast<float2 *>(base)[index / 2] = make_float2(first, second);
        } else if constexpr (KIND == 1) {
            reinterpret_cast<u32 *>(base)[index / 2] = seismic_pack_bf16x2(first, second);
        } else {
            reinterpret_cast<u32 *>(base)[index / 2] = seismic_pack_f16x2(first, second);
        }
    }
};

// The MMA operand type of a 16-bit dense element kind. An F32 element has no
// exact 16-bit operand, so it has none.
template <int KIND> struct OperandOf;
template <> struct OperandOf<1> {
    using type = OpBF16;
};
template <> struct OperandOf<2> {
    using type = OpF16;
};

// The activation element (instantiated only by kernels that bind one).
using Act = Dense<SJ_DENSE_KIND(SJ_ACT)>;

// ---------------------------------------------------------------------------
// Packed weights in the mma16 layout. Every decoder provides
//   GEMV:  Super fetch_superblock(tile, superblock, kblocks, lane) (the lane
//          chunks of 4 k-blocks), Raw raw(super, q, lane) (k-block q; warp-
//          collective), decode<Op>(raw, step, a), Block block(tile,
//          superblock, lane) (packed coefficients, one vector load per row),
//          coefficients(block, q, c)
//   GEMM:  CHUNKS, chunk_source(tile, kb, chunk), from_shared(staged, lane),
//          COEF_WORDS, coef_word(row, kb, word) (the 4-byte words of a row's
//          coefficients that k-block kb reads, staged with the codes),
//          staged_coefficient(words, kb, group, scale, bias)
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
    __device__ static __forceinline__ u32 field(const u32 (&packed)[3], u32 index) {
        const u32 bit = 6 * index;
        const u32 word = bit / 32;
        const u64 wide = (u64)packed[word] | (word < 2 ? (u64)packed[word + 1] << 32 : 0ull);
        return (u32)(wide >> (bit % 32)) & 63u;
    }
    // GEMM staging: the 12 packed scale bytes and the (d, dmin) word of the
    // k-block's superblock.
    static constexpr int COEF_WORDS = 4;
    __device__ __forceinline__ const u8 *coef_word(u64 row, u64 kblock, int word) const {
        const u8 *line = base + row * stride;
        return word < 3 ? line + scales + (kblock / 4) * 12 + 4 * word : line + supers + (kblock / 4) * 4;
    }
    __device__ __forceinline__ void staged_coefficient(const u32 *words, u64 kblock, int group, float &scale,
                                                       float &bias) const {
        const u32 packed[3] = {words[0], words[1], words[2]};
        const u32 index = 2 * ((u32)(kblock % 4) * 2 + (u32)group);
        scale = seismic_f16_to_f32((u16)(words[3] & 0xFFFFu)) * (float)field(packed, index);
        bias = seismic_f16_to_f32((u16)(words[3] >> 16)) * (float)field(packed, index + 1);
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
    // GEMM staging: the k-block's four int8 scales and the aligned word
    // holding the superblock's f16 d (the supers plane is padded to 16 B).
    static constexpr int COEF_WORDS = 2;
    __device__ __forceinline__ const u8 *coef_word(u64 row, u64 kblock, int word) const {
        const u8 *line = base + row * stride;
        return word == 0 ? line + scales + kblock * 4 : line + supers + ((kblock / 4) * 2 & ~3ull);
    }
    __device__ __forceinline__ void staged_coefficient(const u32 *words, u64 kblock, int group, float &scale,
                                                       float &bias) const {
        const u16 d = (u16)(words[1] >> (16 * ((kblock / 4) % 2)));
        scale = seismic_f16_to_f32(d) * (float)(((int)(words[0] << (24 - 8 * group))) >> 24);
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
struct Q8Group {
    const u8 *base;
    u64 stride;
    CodePlane codes_plane;
    u64 supers;

    static constexpr int GROUP = 32;
    static constexpr int GROUPS = 2;
    static constexpr bool BIAS = false;
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
    // GEMM staging: the k-block's two f16 group scales.
    static constexpr int COEF_WORDS = 1;
    __device__ __forceinline__ const u8 *coef_word(u64 row, u64 kblock, int) const {
        return base + row * stride + supers + kblock * 4;
    }
    __device__ __forceinline__ void staged_coefficient(const u32 *words, u64, int group, float &scale,
                                                       float &bias) const {
        scale = seismic_f16_to_f32((u16)(words[0] >> (16 * group)));
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

typedef KQuant45<0> Q4K;
typedef KQuant45<1> Q5K;
typedef KQuant6 Q6K;

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

} // namespace sj

// Values of the decoder types for a tensor prefix `P` (mma16 layout).
#define SJ_PLANE(P, NAME)                                                                           \
    sj::CodePlane { SJ_CAT(P, _PLANE_##NAME##_ROW_OFFSET), SJ_CAT(P, _PLANE_##NAME##_BYTES_PER_ROW) }
#define SJ_MAKE_Q4K(P, pointer)                                                                     \
    sj::Q4K {                                                                                       \
        (const sj::u8 *)(pointer), SJ_CAT(P, _ROW_STRIDE_BYTES), SJ_PLANE(P, CODES_LO), sj::CodePlane{0, 1}, \
            SJ_CAT(P, _PLANE_SCALES_ROW_OFFSET), SJ_CAT(P, _PLANE_SUPERS_ROW_OFFSET)                \
    }
#define SJ_MAKE_Q5K(P, pointer)                                                                     \
    sj::Q5K {                                                                                       \
        (const sj::u8 *)(pointer), SJ_CAT(P, _ROW_STRIDE_BYTES), SJ_PLANE(P, CODES_LO), SJ_PLANE(P, CODES_HI), \
            SJ_CAT(P, _PLANE_SCALES_ROW_OFFSET), SJ_CAT(P, _PLANE_SUPERS_ROW_OFFSET)                \
    }
#define SJ_MAKE_Q6K(P, pointer)                                                                     \
    sj::Q6K {                                                                                       \
        (const sj::u8 *)(pointer), SJ_CAT(P, _ROW_STRIDE_BYTES), SJ_PLANE(P, CODES_LO), SJ_PLANE(P, CODES_HI), \
            SJ_CAT(P, _PLANE_SCALES_ROW_OFFSET), SJ_CAT(P, _PLANE_SUPERS_ROW_OFFSET)                \
    }
#define SJ_MAKE_Q8(P, pointer)                                                                      \
    sj::Q8Group {                                                                                   \
        (const sj::u8 *)(pointer), SJ_CAT(P, _ROW_STRIDE_BYTES), SJ_PLANE(P, CODES),                \
            SJ_CAT(P, _PLANE_SUPERS_ROW_OFFSET)                                                     \
    }

// Slot bindings. Each bound slot must name a tensor in the mma16 layout.
#if defined(SJ_W0)
#if !SJ_HAS(SJ_W0, _LAYOUT_MMA16)
#error "SJ_W0 must be bound in the mma16 layout"
#elif SJ_HAS(SJ_W0, _REPRESENTATION_Q4K)
namespace sj { typedef Q4K W0; }
#define SJ_W0_AT(pointer) SJ_MAKE_Q4K(SJ_W0, pointer)
#elif SJ_HAS(SJ_W0, _REPRESENTATION_Q5K)
namespace sj { typedef Q5K W0; }
#define SJ_W0_AT(pointer) SJ_MAKE_Q5K(SJ_W0, pointer)
#elif SJ_HAS(SJ_W0, _REPRESENTATION_Q6K)
namespace sj { typedef Q6K W0; }
#define SJ_W0_AT(pointer) SJ_MAKE_Q6K(SJ_W0, pointer)
#elif SJ_HAS(SJ_W0, _REPRESENTATION_Q8G32S)
namespace sj { typedef Q8Group W0; }
#define SJ_W0_AT(pointer) SJ_MAKE_Q8(SJ_W0, pointer)
#else
#error "SJ_W0: unsupported weight representation"
#endif
#endif

#if defined(SJ_W1)
#if !SJ_HAS(SJ_W1, _LAYOUT_MMA16)
#error "SJ_W1 must be bound in the mma16 layout"
#elif SJ_HAS(SJ_W1, _REPRESENTATION_Q4K)
namespace sj { typedef Q4K W1; }
#define SJ_W1_AT(pointer) SJ_MAKE_Q4K(SJ_W1, pointer)
#elif SJ_HAS(SJ_W1, _REPRESENTATION_Q5K)
namespace sj { typedef Q5K W1; }
#define SJ_W1_AT(pointer) SJ_MAKE_Q5K(SJ_W1, pointer)
#elif SJ_HAS(SJ_W1, _REPRESENTATION_Q6K)
namespace sj { typedef Q6K W1; }
#define SJ_W1_AT(pointer) SJ_MAKE_Q6K(SJ_W1, pointer)
#elif SJ_HAS(SJ_W1, _REPRESENTATION_Q8G32S)
namespace sj { typedef Q8Group W1; }
#define SJ_W1_AT(pointer) SJ_MAKE_Q8(SJ_W1, pointer)
#else
#error "SJ_W1: unsupported weight representation"
#endif
#endif

#if defined(SJ_W2)
#if !SJ_HAS(SJ_W2, _LAYOUT_MMA16)
#error "SJ_W2 must be bound in the mma16 layout"
#elif SJ_HAS(SJ_W2, _REPRESENTATION_Q4K)
namespace sj { typedef Q4K W2; }
#define SJ_W2_AT(pointer) SJ_MAKE_Q4K(SJ_W2, pointer)
#elif SJ_HAS(SJ_W2, _REPRESENTATION_Q5K)
namespace sj { typedef Q5K W2; }
#define SJ_W2_AT(pointer) SJ_MAKE_Q5K(SJ_W2, pointer)
#elif SJ_HAS(SJ_W2, _REPRESENTATION_Q6K)
namespace sj { typedef Q6K W2; }
#define SJ_W2_AT(pointer) SJ_MAKE_Q6K(SJ_W2, pointer)
#elif SJ_HAS(SJ_W2, _REPRESENTATION_Q8G32S)
namespace sj { typedef Q8Group W2; }
#define SJ_W2_AT(pointer) SJ_MAKE_Q8(SJ_W2, pointer)
#else
#error "SJ_W2: unsupported weight representation"
#endif
#endif

#if defined(SJ_W3)
#if !SJ_HAS(SJ_W3, _LAYOUT_MMA16)
#error "SJ_W3 must be bound in the mma16 layout"
#elif SJ_HAS(SJ_W3, _REPRESENTATION_Q4K)
namespace sj { typedef Q4K W3; }
#define SJ_W3_AT(pointer) SJ_MAKE_Q4K(SJ_W3, pointer)
#elif SJ_HAS(SJ_W3, _REPRESENTATION_Q5K)
namespace sj { typedef Q5K W3; }
#define SJ_W3_AT(pointer) SJ_MAKE_Q5K(SJ_W3, pointer)
#elif SJ_HAS(SJ_W3, _REPRESENTATION_Q6K)
namespace sj { typedef Q6K W3; }
#define SJ_W3_AT(pointer) SJ_MAKE_Q6K(SJ_W3, pointer)
#elif SJ_HAS(SJ_W3, _REPRESENTATION_Q8G32S)
namespace sj { typedef Q8Group W3; }
#define SJ_W3_AT(pointer) SJ_MAKE_Q8(SJ_W3, pointer)
#else
#error "SJ_W3: unsupported weight representation"
#endif
#endif
