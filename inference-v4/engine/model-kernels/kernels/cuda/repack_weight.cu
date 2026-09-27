// Exact GGUF -> resident conversion (the Seismic registry's registered
// `repack`), one kernel per (source format, target layout) selected by the
// element macros; the CUDA form of `metal/repack_weight.metal`, with the same
// tiling: one block of 32 threads converts one 16-row x 32-column tile of the
// [B, N, K] view, and stored-but-unoccupied bytes are written as zero.

#if !defined(SEISMIC_ELEMENT_E_KIND_EXTERNAL) || !defined(SEISMIC_ELEMENT_U_KIND_PACKED)
#error "repack_weight converts an external GGUF source into packed resident storage"
#endif

#if defined(SEISMIC_ELEMENT_E_REPRESENTATION_GGUF_Q4_K) && defined(SEISMIC_ELEMENT_U_REPRESENTATION_Q4K)
#define REPACK_KQUANT 1
#define REPACK_CODE_BITS 4u
#elif defined(SEISMIC_ELEMENT_E_REPRESENTATION_GGUF_Q5_K) && defined(SEISMIC_ELEMENT_U_REPRESENTATION_Q5K)
#define REPACK_KQUANT 1
#define REPACK_CODE_BITS 5u
#elif defined(SEISMIC_ELEMENT_E_REPRESENTATION_GGUF_Q6_K) && defined(SEISMIC_ELEMENT_U_REPRESENTATION_Q6K)
#define REPACK_Q6K 1
#define REPACK_CODE_BITS 6u
#elif defined(SEISMIC_ELEMENT_E_REPRESENTATION_GGUF_Q8_0) && defined(SEISMIC_ELEMENT_U_REPRESENTATION_Q8G32S)
#define REPACK_Q8 1
#define REPACK_CODE_BITS 8u
#elif defined(SEISMIC_ELEMENT_E_REPRESENTATION_GGUF_IQ4_XS) && defined(SEISMIC_ELEMENT_U_REPRESENTATION_IQ4G32)
#define REPACK_IQ4 1
#define REPACK_CODE_BITS 4u
#else
#error "repack_weight binding is not a registered GGUF-to-resident conversion"
#endif

typedef unsigned char u8;
typedef unsigned int u32;
typedef unsigned long long u64;

#define GROUP ((u32)SEISMIC_ELEMENT_E_LOGICAL_GROUP)
#define TILE_ROWS 16u
#define TILE_COLUMNS 32u

// ---- GGUF source packet readers ------------------------------------------

__device__ __forceinline__ u32 source_code(const u8 *in, u32 p) {
#if defined(REPACK_KQUANT) && REPACK_CODE_BITS == 4u
    return (in[16 + (p / 64) * 32 + p % 32] >> ((p % 64 / 32) * 4)) & 15u;
#elif defined(REPACK_KQUANT)
    u32 low = (in[48 + (p / 64) * 32 + p % 32] >> ((p % 64 / 32) * 4)) & 15u;
    u32 high = (in[16 + p % 32] >> (p / 32)) & 1u;
    return low | (high << 4);
#elif defined(REPACK_Q6K)
    u32 low = (in[(p / 128) * 64 + p % 64] >> ((p % 128 / 64) * 4)) & 15u;
    u32 high = (in[128 + (p / 128) * 32 + p % 32] >> ((p % 128 / 32) * 2)) & 3u;
    return low | (high << 4);
#elif defined(REPACK_Q8)
    return (u32)in[2 + p];
#else
    return (in[8 + (p / 32) * 16 + p % 16] >> ((p % 32 / 16) * 4)) & 15u;
#endif
}

#if defined(REPACK_KQUANT)
__device__ __forceinline__ u32 kquant_local(const u8 *in, u32 j, u32 field) {
    u32 index = j % 4;
    u32 low = in[4 + field * 4 + index];
    u32 high = in[12 + index];
    return j < 4 ? (low & 63u) : (((high >> (4 * field)) & 15u) | ((low >> 6) << 4));
}

__device__ __forceinline__ u32 kquant_locals_word(const u8 *in, u32 w) {
    u32 value = 0;
    for (u32 field = 0; field < 16; ++field) {
        int shift = (int)(field * 6) - (int)(w * 32);
        if (shift <= -6 || shift >= 32) continue;
        u32 local = kquant_local(in, field / 2, field % 2);
        value |= shift >= 0 ? local << shift : local >> (-shift);
    }
    return value;
}
#endif

#if defined(REPACK_IQ4)
__device__ __forceinline__ float iq4_scale(const u8 *in, u32 j) {
    float base = seismic_f16_to_f32((unsigned short)(in[0] | (in[1] << 8)));
    u32 low = (in[4 + j / 2] >> ((j % 2) * 4)) & 15u;
    u32 high = ((u32)(in[2] | (in[3] << 8)) >> (2 * j)) & 3u;
    return __fmul_rn(base, (float)((int)(low | (high << 4)) - 32));
}
#endif

__device__ __forceinline__ void write_scales(const u8 *in, u8 *out) {
#if defined(REPACK_KQUANT)
    for (u32 w = 0; w < 3; ++w) reinterpret_cast<u32 *>(out)[w] = kquant_locals_word(in, w);
#elif defined(REPACK_Q6K)
    for (u32 w = 0; w < 4; ++w)
        reinterpret_cast<u32 *>(out)[w] = (u32)in[192 + 4 * w] | ((u32)in[193 + 4 * w] << 8) |
                                          ((u32)in[194 + 4 * w] << 16) | ((u32)in[195 + 4 * w] << 24);
#endif
}

__device__ __forceinline__ void write_supers(const u8 *in, u8 *out) {
#if defined(REPACK_KQUANT)
    reinterpret_cast<u32 *>(out)[0] = (u32)in[0] | ((u32)in[1] << 8) | ((u32)in[2] << 16) | ((u32)in[3] << 24);
#elif defined(REPACK_Q6K)
    reinterpret_cast<unsigned short *>(out)[0] = (unsigned short)(in[208] | (in[209] << 8));
#elif defined(REPACK_Q8)
    reinterpret_cast<unsigned short *>(out)[0] = (unsigned short)(in[0] | (in[1] << 8));
#else
    for (u32 j = 0; j < 8; ++j) reinterpret_cast<float *>(out)[j] = iq4_scale(in, j);
#endif
}

__device__ __forceinline__ u32 pack_eight(const u8 *in, u32 first, u32 shift, u32 bits) {
    u32 value = 0;
    for (u32 i = 0; i < 8; ++i) value |= ((source_code(in, first + i) >> shift) & ((1u << bits) - 1u)) << (i * bits);
    return value;
}

__device__ __forceinline__ u32 pack_four_bytes(const u8 *in, u32 first) {
    u32 value = 0;
    for (u32 i = 0; i < 4; ++i) value |= source_code(in, first + i) << (8 * i);
    return value;
}

__device__ __forceinline__ void zero_bytes(u8 *out, u64 count) {
    for (u64 i = 0; i < count; ++i) out[i] = 0;
}

__device__ __forceinline__ const u8 *source_packet(const u8 *source, const seismic_words_t &seismic_words_value,
                                                   u64 b, u64 n, u64 packet) {
    return source + (b * SEISMIC_SOURCE_STRIDE_0 + n * SEISMIC_SOURCE_STRIDE_1 + packet * SEISMIC_SOURCE_STRIDE_2) *
                        SEISMIC_ELEMENT_E_PACKET_SIZE;
}

#if defined(SEISMIC_ELEMENT_U_LAYOUT_PACKET)

__device__ __forceinline__ u8 *destination_packet(u8 *destination, const seismic_words_t &seismic_words_value,
                                                  u64 b, u64 n, u64 packet) {
    return destination + (b * SEISMIC_RESULT_0_STRIDE_0 + n * SEISMIC_RESULT_0_STRIDE_1 +
                          packet * SEISMIC_RESULT_0_STRIDE_2) * SEISMIC_ELEMENT_U_PACKET_SIZE;
}

__device__ __forceinline__ bool packet_plane_byte(u32 byte) {
    bool inside = byte >= SEISMIC_ELEMENT_U_PLANE_0_OFFSET &&
                  byte < SEISMIC_ELEMENT_U_PLANE_0_OFFSET + SEISMIC_ELEMENT_U_PLANE_0_BYTES_PER_GROUP;
    inside = inside || (byte >= SEISMIC_ELEMENT_U_PLANE_1_OFFSET &&
                        byte < SEISMIC_ELEMENT_U_PLANE_1_OFFSET + SEISMIC_ELEMENT_U_PLANE_1_BYTES_PER_GROUP);
#if SEISMIC_ELEMENT_U_PLANE_COUNT > 2
    inside = inside || (byte >= SEISMIC_ELEMENT_U_PLANE_2_OFFSET &&
                        byte < SEISMIC_ELEMENT_U_PLANE_2_OFFSET + SEISMIC_ELEMENT_U_PLANE_2_BYTES_PER_GROUP);
#endif
#if SEISMIC_ELEMENT_U_PLANE_COUNT > 3
    inside = inside || (byte >= SEISMIC_ELEMENT_U_PLANE_3_OFFSET &&
                        byte < SEISMIC_ELEMENT_U_PLANE_3_OFFSET + SEISMIC_ELEMENT_U_PLANE_3_BYTES_PER_GROUP);
#endif
    return inside;
}

extern "C" __global__ void repack_weight(SEISMIC_KERNEL_PARAMS) {
    const u8 *source = reinterpret_cast<const u8 *>(SEISMIC_PTR(SEISMIC_BUFFER_SOURCE));
    u8 *destination = reinterpret_cast<u8 *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
    const u32 lane = threadIdx.x;
    const u64 b = blockIdx.z;
    const u64 row0 = (u64)blockIdx.y * TILE_ROWS;
    const u32 column0 = blockIdx.x * TILE_COLUMNS;
    const u64 packets = (SEISMIC_DIM_K + GROUP - 1) / GROUP;
    if (column0 >= packets * GROUP) return;
    const u32 packet = column0 / GROUP;
    const u32 within = column0 % GROUP;
    const u32 code_words = REPACK_CODE_BITS;
    for (u32 item = lane; item < TILE_ROWS * code_words; item += TILE_COLUMNS) {
        u64 n = row0 + item / code_words;
        if (n >= SEISMIC_DIM_N) continue;
        const u8 *in = source_packet(source, seismic_words_value, b, n, packet);
        u32 word = within / 32 * code_words + item % code_words;
        u32 first = (word * 32) / REPACK_CODE_BITS;
        u32 last = (word * 32 + 31) / REPACK_CODE_BITS;
        u32 value = 0;
        for (u32 code = first; code <= last; ++code) {
            int shift = (int)(code * REPACK_CODE_BITS) - (int)(word * 32);
            u32 raw = source_code(in, code);
            value |= shift >= 0 ? raw << shift : raw >> (-shift);
        }
        u8 *out = destination_packet(destination, seismic_words_value, b, n, packet);
        reinterpret_cast<u32 *>(out + SEISMIC_ELEMENT_U_PLANE_0_OFFSET)[word] = value;
    }
    if (within != 0 || lane >= TILE_ROWS || row0 + lane >= SEISMIC_DIM_N) return;
    const u64 n = row0 + lane;
    const u8 *in = source_packet(source, seismic_words_value, b, n, packet);
    u8 *out = destination_packet(destination, seismic_words_value, b, n, packet);
#if defined(REPACK_KQUANT) || defined(REPACK_Q6K)
    write_scales(in, out + SEISMIC_ELEMENT_U_PLANE_1_OFFSET);
    write_supers(in, out + SEISMIC_ELEMENT_U_PLANE_2_OFFSET);
#else
    write_supers(in, out + SEISMIC_ELEMENT_U_PLANE_1_OFFSET);
#endif
    for (u32 byte = 0; byte < SEISMIC_ELEMENT_U_PACKET_SIZE; ++byte)
        if (!packet_plane_byte(byte)) out[byte] = 0;
}

#else

__device__ __forceinline__ u8 *row_base(u8 *destination, const seismic_words_t &seismic_words_value, u64 b, u64 n) {
    return destination + (b * SEISMIC_RESULT_0_STRIDE_0 + n * SEISMIC_RESULT_0_STRIDE_1) *
                             SEISMIC_RESULT_0_ROW_STRIDE_BYTES;
}

#if defined(SEISMIC_ELEMENT_U_LAYOUT_MMA16)
// See `metal/repack_weight.metal` and the registry `PackedRowLayout::code_bit`.
__device__ __forceinline__ u32 fragment_code(const u8 *source, const seismic_words_t &seismic_words_value, u64 b,
                                             u64 row0, u64 packets, u32 block, u32 step, u32 lane, u32 slot) {
    u64 n = row0 + lane / 4 + 8 * (slot % 2);
    u64 column = (u64)block * 64 + step * 16 + 2 * (lane % 4) + 8 * ((slot % 4) / 2) + slot / 4;
    if (n >= SEISMIC_DIM_N || column >= packets * GROUP) return 0;
    return source_code(source_packet(source, seismic_words_value, b, n, column / GROUP), (u32)(column % GROUP));
}

__device__ __forceinline__ u32 fragment_slice(const u8 *source, const seismic_words_t &seismic_words_value, u64 b,
                                              u64 row0, u64 packets, u32 block, u32 step, u32 lane, u32 shift,
                                              u32 bits) {
    u32 value = 0;
    for (u32 slot = 0; slot < 8; ++slot)
        value |= ((fragment_code(source, seismic_words_value, b, row0, packets, block, step, lane, slot) >> shift) &
                  ((1u << bits) - 1u)) << (slot * bits);
    return value;
}

__device__ __forceinline__ u8 *tile_plane_byte(u8 *destination, const seismic_words_t &seismic_words_value, u64 b,
                                               u64 row0, u64 position, u64 row_bytes, u64 offset) {
    return row_base(destination, seismic_words_value, b, row0 + position / row_bytes) + offset + position % row_bytes;
}
#endif

extern "C" __global__ void repack_weight(SEISMIC_KERNEL_PARAMS) {
    const u8 *source = reinterpret_cast<const u8 *>(SEISMIC_PTR(SEISMIC_BUFFER_SOURCE));
    u8 *destination = reinterpret_cast<u8 *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
    const u32 lane = threadIdx.x;
    const u64 b = blockIdx.z;
    const u64 row0 = (u64)blockIdx.y * TILE_ROWS;
    const u32 column0 = blockIdx.x * TILE_COLUMNS;
    const u64 packets = (SEISMIC_DIM_K + GROUP - 1) / GROUP;
    const u64 stored_columns = SEISMIC_RESULT_0_ROW_GROUPS * GROUP;
    if (column0 >= stored_columns) return;
    const bool occupied = column0 < packets * GROUP;
    const u32 within = column0 % GROUP;
#if defined(SEISMIC_ELEMENT_U_LAYOUT_MMA16)
    const u64 rows = TILE_ROWS;
#else
    const u64 rows = SEISMIC_DIM_N - row0 < TILE_ROWS ? SEISMIC_DIM_N - row0 : TILE_ROWS;
#endif

#if defined(SEISMIC_ELEMENT_U_LAYOUT_ROWS16)
    if (lane < rows) {
        const u64 n = row0 + lane;
        const u8 *in = source_packet(source, seismic_words_value, b, n, column0 / GROUP);
        u8 *row = row_base(destination, seismic_words_value, b, n);
#if defined(SEISMIC_ELEMENT_U_PLANE_CODES)
        uint4 *codes = reinterpret_cast<uint4 *>(row + SEISMIC_RESULT_0_PLANE_CODES_ROW_OFFSET + column0);
        codes[0] = make_uint4(pack_four_bytes(in, within), pack_four_bytes(in, within + 4),
                              pack_four_bytes(in, within + 8), pack_four_bytes(in, within + 12));
        codes[1] = make_uint4(pack_four_bytes(in, within + 16), pack_four_bytes(in, within + 20),
                              pack_four_bytes(in, within + 24), pack_four_bytes(in, within + 28));
#else
        *reinterpret_cast<uint4 *>(row + SEISMIC_RESULT_0_PLANE_CODES_LO_ROW_OFFSET + column0 / 2) =
            make_uint4(pack_eight(in, within, 0, 4), pack_eight(in, within + 8, 0, 4),
                       pack_eight(in, within + 16, 0, 4), pack_eight(in, within + 24, 0, 4));
#if defined(SEISMIC_ELEMENT_U_PLANE_CODES_HI) && SEISMIC_ELEMENT_U_PLANE_CODES_HI_CODE_BITS == 1
        *reinterpret_cast<u32 *>(row + SEISMIC_RESULT_0_PLANE_CODES_HI_ROW_OFFSET + column0 / 8) =
            pack_eight(in, within, 4, 1) | (pack_eight(in, within + 8, 4, 1) << 8) |
            (pack_eight(in, within + 16, 4, 1) << 16) | (pack_eight(in, within + 24, 4, 1) << 24);
#elif defined(SEISMIC_ELEMENT_U_PLANE_CODES_HI)
        *reinterpret_cast<uint2 *>(row + SEISMIC_RESULT_0_PLANE_CODES_HI_ROW_OFFSET + column0 / 4) =
            make_uint2(pack_eight(in, within, 4, 2) | (pack_eight(in, within + 8, 4, 2) << 16),
                       pack_eight(in, within + 16, 4, 2) | (pack_eight(in, within + 24, 4, 2) << 16));
#endif
#endif
    }
#else
    {
        const u32 block = column0 / 64;
        const u32 step0 = (column0 % 64) / 16;
#if defined(SEISMIC_ELEMENT_U_PLANE_CODES)
        {
            u32 words[4];
            for (u32 half_step = 0; half_step < 2; ++half_step) {
                u32 low = 0, high = 0;
                for (u32 slot = 0; slot < 4; ++slot) {
                    low |= fragment_code(source, seismic_words_value, b, row0, packets, block, step0 + half_step,
                                         lane, slot) << (8 * slot);
                    high |= fragment_code(source, seismic_words_value, b, row0, packets, block, step0 + half_step,
                                          lane, slot + 4) << (8 * slot);
                }
                words[2 * half_step] = low;
                words[2 * half_step + 1] = high;
            }
            u64 position = ((u64)block * 32 + lane) * 32 + 8 * step0;
            *reinterpret_cast<uint4 *>(tile_plane_byte(destination, seismic_words_value, b, row0, position,
                                                       SEISMIC_RESULT_0_PLANE_CODES_BYTES_PER_ROW,
                                                       SEISMIC_RESULT_0_PLANE_CODES_ROW_OFFSET)) =
                make_uint4(words[0], words[1], words[2], words[3]);
        }
#else
        {
            u64 position = ((u64)block * 32 + lane) * 16 + 4 * step0;
            *reinterpret_cast<uint2 *>(tile_plane_byte(destination, seismic_words_value, b, row0, position,
                                                       SEISMIC_RESULT_0_PLANE_CODES_LO_BYTES_PER_ROW,
                                                       SEISMIC_RESULT_0_PLANE_CODES_LO_ROW_OFFSET)) =
                make_uint2(fragment_slice(source, seismic_words_value, b, row0, packets, block, step0, lane, 0, 4),
                           fragment_slice(source, seismic_words_value, b, row0, packets, block, step0 + 1, lane, 0, 4));
        }
#if defined(SEISMIC_ELEMENT_U_PLANE_CODES_HI) && SEISMIC_ELEMENT_U_PLANE_CODES_HI_CODE_BITS == 1
        {
            u64 position = ((u64)block * 32 + lane) * 4 + step0;
            *reinterpret_cast<unsigned short *>(tile_plane_byte(destination, seismic_words_value, b, row0, position,
                                                                SEISMIC_RESULT_0_PLANE_CODES_HI_BYTES_PER_ROW,
                                                                SEISMIC_RESULT_0_PLANE_CODES_HI_ROW_OFFSET)) =
                (unsigned short)(fragment_slice(source, seismic_words_value, b, row0, packets, block, step0, lane, 4,
                                                1) |
                                 (fragment_slice(source, seismic_words_value, b, row0, packets, block, step0 + 1, lane,
                                                 4, 1) << 8));
        }
#elif defined(SEISMIC_ELEMENT_U_PLANE_CODES_HI)
        {
            u64 position = ((u64)block * 32 + lane) * 8 + 2 * step0;
            *reinterpret_cast<u32 *>(tile_plane_byte(destination, seismic_words_value, b, row0, position,
                                                     SEISMIC_RESULT_0_PLANE_CODES_HI_BYTES_PER_ROW,
                                                     SEISMIC_RESULT_0_PLANE_CODES_HI_ROW_OFFSET)) =
                fragment_slice(source, seismic_words_value, b, row0, packets, block, step0, lane, 4, 2) |
                (fragment_slice(source, seismic_words_value, b, row0, packets, block, step0 + 1, lane, 4, 2) << 16);
        }
#endif
#endif
    }
#endif

    if (within == 0 && lane < rows) {
        const u64 n = row0 + lane;
        u8 *row = row_base(destination, seismic_words_value, b, n);
        const u64 group = column0 / GROUP;
        const bool real = n < SEISMIC_DIM_N && occupied;
#if defined(SEISMIC_ELEMENT_U_PLANE_SCALES)
        u8 *scales =
            row + SEISMIC_RESULT_0_PLANE_SCALES_ROW_OFFSET + group * SEISMIC_ELEMENT_U_PLANE_SCALES_BYTES_PER_GROUP;
        if (real) write_scales(source_packet(source, seismic_words_value, b, n, group), scales);
        else zero_bytes(scales, SEISMIC_ELEMENT_U_PLANE_SCALES_BYTES_PER_GROUP);
#endif
        u8 *supers =
            row + SEISMIC_RESULT_0_PLANE_SUPERS_ROW_OFFSET + group * SEISMIC_ELEMENT_U_PLANE_SUPERS_BYTES_PER_GROUP;
        if (real) write_supers(source_packet(source, seismic_words_value, b, n, group), supers);
        else zero_bytes(supers, SEISMIC_ELEMENT_U_PLANE_SUPERS_BYTES_PER_GROUP);
    }

    if (column0 + TILE_COLUMNS == stored_columns && lane < rows) {
        u8 *row = row_base(destination, seismic_words_value, b, row0 + lane);
#if defined(SEISMIC_ELEMENT_U_PLANE_SCALES)
        zero_bytes(row + SEISMIC_RESULT_0_PLANE_SCALES_ROW_OFFSET + SEISMIC_RESULT_0_PLANE_SCALES_BYTES_PER_ROW,
                   SEISMIC_RESULT_0_PLANE_SUPERS_ROW_OFFSET - SEISMIC_RESULT_0_PLANE_SCALES_ROW_OFFSET -
                       SEISMIC_RESULT_0_PLANE_SCALES_BYTES_PER_ROW);
#endif
        zero_bytes(row + SEISMIC_RESULT_0_PLANE_SUPERS_ROW_OFFSET + SEISMIC_RESULT_0_PLANE_SUPERS_BYTES_PER_ROW,
                   SEISMIC_RESULT_0_ROW_STRIDE_BYTES - SEISMIC_RESULT_0_PLANE_SUPERS_ROW_OFFSET -
                       SEISMIC_RESULT_0_PLANE_SUPERS_BYTES_PER_ROW);
    }
}

#endif
