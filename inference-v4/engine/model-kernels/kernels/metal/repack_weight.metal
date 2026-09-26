// Exact GGUF -> resident conversion (the Seismic registry's registered
// `repack`), one kernel per (source format, target layout) selected by the
// element macros: E is the GGUF source, U the resident (representation,
// layout). The entry sees every weight as a [B, N, K] view.
//
// Each threadgroup (32 threads) converts one row tile: 16 rows x 32 columns.
// Every format's group (256 or 32) is a multiple of 32, so a tile covers
// whole q8 packets or one eighth of a k-quant packet; the tile at a group's
// first column also writes that group's coefficients. Code stores are 4-,
// 8- or 16-byte vectors. Stored-but-unoccupied bytes (mma16 padding rows and
// groups, plane alignment tails, packet gaps) are written as zero, so the
// result is fully defined.

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

#define GROUP ((uint)SEISMIC_ELEMENT_E_LOGICAL_GROUP)
#define TILE_ROWS 16u
#define TILE_COLUMNS 32u

// ---- GGUF source packet readers ------------------------------------------

// Raw code of position `p` (0 <= p < GROUP) of one source packet.
inline uint source_code(device const uchar *in, uint p) {
#if defined(REPACK_KQUANT) && REPACK_CODE_BITS == 4u
    return (in[16 + (p / 64) * 32 + p % 32] >> ((p % 64 / 32) * 4)) & 15u;
#elif defined(REPACK_KQUANT)
    uint low = (in[48 + (p / 64) * 32 + p % 32] >> ((p % 64 / 32) * 4)) & 15u;
    uint high = (in[16 + p % 32] >> (p / 32)) & 1u;
    return low | (high << 4);
#elif defined(REPACK_Q6K)
    uint low = (in[(p / 128) * 64 + p % 64] >> ((p % 128 / 64) * 4)) & 15u;
    uint high = (in[128 + (p / 128) * 32 + p % 32] >> ((p % 128 / 32) * 2)) & 3u;
    return low | (high << 4);
#elif defined(REPACK_Q8)
    return uint(in[2 + p]);
#else
    return (in[8 + (p / 32) * 16 + p % 16] >> ((p % 32 / 16) * 4)) & 15u;
#endif
}

#if defined(REPACK_KQUANT)
// Six-bit local coefficient `field` (0 scale, 1 minimum) of sub-block `j`.
inline uint kquant_local(device const uchar *in, uint j, uint field) {
    uint index = j % 4;
    uint low = uint(in[4 + field * 4 + index]);
    uint high = uint(in[12 + index]);
    return j < 4 ? (low & 63u) : (((high >> (4 * field)) & 15u) | ((low >> 6) << 4));
}

// Word `w` of the packed local coefficients: 16 six-bit fields, field
// `2j + f` of sub-block j at bit 12j + 6f (the registry `coefficients` /
// `scales` plane).
inline uint kquant_locals_word(device const uchar *in, uint w) {
    uint value = 0;
    for (uint field = 0; field < 16; ++field) {
        int shift = int(field * 6) - int(w * 32);
        if (shift <= -6 || shift >= 32) continue;
        uint local = kquant_local(in, field / 2, field % 2);
        value |= shift >= 0 ? local << uint(shift) : local >> uint(-shift);
    }
    return value;
}
#endif

#if defined(REPACK_IQ4)
// Resident f32 scale of IQ4_XS sub-block `j` (32 values).
inline float iq4_scale(device const uchar *in, uint j) {
    float base = float(as_type<half>(ushort(uint(in[0]) | (uint(in[1]) << 8))));
    uint low = (uint(in[4 + j / 2]) >> ((j % 2) * 4)) & 15u;
    uint high = ((uint(in[2]) | (uint(in[3]) << 8)) >> (2 * j)) & 3u;
    return base * float(int(low | (high << 4)) - 32);
}
#endif

// The per-group coefficient planes of one packet: `scales` (the packed
// locals) and `supers` (the per-group factors), in registry byte order.
inline void write_scales(device const uchar *in, device uchar *out) {
#if defined(REPACK_KQUANT)
    for (uint w = 0; w < 3; ++w)
        reinterpret_cast<device uint *>(out)[w] = kquant_locals_word(in, w);
#elif defined(REPACK_Q6K)
    for (uint w = 0; w < 4; ++w)
        reinterpret_cast<device uint *>(out)[w] = uint(in[192 + 4 * w]) | (uint(in[193 + 4 * w]) << 8) |
                                                   (uint(in[194 + 4 * w]) << 16) | (uint(in[195 + 4 * w]) << 24);
#endif
}

inline void write_supers(device const uchar *in, device uchar *out) {
#if defined(REPACK_KQUANT)
    reinterpret_cast<device uint *>(out)[0] =
        uint(in[0]) | (uint(in[1]) << 8) | (uint(in[2]) << 16) | (uint(in[3]) << 24);
#elif defined(REPACK_Q6K)
    reinterpret_cast<device ushort *>(out)[0] = ushort(uint(in[208]) | (uint(in[209]) << 8));
#elif defined(REPACK_Q8)
    reinterpret_cast<device ushort *>(out)[0] = ushort(uint(in[0]) | (uint(in[1]) << 8));
#else
    for (uint j = 0; j < 8; ++j)
        reinterpret_cast<device float *>(out)[j] = iq4_scale(in, j);
#endif
}

// Eight consecutive `bits`-wide slices (code >> shift) of codes
// `first .. first + 8`, packed little-endian.
inline uint pack_eight(device const uchar *in, uint first, uint shift, uint bits) {
    uint value = 0;
    for (uint i = 0; i < 8; ++i)
        value |= ((source_code(in, first + i) >> shift) & ((1u << bits) - 1u)) << (i * bits);
    return value;
}

// Four consecutive eight-bit codes `first .. first + 4`.
inline uint pack_four_bytes(device const uchar *in, uint first) {
    uint value = 0;
    for (uint i = 0; i < 4; ++i) value |= source_code(in, first + i) << (8 * i);
    return value;
}

inline void zero_bytes(device uchar *out, ulong count) {
    for (ulong i = 0; i < count; ++i) out[i] = 0;
}

// Source packet `packet` of row `n` of matrix `b`.
inline device const uchar *source_packet(device const uchar *source, constant ulong *seismic_words,
                                         ulong b, ulong n, ulong packet) {
    return source + (b * SEISMIC_SOURCE_STRIDE_0 + n * SEISMIC_SOURCE_STRIDE_1 + packet * SEISMIC_SOURCE_STRIDE_2) *
                        SEISMIC_ELEMENT_E_PACKET_SIZE;
}

#if defined(SEISMIC_ELEMENT_U_LAYOUT_PACKET)

// ---- packet: planes interleaved per packet --------------------------------

inline device uchar *destination_packet(device uchar *destination, constant ulong *seismic_words,
                                        ulong b, ulong n, ulong packet) {
    return destination + (b * SEISMIC_RESULT_0_STRIDE_0 + n * SEISMIC_RESULT_0_STRIDE_1 +
                          packet * SEISMIC_RESULT_0_STRIDE_2) * SEISMIC_ELEMENT_U_PACKET_SIZE;
}

inline bool packet_plane_byte(uint byte) {
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

kernel void repack_weight(
    device const uchar *source [[buffer(SEISMIC_BUFFER_SOURCE)]],
    device uchar *destination [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 tile [[threadgroup_position_in_grid]],
    uint3 thread_position [[thread_position_in_threadgroup]]) {
    const uint lane = thread_position.x;
    const ulong b = tile.z;
    const ulong row0 = ulong(tile.y) * TILE_ROWS;
    const uint column0 = tile.x * TILE_COLUMNS;
    const ulong packets = (SEISMIC_DIM_K + GROUP - 1) / GROUP;
    if (column0 >= packets * GROUP) return;
    const uint packet = column0 / GROUP;
    const uint within = column0 % GROUP;
    // The tile's codes are `bits` words of the packed `words` plane per row.
    const uint code_words = REPACK_CODE_BITS;
    for (uint item = lane; item < TILE_ROWS * code_words; item += TILE_COLUMNS) {
        ulong n = row0 + item / code_words;
        if (n >= SEISMIC_DIM_N) continue;
        device const uchar *in = source_packet(source, seismic_words, b, n, packet);
        uint word = within / 32 * code_words + item % code_words;
        uint first = (word * 32) / REPACK_CODE_BITS;
        uint last = (word * 32 + 31) / REPACK_CODE_BITS;
        uint value = 0;
        for (uint code = first; code <= last; ++code) {
            int shift = int(code * REPACK_CODE_BITS) - int(word * 32);
            uint raw = source_code(in, code);
            value |= shift >= 0 ? raw << uint(shift) : raw >> uint(-shift);
        }
        device uchar *out = destination_packet(destination, seismic_words, b, n, packet);
        reinterpret_cast<device uint *>(out + SEISMIC_ELEMENT_U_PLANE_0_OFFSET)[word] = value;
    }
    if (within != 0 || lane >= TILE_ROWS || row0 + lane >= SEISMIC_DIM_N) return;
    const ulong n = row0 + lane;
    device const uchar *in = source_packet(source, seismic_words, b, n, packet);
    device uchar *out = destination_packet(destination, seismic_words, b, n, packet);
#if defined(REPACK_KQUANT) || defined(REPACK_Q6K)
    write_scales(in, out + SEISMIC_ELEMENT_U_PLANE_1_OFFSET);
    write_supers(in, out + SEISMIC_ELEMENT_U_PLANE_2_OFFSET);
#else
    write_supers(in, out + SEISMIC_ELEMENT_U_PLANE_1_OFFSET);
#endif
    for (uint byte = 0; byte < SEISMIC_ELEMENT_U_PACKET_SIZE; ++byte)
        if (!packet_plane_byte(byte)) out[byte] = 0;
}

#else

// ---- rows16 / mma16: row planes ---------------------------------------------

// Start of stored row `n` of matrix `b`.
inline device uchar *row_base(device uchar *destination, constant ulong *seismic_words, ulong b, ulong n) {
    return destination + (b * SEISMIC_RESULT_0_STRIDE_0 + n * SEISMIC_RESULT_0_STRIDE_1) *
                             SEISMIC_RESULT_0_ROW_STRIDE_BYTES;
}

#if defined(SEISMIC_ELEMENT_U_LAYOUT_MMA16)
// Code of fragment slot `slot` of k16 step `step` (within k-block `block`)
// owned by `lane`: (row lane/4 + 8 (slot % 2), column 64 block + 16 step +
// 2 (lane % 4) + 8 ((slot % 4) / 2) + slot / 4). Zero beyond the matrix's
// rows or the source packets (tile padding).
inline uint fragment_code(device const uchar *source, constant ulong *seismic_words, ulong b, ulong row0,
                          ulong packets, uint block, uint step, uint lane, uint slot) {
    ulong n = row0 + lane / 4 + 8 * (slot % 2);
    ulong column = ulong(block) * 64 + step * 16 + 2 * (lane % 4) + 8 * ((slot % 4) / 2) + slot / 4;
    if (n >= SEISMIC_DIM_N || column >= packets * GROUP) return 0;
    return source_code(source_packet(source, seismic_words, b, n, column / GROUP), uint(column % GROUP));
}

// Eight `bits`-wide slices of one lane's slots of one step.
inline uint fragment_slice(device const uchar *source, constant ulong *seismic_words, ulong b, ulong row0,
                           ulong packets, uint block, uint step, uint lane, uint shift, uint bits) {
    uint value = 0;
    for (uint slot = 0; slot < 8; ++slot)
        value |= ((fragment_code(source, seismic_words, b, row0, packets, block, step, lane, slot) >> shift) &
                  ((1u << bits) - 1u)) << (slot * bits);
    return value;
}

// Destination of byte `position` of a tile's code-plane sequence V: the
// rows' plane payloads (`row_bytes` each) concatenated in row order.
inline device uchar *tile_plane_byte(device uchar *destination, constant ulong *seismic_words, ulong b,
                                     ulong row0, ulong position, ulong row_bytes, ulong offset) {
    return row_base(destination, seismic_words, b, row0 + position / row_bytes) + offset + position % row_bytes;
}
#endif

kernel void repack_weight(
    device const uchar *source [[buffer(SEISMIC_BUFFER_SOURCE)]],
    device uchar *destination [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 tile [[threadgroup_position_in_grid]],
    uint3 thread_position [[thread_position_in_threadgroup]]) {
    const uint lane = thread_position.x;
    const ulong b = tile.z;
    const ulong row0 = ulong(tile.y) * TILE_ROWS;
    const uint column0 = tile.x * TILE_COLUMNS;
    const ulong packets = (SEISMIC_DIM_K + GROUP - 1) / GROUP;
    const ulong stored_columns = SEISMIC_RESULT_0_ROW_GROUPS * GROUP;
    if (column0 >= stored_columns) return;
    const bool occupied = column0 < packets * GROUP;
    const uint within = column0 % GROUP;
#if defined(SEISMIC_ELEMENT_U_LAYOUT_MMA16)
    // Every row of a tile is stored; rows beyond N are zero padding.
    const ulong rows = TILE_ROWS;
#else
    const ulong rows = min(ulong(TILE_ROWS), SEISMIC_DIM_N - row0);
#endif

#if defined(SEISMIC_ELEMENT_U_LAYOUT_ROWS16)
    // Code planes: the code of column c sits at bit `c * bits` of the plane.
    if (lane < rows) {
        const ulong n = row0 + lane;
        device const uchar *in = source_packet(source, seismic_words, b, n, column0 / GROUP);
        device uchar *row = row_base(destination, seismic_words, b, n);
#if defined(SEISMIC_ELEMENT_U_PLANE_CODES)
        device uint4 *codes = reinterpret_cast<device uint4 *>(row + SEISMIC_RESULT_0_PLANE_CODES_ROW_OFFSET + column0);
        codes[0] = uint4(pack_four_bytes(in, within), pack_four_bytes(in, within + 4),
                         pack_four_bytes(in, within + 8), pack_four_bytes(in, within + 12));
        codes[1] = uint4(pack_four_bytes(in, within + 16), pack_four_bytes(in, within + 20),
                         pack_four_bytes(in, within + 24), pack_four_bytes(in, within + 28));
#else
        *reinterpret_cast<device uint4 *>(row + SEISMIC_RESULT_0_PLANE_CODES_LO_ROW_OFFSET + column0 / 2) =
            uint4(pack_eight(in, within, 0, 4), pack_eight(in, within + 8, 0, 4),
                  pack_eight(in, within + 16, 0, 4), pack_eight(in, within + 24, 0, 4));
#if defined(SEISMIC_ELEMENT_U_PLANE_CODES_HI) && SEISMIC_ELEMENT_U_PLANE_CODES_HI_CODE_BITS == 1
        *reinterpret_cast<device uint *>(row + SEISMIC_RESULT_0_PLANE_CODES_HI_ROW_OFFSET + column0 / 8) =
            pack_eight(in, within, 4, 1) | (pack_eight(in, within + 8, 4, 1) << 8) |
            (pack_eight(in, within + 16, 4, 1) << 16) | (pack_eight(in, within + 24, 4, 1) << 24);
#elif defined(SEISMIC_ELEMENT_U_PLANE_CODES_HI)
        *reinterpret_cast<device uint2 *>(row + SEISMIC_RESULT_0_PLANE_CODES_HI_ROW_OFFSET + column0 / 4) =
            uint2(pack_eight(in, within, 4, 2) | (pack_eight(in, within + 8, 4, 2) << 16),
                  pack_eight(in, within + 16, 4, 2) | (pack_eight(in, within + 24, 4, 2) << 16));
#endif
#endif
    }
#else
    // mma16: lane l owns, per 64-column k-block, 4 * bits bytes of each code
    // plane's tile sequence V (registry `PackedRowLayout::code_bit`); this
    // tile is k16 steps `step0` and `step0 + 1` of k-block `block`.
    {
        const uint block = column0 / 64;
        const uint step0 = (column0 % 64) / 16;
#if defined(SEISMIC_ELEMENT_U_PLANE_CODES)
        {
            uint4 value;
            for (uint half_step = 0; half_step < 2; ++half_step) {
                uint low = 0, high = 0;
                for (uint slot = 0; slot < 4; ++slot) {
                    low |= fragment_code(source, seismic_words, b, row0, packets, block, step0 + half_step, lane, slot)
                           << (8 * slot);
                    high |= fragment_code(source, seismic_words, b, row0, packets, block, step0 + half_step, lane,
                                          slot + 4) << (8 * slot);
                }
                value[2 * half_step] = low;
                value[2 * half_step + 1] = high;
            }
            ulong position = (ulong(block) * 32 + lane) * 32 + 8 * step0;
            *reinterpret_cast<device uint4 *>(tile_plane_byte(destination, seismic_words, b, row0, position,
                                                              SEISMIC_RESULT_0_PLANE_CODES_BYTES_PER_ROW,
                                                              SEISMIC_RESULT_0_PLANE_CODES_ROW_OFFSET)) = value;
        }
#else
        {
            ulong position = (ulong(block) * 32 + lane) * 16 + 4 * step0;
            *reinterpret_cast<device uint2 *>(tile_plane_byte(destination, seismic_words, b, row0, position,
                                                              SEISMIC_RESULT_0_PLANE_CODES_LO_BYTES_PER_ROW,
                                                              SEISMIC_RESULT_0_PLANE_CODES_LO_ROW_OFFSET)) =
                uint2(fragment_slice(source, seismic_words, b, row0, packets, block, step0, lane, 0, 4),
                      fragment_slice(source, seismic_words, b, row0, packets, block, step0 + 1, lane, 0, 4));
        }
#if defined(SEISMIC_ELEMENT_U_PLANE_CODES_HI) && SEISMIC_ELEMENT_U_PLANE_CODES_HI_CODE_BITS == 1
        {
            ulong position = (ulong(block) * 32 + lane) * 4 + step0;
            *reinterpret_cast<device ushort *>(tile_plane_byte(destination, seismic_words, b, row0, position,
                                                               SEISMIC_RESULT_0_PLANE_CODES_HI_BYTES_PER_ROW,
                                                               SEISMIC_RESULT_0_PLANE_CODES_HI_ROW_OFFSET)) =
                ushort(fragment_slice(source, seismic_words, b, row0, packets, block, step0, lane, 4, 1) |
                       (fragment_slice(source, seismic_words, b, row0, packets, block, step0 + 1, lane, 4, 1) << 8));
        }
#elif defined(SEISMIC_ELEMENT_U_PLANE_CODES_HI)
        {
            ulong position = (ulong(block) * 32 + lane) * 8 + 2 * step0;
            *reinterpret_cast<device uint *>(tile_plane_byte(destination, seismic_words, b, row0, position,
                                                             SEISMIC_RESULT_0_PLANE_CODES_HI_BYTES_PER_ROW,
                                                             SEISMIC_RESULT_0_PLANE_CODES_HI_ROW_OFFSET)) =
                fragment_slice(source, seismic_words, b, row0, packets, block, step0, lane, 4, 2) |
                (fragment_slice(source, seismic_words, b, row0, packets, block, step0 + 1, lane, 4, 2) << 16);
        }
#endif
#endif
    }
#endif

    // Coefficient planes, once per group (at its first tile), per row.
    if (within == 0 && lane < rows) {
        const ulong n = row0 + lane;
        device uchar *row = row_base(destination, seismic_words, b, n);
        const ulong group = column0 / GROUP;
        const bool real = n < SEISMIC_DIM_N && occupied;
#if defined(SEISMIC_ELEMENT_U_PLANE_SCALES)
        device uchar *scales =
            row + SEISMIC_RESULT_0_PLANE_SCALES_ROW_OFFSET + group * SEISMIC_ELEMENT_U_PLANE_SCALES_BYTES_PER_GROUP;
        if (real) write_scales(source_packet(source, seismic_words, b, n, group), scales);
        else zero_bytes(scales, SEISMIC_ELEMENT_U_PLANE_SCALES_BYTES_PER_GROUP);
#endif
        device uchar *supers =
            row + SEISMIC_RESULT_0_PLANE_SUPERS_ROW_OFFSET + group * SEISMIC_ELEMENT_U_PLANE_SUPERS_BYTES_PER_GROUP;
        if (real) write_supers(source_packet(source, seismic_words, b, n, group), supers);
        else zero_bytes(supers, SEISMIC_ELEMENT_U_PLANE_SUPERS_BYTES_PER_GROUP);
    }

    // Alignment tails of the coefficient planes and of the row, at the row's
    // last stored tile. Code planes are whole multiples of 16 bytes.
    if (column0 + TILE_COLUMNS == stored_columns && lane < rows) {
        device uchar *row = row_base(destination, seismic_words, b, row0 + lane);
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
