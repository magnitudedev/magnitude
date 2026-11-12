// The registered `repack` of an external GGUF source into rows16 storage, the
// Vulkan resident layout, one row tile's place at a time: the
// one placement `repack_weight` imports by and expert streaming's
// conversions write slots by (lib/streaming/window.glsl), so a slot holds
// exactly the bytes a resident import places.
//
// The conversion is data: its word form (`seismic_lang::repack::WordTable`,
// whose docs give the table's format), emitted as SEISMIC_CONVERSION_<C>_*,
// gives every 32-bit word of a source packet's record, the packet's storage
// group's share of each row plane, as the OR of pieces: runs of source bits
// or bits of values its postfix ops compute. The prefix renders it as code
// (SEISMIC_CONVERSION_<C>_UNIT, a unit of the code planes' words, and
// SEISMIC_CONVERSION_<C>_FIXED, the other words: the group roles' shares).
//
// A row tile is 16 rows x one unit of the span's columns
// (SEISMIC_CONVERSION_<C>_UNIT_COLUMNS: 32, or more where 32 columns of a
// code plane are no whole words). A unit divides its group, so a tile covers
// a whole packet or part of a larger one, and a code plane's share of a tile
// row is whole record words (a unit of the span); the tile at a group's
// first column also places that group's other record words.
// Stored-but-unoccupied bytes (plane alignment tails) are written as zero,
// so the result is fully defined.
//
// `REPACK_TILE(NAME, C, S, D)` defines `NAME_tile_columns()`, a tile's
// columns, and `NAME_place(packet, scale, row, column0)` for conversion `C`
// from source element `S` into tensor `D`'s rows16 rows: one row's tile of
// columns from `column0`, `packet` the address of the source
// packet holding that column (a planar source's, of its codes, and `scale`
// the address of the scale its tile holds; unread by any other), `row` the
// address of the stored row; and `NAME_place_shared(packet, scale, row,
// column0)`, the same placement from the packet at byte offset `packet` of
// the group memory (a copy of the source, the packet at a multiple of its
// alignment: the conversion's `_SHARED` code). Placing reads no other invocation's words, so any invocations may
// place any rows' tiles. The includer checks that `S` is external and `D`
// rows16 with no padding groups (`SEISMIC_ELEMENT_<U>_GROUP_MULTIPLE` 1):
// every stored tile holds source packets.
#include <seismic/element.glsl>

// The rows of a conversion's tile as the code computes with them: unsigned.
// `REPACK_TILE_ROWS_INT`, the same count as a signed literal for a
// preprocessor `#if` (glslang's divides by no unsigned one), is the Vulkan
// prefix's (`native::abi`), which states the conversion's `TILE_ROWS` once.
#define REPACK_TILE_ROWS uint(REPACK_TILE_ROWS_INT)

// A planar source's geometry, written once: every reader of one (the
// resident import's repack_weight.comp, the streaming windows) lays a
// matrix's bytes by it. A matrix of `[N, K]` is its codes plane (`N` rows of
// the source's packets per row, `ceil(K / group)`, each `packet - scale`
// code bytes) then its grid of scales (`ceil(N / tile rows)` rows of
// `ceil(K / tile columns)`), as the file stores them
// (`ExternalPacketLayout::packet_pieces`, `matrix_bytes`).
//
// `REPACK_PLANAR_SOURCE(PFX, S, N, K)` defines, for source element prefix `S`
// and the matrix extents `N`, `K`, the functions `PFX_row_packets()`,
// `PFX_grid_columns()`, `PFX_code_bytes()` (a packet's), `PFX_codes_bytes()`
// (the plane's), `PFX_matrix_bytes()`, and, from a matrix's first byte,
// `PFX_code_at(row, packet)` and `PFX_scale_at(row, packet)`. A source of no
// grid has no tile (read as 1x1) and is not asked.
#define REPACK_PLANAR_SOURCE(PFX, S, N, K) \
    uint64_t PFX##_row_packets() { \
        return (uint64_t(K) + uint64_t(S##_LOGICAL_GROUP) - 1ul) / uint64_t(S##_LOGICAL_GROUP); \
    } \
    uint64_t PFX##_grid_columns() { \
        const uint64_t columns = max(uint64_t(S##_GRID_TILE_COLUMNS), 1ul); \
        return (uint64_t(K) + columns - 1ul) / columns; \
    } \
    uint64_t PFX##_code_bytes() { \
        return uint64_t(S##_PACKET_SIZE) - uint64_t(S##_GRID_SCALE_BYTES); \
    } \
    uint64_t PFX##_codes_bytes() { \
        return uint64_t(N) * PFX##_row_packets() * PFX##_code_bytes(); \
    } \
    uint64_t PFX##_matrix_bytes() { \
        const uint64_t rows = max(uint64_t(S##_GRID_TILE_ROWS), 1ul); \
        return PFX##_codes_bytes() + (uint64_t(N) + rows - 1ul) / rows * PFX##_grid_columns() \
            * uint64_t(S##_GRID_SCALE_BYTES); \
    } \
    uint64_t PFX##_code_at(uint64_t row, uint64_t packet) { \
        return (row * PFX##_row_packets() + packet) * PFX##_code_bytes(); \
    } \
    uint64_t PFX##_scale_at(uint64_t row, uint64_t packet) { \
        const uint64_t rows = max(uint64_t(S##_GRID_TILE_ROWS), 1ul), columns = max(uint64_t(S##_GRID_TILE_COLUMNS), 1ul); \
        return PFX##_codes_bytes() \
            + ((row / rows) * PFX##_grid_columns() + packet * uint64_t(S##_LOGICAL_GROUP) / columns) \
            * uint64_t(S##_GRID_SCALE_BYTES); \
    }

// What a conversion reads its source by: a planar source (`S_GRID_PLANAR`)
// its packet's codes and its tile's scale, any other its packet.
#define REPACK_PLANAR_1(planar, plain) planar
#define REPACK_PLANAR_0(planar, plain) plain
#define REPACK_PLANAR_(flag) REPACK_PLANAR_##flag
#define REPACK_PLANAR(flag) REPACK_PLANAR_(flag)

// `count` bytes at `place`, zero.
void repack_zero_bytes(uint64_t place, uint64_t count) {
    for (uint64_t at = 0ul; at < count; ++at)
        element_u8_put(place + at, 0u);
}

// `bytes` bytes of the record words `words` from word `first` at `place`, in
// stores of `unit` bytes (the record's alignments,
// SEISMIC_CONVERSION_<C>_*_ALIGNMENT).
// C, S and D are complete ABI prefixes; paste their suffixes here so
// conversion presence flags are not expanded while forwarding a prefix.
#define REPACK_TILE(NAME, C, S, D) \
    void NAME##_store(uint64_t place, uint words[C##_WORDS], uint first, uint bytes, uint unit) { \
        if (unit == 16u) { \
            [[unroll]] for (uint at = 0u; at < bytes / 4u; at += 4u) \
                element_uvec4_put(place + 4u * at, \
                    uvec4(words[first + at], words[first + at + 1u], words[first + at + 2u], words[first + at + 3u])); \
        } else if (unit == 8u) { \
            [[unroll]] for (uint at = 0u; at < bytes / 4u; at += 2u) \
                element_uvec2_put(place + 4u * at, uvec2(words[first + at], words[first + at + 1u])); \
        } else if (unit == 4u) { \
            [[unroll]] for (uint at = 0u; at < bytes / 4u; ++at) \
                element_u32_put(place + 4u * at, words[first + at]); \
        } else if (unit == 2u) { \
            [[unroll]] for (uint half_word = 0u; half_word < bytes / 2u; ++half_word) \
                element_u16_put(place + 2u * half_word, words[first + half_word / 2u] >> (16u * (half_word % 2u))); \
        } else { \
            [[unroll]] for (uint byte = 0u; byte < bytes; ++byte) \
                element_u8_put(place + byte, words[first + byte / 4u] >> (8u * (byte % 4u))); \
        } \
    } \
    uint NAME##_tile_columns() { \
        return uint(C##_UNIT_COLUMNS); \
    } \
    /* The unit's code words of the tile at `column0`, stored into row `row`. */ \
    void NAME##_place_unit(uint64_t row, uint column0, uint words[C##_WORDS]) { \
        const uint unit = column0 / NAME##_tile_columns(); \
        NAME##_store(row + D##_ROW_CODES + uint64_t(unit) * (4u * C##_CODES_UNIT_WORDS), \
            words, 0u, 4u * C##_CODES_UNIT_WORDS, C##_CODES_ALIGNMENT); \
        NAME##_store(row + D##_ROW_HIGH + uint64_t(unit) * (4u * C##_HIGH_UNIT_WORDS), \
            words, C##_CODES_UNIT_WORDS, 4u * C##_HIGH_UNIT_WORDS, C##_HIGH_ALIGNMENT); \
    } \
    /* The fixed words of the group at `column0`, stored into row `row`. */ \
    void NAME##_place_fixed(uint64_t row, uint column0, uint words[C##_WORDS]) { \
        const uint64_t group = column0 / uint(S##_LOGICAL_GROUP); \
        NAME##_store(row + D##_ROW_SCALES + group * C##_SCALES_BYTES, words, \
            C##_SCALES_WORD, C##_SCALES_BYTES, C##_SCALES_ALIGNMENT); \
        NAME##_store(row + D##_ROW_SUPERS + group * C##_SUPERS_BYTES, words, \
            C##_SUPERS_WORD, C##_SUPERS_BYTES, C##_SUPERS_ALIGNMENT); \
    } \
    /* The row's alignment tails, zero, by its last tile. */ \
    void NAME##_place_tails(uint64_t row, uint column0) { \
        if (uint64_t(column0 + NAME##_tile_columns()) == uint64_t(D##_ROW_GROUPS) * uint(S##_LOGICAL_GROUP)) { \
            repack_zero_bytes(row + D##_ROW_CODES + D##_ROW_CODES_BYTES, D##_ROW_CODES_TAIL); \
            repack_zero_bytes(row + D##_ROW_HIGH + D##_ROW_HIGH_BYTES, D##_ROW_HIGH_TAIL); \
            repack_zero_bytes(row + D##_ROW_SCALES + D##_ROW_SCALES_BYTES, \
                D##_ROW_SCALES_TAIL); \
            repack_zero_bytes(row + D##_ROW_SUPERS + D##_ROW_SUPERS_BYTES, \
                D##_ROW_SUPERS_TAIL); \
        } \
    } \
    void NAME##_place(uint64_t packet, uint64_t scale, uint64_t row, uint column0) { \
        const uint within = column0 % uint(S##_LOGICAL_GROUP); \
        uint words[C##_WORDS]; \
        REPACK_PLANAR(S##_GRID_PLANAR)(C##_UNIT(packet, scale, within / NAME##_tile_columns(), words), \
            C##_UNIT(packet, within / NAME##_tile_columns(), words)); \
        NAME##_place_unit(row, column0, words); \
        if (within == 0u) { \
            REPACK_PLANAR(S##_GRID_PLANAR)(C##_FIXED(packet, scale, words), C##_FIXED(packet, words)); \
            NAME##_place_fixed(row, column0, words); \
        } \
        NAME##_place_tails(row, column0); \
    } \
    void NAME##_place_shared(uint packet, uint scale, uint64_t row, uint column0) { \
        const uint within = column0 % uint(S##_LOGICAL_GROUP); \
        uint words[C##_WORDS]; \
        REPACK_PLANAR(S##_GRID_PLANAR)(C##_UNIT_SHARED(packet, scale, within / NAME##_tile_columns(), words), \
            C##_UNIT_SHARED(packet, within / NAME##_tile_columns(), words)); \
        NAME##_place_unit(row, column0, words); \
        if (within == 0u) { \
            REPACK_PLANAR(S##_GRID_PLANAR)(C##_FIXED_SHARED(packet, scale, words), C##_FIXED_SHARED(packet, words)); \
            NAME##_place_fixed(row, column0, words); \
        } \
        NAME##_place_tails(row, column0); \
    }
