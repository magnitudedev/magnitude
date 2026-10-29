// Exact GGUF -> resident conversion (the Seismic registry's registered
// `repack`): E is the GGUF source, U the resident (representation, layout).
// The entry sees every weight as a [B, N, K] view.
//
// The conversion is data: its word form (`seismic_lang::repack::WordTable`,
// whose docs give the table's format) gives every 32-bit word of a source
// packet's record as the OR of pieces, runs of source bits or bits of values
// its postfix ops compute. The record is U's packet form (`packet`), or the
// packet's storage group's share of each row plane (row layouts). The prefix
// renders it as code: SEISMIC_CONVERSION_E_U_UNIT, a unit of the code
// planes' words, and SEISMIC_CONVERSION_E_U_FIXED, the other words, each
// staging the source bytes it reads through this kernel's `Stage` (the fixed
// words' loaded apart, by SEISMIC_CONVERSION_E_U_FIXED_LOAD); the rest of
// this kernel places records in U's layout.
//
// Each block, one warp, places one row tile: 16 rows x 32 columns, a lane
// per row (the other 16 lanes help stage and store), so the warp runs one
// stream of record words (only their data differ). Every group is a
// multiple of 32 columns, so a tile is one unit of the code planes
// (`WordSpan`): its words are its class's representative's, their runs moved
// by the unit's displacements. The tile at a group's first column also
// places the group's other record words. Stored-but-unoccupied bytes (mma16
// padding rows and groups, plane alignment tails, packet gaps) are written
// as zero, so the result is fully defined.

#if !defined(SEISMIC_ELEMENT_E_KIND_EXTERNAL) || !defined(SEISMIC_ELEMENT_U_KIND_PACKED)
#error "repack_weight converts an external GGUF source into packed resident storage"
#endif

#if !defined(SEISMIC_CONVERSION_E_U)
#error "repack_weight binding is not a registered GGUF-to-resident conversion"
#endif

#if !defined(SEISMIC_ELEMENT_U_LAYOUT_PACKET) && !defined(SEISMIC_ELEMENT_U_LAYOUT_ROWS16) && \
    !defined(SEISMIC_ELEMENT_U_LAYOUT_MMA16)
#error "repack_weight writes packet, rows16 or mma16 storage"
#endif

typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

#define GROUP ((u32)SEISMIC_ELEMENT_E_LOGICAL_GROUP)
// The tile's rows and the warp's lanes, as the conversion's tile plans deal
// staged units to them (`SEISMIC_CONVERSION_E_U_TILE_ROWS`, `_LANES`).
#define TILE_ROWS ((u32)SEISMIC_CONVERSION_E_U_TILE_ROWS)
#define LANES ((u32)SEISMIC_CONVERSION_E_U_LANES)
#define TILE_COLUMNS 32u
#define UNITS (GROUP / TILE_COLUMNS)
static_assert(GROUP % TILE_COLUMNS == 0, "a tile lies in one group");

// A unit's code bytes, the codes' then the high codes' (`WordSpan`), and
// their record words: a unit is whole words, or part of one (a grid entry's
// high part).
#define CODES_UNIT_BYTES SEISMIC_CONVERSION_E_U_CODES_UNIT_BYTES
#define HIGH_UNIT_BYTES SEISMIC_CONVERSION_E_U_HIGH_UNIT_BYTES
#define CODES_WORDS ((CODES_UNIT_BYTES + 3u) / 4u)
#define HIGH_WORDS ((HIGH_UNIT_BYTES + 3u) / 4u)
#define UNIT_WORDS (CODES_WORDS + HIGH_WORDS)
#define WORDS SEISMIC_CONVERSION_E_U_WORDS

// `words` words padded to an odd number (each lane's row of a shared array of
// rows starts in its own bank).
__host__ __device__ constexpr u32 odd_words(u32 words) { return words % 2 == 0 ? words + 1 : words; }

// A staged row's words (`SEISMIC_CONVERSION_E_U_STAGE_WORDS`), padded.
#define STAGE_WORDS odd_words((u32)SEISMIC_CONVERSION_E_U_STAGE_WORDS)

// A lane's loaded units of the fixed words (an array's length: at least 1).
#define FIXED_LOADS (SEISMIC_CONVERSION_E_U_FIXED_LOADS > 0 ? SEISMIC_CONVERSION_E_U_FIXED_LOADS : 1)

// Source packet `packet` of row `n` of matrix `b`.
__device__ __forceinline__ const u8 *source_packet(const u8 *source, const seismic_words_t &seismic_words_value, u64 b,
                                                   u64 n, u64 packet) {
    return source + (b * SEISMIC_SOURCE_STRIDE_0 + n * SEISMIC_SOURCE_STRIDE_1 + packet * SEISMIC_SOURCE_STRIDE_2) *
                        SEISMIC_ELEMENT_E_PACKET_SIZE;
}

// A warp's stage of its tile's rows' source bytes, which the conversion's
// code (SEISMIC_CONVERSION_E_U_UNIT, _FIXED) stages segment by segment of
// its tile plan into `rows`, then reads at `row()`: rows `real` hold source
// packets, row `r`'s at `source + r * row_bytes`.
struct Stage {
    const u8 *source;
    u64 row_bytes;
    u32 real, lane;
    u32 *rows;

    // Lane `lane`'s staged unit `index` of segment bytes [first, first +
    // length) of the tile's rows (copy units of `unit` bytes, the rows'
    // units dealt to the lanes in turn): its row and unit within the row.
    __device__ __forceinline__ void locate(u32 length, u32 unit, u32 index, u32 &r, u32 &k) const {
        const u32 count = length / unit, at = lane + LANES * index;
        r = at / count;
        k = at % count;
    }

    // That unit's bytes of the source (moved by `moved` bytes; zero where its
    // row holds no source packet). Every argument but `moved` is a
    // constant; the generated code loads a batch of units before storing any.
    // Its address is formed whether or not its row is real, so only the
    // access is conditional: NVRTC then predicates each access in place, and
    // a batch's loads stay in flight together (with the address under the
    // condition it branches around every access, issuing one load at a time
    // between its stores: measured against A7, 1.09x the time of the
    // iq3_s and q3_k packet conversions then, into q6k).
    __device__ __forceinline__ u32 load(u32 first, u32 length, u32 unit, int moved, u32 index) const {
        u32 r, k;
        locate(length, unit, index, r, k);
        const u8 *from = source + r * row_bytes + moved + first + k * unit;
        u32 value = 0;
        if (r < real) {
            if (unit == 4) value = *reinterpret_cast<const u32 *>(from);
            else if (unit == 2) value = *reinterpret_cast<const u16 *>(from);
            else value = *from;
        }
        return value;
    }

    // ... stored at staged byte `position` of its row's stage (its address
    // too formed unconditionally).
    __device__ __forceinline__ void store(u32 value, u32 length, u32 position, u32 unit, u32 index) const {
        u32 r, k;
        locate(length, unit, index, r, k);
        u8 *to = reinterpret_cast<u8 *>(rows) + r * 4 * STAGE_WORDS + position + k * unit;
        if (r < real) {
            if (unit == 4) *reinterpret_cast<u32 *>(to) = value;
            else if (unit == 2) *reinterpret_cast<u16 *>(to) = u16(value);
            else *to = u8(value);
        }
    }

    // Row `lane`'s staged bytes, once every lane has staged its share
    // (lanes past the tile's rows read a row's, and place nothing).
    __device__ __forceinline__ const u32 *row() const {
        __syncwarp();
        return rows + lane % TILE_ROWS * STAGE_WORDS;
    }
};

// `bytes` bytes of `words` (little-endian) at `out`, in stores of `unit`
// bytes (a power of two, to 16, that `bytes` and `out` keep: the record's
// alignments, `SEISMIC_CONVERSION_E_U_*_ALIGNMENT`); `bytes` and `unit`
// constants, so every branch but one folds away.
__device__ __forceinline__ void store(u8 *out, const u32 *words, u32 bytes, u32 unit) {
#pragma unroll
    for (u32 at = 0; at < bytes; at += unit) {
        const u32 word = at / 4;
        if (unit == 16) {
            *reinterpret_cast<uint4 *>(out + at) = make_uint4(words[word], words[word + 1], words[word + 2], words[word + 3]);
        } else if (unit == 8) {
            *reinterpret_cast<uint2 *>(out + at) = make_uint2(words[word], words[word + 1]);
        } else if (unit == 4) {
            *reinterpret_cast<u32 *>(out + at) = words[word];
        } else if (unit == 2) {
            *reinterpret_cast<u16 *>(out + at) = u16(words[word] >> (at % 4 * 8));
        } else {
            out[at] = u8(words[word] >> (at % 4 * 8));
        }
    }
}

// A warp's tile: rows row0.. of matrix b at source packet `packet`
// (`occupied` when the tile holds source columns), unit `unit` of the
// group, staging in `stage`. Lane l computes row l's words.
struct Tile {
    const u8 *source;
    const seismic_words_t &seismic_words_value;
    u64 b, row0;
    u32 packet, unit, lane;
    bool occupied;
    u32 *stage;

    // The rows holding source packets.
    __device__ __forceinline__ u32 real() const {
        return occupied ? u32(min(u64(TILE_ROWS), SEISMIC_DIM_N - min(row0, u64(SEISMIC_DIM_N)))) : 0u;
    }

    __device__ __forceinline__ Stage staging() const {
        return Stage{source_packet(source, seismic_words_value, b, row0, packet),
                     u64(SEISMIC_SOURCE_STRIDE_1) * SEISMIC_ELEMENT_E_PACKET_SIZE, real(), lane, stage};
    }

    // Row `lane`'s record words (zero where its row holds no source
    // packet): the tile's unit's at words[0..], then, at a group's first
    // tile, the fixed words at their record words.
    __device__ __forceinline__ void unit_record(u32 *words) const {
        SEISMIC_CONVERSION_E_U_UNIT(staging(), unit, words);
        if (lane >= real())
            for (u32 word = 0; word < UNIT_WORDS; ++word) words[word] = 0;
    }

    // The fixed words' source units. Where every tile is its group's first
    // (a group of one unit), they are loaded before the unit's are staged, so
    // both are in flight together (loaded after the unit's words, every tile
    // waits on memory twice: measured against A7, q4_0 rows16 1.16x its
    // time). Elsewhere a group's first tile loads them once its unit's words
    // are placed: held across the unit, their predicates leave the unit's
    // staging too few to keep a batch's loads in flight on every tile
    // (measured: q6_k rows16 1.14x).
    __device__ __forceinline__ void fixed_load(u32 *loaded) const {
        if (UNITS == 1) SEISMIC_CONVERSION_E_U_FIXED_LOAD(staging(), loaded);
    }

    // At a group's first tile, after its unit's (`loaded` as `fixed_load`
    // left it).
    __device__ __forceinline__ void fixed_record(u32 *loaded, u32 *words) const {
        // The stage is the warp's: every lane is done with what it held.
        __syncwarp();
        if (UNITS != 1) SEISMIC_CONVERSION_E_U_FIXED_LOAD(staging(), loaded);
        SEISMIC_CONVERSION_E_U_FIXED(staging(), loaded, words);
        if (lane >= real())
            for (u32 word = 0; word < WORDS; ++word) words[word] = 0;
    }
};

#if defined(SEISMIC_ELEMENT_U_LAYOUT_PACKET)

// ---- packet: the record is the packet form ---------------------------------

#define CODES_WORD SEISMIC_CONVERSION_E_U_CODES_FIRST
#define CODES_END (CODES_WORD + UNITS * CODES_WORDS)
static_assert(CODES_END * 4 <= SEISMIC_ELEMENT_U_PACKET_SIZE, "the codes are whole words of a packet");

__device__ __forceinline__ u8 *destination_packet(u8 *destination, const seismic_words_t &seismic_words_value, u64 b,
                                                  u64 n, u64 packet) {
    return destination + (b * SEISMIC_RESULT_0_STRIDE_0 + n * SEISMIC_RESULT_0_STRIDE_1 +
                          packet * SEISMIC_RESULT_0_STRIDE_2) * SEISMIC_ELEMENT_U_PACKET_SIZE;
}

// The copy unit of record bytes of `alignment` (a record alignment,
// `SEISMIC_CONVERSION_E_U_*_ALIGNMENT`) from the staged record: to 4 bytes,
// the widest a shared word gives.
#define COPY_UNIT(alignment) ((alignment) < 4u ? (alignment) : 4u)

// The fixed words: the record's but its group's codes.
#define FIXED_WORDS (WORDS - (CODES_END - CODES_WORD))

// A tile's record words are staged per row before being stored, in the
// source stage's memory, from staged word 0: its unit's codes, then (at a
// group's first tile) its fixed words. Only those: a block's shared memory
// bounds the blocks (one warp each) an SM keeps resident, and staging every
// record word (iq3_s's 53 when it converted into q6k: 4416 bytes a block)
// fits 22 blocks in an sm_120 SM's 100 KB, of the 32 it would otherwise run.
#define RECORD_STRIDE odd_words(CODES_WORDS > FIXED_WORDS ? CODES_WORDS : FIXED_WORDS)

// Row `lane`'s words [first, first + count) and [second, second + more),
// staged one after the other from staged word 0.
__device__ __forceinline__ void stage_record(const Tile &tile, const u32 *words, u32 first, u32 count, u32 second,
                                             u32 more) {
    __syncwarp();
    if (tile.lane < TILE_ROWS) {
        for (u32 word = 0; word < count; ++word) tile.stage[tile.lane * RECORD_STRIDE + word] = words[first + word];
        for (u32 word = 0; word < more; ++word)
            tile.stage[tile.lane * RECORD_STRIDE + count + word] = words[second + word];
    }
    __syncwarp();
}

// Packet bytes [low, low + bytes), staged from record byte `slot` on, of
// every real row into its destination packet (rows SEISMIC_RESULT_0_STRIDE_1
// packets apart), coalesced, in copies of `unit` bytes (`COPY_UNIT`); every
// argument but `low` a constant.
__device__ __forceinline__ void store_bytes(const Tile &tile, u8 *out, u32 slot, u32 bytes, u32 unit, u32 low) {
    const seismic_words_t &seismic_words_value = tile.seismic_words_value;
    const u32 units = bytes / unit, real = tile.real();
    const u64 row_bytes = u64(SEISMIC_RESULT_0_STRIDE_1) * SEISMIC_ELEMENT_U_PACKET_SIZE;
#pragma unroll 4
    for (u32 at = tile.lane; at < TILE_ROWS * units; at += LANES) {
        const u32 r = at / units, k = at % units;
        u8 *to = out + r * row_bytes + low + k * unit;
        const u8 *from = reinterpret_cast<const u8 *>(tile.stage + r * RECORD_STRIDE) + slot + k * unit;
        if (r < real) {
            if (unit == 4) *reinterpret_cast<u32 *>(to) = *reinterpret_cast<const u32 *>(from);
            else if (unit == 2) *reinterpret_cast<u16 *>(to) = *reinterpret_cast<const u16 *>(from);
            else *to = *from;
        }
    }
}

extern "C" __global__ void repack_weight(SEISMIC_KERNEL_PARAMS) {
    const u8 *source = reinterpret_cast<const u8 *>(SEISMIC_PTR(SEISMIC_BUFFER_SOURCE));
    u8 *destination = reinterpret_cast<u8 *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
    // The warp's staged source bytes, then its record words.
    __shared__ u32 stage[TILE_ROWS * (STAGE_WORDS > RECORD_STRIDE ? STAGE_WORDS : RECORD_STRIDE)];
    const u32 lane = threadIdx.x;
    const u64 b = blockIdx.z;
    const u64 row0 = u64(blockIdx.y) * TILE_ROWS;
    const u32 column0 = blockIdx.x * TILE_COLUMNS;
    const u64 packets = (SEISMIC_DIM_K + GROUP - 1) / GROUP;
    if (column0 >= packets * GROUP || row0 >= SEISMIC_DIM_N) return;
    const u32 packet = column0 / GROUP;
    const Tile tile{source, seismic_words_value, b, row0, packet, column0 % GROUP / TILE_COLUMNS, lane, true,
                    stage};
    u8 *out = destination_packet(destination, seismic_words_value, b, row0, packet);
    u32 words[WORDS], loaded[FIXED_LOADS];
    tile.fixed_load(loaded);
    tile.unit_record(words);
    stage_record(tile, words, 0, CODES_WORDS, 0, 0);
    store_bytes(tile, out, 0, 4 * CODES_WORDS, COPY_UNIT(SEISMIC_CONVERSION_E_U_CODES_ALIGNMENT),
                4 * (CODES_WORD + tile.unit * CODES_WORDS));
    if (tile.unit != 0) return;
    // Every other record word: the other planes, and the gaps (zero).
    tile.fixed_record(loaded, words);
    constexpr u32 end = 4 * WORDS < SEISMIC_ELEMENT_U_PACKET_SIZE ? 4 * WORDS : SEISMIC_ELEMENT_U_PACKET_SIZE;
    stage_record(tile, words, 0, CODES_WORD, CODES_END, WORDS - CODES_END);
    store_bytes(tile, out, 0, 4 * CODES_WORD, COPY_UNIT(SEISMIC_CONVERSION_E_U_LEADING_ALIGNMENT), 0);
    store_bytes(tile, out, 4 * CODES_WORD, end > 4 * CODES_END ? end - 4 * CODES_END : 0,
                COPY_UNIT(SEISMIC_CONVERSION_E_U_TRAILING_ALIGNMENT), 4 * CODES_END);
}

#else

// ---- rows16 / mma16: the record is the row planes' shares -------------------

#define ROLE_WORD(role) SEISMIC_CONVERSION_E_U##_##role##_WORD
#define ROLE_BYTES(role) SEISMIC_CONVERSION_E_U##_##role##_BYTES
#define ROLE_ALIGNMENT(role) SEISMIC_CONVERSION_E_U##_##role##_ALIGNMENT

// Start of stored row `n` of matrix `b`.
__device__ __forceinline__ u8 *row_base(u8 *destination, const seismic_words_t &seismic_words_value, u64 b, u64 n) {
    return destination + (b * SEISMIC_RESULT_0_STRIDE_0 + n * SEISMIC_RESULT_0_STRIDE_1) *
                             SEISMIC_RESULT_0_ROW_STRIDE_BYTES;
}

// `count` bytes at `out`, zero.
__device__ __forceinline__ void zero_bytes(u8 *out, u64 count) {
    for (u64 byte = 0; byte < count; ++byte) out[byte] = 0;
}

// Rows of an mma16 tile: a row tile's.
#define MMA_ROWS ((u32)SEISMIC_ELEMENT_U_TILE_ROWS)
static_assert(TILE_ROWS % MMA_ROWS == 0, "a row tile holds whole mma16 tiles");

#if defined(SEISMIC_ELEMENT_U_PLANE_RECORDS)

// ---- mma16 records ----------------------------------------------------------
//
// An mma16 tile's 16 rows are one sequence (`PackedRowLayout::record_byte`):
// every stored group's entries, then every group's factors, then zero to the
// tile's end. Entries are 16 bits, in GEMV lane order: entry `e` of row `r`
// of a group, q = e / 8, at byte q / 2 * 512 + (4 (r % 8) + e % 4) * 16 +
// (2 (q % 2) + r / 8) * 4 + 2 (e % 8 / 4); a group's factors are
// [r % 8][r / 8][factor bytes]. A unit's words are two entries each.

#define ENTRY_BITS 16u
#define ENTRY_BYTES ((u64)SEISMIC_ELEMENT_U_PLANE_RECORDS_ENTRY_BYTES)
// The codes role is a group's entries (its units' words), then its factors.
#define ENTRIES_BYTES (GROUP * SEISMIC_ELEMENT_U_PLANE_RECORDS_CODE_BITS / 8u)
#define FACTOR_BYTES (ROLE_BYTES(CODES) - ENTRIES_BYTES)
#define FACTORS_WORD (ROLE_WORD(CODES) + ENTRIES_BYTES / 4u)
static_assert(SEISMIC_CONVERSION_E_U_CODES_FIRST + UNITS * CODES_WORDS == FACTORS_WORD,
              "a records share is its units' entries, then its factors");
static_assert(ENTRY_BYTES == MMA_ROWS * ENTRIES_BYTES, "a tile's group entries are its rows' entries");
static_assert(HIGH_WORDS == 0, "a records unit is its entries");

// Entry `e` of row `r` of a group (`value`) at the group's entries.
__device__ __forceinline__ void place_entry(u8 *entries, u32 r, u32 e, u32 value) {
    const u32 q = e / 8;
    *reinterpret_cast<u16 *>(entries + q / 2 * 512 + (4 * (r % 8) + e % 4) * 16 + (2 * (q % 2) + r / 8) * 4 +
                             2 * (e % 8 / 4)) = u16(value);
}

extern "C" __global__ void repack_weight(SEISMIC_KERNEL_PARAMS) {
    const u8 *source = reinterpret_cast<const u8 *>(SEISMIC_PTR(SEISMIC_BUFFER_SOURCE));
    u8 *destination = reinterpret_cast<u8 *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
    __shared__ u32 stage[TILE_ROWS * STAGE_WORDS];
    const u32 lane = threadIdx.x;
    const u64 b = blockIdx.z;
    const u64 row0 = u64(blockIdx.y) * TILE_ROWS;
    const u32 column0 = blockIdx.x * TILE_COLUMNS;
    const u64 packets = (SEISMIC_DIM_K + GROUP - 1) / GROUP;
    const u64 stored_columns = SEISMIC_RESULT_0_ROW_GROUPS * GROUP;
    // Stored rows: whole mma16 tiles, rows beyond N zero.
    const u64 stored_rows = (SEISMIC_DIM_N + MMA_ROWS - 1) / MMA_ROWS * MMA_ROWS;
    if (column0 >= stored_columns || row0 >= stored_rows) return;
    const u64 rows = min(u64(TILE_ROWS), stored_rows - row0);
    const Tile tile{source, seismic_words_value, b, row0, column0 / GROUP, column0 % GROUP / TILE_COLUMNS, lane,
                    column0 < packets * GROUP, stage};
    // Row `lane`'s mma16 tile.
    u8 *mma = row_base(destination, seismic_words_value, b, row0 + lane - lane % MMA_ROWS);
    const u32 r = lane % MMA_ROWS;
    u32 words[WORDS], loaded[FIXED_LOADS];
    tile.fixed_load(loaded);
    tile.unit_record(words);
    if (lane < rows) {
        u8 *entries = mma + tile.packet * ENTRY_BYTES;
        for (u32 word = 0; word < UNIT_WORDS; ++word) {
            const u32 e = 2 * (tile.unit * UNIT_WORDS + word);
            place_entry(entries, r, e, words[word]);
            place_entry(entries, r, e + 1, words[word] >> ENTRY_BITS);
        }
    }
    if (tile.unit == 0) {
        tile.fixed_record(loaded, words);
        if (lane < rows)
            store(mma + SEISMIC_RESULT_0_ROW_GROUPS * ENTRY_BYTES + tile.packet * (MMA_ROWS * FACTOR_BYTES) +
                      (2 * (r % 8) + r / 8) * FACTOR_BYTES,
                  words + FACTORS_WORD, FACTOR_BYTES, SEISMIC_CONVERSION_E_U_CODES_REST_ALIGNMENT);
    }

    // Each mma16 tile's bytes past its last factors.
    if (column0 + TILE_COLUMNS == stored_columns) {
        for (u64 tile_row = 0; tile_row < rows; tile_row += MMA_ROWS) {
            u8 *tile_bytes = row_base(destination, seismic_words_value, b, row0 + tile_row);
            for (u64 byte = SEISMIC_RESULT_0_ROW_GROUPS * MMA_ROWS * ROLE_BYTES(CODES) + lane;
                 byte < MMA_ROWS * SEISMIC_RESULT_0_ROW_STRIDE_BYTES; byte += LANES)
                tile_bytes[byte] = 0;
        }
    }
}

#else

// ---- rows16 / mma16 planes --------------------------------------------------

#if defined(SEISMIC_ELEMENT_U_LAYOUT_MMA16)
#if !defined(SEISMIC_ELEMENT_U_PLANE_GRIDS)
static_assert(32 % CODES_WORDS == 0 && (HIGH_WORDS == 0 || 32 % HIGH_WORDS == 0), "a code lies in one staged word");
#endif

// Destination of byte `position` of an mma16 tile's code-plane sequence V:
// the tile's rows' plane payloads (`row_bytes` each) concatenated in row
// order.
__device__ __forceinline__ u8 *tile_plane_byte(u8 *destination, const seismic_words_t &seismic_words_value, u64 b,
                                               u64 tile_row0, u64 position, u64 row_bytes, u64 offset) {
    return row_base(destination, seismic_words_value, b, tile_row0 + position / row_bytes) + offset +
           position % row_bytes;
}

// A code role, `bits` a column (a constant), whose words of the warp's rows
// are staged from word `at` of each row: of each of the warp's mma16 tiles,
// this lane's fragment slots of the tile's two k16 steps
// (`PackedRowLayout::code_bit`): slot `j` of a step holds the code of (row
// lane/4 + 8 (j % 2), column 2 (lane % 4) + 8 ((j % 4) / 2) + j / 4) at bit
// `bits` j.
__device__ __forceinline__ void place_fragments(const Tile &tile, u8 *destination, u64 rows, u32 column0,
                                                const u32 *staged, u32 bits, u32 at, u64 offset, u64 row_bytes) {
    if (bits == 0) return;
    const u32 lane = tile.lane;
    const u32 block = column0 / 64, step0 = (column0 % 64) / 16;
    for (u32 tile_row = 0; tile_row < rows; tile_row += MMA_ROWS) {
        u32 value[4] = {};
#pragma unroll
        for (u32 half_step = 0; half_step < 2; ++half_step)
#pragma unroll
            for (u32 slot = 0; slot < 8; ++slot) {
                const u32 row = tile_row + lane / 4 + 8 * (slot % 2);
                const u32 column = 16 * half_step + 2 * (lane % 4) + 8 * ((slot % 4) / 2) + slot / 4;
                const u32 code = (staged[row * UNIT_WORDS + at + column * bits / 32] >> (column * bits % 32)) &
                                 ((1u << bits) - 1u);
                const u32 bit = (8 * half_step + slot) * bits;
                value[bit / 32] |= code << (bit % 32);
            }
        store(tile_plane_byte(destination, tile.seismic_words_value, tile.b, tile.row0 + tile_row,
                              (u64(block) * 32 + lane) * 4 * bits + bits * step0, row_bytes, offset),
              // (`bits` a power of two: the 2 `bits` bytes, at a multiple
              // of them, keep their own alignment, to 16.)
              value, 2 * bits, 2 * bits < 16 ? 2 * bits : 16);
    }
}

// A grids plane's unit (`unit_bytes` bytes, a constant, of `words`, at the
// plane at row offset `offset`, `row_bytes` a row) of the lane's row: grid
// entries and their high parts stay whole, an mma16 tile's sequence being
// [k-block][row][the k-block's two units] (`PackedRowLayout::code_bit`), so
// the row's 32 columns (steps `step0`, `step0 + 1`) are one of its k-block's
// units. Stored in units of `alignment` bytes (its role's).
__device__ __forceinline__ void place_entries(const Tile &tile, u8 *destination, u32 column0, const u32 *words,
                                              u32 unit_bytes, u64 offset, u64 row_bytes, u32 alignment) {
    const seismic_words_t &seismic_words_value = tile.seismic_words_value;
    const u32 r = tile.lane % MMA_ROWS;
    const u32 block = column0 / 64, step0 = (column0 % 64) / 16;
    store(tile_plane_byte(destination, seismic_words_value, tile.b, tile.row0 + tile.lane - r,
                          (u64(block) * MMA_ROWS + r) * (2 * unit_bytes) + step0 / 2 * unit_bytes, row_bytes, offset),
          words, unit_bytes, alignment);
}
#endif

extern "C" __global__ void repack_weight(SEISMIC_KERNEL_PARAMS) {
    const u8 *source = reinterpret_cast<const u8 *>(SEISMIC_PTR(SEISMIC_BUFFER_SOURCE));
    u8 *destination = reinterpret_cast<u8 *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
    __shared__ u32 stage[TILE_ROWS * STAGE_WORDS];
    const u32 lane = threadIdx.x;
#if defined(SEISMIC_ELEMENT_U_LAYOUT_MMA16) && !defined(SEISMIC_ELEMENT_U_PLANE_GRIDS)
    // The code roles' words of the warp's rows (mma16 fragments).
    __shared__ u32 staged[TILE_ROWS * UNIT_WORDS];
#endif
    const u64 b = blockIdx.z;
    const u64 row0 = u64(blockIdx.y) * TILE_ROWS;
    const u32 column0 = blockIdx.x * TILE_COLUMNS;
    const u64 packets = (SEISMIC_DIM_K + GROUP - 1) / GROUP;
    const u64 stored_columns = SEISMIC_RESULT_0_ROW_GROUPS * GROUP;
#if defined(SEISMIC_ELEMENT_U_LAYOUT_MMA16)
    // Stored rows: whole mma16 tiles, rows beyond N zero.
    const u64 stored_rows = (SEISMIC_DIM_N + MMA_ROWS - 1) / MMA_ROWS * MMA_ROWS;
#else
    const u64 stored_rows = SEISMIC_DIM_N;
#endif
    if (column0 >= stored_columns || row0 >= stored_rows) return;
    const u64 rows = min(u64(TILE_ROWS), stored_rows - row0);
    const Tile tile{source, seismic_words_value, b, row0, column0 / GROUP, column0 % GROUP / TILE_COLUMNS, lane,
                    column0 < packets * GROUP, stage};
    u8 *row = row_base(destination, seismic_words_value, b, row0 + lane);
    u32 words[WORDS], loaded[FIXED_LOADS];
    tile.fixed_load(loaded);
    tile.unit_record(words);
#if defined(SEISMIC_ELEMENT_U_LAYOUT_MMA16) && defined(SEISMIC_ELEMENT_U_PLANE_GRIDS)
    // Grid entries and their high parts stay whole (`place_entries`).
    if (lane < rows) {
        place_entries(tile, destination, column0, words, CODES_UNIT_BYTES, SEISMIC_RESULT_0_ROW_CODES,
                      SEISMIC_RESULT_0_ROW_CODES_BYTES, ROLE_ALIGNMENT(CODES));
        place_entries(tile, destination, column0, words + CODES_WORDS, HIGH_UNIT_BYTES, SEISMIC_RESULT_0_ROW_HIGH,
                      SEISMIC_RESULT_0_ROW_HIGH_BYTES, ROLE_ALIGNMENT(HIGH));
    }
#elif defined(SEISMIC_ELEMENT_U_LAYOUT_MMA16)
    // The code roles' words of the warp's rows staged, then placed.
    if (lane < rows)
        for (u32 word = 0; word < UNIT_WORDS; ++word) staged[lane * UNIT_WORDS + word] = words[word];
    __syncwarp();
    place_fragments(tile, destination, rows, column0, staged, CODES_WORDS, 0, SEISMIC_RESULT_0_ROW_CODES,
                    SEISMIC_RESULT_0_ROW_CODES_BYTES);
    place_fragments(tile, destination, rows, column0, staged, HIGH_WORDS, CODES_WORDS, SEISMIC_RESULT_0_ROW_HIGH,
                    SEISMIC_RESULT_0_ROW_HIGH_BYTES);
#else
    // The tile's 32 columns of a row are one unit of each code plane.
    if (lane < rows) {
        store(row + SEISMIC_RESULT_0_ROW_CODES + column0 / TILE_COLUMNS * CODES_UNIT_BYTES, words, CODES_UNIT_BYTES,
              ROLE_ALIGNMENT(CODES));
        store(row + SEISMIC_RESULT_0_ROW_HIGH + column0 / TILE_COLUMNS * HIGH_UNIT_BYTES, words + CODES_WORDS,
              HIGH_UNIT_BYTES, ROLE_ALIGNMENT(HIGH));
    }
#endif

    // Group planes, at the group's first tile: a row's groups are
    // contiguous.
    if (tile.unit == 0) {
        tile.fixed_record(loaded, words);
        if (lane < rows) {
            store(row + SEISMIC_RESULT_0_ROW_SCALES + tile.packet * ROLE_BYTES(SCALES), words + ROLE_WORD(SCALES),
                  ROLE_BYTES(SCALES), ROLE_ALIGNMENT(SCALES));
            store(row + SEISMIC_RESULT_0_ROW_SUPERS + tile.packet * ROLE_BYTES(SUPERS), words + ROLE_WORD(SUPERS),
                  ROLE_BYTES(SUPERS), ROLE_ALIGNMENT(SUPERS));
        }
    }

    // Alignment tails of every plane, from its end to the next plane's start
    // (or the row's end), at the row's last stored tile.
    if (column0 + TILE_COLUMNS == stored_columns && lane < rows) {
        zero_bytes(row + SEISMIC_RESULT_0_ROW_CODES + SEISMIC_RESULT_0_ROW_CODES_BYTES, SEISMIC_RESULT_0_ROW_CODES_TAIL);
        zero_bytes(row + SEISMIC_RESULT_0_ROW_HIGH + SEISMIC_RESULT_0_ROW_HIGH_BYTES, SEISMIC_RESULT_0_ROW_HIGH_TAIL);
        zero_bytes(row + SEISMIC_RESULT_0_ROW_SCALES + SEISMIC_RESULT_0_ROW_SCALES_BYTES,
                   SEISMIC_RESULT_0_ROW_SCALES_TAIL);
        zero_bytes(row + SEISMIC_RESULT_0_ROW_SUPERS + SEISMIC_RESULT_0_ROW_SUPERS_BYTES,
                   SEISMIC_RESULT_0_ROW_SUPERS_TAIL);
    }
}

#endif

#endif
