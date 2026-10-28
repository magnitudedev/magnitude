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
// planes' words, and SEISMIC_CONVERSION_E_U_FIXED, the other words; this
// kernel loads, and places records in U's layout.
//
// Each threadgroup (32 threads) places one row tile: 16 rows x 32 columns.
// Every group is a multiple of 32, so a tile covers a whole packet or part of
// a larger one, and a code plane's share of a tile row is whole record words;
// the tile at a group's first column also places that group's other record
// words. Stored-but-unoccupied bytes (mma16 padding rows and groups, plane
// alignment tails, packet gaps) are written as zero, so the result is fully
// defined.

#if !defined(SEISMIC_ELEMENT_E_KIND_EXTERNAL) || !defined(SEISMIC_ELEMENT_U_KIND_PACKED)
#error "repack_weight converts an external GGUF source into packed resident storage"
#endif

#if !defined(SEISMIC_CONVERSION_E_U)
#error "repack_weight binding is not a registered GGUF-to-resident conversion"
#endif

#if !defined(SEISMIC_ELEMENT_U_LAYOUT_PACKET) && !defined(SEISMIC_ELEMENT_U_LAYOUT_ROWS16) && \
    !defined(SEISMIC_ELEMENT_U_LAYOUT_MMA16) && !defined(SEISMIC_ELEMENT_U_LAYOUT_ROWS32)
#error "repack_weight writes packet, rows16, mma16 or rows32 storage"
#endif

#if defined(SEISMIC_ELEMENT_U_PLANE_RECORDS)
#error "repack_weight does not write mma16 records (CUDA's iq2xxs storage)"
#endif

#define GROUP ((uint)SEISMIC_ELEMENT_E_LOGICAL_GROUP)
#define TILE_ROWS 16u
#define TILE_COLUMNS 32u
static_assert(GROUP % TILE_COLUMNS == 0, "a tile lies in one group");

// A unit's code words: the codes', then the high codes' (`WordSpan`).
#define CODES_WORDS SEISMIC_CONVERSION_E_U_CODES_UNIT_WORDS
#define HIGH_WORDS SEISMIC_CONVERSION_E_U_HIGH_UNIT_WORDS
#define UNIT_WORDS (CODES_WORDS + HIGH_WORDS)
// A high role outside the span (a grid entry's high part, below a byte a
// column) is a share of its group, placed at the group's first tile.
#define HIGH_GROUP_SHARE (HIGH_WORDS == 0 && SEISMIC_CONVERSION_E_U_HIGH_BYTES != 0)

// Helpers are inlined before they are optimized: optimized alone, their
// bodies are general in the constants every call site gives them.
#define ALWAYS_INLINE __attribute__((always_inline)) inline

// Source packet `packet` of row `n` of matrix `b`.
ALWAYS_INLINE device const uchar *source_packet(device const uchar *source, constant ulong *seismic_words,
                                         ulong b, ulong n, ulong packet) {
    return source + (b * SEISMIC_SOURCE_STRIDE_0 + n * SEISMIC_SOURCE_STRIDE_1 + packet * SEISMIC_SOURCE_STRIDE_2) *
                        SEISMIC_ELEMENT_E_PACKET_SIZE;
}

// `bytes` bytes of `words` (little-endian) at `out`, in stores of `unit`
// bytes (a power of two, to 16, that `bytes` and `out` keep: the record's
// alignments, `SEISMIC_CONVERSION_E_U_*_ALIGNMENT`); `bytes` and `unit`
// constants, so every branch but one folds away. The caller computes every
// word before any is stored.
ALWAYS_INLINE void store(device uchar *out, thread const uint *words, uint bytes, uint unit) {
    for (uint at = 0; at < bytes; at += unit) {
        const uint word = at / 4;
        if (unit == 16) {
            *reinterpret_cast<device uint4 *>(out + at) = uint4(words[word], words[word + 1], words[word + 2], words[word + 3]);
        } else if (unit == 8) {
            *reinterpret_cast<device uint2 *>(out + at) = uint2(words[word], words[word + 1]);
        } else if (unit == 4) {
            *reinterpret_cast<device uint *>(out + at) = words[word];
        } else if (unit == 2) {
            *reinterpret_cast<device ushort *>(out + at) = ushort(words[word] >> (at % 4 * 8));
        } else {
            out[at] = uchar(words[word] >> (at % 4 * 8));
        }
    }
}

#if defined(SEISMIC_ELEMENT_U_LAYOUT_PACKET)

// ---- packet: the record is the packet form ---------------------------------

// The codes' (packet plane 0's) first record word and words per 32 columns.
#define CODES_WORD SEISMIC_CONVERSION_E_U_CODES_FIRST
static_assert((CODES_WORD + GROUP / TILE_COLUMNS * CODES_WORDS) * 4 <= SEISMIC_ELEMENT_U_PACKET_SIZE,
              "the codes are whole words of a packet");

ALWAYS_INLINE device uchar *destination_packet(device uchar *destination, constant ulong *seismic_words,
                                        ulong b, ulong n, ulong packet) {
    return destination + (b * SEISMIC_RESULT_0_STRIDE_0 + n * SEISMIC_RESULT_0_STRIDE_1 +
                          packet * SEISMIC_RESULT_0_STRIDE_2) * SEISMIC_ELEMENT_U_PACKET_SIZE;
}

kernel void repack_weight(
    device const uchar *source [[buffer(SEISMIC_BUFFER_SOURCE)]],
    device uchar *destination [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 tile [[threadgroup_position_in_grid]],
    uint3 thread_position [[thread_position_in_threadgroup]]) {
    // Lane l places row l's words: every lane runs one stream of code (a
    // lane's own share of a row would diverge the threadgroup's halves).
    const uint lane = thread_position.x;
    const ulong b = tile.z;
    const ulong n = ulong(tile.y) * TILE_ROWS + lane;
    const uint column0 = tile.x * TILE_COLUMNS;
    const ulong packets = (SEISMIC_DIM_K + GROUP - 1) / GROUP;
    if (lane >= TILE_ROWS || column0 >= packets * GROUP || n >= SEISMIC_DIM_N) return;
    const uint packet = column0 / GROUP;
    const uint within = column0 % GROUP;
    device const uchar *in = source_packet(source, seismic_words, b, n, packet);
    device uchar *out = destination_packet(destination, seismic_words, b, n, packet);
    const uint unit = within / TILE_COLUMNS;
    uint words[SEISMIC_CONVERSION_E_U_WORDS];
    SEISMIC_CONVERSION_E_U_UNIT(in, unit, words);
    store(out + 4 * (CODES_WORD + unit * CODES_WORDS), words, 4 * CODES_WORDS, SEISMIC_CONVERSION_E_U_CODES_ALIGNMENT);
    if (within != 0) return;
    // Every other record word: the other planes, and the gaps (zero), to
    // the packet's end.
    constexpr uint codes_end = CODES_WORD + GROUP / TILE_COLUMNS * CODES_WORDS;
    constexpr uint end = 4 * SEISMIC_CONVERSION_E_U_WORDS < SEISMIC_ELEMENT_U_PACKET_SIZE
                             ? 4 * SEISMIC_CONVERSION_E_U_WORDS
                             : SEISMIC_ELEMENT_U_PACKET_SIZE;
    SEISMIC_CONVERSION_E_U_FIXED(in, words);
    store(out, words, 4 * CODES_WORD, SEISMIC_CONVERSION_E_U_LEADING_ALIGNMENT);
    store(out + 4 * codes_end, words + codes_end, end - 4 * codes_end, SEISMIC_CONVERSION_E_U_TRAILING_ALIGNMENT);
}

#else

// ---- rows16 / rows32 / mma16: the record is the row planes' shares ---------

// A role's share: its first record word and bytes.
#define ROLE_WORD(role) SEISMIC_CONVERSION_E_U##_##role##_WORD
#define ROLE_BYTES(role) SEISMIC_CONVERSION_E_U##_##role##_BYTES
#define ROLE_ALIGNMENT(role) SEISMIC_CONVERSION_E_U##_##role##_ALIGNMENT

// Start of stored row `n` of matrix `b`.
ALWAYS_INLINE device uchar *row_base(device uchar *destination, constant ulong *seismic_words, ulong b, ulong n) {
    return destination + (b * SEISMIC_RESULT_0_STRIDE_0 + n * SEISMIC_RESULT_0_STRIDE_1) *
                             SEISMIC_RESULT_0_ROW_STRIDE_BYTES;
}

// Rows of one stored row tile: `rows32` interleaves every plane across 32
// rows; `rows16` rows stand alone.
#if defined(SEISMIC_ELEMENT_U_LAYOUT_ROWS32)
#define STORED_TILE 32ul
#else
#define STORED_TILE 1ul
#endif

// Unit `unit` (of `unit_bytes` bytes) of stored row `n` in the code plane at
// row offset `offset`: units follow one another along a `rows16` row; in
// `rows32` the tile's rows of one unit are adjacent
// (`PackedRowLayout::code_bit`).
ALWAYS_INLINE device uchar *plane_unit(device uchar *destination, constant ulong *seismic_words, ulong b, ulong n,
                                ulong offset, ulong unit_bytes, ulong unit) {
    return row_base(destination, seismic_words, b, n - n % STORED_TILE) + offset * STORED_TILE +
           (unit * STORED_TILE + n % STORED_TILE) * unit_bytes;
}

// Storage group `group` (of `group_bytes` bytes) of stored row `n` in the
// group plane at row offset `offset` (`row_bytes` a row): a row's groups are
// contiguous in every layout, and a `rows32` tile holds its rows' shares of
// the plane one after another (`PackedRowLayout::packet_bit`).
ALWAYS_INLINE device uchar *plane_group(device uchar *destination, constant ulong *seismic_words, ulong b, ulong n,
                                 ulong offset, ulong row_bytes, ulong group_bytes, ulong group) {
    return row_base(destination, seismic_words, b, n - n % STORED_TILE) + offset * STORED_TILE +
           n % STORED_TILE * row_bytes + group * group_bytes;
}

ALWAYS_INLINE void zero_bytes(device uchar *out, ulong count) {
    for (ulong i = 0; i < count; ++i) out[i] = 0;
}

// Stored row `n`'s share of the `length`-bytes-a-row gap that starts at row
// offset `offset` (a plane's alignment tail), zero: `rows32` tiles hold
// their rows' shares one after another.
ALWAYS_INLINE void zero_tail(device uchar *destination, constant ulong *seismic_words, ulong b, ulong n,
                             ulong offset, ulong length) {
    zero_bytes(row_base(destination, seismic_words, b, n - n % STORED_TILE) + offset * STORED_TILE +
                   (n % STORED_TILE) * length,
               length);
}

#if defined(SEISMIC_ELEMENT_U_LAYOUT_MMA16)
#if !defined(SEISMIC_ELEMENT_U_PLANE_GRIDS)
static_assert(32 % CODES_WORDS == 0 && (HIGH_WORDS == 0 || 32 % HIGH_WORDS == 0), "a code lies in one staged word");
#endif

// Destination of byte `position` of a tile's code-plane sequence V: the
// rows' plane payloads (`row_bytes` each) concatenated in row order.
ALWAYS_INLINE device uchar *tile_plane_byte(device uchar *destination, constant ulong *seismic_words, ulong b,
                                            ulong row0, ulong position, ulong row_bytes, ulong offset) {
    return row_base(destination, seismic_words, b, row0 + position / row_bytes) + offset + position % row_bytes;
}

// A code plane of an mma16 tile, `bits` a column (a constant), lane
// `lane`'s row's `words` of it: the rows' words staged, then lane `lane`'s
// fragment slots of k16 steps `step0` and `step0 + 1` of k-block `block`
// (`PackedRowLayout::code_bit`): slot `j` of a step holds the code of (row
// lane/4 + 8 (j % 2), column 2 (lane % 4) + 8 ((j % 4) / 2) + j / 4) at bit
// `bits` j.
ALWAYS_INLINE void place_fragments(device uchar *destination, constant ulong *seismic_words, ulong b, ulong row0,
                            thread const uint *words, uint bits, ulong offset, ulong row_bytes, uint lane, uint block,
                            uint step0, threadgroup uint *staged) {
    if (bits == 0) return;
    if (lane < TILE_ROWS)
        for (uint word = 0; word < bits; ++word) staged[lane * bits + word] = words[word];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint value[4] = {0u, 0u, 0u, 0u};
    for (uint half_step = 0; half_step < 2; ++half_step)
        for (uint slot = 0; slot < 8; ++slot) {
            const uint row = lane / 4 + 8 * (slot % 2);
            const uint column = 16 * half_step + 2 * (lane % 4) + 8 * ((slot % 4) / 2) + slot / 4;
            const uint code = (staged[row * bits + column * bits / 32] >> (column * bits % 32)) & ((1u << bits) - 1u);
            const uint bit = (8 * half_step + slot) * bits;
            value[bit / 32] |= code << (bit % 32);
        }
    device uchar *out = tile_plane_byte(destination, seismic_words, b, row0,
                                        (ulong(block) * 32 + lane) * 4 * bits + bits * step0, row_bytes, offset);
    // (`bits` a power of two: the 2 `bits` bytes, at a multiple of them,
    // keep their own alignment, to 16.)
    store(out, value, 2 * bits, 2 * bits < 16 ? 2 * bits : 16);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

#if HIGH_GROUP_SHARE
// The group high share at `words` of row `lane` of the tile at `row0`,
// placed as its entries are (`PackedRowLayout::code_bit`): the tile's
// sequence is [k-block][row][the k-block's high bytes].
ALWAYS_INLINE void place_high(device uchar *destination, constant ulong *seismic_words, ulong b, ulong row0,
                              uint lane, uint group, thread const uint *words) {
    constexpr uint kblocks = GROUP / 64, piece = ROLE_BYTES(HIGH) / kblocks;
    static_assert(GROUP % 64 == 0 && ROLE_BYTES(HIGH) % kblocks == 0, "a group's k-blocks hold its high bytes alike");
    for (uint q = 0; q < kblocks; ++q)
        for (uint i = 0; i < piece; ++i) {
            const uint at = q * piece + i;
            *tile_plane_byte(destination, seismic_words, b, row0, (ulong(group * kblocks + q) * TILE_ROWS + lane) * piece + i,
                             SEISMIC_RESULT_0_ROW_HIGH_BYTES, SEISMIC_RESULT_0_ROW_HIGH) = uchar(words[at / 4] >> (8 * (at % 4)));
        }
}
#endif
#else
// A code plane's `words` (`bits` a column, a constant) of the tile's 32
// columns of stored row `n`: one unit of its plane, stored in units of
// `unit` bytes (its role's alignment).
ALWAYS_INLINE void place_unit(device uchar *destination, constant ulong *seismic_words, ulong b, ulong n,
                       thread const uint *words, uint bits, ulong offset, ulong stored_unit, uint unit) {
    if (bits != 0)
        store(plane_unit(destination, seismic_words, b, n, offset, 4ul * bits, stored_unit), words, 4 * bits, unit);
}
#endif

kernel void repack_weight(
    device const uchar *source [[buffer(SEISMIC_BUFFER_SOURCE)]],
    device uchar *destination [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 tile [[threadgroup_position_in_grid]],
    uint3 thread_position [[thread_position_in_threadgroup]]) {
#if defined(SEISMIC_ELEMENT_U_LAYOUT_MMA16) && !defined(SEISMIC_ELEMENT_U_PLANE_GRIDS)
    // A code role's words of the tile's rows.
    threadgroup uint staged[TILE_ROWS * (CODES_WORDS > HIGH_WORDS ? CODES_WORDS : HIGH_WORDS)];
#endif
    const uint lane = thread_position.x;
    const ulong b = tile.z;
    const ulong row0 = ulong(tile.y) * TILE_ROWS;
    const uint column0 = tile.x * TILE_COLUMNS;
    const ulong packets = (SEISMIC_DIM_K + GROUP - 1) / GROUP;
    const ulong stored_columns = SEISMIC_RESULT_0_ROW_GROUPS * GROUP;
    if (column0 >= stored_columns) return;
    const bool occupied = column0 < packets * GROUP;
    const uint group = column0 / GROUP;
    const uint within = column0 % GROUP;
#if defined(SEISMIC_ELEMENT_U_LAYOUT_MMA16)
    // Every row of a tile is stored; rows beyond N are zero padding.
    if (row0 >= (SEISMIC_DIM_N + TILE_ROWS - 1) / TILE_ROWS * TILE_ROWS) return;
    const ulong rows = TILE_ROWS;
#elif defined(SEISMIC_ELEMENT_U_LAYOUT_ROWS32)
    // Every row of a 32-row tile is stored; rows beyond N are zero padding.
    const ulong rows = TILE_ROWS;
#else
    if (row0 >= SEISMIC_DIM_N) return;
    const ulong rows = min(ulong(TILE_ROWS), SEISMIC_DIM_N - row0);
#endif
    // This lane's row: whether it holds a source packet at this tile, and
    // that packet.
    const ulong n = row0 + lane;
    const bool real = n < SEISMIC_DIM_N && occupied;
    device const uchar *in = source_packet(source, seismic_words, b, real ? n : 0ul, real ? group : 0u);

    // Code planes: the tile's unit of each, every word computed before any
    // is stored (a store could alias the source, and the planes' words read
    // the same values).
    const uint unit = within / TILE_COLUMNS;
    uint words[SEISMIC_CONVERSION_E_U_WORDS];
#if defined(SEISMIC_ELEMENT_U_LAYOUT_MMA16)
    if (lane < TILE_ROWS) {
        SEISMIC_CONVERSION_E_U_UNIT(in, unit, words);
        if (!real)
            for (uint word = 0; word < UNIT_WORDS; ++word) words[word] = 0;
    }
    const uint block = column0 / 64;
    const uint step0 = (column0 % 64) / 16;
#if defined(SEISMIC_ELEMENT_U_PLANE_GRIDS)
    // Grid entries stay whole: the tile sequence is [k-block][row][k-block
    // bytes]; row `lane`'s 32 columns (steps `step0`, `step0 + 1`) are half
    // of its k-block bytes.
    if (lane < TILE_ROWS) {
        const ulong position = (ulong(block) * TILE_ROWS + lane) * 8 * CODES_WORDS + step0 / 2 * 4 * CODES_WORDS;
        store(tile_plane_byte(destination, seismic_words, b, row0, position, SEISMIC_RESULT_0_ROW_CODES_BYTES,
                              SEISMIC_RESULT_0_ROW_CODES),
              words, 4 * CODES_WORDS, ROLE_ALIGNMENT(CODES));
    }
#else
    place_fragments(destination, seismic_words, b, row0, words, CODES_WORDS, SEISMIC_RESULT_0_ROW_CODES,
                    SEISMIC_RESULT_0_ROW_CODES_BYTES, lane, block, step0, staged);
    place_fragments(destination, seismic_words, b, row0, words + CODES_WORDS, HIGH_WORDS, SEISMIC_RESULT_0_ROW_HIGH,
                    SEISMIC_RESULT_0_ROW_HIGH_BYTES, lane, block, step0, staged);
#endif
#else
    // The tile's 32 columns of one row are one unit of each code plane.
    if (lane < rows) {
        SEISMIC_CONVERSION_E_U_UNIT(in, unit, words);
        if (!real)
            for (uint word = 0; word < UNIT_WORDS; ++word) words[word] = 0;
        const ulong stored_unit = column0 / TILE_COLUMNS;
        place_unit(destination, seismic_words, b, n, words, CODES_WORDS, SEISMIC_RESULT_0_ROW_CODES, stored_unit,
                   ROLE_ALIGNMENT(CODES));
        place_unit(destination, seismic_words, b, n, words + CODES_WORDS, HIGH_WORDS, SEISMIC_RESULT_0_ROW_HIGH,
                   stored_unit, ROLE_ALIGNMENT(HIGH));
    }
#endif

    if (lane >= rows) return;

    // Group planes, once per group (at its first tile), per row: the
    // fixed words computed (zero for an unoccupied group), then each group
    // role's share stored.
    if (within == 0) {
        if (real) {
            SEISMIC_CONVERSION_E_U_FIXED(in, words);
        } else {
            for (uint word = 0; word < SEISMIC_CONVERSION_E_U_WORDS; ++word) words[word] = 0;
        }
#if HIGH_GROUP_SHARE
#if defined(SEISMIC_ELEMENT_U_LAYOUT_MMA16)
        place_high(destination, seismic_words, b, row0, lane, group, words + ROLE_WORD(HIGH));
#else
        store(plane_group(destination, seismic_words, b, n, SEISMIC_RESULT_0_ROW_HIGH, SEISMIC_RESULT_0_ROW_HIGH_BYTES,
                          ROLE_BYTES(HIGH), group),
              words + ROLE_WORD(HIGH), ROLE_BYTES(HIGH), SEISMIC_CONVERSION_E_U_HIGH_REST_ALIGNMENT);
#endif
#endif
        store(plane_group(destination, seismic_words, b, n, SEISMIC_RESULT_0_ROW_SCALES,
                          SEISMIC_RESULT_0_ROW_SCALES_BYTES, ROLE_BYTES(SCALES), group),
              words + ROLE_WORD(SCALES), ROLE_BYTES(SCALES), ROLE_ALIGNMENT(SCALES));
        store(plane_group(destination, seismic_words, b, n, SEISMIC_RESULT_0_ROW_SUPERS,
                          SEISMIC_RESULT_0_ROW_SUPERS_BYTES, ROLE_BYTES(SUPERS), group),
              words + ROLE_WORD(SUPERS), ROLE_BYTES(SUPERS), ROLE_ALIGNMENT(SUPERS));
    }

    // Alignment tails of every plane, from its end to the next plane's start
    // (or the row's end), at the row's last stored tile.
    if (column0 + TILE_COLUMNS == stored_columns) {
        zero_tail(destination, seismic_words, b, n, SEISMIC_RESULT_0_ROW_CODES + SEISMIC_RESULT_0_ROW_CODES_BYTES,
                  SEISMIC_RESULT_0_ROW_CODES_TAIL);
        zero_tail(destination, seismic_words, b, n, SEISMIC_RESULT_0_ROW_HIGH + SEISMIC_RESULT_0_ROW_HIGH_BYTES,
                  SEISMIC_RESULT_0_ROW_HIGH_TAIL);
        zero_tail(destination, seismic_words, b, n, SEISMIC_RESULT_0_ROW_SCALES + SEISMIC_RESULT_0_ROW_SCALES_BYTES,
                  SEISMIC_RESULT_0_ROW_SCALES_TAIL);
        zero_tail(destination, seismic_words, b, n, SEISMIC_RESULT_0_ROW_SUPERS + SEISMIC_RESULT_0_ROW_SUPERS_BYTES,
                  SEISMIC_RESULT_0_ROW_SUPERS_TAIL);
    }
}

#endif
