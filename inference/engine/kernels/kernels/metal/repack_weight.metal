// Exact GGUF -> resident conversion (the Seismic registry's registered
// `repack`): E is the GGUF source, U the resident (representation, layout).
// The entry sees every weight as a [B, N, K] view.
//
// The conversion is data: its word form (`seismic_lang::repack::WordTable`,
// whose docs give the table's format), emitted as SEISMIC_CONVERSION_E_U_*,
// gives every 32-bit word of a source packet's record as the OR of pieces,
// runs of source bits or bits of values its postfix ops compute. The record
// is U's packet form (`packet`), or the packet's storage group's share of
// each row plane (row layouts). `repack::unit_record` and `repack::fixed_record`
// (`<seismic/repack.h>`) execute it over this kernel's readers of source packets; the rest of this
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

#include <seismic/repack.h>

#define GROUP ((uint)SEISMIC_ELEMENT_E_LOGICAL_GROUP)
#define TILE_ROWS 16u
#define TILE_COLUMNS 32u
static_assert(GROUP % TILE_COLUMNS == 0, "a tile lies in one group");

// ---- the conversion's executor -------------------------------------------
//
// `repack::unit_record` and `repack::fixed_record` execute the table at
// compile time (`<seismic/repack.h>`), reading source packets through this
// kernel's readers. A code plane's units of
// 32 columns of one structure (a class, `WordSpan`) share one compiled unit,
// the class's representative: a unit names its block by its class at compile
// time and moves its runs by its displacements, read from the table at run
// time.

constexpr constant uint conversion[] = {SEISMIC_CONVERSION_E_U_TABLE};
constexpr constant repack::WordForm form = {
    conversion,
    SEISMIC_CONVERSION_E_U_STACK,
    SEISMIC_CONVERSION_E_U_DIRECTORY,
    SEISMIC_CONVERSION_E_U_CLASSES,
};

// A unit's code words: the codes', then the high codes' (`WordSpan`).
#define CODES_WORDS SEISMIC_CONVERSION_E_U_CODES_UNIT_WORDS
#define HIGH_WORDS SEISMIC_CONVERSION_E_U_HIGH_UNIT_WORDS
#define UNIT_WORDS (CODES_WORDS + HIGH_WORDS)

// LENGTH (1..32) bits of the BYTES bytes from `bytes`, from bit `shift` of
// the first.
template <uint BYTES, uint LENGTH> inline uint shifted_bits(device const uchar *bytes, uint shift) {
    if constexpr (BYTES <= 4) {
        uint value = 0;
        repack::each<0, BYTES>([&](auto byte) { value |= uint(bytes[decltype(byte)::value]) << (8 * decltype(byte)::value); });
        return (value >> shift) & repack::low_bits(LENGTH);
    } else {
        ulong value = 0;
        repack::each<0, BYTES>([&](auto byte) { value |= ulong(bytes[decltype(byte)::value]) << (8 * decltype(byte)::value); });
        return uint(value >> shift) & repack::low_bits(LENGTH);
    }
}

// A source packet `in`, its runs read where the table puts them, moved by
// `delta` bits (a group's member, which spans as many bytes).
struct Source {
    device const uchar *in;

    static constexpr repack::WordForm form() { return ::form; }

    template <uint AT, uint K> uint run(int delta) const {
        constexpr uint piece = repack::table_word(form(), AT), bit = repack::piece_first(piece);
        constexpr uint length = repack::piece_length(piece), bytes = (bit % 8 + length + 7) / 8;
        const int at = int(bit) + delta;
        return shifted_bits<bytes, length>(in + (at >> 3), uint(at) & 7u);
    }
};

// A source packet `in` read as a unit of the class whose slot map is MAP
// (indexed by the order the unit's words visit their runs): each run the
// representative's bytes from a base moved by its slot's byte displacement,
// shifted by its bit displacement more (the run spans as many bytes in every
// unit of its class), `displacements` the unit's (`unit_displacements`);
// a member's moved by `delta` bits more.
template <uint MAP> struct Moved {
    device const uchar *in;
    uint displacements;

    static constexpr repack::WordForm form() { return ::form; }

    template <uint AT, uint K> uint run(int delta) const {
        constexpr uint piece = repack::table_word(form(), AT), bit = repack::piece_first(piece);
        constexpr uint length = repack::piece_length(piece), bytes = (bit % 8 + length + 7) / 8;
        // A slot that moves its runs nowhere reads the representative's; one
        // moving them by whole bytes keeps their shifts.
        constexpr uint entry = repack::table_word(form(), MAP + K), slot = repack::slot_of(entry);
        const int at = int(bit) + delta;
        if constexpr (repack::slot_still(entry)) return shifted_bits<bytes, length>(in + (at >> 3), uint(at) & 7u);
        device const uchar *base = in + repack::displacement_bytes(form(), displacements, slot);
        if constexpr (repack::slot_whole(entry)) return shifted_bits<bytes, length>(base + (at >> 3), uint(at) & 7u);
        // The member's bytes, shifted by the bit displacement more (it stays
        // within them).
        else
            return shifted_bits<bytes, length>(
                base + (at >> 3), (uint(at) & 7u) + repack::displacement_bits(form(), displacements, slot));
    }
};

// Record words [FIRST, FIRST + COUNT) (of one fixed block) of the source
// packet `in`, its source bits the table's, into `words`.
template <uint FIRST, uint COUNT> inline void fixed_record(device const uchar *in, thread uint *words) {
    repack::fixed_record<Source, GROUP / TILE_COLUMNS, FIRST, COUNT>(Source{in}, words);
}

// The words of unit `unit` of the code planes' span into `words`: the
// unit's class's representative's, its runs moved by the unit's
// displacements.
inline void unit_record(device const uchar *in, uint unit, thread uint *words) {
    const uint displacements = repack::unit_displacements(form, unit);
    repack::dispatch<0, SEISMIC_CONVERSION_E_U_CLASS_COUNT>(repack::unit_class(form, unit), [&](auto class_index) {
        constexpr uint c = decltype(class_index)::value;
        using Unit = Moved<repack::class_slot_map(form, c)>;
        repack::unit_record<Unit, repack::class_representative(form, c), UNIT_WORDS>(Unit{in, displacements}, words);
    });
}

// Source packet `packet` of row `n` of matrix `b`.
inline device const uchar *source_packet(device const uchar *source, constant ulong *seismic_words,
                                         ulong b, ulong n, ulong packet) {
    return source + (b * SEISMIC_SOURCE_STRIDE_0 + n * SEISMIC_SOURCE_STRIDE_1 + packet * SEISMIC_SOURCE_STRIDE_2) *
                        SEISMIC_ELEMENT_E_PACKET_SIZE;
}

constexpr uint least(uint a, uint b) { return a < b ? a : b; }

// BYTES bytes of `words` (little-endian) at `out`, aligned to ALIGN, in the
// widest stores (to 16 bytes) both allow. The caller computes every word
// before any is stored.
template <uint BYTES, uint ALIGN = repack::alignment(BYTES), typename W>
inline void store(device uchar *out, thread const W &words) {
    constexpr uint unit = least(repack::alignment(BYTES), ALIGN);
    repack::each<0, BYTES / unit>([&](auto index) {
        constexpr uint at = unit * decltype(index)::value, word = at / 4;
        if constexpr (unit == 16) {
            *reinterpret_cast<device uint4 *>(out + at) = uint4(words[word], words[word + 1], words[word + 2], words[word + 3]);
        } else if constexpr (unit == 8) {
            *reinterpret_cast<device uint2 *>(out + at) = uint2(words[word], words[word + 1]);
        } else if constexpr (unit == 4) {
            *reinterpret_cast<device uint *>(out + at) = words[word];
        } else if constexpr (unit == 2) {
            *reinterpret_cast<device ushort *>(out + at) = ushort(words[word] >> (at % 4 * 8));
        } else {
            out[at] = uchar(words[word] >> (at % 4 * 8));
        }
    });
}

#if defined(SEISMIC_ELEMENT_U_LAYOUT_PACKET)

// ---- packet: the record is the packet form ---------------------------------

// The codes' (packet plane 0's) first record word and words per 32 columns.
#define CODES_WORD SEISMIC_CONVERSION_E_U_CODES_FIRST
static_assert((CODES_WORD + GROUP / TILE_COLUMNS * CODES_WORDS) * 4 <= SEISMIC_ELEMENT_U_PACKET_SIZE,
              "the codes are whole words of a packet");

// Record words [FIRST, FIRST + COUNT), `words`, of the destination packet
// at `out` moved by a multiple of MOVED bytes: their bytes before the
// packet's end.
template <uint FIRST, uint COUNT, uint MOVED = 16, typename W>
inline void store_packet_words(device uchar *out, thread const W &words) {
    constexpr uint begin = least(4 * FIRST, SEISMIC_ELEMENT_U_PACKET_SIZE);
    constexpr uint end = least(4 * (FIRST + COUNT), SEISMIC_ELEMENT_U_PACKET_SIZE);
    constexpr uint align = least(least(repack::alignment(SEISMIC_ELEMENT_U_PACKET_SIZE), repack::alignment(begin)), MOVED);
    store<end - begin, align>(out + begin, words);
}

inline device uchar *destination_packet(device uchar *destination, constant ulong *seismic_words,
                                        ulong b, ulong n, ulong packet) {
    return destination + (b * SEISMIC_RESULT_0_STRIDE_0 + n * SEISMIC_RESULT_0_STRIDE_1 +
                          packet * SEISMIC_RESULT_0_STRIDE_2) * SEISMIC_ELEMENT_U_PACKET_SIZE;
}

// Record words [FIRST, FIRST + COUNT) (their source bits the table's) into
// the destination packet `out`.
template <uint FIRST, uint COUNT> inline void place_fixed(device uchar *out, device const uchar *in) {
    uint words[COUNT > 0 ? COUNT : 1];
    fixed_record<FIRST, COUNT>(in, words);
    store_packet_words<FIRST, COUNT>(out, words);
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
    uint words[UNIT_WORDS];
    unit_record(in, unit, words);
    store_packet_words<CODES_WORD, CODES_WORDS, repack::alignment(4 * CODES_WORDS)>(out + unit * CODES_WORDS * 4, words);
    if (within != 0) return;
    // Every other record word: the other planes, and the gaps (zero).
    constexpr uint codes_end = CODES_WORD + GROUP / TILE_COLUMNS * CODES_WORDS;
    place_fixed<0, CODES_WORD>(out, in);
    place_fixed<codes_end, SEISMIC_CONVERSION_E_U_WORDS - codes_end>(out, in);
}

#else

// ---- rows16 / rows32 / mma16: the record is the row planes' shares ---------

// A role's share: its first record word and bytes.
#define ROLE_WORD(role) SEISMIC_CONVERSION_E_U##_##role##_WORD
#define ROLE_BYTES(role) SEISMIC_CONVERSION_E_U##_##role##_BYTES

// Start of stored row `n` of matrix `b`.
inline device uchar *row_base(device uchar *destination, constant ulong *seismic_words, ulong b, ulong n) {
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
inline device uchar *plane_unit(device uchar *destination, constant ulong *seismic_words, ulong b, ulong n,
                                ulong offset, ulong unit_bytes, ulong unit) {
    return row_base(destination, seismic_words, b, n - n % STORED_TILE) + offset * STORED_TILE +
           (unit * STORED_TILE + n % STORED_TILE) * unit_bytes;
}

// Storage group `group` (of `group_bytes` bytes) of stored row `n` in the
// group plane at row offset `offset` (`row_bytes` a row): a row's groups are
// contiguous in every layout, and a `rows32` tile holds its rows' shares of
// the plane one after another (`PackedRowLayout::packet_bit`).
inline device uchar *plane_group(device uchar *destination, constant ulong *seismic_words, ulong b, ulong n,
                                 ulong offset, ulong row_bytes, ulong group_bytes, ulong group) {
    return row_base(destination, seismic_words, b, n - n % STORED_TILE) + offset * STORED_TILE +
           n % STORED_TILE * row_bytes + group * group_bytes;
}

// The share of stored row `n` in the `length`-bytes-a-row gap that starts at
// row offset `offset` (a plane's alignment tail).
inline device uchar *plane_gap(device uchar *destination, constant ulong *seismic_words, ulong b, ulong n,
                               ulong offset, ulong length) {
    return row_base(destination, seismic_words, b, n - n % STORED_TILE) + offset * STORED_TILE +
           (n % STORED_TILE) * length;
}

inline void zero_bytes(device uchar *out, ulong count) {
    for (ulong i = 0; i < count; ++i) out[i] = 0;
}

// A group role's share (BYTES bytes from record word FIRST) at `out`, or
// zero for an unoccupied group: its words computed, then stored.
template <uint FIRST, uint BYTES> inline void place_group(device uchar *out, device const uchar *in, bool real) {
    constexpr uint count = (BYTES + 3) / 4;
    uint words[count > 0 ? count : 1];
    if (real) {
        fixed_record<FIRST, count>(in, words);
    } else {
        for (uint word = 0; word < count; ++word) words[word] = 0;
    }
    store<BYTES>(out, words);
}

#if defined(SEISMIC_ELEMENT_U_LAYOUT_MMA16)
// Destination of byte `position` of a tile's code-plane sequence V: the
// rows' plane payloads (`row_bytes` each) concatenated in row order.
inline device uchar *tile_plane_byte(device uchar *destination, constant ulong *seismic_words, ulong b,
                                     ulong row0, ulong position, ulong row_bytes, ulong offset) {
    return row_base(destination, seismic_words, b, row0 + position / row_bytes) + offset + position % row_bytes;
}

// A code plane of an mma16 tile, BITS a column, lane `lane`'s row's
// `words` of it: the rows' words staged, then lane `lane`'s fragment slots
// of k16 steps `step0` and `step0 + 1` of k-block `block`
// (`PackedRowLayout::code_bit`): slot `j` of a step holds the code of (row
// lane/4 + 8 (j % 2), column 2 (lane % 4) + 8 ((j % 4) / 2) + j / 4) at bit
// BITS j.
template <uint BITS>
inline void place_fragments(device uchar *destination, constant ulong *seismic_words, ulong b, ulong row0,
                            thread const uint *words, ulong offset, ulong row_bytes, uint lane, uint block,
                            uint step0, threadgroup uint *staged) {
    if constexpr (BITS != 0) {
        static_assert(32 % BITS == 0, "a code lies in one staged word");
        if (lane < TILE_ROWS)
            repack::each<0, BITS>([&](auto word) { staged[lane * BITS + decltype(word)::value] = words[decltype(word)::value]; });
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint value[4] = {0u, 0u, 0u, 0u};
        for (uint half_step = 0; half_step < 2; ++half_step)
            for (uint slot = 0; slot < 8; ++slot) {
                const uint row = lane / 4 + 8 * (slot % 2);
                const uint column = 16 * half_step + 2 * (lane % 4) + 8 * ((slot % 4) / 2) + slot / 4;
                const uint code = (staged[row * BITS + column * BITS / 32] >> (column * BITS % 32)) & repack::low_bits(BITS);
                const uint bit = (8 * half_step + slot) * BITS;
                value[bit / 32] |= code << (bit % 32);
            }
        device uchar *out = tile_plane_byte(destination, seismic_words, b, row0,
                                            (ulong(block) * 32 + lane) * 4 * BITS + BITS * step0, row_bytes, offset);
        store<2 * BITS>(out, value);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
#else
// A code plane's `words` (BITS a column) of the tile's 32 columns of stored
// row `n`: one unit of its plane.
template <uint BITS>
inline void place_unit(device uchar *destination, constant ulong *seismic_words, ulong b, ulong n,
                       thread const uint *words, ulong offset, ulong stored_unit) {
    if constexpr (BITS != 0) {
        store<4 * BITS>(plane_unit(destination, seismic_words, b, n, offset, 4ul * BITS, stored_unit), words);
    }
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
    uint words[UNIT_WORDS];
#if defined(SEISMIC_ELEMENT_U_LAYOUT_MMA16)
    if (lane < TILE_ROWS) {
        unit_record(in, unit, words);
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
        store<4 * CODES_WORDS>(tile_plane_byte(destination, seismic_words, b, row0, position,
                                               SEISMIC_RESULT_0_ROW_CODES_BYTES, SEISMIC_RESULT_0_ROW_CODES),
                               words);
    }
#else
    place_fragments<CODES_WORDS>(destination, seismic_words, b, row0, words, SEISMIC_RESULT_0_ROW_CODES,
                                 SEISMIC_RESULT_0_ROW_CODES_BYTES, lane, block, step0, staged);
    place_fragments<HIGH_WORDS>(destination, seismic_words, b, row0, words + CODES_WORDS, SEISMIC_RESULT_0_ROW_HIGH,
                                SEISMIC_RESULT_0_ROW_HIGH_BYTES, lane, block, step0, staged);
#endif
#else
    // The tile's 32 columns of one row are one unit of each code plane.
    if (lane < rows) {
        unit_record(in, unit, words);
        if (!real)
            for (uint word = 0; word < UNIT_WORDS; ++word) words[word] = 0;
        const ulong stored_unit = column0 / TILE_COLUMNS;
        place_unit<CODES_WORDS>(destination, seismic_words, b, n, words, SEISMIC_RESULT_0_ROW_CODES, stored_unit);
        place_unit<HIGH_WORDS>(destination, seismic_words, b, n, words + CODES_WORDS, SEISMIC_RESULT_0_ROW_HIGH,
                               stored_unit);
    }
#endif

    if (lane >= rows) return;

    // Group planes, once per group (at its first tile), per row.
    if (within == 0) {
        place_group<ROLE_WORD(SCALES), ROLE_BYTES(SCALES)>(
            plane_group(destination, seismic_words, b, n, SEISMIC_RESULT_0_ROW_SCALES,
                        SEISMIC_RESULT_0_ROW_SCALES_BYTES, ROLE_BYTES(SCALES), group),
            in, real);
        place_group<ROLE_WORD(SUPERS), ROLE_BYTES(SUPERS)>(
            plane_group(destination, seismic_words, b, n, SEISMIC_RESULT_0_ROW_SUPERS,
                        SEISMIC_RESULT_0_ROW_SUPERS_BYTES, ROLE_BYTES(SUPERS), group),
            in, real);
    }

    // Alignment tails of every plane, from its end to the next plane's start
    // (or the row's end), at the row's last stored tile.
    if (column0 + TILE_COLUMNS == stored_columns) {
        const ulong offsets[4] = {SEISMIC_RESULT_0_ROW_CODES, SEISMIC_RESULT_0_ROW_HIGH, SEISMIC_RESULT_0_ROW_SCALES,
                                  SEISMIC_RESULT_0_ROW_SUPERS};
        const ulong lengths[4] = {SEISMIC_RESULT_0_ROW_CODES_BYTES, SEISMIC_RESULT_0_ROW_HIGH_BYTES,
                                  SEISMIC_RESULT_0_ROW_SCALES_BYTES, SEISMIC_RESULT_0_ROW_SUPERS_BYTES};
        for (uint role = 0; role < 4; ++role) {
            if (lengths[role] == 0) continue;
            const ulong end = offsets[role] + lengths[role];
            ulong next = SEISMIC_RESULT_0_ROW_STRIDE_BYTES;
            for (uint other = 0; other < 4; ++other)
                if (lengths[other] != 0 && offsets[other] >= end) next = min(next, offsets[other]);
            zero_bytes(plane_gap(destination, seismic_words, b, n, end, next - end), next - end);
        }
    }
}

#endif
