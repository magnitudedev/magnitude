// Exact GGUF -> resident conversion (the Seismic registry's registered
// `repack`), as templates over a binding `B` (`REPACK_BINDING`, below): the
// conversion of one source element into one resident element (representation,
// layout), and the geometry of the two tensors it converts between, so one
// kernel holds any number of conversions (`repack_weight` and
// `routed_convert`: one each). Every view is [B, N, K].
//
// The conversion is data: its word form (`seismic_lang::repack::WordTable`,
// whose docs give the table's format) gives every 32-bit word of a source
// packet's record as the OR of pieces, runs of source bits or bits of values
// its postfix ops compute. The record is the resident packet form (`packet`),
// or the packet's storage group's share of each row plane (row layouts). The
// prefix renders it as code: SEISMIC_CONVERSION_<F>_<T>_UNIT, a unit of the
// code planes' words, and SEISMIC_CONVERSION_<F>_<T>_FIXED, the other words
// (`B::unit`, `B::fixed`); `repack_tile` loads, and places records in the
// resident layout.
//
// `repack_tile` places one row tile, 16 rows x one unit of the code planes'
// words (`tile_columns`, SEISMIC_CONVERSION_<F>_<T>_UNIT_COLUMNS: 32, or more
// where 32 columns are no whole words of a plane), with 32 threads, each
// plane's share of the unit's 32-column chunks where the layout puts them. A
// unit divides its group, so a tile covers a whole packet or part of a
// larger one; the tile at a group's first column also places that group's
// other record words. Stored-but-unoccupied bytes (mma16 padding rows and
// groups, plane alignment tails, packet gaps) are written as zero, so the
// result is fully defined. A matrix spans `tiles_x` x `tiles_y` tiles.
//
// The source is read through an addressing policy `A`: `Strided<B>` reads the
// bound source tensor; an includer that stages the source first
// (lib/streaming/window.h) addresses its copy.

#include <seismic/element.h>

// Helpers are inlined before they are optimized: optimized alone, their
// bodies are general in the constants every call site gives them.
#define REPACK_INLINE __attribute__((always_inline)) inline

namespace repack {

enum class Layout : uint { other, packet, rows16, rows32, mma16 };

constant constexpr uint TILE_ROWS = 16u;
// A tile's columns: one unit of the code planes' words, whole chunks of a
// group (and so of 256 columns).
template <class B> constexpr uint tile_columns() {
    constexpr uint columns = B::unit_columns();
    static_assert(columns % B::chunk_columns() == 0 && B::group() % columns == 0 && 256 % columns == 0,
                  "a unit is whole chunks of a group");
    return columns;
}

// A matrix's tiles: whole 256-column and 32-row blocks (`rows32` stores
// whole 32-row tiles).
template <class B> inline ulong tiles_x(constant ulong *seismic_words) {
    return (B::k(seismic_words) + 255) / 256 * (256 / tile_columns<B>());
}
template <class B> inline ulong tiles_y(constant ulong *seismic_words) {
    return (B::n(seismic_words) + 31) / 32 * 2;
}

// The threadgroup words a tile's lanes exchange (`place_fragments`): an
// mma16 code role's words of the tile's rows, but for grid entries; else
// none, declared as one word.
template <class B> constexpr uint exchange_words() {
    return B::layout() == Layout::mma16 && !B::grids()
               ? TILE_ROWS * (B::codes_words() > B::high_words() ? B::codes_words() : B::high_words())
               : 1u;
}

// Addressing of the bound source tensor: packet `packet` of row `n` of
// matrix `b`.
template <class B> struct Strided {
    template <typename In>
    static REPACK_INLINE In packet(In source, constant ulong *seismic_words, ulong b, ulong n, ulong packet) {
        return source + (b * B::source_stride_0(seismic_words) + n * B::source_stride_1(seismic_words) +
                         packet * B::source_stride_2(seismic_words)) *
                            B::source_packet_size();
    }
};

// `bytes` bytes of `words` (little-endian) at `out`, in stores of `unit`
// bytes (a power of two, to 16, that `bytes` and `out` keep: the record's
// alignments, `SEISMIC_CONVERSION_<F>_<T>_*_ALIGNMENT`); `bytes` and `unit`
// constants, so every branch but one folds away. The caller computes every
// word before any is stored.
REPACK_INLINE void store(device uchar *out, thread const uint *words, uint bytes, uint unit) {
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

// ---- packet: the record is the packet form ---------------------------------

template <class B>
REPACK_INLINE device uchar *destination_packet(device uchar *destination, constant ulong *seismic_words, ulong b,
                                               ulong n, ulong packet) {
    return destination + (b * B::out_stride_0(seismic_words) + n * B::out_stride_1(seismic_words) +
                          packet * B::out_stride_2(seismic_words)) * B::packet_size();
}

template <class B, class A, typename In>
REPACK_INLINE void packet_tile(In source, device uchar *destination, constant ulong *seismic_words, ulong b,
                               uint2 tile, uint lane) {
    constexpr uint group = B::group();
    constexpr uint TILE_COLUMNS = tile_columns<B>();
    // The codes' (packet plane 0's) first record word and words per unit.
    constexpr uint codes_word = B::codes_first();
    constexpr uint codes_words = B::codes_words();
    static_assert((codes_word + group / TILE_COLUMNS * codes_words) * 4 <= B::packet_size(),
                  "the codes are whole words of a packet");
    // Lane l places row l's words: every lane runs one stream of code (a
    // lane's own share of a row would diverge the threadgroup's halves).
    const ulong n = ulong(tile.y) * TILE_ROWS + lane;
    const uint column0 = tile.x * TILE_COLUMNS;
    const ulong packets = (B::k(seismic_words) + group - 1) / group;
    if (lane >= TILE_ROWS || column0 >= packets * group || n >= B::n(seismic_words)) return;
    const uint packet = column0 / group;
    const uint within = column0 % group;
    In in = A::packet(source, seismic_words, b, n, packet);
    device uchar *out = destination_packet<B>(destination, seismic_words, b, n, packet);
    const uint unit = within / TILE_COLUMNS;
    uint words[B::words()];
    B::unit(in, unit, words);
    store(out + 4 * (codes_word + unit * codes_words), words, 4 * codes_words, B::codes_alignment());
    if (within != 0) return;
    // Every other record word: the other planes, and the gaps (zero), to
    // the packet's end.
    constexpr uint codes_end = codes_word + group / TILE_COLUMNS * codes_words;
    constexpr uint end = 4 * B::words() < B::packet_size() ? 4 * B::words() : B::packet_size();
    B::fixed(in, words);
    store(out, words, 4 * codes_word, B::leading_alignment());
    store(out + 4 * codes_end, words + codes_end, end - 4 * codes_end, B::trailing_alignment());
}

// ---- rows16 / rows32 / mma16: the record is the row planes' shares ---------

// Start of stored row `n` of matrix `b`.
template <class B>
REPACK_INLINE device uchar *row_base(device uchar *destination, constant ulong *seismic_words, ulong b, ulong n) {
    return destination + (b * B::out_stride_0(seismic_words) + n * B::out_stride_1(seismic_words)) *
                             B::out_row_stride_bytes(seismic_words);
}

// Rows of one stored row tile: `rows32` interleaves every plane across 32
// rows; `rows16` rows stand alone.
template <class B> constexpr ulong stored_tile() {
    return B::layout() == Layout::rows32 ? B::stored_tile_rows() : 1ul;
}

// Unit `unit` (of `unit_bytes` bytes) of stored row `n` in the code plane at
// row offset `offset`: units follow one another along a `rows16` row; in
// `rows32` the tile's rows of one unit are adjacent
// (`PackedRowLayout::code_bit`).
template <class B>
REPACK_INLINE device uchar *plane_unit(device uchar *destination, constant ulong *seismic_words, ulong b, ulong n,
                                       ulong offset, ulong unit_bytes, ulong unit) {
    constexpr ulong stored = stored_tile<B>();
    return row_base<B>(destination, seismic_words, b, n - n % stored) + offset * stored +
           (unit * stored + n % stored) * unit_bytes;
}

// Storage group `group` (of `group_bytes` bytes) of stored row `n` in the
// group plane at row offset `offset` (`row_bytes` a row): a row's groups are
// contiguous in every layout, and a `rows32` tile holds its rows' shares of
// the plane one after another (`PackedRowLayout::packet_bit`).
template <class B>
REPACK_INLINE device uchar *plane_group(device uchar *destination, constant ulong *seismic_words, ulong b, ulong n,
                                        ulong offset, ulong row_bytes, ulong group_bytes, ulong group) {
    constexpr ulong stored = stored_tile<B>();
    return row_base<B>(destination, seismic_words, b, n - n % stored) + offset * stored + n % stored * row_bytes +
           group * group_bytes;
}

REPACK_INLINE void zero_bytes(device uchar *out, ulong count) {
    for (ulong i = 0; i < count; ++i) out[i] = 0;
}

// Stored row `n`'s share of the `length`-bytes-a-row gap that starts at row
// offset `offset` (a plane's alignment tail), zero: `rows32` tiles hold
// their rows' shares one after another.
template <class B>
REPACK_INLINE void zero_tail(device uchar *destination, constant ulong *seismic_words, ulong b, ulong n,
                             ulong offset, ulong length) {
    constexpr ulong stored = stored_tile<B>();
    zero_bytes(row_base<B>(destination, seismic_words, b, n - n % stored) + offset * stored + (n % stored) * length,
               length);
}

// Destination of byte `position` of a tile's code-plane sequence V: the
// rows' plane payloads (`row_bytes` each) concatenated in row order.
template <class B>
REPACK_INLINE device uchar *tile_plane_byte(device uchar *destination, constant ulong *seismic_words, ulong b,
                                            ulong row0, ulong position, ulong row_bytes, ulong offset) {
    return row_base<B>(destination, seismic_words, b, row0 + position / row_bytes) + offset + position % row_bytes;
}

// A code plane of an mma16 tile, `bits` a column (a constant), lane
// `lane`'s row's `words` of it: the rows' words staged, then lane `lane`'s
// fragment slots of k16 steps `step0` and `step0 + 1` of k-block `block`
// (`PackedRowLayout::code_bit`): slot `j` of a step holds the code of
// (`B::fragment_row`, `B::fragment_column`) at bit `bits` j, stored in units
// of `store_unit`. The tile's 32 threads are the whole threadgroup.
template <class B>
REPACK_INLINE void place_fragments(device uchar *destination, constant ulong *seismic_words, ulong b, ulong row0,
                                   thread const uint *words, uint bits, ulong offset, ulong row_bytes, uint lane,
                                   uint block, uint step0, uint store_unit, threadgroup uint *staged) {
    if (bits == 0) return;
    if (lane < TILE_ROWS)
        for (uint word = 0; word < bits; ++word) staged[lane * bits + word] = words[word];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint value[4] = {0u, 0u, 0u, 0u};
    for (uint half_step = 0; half_step < 2; ++half_step)
        for (uint slot = 0; slot < B::step_slots(); ++slot) {
            const uint row = B::fragment_row(lane, slot);
            const uint column = B::step_columns() * half_step + B::fragment_column(lane, slot);
            const uint code = (staged[row * bits + column * bits / 32] >> (column * bits % 32)) & ((1u << bits) - 1u);
            const uint bit = (B::step_slots() * half_step + slot) * bits;
            value[bit / 32] |= code << (bit % 32);
        }
    device uchar *out = tile_plane_byte<B>(destination, seismic_words, b, row0,
                                           B::fragment_byte(block, lane, bits, step0), row_bytes, offset);
    store(out, value, 2 * bits, store_unit);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// A code plane's `words` (`bits` of them, a constant) of a unit of stored
// row `n`: its `chunks` chunks of `chunk` bytes, chunk `c` its bytes [c chunk,
// (c + 1) chunk), placed as units of the plane, the first at `stored_unit`,
// each stored in units of `stores` (its role's chunk bytes and alignment).
template <class B>
REPACK_INLINE void place_unit(device uchar *destination, constant ulong *seismic_words, ulong b, ulong n,
                              thread const uint *words, uint bits, ulong offset, ulong stored_unit, uint chunk,
                              uint stores) {
    constexpr uint chunks = tile_columns<B>() / B::chunk_columns();
    if (bits == 0) return;
    for (uint c = 0; c < chunks; ++c) {
        device uchar *out = plane_unit<B>(destination, seismic_words, b, n, offset, ulong(chunk), stored_unit + c);
        if (chunk % 4 == 0) {
            store(out, words + c * chunk / 4, chunk, stores);
        } else {
            const uint part = words[c * chunk / 4] >> (c * chunk % 4 * 8);
            store(out, &part, chunk, stores);
        }
    }
}

template <class B, class A, typename In>
REPACK_INLINE void row_tile(In source, device uchar *destination, constant ulong *seismic_words, ulong b,
                            uint2 tile, uint lane, threadgroup uint *staged) {
    constexpr uint group = B::group();
    constexpr uint TILE_COLUMNS = tile_columns<B>();
    // A unit's code words: the codes', then the high codes' (`WordSpan`).
    constexpr uint codes_words = B::codes_words();
    constexpr uint high_words = B::high_words();
    constexpr uint unit_words = codes_words + high_words;
    if constexpr (B::layout() == Layout::mma16) {
        static_assert(TILE_COLUMNS == B::chunk_columns(), "an mma16 tile's code planes are 32-column units");
    }
    if constexpr (B::layout() == Layout::mma16 && !B::grids()) {
        static_assert(32 % codes_words == 0 && (high_words == 0 || 32 % high_words == 0),
                      "a code lies in one staged word");
    }
    const ulong row0 = ulong(tile.y) * TILE_ROWS;
    const uint column0 = tile.x * TILE_COLUMNS;
    const ulong dim_n = B::n(seismic_words);
    const ulong packets = (B::k(seismic_words) + group - 1) / group;
    const ulong stored_columns = B::out_row_groups(seismic_words) * group;
    if (column0 >= stored_columns) return;
    const bool occupied = column0 < packets * group;
    const uint group_index = column0 / group;
    const uint within = column0 % group;
    ulong rows;
    if constexpr (B::layout() == Layout::mma16) {
        // Every row of a tile is stored; rows beyond N are zero padding.
        if (row0 >= (dim_n + TILE_ROWS - 1) / TILE_ROWS * TILE_ROWS) return;
        rows = TILE_ROWS;
    } else if constexpr (B::layout() == Layout::rows32) {
        // Every row of a 32-row tile is stored; rows beyond N are zero padding.
        rows = TILE_ROWS;
    } else {
        if (row0 >= dim_n) return;
        rows = min(ulong(TILE_ROWS), dim_n - row0);
    }
    // This lane's row: whether it holds a source packet at this tile, and
    // that packet.
    const ulong n = row0 + lane;
    const bool real = n < dim_n && occupied;
    In in = A::packet(source, seismic_words, b, real ? n : 0ul, real ? group_index : 0u);

    // Code planes: the tile's unit of each, every word computed before any
    // is stored (a store could alias the source, and the planes' words read
    // the same values).
    const uint unit = within / TILE_COLUMNS;
    uint words[B::words()];
    if constexpr (B::layout() == Layout::mma16) {
        if (lane < TILE_ROWS) {
            B::unit(in, unit, words);
            if (!real)
                for (uint word = 0; word < unit_words; ++word) words[word] = 0;
        }
        const uint block = column0 / B::kblock_columns();
        const uint step0 = (column0 % B::kblock_columns()) / B::step_columns();
        if constexpr (B::grids()) {
            // Grid entries stay whole: the tile sequence is [k-block][row][k-block
            // bytes]; row `lane`'s 32 columns (steps `step0`, `step0 + 1`) are half
            // of its k-block bytes.
            if (lane < TILE_ROWS) {
                const ulong position = (ulong(block) * TILE_ROWS + lane) * B::kblock_bytes() + step0 / 2 * 4 * codes_words;
                store(tile_plane_byte<B>(destination, seismic_words, b, row0, position,
                                         B::out_codes_bytes(seismic_words), B::out_codes(seismic_words)),
                      words, 4 * codes_words, B::codes_alignment());
            }
        } else {
            place_fragments<B>(destination, seismic_words, b, row0, words, codes_words, B::out_codes(seismic_words),
                               B::out_codes_bytes(seismic_words), lane, block, step0, B::codes_fragment_alignment(),
                               staged);
            place_fragments<B>(destination, seismic_words, b, row0, words + codes_words, high_words,
                               B::out_high(seismic_words), B::out_high_bytes(seismic_words), lane, block, step0,
                               B::high_fragment_alignment(), staged);
        }
    } else {
        // A unit of each code plane of one row.
        if (lane < rows) {
            B::unit(in, unit, words);
            if (!real)
                for (uint word = 0; word < unit_words; ++word) words[word] = 0;
            const ulong stored_unit = column0 / B::chunk_columns();
            place_unit<B>(destination, seismic_words, b, n, words, codes_words, B::out_codes(seismic_words),
                          stored_unit, B::codes_chunk_bytes(), B::codes_chunk_alignment());
            place_unit<B>(destination, seismic_words, b, n, words + codes_words, high_words,
                          B::out_high(seismic_words), stored_unit, B::high_chunk_bytes(), B::high_chunk_alignment());
        }
    }

    if (lane >= rows) return;

    // Group planes, once per group (at its first tile), per row: the
    // fixed words computed (zero for an unoccupied group), then each group
    // role's share stored.
    if (within == 0) {
        if (real) {
            B::fixed(in, words);
        } else {
            for (uint word = 0; word < B::words(); ++word) words[word] = 0;
        }
        store(plane_group<B>(destination, seismic_words, b, n, B::out_scales(seismic_words),
                             B::out_scales_bytes(seismic_words), B::scales_bytes(), group_index),
              words + B::scales_word(), B::scales_bytes(), B::scales_alignment());
        store(plane_group<B>(destination, seismic_words, b, n, B::out_supers(seismic_words),
                             B::out_supers_bytes(seismic_words), B::supers_bytes(), group_index),
              words + B::supers_word(), B::supers_bytes(), B::supers_alignment());
    }

    // Alignment tails of every plane, from its end to the next plane's start
    // (or the row's end), at the row's last stored tile.
    if (column0 + TILE_COLUMNS == stored_columns) {
        zero_tail<B>(destination, seismic_words, b, n, B::out_codes(seismic_words) + B::out_codes_bytes(seismic_words),
                     B::out_codes_tail(seismic_words));
        zero_tail<B>(destination, seismic_words, b, n, B::out_high(seismic_words) + B::out_high_bytes(seismic_words),
                     B::out_high_tail(seismic_words));
        zero_tail<B>(destination, seismic_words, b, n,
                     B::out_scales(seismic_words) + B::out_scales_bytes(seismic_words),
                     B::out_scales_tail(seismic_words));
        zero_tail<B>(destination, seismic_words, b, n,
                     B::out_supers(seismic_words) + B::out_supers_bytes(seismic_words),
                     B::out_supers_tail(seismic_words));
    }
}

// Place row tile `tile` (columns x, rows y) of matrix `b`, reading the
// source through `A`; `lane` is the thread's index in its 32-thread group,
// `staged` the group's `exchange_words` exchange words. A tile outside the
// matrix's stored extent does nothing.
template <class B, class A = Strided<B>, typename In>
REPACK_INLINE void repack_tile(In source, device uchar *destination, constant ulong *seismic_words, ulong b,
                               uint2 tile, uint lane, threadgroup uint *staged) {
    static_assert(B::converts(), "a repack binding is a registered GGUF-to-resident conversion");
    static_assert(B::layout() != Layout::other, "repack writes packet, rows16, mma16 or rows32 storage");
    static_assert(!B::records(), "repack does not write mma16 records (CUDA's iq2xxs storage)");
    if constexpr (B::layout() == Layout::packet)
        packet_tile<B, A>(source, destination, seismic_words, b, tile, lane);
    else
        row_tile<B, A>(source, destination, seismic_words, b, tile, lane, staged);
}

} // namespace repack

// ---- bindings ----------------------------------------------------------------
//
// `REPACK_BINDING(Name, C, S, D, DIM_N, DIM_K)` defines binding `Name` of
// conversion `C` (the prefix's SEISMIC_CONVERSION_<F>_<T> of the source
// element F into the resident element T) of tensor `S` (ABI prefix such as
// `SEISMIC_SOURCE`) into tensor `D` (such as `SEISMIC_RESULT_0`), both
// [*, N, K] with N = DIM_N and K = DIM_K. A tensor's prefix carries its
// element's facts and geometry; `C`'s macros are the conversion's constants
// and code (`unit`, `fixed`). `C` is defined (as 1) when the conversion is
// registered, so the binding names its macros only by pasting onto it (an
// operand of `##` is not expanded) and reads it bare as the probe. A macro a
// binding does not define is never named (`REPACK_WHEN` and `REPACK_IF` keep
// only the branch their probe selects). A binding of two dense float tensors
// has no conversion (`dense()`: the includer converts each element).

#define REPACK_PICK_1(yes, no) yes
#define REPACK_PICK_0(yes, no) no
#define REPACK_PICK_(flag) REPACK_PICK_##flag
#define REPACK_PICK(flag) REPACK_PICK_(flag)
// `yes` when `<P><suffix>` is defined as 1, else `no`.
#define REPACK_WHEN(P, suffix, yes, no) REPACK_PICK(ELEMENT_HAS(P, suffix))(yes, no)
// The tokens that follow, in parentheses, when `flag` is 1; else nothing.
#define REPACK_KEEP(...) __VA_ARGS__
#define REPACK_DROP(...)
#define REPACK_IF_1 REPACK_KEEP
#define REPACK_IF_0 REPACK_DROP
#define REPACK_IF(flag) ELEMENT_CAT(REPACK_IF_, flag)
// A row-layout geometry word, or 0 for a packet layout or a dense
// representation (which have no row words).
#define REPACK_ROW_WORD(D, WORD)                                                                            \
    REPACK_WHEN(D, _KIND_PACKED, REPACK_WHEN(D, _LAYOUT_PACKET, 0ul, ulong(ELEMENT_CAT(D, WORD))), 0ul)
// The bytes of one unit of tensor P's strides: a row layout's stored row,
// any other's packet (a dense element's packet is one element).
#define REPACK_STRIDE_BYTES(P)                                                                              \
    REPACK_WHEN(P, _KIND_PACKED,                                                                            \
        REPACK_WHEN(P, _LAYOUT_PACKET, ulong(ELEMENT_CAT(P, _PACKET_SIZE)), ulong(ELEMENT_CAT(P, _ROW_STRIDE_BYTES))), \
        ulong(ELEMENT_CAT(P, _PACKET_SIZE)))
// The bytes between the matrices of a stacked [*, N, K] tensor P (a record).
#define REPACK_MATRIX_BYTES(P) (ulong(ELEMENT_CAT(P, _STRIDE_0)) * REPACK_STRIDE_BYTES(P))

#define REPACK_BINDING(Name, C, S, D, DIM_N, DIM_K)                                                         \
    struct Name {                                                                                           \
        /* A dense float source and destination: the element conversion (`streaming::convert_item`). */   \
        static constexpr bool dense() { return ELEMENT_HAS(S, _KIND_DENSE) && ELEMENT_HAS(D, _KIND_DENSE); } \
        static constexpr int source_kind() { return ELEMENT_KIND(S); }                                      \
        static constexpr int destination_kind() { return ELEMENT_KIND(D); }                                 \
        static constexpr uint group() { return uint(ELEMENT_CAT(S, _LOGICAL_GROUP)); }                      \
        static constexpr uint source_packet_size() { return uint(ELEMENT_CAT(S, _PACKET_SIZE)); }           \
        static constexpr repack::Layout layout() {                                                          \
            return ELEMENT_HAS(D, _LAYOUT_PACKET)   ? repack::Layout::packet                                \
                 : ELEMENT_HAS(D, _LAYOUT_ROWS16)   ? repack::Layout::rows16                                \
                 : ELEMENT_HAS(D, _LAYOUT_ROWS32)   ? repack::Layout::rows32                                \
                 : ELEMENT_HAS(D, _LAYOUT_MMA16)    ? repack::Layout::mma16                                 \
                                                    : repack::Layout::other;                                \
        }                                                                                                   \
        static constexpr bool grids() { return ELEMENT_HAS(D, _PLANE_GRIDS); }                              \
        /* A rows32 tile's rows; mma16's facts and placement rules (`PackedRowLayout`), 1 or 0 in */        \
        /* another layout, whose branches never reach them. */                                              \
        static constexpr ulong stored_tile_rows() {                                                         \
            return REPACK_WHEN(D, _LAYOUT_ROWS32, ulong(ELEMENT_CAT(D, _TILE_ROWS)), 1ul);                  \
        }                                                                                                   \
        static constexpr uint kblock_columns() {                                                            \
            return REPACK_WHEN(D, _LAYOUT_MMA16, uint(ELEMENT_CAT(D, _MMA_KBLOCK)), 1u);                    \
        }                                                                                                   \
        static constexpr uint step_columns() {                                                              \
            return REPACK_WHEN(D, _LAYOUT_MMA16, uint(ELEMENT_CAT(D, _MMA_STEP)), 1u);                      \
        }                                                                                                   \
        static constexpr uint step_slots() {                                                                \
            return REPACK_WHEN(D, _LAYOUT_MMA16, uint(ELEMENT_CAT(D, _MMA_STEP_SLOTS)), 0u);                \
        }                                                                                                   \
        static REPACK_INLINE uint fragment_row(uint lane, uint slot) {                                      \
            return REPACK_WHEN(D, _LAYOUT_MMA16, ELEMENT_CAT(D, _MMA_FRAGMENT_ROW)(lane, slot), 0u);        \
        }                                                                                                   \
        static REPACK_INLINE uint fragment_column(uint lane, uint slot) {                                   \
            return REPACK_WHEN(D, _LAYOUT_MMA16, ELEMENT_CAT(D, _MMA_FRAGMENT_COLUMN)(lane, slot), 0u);     \
        }                                                                                                   \
        static REPACK_INLINE ulong fragment_byte(ulong block, ulong lane, ulong bits, ulong step) {         \
            return REPACK_WHEN(D, _LAYOUT_MMA16, ELEMENT_CAT(D, _MMA_FRAGMENT_BYTE)(block, lane, bits, step), 0ul); \
        }                                                                                                   \
        static constexpr bool records() { return ELEMENT_HAS(D, _PLANE_RECORDS); }                          \
        static constexpr uint packet_size() {                                                               \
            return REPACK_WHEN(D, _LAYOUT_PACKET, uint(ELEMENT_CAT(D, _PACKET_SIZE)), 0u);                  \
        }                                                                                                   \
        static ulong n(constant ulong *seismic_words) { return ulong(DIM_N); }                              \
        static ulong k(constant ulong *seismic_words) { return ulong(DIM_K); }                              \
        static ulong source_stride_0(constant ulong *seismic_words) { return ulong(ELEMENT_CAT(S, _STRIDE_0)); } \
        static ulong source_stride_1(constant ulong *seismic_words) { return ulong(ELEMENT_CAT(S, _STRIDE_1)); } \
        static ulong source_stride_2(constant ulong *seismic_words) { return ulong(ELEMENT_CAT(S, _STRIDE_2)); } \
        static ulong out_stride_0(constant ulong *seismic_words) { return ulong(ELEMENT_CAT(D, _STRIDE_0)); }    \
        static ulong out_stride_1(constant ulong *seismic_words) { return ulong(ELEMENT_CAT(D, _STRIDE_1)); }    \
        static ulong out_stride_2(constant ulong *seismic_words) { return ulong(ELEMENT_CAT(D, _STRIDE_2)); }    \
        /* Row layouts: the row and each role's row plane (offset, bytes, alignment tail). */              \
        static ulong out_row_stride_bytes(constant ulong *seismic_words) {                                  \
            return REPACK_ROW_WORD(D, _ROW_STRIDE_BYTES);                                                   \
        }                                                                                                   \
        static ulong out_row_groups(constant ulong *seismic_words) { return REPACK_ROW_WORD(D, _ROW_GROUPS); } \
        static ulong out_codes(constant ulong *seismic_words) { return REPACK_ROW_WORD(D, _ROW_CODES); }    \
        static ulong out_codes_bytes(constant ulong *seismic_words) { return REPACK_ROW_WORD(D, _ROW_CODES_BYTES); } \
        static ulong out_codes_tail(constant ulong *seismic_words) { return REPACK_ROW_WORD(D, _ROW_CODES_TAIL); } \
        static ulong out_high(constant ulong *seismic_words) { return REPACK_ROW_WORD(D, _ROW_HIGH); }      \
        static ulong out_high_bytes(constant ulong *seismic_words) { return REPACK_ROW_WORD(D, _ROW_HIGH_BYTES); } \
        static ulong out_high_tail(constant ulong *seismic_words) { return REPACK_ROW_WORD(D, _ROW_HIGH_TAIL); } \
        static ulong out_scales(constant ulong *seismic_words) { return REPACK_ROW_WORD(D, _ROW_SCALES); }  \
        static ulong out_scales_bytes(constant ulong *seismic_words) {                                      \
            return REPACK_ROW_WORD(D, _ROW_SCALES_BYTES);                                                   \
        }                                                                                                   \
        static ulong out_scales_tail(constant ulong *seismic_words) { return REPACK_ROW_WORD(D, _ROW_SCALES_TAIL); } \
        static ulong out_supers(constant ulong *seismic_words) { return REPACK_ROW_WORD(D, _ROW_SUPERS); }  \
        static ulong out_supers_bytes(constant ulong *seismic_words) {                                      \
            return REPACK_ROW_WORD(D, _ROW_SUPERS_BYTES);                                                   \
        }                                                                                                   \
        static ulong out_supers_tail(constant ulong *seismic_words) { return REPACK_ROW_WORD(D, _ROW_SUPERS_TAIL); } \
        /* The registered conversion C. */                                                                 \
        static constexpr bool converts() { return ELEMENT_IS_ONE(C); }                                      \
        REPACK_IF(ELEMENT_IS_ONE(C))(                                                                       \
            /* The record's words; a unit's code words per code role, and the codes' first record word. */ \
            static constexpr uint words() { return C##_WORDS; }                                             \
            static constexpr uint codes_words() { return C##_CODES_UNIT_WORDS; }                            \
            static constexpr uint high_words() { return C##_HIGH_UNIT_WORDS; }                              \
            static constexpr uint unit_columns() { return C##_UNIT_COLUMNS; }                               \
            static constexpr uint codes_first() { return C##_CODES_FIRST; }                                 \
            /* The alignments the record's bytes are stored in. */                                         \
            static constexpr uint codes_alignment() { return C##_CODES_ALIGNMENT; }                         \
            static constexpr uint leading_alignment() {                                                     \
                return REPACK_WHEN(D, _LAYOUT_PACKET, uint(C##_LEADING_ALIGNMENT), 0u);                     \
            }                                                                                               \
            static constexpr uint trailing_alignment() {                                                    \
                return REPACK_WHEN(D, _LAYOUT_PACKET, uint(C##_TRAILING_ALIGNMENT), 0u);                    \
            }                                                                                               \
            /* Row layouts' code units: a 32-column chunk's columns, bytes and alignment per code */       \
            /* role; mma16's k-block bytes and fragment store units. */                                    \
            static constexpr uint chunk_columns() {                                                         \
                return REPACK_WHEN(D, _LAYOUT_PACKET, 1u, uint(C##_CHUNK_COLUMNS));                         \
            }                                                                                               \
            static constexpr uint codes_chunk_bytes() {                                                     \
                return REPACK_WHEN(D, _LAYOUT_PACKET, 0u, uint(C##_CODES_CHUNK_BYTES));                     \
            }                                                                                               \
            static constexpr uint codes_chunk_alignment() {                                                 \
                return REPACK_WHEN(D, _LAYOUT_PACKET, 0u, uint(C##_CODES_CHUNK_ALIGNMENT));                 \
            }                                                                                               \
            static constexpr uint high_chunk_bytes() {                                                      \
                return REPACK_WHEN(D, _LAYOUT_PACKET, 0u, uint(C##_HIGH_CHUNK_BYTES));                      \
            }                                                                                               \
            static constexpr uint high_chunk_alignment() {                                                  \
                return REPACK_WHEN(D, _LAYOUT_PACKET, 0u, uint(C##_HIGH_CHUNK_ALIGNMENT));                  \
            }                                                                                               \
            static constexpr uint kblock_bytes() {                                                          \
                return REPACK_WHEN(D, _LAYOUT_MMA16, uint(C##_KBLOCK_BYTES), 0u);                           \
            }                                                                                               \
            static constexpr uint codes_fragment_alignment() {                                              \
                return REPACK_WHEN(D, _LAYOUT_MMA16, uint(C##_CODES_FRAGMENT_ALIGNMENT), 0u);               \
            }                                                                                               \
            static constexpr uint high_fragment_alignment() {                                               \
                return REPACK_WHEN(D, _LAYOUT_MMA16, uint(C##_HIGH_FRAGMENT_ALIGNMENT), 0u);                \
            }                                                                                               \
            /* A group role's share: its first record word, bytes and alignment. */                        \
            static constexpr uint scales_word() { return REPACK_WHEN(D, _LAYOUT_PACKET, 0u, uint(C##_SCALES_WORD)); } \
            static constexpr uint scales_bytes() { return REPACK_WHEN(D, _LAYOUT_PACKET, 0u, uint(C##_SCALES_BYTES)); } \
            static constexpr uint scales_alignment() {                                                      \
                return REPACK_WHEN(D, _LAYOUT_PACKET, 0u, uint(C##_SCALES_ALIGNMENT));                      \
            }                                                                                               \
            static constexpr uint supers_word() { return REPACK_WHEN(D, _LAYOUT_PACKET, 0u, uint(C##_SUPERS_WORD)); } \
            static constexpr uint supers_bytes() { return REPACK_WHEN(D, _LAYOUT_PACKET, 0u, uint(C##_SUPERS_BYTES)); } \
            static constexpr uint supers_alignment() {                                                      \
                return REPACK_WHEN(D, _LAYOUT_PACKET, 0u, uint(C##_SUPERS_ALIGNMENT));                      \
            }                                                                                               \
            /* The record's code words of unit `index` of the source packet at `in`; its other words. */   \
            template <typename In>                                                                          \
            static REPACK_INLINE void unit(In in, uint index, thread uint *words) {                         \
                C##_UNIT(in, index, words);                                                                 \
            }                                                                                               \
            template <typename In> static REPACK_INLINE void fixed(In in, thread uint *words) {             \
                C##_FIXED(in, words);                                                                       \
            }                                                                                               \
        )                                                                                                   \
    };
