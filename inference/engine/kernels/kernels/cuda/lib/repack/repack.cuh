// Exact GGUF -> resident conversion (the Seismic registry's registered
// `repack`), as templates over a binding `B`, so one kernel holds any number
// of conversions (`repack_weight`: one; the streamed fill and convert: one
// per tensor or source). A binding (`REPACK_BINDING`, below) names a source
// element F (a GGUF format) and a resident element T (representation,
// layout), whose registered conversion the prefix rendered as
// SEISMIC_CONVERSION_<F>_<T>_*, and two tensors' ABI prefixes for the
// geometry (N, K, strides, row planes). Every view is [B, N, K].
//
// The conversion is data: its word form (`seismic_lang::repack::WordTable`,
// whose docs give the table's format) gives every 32-bit word of a source
// packet's record as the OR of pieces, runs of source bits or bits of values
// its postfix ops compute. The record is T's packet form (`packet`), or the
// packet's storage group's share of each row plane (row layouts). The prefix
// renders it as code: SEISMIC_CONVERSION_<F>_<T>_UNIT, a unit of the code
// planes' words, and SEISMIC_CONVERSION_<F>_<T>_FIXED, the other words, each
// staging the source bytes it reads through `Stage` (the fixed words' loaded
// apart, by SEISMIC_CONVERSION_<F>_<T>_FIXED_LOAD); the rest of this header
// places records in T's layout.
//
// A tile is 16 rows x one unit of the code planes' columns (`WordSpan`,
// SEISMIC_CONVERSION_<F>_<T>_UNIT_COLUMNS: 32, or a multiple where the code
// planes' words need more), placed (`convert_tile`) by lanes of a warp, a
// lane per row, so they run one stream of record words (only their data
// differ): a whole warp (`repack_weight`: the other 16 lanes help stage and
// store), or half a warp, its other half placing another tile at once
// (lib/streaming/window.cuh), but where code planes are placed as mma
// fragments, whose lanes are a whole warp's (`Shape::WARP_TILES`). Every
// group is a multiple of the unit's columns: a tile's words are its class's
// representative's, their runs moved by the unit's displacements. The tile
// at a group's first column also places the group's other record words.
// Stored-but-unoccupied bytes (mma16 padding rows and groups, plane
// alignment tails, packet gaps) are written as zero, so the result is fully
// defined. A matrix spans `tiles_x` x `tiles_y` tiles.
//
// The source is read through an addressing policy, an object of type `A`:
// `Strided<B>` reads the bound source tensor; an includer that stages the
// source first (lib/streaming/window.cuh) addresses its copy. Either gives a
// tile's first row's source packet and the bytes between its rows. The
// result is written through a policy `O`: `Rows<B>` places rows in the bound
// destination tensor; an includer that writes a row tile out whole
// (window.cuh's `Mirror`) places them in its copy. Either gives a stored
// row's start.

#include <seismic/element.cuh>

namespace repack {

using element::u16;
using element::u32;
using element::u64;
using element::u8;

// The record's row-plane roles (`RowPlaneRole`).
enum class Role : u32 { codes, high, scales, supers };

// `words` words padded to an odd number (each lane's row of a shared array of
// rows starts in its own bank).
__host__ __device__ constexpr u32 odd_words(u32 words) { return words % 2 == 0 ? words + 1 : words; }

// A binding's tile and record constants.
template <class B> struct Shape {
    static_assert(B::converts(), "the binding is a registered GGUF-to-resident conversion");
    static_assert(B::external() && B::packed(), "repack converts an external GGUF source into packed resident storage");
    static_assert(B::packet() || B::rows16() || B::mma16(), "repack writes packet, rows16 or mma16 storage");

    static constexpr u32 GROUP = B::group();
    // The tile's rows, and the lanes the conversion's tile plans deal its
    // staged units to in turn (`SEISMIC_CONVERSION_<F>_<T>_TILE_ROWS`,
    // `_LANES`): a lane a row. A tile's placing lanes (`convert_tile`'s
    // LANES: those, or a whole warp) take them in turn (`Stage::past`).
    static constexpr u32 TILE_ROWS = B::tile_rows();
    static constexpr u32 LANES = B::lanes();
    static_assert(LANES == TILE_ROWS, "a tile plan deals a unit a row in turn");
    // A tile's columns: one unit of the code planes' words, dividing every
    // group and so 256.
    static constexpr u32 TILE_COLUMNS = B::unit_columns();
    static constexpr u32 UNITS = GROUP / TILE_COLUMNS;
    static_assert(GROUP % TILE_COLUMNS == 0 && 256 % TILE_COLUMNS == 0, "a tile lies in one group");

    // A unit's code words: the codes', then the high codes' (`WordSpan`).
    static constexpr u32 CODES_WORDS = B::codes_unit_words();
    static constexpr u32 HIGH_WORDS = B::high_unit_words();
    static constexpr u32 UNIT_WORDS = CODES_WORDS + HIGH_WORDS;
    static constexpr u32 WORDS = B::words();

    // A staged row's words (`SEISMIC_CONVERSION_<F>_<T>_STAGE_WORDS`), padded.
    static constexpr u32 STAGE_WORDS = odd_words(B::stage_words());

    // A lane's loaded units of the fixed words (an array's length: at least 1).
    static constexpr u32 FIXED_LOADS = B::fixed_loads() > 0 ? B::fixed_loads() : 1;

    // Packet layout: the record's codes, and its fixed words (the record's but
    // its group's codes).
    static constexpr u32 CODES_WORD = B::codes_first();
    static constexpr u32 CODES_END = CODES_WORD + UNITS * CODES_WORDS;
    static constexpr u32 FIXED_WORDS = WORDS - (CODES_END - CODES_WORD);

    // A packet tile's record words are staged per row before being stored, in
    // the source stage's memory, from staged word 0: its unit's codes, then
    // (at a group's first tile) its fixed words. Only those: a block's shared
    // memory bounds the blocks (one warp each) an SM keeps resident, and
    // staging every record word (iq3_s's 53: 4416 bytes a block) fits 22
    // blocks in an sm_120 SM's 100 KB, of the 32 it would otherwise run.
    static constexpr u32 RECORD_STRIDE = odd_words(CODES_WORDS > FIXED_WORDS ? CODES_WORDS : FIXED_WORDS);

    // Rows of an mma16 tile (row layouts): a row tile's.
    static constexpr u32 MMA_ROWS = B::mma_rows();

    // mma16 code planes are placed as fragments, from the code roles' words of
    // the warp's rows staged after the source stage.
    static constexpr bool FRAGMENTS = B::mma16() && !B::grids() && !B::records();
    static_assert(!FRAGMENTS || TILE_COLUMNS == 32, "fragment units of 32 columns");

    // A tile's shared words: the source stage (packet: then its record
    // words), then (fragments) the staged code words.
    static constexpr u32 STAGE_SHARE =
        TILE_ROWS * (B::packet() && RECORD_STRIDE > STAGE_WORDS ? RECORD_STRIDE : STAGE_WORDS);
    static constexpr u32 SHARED_WORDS = STAGE_SHARE + (FRAGMENTS ? TILE_ROWS * UNIT_WORDS : 0u);

    // The most tiles a warp places at once: one a half warp (a lane a row of
    // each), but where code planes are placed as mma fragments, whose lanes
    // are a whole warp's (`B::fragment_row`).
    static constexpr u32 WARP_TILES = FRAGMENTS ? 1u : 32u / TILE_ROWS;
    static_assert(32 % TILE_ROWS == 0, "a warp's lanes are whole tiles' rows");
};

// The shared words of a warp placing `Shape::WARP_TILES` tiles of binding `B`
// at once (`SHARED_WORDS` each), or 0 when it converts nothing (a dense
// binding).
template <class B> __host__ __device__ constexpr u32 shared_words() {
    if constexpr (B::converts())
        return Shape<B>::WARP_TILES * Shape<B>::SHARED_WORDS;
    else
        return 0u;
}

// The lanes of the calling thread's warp that place its tile, `LANES` of
// them: the warp's aligned share of LANES lanes that holds the thread.
template <u32 LANES> __device__ __forceinline__ u32 lanes_mask() {
    static_assert(LANES > 0 && 32 % LANES == 0, "a tile's lanes are an aligned share of a warp");
    return LANES == 32 ? 0xffffffffu : ((1u << LANES) - 1u) << (threadIdx.x % 32u / LANES * LANES);
}

// Tiles of a matrix: `repack_weight`'s launch (import.seismic). A tile past
// the stored extent places nothing.
template <class B> __device__ __forceinline__ u64 tiles_x(const seismic_words_t &seismic_words_value) {
    return (B::k(seismic_words_value) + 255) / 256 * (256 / Shape<B>::TILE_COLUMNS);
}
template <class B> __device__ __forceinline__ u64 tiles_y(const seismic_words_t &seismic_words_value) {
    return (B::n(seismic_words_value) + Shape<B>::TILE_ROWS - 1) / Shape<B>::TILE_ROWS;
}

// A planar source's geometry (`ExternalPacketLayout::planes`): each matrix
// its codes plane (a row's packets' codes after the row before's), then its
// grid of one scale per tile, a tile row's after the one before's.
namespace planar {
template <class B> __device__ __forceinline__ u64 row_packets(const seismic_words_t &seismic_words_value) {
    return (B::k(seismic_words_value) + B::group() - 1) / B::group();
}
template <class B> __device__ __forceinline__ u64 codes_bytes(const seismic_words_t &seismic_words_value) {
    return B::n(seismic_words_value) * row_packets<B>(seismic_words_value) * B::code_bytes();
}
template <class B> __device__ __forceinline__ u64 grid_columns(const seismic_words_t &seismic_words_value) {
    return (B::k(seismic_words_value) + B::grid_tile_columns() - 1) / B::grid_tile_columns();
}
template <class B> __device__ __forceinline__ u64 grid_bytes(const seismic_words_t &seismic_words_value) {
    return (B::n(seismic_words_value) + B::grid_tile_rows() - 1) / B::grid_tile_rows() * grid_columns<B>(seismic_words_value) *
           B::scale_bytes();
}
// The byte of the scale of packet `packet` of row `n` in the grid.
template <class B> __device__ __forceinline__ u64 scale_at(const seismic_words_t &seismic_words_value, u64 n, u64 packet) {
    return (n / B::grid_tile_rows() * grid_columns<B>(seismic_words_value) + packet * B::group() / B::grid_tile_columns()) *
           B::scale_bytes();
}
} // namespace planar

// Addressing of the bound source tensor at `source`: packet `packet` of row
// `n` of matrix `b`, and the bytes between rows.
template <class B> struct Strided {
    const u8 *source;

    // The packet's codes, or its bytes whole of a source of one plane.
    __device__ __forceinline__ const u8 *packet(const seismic_words_t &seismic_words_value, u64 b, u64 n,
                                                u64 packet) const {
        if constexpr (B::planar()) {
            return source + b * B::source_matrix_bytes(seismic_words_value) +
                   (n * planar::row_packets<B>(seismic_words_value) + packet) * B::code_bytes();
        } else {
            return source + (b * B::source_stride_0(seismic_words_value) + n * B::source_stride_1(seismic_words_value) +
                             packet * B::source_stride_2(seismic_words_value)) *
                                B::source_packet_size();
        }
    }
    __device__ __forceinline__ u64 row_bytes(const seismic_words_t &seismic_words_value) const {
        if constexpr (B::planar())
            return planar::row_packets<B>(seismic_words_value) * B::code_bytes();
        else
            return B::source_stride_1(seismic_words_value) * B::source_packet_size();
    }
    // A planar source's scale of the tile holding packet `packet` of row
    // `n`, in its grid after the matrix's codes.
    __device__ __forceinline__ const u8 *scale(const seismic_words_t &seismic_words_value, u64 b, u64 n,
                                               u64 packet) const {
        return source + b * B::source_matrix_bytes(seismic_words_value) + planar::codes_bytes<B>(seismic_words_value) +
               planar::scale_at<B>(seismic_words_value, n, packet);
    }
    // The bytes between the scales of consecutive rows of a tile: its rows'
    // own where a tile is a row, none where it spans the tile's rows.
    __device__ __forceinline__ u64 scale_row_bytes(const seismic_words_t &seismic_words_value) const {
        return B::grid_tile_rows() == 1u ? planar::grid_columns<B>(seismic_words_value) * B::scale_bytes() : 0u;
    }
};

// Addressing of the bound destination tensor: the start of stored row `n`
// of matrix `b` (a packet layout's row is its packets).
template <class B> struct Rows {
    __device__ static __forceinline__ u8 *row(u8 *destination, const seismic_words_t &seismic_words_value, u64 b,
                                              u64 n) {
        return destination + (b * B::out_stride_0(seismic_words_value) + n * B::out_stride_1(seismic_words_value)) *
                                 B::out_stride_bytes(seismic_words_value);
    }
};

// The stage of a tile's rows' source bytes, by its LANES placing lanes
// (`mask`), which the conversion's code (SEISMIC_CONVERSION_<F>_<T>_UNIT,
// _FIXED) stages segment by segment of its tile plan into `rows`, then reads
// at `row()`: rows `real` hold source packets, row `r`'s at `source + r *
// row_bytes`.
template <class B, u32 LANES> struct Stage {
    typedef Shape<B> S;
    const u8 *source;
    u64 row_bytes;
    u32 real, lane, mask;
    u32 *rows;
    // A planar source's grid: the first row's scale of the tile, and the
    // bytes between its rows' (`Strided::scale`).
    const u8 *scale;
    u64 scale_row_bytes;

    // Whether staged unit `index` of a segment of `length` bytes, copied in
    // units of `unit`, lies past the tile's: the plans deal its units to a
    // lane a row (`Shape::LANES`), so more lanes are done with them before
    // the plans' last indices (every argument a constant: the test folds
    // away, and with it every unit past them).
    __device__ static __forceinline__ bool past(u32 length, u32 unit, u32 index) {
        return LANES * index >= S::TILE_ROWS * (length / unit);
    }

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
    // between its stores: measured against A7, iq3_s 1.09x, q3_k packet
    // 1.09x its time).
    __device__ __forceinline__ u32 load(u32 first, u32 length, u32 unit, int moved, u32 index) const {
        return read(source, row_bytes, first, length, unit, moved, index);
    }
    // A planar source's piece: its codes' bytes `first` on (a packet's, the
    // codes plane's), or its scale's.
    __device__ __forceinline__ u32 load_codes(u32 first, u32 length, u32 unit, int moved, u32 index) const {
        return read(source, row_bytes, first, length, unit, moved, index);
    }
    __device__ __forceinline__ u32 load_scale(u32 first, u32 length, u32 unit, int moved, u32 index) const {
        return read(scale, scale_row_bytes, first, length, unit, moved, index);
    }
    // The unit of `base`'s row `r` (rows `row_step` apart).
    __device__ __forceinline__ u32 read(const u8 *base, u64 row_step, u32 first, u32 length, u32 unit, int moved,
                                        u32 index) const {
        if (past(length, unit, index))
            return 0u;
        u32 r, k;
        locate(length, unit, index, r, k);
        const u8 *from = base + r * row_step + moved + first + k * unit;
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
        if (past(length, unit, index))
            return;
        u32 r, k;
        locate(length, unit, index, r, k);
        u8 *to = reinterpret_cast<u8 *>(rows) + r * 4 * S::STAGE_WORDS + position + k * unit;
        if (r < real) {
            if (unit == 4) *reinterpret_cast<u32 *>(to) = value;
            else if (unit == 2) *reinterpret_cast<u16 *>(to) = u16(value);
            else *to = u8(value);
        }
    }

    // Row `lane`'s staged bytes, once every lane has staged its share
    // (lanes past the tile's rows read a row's, and place nothing).
    __device__ __forceinline__ const u32 *row() const {
        __syncwarp(mask);
        return rows + lane % S::TILE_ROWS * S::STAGE_WORDS;
    }
};

// `bytes` bytes of `words` (little-endian) at `out`, in stores of `unit`
// bytes (a power of two, to 16, that `bytes` and `out` keep: the record's
// alignments, `SEISMIC_CONVERSION_<F>_<T>_*_ALIGNMENT`); `bytes` and `unit`
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

// A tile, placed by LANES lanes (`mask`): rows row0.. of matrix b at source
// packet `packet` (`occupied` when the tile holds source columns), unit
// `unit` of the group, read through `source`, staging in `stage`. Lane l
// computes row l's words.
template <class B, class A, u32 LANES> struct Tile {
    typedef Shape<B> S;
    A source;
    const seismic_words_t &seismic_words_value;
    u64 b, row0;
    u32 packet, unit, lane, mask;
    bool occupied;
    u32 *stage;

    // The rows holding source packets.
    __device__ __forceinline__ u32 real() const {
        const u64 n = B::n(seismic_words_value);
        return occupied ? u32(min(u64(S::TILE_ROWS), n - min(row0, n))) : 0u;
    }

    __device__ __forceinline__ Stage<B, LANES> staging() const {
        if constexpr (B::planar())
            return Stage<B, LANES>{source.packet(seismic_words_value, b, row0, packet),
                                   source.row_bytes(seismic_words_value),
                                   real(),
                                   lane,
                                   mask,
                                   stage,
                                   source.scale(seismic_words_value, b, row0, packet),
                                   source.scale_row_bytes(seismic_words_value)};
        else
            return Stage<B, LANES>{source.packet(seismic_words_value, b, row0, packet),
                                   source.row_bytes(seismic_words_value),
                                   real(),
                                   lane,
                                   mask,
                                   stage,
                                   nullptr,
                                   0u};
    }

    // Row `lane`'s record words (zero where its row holds no source
    // packet): the tile's unit's at words[0..], then, at a group's first
    // tile, the fixed words at their record words.
    __device__ __forceinline__ void unit_record(u32 *words) const {
        B::unit(staging(), unit, words);
        if (lane >= real())
            for (u32 word = 0; word < S::UNIT_WORDS; ++word) words[word] = 0;
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
        if constexpr (S::UNITS == 1) B::fixed_load(staging(), loaded);
    }

    // At a group's first tile, after its unit's (`loaded` as `fixed_load`
    // left it).
    __device__ __forceinline__ void fixed_record(u32 *loaded, u32 *words) const {
        // The stage is the tile's: every lane is done with what it held.
        __syncwarp(mask);
        if constexpr (S::UNITS != 1) B::fixed_load(staging(), loaded);
        B::fixed(staging(), loaded, words);
        if (lane >= real())
            for (u32 word = 0; word < S::WORDS; ++word) words[word] = 0;
    }
};

// ---- packet: the record is the packet form ---------------------------------

template <class B, class O>
__device__ __forceinline__ u8 *destination_packet(u8 *destination, const seismic_words_t &seismic_words_value, u64 b,
                                                  u64 n, u64 packet) {
    return O::row(destination, seismic_words_value, b, n) +
           packet * B::out_stride_2(seismic_words_value) * B::packet_size();
}

// The copy unit of record bytes of `alignment` (a record alignment,
// `SEISMIC_CONVERSION_<F>_<T>_*_ALIGNMENT`) from the staged record: to 4
// bytes, the widest a shared word gives.
__host__ __device__ constexpr u32 copy_unit(u32 alignment) { return alignment < 4u ? alignment : 4u; }

// Row `lane`'s words [first, first + count) and [second, second + more),
// staged one after the other from staged word 0.
template <class B, class A, u32 LANES>
__device__ __forceinline__ void stage_record(const Tile<B, A, LANES> &tile, const u32 *words, u32 first, u32 count,
                                             u32 second, u32 more) {
    typedef Shape<B> S;
    __syncwarp(tile.mask);
    if (tile.lane < S::TILE_ROWS) {
        for (u32 word = 0; word < count; ++word) tile.stage[tile.lane * S::RECORD_STRIDE + word] = words[first + word];
        for (u32 word = 0; word < more; ++word)
            tile.stage[tile.lane * S::RECORD_STRIDE + count + word] = words[second + word];
    }
    __syncwarp(tile.mask);
}

// Packet bytes [low, low + bytes), staged from record byte `slot` on, of
// every real row into its destination packet (rows out_stride_1 packets
// apart), coalesced, in copies of `unit` bytes (`copy_unit`); every argument
// but `low` a constant.
template <class B, class A, u32 LANES>
__device__ __forceinline__ void store_bytes(const Tile<B, A, LANES> &tile, u8 *out, u32 slot, u32 bytes, u32 unit,
                                            u32 low) {
    typedef Shape<B> S;
    const u32 units = bytes / unit, real = tile.real();
    const u64 row_bytes = B::out_stride_1(tile.seismic_words_value) * B::packet_size();
#pragma unroll 4
    for (u32 at = tile.lane; at < S::TILE_ROWS * units; at += LANES) {
        const u32 r = at / units, k = at % units;
        u8 *to = out + r * row_bytes + low + k * unit;
        const u8 *from = reinterpret_cast<const u8 *>(tile.stage + r * S::RECORD_STRIDE) + slot + k * unit;
        if (r < real) {
            if (unit == 4) *reinterpret_cast<u32 *>(to) = *reinterpret_cast<const u32 *>(from);
            else if (unit == 2) *reinterpret_cast<u16 *>(to) = *reinterpret_cast<const u16 *>(from);
            else *to = *from;
        }
    }
}

template <class B, class A, class O, u32 LANES>
__device__ __forceinline__ void packet_tile(const A &source, u8 *destination,
                                            const seismic_words_t &seismic_words_value, u64 b, u32 tile_x, u32 tile_y,
                                            u32 lane, u32 mask, u32 *shared) {
    typedef Shape<B> S;
    static_assert(S::CODES_END * 4 <= B::packet_size(), "the codes are whole words of a packet");
    // `shared`: the tile's staged source bytes, then its record words.
    const u64 row0 = u64(tile_y) * S::TILE_ROWS;
    const u32 column0 = tile_x * S::TILE_COLUMNS;
    const u64 packets = (B::k(seismic_words_value) + S::GROUP - 1) / S::GROUP;
    if (column0 >= packets * S::GROUP || row0 >= B::n(seismic_words_value)) return;
    const u32 packet = column0 / S::GROUP;
    const Tile<B, A, LANES> tile{source, seismic_words_value, b, row0, packet, column0 % S::GROUP / S::TILE_COLUMNS,
                                 lane, mask, true, shared};
    u8 *out = destination_packet<B, O>(destination, seismic_words_value, b, row0, packet);
    u32 words[S::WORDS], loaded[S::FIXED_LOADS];
    tile.fixed_load(loaded);
    tile.unit_record(words);
    stage_record(tile, words, 0, S::CODES_WORDS, 0, 0);
    store_bytes(tile, out, 0, 4 * S::CODES_WORDS, copy_unit(B::role_alignment(Role::codes)),
                4 * (S::CODES_WORD + tile.unit * S::CODES_WORDS));
    if (tile.unit != 0) return;
    // Every other record word: the other planes, and the gaps (zero).
    tile.fixed_record(loaded, words);
    constexpr u32 end = 4 * S::WORDS < B::packet_size() ? 4 * S::WORDS : B::packet_size();
    stage_record(tile, words, 0, S::CODES_WORD, S::CODES_END, S::WORDS - S::CODES_END);
    store_bytes(tile, out, 0, 4 * S::CODES_WORD, copy_unit(B::leading_alignment()), 0);
    store_bytes(tile, out, 4 * S::CODES_WORD, end > 4 * S::CODES_END ? end - 4 * S::CODES_END : 0,
                copy_unit(B::trailing_alignment()), 4 * S::CODES_END);
}

// ---- rows16 / mma16: the record is the row planes' shares -------------------

// `count` bytes at `out`, zero.
__device__ __forceinline__ void zero_bytes(u8 *out, u64 count) {
    for (u64 byte = 0; byte < count; ++byte) out[byte] = 0;
}

// Row `row`'s alignment tail of `role`'s plane: from its end to the next
// plane's start (or the row's end).
template <class B>
__device__ __forceinline__ void zero_tail(u8 *row, Role role, const seismic_words_t &seismic_words_value) {
    zero_bytes(row + B::row_plane(role, seismic_words_value) + B::row_plane_bytes(role, seismic_words_value),
               B::row_plane_tail(role, seismic_words_value));
}

// ---- mma16 records ----------------------------------------------------------
//
// An mma16 tile's 16 rows are one sequence (`PackedRowLayout::record_byte`):
// every stored group's entries, then every group's factors, then zero to the
// tile's end. Entries are 16 bits, in GEMV lane order (`B::record_entry`); a
// group's factors are in lane order too (`B::record_factor`). A unit's words
// are two entries each.

// Entry `e` of row `r` of a group (`value`) at the group's entries.
template <class B> __device__ __forceinline__ void place_entry(u8 *entries, u32 r, u32 e, u32 value) {
    static_assert(B::records_entry_bits() == 16, "a word is two u16 entries");
    constexpr u32 ENTRIES = B::records_kblock_entries();
    *reinterpret_cast<u16 *>(entries + B::record_entry(r, e / ENTRIES, e % ENTRIES)) = u16(value);
}

template <class B, class A, class O, u32 LANES>
__device__ __forceinline__ void records_tile(const A &source, u8 *destination,
                                             const seismic_words_t &seismic_words_value, u64 b, u32 tile_x, u32 tile_y,
                                             u32 lane, u32 mask, u32 *shared) {
    typedef Shape<B> S;
    constexpr u64 ENTRY_BYTES = B::records_entry_bytes();
    // The codes role is a group's entries (its units' words), then its factors.
    constexpr u32 ENTRIES_BYTES = S::GROUP * B::records_code_bits() / 8u;
    constexpr u32 FACTOR_BYTES = B::role_bytes(Role::codes) - ENTRIES_BYTES;
    constexpr u32 FACTORS_WORD = B::role_word(Role::codes) + ENTRIES_BYTES / 4u;
    static_assert(B::codes_first() + S::UNITS * S::CODES_WORDS == FACTORS_WORD,
                  "a records share is its units' entries, then its factors");
    static_assert(ENTRY_BYTES == S::MMA_ROWS * ENTRIES_BYTES, "a tile's group entries are its rows' entries");
    static_assert(S::HIGH_WORDS == 0, "a records unit is its entries");
    const u64 row0 = u64(tile_y) * S::TILE_ROWS;
    const u32 column0 = tile_x * S::TILE_COLUMNS;
    const u64 packets = (B::k(seismic_words_value) + S::GROUP - 1) / S::GROUP;
    const u64 row_groups = B::out_row_groups(seismic_words_value);
    const u64 stored_columns = row_groups * S::GROUP;
    // Stored rows: whole mma16 tiles, rows beyond N zero.
    const u64 stored_rows = (B::n(seismic_words_value) + S::MMA_ROWS - 1) / S::MMA_ROWS * S::MMA_ROWS;
    if (column0 >= stored_columns || row0 >= stored_rows) return;
    const u64 rows = min(u64(S::TILE_ROWS), stored_rows - row0);
    const Tile<B, A, LANES> tile{source, seismic_words_value, b, row0, column0 / S::GROUP,
                                 column0 % S::GROUP / S::TILE_COLUMNS, lane, mask, column0 < packets * S::GROUP, shared};
    // Row `lane`'s mma16 tile.
    u8 *mma = O::row(destination, seismic_words_value, b, row0 + lane - lane % S::MMA_ROWS);
    const u32 r = lane % S::MMA_ROWS;
    u32 words[S::WORDS], loaded[S::FIXED_LOADS];
    tile.fixed_load(loaded);
    tile.unit_record(words);
    if (lane < rows) {
        u8 *entries = mma + tile.packet * ENTRY_BYTES;
        for (u32 word = 0; word < S::UNIT_WORDS; ++word) {
            const u32 e = 2 * (tile.unit * S::UNIT_WORDS + word);
            place_entry<B>(entries, r, e, words[word]);
            place_entry<B>(entries, r, e + 1, words[word] >> B::records_entry_bits());
        }
    }
    if (tile.unit == 0) {
        tile.fixed_record(loaded, words);
        if (lane < rows)
            store(mma + row_groups * ENTRY_BYTES + tile.packet * (S::MMA_ROWS * FACTOR_BYTES) +
                      B::record_factor(r) * FACTOR_BYTES,
                  words + FACTORS_WORD, FACTOR_BYTES, B::codes_rest_alignment());
    }

    // Each mma16 tile's bytes past its last factors.
    if (column0 + S::TILE_COLUMNS == stored_columns) {
        for (u64 tile_row = 0; tile_row < rows; tile_row += S::MMA_ROWS) {
            u8 *tile_bytes = O::row(destination, seismic_words_value, b, row0 + tile_row);
            for (u64 byte = row_groups * S::MMA_ROWS * B::role_bytes(Role::codes) + lane;
                 byte < S::MMA_ROWS * B::out_row_stride_bytes(seismic_words_value); byte += LANES)
                tile_bytes[byte] = 0;
        }
    }
}

// ---- rows16 / mma16 planes --------------------------------------------------

// Destination of byte `position` of an mma16 tile's code-plane sequence V:
// the tile's rows' plane payloads (`row_bytes` each) concatenated in row
// order.
template <class O>
__device__ __forceinline__ u8 *tile_plane_byte(u8 *destination, const seismic_words_t &seismic_words_value, u64 b,
                                               u64 tile_row0, u64 position, u64 row_bytes, u64 offset) {
    return O::row(destination, seismic_words_value, b, tile_row0 + position / row_bytes) + offset +
           position % row_bytes;
}

// A code role, `bits` a column (a constant), whose words of the warp's rows
// are staged from word `at` of each row: of each of the warp's mma16 tiles,
// this lane's fragment slots of the tile's two k16 steps
// (`PackedRowLayout::code_bit`): slot `j` of a step holds the code of
// (`B::fragment_row`, `B::fragment_column`) at bit `bits` j, stored in units
// of `unit`.
template <class B, class A, class O, u32 LANES>
__device__ __forceinline__ void place_fragments(const Tile<B, A, LANES> &tile, u8 *destination, u64 rows, u32 column0,
                                                const u32 *staged, u32 bits, u32 at, u64 offset, u64 row_bytes,
                                                u32 unit) {
    typedef Shape<B> S;
    if (bits == 0) return;
    const u32 lane = tile.lane;
    const u32 block = column0 / B::kblock_columns(), step0 = (column0 % B::kblock_columns()) / B::step_columns();
    for (u32 tile_row = 0; tile_row < rows; tile_row += S::MMA_ROWS) {
        u32 value[4] = {};
#pragma unroll
        for (u32 half_step = 0; half_step < 2; ++half_step)
#pragma unroll
            for (u32 slot = 0; slot < B::step_slots(); ++slot) {
                const u32 row = tile_row + B::fragment_row(lane, slot);
                const u32 column = B::step_columns() * half_step + B::fragment_column(lane, slot);
                const u32 code = (staged[row * S::UNIT_WORDS + at + column * bits / 32] >> (column * bits % 32)) &
                                 ((1u << bits) - 1u);
                const u32 bit = (B::step_slots() * half_step + slot) * bits;
                value[bit / 32] |= code << (bit % 32);
            }
        store(tile_plane_byte<O>(destination, tile.seismic_words_value, tile.b, tile.row0 + tile_row,
                                 B::fragment_byte(block, lane, bits, step0), row_bytes, offset),
              value, 2 * bits, unit);
    }
}

template <class B, class A, class O, u32 LANES>
__device__ __forceinline__ void planes_tile(const A &source, u8 *destination,
                                            const seismic_words_t &seismic_words_value, u64 b, u32 tile_x, u32 tile_y,
                                            u32 lane, u32 mask, u32 *shared) {
    typedef Shape<B> S;
    static_assert(!S::FRAGMENTS || (32 % S::CODES_WORDS == 0 && (S::HIGH_WORDS == 0 || 32 % S::HIGH_WORDS == 0)),
                  "a code lies in one staged word");
    // The code roles' words of the warp's rows (mma16 fragments), after the
    // source stage.
    u32 *staged = shared + S::STAGE_SHARE;
    const u64 row0 = u64(tile_y) * S::TILE_ROWS;
    const u32 column0 = tile_x * S::TILE_COLUMNS;
    const u64 packets = (B::k(seismic_words_value) + S::GROUP - 1) / S::GROUP;
    const u64 stored_columns = B::out_row_groups(seismic_words_value) * S::GROUP;
    // mma16: stored rows are whole mma16 tiles, rows beyond N zero.
    const u64 stored_rows = B::mma16()
                                ? (B::n(seismic_words_value) + S::MMA_ROWS - 1) / S::MMA_ROWS * S::MMA_ROWS
                                : B::n(seismic_words_value);
    if (column0 >= stored_columns || row0 >= stored_rows) return;
    const u64 rows = min(u64(S::TILE_ROWS), stored_rows - row0);
    const Tile<B, A, LANES> tile{source, seismic_words_value, b, row0, column0 / S::GROUP,
                                 column0 % S::GROUP / S::TILE_COLUMNS, lane, mask, column0 < packets * S::GROUP, shared};
    u8 *row = O::row(destination, seismic_words_value, b, row0 + lane);
    u32 words[S::WORDS], loaded[S::FIXED_LOADS];
    tile.fixed_load(loaded);
    tile.unit_record(words);
    if constexpr (B::mma16() && B::grids() && S::TILE_COLUMNS == 32) {
        // Grid entries stay whole: an mma16 tile's sequence is
        // [k-block][row][k-block bytes]; row `r`'s 32 columns (steps `step0`,
        // `step0 + 1`) are half of its k-block bytes.
        if (lane < rows) {
            const u32 r = lane % S::MMA_ROWS;
            const u32 block = column0 / B::kblock_columns(), step0 = (column0 % B::kblock_columns()) / B::step_columns();
            store(tile_plane_byte<O>(destination, seismic_words_value, b, row0 + lane - r,
                                     (u64(block) * S::MMA_ROWS + r) * B::kblock_bytes() + step0 / 2 * 4 * S::CODES_WORDS,
                                     B::row_plane_bytes(Role::codes, seismic_words_value),
                                     B::row_plane(Role::codes, seismic_words_value)),
                  words, 4 * S::CODES_WORDS, B::role_alignment(Role::codes));
        }
    } else if constexpr (B::mma16() && B::grids()) {
        // A unit of whole k-blocks: per k-block the row's entry bytes, and its
        // entries' high bits ([k-block][row][the row's high bytes], placed as
        // the entries are) from the unit's high words, entry e at bit
        // ENTRY_BITS e. A row's k-block bytes are stored in the units they keep
        // (`B::kblock_alignment`).
        static_assert(S::TILE_COLUMNS % B::kblock_columns() == 0, "a unit of whole k-blocks");
        if (lane < rows) {
            constexpr u32 SPANS = S::TILE_COLUMNS / B::kblock_columns();
            constexpr u32 KBLOCK = B::kblock_bytes();
            const u32 r = lane % S::MMA_ROWS;
#pragma unroll
            for (u32 span = 0; span < SPANS; ++span) {
                const u64 block = column0 / B::kblock_columns() + span;
                store(tile_plane_byte<O>(destination, seismic_words_value, b, row0 + lane - r,
                                         (block * S::MMA_ROWS + r) * KBLOCK,
                                         B::row_plane_bytes(Role::codes, seismic_words_value),
                                         B::row_plane(Role::codes, seismic_words_value)),
                      words + span * KBLOCK / 4, KBLOCK, B::kblock_alignment());
                if constexpr (B::grid_high_bits() > 0) {
                    constexpr u32 HIGH = B::grid_high_bits(); // bytes a row's k-block
                    static_assert(4 * S::HIGH_WORDS == SPANS * HIGH && (HIGH == 2 || HIGH == 4),
                                  "a k-block's high bits");
                    const u32 high = words[S::CODES_WORDS + span * HIGH / 4] >> (8 * (span * HIGH % 4));
                    store(tile_plane_byte<O>(destination, seismic_words_value, b, row0 + lane - r,
                                             (block * S::MMA_ROWS + r) * HIGH,
                                             B::row_plane_bytes(Role::high, seismic_words_value),
                                             B::row_plane(Role::high, seismic_words_value)),
                          &high, HIGH, HIGH);
                }
            }
        }
    } else if constexpr (B::mma16()) {
        // The code roles' words of the warp's rows staged, then placed.
        if (lane < rows)
            for (u32 word = 0; word < S::UNIT_WORDS; ++word) staged[lane * S::UNIT_WORDS + word] = words[word];
        __syncwarp(mask);
        place_fragments<B, A, O, LANES>(tile, destination, rows, column0, staged, S::CODES_WORDS, 0,
                        B::row_plane(Role::codes, seismic_words_value),
                        B::row_plane_bytes(Role::codes, seismic_words_value), B::fragment_alignment(Role::codes));
        place_fragments<B, A, O, LANES>(tile, destination, rows, column0, staged, S::HIGH_WORDS, S::CODES_WORDS,
                        B::row_plane(Role::high, seismic_words_value),
                        B::row_plane_bytes(Role::high, seismic_words_value), B::fragment_alignment(Role::high));
    } else {
        // The tile's columns of a row are one unit of each code plane (rows16).
        if (lane < rows) {
            store(row + B::row_plane(Role::codes, seismic_words_value) + column0 / S::TILE_COLUMNS * 4 * S::CODES_WORDS,
                  words, 4 * S::CODES_WORDS, B::role_alignment(Role::codes));
            store(row + B::row_plane(Role::high, seismic_words_value) + column0 / S::TILE_COLUMNS * 4 * S::HIGH_WORDS,
                  words + S::CODES_WORDS, 4 * S::HIGH_WORDS, B::role_alignment(Role::high));
        }
    }

    // Group planes, at the group's first tile: a row's groups are
    // contiguous.
    if (tile.unit == 0) {
        tile.fixed_record(loaded, words);
        if (lane < rows) {
            store(row + B::row_plane(Role::scales, seismic_words_value) + tile.packet * B::role_bytes(Role::scales),
                  words + B::role_word(Role::scales), B::role_bytes(Role::scales), B::role_alignment(Role::scales));
            store(row + B::row_plane(Role::supers, seismic_words_value) + tile.packet * B::role_bytes(Role::supers),
                  words + B::role_word(Role::supers), B::role_bytes(Role::supers), B::role_alignment(Role::supers));
        }
    }

    // Alignment tails of every plane, from its end to the next plane's start
    // (or the row's end), at the row's last stored tile.
    if (column0 + S::TILE_COLUMNS == stored_columns && lane < rows) {
        zero_tail<B>(row, Role::codes, seismic_words_value);
        zero_tail<B>(row, Role::high, seismic_words_value);
        zero_tail<B>(row, Role::scales, seismic_words_value);
        zero_tail<B>(row, Role::supers, seismic_words_value);
    }
}

// Place row tile (`tile_x` columns, `tile_y` rows) of matrix `b`, reading
// the source through `source` (an `A`) and writing through `O`, by LANES
// lanes of a warp (`lanes_mask`; `lane` the thread's among them: a whole
// warp, or a share of it that places a tile of its own, `Shape::WARP_TILES`),
// staging in their `shared` (`Shape::SHARED_WORDS` words). A tile outside the
// matrix's stored extent places nothing.
template <class B, class A = Strided<B>, class O = Rows<B>, u32 LANES = 32>
__device__ __forceinline__ void convert_tile(const A &source, u8 *destination,
                                             const seismic_words_t &seismic_words_value, u64 b, u32 tile_x, u32 tile_y,
                                             u32 lane, u32 *shared) {
    typedef Shape<B> S;
    static_assert(S::TILE_ROWS % (B::packet() ? 1u : S::MMA_ROWS) == 0, "a row tile holds whole mma16 tiles");
    // A lane a row computes its words, and the stage's lanes take every unit
    // the plans deal.
    static_assert(LANES >= S::TILE_ROWS && LANES >= S::LANES, "a tile's lanes hold its rows");
    static_assert(32 / LANES <= S::WARP_TILES, "a warp places at most its WARP_TILES tiles at once");
    const u32 mask = lanes_mask<LANES>();
    if constexpr (B::packet())
        packet_tile<B, A, O, LANES>(source, destination, seismic_words_value, b, tile_x, tile_y, lane, mask, shared);
    else if constexpr (B::records())
        records_tile<B, A, O, LANES>(source, destination, seismic_words_value, b, tile_x, tile_y, lane, mask, shared);
    else
        planes_tile<B, A, O, LANES>(source, destination, seismic_words_value, b, tile_x, tile_y, lane, mask, shared);
}

} // namespace repack

// ---- bindings ----------------------------------------------------------------
//
// `REPACK_BINDING(Name, C, S, D, DIM_N, DIM_K)` defines binding `Name` of
// the conversion of tensor `S` (ABI prefix such as `SEISMIC_SOURCE`) into
// tensor `D` (such as `SEISMIC_RESULT_0`), both [*, N, K] with N = DIM_N
// and K = DIM_K, by `C`, the prefix of the conversion between their element
// parameters (such as `SEISMIC_CONVERSION_E_U`): its constants and code are
// `C`'s macros, the elements' facts and the geometry the tensors'. A binding
// of two dense elements (`dense()`) converts by elements
// (`streaming::convert_item`), and has no registered conversion. `C` is
// defined as 1 where the conversion is registered, so the binding probes it
// unpasted and names its macros only by pasting it whole (`C##_WORDS`: an
// operand of `##` is not expanded). A macro an element or conversion does not
// define is never named (`REPACK_WHEN`, `REPACK_PICK` keep only the branch
// their probe selects): every member is 0 where its element or conversion
// has none.

#define REPACK_PICK_1(yes, no) yes
#define REPACK_PICK_0(yes, no) no
#define REPACK_PICK_(flag) REPACK_PICK_##flag
#define REPACK_PICK(flag) REPACK_PICK_(flag)
// `yes` when `<P><suffix>` is defined as 1, else `no`.
#define REPACK_WHEN(P, suffix, yes, no) REPACK_PICK(ELEMENT_HAS(P, suffix))(yes, no)
// Conversion constant `word` where `registered` (`ELEMENT_IS_ONE(C)`), else 0.
#define REPACK_CONVERSION_WORD(registered, word) REPACK_PICK(registered)((repack::u32)word, 0u)
// A constant only a packet (row) layout's conversion has, or 0.
#define REPACK_PACKET_CONVERSION_WORD(D, registered, word)                                                  \
    REPACK_WHEN(D, _LAYOUT_PACKET, REPACK_CONVERSION_WORD(registered, word), 0u)
#define REPACK_ROWS_CONVERSION_WORD(D, registered, word)                                                    \
    REPACK_WHEN(D, _LAYOUT_PACKET, 0u, REPACK_CONVERSION_WORD(registered, word))
// A fact only a row layout's element has, or 0.
#define REPACK_ROWS_ELEMENT_WORD(D, suffix)                                                                 \
    REPACK_WHEN(D, _KIND_PACKED, REPACK_WHEN(D, _LAYOUT_PACKET, 0u, (repack::u32)ELEMENT_CAT(D, suffix)), 0u)
// A row-layout geometry word of tensor D, or 0 for a packet layout or a
// dense representation (which have no row words).
#define REPACK_ROW_WORD(D, WORD)                                                                            \
    REPACK_WHEN(D, _KIND_PACKED, REPACK_WHEN(D, _LAYOUT_PACKET, 0ull, (repack::u64)ELEMENT_CAT(D, WORD)), 0ull)
// The bytes of one unit of tensor P's strides: a row layout's stored row,
// any other's packet (a dense element's packet is one element).
#define REPACK_STRIDE_BYTES(P)                                                                              \
    REPACK_WHEN(P, _KIND_PACKED,                                                                            \
                REPACK_WHEN(P, _LAYOUT_PACKET, (repack::u64)ELEMENT_CAT(P, _PACKET_SIZE),                   \
                            (repack::u64)ELEMENT_CAT(P, _ROW_STRIDE_BYTES)),                                \
                (repack::u64)ELEMENT_CAT(P, _PACKET_SIZE))
// The bytes between the matrices of a stacked [*, N, K] tensor P (a record).
#define REPACK_MATRIX_BYTES(P) ((repack::u64)ELEMENT_CAT(P, _STRIDE_0) * REPACK_STRIDE_BYTES(P))

#define REPACK_BINDING(Name, C, S, D, DIM_N, DIM_K)                                                         \
    struct Name {                                                                                           \
        /* The elements. */                                                                                 \
        __host__ __device__ static constexpr bool converts() {                                              \
            return REPACK_PICK(ELEMENT_IS_ONE(C))(true, false);                                             \
        }                                                                                                   \
        __host__ __device__ static constexpr bool dense() {                                                 \
            return ELEMENT_HAS(S, _KIND_DENSE) && ELEMENT_HAS(D, _KIND_DENSE); \
        }                                                                                                   \
        __host__ __device__ static constexpr int source_kind() { return ELEMENT_KIND(S); }  \
        __host__ __device__ static constexpr int destination_kind() { return ELEMENT_KIND(D); } \
        __host__ __device__ static constexpr bool external() { return ELEMENT_HAS(S, _KIND_EXTERNAL); } \
        __host__ __device__ static constexpr bool packed() { return ELEMENT_HAS(D, _KIND_PACKED); } \
        __host__ __device__ static constexpr bool packet() { return ELEMENT_HAS(D, _LAYOUT_PACKET); } \
        __host__ __device__ static constexpr bool rows16() { return ELEMENT_HAS(D, _LAYOUT_ROWS16); } \
        __host__ __device__ static constexpr bool mma16() { return ELEMENT_HAS(D, _LAYOUT_MMA16); } \
        __host__ __device__ static constexpr bool grids() { return ELEMENT_HAS(D, _PLANE_GRIDS); } \
        __host__ __device__ static constexpr bool records() { return ELEMENT_HAS(D, _PLANE_RECORDS); } \
        __host__ __device__ static constexpr repack::u32 group() {                                          \
            return (repack::u32)ELEMENT_CAT(S, _LOGICAL_GROUP);                             \
        }                                                                                                   \
        __host__ __device__ static constexpr repack::u32 source_packet_size() {                             \
            return (repack::u32)ELEMENT_CAT(S, _PACKET_SIZE);                               \
        }                                                                                                   \
        /* A source under a companion grid (`ExternalPacketLayout::planes`): its scales' bytes and the */ \
        /* tile of values each scales, 0 without; its packets' codes are the rest. */                      \
        __host__ __device__ static constexpr repack::u32 scale_bytes() {                                    \
            return REPACK_WHEN(S, _KIND_EXTERNAL, (repack::u32)ELEMENT_CAT(S, _GRID_SCALE_BYTES), 0u);     \
        }                                                                                                   \
        __host__ __device__ static constexpr repack::u32 grid_tile_rows() {                                      \
            return REPACK_WHEN(S, _KIND_EXTERNAL, (repack::u32)ELEMENT_CAT(S, _GRID_TILE_ROWS), 0u);       \
        }                                                                                                   \
        __host__ __device__ static constexpr repack::u32 grid_tile_columns() {                                 \
            return REPACK_WHEN(S, _KIND_EXTERNAL, (repack::u32)ELEMENT_CAT(S, _GRID_TILE_COLUMNS), 0u);    \
        }                                                                                                   \
        __host__ __device__ static constexpr bool planar() { return scale_bytes() != 0u; }                  \
        __host__ __device__ static constexpr repack::u32 code_bytes() {                                     \
            return source_packet_size() - scale_bytes();                                                    \
        }                                                                                                   \
        __host__ __device__ static constexpr repack::u32 packet_size() {                                    \
            return REPACK_WHEN(D, _LAYOUT_PACKET,                                           \
                               (repack::u32)ELEMENT_CAT(D, _PACKET_SIZE), 0u);              \
        }                                                                                                   \
        __host__ __device__ static constexpr repack::u32 mma_rows() {                                       \
            return REPACK_ROWS_ELEMENT_WORD(D, _TILE_ROWS);                                                 \
        }                                                                                                   \
        __host__ __device__ static constexpr repack::u64 records_entry_bytes() {                            \
            return REPACK_WHEN(D, _PLANE_RECORDS,                                           \
                               (repack::u64)ELEMENT_CAT(D, _PLANE_RECORDS_ENTRY_BYTES), 0ull); \
        }                                                                                                   \
        __host__ __device__ static constexpr repack::u32 records_code_bits() {                              \
            return REPACK_WHEN(D, _PLANE_RECORDS,                                           \
                               (repack::u32)ELEMENT_CAT(D, _PLANE_RECORDS_CODE_BITS), 0u);  \
        }                                                                                                   \
        __host__ __device__ static constexpr repack::u32 records_entry_bits() {                             \
            return REPACK_WHEN(D, _PLANE_RECORDS,                                           \
                               (repack::u32)ELEMENT_CAT(D, _PLANE_RECORDS_ENTRY_BITS), 0u); \
        }                                                                                                   \
        __host__ __device__ static constexpr repack::u32 records_kblock_entries() {                         \
            return REPACK_WHEN(D, _PLANE_RECORDS,                                           \
                               (repack::u32)ELEMENT_CAT(D, _PLANE_RECORDS_KBLOCK_ENTRIES), 1u); \
        }                                                                                                   \
        /* mma16's facts and placement rules (`PackedRowLayout`); 1 or 0 in another layout, */              \
        /* whose branches never reach them. */                                                              \
        __host__ __device__ static constexpr repack::u32 kblock_columns() {                                 \
            return REPACK_WHEN(D, _LAYOUT_MMA16, (repack::u32)ELEMENT_CAT(D, _MMA_KBLOCK), 1u);             \
        }                                                                                                   \
        __host__ __device__ static constexpr repack::u32 step_columns() {                                   \
            return REPACK_WHEN(D, _LAYOUT_MMA16, (repack::u32)ELEMENT_CAT(D, _MMA_STEP), 1u);               \
        }                                                                                                   \
        __host__ __device__ static constexpr repack::u32 step_slots() {                                     \
            return REPACK_WHEN(D, _LAYOUT_MMA16, (repack::u32)ELEMENT_CAT(D, _MMA_STEP_SLOTS), 0u);         \
        }                                                                                                   \
        __device__ static __forceinline__ repack::u32 fragment_row(repack::u32 lane, repack::u32 slot) {    \
            return REPACK_WHEN(D, _LAYOUT_MMA16, ELEMENT_CAT(D, _MMA_FRAGMENT_ROW)(lane, slot), 0u);        \
        }                                                                                                   \
        __device__ static __forceinline__ repack::u32 fragment_column(repack::u32 lane, repack::u32 slot) { \
            return REPACK_WHEN(D, _LAYOUT_MMA16, ELEMENT_CAT(D, _MMA_FRAGMENT_COLUMN)(lane, slot), 0u);     \
        }                                                                                                   \
        __device__ static __forceinline__ repack::u64 fragment_byte(repack::u64 block, repack::u64 lane,    \
                                                                    repack::u64 bits, repack::u64 step) {   \
            return REPACK_WHEN(D, _LAYOUT_MMA16, ELEMENT_CAT(D, _MMA_FRAGMENT_BYTE)(block, lane, bits, step), 0ull); \
        }                                                                                                   \
        __device__ static __forceinline__ repack::u32 record_entry(repack::u32 r, repack::u32 block,        \
                                                                   repack::u32 entry) {                     \
            return REPACK_WHEN(D, _LAYOUT_MMA16, ELEMENT_CAT(D, _RECORD_ENTRY)(r, block, entry), 0u);       \
        }                                                                                                   \
        __device__ static __forceinline__ repack::u32 record_factor(repack::u32 r) {                        \
            return REPACK_WHEN(D, _LAYOUT_MMA16, ELEMENT_CAT(D, _RECORD_FACTOR)(r), 0u);                    \
        }                                                                                                   \
        /* Each grid entry's bits past its whole bytes (`GRID_HIGH_ENTRY_BITS`), or 0. */                   \
        __host__ __device__ static constexpr repack::u32 grid_high_bits() {                                 \
            return REPACK_WHEN(D, _PLANE_GRID_HIGH,                                         \
                               (repack::u32)ELEMENT_CAT(D, _PLANE_GRID_HIGH_ENTRY_BITS), 0u); \
        }                                                                                                   \
        /* The conversion's constants. */                                                                   \
        __host__ __device__ static constexpr repack::u32 words() { return REPACK_CONVERSION_WORD(ELEMENT_IS_ONE(C), C##_WORDS); } \
        __host__ __device__ static constexpr repack::u32 tile_rows() {                                      \
            return REPACK_CONVERSION_WORD(ELEMENT_IS_ONE(C), C##_TILE_ROWS);                                                \
        }                                                                                                   \
        __host__ __device__ static constexpr repack::u32 lanes() { return REPACK_CONVERSION_WORD(ELEMENT_IS_ONE(C), C##_LANES); } \
        __host__ __device__ static constexpr repack::u32 unit_columns() {                                   \
            return REPACK_CONVERSION_WORD(ELEMENT_IS_ONE(C), C##_UNIT_COLUMNS);                             \
        }                                                                                                   \
        __host__ __device__ static constexpr repack::u32 stage_words() {                                    \
            return REPACK_CONVERSION_WORD(ELEMENT_IS_ONE(C), C##_STAGE_WORDS);                                              \
        }                                                                                                   \
        __host__ __device__ static constexpr repack::u32 fixed_loads() {                                    \
            return REPACK_CONVERSION_WORD(ELEMENT_IS_ONE(C), C##_FIXED_LOADS);                                              \
        }                                                                                                   \
        __host__ __device__ static constexpr repack::u32 codes_first() {                                    \
            return REPACK_CONVERSION_WORD(ELEMENT_IS_ONE(C), C##_CODES_FIRST);                                              \
        }                                                                                                   \
        __host__ __device__ static constexpr repack::u32 codes_unit_words() {                               \
            return REPACK_CONVERSION_WORD(ELEMENT_IS_ONE(C), C##_CODES_UNIT_WORDS);                                         \
        }                                                                                                   \
        __host__ __device__ static constexpr repack::u32 high_unit_words() {                                \
            return REPACK_CONVERSION_WORD(ELEMENT_IS_ONE(C), C##_HIGH_UNIT_WORDS);                                          \
        }                                                                                                   \
        __host__ __device__ static constexpr repack::u32 leading_alignment() {                              \
            return REPACK_PACKET_CONVERSION_WORD(D, ELEMENT_IS_ONE(C), C##_LEADING_ALIGNMENT);                                 \
        }                                                                                                   \
        __host__ __device__ static constexpr repack::u32 trailing_alignment() {                             \
            return REPACK_PACKET_CONVERSION_WORD(D, ELEMENT_IS_ONE(C), C##_TRAILING_ALIGNMENT);                                \
        }                                                                                                   \
        __host__ __device__ static constexpr repack::u32 codes_rest_alignment() {                           \
            return REPACK_ROWS_CONVERSION_WORD(D, ELEMENT_IS_ONE(C), C##_CODES_REST_ALIGNMENT);                                \
        }                                                                                                   \
        /* The units mma16's code bytes are stored in. */                                                   \
        __host__ __device__ static constexpr repack::u32 kblock_bytes() {                                   \
            return REPACK_WHEN(D, _LAYOUT_MMA16, REPACK_CONVERSION_WORD(ELEMENT_IS_ONE(C), C##_KBLOCK_BYTES), 0u); \
        }                                                                                                   \
        __host__ __device__ static constexpr repack::u32 kblock_alignment() {                               \
            return REPACK_WHEN(D, _LAYOUT_MMA16, REPACK_CONVERSION_WORD(ELEMENT_IS_ONE(C), C##_KBLOCK_ALIGNMENT), 0u); \
        }                                                                                                   \
        __host__ __device__ static constexpr repack::u32 fragment_alignment(repack::Role role) {            \
            return role == repack::Role::codes                                                              \
                       ? REPACK_WHEN(D, _LAYOUT_MMA16, REPACK_CONVERSION_WORD(ELEMENT_IS_ONE(C), C##_CODES_FRAGMENT_ALIGNMENT), 0u) \
                       : REPACK_WHEN(D, _LAYOUT_MMA16, REPACK_CONVERSION_WORD(ELEMENT_IS_ONE(C), C##_HIGH_FRAGMENT_ALIGNMENT), 0u); \
        }                                                                                                   \
        /* A role's share of the record (row layouts): first word, bytes, store alignment (codes: */        \
        /* every layout's). */                                                                              \
        __host__ __device__ static constexpr repack::u32 role_word(repack::Role role) {                     \
            return role == repack::Role::codes    ? REPACK_ROWS_CONVERSION_WORD(D, ELEMENT_IS_ONE(C), C##_CODES_WORD)          \
                 : role == repack::Role::high     ? REPACK_ROWS_CONVERSION_WORD(D, ELEMENT_IS_ONE(C), C##_HIGH_WORD)           \
                 : role == repack::Role::scales   ? REPACK_ROWS_CONVERSION_WORD(D, ELEMENT_IS_ONE(C), C##_SCALES_WORD)         \
                                                  : REPACK_ROWS_CONVERSION_WORD(D, ELEMENT_IS_ONE(C), C##_SUPERS_WORD);        \
        }                                                                                                   \
        __host__ __device__ static constexpr repack::u32 role_bytes(repack::Role role) {                    \
            return role == repack::Role::codes    ? REPACK_ROWS_CONVERSION_WORD(D, ELEMENT_IS_ONE(C), C##_CODES_BYTES)         \
                 : role == repack::Role::high     ? REPACK_ROWS_CONVERSION_WORD(D, ELEMENT_IS_ONE(C), C##_HIGH_BYTES)          \
                 : role == repack::Role::scales   ? REPACK_ROWS_CONVERSION_WORD(D, ELEMENT_IS_ONE(C), C##_SCALES_BYTES)        \
                                                  : REPACK_ROWS_CONVERSION_WORD(D, ELEMENT_IS_ONE(C), C##_SUPERS_BYTES);       \
        }                                                                                                   \
        __host__ __device__ static constexpr repack::u32 role_alignment(repack::Role role) {                \
            return role == repack::Role::codes    ? REPACK_CONVERSION_WORD(ELEMENT_IS_ONE(C), C##_CODES_ALIGNMENT)          \
                 : role == repack::Role::high     ? REPACK_ROWS_CONVERSION_WORD(D, ELEMENT_IS_ONE(C), C##_HIGH_ALIGNMENT)      \
                 : role == repack::Role::scales   ? REPACK_ROWS_CONVERSION_WORD(D, ELEMENT_IS_ONE(C), C##_SCALES_ALIGNMENT)    \
                                                  : REPACK_ROWS_CONVERSION_WORD(D, ELEMENT_IS_ONE(C), C##_SUPERS_ALIGNMENT);   \
        }                                                                                                   \
        /* The conversion's code, over a stage `stage` (`repack::Stage`). */                                \
        template <class St>                                                                                 \
        __device__ static __forceinline__ void unit(const St &stage, repack::u32 unit, repack::u32 *words) { \
            REPACK_PICK(ELEMENT_IS_ONE(C))(C##_UNIT(stage, unit, words), (void)0);                          \
        }                                                                                                   \
        template <class St>                                                                                 \
        __device__ static __forceinline__ void fixed_load(const St &stage, repack::u32 *loaded) {           \
            REPACK_PICK(ELEMENT_IS_ONE(C))(C##_FIXED_LOAD(stage, loaded), (void)0);                         \
        }                                                                                                   \
        template <class St>                                                                                 \
        __device__ static __forceinline__ void fixed(const St &stage, const repack::u32 *loaded,            \
                                                     repack::u32 *words) {                                  \
            REPACK_PICK(ELEMENT_IS_ONE(C))(C##_FIXED(stage, loaded, words), (void)0);                       \
        }                                                                                                   \
        /* The tensors' geometry. */                                                                        \
        __device__ static __forceinline__ repack::u64 n(const seismic_words_t &seismic_words_value) {       \
            return (repack::u64)(DIM_N);                                                                    \
        }                                                                                                   \
        __device__ static __forceinline__ repack::u64 k(const seismic_words_t &seismic_words_value) {       \
            return (repack::u64)(DIM_K);                                                                    \
        }                                                                                                   \
        __device__ static __forceinline__ repack::u64 source_stride_0(const seismic_words_t &seismic_words_value) { \
            return (repack::u64)ELEMENT_CAT(S, _STRIDE_0);                                                  \
        }                                                                                                   \
        /* A planar source strides by matrices (its tensor's last two axes are one unit): its rows and */   \
        /* packets are those of its codes plane, a row's `row_packets` packets of `code_bytes`. */           \
        __device__ static __forceinline__ repack::u64 source_stride_1(const seismic_words_t &seismic_words_value) { \
            if (planar()) return (k(seismic_words_value) + group() - 1) / group();                          \
            return (repack::u64)ELEMENT_CAT(S, _STRIDE_1);                                                  \
        }                                                                                                   \
        __device__ static __forceinline__ repack::u64 source_stride_2(const seismic_words_t &seismic_words_value) { \
            if (planar()) return 1u;                                                                        \
            return (repack::u64)ELEMENT_CAT(S, _STRIDE_2);                                                  \
        }                                                                                                   \
        __device__ static __forceinline__ repack::u64 out_stride_0(const seismic_words_t &seismic_words_value) { \
            return (repack::u64)ELEMENT_CAT(D, _STRIDE_0);                                                  \
        }                                                                                                   \
        __device__ static __forceinline__ repack::u64 out_stride_1(const seismic_words_t &seismic_words_value) { \
            return (repack::u64)ELEMENT_CAT(D, _STRIDE_1);                                                  \
        }                                                                                                   \
        __device__ static __forceinline__ repack::u64 out_stride_2(const seismic_words_t &seismic_words_value) { \
            return (repack::u64)ELEMENT_CAT(D, _STRIDE_2);                                                  \
        }                                                                                                   \
        __device__ static __forceinline__ repack::u64 out_row_stride_bytes(                                 \
            const seismic_words_t &seismic_words_value) {                                                   \
            return REPACK_ROW_WORD(D, _ROW_STRIDE_BYTES);                                                   \
        }                                                                                                   \
        /* Each tensor's bytes a stride unit and a record (`REPACK_STRIDE_BYTES`, `_MATRIX_BYTES`). */    \
        __device__ static __forceinline__ repack::u64 out_stride_bytes(const seismic_words_t &seismic_words_value) { \
            return REPACK_STRIDE_BYTES(D);                                                                  \
        }                                                                                                   \
        __device__ static __forceinline__ repack::u64 out_matrix_bytes(const seismic_words_t &seismic_words_value) { \
            return REPACK_MATRIX_BYTES(D);                                                                  \
        }                                                                                                   \
        __device__ static __forceinline__ repack::u64 source_matrix_bytes(                                  \
            const seismic_words_t &seismic_words_value) {                                                   \
            if constexpr (planar())                                                                         \
                return repack::planar::codes_bytes<Name>(seismic_words_value) +                          \
                       repack::planar::grid_bytes<Name>(seismic_words_value);                           \
            else                                                                                            \
                return REPACK_MATRIX_BYTES(S);                                                              \
        }                                                                                                   \
        __device__ static __forceinline__ repack::u64 out_row_groups(const seismic_words_t &seismic_words_value) { \
            return REPACK_ROW_WORD(D, _ROW_GROUPS);                                                         \
        }                                                                                                   \
        /* A role's row plane: its row offset, bytes per row, and tail to the next plane (or the */         \
        /* row's end); 0 without the role. */                                                               \
        __device__ static __forceinline__ repack::u64 row_plane(repack::Role role,                          \
                                                                const seismic_words_t &seismic_words_value) { \
            return role == repack::Role::codes    ? REPACK_ROW_WORD(D, _ROW_CODES)                          \
                 : role == repack::Role::high     ? REPACK_ROW_WORD(D, _ROW_HIGH)                           \
                 : role == repack::Role::scales   ? REPACK_ROW_WORD(D, _ROW_SCALES)                         \
                                                  : REPACK_ROW_WORD(D, _ROW_SUPERS);                        \
        }                                                                                                   \
        __device__ static __forceinline__ repack::u64 row_plane_bytes(                                      \
            repack::Role role, const seismic_words_t &seismic_words_value) {                                \
            return role == repack::Role::codes    ? REPACK_ROW_WORD(D, _ROW_CODES_BYTES)                    \
                 : role == repack::Role::high     ? REPACK_ROW_WORD(D, _ROW_HIGH_BYTES)                     \
                 : role == repack::Role::scales   ? REPACK_ROW_WORD(D, _ROW_SCALES_BYTES)                   \
                                                  : REPACK_ROW_WORD(D, _ROW_SUPERS_BYTES);                  \
        }                                                                                                   \
        __device__ static __forceinline__ repack::u64 row_plane_tail(                                       \
            repack::Role role, const seismic_words_t &seismic_words_value) {                                \
            return role == repack::Role::codes    ? REPACK_ROW_WORD(D, _ROW_CODES_TAIL)                     \
                 : role == repack::Role::high     ? REPACK_ROW_WORD(D, _ROW_HIGH_TAIL)                      \
                 : role == repack::Role::scales   ? REPACK_ROW_WORD(D, _ROW_SCALES_TAIL)                    \
                                                  : REPACK_ROW_WORD(D, _ROW_SUPERS_TAIL);                   \
        }                                                                                                   \
    };
