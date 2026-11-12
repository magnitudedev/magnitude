// Expert streaming's conversion of staged records, by windows; the CUDA form
// of `metal/lib/streaming/window.h`. A block converts whole row tiles of a
// record (its conversion's tile rows, `record_items`),
// one window of packets along them at a time (`Window`): each thread loads
// its share of the window's stored bytes from staging into registers, 16
// bytes a load wherever the records keep whole 16-byte units (`Held`),
// while the block converts the window before it (staging is mapped host
// memory, each load a PCIe round trip on a discrete device, so every row's
// loads are in flight together), then stores them into shared memory, where its warps place the window's column tiles
// by `repack::convert_tile` (lib/repack/repack.cuh), its source the copy,
// two tiles at once a warp, a half warp each, wherever the layout lets half
// a warp place a tile (`repack::Shape::WARP_TILES`). The tiles are placed in
// the block's mirror of the row tile (`Mirror`, the launch's dynamic shared
// memory, MIRROR_BYTES), which the block writes to the record whole once its
// last window is placed (`write_tile`): a tile conversion's own stores are a
// lane per row, narrow and row-strided, which GB10's host-NUMA memory takes
// at a quarter of device memory's rate (148-158 against 35-41 us a 35B
// record), while whole lines take both at device speed (30-36 and 29-35 us,
// 2026-11-03).
// A row tile larger than the mirror is placed in the record directly. The tile
// conversion is `repack_weight`'s, so a slot holds exactly the bytes a
// resident import places. A dense float record (`B::dense()`: `repack`
// between dense representations) converts row by row, each element as
// `import_dense` converts it (decoded to F32, rounded into the destination).
//
// Templates over a repack binding `B` (`REPACK_BINDING`), so a kernel
// converts through as many bindings as it holds, in one `Shared` of them;
// the held window is the bindings' alike, so a block loads one binding's
// window while it converts another's. The includer includes
// lib/streaming/control.cuh first.

#include "../repack/repack.cuh"
#include "../../../streaming/window.hpp"

namespace streaming {

using element::u32;
using element::u64;
using element::u8;

// The unit a window's stored bytes load in where its records keep no whole
// 16-byte units: the widest, to 16 bytes, its packets keep (a conveyor
// record's rows, windows and records are whole packets, from a 16-byte
// aligned conveyor).
template <u32 BYTES> struct LoadUnit;
template <> struct LoadUnit<16> { typedef uint4 type; };
template <> struct LoadUnit<8> { typedef uint2 type; };
template <> struct LoadUnit<4> { typedef u32 type; };
template <> struct LoadUnit<2> { typedef element::u16 type; };
template <> struct LoadUnit<1> { typedef u8 type; };

__host__ __device__ constexpr u32 load_unit(u32 packet) {
    return packet % 16 == 0 ? 16u : packet % 8 == 0 ? 8u : packet % 4 == 0 ? 4u : packet % 2 == 0 ? 2u : 1u;
}

// How a launch windows its records, the same for every item: an item of a
// record is a window of one of its row tiles (`record_items`), the tile's
// rows' stored bytes of `packets` packets each from the window's first,
// which a block copies into its shared memory (the module's WINDOW_BYTES,
// `SEISMIC_CONST_WINDOW_BYTES`), rows `pitch` bytes apart, and converts. The
// windows are every backend's (`streaming/window.hpp`: whole, aligned or
// padded). Its loads are 16 bytes wide (`unit`) wherever every window's
// stored bytes are whole 16-byte units of every record (records, and the
// scratch slots that may hold them, a multiple of 16 bytes apart in 16-byte
// aligned tensors), else the widest the packets keep (`load_unit`):
// - a whole window's one span runs from a row tile's first row to its last
//   row's last packet, 16 rows from a tile's aligned start: whole units of a
//   full tile where rows are 16 bytes apart (stride and row bytes a multiple
//   of 16), or stored back to back (stride the row's bytes), and of a last
//   tile where the record's rows are (N rows' bytes whole units);
// - an aligned window's rows are 16 bytes apart from 16-byte units, whole
//   units where a row's bytes are.
// A window's column tiles are its packets'; a row tile's last window also
// holds the tiles of its padding columns (to `repack::tiles_x`).
template <class B> struct Window {
    typedef repack::Shape<B> S;
    static constexpr u32 ROWS = S::TILE_ROWS;
    static constexpr u32 PACKET = B::source_packet_size();
    static_assert(STREAMING_WINDOW_FITS(ROWS, PACKET, SEISMIC_CONST_WINDOW_BYTES), "a window holds a packet of each row");
    static constexpr u32 NARROW = load_unit(PACKET);

    u32 row_packets, source_packets, tiles_x, packets, windows, pitch, unit;
    bool whole;

    __device__ __forceinline__ explicit Window(const seismic_words_t &seismic_words_value) {
        const u64 n = B::n(seismic_words_value);
        tiles_x = (u32)repack::tiles_x<B>(seismic_words_value);
        row_packets = streaming_window_row_packets((u64)tiles_x * S::TILE_COLUMNS, B::group());
        source_packets = (u32)((B::k(seismic_words_value) + B::group() - 1) / B::group());
        const u64 row = (u64)source_packets * PACKET, stride = B::source_stride_1(seismic_words_value) * PACKET;
        const u32 window = SEISMIC_CONST_WINDOW_BYTES;
        whole = streaming_window_whole(ROWS, (u32)stride, window);
        packets = streaming_window_packets(ROWS, row_packets, PACKET, (u32)stride, window);
        pitch = streaming_window_pitch(ROWS, PACKET, packets, (u32)stride, window);
        windows = streaming_window_count(row_packets, packets);
        const bool aligned = B::source_matrix_bytes(seismic_words_value) % 16 == 0 &&
                             B::out_matrix_bytes(seismic_words_value) % 16 == 0;
        const bool wide = whole ? (stride == row ? n * row % 16 == 0 : stride % 16 == 0 && row % 16 == 0)
                                : stride % 16 == 0 && row % 16 == 0 && pitch == packets * PACKET;
        unit = aligned && wide ? 16u : NARROW;
    }
};

// A window's copy in shared memory as `repack::convert_tile` addresses its
// source: row `n`'s packet `packet` at its place in the copy, rows `pitch`
// bytes apart, from the window's first packet `first`.
template <class B> struct Copy {
    const u8 *copy;
    u64 first;
    u32 pitch;

    __device__ __forceinline__ const u8 *packet(const seismic_words_t &, u64, u64 n, u64 packet) const {
        return copy + (n % Window<B>::ROWS) * pitch + (packet - first) * Window<B>::PACKET;
    }
    __device__ __forceinline__ u64 row_bytes(const seismic_words_t &) const {
        return pitch;
    }
};

// The largest of `words`, at least 1.
__host__ __device__ constexpr u32 most() {
    return 1u;
}
template <class... Rest> __host__ __device__ constexpr u32 most(u32 first, Rest... rest) {
    return first > most(rest...) ? first : most(rest...);
}

// A converting block's whole static shared memory but a written grid's place
// (`seismic_written_block` and `_grid`, two `uint3`s, reserved whether or
// not the launch has one), over its includer's control words `Words` and the
// bindings `Bs` it converts through: the window, the words, and the stages of each warp
// that places tiles (`repack::shared_words`, a warp's WARP_TILES tiles' at
// once, the largest of the bindings'). Those warps are the launch's
// (WINDOW_THREADS, cuda_streaming.seismic), as many as CUDA's 48 KB of static
// shared memory holds beside the rest; any further warps only copy.
template <class Words, class... Bs> struct Shared {
    static constexpr u32 ALIGNMENT = 16u;
    static constexpr u32 WORDS = most(repack::shared_words<Bs>()...);
    static constexpr u32 BLOCK_WARPS = SEISMIC_CONST_WINDOW_THREADS / 32u;
    // CUDA's static shared memory, less the written grid's place, each
    // variable at most ALIGNMENT - 1 bytes of padding from the last.
    static constexpr u32 STATIC_BYTES = 48u * 1024u - (2u * u32(sizeof(uint3)) + 2u * (ALIGNMENT - 1u));
    // The window, the words, then this struct's tail padding.
    static constexpr u32 HELD = SEISMIC_CONST_WINDOW_BYTES + u32(sizeof(Words)) + (ALIGNMENT - 1u);
    static_assert(HELD < STATIC_BYTES, "the window and the words fit");
    static constexpr u32 FIT = (STATIC_BYTES - HELD) / (4u * WORDS);
    static constexpr u32 STAGES = FIT < BLOCK_WARPS ? FIT : BLOCK_WARPS;
    static_assert(STAGES > 0, "a warp's stage fits beside the window");

    alignas(ALIGNMENT) u8 window[SEISMIC_CONST_WINDOW_BYTES];
    Words words;
    u32 stages[STAGES][WORDS];
};

// A packed record's items are its row tiles, each converted window by
// window and written whole from the mirror (`convert_window`); a dense
// record's, its rows.
template <class B> __device__ __forceinline__ u64 record_items(const seismic_words_t &seismic_words_value) {
    if constexpr (B::dense())
        return B::n(seismic_words_value);
    else
        return repack::tiles_y<B>(seismic_words_value);
}

// The element types of a dense binding's source and destination.
template <class B> struct Dense {
    typedef typename element::Of<B::source_kind()>::type Source;
    typedef typename element::Of<B::destination_kind()>::type Destination;
};

// A row tile's destination bytes: its rows are whole and consecutive in each
// layout (a row's packets or planes before the next row's, an mma16 tile's
// rows whole within a row tile, mma16's padding rows the last tile's own).
template <class B> __device__ __forceinline__ u64 tile_bytes(const seismic_words_t &seismic_words_value) {
    return repack::Shape<B>::TILE_ROWS * B::out_stride_1(seismic_words_value) * B::out_stride_bytes(seismic_words_value);
}

// Row `n` of dense staged record `record` of `source` into slot `slot` of
// `destination`, by the whole block: each element decoded to F32 and rounded
// into the destination, `import_dense`'s conversion. Staged elements are read
// with streaming loads, as a window's are.
template <class B>
__device__ __forceinline__ void convert_row(const u8 *source, u8 *destination,
                                            const seismic_words_t &seismic_words_value, u64 record, u64 slot, u64 n) {
    typedef typename Dense<B>::Source Source;
    typedef typename Dense<B>::Destination Destination;
    const typename Source::storage *from =
        reinterpret_cast<const typename Source::storage *>(source + record * B::source_matrix_bytes(seismic_words_value)) +
        n * B::source_stride_1(seismic_words_value);
    u8 *out = destination + slot * B::out_matrix_bytes(seismic_words_value);
    for (u64 k = threadIdx.x; k < B::k(seismic_words_value); k += blockDim.x)
        element::put<Destination>(out, n * B::out_stride_1(seismic_words_value) + k * B::out_stride_2(seismic_words_value),
                                  Source::load(__ldcs(from + k * B::source_stride_2(seismic_words_value))));
}

// The stored record a list entry's `from` names: source record `from` of
// `conveyor`, or, negative, scratch slot -1 - from of `destination`, a slot
// no pass reads, which the host read the stored bytes into at its base
// (`routed_round` in routed.seismic). Either is at record 0 of the base
// returned.
template <class B>
__device__ __forceinline__ const u8 *listed_source(const u8 *conveyor, const u8 *destination,
                                                   const seismic_words_t &seismic_words_value, int from) {
    return from >= 0 ? conveyor + (u64)from * B::source_matrix_bytes(seismic_words_value)
                     : destination + (u64)(-1 - from) * B::out_matrix_bytes(seismic_words_value);
}

// Where item `item` of a packed record lies (`window`'s windowing): its row
// tile and window, the tile's rows below N, the window's first packet and
// its column tiles, its stored bytes (`spans` spans of `span_bytes` bytes,
// `span_stride` apart from `from`: the whole row tile's one span, or its
// rows' shares), and the record's destination.
template <class B> struct Placed {
    u32 row_tile, window_index, rows, packet0, tile0, tiles, spans, span_bytes;
    const u8 *from;
    u64 span_stride;
    u8 *out;

    __device__ __forceinline__ Placed(const Window<B> &window, const u8 *source, u8 *destination,
                                      const seismic_words_t &seismic_words_value, u64 record, u64 slot, u64 item) {
        typedef Window<B> W;
        row_tile = (u32)(item / window.windows);
        window_index = (u32)(item % window.windows);
        const u64 dim_n = B::n(seismic_words_value);
        const u64 row0 = (u64)row_tile * W::ROWS;
        rows = row0 < dim_n ? (u32)(dim_n - row0 < W::ROWS ? dim_n - row0 : W::ROWS) : 0u;
        packet0 = window_index * window.packets;
        const u32 stored = streaming_window_stored(packet0, window.packets, window.source_packets);
        tile0 = packet0 * repack::Shape<B>::UNITS;
        tiles = (window_index + 1 == window.windows ? window.tiles_x : tile0 + window.packets * repack::Shape<B>::UNITS) -
                tile0;
        const u64 row_stride = B::source_stride_1(seismic_words_value) * W::PACKET;
        spans = window.whole ? (rows > 0 ? 1u : 0u) : rows;
        span_bytes = !window.whole ? stored * W::PACKET : rows > 0 ? (rows - 1u) * window.pitch + stored * W::PACKET : 0u;
        span_stride = row_stride;
        from = source + record * B::source_matrix_bytes(seismic_words_value) +
               (row0 * B::source_stride_1(seismic_words_value) + packet0 * B::source_stride_2(seismic_words_value)) *
                   W::PACKET;
        out = destination + slot * B::out_matrix_bytes(seismic_words_value);
    }
};

// The block's mirror of a row tile: the launch's dynamic shared memory.
__device__ __forceinline__ u8 *mirror() {
    extern __shared__ uint4 streaming_mirror[];
    return reinterpret_cast<u8 *>(streaming_mirror);
}

// `repack::convert_tile`'s destination policy into the mirror: a row tile's
// row `n` at its place in the tile.
template <class B> struct Mirror {
    __device__ static __forceinline__ u8 *row(u8 *mirror, const seismic_words_t &seismic_words_value, u64, u64 n) {
        return mirror + (n % repack::Shape<B>::TILE_ROWS) * B::out_stride_1(seismic_words_value) *
                            B::out_stride_bytes(seismic_words_value);
    }
};

// Whether a binding's row tiles are placed through the mirror: they fit it.
template <class B> __device__ __forceinline__ bool mirrored(const seismic_words_t &seismic_words_value) {
    return tile_bytes<B>(seismic_words_value) <= SEISMIC_CONST_MIRROR_BYTES;
}

// The window's tiles of `placed`, its copy in `shared.window`, placed by
// `repack::convert_tile` through `O` into `out` (the record's slot or the
// mirror) by the block's first `Sh::STAGES` warps, each placing its
// WARP_TILES tiles at once, a share of its lanes each staging in its own
// stage words. A warp's tiles at once lie a group apart (`Shape::UNITS`
// tiles: the same unit of consecutive groups), so its lanes run the same
// code.
template <class B, class O, class Sh>
__device__ __forceinline__ void convert_tiles(const Window<B> &window, const Placed<B> &placed, u8 *out,
                                              const seismic_words_t &seismic_words_value, Sh &shared) {
    typedef repack::Shape<B> S;
    constexpr u32 TILES = S::WARP_TILES, LANES = 32u / TILES;
    const u32 warp = threadIdx.x / 32, share = threadIdx.x % 32 / LANES, lane = threadIdx.x % LANES;
    const u32 warps = blockDim.x / 32 < Sh::STAGES ? blockDim.x / 32 : Sh::STAGES;
    const Copy<B> copy{shared.window, placed.packet0, window.pitch};
    // A window's tiles are whole groups (`Placed`): rounds of a warp's tiles
    // at once, each round's at its first's unit.
    const u32 rounds = (placed.tiles + TILES * S::UNITS - 1) / (TILES * S::UNITS) * S::UNITS;
    if (warp < warps)
        for (u32 round = warp; round < rounds; round += warps) {
            const u32 tile = round / S::UNITS * (TILES * S::UNITS) + round % S::UNITS + share * S::UNITS;
            if (tile < placed.tiles) {
                repack::convert_tile<B, Copy<B>, O, LANES>(copy, out, seismic_words_value, 0, placed.tile0 + tile,
                                                           placed.row_tile, lane,
                                                           shared.stages[warp] + share * S::SHARED_WORDS);
                // The share's next tile restages its stage: each of its
                // lanes is done with this one's.
                __syncwarp(repack::lanes_mask<LANES>());
            }
        }
}

// The mirror's first `bytes` bytes at `to`, by the block in `Unit`s.
template <class Unit> __device__ __forceinline__ void write_units(u8 *to, u64 bytes) {
    const u8 *from = mirror();
    for (u64 at = threadIdx.x * sizeof(Unit); at < bytes; at += blockDim.x * sizeof(Unit))
        *reinterpret_cast<Unit *>(to + at) = *reinterpret_cast<const Unit *>(from + at);
}

// The mirror's stored rows of `placed`'s row tile written to the record by
// the block, in the widest units the record keeps: whole lines.
template <class B>
__device__ __forceinline__ void write_tile(const Placed<B> &placed, const seismic_words_t &seismic_words_value) {
    typedef repack::Shape<B> S;
    const u64 row_bytes = B::out_stride_1(seismic_words_value) * B::out_stride_bytes(seismic_words_value);
    // Stored rows: mma16's are whole mma16 tiles, rows beyond N zero.
    const u64 n = B::n(seismic_words_value);
    const u64 stored = B::mma16() ? (n + S::MMA_ROWS - 1) / S::MMA_ROWS * S::MMA_ROWS : n;
    const u64 row0 = (u64)placed.row_tile * S::TILE_ROWS;
    const u64 bytes = (stored - row0 < S::TILE_ROWS ? stored - row0 : S::TILE_ROWS) * row_bytes;
    u8 *to = placed.out + row0 * row_bytes;
    const u64 alignment = reinterpret_cast<u64>(to) | bytes;
    if (alignment % 16 == 0)
        write_units<uint4>(to, bytes);
    else if (alignment % 8 == 0)
        write_units<uint2>(to, bytes);
    else if (alignment % 4 == 0)
        write_units<u32>(to, bytes);
    else if (alignment % 2 == 0)
        write_units<element::u16>(to, bytes);
    else
        write_units<u8>(to, bytes);
}

// A window's stored bytes as the block loads them (`Placed`'s, any
// binding's): `spans` spans of `bytes` bytes, `stride` apart from `from`, in
// units of `unit` bytes.
struct Span {
    const u8 *from;
    u64 stride;
    u32 spans, bytes, unit;
};

template <class B> __device__ __forceinline__ Span span_of(const Window<B> &window, const Placed<B> &placed) {
    return Span{placed.from, placed.span_stride, placed.spans, placed.span_bytes, window.unit};
}

// A thread's share of a window's stored bytes held in registers while the
// block converts the window before it: the loads of one window are in
// flight beside the conversion of the last, so a conveyor read across the
// link streams instead of stopping at every window: 32 records of Q4_K
// convert at 11.0 GB/s (1.714 ms) against the link's 12.9 for a plain
// mapped read (an RTX 3090, 2026-11-02), a measured gap of ~15% the
// convert's own stalls leave. A thread holds units of the window's `unit`
// bytes, unit `u` of its spans' the block's `u % WINDOW_THREADS`th thread's,
// packed into WORDS 16-byte registers whatever the unit (a window is
// WINDOW_BYTES, so every unit's share fits them), so one `Held` serves
// every binding a block converts.
struct Held {
    static constexpr u32 WORDS = (SEISMIC_CONST_WINDOW_BYTES / 16u + SEISMIC_CONST_WINDOW_THREADS - 1u) /
                                 SEISMIC_CONST_WINDOW_THREADS;
    template <u32 UNIT> __host__ __device__ static constexpr u32 units() {
        return (SEISMIC_CONST_WINDOW_BYTES / UNIT + SEISMIC_CONST_WINDOW_THREADS - 1u) / SEISMIC_CONST_WINDOW_THREADS;
    }
    static_assert(units<1>() <= 16u * WORDS && units<2>() * 2u <= 16u * WORDS && units<4>() * 4u <= 16u * WORDS &&
                      units<8>() * 8u <= 16u * WORDS && units<16>() * 16u <= 16u * WORDS,
                  "every unit's share fits the held words");
    uint4 words[WORDS];

    // Component `c` of word `w`.
    __device__ __forceinline__ u32 &component(u32 w, u32 c) {
        return c == 0 ? words[w].x : c == 1 ? words[w].y : c == 2 ? words[w].z : words[w].w;
    }
    __device__ __forceinline__ u32 component(u32 w, u32 c) const {
        return c == 0 ? words[w].x : c == 1 ? words[w].y : c == 2 ? words[w].z : words[w].w;
    }

    // Unit `k` of UNIT bytes, at byte k UNIT of the words.
    template <u32 UNIT> __device__ __forceinline__ void put(u32 k, typename LoadUnit<UNIT>::type value) {
        if constexpr (UNIT == 16) {
            words[k] = value;
        } else if constexpr (UNIT == 8) {
            component(k / 2, k % 2 * 2) = value.x;
            component(k / 2, k % 2 * 2 + 1) = value.y;
        } else if constexpr (UNIT == 4) {
            component(k / 4, k % 4) = value;
        } else {
            constexpr u32 PER = 4u / UNIT, MASK = UNIT == 2 ? 0xffffu : 0xffu;
            const u32 shift = k % PER * UNIT * 8u;
            u32 &word = component(k / (4u * PER), k / PER % 4u);
            word = (word & ~(MASK << shift)) | (u32(value) << shift);
        }
    }
    template <u32 UNIT> __device__ __forceinline__ typename LoadUnit<UNIT>::type get(u32 k) const {
        if constexpr (UNIT == 16) {
            return words[k];
        } else if constexpr (UNIT == 8) {
            return uint2{component(k / 2, k % 2 * 2), component(k / 2, k % 2 * 2 + 1)};
        } else if constexpr (UNIT == 4) {
            return component(k / 4, k % 4);
        } else {
            constexpr u32 PER = 4u / UNIT, MASK = UNIT == 2 ? 0xffffu : 0xffu;
            return typename LoadUnit<UNIT>::type((component(k / (4u * PER), k / PER % 4u) >> (k % PER * UNIT * 8u)) & MASK);
        }
    }

    // Each of the thread's units of `span`, `visit(k, span, unit within the
    // span)`, in order: the span and unit carried from one to the next, not
    // divided out of each.
    template <u32 UNIT, class Visit> __device__ __forceinline__ static void each(const Span &span, Visit visit) {
        const u32 span_units = span.bytes / UNIT;
        if (span_units == 0)
            return;
        const u32 step_spans = SEISMIC_CONST_WINDOW_THREADS / span_units,
                  step_units = SEISMIC_CONST_WINDOW_THREADS % span_units;
        u32 at = threadIdx.x / span_units, unit = threadIdx.x % span_units;
#pragma unroll
        for (u32 k = 0; k < units<UNIT>(); ++k) {
            if (at < span.spans)
                visit(k, at, unit);
            at += step_spans;
            unit += step_units;
            if (unit >= span_units) {
                unit -= span_units;
                ++at;
            }
        }
    }

    template <u32 UNIT> __device__ __forceinline__ void load_as(const Span &span) {
        typedef typename LoadUnit<UNIT>::type Unit;
        each<UNIT>(span, [&](u32 k, u32 at, u32 unit) {
            put<UNIT>(k, __ldcs(reinterpret_cast<const Unit *>(span.from + (u64)at * span.stride + (u64)unit * UNIT)));
        });
    }
    template <u32 UNIT> __device__ __forceinline__ void store_as(const Span &span, u32 pitch, u8 *window) const {
        typedef typename LoadUnit<UNIT>::type Unit;
        each<UNIT>(span, [&](u32 k, u32 at, u32 unit) {
            *reinterpret_cast<Unit *>(window + at * pitch + unit * UNIT) = get<UNIT>(k);
        });
    }

    // Load `span`'s window: issued, not awaited, until stored. A launch of
    // WINDOW_THREADS threads over records that keep their windows' unit
    // (views of whole records of 16-byte aligned tensors; a scratch slot
    // keeps it only where its records do), each checked.
    __device__ __forceinline__ void load(const Span &span) {
        if (blockDim.x != SEISMIC_CONST_WINDOW_THREADS || reinterpret_cast<u64>(span.from) % span.unit != 0u)
            __trap();
        switch (span.unit) {
        case 16: load_as<16>(span); break;
        case 8: load_as<8>(span); break;
        case 4: load_as<4>(span); break;
        case 2: load_as<2>(span); break;
        default: load_as<1>(span); break;
        }
    }

    // Store the units loaded of `span`'s window into the block's window at
    // `window`, its spans `pitch` bytes apart.
    __device__ __forceinline__ void store(const Span &span, u32 pitch, u8 *window) const {
        switch (span.unit) {
        case 16: store_as<16>(span, pitch, window); break;
        case 8: store_as<8>(span, pitch, window); break;
        case 4: store_as<4>(span, pitch, window); break;
        case 2: store_as<2>(span, pitch, window); break;
        default: store_as<1>(span, pitch, window); break;
        }
    }
};

// Where window `window_index` of row tile `row_tile` of the record entry
// (from, to) names lies: from stored record `from` of `source` (negative:
// `listed_source`'s scratch slot) into record `to` of `destination`.
template <class B>
__device__ __forceinline__ Placed<B> place_of(const Window<B> &window, const u8 *source, u8 *destination,
                                              const seismic_words_t &seismic_words_value, int from, int to,
                                              u64 row_tile, u64 window_index) {
    return Placed<B>(window, listed_source<B>(source, destination, seismic_words_value, from), destination,
                     seismic_words_value, 0, (u64)to, row_tile * window.windows + window_index);
}

// Window `window_index` of row tile `row_tile` of entry (from, to), its
// stored bytes already in the block's window (`Held::store`), placed by the
// block: through the mirror where the binding's row tiles fit it, written
// whole once its last window is placed (`write_tile`), else into the record.
template <class B, class Sh>
__device__ __forceinline__ void convert_window(const Window<B> &window, const u8 *source, u8 *destination,
                                               const seismic_words_t &seismic_words_value, int from, int to,
                                               u64 row_tile, u64 window_index, Sh &shared) {
    const Placed<B> placed =
        place_of<B>(window, source, destination, seismic_words_value, from, to, row_tile, window_index);
    if (!mirrored<B>(seismic_words_value)) {
        convert_tiles<B, repack::Rows<B>>(window, placed, placed.out, seismic_words_value, shared);
    } else {
        convert_tiles<B, Mirror<B>>(window, placed, mirror(), seismic_words_value, shared);
        if (placed.window_index + 1 == window.windows) {
            __syncthreads();
            write_tile<B>(placed, seismic_words_value);
        }
    }
}

// Convert the row tiles `first`, `first + step`, ... of a list's packed
// records, whose items (`per_record` a record, `end` in all) are its row
// tiles: each window loaded into registers (`Held`) while the one before it
// converts, each row tile placed in the mirror and written whole
// (`write_tile`), or, larger than the mirror, placed in the record. `pair(at)`,
// called by every thread, gives item `at`'s (from, to), `from` as
// `listed_source` reads it.
template <class B, class Sh, class Pair>
__device__ __forceinline__ void convert_records(const u8 *source, u8 *destination,
                                                const seismic_words_t &seismic_words_value, u64 end, u64 first,
                                                u64 step, u64 per_record, Pair pair, Sh &shared) {
    static_assert(!B::dense(), "a dense record converts row by row");
    const Window<B> window(seismic_words_value);
    const bool through_mirror = mirrored<B>(seismic_words_value);
    const u64 windows = window.windows, last = end * windows;
    // The window after `at`: its row tile's next, else the first of the
    // block's next row tile.
    const auto next = [&](u64 at) { return (at + 1) % windows != 0 ? at + 1 : at + 1 - windows + step * windows; };
    const auto place = [&](u64 at) {
        const int2 fromto = pair(at / windows);
        return place_of<B>(window, source, destination, seismic_words_value, fromto.x, fromto.y,
                           at / windows % per_record, at % windows);
    };
    Held held;
    u64 at = first * windows;
    Placed<B> placed = at < last ? place(at) : Placed<B>(window, source, destination, seismic_words_value, 0, 0, 0);
    Span span = span_of(window, placed);
    if (at < last)
        held.load(span);
    while (at < last) {
        __syncthreads();
        held.store(span, window.pitch, shared.window);
        __syncthreads();
        const Placed<B> current = placed;
        at = next(at);
        if (at < last) {
            placed = place(at);
            span = span_of(window, placed);
            held.load(span);
        }
        if (!through_mirror) {
            convert_tiles<B, repack::Rows<B>>(window, current, current.out, seismic_words_value, shared);
        } else {
            convert_tiles<B, Mirror<B>>(window, current, mirror(), seismic_words_value, shared);
            if (current.window_index + 1 == windows) {
                __syncthreads();
                write_tile<B>(current, seismic_words_value);
            }
        }
    }
    __syncthreads();
}

} // namespace streaming
