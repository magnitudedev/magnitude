// Expert streaming's conversion of staged records, by windows; the CUDA form
// of `metal/lib/streaming/window.h`. A block converts whole row tiles of a
// record (its conversion's tile rows), one window of packets along them at a
// time: each thread loads its share of the window's stored bytes from
// staging into registers in units of the source packet's alignment
// (`Prefetched`), while the block converts the window before it (staging is
// mapped host memory, each load a PCIe round trip on a discrete device, so
// every row's loads are in flight together), then stores them into shared
// memory, where its warps place the window's column tiles by
// `repack::convert_tile` (lib/repack/repack.cuh), its source the copy. The
// tiles are placed in the block's mirror of the row tile (`Mirror`, the
// launch's dynamic shared memory, MIRROR_BYTES), which the block writes to
// the record whole once its last window is placed (`write_tile`): a tile
// conversion's own stores are a lane per row, narrow and row-strided, which
// GB10's host-NUMA memory takes at a quarter of device memory's rate
// (148-158 against 35-41 us a 35B record), while whole lines take both at
// device speed (30-36 and 29-35 us, 2026-11-03).
// A row tile larger than the mirror is placed in the record directly. The tile
// conversion is `repack_weight`'s, so a slot holds exactly the bytes a
// resident import places. A dense float record (`B::dense()`: `repack`
// between dense representations) converts row by row, each element as
// `import_dense` converts it (decoded to F32, rounded into the destination).
//
// Templates over a repack binding `B` (`REPACK_BINDING`), so a kernel
// converts through as many bindings as it holds, in one `Shared` of them.
// The includer includes lib/streaming/control.cuh first.

#include "../repack/repack.cuh"

namespace streaming {

using element::u32;
using element::u64;
using element::u8;

// A window: a tile's rows of `packets()` source packets, each row's share
// padded to 16 bytes, within the module's WINDOW_BYTES
// (`SEISMIC_CONST_WINDOW_BYTES`). A tile reads only its own columns' packets,
// so a window is any whole number of packets. It is the source of the
// window's tiles (`repack::convert_tile`'s addressing): row `n`'s packet
// `packet` at its place in the copy.

template <class B> struct Window {
    __host__ __device__ static constexpr u32 rows() {
        return repack::Shape<B>::TILE_ROWS;
    }
    __host__ __device__ static constexpr u32 packet_bytes() {
        return B::source_packet_size();
    }
    __host__ __device__ static constexpr u32 group() {
        return B::group();
    }
    __host__ __device__ static constexpr u32 packets() {
        return SEISMIC_CONST_WINDOW_BYTES / (rows() * packet_bytes());
    }
    __host__ __device__ static constexpr u32 pitch() {
        return (packets() * packet_bytes() + 15u) / 16u * 16u;
    }
    static_assert(packets() > 0, "a window holds a packet of each row");
    static_assert(rows() * pitch() <= SEISMIC_CONST_WINDOW_BYTES, "a window fits its shared memory");
    // Column tiles of a window.
    __host__ __device__ static constexpr u32 tiles() {
        return packets() * group() / repack::Shape<B>::TILE_COLUMNS;
    }
    // Packet `packet` of row `n` in the window's copy, and the bytes between
    // its rows.
    __device__ static __forceinline__ const u8 *packet(const u8 *window, const seismic_words_t &, u64, u64 n,
                                                       u64 packet) {
        return window + (n % rows()) * pitch() + (packet % packets()) * packet_bytes();
    }
    __device__ static __forceinline__ u64 row_bytes(const seismic_words_t &) {
        return pitch();
    }
};

// The largest of `words`, at least 1.
__host__ __device__ constexpr u32 most() {
    return 1u;
}
template <class... Rest> __host__ __device__ constexpr u32 most(u32 first, Rest... rest) {
    return first > most(rest...) ? first : most(rest...);
}

// A converting block's whole static shared memory but its written grid's
// place (the launch has a written grid: `seismic_written_block` and
// `_grid`), over its includer's control words `Words` and the bindings `Bs`
// it converts through: the window, the words, and a stage for each warp
// that places tiles (`repack::Shape::SHARED_WORDS`, the largest of the
// bindings'). Those warps are the launch's (WINDOW_THREADS,
// cuda_streaming.seismic), as many as CUDA's 48 KB of static shared memory
// holds beside the rest; any further warps only copy.
template <class Words, class... Bs> struct Shared {
    static constexpr u32 ALIGNMENT = 16u;
    static constexpr u32 WORDS = most(repack::shared_words<Bs>()...);
    static constexpr u32 BLOCK_WARPS = SEISMIC_CONST_WINDOW_THREADS / 32u;
    // CUDA's static shared memory, less the written grid's place, each
    // variable at most ALIGNMENT - 1 bytes of padding from the last.
    static constexpr u32 STATIC_BYTES = 48u * 1024u -
        (u32(sizeof(seismic_written_block) + sizeof(seismic_written_grid)) + 2u * (ALIGNMENT - 1u));
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

// Windows along a row, and items of one record.
template <class B> __device__ __forceinline__ u64 record_windows(const seismic_words_t &seismic_words_value) {
    return (repack::tiles_x<B>(seismic_words_value) + Window<B>::tiles() - 1) / Window<B>::tiles();
}

// A packed record's items are its windows; a dense record's, its rows.
template <class B> __device__ __forceinline__ u64 record_items(const seismic_words_t &seismic_words_value) {
    if constexpr (B::dense())
        return B::n(seismic_words_value);
    else
        return repack::tiles_y<B>(seismic_words_value) * record_windows<B>(seismic_words_value);
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

// Where item `item` of a packed record lies: its window's first row and
// rows below N, its first packet and packets below the row's, and the
// record's source and destination.
template <class B> struct Placed {
    u32 row_tile, window_index, rows, stored_packets;
    const u8 *from;
    u64 row_stride;
    u8 *out;

    __device__ __forceinline__ Placed(const u8 *source, u8 *destination, const seismic_words_t &seismic_words_value,
                                      u64 record, u64 slot, u64 item) {
        typedef Window<B> W;
        const u64 windows = record_windows<B>(seismic_words_value);
        row_tile = (u32)(item / windows);
        window_index = (u32)(item % windows);
        const u64 dim_n = B::n(seismic_words_value);
        const u64 packets = (B::k(seismic_words_value) + W::group() - 1) / W::group();
        const u64 packet0 = (u64)window_index * W::packets();
        stored_packets = packet0 < packets ? (u32)(packets - packet0 < W::packets() ? packets - packet0 : W::packets()) : 0u;
        const u64 row0 = (u64)row_tile * W::rows();
        rows = row0 < dim_n ? (u32)(dim_n - row0 < W::rows() ? dim_n - row0 : W::rows()) : 0u;
        row_stride = B::source_stride_1(seismic_words_value) * W::packet_bytes();
        from = source + record * B::source_matrix_bytes(seismic_words_value) +
               (row0 * B::source_stride_1(seismic_words_value) + packet0 * B::source_stride_2(seismic_words_value)) *
                   W::packet_bytes();
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
// mirror), each of the block's first `Sh::STAGES` warps staging in its own
// stage.
template <class B, class O, class Sh>
__device__ __forceinline__ void convert_tiles(const Placed<B> &placed, u8 *out,
                                              const seismic_words_t &seismic_words_value, Sh &shared) {
    typedef Window<B> W;
    const u64 tiles_x = repack::tiles_x<B>(seismic_words_value);
    const u32 warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const u32 warps = blockDim.x / 32 < Sh::STAGES ? blockDim.x / 32 : Sh::STAGES;
    if (warp < warps)
        for (u32 tile = warp; tile < W::tiles(); tile += warps) {
            const u32 tile_x = placed.window_index * W::tiles() + tile;
            if (tile_x < tiles_x) {
                repack::convert_tile<B, W, O>(shared.window, out, seismic_words_value, 0, tile_x, placed.row_tile,
                                              lane, shared.stages[warp]);
                // The warp's next tile restages its stage: every lane is
                // done with this one's.
                __syncwarp();
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

// The unit a window's stored bytes load in: the widest, to 16 bytes, its
// packets keep (a conveyor record's rows, windows and records are whole
// packets, from a 16-byte aligned conveyor).
template <u32 BYTES> struct LoadUnit;
template <> struct LoadUnit<16> { typedef uint4 type; };
template <> struct LoadUnit<8> { typedef uint2 type; };
template <> struct LoadUnit<4> { typedef u32 type; };
template <> struct LoadUnit<2> { typedef element::u16 type; };
template <> struct LoadUnit<1> { typedef u8 type; };

__host__ __device__ constexpr u32 load_unit(u32 packet) {
    return packet % 16 == 0 ? 16u : packet % 8 == 0 ? 8u : packet % 4 == 0 ? 4u : packet % 2 == 0 ? 2u : 1u;
}

// A window's stored bytes held in a thread's registers while the block
// converts the window before it: the loads of one window are in flight
// beside the conversion of the last, so a conveyor read across the link
// streams instead of stopping at every window: 32 records of Q4_K convert
// at 11.0 GB/s (1.714 ms) against the link's 12.9 for a plain mapped read
// (an RTX 3090, 2026-11-02), a measured gap of ~15% the convert's own stalls
// leave. A
// thread holds UNITS units of the window, unit `u` of it the block's
// `u % threads`th thread's.
template <class B> struct Prefetched {
    static constexpr u32 UNIT = load_unit(Window<B>::packet_bytes());
    typedef typename LoadUnit<UNIT>::type Unit;
    static constexpr u32 UNITS = (SEISMIC_CONST_WINDOW_BYTES / UNIT + SEISMIC_CONST_WINDOW_THREADS - 1u) /
                                 SEISMIC_CONST_WINDOW_THREADS;
    Unit units[UNITS];

    // Each of the thread's units of `placed`'s window, `visit(k, row, unit
    // within the row)`, in order: the row and unit carried from one to the
    // next, not divided out of each.
    template <class Visit> __device__ __forceinline__ static void each(const Placed<B> &placed, Visit visit) {
        const u32 row_units = placed.stored_packets * Window<B>::packet_bytes() / UNIT;
        if (row_units == 0)
            return;
        const u32 step_rows = SEISMIC_CONST_WINDOW_THREADS / row_units,
                  step_units = SEISMIC_CONST_WINDOW_THREADS % row_units;
        u32 row = threadIdx.x / row_units, unit = threadIdx.x % row_units;
#pragma unroll
        for (u32 k = 0; k < UNITS; ++k) {
            if (row < placed.rows)
                visit(k, row, unit);
            row += step_rows;
            unit += step_units;
            if (unit >= row_units) {
                unit -= row_units;
                ++row;
            }
        }
    }

    // Load `placed`'s window: issued, not awaited, until stored.
    __device__ __forceinline__ void load(const Placed<B> &placed) {
        each(placed, [&](u32 k, u32 row, u32 unit) {
            units[k] = __ldcs(reinterpret_cast<const Unit *>(placed.from + (u64)row * placed.row_stride + (u64)unit * UNIT));
        });
    }

    // Store the units loaded of `placed`'s window into the block's window.
    template <class Sh> __device__ __forceinline__ void store(const Placed<B> &placed, Sh &shared) const {
        each(placed, [&](u32 k, u32 row, u32 unit) {
            *reinterpret_cast<Unit *>(shared.window + row * Window<B>::pitch() + unit * UNIT) = units[k];
        });
    }
};

// Convert the row tiles `first`, `first + step`, ... of a list's packed
// records, whose items (`per_record` a record, `end` in all) are its row
// tiles' windows in turn: each window loaded into registers (`Prefetched`)
// while the one before it converts, each row tile placed in the mirror and
// written whole (`write_tile`), or, larger than the mirror, placed in the
// record. `pair(at)`, called by every thread, gives item `at`'s (from, to).
// A launch of WINDOW_THREADS threads over a conveyor whose records keep
// `Prefetched`'s unit (a view of whole records of a 16-byte aligned tensor).
template <class B, class Sh, class Pair>
__device__ __forceinline__ void convert_records(const u8 *source, u8 *destination,
                                                const seismic_words_t &seismic_words_value, u64 end, u64 first,
                                                u64 step, u64 per_record, Pair pair, Sh &shared) {
    static_assert(!B::dense(), "a dense record converts row by row");
    if (blockDim.x != SEISMIC_CONST_WINDOW_THREADS || reinterpret_cast<u64>(source) % Prefetched<B>::UNIT != 0u)
        __trap();
    const bool through_mirror = mirrored<B>(seismic_words_value);
    const u64 windows = record_windows<B>(seismic_words_value);
    // The item after `at`: its row tile's next window, else the first of the
    // block's next row tile.
    const auto next = [&](u64 at) { return (at + 1) % windows != 0 ? at + 1 : at + 1 - windows + step * windows; };
    const auto place = [&](u64 at) {
        const int2 fromto = pair(at);
        return Placed<B>(source, destination, seismic_words_value, (u64)fromto.x, (u64)fromto.y, at % per_record);
    };
    Prefetched<B> prefetched;
    u64 at = first * windows;
    Placed<B> placed = at < end ? place(at) : Placed<B>(source, destination, seismic_words_value, 0, 0, 0);
    if (at < end)
        prefetched.load(placed);
    while (at < end) {
        __syncthreads();
        prefetched.store(placed, shared);
        __syncthreads();
        const Placed<B> current = placed;
        at = next(at);
        if (at < end) {
            placed = place(at);
            prefetched.load(placed);
        }
        if (!through_mirror) {
            convert_tiles<B, repack::Rows<B>>(current, current.out, seismic_words_value, shared);
        } else {
            convert_tiles<B, Mirror<B>>(current, mirror(), seismic_words_value, shared);
            if (current.window_index + 1 == windows) {
                __syncthreads();
                write_tile<B>(current, seismic_words_value);
            }
        }
    }
    __syncthreads();
}

} // namespace streaming
