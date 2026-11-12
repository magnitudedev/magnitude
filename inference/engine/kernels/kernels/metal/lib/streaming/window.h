// Expert streaming's conversion of staged records, by windows: one item is a row tile of a
// record (16 rows) and a window of its packets along those rows (`Windows`).
// A threadgroup copies the item's stored bytes from staging into threadgroup
// memory with the widest aligned system-coherent loads (the host writes
// staging within the step's command buffer; GGUF rows are contiguous), then
// its simdgroups place the window's column tiles by `repack_tile`
// (lib/repack/repack.h, the registered conversion), two tiles a simdgroup,
// reading the copy. The tile placement is `repack_weight`'s, so a slot holds
// exactly the bytes a resident import places. A dense float record
// (`B::dense()`: `repack` between dense representations) converts row by
// row, each element as `import_dense` converts it (decoded to F32, rounded
// into the destination).
//
// Templates over a repack binding `B` (`REPACK_BINDING`), so a kernel
// converts through as many bindings as it holds. The includer includes
// `control.h` first.

#include <seismic/slab.h>
#include "../repack/repack.h"
#include "../../../streaming/window.hpp"

#define STAGED volatile coherent(system) device const uchar

namespace streaming {

// How a packed record's row tiles divide into windows, the rule every
// backend asks (`streaming/window.hpp`) over this record's rows: a window
// holds `packets` packets of each of the tile's 16 rows, the rows' shares
// `pitch` bytes apart, within the module's WINDOW_BYTES
// (`SEISMIC_CONST_WINDOW_BYTES`, the includer's threadgroup memory); `whole`
// where the window is the whole tile, copied as one span. A row's packets are
// its stored columns' (`repack::tiles_x`): a window past the source row's
// packets copies none and places zero tiles.
template <class B> struct Windows {
    uint packets;
    uint count;
    uint pitch;
    bool whole;

    static constexpr uint rows() {
        return repack::TILE_ROWS;
    }
    static constexpr uint bytes() {
        return uint(SEISMIC_CONST_WINDOW_BYTES);
    }
    static constexpr uint packet_bytes() {
        return uint(B::source_packet_size());
    }
    // Column tiles of one packet (a tile is whole chunks of a group).
    static constexpr uint packet_tiles() {
        return B::group() / repack::tile_columns<B>();
    }
    static_assert(256 % B::group() == 0, "a 256-column block is whole packets");
    static_assert(STREAMING_WINDOW_FITS(repack::TILE_ROWS, B::source_packet_size(), SEISMIC_CONST_WINDOW_BYTES),
                  "a window holds a packet of each row");

    static Windows of(constant ulong *seismic_words) {
        const uint stored = streaming_window_row_packets(
            ulong(repack::tiles_x<B>(seismic_words)) * repack::tile_columns<B>(), uint(B::group()));
        const uint stride = uint(B::source_stride_1(seismic_words)) * packet_bytes();
        const uint packets = streaming_window_packets(rows(), stored, packet_bytes(), stride, bytes());
        return Windows{packets, streaming_window_count(stored, packets),
                       streaming_window_pitch(rows(), packet_bytes(), packets, stride, bytes()),
                       streaming_window_whole(rows(), stride, bytes())};
    }
};

// An item's window as `repack_tile` reads it (its addressing policy and
// source): the copy, its rows' pitch and a window's packets.
template <class B> struct Window {
    threadgroup const uchar *copy;
    uint pitch;
    uint packets;

    // Packet `packet` of row `n` in the window's copy.
    static threadgroup const uchar *packet(Window window, constant ulong *, ulong, ulong n, ulong packet) {
        return window.copy + (n % repack::TILE_ROWS) * window.pitch +
               (packet % window.packets) * Windows<B>::packet_bytes();
    }
};

// Copy the `bytes` staged bytes at byte `offset` of `source` to `to`
// (16-byte aligned) with the threadgroup's `threads` threads, in the widest
// loads the bytes' address admits.
inline void copy_staged(STAGED *source, ulong offset, threadgroup uchar *to, uint bytes, uint tid, uint threads) {
    STAGED *from = source + offset;
    const ulong address = reinterpret_cast<ulong>(from);
    if (address % 16 == 0) {
        for (uint i = tid; i < bytes / 16; i += threads)
            reinterpret_cast<threadgroup uint4 *>(to)[i] =
                reinterpret_cast<volatile coherent(system) device const uint4 *>(from)[i];
        for (uint i = bytes / 16 * 16 + tid; i < bytes; i += threads)
            to[i] = from[i];
    } else if (address % 4 == 0) {
        for (uint i = tid; i < bytes / 4; i += threads)
            reinterpret_cast<threadgroup uint *>(to)[i] =
                reinterpret_cast<volatile coherent(system) device const uint *>(from)[i];
        for (uint i = bytes / 4 * 4 + tid; i < bytes; i += threads)
            to[i] = from[i];
    } else if (address % 2 == 0) {
        for (uint i = tid; i < bytes / 2; i += threads)
            reinterpret_cast<threadgroup ushort *>(to)[i] =
                reinterpret_cast<volatile coherent(system) device const ushort *>(from)[i];
        for (uint i = bytes / 2 * 2 + tid; i < bytes; i += threads)
            to[i] = from[i];
    } else {
        // Odd-sized packets (17-byte MXFP4) can start a share at an odd offset.
        for (uint i = tid; i < bytes; i += threads)
            to[i] = from[i];
    }
}

// Copy `rows` rows' shares of `bytes` staged bytes, the first at byte
// `offset` of `source` and each the next `stride` bytes on, to `to` (16-byte
// aligned), the next `pitch` bytes on: where every share is whole 16-byte
// loads, the threadgroup's `threads` threads stride over all of them at
// once; else row by row, in the loads each share's address admits.
inline void copy_rows(STAGED *source, ulong offset, ulong stride, threadgroup uchar *to, uint pitch, uint rows,
    uint bytes, uint tid, uint threads) {
    if ((reinterpret_cast<ulong>(source + offset) | stride | pitch | bytes) % 16 == 0) {
        const uint loads = bytes / 16;
        for (uint i = tid; i < rows * loads; i += threads) {
            const uint row = i / loads, load = i % loads;
            reinterpret_cast<threadgroup uint4 *>(to + row * pitch)[load] =
                reinterpret_cast<volatile coherent(system) device const uint4 *>(source + offset + row * stride)[load];
        }
    } else {
        for (uint row = 0; row < rows; ++row)
            copy_staged(source, offset + row * stride, to + row * pitch, bytes, tid, threads);
    }
}

// A packed record's items are its row tiles' windows; a dense record's, its
// rows.
template <class B> inline ulong record_items(constant ulong *seismic_words) {
    if constexpr (B::dense())
        return B::n(seismic_words);
    else
        return repack::tiles_y<B>(seismic_words) * Windows<B>::of(seismic_words).count;
}

// The element types of a dense binding's source and destination.
template <class B> struct Dense {
    typedef typename element::Of<B::source_kind()>::type Source;
    typedef typename element::Of<B::destination_kind()>::type Destination;
};

// Record `record` of a source record port (its records' address table).
inline STAGED *source_record(device const ulong *table, ulong record) {
    return reinterpret_cast<STAGED *>(slab::region<const uchar>(table, record));
}

// Row `n` of the dense source record at `record` into the destination record
// at `out`: each element decoded to F32 and rounded into the destination,
// `import_dense`'s conversion.
template <class B>
inline void convert_row(STAGED *record, device uchar *out, constant ulong *seismic_words, ulong n, uint tid,
    uint threads) {
    typedef typename Dense<B>::Source Source;
    typedef typename Dense<B>::Destination Destination;
    volatile coherent(system) device const typename Source::storage *from =
        reinterpret_cast<volatile coherent(system) device const typename Source::storage *>(record);
    for (ulong k = tid; k < B::k(seismic_words); k += threads)
        element::put<Destination>(out, n * B::out_stride_1(seismic_words) + k * B::out_stride_2(seismic_words),
            Source::load(from[n * B::source_stride_1(seismic_words) + k * B::source_stride_2(seismic_words)]));
}

template <class B>
inline void convert_window(STAGED *record, device uchar *out, constant ulong *seismic_words, ulong item,
    threadgroup uchar *window, uint tid, uint threads, uint simd, uint simdgroups, uint lane);

// Item `item` (< record_items) of the source record at `record` into the
// destination record at `out`, by the whole threadgroup (`threads` threads in
// `simdgroups` simdgroups): a dense record's row by element conversion, a
// packed record's window by `repack_tile`. Every thread calls it; it begins
// and ends with a threadgroup barrier, so `window` is free to reuse after it.
template <class B>
inline void convert_item(STAGED *record, device uchar *out, constant ulong *seismic_words, ulong item,
    threadgroup uchar *window, uint tid, uint threads, uint simd, uint simdgroups, uint lane) {
    if constexpr (B::dense()) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        convert_row<B>(record, out, seismic_words, item, tid, threads);
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    } else {
        convert_window<B>(record, out, seismic_words, item, window, tid, threads, simd, simdgroups, lane);
    }
}

// Item `item` of a packed record: its window, by `repack_tile`, two tiles a
// simdgroup, lanes 0-15 one tile's rows and lanes 16-31 the next's. A
// tile's lanes exchange no words (`repack::exchange_words`): an exchange
// spans the threadgroup, which here holds many tiles; and without one a
// tile's lanes past its 16 rows place nothing, so half a simdgroup is a
// tile's every lane.
template <class B>
inline void convert_window(STAGED *record, device uchar *out, constant ulong *seismic_words, ulong item,
    threadgroup uchar *window, uint tid, uint threads, uint simd, uint simdgroups, uint lane) {
    static_assert(B::layout() != repack::Layout::mma16 || B::grids(),
                  "a streamed record's tiles exchange no words (mma16 code planes do)");
    static_assert(2 * repack::TILE_ROWS == 32, "a simdgroup's lanes are two tiles' rows");
    typedef Windows<B> W;
    const W windows = W::of(seismic_words);
    const uint row_tile = uint(item / windows.count);
    const uint window_index = uint(item % windows.count);
    const ulong dim_n = B::n(seismic_words);
    const ulong packets = (ulong(B::k(seismic_words)) + B::group() - 1) / B::group();
    // The window's stored bytes: rows of the tile below N, its packets below
    // the row's.
    const ulong row0 = ulong(row_tile) * W::rows();
    const uint rows = row0 < dim_n ? uint(min(ulong(W::rows()), dim_n - row0)) : 0u;
    const ulong packet0 = ulong(window_index) * windows.packets;
    const uint stored_packets = streaming_window_stored(uint(packet0), windows.packets, uint(packets));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (rows != 0 && stored_packets != 0) {
        const ulong offset =
            (row0 * B::source_stride_1(seismic_words) + packet0 * B::source_stride_2(seismic_words)) *
            W::packet_bytes();
        if (windows.whole)
            copy_staged(record, offset, window, (rows - 1) * windows.pitch + stored_packets * W::packet_bytes(), tid,
                threads);
        else
            copy_rows(record, offset, B::source_stride_1(seismic_words) * W::packet_bytes(), window, windows.pitch,
                rows, stored_packets * W::packet_bytes(), tid, threads);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const ulong tiles_x = repack::tiles_x<B>(seismic_words);
    const uint tiles = windows.packets * W::packet_tiles();
    const Window<B> source{window, windows.pitch, windows.packets};
    for (uint tile = 2 * simd + lane / repack::TILE_ROWS; tile < tiles; tile += 2 * simdgroups) {
        const ulong tile_x = ulong(window_index) * tiles + tile;
        if (tile_x < tiles_x)
            repack::repack_tile<B, Window<B>>(source, out, seismic_words, 0, uint2(uint(tile_x), row_tile),
                lane % repack::TILE_ROWS, static_cast<threadgroup uint *>(nullptr));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
}

} // namespace streaming
