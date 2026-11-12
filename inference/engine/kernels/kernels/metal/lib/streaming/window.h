// Expert streaming's conversion of staged records, by windows: one item is a row tile of a
// record (16 rows) and a window of its packets along those rows (`Windows`).
// A threadgroup copies the item's stored bytes from staging into threadgroup
// memory with the widest aligned system-coherent loads (the host writes
// staging within the step's command buffer; GGUF rows are contiguous), then
// its simdgroups place the window's column tiles by `repack_tile`
// (lib/repack/repack.h, the registered conversion), two tiles a simdgroup,
// reading the copy. The tile placement is `repack_weight`'s, so a slot holds
// exactly the bytes a resident import places. A dense float record
// (`B::dense()`: `repack` between dense representations) converts runs of
// rows, each element as `import_dense` converts it (decoded to F32, rounded
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
// where the window is the whole tile, copied as one span.
// A planar source (`B::planar()`: a record is each matrix's codes plane, then
// its grid of scales) is windowed over its packets' codes alone
// (`packet_bytes`), and beside them, after them in the threadgroup memory,
// holds the scales of the tiles its packets lie in (`scale_area`: a row's own
// where a grid tile is one row, else one row of the grid's), whose bytes come
// off the codes' (`bytes`): the same shapes for every source, a plain
// source's scale area empty.
// A row's packets are its stored columns' (`repack::tiles_x`): a window past
// the source row's packets copies none and places zero tiles.
template <class B> struct Windows {
    uint packets;
    uint count;
    uint pitch;
    bool whole;

    static constexpr uint rows() {
        return repack::TILE_ROWS;
    }
    // The threadgroup memory's bytes the codes have: its scale area is the rest.
    static constexpr uint bytes() {
        return uint(SEISMIC_CONST_WINDOW_BYTES) - scale_area();
    }
    static constexpr uint packet_bytes() {
        return B::planar() ? uint(B::code_bytes()) : uint(B::source_packet_size());
    }
    // A planar source's scales area: `scale_rows` rows of the most scales a
    // row of a window's packets lies in (its first packet need not be a
    // tile's first).
    static constexpr uint scale_rows() {
        return !B::planar() ? 0u : B::grid_tile_rows() == 1u ? repack::TILE_ROWS : 1u;
    }
    static constexpr uint scales_per_row() {
        constexpr uint most = uint(SEISMIC_CONST_WINDOW_BYTES) / (repack::TILE_ROWS * packet_bytes());
        return !B::planar() ? 0u : (most * B::group() + B::grid_tile_columns() - 1u) / B::grid_tile_columns() + 1u;
    }
    static constexpr uint scale_pitch() {
        return scales_per_row() * B::grid_scale_bytes();
    }
    static constexpr uint scale_area() {
        return (scale_rows() * scale_pitch() + 15u) / 16u * 16u;
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
// source): the copy, its rows' pitch and a window's packets, and a planar
// source's scales (`Windows::bytes` on) from the scale of the window's first
// packet's tile.
template <class B> struct Window {
    threadgroup const uchar *copy;
    uint pitch;
    uint packets;
    threadgroup const uchar *scales;
    ulong first_packet;

    // Packet `packet` of row `n` in the window's copy.
    static threadgroup const uchar *packet(Window window, constant ulong *, ulong, ulong n, ulong packet) {
        return window.copy + (n % repack::TILE_ROWS) * window.pitch +
               (packet % window.packets) * Windows<B>::packet_bytes();
    }
    // The scale of the tile holding packet `index` of row `n` (a planar
    // source's; any other's is its packet, unread).
    static threadgroup const uchar *scale(Window window, constant ulong *, ulong, ulong n, ulong index) {
        if constexpr (B::planar()) {
            const ulong first = window.first_packet * B::group() / B::grid_tile_columns();
            return window.scales +
                   (B::grid_tile_rows() == 1u ? (n % repack::TILE_ROWS) * Windows<B>::scale_pitch() : 0u) +
                   ((index >= window.first_packet ? index * B::group() / B::grid_tile_columns() - first : 0ul)) *
                       B::grid_scale_bytes();
        } else {
            return Window::packet(window, nullptr, 0, n, index);
        }
    }
};

// Copy the `bytes` staged bytes at byte `offset` of `source` to `to`, which
// lies `to_offset` bytes past a 16-byte aligned address, with the
// threadgroup's `threads` threads, in the widest loads both addresses admit.
inline void copy_staged(STAGED *source, ulong offset, threadgroup uchar *to, uint bytes, uint tid, uint threads,
    uint to_offset = 0) {
    STAGED *from = source + offset;
    const ulong address = reinterpret_cast<ulong>(from) | to_offset;
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
// aligned, as the copy's rows are placed `pitch` apart), the next `pitch`
// bytes on: where every share is whole 16-byte
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
            copy_staged(source, offset + row * stride, to + row * pitch, bytes, tid, threads, row * pitch);
    }
}

template <class B> struct Dense {
    typedef typename element::Of<B::source_kind()>::type Source;
    typedef typename element::Of<B::destination_kind()>::type Destination;
};

// Record `record` of a source record port (its records' address table).
inline STAGED *source_record(device const ulong *table, ulong record) {
    return reinterpret_cast<STAGED *>(slab::region<const uchar>(table, record));
}

// A packed record's items are its row tiles' windows; a dense record's, spans
// of its flattened elements (`streaming_dense_span`), each staged through the
// window as a packed record's windows are, by the threadgroup's coherent
// loads, and converted from it.
template <class B> inline ulong record_items(constant ulong *seismic_words, uint threads) {
    if constexpr (B::dense()) {
        const uint element = uint(sizeof(typename Dense<B>::Source::storage));
        return streaming_dense_items(uint(B::n(seismic_words) * B::k(seismic_words)), element,
            streaming_dense_span(element, threads, uint(SEISMIC_CONST_WINDOW_BYTES)));
    } else {
        return repack::tiles_y<B>(seismic_words) * Windows<B>::of(seismic_words).count;
    }
}

// Item `item` of the dense source record at `record` into the destination
// record at `out`: its span staged into `window`, each element then decoded
// to F32 and rounded into the destination, `import_dense`'s conversion, from
// the window.
template <class B>
inline void convert_span(STAGED *record, device uchar *out, constant ulong *seismic_words, ulong item,
    threadgroup uchar *window, uint tid, uint threads) {
    typedef typename Dense<B>::Source Source;
    typedef typename Dense<B>::Destination Destination;
    const uint element = uint(sizeof(typename Source::storage));
    const uint k = uint(B::k(seismic_words)), elements = uint(B::n(seismic_words)) * k;
    const uint span = streaming_dense_span(element, threads, uint(SEISMIC_CONST_WINDOW_BYTES));
    const uint first = uint(item) * (span / element);
    const uint count = min(span / element, elements - first);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    copy_staged(record, ulong(first) * element, window, count * element, tid, threads);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // A thread's row and column step with its element: two divides an item.
    uint row = (first + tid) / k, column = (first + tid) % k;
    const uint row_step = threads / k, column_step = threads % k;
    for (uint e = tid; e < count; e += threads) {
        element::put<Destination>(out, ulong(row) * B::out_stride_1(seismic_words) + ulong(column) * B::out_stride_2(seismic_words),
            Source::load(reinterpret_cast<threadgroup const typename Source::storage *>(window)[e]));
        row += row_step;
        column += column_step;
        if (column >= k) {
            column -= k;
            ++row;
        }
    }
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
        convert_span<B>(record, out, seismic_words, item, window, tid, threads);
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
    if constexpr (B::planar()) {
        // The scales of the tiles the window's packets lie in, from the
        // record's grid, which follows its codes plane.
        if (rows != 0 && stored_packets != 0) {
            const ulong columns = repack::planar::grid_columns<B>(seismic_words);
            const ulong scale0 = packet0 * B::group() / B::grid_tile_columns();
            const uint count = uint((packet0 + stored_packets - 1) * B::group() / B::grid_tile_columns() - scale0 + 1);
            copy_rows(record,
                repack::planar::codes_bytes<B>(seismic_words) +
                    ((row0 / B::grid_tile_rows()) * columns + scale0) * B::grid_scale_bytes(),
                columns * B::grid_scale_bytes(), window + W::bytes(), W::scale_pitch(),
                B::grid_tile_rows() == 1u ? rows : 1u, count * B::grid_scale_bytes(), tid, threads);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const ulong tiles_x = repack::tiles_x<B>(seismic_words);
    const uint tiles = windows.packets * W::packet_tiles();
    const Window<B> source{window, windows.pitch, windows.packets, window + W::bytes(), packet0};
    for (uint tile = 2 * simd + lane / repack::TILE_ROWS; tile < tiles; tile += 2 * simdgroups) {
        const ulong tile_x = ulong(window_index) * tiles + tile;
        if (tile_x < tiles_x)
            repack::repack_tile<B, Window<B>>(source, out, seismic_words, 0, uint2(uint(tile_x), row_tile),
                lane % repack::TILE_ROWS, static_cast<threadgroup uint *>(nullptr));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
}

} // namespace streaming
