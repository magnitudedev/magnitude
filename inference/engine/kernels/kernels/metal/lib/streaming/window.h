// Expert streaming's conversion of staged records, by windows: one item is a row tile of a
// record (16 rows) and a window of its packets along those rows. A
// threadgroup copies the item's stored bytes from staging into threadgroup
// memory with the widest aligned system-coherent loads (the host writes
// staging within the step's command buffer; GGUF rows are contiguous), then
// its simdgroups place the window's column tiles by `repack_tile`
// (lib/repack/repack.h, the registered conversion), reading the copy. The
// tile placement is `repack_weight`'s, so a slot holds exactly the bytes a
// resident import places. A dense float record (`B::dense()`: `repack`
// between dense representations) converts row by row, each element as
// `import_dense` converts it (decoded to F32, rounded into the destination).
//
// Templates over a repack binding `B` (`REPACK_BINDING`), so a kernel
// converts through as many bindings as it holds. The includer includes
// `control.h` first.

#include <seismic/slab.h>
#include "../repack/repack.h"

#define STAGED volatile coherent(system) device const uchar

namespace streaming {

// A window: 16 rows of `packets()` source packets, each row's share padded
// to 16 bytes, within the module's WINDOW_BYTES (`SEISMIC_CONST_WINDOW_BYTES`,
// the includer's threadgroup memory). A tile reads only its own rows' unit
// of one packet, so a window is any whole number of packets.

template <class B> struct Window {
    static constexpr uint rows() {
        return repack::TILE_ROWS;
    }
    static constexpr uint packet_bytes() {
        return uint(B::source_packet_size());
    }
    static constexpr uint group() {
        return uint(B::group());
    }
    static constexpr uint packets() {
        return SEISMIC_CONST_WINDOW_BYTES / (rows() * packet_bytes());
    }
    static constexpr uint row_bytes() {
        return (packets() * packet_bytes() + 15u) / 16u * 16u;
    }
    static_assert(packets() > 0, "a window holds a packet of each row");
    static_assert(rows() * row_bytes() <= SEISMIC_CONST_WINDOW_BYTES, "a window fits its threadgroup memory");
    // Column tiles of a window.
    static constexpr uint tiles() {
        return packets() * group() / repack::tile_columns<B>();
    }
    // Packet `packet` of row `n` in the window's copy.
    template <typename In> static In packet(In source, constant ulong *, ulong, ulong n, ulong packet) {
        return source + (n % rows()) * row_bytes() + (packet % packets()) * packet_bytes();
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
        // Odd-sized packets (17-byte MXFP4) start rows at odd offsets.
        for (uint i = tid; i < bytes; i += threads)
            to[i] = from[i];
    }
}

// Windows along a row, and items of one record.
template <class B> inline ulong record_windows(constant ulong *seismic_words) {
    return (repack::tiles_x<B>(seismic_words) + Window<B>::tiles() - 1) / Window<B>::tiles();
}

// A packed record's items are its windows; a dense record's, its rows.
template <class B> inline ulong record_items(constant ulong *seismic_words) {
    if constexpr (B::dense())
        return B::n(seismic_words);
    else
        return repack::tiles_y<B>(seismic_words) * record_windows<B>(seismic_words);
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

// Item `item` of a packed record: its window, by `repack_tile`, a simdgroup
// a tile. A tile's lanes exchange no words (`repack::exchange_words`): an
// exchange spans the threadgroup, which here holds many tiles.
template <class B>
inline void convert_window(STAGED *record, device uchar *out, constant ulong *seismic_words, ulong item,
    threadgroup uchar *window, uint tid, uint threads, uint simd, uint simdgroups, uint lane) {
    static_assert(B::layout() != repack::Layout::mma16 || B::grids(),
                  "a streamed record's tiles exchange no words (mma16 code planes do)");
    typedef Window<B> W;
    const ulong windows = record_windows<B>(seismic_words);
    const uint row_tile = uint(item / windows);
    const uint window_index = uint(item % windows);
    const ulong dim_n = B::n(seismic_words);
    const ulong packets = (ulong(B::k(seismic_words)) + W::group() - 1) / W::group();
    // The window's stored bytes: rows of the tile below N, its packets below
    // the row's.
    const ulong packet0 = ulong(window_index) * W::packets();
    const uint stored_packets = packet0 < packets ? uint(min(ulong(W::packets()), packets - packet0)) : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint row = 0; row < W::rows(); ++row) {
        const ulong n = ulong(row_tile) * W::rows() + row;
        if (n >= dim_n || stored_packets == 0)
            break;
        copy_staged(record,
            (n * B::source_stride_1(seismic_words) + packet0 * B::source_stride_2(seismic_words)) *
                W::packet_bytes(),
            window + row * W::row_bytes(), stored_packets * W::packet_bytes(), tid, threads);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const ulong tiles_x = repack::tiles_x<B>(seismic_words);
    for (uint tile = simd; tile < W::tiles(); tile += simdgroups) {
        const uint tile_x = window_index * W::tiles() + tile;
        if (tile_x < tiles_x)
            repack::repack_tile<B, W>(static_cast<threadgroup const uchar *>(window), out, seismic_words, 0,
                uint2(tile_x, row_tile), lane, static_cast<threadgroup uint *>(nullptr));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
}

} // namespace streaming
