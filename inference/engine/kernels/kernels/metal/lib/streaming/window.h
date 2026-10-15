// Expert streaming's conversion of staged records, by windows
// (specs/26-10-11/expert-streaming.md 4.3, 4.6): one item is a row tile of a
// record (16 rows) and a window of its packets along those rows. A
// threadgroup copies the item's stored bytes from staging into threadgroup
// memory with the widest aligned system-coherent loads (staging is
// host-written within the step's command buffer; GGUF rows are contiguous),
// then its simdgroups convert the window's column tiles by `repack_tile`
// (lib/repack/repack.h), reading the copy. The tile conversion is
// `repack_weight`'s, so a slot holds exactly the bytes a resident import
// places.
//
// Templates over a repack binding `B` (lib/streaming/bind_*.h), so a kernel
// converts through as many bindings as it holds.

#pragma METAL internals : enable

#include "../repack/repack.h"

namespace streaming {

#define STAGED volatile coherent(system) device const uchar

// A window: 16 rows of `packets()` source packets, each row's share padded
// to 16 bytes, within 16 KiB. A window holds whole 64-column blocks (the
// mma16 fragments read a block's two 32-column packets).
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
    static constexpr uint block() {
        return group() >= 64 ? 1u : 64u / group();
    }
    static constexpr uint fit() {
        return 16384u / (rows() * packet_bytes());
    }
    static constexpr uint packets() {
        return fit() >= block() ? fit() - fit() % block() : block();
    }
    static constexpr uint row_bytes() {
        return (packets() * packet_bytes() + 15u) / 16u * 16u;
    }
    // Threadgroup bytes a window takes.
    static constexpr uint bytes() {
        return rows() * row_bytes();
    }
    // Column tiles of a window.
    static constexpr uint tiles() {
        return packets() * group() / repack::TILE_COLUMNS;
    }
    // Packet `packet` of row `n` in the window's copy.
    template <typename In> static In packet(In source, constant ulong *, ulong, ulong n, ulong packet) {
        return source + (n % rows()) * row_bytes() + (packet % packets()) * packet_bytes();
    }
};

// Copy the `bytes` staged bytes at byte `offset` of the staging binding (a
// pool tensor, allocation-aligned) to `to` (16-byte aligned) with the
// threadgroup's `threads` threads, in the widest loads the offset's
// alignment admits.
inline void copy_staged(STAGED *source, ulong offset, threadgroup uchar *to, uint bytes, uint tid,
    uint threads) {
    STAGED *from = source + offset;
    if (offset % 16 == 0) {
        for (uint i = tid; i < bytes / 16; i += threads)
            reinterpret_cast<threadgroup uint4 *>(to)[i] =
                reinterpret_cast<volatile coherent(system) device const uint4 *>(from)[i];
        for (uint i = bytes / 16 * 16 + tid; i < bytes; i += threads)
            to[i] = from[i];
    } else if (offset % 4 == 0) {
        for (uint i = tid; i < bytes / 4; i += threads)
            reinterpret_cast<threadgroup uint *>(to)[i] =
                reinterpret_cast<volatile coherent(system) device const uint *>(from)[i];
        for (uint i = bytes / 4 * 4 + tid; i < bytes; i += threads)
            to[i] = from[i];
    } else if (offset % 2 == 0) {
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

// Windows along a row and items of one record.
template <class B> inline ulong record_windows(constant ulong *seismic_words) {
    return (repack::Format<B>::tiles_x(seismic_words) + Window<B>::tiles() - 1) / Window<B>::tiles();
}

template <class B> inline ulong record_items(constant ulong *seismic_words) {
    return repack::Format<B>::tiles_y(seismic_words) * record_windows<B>(seismic_words);
}

// Row tiles of a stored matrix, padding tiles of a 32-row layout included.
template <class B> inline uint record_row_tiles(constant ulong *seismic_words) {
    return uint(repack::Format<B>::tiles_y(seismic_words));
}

// The destination matrix's byte stride: a packet layout's matrix stride
// counts packets, a row layout's counts stored rows.
template <class B> inline ulong matrix_bytes(constant ulong *seismic_words) {
    if constexpr (B::layout() == repack::Layout::packet)
        return B::out_stride_0(seismic_words) * B::packet_size();
    else
        return B::out_stride_0(seismic_words) * B::out_row_stride_bytes(seismic_words);
}

// Item `item` (< record_items()) of staging record `record` of `source` into
// slot `slot` of `destination`, by the whole threadgroup (`threads` threads
// in `simdgroups` simdgroups). Every thread calls it; it begins and ends
// with a threadgroup barrier, so `window` is free to reuse after it.
template <class B>
inline void convert_item(STAGED *source, device uchar *destination, constant ulong *seismic_words, ulong record,
    ulong slot, ulong item, threadgroup uchar *window, uint tid, uint threads, uint simd, uint simdgroups,
    uint lane) {
    typedef Window<B> W;
    const ulong windows = record_windows<B>(seismic_words);
    const uint row_tile = uint(item / windows);
    const uint window_index = uint(item % windows);
    const ulong dim_n = B::n(seismic_words);
    const ulong packets = (ulong(B::k(seismic_words)) + W::group() - 1) / W::group();
    const ulong staged = record * B::source_stride_0(seismic_words) * W::packet_bytes();
    device uchar *out = destination + slot * matrix_bytes<B>(seismic_words);
    // The window's stored bytes: rows of the tile below N, its packets
    // below the row's.
    const ulong packet0 = ulong(window_index) * W::packets();
    const uint stored_packets = packet0 < packets ? uint(min(ulong(W::packets()), packets - packet0)) : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint row = 0; row < W::rows(); ++row) {
        const ulong n = ulong(row_tile) * W::rows() + row;
        if (n >= dim_n || stored_packets == 0)
            break;
        copy_staged(source,
            staged + (n * B::source_stride_1(seismic_words) + packet0 * B::source_stride_2(seismic_words)) *
                W::packet_bytes(),
            window + row * W::row_bytes(), stored_packets * W::packet_bytes(), tid, threads);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const ulong tiles_x = repack::Format<B>::tiles_x(seismic_words);
    for (uint tile = simd; tile < W::tiles(); tile += simdgroups) {
        const uint tile_x = window_index * W::tiles() + tile;
        if (tile_x < tiles_x)
            repack::repack_tile<B, W>(static_cast<threadgroup const uchar *>(window), out, seismic_words, 0,
                uint2(tile_x, row_tile), lane);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
}

} // namespace streaming
