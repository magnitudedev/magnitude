// Expert streaming's window rule, written once, in the language MSL, CUDA C++
// and GLSL have in common (streaming/contract.hpp's shim: `STREAMING_FN`,
// `u32`): how a packed record's row tiles divide into the windows a
// threadgroup copies staged bytes into before it places them. Metal's window
// library (`metal/lib/streaming/window.h`), CUDA's
// (`cuda/lib/streaming/window.cuh`) and Vulkan's
// (`vulkan/lib/streaming/window.glsl`) ask it and nothing else.
//
// A window holds `packets` packets of each of a tile's `rows` rows (packet p
// of a row lies in window p / packets), the rows' shares `pitch` bytes apart,
// within `window` bytes. A tile reads only its own rows' unit of one packet,
// so a window is any whole number of packets. A row is `row_packets` packets
// (`streaming_window_row_packets`) of `packet` bytes, stored `stride` bytes
// apart (a row's packets are contiguous, GGUF's [N, K] packets; `stride` is at
// least the row's bytes).
// Of the shapes a record takes, the first that holds:
// - whole: the tile's rows fit (`rows * stride <= window`), so a window is the
//   whole tile, copied as one span from its first row to its last row's last
//   packet, `pitch` the stride: one 16-byte aligned span where the record is,
//   whatever a row's own alignment (V4.1's MXFP4 down rows, 72 17-byte
//   packets, 1,224 bytes, 8 off 16: a 19,584-byte tile);
// - aligned: a stride of whole 16-byte units, so a window's packets are too
//   where they are a multiple of a grain (the fewest packets of whole 16-byte
//   units: 16 over the largest power of two, to 16, a packet's bytes are a
//   multiple of): every row's share is whole 16-byte loads. A window holds
//   the fewest grains that cover the row in as few windows as the most grains
//   the tile's rows fit allow, so the windows are even but for the last's
//   stored packets (V4.1's MXFP4 gate/up rows, 160 packets, 2,720 bytes, a
//   grain of 16 packets: three windows of 64, 64 and 32 packets in 20,480
//   bytes);
// - any other: the most packets the tile's rows fit, each row's share padded
//   to 16 bytes.
// A window past a row's stored packets holds none.
//
// An includer asserts that a window holds a packet of each row
// (`STREAMING_WINDOW_FITS`).
//
// A dense record has no windows: its items are runs of whole rows, each a
// pass of the launch's threads over the run's elements (`streaming_dense_*`).

// Whether `rows` rows of `packet`-byte packets, each share padded to 16 bytes,
// fit a window of `window` bytes with at least one packet each: an integer
// constant expression, which each backend asserts of its records' packets.
#define STREAMING_WINDOW_FITS(rows, packet, window)                                                         \
    ((window) / ((rows) * (packet)) > 0 &&                                                                   \
     (rows) * (((window) / ((rows) * (packet)) * (packet) + 15) / 16 * 16) <= (window))

// The packets of a row the windows cover, whatever the backend: its stored
// columns (the destination's: its tiles' columns) in the source's groups,
// which may exceed the source row's own packets by padding columns. The
// rule's `row_packets` is this and nothing else.
STREAMING_FN u32 streaming_window_row_packets(u64 stored_columns, u32 group) {
    return u32(stored_columns / group);
}

// The packets of a window that begins at packet `first` the source row of
// `source_packets` packets holds: none of a window past them, which copies
// nothing.
STREAMING_FN u32 streaming_window_stored(u32 first, u32 packets, u32 source_packets) {
    return first >= source_packets ? 0u : (source_packets - first < packets ? source_packets - first : packets);
}

// The fewest packets whose bytes are whole 16-byte units.
STREAMING_FN u32 streaming_window_grain(u32 packet) {
    const u32 lowest = packet & (~packet + 1u);
    return 16u / (lowest < 16u ? lowest : 16u);
}

// Whether a row tile is one window.
STREAMING_FN bool streaming_window_whole(u32 rows, u32 stride, u32 window) {
    return rows * stride <= window;
}

// The packets of a row a window holds.
STREAMING_FN u32 streaming_window_packets(u32 rows, u32 row_packets, u32 packet, u32 stride, u32 window) {
    if (streaming_window_whole(rows, stride, window))
        return row_packets;
    const u32 grain = streaming_window_grain(packet);
    const u32 grains = window / (rows * packet * grain);
    if (stride % 16u == 0u && grains > 0u) {
        const u32 row_grains = (row_packets + grain - 1u) / grain;
        const u32 count = (row_grains + grains - 1u) / grains;
        return (row_grains + count - 1u) / count * grain;
    }
    return window / (rows * packet);
}

// The windows along a row.
STREAMING_FN u32 streaming_window_count(u32 row_packets, u32 packets) {
    return (row_packets + packets - 1u) / packets;
}

// The bytes between a window's rows.
STREAMING_FN u32 streaming_window_pitch(u32 rows, u32 packet, u32 packets, u32 stride, u32 window) {
    return streaming_window_whole(rows, stride, window) ? stride : (packets * packet + 15u) / 16u * 16u;
}

// The elements each thread of a dense item converts: an item is a pass of the
// launch's `threads` over `threads * STREAMING_DENSE_PER_THREAD` elements, so
// that the claim, barriers and count an item costs are paid over work and not
// over a row a few elements long.
#define STREAMING_DENSE_PER_THREAD 8u

// The bytes of a dense record's item: a pass of the launch's threads, within
// the window, whole 16-byte units (an element, a power of two of at most four
// bytes, never straddles two items). A record's items split its flattened
// elements, rows and all; the dense source is row-major, one unit-stride run.
STREAMING_FN u32 streaming_dense_span(u32 element, u32 threads, u32 window) {
    const u32 bytes = threads * STREAMING_DENSE_PER_THREAD * element;
    return (bytes < window ? bytes : window) / 16u * 16u;
}

// A dense record's items: its `elements` elements of `element` bytes in
// spans of `span` bytes, the last the rest.
STREAMING_FN u32 streaming_dense_items(u32 elements, u32 element, u32 span) {
    return (elements * element + span - 1u) / span;
}
