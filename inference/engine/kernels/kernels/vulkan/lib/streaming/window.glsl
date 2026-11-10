// Expert streaming's conversion of staged records on Vulkan, by windows; the
// Vulkan form of `metal/lib/streaming/window.h`: one item is a row tile of a
// record (16 rows) and a window of its packets along those rows. The
// workgroup copies the item's stored bytes from staging into its window
// (group memory, SEISMIC_CONST_WINDOW_BYTES from the byte offset the
// includer gives) in the widest loads their address admits, then places the
// window's column tiles by `repack_weight`'s placement
// (lib/repack/repack.glsl, its `_shared` form, reading the copy), so a slot
// holds exactly the bytes a resident import places. The call follows its
// round's wait, which ended at a park the host opened after staging, so the
// staged bytes are read as plain loads.
//
// The window rule is `streaming/window.hpp`'s, asked over the source tensor's
// real row stride (a record's rows and a row's packets are contiguous, GGUF's
// [N, K] packets), and the item count's one owner (`NAME_items`), which the
// grid's workgroups stride over.
//
// Every invocation places: the workgroup's invocations in sixteens, each
// sixteen one column tile at a time, an invocation a row of it (a 32-lane
// subgroup two tiles at once). Placing reads no other invocation's words
// (no subgroup or group-memory exchange: a tile's row is one invocation's
// alone), so a sixteen is any 16 invocations, the halving resting on
// REPACK_TILE_ROWS 16 alone. A dense float record (a `repack` between dense
// representations) converts row by row, each element as `import_dense`
// converts it (decoded to F32, rounded into the destination).
//
// Per binding, `REPACK_TILE(NAME, C, S, D)` followed by
// `STREAMING_PACKED(NAME, S, SRC, D, N, K)` (conversion `C` from source
// element `S` into rows16) or `STREAMING_DENSE(NAME, S, U, SRC,
// D, N, K)` (dense `S` into dense `U`) defines, over source tensor `SRC`'s
// and destination `D`'s strides and the record's [N, K] matrix:
// - `NAME_items()`, a record's items: a packed record's row tiles by its
//   windows, a dense record's spans of elements;
// - `NAME_convert(record, destination, item, window)`, item `item` of the
//   source record at `record` into the destination record at `destination`,
//   by the whole workgroup, a packed record's through the window at byte
//   `window` (a multiple of 16) of the group memory: every invocation calls
//   it, and it begins and ends with a workgroup barrier.
// The includer includes lib/streaming/control.glsl first.
#include "../repack/repack.glsl"
#include "../../../streaming/window.hpp"

// `rows` spans of `span` bytes, from device address `from` (a row each
// `stride` bytes) to group-memory byte `to` (16-byte aligned; a row each
// `pitch` bytes), by the workgroup's invocations at once: in units of the
// widest load (to 16 bytes) every row's start admits, then each row's tail
// bytes.
void streaming_window_copy(uint64_t from, uint64_t stride, uint to, uint pitch, uint span, uint rows) {
    const uint tid = gl_LocalInvocationIndex, threads = gl_WorkGroupSize.x;
    const uint64_t starts = from | (rows > 1u ? stride | uint64_t(pitch) : 0ul);
    const uint unit = starts % 16ul == 0ul ? 16u
        : starts % 8ul == 0ul              ? 8u
        : starts % 4ul == 0ul              ? 4u
        : starts % 2ul == 0ul              ? 2u
                                           : 1u;
    const uint units = span / unit;
    for (uint i = tid; i < rows * units; i += threads) {
        const uint row = i / units, byte = i % units * unit;
        const uint64_t at = from + uint64_t(row) * stride + uint64_t(byte);
        const uint place = to + row * pitch + byte;
        if (unit == 16u) {
            seismic_shared_uvec4[place / 16u] = seismic_uvec4(at).v;
        } else if (unit == 8u) {
            const uvec2 pair = seismic_uvec2(at).v;
            seismic_shared_u32[place / 4u] = pair.x;
            seismic_shared_u32[place / 4u + 1u] = pair.y;
        } else if (unit == 4u) {
            seismic_shared_u32[place / 4u] = seismic_u32(at).v;
        } else if (unit == 2u) {
            seismic_shared_u16[place / 2u] = seismic_u16(at).v;
        } else {
            seismic_shared_u8[place] = seismic_u8(at).v;
        }
    }
    const uint tail = span - units * unit;
    for (uint i = tid; i < rows * tail; i += threads) {
        const uint row = i / tail, byte = units * unit + i % tail;
        seismic_shared_u8[to + row * pitch + byte] = seismic_u8(from + uint64_t(row) * stride + uint64_t(byte)).v;
    }
}

// A dense element `S`'s element kind (`element_at`, `element_put`).
#define STREAMING_DENSE_KIND_(R) ELEMENT_##R
#define STREAMING_DENSE_KIND_OF(R) STREAMING_DENSE_KIND_(R)
#define STREAMING_DENSE_KIND(S) STREAMING_DENSE_KIND_OF(S##_REPRESENTATION)

#define STREAMING_PACKED(NAME, S, SRC, D, N, K) \
    /* A stored row's packets (the window rule's row length), the source */ \
    /* row's own, and the bytes between the source's rows. */ \
    uint NAME##_row_packets() { \
        return streaming_window_row_packets(uint64_t(D##_ROW_GROUPS) * uint64_t(S##_LOGICAL_GROUP), \
            uint(S##_LOGICAL_GROUP)); \
    } \
    uint NAME##_source_packets() { \
        return uint((uint64_t(K) + uint64_t(S##_LOGICAL_GROUP) - 1ul) / uint64_t(S##_LOGICAL_GROUP)); \
    } \
    uint NAME##_stride() { \
        return uint(SRC##_STRIDE_1) * uint(S##_PACKET_SIZE); \
    } \
    /* The packets of a row a window holds, and a row's windows. */ \
    uint NAME##_packets() { \
        return streaming_window_packets(REPACK_TILE_ROWS, NAME##_row_packets(), uint(S##_PACKET_SIZE), \
            NAME##_stride(), uint(SEISMIC_CONST_WINDOW_BYTES)); \
    } \
    uint NAME##_windows() { \
        return streaming_window_count(NAME##_row_packets(), NAME##_packets()); \
    } \
    uint64_t NAME##_items() { \
        return (uint64_t(N) + REPACK_TILE_ROWS - 1u) / REPACK_TILE_ROWS * uint64_t(NAME##_windows()); \
    } \
    void NAME##_convert(uint64_t record, uint64_t destination, uint64_t item, uint window) { \
        barrier(); \
        const uint packet_bytes = uint(S##_PACKET_SIZE); \
        const uint windows = NAME##_windows(), packets = NAME##_packets(), stride = NAME##_stride(); \
        const uint64_t n0 = item / windows * REPACK_TILE_ROWS; \
        const uint rows = uint(min(uint64_t(REPACK_TILE_ROWS), uint64_t(N) - n0)); \
        /* The window's packets of each row, and where they lie in the */ \
        /* window: its rows `pitch` apart. */ \
        const bool whole = streaming_window_whole(REPACK_TILE_ROWS, stride, uint(SEISMIC_CONST_WINDOW_BYTES)); \
        const uint first = uint(item % windows) * packets; \
        const uint stored = streaming_window_stored(first, packets, NAME##_source_packets()); \
        const uint pitch = streaming_window_pitch(REPACK_TILE_ROWS, packet_bytes, packets, stride, \
            uint(SEISMIC_CONST_WINDOW_BYTES)); \
        const uint64_t from = \
            record + (n0 * SRC##_STRIDE_1 + uint64_t(first) * SRC##_STRIDE_2) * uint64_t(packet_bytes); \
        /* A whole tile is one span from its first row to its last row's */ \
        /* last packet, rows at the source's stride, so at the window's. */ \
        if (whole) \
            streaming_window_copy(from, 0ul, window, 0u, (rows - 1u) * stride + stored * packet_bytes, 1u); \
        else \
            streaming_window_copy(from, uint64_t(stride), window, pitch, stored * packet_bytes, rows); \
        memoryBarrierShared(); \
        barrier(); \
        const uint row = gl_LocalInvocationIndex % REPACK_TILE_ROWS; \
        const uint tiles = stored * uint(S##_LOGICAL_GROUP) / NAME##_tile_columns(); \
        if (row < rows) { \
            const uint64_t placed = destination + (n0 + uint64_t(row)) * D##_STRIDE_1 * D##_ROW_STRIDE_BYTES; \
            for (uint tile = gl_LocalInvocationIndex / REPACK_TILE_ROWS; tile < tiles; \
                 tile += gl_WorkGroupSize.x / REPACK_TILE_ROWS) { \
                const uint column = tile * NAME##_tile_columns(); \
                NAME##_place_shared(window + row * pitch + column / uint(S##_LOGICAL_GROUP) * packet_bytes, placed, \
                    first * uint(S##_LOGICAL_GROUP) + column); \
            } \
        } \
        memoryBarrierBuffer(); \
        barrier(); \
    }

#define STREAMING_DENSE(NAME, S, U, SRC, D, N, K) \
    /* The source's element bytes, and a span of a record's flattened elements */ \
    /* an item holds (`streaming_dense_span`): each item is staged through the */ \
    /* window and converted from it. */ \
    uint NAME##_element() { \
        return ELEMENT_BYTES(STREAMING_DENSE_KIND(S)); \
    } \
    uint NAME##_span() { \
        return streaming_dense_span(NAME##_element(), gl_WorkGroupSize.x, uint(SEISMIC_CONST_WINDOW_BYTES)); \
    } \
    uint64_t NAME##_items() { \
        return uint64_t(streaming_dense_items(uint(N) * uint(K), NAME##_element(), NAME##_span())); \
    } \
    void NAME##_convert(uint64_t record, uint64_t destination, uint64_t item, uint window) { \
        /* Vulkan's running work sees no host write (its parks are submission */ \
        /* boundaries), so the span is read in place: no window. */ \
        barrier(); \
        const uint in_span = NAME##_span() / NAME##_element(); \
        const uint first = uint(item) * in_span; \
        const uint count = min(in_span, uint(N) * uint(K) - first); \
        /* A thread's row and column step with its element: two divides an item. */ \
        uint row = (first + gl_LocalInvocationIndex) / uint(K), column = (first + gl_LocalInvocationIndex) % uint(K); \
        const uint row_step = gl_WorkGroupSize.x / uint(K), column_step = gl_WorkGroupSize.x % uint(K); \
        for (uint e = gl_LocalInvocationIndex; e < count; e += gl_WorkGroupSize.x) { \
            const uint64_t n = uint64_t(row), k = uint64_t(column); \
            element_put(STREAMING_DENSE_KIND(U), destination, n * D##_STRIDE_1 + k * D##_STRIDE_2, \
                element_at(STREAMING_DENSE_KIND(S), record, n * SRC##_STRIDE_1 + k * SRC##_STRIDE_2)); \
            row += row_step; \
            column += column_step; \
            if (column >= uint(K)) { \
                column -= uint(K); \
                ++row; \
            } \
        } \
        memoryBarrierBuffer(); \
        barrier(); \
    }
