// Expert streaming's conversion of staged records on Vulkan; the Vulkan form
// of `metal/lib/streaming/window.h`: one item is a row tile of a record (16
// rows) and a window of its column tiles along those rows, one column tile
// to each 32 invocations of the workgroup. Each tile is placed by
// `repack_weight`'s placement (lib/repack/repack.glsl), so a slot holds exactly
// the bytes a resident import places; its 16 invocations of a tile each
// place one row from the row's source packet, read where the host staged it
// (the call follows its round's wait, which spun with a device-scope
// acquire until the host acknowledged the staging, and the word form reads
// packets by address). A dense
// float record (a `repack` between dense representations) converts row by
// row, each element as `import_dense` converts it (decoded to F32, rounded
// into the destination).
//
// Per binding, `REPACK_TILE(NAME, C, S, D)` followed by
// `STREAMING_PACKED(NAME, S, SRC, D, N, K)` (conversion `C` from source
// element `S` into rows16) or `STREAMING_DENSE(NAME, S, U, SRC,
// D, N, K)` (dense `S` into dense `U`) defines, over source tensor `SRC`'s
// and destination `D`'s strides and the record's [N, K] matrix:
// - `NAME_items()`, a record's items: a packed record's row tiles by its
//   windows, a dense record's rows;
// - `NAME_convert(record, destination, item)`, item `item` of the source record at
//   `record` into the destination record at `destination`, by the whole workgroup:
//   every invocation calls it, and it begins and ends with a workgroup
//   barrier.
// The includer includes lib/streaming/control.glsl first.
#include "../repack/repack.glsl"

// Column tiles of a window: one to each 32 invocations.
#define STREAMING_WINDOW_TILES (gl_WorkGroupSize.x / 32u)

// A dense element `S`'s element kind (`element_at`, `element_put`).
#define STREAMING_DENSE_KIND_(R) ELEMENT_##R
#define STREAMING_DENSE_KIND_OF(R) STREAMING_DENSE_KIND_(R)
#define STREAMING_DENSE_KIND(S) STREAMING_DENSE_KIND_OF(S##_REPRESENTATION)

#define STREAMING_PACKED(NAME, S, SRC, D, N, K) \
    uint64_t NAME##_tiles_x() { \
        return uint64_t(D##_ROW_GROUPS) * uint64_t(S##_LOGICAL_GROUP) / NAME##_tile_columns(); \
    } \
    uint64_t NAME##_windows() { \
        return (NAME##_tiles_x() + STREAMING_WINDOW_TILES - 1u) / STREAMING_WINDOW_TILES; \
    } \
    uint64_t NAME##_items() { \
        return (uint64_t(N) + REPACK_TILE_ROWS - 1u) / REPACK_TILE_ROWS * NAME##_windows(); \
    } \
    void NAME##_convert(uint64_t record, uint64_t destination, uint64_t item) { \
        barrier(); \
        const uint64_t windows = NAME##_windows(); \
        const uint lane = gl_LocalInvocationIndex % 32u; \
        const uint tile_x = uint(item % windows) * STREAMING_WINDOW_TILES + gl_LocalInvocationIndex / 32u; \
        const uint64_t n = item / windows * REPACK_TILE_ROWS + lane; \
        if (lane < REPACK_TILE_ROWS && n < uint64_t(N) && uint64_t(tile_x) < NAME##_tiles_x()) { \
            const uint column0 = tile_x * NAME##_tile_columns(); \
            const uint64_t packet = uint64_t(column0 / uint(S##_LOGICAL_GROUP)); \
            NAME##_place(record + (n * SRC##_STRIDE_1 + packet * SRC##_STRIDE_2) \
                    * uint64_t(S##_PACKET_SIZE), \
                destination + n * D##_STRIDE_1 * D##_ROW_STRIDE_BYTES, column0); \
        } \
        memoryBarrierBuffer(); \
        barrier(); \
    }

#define STREAMING_DENSE(NAME, S, U, SRC, D, N, K) \
    uint64_t NAME##_items() { \
        return uint64_t(N); \
    } \
    void NAME##_convert(uint64_t record, uint64_t destination, uint64_t n) { \
        barrier(); \
        for (uint64_t k = gl_LocalInvocationIndex; k < uint64_t(K); k += gl_WorkGroupSize.x) \
            element_put(STREAMING_DENSE_KIND(U), destination, n * D##_STRIDE_1 + k * D##_STRIDE_2, \
                element_at(STREAMING_DENSE_KIND(S), record, n * SRC##_STRIDE_1 + k * SRC##_STRIDE_2)); \
        memoryBarrierBuffer(); \
        barrier(); \
    }
