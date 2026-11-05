// The router logits of the prefill classes (M > DECODE_ROWS), for `routed_route` and
// `routed_select`: one threadgroup of four simdgroups forms a 64 x 32 tile of
// logits[row, column] = sum over k of input[row, k] * column[k], on 8x8 F32
// simdgroup matrices. Every logit is one accumulator's chain of
// multiply-accumulates over the hidden axis in 8-wide steps, k ascending,
// from zero, and keeps its row's and column's places within the 8x8
// fragments (tiles start at multiples of 8): the tile shape never changes a
// bit, and experts with equal router rows tie exactly.
//
// Simdgroup s forms rows 16 s.. of the tile against its 32 columns,
// skipping fragments wholly past the last row or column. It reads its input
// fragments straight into its lanes' matrix elements (no other simdgroup
// reads those rows); the columns, which all four share, are decoded once per
// stage into threadgroup memory. Each stage's reads are issued during the
// stage before, so they are in flight under its multiplies.
//
// The includer defines KERNEL_W0 as the router.

#include "../projection/projection.h"

namespace logits {

typedef element::Act Act;

// Threads of a logits threadgroup, and the rows and logit columns of its
// tile. At Qwen3.6 35B's shape (H 2048, E 256, bf16 rows and router, M4
// Pro; the whole route, 3 launches), 64 x 32 tiles of 32-coordinate stages
// take 174, 175, 179, 184, 201 and 261 µs at 16 to 512 rows; this body's
// 32 x 32 tiles of 64-coordinate stages take 208, 228, 227, 228, 230 and
// 267, so one shape serves every row class. At 512 rows 32 x 64, 32 x 128
// and 64 x 64 tiles take up to 405 µs, and 128- and 256-row tiles of 8 and
// 16 simdgroups sharing a column block 277 and 311. The threadgroup memory
// a stage holds, and so the threadgroups a core keeps, sets it.
constant constexpr uint simdgroups = 4;
constant constexpr uint threads = 32 * simdgroups;
constant constexpr uint rows_per_tile = 16 * simdgroups;
constant constexpr uint columns_per_tile = 32;

// Hidden coordinates per stage: a lane holds 8 input fragments of a stage
// (2 row fragments of 4 steps), and the next stage's reads as many again.
constant constexpr uint depth = 32;

// The floats of a tile's threadgroup memory: the staged columns, then one
// 8 x 8 fragment per simdgroup for the stores.
constant constexpr uint extent = columns_per_tile * (depth + 4) + simdgroups * 64;

// The two elements lane `lane` holds of an 8 x 8 simdgroup matrix, as
// `thread_elements()` orders them: (row, column) and (row, column + 1).
inline uint2 element_of(uint lane) {
    const uint quad = lane / 4;
    return uint2((quad & 4) + (lane / 2) % 4, (quad & 2) * 2 + (lane % 2) * 2);
}

// The tile at rows row0.., columns column0.. of `rows` x `columns` logits
// over H hidden coordinates. `input(row, k)` is the router input of a row
// below `rows`; for a column c below `columns`, `column.fetch(c, k0)` loads
// the 32 values k0 .. k0 + 31 of logit column c (a `Column::fetched`) and
// `column.write(c, k0, fetched, out)` writes them, decoded, to `out`;
// `store(row, c, value)` publishes a logit.
template <uint H, typename Input, typename Column, typename Store>
inline void tile(ulong rows, ulong columns, ulong row0, ulong column0, Input input, thread const Column &column,
    Store store, threadgroup float *shared, uint tid, uint simd, uint lane) {
    constexpr uint D = depth;
    constexpr uint FR = rows_per_tile / (8 * simdgroups);
    constexpr uint steps = D / 8;
    constexpr uint packets = D / 32;
    // One packet per thread at most.
    constexpr uint column_packets = columns_per_tile * packets;
    static_assert(H % D == 0 && column_packets <= threads, "whole stages");
    // Columns stage k-contiguous; fragments load them transposed.
    constexpr uint pitch = D + 4;
    threadgroup float *columns_tile = shared;
    threadgroup float *published = shared + columns_per_tile * pitch + 64 * simd;

    const uint simd_row = simd * 8 * FR;
    // The fragments this simdgroup forms: those with a row below `rows` and
    // a column below `columns`.
    const uint live_rows =
        row0 + simd_row < rows ? uint(metal::min(ulong(FR), (rows - row0 - simd_row + 7) / 8)) : 0;
    const uint live_columns = uint(metal::min(4ul, (columns - column0 + 7) / 8));
    simdgroup_float8x8 accumulators[FR][4];
    PROJECTION_UNROLL
    for (uint i = 0; i < FR; ++i)
        PROJECTION_UNROLL
        for (uint j = 0; j < 4; ++j)
            accumulators[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    // This lane's input elements of a stage (fragment (i, step): rows
    // simd_row + 8 i.., coordinates k0 + 8 step..), and its column packet.
    const uint2 at = element_of(lane);
    const ulong own = column0 + tid / packets;
    const bool decodes = tid < column_packets && own < columns;
    float2 fetched_rows[FR][steps];
    typename Column::fetched fetched_column;
    const auto fetch = [&](ulong k0) {
        PROJECTION_UNROLL
        for (uint i = 0; i < FR; ++i) {
            const ulong row = row0 + simd_row + 8 * i + at.x;
            PROJECTION_UNROLL
            for (uint step = 0; step < steps; ++step) {
                const ulong k = k0 + 8 * step + at.y;
                fetched_rows[i][step] = row < rows ? float2(input(row, k), input(row, k + 1)) : float2(0.0f);
            }
        }
        if (decodes)
            fetched_column = column.fetch(own, k0 + 32 * (tid % packets));
    };
    fetch(0);
    for (ulong k0 = 0; k0 < H; k0 += D) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < column_packets) {
            threadgroup float *out = columns_tile + (tid / packets) * pitch + 32 * (tid % packets);
            if (decodes)
                column.write(own, k0 + 32 * (tid % packets), fetched_column, out);
            else
                for (uint i = 0; i < 32; ++i)
                    out[i] = 0.0f;
        }
        simdgroup_float8x8 a[FR][steps];
        PROJECTION_UNROLL
        for (uint i = 0; i < FR; ++i)
            PROJECTION_UNROLL
            for (uint step = 0; step < steps; ++step) {
                a[i][step].thread_elements()[0] = fetched_rows[i][step].x;
                a[i][step].thread_elements()[1] = fetched_rows[i][step].y;
            }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k0 + D < H)
            fetch(k0 + D);
        PROJECTION_UNROLL
        for (uint step = 0; step < steps; ++step) {
            PROJECTION_UNROLL
            for (uint j = 0; j < 4; ++j) {
                if (j >= live_columns)
                    continue;
                simdgroup_float8x8 b;
                simdgroup_load(b, columns_tile + 8 * j * pitch + 8 * step, pitch, ulong2(0, 0), true);
                PROJECTION_UNROLL
                for (uint i = 0; i < FR; ++i)
                    if (i < live_rows)
                        simdgroup_multiply_accumulate(accumulators[i][j], a[i][step], b, accumulators[i][j]);
            }
        }
    }
    PROJECTION_UNROLL
    for (uint i = 0; i < FR; ++i)
        PROJECTION_UNROLL
        for (uint j = 0; j < 4; ++j) {
            if (i >= live_rows || j >= live_columns)
                continue;
            simdgroup_store(accumulators[i][j], published, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint item = lane; item < 64; item += 32) {
                const ulong row = row0 + simd_row + 8 * i + item / 8;
                const ulong c = column0 + 8 * j + item % 8;
                if (row < rows && c < columns)
                    store(row, c, published[item]);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
}

// Logit columns that are router rows: column c is router row c, its packets
// decoded.
struct RouterColumns {
    typedef typename packets::W0::packet fetched;
    projection::Weights<packets::W0> router;
    fetched fetch(ulong expert, ulong k0) const {
        return router.packet(uint(expert), uint(k0 / 32));
    }
    void write(ulong, ulong, thread const fetched &packet, threadgroup float *out) const {
        PROJECTION_UNROLL
        for (uint step = 0; step < 4; ++step) {
            float4 even, odd;
            packets::W0::values(packet, step, even, odd);
            PROJECTION_UNROLL
            for (uint i = 0; i < 8; ++i)
                out[8 * step + i] = (i & 1u) ? odd[i >> 1] : even[i >> 1];
        }
    }
};

} // namespace logits
