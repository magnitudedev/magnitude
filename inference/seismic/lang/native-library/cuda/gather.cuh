// Seismic native library (CUDA): row gathers of a table in any layout
// (`mma16` lane chunks, a streamed table's `rows16` rows, or dense). Both
// packed layouts give every value the same bits and visit a row's columns in
// one order. `#include <seismic/gather.cuh>`.
//
// The table tensor is bound to the gather slot before this file is
// included:
//     #define KERNEL_T SEISMIC_TABLE
// which defines `packets::T` (the table's gather type) and
// `KERNEL_T_AT(pointer)` (its value for the table at `pointer`). A dense
// table is rank 1 or 2 (`PACKETS_MAKE_DENSE`); a dense binding of another
// rank is an `#error`, so a kernel is not implemented there. A gather
// reads the table through `packets::gather_row` (a row, by the block),
// `packets::gather_row_lanes` (a row, by a warp's lanes) or, for a rows16 or
// dense table, `packets::gather_value` (one element).

#pragma once

#include <seismic/element.cuh>
#include <seismic/packets.cuh>
#include <seismic/rows16.cuh>

namespace packets {

namespace detail {

// Whether a gather type is a dense table, and whether a rows16 one.
template <class T> struct GatherDense {
    static constexpr bool value = false;
};
template <class E> struct GatherDense<Dense<E>> {
    static constexpr bool value = true;
};
template <class T> struct GatherRows16 {
    static constexpr bool value = false;
};
template <u32 LOW, u32 HIGH, class C, class K> struct GatherRows16<rows16::detail::Packed<LOW, HIGH, C, K>> {
    static constexpr bool value = true;
};
template <class G, class K> struct GatherRows16<rows16::detail::Grid<G, K>> {
    static constexpr bool value = true;
};

// `gather_row`'s visit by the participants `start`, `start + step`, ... (a
// static member, which argument-dependent lookup does not reach).
struct Gather {
template <class Table, class Visit>
__device__ static __forceinline__ void row(const Table &table, u64 row, u64 first, u64 width, u64 start, u64 step,
                                           Visit visit) {
    if constexpr (GatherDense<Table>::value) {
        for (u64 column = start; column < width; column += step)
            visit(column, table.value(row, first + column));
    } else if constexpr (GatherRows16<Table>::value) {
        for (u64 item = start; item < width / 64 * 4; item += step) {
            const u64 kb = item / 4;
            const u32 t = (u32)(item % 4);
#pragma unroll
            for (int s = 0; s < 4; ++s)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const u64 column = item16_column(kb, t, s, j);
                    visit(column, table.value(row, first + column));
                }
        }
    } else {
        row_items16(table, row, first, width, start, step, visit);
    }
}
};

} // namespace detail

// `visit(column, value)` for every column of row `row` of `table` from column
// `first` (a multiple of 64, in a packed table) for `width` columns,
// `column` counted from `first`, by the threads of the block. A packed table
// is read by work item (k-block, column pair t), 16 values each, in
// `row_items16`'s order (the mma16 lane chunk's); a dense table a column per
// thread.
template <class Table, class Visit>
__device__ __forceinline__ void gather_row(const Table &table, u64 row, u64 first, u64 width, Visit visit) {
    detail::Gather::row(table, row, first, width, threadIdx.x, blockDim.x, visit);
}

// `gather_row` by the 32 lanes of a warp, lane `lane` (in 0..32) taking
// every 32nd work item (a dense table's column) from its own.
template <class Table, class Visit>
__device__ __forceinline__ void gather_row_lanes(const Table &table, u64 row, u64 first, u64 width, u32 lane,
                                                 Visit visit) {
    detail::Gather::row(table, row, first, width, lane, 32, visit);
}

// Element `i` of row `row` of `table`, alone: a dense or rows16 table's (an
// mma16 table's values lie in lane chunks, read a row at a time by
// `gather_row`).
template <class Table> __device__ __forceinline__ float gather_value(const Table &table, u64 row, u64 i) {
    static_assert(detail::GatherDense<Table>::value || detail::GatherRows16<Table>::value,
                  "a table's single values are read in the dense or rows16 layout, not mma16");
    return table.value(row, i);
}

} // namespace packets

#if defined(KERNEL_T)
#if ELEMENT_HAS(KERNEL_T, _KIND_DENSE)
#if ELEMENT_CAT(KERNEL_T, _RANK) > 2
#error "a dense KERNEL_T must be a rank-1 or rank-2 table"
#endif
namespace packets { typedef detail::Dense<ELEMENT_OF(KERNEL_T)> T; }
#define KERNEL_T_AT(pointer) PACKETS_MAKE_DENSE(KERNEL_T, pointer)
#elif ELEMENT_HAS(KERNEL_T, _LAYOUT_ROWS16)
namespace packets {
typedef rows16::detail::Decoder<
    ELEMENT_CAT(KERNEL_T, _CODES), ELEMENT_CAT(KERNEL_T, _CODE_LOW_BITS), ELEMENT_CAT(KERNEL_T, _CODE_HIGH_BITS),
    ELEMENT_CAT(KERNEL_T, _CODES_OFFSET), ELEMENT_CAT(KERNEL_T, _CODES_TABLE), ELEMENT_CAT(KERNEL_T, _CODES_GRID),
    ELEMENT_CAT(KERNEL_T, _CODES_GRID_INDEX_BITS), ELEMENT_CAT(KERNEL_T, _CODES_GRID_LANES),
    ELEMENT_CAT(KERNEL_T, _CODES_GRID_SIGNS), ELEMENT_CAT(KERNEL_T, _CODES_GRID_DELTA),
    ELEMENT_CAT(KERNEL_T, _CODES_GRID_DELTA_EXPONENT), ELEMENT_CAT(KERNEL_T, _CODES_GRID_SIGN_BIT),
    ELEMENT_CAT(KERNEL_T, _CODES_GRID_SIGN_BITS), ELEMENT_CAT(KERNEL_T, _CODES_GRID_DELTA_BIT),
    PACKETS_COEFFICIENTS(KERNEL_T)>::type T;
}
#define KERNEL_T_AT(pointer)                                                                                   \
    packets::T {                                                                                              \
        rows16::detail::Planes {                                                                              \
            (const packets::u8 *)(pointer), (packets::u64)ELEMENT_CAT(KERNEL_T, _ROW_STRIDE_BYTES),           \
                (packets::u64)ELEMENT_CAT(KERNEL_T, _ROW_CODES), (packets::u64)ELEMENT_CAT(KERNEL_T, _ROW_HIGH), \
                (packets::u64)ELEMENT_CAT(KERNEL_T, _ROW_SCALES), (packets::u64)ELEMENT_CAT(KERNEL_T, _ROW_SUPERS) \
        }                                                                                                     \
    }
#elif ELEMENT_HAS(KERNEL_T, _LAYOUT_MMA16)
namespace packets { typedef PACKETS_DECODER(KERNEL_T) T; }
#if ELEMENT_HAS(KERNEL_T, _PLANE_RECORDS)
#define KERNEL_T_AT(pointer) PACKETS_MAKE_RECORDS(packets::T, KERNEL_T, pointer)
#else
#define KERNEL_T_AT(pointer) PACKETS_MAKE(packets::T, KERNEL_T, pointer)
#endif
#else
#error "KERNEL_T must be dense or bound in the mma16 or rows16 layout"
#endif
#endif
