// Seismic native library (Metal): simdgroup-cooperative row gathers of a
// table in any layout (`rows16`, `rows32` or dense). `#include <seismic/gather.h>`.
//
// The table tensor may be bound to `KERNEL_T` before this file is included
// (matching CUDA's `<seismic/gather.cuh>`):
//     #define KERNEL_T SEISMIC_TABLE
// which binds `KERNEL_W0`, defines `packets::T` (the table's decoder type)
// and `KERNEL_T_AT(pointer, width)` (its `packets::Matrix<packets::T>`).
// A kernel that binds `KERNEL_W0` directly may also include this header and
// pass any `packets::Matrix<W>`.
//
// For a quantized table, a simdgroup of 32 lanes cooperatively loads and
// decodes each 32-element packet once (lanes 0..3 each decoding one 8-element
// step via `W::exact_values`, then distributing the 32 values with
// `simd_shuffle`) rather than 32x per thread. For a dense table, every lane
// loads its own element directly in coalesced 32-lane accesses.

#if defined(KERNEL_T)
#if defined(KERNEL_W0)
#error "define KERNEL_T or KERNEL_W0, not both"
#endif
#define KERNEL_W0 KERNEL_T
#endif

#include <seismic/packets.h>

namespace packets {

namespace detail {

template <typename W>
struct Gather {
    // Cooperatively decode packet `p` of located row `at` across the 32 lanes
    // of a simdgroup, returning element `lane` (0..31) to lane `lane`.
    // All 32 lanes of the simdgroup must call this convergently with the same
    // `at` and `p`.
    static inline float packet_lane(thread const Matrix<W> &table, thread const Located &at, uint p, uint lane) {
        float4 even = 0.0f, odd = 0.0f;
        if (lane < 4u) {
            const typename W::packet k = table.packet(at, p);
            W::exact_values(k, lane, even, odd);
        }
        const ushort src = ushort(lane >> 3);
        const float4 e = metal::simd_shuffle(even, src);
        const float4 o = metal::simd_shuffle(odd, src);
        const float4 vec = (lane & 1u) ? o : e;
        const float2 lo_hi = (lane & 4u) ? vec.zw : vec.xy;
        return (lane & 2u) ? lo_hi.y : lo_hi.x;
    }

    template <typename Visit>
    static inline void row(thread const Matrix<W> &table, thread const Located &at, uint first, uint width,
        uint sg, uint simdgroups, uint lane, Visit visit) {
        const uint first_packet = first / 32u;
        const uint packets = (width + 31u) / 32u;
        for (uint p = sg; p < packets; p += simdgroups) {
            const float value = packet_lane(table, at, first_packet + p, lane);
            const uint column = p * 32u + lane;
            if (column < width)
                visit(column, value);
        }
    }
};

template <typename E>
struct Gather<Dense<E>> {
    static inline float packet_lane(thread const Matrix<Dense<E>> &table, thread const Located &at, uint p, uint lane) {
        const uint col = p * 32u + lane;
        if (col >= table.k)
            return 0.0f;
        const auto values = slab::as<typename E::storage>(Dense<E>::row_values(at));
        return E::load(values[col]);
    }

    template <typename Visit>
    static inline void row(thread const Matrix<Dense<E>> &, thread const Located &at, uint first, uint width,
        uint sg, uint simdgroups, uint lane, Visit visit) {
        const auto values = slab::as<typename E::storage>(Dense<E>::row_values(at));
        const uint step = simdgroups * 32u;
        for (uint column = sg * 32u + lane; column < width; column += step)
            visit(column, E::load(values[first + column]));
    }
};

} // namespace detail

// Cooperatively decode packet `p` of located row `at` across a simdgroup's
// 32 lanes, returning element `lane` to lane `lane`. All 32 lanes must call
// convergently with the same `at` and `p`.
template <typename W>
inline float gather_packet(thread const Matrix<W> &table, thread const Located &at, uint p, uint lane) {
    return detail::Gather<W>::packet_lane(table, at, p, lane);
}

// `visit(column, value)` for every column of row `row` of `table` from column
// `first` (a multiple of 32 in a packed table) for `width` columns, `column`
// counted from `first`, by a threadgroup of `simdgroups` simdgroups (`sg` in
// `0..simdgroups`, `lane` in `0..32`). All 32 lanes of every simdgroup must
// call convergently.
template <typename W, typename Visit>
inline void gather_row(thread const Matrix<W> &table, ulong row, uint first, uint width,
    uint sg, uint simdgroups, uint lane, Visit visit) {
    const Located at = table.locate(row);
    detail::Gather<W>::row(table, at, first, width, sg, simdgroups, lane, visit);
}

// `gather_row` by one simdgroup's 32 lanes.
template <typename W, typename Visit>
inline void gather_row_lanes(thread const Matrix<W> &table, ulong row, uint first, uint width,
    uint lane, Visit visit) {
    const Located at = table.locate(row);
    detail::Gather<W>::row(table, at, first, width, 0u, 1u, lane, visit);
}

} // namespace packets

#if defined(KERNEL_T)
namespace packets { typedef W0 T; }
#define KERNEL_T_AT(pointer, width) packets::Matrix<packets::T>{pointer, KERNEL_W0_LAYOUT(width), width}
#endif
