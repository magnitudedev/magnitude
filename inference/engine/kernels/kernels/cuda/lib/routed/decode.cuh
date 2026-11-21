// The block of the routed projection's decode GEMVs (`choice` and `summed`,
// routed.seismic), for every entry that runs one, the per-row gathered
// entries' `gather_choice` included.
#pragma once
#include "../projection/projection.cuh"

namespace routed {

// The decode GEMV kernels' block: DECODE_WARPS warps (the declarations' 128 threads). A kernel declares it as its
// launch bound (`routed_up` does) so the compiler allocates registers for that block, not for the largest one the
// hardware admits; the others adopt the bound where it is measured to pay.
constexpr int DECODE_WARPS = 4;
constexpr int DECODE_THREADS = DECODE_WARPS * 32;
template <int TPW, int KSPLIT> using DecodeShape = projection::GemvShape<DECODE_WARPS, TPW, KSPLIT, 1>;

// The grid-point readers and points source of a routed decode GEMV over
// weights WA (and optional paired WB). Multi-branch kernels combine their
// branches' readers with `DecodeUnion` so `GemvShared` uses `BlockPoints`
// whenever any branch reads a grid.
template <class WA, class WB = projection::NoWeight>
using DecodeReaders = typename projection::GemvPointReaders<false, WA, WB>::type;
template <class WA, class WB = projection::NoWeight>
using DecodePoints = typename projection::PointsOf<DecodeReaders<WA, WB>>::type;
template <class A, class B> struct DecodeUnion;
template <class... As, class... Bs> struct DecodeUnion<projection::Types<As...>, projection::Types<Bs...>> {
    using type = projection::Types<As..., Bs...>;
};

} // namespace routed
