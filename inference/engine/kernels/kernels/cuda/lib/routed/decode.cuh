// The block of the routed projection's decode GEMVs (`choice` and `summed`,
// routed.seismic), for every entry that runs one, the per-row gathered
// entries' `gather_choice` included.
#pragma once
#include "../projection/projection.cuh"

namespace routed {

// The decode GEMV kernels' block: DECODE_WARPS warps (the declarations' 128 threads).
// Records weights (`GemvRecords`, e.g. iq2xxs) and NB=1 or unpaired entry-order weights
// (`GemvEntries`, e.g. iq1m/iq2xs) need multi-block SM residency (`16 / DECODE_WARPS` =
// 4 blocks/SM, <=128 regs/thread) to hide memory latency; paired NB>1 entry-order and
// MMA/hier weights spill if capped below their natural register count.
constexpr int DECODE_WARPS = 4;
constexpr int DECODE_THREADS = DECODE_WARPS * 32;
template <int NB, class... W> __host__ __device__ constexpr int decode_min_blocks() {
    constexpr int LIVE_WEIGHTS = ((!projection::Same<W, projection::NoWeight>::value ? 1 : 0) + ...);
    constexpr bool HIGH_OCCUPANCY = ((projection::GemvRecords<W>::value
                                      || ((NB == 1 || LIVE_WEIGHTS == 1) && projection::GemvEntries<W>::value)
                                      || projection::Same<W, projection::NoWeight>::value)
                                     && ...);
    return HIGH_OCCUPANCY ? (16 / DECODE_WARPS) : 1;
}
constexpr int DECODE_BATCHES = (SEISMIC_CONST_VERIFY_ROWS + 7) / 8;
static_assert(DECODE_BATCHES * 8 >= SEISMIC_CONST_VERIFY_ROWS, "DecodeGatheredShape must cover VERIFY_ROWS");
template <int TPW, int KSPLIT, int NB = 1> using DecodeShape = projection::GemvShape<DECODE_WARPS, TPW, KSPLIT, NB>;
template <int TPW, int KSPLIT, int NB = DECODE_BATCHES>
using DecodeGatheredShape = DecodeShape<TPW, KSPLIT, NB>;

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
