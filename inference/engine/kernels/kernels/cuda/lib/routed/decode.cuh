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

} // namespace routed
