// Expert streaming, a scratch request's round grids (`routed_scratch_grids`
// in routed.seismic); the CUDA form of `metal/routed_scratch_grids.metal`:
// thread 0 writes them (streaming/grids.hpp), which the step's later launches
// read in order on the same stream.

#include "lib/streaming/control.cuh"
#include "../streaming/round.hpp"
#include "../streaming/grids.hpp"

extern "C" __global__ void routed_scratch_grids(SEISMIC_KERNEL_PARAMS) {
    if (threadIdx.x != 0)
        return;
    streaming_scratch_grids(
        streaming_step(reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_STEP)), SEISMIC_STEP_STRIDE_0),
        reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH)),
        reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_GRIDS)), seismic_words_value);
}
