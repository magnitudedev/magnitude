// Expert streaming, the step's words (`routed_step` in routed.seismic); the
// CUDA form of `metal/routed_step.metal`: one thread writes them
// (streaming/step.hpp). Every later entry of the step reads them in order on
// the same stream.

#include "lib/streaming/control.cuh"
#include "../streaming/step.hpp"

extern "C" __global__ void routed_step(SEISMIC_KERNEL_PARAMS) {
    if (threadIdx.x != 0)
        return;
    streaming_step_words(reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_STEP)), SEISMIC_STEP_STRIDE_0,
        reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_SEQUENCE)), SEISMIC_SEQUENCE_STRIDE_0,
        reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_GUESSES)), SEISMIC_GUESSES_STRIDE_0,
        SEISMIC_GUESSES_STRIDE_1, static_cast<unsigned>(SEISMIC_DIM_T));
}
