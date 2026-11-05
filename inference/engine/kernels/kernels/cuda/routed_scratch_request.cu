// Expert streaming, a scratch-served layer's request (`routed_scratch_request`
// in routed.seismic); the CUDA form of `metal/routed_scratch_request.metal`:
// one block's request (streaming/scratch_request.hpp), its shared words a
// word an expert.

#include "lib/streaming/control.cuh"
#include "../streaming/round.hpp"
#include "../streaming/scratch_request.hpp"

extern "C" __global__ void routed_scratch_request(SEISMIC_KERNEL_PARAMS) {
    __shared__ int counts[SEISMIC_DIM_E];
    streaming_scratch_request(reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES)),
                              reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_STEP)),
                              reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_FLAGS)),
                              reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH)), counts, threadIdx.x,
                              blockDim.x, seismic_words_value);
}
