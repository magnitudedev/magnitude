// Expert streaming, a host-written list staged into device memory
// (`routed_list_stage` in routed.seismic); the CUDA form of
// `metal/routed_list_stage.metal`: the block stages it (streaming/list.hpp),
// which the launches after it read in order on the same stream.

#include "lib/streaming/control.cuh"
#include "../streaming/list.hpp"

extern "C" __global__ void routed_list_stage(SEISMIC_KERNEL_PARAMS) {
    streaming_list_stage(reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_LIST)), SEISMIC_LIST_STRIDE_0,
                         reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_STEP))[0],
                         reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_STAGED)), SEISMIC_STAGED_STRIDE_0,
                         threadIdx.x, blockDim.x);
}
