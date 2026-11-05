// Expert streaming, a merge set's pass's map resolved into device memory
// (`routed_pass_map` in routed.seismic); the CUDA form of
// `metal/routed_pass_map.metal`: the block resolves it (streaming/pass_map.hpp).

#include "lib/streaming/control.cuh"
#include "../streaming/pass_map.hpp"

extern "C" __global__ void routed_pass_map(SEISMIC_KERNEL_PARAMS) {
    streaming_pass_map(reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_MAP)), SEISMIC_MAP_STRIDE_0,
                       SEISMIC_MAP_STRIDE_1, SEISMIC_DIM_E, reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_STEP)),
                       SEISMIC_STEP_STRIDE_0, reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_RESOLVED)),
                       SEISMIC_RESOLVED_STRIDE_0, SEISMIC_RESOLVED_STRIDE_1,
                       reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_PASS_STEP)), SEISMIC_PASS_STEP_STRIDE_0,
                       threadIdx.x, blockDim.x);
}
