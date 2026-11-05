// Expert streaming, a host-written list's launch grid (`routed_list_grid` in
// routed.seismic); the CUDA form of `metal/routed_list_grid.metal`: the
// block stages the list into device memory and thread 0 writes its grid
// (streaming/list.hpp), which the copy after it reads in order on the same
// stream.

#include "lib/streaming/control.cuh"
#include "../streaming/list.hpp"

extern "C" __global__ void routed_list_grid(SEISMIC_KERNEL_PARAMS) {
    const int step = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_STEP))[0];
    streaming_list_stage(reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_LIST)), SEISMIC_LIST_STRIDE_0,
                         step, reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_STAGED)),
                         SEISMIC_STAGED_STRIDE_0, threadIdx.x, blockDim.x);
    if (threadIdx.x != 0)
        return;
    streaming_list_grid(reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_LIST)), SEISMIC_LIST_STRIDE_0,
                        reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_STEP))[0],
                        reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_GRID)), SEISMIC_GRID_STRIDE_0);
}
