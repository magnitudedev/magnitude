// Expert streaming: a streamed layer's tier wait (`routed_tier_wait` in
// routed.seismic); the CUDA form of `metal/routed_tier_wait.metal`, the body
// of the route's wait grid's one block (`_block`): its threads wait on the
// step's slot of the layer's file table, then write the absent choices'
// sources (streaming/tier_wait.hpp).

#include "lib/streaming/control.cuh"
#include "../streaming/tier_wait.hpp"

#ifdef SEISMIC_FORMING_ROUTED_TIER_WAIT
__device__ void routed_tier_wait_block(SEISMIC_KERNEL_PARAMS) {
    streaming_tier_wait(reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_STEP)), SEISMIC_STEP_STRIDE_0,
                        reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_FILE)), SEISMIC_FILE_STRIDE_0,
                        SEISMIC_FILE_STRIDE_1, reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_SOURCES)),
                        SEISMIC_SOURCES_STRIDE_0, SEISMIC_SOURCES_STRIDE_1, SEISMIC_SOURCES_STRIDE_2, threadIdx.x,
                        blockDim.x, seismic_words_value);
}
#endif
