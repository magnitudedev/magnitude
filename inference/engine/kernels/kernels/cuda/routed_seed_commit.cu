// Expert streaming, a prompt group's seeding (`routed_seed_commit` in
// routed.seismic); the CUDA form of `metal/routed_seed_commit.metal`: one
// block's seeding (streaming/seed.hpp).

#include "lib/streaming/control.cuh"
#include "../streaming/list.hpp"
#include "../streaming/round.hpp"
#include "../streaming/seed.hpp"

extern "C" __global__ void routed_seed_commit(SEISMIC_KERNEL_PARAMS) {
    // The count.
    __shared__ int words[STREAMING_SEED_WORDS];
    streaming_seed_commit(reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_LIST)),
                           reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_STEP)), SEISMIC_STEP_STRIDE_0,
                           reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH)),
                           reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_COMMITS)), words, threadIdx.x,
                          blockDim.x, seismic_words_value);
}
