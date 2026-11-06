// Expert streaming, one round of a scratch fill (`routed_scratch_wait` in
// routed.seismic); the CUDA form of `metal/routed_scratch_wait.metal`, the
// body of one block of its written park grid (`_block`, one block while the
// round, or unfollowed the round before, has records to convert, none
// otherwise): thread 0 requests the round's lane of the step's slot of the
// layer's scratch table, then spins until it is answered
// (streaming/round.hpp).

#include "lib/streaming/control.cuh"
#include "../streaming/round.hpp"

#ifdef SEISMIC_FORMING_ROUTED_SCRATCH_WAIT
__device__ void routed_scratch_wait_block(SEISMIC_KERNEL_PARAMS) {
    if (threadIdx.x != 0)
        return;
    const u64 q = streaming_step(reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_STEP)), SEISMIC_STEP_STRIDE_0);
    streaming_round_wait(q, (u64)SEISMIC_PARAM_ROUND, reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH)),
                         seismic_words_value);
}
#endif
