// Expert streaming, a scratch request's round grids (`routed_scratch_grids`
// in routed.seismic): thread 0 writes them (streaming/grids.hpp).

#include "lib/streaming/control.h"
#include "../streaming/round.hpp"
#include "../streaming/grids.hpp"

kernel void routed_scratch_grids(
    device const int *step [[buffer(SEISMIC_BUFFER_STEP)]],
    SYSTEM const int *scratch [[buffer(SEISMIC_BUFFER_SCRATCH)]],
    device int *grids [[buffer(SEISMIC_BUFFER_GRIDS)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint tid [[thread_index_in_threadgroup]]) {
    if (tid != 0)
        return;
    streaming_scratch_grids(streaming_step(step, SEISMIC_STEP_STRIDE_0), scratch, grids, seismic_words);
}
