// Expert streaming, one round of a scratch fill (`routed_scratch_wait` in
// routed.seismic): thread 0 requests the round's lane of the step's slot of
// the layer's scratch table, then spins until it is answered
// (streaming/round.hpp).

#include "lib/streaming/control.h"
#include "../streaming/round.hpp"

kernel void routed_scratch_wait(
    device const int *step [[buffer(SEISMIC_BUFFER_STEP)]],
    SYSTEM int *scratch [[buffer(SEISMIC_BUFFER_SCRATCH)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint tid [[thread_index_in_threadgroup]]) {
    if (tid != 0)
        return;
    streaming_round_wait(streaming_step(step, SEISMIC_STEP_STRIDE_0), ulong(SEISMIC_PARAM_ROUND), scratch,
        seismic_words);
}
