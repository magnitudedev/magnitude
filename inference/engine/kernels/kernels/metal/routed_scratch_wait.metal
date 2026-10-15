// Expert streaming, one round of a scratch fill (`routed_scratch_wait` in
// routed.seismic): thread 0 publishes the step's sequence at the round's
// arrival word, then waits for it at the round's acknowledgement word or the
// failure word, failing the step itself if the host's heartbeat stalls.

#include "lib/streaming/control.h"

kernel void routed_scratch_wait(
    device const int *step [[buffer(SEISMIC_BUFFER_STEP)]],
    SYSTEM int *control [[buffer(SEISMIC_BUFFER_CONTROL)]],
    SYSTEM const int *heartbeat [[buffer(SEISMIC_BUFFER_HEARTBEAT)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint tid [[thread_index_in_threadgroup]]) {
    if (tid != 0) return;
    const int sequence = step[0];
    const uint acknowledgement = uint(SEISMIC_PARAM_ACKNOWLEDGEMENT);
    streaming::system_fence();
    control[uint(SEISMIC_PARAM_ARRIVAL)] = sequence;
    streaming::system_fence();
    streaming::await_word(control + acknowledgement, control + streaming::FAILURE, heartbeat, sequence);
}
