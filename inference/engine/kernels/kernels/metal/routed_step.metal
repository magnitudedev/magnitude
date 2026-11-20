// Expert streaming, the step's words (`routed_step` in routed.seismic): one
// thread writes them (streaming/step.hpp). Every later entry of the step
// reads them in order on the same queue.

#include "lib/streaming/control.h"
#include "../streaming/step.hpp"

kernel void routed_step(
    device int *step [[buffer(SEISMIC_BUFFER_STEP)]],
    device const int *sequence [[buffer(SEISMIC_BUFFER_SEQUENCE)]],
    device const int *guesses [[buffer(SEISMIC_BUFFER_GUESSES)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint tid [[thread_index_in_threadgroup]]) {
    if (tid == 0)
        streaming_step_words(step, SEISMIC_STEP_STRIDE_0, sequence, SEISMIC_SEQUENCE_STRIDE_0, guesses,
            SEISMIC_GUESSES_STRIDE_0, SEISMIC_GUESSES_STRIDE_1, uint(SEISMIC_DIM_T));
}
