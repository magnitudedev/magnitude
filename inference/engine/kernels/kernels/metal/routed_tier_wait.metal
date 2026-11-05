// Expert streaming: a streamed layer's tier wait (`routed_tier_wait` in
// routed.seismic): one threadgroup, its threads waiting on the step's slot
// of the layer's file table, then writing the absent choices' sources
// (streaming/tier_wait.hpp).

#include "lib/streaming/control.h"
#include "../streaming/tier_wait.hpp"

kernel void routed_tier_wait(
    device const int *step [[buffer(SEISMIC_BUFFER_STEP)]],
    SYSTEM const int *file [[buffer(SEISMIC_BUFFER_FILE)]],
    device int *sources [[buffer(SEISMIC_BUFFER_SOURCES)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
    streaming_tier_wait(step, SEISMIC_STEP_STRIDE_0, file, SEISMIC_FILE_STRIDE_0, SEISMIC_FILE_STRIDE_1, sources,
        SEISMIC_SOURCES_STRIDE_0, SEISMIC_SOURCES_STRIDE_1, SEISMIC_SOURCES_STRIDE_2, tid, threads, seismic_words);
}
