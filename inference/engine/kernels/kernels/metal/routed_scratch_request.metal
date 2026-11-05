// Expert streaming, a scratch-served layer's request (`routed_scratch_request`
// in routed.seismic): one threadgroup's request
// (streaming/scratch_request.hpp), its threadgroup words a word an expert.

#include "lib/streaming/control.h"
#include "../streaming/round.hpp"
#include "../streaming/scratch_request.hpp"

kernel void routed_scratch_request(
    device const int *routes [[buffer(SEISMIC_BUFFER_ROUTES)]],
    device const int *step [[buffer(SEISMIC_BUFFER_STEP)]],
    SYSTEM int *flags [[buffer(SEISMIC_BUFFER_FLAGS)]],
    SYSTEM int *scratch [[buffer(SEISMIC_BUFFER_SCRATCH)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
    threadgroup int counts[SEISMIC_DIM_E];
    streaming_scratch_request(routes, step, flags, scratch, counts, tid, threads, seismic_words);
}
