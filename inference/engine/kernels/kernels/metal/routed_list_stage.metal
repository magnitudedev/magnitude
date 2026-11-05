// Expert streaming, a host-written list staged into device memory
// (`routed_list_stage` in routed.seismic): the threadgroup stages it
// (streaming/list.hpp).

#include "lib/streaming/control.h"
#include "../streaming/list.hpp"

kernel void routed_list_stage(
    SYSTEM const int *list [[buffer(SEISMIC_BUFFER_LIST)]],
    device const int *step [[buffer(SEISMIC_BUFFER_STEP)]],
    device int *staged [[buffer(SEISMIC_BUFFER_STAGED)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
    streaming_list_stage(list, SEISMIC_LIST_STRIDE_0, step[0], staged, SEISMIC_STAGED_STRIDE_0, tid, threads);
}
