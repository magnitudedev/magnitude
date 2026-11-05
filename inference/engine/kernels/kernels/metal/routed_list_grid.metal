// Expert streaming, a host-written list's launch grid (`routed_list_grid` in
// routed.seismic): the threadgroup stages the list into device memory and
// thread 0 writes its grid (streaming/list.hpp).

#include "lib/streaming/control.h"
#include "../streaming/list.hpp"

kernel void routed_list_grid(
    SYSTEM const int *list [[buffer(SEISMIC_BUFFER_LIST)]],
    device const int *step [[buffer(SEISMIC_BUFFER_STEP)]],
    device int *grid [[buffer(SEISMIC_BUFFER_GRID)]],
    device int *staged [[buffer(SEISMIC_BUFFER_STAGED)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
    streaming_list_stage(list, SEISMIC_LIST_STRIDE_0, step[0], staged, SEISMIC_STAGED_STRIDE_0, tid, threads);
    if (tid != 0)
        return;
    streaming_list_grid(list, SEISMIC_LIST_STRIDE_0, step[0], grid, SEISMIC_GRID_STRIDE_0);
}
