// Expert streaming, a merge set's pass's map resolved into device memory
// (`routed_pass_map` in routed.seismic): the threadgroup resolves it
// (streaming/pass_map.hpp).

#include "lib/streaming/control.h"
#include "../streaming/pass_map.hpp"

kernel void routed_pass_map(
    SYSTEM const int *map [[buffer(SEISMIC_BUFFER_MAP)]],
    device const int *step [[buffer(SEISMIC_BUFFER_STEP)]],
    device int *resolved [[buffer(SEISMIC_BUFFER_RESOLVED)]],
    device int *pass_step [[buffer(SEISMIC_BUFFER_PASS_STEP)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
    streaming_pass_map(map, SEISMIC_MAP_STRIDE_0, SEISMIC_MAP_STRIDE_1, SEISMIC_DIM_E, step, SEISMIC_STEP_STRIDE_0,
        resolved, SEISMIC_RESOLVED_STRIDE_0, SEISMIC_RESOLVED_STRIDE_1, pass_step, SEISMIC_PASS_STEP_STRIDE_0, tid,
        threads);
}
