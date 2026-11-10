// Expert streaming: a streamed decode layer's landing launch (`routed_land`
// in routed.seismic). Where the bank places (V > 0), every threadgroup
// converts the layer's file misses from staging into the bank's records as
// their reads land (streaming/land.hpp over `lib/streaming/land.h`), its
// thread 0 spinning on the host's count of the step's slot of the layer's
// file table; then threadgroup 0 writes the absent choices' sources
// (streaming/tier_wait.hpp), the slot answered by then. Where the tier
// places, the one threadgroup only sources. A threadgroup of a step that
// requested nothing returns at its first look. The staging and the bank
// are record ports (views of lent records), each record read or written at
// its own base.

#include <seismic/slab.h>
#include "lib/streaming/control.h"
#include "lib/streaming/window.h"

REPACK_BINDING(GateBinding, SEISMIC_CONVERSION_SG_UG, SEISMIC_STAGING_GATE, SEISMIC_GATE, SEISMIC_DIM_F, SEISMIC_DIM_H)
REPACK_BINDING(UpBinding, SEISMIC_CONVERSION_SU_UU, SEISMIC_STAGING_UP, SEISMIC_UP, SEISMIC_DIM_F, SEISMIC_DIM_H)
REPACK_BINDING(DownBinding, SEISMIC_CONVERSION_SD_UD, SEISMIC_STAGING_DOWN, SEISMIC_DOWN, SEISMIC_DIM_H, SEISMIC_DIM_F)

#include "lib/streaming/land.h"
#include "../streaming/tier_wait.hpp"

kernel void routed_land(
    device const int *step [[buffer(SEISMIC_BUFFER_STEP)]],
    SYSTEM int *file [[buffer(SEISMIC_BUFFER_FILE)]],
    device int *sources [[buffer(SEISMIC_BUFFER_SOURCES)]],
    SYSTEM int *landing [[buffer(SEISMIC_BUFFER_LANDING)]],
    device const ulong *staging_gate [[buffer(SEISMIC_BUFFER_STAGING_GATE)]],
    device const ulong *staging_up [[buffer(SEISMIC_BUFFER_STAGING_UP)]],
    device const ulong *staging_down [[buffer(SEISMIC_BUFFER_STAGING_DOWN)]],
    device const ulong *gate [[buffer(SEISMIC_BUFFER_GATE)]],
    device const ulong *up [[buffer(SEISMIC_BUFFER_UP)]],
    device const ulong *down [[buffer(SEISMIC_BUFFER_DOWN)]],
    device int *counters [[buffer(SEISMIC_BUFFER_SCRATCH_COUNTERS)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    threadgroup uchar *window [[threadgroup(0)]],
    uint group [[threadgroup_position_in_grid]],
    uint groups [[threadgroups_per_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup int words[SEISMIC_CONST_LAND_GROUP_WORDS];
    const ulong q = streaming_step(step, SEISMIC_STEP_STRIDE_0);
    if (SEISMIC_DIM_V > 0) {
        streaming::Lander land{{staging_gate, staging_up, staging_down}, {gate, up, down}, SEISMIC_DIM_GV > 0,
            seismic_words, window, tid, threads, simd, simdgroups, lane};
        streaming_land(land, landing, SEISMIC_LANDING_STRIDE_0,
            file + (q % ulong(SEISMIC_DIM_FLIGHT)) * SEISMIC_FILE_STRIDE_0, SEISMIC_FILE_STRIDE_1, q, counters,
            uint(SEISMIC_DIM_L), groups, words, tid);
    }
    if (group == 0)
        streaming_tier_wait(step, SEISMIC_STEP_STRIDE_0, file, SEISMIC_FILE_STRIDE_0, SEISMIC_FILE_STRIDE_1, sources,
            SEISMIC_SOURCES_STRIDE_0, SEISMIC_SOURCES_STRIDE_1, SEISMIC_SOURCES_STRIDE_2, tid, threads, seismic_words);
}
