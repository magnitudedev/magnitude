// Expert streaming, one round of a scratch fill (`routed_round` in
// routed.seismic): every threadgroup's thread 0 stores the round's arrival
// (the request of its lane of the step's slot of the layer's scratch
// table, streaming/round.hpp), then the threadgroups convert the round's
// conveyed records into the scratch tier as their reads land
// (streaming/land.hpp over `lib/streaming/land.h`), spinning on the host's
// count until the lane is answered. A negative `from` names scratch slot -1
// - from, the stored bytes the host read there (a slot no pass reads),
// read through the same coherent pointer as a conveyor record. The conveyor
// and the scratch tier are record ports (views of lent records, the
// scratch tier the scratch lender's), each record read or written at its
// own base.

#include <seismic/slab.h>
#include "lib/streaming/control.h"
#include "lib/streaming/window.h"

REPACK_BINDING(GateBinding, SEISMIC_CONVERSION_SG_UG, SEISMIC_CONVEYOR_GATE, SEISMIC_GATE, SEISMIC_DIM_F, SEISMIC_DIM_H)
REPACK_BINDING(UpBinding, SEISMIC_CONVERSION_SU_UU, SEISMIC_CONVEYOR_UP, SEISMIC_UP, SEISMIC_DIM_F, SEISMIC_DIM_H)
REPACK_BINDING(DownBinding, SEISMIC_CONVERSION_SD_UD, SEISMIC_CONVEYOR_DOWN, SEISMIC_DOWN, SEISMIC_DIM_H, SEISMIC_DIM_F)

#include "lib/streaming/land.h"
#include "../streaming/round.hpp"

kernel void routed_round(
    device const int *step [[buffer(SEISMIC_BUFFER_STEP)]],
    SYSTEM int *scratch [[buffer(SEISMIC_BUFFER_SCRATCH)]],
    SYSTEM int *landing [[buffer(SEISMIC_BUFFER_LANDING)]],
    device const ulong *conveyor_gate [[buffer(SEISMIC_BUFFER_CONVEYOR_GATE)]],
    device const ulong *conveyor_up [[buffer(SEISMIC_BUFFER_CONVEYOR_UP)]],
    device const ulong *conveyor_down [[buffer(SEISMIC_BUFFER_CONVEYOR_DOWN)]],
    device const ulong *gate [[buffer(SEISMIC_BUFFER_GATE)]],
    device const ulong *up [[buffer(SEISMIC_BUFFER_UP)]],
    device const ulong *down [[buffer(SEISMIC_BUFFER_DOWN)]],
    device int *counters [[buffer(SEISMIC_BUFFER_SCRATCH_COUNTERS)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    threadgroup uchar *window [[threadgroup(0)]],
    uint groups [[threadgroups_per_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup int words[SEISMIC_CONST_LAND_GROUP_WORDS];
    const ulong q = streaming_step(step, SEISMIC_STEP_STRIDE_0);
    SYSTEM int *slot = streaming_round_lane(q, ulong(SEISMIC_PARAM_ROUND), scratch, seismic_words);
    if (tid == 0)
        streaming_round_arrive(slot, SEISMIC_SCRATCH_STRIDE_1, q);
    streaming::Lander land{{conveyor_gate, conveyor_up, conveyor_down}, {gate, up, down}, SEISMIC_DIM_GV > 0,
        seismic_words, window, tid, threads, simd, simdgroups, lane};
    streaming_land(land, landing, SEISMIC_LANDING_STRIDE_0, slot, SEISMIC_SCRATCH_STRIDE_1, q, counters,
        uint(SEISMIC_DIM_L), groups, words, tid);
}
