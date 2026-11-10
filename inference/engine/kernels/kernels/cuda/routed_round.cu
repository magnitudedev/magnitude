// Expert streaming, one round of a scratch fill (`routed_round` in
// routed.seismic); the CUDA form of `metal/routed_round.metal`, the body of
// one block of its written park grid (`_block`, LAND_GROUPS blocks while
// the round, or unfollowed the round before, has records, none otherwise):
// thread 0 stores the round's arrival (streaming/round.hpp), then the
// blocks convert the round's conveyed records into the scratch tier as
// their reads land (streaming/land.hpp over `lib/streaming/land.cuh`),
// spinning on the host's count until the round's lane is answered. A
// negative `from` names scratch slot -1 - from, which holds the stored
// bytes the host read there (`listed_source`).

#include "lib/streaming/control.cuh"
#include "lib/streaming/window.cuh"

REPACK_BINDING(GateBinding, SEISMIC_CONVERSION_SG_UG, SEISMIC_CONVEYOR_GATE, SEISMIC_GATE, SEISMIC_DIM_F, SEISMIC_DIM_H)
REPACK_BINDING(UpBinding, SEISMIC_CONVERSION_SU_UU, SEISMIC_CONVEYOR_UP, SEISMIC_UP, SEISMIC_DIM_F, SEISMIC_DIM_H)
REPACK_BINDING(DownBinding, SEISMIC_CONVERSION_SD_UD, SEISMIC_CONVEYOR_DOWN, SEISMIC_DOWN, SEISMIC_DIM_H, SEISMIC_DIM_F)

#include "lib/streaming/land.cuh"
#include "../streaming/round.hpp"

#ifdef SEISMIC_FORMING_ROUTED_ROUND
__device__ void routed_round_block(SEISMIC_KERNEL_PARAMS) {
    __shared__ streaming::LandShared shared;
    const u64 q =
        streaming_step(reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_STEP)), SEISMIC_STEP_STRIDE_0);
    int *slot = streaming_round_lane(q, (u64)SEISMIC_PARAM_ROUND,
                                     reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH)), seismic_words_value);
    if (threadIdx.x == 0)
        streaming_round_arrive(slot, SEISMIC_SCRATCH_STRIDE_1, q);
    streaming::Lander land{{SEISMIC_PTR(SEISMIC_BUFFER_CONVEYOR_GATE), SEISMIC_PTR(SEISMIC_BUFFER_CONVEYOR_UP),
                            SEISMIC_PTR(SEISMIC_BUFFER_CONVEYOR_DOWN)},
                           {SEISMIC_PTR(SEISMIC_BUFFER_GATE), SEISMIC_PTR(SEISMIC_BUFFER_UP),
                            SEISMIC_PTR(SEISMIC_BUFFER_DOWN)},
                           SEISMIC_DIM_GV > 0, seismic_words_value, shared, {}, {}, 0};
    streaming_land(land, reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_LANDING)), SEISMIC_LANDING_STRIDE_0, slot,
                   SEISMIC_SCRATCH_STRIDE_1, q, reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_COUNTERS)),
                   (u32)SEISMIC_DIM_L, gridDim.x * gridDim.y * gridDim.z, shared.words.group, threadIdx.x);
}
#endif
