// Expert streaming: a streamed decode layer's landing launch (`routed_land`
// in routed.seismic); the CUDA form of `metal/routed_land.metal`. Where the
// bank places (V > 0), every block converts the layer's file misses from
// staging into the bank's records as their reads land (streaming/land.hpp
// over `lib/streaming/land.cuh`), its thread 0 spinning on the host's count
// of the step's slot of the layer's file table; then block 0 writes the
// absent choices' sources (streaming/tier_wait.hpp), the slot answered by
// then. Where the tier places, the one block only sources.

#include "lib/streaming/control.cuh"
#include "lib/streaming/window.cuh"

REPACK_BINDING(GateBinding, SEISMIC_CONVERSION_SG_UG, SEISMIC_STAGING_GATE, SEISMIC_GATE, SEISMIC_DIM_F, SEISMIC_DIM_H)
REPACK_BINDING(UpBinding, SEISMIC_CONVERSION_SU_UU, SEISMIC_STAGING_UP, SEISMIC_UP, SEISMIC_DIM_F, SEISMIC_DIM_H)
REPACK_BINDING(DownBinding, SEISMIC_CONVERSION_SD_UD, SEISMIC_STAGING_DOWN, SEISMIC_DOWN, SEISMIC_DIM_H, SEISMIC_DIM_F)

#include "lib/streaming/land.cuh"
#include "../streaming/tier_wait.hpp"

extern "C" __global__ void routed_land(SEISMIC_KERNEL_PARAMS) {
    const int *step = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_STEP));
    int *file = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_FILE));
    const u64 q = streaming_step(step, SEISMIC_STEP_STRIDE_0);
    if (SEISMIC_DIM_V > 0) {
        __shared__ streaming::LandShared shared;
        streaming::Lander land{{SEISMIC_PTR(SEISMIC_BUFFER_STAGING_GATE), SEISMIC_PTR(SEISMIC_BUFFER_STAGING_UP),
                                SEISMIC_PTR(SEISMIC_BUFFER_STAGING_DOWN)},
                               {SEISMIC_PTR(SEISMIC_BUFFER_GATE), SEISMIC_PTR(SEISMIC_BUFFER_UP),
                                SEISMIC_PTR(SEISMIC_BUFFER_DOWN)},
                               SEISMIC_DIM_GV > 0, seismic_words_value, shared, {}, {}, 0};
        streaming_land(land, reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_LANDING)), SEISMIC_LANDING_STRIDE_0,
                       file + (q % (u64)SEISMIC_DIM_FLIGHT) * SEISMIC_FILE_STRIDE_0, SEISMIC_FILE_STRIDE_1, q,
                       reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_COUNTERS)), (u32)SEISMIC_DIM_L,
                       gridDim.x * gridDim.y * gridDim.z, shared.words.group, threadIdx.x);
    }
    if (blockIdx.x == 0)
        streaming_tier_wait(step, SEISMIC_STEP_STRIDE_0, file, SEISMIC_FILE_STRIDE_0, SEISMIC_FILE_STRIDE_1,
                            reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_SOURCES)), SEISMIC_SOURCES_STRIDE_0,
                            SEISMIC_SOURCES_STRIDE_1, SEISMIC_SOURCES_STRIDE_2, threadIdx.x, blockDim.x,
                            seismic_words_value);
}
