// Expert streaming: a streamed decode layer's landing launch (`routed_land`
// in routed.seismic); the CUDA form of `metal/routed_land.metal`, the body
// of one block of the grid its route wrote (`_block`: its blocks where the
// step requested its slot, none otherwise). Where the bank places (V > 0),
// every block converts the layer's file misses from
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
#include "../streaming/site.hpp"

#ifdef SEISMIC_FORMING_ROUTED_LAND
__device__ void routed_land_block(SEISMIC_KERNEL_PARAMS) {
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
    if (blockIdx.x == 0) {
        int *sources = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_SOURCES));
        streaming_tier_wait(step, SEISMIC_STEP_STRIDE_0, file, SEISMIC_FILE_STRIDE_0, SEISMIC_FILE_STRIDE_1,
                            sources, SEISMIC_SOURCES_STRIDE_0,
                            SEISMIC_SOURCES_STRIDE_1, SEISMIC_SOURCES_STRIDE_2, threadIdx.x, blockDim.x,
                            seismic_words_value);
        if (SEISMIC_DIM_SF > 0) {
            __syncthreads();
            const unsigned choices = (unsigned)(SEISMIC_DIM_M * SEISMIC_DIM_K);
            if (streaming_site_any(sources, SEISMIC_SOURCES_STRIDE_0, SEISMIC_SOURCES_STRIDE_1,
                    SEISMIC_SOURCES_STRIDE_2, choices)) {
                const u64 slot = q % (u64)SEISMIC_DIM_SF;
                using A = ELEMENT_OF(SEISMIC_ELEMENT_A);
                const typename A::storage *norm =
                    reinterpret_cast<const typename A::storage *>(SEISMIC_PTR(SEISMIC_BUFFER_NORMALIZED));
                const element::Out<element::F32> rows = SEISMIC_OUT_SITE_ROWS;
                const u64 count = SEISMIC_DIM_M * SEISMIC_DIM_H;
                for (u64 i = threadIdx.x; i < count; i += blockDim.x) {
                    const u64 m = i / SEISMIC_DIM_H, n = i % SEISMIC_DIM_H;
                    rows.put_system(slot * SEISMIC_SITE_ROWS_STRIDE_0 + m * SEISMIC_SITE_ROWS_STRIDE_1
                                        + n * SEISMIC_SITE_ROWS_STRIDE_2,
                                    A::load(norm[m * SEISMIC_NORMALIZED_STRIDE_0 + n * SEISMIC_NORMALIZED_STRIDE_1]));
                }
                streaming_publish_barrier();
                if (threadIdx.x == 0) {
                    streaming_system_fence();
                    streaming_site_publish(q, reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES)),
                        SEISMIC_ROUTES_STRIDE_0, SEISMIC_ROUTES_STRIDE_1,
                        reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_WEIGHTS)), SEISMIC_WEIGHTS_STRIDE_0,
                        SEISMIC_WEIGHTS_STRIDE_1, sources, SEISMIC_SOURCES_STRIDE_0, SEISMIC_SOURCES_STRIDE_1,
                        SEISMIC_SOURCES_STRIDE_2, choices,
                        reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_SITE)) + slot * SEISMIC_SITE_STRIDE_0,
                        SEISMIC_SITE_STRIDE_1);
                }
            }
        }
    }
}
#endif
