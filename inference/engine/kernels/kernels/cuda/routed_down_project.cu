// The device's share of a streamed layer's general decode combine
// (`routed_down_project` in routed.seismic): `routed_down`'s projections
// (`lib/routed/down.cuh`), published into the result for
// `routed_down_sited`.
#define KERNEL_A SEISMIC_ELEMENT_A
#define ROUTED_SOURCE_ROWS SEISMIC_CONST_SOURCE_ROWS
#define ROUTED_PROJECTED_OUT SEISMIC_OUT_PUBLISHED
#define ROUTED_PROJECTED_S0 SEISMIC_PUBLISHED_STRIDE_0
#define ROUTED_PROJECTED_S1 SEISMIC_PUBLISHED_STRIDE_1
#define ROUTED_PROJECTED_S2 SEISMIC_PUBLISHED_STRIDE_2
#define ROUTED_DOWN_MODE ((int)SEISMIC_PARAM_MODE)
#include "lib/streaming/control.cuh"

// `ROUTING_SOURCE` is written by the layer's route kernel earlier in the same
// CUDA stream and never modified by the host, so L2-cached `__ldcg` reads it
// coherently without serializing uncached system-scope `volatile` loads.
__device__ __forceinline__ bool routed_down_left_absent(const int *step, unsigned long long step0,
                                                        const int *routing, unsigned long long routing0,
                                                        unsigned long long routing1, unsigned flight,
                                                        unsigned width, unsigned choice) {
    const unsigned long long q = streaming_step(step, step0);
    const unsigned long long at = (q % (unsigned long long)flight) * routing0 +
                                  (unsigned long long)SEISMIC_CONST_ROUTING_SOURCE(width, choice) * routing1;
    return __ldcg(routing + at) == -1;
}

#define ROUTED_DOWN_LEFT_ABSENT(c)                                                                                    \
    routed_down_left_absent(reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_STEP)), SEISMIC_STEP_STRIDE_0,   \
                            reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTING)),                       \
                            SEISMIC_ROUTING_STRIDE_0, SEISMIC_ROUTING_STRIDE_1, SEISMIC_DIM_FLIGHT, SEISMIC_DIM_P,    \
                            (c))
#include "lib/routed/down.cuh"

#ifdef SEISMIC_FORMING_ROUTED_DOWN_PROJECT_STAGE
__global__ void routed_down_project_stage(SEISMIC_KERNEL_PARAMS) { ROUTED_DOWN_STAGE_BODY }
#endif

#ifdef SEISMIC_FORMING_ROUTED_DOWN_PROJECT_SUMMED
template <int TPW, int KSPLIT>
__launch_bounds__(routed::DECODE_THREADS, routed::decode_min_blocks<1, packets::W0>())
__global__ void routed_down_project_summed(SEISMIC_KERNEL_PARAMS) { ROUTED_DOWN_SUMMED_BODY(TPW, KSPLIT) }
#endif

#ifdef SEISMIC_FORMING_ROUTED_DOWN_PROJECT_GATHERED
template <int KSPLIT, int GATHER_TPW>
__launch_bounds__(routed::DECODE_THREADS, routed::decode_min_blocks<1, packets::W0>())
__global__ void routed_down_project_gathered(SEISMIC_KERNEL_PARAMS) { ROUTED_DOWN_GATHERED_BODY(GATHER_TPW, KSPLIT, 1) }
#endif

#ifdef SEISMIC_FORMING_ROUTED_DOWN_PROJECT_GATHERED16
template <int KSPLIT, int GATHER_TPW>
__launch_bounds__(routed::DECODE_THREADS, routed::decode_min_blocks<routed::DECODE_BATCHES, packets::W0>())
__global__ void routed_down_project_gathered16(SEISMIC_KERNEL_PARAMS) {
    ROUTED_DOWN_GATHERED_BODY(GATHER_TPW, KSPLIT, routed::DECODE_BATCHES)
}
#endif
