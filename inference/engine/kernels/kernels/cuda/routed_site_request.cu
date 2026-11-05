// Expert streaming, a streamed decode layer's request to the CPU
// (`routed_site_request` in routed.seismic); the CUDA form of
// `metal/routed_site_request.metal`: when a choice of the step is the
// CPU's, the block's threads widen the normalized rows into the step's
// slot's F32 rows (mapped host memory) through volatile stores, then thread
// 0 publishes the request in the slot (`streaming/site.hpp`).

#include "lib/streaming/control.cuh"
#include "../streaming/site.hpp"

extern "C" __global__ void routed_site_request(SEISMIC_KERNEL_PARAMS) {
    const int *sources = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_SOURCES));
    const unsigned choices = (unsigned)(SEISMIC_DIM_M * SEISMIC_DIM_K);
    if (!streaming_site_any(sources, SEISMIC_SOURCES_STRIDE_0, SEISMIC_SOURCES_STRIDE_1, SEISMIC_SOURCES_STRIDE_2,
            choices))
        return;
    const u64 q = streaming_step(reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_STEP)), SEISMIC_STEP_STRIDE_0);
    const u64 slot = q % (u64)SEISMIC_DIM_FLIGHT;
    using A = ELEMENT_OF(SEISMIC_ELEMENT_A);
    const typename A::storage *normalized =
        reinterpret_cast<const typename A::storage *>(SEISMIC_PTR(SEISMIC_BUFFER_NORMALIZED));
    volatile float *rows = reinterpret_cast<volatile float *>(SEISMIC_PTR(SEISMIC_BUFFER_SITE_ROWS));
    const u64 count = SEISMIC_DIM_M * SEISMIC_DIM_H;
    for (u64 i = threadIdx.x; i < count; i += blockDim.x) {
        const u64 m = i / SEISMIC_DIM_H, n = i % SEISMIC_DIM_H;
        rows[slot * SEISMIC_SITE_ROWS_STRIDE_0 + m * SEISMIC_SITE_ROWS_STRIDE_1 +
             n * SEISMIC_SITE_ROWS_STRIDE_2] =
            A::load(normalized[m * SEISMIC_NORMALIZED_STRIDE_0 + n * SEISMIC_NORMALIZED_STRIDE_1]);
    }
    streaming_publish_barrier();
    if (threadIdx.x != 0)
        return;
    streaming_system_fence();
    streaming_site_publish(q, reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES)),
        SEISMIC_ROUTES_STRIDE_0, SEISMIC_ROUTES_STRIDE_1,
        reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_WEIGHTS)), SEISMIC_WEIGHTS_STRIDE_0,
        SEISMIC_WEIGHTS_STRIDE_1, sources, SEISMIC_SOURCES_STRIDE_0, SEISMIC_SOURCES_STRIDE_1,
        SEISMIC_SOURCES_STRIDE_2, choices,
        reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_SITE)) + slot * SEISMIC_SITE_STRIDE_0,
        SEISMIC_SITE_STRIDE_1);
}
