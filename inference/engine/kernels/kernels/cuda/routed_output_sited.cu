// The CPU's join of a streamed layer of Qwen's fused decode output
// (`routed_output_sited` in routed.seismic): `routed_output`'s join
// (`lib/routed/output_sum.cuh`) a thread a channel of a row over the
// published projections of `routed_output_project`, its block waiting for
// the CPU's answer (`lib/routed/sited.cuh`) before it joins.
#include "lib/routed/routed.cuh"
#include "lib/routed/sited.cuh"
#define ROUTED_PUBLISHED reinterpret_cast<const element::Act::storage *>(SEISMIC_PTR(SEISMIC_BUFFER_PUBLISHED))
#include "lib/routed/output_sum.cuh"

// Thread (n, m): row m's channel n.
extern "C" __global__ void routed_output_combine(SEISMIC_KERNEL_PARAMS) {
    const projection::u64 n = (projection::u64)blockIdx.x * blockDim.x + threadIdx.x;
    const projection::u64 m = blockIdx.y;
    ROUTED_SITE_AWAIT(m)
    if (n >= SEISMIC_DIM_H)
        return;
    ROUTED_OUTPUT_JOIN(m, n, ROUTED_PUBLISHED + m * (SEISMIC_DIM_K + 1) * SEISMIC_DIM_H);
}
