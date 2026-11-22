// The CPU's join of a streamed layer of Qwen's fused decode output
// (`routed_output_sited` in routed.seismic): `routed_output`'s join
// (`lib/routed/output_sum.cuh`), 4 contiguous channels a thread over the
// published projections of `routed_output_project`, its block waiting for
// the CPU's answer (`lib/routed/sited.cuh`) before it joins.
#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/routed/routed.cuh"
#include "lib/routed/sited.cuh"
#define ROUTED_PUBLISHED reinterpret_cast<const activation::Act::storage *>(SEISMIC_PTR(SEISMIC_BUFFER_PUBLISHED))
#include "lib/routed/output_sum.cuh"

// Thread (t, m): row m's 4 contiguous channels `4*t .. 4*t+3` (scalar tail
// when `n + 4 > H`).
extern "C" __global__ void routed_output_combine(SEISMIC_KERNEL_PARAMS) {
    const projection::u64 n = ((projection::u64)blockIdx.x * blockDim.x + threadIdx.x) * 4ull;
    const projection::u64 m = blockIdx.y;
    ROUTED_SITE_AWAIT(m)
    if (n >= SEISMIC_DIM_H)
        return;
    const activation::Act::storage *pub = ROUTED_PUBLISHED + m * (SEISMIC_DIM_K + 1) * SEISMIC_DIM_H;
    if (n + 4ull <= SEISMIC_DIM_H) {
        ROUTED_OUTPUT_JOIN4(m, n, pub);
    } else {
        for (projection::u64 col = n; col < SEISMIC_DIM_H; ++col)
            ROUTED_OUTPUT_JOIN(m, col, pub);
    }
}
