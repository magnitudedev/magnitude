// The CPU's join of a streamed layer's general decode combine
// (`routed_down_sited` in routed.seismic): `routed_down`'s sum
// (`lib/routed/down_sum.cuh`) over the published projections of
// `routed_down_project`, its block waiting for the CPU's answer
// (`lib/routed/sited.cuh`) before the slot-order sum.
#include "lib/routed/routed.cuh"
#include "lib/routed/sited.cuh"
#define ROUTED_PROJECTED reinterpret_cast<const element::Act::storage *>(SEISMIC_PTR(SEISMIC_BUFFER_PUBLISHED))
#include "lib/routed/down_sum.cuh"

extern "C" __global__ void routed_down_combine(SEISMIC_KERNEL_PARAMS) { ROUTED_DOWN_COMBINE_BODY }
