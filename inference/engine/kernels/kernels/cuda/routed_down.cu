// Decode output of the general routed feed-forward (M <= DECODE_ROWS): the
// down projections (`lib/routed/down.cuh`) into scratch and their sum
// (`lib/routed/down_sum.cuh`), without a CPU site.
#define ROUTED_SOURCE_ROWS SEISMIC_DIM_XS
#define ROUTED_PROJECTED reinterpret_cast<element::Act::storage *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PROJECTED))
#define ROUTED_SITE_AWAIT(m)
#define ROUTED_SITE_ADD(m, n, selected)
#include "lib/routed/down.cuh"
#include "lib/routed/down_sum.cuh"

#ifdef SEISMIC_FORMING_ROUTED_DOWN_COMBINE
__global__ void routed_down_combine(SEISMIC_KERNEL_PARAMS) { ROUTED_DOWN_COMBINE_BODY }
#endif
