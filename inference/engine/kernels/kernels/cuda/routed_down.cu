// Decode output of the general routed feed-forward (M <= VERIFY_ROWS): the
// down projections (`lib/routed/down.cuh`) into scratch and their sum
// (`lib/routed/down_sum.cuh`), without a CPU site.
#define KERNEL_A SEISMIC_ELEMENT_A
#define ROUTED_SOURCE_ROWS SEISMIC_DIM_XS
#define ROUTED_PROJECTED_OUT element::out_over_scratch<activation::Act>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PUBLISHED))
#define ROUTED_PROJECTED reinterpret_cast<const activation::Act::storage *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PUBLISHED))
#define ROUTED_SITE_AWAIT(m)
#define ROUTED_SITE_ADD(m, n, selected)
#include "lib/routed/down.cuh"
#include "lib/routed/down_sum.cuh"

#ifdef SEISMIC_FORMING_ROUTED_DOWN_STAGE
__global__ void routed_down_stage(SEISMIC_KERNEL_PARAMS) { ROUTED_DOWN_STAGE_BODY }
#endif

#ifdef SEISMIC_FORMING_ROUTED_DOWN_SUMMED
template <int TPW, int KSPLIT>
__global__ void routed_down_summed(SEISMIC_KERNEL_PARAMS) { ROUTED_DOWN_SUMMED_BODY(TPW, KSPLIT) }
#endif

#ifdef SEISMIC_FORMING_ROUTED_DOWN_GATHERED
template <int TPW, int KSPLIT>
__global__ void routed_down_gathered(SEISMIC_KERNEL_PARAMS) { ROUTED_DOWN_GATHERED_BODY(TPW, KSPLIT) }
#endif

#ifdef SEISMIC_FORMING_ROUTED_DOWN_COMBINE
__global__ void routed_down_combine(SEISMIC_KERNEL_PARAMS) { ROUTED_DOWN_COMBINE_BODY }
#endif
