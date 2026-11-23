// The device's share of a streamed layer's general decode combine
// (`routed_down_project` in routed.seismic): `routed_down`'s projections
// (`lib/routed/down.cuh`), published into the result for
// `routed_down_sited`.
#define KERNEL_A SEISMIC_ELEMENT_A
#define ROUTED_SOURCE_ROWS SEISMIC_CONST_SOURCE_ROWS
#define ROUTED_PROJECTED_OUT SEISMIC_RESULT_0_OUT
#define ROUTED_PROJECTED_S0 SEISMIC_RESULT_0_STRIDE_0
#define ROUTED_PROJECTED_S1 SEISMIC_RESULT_0_STRIDE_1
#define ROUTED_PROJECTED_S2 SEISMIC_RESULT_0_STRIDE_2
#include "lib/routed/down.cuh"

#ifdef SEISMIC_FORMING_ROUTED_DOWN_PROJECT_STAGE
__global__ void routed_down_project_stage(SEISMIC_KERNEL_PARAMS) { ROUTED_DOWN_STAGE_BODY }
#endif

#ifdef SEISMIC_FORMING_ROUTED_DOWN_PROJECT_SUMMED
template <int TPW, int KSPLIT>
__global__ void routed_down_project_summed(SEISMIC_KERNEL_PARAMS) { ROUTED_DOWN_SUMMED_BODY(TPW, KSPLIT) }
#endif

#ifdef SEISMIC_FORMING_ROUTED_DOWN_PROJECT_GATHERED
template <int KSPLIT, int GATHER_TPW>
__global__ void routed_down_project_gathered(SEISMIC_KERNEL_PARAMS) { ROUTED_DOWN_GATHERED_BODY(GATHER_TPW, KSPLIT) }
#endif
