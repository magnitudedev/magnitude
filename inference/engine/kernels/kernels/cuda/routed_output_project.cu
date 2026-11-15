// The device's share of a streamed layer of Qwen's fused decode output
// (`routed_output_project` in routed.seismic): `routed_output`'s projections
// (`lib/routed/output.cuh`), published into the result for
// `routed_output_sited`; no block joins.
#define ROUTED_SOURCE_ROWS SEISMIC_CONST_SOURCE_ROWS
#define ROUTED_PUBLISHED_OUT SEISMIC_RESULT_0_OUT
#define ROUTED_PUBLISHED_S0 SEISMIC_RESULT_0_STRIDE_0
#define ROUTED_PUBLISHED_S1 SEISMIC_RESULT_0_STRIDE_1
#define ROUTED_PUBLISHED_S2 SEISMIC_RESULT_0_STRIDE_2
#define ROUTED_OUTPUT_ARRIVED(m, blocks, arrival, first, channels)
#define ROUTED_OUTPUT_GATHERED_ARRIVED(blocks, arrival, first, channels)
#include "lib/routed/output.cuh"

#ifdef SEISMIC_FORMING_ROUTED_OUTPUT_PROJECT_STAGE
__global__ void routed_output_project_stage(SEISMIC_KERNEL_PARAMS) { ROUTED_OUTPUT_STAGE_BODY }
#endif

#ifdef SEISMIC_FORMING_ROUTED_OUTPUT_PROJECT_SUMMED
template <int TPW, int KSPLIT>
__global__ void routed_output_project_summed(SEISMIC_KERNEL_PARAMS) { ROUTED_OUTPUT_SUMMED_BODY(TPW, KSPLIT) }
#endif

#ifdef SEISMIC_FORMING_ROUTED_OUTPUT_PROJECT_GATHERED
template <int TPW, int KSPLIT>
__global__ void routed_output_project_gathered(SEISMIC_KERNEL_PARAMS) { ROUTED_OUTPUT_GATHERED_BODY(TPW, KSPLIT) }
#endif
