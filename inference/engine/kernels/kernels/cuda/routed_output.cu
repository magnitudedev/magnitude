// Decode output (M <= VERIFY_ROWS): the fused projections
// (`lib/routed/output.cuh`) into scratch, the last arriving block of a
// channel tile joining them (`lib/routed/output_sum.cuh`), without a CPU
// site.
#define KERNEL_A SEISMIC_ELEMENT_A
#define ROUTED_SOURCE_ROWS SEISMIC_DIM_XS
#define ROUTED_PUBLISHED_OUT element::out_over_scratch<activation::Act>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PUBLISHED))
#define ROUTED_PUBLISHED reinterpret_cast<const activation::Act::storage *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PUBLISHED))
#define ROUTED_SITE_AWAIT(m)
#define ROUTED_SITE_ADD(m, n, selected)
#define ROUTED_ARRIVALS reinterpret_cast<unsigned *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_ARRIVALS))
#define ROUTED_OUTPUT_ARRIVED(m, blocks, arrival, first, channels) \
    ROUTED_OUTPUT_ARRIVED_JOIN(m, blocks, arrival, first, channels)
#define ROUTED_OUTPUT_GATHERED_ARRIVED(blocks, arrival, first, channels) \
    ROUTED_OUTPUT_GATHERED_ARRIVED_JOIN(blocks, arrival, first, channels)
#include "lib/core/arrive.cuh"
#include "lib/routed/output_sum.cuh"
#include "lib/routed/output.cuh"

#ifdef SEISMIC_FORMING_ROUTED_OUTPUT_STAGE
__global__ void routed_output_stage(SEISMIC_KERNEL_PARAMS) { ROUTED_OUTPUT_STAGE_BODY }
#endif

#ifdef SEISMIC_FORMING_ROUTED_OUTPUT_SUMMED
template <int TPW, int KSPLIT>
__global__ void routed_output_summed(SEISMIC_KERNEL_PARAMS) { ROUTED_OUTPUT_SUMMED_BODY(TPW, KSPLIT) }
#endif

#ifdef SEISMIC_FORMING_ROUTED_OUTPUT_GATHERED
template <int KSPLIT, int GATHER_TPW>
__global__ void routed_output_gathered(SEISMIC_KERNEL_PARAMS) { ROUTED_OUTPUT_GATHERED_BODY(GATHER_TPW, KSPLIT) }
#endif
