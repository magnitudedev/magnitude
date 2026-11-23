// Single-launch streamed decode output (M <= VERIFY_ROWS): the fused
// projections (`lib/routed/output.cuh`) into scratch, the last arriving block
// of a channel tile awaiting the CPU site only when `streaming_site_any` is
// true (`lib/routed/sited.cuh`) and joining them (`lib/routed/output_sum.cuh`).
#define KERNEL_A SEISMIC_ELEMENT_A
#define ROUTED_SOURCE_ROWS SEISMIC_CONST_SOURCE_ROWS
#define ROUTED_PUBLISHED_OUT element::out_over_scratch<activation::Act>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PUBLISHED))
#define ROUTED_PUBLISHED reinterpret_cast<const activation::Act::storage *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PUBLISHED))
#define ROUTED_ARRIVALS reinterpret_cast<unsigned *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_ARRIVALS))
#define ROUTED_OUTPUT_ARRIVED(m, blocks, arrival, first, channels) \
    ROUTED_OUTPUT_ARRIVED_JOIN(m, blocks, arrival, first, channels)
#define ROUTED_OUTPUT_GATHERED_ARRIVED(blocks, arrival, first, channels) \
    ROUTED_OUTPUT_GATHERED_ARRIVED_JOIN(blocks, arrival, first, channels)
#include "lib/core/arrive.cuh"
#include "lib/routed/sited.cuh"
#include "lib/routed/output_sum.cuh"
#include "lib/routed/output.cuh"

#ifdef SEISMIC_FORMING_ROUTED_OUTPUT_STREAMED_STAGE
__global__ void routed_output_streamed_stage(SEISMIC_KERNEL_PARAMS) { ROUTED_OUTPUT_STAGE_BODY }
#endif

#ifdef SEISMIC_FORMING_ROUTED_OUTPUT_STREAMED_SUMMED
template <int TPW, int KSPLIT>
__global__ void routed_output_streamed_summed(SEISMIC_KERNEL_PARAMS) { ROUTED_OUTPUT_SUMMED_BODY(TPW, KSPLIT) }
#endif

#ifdef SEISMIC_FORMING_ROUTED_OUTPUT_STREAMED_GATHERED
template <int KSPLIT, int GATHER_TPW>
__global__ void routed_output_streamed_gathered(SEISMIC_KERNEL_PARAMS) { ROUTED_OUTPUT_GATHERED_BODY(GATHER_TPW, KSPLIT) }
#endif
