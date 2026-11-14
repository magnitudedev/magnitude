// The device's share of a streamed layer of Qwen's fused decode output
// (`routed_output_project` in routed.seismic): `routed_output`'s projections
// (`lib/routed/output.cuh`), published into the result for
// `routed_output_sited`; no block joins.
#define ROUTED_SOURCE_ROWS SEISMIC_CONST_SOURCE_ROWS
#define ROUTED_PUBLISHED_OUT SEISMIC_RESULT_0_OUT
#define ROUTED_OUTPUT_ARRIVED(m, blocks, arrival, first, channels)
#define ROUTED_OUTPUT_GATHERED_ARRIVED(blocks, arrival, first, channels)
#include "lib/routed/output.cuh"
