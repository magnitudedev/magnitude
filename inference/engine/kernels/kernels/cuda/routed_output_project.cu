// The device's share of a streamed layer of Qwen's fused decode output
// (`routed_output_project` in routed.seismic): `routed_output`'s projections
// (`lib/routed/output.cuh`), published into the result for
// `routed_output_sited`; no block joins.
#define ROUTED_SOURCE_ROWS SEISMIC_CONST_SOURCE_ROWS
#define ROUTED_PUBLISHED reinterpret_cast<element::Act::storage *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER))
#define ROUTED_OUTPUT_ARRIVED(m, blocks, arrival, first, channels, published)
#define ROUTED_OUTPUT_GATHERED_ARRIVED(blocks, arrival, first, channels, published)
#include "lib/routed/output.cuh"
