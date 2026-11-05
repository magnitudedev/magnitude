// The device's share of a streamed layer's general decode combine
// (`routed_down_project` in routed.seismic): `routed_down`'s projections
// (`lib/routed/down.cuh`), published into the result for
// `routed_down_sited`.
#define ROUTED_SOURCE_ROWS SEISMIC_CONST_SOURCE_ROWS
#define ROUTED_PROJECTED reinterpret_cast<element::Act::storage *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER))
#include "lib/routed/down.cuh"
