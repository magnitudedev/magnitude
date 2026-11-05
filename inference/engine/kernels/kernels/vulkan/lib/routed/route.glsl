// Routing for M rows in four launches. The Vulkan form of
// `metal/routed_route.metal`.
//   routed_route_stage (M > DECODE_ROWS, one workgroup per row): the row's RMS and
//     its normalized values (rounded to the activation dtype), published once.
//   routed_route_gemv (M <= DECODE_ROWS): one subgroup per logit column reads the
//     column's weights once and dots them with every row, normalizing on the
//     fly from RMS partials shared across the workgroup (in the stage
//     launch's sum order); workgroup 0 publishes the normalized rows.
//   routed_route_gemm (M > DECODE_ROWS): 32 rows x 32 columns per workgroup over
//     staged F32 tiles, each invocation 8 outputs in ascending-K FMAs.
//   Logits are [M, E + 1]: column E is the shared-expert gate.
//   routed_route_select (one subgroup per row): the softmax over every
//     expert (ceil(E / 32) experts per lane), K rounds of subgroup argmax
//     (ties to the higher expert index), the optional renormalization, each
//     score times its expert's `expert_scale` entry where that port is
//     present (static ES), and the shared-expert coefficient from column E.
// Every logit accumulates over the hidden axis in the same order for every
// expert, so experts with equal router rows tie exactly.
#define KERNEL_W0 SEISMIC_ROUTER
#include "../core/activation.glsl"
#include "../core/reduce.glsl"
#include <seismic/packets.glsl>
#ifdef ROUTE_PROJECTION
#include "routed.glsl"
#endif
#include "ranked.glsl"

#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_BF16) || defined(SEISMIC_ELEMENT_A_REPRESENTATION_F16) \
    || defined(SEISMIC_ELEMENT_A_REPRESENTATION_F32)
#define ROUTE_ACT ELEMENT_ACT
#else
#error "routed activation must be dense"
#endif
#if defined(SEISMIC_NORM_REPRESENTATION_F32)
#define ROUTE_NORM ELEMENT_F32
#elif defined(SEISMIC_NORM_REPRESENTATION_F16)
#define ROUTE_NORM ELEMENT_F16
#elif defined(SEISMIC_NORM_REPRESENTATION_BF16)
#define ROUTE_NORM ELEMENT_BF16
#else
#error "routed norm must be dense"
#endif
#if defined(SEISMIC_ROUTER_KIND_PACKED) && !defined(SEISMIC_ROUTER_LAYOUT_ROWS16)
#error "packed routed router requires rows16"
#endif

#define ROUTE_THREADS 256u
#define ROUTE_SUBGROUPS 8u
#define ROUTE_K uint(SEISMIC_DIM_K)

#define ROUTE_FN(name) name
#define ROUTE_E uint(SEISMIC_DIM_E)
#define ROUTE_NORM_BUFFER SEISMIC_PTR(SEISMIC_BUFFER_NORM)
#define ROUTE_NORM_STRIDE SEISMIC_NORM_STRIDE_0
#define ROUTE_ROUTER_BUFFER SEISMIC_PTR(SEISMIC_BUFFER_ROUTER)
#define ROUTE_ROUTER_STRIDE_0 SEISMIC_ROUTER_STRIDE_0
#define ROUTE_ROUTER_STRIDE_1 SEISMIC_ROUTER_STRIDE_1
#define ROUTE_ROUTER_ROWS PACKETS_W0_ROWS
#define ROUTE_LOGITS SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_LOGITS)
#define ROUTE_PUBLISH true
#include "route_body.glsl"

ROUTE_NUMERICAL
ROUTE_OUTPUT
