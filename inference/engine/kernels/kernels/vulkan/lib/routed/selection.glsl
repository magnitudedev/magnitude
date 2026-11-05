// General expert selection for M rows in four launches. The Vulkan form of
// `metal/routed_select.metal`.
//   routed_select_stage (M > DECODE_ROWS, one workgroup per row): the row's RMS, the
//     normalized row (norm) published and the router input row (router_norm)
//     into scratch.
//   routed_select_gemv (M <= DECODE_ROWS): one subgroup per logit column reads the
//     column's weights once and dots them with every row, forming the router
//     input on the fly from RMS partials shared across the workgroup (in the
//     stage launch's sum order); workgroup 0 publishes the normalized rows.
//   routed_select_gemm (M > DECODE_ROWS): 32 rows x 32 columns per workgroup over
//     staged F32 tiles of the router input, each invocation 8 outputs in
//     ascending-K FMAs.
//   routed_select_select (one subgroup per row): scores, biased ranking,
//     single-pass top-K (`lib/routed/select.glsl`), normalization, scales.
// Every logit accumulates over the hidden axis in the same order for every
// expert, so experts with equal router rows tie exactly.
#define KERNEL_W0 SEISMIC_ROUTER
#include "../core/activation.glsl"
#include "../core/reduce.glsl"
#include <seismic/packets.glsl>

#include "../core/precise.glsl"
#include "ranked.glsl"

#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_BF16) || defined(SEISMIC_ELEMENT_A_REPRESENTATION_F16) \
    || defined(SEISMIC_ELEMENT_A_REPRESENTATION_F32)
#define SELECT_ACT ELEMENT_ACT
#else
#error "routed activation must be dense"
#endif
#if defined(SEISMIC_NORM_REPRESENTATION_F32)
#define SELECT_NORM ELEMENT_F32
#elif defined(SEISMIC_NORM_REPRESENTATION_F16)
#define SELECT_NORM ELEMENT_F16
#elif defined(SEISMIC_NORM_REPRESENTATION_BF16)
#define SELECT_NORM ELEMENT_BF16
#else
#error "routed norm must be dense"
#endif
#if defined(SEISMIC_ROUTER_NORM_REPRESENTATION_F32)
#define SELECT_ROUTER_NORM ELEMENT_F32
#elif defined(SEISMIC_ROUTER_NORM_REPRESENTATION_F16)
#define SELECT_ROUTER_NORM ELEMENT_F16
#elif defined(SEISMIC_ROUTER_NORM_REPRESENTATION_BF16)
#define SELECT_ROUTER_NORM ELEMENT_BF16
#else
#error "routed router norm must be dense"
#endif
#if defined(SEISMIC_ROUTER_KIND_PACKED) && !defined(SEISMIC_ROUTER_LAYOUT_ROWS16)
#error "packed routed router requires rows16"
#endif

#define SELECT_THREADS 256u
#define SELECT_SUBGROUPS 8u
#define SELECT_K uint(SEISMIC_DIM_K)

#define SELECT_FN(name) name
#define SELECT_E uint(SEISMIC_DIM_E)
#define SELECT_ROUTER_NORM_BUFFER SEISMIC_PTR(SEISMIC_BUFFER_ROUTER_NORM)
#define SELECT_ROUTER_NORM_STRIDE SEISMIC_ROUTER_NORM_STRIDE_0
#define SELECT_ROUTER_BUFFER SEISMIC_PTR(SEISMIC_BUFFER_ROUTER)
#define SELECT_ROUTER_STRIDE_0 SEISMIC_ROUTER_STRIDE_0
#define SELECT_ROUTER_STRIDE_1 SEISMIC_ROUTER_STRIDE_1
#define SELECT_ROUTER_ROWS PACKETS_W0_ROWS
#define SELECT_LOGITS SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_LOGITS)
#define SELECT_BIAS_BUFFER SEISMIC_PTR(SEISMIC_BUFFER_BIAS)
#define SELECT_BIAS_STRIDE SEISMIC_BIAS_STRIDE_0
#define SELECT_PUBLISH true
#include "selection_body.glsl"

SELECT_RANKING
SELECT_NUMERICAL
SELECT_OUTPUT
