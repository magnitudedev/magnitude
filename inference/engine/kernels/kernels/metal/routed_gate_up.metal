// Decode expansion of gated experts (M <= VERIFY_ROWS): threadgroup row y = choice
// (m, k), a K1 paired GEMV over the feature tile `x` of expert routes[m, k]'s
// gate and up rows (its record, `routed::decode_source`: a bank record, or a
// host tier record read in place; a resident layer's, its experts' records)
// with the activation-generic GLU epilogue, the accumulators scaled by the expert's
// `gate_scale`/`up_scale` where those ports are present (static GS, US).
// Every row of a choice with no record is zero.

#define KERNEL_A SEISMIC_ELEMENT_A
#if defined(SEISMIC_FORMING_ROUTED_GATE_UP_RESIDENT) || defined(SEISMIC_FORMING_ROUTED_GATE_UP_GATHERED_RESIDENT)
// A resident layer's: nothing lands in its records while it runs, so plain
// loads read them.
#define PACKETS_CONFIG_PLAIN_RECORDS
#endif
#define KERNEL_W0 SEISMIC_EXPERT_GATE
#define KERNEL_W1 SEISMIC_EXPERT_UP
// The expansion's limit as its F32 bits LB (0: none), which the Glu epilogue
// clamps at.
#if SEISMIC_STATIC_LB != 0
#define PROJECTION_GLU_LIMIT SEISMIC_STATIC_LB
#endif
#define ROUTED_SOURCE_ROWS SEISMIC_DIM_XS
#define ROUTED_GATE_UP_PRODUCT_STRIDE_0 SEISMIC_RESULT_0_STRIDE_0
#define ROUTED_GATE_UP_PRODUCT_STRIDE_1 SEISMIC_RESULT_0_STRIDE_1
#define ROUTED_GATE_UP_PRODUCT_STRIDE_2 SEISMIC_RESULT_0_STRIDE_2
#define ROUTED_GATE_UP_ACTIVE(c, source) true
#include "lib/routed/gate_up.h"

// The ports, the expert stacks': their records' address tables.
#define ROUTED_GATE_UP_PORTS                                                \
    device const uchar *normalized [[buffer(SEISMIC_BUFFER_NORMALIZED)]],   \
    device const int *routes [[buffer(SEISMIC_BUFFER_ROUTES)]],             \
    device const int *sources [[buffer(SEISMIC_BUFFER_SOURCES)]],           \
    device const ulong *expert_gate [[buffer(SEISMIC_BUFFER_EXPERT_GATE)]], \
    device const ulong *expert_up [[buffer(SEISMIC_BUFFER_EXPERT_UP)]],     \
    device const ulong *tier_gate [[buffer(SEISMIC_BUFFER_TIER_GATE)]],     \
    device const ulong *tier_up [[buffer(SEISMIC_BUFFER_TIER_UP)]],         \
    device const float *gate_scale [[buffer(SEISMIC_BUFFER_GATE_SCALE)]],   \
    device const float *up_scale [[buffer(SEISMIC_BUFFER_UP_SCALE)]],       \
    SEISMIC_RESULT_0_PARAM(product_cells),                                  \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],         \
    threadgroup uchar *shared [[threadgroup(0)]],                           \
    uint2 group [[threadgroup_position_in_grid]],                           \
    uint simdgroups [[simdgroups_per_threadgroup]],                         \
    uint sg [[simdgroup_index_in_threadgroup]],                             \
    uint lane [[thread_index_in_simdgroup]]

#ifdef SEISMIC_FORMING_ROUTED_GATE_UP_CHOICE
template <uint ROWS, uint LANES>
kernel void routed_gate_up_choice(ROUTED_GATE_UP_PORTS) {
    ROUTED_GATE_UP_BODY
}
#endif
#ifdef SEISMIC_FORMING_ROUTED_GATE_UP_RESIDENT
// A resident layer's (XS 0), its records read with plain loads.
template <uint ROWS, uint LANES>
kernel void routed_gate_up_resident(ROUTED_GATE_UP_PORTS) {
    ROUTED_GATE_UP_BODY
}
#endif
#ifdef SEISMIC_FORMING_ROUTED_GATE_UP_GATHER
template <uint ROWS, uint LANES>
kernel void routed_gate_up_gather(
    device const int *routes [[buffer(SEISMIC_BUFFER_ROUTES)]],
    device const int *sources [[buffer(SEISMIC_BUFFER_SOURCES)]],
    device routed::Worklist *worklist [[buffer(SEISMIC_BUFFER_SCRATCH_WORKLIST)]],
    device int *gather_grid [[buffer(SEISMIC_BUFFER_SCRATCH_GATHER_GRID)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint lane [[thread_index_in_simdgroup]]) {
    ROUTED_GATE_UP_GATHER_BODY
}
#endif
#ifdef SEISMIC_FORMING_ROUTED_GATE_UP_GATHERED
template <uint ROWS, uint LANES>
kernel void routed_gate_up_gathered(
    ROUTED_GATE_UP_PORTS,
    device const routed::Worklist *worklist [[buffer(SEISMIC_BUFFER_SCRATCH_WORKLIST)]]) {
    ROUTED_GATE_UP_GATHERED_BODY
}
#endif
#ifdef SEISMIC_FORMING_ROUTED_GATE_UP_GATHERED_RESIDENT
// A resident layer's (XS 0), its records read with plain loads.
template <uint ROWS, uint LANES>
kernel void routed_gate_up_gathered_resident(
    ROUTED_GATE_UP_PORTS,
    device const routed::Worklist *worklist [[buffer(SEISMIC_BUFFER_SCRATCH_WORKLIST)]]) {
    ROUTED_GATE_UP_GATHERED_BODY
}
#endif
