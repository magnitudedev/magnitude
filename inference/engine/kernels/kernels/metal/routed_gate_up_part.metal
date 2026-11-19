// `routed_gate_up_part` (routed.seismic): `routed_gate_up`'s streamed launches in two passes over one
// `product`, so the experts a layer holds run while its host lands the absent ones. The early pass
// (`late` 0, before the tier wait) is `routed_gate_up`'s rows into `product`. The late pass (after the
// wait) computes only the choices the route left absent (its routing record's source -1) that the
// wait has since sourced, and leaves every other row as the early pass wrote it. Each choice's
// product is computed once, by the one body (`lib/routed/gate_up.h`), so the sum that follows reads
// the bits of one pass.

#define KERNEL_A SEISMIC_ELEMENT_A
#define KERNEL_W0 SEISMIC_EXPERT_GATE
#define KERNEL_W1 SEISMIC_EXPERT_UP
#if SEISMIC_STATIC_LB != 0
#define PROJECTION_GLU_LIMIT SEISMIC_STATIC_LB
#endif
#include "lib/streaming/control.h"
#define ROUTED_SOURCE_ROWS SEISMIC_CONST_SOURCE_ROWS
#define ROUTED_GATE_UP_PRODUCT_STRIDE_0 SEISMIC_PRODUCT_STRIDE_0
#define ROUTED_GATE_UP_PRODUCT_STRIDE_1 SEISMIC_PRODUCT_STRIDE_1
#define ROUTED_GATE_UP_PRODUCT_STRIDE_2 SEISMIC_PRODUCT_STRIDE_2
// Whether flat choice c runs in this pass: every choice in the early pass
// (`late` 0); in the late pass, only a choice the route left absent
// (`streaming_left_absent`) that the wait has since sourced.
#define ROUTED_GATE_UP_ACTIVE(c, source)                                                                               \
    (int(SEISMIC_PARAM_LATE) == 0                                                                                      \
        || (!(source).none()                                                                                           \
            && streaming_left_absent(step, SEISMIC_STEP_STRIDE_0, routing, SEISMIC_ROUTING_STRIDE_0,                   \
                SEISMIC_ROUTING_STRIDE_1, uint(SEISMIC_DIM_FLIGHT), uint(SEISMIC_DIM_P), (c))))
#include "lib/routed/gate_up.h"

// The ports: the expert stacks' records' address tables, the route's routing ring and step words,
// and the one `product` both passes write.
#define ROUTED_GATE_UP_PART_PORTS                                           \
    device const uchar *normalized [[buffer(SEISMIC_BUFFER_NORMALIZED)]],   \
    device const int *routes [[buffer(SEISMIC_BUFFER_ROUTES)]],             \
    device const int *sources [[buffer(SEISMIC_BUFFER_SOURCES)]],           \
    device const ulong *expert_gate [[buffer(SEISMIC_BUFFER_EXPERT_GATE)]], \
    device const ulong *expert_up [[buffer(SEISMIC_BUFFER_EXPERT_UP)]],     \
    device const ulong *tier_gate [[buffer(SEISMIC_BUFFER_TIER_GATE)]],     \
    device const ulong *tier_up [[buffer(SEISMIC_BUFFER_TIER_UP)]],         \
    device const float *gate_scale [[buffer(SEISMIC_BUFFER_GATE_SCALE)]],   \
    device const float *up_scale [[buffer(SEISMIC_BUFFER_UP_SCALE)]],       \
    SYSTEM const int *routing [[buffer(SEISMIC_BUFFER_ROUTING)]],           \
    device const int *step [[buffer(SEISMIC_BUFFER_STEP)]],                 \
    SEISMIC_OUT_PRODUCT_PARAM(product_cells),                               \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],         \
    threadgroup uchar *shared [[threadgroup(0)]],                           \
    uint2 group [[threadgroup_position_in_grid]],                           \
    uint simdgroups [[simdgroups_per_threadgroup]],                         \
    uint sg [[simdgroup_index_in_threadgroup]],                             \
    uint lane [[thread_index_in_simdgroup]]

#ifdef SEISMIC_FORMING_ROUTED_GATE_UP_PART_CHOICE
template <uint ROWS, uint LANES>
kernel void routed_gate_up_part_choice(ROUTED_GATE_UP_PART_PORTS) {
    ROUTED_GATE_UP_BODY
}
#endif
#ifdef SEISMIC_FORMING_ROUTED_GATE_UP_PART_GATHERED
template <uint ROWS, uint LANES>
kernel void routed_gate_up_part_gathered(ROUTED_GATE_UP_PART_PORTS) {
    ROUTED_GATE_UP_GATHERED_BODY
}
#endif
