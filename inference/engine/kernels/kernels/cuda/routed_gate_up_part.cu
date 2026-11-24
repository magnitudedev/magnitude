// `routed_gate_up_part` (routed.seismic) in CUDA: `routed_gate_up`'s forms in two passes over one
// `product`, so the experts a layer holds run while its host lands the absent ones. The early pass
// (`late` 0, before the tier wait) is `routed_gate_up`'s rows into `product`. The late pass (after
// the wait) computes only the choices the route left absent (its routing record's source -1) that
// the wait has since sourced, and leaves every other row as the early pass wrote it. Each choice's
// product is computed once, by the one body (`lib/routed/gate_up.cuh`), so the sum that follows
// reads the bits of one pass.
#define KERNEL_A SEISMIC_ELEMENT_A
#define KERNEL_W0 SEISMIC_EXPERT_GATE
#define KERNEL_W1 SEISMIC_EXPERT_UP
#if SEISMIC_STATIC_LB != 0
#define PROJECTION_GLU_LIMIT SEISMIC_STATIC_LB
#endif
#include "lib/streaming/control.cuh"

#define ROUTED_GATE_UP_SOURCE_ROWS SEISMIC_CONST_SOURCE_ROWS
#define ROUTED_GATE_UP_PRODUCT_OUT SEISMIC_OUT_PRODUCT
#define ROUTED_GATE_UP_PRODUCT_STRIDE_0 SEISMIC_PRODUCT_STRIDE_0
#define ROUTED_GATE_UP_PRODUCT_STRIDE_1 SEISMIC_PRODUCT_STRIDE_1
// Whether flat choice `c` runs in this pass:
// - `late == 0`: every choice (early pass before tier wait).
// - `late == 1`: only choices the route left absent (`streaming_left_absent`) that the wait has sourced.
// - `late == 2`: only VRAM hits (`source.code >= 0`, `DecodeHits` pass reading `front_sources`).
// - `late == 3`: only sourced misses (`!source.none()`, `DecodeBack` pass reading `back_sources`).
#define ROUTED_GATE_UP_ACTIVE(c, source)                                                                              \
    ((int)SEISMIC_PARAM_LATE == 0 ||                                                                                  \
     ((int)SEISMIC_PARAM_LATE == 2 && (source).code >= 0) ||                                                          \
     ((int)SEISMIC_PARAM_LATE == 3 && !(source).none()) ||                                                            \
     ((int)SEISMIC_PARAM_LATE == 1 && !(source).none() &&                                                             \
      streaming_left_absent(reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_STEP)), SEISMIC_STEP_STRIDE_0,   \
                            reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTING)),                       \
                            SEISMIC_ROUTING_STRIDE_0, SEISMIC_ROUTING_STRIDE_1, SEISMIC_DIM_FLIGHT, SEISMIC_DIM_P,    \
                            (c))))
#include "lib/routed/gate_up.cuh"

#ifdef SEISMIC_FORMING_ROUTED_GATE_UP_PART_CHOICE
template <int TPW, int KSPLIT>
__launch_bounds__(routed::DECODE_THREADS, routed::decode_min_blocks<1, packets::W0, packets::W1>())
__global__ void routed_gate_up_part_choice(SEISMIC_KERNEL_PARAMS) {
    ROUTED_GATE_UP_CHOICE_BODY
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_GATE_UP_PART_STAGE
// Each normalized row as NVFP4 operands, once for every expert.
__global__ void routed_gate_up_part_stage(SEISMIC_KERNEL_PARAMS) {
    ROUTED_GATE_UP_STAGE_BODY
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_GATE_UP_PART_GATHERED
template <int KSPLIT, int GATHER_TPW>
__launch_bounds__(routed::DECODE_THREADS, routed::decode_min_blocks<1, packets::W0, packets::W1>())
__global__ void routed_gate_up_part_gathered(SEISMIC_KERNEL_PARAMS) {
    ROUTED_GATE_UP_GATHERED_BODY(1)
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_GATE_UP_PART_GATHERED16
template <int KSPLIT, int GATHER_TPW>
__launch_bounds__(routed::DECODE_THREADS, routed::decode_min_blocks<routed::DECODE_BATCHES, packets::W0, packets::W1>())
__global__ void routed_gate_up_part_gathered16(SEISMIC_KERNEL_PARAMS) {
    ROUTED_GATE_UP_GATHERED_BODY(routed::DECODE_BATCHES)
}
#endif
