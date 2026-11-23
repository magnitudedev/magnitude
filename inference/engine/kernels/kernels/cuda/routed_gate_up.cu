// Decode expansion of gated experts (M <= VERIFY_ROWS); the CUDA form of
// `metal/routed_gate_up.metal`, with the activation-generic GLU epilogue on
// the accumulators scaled by the expert's gate/up second-level scales where
// those ports are present (static GS, US). Each choice's weights are read
// where its source names (`routed::decode_source`): a bank record, or a
// host tier record read in place over the link.
// Two forms of one entry (GATHERED, a tuning form both valid at every M,
// which the tuning admits between):
// - `routed_gate_up_choice` (GATHERED 0): block row y = choice (m, k), the K1
//   paired GEMV over the choice's gate/up rows for its activation row.
// - `routed_gate_up_gathered` (GATHERED 1): a choice computes only when it
//   is the first choice reading its record, the K1 paired GEMV over that
//   record's gate/up rows for the activation row of every choice reading it
//   (`routed::gather`), so each chosen expert's records are read
//   once a step: a host record read once per row that named it would cost
//   its link bytes again for each (measured on an RTX 3090: 4.2x / 8.0x at 4
//   / 8 rows naming one host expert). A GEMV row's result does not depend on
//   the other rows, so every choice gets its one-row bits.
#define KERNEL_A SEISMIC_ELEMENT_A
#define KERNEL_W0 SEISMIC_EXPERT_GATE
#define KERNEL_W1 SEISMIC_EXPERT_UP
// The expansion's limit as its F32 bits LB (0: none), which the Glu epilogue
// clamps at.
#if SEISMIC_STATIC_LB != 0
#define PROJECTION_GLU_LIMIT SEISMIC_STATIC_LB
#endif

#define ROUTED_GATE_UP_SOURCE_ROWS SEISMIC_DIM_XS
#define ROUTED_GATE_UP_PRODUCT_OUT SEISMIC_RESULT_0_OUT
#define ROUTED_GATE_UP_PRODUCT_STRIDE_0 SEISMIC_RESULT_0_STRIDE_0
#define ROUTED_GATE_UP_PRODUCT_STRIDE_1 SEISMIC_RESULT_0_STRIDE_1
#include "lib/routed/gate_up.cuh"

#ifdef SEISMIC_FORMING_ROUTED_GATE_UP_CHOICE
template <int TPW, int KSPLIT>
__global__ void routed_gate_up_choice(SEISMIC_KERNEL_PARAMS) {
    ROUTED_GATE_UP_CHOICE_BODY
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_GATE_UP_STAGE
// Each normalized row as NVFP4 operands, once for every expert.
__global__ void routed_gate_up_stage(SEISMIC_KERNEL_PARAMS) {
    ROUTED_GATE_UP_STAGE_BODY
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_GATE_UP_GATHERED
template <int KSPLIT, int GATHER_TPW>
__global__ void routed_gate_up_gathered(SEISMIC_KERNEL_PARAMS) {
    ROUTED_GATE_UP_GATHERED_BODY
}
#endif
