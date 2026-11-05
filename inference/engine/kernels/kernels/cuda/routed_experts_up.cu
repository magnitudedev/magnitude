// Grouped up-only expert projections; the CUDA form of
// `metal/routed_experts_up.metal` (the `routed_experts` launches, grids and
// weights' places, with one expanding weight):
//   expand  block (column block, row tile, b) of the used blocks: up GEMM of
//           expert blocks[b] over the block's rows (tile row t reads
//           normalized row order[b, t]) with the activated epilogue into
//           `product` [B * T, F];
//   down    block (column block, row tile, b) of the used blocks: down GEMM
//           of expert blocks[b] into the result [B, T, H].
// Both GEMMs run over the block's live rows only.
#define KERNEL_W0 SEISMIC_EXPERT_UP
#define KERNEL_W1 SEISMIC_EXPERT_DOWN
#include "lib/routed/routed.cuh"
#include "lib/streaming/control.cuh"

// The record of expert `expert` for the step, in a kernel's own scope
// (`ROUTED_EXPERT_SLOT`).
#define EXPERT_SLOT(expert)                                                                                \
    ROUTED_EXPERT_SLOT(reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_MAP)), SEISMIC_DIM_X,      \
                       SEISMIC_MAP_STRIDE_0, SEISMIC_MAP_STRIDE_1,                                         \
                       streaming_step(reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_STEP)),     \
                                      SEISMIC_STEP_STRIDE_0),                                              \
                       expert)

using element::Act;
// The declared shared bytes of both launches (`routed.seismic`).
constexpr int SHARED = 3 * (32 * 144 + 10240) + 4096;
// Whether the up weight takes its rows as NVFP4 operands (the down weight's
// are formed by the quantize launch).
constexpr bool UP_NVFP4 = projection::OperandRecipe<packets::W0_OPERANDS_NVFP4>::NVFP4;

#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_UP_GRIDS
__global__ void routed_experts_up_grids(SEISMIC_KERNEL_PARAMS) {
    routed::expert_grids(SEISMIC_PTR(SEISMIC_BUFFER_ROWS), SEISMIC_PTR(SEISMIC_BUFFER_USED), SEISMIC_DIM_T,
                         projection::gemm_columns(SEISMIC_DIM_F), projection::gemm_columns(SEISMIC_DIM_H),
                         SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGE_GRID), SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_EXPAND_GRID),
                         SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_QUANTIZE_GRID), SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_DOWN_GRID));
}
#endif

// Form each grouped normalized row (from row first[0]) once, as the up
// GEMM's operands; all routed groups reuse it.
#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_UP_STAGE
template <unsigned OPERAND>
__device__ void routed_experts_up_stage_block(SEISMIC_KERNEL_PARAMS) {
    using Operands = projection::GemmOperandsOf<OPERAND, packets::W0>;
    using Raw = projection::Plain<Act, projection::AllRows>;
    const projection::Operands<UP_NVFP4, Raw> source{
        Raw{SEISMIC_PTR(SEISMIC_BUFFER_NORMALIZED), SEISMIC_NORMALIZED_STRIDE_0, projection::AllRows{}}};
    const unsigned first = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_FIRST))[0];
    projection::materialize_operand_row<Operands>(source, first + blockIdx.x, SEISMIC_DIM_H,
        SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED));
}
#endif

// The expert of grouped block `block`.
#define ROUTED_BLOCK_EXPERT(block) \
    (reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_BLOCKS))[(block) * SEISMIC_BLOCKS_STRIDE_0])
// The `order` row of grouped block `block`.
#define ROUTED_BLOCK_ORDER(block) \
    (reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ORDER)) + (block) * SEISMIC_ORDER_STRIDE_0)
#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_UP_EXPAND
template <unsigned OPERAND>
__device__ void routed_experts_up_expand_block(SEISMIC_KERNEL_PARAMS) {
    using Operands = projection::GemmOperandsOf<OPERAND, packets::W0>;
    extern __shared__ uint4 dynamic_shared[];
    const projection::u64 block = blockIdx.z;
    const int expert = ROUTED_BLOCK_EXPERT(block);
    const int *order = ROUTED_BLOCK_ORDER(block);
    const projection::u32 live = routed::block_rows(order, SEISMIC_ORDER_STRIDE_1, (projection::u32)SEISMIC_DIM_T);
    if ((projection::u64)blockIdx.y * routed::TILE_M >= live)
        return;
    const int slot = EXPERT_SLOT(expert);
    const projection::u64 rows = block * SEISMIC_DIM_T;
    const auto epi = projection::scaling<(SEISMIC_DIM_US != 0)>::wrap(
        projection::Activated<typename Operands::Storage, Act>{SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PRODUCT) + rows * SEISMIC_DIM_F * 2,
                                   SEISMIC_DIM_F, (int)SEISMIC_PARAM_ACTIVATION},
        projection::scale_factor(SEISMIC_PTR(SEISMIC_BUFFER_UP_SCALE), SEISMIC_DIM_US,
                                 SEISMIC_UP_SCALE_STRIDE_0, (projection::u64)expert), 1.0f);
    constexpr bool STAGED = OPERAND == 1 || UP_NVFP4;
    const routed::GroupedRows source{STAGED ? SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED) : SEISMIC_PTR(SEISMIC_BUFFER_NORMALIZED),
                                     STAGED ? SEISMIC_DIM_H : SEISMIC_NORMALIZED_STRIDE_0, order,
                                     SEISMIC_ORDER_STRIDE_1};
    routed::routed_gemm<SHARED, Operands>(reinterpret_cast<projection::u8 *>(dynamic_shared), source, live, SEISMIC_DIM_H,
                                     blockIdx.x, SEISMIC_DIM_F,
                                     ROUTED_EXPERT_MATRIX(KERNEL_W0, SEISMIC_PTR(SEISMIC_BUFFER_RECORDS_UP),
                                                          SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_UP), slot, expert),
                                     projection::NoWeight{}, epi);
}
#endif

// Each live product row in place as the down GEMM's NVFP4 operands: one
// block per row of the used blocks' [used * T, F] products.
#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_UP_QUANTIZE
template <unsigned OPERAND>
__device__ void routed_experts_up_quantize_block(SEISMIC_KERNEL_PARAMS) {
    const projection::u64 block = blockIdx.x / SEISMIC_DIM_T, lane = blockIdx.x % SEISMIC_DIM_T;
    if (ROUTED_BLOCK_ORDER(block)[lane * SEISMIC_ORDER_STRIDE_1] < 0)
        return;
    // The product the expand stored, as the down GEMM's operands.
    projection::form_operands_in_place<projection::GemmOperandsOf<OPERAND, packets::W1>,
                                       typename projection::GemmOperandsOf<OPERAND, packets::W0>::Storage,
                                       SEISMIC_DIM_F, 256>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PRODUCT), blockIdx.x);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_UP_DOWN
template <unsigned OPERAND>
__device__ void routed_experts_up_down_block(SEISMIC_KERNEL_PARAMS) {
    using Operands = projection::GemmOperandsOf<OPERAND, packets::W1>;
    extern __shared__ uint4 dynamic_shared[];
    const projection::u64 block = blockIdx.z;
    const int expert = ROUTED_BLOCK_EXPERT(block);
    const projection::u32 live =
        routed::block_rows(ROUTED_BLOCK_ORDER(block), SEISMIC_ORDER_STRIDE_1, (projection::u32)SEISMIC_DIM_T);
    if ((projection::u64)blockIdx.y * routed::TILE_M >= live)
        return;
    const int slot = EXPERT_SLOT(expert);
    const projection::u64 rows = block * SEISMIC_DIM_T;
    const projection::Store<Act> epi{SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER) + block * SEISMIC_RESULT_0_STRIDE_0 * 2,
                                     SEISMIC_RESULT_0_STRIDE_1, 0};
    const projection::ActivationRows source{SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PRODUCT) + rows * SEISMIC_DIM_F * 2,
                                            SEISMIC_DIM_F};
    routed::routed_gemm<SHARED, Operands>(reinterpret_cast<projection::u8 *>(dynamic_shared), source, live, SEISMIC_DIM_F,
                                     blockIdx.x, SEISMIC_DIM_H,
                                     ROUTED_EXPERT_MATRIX(KERNEL_W1, SEISMIC_PTR(SEISMIC_BUFFER_RECORDS_DOWN),
                                                          SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_DOWN), slot, expert),
                                     projection::NoWeight{}, epi);
}
#endif
