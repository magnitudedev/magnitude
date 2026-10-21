// Grouped expert projections; the CUDA form of `metal/routed_experts.metal`.
//   expand  block (column block, row tile, b): paired gate/up GEMM of expert
//           blocks[b] over the block's rows (tile row t reads normalized row
//           order[b, t]) with the activation-generic GLU (`activation`)
//           into `product` [B * T, F];
//   down    block (column block, row tile, b): down GEMM of expert blocks[b]
//           into the result [B, T, H].
// Both GEMMs run over the block's live rows only (the rows before its first
// -1): padding rows are neither read nor published, and a row tile or warp
// holding only padding does no work. Blocks of expert -1 exit.
#define KERNEL_W0 SEISMIC_EXPERT_GATE
#define KERNEL_W1 SEISMIC_EXPERT_UP
#define KERNEL_W2 SEISMIC_EXPERT_DOWN
#include "lib/routed/routed.cuh"

using element::Act;

// The declared shared bytes of the two launches (`routed.seismic`).
constexpr int EXPAND_SHARED = 3 * (32 * 144 + 2 * 10240) + 2 * 4096;
constexpr int DOWN_SHARED = 3 * (32 * 144 + 10240) + 4096;


// Convert each semantic normalized row once; all routed groups reuse it.
#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_STAGE
template <unsigned OPERAND>
__global__ void routed_experts_stage(SEISMIC_KERNEL_PARAMS) {
    using Operands = typename projection::GemmOperands<OPERAND>::type;
    const projection::Plain<Act, projection::AllRows> source{SEISMIC_PTR(SEISMIC_BUFFER_NORMALIZED), SEISMIC_NORMALIZED_STRIDE_0, projection::AllRows{}};
    projection::materialize_operand_row<Operands>(source, blockIdx.x, SEISMIC_DIM_H,
        SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED));
}
#endif

// The expert of grouped block `block` (-1 for an unused block).
#define ROUTED_BLOCK_EXPERT(block) \
    (reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_BLOCKS))[(block) * SEISMIC_BLOCKS_STRIDE_0])
// The `order` row of grouped block `block`.
#define ROUTED_BLOCK_ORDER(block) \
    (reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ORDER)) + (block) * SEISMIC_ORDER_STRIDE_0)

#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_EXPAND
template <unsigned OPERAND>
__global__ void routed_experts_expand(SEISMIC_KERNEL_PARAMS) {
    using Operands = typename projection::GemmOperands<OPERAND>::type;
    extern __shared__ uint4 dynamic_shared[];
    const projection::u64 block = blockIdx.z;
    const int expert = ROUTED_BLOCK_EXPERT(block);
    if (expert < 0 || blockIdx.x >= projection::gemm_columns(SEISMIC_DIM_F))
        return;
    const int *order = ROUTED_BLOCK_ORDER(block);
    const projection::u32 live = routed::block_rows(order, SEISMIC_ORDER_STRIDE_1, (projection::u32)SEISMIC_DIM_T);
    if ((projection::u64)blockIdx.y * routed::TILE_M >= live)
        return;
    const projection::u64 rows = block * SEISMIC_DIM_T;
    const projection::Glu<typename Operands::Storage, Act> epi{SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PRODUCT) + rows * SEISMIC_DIM_F * 2,
                                   SEISMIC_DIM_F, (int)SEISMIC_PARAM_ACTIVATION};
    const routed::GroupedRows source{OPERAND == 1 ? SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED) : SEISMIC_PTR(SEISMIC_BUFFER_NORMALIZED),
                                     OPERAND == 1 ? SEISMIC_DIM_H : SEISMIC_NORMALIZED_STRIDE_0, order,
                                     SEISMIC_ORDER_STRIDE_1};
    routed::routed_gemm<EXPAND_SHARED, Operands>(
        reinterpret_cast<projection::u8 *>(dynamic_shared), source, live, SEISMIC_DIM_H, blockIdx.x, SEISMIC_DIM_F,
        KERNEL_W0_MATRIX(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_GATE), expert),
        KERNEL_W1_MATRIX(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_UP), expert), epi);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_DOWN
template <unsigned OPERAND>
__global__ void routed_experts_down(SEISMIC_KERNEL_PARAMS) {
    using Operands = typename projection::GemmOperands<OPERAND>::type;
    extern __shared__ uint4 dynamic_shared[];
    const projection::u64 block = blockIdx.z;
    const int expert = ROUTED_BLOCK_EXPERT(block);
    if (expert < 0 || blockIdx.x >= projection::gemm_columns(SEISMIC_DIM_H))
        return;
    const projection::u32 live =
        routed::block_rows(ROUTED_BLOCK_ORDER(block), SEISMIC_ORDER_STRIDE_1, (projection::u32)SEISMIC_DIM_T);
    if ((projection::u64)blockIdx.y * routed::TILE_M >= live)
        return;
    const projection::u64 rows = block * SEISMIC_DIM_T;
    const projection::Store<Act> epi{SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER) + block * SEISMIC_RESULT_0_STRIDE_0 * 2,
                             SEISMIC_RESULT_0_STRIDE_1, 0};
    const projection::ActivationRows source{SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PRODUCT) + rows * SEISMIC_DIM_F * 2,
                                            SEISMIC_DIM_F};
    routed::routed_gemm<DOWN_SHARED, Operands>(
        reinterpret_cast<projection::u8 *>(dynamic_shared), source, live, SEISMIC_DIM_F, blockIdx.x, SEISMIC_DIM_H,
        KERNEL_W2_MATRIX(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_DOWN), expert),
        projection::NoWeight{}, epi);
}
#endif
