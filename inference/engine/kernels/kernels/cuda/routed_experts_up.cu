// Grouped up-only expert projections; the CUDA form of
// `metal/routed_experts_up.metal` (the `routed_experts` GEMMs with one
// expanding weight):
//   expand  block (column block, row tile, b): up GEMM of expert blocks[b]
//           over the block's rows (tile row t reads normalized row
//           order[b, t]) with the activated epilogue into `product` [B * T, F];
//   down    block (column block, row tile, b): down GEMM of expert blocks[b]
//           into the result [B, T, H].
// Both GEMMs run over the block's live rows only; blocks of expert -1 exit.
#define KERNEL_W0 SEISMIC_EXPERT_UP
#define KERNEL_W1 SEISMIC_EXPERT_DOWN
#include "lib/routed/routed.cuh"

using GShape = projection::GemmShape<32, 2, 4, 3>;
using element::Act;

// The expert of grouped block `block` (-1 for an unused block).
#define ROUTED_BLOCK_EXPERT(block) \
    (reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_BLOCKS))[(block) * SEISMIC_BLOCKS_STRIDE_0])
// The `order` row of grouped block `block`.
#define ROUTED_BLOCK_ORDER(block) \
    (reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ORDER)) + (block) * SEISMIC_ORDER_STRIDE_0)

extern "C" __global__ void routed_experts_up_expand(SEISMIC_KERNEL_PARAMS) {
    extern __shared__ uint4 dynamic_shared[];
    const projection::u64 block = blockIdx.z;
    const int expert = ROUTED_BLOCK_EXPERT(block);
    if (expert < 0 || blockIdx.x >= projection::gemm_columns(SEISMIC_DIM_F))
        return;
    const int *order = ROUTED_BLOCK_ORDER(block);
    const projection::u32 live = routed::block_rows(order, SEISMIC_ORDER_STRIDE_1, (projection::u32)SEISMIC_DIM_T);
    if ((projection::u64)blockIdx.y * GShape::BM >= live)
        return;
    const projection::u64 rows = block * SEISMIC_DIM_T;
    const auto epi = projection::scaling<true>::wrap(
        projection::Activated<Act>{SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PRODUCT) + rows * SEISMIC_DIM_F * 2,
                                   SEISMIC_DIM_F, (int)SEISMIC_PARAM_ACTIVATION},
        projection::scale_factor(SEISMIC_PTR(SEISMIC_BUFFER_UP_SCALE), SEISMIC_DIM_E,
                                 SEISMIC_UP_SCALE_STRIDE_0, (projection::u64)expert), 1.0f);
    const routed::GroupedRows source{SEISMIC_PTR(SEISMIC_BUFFER_NORMALIZED), SEISMIC_NORMALIZED_STRIDE_0, order,
                                     SEISMIC_ORDER_STRIDE_1};
    projection::gemm_segment<GShape>(reinterpret_cast<projection::u8 *>(dynamic_shared), source, live, SEISMIC_DIM_H,
                                     blockIdx.x, SEISMIC_DIM_F,
                                     KERNEL_W0_MATRIX(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_UP), expert),
                                     projection::NoWeight{}, epi);
}

extern "C" __global__ void routed_experts_up_down(SEISMIC_KERNEL_PARAMS) {
    extern __shared__ uint4 dynamic_shared[];
    const projection::u64 block = blockIdx.z;
    const int expert = ROUTED_BLOCK_EXPERT(block);
    if (expert < 0 || blockIdx.x >= projection::gemm_columns(SEISMIC_DIM_H))
        return;
    const projection::u32 live =
        routed::block_rows(ROUTED_BLOCK_ORDER(block), SEISMIC_ORDER_STRIDE_1, (projection::u32)SEISMIC_DIM_T);
    if ((projection::u64)blockIdx.y * GShape::BM >= live)
        return;
    const projection::u64 rows = block * SEISMIC_DIM_T;
    const projection::Store<Act> epi{SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER) + block * SEISMIC_RESULT_0_STRIDE_0 * 2,
                                     SEISMIC_RESULT_0_STRIDE_1, 0};
    const projection::ActivationRows source{SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PRODUCT) + rows * SEISMIC_DIM_F * 2,
                                            SEISMIC_DIM_F};
    projection::gemm_segment<GShape>(reinterpret_cast<projection::u8 *>(dynamic_shared), source, live, SEISMIC_DIM_F,
                                     blockIdx.x, SEISMIC_DIM_H,
                                     KERNEL_W1_MATRIX(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_DOWN), expert),
                                     projection::NoWeight{}, epi);
}
