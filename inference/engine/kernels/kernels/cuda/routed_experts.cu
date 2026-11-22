// Grouped expert projections; the CUDA form of `metal/routed_experts.metal`.
//   grids   one thread writes the other launches' grids from the grouped
//           rows and used blocks;
//   stage   the grouped rows' operands: their s8 rows and groups (INT8),
//           or their F16 or NVFP4 operand rows;
//   expand  block (column block, row tile, b) of the used blocks: paired
//           gate/up GEMM of expert blocks[b] over the block's rows (tile row
//           t reads normalized row order[b, t]) with the activation-generic
//           GLU (`activation`) on the accumulators scaled by the expert's
//           `gate_scale`/`up_scale` where those ports are present (static
//           GS, US), into `product` [B * T, F];
//   quantize each live product row as the down GEMM's operands: its s8 row
//           and groups (INT8), or its NVFP4 operands in place;
//   down    block (column block, row tile, b) of the used blocks: down GEMM
//           of expert blocks[b] into the result [B, T, H].
// The expert's weights are read where the layer's map places them for the
// step (`EXPERT_SLOT`). Both GEMMs run over the block's live rows only (the
// rows before its first -1): padding rows are neither read nor published,
// and a row tile or warp holding only padding does no work. Every launch but
// the grids is the body of one block of its written grid (`_block`).
#define KERNEL_A SEISMIC_ELEMENT_A
#define KERNEL_W0 SEISMIC_EXPERT_GATE
#define KERNEL_W1 SEISMIC_EXPERT_UP
#define KERNEL_W2 SEISMIC_EXPERT_DOWN
// The expansion's limit as its F32 bits LB (0: none), which the Glu epilogue
// clamps at.
#if SEISMIC_STATIC_LB != 0
#define PROJECTION_GLU_LIMIT SEISMIC_STATIC_LB
#endif
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

using activation::Act;

// Whether the gate and up weights take their rows as NVFP4 operands (the
// down weight's are formed by the quantize launch).
constexpr bool EXPAND_NVFP4 =
    projection::OperandRecipe<packets::W0_OPERANDS_NVFP4, packets::W1_OPERANDS_NVFP4>::NVFP4;
// Whether both GEMMs run on the int8 path (INT8 = 1, which the declaration
// admits where every weight is admitted to it).
#define GROUPED_S8                                                                                         \
    (projection::int8_path<INT8, PACKETS_FORM_INT8(KERNEL_W0), PACKETS_FORM_INT8(KERNEL_W1),               \
                           PACKETS_FORM_INT8(KERNEL_W2)>())
// The order of the int8 rows each GEMM reads (`projection::gemm_s8_entries`),
// as the stage and quantize launches form them: the expansion's of the gate
// and up weights, the down GEMM's of the down weight.
#define EXPAND_ENTRIES (projection::gemm_s8_entries<GROUPED_S8, packets::W0, packets::W1>())
#define DOWN_ENTRIES (projection::gemm_s8_entries<GROUPED_S8, packets::W2>())

#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_GRIDS
template <unsigned BM>
__global__ void routed_experts_grids(SEISMIC_KERNEL_PARAMS) {
    routed::expert_grids<BM>(
                         SEISMIC_PTR(SEISMIC_BUFFER_ROWS), SEISMIC_PTR(SEISMIC_BUFFER_USED),
                         reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_BLOCKS)), SEISMIC_BLOCKS_STRIDE_0,
                         reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ORDER)), SEISMIC_ORDER_STRIDE_0,
                         SEISMIC_ORDER_STRIDE_1, SEISMIC_DIM_T, projection::gemm_columns(SEISMIC_DIM_F),
                         projection::gemm_columns(SEISMIC_DIM_H), SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_TILES),
                         SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGE_GRID),
                         SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_EXPAND_GRID),
                         SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_QUANTIZE_GRID),
                         SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_DOWN_GRID));
}
#endif

// Form each grouped normalized row (from row first[0]) once, as the
// expansion's operands; all routed groups reuse it.
#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_STAGE
template <unsigned INT8, unsigned OPERAND>
__device__ void routed_experts_stage_block(SEISMIC_KERNEL_PARAMS) {
    using Operands = projection::GemmOperandsOf<OPERAND, packets::W0, packets::W1>;
    using Raw = projection::Plain<Act, projection::AllRows>;
    const projection::Operands<EXPAND_NVFP4, Raw> source{
        Raw{SEISMIC_PTR(SEISMIC_BUFFER_NORMALIZED), SEISMIC_NORMALIZED_STRIDE_0, projection::AllRows{}}};
    const unsigned first = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_FIRST))[0];
    if constexpr (GROUPED_S8)
        projection::stage_row_s8<EXPAND_ENTRIES>(
            source, first + blockIdx.x, SEISMIC_DIM_H, SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED),
                                 reinterpret_cast<float2 *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_GROUPS)));
    else
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

#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_EXPAND
template <unsigned INT8, unsigned OPERAND, unsigned BM>
__device__ void routed_experts_expand_block(SEISMIC_KERNEL_PARAMS) {
    constexpr bool S8 = GROUPED_S8;
    using Operands = projection::GemmOperandsOf<OPERAND, packets::W0, packets::W1>;
    extern __shared__ uint4 dynamic_shared[];
    const uint2 tile = reinterpret_cast<const uint2 *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_TILES))[blockIdx.z];
    const projection::u64 block = tile.x;
    const projection::u32 live = tile.y;
    if ((projection::u64)blockIdx.y * BM >= live)
        return;
    const int expert = ROUTED_BLOCK_EXPERT(block);
    const int *order = ROUTED_BLOCK_ORDER(block);
    const int slot = EXPERT_SLOT(expert);
    if (slot == -1)
        return;
    const projection::u64 rows = block * SEISMIC_DIM_T;
    const auto epi = projection::scaling<(SEISMIC_DIM_GS + SEISMIC_DIM_US != 0)>::wrap(
        projection::Glu<typename Operands::Storage, Act>{
            element::out_over_scratch<typename Operands::Storage>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PRODUCT))
                .offset(rows * SEISMIC_DIM_F),
            SEISMIC_DIM_F, (int)SEISMIC_PARAM_ACTIVATION},
        projection::scale_factor(SEISMIC_PTR(SEISMIC_BUFFER_GATE_SCALE), SEISMIC_DIM_GS, SEISMIC_GATE_SCALE_STRIDE_0,
                                 (projection::u64)expert),
        projection::scale_factor(SEISMIC_PTR(SEISMIC_BUFFER_UP_SCALE), SEISMIC_DIM_US, SEISMIC_UP_SCALE_STRIDE_0,
                                 (projection::u64)expert));
    constexpr bool STAGED = S8 || OPERAND == 1 || EXPAND_NVFP4;
    const auto source = routed::grouped_rows(
        projection::gemm_source<S8, EXPAND_ENTRIES>(STAGED ? SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED) : SEISMIC_PTR(SEISMIC_BUFFER_NORMALIZED),
                                    STAGED ? SEISMIC_DIM_H : SEISMIC_NORMALIZED_STRIDE_0,
                                    SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_GROUPS), SEISMIC_DIM_H),
        order, SEISMIC_ORDER_STRIDE_0, SEISMIC_ORDER_STRIDE_1, (projection::u32)SEISMIC_DIM_T);
    routed::routed_gemm<BM, S8, Operands>(
        reinterpret_cast<projection::u8 *>(dynamic_shared), source, live, SEISMIC_DIM_H, blockIdx.x, SEISMIC_DIM_F,
        PACKETS_MATRIX(KERNEL_W0, (slab::Experts{SEISMIC_PTR(SEISMIC_BUFFER_RECORDS_GATE),
                                                    SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_GATE)}
                                      .matrix(slab::Source{slot}))),
        PACKETS_MATRIX(KERNEL_W1, (slab::Experts{SEISMIC_PTR(SEISMIC_BUFFER_RECORDS_UP),
                                                    SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_UP)}
                                      .matrix(slab::Source{slot}))),
        epi);
}
#endif

// Each live product row as the down GEMM's operands: its s8 row and groups
// (INT8), or its NVFP4 operands in place. One block per row of the used
// blocks' [used * T, F] products.
#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_QUANTIZE
template <unsigned INT8, unsigned OPERAND>
__device__ void routed_experts_quantize_block(SEISMIC_KERNEL_PARAMS) {
    const projection::u64 block = blockIdx.x / SEISMIC_DIM_T, lane = blockIdx.x % SEISMIC_DIM_T;
    if (ROUTED_BLOCK_ORDER(block)[lane * SEISMIC_ORDER_STRIDE_1] < 0 || EXPERT_SLOT(ROUTED_BLOCK_EXPERT(block)) == -1)
        return;
    // The product the expand stored, as the down GEMM's operands.
    if constexpr (GROUPED_S8)
        projection::stage_row_s8<DOWN_ENTRIES>(
            projection::Plain<Act, projection::AllRows>{SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PRODUCT), SEISMIC_DIM_F,
                                                        projection::AllRows{}},
            blockIdx.x, SEISMIC_DIM_F, SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PRODUCT_CODES),
            reinterpret_cast<float2 *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PRODUCT_GROUPS)));
    else
        projection::form_operands_in_place<projection::GemmOperandsOf<OPERAND, packets::W2>,
                                           typename projection::GemmOperandsOf<OPERAND, packets::W0, packets::W1>::Storage,
                                           SEISMIC_DIM_F, 256>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PRODUCT), blockIdx.x);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_DOWN
template <unsigned INT8, unsigned OPERAND, unsigned BM>
__device__ void routed_experts_down_block(SEISMIC_KERNEL_PARAMS) {
    constexpr bool S8 = GROUPED_S8;
    using Operands = projection::GemmOperandsOf<OPERAND, packets::W2>;
    extern __shared__ uint4 dynamic_shared[];
    const uint2 tile = reinterpret_cast<const uint2 *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_TILES))[blockIdx.z];
    const projection::u64 block = tile.x;
    const projection::u32 live = tile.y;
    if ((projection::u64)blockIdx.y * BM >= live)
        return;
    const int expert = ROUTED_BLOCK_EXPERT(block);
    const int slot = EXPERT_SLOT(expert);
    if (slot == -1)
        return;
    const projection::u64 rows = block * SEISMIC_DIM_T;
    const routed::BlockStore<Act> epi{SEISMIC_RESULT_0_OUT.offset(block * SEISMIC_RESULT_0_STRIDE_0),
                                      SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1,
                                      (projection::u32)SEISMIC_DIM_T};
    // The block's product rows, or their s8 rows and groups.
    const auto source = projection::gemm_source<S8, DOWN_ENTRIES>(
        S8 ? SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PRODUCT_CODES) + rows * SEISMIC_DIM_F
           : SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PRODUCT) + rows * SEISMIC_DIM_F * 2,
        SEISMIC_DIM_F, SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PRODUCT_GROUPS) + rows * (SEISMIC_DIM_F / 32) * 8,
        SEISMIC_DIM_F);
    routed::routed_gemm<BM, S8, Operands>(
        reinterpret_cast<projection::u8 *>(dynamic_shared), source, live, SEISMIC_DIM_F, blockIdx.x, SEISMIC_DIM_H,
        PACKETS_MATRIX(KERNEL_W2, (slab::Experts{SEISMIC_PTR(SEISMIC_BUFFER_RECORDS_DOWN),
                                                    SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_DOWN)}
                                      .matrix(slab::Source{slot}))),
        projection::NoWeight{}, epi);
}
#endif
