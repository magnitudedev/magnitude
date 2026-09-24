// Grouped expert projections; the CUDA form of `metal/qwen_routed_experts.metal`.
// The K1 GEMM reads contiguous activation rows, so the grouped rows are
// gathered first:
//   gather  one block per grouped row (b, t): normalized row order[b, t]
//           (zeros for padding) into `staged` [B * T, H];
//   expand  block (column block, row tile, b): paired gate/up GEMM of expert
//           blocks[b] with SiLU . mul into `product` [B * T, F];
//   down    block (column block, row tile, b): down GEMM of expert blocks[b]
//           into the result [B, T, H].
// Blocks of expert -1 exit in every launch.
#define SJ_W0 SEISMIC_EXPERT_GATE
#define SJ_W1 SEISMIC_EXPERT_UP
#define SJ_W2 SEISMIC_EXPERT_DOWN
#include "common/projection.cuh"

using GShape = sj::GemmShape<32, 2, 4, 3>;
constexpr int ACT = SJ_DENSE_KIND(SEISMIC_ELEMENT_A);

// Grouped row m (tile row m % T of block m / T) reads normalized row
// order[m / T, m % T]; padding reads 0.
struct Grouped {
    const sj::u8 *x;
    sj::u64 stride;
    const int *order;
    sj::u64 order_block;
    sj::u64 order_lane;
    sj::u64 tile;
    static constexpr bool STAGED = true;
    static constexpr int FACTORS = 0;
    __device__ __forceinline__ void prepare_row(float *, sj::u32, float *) const {}
    __device__ __forceinline__ float value(const float *, sj::u32 m, sj::u64 k) const {
        const int row = order[(m / tile) * order_block + (m % tile) * order_lane];
        return row < 0 ? 0.0f : sj::Dense<ACT>::load(x, (sj::u64)row * stride + k);
    }
};

__device__ __forceinline__ sj::u64 routed_expert_row(sj::u64 expert, sj::u64 rows) {
    return expert * ((rows + 15) / 16 * 16);
}

// The expert of grouped block `block` (-1 for an unused block).
#define ROUTED_BLOCK_EXPERT(block) \
    (reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_BLOCKS))[(block) * SEISMIC_BLOCKS_STRIDE_0])

extern "C" __global__ void qwen_routed_experts_gather(SEISMIC_KERNEL_PARAMS) {
    const sj::u64 grouped = blockIdx.x;
    if (ROUTED_BLOCK_EXPERT(grouped / SEISMIC_DIM_T) < 0)
        return;
    const Grouped pro{SEISMIC_PTR(SEISMIC_BUFFER_NORMALIZED), SEISMIC_NORMALIZED_STRIDE_0,
                      reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ORDER)), SEISMIC_ORDER_STRIDE_0,
                      SEISMIC_ORDER_STRIDE_1, SEISMIC_DIM_T};
    sj::gemm_stage_row(pro, (sj::u32)grouped, SEISMIC_DIM_H, SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED));
}

extern "C" __global__ void qwen_routed_experts_expand(SEISMIC_KERNEL_PARAMS) {
    extern __shared__ uint4 dynamic_shared[];
    const sj::u64 block = blockIdx.z;
    const int expert = ROUTED_BLOCK_EXPERT(block);
    if (expert < 0 || blockIdx.x >= sj::gemm_columns(SEISMIC_DIM_F))
        return;
    const sj::u64 first = routed_expert_row((sj::u64)expert, SEISMIC_DIM_F);
    const sj::u64 rows = block * SEISMIC_DIM_T;
    const sj::SiluMul<ACT> epi{SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PRODUCT) + rows * SEISMIC_DIM_F * 2,
                               SEISMIC_DIM_F};
    sj::gemm_segment<GShape>(
        reinterpret_cast<sj::u8 *>(dynamic_shared), SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED) + rows * SEISMIC_DIM_H * 2,
        SEISMIC_DIM_H, (unsigned)SEISMIC_DIM_T, SEISMIC_DIM_H, blockIdx.x, SEISMIC_DIM_F,
        SJ_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_GATE) + first * SJ_CAT(SJ_W0, _ROW_STRIDE_BYTES)),
        SJ_W1_AT(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_UP) + first * SJ_CAT(SJ_W1, _ROW_STRIDE_BYTES)), epi);
}

extern "C" __global__ void qwen_routed_experts_down(SEISMIC_KERNEL_PARAMS) {
    extern __shared__ uint4 dynamic_shared[];
    const sj::u64 block = blockIdx.z;
    const int expert = ROUTED_BLOCK_EXPERT(block);
    if (expert < 0 || blockIdx.x >= sj::gemm_columns(SEISMIC_DIM_H))
        return;
    const sj::u64 first = routed_expert_row((sj::u64)expert, SEISMIC_DIM_H);
    const sj::u64 rows = block * SEISMIC_DIM_T;
    const sj::Store<ACT> epi{SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER) + block * SEISMIC_RESULT_0_STRIDE_0 * 2,
                             SEISMIC_RESULT_0_STRIDE_1, 0};
    sj::gemm_segment<GShape>(
        reinterpret_cast<sj::u8 *>(dynamic_shared), SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PRODUCT) + rows * SEISMIC_DIM_F * 2,
        SEISMIC_DIM_F, (unsigned)SEISMIC_DIM_T, SEISMIC_DIM_F, blockIdx.x, SEISMIC_DIM_H,
        SJ_W2_AT(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_DOWN) + first * SJ_CAT(SJ_W2, _ROW_STRIDE_BYTES)), sj::NoWeight{},
        epi);
}
