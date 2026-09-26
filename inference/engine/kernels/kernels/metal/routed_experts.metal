// Prefill expert projections over the grouped tables of `routed_group`.
// Block b (T tile rows of expert blocks[b]) is covered by ceil(T / BM) GEMM
// row tiles (BM = TILE_M); blocks of expert -1 exit. A block of at most
// `routed::batched_rows` live rows runs the batched GEMV body instead of the
// GEMM (`routed::expert_paired` / `expert_plain`).
//
// L1 (`routed_experts_expand`): K1 paired projection of the block's
// expert gate/up rows with a row-gather A loader (tile row t reads normalized
// row order[b, t]; padding rows read zeros) and the SiLU . mul epilogue into
// the `product` scratch [B * T, F].
// L2 (`routed_experts_down`): K1 projection of the block's products
// against the expert's down rows, published in A.

#define KERNEL_W0 SEISMIC_EXPERT_GATE
#define KERNEL_W1 SEISMIC_EXPERT_UP
#define KERNEL_W2 SEISMIC_EXPERT_DOWN
#include "lib/routed/routed.h"

kernel void routed_experts_expand(
    device const uchar *normalized [[buffer(SEISMIC_BUFFER_NORMALIZED)]],
    device const int *order [[buffer(SEISMIC_BUFFER_ORDER)]],
    device const int *blocks [[buffer(SEISMIC_BUFFER_BLOCKS)]],
    device const uchar *expert_gate [[buffer(SEISMIC_BUFFER_EXPERT_GATE)]],
    device const uchar *expert_up [[buffer(SEISMIC_BUFFER_EXPERT_UP)]],
    device uchar *product [[buffer(SEISMIC_BUFFER_SCRATCH_PRODUCT)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint BM = SEISMIC_TUNE_TILE_M;
    constexpr uint BN = SEISMIC_TUNE_TILE_N;
    typedef routed::Act A;
    PROJECTION_GEMM_SHARED(tile_memory, BM, BN);
    const uint tn = group.x;
    const ulong subtiles = (SEISMIC_DIM_T + BM - 1) / BM;
    const ulong block = group.y / subtiles;
    const uint tm = uint(group.y % subtiles);
    const int expert = blocks[block * SEISMIC_BLOCKS_STRIDE_0];
    if (expert < 0) return;
    const routed::Grouped in{normalized, SEISMIC_NORMALIZED_STRIDE_0, SEISMIC_NORMALIZED_STRIDE_1,
        uint(SEISMIC_DIM_H), order + block * SEISMIC_ORDER_STRIDE_0, SEISMIC_ORDER_STRIDE_1};
    const auto gate = routed::weights<packets::W0>(expert_gate, KERNEL_W0_LAYOUT(SEISMIC_DIM_H),
        routed::expert_row(ulong(expert), SEISMIC_DIM_F), SEISMIC_DIM_H);
    const auto up = routed::weights<packets::W1>(expert_up, KERNEL_W1_LAYOUT(SEISMIC_DIM_H),
        routed::expert_row(ulong(expert), SEISMIC_DIM_F), SEISMIC_DIM_H);
    const projection::SiluMul<A> out{product + block * SEISMIC_DIM_T * SEISMIC_DIM_F * A::bytes,
        SEISMIC_DIM_F, 1};
    const uint live = routed::block_rows(order + block * SEISMIC_ORDER_STRIDE_0, SEISMIC_ORDER_STRIDE_1,
        uint(SEISMIC_DIM_T));
    routed::expert_paired<packets::W0, packets::W1, BM, BN>(in, out, gate, up, live, uint(SEISMIC_DIM_T),
        uint(SEISMIC_DIM_F), uint(SEISMIC_DIM_H), tm, tn, tile_memory, sg, lane);
}

kernel void routed_experts_down(
    device const int *order [[buffer(SEISMIC_BUFFER_ORDER)]],
    device const int *blocks [[buffer(SEISMIC_BUFFER_BLOCKS)]],
    device const uchar *expert_down [[buffer(SEISMIC_BUFFER_EXPERT_DOWN)]],
    device uchar *output [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    device const uchar *product [[buffer(SEISMIC_BUFFER_SCRATCH_PRODUCT)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint BM = SEISMIC_TUNE_TILE_M;
    constexpr uint BN = SEISMIC_TUNE_TILE_N;
    typedef routed::Act A;
    PROJECTION_GEMM_SHARED(tile_memory, BM, BN);
    const ulong subtiles = (SEISMIC_DIM_T + BM - 1) / BM;
    const ulong block = group.y / subtiles;
    const uint tm = uint(group.y % subtiles);
    const int expert = blocks[block * SEISMIC_BLOCKS_STRIDE_0];
    if (expert < 0) return;
    const auto in = routed::activation(product + block * SEISMIC_DIM_T * SEISMIC_DIM_F * A::bytes,
        SEISMIC_DIM_F, 1, uint(SEISMIC_DIM_F));
    const auto down = routed::weights<packets::W2>(expert_down, KERNEL_W2_LAYOUT(SEISMIC_DIM_F),
        routed::expert_row(ulong(expert), SEISMIC_DIM_H), SEISMIC_DIM_F);
    const projection::Store<A> out{output + block * SEISMIC_RESULT_0_STRIDE_0 * A::bytes,
        SEISMIC_RESULT_0_STRIDE_1, SEISMIC_RESULT_0_STRIDE_2, 0};
    const uint live = routed::block_rows(order + block * SEISMIC_ORDER_STRIDE_0, SEISMIC_ORDER_STRIDE_1,
        uint(SEISMIC_DIM_T));
    routed::expert_plain<packets::W2, BM, BN>(in, out, down, live, uint(SEISMIC_DIM_T), uint(SEISMIC_DIM_H),
        uint(SEISMIC_DIM_F), tm, group.x, tile_memory, sg, lane);
}
