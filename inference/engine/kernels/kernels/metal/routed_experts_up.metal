// Prefill projections of up-only experts over the grouped tables of
// `routed_group` (the `routed_experts` structure with one expanding weight).
// Block b (T tile rows of expert blocks[b]) is covered by ceil(T / BM) GEMM
// row tiles (BM = TILE_M); blocks of expert -1 exit. A block of at most
// `routed::batched_rows` live rows runs the batched GEMV body instead of the
// GEMM (`routed::expert_plain`).
//
// L1 (`routed_experts_up_expand`): K1 projection of the block's expert up
// rows with the row-gather A loader and the activated epilogue A(act(A(up)))
// into the `product` scratch [B * T, F].
// L2 (`routed_experts_up_down`): K1 projection of the block's products
// against the expert's down rows, published in A.

#define KERNEL_W0 SEISMIC_EXPERT_UP
#define KERNEL_W1 SEISMIC_EXPERT_DOWN
#include "lib/routed/routed.h"

kernel void routed_experts_up_expand(
    device const uchar *normalized [[buffer(SEISMIC_BUFFER_NORMALIZED)]],
    device const int *order [[buffer(SEISMIC_BUFFER_ORDER)]],
    device const int *blocks [[buffer(SEISMIC_BUFFER_BLOCKS)]],
    device const uchar *expert_up [[buffer(SEISMIC_BUFFER_EXPERT_UP)]],
    device const float *up_scale [[buffer(SEISMIC_BUFFER_UP_SCALE)]],
    device uchar *product [[buffer(SEISMIC_BUFFER_SCRATCH_PRODUCT)]],
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
    const routed::Grouped in{normalized, SEISMIC_NORMALIZED_STRIDE_0, SEISMIC_NORMALIZED_STRIDE_1,
        uint(SEISMIC_DIM_H), order + block * SEISMIC_ORDER_STRIDE_0, SEISMIC_ORDER_STRIDE_1};
    const auto up = routed::weights<packets::W0>(expert_up, KERNEL_W0_LAYOUT(SEISMIC_DIM_H),
        routed::expert_row(ulong(expert), SEISMIC_DIM_F), SEISMIC_DIM_H);
    const auto out = projection::scaling<true>::wrap(
        projection::Activated<A>{product + block * SEISMIC_DIM_T * SEISMIC_DIM_F * A::bytes,
            SEISMIC_DIM_F, 1, int(SEISMIC_PARAM_ACTIVATION)},
        projection::scale_factor(up_scale, SEISMIC_DIM_E, SEISMIC_UP_SCALE_STRIDE_0, ulong(expert)), 1.0f);
    const uint live = routed::block_rows(order + block * SEISMIC_ORDER_STRIDE_0, SEISMIC_ORDER_STRIDE_1,
        uint(SEISMIC_DIM_T));
    routed::expert_plain<packets::W0, BM, BN>(in, out, up, live, uint(SEISMIC_DIM_T), uint(SEISMIC_DIM_F),
        uint(SEISMIC_DIM_H), tm, group.x, tile_memory, sg, lane);
}

kernel void routed_experts_up_down(
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
    const auto down = routed::weights<packets::W1>(expert_down, KERNEL_W1_LAYOUT(SEISMIC_DIM_F),
        routed::expert_row(ulong(expert), SEISMIC_DIM_H), SEISMIC_DIM_F);
    const projection::Store<A> out{output + block * SEISMIC_RESULT_0_STRIDE_0 * A::bytes,
        SEISMIC_RESULT_0_STRIDE_1, SEISMIC_RESULT_0_STRIDE_2, 0};
    const uint live = routed::block_rows(order + block * SEISMIC_ORDER_STRIDE_0, SEISMIC_ORDER_STRIDE_1,
        uint(SEISMIC_DIM_T));
    routed::expert_plain<packets::W1, BM, BN>(in, out, down, live, uint(SEISMIC_DIM_T), uint(SEISMIC_DIM_H),
        uint(SEISMIC_DIM_F), tm, group.x, tile_memory, sg, lane);
}
