// Prefill projections of up-only experts over the grouped tables of
// `routed_group` (the `routed_experts` structure, its grids and its weights'
// places, with one expanding weight). Used block b (T tile rows of expert
// blocks[b]) is covered by ceil(T / BM) GEMM row tiles (BM = TILE_M),
// multiplied over its live rows only (`routed::expert_plain`).
//
// L1 (`routed_experts_up_expand`): K1 projection of the block's expert up
// rows with the row-gather A loader and the activated epilogue A(act(A(up)))
// into the `product` scratch [B * T, F].
// L2 (`routed_experts_up_down`): K1 projection of the block's products
// against the expert's down rows, published in A.

#define KERNEL_W0 SEISMIC_EXPERT_UP
#define KERNEL_W1 SEISMIC_EXPERT_DOWN
#include "lib/routed/routed.h"
#include "lib/streaming/control.h"

// The rows the up weights take (the normalized rows, or NVFP4 operands that
// `routed_experts_up_stage` forms into `staged`), and the product rows the
// down weights take (NVFP4 operands `routed_experts_up_quantize` forms in
// place).
constant constexpr bool EXPANSION_NVFP4 = projection::OperandRecipe<packets::W0_OPERANDS_NVFP4>::NVFP4;

kernel void routed_experts_up_grids(
    device const int *rows [[buffer(SEISMIC_BUFFER_ROWS)]],
    device const int *used [[buffer(SEISMIC_BUFFER_USED)]],
    device int *stage_grid [[buffer(SEISMIC_BUFFER_SCRATCH_STAGE_GRID)]],
    device int *expand_grid [[buffer(SEISMIC_BUFFER_SCRATCH_EXPAND_GRID)]],
    device int *quantize_grid [[buffer(SEISMIC_BUFFER_SCRATCH_QUANTIZE_GRID)]],
    device int *down_grid [[buffer(SEISMIC_BUFFER_SCRATCH_DOWN_GRID)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]) {
    constexpr uint BM = SEISMIC_TUNE_TILE_M;
    constexpr uint BN = SEISMIC_TUNE_TILE_N;
    routed::expert_grids(uint(rows[0]), uint(used[0]), uint(SEISMIC_DIM_T), (uint(SEISMIC_DIM_T) + BM - 1) / BM,
        (uint(SEISMIC_DIM_F) + BN - 1) / BN, (uint(SEISMIC_DIM_H) + BN - 1) / BN, stage_grid, expand_grid,
        quantize_grid, down_grid);
}

kernel void routed_experts_up_stage(
    device const int *first [[buffer(SEISMIC_BUFFER_FIRST)]],
    device const uchar *normalized [[buffer(SEISMIC_BUFFER_NORMALIZED)]],
    device uchar *staged [[buffer(SEISMIC_BUFFER_SCRATCH_STAGED)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint item [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    PROJECTION_NORMALIZE_SHARED(maxima);
    projection::operands<true>::form<256>(routed::activation(normalized, SEISMIC_NORMALIZED_STRIDE_0,
        SEISMIC_NORMALIZED_STRIDE_1, uint(SEISMIC_DIM_H)), uint(first[0]) + item, staged, uint(SEISMIC_DIM_H), maxima,
        thread_index);
}

kernel void routed_experts_up_quantize(
    device uchar *product [[buffer(SEISMIC_BUFFER_SCRATCH_PRODUCT)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint item [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    PROJECTION_NORMALIZE_SHARED(maxima);
    projection::operands<true>::form_in_place<256, routed::Act>(
        product + ulong(item) * SEISMIC_DIM_F * routed::Act::bytes, uint(SEISMIC_DIM_F), maxima, thread_index);
}

kernel void routed_experts_up_expand(
    device const uchar *normalized [[buffer(SEISMIC_BUFFER_NORMALIZED)]],
    device const uchar *staged [[buffer(SEISMIC_BUFFER_SCRATCH_STAGED)]],
    device const int *order [[buffer(SEISMIC_BUFFER_ORDER)]],
    device const int *blocks [[buffer(SEISMIC_BUFFER_BLOCKS)]],
    device const ulong *expert_up [[buffer(SEISMIC_BUFFER_EXPERT_UP)]],
    SYSTEM const int *map [[buffer(SEISMIC_BUFFER_MAP)]],
    device const int *step [[buffer(SEISMIC_BUFFER_STEP)]],
    device const ulong *records_up [[buffer(SEISMIC_BUFFER_RECORDS_UP)]],
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
    // The record the layer's map holds for the expert at the step
    // (`streaming_held`), or its place by index without a map (X 0).
    const int slot = SEISMIC_DIM_X == 0 ? -1
        : streaming_held(map + ulong(expert) * SEISMIC_MAP_STRIDE_0, SEISMIC_MAP_STRIDE_1,
              streaming_step(step, SEISMIC_STEP_STRIDE_0));
    const routed::Grouped in{EXPANSION_NVFP4 ? staged : normalized,
        EXPANSION_NVFP4 ? SEISMIC_DIM_H : SEISMIC_NORMALIZED_STRIDE_0, EXPANSION_NVFP4 ? 1 : SEISMIC_NORMALIZED_STRIDE_1,
        uint(SEISMIC_DIM_H), order + block * SEISMIC_ORDER_STRIDE_0, SEISMIC_ORDER_STRIDE_1};
    const auto up = routed::weights<packets::W0>(routed::expert_weights(records_up, expert_up, slot, expert),
        KERNEL_W0_LAYOUT(SEISMIC_DIM_H), 0, SEISMIC_DIM_H);
    const auto out = projection::scaling<(SEISMIC_DIM_US != 0)>::wrap(
        projection::Activated<A>{product + block * SEISMIC_DIM_T * SEISMIC_DIM_F * A::bytes,
            SEISMIC_DIM_F, 1, int(SEISMIC_PARAM_ACTIVATION)},
        projection::scale_factor(up_scale, SEISMIC_DIM_US, SEISMIC_UP_SCALE_STRIDE_0, ulong(expert)), 1.0f);
    const uint live = routed::block_rows(order + block * SEISMIC_ORDER_STRIDE_0, SEISMIC_ORDER_STRIDE_1,
        uint(SEISMIC_DIM_T));
    routed::expert_plain<packets::W0, BM, BN>(in, out, up, live, uint(SEISMIC_DIM_T), uint(SEISMIC_DIM_F),
        uint(SEISMIC_DIM_H), tm, group.x, tile_memory, sg, lane);
}

kernel void routed_experts_up_down(
    device const int *order [[buffer(SEISMIC_BUFFER_ORDER)]],
    device const int *blocks [[buffer(SEISMIC_BUFFER_BLOCKS)]],
    device const ulong *expert_down [[buffer(SEISMIC_BUFFER_EXPERT_DOWN)]],
    SYSTEM const int *map [[buffer(SEISMIC_BUFFER_MAP)]],
    device const int *step [[buffer(SEISMIC_BUFFER_STEP)]],
    device const ulong *records_down [[buffer(SEISMIC_BUFFER_RECORDS_DOWN)]],
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
    // The record the layer's map holds for the expert at the step
    // (`streaming_held`), or its place by index without a map (X 0).
    const int slot = SEISMIC_DIM_X == 0 ? -1
        : streaming_held(map + ulong(expert) * SEISMIC_MAP_STRIDE_0, SEISMIC_MAP_STRIDE_1,
              streaming_step(step, SEISMIC_STEP_STRIDE_0));
    const auto in = routed::activation(product + block * SEISMIC_DIM_T * SEISMIC_DIM_F * A::bytes,
        SEISMIC_DIM_F, 1, uint(SEISMIC_DIM_F));
    const auto down = routed::weights<packets::W1>(routed::expert_weights(records_down, expert_down, slot, expert),
        KERNEL_W1_LAYOUT(SEISMIC_DIM_F), 0, SEISMIC_DIM_F);
    const projection::Store<A> out{output + block * SEISMIC_RESULT_0_STRIDE_0 * A::bytes,
        SEISMIC_RESULT_0_STRIDE_1, SEISMIC_RESULT_0_STRIDE_2, 0};
    const uint live = routed::block_rows(order + block * SEISMIC_ORDER_STRIDE_0, SEISMIC_ORDER_STRIDE_1,
        uint(SEISMIC_DIM_T));
    routed::expert_plain<packets::W1, BM, BN>(in, out, down, live, uint(SEISMIC_DIM_T), uint(SEISMIC_DIM_H),
        uint(SEISMIC_DIM_F), tm, group.x, tile_memory, sg, lane);
}
