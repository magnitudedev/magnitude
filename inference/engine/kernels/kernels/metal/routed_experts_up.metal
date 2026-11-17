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
// Each in a form of the grouped projection instead (`projection routed for
// metal`): TALL (`relayout`, `tall`), INT8 (`int8_rows`,
// `int8_coefficients`, `int8`) or PACK (`pack`, `pack_coefficients`,
// `packed`, `unpacked`), the down's the same launches `down_` named.

#define KERNEL_W0 SEISMIC_EXPERT_UP
#define KERNEL_W1 SEISMIC_EXPERT_DOWN
#include "lib/routed/routed.h"
#include "lib/projection/packing.h"
#include "lib/projection/operator.h"
#include "lib/streaming/control.h"

inline int routed_expert_slot(SYSTEM const int *map, device const int *step, int expert,
    constant ulong *seismic_words) {
    return SEISMIC_DIM_X == 0 ? -2 - expert
        : streaming_held(map + ulong(expert) * SEISMIC_MAP_STRIDE_0, SEISMIC_MAP_STRIDE_1,
              streaming_step(step, SEISMIC_STEP_STRIDE_0));
}

// The rows the up weights take (the normalized rows, or NVFP4 operands that
// `routed_experts_up_stage` forms into `staged`), and the product rows the
// down weights take (NVFP4 operands `routed_experts_up_quantize` forms in
// place).
constant constexpr bool EXPANSION_NVFP4 = projection::OperandRecipe<packets::W0_OPERANDS_NVFP4>::NVFP4;

// A block's 32-row tiles and 64-row tall tiles (T static, a multiple of 32
// where a form takes them).
constant constexpr uint BLOCK_TILES = uint(SEISMIC_DIM_T) / 32u;
constant constexpr uint TALL_TILES = (uint(SEISMIC_DIM_T) + 63u) / 64u;

#define EXPERTS_UP_ARGUMENTS                                                                                    \
    device const int *first [[buffer(SEISMIC_BUFFER_FIRST)]],                                                   \
    device const int *rows [[buffer(SEISMIC_BUFFER_ROWS)]],                                                     \
    device const uchar *normalized [[buffer(SEISMIC_BUFFER_NORMALIZED)]],                                       \
    device const int *order [[buffer(SEISMIC_BUFFER_ORDER)]],                                                   \
    device const int *blocks [[buffer(SEISMIC_BUFFER_BLOCKS)]],                                                 \
    device const int *used [[buffer(SEISMIC_BUFFER_USED)]],                                                     \
    device const ulong *expert_up [[buffer(SEISMIC_BUFFER_EXPERT_UP)]],                                         \
    device const ulong *expert_down [[buffer(SEISMIC_BUFFER_EXPERT_DOWN)]],                                     \
    SYSTEM const int *map [[buffer(SEISMIC_BUFFER_MAP)]],                                                       \
    device const int *step [[buffer(SEISMIC_BUFFER_STEP)]],                                                     \
    device const ulong *records_up [[buffer(SEISMIC_BUFFER_RECORDS_UP)]],                                       \
    device const ulong *records_down [[buffer(SEISMIC_BUFFER_RECORDS_DOWN)]],                                   \
    device const float *up_scale [[buffer(SEISMIC_BUFFER_UP_SCALE)]],                                           \
    SEISMIC_RESULT_0_PARAM(output_cells),                                                                       \
    device uchar *staged [[buffer(SEISMIC_BUFFER_SCRATCH_STAGED)]],                                             \
    device uchar *product [[buffer(SEISMIC_BUFFER_SCRATCH_PRODUCT)]],                                           \
    device int *stage_grid [[buffer(SEISMIC_BUFFER_SCRATCH_STAGE_GRID)]],                                       \
    device int *expand_grid [[buffer(SEISMIC_BUFFER_SCRATCH_EXPAND_GRID)]],                                     \
    device int *quantize_grid [[buffer(SEISMIC_BUFFER_SCRATCH_QUANTIZE_GRID)]],                                 \
    device int *down_grid [[buffer(SEISMIC_BUFFER_SCRATCH_DOWN_GRID)]],                                         \
    device int *table_grid [[buffer(SEISMIC_BUFFER_SCRATCH_TABLE_GRID)]],                                       \
    device int *tile_grid [[buffer(SEISMIC_BUFFER_SCRATCH_TILE_GRID)]],                                         \
    device uchar *operand [[buffer(SEISMIC_BUFFER_SCRATCH_OPERAND)]],                                           \
    device uchar *tables [[buffer(SEISMIC_BUFFER_SCRATCH_TABLES)]],                                             \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

// Block `block`'s expert and the source of its weights: the record the
// layer's map holds for it at the step (`streaming_held`), or its place by
// index without a map (X 0).
#define EXPERTS_UP_BLOCK(block)                                                                                 \
    const int expert = blocks[ulong(block) * SEISMIC_BLOCKS_STRIDE_0];                                          \
    const int slot = routed_expert_slot(map, step, expert, seismic_words);                                      \
    if (slot == -1)                                                                                             \
        return;                                                                                                 \
    const slab::Source source = slab::Source{slot}

// The expert's up weights, and its down weights.
#define EXPERTS_UP_EXPANSION                                                                                    \
    const auto up = routed::expert_matrix<packets::W0>(routed::Experts{records_up, expert_up}, source,          \
        KERNEL_W0_LAYOUT(SEISMIC_DIM_H), SEISMIC_DIM_H)
#define EXPERTS_UP_DOWN                                                                                         \
    const auto down = routed::expert_matrix<packets::W1>(routed::Experts{records_down, expert_down}, source,    \
        KERNEL_W1_LAYOUT(SEISMIC_DIM_F), SEISMIC_DIM_F)

// The grouped rows the up weights take, of the blocks from `order` entry
// `at` (a block's own: its entry; every block's positions: 0).
#define EXPERTS_UP_ROWS(at)                                                                                     \
    routed::Grouped{EXPANSION_NVFP4 ? staged : normalized,                                                      \
        EXPANSION_NVFP4 ? SEISMIC_DIM_H : SEISMIC_NORMALIZED_STRIDE_0, EXPANSION_NVFP4 ? 1 : SEISMIC_NORMALIZED_STRIDE_1, \
        uint(SEISMIC_DIM_H), order + (at), SEISMIC_ORDER_STRIDE_1, SEISMIC_ORDER_STRIDE_0, uint(SEISMIC_DIM_T)}
#define EXPERTS_UP_BLOCK_ROWS(block) EXPERTS_UP_ROWS(ulong(block) * SEISMIC_ORDER_STRIDE_0)

// Block `block`'s live rows.
#define EXPERTS_UP_LIVE(block)                                                                                  \
    routed::block_rows(order + ulong(block) * SEISMIC_ORDER_STRIDE_0, SEISMIC_ORDER_STRIDE_1, uint(SEISMIC_DIM_T))

// Block `block`'s products: the activation of its up sums, scaled by its
// expert's `up_scale` where that port is present.
#define EXPERTS_UP_PRODUCTS(block)                                                                              \
    projection::scaling<(SEISMIC_DIM_US != 0)>::wrap(                                                           \
        projection::Activated<routed::Act>{element::out_over_scratch<routed::Act>(product).offset(ulong(block) * SEISMIC_DIM_T * SEISMIC_DIM_F), \
            SEISMIC_DIM_F, 1, int(SEISMIC_PARAM_ACTIVATION)},                                                   \
        projection::scale_factor(up_scale, SEISMIC_DIM_US, SEISMIC_UP_SCALE_STRIDE_0, ulong(expert)), 1.0f)

// Block `block`'s product rows as the down weights take them, and its
// published output rows.
#define EXPERTS_UP_BLOCK_PRODUCTS(block)                                                                        \
    routed::activation(product + ulong(block) * SEISMIC_DIM_T * SEISMIC_DIM_F * routed::Act::bytes,             \
        SEISMIC_DIM_F, 1, uint(SEISMIC_DIM_F))
#define EXPERTS_UP_OUTPUT(block)                                                                                \
    projection::Store<routed::Act>{element::Out<ELEMENT_OF(SEISMIC_RESULT_0)>(output_cells).offset(             \
            ulong(block) * SEISMIC_RESULT_0_STRIDE_0),                                                          \
        SEISMIC_RESULT_0_STRIDE_1, SEISMIC_RESULT_0_STRIDE_2, 0}

// Every block's products at their positions, padding positions zero.
#define EXPERTS_UP_PRODUCT_ROWS                                                                                 \
    routed::GroupedProducts{routed::activation(product, SEISMIC_DIM_F, 1, uint(SEISMIC_DIM_F)), EXPERTS_UP_ROWS(0)}

// The forms' scratch (`routed.h`): the GEMM's operand over every block's
// positions (64-row padded), and the table of weight `weight` (packets W)
// of the GEMM's `weights` (of `n` rows by `k` columns) of block `block`'s
// expert, of `bytes` a weight; the operand alone for its own pass, laid out
// with each run's code sums where `sums`.
#define EXPERTS_UP_OPERAND_ROWS ((ulong(SEISMIC_DIM_B) * SEISMIC_DIM_T + 63u) / 64u * 64u)
#define EXPERTS_UP_TABLE(block, weights, weight, bytes)                                                         \
    routed::weight_table(tables, uint(SEISMIC_DIM_B), routed::ordinals(tables)[block], weights, weight, bytes)
#define EXPERTS_UP_INT8(block, weights, weight, n, k)                                                           \
    routed::int8_forms(operand, EXPERTS_UP_OPERAND_ROWS,                                                        \
        EXPERTS_UP_TABLE(block, weights, weight, routed::int8_table_bytes(n, k)), n, k)
#define EXPERTS_UP_PACKING(block, weights, weight, n, k, W)                                                     \
    routed::packing_forms<projection::packing_scratch>(operand, EXPERTS_UP_OPERAND_ROWS,                        \
        EXPERTS_UP_TABLE(block, weights, weight,                                                                \
            routed::packing_table_bytes(n, k, projection::packing_codes<W>::sums)),                             \
        n, k, projection::packing_codes<W>::sums)
#define EXPERTS_UP_INT8_OPERAND(k) routed::int8_forms(operand, EXPERTS_UP_OPERAND_ROWS, tables, 0, k)
#define EXPERTS_UP_PACKING_OPERAND(k, sums)                                                                     \
    routed::packing_forms<projection::packing_scratch>(operand, EXPERTS_UP_OPERAND_ROWS, tables, 0, k, sums)

// The default grids, the expansion's form's (TALL: `grids`' runtime
// words), and the used blocks' expert ordinals where a form takes tables.
#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_UP_GRIDS
template <uint TILE_M, uint TILE_N>
kernel void routed_experts_up_grids(EXPERTS_UP_ARGUMENTS) {
    const uint t = uint(SEISMIC_DIM_T), f = uint(SEISMIC_DIM_F), h = uint(SEISMIC_DIM_H), blocks_used = uint(used[0]);
    routed::expert_grids(uint(rows[0]), blocks_used, t, (t + TILE_M - 1) / TILE_M, (f + TILE_N - 1) / TILE_N,
        (h + TILE_N - 1) / TILE_N, stage_grid, expand_grid, quantize_grid, down_grid);
    if (int(SEISMIC_RUNTIME_TALL) != 0)
        routed::form_grids(routed::form_of(int(SEISMIC_RUNTIME_INT8), int(SEISMIC_RUNTIME_PACK)), blocks_used, t, f,
            h, 1, expand_grid, table_grid, tile_grid);
    if (int(SEISMIC_RUNTIME_INT8) + int(SEISMIC_RUNTIME_PACK) + int(SEISMIC_RUNTIME_DOWN_INT8)
            + int(SEISMIC_RUNTIME_DOWN_PACK) != 0)
        routed::write_ordinals(blocks, SEISMIC_BLOCKS_STRIDE_0, blocks_used, routed::ordinals(tables));
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_UP_STAGE
kernel void routed_experts_up_stage(EXPERTS_UP_ARGUMENTS,
    uint item [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    PROJECTION_NORMALIZE_SHARED(maxima);
    projection::operands<true>::form<256>(routed::activation(normalized, SEISMIC_NORMALIZED_STRIDE_0,
        SEISMIC_NORMALIZED_STRIDE_1, uint(SEISMIC_DIM_H)), uint(first[0]) + item, staged, uint(SEISMIC_DIM_H), maxima,
        thread_index);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_UP_EXPAND
template <uint TILE_M, uint TILE_N>
kernel void routed_experts_up_expand(EXPERTS_UP_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_GEMM_SHARED(tile_memory, TILE_M, TILE_N);
    const uint subtiles = (uint(SEISMIC_DIM_T) + TILE_M - 1) / TILE_M;
    const uint block = tile.y / subtiles;
    EXPERTS_UP_BLOCK(block);
    EXPERTS_UP_EXPANSION;
    const routed::Grouped in = EXPERTS_UP_BLOCK_ROWS(block);
    const auto out = EXPERTS_UP_PRODUCTS(block);
    routed::expert_plain<packets::W0, TILE_M, TILE_N>(in, out, up, EXPERTS_UP_LIVE(block), uint(SEISMIC_DIM_T),
        uint(SEISMIC_DIM_F), uint(SEISMIC_DIM_H), tile.y % subtiles, tile.x, tile_memory, sg, lane);
}
#endif

// TALL: each block's rows in the tall GEMM's order, then its 64-row tall
// tiles (those past its live rows absent).
#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_UP_RELAYOUT
kernel void routed_experts_up_relayout(EXPERTS_UP_ARGUMENTS, PROJECTION_ITEM_ARGUMENTS) {
    EXPERTS_UP_BLOCK(position.y);
    (void)source;
    const routed::Grouped in = EXPERTS_UP_BLOCK_ROWS(position.y);
    projection::tall_relayout(in, routed::block_fragments(operand, position.y, uint(SEISMIC_DIM_T),
        uint(SEISMIC_DIM_H)), uint(SEISMIC_DIM_T), uint(SEISMIC_DIM_H), position.x);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_UP_TALL
template <uint TALL_K, uint STAGERS>
kernel void routed_experts_up_tall(EXPERTS_UP_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_GEMM_TALL_SHARED(shared, TALL_K);
    const uint block = tile.y / TALL_TILES, tm = tile.y % TALL_TILES;
    if (tm * 64u >= EXPERTS_UP_LIVE(block))
        return;
    EXPERTS_UP_BLOCK(block);
    EXPERTS_UP_EXPANSION;
    const routed::Grouped in = EXPERTS_UP_BLOCK_ROWS(block);
    const auto out = EXPERTS_UP_PRODUCTS(block);
    projection::gemm_tall<packets::W0, 64, TALL_K, STAGERS>(projection::tall_operand(in,
        routed::block_fragments(operand, block, uint(SEISMIC_DIM_T), uint(SEISMIC_DIM_H))), out, up,
        uint(SEISMIC_DIM_T), uint(SEISMIC_DIM_F), uint(SEISMIC_DIM_H), tm, tile.x, shared, sg, lane);
}
#endif

// INT8: the grouped rows quantized at their positions, each used expert's
// tables (by its first block), then each block's 32-row int8 tiles.
#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_UP_INT8_ROWS
kernel void routed_experts_up_int8_rows(EXPERTS_UP_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    if constexpr (!projection::int8_path<packets::W0>())
        return;
    EXPERTS_UP_BLOCK(index / uint(SEISMIC_DIM_T));
    (void)source;
    const routed::Grouped in = EXPERTS_UP_ROWS(0);
    projection::int8_quantize<projection::int8_codes<packets::W0>::interleaved>(in,
        EXPERTS_UP_INT8_OPERAND(uint(SEISMIC_DIM_H)), uint(SEISMIC_DIM_B) * uint(SEISMIC_DIM_T), uint(SEISMIC_DIM_H),
        index, thread_index, lane, 1.0f);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_UP_INT8_COEFFICIENTS
kernel void routed_experts_up_int8_coefficients(EXPERTS_UP_ARGUMENTS, PROJECTION_ITEM_ARGUMENTS) {
    if constexpr (!projection::int8_path<packets::W0>())
        return;
    if (!routed::first_block(blocks, SEISMIC_BLOCKS_STRIDE_0, position.y))
        return;
    EXPERTS_UP_BLOCK(position.y);
    EXPERTS_UP_EXPANSION;
    const uint f = uint(SEISMIC_DIM_F), h = uint(SEISMIC_DIM_H);
    projection::int8_coefficients(up, EXPERTS_UP_INT8(position.y, 1, 0, f, h), f, h, position.x);
}
#endif

// The default form's 32 x 64 tiles only where the launch takes that form:
// threadgroup memory a kernel does not use still slows its products.
#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_UP_INT8
kernel void routed_experts_up_int8(EXPERTS_UP_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    constexpr bool PATH = projection::int8_path<packets::W0>();
    threadgroup float4 shared_words[PATH ? 1 : projection::gemm_tile<32, 64>::bytes / 16];
    threadgroup uchar *shared = reinterpret_cast<threadgroup uchar *>(shared_words);
    const uint block = tile.y / BLOCK_TILES, first_row = block * uint(SEISMIC_DIM_T);
    const uint f = uint(SEISMIC_DIM_F), h = uint(SEISMIC_DIM_H);
    EXPERTS_UP_BLOCK(block);
    EXPERTS_UP_EXPANSION;
    const routed::Grouped in = EXPERTS_UP_ROWS(0);
    const auto out = routed::positioned(EXPERTS_UP_PRODUCTS(block), first_row);
    projection::gemm_int8<packets::W0, 32, PATH>(in, out, up, EXPERTS_UP_INT8(block, 1, 0, f, h),
        first_row + EXPERTS_UP_LIVE(block), f, h, tile.y, tile.x, shared, sg, lane);
}
#endif

// PACK: the grouped rows as integer codes at their positions, each used
// expert's tables (by its first block), then each block's 32-token packed
// tiles; on a device with tensor operations the default form's 32 x 64
// tiles instead (`unpacked`), in the same grid.
#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_UP_PACK
kernel void routed_experts_up_pack(EXPERTS_UP_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    if constexpr (!projection::packing_path<packets::W0>())
        return;
    EXPERTS_UP_BLOCK(index / (uint(SEISMIC_DIM_T) / 4u));
    (void)source;
    threadgroup float peaks[projection::packing_peaks()];
    const routed::Grouped in = EXPERTS_UP_ROWS(0);
    const projection::packing_bounds<0> bounds{};
    typedef projection::packing_codes<packets::W0> coded;
    projection::packing_operand<coded::centre, coded::folds, coded::sums, (coded::residue != 0 ? 1 : 0), 0>(in,
        EXPERTS_UP_PACKING_OPERAND(uint(SEISMIC_DIM_H), coded::sums), bounds, uint(SEISMIC_DIM_B) * uint(SEISMIC_DIM_T),
        uint(SEISMIC_DIM_H), index, peaks, thread_index, sg, lane);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_UP_PACK_COEFFICIENTS
kernel void routed_experts_up_pack_coefficients(EXPERTS_UP_ARGUMENTS, PROJECTION_TILE_ARGUMENTS,
    uint thread_index [[thread_index_in_threadgroup]]) {
    if constexpr (!projection::packing_path<packets::W0>())
        return;
    threadgroup float peaks[8];
    if (!routed::first_block(blocks, SEISMIC_BLOCKS_STRIDE_0, tile.y))
        return;
    EXPERTS_UP_BLOCK(tile.y);
    EXPERTS_UP_EXPANSION;
    const uint f = uint(SEISMIC_DIM_F), h = uint(SEISMIC_DIM_H);
    projection::packing_tile_coefficients(up, EXPERTS_UP_PACKING(tile.y, 1, 0, f, h, packets::W0), h, tile.x, peaks,
        thread_index, sg, lane);
}
#endif

// A threadgroup's four simdgroups take the 32-token tile's four 32 weight
// rows.
#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_UP_PACKED
template <uint WEIGHTS_AHEAD>
kernel void routed_experts_up_packed(EXPERTS_UP_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    if constexpr (!projection::packing_path<packets::W0>())
        return;
    const uint block = tile.y / BLOCK_TILES, first_row = block * uint(SEISMIC_DIM_T);
    const uint f = uint(SEISMIC_DIM_F), h = uint(SEISMIC_DIM_H);
    EXPERTS_UP_BLOCK(block);
    EXPERTS_UP_EXPANSION;
    const auto out = routed::positioned(EXPERTS_UP_PRODUCTS(block), first_row);
    projection::gemm_packed_tile<packets::W0, 1, 2, WEIGHTS_AHEAD>(
        out, up, EXPERTS_UP_PACKING(block, 1, 0, f, h, packets::W0),
        first_row + EXPERTS_UP_LIVE(block), tile.y * 32u, tile.x * 128u + sg * 32u, f, h, lane);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_UP_UNPACKED
kernel void routed_experts_up_unpacked(EXPERTS_UP_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    if constexpr (projection::packing_path<packets::W0>())
        return;
    PROJECTION_GEMM_SHARED(shared, 32, 64);
    const uint block = tile.y / BLOCK_TILES, first_row = block * uint(SEISMIC_DIM_T);
    EXPERTS_UP_BLOCK(block);
    EXPERTS_UP_EXPANSION;
    const routed::Grouped in = EXPERTS_UP_ROWS(0);
    const auto out = routed::positioned(EXPERTS_UP_PRODUCTS(block), first_row);
    projection::gemm_halves<packets::W0>(in, out, up, first_row + EXPERTS_UP_LIVE(block), uint(SEISMIC_DIM_F),
        uint(SEISMIC_DIM_H), tile.y, tile.x, shared, sg, lane);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_UP_QUANTIZE
kernel void routed_experts_up_quantize(EXPERTS_UP_ARGUMENTS,
    uint item [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    const ulong block = ulong(item) / SEISMIC_DIM_T;
    if (routed_expert_slot(map, step, blocks[block * SEISMIC_BLOCKS_STRIDE_0], seismic_words) == -1)
        return;
    PROJECTION_NORMALIZE_SHARED(maxima);
    projection::operands<true>::form_in_place<256, routed::Act>(
        product + ulong(item) * SEISMIC_DIM_F * routed::Act::bytes, uint(SEISMIC_DIM_F), maxima, thread_index);
}
#endif

// The down's form's grids, in the expansion's form's shared grids once its
// launches have run.
#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_UP_DOWN_GRIDS
kernel void routed_experts_up_down_grids(EXPERTS_UP_ARGUMENTS) {
    routed::form_grids(routed::form_of(int(SEISMIC_RUNTIME_DOWN_INT8), int(SEISMIC_RUNTIME_DOWN_PACK)),
        uint(used[0]), uint(SEISMIC_DIM_T), uint(SEISMIC_DIM_H), uint(SEISMIC_DIM_F), 1, down_grid, table_grid,
        tile_grid);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_UP_DOWN
template <uint TILE_M, uint TILE_N>
kernel void routed_experts_up_down(EXPERTS_UP_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_GEMM_SHARED(tile_memory, TILE_M, TILE_N);
    const uint subtiles = (uint(SEISMIC_DIM_T) + TILE_M - 1) / TILE_M;
    const uint block = tile.y / subtiles;
    EXPERTS_UP_BLOCK(block);
    EXPERTS_UP_DOWN;
    const auto in = EXPERTS_UP_BLOCK_PRODUCTS(block);
    const auto out = EXPERTS_UP_OUTPUT(block);
    routed::expert_plain<packets::W1, TILE_M, TILE_N>(in, out, down, EXPERTS_UP_LIVE(block), uint(SEISMIC_DIM_T),
        uint(SEISMIC_DIM_H), uint(SEISMIC_DIM_F), tile.y % subtiles, tile.x, tile_memory, sg, lane);
}
#endif

// The down GEMM's forms, over the blocks' products.
#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_UP_DOWN_RELAYOUT
kernel void routed_experts_up_down_relayout(EXPERTS_UP_ARGUMENTS, PROJECTION_ITEM_ARGUMENTS) {
    EXPERTS_UP_BLOCK(position.y);
    (void)source;
    const auto in = EXPERTS_UP_BLOCK_PRODUCTS(position.y);
    projection::tall_relayout(in, routed::block_fragments(operand, position.y, uint(SEISMIC_DIM_T),
        uint(SEISMIC_DIM_F)), uint(SEISMIC_DIM_T), uint(SEISMIC_DIM_F), position.x);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_UP_DOWN_TALL
template <uint DOWN_TALL_K, uint DOWN_STAGERS>
kernel void routed_experts_up_down_tall(EXPERTS_UP_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_GEMM_TALL_SHARED(shared, DOWN_TALL_K);
    const uint block = tile.y / TALL_TILES, tm = tile.y % TALL_TILES;
    if (tm * 64u >= EXPERTS_UP_LIVE(block))
        return;
    EXPERTS_UP_BLOCK(block);
    EXPERTS_UP_DOWN;
    const auto in = EXPERTS_UP_BLOCK_PRODUCTS(block);
    const auto out = EXPERTS_UP_OUTPUT(block);
    projection::gemm_tall<packets::W1, 64, DOWN_TALL_K, DOWN_STAGERS>(projection::tall_operand(in,
        routed::block_fragments(operand, block, uint(SEISMIC_DIM_T), uint(SEISMIC_DIM_F))), out, down,
        uint(SEISMIC_DIM_T), uint(SEISMIC_DIM_H), uint(SEISMIC_DIM_F), tm, tile.x, shared, sg, lane);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_UP_DOWN_INT8_ROWS
kernel void routed_experts_up_down_int8_rows(EXPERTS_UP_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    if constexpr (!projection::int8_path<packets::W1>())
        return;
    EXPERTS_UP_BLOCK(index / uint(SEISMIC_DIM_T));
    (void)source;
    const routed::GroupedProducts in = EXPERTS_UP_PRODUCT_ROWS;
    projection::int8_quantize<projection::int8_codes<packets::W1>::interleaved>(in,
        EXPERTS_UP_INT8_OPERAND(uint(SEISMIC_DIM_F)), uint(SEISMIC_DIM_B) * uint(SEISMIC_DIM_T), uint(SEISMIC_DIM_F),
        index, thread_index, lane, 1.0f);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_UP_DOWN_INT8_COEFFICIENTS
kernel void routed_experts_up_down_int8_coefficients(EXPERTS_UP_ARGUMENTS, PROJECTION_ITEM_ARGUMENTS) {
    if constexpr (!projection::int8_path<packets::W1>())
        return;
    if (!routed::first_block(blocks, SEISMIC_BLOCKS_STRIDE_0, position.y))
        return;
    EXPERTS_UP_BLOCK(position.y);
    EXPERTS_UP_DOWN;
    const uint h = uint(SEISMIC_DIM_H), f = uint(SEISMIC_DIM_F);
    projection::int8_coefficients(down, EXPERTS_UP_INT8(position.y, 1, 0, h, f), h, f, position.x);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_UP_DOWN_INT8
kernel void routed_experts_up_down_int8(EXPERTS_UP_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    constexpr bool PATH = projection::int8_path<packets::W1>();
    threadgroup float4 shared_words[PATH ? 1 : projection::gemm_tile<32, 64>::bytes / 16];
    threadgroup uchar *shared = reinterpret_cast<threadgroup uchar *>(shared_words);
    const uint block = tile.y / BLOCK_TILES, first_row = block * uint(SEISMIC_DIM_T);
    const uint h = uint(SEISMIC_DIM_H), f = uint(SEISMIC_DIM_F);
    EXPERTS_UP_BLOCK(block);
    EXPERTS_UP_DOWN;
    const auto in = routed::activation(product, SEISMIC_DIM_F, 1, f);
    const auto out = routed::positioned(EXPERTS_UP_OUTPUT(block), first_row);
    projection::gemm_int8<packets::W1, 32, PATH>(in, out, down, EXPERTS_UP_INT8(block, 1, 0, h, f),
        first_row + EXPERTS_UP_LIVE(block), h, f, tile.y, tile.x, shared, sg, lane);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_UP_DOWN_PACK
kernel void routed_experts_up_down_pack(EXPERTS_UP_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    if constexpr (!projection::packing_path<packets::W1>())
        return;
    EXPERTS_UP_BLOCK(index / (uint(SEISMIC_DIM_T) / 4u));
    (void)source;
    threadgroup float peaks[projection::packing_peaks()];
    const routed::GroupedProducts in = EXPERTS_UP_PRODUCT_ROWS;
    const projection::packing_bounds<0> bounds{};
    typedef projection::packing_codes<packets::W1> coded;
    projection::packing_operand<coded::centre, coded::folds, coded::sums, (coded::residue != 0 ? 1 : 0), 0>(in,
        EXPERTS_UP_PACKING_OPERAND(uint(SEISMIC_DIM_F), coded::sums), bounds, uint(SEISMIC_DIM_B) * uint(SEISMIC_DIM_T),
        uint(SEISMIC_DIM_F), index, peaks, thread_index, sg, lane);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_UP_DOWN_PACK_COEFFICIENTS
kernel void routed_experts_up_down_pack_coefficients(EXPERTS_UP_ARGUMENTS, PROJECTION_TILE_ARGUMENTS,
    uint thread_index [[thread_index_in_threadgroup]]) {
    if constexpr (!projection::packing_path<packets::W1>())
        return;
    threadgroup float peaks[8];
    if (!routed::first_block(blocks, SEISMIC_BLOCKS_STRIDE_0, tile.y))
        return;
    EXPERTS_UP_BLOCK(tile.y);
    EXPERTS_UP_DOWN;
    const uint h = uint(SEISMIC_DIM_H), f = uint(SEISMIC_DIM_F);
    projection::packing_tile_coefficients(down, EXPERTS_UP_PACKING(tile.y, 1, 0, h, f, packets::W1), f, tile.x, peaks,
        thread_index, sg, lane);
}
#endif

// A threadgroup's four simdgroups take the 32-token tile's four 32 weight
// rows.
#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_UP_DOWN_PACKED
template <uint DOWN_WEIGHTS_AHEAD>
kernel void routed_experts_up_down_packed(EXPERTS_UP_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    if constexpr (!projection::packing_path<packets::W1>())
        return;
    const uint block = tile.y / BLOCK_TILES, first_row = block * uint(SEISMIC_DIM_T);
    const uint h = uint(SEISMIC_DIM_H), f = uint(SEISMIC_DIM_F);
    EXPERTS_UP_BLOCK(block);
    EXPERTS_UP_DOWN;
    const auto out = routed::positioned(EXPERTS_UP_OUTPUT(block), first_row);
    projection::gemm_packed_tile<packets::W1, 1, 2, DOWN_WEIGHTS_AHEAD>(
        out, down, EXPERTS_UP_PACKING(block, 1, 0, h, f, packets::W1),
        first_row + EXPERTS_UP_LIVE(block), tile.y * 32u, tile.x * 128u + sg * 32u, h, f, lane);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_UP_DOWN_UNPACKED
kernel void routed_experts_up_down_unpacked(EXPERTS_UP_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    if constexpr (projection::packing_path<packets::W1>())
        return;
    PROJECTION_GEMM_SHARED(shared, 32, 64);
    const uint block = tile.y / BLOCK_TILES, first_row = block * uint(SEISMIC_DIM_T);
    EXPERTS_UP_BLOCK(block);
    EXPERTS_UP_DOWN;
    const auto in = routed::activation(product, SEISMIC_DIM_F, 1, uint(SEISMIC_DIM_F));
    const auto out = routed::positioned(EXPERTS_UP_OUTPUT(block), first_row);
    projection::gemm_halves<packets::W1>(in, out, down, first_row + EXPERTS_UP_LIVE(block), uint(SEISMIC_DIM_H),
        uint(SEISMIC_DIM_F), tile.y, tile.x, shared, sg, lane);
}
#endif
