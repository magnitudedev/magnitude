// Prefill expert projections over the grouped tables of `routed_group`.
// `routed_experts_grids` writes the other launches' grids from the grouped
// rows and used blocks. Used block b (T tile rows of expert blocks[b]) is
// covered by ceil(T / BM) GEMM row tiles (BM = TILE_M), its expert's weights
// read where the layer's map places them for the step (`routed_expert_slot`), and
// multiplied over its live rows only (`routed::expert_paired` /
// `expert_plain`).
//
// L1 (`routed_experts_expand`): K1 paired projection of the block's
// expert gate/up rows with a row-gather A loader (tile row t reads normalized
// row order[b, t]; padding rows read zeros) and the activation-generic GLU
// epilogue (`activation`), the accumulators scaled by the expert's
// `gate_scale`/`up_scale` where those ports are present (static GS, US),
// into the `product` scratch [B * T, F].
// L2 (`routed_experts_down`): K1 projection of the block's products
// against the expert's down rows, published in A.
// Each in a form of the grouped projection instead (`projection routed for
// metal`): TALL (`relayout`, `tall`), INT8 (`int8_rows`,
// `int8_coefficients`, `int8`) or PACK (`pack`, `pack_coefficients`,
// `packed`, `unpacked`), the down's the same launches `down_` named.

#define KERNEL_A SEISMIC_ELEMENT_A
#define KERNEL_W0 SEISMIC_EXPERT_GATE
#define KERNEL_W1 SEISMIC_EXPERT_UP
#define KERNEL_W2 SEISMIC_EXPERT_DOWN
// The expansion's limit as its F32 bits LB (0: none), which the Glu epilogue
// clamps at.
#if SEISMIC_STATIC_LB != 0
#define PROJECTION_GLU_LIMIT SEISMIC_STATIC_LB
#endif
#include "lib/routed/routed.h"
#include "lib/projection/packing.h"
#include "lib/projection/operator.h"
#include "lib/streaming/control.h"

inline int routed_expert_slot(device const int *map, device const int *step, int expert,
    constant ulong *seismic_words) {
    return SEISMIC_DIM_X == 0 ? -2 - expert
        : streaming_held_input(map + ulong(expert) * SEISMIC_MAP_STRIDE_0, SEISMIC_MAP_STRIDE_1,
              streaming_step(step, SEISMIC_STEP_STRIDE_0));
}

// The rows the gate/up weights take (the normalized rows, or NVFP4 operands
// that `routed_experts_stage` forms into `staged`), and the product rows the
// down weights take (NVFP4 operands `routed_experts_quantize` forms in
// place).
constant constexpr bool EXPANSION_NVFP4 =
    projection::OperandRecipe<packets::W0_OPERANDS_NVFP4, packets::W1_OPERANDS_NVFP4>::NVFP4;
constant constexpr bool DOWN_NVFP4 = projection::OperandRecipe<packets::W2_OPERANDS_NVFP4>::NVFP4;

// A block's 32-row tiles and 64-row tall tiles (T static, a multiple of 32
// where a form takes them).
constant constexpr uint BLOCK_TILES = uint(SEISMIC_DIM_T) / 32u;
constant constexpr uint TALL_TILES = (uint(SEISMIC_DIM_T) + 63u) / 64u;

#define EXPERTS_ARGUMENTS                                                                                       \
    device const int *first [[buffer(SEISMIC_BUFFER_FIRST)]],                                                   \
    device const int *rows [[buffer(SEISMIC_BUFFER_ROWS)]],                                                     \
    device const uchar *normalized [[buffer(SEISMIC_BUFFER_NORMALIZED)]],                                       \
    device const int *order [[buffer(SEISMIC_BUFFER_ORDER)]],                                                   \
    device const int *blocks [[buffer(SEISMIC_BUFFER_BLOCKS)]],                                                 \
    device const int *used [[buffer(SEISMIC_BUFFER_USED)]],                                                     \
    device const ulong *expert_gate [[buffer(SEISMIC_BUFFER_EXPERT_GATE)]],                                     \
    device const ulong *expert_up [[buffer(SEISMIC_BUFFER_EXPERT_UP)]],                                         \
    device const ulong *expert_down [[buffer(SEISMIC_BUFFER_EXPERT_DOWN)]],                                     \
    device const int *map [[buffer(SEISMIC_BUFFER_MAP)]],                                                       \
    device const int *step [[buffer(SEISMIC_BUFFER_STEP)]],                                                     \
    device const ulong *records_gate [[buffer(SEISMIC_BUFFER_RECORDS_GATE)]],                                   \
    device const ulong *records_up [[buffer(SEISMIC_BUFFER_RECORDS_UP)]],                                       \
    device const ulong *records_down [[buffer(SEISMIC_BUFFER_RECORDS_DOWN)]],                                   \
    device const float *gate_scale [[buffer(SEISMIC_BUFFER_GATE_SCALE)]],                                       \
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
// layer's map holds for it at the step (`streaming_held_input`), or its place by
// index without a map (X 0).
#define EXPERTS_BLOCK(block)                                                                                    \
    const int expert = blocks[ulong(block) * SEISMIC_BLOCKS_STRIDE_0];                                          \
    const int slot = routed_expert_slot(map, step, expert, seismic_words);                                      \
    if (slot == -1)                                                                                             \
        return;                                                                                                 \
    const slab::Source source = slab::Source{slot}

// The expert's gate and up weights, and its down weights.
#define EXPERTS_EXPANSION                                                                                       \
    const auto gate = routed::expert_matrix<packets::W0>(routed::Experts{records_gate, expert_gate}, source,    \
        KERNEL_W0_LAYOUT(SEISMIC_DIM_H), SEISMIC_DIM_H);                                                        \
    const auto up = routed::expert_matrix<packets::W1>(routed::Experts{records_up, expert_up}, source,          \
        KERNEL_W1_LAYOUT(SEISMIC_DIM_H), SEISMIC_DIM_H)
#define EXPERTS_DOWN                                                                                            \
    const auto down = routed::expert_matrix<packets::W2>(routed::Experts{records_down, expert_down}, source,    \
        KERNEL_W2_LAYOUT(SEISMIC_DIM_F), SEISMIC_DIM_F)

// The grouped rows the gate/up weights take, of the blocks from `order`
// entry `at` (a block's own: its entry; every block's positions: 0).
#define EXPERTS_ROWS(at)                                                                                        \
    routed::Grouped{EXPANSION_NVFP4 ? staged : normalized,                                                      \
        EXPANSION_NVFP4 ? SEISMIC_DIM_H : SEISMIC_NORMALIZED_STRIDE_0, EXPANSION_NVFP4 ? 1 : SEISMIC_NORMALIZED_STRIDE_1, \
        uint(SEISMIC_DIM_H), order + (at), SEISMIC_ORDER_STRIDE_1, SEISMIC_ORDER_STRIDE_0, uint(SEISMIC_DIM_T)}
#define EXPERTS_BLOCK_ROWS(block) EXPERTS_ROWS(ulong(block) * SEISMIC_ORDER_STRIDE_0)

// Block `block`'s live rows.
#define EXPERTS_LIVE(block)                                                                                     \
    routed::block_rows(order + ulong(block) * SEISMIC_ORDER_STRIDE_0, SEISMIC_ORDER_STRIDE_1, uint(SEISMIC_DIM_T))

// Block `block`'s products: the GLU of its gate and up sums, scaled by its
// expert's `gate_scale`/`up_scale` where those ports are present.
#define EXPERTS_PRODUCTS(block)                                                                                 \
    projection::scaling<(SEISMIC_DIM_GS + SEISMIC_DIM_US != 0)>::wrap(                                          \
        projection::Glu<routed::Act>{element::out_over_scratch<routed::Act>(product).offset(ulong(block) * SEISMIC_DIM_T * SEISMIC_DIM_F), \
            SEISMIC_DIM_F, 1, int(SEISMIC_PARAM_ACTIVATION)},                                                   \
        projection::scale_factor(gate_scale, SEISMIC_DIM_GS, SEISMIC_GATE_SCALE_STRIDE_0, ulong(expert)),       \
        projection::scale_factor(up_scale, SEISMIC_DIM_US, SEISMIC_UP_SCALE_STRIDE_0, ulong(expert)))

// Block `block`'s product rows as the down weights take them, and its
// published output rows.
#define EXPERTS_BLOCK_PRODUCTS(block)                                                                           \
    routed::activation(product + ulong(block) * SEISMIC_DIM_T * SEISMIC_DIM_F * routed::Act::bytes,             \
        SEISMIC_DIM_F, 1, uint(SEISMIC_DIM_F))
#define EXPERTS_OUTPUT(block)                                                                                   \
    projection::Store<routed::Act>{element::Out<ELEMENT_OF(SEISMIC_RESULT_0)>(output_cells).offset(             \
            ulong(block) * SEISMIC_RESULT_0_STRIDE_0),                                                          \
        SEISMIC_RESULT_0_STRIDE_1, SEISMIC_RESULT_0_STRIDE_2, 0}

// Every block's products at their positions, padding positions zero.
#define EXPERTS_PRODUCT_ROWS                                                                                    \
    routed::GroupedProducts{routed::activation(product, SEISMIC_DIM_F, 1, uint(SEISMIC_DIM_F)), EXPERTS_ROWS(0)}

// The forms' scratch (`routed.h`): the GEMM's operand over every block's
// positions (64-row padded), and the table of weight `weight` (packets W)
// of the GEMM's `weights` (of `n` rows by `k` columns) of block `block`'s
// expert, of `bytes` a weight; the operand alone for its own pass, laid out
// with each run's code sums where `sums`.
#define EXPERTS_OPERAND_ROWS ((ulong(SEISMIC_DIM_B) * SEISMIC_DIM_T + 63u) / 64u * 64u)
#define EXPERTS_TABLE(block, weights, weight, bytes)                                                            \
    routed::weight_table(tables, uint(SEISMIC_DIM_B), routed::block_ordinal(tables, block), weights, weight, bytes)
#define EXPERTS_INT8(block, weights, weight, n, k)                                                              \
    routed::int8_forms(operand, EXPERTS_OPERAND_ROWS,                                                           \
        EXPERTS_TABLE(block, weights, weight, routed::int8_table_bytes(n, k)), n, k)
#define EXPERTS_PACKING(block, weights, weight, n, k, W)                                                        \
    routed::packing_forms<projection::packing_scratch>(operand, EXPERTS_OPERAND_ROWS,                           \
        EXPERTS_TABLE(block, weights, weight,                                                                   \
            routed::packing_table_bytes(n, k, projection::packing_codes<W>::sums)),                             \
        n, k, projection::packing_codes<W>::sums)
#define EXPERTS_INT8_OPERAND(k) routed::int8_forms(operand, EXPERTS_OPERAND_ROWS, tables, 0, k)
#define EXPERTS_PACKING_OPERAND(k, sums)                                                                        \
    routed::packing_forms<projection::packing_scratch>(operand, EXPERTS_OPERAND_ROWS, tables, 0, k, sums)

// The default grids, the expansion's form's (TALL: `grids`' runtime
// words), and the used blocks' expert ordinals where a form takes tables.
#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_GRIDS
template <uint TILE_M, uint TILE_N>
kernel void routed_experts_grids(EXPERTS_ARGUMENTS) {
    const uint t = uint(SEISMIC_DIM_T), f = uint(SEISMIC_DIM_F), h = uint(SEISMIC_DIM_H), blocks_used = uint(used[0]);
    routed::expert_grids(uint(rows[0]), blocks_used, t, (t + TILE_M - 1) / TILE_M, (f * 2 + TILE_N - 1) / TILE_N,
        (h + TILE_N - 1) / TILE_N, stage_grid, expand_grid, quantize_grid, down_grid);
    if (int(SEISMIC_RUNTIME_TALL) != 0)
        routed::form_grids(routed::form_of(int(SEISMIC_RUNTIME_INT8), int(SEISMIC_RUNTIME_PACK)), blocks_used, t, f,
            h, 2, expand_grid, table_grid, tile_grid);
    if (int(SEISMIC_RUNTIME_INT8) + int(SEISMIC_RUNTIME_PACK) + int(SEISMIC_RUNTIME_DOWN_INT8)
            + int(SEISMIC_RUNTIME_DOWN_PACK) != 0)
        routed::write_ordinals(blocks, SEISMIC_BLOCKS_STRIDE_0, blocks_used, routed::ordinals(tables));
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_STAGE
kernel void routed_experts_stage(EXPERTS_ARGUMENTS,
    uint item [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    PROJECTION_NORMALIZE_SHARED(maxima);
    projection::operands<true>::form<256>(routed::activation(normalized, SEISMIC_NORMALIZED_STRIDE_0,
        SEISMIC_NORMALIZED_STRIDE_1, uint(SEISMIC_DIM_H)), uint(first[0]) + item, staged, uint(SEISMIC_DIM_H), maxima,
        thread_index);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_EXPAND
template <uint TILE_M, uint TILE_N>
kernel void routed_experts_expand(EXPERTS_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_GEMM_SHARED(tile_memory, TILE_M, TILE_N);
    const uint subtiles = (uint(SEISMIC_DIM_T) + TILE_M - 1) / TILE_M;
    const uint block = tile.y / subtiles;
    EXPERTS_BLOCK(block);
    EXPERTS_EXPANSION;
    const routed::Grouped in = EXPERTS_BLOCK_ROWS(block);
    const auto out = EXPERTS_PRODUCTS(block);
    routed::expert_paired<packets::W0, packets::W1, TILE_M, TILE_N>(in, out, gate, up, EXPERTS_LIVE(block),
        uint(SEISMIC_DIM_T), uint(SEISMIC_DIM_F), uint(SEISMIC_DIM_H), tile.y % subtiles, tile.x, tile_memory, sg,
        lane);
}
#endif

// TALL: each block's rows in the tall GEMM's order, then its 64-row tall
// tiles (those past its live rows absent).
#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_RELAYOUT
kernel void routed_experts_relayout(EXPERTS_ARGUMENTS, PROJECTION_ITEM_ARGUMENTS) {
    EXPERTS_BLOCK(position.y);
    (void)source;
    const routed::Grouped in = EXPERTS_BLOCK_ROWS(position.y);
    projection::tall_relayout(in, routed::block_fragments(operand, position.y, uint(SEISMIC_DIM_T),
        uint(SEISMIC_DIM_H)), uint(SEISMIC_DIM_T), uint(SEISMIC_DIM_H), position.x);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_TALL
template <uint TALL_K, uint STAGERS>
kernel void routed_experts_tall(EXPERTS_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_GEMM_TALL_SHARED(shared, TALL_K);
    const uint block = tile.y / TALL_TILES, tm = tile.y % TALL_TILES;
    if (tm * 64u >= EXPERTS_LIVE(block))
        return;
    EXPERTS_BLOCK(block);
    EXPERTS_EXPANSION;
    const routed::Grouped in = EXPERTS_BLOCK_ROWS(block);
    const auto out = EXPERTS_PRODUCTS(block);
    projection::gemm_tall_paired<packets::W0, packets::W1, 64, TALL_K, STAGERS>(projection::tall_operand(in,
        routed::block_fragments(operand, block, uint(SEISMIC_DIM_T), uint(SEISMIC_DIM_H))), out, gate, up,
        uint(SEISMIC_DIM_T), uint(SEISMIC_DIM_F), uint(SEISMIC_DIM_H), tm, tile.x, shared, sg, lane);
}
#endif

// INT8: the grouped rows quantized at their positions, each used expert's
// tables (by its first block), then each block's 32-row int8 tiles.
#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_INT8_ROWS
kernel void routed_experts_int8_rows(EXPERTS_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    if constexpr (!PROJECTION_PAIRED_INT8_PATH)
        return;
    EXPERTS_BLOCK(index / uint(SEISMIC_DIM_T));
    (void)source;
    const routed::Grouped in = EXPERTS_ROWS(0);
    projection::int8_quantize<packets::int8<packets::W0>::interleaved>(in,
        EXPERTS_INT8_OPERAND(uint(SEISMIC_DIM_H)), uint(SEISMIC_DIM_B) * uint(SEISMIC_DIM_T), uint(SEISMIC_DIM_H),
        index, thread_index, lane, 1.0f);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_INT8_COEFFICIENTS
kernel void routed_experts_int8_coefficients(EXPERTS_ARGUMENTS, PROJECTION_ITEM_ARGUMENTS) {
    if constexpr (!PROJECTION_PAIRED_INT8_PATH)
        return;
    if (!routed::first_block(blocks, SEISMIC_BLOCKS_STRIDE_0, position.y))
        return;
    EXPERTS_BLOCK(position.y);
    EXPERTS_EXPANSION;
    const uint f = uint(SEISMIC_DIM_F), h = uint(SEISMIC_DIM_H);
    projection::int8_coefficients(gate, EXPERTS_INT8(position.y, 2, 0, f, h), f, h, position.x);
    projection::int8_coefficients(up, EXPERTS_INT8(position.y, 2, 1, f, h), f, h, position.x);
}
#endif

// The default form's 32 x 64 tiles only where the launch takes that form:
// threadgroup memory a kernel does not use still slows its products.
#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_INT8
kernel void routed_experts_int8(EXPERTS_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    constexpr bool PATH = PROJECTION_PAIRED_INT8_PATH;
    threadgroup float exchange[PATH ? 2 * 32 * 32 : 1];
    threadgroup float4 shared_words[PATH ? 1 : projection::gemm_tile<32, 64>::bytes / 16];
    threadgroup uchar *shared = reinterpret_cast<threadgroup uchar *>(shared_words);
    const uint block = tile.y / BLOCK_TILES, first_row = block * uint(SEISMIC_DIM_T);
    const uint f = uint(SEISMIC_DIM_F), h = uint(SEISMIC_DIM_H);
    EXPERTS_BLOCK(block);
    EXPERTS_EXPANSION;
    const routed::Grouped in = EXPERTS_ROWS(0);
    const auto out = routed::positioned(EXPERTS_PRODUCTS(block), first_row);
    projection::gemm_int8_paired<packets::W0, packets::W1, PATH, 32>(in, out, gate, up,
        EXPERTS_INT8(block, 2, 0, f, h), EXPERTS_INT8(block, 2, 1, f, h), first_row + EXPERTS_LIVE(block), f, h,
        tile.y, tile.x, shared, exchange, sg, lane);
}
#endif

// PACK: the grouped rows as integer codes at their positions, each used
// expert's tables (by its first block), then each block's 32-token packed
// tiles; on a device with tensor operations the default form's 32 x 64
// tiles instead (`unpacked`), in the same grid.
#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_PACK
kernel void routed_experts_pack(EXPERTS_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    PROJECTION_PAIRED_PACKING_ADMITTED;
    if constexpr (!PROJECTION_PAIRED_PACKING_PATH)
        return;
    EXPERTS_BLOCK(index / (uint(SEISMIC_DIM_T) / 4u));
    (void)source;
    threadgroup float peaks[projection::packing_peaks()];
    const routed::Grouped in = EXPERTS_ROWS(0);
    const projection::packing_bounds<0> bounds{};
    typedef projection::packing_shared<1, packets::W0, 1, packets::W1, 1> coded;
    projection::packing_operand<coded::centre, coded::folds, coded::sums, coded::residue, 0>(in,
        EXPERTS_PACKING_OPERAND(uint(SEISMIC_DIM_H), coded::sums), bounds, uint(SEISMIC_DIM_B) * uint(SEISMIC_DIM_T),
        uint(SEISMIC_DIM_H), index, peaks, thread_index, sg, lane);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_PACK_COEFFICIENTS
kernel void routed_experts_pack_coefficients(EXPERTS_ARGUMENTS, PROJECTION_TILE_ARGUMENTS,
    uint thread_index [[thread_index_in_threadgroup]]) {
    PROJECTION_PAIRED_PACKING_ADMITTED;
    if constexpr (!PROJECTION_PAIRED_PACKING_PATH)
        return;
    threadgroup float peaks0[8];
    threadgroup float peaks1[8];
    if (!routed::first_block(blocks, SEISMIC_BLOCKS_STRIDE_0, tile.y))
        return;
    EXPERTS_BLOCK(tile.y);
    EXPERTS_EXPANSION;
    const uint f = uint(SEISMIC_DIM_F), h = uint(SEISMIC_DIM_H);
    projection::packing_tile_coefficients(gate, EXPERTS_PACKING(tile.y, 2, 0, f, h, packets::W0), h, tile.x, peaks0,
        thread_index, sg, lane);
    projection::packing_tile_coefficients(up, EXPERTS_PACKING(tile.y, 2, 1, f, h, packets::W1), h, tile.x, peaks1,
        thread_index, sg, lane);
}
#endif

// A threadgroup's two simdgroup pairs take the 32-token tile's two 32
// features, or coalesce consecutive 32-row blocks of the same expert into a
// 64-token (`PACK_TOKENS = 4`) tile over half the features per threadgroup.
#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_PACKED
template <uint WEIGHTS_AHEAD>
kernel void routed_experts_packed(EXPERTS_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_PAIRED_PACKING_ADMITTED;
    if constexpr (!PROJECTION_PAIRED_PACKING_PATH)
        return;
    threadgroup float exchange[projection::packing_simdgroups / 2 * projection::packing_exchange()];
    const uint block = tile.y / BLOCK_TILES, first_row = block * uint(SEISMIC_DIM_T);
    const uint f = uint(SEISMIC_DIM_F), h = uint(SEISMIC_DIM_H), pair = sg / 2u;
    EXPERTS_BLOCK(block);
    EXPERTS_EXPANSION;
    if constexpr (BLOCK_TILES == 1u) {
        const uint role = routed::block_pair_role(tables, block);
        if (role != 0u) {
            const uint first_block = (role == routed::ORDINAL_PAIR_ODD) ? (block - 1u) : block;
            const uint pair_first_row = first_block * 32u;
            const uint pair_m_rows = (first_block + 1u) * 32u + EXPERTS_LIVE(first_block + 1u);
            const uint side = (role == routed::ORDINAL_PAIR_ODD) ? 1u : 0u;
            const auto out = routed::positioned(EXPERTS_PRODUCTS(first_block), pair_first_row);
            projection::gemm_packed_paired_tile<packets::W0, packets::W1, 1, 4, WEIGHTS_AHEAD>(out, gate, up,
                EXPERTS_PACKING(block, 2, 0, f, h, packets::W0),
                EXPERTS_PACKING(block, 2, 1, f, h, packets::W1), pair_m_rows,
                pair_first_row, tile.x * 64u + side * 32u + pair * 16u, f, h, pair, exchange, sg, lane);
            return;
        }
    }
    const auto out = routed::positioned(EXPERTS_PRODUCTS(block), first_row);
    projection::gemm_packed_paired_tile<packets::W0, packets::W1, 1, 2, WEIGHTS_AHEAD>(out, gate, up,
        EXPERTS_PACKING(block, 2, 0, f, h, packets::W0),
        EXPERTS_PACKING(block, 2, 1, f, h, packets::W1), first_row + EXPERTS_LIVE(block),
        tile.y * 32u, tile.x * 64u + pair * 32u, f, h, pair, exchange, sg, lane);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_UNPACKED
kernel void routed_experts_unpacked(EXPERTS_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_PAIRED_PACKING_ADMITTED;
    if constexpr (PROJECTION_PAIRED_PACKING_PATH)
        return;
    PROJECTION_GEMM_SHARED(shared, 32, 64);
    const uint block = tile.y / BLOCK_TILES, first_row = block * uint(SEISMIC_DIM_T);
    EXPERTS_BLOCK(block);
    EXPERTS_EXPANSION;
    const routed::Grouped in = EXPERTS_ROWS(0);
    const auto out = routed::positioned(EXPERTS_PRODUCTS(block), first_row);
    projection::gemm_paired_halves<packets::W0, packets::W1>(in, out, gate, up, first_row + EXPERTS_LIVE(block),
        uint(SEISMIC_DIM_F), uint(SEISMIC_DIM_H), tile.y, tile.x, shared, sg, lane);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_QUANTIZE
kernel void routed_experts_quantize(EXPERTS_ARGUMENTS,
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
#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_DOWN_GRIDS
kernel void routed_experts_down_grids(EXPERTS_ARGUMENTS) {
    routed::form_grids(routed::form_of(int(SEISMIC_RUNTIME_DOWN_INT8), int(SEISMIC_RUNTIME_DOWN_PACK)),
        uint(used[0]), uint(SEISMIC_DIM_T), uint(SEISMIC_DIM_H), uint(SEISMIC_DIM_F), 1, down_grid, table_grid,
        tile_grid);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_DOWN
template <uint TILE_M, uint TILE_N>
kernel void routed_experts_down(EXPERTS_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_GEMM_SHARED(tile_memory, TILE_M, TILE_N);
    const uint subtiles = (uint(SEISMIC_DIM_T) + TILE_M - 1) / TILE_M;
    const uint block = tile.y / subtiles;
    EXPERTS_BLOCK(block);
    EXPERTS_DOWN;
    const auto in = EXPERTS_BLOCK_PRODUCTS(block);
    const auto out = EXPERTS_OUTPUT(block);
    routed::expert_plain<packets::W2, TILE_M, TILE_N>(in, out, down, EXPERTS_LIVE(block), uint(SEISMIC_DIM_T),
        uint(SEISMIC_DIM_H), uint(SEISMIC_DIM_F), tile.y % subtiles, tile.x, tile_memory, sg, lane);
}
#endif

// The down GEMM's forms, over the blocks' products.
#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_DOWN_RELAYOUT
kernel void routed_experts_down_relayout(EXPERTS_ARGUMENTS, PROJECTION_ITEM_ARGUMENTS) {
    EXPERTS_BLOCK(position.y);
    (void)source;
    const auto in = EXPERTS_BLOCK_PRODUCTS(position.y);
    projection::tall_relayout(in, routed::block_fragments(operand, position.y, uint(SEISMIC_DIM_T),
        uint(SEISMIC_DIM_F)), uint(SEISMIC_DIM_T), uint(SEISMIC_DIM_F), position.x);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_DOWN_TALL
template <uint DOWN_TALL_K, uint DOWN_STAGERS>
kernel void routed_experts_down_tall(EXPERTS_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_GEMM_TALL_SHARED(shared, DOWN_TALL_K);
    const uint block = tile.y / TALL_TILES, tm = tile.y % TALL_TILES;
    if (tm * 64u >= EXPERTS_LIVE(block))
        return;
    EXPERTS_BLOCK(block);
    EXPERTS_DOWN;
    const auto in = EXPERTS_BLOCK_PRODUCTS(block);
    const auto out = EXPERTS_OUTPUT(block);
    projection::gemm_tall<packets::W2, 64, DOWN_TALL_K, DOWN_STAGERS>(projection::tall_operand(in,
        routed::block_fragments(operand, block, uint(SEISMIC_DIM_T), uint(SEISMIC_DIM_F))), out, down,
        uint(SEISMIC_DIM_T), uint(SEISMIC_DIM_H), uint(SEISMIC_DIM_F), tm, tile.x, shared, sg, lane);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_DOWN_INT8_ROWS
kernel void routed_experts_down_int8_rows(EXPERTS_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    if constexpr (!projection::int8_path<packets::W2>())
        return;
    EXPERTS_BLOCK(index / uint(SEISMIC_DIM_T));
    (void)source;
    const routed::GroupedProducts in = EXPERTS_PRODUCT_ROWS;
    projection::int8_quantize<packets::int8<packets::W2>::interleaved>(in,
        EXPERTS_INT8_OPERAND(uint(SEISMIC_DIM_F)), uint(SEISMIC_DIM_B) * uint(SEISMIC_DIM_T), uint(SEISMIC_DIM_F),
        index, thread_index, lane, 1.0f);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_DOWN_INT8_COEFFICIENTS
kernel void routed_experts_down_int8_coefficients(EXPERTS_ARGUMENTS, PROJECTION_ITEM_ARGUMENTS) {
    if constexpr (!projection::int8_path<packets::W2>())
        return;
    if (!routed::first_block(blocks, SEISMIC_BLOCKS_STRIDE_0, position.y))
        return;
    EXPERTS_BLOCK(position.y);
    EXPERTS_DOWN;
    const uint h = uint(SEISMIC_DIM_H), f = uint(SEISMIC_DIM_F);
    projection::int8_coefficients(down, EXPERTS_INT8(position.y, 1, 0, h, f), h, f, position.x);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_DOWN_INT8
kernel void routed_experts_down_int8(EXPERTS_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    constexpr bool PATH = projection::int8_path<packets::W2>();
    threadgroup float4 shared_words[PATH ? 1 : projection::gemm_tile<32, 64>::bytes / 16];
    threadgroup uchar *shared = reinterpret_cast<threadgroup uchar *>(shared_words);
    const uint block = tile.y / BLOCK_TILES, first_row = block * uint(SEISMIC_DIM_T);
    const uint h = uint(SEISMIC_DIM_H), f = uint(SEISMIC_DIM_F);
    EXPERTS_BLOCK(block);
    EXPERTS_DOWN;
    const auto in = routed::activation(product, SEISMIC_DIM_F, 1, f);
    const auto out = routed::positioned(EXPERTS_OUTPUT(block), first_row);
    projection::gemm_int8<packets::W2, 32, PATH>(in, out, down, EXPERTS_INT8(block, 1, 0, h, f),
        first_row + EXPERTS_LIVE(block), h, f, tile.y, tile.x, shared, sg, lane);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_DOWN_PACK
kernel void routed_experts_down_pack(EXPERTS_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    if constexpr (!projection::packing_path<packets::W2>())
        return;
    EXPERTS_BLOCK(index / (uint(SEISMIC_DIM_T) / 4u));
    (void)source;
    threadgroup float peaks[projection::packing_peaks()];
    const routed::GroupedProducts in = EXPERTS_PRODUCT_ROWS;
    const projection::packing_bounds<0> bounds{};
    typedef projection::packing_codes<packets::W2> coded;
    projection::packing_operand<coded::centre, coded::folds, coded::sums, (coded::residue != 0 ? 1 : 0), 0>(in,
        EXPERTS_PACKING_OPERAND(uint(SEISMIC_DIM_F), coded::sums), bounds, uint(SEISMIC_DIM_B) * uint(SEISMIC_DIM_T),
        uint(SEISMIC_DIM_F), index, peaks, thread_index, sg, lane);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_DOWN_PACK_COEFFICIENTS
kernel void routed_experts_down_pack_coefficients(EXPERTS_ARGUMENTS, PROJECTION_TILE_ARGUMENTS,
    uint thread_index [[thread_index_in_threadgroup]]) {
    if constexpr (!projection::packing_path<packets::W2>())
        return;
    threadgroup float peaks[8];
    if (!routed::first_block(blocks, SEISMIC_BLOCKS_STRIDE_0, tile.y))
        return;
    EXPERTS_BLOCK(tile.y);
    EXPERTS_DOWN;
    const uint h = uint(SEISMIC_DIM_H), f = uint(SEISMIC_DIM_F);
    projection::packing_tile_coefficients(down, EXPERTS_PACKING(tile.y, 1, 0, h, f, packets::W2), f, tile.x, peaks,
        thread_index, sg, lane);
}
#endif

// A threadgroup's four simdgroups take the 32-token tile's four 32 weight
// rows, or coalesce consecutive 32-row blocks of the same expert into a
// 64-token (`PACK_TOKENS = 4`) tile over half the weight rows per threadgroup.
#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_DOWN_PACKED
template <uint DOWN_WEIGHTS_AHEAD>
kernel void routed_experts_down_packed(EXPERTS_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    if constexpr (!projection::packing_path<packets::W2>())
        return;
    const uint block = tile.y / BLOCK_TILES, first_row = block * uint(SEISMIC_DIM_T);
    const uint h = uint(SEISMIC_DIM_H), f = uint(SEISMIC_DIM_F);
    EXPERTS_BLOCK(block);
    EXPERTS_DOWN;
    if constexpr (BLOCK_TILES == 1u) {
        const uint role = routed::block_pair_role(tables, block);
        if (role != 0u) {
            const uint first_block = (role == routed::ORDINAL_PAIR_ODD) ? (block - 1u) : block;
            const uint pair_first_row = first_block * 32u;
            const uint pair_m_rows = (first_block + 1u) * 32u + EXPERTS_LIVE(first_block + 1u);
            const uint side = (role == routed::ORDINAL_PAIR_ODD) ? 1u : 0u;
            const routed::BlockOutput<ELEMENT_OF(SEISMIC_RESULT_0)> out{
                element::Out<ELEMENT_OF(SEISMIC_RESULT_0)>(output_cells),
                SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1, SEISMIC_RESULT_0_STRIDE_2, uint(SEISMIC_DIM_T)};
            projection::gemm_packed_tile<packets::W2, 1, 4, DOWN_WEIGHTS_AHEAD>(
                out, down, EXPERTS_PACKING(block, 1, 0, h, f, packets::W2),
                pair_m_rows, pair_first_row, tile.x * 128u + side * 64u + sg * 16u, h, f, lane);
            return;
        }
    }
    const auto out = routed::positioned(EXPERTS_OUTPUT(block), first_row);
    projection::gemm_packed_tile<packets::W2, 1, 2, DOWN_WEIGHTS_AHEAD>(
        out, down, EXPERTS_PACKING(block, 1, 0, h, f, packets::W2),
        first_row + EXPERTS_LIVE(block), tile.y * 32u, tile.x * 128u + sg * 32u, h, f, lane);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_EXPERTS_DOWN_UNPACKED
kernel void routed_experts_down_unpacked(EXPERTS_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    if constexpr (projection::packing_path<packets::W2>())
        return;
    PROJECTION_GEMM_SHARED(shared, 32, 64);
    const uint block = tile.y / BLOCK_TILES, first_row = block * uint(SEISMIC_DIM_T);
    EXPERTS_BLOCK(block);
    EXPERTS_DOWN;
    const auto in = routed::activation(product, SEISMIC_DIM_F, 1, uint(SEISMIC_DIM_F));
    const auto out = routed::positioned(EXPERTS_OUTPUT(block), first_row);
    projection::gemm_halves<packets::W2>(in, out, down, first_row + EXPERTS_LIVE(block), uint(SEISMIC_DIM_H),
        uint(SEISMIC_DIM_F), tile.y, tile.x, shared, sg, lane);
}
#endif
