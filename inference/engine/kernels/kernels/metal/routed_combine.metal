// Prefill combine. L1 (`routed_combine_expand`): the K1 paired GEMM of
// the shared expert's gate/up over all M rows with SiLU . mul into the
// `product` scratch. L2 (`routed_combine_down`): the K1 GEMM of
// the shared down projection into the `shared_output` scratch [M, H], in A.
// Each past 64 rows in a form of the grouped projection instead (`projection
// routed for metal`), dense's launches over every row: TALL (`relayout`,
// `tall`), INT8 (`int8_rows`, `int8_coefficients`, `int8`) or PACK
// (`pack`, `pack_coefficients`, `packed`, `unpacked`), the down's the same
// launches `down_` named.
// L3 (`routed_combine`): the grouped expert outputs of each row
// unpermuted in slot order, published as
//     residual + selected + shared * coefficient.
// An L3 thread owns 8 consecutive columns of one row: it reads each choice's
// position and score once and the choice's 8 expert outputs as one 16-byte
// load, so the gather moves whole rows of expert output.

#define KERNEL_W0 SEISMIC_SHARED_DOWN
#define KERNEL_W1 SEISMIC_SHARED_GATE
#define KERNEL_W2 SEISMIC_SHARED_UP
#include "lib/routed/routed.h"
#include "lib/projection/packing.h"
#include "lib/projection/operator.h"

// Threads of an L3 threadgroup (8 columns each).
constant constexpr uint combine_threads = 256;

#define COMBINE_ARGUMENTS                                                                                       \
    device const float *residual [[buffer(SEISMIC_BUFFER_RESIDUAL)]],                                           \
    device const float *inject [[buffer(SEISMIC_BUFFER_INJECT)]],                                               \
    device const uchar *expert_output [[buffer(SEISMIC_BUFFER_EXPERT_OUTPUT)]],                                 \
    device const int *inverse [[buffer(SEISMIC_BUFFER_INVERSE)]],                                               \
    device const float *scores [[buffer(SEISMIC_BUFFER_SCORES)]],                                               \
    device const uchar *normalized [[buffer(SEISMIC_BUFFER_NORMALIZED)]],                                       \
    device const float *coefficient [[buffer(SEISMIC_BUFFER_COEFFICIENT)]],                                     \
    device const uchar *shared_gate [[buffer(SEISMIC_BUFFER_SHARED_GATE)]],                                     \
    device const uchar *shared_up [[buffer(SEISMIC_BUFFER_SHARED_UP)]],                                         \
    device const uchar *shared_down [[buffer(SEISMIC_BUFFER_SHARED_DOWN)]],                                     \
    SEISMIC_RESULT_0_PARAM(value_cells),                                                                        \
    device uchar *shared_product [[buffer(SEISMIC_BUFFER_SCRATCH_PRODUCT)]],                                    \
    device uchar *shared_output [[buffer(SEISMIC_BUFFER_SCRATCH_SHARED_OUTPUT)]],                               \
    device uchar *operand [[buffer(SEISMIC_BUFFER_SCRATCH_OPERAND)]],                                           \
    device uchar *tables [[buffer(SEISMIC_BUFFER_SCRATCH_TABLES)]],                                             \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

// The expansion: every row against the shared gate and up weights, into
// the products.
#define COMBINE_EXPANSION                                                                                       \
    const uint m = uint(SEISMIC_DIM_M), n = uint(SEISMIC_DIM_S), k = uint(SEISMIC_DIM_H);                       \
    const auto in = routed::activation(normalized, SEISMIC_NORMALIZED_STRIDE_0, SEISMIC_NORMALIZED_STRIDE_1, k);  \
    const auto gate = routed::weights<packets::W1>(shared_gate, KERNEL_W1_LAYOUT(SEISMIC_DIM_H), 0, k);         \
    const auto up = routed::weights<packets::W2>(shared_up, KERNEL_W2_LAYOUT(SEISMIC_DIM_H), 0, k);             \
    const projection::SiluMul<routed::Act> out{element::out_over_scratch<routed::Act>(shared_product), SEISMIC_DIM_S, 1}

// The down: the products against the shared down weights, into
// `shared_output`.
#define COMBINE_DOWN                                                                                            \
    const uint m = uint(SEISMIC_DIM_M), n = uint(SEISMIC_DIM_H), k = uint(SEISMIC_DIM_S);                       \
    const auto in = routed::activation(shared_product, SEISMIC_DIM_S, 1, k);                                    \
    const auto down = routed::weights<packets::W0>(shared_down, KERNEL_W0_LAYOUT(SEISMIC_DIM_S), 0, k);         \
    const projection::Store<routed::Act> out{element::out_over_scratch<routed::Act>(shared_output), SEISMIC_DIM_H, 1, 0}

// The forms' scratch (`routed.h`): the GEMM's operand over every row
// (128-row padded), with the table of weight `weight` (packets W) of the
// GEMM's `weights`, of `n` rows by `k` columns; the operand alone for its own
// pass, laid out with each run's code sums where `sums`.
#define COMBINE_OPERAND_ROWS ((ulong(SEISMIC_DIM_M) + 127u) / 128u * 128u)
#define COMBINE_INT8(weights, weight)                                                                           \
    routed::int8_forms(operand, COMBINE_OPERAND_ROWS,                                                           \
        routed::weight_table(tables, 0, 0, weights, weight, routed::int8_table_bytes(n, k)), n, k)
#define COMBINE_PACKING(weights, weight, W)                                                                     \
    routed::packing_forms<projection::packing_scratch>(operand, COMBINE_OPERAND_ROWS,                           \
        routed::weight_table(tables, 0, 0, weights, weight,                                                     \
            routed::packing_table_bytes(n, k, projection::packing_codes<W>::sums)),                             \
        n, k, projection::packing_codes<W>::sums)
#define COMBINE_INT8_OPERAND routed::int8_forms(operand, COMBINE_OPERAND_ROWS, tables, 0, k)
#define COMBINE_PACKING_OPERAND(sums)                                                                           \
    routed::packing_forms<projection::packing_scratch>(operand, COMBINE_OPERAND_ROWS, tables, 0, k, sums)

#ifdef SEISMIC_FORMING_ROUTED_COMBINE_EXPAND
template <uint TILE_M, uint TILE_N>
kernel void routed_combine_expand(COMBINE_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_GEMM_SHARED(tile_memory, TILE_M, TILE_N);
    COMBINE_EXPANSION;
    projection::gemm_paired<packets::W1, packets::W2, TILE_M, TILE_N>(in, out, gate, up, m, n, k, tile.y, tile.x,
        tile_memory, sg, lane);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_COMBINE_RELAYOUT
kernel void routed_combine_relayout(COMBINE_ARGUMENTS, PROJECTION_ITEM_ARGUMENTS) {
    COMBINE_EXPANSION;
    projection::tall_relayout(in, operand, m, k, position.x);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_COMBINE_TALL
template <uint TALL_M, uint TALL_K, uint STAGERS>
kernel void routed_combine_tall(COMBINE_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_GEMM_TALL_SHARED(shared, TALL_K);
    COMBINE_EXPANSION;
    projection::gemm_tall_paired<packets::W1, packets::W2, TALL_M, TALL_K, STAGERS>(
        projection::tall_operand(in, operand), out, gate, up, m, n, k, tile.y, tile.x, shared, sg, lane);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_COMBINE_INT8_ROWS
kernel void routed_combine_int8_rows(COMBINE_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    if constexpr (!projection::int8_path<packets::W1, packets::W2>())
        return;
    COMBINE_EXPANSION;
    projection::int8_quantize<projection::int8_codes<packets::W1>::interleaved>(in, COMBINE_INT8_OPERAND, m, k,
        index, thread_index, lane, 1.0f);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_COMBINE_INT8_COEFFICIENTS
kernel void routed_combine_int8_coefficients(COMBINE_ARGUMENTS, PROJECTION_ITEM_ARGUMENTS) {
    if constexpr (!projection::int8_path<packets::W1, packets::W2>())
        return;
    COMBINE_EXPANSION;
    projection::int8_coefficients(gate, COMBINE_INT8(2, 0), n, k, position.x);
    projection::int8_coefficients(up, COMBINE_INT8(2, 1), n, k, position.x);
}
#endif

// The default form's tile only where the launch takes that form:
// threadgroup memory a kernel does not use still slows its products.
#ifdef SEISMIC_FORMING_ROUTED_COMBINE_INT8
kernel void routed_combine_int8(COMBINE_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    constexpr bool PATH = projection::int8_path<packets::W1, packets::W2>();
    threadgroup float exchange[PATH ? 2 * 32 * 32 : 1];
    threadgroup float4 shared_words[PATH ? 1 : projection::gemm_tile<64, 64>::bytes / 16];
    threadgroup uchar *shared = reinterpret_cast<threadgroup uchar *>(shared_words);
    COMBINE_EXPANSION;
    projection::gemm_int8_paired<packets::W1, packets::W2, PATH, 64>(in, out, gate, up, COMBINE_INT8(2, 0),
        COMBINE_INT8(2, 1), m, n, k, tile.y, tile.x, shared, exchange, sg, lane);
}
#endif

// The operand serves both weights, which carry no packing bound; on a device
// without the packed path, the staged tiles of the default form.
#ifdef SEISMIC_FORMING_ROUTED_COMBINE_PACK
kernel void routed_combine_pack(COMBINE_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    if constexpr (!projection::packing_path<packets::W1, packets::W2>())
        return;
    threadgroup float4 peaks[8];
    COMBINE_EXPANSION;
    typedef projection::packing_shared<1, packets::W1, 1, packets::W2, 1> coded;
    const projection::packing_bounds<0> bounds{};
    projection::packing_operand<coded::centre, coded::folds, coded::sums, coded::residue, 0>(in,
        COMBINE_PACKING_OPERAND(coded::sums), bounds, m, k, index, peaks, thread_index, sg, lane);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_COMBINE_PACK_COEFFICIENTS
kernel void routed_combine_pack_coefficients(COMBINE_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    if constexpr (!projection::packing_path<packets::W1, packets::W2>())
        return;
    threadgroup float peaks0[8];
    threadgroup float peaks1[8];
    COMBINE_EXPANSION;
    projection::packing_tile_coefficients(gate, COMBINE_PACKING(2, 0, packets::W1), k, index, peaks0, thread_index,
        sg, lane);
    projection::packing_tile_coefficients(up, COMBINE_PACKING(2, 1, packets::W2), k, index, peaks1, thread_index,
        sg, lane);
}
#endif

// The operand's rows padded to the paired tile's 64.
#ifdef SEISMIC_FORMING_ROUTED_COMBINE_PACKED
template <uint PACK_TOKENS, uint WEIGHTS_AHEAD>
kernel void routed_combine_packed(COMBINE_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    if constexpr (!projection::packing_path<packets::W1, packets::W2>())
        return;
    threadgroup float4 exchange[projection::packing_simdgroups / 2 * projection::packing_exchange];
    COMBINE_EXPANSION;
    projection::gemm_packed_paired<packets::W1, packets::W2, 1, PACK_TOKENS, WEIGHTS_AHEAD>(out, gate, up,
        COMBINE_PACKING(2, 0, packets::W1), COMBINE_PACKING(2, 1, packets::W2), m, (m + 63u) / 64u * 64u, n, k,
        index,
        projection::packing_simdgroups, exchange, sg, lane);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_COMBINE_UNPACKED
kernel void routed_combine_unpacked(COMBINE_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    if constexpr (projection::packing_path<packets::W1, packets::W2>())
        return;
    PROJECTION_GEMM_SHARED(shared, 64, 64);
    COMBINE_EXPANSION;
    projection::gemm_paired<packets::W1, packets::W2, 64, 64>(in, out, gate, up, m, n, k, tile.y, tile.x, shared,
        sg, lane);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_COMBINE_DOWN
template <uint TILE_M, uint TILE_N>
kernel void routed_combine_down(COMBINE_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_GEMM_SHARED(tile_memory, TILE_M, TILE_N);
    COMBINE_DOWN;
    projection::gemm<packets::W0, TILE_M, TILE_N>(in, out, down, m, n, k, tile.y, tile.x, tile_memory, sg, lane);
}
#endif

// The down GEMM's forms, over the products.
#ifdef SEISMIC_FORMING_ROUTED_COMBINE_DOWN_RELAYOUT
kernel void routed_combine_down_relayout(COMBINE_ARGUMENTS, PROJECTION_ITEM_ARGUMENTS) {
    COMBINE_DOWN;
    projection::tall_relayout(in, operand, m, k, position.x);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_COMBINE_DOWN_TALL
template <uint DOWN_TALL_M, uint DOWN_TALL_K, uint DOWN_STAGERS>
kernel void routed_combine_down_tall(COMBINE_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_GEMM_TALL_SHARED(shared, DOWN_TALL_K);
    COMBINE_DOWN;
    projection::gemm_tall<packets::W0, DOWN_TALL_M, DOWN_TALL_K, DOWN_STAGERS>(projection::tall_operand(in,
        operand), out, down, m, n, k, tile.y, tile.x, shared, sg, lane);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_COMBINE_DOWN_INT8_ROWS
kernel void routed_combine_down_int8_rows(COMBINE_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    if constexpr (!projection::int8_path<packets::W0>())
        return;
    COMBINE_DOWN;
    projection::int8_quantize<projection::int8_codes<packets::W0>::interleaved>(in, COMBINE_INT8_OPERAND, m, k,
        index, thread_index, lane, 1.0f);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_COMBINE_DOWN_INT8_COEFFICIENTS
kernel void routed_combine_down_int8_coefficients(COMBINE_ARGUMENTS, PROJECTION_ITEM_ARGUMENTS) {
    if constexpr (!projection::int8_path<packets::W0>())
        return;
    COMBINE_DOWN;
    projection::int8_coefficients(down, COMBINE_INT8(1, 0), n, k, position.x);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_COMBINE_DOWN_INT8
template <uint INT8_TILE>
kernel void routed_combine_down_int8(COMBINE_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    constexpr bool PATH = projection::int8_path<packets::W0>();
    threadgroup float4 shared_words[PATH ? 1 : projection::gemm_tile<64, 64>::bytes / 16];
    threadgroup uchar *shared = reinterpret_cast<threadgroup uchar *>(shared_words);
    COMBINE_DOWN;
    projection::gemm_int8<packets::W0, INT8_TILE, PATH>(in, out, down, COMBINE_INT8(1, 0), m, n, k, tile.x, tile.y,
        shared, sg, lane);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_COMBINE_DOWN_PACK
kernel void routed_combine_down_pack(COMBINE_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    if constexpr (!projection::packing_path<packets::W0>())
        return;
    threadgroup float4 peaks[8];
    COMBINE_DOWN;
    typedef projection::packing_codes<packets::W0> coded;
    const projection::packing_bounds<0> bounds{};
    projection::packing_operand<coded::centre, coded::folds, coded::sums, (coded::residue != 0 ? 1 : 0), 0>(in,
        COMBINE_PACKING_OPERAND(coded::sums), bounds, m, k, index, peaks, thread_index, sg, lane);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_COMBINE_DOWN_PACK_COEFFICIENTS
kernel void routed_combine_down_pack_coefficients(COMBINE_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    if constexpr (!projection::packing_path<packets::W0>())
        return;
    threadgroup float peaks[8];
    COMBINE_DOWN;
    projection::packing_tile_coefficients(
        down, COMBINE_PACKING(1, 0, packets::W0), k, index, peaks, thread_index, sg, lane);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_COMBINE_DOWN_PACKED
template <uint DOWN_PACK_TOKENS, uint DOWN_WEIGHTS_AHEAD>
kernel void routed_combine_down_packed(COMBINE_ARGUMENTS, PROJECTION_THREADGROUP_ARGUMENTS) {
    if constexpr (!projection::packing_path<packets::W0>())
        return;
    COMBINE_DOWN;
    projection::gemm_packed<packets::W0, 1, DOWN_PACK_TOKENS, DOWN_WEIGHTS_AHEAD>(
        out, down, COMBINE_PACKING(1, 0, packets::W0),
        m, (m + 127u) / 128u * 128u, n, k, index * projection::packing_simdgroups + sg, lane);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_COMBINE_DOWN_UNPACKED
kernel void routed_combine_down_unpacked(COMBINE_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    if constexpr (projection::packing_path<packets::W0>())
        return;
    PROJECTION_GEMM_SHARED(shared, 64, 64);
    COMBINE_DOWN;
    projection::gemm<packets::W0, 64, 64>(in, out, down, m, n, k, tile.y, tile.x, shared, sg, lane);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_COMBINE
kernel void routed_combine(COMBINE_ARGUMENTS,
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    typedef routed::Act A;
    typedef typename A::storage S;
    const element::Out<ELEMENT_OF(SEISMIC_RESULT_0)> value(value_cells);
    const uint m = group.y;
    const uint n = 8u * (group.x * combine_threads + thread_index);
    const uint columns = uint(SEISMIC_DIM_H);
    if (n >= columns)
        return;
    const bool vector = SEISMIC_EXPERT_OUTPUT_STRIDE_2 == 1 && (SEISMIC_EXPERT_OUTPUT_STRIDE_1 & 7u) == 0
        && (SEISMIC_EXPERT_OUTPUT_STRIDE_0 & 7u) == 0;
    float4 selected_even = float4(0.0f), selected_odd = float4(0.0f);
    for (ulong k = 0; k < SEISMIC_DIM_K; ++k) {
        const ulong position = ulong(inverse[m * SEISMIC_INVERSE_STRIDE_0 + k * SEISMIC_INVERSE_STRIDE_1]);
        const float score = scores[m * SEISMIC_SCORES_STRIDE_0 + k * SEISMIC_SCORES_STRIDE_1];
        device const S *row = reinterpret_cast<device const S *>(expert_output)
            + (position / SEISMIC_DIM_T) * SEISMIC_EXPERT_OUTPUT_STRIDE_0
            + (position % SEISMIC_DIM_T) * SEISMIC_EXPERT_OUTPUT_STRIDE_1;
        float4 even, odd;
        projection::load8_storage<A>(row, SEISMIC_EXPERT_OUTPUT_STRIDE_2, n, columns, vector, even, odd);
        selected_even = metal::fma(float4(score), even, selected_even);
        selected_odd = metal::fma(float4(score), odd, selected_odd);
    }
    float4 shared_even, shared_odd;
    projection::load8_storage<A>(reinterpret_cast<device const S *>(shared_output) + ulong(m) * columns, 1, n,
        columns, (columns & 7u) == 0, shared_even, shared_odd);
    const float c = coefficient[m * SEISMIC_COEFFICIENT_STRIDE_0];
    const float selected[8] = {selected_even.x, selected_odd.x, selected_even.y, selected_odd.y,
        selected_even.z, selected_odd.z, selected_even.w, selected_odd.w};
    const float projected[8] = {shared_even.x, shared_odd.x, shared_even.y, shared_odd.y, shared_even.z,
        shared_odd.z, shared_even.w, shared_odd.w};
    for (uint i = 0; i < 8u && n + i < columns; ++i)
        projection::routed_join<SEISMIC_DIM_SN, (SEISMIC_DIM_IN != 0)>(value.offset(m * SEISMIC_RESULT_0_STRIDE_0),
            SEISMIC_RESULT_0_STRIDE_1, residual + m * SEISMIC_RESIDUAL_STRIDE_0, SEISMIC_RESIDUAL_STRIDE_1,
            inject + m * SEISMIC_INJECT_STRIDE_0, columns, n + i, selected[i], projected[i] * c);
}
#endif
