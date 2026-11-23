// Decode expansion (M <= VERIFY_ROWS): the K selected experts' gate/up projections and
// the shared expert's, each with the SiLU . mul epilogue, in one launch; a
// choice's gate/up accumulators carry its expert's `gate_scale` and
// `up_scale` where those ports are present (static GS, US).
// Threadgroup row y < M * K is choice (m, k): a K1 paired GEMV over the
// feature tile `x` of expert routes[m, k]'s gate and up rows, or, gathered
// (`routed_expand_gathered`, `routed::gather`), the rows of every choice of
// its expert where it is the expert's first. Row y = M * K is the shared
// expert, whose weights feed all M activation rows. The expert stacks are
// record ports, bound as their records' address tables, as every routed
// launch's are: the expand form decodes a held layer's stacks and a whole
// bank's home records alike, the expert itself its record. Nothing lands in
// either while it runs, so plain loads read them.

#define KERNEL_A SEISMIC_ELEMENT_A
#define PACKETS_CONFIG_PLAIN_RECORDS
#define KERNEL_W0 SEISMIC_EXPERT_GATE
#define KERNEL_W1 SEISMIC_EXPERT_UP
#define KERNEL_W2 SEISMIC_SHARED_GATE
#define KERNEL_W3 SEISMIC_SHARED_UP
#include "lib/routed/routed.h"
#include "lib/projection/gemv.h"

// The rows the experts' and the shared expert's weights take (the
// normalized rows, or NVFP4 operands), each pair's own, formed while its
// GEMV stages them.
constant constexpr bool EXPERT_NVFP4 =
    projection::OperandRecipe<packets::W0_OPERANDS_NVFP4, packets::W1_OPERANDS_NVFP4>::NVFP4;
constant constexpr bool SHARED_NVFP4 =
    projection::OperandRecipe<packets::W2_OPERANDS_NVFP4, packets::W3_OPERANDS_NVFP4>::NVFP4;

#define ROUTED_EXPAND_PORTS                                                                         \
    device const uchar *normalized [[buffer(SEISMIC_BUFFER_NORMALIZED)]],                           \
    device const int *routes [[buffer(SEISMIC_BUFFER_ROUTES)]],                                     \
    device const ulong *expert_gate [[buffer(SEISMIC_BUFFER_EXPERT_GATE)]],                         \
    device const ulong *expert_up [[buffer(SEISMIC_BUFFER_EXPERT_UP)]],                             \
    device const uchar *shared_gate [[buffer(SEISMIC_BUFFER_SHARED_GATE)]],                         \
    device const uchar *shared_up [[buffer(SEISMIC_BUFFER_SHARED_UP)]],                             \
    device const float *gate_scale [[buffer(SEISMIC_BUFFER_GATE_SCALE)]],                           \
    device const float *up_scale [[buffer(SEISMIC_BUFFER_UP_SCALE)]],                               \
    SEISMIC_RESULT_0_PARAM(expert_product_cells),                                                   \
    SEISMIC_RESULT_1_PARAM(shared_product_cells),                                                   \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],                                 \
    threadgroup uchar *shared [[threadgroup(0)]],                                                   \
    uint2 group [[threadgroup_position_in_grid]],                                                   \
    uint simdgroups [[simdgroups_per_threadgroup]],                                                 \
    uint sg [[simdgroup_index_in_threadgroup]],                                                     \
    uint lane [[thread_index_in_simdgroup]]

// The results' owner handles.
#define ROUTED_EXPAND_RESULTS                                                                       \
    const element::Out<ELEMENT_OF(SEISMIC_RESULT_0)> expert_product(expert_product_cells);           \
    const element::Out<ELEMENT_OF(SEISMIC_RESULT_1)> shared_product(shared_product_cells)

// The shared expert's row (y = M * K): its weights over every activation row
// (a macro: its operand staging is threadgroup memory of the kernel's scope).
#define ROUTED_EXPAND_SHARED_ROW(R, L, FOR_ROWS)                                                        \
    if (ulong(tile) * simdgroups * R * (32u / L) >= SEISMIC_DIM_S) return;                              \
    const auto gate = routed::weights<packets::W2>(shared_gate, KERNEL_W2_LAYOUT(SEISMIC_DIM_H), 0,     \
        SEISMIC_DIM_H);                                                                                 \
    const auto up = routed::weights<packets::W3>(shared_up, KERNEL_W3_LAYOUT(SEISMIC_DIM_H), 0,         \
        SEISMIC_DIM_H);                                                                                 \
    const uint rows = uint(SEISMIC_DIM_M);                                                              \
    PROJECTION_OPERANDS_SHARED(globals, SHARED_NVFP4);                                                  \
    for (uint base = 0u; base < rows; base += 4u) {                                                     \
        const uint chunk = min(4u, rows - base);                                                        \
        if (base > 0u)                                                                                  \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                            \
        const auto in = routed::activation(normalized + ulong(base) * SEISMIC_NORMALIZED_STRIDE_0 * A::bytes, \
            SEISMIC_NORMALIZED_STRIDE_0, SEISMIC_NORMALIZED_STRIDE_1, uint(SEISMIC_DIM_H));             \
        const projection::SiluMul<A> out{shared_product.offset(ulong(base) * SEISMIC_RESULT_1_STRIDE_0), \
            SEISMIC_RESULT_1_STRIDE_0, SEISMIC_RESULT_1_STRIDE_1};                                      \
        const auto x = projection::operands<SHARED_NVFP4>::rows(in, chunk, uint(SEISMIC_DIM_H), globals, \
            simdgroups, sg, lane);                                                                      \
        FOR_ROWS(chunk,                                                                                 \
            (projection::gemv_paired_runtime<packets::W2, packets::W3, R, MAXM, L, 128u>(x, out, gate,  \
                up, chunk, SEISMIC_DIM_S, SEISMIC_DIM_H, tile, shared, simdgroups, sg, lane)));         \
    }

#ifdef SEISMIC_FORMING_ROUTED_EXPAND_CHOICE
template <uint ROWS, uint LANES>
kernel void routed_expand_choice(ROUTED_EXPAND_PORTS) {
    constexpr uint R = ROWS;
    constexpr uint L = LANES;
    typedef routed::Act A;
    ROUTED_EXPAND_RESULTS;
    const uint tile = group.x;
    if (group.y < SEISMIC_DIM_M * SEISMIC_DIM_K) {
        if (ulong(tile) * simdgroups * R * (32u / L) >= SEISMIC_DIM_F) return;
        const ulong m = group.y / SEISMIC_DIM_K, k = group.y % SEISMIC_DIM_K;
        const int expert = routes[m * SEISMIC_ROUTES_STRIDE_0 + k * SEISMIC_ROUTES_STRIDE_1];
        const auto in = routed::activation(normalized + m * SEISMIC_NORMALIZED_STRIDE_0 * A::bytes,
            SEISMIC_NORMALIZED_STRIDE_0, SEISMIC_NORMALIZED_STRIDE_1, uint(SEISMIC_DIM_H));
        const auto gate = routed::record_weights<packets::W0>(expert_gate, expert, KERNEL_W0_LAYOUT(SEISMIC_DIM_H),
            SEISMIC_DIM_H);
        const auto up = routed::record_weights<packets::W1>(expert_up, expert, KERNEL_W1_LAYOUT(SEISMIC_DIM_H),
            SEISMIC_DIM_H);
        const auto out = projection::scaling<(SEISMIC_DIM_GS + SEISMIC_DIM_US != 0)>::wrap(
            projection::SiluMul<A>{
                expert_product.offset(m * SEISMIC_RESULT_0_STRIDE_0 + k * SEISMIC_RESULT_0_STRIDE_1), 0,
                SEISMIC_RESULT_0_STRIDE_2},
            projection::scale_factor(gate_scale, SEISMIC_DIM_GS, SEISMIC_GATE_SCALE_STRIDE_0, ulong(expert)),
            projection::scale_factor(up_scale, SEISMIC_DIM_US, SEISMIC_UP_SCALE_STRIDE_0, ulong(expert)));
        PROJECTION_OPERANDS_SHARED(globals, EXPERT_NVFP4);
        const auto x = projection::operands<EXPERT_NVFP4>::rows(in, 1u, uint(SEISMIC_DIM_H), globals, simdgroups,
            sg, lane);
        projection::gemv_paired_runtime<packets::W0, packets::W1, R, 1, L>(x, out, gate, up, 1,
            SEISMIC_DIM_F, SEISMIC_DIM_H, tile, shared, simdgroups, sg, lane);
        return;
    }
    ROUTED_EXPAND_SHARED_ROW(R, L, PROJECTION_FOR_ROWS)
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_EXPAND_GATHER
template <uint ROWS, uint LANES>
kernel void routed_expand_gather(
    device const int *routes [[buffer(SEISMIC_BUFFER_ROUTES)]],
    device routed::Worklist *worklist [[buffer(SEISMIC_BUFFER_SCRATCH_WORKLIST)]],
    device int *gather_grid [[buffer(SEISMIC_BUFFER_SCRATCH_GATHER_GRID)]],
    device int *gather_one_grid [[buffer(SEISMIC_BUFFER_SCRATCH_GATHER_ONE_GRID)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup int cache[routed::WORKLIST_CACHE_WORDS];
    const uint max_n = uint(SEISMIC_DIM_F) > uint(SEISMIC_DIM_S) ? uint(SEISMIC_DIM_F) : uint(SEISMIC_DIM_S);
    const uint per_group = uint(SEISMIC_RUNTIME_SIMDGROUPS) * ROWS * (32u / LANES);
    const uint x_groups = (max_n + per_group - 1u) / per_group;
    routed::build_worklist(routes, SEISMIC_ROUTES_STRIDE_0, SEISMIC_ROUTES_STRIDE_1, uint(SEISMIC_DIM_M),
        uint(SEISMIC_DIM_K), x_groups, 1u, worklist, gather_grid, gather_one_grid, cache, lane);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_EXPAND_GATHERED_ONE
template <uint ROWS, uint LANES>
kernel void routed_expand_gathered_one(
    ROUTED_EXPAND_PORTS,
    device const routed::Worklist *worklist [[buffer(SEISMIC_BUFFER_SCRATCH_WORKLIST)]]) {
    constexpr uint R = ROWS;
    constexpr uint L = LANES;
    typedef routed::Act A;
    ROUTED_EXPAND_RESULTS;
    const uint tile = group.x;
    if (ulong(tile) * simdgroups * R * (32u / L) >= SEISMIC_DIM_F) return;
    device const routed::WorkEntry &entry = worklist->entries[group.y];
    const uint packed = entry.packed[0];
    const ulong m = packed >> 16u, k = packed & 0xffffu;
    const int expert = entry.expert;
    const auto in = routed::activation(normalized + m * SEISMIC_NORMALIZED_STRIDE_0 * A::bytes,
        SEISMIC_NORMALIZED_STRIDE_0, SEISMIC_NORMALIZED_STRIDE_1, uint(SEISMIC_DIM_H));
    const auto gate = routed::record_weights<packets::W0>(expert_gate, expert, KERNEL_W0_LAYOUT(SEISMIC_DIM_H),
        SEISMIC_DIM_H);
    const auto up = routed::record_weights<packets::W1>(expert_up, expert, KERNEL_W1_LAYOUT(SEISMIC_DIM_H),
        SEISMIC_DIM_H);
    const auto out = projection::scaling<(SEISMIC_DIM_GS + SEISMIC_DIM_US != 0)>::wrap(
        projection::SiluMul<A>{
            expert_product.offset(m * SEISMIC_RESULT_0_STRIDE_0 + k * SEISMIC_RESULT_0_STRIDE_1), 0,
            SEISMIC_RESULT_0_STRIDE_2},
        projection::scale_factor(gate_scale, SEISMIC_DIM_GS, SEISMIC_GATE_SCALE_STRIDE_0, ulong(expert)),
        projection::scale_factor(up_scale, SEISMIC_DIM_US, SEISMIC_UP_SCALE_STRIDE_0, ulong(expert)));
    PROJECTION_OPERANDS_SHARED(globals, EXPERT_NVFP4);
    const auto x = projection::operands<EXPERT_NVFP4>::rows(in, 1u, uint(SEISMIC_DIM_H), globals, simdgroups, sg, lane);
    projection::gemv_paired_runtime<packets::W0, packets::W1, R, 1, L>(x, out, gate, up, 1,
        SEISMIC_DIM_F, SEISMIC_DIM_H, tile, shared, simdgroups, sg, lane);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_EXPAND_GATHERED
template <uint ROWS, uint LANES>
kernel void routed_expand_gathered(
    ROUTED_EXPAND_PORTS,
    device const routed::Worklist *worklist [[buffer(SEISMIC_BUFFER_SCRATCH_WORKLIST)]]) {
    constexpr uint R = ROWS;
    constexpr uint L = LANES;
    typedef routed::Act A;
    ROUTED_EXPAND_RESULTS;
    const uint tile = group.x;
    if (group.y < worklist->multi_count) {
        if (ulong(tile) * simdgroups * R * (32u / L) >= SEISMIC_DIM_F) return;
        const routed::Gathered chosen = routed::unpack_work_entry(
            worklist->entries[worklist->capacity - 1u - group.y], SEISMIC_RESULT_0_STRIDE_0,
            SEISMIC_RESULT_0_STRIDE_1);
        const auto gate = routed::record_weights<packets::W0>(expert_gate, chosen.expert,
            KERNEL_W0_LAYOUT(SEISMIC_DIM_H), SEISMIC_DIM_H);
        const auto up = routed::record_weights<packets::W1>(expert_up, chosen.expert,
            KERNEL_W1_LAYOUT(SEISMIC_DIM_H), SEISMIC_DIM_H);
        const float gate_sc = projection::scale_factor(gate_scale, SEISMIC_DIM_GS, SEISMIC_GATE_SCALE_STRIDE_0,
            ulong(chosen.expert));
        const float up_sc = projection::scale_factor(up_scale, SEISMIC_DIM_US, SEISMIC_UP_SCALE_STRIDE_0,
            ulong(chosen.expert));
        const uint rows = chosen.count;
        PROJECTION_OPERANDS_SHARED(globals, EXPERT_NVFP4);
        for (uint base = 0u; base < rows; base += 4u) {
            const uint chunk = min(4u, rows - base);
            if (base > 0u)
                threadgroup_barrier(mem_flags::mem_threadgroup);
            const projection::Plain<A, routed::Listed> in{normalized, SEISMIC_NORMALIZED_STRIDE_0,
                SEISMIC_NORMALIZED_STRIDE_1, uint(SEISMIC_DIM_H), routed::Listed{chosen.rows + base}};
            const auto out = projection::scaling<(SEISMIC_DIM_GS + SEISMIC_DIM_US != 0)>::wrap(
                routed::ChoiceRows<projection::SiluMul<A>>{
                    projection::SiluMul<A>{expert_product, 1, SEISMIC_RESULT_0_STRIDE_2}, chosen.products + base},
                gate_sc, up_sc);
            const auto x = projection::operands<EXPERT_NVFP4>::rows(in, chunk, uint(SEISMIC_DIM_H), globals,
                simdgroups, sg, lane);
            PROJECTION_FOR_ROWS_24(chunk,
                (projection::gemv_paired_runtime<packets::W0, packets::W1, R, MAXM, L, 128u>(x, out, gate, up, chunk,
                    SEISMIC_DIM_F, SEISMIC_DIM_H, tile, shared, simdgroups, sg, lane)));
        }
        return;
    }
    ROUTED_EXPAND_SHARED_ROW(R, L, PROJECTION_FOR_ROWS_24)
}
#endif
