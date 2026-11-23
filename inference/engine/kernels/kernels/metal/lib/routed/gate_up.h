// Decode expansion of gated experts (M <= VERIFY_ROWS), shared by
// `routed_gate_up` and `routed_gate_up_part`: threadgroup row y = choice
// (m, k), a K1 paired GEMV over the feature tile of expert routes[m, k]'s gate
// and up rows (its record, `routed::decode_source`: a bank record, or a host
// tier record read in place; a resident layer's, its expert's record) with the
// activation-generic GLU epilogue, the accumulators scaled by the expert's
// `gate_scale`/`up_scale` where those ports are present (static GS, US). Every
// active choice with no record zeroes its row; gathered (GATHERED,
// `routed::gather`), each active record is read once across the active choices
// sharing it.
//
// The includer defines KERNEL_W0, KERNEL_W1, optional PROJECTION_GLU_LIMIT,
// ROUTED_SOURCE_ROWS, ROUTED_GATE_UP_PRODUCT_STRIDE_0..2 and
// ROUTED_GATE_UP_ACTIVE(c, source) (whether flat choice c, reading `source`,
// runs in this pass).
#pragma once
#include "routed.h"
#include "../projection/gemv.h"

// The row both weights take (the normalized row, or NVFP4 operands), formed
// while the GEMV stages it.
constant constexpr bool NVFP4 =
    projection::OperandRecipe<packets::W0_OPERANDS_NVFP4, packets::W1_OPERANDS_NVFP4>::NVFP4;
typedef projection::operands<NVFP4> Operands;

// The per-choice body (`routed::decode_source`, `routed::expert_matrix`).
#define ROUTED_GATE_UP_BODY                                                                                            \
    typedef routed::Act A;                                                                                             \
    const uint tile = group.x;                                                                                         \
    if (ulong(tile) * simdgroups * ROWS * (32u / LANES) >= SEISMIC_DIM_F)                                              \
        return;                                                                                                        \
    const ulong m = group.y / SEISMIC_DIM_K, k = group.y % SEISMIC_DIM_K;                                              \
    const int expert = routes[m * SEISMIC_ROUTES_STRIDE_0 + k * SEISMIC_ROUTES_STRIDE_1];                              \
    const slab::Source source = routed::decode_source(sources, ROUTED_SOURCE_ROWS, SEISMIC_SOURCES_STRIDE_1,           \
        SEISMIC_SOURCES_STRIDE_2, m, k, expert);                                                                       \
    if (!ROUTED_GATE_UP_ACTIVE(uint(group.y), source))                                                                 \
        return;                                                                                                        \
    const element::Out<A> choice = element::Out<A>(product_cells).offset(                                              \
        m * ROUTED_GATE_UP_PRODUCT_STRIDE_0 + k * ROUTED_GATE_UP_PRODUCT_STRIDE_1);                                    \
    /* A choice reading no record (the CPU's, a skipped front choice, or a                                             \
       failed one): its first threadgroup zeroes its row. */                                                           \
    if (source.none()) {                                                                                               \
        if (tile == 0)                                                                                                 \
            routed::zero_row(choice, ROUTED_GATE_UP_PRODUCT_STRIDE_2, uint(SEISMIC_DIM_F), sg * 32 + lane,             \
                simdgroups * 32);                                                                                      \
        return;                                                                                                        \
    }                                                                                                                  \
    const auto in = routed::activation(normalized + m * SEISMIC_NORMALIZED_STRIDE_0 * A::bytes,                        \
        SEISMIC_NORMALIZED_STRIDE_0, SEISMIC_NORMALIZED_STRIDE_1, uint(SEISMIC_DIM_H));                                \
    const auto gate = routed::expert_matrix<packets::W0>(routed::Experts{expert_gate, tier_gate}, source,              \
        KERNEL_W0_LAYOUT(SEISMIC_DIM_H), SEISMIC_DIM_H);                                                               \
    const auto up = routed::expert_matrix<packets::W1>(routed::Experts{expert_up, tier_up}, source,                    \
        KERNEL_W1_LAYOUT(SEISMIC_DIM_H), SEISMIC_DIM_H);                                                               \
    const auto out = projection::scaling<(SEISMIC_DIM_GS + SEISMIC_DIM_US != 0)>::wrap(                                \
        projection::Glu<A>{choice, 0, ROUTED_GATE_UP_PRODUCT_STRIDE_2, int(SEISMIC_PARAM_ACTIVATION)},                 \
        projection::scale_factor(gate_scale, SEISMIC_DIM_GS, SEISMIC_GATE_SCALE_STRIDE_0, ulong(expert)),              \
        projection::scale_factor(up_scale, SEISMIC_DIM_US, SEISMIC_UP_SCALE_STRIDE_0, ulong(expert)));                 \
    PROJECTION_OPERANDS_SHARED(globals, NVFP4);                                                                        \
    const auto x = Operands::rows(in, 1u, uint(SEISMIC_DIM_H), globals, simdgroups, sg, lane);                         \
    projection::gemv_paired_runtime<packets::W0, packets::W1, ROWS, 1, LANES>(x, out, gate, up, 1,                     \
        SEISMIC_DIM_F, SEISMIC_DIM_H, tile, shared, simdgroups, sg, lane);

// The compact worklist builder (`routed::build_worklist`): emits unique active
// `(expert, source)` entries and the indirect grid `(CHOICE_GROUPS, count, 1)`.
#define ROUTED_GATE_UP_GATHER_BODY                                                                                     \
    threadgroup int cache[routed::WORKLIST_CACHE_WORDS];                                                               \
    const uint per_group = uint(SEISMIC_RUNTIME_SIMDGROUPS) * ROWS * (32u / LANES);                                    \
    const uint x_groups = (uint(SEISMIC_DIM_F) + per_group - 1u) / per_group;                                          \
    routed::build_worklist(routes, SEISMIC_ROUTES_STRIDE_0, SEISMIC_ROUTES_STRIDE_1, sources, ROUTED_SOURCE_ROWS,      \
        SEISMIC_SOURCES_STRIDE_1, SEISMIC_SOURCES_STRIDE_2, uint(SEISMIC_DIM_M), uint(SEISMIC_DIM_K), x_groups, 0u,    \
        worklist, gather_grid, gather_one_grid, cache, lane,                                                           \
        [&](uint c, slab::Source source) { return ROUTED_GATE_UP_ACTIVE(c, source); });

// Singleton gathered body (`r == 1`): instantiates ONLY `MAXM = 1`, matching
// the 1-row `choice` GEMV register count and occupancy exactly.
#define ROUTED_GATE_UP_GATHERED_ONE_BODY                                                                               \
    typedef routed::Act A;                                                                                             \
    const element::Out<A> product(product_cells);                                                                      \
    const uint tile = group.x;                                                                                         \
    if (ulong(tile) * simdgroups * ROWS * (32u / LANES) >= SEISMIC_DIM_F)                                              \
        return;                                                                                                        \
    device const routed::WorkEntry &entry = worklist->entries[group.y];                                                \
    const uint packed = entry.packed[0];                                                                               \
    const ulong m = packed >> 16u, k = packed & 0xffffu;                                                               \
    const slab::Source source{entry.source};                                                                           \
    const auto product_row = product.offset(                                                                           \
        m * ROUTED_GATE_UP_PRODUCT_STRIDE_0 + k * ROUTED_GATE_UP_PRODUCT_STRIDE_1);                                    \
    if (source.none()) {                                                                                               \
        if (tile == 0)                                                                                                 \
            routed::zero_row(product_row, ROUTED_GATE_UP_PRODUCT_STRIDE_2, uint(SEISMIC_DIM_F),                        \
                sg * 32 + lane, simdgroups * 32);                                                                      \
        return;                                                                                                        \
    }                                                                                                                  \
    const auto in = routed::activation(normalized + m * SEISMIC_NORMALIZED_STRIDE_0 * A::bytes,                        \
        SEISMIC_NORMALIZED_STRIDE_0, SEISMIC_NORMALIZED_STRIDE_1, uint(SEISMIC_DIM_H));                                \
    const auto gate = routed::expert_matrix<packets::W0>(routed::Experts{expert_gate, tier_gate}, source,              \
        KERNEL_W0_LAYOUT(SEISMIC_DIM_H), SEISMIC_DIM_H);                                                               \
    const auto up = routed::expert_matrix<packets::W1>(routed::Experts{expert_up, tier_up}, source,                    \
        KERNEL_W1_LAYOUT(SEISMIC_DIM_H), SEISMIC_DIM_H);                                                               \
    const auto out = projection::scaling<(SEISMIC_DIM_GS + SEISMIC_DIM_US != 0)>::wrap(                                \
        projection::Glu<A>{product_row, 0, ROUTED_GATE_UP_PRODUCT_STRIDE_2, int(SEISMIC_PARAM_ACTIVATION)},            \
        projection::scale_factor(gate_scale, SEISMIC_DIM_GS, SEISMIC_GATE_SCALE_STRIDE_0, ulong(entry.expert)),        \
        projection::scale_factor(up_scale, SEISMIC_DIM_US, SEISMIC_UP_SCALE_STRIDE_0, ulong(entry.expert)));           \
    PROJECTION_OPERANDS_SHARED(globals, NVFP4);                                                                        \
    const auto x = Operands::rows(in, 1u, uint(SEISMIC_DIM_H), globals, simdgroups, sg, lane);                         \
    projection::gemv_paired_runtime<packets::W0, packets::W1, ROWS, 1, LANES>(x, out, gate, up, 1, SEISMIC_DIM_F,      \
        SEISMIC_DIM_H, tile, shared, simdgroups, sg, lane);

// Multi-row gathered body (`r >= 2`): reads `worklist->entries[capacity - 1 - group.y]`
// and projects all `r` rows in chunks of <= 4 with `MAXM in {2, 4}` and 8 KB staging.
#define ROUTED_GATE_UP_GATHERED_BODY                                                                                   \
    typedef routed::Act A;                                                                                             \
    const element::Out<A> product(product_cells);                                                                      \
    const uint tile = group.x;                                                                                         \
    if (ulong(tile) * simdgroups * ROWS * (32u / LANES) >= SEISMIC_DIM_F)                                              \
        return;                                                                                                        \
    device const routed::WorkEntry &entry = worklist->entries[worklist->capacity - 1u - group.y];                      \
    const routed::Gathered chosen =                                                                                    \
        routed::unpack_work_entry(entry, ROUTED_GATE_UP_PRODUCT_STRIDE_0, ROUTED_GATE_UP_PRODUCT_STRIDE_1);            \
    const slab::Source source{entry.source};                                                                           \
    if (source.none()) {                                                                                               \
        if (tile == 0)                                                                                                 \
            for (uint r = 0; r < chosen.count; ++r)                                                                    \
                routed::zero_row(product.offset(chosen.products[r]), ROUTED_GATE_UP_PRODUCT_STRIDE_2,                  \
                    uint(SEISMIC_DIM_F), sg * 32 + lane, simdgroups * 32);                                             \
        return;                                                                                                        \
    }                                                                                                                  \
    const auto gate = routed::expert_matrix<packets::W0>(routed::Experts{expert_gate, tier_gate}, source,              \
        KERNEL_W0_LAYOUT(SEISMIC_DIM_H), SEISMIC_DIM_H);                                                               \
    const auto up = routed::expert_matrix<packets::W1>(routed::Experts{expert_up, tier_up}, source,                    \
        KERNEL_W1_LAYOUT(SEISMIC_DIM_H), SEISMIC_DIM_H);                                                               \
    const float gate_sc = projection::scale_factor(gate_scale, SEISMIC_DIM_GS, SEISMIC_GATE_SCALE_STRIDE_0,            \
        ulong(chosen.expert));                                                                                         \
    const float up_sc = projection::scale_factor(up_scale, SEISMIC_DIM_US, SEISMIC_UP_SCALE_STRIDE_0,                  \
        ulong(chosen.expert));                                                                                         \
    const uint rows = chosen.count;                                                                                    \
    PROJECTION_OPERANDS_SHARED(globals, NVFP4);                                                                        \
    for (uint base = 0u; base < rows; base += 4u) {                                                                    \
        const uint chunk = min(4u, rows - base);                                                                       \
        if (base > 0u)                                                                                                 \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                                           \
        const projection::Plain<A, routed::Listed> in{normalized, SEISMIC_NORMALIZED_STRIDE_0,                         \
            SEISMIC_NORMALIZED_STRIDE_1, uint(SEISMIC_DIM_H), routed::Listed{chosen.rows + base}};                     \
        const auto out = projection::scaling<(SEISMIC_DIM_GS + SEISMIC_DIM_US != 0)>::wrap(                            \
            routed::ChoiceRows<projection::Glu<A>>{                                                                    \
                projection::Glu<A>{product, 1, ROUTED_GATE_UP_PRODUCT_STRIDE_2, int(SEISMIC_PARAM_ACTIVATION)},        \
                chosen.products + base},                                                                               \
            gate_sc, up_sc);                                                                                           \
        const auto x = Operands::rows(in, chunk, uint(SEISMIC_DIM_H), globals, simdgroups, sg, lane);                  \
        PROJECTION_FOR_ROWS_24(chunk,                                                                                  \
            (projection::gemv_paired_runtime<packets::W0, packets::W1, ROWS, MAXM, LANES, 128u>(x, out, gate, up,      \
                chunk, SEISMIC_DIM_F, SEISMIC_DIM_H, tile, shared, simdgroups, sg, lane)));                            \
    }
