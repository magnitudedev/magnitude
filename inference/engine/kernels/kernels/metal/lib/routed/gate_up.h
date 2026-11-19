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

// The gathered body (`routed::gather`): choice y's threadgroups run where it
// is the first active choice of its expert and source, over the rows of every
// active choice sharing them.
#define ROUTED_GATE_UP_GATHERED_BODY                                                                                   \
    typedef routed::Act A;                                                                                             \
    const element::Out<A> product(product_cells);                                                                      \
    const uint tile = group.x;                                                                                         \
    if (ulong(tile) * simdgroups * ROWS * (32u / LANES) >= SEISMIC_DIM_F)                                              \
        return;                                                                                                        \
    const routed::Gathered chosen = routed::gather(routes, SEISMIC_ROUTES_STRIDE_0, SEISMIC_ROUTES_STRIDE_1,           \
        sources, ROUTED_SOURCE_ROWS, SEISMIC_SOURCES_STRIDE_1, SEISMIC_SOURCES_STRIDE_2, uint(SEISMIC_DIM_M),          \
        uint(SEISMIC_DIM_K), group.y, ROUTED_GATE_UP_PRODUCT_STRIDE_0, ROUTED_GATE_UP_PRODUCT_STRIDE_1,                \
        [&](uint c, slab::Source source) { return ROUTED_GATE_UP_ACTIVE(c, source); });                                \
    if (chosen.count == 0)                                                                                             \
        return;                                                                                                        \
    const slab::Source source = routed::decode_source(sources, ROUTED_SOURCE_ROWS, SEISMIC_SOURCES_STRIDE_1,           \
        SEISMIC_SOURCES_STRIDE_2, group.y / SEISMIC_DIM_K, group.y % SEISMIC_DIM_K, chosen.expert);                    \
    if (source.none()) {                                                                                               \
        if (tile == 0)                                                                                                 \
            for (uint r = 0; r < chosen.count; ++r)                                                                    \
                routed::zero_row(product.offset(chosen.products[r]), ROUTED_GATE_UP_PRODUCT_STRIDE_2,                  \
                    uint(SEISMIC_DIM_F), sg * 32 + lane, simdgroups * 32);                                             \
        return;                                                                                                        \
    }                                                                                                                  \
    const projection::Plain<A, routed::Listed> in{normalized, SEISMIC_NORMALIZED_STRIDE_0,                             \
        SEISMIC_NORMALIZED_STRIDE_1, uint(SEISMIC_DIM_H), routed::Listed{chosen.rows}};                                \
    const auto gate = routed::expert_matrix<packets::W0>(routed::Experts{expert_gate, tier_gate}, source,              \
        KERNEL_W0_LAYOUT(SEISMIC_DIM_H), SEISMIC_DIM_H);                                                               \
    const auto up = routed::expert_matrix<packets::W1>(routed::Experts{expert_up, tier_up}, source,                    \
        KERNEL_W1_LAYOUT(SEISMIC_DIM_H), SEISMIC_DIM_H);                                                               \
    const auto out = projection::scaling<(SEISMIC_DIM_GS + SEISMIC_DIM_US != 0)>::wrap(                                \
        routed::ChoiceRows<projection::Glu<A>>{                                                                        \
            projection::Glu<A>{product, 1, ROUTED_GATE_UP_PRODUCT_STRIDE_2, int(SEISMIC_PARAM_ACTIVATION)},            \
            chosen.products},                                                                                          \
        projection::scale_factor(gate_scale, SEISMIC_DIM_GS, SEISMIC_GATE_SCALE_STRIDE_0, ulong(chosen.expert)),       \
        projection::scale_factor(up_scale, SEISMIC_DIM_US, SEISMIC_UP_SCALE_STRIDE_0, ulong(chosen.expert)));          \
    const uint rows = chosen.count;                                                                                    \
    PROJECTION_OPERANDS_SHARED(globals, NVFP4);                                                                        \
    const auto x = Operands::rows(in, rows, uint(SEISMIC_DIM_H), globals, simdgroups, sg, lane);                       \
    PROJECTION_FOR_ROWS(rows,                                                                                          \
        (projection::gemv_paired_runtime<packets::W0, packets::W1, ROWS, MAXM, LANES>(x, out, gate, up, rows,          \
            SEISMIC_DIM_F, SEISMIC_DIM_H, tile, shared, simdgroups, sg, lane)));
