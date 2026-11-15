// Decode expansion of gated experts (M <= VERIFY_ROWS): threadgroup row y = choice
// (m, k), a K1 paired GEMV over the feature tile `x` of expert routes[m, k]'s
// gate and up rows (its record, `routed::decode_source`: a bank record, or a
// host tier record read in place; a resident layer's, its experts' records)
// with the activation-generic GLU epilogue, the accumulators scaled by the expert's
// `gate_scale`/`up_scale` where those ports are present (static GS, US).
// Every row of a choice with no record is zero.

#if defined(SEISMIC_FORMING_ROUTED_GATE_UP_RESIDENT) || defined(SEISMIC_FORMING_ROUTED_GATE_UP_GATHERED_RESIDENT)
// A resident layer's: nothing lands in its records while it runs, so plain
// loads read them.
#define PACKETS_PLAIN_RECORDS
#endif
#define KERNEL_W0 SEISMIC_EXPERT_GATE
#define KERNEL_W1 SEISMIC_EXPERT_UP
// The expansion's limit as its F32 bits LB (0: none), which the Glu epilogue
// clamps at.
#if SEISMIC_STATIC_LB != 0
#define PROJECTION_GLU_LIMIT SEISMIC_STATIC_LB
#endif
#include "lib/routed/routed.h"
#include "lib/projection/gemv.h"

// The row both weights take (the normalized row, or NVFP4 operands), formed
// while the GEMV stages it.
constant constexpr bool NVFP4 =
    projection::OperandRecipe<packets::W0_OPERANDS_NVFP4, packets::W1_OPERANDS_NVFP4>::NVFP4;
typedef projection::operands<NVFP4> Operands;

// The ports, the expert stacks': their records' address tables.
#define ROUTED_GATE_UP_PORTS                                         \
    device const uchar *normalized [[buffer(SEISMIC_BUFFER_NORMALIZED)]],   \
    device const int *routes [[buffer(SEISMIC_BUFFER_ROUTES)]],             \
    device const int *sources [[buffer(SEISMIC_BUFFER_SOURCES)]],           \
    device const ulong *expert_gate [[buffer(SEISMIC_BUFFER_EXPERT_GATE)]], \
    device const ulong *expert_up [[buffer(SEISMIC_BUFFER_EXPERT_UP)]],     \
    device const ulong *tier_gate [[buffer(SEISMIC_BUFFER_TIER_GATE)]],     \
    device const ulong *tier_up [[buffer(SEISMIC_BUFFER_TIER_UP)]],         \
    device const float *gate_scale [[buffer(SEISMIC_BUFFER_GATE_SCALE)]],   \
    device const float *up_scale [[buffer(SEISMIC_BUFFER_UP_SCALE)]],       \
    SEISMIC_RESULT_0_PARAM(product_cells),                                  \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],         \
    threadgroup uchar *shared [[threadgroup(0)]],                           \
    uint2 group [[threadgroup_position_in_grid]],                           \
    uint simdgroups [[simdgroups_per_threadgroup]],                         \
    uint sg [[simdgroup_index_in_threadgroup]],                             \
    uint lane [[thread_index_in_simdgroup]]

// The body (`routed::decode_source`, `routed::expert_matrix`).
#define ROUTED_GATE_UP_BODY                                                                                            \
    typedef routed::Act A;                                                                                             \
    const uint tile = group.x;                                                                                         \
    if (ulong(tile) * simdgroups * ROWS * (32u / LANES) >= SEISMIC_DIM_F)                                              \
        return;                                                                                                        \
    const ulong m = group.y / SEISMIC_DIM_K, k = group.y % SEISMIC_DIM_K;                                              \
    const int expert = routes[m * SEISMIC_ROUTES_STRIDE_0 + k * SEISMIC_ROUTES_STRIDE_1];                              \
    const slab::Source source = routed::decode_source(sources, SEISMIC_DIM_XS, SEISMIC_SOURCES_STRIDE_1,               \
        SEISMIC_SOURCES_STRIDE_2, m, k, expert);                                                                       \
    const element::Out<ELEMENT_OF(SEISMIC_RESULT_0)> choice =                                                          \
        element::Out<ELEMENT_OF(SEISMIC_RESULT_0)>(product_cells).offset(                                              \
            m * SEISMIC_RESULT_0_STRIDE_0 + k * SEISMIC_RESULT_0_STRIDE_1);                                            \
    /* A choice reading no record (the CPU's, or a failed one): its first                                              \
       threadgroup zeroes its row. */                                                                                  \
    if (source.none()) {                                                                                               \
        if (tile == 0)                                                                                                 \
            routed::zero_row(choice, SEISMIC_RESULT_0_STRIDE_2, uint(SEISMIC_DIM_F), sg * 32 + lane, simdgroups * 32); \
        return;                                                                                                        \
    }                                                                                                                  \
    const auto in = routed::activation(normalized + m * SEISMIC_NORMALIZED_STRIDE_0 * A::bytes,                        \
        SEISMIC_NORMALIZED_STRIDE_0, SEISMIC_NORMALIZED_STRIDE_1, uint(SEISMIC_DIM_H));                                \
    const auto gate = routed::expert_matrix<packets::W0>(routed::Experts{expert_gate, tier_gate}, source,              \
        KERNEL_W0_LAYOUT(SEISMIC_DIM_H), SEISMIC_DIM_H);                                                               \
    const auto up = routed::expert_matrix<packets::W1>(routed::Experts{expert_up, tier_up}, source,                    \
        KERNEL_W1_LAYOUT(SEISMIC_DIM_H), SEISMIC_DIM_H);                                                               \
    const auto out = projection::scaling<(SEISMIC_DIM_GS + SEISMIC_DIM_US != 0)>::wrap(                                \
        projection::Glu<A>{choice, 0, SEISMIC_RESULT_0_STRIDE_2, int(SEISMIC_PARAM_ACTIVATION)},                       \
        projection::scale_factor(gate_scale, SEISMIC_DIM_GS, SEISMIC_GATE_SCALE_STRIDE_0, ulong(expert)),              \
        projection::scale_factor(up_scale, SEISMIC_DIM_US, SEISMIC_UP_SCALE_STRIDE_0, ulong(expert)));                 \
    PROJECTION_OPERANDS_SHARED(globals, NVFP4);                                                                        \
    const auto x = Operands::rows(in, 1u, uint(SEISMIC_DIM_H), globals, simdgroups, sg, lane);                         \
    projection::gemv_paired_runtime<packets::W0, packets::W1, ROWS, 1, LANES>(x, out, gate, up, 1,                     \
        SEISMIC_DIM_F, SEISMIC_DIM_H, tile, shared, simdgroups, sg, lane);

// The gathered body (`routed::gather`): choice y's threadgroups run where it
// is its expert's first choice, over the rows of every choice of the expert.
#define ROUTED_GATE_UP_GATHERED_BODY                                                                                   \
    typedef routed::Act A;                                                                                             \
    const element::Out<ELEMENT_OF(SEISMIC_RESULT_0)> product(product_cells);                                           \
    const uint tile = group.x;                                                                                         \
    if (ulong(tile) * simdgroups * ROWS * (32u / LANES) >= SEISMIC_DIM_F)                                              \
        return;                                                                                                        \
    const routed::Gathered chosen = routed::gather(routes, SEISMIC_ROUTES_STRIDE_0, SEISMIC_ROUTES_STRIDE_1,           \
        uint(SEISMIC_DIM_M), uint(SEISMIC_DIM_K), group.y, SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1);       \
    if (chosen.count == 0)                                                                                             \
        return;                                                                                                        \
    /* Every choice of an expert has its one source. */                                                                \
    const slab::Source source = routed::decode_source(sources, SEISMIC_DIM_XS, SEISMIC_SOURCES_STRIDE_1,               \
        SEISMIC_SOURCES_STRIDE_2, group.y / SEISMIC_DIM_K, group.y % SEISMIC_DIM_K, chosen.expert);                    \
    if (source.none()) {                                                                                               \
        if (tile == 0)                                                                                                 \
            for (uint r = 0; r < chosen.count; ++r)                                                                    \
                routed::zero_row(product.offset(chosen.products[r]), SEISMIC_RESULT_0_STRIDE_2,            \
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
            projection::Glu<A>{product, 1, SEISMIC_RESULT_0_STRIDE_2, int(SEISMIC_PARAM_ACTIVATION)}, chosen.products},\
        projection::scale_factor(gate_scale, SEISMIC_DIM_GS, SEISMIC_GATE_SCALE_STRIDE_0, ulong(chosen.expert)),       \
        projection::scale_factor(up_scale, SEISMIC_DIM_US, SEISMIC_UP_SCALE_STRIDE_0, ulong(chosen.expert)));          \
    const uint rows = chosen.count;                                                                                    \
    PROJECTION_OPERANDS_SHARED(globals, NVFP4);                                                                        \
    const auto x = Operands::rows(in, rows, uint(SEISMIC_DIM_H), globals, simdgroups, sg, lane);                       \
    PROJECTION_FOR_ROWS(rows,                                                                                          \
        (projection::gemv_paired_runtime<packets::W0, packets::W1, ROWS, MAXM, LANES>(x, out, gate, up, rows,          \
            SEISMIC_DIM_F, SEISMIC_DIM_H, tile, shared, simdgroups, sg, lane)));

#ifdef SEISMIC_FORMING_ROUTED_GATE_UP_CHOICE
template <uint ROWS, uint LANES>
kernel void routed_gate_up_choice(ROUTED_GATE_UP_PORTS) {
    ROUTED_GATE_UP_BODY
}
#endif
#ifdef SEISMIC_FORMING_ROUTED_GATE_UP_RESIDENT
// A resident layer's (XS 0), its records read with plain loads.
template <uint ROWS, uint LANES>
kernel void routed_gate_up_resident(ROUTED_GATE_UP_PORTS) {
    ROUTED_GATE_UP_BODY
}
#endif
#ifdef SEISMIC_FORMING_ROUTED_GATE_UP_GATHERED
template <uint ROWS, uint LANES>
kernel void routed_gate_up_gathered(ROUTED_GATE_UP_PORTS) {
    ROUTED_GATE_UP_GATHERED_BODY
}
#endif
#ifdef SEISMIC_FORMING_ROUTED_GATE_UP_GATHERED_RESIDENT
// A resident layer's (XS 0), its records read with plain loads.
template <uint ROWS, uint LANES>
kernel void routed_gate_up_gathered_resident(ROUTED_GATE_UP_PORTS) {
    ROUTED_GATE_UP_GATHERED_BODY
}
#endif
