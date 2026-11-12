// Decode expansion of up-only experts (M <= DECODE_ROWS): threadgroup row y = choice
// (m, k), a K1 GEMV over the feature tile `x` of expert routes[m, k]'s up rows
// (its record, `routed::decode_source`; a resident layer's, its expert's
// record) with the activated epilogue
// A(act(A(up))). Every row of a choice with no record is zero.

#ifdef SEISMIC_FORMING_ROUTED_UP_RESIDENT
// A resident layer's: nothing lands in its records while it runs, so plain
// loads read them.
#define PACKETS_PLAIN_RECORDS
#endif
#define KERNEL_W0 SEISMIC_EXPERT_UP
#include "lib/routed/routed.h"
#include "lib/projection/gemv.h"

// The up weight's operand row (the normalized row, or NVFP4 operands),
// formed while the GEMV stages it.
constant constexpr bool NVFP4 = projection::OperandRecipe<packets::W0_OPERANDS_NVFP4>::NVFP4;
typedef projection::operands<NVFP4> Operands;

// The ports, the expert stack's: its records' address table.
#define ROUTED_UP_PORTS                                               \
    device const uchar *normalized [[buffer(SEISMIC_BUFFER_NORMALIZED)]],    \
    device const int *routes [[buffer(SEISMIC_BUFFER_ROUTES)]],              \
    device const int *sources [[buffer(SEISMIC_BUFFER_SOURCES)]],            \
    device const ulong *expert_up [[buffer(SEISMIC_BUFFER_EXPERT_UP)]],      \
    device const ulong *tier_up [[buffer(SEISMIC_BUFFER_TIER_UP)]],          \
    device const float *up_scale [[buffer(SEISMIC_BUFFER_UP_SCALE)]],        \
    device uchar *product [[buffer(SEISMIC_RESULT_0_BUFFER)]],               \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],          \
    threadgroup uchar *shared [[threadgroup(0)]],                            \
    uint2 group [[threadgroup_position_in_grid]],                            \
    uint simdgroups [[simdgroups_per_threadgroup]],                          \
    uint sg [[simdgroup_index_in_threadgroup]],                              \
    uint lane [[thread_index_in_simdgroup]]

// The body (`routed::choice_record`, `routed::choice_weights`).
#define ROUTED_UP_BODY                                                                                                 \
    typedef routed::Act A;                                                                                             \
    const uint tile = group.x;                                                                                         \
    if (ulong(tile) * simdgroups * ROWS * (32u / LANES) >= SEISMIC_DIM_F)                                              \
        return;                                                                                                        \
    const ulong m = group.y / SEISMIC_DIM_K, k = group.y % SEISMIC_DIM_K;                                              \
    const int expert = routes[m * SEISMIC_ROUTES_STRIDE_0 + k * SEISMIC_ROUTES_STRIDE_1];                              \
    const int record = routed::choice_record(expert_up, sources, SEISMIC_DIM_XS, SEISMIC_SOURCES_STRIDE_1,             \
        SEISMIC_SOURCES_STRIDE_2, m, k, expert);                                                                       \
    device uchar *choice = product + (m * SEISMIC_RESULT_0_STRIDE_0 + k * SEISMIC_RESULT_0_STRIDE_1) * A::bytes;       \
    /* A choice reading no record (the CPU's, or a failed one): its first                                              \
       threadgroup zeroes its row. */                                                                                  \
    if (record == -1) {                                                                                                \
        if (tile == 0)                                                                                                 \
            routed::zero_row(choice, SEISMIC_RESULT_0_STRIDE_2, uint(SEISMIC_DIM_F), sg * 32 + lane, simdgroups * 32); \
        return;                                                                                                        \
    }                                                                                                                  \
    const auto in = routed::activation(normalized + m * SEISMIC_NORMALIZED_STRIDE_0 * A::bytes,                        \
        SEISMIC_NORMALIZED_STRIDE_0, SEISMIC_NORMALIZED_STRIDE_1, uint(SEISMIC_DIM_H));                                \
    const auto up = routed::choice_weights<packets::W0>(expert_up, tier_up, record,                                    \
        KERNEL_W0_LAYOUT(SEISMIC_DIM_H), SEISMIC_DIM_F, SEISMIC_DIM_H);                                                \
    const auto out = projection::scaling<(SEISMIC_DIM_US != 0)>::wrap(                                                 \
        projection::Activated<A>{choice, 0, SEISMIC_RESULT_0_STRIDE_2, int(SEISMIC_PARAM_ACTIVATION)},                 \
        projection::scale_factor(up_scale, SEISMIC_DIM_US, SEISMIC_UP_SCALE_STRIDE_0, ulong(expert)), 1.0f);           \
    PROJECTION_OPERANDS_SHARED(globals, NVFP4);                                                                        \
    const auto x = Operands::rows(in, 1u, uint(SEISMIC_DIM_H), globals, simdgroups, sg, lane);                         \
    projection::gemv_runtime<packets::W0, ROWS, 1, LANES>(x, out, up, 1, SEISMIC_DIM_F, SEISMIC_DIM_H, tile,           \
        shared, simdgroups, sg, lane);

#ifdef SEISMIC_FORMING_ROUTED_UP
template <uint ROWS, uint LANES>
kernel void routed_up(ROUTED_UP_PORTS) {
    ROUTED_UP_BODY
}
#endif
#ifdef SEISMIC_FORMING_ROUTED_UP_RESIDENT
// A resident layer's (XS 0), its records read with plain loads.
template <uint ROWS, uint LANES>
kernel void routed_up_resident(ROUTED_UP_PORTS) {
    ROUTED_UP_BODY
}
#endif
