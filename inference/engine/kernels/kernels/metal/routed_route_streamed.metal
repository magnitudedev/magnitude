// Expert streaming's decode routing (`routed_route_streamed`) on the `dense`
// projection operator, segmented over `(router, shared_router, (shared_gate, shared_up))`,
// followed by resolve, assignment and front-ring publication in the selecting threadgroup.
#define KERNEL_A SEISMIC_ELEMENT_A
#define KERNEL_W0 SEISMIC_ROUTER
#define KERNEL_W1 SEISMIC_SHARED_ROUTER
#define KERNEL_W2 SEISMIC_SHARED_GATE
#define KERNEL_W3 SEISMIC_SHARED_UP
#define PROJECTION_N0 SEISMIC_DIM_E
#define PROJECTION_N1 1
#define PROJECTION_N2 SEISMIC_DIM_S
#define PROJECTION_PARTNER_2 w_up
#define PROJECTION_PARTNER_PACKET_2 packets::W3
#include "lib/projection/projection.h"
#include "lib/projection/gemv.h"
#include "lib/routed/route.h"
#include "lib/streaming/control.h"
#include "../streaming/hint.hpp"
#include "../streaming/resolve.hpp"
#include "lib/routed/keys.h"

typedef activation::Act Act;
typedef ELEMENT_OF(SEISMIC_NORM) Norm;
constant constexpr bool NVFP4_SHARED = projection::OperandRecipe<
    packets::W2_OPERANDS_NVFP4, packets::W3_OPERANDS_NVFP4>::NVFP4;

#define PROJECTION_ARGUMENTS                                                                \
    device const float *residual [[buffer(SEISMIC_BUFFER_RESIDUAL)]],                       \
    device const uchar *norm [[buffer(SEISMIC_BUFFER_NORM)]],                               \
    device const uchar *router [[buffer(SEISMIC_BUFFER_ROUTER)]],                           \
    device const uchar *shared_router [[buffer(SEISMIC_BUFFER_SHARED_ROUTER)]],             \
    device const uchar *shared_gate [[buffer(SEISMIC_BUFFER_SHARED_GATE)]],                 \
    device const uchar *shared_up [[buffer(SEISMIC_BUFFER_SHARED_UP)]],                     \
    device int *routes [[buffer(SEISMIC_BUFFER_ROUTES)]],                                   \
    SEISMIC_OUT_SCORES_PARAM(scores_cells),                                                 \
    device const float *expert_scale [[buffer(SEISMIC_BUFFER_EXPERT_SCALE)]],               \
    SYSTEM const int *map [[buffer(SEISMIC_BUFFER_MAP)]],                                   \
    SYSTEM const int *tier [[buffer(SEISMIC_BUFFER_TIER)]],                                 \
    device const int *step [[buffer(SEISMIC_BUFFER_STEP)]],                                 \
    device int *sources [[buffer(SEISMIC_BUFFER_SOURCES)]],                                 \
    SYSTEM int *routing [[buffer(SEISMIC_BUFFER_ROUTING)]],                                 \
    SYSTEM int *keys [[buffer(SEISMIC_BUFFER_KEYS)]],                                       \
    SYSTEM int *file [[buffer(SEISMIC_BUFFER_FILE)]],                                       \
    SYSTEM const int *policy [[buffer(SEISMIC_BUFFER_POLICY)]],                             \
    SYSTEM int *front [[buffer(SEISMIC_BUFFER_FRONT)]],                                     \
    device int *land_grid [[buffer(SEISMIC_BUFFER_LAND_GRID)]],                             \
    SEISMIC_RESULT_0_PARAM(normalized_cells),                                               \
    SEISMIC_RESULT_1_PARAM(coefficient_cells),                                              \
    SEISMIC_RESULT_2_PARAM(shared_product_cells),                                           \
    arrive::coherent_device<float> logits [[buffer(SEISMIC_BUFFER_SCRATCH_LOGITS)]],        \
    device atomic_uint *arrivals [[buffer(SEISMIC_BUFFER_SCRATCH_ARRIVALS)]],               \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

#define PROJECTION_ORDERED_PUBLISH 1
#define PROJECTION_ORDERED_PUB_OUT element::Out<Act>(normalized_cells)
#define PROJECTION_ORDERED_PUB_S0 SEISMIC_RESULT_0_STRIDE_0
#define PROJECTION_ORDERED_PUB_S1 SEISMIC_RESULT_0_STRIDE_1

#define PROJECTION_OPERANDS                                                                 \
    const uint m = uint(SEISMIC_DIM_M), k = uint(SEISMIC_DIM_H);                            \
    const typename projection::OrderedInputOf<Act, Norm, projection::AllRows,               \
        uint(SEISMIC_DIM_NM), uint(SEISMIC_DIM_PT)>::type raw{                              \
        residual, SEISMIC_RESIDUAL_STRIDE_0, SEISMIC_RESIDUAL_STRIDE_1, norm,               \
        SEISMIC_NORM_STRIDE_0, as_type<float>(uint(SEISMIC_PARAM_EPS)), k, {}};             \
    const projection::Weights<packets::W0> w0{{router, KERNEL_W0_LAYOUT(k), k}};            \
    const projection::Weights<packets::W1> w1{{shared_router, KERNEL_W1_LAYOUT(k), k}};     \
    const projection::Weights<packets::W2> w2{{shared_gate, KERNEL_W2_LAYOUT(k), k}};       \
    const projection::Weights<packets::W3> w_up{{shared_up, KERNEL_W3_LAYOUT(k), k}};       \
    const projection::CoherentStore out0{logits, SEISMIC_DIM_E + 1ul, 1ul, 0ul}; \
    const projection::CoherentStore out1{logits, SEISMIC_DIM_E + 1ul, 1ul, SEISMIC_DIM_E}; \
    const projection::SiluMul<Act> out2{element::Out<Act>(shared_product_cells),            \
        SEISMIC_RESULT_2_STRIDE_0, SEISMIC_RESULT_2_STRIDE_1}

#define PROJECTION_SEGMENT_NVFP4(s) ((s) == 2 ? NVFP4_SHARED : false)
#define PROJECTION_SEGMENT_INPUT(s, name) const auto name = raw

#define PROJECTION_ORDERED_EPILOGUE(ordered_tiles)                                          \
    do {                                                                                    \
        const uint tid = sg * 32u + lane;                                                   \
        if (arrive::last(arrivals, ordered_tiles, reinterpret_cast<threadgroup uint *>(shared), tid)) { \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                \
            const element::Out<element::F32> scores_out(scores_cells);                      \
            ROUTED_KEYS_RECORD;                                                             \
            for (uint r = sg; r < m; r += simdgroups) {                                     \
                select_row(r, [&](uint e) { return logits[logit_index(r, e)]; },            \
                    expert_scale, routes, scores_out,                                       \
                    element::Out<element::F32>(coefficient_cells), seismic_words, lane,     \
                    ROUTED_ROW_KEYS(r));                                                    \
            }                                                                               \
            threadgroup int *words = reinterpret_cast<threadgroup int *>(shared);           \
            const bool whole = streaming_load(STREAMING_AT(policy,                          \
                SEISMIC_CONST_POLICY_WHOLE * SEISMIC_POLICY_STRIDE_0)) != 0;                \
            streaming_resolve(routes, SEISMIC_ROUTES_STRIDE_0, SEISMIC_ROUTES_STRIDE_1,     \
                m * uint(SEISMIC_DIM_K),                                                    \
                reinterpret_cast<device const int *>(scores_out.stored()),                  \
                SEISMIC_SCORES_STRIDE_0, SEISMIC_SCORES_STRIDE_1,                           \
                step, SEISMIC_STEP_STRIDE_0, whole,                                         \
                map, SEISMIC_MAP_STRIDE_0, SEISMIC_MAP_STRIDE_1,                            \
                tier, SEISMIC_TIER_STRIDE_0, uint(SEISMIC_DIM_TM), uint(SEISMIC_PARAM_SITED), \
                sources, SEISMIC_SOURCES_STRIDE_0, SEISMIC_SOURCES_STRIDE_1, SEISMIC_SOURCES_STRIDE_2, \
                routing, SEISMIC_ROUTING_STRIDE_0, SEISMIC_ROUTING_STRIDE_1,                \
                uint(SEISMIC_DIM_FLIGHT), uint(SEISMIC_DIM_P),                              \
                file, SEISMIC_FILE_STRIDE_0, SEISMIC_FILE_STRIDE_1,                         \
                policy, front, SEISMIC_FRONT_STRIDE_0,                                      \
                land_grid, SEISMIC_LAND_GRID_STRIDE_0, int(SEISMIC_PARAM_LAND_GROUPS), words, tid, simdgroups * 32u);       \
        }                                                                                   \
    } while (0)

#include "lib/projection/operator.h"

kernel void routed_route_streamed_ordered_gemv1(PROJECTION_ARGUMENTS, PROJECTION_ORDERED_GEMV_ARGS) {
    constexpr uint simdgroups = SEISMIC_TUNE_SIMDGROUPS, ROWS = SEISMIC_TUNE_ROWS, LANES = SEISMIC_TUNE_LANES;
    PROJECTION_SEGMENTED_ORDERED_GEMV_MIXED(1u);
}

kernel void routed_route_streamed_ordered_gemv2(PROJECTION_ARGUMENTS, PROJECTION_ORDERED_GEMV_ARGS) {
    constexpr uint simdgroups = SEISMIC_TUNE_SIMDGROUPS, ROWS = SEISMIC_TUNE_ROWS, LANES = SEISMIC_TUNE_LANES;
    PROJECTION_SEGMENTED_ORDERED_GEMV_MIXED(2u);
}

kernel void routed_route_streamed_ordered_gemv4(PROJECTION_ARGUMENTS, PROJECTION_ORDERED_GEMV_ARGS) {
    constexpr uint simdgroups = SEISMIC_TUNE_SIMDGROUPS, ROWS = SEISMIC_TUNE_ROWS, LANES = SEISMIC_TUNE_LANES;
    PROJECTION_SEGMENTED_ORDERED_GEMV_MIXED(4u);
}

kernel void routed_route_streamed_ordered_gemv8(PROJECTION_ARGUMENTS, PROJECTION_ORDERED_GEMV_ARGS) {
    constexpr uint simdgroups = SEISMIC_TUNE_SIMDGROUPS, ROWS = SEISMIC_TUNE_ROWS, LANES = SEISMIC_TUNE_LANES;
    PROJECTION_SEGMENTED_ORDERED_GEMV_MIXED(8u);
}
