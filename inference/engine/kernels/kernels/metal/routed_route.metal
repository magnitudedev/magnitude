// Routing (`routed_route`) on the `dense` projection operator, segmented over
// `(router, shared_router)` in the CPU 8-lane reference order.
#define KERNEL_A SEISMIC_ELEMENT_A
#define KERNEL_W0 SEISMIC_ROUTER
#define KERNEL_W1 SEISMIC_SHARED_ROUTER
#define PROJECTION_N0 SEISMIC_DIM_E
#define PROJECTION_N1 1
#include "lib/projection/projection.h"
#include "lib/routed/route.h"

typedef activation::Act Act;
typedef ELEMENT_OF(SEISMIC_NORM) Norm;

#define PROJECTION_ARGUMENTS                                                                \
    device const float *residual [[buffer(SEISMIC_BUFFER_RESIDUAL)]],                       \
    device const uchar *norm [[buffer(SEISMIC_BUFFER_NORM)]],                               \
    device const uchar *router [[buffer(SEISMIC_BUFFER_ROUTER)]],                           \
    device const uchar *shared_router [[buffer(SEISMIC_BUFFER_SHARED_ROUTER)]],             \
    device int *routes [[buffer(SEISMIC_BUFFER_ROUTES)]],                                   \
    SEISMIC_OUT_SCORES_PARAM(scores_cells),                                                 \
    device const float *expert_scale [[buffer(SEISMIC_BUFFER_EXPERT_SCALE)]],               \
    SEISMIC_RESULT_0_PARAM(normalized_cells),                                               \
    SEISMIC_RESULT_1_PARAM(coefficient_cells),                                              \
    arrive::coherent_device<float> logits [[buffer(SEISMIC_BUFFER_SCRATCH_LOGITS)]],        \
    device atomic_uint *arrivals [[buffer(SEISMIC_BUFFER_SCRATCH_ARRIVALS)]],               \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

#define PROJECTION_STAGE_K SEISMIC_DIM_H
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
    const projection::Plain<Act, projection::AllRows> in{                                   \
        element::Out<Act>(normalized_cells).stored(), \
        SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1, k, {}};                       \
    const projection::Weights<packets::W0> w0{{router, KERNEL_W0_LAYOUT(k), k}};            \
    const projection::Weights<packets::W1> w1{{shared_router, KERNEL_W1_LAYOUT(k), k}};     \
    const projection::CoherentStore out0{logits, SEISMIC_DIM_E + 1ul, 1ul, 0ul}; \
    const projection::CoherentStore out1{logits, SEISMIC_DIM_E + 1ul, 1ul, SEISMIC_DIM_E}

#define PROJECTION_GEMV_INPUT(x) const auto x = raw

#define PROJECTION_ORDERED_EPILOGUE(ordered_tiles)                                          \
    do {                                                                                    \
        if (arrive::last(arrivals, ordered_tiles, reinterpret_cast<threadgroup uint *>(shared), sg * 32u + lane)) {           \
            for (uint r = sg; r < m; r += simdgroups) {                                     \
                select_row(r, [&](uint e) { return logits[logit_index(r, e)]; },            \
                    expert_scale, routes, element::Out<element::F32>(scores_cells),         \
                    element::Out<element::F32>(coefficient_cells), seismic_words, lane);    \
            }                                                                               \
        }                                                                                   \
    } while (0)

#include "lib/projection/operator.h"

kernel void routed_route_ordered_stage(PROJECTION_ARGUMENTS, PROJECTION_ORDERED_STAGE_ARGS) {
    PROJECTION_OPERANDS;
    PROJECTION_ORDERED_STAGE_BODY(raw, element::Out<Act>(normalized_cells),
        SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1);
}

kernel void routed_route_ordered_gemv(PROJECTION_ARGUMENTS, PROJECTION_ORDERED_GEMV_ARGS) {
    constexpr uint simdgroups = SEISMIC_TUNE_SIMDGROUPS;
    PROJECTION_SEGMENTED_ORDERED_GEMV_ALL;
}

kernel void routed_route_ordered_gemm(PROJECTION_ARGUMENTS, PROJECTION_ORDERED_GEMM_ARGS) {
    PROJECTION_SEGMENTED_ORDERED_GEMM;
}

kernel void routed_route_select(PROJECTION_ARGUMENTS,
    uint group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const ulong row = ulong(group) * 8ul + sg;
    if (row < SEISMIC_DIM_M) {
        select_row(row, [&](uint e) { return logits[logit_index(row, e)]; },
            expert_scale, routes, element::Out<element::F32>(scores_cells),
            element::Out<element::F32>(coefficient_cells), seismic_words, lane);
    }
}
