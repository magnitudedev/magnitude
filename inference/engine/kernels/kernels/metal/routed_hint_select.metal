// Ahead-of-step general-selection hints (`routed_hint_select`) on the `dense`
// projection operator, segmented over `(next_router, after_router)` in the CPU
// 8-lane reference order; segment 1 is zeroed when `AHEAD < 2`.
#define KERNEL_A SEISMIC_ELEMENT_A
#define KERNEL_W0 SEISMIC_NEXT_ROUTER
#define KERNEL_W1 SEISMIC_AFTER_ROUTER
#define PROJECTION_N0 SEISMIC_DIM_EN
#define PROJECTION_N1 SEISMIC_DIM_EN
#define PROJECTION_ZEROED(s) ((s) == 1 && SEISMIC_DIM_AHEAD < 2)
#include "lib/projection/projection.h"
#include "lib/core/arrive.h"
#include "lib/routed/select.h"
#include "lib/streaming/control.h"
#include "../streaming/hint.hpp"

typedef activation::Act Act;
typedef ELEMENT_OF(SEISMIC_NEXT_ROUTER_NORM) RouterNorm;

#define PROJECTION_ARGUMENTS                                                                \
    device const float *residual [[buffer(SEISMIC_BUFFER_RESIDUAL)]],                       \
    device const int *step [[buffer(SEISMIC_BUFFER_STEP)]],                                 \
    device const uchar *next_router_norm [[buffer(SEISMIC_BUFFER_NEXT_ROUTER_NORM)]],       \
    device const uchar *next_router [[buffer(SEISMIC_BUFFER_NEXT_ROUTER)]],                 \
    device const float *next_bias [[buffer(SEISMIC_BUFFER_NEXT_BIAS)]],                     \
    device const uchar *after_router_norm [[buffer(SEISMIC_BUFFER_AFTER_ROUTER_NORM)]],     \
    device const uchar *after_router [[buffer(SEISMIC_BUFFER_AFTER_ROUTER)]],               \
    device const float *after_bias [[buffer(SEISMIC_BUFFER_AFTER_BIAS)]],                   \
    SYSTEM int *rings [[buffer(SEISMIC_BUFFER_RINGS)]],                                     \
    arrive::coherent_device<float> logits [[buffer(SEISMIC_BUFFER_SCRATCH_LOGITS)]],        \
    device atomic_uint *arrivals [[buffer(SEISMIC_BUFFER_SCRATCH_ARRIVALS)]],               \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

#define PROJECTION_OPERANDS                                                                 \
    const uint m = uint(SEISMIC_DIM_M), k = uint(SEISMIC_DIM_H);                            \
    const float eps = as_type<float>(uint(SEISMIC_PARAM_EPSILON));                          \
    const projection::OrderedRms<Act, RouterNorm, projection::AllRows> raw0{                \
        residual, SEISMIC_RESIDUAL_STRIDE_0, SEISMIC_RESIDUAL_STRIDE_1, next_router_norm,   \
        SEISMIC_NEXT_ROUTER_NORM_STRIDE_0, eps, k, {}};                                     \
    const projection::OrderedRms<Act, RouterNorm, projection::AllRows> raw1{                \
        residual, SEISMIC_RESIDUAL_STRIDE_0, SEISMIC_RESIDUAL_STRIDE_1, after_router_norm,  \
        SEISMIC_AFTER_ROUTER_NORM_STRIDE_0, eps, k, {}};                                    \
    const projection::Weights<packets::W0> w0{{next_router, KERNEL_W0_LAYOUT(k), k}};       \
    const projection::Weights<packets::W1> w1{{after_router, KERNEL_W1_LAYOUT(k), k}};      \
    const projection::CoherentStore out0{logits, SEISMIC_DIM_EN, 1ul, 0ul}; \
    const projection::CoherentStore out1{logits + SEISMIC_DIM_M * SEISMIC_DIM_EN, SEISMIC_DIM_EN, 1ul, 0ul}

#define PROJECTION_SEGMENT_INPUT_0(name) const auto name = raw0
#define PROJECTION_SEGMENT_INPUT_1(name) const auto name = raw1
#define PROJECTION_SEGMENT_INPUT(s, name) PROJECTION_SEGMENT_INPUT_##s(name)

#define PROJECTION_ORDERED_EPILOGUE(ordered_tiles)                                          \
    do {                                                                                    \
        const uint tid = sg * 32u + lane;                                                   \
        if (arrive::last(arrivals, ordered_tiles, reinterpret_cast<threadgroup uint *>(shared), tid)) { \
            threadgroup_barrier(mem_flags::mem_threadgroup);                       \
            threadgroup int *words = reinterpret_cast<threadgroup int *>(shared);           \
            const int seq = step[SEISMIC_CONST_STEP_LOW * SEISMIC_STEP_STRIDE_0];           \
            constexpr uint K = uint(SEISMIC_DIM_K);                                         \
            for (uint lead = 0u; lead < uint(SEISMIC_DIM_AHEAD); ++lead) {                  \
                arrive::coherent_device<float> plane = logits + ulong(lead) * SEISMIC_DIM_M * SEISMIC_DIM_EN; \
                device const float *bias_p = lead == 0u ? next_bias : after_bias;           \
                const ulong bias_s = lead == 0u ? SEISMIC_NEXT_BIAS_STRIDE_0 : SEISMIC_AFTER_BIAS_STRIDE_0; \
                const int score_fn = int(SEISMIC_PARAM_SCORE); \
                for (uint r = sg; r < m; r += simdgroups) {                                 \
                    const select::Selected<K> sel = select::rank<uint(SEISMIC_DIM_EN), K>(  \
                        score_fn,                                                           \
                        [&](uint e) { return plane[ulong(r) * SEISMIC_DIM_EN + e]; },       \
                        [&](int e) { return bias_p[ulong(e) * bias_s]; }, lane);            \
                    if (lane == 0u) {                                                       \
                        for (uint c = 0u; c < K; ++c)                                       \
                            words[STREAMING_HINT_CHOSEN(r) + c] = sel.winners[K - 1u - c];  \
                    }                                                                       \
                }                                                                           \
                streaming_hint(m, rings + ulong(lead) * SEISMIC_RINGS_STRIDE_0,             \
                    SEISMIC_RINGS_STRIDE_1, seq, words, tid, simdgroups * 32u);             \
                streaming_barrier();                                                        \
            }                                                                               \
        }                                                                                   \
    } while (0)

#include "lib/projection/operator.h"

kernel void routed_hint_select_ordered_gemv(PROJECTION_ARGUMENTS, PROJECTION_ORDERED_GEMV_ARGS) {
    constexpr uint simdgroups = SEISMIC_TUNE_SIMDGROUPS;
    PROJECTION_SEGMENTED_ORDERED_GEMV_ALL;
}
