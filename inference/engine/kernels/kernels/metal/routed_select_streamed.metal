// Expert streaming's decode general selection (`routed_select_streamed`) on the
// `dense` projection operator, single over `router`, followed by resolve,
// assignment and front-ring publication in the selecting threadgroup.
#define KERNEL_A SEISMIC_ELEMENT_A
#define KERNEL_W0 SEISMIC_ROUTER
#include "lib/projection/projection.h"
#include "lib/routed/selection.h"
#include "lib/streaming/control.h"
#include "../streaming/hint.hpp"
#include "../streaming/resolve.hpp"
#include "lib/routed/keys.h"

typedef activation::Act Act;
typedef ELEMENT_OF(SEISMIC_NORM) Norm;
typedef ELEMENT_OF(SEISMIC_ROUTER_NORM) RouterNorm;

#define PROJECTION_ARGUMENTS                                                                \
    device const float *residual [[buffer(SEISMIC_BUFFER_RESIDUAL)]],                       \
    device const uchar *norm [[buffer(SEISMIC_BUFFER_NORM)]],                               \
    device const uchar *router_norm [[buffer(SEISMIC_BUFFER_ROUTER_NORM)]],                 \
    device const uchar *router [[buffer(SEISMIC_BUFFER_ROUTER)]],                           \
    device const float *bias [[buffer(SEISMIC_BUFFER_BIAS)]],                               \
    device const float *expert_scale [[buffer(SEISMIC_BUFFER_EXPERT_SCALE)]],               \
    device int *routes [[buffer(SEISMIC_BUFFER_ROUTES)]],                                   \
    SEISMIC_OUT_WEIGHTS_PARAM(weights_cells),                                               \
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
    arrive::coherent_device<float> logits [[buffer(SEISMIC_BUFFER_SCRATCH_LOGITS)]],        \
    device atomic_uint *arrivals [[buffer(SEISMIC_BUFFER_SCRATCH_ARRIVALS)]],               \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

#define PROJECTION_OPERANDS(group)                                                          \
    const uint m = uint(SEISMIC_DIM_M), n = uint(SEISMIC_DIM_E), k = uint(SEISMIC_DIM_H);   \
    const float eps = as_type<float>(uint(SEISMIC_PARAM_EPSILON));                          \
    const projection::OrderedRms<Act, RouterNorm, projection::AllRows> router_raw{          \
        residual, SEISMIC_RESIDUAL_STRIDE_0, SEISMIC_RESIDUAL_STRIDE_1, router_norm,        \
        SEISMIC_ROUTER_NORM_STRIDE_0, eps, k, {}};                                          \
    const projection::OrderedRms<Act, Norm, projection::AllRows> norm_raw{                  \
        residual, SEISMIC_RESIDUAL_STRIDE_0, SEISMIC_RESIDUAL_STRIDE_1, norm,               \
        SEISMIC_NORM_STRIDE_0, eps, k, {}};                                                 \
    const projection::Weights<packets::W0> w{{router, KERNEL_W0_LAYOUT(k), k}};             \
    const projection::CoherentStore out{logits, SEISMIC_DIM_E, 1ul, 0ul}

#define PROJECTION_GEMV_INPUT(x) const auto x = router_raw

#define PROJECTION_ORDERED_FIRST_TILE()                                                     \
    do {                                                                                    \
        threadgroup const float *smem_inv = reinterpret_cast<threadgroup const float *>(shared); \
        const uint tid = sg * 32u + lane, threads = simdgroups * 32u;                       \
        const element::Out<Act> pub(normalized_cells);                                      \
        if (SEISMIC_RESULT_0_STRIDE_1 == 1ul && (SEISMIC_RESULT_0_STRIDE_0 & 7ul) == 0ul && (k & 7u) == 0u) { \
            const uint vecs = k >> 3u;                                                      \
            for (uint idx = tid; idx < m * vecs; idx += threads) {                          \
                const uint r = idx / vecs, c = (idx % vecs) * 8u;                           \
                float4 even, odd;                                                           \
                norm_raw.load8(r, c, smem_inv[r], even, odd);                               \
                pub.put8(ulong(r) * SEISMIC_RESULT_0_STRIDE_0 + ulong(c), even, odd);       \
            }                                                                               \
        } else {                                                                            \
            for (uint idx = tid; idx < m * k; idx += threads) {                             \
                const uint r = idx / k, c = idx % k;                                        \
                pub.put(ulong(r) * SEISMIC_RESULT_0_STRIDE_0 + ulong(c) * SEISMIC_RESULT_0_STRIDE_1, \
                    norm_raw.ordered_at(r, c, smem_inv[r]));                                \
            }                                                                               \
        }                                                                                   \
    } while (0)

#define PROJECTION_ORDERED_EPILOGUE(ordered_tiles)                                          \
    do {                                                                                    \
        const uint tid = sg * 32u + lane;                                                   \
        if (arrive::last(arrivals, ordered_tiles, reinterpret_cast<threadgroup uint *>(shared), tid)) { \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                \
            const element::Out<element::F32> weights_out(weights_cells);                    \
            ROUTED_KEYS_RECORD;                                                             \
            for (uint r = sg; r < m; r += simdgroups) {                                     \
                select_row(r, logits, bias, expert_scale, routes,                           \
                    weights_out, seismic_words, lane, ROUTED_ROW_KEYS(r));                  \
            }                                                                               \
            threadgroup int *words = reinterpret_cast<threadgroup int *>(shared);           \
            const bool whole = streaming_load(STREAMING_AT(policy,                          \
                SEISMIC_CONST_POLICY_WHOLE * SEISMIC_POLICY_STRIDE_0)) != 0;                \
            streaming_resolve(routes, SEISMIC_ROUTES_STRIDE_0, SEISMIC_ROUTES_STRIDE_1,     \
                m * uint(SEISMIC_DIM_K),                                                    \
                reinterpret_cast<device const int *>(weights_out.stored()),                 \
                SEISMIC_WEIGHTS_STRIDE_0, SEISMIC_WEIGHTS_STRIDE_1,                         \
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

kernel void routed_select_streamed_ordered_gemv(PROJECTION_ARGUMENTS, PROJECTION_ORDERED_GEMV_ARGS) {
    constexpr uint simdgroups = SEISMIC_TUNE_SIMDGROUPS;
    PROJECTION_ORDERED_GEMV_SINGLE;
}
