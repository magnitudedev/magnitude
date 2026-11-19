// General expert selection (`routed_select`) on the `dense` projection
// operator, single over `router` in the CPU 8-lane reference order; `norm` and
// `router_norm` share the row's RMS inverse.
#define KERNEL_A SEISMIC_ELEMENT_A
#define KERNEL_W0 SEISMIC_ROUTER
#include "lib/projection/projection.h"
#include "lib/routed/selection.h"

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
    SEISMIC_RESULT_0_PARAM(normalized_cells),                                               \
    arrive::coherent_device<float> logits [[buffer(SEISMIC_BUFFER_SCRATCH_LOGITS)]],        \
    device uchar *staged [[buffer(SEISMIC_BUFFER_SCRATCH_STAGED)]],                         \
    device atomic_uint *arrivals [[buffer(SEISMIC_BUFFER_SCRATCH_ARRIVALS)]],               \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

#define PROJECTION_STAGE_K SEISMIC_DIM_H

#define PROJECTION_OPERANDS(group)                                                          \
    const uint m = uint(SEISMIC_DIM_M), n = uint(SEISMIC_DIM_E), k = uint(SEISMIC_DIM_H);   \
    const float eps = as_type<float>(uint(SEISMIC_PARAM_EPSILON));                          \
    const projection::OrderedRms<Act, RouterNorm, projection::AllRows> router_raw{          \
        residual, SEISMIC_RESIDUAL_STRIDE_0, SEISMIC_RESIDUAL_STRIDE_1, router_norm,        \
        SEISMIC_ROUTER_NORM_STRIDE_0, eps, k, {}};                                          \
    const projection::OrderedRms<Act, Norm, projection::AllRows> norm_raw{                  \
        residual, SEISMIC_RESIDUAL_STRIDE_0, SEISMIC_RESIDUAL_STRIDE_1, norm,               \
        SEISMIC_NORM_STRIDE_0, eps, k, {}};                                                 \
    const projection::Plain<Act, projection::AllRows> in{staged, k, 1, k, {}};              \
    const projection::Weights<packets::W0> w{{router, KERNEL_W0_LAYOUT(k), k}};             \
    const projection::CoherentStore out{logits, SEISMIC_DIM_E, 1ul, 0ul}

#define PROJECTION_GEMV_INPUT(x) const auto x = router_raw

#define PROJECTION_STAGE_EXTRA(row, row_inv)                                                \
    do {                                                                                    \
        const uint tid = sg * 32u + lane;                                                   \
        const uint h = uint(SEISMIC_DIM_H), h8 = h & ~7u;                                   \
        const element::Out<Act> pub(normalized_cells);                                      \
        if (SEISMIC_RESULT_0_STRIDE_1 == 1ul && (SEISMIC_RESULT_0_STRIDE_0 & 7ul) == 0ul) { \
            for (uint c = tid * 8u; c < h8; c += 2048u) {                                   \
                float4 even, odd;                                                           \
                norm_raw.load8(row, c, (row_inv), even, odd);                               \
                pub.put8(ulong(row) * SEISMIC_RESULT_0_STRIDE_0 + ulong(c), even, odd);     \
            }                                                                               \
            for (uint c = h8 + tid; c < h; c += 256u)                                       \
                pub.put(ulong(row) * SEISMIC_RESULT_0_STRIDE_0 + ulong(c),                  \
                    norm_raw.ordered_at(row, c, (row_inv)));                                \
        } else {                                                                            \
            for (uint c = tid; c < h; c += 256u) {                                          \
                pub.put(ulong(row) * SEISMIC_RESULT_0_STRIDE_0 + ulong(c) * SEISMIC_RESULT_0_STRIDE_1, \
                    norm_raw.ordered_at(row, c, (row_inv)));                                \
            }                                                                               \
        }                                                                                   \
    } while (0)

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
        if (arrive::last(arrivals, ordered_tiles, reinterpret_cast<threadgroup uint *>(shared), sg * 32u + lane)) {           \
            for (uint r = sg; r < m; r += simdgroups) {                                     \
                select_row(r, logits, bias, expert_scale, routes,                           \
                    element::Out<element::F32>(weights_cells), seismic_words, lane);        \
            }                                                                               \
        }                                                                                   \
    } while (0)

#include "lib/projection/operator.h"

kernel void routed_select_ordered_stage(PROJECTION_ARGUMENTS, PROJECTION_ORDERED_STAGE_ARGS) {
    PROJECTION_OPERANDS(0u);
    PROJECTION_ORDERED_STAGE_BODY(router_raw, element::out_over_scratch<Act>(staged), ulong(k), 1ul);
}

kernel void routed_select_ordered_gemv(PROJECTION_ARGUMENTS, PROJECTION_ORDERED_GEMV_ARGS) {
    constexpr uint simdgroups = SEISMIC_TUNE_SIMDGROUPS;
    PROJECTION_ORDERED_GEMV_SINGLE;
}

kernel void routed_select_ordered_gemm(PROJECTION_ARGUMENTS, PROJECTION_ORDERED_GEMM_ARGS) {
    PROJECTION_ORDERED_GEMM_SINGLE;
}

kernel void routed_select_select(PROJECTION_ARGUMENTS,
    uint group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const ulong row = ulong(group) * 8ul + sg;
    if (row < SEISMIC_DIM_M) {
        select_row(row, logits, bias, expert_scale, routes,
            element::Out<element::F32>(weights_cells), seismic_words, lane);
    }
}
