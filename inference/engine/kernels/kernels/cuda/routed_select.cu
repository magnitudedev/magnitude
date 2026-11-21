// General expert selection (`routed_select`) on the `dense` projection
// operator, single over `router` in the CPU 8-lane reference order; `norm` and
// `router_norm` share the row's RMS inverse.
#define KERNEL_A SEISMIC_ELEMENT_A
#define KERNEL_W0 SEISMIC_ROUTER
#include "lib/projection/operator.cuh"
#include "lib/routed/selection.cuh"

SEISMIC_PROGRAMMATIC_DEPENDENCY

using NormElem = ELEMENT_OF(SEISMIC_NORM);
using RouterRaw = projection::OrderedRms<ELEMENT_OF(SEISMIC_ROUTER_NORM), projection::AllRows>;
using NormRaw = projection::OrderedRms<NormElem, projection::AllRows>;
using LogitStore = projection::Store<element::F32>;

extern "C" __global__ void routed_select_ordered_stage(SEISMIC_KERNEL_PARAMS) {
    seismic_dependency_start();
    const float eps = __uint_as_float(static_cast<unsigned>(SEISMIC_PARAM_EPSILON));
    const RouterRaw router_raw{reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL)),
                               SEISMIC_RESIDUAL_STRIDE_0, SEISMIC_PTR(SEISMIC_BUFFER_ROUTER_NORM), eps, SEISMIC_DIM_H,
                               projection::AllRows{}, SEISMIC_RESIDUAL_STRIDE_1, SEISMIC_ROUTER_NORM_STRIDE_0};
    const NormRaw norm_raw{reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL)),
                           SEISMIC_RESIDUAL_STRIDE_0, SEISMIC_PTR(SEISMIC_BUFFER_NORM), eps, SEISMIC_DIM_H,
                           projection::AllRows{}, SEISMIC_RESIDUAL_STRIDE_1, SEISMIC_NORM_STRIDE_0};
    const element::Out<activation::Act> staged_out = element::out_over_scratch<activation::Act>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED));
    const float inv =
        projection::ordered_stage_row(router_raw, blockIdx.x, SEISMIC_DIM_H, staged_out, SEISMIC_DIM_H, 1);
    routed::publish_ordered_norm_row<activation::Act, NormElem>(
        norm_raw, blockIdx.x, inv, SEISMIC_RESULT_0_OUT, SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1);
}

extern "C" __global__ void routed_select_ordered_gemv(SEISMIC_KERNEL_PARAMS) {
    seismic_dependency_start();
    extern __shared__ uint4 dynamic_shared[];
    projection::u8 *shared_raw = reinterpret_cast<projection::u8 *>(dynamic_shared);
    const unsigned m = static_cast<unsigned>(SEISMIC_DIM_M);
    const unsigned long long k = SEISMIC_DIM_H;
    const float eps = __uint_as_float(static_cast<unsigned>(SEISMIC_PARAM_EPSILON));
    const RouterRaw router_raw{reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL)),
                               SEISMIC_RESIDUAL_STRIDE_0, SEISMIC_PTR(SEISMIC_BUFFER_ROUTER_NORM), eps, k,
                               projection::AllRows{}, SEISMIC_RESIDUAL_STRIDE_1, SEISMIC_ROUTER_NORM_STRIDE_0};
    const NormRaw norm_raw{reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL)),
                           SEISMIC_RESIDUAL_STRIDE_0, SEISMIC_PTR(SEISMIC_BUFFER_NORM), eps, k,
                           projection::AllRows{}, SEISMIC_RESIDUAL_STRIDE_1, SEISMIC_NORM_STRIDE_0};
    const element::Out<element::F32> logits_out = element::out_over_scratch<element::F32>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_LOGITS));
    projection::op::ordered_gemv_block<false>(
        shared_raw, router_raw, SEISMIC_RESULT_0_OUT, 0, 1, false, m, k, SEISMIC_DIM_E, blockIdx.x,
        KERNEL_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_ROUTER)), LogitStore{logits_out, SEISMIC_DIM_E, 0});
    if (blockIdx.x == 0) {
        const float *smem_inv = reinterpret_cast<const float *>(shared_raw);
        routed::publish_ordered_norm_rows<activation::Act, NormElem>(
            norm_raw, m, smem_inv, SEISMIC_RESULT_0_OUT, SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1);
    }
    const unsigned ordered_tiles = static_cast<unsigned>((SEISMIC_DIM_E + 31ull) / 32ull);
    __shared__ unsigned last;
    unsigned *arrivals = reinterpret_cast<unsigned *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_ARRIVALS));
    if (!arrive::last(arrivals, ordered_tiles, &last))
        return;
    const float *logits = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_LOGITS));
    const float *bias = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_BIAS));
    const float *expert_scale = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_SCALE));
    int *routes = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES));
    const auto weights = SEISMIC_OUT_WEIGHTS;
    for (unsigned row = threadIdx.x / 32; row < m; row += blockDim.x / 32) {
        const float *row_logits = logits + static_cast<unsigned long long>(row) * SEISMIC_DIM_E;
        select_row(row, [&](unsigned e) { return __ldcg(row_logits + e); }, bias, expert_scale, routes, weights,
                   seismic_words_value);
    }
}

extern "C" __global__ void routed_select_ordered_gemm(SEISMIC_KERNEL_PARAMS) {
    seismic_dependency_start();
    extern __shared__ uint4 dynamic_shared[];
    const element::Out<element::F32> logits_out = element::out_over_scratch<element::F32>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_LOGITS));
    projection::op::ordered_gemm_block(
        reinterpret_cast<projection::u8 *>(dynamic_shared), SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED), SEISMIC_DIM_H,
        1, static_cast<unsigned>(SEISMIC_DIM_M), SEISMIC_DIM_H, SEISMIC_DIM_E, blockIdx.y, blockIdx.x,
        KERNEL_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_ROUTER)), LogitStore{logits_out, SEISMIC_DIM_E, 0});
}

extern "C" __global__ void routed_select_select(SEISMIC_KERNEL_PARAMS) {
    seismic_dependency_start();
    const unsigned long long row = static_cast<unsigned long long>(blockIdx.x) * 8ull + threadIdx.x / 32;
    if (row >= SEISMIC_DIM_M)
        return;
    const float *row_logits =
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_LOGITS)) + row * SEISMIC_DIM_E;
    select_row(row, [&](unsigned e) { return row_logits[e]; },
               reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_BIAS)),
               reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_SCALE)),
               reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES)), SEISMIC_OUT_WEIGHTS, seismic_words_value);
}
