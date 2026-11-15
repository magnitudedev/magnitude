// Routing (`routed_route`) on the `dense` projection operator, segmented over
// `(router, shared_router)` in the CPU 8-lane reference order.
#define KERNEL_W0 SEISMIC_ROUTER
#define KERNEL_W1 SEISMIC_SHARED_ROUTER
#include "lib/projection/operator.cuh"
#include "lib/routed/route.cuh"

SEISMIC_PROGRAMMATIC_DEPENDENCY

using Raw = projection::OrderedInput<ELEMENT_OF(SEISMIC_NORM), projection::AllRows, SEISMIC_DIM_NM, SEISMIC_DIM_PT>;
using LogitStore = projection::Store<element::F32>;
using Segments = projection::op::SegmentSet<0, projection::op::Types<packets::W0, packets::W1>,
                                            projection::op::Types<LogitStore, LogitStore>,
                                            projection::op::empty_mask(SEISMIC_DIM_E, 1)>;

#define MAKE_RAW                                                                                              \
    Raw {                                                                                                     \
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL)), SEISMIC_RESIDUAL_STRIDE_0,      \
            SEISMIC_PTR(SEISMIC_BUFFER_NORM), __uint_as_float(static_cast<unsigned>(SEISMIC_PARAM_EPS)),      \
            SEISMIC_DIM_H, projection::AllRows{}, SEISMIC_RESIDUAL_STRIDE_1, SEISMIC_NORM_STRIDE_0            \
    }

#define MAKE_SEGMENTS(logits_out)                                                                             \
    Segments {                                                                                                \
        {SEISMIC_DIM_E, 1},                                                                                   \
            {KERNEL_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_ROUTER)),                                                \
             KERNEL_W1_AT(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_ROUTER))},                                        \
        {LogitStore{logits_out, SEISMIC_DIM_E + 1, 0},                                                        \
         LogitStore{logits_out, SEISMIC_DIM_E + 1, SEISMIC_DIM_E}}                                            \
    }

extern "C" __global__ void routed_route_ordered_stage(SEISMIC_KERNEL_PARAMS) {
    seismic_dependency_start();
    projection::ordered_stage_row(MAKE_RAW, blockIdx.x, SEISMIC_DIM_H, SEISMIC_RESULT_0_OUT,
                                  SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1);
}

extern "C" __global__ void routed_route_ordered_gemv(SEISMIC_KERNEL_PARAMS) {
    extern __shared__ uint4 dynamic_shared[];
    const Raw raw = MAKE_RAW;
    const element::Out<element::F32> logits_out = element::out_over_scratch<element::F32>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_LOGITS));
    const unsigned ordered_tiles = static_cast<unsigned>(
        projection::op::ordered_gemv_segmented<true, SEISMIC_DIM_H>(
            reinterpret_cast<projection::u8 *>(dynamic_shared), raw, raw, nullptr, SEISMIC_RESULT_0_OUT,
            SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1, static_cast<unsigned>(SEISMIC_DIM_M),
            MAKE_SEGMENTS(logits_out), 0u));
    __shared__ unsigned last;
    unsigned *arrivals = reinterpret_cast<unsigned *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_ARRIVALS));
    if (!arrive::last(arrivals, ordered_tiles, &last))
        return;
    const float *logits = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_LOGITS));
    const route::Selection sel = ROUTE_SELECTION;
    const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32, warps = blockDim.x / 32;
    for (unsigned row = warp; row < static_cast<unsigned>(SEISMIC_DIM_M); row += warps) {
        const float *row_logits = logits + static_cast<unsigned long long>(row) * (SEISMIC_DIM_E + 1);
        route::select_row([&](unsigned long long e) { return __ldcg(row_logits + e); }, sel, row, lane,
                          seismic_words_value);
    }
}

extern "C" __global__ void routed_route_ordered_gemm(SEISMIC_KERNEL_PARAMS) {
    extern __shared__ uint4 dynamic_shared[];
    const element::Out<element::F32> logits_out = element::out_over_scratch<element::F32>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_LOGITS));
    projection::op::ordered_gemm_segmented(
        reinterpret_cast<projection::u8 *>(dynamic_shared), SEISMIC_RESULT_0_OUT.stored(),
        SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1, static_cast<unsigned>(SEISMIC_DIM_M), SEISMIC_DIM_H,
        MAKE_SEGMENTS(logits_out), 0u);
}

extern "C" __global__ void routed_route_select(SEISMIC_KERNEL_PARAMS) {
    seismic_dependency_start();
    const unsigned long long row = static_cast<unsigned long long>(blockIdx.x) * 8ull + threadIdx.x / 32;
    if (row >= SEISMIC_DIM_M)
        return;
    const float *row_logits =
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_LOGITS)) + row * (SEISMIC_DIM_E + 1);
    route::select_row([&](unsigned long long e) { return row_logits[e]; }, ROUTE_SELECTION, row, threadIdx.x % 32,
                      seismic_words_value);
}
