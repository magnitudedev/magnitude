// Decode routing with the shared expert's gate/up expansion (`routed_route_shared`)
// on the `dense` projection operator, segmented over
// `(router, shared_router, (shared_gate, shared_up))`.
#define KERNEL_W0 SEISMIC_ROUTER
#define KERNEL_W1 SEISMIC_SHARED_ROUTER
#define KERNEL_W2 SEISMIC_SHARED_GATE
#define KERNEL_W3 SEISMIC_SHARED_UP
#include "lib/projection/operator.cuh"
#include "lib/routed/route.cuh"

SEISMIC_PROGRAMMATIC_DEPENDENCY

using Raw = projection::OrderedInput<ELEMENT_OF(SEISMIC_NORM), projection::AllRows, SEISMIC_DIM_NM, SEISMIC_DIM_PT>;
constexpr bool NVFP4_SHARED =
    projection::OperandRecipe<packets::W2_OPERANDS_NVFP4, packets::W3_OPERANDS_NVFP4>::NVFP4;
using UnordPro = projection::Operands<NVFP4_SHARED, Raw>;
using LogitStore = projection::Store<element::F32>;
using SharedEpi = projection::SiluMul<element::Act>;
using Segments = projection::op::SegmentSet<0, projection::op::Types<packets::W0, packets::W1, packets::W2, packets::W3>,
                                            projection::op::Types<LogitStore, LogitStore, SharedEpi>,
                                            projection::op::empty_mask(SEISMIC_DIM_E, 1, SEISMIC_DIM_S)>;

#define MAKE_RAW                                                                                              \
    Raw {                                                                                                     \
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL)), SEISMIC_RESIDUAL_STRIDE_0,      \
            SEISMIC_PTR(SEISMIC_BUFFER_NORM), __uint_as_float(static_cast<unsigned>(SEISMIC_PARAM_EPS)),      \
            SEISMIC_DIM_H, projection::AllRows{}, SEISMIC_RESIDUAL_STRIDE_1, SEISMIC_NORM_STRIDE_0            \
    }

#define MAKE_SEGMENTS(logits_out)                                                                             \
    Segments {                                                                                                \
        {SEISMIC_DIM_E, 1, SEISMIC_DIM_S},                                                                    \
            {KERNEL_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_ROUTER)),                                                \
             KERNEL_W1_AT(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_ROUTER)),                                        \
             KERNEL_W2_AT(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_GATE)),                                          \
             KERNEL_W3_AT(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_UP))},                                            \
        {LogitStore{logits_out, SEISMIC_DIM_E + 1, 0},                                                        \
         LogitStore{logits_out, SEISMIC_DIM_E + 1, SEISMIC_DIM_E},                                            \
         SharedEpi{SEISMIC_RESULT_2_OUT, SEISMIC_RESULT_2_STRIDE_0}}                                          \
    }

extern "C" __global__ void routed_route_shared_ordered_stage(SEISMIC_KERNEL_PARAMS) {
    seismic_dependency_start();
    const UnordPro unord_pro{MAKE_RAW};
    projection::form_row(unord_pro, blockIdx.x, SEISMIC_DIM_H, SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED));
}

extern "C" __global__ void routed_route_shared_ordered_gemv(SEISMIC_KERNEL_PARAMS) {
    extern __shared__ uint4 dynamic_shared[];
    const Raw raw = MAKE_RAW;
    const UnordPro unord_pro{raw};
    const element::Out<element::F32> logits_out = element::out_over_scratch<element::F32>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_LOGITS));
    const Segments segs = MAKE_SEGMENTS(logits_out);
    const unsigned ordered_tiles = static_cast<unsigned>(
        projection::op::ordered_gemv_segmented<true, SEISMIC_DIM_H, SEISMIC_TUNE_TPW, SEISMIC_TUNE_KSPLIT>(
            reinterpret_cast<projection::u8 *>(dynamic_shared), raw, unord_pro,
            SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED), SEISMIC_RESULT_0_OUT, SEISMIC_RESULT_0_STRIDE_0,
            SEISMIC_RESULT_0_STRIDE_1, static_cast<unsigned>(SEISMIC_DIM_M), segs, 0u));
    if (blockIdx.x >= ordered_tiles)
        return;
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
