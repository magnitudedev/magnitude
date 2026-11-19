// Ahead-of-step expert hints (`routed_hint_route`) on the `dense` projection
// operator, segmented over `(next_router, after_router)` in the CPU 8-lane
// reference order; segment 1 is zeroed when `AHEAD < 2`.
#define KERNEL_A SEISMIC_ELEMENT_A
#define KERNEL_W0 SEISMIC_NEXT_ROUTER
#define KERNEL_W1 SEISMIC_AFTER_ROUTER
#include "lib/projection/operator.cuh"
#include "lib/core/activation.cuh"
#include "lib/core/arrive.cuh"
#include "lib/routed/ranked.cuh"
#include "lib/streaming/control.cuh"
#include "../streaming/hint.hpp"

SEISMIC_PROGRAMMATIC_DEPENDENCY

using Raw = projection::OrderedInput<ELEMENT_OF(SEISMIC_NEXT_NORM), projection::AllRows, SEISMIC_DIM_NM, SEISMIC_DIM_PT>;
using LogitStore = projection::Store<element::F32>;
using Segments = projection::op::SegmentSet<1, projection::op::Types<packets::W0, packets::W1>,
                                            projection::op::Types<LogitStore, LogitStore>,
                                            projection::op::empty_mask(SEISMIC_DIM_EN, SEISMIC_DIM_EN)>;

struct HintPro {
    Raw raw0, raw1;
    __device__ __forceinline__ Raw operator()(projection::op::SegmentIndex<0>) const { return raw0; }
    template <int I> __device__ __forceinline__ Raw operator()(projection::op::SegmentIndex<I>) const { return raw1; }
};

extern "C" __global__ void routed_hint_route_ordered_gemv(SEISMIC_KERNEL_PARAMS) {
    extern __shared__ uint4 dynamic_shared[];
    const unsigned m = static_cast<unsigned>(SEISMIC_DIM_M);
    const float eps = __uint_as_float(static_cast<unsigned>(SEISMIC_PARAM_EPS));
    const Raw raw0{reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL)), SEISMIC_RESIDUAL_STRIDE_0,
                   SEISMIC_PTR(SEISMIC_BUFFER_NEXT_NORM), eps, SEISMIC_DIM_H, projection::AllRows{},
                   SEISMIC_RESIDUAL_STRIDE_1, SEISMIC_NEXT_NORM_STRIDE_0};
    const Raw raw1{reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL)), SEISMIC_RESIDUAL_STRIDE_0,
                   SEISMIC_PTR(SEISMIC_BUFFER_AFTER_NORM), eps, SEISMIC_DIM_H, projection::AllRows{},
                   SEISMIC_RESIDUAL_STRIDE_1, SEISMIC_AFTER_NORM_STRIDE_0};
    const element::Out<element::F32> logits0 = element::out_over_scratch<element::F32>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_LOGITS));
    const element::Out<element::F32> logits1 = logits0.offset(SEISMIC_DIM_M * SEISMIC_DIM_EN);
    const Segments segs{{SEISMIC_DIM_EN, SEISMIC_DIM_EN},
                        {KERNEL_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_NEXT_ROUTER)),
                         KERNEL_W1_AT(SEISMIC_PTR(SEISMIC_BUFFER_AFTER_ROUTER))},
                        {LogitStore{logits0, SEISMIC_DIM_EN, 0}, LogitStore{logits1, SEISMIC_DIM_EN, 0}}};
    const unsigned zero_mask = SEISMIC_DIM_AHEAD < 2 ? 2u : 0u;
    const unsigned ordered_tiles = static_cast<unsigned>(
        projection::op::ordered_gemv_segmented_each<false, SEISMIC_DIM_H>(
            reinterpret_cast<projection::u8 *>(dynamic_shared), HintPro{raw0, raw1}, raw0, nullptr,
            element::Out<activation::Act>(nullptr), 0, 1, m, segs, zero_mask));
    __shared__ unsigned last;
    unsigned *arrivals = reinterpret_cast<unsigned *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_ARRIVALS));
    if (!arrive::last(arrivals, ordered_tiles, &last))
        return;
    int *words = reinterpret_cast<int *>(dynamic_shared);
    const float *logits = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_LOGITS));
    const int *step = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_STEP));
    int *rings = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_RINGS));
    const int seq = step[SEISMIC_CONST_STEP_LOW * SEISMIC_STEP_STRIDE_0];
    const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32, warps = blockDim.x / 32;
    for (unsigned lead = 0; lead < static_cast<unsigned>(SEISMIC_DIM_AHEAD); ++lead) {
        const float *plane = logits + static_cast<unsigned long long>(lead) * SEISMIC_DIM_M * SEISMIC_DIM_EN;
        for (unsigned row = warp; row < m; row += warps) {
            const float *row_logits = plane + static_cast<unsigned long long>(row) * SEISMIC_DIM_EN;
            const auto ranked = selection::rank_row<static_cast<unsigned>(SEISMIC_DIM_EN), static_cast<unsigned>(SEISMIC_DIM_K)>(
                [&](unsigned e) { return __ldcg(row_logits + e); }, lane);
            if (lane == 0) {
                for (unsigned c = 0; c < static_cast<unsigned>(SEISMIC_DIM_K); ++c)
                    words[STREAMING_HINT_CHOSEN(row) + c] = ranked.winners[c];
            }
        }
        streaming_hint(m, rings + static_cast<unsigned long long>(lead) * SEISMIC_RINGS_STRIDE_0,
                       SEISMIC_RINGS_STRIDE_1, seq, words, threadIdx.x, blockDim.x);
        streaming_barrier();
    }
}
