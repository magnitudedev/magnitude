// Expert streaming's decode general selection (`routed_select_streamed`) on the
// `dense` projection operator, single over `router`, followed by resolve and
// assignment in the selecting block.
#define KERNEL_A SEISMIC_ELEMENT_A
#define KERNEL_W0 SEISMIC_ROUTER
#include "lib/projection/operator.cuh"
#include "lib/routed/selection.cuh"
#include "lib/streaming/control.cuh"
#include "../streaming/resolve.hpp"
#include "lib/routed/resolve.cuh"

SEISMIC_PROGRAMMATIC_DEPENDENCY

using RouterRaw = projection::OrderedRms<ELEMENT_OF(SEISMIC_ROUTER_NORM), projection::AllRows>;
using NormRaw = projection::OrderedRms<ELEMENT_OF(SEISMIC_NORM), projection::AllRows>;
using LogitStore = projection::Store<element::F32>;

extern "C" __global__ void routed_select_streamed_ordered_gemv(SEISMIC_KERNEL_PARAMS) {
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
        const auto pub_out = SEISMIC_RESULT_0_OUT;
        for (unsigned long long idx = threadIdx.x; idx < m * k; idx += blockDim.x) {
            const unsigned r = static_cast<unsigned>(idx / k);
            const unsigned long long c = idx % k;
            pub_out.put(static_cast<unsigned long long>(r) * SEISMIC_RESULT_0_STRIDE_0 + c * SEISMIC_RESULT_0_STRIDE_1,
                        norm_raw.ordered_at(r, c, smem_inv[r]));
        }
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
    int *words = reinterpret_cast<int *>(dynamic_shared);
    const ResolvePorts bank = ROUTED_RESOLVE_PORTS;
    const bool whole =
        streaming_load(STREAMING_AT(bank.policy, SEISMIC_CONST_POLICY_WHOLE * SEISMIC_POLICY_STRIDE_0)) != 0;
    routed_resolve(bank, whole, words, seismic_words_value);
}
