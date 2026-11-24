// Expert streaming's decode routing (`routed_route_streamed`) on the `dense`
// projection operator, segmented over `(router, shared_router, (shared_gate, shared_up))`,
// followed by resolve and assignment in the selecting block.
#define KERNEL_A SEISMIC_ELEMENT_A
#define KERNEL_W0 SEISMIC_ROUTER
#define KERNEL_W1 SEISMIC_SHARED_ROUTER
#define KERNEL_W2 SEISMIC_SHARED_GATE
#define KERNEL_W3 SEISMIC_SHARED_UP
#include "lib/projection/operator.cuh"
#include "lib/routed/route.cuh"
#include "lib/streaming/control.cuh"
#include "../streaming/hint.hpp"
#include "../streaming/resolve.hpp"
#include "../streaming/site.hpp"
#include "lib/routed/resolve.cuh"

SEISMIC_PROGRAMMATIC_DEPENDENCY

using Raw = projection::OrderedInput<ELEMENT_OF(SEISMIC_NORM), projection::AllRows, SEISMIC_DIM_NM, SEISMIC_DIM_PT>;
constexpr bool NVFP4_SHARED =
    projection::OperandRecipe<packets::W2_OPERANDS_NVFP4, packets::W3_OPERANDS_NVFP4>::NVFP4;
using UnordPro = projection::Operands<NVFP4_SHARED, Raw>;
using LogitStore = projection::Store<element::F32>;
using SharedEpi = projection::SiluMul<activation::Act>;
using Segments = projection::op::SegmentSet<0, projection::op::Types<packets::W0, packets::W1, packets::W2, packets::W3>,
                                            projection::op::Types<LogitStore, LogitStore, SharedEpi>,
                                            projection::op::empty_mask(SEISMIC_DIM_E, 1, SEISMIC_DIM_S)>;

#define MAKE_RAW                                                                                              \
    Raw {                                                                                                     \
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL)), SEISMIC_RESIDUAL_STRIDE_0,      \
            SEISMIC_PTR(SEISMIC_BUFFER_NORM), __uint_as_float(static_cast<unsigned>(SEISMIC_PARAM_EPS)),      \
            SEISMIC_DIM_H, projection::AllRows{}, SEISMIC_RESIDUAL_STRIDE_1, SEISMIC_NORM_STRIDE_0            \
    }

constexpr unsigned long long LOGITS_STRIDE = (SEISMIC_DIM_E + 4ull) & ~3ull;

#define MAKE_SEGMENTS(logits_out)                                                                             \
    Segments {                                                                                                \
        {SEISMIC_DIM_E, 1, SEISMIC_DIM_S},                                                                    \
            {KERNEL_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_ROUTER)),                                                \
             KERNEL_W1_AT(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_ROUTER)),                                        \
             KERNEL_W2_AT(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_GATE)),                                          \
             KERNEL_W3_AT(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_UP))},                                            \
        {LogitStore{logits_out, LOGITS_STRIDE, 0},                                                            \
         LogitStore{logits_out, LOGITS_STRIDE, SEISMIC_DIM_E},                                                \
         SharedEpi{SEISMIC_RESULT_2_OUT, SEISMIC_RESULT_2_STRIDE_0}}                                          \
    }

extern "C" __global__ void routed_route_streamed_ordered_stage(SEISMIC_KERNEL_PARAMS) {
    seismic_dependency_start();
    const UnordPro unord_pro{MAKE_RAW};
    projection::form_row(unord_pro, blockIdx.x, SEISMIC_DIM_H, SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED));
}

extern "C" __global__ void routed_route_streamed_ordered_gemv(SEISMIC_KERNEL_PARAMS) {
    extern __shared__ uint4 dynamic_shared[];
    const Raw raw = MAKE_RAW;
    const UnordPro unord_pro{raw};
    const element::Out<element::F32> logits_out = element::out_over_scratch<element::F32>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_LOGITS));
    const Segments segs = MAKE_SEGMENTS(logits_out);
    const unsigned ordered_tiles = static_cast<unsigned>(
        projection::op::ordered_gemv_segmented<true, SEISMIC_DIM_H, SEISMIC_DIM_E + 1, SEISMIC_TUNE_TPW, SEISMIC_TUNE_KSPLIT>(
            reinterpret_cast<projection::u8 *>(dynamic_shared), raw, unord_pro,
            SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED), SEISMIC_RESULT_0_OUT, SEISMIC_RESULT_0_STRIDE_0,
            SEISMIC_RESULT_0_STRIDE_1, static_cast<unsigned>(SEISMIC_DIM_M), segs, 0u));
    if (blockIdx.x >= ordered_tiles)
        return;
    __shared__ unsigned last;
    unsigned *arrivals = reinterpret_cast<unsigned *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_ARRIVALS));
    if (!arrive::last(arrivals, ordered_tiles, &last))
        return;
    float *logits = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_LOGITS));
    const route::Selection sel = ROUTE_SELECTION;
    const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32, warps = blockDim.x / 32;
    // Stage each row's ranked keys in place in device scratch `logits[row, 0..E)`
    // (dead once `rank_row` loads it) so zero pinned-host PCIe writes sit in
    // the write buffer when `routed_resolve` executes `fence.acq_rel.sys`.
    for (unsigned row = warp; row < static_cast<unsigned>(SEISMIC_DIM_M); row += warps) {
        float *row_logits = logits + static_cast<unsigned long long>(row) * LOGITS_STRIDE;
        route::select_row([&](unsigned long long e) { return __ldcg(row_logits + e); }, sel, row, lane,
                          seismic_words_value, [&](unsigned e, float key) { row_logits[e] = key; });
    }
    int *words = reinterpret_cast<int *>(dynamic_shared);
    const ResolvePorts bank =
        ROUTED_RESOLVE_PORTS(SEISMIC_OUT_SCORES.stored(), SEISMIC_SCORES_STRIDE_0, SEISMIC_SCORES_STRIDE_1);
    routed_resolve(bank, false, words, seismic_words_value);
    if (SEISMIC_DIM_SF > 0 && SEISMIC_PARAM_SITED != 0) {
        __syncthreads();
        if (SEISMIC_PARAM_SITED == 2 || bank.land_grid[0] == 0) {
            const unsigned choices = static_cast<unsigned>(SEISMIC_DIM_M) * static_cast<unsigned>(SEISMIC_DIM_K);
            __shared__ unsigned any_sited;
            if (threadIdx.x == 0)
                any_sited = 0u;
            __syncthreads();
            bool local_sited = false;
            for (unsigned c = threadIdx.x; c < choices; c += blockDim.x) {
                if (streaming_site_choice(bank.sources, SEISMIC_SOURCES_STRIDE_0, SEISMIC_SOURCES_STRIDE_1,
                        SEISMIC_SOURCES_STRIDE_2, c))
                    local_sited = true;
            }
            if (__any_sync(0xffffffffu, local_sited) && (threadIdx.x & 31u) == 0u)
                atomicOr(&any_sited, 1u);
            __syncthreads();
            if (any_sited != 0u) {
                const u64 q = streaming_step(bank.step, SEISMIC_STEP_STRIDE_0);
                const u64 slot = q % (u64)SEISMIC_DIM_SF;
                using A = ELEMENT_OF(SEISMIC_ELEMENT_A);
                streaming::copy_site_rows<A>(
                    reinterpret_cast<const typename A::storage *>(SEISMIC_RESULT_0_OUT.stored()),
                    SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1, SEISMIC_OUT_SITE_ROWS,
                    slot * SEISMIC_SITE_ROWS_STRIDE_0, SEISMIC_SITE_ROWS_STRIDE_1, SEISMIC_SITE_ROWS_STRIDE_2,
                    SEISMIC_DIM_M, SEISMIC_DIM_H);
                streaming_publish_barrier();
                if (threadIdx.x == 0) {
                    streaming_system_fence();
                    streaming_site_publish(q, bank.routes, SEISMIC_ROUTES_STRIDE_0, SEISMIC_ROUTES_STRIDE_1,
                        bank.weights, bank.weights0, bank.weights1, bank.sources, SEISMIC_SOURCES_STRIDE_0,
                        SEISMIC_SOURCES_STRIDE_1, SEISMIC_SOURCES_STRIDE_2, choices,
                        reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_SITE)) + slot * SEISMIC_SITE_STRIDE_0,
                        SEISMIC_SITE_STRIDE_1);
                }
            }
        }
    }
    __syncthreads();
    ROUTED_KEYS_RECORD;
    routed_write_keys_rows(keys_record, SEISMIC_KEYS_STRIDE_1, SEISMIC_KEYS_STRIDE_2, logits, LOGITS_STRIDE,
                           static_cast<unsigned>(SEISMIC_DIM_M), static_cast<unsigned>(SEISMIC_DIM_E));
}
