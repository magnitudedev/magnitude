// The CPU's answer in a streamed decode combine (`routed_down_sited`,
// `routed_output_sited`), the sums' answer hooks (`lib/routed/down_sum.cuh`,
// `lib/routed/output_sum.cuh`): thread 0 spins on the step's slot of the
// layer's site table when a choice of the step is the CPU's
// (`streaming_site_any`, `streaming_await_slot`) and shares the slot in
// shared memory; then every thread, past the block's barrier and a system
// fence, adds row m's answer (`site_output[slot, m, n]`) to its sum where
// the CPU answered the step. In the kernel's own scope, where its ports are.
#include "../streaming/control.cuh"
#include "../../../streaming/site.hpp"

// The step (its pair).
#define ROUTED_SITE_STEP                                                                                   \
    streaming_step(reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_STEP)), SEISMIC_STEP_STRIDE_0)

#define ROUTED_SITE_AWAIT(m)                                                                               \
    __shared__ int site_answered;                                                                          \
    __shared__ u64 site_slot;                                                                              \
    if (threadIdx.x == 0) {                                                                                \
        const int *sources = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_SOURCES));           \
        if (streaming_site_any(sources, SEISMIC_SOURCES_STRIDE_0, SEISMIC_SOURCES_STRIDE_1,                \
                SEISMIC_SOURCES_STRIDE_2, (unsigned)(SEISMIC_DIM_M * SEISMIC_DIM_K))) {                    \
            const u64 site_step = ROUTED_SITE_STEP;                                                        \
            const u64 slot = site_step % (u64)SEISMIC_DIM_FLIGHT;                                          \
            site_slot = slot;                                                                              \
            site_answered = streaming_await_slot(                                                          \
                reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_SITE)) + slot * SEISMIC_SITE_STRIDE_0,  \
                SEISMIC_SITE_STRIDE_1, site_step);                                                         \
        } else {                                                                                           \
            site_slot = 0;                                                                                 \
            site_answered = 0;                                                                             \
        }                                                                                                  \
    }                                                                                                      \
    __syncthreads();                                                                                       \
    const bool answered = site_answered != 0;                                                              \
    const u64 answered_slot = site_slot;                                                                   \
    if (answered)                                                                                          \
        streaming_system_fence();

// Row m's answer at channel n added to `selected` where the CPU answered.
#define ROUTED_SITE_ADD(m, n, selected)                                                                    \
    if (answered)                                                                                          \
        (selected) += *reinterpret_cast<const volatile float *>(                                           \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SITE_OUTPUT)) +                     \
            answered_slot * SEISMIC_SITE_OUTPUT_STRIDE_0 + (m) * SEISMIC_SITE_OUTPUT_STRIDE_1 +            \
            (n) * SEISMIC_SITE_OUTPUT_STRIDE_2);
