// The CPU's answer in a streamed decode combine (`routed_down_sited`,
// `routed_output_sited`), the sums' answer hooks (`lib/routed/down_sum.cuh`,
// `lib/routed/output_sum.cuh`): thread 0 spins on the step's slot of the
// layer's site table when a choice of the step is the CPU's
// (`streaming_site_any`, `streaming_await_slot`); then every thread, past
// the block's barrier and a system fence, adds row m's answer
// (`site_output[slot, m, n]`) to its sum where the CPU answered the step.
// In the kernel's own scope, where its ports are.
#include "../streaming/control.cuh"
#include "../../../streaming/site.hpp"

// The step (its pair) and its slot.
#define ROUTED_SITE_STEP                                                                                   \
    streaming_step(reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_STEP)), SEISMIC_STEP_STRIDE_0)
#define ROUTED_SITE_SLOT (ROUTED_SITE_STEP % (u64)SEISMIC_DIM_FLIGHT)

#define ROUTED_SITE_AWAIT(m)                                                                               \
    __shared__ int site_answered;                                                                          \
    if (threadIdx.x == 0) {                                                                                \
        const int *sources = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_SOURCES));           \
        const u64 site_step = ROUTED_SITE_STEP;                                                            \
        site_answered = streaming_site_any(sources, SEISMIC_SOURCES_STRIDE_0, SEISMIC_SOURCES_STRIDE_1,    \
                            SEISMIC_SOURCES_STRIDE_2, (unsigned)(SEISMIC_DIM_M * SEISMIC_DIM_K)) &&        \
            streaming_await_slot(reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_SITE)) +         \
                                     (site_step % (u64)SEISMIC_DIM_FLIGHT) * SEISMIC_SITE_STRIDE_0,        \
                SEISMIC_SITE_STRIDE_1, site_step);                                                         \
    }                                                                                                      \
    __syncthreads();                                                                                       \
    const bool answered = site_answered != 0;                                                              \
    if (answered)                                                                                          \
        streaming_system_fence();

// Row m's answer at channel n added to `selected` where the CPU answered.
#define ROUTED_SITE_ADD(m, n, selected)                                                                    \
    if (answered)                                                                                          \
        (selected) += *reinterpret_cast<const volatile float *>(                                           \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SITE_OUTPUT)) +                     \
            ROUTED_SITE_SLOT * SEISMIC_SITE_OUTPUT_STRIDE_0 + (m) * SEISMIC_SITE_OUTPUT_STRIDE_1 +         \
            (n) * SEISMIC_SITE_OUTPUT_STRIDE_2);
