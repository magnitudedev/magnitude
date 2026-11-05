// `parked_scale` (route.seismic): the kernel parks on the host's answer word,
// read through a `volatile coherent(system)` pointer under system-scope
// fences, then scales `x` by it.

#pragma METAL internals : enable
#ifndef __METAL_MEMORY_SCOPE_SYSTEM__
#define __METAL_MEMORY_SCOPE_SYSTEM__ 3
#endif

constant metal::thread_scope system_scope =
    static_cast<metal::thread_scope>(__METAL_MEMORY_SCOPE_SYSTEM__);

// Spins before the kernel stops waiting: a bound, so an unanswered park ends
// the launch (and the test fails on its result) rather than holding the
// device.
constant uint PATIENCE = 1u << 24;

kernel void parked_scale(
    volatile coherent(system) device const int *answer [[buffer(SEISMIC_BUFFER_ANSWER)]],
    device const float *x [[buffer(SEISMIC_BUFFER_X)]],
    device float *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]) {
    int seen = 0;
    for (uint spin = 0; spin < PATIENCE && seen == 0; ++spin) {
        metal::atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst, system_scope);
        seen = answer[0];
    }
    for (ulong index = 0; index < SEISMIC_DIM_N; ++index)
        result[index * SEISMIC_RESULT_0_STRIDE_0] = x[index * SEISMIC_X_STRIDE_0] * float(seen);
}
