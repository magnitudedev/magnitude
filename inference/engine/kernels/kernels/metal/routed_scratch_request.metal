// Expert streaming, a scratch-served step's request (`routed_scratch_request`
// in routed.seismic): each expert's flag, the number of choices that route to
// it (0 when none), then the layer and the step's sequence in the load's
// scratch request region, under system-scope fences. It does not wait.

#include "lib/streaming/control.h"

kernel void routed_scratch_request(
    device const int *routes [[buffer(SEISMIC_BUFFER_ROUTES)]],
    device const int *step [[buffer(SEISMIC_BUFFER_STEP)]],
    SYSTEM int *flags [[buffer(SEISMIC_BUFFER_FLAGS)]],
    SYSTEM int *control [[buffer(SEISMIC_BUFFER_CONTROL)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint tid [[thread_index_in_threadgroup]]) {
    // E <= 1024 (routed.seismic): one word per expert.
    threadgroup atomic_int counts[1024];
    const uint experts = uint(SEISMIC_DIM_E);
    for (uint e = tid; e < experts; e += 1024u)
        atomic_store_explicit(&counts[e], 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const ulong choices = SEISMIC_DIM_M * SEISMIC_DIM_K;
    for (ulong c = tid; c < choices; c += 1024u) {
        const ulong m = c / SEISMIC_DIM_K, k = c % SEISMIC_DIM_K;
        const int expert = routes[m * SEISMIC_ROUTES_STRIDE_0 + k * SEISMIC_ROUTES_STRIDE_1];
        atomic_fetch_add_explicit(&counts[uint(expert)], 1, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < experts; e += 1024u)
        flags[e * SEISMIC_FLAGS_STRIDE_0] = atomic_load_explicit(&counts[e], memory_order_relaxed);
    streaming::system_fence();
    threadgroup_barrier(mem_flags::mem_device);
    if (tid != 0) return;
    streaming::system_fence();
    control[streaming::SCRATCH_LAYER] = int(SEISMIC_PARAM_LAYER);
    streaming::system_fence();
    control[streaming::REQUEST] = step[0];
    streaming::system_fence();
}
