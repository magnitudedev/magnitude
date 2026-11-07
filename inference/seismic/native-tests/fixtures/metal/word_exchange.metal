// The device's side of a host-shared word exchange (route.seismic
// `word_exchange`). Words the host writes while the kernel runs are read
// through a `volatile coherent(system)` pointer under system-scope fences:
// shared storage alone does not order a host write before a device read.

#pragma METAL internals : enable

// Metal's system memory scope (AIR scope 3).
constant metal::thread_scope system_scope =
    static_cast<metal::thread_scope>(3);

inline void system_fence() {
    metal::atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst,
                               system_scope);
}

// Spins before the kernel gives up on the host: a bound, so an unanswered
// exchange ends the launch (and the test fails on word 2) rather than holding
// the device.
constant uint PATIENCE = 1u << 24;

kernel void word_exchange(
    volatile coherent(system) device int *exchange [[buffer(SEISMIC_BUFFER_EXCHANGE)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]) {
    const ulong stride = SEISMIC_EXCHANGE_STRIDE_0;
    exchange[3 * stride] = 42;
    system_fence();
    exchange[0] = 1;
    system_fence();
    int seen = 0;
    for (uint spin = 0; spin < PATIENCE && seen == 0; ++spin) {
        system_fence();
        seen = exchange[stride];
    }
    exchange[2 * stride] = seen;
}
