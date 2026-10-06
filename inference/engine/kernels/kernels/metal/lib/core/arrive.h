// Arrival counting across the threadgroups of one launch, over `sync`
// scratch (zero whenever a launch starts; see the `sync` scratch contract).
// A threadgroup that has finished its device stores arrives; exactly one
// threadgroup, the last to arrive, is told so. It sees every other arrived
// threadgroup's device stores, and it restores the counter to zero, so the
// next launch over the same scratch starts from zero. No threadgroup ever
// waits for another, so the threadgroups need not be co-resident.

namespace arrive {

// Whether this threadgroup is the last of `expected` to arrive at `counter`.
// Every thread of the threadgroup must call it, after its device stores;
// `flag` is one threadgroup word. Every thread receives the result.
inline bool last(device atomic_uint *counter, uint expected, threadgroup uint *flag, uint thread_index) {
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (thread_index == 0) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        const uint ticket = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
        const bool is_last = ticket == expected - 1u;
        if (is_last) {
            atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
            atomic_store_explicit(counter, 0u, memory_order_relaxed);
        }
        *flag = is_last ? 1u : 0u;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    return *flag != 0u;
}

} // namespace arrive
