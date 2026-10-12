// Arrival counting across the threadgroups of one launch, over `sync`
// scratch (zero whenever a launch starts; see the `sync` scratch contract).
// A threadgroup that has finished its device stores arrives; exactly one
// threadgroup, the last to arrive, is told so, and it restores the counter
// to zero, so the next launch over the same scratch starts from zero. No
// threadgroup ever waits for another, so the threadgroups need not be
// co-resident.
//
// The last threadgroup sees the other threadgroups' stores only through
// memory declared `coherent_device`: the fences order the counter, but a
// plain device load may be served from the reading core's non-coherent cache
// and miss another core's stores. M1 Max-class GPUs (g13s, g13d) do miss
// them, deterministically.

namespace arrive {

// A buffer that a launch's threadgroups store to and its last-arriving
// threadgroup then loads from.
template <typename T>
using coherent_device = coherent(device) device T *;

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
