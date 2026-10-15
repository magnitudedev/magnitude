// Expert streaming's device side of the host handshake
// (specs/26-10-11/expert-streaming.md 4.3).
//
// Words the host reads or writes while a step runs are accessed through
// `volatile coherent(system)` pointers under system-scope fences. Shared
// storage alone does not order a host write before a device read, and plain
// device loads of host-rewritten bytes can return stale data within one
// command buffer (expert-streaming-theory.md §8).

#pragma METAL internals : enable
#ifndef __METAL_MEMORY_SCOPE_SYSTEM__
#define __METAL_MEMORY_SCOPE_SYSTEM__ 3
#endif
namespace metal {
constexpr constant metal::thread_scope thread_scope_system =
    static_cast<thread_scope>(__METAL_MEMORY_SCOPE_SYSTEM__);
}

#define SYSTEM volatile coherent(system) device

namespace streaming {

inline void system_fence() {
    metal::atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst,
        metal::thread_scope_system);
}

// Words of a layer's request region (routed.seismic).
constant uint REQUEST = 0, ACKNOWLEDGED = 1, FAILURE = 2, COUNT = 3, EXPERTS = 4;

// Words of the load's scratch request region (routed.seismic): the request
// and failure words as above.
constant uint SCRATCH_LAYER = 1;

// The host fails a park it cannot serve within one second. A host that died
// (a killed process) answers nothing, so a park also watches the host's
// heartbeat word, which the host advances while it serves submitted steps: a
// heartbeat that stays still for PATIENCE spins fails the park. A live host is
// never abandoned however long its reads take, and a dead one never holds
// the device.
constant uint PATIENCE = 1u << 22;

// Waits until `word` holds `sequence` or the failure word does; on a stalled
// heartbeat writes the failure itself. True when the word arrived and the
// step has not failed.
inline bool await_word(SYSTEM const int *word, SYSTEM int *failure, SYSTEM const int *heartbeat,
                       int sequence) {
    int beat = *heartbeat;
    for (uint still = 0; still < PATIENCE; ++still) {
        system_fence();
        if (*word == sequence || *failure == sequence) break;
        const int now = *heartbeat;
        if (now != beat) {
            beat = now;
            still = 0;
        }
    }
    system_fence();
    if (*word != sequence && *failure != sequence) {
        *failure = sequence;
        system_fence();
    }
    return *word == sequence && *failure != sequence;
}

// Waits until the host acknowledges `sequence` in a layer's request region
// (or fails it); true when acknowledged without failure.
inline bool await_acknowledgement(SYSTEM int *control, SYSTEM const int *heartbeat, int sequence) {
    return await_word(control + ACKNOWLEDGED, control + FAILURE, heartbeat, sequence);
}

} // namespace streaming
