// Grid barriers of a resident launch (`launch K resident (BARRIER)`), in
// the prefix of every Metal source with one: every threadgroup of its grid
// waits at `seismic_grid::sync` until all have arrived, so one launch runs
// phases that read what the whole grid wrote before. The runtime sizes the
// grid (`RESIDENT`, at most the launch's `threadgroups`) so that every
// threadgroup is on the device at once, and orders no other launch of its
// queue beside it; nothing on Metal guarantees co-residency, so every wait
// is bounded. A threadgroup that waits past the bound gives up: it sets the
// barrier's give-up word, and every later `sync` over the barrier returns
// false at once. The word stays set; the runtime reads it once the work
// completes and loses the device (`NativeSubmission::wait`). The kernel
// still runs `depart` in every threadgroup, whether or not a wait gave up.
//
// The barrier is RESIDENT_BARRIER_BYTES of `sync` scratch, zero whenever
// a launch starts, as `seismic_lang::registry` lays it out: arrivals,
// departures, the give-up word (SEISMIC_RESIDENT_GIVE_UP_WORD) and a pad.
// The last threadgroup to depart restores arrivals and departures to zero.
//
// Stores before a `sync` are seen after it only through memory declared
// `coherent(device)`: a plain device load may be served from the reading
// core's non-coherent cache.

namespace seismic_grid {

// Spins a wait allows before it gives up. On an M4 Pro 2^20 spins, reading
// the give-up word every 256, took 399 ms to detect a give-up (~380 ns a
// spin under contention), well inside a call's 2 s; a co-resident grid's
// wait is ~0.3 us.
constant uint SPINS = 1u << 20;
// Spins between reads of the give-up word, so the wait's loop is the
// arrivals' load alone.
constant uint GIVE_UP_STRIDE = 256u;

constant uint ARRIVALS = 0u;
constant uint DEPARTURES = 1u;
constant uint GIVE_UP = SEISMIC_RESIDENT_GIVE_UP_WORD;

// One threadgroup's view of the barrier: its words, the grid's threadgroups
// and the barriers it has passed. Formed by `SEISMIC_GRID_BARRIER`.
struct Barrier {
    device atomic_uint *words;
    uint groups;
    uint passed;
};

// Wait until every threadgroup has arrived here. Every thread of the
// threadgroup calls it, after its device stores; `flag` is one threadgroup
// word. Returns false, to every thread, once a wait over the barrier has
// given up.
inline bool sync(thread Barrier &barrier, threadgroup uint *flag, uint thread_index) {
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    barrier.passed += 1u;
    if (thread_index == 0) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        atomic_fetch_add_explicit(&barrier.words[ARRIVALS], 1u, memory_order_relaxed);
        const uint target = barrier.groups * barrier.passed;
        uint held = atomic_load_explicit(&barrier.words[GIVE_UP], memory_order_relaxed) == 0u ? 1u : 0u;
        for (uint spin = 1; held != 0u && atomic_load_explicit(&barrier.words[ARRIVALS], memory_order_relaxed) < target;
             ++spin) {
            if (spin % GIVE_UP_STRIDE != 0u)
                continue;
            if (spin >= SPINS || atomic_load_explicit(&barrier.words[GIVE_UP], memory_order_relaxed) != 0u) {
                atomic_store_explicit(&barrier.words[GIVE_UP], 1u, memory_order_relaxed);
                held = 0u;
            }
        }
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        *flag = held;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    return *flag != 0u;
}

// Leave the launch's barriers: every threadgroup calls it once, last; the
// last to depart restores arrivals and departures for the next launch.
inline void depart(thread const Barrier &barrier, uint thread_index) {
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (thread_index == 0) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        const uint ticket = atomic_fetch_add_explicit(&barrier.words[DEPARTURES], 1u, memory_order_relaxed);
        if (ticket == barrier.groups - 1u) {
            atomic_store_explicit(&barrier.words[ARRIVALS], 0u, memory_order_relaxed);
            atomic_store_explicit(&barrier.words[DEPARTURES], 0u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
}

} // namespace seismic_grid

// Resident kernel KERNEL's barrier over `words` (its BARRIER scratch), of
// `groups` threadgroups (`threadgroups_per_grid`); KERNEL is the kernel's
// macro name (upper case, as SEISMIC_FORMING_<KERNEL>).
// SEISMIC_RESIDENT_<KERNEL> is defined for resident kernels alone, so no
// other kernel forms one.
#define SEISMIC_GRID_BARRIER(KERNEL, words, groups) \
    (seismic_grid::Barrier{(words), (groups) * SEISMIC_RESIDENT_##KERNEL, 0u})
