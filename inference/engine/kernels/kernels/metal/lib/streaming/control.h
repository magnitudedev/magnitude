// Expert streaming's device side of the host-shared words (routed.seismic),
// at the indices
// streaming_words.seismic declares (`SEISMIC_CONST_*`): Metal's shim under
// the protocol every backend shares (`kernels/streaming/`, its contract
// `streaming/contract.hpp`), and the waits in the kernel
// (`streaming/park.hpp`), which it includes.
//
// Words the host reads or writes while a step runs are accessed through
// `volatile coherent(system)` pointers (`SYSTEM`) under system-scope fences;
// the host's side is `Tensor::load_acquire` / `Tensor::store_release`.
// Shared storage alone does not order a host write before a device read, and
// plain device loads of host-rewritten bytes can return stale data within one
// command buffer.

// <seismic/slab.h> enables `coherent(system)` pointers.
#include <seismic/slab.h>

#define SYSTEM volatile coherent(system) device

// The shim's qualifiers and word handles (streaming/contract.hpp).
#define STREAMING_FN inline
#define STREAMING_WORDS_PARAM , constant ulong *seismic_words
#define STREAMING_WORDS_ARG , seismic_words
#define STREAMING_SYSTEM_WORDS SYSTEM int *
#define STREAMING_SYSTEM_READ SYSTEM const int *
#define STREAMING_DEVICE_WORDS device int *
#define STREAMING_DEVICE_READ device const int *
#define STREAMING_THREAD thread
#define STREAMING_AT(handle, index) ((handle) + (index))
#define STREAMING_GROUP_PARAM threadgroup int *streaming_group
#define STREAMING_GROUP(index) streaming_group[index]
#define STREAMING_GROUP_ADD(index, value) streaming_group_add(&streaming_group[index], value)
#define STREAMING_RECORDS_READ device const ulong *
#define STREAMING_RECORDS device const ulong *

typedef uint u32;
typedef ulong u64;

namespace streaming {

// Metal's system memory scope (AIR scope 3), stated as its value: a kernel's
// conditions read no toolchain-predefined macro, so coverage's skeleton
// preprocessing decides every group as the toolchain does.
constant metal::thread_scope system_scope = static_cast<metal::thread_scope>(3);

} // namespace streaming

inline void streaming_system_fence() {
    metal::atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst, streaming::system_scope);
}

// A host-shared word as the host last wrote it, and a store the host reads.
inline int streaming_load(SYSTEM const int *word) {
    return *word;
}

inline void streaming_store(SYSTEM int *word, int value) {
    *word = value;
}

// An acquire load of a host-shared word: the load, then a system-scope
// fence, so every later read follows it.
inline int streaming_load_acquire(SYSTEM const int *word) {
    const int value = *word;
    streaming_system_fence();
    return value;
}

// One 64-bit access of the u64 at an even word: an acquire load, and a
// store after a system-scope fence (release).
inline ulong streaming_load_u64_acquire(SYSTEM const int *word) {
    const ulong value = *reinterpret_cast<SYSTEM const ulong *>(word);
    streaming_system_fence();
    return value;
}

inline void streaming_store_u64_release(SYSTEM int *word, ulong value) {
    streaming_system_fence();
    *reinterpret_cast<SYSTEM ulong *>(word) = value;
}

inline int streaming_device_load(device const int *word) {
    return *word;
}

inline void streaming_device_store(device int *word, int value) {
    *word = value;
}

inline int streaming_device_add(device int *word, int value) {
    return atomic_fetch_add_explicit((device atomic_int *)word, value, memory_order_relaxed);
}

// MSL's compare-and-swap is weak only: it retries a spurious failure (the
// word read as `expected`), so it fails only where the word differs.
inline int streaming_device_cas(device int *word, int expected, int desired) {
    int seen = expected;
    while (!atomic_compare_exchange_weak_explicit((device atomic_int *)word, &seen, desired, memory_order_relaxed,
                                                  memory_order_relaxed)) {
        if (seen != expected)
            return seen;
    }
    return expected;
}

// The threadgroup's barrier over its threadgroup memory, and over its device
// memory too.
inline void streaming_barrier() {
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void streaming_device_barrier() {
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
}

// A barrier after which thread 0's system-scope stores follow every
// thread's earlier ones: a threadgroup barrier alone orders no thread's
// system-memory stores before another's, so each thread fences first.
inline void streaming_publish_barrier() {
    streaming_system_fence();
    streaming_device_barrier();
}

// A relaxed atomic add to a threadgroup word.
inline void streaming_group_add(threadgroup int *word, int value) {
    atomic_fetch_add_explicit((threadgroup atomic_int *)word, value, memory_order_relaxed);
}

// Copies `bytes` device bytes from `from` to `to`, each in its own space
// (<seismic/slab.h>), unit `u` by lane `u % lanes` of `lanes` (the caller's
// `lane`): 16-byte units when both addresses admit them, then the tail bytes.
template <typename From, typename To>
inline void streaming_copy_bytes(From from, To to, ulong bytes, ulong lane, ulong lanes) {
    const ulong vectors = (reinterpret_cast<ulong>(from) | reinterpret_cast<ulong>(to)) % 16 == 0 ? bytes / 16 : 0;
    for (ulong v = lane; v < vectors; v += lanes)
        slab::as<uint4>(to)[v] = slab::as<uint4>(from)[v];
    for (ulong b = vectors * 16 + lane; b < bytes; b += lanes)
        to[b] = from[b];
}

#include "../../../streaming/contract.hpp"
#include "../../../streaming/park.hpp"
