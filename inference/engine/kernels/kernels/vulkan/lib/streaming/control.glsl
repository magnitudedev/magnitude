// Expert streaming's device side of the host-shared words (routed.seismic),
// at the indices streaming_words.seismic declares (`SEISMIC_CONST_*`): the
// Vulkan shim under the protocol every backend shares (`kernels/streaming/`,
// its contract `streaming/contract.hpp`).
//
// A wait does not spin (`SEISMIC_PARKS_IN_KERNELS` 0, `streaming/park.hpp`):
// the Vulkan spec makes no host write visible to running work, so the call that
// waits follows a park, a submission boundary the host opens once the answer
// is written, and reads the awaited words once. Host-shared words are host-coherent memory, read and written
// through coherent, volatile references, and an awaited word is read by a
// device-scope acquire load (the Vulkan memory model's `atomicLoad`), so
// every read after it sees what the host wrote before the park's answer.
// Words are addressed by their device addresses.
//
// A call's group words are its group memory as `seismic_shared_i32` (the
// prefix's view of the native's `shared_bytes`): a protocol function takes
// the index of its first word.

#include <seismic/element.glsl>

layout(buffer_reference, scalar, buffer_reference_align = 4) coherent volatile buffer streaming_word_t { int v; };
layout(buffer_reference, scalar, buffer_reference_align = 8) coherent volatile buffer streaming_pair_t { uint64_t v; };

#define STREAMING_FN
#define STREAMING_WORDS_PARAM
#define STREAMING_WORDS_ARG
#define STREAMING_SYSTEM_WORDS uint64_t
#define STREAMING_SYSTEM_READ uint64_t
#define STREAMING_DEVICE_WORDS uint64_t
#define STREAMING_DEVICE_READ uint64_t
#define STREAMING_ATOMIC_WORDS uint64_t
#define STREAMING_THREAD
#define STREAMING_AT(handle, index) ((handle) + uint64_t(index) * 4ul)
#define STREAMING_GROUP_PARAM uint streaming_group
#define STREAMING_GROUP(index) seismic_shared_i32[streaming_group + uint(index)]
#define STREAMING_GROUP_ADD(index, value) atomicAdd(STREAMING_GROUP(index), value)
#define STREAMING_RECORDS uint64_t
#define u32 uint
#define u64 uint64_t

// `value` % `divisor` of a step counter by the flight depth, and `value` /
// `divisor` of a byte count by a part size whose quotient is below 2^32.
// The flight depth is a small count (below 2^16), as seismic_divide takes.
uint64_t streaming_modulo(uint64_t value, uint64_t divisor) {
    uint64_t quotient;
    uint remainder;
    seismic_divide(value, uint(divisor), quotient, remainder);
    return uint64_t(remainder);
}
// The quotient is proved below 2^32, as seismic_narrow_divide takes.
uint streaming_narrow_quotient(uint64_t value, uint64_t divisor) {
    uint64_t rest = value;
    return seismic_narrow_divide(rest, divisor);
}

int streaming_load(uint64_t word) {
    return streaming_word_t(word).v;
}

void streaming_store(uint64_t word, int value) {
    streaming_word_t(word).v = value;
}

void streaming_store_relaxed(uint64_t word, int value) {
    streaming_word_t(word).v = value;
}

// An acquire load of a host-shared word at device scope.
int streaming_load_acquire(uint64_t word) {
    return atomicLoad(streaming_word_t(word).v, gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsAcquire);
}

// One 64-bit access of the u64 at an even word, at device scope: a relaxed
// load, an acquire load, and a release store.
uint64_t streaming_load_u64(uint64_t word) {
    return streaming_pair_t(word).v;
}

uint64_t streaming_load_u64_acquire(uint64_t word) {
    return atomicLoad(streaming_pair_t(word).v, gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsAcquire);
}

void streaming_store_u64_release(uint64_t word, uint64_t value) {
    atomicStore(streaming_pair_t(word).v, value, gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsRelease);
}

void streaming_acquire_fence() {
    memoryBarrierBuffer();
}

int streaming_device_load(uint64_t word) {
    return streaming_word_t(word).v;
}

void streaming_device_store(uint64_t word, int value) {
    streaming_word_t(word).v = value;
}

int streaming_atomic_load(uint64_t word) {
    return atomicLoad(streaming_word_t(word).v, gl_ScopeDevice, gl_StorageSemanticsNone, gl_SemanticsRelaxed);
}

void streaming_atomic_store(uint64_t word, int value) {
    atomicStore(streaming_word_t(word).v, value, gl_ScopeDevice, gl_StorageSemanticsNone, gl_SemanticsRelaxed);
}

int streaming_atomic_add(uint64_t word, int value) {
    return atomicAdd(streaming_word_t(word).v, value);
}

int streaming_atomic_max(uint64_t word, int value) {
    return atomicMax(streaming_word_t(word).v, value);
}

int streaming_atomic_exchange(uint64_t word, int value) {
    return atomicExchange(streaming_word_t(word).v, value);
}

void streaming_device_fence() {
    memoryBarrierBuffer();
}

// Orders this invocation's accesses of host-shared words.
void streaming_system_fence() {
    memoryBarrierBuffer();
}

void streaming_barrier() {
    memoryBarrierShared();
    barrier();
}

void streaming_device_barrier() {
    memoryBarrierShared();
    memoryBarrierBuffer();
    barrier();
}

void streaming_publish_barrier() {
    streaming_device_barrier();
}

// Copies `bytes` device bytes from `from` to `to`, unit `u` by lane
// `u % lanes` of `lanes` (the caller's `lane`): 16-byte units when every
// address admits them, then the tail bytes. Its counters are 32-bit
// (streaming/contract.hpp).
void streaming_copy_bytes(uint64_t from, uint64_t to, uint64_t bytes, uint64_t lane, uint64_t lanes) {
    const uint vectors = ((from | to) & 15ul) == 0ul ? uint(bytes >> 4ul) : 0u;
    for (uint v = uint(lane); v < vectors; v += uint(lanes))
        seismic_uvec4(to + uint64_t(v) * 16ul).v = seismic_uvec4(from + uint64_t(v) * 16ul).v;
    for (uint b = vectors * 16u + uint(lane); b < uint(bytes); b += uint(lanes))
        seismic_u8(to + uint64_t(b)).v = seismic_u8(from + uint64_t(b)).v;
}

#include "../../../streaming/contract.hpp"
#include "../../../streaming/park.hpp"
