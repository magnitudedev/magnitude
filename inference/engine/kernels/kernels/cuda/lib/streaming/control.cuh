// Expert streaming's device side of the host-shared words (routed.seismic),
// at the indices
// streaming_words.seismic declares (`SEISMIC_CONST_*`); the CUDA form of
// `metal/lib/streaming/control.h`, the waits included: a waiting launch
// spins on the words its host answers, as on every backend, so one protocol
// and one step graph serve them all.
//
// Host-shared words (`Tensor::shared`: mapped pinned host memory) are read
// and written through volatile accesses, which PTX makes relaxed at system
// scope, under system-scope fences (`__threadfence_system`); the host's side
// is `Tensor::load_acquire` / `Tensor::store_release`. A publication's data
// precede its sequence by a fence, and a wait reads the awaited word by an
// acquire load (`ld.acquire.sys`), so its reads of what its host wrote
// follow it.
//
// It is CUDA's shim under the protocol every backend shares
// (`kernels/streaming/`, its contract `streaming/contract.hpp`), and includes
// the waits in the kernel (`streaming/park.hpp`).

#include <seismic/element.cuh>

// The shim's qualifiers and word handles (streaming/contract.hpp):
// host-shared words are plain pointers, accessed by `streaming_load` and
// `streaming_store`.
#define STREAMING_FN __device__ __forceinline__
#define STREAMING_WORDS_PARAM , const seismic_words_t &seismic_words_value
#define STREAMING_WORDS_ARG , seismic_words_value
#define STREAMING_SYSTEM_WORDS int *
#define STREAMING_SYSTEM_READ const int *
#define STREAMING_DEVICE_WORDS int *
#define STREAMING_DEVICE_READ const int *
#define STREAMING_THREAD
#define STREAMING_AT(handle, index) ((handle) + (index))
#define STREAMING_GROUP_PARAM int *streaming_group
#define STREAMING_GROUP(index) streaming_group[index]
#define STREAMING_GROUP_ADD(index, value) atomicAdd(&streaming_group[index], value)
#define STREAMING_RECORDS_READ const unsigned char *
#define STREAMING_RECORDS unsigned char *

using element::u32;
using element::u64;

// A host-shared word as the host last wrote it.
__device__ __forceinline__ int streaming_load(const int *word) {
    return *reinterpret_cast<const volatile int *>(word);
}

__device__ __forceinline__ void streaming_store(int *word, int value) {
    *reinterpret_cast<volatile int *>(word) = value;
}

// An acquire load of a host-shared word at system scope.
__device__ __forceinline__ int streaming_load_acquire(const int *word) {
    int value;
    asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(value) : "l"(word) : "memory");
    return value;
}

// One 64-bit access of the u64 at an even word, at system scope: an acquire
// load, and a release store.
__device__ __forceinline__ unsigned long long streaming_load_u64_acquire(const int *word) {
    unsigned long long value;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(value) : "l"(word) : "memory");
    return value;
}

__device__ __forceinline__ void streaming_store_u64_release(int *word, unsigned long long value) {
    asm volatile("st.release.sys.global.u64 [%0], %1;" : : "l"(word), "l"(value) : "memory");
}

__device__ __forceinline__ int streaming_device_load(const int *word) {
    return *word;
}

__device__ __forceinline__ void streaming_device_store(int *word, int value) {
    *word = value;
}

__device__ __forceinline__ void streaming_system_fence() {
    __threadfence_system();
}

// The block's barrier; `__syncthreads` orders shared and global memory
// alike, so the shim's three barriers are one.
__device__ __forceinline__ void streaming_barrier() {
    __syncthreads();
}

__device__ __forceinline__ void streaming_device_barrier() {
    __syncthreads();
}

// A barrier after which thread 0's system-scope stores follow every
// thread's earlier ones: PTX's `fence.sys` is cumulative, so the fence
// thread 0 issues after the barrier orders every store the barrier made
// visible to it, and no other thread fences.
__device__ __forceinline__ void streaming_publish_barrier() {
    __syncthreads();
}

namespace streaming {

// Copies `rows` rows of `bytes` bytes, `from_stride` bytes apart at `from`,
// to rows `to_stride` bytes apart at `to`: the rows' units in row-major
// order, unit `u` by thread `u % threads` of `threads` (the caller's
// `thread`), each thread with UNROLL units in flight (a load of mapped host
// memory is a PCIe round trip, so every row's loads are in flight
// together). Units are as wide as every address and stride admit (16 bytes for records and
// rows), a row's tail past its last whole unit then copied a byte a unit.
// Odd-sized packets (17-byte MXFP4) start rows at odd offsets.
template <class Unit>
__device__ __forceinline__ void copy_units(const unsigned char *from, unsigned long long from_stride, unsigned char *to,
                                           unsigned long long to_stride, unsigned long long rows,
                                           unsigned long long units, unsigned long long thread,
                                           unsigned long long threads) {
    typedef unsigned long long u64;
    constexpr unsigned UNROLL = 4;
    if (units == 0)
        return;
    // A thread's next unit is `threads` units on: `step_rows` rows and
    // `step_units` units, carried into the next row past the last.
    const u64 step_rows = threads / units, step_units = threads % units;
    u64 row = thread / units, unit = thread % units;
    while (row < rows) {
        // Each unit's byte offsets into `from` and into `to`.
        u64 in[UNROLL], out[UNROLL];
        bool live[UNROLL];
#pragma unroll
        for (unsigned u = 0; u < UNROLL; ++u) {
            live[u] = row < rows;
            in[u] = row * from_stride + unit * sizeof(Unit);
            out[u] = row * to_stride + unit * sizeof(Unit);
            row += step_rows;
            unit += step_units;
            if (unit >= units) {
                unit -= units;
                ++row;
            }
        }
        Unit held[UNROLL];
#pragma unroll
        for (unsigned u = 0; u < UNROLL; ++u)
            if (live[u])
                held[u] = __ldcs(reinterpret_cast<const Unit *>(from + in[u]));
#pragma unroll
        for (unsigned u = 0; u < UNROLL; ++u)
            if (live[u])
                *reinterpret_cast<Unit *>(to + out[u]) = held[u];
    }
}

template <class Unit>
__device__ __forceinline__ void copy_rows_as(const unsigned char *from, unsigned long long from_stride,
                                             unsigned char *to, unsigned long long to_stride, unsigned long long rows,
                                             unsigned long long bytes, unsigned long long thread,
                                             unsigned long long threads) {
    const unsigned long long whole = bytes / sizeof(Unit) * sizeof(Unit);
    copy_units<Unit>(from, from_stride, to, to_stride, rows, bytes / sizeof(Unit), thread, threads);
    if (whole != bytes)
        copy_units<unsigned char>(from + whole, from_stride, to + whole, to_stride, rows, bytes - whole, thread,
                                  threads);
}

__device__ __forceinline__ void copy_rows(const unsigned char *from, unsigned long long from_stride, unsigned char *to,
                                          unsigned long long to_stride, unsigned long long rows,
                                          unsigned long long bytes, unsigned long long thread,
                                          unsigned long long threads) {
    const unsigned long long alignment = reinterpret_cast<unsigned long long>(from) |
                                         reinterpret_cast<unsigned long long>(to) | from_stride | to_stride;
    if (alignment % 16 == 0)
        copy_rows_as<uint4>(from, from_stride, to, to_stride, rows, bytes, thread, threads);
    else if (alignment % 4 == 0)
        copy_rows_as<unsigned>(from, from_stride, to, to_stride, rows, bytes, thread, threads);
    else if (alignment % 2 == 0)
        copy_rows_as<unsigned short>(from, from_stride, to, to_stride, rows, bytes, thread, threads);
    else
        copy_rows_as<unsigned char>(from, from_stride, to, to_stride, rows, bytes, thread, threads);
}

} // namespace streaming

// One row of `bytes` bytes (streaming/contract.hpp).
__device__ __forceinline__ void streaming_copy_bytes(const unsigned char *from, unsigned char *to,
                                                     unsigned long long bytes, unsigned long long lane,
                                                     unsigned long long lanes) {
    streaming::copy_rows(from, 0, to, 0, 1, bytes, lane, lanes);
}

#include "../../../streaming/contract.hpp"
#include "../../../streaming/park.hpp"
