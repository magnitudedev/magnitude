// Arrival counting across the blocks of one launch, over `sync` scratch (zero
// whenever a launch starts; see the `sync` scratch contract); the CUDA form
// of `metal/lib/core/arrive.h`. A block that has finished its device stores
// arrives; exactly one block, the last to arrive, is told so, and it restores
// the counter to zero, so the next launch over the same scratch starts from
// zero. No block ever waits for another, so the blocks need not be
// co-resident.
//
// The barrier makes every thread's stores the arriving thread's, and its
// device-scope fence, cumulative, orders them before the arrival; the last
// block's fence orders the arrivals before its reads.

namespace arrive {

// Whether this block is the last of `expected` to arrive at `counter`. Every
// thread of the block must call it, after its device stores; `flag` is one
// shared word. Every thread receives the result.
__device__ __forceinline__ bool last(unsigned *counter, unsigned expected, unsigned *flag) {
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        const bool is_last = atomicAdd(counter, 1u) == expected - 1u;
        if (is_last) {
            __threadfence();
            atomicExch(counter, 0u);
        }
        *flag = is_last ? 1u : 0u;
    }
    __syncthreads();
    return *flag != 0u;
}

} // namespace arrive
