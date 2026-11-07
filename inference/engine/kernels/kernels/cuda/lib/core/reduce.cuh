// Fixed-order reductions over a warp or a block. A block reduction first
// reduces each warp (the prelude's fixed xor-butterfly `seismic_warp_*`),
// then combines the warp partials in index order, so its result depends only
// on the block's width, never on scheduling. Every thread receives the
// result.

namespace reduce {

// Sum of one value per thread of the block. `partials` holds one float per
// warp (at most 32); the leading barrier lets consecutive calls share it.
// Every thread of the block must call it.
__device__ __forceinline__ float group_sum(float value, float *partials) {
    value = seismic_warp_sum_f32(value);
    const unsigned warps = blockDim.x / 32;
    __syncthreads();
    if (threadIdx.x % 32 == 0)
        partials[threadIdx.x / 32] = value;
    __syncthreads();
    float total = 0.0f;
    for (unsigned w = 0; w < warps; ++w)
        total += partials[w];
    return total;
}

// Maximum of one value per thread, as `group_sum`.
__device__ __forceinline__ float group_max(float value, float *partials) {
    value = seismic_warp_max_f32(value);
    const unsigned warps = blockDim.x / 32;
    __syncthreads();
    if (threadIdx.x % 32 == 0)
        partials[threadIdx.x / 32] = value;
    __syncthreads();
    float maximum = partials[0];
    for (unsigned w = 1; w < warps; ++w)
        maximum = fmaxf(maximum, partials[w]);
    return maximum;
}

// Tie rules of an argmax: among candidates of equal score, the higher or the
// lower index wins.
struct HigherIndex {
    __device__ static __forceinline__ int pick(int candidate) { return seismic_redux_max_s32(candidate); }
    static constexpr int none = -1;
};
struct LowerIndex {
    __device__ static __forceinline__ int pick(int candidate) { return seismic_redux_min_s32(candidate); }
    static constexpr int none = 0x7fffffff;
};

// The warp's maximum score and the index TIES picks among the lanes holding
// it. Every lane receives both; all 32 lanes must be convergent.
template <class TIES> __device__ __forceinline__ int argmax(float score, int index, float &best) {
    best = seismic_warp_max_f32(score);
    return TIES::pick(score == best ? index : TIES::none);
}

// Integer prefix sums over a warp or a block, exact in any order, and the
// warp-cooperative walks over a table of span lengths built on them.

// The inclusive prefix sum of `value` over lanes 0..lane. All 32 lanes must
// be convergent.
template <class T> __device__ __forceinline__ T lanes_inclusive_add(T value, int lane) {
#pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        const T below = __shfl_up_sync(0xffffffffu, value, offset);
        if (lane >= offset)
            value += below;
    }
    return value;
}

// The sum of `value` over the block's threads before this one, and in `total`
// the sum over all of them. `partials` holds one word per warp (at most 32);
// the leading barrier lets consecutive calls share it. Every thread of the
// block must call it.
__device__ __forceinline__ unsigned group_exclusive_add(unsigned value, unsigned *partials, unsigned &total) {
    const int lane = static_cast<int>(threadIdx.x % 32);
    const unsigned warp = threadIdx.x / 32;
    const unsigned warps = blockDim.x / 32;
    const unsigned inclusive = lanes_inclusive_add(value, lane);
    __syncthreads();
    if (lane == 31)
        partials[warp] = inclusive;
    __syncthreads();
    unsigned before = 0, sum = 0;
    for (unsigned w = 0; w < warps; ++w) {
        const unsigned partial = partials[w];
        before += w < warp ? partial : 0u;
        sum += partial;
    }
    total = sum;
    return before + inclusive - value;
}

// Walks over spans 0..count-1 whose lengths are `length(s)` (>= 0, in any
// unit: rows, keys, key tiles). Every lane of the warp calls them with the
// same arguments, convergent, and receives the same result.

// The sum of the lengths: lane l sums spans l, l + 32, ..., then the warp sum.
template <class Length>
__device__ __forceinline__ long long span_total(unsigned long long count, const Length &length, int lane) {
    long long total = 0;
    for (unsigned long long s = static_cast<unsigned long long>(lane); s < count; s += 32)
        total += static_cast<long long>(length(s));
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        total += __shfl_xor_sync(0xffffffffu, total, offset);
    return total;
}

// A span and its offset: the lengths of the spans before it.
struct Located {
    unsigned long long span;
    long long offset;
};

// The first span s with offset(s) + length(s) > `at`, and offset(s); (count,
// total) when there is none. Spans are taken 32 at a time: an inclusive warp
// prefix of their lengths plus the offset carried from earlier chunks, a
// ballot of the spans ending past `at`, the first of them broadcast. A walk
// over the units from `at` on starts there: the spans before it hold none of
// them.
template <class Length>
__device__ __forceinline__ Located span_locate(unsigned long long count, long long at, const Length &length,
                                               int lane) {
    long long carried = 0;
    for (unsigned long long base = 0; base < count; base += 32) {
        const unsigned long long s = base + static_cast<unsigned long long>(lane);
        const long long own = s < count ? static_cast<long long>(length(s)) : 0;
        const long long inclusive = lanes_inclusive_add(own, lane);
        const long long end = carried + inclusive;
        const unsigned hits = __ballot_sync(0xffffffffu, s < count && end > at);
        if (hits != 0u) {
            const int first = __ffs(hits) - 1;
            return Located{base + static_cast<unsigned long long>(first),
                           __shfl_sync(0xffffffffu, end - own, first)};
        }
        carried += __shfl_sync(0xffffffffu, inclusive, 31);
    }
    return Located{count, carried};
}

} // namespace reduce
