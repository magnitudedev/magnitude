// Fixed-order reductions over a warp or a block. A block reduction first
// reduces each warp (the prelude's fixed xor-butterfly `seismic_warp_*`),
// then combines the warp partials in index order, so its result depends only
// on the block's width, never on scheduling. Every thread receives the
// result.
#pragma once

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

// Sum of N independent values per thread of the block in a single barrier pair.
// `partials` holds N * warps floats (at most 32 * N); the leading barrier lets
// consecutive calls share it. Every thread of the block must call it.
template <unsigned N>
__device__ __forceinline__ void group_sum_n(float (&values)[N], float *partials) {
#pragma unroll
    for (unsigned k = 0; k < N; ++k)
        values[k] = seismic_warp_sum_f32(values[k]);
    const unsigned warps = blockDim.x / 32;
    const unsigned warp = threadIdx.x / 32;
    __syncthreads();
    if (threadIdx.x % 32 == 0) {
#pragma unroll
        for (unsigned k = 0; k < N; ++k)
            partials[k * warps + warp] = values[k];
    }
    __syncthreads();
#pragma unroll
    for (unsigned k = 0; k < N; ++k) {
        float total = 0.0f;
        for (unsigned w = 0; w < warps; ++w)
            total += partials[k * warps + w];
        values[k] = total;
    }
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

// Single-pass online softmax state: running maximum and sum of exp(x - max).
// Empty state is {-inf, 0}; -inf inputs never produce NaN, while NaN inputs
// propagate through `sum`.
struct OnlineSoftmax {
    float max;
    float sum;

    static __device__ __forceinline__ OnlineSoftmax empty() {
        return OnlineSoftmax{-__int_as_float(0x7f800000), 0.0f};
    }

    __device__ __forceinline__ void add(float x) {
        if (x > max) {
            sum = (sum != 0.0f ? sum * expf(max - x) : 0.0f) + 1.0f;
            max = x;
        } else if (x != -__int_as_float(0x7f800000)) {
            sum += expf(x - max);
        }
    }

    static __device__ __forceinline__ OnlineSoftmax merge(OnlineSoftmax a, OnlineSoftmax b) {
        if (a.sum == 0.0f)
            return b;
        if (b.sum == 0.0f)
            return a;
        const float m = fmaxf(a.max, b.max);
        return OnlineSoftmax{m, a.sum * expf(a.max - m) + b.sum * expf(b.max - m)};
    }
};

// Online softmax reduction over a warp. All 32 lanes must be convergent; every
// lane receives the combined state.
__device__ __forceinline__ OnlineSoftmax warp_online_softmax(OnlineSoftmax value) {
    const float m = seismic_warp_max_f32(value.max);
    const float scaled = value.sum != 0.0f ? value.sum * expf(value.max - m) : 0.0f;
    return OnlineSoftmax{m, seismic_warp_sum_f32(scaled)};
}

// Online softmax reduction over a block, in a single barrier pair. `partials`
// holds one OnlineSoftmax per warp (at most 32); the leading barrier lets
// consecutive calls share it. Every thread of the block must call it.
__device__ __forceinline__ OnlineSoftmax group_online_softmax(OnlineSoftmax value, OnlineSoftmax *partials) {
    value = warp_online_softmax(value);
    const unsigned warps = blockDim.x / 32;
    __syncthreads();
    if (threadIdx.x % 32 == 0)
        partials[threadIdx.x / 32] = value;
    __syncthreads();
    float maximum = partials[0].max;
    for (unsigned w = 1; w < warps; ++w)
        maximum = fmaxf(maximum, partials[w].max);
    float total = 0.0f;
    for (unsigned w = 0; w < warps; ++w) {
        const OnlineSoftmax p = partials[w];
        if (p.sum != 0.0f)
            total += p.sum * expf(p.max - maximum);
    }
    return OnlineSoftmax{maximum, total};
}

// Maximum of one value per thread, each non-negative or NaN, NaN if any is
// (a group's peak, `packets::peak_with`), as `group_max`: such values'
// bits order as the values do with every NaN above +Inf, so integer maxima
// reduce them, one `redux.sync` a warp.
__device__ __forceinline__ float group_peak(float value, float *partials) {
    unsigned bits = seismic_redux_max_u32(__float_as_uint(value));
    const unsigned warps = blockDim.x / 32;
    __syncthreads();
    if (threadIdx.x % 32 == 0)
        partials[threadIdx.x / 32] = __uint_as_float(bits);
    __syncthreads();
    for (unsigned w = 0; w < warps; ++w)
        bits = max(bits, __float_as_uint(partials[w]));
    return __uint_as_float(bits);
}

// A (score, index) candidate for argmax reductions.
struct Candidate {
    float score;
    int index;
};

// Tie rules of an argmax: among candidates of equal score, the higher or the
// lower index wins.
struct HigherIndex {
    __device__ static __forceinline__ int pick(int candidate) { return seismic_redux_max_s32(candidate); }
    __device__ static __forceinline__ bool wins_tie(int a, int b) { return a > b; }
    static constexpr int none = -1;
};
struct LowerIndex {
    __device__ static __forceinline__ int pick(int candidate) { return seismic_redux_min_s32(candidate); }
    __device__ static __forceinline__ bool wins_tie(int a, int b) { return a < b; }
    static constexpr int none = 0x7fffffff;
};

// The warp's maximum score and the index TIES picks among the lanes holding
// it. Every lane receives both; all 32 lanes must be convergent.
template <class TIES> __device__ __forceinline__ int argmax(float score, int index, float &best) {
    best = seismic_warp_max_f32(score);
    return TIES::pick(score == best ? index : TIES::none);
}

// Maximum score and winning index under TIES across the block. `partials`
// holds one Candidate per warp (at most 32); the leading barrier lets
// consecutive calls share it. Every thread of the block must call it.
template <class TIES>
__device__ __forceinline__ Candidate group_argmax(Candidate local, Candidate *partials) {
    float warp_score;
    const int warp_index = argmax<TIES>(local.score, local.index, warp_score);
    const unsigned warps = blockDim.x / 32;
    __syncthreads();
    if (threadIdx.x % 32 == 0)
        partials[threadIdx.x / 32] = Candidate{warp_score, warp_index};
    __syncthreads();
    Candidate best = partials[0];
    for (unsigned w = 1; w < warps; ++w) {
        const Candidate c = partials[w];
        if (best.score != best.score || c.score > best.score ||
            (c.score == best.score && TIES::wins_tie(c.index, best.index)))
            best = c;
    }
    return best;
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
