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

} // namespace reduce
