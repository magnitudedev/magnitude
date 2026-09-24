// Fixed-order reductions over a simdgroup or a threadgroup. A threadgroup
// reduction first reduces each simdgroup (`simd_sum`/`simd_max`), then
// combines the simdgroup partials in index order, so its result depends only
// on the threadgroup's width, never on scheduling. Every thread receives the
// result.

namespace reduce {

// Sum of one value per thread of a threadgroup of SIMDGROUPS simdgroups.
// `partials` holds SIMDGROUPS floats; the leading barrier lets consecutive
// calls share it. Every thread of the threadgroup must call it.
template <uint SIMDGROUPS>
inline float group_sum(float value, threadgroup float *partials, uint sg, uint lane) {
    value = simd_sum(value);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0)
        partials[sg] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint j = 0; j < SIMDGROUPS; ++j)
        total += partials[j];
    return total;
}

// Maximum of one value per thread, as `group_sum`.
template <uint SIMDGROUPS>
inline float group_max(float value, threadgroup float *partials, uint sg, uint lane) {
    value = simd_max(value);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0)
        partials[sg] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maximum = partials[0];
    for (uint j = 1; j < SIMDGROUPS; ++j)
        maximum = metal::max(maximum, partials[j]);
    return maximum;
}

// Tie rules of an argmax: among candidates of equal score, the higher or the
// lower index wins.
struct HigherIndex {
    static int pick(int candidate) { return simd_max(candidate); }
    static constant constexpr int none = -1;
};
struct LowerIndex {
    static int pick(int candidate) { return simd_min(candidate); }
    static constant constexpr int none = 0x7fffffff;
};

// The simdgroup's maximum score and the index TIES picks among the lanes
// holding it. Every lane receives both.
template <typename TIES>
inline int argmax(float score, int index, thread float &best) {
    best = simd_max(score);
    return TIES::pick(score == best ? index : TIES::none);
}

} // namespace reduce
