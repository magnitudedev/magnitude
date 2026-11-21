// Fixed-order reductions over a simdgroup or a threadgroup. A threadgroup
// reduction first reduces each simdgroup (`simd_sum`/`simd_max`), then
// combines the simdgroup partials in index order, so its result depends only
// on the threadgroup's width, never on scheduling. Every thread receives the
// result.
#pragma once

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

// Sum of N independent values per thread of a threadgroup of SIMDGROUPS
// simdgroups in a single barrier pair. `partials` holds N * SIMDGROUPS floats;
// the leading barrier lets consecutive calls share it. Every thread of the
// threadgroup must call it.
template <uint N, uint SIMDGROUPS>
inline void group_sum_n(thread float (&values)[N], threadgroup float *partials, uint sg, uint lane) {
    for (uint k = 0; k < N; ++k)
        values[k] = simd_sum(values[k]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        for (uint k = 0; k < N; ++k)
            partials[k * SIMDGROUPS + sg] = values[k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 0; k < N; ++k) {
        float total = 0.0f;
        for (uint j = 0; j < SIMDGROUPS; ++j)
            total += partials[k * SIMDGROUPS + j];
        values[k] = total;
    }
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

// A (score, index) candidate for argmax reductions.
struct Candidate {
    float score;
    int index;
};

// Tie rules of an argmax: among candidates of equal score, the higher or the
// lower index wins.
struct HigherIndex {
    static int pick(int candidate) { return simd_max(candidate); }
    static inline bool wins_tie(int a, int b) { return a > b; }
    static constant constexpr int none = -1;
};
struct LowerIndex {
    static int pick(int candidate) { return simd_min(candidate); }
    static inline bool wins_tie(int a, int b) { return a < b; }
    static constant constexpr int none = 0x7fffffff;
};

// The simdgroup's maximum score and the index TIES picks among the lanes
// holding it. Every lane receives both.
template <typename TIES>
inline int argmax(float score, int index, thread float &best) {
    best = simd_max(score);
    return TIES::pick(score == best ? index : TIES::none);
}

// Integer prefix sums over a threadgroup, exact in any order, and the
// simdgroup-cooperative walks over a table of span lengths built on them.

// The sum of `value` over the threadgroup's threads before this one, and in
// `total` the sum over all of them. `partials` holds SIMDGROUPS uints (at
// most 32); the leading barrier lets consecutive calls share it. Every
// thread of the threadgroup must call it.
template <uint SIMDGROUPS>
inline uint group_exclusive_add(uint value, threadgroup uint *partials, uint sg, uint lane, thread uint &total) {
    static_assert(SIMDGROUPS >= 1 && SIMDGROUPS <= 32, "at most 32 simdgroups");
    const uint inclusive = simd_prefix_inclusive_sum(value);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 31)
        partials[sg] = inclusive;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint sg_sum = lane < SIMDGROUPS ? partials[lane] : 0u;
    const uint sg_before = simd_prefix_exclusive_sum(sg_sum);
    total = simd_sum(sg_sum);
    const uint before = simd_shuffle(sg_before, ushort(sg));
    return before + inclusive - value;
}

// Walks over spans 0..count-1 whose lengths are `length(s)` (in any unit:
// rows, keys, key tiles). Every lane of the simdgroup calls them with the
// same arguments, convergent, and receives the same result.

// The sum of the lengths: lane l sums spans l, l + 32, ..., then `simd_sum`.
template <typename Length>
inline uint span_total(ulong count, Length length, uint lane) {
    uint total = 0;
    for (ulong s = lane; s < count; s += 32)
        total += length(s);
    return simd_sum(total);
}

// A span and its offset: the lengths of the spans before it.
struct Located {
    ulong span;
    uint offset;
};

// The first span s with offset(s) + length(s) > `at`, and offset(s); (count,
// total) when there is none. Spans are taken 32 at a time: an exclusive
// simdgroup prefix of their lengths plus the offset carried from earlier
// chunks, a ballot of the spans ending past `at`, the first of them broadcast.
template <typename Length>
inline Located span_locate(ulong count, uint at, Length length, uint lane) {
    uint carried = 0;
    for (ulong base = 0; base < count; base += 32) {
        const ulong s = base + lane;
        const uint own = s < count ? length(s) : 0u;
        const uint before = carried + simd_prefix_exclusive_sum(own);
        const uint hits = uint(uint64_t(simd_ballot(s < count && at < before + own)));
        if (hits != 0u) {
            const uint first = metal::ctz(hits);
            return Located{base + first, simd_shuffle(before, ushort(first))};
        }
        carried += simd_sum(own);
    }
    return Located{count, carried};
}

// Maximum score and winning index under TIES across a threadgroup of
// SIMDGROUPS simdgroups. `partials` holds SIMDGROUPS Candidates; the leading
// barrier lets consecutive calls share it. Every thread of the threadgroup
// must call it.
template <typename TIES, uint SIMDGROUPS>
inline Candidate group_argmax(Candidate local, threadgroup Candidate *partials, uint sg, uint lane) {
    float sg_score;
    const int sg_index = argmax<TIES>(local.score, local.index, sg_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0)
        partials[sg] = Candidate{sg_score, sg_index};
    threadgroup_barrier(mem_flags::mem_threadgroup);
    Candidate best = partials[0];
    for (uint j = 1; j < SIMDGROUPS; ++j) {
        const Candidate c = partials[j];
        if (best.score != best.score || c.score > best.score ||
            (c.score == best.score && TIES::wins_tie(c.index, best.index)))
            best = c;
    }
    return best;
}

} // namespace reduce
