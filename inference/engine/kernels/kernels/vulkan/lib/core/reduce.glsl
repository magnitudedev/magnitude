// Fixed-order reductions over a subgroup or a workgroup. The counterpart of
// `metal/lib/core/reduce.h` and `cuda/lib/core/reduce.cuh`.
//
// Subgroups are the prelude's 32-lane logical subgroups. A subgroup sum is the
// prelude's fixed xor butterfly (`seismic_subgroup_sum_f32`: xor 16, 8, 4, 2,
// 1), never `subgroupAdd`, whose order is implementation-defined. A workgroup
// reduction first reduces each subgroup, then combines the subgroup partials
// in index order, so its result depends only on the workgroup's width, never
// on scheduling or the device. Every invocation receives the result.
//
// Workgroup reductions keep their partials in the shared region: `partials`
// is an offset in floats into `seismic_shared_f32` with room for one float
// per subgroup; the leading barrier lets consecutive calls share it. Every
// invocation of the workgroup must call them.

// Sum of `value` over the aligned groups of `width` lanes (a power of two,
// at most 32), in the butterfly order xor width/2, ..., 1.
float reduce_lanes_sum(float value, const uint width) {
    [[unroll]] for (uint mask = width / 2u; mask > 0u; mask >>= 1)
        value += subgroupShuffleXor(value, mask);
    return value;
}

float reduce_group_sum(float value, const uint partials) {
    value = seismic_subgroup_sum_f32(value);
    barrier();
    if (SEISMIC_LANE == 0u)
        seismic_shared_f32[partials + SEISMIC_SUBGROUP] = value;
    barrier();
    float total = 0.0;
    for (uint s = 0u; s < SEISMIC_SUBGROUPS; ++s)
        total += seismic_shared_f32[partials + s];
    return total;
}

float reduce_group_max(float value, const uint partials) {
    value = seismic_subgroup_max(value);
    barrier();
    if (SEISMIC_LANE == 0u)
        seismic_shared_f32[partials + SEISMIC_SUBGROUP] = value;
    barrier();
    float maximum = seismic_shared_f32[partials];
    for (uint s = 1u; s < SEISMIC_SUBGROUPS; ++s)
        maximum = max(maximum, seismic_shared_f32[partials + s]);
    return maximum;
}

// Tie rules of an argmax: among candidates of equal score, the higher or the
// lower index wins.
#define REDUCE_HIGHER_INDEX 0
#define REDUCE_LOWER_INDEX 1

// The subgroup's maximum score (into `best`) and the index `ties` picks among
// the lanes holding it. Every lane receives both.
int reduce_argmax(float score, int index, const int ties, out float best) {
    best = seismic_subgroup_max(score);
    if (ties == REDUCE_HIGHER_INDEX)
        return seismic_subgroup_max(score == best ? index : -1);
    return seismic_subgroup_min(score == best ? index : 0x7fffffff);
}

// Integer scans over the logical subgroup, exact in any order. Every lane of
// the logical subgroup calls them and receives its own result.

// The logical subgroup's lanes for which `predicate` holds: bit l for lane l.
uint reduce_lanes_ballot(bool predicate) {
    const uvec4 lanes = subgroupBallot(predicate);
#if SEISMIC_SUBGROUP_LANES == 32
    return lanes.x;
#else
    return SEISMIC_LANE_BASE == 0u ? lanes.x : lanes.y;
#endif
}

// The sum of `value` over the lanes below this one.
uint reduce_lanes_exclusive_add(uint value) {
    uint inclusive = value;
    [[unroll]] for (uint delta = 1u; delta < 32u; delta <<= 1) {
        const uint below = subgroupShuffleUp(inclusive, delta);
        if (SEISMIC_LANE >= delta)
            inclusive += below;
    }
    return inclusive - value;
}

// One chunk of 32 consecutive entries of a search through their extents:
// lane l holds the extent `length` of entry l of the chunk, and `carried` the
// extents of every entry before the chunk. When some entry of the chunk ends
// past `key` (its first offset plus its extent exceeds it), returns true with
// the first such entry's lane in `found` and its first offset in `carried`;
// otherwise adds the chunk's extents to `carried` and returns false. Every
// lane receives the same results.
bool reduce_lanes_locate(uint length, uint key, inout uint carried, out uint found) {
    const uint before = reduce_lanes_exclusive_add(length);
    const uint hits = reduce_lanes_ballot(carried + before + length > key);
    if (hits == 0u) {
        carried += seismic_subgroup_add(length);
        found = 0u;
        return false;
    }
    found = uint(findLSB(hits));
    carried += seismic_shuffle(before, found);
    return true;
}

// The sum of `value` over the workgroup's invocations before this one, in
// (subgroup, lane) order, and in `total` the sum over all of them. `partials`
// is an offset in words into `seismic_shared_u32` with room for one word per
// subgroup; the leading barrier lets consecutive calls share it. Every
// invocation of the workgroup must call it.
uint reduce_group_exclusive_add(uint value, const uint partials, out uint total) {
    const uint before = reduce_lanes_exclusive_add(value);
    barrier();
    if (SEISMIC_LANE == 31u)
        seismic_shared_u32[partials + SEISMIC_SUBGROUP] = before + value;
    barrier();
    uint earlier = 0u;
    total = 0u;
    for (uint s = 0u; s < SEISMIC_SUBGROUPS; ++s) {
        const uint sum = seismic_shared_u32[partials + s];
        earlier += s < SEISMIC_SUBGROUP ? sum : 0u;
        total += sum;
    }
    return earlier + before;
}
