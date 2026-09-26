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
