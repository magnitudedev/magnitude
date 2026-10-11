// Groups the M * K routed choices by expert into T-row tiles. One
// threadgroup of E * PARTS threads, PARTS = 1024 / E (the most whole parts
// per expert a 1024-thread threadgroup holds). The routes are staged once
// into threadgroup memory as 16-bit experts, padded with 0xffff to whole
// 8-choice vectors; thread (part, expert) owns the choices of `part` (a
// contiguous slice of the flat choice order, whole vectors) that route to
// `expert` and walks its slice a vector (one 16-byte broadcast load) at a
// time. Counting and placement walk the same slice in order, so every
// expert's rows keep their flat (row, choice) order and the tables are
// deterministic. The experts' first tiles are an exclusive threadgroup scan
// of their tile counts.

constant constexpr uint group_experts = SEISMIC_DIM_E;
constant constexpr uint group_parts = 1024u / group_experts;
constant constexpr uint group_threads = group_experts * group_parts;
constant constexpr uint group_simdgroups = (group_threads + 31u) / 32u;

// The count of `expert` among the 8 staged experts of `vector`.
inline uint group_matches(ushort4 low, ushort4 high, ushort expert) {
    return uint(low.x == expert) + uint(low.y == expert) + uint(low.z == expert) + uint(low.w == expert)
        + uint(high.x == expert) + uint(high.y == expert) + uint(high.z == expert) + uint(high.w == expert);
}

kernel void routed_group(
    device const int *routes [[buffer(SEISMIC_BUFFER_ROUTES)]],
    device int *counts [[buffer(SEISMIC_BUFFER_COUNTS)]],
    device int *order [[buffer(SEISMIC_BUFFER_ORDER)]],
    device int *inverse [[buffer(SEISMIC_BUFFER_INVERSE)]],
    device int *blocks [[buffer(SEISMIC_BUFFER_BLOCKS)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    threadgroup uchar *memory [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint K = SEISMIC_DIM_K;
    const uint tile = uint(SEISMIC_DIM_T);
    const uint choices = uint(SEISMIC_DIM_M) * K;
    const uint vectors = (choices + 7u) / 8u;
    threadgroup ushort4 *staged = reinterpret_cast<threadgroup ushort4 *>(memory);
    threadgroup ushort *halves = reinterpret_cast<threadgroup ushort *>(memory);
    threadgroup uint *slices = reinterpret_cast<threadgroup uint *>(memory + vectors * 16u);
    threadgroup uint *starts = slices + group_threads;
    threadgroup uint *sums = starts + group_experts;
    threadgroup uint *used = sums + group_simdgroups;
    const uint expert = tid % group_experts;
    const uint part = tid / group_experts;
    const uint span = (vectors + group_parts - 1u) / group_parts;
    const uint begin = metal::min(part * span, vectors);
    const uint end = metal::min(begin + span, vectors);
    const ushort mine = ushort(expert);

    for (uint flat = tid; flat < vectors * 8u; flat += group_threads)
        halves[flat] = flat < choices
            ? ushort(routes[ulong(flat / K) * SEISMIC_ROUTES_STRIDE_0 + ulong(flat % K) * SEISMIC_ROUTES_STRIDE_1])
            : ushort(0xffff);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint count = 0;
    for (uint vector = begin; vector < end; ++vector)
        count += group_matches(staged[2u * vector], staged[2u * vector + 1u], mine);
    slices[expert * group_parts + part] = count;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint total = 0;
    uint preceding = 0;
    for (uint slice = 0; slice < group_parts; ++slice) {
        const uint value = slices[expert * group_parts + slice];
        preceding += slice < part ? value : 0;
        total += value;
    }
    // Exclusive scan of the experts' tile counts (thread e < E holds expert e).
    const uint tiles = (total + tile - 1u) / tile;
    const uint own = tid < group_experts ? tiles : 0u;
    const uint inclusive = metal::simd_prefix_inclusive_sum(own);
    if (lane == 31u)
        sums[sg] = inclusive;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const uint value = lane < group_simdgroups ? sums[lane] : 0u;
        const uint scanned = metal::simd_prefix_inclusive_sum(value);
        if (lane < group_simdgroups)
            sums[lane] = scanned;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < group_experts) {
        const uint scanned = inclusive + (sg > 0 ? sums[sg - 1u] : 0u);
        starts[tid] = scanned - own;
        if (tid == group_experts - 1u)
            used[0] = scanned;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint base = starts[expert] * tile;
    uint position = base + preceding;
    for (uint vector = begin; vector < end; ++vector) {
        const ushort4 low = staged[2u * vector], high = staged[2u * vector + 1u];
        if (group_matches(low, high, mine) == 0)
            continue;
        const ushort values[8] = {low.x, low.y, low.z, low.w, high.x, high.y, high.z, high.w};
        for (uint i = 0; i < 8u; ++i) {
            if (values[i] != mine)
                continue;
            const uint flat = vector * 8u + i;
            const uint row = flat / K;
            order[ulong(position / tile) * SEISMIC_ORDER_STRIDE_0 + ulong(position % tile) * SEISMIC_ORDER_STRIDE_1]
                = int(row);
            inverse[ulong(row) * SEISMIC_INVERSE_STRIDE_0 + ulong(flat % K) * SEISMIC_INVERSE_STRIDE_1] = int(position);
            ++position;
        }
    }
    if (part == 0) {
        counts[expert * SEISMIC_COUNTS_STRIDE_0] = int(total);
        for (uint padding = base + total; padding < base + tiles * tile; ++padding)
            order[ulong(padding / tile) * SEISMIC_ORDER_STRIDE_0 + ulong(padding % tile) * SEISMIC_ORDER_STRIDE_1] =
                -1;
        for (uint block = starts[expert]; block < starts[expert] + tiles; ++block)
            blocks[block * SEISMIC_BLOCKS_STRIDE_0] = int(expert);
    }
    for (uint flat = used[0] * tile + tid; flat < uint(SEISMIC_DIM_B) * tile; flat += group_threads) {
        const uint block = flat / tile;
        const uint lane_in_tile = flat % tile;
        if (lane_in_tile == 0)
            blocks[block * SEISMIC_BLOCKS_STRIDE_0] = -1;
        order[ulong(block) * SEISMIC_ORDER_STRIDE_0 + ulong(lane_in_tile) * SEISMIC_ORDER_STRIDE_1] = -1;
    }
}
