// Groups the choices of the rows[0] rows from row first[0] by expert into
// T-row tiles, in
// chunks of GROUP_CHOICES flat choices, one threadgroup each, so no launch is
// bounded by the row count.
//
// `routed_group_count`: chunk c's choices per expert, partials[c, e]
// (threadgroup atomics: a count is order-free).
// `routed_group_place`: one threadgroup of E * PARTS threads per chunk,
// PARTS = 1024 / E (the most whole parts per expert a 1024-thread
// threadgroup holds). Thread e < E sums expert e's partials over every chunk
// (its tiles) and over the chunks before this one; the experts' first tiles
// are an exclusive threadgroup scan of their tile counts. The chunk's routes
// are staged as 16-bit experts, padded with 0xffff to whole 8-choice
// vectors; thread (part, expert) owns the choices of `part` (a contiguous
// slice of the chunk, whole vectors) that route to `expert` and walks its
// slice a vector (one 16-byte broadcast load) at a time. Counting and
// placement walk the same slice in order, so every expert's rows keep their
// flat (row, choice) order and the tables are deterministic. The first
// chunk's threadgroup writes the counts, the experts' blocks, their last
// tiles' padding and `used`.

constant constexpr uint group_experts = SEISMIC_DIM_E;
constant constexpr uint group_parts = 1024u / group_experts;
constant constexpr uint group_threads = group_experts * group_parts;
constant constexpr uint group_simdgroups = (group_threads + 31u) / 32u;
constant constexpr uint group_choices = SEISMIC_CONST_GROUP_CHOICES;

// The flat index of the first grouped choice: row first[0]'s.
inline uint group_base(device const int *first, constant ulong *seismic_words) {
    return uint(first[0]) * uint(SEISMIC_DIM_K);
}

// The grouped choices: K of each of the rows[0] rows from the first (within
// the M rows: the contract bounds first[0] + rows[0]).
inline uint group_live(device const int *rows, constant ulong *seismic_words) {
    return uint(rows[0]) * uint(SEISMIC_DIM_K);
}

// The count of `expert` among the 8 staged experts of `vector`.
inline uint group_matches(ushort4 low, ushort4 high, ushort expert) {
    return uint(low.x == expert) + uint(low.y == expert) + uint(low.z == expert) + uint(low.w == expert)
        + uint(high.x == expert) + uint(high.y == expert) + uint(high.z == expert) + uint(high.w == expert);
}

inline int group_route(device const int *routes, uint flat, constant ulong *seismic_words) {
    constexpr uint K = SEISMIC_DIM_K;
    return routes[ulong(flat / K) * SEISMIC_ROUTES_STRIDE_0 + ulong(flat % K) * SEISMIC_ROUTES_STRIDE_1];
}

kernel void routed_group_count(
    device const int *first_row [[buffer(SEISMIC_BUFFER_FIRST)]],
    device const int *rows [[buffer(SEISMIC_BUFFER_ROWS)]],
    device const int *routes [[buffer(SEISMIC_BUFFER_ROUTES)]],
    device uint *partials [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIALS)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    threadgroup atomic_uint *histogram [[threadgroup(0)]],
    uint chunk [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
    const uint base = group_base(first_row, seismic_words);
    const uint live = group_live(rows, seismic_words);
    const uint begin = chunk * group_choices;
    if (begin >= live)
        return;
    const uint end = metal::min(begin + group_choices, live);
    for (uint expert = tid; expert < group_experts; expert += threads)
        atomic_store_explicit(&histogram[expert], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint flat = begin + tid; flat < end; flat += threads)
        atomic_fetch_add_explicit(&histogram[group_route(routes, base + flat, seismic_words)], 1u,
            memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint expert = tid; expert < group_experts; expert += threads)
        partials[ulong(chunk) * group_experts + expert] = atomic_load_explicit(&histogram[expert], memory_order_relaxed);
}

kernel void routed_group_place(
    device const int *first_row [[buffer(SEISMIC_BUFFER_FIRST)]],
    device const int *rows [[buffer(SEISMIC_BUFFER_ROWS)]],
    device const int *routes [[buffer(SEISMIC_BUFFER_ROUTES)]],
    device int *counts [[buffer(SEISMIC_BUFFER_COUNTS)]],
    device int *order [[buffer(SEISMIC_BUFFER_ORDER)]],
    device int *inverse [[buffer(SEISMIC_BUFFER_INVERSE)]],
    device int *blocks [[buffer(SEISMIC_BUFFER_BLOCKS)]],
    device int *used [[buffer(SEISMIC_BUFFER_USED)]],
    device const uint *partials [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIALS)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    threadgroup uchar *memory [[threadgroup(0)]],
    uint chunk [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint tile = uint(SEISMIC_DIM_T);
    const uint live = group_live(rows, seismic_words);
    const uint chunks = (live + group_choices - 1u) / group_choices;
    // The first chunk's threadgroup writes the per-expert tables even when
    // no choice is grouped.
    if (chunk >= metal::max(chunks, 1u))
        return;
    const uint base = group_base(first_row, seismic_words);
    const uint first = chunk * group_choices;
    const uint choices = metal::min(group_choices, live - metal::min(first, live));
    const uint vectors = (choices + 7u) / 8u;
    threadgroup ushort4 *staged = reinterpret_cast<threadgroup ushort4 *>(memory);
    threadgroup ushort *halves = reinterpret_cast<threadgroup ushort *>(memory);
    threadgroup uint *slices = reinterpret_cast<threadgroup uint *>(memory + group_choices * 2u);
    threadgroup uint *bases = slices + group_threads;
    threadgroup uint *sums = bases + group_experts;
    const uint expert = tid % group_experts;
    const uint part = tid / group_experts;
    const uint span = (vectors + group_parts - 1u) / group_parts;
    const uint begin = metal::min(part * span, vectors);
    const uint end = metal::min(begin + span, vectors);
    const ushort mine = ushort(expert);

    for (uint flat = tid; flat < vectors * 8u; flat += group_threads)
        halves[flat] =
            flat < choices ? ushort(group_route(routes, base + first + flat, seismic_words)) : ushort(0xffff);
    // Expert e's choices in every chunk and in the chunks before this one.
    uint total = 0, before = 0;
    if (part == 0)
        for (uint other = 0; other < chunks; ++other) {
            const uint value = partials[ulong(other) * group_experts + expert];
            total += value;
            before += other < chunk ? value : 0u;
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint count = 0;
    for (uint vector = begin; vector < end; ++vector)
        count += group_matches(staged[2u * vector], staged[2u * vector + 1u], mine);
    slices[expert * group_parts + part] = count;
    // Exclusive scan of the experts' tile counts (thread e < E holds expert e).
    const uint tiles = (total + tile - 1u) / tile;
    const uint inclusive = metal::simd_prefix_inclusive_sum(tiles);
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
    const uint start = inclusive + (sg > 0 ? sums[sg - 1u] : 0u) - tiles;
    if (part == 0) {
        bases[expert] = start * tile + before;
        if (chunk == 0) {
            counts[expert * SEISMIC_COUNTS_STRIDE_0] = int(total);
            for (uint padding = start * tile + total; padding < (start + tiles) * tile; ++padding)
                order[ulong(padding / tile) * SEISMIC_ORDER_STRIDE_0 + ulong(padding % tile) * SEISMIC_ORDER_STRIDE_1]
                    = -1;
            for (uint block = start; block < start + tiles; ++block)
                blocks[block * SEISMIC_BLOCKS_STRIDE_0] = int(expert);
            if (expert == group_experts - 1u)
                used[0] = int(start + tiles);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint position = bases[expert];
    for (uint slice = 0; slice < part; ++slice)
        position += slices[expert * group_parts + slice];
    for (uint vector = begin; vector < end; ++vector) {
        const ushort4 low = staged[2u * vector], high = staged[2u * vector + 1u];
        if (group_matches(low, high, mine) == 0)
            continue;
        const ushort values[8] = {low.x, low.y, low.z, low.w, high.x, high.y, high.z, high.w};
        for (uint i = 0; i < 8u; ++i) {
            if (values[i] != mine)
                continue;
            const uint flat = base + first + vector * 8u + i;
            const uint row = flat / uint(SEISMIC_DIM_K);
            order[ulong(position / tile) * SEISMIC_ORDER_STRIDE_0 + ulong(position % tile) * SEISMIC_ORDER_STRIDE_1]
                = int(row);
            inverse[ulong(row) * SEISMIC_INVERSE_STRIDE_0 + ulong(flat % SEISMIC_DIM_K) * SEISMIC_INVERSE_STRIDE_1]
                = int(position);
            ++position;
        }
    }
}
