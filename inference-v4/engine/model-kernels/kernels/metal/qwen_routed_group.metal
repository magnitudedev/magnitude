// Groups the M * K routed choices by expert into T-row tiles. One
// threadgroup. The routes are staged once into threadgroup memory; thread
// (part, expert) owns the choices of `part` (a contiguous slice of the flat
// choice order) that route to `expert`. Counting and placement walk the same
// slice in order, so every expert's rows keep their flat (row, choice) order
// and the tables are deterministic. Threads of one simdgroup share a part, so
// their slice reads are broadcasts.

kernel void qwen_routed_group(
    device const int *routes [[buffer(SEISMIC_BUFFER_ROUTES)]],
    device int *counts [[buffer(SEISMIC_BUFFER_COUNTS)]],
    device int *order [[buffer(SEISMIC_BUFFER_ORDER)]],
    device int *inverse [[buffer(SEISMIC_BUFFER_INVERSE)]],
    device int *blocks [[buffer(SEISMIC_BUFFER_BLOCKS)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    threadgroup uint *slices [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint parts = SEISMIC_TUNE_PARTS;
    constexpr uint experts = SEISMIC_DIM_E;
    constexpr uint threads = experts * parts;
    constexpr uint K = SEISMIC_DIM_K;
    const uint tile = uint(SEISMIC_DIM_T);
    threadgroup uint *starts = slices + threads;
    threadgroup uint *used = starts + experts;
    threadgroup ushort *staged = reinterpret_cast<threadgroup ushort *>(used + 4);
    const uint expert = tid % experts;
    const uint part = tid / experts;
    const uint choices = uint(SEISMIC_DIM_M) * K;
    const uint span = (choices + parts - 1) / parts;
    const uint begin = metal::min(part * span, choices);
    const uint end = metal::min(begin + span, choices);

    for (uint flat = tid; flat < choices; flat += threads)
        staged[flat] = ushort(routes[ulong(flat / K) * SEISMIC_ROUTES_STRIDE_0
            + ulong(flat % K) * SEISMIC_ROUTES_STRIDE_1]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint count = 0;
    for (uint flat = begin; flat < end; ++flat)
        count += staged[flat] == expert;
    slices[expert * parts + part] = count;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        uint block = 0;
        for (uint index = 0; index < experts; ++index) {
            starts[index] = block;
            uint total = 0;
            for (uint slice = 0; slice < parts; ++slice) total += slices[index * parts + slice];
            block += (total + tile - 1) / tile;
        }
        used[0] = block;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint total = 0;
    uint preceding = 0;
    for (uint slice = 0; slice < parts; ++slice) {
        const uint value = slices[expert * parts + slice];
        preceding += slice < part ? value : 0;
        total += value;
    }
    const uint base = starts[expert] * tile;
    uint position = base + preceding;
    for (uint flat = begin; flat < end; ++flat) {
        if (staged[flat] != expert)
            continue;
        const uint row = flat / K;
        order[ulong(position / tile) * SEISMIC_ORDER_STRIDE_0 + ulong(position % tile) * SEISMIC_ORDER_STRIDE_1] =
            int(row);
        inverse[ulong(row) * SEISMIC_INVERSE_STRIDE_0 + ulong(flat % K) * SEISMIC_INVERSE_STRIDE_1] = int(position);
        ++position;
    }
    if (part == 0) {
        counts[expert * SEISMIC_COUNTS_STRIDE_0] = int(total);
        const uint tiles = (total + tile - 1) / tile;
        for (uint padding = base + total; padding < base + tiles * tile; ++padding)
            order[ulong(padding / tile) * SEISMIC_ORDER_STRIDE_0 + ulong(padding % tile) * SEISMIC_ORDER_STRIDE_1] =
                -1;
        for (uint block = starts[expert]; block < starts[expert] + tiles; ++block)
            blocks[block * SEISMIC_BLOCKS_STRIDE_0] = int(expert);
    }
    for (uint flat = used[0] * tile + tid; flat < uint(SEISMIC_DIM_B) * tile; flat += threads) {
        const uint block = flat / tile;
        const uint lane = flat % tile;
        if (lane == 0)
            blocks[block * SEISMIC_BLOCKS_STRIDE_0] = -1;
        order[ulong(block) * SEISMIC_ORDER_STRIDE_0 + ulong(lane) * SEISMIC_ORDER_STRIDE_1] = -1;
    }
}
