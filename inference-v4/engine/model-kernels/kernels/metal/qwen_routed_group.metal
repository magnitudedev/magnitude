// Groups the M * K routed choices by expert into T-row tiles. One
// threadgroup; thread (expert, part) owns the choices of `part` (a contiguous
// slice of the flat choice order) that route to `expert`. Counting and
// placement walk the same slice in order, so every expert's rows keep their
// flat (row, choice) order and the tables are deterministic.

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
    constexpr uint threads = SEISMIC_DIM_E * parts;
    threadgroup uint *starts = slices + threads;
    threadgroup uint *used = starts + SEISMIC_DIM_E;
    const uint expert = tid / parts;
    const uint part = tid % parts;
    const ulong tile = SEISMIC_DIM_T;
    const ulong choices = SEISMIC_DIM_M * SEISMIC_DIM_K;
    const ulong span = (choices + parts - 1) / parts;
    const ulong begin = metal::min(ulong(part) * span, choices);
    const ulong end = metal::min(begin + span, choices);

    uint count = 0;
    for (ulong flat = begin; flat < end; ++flat) {
        const ulong row = flat / SEISMIC_DIM_K;
        const ulong choice = flat % SEISMIC_DIM_K;
        count += routes[row * SEISMIC_ROUTES_STRIDE_0 + choice * SEISMIC_ROUTES_STRIDE_1] == int(expert);
    }
    slices[tid] = count;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        uint block = 0;
        for (uint index = 0; index < SEISMIC_DIM_E; ++index) {
            starts[index] = block;
            uint total = 0;
            for (uint slice = 0; slice < parts; ++slice) total += slices[index * parts + slice];
            block += uint((ulong(total) + tile - 1) / tile);
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
    const ulong base = ulong(starts[expert]) * tile;
    ulong position = base + preceding;
    for (ulong flat = begin; flat < end; ++flat) {
        const ulong row = flat / SEISMIC_DIM_K;
        const ulong choice = flat % SEISMIC_DIM_K;
        if (routes[row * SEISMIC_ROUTES_STRIDE_0 + choice * SEISMIC_ROUTES_STRIDE_1] != int(expert))
            continue;
        order[(position / tile) * SEISMIC_ORDER_STRIDE_0 + (position % tile) * SEISMIC_ORDER_STRIDE_1] = int(row);
        inverse[row * SEISMIC_INVERSE_STRIDE_0 + choice * SEISMIC_INVERSE_STRIDE_1] = int(position);
        ++position;
    }
    if (part == 0) {
        counts[expert * SEISMIC_COUNTS_STRIDE_0] = int(total);
        const ulong tiles = (ulong(total) + tile - 1) / tile;
        for (ulong padding = base + total; padding < base + tiles * tile; ++padding)
            order[(padding / tile) * SEISMIC_ORDER_STRIDE_0 + (padding % tile) * SEISMIC_ORDER_STRIDE_1] = -1;
        for (ulong block = starts[expert]; block < starts[expert] + tiles; ++block)
            blocks[block * SEISMIC_BLOCKS_STRIDE_0] = int(expert);
    }
    for (ulong block = ulong(used[0]) + tid; block < SEISMIC_DIM_B; block += threads) {
        blocks[block * SEISMIC_BLOCKS_STRIDE_0] = -1;
        for (ulong lane = 0; lane < tile; ++lane)
            order[block * SEISMIC_ORDER_STRIDE_0 + lane * SEISMIC_ORDER_STRIDE_1] = -1;
    }
}
