// Groups the M * K routed choices by expert into T-row tiles; the CUDA form of
// `metal/qwen_routed_group.metal`. One block. The routes are staged once into
// shared memory; thread (part, expert) owns the choices of `part` (a
// contiguous slice of the flat choice order) that route to `expert`, so every
// expert's rows keep their flat (row, choice) order. Threads of one warp share
// a part, so their slice reads are broadcasts.

extern "C" __global__ void qwen_routed_group(SEISMIC_KERNEL_PARAMS) {
    const int *routes = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES));
    int *counts = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_COUNTS));
    int *order = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_ORDER));
    int *inverse = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_INVERSE));
    int *blocks = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_BLOCKS));
    extern __shared__ unsigned slices[];
    constexpr unsigned parts = SEISMIC_TUNE_PARTS;
    constexpr unsigned experts = SEISMIC_DIM_E;
    constexpr unsigned threads = experts * parts;
    constexpr unsigned K = SEISMIC_DIM_K;
    const unsigned tile = static_cast<unsigned>(SEISMIC_DIM_T);
    unsigned *starts = slices + threads;
    unsigned *used = starts + experts;
    unsigned short *staged = reinterpret_cast<unsigned short *>(used + 4);
    const unsigned thread = threadIdx.x;
    const unsigned expert = thread % experts;
    const unsigned part = thread / experts;
    const unsigned choices = static_cast<unsigned>(SEISMIC_DIM_M) * K;
    const unsigned span = (choices + parts - 1) / parts;
    const unsigned begin = min(part * span, choices);
    const unsigned end = min(begin + span, choices);

    for (unsigned flat = thread; flat < choices; flat += threads)
        staged[flat] = static_cast<unsigned short>(
            routes[static_cast<unsigned long long>(flat / K) * SEISMIC_ROUTES_STRIDE_0 +
                   static_cast<unsigned long long>(flat % K) * SEISMIC_ROUTES_STRIDE_1]);
    __syncthreads();

    unsigned count = 0;
    for (unsigned flat = begin; flat < end; ++flat)
        count += staged[flat] == expert;
    slices[expert * parts + part] = count;
    __syncthreads();

    if (thread == 0) {
        unsigned block = 0;
        for (unsigned index = 0; index < experts; ++index) {
            starts[index] = block;
            unsigned total = 0;
            for (unsigned slice = 0; slice < parts; ++slice) total += slices[index * parts + slice];
            block += (total + tile - 1) / tile;
        }
        used[0] = block;
    }
    __syncthreads();

    unsigned total = 0;
    unsigned preceding = 0;
    for (unsigned slice = 0; slice < parts; ++slice) {
        const unsigned value = slices[expert * parts + slice];
        preceding += slice < part ? value : 0;
        total += value;
    }
    const unsigned base = starts[expert] * tile;
    unsigned position = base + preceding;
    for (unsigned flat = begin; flat < end; ++flat) {
        if (staged[flat] != expert)
            continue;
        const unsigned row = flat / K;
        order[static_cast<unsigned long long>(position / tile) * SEISMIC_ORDER_STRIDE_0 +
              static_cast<unsigned long long>(position % tile) * SEISMIC_ORDER_STRIDE_1] = static_cast<int>(row);
        inverse[static_cast<unsigned long long>(row) * SEISMIC_INVERSE_STRIDE_0 +
                static_cast<unsigned long long>(flat % K) * SEISMIC_INVERSE_STRIDE_1] = static_cast<int>(position);
        ++position;
    }
    if (part == 0) {
        counts[expert * SEISMIC_COUNTS_STRIDE_0] = static_cast<int>(total);
        const unsigned tiles = (total + tile - 1) / tile;
        for (unsigned padding = base + total; padding < base + tiles * tile; ++padding)
            order[static_cast<unsigned long long>(padding / tile) * SEISMIC_ORDER_STRIDE_0 +
                  static_cast<unsigned long long>(padding % tile) * SEISMIC_ORDER_STRIDE_1] = -1;
        for (unsigned block = starts[expert]; block < starts[expert] + tiles; ++block)
            blocks[block * SEISMIC_BLOCKS_STRIDE_0] = static_cast<int>(expert);
    }
    for (unsigned flat = used[0] * tile + thread; flat < static_cast<unsigned>(SEISMIC_DIM_B) * tile; flat += threads) {
        const unsigned block = flat / tile;
        const unsigned lane = flat % tile;
        if (lane == 0)
            blocks[block * SEISMIC_BLOCKS_STRIDE_0] = -1;
        order[static_cast<unsigned long long>(block) * SEISMIC_ORDER_STRIDE_0 +
              static_cast<unsigned long long>(lane) * SEISMIC_ORDER_STRIDE_1] = -1;
    }
}
