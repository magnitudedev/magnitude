// Groups the M * K routed choices by expert into T-row tiles; the CUDA form of
// `metal/qwen_routed_group.metal`. One block; thread (expert, part) owns the
// choices of `part` (a contiguous slice of the flat choice order) that route
// to `expert`, so every expert's rows keep their flat (row, choice) order.

extern "C" __global__ void qwen_routed_group(SEISMIC_KERNEL_PARAMS) {
    const int *routes = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES));
    int *counts = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_COUNTS));
    int *order = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_ORDER));
    int *inverse = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_INVERSE));
    int *blocks = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_BLOCKS));
    extern __shared__ unsigned slices[];
    constexpr unsigned parts = SEISMIC_TUNE_PARTS;
    constexpr unsigned threads = SEISMIC_DIM_E * parts;
    unsigned *starts = slices + threads;
    unsigned *used = starts + SEISMIC_DIM_E;
    const unsigned thread = threadIdx.x;
    const unsigned expert = thread / parts;
    const unsigned part = thread % parts;
    const unsigned long long tile = SEISMIC_DIM_T;
    const unsigned long long choices = SEISMIC_DIM_M * SEISMIC_DIM_K;
    const unsigned long long span = (choices + parts - 1) / parts;
    const unsigned long long start = static_cast<unsigned long long>(part) * span;
    const unsigned long long begin = start < choices ? start : choices;
    const unsigned long long end = begin + span < choices ? begin + span : choices;

    unsigned count = 0;
    for (unsigned long long flat = begin; flat < end; ++flat) {
        const unsigned long long row = flat / SEISMIC_DIM_K;
        const unsigned long long choice = flat % SEISMIC_DIM_K;
        count += routes[row * SEISMIC_ROUTES_STRIDE_0 + choice * SEISMIC_ROUTES_STRIDE_1] == static_cast<int>(expert);
    }
    slices[thread] = count;
    __syncthreads();

    if (thread == 0) {
        unsigned block = 0;
        for (unsigned index = 0; index < SEISMIC_DIM_E; ++index) {
            starts[index] = block;
            unsigned total = 0;
            for (unsigned slice = 0; slice < parts; ++slice) total += slices[index * parts + slice];
            block += static_cast<unsigned>((total + tile - 1) / tile);
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
    const unsigned long long base = static_cast<unsigned long long>(starts[expert]) * tile;
    unsigned long long position = base + preceding;
    for (unsigned long long flat = begin; flat < end; ++flat) {
        const unsigned long long row = flat / SEISMIC_DIM_K;
        const unsigned long long choice = flat % SEISMIC_DIM_K;
        if (routes[row * SEISMIC_ROUTES_STRIDE_0 + choice * SEISMIC_ROUTES_STRIDE_1] != static_cast<int>(expert))
            continue;
        order[(position / tile) * SEISMIC_ORDER_STRIDE_0 + (position % tile) * SEISMIC_ORDER_STRIDE_1] =
            static_cast<int>(row);
        inverse[row * SEISMIC_INVERSE_STRIDE_0 + choice * SEISMIC_INVERSE_STRIDE_1] = static_cast<int>(position);
        ++position;
    }
    if (part == 0) {
        counts[expert * SEISMIC_COUNTS_STRIDE_0] = static_cast<int>(total);
        const unsigned long long tiles = (total + tile - 1) / tile;
        for (unsigned long long padding = base + total; padding < base + tiles * tile; ++padding)
            order[(padding / tile) * SEISMIC_ORDER_STRIDE_0 + (padding % tile) * SEISMIC_ORDER_STRIDE_1] = -1;
        for (unsigned long long block = starts[expert]; block < starts[expert] + tiles; ++block)
            blocks[block * SEISMIC_BLOCKS_STRIDE_0] = static_cast<int>(expert);
    }
    for (unsigned long long block = used[0] + thread; block < SEISMIC_DIM_B; block += threads) {
        blocks[block * SEISMIC_BLOCKS_STRIDE_0] = -1;
        for (unsigned long long lane = 0; lane < tile; ++lane)
            order[block * SEISMIC_ORDER_STRIDE_0 + lane * SEISMIC_ORDER_STRIDE_1] = -1;
    }
}
