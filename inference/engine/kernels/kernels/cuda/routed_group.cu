// Groups the M * K routed choices by expert into T-row tiles; the CUDA form of
// `metal/routed_group.metal`. One block of E * PARTS threads, PARTS = 1024 / E
// (the most whole parts per expert a 1024-thread block holds). The routes are
// staged once into shared memory as 16-bit experts, padded with 0xffff to
// whole 8-choice vectors; thread (part, expert) owns the choices of `part` (a
// contiguous slice of the flat choice order, whole vectors) that route to
// `expert` and walks its slice a vector (one 16-byte broadcast load) at a
// time, so every expert's rows keep their flat (row, choice) order. The
// experts' first tiles are an exclusive block scan of their tile counts.

namespace {

constexpr unsigned experts = SEISMIC_DIM_E;
constexpr unsigned parts = 1024 / experts;
constexpr unsigned threads = experts * parts;
constexpr unsigned warps = (threads + 31) / 32;
constexpr unsigned K = SEISMIC_DIM_K;

// The count of `expert` among the 8 staged experts of `vector`.
__device__ __forceinline__ unsigned matches(uint4 vector, unsigned expert) {
    const unsigned pair = expert | (expert << 16);
    unsigned count = 0;
    const unsigned words[4] = {vector.x, vector.y, vector.z, vector.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned equal = __vcmpeq2(words[i], pair);
        count += __popc(equal) / 16;
    }
    return count;
}

} // namespace

extern "C" __global__ void routed_group(SEISMIC_KERNEL_PARAMS) {
    const int *routes = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES));
    int *counts = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_COUNTS));
    int *order = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_ORDER));
    int *inverse = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_INVERSE));
    int *blocks = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_BLOCKS));
    extern __shared__ uint4 group_shared[];
    uint4 *staged = group_shared;
    const unsigned tile = static_cast<unsigned>(SEISMIC_DIM_T);
    const unsigned choices = static_cast<unsigned>(SEISMIC_DIM_M) * K;
    const unsigned vectors = (choices + 7) / 8;
    unsigned *slices = reinterpret_cast<unsigned *>(staged + vectors);
    unsigned *starts = slices + threads;
    unsigned *sums = starts + experts;
    unsigned *used = sums + warps;
    const unsigned thread = threadIdx.x;
    const unsigned lane = thread % 32, warp = thread / 32;
    const unsigned expert = thread % experts;
    const unsigned part = thread / experts;
    const unsigned span = (vectors + parts - 1) / parts;
    const unsigned begin = min(part * span, vectors);
    const unsigned end = min(begin + span, vectors);

    unsigned short *halves = reinterpret_cast<unsigned short *>(staged);
    for (unsigned flat = thread; flat < vectors * 8; flat += threads)
        halves[flat] = flat < choices
            ? static_cast<unsigned short>(
                  routes[static_cast<unsigned long long>(flat / K) * SEISMIC_ROUTES_STRIDE_0 +
                         static_cast<unsigned long long>(flat % K) * SEISMIC_ROUTES_STRIDE_1])
            : static_cast<unsigned short>(0xffffu);
    __syncthreads();

    unsigned count = 0;
    for (unsigned vector = begin; vector < end; ++vector)
        count += matches(staged[vector], expert);
    slices[expert * parts + part] = count;
    __syncthreads();

    unsigned total = 0;
    unsigned preceding = 0;
    for (unsigned slice = 0; slice < parts; ++slice) {
        const unsigned value = slices[expert * parts + slice];
        preceding += slice < part ? value : 0;
        total += value;
    }
    // Exclusive scan of the experts' tile counts (thread e < E holds expert e).
    const unsigned tiles = (total + tile - 1) / tile;
    const unsigned own = thread < experts ? tiles : 0;
    unsigned inclusive = own;
#pragma unroll
    for (unsigned offset = 1; offset < 32; offset *= 2) {
        const unsigned value = __shfl_up_sync(0xffffffffu, inclusive, offset);
        if (lane >= offset)
            inclusive += value;
    }
    if (lane == 31)
        sums[warp] = inclusive;
    __syncthreads();
    if (warp == 0) {
        unsigned value = lane < warps ? sums[lane] : 0;
#pragma unroll
        for (unsigned offset = 1; offset < 32; offset *= 2) {
            const unsigned other = __shfl_up_sync(0xffffffffu, value, offset);
            if (lane >= offset)
                value += other;
        }
        if (lane < warps)
            sums[lane] = value;
    }
    __syncthreads();
    if (thread < experts) {
        const unsigned scanned = inclusive + (warp > 0 ? sums[warp - 1] : 0);
        starts[thread] = scanned - own;
        if (thread == experts - 1)
            used[0] = scanned;
    }
    __syncthreads();

    const unsigned base = starts[expert] * tile;
    unsigned position = base + preceding;
    for (unsigned vector = begin; vector < end; ++vector) {
        const uint4 packed = staged[vector];
        if (matches(packed, expert) == 0)
            continue;
        const unsigned words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
        for (unsigned i = 0; i < 8; ++i) {
            if (((words[i / 2] >> (16 * (i % 2))) & 0xffffu) != expert)
                continue;
            const unsigned flat = vector * 8 + i;
            const unsigned row = flat / K;
            order[static_cast<unsigned long long>(position / tile) * SEISMIC_ORDER_STRIDE_0 +
                  static_cast<unsigned long long>(position % tile) * SEISMIC_ORDER_STRIDE_1] = static_cast<int>(row);
            inverse[static_cast<unsigned long long>(row) * SEISMIC_INVERSE_STRIDE_0 +
                    static_cast<unsigned long long>(flat % K) * SEISMIC_INVERSE_STRIDE_1] = static_cast<int>(position);
            ++position;
        }
    }
    if (part == 0) {
        counts[expert * SEISMIC_COUNTS_STRIDE_0] = static_cast<int>(total);
        for (unsigned padding = base + total; padding < base + tiles * tile; ++padding)
            order[static_cast<unsigned long long>(padding / tile) * SEISMIC_ORDER_STRIDE_0 +
                  static_cast<unsigned long long>(padding % tile) * SEISMIC_ORDER_STRIDE_1] = -1;
        for (unsigned block = starts[expert]; block < starts[expert] + tiles; ++block)
            blocks[block * SEISMIC_BLOCKS_STRIDE_0] = static_cast<int>(expert);
    }
    for (unsigned flat = used[0] * tile + thread; flat < static_cast<unsigned>(SEISMIC_DIM_B) * tile; flat += threads) {
        const unsigned block = flat / tile;
        const unsigned lane_in_tile = flat % tile;
        if (lane_in_tile == 0)
            blocks[block * SEISMIC_BLOCKS_STRIDE_0] = -1;
        order[static_cast<unsigned long long>(block) * SEISMIC_ORDER_STRIDE_0 +
              static_cast<unsigned long long>(lane_in_tile) * SEISMIC_ORDER_STRIDE_1] = -1;
    }
}
