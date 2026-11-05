// Groups the choices of the rows[0] rows from row first[0] by expert into T-row tiles;
// the CUDA form of `metal/routed_group.metal`. `routed_group_count` counts
// each chunk of GROUP_CHOICES flat choices per expert (shared atomics: a
// count is order-free); `routed_group_place` runs one block of E * PARTS
// threads per chunk, PARTS = 1024 / E. Thread e < E sums expert e's partials
// over every chunk and over the chunks before this one; the experts' first
// tiles are an exclusive block scan of their tile counts. The chunk's routes
// are staged as 16-bit experts, padded with 0xffff to whole 8-choice vectors;
// thread (part, expert) owns the choices of `part` (a contiguous slice of the
// chunk, whole vectors) that route to `expert` and walks its slice a vector
// at a time, so every expert's rows keep their flat (row, choice) order. The
// first chunk's block writes the counts, the experts' blocks, their last
// tiles' padding and `used`.

namespace {

constexpr unsigned experts = SEISMIC_DIM_E;
constexpr unsigned parts = 1024 / experts;
constexpr unsigned threads = experts * parts;
constexpr unsigned warps = (threads + 31) / 32;
constexpr unsigned K = SEISMIC_DIM_K;
constexpr unsigned chunk_choices = SEISMIC_CONST_GROUP_CHOICES;

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

// The flat index of the first grouped choice: row first[0]'s.
#define GROUP_BASE (static_cast<unsigned>(reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_FIRST))[0]) * K)
// The grouped choices: K of each of the rows[0] rows from the first (within
// the M rows: the contract bounds first[0] + rows[0]).
#define GROUP_LIVE (static_cast<unsigned>(reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROWS))[0]) * K)
#define GROUP_ROUTE(flat)                                                                                             \
    (reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES))[static_cast<unsigned long long>((flat) / K) *   \
                                                                           SEISMIC_ROUTES_STRIDE_0 +                   \
                                                                       static_cast<unsigned long long>((flat) % K) *   \
                                                                           SEISMIC_ROUTES_STRIDE_1])

extern "C" __global__ void routed_group_count(SEISMIC_KERNEL_PARAMS) {
    unsigned *partials = reinterpret_cast<unsigned *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS));
    extern __shared__ unsigned histogram[];
    const unsigned base = GROUP_BASE;
    const unsigned live = GROUP_LIVE;
    const unsigned begin = blockIdx.x * chunk_choices;
    if (begin >= live)
        return;
    const unsigned end = min(begin + chunk_choices, live);
    for (unsigned expert = threadIdx.x; expert < experts; expert += blockDim.x)
        histogram[expert] = 0;
    __syncthreads();
    for (unsigned flat = begin + threadIdx.x; flat < end; flat += blockDim.x)
        atomicAdd(&histogram[GROUP_ROUTE(base + flat)], 1u);
    __syncthreads();
    for (unsigned expert = threadIdx.x; expert < experts; expert += blockDim.x)
        partials[static_cast<unsigned long long>(blockIdx.x) * experts + expert] = histogram[expert];
}

extern "C" __global__ void routed_group_place(SEISMIC_KERNEL_PARAMS) {
    int *counts = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_COUNTS));
    int *order = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_ORDER));
    int *inverse = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_INVERSE));
    int *blocks = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_BLOCKS));
    int *used = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_USED));
    const unsigned *partials = reinterpret_cast<const unsigned *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS));
    extern __shared__ uint4 group_shared[];
    const unsigned tile = static_cast<unsigned>(SEISMIC_DIM_T);
    const unsigned live = GROUP_LIVE;
    const unsigned chunks = (live + chunk_choices - 1) / chunk_choices;
    const unsigned chunk = blockIdx.x;
    // The first chunk's block writes the per-expert tables even when no
    // choice is grouped.
    if (chunk >= max(chunks, 1u))
        return;
    const unsigned base = GROUP_BASE;
    const unsigned first = chunk * chunk_choices;
    const unsigned choices = min(chunk_choices, live - min(first, live));
    const unsigned vectors = (choices + 7) / 8;
    uint4 *staged = group_shared;
    unsigned *slices = reinterpret_cast<unsigned *>(group_shared + chunk_choices / 8);
    unsigned *bases = slices + threads;
    unsigned *sums = bases + experts;
    const unsigned thread = threadIdx.x;
    const unsigned lane = thread % 32, warp = thread / 32;
    const unsigned expert = thread % experts;
    const unsigned part = thread / experts;
    const unsigned span = (vectors + parts - 1) / parts;
    const unsigned begin = min(part * span, vectors);
    const unsigned end = min(begin + span, vectors);

    unsigned short *halves = reinterpret_cast<unsigned short *>(staged);
    for (unsigned flat = thread; flat < vectors * 8; flat += threads)
        halves[flat] = flat < choices ? static_cast<unsigned short>(GROUP_ROUTE(base + first + flat))
                                      : static_cast<unsigned short>(0xffffu);
    // Expert e's choices in every chunk and in the chunks before this one.
    unsigned total = 0, before = 0;
    if (part == 0)
        for (unsigned other = 0; other < chunks; ++other) {
            const unsigned value = partials[static_cast<unsigned long long>(other) * experts + expert];
            total += value;
            before += other < chunk ? value : 0;
        }
    __syncthreads();
    unsigned count = 0;
    for (unsigned vector = begin; vector < end; ++vector)
        count += matches(staged[vector], expert);
    slices[expert * parts + part] = count;

    // Exclusive scan of the experts' tile counts (thread e < E holds expert e).
    const unsigned tiles = (total + tile - 1) / tile;
    unsigned inclusive = tiles;
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
    const unsigned start = inclusive + (warp > 0 ? sums[warp - 1] : 0) - tiles;
    if (part == 0) {
        bases[expert] = start * tile + before;
        if (chunk == 0) {
            counts[expert * SEISMIC_COUNTS_STRIDE_0] = static_cast<int>(total);
            for (unsigned padding = start * tile + total; padding < (start + tiles) * tile; ++padding)
                order[static_cast<unsigned long long>(padding / tile) * SEISMIC_ORDER_STRIDE_0 +
                      static_cast<unsigned long long>(padding % tile) * SEISMIC_ORDER_STRIDE_1] = -1;
            for (unsigned block = start; block < start + tiles; ++block)
                blocks[block * SEISMIC_BLOCKS_STRIDE_0] = static_cast<int>(expert);
            if (expert == experts - 1)
                used[0] = static_cast<int>(start + tiles);
        }
    }
    __syncthreads();

    unsigned position = bases[expert];
    for (unsigned slice = 0; slice < part; ++slice)
        position += slices[expert * parts + slice];
    for (unsigned vector = begin; vector < end; ++vector) {
        const uint4 packed = staged[vector];
        if (matches(packed, expert) == 0)
            continue;
        const unsigned words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
        for (unsigned i = 0; i < 8; ++i) {
            if (((words[i / 2] >> (16 * (i % 2))) & 0xffffu) != expert)
                continue;
            const unsigned flat = base + first + vector * 8 + i;
            const unsigned row = flat / K;
            order[static_cast<unsigned long long>(position / tile) * SEISMIC_ORDER_STRIDE_0 +
                  static_cast<unsigned long long>(position % tile) * SEISMIC_ORDER_STRIDE_1] = static_cast<int>(row);
            inverse[static_cast<unsigned long long>(row) * SEISMIC_INVERSE_STRIDE_0 +
                    static_cast<unsigned long long>(flat % K) * SEISMIC_INVERSE_STRIDE_1] = static_cast<int>(position);
            ++position;
        }
    }
}
