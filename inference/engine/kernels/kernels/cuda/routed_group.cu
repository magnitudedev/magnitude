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
constexpr unsigned expert_warps = (experts + 31) / 32;
constexpr unsigned K = SEISMIC_DIM_K;
constexpr unsigned chunk_choices = SEISMIC_CONST_GROUP_CHOICES;
constexpr unsigned SUB_CHOICES = 2048;

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

    // Shared memory layout (2 * 2048 * 4 + 32 * 32 * 2 + E * 12 + 128 bytes):
    //   keys0[2048], keys1[2048], hist[32][32] u16, bases[E], sub_start[E], sub_end[E], sums[32].
    unsigned *keys0 = reinterpret_cast<unsigned *>(group_shared);
    unsigned *keys1 = keys0 + SUB_CHOICES;
    unsigned short *hist = reinterpret_cast<unsigned short *>(keys1 + SUB_CHOICES);
    unsigned *bases = reinterpret_cast<unsigned *>(hist + 32u * 32u);
    unsigned *sub_start = bases + experts;
    unsigned *sub_end = sub_start + experts;
    unsigned *sums = sub_end + experts;

    const unsigned thread = threadIdx.x;
    const unsigned lane = thread & 31u;
    const unsigned warp = thread >> 5u;
    const unsigned warps = blockDim.x >> 5u;

    // Expert e's choices in every chunk and in the chunks before this one.
    unsigned total = 0u, before = 0u;
    if (chunks > 1u) {
        if (thread < experts) {
            for (unsigned other = 0u; other < chunks; ++other) {
                const unsigned value = partials[static_cast<unsigned long long>(other) * experts + thread];
                total += value;
                before += other < chunk ? value : 0u;
            }
        }
    } else {
        if (thread < experts)
            bases[thread] = 0u;
        __syncthreads();
        for (unsigned flat = thread; flat < choices; flat += blockDim.x)
            atomicAdd(&bases[GROUP_ROUTE(base + first + flat)], 1u);
        __syncthreads();
        if (thread < experts)
            total = bases[thread];
    }

    // Exclusive scan of the experts' tile counts (thread e < E holds expert e).
    const unsigned tiles = thread < experts ? (total + tile - 1u) / tile : 0u;
    unsigned inclusive = tiles;
    if (warp < expert_warps) {
#pragma unroll
        for (unsigned offset = 1u; offset < 32u; offset *= 2u) {
            const unsigned value = __shfl_up_sync(0xffffffffu, inclusive, offset);
            if (lane >= offset)
                inclusive += value;
        }
        if constexpr (expert_warps > 1) {
            if (lane == 31u && warp < expert_warps - 1u)
                sums[warp] = inclusive;
        }
    }
    if constexpr (expert_warps > 1) {
        __syncthreads();
        if (warp == 0u) {
            unsigned value = lane < expert_warps - 1u ? sums[lane] : 0u;
#pragma unroll
            for (unsigned offset = 1u; offset < 32u; offset *= 2u) {
                const unsigned other = __shfl_up_sync(0xffffffffu, value, offset);
                if (lane >= offset)
                    value += other;
            }
            if (lane < expert_warps - 1u)
                sums[lane] = value;
        }
        __syncthreads();
    }
    if (thread < experts) {
        const unsigned start = inclusive + (expert_warps > 1 && warp > 0u ? sums[warp - 1u] : 0u) - tiles;
        bases[thread] = start * tile + before;
        if (chunk == 0u) {
            counts[thread * SEISMIC_COUNTS_STRIDE_0] = static_cast<int>(total);
            for (unsigned padding = start * tile + total; padding < (start + tiles) * tile; ++padding)
                order[static_cast<unsigned long long>(padding / tile) * SEISMIC_ORDER_STRIDE_0 +
                      static_cast<unsigned long long>(padding % tile) * SEISMIC_ORDER_STRIDE_1] = -1;
            for (unsigned block = start; block < start + tiles; ++block)
                blocks[block * SEISMIC_BLOCKS_STRIDE_0] = static_cast<int>(thread);
            if (thread == experts - 1u)
                used[0] = static_cast<int>(start + tiles);
        }
    }
    __syncthreads();

    // O(N) stable order-preserving scatter in sub-chunks of SUB_CHOICES:
    // 2-pass 5-bit stable radix sort on (expert << 16 | local_flat) in shared
    // memory (covers E <= 1024), then direct O(1) placement per choice.
    for (unsigned sub_first = 0u; sub_first < choices; sub_first += SUB_CHOICES) {
        const unsigned sub_n = min(SUB_CHOICES, choices - sub_first);
        for (unsigned i = thread; i < sub_n; i += blockDim.x) {
            const unsigned ex = static_cast<unsigned>(GROUP_ROUTE(base + first + sub_first + i));
            keys0[i] = (ex << 16u) | i;
        }
        __syncthreads();

        const unsigned per_warp = (sub_n + warps - 1u) / warps;
        const unsigned w_lo = min(warp * per_warp, sub_n);
        const unsigned w_hi = min(w_lo + per_warp, sub_n);

#pragma unroll
        for (unsigned pass = 0u; pass < 2u; ++pass) {
            const unsigned shift = 16u + pass * 5u;
            const unsigned *src = (pass == 0u) ? keys0 : keys1;
            unsigned *dst = (pass == 0u) ? keys1 : keys0;

            for (unsigned i = thread; i < 32u * 32u; i += blockDim.x)
                hist[i] = 0u;
            __syncthreads();

            for (unsigned pos = w_lo; pos < w_hi; pos += 32u) {
                const bool act = (pos + lane) < w_hi;
                const unsigned bin = act ? ((src[pos + lane] >> shift) & 31u) : 32u;
                const unsigned same = __match_any_sync(0xffffffffu, bin);
                if (act && (same & ((1u << lane) - 1u)) == 0u)
                    hist[bin * 32u + warp] += static_cast<unsigned short>(__popc(same));
            }
            __syncthreads();

            // Exclusive prefix sum across 32 bins x 32 warps.
            for (unsigned b = warp; b < 32u; b += warps) {
                unsigned val = static_cast<unsigned>(hist[b * 32u + lane]);
                unsigned inc = val;
#pragma unroll
                for (unsigned offset = 1u; offset < 32u; offset *= 2u) {
                    const unsigned up = __shfl_up_sync(0xffffffffu, inc, offset);
                    if (lane >= offset)
                        inc += up;
                }
                hist[b * 32u + lane] = static_cast<unsigned short>(inc - val);
                if (lane == 31u)
                    sums[b] = inc;
            }
            __syncthreads();

            if (warp == 0u) {
                unsigned val = sums[lane];
                unsigned inc = val;
#pragma unroll
                for (unsigned offset = 1u; offset < 32u; offset *= 2u) {
                    const unsigned up = __shfl_up_sync(0xffffffffu, inc, offset);
                    if (lane >= offset)
                        inc += up;
                }
                sums[lane] = inc - val;
            }
            __syncthreads();

            for (unsigned b = warp; b < 32u; b += warps)
                hist[b * 32u + lane] = static_cast<unsigned short>(static_cast<unsigned>(hist[b * 32u + lane]) + sums[b]);
            __syncthreads();

            for (unsigned pos = w_lo; pos < w_hi; pos += 32u) {
                const bool act = (pos + lane) < w_hi;
                const unsigned key = act ? src[pos + lane] : 0u;
                const unsigned bin = act ? ((key >> shift) & 31u) : 32u;
                const unsigned same = __match_any_sync(0xffffffffu, bin);
                const unsigned r = __popc(same & ((1u << lane) - 1u));
                const unsigned base_off = act ? static_cast<unsigned>(hist[bin * 32u + warp]) : 0u;
                if (act)
                    dst[base_off + r] = key;
                __syncwarp();
                if (act && r == 0u)
                    hist[bin * 32u + warp] = static_cast<unsigned short>(base_off + __popc(same));
                __syncwarp();
            }
            __syncthreads();
        }

        // keys0[0..sub_n) is now stably sorted by (expert, local_flat).
        for (unsigned i = thread; i < sub_n; i += blockDim.x) {
            const unsigned ex = keys0[i] >> 16u;
            if (i == 0u || (keys0[i - 1u] >> 16u) != ex)
                sub_start[ex] = i;
            if (i + 1u == sub_n || (keys0[i + 1u] >> 16u) != ex)
                sub_end[ex] = i + 1u;
        }
        __syncthreads();

        for (unsigned i = thread; i < sub_n; i += blockDim.x) {
            const unsigned key = keys0[i];
            const unsigned ex = key >> 16u;
            const unsigned local_flat = key & 0xffffu;
            const unsigned position = bases[ex] + (i - sub_start[ex]);
            const unsigned flat = base + first + sub_first + local_flat;
            const unsigned row = flat / K;
            order[static_cast<unsigned long long>(position / tile) * SEISMIC_ORDER_STRIDE_0 +
                  static_cast<unsigned long long>(position % tile) * SEISMIC_ORDER_STRIDE_1] = static_cast<int>(row);
            inverse[static_cast<unsigned long long>(row) * SEISMIC_INVERSE_STRIDE_0 +
                    static_cast<unsigned long long>(flat % K) * SEISMIC_INVERSE_STRIDE_1] = static_cast<int>(position);
        }
        __syncthreads();

        for (unsigned i = thread; i < sub_n; i += blockDim.x) {
            const unsigned ex = keys0[i] >> 16u;
            if (i == 0u || (keys0[i - 1u] >> 16u) != ex)
                bases[ex] += sub_end[ex] - sub_start[ex];
        }
        __syncthreads();
    }
}
