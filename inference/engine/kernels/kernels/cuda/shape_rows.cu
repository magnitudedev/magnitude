// shape_rows: penalties and temperature, then top-k (ties at the cutoff kept),
// min-p against the row maximum, and top-p over the survivors in
// (value descending, token ascending) order.
//   shape_rows_prepare (PARTS x Sx): penalized/tempered values in place over
//     `logits` (-inf where a constrained row's mask rejects a finite value;
//     each thread reads and rewrites its own tokens) and the partition
//     maxima.
//   shape_rows_select (Sx): the cutoffs. Top-k is a count-weighted radix select of
//     the k-th largest key; top-p a mass-weighted radix select of the first
//     token whose inclusive mass reaches ceil(top_p * total), then its rank
//     among equal keys by token index. Masses are exp(v - max) in 2^-32
//     fixed point, so every sum is order-independent and exact.
//   shape_rows_apply (PARTS x Sx): writes -inf over every removed token.
// A zero temperature leaves the penalized values unshaped, as does a tempered
// NaN or +inf anywhere in the row.

#include <seismic/element.cuh>
#include "lib/core/order.cuh"

typedef unsigned int u32;
typedef unsigned long long u64;

#define INF __int_as_float(0x7F800000)

// Scratch per row: PARTS partition-maximum keys, then the selection state.
struct Selection {
    u32 maximum;   // ordered key of the row maximum
    u32 top_k_key; // keys below are removed (0: none)
    u32 top_p_key; // removed: key below, or equal with token above top_p_token
    u32 top_p_token;
    u32 top_p_mode; // 0 inactive, 1 cutoff, 2 every token removed
};

__device__ __forceinline__ bool finite(float value) { return value > -INF && value < INF; }

// Partition "maximum" of a partition holding NaN or +inf, above every finite
// key: the row is left unshaped (sampling reports it).
#define UNSHAPED 0xFFFFFFFFu

// ceil(top_p * total) exactly: top_p = m * 2^-shift (m a 24-bit integer), so
// the product is a 128-bit integer shifted right with rounding up. A token
// survives top-p when the mass before it is below this target (the portable
// `preceding < top_p`).
__device__ u64 top_p_target(float top_p, u64 total) {
    if (!(top_p > 0.0f) || total == 0)
        return 0;
    const u32 bits = __float_as_uint(top_p);
    const u32 biased = bits >> 23;
    const u64 m = biased == 0 ? (u64)(bits & 0x7FFFFFu) : (u64)((bits & 0x7FFFFFu) | 0x800000u);
    const int shift = biased == 0 ? 149 : 150 - (int)biased;
    const u64 low = m * total;
    const u64 high = __umul64hi(m, total);
    if (shift <= 0)
        return low << -shift;
    if (shift >= 128)
        return 1;
    u64 quotient;
    bool remainder;
    if (shift >= 64) {
        quotient = shift == 64 ? high : high >> (shift - 64);
        remainder = low != 0 || (shift > 64 && (high & ((1ull << (shift - 64)) - 1)) != 0);
    } else {
        quotient = (low >> shift) | (high << (64 - shift));
        remainder = (low & ((1ull << shift) - 1)) != 0;
    }
    return quotient + (remainder ? 1 : 0);
}

// Strides may be argument words, which only kernel bodies can read.
#define parameter(params, row, index) (params)[(row) * SEISMIC_PARAMS_STRIDE_0 + (index) * SEISMIC_PARAMS_STRIDE_1]

__device__ __forceinline__ u64 slice_begin(u64 part) {
    const u64 span = (SEISMIC_DIM_V + SEISMIC_TUNE_PARTS - 1) / SEISMIC_TUNE_PARTS;
    return part * span < SEISMIC_DIM_V ? part * span : SEISMIC_DIM_V;
}
__device__ __forceinline__ u64 slice_end(u64 part) {
    const u64 span = (SEISMIC_DIM_V + SEISMIC_TUNE_PARTS - 1) / SEISMIC_TUNE_PARTS;
    const u64 end = slice_begin(part) + span;
    return end < SEISMIC_DIM_V ? end : SEISMIC_DIM_V;
}

__device__ __forceinline__ u32 *row_scratch(unsigned char *scratch, u64 row) {
    return reinterpret_cast<u32 *>(scratch) + row * (SEISMIC_TUNE_PARTS + 8);
}

extern "C" __global__ void shape_rows_prepare(SEISMIC_KERNEL_PARAMS) {
    const element::Out<element::F32> logits = SEISMIC_OUT_LOGITS;
    const float *params = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_PARAMS));
    const int *history = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_HISTORY));
    extern __shared__ int recent[];
    __shared__ u32 warp_maximum[32];
    const u64 row = blockIdx.y;
    const u64 part = blockIdx.x;
    const float temperature = parameter(params, row, 0);
    const float repetition = parameter(params, row, 4);
    const float presence = parameter(params, row, 5);
    const float frequency = parameter(params, row, 6);
    const bool penalized = SEISMIC_DIM_HN > 0 && (repetition != 1.0f || presence != 0.0f || frequency != 0.0f);
    if (penalized) {
        for (u64 h = threadIdx.x; h < SEISMIC_DIM_HN; h += blockDim.x)
            recent[h] = history[row * SEISMIC_HISTORY_STRIDE_0 + h * SEISMIC_HISTORY_STRIDE_1];
        __syncthreads();
    }
    // A constrained row removes every rejected value but NaN and +inf, which
    // keep the row unshaped for sampling to report.
    const bool masked =
        reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_CONSTRAINED))[row * SEISMIC_CONSTRAINED_STRIDE_0] != 0;
    const u32 *bits = reinterpret_cast<const u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_MASK)) + row * SEISMIC_MASK_STRIDE_0;
    const bool scale_temp = temperature != 0.0f && temperature != 1.0f;
    const bool write_back = penalized || masked || scale_temp;
    u32 maximum = 0u;
    for (u64 token = slice_begin(part) + threadIdx.x; token < slice_end(part); token += blockDim.x) {
        const u64 slot = row * SEISMIC_LOGITS_STRIDE_0 + token * SEISMIC_LOGITS_STRIDE_1;
        float value = logits.at(slot);
        if (penalized) {
            int count = 0;
            for (u64 h = 0; h < SEISMIC_DIM_HN; ++h)
                count += recent[h] == (int)token ? 1 : 0;
            if (count > 0) {
                value = value < 0.0f ? value * repetition : value / repetition;
                value = value - presence - frequency * (float)count;
            }
        }
        if (masked && ((bits[(token / 32) * SEISMIC_MASK_STRIDE_1] >> (token % 32)) & 1u) == 0u && value < INF)
            value = -INF;
        if (scale_temp)
            value = value / temperature;
        if (write_back)
            logits.put(slot, value);
        if (finite(value))
            maximum = max(maximum, order::finite_key(value));
        else if (value != -INF)
            maximum = UNSHAPED;
    }
    maximum = seismic_redux_max_u32(maximum);
    if (threadIdx.x % 32 == 0)
        warp_maximum[threadIdx.x / 32] = maximum;
    __syncthreads();
    if (threadIdx.x < 32) {
        maximum = threadIdx.x < blockDim.x / 32 ? warp_maximum[threadIdx.x] : 0u;
        maximum = seismic_redux_max_u32(maximum);
        if (threadIdx.x == 0)
            row_scratch(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STATE), row)[part] = maximum;
    }
}

// ---------------------------------------------------------------------------
// Selection (one block of SELECT_THREADS per row).

#define SELECT_THREADS 1024
#define BINS 2048
#define COMPACT_CAP 2048u

union SelectScratch {
    u64 histogram[BINS];
    struct {
        u32 keys[COMPACT_CAP];
        u32 tokens[COMPACT_CAP];
    } compact;
};

struct Survivor {
    element::Stored<element::F32> line;
    u64 stride;
    float maximum;
    u32 top_k_key;
    float min_p;
    // The finite value at `token` when it survives top-k and min-p.
    __device__ __forceinline__ bool at(u64 token, float &value, u32 &key) const {
        value = line.at(token * stride);
        if (!finite(value))
            return false;
        key = order::finite_key(value);
        return key >= top_k_key && !(min_p > 0.0f && expf(value - maximum) < min_p);
    }
    __device__ __forceinline__ bool at_mass(u64 token, u32 &key, u64 &mass_out) const {
        const float value = line.at(token * stride);
        if (!finite(value))
            return false;
        key = order::finite_key(value);
        if (key < top_k_key)
            return false;
        const float e = expf(value - maximum);
        if (min_p > 0.0f && e < min_p)
            return false;
        mass_out = (u64)(e * 4294967296.0f);
        return true;
    }
    __device__ __forceinline__ u64 mass(float value) const {
        return (u64)(expf(value - maximum) * 4294967296.0f);
    }
};

__device__ __forceinline__ bool comes_before(u32 ka, u32 ta, u32 kb, u32 tb) {
    return ka > kb || (ka == kb && ta < tb);
}

// Parallel bitonic sort of `count <= COMPACT_CAP` (key, token) pairs in shared
// memory in (key descending, token ascending) order.
__device__ void bitonic_sort_compact(u32 count, u32 *keys, u32 *tokens) {
    if (count <= 1u)
        return;
    u32 n = 1u;
    while (n < count)
        n <<= 1;
    for (u32 i = count + threadIdx.x; i < n; i += blockDim.x) {
        keys[i] = 0u;
        tokens[i] = 0xFFFFFFFFu;
    }
    __syncthreads();
    const u32 half = n >> 1;
    const u32 tid = threadIdx.x;
    for (u32 k = 2u; k <= n; k <<= 1) {
        for (u32 j = k >> 1; j > 0u; j >>= 1) {
            if (tid < half) {
                const u32 low_mask = j - 1u;
                const u32 i = ((tid & ~low_mask) << 1) | (tid & low_mask);
                const u32 p = i | j;
                const u32 ki = keys[i];
                const u32 ti = tokens[i];
                const u32 kp = keys[p];
                const u32 tp = tokens[p];
                const bool swap = (i & k) == 0u ? comes_before(kp, tp, ki, ti) : comes_before(ki, ti, kp, tp);
                if (swap) {
                    keys[i] = kp;
                    tokens[i] = tp;
                    keys[p] = ki;
                    tokens[p] = ti;
                }
            }
            __syncthreads();
        }
    }
}

// Top-p cutoff directly from `count <= COMPACT_CAP` candidates sorted in
// (key descending, token ascending) order.
__device__ void solve_top_p_sorted(u32 count, const Survivor &survivor, float top_p, const u32 *keys, const u32 *tokens,
                                   u64 *reduce, Selection *selection, u32 &mode, u32 &top_p_key) {
    __shared__ u32 shared_key;
    const u32 i0 = 2u * threadIdx.x;
    const u32 i1 = i0 + 1u;
    u64 m0 = 0ull, m1 = 0ull;
    if (i0 < count) {
        const u32 k = keys[i0];
        if (k >= survivor.top_k_key) {
            const float e = expf(order::value(k) - survivor.maximum);
            if (!(survivor.min_p > 0.0f && e < survivor.min_p))
                m0 = (u64)(e * 4294967296.0f);
        }
    }
    if (i1 < count) {
        const u32 k = keys[i1];
        if (k >= survivor.top_k_key) {
            const float e = expf(order::value(k) - survivor.maximum);
            if (!(survivor.min_p > 0.0f && e < survivor.min_p))
                m1 = (u64)(e * 4294967296.0f);
        }
    }
    const u64 thread_sum = m0 + m1;
    u64 warp_inc = thread_sum;
    const u32 lane = threadIdx.x & 31u;
    const u32 warp = threadIdx.x >> 5;
    for (int offset = 1; offset < 32; offset <<= 1) {
        const u64 n = __shfl_up_sync(0xFFFFFFFFu, warp_inc, offset);
        if (lane >= (u32)offset)
            warp_inc += n;
    }
    if (lane == 31u)
        reduce[warp] = warp_inc;
    __syncthreads();
    if (warp == 0u) {
        u64 wv = reduce[lane];
        for (int offset = 1; offset < 32; offset <<= 1) {
            const u64 n = __shfl_up_sync(0xFFFFFFFFu, wv, offset);
            if (lane >= (u32)offset)
                wv += n;
        }
        reduce[lane] = wv;
    }
    __syncthreads();
    const u64 total = reduce[31];
    const u64 target = top_p_target(top_p, total);
    if (target == 0) {
        mode = 2u;
        return;
    }
    const u64 warp_base = warp > 0u ? reduce[warp - 1u] : 0ull;
    const u64 excl_i0 = warp_base + (warp_inc - thread_sum);
    const u64 incl_i0 = excl_i0 + m0;
    const u64 incl_i1 = incl_i0 + m1;
    if (i0 < count && excl_i0 < target && target <= incl_i0) {
        selection->top_p_token = tokens[i0];
        shared_key = keys[i0];
    } else if (i1 < count && incl_i0 < target && target <= incl_i1) {
        selection->top_p_token = tokens[i1];
        shared_key = keys[i1];
    }
    __syncthreads();
    top_p_key = shared_key;
    mode = 1u;
}

// Descending digit of a weighted histogram at which the running weight
// `above` + bins reaches `target`; adds the weight of higher bins to `above`.
// Returns BINS when the whole histogram stays below the target.
__device__ u32 descend(const u64 *histogram, u32 bins, u64 target, u64 &above, u64 *lane_totals) {
    // Warp 0: lane l owns bins [bins - (l + 1) * per, bins - l * per), highest first.
    __shared__ u32 digit;
    __shared__ u64 base;
    if (threadIdx.x < 32) {
        const u32 lane = threadIdx.x;
        const u32 per = bins / 32;
        u64 total = 0;
        for (u32 i = 0; i < per; ++i)
            total += histogram[bins - 1 - (lane * per + i)];
        lane_totals[lane] = total;
        __syncwarp();
        if (lane == 0) {
            u64 running = above;
            digit = BINS;
            for (u32 l = 0; l < 32; ++l) {
                if (running + lane_totals[l] >= target) {
                    for (u32 i = 0; i < per; ++i) {
                        const u32 bin = bins - 1 - (l * per + i);
                        if (running + histogram[bin] >= target) {
                            digit = bin;
                            break;
                        }
                        running += histogram[bin];
                    }
                    break;
                }
                running += lane_totals[l];
            }
            base = running;
        }
    }
    __syncthreads();
    above = base;
    return digit;
}

// Radix select over the keys of survivors (weight 1 or mass) starting from
// `start_level` with established `prefix`/`mask`/`above`: finds the key K with
// weight(key > K) < target <= weight(key >= K). Returns false when the total
// weight is below the target; `above` receives weight(key > K).
template <bool MASS>
__device__ bool select_key_from(const Survivor &survivor, u64 target, int start_level, u32 prefix, u32 mask, u64 &above,
                                u32 &key_out, u64 *histogram, u64 *lane_totals) {
    const int shifts[3] = {21, 10, 0};
    const u32 widths[3] = {11, 11, 10};
    for (int level = start_level; level < 3; ++level) {
        const u32 bins = 1u << widths[level];
        for (u32 bin = threadIdx.x; bin < bins; bin += blockDim.x)
            histogram[bin] = 0;
        __syncthreads();
        for (u64 token = threadIdx.x; token < SEISMIC_DIM_V; token += blockDim.x) {
            if (!MASS) {
                float value;
                u32 key;
                if (survivor.at(token, value, key) && (key & mask) == prefix) {
                    const u32 bin = (key >> shifts[level]) & (bins - 1);
                    const u32 active = __activemask();
                    const u32 peers = __match_any_sync(active, bin);
                    if (__ffs(peers) - 1 == (int)(threadIdx.x & 31u))
                        atomicAdd(&histogram[bin], (u64)__popc(peers));
                }
            } else {
                u32 key;
                u64 w;
                if (survivor.at_mass(token, key, w) && (key & mask) == prefix)
                    atomicAdd(&histogram[(key >> shifts[level]) & (bins - 1)], w);
            }
        }
        __syncthreads();
        const u32 digit = descend(histogram, bins, target, above, lane_totals);
        if (digit == BINS)
            return false;
        prefix |= digit << shifts[level];
        mask |= (bins - 1) << shifts[level];
        __syncthreads();
    }
    key_out = prefix;
    return true;
}

extern "C" __global__ void shape_rows_select(SEISMIC_KERNEL_PARAMS) {
    const float *params = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_PARAMS));
    const element::Out<element::F32> logits = SEISMIC_OUT_LOGITS;
    __shared__ SelectScratch smem;
    __shared__ u64 lane_totals[32];
    __shared__ u64 reduce[32];
    __shared__ u32 chunk_counts[SELECT_THREADS];
    __shared__ u32 cand_count;
    const u64 row = blockIdx.x;
    u32 *scratch = row_scratch(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STATE), row);
    Selection *selection = reinterpret_cast<Selection *>(scratch + SEISMIC_TUNE_PARTS);
    if (parameter(params, row, 0) == 0.0f)
        return;
    const int top_k = (int)parameter(params, row, 1);
    const float top_p = parameter(params, row, 2);
    u32 maximum = 0u;
    for (u64 part = 0; part < SEISMIC_TUNE_PARTS; ++part)
        maximum = max(maximum, scratch[part]);
    if (maximum == UNSHAPED) {
        if (threadIdx.x == 0)
            selection->maximum = UNSHAPED;
        return;
    }
    Survivor survivor{logits.view(row * SEISMIC_LOGITS_STRIDE_0), SEISMIC_LOGITS_STRIDE_1, order::value(maximum), 0u, 0.0f};
    u32 top_k_key = 0u;
    u64 above = 0;
    bool compacted = false;
    u32 compacted_count = 0u;
    if (top_k > 0 && (u64)top_k < SEISMIC_DIM_V && maximum != 0u) {
        for (u32 bin = threadIdx.x; bin < BINS; bin += blockDim.x)
            smem.histogram[bin] = 0;
        __syncthreads();
        for (u64 token = threadIdx.x; token < SEISMIC_DIM_V; token += blockDim.x) {
            const float value = survivor.line.at(token * survivor.stride);
            if (finite(value)) {
                const u32 bin = order::finite_key(value) >> 21;
                const u32 active = __activemask();
                const u32 peers = __match_any_sync(active, bin);
                if (__ffs(peers) - 1 == (int)(threadIdx.x & 31u))
                    atomicAdd(&smem.histogram[bin], (u64)__popc(peers));
            }
        }
        __syncthreads();
        const u32 digit0 = descend(smem.histogram, BINS, (u64)top_k, above, lane_totals);
        if (digit0 != BINS) {
            const u64 in_bin0 = smem.histogram[digit0];
            __syncthreads();
            if (above + in_bin0 <= COMPACT_CAP) {
                if (threadIdx.x == 0)
                    cand_count = 0u;
                __syncthreads();
                for (u64 token = threadIdx.x; token < SEISMIC_DIM_V; token += blockDim.x) {
                    const float value = survivor.line.at(token * survivor.stride);
                    if (finite(value)) {
                        const u32 key = order::finite_key(value);
                        if ((key >> 21) >= digit0) {
                            const u32 idx = atomicAdd(&cand_count, 1u);
                            smem.compact.keys[idx] = key;
                            smem.compact.tokens[idx] = (u32)token;
                        }
                    }
                }
                __syncthreads();
                compacted_count = cand_count;
                bitonic_sort_compact(compacted_count, smem.compact.keys, smem.compact.tokens);
                top_k_key = smem.compact.keys[top_k - 1];
                compacted = true;
            } else {
                u32 key;
                if (select_key_from<false>(survivor, (u64)top_k, 1, digit0 << 21, 0x7FFu << 21, above, key,
                                           smem.histogram, lane_totals))
                    top_k_key = key;
            }
        }
    }
    survivor.top_k_key = top_k_key;
    survivor.min_p = parameter(params, row, 3);
    u32 mode = 0u, top_p_key = 0u;
    if (top_p < 1.0f && maximum != 0u) {
        if (compacted) {
            solve_top_p_sorted(compacted_count, survivor, top_p, smem.compact.keys, smem.compact.tokens, reduce,
                               selection, mode, top_p_key);
        } else {
            if (threadIdx.x == 0)
                cand_count = 0u;
            __syncthreads();
            u64 total = 0;
            for (u64 token = threadIdx.x; token < SEISMIC_DIM_V; token += blockDim.x) {
                u32 key;
                u64 w;
                if (survivor.at_mass(token, key, w)) {
                    total += w;
                    if (*(volatile u32 *)&cand_count <= COMPACT_CAP) {
                        const u32 idx = atomicAdd(&cand_count, 1u);
                        if (idx < COMPACT_CAP) {
                            smem.compact.keys[idx] = key;
                            smem.compact.tokens[idx] = (u32)token;
                        }
                    }
                }
            }
            for (int offset = 16; offset > 0; offset >>= 1)
                total += __shfl_xor_sync(0xFFFFFFFFu, total, offset);
            if (threadIdx.x % 32 == 0)
                reduce[threadIdx.x / 32] = total;
            __syncthreads();
            total = 0;
            for (u32 w = 0; w < blockDim.x / 32; ++w)
                total += reduce[w];
            const u32 count = cand_count;
            __syncthreads();
            const u64 target = top_p_target(top_p, total);
            if (target == 0) {
                mode = 2u;
            } else if (count <= COMPACT_CAP) {
                bitonic_sort_compact(count, smem.compact.keys, smem.compact.tokens);
                solve_top_p_sorted(count, survivor, top_p, smem.compact.keys, smem.compact.tokens, reduce, selection,
                                   mode, top_p_key);
            } else {
                u32 key;
                above = 0;
                select_key_from<true>(survivor, target, 0, 0u, 0u, above, key, smem.histogram, lane_totals);
                // Rank among equal keys: the tied token whose inclusive mass
                // first reaches the target, counted in ascending token order.
                const u64 each = survivor.mass(order::value(key));
                // above < target <= above + ties * each, so each > 0 and rank < ties.
                const u64 rank = (target - above + each - 1) / each - 1;
                const u64 chunk = (SEISMIC_DIM_V + blockDim.x - 1) / blockDim.x;
                const u64 first = threadIdx.x * chunk;
                u32 tie_count = 0;
                for (u64 token = first; token < first + chunk && token < SEISMIC_DIM_V; ++token) {
                    float value;
                    u32 candidate;
                    tie_count += survivor.at(token, value, candidate) && candidate == key ? 1u : 0u;
                }
                chunk_counts[threadIdx.x] = tie_count;
                __syncthreads();
                u64 before = 0;
                for (u32 t = 0; t < threadIdx.x; ++t)
                    before += chunk_counts[t];
                if (rank >= before && rank < before + tie_count) {
                    u64 seen = before;
                    for (u64 token = first; token < first + chunk && token < SEISMIC_DIM_V; ++token) {
                        float value;
                        u32 candidate;
                        if (survivor.at(token, value, candidate) && candidate == key) {
                            if (seen == rank)
                                selection->top_p_token = (u32)token;
                            ++seen;
                        }
                    }
                }
                mode = 1u;
                top_p_key = key;
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        selection->maximum = maximum;
        selection->top_k_key = top_k_key;
        selection->top_p_key = top_p_key;
        selection->top_p_mode = mode;
    }
}

extern "C" __global__ void shape_rows_apply(SEISMIC_KERNEL_PARAMS) {
    const float *params = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_PARAMS));
    const element::Out<element::F32> logits = SEISMIC_OUT_LOGITS;
    const u64 row = blockIdx.y;
    const u64 part = blockIdx.x;
    if (parameter(params, row, 0) == 0.0f)
        return;
    const Selection selection =
        *reinterpret_cast<const Selection *>(row_scratch(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STATE), row) + SEISMIC_TUNE_PARTS);
    if (selection.maximum == UNSHAPED)
        return;
    const float maximum = order::value(selection.maximum);
    const float min_p = parameter(params, row, 3);
    for (u64 token = slice_begin(part) + threadIdx.x; token < slice_end(part); token += blockDim.x) {
        const u64 slot = row * SEISMIC_LOGITS_STRIDE_0 + token * SEISMIC_LOGITS_STRIDE_1;
        const float value = logits.at(slot);
        if (!finite(value))
            continue;
        const u32 key = order::finite_key(value);
        const bool removed =
            key < selection.top_k_key || (min_p > 0.0f && expf(value - maximum) < min_p) ||
            selection.top_p_mode == 2u ||
            (selection.top_p_mode == 1u &&
             (key < selection.top_p_key || (key == selection.top_p_key && (u32)token > selection.top_p_token)));
        if (removed)
            logits.put(slot, -INF);
    }
}
