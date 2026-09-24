// shape_rows: penalties and temperature, then top-k (ties at the cutoff kept),
// min-p against the row maximum, and top-p over the survivors in
// (value descending, token ascending) order.
//   shape_rows_prepare (PARTS x Sx): penalized/tempered values into `out` and the
//     partition maxima.
//   shape_rows_select (Sx): the cutoffs. Top-k is a count-weighted radix select of
//     the k-th largest key; top-p a mass-weighted radix select of the first
//     token whose inclusive mass reaches ceil(top_p * total), then its rank
//     among equal keys by token index. Masses are exp(v - max) in 2^-32
//     fixed point, so every sum is order-independent and exact.
//   shape_rows_apply (PARTS x Sx): writes -inf over every removed token.
// A zero temperature leaves the penalized values unshaped.

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

__device__ __forceinline__ u32 ordered_key(float value) {
    // Numeric equality treats both zero encodings as one tie.
    const u32 bits = __float_as_uint(value == 0.0f ? 0.0f : value);
    return (bits & 0x80000000u) != 0 ? ~bits : bits ^ 0x80000000u;
}
__device__ __forceinline__ float key_value(u32 key) {
    return __uint_as_float((key & 0x80000000u) != 0 ? key ^ 0x80000000u : ~key);
}
__device__ __forceinline__ bool finite(float value) { return value > -INF && value < INF; }

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
    const float *logits = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_LOGITS));
    const float *params = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_PARAMS));
    const int *history = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_HISTORY));
    float *out = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_OUT));
    extern __shared__ int recent[];
    __shared__ u32 warp_maximum[32];
    const u64 row = blockIdx.y;
    const u64 part = blockIdx.x;
    for (u64 h = threadIdx.x; h < SEISMIC_DIM_HN; h += blockDim.x)
        recent[h] = history[row * SEISMIC_HISTORY_STRIDE_0 + h * SEISMIC_HISTORY_STRIDE_1];
    __syncthreads();
    const float temperature = parameter(params, row, 0);
    const float repetition = parameter(params, row, 4);
    const float presence = parameter(params, row, 5);
    const float frequency = parameter(params, row, 6);
    u32 maximum = 0u;
    for (u64 token = slice_begin(part) + threadIdx.x; token < slice_end(part); token += blockDim.x) {
        int count = 0;
        for (u64 h = 0; h < SEISMIC_DIM_HN; ++h)
            count += recent[h] == (int)token ? 1 : 0;
        float value = logits[row * SEISMIC_LOGITS_STRIDE_0 + token * SEISMIC_LOGITS_STRIDE_1];
        if (count > 0) {
            value = value < 0.0f ? value * repetition : value / repetition;
            value = value - presence - frequency * (float)count;
        }
        if (temperature != 0.0f)
            value = value / temperature;
        out[row * SEISMIC_OUT_STRIDE_0 + token * SEISMIC_OUT_STRIDE_1] = value;
        if (finite(value))
            maximum = max(maximum, ordered_key(value));
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

struct Survivor {
    const float *line;
    u64 stride;
    float maximum;
    u32 top_k_key;
    float min_p;
    // The finite value at `token` when it survives top-k and min-p.
    __device__ __forceinline__ bool at(u64 token, float &value, u32 &key) const {
        value = line[token * stride];
        if (!finite(value))
            return false;
        key = ordered_key(value);
        return key >= top_k_key && !(min_p > 0.0f && expf(value - maximum) < min_p);
    }
    __device__ __forceinline__ u64 mass(float value) const {
        return (u64)(expf(value - maximum) * 4294967296.0f);
    }
};

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

// Radix select over the keys of survivors (weight 1 or mass): the key K with
// weight(key > K) < target <= weight(key >= K). Returns false when the total
// weight is below the target; `above` receives weight(key > K).
template <bool MASS>
__device__ bool select_key(const Survivor &survivor, u64 target, u32 &key_out, u64 &above, u64 *histogram,
                           u64 *lane_totals) {
    const int shifts[3] = {21, 10, 0};
    const u32 widths[3] = {11, 11, 10};
    u32 prefix = 0u;
    u32 mask = 0u;
    above = 0;
    for (int level = 0; level < 3; ++level) {
        const u32 bins = 1u << widths[level];
        for (u32 bin = threadIdx.x; bin < bins; bin += blockDim.x)
            histogram[bin] = 0;
        __syncthreads();
        for (u64 token = threadIdx.x; token < SEISMIC_DIM_V; token += blockDim.x) {
            float value;
            u32 key;
            if (survivor.at(token, value, key) && (key & mask) == prefix)
                atomicAdd(&histogram[(key >> shifts[level]) & (bins - 1)], MASS ? survivor.mass(value) : 1ull);
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
    const float *out = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_OUT));
    __shared__ u64 histogram[BINS];
    __shared__ u64 lane_totals[32];
    __shared__ u64 reduce[32];
    __shared__ u32 chunk_counts[SELECT_THREADS];
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
    Survivor survivor{out + row * SEISMIC_OUT_STRIDE_0, SEISMIC_OUT_STRIDE_1, key_value(maximum), 0u, 0.0f};
    u32 top_k_key = 0u;
    u64 above;
    if (top_k > 0 && (u64)top_k < SEISMIC_DIM_V && maximum != 0u) {
        u32 key;
        if (select_key<false>(survivor, (u64)top_k, key, above, histogram, lane_totals))
            top_k_key = key;
    }
    survivor.top_k_key = top_k_key;
    survivor.min_p = parameter(params, row, 3);
    u32 mode = 0u, top_p_key = 0u, top_p_token = 0u;
    if (top_p < 1.0f && maximum != 0u) {
        u64 total = 0;
        for (u64 token = threadIdx.x; token < SEISMIC_DIM_V; token += blockDim.x) {
            float value;
            u32 key;
            if (survivor.at(token, value, key))
                total += survivor.mass(value);
        }
        for (int offset = 16; offset > 0; offset >>= 1)
            total += __shfl_xor_sync(0xFFFFFFFFu, total, offset);
        if (threadIdx.x % 32 == 0)
            reduce[threadIdx.x / 32] = total;
        __syncthreads();
        total = 0;
        for (u32 w = 0; w < blockDim.x / 32; ++w)
            total += reduce[w];
        const u64 target = (u64)ceil((double)top_p * (double)total);
        if (target == 0) {
            mode = 2u;
        } else {
            u32 key;
            select_key<true>(survivor, target, key, above, histogram, lane_totals);
            // Rank among equal keys: the tied token whose inclusive mass
            // first reaches the target, counted in ascending token order.
            const u64 each = survivor.mass(key_value(key));
            // above < target <= above + ties * each, so each > 0 and rank < ties.
            const u64 rank = (target - above + each - 1) / each - 1;
            const u64 chunk = (SEISMIC_DIM_V + blockDim.x - 1) / blockDim.x;
            const u64 first = threadIdx.x * chunk;
            u32 count = 0;
            for (u64 token = first; token < first + chunk && token < SEISMIC_DIM_V; ++token) {
                float value;
                u32 candidate;
                count += survivor.at(token, value, candidate) && candidate == key ? 1u : 0u;
            }
            chunk_counts[threadIdx.x] = count;
            __syncthreads();
            u64 before = 0;
            for (u32 t = 0; t < threadIdx.x; ++t)
                before += chunk_counts[t];
            if (rank >= before && rank < before + count) {
                u64 seen = before;
                for (u64 token = first; token < first + chunk && token < SEISMIC_DIM_V; ++token) {
                    float value;
                    u32 candidate;
                    if (survivor.at(token, value, candidate) && candidate == key) {
                        if (seen == rank) {
                            top_p_token = (u32)token;
                            selection->top_p_token = top_p_token;
                        }
                        ++seen;
                    }
                }
            }
            mode = 1u;
            top_p_key = key;
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
    float *out = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_OUT));
    const u64 row = blockIdx.y;
    const u64 part = blockIdx.x;
    if (parameter(params, row, 0) == 0.0f)
        return;
    const Selection selection =
        *reinterpret_cast<const Selection *>(row_scratch(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STATE), row) + SEISMIC_TUNE_PARTS);
    const float maximum = key_value(selection.maximum);
    const float min_p = parameter(params, row, 3);
    for (u64 token = slice_begin(part) + threadIdx.x; token < slice_end(part); token += blockDim.x) {
        float *slot = out + row * SEISMIC_OUT_STRIDE_0 + token * SEISMIC_OUT_STRIDE_1;
        const float value = *slot;
        if (!finite(value))
            continue;
        const u32 key = ordered_key(value);
        const bool removed =
            key < selection.top_k_key || (min_p > 0.0f && expf(value - maximum) < min_p) ||
            selection.top_p_mode == 2u ||
            (selection.top_p_mode == 1u &&
             (key < selection.top_p_key || (key == selection.top_p_key && (u32)token > selection.top_p_token)));
        if (removed)
            *slot = -INF;
    }
}
