// sample_rows: two-launch partition/merge selection.
//   sample_rows_partition (PARTS x M blocks): each block scans its contiguous
//     vocabulary slice, applies the mask of a constrained row and the
//     Philox4x32-10 Gumbel draw
//     (counter = token, draws[3..5]; key = draws[1..2]) and keeps its winner
//     (score bits, token, nonfinite flag), ties to the lower token.
//   sample_rows_merge (M threads): merges the partitions in ascending order,
//     preserving the winning score bits, and publishes (token, status):
//     0 success, 1 empty, 2 nonfinite source.
// The token counter is the vocabulary coordinate, so partitioning never
// changes a draw and the result is independent of PARTS and WIDTH.

typedef unsigned int u32;
typedef unsigned long long u64;

#define NO_TOKEN 0xFFFFFFFFu

__device__ __forceinline__ float gumbel_score(float value, u32 token, const u32 *draw, u64 stride) {
    if (draw[0] != 1u)
        return value;
    u32 c0 = token, c1 = draw[3 * stride], c2 = draw[4 * stride], c3 = draw[5 * stride];
    u32 k0 = draw[stride], k1 = draw[2 * stride];
#pragma unroll
    for (int round = 0; round < 10; ++round) {
        const u32 hi0 = __umulhi(3528531795u, c0), lo0 = 3528531795u * c0;
        const u32 hi1 = __umulhi(3449720151u, c2), lo1 = 3449720151u * c2;
        const u32 next0 = hi1 ^ c1 ^ k0;
        const u32 next2 = hi0 ^ c3 ^ k1;
        c0 = next0;
        c1 = lo1;
        c2 = next2;
        c3 = lo0;
        k0 += 2654435769u;
        k1 += 3144134277u;
    }
    const float uniform = ((float)(c0 >> 9) + 0.5f) * 0.00000011920928955078125f;
    return value - logf(-logf(uniform));
}

struct Best {
    float score;
    u32 token;
    u32 bad;
};

// `a` replaces `b` when it is a candidate with a higher score, or an equal
// score at a lower token.
__device__ __forceinline__ bool better(float score, u32 token, float best_score, u32 best_token) {
    return token != NO_TOKEN &&
           (best_token == NO_TOKEN || score > best_score || (score == best_score && token < best_token));
}

__device__ __forceinline__ Best warp_best(Best value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float score = __shfl_xor_sync(0xFFFFFFFFu, value.score, offset);
        const u32 token = __shfl_xor_sync(0xFFFFFFFFu, value.token, offset);
        const u32 bad = __shfl_xor_sync(0xFFFFFFFFu, value.bad, offset);
        if (better(score, token, value.score, value.token)) {
            value.score = score;
            value.token = token;
        }
        value.bad |= bad;
    }
    return value;
}

extern "C" __global__ void sample_rows_partition(SEISMIC_KERNEL_PARAMS) {
    const float *logits = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_LOGITS));
    const u32 *mask = reinterpret_cast<const u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_MASK));
    const u32 *draws = reinterpret_cast<const u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_DRAWS));
    u32 *partials = reinterpret_cast<u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS));
    __shared__ Best warps[32];
    const u64 row = blockIdx.y;
    const u64 part = blockIdx.x;
    const u64 V = SEISMIC_DIM_V;
    const u64 span = (V + SEISMIC_TUNE_PARTS - 1) / SEISMIC_TUNE_PARTS;
    const u64 begin = part * span < V ? part * span : V;
    const u64 end = begin + span < V ? begin + span : V;
    const float *line = logits + row * SEISMIC_LOGITS_STRIDE_0;
    const u32 *bits = mask + row * SEISMIC_MASK_STRIDE_0;
    const u32 *draw = draws + row * SEISMIC_DRAWS_STRIDE_0;
    // An unconstrained row admits every token; its mask row is never read.
    const bool masked =
        reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_CONSTRAINED))[row * SEISMIC_CONSTRAINED_STRIDE_0] != 0;
    Best best{-__int_as_float(0x7F800000), NO_TOKEN, 0u};
    for (u64 token = begin + threadIdx.x; token < end; token += blockDim.x) {
        const float value = line[token * SEISMIC_LOGITS_STRIDE_1];
        best.bad |= (value != value || value == __int_as_float(0x7F800000)) ? 1u : 0u;
        if (!(value > -__int_as_float(0x7F800000) && value < __int_as_float(0x7F800000)))
            continue;
        if (masked && ((bits[(token / 32) * SEISMIC_MASK_STRIDE_1] >> (token % 32)) & 1u) == 0u)
            continue;
        const float score = gumbel_score(value, (u32)token, draw, SEISMIC_DRAWS_STRIDE_1);
        if (better(score, (u32)token, best.score, best.token)) {
            best.score = score;
            best.token = (u32)token;
        }
    }
    best = warp_best(best);
    if (threadIdx.x % 32 == 0)
        warps[threadIdx.x / 32] = best;
    __syncthreads();
    if (threadIdx.x < 32) {
        best = threadIdx.x < blockDim.x / 32 ? warps[threadIdx.x] : Best{-__int_as_float(0x7F800000), NO_TOKEN, 0u};
        best = warp_best(best);
        if (threadIdx.x == 0) {
            u32 *out = partials + (row * SEISMIC_TUNE_PARTS + part) * 3;
            out[0] = __float_as_uint(best.score);
            out[1] = best.token;
            out[2] = best.bad;
        }
    }
}

extern "C" __global__ void sample_rows_merge(SEISMIC_KERNEL_PARAMS) {
    const u32 *partials = reinterpret_cast<const u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS));
    int *result = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_RESULT));
    const u64 row = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= SEISMIC_DIM_M)
        return;
    u32 best_bits = 0u;
    u32 best_token = NO_TOKEN;
    u32 bad = 0u;
    for (u64 part = 0; part < SEISMIC_TUNE_PARTS; ++part) {
        const u32 *in = partials + (row * SEISMIC_TUNE_PARTS + part) * 3;
        bad |= in[2];
        if (better(__uint_as_float(in[0]), in[1], __uint_as_float(best_bits), best_token)) {
            best_bits = in[0];
            best_token = in[1];
        }
    }
    const int status = bad != 0u ? 2 : (best_token == NO_TOKEN ? 1 : 0);
    result[row * SEISMIC_RESULT_STRIDE_0] = status == 0 ? (int)best_token : -1;
    result[row * SEISMIC_RESULT_STRIDE_0 + SEISMIC_RESULT_STRIDE_1] = status;
}
