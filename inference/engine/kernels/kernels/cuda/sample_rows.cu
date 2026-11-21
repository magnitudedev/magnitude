// sample_rows: two-launch partition/merge selection.
//   sample_rows_partition (PARTS x M blocks): each block scans its contiguous
//     vocabulary slice, applies the mask of a constrained row and the
//     Philox4x32-10 Gumbel draw
//     (counter = token, draws[3..5]; key = draws[1..2]) and keeps its winner
//     (score bits, token, nonfinite flag), ties to the lower token.
//   sample_rows_merge (M warps): one warp per row merges the partitions,
//     preserving the winning score bits, and publishes (token, status):
//     0 success, 1 empty, 2 nonfinite source.
// The token counter is the vocabulary coordinate, so partitioning never
// changes a draw and the result is independent of PARTS and WIDTH.

typedef unsigned int u32;
typedef unsigned long long u64;

#define NO_TOKEN 0xFFFFFFFFu

#include "lib/core/gumbel.cuh"

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

template <bool NOISY>
__device__ __forceinline__ void consider_token(
    float value, u32 token, bool masked,
    const u32 *bits, u64 mask_stride,
    const u32 *draw, u64 draws_stride,
    Best &best) {
    const float pos_inf = __int_as_float(0x7F800000);
    best.bad |= (value != value || value == pos_inf) ? 1u : 0u;
    if (!(value > -pos_inf && value < pos_inf))
        return;
    if (masked && ((bits[(token >> 5) * mask_stride] >> (token & 31u)) & 1u) == 0u)
        return;
    const float score = NOISY ? (value + gumbel::noise(token, draw, draws_stride)) : value;
    if (better(score, token, best.score, best.token)) {
        best.score = score;
        best.token = token;
    }
}

template <bool NOISY>
__device__ __forceinline__ Best scan_partition(
    const float *line, u64 logits_stride,
    u64 begin, u64 end, bool masked,
    const u32 *bits, u64 mask_stride,
    const u32 *draw, u64 draws_stride) {
    Best best{-__int_as_float(0x7F800000), NO_TOKEN, 0u};
    if (logits_stride == 1u && (reinterpret_cast<u64>(line) & 3u) == 0u) {
        const u64 begin_addr = reinterpret_cast<u64>(line + begin);
        const u64 pad = ((16u - (begin_addr & 15u)) & 15u) >> 2;
        const u64 vec_begin = begin + pad < end ? begin + pad : end;
        const u64 vec_end = vec_begin + ((end - vec_begin) & ~3ull);
        if ((u64)threadIdx.x < vec_begin - begin) {
            const u64 token = begin + threadIdx.x;
            consider_token<NOISY>(line[token], (u32)token, masked, bits, mask_stride, draw, draws_stride, best);
        }
        for (u64 token = vec_begin + (u64)threadIdx.x * 4u; token < vec_end; token += (u64)blockDim.x * 4u) {
            const float4 values = *reinterpret_cast<const float4 *>(line + token);
            consider_token<NOISY>(values.x, (u32)token, masked, bits, mask_stride, draw, draws_stride, best);
            consider_token<NOISY>(values.y, (u32)(token + 1u), masked, bits, mask_stride, draw, draws_stride, best);
            consider_token<NOISY>(values.z, (u32)(token + 2u), masked, bits, mask_stride, draw, draws_stride, best);
            consider_token<NOISY>(values.w, (u32)(token + 3u), masked, bits, mask_stride, draw, draws_stride, best);
        }
        if ((u64)threadIdx.x < end - vec_end) {
            const u64 token = vec_end + threadIdx.x;
            consider_token<NOISY>(line[token], (u32)token, masked, bits, mask_stride, draw, draws_stride, best);
        }
    } else {
        for (u64 token = begin + threadIdx.x; token < end; token += blockDim.x) {
            consider_token<NOISY>(
                line[token * logits_stride], (u32)token, masked, bits, mask_stride, draw, draws_stride, best);
        }
    }
    return best;
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
    const bool noisy = draw[0] == 1u;
    Best best = noisy
        ? scan_partition<true>(
            line, SEISMIC_LOGITS_STRIDE_1, begin, end, masked,
            bits, SEISMIC_MASK_STRIDE_1, draw, SEISMIC_DRAWS_STRIDE_1)
        : scan_partition<false>(
            line, SEISMIC_LOGITS_STRIDE_1, begin, end, masked,
            bits, SEISMIC_MASK_STRIDE_1, draw, SEISMIC_DRAWS_STRIDE_1);
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
    const u64 row = (u64)blockIdx.x;
    const u32 lane = threadIdx.x;
    Best best{-__int_as_float(0x7F800000), NO_TOKEN, 0u};
    for (u64 part = lane; part < SEISMIC_TUNE_PARTS; part += 32u) {
        const u32 *in = partials + (row * SEISMIC_TUNE_PARTS + part) * 3;
        const float score = __uint_as_float(in[0]);
        const u32 token = in[1];
        best.bad |= in[2];
        if (better(score, token, best.score, best.token)) {
            best.score = score;
            best.token = token;
        }
    }
    best = warp_best(best);
    if (lane == 0u) {
        const int status = best.bad != 0u ? 2 : (best.token == NO_TOKEN ? 1 : 0);
        result[row * SEISMIC_RESULT_STRIDE_0] = status == 0 ? (int)best.token : -1;
        result[row * SEISMIC_RESULT_STRIDE_0 + SEISMIC_RESULT_STRIDE_1] = status;
    }
}
