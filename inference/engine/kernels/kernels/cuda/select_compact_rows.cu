// select_compact_rows: fused single-launch shortlist selection + softmax
// confidence directly over `[M, S]` compact scores and canonical `token_rows`,
// avoiding scattering into `[M, V]` and scanning the full vocabulary.
// One 64-thread (2-warp) block per row: exact parity with `place_logits_rows`
// -> `sample_rows` -> `token_confidence`.
#include <seismic/element.cuh>
#include "lib/core/gumbel.cuh"
#include "lib/core/reduce.cuh"

typedef unsigned int u32;
typedef unsigned long long u64;

#define NO_TOKEN 0xFFFFFFFFu

namespace {

struct CompactBest {
    float score;
    float logit;
    u32 token;
    u32 bad;
};

__device__ __forceinline__ bool better(float score, u32 token, float best_score, u32 best_token) {
    return token != NO_TOKEN &&
           (best_token == NO_TOKEN || score > best_score || (score == best_score && token < best_token));
}

__device__ __forceinline__ CompactBest warp_best(CompactBest value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float score = __shfl_xor_sync(0xFFFFFFFFu, value.score, offset);
        const float logit = __shfl_xor_sync(0xFFFFFFFFu, value.logit, offset);
        const u32 token = __shfl_xor_sync(0xFFFFFFFFu, value.token, offset);
        const u32 bad = __shfl_xor_sync(0xFFFFFFFFu, value.bad, offset);
        if (better(score, token, value.score, value.token)) {
            value.score = score;
            value.logit = logit;
            value.token = token;
        }
        value.bad |= bad;
    }
    return value;
}

} // namespace

extern "C" __global__ void select_compact_rows(SEISMIC_KERNEL_PARAMS) {
    __shared__ CompactBest warp_winners[2];
    __shared__ reduce::OnlineSoftmax warp_softmax[2];

    const float *scores = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCORES));
    const int *token_rows = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_TOKEN_ROWS));
    const u32 *mask = reinterpret_cast<const u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_MASK));
    const int *constrained = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_CONSTRAINED));
    const u32 *draws = reinterpret_cast<const u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_DRAWS));
    int *result = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_RESULT));
    const auto confidence = SEISMIC_OUT_CONFIDENCE;

    const u64 row = (u64)blockIdx.x;
    const u64 W = SEISMIC_DIM_W;
    const u64 S = SEISMIC_DIM_S;
    const float *score_row = scores + row * SEISMIC_SCORES_STRIDE_0;
    const int *tok_row = token_rows + row * SEISMIC_TOKEN_ROWS_STRIDE_0;
    const u32 *bits = mask + row * SEISMIC_MASK_STRIDE_0;
    const u32 *draw = draws + row * SEISMIC_DRAWS_STRIDE_0;
    const bool masked = constrained[row * SEISMIC_CONSTRAINED_STRIDE_0] != 0;
    const bool noisy = draw[0] == 1u;
    const float pos_inf = __int_as_float(0x7F800000);

    CompactBest best{-pos_inf, -pos_inf, NO_TOKEN, 0u};
    reduce::OnlineSoftmax sm = reduce::OnlineSoftmax::empty();

    for (u64 col = (u64)threadIdx.x; col < S; col += (u64)blockDim.x) {
        const int raw_token = tok_row[col * SEISMIC_TOKEN_ROWS_STRIDE_1];
        if (raw_token < 0 || ((u64)raw_token >> 5) >= W)
            continue;
        const u32 token = (u32)raw_token;
        const float value = score_row[col * SEISMIC_SCORES_STRIDE_1];
        best.bad |= (value != value || value == pos_inf) ? 1u : 0u;
        sm.add(value);
        if (!(value > -pos_inf && value < pos_inf))
            continue;
        if (masked && ((bits[(token >> 5) * SEISMIC_MASK_STRIDE_1] >> (token & 31u)) & 1u) == 0u)
            continue;
        const float candidate = noisy ? (value + gumbel::noise(token, draw, SEISMIC_DRAWS_STRIDE_1)) : value;
        if (better(candidate, token, best.score, best.token)) {
            best.score = candidate;
            best.logit = value;
            best.token = token;
        }
    }

    best = warp_best(best);
    sm = reduce::warp_online_softmax(sm);

    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31u;
    if (lane == 0u) {
        warp_winners[warp] = best;
        warp_softmax[warp] = sm;
    }
    __syncthreads();

    if (threadIdx.x == 0u) {
        CompactBest final_best = warp_winners[0];
        const CompactBest other = warp_winners[1];
        if (better(other.score, other.token, final_best.score, final_best.token)) {
            final_best.score = other.score;
            final_best.logit = other.logit;
            final_best.token = other.token;
        }
        final_best.bad |= other.bad;

        const reduce::OnlineSoftmax final_sm = reduce::OnlineSoftmax::merge(warp_softmax[0], warp_softmax[1]);
        const int status = final_best.bad != 0u ? 2 : (final_best.token == NO_TOKEN ? 1 : 0);
        result[row * SEISMIC_RESULT_STRIDE_0] = status == 0 ? (int)final_best.token : -1;
        result[row * SEISMIC_RESULT_STRIDE_0 + SEISMIC_RESULT_STRIDE_1] = status;
        confidence.put(
            row * SEISMIC_CONFIDENCE_STRIDE_0,
            status == 0 && final_sm.sum != 0.0f ? expf(final_best.logit - final_sm.max) / final_sm.sum : 0.0f);
    }
}
