// General expert selection for M rows; the CUDA form of
// `metal/routed_select.metal`, in four launches: `routed_select_stage`
// (M > 8, one block per row) publishes the normalized rows and the router
// input rows once; `routed_select_gemv` (M <= 8, one warp per logit column,
// forming the router input on the fly and publishing the normalized rows) or
// `routed_select_gemm` (32 rows x 32 columns x one hidden share of SPLIT per
// block over shared tiles) forms the [M, E] logits as SPLIT partial planes;
// `routed_select_select` (one warp per row) sums the planes, forms the
// scores, ranks them with the selection bias (single-pass top-K,
// `lib/routed/select.cuh`) and publishes the normalized, scaled weights.
// Every logit accumulates over the hidden axis in the same order for every
// expert, so experts with equal router rows tie exactly.

#include "lib/core/activation.cuh"
#include "lib/core/reduce.cuh"
#include "lib/routed/select.cuh"
#if defined(SEISMIC_ROUTER_KIND_PACKED)
#define KERNEL_W0 SEISMIC_ROUTER
#include <seismic/packets.cuh>
#endif

namespace {

using element::Act;
typedef ELEMENT_OF(SEISMIC_NORM) Norm;
typedef ELEMENT_OF(SEISMIC_ROUTER_NORM) RouterNorm;
#if !defined(SEISMIC_ROUTER_KIND_PACKED)
typedef ELEMENT_OF(SEISMIC_ROUTER) Router;
#endif

} // namespace

constexpr unsigned SELECT_THREADS = 256;
constexpr unsigned SELECT_WARPS = SELECT_THREADS / 32;

extern "C" __global__ void routed_select_stage(SEISMIC_KERNEL_PARAMS) {
    const float *residual = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL));
    const unsigned char *norm = SEISMIC_PTR(SEISMIC_BUFFER_NORM);
    const unsigned char *router_norm = SEISMIC_PTR(SEISMIC_BUFFER_ROUTER_NORM);
    unsigned char *normalized = SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER);
    unsigned char *router_rows = SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_ROUTER_ROWS);
    __shared__ float partials[SELECT_WARPS];
    const unsigned thread = threadIdx.x;
    const unsigned long long row = blockIdx.x;
    const float eps = __uint_as_float(static_cast<unsigned>(SEISMIC_PARAM_EPSILON));
    float squares = 0.0f;
    for (unsigned source = thread; source < SEISMIC_DIM_H; source += SELECT_THREADS) {
        const float value = residual[row * SEISMIC_RESIDUAL_STRIDE_0 + source * SEISMIC_RESIDUAL_STRIDE_1];
        squares = seismic_fma_rn(value, value, squares);
    }
    const float total = reduce::group_sum(squares, partials);
    const float inverse = rsqrtf(seismic_add_rn(total / static_cast<float>(SEISMIC_DIM_H), eps));
    for (unsigned source = thread; source < SEISMIC_DIM_H; source += SELECT_THREADS) {
        const float value = residual[row * SEISMIC_RESIDUAL_STRIDE_0 + source * SEISMIC_RESIDUAL_STRIDE_1];
        const float scaled = seismic_mul_rn(value, inverse);
        element::put<Act>(normalized, row * SEISMIC_RESULT_0_STRIDE_0 + source * SEISMIC_RESULT_0_STRIDE_1,
            Act::round(seismic_mul_rn(scaled, element::at<Norm>(norm, source * SEISMIC_NORM_STRIDE_0))));
        element::put<Act>(router_rows, row * SEISMIC_DIM_H + source,
            Act::round(seismic_mul_rn(scaled, element::at<RouterNorm>(router_norm, source * SEISMIC_ROUTER_NORM_STRIDE_0))));
    }
}

// M <= 8, one launch before the selection: the block computes its rows' RMS
// inverses (in the stage launch's order), then each warp forms one logit
// column, forming each router input element as it reads it; block 0
// publishes the normalized rows.
extern "C" __global__ void routed_select_gemv(SEISMIC_KERNEL_PARAMS) {
    const float *residual = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL));
    const unsigned char *norm = SEISMIC_PTR(SEISMIC_BUFFER_NORM);
    const unsigned char *router_norm = SEISMIC_PTR(SEISMIC_BUFFER_ROUTER_NORM);
    const unsigned char *router = SEISMIC_PTR(SEISMIC_BUFFER_ROUTER);
    unsigned char *normalized = SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER);
    float *logits = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_LOGITS));
    __shared__ float parts[8][SELECT_WARPS];
    __shared__ float inverses[8];
    const unsigned thread = threadIdx.x, lane = thread % 32, warp = thread / 32;
    const unsigned rows = static_cast<unsigned>(SEISMIC_DIM_M);
    const float eps = __uint_as_float(static_cast<unsigned>(SEISMIC_PARAM_EPSILON));
    for (unsigned item = warp; item < rows * SELECT_WARPS; item += SEISMIC_TUNE_SIMDGROUPS) {
        const unsigned row = item / SELECT_WARPS, part = item % SELECT_WARPS;
        float squares = 0.0f;
        for (unsigned source = part * 32 + lane; source < SEISMIC_DIM_H; source += SELECT_THREADS) {
            const float value = residual[row * SEISMIC_RESIDUAL_STRIDE_0 + source * SEISMIC_RESIDUAL_STRIDE_1];
            squares = seismic_fma_rn(value, value, squares);
        }
        squares = seismic_warp_sum_f32(squares);
        if (lane == 0)
            parts[row][part] = squares;
    }
    __syncthreads();
    if (thread < rows) {
        float total = 0.0f;
        for (unsigned part = 0; part < SELECT_WARPS; ++part)
            total = seismic_add_rn(total, parts[thread][part]);
        inverses[thread] = rsqrtf(seismic_add_rn(total / static_cast<float>(SEISMIC_DIM_H), eps));
    }
    __syncthreads();
    if (blockIdx.x == 0) {
        for (unsigned item = thread; item < rows * static_cast<unsigned>(SEISMIC_DIM_H); item += 32 * SEISMIC_TUNE_SIMDGROUPS) {
            const unsigned long long row = item / SEISMIC_DIM_H, source = item % SEISMIC_DIM_H;
            const float value = residual[row * SEISMIC_RESIDUAL_STRIDE_0 + source * SEISMIC_RESIDUAL_STRIDE_1];
            element::put<Act>(normalized, row * SEISMIC_RESULT_0_STRIDE_0 + source * SEISMIC_RESULT_0_STRIDE_1,
                Act::round(seismic_mul_rn(seismic_mul_rn(value, inverses[row]),
                    element::at<Norm>(norm, source * SEISMIC_NORM_STRIDE_0))));
        }
    }
    const unsigned long long expert = static_cast<unsigned long long>(blockIdx.x) * SEISMIC_TUNE_SIMDGROUPS + warp;
    if (expert >= SEISMIC_DIM_E)
        return;
    float sums[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#if defined(SEISMIC_ROUTER_KIND_PACKED)
    const auto router_rows = KERNEL_W0_AT(router);
    // One lane decodes a 16-value register chunk of an mma16 router row.
    for (unsigned item = lane; item < SEISMIC_DIM_H / 64 * 4; item += 32) {
        const unsigned long long kb = item / 4;
        const unsigned t = item % 4;
        const unsigned offsets[4] = {2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9};
        float values[16];
        packets::row_values16(router_rows, expert, kb, t, values);
#pragma unroll
        for (unsigned s = 0; s < 4; ++s)
#pragma unroll
            for (unsigned j = 0; j < 4; ++j) {
                const unsigned long long source = 64 * kb + 16 * s + offsets[j];
                const float scale = element::at<RouterNorm>(router_norm, source * SEISMIC_ROUTER_NORM_STRIDE_0);
#pragma unroll
                for (unsigned row = 0; row < 8; ++row)
                    if (row < rows) {
                        const float value = residual[row * SEISMIC_RESIDUAL_STRIDE_0 + source * SEISMIC_RESIDUAL_STRIDE_1];
                        sums[row] = seismic_fma_rn(Act::round(seismic_mul_rn(seismic_mul_rn(value, inverses[row]),
                            scale)), values[4 * s + j], sums[row]);
                    }
            }
    }
#else
    // Dense lane l owns sources 4l + 128 i .. + 3.
    for (unsigned base = 4 * lane; base < SEISMIC_DIM_H; base += 128) {
        float weight[4], scale[4];
#pragma unroll
        for (unsigned j = 0; j < 4; ++j) {
            weight[j] = element::at<Router>(router, expert * SEISMIC_ROUTER_STRIDE_0 + (base + j) * SEISMIC_ROUTER_STRIDE_1);
            scale[j] = element::at<RouterNorm>(router_norm, (base + j) * SEISMIC_ROUTER_NORM_STRIDE_0);
        }
#pragma unroll
        for (unsigned row = 0; row < 8; ++row) {
            if (row < rows) {
#pragma unroll
                for (unsigned j = 0; j < 4; ++j) {
                    const float value = residual[row * SEISMIC_RESIDUAL_STRIDE_0 + (base + j) * SEISMIC_RESIDUAL_STRIDE_1];
                    sums[row] = seismic_fma_rn(Act::round(seismic_mul_rn(seismic_mul_rn(value, inverses[row]),
                        scale[j])), weight[j], sums[row]);
                }
            }
        }
    }
#endif
#pragma unroll
    for (unsigned row = 0; row < 8; ++row) {
        const float sum = seismic_warp_sum_f32(sums[row]);
        if (row < rows && lane == 0)
            logits[row * SEISMIC_DIM_E + expert] = sum;
    }
}

// Rows and logit columns per GEMM block, and hidden coordinates per stage.
constexpr unsigned SELECT_TILE = 32;
constexpr unsigned SELECT_DEPTH = 128;

extern "C" __global__ void routed_select_gemm(SEISMIC_KERNEL_PARAMS) {
    const unsigned char *router = SEISMIC_PTR(SEISMIC_BUFFER_ROUTER);
    const unsigned char *router_input = SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_ROUTER_ROWS);
    float *logits = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_LOGITS));
#if defined(SEISMIC_ROUTER_KIND_PACKED)
    const auto router_rows = KERNEL_W0_AT(router);
#endif
    __shared__ float rows_tile[SELECT_TILE][SELECT_DEPTH + 1];
    __shared__ float experts_tile[SELECT_TILE][SELECT_DEPTH + 1];
    const unsigned thread = threadIdx.x;
    const unsigned long long expert0 = static_cast<unsigned long long>(blockIdx.x) * SELECT_TILE;
    const unsigned long long row0 = static_cast<unsigned long long>(blockIdx.y) * SELECT_TILE;
    // Block z forms the partial logits of hidden share z of SPLIT into plane z.
    const unsigned long long part = blockIdx.z;
    const unsigned long long kbegin = SEISMIC_DIM_H / 32 * part / SEISMIC_TUNE_SPLIT * 32;
    const unsigned long long kend = SEISMIC_DIM_H / 32 * (part + 1) / SEISMIC_TUNE_SPLIT * 32;
    logits += part * SEISMIC_DIM_M * SEISMIC_DIM_E;
    const unsigned r = thread / 8, e0 = 4 * (thread % 8);
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (unsigned long long k0 = kbegin; k0 < kend; k0 += SELECT_DEPTH) {
        const unsigned depth = static_cast<unsigned>(min(static_cast<unsigned long long>(SELECT_DEPTH), kend - k0));
        __syncthreads();
        for (unsigned item = thread; item < SELECT_TILE * SELECT_DEPTH; item += SELECT_THREADS) {
            const unsigned i = item / SELECT_DEPTH, k = item % SELECT_DEPTH;
            if (k >= depth)
                continue;
            const unsigned long long row = row0 + i, expert = expert0 + i;
            rows_tile[i][k] = row < SEISMIC_DIM_M ? element::at<Act>(router_input, row * SEISMIC_DIM_H + k0 + k) : 0.0f;
#if !defined(SEISMIC_ROUTER_KIND_PACKED)
            experts_tile[i][k] = expert < SEISMIC_DIM_E
                ? element::at<Router>(router, expert * SEISMIC_ROUTER_STRIDE_0 + (k0 + k) * SEISMIC_ROUTER_STRIDE_1)
                : 0.0f;
#endif
        }
#if defined(SEISMIC_ROUTER_KIND_PACKED)
        const unsigned long long first_kb = k0 / 64;
        const unsigned blocks = static_cast<unsigned>((k0 + depth + 63) / 64 - first_kb);
        for (unsigned item = thread; item < SELECT_TILE * blocks * 4; item += SELECT_THREADS) {
            const unsigned i = item / (blocks * 4);
            const unsigned piece = item % (blocks * 4);
            const unsigned long long expert = expert0 + i;
            const unsigned long long kb = first_kb + piece / 4;
            const unsigned t = piece % 4;
            const unsigned offsets[4] = {2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9};
            float values[16];
            if (expert < SEISMIC_DIM_E)
                packets::row_values16(router_rows, expert, kb, t, values);
#pragma unroll
            for (unsigned s = 0; s < 4; ++s)
#pragma unroll
                for (unsigned j = 0; j < 4; ++j) {
                    const unsigned long long source = 64 * kb + 16 * s + offsets[j];
                    if (source < k0 || source >= k0 + depth) continue;
                    experts_tile[i][source - k0] = expert < SEISMIC_DIM_E ? values[4 * s + j] : 0.0f;
                }
        }
#endif
        __syncthreads();
#pragma unroll 8
        for (unsigned k = 0; k < depth; ++k) {
            const float x = rows_tile[r][k];
#pragma unroll
            for (unsigned j = 0; j < 4; ++j)
                acc[j] = seismic_fma_rn(x, experts_tile[e0 + j][k], acc[j]);
        }
    }
    const unsigned long long row = row0 + r;
#pragma unroll
    for (unsigned j = 0; j < 4; ++j) {
        const unsigned long long expert = expert0 + e0 + j;
        if (row < SEISMIC_DIM_M && expert < SEISMIC_DIM_E)
            logits[row * SEISMIC_DIM_E + expert] = acc[j];
    }
}

extern "C" __global__ void routed_select_select(SEISMIC_KERNEL_PARAMS) {
    const float *bias = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_BIAS));
    const float *expert_scale = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_SCALE));
    int *routes = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES));
    float *weights = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_WEIGHTS));
    const float *logits = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_LOGITS));
    constexpr unsigned P = SEISMIC_DIM_E / 32;
    const unsigned lane = threadIdx.x % 32;
    const unsigned long long row = static_cast<unsigned long long>(blockIdx.x) * SELECT_WARPS + threadIdx.x / 32;
    if (row >= SEISMIC_DIM_M)
        return;
    // Logit column e: the sum of its partial planes in plane order (the GEMV
    // forms plane 0 whole).
    const unsigned long long parts = SEISMIC_DIM_M > 8 ? SEISMIC_TUNE_SPLIT : 1;
    const unsigned long long plane = SEISMIC_DIM_M * SEISMIC_DIM_E;
    float score[P], ranked[P];
    int expert[P];
#pragma unroll
    for (unsigned i = 0; i < P; ++i) {
        expert[i] = static_cast<int>(lane + 32 * i);
        float sum = logits[row * SEISMIC_DIM_E + lane + 32 * i];
        for (unsigned long long p = 1; p < parts; ++p)
            sum = seismic_add_rn(sum, logits[p * plane + row * SEISMIC_DIM_E + lane + 32 * i]);
        score[i] = sum;
    }
    select::scores<P>(static_cast<int>(SEISMIC_PARAM_SCORE), score);
#pragma unroll
    for (unsigned i = 0; i < P; ++i)
        ranked[i] = seismic_add_rn(score[i], bias[static_cast<unsigned long long>(expert[i]) * SEISMIC_BIAS_STRIDE_0]);
    select::sort<P>(ranked, score, expert);

    float chosen[SEISMIC_DIM_K];
    int winners[SEISMIC_DIM_K];
    for (unsigned rank = 0; rank < SEISMIC_DIM_K; ++rank) {
        float best;
        const int winner = reduce::argmax<reduce::HigherIndex>(ranked[0], expert[0], best);
        // The winner's unbiased score, from its lane.
        const float winner_score = __shfl_sync(0xffffffffu, score[0], static_cast<unsigned>(winner) % 32u);
        if (expert[0] == winner)
            select::pop<P>(ranked, score, expert);
        chosen[SEISMIC_DIM_K - 1 - rank] = winner_score;
        winners[SEISMIC_DIM_K - 1 - rank] = winner;
    }
    // The slot-order sum: slot s holds rank K - 1 - s.
    float denominator = 0.0f;
    for (unsigned slot = 0; slot < SEISMIC_DIM_K; ++slot)
        denominator = seismic_add_rn(denominator, chosen[slot]);
    if (lane == 0) {
        const int normalization = static_cast<int>(SEISMIC_PARAM_NORMALIZATION);
        const float scale = __uint_as_float(static_cast<unsigned>(SEISMIC_PARAM_SCALE));
        const float epsilon = __uint_as_float(static_cast<unsigned>(SEISMIC_PARAM_NORMALIZATION_EPSILON));
        for (unsigned slot = 0; slot < SEISMIC_DIM_K; ++slot) {
            const float weight = select::normalized(normalization, chosen[slot], denominator, epsilon);
            routes[row * SEISMIC_ROUTES_STRIDE_0 + slot * SEISMIC_ROUTES_STRIDE_1] = winners[slot];
            weights[row * SEISMIC_WEIGHTS_STRIDE_0 + slot * SEISMIC_WEIGHTS_STRIDE_1] = seismic_mul_rn(
                seismic_mul_rn(weight, scale),
                expert_scale[static_cast<unsigned long long>(winners[slot]) * SEISMIC_EXPERT_SCALE_STRIDE_0]);
        }
    }
}
