// Routing for M rows; the CUDA form of `metal/routed_route.metal`, in
// four launches: `routed_route_stage` (M > 8, one block per row) publishes
// the normalized rows once; `routed_route_gemv` (M <= 8, one warp per
// logit column, normalizing on the fly and publishing the normalized rows) or
// `routed_route_gemm` (32 rows x 32 columns x one hidden share of SPLIT
// per block over shared tiles) forms the [M, E + 1] logits, column E being
// the shared-expert gate, as SPLIT partial planes; `routed_route_select`
// (one warp per row) sums the planes, forms the softmax, selects the K winners
// by K rounds of warp argmax (ties to the higher expert index) and computes the
// shared-expert coefficient from column E. Every logit accumulates over the
// hidden axis in the same order for every expert, so experts with equal router
// rows tie exactly.

#include "lib/core/activation.cuh"
#include "lib/core/reduce.cuh"

namespace {

using element::Act;
// The dense norm and router elements.
typedef ELEMENT_OF(SEISMIC_NORM) Norm;
typedef ELEMENT_OF(SEISMIC_ROUTER) Router;

} // namespace

constexpr unsigned ROUTE_THREADS = 256;
constexpr unsigned ROUTE_WARPS = ROUTE_THREADS / 32;

extern "C" __global__ void routed_route_stage(SEISMIC_KERNEL_PARAMS) {
    const float *residual = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL));
    const unsigned char *norm = SEISMIC_PTR(SEISMIC_BUFFER_NORM);
    unsigned char *normalized = SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER);
    __shared__ float partials[ROUTE_WARPS];
    const unsigned thread = threadIdx.x;
    const unsigned long long row = blockIdx.x;
    const float eps = __uint_as_float(static_cast<unsigned>(SEISMIC_PARAM_EPS));
    float squares = 0.0f;
    for (unsigned source = thread; source < SEISMIC_DIM_H; source += ROUTE_THREADS) {
        const float value = residual[row * SEISMIC_RESIDUAL_STRIDE_0 + source * SEISMIC_RESIDUAL_STRIDE_1];
        squares = seismic_fma_rn(value, value, squares);
    }
    const float total = reduce::group_sum(squares, partials);
    const float inverse = rsqrtf(seismic_add_rn(total / static_cast<float>(SEISMIC_DIM_H), eps));
    for (unsigned source = thread; source < SEISMIC_DIM_H; source += ROUTE_THREADS) {
        const float value = residual[row * SEISMIC_RESIDUAL_STRIDE_0 + source * SEISMIC_RESIDUAL_STRIDE_1];
        element::put<Act>(normalized, row * SEISMIC_RESULT_0_STRIDE_0 + source * SEISMIC_RESULT_0_STRIDE_1,
            Act::round(seismic_mul_rn(seismic_mul_rn(value, inverse),
                element::at<Norm>(norm, source * SEISMIC_NORM_STRIDE_0))));
    }
}

// The router logits scratch is SPLIT planes [M, E + 1]: column E holds the
// shared-expert gate logit (normalized . shared_router), formed like an
// expert's logit. The GEMV (M <= 8) writes plane 0 whole; the GEMM (M > 8)
// writes plane p with the hidden share p of SPLIT, and the selection sums the
// planes in order.
__device__ __forceinline__ unsigned long long routed_logit_index(unsigned long long row, unsigned long long expert) {
    return row * (SEISMIC_DIM_E + 1) + expert;
}

// The weight of logit column `expert` at hidden coordinate `source`.
__device__ __forceinline__ float routed_column_weight(const unsigned char *router, const float *shared_router,
    unsigned long long expert, unsigned long long source) {
    return expert < SEISMIC_DIM_E
        ? element::at<Router>(router, expert * SEISMIC_ROUTER_STRIDE_0 + source * SEISMIC_ROUTER_STRIDE_1)
        : shared_router[source * SEISMIC_SHARED_ROUTER_STRIDE_0];
}

// M <= 8, one launch before the selection: the block computes its rows' RMS
// inverses (in the normalize launch's order), then each warp forms one logit
// column, normalizing each element as it reads it; block 0 publishes the
// normalized rows.
extern "C" __global__ void routed_route_gemv(SEISMIC_KERNEL_PARAMS) {
    const float *residual = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL));
    const unsigned char *norm = SEISMIC_PTR(SEISMIC_BUFFER_NORM);
    const unsigned char *router = SEISMIC_PTR(SEISMIC_BUFFER_ROUTER);
    const float *shared_router = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_ROUTER));
    unsigned char *normalized = SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER);
    float *logits = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_LOGITS));
    __shared__ float parts[8][ROUTE_WARPS];
    __shared__ float inverses[8];
    const unsigned thread = threadIdx.x, lane = thread % 32, warp = thread / 32;
    const unsigned rows = static_cast<unsigned>(SEISMIC_DIM_M);
    const float eps = __uint_as_float(static_cast<unsigned>(SEISMIC_PARAM_EPS));
    // The normalize launch's sum order (256 strided partials, warp sums, then
    // the eight warp partials in order): item (row, part) is part `part`'s
    // warp sum, and the items spread over every warp.
    for (unsigned item = warp; item < rows * ROUTE_WARPS; item += SEISMIC_TUNE_SIMDGROUPS) {
        const unsigned row = item / ROUTE_WARPS, part = item % ROUTE_WARPS;
        float squares = 0.0f;
        for (unsigned source = part * 32 + lane; source < SEISMIC_DIM_H; source += ROUTE_THREADS) {
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
        for (unsigned part = 0; part < ROUTE_WARPS; ++part)
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
    if (expert > SEISMIC_DIM_E)
        return;
    // Lane l owns sources 4l + 128 i .. + 3 (four independent loads per
    // step), accumulated in source order within the lane.
    float sums[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (unsigned base = 4 * lane; base < SEISMIC_DIM_H; base += 128) {
        float weight[4], scale[4];
#pragma unroll
        for (unsigned j = 0; j < 4; ++j) {
            weight[j] = routed_column_weight(router, shared_router, expert, base + j);
            scale[j] = element::at<Norm>(norm, (base + j) * SEISMIC_NORM_STRIDE_0);
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
#pragma unroll
    for (unsigned row = 0; row < 8; ++row) {
        const float sum = seismic_warp_sum_f32(sums[row]);
        if (row < rows && lane == 0)
            logits[routed_logit_index(row, expert)] = sum;
    }
}

// Rows and logit columns per GEMM block, and hidden coordinates per stage.
constexpr unsigned ROUTE_TILE = 32;
constexpr unsigned ROUTE_DEPTH = 128;

extern "C" __global__ void routed_route_gemm(SEISMIC_KERNEL_PARAMS) {
    const unsigned char *router = SEISMIC_PTR(SEISMIC_BUFFER_ROUTER);
    const float *shared_router = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_ROUTER));
    const unsigned char *normalized = SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER);
    float *logits = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_LOGITS));
    // Row pitch ROUTE_DEPTH + 1 = 1 (mod 32 banks): the eight column groups of
    // a warp read eight banks.
    __shared__ float rows_tile[ROUTE_TILE][ROUTE_DEPTH + 1];
    __shared__ float experts_tile[ROUTE_TILE][ROUTE_DEPTH + 1];
    const unsigned thread = threadIdx.x;
    const unsigned long long expert0 = static_cast<unsigned long long>(blockIdx.x) * ROUTE_TILE;
    const unsigned long long row0 = static_cast<unsigned long long>(blockIdx.y) * ROUTE_TILE;
    // Block z forms the partial logits of hidden share z of SPLIT into plane z.
    const unsigned long long part = blockIdx.z;
    const unsigned long long kbegin = SEISMIC_DIM_H / 32 * part / SEISMIC_TUNE_SPLIT * 32;
    const unsigned long long kend = SEISMIC_DIM_H / 32 * (part + 1) / SEISMIC_TUNE_SPLIT * 32;
    logits += part * SEISMIC_DIM_M * (SEISMIC_DIM_E + 1);
    // Thread t owns row t / 8 and experts 4 (t % 8) .. 4 (t % 8) + 3; each
    // partial logit accumulates over its share in ascending order.
    const unsigned r = thread / 8, e0 = 4 * (thread % 8);
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (unsigned long long k0 = kbegin; k0 < kend; k0 += ROUTE_DEPTH) {
        const unsigned depth = static_cast<unsigned>(min(static_cast<unsigned long long>(ROUTE_DEPTH), kend - k0));
        __syncthreads();
        for (unsigned item = thread; item < ROUTE_TILE * ROUTE_DEPTH; item += ROUTE_THREADS) {
            const unsigned i = item / ROUTE_DEPTH, k = item % ROUTE_DEPTH;
            if (k >= depth)
                continue;
            const unsigned long long row = row0 + i, expert = expert0 + i;
            rows_tile[i][k] = row < SEISMIC_DIM_M
                ? element::at<Act>(normalized, row * SEISMIC_RESULT_0_STRIDE_0 + (k0 + k) * SEISMIC_RESULT_0_STRIDE_1)
                : 0.0f;
            experts_tile[i][k] = expert <= SEISMIC_DIM_E
                ? routed_column_weight(router, shared_router, expert, k0 + k)
                : 0.0f;
        }
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
        if (row < SEISMIC_DIM_M && expert <= SEISMIC_DIM_E)
            logits[routed_logit_index(row, expert)] = acc[j];
    }
}

// Descending probability, ties to the higher expert index.
__device__ __forceinline__ bool routed_precedes(float probability, int expert, float best, int best_expert) {
    return probability > best || (probability == best && expert > best_expert);
}

extern "C" __global__ void routed_route_select(SEISMIC_KERNEL_PARAMS) {
    int *routes = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES));
    float *scores = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCORES));
    float *coefficient = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_1_BUFFER));
    const float *logits = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_LOGITS));
    constexpr unsigned PER_LANE = SEISMIC_DIM_E / 32;
    const unsigned lane = threadIdx.x % 32;
    const unsigned long long row = static_cast<unsigned long long>(blockIdx.x) * ROUTE_WARPS + threadIdx.x / 32;
    if (row >= SEISMIC_DIM_M)
        return;
    const float negative_infinity = -__int_as_float(0x7f800000);
    const float *values = logits + routed_logit_index(row, 0);
    // Logit column e: the sum of its partial planes in plane order (the GEMV
    // forms plane 0 whole).
    const unsigned long long parts = SEISMIC_DIM_M > 8 ? SEISMIC_TUNE_SPLIT : 1;
    const unsigned long long plane = SEISMIC_DIM_M * (SEISMIC_DIM_E + 1);
    const auto logit = [&](unsigned long long expert) {
        float sum = values[expert];
        for (unsigned long long p = 1; p < parts; ++p)
            sum = seismic_add_rn(sum, values[p * plane + expert]);
        return sum;
    };
    // Lane `lane` holds experts lane + 32 i.
    float probability[PER_LANE];
    float maximum = negative_infinity;
#pragma unroll
    for (unsigned i = 0; i < PER_LANE; ++i) {
        probability[i] = logit(lane + 32 * i);
        maximum = fmaxf(maximum, probability[i]);
    }
    maximum = seismic_warp_max_f32(maximum);
    float total = 0.0f;
#pragma unroll
    for (unsigned i = 0; i < PER_LANE; ++i) {
        probability[i] = expf(probability[i] - maximum);
        total = seismic_add_rn(total, probability[i]);
    }
    total = seismic_warp_sum_f32(total);
#pragma unroll
    for (unsigned i = 0; i < PER_LANE; ++i)
        probability[i] = probability[i] / total;

    float chosen[SEISMIC_DIM_K];
    for (unsigned rank = 0; rank < SEISMIC_DIM_K; ++rank) {
        float best = negative_infinity;
        int best_expert = -1;
#pragma unroll
        for (unsigned i = 0; i < PER_LANE; ++i) {
            const int expert = static_cast<int>(lane + 32 * i);
            if (routed_precedes(probability[i], expert, best, best_expert)) {
                best = probability[i];
                best_expert = expert;
            }
        }
        float winner_probability;
        const int winner = reduce::argmax<reduce::HigherIndex>(best, best_expert, winner_probability);
#pragma unroll
        for (unsigned i = 0; i < PER_LANE; ++i)
            if (winner >= 0 && static_cast<unsigned>(winner) == lane + 32 * i)
                probability[i] = negative_infinity;
        chosen[rank] = winner_probability;
        if (lane == 0) {
            const unsigned long long slot = SEISMIC_DIM_K - 1 - rank;
            routes[row * SEISMIC_ROUTES_STRIDE_0 + slot * SEISMIC_ROUTES_STRIDE_1] = winner;
            scores[row * SEISMIC_SCORES_STRIDE_0 + slot * SEISMIC_SCORES_STRIDE_1] = winner_probability;
        }
    }
    if (SEISMIC_PARAM_NORMALIZE != 0) {
        // The slot-order sum: slot s holds rank K - 1 - s.
        float denominator = 0.0f;
        for (unsigned slot = 0; slot < SEISMIC_DIM_K; ++slot)
            denominator = seismic_add_rn(denominator, chosen[SEISMIC_DIM_K - 1 - slot]);
        if (lane == 0)
            for (unsigned slot = 0; slot < SEISMIC_DIM_K; ++slot)
                scores[row * SEISMIC_SCORES_STRIDE_0 + slot * SEISMIC_SCORES_STRIDE_1] =
                    chosen[SEISMIC_DIM_K - 1 - slot] / denominator;
    }

    if (lane == 0)
        coefficient[row * SEISMIC_RESULT_1_STRIDE_0] = 1.0f / (1.0f + expf(-logit(SEISMIC_DIM_E)));
}
