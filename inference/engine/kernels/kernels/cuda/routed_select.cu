// General expert selection for M rows; the CUDA form of
// `metal/routed_select.metal`, in four launches: `routed_select_stage`
// (M > DECODE_ROWS, one block per row) publishes the normalized rows and the router
// input rows once; `routed_select_gemv` (M <= DECODE_ROWS, one warp per logit column
// over the block's hidden share of SPLIT, whose router input rows the block
// forms in shared memory, publishing the share's normalized rows) or
// `routed_select_gemm` (32 rows x 32 columns x one hidden share of SPLIT per
// block over shared tiles) forms the [M, E] logits as SPLIT partial planes;
// `routed_select_select` (one warp per row) sums the planes, forms the
// scores, ranks them with the selection bias (single-pass top-K,
// `lib/routed/select.cuh`) and publishes the normalized, scaled weights.
// The decode GEMV and the selection are `lib/routed/selection.cuh`'s, which
// `routed_select_streamed` shares. Every logit accumulates over the hidden
// axis in the same order for every expert, so experts with equal router
// rows tie exactly.

#include "lib/routed/selection.cuh"

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

// M <= DECODE_ROWS, one launch before the selection: block (x, z) computes its rows'
// RMS inverses (in the stage launch's order), stages the router input rows of
// hidden share z of SPLIT in shared memory, and forms the partial logits of
// SIMDGROUPS logit columns over that share into plane z, one warp per column;
// the blocks of x = 0 publish the share's normalized rows.
extern "C" __global__ void routed_select_gemv(SEISMIC_KERNEL_PARAMS) {
    const float *residual = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL));
    const unsigned char *norm = SEISMIC_PTR(SEISMIC_BUFFER_NORM);
    const unsigned char *router_norm = SEISMIC_PTR(SEISMIC_BUFFER_ROUTER_NORM);
    const unsigned char *router = SEISMIC_PTR(SEISMIC_BUFFER_ROUTER);
    unsigned char *normalized = SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER);
    float *logits = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_LOGITS));
    __shared__ float parts[SEISMIC_CONST_DECODE_ROWS][SELECT_WARPS];
    __shared__ float inverses[SEISMIC_CONST_DECODE_ROWS];
    __shared__ alignas(16) Act::storage staged[SEISMIC_CONST_DECODE_ROWS][GEMV_SHARE];
    decode_inverses(residual, parts, inverses, seismic_words_value);
    select_columns(residual, inverses, router_norm, SEISMIC_ROUTER_NORM_STRIDE_0, router, SEISMIC_ROUTER_STRIDE_0, norm,
                   blockIdx.x == 0 ? normalized : nullptr, logits, blockIdx.x, staged, seismic_words_value);
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
    const unsigned long long row = static_cast<unsigned long long>(blockIdx.x) * SELECT_WARPS + threadIdx.x / 32;
    if (row >= SEISMIC_DIM_M)
        return;
    select_row(row, logits, bias, expert_scale, routes, weights, seismic_words_value);
}
