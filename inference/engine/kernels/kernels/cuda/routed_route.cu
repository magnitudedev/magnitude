// Routing for M rows; the CUDA form of `metal/routed_route.metal`, in
// four launches: `routed_route_stage` (M > DECODE_ROWS, one block per row) publishes
// the normalized rows once; `routed_route_gemv` (M <= DECODE_ROWS, one warp per
// logit column, normalizing on the fly and publishing the normalized rows,
// `lib/routed/route.cuh`) or `routed_route_gemm` (32 rows x 32 columns x
// one hidden share of SPLIT per block over shared tiles) forms the
// [M, E + 1] logits, column E being the shared-expert gate, as SPLIT
// partial planes; the selection (one warp per row: the last GEMV block to
// arrive, or `routed_route_select` after the GEMM) sums the planes, forms
// the softmax, selects the K winners by K rounds of warp argmax
// (`route::select_row`) and computes the shared-expert coefficient from
// column E. Every logit accumulates over the hidden axis in the same order
// for every expert, so experts with equal router rows tie exactly.

#include "lib/routed/route.cuh"

SEISMIC_PROGRAMMATIC_DEPENDENCY

// Row `row`'s RMS inverse (the whole block reduces it), or 0 without a norm:
// the sum order `route::decode_inverses` keeps.
__device__ __forceinline__ float routed_inverse(const float *residual, unsigned long long row, float eps,
                                                const seismic_words_t &seismic_words_value) {
    if constexpr (!route::NORMED) {
        return 0.0f;
    } else {
        __shared__ float partials[route::WARPS];
        float squares = 0.0f;
        for (unsigned source = threadIdx.x; source < SEISMIC_DIM_H; source += route::THREADS) {
            const float value = residual[row * SEISMIC_RESIDUAL_STRIDE_0 + source * SEISMIC_RESIDUAL_STRIDE_1];
            squares = seismic_fma_rn(value, value, squares);
        }
        const float total = reduce::group_sum(squares, partials);
        return rsqrtf(seismic_add_rn(total / static_cast<float>(SEISMIC_DIM_H), eps));
    }
}

extern "C" __global__ void routed_route_stage(SEISMIC_KERNEL_PARAMS) {
    seismic_dependency_start();
    const route::Input in = ROUTE_INPUT;
    unsigned char *normalized = SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER);
    const unsigned thread = threadIdx.x;
    const unsigned long long row = blockIdx.x;
    const float inverse =
        routed_inverse(in.residual, row, __uint_as_float(static_cast<unsigned>(SEISMIC_PARAM_EPS)), seismic_words_value);
    for (unsigned source = thread; source < SEISMIC_DIM_H; source += route::THREADS)
        element::put<route::Act>(normalized, row * SEISMIC_RESULT_0_STRIDE_0 + source * SEISMIC_RESULT_0_STRIDE_1,
            in.value(row, source, inverse, seismic_words_value));
}

// The router logits scratch is SPLIT planes [M, E + 1]: the GEMV (M <= DECODE_ROWS)
// writes plane 0 whole; the GEMM (M > DECODE_ROWS) writes plane p with the hidden
// share p of SPLIT, and the selection sums the planes in order.

// M <= DECODE_ROWS, one launch of router blocks (`route::route_block`): the block
// computes its rows' RMS inverses (in the normalize launch's order), then
// each warp forms one logit column; block 0 publishes the normalized rows;
// the last block to arrive selects.
extern "C" __global__ void routed_route_gemv(SEISMIC_KERNEL_PARAMS) {
    __shared__ float parts[SEISMIC_CONST_DECODE_ROWS][route::WARPS];
    __shared__ float inverses[SEISMIC_CONST_DECODE_ROWS];
    __shared__ unsigned last;
    route::route_block(ROUTE_INPUT, route::FormRows{}, ROUTE_COLUMNS, ROUTE_SELECTION,
        SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER), SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1,
        reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_LOGITS)),
        reinterpret_cast<unsigned *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_ARRIVALS)), blockIdx.x, parts, inverses,
        &last, seismic_words_value);
}

// Rows and logit columns per GEMM block, and hidden coordinates per stage.
constexpr unsigned ROUTE_TILE = 32;
constexpr unsigned ROUTE_DEPTH = 128;

extern "C" __global__ void routed_route_gemm(SEISMIC_KERNEL_PARAMS) {
    seismic_dependency_start();
    const route::Columns columns = ROUTE_COLUMNS;
    const unsigned char *normalized = SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER);
    float *logits = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_LOGITS));
#if defined(SEISMIC_ROUTER_KIND_PACKED)
    const auto router_rows = KERNEL_W0_AT(columns.router);
#endif
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
        for (unsigned item = thread; item < ROUTE_TILE * ROUTE_DEPTH; item += route::THREADS) {
            const unsigned i = item / ROUTE_DEPTH, k = item % ROUTE_DEPTH;
            if (k >= depth)
                continue;
            const unsigned long long row = row0 + i, expert = expert0 + i;
            rows_tile[i][k] = row < SEISMIC_DIM_M
                ? element::at<route::Act>(normalized, row * SEISMIC_RESULT_0_STRIDE_0 + (k0 + k) * SEISMIC_RESULT_0_STRIDE_1)
                : 0.0f;
#if !defined(SEISMIC_ROUTER_KIND_PACKED)
            experts_tile[i][k] = expert <= SEISMIC_DIM_E
                ? route::weight(columns, expert, k0 + k, seismic_words_value) : 0.0f;
#endif
        }
#if defined(SEISMIC_ROUTER_KIND_PACKED)
        // Each item decodes 16 values into one expert's shared-memory row.
        const unsigned long long first_kb = k0 / 64;
        const unsigned blocks = static_cast<unsigned>((k0 + depth + 63) / 64 - first_kb);
        for (unsigned item = thread; item < ROUTE_TILE * blocks * 4; item += route::THREADS) {
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
                    const unsigned k = static_cast<unsigned>(source - k0);
                    experts_tile[i][k] = expert < SEISMIC_DIM_E ? values[4 * s + j]
                        : expert == SEISMIC_DIM_E
                            ? columns.shared_router[(k0 + k) * columns.shared_router0] : 0.0f;
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
        if (row < SEISMIC_DIM_M && expert <= SEISMIC_DIM_E)
            logits[route::logit_index(row, expert, seismic_words_value)] = acc[j];
    }
}

// M > DECODE_ROWS: the selection after the GEMM, each logit the sum of its SPLIT
// partial planes in plane order (L2 reads: other blocks wrote them).
extern "C" __global__ void routed_route_select(SEISMIC_KERNEL_PARAMS) {
    seismic_dependency_start();
    const unsigned long long row = static_cast<unsigned long long>(blockIdx.x) * route::WARPS + threadIdx.x / 32;
    if (row >= SEISMIC_DIM_M)
        return;
    const unsigned long long plane = SEISMIC_DIM_M * (SEISMIC_DIM_E + 1);
    const float *values = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_LOGITS))
        + route::logit_index(row, 0, seismic_words_value);
    const auto logit = [&](unsigned long long expert) {
        float sum = __ldcg(values + expert);
        for (unsigned long long p = 1; p < SEISMIC_TUNE_SPLIT; ++p)
            sum = seismic_add_rn(sum, __ldcg(values + p * plane + expert));
        return sum;
    };
    route::select_row(logit, ROUTE_SELECTION, row, threadIdx.x % 32, seismic_words_value);
}
