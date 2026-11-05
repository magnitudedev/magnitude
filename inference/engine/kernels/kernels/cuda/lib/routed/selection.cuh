// The decode selection of the `routed_select` contract, shared by the
// `routed_select` and `routed_select_streamed` entries; the CUDA form of
// `metal/lib/routed/selection.h`. Both entries bind its parameters
// (`residual`, `norm`, `router_norm`, `router`, `bias`, `expert_scale` of
// static extent ES, `routes`, `weights`, `epsilon`, `score`,
// `normalization`, `normalization_epsilon`, `scale`), its result
// (`normalized`), its `logits` scratch and SIMDGROUPS and SPLIT under the
// same names. Every function takes the entry's words.
//   decode_inverses: a GEMV block's RMS inverses of the decode rows
//     (M <= DECODE_ROWS), in the stage launch's order.
//   select_columns: a GEMV block's SIMDGROUPS logit columns of the decode
//     rows over hidden share blockIdx.z of SPLIT, through a router norm and
//     router, into partial plane z of a layer's [SPLIT, M, E] logits, one
//     warp per column; the block forms the share's router input rows in
//     shared memory and, given `normalized`, publishes the share's
//     normalized rows.
//   select_row: one row's choices by one warp (`select::rank_row` over the
//     sum of the row's partial planes in plane order), their normalization
//     and scales into `routes` and `weights`; lane 0 then hands `bind` the
//     row's routes.

#include "../core/activation.cuh"
#include "../core/reduce.cuh"
#include "select.cuh"
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

// The hidden coordinates of one share of SPLIT: whole 64-value packets of a
// packed router, whole 32-value runs of a dense one.
#if defined(SEISMIC_ROUTER_KIND_PACKED)
constexpr unsigned SHARE_GRAIN = 64;
#else
constexpr unsigned SHARE_GRAIN = 32;
#endif
constexpr unsigned GEMV_SHARE = (static_cast<unsigned>(SEISMIC_DIM_H) / SHARE_GRAIN + SEISMIC_TUNE_SPLIT - 1) /
                                SEISMIC_TUNE_SPLIT * SHARE_GRAIN;

// Every thread of a GEMV block (32 SIMDGROUPS threads) calls it; `parts`
// and `inverses` are the block's.
__device__ __forceinline__ void decode_inverses(const float *residual, float (*parts)[SELECT_WARPS], float *inverses,
                                                const seismic_words_t &seismic_words_value) {
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
}

// Column group `group`'s logits over the router input
// round_A(r * inverse * router_norm) (`router_norm0` its stride, `router0`
// a dense router's row stride) into `logits`, a router of EXPERTS experts'
// [SPLIT, M, EXPERTS] partial planes (the entry's own by default; the
// streamed selection's hints rank the next layer's EN), publishing the
// share's rows round_A(r * inverse * norm) into `normalized` unless it is
// null; `staged` is the block's router input rows. Every thread of the block
// calls it.
template <unsigned EXPERTS = static_cast<unsigned>(SEISMIC_DIM_E)>
__device__ __forceinline__ void select_columns(const float *residual, const float *inverses,
                                               const unsigned char *router_norm, unsigned long long router_norm0,
                                               const unsigned char *router, [[maybe_unused]] unsigned long long router0,
                                               const unsigned char *norm, unsigned char *normalized, float *logits,
                                               unsigned group,
                                               Act::storage (*staged)[GEMV_SHARE],
                                               const seismic_words_t &seismic_words_value) {
    const unsigned thread = threadIdx.x, lane = thread % 32, warp = thread / 32;
    const unsigned rows = static_cast<unsigned>(SEISMIC_DIM_M);
    const unsigned grains = static_cast<unsigned>(SEISMIC_DIM_H) / SHARE_GRAIN;
    const unsigned share = blockIdx.z;
    const unsigned kbegin = grains * share / SEISMIC_TUNE_SPLIT * SHARE_GRAIN;
    const unsigned width = grains * (share + 1) / SEISMIC_TUNE_SPLIT * SHARE_GRAIN - kbegin;
    for (unsigned item = thread; item < rows * width; item += 32 * SEISMIC_TUNE_SIMDGROUPS) {
        const unsigned row = item / width, k = item % width;
        const unsigned long long source = kbegin + k;
        const float scaled =
            seismic_mul_rn(residual[row * SEISMIC_RESIDUAL_STRIDE_0 + source * SEISMIC_RESIDUAL_STRIDE_1], inverses[row]);
        staged[row][k] = Act::store(seismic_mul_rn(scaled, element::at<RouterNorm>(router_norm, source * router_norm0)));
        if (normalized != nullptr)
            element::put<Act>(normalized, row * SEISMIC_RESULT_0_STRIDE_0 + source * SEISMIC_RESULT_0_STRIDE_1,
                Act::round(seismic_mul_rn(scaled, element::at<Norm>(norm, source * SEISMIC_NORM_STRIDE_0))));
    }
    __syncthreads();
    logits += static_cast<unsigned long long>(share) * SEISMIC_DIM_M * EXPERTS;
    const unsigned long long expert = static_cast<unsigned long long>(group) * SEISMIC_TUNE_SIMDGROUPS + warp;
    if (expert >= EXPERTS)
        return;
    float sums[SEISMIC_CONST_DECODE_ROWS] = {};
#if defined(SEISMIC_ROUTER_KIND_PACKED)
    const auto router_rows = KERNEL_W0_AT(router);
    // One lane decodes a 16-value register chunk of an mma16 router row.
    for (unsigned item = lane; item < width / 64 * 4; item += 32) {
        const unsigned long long kb = kbegin / 64 + item / 4;
        const unsigned t = item % 4;
        const unsigned offsets[4] = {2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9};
        float values[16];
        packets::row_values16(router_rows, expert, kb, t, values);
#pragma unroll
        for (unsigned s = 0; s < 4; ++s)
#pragma unroll
            for (unsigned j = 0; j < 4; ++j) {
                const unsigned k = 64 * (item / 4) + 16 * s + offsets[j];
#pragma unroll
                for (unsigned row = 0; row < SEISMIC_CONST_DECODE_ROWS; ++row)
                    if (row < rows)
                        sums[row] = seismic_fma_rn(Act::load(staged[row][k]), values[4 * s + j], sums[row]);
            }
    }
#else
    // Dense lane l owns share sources 8l + 256 i .. + 7, one span of the
    // canonical router row and of each staged row.
#pragma unroll 4
    for (unsigned base = 8 * lane; base < width; base += 256) {
        float weight[8];
        element::span<Router, 8, true>(router, expert * router0 + kbegin + base, weight);
#pragma unroll
        for (unsigned row = 0; row < SEISMIC_CONST_DECODE_ROWS; ++row) {
            if (row < rows) {
                float input[8];
                element::span<Act, 8, false>(reinterpret_cast<const unsigned char *>(staged[row]), base, input);
#pragma unroll
                for (unsigned j = 0; j < 8; ++j)
                    sums[row] = seismic_fma_rn(input[j], weight[j], sums[row]);
            }
        }
    }
#endif
#pragma unroll
    for (unsigned row = 0; row < SEISMIC_CONST_DECODE_ROWS; ++row) {
        const float sum = seismic_warp_sum_f32(sums[row]);
        if (row < rows && lane == 0)
            logits[row * EXPERTS + expert] = sum;
    }
}

// Logit `expert` of row `row`: the sum of its partial planes of a layer's
// [SPLIT, M, EXPERTS] `logits` in plane order.
template <unsigned EXPERTS = static_cast<unsigned>(SEISMIC_DIM_E)>
__device__ __forceinline__ float select_logit(const float *logits, unsigned long long row, unsigned expert,
                                              const seismic_words_t &seismic_words_value) {
    const unsigned long long plane = SEISMIC_DIM_M * EXPERTS;
    float sum = logits[row * EXPERTS + expert];
    for (unsigned long long p = 1; p < SEISMIC_TUNE_SPLIT; ++p)
        sum = seismic_add_rn(sum, logits[p * plane + row * EXPERTS + expert]);
    return sum;
}

// One row's selection by one warp; lane 0 stores the row's routes and
// weights in slot order.
__device__ __forceinline__ void select_row(unsigned long long row, const float *logits, const float *bias,
                                           const float *expert_scale, int *routes, float *weights,
                                           const seismic_words_t &seismic_words_value) {
    constexpr unsigned K = SEISMIC_DIM_K;
    const unsigned lane = threadIdx.x % 32;
    const select::Selected<K> selected = select::rank_row<static_cast<unsigned>(SEISMIC_DIM_E), K>(
        static_cast<int>(SEISMIC_PARAM_SCORE),
        [&](unsigned expert) { return select_logit(logits, row, expert, seismic_words_value); },
        [&](int expert) { return bias[static_cast<unsigned long long>(expert) * SEISMIC_BIAS_STRIDE_0]; }, lane);
    // The slot-order sum: slot s holds rank K - 1 - s.
    float denominator = 0.0f;
    for (unsigned slot = 0; slot < K; ++slot)
        denominator = seismic_add_rn(denominator, selected.chosen[slot]);
    if (lane == 0) {
        const int normalization = static_cast<int>(SEISMIC_PARAM_NORMALIZATION);
        const float scale = __uint_as_float(static_cast<unsigned>(SEISMIC_PARAM_SCALE));
        const float epsilon = __uint_as_float(static_cast<unsigned>(SEISMIC_PARAM_NORMALIZATION_EPSILON));
        for (unsigned slot = 0; slot < K; ++slot) {
            float weight = seismic_mul_rn(select::normalized(normalization, selected.chosen[slot], denominator, epsilon), scale);
            if (SEISMIC_DIM_ES != 0)
                weight = seismic_mul_rn(
                    weight, expert_scale[static_cast<unsigned long long>(selected.winners[slot]) * SEISMIC_EXPERT_SCALE_STRIDE_0]);
            routes[row * SEISMIC_ROUTES_STRIDE_0 + slot * SEISMIC_ROUTES_STRIDE_1] = selected.winners[slot];
            weights[row * SEISMIC_WEIGHTS_STRIDE_0 + slot * SEISMIC_WEIGHTS_STRIDE_1] = weight;
        }
    }
}
