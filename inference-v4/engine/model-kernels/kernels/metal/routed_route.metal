// Routing for M rows in four launches.
//   routed_route_stage (one threadgroup per row): the row's RMS and its
//     normalized values (rounded to the activation dtype), published once.
//   routed_route_gemv (M <= 8): one simdgroup per logit column reads
//     the column's weights once and dots them with every row, normalizing
//     on the fly from RMS partials shared across the threadgroup.
//   routed_route_gemm (M > 8): 32 rows x 32 columns per threadgroup on
//     8x8 F32 simdgroup matrices over staged tiles.
//   Logits are [M, E + 1]: column E is the shared-expert gate.
//   routed_route_select (one simdgroup per row): the softmax over every expert
//     (E / 32 experts per lane), K rounds of simdgroup argmax (ties to the
//     higher expert index), the optional renormalization and the
//     shared-expert coefficient from column E.
// Every logit accumulates over the hidden axis in the same order for every
// expert, so experts with equal router rows tie exactly.

#include "lib/core/activation.h"
#include "lib/core/reduce.h"

typedef element::Act Act;
// The dense norm and router elements.
typedef ELEMENT_OF(SEISMIC_NORM) Norm;
typedef ELEMENT_OF(SEISMIC_ROUTER) Router;

constant constexpr uint route_threads = 256;
constant constexpr uint route_simdgroups = route_threads / 32;

kernel void routed_route_stage(
    device const float *residual [[buffer(SEISMIC_BUFFER_RESIDUAL)]],
    device const uchar *norm [[buffer(SEISMIC_BUFFER_NORM)]],
    device uchar *normalized [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint row_index [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partials[route_simdgroups];
    const ulong row = ulong(row_index);
    const float eps = as_type<float>(uint(SEISMIC_PARAM_EPS));
    float squares = 0.0f;
    for (uint source = tid; source < SEISMIC_DIM_H; source += route_threads) {
        const float value = residual[row * SEISMIC_RESIDUAL_STRIDE_0 + source * SEISMIC_RESIDUAL_STRIDE_1];
        squares = metal::fma(value, value, squares);
    }
    const float total = reduce::group_sum<route_simdgroups>(squares, partials, simd, lane);
    const float inverse = metal::rsqrt(total / float(SEISMIC_DIM_H) + eps);
    for (uint source = tid; source < SEISMIC_DIM_H; source += route_threads) {
        const float value = residual[row * SEISMIC_RESIDUAL_STRIDE_0 + source * SEISMIC_RESIDUAL_STRIDE_1];
        element::put<Act>(normalized, row * SEISMIC_RESULT_0_STRIDE_0 + source * SEISMIC_RESULT_0_STRIDE_1,
            Act::round(value * inverse * element::at<Norm>(norm, source * SEISMIC_NORM_STRIDE_0)));
    }
}

// The router logits scratch is [M, E + 1]: column E holds the shared-expert
// gate logit (normalized . shared_router), formed like an expert's logit.
inline ulong logit_index(ulong row, ulong expert) {
    return row * (SEISMIC_DIM_E + 1) + expert;
}

// The weight of logit column `expert` at hidden coordinate `source`.
inline float column_weight(device const uchar *router, device const float *shared_router, ulong expert,
    ulong source, constant ulong *seismic_words) {
    return expert < SEISMIC_DIM_E
        ? element::at<Router>(router,expert * SEISMIC_ROUTER_STRIDE_0 + source * SEISMIC_ROUTER_STRIDE_1)
        : shared_router[source * SEISMIC_SHARED_ROUTER_STRIDE_0];
}

// M <= 8, one launch before the selection: the threadgroup computes its rows'
// RMS inverses (in the normalize launch's order), then each simdgroup forms
// one logit column, normalizing each element as it reads it; threadgroup 0
// publishes the normalized rows.
kernel void routed_route_gemv(
    device const float *residual [[buffer(SEISMIC_BUFFER_RESIDUAL)]],
    device const uchar *norm [[buffer(SEISMIC_BUFFER_NORM)]],
    device const uchar *router [[buffer(SEISMIC_BUFFER_ROUTER)]],
    device const float *shared_router [[buffer(SEISMIC_BUFFER_SHARED_ROUTER)]],
    device uchar *normalized [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    device float *logits [[buffer(SEISMIC_BUFFER_SCRATCH_LOGITS)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float parts[8][route_simdgroups];
    threadgroup float inverses[8];
    const uint rows = uint(SEISMIC_DIM_M);
    const float eps = as_type<float>(uint(SEISMIC_PARAM_EPS));
    // The normalize launch's sum order (256 strided partials, simdgroup sums,
    // then the eight simdgroup partials in order): item (row, part) is part
    // `part`'s simdgroup sum, and the items spread over every simdgroup.
    for (uint item = simd; item < rows * route_simdgroups; item += SEISMIC_TUNE_SIMDGROUPS) {
        const uint row = item / route_simdgroups, part = item % route_simdgroups;
        float squares = 0.0f;
        for (uint source = part * 32 + lane; source < SEISMIC_DIM_H; source += route_threads) {
            const float value = residual[ulong(row) * SEISMIC_RESIDUAL_STRIDE_0 + source * SEISMIC_RESIDUAL_STRIDE_1];
            squares = metal::fma(value, value, squares);
        }
        squares = simd_sum(squares);
        if (lane == 0)
            parts[row][part] = squares;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < rows) {
        float total = 0.0f;
        for (uint part = 0; part < route_simdgroups; ++part)
            total += parts[tid][part];
        inverses[tid] = metal::rsqrt(total / float(SEISMIC_DIM_H) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (group == 0) {
        for (uint item = tid; item < rows * uint(SEISMIC_DIM_H); item += 32 * SEISMIC_TUNE_SIMDGROUPS) {
            const ulong row = item / SEISMIC_DIM_H, source = item % SEISMIC_DIM_H;
            const float value = residual[row * SEISMIC_RESIDUAL_STRIDE_0 + source * SEISMIC_RESIDUAL_STRIDE_1];
            element::put<Act>(normalized, row * SEISMIC_RESULT_0_STRIDE_0 + source * SEISMIC_RESULT_0_STRIDE_1,
                Act::round(value * inverses[row] * element::at<Norm>(norm, source * SEISMIC_NORM_STRIDE_0)));
        }
    }
    const ulong expert = ulong(group) * SEISMIC_TUNE_SIMDGROUPS + simd;
    if (expert > SEISMIC_DIM_E)
        return;
    // Lane l owns sources 4l + 128 i .. + 3 (four independent loads per
    // step), accumulated in source order within the lane.
    float sums[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint base = 4 * lane; base < SEISMIC_DIM_H; base += 128) {
        float weight[4], scale[4];
        for (uint j = 0; j < 4; ++j) {
            weight[j] = column_weight(router, shared_router, expert, base + j, seismic_words);
            scale[j] = element::at<Norm>(norm,(base + j) * SEISMIC_NORM_STRIDE_0);
        }
        for (uint row = 0; row < 8; ++row) {
            if (row < rows) {
                for (uint j = 0; j < 4; ++j) {
                    const float value = residual[ulong(row) * SEISMIC_RESIDUAL_STRIDE_0
                        + (base + j) * SEISMIC_RESIDUAL_STRIDE_1];
                    sums[row] = metal::fma(Act::round(value * inverses[row] * scale[j]), weight[j], sums[row]);
                }
            }
        }
    }
    for (uint row = 0; row < 8; ++row) {
        if (row < rows) {
            const float sum = simd_sum(sums[row]);
            if (lane == 0)
                logits[logit_index(row, expert)] = sum;
        }
    }
}

// Rows and logit columns per GEMM threadgroup, and hidden coordinates per
// stage (32 KB of threadgroup memory bounds the stage at 64).
constant constexpr uint route_tile = 32;
constant constexpr uint route_pitch = route_tile + 4;
constant constexpr uint route_depth = 64;
constant constexpr uint route_rows_pitch = route_depth + 4;

kernel void routed_route_gemm(
    device const uchar *router [[buffer(SEISMIC_BUFFER_ROUTER)]],
    device const float *shared_router [[buffer(SEISMIC_BUFFER_SHARED_ROUTER)]],
    device const uchar *normalized [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    device float *logits [[buffer(SEISMIC_BUFFER_SCRATCH_LOGITS)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float rows_tile[route_tile * route_rows_pitch];
    threadgroup float experts_tile[route_depth * route_pitch];
    const ulong expert0 = ulong(group.x) * route_tile;
    const ulong row0 = ulong(group.y) * route_tile;
    simdgroup_float8x8 accumulators[4];
    for (uint j = 0; j < 4; ++j)
        accumulators[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (ulong k0 = 0; k0 < SEISMIC_DIM_H; k0 += route_depth) {
        // A multiple of 8 (H % 32 == 0).
        const uint depth = uint(metal::min(ulong(route_depth), SEISMIC_DIM_H - k0));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // rows_tile[r][k] = normalized[row0 + r][k0 + k]; experts_tile[k][e] = column expert0 + e at k0 + k.
        for (uint item = tid; item < route_tile * depth; item += 128) {
            const uint r = item / depth, k = item % depth;
            const ulong row = row0 + r, expert = expert0 + r;
            rows_tile[r * route_rows_pitch + k] = row < SEISMIC_DIM_M
                ? element::at<Act>(normalized,row * SEISMIC_RESULT_0_STRIDE_0 + (k0 + k) * SEISMIC_RESULT_0_STRIDE_1)
                : 0.0f;
            experts_tile[k * route_pitch + r] = expert <= SEISMIC_DIM_E
                ? column_weight(router, shared_router, expert, k0 + k, seismic_words)
                : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < depth; kk += 8) {
            simdgroup_float8x8 a;
            simdgroup_load(a, rows_tile + (8 * simd) * route_rows_pitch + kk, route_rows_pitch);
            for (uint j = 0; j < 4; ++j) {
                simdgroup_float8x8 b;
                simdgroup_load(b, experts_tile + kk * route_pitch + 8 * j, route_pitch);
                simdgroup_multiply_accumulate(accumulators[j], a, b, accumulators[j]);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0; j < 4; ++j)
        simdgroup_store(accumulators[j], rows_tile + (8 * simd) * route_pitch + 8 * j, route_pitch);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint item = tid; item < route_tile * route_tile; item += 128) {
        const uint r = item / route_tile, e = item % route_tile;
        const ulong row = row0 + r, expert = expert0 + e;
        if (row < SEISMIC_DIM_M && expert <= SEISMIC_DIM_E)
            logits[logit_index(row, expert)] = rows_tile[r * route_pitch + e];
    }
}

// Descending probability, ties to the higher expert index.
inline bool precedes(float probability, int expert, float best, int best_expert) {
    return probability > best || (probability == best && expert > best_expert);
}

kernel void routed_route_select(
    device int *routes [[buffer(SEISMIC_BUFFER_ROUTES)]],
    device float *scores [[buffer(SEISMIC_BUFFER_SCORES)]],
    device float *coefficient [[buffer(SEISMIC_RESULT_1_BUFFER)]],
    device const float *logits [[buffer(SEISMIC_BUFFER_SCRATCH_LOGITS)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint per_lane = SEISMIC_DIM_E / 32;
    const ulong row = ulong(group) * route_simdgroups + simd;
    if (row >= SEISMIC_DIM_M)
        return;
    device const float *values = logits + logit_index(row, 0);
    // Lane `lane` holds experts lane + 32 i.
    float probability[per_lane];
    float maximum = -INFINITY;
    for (uint i = 0; i < per_lane; ++i) {
        probability[i] = values[lane + 32 * i];
        maximum = metal::max(maximum, probability[i]);
    }
    maximum = simd_max(maximum);
    float total = 0.0f;
    for (uint i = 0; i < per_lane; ++i) {
        probability[i] = metal::exp(probability[i] - maximum);
        total += probability[i];
    }
    total = simd_sum(total);
    for (uint i = 0; i < per_lane; ++i)
        probability[i] = probability[i] / total;

    float chosen[SEISMIC_DIM_K];
    for (uint rank = 0; rank < SEISMIC_DIM_K; ++rank) {
        float best = -INFINITY;
        int best_expert = -1;
        for (uint i = 0; i < per_lane; ++i) {
            const int expert = int(lane + 32 * i);
            if (precedes(probability[i], expert, best, best_expert)) {
                best = probability[i];
                best_expert = expert;
            }
        }
        float winner_probability;
        const int winner = reduce::argmax<reduce::HigherIndex>(best, best_expert, winner_probability);
        if (winner >= 0 && uint(winner) % 32 == lane)
            probability[uint(winner) / 32] = -INFINITY;
        chosen[rank] = winner_probability;
        if (lane == 0) {
            const ulong slot = SEISMIC_DIM_K - 1 - rank;
            routes[row * SEISMIC_ROUTES_STRIDE_0 + slot * SEISMIC_ROUTES_STRIDE_1] = winner;
            scores[row * SEISMIC_SCORES_STRIDE_0 + slot * SEISMIC_SCORES_STRIDE_1] = winner_probability;
        }
    }
    if (SEISMIC_PARAM_NORMALIZE != 0) {
        // The slot-order sum: slot s holds rank K - 1 - s.
        float denominator = 0.0f;
        for (uint slot = 0; slot < SEISMIC_DIM_K; ++slot)
            denominator += chosen[SEISMIC_DIM_K - 1 - slot];
        if (lane == 0) {
            for (uint slot = 0; slot < SEISMIC_DIM_K; ++slot)
                scores[row * SEISMIC_SCORES_STRIDE_0 + slot * SEISMIC_SCORES_STRIDE_1] =
                    chosen[SEISMIC_DIM_K - 1 - slot] / denominator;
        }
    }

    if (lane == 0)
        coefficient[row * SEISMIC_RESULT_1_STRIDE_0] = 1.0f / (1.0f + metal::exp(-values[SEISMIC_DIM_E]));
}
