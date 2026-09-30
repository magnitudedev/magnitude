// General expert selection for M rows in four launches (the `routed_route`
// structure without the shared-expert column):
//   routed_select_stage (M > 8, one threadgroup per row): the row's RMS and
//     its two normalized forms, published once: `normalized` (norm, the
//     experts' input) and the router input (router_norm) into scratch.
//   routed_select_gemv (M <= 8): the router logits (`lib/routed/router.h`),
//     one simdgroup per logit column over the router input each threadgroup
//     forms from RMS partials of its own; threadgroup 0 publishes
//     `normalized`.
//   routed_select_gemm (M > 8): 32 rows x 32 columns per threadgroup on 8x8
//     F32 simdgroup matrices over the staged router input.
//   routed_select_select (one simdgroup per row): scores, biased ranking,
//     single-pass top-K (`lib/routed/select.h`), normalization, scales.
// Every logit accumulates over the hidden axis in the same order for every
// expert, so experts with equal router rows tie exactly.

#define KERNEL_W0 SEISMIC_ROUTER
#include "lib/routed/router.h"
#include "lib/routed/select.h"

typedef element::Act Act;
typedef ELEMENT_OF(SEISMIC_NORM) Norm;
typedef ELEMENT_OF(SEISMIC_ROUTER_NORM) RouterNorm;
typedef routing::Packets<packets::W0>::type Router;

constant constexpr uint select_threads = routing::stage_threads;
constant constexpr uint select_simdgroups = routing::stage_simdgroups;

kernel void routed_select_stage(
    device const float *residual [[buffer(SEISMIC_BUFFER_RESIDUAL)]],
    device const uchar *norm [[buffer(SEISMIC_BUFFER_NORM)]],
    device const uchar *router_norm [[buffer(SEISMIC_BUFFER_ROUTER_NORM)]],
    device uchar *normalized [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    device uchar *router_rows [[buffer(SEISMIC_BUFFER_SCRATCH_ROUTER_ROWS)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint row_index [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partials[select_simdgroups];
    const ulong row = ulong(row_index);
    const float eps = as_type<float>(uint(SEISMIC_PARAM_EPSILON));
    float squares = 0.0f;
    for (uint source = tid; source < SEISMIC_DIM_H; source += select_threads) {
        const float value = residual[row * SEISMIC_RESIDUAL_STRIDE_0 + source * SEISMIC_RESIDUAL_STRIDE_1];
        squares = metal::fma(value, value, squares);
    }
    const float total = reduce::group_sum<select_simdgroups>(squares, partials, simd, lane);
    const float inverse = metal::rsqrt(total / float(SEISMIC_DIM_H) + eps);
    for (uint source = tid; source < SEISMIC_DIM_H; source += select_threads) {
        const float value = residual[row * SEISMIC_RESIDUAL_STRIDE_0 + source * SEISMIC_RESIDUAL_STRIDE_1];
        element::put<Act>(normalized, row * SEISMIC_RESULT_0_STRIDE_0 + source * SEISMIC_RESULT_0_STRIDE_1,
            Act::round(value * inverse * element::at<Norm>(norm, source * SEISMIC_NORM_STRIDE_0)));
        element::put<Act>(router_rows, row * SEISMIC_DIM_H + source,
            Act::round(value * inverse * element::at<RouterNorm>(router_norm, source * SEISMIC_ROUTER_NORM_STRIDE_0)));
    }
}

// M <= 8, one launch before the selection: the threadgroup computes its rows'
// RMS inverses (in the stage launch's order); threadgroup 0 publishes the
// normalized rows; each threadgroup then forms the logits of SIMDGROUPS
// router rows over the router input (`routing::Gemv`).
kernel void routed_select_gemv(
    device const float *residual [[buffer(SEISMIC_BUFFER_RESIDUAL)]],
    device const uchar *norm [[buffer(SEISMIC_BUFFER_NORM)]],
    device const uchar *router_norm [[buffer(SEISMIC_BUFFER_ROUTER_NORM)]],
    device const uchar *router [[buffer(SEISMIC_BUFFER_ROUTER)]],
    device uchar *normalized [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    device float *logits [[buffer(SEISMIC_BUFFER_SCRATCH_LOGITS)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    threadgroup uchar *shared [[threadgroup(0)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float parts[8 * select_simdgroups];
    threadgroup float inverses[8];
    const uint rows = uint(SEISMIC_DIM_M);
    const uint k = uint(SEISMIC_DIM_H);
    routing::inverses(residual, SEISMIC_RESIDUAL_STRIDE_0, SEISMIC_RESIDUAL_STRIDE_1, k,
        as_type<float>(uint(SEISMIC_PARAM_EPSILON)), rows, parts, inverses, SEISMIC_TUNE_SIMDGROUPS, simd, lane,
        tid);
    if (group == 0) {
        for (uint item = tid; item < rows * uint(SEISMIC_DIM_H); item += 32 * SEISMIC_TUNE_SIMDGROUPS) {
            const ulong row = item / SEISMIC_DIM_H, source = item % SEISMIC_DIM_H;
            const float value = residual[row * SEISMIC_RESIDUAL_STRIDE_0 + source * SEISMIC_RESIDUAL_STRIDE_1];
            element::put<Act>(normalized, row * SEISMIC_RESULT_0_STRIDE_0 + source * SEISMIC_RESULT_0_STRIDE_1,
                Act::round(value * inverses[row] * element::at<Norm>(norm, source * SEISMIC_NORM_STRIDE_0)));
        }
    }
    const routing::Rows<RouterNorm> in{residual, SEISMIC_RESIDUAL_STRIDE_0, SEISMIC_RESIDUAL_STRIDE_1,
        router_norm, SEISMIC_ROUTER_NORM_STRIDE_0, inverses, k};
    const projection::Weights<Router> weights{router, KERNEL_W0_LAYOUT(k), k, nullptr};
    const routing::Logits out{logits, SEISMIC_DIM_E, 0};
    routing::Gemv<Act::bytes>::columns(in, weights, out, rows, uint(SEISMIC_DIM_E), group, shared,
        SEISMIC_TUNE_SIMDGROUPS, simd, lane);
}

// Rows and logit columns per GEMM threadgroup, and hidden coordinates per
// stage (32 KB of threadgroup memory bounds the stage at 64).
constant constexpr uint select_tile = 32;
constant constexpr uint select_pitch = select_tile + 4;
constant constexpr uint select_depth = 64;
constant constexpr uint select_rows_pitch = select_depth + 4;

kernel void routed_select_gemm(
    device const uchar *router [[buffer(SEISMIC_BUFFER_ROUTER)]],
    device const uchar *router_rows [[buffer(SEISMIC_BUFFER_SCRATCH_ROUTER_ROWS)]],
    device float *logits [[buffer(SEISMIC_BUFFER_SCRATCH_LOGITS)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float rows_tile[select_tile * select_rows_pitch];
    threadgroup float experts_tile[select_depth * select_pitch];
    projection::Weights<packets::W0> weights{router, KERNEL_W0_LAYOUT(uint(SEISMIC_DIM_H)),
        uint(SEISMIC_DIM_H), nullptr};
    const ulong expert0 = ulong(group.x) * select_tile;
    const ulong row0 = ulong(group.y) * select_tile;
    simdgroup_float8x8 accumulators[4];
    for (uint j = 0; j < 4; ++j)
        accumulators[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (ulong k0 = 0; k0 < SEISMIC_DIM_H; k0 += select_depth) {
        // A multiple of 8 (H % 32 == 0).
        const uint depth = uint(metal::min(ulong(select_depth), SEISMIC_DIM_H - k0));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint item = tid; item < select_tile * depth; item += 128) {
            const uint r = item / depth, k = item % depth;
            const ulong row = row0 + r;
            rows_tile[r * select_rows_pitch + k] = row < SEISMIC_DIM_M
                ? element::at<Act>(router_rows, row * SEISMIC_DIM_H + k0 + k) : 0.0f;
        }
        // One thread decodes a packet for one expert column.
        for (uint item = tid; item < select_tile * (depth / 32); item += 128) {
            const uint r = item / (depth / 32), packet_in_tile = item % (depth / 32);
            const ulong expert = expert0 + r, source0 = k0 + 32ul * packet_in_tile;
            if (expert < SEISMIC_DIM_E) {
                typename packets::W0::packet packet = weights.packet(uint(expert), uint(source0 / 32));
                for (uint step = 0; step < 4; ++step) {
                    float4 even, odd;
                    packets::W0::codes(packet, step, even, odd);
                    for (uint i = 0; i < 8; ++i) {
                        const float code = (i & 1u) ? odd[i >> 1] : even[i >> 1];
                        experts_tile[(32 * packet_in_tile + 8 * step + i) * select_pitch + r] =
                            packets::W0::value(packet, step, code);
                    }
                }
            } else {
                for (uint i = 0; i < 32; ++i)
                    experts_tile[(32 * packet_in_tile + i) * select_pitch + r] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < depth; kk += 8) {
            simdgroup_float8x8 a;
            simdgroup_load(a, rows_tile + (8 * simd) * select_rows_pitch + kk, select_rows_pitch);
            for (uint j = 0; j < 4; ++j) {
                simdgroup_float8x8 b;
                simdgroup_load(b, experts_tile + kk * select_pitch + 8 * j, select_pitch);
                simdgroup_multiply_accumulate(accumulators[j], a, b, accumulators[j]);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0; j < 4; ++j)
        simdgroup_store(accumulators[j], rows_tile + (8 * simd) * select_pitch + 8 * j, select_pitch);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint item = tid; item < select_tile * select_tile; item += 128) {
        const uint r = item / select_tile, e = item % select_tile;
        const ulong row = row0 + r, expert = expert0 + e;
        if (row < SEISMIC_DIM_M && expert < SEISMIC_DIM_E)
            logits[row * SEISMIC_DIM_E + expert] = rows_tile[r * select_pitch + e];
    }
}

kernel void routed_select_select(
    device const float *bias [[buffer(SEISMIC_BUFFER_BIAS)]],
    device const float *expert_scale [[buffer(SEISMIC_BUFFER_EXPERT_SCALE)]],
    device int *routes [[buffer(SEISMIC_BUFFER_ROUTES)]],
    device float *weights [[buffer(SEISMIC_BUFFER_WEIGHTS)]],
    device const float *logits [[buffer(SEISMIC_BUFFER_SCRATCH_LOGITS)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint P = SEISMIC_DIM_E / 32;
    const ulong row = ulong(group) * select_simdgroups + simd;
    if (row >= SEISMIC_DIM_M)
        return;
    const int normalization = int(SEISMIC_PARAM_NORMALIZATION);
    float score[P], ranked[P];
    int expert[P];
    // Every loop over a register array unrolls: a dynamically indexed array
    // lives in stack memory.
    PROJECTION_UNROLL
    for (uint i = 0; i < P; ++i) {
        expert[i] = int(lane + 32 * i);
        score[i] = logits[row * SEISMIC_DIM_E + lane + 32 * i];
    }
    select::scores<P>(int(SEISMIC_PARAM_SCORE), score);
    PROJECTION_UNROLL
    for (uint i = 0; i < P; ++i)
        ranked[i] = score[i] + bias[ulong(expert[i]) * SEISMIC_BIAS_STRIDE_0];
    select::sort<P>(ranked, score, expert);

    float chosen[SEISMIC_DIM_K];
    int winners[SEISMIC_DIM_K];
    PROJECTION_UNROLL
    for (uint rank = 0; rank < SEISMIC_DIM_K; ++rank) {
        float best;
        const int winner = reduce::argmax<reduce::HigherIndex>(ranked[0], expert[0], best);
        // The winner's unbiased score, from its lane.
        const float winner_score = simd_broadcast(score[0], ushort(uint(winner) % 32u));
        if (expert[0] == winner) {
            _Pragma("clang loop unroll(full)")
            for (uint i = 0; i + 1u < P; ++i) {
                ranked[i] = ranked[i + 1u];
                score[i] = score[i + 1u];
                expert[i] = expert[i + 1u];
            }
            ranked[P - 1] = -INFINITY;
            expert[P - 1] = -1;
        }
        chosen[SEISMIC_DIM_K - 1 - rank] = winner_score;
        winners[SEISMIC_DIM_K - 1 - rank] = winner;
    }
    // The slot-order sum: slot s holds rank K - 1 - s.
    float denominator = 0.0f;
    PROJECTION_UNROLL
    for (uint slot = 0; slot < SEISMIC_DIM_K; ++slot)
        denominator += chosen[slot];
    if (lane == 0) {
        const float scale = as_type<float>(uint(SEISMIC_PARAM_SCALE));
        const float epsilon = as_type<float>(uint(SEISMIC_PARAM_NORMALIZATION_EPSILON));
        PROJECTION_UNROLL
        for (uint slot = 0; slot < SEISMIC_DIM_K; ++slot) {
            const float weight = select::normalized(normalization, chosen[slot], denominator, epsilon);
            routes[row * SEISMIC_ROUTES_STRIDE_0 + slot * SEISMIC_ROUTES_STRIDE_1] = winners[slot];
            weights[row * SEISMIC_WEIGHTS_STRIDE_0 + slot * SEISMIC_WEIGHTS_STRIDE_1] =
                weight * scale * expert_scale[ulong(winners[slot]) * SEISMIC_EXPERT_SCALE_STRIDE_0];
        }
    }
}
