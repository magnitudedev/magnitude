// Routing for M rows: one launch at M <= 8 (gemv), three above (stage, gemm,
// select).
//   routed_route_stage (one threadgroup per row): the row's RMS and its
//     normalized values (rounded to the activation dtype), published once.
//   routed_route_gemv (M <= 8): the router logits (`lib/routed/router.h`,
//     `lib/routed/route.h`), one simdgroup per logit column over the
//     normalized rows each threadgroup forms from RMS partials of its own;
//     one more threadgroup forms the shared-expert gate column; the last
//     threadgroup to arrive selects.
//   routed_route_gemm (M > 8): 32 rows x 32 columns per threadgroup on
//     8x8 F32 simdgroup matrices over staged tiles.
//   Logits are [M, E + 1]: column E is the shared-expert gate.
//   routed_route_select (one simdgroup per row): the softmax over every expert
//     (E / 32 experts per lane), K rounds of simdgroup argmax (ties to the
//     higher expert index), the optional renormalization and the
//     shared-expert coefficient from column E.
// Every logit accumulates over the hidden axis in the same order for every
// expert, so experts with equal router rows tie exactly.

#define KERNEL_W0 SEISMIC_ROUTER
#include "lib/routed/route.h"

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

// M <= 8, the only launch: the router threadgroups of `lib/routed/route.h`.
kernel void routed_route_gemv(
    device const float *residual [[buffer(SEISMIC_BUFFER_RESIDUAL)]],
    device const uchar *norm [[buffer(SEISMIC_BUFFER_NORM)]],
    device const uchar *router [[buffer(SEISMIC_BUFFER_ROUTER)]],
    device const float *shared_router [[buffer(SEISMIC_BUFFER_SHARED_ROUTER)]],
    device int *routes [[buffer(SEISMIC_BUFFER_ROUTES)]],
    device float *scores [[buffer(SEISMIC_BUFFER_SCORES)]],
    device uchar *normalized [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    device float *coefficient [[buffer(SEISMIC_RESULT_1_BUFFER)]],
    device float *logits [[buffer(SEISMIC_BUFFER_SCRATCH_LOGITS)]],
    device atomic_uint *arrivals [[buffer(SEISMIC_BUFFER_SCRATCH_ARRIVALS)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    threadgroup uchar *shared [[threadgroup(0)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float parts[8 * route_simdgroups];
    threadgroup float inverses[8];
    threadgroup uint last;
    const routing::Rows<Norm> in = decode_rows(residual, norm, parts, inverses, seismic_words, simd, lane, tid);
    route_group(in, router, shared_router, routes, scores, normalized, coefficient, logits, arrivals, seismic_words,
        shared, &last, group, tid, simd, lane);
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
    projection::Weights<packets::W0> router_rows{router, KERNEL_W0_LAYOUT(uint(SEISMIC_DIM_H)),
        uint(SEISMIC_DIM_H), nullptr};
    const ulong expert0 = ulong(group.x) * route_tile;
    const ulong row0 = ulong(group.y) * route_tile;
    simdgroup_float8x8 accumulators[4];
    for (uint j = 0; j < 4; ++j)
        accumulators[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (ulong k0 = 0; k0 < SEISMIC_DIM_H; k0 += route_depth) {
        // A multiple of 8 (H % 32 == 0).
        const uint depth = uint(metal::min(ulong(route_depth), SEISMIC_DIM_H - k0));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // rows_tile[r][k] = normalized[row0 + r][k0 + k].
        for (uint item = tid; item < route_tile * depth; item += 128) {
            const uint r = item / depth, k = item % depth;
            const ulong row = row0 + r;
            rows_tile[r * route_rows_pitch + k] = row < SEISMIC_DIM_M
                ? element::at<Act>(normalized,row * SEISMIC_RESULT_0_STRIDE_0 + (k0 + k) * SEISMIC_RESULT_0_STRIDE_1)
                : 0.0f;
        }
        // One thread decodes a packet for one expert column, publishing all
        // 32 weights of that packet into the matrix tile.
        for (uint item = tid; item < route_tile * (depth / 32); item += 128) {
            const uint r = item / (depth / 32), packet_in_tile = item % (depth / 32);
            const ulong expert = expert0 + r, source0 = k0 + 32ul * packet_in_tile;
            if (expert < SEISMIC_DIM_E) {
                typename packets::W0::packet packet = router_rows.packet(uint(expert), uint(source0 / 32));
                for (uint step = 0; step < 4; ++step) {
                    float4 even, odd;
                    packets::W0::codes(packet, step, even, odd);
                    for (uint i = 0; i < 8; ++i) {
                        const float code = (i & 1u) ? odd[i >> 1] : even[i >> 1];
                        experts_tile[(32 * packet_in_tile + 8 * step + i) * route_pitch + r] =
                            packets::W0::value(packet, step, code);
                    }
                }
            } else {
                for (uint i = 0; i < 32; ++i)
                    experts_tile[(32 * packet_in_tile + i) * route_pitch + r] = expert == SEISMIC_DIM_E
                        ? shared_router[(source0 + i) * SEISMIC_SHARED_ROUTER_STRIDE_0] : 0.0f;
            }
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

// M > 8: one simdgroup per row, after the GEMM. (M <= 8 selects in the last
// GEMV threadgroup to arrive.)
kernel void routed_route_select(
    device int *routes [[buffer(SEISMIC_BUFFER_ROUTES)]],
    device float *scores [[buffer(SEISMIC_BUFFER_SCORES)]],
    device float *coefficient [[buffer(SEISMIC_RESULT_1_BUFFER)]],
    device const float *logits [[buffer(SEISMIC_BUFFER_SCRATCH_LOGITS)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const ulong row = ulong(group) * route_simdgroups + simd;
    if (row >= SEISMIC_DIM_M)
        return;
    select_row(row, [&](ulong at) { return logits[at]; }, routes, scores, coefficient, seismic_words, lane);
}
