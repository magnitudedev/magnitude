// Routing for M rows: one launch at M <= DECODE_ROWS (gemv), three above (stage, gemm,
// select).
//   routed_route_stage (one threadgroup per row): the row's RMS and its
//     normalized values (rounded to the activation dtype), published once.
//   routed_route_gemv (M <= DECODE_ROWS): the router logits (`lib/routed/router.h`,
//     `lib/routed/route.h`), one simdgroup per logit column over the
//     normalized rows each threadgroup forms from RMS partials of its own;
//     one more threadgroup forms the shared-expert gate column; the last
//     threadgroup to arrive selects.
//   routed_route_gemm (M > DECODE_ROWS): the logits tiles of `lib/routed/logits.h`.
//   Logits are [M, E + 1]: column E is the shared-expert gate.
//   routed_route_select (one simdgroup per row): the softmax over every expert
//     (ceil(E / 32) experts per lane), K rounds of simdgroup argmax (ties to the
//     higher expert index), the optional renormalization and the
//     shared-expert coefficient from column E.
// Every logit accumulates over the hidden axis in the same order for every
// expert, so experts with equal router rows tie exactly.

#define KERNEL_W0 SEISMIC_ROUTER
#include "lib/routed/route.h"
#include "lib/routed/logits.h"

kernel void routed_route_stage(
    device const float *residual [[buffer(SEISMIC_BUFFER_RESIDUAL)]],
    device const uchar *norm [[buffer(SEISMIC_BUFFER_NORM)]],
    device uchar *normalized [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint row_index [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const ulong row = ulong(row_index);
    threadgroup float partials[route_simdgroups];
    if (SEISMIC_DIM_NM == 0) {
        // `form_input` without a norm: the sum in part order of the partials.
        for (uint source = tid; source < SEISMIC_DIM_H; source += route_threads) {
            float total = 0.0f;
            for (ulong part = 0; part < SEISMIC_DIM_PT; ++part)
                total += residual[row * SEISMIC_RESIDUAL_STRIDE_0
                    + (part * SEISMIC_DIM_H + source) * SEISMIC_RESIDUAL_STRIDE_1];
            element::put<Act>(normalized, row * SEISMIC_RESULT_0_STRIDE_0 + source * SEISMIC_RESULT_0_STRIDE_1,
                Act::round(total));
        }
        return;
    }
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

// M <= DECODE_ROWS, the only launch: the router threadgroups of `lib/routed/route.h`.
kernel void routed_route_gemv(
    device const float *residual [[buffer(SEISMIC_BUFFER_RESIDUAL)]],
    device const uchar *norm [[buffer(SEISMIC_BUFFER_NORM)]],
    device const uchar *router [[buffer(SEISMIC_BUFFER_ROUTER)]],
    device const float *shared_router [[buffer(SEISMIC_BUFFER_SHARED_ROUTER)]],
    device int *routes [[buffer(SEISMIC_BUFFER_ROUTES)]],
    device float *scores [[buffer(SEISMIC_BUFFER_SCORES)]],
    device const float *expert_scale [[buffer(SEISMIC_BUFFER_EXPERT_SCALE)]],
    device uchar *normalized [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    device float *coefficient [[buffer(SEISMIC_RESULT_1_BUFFER)]],
    arrive::coherent_device<float> logits [[buffer(SEISMIC_BUFFER_SCRATCH_LOGITS)]],
    device atomic_uint *arrivals [[buffer(SEISMIC_BUFFER_SCRATCH_ARRIVALS)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    threadgroup uchar *shared [[threadgroup(0)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float parts[SEISMIC_CONST_DECODE_ROWS * route_simdgroups];
    threadgroup float inverses[SEISMIC_CONST_DECODE_ROWS];
    threadgroup uint last;
    const RouteRows in = decode_rows(residual, norm, parts, inverses, seismic_words, simd, lane, tid);
    route_group(in, router, shared_router, expert_scale, routes, scores, normalized, coefficient, logits, arrivals,
        seismic_words, shared, &last, group, tid, simd, lane);
}

// The GEMM's logit columns: the router rows, then the shared-expert gate row
// (dense F32, read as it is written).
struct RouteColumns {
    typedef logits::RouterColumns::fetched fetched;
    logits::RouterColumns rows;
    device const float *shared_router;
    fetched fetch(ulong expert, ulong k0) const {
        return expert < SEISMIC_DIM_E ? rows.fetch(expert, k0) : fetched{};
    }
    void write(ulong expert, ulong k0, thread const fetched &packet, threadgroup float *out) const {
        if (expert < SEISMIC_DIM_E)
            rows.write(expert, k0, packet, out);
        else
            for (uint i = 0; i < 32; ++i)
                out[i] = shared_router[(k0 + i) * SEISMIC_SHARED_ROUTER_STRIDE_0];
    }
};

// M > DECODE_ROWS: the logits tiles (`lib/routed/logits.h`) over the E + 1 columns,
// column E the shared-expert gate.
kernel void routed_route_gemm(
    device const uchar *router [[buffer(SEISMIC_BUFFER_ROUTER)]],
    device const float *shared_router [[buffer(SEISMIC_BUFFER_SHARED_ROUTER)]],
    device const uchar *normalized [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    arrive::coherent_device<float> logits [[buffer(SEISMIC_BUFFER_SCRATCH_LOGITS)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint H = SEISMIC_DIM_H;
    threadgroup float shared[logits::extent];
    const RouteColumns columns{{{router, KERNEL_W0_LAYOUT(H), H, nullptr}}, shared_router};
    logits::tile<H>(SEISMIC_DIM_M, SEISMIC_DIM_E + 1, ulong(group.y) * logits::rows_per_tile,
        ulong(group.x) * logits::columns_per_tile,
        [&](ulong row, ulong k) {
            return element::at<Act>(normalized, row * SEISMIC_RESULT_0_STRIDE_0 + k * SEISMIC_RESULT_0_STRIDE_1);
        },
        columns, [&](ulong row, ulong expert, float value) { logits[logit_index(row, expert)] = value; }, shared, tid,
        simd, lane);
}

// M > DECODE_ROWS: one simdgroup per row, after the GEMM. (M <= DECODE_ROWS selects in the last
// GEMV threadgroup to arrive.)
kernel void routed_route_select(
    device int *routes [[buffer(SEISMIC_BUFFER_ROUTES)]],
    device float *scores [[buffer(SEISMIC_BUFFER_SCORES)]],
    device const float *expert_scale [[buffer(SEISMIC_BUFFER_EXPERT_SCALE)]],
    device float *coefficient [[buffer(SEISMIC_RESULT_1_BUFFER)]],
    device const float *logits [[buffer(SEISMIC_BUFFER_SCRATCH_LOGITS)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const ulong row = ulong(group) * route_simdgroups + simd;
    if (row >= SEISMIC_DIM_M)
        return;
    select_row(row, [&](ulong at) { return logits[at]; }, expert_scale, routes, scores, coefficient, seismic_words,
        lane);
}
