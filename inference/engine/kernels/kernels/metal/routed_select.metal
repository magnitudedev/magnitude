// General expert selection for M rows (the `routed_route` structure without
// the shared-expert column):
//   routed_select_stage (M > DECODE_ROWS, one threadgroup per row): the row's RMS and
//     its two normalized forms, published once: `normalized` (norm, the
//     experts' input) and the router input (router_norm) into scratch.
//   routed_select_gemv (M <= DECODE_ROWS, the decode's one launch): the
//     router logits (`lib/routed/router.h`), one simdgroup per logit column
//     over the router input each threadgroup forms from RMS partials of its
//     own; threadgroup 0 publishes `normalized`; the last threadgroup to
//     arrive (`arrivals`, sync scratch) selects every row.
//   routed_select_gemm (M > DECODE_ROWS): the logits tiles of `lib/routed/logits.h`
//     over the staged router input.
//   routed_select_select (M > DECODE_ROWS, one simdgroup per row): scores,
//     biased ranking, single-pass top-K (`lib/routed/select.h`),
//     normalization, scales.
// The decode launches' bodies are `lib/routed/selection.h`'s, which
// `routed_select_streamed` shares.
// Every logit accumulates over the hidden axis in the same order for every
// expert, so experts with equal router rows tie exactly.

#define KERNEL_W0 SEISMIC_ROUTER
#include "lib/routed/selection.h"
#include "lib/routed/logits.h"

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

// M <= DECODE_ROWS, the whole selection in one launch: the threadgroup
// computes its rows' RMS inverses (in the stage launch's order); threadgroup
// 0 publishes the normalized rows; each threadgroup then forms the logits of
// SIMDGROUPS router rows over the router input (`routing::Gemv`); the last
// to arrive selects every row, a simdgroup a row.
kernel void routed_select_gemv(
    device const float *residual [[buffer(SEISMIC_BUFFER_RESIDUAL)]],
    device const uchar *norm [[buffer(SEISMIC_BUFFER_NORM)]],
    device const uchar *router_norm [[buffer(SEISMIC_BUFFER_ROUTER_NORM)]],
    device const uchar *router [[buffer(SEISMIC_BUFFER_ROUTER)]],
    device const float *bias [[buffer(SEISMIC_BUFFER_BIAS)]],
    device const float *expert_scale [[buffer(SEISMIC_BUFFER_EXPERT_SCALE)]],
    device int *routes [[buffer(SEISMIC_BUFFER_ROUTES)]],
    device float *weights [[buffer(SEISMIC_BUFFER_WEIGHTS)]],
    device uchar *normalized [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    arrive::coherent_device<float> logits [[buffer(SEISMIC_BUFFER_SCRATCH_LOGITS)]],
    device atomic_uint *arrivals [[buffer(SEISMIC_BUFFER_SCRATCH_ARRIVALS)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    threadgroup uchar *shared [[threadgroup(0)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float parts[SEISMIC_CONST_DECODE_ROWS * select_simdgroups];
    threadgroup float inverses[SEISMIC_CONST_DECODE_ROWS];
    threadgroup uint last;
    decode_inverses(residual, parts, inverses, seismic_words, simd, lane, tid);
    if (group == 0)
        publish_normalized(residual, norm, normalized, inverses, seismic_words, tid);
    select_columns(residual, router_norm, SEISMIC_ROUTER_NORM_STRIDE_0, router, logits, inverses, group, shared,
        seismic_words, simd, lane);
    if (!arrive::last(arrivals, (uint(SEISMIC_DIM_E) + SEISMIC_TUNE_SIMDGROUPS - 1) / SEISMIC_TUNE_SIMDGROUPS, &last,
            tid))
        return;
    for (uint row = simd; row < uint(SEISMIC_DIM_M); row += SEISMIC_TUNE_SIMDGROUPS)
        select_row(row, logits, bias, expert_scale, routes, weights, seismic_words, lane);
}

// M > DECODE_ROWS: the logits tiles (`lib/routed/logits.h`) over the staged router
// input.
kernel void routed_select_gemm(
    device const uchar *router [[buffer(SEISMIC_BUFFER_ROUTER)]],
    device const uchar *router_rows [[buffer(SEISMIC_BUFFER_SCRATCH_ROUTER_ROWS)]],
    device float *logits [[buffer(SEISMIC_BUFFER_SCRATCH_LOGITS)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint H = SEISMIC_DIM_H;
    threadgroup float shared[logits::extent];
    const logits::RouterColumns columns{{router, KERNEL_W0_LAYOUT(H), H, nullptr}};
    logits::tile<H>(SEISMIC_DIM_M, SEISMIC_DIM_E, ulong(group.y) * logits::rows_per_tile,
        ulong(group.x) * logits::columns_per_tile,
        [&](ulong row, ulong k) { return element::at<Act>(router_rows, row * H + k); }, columns,
        [&](ulong row, ulong expert, float value) { logits[row * SEISMIC_DIM_E + expert] = value; }, shared, tid,
        simd, lane);
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
    const ulong row = ulong(group) * select_simdgroups + simd;
    if (row >= SEISMIC_DIM_M)
        return;
    select_row(row, logits, bias, expert_scale, routes, weights, seismic_words, lane);
}
