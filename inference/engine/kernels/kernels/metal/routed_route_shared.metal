// Decode routing (M <= 8) with the shared expert's gate/up in one launch.
// Threadgroups [0, route_groups) are `routed_route`'s router threadgroups
// (`lib/routed/route.h`), whose selection counts only them. The shared
// threadgroups after them each form the normalized rows as a router
// threadgroup does (`decode_rows`: the same RMS sum order and rounding as
// the published `normalized`), then run `routed_expand`'s shared row: a K1
// paired GEMV of ROWS weight rows per lane group of LANES lanes over the
// shared expert's gate and up rows, with the SiLU . mul epilogue.

#define KERNEL_W0 SEISMIC_ROUTER
#define KERNEL_W1 SEISMIC_SHARED_GATE
#define KERNEL_W2 SEISMIC_SHARED_UP
#include "lib/routed/route.h"

kernel void routed_route_shared(
    device const float *residual [[buffer(SEISMIC_BUFFER_RESIDUAL)]],
    device const uchar *norm [[buffer(SEISMIC_BUFFER_NORM)]],
    device const uchar *router [[buffer(SEISMIC_BUFFER_ROUTER)]],
    device const float *shared_router [[buffer(SEISMIC_BUFFER_SHARED_ROUTER)]],
    device const uchar *shared_gate [[buffer(SEISMIC_BUFFER_SHARED_GATE)]],
    device const uchar *shared_up [[buffer(SEISMIC_BUFFER_SHARED_UP)]],
    device int *routes [[buffer(SEISMIC_BUFFER_ROUTES)]],
    device float *scores [[buffer(SEISMIC_BUFFER_SCORES)]],
    device uchar *normalized [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    device float *coefficient [[buffer(SEISMIC_RESULT_1_BUFFER)]],
    device uchar *shared_product [[buffer(SEISMIC_RESULT_2_BUFFER)]],
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
    if (group < route_groups) {
        route_group(in, router, shared_router, routes, scores, normalized, coefficient, logits, arrivals,
            seismic_words, shared, &last, group, tid, simd, lane);
        return;
    }
    const uint k = uint(SEISMIC_DIM_H);
    const projection::Weights<packets::W1> gate{shared_gate, KERNEL_W1_LAYOUT(k), k, nullptr};
    const projection::Weights<packets::W2> up{shared_up, KERNEL_W2_LAYOUT(k), k, nullptr};
    const projection::SiluMul<Act> out{shared_product, SEISMIC_RESULT_2_STRIDE_0, SEISMIC_RESULT_2_STRIDE_1};
    const uint rows = uint(SEISMIC_DIM_M);
    PROJECTION_FOR_ROWS(rows,
        (projection::gemv_paired_runtime<packets::W1, packets::W2, SEISMIC_TUNE_ROWS, MAXM, SEISMIC_TUNE_LANES>(in,
            out, gate, up, rows, SEISMIC_DIM_S, k, group - route_groups, shared, SEISMIC_TUNE_SIMDGROUPS, simd,
            lane)));
}
