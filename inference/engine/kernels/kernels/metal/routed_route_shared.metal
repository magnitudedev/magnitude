// Decode routing (M <= DECODE_ROWS) with the shared expert's gate/up in one launch.
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
#include "lib/routed/shared.h"

kernel void routed_route_shared(
    device const float *residual [[buffer(SEISMIC_BUFFER_RESIDUAL)]],
    device const uchar *norm [[buffer(SEISMIC_BUFFER_NORM)]],
    device const uchar *router [[buffer(SEISMIC_BUFFER_ROUTER)]],
    device const float *shared_router [[buffer(SEISMIC_BUFFER_SHARED_ROUTER)]],
    device const uchar *shared_gate [[buffer(SEISMIC_BUFFER_SHARED_GATE)]],
    device const uchar *shared_up [[buffer(SEISMIC_BUFFER_SHARED_UP)]],
    device int *routes [[buffer(SEISMIC_BUFFER_ROUTES)]],
    device float *scores [[buffer(SEISMIC_BUFFER_SCORES)]],
    device const float *expert_scale [[buffer(SEISMIC_BUFFER_EXPERT_SCALE)]],
    device uchar *normalized [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    device float *coefficient [[buffer(SEISMIC_RESULT_1_BUFFER)]],
    device uchar *shared_product [[buffer(SEISMIC_RESULT_2_BUFFER)]],
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
    if (group < route_groups) {
        route_group(in, router, shared_router, expert_scale, routes, scores, normalized, coefficient, logits,
            arrivals, seismic_words, shared, &last, group, tid, simd, lane);
        return;
    }
    PROJECTION_OPERANDS_SHARED(globals, shared_nvfp4);
    shared_group(in, shared_gate, shared_up, shared_product, seismic_words, group - route_groups, globals, shared, simd, lane);
}
