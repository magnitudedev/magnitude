// Expert streaming's decode routing (`routed_route_streamed` in
// routed.seismic): `routed_route_shared`'s launch, whose last router
// threadgroup to arrive selects, and the next two layers' hints (AHEAD 2).
//   [0, route_groups)               the router threadgroups
//                                   (`lib/routed/route.h`).
//   [.., + AHEAD * predict_groups)  per lead (one layer ahead, then two),
//                                   predict_groups threadgroups of that
//                                   layer's router GEMV (EN experts) over
//                                   this layer's residual normed by that
//                                   layer's norm, into the lead's [M, EN +
//                                   1] logits after this layer's [M, E + 1];
//                                   the last of a lead's to arrive ranks
//                                   its rows and publishes its hints into
//                                   the lead's ring row
//                                   (`streaming/hint.hpp`).
//   [.., + shared_groups)           the shared expert's gate/up threadgroups.
// The hints dispatch before the shared expert, so their serial tail (the
// arrival, ranking and publication) runs beside the shared product rather
// than after it. While the next layer's group is whole the launch encoded is
// the `_alone` kernel, the same body compiled without the hint threadgroups
// or their code.
// Every threadgroup forms the RMS inverses as `routed_route_shared` does, so
// the routing and the shared product carry its bits. The three groups count
// their arrivals apart and none waits for another.
//
// The selecting threadgroup then resolves (`streaming/resolve.hpp`): a
// layer of a group that is not whole (`routed_route_streamed`) its sources
// by the step-stamped map, its misses' assignment between the GPU and the
// CPU, routing record, grids and, on a miss, the request; a layer of a
// whole group (`routed_route_whole`) its hits, routing record and grids
// alone.

#define KERNEL_W0 SEISMIC_ROUTER
#define KERNEL_W1 SEISMIC_SHARED_GATE
#define KERNEL_W2 SEISMIC_SHARED_UP
#include "lib/routed/route.h"
#include "lib/routed/shared.h"
#include "lib/streaming/control.h"
#include "../streaming/hint.hpp"
#include "../streaming/resolve.hpp"

// The next streamed layer's experts (EN, static) and the threadgroups of its
// router GEMV.
constant constexpr uint next_experts = uint(SEISMIC_DIM_EN);
constant constexpr uint predict_groups = (next_experts + SEISMIC_TUNE_SIMDGROUPS - 1) / SEISMIC_TUNE_SIMDGROUPS;

// The ports of one lead's prediction: that layer's norm (and its stride)
// and router.
struct Lead {
    device const uchar *norm;
    ulong norm_stride;
    device const uchar *router;
};

// Hint threadgroup `group` (of predict_groups) of lead `lead` (0: one layer
// ahead, 1: two): that layer's router GEMV over this residual's inverses
// with its norm, then, in the last of the lead's threadgroups to arrive, its
// rows ranked and the hints published into the lead's ring row. Its choices
// and keep flags take `shared` once the GEMV is done with it
// (`shared_bytes` holds both).
inline void hint_group(device const float *residual, thread const Lead &ahead, uint lead,
    SYSTEM int *rings, int sequence,
    arrive::coherent_device<float> logits, device atomic_uint *arrivals, threadgroup uchar *shared,
    threadgroup float *inverses, threadgroup uint *last, constant ulong *seismic_words, uint group, uint tid,
    uint simd, uint lane) {
    const uint rows = uint(SEISMIC_DIM_M);
    threadgroup int *words = reinterpret_cast<threadgroup int *>(shared);
    const RouteRows predicted_in{residual, SEISMIC_RESIDUAL_STRIDE_0, SEISMIC_RESIDUAL_STRIDE_1, ahead.norm,
        ahead.norm_stride, inverses, uint(SEISMIC_DIM_H)};
    arrive::coherent_device<float> predicted = logits + ulong(SEISMIC_DIM_M) * (SEISMIC_DIM_E + 1)
        + ulong(lead) * SEISMIC_DIM_M * (SEISMIC_DIM_EN + 1);
    router_columns<next_experts>(predicted_in, ahead.router, predicted, rows, group, shared, simd, lane);
    // The arrival's barrier also ends every thread's use of `shared`.
    if (!arrive::last(arrivals + 1 + lead, predict_groups, last, tid))
        return;
    const auto logit = [&](ulong at) { return predicted[at]; };
    for (uint row = simd; row < rows; row += SEISMIC_TUNE_SIMDGROUPS) {
        const Ranked ranked = rank_row<next_experts>(row, logit, lane);
        if (lane == 0) {
            PROJECTION_UNROLL
            for (uint r = 0; r < SEISMIC_DIM_K; ++r)
                words[STREAMING_HINT_CHOSEN(row) + r] = ranked.winners[r];
        }
    }
    streaming_hint(rows, rings + ulong(lead) * SEISMIC_RINGS_STRIDE_0, SEISMIC_RINGS_STRIDE_1, sequence, words, tid,
        32 * SEISMIC_TUNE_SIMDGROUPS);
}

// The threadgroups past the router's, which both launch bodies share: the
// next layer's hint threadgroups (`HINTS`), then the shared expert's. True
// when `group` is one of them, its work done; false for a router
// threadgroup.
template <bool HINTS>
inline bool side_group(thread const RouteRows &in, device const float *residual, device const uchar *shared_gate,
    device const uchar *shared_up, device const int *step, thread const Lead (&leads)[2],
    SYSTEM int *rings, device uchar *shared_product,
    arrive::coherent_device<float> logits, device atomic_uint *arrivals, constant ulong *seismic_words,
    threadgroup uchar *shared, threadgroup float *globals, threadgroup float *inverses, threadgroup uint *last,
    uint group, uint tid, uint simd, uint lane) {
    if (group < route_groups)
        return false;
    const uint predicting = HINTS ? uint(SEISMIC_DIM_AHEAD) * predict_groups : 0;
    if (HINTS && group < route_groups + predicting) {
        const uint lead = (group - route_groups) / predict_groups;
        hint_group(residual, leads[lead], lead, rings, step[0], logits, arrivals, shared, inverses,
            last, seismic_words, (group - route_groups) % predict_groups, tid, simd, lane);
    }
    else
        shared_group(in, shared_gate, shared_up, shared_product, seismic_words, group - route_groups - predicting,
            globals, shared, simd, lane);
    return true;
}

// The bank's words the resolve reads and writes (`streaming/resolve.hpp`).
struct ResolvePorts {
    SYSTEM const int *map;
    SYSTEM const int *tier;
    device int *sources;
    SYSTEM int *file;
    device int *grids;
    SYSTEM const int *policy;
};

// The launch's body, with the next layer's hint threadgroups (`HINTS`) or
// without them; its selecting threadgroup resolves, a whole group's layer
// (`WHOLE`) its sources and routing record alone.
template <bool HINTS, bool WHOLE>
inline void route_streamed(device const float *residual, device const uchar *norm, device const uchar *router,
    device const float *shared_router, device const uchar *shared_gate, device const uchar *shared_up,
    device int *routes, device float *scores, device const float *expert_scale, thread const ResolvePorts &bank,
    device const int *step, thread const Lead (&leads)[2],
    SYSTEM int *rings, SYSTEM int *routing, device uchar *normalized,
    device float *coefficient, device uchar *shared_product, arrive::coherent_device<float> logits,
    device atomic_uint *arrivals, constant ulong *seismic_words, threadgroup uchar *shared,
    threadgroup float *globals, threadgroup float *parts, threadgroup float *inverses, threadgroup uint *last,
    threadgroup int *words, uint group, uint tid, uint simd, uint lane) {
    const RouteRows in = decode_rows(residual, norm, parts, inverses, seismic_words, simd, lane, tid);
    if (side_group<HINTS>(in, residual, shared_gate, shared_up, step, leads, rings,
            shared_product, logits, arrivals, seismic_words, shared, globals, inverses, last, group, tid, simd, lane))
        return;
    route_group(in, router, shared_router, expert_scale, routes, scores, normalized, coefficient, logits, arrivals,
        seismic_words, shared, last, group, tid, simd, lane);
    if (*last == 0u)
        return;
    // Resolve: the selection's routes are this threadgroup's own stores.
    streaming_resolve(routes, SEISMIC_ROUTES_STRIDE_0, SEISMIC_ROUTES_STRIDE_1, uint(SEISMIC_DIM_M) * uint(SEISMIC_DIM_K),
        step, SEISMIC_STEP_STRIDE_0, WHOLE, bank.map, SEISMIC_MAP_STRIDE_0, SEISMIC_MAP_STRIDE_1, bank.tier,
        SEISMIC_TIER_STRIDE_0, uint(SEISMIC_DIM_TM), uint(SEISMIC_PARAM_SITED),
        bank.sources, SEISMIC_SOURCES_STRIDE_0, SEISMIC_SOURCES_STRIDE_1, SEISMIC_SOURCES_STRIDE_2, routing,
        SEISMIC_ROUTING_STRIDE_0, SEISMIC_ROUTING_STRIDE_1, uint(SEISMIC_DIM_FLIGHT), uint(SEISMIC_DIM_P), bank.grids,
        SEISMIC_GRIDS_STRIDE_0, SEISMIC_GRIDS_STRIDE_1, bank.file, SEISMIC_FILE_STRIDE_0, SEISMIC_FILE_STRIDE_1,
        bank.policy, words,
        tid, 32 * SEISMIC_TUNE_SIMDGROUPS);
}

// The ports every launch of the entry binds and its kernels declare.
#define ROUTED_ROUTE_PORTS                                                                                         \
    device const float *residual [[buffer(SEISMIC_BUFFER_RESIDUAL)]],                                              \
        device const uchar *norm [[buffer(SEISMIC_BUFFER_NORM)]],                                                  \
        device const uchar *router [[buffer(SEISMIC_BUFFER_ROUTER)]],                                              \
        device const float *shared_router [[buffer(SEISMIC_BUFFER_SHARED_ROUTER)]],                                \
        device const uchar *shared_gate [[buffer(SEISMIC_BUFFER_SHARED_GATE)]],                                    \
        device const uchar *shared_up [[buffer(SEISMIC_BUFFER_SHARED_UP)]],                                        \
        device int *routes [[buffer(SEISMIC_BUFFER_ROUTES)]], device float *scores [[buffer(SEISMIC_BUFFER_SCORES)]], \
        device const float *expert_scale [[buffer(SEISMIC_BUFFER_EXPERT_SCALE)]],                                  \
        device const int *step [[buffer(SEISMIC_BUFFER_STEP)]],                                                    \
        device const uchar *next_norm [[buffer(SEISMIC_BUFFER_NEXT_NORM)]],                                        \
        device const uchar *next_router [[buffer(SEISMIC_BUFFER_NEXT_ROUTER)]],                                    \
        device const uchar *after_norm [[buffer(SEISMIC_BUFFER_AFTER_NORM)]],                                      \
        device const uchar *after_router [[buffer(SEISMIC_BUFFER_AFTER_ROUTER)]],                                  \
        SYSTEM int *rings [[buffer(SEISMIC_BUFFER_RINGS)]], SYSTEM int *routing [[buffer(SEISMIC_BUFFER_ROUTING)]], \
        device uchar *normalized [[buffer(SEISMIC_RESULT_0_BUFFER)]],                                              \
        device float *coefficient [[buffer(SEISMIC_RESULT_1_BUFFER)]],                                             \
        device uchar *shared_product [[buffer(SEISMIC_RESULT_2_BUFFER)]],                                          \
        arrive::coherent_device<float> logits [[buffer(SEISMIC_BUFFER_SCRATCH_LOGITS)]],                           \
        device atomic_uint *arrivals [[buffer(SEISMIC_BUFFER_SCRATCH_ARRIVALS)]],                                  \
        constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]], threadgroup uchar *shared [[threadgroup(0)]], \
        uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],                     \
        uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]

// The launches' kernels: the shared ports, then the bank's words the
// resolve reads and writes.
#define ROUTED_ROUTE_STREAMED(NAME, HINTS, WHOLE)                                                                  \
    kernel void NAME(ROUTED_ROUTE_PORTS, SYSTEM const int *map [[buffer(SEISMIC_BUFFER_MAP)]],                     \
        SYSTEM const int *tier [[buffer(SEISMIC_BUFFER_TIER)]],                                                     \
        device int *sources [[buffer(SEISMIC_BUFFER_SOURCES)]],                                                     \
        SYSTEM int *file [[buffer(SEISMIC_BUFFER_FILE)]],                                                           \
        device int *grids [[buffer(SEISMIC_BUFFER_GRIDS)]], SYSTEM const int *policy [[buffer(SEISMIC_BUFFER_POLICY)]]) { \
        threadgroup float parts[SEISMIC_CONST_DECODE_ROWS * route_simdgroups];                                     \
        threadgroup float inverses[SEISMIC_CONST_DECODE_ROWS];                                                     \
        threadgroup uint last;                                                                                     \
        threadgroup int resolved[STREAMING_RESOLVE_WORDS(SEISMIC_CONST_DECODE_ROWS * SEISMIC_DIM_K)];              \
        PROJECTION_OPERANDS_SHARED(globals, shared_nvfp4);                                                         \
        const ResolvePorts bank{map, tier, sources, file, grids, policy};                                       \
        const Lead leads[2] = {{next_norm, SEISMIC_NEXT_NORM_STRIDE_0, next_router},                               \
            {after_norm, SEISMIC_AFTER_NORM_STRIDE_0, after_router}};                                              \
        route_streamed<HINTS, WHOLE>(residual, norm, router, shared_router, shared_gate, shared_up, routes,        \
            scores, expert_scale, bank, step, leads, rings, routing, normalized,        \
            coefficient, shared_product, logits, arrivals, seismic_words, shared, globals, parts, inverses, &last, \
            resolved, group, tid, simd, lane);                                                                     \
    }

ROUTED_ROUTE_STREAMED(routed_route_streamed, true, false)
ROUTED_ROUTE_STREAMED(routed_route_streamed_alone, false, false)
ROUTED_ROUTE_STREAMED(routed_route_whole, true, true)
ROUTED_ROUTE_STREAMED(routed_route_whole_alone, false, true)
