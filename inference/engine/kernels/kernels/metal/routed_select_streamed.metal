// Expert streaming's decode selection of the general routed form
// (`routed_select_streamed` in routed.seismic) in one launch:
// `routed_select`'s decode launch (`lib/routed/selection.h`), so its bits,
// then resolve and the next two layers' hints (AHEAD 2).
//   [0, columns_groups)             `routed_select_gemv`'s threadgroups; the
//                                   last to arrive selects every row, a
//                                   simdgroup a row (M <= DECODE_ROWS), then
//                                   resolves (`streaming/resolve.hpp`), or,
//                                   of a whole group's layer
//                                   (`routed_select_whole`), resolves its
//                                   hits and routing record alone.
//   [.., + AHEAD * next_groups)     per lead (one layer ahead, then two),
//                                   next_groups threadgroups forming that
//                                   layer's logits over this residual
//                                   through its router norm and router, into
//                                   the lead's [M, EN] plane of the scratch
//                                   after this layer's [M, E]; the last of a
//                                   lead's to arrive ranks the lead's
//                                   layer's rows as its selection ranks them
//                                   (`select::rank` with that layer's bias)
//                                   and publishes the hints into the lead's
//                                   ring row (`streaming/hint.hpp`).
// The router and each lead count their arrivals apart (`arrivals`, sync
// scratch) and none waits for another. While the next layer holds every
// expert, the launch is encoded without the lead threadgroups, which follow
// the router's.

#define KERNEL_W0 SEISMIC_ROUTER
#include "lib/routed/selection.h"
#include "lib/streaming/control.h"
#include "../streaming/hint.hpp"
#include "../streaming/resolve.hpp"

// E and EN (the next streamed layer's experts) are static.
constant constexpr uint columns_groups = (uint(SEISMIC_DIM_E) + SEISMIC_TUNE_SIMDGROUPS - 1) / SEISMIC_TUNE_SIMDGROUPS;
constant constexpr uint next_experts = uint(SEISMIC_DIM_EN);
constant constexpr uint next_groups = (next_experts + SEISMIC_TUNE_SIMDGROUPS - 1) / SEISMIC_TUNE_SIMDGROUPS;
constant constexpr uint launch_threads = 32 * SEISMIC_TUNE_SIMDGROUPS;

// The last of lead `lead`'s threadgroups: the hints of the layer `lead` + 1
// ahead, by its selection bias (`bias`, of stride `bias_stride`), through
// its group words (`STREAMING_HINT_WORDS`) into the lead's ring row.
inline void hint_group(arrive::coherent_device<float> logits, uint lead, device const float *bias,
    ulong bias_stride, SYSTEM int *rings, int sequence, threadgroup int *words, constant ulong *seismic_words,
    uint tid, uint simd, uint lane) {
    constexpr uint K = SEISMIC_DIM_K;
    const uint rows = uint(SEISMIC_DIM_M);
    arrive::coherent_device<float> predicted = logits + ulong(rows) * SEISMIC_DIM_E + ulong(lead) * rows * next_experts;
    for (uint row = simd; row < rows; row += SEISMIC_TUNE_SIMDGROUPS) {
        const select::Selected<K> selected = select::rank<next_experts, K>(int(SEISMIC_PARAM_SCORE),
            [&](uint expert) { return predicted[ulong(row) * next_experts + expert]; },
            [&](int expert) { return bias[ulong(expert) * bias_stride]; }, lane);
        if (lane == 0) {
            PROJECTION_UNROLL
            for (uint r = 0; r < K; ++r)
                words[STREAMING_HINT_CHOSEN(row) + r] = selected.winners[K - 1 - r];
        }
    }
    streaming_hint(rows, rings + ulong(lead) * SEISMIC_RINGS_STRIDE_0, SEISMIC_RINGS_STRIDE_1, sequence, words, tid,
        launch_threads);
}

// The router's threadgroups select and resolve (WHOLE: its sources and
// routing record alone); a lead's publish its hints.
#define ROUTED_SELECT_STREAMED(NAME, WHOLE)                                                                        \
    kernel void NAME(device const float *residual [[buffer(SEISMIC_BUFFER_RESIDUAL)]],                            \
        device const uchar *norm [[buffer(SEISMIC_BUFFER_NORM)]],                                                  \
        device const uchar *router_norm [[buffer(SEISMIC_BUFFER_ROUTER_NORM)]],                                    \
        device const uchar *router [[buffer(SEISMIC_BUFFER_ROUTER)]],                                              \
        device const uchar *next_router_norm [[buffer(SEISMIC_BUFFER_NEXT_ROUTER_NORM)]],                          \
        device const uchar *next_router [[buffer(SEISMIC_BUFFER_NEXT_ROUTER)]],                                    \
        device const uchar *after_router_norm [[buffer(SEISMIC_BUFFER_AFTER_ROUTER_NORM)]],                        \
        device const uchar *after_router [[buffer(SEISMIC_BUFFER_AFTER_ROUTER)]],                                  \
        device const float *bias [[buffer(SEISMIC_BUFFER_BIAS)]],                                                  \
        device const float *expert_scale [[buffer(SEISMIC_BUFFER_EXPERT_SCALE)]],                                  \
        device int *routes [[buffer(SEISMIC_BUFFER_ROUTES)]], device float *weights [[buffer(SEISMIC_BUFFER_WEIGHTS)]], \
        device const int *step [[buffer(SEISMIC_BUFFER_STEP)]],                                                    \
        device const float *next_bias [[buffer(SEISMIC_BUFFER_NEXT_BIAS)]],                                        \
        device const float *after_bias [[buffer(SEISMIC_BUFFER_AFTER_BIAS)]],                                      \
        SYSTEM int *rings [[buffer(SEISMIC_BUFFER_RINGS)]], SYSTEM int *routing [[buffer(SEISMIC_BUFFER_ROUTING)]], \
        SYSTEM const int *map [[buffer(SEISMIC_BUFFER_MAP)]], SYSTEM const int *tier [[buffer(SEISMIC_BUFFER_TIER)]], \
        device int *sources [[buffer(SEISMIC_BUFFER_SOURCES)]], SYSTEM int *file [[buffer(SEISMIC_BUFFER_FILE)]],  \
        SYSTEM const int *policy [[buffer(SEISMIC_BUFFER_POLICY)]],                                                \
        device uchar *normalized [[buffer(SEISMIC_RESULT_0_BUFFER)]],                                              \
        arrive::coherent_device<float> logits [[buffer(SEISMIC_BUFFER_SCRATCH_LOGITS)]],                           \
        device atomic_uint *arrivals [[buffer(SEISMIC_BUFFER_SCRATCH_ARRIVALS)]],                                  \
        constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]], threadgroup uchar *shared [[threadgroup(0)]], \
        uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],                     \
        uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {                   \
        constexpr uint K = SEISMIC_DIM_K;                                                                          \
        const uint rows = uint(SEISMIC_DIM_M);                                                                     \
        threadgroup float parts[SEISMIC_CONST_DECODE_ROWS * select_simdgroups];                                    \
        threadgroup float inverses[SEISMIC_CONST_DECODE_ROWS];                                                     \
        threadgroup uint last;                                                                                     \
        /* The resolve's or the hints' group words take `shared` once the GEMV is done with it (the arrival's  \
           barrier); `shared_bytes` holds both. */                                                                \
        threadgroup int *words = reinterpret_cast<threadgroup int *>(shared);                                      \
        decode_inverses(residual, parts, inverses, seismic_words, simd, lane, tid);                                \
        if (group >= columns_groups) {                                                                             \
            const uint lead = (group - columns_groups) / next_groups;                                              \
            const uint tile = (group - columns_groups) % next_groups;                                              \
            arrive::coherent_device<float> predicted =                                                             \
                logits + ulong(rows) * SEISMIC_DIM_E + ulong(lead) * rows * next_experts;                          \
            if (lead == 0)                                                                                         \
                select_columns<next_experts>(residual, next_router_norm, SEISMIC_NEXT_ROUTER_NORM_STRIDE_0,        \
                    next_router, predicted, inverses, tile, shared, seismic_words, simd, lane);                    \
            else                                                                                                   \
                select_columns<next_experts>(residual, after_router_norm, SEISMIC_AFTER_ROUTER_NORM_STRIDE_0,      \
                    after_router, predicted, inverses, tile, shared, seismic_words, simd, lane);                   \
            if (!arrive::last(arrivals + 1 + lead, next_groups, &last, tid))                                       \
                return;                                                                                            \
            if (lead == 0)                                                                                         \
                hint_group(logits, 0, next_bias, SEISMIC_NEXT_BIAS_STRIDE_0, rings, step[0], words, seismic_words, \
                    tid, simd, lane);                                                                              \
            else                                                                                                   \
                hint_group(logits, 1, after_bias, SEISMIC_AFTER_BIAS_STRIDE_0, rings, step[0], words,              \
                    seismic_words, tid, simd, lane);                                                               \
            return;                                                                                                \
        }                                                                                                          \
        if (group == 0)                                                                                            \
            publish_normalized(residual, norm, normalized, inverses, seismic_words, tid);                          \
        select_columns(residual, router_norm, SEISMIC_ROUTER_NORM_STRIDE_0, router, logits, inverses, group,       \
            shared, seismic_words, simd, lane);                                                                    \
        if (!arrive::last(arrivals, columns_groups, &last, tid))                                                   \
            return;                                                                                                \
        for (uint row = simd; row < rows; row += SEISMIC_TUNE_SIMDGROUPS)                                          \
            select_row(row, logits, bias, expert_scale, routes, weights, seismic_words, lane);                     \
        /* Resolve: the selection's routes are this threadgroup's own stores. */                                   \
        streaming_resolve(routes, SEISMIC_ROUTES_STRIDE_0, SEISMIC_ROUTES_STRIDE_1, rows * K, step,                \
            SEISMIC_STEP_STRIDE_0, WHOLE, map, SEISMIC_MAP_STRIDE_0, SEISMIC_MAP_STRIDE_1, tier,                   \
            SEISMIC_TIER_STRIDE_0, uint(SEISMIC_DIM_TM), uint(SEISMIC_PARAM_SITED),                                \
            sources, SEISMIC_SOURCES_STRIDE_0, SEISMIC_SOURCES_STRIDE_1, SEISMIC_SOURCES_STRIDE_2, routing,         \
            SEISMIC_ROUTING_STRIDE_0, SEISMIC_ROUTING_STRIDE_1, uint(SEISMIC_DIM_FLIGHT), uint(SEISMIC_DIM_P),       \
            file, SEISMIC_FILE_STRIDE_0,                                                                           \
            SEISMIC_FILE_STRIDE_1, policy,                                                                         \
            words, tid, launch_threads);                                                                           \
    }

ROUTED_SELECT_STREAMED(routed_select_streamed, false)
ROUTED_SELECT_STREAMED(routed_select_whole, true)
