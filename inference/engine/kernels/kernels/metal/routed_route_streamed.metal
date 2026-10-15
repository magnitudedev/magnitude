// Expert streaming, steps 1 and 2 (`routed_route_streamed` in routed.seismic):
// a streamed layer's routing, resolve and prefetch hints in one launch.
//   [0, route_groups)           `routed_route_shared`'s router threadgroups
//                               (`lib/routed/route.h`); the last of them to
//                               arrive selects, then resolves.
//   [.., + shared_groups)       its shared-expert gate/up threadgroups.
//   [.., + 2 * predict_groups)  the router GEMV of the near, then the far
//                               layer ahead, over this layer's residual with
//                               that layer's norm; the last of a distance's
//                               threadgroups to arrive selects its
//                               predictions and publishes the hints.
// Every threadgroup forms the RMS inverses as a router threadgroup does, so
// the routing and the shared product carry `routed_route_shared`'s bits. The
// three groups count their arrivals apart and none waits for another.

#define KERNEL_W0 SEISMIC_ROUTER
#define KERNEL_W1 SEISMIC_SHARED_GATE
#define KERNEL_W2 SEISMIC_SHARED_UP
#include "lib/routed/route.h"
#include "lib/streaming/control.h"

static_assert(element::Act::bytes == 2, "routed_route_streamed requires a bf16 or f16 activation element");

constant constexpr uint predict_groups = (uint(SEISMIC_DIM_E) + SEISMIC_TUNE_SIMDGROUPS - 1) / SEISMIC_TUNE_SIMDGROUPS;
constant constexpr uint shared_groups = (uint(SEISMIC_DIM_S) + SEISMIC_TUNE_SIMDGROUPS * SEISMIC_TUNE_ROWS * 32
        / SEISMIC_TUNE_LANES - 1) / (SEISMIC_TUNE_SIMDGROUPS * SEISMIC_TUNE_ROWS * 32 / SEISMIC_TUNE_LANES);
constant constexpr uint threads = 32 * SEISMIC_TUNE_SIMDGROUPS;
// A distance's predictions: at most 8 decode rows of K choices.
constant constexpr uint predictions = 8 * SEISMIC_DIM_K;

// The K largest of row `row`'s logits (ties to the higher expert), in rank
// order, to chosen[row * K + rank]; one simdgroup.
inline void predict_row(arrive::coherent_device<const float> logits, ulong row, threadgroup int *chosen, uint lane) {
    constexpr uint per_lane = SEISMIC_DIM_E / 32;
    float value[per_lane];
    PROJECTION_UNROLL
    for (uint i = 0; i < per_lane; ++i)
        value[i] = logits[logit_index(row, lane + 32 * i)];
    for (uint rank = 0; rank < SEISMIC_DIM_K; ++rank) {
        float best = -INFINITY;
        int best_expert = -1;
        PROJECTION_UNROLL
        for (uint i = 0; i < per_lane; ++i) {
            const int expert = int(lane + 32 * i);
            if (precedes(value[i], expert, best, best_expert)) {
                best = value[i];
                best_expert = expert;
            }
        }
        float winner_value;
        const int winner = reduce::argmax<reduce::HigherIndex>(best, best_expert, winner_value);
        PROJECTION_UNROLL
        for (uint i = 0; i < per_lane; ++i)
            if (winner == int(lane + 32 * i))
                value[i] = -INFINITY;
        if (lane == 0)
            chosen[row * SEISMIC_DIM_K + rank] = winner;
    }
}

// The last threadgroup of a distance: its predictions, then each one absent
// from the layer's slot map and not chosen earlier (row, then rank order)
// published to `ring`; one system-scope fence; the sequence.
inline void hint(arrive::coherent_device<const float> logits, SYSTEM const int *map, ulong map0, SYSTEM int *ring, ulong ring0,
    int sequence, uint rows, threadgroup int *chosen, threadgroup uint *keep, uint simd, uint lane, uint tid) {
    for (uint row = simd; row < rows; row += SEISMIC_TUNE_SIMDGROUPS)
        predict_row(logits, row, chosen, lane);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint total = rows * SEISMIC_DIM_K;
    for (uint c = tid; c < total; c += threads) {
        const int expert = chosen[c];
        bool kept = map[ulong(expert) * map0] < 0;
        for (uint earlier = 0; earlier < c && kept; ++earlier)
            kept = chosen[earlier] != expert;
        keep[c] = kept ? 1u : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid != 0)
        return;
    int count = 0;
    for (uint c = 0; c < total; ++c)
        if (keep[c] != 0u)
            ring[(2 + ulong(count++)) * ring0] = chosen[c];
    ring[1 * ring0] = count;
    streaming::system_fence();
    ring[0] = sequence;
}

kernel void routed_route_streamed(
    device const float *residual [[buffer(SEISMIC_BUFFER_RESIDUAL)]],
    device const uchar *norm [[buffer(SEISMIC_BUFFER_NORM)]],
    device const uchar *router [[buffer(SEISMIC_BUFFER_ROUTER)]],
    device const float *shared_router [[buffer(SEISMIC_BUFFER_SHARED_ROUTER)]],
    device const uchar *shared_gate [[buffer(SEISMIC_BUFFER_SHARED_GATE)]],
    device const uchar *shared_up [[buffer(SEISMIC_BUFFER_SHARED_UP)]],
    device int *routes [[buffer(SEISMIC_BUFFER_ROUTES)]],
    device float *scores [[buffer(SEISMIC_BUFFER_SCORES)]],
    SYSTEM const int *slot_map [[buffer(SEISMIC_BUFFER_SLOT_MAP)]],
    SYSTEM int *stamps [[buffer(SEISMIC_BUFFER_STAMPS)]],
    device const int *step [[buffer(SEISMIC_BUFFER_STEP)]],
    SYSTEM int *control [[buffer(SEISMIC_BUFFER_CONTROL)]],
    device const uchar *near_norm [[buffer(SEISMIC_BUFFER_NEAR_NORM)]],
    device const uchar *near_router [[buffer(SEISMIC_BUFFER_NEAR_ROUTER)]],
    SYSTEM const int *near_map [[buffer(SEISMIC_BUFFER_NEAR_MAP)]],
    device const uchar *far_norm [[buffer(SEISMIC_BUFFER_FAR_NORM)]],
    device const uchar *far_router [[buffer(SEISMIC_BUFFER_FAR_ROUTER)]],
    SYSTEM const int *far_map [[buffer(SEISMIC_BUFFER_FAR_MAP)]],
    SYSTEM int *near_ring [[buffer(SEISMIC_BUFFER_NEAR_RING)]],
    SYSTEM int *far_ring [[buffer(SEISMIC_BUFFER_FAR_RING)]],
    device uchar *normalized [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    device float *coefficient [[buffer(SEISMIC_RESULT_1_BUFFER)]],
    device uchar *shared_product [[buffer(SEISMIC_RESULT_2_BUFFER)]],
    device int *slots [[buffer(SEISMIC_RESULT_3_BUFFER)]],
    arrive::coherent_device<float> logits [[buffer(SEISMIC_BUFFER_SCRATCH_LOGITS)]],
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
    threadgroup int missing;
    threadgroup int chosen[predictions];
    threadgroup uint keep[predictions];
    const RouteRows in = decode_rows(residual, norm, parts, inverses, seismic_words, simd, lane, tid);
    const uint rows = uint(SEISMIC_DIM_M);
    const uint k = uint(SEISMIC_DIM_H);
    const int sequence = step[0];

    if (group < route_groups) {
        route_group(in, router, shared_router, routes, scores, normalized, coefficient, logits, arrivals,
            seismic_words, shared, &last, group, tid, simd, lane);
        if (last == 0u)
            return;
        // Resolve: the selection's routes are this threadgroup's own stores.
        if (tid == 0)
            missing = 0;
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        const uint choices = rows * uint(SEISMIC_DIM_K);
        for (uint c = tid; c < choices; c += threads) {
            const ulong m = c / SEISMIC_DIM_K, choice = c % SEISMIC_DIM_K;
            const int expert = routes[m * SEISMIC_ROUTES_STRIDE_0 + choice * SEISMIC_ROUTES_STRIDE_1];
            const int slot = slot_map[ulong(expert) * SEISMIC_SLOT_MAP_STRIDE_0];
            slots[m * SEISMIC_RESULT_3_STRIDE_0 + choice * SEISMIC_RESULT_3_STRIDE_1] = slot;
            if (slot >= 0)
                stamps[ulong(slot) * SEISMIC_STAMPS_STRIDE_0] = sequence;
            else
                missing = 1;
        }
        // The slots and stamps every thread wrote, for the publication.
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        // With every choice resident the host is not involved: no
        // publication, so no system-scope fence.
        if (tid != 0 || missing == 0)
            return;
        streaming::system_fence();
        // Each choice's expert as resolve found it: an absent one as itself,
        // a resident one as -1 - expert. The host places exactly the
        // choices this launch found absent, whatever it maps meanwhile.
        for (uint c = 0; c < choices; ++c) {
            const ulong m = c / SEISMIC_DIM_K, choice = c % SEISMIC_DIM_K;
            const int expert = routes[m * SEISMIC_ROUTES_STRIDE_0 + choice * SEISMIC_ROUTES_STRIDE_1];
            const int slot = slots[m * SEISMIC_RESULT_3_STRIDE_0 + choice * SEISMIC_RESULT_3_STRIDE_1];
            control[streaming::EXPERTS + c] = slot >= 0 ? -1 - expert : expert;
        }
        control[streaming::COUNT] = int(choices);
        streaming::system_fence();
        control[streaming::REQUEST] = sequence;
        streaming::system_fence();
        return;
    }
    if (group < route_groups + shared_groups) {
        const projection::Weights<packets::W1> gate{shared_gate, KERNEL_W1_LAYOUT(k), k, nullptr};
        const projection::Weights<packets::W2> up{shared_up, KERNEL_W2_LAYOUT(k), k, nullptr};
        const projection::SiluMul<Act> out{shared_product, SEISMIC_RESULT_2_STRIDE_0, SEISMIC_RESULT_2_STRIDE_1};
        PROJECTION_FOR_ROWS(rows,
            (projection::gemv_paired_runtime<packets::W1, packets::W2, SEISMIC_TUNE_ROWS, MAXM, SEISMIC_TUNE_LANES>(
                in, out, gate, up, rows, SEISMIC_DIM_S, k, group - route_groups, shared, SEISMIC_TUNE_SIMDGROUPS,
                simd, lane)));
        return;
    }

    // A distance ahead: 0 near, 1 far.
    const uint ahead = group - route_groups - shared_groups;
    const uint distance = ahead / predict_groups;
    if (int(distance) >= int(SEISMIC_PARAM_AHEAD))
        return;
    const bool far = distance == 1;
    const RouteRows predicted_in{residual, SEISMIC_RESIDUAL_STRIDE_0, SEISMIC_RESIDUAL_STRIDE_1,
        far ? far_norm : near_norm, far ? SEISMIC_FAR_NORM_STRIDE_0 : SEISMIC_NEAR_NORM_STRIDE_0, inverses, k};
    const projection::Weights<Router> weights{far ? far_router : near_router, KERNEL_W0_LAYOUT(k), k, nullptr};
    arrive::coherent_device<float> predicted = logits + (1 + distance) * ulong(SEISMIC_DIM_M) * (SEISMIC_DIM_E + 1);
    const routing::Logits out{predicted, SEISMIC_DIM_E + 1, 0};
    routing::Gemv<Act::bytes>::columns(predicted_in, weights, out, rows, uint(SEISMIC_DIM_E), ahead % predict_groups,
        shared, SEISMIC_TUNE_SIMDGROUPS, simd, lane);
    if (!arrive::last(arrivals + 1 + distance, predict_groups, &last, tid))
        return;
    if (far)
        hint(predicted, far_map, SEISMIC_FAR_MAP_STRIDE_0, far_ring, SEISMIC_FAR_RING_STRIDE_0, sequence, rows,
            chosen, keep, simd, lane, tid);
    else
        hint(predicted, near_map, SEISMIC_NEAR_MAP_STRIDE_0, near_ring, SEISMIC_NEAR_RING_STRIDE_0, sequence,
            rows, chosen, keep, simd, lane, tid);
}
