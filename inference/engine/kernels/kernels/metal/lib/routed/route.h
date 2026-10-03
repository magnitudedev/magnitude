// The decode routing of the `routed_route` contract, shared by the
// `routed_route` and `routed_route_shared` entries: both bind its parameters
// (`residual`, `norm`, `router`, `shared_router`, `routes`, `scores`, `eps`,
// `normalize`), results (`normalized`, `coefficient`), scratch (`logits`,
// `arrivals`) and SIMDGROUPS under the same names. The includer defines
// KERNEL_W0 as the router.
//   decode_rows: a threadgroup's router input over the decode rows (M <= 8),
//     from RMS inverses of its own in the stage launch's order.
//   route_group: one router threadgroup of the decode rows. Threadgroup 0
//     publishes the normalized rows; every threadgroup but the last forms
//     the logits of SIMDGROUPS router rows, the last the shared-expert gate
//     column; the last threadgroup to arrive (`arrive::last`) then selects
//     every row, one simdgroup per row (`select_row`).
// The router logits scratch is [M, E + 1]: column E holds the shared-expert
// gate logit (normalized . shared_router), formed like an expert's logit.

#include "router.h"
#include "../core/arrive.h"
#include "../core/reduce.h"

typedef element::Act Act;
// The norm is dense; router rows may be dense or packed rows16.
typedef ELEMENT_OF(SEISMIC_NORM) Norm;
typedef routing::Packets<packets::W0>::type Router;

constant constexpr uint route_threads = routing::stage_threads;
constant constexpr uint route_simdgroups = routing::stage_simdgroups;

inline ulong logit_index(ulong row, ulong expert) {
    return row * (SEISMIC_DIM_E + 1) + expert;
}

// Descending probability, ties to the higher expert index.
inline bool precedes(float probability, int expert, float best, int best_expert) {
    return probability > best || (probability == best && expert > best_expert);
}

// The selection of one row by one simdgroup, reading the row's logits
// through `logit(index)`: the softmax over every expert (E / 32 experts per
// lane), K rounds of simdgroup argmax (ties to the higher expert index), the
// optional renormalization and the shared-expert coefficient from column E.
// Every register array is indexed by unrolled loops only: a dynamically
// indexed one lives in stack memory, and its K dependent rounds then dominate
// the routing.
template <typename Logit>
inline void select_row(ulong row, Logit logit, device int *routes, device float *scores,
    device float *coefficient, constant ulong *seismic_words, uint lane) {
    constexpr uint per_lane = SEISMIC_DIM_E / 32;
    constexpr uint K = SEISMIC_DIM_K;
    // Lane `lane` holds experts lane + 32 i.
    float probability[per_lane];
    float maximum = -INFINITY;
    PROJECTION_UNROLL
    for (uint i = 0; i < per_lane; ++i) {
        probability[i] = logit(logit_index(row, lane + 32 * i));
        maximum = metal::max(maximum, probability[i]);
    }
    maximum = simd_max(maximum);
    float total = 0.0f;
    PROJECTION_UNROLL
    for (uint i = 0; i < per_lane; ++i) {
        probability[i] = metal::exp(probability[i] - maximum);
        total += probability[i];
    }
    total = simd_sum(total);
    PROJECTION_UNROLL
    for (uint i = 0; i < per_lane; ++i)
        probability[i] = probability[i] / total;

    float chosen[K];
    int winners[K];
    PROJECTION_UNROLL
    for (uint rank = 0; rank < K; ++rank) {
        float best = -INFINITY;
        int best_expert = -1;
        PROJECTION_UNROLL
        for (uint i = 0; i < per_lane; ++i) {
            const int expert = int(lane + 32 * i);
            if (precedes(probability[i], expert, best, best_expert)) {
                best = probability[i];
                best_expert = expert;
            }
        }
        float winner_probability;
        const int winner = reduce::argmax<reduce::HigherIndex>(best, best_expert, winner_probability);
        PROJECTION_UNROLL
        for (uint i = 0; i < per_lane; ++i)
            if (winner == int(lane + 32 * i))
                probability[i] = -INFINITY;
        chosen[rank] = winner_probability;
        winners[rank] = winner;
    }
    // Slot s holds rank K - 1 - s; the renormalization divides by the
    // slot-order sum.
    float denominator = 0.0f;
    PROJECTION_UNROLL
    for (uint slot = 0; slot < K; ++slot)
        denominator += chosen[K - 1 - slot];
    if (lane == 0) {
        PROJECTION_UNROLL
        for (uint slot = 0; slot < K; ++slot) {
            const float probability = chosen[K - 1 - slot];
            routes[row * SEISMIC_ROUTES_STRIDE_0 + slot * SEISMIC_ROUTES_STRIDE_1] = winners[K - 1 - slot];
            scores[row * SEISMIC_SCORES_STRIDE_0 + slot * SEISMIC_SCORES_STRIDE_1] =
                SEISMIC_PARAM_NORMALIZE != 0 ? probability / denominator : probability;
        }
        coefficient[row * SEISMIC_RESULT_1_STRIDE_0] =
            1.0f / (1.0f + metal::exp(-logit(logit_index(row, SEISMIC_DIM_E))));
    }
}

// The shared-expert gate row: one dense, contiguous F32 row of H values (a
// weight, like the router rows).
typedef routing::EagerDense<element::F32> SharedGate;

// Router threadgroups of the decode rows: ceil(E / SIMDGROUPS) of router
// rows, then the shared-expert gate column.
constant constexpr uint route_groups = (uint(SEISMIC_DIM_E) + SEISMIC_TUNE_SIMDGROUPS - 1) / SEISMIC_TUNE_SIMDGROUPS + 1;

// The router input of the decode rows in this threadgroup: the rows' RMS
// inverses (in the stage launch's order) reduced into `inverses` (8 floats)
// through `parts` (8 * route_simdgroups floats). Every thread calls it.
inline routing::Rows<Norm> decode_rows(device const float *residual, device const uchar *norm,
    threadgroup float *parts, threadgroup float *inverses, constant ulong *seismic_words, uint simd, uint lane,
    uint tid) {
    routing::inverses(residual, SEISMIC_RESIDUAL_STRIDE_0, SEISMIC_RESIDUAL_STRIDE_1, uint(SEISMIC_DIM_H),
        as_type<float>(uint(SEISMIC_PARAM_EPS)), uint(SEISMIC_DIM_M), parts, inverses, SEISMIC_TUNE_SIMDGROUPS, simd,
        lane, tid);
    return routing::Rows<Norm>{residual, SEISMIC_RESIDUAL_STRIDE_0, SEISMIC_RESIDUAL_STRIDE_1, norm,
        SEISMIC_NORM_STRIDE_0, inverses, uint(SEISMIC_DIM_H)};
}

// Router threadgroup `group` (< route_groups) over the decode rows `in`.
// Every thread calls it; `last` is one threadgroup word.
inline void route_group(thread const routing::Rows<Norm> &in, device const uchar *router,
    device const float *shared_router, device int *routes, device float *scores, device uchar *normalized,
    device float *coefficient, device float *logits, device atomic_uint *arrivals, constant ulong *seismic_words,
    threadgroup uchar *shared, threadgroup uint *last, uint group, uint tid, uint simd, uint lane) {
    const uint rows = uint(SEISMIC_DIM_M);
    const uint k = uint(SEISMIC_DIM_H);
    if (group == 0) {
        for (uint item = tid; item < rows * k; item += 32 * SEISMIC_TUNE_SIMDGROUPS) {
            const ulong row = item / k, source = item % k;
            element::put<Act>(normalized, row * SEISMIC_RESULT_0_STRIDE_0 + source * SEISMIC_RESULT_0_STRIDE_1,
                in.value(uint(row), source));
        }
    }
    if (group + 1 < route_groups) {
        const projection::Weights<Router> weights{router, KERNEL_W0_LAYOUT(k), k, nullptr};
        const routing::Logits out{logits, SEISMIC_DIM_E + 1, 0};
        routing::Gemv<Act::bytes>::columns(in, weights, out, rows, uint(SEISMIC_DIM_E), group, shared,
            SEISMIC_TUNE_SIMDGROUPS, simd, lane);
    } else {
        const projection::Weights<SharedGate> gate{reinterpret_cast<device const uchar *>(shared_router),
            PACKETS_ROWS16_DENSE(element::F32, k), k, nullptr};
        const routing::Logits out{logits, SEISMIC_DIM_E + 1, SEISMIC_DIM_E};
        routing::Gemv<Act::bytes>::columns(in, gate, out, rows, 1, 0, shared, SEISMIC_TUNE_SIMDGROUPS, simd,
            lane);
    }
    if (!arrive::last(arrivals, route_groups, last, tid))
        return;
    for (uint row = simd; row < rows; row += SEISMIC_TUNE_SIMDGROUPS)
        select_row(row, [&](ulong at) { return logits[at]; }, routes, scores, coefficient, seismic_words, lane);
}
