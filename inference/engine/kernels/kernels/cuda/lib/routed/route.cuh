// The routing of the `routed_route` contract, shared by the `routed_route`,
// `routed_route_shared` and `routed_route_streamed` entries; the CUDA form
// of `metal/lib/routed/route.h`. Each binds its parameters (`residual`,
// `norm`, `router`, `shared_router`, `routes`, `scores`, `eps`, `normalize`,
// `expert_scale` of static extent ES), results (`normalized`, `coefficient`)
// and scratch (`logits`, `arrivals`) under the same names, and SIMDGROUPS.
// KERNEL_W0 is the router. Every function takes the entry's words.
//   Input: the router input (`form_input`) of a row through a norm (the
//     layer's, or a layer ahead's: its input form is this layer's), each
//     value rounded to A.
//   decode_inverses: a GEMV block's RMS inverses of the decode rows
//     (M <= DECODE_ROWS), in the stage launch's sum order.
//   Columns, weight: a router's logit columns, column E the shared-expert
//     gate (`shared_router`), formed like an expert's logit.
//   column: one warp's logit column over the decode rows, normalizing each
//     element as it reads it, into a [M, E + 1] logits plane.
//   rank_row: one row's K winners by one warp (`rank_experts`).
//   select_row: a row's routes, scores and coefficient from its ranking;
//     lane 0 then hands `bind` the row's routes.
//   BLOCKS, route_block: the router blocks of the decode rows: block 0
//     publishes the normalized rows, each warp forms one logit column, and
//     the last block to arrive (`arrive::last`) selects every row, one warp
//     per row.
// Every logit accumulates over the hidden axis in the same order for every
// column, so experts with equal router rows tie exactly, and a column's
// bits never depend on the block or warp that forms it (SIMDGROUPS).

#include "../core/activation.cuh"
#include "../core/arrive.cuh"
#include "../core/reduce.cuh"
#include "ranked.cuh"
// A packed router's decoder. The include inliner does not evaluate `#if`, so
// a conditional include follows the unconditional ones (`activation.cuh`
// brings <seismic/element.cuh>), and an includer whose other weights need
// the packets includes them unconditionally first (this one then inlines
// nothing).
#if defined(SEISMIC_ROUTER_KIND_PACKED)
#define KERNEL_W0 SEISMIC_ROUTER
#include <seismic/packets.cuh>
#endif

namespace route {

using element::Act;
// The norm is dense; packed routers use the resident mma16 packet decoder.
typedef ELEMENT_OF(SEISMIC_NORM) Norm;
#if !defined(SEISMIC_ROUTER_KIND_PACKED)
typedef ELEMENT_OF(SEISMIC_ROUTER) Router;
#endif

// The stage launch's block: its sum order fixes every RMS inverse.
constexpr unsigned THREADS = 256;
constexpr unsigned WARPS = THREADS / 32;
// Whether the input is RMS-normalized (`form_input` with a norm), else the
// sum of a mix site's partials.
constexpr bool NORMED = SEISMIC_DIM_NM != 0;

// The router input of the residual's rows through `norm` (stride `norm0`):
// at (row, source) the residual times its row's RMS inverse and the norm, or
// (no norm) the sum in part order of a mix site's PT partials, rounded to A.
struct Input {
    const float *residual;
    const unsigned char *norm;
    unsigned long long norm0;

    __device__ __forceinline__ float value(unsigned long long row, unsigned long long source, float inverse,
                                           const seismic_words_t &seismic_words_value) const {
        if constexpr (!NORMED) {
            float sum = 0.0f;
            for (unsigned long long part = 0; part < SEISMIC_DIM_PT; ++part)
                sum = seismic_add_rn(sum, residual[row * SEISMIC_RESIDUAL_STRIDE_0
                                                   + (part * SEISMIC_DIM_H + source) * SEISMIC_RESIDUAL_STRIDE_1]);
            return Act::round(sum);
        } else {
            const float value = residual[row * SEISMIC_RESIDUAL_STRIDE_0 + source * SEISMIC_RESIDUAL_STRIDE_1];
            return Act::round(seismic_mul_rn(seismic_mul_rn(value, inverse),
                                             element::at<Norm>(norm, source * norm0)));
        }
    }
};

// The decode rows' RMS inverses into `inverses` (DECODE_ROWS floats; 0 without a
// norm), in the stage launch's sum order (256 strided partials, warp sums,
// then the eight warp partials in order): item (row, part) is part `part`'s
// warp sum, and the items spread over every warp of the block, through
// `parts`. Every thread of the block calls it; the inverses are the block's
// when it returns.
__device__ __forceinline__ void decode_inverses(const float *residual, float (*parts)[WARPS], float *inverses,
                                                const seismic_words_t &seismic_words_value) {
    const unsigned thread = threadIdx.x, lane = thread % 32, warp = thread / 32, warps = blockDim.x / 32;
    const unsigned rows = static_cast<unsigned>(SEISMIC_DIM_M);
    if constexpr (NORMED) {
        const float eps = __uint_as_float(static_cast<unsigned>(SEISMIC_PARAM_EPS));
        for (unsigned item = warp; item < rows * WARPS; item += warps) {
            const unsigned row = item / WARPS, part = item % WARPS;
            float squares = 0.0f;
            for (unsigned source = part * 32 + lane; source < SEISMIC_DIM_H; source += THREADS) {
                const float value = residual[row * SEISMIC_RESIDUAL_STRIDE_0 + source * SEISMIC_RESIDUAL_STRIDE_1];
                squares = seismic_fma_rn(value, value, squares);
            }
            squares = seismic_warp_sum_f32(squares);
            if (lane == 0)
                parts[row][part] = squares;
        }
        __syncthreads();
        if (thread < rows) {
            float total = 0.0f;
            for (unsigned part = 0; part < WARPS; ++part)
                total = seismic_add_rn(total, parts[thread][part]);
            inverses[thread] = rsqrtf(seismic_add_rn(total / static_cast<float>(SEISMIC_DIM_H), eps));
        }
    } else {
        if (thread < SEISMIC_CONST_DECODE_ROWS)
            inverses[thread] = 0.0f;
    }
    __syncthreads();
}

// The logit columns of a router (the layer's, or a layer ahead's: the same
// element), then the shared-expert gate row as column E.
struct Columns {
    const unsigned char *router;
    unsigned long long router0, router1;
    const float *shared_router;
    unsigned long long shared_router0;
};

// The entry's router input (through its own norm) and logit columns.
#define ROUTE_INPUT                                                                                       \
    route::Input {                                                                                        \
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL)), SEISMIC_PTR(SEISMIC_BUFFER_NORM), \
            SEISMIC_NORM_STRIDE_0                                                                         \
    }
#define ROUTE_COLUMNS                                                                                     \
    route::Columns {                                                                                      \
        SEISMIC_PTR(SEISMIC_BUFFER_ROUTER), SEISMIC_ROUTER_STRIDE_0, SEISMIC_ROUTER_STRIDE_1,             \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_ROUTER)),                   \
            SEISMIC_SHARED_ROUTER_STRIDE_0                                                                \
    }

// Each of a router's functions below takes its expert count EXPERTS: the
// entry's own router's (E, the default), or a layer ahead's (the streamed
// route's hints rank the next layer's EN).
#if !defined(SEISMIC_ROUTER_KIND_PACKED)
// Dense weight of logit column `expert` at hidden coordinate `source`.
template <unsigned EXPERTS = static_cast<unsigned>(SEISMIC_DIM_E)>
__device__ __forceinline__ float weight(const Columns &w,unsigned long long expert, unsigned long long source,
                                        const seismic_words_t &seismic_words_value) {
    return expert < EXPERTS ? element::at<Router>(w.router, expert * w.router0 + source * w.router1)
                            : w.shared_router[source * w.shared_router0];
}
#endif

// Streams a dense column `expert` into L2 while the launch before finishes.
template <unsigned EXPERTS = static_cast<unsigned>(SEISMIC_DIM_E)>
__device__ __forceinline__ void prefetch(const Columns &w,unsigned long long expert, unsigned lane,
                                         const seismic_words_t &seismic_words_value) {
#if !defined(SEISMIC_ROUTER_KIND_PACKED)
    if (expert <= EXPERTS && w.router1 == 1 && w.shared_router0 == 1) {
        const unsigned char *column = expert < EXPERTS
            ? w.router + expert * w.router0 * Router::bytes
            : reinterpret_cast<const unsigned char *>(w.shared_router);
        const unsigned long long bytes = SEISMIC_DIM_H * (expert < EXPERTS ? Router::bytes : 4u);
        for (unsigned long long byte = 128ull * lane; byte < bytes; byte += 128ull * 32)
            seismic_prefetch_l2(column + byte);
    }
#endif
}

// Index of (row, column) in a [M, EXPERTS + 1] logits plane.
template <unsigned EXPERTS = static_cast<unsigned>(SEISMIC_DIM_E)>
__device__ __forceinline__ unsigned long long logit_index(unsigned long long row, unsigned long long expert,
                                                          const seismic_words_t &seismic_words_value) {
    return row * (EXPERTS + 1) + expert;
}

// Logit column `expert` (<= EXPERTS) of the decode rows `in` (their RMS
// `inverses`) through `w`, by one warp, into `logits` (a [M, EXPERTS + 1]
// plane).
template <unsigned EXPERTS = static_cast<unsigned>(SEISMIC_DIM_E)>
__device__ __forceinline__ void column(const Input &in, const float *inverses, const Columns &w,
                                       unsigned long long expert, float *logits, unsigned lane,
                                       const seismic_words_t &seismic_words_value) {
    const unsigned rows = static_cast<unsigned>(SEISMIC_DIM_M);
    float sums[SEISMIC_CONST_DECODE_ROWS] = {};
#if defined(SEISMIC_ROUTER_KIND_PACKED)
    const auto router_rows = KERNEL_W0_AT(w.router);
    // One lane decodes a 16-value register chunk of an mma16 router row.
    for (unsigned item = lane; item < SEISMIC_DIM_H / 64 * 4; item += 32) {
        const unsigned long long kb = item / 4;
        const unsigned t = item % 4;
        const unsigned offsets[4] = {2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9};
        float values[16];
        if (expert < EXPERTS)
            packets::row_values16(router_rows, expert, kb, t, values);
#pragma unroll
        for (unsigned s = 0; s < 4; ++s)
#pragma unroll
            for (unsigned j = 0; j < 4; ++j) {
                const unsigned long long source = 64 * kb + 16 * s + offsets[j];
                const float weight = expert < EXPERTS
                    ? values[4 * s + j] : w.shared_router[source * w.shared_router0];
#pragma unroll
                for (unsigned row = 0; row < SEISMIC_CONST_DECODE_ROWS; ++row)
                    if (row < rows)
                        sums[row] = seismic_fma_rn(in.value(row, source, inverses[row], seismic_words_value),
                            weight, sums[row]);
            }
    }
#else
    // Dense lane l owns sources 4l + 128 i .. + 3. The loads of BATCH
    // consecutive i are issued before their products, so a warp waits on
    // memory once per batch rather than once per i; each row still
    // accumulates its sources in i order, then j order.
    constexpr unsigned BATCH = 4;
    if constexpr (!NORMED) {
        // The mix site's partials: each source's input sums PT reads.
        for (unsigned first = 4 * lane; first < SEISMIC_DIM_H; first += 128) {
#pragma unroll
            for (unsigned j = 0; j < 4; ++j) {
                const float weight = first + j < SEISMIC_DIM_H
                    ? route::weight<EXPERTS>(w, expert, first + j, seismic_words_value) : 0.0f;
#pragma unroll
                for (unsigned row = 0; row < SEISMIC_CONST_DECODE_ROWS; ++row)
                    if (row < rows && first + j < SEISMIC_DIM_H)
                        sums[row] = seismic_fma_rn(in.value(row, first + j, 0.0f, seismic_words_value), weight,
                            sums[row]);
            }
        }
    } else {
        for (unsigned first = 4 * lane; first < SEISMIC_DIM_H; first += 128 * BATCH) {
            float weight[BATCH][4], scale[BATCH][4];
#pragma unroll
            for (unsigned b = 0; b < BATCH; ++b) {
                const unsigned base = first + 128 * b;
#pragma unroll
                for (unsigned j = 0; j < 4; ++j) {
                    weight[b][j] = base < SEISMIC_DIM_H ? route::weight<EXPERTS>(w, expert, base + j, seismic_words_value) : 0.0f;
                    scale[b][j] = base < SEISMIC_DIM_H ? element::at<Norm>(in.norm, (base + j) * in.norm0) : 0.0f;
                }
            }
#pragma unroll
            for (unsigned row = 0; row < SEISMIC_CONST_DECODE_ROWS; ++row) {
                if (row < rows) {
                    float value[BATCH][4];
#pragma unroll
                    for (unsigned b = 0; b < BATCH; ++b) {
                        const unsigned base = first + 128 * b;
#pragma unroll
                        for (unsigned j = 0; j < 4; ++j)
                            value[b][j] = base < SEISMIC_DIM_H
                                ? in.residual[row * SEISMIC_RESIDUAL_STRIDE_0 + (base + j) * SEISMIC_RESIDUAL_STRIDE_1]
                                : 0.0f;
                    }
#pragma unroll
                    for (unsigned b = 0; b < BATCH; ++b) {
                        if (first + 128 * b >= SEISMIC_DIM_H)
                            break;
#pragma unroll
                        for (unsigned j = 0; j < 4; ++j)
                            sums[row] = seismic_fma_rn(Act::round(seismic_mul_rn(seismic_mul_rn(value[b][j],
                                inverses[row]), scale[b][j])), weight[b][j], sums[row]);
                    }
                }
            }
        }
    }
#endif
#pragma unroll
    for (unsigned row = 0; row < SEISMIC_CONST_DECODE_ROWS; ++row) {
        const float sum = seismic_warp_sum_f32(sums[row]);
        if (row < rows && lane == 0)
            logits[logit_index<EXPERTS>(row, expert, seismic_words_value)] = sum;
    }
}

// One row's K winners by rank (rank 0 the most probable) and their ranked
// probabilities, as every lane of the warp holds them.
struct Ranked {
    float chosen[SEISMIC_DIM_K];
    int winners[SEISMIC_DIM_K];
};

// The ranking of one row by one warp, reading its logit of expert e as
// `logit(e)`: the softmax over every expert (lane `lane` holds its
// `selection::candidates`, ranked.cuh; an excluded entry is never read or
// summed), each probability ranked, then K rounds of warp argmax by
// `selection::precedes`, each taking its winner out of the ranking. An entry
// out of the ranking (excluded, or a round's taken winner) holds
// `selection::out`, so the rounds keep no expert array: entry i is expert
// lane + 32 i.
template <unsigned EXPERTS = static_cast<unsigned>(SEISMIC_DIM_E), class Logit>
__device__ __forceinline__ Ranked rank_row(Logit logit, unsigned lane) {
    // PER_LANE when 32 divides EXPERTS.
    constexpr unsigned PER_LANE = (EXPERTS + 31) / 32;
    const unsigned held = selection::candidates<EXPERTS>(lane);
    const float negative_infinity = -__int_as_float(0x7f800000);
    float probability[PER_LANE];
    float maximum = negative_infinity;
#pragma unroll
    for (unsigned i = 0; i < PER_LANE; ++i) {
        probability[i] = i < held ? logit(lane + 32 * i) : negative_infinity;
        maximum = fmaxf(maximum, probability[i]);
    }
    maximum = seismic_warp_max_f32(maximum);
    float total = 0.0f;
#pragma unroll
    for (unsigned i = 0; i < PER_LANE; ++i) {
        if (i < held) {
            probability[i] = expf(probability[i] - maximum);
            total = seismic_add_rn(total, probability[i]);
        }
    }
    total = seismic_warp_sum_f32(total);
#pragma unroll
    for (unsigned i = 0; i < PER_LANE; ++i)
        probability[i] = i < held ? selection::ranked(probability[i] / total) : selection::out();

    Ranked ranked;
    for (unsigned rank = 0; rank < SEISMIC_DIM_K; ++rank) {
        float best = negative_infinity;
        int best_expert = -1;
#pragma unroll
        for (unsigned i = 0; i < PER_LANE; ++i) {
            if (selection::precedes(probability[i], static_cast<int>(lane + 32 * i), best, best_expert)) {
                best = probability[i];
                best_expert = static_cast<int>(lane + 32 * i);
            }
        }
        float winner_probability;
        const int winner = reduce::argmax<reduce::HigherIndex>(best, best_expert, winner_probability);
#pragma unroll
        for (unsigned i = 0; i < PER_LANE; ++i)
            if (winner == static_cast<int>(lane + 32 * i))
                probability[i] = selection::out();
        ranked.chosen[rank] = winner_probability;
        ranked.winners[rank] = winner;
    }
    return ranked;
}

// The selection's outputs.
struct Selection {
    int *routes;
    unsigned long long route_row, route_slot;
    float *scores;
    unsigned long long score_row, score_slot;
    float *coefficient;
    unsigned long long coefficient_row;
    bool normalize;
    // The `expert_scale` port (static extent ES; read only where present).
    const float *expert_scale;
    unsigned long long expert_scale_stride;
};

#define ROUTE_SELECTION                                                                                   \
    route::Selection {                                                                                    \
        reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES)), SEISMIC_ROUTES_STRIDE_0,             \
            SEISMIC_ROUTES_STRIDE_1, reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCORES)),       \
            SEISMIC_SCORES_STRIDE_0, SEISMIC_SCORES_STRIDE_1,                                             \
            reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_1_BUFFER)), SEISMIC_RESULT_1_STRIDE_0,   \
            SEISMIC_PARAM_NORMALIZE != 0,                                                                 \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_SCALE)),                    \
            SEISMIC_EXPERT_SCALE_STRIDE_0                                                                 \
    }

// The selection of `row` by one warp (`rank_row` over `logit`): each score
// optionally renormalized by the slot-order sum (slot s holds rank K - 1 -
// s), then multiplied by its expert's `expert_scale` where the port is
// present, and the shared-expert coefficient from column E, stored by lane
// 0 in slot order.
template <class Logit>
__device__ __forceinline__ void select_row(Logit logit, const Selection &out, unsigned long long row, unsigned lane,
                                           const seismic_words_t &seismic_words_value) {
    constexpr unsigned K = SEISMIC_DIM_K;
    const Ranked ranked = rank_row(logit, lane);
    float denominator = 0.0f;
    if (out.normalize)
        for (unsigned slot = 0; slot < K; ++slot)
            denominator = seismic_add_rn(denominator, ranked.chosen[K - 1 - slot]);
    if (lane != 0)
        return;
    for (unsigned slot = 0; slot < K; ++slot) {
        const unsigned rank = K - 1 - slot;
        out.routes[row * out.route_row + slot * out.route_slot] = ranked.winners[rank];
        float score = out.normalize ? ranked.chosen[rank] / denominator : ranked.chosen[rank];
        if (SEISMIC_DIM_ES != 0)
            score = seismic_mul_rn(score,
                out.expert_scale[static_cast<unsigned long long>(ranked.winners[rank]) * out.expert_scale_stride]);
        out.scores[row * out.score_row + slot * out.score_slot] = score;
    }
    out.coefficient[row * out.coefficient_row] = 1.0f / (1.0f + expf(-logit(SEISMIC_DIM_E)));
}

// Router blocks of the decode rows: a warp a logit column, the
// shared-expert gate column E in the last block.
constexpr unsigned BLOCKS = (static_cast<unsigned>(SEISMIC_DIM_E) + SEISMIC_TUNE_SIMDGROUPS) / SEISMIC_TUNE_SIMDGROUPS;

// Router block `group` (of BLOCKS) over the
// decode rows `in`: warp w's logit column group SIMDGROUPS + w (<= E)
// streams into L2 while the launch before finishes; then the block forms
// the rows' RMS inverses (`decode_inverses`, through its `parts` and
// `inverses`), block 0 publishes the normalized rows, each warp forms its
// column into `logits` ([M, E + 1]), and the last block to arrive selects
// every row, one warp per row. Every thread calls it; true in the selecting
// block.
__device__ __forceinline__ bool route_block(const Input &in, const Columns &w, const Selection &out,
                                            unsigned char *normalized, unsigned long long normalized0,
                                            unsigned long long normalized1, float *logits, unsigned *arrivals,
                                            unsigned group, float (*parts)[WARPS],
                                            float *inverses, unsigned *last,
                                            const seismic_words_t &seismic_words_value) {
    const unsigned thread = threadIdx.x, lane = thread % 32, warp = thread / 32, warps = blockDim.x / 32;
    const unsigned rows = static_cast<unsigned>(SEISMIC_DIM_M);
    const unsigned long long expert = static_cast<unsigned long long>(group) * warps + warp;
    prefetch(w, expert, lane, seismic_words_value);
    seismic_dependency_start();
    decode_inverses(in.residual, parts, inverses, seismic_words_value);
    if (group == 0) {
        for (unsigned item = thread; item < rows * static_cast<unsigned>(SEISMIC_DIM_H); item += blockDim.x) {
            const unsigned long long row = item / SEISMIC_DIM_H, source = item % SEISMIC_DIM_H;
            element::put<Act>(normalized, row * normalized0 + source * normalized1,
                in.value(row, source, inverses[row], seismic_words_value));
        }
    }
    if (expert <= SEISMIC_DIM_E)
        column(in, inverses, w, expert, logits, lane, seismic_words_value);
    if (!arrive::last(arrivals, BLOCKS, last))
        return false;
    for (unsigned row = warp; row < rows; row += warps) {
        const float *values = logits + logit_index(row, 0, seismic_words_value);
        select_row([&](unsigned long long expert) { return __ldcg(values + expert); }, out, row, lane,
            seismic_words_value);
    }
    return true;
}

} // namespace route
