// Expert streaming's decode routing (`routed_route_streamed` in
// routed.seismic); the CUDA form of `metal/routed_route_streamed.metal`:
// `routed_route_shared`'s launch, whose last router block to arrive
// selects, and the next two layers' hints (AHEAD 2).
//   [0, route::BLOCKS)               the router blocks
//                                    (`lib/routed/route.cuh`).
//   [.., + AHEAD * PREDICT_BLOCKS)   the router GEMV of the next layer (EN
//                                    experts) over this layer's residual
//                                    normed by that layer's norm, into the
//                                    logits after this layer's [M, E + 1];
//                                    the last of them to
//                                    arrive ranks its rows and publishes the
//                                    hints (`streaming/hint.hpp`).
//   [.., + shared blocks)            the shared expert's gate/up blocks
//                                    (`lib/routed/shared.cuh`).
// The hints dispatch before the shared expert, so their serial tail (the
// arrival, ranking and publication) runs beside the shared product rather
// than after it. While the next layer's group is whole the launch encoded is
// the `_alone` kernel, the same body compiled without the hint blocks or
// their code.
// Every block forms the RMS inverses as `routed_route_shared` does, so the
// routing and the shared product carry its bits. The three groups count
// their arrivals apart and none waits for another.
//
// The selecting block then resolves and assigns (`streaming/resolve.hpp`):
// a layer of a group that is not whole (`routed_route_streamed`) its
// sources, routing record, the tier wait's grid and, on a miss, the
// request; a layer of a whole group (`routed_route_whole`) its hits and
// routing record alone.

#define KERNEL_W0 SEISMIC_ROUTER
#define KERNEL_W1 SEISMIC_SHARED_GATE
#define KERNEL_W2 SEISMIC_SHARED_UP
#include <seismic/packets.cuh>
#include "lib/routed/route.cuh"
#include "lib/routed/shared.cuh"
#include "lib/streaming/control.cuh"
#include "../streaming/hint.hpp"
#include "../streaming/resolve.hpp"
#include "lib/routed/resolve.cuh"

SEISMIC_PROGRAMMATIC_DEPENDENCY

// The next streamed layer's experts (EN, static) and its router blocks: its
// EN logit columns.
constexpr unsigned NEXT_EXPERTS = static_cast<unsigned>(SEISMIC_DIM_EN);
constexpr unsigned PREDICT_BLOCKS = (NEXT_EXPERTS + SEISMIC_TUNE_SIMDGROUPS - 1) / SEISMIC_TUNE_SIMDGROUPS;

// The ports the launch's kernels bind.
struct Ports {
    route::Input in;
    route::Columns columns;
    route::Selection selection;
    unsigned char *normalized;
    const unsigned char *shared_gate, *shared_up;
    unsigned char *shared_product;
    const int *routes;
    const int *step;
    ResolvePorts bank;
    // Each lead's prediction (one layer ahead, then two): that layer's
    // router input and columns.
    route::Input ahead[2];
    route::Columns ahead_columns[2];
    int *rings;
    float *logits;
    unsigned *arrivals;
};

#define ROUTED_ROUTE_PORTS                                                                                 \
    Ports {                                                                                                \
        ROUTE_INPUT, ROUTE_COLUMNS, ROUTE_SELECTION, SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER),                 \
            SEISMIC_PTR(SEISMIC_BUFFER_SHARED_GATE), SEISMIC_PTR(SEISMIC_BUFFER_SHARED_UP),               \
            SEISMIC_PTR(SEISMIC_RESULT_2_BUFFER), reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES)), \
            reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_STEP)), ROUTED_RESOLVE_PORTS,        \
            {route::Input{reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL)),          \
                          SEISMIC_PTR(SEISMIC_BUFFER_NEXT_NORM), SEISMIC_NEXT_NORM_STRIDE_0},             \
             route::Input{reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL)),          \
                          SEISMIC_PTR(SEISMIC_BUFFER_AFTER_NORM), SEISMIC_AFTER_NORM_STRIDE_0}},          \
            {route::Columns{SEISMIC_PTR(SEISMIC_BUFFER_NEXT_ROUTER), SEISMIC_NEXT_ROUTER_STRIDE_0,        \
                            SEISMIC_NEXT_ROUTER_STRIDE_1,                                                 \
                            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_ROUTER)),   \
                            SEISMIC_SHARED_ROUTER_STRIDE_0},                                              \
             route::Columns{SEISMIC_PTR(SEISMIC_BUFFER_AFTER_ROUTER), SEISMIC_AFTER_ROUTER_STRIDE_0,      \
                            SEISMIC_AFTER_ROUTER_STRIDE_1,                                                \
                            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_ROUTER)),   \
                            SEISMIC_SHARED_ROUTER_STRIDE_0}},                                             \
            reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_RINGS)),                                   \
            reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_LOGITS)),                        \
            reinterpret_cast<unsigned *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_ARRIVALS))                    \
    }

// Hint block `group` (of PREDICT_BLOCKS) of lead `lead` (0: one layer
// ahead, 1: two): that layer's router GEMV over this residual's inverses
// into the lead's [M, EN + 1] logits plane after this layer's [M, E + 1],
// then, in the last of the lead's blocks to arrive (arrival counter 1 +
// lead), its rows ranked (`route::rank_row`) and the hints published into
// the lead's ring row. Every thread calls it.
__device__ __forceinline__ void hint_block(const Ports &ports, unsigned lead, unsigned group,
                                           float (*parts)[route::WARPS], float *inverses, unsigned *last,
                                           const seismic_words_t &seismic_words_value) {
    constexpr unsigned K = SEISMIC_DIM_K;
    __shared__ int words[STREAMING_HINT_WORDS(SEISMIC_CONST_DECODE_ROWS)];
    const route::Input &next = ports.ahead[lead];
    const route::Columns &next_columns = ports.ahead_columns[lead];
    float *predicted = ports.logits + SEISMIC_DIM_M * (SEISMIC_DIM_E + 1) + lead * SEISMIC_DIM_M * (SEISMIC_DIM_EN + 1);
    const unsigned lane = threadIdx.x % 32;
    const unsigned long long expert = static_cast<unsigned long long>(group) * (blockDim.x / 32) + threadIdx.x / 32;
    if (expert < NEXT_EXPERTS)
        route::prefetch<NEXT_EXPERTS>(next_columns, expert, lane, seismic_words_value);
    seismic_dependency_start();
    route::decode_inverses(next.residual, parts, inverses, seismic_words_value);
    if (expert < NEXT_EXPERTS)
        route::column<NEXT_EXPERTS>(next, inverses, next_columns, expert, predicted, lane,
            seismic_words_value);
    if (!arrive::last(ports.arrivals + 1 + lead, PREDICT_BLOCKS, last))
        return;
    const unsigned rows = static_cast<unsigned>(SEISMIC_DIM_M);
    for (unsigned row = threadIdx.x / 32; row < rows; row += blockDim.x / 32) {
        const float *values = predicted + route::logit_index<NEXT_EXPERTS>(row, 0, seismic_words_value);
        const route::Ranked ranked = route::rank_row<NEXT_EXPERTS>(
            [&](unsigned long long column) { return __ldcg(values + column); }, lane);
        if (lane == 0)
            for (unsigned r = 0; r < K; ++r)
                words[STREAMING_HINT_CHOSEN(row) + r] = ranked.winners[r];
    }
    streaming_hint(rows, ports.rings + lead * SEISMIC_RINGS_STRIDE_0, SEISMIC_RINGS_STRIDE_1,
        ports.step[0], words, threadIdx.x, blockDim.x);
}

// The launch's body: with the next layer's hint blocks (HINTS) or without
// them; resolving a layer of a group that is not whole, or of a whole one
// its sources and routing record alone (WHOLE). `memory` is the launch's dynamic shared
// memory.
template <bool HINTS, bool WHOLE>
__device__ __forceinline__ void route_streamed(const Ports &ports, projection::u8 *memory,
                                               const seismic_words_t &seismic_words_value) {
    __shared__ float parts[SEISMIC_CONST_DECODE_ROWS][route::WARPS];
    __shared__ float inverses[SEISMIC_CONST_DECODE_ROWS];
    __shared__ unsigned last;
    const unsigned predicting = HINTS ? static_cast<unsigned>(SEISMIC_DIM_AHEAD) * PREDICT_BLOCKS : 0u;
    const unsigned group = blockIdx.x;
    if (HINTS && group >= route::BLOCKS && group < route::BLOCKS + predicting) {
        hint_block(ports, (group - route::BLOCKS) / PREDICT_BLOCKS, (group - route::BLOCKS) % PREDICT_BLOCKS, parts,
            inverses, &last, seismic_words_value);
        return;
    }
    if (group >= route::BLOCKS) {
        shared_expert::block(ports.in, ports.shared_gate, ports.shared_up, ports.shared_product, parts, inverses,
            memory, group - route::BLOCKS - predicting, seismic_words_value);
        return;
    }
    __shared__ int resolved[STREAMING_RESOLVE_WORDS(SEISMIC_CONST_DECODE_ROWS * SEISMIC_DIM_K)];
    if (!route::route_block(ports.in, ports.columns, ports.selection, ports.normalized, SEISMIC_RESULT_0_STRIDE_0,
            SEISMIC_RESULT_0_STRIDE_1, ports.logits, ports.arrivals, group, parts, inverses, &last, seismic_words_value))
        return;
    // Resolve: the selection's routes are this block's own stores.
    routed_resolve(ports.bank, WHOLE, resolved, seismic_words_value);
}

#define ROUTED_ROUTE_STREAMED(NAME, HINTS, WHOLE)                                                          \
    extern "C" __global__ void NAME(SEISMIC_KERNEL_PARAMS) {                                               \
        extern __shared__ uint4 memory[];                                                                  \
        route_streamed<HINTS, WHOLE>(ROUTED_ROUTE_PORTS, reinterpret_cast<projection::u8 *>(memory),       \
            seismic_words_value);                                                                          \
    }

ROUTED_ROUTE_STREAMED(routed_route_streamed, true, false)
ROUTED_ROUTE_STREAMED(routed_route_streamed_alone, false, false)
ROUTED_ROUTE_STREAMED(routed_route_whole, true, true)
ROUTED_ROUTE_STREAMED(routed_route_whole_alone, false, true)
