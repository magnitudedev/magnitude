// Expert streaming's decode selection of the general routed form
// (`routed_select_streamed` in routed.seismic); the CUDA form of
// `metal/routed_select_streamed.metal`: `routed_select`'s decode launches
// (`lib/routed/selection.cuh`), so its bits, then resolve and the next two
// layers' hints (AHEAD 2).
//   routed_select_streamed_gemv: `routed_select_gemv`'s blocks, then per
//     lead (one layer ahead, then two) ceil(EN / SIMDGROUPS) forming that
//     layer's logits over this residual through its router norm and router,
//     into the lead's [SPLIT, M, EN] partial planes after this layer's
//     [SPLIT, M, E].
//   routed_select_streamed_select: block 0 selects every row, one warp per
//     row (M <= DECODE_ROWS), then resolves and assigns
//     (`streaming/resolve.hpp`); block 1 + lead ranks the lead's layer's
//     rows as its selection ranks them (`select::rank_row` with that layer's
//     bias) and publishes the hints into the lead's ring row
//     (`streaming/hint.hpp`).
//   routed_select_whole: a whole group's layer's, as the select launch but
//     block 0 resolves its hits and routing record alone.
// While the next layer holds every expert, each launch is encoded without
// the next layer's blocks, which follow this layer's.

#include "lib/routed/selection.cuh"
#include "lib/streaming/control.cuh"
#include "../streaming/hint.hpp"
#include "../streaming/resolve.hpp"
#include "lib/routed/resolve.cuh"

// Block 0's group words: the resolve's, or a hint block's the hints'.
constexpr unsigned SELECT_WORDS =
    STREAMING_RESOLVE_WORDS(SEISMIC_CONST_DECODE_ROWS * SEISMIC_DIM_K) > STREAMING_HINT_WORDS(SEISMIC_CONST_DECODE_ROWS)
        ? STREAMING_RESOLVE_WORDS(SEISMIC_CONST_DECODE_ROWS * SEISMIC_DIM_K)
        : STREAMING_HINT_WORDS(SEISMIC_CONST_DECODE_ROWS);

// E and EN (the next streamed layer's experts) are static.
constexpr unsigned COLUMN_GROUPS =
    (static_cast<unsigned>(SEISMIC_DIM_E) + SEISMIC_TUNE_SIMDGROUPS - 1) / SEISMIC_TUNE_SIMDGROUPS;
constexpr unsigned NEXT_EXPERTS = static_cast<unsigned>(SEISMIC_DIM_EN);
constexpr unsigned NEXT_GROUPS = (NEXT_EXPERTS + SEISMIC_TUNE_SIMDGROUPS - 1) / SEISMIC_TUNE_SIMDGROUPS;
// Where lead `lead`'s partial planes begin: after this layer's [SPLIT, M,
// E] and each earlier lead's [SPLIT, M, EN].
__device__ __forceinline__ unsigned long long predicted_at(unsigned lead, const seismic_words_t &seismic_words_value) {
    return static_cast<unsigned long long>(SEISMIC_TUNE_SPLIT) * SEISMIC_DIM_M * (SEISMIC_DIM_E + lead * NEXT_EXPERTS);
}

extern "C" __global__ void routed_select_streamed_gemv(SEISMIC_KERNEL_PARAMS) {
    const float *residual = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL));
    const unsigned char *norm = SEISMIC_PTR(SEISMIC_BUFFER_NORM);
    const unsigned char *router_norm = SEISMIC_PTR(SEISMIC_BUFFER_ROUTER_NORM);
    const unsigned char *router = SEISMIC_PTR(SEISMIC_BUFFER_ROUTER);
    const unsigned char *next_router_norm = SEISMIC_PTR(SEISMIC_BUFFER_NEXT_ROUTER_NORM);
    const unsigned char *next_router = SEISMIC_PTR(SEISMIC_BUFFER_NEXT_ROUTER);
    const unsigned char *after_router_norm = SEISMIC_PTR(SEISMIC_BUFFER_AFTER_ROUTER_NORM);
    const unsigned char *after_router = SEISMIC_PTR(SEISMIC_BUFFER_AFTER_ROUTER);
    unsigned char *normalized = SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER);
    float *logits = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_LOGITS));
    __shared__ float parts[SEISMIC_CONST_DECODE_ROWS][SELECT_WARPS];
    __shared__ float inverses[SEISMIC_CONST_DECODE_ROWS];
    __shared__ alignas(16) Act::storage staged[SEISMIC_CONST_DECODE_ROWS][GEMV_SHARE];
    decode_inverses(residual, parts, inverses, seismic_words_value);
    if (blockIdx.x < COLUMN_GROUPS) {
        select_columns(residual, inverses, router_norm, SEISMIC_ROUTER_NORM_STRIDE_0, router, SEISMIC_ROUTER_STRIDE_0,
                       norm, blockIdx.x == 0 ? normalized : nullptr, logits, blockIdx.x, staged, seismic_words_value);
        return;
    }
    const unsigned lead = (blockIdx.x - COLUMN_GROUPS) / NEXT_GROUPS;
    const unsigned group = (blockIdx.x - COLUMN_GROUPS) % NEXT_GROUPS;
    if (lead == 0)
        select_columns<NEXT_EXPERTS>(residual, inverses, next_router_norm, SEISMIC_NEXT_ROUTER_NORM_STRIDE_0,
                                     next_router, SEISMIC_NEXT_ROUTER_STRIDE_0, norm, nullptr,
                                     logits + predicted_at(0, seismic_words_value), group, staged, seismic_words_value);
    else
        select_columns<NEXT_EXPERTS>(residual, inverses, after_router_norm, SEISMIC_AFTER_ROUTER_NORM_STRIDE_0,
                                     after_router, SEISMIC_AFTER_ROUTER_STRIDE_0, norm, nullptr,
                                     logits + predicted_at(1, seismic_words_value), group, staged, seismic_words_value);
}

// Block 1 + `lead` of a select launch: the hints of the layer `lead` + 1
// ahead, by its selection bias (`bias`, of stride `bias_stride`), through
// its group words (`STREAMING_HINT_WORDS`) into the lead's ring row.
__device__ __forceinline__ void hint_block(const float *logits, unsigned lead, const float *bias,
                                           unsigned long long bias_stride, int *rings, int sequence, int *words,
                                           const seismic_words_t &seismic_words_value) {
    constexpr unsigned K = SEISMIC_DIM_K;
    const unsigned lane = threadIdx.x % 32;
    const float *predicted = logits + predicted_at(lead, seismic_words_value);
    const unsigned rows = static_cast<unsigned>(SEISMIC_DIM_M);
    for (unsigned row = threadIdx.x / 32; row < rows; row += blockDim.x / 32) {
        const select::Selected<K> selected = select::rank_row<NEXT_EXPERTS, K>(
            static_cast<int>(SEISMIC_PARAM_SCORE),
            [&](unsigned expert) { return select_logit<NEXT_EXPERTS>(predicted, row, expert, seismic_words_value); },
            [&](int expert) { return bias[static_cast<unsigned long long>(expert) * bias_stride]; },
            lane);
        if (lane == 0)
            for (unsigned r = 0; r < K; ++r)
                words[STREAMING_HINT_CHOSEN(row) + r] = selected.winners[K - 1 - r];
    }
    streaming_hint(rows, rings + lead * SEISMIC_RINGS_STRIDE_0, SEISMIC_RINGS_STRIDE_1, sequence, words,
                   threadIdx.x, blockDim.x);
}

// The ports both select launches read.
#define ROUTED_SELECT_PORTS                                                                                        \
    const float *bias = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_BIAS));                         \
    const float *expert_scale = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_SCALE));         \
    int *routes = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES));                                     \
    float *weights = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_WEIGHTS));                               \
    const int *step = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_STEP));                             \
    const float *next_bias = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_NEXT_BIAS));               \
    const float *after_bias = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_AFTER_BIAS));             \
    int *rings = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_RINGS));                                       \
    const float *logits = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_LOGITS));             \
    const unsigned warp = threadIdx.x / 32;                                                                        \
    const unsigned rows = static_cast<unsigned>(SEISMIC_DIM_M)

// Block 0 selects every row, then resolves and assigns (WHOLE: its hits and
// routing record alone); block 1 + lead publishes the lead's hints. In each kernel's
// own scope, where its ports are.
#define ROUTED_SELECT_STREAMED(WHOLE)                                                                              \
    ROUTED_SELECT_PORTS;                                                                                           \
    __shared__ int words[SELECT_WORDS];                                                                            \
    if (blockIdx.x >= 1) {                                                                                         \
        const unsigned lead = blockIdx.x - 1;                                                                      \
        hint_block(logits, lead, lead == 0 ? next_bias : after_bias,                                               \
            lead == 0 ? SEISMIC_NEXT_BIAS_STRIDE_0 : SEISMIC_AFTER_BIAS_STRIDE_0, rings, step[0], words,           \
            seismic_words_value);                                                                                  \
        return;                                                                                                    \
    }                                                                                                              \
    if (warp < rows)                                                                                               \
        select_row(warp, logits, bias, expert_scale, routes, weights, seismic_words_value);             \
    /* Resolve: the selection's routes are this block's own stores. */                                             \
    routed_resolve(ROUTED_RESOLVE_PORTS, (WHOLE), words, seismic_words_value)

extern "C" __global__ void routed_select_streamed_select(SEISMIC_KERNEL_PARAMS) {
    ROUTED_SELECT_STREAMED(false);
}

extern "C" __global__ void routed_select_whole(SEISMIC_KERNEL_PARAMS) {
    ROUTED_SELECT_STREAMED(true);
}
