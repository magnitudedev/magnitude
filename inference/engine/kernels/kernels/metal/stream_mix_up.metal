// stream_mix_up: the low-rank rows through `up`, each output column the gate
// of its stream's column, publishing the per-stream partials (`Mix`).
#define KERNEL_W0 SEISMIC_UP_WEIGHT
#include "lib/projection/projection.h"

typedef element::Act activation;
typedef ELEMENT_OF(SEISMIC_NORM) norm_element;

#define STREAM_MIX_UP_ARGUMENTS                                                         \
    device const uchar *low [[buffer(SEISMIC_BUFFER_LOW)]],                             \
    device const uchar *up_weight [[buffer(SEISMIC_BUFFER_UP_WEIGHT)]],                 \
    device const float *hidden [[buffer(SEISMIC_BUFFER_HIDDEN)]],                       \
    device const float *inverse [[buffer(SEISMIC_BUFFER_INVERSE)]],                     \
    device const uchar *norm [[buffer(SEISMIC_BUFFER_NORM)]],                           \
    device float *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],                           \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

#define STREAM_MIX_UP_OPERANDS                                                          \
    const uint k = uint(SEISMIC_DIM_LR);                                                 \
    const uint n = uint(SEISMIC_DIM_S * SEISMIC_DIM_H);                                 \
    projection::Plain<activation, projection::AllRows> in{low, SEISMIC_LOW_STRIDE_0,    \
        SEISMIC_LOW_STRIDE_1, k, {}};                                                   \
    projection::Mix<norm_element> out{result, SEISMIC_RESULT_0_STRIDE_0,                \
        SEISMIC_RESULT_0_STRIDE_1, hidden, SEISMIC_HIDDEN_STRIDE_0, SEISMIC_HIDDEN_STRIDE_2, \
        inverse, SEISMIC_INVERSE_STRIDE_0, SEISMIC_INVERSE_STRIDE_1, norm,              \
        SEISMIC_NORM_STRIDE_1, uint(SEISMIC_DIM_H), float(SEISMIC_DIM_S)};              \
    projection::Weights<packets::W0> w{up_weight, KERNEL_W0_LAYOUT(k), k}

#ifdef SEISMIC_FORMING_STREAM_MIX_UP_GEMV
template <uint ROWS, uint LANES>
kernel void stream_mix_up_gemv(STREAM_MIX_UP_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    STREAM_MIX_UP_OPERANDS;
    const uint rows = uint(SEISMIC_DIM_M);
    PROJECTION_FOR_ROWS(rows, projection::gemv_runtime<packets::W0, ROWS, MAXM, LANES>(
        in, out, w, rows, n, k, tile, shared, simdgroups, sg, lane));
}
#endif

#ifdef SEISMIC_FORMING_STREAM_MIX_UP_BATCH
template <uint BATCH_ROWS>
kernel void stream_mix_up_batch(STREAM_MIX_UP_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    STREAM_MIX_UP_OPERANDS;
    projection::gemv_batch_runtime<packets::W0, BATCH_ROWS>(in, out, w, uint(SEISMIC_DIM_M), n, k, tile,
        shared, simdgroups, sg, lane);
}
#endif

#ifdef SEISMIC_FORMING_STREAM_MIX_UP_GEMM
template <uint TILE_M, uint TILE_N>
kernel void stream_mix_up_gemm(STREAM_MIX_UP_ARGUMENTS,
    uint2 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    PROJECTION_GEMM_SHARED(shared, TILE_M, TILE_N);
    STREAM_MIX_UP_OPERANDS;
    projection::gemm<packets::W0, TILE_M, TILE_N>(in, out, w, uint(SEISMIC_DIM_M), n, k, tile.y, tile.x,
        shared, sg, lane);
}
#endif
