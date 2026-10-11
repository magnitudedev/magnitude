// head_logits_rows: the vocabulary projection of already-normalized feature
// rows (the draft head's readout) into F32 logits.
#define KERNEL_W0 SEISMIC_WEIGHT
#include "lib/projection/projection.h"

typedef element::Act activation;

#define HEAD_LOGITS_ARGUMENTS                                                           \
    device const uchar *features [[buffer(SEISMIC_BUFFER_FEATURES)]],                   \
    device const uchar *weight [[buffer(SEISMIC_BUFFER_WEIGHT)]],                       \
    device uchar *logits [[buffer(SEISMIC_RESULT_0_BUFFER)]],                           \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

#define HEAD_LOGITS_OPERANDS                                                            \
    const uint k = uint(SEISMIC_DIM_D);                                                 \
    projection::Plain<activation, projection::AllRows> in{features, SEISMIC_FEATURES_STRIDE_0, \
        SEISMIC_FEATURES_STRIDE_1, k, {}};                                              \
    projection::Store<element::F32> out{logits, SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1, 0}; \
    projection::Weights<packets::W0> w{weight, KERNEL_W0_LAYOUT(k), k}

#ifdef SEISMIC_FORMING_HEAD_LOGITS_ROWS_GEMV
template <uint ROWS, uint LANES>
kernel void head_logits_rows_gemv(HEAD_LOGITS_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    HEAD_LOGITS_OPERANDS;
    uint rows = uint(SEISMIC_DIM_O);
    PROJECTION_FOR_ROWS(rows,
        projection::gemv_runtime<packets::W0, ROWS, MAXM, LANES>(
            in, out, w, rows, uint(SEISMIC_DIM_V), k, tile, shared, simdgroups, sg, lane));
}
#endif

#ifdef SEISMIC_FORMING_HEAD_LOGITS_ROWS_BATCH
template <uint BATCH_ROWS>
kernel void head_logits_rows_batch(HEAD_LOGITS_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    HEAD_LOGITS_OPERANDS;
    projection::gemv_batch_runtime<packets::W0, BATCH_ROWS>(in, out, w,
        uint(SEISMIC_DIM_O), uint(SEISMIC_DIM_V), k, tile, shared, simdgroups, sg, lane);
}
#endif

#ifdef SEISMIC_FORMING_HEAD_LOGITS_ROWS_GEMM
template <uint TILE_M, uint TILE_N>
kernel void head_logits_rows_gemm(HEAD_LOGITS_ARGUMENTS,
    uint2 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    PROJECTION_GEMM_SHARED(shared, TILE_M, TILE_N);
    HEAD_LOGITS_OPERANDS;
    projection::gemm<packets::W0, TILE_M, TILE_N>(in, out, w, uint(SEISMIC_DIM_O),
        uint(SEISMIC_DIM_V), k, tile.y, tile.x, shared, sg, lane);
}
#endif
