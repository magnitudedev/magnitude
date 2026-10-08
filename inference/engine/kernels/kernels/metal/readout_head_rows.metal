// readout_head_rows: final RMS prologue over the `out_rows` hidden rows and the
// vocabulary projection into F32 logits. Batched rows normalize once and use
// the shared compressed-code matrix form where supported. Logits are scaled by a present `weight_scale`
// (static WS = 1) and softcapped from the accumulator when `softcap` > 0.
#define KERNEL_W0 SEISMIC_WEIGHT
#include "lib/projection/projection.h"

typedef element::Act activation;
typedef ELEMENT_OF(SEISMIC_NORM) norm_element;

#define HEAD_ROWS_ARGUMENTS                                                             \
    device const float *hidden [[buffer(SEISMIC_BUFFER_HIDDEN)]],                       \
    device const uchar *norm [[buffer(SEISMIC_BUFFER_NORM)]],                           \
    device const uchar *weight [[buffer(SEISMIC_BUFFER_WEIGHT)]],                       \
    device const int *out_rows [[buffer(SEISMIC_BUFFER_OUT_ROWS)]],                     \
    device uchar *logits [[buffer(SEISMIC_RESULT_0_BUFFER)]],                           \
    device uchar *normalized [[buffer(SEISMIC_BUFFER_SCRATCH_NORMALIZED)]],             \
    device const float *weight_scale [[buffer(SEISMIC_BUFFER_WEIGHT_SCALE)]],           \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

#define HEAD_ROWS_OPERANDS                                                              \
    const uint k = uint(SEISMIC_DIM_D);                                                 \
    projection::Rms<activation, norm_element, projection::SelectedRows> in{hidden, SEISMIC_HIDDEN_STRIDE_0, \
        SEISMIC_HIDDEN_STRIDE_1, norm, SEISMIC_NORM_STRIDE_0,                           \
        as_type<float>(uint(SEISMIC_PARAM_EPSILON)), k, {out_rows}};                    \
    const auto out = projection::scaling<(SEISMIC_DIM_WS != 0)>::wrap(                  \
        projection::Logits{logits, SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1, \
            as_type<float>(uint(SEISMIC_PARAM_SOFTCAP))},                               \
        projection::scale_factor(weight_scale, SEISMIC_DIM_WS, 0, 0), 1.0f);            \
    projection::Weights<packets::W0> w{weight, KERNEL_W0_LAYOUT(k), k}

// The GEMV of a launch that serves COUNT (ONE, SEVERAL) rows.
#define HEAD_ROWS_GEMV(ROWS, LANES, COUNT)                                              \
    HEAD_ROWS_OPERANDS;                                                                 \
    uint rows = uint(SEISMIC_DIM_O);                                                    \
    PROJECTION_SQUARES_SHARED(squares, decltype(in)::parts);                            \
    projection::threadgroup_squares_runtime(in, rows, squares, simdgroups, sg, lane);   \
    const auto x = projection::shared_norm(in, squares);                                \
    PROJECTION_FOR_##COUNT##_ROWS(rows,                                                 \
        projection::gemv_runtime<packets::W0, ROWS, MAXM, LANES>(                       \
            x, out, w, rows, uint(SEISMIC_DIM_V), k, tile, shared, simdgroups, sg, lane))

// One row.
#ifdef SEISMIC_FORMING_READOUT_HEAD_ROWS_GEMV
template <uint ROWS, uint LANES>
kernel void readout_head_rows_gemv(HEAD_ROWS_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    HEAD_ROWS_GEMV(ROWS, LANES, ONE);
}
#endif

// Three rows up to BATCH_FROM: the same GEMV under this launch's mapping.
#ifdef SEISMIC_FORMING_READOUT_HEAD_ROWS_GEMV_ROWS
template <uint ROWS, uint LANES>
kernel void readout_head_rows_gemv_rows(HEAD_ROWS_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    HEAD_ROWS_GEMV(ROWS, LANES, SEVERAL);
}
#endif

// Two rows: the same GEMV under this launch's mapping.
#ifdef SEISMIC_FORMING_READOUT_HEAD_ROWS_GEMV_PAIR
template <uint ROWS, uint LANES>
kernel void readout_head_rows_gemv_pair(HEAD_ROWS_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    HEAD_ROWS_GEMV(ROWS, LANES, PAIR);
}
#endif

#ifdef SEISMIC_FORMING_READOUT_HEAD_ROWS_BATCH
template <uint BATCH_ROWS, uint BATCH_PARTS>
kernel void readout_head_rows_batch(HEAD_ROWS_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    HEAD_ROWS_OPERANDS;
    projection::Plain<activation, projection::AllRows> x{normalized, k, 1, k, {}};
    if constexpr (projection::matrix_codes<packets::W0>::available) {
        projection::gemv_matrix<packets::W0, BATCH_PARTS>(x, out, w,
            uint(SEISMIC_DIM_O), uint(SEISMIC_DIM_V), k, tile, shared, simdgroups, sg, lane);
        return;
    }
    if (tile * simdgroups * BATCH_ROWS * 8u >= uint(SEISMIC_DIM_V))
        return;
    projection::gemv_batch_runtime<packets::W0, BATCH_ROWS>(x, out, w,
        uint(SEISMIC_DIM_O), uint(SEISMIC_DIM_V), k, tile, shared, simdgroups, sg, lane);
}
#endif

#ifdef SEISMIC_FORMING_READOUT_HEAD_ROWS_STAGE
kernel void readout_head_rows_stage(HEAD_ROWS_ARGUMENTS,
    uint item [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    PROJECTION_NORMALIZE_SHARED(norms);
    HEAD_ROWS_OPERANDS;
    projection::device_normalize<256>(in, item, normalized, k, norms, thread_index);
}
#endif

#ifdef SEISMIC_FORMING_READOUT_HEAD_ROWS_GEMM
template <uint TILE_M, uint TILE_N>
kernel void readout_head_rows_gemm(HEAD_ROWS_ARGUMENTS,
    uint2 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    PROJECTION_GEMM_SHARED(shared, TILE_M, TILE_N);
    HEAD_ROWS_OPERANDS;
    projection::Plain<activation, projection::AllRows> x{normalized, k, 1, k, {}};
    projection::gemm<packets::W0, TILE_M, TILE_N>(x, out, w, uint(SEISMIC_DIM_O),
        uint(SEISMIC_DIM_V), k, tile.y, tile.x, shared, sg, lane);
}
#endif
