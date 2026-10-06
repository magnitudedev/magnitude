// dense_expand: RMS prologue over the `out_rows` residual rows, the
// paired gate/up projection, and act(gate)·up (`activation`: SiLU or
// GELU-tanh).
#define KERNEL_W0 SEISMIC_GATE_WEIGHT
#define KERNEL_W1 SEISMIC_UP_WEIGHT
#include "lib/projection/projection.h"

typedef element::Act activation;
typedef ELEMENT_OF(SEISMIC_NORM) norm_element;

#define DENSE_EXPAND_ARGUMENTS                                                          \
    device const float *residual [[buffer(SEISMIC_BUFFER_RESIDUAL)]],                   \
    device const uchar *norm [[buffer(SEISMIC_BUFFER_NORM)]],                           \
    device const uchar *gate_weight [[buffer(SEISMIC_BUFFER_GATE_WEIGHT)]],             \
    device const uchar *up_weight [[buffer(SEISMIC_BUFFER_UP_WEIGHT)]],                 \
    device const int *out_rows [[buffer(SEISMIC_BUFFER_OUT_ROWS)]],                     \
    device uchar *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],                           \
    device uchar *normalized [[buffer(SEISMIC_BUFFER_SCRATCH_NORMALIZED)]],             \
    device uchar *quantized [[buffer(SEISMIC_BUFFER_SCRATCH_QUANTIZED)]],               \
    device float *row_scales [[buffer(SEISMIC_BUFFER_SCRATCH_ROW_SCALES)]],             \
    device half *block_sums [[buffer(SEISMIC_BUFFER_SCRATCH_BLOCK_SUMS)]],              \
    device float *gate_coefficients [[buffer(SEISMIC_BUFFER_SCRATCH_GATE_COEFFICIENTS)]], \
    device half *gate_biases [[buffer(SEISMIC_BUFFER_SCRATCH_GATE_BIASES)]],            \
    device float *up_coefficients [[buffer(SEISMIC_BUFFER_SCRATCH_UP_COEFFICIENTS)]],   \
    device half *up_biases [[buffer(SEISMIC_BUFFER_SCRATCH_UP_BIASES)]],                \
    device const float *gate_scale [[buffer(SEISMIC_BUFFER_GATE_SCALE)]],               \
    device const float *up_scale [[buffer(SEISMIC_BUFFER_UP_SCALE)]],                   \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

#define DENSE_EXPAND_OPERANDS                                                           \
    projection::Rms<activation, norm_element, projection::SelectedRows> in{residual,    \
        SEISMIC_RESIDUAL_STRIDE_0, SEISMIC_RESIDUAL_STRIDE_1, norm, SEISMIC_NORM_STRIDE_0, \
        as_type<float>(uint(SEISMIC_PARAM_EPS)), uint(SEISMIC_DIM_H), {out_rows}};      \
    const auto out = projection::scaling<(SEISMIC_DIM_GS != 0 || SEISMIC_DIM_US != 0)>::wrap( \
        projection::Glu<activation>{result, SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1, \
            int(SEISMIC_PARAM_ACTIVATION)},                                             \
        projection::scale_factor(gate_scale, SEISMIC_DIM_GS, 0, 0),                     \
        projection::scale_factor(up_scale, SEISMIC_DIM_US, 0, 0));                      \
    projection::Weights<packets::W0> gate{gate_weight, KERNEL_W0_LAYOUT(SEISMIC_DIM_H), \
        uint(SEISMIC_DIM_H)};                                                           \
    projection::Weights<packets::W1> up{up_weight, KERNEL_W1_LAYOUT(SEISMIC_DIM_H), uint(SEISMIC_DIM_H)}

#ifdef SEISMIC_FORMING_DENSE_EXPAND_GEMV
template <uint ROWS, uint LANES>
kernel void dense_expand_gemv(DENSE_EXPAND_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    DENSE_EXPAND_OPERANDS;
    uint rows = uint(SEISMIC_DIM_O);
    PROJECTION_SQUARES_SHARED(squares, decltype(in)::parts);
    projection::threadgroup_squares_runtime(in, rows, squares, simdgroups, sg, lane);
    const auto x = projection::shared_norm(in, squares);
    PROJECTION_FOR_ROWS(rows,
        projection::gemv_paired_runtime<packets::W0, packets::W1, ROWS, MAXM, LANES>(
            x, out, gate, up, rows, uint(SEISMIC_DIM_F), uint(SEISMIC_DIM_H), tile, shared,
            simdgroups, sg, lane));
}
#endif

#ifdef SEISMIC_FORMING_DENSE_EXPAND_BATCH
template <uint BATCH_ROWS>
kernel void dense_expand_batch(DENSE_EXPAND_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    DENSE_EXPAND_OPERANDS;
    PROJECTION_SQUARES_SHARED(squares, decltype(in)::parts);
    projection::threadgroup_squares_runtime(in, uint(SEISMIC_DIM_O), squares, simdgroups, sg, lane);
    const auto x = projection::shared_norm(in, squares);
    projection::gemv_batch_paired_runtime<packets::W0, packets::W1, BATCH_ROWS>(x, out,
        gate, up, uint(SEISMIC_DIM_O), uint(SEISMIC_DIM_F), uint(SEISMIC_DIM_H), tile, shared,
        simdgroups, sg, lane);
}
#endif

#ifdef SEISMIC_FORMING_DENSE_EXPAND_NORMALIZE
kernel void dense_expand_normalize(DENSE_EXPAND_ARGUMENTS,
    uint item [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    PROJECTION_NORMALIZE_SHARED(norms);
    DENSE_EXPAND_OPERANDS;
    projection::device_normalize<256>(in, item, normalized, uint(SEISMIC_DIM_H), norms, thread_index);
}
#endif

#define DENSE_EXPAND_GEMM(TM, TN)                                                       \
    PROJECTION_GEMM_SHARED(shared, TM, TN);                                             \
    DENSE_EXPAND_OPERANDS;                                                              \
    projection::Plain<activation, projection::AllRows> x{normalized, SEISMIC_DIM_H, 1, uint(SEISMIC_DIM_H), {}}; \
    projection::gemm_paired<packets::W0, packets::W1, TM, TN>(x, out, gate, up, uint(SEISMIC_DIM_O), \
        uint(SEISMIC_DIM_F), uint(SEISMIC_DIM_H), tile.y, tile.x, shared, sg, lane)

// 17..64 rows: the fixed small-row tile.
#ifdef SEISMIC_FORMING_DENSE_EXPAND_GEMM_SMALL
kernel void dense_expand_gemm_small(DENSE_EXPAND_ARGUMENTS,
    uint2 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    DENSE_EXPAND_GEMM(projection::small_tile_m, projection::small_tile_n);
}
#endif

#ifdef SEISMIC_FORMING_DENSE_EXPAND_GEMM
template <uint TILE_M, uint TILE_N>
kernel void dense_expand_gemm(DENSE_EXPAND_ARGUMENTS,
    uint2 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    DENSE_EXPAND_GEMM(TILE_M, TILE_N);
}
#endif

// The TALL form past 64 rows: the normalized rows in the tall GEMM's order,
// then its tiles.
#ifdef SEISMIC_FORMING_DENSE_EXPAND_NORMALIZE_TALL
kernel void dense_expand_normalize_tall(DENSE_EXPAND_ARGUMENTS,
    uint item [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    PROJECTION_NORMALIZE_SHARED(norms);
    DENSE_EXPAND_OPERANDS;
    projection::device_normalize<256, projection::TallOrder<activation>>(in, item, normalized, uint(SEISMIC_DIM_H),
        norms, thread_index);
}
#endif

#ifdef SEISMIC_FORMING_DENSE_EXPAND_TALL
template <uint TALL_M, uint TALL_K, uint STAGERS>
kernel void dense_expand_tall(DENSE_EXPAND_ARGUMENTS,
    uint2 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    PROJECTION_GEMM_TALL_SHARED(shared, TALL_K);
    DENSE_EXPAND_OPERANDS;
    projection::Plain<activation, projection::AllRows> x{normalized, SEISMIC_DIM_H, 1, uint(SEISMIC_DIM_H), {}};
    projection::gemm_tall_paired<packets::W0, packets::W1, TALL_M, TALL_K, STAGERS>(
        projection::tall_operand(x, normalized), out, gate, up, uint(SEISMIC_DIM_O), uint(SEISMIC_DIM_F),
        uint(SEISMIC_DIM_H), tile.y, tile.x, shared, sg, lane);
}
#endif

// The INT8 form past 64 rows: the normalized rows (`dense_expand_normalize`)
// quantized per (row, 32 columns), both weights' block scales and biases,
// then the paired int8 tiles.
#define DENSE_EXPAND_INT8_SCRATCH                                                        \
    const projection::int8_scratch gate_scratch{quantized, row_scales, block_sums, gate_coefficients, gate_biases}; \
    const projection::int8_scratch up_scratch{quantized, row_scales, block_sums, up_coefficients, up_biases}
#define DENSE_EXPAND_INT8_AVAILABLE \
    (projection::int8_codes<packets::W0>::available && projection::int8_codes<packets::W1>::available)

#ifdef SEISMIC_FORMING_DENSE_EXPAND_QUANTIZE
kernel void dense_expand_quantize(DENSE_EXPAND_ARGUMENTS,
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    if constexpr (!DENSE_EXPAND_INT8_AVAILABLE)
        return;
    DENSE_EXPAND_INT8_SCRATCH;
    projection::Plain<activation, projection::AllRows> x{normalized, SEISMIC_DIM_H, 1, uint(SEISMIC_DIM_H), {}};
    projection::int8_quantize(x, gate_scratch, uint(SEISMIC_DIM_O), uint(SEISMIC_DIM_H), row, thread_index, lane,
        1.0f);
}
#endif

#ifdef SEISMIC_FORMING_DENSE_EXPAND_COEFFICIENTS
kernel void dense_expand_coefficients(DENSE_EXPAND_ARGUMENTS,
    uint item [[thread_position_in_grid]]) {
    if constexpr (!DENSE_EXPAND_INT8_AVAILABLE)
        return;
    DENSE_EXPAND_OPERANDS;
    DENSE_EXPAND_INT8_SCRATCH;
    projection::int8_coefficients(gate, gate_scratch, uint(SEISMIC_DIM_F), uint(SEISMIC_DIM_H), item);
    projection::int8_coefficients(up, up_scratch, uint(SEISMIC_DIM_F), uint(SEISMIC_DIM_H), item);
}
#endif

#ifdef SEISMIC_FORMING_DENSE_EXPAND_INT8
kernel void dense_expand_int8(DENSE_EXPAND_ARGUMENTS,
    uint2 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float exchange[DENSE_EXPAND_INT8_AVAILABLE ? 2 * 32 * 32 : 1];
    PROJECTION_GEMM_SHARED(shared, 64, 64);
    DENSE_EXPAND_OPERANDS;
    DENSE_EXPAND_INT8_SCRATCH;
    projection::Plain<activation, projection::AllRows> x{normalized, SEISMIC_DIM_H, 1, uint(SEISMIC_DIM_H), {}};
    projection::gemm_int8_paired<packets::W0, packets::W1>(x, out, gate, up, gate_scratch, up_scratch,
        uint(SEISMIC_DIM_O), uint(SEISMIC_DIM_F), uint(SEISMIC_DIM_H), tile.y, tile.x, shared, exchange, sg, lane);
}
#endif
