// qwen_recurrent_output: gated per-head RMS·SiLU(z) prologue over the mixed
// recurrence output, the output projection, and the F32 residual.
#define KERNEL_W0 SEISMIC_OUTPUT_WEIGHT
#include "common/projection.h"

typedef element::Act activation;
typedef ELEMENT_OF(SEISMIC_RECURRENT_NORM) norm_element;
// The GEMV and batched launches reduce each head's norm across the staging
// lanes that load it (projection::LaneNorm): W / 8 lanes, a power of two.
static_assert(SEISMIC_DIM_W % 8 == 0 && SEISMIC_DIM_W <= 256 && ((SEISMIC_DIM_W / 8) & (SEISMIC_DIM_W / 8 - 1)) == 0,
    "qwen_recurrent_output reduces a head over W / 8 lanes, a power of two up to 32");

#define RECURRENT_OUTPUT_ARGUMENTS                                                      \
    device const float *hidden [[buffer(SEISMIC_BUFFER_HIDDEN)]],                       \
    device const uchar *projected [[buffer(SEISMIC_BUFFER_PROJECTION)]],                \
    device const uchar *mixed [[buffer(SEISMIC_BUFFER_MIXED)]],                         \
    device const uchar *recurrent_norm [[buffer(SEISMIC_BUFFER_RECURRENT_NORM)]],       \
    device const uchar *output_weight [[buffer(SEISMIC_BUFFER_OUTPUT_WEIGHT)]],         \
    device float *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],                           \
    device uchar *normalized [[buffer(SEISMIC_BUFFER_SCRATCH_NORMALIZED)]],             \
    device float *partials [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIALS)]],                 \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

#define RECURRENT_OUTPUT_OPERANDS                                                       \
    const uint k = uint(SEISMIC_DIM_NV * SEISMIC_DIM_W);                                \
    projection::GatedRms<activation, norm_element, projection::AllRows> in{mixed, SEISMIC_MIXED_STRIDE_0, \
        SEISMIC_MIXED_STRIDE_1, SEISMIC_MIXED_STRIDE_2, projected,                      \
        SEISMIC_PROJECTION_STRIDE_0, SEISMIC_PROJECTION_STRIDE_1,                       \
        ulong((2 * SEISMIC_DIM_NK + SEISMIC_DIM_NV) * SEISMIC_DIM_W), recurrent_norm,   \
        SEISMIC_RECURRENT_NORM_STRIDE_0, as_type<float>(uint(SEISMIC_PARAM_EPSILON)),   \
        uint(SEISMIC_DIM_NV), uint(SEISMIC_DIM_W), {}};                                 \
    projection::Residual<activation, projection::AllRows> out{result, SEISMIC_RESULT_0_STRIDE_0, \
        SEISMIC_RESULT_0_STRIDE_1, hidden, SEISMIC_HIDDEN_STRIDE_0,                     \
        SEISMIC_HIDDEN_STRIDE_1, {}};                                                   \
    projection::Weights<packets::W0> w{output_weight, KERNEL_W0_LAYOUT(k), k}

kernel void qwen_recurrent_output_gemv(RECURRENT_OUTPUT_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    RECURRENT_OUTPUT_OPERANDS;
    uint rows = uint(SEISMIC_DIM_M);
    projection::LaneNorm<decltype(in)> x{in};
    PROJECTION_FOR_ROWS(rows,
        projection::gemv<packets::W0, SEISMIC_TUNE_SIMDGROUPS, SEISMIC_TUNE_ROWS, MAXM, SEISMIC_TUNE_LANES>(
            x, out, w, rows, uint(SEISMIC_DIM_H), k, tile, shared, sg, lane));
}

kernel void qwen_recurrent_output_batch(RECURRENT_OUTPUT_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    RECURRENT_OUTPUT_OPERANDS;
    projection::LaneNorm<decltype(in)> x{in};
    projection::gemv_batch<packets::W0, SEISMIC_TUNE_BATCH_SIMDGROUPS, SEISMIC_TUNE_BATCH_ROWS>(x, out, w,
        uint(SEISMIC_DIM_M), uint(SEISMIC_DIM_H), k, tile, shared, sg, lane);
}

kernel void qwen_recurrent_output_stage(RECURRENT_OUTPUT_ARGUMENTS,
    uint item [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    PROJECTION_NORMALIZE_SHARED(norms);
    RECURRENT_OUTPUT_OPERANDS;
    projection::device_normalize<32>(in, item, normalized, k, norms, thread_index);
}

kernel void qwen_recurrent_output_gemm(RECURRENT_OUTPUT_ARGUMENTS,
    uint3 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint TM = SEISMIC_TUNE_TILE_M, TN = SEISMIC_TUNE_TILE_N;
    PROJECTION_GEMM_SHARED(shared, TM, TN);
    RECURRENT_OUTPUT_OPERANDS;
    projection::Plain<activation, projection::AllRows> x{normalized, k, 1, k, {}};
    const uint m = uint(SEISMIC_DIM_M), n = uint(SEISMIC_DIM_H);
    if (SEISMIC_TUNE_SPLIT == 1)
        projection::gemm<packets::W0, TM, TN>(x, out, w, m, n, k, tile.y, tile.x, shared, sg, lane);
    else
        projection::gemm_part<packets::W0, TM, TN>(x, partials, w, m, n, k, SEISMIC_TUNE_SPLIT, tile.z,
            tile.y, tile.x, shared, sg, lane);
}

kernel void qwen_recurrent_output_finalize(RECURRENT_OUTPUT_ARGUMENTS,
    uint index [[thread_position_in_grid]]) {
    RECURRENT_OUTPUT_OPERANDS;
    projection::gemm_reduce(out, partials, uint(SEISMIC_DIM_M), uint(SEISMIC_DIM_H), SEISMIC_TUNE_SPLIT, index);
}
