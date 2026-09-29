// per_layer_gate: `stage` rounds the F32 hidden rows to A in scratch; the
// projection launches (`project_rows`' row classes) project them through
// the gate rows with the activated-product epilogue
// A(A(act(A(gate))) * inputs[m, layer, p]).
#define KERNEL_W0 SEISMIC_GATE_WEIGHT
#include "lib/projection/projection.h"

typedef element::Act activation;

#define PER_LAYER_GATE_ARGUMENTS                                                        \
    device const float *hidden [[buffer(SEISMIC_BUFFER_HIDDEN)]],                       \
    device const uchar *gate_weight [[buffer(SEISMIC_BUFFER_GATE_WEIGHT)]],             \
    device const float *inputs [[buffer(SEISMIC_BUFFER_INPUTS)]],                       \
    device uchar *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],                           \
    device uchar *rounded [[buffer(SEISMIC_BUFFER_SCRATCH_ROUNDED)]],                   \
    device const float *gate_scale [[buffer(SEISMIC_BUFFER_GATE_SCALE)]],               \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

#define PER_LAYER_GATE_OPERANDS                                                         \
    const uint k = uint(SEISMIC_DIM_D);                                                 \
    projection::Plain<activation, projection::AllRows> in{rounded, SEISMIC_DIM_D, 1, k, {}}; \
    const auto out = projection::scaling<(SEISMIC_DIM_GS != 0)>::wrap(                  \
        projection::ActivatedMul<activation>{result, SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1, \
            inputs + ulong(int(SEISMIC_PARAM_LAYER)) * SEISMIC_INPUTS_STRIDE_1, SEISMIC_INPUTS_STRIDE_0, \
            SEISMIC_INPUTS_STRIDE_2, int(SEISMIC_PARAM_ACTIVATION)},                    \
        projection::scale_factor(gate_scale, SEISMIC_DIM_GS, 0, 0), 1.0f);              \
    projection::Weights<packets::W0> w{gate_weight, KERNEL_W0_LAYOUT(k), k}

#ifdef SEISMIC_FORMING_PER_LAYER_GATE_STAGE
kernel void per_layer_gate_stage(PER_LAYER_GATE_ARGUMENTS,
    uint item [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    device typename activation::storage *row =
        reinterpret_cast<device typename activation::storage *>(rounded) + ulong(item) * SEISMIC_DIM_D;
    for (uint i = thread_index; i < uint(SEISMIC_DIM_D); i += 256u)
        row[i] = activation::store(hidden[ulong(item) * SEISMIC_HIDDEN_STRIDE_0 + ulong(i) * SEISMIC_HIDDEN_STRIDE_1]);
}
#endif

#ifdef SEISMIC_FORMING_PER_LAYER_GATE_GEMV
template <uint ROWS, uint LANES>
kernel void per_layer_gate_gemv(PER_LAYER_GATE_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    PER_LAYER_GATE_OPERANDS;
    uint rows = uint(SEISMIC_DIM_M);
    PROJECTION_FOR_ROWS(rows,
        projection::gemv_runtime<packets::W0, ROWS, MAXM, LANES>(
            in, out, w, rows, uint(SEISMIC_DIM_P), k, tile, shared, simdgroups, sg, lane));
}
#endif

#ifdef SEISMIC_FORMING_PER_LAYER_GATE_BATCH
template <uint BATCH_ROWS>
kernel void per_layer_gate_batch(PER_LAYER_GATE_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    PER_LAYER_GATE_OPERANDS;
    projection::gemv_batch_runtime<packets::W0, BATCH_ROWS>(in, out, w,
        uint(SEISMIC_DIM_M), uint(SEISMIC_DIM_P), k, tile, shared, simdgroups, sg, lane);
}
#endif

#define PER_LAYER_GATE_GEMM(TM, TN)                                                     \
    PROJECTION_GEMM_SHARED(shared, TM, TN);                                             \
    PER_LAYER_GATE_OPERANDS;                                                            \
    projection::gemm<packets::W0, TM, TN>(in, out, w, uint(SEISMIC_DIM_M), uint(SEISMIC_DIM_P), k, tile.y, \
        tile.x, shared, sg, lane)

// 17..64 rows: the fixed small-row tile.
#ifdef SEISMIC_FORMING_PER_LAYER_GATE_GEMM_SMALL
kernel void per_layer_gate_gemm_small(PER_LAYER_GATE_ARGUMENTS,
    uint2 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    PER_LAYER_GATE_GEMM(projection::small_tile_m, projection::small_tile_n);
}
#endif

#ifdef SEISMIC_FORMING_PER_LAYER_GATE_GEMM
template <uint TILE_M, uint TILE_N>
kernel void per_layer_gate_gemm(PER_LAYER_GATE_ARGUMENTS,
    uint2 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    PER_LAYER_GATE_GEMM(TILE_M, TILE_N);
}
#endif
