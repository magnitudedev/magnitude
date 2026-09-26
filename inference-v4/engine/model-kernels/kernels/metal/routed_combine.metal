// Prefill combine. L1 (`routed_combine_shared`): the K1 paired GEMM of
// the shared expert's gate/up over all M rows with SiLU . mul into the
// `shared_product` scratch. L2 (`routed_combine_down`): the K1 GEMM of
// the shared down projection into the `shared_output` scratch [M, H], in A.
// L3 (`routed_combine`): the grouped expert outputs of each row
// unpermuted in slot order, published as
//     residual + selected + shared * coefficient.
// An L3 thread owns 8 consecutive columns of one row: it reads each choice's
// position and score once and the choice's 8 expert outputs as one 16-byte
// load, so the gather moves whole rows of expert output.

#define KERNEL_W0 SEISMIC_SHARED_DOWN
#define KERNEL_W1 SEISMIC_SHARED_GATE
#define KERNEL_W2 SEISMIC_SHARED_UP
#include "lib/routed/routed.h"

// Threads of an L3 threadgroup (8 columns each).
constant constexpr uint combine_threads = 256;

kernel void routed_combine_shared(
    device const uchar *normalized [[buffer(SEISMIC_BUFFER_NORMALIZED)]],
    device const uchar *shared_gate [[buffer(SEISMIC_BUFFER_SHARED_GATE)]],
    device const uchar *shared_up [[buffer(SEISMIC_BUFFER_SHARED_UP)]],
    device uchar *shared_product [[buffer(SEISMIC_BUFFER_SCRATCH_SHARED_PRODUCT)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint BM = SEISMIC_TUNE_TILE_M;
    constexpr uint BN = SEISMIC_TUNE_TILE_N;
    PROJECTION_GEMM_SHARED(tile_memory, BM, BN);
    const auto in = routed::activation(normalized, SEISMIC_NORMALIZED_STRIDE_0, SEISMIC_NORMALIZED_STRIDE_1,
        uint(SEISMIC_DIM_H));
    const auto gate = routed::weights<packets::W1>(shared_gate, KERNEL_W1_LAYOUT(SEISMIC_DIM_H), 0,
        SEISMIC_DIM_H);
    const auto up = routed::weights<packets::W2>(shared_up, KERNEL_W2_LAYOUT(SEISMIC_DIM_H), 0, SEISMIC_DIM_H);
    const projection::SiluMul<routed::Act> out{shared_product, SEISMIC_DIM_S, 1};
    projection::gemm_paired<packets::W1, packets::W2, BM, BN>(in, out, gate, up, uint(SEISMIC_DIM_M),
        uint(SEISMIC_DIM_S), uint(SEISMIC_DIM_H), group.y, group.x, tile_memory, sg, lane);
}

kernel void routed_combine_down(
    device const uchar *shared_down [[buffer(SEISMIC_BUFFER_SHARED_DOWN)]],
    device const uchar *shared_product [[buffer(SEISMIC_BUFFER_SCRATCH_SHARED_PRODUCT)]],
    device uchar *shared_output [[buffer(SEISMIC_BUFFER_SCRATCH_SHARED_OUTPUT)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint BM = SEISMIC_TUNE_TILE_M;
    constexpr uint BN = SEISMIC_TUNE_TILE_N;
    PROJECTION_GEMM_SHARED(tile_memory, BM, BN);
    const auto in = routed::activation(shared_product, SEISMIC_DIM_S, 1, uint(SEISMIC_DIM_S));
    const auto down = routed::weights<packets::W0>(shared_down, KERNEL_W0_LAYOUT(SEISMIC_DIM_S), 0,
        SEISMIC_DIM_S);
    const projection::Store<routed::Act> out{shared_output, SEISMIC_DIM_H, 1, 0};
    projection::gemm<packets::W0, BM, BN>(in, out, down, uint(SEISMIC_DIM_M), uint(SEISMIC_DIM_H),
        uint(SEISMIC_DIM_S), group.y, group.x, tile_memory, sg, lane);
}

kernel void routed_combine(
    device const float *residual [[buffer(SEISMIC_BUFFER_RESIDUAL)]],
    device const uchar *expert_output [[buffer(SEISMIC_BUFFER_EXPERT_OUTPUT)]],
    device const int *inverse [[buffer(SEISMIC_BUFFER_INVERSE)]],
    device const float *scores [[buffer(SEISMIC_BUFFER_SCORES)]],
    device const float *coefficient [[buffer(SEISMIC_BUFFER_COEFFICIENT)]],
    device float *value [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    device const uchar *shared_output [[buffer(SEISMIC_BUFFER_SCRATCH_SHARED_OUTPUT)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    typedef routed::Act A;
    typedef typename A::storage S;
    const uint m = group.y;
    const uint n = 8u * (group.x * combine_threads + thread_index);
    const uint columns = uint(SEISMIC_DIM_H);
    if (n >= columns)
        return;
    const bool vector = SEISMIC_EXPERT_OUTPUT_STRIDE_2 == 1 && (SEISMIC_EXPERT_OUTPUT_STRIDE_1 & 7u) == 0
        && (SEISMIC_EXPERT_OUTPUT_STRIDE_0 & 7u) == 0;
    float4 selected_even = float4(0.0f), selected_odd = float4(0.0f);
    for (ulong k = 0; k < SEISMIC_DIM_K; ++k) {
        const ulong position = ulong(inverse[m * SEISMIC_INVERSE_STRIDE_0 + k * SEISMIC_INVERSE_STRIDE_1]);
        const float score = scores[m * SEISMIC_SCORES_STRIDE_0 + k * SEISMIC_SCORES_STRIDE_1];
        device const S *row = reinterpret_cast<device const S *>(expert_output)
            + (position / SEISMIC_DIM_T) * SEISMIC_EXPERT_OUTPUT_STRIDE_0
            + (position % SEISMIC_DIM_T) * SEISMIC_EXPERT_OUTPUT_STRIDE_1;
        float4 even, odd;
        projection::load8_storage<A>(row, SEISMIC_EXPERT_OUTPUT_STRIDE_2, n, columns, vector, even, odd);
        selected_even = metal::fma(float4(score), even, selected_even);
        selected_odd = metal::fma(float4(score), odd, selected_odd);
    }
    float4 shared_even, shared_odd;
    projection::load8_storage<A>(reinterpret_cast<device const S *>(shared_output) + ulong(m) * columns, 1, n,
        columns, (columns & 7u) == 0, shared_even, shared_odd);
    const float c = coefficient[m * SEISMIC_COEFFICIENT_STRIDE_0];
    const float selected[8] = {selected_even.x, selected_odd.x, selected_even.y, selected_odd.y,
        selected_even.z, selected_odd.z, selected_even.w, selected_odd.w};
    const float projected[8] = {shared_even.x, shared_odd.x, shared_even.y, shared_odd.y, shared_even.z,
        shared_odd.z, shared_even.w, shared_odd.w};
    for (uint i = 0; i < 8u && n + i < columns; ++i) {
        const ulong column = n + i;
        value[m * SEISMIC_RESULT_0_STRIDE_0 + column * SEISMIC_RESULT_0_STRIDE_1] =
            residual[m * SEISMIC_RESIDUAL_STRIDE_0 + column * SEISMIC_RESIDUAL_STRIDE_1] + selected[i]
            + projected[i] * c;
    }
}
