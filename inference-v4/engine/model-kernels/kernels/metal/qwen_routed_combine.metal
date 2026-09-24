// Prefill combine. L1 (`qwen_routed_combine_shared`): the K1 paired GEMM of
// the shared expert's gate/up over all M rows with SiLU . mul into the
// `shared_product` scratch. L2 (`qwen_routed_combine`): the K1 GEMM of the
// shared down projection, whose epilogue unpermutes the grouped expert
// outputs of each row in slot order and publishes
//     residual + selected + round_A(shared) * coefficient.

#define ROUTED_W0 SEISMIC_SHARED_DOWN
#define ROUTED_W1 SEISMIC_SHARED_GATE
#define ROUTED_W2 SEISMIC_SHARED_UP
#include "common/routed.h"

namespace routed {

struct output_combined {
    device float *y;
    device const float *residual;
    device const uchar *expert_output;
    device const int *inverse;
    device const float *scores;
    device const float *coefficient;
    constant ulong *seismic_words;
    // Row m's grouped output of choice k starts at element `row_of(m, k)`.
    ulong row_of(uint m, ulong k) const {
        const ulong position = ulong(inverse[m * SEISMIC_INVERSE_STRIDE_0 + k * SEISMIC_INVERSE_STRIDE_1]);
        return (position / SEISMIC_DIM_T) * SEISMIC_EXPERT_OUTPUT_STRIDE_0
            + (position % SEISMIC_DIM_T) * SEISMIC_EXPERT_OUTPUT_STRIDE_1;
    }
    float score(uint m, ulong k) const { return scores[m * SEISMIC_SCORES_STRIDE_0 + k * SEISMIC_SCORES_STRIDE_1]; }
    void publish(uint m, uint n, float selected, float value) const {
        y[m * SEISMIC_RESULT_0_STRIDE_0 + n * SEISMIC_RESULT_0_STRIDE_1] =
            residual[m * SEISMIC_RESIDUAL_STRIDE_0 + n * SEISMIC_RESIDUAL_STRIDE_1] + selected
            + Act::round(value) * coefficient[m * SEISMIC_COEFFICIENT_STRIDE_0];
    }
    void store(uint m, uint n, float value) const {
        device const typename Act::storage *outputs =
            reinterpret_cast<device const typename Act::storage *>(expert_output);
        float selected = 0.0f;
        for (ulong k = 0; k < SEISMIC_DIM_K; ++k)
            selected = metal::fma(score(m, k),
                Act::load(outputs[row_of(m, k) + ulong(n) * SEISMIC_EXPERT_OUTPUT_STRIDE_2]), selected);
        publish(m, n, selected, value);
    }
    // Columns n and n + 1 share each choice's position and score; with unit
    // column stride their two outputs are one 4-byte load.
    void store2(uint m, uint n, float first, float second) const {
        device const typename Act::storage *outputs =
            reinterpret_cast<device const typename Act::storage *>(expert_output);
        float2 selected = float2(0.0f);
        for (ulong k = 0; k < SEISMIC_DIM_K; ++k) {
            const ulong at = row_of(m, k) + ulong(n) * SEISMIC_EXPERT_OUTPUT_STRIDE_2;
            float2 projected;
            if (SEISMIC_EXPERT_OUTPUT_STRIDE_2 == 1 && (at & 1u) == 0) {
                const uint word = *reinterpret_cast<device const uint *>(outputs + at);
                projected = float2(Act::load(as_type<typename Act::storage>(ushort(word & 0xffffu))),
                    Act::load(as_type<typename Act::storage>(ushort(word >> 16))));
            } else {
                projected = float2(Act::load(outputs[at]),
                    Act::load(outputs[at + SEISMIC_EXPERT_OUTPUT_STRIDE_2]));
            }
            selected = metal::fma(float2(score(m, k)), projected, selected);
        }
        publish(m, n, selected.x, first);
        publish(m, n + 1u, selected.y, second);
    }
};

} // namespace routed

kernel void qwen_routed_combine_shared(
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
    const auto gate = routed::rows_from<routed::W1>(shared_gate, ROUTED_W1_LAYOUT(SEISMIC_DIM_H), 0,
        SEISMIC_DIM_H);
    const auto up = routed::rows_from<routed::W2>(shared_up, ROUTED_W2_LAYOUT(SEISMIC_DIM_H), 0, SEISMIC_DIM_H);
    const projection::output_paired<routed::Act> out{shared_product, SEISMIC_DIM_S, 1};
    projection::gemm_paired<routed::W1, routed::W2, BM, BN>(in, out, gate, up, uint(SEISMIC_DIM_M),
        uint(SEISMIC_DIM_S), uint(SEISMIC_DIM_H), group.y, group.x, tile_memory, sg, lane);
}

kernel void qwen_routed_combine(
    device const float *residual [[buffer(SEISMIC_BUFFER_RESIDUAL)]],
    device const uchar *expert_output [[buffer(SEISMIC_BUFFER_EXPERT_OUTPUT)]],
    device const int *inverse [[buffer(SEISMIC_BUFFER_INVERSE)]],
    device const float *scores [[buffer(SEISMIC_BUFFER_SCORES)]],
    device const float *coefficient [[buffer(SEISMIC_BUFFER_COEFFICIENT)]],
    device const uchar *shared_down [[buffer(SEISMIC_BUFFER_SHARED_DOWN)]],
    device float *value [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    device const uchar *shared_product [[buffer(SEISMIC_BUFFER_SCRATCH_SHARED_PRODUCT)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint BM = SEISMIC_TUNE_TILE_M;
    constexpr uint BN = SEISMIC_TUNE_TILE_N;
    PROJECTION_GEMM_SHARED(tile_memory, BM, BN);
    const auto in = routed::activation(shared_product, SEISMIC_DIM_S, 1, uint(SEISMIC_DIM_S));
    const auto down = routed::rows_from<routed::W0>(shared_down, ROUTED_W0_LAYOUT(SEISMIC_DIM_S), 0,
        SEISMIC_DIM_S);
    const routed::output_combined out{value, residual, expert_output, inverse, scores, coefficient,
        seismic_words};
    projection::gemm<routed::W0, BM, BN>(in, out, down, uint(SEISMIC_DIM_M), uint(SEISMIC_DIM_H),
        uint(SEISMIC_DIM_S), group.y, group.x, tile_memory, sg, lane);
}
