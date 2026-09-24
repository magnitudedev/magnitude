// Decode expansion (M <= 8): the K selected experts' gate/up projections and
// the shared expert's, each with the SiLU . mul epilogue, in one launch.
// Threadgroup row y < M * K is choice (m, k): a K1 paired GEMV over the
// feature tile `x` of expert routes[m, k]'s gate and up rows. Row y = M * K
// is the shared expert, whose weights feed all M activation rows.

#define KERNEL_W0 SEISMIC_EXPERT_GATE
#define KERNEL_W1 SEISMIC_EXPERT_UP
#define KERNEL_W2 SEISMIC_SHARED_GATE
#define KERNEL_W3 SEISMIC_SHARED_UP
#include "common/routed.h"

kernel void qwen_routed_expand(
    device const uchar *normalized [[buffer(SEISMIC_BUFFER_NORMALIZED)]],
    device const int *routes [[buffer(SEISMIC_BUFFER_ROUTES)]],
    device const uchar *expert_gate [[buffer(SEISMIC_BUFFER_EXPERT_GATE)]],
    device const uchar *expert_up [[buffer(SEISMIC_BUFFER_EXPERT_UP)]],
    device const uchar *shared_gate [[buffer(SEISMIC_BUFFER_SHARED_GATE)]],
    device const uchar *shared_up [[buffer(SEISMIC_BUFFER_SHARED_UP)]],
    device uchar *expert_product [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    device uchar *shared_product [[buffer(SEISMIC_RESULT_1_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    threadgroup uchar *shared [[threadgroup(0)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint SG = SEISMIC_TUNE_SIMDGROUPS;
    constexpr uint R = SEISMIC_TUNE_ROWS;
    constexpr uint L = SEISMIC_TUNE_LANES;
    // Weight rows per threadgroup.
    constexpr uint TILE = projection::gemv_threadgroup_rows<SG, R, L>();
    typedef routed::Act A;
    const uint tile = group.x;
    const ulong choices = SEISMIC_DIM_M * SEISMIC_DIM_K;

    if (group.y < choices) {
        if (ulong(tile) * TILE >= SEISMIC_DIM_F) return;
        const ulong m = group.y / SEISMIC_DIM_K, k = group.y % SEISMIC_DIM_K;
        const ulong expert = ulong(routes[m * SEISMIC_ROUTES_STRIDE_0 + k * SEISMIC_ROUTES_STRIDE_1]);
        const auto in = routed::activation(normalized + m * SEISMIC_NORMALIZED_STRIDE_0 * A::bytes,
            SEISMIC_NORMALIZED_STRIDE_0, SEISMIC_NORMALIZED_STRIDE_1, uint(SEISMIC_DIM_H));
        const auto gate = routed::weights<packets::W0>(expert_gate, KERNEL_W0_LAYOUT(SEISMIC_DIM_H),
            routed::expert_row(expert, SEISMIC_DIM_F), SEISMIC_DIM_H);
        const auto up = routed::weights<packets::W1>(expert_up, KERNEL_W1_LAYOUT(SEISMIC_DIM_H),
            routed::expert_row(expert, SEISMIC_DIM_F), SEISMIC_DIM_H);
        const projection::SiluMul<A> out{expert_product
            + (m * SEISMIC_RESULT_0_STRIDE_0 + k * SEISMIC_RESULT_0_STRIDE_1) * A::bytes,
            0, SEISMIC_RESULT_0_STRIDE_2};
        projection::gemv_paired<packets::W0, packets::W1, SG, R, 1, L>(in, out, gate, up, 1,
            SEISMIC_DIM_F, SEISMIC_DIM_H, tile, shared, sg, lane);
        return;
    }

    if (ulong(tile) * TILE >= SEISMIC_DIM_S) return;
    const auto in = routed::activation(normalized, SEISMIC_NORMALIZED_STRIDE_0, SEISMIC_NORMALIZED_STRIDE_1,
        uint(SEISMIC_DIM_H));
    const auto gate = routed::weights<packets::W2>(shared_gate, KERNEL_W2_LAYOUT(SEISMIC_DIM_H), 0,
        SEISMIC_DIM_H);
    const auto up = routed::weights<packets::W3>(shared_up, KERNEL_W3_LAYOUT(SEISMIC_DIM_H), 0,
        SEISMIC_DIM_H);
    const projection::SiluMul<A> out{shared_product, SEISMIC_RESULT_1_STRIDE_0,
        SEISMIC_RESULT_1_STRIDE_1};
    const uint rows = uint(SEISMIC_DIM_M);
    PROJECTION_FOR_ROWS(rows,
        (projection::gemv_paired<packets::W2, packets::W3, SG, R, MAXM, L>(in, out, gate, up, rows,
            SEISMIC_DIM_S, SEISMIC_DIM_H, tile, shared, sg, lane)));
}
