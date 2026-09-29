// Decode expansion of gated experts (M <= 8): threadgroup row y = choice
// (m, k), a K1 paired GEMV over the feature tile `x` of expert routes[m, k]'s
// gate and up rows with the activation-generic GLU epilogue.

#define KERNEL_W0 SEISMIC_EXPERT_GATE
#define KERNEL_W1 SEISMIC_EXPERT_UP
#include "lib/routed/routed.h"

#ifdef SEISMIC_FORMING_ROUTED_GATE_UP
template <uint ROWS, uint LANES>
kernel void routed_gate_up(
    device const uchar *normalized [[buffer(SEISMIC_BUFFER_NORMALIZED)]],
    device const int *routes [[buffer(SEISMIC_BUFFER_ROUTES)]],
    device const uchar *expert_gate [[buffer(SEISMIC_BUFFER_EXPERT_GATE)]],
    device const uchar *expert_up [[buffer(SEISMIC_BUFFER_EXPERT_UP)]],
    device uchar *product [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    threadgroup uchar *shared [[threadgroup(0)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    typedef routed::Act A;
    const uint tile = group.x;
    if (ulong(tile) * simdgroups * ROWS * (32u / LANES) >= SEISMIC_DIM_F)
        return;
    const ulong m = group.y / SEISMIC_DIM_K, k = group.y % SEISMIC_DIM_K;
    const ulong expert = ulong(routes[m * SEISMIC_ROUTES_STRIDE_0 + k * SEISMIC_ROUTES_STRIDE_1]);
    const auto in = routed::activation(normalized + m * SEISMIC_NORMALIZED_STRIDE_0 * A::bytes,
        SEISMIC_NORMALIZED_STRIDE_0, SEISMIC_NORMALIZED_STRIDE_1, uint(SEISMIC_DIM_H));
    const auto gate = routed::weights<packets::W0>(expert_gate, KERNEL_W0_LAYOUT(SEISMIC_DIM_H),
        routed::expert_row(expert, SEISMIC_DIM_F), SEISMIC_DIM_H);
    const auto up = routed::weights<packets::W1>(expert_up, KERNEL_W1_LAYOUT(SEISMIC_DIM_H),
        routed::expert_row(expert, SEISMIC_DIM_F), SEISMIC_DIM_H);
    const projection::Glu<A> out{product + (m * SEISMIC_RESULT_0_STRIDE_0 + k * SEISMIC_RESULT_0_STRIDE_1) * A::bytes,
        0, SEISMIC_RESULT_0_STRIDE_2, int(SEISMIC_PARAM_ACTIVATION)};
    projection::gemv_paired_runtime<packets::W0, packets::W1, ROWS, 1, LANES>(in, out, gate, up, 1,
        SEISMIC_DIM_F, SEISMIC_DIM_H, tile, shared, simdgroups, sg, lane);
}
#endif
