// Decode output of the general routed feed-forward (M <= 8): threadgroup
// (x, m) owns the TILE output channels of tile x for row m and runs K
// simdgroups side by side, simdgroup k projecting choice k through its
// expert's down rows (`routed::project_channels`: each projection is the K1
// GEMV's sum for the same LANES). The projections meet in threadgroup
// memory; after one barrier, the thread of each channel publishes
//     base + selected
// in R, where `selected` accumulates weight * round_A(projection) in slot
// order.

#define KERNEL_W0 SEISMIC_EXPERT_DOWN
#include "lib/routed/routed.h"

#ifdef SEISMIC_FORMING_ROUTED_DOWN
template <uint ROWS, uint LANES>
kernel void routed_down(
    device const float *base [[buffer(SEISMIC_BUFFER_BASE)]],
    device const uchar *product [[buffer(SEISMIC_BUFFER_PRODUCT)]],
    device const int *routes [[buffer(SEISMIC_BUFFER_ROUTES)]],
    device const float *weights [[buffer(SEISMIC_BUFFER_WEIGHTS)]],
    device const uchar *expert_down [[buffer(SEISMIC_BUFFER_EXPERT_DOWN)]],
    device uchar *value [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint R = ROWS;
    constexpr uint L = LANES;
    // Output channels per threadgroup.
    constexpr uint TILE = (32u / L) * R;
    constexpr uint K = uint(SEISMIC_DIM_K);
    typedef routed::Act A;
    typedef ELEMENT_OF(SEISMIC_ELEMENT_R) Published;
    threadgroup float projected[K][TILE];
    const ulong m = group.y;
    const uint first = group.x * TILE;
    const uint rows = uint(SEISMIC_DIM_H);

    const ulong k = sg;
    const ulong expert = ulong(routes[m * SEISMIC_ROUTES_STRIDE_0 + k * SEISMIC_ROUTES_STRIDE_1]);
    const auto in = routed::activation(product + (m * SEISMIC_PRODUCT_STRIDE_0 + k * SEISMIC_PRODUCT_STRIDE_1) * A::bytes,
        0, SEISMIC_PRODUCT_STRIDE_2, uint(SEISMIC_DIM_F));
    const auto down = routed::weights<packets::W0>(expert_down, KERNEL_W0_LAYOUT(SEISMIC_DIM_F),
        routed::expert_row(expert, SEISMIC_DIM_H), SEISMIC_DIM_F);
    routed::project_channels<packets::W0, R, L, uint(SEISMIC_DIM_F)>(in, down, first, rows, projected[k], lane);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n = first + tid;
    if (tid >= TILE || n >= rows)
        return;
    float selected = 0.0f;
    for (uint choice = 0; choice < K; ++choice)
        selected = metal::fma(weights[m * SEISMIC_WEIGHTS_STRIDE_0 + choice * SEISMIC_WEIGHTS_STRIDE_1],
            A::round(projected[choice][tid]), selected);
    reinterpret_cast<device typename Published::storage *>(value)[m * SEISMIC_RESULT_0_STRIDE_0
        + n * SEISMIC_RESULT_0_STRIDE_1] = Published::store(base[m * SEISMIC_BASE_STRIDE_0 + n * SEISMIC_BASE_STRIDE_1]
        + selected);
}
#endif
