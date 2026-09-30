// Decode output (M <= 8): threadgroup (x, m) owns the TILE output channels
// of tile x for row m and runs K + 1 simdgroups side by side: simdgroup k < K
// projects choice k through its expert's down rows, simdgroup K the shared
// expert's down rows. Each lane group (LANES lanes) of a simdgroup owns ROWS
// channels; its lanes own the packets sub, sub + LANES, ... of every channel's
// row (`routed::project_channels`). Every projection is the K1 GEMV's sum for
// the same LANES (`projection::gemv_body`).
//
// The projections meet in threadgroup memory; after one barrier, the thread
// of each channel publishes
//     residual + selected + round_A(shared) * coefficient,
// where `selected` accumulates score * round_A(projection) in slot order.

#define KERNEL_W0 SEISMIC_EXPERT_DOWN
#define KERNEL_W1 SEISMIC_SHARED_DOWN
#include "lib/routed/routed.h"

#ifdef SEISMIC_FORMING_ROUTED_OUTPUT
template <uint ROWS, uint LANES>
kernel void routed_output(
    device const float *residual [[buffer(SEISMIC_BUFFER_RESIDUAL)]],
    device const uchar *expert_product [[buffer(SEISMIC_BUFFER_EXPERT_PRODUCT)]],
    device const uchar *shared_product [[buffer(SEISMIC_BUFFER_SHARED_PRODUCT)]],
    device const int *routes [[buffer(SEISMIC_BUFFER_ROUTES)]],
    device const float *scores [[buffer(SEISMIC_BUFFER_SCORES)]],
    device const float *coefficient [[buffer(SEISMIC_BUFFER_COEFFICIENT)]],
    device const uchar *expert_down [[buffer(SEISMIC_BUFFER_EXPERT_DOWN)]],
    device const uchar *shared_down [[buffer(SEISMIC_BUFFER_SHARED_DOWN)]],
    device float *value [[buffer(SEISMIC_RESULT_0_BUFFER)]],
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
    threadgroup float projected[K + 1][TILE];
    const ulong m = group.y;
    const uint first = group.x * TILE;
    const uint rows = uint(SEISMIC_DIM_H);

    if (sg < K) {
        const ulong k = sg;
        const ulong expert = ulong(routes[m * SEISMIC_ROUTES_STRIDE_0 + k * SEISMIC_ROUTES_STRIDE_1]);
        const auto in = routed::activation(expert_product
                + (m * SEISMIC_EXPERT_PRODUCT_STRIDE_0 + k * SEISMIC_EXPERT_PRODUCT_STRIDE_1) * A::bytes,
            0, SEISMIC_EXPERT_PRODUCT_STRIDE_2, uint(SEISMIC_DIM_F));
        const auto down = routed::weights<packets::W0>(expert_down, KERNEL_W0_LAYOUT(SEISMIC_DIM_F),
            routed::expert_row(expert, SEISMIC_DIM_H), SEISMIC_DIM_F);
        routed::project_channels<packets::W0, R, L, uint(SEISMIC_DIM_F)>(in, down, first, rows, projected[k],
            lane);
    } else {
        const auto in = routed::activation(shared_product + m * SEISMIC_SHARED_PRODUCT_STRIDE_0 * A::bytes, 0,
            SEISMIC_SHARED_PRODUCT_STRIDE_1, uint(SEISMIC_DIM_S));
        const auto down = routed::weights<packets::W1>(shared_down, KERNEL_W1_LAYOUT(SEISMIC_DIM_S), 0,
            SEISMIC_DIM_S);
        routed::project_channels<packets::W1, R, L, uint(SEISMIC_DIM_S)>(in, down, first, rows, projected[K],
            lane);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n = first + tid;
    if (tid >= TILE || n >= rows)
        return;
    float selected = 0.0f;
    for (uint k = 0; k < K; ++k)
        selected = metal::fma(scores[m * SEISMIC_SCORES_STRIDE_0 + k * SEISMIC_SCORES_STRIDE_1],
            A::round(projected[k][tid]), selected);
    value[m * SEISMIC_RESULT_0_STRIDE_0 + n * SEISMIC_RESULT_0_STRIDE_1] =
        residual[m * SEISMIC_RESIDUAL_STRIDE_0 + n * SEISMIC_RESIDUAL_STRIDE_1] + selected
        + A::round(projected[K][tid]) * coefficient[m * SEISMIC_COEFFICIENT_STRIDE_0];
}
#endif
