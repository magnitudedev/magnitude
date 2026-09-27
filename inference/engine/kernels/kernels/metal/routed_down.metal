// Decode output of the general routed feed-forward (M <= 8): threadgroup
// (x, m) owns output channels of tile x for row m. It runs the K1 GEMV of
// each choice's down projection in slot order, weighting each published
// (A-rounded) projection by its weight; the last choice publishes
//     base + selected.
// The GEMV stores every output channel from one fixed lane, so that lane
// carries the channel's running sum in registers across the choices.

#define KERNEL_W0 SEISMIC_EXPERT_DOWN
#include "lib/routed/routed.h"

namespace routed {

// Accumulates weight * round_A(projection) into the storing lane's registers.
struct CarryEpi {
    thread float *selected;
    uint first;
    float weight;
    void store(uint, uint n, float value) const {
        selected[n - first] = metal::fma(weight, Act::round(value), selected[n - first]);
    }
};

// The last choice: accumulates, then publishes base + selected in R.
template <typename R>
struct BaseEpi {
    thread const float *selected;
    uint first;
    float weight;
    device typename R::storage *y;
    ulong y_stride;
    device const float *base;
    ulong base_stride;
    void store(uint, uint n, float value) const {
        y[n * y_stride] = R::store(base[n * base_stride] + metal::fma(weight, Act::round(value), selected[n - first]));
    }
};

} // namespace routed

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
    threadgroup uchar *shared [[threadgroup(0)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint R = ROWS;
    constexpr uint L = LANES;
    typedef routed::Act A;
    const uint tile = group.x;
    const ulong m = group.y;
    // The R output channels of this lane's group (the GEMV's row ownership).
    const uint first = ((tile * simdgroups + sg) * (32u / L) + lane / L) * R;
    float selected[R];
    for (uint r = 0; r < R; ++r)
        selected[r] = 0.0f;
    for (ulong k = 0; k < SEISMIC_DIM_K; ++k) {
        const ulong expert = ulong(routes[m * SEISMIC_ROUTES_STRIDE_0 + k * SEISMIC_ROUTES_STRIDE_1]);
        const auto in = routed::activation(product + (m * SEISMIC_PRODUCT_STRIDE_0 + k * SEISMIC_PRODUCT_STRIDE_1) * A::bytes,
            0, SEISMIC_PRODUCT_STRIDE_2, uint(SEISMIC_DIM_F));
        const auto down = routed::weights<packets::W0>(expert_down, KERNEL_W0_LAYOUT(SEISMIC_DIM_F),
            routed::expert_row(expert, SEISMIC_DIM_H), SEISMIC_DIM_F);
        const float weight = weights[m * SEISMIC_WEIGHTS_STRIDE_0 + k * SEISMIC_WEIGHTS_STRIDE_1];
        if (k + 1 < SEISMIC_DIM_K) {
            const routed::CarryEpi out{selected, first, weight};
            projection::gemv_runtime<packets::W0, R, 1, L>(in, out, down, 1, SEISMIC_DIM_H, SEISMIC_DIM_F, tile,
                shared, simdgroups, sg, lane);
        } else {
            typedef ELEMENT_OF(SEISMIC_ELEMENT_R) Published;
            const routed::BaseEpi<Published> out{selected, first, weight,
                reinterpret_cast<device typename Published::storage *>(value) + m * SEISMIC_RESULT_0_STRIDE_0,
                SEISMIC_RESULT_0_STRIDE_1, base + m * SEISMIC_BASE_STRIDE_0, SEISMIC_BASE_STRIDE_1};
            projection::gemv_runtime<packets::W0, R, 1, L>(in, out, down, 1, SEISMIC_DIM_H, SEISMIC_DIM_F, tile,
                shared, simdgroups, sg, lane);
        }
    }
}
#endif
