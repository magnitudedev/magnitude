// Decode output (M <= 8): threadgroup (x, m) owns output channels of tile x
// for row m. It runs the K1 GEMV of each choice's down projection in slot
// order, weighting each published (A-rounded) projection by its score, then
// the shared expert's down projection, and publishes
//     residual + selected + round_A(shared) * coefficient.
// The GEMV stores every output channel from one fixed lane, so that lane
// carries the channel's running sum in registers across the choices.

#define KERNEL_W0 SEISMIC_EXPERT_DOWN
#define KERNEL_W1 SEISMIC_SHARED_DOWN
#include "common/routed.h"

namespace routed {

// Accumulates score * round_A(projection) into the storing lane's registers.
template <uint R>
struct SelectEpi {
    thread float *selected;
    uint first;
    float score;
    void store(uint, uint n, float value) const {
        selected[n - first] = metal::fma(score, Act::round(value), selected[n - first]);
    }
};

// Publishes the channel from the carried selection and the shared projection.
template <uint R>
struct FinalEpi {
    thread const float *selected;
    uint first;
    device float *y;
    ulong y_stride;
    device const float *residual;
    ulong residual_stride;
    float coefficient;
    void store(uint, uint n, float value) const {
        y[n * y_stride] = residual[n * residual_stride] + selected[n - first]
            + Act::round(value) * coefficient;
    }
};

} // namespace routed

kernel void qwen_routed_output(
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
    threadgroup uchar *shared [[threadgroup(0)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint SG = SEISMIC_TUNE_SIMDGROUPS;
    constexpr uint R = SEISMIC_TUNE_ROWS;
    constexpr uint L = SEISMIC_TUNE_LANES;
    typedef routed::Act A;
    const uint tile = group.x;
    const ulong m = group.y;
    // The R output channels of this lane's group (the GEMV's row ownership).
    const uint first = ((tile * SG + sg) * (32u / L) + lane / L) * R;
    float selected[R];
    for (uint r = 0; r < R; ++r) selected[r] = 0.0f;

    for (ulong k = 0; k < SEISMIC_DIM_K; ++k) {
        const ulong expert = ulong(routes[m * SEISMIC_ROUTES_STRIDE_0 + k * SEISMIC_ROUTES_STRIDE_1]);
        const auto in = routed::activation(expert_product
                + (m * SEISMIC_EXPERT_PRODUCT_STRIDE_0 + k * SEISMIC_EXPERT_PRODUCT_STRIDE_1) * A::bytes,
            0, SEISMIC_EXPERT_PRODUCT_STRIDE_2, uint(SEISMIC_DIM_F));
        const auto down = routed::weights<packets::W0>(expert_down, KERNEL_W0_LAYOUT(SEISMIC_DIM_F),
            routed::expert_row(expert, SEISMIC_DIM_H), SEISMIC_DIM_F);
        const routed::SelectEpi<R> out{selected, first,
            scores[m * SEISMIC_SCORES_STRIDE_0 + k * SEISMIC_SCORES_STRIDE_1]};
        projection::gemv<packets::W0, SG, R, 1, L>(in, out, down, 1, SEISMIC_DIM_H, SEISMIC_DIM_F, tile,
            shared, sg, lane);
    }

    const auto in = routed::activation(shared_product + m * SEISMIC_SHARED_PRODUCT_STRIDE_0 * A::bytes, 0,
        SEISMIC_SHARED_PRODUCT_STRIDE_1, uint(SEISMIC_DIM_S));
    const auto down = routed::weights<packets::W1>(shared_down, KERNEL_W1_LAYOUT(SEISMIC_DIM_S), 0,
        SEISMIC_DIM_S);
    const routed::FinalEpi<R> out{selected, first, value + m * SEISMIC_RESULT_0_STRIDE_0,
        SEISMIC_RESULT_0_STRIDE_1, residual + m * SEISMIC_RESIDUAL_STRIDE_0, SEISMIC_RESIDUAL_STRIDE_1,
        coefficient[m * SEISMIC_COEFFICIENT_STRIDE_0]};
    projection::gemv<packets::W1, SG, R, 1, L>(in, out, down, 1, SEISMIC_DIM_H, SEISMIC_DIM_S, tile, shared,
        sg, lane);
}
