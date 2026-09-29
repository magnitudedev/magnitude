// Decode output of the general routed feed-forward (M <= 8); the CUDA form
// of `metal/routed_down.metal`. Block (x, m) owns the output channels of tile
// group x for row m: the K1 GEMV of each choice's down projection in slot
// order, each published (A-rounded) projection weighted by its weight; the
// last choice publishes base + selected. The GEMV hands every channel to one
// fixed lane, which carries the channel's running sum in registers across
// the choices.
#define KERNEL_W0 SEISMIC_EXPERT_DOWN
#include "lib/routed/routed.cuh"

using Pro = projection::Plain<ELEMENT_OF(SEISMIC_ELEMENT_A), projection::AllRows>;
using Published = ELEMENT_OF(SEISMIC_ELEMENT_R);

// The carried slot of channel n: its tile within the group and its row half.
__device__ __forceinline__ unsigned routed_slot(projection::u64 n, projection::u64 first) {
    const projection::u64 local = n - first;
    return (unsigned)((local / 16) * 2 + (local % 16) / 8);
}

struct CarryEpi {
    float *selected;
    projection::u64 first;
    float weight;
    __device__ __forceinline__ void operator()(unsigned, projection::u64 n, float value, float) const {
        float &carried = selected[routed_slot(n, first)];
        carried = __fmaf_rn(weight, element::Act::round(value), carried);
    }
};

// The last choice: accumulates, then publishes base + selected in R.
struct BaseEpi {
    const float *selected;
    projection::u64 first;
    float weight;
    projection::u8 *out;
    const float *base;
    __device__ __forceinline__ void operator()(unsigned, projection::u64 n, float value, float) const {
        element::put<Published>(
            out, n, base[n] + __fmaf_rn(weight, element::Act::round(value), selected[routed_slot(n, first)]));
    }
};

#ifdef SEISMIC_FORMING_ROUTED_DOWN
template <int TPW, int KSPLIT>
__global__ void routed_down(SEISMIC_KERNEL_PARAMS) {
    using Shape = projection::GemvShape<4, TPW, KSPLIT, 1>;
    constexpr int CARRIED = 2 * TPW;
    __shared__ projection::GemvShared<Shape, Pro> shared;
    const projection::u64 group = Shape::tile_group();
    if (group >= projection::gemv_groups<Shape>(SEISMIC_DIM_H))
        return;
    const projection::u64 m = blockIdx.y;
    const projection::u64 first = group * Shape::TPW * 16;
    const int *routes = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES));
    const float *weights = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_WEIGHTS));
    float selected[CARRIED];
#pragma unroll
    for (int slot = 0; slot < CARRIED; ++slot)
        selected[slot] = 0.0f;
    for (projection::u64 k = 0; k < SEISMIC_DIM_K; ++k) {
        // The K-split reduction area is reused by every choice.
        if (Shape::KSPLIT > 1)
            projection::named_barrier(1 + Shape::group(), Shape::KSPLIT * 32);
        const projection::u64 expert = (projection::u64)routes[m * SEISMIC_ROUTES_STRIDE_0 + k * SEISMIC_ROUTES_STRIDE_1];
        const Pro pro{SEISMIC_PTR(SEISMIC_BUFFER_PRODUCT) + (m * SEISMIC_PRODUCT_STRIDE_0 + k * SEISMIC_PRODUCT_STRIDE_1) * 2,
                      0, projection::AllRows{}};
        const float weight = weights[m * SEISMIC_WEIGHTS_STRIDE_0 + k * SEISMIC_WEIGHTS_STRIDE_1];
        const auto down = KERNEL_W0_MATRIX(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_DOWN), expert);
        if (k + 1 < SEISMIC_DIM_K) {
            const CarryEpi epi{selected, first, weight};
            projection::gemv_segment<Shape>(shared, pro, 1u, SEISMIC_DIM_F / 64, group, SEISMIC_DIM_H, down,
                                            projection::NoWeight{}, epi);
        } else {
            const BaseEpi epi{selected, first, weight,
                              SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER) + m * SEISMIC_RESULT_0_STRIDE_0 * Published::bytes,
                              reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_BASE)) + m * SEISMIC_BASE_STRIDE_0};
            projection::gemv_segment<Shape>(shared, pro, 1u, SEISMIC_DIM_F / 64, group, SEISMIC_DIM_H, down,
                                            projection::NoWeight{}, epi);
        }
    }
}
#endif
