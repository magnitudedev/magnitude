// Decode output of the general routed feed-forward (M <= 8); the CUDA form
// of `metal/routed_down.metal`. The choices' down projections run in parallel
// (block (x, m K + k) owns the output channels of tile group x for choice k
// of row m) and publish each A-rounded projection in `projected`; the combine
// then weights them in slot order and publishes base + selected, the sum the
// portable body forms.
#define KERNEL_W0 SEISMIC_EXPERT_DOWN
#include "lib/routed/routed.cuh"

using Pro = projection::Plain<ELEMENT_OF(SEISMIC_ELEMENT_A), projection::AllRows>;
using Published = ELEMENT_OF(SEISMIC_ELEMENT_R);

// One choice's projection, A-rounded, into its [M, K, H] slot.
struct ProjectedEpi {
    float *projected;
    __device__ __forceinline__ void operator()(unsigned, projection::u64 n, float value, float) const {
        projected[n] = element::Act::round(value);
    }
};

#ifdef SEISMIC_FORMING_ROUTED_DOWN
template <int TPW, int KSPLIT>
__global__ void routed_down(SEISMIC_KERNEL_PARAMS) {
    using Shape = projection::GemvShape<4, TPW, KSPLIT, 1>;
    __shared__ projection::GemvShared<Shape, Pro> shared;
    const projection::u64 group = Shape::tile_group();
    if (group >= projection::gemv_groups<Shape>(SEISMIC_DIM_H))
        return;
    const projection::u64 m = blockIdx.y / SEISMIC_DIM_K, k = blockIdx.y % SEISMIC_DIM_K;
    const int *routes = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES));
    const projection::u64 expert = (projection::u64)routes[m * SEISMIC_ROUTES_STRIDE_0 + k * SEISMIC_ROUTES_STRIDE_1];
    const Pro pro{SEISMIC_PTR(SEISMIC_BUFFER_PRODUCT) + (m * SEISMIC_PRODUCT_STRIDE_0 + k * SEISMIC_PRODUCT_STRIDE_1) * 2,
                  0, projection::AllRows{}};
    const auto down = KERNEL_W0_MATRIX(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_DOWN), expert);
    float *projected = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PROJECTED)) +
                       (m * SEISMIC_DIM_K + k) * SEISMIC_DIM_H;
    projection::gemv_segment<Shape>(shared, pro, 1u, SEISMIC_DIM_F / 64, group, SEISMIC_DIM_H, down,
                                    projection::NoWeight{}, ProjectedEpi{projected});
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_DOWN_COMBINE
// Thread (n, m): R(base + sum_k w_k * projected_k), the sum in slot order.
__global__ void routed_down_combine(SEISMIC_KERNEL_PARAMS) {
    const projection::u64 n = (projection::u64)blockIdx.x * blockDim.x + threadIdx.x;
    const projection::u64 m = blockIdx.y;
    if (n >= SEISMIC_DIM_H)
        return;
    const float *weights = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_WEIGHTS));
    const float *projected = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PROJECTED)) +
                             m * SEISMIC_DIM_K * SEISMIC_DIM_H;
    float selected = 0.0f;
    for (projection::u64 k = 0; k < SEISMIC_DIM_K; ++k)
        selected = __fmaf_rn(weights[m * SEISMIC_WEIGHTS_STRIDE_0 + k * SEISMIC_WEIGHTS_STRIDE_1],
                             projected[k * SEISMIC_DIM_H + n], selected);
    const float *base = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_BASE)) + m * SEISMIC_BASE_STRIDE_0;
    element::put<Published>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER) + m * SEISMIC_RESULT_0_STRIDE_0 * Published::bytes, n,
                            base[n] + selected);
}
#endif
