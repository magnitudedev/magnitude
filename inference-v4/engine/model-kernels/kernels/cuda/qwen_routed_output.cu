// Decode output (M <= 8); the CUDA form of `metal/qwen_routed_output.metal`.
// Block (x, m) owns the output channels of tile group x for row m: the K1
// GEMV of each choice's down projection in slot order, each published
// (A-rounded) projection weighted by its score, then the shared expert's
// down projection; the channel is residual + selected + round_A(shared) * c.
// The GEMV hands every channel to one fixed lane, which carries the channel's
// running sum in registers across the choices.
#define KERNEL_W0 SEISMIC_EXPERT_DOWN
#define KERNEL_W1 SEISMIC_SHARED_DOWN
#include "common/routed.cuh"

using Shape = projection::GemvShape<4, SEISMIC_TUNE_TPW, SEISMIC_TUNE_KSPLIT, 1>;
using Pro = projection::Plain<ELEMENT_OF(SEISMIC_ELEMENT_A), projection::AllRows>;
constexpr int CARRIED = 2 * SEISMIC_TUNE_TPW;

// The carried slot of channel n: its tile within the group and its row half.
__device__ __forceinline__ unsigned routed_slot(projection::u64 n, projection::u64 first) {
    const projection::u64 local = n - first;
    return (unsigned)((local / 16) * 2 + (local % 16) / 8);
}

struct SelectEpi {
    float *selected;
    projection::u64 first;
    float score;
    __device__ __forceinline__ void operator()(unsigned, projection::u64 n, float value, float) const {
        float &carried = selected[routed_slot(n, first)];
        carried = __fmaf_rn(score, element::Act::round(value), carried);
    }
};

struct FinalEpi {
    const float *selected;
    projection::u64 first;
    float *out;
    const float *residual;
    float coefficient;
    __device__ __forceinline__ void operator()(unsigned, projection::u64 n, float value, float) const {
        out[n] = residual[n] + selected[routed_slot(n, first)] + element::Act::round(value) * coefficient;
    }
};

extern "C" __global__ void qwen_routed_output(SEISMIC_KERNEL_PARAMS) {
    __shared__ projection::GemvShared<Shape, Pro> shared;
    const projection::u64 group = Shape::tile_group();
    if (group >= projection::gemv_groups<Shape>(SEISMIC_DIM_H))
        return;
    const projection::u64 m = blockIdx.y;
    const projection::u64 first = group * Shape::TPW * 16;
    const int *routes = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES));
    const float *scores = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCORES));
    float selected[CARRIED];
#pragma unroll
    for (int slot = 0; slot < CARRIED; ++slot)
        selected[slot] = 0.0f;

    for (projection::u64 k = 0; k < SEISMIC_DIM_K; ++k) {
        // The K-split reduction area is reused by every choice.
        if (Shape::KSPLIT > 1)
            projection::named_barrier(1 + Shape::group(), Shape::KSPLIT * 32);
        const projection::u64 expert = (projection::u64)routes[m * SEISMIC_ROUTES_STRIDE_0 + k * SEISMIC_ROUTES_STRIDE_1];
        const projection::u64 row = routed::expert_row(expert, SEISMIC_DIM_H);
        const Pro pro{SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_PRODUCT)
                          + (m * SEISMIC_EXPERT_PRODUCT_STRIDE_0 + k * SEISMIC_EXPERT_PRODUCT_STRIDE_1) * 2,
                      0, projection::AllRows{}};
        const SelectEpi epi{selected, first, scores[m * SEISMIC_SCORES_STRIDE_0 + k * SEISMIC_SCORES_STRIDE_1]};
        projection::gemv_segment<Shape>(shared, pro, 1u, SEISMIC_DIM_F / 64, group, SEISMIC_DIM_H,
                                KERNEL_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_DOWN) + row * ELEMENT_CAT(KERNEL_W0, _ROW_STRIDE_BYTES)),
                                projection::NoWeight{}, epi);
    }

    if (Shape::KSPLIT > 1)
        projection::named_barrier(1 + Shape::group(), Shape::KSPLIT * 32);
    const Pro pro{SEISMIC_PTR(SEISMIC_BUFFER_SHARED_PRODUCT) + m * SEISMIC_SHARED_PRODUCT_STRIDE_0 * 2, 0,
                  projection::AllRows{}};
    const FinalEpi epi{selected, first,
                       reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER)) + m * SEISMIC_RESULT_0_STRIDE_0,
                       reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL)) + m * SEISMIC_RESIDUAL_STRIDE_0,
                       reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_COEFFICIENT))[m * SEISMIC_COEFFICIENT_STRIDE_0]};
    projection::gemv_segment<Shape>(shared, pro, 1u, SEISMIC_DIM_S / 64, group, SEISMIC_DIM_H,
                            KERNEL_W1_AT(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_DOWN)), projection::NoWeight{}, epi);
}
