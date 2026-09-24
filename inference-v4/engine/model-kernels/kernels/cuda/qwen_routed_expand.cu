// Decode expansion (M <= 8); the CUDA form of `metal/qwen_routed_expand.metal`.
// Block row y < M * K is choice (m, k): the K1 paired GEMV over expert
// routes[m, k]'s gate/up rows for one activation row. Row y = M * K is the
// shared expert over all M rows.
#define KERNEL_W0 SEISMIC_EXPERT_GATE
#define KERNEL_W1 SEISMIC_EXPERT_UP
#define KERNEL_W2 SEISMIC_SHARED_GATE
#define KERNEL_W3 SEISMIC_SHARED_UP
#include "common/routed.cuh"

using Shape = projection::GemvShape<4, SEISMIC_TUNE_TPW, SEISMIC_TUNE_KSPLIT, 1>;
using Pro = projection::Plain<ELEMENT_OF(SEISMIC_ELEMENT_A), projection::AllRows>;
using Epi = projection::SiluMul<ELEMENT_OF(SEISMIC_ELEMENT_A)>;

extern "C" __global__ void qwen_routed_expand(SEISMIC_KERNEL_PARAMS) {
    __shared__ projection::GemvShared<Shape, Pro> shared;
    const projection::u8 *normalized = SEISMIC_PTR(SEISMIC_BUFFER_NORMALIZED);
    const int *routes = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES));
    const projection::u64 group = Shape::tile_group();
    const projection::u64 choices = SEISMIC_DIM_M * SEISMIC_DIM_K;

    if (blockIdx.y < choices) {
        if (group >= projection::gemv_groups<Shape>(SEISMIC_DIM_F))
            return;
        const projection::u64 m = blockIdx.y / SEISMIC_DIM_K, k = blockIdx.y % SEISMIC_DIM_K;
        const projection::u64 expert = (projection::u64)routes[m * SEISMIC_ROUTES_STRIDE_0 + k * SEISMIC_ROUTES_STRIDE_1];
        const projection::u64 first = routed::expert_row(expert, SEISMIC_DIM_F);
        const Pro pro{normalized + m * SEISMIC_NORMALIZED_STRIDE_0 * 2, SEISMIC_NORMALIZED_STRIDE_0, projection::AllRows{}};
        const Epi epi{SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER)
                          + (m * SEISMIC_RESULT_0_STRIDE_0 + k * SEISMIC_RESULT_0_STRIDE_1) * 2,
                      0};
        projection::gemv_segment<Shape>(shared, pro, 1u, SEISMIC_DIM_H / 64, group, SEISMIC_DIM_F,
                                KERNEL_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_GATE) + first * ELEMENT_CAT(KERNEL_W0, _ROW_STRIDE_BYTES)),
                                KERNEL_W1_AT(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_UP) + first * ELEMENT_CAT(KERNEL_W1, _ROW_STRIDE_BYTES)),
                                epi);
        return;
    }

    if (group >= projection::gemv_groups<Shape>(SEISMIC_DIM_S))
        return;
    const Pro pro{normalized, SEISMIC_NORMALIZED_STRIDE_0, projection::AllRows{}};
    const Epi epi{SEISMIC_PTR(SEISMIC_RESULT_1_BUFFER), SEISMIC_RESULT_1_STRIDE_0};
    projection::gemv_segment<Shape>(shared, pro, (unsigned)SEISMIC_DIM_M, SEISMIC_DIM_H / 64, group, SEISMIC_DIM_S,
                            KERNEL_W2_AT(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_GATE)),
                            KERNEL_W3_AT(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_UP)), epi);
}
