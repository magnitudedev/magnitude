// Decode expansion of up-only experts (M <= 8); the CUDA form of
// `metal/routed_up.metal`. Block row y = choice (m, k): the K1 GEMV over
// expert routes[m, k]'s up rows for one activation row, with the activated
// epilogue A(act(A(up))).
#define KERNEL_W0 SEISMIC_EXPERT_UP
#include "lib/routed/routed.cuh"

using Pro = projection::Plain<ELEMENT_OF(SEISMIC_ELEMENT_A), projection::AllRows>;
using Epi = projection::Activated<ELEMENT_OF(SEISMIC_ELEMENT_A)>;

#ifdef SEISMIC_FORMING_ROUTED_UP
template <int TPW, int KSPLIT>
__global__ void routed_up(SEISMIC_KERNEL_PARAMS) {
    using Shape = projection::GemvShape<4, TPW, KSPLIT, 1>;
    __shared__ projection::GemvShared<Shape, Pro> shared;
    const projection::u8 *normalized = SEISMIC_PTR(SEISMIC_BUFFER_NORMALIZED);
    const int *routes = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES));
    const projection::u64 group = Shape::tile_group();
    if (group >= projection::gemv_groups<Shape>(SEISMIC_DIM_F))
        return;
    const projection::u64 m = blockIdx.y / SEISMIC_DIM_K, k = blockIdx.y % SEISMIC_DIM_K;
    const projection::u64 expert = (projection::u64)routes[m * SEISMIC_ROUTES_STRIDE_0 + k * SEISMIC_ROUTES_STRIDE_1];
    const Pro pro{normalized + m * SEISMIC_NORMALIZED_STRIDE_0 * 2, SEISMIC_NORMALIZED_STRIDE_0, projection::AllRows{}};
    const auto epi = projection::scaling<true>::wrap(
        Epi{SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER) + (m * SEISMIC_RESULT_0_STRIDE_0 + k * SEISMIC_RESULT_0_STRIDE_1) * 2,
            0, (int)SEISMIC_PARAM_ACTIVATION},
        projection::scale_factor(SEISMIC_PTR(SEISMIC_BUFFER_UP_SCALE), SEISMIC_DIM_E,
                                 SEISMIC_UP_SCALE_STRIDE_0, expert), 1.0f);
    projection::gemv_segment<Shape>(shared, pro, 1u, SEISMIC_DIM_H / 64, group, SEISMIC_DIM_F,
                                    KERNEL_W0_MATRIX(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_UP), expert),
                                    projection::NoWeight{}, epi);
}
#endif
