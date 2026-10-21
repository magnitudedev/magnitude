// Decode expansion of gated experts (M <= 8); the CUDA form of
// `metal/routed_gate_up.metal`. Block row y = choice (m, k): the K1 paired
// GEMV over expert routes[m, k]'s gate/up rows for one activation row, with
// the activation-generic GLU epilogue on the accumulators scaled by the
// expert's gate/up second-level scales where those ports are present (static
// GS, US).
#define KERNEL_W0 SEISMIC_EXPERT_GATE
#define KERNEL_W1 SEISMIC_EXPERT_UP
#include "lib/routed/routed.cuh"

// A block's one row, read in place unless the experts take it as NVFP4
// operands, which the block forms in its shared memory.
using Raw = projection::Plain<ELEMENT_OF(SEISMIC_ELEMENT_A), projection::AllRows>;
using Pro = projection::Operands<
    projection::OperandRecipe<packets::W0_OPERANDS_NVFP4, packets::W1_OPERANDS_NVFP4>::NVFP4, Raw>;
using Source = projection::GemvSource<Pro>;
using Epi = projection::Glu<ELEMENT_OF(SEISMIC_ELEMENT_A)>;

#ifdef SEISMIC_FORMING_ROUTED_GATE_UP
template <int TPW, int KSPLIT>
__global__ void routed_gate_up(SEISMIC_KERNEL_PARAMS) {
    using Shape = projection::GemvShape<4, TPW, KSPLIT, 1>;
    __shared__ projection::GemvShared<Shape, Source::type> shared;
    extern __shared__ uint4 dynamic_shared[];
    const projection::u8 *normalized = SEISMIC_PTR(SEISMIC_BUFFER_NORMALIZED);
    const int *routes = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES));
    const projection::u64 m = blockIdx.y / SEISMIC_DIM_K, k = blockIdx.y % SEISMIC_DIM_K;
    // Block-collective: before any warp leaves.
    const Source::type pro = Source::make(
        Pro{Raw{normalized + m * SEISMIC_NORMALIZED_STRIDE_0 * 2, SEISMIC_NORMALIZED_STRIDE_0, projection::AllRows{}}},
        reinterpret_cast<projection::u8 *>(dynamic_shared), 1u, SEISMIC_DIM_H, nullptr);
    const projection::u64 group = Shape::tile_group();
    if (group >= projection::gemv_groups<Shape>(SEISMIC_DIM_F))
        return;
    const projection::u64 expert = (projection::u64)routes[m * SEISMIC_ROUTES_STRIDE_0 + k * SEISMIC_ROUTES_STRIDE_1];
    const auto epi = projection::scaling<(SEISMIC_DIM_GS + SEISMIC_DIM_US != 0)>::wrap(
        Epi{SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER) + (m * SEISMIC_RESULT_0_STRIDE_0 + k * SEISMIC_RESULT_0_STRIDE_1) * 2,
            0, (int)SEISMIC_PARAM_ACTIVATION},
        projection::scale_factor(SEISMIC_PTR(SEISMIC_BUFFER_GATE_SCALE), SEISMIC_DIM_GS,
                                 SEISMIC_GATE_SCALE_STRIDE_0, expert),
        projection::scale_factor(SEISMIC_PTR(SEISMIC_BUFFER_UP_SCALE), SEISMIC_DIM_US,
                                 SEISMIC_UP_SCALE_STRIDE_0, expert));
    projection::gemv_segment<Shape>(shared, pro, 1u, SEISMIC_DIM_H / 64, group, SEISMIC_DIM_F,
                                    KERNEL_W0_MATRIX(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_GATE), expert),
                                    KERNEL_W1_MATRIX(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_UP), expert), epi);
}
#endif
