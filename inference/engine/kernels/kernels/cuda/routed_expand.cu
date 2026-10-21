// Decode expansion (M <= 8); the CUDA form of `metal/routed_expand.metal`.
// Block row y < M * K is choice (m, k); row y = M * K is the shared expert
// over all M rows.
// - `routed_expand` (one row): choice k is the K1 paired GEMV over expert
//   routes[0, k]'s gate/up rows.
// - `routed_expand_gathered` (several rows): a choice computes only when it
//   is its expert's first choice, the K1 paired GEMV over that expert's
//   gate/up rows for the activation row of every choice of the expert
//   (`routed::expert_choices`), so each chosen expert streams once. A GEMV
//   row's result does not depend on the other rows, so every choice gets its
//   one-row bits.
// A choice's gate/up accumulators carry its expert's second-level scales
// (`gate_scale`, `up_scale`, static GS, US) before SiLU . mul where those
// ports are present; the shared expert's none.
#define KERNEL_W0 SEISMIC_EXPERT_GATE
#define KERNEL_W1 SEISMIC_EXPERT_UP
#define KERNEL_W2 SEISMIC_SHARED_GATE
#define KERNEL_W3 SEISMIC_SHARED_UP
#include "lib/routed/routed.cuh"

SEISMIC_PROGRAMMATIC_DEPENDENCY

using Pro = projection::Plain<ELEMENT_OF(SEISMIC_ELEMENT_A), projection::AllRows>;
using Epi = projection::SiluMul<ELEMENT_OF(SEISMIC_ELEMENT_A)>;

// The rows the routed and the shared experts read: the normalized rows, or
// where their weights take NVFP4 operands the rows the stage launch formed.
constexpr bool ROUTED_NVFP4 =
    projection::OperandRecipe<packets::W0_OPERANDS_NVFP4, packets::W1_OPERANDS_NVFP4>::NVFP4;
constexpr bool SHARED_NVFP4 =
    projection::OperandRecipe<packets::W2_OPERANDS_NVFP4, packets::W3_OPERANDS_NVFP4>::NVFP4;
#define ROWS_OF(NVFP4) ((NVFP4) ? SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_OPERANDS) : SEISMIC_PTR(SEISMIC_BUFFER_NORMALIZED))
#define STRIDE_OF(NVFP4) ((NVFP4) ? SEISMIC_DIM_H : SEISMIC_NORMALIZED_STRIDE_0)

#ifdef SEISMIC_FORMING_ROUTED_EXPAND_STAGE
// Each normalized row as NVFP4 operands, once for every expert.
__global__ void routed_expand_stage(SEISMIC_KERNEL_PARAMS) {
    seismic_dependency_start();
    projection::form_row(projection::Nvfp4Rows<Pro>{Pro{SEISMIC_PTR(SEISMIC_BUFFER_NORMALIZED),
                                                        SEISMIC_NORMALIZED_STRIDE_0, projection::AllRows{}}},
                         blockIdx.x, SEISMIC_DIM_H, SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_OPERANDS));
}
#endif

// Expert `expert`'s gate/up scales around a choice's epilogue (a macro: the
// buffers are kernel parameters).
#define EXPERT_SCALED(epi, expert)                                                                          \
    projection::scaling<(SEISMIC_DIM_GS + SEISMIC_DIM_US != 0)>::wrap(                                      \
        (epi),                                                                                              \
        projection::scale_factor(SEISMIC_PTR(SEISMIC_BUFFER_GATE_SCALE), SEISMIC_DIM_GS,                    \
                                 SEISMIC_GATE_SCALE_STRIDE_0, (expert)),                                    \
        projection::scale_factor(SEISMIC_PTR(SEISMIC_BUFFER_UP_SCALE), SEISMIC_DIM_US,                      \
                                 SEISMIC_UP_SCALE_STRIDE_0, (expert)))

#ifdef SEISMIC_FORMING_ROUTED_EXPAND
template <int TPW, int KSPLIT>
__global__ void routed_expand(SEISMIC_KERNEL_PARAMS) {
    using Shape = projection::GemvShape<4, TPW, KSPLIT, 1>;
    __shared__ projection::GemvShared<Shape, Pro> shared;
    const int *routes = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES));
    const projection::u64 group = Shape::tile_group();
    const projection::u64 choices = SEISMIC_DIM_M * SEISMIC_DIM_K;

    // A choice's expert comes from the launch before, so its blocks wait
    // first; the shared expert's issue their first weight loads before.
    if (blockIdx.y < choices) {
        seismic_dependency_start();
        if (group >= projection::gemv_groups<Shape>(SEISMIC_DIM_F))
            return;
        const projection::u64 m = blockIdx.y / SEISMIC_DIM_K, k = blockIdx.y % SEISMIC_DIM_K;
        const projection::u64 expert = (projection::u64)routes[m * SEISMIC_ROUTES_STRIDE_0 + k * SEISMIC_ROUTES_STRIDE_1];
        const Pro pro{ROWS_OF(ROUTED_NVFP4) + m * STRIDE_OF(ROUTED_NVFP4) * 2, STRIDE_OF(ROUTED_NVFP4),
                      projection::AllRows{}};
        const Epi epi{SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER)
                          + (m * SEISMIC_RESULT_0_STRIDE_0 + k * SEISMIC_RESULT_0_STRIDE_1) * 2,
                      0};
        projection::gemv_segment<Shape>(shared, pro, 1u, SEISMIC_DIM_H / 64, group, SEISMIC_DIM_F,
                                KERNEL_W0_MATRIX(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_GATE), expert),
                                KERNEL_W1_MATRIX(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_UP), expert),
                                EXPERT_SCALED(epi, expert));
        return;
    }

    const Pro pro{ROWS_OF(SHARED_NVFP4), STRIDE_OF(SHARED_NVFP4), projection::AllRows{}};
    const Epi epi{SEISMIC_PTR(SEISMIC_RESULT_1_BUFFER), SEISMIC_RESULT_1_STRIDE_0};
    projection::gemv_segment_ready<Shape>(shared, projection::after_dependency(pro), (unsigned)SEISMIC_DIM_M, SEISMIC_DIM_H / 64, group, SEISMIC_DIM_S,
                            KERNEL_W2_AT(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_GATE)),
                            KERNEL_W3_AT(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_UP)), epi);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_EXPAND_GATHERED
// SiluMul into each GEMV row's choice: row r at element offset offsets[r].
struct ChoiceEpi {
    projection::u8 *out;
    const projection::u64 *offsets;
    __device__ __forceinline__ void operator()(unsigned m, projection::u64 n, float gate, float up) const {
        element::put<ELEMENT_OF(SEISMIC_ELEMENT_A)>(out, offsets[m] + n, Epi::value(gate, up));
    }
};

template <int TPW, int KSPLIT>
__global__ void routed_expand_gathered(SEISMIC_KERNEL_PARAMS) {
    using Shape = projection::GemvShape<4, TPW, KSPLIT, 1>;
    __shared__ projection::GemvShared<Shape, Pro> shared;
    const int *routes = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES));
    const projection::u64 group = Shape::tile_group();
    const projection::u64 choices = SEISMIC_DIM_M * SEISMIC_DIM_K;

    // A choice's expert comes from the launch before, so its blocks wait
    // first; the shared expert's issue their first weight loads before.
    if (blockIdx.y < choices) {
        seismic_dependency_start();
        const unsigned choice = blockIdx.y, slots = (unsigned)SEISMIC_DIM_K;
        if (!routed::first_choice(routes, SEISMIC_ROUTES_STRIDE_0, SEISMIC_ROUTES_STRIDE_1, slots, choice))
            return;
        __shared__ projection::u64 rows_at[8], products_at[8];
        __shared__ unsigned count;
        if (threadIdx.x == 0) {
            projection::u32 chosen[8];
            count = routed::expert_choices(routes, SEISMIC_ROUTES_STRIDE_0, SEISMIC_ROUTES_STRIDE_1,
                                           (unsigned)SEISMIC_DIM_M, slots, choice, chosen);
            for (unsigned r = 0; r < count; ++r) {
                const projection::u64 m = chosen[r] / slots, k = chosen[r] % slots;
                rows_at[r] = m * STRIDE_OF(ROUTED_NVFP4);
                products_at[r] = m * SEISMIC_RESULT_0_STRIDE_0 + k * SEISMIC_RESULT_0_STRIDE_1;
            }
        }
        __syncthreads();
        if (group >= projection::gemv_groups<Shape>(SEISMIC_DIM_F))
            return;
        const projection::u64 expert =
            (projection::u64)routed::route(routes, SEISMIC_ROUTES_STRIDE_0, SEISMIC_ROUTES_STRIDE_1, slots, choice);
        projection::gemv_segment<Shape>(shared, routed::GatheredRows{ROWS_OF(ROUTED_NVFP4), rows_at}, count, SEISMIC_DIM_H / 64,
                                        group, SEISMIC_DIM_F,
                                        KERNEL_W0_MATRIX(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_GATE), expert),
                                        KERNEL_W1_MATRIX(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_UP), expert),
                                        EXPERT_SCALED((ChoiceEpi{SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER), products_at}),
                                                      expert));
        return;
    }

    const Pro pro{ROWS_OF(SHARED_NVFP4), STRIDE_OF(SHARED_NVFP4), projection::AllRows{}};
    const Epi epi{SEISMIC_PTR(SEISMIC_RESULT_1_BUFFER), SEISMIC_RESULT_1_STRIDE_0};
    projection::gemv_segment_ready<Shape>(shared, projection::after_dependency(pro), (unsigned)SEISMIC_DIM_M, SEISMIC_DIM_H / 64, group, SEISMIC_DIM_S,
                            KERNEL_W2_AT(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_GATE)),
                            KERNEL_W3_AT(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_UP)), epi);
}
#endif
