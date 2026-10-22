// Decode output of the general routed feed-forward (M <= 8); the CUDA form
// of `metal/routed_down.metal`. The choices' down projections run in parallel
// (block (x, m K + k) owns the output channels of tile group x for choice k
// of row m) and publish each A-rounded projection in `projected`; the combine
// then weights them in slot order and publishes base + selected, the sum the
// portable body forms.
#define KERNEL_W0 SEISMIC_EXPERT_DOWN
#include "lib/routed/routed.cuh"

// A block's one product row, read in place unless the experts take it as
// NVFP4 operands, which the block forms in its shared memory.
using Raw = projection::Plain<ELEMENT_OF(SEISMIC_ELEMENT_A), projection::AllRows>;
using Pro = projection::Operands<projection::OperandRecipe<packets::W0_OPERANDS_NVFP4>::NVFP4, Raw>;
using Source = projection::GemvSource<Pro>;
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
    __shared__ projection::GemvShared<Shape, Source::type> shared;
    extern __shared__ uint4 dynamic_shared[];
    const projection::u64 m = blockIdx.y / SEISMIC_DIM_K, k = blockIdx.y % SEISMIC_DIM_K;
    // Block-collective: before any warp leaves.
    const Source::type pro = Source::make(
        Pro{Raw{SEISMIC_PTR(SEISMIC_BUFFER_PRODUCT) + (m * SEISMIC_PRODUCT_STRIDE_0 + k * SEISMIC_PRODUCT_STRIDE_1) * 2,
                0, projection::AllRows{}}},
        reinterpret_cast<projection::u8 *>(dynamic_shared), 1u, SEISMIC_DIM_F, nullptr);
    const projection::u64 group = Shape::tile_group();
    if (group >= projection::gemv_groups<Shape>(SEISMIC_DIM_H))
        return;
    const int *routes = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES));
    const projection::u64 expert = (projection::u64)routes[m * SEISMIC_ROUTES_STRIDE_0 + k * SEISMIC_ROUTES_STRIDE_1];
    const auto down = KERNEL_W0_MATRIX(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_DOWN), expert);
    float *projected = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PROJECTED)) +
                       (m * SEISMIC_DIM_K + k) * SEISMIC_DIM_H;
    projection::gemv_segment<Shape>(shared, pro, 1u, SEISMIC_DIM_F / 64, group, SEISMIC_DIM_H, down,
                                    projection::NoWeight{}, ProjectedEpi{projected});
}
#endif

// The product rows the gathered GEMV reads: in place ([M, K, F]), or where
// the experts take NVFP4 operands the rows `routed_down_stage` formed
// ([M K, F]).
constexpr bool NVFP4 = projection::OperandRecipe<packets::W0_OPERANDS_NVFP4>::NVFP4;

#ifdef SEISMIC_FORMING_ROUTED_DOWN_STAGE
// Choice (m, k)'s product row as NVFP4 operands, block m K + k.
__global__ void routed_down_stage(SEISMIC_KERNEL_PARAMS) {
    const projection::u64 m = blockIdx.x / SEISMIC_DIM_K, k = blockIdx.x % SEISMIC_DIM_K;
    const Raw row{SEISMIC_PTR(SEISMIC_BUFFER_PRODUCT) + (m * SEISMIC_PRODUCT_STRIDE_0 + k * SEISMIC_PRODUCT_STRIDE_1) * 2,
                  0, projection::AllRows{}};
    projection::form_row(projection::Nvfp4Rows<Raw>{row}, 0u, SEISMIC_DIM_F,
                         SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_OPERANDS) + blockIdx.x * SEISMIC_DIM_F * 2);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_DOWN_GATHERED
// Each GEMV row's projection, A-rounded, into its choice's [M, K, H] slot.
struct ChoiceProjectedEpi {
    float *projected;
    const projection::u64 *slots;
    __device__ __forceinline__ void operator()(unsigned r, projection::u64 n, float value, float) const {
        projected[slots[r] * SEISMIC_DIM_H + n] = element::Act::round(value);
    }
};

// A choice computes only when it is its expert's first choice: the K1 GEMV
// over that expert's down rows for the product row of every choice of the
// expert (`routed::expert_choices`), so each chosen expert streams once and
// every choice gets its one-row bits.
template <int TPW, int KSPLIT>
__global__ void routed_down_gathered(SEISMIC_KERNEL_PARAMS) {
    using Shape = projection::GemvShape<4, TPW, KSPLIT, 1>;
    __shared__ projection::GemvShared<Shape, Raw> shared;
    const int *routes = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES));
    const unsigned choice = blockIdx.y, slots = (unsigned)SEISMIC_DIM_K;
    if (!routed::first_choice(routes, SEISMIC_ROUTES_STRIDE_0, SEISMIC_ROUTES_STRIDE_1, slots, choice))
        return;
    __shared__ projection::u64 rows_at[8], slots_at[8];
    __shared__ unsigned count;
    if (threadIdx.x == 0) {
        projection::u32 chosen[8];
        count = routed::expert_choices(routes, SEISMIC_ROUTES_STRIDE_0, SEISMIC_ROUTES_STRIDE_1,
                                       (unsigned)SEISMIC_DIM_M, slots, choice, chosen);
        for (unsigned r = 0; r < count; ++r) {
            const projection::u64 m = chosen[r] / slots, k = chosen[r] % slots;
            rows_at[r] = NVFP4 ? chosen[r] * SEISMIC_DIM_F : m * SEISMIC_PRODUCT_STRIDE_0 + k * SEISMIC_PRODUCT_STRIDE_1;
            slots_at[r] = chosen[r];
        }
    }
    __syncthreads();
    const projection::u64 group = Shape::tile_group();
    if (group >= projection::gemv_groups<Shape>(SEISMIC_DIM_H))
        return;
    const projection::u64 expert =
        (projection::u64)routed::route(routes, SEISMIC_ROUTES_STRIDE_0, SEISMIC_ROUTES_STRIDE_1, slots, choice);
    const projection::u8 *rows =
        NVFP4 ? SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_OPERANDS) : SEISMIC_PTR(SEISMIC_BUFFER_PRODUCT);
    projection::gemv_segment<Shape>(shared, routed::GatheredRows{rows, rows_at}, count, SEISMIC_DIM_F / 64, group,
                                    SEISMIC_DIM_H, KERNEL_W0_MATRIX(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_DOWN), expert),
                                    projection::NoWeight{},
                                    ChoiceProjectedEpi{reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PROJECTED)),
                                                       slots_at});
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
