// Decode expansion of up-only experts (M <= DECODE_ROWS); the CUDA form of
// `metal/routed_up.metal`, with the activated epilogue A(act(A(up))).
// Two forms of one entry (GATHERED, a tuning form both valid at every M,
// which the tuning admits between):
// - `routed_up` (GATHERED 0): block row y = choice (m, k), the K1 GEMV over
//   choice's up rows where its source names (`routed::decode_source`: a
//   bank record, or a host tier record read in place); both forms read an
//   expert's weights there, its scale by expert, and give a choice with no
//   record a zero product.
// - `routed_up_gathered` (GATHERED 1): a choice computes
//   only when it is its expert's first choice, the K1 GEMV over that
//   expert's up rows for the activation row of every choice of the expert
//   (`routed::expert_choices`), so each chosen expert streams once. A GEMV
//   row's result does not depend on the other rows, so every choice gets its
//   one-row bits.
#define KERNEL_W0 SEISMIC_EXPERT_UP
#include "lib/routed/routed.cuh"

// A block's one row, read in place unless the experts take it as NVFP4
// operands, which the block forms in its shared memory.
using Raw = projection::Plain<ELEMENT_OF(SEISMIC_ELEMENT_A), projection::AllRows>;
using Pro = projection::Operands<projection::OperandRecipe<packets::W0_OPERANDS_NVFP4>::NVFP4, Raw>;
using Source = projection::GemvSource<Pro>;
using Epi = projection::Activated<ELEMENT_OF(SEISMIC_ELEMENT_A)>;

// Where choice (m, k) of `expert` reads its up weights: a bank record, or
// its tier record in place. In the kernel's own scope, where its ports are.
#define UP_WEIGHTS(m, k, expert)                                                                           \
    routed::Place::of(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_UP), SEISMIC_PTR(SEISMIC_BUFFER_TIER_UP),          \
                      routed::decode_source(SEISMIC_PTR(SEISMIC_BUFFER_SOURCES), SEISMIC_DIM_XS,           \
                                            SEISMIC_SOURCES_STRIDE_1, SEISMIC_SOURCES_STRIDE_2, (m), (k),  \
                                            (expert)))

#ifdef SEISMIC_FORMING_ROUTED_UP
template <int TPW, int KSPLIT>
__global__ void routed_up(SEISMIC_KERNEL_PARAMS) {
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
    const projection::u64 expert = (projection::u64)routes[m * SEISMIC_ROUTES_STRIDE_0 + k * SEISMIC_ROUTES_STRIDE_1];
    const routed::Place weights = UP_WEIGHTS(m, k, (int)expert);
    projection::u8 *product =
        SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER) + (m * SEISMIC_RESULT_0_STRIDE_0 + k * SEISMIC_RESULT_0_STRIDE_1) * 2;
    // A choice with no record has a zero product.
    if (weights.stack == nullptr) {
        routed::zero_block_rows<Shape>(SEISMIC_DIM_F, [&](projection::u64 n) {
            element::put<ELEMENT_OF(SEISMIC_ELEMENT_A)>(product, n, 0.0f);
        });
        return;
    }
    const projection::u64 group = Shape::tile_group();
    const auto epi = projection::scaling<(SEISMIC_DIM_US != 0)>::wrap(
        Epi{product, 0, (int)SEISMIC_PARAM_ACTIVATION},
        projection::scale_factor(SEISMIC_PTR(SEISMIC_BUFFER_UP_SCALE), SEISMIC_DIM_US,
                                 SEISMIC_UP_SCALE_STRIDE_0, expert), 1.0f);
    projection::gemv_segment<Shape>(shared, pro, 1u, SEISMIC_DIM_H / 64, group, SEISMIC_DIM_F,
                                    KERNEL_W0_MATRIX(weights.stack, weights.record), projection::NoWeight{}, epi);
}
#endif

// The rows the gathered GEMV reads: the normalized rows, or where the experts
// take NVFP4 operands the rows `routed_up_stage` formed.
constexpr bool NVFP4 = projection::OperandRecipe<packets::W0_OPERANDS_NVFP4>::NVFP4;
#define GATHERED_ROWS (NVFP4 ? SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_OPERANDS) : SEISMIC_PTR(SEISMIC_BUFFER_NORMALIZED))
#define GATHERED_STRIDE (NVFP4 ? SEISMIC_DIM_H : SEISMIC_NORMALIZED_STRIDE_0)

#ifdef SEISMIC_FORMING_ROUTED_UP_STAGE
// Each normalized row as NVFP4 operands, once for every expert.
__global__ void routed_up_stage(SEISMIC_KERNEL_PARAMS) {
    projection::form_row(projection::Nvfp4Rows<Raw>{Raw{SEISMIC_PTR(SEISMIC_BUFFER_NORMALIZED),
                                                        SEISMIC_NORMALIZED_STRIDE_0, projection::AllRows{}}},
                         blockIdx.x, SEISMIC_DIM_H, SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_OPERANDS));
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_UP_GATHERED
// The activated product into each GEMV row's choice: row r at element offset
// offsets[r].
struct ChoiceEpi {
    Epi activated;
    const projection::u64 *offsets;
    __device__ __forceinline__ void operator()(unsigned m, projection::u64 n, float projected, float) const {
        element::put<ELEMENT_OF(SEISMIC_ELEMENT_A)>(activated.out, offsets[m] + n, activated.value(projected));
    }
    __device__ __forceinline__ void pair(unsigned m, projection::u64 n, float first, float second, float,
                                         float) const {
        element::put2<ELEMENT_OF(SEISMIC_ELEMENT_A)>(activated.out, offsets[m] + n, activated.value(first),
                                                     activated.value(second));
    }
};

template <int TPW, int KSPLIT>
__global__ void routed_up_gathered(SEISMIC_KERNEL_PARAMS) {
    using Shape = projection::GemvShape<4, TPW, KSPLIT, 1>;
    __shared__ projection::GemvShared<Shape, Raw> shared;
    const int *routes = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES));
    const unsigned choice = blockIdx.y, slots = (unsigned)SEISMIC_DIM_K;
    if (!routed::first_choice(routes, SEISMIC_ROUTES_STRIDE_0, SEISMIC_ROUTES_STRIDE_1, slots, choice))
        return;
    __shared__ projection::u64 rows_at[SEISMIC_CONST_DECODE_ROWS], products_at[SEISMIC_CONST_DECODE_ROWS];
    __shared__ unsigned count;
    if (threadIdx.x == 0) {
        projection::u32 chosen[SEISMIC_CONST_DECODE_ROWS];
        count = routed::expert_choices(routes, SEISMIC_ROUTES_STRIDE_0, SEISMIC_ROUTES_STRIDE_1,
                                       (unsigned)SEISMIC_DIM_M, slots, choice, chosen);
        for (unsigned r = 0; r < count; ++r) {
            const projection::u64 m = chosen[r] / slots, k = chosen[r] % slots;
            rows_at[r] = m * GATHERED_STRIDE;
            products_at[r] = m * SEISMIC_RESULT_0_STRIDE_0 + k * SEISMIC_RESULT_0_STRIDE_1;
        }
    }
    __syncthreads();
    const projection::u64 expert =
        (projection::u64)routed::route(routes, SEISMIC_ROUTES_STRIDE_0, SEISMIC_ROUTES_STRIDE_1, slots, choice);
    const routed::Place weights = UP_WEIGHTS(choice / slots, choice % slots, (int)expert);
    // An expert with no record (every choice of an expert has its one
    // source) has a zero product for each of its choices.
    if (weights.stack == nullptr) {
        for (unsigned r = 0; r < count; ++r)
            routed::zero_block_rows<Shape>(SEISMIC_DIM_F, [&](projection::u64 n) {
                element::put<ELEMENT_OF(SEISMIC_ELEMENT_A)>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER), products_at[r] + n,
                                                            0.0f);
            });
        return;
    }
    const projection::u64 group = Shape::tile_group();
    const auto epi = projection::scaling<(SEISMIC_DIM_US != 0)>::wrap(
        ChoiceEpi{Epi{SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER), 0, (int)SEISMIC_PARAM_ACTIVATION}, products_at},
        projection::scale_factor(SEISMIC_PTR(SEISMIC_BUFFER_UP_SCALE), SEISMIC_DIM_US, SEISMIC_UP_SCALE_STRIDE_0,
                                 expert),
        1.0f);
    projection::gemv_segment<Shape>(shared, routed::GatheredRows{GATHERED_ROWS, rows_at}, count, SEISMIC_DIM_H / 64,
                                    group, SEISMIC_DIM_F,
                                    KERNEL_W0_MATRIX(weights.stack, weights.record), projection::NoWeight{}, epi);
}
#endif
