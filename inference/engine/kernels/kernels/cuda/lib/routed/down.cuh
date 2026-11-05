// The decode down projections of the general routed feed-forward (M <=
// DECODE_ROWS), shared by `routed_down` and `routed_down_project`; the CUDA
// form of `metal/lib/routed/down.h`. The choices' down projections run in
// parallel (block (x, m K + k) owns the output channels of tile group x for
// choice k of row m) and publish each projection in A at `ROUTED_PROJECTED`
// (`[M, K, H]`, row-major, of `element::Act::storage`); the sum is
// `lib/routed/down_sum.cuh`'s. Each choice's down weights are read where its
// source names (`routed::decode_source`): a bank record, or a host tier
// record read in place; a choice with no record (the CPU's, or a failed
// one) projects zero.
//
// The includer defines ROUTED_SOURCE_ROWS (the sources' rows: the entry's
// XS, or SOURCE_ROWS where its layers stream) and ROUTED_PROJECTED (the
// projections' storage, in the kernel's scope).
#define KERNEL_W0 SEISMIC_EXPERT_DOWN
#include "routed.cuh"

// A block's one product row, read in place unless the experts take it as
// NVFP4 operands, which the block forms in its shared memory.
using Raw = projection::Plain<ELEMENT_OF(SEISMIC_ELEMENT_A), projection::AllRows>;
using Pro = projection::Operands<projection::OperandRecipe<packets::W0_OPERANDS_NVFP4>::NVFP4, Raw>;
using Source = projection::GemvSource<Pro>;

// Where choice (m, k) of `expert` reads its down weights: a bank record, or
// its tier record in place. In the kernel's own scope, where its ports are.
#define DOWN_WEIGHTS(m, k, expert)                                                                         \
    routed::Place::of(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_DOWN), SEISMIC_PTR(SEISMIC_BUFFER_TIER_DOWN),      \
                      routed::decode_source(SEISMIC_PTR(SEISMIC_BUFFER_SOURCES), ROUTED_SOURCE_ROWS,           \
                                            SEISMIC_SOURCES_STRIDE_1, SEISMIC_SOURCES_STRIDE_2, (m), (k),  \
                                            (expert)))


// One choice's projection, A-rounded, into its [M, K, H] slot.
struct ProjectedEpi {
    element::Act::storage *projected;
    __device__ __forceinline__ void operator()(unsigned, projection::u64 n, float value, float) const {
        projected[n] = element::Act::store(value);
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
    const int *routes = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES));
    const routed::Place down = DOWN_WEIGHTS(m, k, routes[m * SEISMIC_ROUTES_STRIDE_0 + k * SEISMIC_ROUTES_STRIDE_1]);
    element::Act::storage *projected = ROUTED_PROJECTED + (m * SEISMIC_DIM_K + k) * SEISMIC_DIM_H;
    if (down.stack == nullptr) {
        // A choice reading no record: a zero
        // projection.
        routed::zero_block_rows<Shape>(SEISMIC_DIM_H, [&](projection::u64 n) { projected[n] = element::Act::store(0.0f); });
        return;
    }
    const projection::u64 group = Shape::tile_group();
    if (group >= projection::gemv_groups<Shape>(SEISMIC_DIM_H))
        return;
    projection::gemv_segment<Shape>(shared, pro, 1u, SEISMIC_DIM_F / 64, group, SEISMIC_DIM_H,
                                    KERNEL_W0_MATRIX(down.stack, down.record), projection::NoWeight{},
                                    ProjectedEpi{projected});
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
    element::Act::storage *projected;
    const projection::u64 *slots;
    __device__ __forceinline__ void operator()(unsigned r, projection::u64 n, float value, float) const {
        projected[slots[r] * SEISMIC_DIM_H + n] = element::Act::store(value);
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
    __shared__ projection::u64 rows_at[SEISMIC_CONST_DECODE_ROWS], slots_at[SEISMIC_CONST_DECODE_ROWS];
    __shared__ unsigned count;
    if (threadIdx.x == 0) {
        projection::u32 chosen[SEISMIC_CONST_DECODE_ROWS];
        count = routed::expert_choices(routes, SEISMIC_ROUTES_STRIDE_0, SEISMIC_ROUTES_STRIDE_1,
                                       (unsigned)SEISMIC_DIM_M, slots, choice, chosen);
        for (unsigned r = 0; r < count; ++r) {
            const projection::u64 m = chosen[r] / slots, k = chosen[r] % slots;
            rows_at[r] = NVFP4 ? chosen[r] * SEISMIC_DIM_F : m * SEISMIC_PRODUCT_STRIDE_0 + k * SEISMIC_PRODUCT_STRIDE_1;
            slots_at[r] = chosen[r];
        }
    }
    __syncthreads();
    const routed::Place down = DOWN_WEIGHTS(choice / slots, choice % slots,
                                            routed::route(routes, SEISMIC_ROUTES_STRIDE_0, SEISMIC_ROUTES_STRIDE_1,
                                                          slots, choice));
    element::Act::storage *projected = ROUTED_PROJECTED;
    if (down.stack == nullptr) {
        // An expert reading no record: each of its
        // choices' projections is zero.
        for (unsigned r = 0; r < count; ++r)
            routed::zero_block_rows<Shape>(SEISMIC_DIM_H,
                                           [&](projection::u64 n) { projected[slots_at[r] * SEISMIC_DIM_H + n] = element::Act::store(0.0f); });
        return;
    }
    const projection::u64 group = Shape::tile_group();
    if (group >= projection::gemv_groups<Shape>(SEISMIC_DIM_H))
        return;
    const projection::u8 *rows =
        NVFP4 ? SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_OPERANDS) : SEISMIC_PTR(SEISMIC_BUFFER_PRODUCT);
    projection::gemv_segment<Shape>(shared, routed::GatheredRows{rows, rows_at}, count, SEISMIC_DIM_F / 64, group,
                                    SEISMIC_DIM_H,
                                    KERNEL_W0_MATRIX(down.stack, down.record), projection::NoWeight{},
                                    ChoiceProjectedEpi{projected, slots_at});
}
#endif
