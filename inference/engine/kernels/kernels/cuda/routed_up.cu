// Decode expansion of up-only experts (M <= VERIFY_ROWS); the CUDA form of
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
//   (`routed::gather`), so each chosen expert streams once. A GEMV
//   row's result does not depend on the other rows, so every choice gets its
//   one-row bits.
#define KERNEL_W0 SEISMIC_EXPERT_UP
#include "lib/routed/routed.cuh"

// A block's one row, as the experts take it (NVFP4 operands or A), which the
// block forms in its shared memory (`shared_row`).
using Raw = projection::Plain<ELEMENT_OF(SEISMIC_ELEMENT_A), projection::AllRows>;
using Pro = projection::Operands<projection::OperandRecipe<packets::W0_OPERANDS_NVFP4>::NVFP4, Raw>;
using Epi = projection::Activated<ELEMENT_OF(SEISMIC_ELEMENT_A)>;

// Where choice (m, k) of `expert` reads its up weights: a bank record, or
// its tier record in place. In the kernel's own scope, where its ports are.
#define UP_WEIGHTS(m, k, expert)                                                                           \
    (slab::Experts{SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_UP), SEISMIC_PTR(SEISMIC_BUFFER_TIER_UP)}                  \
         .matrix(routed::decode_source(SEISMIC_PTR(SEISMIC_BUFFER_SOURCES), SEISMIC_DIM_XS, SEISMIC_SOURCES_STRIDE_1, \
                                       SEISMIC_SOURCES_STRIDE_2, (m), (k), (expert))))

#ifdef SEISMIC_FORMING_ROUTED_UP_CHOICE
template <int TPW, int KSPLIT>
__launch_bounds__(routed::DECODE_THREADS) __global__ void routed_up_choice(SEISMIC_KERNEL_PARAMS) {
    using Shape = routed::DecodeShape<TPW, KSPLIT>;
    __shared__ projection::GemvShared<Shape, projection::SharedRow> shared;
    const projection::u8 *normalized = SEISMIC_PTR(SEISMIC_BUFFER_NORMALIZED);
    const int *routes = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES));
    const projection::u64 m = blockIdx.y / SEISMIC_DIM_K, k = blockIdx.y % SEISMIC_DIM_K;
    // Block-collective: before any warp leaves.
    const projection::SharedRow pro = projection::shared_row(
        Pro{Raw{normalized + m * SEISMIC_NORMALIZED_STRIDE_0 * 2, SEISMIC_NORMALIZED_STRIDE_0, projection::AllRows{}}},
        SEISMIC_DIM_H);
    const projection::u64 expert = (projection::u64)routes[m * SEISMIC_ROUTES_STRIDE_0 + k * SEISMIC_ROUTES_STRIDE_1];
    const slab::Matrix weights = UP_WEIGHTS(m, k, (int)expert);
    const element::Out<ELEMENT_OF(SEISMIC_ELEMENT_A)> product =
        SEISMIC_RESULT_0_OUT.offset(m * SEISMIC_RESULT_0_STRIDE_0 + k * SEISMIC_RESULT_0_STRIDE_1);
    // A choice with no record has a zero product.
    if (weights.stack == nullptr) {
        routed::zero_block_rows<Shape>(SEISMIC_DIM_F, [&](projection::u64 n) {
            product.put(n, 0.0f);
        });
        return;
    }
    const projection::u64 group = Shape::tile_group();
    const auto epi = projection::scaling<(SEISMIC_DIM_US != 0)>::wrap(
        Epi{product, 0, (int)SEISMIC_PARAM_ACTIVATION},
        projection::scale_factor(SEISMIC_PTR(SEISMIC_BUFFER_UP_SCALE), SEISMIC_DIM_US,
                                 SEISMIC_UP_SCALE_STRIDE_0, expert), 1.0f);
    projection::gemv_segment<Shape>(shared, pro, 1u, SEISMIC_DIM_H / 64, group, SEISMIC_DIM_F,
                                    ROUTED_MATRIX(KERNEL_W0, weights), projection::NoWeight{}, epi);
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
template <int TPW, int KSPLIT>
__global__ void routed_up_gathered(SEISMIC_KERNEL_PARAMS) {
    using Shape = routed::DecodeShape<TPW, KSPLIT>;
    __shared__ projection::GemvShared<Shape, Raw> shared;
    __shared__ routed::Gathered table;
    const int *routes = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES));
    const unsigned choice = blockIdx.y, slots = (unsigned)SEISMIC_DIM_K;
    if (!routed::gather(table, routes, SEISMIC_ROUTES_STRIDE_0, SEISMIC_ROUTES_STRIDE_1,
                        (unsigned)SEISMIC_DIM_M, slots, choice, GATHERED_STRIDE, SEISMIC_RESULT_0_STRIDE_0,
                        SEISMIC_RESULT_0_STRIDE_1))
        return;
    const projection::u64 expert = (projection::u64)table.expert;
    const slab::Matrix weights = UP_WEIGHTS(choice / slots, choice % slots, (int)expert);
    // An expert with no record (every choice of an expert has its one
    // source) has a zero product for each of its choices.
    if (weights.stack == nullptr) {
        for (unsigned r = 0; r < table.count; ++r)
            routed::zero_block_rows<Shape>(SEISMIC_DIM_F, [&](projection::u64 n) {
                SEISMIC_RESULT_0_OUT.put(table.products[r] + n, 0.0f);
            });
        return;
    }
    const projection::u64 group = Shape::tile_group();
    const auto epi = projection::scaling<(SEISMIC_DIM_US != 0)>::wrap(
        routed::ChoiceRows<Epi>{Epi{SEISMIC_RESULT_0_OUT, 0, (int)SEISMIC_PARAM_ACTIVATION},
                                table.products},
        projection::scale_factor(SEISMIC_PTR(SEISMIC_BUFFER_UP_SCALE), SEISMIC_DIM_US, SEISMIC_UP_SCALE_STRIDE_0,
                                 expert),
        1.0f);
    projection::gemv_segment<Shape>(shared, routed::GatheredRows{GATHERED_ROWS, table.rows}, table.count,
                                    SEISMIC_DIM_H / 64, group, SEISMIC_DIM_F,
                                    ROUTED_MATRIX(KERNEL_W0, weights), projection::NoWeight{}, epi);
}
#endif
