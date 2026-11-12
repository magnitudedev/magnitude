// Decode expansion of gated experts (M <= DECODE_ROWS); the CUDA form of
// `metal/routed_gate_up.metal`, with the activation-generic GLU epilogue on
// the accumulators scaled by the expert's gate/up second-level scales where
// those ports are present (static GS, US). Each choice's weights are read
// where its source names (`routed::decode_source`): a bank record, or a
// host tier record read in place over the link.
// Two forms of one entry (GATHERED, a tuning form both valid at every M,
// which the tuning admits between):
// - `routed_gate_up` (GATHERED 0): block row y = choice (m, k), the K1 paired
//   GEMV over the choice's gate/up rows for its activation row.
// - `routed_gate_up_gathered` (GATHERED 1): a choice computes only when it
//   is its expert's first choice, the K1 paired GEMV over that expert's
//   gate/up rows for the activation row of every choice of the expert
//   (`routed::expert_choices`), so each chosen expert's records are read
//   once a step: a host record read once per row that named it would cost
//   its link bytes again for each (measured on an RTX 3090: 4.2x / 8.0x at 4
//   / 8 rows naming one host expert). A GEMV row's result does not depend on
//   the other rows, so every choice gets its one-row bits.
#define KERNEL_W0 SEISMIC_EXPERT_GATE
#define KERNEL_W1 SEISMIC_EXPERT_UP
// The expansion's limit as its F32 bits LB (0: none), which the Glu epilogue
// clamps at.
#if SEISMIC_STATIC_LB != 0
#define PROJECTION_GLU_LIMIT SEISMIC_STATIC_LB
#endif
#include "lib/routed/routed.cuh"

// A block's one row, read in place unless the experts take it as NVFP4
// operands, which the block forms in its shared memory.
using Raw = projection::Plain<ELEMENT_OF(SEISMIC_ELEMENT_A), projection::AllRows>;
using Pro = projection::Operands<
    projection::OperandRecipe<packets::W0_OPERANDS_NVFP4, packets::W1_OPERANDS_NVFP4>::NVFP4, Raw>;
using Source = projection::GemvSource<Pro>;
using Epi = projection::Glu<ELEMENT_OF(SEISMIC_ELEMENT_A)>;

// Where choice (m, k) of `expert` reads its gate and up weights: a bank
// record, or its tier records in place. In the kernel's own scope, where
// its ports are.
struct Weights {
    routed::Place gate, up;
    __device__ __forceinline__ bool none() const { return gate.stack == nullptr; }
};
#define CHOICE_WEIGHTS(m, k, expert)                                                                       \
    [&] {                                                                                                  \
        const int source = routed::decode_source(SEISMIC_PTR(SEISMIC_BUFFER_SOURCES), SEISMIC_DIM_XS,      \
                                                 SEISMIC_SOURCES_STRIDE_1, SEISMIC_SOURCES_STRIDE_2, (m),  \
                                                 (k), (expert));                                           \
        return Weights{routed::Place::of(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_GATE),                          \
                                         SEISMIC_PTR(SEISMIC_BUFFER_TIER_GATE), source),                   \
                       routed::Place::of(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_UP),                            \
                                         SEISMIC_PTR(SEISMIC_BUFFER_TIER_UP), source)};                    \
    }()

#ifdef SEISMIC_FORMING_ROUTED_GATE_UP
template <int TPW, int KSPLIT>
__global__ void routed_gate_up(SEISMIC_KERNEL_PARAMS) {
    using Shape = routed::DecodeShape<TPW, KSPLIT>;
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
    const Weights weights = CHOICE_WEIGHTS(m, k, (int)expert);
    projection::u8 *product =
        SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER) + (m * SEISMIC_RESULT_0_STRIDE_0 + k * SEISMIC_RESULT_0_STRIDE_1) * 2;
    // A choice with no record has a zero product.
    if (weights.none()) {
        routed::zero_block_rows<Shape>(SEISMIC_DIM_F, [&](projection::u64 n) {
            element::put<ELEMENT_OF(SEISMIC_ELEMENT_A)>(product, n, 0.0f);
        });
        return;
    }
    const projection::u64 group = Shape::tile_group();
    const auto epi = projection::scaling<(SEISMIC_DIM_GS + SEISMIC_DIM_US != 0)>::wrap(
        Epi{product, 0, (int)SEISMIC_PARAM_ACTIVATION},
        projection::scale_factor(SEISMIC_PTR(SEISMIC_BUFFER_GATE_SCALE), SEISMIC_DIM_GS,
                                 SEISMIC_GATE_SCALE_STRIDE_0, expert),
        projection::scale_factor(SEISMIC_PTR(SEISMIC_BUFFER_UP_SCALE), SEISMIC_DIM_US,
                                 SEISMIC_UP_SCALE_STRIDE_0, expert));
    projection::gemv_segment<Shape>(shared, pro, 1u, SEISMIC_DIM_H / 64, group, SEISMIC_DIM_F,
                                    KERNEL_W0_MATRIX(weights.gate.stack, weights.gate.record),
                                    KERNEL_W1_MATRIX(weights.up.stack, weights.up.record), epi);
}
#endif

// The rows the gathered GEMV reads: the normalized rows, or where the experts
// take NVFP4 operands the rows `routed_gate_up_stage` formed.
constexpr bool NVFP4 = projection::OperandRecipe<packets::W0_OPERANDS_NVFP4, packets::W1_OPERANDS_NVFP4>::NVFP4;
#define GATHERED_ROWS (NVFP4 ? SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_OPERANDS) : SEISMIC_PTR(SEISMIC_BUFFER_NORMALIZED))
#define GATHERED_STRIDE (NVFP4 ? SEISMIC_DIM_H : SEISMIC_NORMALIZED_STRIDE_0)

#ifdef SEISMIC_FORMING_ROUTED_GATE_UP_STAGE
// Each normalized row as NVFP4 operands, once for every expert.
__global__ void routed_gate_up_stage(SEISMIC_KERNEL_PARAMS) {
    projection::form_row(projection::Nvfp4Rows<Raw>{Raw{SEISMIC_PTR(SEISMIC_BUFFER_NORMALIZED),
                                                        SEISMIC_NORMALIZED_STRIDE_0, projection::AllRows{}}},
                         blockIdx.x, SEISMIC_DIM_H, SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_OPERANDS));
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_GATE_UP_GATHERED
// The GLU product into each GEMV row's choice: row r at element offset
// offsets[r].
struct ChoiceEpi {
    Epi glu;
    const projection::u64 *offsets;
    __device__ __forceinline__ void operator()(unsigned m, projection::u64 n, float gate, float up) const {
        element::put<ELEMENT_OF(SEISMIC_ELEMENT_A)>(glu.out, offsets[m] + n, glu.value(gate, up));
    }
    __device__ __forceinline__ void pair(unsigned m, projection::u64 n, float gate0, float gate1, float up0,
                                         float up1) const {
        element::put2<ELEMENT_OF(SEISMIC_ELEMENT_A)>(glu.out, offsets[m] + n, glu.value(gate0, up0),
                                                     glu.value(gate1, up1));
    }
};

template <int TPW, int KSPLIT>
__global__ void routed_gate_up_gathered(SEISMIC_KERNEL_PARAMS) {
    using Shape = routed::DecodeShape<TPW, KSPLIT>;
    __shared__ projection::GemvShared<Shape, Raw> shared;
    const int *routes = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES));
    const unsigned at = blockIdx.y, slots = (unsigned)SEISMIC_DIM_K;
    if (!routed::first_choice(routes, SEISMIC_ROUTES_STRIDE_0, SEISMIC_ROUTES_STRIDE_1, slots, at))
        return;
    __shared__ projection::u64 rows_at[SEISMIC_CONST_DECODE_ROWS], products_at[SEISMIC_CONST_DECODE_ROWS];
    __shared__ unsigned count;
    if (threadIdx.x == 0) {
        projection::u32 chosen[SEISMIC_CONST_DECODE_ROWS];
        count = routed::expert_choices(routes, SEISMIC_ROUTES_STRIDE_0, SEISMIC_ROUTES_STRIDE_1,
                                       (unsigned)SEISMIC_DIM_M, slots, at, chosen);
        for (unsigned r = 0; r < count; ++r) {
            const projection::u64 m = chosen[r] / slots, k = chosen[r] % slots;
            rows_at[r] = m * GATHERED_STRIDE;
            products_at[r] = m * SEISMIC_RESULT_0_STRIDE_0 + k * SEISMIC_RESULT_0_STRIDE_1;
        }
    }
    __syncthreads();
    const projection::u64 expert =
        (projection::u64)routed::route(routes, SEISMIC_ROUTES_STRIDE_0, SEISMIC_ROUTES_STRIDE_1, slots, at);
    const Weights weights = CHOICE_WEIGHTS(at / slots, at % slots, (int)expert);
    // An expert with no record (every choice of an expert has its one
    // source) has a zero product for each of its choices.
    if (weights.none()) {
        for (unsigned r = 0; r < count; ++r)
            routed::zero_block_rows<Shape>(SEISMIC_DIM_F, [&](projection::u64 n) {
                element::put<ELEMENT_OF(SEISMIC_ELEMENT_A)>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER), products_at[r] + n,
                                                            0.0f);
            });
        return;
    }
    const projection::u64 group = Shape::tile_group();
    const auto epi = projection::scaling<(SEISMIC_DIM_GS + SEISMIC_DIM_US != 0)>::wrap(
        ChoiceEpi{Epi{SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER), 0, (int)SEISMIC_PARAM_ACTIVATION}, products_at},
        projection::scale_factor(SEISMIC_PTR(SEISMIC_BUFFER_GATE_SCALE), SEISMIC_DIM_GS,
                                 SEISMIC_GATE_SCALE_STRIDE_0, expert),
        projection::scale_factor(SEISMIC_PTR(SEISMIC_BUFFER_UP_SCALE), SEISMIC_DIM_US, SEISMIC_UP_SCALE_STRIDE_0,
                                 expert));
    projection::gemv_segment<Shape>(shared, routed::GatheredRows{GATHERED_ROWS, rows_at}, count, SEISMIC_DIM_H / 64,
                                    group, SEISMIC_DIM_F, KERNEL_W0_MATRIX(weights.gate.stack, weights.gate.record),
                                    KERNEL_W1_MATRIX(weights.up.stack, weights.up.record), epi);
}
#endif
