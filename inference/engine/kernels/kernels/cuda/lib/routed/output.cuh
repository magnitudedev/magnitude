// The decode output projections of Qwen's fused routed feed-forward (M <=
// DECODE_ROWS), shared by `routed_output` and `routed_output_project`; the
// CUDA form of `metal/lib/routed/output.h`.
// Block (x, y) works on the output channels of tile group x for y =
// m * (K + 1) + slot (slot < K is choice (m, slot), slot K the shared
// expert) and publishes each projection in A at ROUTED_PUBLISHED
// (`[M, K + 1, H]`, row-major, of `element::Act::storage`). Then the includer's hook
// runs: `routed_output`'s last block of a channel tile to arrive joins its
// channels (`lib/routed/output_sum.cuh`), the arithmetic of one block carrying
// every slot in turn; `routed_output_project` leaves the join to
// `routed_output_sited`. Running the projections side by side keeps enough
// weight streams in flight to reach bandwidth.
// - `routed_output` (one row): block (x, slot) projects slot's down
//   projection; the hook `ROUTED_OUTPUT_ARRIVED(m, blocks, arrival, first,
//   channels, published)` follows, with the K + 1 blocks of tile x of row m.
// - `routed_output_gathered` (several rows): choice (m, slot) projects only
//   when it is its expert's first choice, the down projection of every
//   choice of that expert (`routed::expert_choices`), so each chosen expert
//   streams once; slot K of row 0 projects the shared expert for every row.
//   The hook `ROUTED_OUTPUT_GATHERED_ARRIVED(blocks, arrival, first,
//   channels, published)` follows, with tile x's M * (K + 1) blocks.
// Each choice's down weights are read where its source names
// (`routed::decode_source`): a bank record, or a host tier record read in
// place; a choice with no record (the CPU's, or a failed one) projects
// zero.
//
// The includer defines ROUTED_SOURCE_ROWS, ROUTED_PUBLISHED (in the kernel's
// scope) and the two hooks.
#define KERNEL_W0 SEISMIC_EXPERT_DOWN
#define KERNEL_W1 SEISMIC_SHARED_DOWN
#include "routed.cuh"

SEISMIC_PROGRAMMATIC_DEPENDENCY

using Pro = projection::Plain<ELEMENT_OF(SEISMIC_ELEMENT_A), projection::AllRows>;

// The product rows the experts' and the shared expert's down projections
// read: as they are published, or where the down weight takes NVFP4 operands
// the rows the stage launch formed (choice (m, k) at row m K + k).
constexpr bool EXPERT_NVFP4 = projection::OperandRecipe<packets::W0_OPERANDS_NVFP4>::NVFP4;
constexpr bool SHARED_NVFP4 = projection::OperandRecipe<packets::W1_OPERANDS_NVFP4>::NVFP4;
#define EXPERT_ROWS \
    (EXPERT_NVFP4 ? SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_EXPERT_OPERANDS) : SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_PRODUCT))
#define EXPERT_STRIDE_0 (EXPERT_NVFP4 ? SEISMIC_DIM_K * SEISMIC_DIM_F : SEISMIC_EXPERT_PRODUCT_STRIDE_0)
#define EXPERT_STRIDE_1 (EXPERT_NVFP4 ? SEISMIC_DIM_F : SEISMIC_EXPERT_PRODUCT_STRIDE_1)
#define SHARED_ROWS \
    (SHARED_NVFP4 ? SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_SHARED_OPERANDS) : SEISMIC_PTR(SEISMIC_BUFFER_SHARED_PRODUCT))
#define SHARED_STRIDE_0 (SHARED_NVFP4 ? SEISMIC_DIM_S : SEISMIC_SHARED_PRODUCT_STRIDE_0)

#ifdef SEISMIC_FORMING_ROUTED_OUTPUT_STAGE
// Block b < M K: choice (b / K, b % K)'s product row; else row b - M K of
// the shared product. Each as NVFP4 operands where its down weight takes
// them.
__global__ void routed_output_stage(SEISMIC_KERNEL_PARAMS) {
    seismic_dependency_start();
    const projection::u64 K = SEISMIC_DIM_K, b = blockIdx.x;
    if (b < SEISMIC_DIM_M * K) {
        if constexpr (EXPERT_NVFP4) {
            const projection::u64 m = b / K, k = b % K;
            const Pro row{SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_PRODUCT) +
                              (m * SEISMIC_EXPERT_PRODUCT_STRIDE_0 + k * SEISMIC_EXPERT_PRODUCT_STRIDE_1) * 2,
                          0, projection::AllRows{}};
            projection::form_row(projection::Nvfp4Rows<Pro>{row}, 0, SEISMIC_DIM_F,
                                 SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_EXPERT_OPERANDS) + b * SEISMIC_DIM_F * 2);
        }
    } else if constexpr (SHARED_NVFP4) {
        const projection::u64 m = b - SEISMIC_DIM_M * K;
        const Pro row{SEISMIC_PTR(SEISMIC_BUFFER_SHARED_PRODUCT) + m * SEISMIC_SHARED_PRODUCT_STRIDE_0 * 2, 0,
                      projection::AllRows{}};
        projection::form_row(projection::Nvfp4Rows<Pro>{row}, 0, SEISMIC_DIM_S,
                             SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_SHARED_OPERANDS) + m * SEISMIC_DIM_S * 2);
    }
}
#endif

// Where choice (m, k) of `expert` reads its down weights: a bank record, or
// its tier record in place. In the kernel's own scope, where its ports are.
#define DOWN_WEIGHTS(m, k, expert)                                                                         \
    routed::Place::of(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_DOWN), SEISMIC_PTR(SEISMIC_BUFFER_TIER_DOWN),      \
                      routed::decode_source(SEISMIC_PTR(SEISMIC_BUFFER_SOURCES), ROUTED_SOURCE_ROWS,           \
                                            SEISMIC_SOURCES_STRIDE_1, SEISMIC_SOURCES_STRIDE_2, (m), (k),  \
                                            (expert)))


struct PublishEpi {
    element::Act::storage *published;
    __device__ __forceinline__ void operator()(unsigned, projection::u64 n, float value, float) const {
        published[n] = element::Act::store(value);
    }
};

#ifdef SEISMIC_FORMING_ROUTED_OUTPUT
template <int TPW, int KSPLIT>
__global__ void routed_output(SEISMIC_KERNEL_PARAMS) {
    using Shape = routed::DecodeShape<TPW, KSPLIT>;
    constexpr projection::u64 CHANNELS = Shape::GROUPS * TPW * 16;
    __shared__ projection::GemvShared<Shape, Pro> shared;
    const projection::u64 H = SEISMIC_DIM_H, K = SEISMIC_DIM_K;
    const projection::u64 m = blockIdx.y / (K + 1), slot = blockIdx.y % (K + 1);
    const projection::u64 group = Shape::tile_group();
    element::Act::storage *published = ROUTED_PUBLISHED + m * (K + 1) * H;

    const PublishEpi epi{published + slot * H};
    // The arrival and the combine follow the launch before.
    if (slot < K) {
        const int *routes = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES));
        const routed::Place down =
            DOWN_WEIGHTS(m, slot, routes[m * SEISMIC_ROUTES_STRIDE_0 + slot * SEISMIC_ROUTES_STRIDE_1]);
        if (down.stack == nullptr) {
            // A choice reading no record: each
            // block publishes its channels of a zero projection.
            seismic_dependency_start();
            routed::zero_block_rows<Shape>(H, [&](projection::u64 n) { epi.published[n] = element::Act::store(0.0f); });
        } else {
            const Pro pro{EXPERT_ROWS + (m * EXPERT_STRIDE_0 + slot * EXPERT_STRIDE_1) * 2, 0, projection::AllRows{}};
            projection::gemv_segment_ready<Shape>(shared, projection::after_dependency(pro), 1u, SEISMIC_DIM_F / 64, group, H,
                                            KERNEL_W0_MATRIX(down.stack, down.record),
                                            projection::NoWeight{}, epi);
        }
    } else {
        const Pro pro{SHARED_ROWS + m * SHARED_STRIDE_0 * 2, 0, projection::AllRows{}};
        projection::gemv_segment_ready<Shape>(shared, projection::after_dependency(pro), 1u, SEISMIC_DIM_S / 64, group, H,
                                        KERNEL_W1_AT(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_DOWN)),
                                        projection::NoWeight{}, epi);
    }

    ROUTED_OUTPUT_ARRIVED(m, static_cast<unsigned>(K) + 1u, m * gridDim.x + blockIdx.x, blockIdx.x * CHANNELS, CHANNELS,
                          published)
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_OUTPUT_GATHERED
// Each GEMV row's projection A-rounded at its published offset.
struct GatheredPublishEpi {
    element::Act::storage *published;
    const projection::u64 *offsets;
    __device__ __forceinline__ void operator()(unsigned m, projection::u64 n, float value, float) const {
        published[offsets[m] + n] = element::Act::store(value);
    }
};

template <int TPW, int KSPLIT>
__global__ void routed_output_gathered(SEISMIC_KERNEL_PARAMS) {
    using Shape = routed::DecodeShape<TPW, KSPLIT>;
    constexpr projection::u64 CHANNELS = Shape::GROUPS * TPW * 16;
    __shared__ projection::GemvShared<Shape, routed::GatheredRows> shared;
    __shared__ projection::u64 rows_at[SEISMIC_CONST_DECODE_ROWS], published_at[SEISMIC_CONST_DECODE_ROWS];
    __shared__ unsigned count;
    const projection::u64 H = SEISMIC_DIM_H, K = SEISMIC_DIM_K, M = SEISMIC_DIM_M;
    const projection::u64 m = blockIdx.y / (K + 1), slot = blockIdx.y % (K + 1);
    const projection::u64 group = Shape::tile_group();
    element::Act::storage *published = ROUTED_PUBLISHED;
    const int *routes = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES));

    // The routes come from the launch before the expansion, which has
    // completed (the expansion lets this launch start only after its own
    // wait), so every block reads them and issues its first weight loads
    // before waiting.
    const unsigned choice = (unsigned)(m * K + slot);
    const bool projects = slot < K
        ? routed::first_choice(routes, SEISMIC_ROUTES_STRIDE_0, SEISMIC_ROUTES_STRIDE_1, (unsigned)K, choice)
        : m == 0;
    if (projects && threadIdx.x == 0) {
        if (slot < K) {
            projection::u32 chosen[SEISMIC_CONST_DECODE_ROWS];
            count = routed::expert_choices(routes, SEISMIC_ROUTES_STRIDE_0, SEISMIC_ROUTES_STRIDE_1, (unsigned)M,
                                           (unsigned)K, choice, chosen);
            for (unsigned r = 0; r < count; ++r) {
                const projection::u64 row = chosen[r] / K, choice_slot = chosen[r] % K;
                rows_at[r] = row * EXPERT_STRIDE_0 + choice_slot * EXPERT_STRIDE_1;
                published_at[r] = (row * (K + 1) + choice_slot) * H;
            }
        } else {
            count = (unsigned)M;
            for (unsigned r = 0; r < count; ++r) {
                rows_at[r] = r * SHARED_STRIDE_0;
                published_at[r] = (r * (K + 1) + K) * H;
            }
        }
    }
    __syncthreads();
    if (projects) {
        const GatheredPublishEpi epi{published, published_at};
        if (slot < K) {
            const routed::Place down = DOWN_WEIGHTS(
                m, slot, routed::route(routes, SEISMIC_ROUTES_STRIDE_0, SEISMIC_ROUTES_STRIDE_1, (unsigned)K, choice));
            if (down.stack == nullptr) {
                // An expert reading no record: each
                // block publishes its channels of each of its choices' zero
                // projections.
                seismic_dependency_start();
                for (unsigned r = 0; r < count; ++r)
                    routed::zero_block_rows<Shape>(H, [&](projection::u64 n) { published[published_at[r] + n] = element::Act::store(0.0f); });
            } else {
                const routed::GatheredRows rows{EXPERT_ROWS, rows_at};
                projection::gemv_segment_ready<Shape>(
                    shared, projection::after_dependency(rows), count, SEISMIC_DIM_F / 64, group, H,
                    KERNEL_W0_MATRIX(down.stack, down.record), projection::NoWeight{}, epi);
            }
        } else {
            const routed::GatheredRows rows{SHARED_ROWS, rows_at};
            projection::gemv_segment_ready<Shape>(shared, projection::after_dependency(rows), count,
                                                  SEISMIC_DIM_S / 64, group, H,
                                                  KERNEL_W1_AT(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_DOWN)),
                                                  projection::NoWeight{}, epi);
        }
    } else {
        seismic_dependency_start();
    }

    ROUTED_OUTPUT_GATHERED_ARRIVED(gridDim.y, blockIdx.x, blockIdx.x * CHANNELS, CHANNELS, published)
}
#endif
