// The decode output projections of Qwen's fused routed feed-forward (M <=
// VERIFY_ROWS), shared by `routed_output` and `routed_output_project`; the
// CUDA form of `metal/lib/routed/output.h`.
#define KERNEL_W0 SEISMIC_EXPERT_DOWN
#define KERNEL_W1 SEISMIC_SHARED_DOWN
#include "routed.cuh"

#ifndef ROUTED_PUBLISHED_S0
#define ROUTED_PUBLISHED_S0 ((SEISMIC_DIM_K + 1) * SEISMIC_DIM_H)
#define ROUTED_PUBLISHED_S1 SEISMIC_DIM_H
#define ROUTED_PUBLISHED_S2 1ull
#endif

SEISMIC_PROGRAMMATIC_DEPENDENCY

using Pro = projection::Plain<ELEMENT_OF(SEISMIC_ELEMENT_A), projection::AllRows>;

constexpr bool EXPERT_NVFP4 = projection::OperandRecipe<packets::W0_OPERANDS_NVFP4>::NVFP4;
constexpr bool SHARED_NVFP4 = projection::OperandRecipe<packets::W1_OPERANDS_NVFP4>::NVFP4;
#define EXPERT_ROWS \
    (EXPERT_NVFP4 ? SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_EXPERT_DOWN_OPERANDS) : SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_PRODUCT))
#define EXPERT_STRIDE_0 (EXPERT_NVFP4 ? SEISMIC_DIM_K * SEISMIC_DIM_F : SEISMIC_EXPERT_PRODUCT_STRIDE_0)
#define EXPERT_STRIDE_1 (EXPERT_NVFP4 ? SEISMIC_DIM_F : SEISMIC_EXPERT_PRODUCT_STRIDE_1)
#define SHARED_ROWS \
    (SHARED_NVFP4 ? SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_SHARED_DOWN_OPERANDS) : SEISMIC_PTR(SEISMIC_BUFFER_SHARED_PRODUCT))
#define SHARED_STRIDE_0 (SHARED_NVFP4 ? SEISMIC_DIM_S : SEISMIC_SHARED_PRODUCT_STRIDE_0)

// Where choice (m, k) of `expert` reads its down weights: a bank record, or
// its tier record in place. In the kernel's own scope, where its ports are.
#define DOWN_SOURCE(m, k, expert)                                                                          \
    routed::decode_source(SEISMIC_PTR(SEISMIC_BUFFER_SOURCES), ROUTED_SOURCE_ROWS, SEISMIC_SOURCES_STRIDE_1, \
                          SEISMIC_SOURCES_STRIDE_2, (m), (k), (expert))
#define DOWN_WEIGHTS(m, k, expert)                                                                         \
    (slab::Experts{SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_DOWN), SEISMIC_PTR(SEISMIC_BUFFER_TIER_DOWN)}         \
         .matrix(DOWN_SOURCE(m, k, expert)))

struct PublishEpi {
    element::Out<activation::Act> published;
    projection::u64 s2;
    __device__ __forceinline__ void operator()(unsigned, projection::u64 n, float value, float) const {
        published.put(n * s2, value);
    }
};

struct GatheredPublishEpi {
    element::Out<activation::Act> published;
    const projection::u64 *offsets;
    projection::u64 s2;
    __device__ __forceinline__ void operator()(unsigned m, projection::u64 n, float value, float) const {
        published.put(offsets[m] + n * s2, value);
    }
};

#define ROUTED_OUTPUT_STAGE_BODY                                                                                   \
    seismic_dependency_start();                                                                                    \
    const projection::u64 K = SEISMIC_DIM_K, b = blockIdx.x;                                                       \
    if (b < SEISMIC_DIM_M * K) {                                                                                   \
        if constexpr (EXPERT_NVFP4) {                                                                              \
            const projection::u64 m = b / K, k = b % K;                                                            \
            const Pro row{SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_PRODUCT) +                                             \
                              (m * SEISMIC_EXPERT_PRODUCT_STRIDE_0 + k * SEISMIC_EXPERT_PRODUCT_STRIDE_1) * 2,     \
                          0, projection::AllRows{}};                                                               \
            projection::form_row(projection::Nvfp4Rows<Pro>{row}, 0, SEISMIC_DIM_F,                                \
                                 SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_EXPERT_DOWN_OPERANDS) + b * SEISMIC_DIM_F * 2); \
        }                                                                                                          \
    } else if constexpr (SHARED_NVFP4) {                                                                           \
        const projection::u64 m = b - SEISMIC_DIM_M * K;                                                           \
        const Pro row{SEISMIC_PTR(SEISMIC_BUFFER_SHARED_PRODUCT) + m * SEISMIC_SHARED_PRODUCT_STRIDE_0 * 2, 0,     \
                      projection::AllRows{}};                                                                      \
        projection::form_row(projection::Nvfp4Rows<Pro>{row}, 0, SEISMIC_DIM_S,                                    \
                             SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_SHARED_DOWN_OPERANDS) + m * SEISMIC_DIM_S * 2);    \
    }

#define ROUTED_OUTPUT_SUMMED_BODY(TPW, KSPLIT)                                                                     \
    using Shape = routed::DecodeShape<TPW, KSPLIT>;                                                                \
    constexpr projection::u64 CHANNELS = Shape::GROUPS * TPW * 16;                                                 \
    __shared__ projection::GemvShared<Shape, projection::SharedRow> shared;                                        \
    const projection::u64 H = SEISMIC_DIM_H, K = SEISMIC_DIM_K;                                                    \
    const projection::u64 m = blockIdx.y / (K + 1), slot = blockIdx.y % (K + 1);                                   \
    const projection::u64 group = Shape::tile_group();                                                             \
    const projection::u64 s2 = ROUTED_PUBLISHED_S2;                                                                \
    const PublishEpi epi{ROUTED_PUBLISHED_OUT.offset(m * (ROUTED_PUBLISHED_S0) + slot * (ROUTED_PUBLISHED_S1)), s2}; \
    if (slot < K) {                                                                                                \
        const int *routes = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES));                     \
        const slab::Matrix down =                                                                                  \
            DOWN_WEIGHTS(m, slot, routes[m * SEISMIC_ROUTES_STRIDE_0 + slot * SEISMIC_ROUTES_STRIDE_1]);           \
        if (down.stack == nullptr) {                                                                               \
            seismic_dependency_start();                                                                            \
            routed::zero_block_rows<Shape>(H, [&](projection::u64 n) { epi.published.put(n * s2, 0.0f); });        \
        } else {                                                                                                   \
            const Pro pro{EXPERT_ROWS + (m * EXPERT_STRIDE_0 + slot * EXPERT_STRIDE_1) * 2, 0,                     \
                          projection::AllRows{}};                                                                  \
            projection::gemv_segment_ready<Shape>(shared, projection::shared_row_after_dependency(pro, SEISMIC_DIM_F), \
                                                  1u, SEISMIC_DIM_F / 64, group, H,                                \
                                                  PACKETS_MATRIX(KERNEL_W0, down),                       \
                                                  projection::NoWeight{}, epi);                                    \
        }                                                                                                          \
    } else {                                                                                                       \
        const Pro pro{SHARED_ROWS + m * SHARED_STRIDE_0 * 2, 0, projection::AllRows{}};                            \
        projection::gemv_segment_ready<Shape>(shared, projection::shared_row_after_dependency(pro, SEISMIC_DIM_S), \
                                              1u, SEISMIC_DIM_S / 64, group, H,                                    \
                                              KERNEL_W1_AT(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_DOWN)),               \
                                              projection::NoWeight{}, epi);                                        \
    }                                                                                                              \
    ROUTED_OUTPUT_ARRIVED(m, static_cast<unsigned>(K) + 1u, m * gridDim.x + blockIdx.x, blockIdx.x * CHANNELS, CHANNELS)

#define ROUTED_OUTPUT_GATHERED_BODY(TPW, KSPLIT)                                                                   \
    using Shape = routed::DecodeShape<TPW, KSPLIT>;                                                                \
    constexpr projection::u64 CHANNELS = Shape::GROUPS * TPW * 16;                                                 \
    __shared__ projection::GemvShared<Shape, routed::GatheredRows> shared;                                         \
    __shared__ projection::u64 rows_at[SEISMIC_CONST_VERIFY_ROWS], published_at[SEISMIC_CONST_VERIFY_ROWS];        \
    __shared__ unsigned count;                                                                                     \
    const projection::u64 H = SEISMIC_DIM_H, K = SEISMIC_DIM_K, M = SEISMIC_DIM_M;                                 \
    const projection::u64 m = blockIdx.y / (K + 1), slot = blockIdx.y % (K + 1);                                   \
    const projection::u64 group = Shape::tile_group();                                                             \
    const projection::u64 s2 = ROUTED_PUBLISHED_S2;                                                                \
    const element::Out<activation::Act> published = ROUTED_PUBLISHED_OUT;                                             \
    const int *routes = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES));                         \
    const unsigned choice = (unsigned)(m * K + slot);                                                              \
    const auto record = [&](unsigned c) {                                                                          \
        const int ex = routed::route(routes, SEISMIC_ROUTES_STRIDE_0, SEISMIC_ROUTES_STRIDE_1, (unsigned)K, c);    \
        return routed::ChoiceKey{ex, DOWN_SOURCE(c / (unsigned)K, c % (unsigned)K, ex).code, true};                \
    };                                                                                                             \
    const bool projects = slot < K ? routed::first_choice(record, choice) : m == 0;                                \
    if (projects && threadIdx.x == 0) {                                                                            \
        if (slot < K) {                                                                                            \
            projection::u32 chosen[SEISMIC_CONST_VERIFY_ROWS];                                                     \
            count = routed::record_choices(record, (unsigned)(M * K), choice, chosen);                             \
            for (unsigned r = 0; r < count; ++r) {                                                                 \
                const projection::u64 row = chosen[r] / K, choice_slot = chosen[r] % K;                            \
                rows_at[r] = row * EXPERT_STRIDE_0 + choice_slot * EXPERT_STRIDE_1;                                \
                published_at[r] = row * (ROUTED_PUBLISHED_S0) + choice_slot * (ROUTED_PUBLISHED_S1);               \
            }                                                                                                      \
        } else {                                                                                                   \
            count = (unsigned)M;                                                                                   \
            for (unsigned r = 0; r < count; ++r) {                                                                 \
                rows_at[r] = r * SHARED_STRIDE_0;                                                                  \
                published_at[r] = r * (ROUTED_PUBLISHED_S0) + K * (ROUTED_PUBLISHED_S1);                           \
            }                                                                                                      \
        }                                                                                                          \
    }                                                                                                              \
    __syncthreads();                                                                                               \
    if (projects) {                                                                                                \
        const GatheredPublishEpi epi{published, published_at, s2};                                                 \
        if (slot < K) {                                                                                            \
            const slab::Matrix down = DOWN_WEIGHTS(                                                                \
                m, slot, routed::route(routes, SEISMIC_ROUTES_STRIDE_0, SEISMIC_ROUTES_STRIDE_1, (unsigned)K, choice)); \
            if (down.stack == nullptr) {                                                                           \
                seismic_dependency_start();                                                                        \
                for (unsigned r = 0; r < count; ++r)                                                               \
                    routed::zero_block_rows<Shape>(H, [&](projection::u64 n) { published.put(published_at[r] + n * s2, 0.0f); }); \
            } else {                                                                                               \
                const routed::GatheredRows rows{EXPERT_ROWS, rows_at};                                             \
                projection::gemv_segment_ready<Shape>(                                                             \
                    shared, projection::after_dependency(rows), count, SEISMIC_DIM_F / 64, group, H,               \
                    PACKETS_MATRIX(KERNEL_W0, down), projection::NoWeight{}, epi);                                 \
            }                                                                                                      \
        } else {                                                                                                   \
            const routed::GatheredRows rows{SHARED_ROWS, rows_at};                                                 \
            projection::gemv_segment_ready<Shape>(shared, projection::after_dependency(rows), count,               \
                                                  SEISMIC_DIM_S / 64, group, H,                                    \
                                                  KERNEL_W1_AT(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_DOWN)),           \
                                                  projection::NoWeight{}, epi);                                    \
        }                                                                                                          \
    } else {                                                                                                       \
        seismic_dependency_start();                                                                                \
    }                                                                                                              \
    ROUTED_OUTPUT_GATHERED_ARRIVED(gridDim.y, blockIdx.x, blockIdx.x * CHANNELS, CHANNELS)
