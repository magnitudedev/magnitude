// The decode down projections of the general routed feed-forward (M <=
// VERIFY_ROWS), shared by `routed_down` and `routed_down_project`; the CUDA
// form of `metal/lib/routed/down.h`. The choices' down projections run in
// parallel (block (x, m K + k) owns the output channels of tile group x for
// choice k of row m) and publish each projection in A through
// `ROUTED_PROJECTED_OUT`; the sum is `lib/routed/down_sum.cuh`'s.
#define KERNEL_W0 SEISMIC_EXPERT_DOWN
#include "routed.cuh"

SEISMIC_PROGRAMMATIC_DEPENDENCY

#ifndef ROUTED_PROJECTED_S0
#define ROUTED_PROJECTED_S0 (SEISMIC_DIM_K * SEISMIC_DIM_H)
#define ROUTED_PROJECTED_S1 SEISMIC_DIM_H
#define ROUTED_PROJECTED_S2 1ull
#endif

using Raw = projection::Plain<ELEMENT_OF(SEISMIC_ELEMENT_A), projection::AllRows>;
using Pro = projection::Operands<projection::OperandRecipe<packets::W0_OPERANDS_NVFP4>::NVFP4, Raw>;

// Where choice (m, k) of `expert` reads its down weights: a bank record, or
// its tier record in place. In the kernel's own scope, where its ports are.
#define DOWN_SOURCE(m, k, expert)                                                                          \
    routed::decode_source(SEISMIC_PTR(SEISMIC_BUFFER_SOURCES), ROUTED_SOURCE_ROWS, SEISMIC_SOURCES_STRIDE_1, \
                          SEISMIC_SOURCES_STRIDE_2, (m), (k), (expert))
#define DOWN_WEIGHTS(m, k, expert)                                                                         \
    (slab::Experts{SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_DOWN), SEISMIC_PTR(SEISMIC_BUFFER_TIER_DOWN)}         \
         .matrix(DOWN_SOURCE(m, k, expert)))

struct ProjectedEpi {
    element::Out<activation::Act> projected;
    projection::u64 s2;
    __device__ __forceinline__ void operator()(unsigned, projection::u64 n, float value, float) const {
        projected.put(n * s2, value);
    }
};

constexpr bool NVFP4 = projection::OperandRecipe<packets::W0_OPERANDS_NVFP4>::NVFP4;

struct ChoiceProjectedEpi {
    element::Out<activation::Act> projected;
    const projection::u64 *offsets;
    projection::u64 s2;
    __device__ __forceinline__ void operator()(unsigned r, projection::u64 n, float value, float) const {
        projected.put(offsets[r] + n * s2, value);
    }
};

#define ROUTED_DOWN_SUMMED_BODY(TPW, KSPLIT)                                                                       \
    using Shape = routed::DecodeShape<TPW, KSPLIT>;                                                                \
    using Readers = routed::DecodeReaders<packets::W0>;                                                            \
    __shared__ projection::GemvShared<Shape, projection::SharedRow, 0, routed::DecodePoints<packets::W0>> shared;  \
    const projection::u64 m = blockIdx.y / SEISMIC_DIM_K, k = blockIdx.y % SEISMIC_DIM_K;                          \
    const int *routes = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES));                         \
    const slab::Matrix down = DOWN_WEIGHTS(m, k, routes[m * SEISMIC_ROUTES_STRIDE_0 + k * SEISMIC_ROUTES_STRIDE_1]); \
    const projection::u64 s2 = ROUTED_PROJECTED_S2;                                                                \
    const element::Out<activation::Act> projected =                                                                   \
        ROUTED_PROJECTED_OUT.offset(m * (ROUTED_PROJECTED_S0) + k * (ROUTED_PROJECTED_S1));                        \
    if (down.stack == nullptr) {                                                                                   \
        seismic_dependency_start();                                                                                \
        routed::zero_block_rows<Shape>(SEISMIC_DIM_H, [&](projection::u64 n) { projected.put(n * s2, 0.0f); });    \
        return;                                                                                                    \
    }                                                                                                              \
    projection::stage_points(Readers{});                                                                           \
    const Pro pro{Raw{SEISMIC_PTR(SEISMIC_BUFFER_PRODUCT) +                                                        \
                          (m * SEISMIC_PRODUCT_STRIDE_0 + k * SEISMIC_PRODUCT_STRIDE_1) * 2,                       \
                      0, projection::AllRows{}}};                                                                  \
    const projection::u64 group = Shape::tile_group();                                                             \
    projection::gemv_segment_ready<Shape>(shared, projection::shared_row_after_dependency(pro, SEISMIC_DIM_F),     \
                                          1u, SEISMIC_DIM_F / 64, group, SEISMIC_DIM_H,                            \
                                          PACKETS_MATRIX(KERNEL_W0, down), projection::NoWeight{},                 \
                                          ProjectedEpi{projected, s2});

#define ROUTED_DOWN_STAGE_BODY                                                                                     \
    seismic_dependency_start();                                                                                    \
    const projection::u64 m = blockIdx.x / SEISMIC_DIM_K, k = blockIdx.x % SEISMIC_DIM_K;                          \
    const Raw row{SEISMIC_PTR(SEISMIC_BUFFER_PRODUCT) +                                                            \
                      (m * SEISMIC_PRODUCT_STRIDE_0 + k * SEISMIC_PRODUCT_STRIDE_1) * 2,                           \
                  0, projection::AllRows{}};                                                                       \
    projection::form_row(projection::Nvfp4Rows<Raw>{row}, 0u, SEISMIC_DIM_F,                                       \
                         SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_EXPERT_DOWN_OPERANDS) + blockIdx.x * SEISMIC_DIM_F * 2);

#define ROUTED_DOWN_GATHERED_BODY(TPW, KSPLIT)                                                                     \
    using Shape = routed::DecodeShape<TPW, KSPLIT>;                                                                \
    using Readers = routed::DecodeReaders<packets::W0>;                                                            \
    __shared__ projection::GemvShared<Shape, Raw, 0, routed::DecodePoints<packets::W0>> shared;                    \
    __shared__ routed::Gathered table;                                                                             \
    const int *routes = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES));                         \
    const unsigned choice = blockIdx.y, slots = (unsigned)SEISMIC_DIM_K;                                           \
    const auto record = [&](unsigned c) {                                                                          \
        const int ex = routed::route(routes, SEISMIC_ROUTES_STRIDE_0, SEISMIC_ROUTES_STRIDE_1, slots, c);          \
        return routed::ChoiceKey{ex, DOWN_SOURCE(c / slots, c % slots, ex).code, true};                            \
    };                                                                                                             \
    const projection::u64 row0 = NVFP4 ? SEISMIC_DIM_K * SEISMIC_DIM_F : SEISMIC_PRODUCT_STRIDE_0;                 \
    const projection::u64 row1 = NVFP4 ? SEISMIC_DIM_F : SEISMIC_PRODUCT_STRIDE_1;                                 \
    if (!routed::gather(table, record, (unsigned)SEISMIC_DIM_M, slots, choice, row0, row1,                         \
                        ROUTED_PROJECTED_S0, ROUTED_PROJECTED_S1)) {                                               \
        seismic_dependency_start();                                                                                \
        return;                                                                                                    \
    }                                                                                                              \
    const slab::Matrix down = DOWN_WEIGHTS(choice / slots, choice % slots, table.expert);                          \
    const projection::u64 s2 = ROUTED_PROJECTED_S2;                                                                \
    const element::Out<activation::Act> projected = ROUTED_PROJECTED_OUT;                                          \
    if (down.stack == nullptr) {                                                                                   \
        seismic_dependency_start();                                                                                \
        for (unsigned r = 0; r < table.count; ++r)                                                                 \
            routed::zero_block_rows<Shape>(                                                                        \
                SEISMIC_DIM_H, [&](projection::u64 n) { projected.put(table.products[r] + n * s2, 0.0f); });       \
        return;                                                                                                    \
    }                                                                                                              \
    projection::stage_points(Readers{});                                                                           \
    const projection::u64 group = Shape::tile_group();                                                             \
    const projection::u8 *rows =                                                                                   \
        NVFP4 ? SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_EXPERT_DOWN_OPERANDS) : SEISMIC_PTR(SEISMIC_BUFFER_PRODUCT);    \
    projection::gemv_segment_ready<Shape>(shared, projection::after_dependency(routed::GatheredRows{rows, table.rows}), \
                                          table.count, SEISMIC_DIM_F / 64, group, SEISMIC_DIM_H,                   \
                                          PACKETS_MATRIX(KERNEL_W0, down), projection::NoWeight{},                 \
                                          ChoiceProjectedEpi{projected, table.products, s2});
