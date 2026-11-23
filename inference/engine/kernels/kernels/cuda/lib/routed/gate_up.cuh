// Decode expansion of gated experts (M <= VERIFY_ROWS) in CUDA: what `routed_gate_up` and
// `routed_gate_up_part` share. The includer defines KERNEL_W0, KERNEL_W1 and the glu limit, then
// ROUTED_GATE_UP_SOURCE_ROWS, the product's Out and strides (ROUTED_GATE_UP_PRODUCT_OUT,
// ROUTED_GATE_UP_PRODUCT_STRIDE_0/1), a two-pass form also ROUTED_GATE_UP_ACTIVE(c, source)
// (whether flat choice c runs in this pass; undefined, every choice runs), and places the three
// kernel bodies (ROUTED_GATE_UP_CHOICE_BODY, ROUTED_GATE_UP_STAGE_BODY,
// ROUTED_GATE_UP_GATHERED_BODY) in its forms of the kernels.
#pragma once
#include "routed.cuh"

#ifdef ROUTED_GATE_UP_ACTIVE
#define ROUTED_GATE_UP_SKIPS(c, source) (!ROUTED_GATE_UP_ACTIVE(c, source))
#define ROUTED_GATE_UP_IS_ACTIVE(c, source) ROUTED_GATE_UP_ACTIVE(c, source)
#else
#define ROUTED_GATE_UP_SKIPS(c, source) false
#define ROUTED_GATE_UP_IS_ACTIVE(c, source) true
#endif

// A block's one row, as the experts take it (NVFP4 operands or A), which the
// block forms in its shared memory (`shared_row`).
using Raw = projection::Plain<ELEMENT_OF(SEISMIC_ELEMENT_A), projection::AllRows>;
using Pro = projection::Operands<
    projection::OperandRecipe<packets::W0_OPERANDS_NVFP4, packets::W1_OPERANDS_NVFP4>::NVFP4, Raw>;
using Epi = projection::Glu<ELEMENT_OF(SEISMIC_ELEMENT_A)>;

// Where choice (m, k) of `expert` reads its gate and up weights: a bank
// record, or its tier records in place. In the kernel's own scope, where
// its ports are.
struct Weights {
    slab::Matrix gate, up;
    __device__ __forceinline__ bool none() const { return gate.stack == nullptr; }
};
#define CHOICE_SOURCE(m, k, expert)                                                                             \
    routed::decode_source(SEISMIC_PTR(SEISMIC_BUFFER_SOURCES), ROUTED_GATE_UP_SOURCE_ROWS,                      \
                          SEISMIC_SOURCES_STRIDE_1, SEISMIC_SOURCES_STRIDE_2, (m), (k), (expert))
#define CHOICE_WEIGHTS(source)                                                                                  \
    Weights{slab::Experts{SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_GATE),                                              \
                          SEISMIC_PTR(SEISMIC_BUFFER_TIER_GATE)}.matrix(source),                                \
            slab::Experts{SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_UP),                                                \
                          SEISMIC_PTR(SEISMIC_BUFFER_TIER_UP)}.matrix(source)}

// The rows the gathered GEMV reads: the normalized rows, or where the experts
// take NVFP4 operands the rows `routed_gate_up_stage` formed.
constexpr bool NVFP4 = projection::OperandRecipe<packets::W0_OPERANDS_NVFP4, packets::W1_OPERANDS_NVFP4>::NVFP4;
#define GATHERED_ROWS (NVFP4 ? SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_OPERANDS) : SEISMIC_PTR(SEISMIC_BUFFER_NORMALIZED))
#define GATHERED_STRIDE (NVFP4 ? SEISMIC_DIM_H : SEISMIC_NORMALIZED_STRIDE_0)

#define ROUTED_GATE_UP_CHOICE_BODY                                                                              \
    using Shape = routed::DecodeShape<TPW, KSPLIT>;                                                             \
    using Readers = routed::DecodeReaders<packets::W0, packets::W1>;                                            \
    __shared__ projection::GemvShared<Shape, projection::SharedRow, 0,                                          \
                                      routed::DecodePoints<packets::W0, packets::W1>> shared;                   \
    const projection::u8 *normalized = SEISMIC_PTR(SEISMIC_BUFFER_NORMALIZED);                                  \
    const int *routes = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES));                      \
    const projection::u64 m = blockIdx.y / SEISMIC_DIM_K, k = blockIdx.y % SEISMIC_DIM_K;                       \
    const projection::u64 expert = (projection::u64)routes[m * SEISMIC_ROUTES_STRIDE_0 + k * SEISMIC_ROUTES_STRIDE_1]; \
    const slab::Source source = CHOICE_SOURCE(m, k, (int)expert);                                               \
    if (ROUTED_GATE_UP_SKIPS((unsigned)blockIdx.y, source))                                                     \
        return;                                                                                                 \
    /* Block-collective: before any warp leaves. */                                                             \
    const projection::SharedRow pro = projection::shared_row(                                                   \
        Pro{Raw{normalized + m * SEISMIC_NORMALIZED_STRIDE_0 * 2, SEISMIC_NORMALIZED_STRIDE_0,                  \
                projection::AllRows{}}},                                                                        \
        SEISMIC_DIM_H);                                                                                         \
    const Weights weights = CHOICE_WEIGHTS(source);                                                             \
    const element::Out<ELEMENT_OF(SEISMIC_ELEMENT_A)> product =                                                 \
        ROUTED_GATE_UP_PRODUCT_OUT.offset(m * ROUTED_GATE_UP_PRODUCT_STRIDE_0 + k * ROUTED_GATE_UP_PRODUCT_STRIDE_1); \
    if (weights.none()) {                                                                                       \
        routed::zero_block_rows<Shape>(SEISMIC_DIM_F, [&](projection::u64 n) {                                  \
            product.put(n, 0.0f);                                                                               \
        });                                                                                                     \
        return;                                                                                                 \
    }                                                                                                           \
    projection::stage_points(Readers{});                                                                        \
    const projection::u64 group = Shape::tile_group();                                                          \
    const auto epi = projection::scaling<(SEISMIC_DIM_GS + SEISMIC_DIM_US != 0)>::wrap(                         \
        Epi{product, 0, (int)SEISMIC_PARAM_ACTIVATION},                                                         \
        projection::scale_factor(SEISMIC_PTR(SEISMIC_BUFFER_GATE_SCALE), SEISMIC_DIM_GS,                        \
                                 SEISMIC_GATE_SCALE_STRIDE_0, expert),                                          \
        projection::scale_factor(SEISMIC_PTR(SEISMIC_BUFFER_UP_SCALE), SEISMIC_DIM_US,                          \
                                 SEISMIC_UP_SCALE_STRIDE_0, expert));                                           \
    projection::gemv_segment<Shape>(shared, pro, 1u, SEISMIC_DIM_H / 64, group, SEISMIC_DIM_F,                  \
                                    PACKETS_MATRIX(KERNEL_W0, weights.gate),                                     \
                                    PACKETS_MATRIX(KERNEL_W1, weights.up), epi);

#define ROUTED_GATE_UP_STAGE_BODY                                                                               \
    projection::form_row(projection::Nvfp4Rows<Raw>{Raw{SEISMIC_PTR(SEISMIC_BUFFER_NORMALIZED),                 \
                                                        SEISMIC_NORMALIZED_STRIDE_0, projection::AllRows{}}},   \
                         blockIdx.x, SEISMIC_DIM_H, SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_OPERANDS));

#define ROUTED_GATE_UP_GATHERED_BODY                                                                            \
    using Shape = routed::DecodeGatheredShape<TPW, KSPLIT>;                                                     \
    using Readers = routed::DecodeReaders<packets::W0, packets::W1>;                                            \
    __shared__ projection::GemvShared<Shape, Raw, 0, routed::DecodePoints<packets::W0, packets::W1>> shared;    \
    __shared__ routed::Gathered table;                                                                          \
    const int *routes = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES));                      \
    const unsigned at = blockIdx.y, slots = (unsigned)SEISMIC_DIM_K;                                            \
    const auto record = [&](unsigned c) {                                                                       \
        const int ex = routed::route(routes, SEISMIC_ROUTES_STRIDE_0, SEISMIC_ROUTES_STRIDE_1, slots, c);       \
        const slab::Source src = CHOICE_SOURCE(c / slots, c % slots, ex);                                       \
        return routed::ChoiceKey{ex, src.code, ROUTED_GATE_UP_IS_ACTIVE(c, src)};                               \
    };                                                                                                          \
    if (!routed::gather(table, record, (unsigned)SEISMIC_DIM_M, slots, at, GATHERED_STRIDE, 0,                  \
                        ROUTED_GATE_UP_PRODUCT_STRIDE_0, ROUTED_GATE_UP_PRODUCT_STRIDE_1))                      \
        return;                                                                                                 \
    const projection::u64 expert = (projection::u64)table.expert;                                               \
    const Weights weights = CHOICE_WEIGHTS(CHOICE_SOURCE(at / slots, at % slots, (int)expert));                 \
    if (weights.none()) {                                                                                       \
        for (unsigned r = 0; r < table.count; ++r)                                                              \
            routed::zero_block_rows<Shape>(SEISMIC_DIM_F, [&](projection::u64 n) {                              \
                ROUTED_GATE_UP_PRODUCT_OUT.put(table.products[r] + n, 0.0f);                                    \
            });                                                                                                 \
        return;                                                                                                 \
    }                                                                                                           \
    projection::stage_points(Readers{});                                                                        \
    const projection::u64 group = Shape::tile_group();                                                          \
    const auto epi = projection::scaling<(SEISMIC_DIM_GS + SEISMIC_DIM_US != 0)>::wrap(                         \
        routed::ChoiceRows<Epi>{Epi{ROUTED_GATE_UP_PRODUCT_OUT, 0, (int)SEISMIC_PARAM_ACTIVATION},              \
                                table.products},                                                                \
        projection::scale_factor(SEISMIC_PTR(SEISMIC_BUFFER_GATE_SCALE), SEISMIC_DIM_GS,                        \
                                 SEISMIC_GATE_SCALE_STRIDE_0, expert),                                          \
        projection::scale_factor(SEISMIC_PTR(SEISMIC_BUFFER_UP_SCALE), SEISMIC_DIM_US,                          \
                                 SEISMIC_UP_SCALE_STRIDE_0, expert));                                           \
    projection::gemv_segment<Shape>(shared, routed::GatheredRows{GATHERED_ROWS, table.rows}, table.count,       \
                                    SEISMIC_DIM_H / 64, group, SEISMIC_DIM_F,                                   \
                                    PACKETS_MATRIX(KERNEL_W0, weights.gate),                                     \
                                    PACKETS_MATRIX(KERNEL_W1, weights.up), epi);
