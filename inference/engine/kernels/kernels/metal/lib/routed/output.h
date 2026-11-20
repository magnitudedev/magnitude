// The decode output projections of Qwen's fused routed feed-forward (M <=
// VERIFY_ROWS), shared by `routed_output` and `routed_output_project`: threadgroup (x, m)
// owns the TILE output channels of tile x for row m and runs K + 1
// simdgroups side by side: simdgroup k < K projects choice k through its
// expert's down rows, simdgroup K the shared
// expert's down rows. Each lane group (LANES lanes) of a simdgroup owns ROWS
// channels; its lanes own the packets sub, sub + LANES, ... of every
// channel's row (`routed::project_channels`).
//
// The projections meet in threadgroup memory; their join is
// `lib/routed/output_sum.h`'s. Gathered (`routed_output`'s GATHERED,
// `routed::output_project_gathered`), each expert's first choice simdgroup
// projects every choice of that expert in one weight pass, and row 0's shared
// expert simdgroup projects every row in one weight pass, publishing directly
// to `published`.

#define KERNEL_W0 SEISMIC_EXPERT_DOWN
#define KERNEL_W1 SEISMIC_SHARED_DOWN
// The includer defines ROUTED_SOURCE_ROWS, the sources' rows: the entry's
// XS (0 for a resident layer), or SOURCE_ROWS where the entry's layers
// stream.
#include "channels.h"

namespace routed {

template <uint ROWS, uint LANES> constexpr uint output_tile() {
    return (32u / LANES) * ROWS;
}

// The product packets each simdgroup stages at once (`project_channels`).
template <uint LANES> constexpr uint output_stage() {
    constexpr uint COLUMNS = uint(SEISMIC_DIM_F) > uint(SEISMIC_DIM_S) ? uint(SEISMIC_DIM_F) : uint(SEISMIC_DIM_S);
    return stage_packets<uint(SEISMIC_DIM_K) + 1u, LANES, (COLUMNS + 31u) / 32u, false>();
}

// Each simdgroup's region of the projections in threadgroup memory, in
// floats (`projection_region`).
template <uint ROWS, uint LANES> constexpr uint output_region() {
    return projection_region<output_tile<ROWS, LANES>(), output_stage<LANES>()>();
}

// Slot `slot`'s projection of row m, by one simdgroup: choice k = slot < K
// (its down rows of `expert_down`, `decode_source`, `expert_matrix`: a bank
// record of its table or a host tier record of `tier_down` read in place),
// or the shared expert (slot K), into `projected` (TILE channels from
// `first`; its `output_region()` floats, a stage of its product while it
// projects).
template <uint ROWS, uint LANES>
inline void output_project(device const uchar *expert_product, device const uchar *shared_product,
    device const int *routes, device const int *sources, device const ulong *expert_down,
    device const ulong *tier_down, device const uchar *shared_down, ulong m, uint first, uint slot, uint lane,
    threadgroup float *projected, constant ulong *seismic_words) {
    constexpr uint TILE = output_tile<ROWS, LANES>();
    constexpr uint K = uint(SEISMIC_DIM_K);
    const uint rows = uint(SEISMIC_DIM_H);
    if (slot < K) {
        const ulong k = slot;
        const slab::Source source = decode_source(sources, ROUTED_SOURCE_ROWS, SEISMIC_SOURCES_STRIDE_1,
            SEISMIC_SOURCES_STRIDE_2, m, k, routes[m * SEISMIC_ROUTES_STRIDE_0 + k * SEISMIC_ROUTES_STRIDE_1]);
        if (source.none()) {
            // A choice reading no record (the CPU's, or a failed one): a zero
            // projection.
            for (uint channel = lane; channel < TILE; channel += 32)
                projected[channel] = 0.0f;
            return;
        }
        const auto in = activation(expert_product
                + (m * SEISMIC_EXPERT_PRODUCT_STRIDE_0 + k * SEISMIC_EXPERT_PRODUCT_STRIDE_1) * Act::bytes,
            0, SEISMIC_EXPERT_PRODUCT_STRIDE_2, uint(SEISMIC_DIM_F));
        const auto down = expert_matrix<packets::W0>(Experts{expert_down, tier_down}, source,
            KERNEL_W0_LAYOUT(SEISMIC_DIM_F), SEISMIC_DIM_F);
        // The down weights' operand rows (the product rows, or NVFP4
        // operands), which each simdgroup forms alone.
        const auto x = projection::operands<projection::OperandRecipe<packets::W0_OPERANDS_NVFP4>::NVFP4>::lone_row(
            in, uint(SEISMIC_DIM_F), lane);
        project_channels<packets::W0, ROWS, LANES, uint(SEISMIC_DIM_F), output_stage<LANES>()>(x, down, first, rows,
            uint(SEISMIC_DIM_F), projected, lane);
        return;
    }
    const auto in = activation(shared_product + m * SEISMIC_SHARED_PRODUCT_STRIDE_0 * Act::bytes, 0,
        SEISMIC_SHARED_PRODUCT_STRIDE_1, uint(SEISMIC_DIM_S));
    const auto down = weights<packets::W1>(shared_down, KERNEL_W1_LAYOUT(SEISMIC_DIM_S), 0, SEISMIC_DIM_S);
    const auto x = projection::operands<projection::OperandRecipe<packets::W1_OPERANDS_NVFP4>::NVFP4>::lone_row(
        in, uint(SEISMIC_DIM_S), lane);
    project_channels<packets::W1, ROWS, LANES, uint(SEISMIC_DIM_S), output_stage<LANES>()>(x, down, first, rows,
        uint(SEISMIC_DIM_S), projected, lane);
}

// Gathered decode output projections: `sg < K` projects every choice sharing
// `(m, sg)`'s expert and source in one weight pass (`project_gathered_expert`);
// `sg == K` at `m == 0` projects all `m_rows` rows through `shared_down` in
// one weight pass. Each slot is published directly in A into `published`.
template <uint ROWS, uint LANES, typename Published>
inline void output_project_gathered(device const uchar *expert_product, device const uchar *shared_product,
    device const int *routes, device const int *sources, device const ulong *expert_down,
    device const ulong *tier_down, device const uchar *shared_down, uint m_rows, uint m, uint first, uint sg,
    uint lane, threadgroup float *projected, Published published, constant ulong *seismic_words) {
    constexpr uint K = uint(SEISMIC_DIM_K);
    constexpr uint SLOTS = K + 1u;
    constexpr uint H = uint(SEISMIC_DIM_H);
    if (sg < K) {
        constexpr bool NVFP4 = projection::OperandRecipe<packets::W0_OPERANDS_NVFP4>::NVFP4;
        project_gathered_expert<packets::W0, ROWS, LANES, K, SLOTS, uint(SEISMIC_DIM_F), output_stage<LANES>(), H,
            NVFP4>(expert_product, SEISMIC_EXPERT_PRODUCT_STRIDE_0, SEISMIC_EXPERT_PRODUCT_STRIDE_1,
            SEISMIC_EXPERT_PRODUCT_STRIDE_2, routes, SEISMIC_ROUTES_STRIDE_0, SEISMIC_ROUTES_STRIDE_1, sources,
            ROUTED_SOURCE_ROWS, SEISMIC_SOURCES_STRIDE_1, SEISMIC_SOURCES_STRIDE_2, expert_down, tier_down,
            KERNEL_W0_LAYOUT(SEISMIC_DIM_F), m_rows, m, first, sg, lane, projected, published);
        return;
    }
    if (m == 0 && m_rows > 0) {
        uint products[SEISMIC_CONST_VERIFY_ROWS];
        for (uint row = 0; row < m_rows; ++row)
            products[row] = row * SLOTS + K;
        const auto in = activation(shared_product, SEISMIC_SHARED_PRODUCT_STRIDE_0, SEISMIC_SHARED_PRODUCT_STRIDE_1,
            uint(SEISMIC_DIM_S));
        const auto down = weights<packets::W1>(shared_down, KERNEL_W1_LAYOUT(SEISMIC_DIM_S), 0, SEISMIC_DIM_S);
        constexpr bool NVFP4 = projection::OperandRecipe<packets::W1_OPERANDS_NVFP4>::NVFP4;
        PROJECTION_FOR_ROWS(m_rows,
            (project_channels_published<packets::W1, ROWS, LANES, MAXM, uint(SEISMIC_DIM_S), output_stage<LANES>(), H,
                NVFP4>(in, down, m_rows, products, first, H, uint(SEISMIC_DIM_S), projected, published, lane)));
    }
}

} // namespace routed
