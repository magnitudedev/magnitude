// The decode down projections of the general routed feed-forward (M <=
// DECODE_ROWS), shared by `routed_down` and `routed_down_project`:
// threadgroup (x, m) owns the TILE output channels of tile x for row m and
// runs K simdgroups side by side, simdgroup k projecting choice k through its
// expert's down rows (`routed::project_channels`). The projections meet in
// threadgroup memory; their sum is `lib/routed/down_sum.h`'s.

#define KERNEL_W0 SEISMIC_EXPERT_DOWN
// The includer defines ROUTED_SOURCE_ROWS, the sources' rows: the entry's
// XS (0 for a resident layer), or SOURCE_ROWS where the entry's layers
// stream.
#include "channels.h"

namespace routed {

// Output channels per threadgroup.
template <uint ROWS, uint LANES> constexpr uint down_tile() {
    return (32u / LANES) * ROWS;
}

// The product packets each simdgroup stages at once (`project_channels`),
// at 16 lanes too where the includer defines ROUTED_DOWN_STAGES_AT_16
// (`routed_down`).
template <uint LANES> constexpr uint down_stage() {
#ifdef ROUTED_DOWN_STAGES_AT_16
    constexpr bool AT_16 = true;
#else
    constexpr bool AT_16 = false;
#endif
    return stage_packets<uint(SEISMIC_DIM_K), LANES, (uint(SEISMIC_DIM_F) + 31u) / 32u, AT_16>();
}

// Each simdgroup's region of the projections in threadgroup memory, in
// floats (`projection_region`).
template <uint ROWS, uint LANES> constexpr uint down_region() {
    return projection_region<down_tile<ROWS, LANES>(), down_stage<LANES>()>();
}

// Simdgroup `sg`'s projection of choice k = sg of row m into `projected`
// (TILE channels from `first`; its `down_region()` floats, a stage of its
// product while it projects), its down rows of `expert_down`
// (`choice_record`, `choice_weights`): a bank record of its table or a host
// tier record of `tier_down` read in place.
template <uint ROWS, uint LANES>
inline void down_project(device const uchar *product, device const int *routes, device const int *sources,
    device const ulong *expert_down, device const ulong *tier_down, ulong m, uint first, uint sg, uint lane,
    threadgroup float *projected, constant ulong *seismic_words) {
    constexpr uint TILE = down_tile<ROWS, LANES>();
    const ulong k = sg;
    const int record = choice_record(expert_down, sources, ROUTED_SOURCE_ROWS, SEISMIC_SOURCES_STRIDE_1,
        SEISMIC_SOURCES_STRIDE_2, m, k, routes[m * SEISMIC_ROUTES_STRIDE_0 + k * SEISMIC_ROUTES_STRIDE_1]);
    if (record == -1) {
        // A choice reading no record (the CPU's, or a failed one): a zero
        // projection.
        for (uint channel = lane; channel < TILE; channel += 32)
            projected[channel] = 0.0f;
        return;
    }
    const auto in = activation(product + (m * SEISMIC_PRODUCT_STRIDE_0 + k * SEISMIC_PRODUCT_STRIDE_1) * Act::bytes, 0,
        SEISMIC_PRODUCT_STRIDE_2, uint(SEISMIC_DIM_F));
    const auto down = choice_weights<packets::W0>(expert_down, tier_down, record, KERNEL_W0_LAYOUT(SEISMIC_DIM_F),
        SEISMIC_DIM_H, SEISMIC_DIM_F);
    // The down weight's operand row (the product row, or NVFP4 operands),
    // which this choice's simdgroup forms alone.
    const auto x = projection::operands<projection::OperandRecipe<packets::W0_OPERANDS_NVFP4>::NVFP4>::lone_row(
        in, uint(SEISMIC_DIM_F), lane);
    project_channels<packets::W0, ROWS, LANES, uint(SEISMIC_DIM_F), down_stage<LANES>()>(x, down, first,
        uint(SEISMIC_DIM_H), uint(SEISMIC_DIM_F), projected, lane);
}

} // namespace routed
