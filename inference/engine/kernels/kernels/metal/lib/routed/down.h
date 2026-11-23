// The decode down projections of the general routed feed-forward (M <=
// VERIFY_ROWS), shared by `routed_down` and `routed_down_project`:
// threadgroup (x, m) owns the TILE output channels of tile x for row m and
// runs K simdgroups side by side, simdgroup k projecting choice k through its
// expert's down rows (`routed::project_channels`). The projections meet in
// memory; their sum is `lib/routed/down_sum.h`'s. Gathered (`routed_down`'s
// GATHERED, `routed::down_project_gathered`), each expert's first choice
// simdgroup projects every choice of that expert in one weight pass,
// publishing directly to `published`.

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
// staged at 16 lanes too past 2 packets a lane (`stage_packets`).
template <uint LANES> constexpr uint down_stage() {
    return stage_packets<uint(SEISMIC_DIM_K), LANES, (uint(SEISMIC_DIM_F) + 31u) / 32u, true>();
}

// Each simdgroup's region of the projections in threadgroup memory, in
// floats (`projection_region`).
template <uint ROWS, uint LANES> constexpr uint down_region() {
    return projection_region<down_tile<ROWS, LANES>(), down_stage<LANES>()>();
}

// Choice k of row m's projection into `projected` (TILE channels from
// `first`; its `down_region()` floats, a stage of its product while it
// projects), by one simdgroup, its down rows of `expert_down`
// (`decode_source`, `expert_matrix`): a bank record of its table or a host
// tier record of `tier_down` read in place.
template <uint ROWS, uint LANES>
inline void down_project(device const uchar *product, device const int *routes, device const int *sources,
    device const ulong *expert_down, device const ulong *tier_down, ulong m, uint first, ulong k, uint lane,
    threadgroup float *projected, constant ulong *seismic_words) {
    constexpr uint TILE = down_tile<ROWS, LANES>();
    const slab::Source source = decode_source(sources, ROUTED_SOURCE_ROWS, SEISMIC_SOURCES_STRIDE_1,
        SEISMIC_SOURCES_STRIDE_2, m, k, routes[m * SEISMIC_ROUTES_STRIDE_0 + k * SEISMIC_ROUTES_STRIDE_1]);
    if (source.none()) {
        // A choice reading no record (the CPU's, or a failed one): a zero
        // projection.
        for (uint channel = lane; channel < TILE; channel += 32)
            projected[channel] = 0.0f;
        return;
    }
    const auto in = activation(product + (m * SEISMIC_PRODUCT_STRIDE_0 + k * SEISMIC_PRODUCT_STRIDE_1) * Act::bytes, 0,
        SEISMIC_PRODUCT_STRIDE_2, uint(SEISMIC_DIM_F));
    const auto down = expert_matrix<packets::W0>(Experts{expert_down, tier_down}, source,
        KERNEL_W0_LAYOUT(SEISMIC_DIM_F), SEISMIC_DIM_F);
    // The down weight's operand row (the product row, or NVFP4 operands),
    // which this choice's simdgroup forms alone.
    const auto x = projection::operands<projection::OperandRecipe<packets::W0_OPERANDS_NVFP4>::NVFP4>::lone_row(
        in, uint(SEISMIC_DIM_F), lane);
    project_channels<packets::W0, ROWS, LANES, uint(SEISMIC_DIM_F), down_stage<LANES>()>(x, down, first,
        uint(SEISMIC_DIM_H), uint(SEISMIC_DIM_F), projected, lane);
}

// Gathered decode down projections: threadgroup `(group.x, u)` owns unique
// expert `u < worklist->count`, and all `GATHERED_SUMMED_SIMDGROUPS` (4)
// simdgroups cooperatively stage the expert's rows once into `tg` and split
// the threadgroup's `4 * TILE` output channels (`sg_first = group_first + sg * TILE`).
// Each choice's slot is published directly in A into `published`.
template <uint ROWS, uint LANES, bool ONE = false, typename Published>
inline void down_project_gathered(device const uchar *product, device const Worklist *worklist,
    device const ulong *expert_down, device const ulong *tier_down, uint u, uint group_first, uint tid, uint sg,
    uint lane, threadgroup GatheredSummedShared<LANES, ONE> &tg, Published published, constant ulong *seismic_words) {
    constexpr uint K = uint(SEISMIC_DIM_K);
    constexpr uint H = uint(SEISMIC_DIM_H);
    constexpr uint TILE = down_tile<ROWS, LANES>();
    const uint sg_first = group_first + sg * TILE;
    constexpr bool NVFP4 = projection::OperandRecipe<packets::W0_OPERANDS_NVFP4>::NVFP4;
    const uint idx = ONE ? u : (worklist->capacity - 1u - u);
    project_gathered_expert<packets::W0, ROWS, LANES, K, uint(SEISMIC_DIM_F), H, NVFP4, ONE>(
        product, SEISMIC_PRODUCT_STRIDE_0, SEISMIC_PRODUCT_STRIDE_1, SEISMIC_PRODUCT_STRIDE_2, worklist->entries[idx],
        expert_down, tier_down, KERNEL_W0_LAYOUT(SEISMIC_DIM_F), sg_first, tid, lane, tg, published);
}

} // namespace routed
