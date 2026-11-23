// The device's share of a streamed layer's general decode combine
// (`routed_down_project` in routed.seismic): `routed_down`'s side-by-side and
// gathered projections (`lib/routed/down.h`), published in A into result 0
// for `routed_down_sited`.

#define KERNEL_A SEISMIC_ELEMENT_A
#define ROUTED_SOURCE_ROWS SEISMIC_CONST_SOURCE_ROWS
#include "lib/routed/down.h"

#define ROUTED_DOWN_PROJECT_PORTS                                               \
    device const uchar *product [[buffer(SEISMIC_BUFFER_PRODUCT)]],             \
    device const int *routes [[buffer(SEISMIC_BUFFER_ROUTES)]],                 \
    device const int *sources [[buffer(SEISMIC_BUFFER_SOURCES)]],               \
    device const ulong *expert_down [[buffer(SEISMIC_BUFFER_EXPERT_DOWN)]],     \
    device const ulong *tier_down [[buffer(SEISMIC_BUFFER_TIER_DOWN)]],         \
    SEISMIC_RESULT_0_PARAM(published_cells),                                    \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],             \
    uint2 group [[threadgroup_position_in_grid]],                               \
    uint tid [[thread_index_in_threadgroup]],                                   \
    uint sg [[simdgroup_index_in_threadgroup]],                                 \
    uint lane [[thread_index_in_simdgroup]]

#ifdef SEISMIC_FORMING_ROUTED_DOWN_PROJECT_SUMMED
template <uint ROWS, uint LANES>
kernel void routed_down_project_summed(ROUTED_DOWN_PROJECT_PORTS) {
    constexpr uint TILE = routed::down_tile<ROWS, LANES>();
    const element::Out<ELEMENT_OF(SEISMIC_RESULT_0)> published(published_cells);
    threadgroup float projected[uint(SEISMIC_DIM_K)][routed::down_region<ROWS, LANES>()];
    const ulong m = group.y;
    const uint first = group.x * TILE;
    routed::down_project<ROWS, LANES>(product, routes, sources, expert_down, tier_down, m, first, sg, lane,
        projected[sg], seismic_words);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint n = first + tid;
    if (tid >= TILE || n >= uint(SEISMIC_DIM_H))
        return;
    for (uint k = 0; k < uint(SEISMIC_DIM_K); ++k)
        published.put(m * SEISMIC_RESULT_0_STRIDE_0 + ulong(k) * SEISMIC_RESULT_0_STRIDE_1
            + ulong(n) * SEISMIC_RESULT_0_STRIDE_2, projected[k][tid]);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_DOWN_PROJECT_GATHER
template <uint ROWS, uint LANES>
kernel void routed_down_project_gather(
    device const int *routes [[buffer(SEISMIC_BUFFER_ROUTES)]],
    device const int *sources [[buffer(SEISMIC_BUFFER_SOURCES)]],
    device routed::Worklist *worklist [[buffer(SEISMIC_BUFFER_SCRATCH_WORKLIST)]],
    device int *gather_grid [[buffer(SEISMIC_BUFFER_SCRATCH_GATHER_GRID)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup int cache[2u * routed::WORKLIST_CACHE];
    constexpr uint GROUP_TILE = routed::GATHERED_SUMMED_SIMDGROUPS * routed::down_tile<ROWS, LANES>();
    const uint x_groups = (uint(SEISMIC_DIM_H) + GROUP_TILE - 1u) / GROUP_TILE;
    routed::build_worklist(routes, SEISMIC_ROUTES_STRIDE_0, SEISMIC_ROUTES_STRIDE_1, sources, ROUTED_SOURCE_ROWS,
        SEISMIC_SOURCES_STRIDE_1, SEISMIC_SOURCES_STRIDE_2, uint(SEISMIC_DIM_M), uint(SEISMIC_DIM_K), x_groups, 0u,
        worklist, gather_grid, cache, lane);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_DOWN_PROJECT_GATHERED
template <uint ROWS, uint LANES>
kernel void routed_down_project_gathered(
    ROUTED_DOWN_PROJECT_PORTS,
    device const routed::Worklist *worklist [[buffer(SEISMIC_BUFFER_SCRATCH_WORKLIST)]]) {
    constexpr uint TILE = routed::down_tile<ROWS, LANES>();
    constexpr uint SLOTS = uint(SEISMIC_DIM_K);
    threadgroup routed::GatheredSummedShared<LANES> tg;
    const uint first = group.x * (routed::GATHERED_SUMMED_SIMDGROUPS * TILE);
    const routed::OutPublishedSlots<SLOTS> published{
        element::Out<ELEMENT_OF(SEISMIC_RESULT_0)>(published_cells),
        SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1, SEISMIC_RESULT_0_STRIDE_2};
    routed::down_project_gathered<ROWS, LANES>(product, worklist, expert_down, tier_down, group.y, first, tid, sg,
        lane, tg, published, seismic_words);
}
#endif
