// The device's share of a streamed layer's general decode combine
// (`routed_down_project` in routed.seismic): `routed_down`'s side-by-side and
// gathered projections (`lib/routed/down.h`), published in A into result 0
// for `routed_down_sited`.

#define KERNEL_A SEISMIC_ELEMENT_A
#define ROUTED_SOURCE_ROWS SEISMIC_CONST_SOURCE_ROWS
#include "lib/streaming/control.h"
#include "lib/routed/down.h"

inline bool down_project_active(int mode, uint c, slab::Source src, device const int *step,
    SYSTEM const int *routing, constant ulong *seismic_words) {
    if (mode == 2)
        return !src.none();
    if (mode == 3)
        return !src.none()
            && streaming_left_absent(step, SEISMIC_STEP_STRIDE_0, routing, SEISMIC_ROUTING_STRIDE_0,
                SEISMIC_ROUTING_STRIDE_1, uint(SEISMIC_DIM_FLIGHT), uint(SEISMIC_DIM_P), c);
    return true;
}

#define ROUTED_DOWN_PROJECT_PORTS                                               \
    device const uchar *product [[buffer(SEISMIC_BUFFER_PRODUCT)]],             \
    device const int *routes [[buffer(SEISMIC_BUFFER_ROUTES)]],                 \
    device const int *sources [[buffer(SEISMIC_BUFFER_SOURCES)]],               \
    device const ulong *expert_down [[buffer(SEISMIC_BUFFER_EXPERT_DOWN)]],     \
    device const ulong *tier_down [[buffer(SEISMIC_BUFFER_TIER_DOWN)]],         \
    SYSTEM const int *routing [[buffer(SEISMIC_BUFFER_ROUTING)]],               \
    device const int *step [[buffer(SEISMIC_BUFFER_STEP)]],                     \
    SEISMIC_OUT_PUBLISHED_PARAM(published_cells),                               \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],             \
    uint2 group [[threadgroup_position_in_grid]],                               \
    uint tid [[thread_index_in_threadgroup]],                                   \
    uint sg [[simdgroup_index_in_threadgroup]],                                 \
    uint lane [[thread_index_in_simdgroup]]

#ifdef SEISMIC_FORMING_ROUTED_DOWN_PROJECT_SUMMED
template <uint ROWS, uint LANES>
kernel void routed_down_project_summed(ROUTED_DOWN_PROJECT_PORTS) {
    constexpr uint TILE = routed::down_tile<ROWS, LANES>();
    const element::Out<ELEMENT_OF(SEISMIC_ELEMENT_A)> published(published_cells);
    threadgroup float projected[uint(SEISMIC_DIM_K)][routed::down_region<ROWS, LANES>()];
    threadgroup uchar active_mask[uint(SEISMIC_DIM_K)];
    const ulong m = group.y;
    const uint first = group.x * TILE;
    const int mode = int(SEISMIC_PARAM_MODE);
    const slab::Source src = routed::decode_source(sources, ROUTED_SOURCE_ROWS, SEISMIC_SOURCES_STRIDE_1,
        SEISMIC_SOURCES_STRIDE_2, m, sg, routes[m * SEISMIC_ROUTES_STRIDE_0 + ulong(sg) * SEISMIC_ROUTES_STRIDE_1]);
    int act_i = 0;
    if (lane == 0u) {
        act_i = down_project_active(mode, uint(m * SEISMIC_DIM_K + ulong(sg)), src, step, routing, seismic_words) ? 1 : 0;
        active_mask[sg] = uchar(act_i);
    }
    const bool act = metal::simd_broadcast_first(act_i) != 0;
    if (act) {
        if (mode == 1 && src.code < 0) {
            for (uint channel = lane; channel < TILE; channel += 32u)
                projected[sg][channel] = 0.0f;
        } else {
            routed::down_project<ROWS, LANES>(product, routes, sources, expert_down, tier_down, m, first, sg, lane,
                projected[sg], seismic_words);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint n = first + tid;
    if (tid >= TILE || n >= uint(SEISMIC_DIM_H))
        return;
    for (uint k = 0; k < uint(SEISMIC_DIM_K); ++k) {
        if (active_mask[k])
            published.put(m * SEISMIC_PUBLISHED_STRIDE_0 + ulong(k) * SEISMIC_PUBLISHED_STRIDE_1
                + ulong(n) * SEISMIC_PUBLISHED_STRIDE_2, projected[k][tid]);
    }
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_DOWN_PROJECT_GATHER
template <uint ROWS, uint LANES>
kernel void routed_down_project_gather(
    device const int *routes [[buffer(SEISMIC_BUFFER_ROUTES)]],
    device const int *sources [[buffer(SEISMIC_BUFFER_SOURCES)]],
    SYSTEM const int *routing [[buffer(SEISMIC_BUFFER_ROUTING)]],
    device const int *step [[buffer(SEISMIC_BUFFER_STEP)]],
    device routed::Worklist *worklist [[buffer(SEISMIC_BUFFER_SCRATCH_WORKLIST)]],
    device int *gather_grid [[buffer(SEISMIC_BUFFER_SCRATCH_GATHER_GRID)]],
    device int *gather_one_grid [[buffer(SEISMIC_BUFFER_SCRATCH_GATHER_ONE_GRID)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup int cache[routed::WORKLIST_CACHE_WORDS];
    constexpr uint GROUP_TILE = routed::GATHERED_SUMMED_SIMDGROUPS * routed::down_tile<ROWS, LANES>();
    const uint x_groups = (uint(SEISMIC_DIM_H) + GROUP_TILE - 1u) / GROUP_TILE;
    const int mode = int(SEISMIC_PARAM_MODE);
    routed::build_worklist(routes, SEISMIC_ROUTES_STRIDE_0, SEISMIC_ROUTES_STRIDE_1, sources, ROUTED_SOURCE_ROWS,
        SEISMIC_SOURCES_STRIDE_1, SEISMIC_SOURCES_STRIDE_2, uint(SEISMIC_DIM_M), uint(SEISMIC_DIM_K), x_groups, 0u,
        worklist, gather_grid, gather_one_grid, cache, lane,
        [&](uint c, slab::Source src) {
            return down_project_active(mode, c, src, step, routing, seismic_words);
        },
        mode == 1);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_DOWN_PROJECT_GATHERED_ONE
template <uint ROWS, uint LANES>
kernel void routed_down_project_gathered_one(
    ROUTED_DOWN_PROJECT_PORTS,
    device const routed::Worklist *worklist [[buffer(SEISMIC_BUFFER_SCRATCH_WORKLIST)]]) {
    constexpr uint TILE = routed::down_tile<ROWS, LANES>();
    constexpr uint SLOTS = uint(SEISMIC_DIM_K);
    threadgroup routed::GatheredSummedShared<LANES, true, false> tg;
    const uint first = group.x * (routed::GATHERED_SUMMED_SIMDGROUPS * TILE);
    const routed::OutPublishedSlots<SLOTS> published{
        element::Out<ELEMENT_OF(SEISMIC_ELEMENT_A)>(published_cells),
        SEISMIC_PUBLISHED_STRIDE_0, SEISMIC_PUBLISHED_STRIDE_1, SEISMIC_PUBLISHED_STRIDE_2};
    routed::down_project_gathered<ROWS, LANES, true, false>(product, worklist, expert_down, tier_down, group.x,
        group.y, first, tid, sg, lane, tg, nullptr, published, seismic_words);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_DOWN_PROJECT_GATHERED
template <uint ROWS, uint LANES, uint MATRIX>
kernel void routed_down_project_gathered(
    ROUTED_DOWN_PROJECT_PORTS,
    device const routed::Worklist *worklist [[buffer(SEISMIC_BUFFER_SCRATCH_WORKLIST)]],
    threadgroup uchar *shared [[threadgroup(0)]]) {
    constexpr uint TILE = routed::down_tile<ROWS, LANES>();
    constexpr uint SLOTS = uint(SEISMIC_DIM_K);
    threadgroup routed::GatheredSummedShared<LANES, false, (MATRIX != 0)> tg;
    const uint first = group.x * (routed::GATHERED_SUMMED_SIMDGROUPS * TILE);
    const routed::OutPublishedSlots<SLOTS> published{
        element::Out<ELEMENT_OF(SEISMIC_ELEMENT_A)>(published_cells),
        SEISMIC_PUBLISHED_STRIDE_0, SEISMIC_PUBLISHED_STRIDE_1, SEISMIC_PUBLISHED_STRIDE_2};
    routed::down_project_gathered<ROWS, LANES, false, (MATRIX != 0)>(product, worklist, expert_down, tier_down,
        group.x, group.y, first, tid, sg, lane, tg, shared, published, seismic_words);
}
#endif
