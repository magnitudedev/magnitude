// The device's share of a streamed layer's general decode combine
// (`routed_down_project` in routed.seismic): `routed_down`'s projections
// (`lib/routed/down.h`), each channel's thread then publishing every
// choice's projection in A, for `routed_down_sited`.

#define ROUTED_SOURCE_ROWS SEISMIC_CONST_SOURCE_ROWS
#include "lib/routed/down.h"

#ifdef SEISMIC_FORMING_ROUTED_DOWN_PROJECT
template <uint ROWS, uint LANES>
kernel void routed_down_project(
    device const uchar *product [[buffer(SEISMIC_BUFFER_PRODUCT)]],
    device const int *routes [[buffer(SEISMIC_BUFFER_ROUTES)]],
    device const int *sources [[buffer(SEISMIC_BUFFER_SOURCES)]],
    device const ulong *expert_down [[buffer(SEISMIC_BUFFER_EXPERT_DOWN)]],
    device const ulong *tier_down [[buffer(SEISMIC_BUFFER_TIER_DOWN)]],
    device uchar *published [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint TILE = routed::down_tile<ROWS, LANES>();
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
        reinterpret_cast<device typename routed::Act::storage *>(published)[m * SEISMIC_RESULT_0_STRIDE_0
            + ulong(k) * SEISMIC_RESULT_0_STRIDE_1 + ulong(n) * SEISMIC_RESULT_0_STRIDE_2] = routed::Act::store(projected[k][tid]);
}
#endif
