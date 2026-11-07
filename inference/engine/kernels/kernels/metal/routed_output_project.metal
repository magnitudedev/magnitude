// The device's share of a streamed layer of Qwen's fused decode output
// (`routed_output_project` in routed.seismic): `routed_output`'s projections
// (`lib/routed/output.h`), each channel's thread then publishing every
// choice's projection and the shared expert's (slot K), in A, for `routed_output_sited`.

#define ROUTED_SOURCE_ROWS SEISMIC_CONST_SOURCE_ROWS
#include "lib/routed/output.h"

#ifdef SEISMIC_FORMING_ROUTED_OUTPUT_PROJECT
template <uint ROWS, uint LANES>
kernel void routed_output_project(
    device const uchar *expert_product [[buffer(SEISMIC_BUFFER_EXPERT_PRODUCT)]],
    device const uchar *shared_product [[buffer(SEISMIC_BUFFER_SHARED_PRODUCT)]],
    device const int *routes [[buffer(SEISMIC_BUFFER_ROUTES)]],
    device const int *sources [[buffer(SEISMIC_BUFFER_SOURCES)]],
    device const ulong *expert_down [[buffer(SEISMIC_BUFFER_EXPERT_DOWN)]],
    device const ulong *tier_down [[buffer(SEISMIC_BUFFER_TIER_DOWN)]],
    device const uchar *shared_down [[buffer(SEISMIC_BUFFER_SHARED_DOWN)]],
    device uchar *published [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint TILE = routed::output_tile<ROWS, LANES>();
    threadgroup float projected[uint(SEISMIC_DIM_K) + 1][TILE];
    threadgroup uint4 staged[uint(SEISMIC_DIM_K) + 1][4 * routed::output_stage()];
    const ulong m = group.y;
    const uint first = group.x * TILE;
    routed::output_project<ROWS, LANES>(expert_product, shared_product, routes, sources, expert_down, tier_down,
        shared_down, m, first, sg, lane, staged[sg], projected[sg], seismic_words);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint n = first + tid;
    if (tid >= TILE || n >= uint(SEISMIC_DIM_H))
        return;
    for (uint k = 0; k <= uint(SEISMIC_DIM_K); ++k)
        reinterpret_cast<device typename routed::Act::storage *>(published)[m * SEISMIC_RESULT_0_STRIDE_0
            + ulong(k) * SEISMIC_RESULT_0_STRIDE_1 + ulong(n) * SEISMIC_RESULT_0_STRIDE_2] = routed::Act::store(projected[k][tid]);
}
#endif
