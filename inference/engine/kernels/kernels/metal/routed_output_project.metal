// The device's share of a streamed layer of Qwen's fused decode output
// (`routed_output_project` in routed.seismic): `routed_output`'s side-by-side
// and gathered projections (`lib/routed/output.h`), published in A into
// result 0 for `routed_output_sited`.

#define KERNEL_A SEISMIC_ELEMENT_A
#define ROUTED_SOURCE_ROWS SEISMIC_CONST_SOURCE_ROWS
#include "lib/routed/output.h"

#define ROUTED_OUTPUT_PROJECT_PORTS                                                 \
    device const uchar *expert_product [[buffer(SEISMIC_BUFFER_EXPERT_PRODUCT)]],   \
    device const uchar *shared_product [[buffer(SEISMIC_BUFFER_SHARED_PRODUCT)]],   \
    device const int *routes [[buffer(SEISMIC_BUFFER_ROUTES)]],                     \
    device const int *sources [[buffer(SEISMIC_BUFFER_SOURCES)]],                   \
    device const ulong *expert_down [[buffer(SEISMIC_BUFFER_EXPERT_DOWN)]],         \
    device const ulong *tier_down [[buffer(SEISMIC_BUFFER_TIER_DOWN)]],             \
    device const uchar *shared_down [[buffer(SEISMIC_BUFFER_SHARED_DOWN)]],         \
    SEISMIC_RESULT_0_PARAM(published_cells),                                        \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],                 \
    uint2 group [[threadgroup_position_in_grid]],                                   \
    uint tid [[thread_index_in_threadgroup]],                                       \
    uint sg [[simdgroup_index_in_threadgroup]],                                     \
    uint lane [[thread_index_in_simdgroup]]

#ifdef SEISMIC_FORMING_ROUTED_OUTPUT_PROJECT_SUMMED
template <uint ROWS, uint LANES>
kernel void routed_output_project_summed(ROUTED_OUTPUT_PROJECT_PORTS) {
    constexpr uint TILE = routed::output_tile<ROWS, LANES>();
    const element::Out<ELEMENT_OF(SEISMIC_RESULT_0)> published(published_cells);
    threadgroup float projected[uint(SEISMIC_DIM_K) + 1][routed::output_region<ROWS, LANES>()];
    const ulong m = group.y;
    const uint first = group.x * TILE;
    routed::output_project<ROWS, LANES>(expert_product, shared_product, routes, sources, expert_down, tier_down,
        shared_down, m, first, sg, lane, projected[sg], seismic_words);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint n = first + tid;
    if (tid >= TILE || n >= uint(SEISMIC_DIM_H))
        return;
    for (uint k = 0; k <= uint(SEISMIC_DIM_K); ++k)
        published.put(m * SEISMIC_RESULT_0_STRIDE_0 + ulong(k) * SEISMIC_RESULT_0_STRIDE_1
            + ulong(n) * SEISMIC_RESULT_0_STRIDE_2, projected[k][tid]);
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_OUTPUT_PROJECT_GATHERED
template <uint ROWS, uint LANES>
kernel void routed_output_project_gathered(ROUTED_OUTPUT_PROJECT_PORTS) {
    constexpr uint TILE = routed::output_tile<ROWS, LANES>();
    constexpr uint SLOTS = uint(SEISMIC_DIM_K) + 1;
    threadgroup float projected[SLOTS][routed::output_region<ROWS, LANES>()];
    const uint first = group.x * TILE;
    const routed::OutPublishedSlots<SLOTS> published{
        element::Out<ELEMENT_OF(SEISMIC_RESULT_0)>(published_cells),
        SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1, SEISMIC_RESULT_0_STRIDE_2};
    routed::output_project_gathered<ROWS, LANES>(expert_product, shared_product, routes, sources, expert_down,
        tier_down, shared_down, uint(SEISMIC_DIM_M), group.y, first, sg, lane, projected[sg], published,
        seismic_words);
}
#endif
