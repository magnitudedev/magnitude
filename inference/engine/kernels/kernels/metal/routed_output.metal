// Decode output (M <= DECODE_ROWS): the fused projections (`lib/routed/output.h`)
// and their join (`lib/routed/output_sum.h`), without a CPU site.

#define ROUTED_SOURCE_ROWS SEISMIC_DIM_XS
#include "lib/routed/output.h"
#include "lib/routed/output_sum.h"

// The ports, the expert stack's of element type STACK: its records' address
// table (`ulong`), or a resident layer's stack (`uchar`).
#define ROUTED_OUTPUT_PORTS(STACK)                                                \
    device const float *residual [[buffer(SEISMIC_BUFFER_RESIDUAL)]],             \
    device const float *inject [[buffer(SEISMIC_BUFFER_INJECT)]],                 \
    device const uchar *expert_product [[buffer(SEISMIC_BUFFER_EXPERT_PRODUCT)]], \
    device const uchar *shared_product [[buffer(SEISMIC_BUFFER_SHARED_PRODUCT)]], \
    device const int *routes [[buffer(SEISMIC_BUFFER_ROUTES)]],                   \
    device const int *sources [[buffer(SEISMIC_BUFFER_SOURCES)]],                 \
    device const float *scores [[buffer(SEISMIC_BUFFER_SCORES)]],                 \
    device const float *coefficient [[buffer(SEISMIC_BUFFER_COEFFICIENT)]],       \
    device const STACK *expert_down [[buffer(SEISMIC_BUFFER_EXPERT_DOWN)]],       \
    device const ulong *tier_down [[buffer(SEISMIC_BUFFER_TIER_DOWN)]],           \
    device const uchar *shared_down [[buffer(SEISMIC_BUFFER_SHARED_DOWN)]],       \
    device float *value [[buffer(SEISMIC_RESULT_0_BUFFER)]],                      \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],               \
    uint2 group [[threadgroup_position_in_grid]],                                 \
    uint tid [[thread_index_in_threadgroup]],                                     \
    uint sg [[simdgroup_index_in_threadgroup]],                                   \
    uint lane [[thread_index_in_simdgroup]]

// The body, of either binding of the stack (`routed::output_project`).
#define ROUTED_OUTPUT_BODY                                                                                       \
    constexpr uint TILE = routed::output_tile<ROWS, LANES>();                                                    \
    constexpr uint REGION = routed::output_region<ROWS, LANES>();                                                \
    threadgroup float projected[uint(SEISMIC_DIM_K) + 1][REGION];                                                \
    const ulong m = group.y;                                                                                     \
    const uint first = group.x * TILE;                                                                           \
    routed::output_project<ROWS, LANES>(expert_product, shared_product, routes, sources, expert_down, tier_down, \
        shared_down, m, first, sg, lane, projected[sg], seismic_words);                                          \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                                             \
    const uint n = first + tid;                                                                                  \
    if (tid >= TILE || n >= uint(SEISMIC_DIM_H))                                                                 \
        return;                                                                                                  \
    routed::output_publish(residual, inject, scores, coefficient, value, m, n,                                   \
        routed::TileProjections{&projected[0][0], REGION, tid}, routed::Answer{false, 0.0f}, seismic_words);

#ifdef SEISMIC_FORMING_ROUTED_OUTPUT
template <uint ROWS, uint LANES>
kernel void routed_output(ROUTED_OUTPUT_PORTS(ulong)) {
    ROUTED_OUTPUT_BODY
}
#endif
#ifdef SEISMIC_FORMING_ROUTED_OUTPUT_RESIDENT
// A resident layer's (XS 0): its stack bound as it lies, read with plain
// loads, as nothing lands in it while it runs.
template <uint ROWS, uint LANES>
kernel void routed_output_resident(ROUTED_OUTPUT_PORTS(uchar)) {
    ROUTED_OUTPUT_BODY
}
#endif
