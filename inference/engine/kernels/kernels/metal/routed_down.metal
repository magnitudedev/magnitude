// Decode output of the general routed feed-forward (M <= DECODE_ROWS): the
// down projections (`lib/routed/down.h`) and their sum
// (`lib/routed/down_sum.h`), without a CPU site.

#ifdef SEISMIC_FORMING_ROUTED_DOWN_WHOLE
// A whole layer's: nothing lands in its records while it runs, so plain
// loads read them (8% of the launch at Gemma 26B's shape).
#define PACKETS_PLAIN_RECORDS
#endif
#define ROUTED_SOURCE_ROWS SEISMIC_DIM_XS
// Staged at 16 lanes too, past 2 packets a lane (`routed::stage_packets`).
#define ROUTED_DOWN_STAGES_AT_16
#include "lib/routed/down.h"
#include "lib/routed/down_sum.h"

// The ports, the expert stack's of element type STACK: its records' address
// table (`ulong`), or a resident layer's stack (`uchar`).
#define ROUTED_DOWN_PORTS(STACK)                                                                                  \
    device const float *base [[buffer(SEISMIC_BUFFER_BASE)]],                                                     \
        device const uchar *product [[buffer(SEISMIC_BUFFER_PRODUCT)]],                                           \
        device const int *routes [[buffer(SEISMIC_BUFFER_ROUTES)]],                                               \
        device const int *sources [[buffer(SEISMIC_BUFFER_SOURCES)]],                                             \
        device const float *weights [[buffer(SEISMIC_BUFFER_WEIGHTS)]],                                           \
        device const STACK *expert_down [[buffer(SEISMIC_BUFFER_EXPERT_DOWN)]],                                   \
        device const ulong *tier_down [[buffer(SEISMIC_BUFFER_TIER_DOWN)]],                                       \
        device uchar *value [[buffer(SEISMIC_RESULT_0_BUFFER)]],                                                  \
        constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],                                           \
        uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],                   \
        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]

// The body, of either binding of the stack (`routed::down_project`).
#define ROUTED_DOWN_BODY                                                                                          \
    constexpr uint TILE = routed::down_tile<ROWS, LANES>();                                                       \
    constexpr uint REGION = routed::down_region<ROWS, LANES>();                                                   \
    threadgroup float projected[uint(SEISMIC_DIM_K)][REGION];                                                     \
    const ulong m = group.y;                                                                                      \
    const uint first = group.x * TILE;                                                                            \
    routed::down_project<ROWS, LANES>(product, routes, sources, expert_down, tier_down, m, first, sg, lane,       \
        projected[sg], seismic_words);                                                                            \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                                              \
    const uint n = first + tid;                                                                                   \
    if (tid >= TILE || n >= uint(SEISMIC_DIM_H))                                                                  \
        return;                                                                                                   \
    routed::down_publish(base, weights, value, m, n, routed::TileProjections{&projected[0][0], REGION, tid},      \
        routed::Answer{false, 0.0f}, seismic_words);

#ifdef SEISMIC_FORMING_ROUTED_DOWN
template <uint ROWS, uint LANES>
kernel void routed_down(ROUTED_DOWN_PORTS(ulong)) {
    ROUTED_DOWN_BODY
}
#endif
#ifdef SEISMIC_FORMING_ROUTED_DOWN_WHOLE
template <uint ROWS, uint LANES>
kernel void routed_down_whole(ROUTED_DOWN_PORTS(ulong)) {
    ROUTED_DOWN_BODY
}
#endif
#ifdef SEISMIC_FORMING_ROUTED_DOWN_RESIDENT
// A resident layer's (XS 0): its stack bound as it lies, read with plain
// loads, as nothing lands in it while it runs.
template <uint ROWS, uint LANES>
kernel void routed_down_resident(ROUTED_DOWN_PORTS(uchar)) {
    ROUTED_DOWN_BODY
}
#endif
