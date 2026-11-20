// Decode output (M <= VERIFY_ROWS): the fused projections (`lib/routed/output.h`)
// and their join (`lib/routed/output_sum.h`), without a CPU site. Gathered
// (GATHERED), each slot's projection is published in A (`published`) and the
// last threadgroup of a channel tile to arrive (`arrivals`) joins the tile's
// channels of every row.

#define KERNEL_A SEISMIC_ELEMENT_A
#if defined(SEISMIC_FORMING_ROUTED_OUTPUT_RESIDENT) || defined(SEISMIC_FORMING_ROUTED_OUTPUT_GATHERED_RESIDENT)
// A resident layer's: nothing lands in its records while it runs, so plain
// loads read them.
#define PACKETS_CONFIG_PLAIN_RECORDS
#endif
#define ROUTED_SOURCE_ROWS SEISMIC_DIM_XS
#include "lib/routed/output.h"
#include "lib/routed/output_sum.h"
#include "lib/core/arrive.h"

// The ports, the expert stack's: its records' address table.
#define ROUTED_OUTPUT_PORTS                                                        \
    device const float *residual [[buffer(SEISMIC_BUFFER_RESIDUAL)]],             \
    device const float *inject [[buffer(SEISMIC_BUFFER_INJECT)]],                 \
    device const uchar *expert_product [[buffer(SEISMIC_BUFFER_EXPERT_PRODUCT)]], \
    device const uchar *shared_product [[buffer(SEISMIC_BUFFER_SHARED_PRODUCT)]], \
    device const int *routes [[buffer(SEISMIC_BUFFER_ROUTES)]],                   \
    device const int *sources [[buffer(SEISMIC_BUFFER_SOURCES)]],                 \
    device const float *scores [[buffer(SEISMIC_BUFFER_SCORES)]],                 \
    device const float *coefficient [[buffer(SEISMIC_BUFFER_COEFFICIENT)]],       \
    device const ulong *expert_down [[buffer(SEISMIC_BUFFER_EXPERT_DOWN)]],       \
    device const ulong *tier_down [[buffer(SEISMIC_BUFFER_TIER_DOWN)]],           \
    device const uchar *shared_down [[buffer(SEISMIC_BUFFER_SHARED_DOWN)]],       \
    SEISMIC_RESULT_0_PARAM(value_cells),                                          \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],               \
    uint2 group [[threadgroup_position_in_grid]],                                 \
    uint tid [[thread_index_in_threadgroup]],                                     \
    uint sg [[simdgroup_index_in_threadgroup]],                                   \
    uint lane [[thread_index_in_simdgroup]]

// The body (`routed::output_project`).
#define ROUTED_OUTPUT_BODY                                                                                       \
    constexpr uint TILE = routed::output_tile<ROWS, LANES>();                                                    \
    constexpr uint REGION = routed::output_region<ROWS, LANES>();                                                \
    threadgroup float projected[uint(SEISMIC_DIM_K) + 1][REGION];                                                \
    const ulong m = group.y;                                                                                     \
    const uint first = group.x * TILE;                                                                           \
    const element::Out<ELEMENT_OF(SEISMIC_RESULT_0)> value(value_cells);                                         \
    routed::output_project<ROWS, LANES>(expert_product, shared_product, routes, sources, expert_down, tier_down, \
        shared_down, m, first, sg, lane, projected[sg], seismic_words);                                          \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                                             \
    const uint n = first + tid;                                                                                  \
    if (tid >= TILE || n >= uint(SEISMIC_DIM_H))                                                                 \
        return;                                                                                                  \
    routed::output_publish(residual, inject, scores, coefficient, value, m, n,                                   \
        routed::TileProjections{&projected[0][0], REGION, tid}, routed::Answer{false, 0.0f}, seismic_words);

// The gathered ports: the side-by-side form's, the published projections and
// the channel tiles' arrival counters.
#define ROUTED_OUTPUT_GATHERED_PORTS                                                   \
    ROUTED_OUTPUT_PORTS,                                                               \
    routed::PublishedSlots published [[buffer(SEISMIC_BUFFER_SCRATCH_PUBLISHED)]],    \
    device atomic_uint *arrivals [[buffer(SEISMIC_BUFFER_SCRATCH_ARRIVALS)]]

// The gathered body (`routed::output_project_gathered`): threadgroup (x, m)'s
// simdgroups publish their batched projections of tile x; the last of the
// tile's M threadgroups to arrive joins the tile's channels of every row.
#define ROUTED_OUTPUT_GATHERED_BODY                                                                              \
    constexpr uint TILE = routed::output_tile<ROWS, LANES>();                                                    \
    constexpr uint REGION = routed::output_region<ROWS, LANES>();                                                \
    constexpr uint SLOTS = uint(SEISMIC_DIM_K) + 1;                                                              \
    constexpr uint H = uint(SEISMIC_DIM_H);                                                                      \
    threadgroup float projected[SLOTS][REGION];                                                                  \
    threadgroup uint last;                                                                                       \
    const uint first = group.x * TILE;                                                                           \
    const element::Out<ELEMENT_OF(SEISMIC_RESULT_0)> value(value_cells);                                         \
    routed::output_project_gathered<ROWS, LANES>(expert_product, shared_product, routes, sources, expert_down,   \
        tier_down, shared_down, uint(SEISMIC_DIM_M), group.y, first, sg, lane, projected[sg], published,         \
        seismic_words);                                                                                          \
    if (!arrive::last(arrivals + group.x, uint(SEISMIC_DIM_M), &last, tid))                                      \
        return;                                                                                                  \
    for (uint item = tid; item < uint(SEISMIC_DIM_M) * TILE; item += 32 * SLOTS) {                               \
        const ulong m = item / TILE;                                                                             \
        const uint n = first + item % TILE;                                                                      \
        if (n < H)                                                                                               \
            routed::output_publish(residual, inject, scores, coefficient, value, m, n,                           \
                routed::ArrivedProjections<SLOTS, H>{published, m, n}, routed::Answer{false, 0.0f},              \
                seismic_words);                                                                                  \
    }

#ifdef SEISMIC_FORMING_ROUTED_OUTPUT_SUMMED
template <uint ROWS, uint LANES>
kernel void routed_output_summed(ROUTED_OUTPUT_PORTS) {
    ROUTED_OUTPUT_BODY
}
#endif
#ifdef SEISMIC_FORMING_ROUTED_OUTPUT_RESIDENT
// A resident layer's (XS 0), its records read with plain loads.
template <uint ROWS, uint LANES>
kernel void routed_output_resident(ROUTED_OUTPUT_PORTS) {
    ROUTED_OUTPUT_BODY
}
#endif
#ifdef SEISMIC_FORMING_ROUTED_OUTPUT_GATHERED
template <uint ROWS, uint LANES>
kernel void routed_output_gathered(ROUTED_OUTPUT_GATHERED_PORTS) {
    ROUTED_OUTPUT_GATHERED_BODY
}
#endif
#ifdef SEISMIC_FORMING_ROUTED_OUTPUT_GATHERED_RESIDENT
// A resident layer's (XS 0), its records read with plain loads.
template <uint ROWS, uint LANES>
kernel void routed_output_gathered_resident(ROUTED_OUTPUT_GATHERED_PORTS) {
    ROUTED_OUTPUT_GATHERED_BODY
}
#endif
