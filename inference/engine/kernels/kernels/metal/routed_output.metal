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
    const uint c = tid * 4u;                                                                                     \
    const uint n = first + c;                                                                                    \
    if (c >= TILE || n >= uint(SEISMIC_DIM_H))                                                                   \
        return;                                                                                                  \
    if (c + 4u <= TILE && n + 4u <= uint(SEISMIC_DIM_H)) {                                                       \
        routed::output_publish4(residual, inject, scores, coefficient, value, m, n,                              \
            routed::TileProjections{&projected[0][0], REGION, c}, routed::Answer4{false, float4(0.0f)},          \
            seismic_words);                                                                                      \
    } else {                                                                                                     \
        const uint end = metal::min(TILE, uint(SEISMIC_DIM_H) - first);                                          \
        for (uint i = c; i < end; ++i)                                                                           \
            routed::output_publish(residual, inject, scores, coefficient, value, m, first + i,                   \
                routed::TileProjections{&projected[0][0], REGION, i}, routed::Answer{false, 0.0f},               \
                seismic_words);                                                                                  \
    }

// The gathered ports: the side-by-side form's, the compact worklist, the
// published projections and the channel tiles' arrival counters.
#define ROUTED_OUTPUT_GATHERED_PORTS                                                   \
    ROUTED_OUTPUT_PORTS,                                                               \
    device const routed::Worklist *worklist [[buffer(SEISMIC_BUFFER_SCRATCH_WORKLIST)]], \
    routed::PublishedSlots published [[buffer(SEISMIC_BUFFER_SCRATCH_PUBLISHED)]],    \
    device atomic_uint *arrivals [[buffer(SEISMIC_BUFFER_SCRATCH_ARRIVALS)]]

// The gathered body (`routed::output_project_gathered`): threadgroup (x, u)'s
// `GATHERED_SUMMED_SIMDGROUPS` (4) simdgroups cooperatively stage the expert's
// rows and split the tile's `4 * TILE` channels for unique expert `u` (or the
// shared expert at `u == worklist->count`); the last of the tile's
// `worklist->count + 1` threadgroups to arrive joins the tile's channels of
// every row, four contiguous channels per thread with a scalar tail.
#define ROUTED_OUTPUT_GATHERED_BODY                                                                              \
    constexpr uint TILE = routed::output_tile<ROWS, LANES>();                                                    \
    constexpr uint SLOTS = uint(SEISMIC_DIM_K) + 1;                                                              \
    constexpr uint SG_COUNT = routed::GATHERED_SUMMED_SIMDGROUPS;                                                \
    constexpr uint GROUP_TILE = SG_COUNT * TILE;                                                                 \
    constexpr uint H = uint(SEISMIC_DIM_H);                                                                      \
    threadgroup routed::GatheredSummedShared<LANES> tg;                                                          \
    threadgroup uint last;                                                                                       \
    const uint first = group.x * GROUP_TILE;                                                                     \
    const element::Out<ELEMENT_OF(SEISMIC_RESULT_0)> value(value_cells);                                         \
    routed::output_project_gathered<ROWS, LANES>(expert_product, shared_product, worklist, expert_down,          \
        tier_down, shared_down, uint(SEISMIC_DIM_M), group.y, first, tid, sg, lane, tg, published,               \
        seismic_words);                                                                                          \
    if (!arrive::last(arrivals + group.x, worklist->count + 1u, &last, tid))                                     \
        return;                                                                                                  \
    const uint rows = uint(SEISMIC_DIM_M);                                                                       \
    if ((GROUP_TILE & 3u) == 0u && GROUP_TILE != 0u && first + GROUP_TILE <= H) {                                \
        constexpr uint QUADS = (GROUP_TILE >= 4u && (GROUP_TILE & 3u) == 0u) ? (GROUP_TILE >> 2u) : 1u;          \
        for (uint item = tid; item < rows * QUADS; item += 32u * SG_COUNT) {                                     \
            const ulong m = item / QUADS;                                                                        \
            const uint n = first + (item % QUADS) * 4u;                                                          \
            routed::output_publish4(residual, inject, scores, coefficient, value, m, n,                          \
                routed::ArrivedProjections<SLOTS, H>{published, m, n}, routed::Answer4{false, float4(0.0f)},     \
                seismic_words);                                                                                  \
        }                                                                                                        \
    } else if (first < H) {                                                                                      \
        const uint span = metal::min(GROUP_TILE, H - first);                                                     \
        const uint quads = span >> 2u;                                                                           \
        if (quads != 0u) {                                                                                       \
            for (uint item = tid; item < rows * quads; item += 32u * SG_COUNT) {                                 \
                const ulong m = item / quads;                                                                    \
                const uint n = first + (item - uint(m) * quads) * 4u;                                            \
                routed::output_publish4(residual, inject, scores, coefficient, value, m, n,                      \
                    routed::ArrivedProjections<SLOTS, H>{published, m, n},                                       \
                    routed::Answer4{false, float4(0.0f)}, seismic_words);                                        \
            }                                                                                                    \
        }                                                                                                        \
        const uint tail = span & 3u;                                                                             \
        if (tail != 0u) {                                                                                        \
            const uint tail_base = first + (quads << 2u);                                                        \
            for (uint item = tid; item < rows * tail; item += 32u * SG_COUNT) {                                  \
                const ulong m = item / tail;                                                                     \
                const uint n = tail_base + (item - uint(m) * tail);                                              \
                routed::output_publish(residual, inject, scores, coefficient, value, m, n,                       \
                    routed::ArrivedProjections<SLOTS, H>{published, m, n}, routed::Answer{false, 0.0f},          \
                    seismic_words);                                                                              \
            }                                                                                                    \
        }                                                                                                        \
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
#ifdef SEISMIC_FORMING_ROUTED_OUTPUT_GATHER
template <uint ROWS, uint LANES>
kernel void routed_output_gather(
    device const int *routes [[buffer(SEISMIC_BUFFER_ROUTES)]],
    device const int *sources [[buffer(SEISMIC_BUFFER_SOURCES)]],
    device routed::Worklist *worklist [[buffer(SEISMIC_BUFFER_SCRATCH_WORKLIST)]],
    device int *gather_grid [[buffer(SEISMIC_BUFFER_SCRATCH_GATHER_GRID)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup int cache[2u * routed::WORKLIST_CACHE];
    constexpr uint GROUP_TILE = routed::GATHERED_SUMMED_SIMDGROUPS * routed::output_tile<ROWS, LANES>();
    const uint x_groups = (uint(SEISMIC_DIM_H) + GROUP_TILE - 1u) / GROUP_TILE;
    routed::build_worklist(routes, SEISMIC_ROUTES_STRIDE_0, SEISMIC_ROUTES_STRIDE_1, sources, ROUTED_SOURCE_ROWS,
        SEISMIC_SOURCES_STRIDE_1, SEISMIC_SOURCES_STRIDE_2, uint(SEISMIC_DIM_M), uint(SEISMIC_DIM_K), x_groups, 1u,
        worklist, gather_grid, cache, lane);
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
