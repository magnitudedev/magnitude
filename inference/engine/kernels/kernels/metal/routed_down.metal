// Decode output of the general routed feed-forward (M <= VERIFY_ROWS): the
// down projections (`lib/routed/down.h`) and their sum
// (`lib/routed/down_sum.h`), without a CPU site. Gathered (GATHERED), each
// choice's projection is published in A (`published`) and the last
// threadgroup of a channel tile to arrive (`arrivals`) sums the tile's
// channels of every row.

#define KERNEL_A SEISMIC_ELEMENT_A
#if defined(SEISMIC_FORMING_ROUTED_DOWN_WHOLE) || defined(SEISMIC_FORMING_ROUTED_DOWN_RESIDENT)                   \
    || defined(SEISMIC_FORMING_ROUTED_DOWN_GATHERED_WHOLE) || defined(SEISMIC_FORMING_ROUTED_DOWN_GATHERED_RESIDENT)
// A whole or a resident layer's: nothing lands in its records while it runs,
// so plain loads read them (8% of the launch at Gemma 26B's shape).
#define PACKETS_CONFIG_PLAIN_RECORDS
#endif
#define ROUTED_SOURCE_ROWS SEISMIC_DIM_XS
#include "lib/routed/down.h"
#include "lib/routed/down_sum.h"
#include "lib/core/arrive.h"

namespace {

// Four adjacent channels c .. c + 3 of a resident combine's projections in
// threadgroup memory, A-rounded.
struct TileProjections4 {
    threadgroup const float *projected;
    uint stride;
    uint c;
    float4 at4(uint k) const thread {
        threadgroup const float *at = projected + k * stride + c;
        return float4(
            routed::Act::round(at[0]),
            routed::Act::round(at[1]),
            routed::Act::round(at[2]),
            routed::Act::round(at[3]));
    }
};

// Four adjacent channels n .. n + 3 of row m's arrived projections in A,
// read through coherent device loads (`routed::PublishedSlots`).
template <uint SLOTS, uint H>
struct ArrivedProjections4 {
    routed::PublishedSlots published;
    ulong m;
    uint n;
    float4 at4(uint k) const thread {
        const ulong offset = (m * SLOTS + k) * H + n;
        if ((reinterpret_cast<ulong>(published + offset) & 7ul) == 0ul) {
            const metal::vec<typename routed::Act::storage, 4> raw =
                *reinterpret_cast<coherent(device) device const metal::vec<typename routed::Act::storage, 4> *>(
                    published + offset);
            return float4(
                routed::Act::load(raw.x),
                routed::Act::load(raw.y),
                routed::Act::load(raw.z),
                routed::Act::load(raw.w));
        }
        return float4(
            routed::Act::load(published[offset + 0ul]),
            routed::Act::load(published[offset + 1ul]),
            routed::Act::load(published[offset + 2ul]),
            routed::Act::load(published[offset + 3ul]));
    }
};

} // namespace

// The ports, the expert stack's: its records' address table.
#define ROUTED_DOWN_PORTS                                                                                  \
    device const float *base [[buffer(SEISMIC_BUFFER_BASE)]],                                                     \
        device const uchar *product [[buffer(SEISMIC_BUFFER_PRODUCT)]],                                           \
        device const int *routes [[buffer(SEISMIC_BUFFER_ROUTES)]],                                               \
        device const int *sources [[buffer(SEISMIC_BUFFER_SOURCES)]],                                             \
        device const float *weights [[buffer(SEISMIC_BUFFER_WEIGHTS)]],                                           \
        device const ulong *expert_down [[buffer(SEISMIC_BUFFER_EXPERT_DOWN)]],                                   \
        device const ulong *tier_down [[buffer(SEISMIC_BUFFER_TIER_DOWN)]],                                       \
        SEISMIC_RESULT_0_PARAM(value_cells),                                                                      \
        constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],                                           \
        uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],                   \
        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]

// The body (`routed::down_project`).
#define ROUTED_DOWN_BODY                                                                                          \
    constexpr uint TILE = routed::down_tile<ROWS, LANES>();                                                       \
    constexpr uint REGION = routed::down_region<ROWS, LANES>();                                                   \
    constexpr uint H = uint(SEISMIC_DIM_H);                                                                       \
    threadgroup float projected[uint(SEISMIC_DIM_K)][REGION];                                                     \
    const ulong m = group.y;                                                                                      \
    const uint first = group.x * TILE;                                                                            \
    const element::Out<ELEMENT_OF(SEISMIC_RESULT_0)> value(value_cells);                                          \
    routed::down_project<ROWS, LANES>(product, routes, sources, expert_down, tier_down, m, first, sg, lane,       \
        projected[sg], seismic_words);                                                                            \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                                              \
    const uint c = tid * 4u;                                                                                      \
    if (c >= TILE)                                                                                                \
        return;                                                                                                   \
    const uint n = first + c;                                                                                     \
    if (n >= H)                                                                                                   \
        return;                                                                                                   \
    if (c + 4u <= TILE && n + 4u <= H) {                                                                          \
        routed::down_publish4(base, weights, value, m, n, TileProjections4{&projected[0][0], REGION, c},          \
            routed::Answer4{false, float4(0.0f)}, seismic_words);                                                 \
    } else {                                                                                                      \
        for (uint i = 0u; i < 4u && c + i < TILE && n + i < H; ++i)                                               \
            routed::down_publish(base, weights, value, m, n + i,                                                  \
                routed::TileProjections{&projected[0][0], REGION, c + i}, routed::Answer{false, 0.0f},            \
                seismic_words);                                                                                   \
    }

// The gathered ports: the side-by-side form's, the published projections and
// the channel tiles' arrival counters.
#define ROUTED_DOWN_GATHERED_PORTS                                                  \
    ROUTED_DOWN_PORTS,                                                              \
        routed::PublishedSlots published [[buffer(SEISMIC_BUFFER_SCRATCH_PUBLISHED)]], \
        device atomic_uint *arrivals [[buffer(SEISMIC_BUFFER_SCRATCH_ARRIVALS)]]

// The gathered body (`routed::down_project_gathered`): threadgroup (x, m)'s
// simdgroups publish their batched projections of tile x; the last of the
// tile's M threadgroups to arrive sums the tile's channels of every row.
#define ROUTED_DOWN_GATHERED_BODY                                                                                 \
    constexpr uint TILE = routed::down_tile<ROWS, LANES>();                                                       \
    constexpr uint REGION = routed::down_region<ROWS, LANES>();                                                   \
    constexpr uint SLOTS = uint(SEISMIC_DIM_K);                                                                   \
    constexpr uint H = uint(SEISMIC_DIM_H);                                                                       \
    constexpr uint QUADS = (TILE + 3u) / 4u;                                                                      \
    threadgroup float projected[SLOTS][REGION];                                                                   \
    threadgroup uint last;                                                                                        \
    const uint first = group.x * TILE;                                                                            \
    const element::Out<ELEMENT_OF(SEISMIC_RESULT_0)> value(value_cells);                                          \
    routed::down_project_gathered<ROWS, LANES>(product, routes, sources, expert_down, tier_down,                  \
        uint(SEISMIC_DIM_M), group.y, first, sg, lane, projected[sg], published, seismic_words);                  \
    if (!arrive::last(arrivals + group.x, uint(SEISMIC_DIM_M), &last, tid))                                       \
        return;                                                                                                   \
    for (uint item = tid; item < uint(SEISMIC_DIM_M) * QUADS; item += 32u * SLOTS) {                              \
        const ulong m = item / QUADS;                                                                             \
        const uint c = (item % QUADS) * 4u;                                                                       \
        const uint n = first + c;                                                                                 \
        if (n >= H)                                                                                               \
            continue;                                                                                             \
        if (c + 4u <= TILE && n + 4u <= H) {                                                                      \
            routed::down_publish4(base, weights, value, m, n, ArrivedProjections4<SLOTS, H>{published, m, n},     \
                routed::Answer4{false, float4(0.0f)}, seismic_words);                                             \
        } else {                                                                                                  \
            for (uint i = 0u; i < 4u && c + i < TILE && n + i < H; ++i)                                           \
                routed::down_publish(base, weights, value, m, n + i,                                              \
                    routed::ArrivedProjections<SLOTS, H>{published, m, n + i}, routed::Answer{false, 0.0f},       \
                    seismic_words);                                                                               \
        }                                                                                                         \
    }

#ifdef SEISMIC_FORMING_ROUTED_DOWN_SUMMED
template <uint ROWS, uint LANES>
kernel void routed_down_summed(ROUTED_DOWN_PORTS) {
    ROUTED_DOWN_BODY
}
#endif
#ifdef SEISMIC_FORMING_ROUTED_DOWN_WHOLE
template <uint ROWS, uint LANES>
kernel void routed_down_whole(ROUTED_DOWN_PORTS) {
    ROUTED_DOWN_BODY
}
#endif
#ifdef SEISMIC_FORMING_ROUTED_DOWN_RESIDENT
// A resident layer's (XS 0), as the whole layer's: its records read with
// plain loads.
template <uint ROWS, uint LANES>
kernel void routed_down_resident(ROUTED_DOWN_PORTS) {
    ROUTED_DOWN_BODY
}
#endif
#ifdef SEISMIC_FORMING_ROUTED_DOWN_GATHERED
template <uint ROWS, uint LANES>
kernel void routed_down_gathered(ROUTED_DOWN_GATHERED_PORTS) {
    ROUTED_DOWN_GATHERED_BODY
}
#endif
#ifdef SEISMIC_FORMING_ROUTED_DOWN_GATHERED_WHOLE
template <uint ROWS, uint LANES>
kernel void routed_down_gathered_whole(ROUTED_DOWN_GATHERED_PORTS) {
    ROUTED_DOWN_GATHERED_BODY
}
#endif
#ifdef SEISMIC_FORMING_ROUTED_DOWN_GATHERED_RESIDENT
// A resident layer's (XS 0), as the whole layer's: its records read with
// plain loads.
template <uint ROWS, uint LANES>
kernel void routed_down_gathered_resident(ROUTED_DOWN_GATHERED_PORTS) {
    ROUTED_DOWN_GATHERED_BODY
}
#endif
