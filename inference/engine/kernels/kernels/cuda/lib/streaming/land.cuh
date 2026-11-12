// Expert streaming's landing launch on CUDA (`routed_land`,
// `routed_round`); the CUDA form of `metal/lib/streaming/land.h`: the
// protocol every backend shares (streaming/land.hpp) over the window
// library's conversion (`window.cuh`). An item is a whole row tile of a
// routed tensor's record (or a dense record's span of elements, `record_items`): the
// claim's first window is loaded into registers (`Held`) as the claim is
// begun (load hook); each window but the last is stored into the block's
// window, the next loaded and the stored one placed (stage hook); the last
// is stored, and, between the stage and the place, the block loads the next
// claim's first window where its entry has landed (streaming/land.hpp), so
// its loads are in flight while the last window places and the tile's
// mirror is written whole (place hook, `convert_window`).
//
// The includer includes control.cuh and window.cuh, defines the routed
// tensors' bindings `GateBinding`, `UpBinding` and `DownBinding`
// (`REPACK_BINDING`), then includes this file.

namespace streaming {

// A landing launch's group words (streaming/land.hpp), in its `Shared`.
struct LandWords {
    int group[SEISMIC_CONST_LAND_GROUP_WORDS];
};

typedef Shared<LandWords, GateBinding, UpBinding, DownBinding> LandShared;

// A landing launch's conversion context: each routed tensor's source and
// destination (gate, up, down), whether the record has a gate, the entry's
// words, the block's shared memory, and the window the thread holds: its
// span and its rows' pitch in the block's window.
struct Lander {
    const u8 *source[3];
    u8 *destination[3];
    bool gated;
    const seismic_words_t &words;
    LandShared &shared;
    Held held;
    Span span;
    u32 pitch;
    // The window's scales (a planar source's): in registers, and where they go.
    HeldScales<most(scale_iters<GateBinding>(), scale_iters<UpBinding>(), scale_iters<DownBinding>())> scales;
    ScaleSpan scale_span;
};

// Load window `window_index` of row tile `row_tile` of entry (from, to)'s
// record of `tensor` through binding `B` into the thread's registers.
template <class B>
__device__ __forceinline__ void land_load(Lander &land, u32 tensor, int from, int to, u64 row_tile, u64 window_index) {
    const Window<B> window(land.words);
    const Placed<B> placed =
        place_of<B>(window, land.source[tensor], land.destination[tensor], land.words, from, to, row_tile, window_index);
    land.span = span_of(window, placed);
    land.pitch = window.pitch;
    land.held.load(land.span);
    land.scale_span = scale_span_of(placed);
    land.scales.load(land.scale_span);
}

// The held window into the block's window, between barriers.
__device__ __forceinline__ void land_store(Lander &land) {
    __syncthreads();
    land.held.store(land.span, land.pitch, land.shared.window);
    land.scales.store(land.scale_span, land.shared.window);
    __syncthreads();
}

template <class B> __device__ __forceinline__ u32 land_items(const Lander &land) {
    return u32(record_items<B>(land.words, blockDim.x));
}

// The dense record's item `item`, as a span of the staged record's bytes, in
// the widest unit its address and size admit (the last item's rest may narrow it).
template <class B>
__device__ __forceinline__ Span dense_span_of(Lander &land, u32 tensor, int from, u32 item) {
    const DenseItem span = dense_item_of<B>(land.words, item);
    const u32 bytes = span.count * dense_element<B>();
    const u8 *at = listed_source<B>(land.source[tensor], land.destination[tensor], land.words, from) +
                   (u64)span.first * dense_element<B>();
    return Span{at, 0, 1, bytes, load_unit(u32(reinterpret_cast<u64>(at)) | bytes)};
}

template <class B> __device__ __forceinline__ void land_begin(Lander &land, u32 tensor, int from, int to, u32 item) {
    if constexpr (B::dense()) {
        land.span = dense_span_of<B>(land, tensor, from, item);
        land.pitch = land.span.bytes;
        land.held.load(land.span);
        land.scale_span = ScaleSpan{nullptr, 0, 0, 1, 0, 0, 0};
        land.scales.load(land.scale_span);
    } else {
        land_load<B>(land, tensor, from, to, item, 0);
    }
}

template <class B> __device__ __forceinline__ void land_stage(Lander &land, u32 tensor, int from, int to, u32 item) {
    if constexpr (B::dense()) {
        land_store(land);
    } else {
        const Window<B> window(land.words);
        for (u32 at = 0; at + 1 < window.windows; ++at) {
            land_store(land);
            land_load<B>(land, tensor, from, to, item, at + 1);
            convert_window<B>(window, land.source[tensor], land.destination[tensor], land.words, from, to, item, at,
                              land.shared);
        }
        land_store(land);
    }
}

template <class B> __device__ __forceinline__ void land_place(Lander &land, u32 tensor, int from, int to, u32 item) {
    __syncthreads();
    if constexpr (B::dense()) {
        convert_span<B>(land.shared.window, land.destination[tensor], land.words, (u64)to, item);
    } else {
        const Window<B> window(land.words);
        convert_window<B>(window, land.source[tensor], land.destination[tensor], land.words, from, to, item,
                          window.windows - 1, land.shared);
    }
    __syncthreads();
}

} // namespace streaming

#define STREAMING_LAND_PARAM streaming::Lander &land
#define STREAMING_LAND_ARG land

#include "../../../streaming/land.hpp"

__device__ __forceinline__ u32 streaming_land_items(STREAMING_LAND_PARAM, u32 tensor) {
    if (tensor == STREAMING_LAND_GATE)
        return land.gated ? streaming::land_items<GateBinding>(land) : 0u;
    if (tensor == STREAMING_LAND_UP)
        return streaming::land_items<UpBinding>(land);
    return streaming::land_items<DownBinding>(land);
}

__device__ __forceinline__ void streaming_land_load(STREAMING_LAND_PARAM, u32 tensor, int from, int to, u32 item) {
    if (tensor == STREAMING_LAND_GATE)
        streaming::land_begin<GateBinding>(land, tensor, from, to, item);
    else if (tensor == STREAMING_LAND_UP)
        streaming::land_begin<UpBinding>(land, tensor, from, to, item);
    else
        streaming::land_begin<DownBinding>(land, tensor, from, to, item);
}

__device__ __forceinline__ void streaming_land_stage(STREAMING_LAND_PARAM, u32 tensor, int from, int to, u32 item) {
    if (tensor == STREAMING_LAND_GATE)
        streaming::land_stage<GateBinding>(land, tensor, from, to, item);
    else if (tensor == STREAMING_LAND_UP)
        streaming::land_stage<UpBinding>(land, tensor, from, to, item);
    else
        streaming::land_stage<DownBinding>(land, tensor, from, to, item);
}

__device__ __forceinline__ void streaming_land_place(STREAMING_LAND_PARAM, u32 tensor, int from, int to, u32 item) {
    if (tensor == STREAMING_LAND_GATE)
        streaming::land_place<GateBinding>(land, tensor, from, to, item);
    else if (tensor == STREAMING_LAND_UP)
        streaming::land_place<UpBinding>(land, tensor, from, to, item);
    else
        streaming::land_place<DownBinding>(land, tensor, from, to, item);
}
