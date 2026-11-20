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
// destination record tables (gate, up, down), whether the record has a gate, the entry's
// words, the block's shared memory, and the window the thread holds: its
// span and its rows' pitch in the block's window.
struct Lander {
    const u8 *source[3];
    const u8 *destination[3];
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

// Canonical conversion prototype per routed tensor: bindings sharing one
// `(SourceRep, DestRep)` conversion (`Conversion`) collapse to a single
// prototype type, so identical conversions instantiate once by construction
// while per-tensor dimensions and strides travel in `repack::Geometry`.
typedef GateBinding GateProto;
typedef typename element::Select<element::Same<UpBinding::Conversion, GateBinding::Conversion>::value, GateBinding,
                                 UpBinding>::type UpProto;
typedef typename element::Select<
    element::Same<DownBinding::Conversion, GateBinding::Conversion>::value, GateBinding,
    typename element::Select<element::Same<DownBinding::Conversion, UpBinding::Conversion>::value, UpBinding,
                             DownBinding>::type>::type DownProto;

// Load window `window_index` of row tile `row_tile` of entry (from, to)'s
// record of `tensor` through binding `B` into the thread's registers.
template <class B>
__device__ __forceinline__ void land_load(Lander &land, const repack::Geometry &geometry, u32 tensor, int from, int to,
                                          u64 row_tile, u64 window_index) {
    const Window<B> window(geometry);
    const Placed<B> placed =
        place_of<B>(window, land.source[tensor], land.destination[tensor], geometry, from, to, row_tile, window_index);
    land.span = span_of(window, placed);
    land.pitch = window.pitch;
    land.held.load<min_load_unit<B>()>(land.span);
    land.scale_span = scale_span_of(placed);
    land.scales.load(land.scale_span);
}

// The held window into the block's window, between barriers.
template <class B> __device__ __forceinline__ void land_store(Lander &land) {
    __syncthreads();
    land.held.store<min_load_unit<B>()>(land.span, land.pitch, land.shared.window);
    land.scales.store(land.scale_span, land.shared.window);
    __syncthreads();
}

template <class B> __device__ __forceinline__ u32 land_items(const repack::Geometry &geometry) {
    return u32(record_items<B>(geometry, blockDim.x));
}

// The dense record's item `item`, as a span of the staged record's bytes, in
// the widest unit its address and size admit (the last item's rest may narrow it).
template <class B>
__device__ __forceinline__ Span dense_span_of(Lander &land, const repack::Geometry &geometry, u32 tensor, int from,
                                              u32 item) {
    const DenseItem span = dense_item_of<B>(geometry, item);
    const u32 bytes = span.count * dense_element<B>();
    const u8 *at = listed_source(land.source[tensor], land.destination[tensor], from) +
                   (u64)span.first * dense_element<B>();
    return Span{at, 0, 1, bytes, load_unit(u32(reinterpret_cast<u64>(at)) | bytes)};
}

template <class B>
__device__ __forceinline__ void land_begin(Lander &land, const repack::Geometry &geometry, u32 tensor, int from, int to,
                                           u32 item) {
    if constexpr (B::dense()) {
        land.span = dense_span_of<B>(land, geometry, tensor, from, item);
        land.pitch = land.span.bytes;
        land.held.load<min_load_unit<B>()>(land.span);
        land.scale_span = ScaleSpan{nullptr, 0, 0, 1, 0, 0, 0};
        land.scales.load(land.scale_span);
    } else {
        land_load<B>(land, geometry, tensor, from, to, item, 0);
    }
}

template <class B>
__device__ __forceinline__ void land_stage(Lander &land, const repack::Geometry &geometry, u32 tensor, int from, int to,
                                           u32 item) {
    if constexpr (B::dense()) {
        land_store<B>(land);
    } else {
        const Window<B> window(geometry);
        for (u32 at = 0; at + 1 < window.windows; ++at) {
            land_store<B>(land);
            land_load<B>(land, geometry, tensor, from, to, item, at + 1);
            convert_window<B>(window, land.source[tensor], land.destination[tensor], geometry, from, to, item, at,
                              land.shared);
        }
        land_store<B>(land);
    }
}

template <class B>
__device__ __forceinline__ void land_place(Lander &land, const repack::Geometry &geometry, u32 tensor, int from, int to,
                                           u32 item) {
    __syncthreads();
    if constexpr (B::dense()) {
        convert_span<B>(land.shared.window, land.destination[tensor], geometry, (u64)to, item);
    } else {
        const Window<B> window(geometry);
        convert_window<B>(window, land.source[tensor], land.destination[tensor], geometry, from, to, item,
                          window.windows - 1, land.shared);
    }
    __syncthreads();
}

} // namespace streaming

#define STREAMING_LAND_PARAM streaming::Lander &land
#define STREAMING_LAND_ARG land

#include "../../../streaming/land.hpp"

namespace streaming {

__device__ __forceinline__ repack::Geometry land_geometry(const seismic_words_t &words, u32 tensor) {
    if (tensor == STREAMING_LAND_GATE)
        return repack::Geometry::of<GateBinding>(words);
    if (tensor == STREAMING_LAND_UP)
        return repack::Geometry::of<UpBinding>(words);
    return repack::Geometry::of<DownBinding>(words);
}

// Dispatch `tensor` to its canonical `RuntimeBinding<Proto>`: identical
// conversions across gate, up and down collapse to one branch-free
// instantiation by construction, and only distinct conversions emit separate
// instantiations.
template <template <class> class Op, class... Args>
__device__ __forceinline__ auto dispatch_tensor(u32 tensor, Args &&...args) {
    constexpr bool up_same = element::Same<UpProto, GateProto>::value;
    constexpr bool down_same_gate = element::Same<DownProto, GateProto>::value;
    constexpr bool down_same_up = element::Same<DownProto, UpProto>::value;
    if constexpr (up_same && down_same_gate) {
        return Op<repack::RuntimeBinding<GateProto>>::run(static_cast<Args &&>(args)...);
    } else if constexpr (up_same) {
        if (tensor == STREAMING_LAND_DOWN)
            return Op<repack::RuntimeBinding<DownProto>>::run(static_cast<Args &&>(args)...);
        return Op<repack::RuntimeBinding<GateProto>>::run(static_cast<Args &&>(args)...);
    } else if constexpr (down_same_gate) {
        if (tensor == STREAMING_LAND_UP)
            return Op<repack::RuntimeBinding<UpProto>>::run(static_cast<Args &&>(args)...);
        return Op<repack::RuntimeBinding<GateProto>>::run(static_cast<Args &&>(args)...);
    } else if constexpr (down_same_up) {
        if (tensor == STREAMING_LAND_GATE)
            return Op<repack::RuntimeBinding<GateProto>>::run(static_cast<Args &&>(args)...);
        return Op<repack::RuntimeBinding<UpProto>>::run(static_cast<Args &&>(args)...);
    } else {
        if (tensor == STREAMING_LAND_GATE)
            return Op<repack::RuntimeBinding<GateProto>>::run(static_cast<Args &&>(args)...);
        if (tensor == STREAMING_LAND_UP)
            return Op<repack::RuntimeBinding<UpProto>>::run(static_cast<Args &&>(args)...);
        return Op<repack::RuntimeBinding<DownProto>>::run(static_cast<Args &&>(args)...);
    }
}

template <class B> struct LandItemsOp {
    __device__ static __forceinline__ u32 run(const repack::Geometry &geometry) { return land_items<B>(geometry); }
};
template <class B> struct LandBeginOp {
    __device__ static __forceinline__ void run(Lander &land, const repack::Geometry &geometry, u32 tensor, int from,
                                               int to, u32 item) {
        land_begin<B>(land, geometry, tensor, from, to, item);
    }
};
template <class B> struct LandStageOp {
    __device__ static __forceinline__ void run(Lander &land, const repack::Geometry &geometry, u32 tensor, int from,
                                               int to, u32 item) {
        land_stage<B>(land, geometry, tensor, from, to, item);
    }
};
template <class B> struct LandPlaceOp {
    __device__ static __forceinline__ void run(Lander &land, const repack::Geometry &geometry, u32 tensor, int from,
                                               int to, u32 item) {
        land_place<B>(land, geometry, tensor, from, to, item);
    }
};

} // namespace streaming

__device__ __forceinline__ u32 streaming_land_items(STREAMING_LAND_PARAM, u32 tensor) {
    if (tensor == STREAMING_LAND_GATE && !land.gated)
        return 0u;
    const repack::Geometry geometry = streaming::land_geometry(land.words, tensor);
    return streaming::dispatch_tensor<streaming::LandItemsOp>(tensor, geometry);
}

__device__ __forceinline__ void streaming_land_load(STREAMING_LAND_PARAM, u32 tensor, int from, int to, u32 item) {
    const repack::Geometry geometry = streaming::land_geometry(land.words, tensor);
    streaming::dispatch_tensor<streaming::LandBeginOp>(tensor, land, geometry, tensor, from, to, item);
}

__device__ __forceinline__ void streaming_land_stage(STREAMING_LAND_PARAM, u32 tensor, int from, int to, u32 item) {
    const repack::Geometry geometry = streaming::land_geometry(land.words, tensor);
    streaming::dispatch_tensor<streaming::LandStageOp>(tensor, land, geometry, tensor, from, to, item);
}

__device__ __forceinline__ void streaming_land_place(STREAMING_LAND_PARAM, u32 tensor, int from, int to, u32 item) {
    const repack::Geometry geometry = streaming::land_geometry(land.words, tensor);
    streaming::dispatch_tensor<streaming::LandPlaceOp>(tensor, land, geometry, tensor, from, to, item);
}
