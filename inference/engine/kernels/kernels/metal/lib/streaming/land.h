// Expert streaming's landing launch on Metal (`routed_land`,
// `routed_round`): the protocol every backend shares (streaming/land.hpp)
// over the window library's conversion (`window.h`): an item is one window
// of a row tile of a routed tensor's record (or a dense record's run of
// rows, `record_items`), converted by the whole threadgroup (`convert_item`),
// byte-equal to a resident import. A window is copied and placed in one
// step (the place hook): Metal reads staging in place, holding nothing in
// registers across claims.
//
// The includer includes control.h and window.h, defines the routed
// tensors' bindings `GateBinding`, `UpBinding` and `DownBinding`
// (`REPACK_BINDING`), then includes this file.

namespace streaming {

// A landing launch's conversion context: each routed tensor's source and
// destination record tables (gate, up, down), whether the record has a
// gate, the entry's words, the window and the thread's place.
struct Lander {
    device const ulong *source[3];
    device const ulong *destination[3];
    bool gated;
    constant ulong *words;
    threadgroup uchar *window;
    uint tid, threads, simd, simdgroups, lane;
};

// Item `item` of an entry's record of routed tensor `tensor` through
// binding `B`: from source record `from`, or for a negative `from` the
// stored bytes the host read into destination record -1 - from (a round's
// scratch slot no pass reads), into destination record `to`.
template <class B> inline void land_item(thread const Lander &land, uint tensor, int from, int to, uint item) {
    STAGED *source = from >= 0 ? source_record(land.source[tensor], ulong(from))
                               : source_record(land.destination[tensor], ulong(-1 - from));
    convert_item<B>(source, slab::region<uchar>(land.destination[tensor], ulong(to)), land.words, item, land.window,
        land.tid, land.threads, land.simd, land.simdgroups, land.lane);
}

} // namespace streaming

#define STREAMING_LAND_PARAM thread streaming::Lander &land
#define STREAMING_LAND_ARG land

#include "../../../streaming/land.hpp"

inline u32 streaming_land_items(STREAMING_LAND_PARAM, u32 tensor) {
    if (tensor == STREAMING_LAND_GATE)
        return land.gated ? u32(streaming::record_items<GateBinding>(land.words, land.threads)) : 0u;
    if (tensor == STREAMING_LAND_UP)
        return u32(streaming::record_items<UpBinding>(land.words, land.threads));
    return u32(streaming::record_items<DownBinding>(land.words, land.threads));
}

inline void streaming_land_load(STREAMING_LAND_PARAM, u32, int, int, u32) {}

inline void streaming_land_stage(STREAMING_LAND_PARAM, u32, int, int, u32) {}

inline void streaming_land_place(STREAMING_LAND_PARAM, u32 tensor, int from, int to, u32 item) {
    if (tensor == STREAMING_LAND_GATE)
        streaming::land_item<GateBinding>(land, tensor, from, to, item);
    else if (tensor == STREAMING_LAND_UP)
        streaming::land_item<UpBinding>(land, tensor, from, to, item);
    else
        streaming::land_item<DownBinding>(land, tensor, from, to, item);
}
