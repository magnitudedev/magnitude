// Expert streaming's landing launch on Vulkan (`routed_land`,
// `routed_round`); the Vulkan form of `metal/lib/streaming/land.h`: the
// protocol every backend shares (streaming/land.hpp) over the window
// library's conversion (`window.glsl`), an item one window of a row tile of
// a routed tensor's record (or a dense record's span of elements), converted by the
// whole workgroup, copied and placed in one step (the place hook). The call
// follows its park, which the host opened once every entry was appended,
// so the staged bytes are read as plain loads.
//
// The includer includes control.glsl and window.glsl, names its source
// tensors (`LAND_SOURCE_GATE`, `_UP`, `_DOWN`: their macro prefixes, and
// `LAND_SOURCE_GATE_BUFFER` ... their buffers), then includes this file.
// Its group memory holds the landing words (LAND_GROUP_WORDS) from word 0,
// then a packed record's window (`STREAMING_LAND_WINDOW`).

// A window's first byte in the group memory: past the landing words, at a
// 16-byte boundary (a window is copied in 16-byte loads).
#if SEISMIC_CONST_LAND_WINDOW_AT != 4 * SEISMIC_CONST_LAND_GROUP_WORDS || SEISMIC_CONST_LAND_WINDOW_AT % 16 != 0
#error "LAND_WINDOW_AT is the landing words' bytes, a 16-byte boundary"
#endif
#define STREAMING_LAND_WINDOW uint(SEISMIC_CONST_LAND_WINDOW_AT)

// The window library's per-binding definitions, their source named through
// the includer's macros (expanded here, before the library pastes them).
#define STREAMING_LAND_PACKED(NAME, S, SRC, D, N, K) STREAMING_PACKED(NAME, S, SRC, D, N, K)
#define STREAMING_LAND_DENSE(NAME, S, U, SRC, D, N, K) STREAMING_DENSE(NAME, S, U, SRC, D, N, K)

// rows16 stores exactly the source's groups (every stored tile holds source
// packets), and a window holds a packet of each row.
#if defined(SEISMIC_ELEMENT_UG_KIND_DENSE)
STREAMING_LAND_DENSE(land_gate, SEISMIC_ELEMENT_SG, SEISMIC_ELEMENT_UG, LAND_SOURCE_GATE, SEISMIC_GATE,
    SEISMIC_DIM_F, SEISMIC_DIM_H)
#else
#if SEISMIC_ELEMENT_UG_GROUP_MULTIPLE != 1 || \
    !STREAMING_WINDOW_FITS(REPACK_TILE_ROWS_INT, SEISMIC_ELEMENT_SG_PACKET_SIZE, SEISMIC_CONST_WINDOW_BYTES)
#error "rows16 stores no padding groups, and a window holds a source packet of each row"
#endif
REPACK_TILE(land_gate, SEISMIC_CONVERSION_SG_UG, SEISMIC_ELEMENT_SG, SEISMIC_GATE)
STREAMING_LAND_PACKED(land_gate, SEISMIC_ELEMENT_SG, LAND_SOURCE_GATE, SEISMIC_GATE, SEISMIC_DIM_F, SEISMIC_DIM_H)
#endif

#if defined(SEISMIC_ELEMENT_UU_KIND_DENSE)
STREAMING_LAND_DENSE(land_up, SEISMIC_ELEMENT_SU, SEISMIC_ELEMENT_UU, LAND_SOURCE_UP, SEISMIC_UP,
    SEISMIC_DIM_F, SEISMIC_DIM_H)
#else
#if SEISMIC_ELEMENT_UU_GROUP_MULTIPLE != 1 || \
    !STREAMING_WINDOW_FITS(REPACK_TILE_ROWS_INT, SEISMIC_ELEMENT_SU_PACKET_SIZE, SEISMIC_CONST_WINDOW_BYTES)
#error "rows16 stores no padding groups, and a window holds a source packet of each row"
#endif
REPACK_TILE(land_up, SEISMIC_CONVERSION_SU_UU, SEISMIC_ELEMENT_SU, SEISMIC_UP)
STREAMING_LAND_PACKED(land_up, SEISMIC_ELEMENT_SU, LAND_SOURCE_UP, SEISMIC_UP, SEISMIC_DIM_F, SEISMIC_DIM_H)
#endif

#if defined(SEISMIC_ELEMENT_UD_KIND_DENSE)
STREAMING_LAND_DENSE(land_down, SEISMIC_ELEMENT_SD, SEISMIC_ELEMENT_UD, LAND_SOURCE_DOWN, SEISMIC_DOWN,
    SEISMIC_DIM_H, SEISMIC_DIM_F)
#else
#if SEISMIC_ELEMENT_UD_GROUP_MULTIPLE != 1 || \
    !STREAMING_WINDOW_FITS(REPACK_TILE_ROWS_INT, SEISMIC_ELEMENT_SD_PACKET_SIZE, SEISMIC_CONST_WINDOW_BYTES)
#error "rows16 stores no padding groups, and a window holds a source packet of each row"
#endif
REPACK_TILE(land_down, SEISMIC_CONVERSION_SD_UD, SEISMIC_ELEMENT_SD, SEISMIC_DOWN)
STREAMING_LAND_PACKED(land_down, SEISMIC_ELEMENT_SD, LAND_SOURCE_DOWN, SEISMIC_DOWN, SEISMIC_DIM_H, SEISMIC_DIM_F)
#endif

// The conversion holds no state across claims: its context is the
// window's first byte.
#define STREAMING_LAND_PARAM uint land_window
#define STREAMING_LAND_ARG land_window

#include "../../../streaming/land.hpp"

// The stored record an entry's `from` names in a source of buffer `source`
// beside destination buffer `destination`: source record `from`, or for a
// negative `from` destination record -1 - from, which holds the stored
// bytes the host read there (a round's scratch slot no pass reads).
uint64_t streaming_land_source(uint64_t source, uint64_t destination, int from) {
    return from >= 0 ? slab_region(source, uint64_t(from)) : slab_region(destination, uint64_t(-1 - from));
}

uint streaming_land_items(uint land_window, uint tensor) {
    if (tensor == STREAMING_LAND_GATE)
        return SEISMIC_DIM_GV > 0 ? uint(land_gate_items()) : 0u;
    if (tensor == STREAMING_LAND_UP)
        return uint(land_up_items());
    return uint(land_down_items());
}

void streaming_land_load(uint land_window, uint tensor, int from, int to, uint item) {
}

void streaming_land_stage(uint land_window, uint tensor, int from, int to, uint item) {
}

void streaming_land_place(uint land_window, uint tensor, int from, int to, uint item) {
    if (tensor == STREAMING_LAND_GATE) {
        const uint64_t destination = SEISMIC_PTR(SEISMIC_BUFFER_GATE);
        land_gate_convert(streaming_land_source(SEISMIC_PTR(LAND_SOURCE_GATE_BUFFER), destination, from),
            slab_region(destination, uint64_t(to)), uint64_t(item), land_window);
    } else if (tensor == STREAMING_LAND_UP) {
        const uint64_t destination = SEISMIC_PTR(SEISMIC_BUFFER_UP);
        land_up_convert(streaming_land_source(SEISMIC_PTR(LAND_SOURCE_UP_BUFFER), destination, from),
            slab_region(destination, uint64_t(to)), uint64_t(item), land_window);
    } else {
        const uint64_t destination = SEISMIC_PTR(SEISMIC_BUFFER_DOWN);
        land_down_convert(streaming_land_source(SEISMIC_PTR(LAND_SOURCE_DOWN_BUFFER), destination, from),
            slab_region(destination, uint64_t(to)), uint64_t(item), land_window);
    }
}
