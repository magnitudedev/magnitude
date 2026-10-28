// Seismic native library (Vulkan): packet decoders for weights stored in the
// `rows16` layout (the Vulkan resident layout). `#include <seismic/packets.glsl>`.
// The counterpart of <seismic/packets.h> (Metal).
//
// A packet is 32 consecutive logical elements of one weight row: its slice of
// each row plane (its codes in the low code plane, their high bits in the
// high plane, or its four grid entries; its local coefficient codes and its
// factors). A packet is consumed in four 8-element sub-steps. Each sub-step
// decodes its codes as exact small integers in two vec4s, `even` (elements
// 0,2,4,6) and `odd` (1,3,5,7), by splitting code bytes on nibble
// boundaries: every decoded code is an exact integer, so the decode has no
// rounding or subnormal case.
//
// A packet's logical value is `scale(g) * (code - offset)`, plus a stored
// bias(g), for its coefficient group g, with one F32 rounding
// (`packets_values`). Dot products are factored:
// `scale * dot(code, x) + bias * sum(x)` (`packets_bias`),
// where the activation sums per 16 elements are computed once per staged
// activation and reused by every weight row.
//
// GLSL has no templates: a weight's decoder is composed from its slot's axes,
// the ABI's `<T>(F)`, the registry's decode axes as one tuple (seismic/runtime
// native ABI `render_axes_tuple`), read with its selectors
// `SEISMIC_AXIS_<AXIS>`. A slot's kind is its representation's identity (two
// slots of one representation share a kind), passed as the first argument of
// every operation; operations read the kind's axes as constants
// (`PACKETS_AXIS`). Weight tensors are bound to slots before this file is
// included, as on Metal and CUDA:
//     #define KERNEL_W0 SEISMIC_GATE_WEIGHT
// which defines `PACKETS_W0` (the slot's kind), `PACKETS_W0_ROWS` (its
// `packets_rows16`, a dense tensor's rows its axis 0) and
// `PACKETS_W0_STACKED_ROWS` (that of one matrix of a stacked [B, N, K]
// tensor, such as one expert's). Slots W0..W3 exist. Code that loads or
// decodes an axis value, here and in the libraries over this file, is a
// preprocessor group of the bound slots' values (`PACKETS_ANY`), so an
// entry's pipelines hold its own forms' paths alone: drivers compile code a
// constant makes dead at full cost. RADV took 9.8 s and 1.96 GB to create a
// projection GEMM pipeline staging every kind instead of 0.5 s and 213 MB
// (seismic/hardware/rdna3-vulkan.md), and an RTX 3090's NVIDIA compiler
// 4.4 s instead of 2.4 s for one staging two. The axes reach `#if` through
// the prefix's expansion alone: glslang pastes no tokens there. A component
// that dispatches on an axis states the values it implements, and a bound
// slot outside them is an `#error`.
//
// This file is independent of any entry ABI.
#include <seismic/element.glsl>
#include <seismic/tables.glsl>

#define PACKETS_CAT_(a, b) a##b
#define PACKETS_CAT(a, b) PACKETS_CAT_(a, b)
// The `packets_rows16` of a packed tensor (its row planes by role) and of a
// dense one (its rows its axis AXIS, each of its elements' bytes).
#define PACKETS_ROWS_PACKED(T) packets_rows16(PACKETS_CAT(T, _ROW_STRIDE_BYTES), PACKETS_CAT(T, _ROW_CODES), \
    PACKETS_CAT(T, _ROW_HIGH), PACKETS_CAT(T, _ROW_SCALES), PACKETS_CAT(T, _ROW_SUPERS))
#define PACKETS_ROWS_DENSE(T, AXIS) \
    packets_rows16(PACKETS_CAT(T, _STRIDE_##AXIS) * uint64_t(PACKETS_CAT(T, _PACKET_SIZE)), 0ul, 0ul, 0ul, 0ul)

// Slot bindings: each slot's axes (`PACKETS_Wn_AXES`), kind and rows.
#if defined(KERNEL_W0)
#define PACKETS_W0_AXES KERNEL_W0
#else
#define PACKETS_W0_AXES SEISMIC_AXES_NONE
#endif
#define PACKETS_W0 PACKETS_W0_AXES(SEISMIC_AXIS_ID)
#if defined(KERNEL_W0) && PACKETS_W0_AXES(SEISMIC_AXIS_FORM) == SEISMIC_FORM_DENSE
#define PACKETS_W0_ROWS PACKETS_ROWS_DENSE(KERNEL_W0, 0)
#define PACKETS_W0_STACKED_ROWS PACKETS_ROWS_DENSE(KERNEL_W0, 1)
#elif defined(KERNEL_W0) && PACKETS_W0_AXES(SEISMIC_AXIS_FORM) == SEISMIC_FORM_ROWS
#define PACKETS_W0_ROWS PACKETS_ROWS_PACKED(KERNEL_W0)
#define PACKETS_W0_STACKED_ROWS PACKETS_ROWS_PACKED(KERNEL_W0)
#elif defined(KERNEL_W0)
#error "weight slot W0 binds a representation with neither dense nor packed rows"
#endif
#if defined(KERNEL_W1)
#define PACKETS_W1_AXES KERNEL_W1
#else
#define PACKETS_W1_AXES SEISMIC_AXES_NONE
#endif
#define PACKETS_W1 PACKETS_W1_AXES(SEISMIC_AXIS_ID)
#if defined(KERNEL_W1) && PACKETS_W1_AXES(SEISMIC_AXIS_FORM) == SEISMIC_FORM_DENSE
#define PACKETS_W1_ROWS PACKETS_ROWS_DENSE(KERNEL_W1, 0)
#define PACKETS_W1_STACKED_ROWS PACKETS_ROWS_DENSE(KERNEL_W1, 1)
#elif defined(KERNEL_W1) && PACKETS_W1_AXES(SEISMIC_AXIS_FORM) == SEISMIC_FORM_ROWS
#define PACKETS_W1_ROWS PACKETS_ROWS_PACKED(KERNEL_W1)
#define PACKETS_W1_STACKED_ROWS PACKETS_ROWS_PACKED(KERNEL_W1)
#elif defined(KERNEL_W1)
#error "weight slot W1 binds a representation with neither dense nor packed rows"
#endif
#if defined(KERNEL_W2)
#define PACKETS_W2_AXES KERNEL_W2
#else
#define PACKETS_W2_AXES SEISMIC_AXES_NONE
#endif
#define PACKETS_W2 PACKETS_W2_AXES(SEISMIC_AXIS_ID)
#if defined(KERNEL_W2) && PACKETS_W2_AXES(SEISMIC_AXIS_FORM) == SEISMIC_FORM_DENSE
#define PACKETS_W2_ROWS PACKETS_ROWS_DENSE(KERNEL_W2, 0)
#define PACKETS_W2_STACKED_ROWS PACKETS_ROWS_DENSE(KERNEL_W2, 1)
#elif defined(KERNEL_W2) && PACKETS_W2_AXES(SEISMIC_AXIS_FORM) == SEISMIC_FORM_ROWS
#define PACKETS_W2_ROWS PACKETS_ROWS_PACKED(KERNEL_W2)
#define PACKETS_W2_STACKED_ROWS PACKETS_ROWS_PACKED(KERNEL_W2)
#elif defined(KERNEL_W2)
#error "weight slot W2 binds a representation with neither dense nor packed rows"
#endif
#if defined(KERNEL_W3)
#define PACKETS_W3_AXES KERNEL_W3
#else
#define PACKETS_W3_AXES SEISMIC_AXES_NONE
#endif
#define PACKETS_W3 PACKETS_W3_AXES(SEISMIC_AXIS_ID)
#if defined(KERNEL_W3) && PACKETS_W3_AXES(SEISMIC_AXIS_FORM) == SEISMIC_FORM_DENSE
#define PACKETS_W3_ROWS PACKETS_ROWS_DENSE(KERNEL_W3, 0)
#define PACKETS_W3_STACKED_ROWS PACKETS_ROWS_DENSE(KERNEL_W3, 1)
#elif defined(KERNEL_W3) && PACKETS_W3_AXES(SEISMIC_AXIS_FORM) == SEISMIC_FORM_ROWS
#define PACKETS_W3_ROWS PACKETS_ROWS_PACKED(KERNEL_W3)
#define PACKETS_W3_STACKED_ROWS PACKETS_ROWS_PACKED(KERNEL_W3)
#elif defined(KERNEL_W3)
#error "weight slot W3 binds a representation with neither dense nor packed rows"
#endif

// A kind's axis `SEL` (a selector `SEISMIC_AXIS_*`; a constant: `kind` is one
// of the slots').
#define PACKETS_AXIS(kind, SEL) ((kind) == PACKETS_W0 ? PACKETS_W0_AXES(SEL) : (kind) == PACKETS_W1 \
    ? PACKETS_W1_AXES(SEL) : (kind) == PACKETS_W2 ? PACKETS_W2_AXES(SEL) : PACKETS_W3_AXES(SEL))
// Whether slot axes `AXES` are packed rows with axis `SEL` equal to `V`, and
// whether a bound slot is: preprocessor conditions.
#define PACKETS_SLOT(AXES, SEL, V) (AXES(SEISMIC_AXIS_FORM) == SEISMIC_FORM_ROWS && AXES(SEL) == (V))
#define PACKETS_ANY(SEL, V) (PACKETS_SLOT(PACKETS_W0_AXES, SEL, V) || PACKETS_SLOT(PACKETS_W1_AXES, SEL, V) \
    || PACKETS_SLOT(PACKETS_W2_AXES, SEL, V) || PACKETS_SLOT(PACKETS_W3_AXES, SEL, V))
#define PACKETS_ANY_FORM(FORM) (PACKETS_W0_AXES(SEISMIC_AXIS_FORM) == (FORM) || PACKETS_W1_AXES(SEISMIC_AXIS_FORM) == (FORM) \
    || PACKETS_W2_AXES(SEISMIC_AXIS_FORM) == (FORM) || PACKETS_W3_AXES(SEISMIC_AXIS_FORM) == (FORM))
#define PACKETS_ANY_DENSE PACKETS_ANY_FORM(SEISMIC_FORM_DENSE)
#define PACKETS_ANY_PACKED PACKETS_ANY_FORM(SEISMIC_FORM_ROWS)
// Whether every bound slot's axes satisfy `PRED(AXES)`: a component's domain.
#define PACKETS_ALL(PRED) (PRED(PACKETS_W0_AXES) && PRED(PACKETS_W1_AXES) && PRED(PACKETS_W2_AXES) \
    && PRED(PACKETS_W3_AXES))
// Whether some bound slot's axes satisfy `PRED(AXES)`, and whether a kind's
// do (a constant: `kind` is one of the slots').
#define PACKETS_SOME(PRED) (PRED(PACKETS_W0_AXES) || PRED(PACKETS_W1_AXES) || PRED(PACKETS_W2_AXES) \
    || PRED(PACKETS_W3_AXES))
#define PACKETS_OF(kind, PRED) ((kind) == PACKETS_W0 ? PRED(PACKETS_W0_AXES) : (kind) == PACKETS_W1 \
    ? PRED(PACKETS_W1_AXES) : (kind) == PACKETS_W2 ? PRED(PACKETS_W2_AXES) : PRED(PACKETS_W3_AXES))
// Whether slot axes `AXES` have coefficient groups that carry a bias term: a
// minimum or stored bias (`COEFF_BIAS`) or the offset of offset codes
// (`packets_bias`).
#define PACKETS_BIASED(AXES) (AXES(SEISMIC_AXIS_COEFF_BIAS) != 0 || AXES(SEISMIC_AXIS_CODES) == SEISMIC_CODES_OFFSET)
#define PACKETS_ANY_BIASED PACKETS_SOME(PACKETS_BIASED)

// A kind's form, and whether it is dense or has a bias term.
bool packets_dense(const int kind) {
    return PACKETS_AXIS(kind, SEISMIC_AXIS_FORM) == SEISMIC_FORM_DENSE;
}

bool packets_biased(const int kind) { return PACKETS_OF(kind, PACKETS_BIASED); }

// Coefficient groups per packet (one per 32 or more elements, two per 16).
uint packets_groups(const int kind) {
    return packets_dense(kind) ? 1u : 32u / min(32u, uint(PACKETS_AXIS(kind, SEISMIC_AXIS_COEFF_GROUP)));
}

// A dense kind's element kind.
int packets_element(const int kind) {
    const int dtype = PACKETS_AXIS(kind, SEISMIC_AXIS_DTYPE);
    return dtype == SEISMIC_DTYPE_BF16 ? ELEMENT_BF16 : dtype == SEISMIC_DTYPE_F16 ? ELEMENT_F16 : ELEMENT_F32;
}

// ---------------------------------------------------------------------------
// Weight row geometry. `stride` is the byte distance between rows; plane
// offsets are within a row. Planes a representation lacks are unused.

struct packets_rows16 {
    uint64_t stride;
    uint64_t codes;   // the low code plane, the grid entries or the values (dense)
    uint64_t high;    // the high code plane
    uint64_t scales;  // the local coefficient codes
    uint64_t supers;  // the direct coefficients or the factors
};

// Whether slot axes `AXES` have two biased coefficient groups a packet. A
// packet holds a second bias only where a bound slot does: a launch whose
// slots bind several kinds holds one packet for all of them (a staged GEMM
// holds several), and a bias none of them has would be a register each.
#define PACKETS_TWO_BIASES(AXES) (AXES(SEISMIC_AXIS_FORM) == SEISMIC_FORM_ROWS \
    && AXES(SEISMIC_AXIS_COEFF_BIAS) != 0 && AXES(SEISMIC_AXIS_COEFF_GROUP) < 32)

// One loaded packet. A group's coefficients are scalar fields, read and
// written through `packets_group_scale`, `packets_group_bias` and
// `packets_set_group`: held as one vec2, the scales spilled the q4k/q6k
// staged GEMM to 32 bytes of local memory instead of 16 and slowed it 3%
// (NVIDIA GB10, driver 580.159.03).
struct packets_packet {
    uvec4 low;       // the low plane's codes (8-bit: the first 16) or the grid entries
    uvec4 second;    // 8-bit codes: the last 16
    uvec2 high;      // the high plane's codes: one bit in .x, two in .xy
    float scale0;    // the scales of the packet's (one or two) coefficient groups
    float scale1;
    float bias0;     // their biases (a minimum or stored bias)
#if PACKETS_SOME(PACKETS_TWO_BIASES)
    float bias1;
#endif
    uint64_t values; // dense: the packet's first element
    uint valid;      // dense: elements of this packet inside the row
};

// Coefficient group `g`'s scale and bias (a group's bias is the first's
// where no bound slot has a second).
float packets_group_scale(packets_packet packet, uint g) { return g == 0u ? packet.scale0 : packet.scale1; }

float packets_group_bias(packets_packet packet, uint g) {
#if PACKETS_SOME(PACKETS_TWO_BIASES)
    return g == 0u ? packet.bias0 : packet.bias1;
#else
    return packet.bias0;
#endif
}

// Set coefficient group `g`'s scale, and its bias where `biased`.
void packets_set_group(inout packets_packet packet, uint g, float scale, const bool biased, float bias) {
    if (g == 0u) {
        packet.scale0 = scale;
        if (biased)
            packet.bias0 = bias;
    } else {
        packet.scale1 = scale;
#if PACKETS_SOME(PACKETS_TWO_BIASES)
        if (biased)
            packet.bias1 = bias;
#endif
    }
}

// ---------------------------------------------------------------------------
// Code planes (`registry::code_planes`): a low plane of LOW bits a code and a
// high plane of HIGH, a code's bits one byte of four (`packets_codes`), or a
// grid code's entries: four of LOW bits a packet.

// The code planes this file reads: 2-bit low planes (with a high plane of
// up to one bit), 4-bit low planes (up to two), 8-bit low planes, and 16- or
// 24-bit grid entries with a high part below a byte or none.
#define PACKETS_READS_PLANES(AXES) (AXES(SEISMIC_AXIS_FORM) != SEISMIC_FORM_ROWS \
    || (AXES(SEISMIC_AXIS_CODES) == SEISMIC_CODES_GRID && AXES(SEISMIC_AXIS_CODE_HIGH_BITS) < 8 \
        && (AXES(SEISMIC_AXIS_CODE_LOW_BITS) == 16 || AXES(SEISMIC_AXIS_CODE_LOW_BITS) == 24)) \
    || (AXES(SEISMIC_AXIS_CODES) != SEISMIC_CODES_GRID \
        && ((AXES(SEISMIC_AXIS_CODE_LOW_BITS) == 2 && AXES(SEISMIC_AXIS_CODE_HIGH_BITS) <= 1) \
            || (AXES(SEISMIC_AXIS_CODE_LOW_BITS) == 4 && AXES(SEISMIC_AXIS_CODE_HIGH_BITS) <= 2) \
            || (AXES(SEISMIC_AXIS_CODE_LOW_BITS) == 8 && AXES(SEISMIC_AXIS_CODE_HIGH_BITS) == 0))))
#if !PACKETS_ALL(PACKETS_READS_PLANES)
#error "packets: a weight slot's code planes are of a width this file does not read"
#endif

// Bytes 0,2,4,6 / 1,3,5,7 of a 16-bit field of 2-bit high codes, moved to
// bits 4..5 of four bytes.
void packets_split_high2(uint field, out uint even, out uint odd) {
    uint t = (field | (field << 8)) & 0x00ff00ffu;
    t = (t | (t << 4)) & 0x0f0f0f0fu;
    even = (t & 0x03030303u) << 4;
    odd = (t & 0x0c0c0c0cu) << 2;
}

#if PACKETS_ANY(SEISMIC_AXIS_CODE_LOW_BITS, 2)
// The same for a 16-bit field of 2-bit codes, in bits 0..1.
void packets_split_crumbs(uint field, out uint even, out uint odd) {
    packets_split_high2(field, even, odd);
    even >>= 4u;
    odd >>= 4u;
}
#endif

// The same for an 8-bit field of 1-bit high codes, moved to bit 4.
void packets_split_high1(uint field, out uint even, out uint odd) {
    uint t = (field | (field << 12)) & 0x000f000fu;
    t = (t | (t << 6)) & 0x03030303u;
    even = (t & 0x01010101u) << 4;
    odd = (t & 0x02020202u) << 3;
}

vec4 packets_unsigned_bytes(uint word) { return vec4(unpack8(word)); }

// ---------------------------------------------------------------------------
// Interpretations of codes (`registry::CodeInterpretation`): four codes, a
// byte each, as exact values. Unsigned and offset codes are the bytes (or
// an offset code less its offset, `centred`: codes of at most 7 bits and an
// offset of at most 128, so that the difference is a signed byte), two's
// complement codes (bytes) the signed bytes, and a codebook's codes
// (nibbles) the codebook's values (`CODEBOOKS_TABLE_<ID>`, `PACKETS_CODEBOOK`). A product
// operand takes an offset off as the exact bias -(offset scale)
// (`packets_values`): the offset is a power of two, without a stored bias.

#define PACKETS_READS_CODES(AXES) (AXES(SEISMIC_AXIS_FORM) != SEISMIC_FORM_ROWS \
    || AXES(SEISMIC_AXIS_CODES) == SEISMIC_CODES_UNSIGNED \
    || (AXES(SEISMIC_AXIS_CODES) == SEISMIC_CODES_OFFSET \
        && AXES(SEISMIC_AXIS_CODE_LOW_BITS) + AXES(SEISMIC_AXIS_CODE_HIGH_BITS) <= 7 \
        && AXES(SEISMIC_AXIS_CODES_OFFSET) <= 128 \
        && (AXES(SEISMIC_AXIS_CODES_OFFSET) & (AXES(SEISMIC_AXIS_CODES_OFFSET) - 1)) == 0 \
        && AXES(SEISMIC_AXIS_COEFF_BIAS) == 0) \
    || (AXES(SEISMIC_AXIS_CODES) == SEISMIC_CODES_TWOS \
        && AXES(SEISMIC_AXIS_CODE_LOW_BITS) + AXES(SEISMIC_AXIS_CODE_HIGH_BITS) == 8) \
    || (AXES(SEISMIC_AXIS_CODES) == SEISMIC_CODES_TABLE \
        && AXES(SEISMIC_AXIS_CODE_LOW_BITS) + AXES(SEISMIC_AXIS_CODE_HIGH_BITS) == 4) \
    || AXES(SEISMIC_AXIS_CODES) == SEISMIC_CODES_GRID)
#if !PACKETS_ALL(PACKETS_READS_CODES)
#error "packets: a weight slot's codes are of an interpretation this file does not read"
#endif

// The codebook value of `code` of slot axes `AXES` (`CODEBOOKS_TABLE_<id>`).
#define PACKETS_CODEBOOK(AXES) PACKETS_CAT(CODEBOOKS_TABLE_, AXES(SEISMIC_AXIS_CODES_TABLE))

#if PACKETS_ANY(SEISMIC_AXIS_CODES, SEISMIC_CODES_TABLE)
float packets_codebook(const int kind, uint code) {
#if PACKETS_SLOT(PACKETS_W0_AXES, SEISMIC_AXIS_CODES, SEISMIC_CODES_TABLE)
    if (kind == PACKETS_W0)
        return PACKETS_CODEBOOK(PACKETS_W0_AXES)[code];
#endif
#if PACKETS_SLOT(PACKETS_W1_AXES, SEISMIC_AXIS_CODES, SEISMIC_CODES_TABLE)
    if (kind == PACKETS_W1)
        return PACKETS_CODEBOOK(PACKETS_W1_AXES)[code];
#endif
#if PACKETS_SLOT(PACKETS_W2_AXES, SEISMIC_AXIS_CODES, SEISMIC_CODES_TABLE)
    if (kind == PACKETS_W2)
        return PACKETS_CODEBOOK(PACKETS_W2_AXES)[code];
#endif
#if PACKETS_SLOT(PACKETS_W3_AXES, SEISMIC_AXIS_CODES, SEISMIC_CODES_TABLE)
    if (kind == PACKETS_W3)
        return PACKETS_CODEBOOK(PACKETS_W3_AXES)[code];
#endif
    return 0.0;
}
#endif

// Four codes, a byte each, as exact values, offset codes less their offset
// where `centred` (per byte: 128 + code - offset, without borrows, with its
// top bit flipped: code - offset as a signed byte).
vec4 packets_interpret(const int kind, uint bytes, const bool centred) {
    const int interpretation = PACKETS_AXIS(kind, SEISMIC_AXIS_CODES);
#if PACKETS_ANY(SEISMIC_AXIS_CODES, SEISMIC_CODES_OFFSET)
    if (centred && interpretation == SEISMIC_CODES_OFFSET)
        return vec4(unpack8(int(((bytes | 0x80808080u) - uint(PACKETS_AXIS(kind, SEISMIC_AXIS_CODES_OFFSET)) * 0x01010101u)
            ^ 0x80808080u)));
#endif
#if PACKETS_ANY(SEISMIC_AXIS_CODES, SEISMIC_CODES_TABLE)
    if (interpretation == SEISMIC_CODES_TABLE)
        return vec4(packets_codebook(kind, bytes & 0xffu), packets_codebook(kind, (bytes >> 8) & 0xffu),
            packets_codebook(kind, (bytes >> 16) & 0xffu), packets_codebook(kind, bytes >> 24));
#endif
#if PACKETS_ANY(SEISMIC_AXIS_CODES, SEISMIC_CODES_TWOS)
    if (interpretation == SEISMIC_CODES_TWOS)
        return vec4(unpack8(int(bytes)));
#endif
    return packets_unsigned_bytes(bytes);
}

// ---------------------------------------------------------------------------
// Grid codes (`registry::GridCodes`). An entry codes eight values: grid
// indices name their magnitudes (<seismic/tables.glsl>), and eight sign
// bits, or seven completed by an eighth of even parity, their signs. Its
// whole bytes are in the grids plane, a high part below a byte in the high
// plane.

// The grid table of slot axes `AXES` (`GRIDS_TABLE_<id>`).
#define PACKETS_GRID_TABLE(AXES) PACKETS_CAT(GRIDS_TABLE_, AXES(SEISMIC_AXIS_CODES_GRID))

// Whether slot axes `AXES` are grid codes whose entries have a high part in
// the high plane (`registry::RowPlaneContent::GridHigh`, placed as a group's
// bytes: a packet's four entries' high bits at bit 4 HIGH p of a row's),
// held in a packet's `high.x`.
#define PACKETS_GRID_HIGH(AXES) (AXES(SEISMIC_AXIS_FORM) == SEISMIC_FORM_ROWS \
    && AXES(SEISMIC_AXIS_CODES) == SEISMIC_CODES_GRID && AXES(SEISMIC_AXIS_CODE_HIGH_BITS) != 0)

#if PACKETS_ANY(SEISMIC_AXIS_CODES, SEISMIC_CODES_GRID)
// Entry `step` of a packet's four entries (`LOW` bits each).
uint packets_grid_entry(const int kind, uvec4 entries, uint step) {
#if PACKETS_ANY(SEISMIC_AXIS_CODE_LOW_BITS, 24)
    if (PACKETS_AXIS(kind, SEISMIC_AXIS_CODE_LOW_BITS) == 24) {
        const uint word = (3u * step) >> 2, shift = (24u * step) & 31u;
        const uint spill = shift > 8u ? entries[word + 1u] << (32u - shift) : 0u;
        return ((entries[word] >> shift) | spill) & 0xffffffu;
    }
#endif
    return (entries[step >> 1] >> (16u * (step & 1u))) & 0xffffu;
}

// Entry `step` of a packet's four entries: its whole bytes, then its high
// part.
uint packets_grid_code(const int kind, packets_packet packet, uint step) {
    const uint entry = packets_grid_entry(kind, packet.low, step);
#if PACKETS_SOME(PACKETS_GRID_HIGH)
    const uint high = uint(PACKETS_AXIS(kind, SEISMIC_AXIS_CODE_HIGH_BITS));
    if (high != 0u)
        return entry | (((packet.high.x >> (high * step)) & ((1u << high) - 1u))
            << uint(PACKETS_AXIS(kind, SEISMIC_AXIS_CODE_LOW_BITS)));
#endif
    return entry;
}

// An entry's grid point (lanes 0..3 in x, 4..7 in y): one index of an
// eight-lane grid, or two of a four-lane grid, the first lowest. A grid's
// points are `uvec2`s or `uint`s by its lanes, so a slot's lanes choose its
// code here, in the preprocessor.
#define PACKETS_GRID_MASK(AXES) ((1u << uint(AXES(SEISMIC_AXIS_CODES_GRID_INDEX_BITS))) - 1u)
#define PACKETS_GRID_POINT8(AXES, entry) return PACKETS_GRID_TABLE(AXES)[(entry) & PACKETS_GRID_MASK(AXES)];
#define PACKETS_GRID_POINT4(AXES, entry) \
    return uvec2(PACKETS_GRID_TABLE(AXES)[(entry) & PACKETS_GRID_MASK(AXES)], \
        PACKETS_GRID_TABLE(AXES)[((entry) >> AXES(SEISMIC_AXIS_CODES_GRID_INDEX_BITS)) & PACKETS_GRID_MASK(AXES)]);
uvec2 packets_grid_point(const int kind, uint entry) {
#if PACKETS_SLOT(PACKETS_W0_AXES, SEISMIC_AXIS_CODES, SEISMIC_CODES_GRID)
    if (kind == PACKETS_W0) {
#if PACKETS_W0_AXES(SEISMIC_AXIS_CODES_GRID_LANES) == 8
        PACKETS_GRID_POINT8(PACKETS_W0_AXES, entry)
#else
        PACKETS_GRID_POINT4(PACKETS_W0_AXES, entry)
#endif
    }
#endif
#if PACKETS_SLOT(PACKETS_W1_AXES, SEISMIC_AXIS_CODES, SEISMIC_CODES_GRID)
    if (kind == PACKETS_W1) {
#if PACKETS_W1_AXES(SEISMIC_AXIS_CODES_GRID_LANES) == 8
        PACKETS_GRID_POINT8(PACKETS_W1_AXES, entry)
#else
        PACKETS_GRID_POINT4(PACKETS_W1_AXES, entry)
#endif
    }
#endif
#if PACKETS_SLOT(PACKETS_W2_AXES, SEISMIC_AXIS_CODES, SEISMIC_CODES_GRID)
    if (kind == PACKETS_W2) {
#if PACKETS_W2_AXES(SEISMIC_AXIS_CODES_GRID_LANES) == 8
        PACKETS_GRID_POINT8(PACKETS_W2_AXES, entry)
#else
        PACKETS_GRID_POINT4(PACKETS_W2_AXES, entry)
#endif
    }
#endif
#if PACKETS_SLOT(PACKETS_W3_AXES, SEISMIC_AXIS_CODES, SEISMIC_CODES_GRID)
    if (kind == PACKETS_W3) {
#if PACKETS_W3_AXES(SEISMIC_AXIS_CODES_GRID_LANES) == 8
        PACKETS_GRID_POINT8(PACKETS_W3_AXES, entry)
#else
        PACKETS_GRID_POINT4(PACKETS_W3_AXES, entry)
#endif
    }
#endif
    return uvec2(0u);
}

// The bit of an entry past its indices: its first sign bit.
uint packets_grid_signs_shift(const int kind) {
    return uint(PACKETS_AXIS(kind, SEISMIC_AXIS_CODES_GRID_INDEX_BITS))
        * (8u / uint(PACKETS_AXIS(kind, SEISMIC_AXIS_CODES_GRID_LANES)));
}

// The eight lane signs of `entry` (`registry::GridSigns`): its eight sign
// bits, or seven and an eighth that makes the count of set bits even.
uint packets_grid_signs(const int kind, uint entry) {
    const uint signs = entry >> packets_grid_signs_shift(kind);
#if PACKETS_ANY(SEISMIC_AXIS_CODES_GRID_SIGNS, 8)
    if (PACKETS_AXIS(kind, SEISMIC_AXIS_CODES_GRID_SIGNS) == 8)
        return signs & 255u;
#endif
    return (signs & 127u) | ((uint(bitCount(signs & 127u)) & 1u) << 7);
}

// The eight signed codes of `point` under its eight lane signs `signs`, as
// (0,2,4,6), (1,3,5,7).
void packets_grid_codes(uvec2 point, uint signs, out vec4 even, out vec4 odd) {
    const vec4 low = packets_unsigned_bytes(point.x), high = packets_unsigned_bytes(point.y);
    even = uintBitsToFloat(floatBitsToUint(vec4(low.x, low.z, high.x, high.z)) ^
                           (((uvec4(signs) >> uvec4(0u, 2u, 4u, 6u)) & 1u) << 31));
    odd = uintBitsToFloat(floatBitsToUint(vec4(low.y, low.w, high.y, high.w)) ^
                          (((uvec4(signs) >> uvec4(1u, 3u, 5u, 7u)) & 1u) << 31));
}
#endif

// ---------------------------------------------------------------------------
// Coefficients. Direct: a scale (and a bias) per GROUP values, stored per
// storage group as its scales, then its biases. Hierarchical: per GROUP
// values a LOCAL_BITS-bit scale code (and a minimum code after it),
// interpreted by LOCAL, under a factor (and a minimum factor) per
// FACTOR_GROUP values: scale = factor 2^EXPONENT L(scale code) and bias =
// BIAS_SIGN minimum factor 2^EXPONENT L(minimum code). Scales carry the
// codes' exponent (`CODES_EXPONENT`).

// Bytes of one scale of encoding `scale` (`SEISMIC_SCALE_*`).
uint packets_scale_bytes(const int scale) {
    return scale == SEISMIC_SCALE_F32 ? 4u : scale == SEISMIC_SCALE_F16 || scale == SEISMIC_SCALE_BF16 ? 2u : 1u;
}

// E8M0: 2^(e - 127) (e = 0 is the F32 subnormal 2^-127); 0xff is NaN.
float packets_e8m0(uint e) {
    return uintBitsToFloat(e == 0u ? 0x00400000u : e == 255u ? 0x7fc00000u : e << 23);
}

// UE4M3 (sign bit ignored; 0x7f is NaN): (1 + m/8) 2^(e - 7), or m 2^-9 when
// e = 0.
float packets_ue4m3(uint raw) {
    raw &= 0x7fu;
    const uint e = raw >> 3, m = raw & 7u;
    const float normal = uintBitsToFloat(((e + 120u) << 23) | (m << 20));
    return raw == 0x7fu ? uintBitsToFloat(0x7fc00000u) : e == 0u ? float(m) * 0.001953125 : normal;
}

// The value of a scale of encoding `scale` stored as the low bytes of `raw`.
float packets_scale_value(const int scale, uint raw) {
#if PACKETS_ANY(SEISMIC_AXIS_COEFF_SCALE, SEISMIC_SCALE_F32)
    if (scale == SEISMIC_SCALE_F32)
        return uintBitsToFloat(raw);
#endif
#if PACKETS_ANY(SEISMIC_AXIS_COEFF_SCALE, SEISMIC_SCALE_F16)
    if (scale == SEISMIC_SCALE_F16)
        return seismic_f16_to_f32(uint16_t(raw));
#endif
#if PACKETS_ANY(SEISMIC_AXIS_COEFF_SCALE, SEISMIC_SCALE_BF16)
    if (scale == SEISMIC_SCALE_BF16)
        return uintBitsToFloat(raw << 16);
#endif
#if PACKETS_ANY(SEISMIC_AXIS_COEFF_SCALE, SEISMIC_SCALE_E8M0)
    if (scale == SEISMIC_SCALE_E8M0)
        return packets_e8m0(raw & 0xffu);
#endif
#if PACKETS_ANY(SEISMIC_AXIS_COEFF_SCALE, SEISMIC_SCALE_UE4M3)
    if (scale == SEISMIC_SCALE_UE4M3)
        return packets_ue4m3(raw & 0xffu);
#endif
    return 0.0;
}

// The scale of encoding `scale` at `at`, and the two adjacent ones there,
// each read as one load.
float packets_scale_at(const int scale, uint64_t at) {
    const uint bytes = packets_scale_bytes(scale);
    return packets_scale_value(scale,
        bytes == 4u ? element_u32_at(at) : bytes == 2u ? element_u16_at(at) : element_u8_at(at));
}

vec2 packets_scale_pair_at(const int scale, uint64_t at) {
    const uint bytes = packets_scale_bytes(scale);
    if (bytes == 4u) {
        const uvec2 two = element_uvec2_at(at);
        return vec2(packets_scale_value(scale, two.x), packets_scale_value(scale, two.y));
    }
    const uint two = bytes == 2u ? element_u32_at(at) : element_u16_at(at);
    return vec2(packets_scale_value(scale, two), packets_scale_value(scale, two >> (8u * bytes)));
}

// A packet's local field bits (its groups' scale codes, and minimum codes,
// of a coefficient group of 16 or 32: two groups or one), so that no `#if`
// divides.
#define PACKETS_FIELD_BITS(AXES) (AXES(SEISMIC_AXIS_COEFF_LOCAL_BITS) * (1 + AXES(SEISMIC_AXIS_COEFF_BIAS)) \
    * (1 + (AXES(SEISMIC_AXIS_COEFF_GROUP) == 16)))
// Whether a field of F bits read from the plane fits the reader's four
// bytes: it starts at a multiple of F's lowest set bit (at most 8) within a
// byte, so it reaches F + 8 - that bit (`packets_coefficients`), at most 32.
#define PACKETS_FIELD_SPANS(F) ((F) <= 25 || ((F) % 2 == 0 && (F) <= 26) || ((F) % 4 == 0 && (F) <= 28) \
    || (F) % 8 == 0)
// The coefficient structures this file reads: direct ones per 16 values or a
// multiple of 32 (with a bias, each group's scale then its bias), and
// hierarchical ones per 16 or 32 values whose local codes are unsigned, two's
// complement, offset or odd, either carried (without a minimum) by at most a
// group's grid entries or in the scales plane, a packet's field below a word
// and within the reader's four bytes.
#define PACKETS_READS_COEFFICIENTS(AXES) (AXES(SEISMIC_AXIS_FORM) != SEISMIC_FORM_ROWS \
    || (AXES(SEISMIC_AXIS_COEFF) == SEISMIC_COEFF_DIRECT \
        && (AXES(SEISMIC_AXIS_COEFF_GROUP) == 16 || AXES(SEISMIC_AXIS_COEFF_GROUP) % 32 == 0) \
        && (AXES(SEISMIC_AXIS_COEFF_BIAS) == 0 || AXES(SEISMIC_AXIS_LOGICAL_GROUP) == AXES(SEISMIC_AXIS_COEFF_GROUP))) \
    || (AXES(SEISMIC_AXIS_COEFF) == SEISMIC_COEFF_HIER \
        && (AXES(SEISMIC_AXIS_COEFF_GROUP) == 16 || AXES(SEISMIC_AXIS_COEFF_GROUP) == 32) \
        && (AXES(SEISMIC_AXIS_COEFF_LOCAL) == SEISMIC_CODES_UNSIGNED \
            || AXES(SEISMIC_AXIS_COEFF_LOCAL) == SEISMIC_CODES_TWOS \
            || AXES(SEISMIC_AXIS_COEFF_LOCAL) == SEISMIC_CODES_OFFSET \
            || AXES(SEISMIC_AXIS_COEFF_LOCAL) == SEISMIC_CODES_ODD) \
        && ((AXES(SEISMIC_AXIS_COEFF_CARRIED) == 0 \
                && PACKETS_FIELD_BITS(AXES) < 32 && PACKETS_FIELD_SPANS(PACKETS_FIELD_BITS(AXES))) \
            || (AXES(SEISMIC_AXIS_COEFF_CARRIED) != 0 && AXES(SEISMIC_AXIS_CODES) == SEISMIC_CODES_GRID \
                && AXES(SEISMIC_AXIS_COEFF_BIAS) == 0 \
                && 8 * AXES(SEISMIC_AXIS_COEFF_LOCAL_BITS) <= AXES(SEISMIC_AXIS_COEFF_GROUP)))))
#if !PACKETS_ALL(PACKETS_READS_COEFFICIENTS)
#error "packets: a weight slot's coefficients are of a structure this file does not read"
#endif

// A local coefficient code's value: unsigned, two's complement, offset, or
// odd (2c + 1).
float packets_local_value(const int kind, uint code) {
    const int local = PACKETS_AXIS(kind, SEISMIC_AXIS_COEFF_LOCAL);
    const uint bits = uint(PACKETS_AXIS(kind, SEISMIC_AXIS_COEFF_LOCAL_BITS));
    if (local == SEISMIC_CODES_TWOS)
        return float(int(code << (32u - bits)) >> (32u - bits));
    if (local == SEISMIC_CODES_OFFSET)
        return float(int(code) - PACKETS_AXIS(kind, SEISMIC_AXIS_COEFF_LOCAL_OFFSET));
    if (local == SEISMIC_CODES_ODD)
        return float(2u * code + 1u);
    return float(code);
}

// Packet `p`'s coefficients into `packet`. A local field is addressed by `p`
// itself, never as its group and offset added back together: NVIDIA's
// aarch64 compiler (GB10, driver 580) traps in pipeline creation folding such
// an add of a value's own split bit fields (`(p & ~7) + (p & 7)`, as an
// address or a value) in the GEMV, while `p` itself, or the fields joined
// with `|`, compile.
void packets_coefficients(const int kind, uint64_t row, packets_rows16 geometry, uint p, inout packets_packet packet) {
    const int coeff = PACKETS_AXIS(kind, SEISMIC_AXIS_COEFF), scale = PACKETS_AXIS(kind, SEISMIC_AXIS_COEFF_SCALE);
    const bool bias = PACKETS_AXIS(kind, SEISMIC_AXIS_COEFF_BIAS) != 0;
    const uint bytes = packets_scale_bytes(scale), groups = packets_groups(kind);
    const int code_exponent = PACKETS_AXIS(kind, SEISMIC_AXIS_CODES_EXPONENT);
#if PACKETS_ANY(SEISMIC_AXIS_COEFF, SEISMIC_COEFF_DIRECT)
    if (coeff == SEISMIC_COEFF_DIRECT) {
        // Packet p's first group, without a product that could wrap (which
        // keeps a compiler from folding it).
        const uint group = uint(PACKETS_AXIS(kind, SEISMIC_AXIS_COEFF_GROUP));
        const uint first = group < 32u ? p * (32u / group) : p / (group / 32u);
        if (groups == 2u && !bias) {
            // The packet's two scales are adjacent: one load.
            const vec2 stored = packets_scale_pair_at(scale, row + geometry.supers + uint64_t(first * bytes));
            const vec2 scales = code_exponent != 0 ? ldexp(stored, ivec2(code_exponent)) : stored;
            packet.scale0 = scales.x;
            packet.scale1 = scales.y;
            return;
        }
        [[unroll]] for (uint g = 0u; g < 2u; ++g) {
            if (g < groups) {
                // A group's scale, then its bias (a storage group is one
                // coefficient group).
                const uint64_t at = row + geometry.supers + uint64_t(first + g) * (bias ? 2u * bytes : bytes);
                const float stored = packets_scale_at(scale, at);
                packets_set_group(packet, g, code_exponent != 0 ? ldexp(stored, code_exponent) : stored, bias,
                    bias ? packets_scale_at(scale, at + uint64_t(bytes)) : 0.0);
            }
        }
        return;
    }
#endif
#if PACKETS_ANY(SEISMIC_AXIS_COEFF, SEISMIC_COEFF_HIER)
    if (coeff == SEISMIC_COEFF_HIER) {
        const uint per = uint(PACKETS_AXIS(kind, SEISMIC_AXIS_COEFF_FACTOR_GROUP)) / 32u; // packets a factor
        const uint64_t factors = row + geometry.supers + uint64_t(bias ? 2u * bytes : bytes) * uint64_t(p / per);
        const int exponent = PACKETS_AXIS(kind, SEISMIC_AXIS_COEFF_EXPONENT);
        // The scale factor and the minimum factor after it, one load.
        const vec2 stored = bias ? packets_scale_pair_at(scale, factors) : vec2(packets_scale_at(scale, factors), 0.0);
        const float factor = exponent + code_exponent != 0 ? ldexp(stored.x, exponent + code_exponent) : stored.x;
        const float minimum = bias ? float(PACKETS_AXIS(kind, SEISMIC_AXIS_COEFF_BIAS_SIGN))
            * (exponent != 0 ? ldexp(stored.y, exponent) : stored.y) : 0.0;
        // The packet's local field: per group a scale code, then its minimum
        // code, LOCAL_BITS each.
        const uint bits = uint(PACKETS_AXIS(kind, SEISMIC_AXIS_COEFF_LOCAL_BITS));
        const uint width = bits * (bias ? 2u : 1u), mask = (1u << bits) - 1u;
        uint field = 0u;
#if PACKETS_ANY(SEISMIC_AXIS_CODES, SEISMIC_CODES_GRID)
        if (PACKETS_AXIS(kind, SEISMIC_AXIS_COEFF_CARRIED) != 0) {
            // Carried in the entries: bit j of a group's code is its entry j's
            // carried bit (a packet's four entries, 4 / groups a group).
            const uint carried = uint(PACKETS_AXIS(kind, SEISMIC_AXIS_COEFF_CARRIED));
            [[unroll]] for (uint g = 0u; g < 2u; ++g) {
                if (g < groups) {
                    [[unroll]] for (uint j = 0u; j < 4u; ++j) {
                        if (j < bits)
                            field |= ((packets_grid_code(kind, packet, g * (4u / groups) + j) >> carried) & 1u)
                                << (g * width + j);
                    }
                }
            }
        } else
#endif
        {
            // Factor groups' fields follow one another, so the packet's field
            // is at bit `width groups` p of the plane, addressed by `p`
            // itself (see above): a 16-bit field is one aligned load, any
            // other the bytes its widest span reaches. A field starts at a
            // multiple of its width's lowest set bit (at most 8) within a
            // byte, so it reaches 8 less that bit past its width.
            const uint field_bits = width * groups;
            const uint bit = p * field_bits, shift = bit & 7u;
            const uint reach = field_bits + 8u - min(field_bits & (0u - field_bits), 8u);
            const uint64_t at = row + geometry.scales + uint64_t(bit >> 3);
            if (field_bits == 16u) {
                field = element_u16_at(at);
            } else {
                field = element_u8_at(at);
                if (reach > 8u)
                    field |= element_u8_at(at + 1ul) << 8;
                if (reach > 16u)
                    field |= element_u8_at(at + 2ul) << 16;
                if (reach > 24u)
                    field |= element_u8_at(at + 3ul) << 24;
                field >>= shift;
            }
        }
        [[unroll]] for (uint g = 0u; g < 2u; ++g) {
            if (g < groups) {
                packets_set_group(packet, g, factor * packets_local_value(kind, (field >> (g * width)) & mask), bias,
                    bias ? minimum * packets_local_value(kind, (field >> (g * width + bits)) & mask) : 0.0);
            }
        }
    }
#endif
}

// Packet `p` of the row at `row`; `k` is the row length (dense packets may be
// partial; packed kinds ignore it).
packets_packet packets_load(const int kind, uint64_t row, packets_rows16 geometry, uint p, uint k) {
    packets_packet packet;
    packet.low = uvec4(0u);
    packet.second = uvec4(0u);
    packet.high = uvec2(0u);
    packet.scale0 = 0.0;
    packet.scale1 = 0.0;
    packet.bias0 = 0.0;
#if PACKETS_SOME(PACKETS_TWO_BIASES)
    packet.bias1 = 0.0;
#endif
    packet.values = 0ul;
    packet.valid = 0u;
#if PACKETS_ANY_DENSE
    if (packets_dense(kind)) {
        packet.values = row + geometry.codes + uint64_t(p) * 32ul * ELEMENT_BYTES(packets_element(kind));
        packet.valid = min(32u, k - 32u * p);
        return packet;
    }
#endif
#if PACKETS_ANY_PACKED
    const uint low = uint(PACKETS_AXIS(kind, SEISMIC_AXIS_CODE_LOW_BITS));
    const uint high = uint(PACKETS_AXIS(kind, SEISMIC_AXIS_CODE_HIGH_BITS));
    // The packet's codes: 32 codes or 4 grid entries of `low` bits.
    const bool grid = PACKETS_AXIS(kind, SEISMIC_AXIS_CODES) == SEISMIC_CODES_GRID;
    const uint64_t codes = row + geometry.codes + uint64_t((grid ? 4u : 32u) * low / 8u) * p;
#if PACKETS_ANY(SEISMIC_AXIS_CODE_LOW_BITS, 2) || PACKETS_ANY(SEISMIC_AXIS_CODE_LOW_BITS, 16)
    // Packets of 8 bytes of codes: 32 codes of 2 bits or 4 grid entries of 16.
    if (low == 2u || low == 16u)
        packet.low = uvec4(element_uvec2_at(codes), 0u, 0u);
#endif
#if PACKETS_ANY(SEISMIC_AXIS_CODE_LOW_BITS, 24)
    if (low == 24u)
        packet.low = uvec4(element_u32_at(codes), element_u32_at(codes + 4ul), element_u32_at(codes + 8ul), 0u);
#endif
#if PACKETS_ANY(SEISMIC_AXIS_CODE_LOW_BITS, 4) || PACKETS_ANY(SEISMIC_AXIS_CODE_LOW_BITS, 8)
    if (low == 4u || low == 8u)
        packet.low = element_uvec4_at(codes);
#endif
#if PACKETS_ANY(SEISMIC_AXIS_CODE_LOW_BITS, 8)
    if (low == 8u)
        packet.second = element_uvec4_at(codes + 16ul);
#endif
#if PACKETS_SOME(PACKETS_GRID_HIGH)
    // Grid entries' high bits: 4 `high` bits at bit 4 `high` p of the row's
    // high plane, the bytes its widest span reaches; the high plane holds no
    // codes' high bits then.
    if (grid && high != 0u) {
        const uint bits = 4u * high, bit = bits * p, shift = bit & 7u;
        const uint64_t at = row + geometry.high + uint64_t(bit >> 3);
        uint field = element_u8_at(at);
        if (shift + bits > 8u)
            field |= element_u8_at(at + 1ul) << 8;
        if (shift + bits > 16u)
            field |= element_u8_at(at + 2ul) << 16;
        if (shift + bits > 24u)
            field |= element_u8_at(at + 3ul) << 24;
        packet.high.x = (field >> shift) & ((1u << bits) - 1u);
    }
#define PACKETS_CODES_HIGH(bits) (!grid && high == (bits))
#else
#define PACKETS_CODES_HIGH(bits) (high == (bits))
#endif
#if PACKETS_ANY(SEISMIC_AXIS_CODE_HIGH_BITS, 1)
    if (PACKETS_CODES_HIGH(1u))
        packet.high.x = element_u32_at(row + geometry.high + 4ul * p);
#endif
#if PACKETS_ANY(SEISMIC_AXIS_CODE_HIGH_BITS, 2)
    if (PACKETS_CODES_HIGH(2u))
        packet.high = element_uvec2_at(row + geometry.high + 8ul * p);
#endif
#undef PACKETS_CODES_HIGH
    packets_coefficients(kind, row, geometry, p, packet);
#endif
    return packet;
}

// Element `i` of a dense packet (zero past the row).
float packets_dense_element(const int kind, packets_packet packet, uint i) {
    return i < packet.valid ? element_at(packets_element(kind), packet.values, i) : 0.0;
}

// The codes of sub-step `step` (elements 8 step ..) as (0,2,4,6), (1,3,5,7),
// offset codes less their offset where `centred` (a dense packet's are its
// values).
void packets_decode(const int kind, packets_packet packet, uint step, const bool centred, out vec4 even,
    out vec4 odd) {
#if PACKETS_ANY_DENSE
    if (packets_dense(kind)) {
        const uint first = 8u * step;
        if (first + 8u <= packet.valid) {
            element_load8(packets_element(kind), packet.values, first, even, odd);
        } else {
            even = vec4(packets_dense_element(kind, packet, first), packets_dense_element(kind, packet, first + 2u),
                packets_dense_element(kind, packet, first + 4u), packets_dense_element(kind, packet, first + 6u));
            odd = vec4(packets_dense_element(kind, packet, first + 1u), packets_dense_element(kind, packet, first + 3u),
                packets_dense_element(kind, packet, first + 5u), packets_dense_element(kind, packet, first + 7u));
        }
        return;
    }
#endif
#if PACKETS_ANY_PACKED
#if PACKETS_ANY(SEISMIC_AXIS_CODES, SEISMIC_CODES_GRID)
    if (PACKETS_AXIS(kind, SEISMIC_AXIS_CODES) == SEISMIC_CODES_GRID) {
        const uint entry = packets_grid_code(kind, packet, step);
        packets_grid_codes(packets_grid_point(kind, entry), packets_grid_signs(kind, entry), even, odd);
        return;
    }
#endif
#if PACKETS_ANY(SEISMIC_AXIS_CODE_LOW_BITS, 8)
    if (PACKETS_AXIS(kind, SEISMIC_AXIS_CODE_LOW_BITS) == 8) {
        const uvec4 words = step < 2u ? packet.low : packet.second;
        const uint base = (step & 1u) * 2u;
        const vec4 a = packets_interpret(kind, words[base], centred);
        const vec4 b = packets_interpret(kind, words[base + 1u], centred);
        even = vec4(a.x, a.z, b.x, b.z);
        odd = vec4(a.y, a.w, b.y, b.w);
        return;
    }
#endif
#if PACKETS_ANY(SEISMIC_AXIS_CODE_LOW_BITS, 2)
    if (PACKETS_AXIS(kind, SEISMIC_AXIS_CODE_LOW_BITS) == 2) {
        // Step `step`'s eight codes: bits 16 (step % 2).. of word step / 2,
        // and their high bits at bit 2.
        uint e, o;
        packets_split_crumbs((packet.low[step >> 1] >> (16u * (step & 1u))) & 0xffffu, e, o);
#if PACKETS_ANY(SEISMIC_AXIS_CODE_HIGH_BITS, 1)
        if (PACKETS_AXIS(kind, SEISMIC_AXIS_CODE_HIGH_BITS) == 1) {
            uint he, ho;
            packets_split_high1((packet.high.x >> (8u * step)) & 0xffu, he, ho);
            e |= he >> 2u;
            o |= ho >> 2u;
        }
#endif
        even = packets_interpret(kind, e, centred);
        odd = packets_interpret(kind, o, centred);
        return;
    }
#endif
    const uint word = packet.low[step];
    uint he = 0u, ho = 0u;
#if PACKETS_ANY(SEISMIC_AXIS_CODE_HIGH_BITS, 1)
    if (PACKETS_AXIS(kind, SEISMIC_AXIS_CODE_HIGH_BITS) == 1)
        packets_split_high1((packet.high.x >> (8u * step)) & 0xffu, he, ho);
#endif
#if PACKETS_ANY(SEISMIC_AXIS_CODE_HIGH_BITS, 2)
    if (PACKETS_AXIS(kind, SEISMIC_AXIS_CODE_HIGH_BITS) == 2)
        packets_split_high2((packet.high[step >> 1] >> (16u * (step & 1u))) & 0xffffu, he, ho);
#endif
    even = packets_interpret(kind, (word & 0x0f0f0f0fu) | he, centred);
    odd = packets_interpret(kind, ((word >> 4) & 0x0f0f0f0fu) | ho, centred);
#endif
}

// The codes of sub-step `step`, for factored dot products (`packets_bias`).
void packets_codes(const int kind, packets_packet packet, uint step, out vec4 even, out vec4 odd) {
    packets_decode(kind, packet, step, false, even, odd);
}

float packets_scale(const int kind, packets_packet packet, uint step) {
    if (packets_groups(kind) == 2u)
        return packets_group_scale(packet, step >> 1);
    return packets_dense(kind) ? 1.0 : packet.scale0;
}

// The bias of coefficient group `group` in factored dot products,
// scale * dot(code, x) + bias * sum(x): its minimum or stored bias, or an
// offset code's exact bias -(offset * scale).
float packets_bias(const int kind, packets_packet packet, uint group) {
    if (PACKETS_AXIS(kind, SEISMIC_AXIS_CODES) == SEISMIC_CODES_OFFSET)
        return -(float(PACKETS_AXIS(kind, SEISMIC_AXIS_CODES_OFFSET)) * packets_group_scale(packet, group));
    return PACKETS_AXIS(kind, SEISMIC_AXIS_COEFF_BIAS) != 0 ? packets_group_bias(packet, group) : 0.0;
}

// The logical value of centred code `code` of sub-step `step`: scale *
// (code - offset), the exact integer difference (decoded so) under one
// multiply, with a stored bias by one fma (a dense packet's code is its
// value).
float packets_value(const int kind, packets_packet packet, uint step, float code) {
    if (packets_dense(kind))
        return code;
    const float scale = packets_scale(kind, packet, step);
    if (PACKETS_AXIS(kind, SEISMIC_AXIS_COEFF_BIAS) != 0)
        return seismic_fma_rn(scale, code, packets_group_bias(packet, packets_groups(kind) == 2u ? step >> 1 : 0u));
    return scale * code;
}

// The values of sub-step `step` as (0,2,4,6), (1,3,5,7), as product
// operands: scale * code with one rounding, plus by the same fma a stored
// bias or an offset code's exact bias (`packets_bias`), so a value of zero
// is +0. Codes are not centred: the offset costs nothing past the fma.
void packets_values(const int kind, packets_packet packet, uint step, out vec4 even, out vec4 odd) {
    packets_decode(kind, packet, step, false, even, odd);
    if (packets_dense(kind))
        return;
    const float scale = packets_scale(kind, packet, step);
    if (PACKETS_AXIS(kind, SEISMIC_AXIS_COEFF_BIAS) != 0
        || PACKETS_AXIS(kind, SEISMIC_AXIS_CODES) == SEISMIC_CODES_OFFSET) {
        const float bias = packets_bias(kind, packet, packets_groups(kind) == 2u ? step >> 1 : 0u);
        [[unroll]] for (uint i = 0u; i < 4u; ++i) {
            even[i] = seismic_fma_rn(scale, even[i], bias);
            odd[i] = seismic_fma_rn(scale, odd[i], bias);
        }
    } else {
        even *= scale;
        odd *= scale;
    }
}

// The logical values of sub-step `step` exactly, as a gather publishes
// them (`packets_value`): a zero keeps the sign of scale * 0.
void packets_exact_values(const int kind, packets_packet packet, uint step, out vec4 even, out vec4 odd) {
    packets_decode(kind, packet, step, true, even, odd);
    [[unroll]] for (uint i = 0u; i < 4u; ++i) {
        even[i] = packets_value(kind, packet, step, even[i]);
        odd[i] = packets_value(kind, packet, step, odd[i]);
    }
}

// One decoded logical weight value (used by gathers such as the embedding):
// its centred code, then its value.
float packets_value_at(const int kind, packets_packet packet, uint i) {
    vec4 even, odd;
    packets_decode(kind, packet, i >> 3, true, even, odd);
    const uint lane = (i & 7u) >> 1;
    return packets_value(kind, packet, i >> 3, (i & 1u) != 0u ? odd[lane] : even[lane]);
}
