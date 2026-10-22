// Seismic native library (Vulkan): packet decoders for weights stored in the
// `rows16` layout (the Vulkan resident layout). `#include <seismic/packets.glsl>`.
// The counterpart of <seismic/packets.h> (Metal).
//
// A packet is 32 consecutive logical elements of one weight row. In `rows16`
// a q4k/q5k/q6k packet is one 16-byte load of its low-nibble code plane
// (codes 2i/2i+1 in the low/high nibble of byte i), plus its slice of the
// high-bit plane (q5k: one u32, q6k: one uvec2), its local coefficients and
// its super factors; a q8 packet is two 16-byte loads of int8 codes; a
// vector-coded (iq2xs, iq3xxs, iq2xxs) packet is its four grid entries, one a
// sub-step, decoded through the grid. A packet is consumed in four
// 8-element sub-steps. Each sub-step decodes its codes as
// exact small integers in two vec4s, `even` (elements 0,2,4,6) and `odd`
// (1,3,5,7), by splitting code bytes on nibble boundaries: every decoded code
// is an exact integer, so the decode has no rounding or subnormal case.
//
// A packet's logical value is `scale(g) * code + bias(g)` for its coefficient
// group g. Dot products are factored: `scale * dot(code, x) + bias * sum(x)`,
// where the activation sums per 16 elements are computed once per staged
// activation and reused by every weight row.
//
// GLSL has no templates: a weight's packet type is a constant kind
// (`PACKETS_Q4K`, ...) passed as the first argument, and one `packets_packet`
// struct holds the fields of every kind (a kind leaves the others unused).
// Weight tensors are bound to slots before this file is included, as on
// Metal and CUDA:
//     #define KERNEL_W0 SEISMIC_GATE_WEIGHT
// which defines `PACKETS_W0` (the kind of the bound representation),
// `PACKETS_W0_ROWS` (its `packets_rows16`, a dense tensor's rows its axis 0)
// and `PACKETS_W0_STACKED_ROWS` (that of one matrix of a stacked [B, N, K]
// tensor, such as one expert's). Slots W0..W3 exist. Code that loads or
// decodes a kind, here and in the libraries over this file, is a
// preprocessor group of the bound slots' kinds (`PACKETS_ANY`), so an
// entry's pipelines hold its own kinds' paths alone: drivers compile code a
// constant makes dead at full cost. RADV took 9.8 s and 1.96 GB to create a
// projection GEMM pipeline staging every kind instead of 0.5 s and 213 MB
// (seismic/hardware/rdna3-vulkan.md), and an RTX 3090's NVIDIA compiler
// 4.4 s instead of 2.4 s for one staging two.
//
// This file is independent of any entry ABI.
#include <seismic/element.glsl>
#include <seismic/grids.glsl>

// Kinds start at 1: 0 is the kind of a representation this file has none
// for (an undefined `SEISMIC_REPRESENTATION_*` is 0 in a preprocessor
// condition).
#define PACKETS_Q4K 1
#define PACKETS_Q5K 2
#define PACKETS_Q6K 3
#define PACKETS_Q8 4
#define PACKETS_BF16 5
#define PACKETS_F16 6
#define PACKETS_F32 7
#define PACKETS_IQ4 8
// The rest of the 4-bit coded family (iq4g32 is PACKETS_IQ4) and q5g32.
#define PACKETS_Q4G32S 9
#define PACKETS_MXFP4 10
#define PACKETS_NVFP4 11
#define PACKETS_Q5G32S 12
#define PACKETS_Q5G32 13
// The vector-coded kinds.
#define PACKETS_IQ2XS 14
#define PACKETS_IQ3XXS 15
// IQ4_NL blocks of a row whose width is not a multiple of 256 (iq4g32s): the
// IQ4 table with one f16 scale per 32.
#define PACKETS_IQ4S 16
// The vector-coded IQ2_XXS kind.
#define PACKETS_IQ2XXS 17

#define PACKETS_BIT(K) (1 << (K))

// The kind of each representation, by the name the ABI gives it
// (`SEISMIC_REPRESENTATION_<NAME>`, which a tensor's prefix `<T>()` expands
// to without pasting, so a preprocessor condition can test it), and the row
// geometry of each, by its pasted ABI name (`<T>_REPRESENTATION`). A packed
// tensor in its packet layout has no rows, so its prefix names no kind here.
#define SEISMIC_REPRESENTATION_Q4K PACKETS_Q4K
#define SEISMIC_REPRESENTATION_Q5K PACKETS_Q5K
#define SEISMIC_REPRESENTATION_Q6K PACKETS_Q6K
#define SEISMIC_REPRESENTATION_Q8G32S PACKETS_Q8
#define SEISMIC_REPRESENTATION_BF16 PACKETS_BF16
#define SEISMIC_REPRESENTATION_F16 PACKETS_F16
#define SEISMIC_REPRESENTATION_F32 PACKETS_F32
#define SEISMIC_REPRESENTATION_IQ4G32 PACKETS_IQ4
#define SEISMIC_REPRESENTATION_Q4G32S PACKETS_Q4G32S
#define SEISMIC_REPRESENTATION_MXFP4G32 PACKETS_MXFP4
#define SEISMIC_REPRESENTATION_NVFP4G16 PACKETS_NVFP4
#define SEISMIC_REPRESENTATION_Q5G32S PACKETS_Q5G32S
#define SEISMIC_REPRESENTATION_Q5G32 PACKETS_Q5G32
#define SEISMIC_REPRESENTATION_IQ2XS PACKETS_IQ2XS
#define SEISMIC_REPRESENTATION_IQ3XXS PACKETS_IQ3XXS
#define SEISMIC_REPRESENTATION_IQ4G32S PACKETS_IQ4S
#define SEISMIC_REPRESENTATION_IQ2XXS PACKETS_IQ2XXS
#define PACKETS_ROWS_OF_Q4K(T) packets_rows16(T##_ROW_STRIDE_BYTES, T##_PLANE_CODES_LO_ROW_OFFSET, 0ul, T##_PLANE_SCALES_ROW_OFFSET, T##_PLANE_SUPERS_ROW_OFFSET)
#define PACKETS_ROWS_OF_Q5K(T) PACKETS_ROWS_HIGH(T)
#define PACKETS_ROWS_OF_Q6K(T) PACKETS_ROWS_HIGH(T)
#define PACKETS_ROWS_OF_Q8G32S(T) packets_rows16(T##_ROW_STRIDE_BYTES, T##_PLANE_CODES_ROW_OFFSET, 0ul, 0ul, T##_PLANE_SUPERS_ROW_OFFSET)
#define PACKETS_ROWS_OF_IQ4G32(T) PACKETS_ROWS_C4(T)
#define PACKETS_ROWS_OF_Q4G32S(T) PACKETS_ROWS_C4(T)
#define PACKETS_ROWS_OF_MXFP4G32(T) PACKETS_ROWS_C4(T)
#define PACKETS_ROWS_OF_NVFP4G16(T) PACKETS_ROWS_C4(T)
#define PACKETS_ROWS_OF_Q5G32S(T) PACKETS_ROWS_Q5(T)
#define PACKETS_ROWS_OF_Q5G32(T) PACKETS_ROWS_Q5(T)
#define PACKETS_ROWS_OF_IQ2XS(T) PACKETS_ROWS_GRID(T)
#define PACKETS_ROWS_OF_IQ3XXS(T) PACKETS_ROWS_GRID(T)
#define PACKETS_ROWS_OF_IQ4G32S(T) PACKETS_ROWS_C4(T)
// iq2xxs: its entries carry the scale codes; no scales plane.
#define PACKETS_ROWS_OF_IQ2XXS(T) packets_rows16(T##_ROW_STRIDE_BYTES, T##_PLANE_GRIDS_ROW_OFFSET, 0ul, 0ul, T##_PLANE_SUPERS_ROW_OFFSET)
#define PACKETS_STACKED_ROWS_OF_Q4K(T) PACKETS_ROWS_OF_Q4K(T)
#define PACKETS_STACKED_ROWS_OF_Q5K(T) PACKETS_ROWS_OF_Q5K(T)
#define PACKETS_STACKED_ROWS_OF_Q6K(T) PACKETS_ROWS_OF_Q6K(T)
#define PACKETS_STACKED_ROWS_OF_Q8G32S(T) PACKETS_ROWS_OF_Q8G32S(T)
#define PACKETS_STACKED_ROWS_OF_IQ4G32(T) PACKETS_ROWS_OF_IQ4G32(T)
#define PACKETS_STACKED_ROWS_OF_Q4G32S(T) PACKETS_ROWS_OF_Q4G32S(T)
#define PACKETS_STACKED_ROWS_OF_MXFP4G32(T) PACKETS_ROWS_OF_MXFP4G32(T)
#define PACKETS_STACKED_ROWS_OF_NVFP4G16(T) PACKETS_ROWS_OF_NVFP4G16(T)
#define PACKETS_STACKED_ROWS_OF_Q5G32S(T) PACKETS_ROWS_OF_Q5G32S(T)
#define PACKETS_STACKED_ROWS_OF_Q5G32(T) PACKETS_ROWS_OF_Q5G32(T)
#define PACKETS_STACKED_ROWS_OF_IQ2XS(T) PACKETS_ROWS_OF_IQ2XS(T)
#define PACKETS_STACKED_ROWS_OF_IQ3XXS(T) PACKETS_ROWS_OF_IQ3XXS(T)
#define PACKETS_STACKED_ROWS_OF_IQ4G32S(T) PACKETS_ROWS_OF_IQ4G32S(T)
#define PACKETS_STACKED_ROWS_OF_IQ2XXS(T) PACKETS_ROWS_OF_IQ2XXS(T)
#define PACKETS_ROWS_HIGH(T) packets_rows16(T##_ROW_STRIDE_BYTES, T##_PLANE_CODES_LO_ROW_OFFSET, T##_PLANE_CODES_HI_ROW_OFFSET, T##_PLANE_SCALES_ROW_OFFSET, T##_PLANE_SUPERS_ROW_OFFSET)
#define PACKETS_ROWS_C4(T) packets_rows16(T##_ROW_STRIDE_BYTES, T##_PLANE_CODES_LO_ROW_OFFSET, 0ul, 0ul, T##_PLANE_SUPERS_ROW_OFFSET)
#define PACKETS_ROWS_Q5(T) packets_rows16(T##_ROW_STRIDE_BYTES, T##_PLANE_CODES_LO_ROW_OFFSET, T##_PLANE_CODES_HI_ROW_OFFSET, 0ul, T##_PLANE_SUPERS_ROW_OFFSET)
#define PACKETS_ROWS_GRID(T) packets_rows16(T##_ROW_STRIDE_BYTES, T##_PLANE_GRIDS_ROW_OFFSET, 0ul, T##_PLANE_SCALES_ROW_OFFSET, T##_PLANE_SUPERS_ROW_OFFSET)
// A dense tensor's rows are its axis 0, or its axis 1 for one matrix of a
// stack, each of its elements' bytes; a row layout's rows are the same
// either way.
#define PACKETS_ROWS_OF_BF16(T) PACKETS_ROWS_DENSE(T, 0)
#define PACKETS_ROWS_OF_F16(T) PACKETS_ROWS_DENSE(T, 0)
#define PACKETS_ROWS_OF_F32(T) PACKETS_ROWS_DENSE(T, 0)
#define PACKETS_STACKED_ROWS_OF_BF16(T) PACKETS_ROWS_DENSE(T, 1)
#define PACKETS_STACKED_ROWS_OF_F16(T) PACKETS_ROWS_DENSE(T, 1)
#define PACKETS_STACKED_ROWS_OF_F32(T) PACKETS_ROWS_DENSE(T, 1)
#define PACKETS_ROWS_DENSE(T, AXIS) packets_rows16(T##_STRIDE_##AXIS * uint64_t(T##_PACKET_SIZE), 0ul, 0ul, 0ul, 0ul)
#define PACKETS_CAT_(a, b) a##b
#define PACKETS_CAT(a, b) PACKETS_CAT_(a, b)
// The `packets_rows16` and stacked `packets_rows16` of the tensor of ABI
// prefix name T (a slot, which names it, expands first).
#define PACKETS_ROWS(T) PACKETS_ROWS_(T)
#define PACKETS_STACKED_ROWS(T) PACKETS_STACKED_ROWS_(T)
#define PACKETS_ROWS_(T) PACKETS_CAT(PACKETS_ROWS_OF_, T##_REPRESENTATION)(T)
#define PACKETS_STACKED_ROWS_(T) PACKETS_CAT(PACKETS_STACKED_ROWS_OF_, T##_REPRESENTATION)(T)

// Slot bindings. A slot whose representation has no packet kind is
// rejected and holds no kind, so later conditions stay evaluable.
#if defined(KERNEL_W0)
#define PACKETS_W0 KERNEL_W0()
#if PACKETS_W0 == 0
#error "weight slot W0 binds a representation without a packet kind"
#define PACKETS_W0_BIT 0
#else
#define PACKETS_W0_ROWS PACKETS_ROWS(KERNEL_W0)
#define PACKETS_W0_STACKED_ROWS PACKETS_STACKED_ROWS(KERNEL_W0)
#define PACKETS_W0_BIT PACKETS_BIT(PACKETS_W0)
#endif
#else
#define PACKETS_W0_BIT 0
#endif
#if defined(KERNEL_W1)
#define PACKETS_W1 KERNEL_W1()
#if PACKETS_W1 == 0
#error "weight slot W1 binds a representation without a packet kind"
#define PACKETS_W1_BIT 0
#else
#define PACKETS_W1_ROWS PACKETS_ROWS(KERNEL_W1)
#define PACKETS_W1_STACKED_ROWS PACKETS_STACKED_ROWS(KERNEL_W1)
#define PACKETS_W1_BIT PACKETS_BIT(PACKETS_W1)
#endif
#else
#define PACKETS_W1_BIT 0
#endif
#if defined(KERNEL_W2)
#define PACKETS_W2 KERNEL_W2()
#if PACKETS_W2 == 0
#error "weight slot W2 binds a representation without a packet kind"
#define PACKETS_W2_BIT 0
#else
#define PACKETS_W2_ROWS PACKETS_ROWS(KERNEL_W2)
#define PACKETS_W2_STACKED_ROWS PACKETS_STACKED_ROWS(KERNEL_W2)
#define PACKETS_W2_BIT PACKETS_BIT(PACKETS_W2)
#endif
#else
#define PACKETS_W2_BIT 0
#endif
#if defined(KERNEL_W3)
#define PACKETS_W3 KERNEL_W3()
#if PACKETS_W3 == 0
#error "weight slot W3 binds a representation without a packet kind"
#define PACKETS_W3_BIT 0
#else
#define PACKETS_W3_ROWS PACKETS_ROWS(KERNEL_W3)
#define PACKETS_W3_STACKED_ROWS PACKETS_STACKED_ROWS(KERNEL_W3)
#define PACKETS_W3_BIT PACKETS_BIT(PACKETS_W3)
#endif
#else
#define PACKETS_W3_BIT 0
#endif
// The kinds of the bound slots.
#define PACKETS_KINDS (PACKETS_W0_BIT | PACKETS_W1_BIT | PACKETS_W2_BIT | PACKETS_W3_BIT)
// Whether a bound slot holds a kind of `MASK`, or kind `K`: preprocessor
// conditions.
#define PACKETS_ANY(MASK) ((PACKETS_KINDS & (MASK)) != 0)
#define PACKETS_HOLDS(K) PACKETS_ANY(PACKETS_BIT(K))

// Kinds that share a decoding.
#define PACKETS_IQ4_TABLE_KINDS (PACKETS_BIT(PACKETS_IQ4) | PACKETS_BIT(PACKETS_IQ4S))
#define PACKETS_E2M1_KINDS (PACKETS_BIT(PACKETS_MXFP4) | PACKETS_BIT(PACKETS_NVFP4))
#define PACKETS_TABLE_KINDS (PACKETS_IQ4_TABLE_KINDS | PACKETS_E2M1_KINDS)
#define PACKETS_GRID_KINDS (PACKETS_BIT(PACKETS_IQ2XS) | PACKETS_BIT(PACKETS_IQ3XXS) | PACKETS_BIT(PACKETS_IQ2XXS))
#define PACKETS_KQUANT4_KINDS (PACKETS_BIT(PACKETS_Q4K) | PACKETS_BIT(PACKETS_Q5K))
#define PACKETS_NIBBLE_KINDS (PACKETS_BIT(PACKETS_Q4K) | PACKETS_BIT(PACKETS_Q4G32S))
#define PACKETS_HIGH1_KINDS (PACKETS_BIT(PACKETS_Q5K) | PACKETS_BIT(PACKETS_Q5G32S) | PACKETS_BIT(PACKETS_Q5G32))
#define PACKETS_Q5G32_KINDS (PACKETS_BIT(PACKETS_Q5G32S) | PACKETS_BIT(PACKETS_Q5G32))
#define PACKETS_BIASED_KINDS (PACKETS_BIT(PACKETS_Q4K) | PACKETS_BIT(PACKETS_Q5K) | PACKETS_BIT(PACKETS_Q6K) \
    | PACKETS_BIT(PACKETS_Q4G32S) | PACKETS_Q5G32_KINDS)
#define PACKETS_TWO_GROUP_KINDS (PACKETS_BIT(PACKETS_Q6K) | PACKETS_BIT(PACKETS_NVFP4) | PACKETS_BIT(PACKETS_IQ2XS))
#define PACKETS_DENSE_KINDS (PACKETS_BIT(PACKETS_BF16) | PACKETS_BIT(PACKETS_F16) | PACKETS_BIT(PACKETS_F32))

// Whether `kind` is one of `mask`, or `candidate`, that a bound slot holds.
// Value tests select between expressions with these; code that loads or
// decodes a kind's packets sits in a preprocessor group of its condition
// (`#if PACKETS_ANY(...)`), so a pipeline never holds a kind no slot binds.
bool packets_in(const int kind, const int mask) {
    return (PACKETS_KINDS & mask & PACKETS_BIT(kind)) != 0;
}

bool packets_is(const int kind, const int candidate) {
    return packets_in(kind, PACKETS_BIT(candidate));
}

// Coefficient groups per packet (q6k, nvfp4g16, iq2xs: one per 16 elements).
uint packets_groups(const int kind) {
    return packets_in(kind, PACKETS_TWO_GROUP_KINDS) ? 2u : 1u;
}

// Vector codes through a grid.
bool packets_grid(const int kind) {
    return packets_in(kind, PACKETS_GRID_KINDS);
}

// The kind has a per-group bias term (the factored `bias * sum(x)`).
bool packets_biased(const int kind) {
    return packets_in(kind, PACKETS_BIASED_KINDS);
}

// Nibble codes with a fifth bit in the high plane.
bool packets_high1(const int kind) {
    return packets_in(kind, PACKETS_HIGH1_KINDS);
}

// Dense weights: the element kind of their storage.
bool packets_dense(const int kind) {
    return packets_in(kind, PACKETS_DENSE_KINDS);
}

int packets_element(const int kind) {
    return packets_is(kind, PACKETS_BF16) ? ELEMENT_BF16 : packets_is(kind, PACKETS_F16) ? ELEMENT_F16 : ELEMENT_F32;
}

// ---------------------------------------------------------------------------
// Weight row geometry. `stride` is the byte distance between rows; plane
// offsets are within a row. Planes a representation lacks are unused.

struct packets_rows16 {
    uint64_t stride;
    uint64_t codes;   // codes_lo (k-quants), codes (q8), grids or the values (dense)
    uint64_t high;    // codes_hi (q5k, q6k)
    uint64_t scales;  // packed local coefficients (k-quants)
    uint64_t supers;  // super factors (k-quants) or group scales (q8)
};

// One loaded packet of any kind.
struct packets_packet {
    uvec4 low;       // codes_lo (k-quants), the first 16 codes (q8) or the grid entries
    uvec4 second;    // the last 16 codes (q8)
    uvec2 high;      // codes_hi: q5k in .x, q6k in .xy
    float scale0;
    float scale1;    // q6k second group
    float bias;      // q4k/q5k
    uint64_t values; // dense: the packet's first element
    uint valid;      // dense: elements of this packet inside the row
};

// Bytes 0,2,4,6 / 1,3,5,7 of a 16-bit field of 2-bit high codes, moved to
// bits 4..5 of four bytes.
void packets_split_high2(uint field, out uint even, out uint odd) {
    uint t = (field | (field << 8)) & 0x00ff00ffu;
    t = (t | (t << 4)) & 0x0f0f0f0fu;
    even = (t & 0x03030303u) << 4;
    odd = (t & 0x0c0c0c0cu) << 2;
}

// The same for an 8-bit field of 1-bit high codes, moved to bit 4.
void packets_split_high1(uint field, out uint even, out uint odd) {
    uint t = (field | (field << 12)) & 0x000f000fu;
    t = (t | (t << 6)) & 0x03030303u;
    even = (t & 0x01010101u) << 4;
    odd = (t & 0x02020202u) << 3;
}

vec4 packets_unsigned_bytes(uint word) { return vec4(unpack8(word)); }
vec4 packets_signed_bytes(uint word) { return vec4(unpack8(int(word))); }

// Registry iq4g32 code table, shared by GGUF IQ4_XS and IQ4_NL imports.
float packets_iq4_code(uint code) {
    const int values[16] = int[16](-127, -104, -83, -65, -49, -35, -22, -10,
        1, 13, 25, 38, 53, 69, 89, 113);
    return float(values[code]);
}

// E2M1 doubled (GGUF kvalues_mxfp4), shared by mxfp4g32 and nvfp4g16; their
// scales carry the compensating factor 1/2.
float packets_e2m1_code(uint code) {
    const int values[16] = int[16](0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    return float(values[code]);
}

// The eight table values of one step word's codes, as (0,2,4,6), (1,3,5,7).
void packets_table_codes(const int kind, uint word, out vec4 even, out vec4 odd) {
#if PACKETS_ANY(PACKETS_IQ4_TABLE_KINDS)
    if (packets_in(kind, PACKETS_IQ4_TABLE_KINDS)) {
        even = vec4(packets_iq4_code(word & 15u), packets_iq4_code((word >> 8u) & 15u),
            packets_iq4_code((word >> 16u) & 15u), packets_iq4_code((word >> 24u) & 15u));
        odd = vec4(packets_iq4_code((word >> 4u) & 15u), packets_iq4_code((word >> 12u) & 15u),
            packets_iq4_code((word >> 20u) & 15u), packets_iq4_code((word >> 28u) & 15u));
        return;
    }
#endif
#if PACKETS_ANY(PACKETS_E2M1_KINDS)
    if (packets_in(kind, PACKETS_E2M1_KINDS)) {
        even = vec4(packets_e2m1_code(word & 15u), packets_e2m1_code((word >> 8u) & 15u),
            packets_e2m1_code((word >> 16u) & 15u), packets_e2m1_code((word >> 24u) & 15u));
        odd = vec4(packets_e2m1_code((word >> 4u) & 15u), packets_e2m1_code((word >> 12u) & 15u),
            packets_e2m1_code((word >> 20u) & 15u), packets_e2m1_code((word >> 28u) & 15u));
    }
#endif
}

// E8M0 2^(e - 127) halved: 2^(e - 128) (e < 2 are F32 subnormals); 0xff is NaN.
float packets_e8m0_half(uint e) {
    return uintBitsToFloat(e < 2u ? 0x00200000u << e : e == 255u ? 0x7fc00000u : (e - 1u) << 23);
}

// UE4M3 (sign bit ignored; 0x7f is NaN) halved: (1 + m/8) 2^(e - 8), or
// m 2^-10 when e = 0.
float packets_ue4m3_half(uint raw) {
    raw &= 0x7fu;
    const uint e = raw >> 3, m = raw & 7u;
    const float normal = uintBitsToFloat(((e + 119u) << 23) | (m << 20));
    return raw == 0x7fu ? uintBitsToFloat(0x7fc00000u) : e == 0u ? float(m) * 0.0009765625 : normal;
}

// Vector codes (the registry's grid codes: iq2xs, iq3xxs, iq2xxs). An entry
// of the grids plane codes eight values: grid indices name their magnitudes
// (<seismic/grids.glsl>), and seven sign bits, completed by an eighth of even
// parity, their signs. Every 16 (iq2xs) or 32 (iq3xxs, iq2xxs) values have
// an odd scale code 2s + 1 and every 256 one factor: value = factor *
// (2s + 1) * code, every product exact. iq2xs and iq3xxs keep s in the
// scales plane under an f32 factor; iq2xxs carries s in its entries (bit j
// of s is bit 15 of entry j) under an f16 d, the factor d / 8.

// Entry `step` of a packet's entries: iq2xs sixteen-bit entries (a 9-bit
// index into the 512-point grid, then the signs), iq2xxs sixteen-bit entries
// (an 8-bit index into the 256-point grid, then the signs, then a scale code
// bit), iq3xxs
// three-byte entries (two 8-bit indices into the 256-point grid of four
// values, then the signs). Its grid point (lanes 0..3 in x, 4..7 in y) and
// its sign bits.
void packets_grid_entry(const int kind, uvec4 entries, uint step, out uvec2 point, out uint signs) {
#if PACKETS_HOLDS(PACKETS_IQ2XS)
    if (packets_is(kind, PACKETS_IQ2XS)) {
        const uint entry = (entries[step >> 1] >> (16u * (step & 1u))) & 0xffffu;
        point = grids_iq2xs[entry & 511u];
        signs = entry >> 9;
        return;
    }
#endif
#if PACKETS_HOLDS(PACKETS_IQ2XXS)
    if (packets_is(kind, PACKETS_IQ2XXS)) {
        const uint entry = (entries[step >> 1] >> (16u * (step & 1u))) & 0xffffu;
        point = grids_iq2xxs[entry & 255u];
        signs = (entry >> 8) & 127u;
        return;
    }
#endif
#if PACKETS_HOLDS(PACKETS_IQ3XXS)
    if (packets_is(kind, PACKETS_IQ3XXS)) {
        const uint word = (3u * step) >> 2, shift = (24u * step) & 31u;
        const uint spill = shift > 8u ? entries[word + 1u] << (32u - shift) : 0u;
        const uint entry = ((entries[word] >> shift) | spill) & 0xffffffu;
        point = uvec2(grids_iq3xxs[entry & 255u], grids_iq3xxs[(entry >> 8) & 255u]);
        signs = (entry >> 16) & 127u;
    }
#endif
}

// The eight signed codes of `point` under sign bits `signs` (the eighth makes
// the count of set bits even), as (0,2,4,6), (1,3,5,7).
void packets_grid_codes(uvec2 point, uint signs, out vec4 even, out vec4 odd) {
    signs |= (uint(bitCount(signs)) & 1u) << 7;
    const vec4 low = packets_unsigned_bytes(point.x), high = packets_unsigned_bytes(point.y);
    even = uintBitsToFloat(floatBitsToUint(vec4(low.x, low.z, high.x, high.z)) ^
                           (((uvec4(signs) >> uvec4(0u, 2u, 4u, 6u)) & 1u) << 31));
    odd = uintBitsToFloat(floatBitsToUint(vec4(low.y, low.w, high.y, high.w)) ^
                          (((uvec4(signs) >> uvec4(1u, 3u, 5u, 7u)) & 1u) << 31));
}

// The (scale6, min6) pair of k-quant packet `p` and its super factors. The
// pair's 12 bits sit at bit 12 p of the scale plane (twelve bytes a super
// block of eight packets): from byte (3 p) >> 1, at the nibble p & 1. Both
// are written over `p` itself, as the grid scale fields are
// (`packets_load`): NVIDIA's aarch64 compiler traps on a value's split
// bit fields added back together.
void packets_kquant_coefficients(uint64_t row, packets_rows16 geometry, uint p, out float scale, out float bias) {
    const uint64_t fields = row + geometry.scales + uint64_t((3u * p) >> 1);
    const uint pair = (element_u8_at(fields) | (element_u8_at(fields + 1ul) << 8)) >> ((p & 1u) * 4u);
    const vec2 factors = unpackHalf2x16(element_u32_at(row + geometry.supers + 4ul * uint64_t(p >> 3)));
    scale = factors.x * float(pair & 63u);
    bias = -(factors.y * float((pair >> 6) & 63u));
}

// Packet `p` of the row at `row`; `k` is the row length (dense packets may be
// partial; quantized kinds ignore it).
packets_packet packets_load(const int kind, uint64_t row, packets_rows16 geometry, uint p, uint k) {
    packets_packet packet;
    packet.low = uvec4(0u);
    packet.second = uvec4(0u);
    packet.high = uvec2(0u);
    packet.scale0 = 0.0;
    packet.scale1 = 0.0;
    packet.bias = 0.0;
    packet.values = 0ul;
    packet.valid = 0u;
#if PACKETS_ANY(PACKETS_KQUANT4_KINDS)
    if (packets_in(kind, PACKETS_KQUANT4_KINDS)) {
        packet.low = element_uvec4_at(row + geometry.codes + 16ul * p);
#if PACKETS_HOLDS(PACKETS_Q5K)
        if (packets_is(kind, PACKETS_Q5K))
            packet.high.x = element_u32_at(row + geometry.high + 4ul * p);
#endif
        packets_kquant_coefficients(row, geometry, p, packet.scale0, packet.bias);
        return packet;
    }
#endif
#if PACKETS_HOLDS(PACKETS_Q6K)
    if (packets_is(kind, PACKETS_Q6K)) {
        packet.low = element_uvec4_at(row + geometry.codes + 16ul * p);
        packet.high = element_uvec2_at(row + geometry.high + 8ul * p);
        const uint local = element_u16_at(row + geometry.scales + 2ul * p);
        const float d = seismic_f16_to_f32(uint16_t(element_u16_at(row + geometry.supers + 2ul * (p >> 3))));
        packet.scale0 = d * float(int(local << 24) >> 24);
        packet.scale1 = d * float(int(local << 16) >> 24);
        return packet;
    }
#endif
#if PACKETS_HOLDS(PACKETS_IQ4)
    if (packets_is(kind, PACKETS_IQ4)) {
        packet.low = element_uvec4_at(row + geometry.codes + 16ul * p);
        packet.scale0 = element_f32_at(row + geometry.supers + 4ul * p);
        return packet;
    }
#endif
#if PACKETS_HOLDS(PACKETS_IQ4S)
    if (packets_is(kind, PACKETS_IQ4S)) {
        packet.low = element_uvec4_at(row + geometry.codes + 16ul * p);
        packet.scale0 = seismic_f16_to_f32(uint16_t(element_u16_at(row + geometry.supers + 2ul * p)));
        return packet;
    }
#endif
#if PACKETS_HOLDS(PACKETS_Q4G32S)
    if (packets_is(kind, PACKETS_Q4G32S)) {
        // Codes 0..15 with the exact bias -8 d: value = d * (code - 8).
        packet.low = element_uvec4_at(row + geometry.codes + 16ul * p);
        packet.scale0 = seismic_f16_to_f32(uint16_t(element_u16_at(row + geometry.supers + 2ul * p)));
        packet.bias = -8.0 * packet.scale0;
        return packet;
    }
#endif
#if PACKETS_HOLDS(PACKETS_MXFP4)
    if (packets_is(kind, PACKETS_MXFP4)) {
        packet.low = element_uvec4_at(row + geometry.codes + 16ul * p);
        packet.scale0 = packets_e8m0_half(element_u8_at(row + geometry.supers + uint64_t(p)));
        return packet;
    }
#endif
#if PACKETS_HOLDS(PACKETS_NVFP4)
    if (packets_is(kind, PACKETS_NVFP4)) {
        packet.low = element_uvec4_at(row + geometry.codes + 16ul * p);
        const uint fields = element_u16_at(row + geometry.supers + 2ul * p);
        packet.scale0 = packets_ue4m3_half(fields & 0xffu);
        packet.scale1 = packets_ue4m3_half(fields >> 8);
        return packet;
    }
#endif
#if PACKETS_HOLDS(PACKETS_Q5G32S)
    if (packets_is(kind, PACKETS_Q5G32S)) {
        // The exact bias -16 d.
        packet.low = element_uvec4_at(row + geometry.codes + 16ul * p);
        packet.high.x = element_u32_at(row + geometry.high + 4ul * p);
        packet.scale0 = seismic_f16_to_f32(uint16_t(element_u16_at(row + geometry.supers + 2ul * p)));
        packet.bias = -16.0 * packet.scale0;
        return packet;
    }
#endif
#if PACKETS_HOLDS(PACKETS_Q5G32)
    if (packets_is(kind, PACKETS_Q5G32)) {
        // The f16 minimum m.
        packet.low = element_uvec4_at(row + geometry.codes + 16ul * p);
        packet.high.x = element_u32_at(row + geometry.high + 4ul * p);
        const uint64_t fields = row + geometry.supers + 4ul * p;
        packet.scale0 = seismic_f16_to_f32(uint16_t(element_u16_at(fields)));
        packet.bias = seismic_f16_to_f32(uint16_t(element_u16_at(fields + 2ul)));
        return packet;
    }
#endif
#if PACKETS_HOLDS(PACKETS_Q8)
    if (packets_is(kind, PACKETS_Q8)) {
        const uint64_t codes = row + geometry.codes + 32ul * p;
        packet.low = element_uvec4_at(codes);
        packet.second = element_uvec4_at(codes + 16ul);
        packet.scale0 = seismic_f16_to_f32(uint16_t(element_u16_at(row + geometry.supers + 2ul * p)));
        return packet;
    }
#endif
#if PACKETS_ANY(PACKETS_GRID_KINDS)
    if (packets_grid(kind)) {
        // The packet's four entries (8 or 12 bytes), its scale code field
        // (iq2xs: byte p, two codes; iq3xxs: nibble p of the plane; iq2xxs:
        // its entries' top bits)
        // and its group's factor (eight packets a group). The field's byte
        // is addressed by `p` itself, never as its group and offset added
        // back together: NVIDIA's aarch64 compiler (GB10, driver 580) traps
        // in pipeline creation folding such an add of a value's own split
        // bit fields (`(p & ~7) + (p & 7)`, as an address or a value) in
        // the GEMV, while `p` itself, or the fields joined with `|`,
        // compile.
        uint field = 0u;
#if PACKETS_HOLDS(PACKETS_IQ2XS)
        if (packets_is(kind, PACKETS_IQ2XS)) {
            packet.low = uvec4(element_uvec2_at(row + geometry.codes + 8ul * p), 0u, 0u);
            field = element_u8_at(row + geometry.scales + uint64_t(p));
        }
#endif
#if PACKETS_HOLDS(PACKETS_IQ2XXS)
        // iq2xxs carries its scale code in its entries: bit j of the code is
        // bit 15 of entry j, under its block's f16 d, the factor d / 8.
        if (packets_is(kind, PACKETS_IQ2XXS)) {
            const uvec2 entries = element_uvec2_at(row + geometry.codes + 8ul * p);
            packet.low = uvec4(entries, 0u, 0u);
            field = ((entries.x >> 15) & 1u) | ((entries.x >> 30) & 2u) | ((entries.y >> 13) & 4u)
                | ((entries.y >> 28) & 8u);
        }
#endif
#if PACKETS_HOLDS(PACKETS_IQ3XXS)
        if (packets_is(kind, PACKETS_IQ3XXS)) {
            const uint64_t entries = row + geometry.codes + 12ul * p;
            packet.low = uvec4(element_u32_at(entries), element_u32_at(entries + 4ul), element_u32_at(entries + 8ul), 0u);
            field = (element_u8_at(row + geometry.scales + uint64_t(p >> 1)) >> (4u * (p & 1u))) & 15u;
        }
#endif
        const float factor = packets_is(kind, PACKETS_IQ2XXS)
            ? 0.125 * seismic_f16_to_f32(uint16_t(element_u16_at(row + geometry.supers + 2ul * uint64_t(p >> 3))))
            : element_f32_at(row + geometry.supers + 4ul * uint64_t(p >> 3));
        packet.scale0 = factor * float(2u * (field & 15u) + 1u);
        packet.scale1 = factor * float(2u * (field >> 4) + 1u);
        return packet;
    }
#endif
#if PACKETS_ANY(PACKETS_DENSE_KINDS)
    if (packets_dense(kind)) {
        packet.values = row + geometry.codes + uint64_t(p) * 32ul * ELEMENT_BYTES(packets_element(kind));
        packet.valid = min(32u, k - 32u * p);
    }
#endif
    return packet;
}

// Element `i` of a dense packet (zero past the row).
float packets_dense_element(const int kind, packets_packet packet, uint i) {
    return i < packet.valid ? element_at(packets_element(kind), packet.values, i) : 0.0;
}

// The codes of sub-step `step` (elements 8 step ..) as (0,2,4,6), (1,3,5,7).
void packets_codes(const int kind, packets_packet packet, uint step, out vec4 even, out vec4 odd) {
#if PACKETS_ANY(PACKETS_TABLE_KINDS)
    if (packets_in(kind, PACKETS_TABLE_KINDS)) {
        packets_table_codes(kind, packet.low[step], even, odd);
        return;
    }
#endif
#if PACKETS_ANY(PACKETS_GRID_KINDS)
    if (packets_grid(kind)) {
        uvec2 point;
        uint signs;
        packets_grid_entry(kind, packet.low, step, point, signs);
        packets_grid_codes(point, signs, even, odd);
        return;
    }
#endif
#if PACKETS_ANY(PACKETS_NIBBLE_KINDS)
    if (packets_in(kind, PACKETS_NIBBLE_KINDS)) {
        const uint word = packet.low[step];
        even = packets_unsigned_bytes(word & 0x0f0f0f0fu);
        odd = packets_unsigned_bytes((word >> 4) & 0x0f0f0f0fu);
        return;
    }
#endif
#if PACKETS_ANY(PACKETS_HIGH1_KINDS)
    if (packets_high1(kind)) {
        const uint word = packet.low[step];
        uint he, ho;
        packets_split_high1((packet.high.x >> (8u * step)) & 0xffu, he, ho);
        even = packets_unsigned_bytes((word & 0x0f0f0f0fu) | he);
        odd = packets_unsigned_bytes(((word >> 4) & 0x0f0f0f0fu) | ho);
        return;
    }
#endif
#if PACKETS_HOLDS(PACKETS_Q6K)
    if (packets_is(kind, PACKETS_Q6K)) {
        const uint word = packet.low[step];
        uint he, ho;
        packets_split_high2((packet.high[step >> 1] >> (16u * (step & 1u))) & 0xffffu, he, ho);
        even = packets_unsigned_bytes((word & 0x0f0f0f0fu) | he);
        odd = packets_unsigned_bytes(((word >> 4) & 0x0f0f0f0fu) | ho);
        return;
    }
#endif
#if PACKETS_HOLDS(PACKETS_Q8)
    if (packets_is(kind, PACKETS_Q8)) {
        const uvec4 words = step < 2u ? packet.low : packet.second;
        const uint base = (step & 1u) * 2u;
        const vec4 a = packets_signed_bytes(words[base]);
        const vec4 b = packets_signed_bytes(words[base + 1u]);
        even = vec4(a.x, a.z, b.x, b.z);
        odd = vec4(a.y, a.w, b.y, b.w);
        return;
    }
#endif
#if PACKETS_ANY(PACKETS_DENSE_KINDS)
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
    }
#endif
}

float packets_scale(const int kind, packets_packet packet, uint step) {
    if (packets_groups(kind) == 2u)
        return step < 2u ? packet.scale0 : packet.scale1;
    return packets_dense(kind) ? 1.0 : packet.scale0;
}

// The bias of coefficient group `group` (q6k: -32 * scale of the group).
float packets_bias(const int kind, packets_packet packet, uint group) {
    if (packets_is(kind, PACKETS_Q6K))
        return -32.0 * (group == 0u ? packet.scale0 : packet.scale1);
    if (packets_biased(kind))
        return packet.bias;
    return 0.0;
}

// The logical value of a decoded code of sub-step `step`.
float packets_value(const int kind, packets_packet packet, uint step, float code) {
    if (packets_is(kind, PACKETS_Q6K))
        return packets_scale(kind, packet, step) * (code - 32.0);
    if (packets_biased(kind))
        return seismic_fma_rn(packet.scale0, code, packet.bias);
    if (packets_dense(kind))
        return code;
    return packets_scale(kind, packet, step) * code;
}

// One decoded logical weight value (used by gathers such as the embedding).
float packets_value_at(const int kind, packets_packet packet, uint i) {
    vec4 even, odd;
    packets_codes(kind, packet, i >> 3, even, odd);
    const uint lane = (i & 7u) >> 1;
    const float code = (i & 1u) != 0u ? odd[lane] : even[lane];
    return packets_value(kind, packet, i >> 3, code);
}
