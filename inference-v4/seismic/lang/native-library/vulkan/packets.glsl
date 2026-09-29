// Seismic native library (Vulkan): packet decoders for weights stored in the
// `rows16` layout (the Vulkan resident layout). `#include <seismic/packets.glsl>`.
// The counterpart of <seismic/packets.h> (Metal).
//
// A packet is 32 consecutive logical elements of one weight row. In `rows16`
// a q4k/q5k/q6k packet is one 16-byte load of its low-nibble code plane
// (codes 2i/2i+1 in the low/high nibble of byte i), plus its slice of the
// high-bit plane (q5k: one u32, q6k: one uvec2), its local coefficients and
// its super factors; a q8 packet is two 16-byte loads of int8 codes. A packet
// is consumed in four 8-element sub-steps. Each sub-step decodes its codes as
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
// Kinds are compile-time constants at every call, so the driver folds the
// kind switches and drops the unused fields. An entry maps each weight
// tensor's representation to a kind and a `packets_rows16` with a
// `defined(...)` ladder over its ABI prefix names.
//
// This file is independent of any entry ABI.
#include <seismic/element.glsl>

#define PACKETS_Q4K 0
#define PACKETS_Q5K 1
#define PACKETS_Q6K 2
#define PACKETS_Q8 3
#define PACKETS_BF16 4
#define PACKETS_F16 5
#define PACKETS_F32 6

// Coefficient groups per packet (q6k: one per 16 elements).
uint packets_groups(const int kind) { return kind == PACKETS_Q6K ? 2u : 1u; }

// The kind has a per-group bias term (the factored `bias * sum(x)`).
bool packets_biased(const int kind) { return kind <= PACKETS_Q6K; }

// Dense weights: the element kind of their storage.
bool packets_dense(const int kind) { return kind >= PACKETS_BF16; }

int packets_element(const int kind) {
    return kind == PACKETS_BF16 ? ELEMENT_BF16 : kind == PACKETS_F16 ? ELEMENT_F16 : ELEMENT_F32;
}

// ---------------------------------------------------------------------------
// Weight row geometry. `stride` is the byte distance between rows; plane
// offsets are within a row. Planes a representation lacks are unused.

struct packets_rows16 {
    uint64_t stride;
    uint64_t codes;   // codes_lo (k-quants), codes (q8) or the values (dense)
    uint64_t high;    // codes_hi (q5k, q6k)
    uint64_t scales;  // packed local coefficients (k-quants)
    uint64_t supers;  // super factors (k-quants) or group scales (q8)
};

// The row geometry of a weight tensor from its ABI prefix name T (for
// example `SEISMIC_GATE_WEIGHT`), per kind family; dense weights take their
// element size.
#define PACKETS_ROWS16_Q4K(T) packets_rows16(T##_ROW_STRIDE_BYTES, T##_PLANE_CODES_LO_ROW_OFFSET, 0ul, T##_PLANE_SCALES_ROW_OFFSET, T##_PLANE_SUPERS_ROW_OFFSET)
#define PACKETS_ROWS16_HIGH(T) packets_rows16(T##_ROW_STRIDE_BYTES, T##_PLANE_CODES_LO_ROW_OFFSET, T##_PLANE_CODES_HI_ROW_OFFSET, T##_PLANE_SCALES_ROW_OFFSET, T##_PLANE_SUPERS_ROW_OFFSET)
#define PACKETS_ROWS16_Q8(T) packets_rows16(T##_ROW_STRIDE_BYTES, T##_PLANE_CODES_ROW_OFFSET, 0ul, 0ul, T##_PLANE_SUPERS_ROW_OFFSET)
#define PACKETS_ROWS16_DENSE(T, BYTES) packets_rows16(T##_STRIDE_0 * uint64_t(BYTES), 0ul, 0ul, 0ul, 0ul)

// One loaded packet of any kind.
struct packets_packet {
    uvec4 low;       // codes_lo (k-quants) or the first 16 codes (q8)
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

// The (scale6, min6) pair of k-quant packet `p` and its super factors.
void packets_kquant_coefficients(uint64_t row, packets_rows16 geometry, uint p, out float scale, out float bias) {
    const uint block = p >> 3, local = p & 7u;
    const uint64_t fields = row + geometry.scales + 12ul * block + ((3u * local) >> 1);
    const uint pair = (element_u8_at(fields) | (element_u8_at(fields + 1ul) << 8)) >> ((local & 1u) * 4u);
    const vec2 factors = unpackHalf2x16(element_u32_at(row + geometry.supers + 4ul * block));
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
    if (kind == PACKETS_Q4K || kind == PACKETS_Q5K) {
        packet.low = element_uvec4_at(row + geometry.codes + 16ul * p);
        if (kind == PACKETS_Q5K)
            packet.high.x = element_u32_at(row + geometry.high + 4ul * p);
        packets_kquant_coefficients(row, geometry, p, packet.scale0, packet.bias);
    } else if (kind == PACKETS_Q6K) {
        packet.low = element_uvec4_at(row + geometry.codes + 16ul * p);
        packet.high = element_uvec2_at(row + geometry.high + 8ul * p);
        const uint local = element_u16_at(row + geometry.scales + 2ul * p);
        const float d = seismic_f16_to_f32(uint16_t(element_u16_at(row + geometry.supers + 2ul * (p >> 3))));
        packet.scale0 = d * float(int(local << 24) >> 24);
        packet.scale1 = d * float(int(local << 16) >> 24);
    } else if (kind == PACKETS_Q8) {
        const uint64_t codes = row + geometry.codes + 32ul * p;
        packet.low = element_uvec4_at(codes);
        packet.second = element_uvec4_at(codes + 16ul);
        packet.scale0 = seismic_f16_to_f32(uint16_t(element_u16_at(row + geometry.supers + 2ul * p)));
    } else {
        packet.values = row + geometry.codes + uint64_t(p) * 32ul * ELEMENT_BYTES(packets_element(kind));
        packet.valid = min(32u, k - 32u * p);
    }
    return packet;
}

// Element `i` of a dense packet (zero past the row).
float packets_dense_element(const int kind, packets_packet packet, uint i) {
    return i < packet.valid ? element_at(packets_element(kind), packet.values, i) : 0.0;
}

// The codes of sub-step `step` (elements 8 step ..) as (0,2,4,6), (1,3,5,7).
void packets_codes(const int kind, packets_packet packet, uint step, out vec4 even, out vec4 odd) {
    if (kind == PACKETS_Q4K) {
        const uint word = packet.low[step];
        even = packets_unsigned_bytes(word & 0x0f0f0f0fu);
        odd = packets_unsigned_bytes((word >> 4) & 0x0f0f0f0fu);
    } else if (kind == PACKETS_Q5K) {
        const uint word = packet.low[step];
        uint he, ho;
        packets_split_high1((packet.high.x >> (8u * step)) & 0xffu, he, ho);
        even = packets_unsigned_bytes((word & 0x0f0f0f0fu) | he);
        odd = packets_unsigned_bytes(((word >> 4) & 0x0f0f0f0fu) | ho);
    } else if (kind == PACKETS_Q6K) {
        const uint word = packet.low[step];
        uint he, ho;
        packets_split_high2((packet.high[step >> 1] >> (16u * (step & 1u))) & 0xffffu, he, ho);
        even = packets_unsigned_bytes((word & 0x0f0f0f0fu) | he);
        odd = packets_unsigned_bytes(((word >> 4) & 0x0f0f0f0fu) | ho);
    } else if (kind == PACKETS_Q8) {
        const uvec4 words = step < 2u ? packet.low : packet.second;
        const uint base = (step & 1u) * 2u;
        const vec4 a = packets_signed_bytes(words[base]);
        const vec4 b = packets_signed_bytes(words[base + 1u]);
        even = vec4(a.x, a.z, b.x, b.z);
        odd = vec4(a.y, a.w, b.y, b.w);
    } else {
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
}

float packets_scale(const int kind, packets_packet packet, uint step) {
    if (kind == PACKETS_Q6K)
        return step < 2u ? packet.scale0 : packet.scale1;
    return packets_dense(kind) ? 1.0 : packet.scale0;
}

// The bias of coefficient group `group` (q6k: -32 * scale of the group).
float packets_bias(const int kind, packets_packet packet, uint group) {
    if (kind == PACKETS_Q4K || kind == PACKETS_Q5K)
        return packet.bias;
    if (kind == PACKETS_Q6K)
        return -32.0 * (group == 0u ? packet.scale0 : packet.scale1);
    return 0.0;
}

// The logical value of a decoded code of sub-step `step`.
float packets_value(const int kind, packets_packet packet, uint step, float code) {
    if (kind == PACKETS_Q4K || kind == PACKETS_Q5K)
        return seismic_fma_rn(packet.scale0, code, packet.bias);
    if (kind == PACKETS_Q6K)
        return packets_scale(kind, packet, step) * (code - 32.0);
    if (kind == PACKETS_Q8)
        return packet.scale0 * code;
    return code;
}

// One decoded logical weight value (used by gathers such as the embedding).
float packets_value_at(const int kind, packets_packet packet, uint i) {
    vec4 even, odd;
    packets_codes(kind, packet, i >> 3, even, odd);
    const uint lane = (i & 7u) >> 1;
    const float code = (i & 1u) != 0u ? odd[lane] : even[lane];
    return packets_value(kind, packet, i >> 3, code);
}
