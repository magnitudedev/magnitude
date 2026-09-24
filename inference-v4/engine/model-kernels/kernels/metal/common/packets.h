// Packet decoders for weights stored in the `rows16` layout, and the weight
// slot bindings every projection entry uses. Dense element types (activations,
// norm vectors, dense weights) are `element` (element.h).
//
// A packet is 32 consecutive logical elements of one weight row. In `rows16`
// a q4k/q5k/q6k packet is one 16-byte load of its low-nibble code plane
// (codes 2i/2i+1 in the low/high nibble of byte i), plus its slice of the
// high-bit plane (q5k: one u32, q6k: one u2), its local coefficients and its
// super factors; a q8 packet is two 16-byte loads of int8 codes. A packet is
// consumed in four 8-element sub-steps. Each sub-step decodes its codes as
// exact small integers in two float4s, `even` (elements 0,2,4,6) and `odd`
// (1,3,5,7), by splitting code bytes on nibble boundaries: no per-element
// shifts, no activation prescale, and every decoded value is an exact
// integer, so no rounding or subnormal case exists in the decode. Code bytes
// become floats without integer conversions (`byte_floats`): the M = 1
// projections are bound by the decode's instruction issue, and an integer
// conversion costs several times a half subtraction.
//
// A packet's logical value is `scale(g) * code + bias(g)` for its coefficient
// group g, dequantized with one F32 rounding (`value`, or the fused
// `fma(scale, code, bias)` of whole steps).
//
// Weight tensors are bound to slots before this file is included:
//     #define KERNEL_W0 SEISMIC_GATE_WEIGHT
// which defines `packets::W0` (the decoder of the bound representation) and
// `KERNEL_W0_LAYOUT(k)` (its `Rows16` geometry for rows of `k` logical
// values). Slots W0..W3 exist. A bound weight is dense (f32, bf16, f16) or
// packed in the rows16 layout.

#include "common/element.h"

namespace packets {

// ---------------------------------------------------------------------------
// Weight row geometry. `stride` is the byte distance between rows; plane
// offsets are within a row. Planes a representation lacks are unused.

struct Rows16 {
    ulong stride;
    ulong codes;    // codes_lo (k-quants), codes (q8) or the values (dense)
    ulong high;     // codes_hi (q5k, q6k)
    ulong scales;   // packed local coefficients (k-quants)
    ulong supers;   // super factors (k-quants) or group scales (q8)
};

// Bytes 0,2,4,6 / 1,3,5,7 of a 16-bit field of 2-bit high codes, moved to
// bits 4..5 of four bytes.
inline void split_high2(uint field, thread uint &even, thread uint &odd) {
    uint t = (field | (field << 8)) & 0x00ff00ffu;
    t = (t | (t << 4)) & 0x0f0f0f0fu;
    even = (t & 0x03030303u) << 4;
    odd = (t & 0x0c0c0c0cu) << 2;
}

// The same for an 8-bit field of 1-bit high codes, moved to bit 4.
inline void split_high1(uint field, thread uint &even, thread uint &odd) {
    uint t = (field | (field << 12)) & 0x000f000fu;
    t = (t | (t << 6)) & 0x03030303u;
    even = (t & 0x01010101u) << 4;
    odd = (t & 0x02020202u) << 3;
}

inline void split_nibbles(uint word, thread uint &even, thread uint &odd) {
    even = word & 0x0f0f0f0fu;
    odd = (word >> 4) & 0x0f0f0f0fu;
}

// The four bytes of `word` as exact floats, less `offset - 1024`: alternate
// bytes become the low mantissa bits of a half2 whose exponent makes each
// lane 1024 + byte, and one half subtraction removes `offset`.
inline float4 byte_floats(uint word, half offset) {
    half2 even = as_type<half2>((word & 0x00ff00ffu) | 0x64006400u) - half2(offset);
    half2 odd = as_type<half2>(((word >> 8) & 0x00ff00ffu) | 0x64006400u) - half2(offset);
    return float4(even.x, odd.x, even.y, odd.y);
}
inline float4 unsigned_bytes(uint word) { return byte_floats(word, 1024.0h); }
// Signed bytes: flipping the sign bits maps -128..127 onto 0..255.
inline float4 signed_bytes(uint word) { return byte_floats(word ^ 0x80808080u, 1152.0h); }

// Two codes (each below 1024; the second in bits 16..) as exact floats, the
// same way.
inline float2 code_pair(uint two, half offset) {
    return float2(as_type<half2>(two | 0x64006400u) - half2(offset));
}

// ---------------------------------------------------------------------------
// q4k: 4-bit unsigned codes, groups of 32 with (scale6, min6) and per-256
// (d, dmin): value = d*scale6*code - dmin*min6.

struct Q4K {
    static constant constexpr uint groups = 1;       // coefficient groups per packet
    struct packet {
        uint4 low;
        float scale;
        float bias;
    };
    static packet load(device const uchar *row, Rows16 layout, uint p) {
        packet k;
        k.low = *reinterpret_cast<device const uint4 *>(row + layout.codes + 16ul * p);
        uint block = p >> 3, local = p & 7u;
        device const uchar *fields = row + layout.scales + 12ul * block + ((3u * local) >> 1);
        uint pair = (uint(fields[0]) | (uint(fields[1]) << 8)) >> ((local & 1u) * 4u);
        half2 factors = *reinterpret_cast<device const half2 *>(row + layout.supers + 4ul * block);
        k.scale = float(factors.x) * float(pair & 63u);
        k.bias = -(float(factors.y) * float((pair >> 6) & 63u));
        return k;
    }
    static void codes(thread const packet &k, uint step, thread float4 &even, thread float4 &odd) {
        uint e, o;
        split_nibbles(k.low[step], e, o);
        even = unsigned_bytes(e);
        odd = unsigned_bytes(o);
    }
    // The codes of columns 8 * step + 2 * j and + 1 (j < 4).
    static float2 pair(thread const packet &k, uint step, uint j) {
        uint b = (k.low[step] >> (8u * j)) & 0xffu;
        return code_pair((b & 15u) | ((b >> 4) << 16), 1024.0h);
    }
    static float scale(thread const packet &k, uint) { return k.scale; }
    static float bias(thread const packet &k, uint) { return k.bias; }
    static float value(thread const packet &k, uint, float code) {
        return metal::fma(k.scale, code, k.bias);
    }
};

// q5k: q4k plus one high bit per code (code = low + 16*high).
struct Q5K {
    static constant constexpr uint groups = 1;
    struct packet {
        uint4 low;
        uint high;
        float scale;
        float bias;
    };
    static packet load(device const uchar *row, Rows16 layout, uint p) {
        packet k;
        k.low = *reinterpret_cast<device const uint4 *>(row + layout.codes + 16ul * p);
        k.high = *reinterpret_cast<device const uint *>(row + layout.high + 4ul * p);
        uint block = p >> 3, local = p & 7u;
        device const uchar *fields = row + layout.scales + 12ul * block + ((3u * local) >> 1);
        uint pair = (uint(fields[0]) | (uint(fields[1]) << 8)) >> ((local & 1u) * 4u);
        half2 factors = *reinterpret_cast<device const half2 *>(row + layout.supers + 4ul * block);
        k.scale = float(factors.x) * float(pair & 63u);
        k.bias = -(float(factors.y) * float((pair >> 6) & 63u));
        return k;
    }
    static void codes(thread const packet &k, uint step, thread float4 &even, thread float4 &odd) {
        uint e, o, he, ho;
        split_nibbles(k.low[step], e, o);
        split_high1((k.high >> (8u * step)) & 0xffu, he, ho);
        even = unsigned_bytes(e | he);
        odd = unsigned_bytes(o | ho);
    }
    static float2 pair(thread const packet &k, uint step, uint j) {
        uint b = (k.low[step] >> (8u * j)) & 0xffu;
        uint h = (k.high >> (8u * step + 2u * j)) & 3u;
        return code_pair((b & 15u) | ((h & 1u) << 4) | (((b >> 4) | ((h >> 1) << 4)) << 16), 1024.0h);
    }
    static float scale(thread const packet &k, uint) { return k.scale; }
    static float bias(thread const packet &k, uint) { return k.bias; }
    static float value(thread const packet &k, uint, float code) {
        return metal::fma(k.scale, code, k.bias);
    }
};

// q6k: 6-bit codes (low nibble + two high bits) offset by 32, int8 scales per
// 16 and per-256 d: value = d*scale8*(code - 32).
struct Q6K {
    static constant constexpr uint groups = 2;
    struct packet {
        uint4 low;
        uint2 high;
        float scale0;
        float scale1;
    };
    static packet load(device const uchar *row, Rows16 layout, uint p) {
        packet k;
        k.low = *reinterpret_cast<device const uint4 *>(row + layout.codes + 16ul * p);
        k.high = *reinterpret_cast<device const uint2 *>(row + layout.high + 8ul * p);
        char2 local = *reinterpret_cast<device const char2 *>(row + layout.scales + 2ul * p);
        float d = float(*reinterpret_cast<device const half *>(row + layout.supers + 2ul * (p >> 3)));
        k.scale0 = d * float(local.x);
        k.scale1 = d * float(local.y);
        return k;
    }
    static void codes(thread const packet &k, uint step, thread float4 &even, thread float4 &odd) {
        uint e, o, he, ho;
        split_nibbles(k.low[step], e, o);
        split_high2((k.high[step >> 1] >> (16u * (step & 1u))) & 0xffffu, he, ho);
        even = unsigned_bytes(e | he);
        odd = unsigned_bytes(o | ho);
    }
    static float2 pair(thread const packet &k, uint step, uint j) {
        uint b = (k.low[step] >> (8u * j)) & 0xffu;
        uint h = (k.high[step >> 1] >> (16u * (step & 1u) + 4u * j)) & 0xfu;
        return code_pair((b & 15u) | ((h & 3u) << 4) | (((b >> 4) | ((h >> 2) << 4)) << 16), 1024.0h);
    }
    static float scale(thread const packet &k, uint step) { return step < 2 ? k.scale0 : k.scale1; }
    static float bias(thread const packet &k, uint group) {
        return -32.0f * (group == 0 ? k.scale0 : k.scale1);
    }
    static float value(thread const packet &k, uint step, float code) {
        return scale(k, step) * (code - 32.0f);
    }
};

// q8 (q8g32s): int8 codes, one f16 scale per 32.
struct Q8 {
    static constant constexpr uint groups = 1;
    struct packet {
        uint4 first;
        uint4 second;
        float scale;
    };
    static packet load(device const uchar *row, Rows16 layout, uint p) {
        packet k;
        device const uint4 *codes = reinterpret_cast<device const uint4 *>(row + layout.codes + 32ul * p);
        k.first = codes[0];
        k.second = codes[1];
        k.scale = float(*reinterpret_cast<device const half *>(row + layout.supers + 2ul * p));
        return k;
    }
    static void codes(thread const packet &k, uint step, thread float4 &even, thread float4 &odd) {
        uint4 words = step < 2 ? k.first : k.second;
        uint base = (step & 1u) * 2u;
        float4 a = signed_bytes(words[base]);
        float4 b = signed_bytes(words[base + 1]);
        even = float4(a.x, a.z, b.x, b.z);
        odd = float4(a.y, a.w, b.y, b.w);
    }
    static float2 pair(thread const packet &k, uint step, uint j) {
        uint4 words = step < 2 ? k.first : k.second;
        uint word = words[(step & 1u) * 2u + (j >> 1)];
        uint flipped = (word ^ 0x80808080u) >> ((j & 1u) * 16u);
        return code_pair((flipped & 0xffu) | ((flipped & 0xff00u) << 8), 1152.0h);
    }
    static float scale(thread const packet &k, uint) { return k.scale; }
    static float bias(thread const packet &, uint) { return 0.0f; }
    static float value(thread const packet &k, uint, float code) { return k.scale * code; }
};

// Dense weights of element type E (element::Bf16, F16 or F32). The row's
// final packet may be partial when K is not a multiple of 32; its missing
// elements decode as zero.
template <typename E>
struct Dense {
    static constant constexpr uint groups = 1;
    struct packet {
        device const uchar *values;
        uint valid;   // elements of this packet inside the row
    };
    static packet load(device const uchar *row, Rows16 layout, uint p, uint k) {
        packet out;
        out.values = row + layout.codes + ulong(p) * 32ul * E::bytes;
        out.valid = min(32u, k - 32u * p);
        return out;
    }
    static float element(thread const packet &k, uint i) {
        return i < k.valid
            ? E::load(reinterpret_cast<device const typename E::storage *>(k.values)[i])
            : 0.0f;
    }
    static void codes(thread const packet &k, uint step, thread float4 &even, thread float4 &odd) {
        uint first = 8u * step;
        if (first + 8u <= k.valid) {
            device const typename E::storage *v =
                reinterpret_cast<device const typename E::storage *>(k.values) + first;
            even = float4(E::load(v[0]), E::load(v[2]), E::load(v[4]), E::load(v[6]));
            odd = float4(E::load(v[1]), E::load(v[3]), E::load(v[5]), E::load(v[7]));
        } else {
            even = float4(element(k, first), element(k, first + 2), element(k, first + 4),
                element(k, first + 6));
            odd = float4(element(k, first + 1), element(k, first + 3), element(k, first + 5),
                element(k, first + 7));
        }
    }
    static float2 pair(thread const packet &k, uint step, uint j) {
        return float2(element(k, 8u * step + 2u * j), element(k, 8u * step + 2u * j + 1u));
    }
    static float scale(thread const packet &, uint) { return 1.0f; }
    static float bias(thread const packet &, uint) { return 0.0f; }
    static float value(thread const packet &, uint, float code) { return code; }
};

// Uniform packet loading: quantized packets ignore the row length.
template <typename W>
struct Loader {
    static typename W::packet load(device const uchar *row, Rows16 layout, uint p, uint) {
        return W::load(row, layout, p);
    }
};
template <typename E>
struct Loader<Dense<E>> {
    static typename Dense<E>::packet load(device const uchar *row, Rows16 layout, uint p, uint k) {
        return Dense<E>::load(row, layout, p, k);
    }
};

// One decoded logical weight value (used by gathers such as the embedding).
template <typename W>
inline float value_at(thread const typename W::packet &k, uint i) {
    float4 even, odd;
    W::codes(k, i >> 3, even, odd);
    uint lane = (i & 7u) >> 1;
    float code = (i & 1u) ? odd[lane] : even[lane];
    return W::value(k, i >> 3, code);
}

} // namespace packets

// ---------------------------------------------------------------------------
// Weight slots. Layout names are formed by token pasting from the bound
// prefix; a dense weight row is `k` contiguous elements.

#define PACKETS_ROWS16_Q4K(P)                                                                        \
    packets::Rows16 { ELEMENT_CAT(P, _ROW_STRIDE_BYTES), ELEMENT_CAT(P, _PLANE_CODES_LO_ROW_OFFSET), 0, \
        ELEMENT_CAT(P, _PLANE_SCALES_ROW_OFFSET), ELEMENT_CAT(P, _PLANE_SUPERS_ROW_OFFSET) }
#define PACKETS_ROWS16_HIGH(P)                                                                       \
    packets::Rows16 { ELEMENT_CAT(P, _ROW_STRIDE_BYTES), ELEMENT_CAT(P, _PLANE_CODES_LO_ROW_OFFSET),   \
        ELEMENT_CAT(P, _PLANE_CODES_HI_ROW_OFFSET), ELEMENT_CAT(P, _PLANE_SCALES_ROW_OFFSET),           \
        ELEMENT_CAT(P, _PLANE_SUPERS_ROW_OFFSET) }
#define PACKETS_ROWS16_Q8(P)                                                                         \
    packets::Rows16 { ELEMENT_CAT(P, _ROW_STRIDE_BYTES), ELEMENT_CAT(P, _PLANE_CODES_ROW_OFFSET), 0, 0, \
        ELEMENT_CAT(P, _PLANE_SUPERS_ROW_OFFSET) }
#define PACKETS_ROWS16_DENSE(E, k) packets::Rows16 { ulong(k) * E::bytes, 0, 0, 0, 0 }

// Binds slot `SLOT` to the decoder type `TYPE`.
#define PACKETS_BIND(SLOT, TYPE) namespace packets { typedef TYPE SLOT; }

#if defined(KERNEL_W0)
#if ELEMENT_HAS(KERNEL_W0, _REPRESENTATION_F32)
PACKETS_BIND(W0, packets::Dense<element::F32>)
#define KERNEL_W0_LAYOUT(k) PACKETS_ROWS16_DENSE(element::F32, k)
#elif ELEMENT_HAS(KERNEL_W0, _REPRESENTATION_BF16)
PACKETS_BIND(W0, packets::Dense<element::Bf16>)
#define KERNEL_W0_LAYOUT(k) PACKETS_ROWS16_DENSE(element::Bf16, k)
#elif ELEMENT_HAS(KERNEL_W0, _REPRESENTATION_F16)
PACKETS_BIND(W0, packets::Dense<element::F16>)
#define KERNEL_W0_LAYOUT(k) PACKETS_ROWS16_DENSE(element::F16, k)
#elif !ELEMENT_HAS(KERNEL_W0, _LAYOUT_ROWS16)
#error "KERNEL_W0 must be dense or in the rows16 layout"
#elif ELEMENT_HAS(KERNEL_W0, _REPRESENTATION_Q4K)
PACKETS_BIND(W0, packets::Q4K)
#define KERNEL_W0_LAYOUT(k) PACKETS_ROWS16_Q4K(KERNEL_W0)
#elif ELEMENT_HAS(KERNEL_W0, _REPRESENTATION_Q5K)
PACKETS_BIND(W0, packets::Q5K)
#define KERNEL_W0_LAYOUT(k) PACKETS_ROWS16_HIGH(KERNEL_W0)
#elif ELEMENT_HAS(KERNEL_W0, _REPRESENTATION_Q6K)
PACKETS_BIND(W0, packets::Q6K)
#define KERNEL_W0_LAYOUT(k) PACKETS_ROWS16_HIGH(KERNEL_W0)
#elif ELEMENT_HAS(KERNEL_W0, _REPRESENTATION_Q8G32S)
PACKETS_BIND(W0, packets::Q8)
#define KERNEL_W0_LAYOUT(k) PACKETS_ROWS16_Q8(KERNEL_W0)
#else
#error "KERNEL_W0: unsupported weight representation"
#endif
#endif

#if defined(KERNEL_W1)
#if ELEMENT_HAS(KERNEL_W1, _REPRESENTATION_F32)
PACKETS_BIND(W1, packets::Dense<element::F32>)
#define KERNEL_W1_LAYOUT(k) PACKETS_ROWS16_DENSE(element::F32, k)
#elif ELEMENT_HAS(KERNEL_W1, _REPRESENTATION_BF16)
PACKETS_BIND(W1, packets::Dense<element::Bf16>)
#define KERNEL_W1_LAYOUT(k) PACKETS_ROWS16_DENSE(element::Bf16, k)
#elif ELEMENT_HAS(KERNEL_W1, _REPRESENTATION_F16)
PACKETS_BIND(W1, packets::Dense<element::F16>)
#define KERNEL_W1_LAYOUT(k) PACKETS_ROWS16_DENSE(element::F16, k)
#elif !ELEMENT_HAS(KERNEL_W1, _LAYOUT_ROWS16)
#error "KERNEL_W1 must be dense or in the rows16 layout"
#elif ELEMENT_HAS(KERNEL_W1, _REPRESENTATION_Q4K)
PACKETS_BIND(W1, packets::Q4K)
#define KERNEL_W1_LAYOUT(k) PACKETS_ROWS16_Q4K(KERNEL_W1)
#elif ELEMENT_HAS(KERNEL_W1, _REPRESENTATION_Q5K)
PACKETS_BIND(W1, packets::Q5K)
#define KERNEL_W1_LAYOUT(k) PACKETS_ROWS16_HIGH(KERNEL_W1)
#elif ELEMENT_HAS(KERNEL_W1, _REPRESENTATION_Q6K)
PACKETS_BIND(W1, packets::Q6K)
#define KERNEL_W1_LAYOUT(k) PACKETS_ROWS16_HIGH(KERNEL_W1)
#elif ELEMENT_HAS(KERNEL_W1, _REPRESENTATION_Q8G32S)
PACKETS_BIND(W1, packets::Q8)
#define KERNEL_W1_LAYOUT(k) PACKETS_ROWS16_Q8(KERNEL_W1)
#else
#error "KERNEL_W1: unsupported weight representation"
#endif
#endif

#if defined(KERNEL_W2)
#if ELEMENT_HAS(KERNEL_W2, _REPRESENTATION_F32)
PACKETS_BIND(W2, packets::Dense<element::F32>)
#define KERNEL_W2_LAYOUT(k) PACKETS_ROWS16_DENSE(element::F32, k)
#elif ELEMENT_HAS(KERNEL_W2, _REPRESENTATION_BF16)
PACKETS_BIND(W2, packets::Dense<element::Bf16>)
#define KERNEL_W2_LAYOUT(k) PACKETS_ROWS16_DENSE(element::Bf16, k)
#elif ELEMENT_HAS(KERNEL_W2, _REPRESENTATION_F16)
PACKETS_BIND(W2, packets::Dense<element::F16>)
#define KERNEL_W2_LAYOUT(k) PACKETS_ROWS16_DENSE(element::F16, k)
#elif !ELEMENT_HAS(KERNEL_W2, _LAYOUT_ROWS16)
#error "KERNEL_W2 must be dense or in the rows16 layout"
#elif ELEMENT_HAS(KERNEL_W2, _REPRESENTATION_Q4K)
PACKETS_BIND(W2, packets::Q4K)
#define KERNEL_W2_LAYOUT(k) PACKETS_ROWS16_Q4K(KERNEL_W2)
#elif ELEMENT_HAS(KERNEL_W2, _REPRESENTATION_Q5K)
PACKETS_BIND(W2, packets::Q5K)
#define KERNEL_W2_LAYOUT(k) PACKETS_ROWS16_HIGH(KERNEL_W2)
#elif ELEMENT_HAS(KERNEL_W2, _REPRESENTATION_Q6K)
PACKETS_BIND(W2, packets::Q6K)
#define KERNEL_W2_LAYOUT(k) PACKETS_ROWS16_HIGH(KERNEL_W2)
#elif ELEMENT_HAS(KERNEL_W2, _REPRESENTATION_Q8G32S)
PACKETS_BIND(W2, packets::Q8)
#define KERNEL_W2_LAYOUT(k) PACKETS_ROWS16_Q8(KERNEL_W2)
#else
#error "KERNEL_W2: unsupported weight representation"
#endif
#endif

#if defined(KERNEL_W3)
#if ELEMENT_HAS(KERNEL_W3, _REPRESENTATION_F32)
PACKETS_BIND(W3, packets::Dense<element::F32>)
#define KERNEL_W3_LAYOUT(k) PACKETS_ROWS16_DENSE(element::F32, k)
#elif ELEMENT_HAS(KERNEL_W3, _REPRESENTATION_BF16)
PACKETS_BIND(W3, packets::Dense<element::Bf16>)
#define KERNEL_W3_LAYOUT(k) PACKETS_ROWS16_DENSE(element::Bf16, k)
#elif ELEMENT_HAS(KERNEL_W3, _REPRESENTATION_F16)
PACKETS_BIND(W3, packets::Dense<element::F16>)
#define KERNEL_W3_LAYOUT(k) PACKETS_ROWS16_DENSE(element::F16, k)
#elif !ELEMENT_HAS(KERNEL_W3, _LAYOUT_ROWS16)
#error "KERNEL_W3 must be dense or in the rows16 layout"
#elif ELEMENT_HAS(KERNEL_W3, _REPRESENTATION_Q4K)
PACKETS_BIND(W3, packets::Q4K)
#define KERNEL_W3_LAYOUT(k) PACKETS_ROWS16_Q4K(KERNEL_W3)
#elif ELEMENT_HAS(KERNEL_W3, _REPRESENTATION_Q5K)
PACKETS_BIND(W3, packets::Q5K)
#define KERNEL_W3_LAYOUT(k) PACKETS_ROWS16_HIGH(KERNEL_W3)
#elif ELEMENT_HAS(KERNEL_W3, _REPRESENTATION_Q6K)
PACKETS_BIND(W3, packets::Q6K)
#define KERNEL_W3_LAYOUT(k) PACKETS_ROWS16_HIGH(KERNEL_W3)
#elif ELEMENT_HAS(KERNEL_W3, _REPRESENTATION_Q8G32S)
PACKETS_BIND(W3, packets::Q8)
#define KERNEL_W3_LAYOUT(k) PACKETS_ROWS16_Q8(KERNEL_W3)
#else
#error "KERNEL_W3: unsupported weight representation"
#endif
#endif
