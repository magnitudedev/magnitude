// Packet decoders for weights stored in the `rows16` layout, and the dense
// activation/vector element types every projection shares.
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
// integer, so no rounding or subnormal case exists in the decode.
//
// A packet's logical value is `scale(g) * code + bias(g)` for its coefficient
// group g. Dot products are factored: `scale * dot(code, x) + bias * sum(x)`,
// where the activation sums per 16 elements are computed once per staged
// activation and reused by every weight row.
//
// This file is independent of any entry ABI; entries bind their layout
// constants into `rows16` and choose the packet type.

namespace packets {

// ---------------------------------------------------------------------------
// Dense element types (activations, norm vectors, dense weights).

struct bf16 {
    typedef ushort storage;
    typedef bfloat native;   // the MSL scalar of the storage (matrix operands)
    static constant constexpr uint bytes = 2;
    static float load(storage value) { return as_type<float>(uint(value) << 16); }
    static storage store(float value) {
        uint bits = as_type<uint>(value);
        bits += 0x7fffu + ((bits >> 16) & 1u);
        return ushort(bits >> 16);
    }
    static float round(float value) { return load(store(value)); }
    // Eight consecutive elements packed in a uint4: element 2i in the low
    // half of word i. Returns (0,2,4,6) and (1,3,5,7).
    static void split8(uint4 words, thread float4 &even, thread float4 &odd) {
        even = as_type<float4>(words << 16);
        odd = as_type<float4>(words & 0xffff0000u);
    }
    // The inverse of `split8` for values already exact in bf16.
    static uint4 pack8(float4 even, float4 odd) {
        return (as_type<uint4>(odd) & 0xffff0000u) | (as_type<uint4>(even) >> 16);
    }
};

struct f16 {
    typedef half storage;
    typedef half native;
    static constant constexpr uint bytes = 2;
    static float load(storage value) { return float(value); }
    static storage store(float value) { return half(value); }
    static float round(float value) { return float(half(value)); }
    static void split8(uint4 words, thread float4 &even, thread float4 &odd) {
        half2 a = as_type<half2>(words.x), b = as_type<half2>(words.y);
        half2 c = as_type<half2>(words.z), d = as_type<half2>(words.w);
        even = float4(a.x, b.x, c.x, d.x);
        odd = float4(a.y, b.y, c.y, d.y);
    }
    static uint4 pack8(float4 even, float4 odd) {
        return uint4(as_type<uint>(half2(even.x, odd.x)), as_type<uint>(half2(even.y, odd.y)),
            as_type<uint>(half2(even.z, odd.z)), as_type<uint>(half2(even.w, odd.w)));
    }
};

struct f32 {
    typedef float storage;
    static constant constexpr uint bytes = 4;
    static float load(storage value) { return value; }
    static storage store(float value) { return value; }
    static float round(float value) { return value; }
};

// A dense vector (norm weights) of element type E.
template <typename E>
inline float vector_at(device const uchar *base, ulong index) {
    return E::load(reinterpret_cast<device const typename E::storage *>(base)[index]);
}

// ---------------------------------------------------------------------------
// Weight row geometry. `stride` is the byte distance between rows; plane
// offsets are within a row. Planes a representation lacks are unused.

struct rows16 {
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

inline float4 unsigned_bytes(uint word) { return float4(as_type<uchar4>(word)); }

// ---------------------------------------------------------------------------
// q4k: 4-bit unsigned codes, groups of 32 with (scale6, min6) and per-256
// (d, dmin): value = d*scale6*code - dmin*min6.

struct q4k {
    static constant constexpr uint groups = 1;       // coefficient groups per packet
    static constant constexpr uint group_size = 32;
    static constant constexpr bool biased = true;
    struct packet {
        uint4 low;
        float scale;
        float bias;
    };
    static packet load(device const uchar *row, rows16 layout, uint p) {
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
        return float2(float(b & 15u), float(b >> 4));
    }
    static float scale(thread const packet &k, uint) { return k.scale; }
    static float bias(thread const packet &k, uint) { return k.bias; }
    static float value(thread const packet &k, uint, float code) {
        return metal::fma(k.scale, code, k.bias);
    }
};

// q5k: q4k plus one high bit per code (code = low + 16*high).
struct q5k {
    static constant constexpr uint groups = 1;
    static constant constexpr uint group_size = 32;
    static constant constexpr bool biased = true;
    struct packet {
        uint4 low;
        uint high;
        float scale;
        float bias;
    };
    static packet load(device const uchar *row, rows16 layout, uint p) {
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
        return float2(float((b & 15u) | ((h & 1u) << 4)), float((b >> 4) | ((h >> 1) << 4)));
    }
    static float scale(thread const packet &k, uint) { return k.scale; }
    static float bias(thread const packet &k, uint) { return k.bias; }
    static float value(thread const packet &k, uint, float code) {
        return metal::fma(k.scale, code, k.bias);
    }
};

// q6k: 6-bit codes (low nibble + two high bits) offset by 32, int8 scales per
// 16 and per-256 d: value = d*scale8*(code - 32).
struct q6k {
    static constant constexpr uint groups = 2;
    static constant constexpr uint group_size = 16;
    static constant constexpr bool biased = true;
    struct packet {
        uint4 low;
        uint2 high;
        float scale0;
        float scale1;
    };
    static packet load(device const uchar *row, rows16 layout, uint p) {
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
        return float2(float((b & 15u) | ((h & 3u) << 4)), float((b >> 4) | ((h >> 2) << 4)));
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
struct q8 {
    static constant constexpr uint groups = 1;
    static constant constexpr uint group_size = 32;
    static constant constexpr bool biased = false;
    struct packet {
        uint4 first;
        uint4 second;
        float scale;
    };
    static packet load(device const uchar *row, rows16 layout, uint p) {
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
        float4 a = float4(as_type<char4>(words[base]));
        float4 b = float4(as_type<char4>(words[base + 1]));
        even = float4(a.x, a.z, b.x, b.z);
        odd = float4(a.y, a.w, b.y, b.w);
    }
    static float2 pair(thread const packet &k, uint step, uint j) {
        uint4 words = step < 2 ? k.first : k.second;
        uint word = words[(step & 1u) * 2u + (j >> 1)];
        char4 codes = as_type<char4>(word);
        return (j & 1u) ? float2(codes.z, codes.w) : float2(codes.x, codes.y);
    }
    static float scale(thread const packet &k, uint) { return k.scale; }
    static float bias(thread const packet &, uint) { return 0.0f; }
    static float value(thread const packet &k, uint, float code) { return k.scale * code; }
};

// Dense weights of element type E (bf16, f16 or f32). The row's final packet
// may be partial when K is not a multiple of 32; its missing elements decode
// as zero.
template <typename E>
struct dense {
    static constant constexpr uint groups = 1;
    static constant constexpr uint group_size = 32;
    static constant constexpr bool biased = false;
    struct packet {
        device const uchar *values;
        uint valid;   // elements of this packet inside the row
    };
    static packet load(device const uchar *row, rows16 layout, uint p, uint k) {
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
struct loader {
    static typename W::packet load(device const uchar *row, rows16 layout, uint p, uint) {
        return W::load(row, layout, p);
    }
};
template <typename E>
struct loader<dense<E>> {
    static typename dense<E>::packet load(device const uchar *row, rows16 layout, uint p, uint k) {
        return dense<E>::load(row, layout, p, k);
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
