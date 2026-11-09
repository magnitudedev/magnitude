// Trained block roundings of stored rows on Metal, evaluated exactly as
// `rounding.seismic` writes them: each run of `block` values x becomes
// code(clamp(x / s, -m, m)) * s, s the run's scale from
// a = max(max|x|, amax_floor), the quotient a division (Metal compiles
// without fast math, so it is IEEE's). The counterpart of
// `cpu/lib/core/rounding.rs`.
//
// This file is independent of any entry ABI.

namespace rounding {

// A rounding as the entries take it (`rounding.seismic`'s codes): `block` 0
// rounds nothing; `element` 0 E4M3, 1 E2M1; `scale` 0 UE8M0, 1 E4M3.
struct Rounding {
    int block;
    int element;
    int scale;
    float amax_floor;
};

// v to the nearest code of `mantissa` bits and smallest normal exponent
// `min_exponent`, ties to even (`round_codes`).
inline float round_codes(float v, uint mantissa, int min_exponent) {
    const uint biased = (as_type<uint>(v) >> 23) & 255u;
    const float step = as_type<float>((metal::max(biased, uint(min_exponent + 127)) - mantissa) << 23);
    return metal::rint(v / step) * step;
}

// The least power of two at or above positive normal v.
inline float power_of_two_ceil(float v) {
    const uint bits = as_type<uint>(v);
    return as_type<float>(((bits >> 23) + ((bits & 0x007fffffu) != 0u ? 1u : 0u)) << 23);
}

inline float element_max(int element) {
    return element == 0 ? 448.0f : 6.0f;
}

// The scale of a run whose largest magnitude is `amax`.
inline float block_scale(float amax, Rounding rounding) {
    const float m = element_max(rounding.element);
    const float a = metal::max(amax, rounding.amax_floor);
    return rounding.scale == 1 ? round_codes(a / m, 3u, -6) : power_of_two_ceil(a * (1.0f / m));
}

// One value of a run with scale `s`.
inline float round_element(float x, float s, int element) {
    const float m = element_max(element);
    const float q = x / s;
    const float clamped = q > m ? m : (q < -m ? -m : q);
    return element == 0 ? round_codes(clamped, 3u, -6) * s : round_codes(clamped, 1u, 0) * s;
}

// A row of 32 E values spread over a simdgroup, lane l holding columns
// [l E, (l + 1) E), rounded in place run by run. A run of `block` values
// lies within one lane (block <= E) or spans block / E aligned lanes, whose
// largest magnitude a butterfly over those lanes forms. The whole simdgroup
// calls it.
template <uint E>
inline void round_lanes(thread float (&x)[E], Rounding rounding) {
    if (rounding.block == 0)
        return;
    const uint block = uint(rounding.block);
    if (block <= E) {
        for (uint first = 0; first < E; first += block) {
            float amax = 0.0f;
            for (uint i = 0; i < block; ++i)
                amax = metal::max(amax, metal::abs(x[first + i]));
            const float s = block_scale(amax, rounding);
            for (uint i = 0; i < block; ++i)
                x[first + i] = round_element(x[first + i], s, rounding.element);
        }
        return;
    }
    float amax = 0.0f;
    for (uint i = 0; i < E; ++i)
        amax = metal::max(amax, metal::abs(x[i]));
    for (ushort offset = 1; offset < ushort(block / E); offset <<= 1)
        amax = metal::max(amax, metal::simd_shuffle_xor(amax, offset));
    const float s = block_scale(amax, rounding);
    for (uint i = 0; i < E; ++i)
        x[i] = round_element(x[i], s, rounding.element);
}

} // namespace rounding
