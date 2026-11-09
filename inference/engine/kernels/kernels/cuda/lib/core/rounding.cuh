// Trained block roundings of stored rows on CUDA, evaluated exactly as
// `rounding.seismic` writes them: each run of `block` values x becomes
// code(clamp(x / s, -m, m)) * s, s the run's scale from
// a = max(max|x|, amax_floor), the quotient a division (the natives compile
// with --prec-div=true, --fmad=false, --ftz=false, so it is IEEE's). The
// counterpart of `metal/lib/core/rounding.h` and `cpu/lib/core/rounding.rs`.
//
// This file is independent of any entry ABI.

#pragma once

namespace rounding {

// A rounding as the entries take it (`rounding.seismic`'s codes): `block` 0
// rounds nothing; `element` 0 E4M3, 1 E2M1; `scale` 0 UE8M0, 1 E4M3.
struct Rounding {
    int block;
    int element;
    int scale;
    float amax_floor;
};

// No rounding (block 0).
__device__ __forceinline__ Rounding none() { return Rounding{0, 0, 0, 0.0f}; }

// v to the nearest code of `mantissa` bits and smallest normal exponent
// `min_exponent`, ties to even (`round_codes`).
__device__ __forceinline__ float round_codes(float v, unsigned mantissa, int min_exponent) {
    const unsigned biased = (__float_as_uint(v) >> 23) & 255u;
    const float step = __uint_as_float((max(biased, static_cast<unsigned>(min_exponent + 127)) - mantissa) << 23);
    return rintf(v / step) * step;
}

// The least power of two at or above positive normal v.
__device__ __forceinline__ float power_of_two_ceil(float v) {
    const unsigned bits = __float_as_uint(v);
    return __uint_as_float(((bits >> 23) + ((bits & 0x007fffffu) != 0u ? 1u : 0u)) << 23);
}

__device__ __forceinline__ float element_max(int element) { return element == 0 ? 448.0f : 6.0f; }

// The scale of a run whose largest magnitude is `amax`.
__device__ __forceinline__ float block_scale(float amax, Rounding rounding) {
    const float m = element_max(rounding.element);
    const float a = fmaxf(amax, rounding.amax_floor);
    return rounding.scale == 1 ? round_codes(a / m, 3u, -6) : power_of_two_ceil(a * (1.0f / m));
}

// One value of a run with scale `s`.
__device__ __forceinline__ float round_element(float x, float s, int element) {
    const float m = element_max(element);
    const float q = x / s;
    const float clamped = q > m ? m : (q < -m ? -m : q);
    return element == 0 ? round_codes(clamped, 3u, -6) * s : round_codes(clamped, 1u, 0) * s;
}

// A row of 32 E values spread over a warp, lane l holding columns
// [l E, (l + 1) E), rounded in place run by run. A run of `block` values
// lies within one lane (block <= E) or spans block / E aligned lanes, whose
// largest magnitude a butterfly over those lanes forms. The whole warp calls
// it.
template <int E>
__device__ __forceinline__ void round_lanes(float (&x)[E], Rounding rounding) {
    if (rounding.block == 0) return;
    const int block = rounding.block;
    if (block <= E) {
        for (int first = 0; first < E; first += block) {
            float amax = 0.0f;
            for (int i = 0; i < block; ++i) amax = fmaxf(amax, fabsf(x[first + i]));
            const float s = block_scale(amax, rounding);
            for (int i = 0; i < block; ++i) x[first + i] = round_element(x[first + i], s, rounding.element);
        }
        return;
    }
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < E; ++i) amax = fmaxf(amax, fabsf(x[i]));
    for (unsigned offset = 1; offset < static_cast<unsigned>(block / E); offset <<= 1)
        amax = fmaxf(amax, seismic_shfl_xor_f32(amax, offset));
    const float s = block_scale(amax, rounding);
#pragma unroll
    for (int i = 0; i < E; ++i) x[i] = round_element(x[i], s, rounding.element);
}

// A lane's values of a rounded row (block > 0) as the model holds them in
// its activation element `A` before the rounding takes them (the portable
// `held_in_activation`): each rounded once to A. An unrounded row holds
// nothing.
template <typename A, int E>
__device__ __forceinline__ void hold_lanes(float (&x)[E], Rounding rounding) {
    if (rounding.block == 0) return;
#pragma unroll
    for (int i = 0; i < E; ++i) x[i] = A::round(x[i]);
}

}  // namespace rounding
