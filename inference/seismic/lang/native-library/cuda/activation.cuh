// Seismic native library (CUDA): the per-group activation rules of the
// operand forms whose products quantize activation rows, over the CUDA
// prelude. `#include <seismic/activation.cuh>`. A kernel owns the rows'
// layout, the lanes a group's peak is reduced over and the stores; these
// functions are each rule's arithmetic, which every backend's matches bit for
// bit (`seismic-native-cpu` `quant.rs`).
//
// Non-finite activations: a NaN or ±Inf in a group makes its scale NaN and
// its codes 0, so its products are NaN, as the dense product's would be. A
// group's peak takes the NaN-propagating maximum (`peak_with`: one FMNMX, as
// `fmaxf`), and its scale maps a +Inf peak to NaN once per group.

#pragma once

#include <seismic/element.cuh>

namespace packets {

using element::u32;

namespace detail {

// The smaller of a and b, NaN if either is.
__device__ __forceinline__ float min_nan(float a, float b) {
    float result;
    asm("min.NaN.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
    return result;
}

// v >= 0 onto UE4M3 (exponent bias 7, three mantissa bits, subnormal
// quantum 2^-9), saturating at 448; NaN stays NaN.
__device__ __forceinline__ float ue4m3(float v) {
    const u32 exponent = max((__float_as_uint(v) >> 23) & 255u, 121u);
    const float quantum = __uint_as_float((exponent - 3u) << 23);
    return min_nan(__fmul_rn(rintf(__fdiv_rn(v, quantum)), quantum), 448.0f);
}

// v onto E2M1's magnitudes 0, 0.5, 1, 1.5, 2, 3, 4 and 6 with v's sign,
// saturating at 6; NaN stays NaN.
__device__ __forceinline__ float e2m1(float v) {
    const float a = fabsf(v);
    const float m = a <= 0.25f   ? 0.0f
                    : a < 0.75f  ? 0.5f
                    : a <= 1.25f ? 1.0f
                    : a < 1.75f  ? 1.5f
                    : a <= 2.5f  ? 2.0f
                    : a < 3.5f   ? 3.0f
                    : a <= 5.0f  ? 4.0f
                    : a > 5.0f   ? 6.0f
                                 : a;
    return v < 0.0f ? -m : m;
}

} // namespace detail

// The larger of `peak` and |x|, NaN if either is: a group's peak, taken over
// its values in any order.
__device__ __forceinline__ float peak_with(float peak, float x) {
    float result;
    asm("max.NaN.f32 %0, %1, %2;" : "=f"(result) : "f"(peak), "f"(fabsf(x)));
    return result;
}

// ---------------------------------------------------------------------------
// int8 activations (the INT8 form): a row is quantized per 32-value group,
// x ~= d * q.

// The scale of a group whose largest magnitude is `maximum` (`peak_with`'s):
// `maximum / 127` rounded to nearest, or where that is subnormal, rounded up
// onto the subnormal grid, so `maximum / d <= 127` and `d` is zero only for
// an all-zero group; NaN where `maximum` is NaN or +Inf.
__device__ __forceinline__ float int8_scale_of(float maximum) {
    if (!(maximum < __int_as_float(0x7f800000)))
        return __int_as_float(0x7fffffff);
    const float least_normal = __uint_as_float(0x00800000u);
    float d = __fdiv_rn(maximum, 127.0f);
    // F64 holds `d * 127` exactly.
    if (d < least_normal && (double)d * 127.0 < (double)maximum)
        d = __uint_as_float(__float_as_uint(d) + 1u);
    return d;
}

// The code of `x` under the scale `d` (`int8_scale_of`): the exact quotient
// rounded once, half away from zero; a NaN scale, and a zero one, code 0. Where
// `d` is normal the F32 quotient is within half an ulp of the exact one, so
// only a quotient that lands on a half decides by the sign of its remainder;
// where `d` is subnormal the F64 quotient of two F32 values rounds like the
// exact one.
__device__ __forceinline__ int int8_code_of(float x, float d) {
    if (d == 0.0f)
        return 0;
    const float magnitude = fabsf(x);
    float rounded;
    if (d < __uint_as_float(0x00800000u)) {
        const double quotient = (double)magnitude / (double)d;
        if (!(quotient == quotient))
            return 0;
        rounded = (float)round(quotient);
    } else {
        const float quotient = __fdiv_rn(magnitude, d);
        if (!(quotient == quotient))
            return 0;
        rounded = roundf(quotient);
        // A tie in F32 is below the exact quotient where the remainder is negative.
        if (quotient - truncf(quotient) == 0.5f && fmaf(-quotient, d, magnitude) < 0.0f)
            rounded = truncf(quotient);
    }
    return x < 0.0f ? -(int)rounded : (int)rounded;
}

// ---------------------------------------------------------------------------
// NVFP4 operand rows (docs/precision.md, "NVFP4 operands"; the portable
// `nvfp4_row`): a product over NVFP4 weights takes each row x as
// q * g * s, which a 16-bit activation holds exactly. g, the row's global
// scale, is the least power of two at or above amax(|x|) / (6 * 448); s, a
// 16-value block's scale, is amax(|block|) / (6 * g) rounded to nearest even
// onto UE4M3; q is x / (g * s) rounded to nearest even onto E2M1, saturating
// at 6 (zero where s is). Every operation rounds once, to nearest even. The
// row is g's group: a NaN or ±Inf in the row makes g NaN, so every block's s
// is NaN (UE4M3's NaN, 0x7F) and every code 0.
namespace nvfp4 {

// The global scale of a row whose peak is `amax` (`peak_with`'s): NaN where
// amax is NaN or +Inf.
__device__ __forceinline__ float global_scale(float amax) {
    if (!(amax < __int_as_float(0x7f800000)))
        return __int_as_float(0x7fffffff);
    const u32 bits = __float_as_uint(__fdiv_rn(amax, 2688.0f));
    const u32 exponent = (bits >> 23) + ((bits & 0x7fffffu) != 0u ? 1u : 0u);
    return __uint_as_float(min(max(exponent, 1u), 254u) << 23);
}

// The UE4M3 scale s of a block whose largest magnitude is `maximum`, under
// the row's global scale g; NaN under a NaN g (the block's own peak need not
// carry NaN).
__device__ __forceinline__ float block_scale(float maximum, float g) {
    return detail::ue4m3(__fdiv_rn(maximum, __fmul_rn(6.0f, g)));
}

// The E2M1 code value q of x under the block's operand scale d = g * s:
// zero where d is not positive (a zero block, or a NaN row).
__device__ __forceinline__ float code_value(float x, float d) {
    return d > 0.0f ? detail::e2m1(__fdiv_rn(x, d)) : 0.0f;
}

// The field of s, a UE4M3 value (`block_scale`'s): exponent bias 7 above the
// subnormal quantum 2^-9; a normal s's F32 exponent and three mantissa bits
// rebiased, and NaN's 0x7F (whose F32 exponent field exceeds every finite
// s's).
__device__ __forceinline__ u32 ue4m3_field(float s) {
    return s < 0.015625f ? (u32)__fmul_rn(s, 512.0f) : min((__float_as_uint(s) >> 20) - (120u << 3), 0x7fu);
}

// The code of q, an E2M1 value (`code_value`'s): the magnitude's index among
// 0, 0.5, 1, 1.5, 2, 3, 4, 6 with q's sign at bit 3 (zero positive).
__device__ __forceinline__ u32 e2m1_code(float q) {
    const float a = fabsf(q);
    const u32 index = a < 2.0f ? (u32)__fmul_rn(a, 2.0f) : a < 4.0f ? 2u + (u32)a : a < 6.0f ? 6u : 7u;
    return index | (q < 0.0f ? 8u : 0u);
}

} // namespace nvfp4

} // namespace packets
