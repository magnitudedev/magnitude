// Seismic native library (Vulkan): the per-group activation rules of the
// products over packed weights. `#include <seismic/activation.glsl>`. A
// kernel forms its activation operands (their layout, staging and the lanes
// its peaks reduce over) and calls these for each group's values.

// ---------------------------------------------------------------------------
// Non-finite activations (`seismic-native-cpu` `quant.rs`): a NaN or ±Inf in
// a group makes its scale NaN and its codes 0, so its products are NaN, as
// the dense product's would be. A group's peak is the maximum of its
// magnitudes' bits, whose integer order is the magnitudes' with every NaN
// above +Inf, so a NaN wins (SPIR-V's float maxima leave a NaN operand's
// result undefined); its scale maps a +Inf peak to NaN once per group.

// The peak of `held` (a peak's bits) and eight values' magnitudes.
uint nvfp4_peak(uint held, vec4 even, vec4 odd) {
    const uvec4 m = max(floatBitsToUint(even) & 0x7fffffffu, floatBitsToUint(odd) & 0x7fffffffu);
    return max(held, max(max(m.x, m.y), max(m.z, m.w)));
}

// ---------------------------------------------------------------------------
// NVFP4 operands (docs/precision.md, "NVFP4 operands"; the portable
// `nvfp4_row`): a product over NVFP4 weights takes each row x as q * g * s,
// g the row's global scale (the least power of two with 2688 g >= amax|x|),
// s a 16-value block's UE4M3 scale (nearest amax|block| / (6 g)) and q the
// E2M1 code nearest x / (g s), saturating at 6, every rounding to nearest
// even. The portable body divides with F32's correctly rounded quotient,
// which rounds onto each grid as the exact quotient does; GLSL's division is
// not correctly rounded, so every rounding here is decided without one:
// each grid point or midpoint times its divisor is exact in F32, and the
// dividend against it decides the exact quotient. The row is g's group: a
// NaN or ±Inf in the row makes g NaN (its peak `nvfp4_peak`'s), so
// every block's g s and every operand of the row is NaN.

// 2^(e - 127) for a biased exponent e in 1 .. 254.
float nvfp4_detail_power(uint e) { return uintBitsToFloat(e << 23); }

// The global scale of a row whose largest magnitude is amax: the binade of
// an approximate amax / 2688, corrected by one exact comparison. A finite
// row's g is at most 2^117, so its inverse is normal; a NaN or +Inf amax
// gives NaN.
float nvfp4_global_scale(float amax) {
    if (!(amax < uintBitsToFloat(0x7f800000u)))
        return uintBitsToFloat(0x7fc00000u);
    const uint bits = floatBitsToUint(amax * (1.0 / 2688.0));
    const uint near = clamp((bits >> 23) + ((bits & 0x7fffffu) != 0u ? 1u : 0u), 1u, 254u);
    if (near > 1u && 2688.0 * nvfp4_detail_power(near - 1u) >= amax)
        return nvfp4_detail_power(near - 1u);
    if (near < 254u && 2688.0 * nvfp4_detail_power(near) < amax)
        return nvfp4_detail_power(near + 1u);
    return nvfp4_detail_power(near);
}

// g s for a block whose largest magnitude is block_max: s is the UE4M3
// value (exponent bias 7, three mantissa bits, subnormal quantum 2^-9)
// nearest w / 6 for w = block_max / g (exact: g is a power of two). The
// quantum is that of an approximate w / 6's binade (both binades' grids
// hold the boundary, the nearest value to any quotient that rounds across
// it); t = w / quantum is exact, and the approximate index t / 6 is
// corrected against its exact midpoints 6 index +- 3. A finite row's w is
// at most 2688, so s is at most 448. A NaN g gives NaN.
float nvfp4_scale(float block_max, float g) {
    const float w = block_max * nvfp4_detail_power(254u - (floatBitsToUint(g) >> 23));
    const uint exponent = max((floatBitsToUint(w * (1.0 / 6.0)) >> 23) & 255u, 121u);
    const float t = w * nvfp4_detail_power(257u - exponent);
    const int near = int(roundEven(t * (1.0 / 6.0)));
    const bool odd = (near & 1) != 0;
    const float above = float(6 * near + 3), below = float(6 * near - 3);
    const int index = (t > above || (t == above && odd)) ? near + 1
        : (t < below || (t == below && odd))             ? near - 1
                                                         : near;
    return g * (float(index) * nvfp4_detail_power(exponent - 3u));
}

// x's E2M1 code nearest x / d, saturating at 6, times d: each threshold
// times d is exact, so |x| against it decides the exact quotient (ties to
// the even code). A zero d gives (positive) zero, a NaN d NaN.
float nvfp4_value(float x, float d) {
    if (d == 0.0)
        return 0.0;
    const float a = abs(x);
    const float m = a <= 0.25 * d ? 0.0
        : a < 0.75 * d            ? 0.5
        : a <= 1.25 * d           ? 1.0
        : a < 1.75 * d            ? 1.5
        : a <= 2.5 * d            ? 2.0
        : a < 3.5 * d             ? 3.0
        : a <= 5.0 * d            ? 4.0
                                  : 6.0;
    return (x < 0.0 ? -m : m) * d;
}

vec4 nvfp4_values(vec4 x, float d) {
    return vec4(nvfp4_value(x.x, d), nvfp4_value(x.y, d), nvfp4_value(x.z, d), nvfp4_value(x.w, d));
}

float nvfp4_magnitude(vec4 even, vec4 odd) {
    const vec4 m = max(abs(even), abs(odd));
    return max(max(m.x, m.y), max(m.z, m.w));
}
