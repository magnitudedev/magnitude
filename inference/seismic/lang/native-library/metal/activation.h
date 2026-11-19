// Seismic native library (Metal): the per-group activation rules of the
// operand forms, `#include <seismic/activation.h>`: how a group of
// activation values becomes its scale and codes for the int8 product
// (`int8_scale_of`, `int8_code_of`), for NVFP4 operand rows (`nvfp4`) and
// for the packed form's lanes (`packing_lane_scales`). A kernel keeps the
// layout of its fragments and their stores, its fused prologue and the lane
// mapping of a group's peak (which lanes it joins); the rule a group's
// values are coded by is this file's, held to the CPU reference
// (`seismic-native-cpu` `quant.rs`) bit for bit.
//
// Non-finite activations: a NaN or ±Inf in a group makes its scale NaN and
// its codes 0, so its products are NaN, as the dense product's would be. A
// group's peak is the maximum of its magnitudes' bits (`magnitude_peak`),
// whose integer order is the magnitudes' with every NaN above +Inf, so a NaN
// wins (`metal::max` of floats drops it); its scale maps a +Inf peak to NaN
// once per group.

namespace packets {

// The peak of `held` (a peak's bits) and eight values' magnitudes.
inline uint magnitude_peak(uint held, float4 even, float4 odd) {
    const uint4 m = metal::max(as_type<uint4>(even) & 0x7fffffffu, as_type<uint4>(odd) & 0x7fffffffu);
    return metal::max(held, metal::max(metal::max(m.x, m.y), metal::max(m.z, m.w)));
}

// ---------------------------------------------------------------------------
// int8: a 32-column block's scale and codes.

// A 32-column block's scale: its largest magnitude / 127 rounded to nearest
// F16 (the scale the fold stores), or where that is subnormal (below 2^-14),
// rounded up onto the F16 grid, so `peak / d <= 127` and `d` is zero only for
// an all-zero block; NaN where `peak` is NaN or +Inf. The CPU's rule on F16's
// grid (`quant.rs`).
inline float int8_scale_of(float peak) {
    if (!(peak < INFINITY))
        return NAN;
    const half stored = half(peak / 127.0f);
    const float d = float(stored);
    // `d * 127` is exact: 18 bits.
    if (d < 0x1p-14f && d * 127.0f < peak)
        return float(as_type<half>(ushort(as_type<ushort>(stored) + 1u)));
    return d;
}

// The code of `x` under `d`, a scale of `int8_scale_of`: the exact quotient
// rounded once, half away from zero; a NaN scale and a zero one code 0. The
// quotient picks the candidate and the code's two boundaries decide it
// exactly: (2n +- 1) d / 2 is at most 19 bits, so exact in F32.
inline int int8_code_of(float x, float d) {
    const float a = metal::abs(x);
    const float quotient = a / d;
    if (!(quotient < 255.0f))
        return 0;
    int n = int(quotient + 0.5f);
    if (a >= float(2 * n + 1) * d * 0.5f)
        ++n;
    else if (a < float(2 * n - 1) * d * 0.5f)
        --n;
    return x < 0.0f ? -n : n;
}

// ---------------------------------------------------------------------------
// NVFP4 operands (docs/precision.md, "NVFP4 operands"; the portable
// `nvfp4_row`): a product over NVFP4 weights takes each row x as q * g * s,
// g the row's global scale (the least power of two with 2688 g >= amax|x|),
// s a 16-value block's UE4M3 scale (nearest amax|block| / (6 g)) and q the
// E2M1 code nearest x / (g s), saturating at 6, every rounding to nearest
// even. The portable body divides with F32's correctly rounded quotient,
// which rounds onto each grid as the exact quotient does; Metal's division
// is not correctly rounded, so every rounding here is decided without one:
// each grid point or midpoint times its divisor is exact in F32, and the
// dividend against it decides the exact quotient. The row is g's group: a
// NaN or ±Inf in the row makes g NaN (its peak `magnitude_peak`'s), so
// every block's g s and every operand of the row is NaN.

namespace nvfp4 {

// 2^(e - 127) for a biased exponent e in 1 .. 254.
inline float power(uint e) { return as_type<float>(e << 23); }

// The global scale of a row whose largest magnitude is amax: the binade of
// an approximate amax / 2688, corrected by one exact comparison. A finite
// row's g is at most 2^117, so its inverse is normal; a NaN or +Inf amax
// gives NaN.
inline float global_scale(float amax) {
    if (!(amax < INFINITY))
        return NAN;
    const uint bits = as_type<uint>(amax * (1.0f / 2688.0f));
    const uint near = metal::clamp((bits >> 23) + ((bits & 0x7fffffu) != 0u ? 1u : 0u), 1u, 254u);
    if (near > 1u && 2688.0f * power(near - 1u) >= amax)
        return power(near - 1u);
    if (near < 254u && 2688.0f * power(near) < amax)
        return power(near + 1u);
    return power(near);
}

// g s for a block whose largest magnitude is block_max: s is the UE4M3
// value (exponent bias 7, three mantissa bits, subnormal quantum 2^-9)
// nearest w / 6 for w = block_max / g (exact: g is a power of two). The
// quantum is that of an approximate w / 6's binade (both binades' grids
// hold the boundary, the nearest value to any quotient that rounds across
// it); t = w / quantum is exact, and the approximate index t / 6 is
// corrected against its exact midpoints 6 index +- 3. A finite row's w is
// at most 2688, so s is at most 448. A NaN g gives NaN.
inline float scale(float block_max, float g) {
    const float w = block_max * power(254u - (as_type<uint>(g) >> 23));
    const uint exponent = metal::max((as_type<uint>(w * (1.0f / 6.0f)) >> 23) & 255u, 121u);
    const float t = w * power(257u - exponent);
    const int near = int(metal::rint(t * (1.0f / 6.0f)));
    const bool odd = (near & 1) != 0;
    const float above = float(6 * near + 3), below = float(6 * near - 3);
    const int index = (t > above || (t == above && odd)) ? near + 1
        : (t < below || (t == below && odd))             ? near - 1
                                                         : near;
    return g * (float(index) * power(exponent - 3u));
}

// x's E2M1 code nearest x / d, saturating at 6, times d: each threshold
// times d is exact, so |x| against it decides the exact quotient (ties to
// the even code). A zero d gives (positive) zero, a NaN d NaN.
inline float value(float x, float d) {
    if (d == 0.0f)
        return 0.0f;
    const float a = metal::abs(x);
    const float m = a <= 0.25f * d ? 0.0f
        : a < 0.75f * d            ? 0.5f
        : a <= 1.25f * d           ? 1.0f
        : a < 1.75f * d            ? 1.5f
        : a <= 2.5f * d            ? 2.0f
        : a < 3.5f * d             ? 3.0f
        : a <= 5.0f * d            ? 4.0f
                                   : 6.0f;
    return (x < 0.0f ? -m : m) * d;
}

inline float4 values(float4 x, float d) {
    return float4(value(x.x, d), value(x.y, d), value(x.z, d), value(x.w, d));
}

// A block's largest magnitude of eight values (a block's peak: a NaN in it
// has made the row's g NaN already).
inline float magnitude(float4 even, float4 odd) {
    const float4 m = metal::max(metal::abs(even), metal::abs(odd));
    return metal::max(metal::max(m.x, m.y), metal::max(m.z, m.w));
}

} // namespace nvfp4

// ---------------------------------------------------------------------------
// The packed form's lane scales: each token of an element (a lane, by the
// kernel's fields F: `F::limit(l, n)`, a lane's sum limit over a run of n
// columns, and `F::codes(l)`, the largest code it holds) is coded under a
// scale (64 / gain, what the fold multiplies) certified by the format's
// bound and, BOUNDED, by the tensors' bound of the run.

// The gain limits of a run of n = 32 / folds columns against codes minus
// `centre` (each at most `centre` in magnitude): a run's sum is at most
// centre |q|_1, and codes rounded to multiples of s have
// |q|_1 <= g |x|_1 + n s / 2, so the largest gain under a sum limit is
// (limit / centre - n s / 2) / |x|_1. Returned as what a run's scale
// (64 / gain) is per unit of |x|_1: 64 / limit over 1 - 2^-10, which covers
// the F32 rounding of the norm (at most 2^-19 of a sum of 32 magnitudes, in
// any order) and of the products with it. One lane's, for a sum limit and
// code step s.
inline float packing_limit(float limit, uint centre, uint folds, float step) {
    const float n = 32.0f / float(folds);
    return (64.0f / 0.9990234375f) / (limit / float(centre) - 0.5f * n * step);
}

// The certified scale of lane `l` of fields F for a token of norms `norms`
// (the sum of its magnitudes, the largest, the sum of its squares) in a run
// of n = 32 / FOLDS columns, coded in multiples of `step`, against codes
// minus `centre`. The format's bound certifies `packing_limit` per unit of
// |x|_1; with the tensors' bound B^2 of the run (`squares`, BOUNDED),
// Cauchy-Schwarz with the rounding of the codes inside,
// (g |x|_2 + sqrt(n) s / 2) B <= limit, certifies its own (none in a run
// whose codes are all the centre: the inverse is then infinite, and so is
// the room), and the smaller of the two scales (the larger gain) holds. Not
// yet held, nor capped by the lane's codes (`packing_lane_scales`).
template <typename F, uint FOLDS, bool BOUNDED>
inline float packing_lane_scale(uint l, float3 norms, float squares, uint centre,
    float step) {
    const float n = 32.0f / float(FOLDS), limit = F::limit(l, n);
    const float certified = norms.x * packing_limit(limit, centre, FOLDS, step);
    if (!BOUNDED)
        return certified;
    const float root = FOLDS == 1 ? 5.65685424949238f : 4.0f;
    const float room = limit * metal::rsqrt(squares) - 0.5f * root * step;
    return metal::min(certified, (64.0f / 0.9990234375f) * metal::sqrt(norms.z) / room);
}

// A scale as the fold holds it, in 16 bits: its fraction rounded up to 8
// bits, so that the gain, the reciprocal of what is held, is at most the one
// admitted.
inline float packing_held(float scale) { return as_type<float>((as_type<uint>(scale) + 0x7fffu) & 0xffff8000u); }
// The gain of a held scale: zero for a zero or NaN scale.
inline float packing_gain(float scale) { return scale > 0.0f ? 64.0f / scale : 0.0f; }

// The held scales of an element's two tokens of one run (`scales`, by lane:
// the bottom one, then the top), from their norms (`norms`, by lane), of
// fields F against codes minus `centre`; returned, the step of the bottom
// lane's codes. Each lane's scale is certified (`packing_lane_scale`). A
// lane's codes stay within its `codes`, and the top lane's within 2 codes + 1
// beside even bottom codes (the element is then even, so exact below 2^25);
// the element takes the wider top where it lowers the sum of the top and
// bottom tokens' squared relative steps (64 / gain over |x|_1, the bottom
// token's doubled by its step), which only a top capped by its codes can.
template <typename F, uint FOLDS, bool BOUNDED>
inline float packing_lane_scales(thread const float3 (&norms)[2], float squares, uint centre,
    thread float (&scales)[2]) {
    // Each lane's certified scale, then the bottom lane's with even codes.
    const float certified_top = packing_lane_scale<F, FOLDS, BOUNDED>(1, norms[1], squares, centre, 1.0f);
    const float certified_low = packing_lane_scale<F, FOLDS, BOUNDED>(0, norms[0], squares, centre, 1.0f);
    const float certified_even = packing_lane_scale<F, FOLDS, BOUNDED>(0, norms[0], squares, centre, 2.0f);
    constexpr float NARROW = 64.0f / F::codes(1), WIDE = 64.0f / (2.0f * F::codes(1) + 1.0f);
    constexpr float BOTTOM = 64.0f / F::codes(0);
    const float3 top = norms[1], low = norms[0];
    const float narrow = packing_held(metal::max(certified_top, top.y * NARROW));
    const float wide = packing_held(metal::max(certified_top, top.y * WIDE));
    const float fine = packing_held(metal::max(certified_low, low.y * BOTTOM));
    const float even = packing_held(metal::max(certified_even, low.y * BOTTOM));
    const bool widen = narrow * narrow * low.x * low.x + fine * fine * top.x * top.x
        > wide * wide * low.x * low.x + 4.0f * even * even * top.x * top.x;
    scales[1] = widen ? wide : narrow;
    scales[0] = widen ? even : fine;
    return widen ? 2.0f : 1.0f;
}

} // namespace packets
