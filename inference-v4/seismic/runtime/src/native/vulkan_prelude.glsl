// ---------------------------------------------------------------------------
// Seismic Vulkan device library, the counterpart of the CUDA prelude. Every
// native Vulkan source receives it after `#version`, the extensions and the
// device feature macros, ahead of its generated ABI macros.
//
// Every floating-point operation of the formed module is `NoContraction`, and
// the module runs with RTE rounding and signed-zero/Inf/NaN preservation for
// fp16 and fp32, so `+` and `*` are single correctly rounded operations (on
// the NVIDIA proprietary driver fp32 RTE is the probed default rather than a
// declared mode, §16.1). Float division is always `seismic_div_rn`, never a
// bare `/`: GLSL `/` is only 2.5 ULP, and NVIDIA's compiler traps on it
// under `RoundingModeRTE 32`.
// Subgroup helpers assume the subgroup width of 32 every pipeline requires,
// and must be executed by all 32 lanes convergently.
// ---------------------------------------------------------------------------

// Element references: one element per block, so a reference indexes like a
// pointer (`seismic_f32(address)[i].v`) and advances by the element size.
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer seismic_f32 { float v; };
layout(buffer_reference, scalar, buffer_reference_align = 2) buffer seismic_f16 { float16_t v; };
layout(buffer_reference, scalar, buffer_reference_align = 2) buffer seismic_bf16 { uint16_t v; };
layout(buffer_reference, scalar, buffer_reference_align = 1) buffer seismic_u8 { uint8_t v; };
layout(buffer_reference, scalar, buffer_reference_align = 1) buffer seismic_i8 { int8_t v; };
layout(buffer_reference, scalar, buffer_reference_align = 2) buffer seismic_u16 { uint16_t v; };
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer seismic_u32 { uint v; };
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer seismic_i32 { int v; };
layout(buffer_reference, scalar, buffer_reference_align = 8) buffer seismic_u64 { uint64_t v; };
layout(buffer_reference, scalar, buffer_reference_align = 16) buffer seismic_uvec4 { uvec4 v; };

// Conversions. bf16 rounds to nearest even in integer code (NaN becomes the
// canonical 0x7fff, as CUDA's cvt.rn.bf16.f32); fp16 conversions round by the
// module's RTE mode.
float seismic_bf16_to_f32(uint16_t value) {
    return uintBitsToFloat(uint(value) << 16);
}
uint16_t seismic_f32_to_bf16(float value) {
    const uint bits = floatBitsToUint(value);
    if ((bits & 0x7fffffffu) > 0x7f800000u)
        return uint16_t(0x7fffu);
    return uint16_t((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}
float seismic_f16_to_f32(uint16_t value) {
    return float(uint16BitsToFloat16(value));
}
uint16_t seismic_f32_to_f16(float value) {
    return float16BitsToUint16(float16_t(value));
}

// Packed pairs: `lo` occupies bits 0..15 (the lower address in memory), `hi`
// bits 16..31. Packing rounds to nearest even.
uint seismic_pack_bf16x2(float lo, float hi) {
    return uint(seismic_f32_to_bf16(lo)) | (uint(seismic_f32_to_bf16(hi)) << 16);
}
uint seismic_pack_f16x2(float lo, float hi) {
    return uint(seismic_f32_to_f16(lo)) | (uint(seismic_f32_to_f16(hi)) << 16);
}
vec2 seismic_unpack_bf16x2(uint pair) {
    return vec2(uintBitsToFloat(pair << 16), uintBitsToFloat(pair & 0xffff0000u));
}
vec2 seismic_unpack_f16x2(uint pair) {
    return vec2(seismic_f16_to_f32(uint16_t(pair & 0xffffu)), seismic_f16_to_f32(uint16_t(pair >> 16)));
}

// Exact integer helpers of the correctly rounded operations below.

// Round the 27-bit significand `q` (leading one at bit 26, three extra bits)
// with sticky bit `sticky` at biased exponent `exponent` to nearest even, and
// pack it (subnormal and overflowing results included).
float seismic_round_pack(uint sign, int exponent, uint q, bool sticky) {
    if (exponent >= 255)
        return uintBitsToFloat(sign | 0x7f800000u);
    if (exponent <= 0) {
        const uint shift = uint(min(1 - exponent, 31));
        sticky = sticky || (q & ((1u << shift) - 1u)) != 0u;
        q >>= shift;
        exponent = 0;
    }
    const uint rest = q & 7u;
    uint significand = q >> 3;
    if (rest > 4u || (rest == 4u && (sticky || (significand & 1u) != 0u)))
        significand += 1u;
    const uint base = exponent == 0 ? 0u : uint(exponent - 1) << 23;
    return uintBitsToFloat(sign | min(base + significand, 0x7f800000u));
}

// A finite nonzero operand as a 24-bit significand (leading one at bit 23)
// and an unbiased exponent.
uint seismic_normalize(uint bits, out int exponent) {
    const uint field = (bits >> 23) & 0xffu;
    const uint significand = bits & 0x7fffffu;
    if (field == 0u) {
        const int shift = 23 - findMSB(significand);
        exponent = -126 - shift;
        return significand << shift;
    }
    exponent = int(field) - 127;
    return significand | 0x800000u;
}

// Explicitly rounded arithmetic. `seismic_fma_rn` is one fused, correctly
// rounded multiply-add: formation binds `fma` to `OpFmaKHR` where the
// device has `VK_KHR_shader_fma`, and a device without it opens only if its
// `fma` is fused (§16.1).
float seismic_add_rn(float a, float b) {
    return a + b;
}
float seismic_mul_rn(float a, float b) {
    return a * b;
}
float seismic_fma_rn(float a, float b, float c) {
    return fma(a, b, c);
}

// Correctly rounded fp32 division and square root (CUDA's `/` and `sqrtf`
// under --prec-div/--prec-sqrt). Integer long division and square root handle
// every operand; division of operands and quotient well inside the normal
// range takes a fused exact-residual fast path.

float seismic_div_exact(float a, float b) {
    const uint ua = floatBitsToUint(a), ub = floatBitsToUint(b);
    const uint sign = (ua ^ ub) & 0x80000000u;
    const float nan = uintBitsToFloat(0x7fffffffu);
    if (isnan(a) || isnan(b) || (isinf(a) && isinf(b)))
        return nan;
    const bool zero_a = (ua & 0x7fffffffu) == 0u, zero_b = (ub & 0x7fffffffu) == 0u;
    if (zero_a && zero_b)
        return nan;
    if (isinf(a) || zero_b)
        return uintBitsToFloat(sign | 0x7f800000u);
    if (isinf(b) || zero_a)
        return uintBitsToFloat(sign);
    int ea, eb;
    const uint ma = seismic_normalize(ua, ea), mb = seismic_normalize(ub, eb);
    // ma / mb lies in (1/2, 2): 27 or 28 quotient bits.
    const uint64_t numerator = uint64_t(ma) << 27;
    uint64_t q = numerator / uint64_t(mb);
    bool sticky = numerator % uint64_t(mb) != 0ul;
    int exponent = ea - eb + 127 - 1;
    if (q >= (1ul << 27)) {
        sticky = sticky || (q & 1ul) != 0ul;
        q >>= 1;
        exponent += 1;
    }
    return seismic_round_pack(sign, exponent, uint(q), sticky);
}

float seismic_div_rn(float a, float b) {
    // Operands and quotient within 2^+-100 keep every residual exact.
    const float q0 = a / b;
    const float low = 7.8886091e-31, high = 1.2676506e30;
    const bool fast = abs(a) >= low && abs(a) <= high && abs(b) >= low && abs(b) <= high
        && abs(q0) >= low && abs(q0) <= high;
    if (!fast)
        return seismic_div_exact(a, b);
    // One correction from the exact residual, then the neighbor whose exact
    // residual is smallest (division has no ties).
    const float q1 = q0 + fma(-q0, b, a) / b;
    const uint bits = floatBitsToUint(q1);
    float best = q1;
    float residual = abs(fma(-q1, b, a));
    for (int step = -1; step <= 1; step += 2) {
        const float candidate = uintBitsToFloat(uint(int(bits) + step));
        const float candidate_residual = abs(fma(-candidate, b, a));
        if (candidate_residual < residual) {
            best = candidate;
            residual = candidate_residual;
        }
    }
    return best;
}

float seismic_sqrt_rn(float x) {
    const uint bits = floatBitsToUint(x);
    if (isnan(x) || (x < 0.0))
        return uintBitsToFloat(0x7fffffffu);
    if (isinf(x) || (bits & 0x7fffffffu) == 0u)
        return x;
    int exponent;
    const uint significand = seismic_normalize(bits, exponent);
    // x = significand * 2^(exponent - 23) = M * 2^t with t even.
    int t = exponent - 23 - 30;
    uint64_t m = uint64_t(significand) << 30;
    if ((t & 1) != 0) {
        m <<= 1;
        t -= 1;
    }
    uint64_t rest = m, root = 0ul, one = 1ul << 62;
    while (one > rest)
        one >>= 2;
    while (one != 0ul) {
        if (rest >= root + one) {
            rest -= root + one;
            root = (root >> 1) + one;
        } else {
            root >>= 1;
        }
        one >>= 2;
    }
    bool sticky = rest != 0ul;
    int biased = 26 + t / 2 + 127;
    if (root >= (1ul << 27)) {
        sticky = sticky || (root & 1ul) != 0ul;
        root >>= 1;
        biased += 1;
    }
    return seismic_round_pack(0u, biased, uint(root), sticky);
}

// Hardware approximations. Not correctly rounded: admitted only where an
// entry's precision gate allows them. Their divisions are `seismic_div_rn`,
// as every float division is (a bare fp32 `/` does not form on the NVIDIA
// proprietary driver under `RoundingModeRTE 32`, §16.1).
float seismic_ex2_approx(float x) {
    return exp2(x);
}
float seismic_lg2_approx(float x) {
    return log2(x);
}
float seismic_rcp_approx(float x) {
    return seismic_div_rn(1.0, x);
}
float seismic_rsqrt_approx(float x) {
    return inversesqrt(x);
}
float seismic_tanh_approx(float x) {
    // tanh(x) = 1 - 2 / (2^(2x log2 e) + 1), saturated where the exponential
    // overflows.
    const float clamped = clamp(x, -15.0, 15.0);
    return 1.0 - seismic_div_rn(2.0, exp2(2.8853900817779268 * clamped) + 1.0);
}

// Subgroup shuffles (`subgroupShuffle*`), over the whole subgroup.
uint seismic_shfl_xor_u32(uint value, uint lane_mask) {
    return subgroupShuffleXor(value, lane_mask);
}
uint seismic_shfl_idx_u32(uint value, uint source_lane) {
    return subgroupShuffle(value, source_lane);
}
uint seismic_shfl_down_u32(uint value, uint delta) {
    return subgroupShuffleDown(value, delta);
}
uint seismic_shfl_up_u32(uint value, uint delta) {
    return subgroupShuffleUp(value, delta);
}
float seismic_shfl_xor_f32(float value, uint lane_mask) {
    return subgroupShuffleXor(value, lane_mask);
}
float seismic_shfl_idx_f32(float value, uint source_lane) {
    return subgroupShuffle(value, source_lane);
}
float seismic_shfl_down_f32(float value, uint delta) {
    return subgroupShuffleDown(value, delta);
}
float seismic_shfl_up_f32(float value, uint delta) {
    return subgroupShuffleUp(value, delta);
}

// Butterfly reductions: every lane receives the result, combined in the fixed
// order xor 16, 8, 4, 2, 1 (bitwise identical on every lane and every run).
// `subgroupAdd` on floats is not used: its order is implementation-defined.
float seismic_subgroup_sum_f32(float value) {
    [[unroll]] for (uint mask = 16u; mask > 0u; mask >>= 1)
        value = value + subgroupShuffleXor(value, mask);
    return value;
}
float seismic_subgroup_max_f32(float value) {
    [[unroll]] for (uint mask = 16u; mask > 0u; mask >>= 1)
        value = max(value, subgroupShuffleXor(value, mask));
    return value;
}

// Integer subgroup reductions, where the order does not matter; every lane
// receives the result.
uint seismic_redux_add_u32(uint value) {
    return subgroupAdd(value);
}
uint seismic_redux_min_u32(uint value) {
    return subgroupMin(value);
}
uint seismic_redux_max_u32(uint value) {
    return subgroupMax(value);
}
int seismic_redux_add_s32(int value) {
    return subgroupAdd(value);
}
int seismic_redux_min_s32(int value) {
    return subgroupMin(value);
}
int seismic_redux_max_s32(int value) {
    return subgroupMax(value);
}

// Four-way byte dot products accumulated into `acc` without saturation (as
// CUDA's dp4a): bytes of `a` and `b` in little-endian order, signed (s8) or
// unsigned (u8).
int seismic_dp4a_s8(int a, int b, int acc) {
    return acc + dotPacked4x8EXT(a, b);
}
int seismic_dp4a_u8s8(uint a, int b, int acc) {
    return acc + dotPacked4x8EXT(a, b);
}
