// Trained block roundings of stored rows on Vulkan, evaluated exactly as
// `rounding.seismic` writes them: each run of `block` values x becomes
// code(clamp(x / s, -m, m)) * s, s the run's scale from
// a = max(max|x|, amax_floor), every quotient `seismic_div_rn` (IEEE's). The
// counterpart of `metal/lib/core/rounding.h` and `cuda/lib/core/rounding.cuh`.
//
// This file is independent of any entry ABI. The lane forms take a row held
// by a logical subgroup, lane l holding columns [l E, (l + 1) E),
// E = `LANE_E`, which the including entry defines; `rounding_hold_lanes`
// rounds to the entry's activation element (`ELEMENT_ACT`).

// A rounding as the entries take it (`rounding.seismic`'s codes): `block` 0
// rounds nothing; `element` 0 E4M3, 1 E2M1; `scale` 0 UE8M0, 1 E4M3.
struct rounding_t {
    int block;
    int element;
    int scale;
    float amax_floor;
};

// The rounding of an entry's four scalar words: its block, element and scale
// codes, and its amax floor's F32 bits.
rounding_t rounding_of(uint64_t block, uint64_t element, uint64_t scale, uint64_t amax_floor) {
    return rounding_t(int(uint(block)), int(uint(element)), int(uint(scale)), uintBitsToFloat(uint(amax_floor)));
}

// No rounding (block 0).
rounding_t rounding_none() { return rounding_t(0, 0, 0, 0.0); }

// v to the nearest code of `mantissa` bits and smallest normal exponent
// `min_exponent`, ties to even (`round_codes`).
float rounding_codes(float v, uint mantissa, int min_exponent) {
    const uint biased = (floatBitsToUint(v) >> 23) & 255u;
    const float step = uintBitsToFloat((max(biased, uint(min_exponent + 127)) - mantissa) << 23);
    return roundEven(seismic_div_rn(v, step)) * step;
}

// The least power of two at or above positive normal v.
float rounding_power_of_two_ceil(float v) {
    const uint bits = floatBitsToUint(v);
    return uintBitsToFloat(((bits >> 23) + ((bits & 0x007fffffu) != 0u ? 1u : 0u)) << 23);
}

float rounding_element_max(int element) {
    return element == 0 ? 448.0 : 6.0;
}

// The scale of a run whose largest magnitude is `amax`.
float rounding_block_scale(float amax, rounding_t rounding) {
    const float m = rounding_element_max(rounding.element);
    const float a = max(amax, rounding.amax_floor);
    return rounding.scale == 1 ? rounding_codes(seismic_div_rn(a, m), 3u, -6)
                               : rounding_power_of_two_ceil(a * seismic_div_rn(1.0, m));
}

// One value of a run with scale `s`.
float rounding_element(float x, float s, int element) {
    const float m = rounding_element_max(element);
    const float q = seismic_div_rn(x, s);
    const float clamped = q > m ? m : (q < -m ? -m : q);
    return element == 0 ? rounding_codes(clamped, 3u, -6) * s : rounding_codes(clamped, 1u, 0) * s;
}

#ifdef LANE_E

// The row rounded in place run by run. A run of `block` values lies within
// one lane (block <= E) or spans block / E aligned lanes, whose largest
// magnitude a butterfly over those lanes forms. The whole logical subgroup
// calls it.
void rounding_lanes(inout float x[LANE_E], rounding_t rounding) {
    if (rounding.block == 0)
        return;
    const uint block = uint(rounding.block);
    if (block <= LANE_E) {
        for (uint first = 0u; first < LANE_E; first += block) {
            float amax = 0.0;
            for (uint i = 0u; i < block; ++i)
                amax = max(amax, abs(x[first + i]));
            const float s = rounding_block_scale(amax, rounding);
            for (uint i = 0u; i < block; ++i)
                x[first + i] = rounding_element(x[first + i], s, rounding.element);
        }
        return;
    }
    float amax = 0.0;
    [[unroll]] for (uint i = 0u; i < LANE_E; ++i)
        amax = max(amax, abs(x[i]));
    for (uint offset = 1u; offset < block / LANE_E; offset <<= 1)
        amax = max(amax, seismic_shfl_xor_f32(amax, offset));
    const float s = rounding_block_scale(amax, rounding);
    [[unroll]] for (uint i = 0u; i < LANE_E; ++i)
        x[i] = rounding_element(x[i], s, rounding.element);
}

// A lane's values of a rounded row (block > 0) as the model holds them in its
// activation element before the rounding takes them (the portable
// `held_in_activation`): each rounded once to A. An unrounded row holds
// nothing.
void rounding_hold_lanes(inout float x[LANE_E], rounding_t rounding) {
    if (rounding.block == 0)
        return;
    [[unroll]] for (uint i = 0u; i < LANE_E; ++i)
        x[i] = element_round(ELEMENT_ACT, x[i]);
}

#endif
