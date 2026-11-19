//! The NVFP4 rows rule: activation rows as the NVFP4 operands a weight's
//! W4A4 recipe multiplies (docs/precision.md, "NVFP4 operands"; the portable
//! `nvfp4_row`, which every backend's staging follows).

use super::largest;

/// The row `values` as NVFP4 operands, in place: g, the least power of two at or above
/// amax(|row|) / (6 * 448); per 16-value block s, amax(|block|) / (6 * g)
/// onto UE4M3; each value q * g * s with q, x / (g * s) onto E2M1,
/// saturating at 6 (zero where s is). Every step rounds to nearest even, and
/// q * g * s is exact in F32. A NaN or ±Inf in the row makes amax NaN or
/// +Inf ([`largest`]), and so g NaN: every value of the row is then NaN.
#[inline(always)]
pub fn row(values: &mut [f32]) {
    let amax = largest(values);
    let bits = (amax / 2688.0).to_bits();
    let g = if amax.is_finite() {
        f32::from_bits(((bits >> 23) + u32::from(bits & 0x7f_ffff != 0)).clamp(1, 254) << 23)
    } else {
        f32::NAN
    };
    for block in values.chunks_mut(16) {
        let maximum = block.iter().fold(0.0f32, |held, value| held.max(value.abs()));
        let d = g * ue4m3(maximum / (6.0 * g));
        for value in block {
            *value = if d == 0.0 { 0.0 } else { e2m1(*value / d) * d };
        }
    }
}

/// v >= 0 onto UE4M3 (exponent bias 7, three mantissa bits, subnormal
/// quantum 2^-9), saturating at 448.
#[inline(always)]
fn ue4m3(v: f32) -> f32 {
    let exponent = ((v.to_bits() >> 23) & 255).max(121);
    let quantum = f32::from_bits((exponent - 3) << 23);
    ((v / quantum).round_ties_even() * quantum).min(448.0)
}

/// v onto E2M1's magnitudes with v's sign, saturating at 6; NaN stays NaN.
#[inline(always)]
fn e2m1(v: f32) -> f32 {
    let a = v.abs();
    let m = if a <= 0.25 {
        0.0
    } else if a < 0.75 {
        0.5
    } else if a <= 1.25 {
        1.0
    } else if a < 1.75 {
        1.5
    } else if a <= 2.5 {
        2.0
    } else if a < 3.5 {
        3.0
    } else if a <= 5.0 {
        4.0
    } else if a > 5.0 {
        6.0
    } else {
        a
    };
    if v < 0.0 { -m } else { m }
}
