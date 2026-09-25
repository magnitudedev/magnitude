use super::attention::GROUP;
use seismic::cpu::{Dense, F16};

/// The portable `affine_encode` of one vector `x` at `bits` code bits: per
/// group of GROUP values, zero = f16(min), scale = f16((max - min) / L) and
/// code = min(L, u32(fma(x - zero, 1 / scale, 0.5))) (0 when scale is 0),
/// packed 32 / bits codes per word; (scale, zero) F16 bits per group.
pub fn affine_encode(x: &[f32], bits: u32, codes: &mut [u32], coefficients: &mut [u16]) {
    let levels = (1u32 << bits) - 1;
    let per = (32 / bits) as usize;
    codes.fill(0);
    for (group, values) in x.chunks_exact(GROUP).enumerate() {
        let (low, high) = values
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(low, high), value| (low.min(*value), high.max(*value)));
        let zero = F16::narrow(low);
        let scale = F16::narrow((high - low) / levels as f32);
        let step = F16::widen(scale);
        let inverse = if step > 0.0 { 1.0 / step } else { 0.0 };
        let base = F16::widen(zero);
        for (offset, value) in values.iter().enumerate() {
            let i = group * GROUP + offset;
            // `as` saturates: a negative or NaN code is 0, as the portable
            // `u32` conversion.
            let code = ((*value - base).mul_add(inverse, 0.5) as u32).min(levels);
            codes[i / per] |= code << ((i % per) as u32 * bits);
        }
        coefficients[2 * group] = scale;
        coefficients[2 * group + 1] = zero;
    }
}

/// The portable `affine_decode` of one vector at `bits` code bits:
/// `fma(code, scale, zero)` with the value's group pair.
#[inline(always)]
pub fn affine_decode(codes: &[u32], coefficients: &[u16], bits: u32, out: &mut [f32]) {
    let mask = (1u32 << bits) - 1;
    let per = (32 / bits) as usize;
    for (group, values) in out.chunks_exact_mut(GROUP).enumerate() {
        let scale = F16::widen(coefficients[2 * group]);
        let zero = F16::widen(coefficients[2 * group + 1]);
        for (offset, target) in values.iter_mut().enumerate() {
            let i = group * GROUP + offset;
            let code = (codes[i / per] >> ((i % per) as u32 * bits)) & mask;
            *target = (code as f32).mul_add(scale, zero);
        }
    }
}
