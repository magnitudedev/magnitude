use seismic::cpu::{Dense, F16};

/// The portable `affine_encode` of one vector `x` at `bits` code bits: per
/// group of GROUP values, zero = f16(min), scale = f16((max - min) / L) and
/// code = min(L, u32(fma(x - zero, 1 / scale, 0.5))) (0 when scale is 0),
/// packed 32 / bits codes per word; (scale, zero) F16 bits per group.
#[inline(always)]
pub fn affine_encode(x: &[f32], bits: u32, codes: &mut [u32], coefficients: &mut [u16]) {
    let levels = (1u32 << bits) - 1;
    let per = (32 / bits) as usize;
    let group_width = 2 * x.len() / coefficients.len();
    codes.fill(0);
    for (group, values) in x.chunks_exact(group_width).enumerate() {
        let (low, high) = values
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(low, high), value| (low.min(*value), high.max(*value)));
        let zero = F16::narrow(low);
        let scale = F16::narrow((high - low) / levels as f32);
        let step = F16::widen(scale);
        let inverse = if step > 0.0 { 1.0 / step } else { 0.0 };
        let base = F16::widen(zero);
        for (offset, value) in values.iter().enumerate() {
            let i = group * group_width + offset;
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
/// `fma(code, scale, zero)` with the value's group pair. A group's codes are
/// whole words (32 / bits codes each), walked word by word, so no value
/// divides; the vector tiers decode in registers instead
/// (`attention::Rows`).
#[inline(always)]
pub fn affine_decode(codes: &[u32], coefficients: &[u16], bits: u32, out: &mut [f32]) {
    let mask = (1u32 << bits) - 1;
    let per = (32 / bits) as usize;
    let group_width = 2 * out.len() / coefficients.len();
    assert_eq!(group_width % per, 0, "a group of {group_width} values is not whole {bits}-bit words");
    let words = codes.chunks_exact(group_width / per);
    for ((values, words), pair) in out.chunks_exact_mut(group_width).zip(words).zip(coefficients.chunks_exact(2)) {
        let (scale, zero) = (F16::widen(pair[0]), F16::widen(pair[1]));
        for (targets, word) in values.chunks_exact_mut(per).zip(words) {
            for (index, target) in targets.iter_mut().enumerate() {
                let code = (word >> (index as u32 * bits)) & mask;
                *target = (code as f32).mul_add(scale, zero);
            }
        }
    }
}
