use seismic::cpu::{Dense, F16};

/// One group's quantization at `levels` code levels: zero = f16(min), scale
/// = f16((max - min) / L), and code = min(L, u32(fma(x - zero, 1 / scale,
/// 0.5))) (0 when scale is 0).
struct GroupCodec {
    /// The (scale, zero) F16 bits.
    scale: u16,
    zero: u16,
    base: f32,
    inverse: f32,
    levels: u32,
}

impl GroupCodec {
    #[inline(always)]
    fn of(values: &[f32], levels: u32) -> Self {
        let (low, high) = values
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(low, high), value| (low.min(*value), high.max(*value)));
        let zero = F16::narrow(low);
        let scale = F16::narrow((high - low) / levels as f32);
        let step = F16::widen(scale);
        let inverse = if step > 0.0 { 1.0 / step } else { 0.0 };
        Self { scale, zero, base: F16::widen(zero), inverse, levels }
    }

    #[inline(always)]
    fn code(&self, value: f32) -> u32 {
        // `as` saturates: a negative or NaN code is 0, as the portable `u32`
        // conversion.
        ((value - self.base).mul_add(self.inverse, 0.5) as u32).min(self.levels)
    }
}

/// The portable `affine_encode` of one vector `x` at `bits` code bits: per
/// group of GROUP values its codes (`GroupCodec`), packed 32 / bits codes per
/// word, and its (scale, zero) F16 bits.
#[inline(always)]
pub fn affine_encode(x: &[f32], bits: u32, codes: &mut [u32], coefficients: &mut [u16]) {
    let levels = (1u32 << bits) - 1;
    let per = (32 / bits) as usize;
    let group_width = 2 * x.len() / coefficients.len();
    codes.fill(0);
    for (group, values) in x.chunks_exact(group_width).enumerate() {
        let codec = GroupCodec::of(values, levels);
        for (offset, value) in values.iter().enumerate() {
            let i = group * group_width + offset;
            codes[i / per] |= codec.code(*value) << ((i % per) as u32 * bits);
        }
        coefficients[2 * group] = codec.scale;
        coefficients[2 * group + 1] = codec.zero;
    }
}

/// `x` replaced by what its `affine_encode` encoding (at `bits` code bits,
/// `coefficients` F16 coefficients) decodes to (`affine_decode`): the vector
/// a history row holding that encoding reads. The encode is deterministic,
/// so this is what a row appended from the same `x` reads, with no encoding
/// held in between.
#[inline(always)]
pub fn affine_round_trip(x: &mut [f32], bits: u32, coefficients: usize) {
    let levels = (1u32 << bits) - 1;
    let group_width = 2 * x.len() / coefficients;
    for values in x.chunks_exact_mut(group_width) {
        let codec = GroupCodec::of(values, levels);
        let (scale, zero) = (F16::widen(codec.scale), F16::widen(codec.zero));
        for value in values.iter_mut() {
            *value = (codec.code(*value) as f32).mul_add(scale, zero);
        }
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
