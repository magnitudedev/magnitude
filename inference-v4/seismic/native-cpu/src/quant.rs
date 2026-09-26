//! Activations quantized for integer dot products.
//!
//! A row of activations is quantized once per call into blocks of
//! [`Q8_BLOCK`] values: an `f32` scale `d = max|x| / 127`, the codes
//! `round(x / d)` in `[-127, 127]`, and the sums of each 16 consecutive codes
//! (which the k-quant min terms use). A final partial block is padded with
//! zero codes. Every step is scalar-exact or elementwise, so every tier
//! produces the same codes.
//!
//! Quantizing activations changes a projection's arithmetic: an entry offers
//! it as an `arithmetic` tuning parameter, validated like any other.

/// Values per quantized activation block (the k-quant super block).
pub const Q8_BLOCK: usize = 256;

/// One block of quantized activations.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Q8Block {
    pub d: f32,
    pub codes: [i8; Q8_BLOCK],
    /// Sums of codes `16 j .. 16 j + 16`.
    pub sums: [i16; Q8_BLOCK / 16],
}

impl Q8Block {
    pub const ZERO: Q8Block = Q8Block {
        d: 0.0,
        codes: [0; Q8_BLOCK],
        sums: [0; Q8_BLOCK / 16],
    };
}

/// Blocks needed for a row of `k` values.
pub const fn blocks(k: usize) -> usize {
    k.div_ceil(Q8_BLOCK)
}

/// Quantizes `x` into `out` (`blocks(x.len())` blocks).
#[inline(always)]
pub fn quantize(x: &[f32], out: &mut [Q8Block]) {
    let out = &mut out[..blocks(x.len())];
    for (values, block) in x.chunks(Q8_BLOCK).zip(out.iter_mut()) {
        #[cfg(target_arch = "aarch64")]
        if let Ok(values) = <&[f32; Q8_BLOCK]>::try_from(values) {
            neon::quantize_block(values, block);
            continue;
        }
        let maximum = values
            .iter()
            .fold(0.0f32, |maximum, value| maximum.max(value.abs()));
        let d = maximum / 127.0;
        let inverse = if d == 0.0 { 0.0 } else { 1.0 / d };
        block.d = d;
        for (code, value) in block.codes.iter_mut().zip(values) {
            *code = (value * inverse).round() as i8;
        }
        for code in &mut block.codes[values.len()..] {
            *code = 0;
        }
        for (j, sum) in block.sums.iter_mut().enumerate() {
            *sum = block.codes[16 * j..16 * j + 16]
                .iter()
                .map(|code| i16::from(*code))
                .sum();
        }
    }
}

/// A whole block quantized with the scalar path's exact operations:
/// `f32::max` ignores NaN as `fmaxnm` does, `round` rounds half away from
/// zero as `frinta` does, and `as i8` saturates (NaN to 0) as the
/// conversion and saturating narrows do.
#[cfg(target_arch = "aarch64")]
mod neon {
    use super::{Q8Block, Q8_BLOCK};
    use std::arch::aarch64::*;

    #[inline(always)]
    pub(super) fn quantize_block(values: &[f32; Q8_BLOCK], block: &mut Q8Block) {
        // SAFETY: every load and store stays inside `values` and `block`.
        unsafe {
            let input = values.as_ptr();
            let mut maximum = vdupq_n_f32(0.0);
            for quarter in 0..Q8_BLOCK / 4 {
                maximum = vmaxnmq_f32(maximum, vabsq_f32(vld1q_f32(input.add(4 * quarter))));
            }
            let d = vmaxnmvq_f32(maximum) / 127.0;
            let inverse = if d == 0.0 { 0.0 } else { 1.0 / d };
            block.d = d;
            let scale = vdupq_n_f32(inverse);
            for sixteen in 0..Q8_BLOCK / 16 {
                let at = input.add(16 * sixteen);
                let codes: [int32x4_t; 4] = std::array::from_fn(|quarter| {
                    vcvtq_s32_f32(vrndaq_f32(vmulq_f32(vld1q_f32(at.add(4 * quarter)), scale)))
                });
                let low = vcombine_s16(vqmovn_s32(codes[0]), vqmovn_s32(codes[1]));
                let high = vcombine_s16(vqmovn_s32(codes[2]), vqmovn_s32(codes[3]));
                let bytes = vcombine_s8(vqmovn_s16(low), vqmovn_s16(high));
                vst1q_s8(block.codes.as_mut_ptr().add(16 * sixteen), bytes);
                block.sums[sixteen] = vaddlvq_s8(bytes);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The vector block quantization matches the scalar path bit for bit,
    /// including ties, saturation, zeros and NaN.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn vector_blocks_match_the_scalar_quantization() {
        let mut values = (0..Q8_BLOCK)
            .map(|i| ((i * 7919 % 509) as f32 - 254.0) * 0.0371)
            .collect::<Vec<_>>();
        values[3] = 254.0 * 0.0371 / 127.0 * 2.5;
        values[9] = f32::NAN;
        values[17] = -0.0;
        for scale in [1.0f32, 1e-30, 0.0, 3.0e6] {
            let x = values.iter().map(|v| v * scale).collect::<Vec<_>>();
            let mut vector = [Q8Block::ZERO];
            neon::quantize_block(x.as_slice().try_into().unwrap(), &mut vector[0]);
            // The scalar operations over the whole block.
            let mut whole = Q8Block::ZERO;
            let maximum = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let d = maximum / 127.0;
            let inverse = if d == 0.0 { 0.0 } else { 1.0 / d };
            whole.d = d;
            for (code, value) in whole.codes.iter_mut().zip(&x) {
                *code = (value * inverse).round() as i8;
            }
            for (j, sum) in whole.sums.iter_mut().enumerate() {
                *sum = whole.codes[16 * j..16 * j + 16]
                    .iter()
                    .map(|c| i16::from(*c))
                    .sum();
            }
            assert_eq!(vector[0].d.to_bits(), whole.d.to_bits(), "scale {scale}");
            assert_eq!(vector[0].codes, whole.codes, "scale {scale}");
            assert_eq!(vector[0].sums, whole.sums, "scale {scale}");
        }
    }

    #[test]
    fn codes_reconstruct_within_half_a_step() {
        assert_eq!(std::mem::size_of::<Q8Block>(), 292);
        let x = (0..600)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.013)
            .collect::<Vec<_>>();
        let mut blocks = vec![Q8Block::ZERO; blocks(x.len())];
        quantize(&x, &mut blocks);
        for (i, value) in x.iter().enumerate() {
            let block = &blocks[i / Q8_BLOCK];
            let decoded = block.d * f32::from(block.codes[i % Q8_BLOCK]);
            assert!((decoded - value).abs() <= block.d * 0.5 + 1e-7, "{i}");
        }
        assert!(blocks[2].codes[600 - 512..].iter().all(|code| *code == 0));
        let sum: i16 = blocks[0].codes[..16]
            .iter()
            .map(|code| i16::from(*code))
            .sum();
        assert_eq!(blocks[0].sums[0], sum);
    }
}
