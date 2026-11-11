//! Activations quantized for integer dot products.
//!
//! A row of activations is quantized once per call into blocks of
//! [`Q8_BLOCK`] values: an `f32` scale `d`, the codes `round(x / d)` in
//! `[-127, 127]`, and the sums of each 16 consecutive codes (which the
//! k-quant min terms use). A final partial block is padded with zero codes.
//!
//! The scale is `max|x| / 127` rounded to nearest; where that is subnormal
//! (`max|x| < 127 · 2^-126`) it is rounded up onto the subnormal grid
//! instead, so that every `|x / d| ≤ 127` and `d` is zero only when the block
//! is. Each code is the stored scale's exact quotient rounded once, half away
//! from zero (CUDA's INT8 staging rounds half to even), and NaN codes 0.
//! Every step is scalar-exact, so every tier produces the same block.
//!
//! Quantizing activations changes a projection's arithmetic: an entry offers
//! it as an `arithmetic` tuning parameter, validated like any other.

use seismic_lang::forms::{CPU_Q8_ALIGNMENT, CPU_Q8_BYTES, CPU_Q8_VALUES};

/// Values per quantized activation block (the k-quant super block).
pub const Q8_BLOCK: usize = CPU_Q8_VALUES as usize;

/// Bytes of one quantized activation block, which declarations size their
/// scratch by as `Q8_BYTES`.
pub const Q8_BYTES: usize = CPU_Q8_BYTES as usize;

/// One block of quantized activations, the CPU INT8 form's
/// (`forms::CPU_Q8_BYTES`): its codes first and the block aligned to
/// `forms::CPU_Q8_ALIGNMENT`, so every packet's codes are whole vectors on
/// one cache line wherever the block lies.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct Q8Block {
    pub(crate) codes: [i8; Q8_BLOCK],
    /// Sums of codes `16 j .. 16 j + 16`.
    pub(crate) sums: [i16; Q8_BLOCK / 16],
    pub(crate) d: f32,
}

const _: () = assert!(
    std::mem::align_of::<Q8Block>() as u64 == CPU_Q8_ALIGNMENT
        && std::mem::size_of::<Q8Block>() as u64 == CPU_Q8_BYTES,
    "a Q8 block is the CPU INT8 form's"
);

const _: () = assert!(
    CPU_Q8_ALIGNMENT as usize == crate::isa::Tier::widest_vector(&crate::isa::Tier::EVERY),
    "the CPU INT8 form aligns its blocks to the widest vector of any tier"
);

impl Q8Block {
    pub const ZERO: Q8Block = Q8Block {
        codes: [0; Q8_BLOCK],
        sums: [0; Q8_BLOCK / 16],
        d: 0.0,
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
        let d = scale(largest(values));
        block.d = d;
        // The quotient of two F32 values is exact enough in F64 that rounding
        // it rounds the exact quotient; 0 / 0 and NaN code 0.
        for (code, value) in block.codes.iter_mut().zip(values) {
            *code = (f64::from(*value) / f64::from(d)).round() as i8;
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

/// The scale of a block whose largest magnitude is `maximum`:
/// `maximum / 127` rounded to nearest, or where that is subnormal, rounded
/// up onto the subnormal grid, so `maximum / d ≤ 127` and `d > 0` for every
/// nonzero `maximum`. A normal `d` rounded down leaves `maximum / d` within
/// one F32 ulp of 127, which still rounds to 127.
#[inline(always)]
pub(crate) fn scale(maximum: f32) -> f32 {
    let d = maximum / 127.0;
    // F64 holds `d · 127` exactly.
    if d < f32::MIN_POSITIVE && f64::from(d) * 127.0 < f64::from(maximum) {
        f32::from_bits(d.to_bits() + 1)
    } else {
        d
    }
}

/// The largest magnitude of `values`, ignoring NaN (as `fmaxnm` does).
#[inline(always)]
fn largest(values: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    if let Ok(values) = <&[f32; Q8_BLOCK]>::try_from(values) {
        return neon::largest(values);
    }
    values
        .iter()
        .fold(0.0f32, |maximum, value| maximum.max(value.abs()))
}

/// A whole block's largest magnitude by `fmaxnm`, which ignores NaN as
/// `f32::max` does.
#[cfg(target_arch = "aarch64")]
mod neon {
    use super::Q8_BLOCK;
    use std::arch::aarch64::*;

    #[inline(always)]
    pub(super) fn largest(values: &[f32; Q8_BLOCK]) -> f32 {
        // SAFETY: every load stays inside `values`.
        unsafe {
            let input = values.as_ptr();
            let mut maximum = vdupq_n_f32(0.0);
            for quarter in 0..Q8_BLOCK / 4 {
                maximum = vmaxnmq_f32(maximum, vabsq_f32(vld1q_f32(input.add(4 * quarter))));
            }
            vmaxnmvq_f32(maximum)
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// The vector block maximum matches the scalar fold bit for bit,
    /// including zeros and NaN.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn vector_maxima_match_the_scalar_fold() {
        let mut values = (0..Q8_BLOCK)
            .map(|i| ((i * 7919 % 509) as f32 - 254.0) * 0.0371)
            .collect::<Vec<_>>();
        values[9] = f32::NAN;
        values[17] = -0.0;
        for scale in [1.0f32, 1e-30, 1e-42, 0.0, 3.0e6] {
            let x = values.iter().map(|v| v * scale).collect::<Vec<_>>();
            let maximum = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            assert_eq!(neon::largest(x.as_slice().try_into().unwrap()).to_bits(), maximum.to_bits(), "scale {scale}");
        }
    }

    /// Asserts each of `blocks` is `x`'s exactly: the scale rule, every code
    /// the exact quotient by the stored scale rounded half away from zero
    /// (checked in F64, whose quotient of F32 values rounds the same), and
    /// the code sums.
    pub(crate) fn assert_exact(x: &[f32], blocks: &[Q8Block]) {
        for (b, (values, block)) in x.chunks(Q8_BLOCK).zip(blocks).enumerate() {
            let maximum = values.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let (d, top) = (f64::from(block.d), f64::from(maximum));
            assert_eq!(block.d == 0.0, maximum == 0.0, "block {b}: scale {:e} of maximum {maximum:e}", block.d);
            if block.d < f32::MIN_POSITIVE {
                // Rounded up onto the subnormal grid: the least step at or
                // above maximum / 127.
                assert!(d * 127.0 >= top, "block {b}");
                assert!(block.d == 0.0 || f64::from(f32::from_bits(block.d.to_bits() - 1)) * 127.0 < top, "block {b}");
            } else {
                assert_eq!(block.d.to_bits(), (maximum / 127.0).to_bits(), "block {b}");
            }
            for (i, code) in block.codes.iter().enumerate() {
                // NaN, and 0 / 0, code 0.
                let quotient = values.get(i).map_or(0.0, |x| (f64::from(*x) / d).round());
                let expected = if quotient.is_nan() { 0.0 } else { quotient };
                assert!(expected.abs() <= 127.0, "block {b} code {i}");
                assert_eq!(f64::from(*code), expected, "block {b} code {i}: {:?} / {d:e}", values.get(i));
            }
            for (j, sum) in block.sums.iter().enumerate() {
                assert_eq!(*sum, block.codes[16 * j..16 * j + 16].iter().map(|c| i16::from(*c)).sum::<i16>());
            }
        }
    }

    #[test]
    fn codes_reconstruct_within_half_a_step() {
        let x = (0..600)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.013)
            .collect::<Vec<_>>();
        let mut blocks = vec![Q8Block::ZERO; blocks(x.len())];
        quantize(&x, &mut blocks);
        assert_exact(&x, &blocks);
        for (i, value) in x.iter().enumerate() {
            let block = &blocks[i / Q8_BLOCK];
            let decoded = block.d * f32::from(block.codes[i % Q8_BLOCK]);
            assert!((decoded - value).abs() <= block.d * 0.5 + 1e-7, "{i}");
        }
        assert!(blocks[2].codes[600 - 512..].iter().all(|code| *code == 0));
    }

    /// A value whose exact quotient lies just under a half step codes down,
    /// where a product with the rounded reciprocal reaches the half and codes
    /// up (0.16743414 / (1.8490553 / 127) = 11.4999…, coded 12 that way).
    #[test]
    fn a_near_tie_codes_by_the_exact_quotient() {
        let mut x = vec![0.0f32; Q8_BLOCK];
        x[0] = 1.8490553;
        x[1] = 0.16743414;
        let mut blocks = [Q8Block::ZERO];
        quantize(&x, &mut blocks);
        assert_eq!(blocks[0].codes[1], 11);
        assert_exact(&x, &blocks);
    }

    /// Blocks whose scales are subnormal, from just under the normal range
    /// to maxima below 64 · 2^-149 (where `maximum / 127` rounds to zero),
    /// code every value by the stored scale within [-127, 127], and no
    /// nonzero block stores a zero scale. Each block also carries NaN, ±0
    /// and the least subnormal.
    #[test]
    fn subnormal_scales_code_by_the_stored_scale() {
        let mut x = Vec::new();
        for maximum in [127.0 * f32::MIN_POSITIVE * 0.999, 1.49e-36, 3.7e-37, 2.0f32.powi(-122), 1e-40, 64.0 * 2.0f32.powi(-149) * 0.9, 2.0f32.powi(-149)] {
            let mut block = (0..Q8_BLOCK)
                .map(|i| maximum * (((i * 7919 % 509) as f32 - 254.0) / 254.0))
                .collect::<Vec<_>>();
            block[0] = maximum;
            block[5] = f32::NAN;
            block[6] = -0.0;
            block[7] = -f32::from_bits(1);
            x.extend(block);
        }
        let mut blocks = vec![Q8Block::ZERO; blocks(x.len())];
        quantize(&x, &mut blocks);
        assert_exact(&x, &blocks);
        assert!(blocks.iter().all(|block| block.d > 0.0 && block.d < f32::MIN_POSITIVE));
    }
}
