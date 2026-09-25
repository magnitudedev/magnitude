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

#[cfg(test)]
mod tests {
    use super::*;

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
