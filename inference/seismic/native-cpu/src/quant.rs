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
//! from zero, and NaN codes 0. Every step is scalar-exact, so every tier
//! produces the same block, and CUDA's and Metal's INT8 staging (over groups
//! of 32, Metal's scale on F16's grid) follow the same rule, which
//! `tests/int8_rule` in `magnitude-kernels` checks against this quantizer.
//!
//! Quantizing activations changes a projection's arithmetic: an entry offers
//! it as an `arithmetic` tuning parameter, validated like any other.

use crate::element::{f16_to_f32, f32_to_f16};
use seismic_lang::forms::{CPU_Q8_ALIGNMENT, CPU_Q8_BYTES, CPU_Q8_ROWS, CPU_Q8_VALUES};

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

    /// The block's codes.
    pub fn codes(&self) -> &[i8; Q8_BLOCK] {
        &self.codes
    }

    /// The block's scale.
    pub fn scale(&self) -> f32 {
        self.d
    }

    /// Blocks as the one-row tiles they are: a block's codes, sums and
    /// scale are a one-row [`Q8Tile`]'s, byte for byte.
    pub fn as_tiles(blocks: &[Q8Block]) -> &[Q8Tile<1>] {
        const {
            assert!(
                std::mem::size_of::<Q8Block>() == std::mem::size_of::<Q8Tile<1>>()
                    && std::mem::align_of::<Q8Block>() == std::mem::align_of::<Q8Tile<1>>()
                    && std::mem::offset_of!(Q8Block, sums) == std::mem::offset_of!(Q8Tile<1>, sums)
                    && std::mem::offset_of!(Q8Block, d) == std::mem::offset_of!(Q8Tile<1>, d),
                "a block is a one-row tile"
            )
        };
        // SAFETY: the two are `repr(C)` of the same fields at the same
        // offsets (asserted), every bit pattern valid for both.
        unsafe { std::slice::from_raw_parts(blocks.as_ptr().cast(), blocks.len()) }
    }
}

/// Activation rows of one [`Q8Tile`]: the tile kernel's strip.
pub const TILE_ROWS: usize = CPU_Q8_ROWS as usize;

/// One block of `R` activation rows ([`TILE_ROWS`], or one: a
/// [`Q8Block`], `Q8Block::as_tiles`), interleaved for the tile kernel
/// (`weights::tile`): each row's codes, sums and scale exactly as its
/// [`Q8Block`], placed so that every ISA loads a strip's operands whole
/// (AVX2 broadcasts a row's four codes, AVX-512 its eight, NEON `ld2` yields
/// four rows' four codes a vector, SMMLA a row pair's eight; x86 broadcasts
/// a row's adjacent sums). Rows past an activation matrix's end are zero.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct Q8Tile<const R: usize = TILE_ROWS> {
    /// `codes[s][r]`: row `r`'s codes `8 s .. 8 s + 8`.
    pub(crate) codes: [[[i8; 8]; R]; Q8_BLOCK / 8],
    /// `sums[r]`: row `r`'s `Q8Block::sums`.
    pub(crate) sums: [[i16; Q8_BLOCK / 16]; R],
    pub(crate) d: [f32; R],
}

const _: () = assert!(
    std::mem::size_of::<Q8Tile>() <= TILE_ROWS * Q8_BYTES && std::mem::align_of::<Q8Tile>() == CPU_Q8_ALIGNMENT as usize,
    "a tile fits its rows' blocks' bytes (`forms::CPU_Q8_ROWS`)"
);

impl Q8Tile {
    pub const ZERO: Q8Tile = Q8Tile {
        codes: [[[0; 8]; TILE_ROWS]; Q8_BLOCK / 8],
        sums: [[0; Q8_BLOCK / 16]; TILE_ROWS],
        d: [0.0; TILE_ROWS],
    };

    /// Places `block` as row `row` of the tile at `tile`, writing that row's
    /// bytes alone: the tile's other rows may be written at the same time.
    ///
    /// # Safety
    /// `tile` is valid for writes, and nothing else writes row `row` of it.
    #[inline(always)]
    pub unsafe fn place(tile: *mut Q8Tile, row: usize, block: &Q8Block) {
        unsafe {
            for (step, codes) in block.codes.chunks_exact(8).enumerate() {
                std::ptr::addr_of_mut!((*tile).codes[step][row]).cast::<i8>().copy_from_nonoverlapping(codes.as_ptr(), 8);
            }
            std::ptr::addr_of_mut!((*tile).sums[row]).write(block.sums);
            std::ptr::addr_of_mut!((*tile).d[row]).write(block.d);
        }
    }
}

/// Tiles of `m` activation rows of `blocks` blocks each: `[row group][block]`.
pub const fn tiles(m: usize, blocks: usize) -> usize {
    m.div_ceil(TILE_ROWS) * blocks
}

/// The `m` quantized rows `rows` (`blocks` blocks each, row after row) as
/// tiles (`[row group][block]`, [`tiles`] of them), rows past `m` zero.
pub fn interleave(rows: &[Q8Block], m: usize, blocks: usize, out: &mut [Q8Tile]) {
    assert!(rows.len() == m * blocks && out.len() == tiles(m, blocks), "{m} rows of {blocks} blocks");
    out.fill(Q8Tile::ZERO);
    for (row, blocks_of) in rows.chunks_exact(blocks).enumerate() {
        let group = &mut out[row / TILE_ROWS * blocks..][..blocks];
        for (tile, block) in group.iter_mut().zip(blocks_of) {
            // SAFETY: `tile` is this call's alone.
            unsafe { Q8Tile::place(tile, row % TILE_ROWS, block) };
        }
    }
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
        block.d = quantize_group::<F32Grid>(values, &mut block.codes[..values.len()]);
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

/// The format a block's scale is stored in, whose grid the rule rounds it
/// to: F32 for the CPU, F16 for Metal's INT8 form.
pub trait ScaleGrid {
    /// The least normal value of the grid.
    const LEAST_NORMAL: f32;
    /// `value` rounded to nearest on the grid.
    fn nearest(value: f32) -> f32;
    /// The grid value above `scale`, a subnormal or zero one.
    fn above(scale: f32) -> f32;
}

/// F32 scales.
pub struct F32Grid;

impl ScaleGrid for F32Grid {
    const LEAST_NORMAL: f32 = f32::MIN_POSITIVE;
    #[inline(always)]
    fn nearest(value: f32) -> f32 {
        value
    }
    #[inline(always)]
    fn above(scale: f32) -> f32 {
        f32::from_bits(scale.to_bits() + 1)
    }
}

/// F16 scales.
pub struct F16Grid;

impl ScaleGrid for F16Grid {
    const LEAST_NORMAL: f32 = f32::from_bits(0x3880_0000);
    #[inline(always)]
    fn nearest(value: f32) -> f32 {
        f16_to_f32(f32_to_f16(value))
    }
    #[inline(always)]
    fn above(scale: f32) -> f32 {
        f16_to_f32(f32_to_f16(scale) + 1)
    }
}

/// The scale of a group whose largest magnitude is `maximum`, on grid `G`:
/// `maximum / 127` rounded to nearest, or where that is subnormal, rounded
/// up onto the subnormal grid, so `maximum / d ≤ 127` and `d > 0` for every
/// nonzero `maximum`. A normal `d` rounded down leaves `maximum / d` within
/// one ulp of 127, which still rounds to 127.
#[inline(always)]
pub fn scale_on<G: ScaleGrid>(maximum: f32) -> f32 {
    let d = G::nearest(maximum / 127.0);
    // F64 holds `d · 127` exactly.
    if d < G::LEAST_NORMAL && f64::from(d) * 127.0 < f64::from(maximum) {
        G::above(d)
    } else {
        d
    }
}

/// The code of `value` under `scale`: the exact quotient (an F64 quotient of
/// two F32 values rounds like it) rounded once, half away from zero; 0 / 0
/// and NaN code 0.
#[inline(always)]
pub fn code(value: f32, scale: f32) -> i8 {
    (f64::from(value) / f64::from(scale)).round() as i8
}

/// Quantizes a group on grid `G`: its scale, and its codes into `codes`
/// (`values.len()` of them).
#[inline(always)]
pub fn quantize_group<G: ScaleGrid>(values: &[f32], codes: &mut [i8]) -> f32 {
    let d = scale_on::<G>(largest(values));
    for (code_out, value) in codes.iter_mut().zip(values) {
        *code_out = code(*value, d);
    }
    d
}

/// The largest magnitude of `values`, ignoring NaN (as `fmaxnm` does).
#[inline(always)]
pub fn largest(values: &[f32]) -> f32 {
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
