//! The per-group activation rules: activations quantized for integer dot
//! products (below), and rows as NVFP4 operands ([`nvfp4`]).
//!
//! A row of activations is quantized once per call into blocks of
//! [`Q8_BLOCK`] values: eight [`Q8_GROUP`]-value (32-value) sub-blocks, each
//! with its own `f32` scale `d[g]` and codes `round(x / d[g])` in
//! `[-127, 127]`, and the sums of each 16 consecutive codes (which the
//! k-quant min terms use). A final partial block is padded with zero codes
//! and zero scales.
//!
//! Each sub-block's scale is `max|x| / 127` rounded to nearest; where that is
//! subnormal (`max|x| < 127 · 2^-126`) it is rounded up onto the subnormal grid
//! instead, so that every `|x / d| ≤ 127` and `d` is zero only when the
//! sub-block is. Each code is the stored scale's exact quotient rounded once,
//! half away from zero. A NaN or ±Inf value makes its sub-block's scale NaN
//! and every code in that sub-block 0, so the product is NaN. Every step is
//! scalar-exact, so every tier produces the same block, and CUDA's and
//! Metal's INT8 staging (over groups of 32, Metal's scale on F16's grid)
//! follow the same rule, which `tests/int8_rule` in `magnitude-kernels`
//! checks against this quantizer.
//!
//! Quantizing activations changes a projection's arithmetic: an entry offers
//! it as an `arithmetic` tuning parameter, validated like any other.

use crate::element::{f16_to_f32, f32_to_f16};
use seismic_lang::cpu::{CPU_Q8_ALIGNMENT, CPU_Q8_BYTES, CPU_Q8_GROUP, CPU_Q8_ROWS, CPU_Q8_VALUES};

pub mod nvfp4;

/// Values per quantized activation block (the k-quant super block).
pub const Q8_BLOCK: usize = CPU_Q8_VALUES as usize;

/// Values per quantized activation scale group (one packet).
pub const Q8_GROUP: usize = CPU_Q8_GROUP as usize;

/// Scale groups per quantized activation block.
pub const Q8_GROUPS: usize = Q8_BLOCK / Q8_GROUP;

/// Bytes of one quantized activation block, which declarations size their
/// scratch by as `Q8_BYTES`.
pub const Q8_BYTES: usize = CPU_Q8_BYTES as usize;

/// One block of quantized activations, the CPU INT8 form's
/// (`cpu::CPU_Q8_BYTES`): its codes first and the block aligned to
/// `cpu::CPU_Q8_ALIGNMENT`, so every packet's codes are whole vectors on
/// one cache line wherever the block lies.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct Q8Block {
    pub(crate) codes: [i8; Q8_BLOCK],
    /// Sums of codes `16 j .. 16 j + 16`.
    pub(crate) sums: [i16; Q8_BLOCK / 16],
    /// Scale of each 32-value sub-block (`32 g .. 32 g + 32`).
    pub(crate) d: [f32; Q8_GROUPS],
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
        d: [0.0; Q8_GROUPS],
    };

    /// The block's codes.
    pub fn codes(&self) -> &[i8; Q8_BLOCK] {
        &self.codes
    }

    /// The block's per-sub-block scales (one per [`Q8_GROUP`] values).
    pub fn scales(&self) -> &[f32; Q8_GROUPS] {
        &self.d
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
    /// `d[r]`: row `r`'s `Q8Block::d`.
    pub(crate) d: [[f32; Q8_GROUPS]; R],
}

const _: () = assert!(
    std::mem::size_of::<Q8Tile>() <= TILE_ROWS * Q8_BYTES && std::mem::align_of::<Q8Tile>() == CPU_Q8_ALIGNMENT as usize,
    "a tile fits its rows' blocks' bytes (`cpu::CPU_Q8_ROWS`)"
);

impl Q8Tile {
    pub const ZERO: Q8Tile = Q8Tile {
        codes: [[[0; 8]; TILE_ROWS]; Q8_BLOCK / 8],
        sums: [[0; Q8_BLOCK / 16]; TILE_ROWS],
        d: [[0.0; Q8_GROUPS]; TILE_ROWS],
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
    if blocks == 0 {
        return;
    }
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
        for (g, d) in block.d.iter_mut().enumerate() {
            let start = g * Q8_GROUP;
            let codes = &mut block.codes[start..start + Q8_GROUP];
            if start < values.len() {
                let end = (start + Q8_GROUP).min(values.len());
                let group = &values[start..end];
                *d = quantize_group::<F32Grid>(group, &mut codes[..group.len()]);
                for code in &mut codes[group.len()..] {
                    *code = 0;
                }
            } else {
                *d = 0.0;
                codes.fill(0);
            }
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
/// and a NaN scale code 0.
#[inline(always)]
pub fn code(value: f32, scale: f32) -> i8 {
    (f64::from(value) / f64::from(scale)).round() as i8
}

/// Quantizes a group on grid `G`: its scale, and its codes into `codes`
/// (`values.len()` of them). A NaN or ±Inf value makes the group's scale
/// NaN and every code 0, so the product is NaN. The codes are taken under
/// the scale of the group's [`largest`] magnitude as it is, NaN or +Inf for
/// such a group, under either of which every code is 0 (a finite value over
/// +Inf rounds to 0, and ±Inf over +Inf is NaN, which codes 0); only the
/// returned scale is then made NaN, off the codes' path.
#[inline(always)]
pub fn quantize_group<G: ScaleGrid>(values: &[f32], codes: &mut [i8]) -> f32 {
    let maximum = largest(values);
    let d = scale_on::<G>(maximum);
    for (code_out, value) in codes.iter_mut().zip(values) {
        *code_out = code(*value, d);
    }
    if maximum.is_finite() { d } else { f32::NAN }
}

/// The largest magnitude of `values`: NaN where any value is NaN, else
/// +Inf where any is ±Inf. The maximum is over the magnitudes' bits, whose
/// integer order is the magnitudes' with every NaN above +Inf, so a NaN
/// wins for the instructions of a float maximum (an AND and an integer
/// maximum in place of an absolute value and a float maximum).
#[inline(always)]
pub fn largest(values: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    let bits = neon::magnitudes(values);
    #[cfg(not(target_arch = "aarch64"))]
    let bits = magnitudes(values);
    f32::from_bits(bits)
}

/// The bits of an F32's magnitude.
const MAGNITUDE: u32 = 0x7fff_ffff;

/// The integer maximum of the magnitudes' bits of `values` (0 for none).
#[inline(always)]
fn magnitudes(values: &[f32]) -> u32 {
    values.iter().fold(0, |maximum, value| maximum.max(value.to_bits() & MAGNITUDE))
}

/// [`magnitudes`] by vectors of four, the values past the last whole
/// vector seeding the maximum: one path for every length, so the maximum
/// stays in the vector register the scale's division reads.
#[cfg(target_arch = "aarch64")]
mod neon {
    use super::MAGNITUDE;
    use std::arch::aarch64::*;

    #[inline(always)]
    pub(super) fn magnitudes(values: &[f32]) -> u32 {
        let vectors = values.chunks_exact(4);
        // SAFETY: each load reads one whole chunk of four.
        unsafe {
            let magnitude = vdupq_n_u32(MAGNITUDE);
            let mut maximum = vdupq_n_u32(super::magnitudes(vectors.remainder()));
            for vector in vectors {
                maximum = vmaxq_u32(maximum, vandq_u32(vld1q_u32(vector.as_ptr().cast()), magnitude));
            }
            vmaxvq_u32(maximum)
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// The vector maximum matches the scalar fold bit for bit at a whole
    /// block, a group of 32 and a length past whole vectors, including
    /// zeros; it is NaN where a NaN is among the values, and +Inf where
    /// ±Inf is without one.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn vector_maxima_match_the_scalar_fold() {
        let mut values = (0..Q8_BLOCK)
            .map(|i| ((i * 7919 % 509) as f32 - 254.0) * 0.0371)
            .collect::<Vec<_>>();
        values[17] = -0.0;
        for len in [Q8_BLOCK, 32, 39] {
            for scale in [1.0f32, 1e-30, 1e-42, 0.0, 3.0e6] {
                let x = values[..len].iter().map(|v| v * scale).collect::<Vec<_>>();
                let maximum = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                assert_eq!(magnitudes(&x), maximum.to_bits(), "{len} values, scale {scale}");
                assert_eq!(neon::magnitudes(&x), maximum.to_bits(), "{len} values, scale {scale}");
            }
            for (non_finite, nan) in [(f32::NAN, true), (-f32::NAN, true), (f32::INFINITY, false), (f32::NEG_INFINITY, false)] {
                for at in [0, 9, len - 1] {
                    let mut x = values[..len].to_vec();
                    x[at] = non_finite;
                    for bits in [magnitudes(&x), neon::magnitudes(&x)] {
                        let maximum = f32::from_bits(bits);
                        assert!(if nan { maximum.is_nan() } else { maximum == f32::INFINITY }, "{non_finite} at {at} of {len}");
                    }
                    x[(at + 1) % len] = f32::NAN;
                    assert!(f32::from_bits(neon::magnitudes(&x)).is_nan(), "NaN with {non_finite} at {at} of {len}");
                }
            }
        }
    }

    /// Asserts each of `blocks` is `x`'s exactly: the scale rule per 32-value
    /// sub-block, every code the exact quotient by the stored sub-block scale
    /// rounded half away from zero (checked in F64, whose quotient of F32
    /// values rounds the same), and the code sums. A sub-block with a NaN or
    /// ±Inf has a NaN scale and zero codes.
    pub(crate) fn assert_exact(x: &[f32], blocks: &[Q8Block]) {
        for (b, (values, block)) in x.chunks(Q8_BLOCK).zip(blocks).enumerate() {
            for (g, &scale) in block.d.iter().enumerate() {
                let start = g * Q8_GROUP;
                let group = if start < values.len() {
                    &values[start..(start + Q8_GROUP).min(values.len())]
                } else {
                    &[]
                };
                let codes = &block.codes[start..start + Q8_GROUP];
                if group.iter().any(|value| !value.is_finite()) {
                    assert!(scale.is_nan(), "block {b} group {g}: scale {:e} of a non-finite group", scale);
                    assert!(
                        codes.iter().all(|code| *code == 0)
                            && block.sums[2 * g] == 0
                            && block.sums[2 * g + 1] == 0,
                        "block {b} group {g}"
                    );
                    continue;
                }
                let maximum = group.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let (d, top) = (f64::from(scale), f64::from(maximum));
                assert_eq!(scale == 0.0, maximum == 0.0, "block {b} group {g}: scale {:e} of maximum {maximum:e}", scale);
                if scale < f32::MIN_POSITIVE {
                    // Rounded up onto the subnormal grid: the least step at or
                    // above maximum / 127.
                    assert!(d * 127.0 >= top, "block {b} group {g}");
                    assert!(scale == 0.0 || f64::from(f32::from_bits(scale.to_bits() - 1)) * 127.0 < top, "block {b} group {g}");
                } else {
                    assert_eq!(scale.to_bits(), (maximum / 127.0).to_bits(), "block {b} group {g}");
                }
                for (i, code) in codes.iter().enumerate() {
                    // 0 / 0 codes 0.
                    let quotient = group.get(i).map_or(0.0, |x| (f64::from(*x) / d).round());
                    let expected = if quotient.is_nan() { 0.0 } else { quotient };
                    assert!(expected.abs() <= 127.0, "block {b} group {g} code {i}");
                    assert_eq!(f64::from(*code), expected, "block {b} group {g} code {i}: {:?} / {d:e}", group.get(i));
                }
            }
            for (j, sum) in block.sums.iter().enumerate() {
                assert_eq!(*sum, block.codes[16 * j..16 * j + 16].iter().map(|c| i16::from(*c)).sum::<i16>());
            }
        }
    }

    /// A NaN or ±Inf value makes its sub-block's scale NaN and every code 0,
    /// so the product is NaN; the row's other sub-blocks are unaffected.
    #[test]
    fn non_finite_values_make_their_block_nan() {
        for non_finite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut x = (0..3 * Q8_BLOCK).map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.013).collect::<Vec<_>>();
            x[Q8_BLOCK + 70] = non_finite;
            let mut blocks = vec![Q8Block::ZERO; blocks(x.len())];
            quantize(&x, &mut blocks);
            assert!(
                blocks[1].d[70 / Q8_GROUP].is_nan()
                    && blocks[0].d.iter().all(|d| d.is_finite())
                    && blocks[2].d.iter().all(|d| d.is_finite()),
                "{non_finite}"
            );
            assert_exact(&x, &blocks);
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
            let d = block.d[(i % Q8_BLOCK) / Q8_GROUP];
            let decoded = d * f32::from(block.codes[i % Q8_BLOCK]);
            assert!((decoded - value).abs() <= d * 0.5 + 1e-7, "{i}");
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
    /// nonzero sub-block stores a zero scale. Each block also carries ±0 and
    /// the least subnormal.
    #[test]
    fn subnormal_scales_code_by_the_stored_scale() {
        let mut x = Vec::new();
        for maximum in [127.0 * f32::MIN_POSITIVE * 0.999, 1.49e-36, 3.7e-37, 2.0f32.powi(-122), 1e-40, 64.0 * 2.0f32.powi(-149) * 0.9, 2.0f32.powi(-149)] {
            let mut block = (0..Q8_BLOCK)
                .map(|i| maximum * (((i * 7919 % 509) as f32 - 254.0) / 254.0))
                .collect::<Vec<_>>();
            block[0] = maximum;
            block[6] = -0.0;
            block[7] = -f32::from_bits(1);
            x.extend(block);
        }
        let mut blocks = vec![Q8Block::ZERO; blocks(x.len())];
        quantize(&x, &mut blocks);
        assert_exact(&x, &blocks);
        assert!(blocks.iter().all(|block| block.d.iter().all(|&d| d > 0.0 && d < f32::MIN_POSITIVE)));
    }

    #[test]
    fn interleave_zero_blocks_is_noop() {
        interleave(&[], 0, 0, &mut []);
        interleave(&[], 4, 0, &mut []);
    }
}
