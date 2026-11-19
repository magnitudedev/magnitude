// The affine K8/V4 history codec on CPU (`affine_k8_uniform_v4`, the
// reference `affine_encode` / `affine_decode` in `attention.seismic`): the
// encode a row's append and every read of a fresh row take, the portable
// decode, and the rows the vector tiers absorb codes from (`AffineRows`).
// Its group and code widths are attention.seismic's AFFINE_* constants.

use super::attention::{lines, Rows, F8};
use seismic::cpu::{slab::{SlabTensor, SlabWalk}, Dense, Element, Store, F16, U32};

/// Values per (scale, zero) pair (attention.seismic's AFFINE_GROUP).
pub const GROUP: usize = crate::constants::AFFINE_GROUP as usize;
/// Code bits of a key (AFFINE_KEY_BITS) and of a value (AFFINE_VALUE_BITS).
pub const KEY_BITS: u32 = crate::constants::AFFINE_KEY_BITS as u32;
pub const VALUE_BITS: u32 = crate::constants::AFFINE_VALUE_BITS as u32;

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
fn affine_encode(x: &[f32], bits: u32, codes: &mut [u32], mut coefficients: Store<'_, F16>) {
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
        coefficients.set(2 * group, F16::widen(codec.scale));
        coefficients.set(2 * group + 1, F16::widen(codec.zero));
    }
}

/// `x` replaced by what its `affine_encode` encoding (at `bits` code bits,
/// in groups of [`GROUP`]) decodes to (`affine_decode`): the vector a
/// history row holding that encoding reads. The encode is deterministic, so
/// this is what a row appended from the same `x` reads, with no encoding
/// held in between.
#[inline(always)]
pub fn affine_round_trip(x: &mut [f32], bits: u32) {
    let levels = (1u32 << bits) - 1;
    for values in x.chunks_exact_mut(GROUP) {
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
fn affine_decode(codes: &[u32], coefficients: &[u16], bits: u32, out: &mut [f32]) {
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

/// Affine history rows of `BITS`-bit codes (8 or 4) in GROUP-column groups
/// of kv head `head` in slab-stored history: codes in `codes`, (scale,
/// zero) F16 pairs in `coefficients`, both of one slab size.
pub struct AffineRows<'a, const BITS: u32, C: Element<Storage = u16>> {
    codes: SlabTensor<'a, U32, 3>,
    coefficients: SlabTensor<'a, C, 3>,
    head: usize,
}

impl<const BITS: u32, C: Element<Storage = u16>> Rows for AffineRows<'_, BITS, C> {
    #[inline(always)]
    fn coefficients(&self, w: usize) -> usize {
        2 * w / GROUP
    }

    #[inline(always)]
    fn load(&self, index: usize, out: &mut [f32]) {
        let (codes, coefficients) = (self.codes.row([index, self.head, 0]), self.coefficients.row([index, self.head, 0]));
        affine_decode(codes, coefficients, BITS, out);
    }

    type Walk = SlabWalk;

    #[inline(always)]
    fn walk(&self, first: usize) -> SlabWalk {
        self.codes.walk(first)
    }

    #[inline(always)]
    unsafe fn next(&self, walk: &mut SlabWalk, w: usize, coefficients: &mut [f32]) -> *const u8 {
        const { assert!(BITS == 8 || BITS == 4) };
        let index = [walk.index(), self.head, 0];
        let (codes, pairs) = (self.codes.row_at(walk, index), self.coefficients.row_at(walk, index));
        assert!(codes.len() * 32 >= w * BITS as usize && pairs.len() >= coefficients.len());
        // SAFETY: the caller's tier.
        unsafe { widen_f16(&pairs[..coefficients.len()], coefficients) };
        walk.step();
        codes.as_ptr().cast()
    }

    #[inline(always)]
    fn prefetch(&self, walk: &mut SlabWalk, w: usize) {
        let index = [walk.index(), self.head, 0];
        lines(&self.codes.row_at(walk, index)[..w * BITS as usize / 32]);
        lines(&self.coefficients.row_at(walk, index)[..2 * w / GROUP]);
        walk.step();
    }

    const GROUP_CHUNKS: usize = GROUP / 8;

    /// The group's (scale, zero), each in every lane.
    type Scale = (F8, F8);

    #[inline(always)]
    unsafe fn scale(coefficients: &[f32], group: usize) -> (F8, F8) {
        unsafe { (F8::splat(coefficients[2 * group]), F8::splat(coefficients[2 * group + 1])) }
    }

    #[inline(always)]
    unsafe fn chunk(row: *const u8, (scale, zero): (F8, F8), c: usize) -> F8 {
        unsafe {
            let codes = if BITS == 8 {
                F8::bytes(row.add(8 * c))
            } else {
                F8::nibbles(row.cast::<u32>().add(c).read_unaligned())
            };
            zero.fma(codes, scale)
        }
    }
}

/// One kv head's prepared key and value (`key` and `value`, W floats each)
/// encoded into affine history row `destination`: keys at [`KEY_BITS`],
/// values at [`VALUE_BITS`].
///
/// # Safety
/// `destination` is a freshly reserved row no other work item of the call
/// reads or writes.
#[inline(always)]
pub unsafe fn store_affine_head(
    key: &[f32],
    value: &[f32],
    (key_codes, key_coefficients): (&SlabTensor<'_, U32, 3>, &SlabTensor<'_, F16, 3>),
    (value_codes, value_coefficients): (&SlabTensor<'_, U32, 3>, &SlabTensor<'_, F16, 3>),
    destination: usize,
    head: usize,
) {
    // SAFETY: the caller's.
    unsafe {
        affine_encode(
            key,
            KEY_BITS,
            key_codes.row_mut([destination, head, 0]),
            key_coefficients.row_mut([destination, head, 0]),
        );
        affine_encode(
            value,
            VALUE_BITS,
            value_codes.row_mut([destination, head, 0]),
            value_coefficients.row_mut([destination, head, 0]),
        );
    }
}

/// The K8/V4 rows of kv head `head` in slab-stored affine history.
#[inline(always)]
pub fn affine_rows<'a, const BITS: u32, C: Element<Storage = u16>>(
    codes: &SlabTensor<'a, U32, 3>,
    coefficients: &SlabTensor<'a, C, 3>,
    head: usize,
) -> AffineRows<'a, BITS, C> {
    assert_eq!(codes.rows_per_slab(), coefficients.rows_per_slab());
    AffineRows { codes: *codes, coefficients: *coefficients, head }
}

/// F16 `values` widened into `out`, eight at a time (`Dense::widen` bit for
/// bit).
///
/// # Safety
/// The tier runs [`F8`].
#[inline(always)]
unsafe fn widen_f16(values: &[u16], out: &mut [f32]) {
    assert_eq!(values.len(), out.len());
    let chunks = values.chunks_exact(8);
    let tail = chunks.remainder();
    for (chunk, target) in chunks.zip(out.chunks_exact_mut(8)) {
        // SAFETY: eight values in, eight out.
        unsafe { F8::widen::<F16>(chunk.as_ptr()).store(target.as_mut_ptr()) };
    }
    if !tail.is_empty() {
        // The tail's words gathered in one register and stored whole, which
        // the widen's 16-byte load forwards from (a padded copy's narrow
        // stores cannot forward).
        let words = tail.iter().rev().fold(0u128, |words, value| words << 16 | u128::from(*value));
        let mut widened = [0.0f32; 8];
        // SAFETY: eight values in (the tail's, then zeros), eight out.
        unsafe { F8::widen::<F16>((&words as *const u128).cast()).store(widened.as_mut_ptr()) };
        let at = values.len() - tail.len();
        out[at..].copy_from_slice(&widened[..tail.len()]);
    }
}

/// The code-expanding lane operations of [`AffineRows::chunk`] on AVX2.
#[cfg(target_arch = "x86_64")]
impl F8 {
    /// The eight 8-bit codes at `at`.
    #[inline(always)]
    unsafe fn bytes(at: *const u8) -> Self {
        use std::arch::x86_64::*;
        unsafe { Self(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(at.cast())))) }
    }

    /// The eight 4-bit codes of `word`, code k at bits 4k.
    #[inline(always)]
    unsafe fn nibbles(word: u32) -> Self {
        use std::arch::x86_64::*;
        unsafe {
            let shifted = _mm256_srlv_epi32(_mm256_set1_epi32(word as i32), _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28));
            Self(_mm256_cvtepi32_ps(_mm256_and_si256(shifted, _mm256_set1_epi32(15))))
        }
    }
}

/// The code-expanding lane operations of [`AffineRows::chunk`] on NEON.
#[cfg(target_arch = "aarch64")]
impl F8 {
    /// The eight 8-bit codes at `at`.
    #[inline(always)]
    unsafe fn bytes(at: *const u8) -> Self {
        use std::arch::aarch64::*;
        unsafe { Self::codes(vld1_u8(at)) }
    }

    /// The eight 4-bit codes of `word`, code k at bits 4k.
    #[inline(always)]
    unsafe fn nibbles(word: u32) -> Self {
        use std::arch::aarch64::*;
        unsafe {
            let bytes = vcreate_u8(u64::from(word));
            Self::codes(vzip1_u8(vand_u8(bytes, vdup_n_u8(15)), vshr_n_u8::<4>(bytes)))
        }
    }

    /// Eight codes, one a byte, widened to F32.
    #[inline(always)]
    unsafe fn codes(codes: std::arch::aarch64::uint8x8_t) -> Self {
        use std::arch::aarch64::*;
        unsafe {
            let wide = vmovl_u8(codes);
            Self([vcvtq_f32_u32(vmovl_u16(vget_low_u16(wide))), vcvtq_f32_u32(vmovl_high_u16(wide))])
        }
    }
}
