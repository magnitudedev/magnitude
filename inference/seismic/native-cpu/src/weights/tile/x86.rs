//! The x86 integer core: the [`Units`] of a block of eight weight rows
//! against a strip of activation rows, from the codes as unsigned bytes
//! shifted by their least value (or signed, in the sign form).
//!
//! One loop serves every x86 tier, over two lane shapes ([`Lanes`]):
//! - [`Step`] (256-bit): lane `r` row `r`, a weight vector one step of four
//!   columns, an activation row's four codes broadcast;
//! - [`Pair`] (512-bit): lane `2 r + h` row `r` at step `2 p + h`, a weight
//!   vector two steps, an activation row's eight codes broadcast as one
//!   64-bit word (`Q8Tile::codes[p][a]`, which alternates the two steps'
//!   words), each row's two lanes added once a block;
//!
//! and three products ([`Multiply`]): the sign form ([`Sign`], step lanes
//! only: `maddubs` of the magnitudes and the signed activations), byte
//! products into 16-bit lanes ([`Bytes16`], `maddubs`), and VNNI dot
//! products into 32-bit lanes ([`Dots32`], `vpdpbusd`); codes are shifted
//! by the one rule of every x86 dot (`packed_x86::shift`). A chunk of
//! [`Chunks::together`] products is widened by each row's unit integer,
//! `vpmaddwd` and an add or VNNI's `vpdpwssd`, exact while the chunk's lanes
//! fit 16 bits: `J · p · 127 · span ≤ 32767` for `p` products a lane an
//! instruction (2 or 4), the activation codes being at most 127 in magnitude
//! (`quant::quantize`) and `span` the shifted codes' range. With integer unit
//! coefficients ([`scaled`]) the widened chunks of every unit accumulate into
//! the block's `scaled` in registers, and the shift's correction (`least Σ_u
//! widen_u Σ a`) and the `minimums` (`Σ_u minimum_u Σ a`) are the same
//! widening over pairs of 16-value groups' codes (`Side::pairs`) against the
//! block sums (`Q8Tile::sums`); direct coefficients' units ([`units`]) stay
//! apart, on step lanes.

use super::*;
use packed_x86::Products;
use std::arch::x86_64::*;

/// Shifted codes' byte products: `products`, `together` instructions a
/// chunk (dividing a unit's), the codes less `least`
/// (`packed_x86::shift`).
#[derive(Clone, Copy)]
pub(super) struct Chunks {
    products: Products,
    together: usize,
    least: i32,
}

/// An instance on step lanes: the sign form (a step's products widened at
/// once) or shifted codes.
#[derive(Clone, Copy)]
pub(super) enum Steps {
    Sign,
    Shifted(Chunks),
}

/// The integer-unit core's instance: on step lanes, or shifted codes on
/// pair lanes.
#[derive(Clone, Copy)]
pub(super) enum Form {
    Steps(Steps),
    Pairs(Chunks),
}

impl Steps {
    const fn least(self) -> i32 {
        match self {
            Self::Sign => 0,
            Self::Shifted(chunks) => chunks.least,
        }
    }

    /// The chunk size and shift of the instance's loop: the sign form widens
    /// each step at once, unshifted.
    const fn chunking(self) -> (usize, i32) {
        match self {
            Self::Sign => (1, 0),
            Self::Shifted(chunks) => (chunks.together, chunks.least),
        }
    }

    /// Whether the instance multiplies with `products` (not the sign form).
    const fn shifted_by(self, wanted: Products) -> bool {
        matches!(self, Self::Shifted(chunks) if chunks.products as u8 == wanted as u8)
    }
}

impl Form {
    const fn least(self) -> i32 {
        self.chunking().1
    }

    /// As [`Steps::chunking`].
    const fn chunking(self) -> (usize, i32) {
        match self {
            Self::Steps(steps) => steps.chunking(),
            Self::Pairs(chunks) => (chunks.together, chunks.least),
        }
    }

    /// Whether the instance is on pair lanes multiplying with `products`.
    const fn paired_by(self, wanted: Products) -> bool {
        matches!(self, Self::Pairs(chunks) if chunks.products as u8 == wanted as u8)
    }
}

/// The most instructions `j`, a power of two dividing `ops`, whose products
/// add in a lane that fits 16 bits: `j · per · 127 · span ≤ 32767`; zero
/// when not one does.
const fn most(ops: usize, per: i32, span: i32) -> usize {
    let mut j = ops;
    while j > 0 && j as i32 * per * 127 * span > 32767 {
        j /= 2;
    }
    j
}

/// Products `products` of codes `C` in chunks fitting 16-bit lanes of `ops`
/// instructions a unit, if the codes shift into them and a chunk fits.
const fn chunks<C: Codes>(products: Products, ops: usize) -> Option<Chunks> {
    let Some(least) = packed_x86::shift::<C>(products) else {
        return None;
    };
    let per = match products {
        Products::Bytes16 => 2,
        Products::Dots32 => 4,
    };
    match most(ops, per, C::GREATEST - C::LEAST) {
        0 => None,
        together => Some(Chunks { products, together, least }),
    }
}

/// The direct-coefficient core's instance for codes `C` and tier `V`: the
/// tier's products of codes that shift into them, else the sign form. Each
/// unit's VNNI sum stays apart unwidened, so in one chunk.
pub(super) const fn unit_form<C: Codes, K: Coefficients, V: Vectors>() -> Steps {
    let (products, steps) = (Products::of::<V>(), K::UNIT / 4);
    match products {
        Products::Dots32 => match packed_x86::shift::<C>(products) {
            Some(least) => Steps::Shifted(Chunks { products, together: steps, least }),
            None => Steps::Sign,
        },
        Products::Bytes16 => match chunks::<C>(products, steps) {
            Some(chunks) => Steps::Shifted(chunks),
            None => Steps::Sign,
        },
    }
}

/// The integer-unit core's instance for codes `C`, coefficients `K` and
/// tier `V`, on pair lanes with 512-bit vectors: of the products the codes
/// shift into and a chunk fits, the one whose unit costs the fewest
/// multiplier instructions plus instructions in all (16-bit lanes: `ops`
/// products, `ops - chunks` adds and `chunks` widenings; 32-bit lanes: `ops`
/// products and `chunks` widenings); else the sign form.
pub(super) const fn block_form<C: Codes, K: Coefficients, V: Vectors>() -> Form {
    let wide = V::TIER.vector_bytes() == 64;
    let ops = K::UNIT / 4 / if wide { 2 } else { 1 };
    let bytes16 = chunks::<C>(Products::Bytes16, ops);
    let dots32 = if matches!(Products::of::<V>(), Products::Dots32) { chunks::<C>(Products::Dots32, ops) } else { None };
    let best = if cost(bytes16, ops) < cost(dots32, ops) { bytes16 } else { dots32 };
    match best {
        None => Form::Steps(Steps::Sign),
        Some(chunks) if wide => Form::Pairs(chunks),
        Some(chunks) => Form::Steps(Steps::Shifted(chunks)),
    }
}

/// A unit's multiplier instructions plus instructions in all, in
/// [`block_form`]'s chunks.
const fn cost(chunks: Option<Chunks>, ops: usize) -> usize {
    match chunks {
        None => usize::MAX,
        Some(Chunks { products: Products::Bytes16, together, .. }) => 3 * ops + ops / together,
        Some(Chunks { products: Products::Dots32, together, .. }) => 2 * (ops + ops / together),
    }
}

/// The value the core's codes `C` are shifted by, for coefficients `K` on
/// tier `V`: [`Chunks::least`], or 0 in the sign form.
pub(super) const fn least<C: Codes, K: Coefficients, V: Vectors>() -> i32 {
    if K::INTEGER_UNITS {
        block_form::<C, K, V>().least()
    } else {
        unit_form::<C, K, V>().least()
    }
}

/// `[0, 0, 1, 1, .., 7, 7]`: each 32-bit lane of a 256-bit vector twice.
const DUPLICATE: [i32; 16] = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7];

/// `[0, 16, 1, 17, .., 7, 23]`: two 256-bit vectors' lanes interleaved.
const INTERLEAVE: [i32; 16] = [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23];

/// `[0, 8, 1, 9, .., 7, 15]`: the two halves of one vector interleaved.
pub(super) const INTERLEAVE_ONE: [i32; 16] = [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15];

/// Steps `2 pair` and `2 pair + 1` of a rows8 panel span of `BITS`-bit
/// codes from `units` as [`Source::x86_pair`] takes them: each row's word
/// of the two steps' slots, or of the two units of 8-bit codes.
///
/// # Safety
/// The tier has AVX-512 (BW); the steps lie in the span.
#[inline(always)]
pub(super) unsafe fn panel_pair(units: *const u8, pair: usize, bits: u32) -> __m512i {
    let slots = 8 / bits as usize;
    unsafe {
        if bits == 8 {
            let (first, second) = (_mm256_loadu_si256(units.add(64 * pair).cast()), _mm256_loadu_si256(units.add(64 * pair + 32).cast()));
            return _mm512_permutex2var_epi32(_mm512_castsi256_si512(first), load(&INTERLEAVE), _mm512_castsi256_si512(second));
        }
        // Both steps lie in one unit, at slots `2 pair % slots` and the next.
        let bytes = _mm256_loadu_si256(units.add(32 * (2 * pair / slots)).cast());
        let words = _mm512_permutexvar_epi32(load(&DUPLICATE), _mm512_castsi256_si512(bytes));
        let shift = (2 * pair % slots) as i64 * bits as i64;
        let codes = _mm512_srlv_epi32(words, _mm512_set1_epi64(((shift + bits as i64) << 32) | shift));
        _mm512_and_si512(codes, _mm512_set1_epi8(((1u16 << bits) - 1) as u8 as i8))
    }
}

/// A 512-bit vector of sixteen lanes from `lanes`.
#[inline(always)]
unsafe fn load(lanes: &[i32; 16]) -> __m512i {
    unsafe { _mm512_loadu_si512(lanes.as_ptr().cast()) }
}

/// The shape of the core's accumulators.
trait Lanes {
    type V: Copy;
    /// Steps of four columns a weight vector holds.
    const STEPS: usize;
    /// Pairs of 16-value groups a side vector holds.
    const PAIRS: usize;

    unsafe fn zero() -> Self::V;

    /// The weight vector of step `step` (of the first, for two).
    unsafe fn weights<S: Source>(source: S, step: usize) -> Self::V;

    /// Activation row `a`'s codes of step `step`, as the weight vector's
    /// lanes take them.
    unsafe fn activations<const R: usize>(x: &Q8Tile<R>, step: usize, a: usize) -> Self::V;

    /// `acc` plus `sums`, per 32-bit lane.
    unsafe fn add(acc: Self::V, sums: Self::V) -> Self::V;

    /// `acc` plus, per 32-bit lane, the products of the 16-bit halves of
    /// `pairs` and `codes`.
    unsafe fn widen<T: Vectors>(acc: Self::V, pairs: Self::V, codes: Self::V) -> Self::V;

    /// Each row's code `codes[r]` in its lanes: in both 16-bit halves, or in
    /// the low half with the high zero (`low`).
    unsafe fn codes(codes: &[i32; 8], low: bool) -> Self::V;

    /// Side vector `slot`: the group pairs `pairs` holds for it, of which
    /// `0..live` are the block's.
    unsafe fn side(pairs: &[[i32; 8]; Q8_BLOCK / PACKET], slot: usize, live: usize) -> Self::V;

    /// Activation row `a`'s block sums of side vector `slot`'s group pairs.
    unsafe fn sums<const R: usize>(x: &Q8Tile<R>, slot: usize, a: usize) -> Self::V;

    /// Each row's lanes added.
    unsafe fn rows(acc: Self::V) -> __m256i;
}

/// 256-bit lanes, a row each: [`Lanes`].
struct Step;

impl Lanes for Step {
    type V = __m256i;
    const STEPS: usize = 1;
    const PAIRS: usize = 1;

    #[inline(always)]
    unsafe fn zero() -> __m256i {
        unsafe { _mm256_setzero_si256() }
    }

    #[inline(always)]
    unsafe fn weights<S: Source>(source: S, step: usize) -> __m256i {
        unsafe { source.x86(step) }
    }

    #[inline(always)]
    unsafe fn activations<const R: usize>(x: &Q8Tile<R>, step: usize, a: usize) -> __m256i {
        unsafe { _mm256_set1_epi32(x.codes[step / 2][a].as_ptr().add(4 * (step % 2)).cast::<i32>().read_unaligned()) }
    }

    #[inline(always)]
    unsafe fn add(acc: __m256i, sums: __m256i) -> __m256i {
        unsafe { _mm256_add_epi32(acc, sums) }
    }

    #[inline(always)]
    unsafe fn widen<T: Vectors>(acc: __m256i, pairs: __m256i, codes: __m256i) -> __m256i {
        unsafe {
            if T::QUANTIZED_DOTS == 2 {
                packed_x86::dpwssd::<T>(acc, pairs, codes)
            } else {
                _mm256_add_epi32(acc, _mm256_madd_epi16(pairs, codes))
            }
        }
    }

    #[inline(always)]
    unsafe fn codes(codes: &[i32; 8], low: bool) -> __m256i {
        unsafe {
            let codes = _mm256_loadu_si256(codes.as_ptr().cast());
            let codes_low = _mm256_and_si256(codes, _mm256_set1_epi32(0xffff));
            if low {
                codes_low
            } else {
                _mm256_or_si256(codes_low, _mm256_slli_epi32::<16>(codes))
            }
        }
    }

    #[inline(always)]
    unsafe fn side(pairs: &[[i32; 8]; Q8_BLOCK / PACKET], slot: usize, _: usize) -> __m256i {
        unsafe { _mm256_loadu_si256(pairs[slot].as_ptr().cast()) }
    }

    #[inline(always)]
    unsafe fn sums<const R: usize>(x: &Q8Tile<R>, slot: usize, a: usize) -> __m256i {
        unsafe { _mm256_set1_epi32(x.sums[a].as_ptr().add(2 * slot).cast::<i32>().read_unaligned()) }
    }

    #[inline(always)]
    unsafe fn rows(acc: __m256i) -> __m256i {
        acc
    }
}

/// 512-bit lanes, a row's two steps each: [`Lanes`].
struct Pair;

impl Lanes for Pair {
    type V = __m512i;
    const STEPS: usize = 2;
    const PAIRS: usize = 2;

    #[inline(always)]
    unsafe fn zero() -> __m512i {
        unsafe { _mm512_setzero_si512() }
    }

    #[inline(always)]
    unsafe fn weights<S: Source>(source: S, step: usize) -> __m512i {
        unsafe { source.x86_pair(step / 2) }
    }

    #[inline(always)]
    unsafe fn activations<const R: usize>(x: &Q8Tile<R>, step: usize, a: usize) -> __m512i {
        unsafe { _mm512_set1_epi64(x.codes[step / 2][a].as_ptr().cast::<i64>().read_unaligned()) }
    }

    #[inline(always)]
    unsafe fn add(acc: __m512i, sums: __m512i) -> __m512i {
        unsafe { _mm512_add_epi32(acc, sums) }
    }

    #[inline(always)]
    unsafe fn widen<T: Vectors>(acc: __m512i, pairs: __m512i, codes: __m512i) -> __m512i {
        unsafe {
            if T::QUANTIZED_DOTS == 2 {
                _mm512_dpwssd_epi32(acc, pairs, codes)
            } else {
                _mm512_add_epi32(acc, _mm512_madd_epi16(pairs, codes))
            }
        }
    }

    #[inline(always)]
    unsafe fn codes(codes: &[i32; 8], low: bool) -> __m512i {
        unsafe { _mm512_permutexvar_epi32(load(&DUPLICATE), _mm512_castsi256_si512(Step::codes(codes, low))) }
    }

    #[inline(always)]
    unsafe fn side(pairs: &[[i32; 8]; Q8_BLOCK / PACKET], slot: usize, live: usize) -> __m512i {
        unsafe {
            let first = _mm256_loadu_si256(pairs[2 * slot].as_ptr().cast());
            let second = if 2 * slot + 1 < live { _mm256_loadu_si256(pairs[2 * slot + 1].as_ptr().cast()) } else { _mm256_setzero_si256() };
            _mm512_permutex2var_epi32(_mm512_castsi256_si512(first), load(&INTERLEAVE), _mm512_castsi256_si512(second))
        }
    }

    #[inline(always)]
    unsafe fn sums<const R: usize>(x: &Q8Tile<R>, slot: usize, a: usize) -> __m512i {
        unsafe { _mm512_set1_epi64(x.sums[a].as_ptr().add(4 * slot).cast::<i64>().read_unaligned()) }
    }

    #[inline(always)]
    unsafe fn rows(acc: __m512i) -> __m256i {
        unsafe { _mm512_cvtepi64_epi32(_mm512_add_epi32(acc, _mm512_srli_epi64::<32>(acc))) }
    }
}

/// How an instance multiplies a weight vector by an activation vector on
/// lanes `L`.
trait Multiply<L: Lanes> {
    /// Whether the products add in 32-bit lanes (VNNI's), each widened by
    /// its low 16-bit half; else in 16-bit lanes.
    const DOTS32: bool;

    /// `acc` plus the products of `weights` and `activations`.
    unsafe fn product<T: Vectors>(acc: L::V, weights: L::V, activations: L::V) -> L::V;
}

/// The sign form: `maddubs` of the codes' magnitudes and the activations
/// signed by the codes, into 16-bit lanes.
struct Sign;

/// `maddubs` of shifted codes into 16-bit lanes.
struct Bytes16;

/// `vpdpbusd` of shifted codes into 32-bit lanes.
struct Dots32;

impl Multiply<Step> for Sign {
    const DOTS32: bool = false;

    #[inline(always)]
    unsafe fn product<T: Vectors>(acc: __m256i, weights: __m256i, activations: __m256i) -> __m256i {
        unsafe { _mm256_add_epi16(acc, _mm256_maddubs_epi16(_mm256_abs_epi8(weights), _mm256_sign_epi8(activations, weights))) }
    }
}

impl Multiply<Step> for Bytes16 {
    const DOTS32: bool = false;

    #[inline(always)]
    unsafe fn product<T: Vectors>(acc: __m256i, weights: __m256i, activations: __m256i) -> __m256i {
        unsafe { _mm256_add_epi16(acc, _mm256_maddubs_epi16(weights, activations)) }
    }
}

impl Multiply<Step> for Dots32 {
    const DOTS32: bool = true;

    #[inline(always)]
    unsafe fn product<T: Vectors>(acc: __m256i, weights: __m256i, activations: __m256i) -> __m256i {
        unsafe { packed_x86::dpbusd::<T>(acc, weights, activations) }
    }
}

impl Multiply<Pair> for Bytes16 {
    const DOTS32: bool = false;

    #[inline(always)]
    unsafe fn product<T: Vectors>(acc: __m512i, weights: __m512i, activations: __m512i) -> __m512i {
        unsafe { _mm512_add_epi16(acc, _mm512_maddubs_epi16(weights, activations)) }
    }
}

impl Multiply<Pair> for Dots32 {
    const DOTS32: bool = true;

    #[inline(always)]
    unsafe fn product<T: Vectors>(acc: __m512i, weights: __m512i, activations: __m512i) -> __m512i {
        unsafe { _mm512_dpbusd_epi32(acc, weights, activations) }
    }
}

/// `acc` plus unit `u`'s products of the source against each activation row
/// of the strip, `ops` instructions a unit in chunks of `together`, each
/// chunk widened by `codes`, or added unwidened (`None`).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn unit<L: Lanes, P: Multiply<L>, V: Vectors, S: Source, const W: usize, const R: usize>(
    source: S,
    x: &Q8Tile<R>,
    u: usize,
    ops: usize,
    together: usize,
    codes: Option<L::V>,
    acc: &mut [L::V; W],
) {
    unsafe {
        for chunk in 0..ops / together {
            let mut products = [L::zero(); W];
            for op in 0..together {
                let step = (u * ops + chunk * together + op) * L::STEPS;
                let weights = L::weights(source, step);
                for (a, products) in products.iter_mut().enumerate() {
                    *products = P::product::<V>(*products, weights, L::activations(x, step, a));
                }
            }
            for (acc, products) in acc.iter_mut().zip(products) {
                *acc = match codes {
                    Some(codes) => L::widen::<V>(*acc, products, codes),
                    None => L::add(*acc, products),
                };
            }
        }
    }
}

/// Direct coefficients' [`Sums`] of the source's first `packets` packets
/// against the strip `x`, on step lanes in [`unit_form`]'s instance.
///
/// # Safety
/// The tier has AVX2; the packets lie in the source's block.
#[inline(always)]
pub(super) unsafe fn units<C: Codes, K: Coefficients, V: Vectors, S: Source, const W: usize, const R: usize>(
    source: S,
    x: &Q8Tile<R>,
    packets: usize,
) -> Sums<W> {
    // One instance's arm alone is instantiated: each condition is a
    // constant of the instance, so the other arms are never compiled.
    let (together, least) = const { unit_form::<C, K, V>().chunking() };
    unsafe {
        if const { matches!(unit_form::<C, K, V>(), Steps::Sign) } {
            unit_sums::<Sign, K, V, S, W, R>(source, x, packets, together, least)
        } else if const { unit_form::<C, K, V>().shifted_by(Products::Bytes16) } {
            unit_sums::<Bytes16, K, V, S, W, R>(source, x, packets, together, least)
        } else if const { unit_form::<C, K, V>().shifted_by(Products::Dots32) } {
            unit_sums::<Dots32, K, V, S, W, R>(source, x, packets, together, least)
        } else {
            unreachable!("every instance form is one of the arms")
        }
    }
}

#[inline(always)]
unsafe fn unit_sums<P: Multiply<Step>, K: Coefficients, V: Vectors, S: Source, const W: usize, const R: usize>(
    source: S,
    x: &Q8Tile<R>,
    packets: usize,
    together: usize,
    least: i32,
) -> Sums<W> {
    let mut sums = Sums { units: [[[0; 8]; W]; UNITS], count: packets * PACKET / K::UNIT };
    unsafe {
        // 16-bit chunks widen by one; VNNI's 32-bit sums add as they are.
        let ones = if P::DOTS32 { None } else { Some(Step::codes(&[1; 8], false)) };
        for u in 0..sums.count {
            let mut acc = [Step::zero(); W];
            unit::<Step, P, V, S, W, R>(source, x, u, K::UNIT / 4, together, ones, &mut acc);
            for (a, (out, acc)) in sums.units[u].iter_mut().zip(acc).enumerate() {
                let acc = if least != 0 { _mm256_add_epi32(acc, _mm256_set1_epi32(least * unit_sum(x, K::UNIT, u, a))) } else { acc };
                _mm256_storeu_si256(out.as_mut_ptr().cast(), acc);
            }
        }
    }
    sums
}

/// Integer-unit coefficients' [`Scaled`] of the source's first `packets`
/// packets against the strip `x`, in [`block_form`]'s instance.
///
/// # Safety
/// The tier has AVX2 (and AVX-512 for pair lanes); the packets lie in the
/// source's block.
#[inline(always)]
pub(super) unsafe fn scaled<C: Codes, K: Coefficients, V: Vectors, S: Source, const W: usize, const R: usize>(
    source: S,
    side: &Side,
    x: &Q8Tile<R>,
    packets: usize,
) -> Scaled<W> {
    // Each row's unit integer in a 16-bit half (`Lanes::codes`).
    const { assert!(K::MAGNITUDE <= i16::MAX as i32, "unit integers fit 16 bits") };
    // As in [`units`]: the instance's arm alone is compiled.
    let (together, least) = const { block_form::<C, K, V>().chunking() };
    unsafe {
        if const { matches!(block_form::<C, K, V>(), Form::Steps(Steps::Sign)) } {
            block::<Step, Sign, K, V, S, W, R>(source, side, x, packets, together, least)
        } else if const { matches!(block_form::<C, K, V>(), Form::Steps(steps) if steps.shifted_by(Products::Bytes16)) } {
            block::<Step, Bytes16, K, V, S, W, R>(source, side, x, packets, together, least)
        } else if const { matches!(block_form::<C, K, V>(), Form::Steps(steps) if steps.shifted_by(Products::Dots32)) } {
            block::<Step, Dots32, K, V, S, W, R>(source, side, x, packets, together, least)
        } else if const { block_form::<C, K, V>().paired_by(Products::Bytes16) } {
            block::<Pair, Bytes16, K, V, S, W, R>(source, side, x, packets, together, least)
        } else if const { block_form::<C, K, V>().paired_by(Products::Dots32) } {
            block::<Pair, Dots32, K, V, S, W, R>(source, side, x, packets, together, least)
        } else {
            unreachable!("every instance form is one of the arms")
        }
    }
}

#[inline(always)]
unsafe fn block<L: Lanes, P: Multiply<L>, K: Coefficients, V: Vectors, S: Source, const W: usize, const R: usize>(
    source: S,
    side: &Side,
    x: &Q8Tile<R>,
    packets: usize,
    together: usize,
    least: i32,
) -> Scaled<W> {
    let (count, ops) = (packets * PACKET / K::UNIT, K::UNIT / 4 / L::STEPS);
    let mut out = Scaled { scaled: [[0; 8]; W], minimums: [[0; 8]; W] };
    unsafe {
        let mut scaled = [L::zero(); W];
        for u in 0..count {
            let codes = L::codes(&side.widen[u], P::DOTS32);
            unit::<L, P, V, S, W, R>(source, x, u, ops, together, Some(codes), &mut scaled);
        }
        // The shift's correction into `scaled`, and the minimums.
        let mut minimums = [L::zero(); W];
        if least != 0 || K::BIASED {
            for slot in 0..packets.div_ceil(L::PAIRS) {
                let (shift, minimum) = (L::side(&side.pairs[0], slot, packets), L::side(&side.pairs[1], slot, packets));
                for a in 0..W {
                    let sums = L::sums(x, slot, a);
                    if least != 0 {
                        scaled[a] = L::widen::<V>(scaled[a], sums, shift);
                    }
                    if K::BIASED {
                        minimums[a] = L::widen::<V>(minimums[a], sums, minimum);
                    }
                }
            }
        }
        for a in 0..W {
            _mm256_storeu_si256(out.scaled[a].as_mut_ptr().cast(), L::rows(scaled[a]));
            if K::BIASED {
                _mm256_storeu_si256(out.minimums[a].as_mut_ptr().cast(), L::rows(minimums[a]));
            }
        }
    }
    out
}
