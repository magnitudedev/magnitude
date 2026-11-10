//! The tile kernel: weight rows against one or more quantized activation
//! rows, each weight block decoded once for all of them.
//!
//! Per eight weight rows and activation block, the integer core multiplies
//! the rows' codes a step of four columns at a time (`[row][4 codes]`: one
//! vector on AVX2, two on NEON) against each activation row of a strip of
//! up to eight ([`Q8Tile`]; a [`Q8Block`] is a one-row one), and the
//! combine applies the rows' coefficients ([`Side`]). A rows8 tile's code
//! spans are such steps already (its panels, `registry::Layout::Rows8`):
//! read in place for one strip ([`Spans`]), else decoded once into a
//! [`Panel`]; any other rows' packets are transposed into one.
//!
//! The integer core forms, per row and activation row, the exact integers
//! the single-row dots form, the coefficient component's own
//! (`Coefficients::Units`): with integer unit coefficients the block's
//! `scaled` and `minimums` ([`Scaled`]), else each unit's (a coefficient
//! group, at most a packet: `Coefficients::UNIT`) `Σ code * activation`
//! ([`Sums`]). The combine then applies the coefficient
//! component's own `f32` arithmetic in its own order, every row a lane
//! (`Coefficients::tile_combine`). So every output is the single-row dot's
//! bit for bit, whatever the instructions, the source and the strip.
//!
//! x86 takes the codes as every x86 dot does (`packed_x86::shift`): unsigned
//! bytes shifted by their least value, whose sum against the activations it
//! corrects by the block sums, added in 16- or 32-bit lanes before widening
//! by the units' integers (`x86`); codes whose range exceeds the byte
//! products' take the sign form, one step at a time.

use super::*;
use crate::quant::Q8Tile;

#[cfg(target_arch = "x86_64")]
mod x86;

/// Units of a block at most: sixteen-value groups.
pub(super) const UNITS: usize = Q8_BLOCK / 16;

/// Steps of four columns of a block.
const STEPS: usize = Q8_BLOCK / 4;

/// Eight rows' codes of one block: `[k/4][row][4 codes]`, as the core
/// takes them (x86: the values less [`least`]; else the values).
#[repr(C, align(64))]
pub(super) struct Panel([[i8; 32]; STEPS]);

/// Eight rows' coefficients of one block, per unit and row where they
/// vary by unit: the integer each unit's sum is widened by (a scale code,
/// or 1), the minimum codes and the factors (scale and bias factor) of
/// hierarchical coefficients, the scales and biases of direct ones; and, for
/// the x86 core, per pair of 16-value groups the codes the block sums are
/// widened by ([`group_pairs`]): `[0]` the scale codes times the codes'
/// shift, `[1]` the minimum codes.
pub struct Side {
    pub(super) widen: [[i32; 8]; UNITS],
    pub(super) minimums: [[i32; 8]; UNITS],
    pub(super) factors: [[f32; 8]; 2],
    pub(super) scales: [[f32; 8]; UNITS],
    pub(super) biases: [[f32; 8]; UNITS],
    #[cfg(target_arch = "x86_64")]
    pub(super) pairs: [[[i32; 8]; Q8_BLOCK / PACKET]; 2],
}

/// Direct coefficients' exact integers: per unit `u` of the first `count`,
/// activation row `a` and weight row `r`, `units[u][a][r] = Σ code a` of
/// true code values.
pub struct Sums<const W: usize> {
    pub(super) units: [[[i32; 8]; W]; UNITS],
    pub(super) count: usize,
}

/// Integer-unit coefficients' exact integers: per activation row and weight
/// row the block's `scaled` (`Σ_u widen_u Σ code a`, of true code values)
/// and `minimums` (`Σ_u minimum_u Σ a`).
pub struct Scaled<const W: usize> {
    pub(super) scaled: [[i32; 8]; W],
    pub(super) minimums: [[i32; 8]; W],
}

/// A coefficient component's exact integers of a block against a strip of
/// `W` activation rows (`Coefficients::Units`).
pub trait Integers<const W: usize>: Sized {
    /// From each of the first `count` units' sum `units[u][a][r]` widened by
    /// its integer (`Side::widen`).
    fn of_units<K: Coefficients, const R: usize>(units: [[[i32; 8]; W]; UNITS], count: usize, side: &Side, x: &Q8Tile<R>) -> Self;

    /// The x86 core's, of the source's first `packets` packets.
    ///
    /// # Safety
    /// The tier has AVX2 (and AVX-512 for pair lanes); the packets lie in
    /// the source's block.
    #[cfg(target_arch = "x86_64")]
    unsafe fn x86<C: Codes, K: Coefficients, V: Vectors, S: Source, const R: usize>(source: S, side: &Side, x: &Q8Tile<R>, packets: usize) -> Self;
}

impl<const W: usize> Integers<W> for Sums<W> {
    #[inline(always)]
    fn of_units<K: Coefficients, const R: usize>(units: [[[i32; 8]; W]; UNITS], count: usize, _: &Side, _: &Q8Tile<R>) -> Self {
        Self { units, count }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn x86<C: Codes, K: Coefficients, V: Vectors, S: Source, const R: usize>(source: S, _: &Side, x: &Q8Tile<R>, packets: usize) -> Self {
        unsafe { x86::units::<C, K, V, S, W, R>(source, x, packets) }
    }
}

impl<const W: usize> Integers<W> for Scaled<W> {
    /// The units added into `scaled`, and `minimums` from the block sums.
    #[inline(always)]
    fn of_units<K: Coefficients, const R: usize>(units: [[[i32; 8]; W]; UNITS], count: usize, side: &Side, x: &Q8Tile<R>) -> Self {
        let mut out = Self { scaled: [[0; 8]; W], minimums: [[0; 8]; W] };
        for a in 0..W {
            for u in 0..count {
                let codes = unit_sum(x, K::UNIT, u, a);
                for r in 0..8 {
                    out.scaled[a][r] += units[u][a][r];
                    out.minimums[a][r] += side.minimums[u][r] * codes;
                }
            }
        }
        out
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn x86<C: Codes, K: Coefficients, V: Vectors, S: Source, const R: usize>(source: S, side: &Side, x: &Q8Tile<R>, packets: usize) -> Self {
        unsafe { x86::scaled::<C, K, V, S, W, R>(source, side, x, packets) }
    }
}

/// The sum of activation row `a`'s codes over unit `u` of `unit` values.
#[inline(always)]
pub(super) fn unit_sum<const R: usize>(x: &Q8Tile<R>, unit: usize, u: usize, a: usize) -> i32 {
    // A loop, not an iterator chain: its `fold` would compile out of line.
    let (per, mut sum) = (unit / 16, 0);
    for j in u * per..(u + 1) * per {
        sum += i32::from(x.sums[a][j]);
    }
    sum
}

/// The eight scales from `first`, one after another, each as
/// [`Scale::read`] reads it.
///
/// # Safety
/// `first` addresses eight scales.
#[inline(always)]
pub(super) unsafe fn read8<S: Scale, V: Vectors>(first: *const u8) -> [f32; 8] {
    let mut scales = [0.0f32; 8];
    #[cfg(target_arch = "x86_64")]
    if V::QUANTIZED_DOTS >= 1 {
        unsafe { std::arch::x86_64::_mm256_storeu_ps(scales.as_mut_ptr(), S::read8(first, S::BYTES)) };
        return scales;
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let [low, high] = S::read8_neon(first, S::BYTES);
        std::arch::aarch64::vst1q_f32(scales.as_mut_ptr(), low);
        std::arch::aarch64::vst1q_f32(scales.as_mut_ptr().add(4), high);
        return scales;
    }
    #[allow(unreachable_code)]
    {
        for (r, scale) in scales.iter_mut().enumerate() {
            *scale = unsafe { S::read(first.add(r * S::BYTES)) };
        }
        scales
    }
}

/// The value the core takes codes `C` less, which it adds back against the
/// block sums: nonzero only for x86's shifted codes (`x86::least`).
const fn least<C: Codes, K: Coefficients, V: Vectors>() -> i32 {
    #[cfg(target_arch = "x86_64")]
    if V::QUANTIZED_DOTS >= 1 {
        return x86::least::<C, K, V>();
    }
    0
}

/// Lane `r` of pair `i` of `out`, for pairs `0..packets`: the codes
/// (`codes[u][r]`, per unit of `unit` values) of row `r`'s 16-value groups
/// `2 i` and `2 i + 1`, times `factor`, as the low and high 16-bit halves.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn group_pairs(codes: &[[i32; 8]; UNITS], unit: usize, factor: i32, packets: usize, out: &mut [[i32; 8]; Q8_BLOCK / PACKET]) {
    for (i, out) in out.iter_mut().enumerate().take(packets) {
        let (low, high) = (&codes[2 * i * 16 / unit], &codes[(2 * i + 1) * 16 / unit]);
        for r in 0..8 {
            // Each product fits 16 bits (asserted where `tile` makes them).
            out[r] = ((low[r] * factor) & 0xffff) | ((high[r] * factor) << 16);
        }
    }
}

/// Where the core reads its weight steps.
pub trait Source: Copy {
    /// Step `step`'s codes of the eight rows as the AVX2 core takes them.
    ///
    /// # Safety
    /// The tier has AVX2; `step` lies in the source's block.
    #[cfg(target_arch = "x86_64")]
    unsafe fn x86(self, step: usize) -> std::arch::x86_64::__m256i;

    /// Steps `2 pair` and `2 pair + 1` as the AVX-512 core takes them: lane
    /// `2 r + h` row `r`'s four codes of step `2 pair + h`.
    ///
    /// # Safety
    /// The tier has AVX-512 (BW); both steps lie in the source's block.
    #[cfg(target_arch = "x86_64")]
    unsafe fn x86_pair(self, pair: usize) -> std::arch::x86_64::__m512i;

    /// Step `step`'s values of rows `4 quad .. 4 quad + 4`.
    ///
    /// # Safety
    /// `step` lies in the source's block.
    #[cfg(target_arch = "aarch64")]
    unsafe fn neon(self, step: usize, quad: usize) -> std::arch::aarch64::int8x16_t;

    /// Step `step`'s values, without quantized dots.
    fn values(self, step: usize) -> [i8; 32];
}

/// A decoded panel.
#[derive(Clone, Copy)]
struct Decoded<'a>(&'a Panel);

impl Source for Decoded<'_> {
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn x86(self, step: usize) -> std::arch::x86_64::__m256i {
        unsafe { std::arch::x86_64::_mm256_load_si256(self.0 .0[step].as_ptr().cast()) }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn x86_pair(self, pair: usize) -> std::arch::x86_64::__m512i {
        use std::arch::x86_64::*;
        unsafe {
            let steps = _mm512_loadu_si512(self.0 .0[2 * pair].as_ptr().cast());
            _mm512_permutexvar_epi32(_mm512_loadu_si512(x86::INTERLEAVE_ONE.as_ptr().cast()), steps)
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn neon(self, step: usize, quad: usize) -> std::arch::aarch64::int8x16_t {
        unsafe { std::arch::aarch64::vld1q_s8(self.0 .0[step][16 * quad..].as_ptr()) }
    }

    #[inline(always)]
    fn values(self, step: usize) -> [i8; 32] {
        self.0 .0[step]
    }
}

/// A rows8 tile's code spans of one block (its low plane's, and its high
/// plane's), a step decoded where read: its unit's slot, the high bits
/// joined, each raw code's value (`Codes::PANEL`), as the core of
/// coefficients `K` on tier `V` takes them.
struct Spans<C, K, V> {
    low: *const u8,
    high: *const u8,
    codes: PhantomData<(C, K, V)>,
}

impl<C, K, V> Clone for Spans<C, K, V> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<C, K, V> Copy for Spans<C, K, V> {}

impl<C: Codes, K: Coefficients, V: Vectors> Spans<C, K, V> {
    /// How the codes lie in the spans (no spans hold grid codes, whose
    /// rows are never `Row::panel_tile`).
    const PANEL: PanelCodes = match C::PANEL {
        Some(panel) => panel,
        None => PanelCodes { low: 8, high: 0, table: None, offset: 0 },
    };

    /// The x86 operand of each raw code: its value, less [`least`].
    #[cfg(target_arch = "x86_64")]
    const OPERANDS: [i8; 16] = {
        let (mut operands, mut raw) = ([0i8; 16], 0);
        while raw < 16 {
            if let Some(table) = Self::PANEL.table {
                operands[raw] = (table[raw] as i32 - least::<C, K, V>()) as u8 as i8;
            }
            raw += 1;
        }
        operands
    };

    /// The x86 operand's offset from an untabled raw code.
    #[cfg(target_arch = "x86_64")]
    const OFFSET: i8 = (Self::PANEL.offset as i32 - least::<C, K, V>()) as u8 as i8;
}

impl<C: Codes, K: Coefficients, V: Vectors> Source for Spans<C, K, V> {
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn x86(self, step: usize) -> std::arch::x86_64::__m256i {
        use std::arch::x86_64::*;
        let panel = Self::PANEL;
        unsafe {
            let mut raw = packed_x86::panel_step(self.low, step, panel.low);
            if panel.high > 0 {
                let high = packed_x86::panel_step(self.high, step, panel.high);
                raw = _mm256_or_si256(raw, _mm256_sll_epi16(high, _mm_cvtsi32_si128(panel.low as i32)));
            }
            if panel.table.is_some() {
                _mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128(Self::OPERANDS.as_ptr().cast())), raw)
            } else if Self::OFFSET != 0 {
                _mm256_add_epi8(raw, _mm256_set1_epi8(Self::OFFSET))
            } else {
                raw
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn x86_pair(self, pair: usize) -> std::arch::x86_64::__m512i {
        use std::arch::x86_64::*;
        let panel = Self::PANEL;
        unsafe {
            let mut raw = x86::panel_pair(self.low, pair, panel.low);
            if panel.high > 0 {
                let high = x86::panel_pair(self.high, pair, panel.high);
                raw = _mm512_or_si512(raw, _mm512_sll_epi16(high, _mm_cvtsi32_si128(panel.low as i32)));
            }
            if panel.table.is_some() {
                _mm512_shuffle_epi8(_mm512_broadcast_i32x4(_mm_loadu_si128(Self::OPERANDS.as_ptr().cast())), raw)
            } else if Self::OFFSET != 0 {
                _mm512_add_epi8(raw, _mm512_set1_epi8(Self::OFFSET))
            } else {
                raw
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn neon(self, step: usize, quad: usize) -> std::arch::aarch64::int8x16_t {
        use std::arch::aarch64::*;
        let panel = Self::PANEL;
        unsafe {
            let mut raw = packed_neon::panel_step(self.low, step, quad, panel.low);
            if panel.high > 0 {
                let high = packed_neon::panel_step(self.high, step, quad, panel.high);
                raw = vorrq_u8(raw, vshlq_u8(high, vdupq_n_s8(panel.low as i8)));
            }
            match panel.table {
                Some(table) => vqtbl1q_s8(vld1q_s8(table.as_ptr()), raw),
                None => vaddq_s8(vreinterpretq_s8_u8(raw), vdupq_n_s8(panel.offset)),
            }
        }
    }

    fn values(self, _: usize) -> [i8; 32] {
        unreachable!("spans are read in place only with quantized dots")
    }
}

/// `out[a n + r] = (row first + r) · activation a` for the `n` rows from
/// `first` and the `m` activation rows of `x` (`m.div_ceil(R) * blocks`
/// tiles of `R` rows, `[row group][block]`): each output the single-row
/// quantized dot's, bit for bit.
///
/// # Safety
/// `first` addresses `n` rows of the format with `geometry` and `k` values.
#[inline(always)]
pub(super) unsafe fn tile<C: Codes, K: Coefficients, const STORAGE: usize, V: Vectors, const R: usize>(
    first: *const u8,
    geometry: &RowGeometry,
    n: usize,
    x: &[Q8Tile<R>],
    m: usize,
    k: usize,
    out: &mut [f32],
) {
    let (blocks, strips) = (crate::quant::blocks(k), m.div_ceil(R));
    assert!(x.len() == strips * blocks && out.len() == m * n, "{m} x {n} products of {blocks} blocks");
    out.fill(0.0);
    let mut panel = Panel([[0; 32]; STEPS]);
    let mut side = Side {
        widen: [[1; 8]; UNITS],
        minimums: [[0; 8]; UNITS],
        factors: [[0.0; 8]; 2],
        scales: [[0.0; 8]; UNITS],
        biases: [[0.0; 8]; UNITS],
        #[cfg(target_arch = "x86_64")]
        pairs: [[[0; 8]; Q8_BLOCK / PACKET]; 2],
    };
    // The group pairs' codes fit 16 bits.
    const { assert!(least::<C, K, V>().unsigned_abs() as i32 * K::MAGNITUDE <= i16::MAX as i32, "group pairs fit 16 bits") };
    let least = const { least::<C, K, V>() };
    for group in (0..n).step_by(8) {
        let live = (n - group).min(8);
        let mut rows = geometry.rows::<8>(first.wrapping_add(group * geometry.stride));
        let tiled = C::PANEL.is_some() && Row::panel_tile(&rows, live);
        // Rows past the operand read the group's first again; their sums
        // are never stored.
        for r in live..8 {
            rows[r] = rows[0];
        }
        for b in 0..blocks {
            let packets = block_packets(b, k).len();
            unsafe { K::tile_side::<STORAGE, V>(&rows, b, packets, tiled, &mut side) };
            #[cfg(target_arch = "x86_64")]
            if K::INTEGER_UNITS && V::QUANTIZED_DOTS >= 1 {
                if least != 0 {
                    group_pairs(&side.widen, K::UNIT, least, packets, &mut side.pairs[0]);
                }
                if K::BIASED {
                    group_pairs(&side.minimums, K::UNIT, 1, packets, &mut side.pairs[1]);
                }
            }
            // Strip `s` against `source`, as wide as its rows.
            macro_rules! strip {
                ($source:expr, $s:expr) => {{
                    let (x, width) = (&x[$s * blocks + b], (m - R * $s).min(R));
                    let out = &mut out[R * $s * n + group..];
                    // A strip is at most `R` rows wide: the widths above `R`
                    // are never compiled.
                    macro_rules! wide {
                        ($w:literal) => {
                            if const { $w <= R } {
                                unsafe { strip::<C, K, V, _, $w, R>($source, &side, x, packets, width, live, n, out) }
                            } else {
                                unreachable!("a strip is at most {R} rows wide")
                            }
                        };
                    }
                    match width {
                        1 => wide!(1),
                        2 => wide!(2),
                        3 | 4 => wide!(4),
                        _ => wide!(8),
                    }
                }};
            }
            if tiled && V::QUANTIZED_DOTS >= 1 {
                let at = At::group(&rows[0], b * (Q8_BLOCK / STORAGE));
                let spans = Spans::<C, K, V> {
                    low: at.plane(RowPlaneRole::Codes),
                    high: at.plane(RowPlaneRole::High),
                    codes: PhantomData,
                };
                if strips == 1 {
                    strip!(spans, 0);
                    continue;
                }
                unsafe { fill::<V, _>(spans, packets * 8, &mut panel) };
            } else {
                unsafe { convert::<C, STORAGE, V>(&rows, b, packets, tiled, least, &mut panel) };
            }
            for s in 0..strips {
                strip!(Decoded(&panel), s);
            }
        }
    }
}

/// One strip of `width` (at most `W`) activation rows against the source's
/// steps of the first `packets` packets, each product of the block added to
/// `out[a n + r]`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn strip<C: Codes, K: Coefficients, V: Vectors, S: Source, const W: usize, const R: usize>(
    source: S,
    side: &Side,
    x: &Q8Tile<R>,
    packets: usize,
    width: usize,
    live: usize,
    n: usize,
    out: &mut [f32],
) {
    let units = unsafe { integers::<C, K, V, S, W, R>(source, side, x, packets) };
    let values = K::tile_combine::<W, R>(side, &units, x);
    for (a, values) in values.iter().enumerate().take(width) {
        for (sum, value) in out[a * n..a * n + live].iter_mut().zip(values) {
            *sum += value;
        }
    }
}

/// The integer core: the coefficient component's exact integers
/// (`Coefficients::Units`) of the first `packets` packets of the source
/// against the strip.
#[inline(always)]
unsafe fn integers<C: Codes, K: Coefficients, V: Vectors, S: Source, const W: usize, const R: usize>(
    source: S,
    side: &Side,
    x: &Q8Tile<R>,
    packets: usize,
) -> K::Units<W> {
    #[cfg(target_arch = "x86_64")]
    if V::QUANTIZED_DOTS >= 1 {
        return unsafe { <K::Units<W> as Integers<W>>::x86::<C, K, V, S, R>(source, side, x, packets) };
    }
    let mut units = [[[0i32; 8]; W]; UNITS];
    let (count, steps) = (packets * PACKET / K::UNIT, K::UNIT / 4);
    #[cfg(target_arch = "aarch64")]
    if V::QUANTIZED_DOTS >= 1 {
        unsafe { core_neon::<S, W, R>(source, side, x, count, steps, &mut units) };
        return <K::Units<W> as Integers<W>>::of_units::<K, R>(units, count, side, x);
    }
    for (u, units) in units.iter_mut().enumerate().take(count) {
        for (a, units) in units.iter_mut().enumerate() {
            for (r, sum) in units.iter_mut().enumerate() {
                let mut dot = 0i32;
                for step in u * steps..(u + 1) * steps {
                    let values = source.values(step);
                    for i in 0..4 {
                        dot += i32::from(values[4 * r + i]) * i32::from(x.codes[step / 2][a][4 * (step % 2) + i]);
                    }
                }
                *sum = dot * side.widen[u][r];
            }
        }
    }
    <K::Units<W> as Integers<W>>::of_units::<K, R>(units, count, side, x)
}

/// The NEON core: per four rows and step a weight vector against each
/// activation row's four codes by element `sdot` (a lane of the strip's
/// `ld2` vectors; of one row's sixteen codes alone), each unit's sums
/// widened by each row's integer.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn core_neon<S: Source, const W: usize, const R: usize>(
    source: S,
    side: &Side,
    x: &Q8Tile<R>,
    count: usize,
    steps: usize,
    units: &mut [[[i32; 8]; W]; UNITS],
) {
    use std::arch::aarch64::*;
    unsafe {
        for quad in 0..2 {
            for u in 0..count {
                let mut sums = [vdupq_n_s32(0); W];
                if R == 1 {
                    // One row: steps `4 q .. 4 q + 4`'s codes a vector.
                    for four in (u * steps / 4)..((u + 1) * steps / 4) {
                        let codes = vld1q_s8(x.codes[2 * four].as_ptr().cast());
                        for lane in 0..4 {
                            sums[0] = by_lane(sums[0], source.neon(4 * four + lane, quad), codes, lane);
                        }
                    }
                } else {
                    for pair in (u * steps / 2)..((u + 1) * steps / 2) {
                        // Rows 0..4 and 4..8 of the strip: four codes a
                        // lane, columns 8 pair .. + 4 then + 4 .. + 8.
                        let codes = x.codes[pair].as_ptr().cast::<i32>();
                        let low = vld2q_s32(codes);
                        let high = if W > 4 { vld2q_s32(codes.add(8)) } else { low };
                        for half in 0..2 {
                            let weights = source.neon(2 * pair + half, quad);
                            let codes = if half == 0 { [low.0, high.0] } else { [low.1, high.1] };
                            for (a, sums) in sums.iter_mut().enumerate() {
                                *sums = by_lane(*sums, weights, vreinterpretq_s8_s32(codes[a / 4]), a % 4);
                            }
                        }
                    }
                }
                let widen = vld1q_s32(side.widen[u][4 * quad..].as_ptr());
                for (out, sums) in units[u].iter_mut().zip(sums) {
                    vst1q_s32(out[4 * quad..].as_mut_ptr(), vmulq_s32(sums, widen));
                }
            }
        }
    }
}

/// `acc` plus the by-element `sdot` of `weights` and lane `lane` of
/// `codes`. (A function, not a closure: a closure would not take the
/// caller's features, and its `sdot` would not inline.)
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn by_lane(
    acc: std::arch::aarch64::int32x4_t,
    weights: std::arch::aarch64::int8x16_t,
    codes: std::arch::aarch64::int8x16_t,
    lane: usize,
) -> std::arch::aarch64::int32x4_t {
    unsafe {
        match lane {
            0 => packed_neon::sdot_lane::<0>(acc, weights, codes),
            1 => packed_neon::sdot_lane::<1>(acc, weights, codes),
            2 => packed_neon::sdot_lane::<2>(acc, weights, codes),
            _ => packed_neon::sdot_lane::<3>(acc, weights, codes),
        }
    }
}

/// The first `steps` steps of `source` decoded into `panel`, once for
/// every strip.
#[inline(always)]
unsafe fn fill<V: Vectors, S: Source>(source: S, steps: usize, panel: &mut Panel) {
    for step in 0..steps {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm256_store_si256(panel.0[step].as_mut_ptr().cast(), source.x86(step));
        }
        #[cfg(target_arch = "aarch64")]
        for quad in 0..2 {
            unsafe { std::arch::aarch64::vst1q_s8(panel.0[step][16 * quad..].as_mut_ptr(), source.neon(step, quad)) };
        }
    }
}

/// Packets `0..packets` of block `b` of the eight `rows` into `panel`:
/// each row's packet as the core takes it (x86: the values less `least`,
/// [`least`]; else the values), transposed so a step's codes of the eight
/// rows lie together; or, `tiled` (a rows8 tile's lanes, without quantized
/// dots), its panel spans' values.
#[inline(always)]
unsafe fn convert<C: Codes, const STORAGE: usize, V: Vectors>(rows: &[Row; 8], b: usize, packets: usize, tiled: bool, least: i32, panel: &mut Panel) {
    #[cfg(not(target_arch = "x86_64"))]
    let _ = least;
    let mut blocks = [Block::of::<STORAGE>(&rows[0], b); 8];
    for (block, row) in blocks.iter_mut().zip(rows).skip(1) {
        *block = Block::of::<STORAGE>(row, b);
    }
    #[cfg(target_arch = "x86_64")]
    if V::QUANTIZED_DOTS >= 1 {
        use std::arch::x86_64::*;
        unsafe {
            for local in packets_of(packets) {
                let mut codes = [_mm256_setzero_si256(); 8];
                for r in 0..8 {
                    let at = blocks[r].at::<STORAGE>(local);
                    codes[r] = if least != 0 { C::x86_shifted(at) } else { C::x86(at) };
                }
                // The 8 x 8 transpose of four-code words: step j of the
                // packet holds word j of each row.
                // (Loops, not closures: a closure would not take the
                // caller's features, and its intrinsics would not inline.)
                let mut t = [_mm256_setzero_si256(); 8];
                for p in 0..4 {
                    t[2 * p] = _mm256_unpacklo_epi32(codes[2 * p], codes[2 * p + 1]);
                    t[2 * p + 1] = _mm256_unpackhi_epi32(codes[2 * p], codes[2 * p + 1]);
                }
                // Words j and j + 4 of rows 0..4 in `q[j]`, of rows 4..8 in
                // `q[4 + j]`.
                let mut q = [_mm256_setzero_si256(); 8];
                for half in 0..2 {
                    let (a, b) = (4 * half, 4 * half + 2);
                    q[4 * half] = _mm256_unpacklo_epi64(t[a], t[b]);
                    q[4 * half + 1] = _mm256_unpackhi_epi64(t[a], t[b]);
                    q[4 * half + 2] = _mm256_unpacklo_epi64(t[a + 1], t[b + 1]);
                    q[4 * half + 3] = _mm256_unpackhi_epi64(t[a + 1], t[b + 1]);
                }
                for j in 0..4 {
                    let steps = &mut panel.0[8 * local..];
                    _mm256_store_si256(steps[j].as_mut_ptr().cast(), _mm256_permute2x128_si256::<0x20>(q[j], q[4 + j]));
                    _mm256_store_si256(steps[4 + j].as_mut_ptr().cast(), _mm256_permute2x128_si256::<0x31>(q[j], q[4 + j]));
                }
            }
        }
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if V::QUANTIZED_DOTS >= 1 {
        use std::arch::aarch64::*;
        unsafe {
            for local in packets_of(packets) {
                let mut codes = [[vdupq_n_s8(0); 2]; 8];
                for r in 0..8 {
                    codes[r] = C::neon(blocks[r].at::<STORAGE>(local));
                }
                for half in 0..2 {
                    for quad in 0..2 {
                        let four = [0, 1, 2, 3].map(|r| codes[4 * quad + r][half]);
                        for (c, chunk) in packed_neon::chunks4(four).into_iter().enumerate() {
                            vst1q_s8(panel.0[8 * local + 4 * half + c][16 * quad..].as_mut_ptr(), chunk);
                        }
                    }
                }
            }
        }
        return;
    }
    if tiled {
        // The spans' units are the panel's steps, each raw code's value.
        let codes = C::PANEL.expect("codes of rows8 panels");
        let block = blocks[0].at::<STORAGE>(0);
        let (low, high) = (block.plane(RowPlaneRole::Codes), block.plane(RowPlaneRole::High));
        let raw = |units: *const u8, bits: u32, step: usize, byte: usize| unsafe {
            let slots = 8 / bits as usize;
            (*units.add(32 * (step / slots) + byte) >> (step % slots * bits as usize)) & ((1u16 << bits) - 1) as u8
        };
        for step in 0..8 * packets {
            for byte in 0..32 {
                let mut code = raw(low, codes.low, step, byte);
                if codes.high > 0 {
                    code |= raw(high, codes.high, step, byte) << codes.low;
                }
                panel.0[step][byte] = codes.value(code);
            }
        }
        return;
    }
    for local in packets_of(packets) {
        for (r, block) in blocks.iter().enumerate() {
            let values = unsafe { C::values(block.at::<STORAGE>(local)) };
            for (j, word) in values.chunks_exact(4).enumerate() {
                panel.0[8 * local + j][4 * r..4 * r + 4].copy_from_slice(word);
            }
        }
    }
}
