// Shared pieces of the gated attention entries on CPU
// (`gated_attention_decode`, `gated_attention_prefill` and their affine K8/V4
// history forms `gated_attention_{decode,prefill}_k8v4`): the per-row
// preparation (RMS norm and partial M-RoPE of every query and key head), a
// row's key walk, decode partition bounds, the online-softmax absorb, the
// fixed-order merge of partial states, the sigmoid gate, and the affine codec
// (encode on append, decode on read). The counterpart of
// `metal/lib/attention/attention.h`, `cuda/lib/attention/*.cuh` and
// `vulkan/lib/attention/*.glsl`; each entry keeps its own launch structure and
// calls these.
//
// Numerics follow the portable bodies (`attention.seismic`): prepared queries
// and keys are F32 rounded once to the activation element; scores, the
// softmax state and the output accumulate in F32; the history codec decodes
// in F32 (code * scale + zero). The online softmax absorbs keys in blocks of
// BLOCK (the portable `attention_update` absorbs a span at a time), and decode
// partitions merge in partition order: the reassociations every native
// backend is qualified under.

use super::super::core::{activation, reduce};
use seismic::cpu::{math, Dense, Tensor, F16};
use std::ops::Range;

/// Keys scored together before their values are absorbed.
pub const BLOCK: usize = 32;

/// The fewest keys of a decode partition: shorter rows use fewer partitions.
pub const PARTITION_KEYS: usize = 128;

/// Columns per affine (scale, zero) pair: the codec's group.
pub const GROUP: usize = 32;
/// Code bits of affine history keys and values.
pub const KEY_BITS: u32 = 8;
pub const VALUE_BITS: u32 = 4;

/// The head geometry of an entry: KV key/value heads of G query heads each,
/// rows of W = 2P + S columns whose first 2P rotate.
#[derive(Clone, Copy, Debug)]
pub struct Heads {
    pub kv: usize,
    pub g: usize,
    pub p: usize,
    pub w: usize,
}

impl Heads {
    pub fn new(kv: u64, g: u64, p: u64, s: u64) -> Self {
        Self { kv: kv as usize, g: g as usize, p: p as usize, w: (2 * p + s) as usize }
    }

    /// The geometry of an affine K8/V4 history entry, whose vectors are
    /// whole codec groups (the portable codec leaves a partial group's
    /// columns undefined); the declarations' `where` admits only those.
    pub fn affine(self) -> Self {
        assert!(self.w % GROUP == 0, "affine history needs head widths of whole {GROUP}-column groups, not {}", self.w);
        self
    }

    /// Query heads of a row.
    pub fn queries(&self) -> usize {
        self.kv * self.g
    }
}

/// The cosines and sines of a row's P rotary pairs: pair p by the row's
/// coordinate on axis `components[p]` at `frequencies[p]`.
pub fn angles(coordinates: &[i32], components: &[i32], frequencies: &[f32], cosines: &mut [f32], sines: &mut [f32]) {
    for (pair, (cosine, sine)) in cosines.iter_mut().zip(sines.iter_mut()).enumerate() {
        let angle = coordinates[components[pair] as usize] as f32 * frequencies[pair];
        (*sine, *cosine) = math::sin_cos(angle);
    }
}

/// The portable `norm_rotary_table` of one head row `x`, published to `A`:
/// RMS-normalized in F32 with `norm`, its first 2P columns rotated by the
/// row's `cosines` and `sines`.
pub fn norm_rotary<A: Dense>(
    x: &[A::Storage],
    norm: &[f32],
    cosines: &[f32],
    sines: &[f32],
    epsilon: f32,
    out: &mut [f32],
) {
    let (w, p) = (x.len(), cosines.len());
    let out = &mut out[..w];
    activation::widen::<A>(x, out);
    let inverse = reduce::rms_inverse(out, epsilon);
    for (value, weight) in out.iter_mut().zip(&norm[..w]) {
        *value = *value * inverse * weight;
    }
    for pair in 0..p {
        let (low, high) = (out[pair], out[pair + p]);
        let (cosine, sine) = (cosines[pair], sines[pair]);
        out[pair] = low * cosine - high * sine;
        out[pair + p] = high * cosine + low * sine;
    }
    for value in out.iter_mut() {
        *value = activation::publish::<A>(*value);
    }
}

/// Prepares one row: its KV * G queries (the first W columns of each query
/// head's query|gate pair) into `queries` and its KV keys into `keys`, F32
/// values of the activation element.
#[allow(clippy::too_many_arguments)]
pub fn prepare_row<A: Dense>(
    heads: Heads,
    query_gate: &[A::Storage],
    key: &[A::Storage],
    query_norm: &[f32],
    key_norm: &[f32],
    cosines: &[f32],
    sines: &[f32],
    epsilon: f32,
    queries: &mut [f32],
    keys: &mut [f32],
) {
    let w = heads.w;
    for head in 0..heads.queries() {
        let raw = &query_gate[head * 2 * w..][..w];
        norm_rotary::<A>(raw, query_norm, cosines, sines, epsilon, &mut queries[head * w..][..w]);
    }
    for head in 0..heads.kv {
        norm_rotary::<A>(&key[head * w..][..w], key_norm, cosines, sines, epsilon, &mut keys[head * w..][..w]);
    }
}

/// Where a run of a row's keys lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Keys {
    /// History rows.
    History,
    /// Rows of this batch (prepared keys, projected values).
    Fresh,
}

/// The number of keys a row sees: its admitted history spans, then its
/// fresh span.
pub fn total(spans: impl Iterator<Item = (i32, i32)>, fresh: (i32, i32)) -> usize {
    spans.map(|(lo, hi)| (hi - lo) as usize).sum::<usize>() + (fresh.1 - fresh.0).max(0) as usize
}

/// Calls `run(keys, first, count)` for each run of the row's key sequence
/// (its admitted history spans in order, then its fresh span) inside
/// `range`, positions counted along the sequence.
pub fn walk(
    spans: impl Iterator<Item = (i32, i32)>,
    fresh: (i32, i32),
    range: Range<usize>,
    mut run: impl FnMut(Keys, usize, usize),
) {
    let mut position = 0usize;
    let mut visit = |keys: Keys, lo: i32, hi: i32| {
        let count = (hi - lo) as usize;
        let start = range.start.max(position);
        let end = range.end.min(position + count);
        if start < end {
            run(keys, lo as usize + (start - position), end - start);
        }
        position += count;
    };
    for (lo, hi) in spans {
        visit(Keys::History, lo, hi);
    }
    if fresh.1 > fresh.0 {
        visit(Keys::Fresh, fresh.0, fresh.1);
    }
}

/// The keys of decode partition `partition` of a row seeing `total` keys:
/// consecutive partitions of max(PARTITION_KEYS, ceil(total / parts)) keys.
pub fn partition(total: usize, parts: usize, partition: usize) -> Range<usize> {
    let keys = PARTITION_KEYS.max(total.div_ceil(parts));
    (partition * keys).min(total)..((partition + 1) * keys).min(total)
}

/// The online-softmax state of G query heads: maximum and denominator per
/// head, the unnormalized output [G][W].
pub struct Online<'s> {
    pub maximum: &'s mut [f32],
    pub denominator: &'s mut [f32],
    pub accumulator: &'s mut [f32],
}

impl Online<'_> {
    /// The state before any key: maximum -inf, denominator and output 0.
    pub fn reset(&mut self) {
        self.maximum.fill(f32::NEG_INFINITY);
        self.denominator.fill(0.0);
        self.accumulator.fill(0.0);
    }

    /// Absorbs `count` keys against the G `queries` [G][W]:
    /// `load_key(j, out)` / `load_value(j, out)` give key and value j as F32.
    /// Blocks of BLOCK keys are scored, then absorbed as the portable
    /// `attention_update` absorbs a span. `scores` holds G * BLOCK values,
    /// `row` W.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub fn absorb(
        &mut self,
        queries: &[f32],
        scale: f32,
        count: usize,
        scores: &mut [f32],
        row: &mut [f32],
        mut load_key: impl FnMut(usize, &mut [f32]),
        mut load_value: impl FnMut(usize, &mut [f32]),
    ) {
        let g = self.maximum.len();
        let w = row.len();
        let mut first = 0;
        while first < count {
            let n = BLOCK.min(count - first);
            for j in 0..n {
                load_key(first + j, row);
                for head in 0..g {
                    scores[head * BLOCK + j] = reduce::dot(&queries[head * w..][..w], row) * scale;
                }
            }
            for head in 0..g {
                let scores = &mut scores[head * BLOCK..][..n];
                let next = self.maximum[head].max(reduce::max(scores));
                for score in scores.iter_mut() {
                    *score = math::exp(*score - next);
                }
                let carry = math::exp(self.maximum[head] - next);
                self.denominator[head] = self.denominator[head].mul_add(carry, reduce::sum(scores));
                for value in &mut self.accumulator[head * w..][..w] {
                    *value *= carry;
                }
                self.maximum[head] = next;
            }
            for j in 0..n {
                load_value(first + j, row);
                for head in 0..g {
                    let weight = scores[head * BLOCK + j];
                    for (value, v) in self.accumulator[head * w..][..w].iter_mut().zip(row.iter()) {
                        *value = weight.mul_add(*v, *value);
                    }
                }
            }
            first += n;
        }
    }
}

/// One history vector in affine storage: packed codes and (scale, zero)
/// F16 pairs per GROUP columns.
pub type AffineRow<'a> = (&'a [u32], &'a [u16]);

impl Online<'_> {
    /// [`Online::absorb`] over affine K8/V4 history, with the same result
    /// bits: keys and values decode in registers straight into the score
    /// dots and value updates. Every score is the same eight-lane fused dot
    /// combined by the same tree, and every accumulator element takes the
    /// same fused updates in key order; only the loops are tiled, so no row
    /// round-trips through memory and each accumulator block stays in
    /// registers across a block of keys.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub fn absorb_affine<'h>(
        &mut self,
        queries: &[f32],
        scale: f32,
        count: usize,
        scores: &mut [f32],
        row: &mut [f32],
        key: impl Fn(usize) -> AffineRow<'h>,
        value: impl Fn(usize) -> AffineRow<'h>,
    ) {
        #[cfg(target_arch = "aarch64")]
        affine_neon::absorb(self, queries, scale, count, scores, row, key, value);
        #[cfg(not(target_arch = "aarch64"))]
        self.absorb(
            queries,
            scale,
            count,
            scores,
            row,
            |j, out| {
                let (codes, coefficients) = key(j);
                affine_decode(codes, coefficients, KEY_BITS, out)
            },
            |j, out| {
                let (codes, coefficients) = value(j);
                affine_decode(codes, coefficients, VALUE_BITS, out)
            },
        );
    }
}

#[cfg(target_arch = "aarch64")]
mod affine_neon {
    use super::{math, reduce, AffineRow, Online, BLOCK, GROUP};
    use seismic::cpu::{Dense, F16};
    use std::arch::aarch64::*;

    /// Value columns updated together.
    const COLUMNS: usize = 16;

    /// Sixteen codes as `code * scale + zero`, fused.
    #[inline(always)]
    unsafe fn decode16(codes: uint8x16_t, scale: float32x4_t, zero: float32x4_t) -> [float32x4_t; 4] {
        unsafe {
            let (low, high) = (vmovl_u8(vget_low_u8(codes)), vmovl_u8(vget_high_u8(codes)));
            [
                vfmaq_f32(zero, vcvtq_f32_u32(vmovl_u16(vget_low_u16(low))), scale),
                vfmaq_f32(zero, vcvtq_f32_u32(vmovl_u16(vget_high_u16(low))), scale),
                vfmaq_f32(zero, vcvtq_f32_u32(vmovl_u16(vget_low_u16(high))), scale),
                vfmaq_f32(zero, vcvtq_f32_u32(vmovl_u16(vget_high_u16(high))), scale),
            ]
        }
    }

    /// The (scale, zero) of `group` as broadcast vectors.
    #[inline(always)]
    fn coefficients(pairs: &[u16], group: usize) -> (float32x4_t, float32x4_t) {
        unsafe {
            (
                vdupq_n_f32(F16::widen(pairs[2 * group])),
                vdupq_n_f32(F16::widen(pairs[2 * group + 1])),
            )
        }
    }

    /// `reduce::combine` of eight lanes held as (lanes 0..4, lanes 4..8).
    #[inline(always)]
    fn combine(low: float32x4_t, high: float32x4_t) -> f32 {
        unsafe {
            let pairs = vaddq_f32(low, high);
            (vgetq_lane_f32(pairs, 0) + vgetq_lane_f32(pairs, 2))
                + (vgetq_lane_f32(pairs, 1) + vgetq_lane_f32(pairs, 3))
        }
    }

    /// Dispatches on the query heads per key/value head, so each head's
    /// accumulators are registers rather than a runtime-indexed array.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn absorb<'h>(
        state: &mut Online<'_>,
        queries: &[f32],
        scale: f32,
        count: usize,
        scores: &mut [f32],
        row: &mut [f32],
        key: impl Fn(usize) -> AffineRow<'h>,
        value: impl Fn(usize) -> AffineRow<'h>,
    ) {
        match state.maximum.len() {
            1 => heads::<1>(state, queries, scale, count, scores, row.len(), key, value),
            2 => heads::<2>(state, queries, scale, count, scores, row.len(), key, value),
            4 => heads::<4>(state, queries, scale, count, scores, row.len(), key, value),
            8 => heads::<8>(state, queries, scale, count, scores, row.len(), key, value),
            _ => state.absorb(
                queries,
                scale,
                count,
                scores,
                row,
                |j, out| {
                    let (codes, coefficients) = key(j);
                    super::affine_decode(codes, coefficients, super::KEY_BITS, out)
                },
                |j, out| {
                    let (codes, coefficients) = value(j);
                    super::affine_decode(codes, coefficients, super::VALUE_BITS, out)
                },
            ),
        }
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn heads<'h, const G: usize>(
        state: &mut Online<'_>,
        queries: &[f32],
        scale: f32,
        count: usize,
        scores: &mut [f32],
        w: usize,
        key: impl Fn(usize) -> AffineRow<'h>,
        value: impl Fn(usize) -> AffineRow<'h>,
    ) {
        let g = G;
        assert!(state.maximum.len() == G && w % GROUP == 0 && queries.len() >= g * w);
        assert!(state.accumulator.len() >= g * w && scores.len() >= g * BLOCK);
        let empty: AffineRow<'h> = (&[], &[]);
        let mut first = 0;
        while first < count {
            let n = BLOCK.min(count - first);
            for j in 0..n {
                let (codes, pairs) = key(first + j);
                assert!(codes.len() * 4 >= w && pairs.len() * GROUP >= 2 * w);
                let bytes = codes.as_ptr().cast::<u8>();
                // SAFETY: the assertions bound every code byte and query read
                // below; the lanes follow `reduce::dot` element by element.
                unsafe {
                    let mut low = [vdupq_n_f32(0.0); G];
                    let mut high = [vdupq_n_f32(0.0); G];
                    for group in 0..w / GROUP {
                        let (s, z) = coefficients(pairs, group);
                        let packed = bytes.add(group * GROUP);
                        let keys = [
                            decode16(vld1q_u8(packed), s, z),
                            decode16(vld1q_u8(packed.add(16)), s, z),
                        ];
                        for chunk in 0..GROUP / 8 {
                            let column = group * GROUP + chunk * 8;
                            let half = &keys[chunk / 2];
                            let (a, b) = (half[(chunk % 2) * 2], half[(chunk % 2) * 2 + 1]);
                            for head in 0..g {
                                let q = queries.as_ptr().add(head * w + column);
                                low[head] = vfmaq_f32(low[head], vld1q_f32(q), a);
                                high[head] = vfmaq_f32(high[head], vld1q_f32(q.add(4)), b);
                            }
                        }
                    }
                    for head in 0..g {
                        scores[head * BLOCK + j] = combine(low[head], high[head]) * scale;
                    }
                }
            }
            for head in 0..g {
                let scores = &mut scores[head * BLOCK..][..n];
                let next = state.maximum[head].max(reduce::max(scores));
                for score in scores.iter_mut() {
                    *score = math::exp(*score - next);
                }
                let carry = math::exp(state.maximum[head] - next);
                state.denominator[head] = state.denominator[head].mul_add(carry, reduce::sum(scores));
                for value in &mut state.accumulator[head * w..][..w] {
                    *value *= carry;
                }
                state.maximum[head] = next;
            }
            let mut rows = [empty; BLOCK];
            for (j, row) in rows[..n].iter_mut().enumerate() {
                *row = value(first + j);
                assert!(row.0.len() * 8 >= w && row.1.len() * GROUP >= 2 * w);
            }
            for column in (0..w).step_by(COLUMNS) {
                let (group, half) = (column / GROUP, (column % GROUP) / COLUMNS);
                // SAFETY: `column + COLUMNS <= w` inside every head's
                // accumulator block, and every value row holds its w / 2
                // code bytes (asserted above).
                unsafe {
                    let mut blocks = [[vdupq_n_f32(0.0); 4]; G];
                    for head in 0..g {
                        let at = state.accumulator.as_ptr().add(head * w + column);
                        for r in 0..4 {
                            blocks[head][r] = vld1q_f32(at.add(4 * r));
                        }
                    }
                    for (j, (codes, pairs)) in rows[..n].iter().enumerate() {
                        let (s, z) = coefficients(pairs, group);
                        let nibbles = vld1_u8(codes.as_ptr().cast::<u8>().add(group * GROUP / 2 + half * 8));
                        let (lo, hi) = (vand_u8(nibbles, vdup_n_u8(15)), vshr_n_u8(nibbles, 4));
                        let decoded = decode16(vcombine_u8(vzip1_u8(lo, hi), vzip2_u8(lo, hi)), s, z);
                        for head in 0..g {
                            let weight = vdupq_n_f32(scores[head * BLOCK + j]);
                            for r in 0..4 {
                                blocks[head][r] = vfmaq_f32(blocks[head][r], decoded[r], weight);
                            }
                        }
                    }
                    for head in 0..g {
                        let at = state.accumulator.as_mut_ptr().add(head * w + column);
                        for r in 0..4 {
                            vst1q_f32(at.add(4 * r), blocks[head][r]);
                        }
                    }
                }
            }
            first += n;
        }
    }
}

/// The fresh rows of a batch as keys and values: prepared keys [M][KV][W]
/// (F32 values of the activation element) and the projected values
/// [M, KV * W].
pub struct Fresh<'a, A: Dense> {
    pub keys: &'a [f32],
    pub values: Tensor<'a, A, 2>,
}

/// Absorbs the part `range` of a row's key sequence (its admitted history
/// `spans`, then its `fresh` span) for kv head `kv_head` into `state`:
/// history rows through `history_key(token, out)` / `history_value(token,
/// out)`, fresh rows from `rows`. `scores` holds G * BLOCK values, `row` W.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn attend<A: Dense>(
    state: &mut Online<'_>,
    heads: Heads,
    kv_head: usize,
    queries: &[f32],
    scale: f32,
    scores: &mut [f32],
    row: &mut [f32],
    spans: impl Iterator<Item = (i32, i32)>,
    fresh: (i32, i32),
    range: Range<usize>,
    rows: &Fresh<'_, A>,
    mut history_key: impl FnMut(usize, &mut [f32]),
    mut history_value: impl FnMut(usize, &mut [f32]),
) {
    let (kv, w) = (heads.kv, heads.w);
    walk(spans, fresh, range, |keys, first, count| match keys {
        Keys::History => state.absorb(
            queries,
            scale,
            count,
            scores,
            row,
            |j, out| history_key(first + j, out),
            |j, out| history_value(first + j, out),
        ),
        Keys::Fresh => state.absorb(
            queries,
            scale,
            count,
            scores,
            row,
            |j, out| out.copy_from_slice(&rows.keys[((first + j) * kv + kv_head) * w..][..w]),
            |j, out| activation::widen::<A>(rows.values.span([first + j, kv_head * w], w), out),
        ),
    });
}

/// [`attend`] with affine K8/V4 history rows `history_key(token)` and
/// `history_value(token)`, absorbed by [`Online::absorb_affine`].
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn attend_affine<'h, A: Dense>(
    state: &mut Online<'_>,
    heads: Heads,
    kv_head: usize,
    queries: &[f32],
    scale: f32,
    scores: &mut [f32],
    row: &mut [f32],
    spans: impl Iterator<Item = (i32, i32)>,
    fresh: (i32, i32),
    range: Range<usize>,
    rows: &Fresh<'_, A>,
    history_key: impl Fn(usize) -> AffineRow<'h>,
    history_value: impl Fn(usize) -> AffineRow<'h>,
) {
    let (kv, w) = (heads.kv, heads.w);
    walk(spans, fresh, range, |keys, first, count| match keys {
        Keys::History => state.absorb_affine(
            queries,
            scale,
            count,
            scores,
            row,
            |j| history_key(first + j),
            |j| history_value(first + j),
        ),
        Keys::Fresh => state.absorb(
            queries,
            scale,
            count,
            scores,
            row,
            |j, out| out.copy_from_slice(&rows.keys[((first + j) * kv + kv_head) * w..][..w]),
            |j, out| activation::widen::<A>(rows.values.span([first + j, kv_head * w], w), out),
        ),
    });
}

/// Merges the partial states of one query head over its `parts` partitions
/// in partition order: partition `part` has output `partial(part)`, maximum
/// `maximum(part)` and denominator `denominator(part)`; an empty partition
/// (maximum -inf) contributes nothing. Writes the unnormalized output and
/// returns the denominator.
pub fn merge<'p>(
    parts: usize,
    maximum: impl Fn(usize) -> f32,
    denominator: impl Fn(usize) -> f32,
    partial: impl Fn(usize) -> &'p [f32],
    out: &mut [f32],
) -> f32 {
    let overall = (0..parts).map(&maximum).fold(f32::NEG_INFINITY, f32::max);
    out.fill(0.0);
    let mut total = 0.0f32;
    for part in 0..parts {
        let part_maximum = maximum(part);
        if part_maximum == f32::NEG_INFINITY {
            continue;
        }
        let factor = math::exp(part_maximum - overall);
        total = denominator(part).mul_add(factor, total);
        for (value, partial) in out.iter_mut().zip(partial(part)) {
            *value = partial.mul_add(factor, *value);
        }
    }
    total
}

/// The gated output of one query head, stored to `A`:
/// `(accumulator / max(denominator, 1e-30)) / (1 + exp(-gate))`.
pub fn gate<A: Dense>(accumulator: &[f32], denominator: f32, gate: &[A::Storage], out: &mut [A::Storage]) {
    let denominator = denominator.max(1e-30);
    for ((target, value), gate) in out.iter_mut().zip(accumulator).zip(gate) {
        let attended = value / denominator;
        *target = A::narrow(attended / (1.0 + math::exp(-A::widen(*gate))));
    }
}

pub use super::history::{affine_encode, affine_decode};
