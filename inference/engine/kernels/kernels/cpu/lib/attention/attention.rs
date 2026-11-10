// Shared pieces of the attention family entries on CPU (`attention_decode`,
// `attention_prefill` and their affine K8/V4 history forms
// `attention_{decode,prefill}_k8v4`): the per-row preparation (optional RMS
// norm and amplitude-scaled partial M-RoPE of every query and key head), a
// row's key walk, decode partition bounds, the online-softmax absorb, the
// fixed-order merge of partial states, the output gate, and the affine codec
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

use super::super::core::{activation, reduce, rounding::Rounding};
use seismic::cpu::{math, slab::SlabTensor, Dense, F16};
use std::ops::Range;

/// Keys scored together before their values are absorbed.
pub const BLOCK: usize = 32;

/// The fewest keys of a decode partition: shorter rows use fewer partitions.
pub const PARTITION_KEYS: usize = 128;

/// Columns of the native SIMD affine group32 implementation. Its entry
/// eligibility checks the descriptor coefficient extent before dispatch.
pub const GROUP: usize = 32;
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

/// Which two columns of a head row each rotary pair turns (the portable
/// `rotate_pairs`): pair p turns (p, p + P) or, with `adjacent` (RA = 1),
/// (2p, 2p + 1), within the leading 2P columns or, with `trailing` (RT = 1),
/// the trailing ones.
#[derive(Clone, Copy, Debug)]
pub struct Layout {
    pub adjacent: bool,
    pub trailing: bool,
}

impl Layout {
    /// Pairs (p, p + P) over the leading 2P columns.
    pub const HALF_SPLIT_LEADING: Self = Self { adjacent: false, trailing: false };

    pub fn new(ra: u64, rt: u64) -> Self {
        Self { adjacent: ra == 1, trailing: rt == 1 }
    }

    /// The (lo, hi) columns of pair `pair` of `p` in a row of `w` columns.
    #[inline(always)]
    pub fn columns(self, pair: usize, p: usize, w: usize) -> (usize, usize) {
        let first = if self.trailing { w - 2 * p } else { 0 };
        if self.adjacent {
            (first + 2 * pair, first + 2 * pair + 1)
        } else {
            (first + pair, first + pair + p)
        }
    }
}

/// The structure of an `attention_*` entry (see `attention.seismic`): its
/// heads, I output gates interleaved after each query head's W columns (0 or
/// W), U separate gates per query head (0, 1 or W), whether the layer has
/// fresh keys and values of its own (F = 1), whether queries (NQ = 1), keys
/// (N = 1) and values (NV = 1) are RMS-normalized, the rotary pairs' layout
/// (RA, RT), whether attended rows are turned back by the output rotary
/// (RO = 1), and whether values are the stored keys (VK = 1: the fresh value
/// is the prepared key, history values are read from the key plane, and no
/// value is taken, normalized or stored). The declarations' `where` admits
/// only these forms.
#[derive(Clone, Copy, Debug)]
pub struct Form {
    pub heads: Heads,
    pub interleaved: usize,
    pub separate: usize,
    pub fresh: bool,
    pub query_norm: bool,
    pub key_norm: bool,
    pub value_norm: bool,
    pub layout: Layout,
    pub output_rotary: bool,
    pub stored_key: bool,
    /// The trained rounding of the prepared key (KR = 1): applied in F32
    /// before the key is published, attended or stored, to the key held in
    /// the activation element after its norm and after its rotation.
    pub key_rounding: Option<Rounding>,
}

impl Form {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kv: u64,
        g: u64,
        p: u64,
        s: u64,
        i: u64,
        u: u64,
        f: u64,
        nq: u64,
        n: u64,
        nv: u64,
        layout: Layout,
        ro: u64,
        vk: u64,
    ) -> Self {
        Self {
            heads: Heads::new(kv, g, p, s),
            interleaved: i as usize,
            separate: u as usize,
            fresh: f == 1,
            query_norm: nq == 1,
            key_norm: n == 1,
            value_norm: nv == 1,
            layout,
            output_rotary: ro == 1,
            stored_key: vk == 1,
            key_rounding: None,
        }
    }

    /// The form with its prepared keys rounded as the dense entries' `KR`
    /// and four `key_*` scalars say: KR = 1 rounds by the scalars, which then
    /// name a rounding, and KR = 0 rounds nothing, the scalars naming none.
    pub fn with_key_rounding(self, kr: u64, block: i32, element: i32, scale: i32, amax_floor: f32) -> Self {
        let rounding = Rounding::of(block, element, scale, amax_floor);
        assert_eq!(kr == 1, rounding.is_some(), "KR {kr} beside key rounding block {block}");
        Self { key_rounding: rounding, ..self }
    }

    /// Whether the row takes a value of its own: a fresh layer whose values
    /// are not its stored keys.
    pub fn own_values(self) -> bool {
        self.fresh && !self.stored_key
    }

    /// The form over affine K8/V4 history ([`Heads::affine`]).
    pub fn affine(self) -> Self {
        Self { heads: self.heads.affine(), ..self }
    }
}

/// [`angles`] with each pair's cosine and sine scaled by its amplitude.
pub fn scaled_angles(
    coordinates: &[i32],
    components: &[i32],
    frequencies: &[f32],
    amplitudes: &[f32],
    cosines: &mut [f32],
    sines: &mut [f32],
) {
    angles(coordinates, components, frequencies, cosines, sines);
    for ((cosine, sine), amplitude) in cosines.iter_mut().zip(sines.iter_mut()).zip(amplitudes) {
        *cosine *= amplitude;
        *sine *= amplitude;
    }
}

/// The portable `head_rotary` of one head row `x`, published to `A`:
/// RMS-normalized in F32 with `norm` when present, its pairs (in `layout`)
/// rotated by the row's (amplitude-scaled) `cosines` and `sines`.
pub fn prepare_head<A: Dense>(
    x: &[A::Storage],
    norm: Option<&[f32]>,
    layout: Layout,
    cosines: &[f32],
    sines: &[f32],
    epsilon: f32,
    out: &mut [f32],
) {
    prepare_rounded_head::<A>(x, norm, layout, cosines, sines, epsilon, None, None, out);
}

/// `values` as `hold` holds them (each value through it), or as they are
/// without one: a rounded row's hold in its activation element
/// (`activation::publish`), the portable `held_in_activation`.
pub fn held(hold: Option<fn(f32) -> f32>, values: &mut [f32]) {
    if let Some(hold) = hold {
        values.iter_mut().for_each(|value| *value = hold(*value));
    }
}

/// [`prepare_head`] with the rotated row rounded by `rounding` in F32
/// before it is published (the portable `block_round` of `head_rotary`),
/// and, with `hold`, every value held by it after the norm (when there is
/// one) and after the rotation, before the rounding (a stored row's BF16
/// holds, `activation::publish` of the activation element).
#[allow(clippy::too_many_arguments)]
pub fn prepare_rounded_head<A: Dense>(
    x: &[A::Storage],
    norm: Option<&[f32]>,
    layout: Layout,
    cosines: &[f32],
    sines: &[f32],
    epsilon: f32,
    rounding: Option<Rounding>,
    hold: Option<fn(f32) -> f32>,
    out: &mut [f32],
) {
    let (w, p) = (x.len(), cosines.len());
    let out = &mut out[..w];
    activation::widen::<A>(x, out);
    if let Some(norm) = norm {
        let inverse = reduce::rms_inverse(out, epsilon);
        for (value, weight) in out.iter_mut().zip(&norm[..w]) {
            *value = *value * inverse * weight;
        }
        held(hold, out);
    }
    for pair in 0..p {
        let (lo, hi) = layout.columns(pair, p, w);
        let (low, high) = (out[lo], out[hi]);
        let (cosine, sine) = (cosines[pair], sines[pair]);
        out[lo] = low * cosine - high * sine;
        out[hi] = high * cosine + low * sine;
    }
    held(hold, out);
    if let Some(rounding) = rounding {
        rounding.apply(out);
    }
    for value in out.iter_mut() {
        *value = activation::publish::<A>(*value);
    }
}

/// The row's amplitude-scaled (cosines, sines) for its output rotary, in
/// `angles` (2P floats), when `form` turns its outputs back (RO = 1); none
/// otherwise (`angles` is then empty).
pub fn output_angles<'a>(
    form: Form,
    coordinates: &[i32],
    components: &[i32],
    frequencies: &[f32],
    amplitudes: &[f32],
    angles: &'a mut [f32],
) -> Option<(&'a [f32], &'a [f32])> {
    if !form.output_rotary {
        return None;
    }
    let (cosines, sines) = angles.split_at_mut(form.heads.p);
    scaled_angles(coordinates, components, frequencies, amplitudes, cosines, sines);
    Some((cosines, sines))
}

/// The portable `output_rotary` of one query head's published output `out`
/// (RO = 1, ungated, so the published attended row): its pairs (in
/// `layout`) turned back by the row's negated angles, the amplitude-scaled
/// `cosines` and `sines` of [`scaled_angles`], in F32 and published again.
pub fn output_rotary<A: Dense>(out: &mut [A::Storage], layout: Layout, cosines: &[f32], sines: &[f32]) {
    let (w, p) = (out.len(), cosines.len());
    for pair in 0..p {
        let (lo, hi) = layout.columns(pair, p, w);
        let (low, high) = (A::widen(out[lo]), A::widen(out[hi]));
        let (cosine, sine) = (cosines[pair], -sines[pair]);
        out[lo] = A::narrow(low * cosine - high * sine);
        out[hi] = A::narrow(high * cosine + low * sine);
    }
}

/// Prepares one row of an `attention_*` entry: its KV * G queries (the first
/// W columns of each `query(head)`) into `queries`, and when the layer has
/// fresh rows its KV keys into `keys` and its KV values (`head_norm`ed with
/// `value_norm` when present, or the prepared keys for stored-key values,
/// whose `value` is empty) into `values`, all F32 values of the activation
/// element.
#[allow(clippy::too_many_arguments)]
pub fn prepare_form_row<'q, A: Dense>(
    form: Form,
    query: impl Fn(usize) -> &'q [A::Storage],
    key: &[A::Storage],
    value: &[A::Storage],
    norms: (Option<&[f32]>, Option<&[f32]>, Option<&[f32]>),
    cosines: &[f32],
    sines: &[f32],
    epsilon: f32,
    queries: &mut [f32],
    keys: &mut [f32],
    values: &mut [f32],
) where
    A::Storage: 'q,
{
    let w = form.heads.w;
    let (query_norm, key_norm, value_norm) = norms;
    for head in 0..form.heads.queries() {
        prepare_head::<A>(&query(head)[..w], query_norm, form.layout, cosines, sines, epsilon, &mut queries[head * w..][..w]);
    }
    if !form.fresh {
        return;
    }
    for head in 0..form.heads.kv {
        prepare_rounded_head::<A>(
            &key[head * w..][..w],
            key_norm,
            form.layout,
            cosines,
            sines,
            epsilon,
            form.key_rounding,
            form.key_rounding.map(|_| activation::publish::<A> as fn(f32) -> f32),
            &mut keys[head * w..][..w],
        );
    }
    if form.stored_key {
        values[..form.heads.kv * w].copy_from_slice(&keys[..form.heads.kv * w]);
        return;
    }
    for head in 0..form.heads.kv {
        prepare_head::<A>(&value[head * w..][..w], value_norm, form.layout, &[], &[], epsilon, &mut values[head * w..][..w]);
    }
}

/// Appends one row's prepared keys and values ([`prepare_form_row`]) at
/// dense history row `destination`: the key plane alone for stored-key
/// values, whose value plane has no rows.
///
/// # Safety
/// `destination` is a freshly reserved row no other work item of the call
/// reads or writes.
pub unsafe fn store_dense_row<A: Dense>(
    form: Form,
    keys: &[f32],
    values: &[f32],
    history_key: &SlabTensor<'_, A, 3>,
    history_value: &SlabTensor<'_, A, 3>,
    destination: usize,
) {
    let w = form.heads.w;
    for head in 0..form.heads.kv {
        // SAFETY: the caller's.
        activation::store::<A>(&keys[head * w..][..w], unsafe { history_key.row_mut([destination, head, 0]) });
        if !form.stored_key {
            // SAFETY: the caller's.
            activation::store::<A>(&values[head * w..][..w], unsafe { history_value.row_mut([destination, head, 0]) });
        }
    }
}

/// What a sink logit `sink` adds to the denominator of a head's softmax whose
/// running maximum is `maximum` (`exp(sink - maximum)`, the portable
/// `attention_rows`): infinite for a head that read no key, whose output
/// stays zero.
pub fn sink_weight(sink: f32, maximum: f32) -> f32 {
    if maximum == f32::NEG_INFINITY {
        f32::INFINITY
    } else {
        math::exp(sink - maximum)
    }
}

/// One query head's output stored to `A`: `accumulator / max(denominator,
/// 1e-30)`, column c times its gate c % U (sigmoid, or softplus when
/// `softplus`); no gates pass the row through.
pub fn gate_row<A: Dense>(
    accumulator: &[f32],
    denominator: f32,
    gates: &[A::Storage],
    softplus: bool,
    out: &mut [A::Storage],
) {
    let denominator = denominator.max(1e-30);
    let u = gates.len();
    for (column, (target, value)) in out.iter_mut().zip(accumulator).enumerate() {
        let attended = value / denominator;
        *target = A::narrow(if u == 0 {
            attended
        } else {
            let gate = A::widen(gates[column % u]);
            if softplus {
                attended * math::softplus(gate)
            } else {
                attended / (1.0 + math::exp(-gate))
            }
        });
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
/// `range` whose keys `selected` reads, positions counted along the
/// sequence.
pub fn walk(
    spans: impl Iterator<Item = (i32, i32)>,
    fresh: (i32, i32),
    range: Range<usize>,
    selected: impl Fn(usize) -> bool,
    mut run: impl FnMut(Keys, usize, usize),
) {
    let mut position = 0usize;
    let mut visit = |keys: Keys, lo: i32, hi: i32| {
        let count = (hi - lo) as usize;
        let mut start = range.start.max(position);
        let end = range.end.min(position + count);
        while start < end {
            if !selected(start) {
                start += 1;
                continue;
            }
            let mut stop = start + 1;
            while stop < end && selected(stop) {
                stop += 1;
            }
            run(keys, lo as usize + (start - position), stop - start);
            start = stop;
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
        let mut first = 0;
        while first < count {
            let n = BLOCK.min(count - first);
            for j in 0..n {
                load_key(first + j, row);
                score(self.maximum.len(), queries, scale, row, scores, j);
            }
            self.absorb_scored(scores, n, row, |j, out| load_value(first + j, out));
            first += n;
        }
    }

    /// Absorbs a block of `n` keys whose scores are in `scores` (G * BLOCK,
    /// key j of head h at h * BLOCK + j): the online-softmax update, then
    /// value j (`load_value(j, out)`, F32 into `row`) into each head's
    /// output in key order.
    #[inline(always)]
    pub fn absorb_scored(
        &mut self,
        scores: &mut [f32],
        n: usize,
        row: &mut [f32],
        mut load_value: impl FnMut(usize, &mut [f32]),
    ) {
        let g = self.maximum.len();
        let w = row.len();
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
            load_value(j, row);
            for head in 0..g {
                let weight = scores[head * BLOCK + j];
                for (value, v) in self.accumulator[head * w..][..w].iter_mut().zip(row.iter()) {
                    *value = weight.mul_add(*v, *value);
                }
            }
        }
    }
}

/// Key `j`'s scores against the `g` `queries` [G][W] (`row` holding it),
/// scaled, into `scores` at head * BLOCK + j.
#[inline(always)]
fn score(g: usize, queries: &[f32], scale: f32, row: &[f32], scores: &mut [f32], j: usize) {
    let w = row.len();
    for head in 0..g {
        scores[head * BLOCK + j] = reduce::dot(&queries[head * w..][..w], row) * scale;
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
                affine_decode(codes, coefficients, (32 * codes.len() / out.len()) as u32, out)
            },
            |j, out| {
                let (codes, coefficients) = value(j);
                affine_decode(codes, coefficients, (32 * codes.len() / out.len()) as u32, out)
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
                    super::affine_decode(codes, coefficients, (32 * codes.len() / out.len()) as u32, out)
                },
                |j, out| {
                    let (codes, coefficients) = value(j);
                    super::affine_decode(codes, coefficients, (32 * codes.len() / out.len()) as u32, out)
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

/// Absorbs the part `range` of a row's key sequence (its admitted history
/// `spans`, then its `fresh` span) into `state`: history rows through
/// `history_key(token, out)` / `history_value(token, out)`, fresh rows through
/// `fresh_key(token, out)` / `fresh_value(token, out)`. The keys `selected`
/// reads absorb in blocks of BLOCK consecutive ones from the range's start,
/// history and fresh alike: a key's block, and so the row's reduction order,
/// depends on its position alone, never on which keys are still fresh (how
/// many rows share the launch) or where its spans fall. `scores` holds G *
/// BLOCK values, `row` W.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn attend_rows(
    state: &mut Online<'_>,
    queries: &[f32],
    scale: f32,
    scores: &mut [f32],
    row: &mut [f32],
    spans: impl Iterator<Item = (i32, i32)>,
    fresh: (i32, i32),
    range: Range<usize>,
    selected: impl Fn(usize) -> bool,
    mut history_key: impl FnMut(usize, &mut [f32]),
    mut history_value: impl FnMut(usize, &mut [f32]),
    mut fresh_key: impl FnMut(usize, &mut [f32]),
    mut fresh_value: impl FnMut(usize, &mut [f32]),
) {
    // The block's keys so far: where each lives.
    let mut pending = [(Keys::History, 0usize); BLOCK];
    let mut n = 0;
    walk(spans, fresh, range, selected, |keys, first, count| {
        for token in first..first + count {
            match keys {
                Keys::History => history_key(token, row),
                Keys::Fresh => fresh_key(token, row),
            }
            score(state.maximum.len(), queries, scale, row, scores, n);
            pending[n] = (keys, token);
            n += 1;
            if n == BLOCK {
                state.absorb_scored(scores, n, row, |j, out| match pending[j] {
                    (Keys::History, token) => history_value(token, out),
                    (Keys::Fresh, token) => fresh_value(token, out),
                });
                n = 0;
            }
        }
    });
    if n > 0 {
        state.absorb_scored(scores, n, row, |j, out| match pending[j] {
            (Keys::History, token) => history_value(token, out),
            (Keys::Fresh, token) => fresh_value(token, out),
        });
    }
}

/// Absorbs a row's index-list picks (`selected_rows`, -1 for none), each
/// row `load(pick, out)` of the source's one plane read as key and value,
/// into `state` as [`attend_rows`] absorbs its keys (the portable
/// `attention_rows`' index-list source).
#[inline(always)]
pub fn attend_picks(
    state: &mut Online<'_>,
    queries: &[f32],
    scale: f32,
    scores: &mut [f32],
    row: &mut [f32],
    picks: &[i32],
    load: impl Fn(usize, &mut [f32]),
) {
    let mut first = 0;
    while first < picks.len() {
        if picks[first] < 0 {
            first += 1;
            continue;
        }
        let end = picks[first..].iter().position(|pick| *pick < 0).map_or(picks.len(), |run| first + run);
        let run = &picks[first..end];
        // A pick is its own value.
        state.absorb(
            queries,
            scale,
            run.len(),
            scores,
            row,
            |j, out| load(run[j] as usize, out),
            |j, out| load(run[j] as usize, out),
        );
        first = end;
    }
}

/// Whether a row's WorkList `bitmap` (one bit per block of `block_rows`
/// keys, word block / 32) reads key `ordinal`; a block past the bitmap is
/// not read, and an empty bitmap reads every key (the portable `selected`
/// under `B > 0`, and the Visible form).
#[inline(always)]
pub fn selected(bitmap: &[u32], block_rows: i32, ordinal: usize) -> bool {
    if bitmap.is_empty() {
        return true;
    }
    if block_rows <= 0 {
        return false;
    }
    let block = ordinal / block_rows as usize;
    bitmap.get(block / 32).is_some_and(|word| word >> (block % 32) & 1 != 0)
}

/// [`attend_rows`] with affine K8/V4 history rows `history_key(token)` and
/// `history_value(token)`, absorbed by [`Online::absorb_affine`].
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn attend_affine_rows<'h>(
    state: &mut Online<'_>,
    queries: &[f32],
    scale: f32,
    scores: &mut [f32],
    row: &mut [f32],
    spans: impl Iterator<Item = (i32, i32)>,
    fresh: (i32, i32),
    range: Range<usize>,
    history_key: impl Fn(usize) -> AffineRow<'h>,
    history_value: impl Fn(usize) -> AffineRow<'h>,
    mut fresh_key: impl FnMut(usize, &mut [f32]),
    mut fresh_value: impl FnMut(usize, &mut [f32]),
) {
    walk(spans, fresh, range, |_| true, |keys, first, count| match keys {
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
            |j, out| fresh_key(first + j, out),
            |j, out| fresh_value(first + j, out),
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

pub use super::history::{affine_encode, affine_decode};
