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
use seismic::cpu::{math, slab::{SlabTensor, SlabWalk}, Dense, Element, Isa, F16, F32, U32};
use std::{marker::PhantomData, ops::Range};

/// How many keys ahead of its scoring a key's rows are prefetched: a key
/// takes a few hundred nanoseconds on a vector tier, so two keys cover a
/// memory access, and more would only evict.
const PREFETCH_KEYS: usize = 2;

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
    #[inline(always)]
    pub fn new(kv: u64, g: u64, p: u64, s: u64) -> Self {
        Self { kv: kv as usize, g: g as usize, p: p as usize, w: (2 * p + s) as usize }
    }

    /// The geometry of an affine K8/V4 history entry, whose vectors are
    /// whole codec groups (the portable codec leaves a partial group's
    /// columns undefined); the declarations' `where` admits only those.
    #[inline(always)]
    pub fn affine(self) -> Self {
        assert!(self.w % GROUP == 0, "affine history needs head widths of whole {GROUP}-column groups, not {}", self.w);
        self
    }

    /// Query heads of a row.
    #[inline(always)]
    pub fn queries(&self) -> usize {
        self.kv * self.g
    }
}

/// The cosines and sines of a row's P rotary pairs: pair p by the row's
/// coordinate on axis `components[p]` at `frequencies[p]`.
#[inline(always)]
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

    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
    pub fn with_key_rounding(self, kr: u64, block: i32, element: i32, scale: i32, amax_floor: f32) -> Self {
        let rounding = Rounding::of(block, element, scale, amax_floor);
        assert_eq!(kr == 1, rounding.is_some(), "KR {kr} beside key rounding block {block}");
        Self { key_rounding: rounding, ..self }
    }

    /// Whether the row takes a value of its own: a fresh layer whose values
    /// are not its stored keys.
    #[inline(always)]
    pub fn own_values(self) -> bool {
        self.fresh && !self.stored_key
    }

    /// The form over affine K8/V4 history ([`Heads::affine`]).
    #[inline(always)]
    pub fn affine(self) -> Self {
        Self { heads: self.heads.affine(), ..self }
    }
}

/// [`angles`] with each pair's cosine and sine scaled by its amplitude.
#[inline(always)]
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
#[inline(always)]
pub fn prepare_head<A: Dense>(
    x: &[A::Storage],
    norm: Option<&[f32]>,
    layout: Layout,
    cosines: &[f32],
    sines: &[f32],
    epsilon: f32,
    out: &mut [f32],
) {
    prepare_rounded_head::<A>(x, norm, layout, cosines, sines, epsilon, None, None::<fn(f32) -> f32>, out);
}

/// `values` as `hold` holds them (each value through it), or as they are
/// without one: a rounded row's hold in its activation element
/// (`activation::publish`), the portable `held_in_activation`.
#[inline(always)]
pub fn held(hold: Option<impl Fn(f32) -> f32>, values: &mut [f32]) {
    if let Some(hold) = hold {
        values.iter_mut().for_each(|value| *value = hold(*value));
    }
}

/// [`prepare_head`] with the rotated row rounded by `rounding` in F32
/// before it is published (the portable `block_round` of `head_rotary`),
/// and, with `hold`, every value held by it after the norm (when there is
/// one) and after the rotation, before the rounding (a stored row's BF16
/// holds, `activation::publish` of the activation element).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn prepare_rounded_head<A: Dense>(
    x: &[A::Storage],
    norm: Option<&[f32]>,
    layout: Layout,
    cosines: &[f32],
    sines: &[f32],
    epsilon: f32,
    rounding: Option<Rounding>,
    hold: Option<impl Fn(f32) -> f32 + Copy>,
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
#[inline(always)]
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
#[inline(always)]
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

/// The `count` query heads from `first` of one row, prepared (the first W
/// columns of each `query(head)`) into `queries`, W floats a head.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn prepare_queries<'q, A: Dense>(
    form: Form,
    first: usize,
    count: usize,
    query: impl Fn(usize) -> &'q [A::Storage],
    query_norm: Option<&[f32]>,
    cosines: &[f32],
    sines: &[f32],
    epsilon: f32,
    queries: &mut [f32],
) where
    A::Storage: 'q,
{
    let w = form.heads.w;
    for local in 0..count {
        prepare_head::<A>(&query(first + local)[..w], query_norm, form.layout, cosines, sines, epsilon, &mut queries[local * w..][..w]);
    }
}

/// One kv head of a layer with fresh rows, prepared from the head's `key` and
/// `value` columns (W each; `value` is empty for stored-key values) into
/// `keys` and `values` (W floats each): the key rounded and rotated, the
/// value `head_norm`ed with `value_norm` when present, or the prepared key
/// for stored-key values.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn prepare_kv_head<A: Dense>(
    form: Form,
    key: &[A::Storage],
    value: &[A::Storage],
    key_norm: Option<&[f32]>,
    value_norm: Option<&[f32]>,
    cosines: &[f32],
    sines: &[f32],
    epsilon: f32,
    keys: &mut [f32],
    values: &mut [f32],
) {
    let w = form.heads.w;
    prepare_rounded_head::<A>(
        &key[..w],
        key_norm,
        form.layout,
        cosines,
        sines,
        epsilon,
        form.key_rounding,
        form.key_rounding.map(|_| activation::publish::<A>),
        &mut keys[..w],
    );
    if form.stored_key {
        values[..w].copy_from_slice(&keys[..w]);
    } else {
        prepare_head::<A>(&value[..w], value_norm, form.layout, &[], &[], epsilon, &mut values[..w]);
    }
}

/// What one decode work item (kv head `kv_head` of row `row`) prepares itself
/// before it attends, so no launch precedes the attend: the row's G query
/// heads of the kv head into `queries` (W floats a head), and for a layer with
/// fresh rows the kv head's key and value of each row it reads (`fresh`, the
/// row's fresh span) into `keys` and `values` at row r's W floats, plus the
/// row's own when `own_row` (its history append). `angles(r, cosines, sines)`
/// writes row r's amplitude-scaled angles; `key(r)` and `value(r)` are batch
/// row r's key and value rows over all kv heads (`value` empty for stored-key
/// values). Each head is prepared by the arithmetic of [`prepare_form_row`].
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn prepare_group<'q, A: Dense>(
    form: Form,
    kv_head: usize,
    row: usize,
    own_row: bool,
    fresh: Range<i32>,
    mut angles: impl FnMut(usize, &mut [f32], &mut [f32]),
    query: impl Fn(usize) -> &'q [A::Storage],
    key: impl Fn(usize) -> &'q [A::Storage],
    value: impl Fn(usize) -> &'q [A::Storage],
    norms: (Option<&[f32]>, Option<&[f32]>, Option<&[f32]>),
    epsilon: f32,
    (cosines, sines): (&mut [f32], &mut [f32]),
    queries: &mut [f32],
    keys: &mut [f32],
    values: &mut [f32],
) where
    A::Storage: 'q,
{
    let (g, w) = (form.heads.g, form.heads.w);
    let (query_norm, key_norm, value_norm) = norms;
    angles(row, cosines, sines);
    prepare_queries::<A>(form, kv_head * g, g, query, query_norm, cosines, sines, epsilon, queries);
    if !form.fresh {
        return;
    }
    // The row's own angles are held; its rows come first so they are not
    // computed twice.
    let own = own_row || fresh.contains(&(row as i32));
    let others = fresh.filter(move |&r| !(own && r as usize == row)).map(|r| r as usize);
    let mut held = row;
    for r in own.then_some(row).into_iter().chain(others) {
        if r != held {
            angles(r, cosines, sines);
            held = r;
        }
        let value = if form.stored_key { &[][..] } else { &value(r)[kv_head * w..] };
        prepare_kv_head::<A>(
            form,
            &key(r)[kv_head * w..],
            value,
            key_norm,
            value_norm,
            cosines,
            sines,
            epsilon,
            &mut keys[r * w..],
            &mut values[r * w..],
        );
    }
}

/// Prepares one row of an `attention_*` entry: its KV * G queries (the first
/// W columns of each `query(head)`) into `queries`, and when the layer has
/// fresh rows its KV keys into `keys` and its KV values (`head_norm`ed with
/// `value_norm` when present, or the prepared keys for stored-key values,
/// whose `value` is empty) into `values`, all F32 values of the activation
/// element.
#[inline(always)]
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
    prepare_queries::<A>(form, 0, form.heads.queries(), query, query_norm, cosines, sines, epsilon, queries);
    if !form.fresh {
        return;
    }
    for head in 0..form.heads.kv {
        let value = if form.stored_key { &[][..] } else { &value[head * w..] };
        prepare_kv_head::<A>(
            form,
            &key[head * w..],
            value,
            key_norm,
            value_norm,
            cosines,
            sines,
            epsilon,
            &mut keys[head * w..],
            &mut values[head * w..],
        );
    }
}

/// Appends one kv head's prepared key and value (`key` and `value`, W floats
/// each) at dense history row `destination`: the key plane alone for
/// stored-key values, whose value plane has no rows.
///
/// # Safety
/// `destination` is a freshly reserved row no other work item of the call
/// reads or writes.
#[inline(always)]
pub unsafe fn store_dense_head<A: Dense>(
    form: Form,
    key: &[f32],
    value: &[f32],
    history_key: &SlabTensor<'_, A, 3>,
    history_value: &SlabTensor<'_, A, 3>,
    destination: usize,
    head: usize,
) {
    // SAFETY: the caller's.
    activation::store::<A>(key, unsafe { history_key.row_mut([destination, head, 0]) });
    if !form.stored_key {
        // SAFETY: the caller's.
        activation::store::<A>(value, unsafe { history_value.row_mut([destination, head, 0]) });
    }
}

/// Appends one row's prepared keys and values ([`prepare_form_row`]) at
/// dense history row `destination`, each kv head by [`store_dense_head`].
///
/// # Safety
/// As [`store_dense_head`].
#[inline(always)]
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
        unsafe {
            store_dense_head::<A>(
                form,
                &keys[head * w..][..w],
                if form.stored_key { &[][..] } else { &values[head * w..][..w] },
                history_key,
                history_value,
                destination,
                head,
            );
        }
    }
}

/// What a sink logit `sink` adds to the denominator of a head's softmax whose
/// running maximum is `maximum` (`exp(sink - maximum)`, the portable
/// `attention_rows`): infinite for a head that read no key, whose output
/// stays zero.
#[inline(always)]
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
#[inline(always)]
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
#[inline(always)]
pub fn total(spans: impl Iterator<Item = (i32, i32)>, fresh: (i32, i32)) -> usize {
    spans.map(|(lo, hi)| (hi - lo) as usize).sum::<usize>() + (fresh.1 - fresh.0).max(0) as usize
}

/// The runs of a row's key sequence (its admitted history spans in order,
/// then its fresh span) inside `range` whose keys `selected` reads, each
/// `(keys, first, count)` with positions counted along the sequence. An
/// iterator rather than a callback, so the absorb a caller runs per run
/// inlines into its launch's tier boundary (a closure would be compiled
/// out of line, without the tier's features).
#[inline(always)]
pub fn walk<S: Iterator<Item = (i32, i32)>, F: Fn(usize) -> bool>(
    spans: S,
    fresh: (i32, i32),
    range: Range<usize>,
    selected: F,
) -> Runs<S, F> {
    Runs { spans, fresh: (fresh.1 > fresh.0).then_some(fresh), range, selected, position: 0, span: None }
}

/// The iterator of [`walk`].
pub struct Runs<S, F> {
    spans: S,
    fresh: Option<(i32, i32)>,
    range: Range<usize>,
    selected: F,
    /// Sequence positions before the current span.
    position: usize,
    /// The current span: its keys, first row, base position, and the
    /// positions [start, end) of it still to walk.
    span: Option<(Keys, i32, usize, usize, usize)>,
}

impl<S: Iterator<Item = (i32, i32)>, F: Fn(usize) -> bool> Iterator for Runs<S, F> {
    type Item = (Keys, usize, usize);

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some((keys, lo, base, start, end)) = &mut self.span {
                while *start < *end && !(self.selected)(*start) {
                    *start += 1;
                }
                if *start < *end {
                    let mut stop = *start + 1;
                    while stop < *end && (self.selected)(stop) {
                        stop += 1;
                    }
                    let run = (*keys, *lo as usize + (*start - *base), stop - *start);
                    *start = stop;
                    return Some(run);
                }
                self.span = None;
            }
            let (keys, (lo, hi)) = match self.spans.next() {
                Some(span) => (Keys::History, span),
                None => (Keys::Fresh, self.fresh.take()?),
            };
            let (base, count) = (self.position, (hi - lo) as usize);
            self.position += count;
            let (start, end) = (self.range.start.max(base), self.range.end.min(base + count));
            self.span = Some((keys, lo, base, start, end));
        }
    }
}

/// The keys of decode partition `partition` of a row seeing `total` keys:
/// consecutive partitions of max(PARTITION_KEYS, ceil(total / parts)) keys.
#[inline(always)]
pub fn partition(total: usize, parts: usize, partition: usize) -> Range<usize> {
    let keys = PARTITION_KEYS.max(total.div_ceil(parts));
    (partition * keys).min(total)..((partition + 1) * keys).min(total)
}

/// Runs `$body` with `$tile` the query heads of the next tile, at most eight
/// of the `$left` still to go, so a tile's lanes are registers.
macro_rules! tile {
    ($left:expr, $tile:ident => $body:expr) => {
        match $left {
            1 => { const $tile: usize = 1; $body }
            2 => { const $tile: usize = 2; $body }
            3 => { const $tile: usize = 3; $body }
            4 => { const $tile: usize = 4; $body }
            5 => { const $tile: usize = 5; $body }
            6 => { const $tile: usize = 6; $body }
            7 => { const $tile: usize = 7; $body }
            _ => { const $tile: usize = 8; $body }
        }
    };
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
    #[inline(always)]
    pub fn reset(&mut self) {
        self.maximum.fill(f32::NEG_INFINITY);
        self.denominator.fill(0.0);
        self.accumulator.fill(0.0);
    }

    /// Scores key `row` (F32, its W columns) against the G `queries`
    /// [G][W], scaled, into column `j` of the scores.
    #[inline(always)]
    fn score_row(&self, queries: &[f32], scale: f32, row: &[f32], scores: &mut [f32], j: usize) {
        let w = row.len();
        for head in 0..self.maximum.len() {
            scores[head * BLOCK + j] = reduce::dot(&queries[head * w..][..w], row) * scale;
        }
    }

    /// Absorbs value `row` (F32) into each head's output, weighted by its
    /// weight in column `j` of the scores.
    #[inline(always)]
    fn absorb_value(&mut self, scores: &[f32], row: &[f32], j: usize) {
        let w = row.len();
        for head in 0..self.maximum.len() {
            let weight = scores[head * BLOCK + j];
            for (value, v) in self.accumulator[head * w..][..w].iter_mut().zip(row.iter()) {
                *value = weight.mul_add(*v, *value);
            }
        }
    }

    /// The online-softmax step of a block of `n` scored keys: each head's
    /// maximum grows to cover its scores, which become their weights
    /// exp(score - maximum), and its denominator and output carry to the new
    /// maximum, the denominator taking the weights' sum.
    #[inline(always)]
    fn rescale(&mut self, scores: &mut [f32], n: usize) {
        let w = self.accumulator.len() / self.maximum.len();
        for head in 0..self.maximum.len() {
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
    }

    /// The portable body's part of a run: its `count` keys from row `first`
    /// of `key` (each F32 through `row`) scored into `block` as `keys`,
    /// which absorbs whenever it fills ([`Online::flush_portable`]).
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn take_portable<K: Rows, H: Rows, F: Rows>(
        &mut self,
        queries: &[f32],
        scale: f32,
        first: usize,
        count: usize,
        scores: &mut [f32],
        row: &mut [f32],
        keys: Keys,
        key: &K,
        block: &mut Block<usize>,
        values: (&H, &F),
    ) {
        for index in first..first + count {
            key.load(index, row);
            self.score_row(queries, scale, row, scores, block.n);
            block.push(keys, index);
            if block.n == BLOCK {
                self.flush_portable(scores, row, block, values);
            }
        }
    }

    /// Absorbs the portable body's `block`: the online-softmax step, then
    /// each key's value (from `history` or `fresh`, F32 through `row`) in
    /// key order.
    #[inline(always)]
    fn flush_portable<H: Rows, F: Rows>(
        &mut self,
        scores: &mut [f32],
        row: &mut [f32],
        block: &mut Block<usize>,
        (history, fresh): (&H, &F),
    ) {
        self.rescale(scores, block.n);
        for j in 0..block.n {
            if j < block.split {
                history.load(block.values[j], row);
            } else {
                fresh.load(block.values[j], row);
            }
            self.absorb_value(scores, row, j);
        }
        block.clear();
    }

    /// The vector body's part of a run: [`Online::take_portable`]'s result
    /// bits with no row through memory. A score is `reduce::dot`'s eight
    /// lanes (column c fused into lane c % 8, in column order) combined by
    /// its tree, for up to eight heads per decoded key chunk; each key's
    /// value row is resolved as it is scored (its widened coefficients at
    /// `per` floats a key in `value_coefficients`), and absorbed by
    /// [`Online::flush_vectors`] when the block fills, with value rows `H`
    /// for its history keys and `F` for its fresh ones.
    ///
    /// # Safety
    /// The tier runs [`F8`], and the head width is whole 8-column chunks.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn take_vectors<K: Rows, R: Rows, H: Rows, F: Rows>(
        &mut self,
        queries: &[f32],
        scale: f32,
        first: usize,
        count: usize,
        scores: &mut [f32],
        (key_coefficients, value_coefficients, per): (&mut [f32], &mut [f32], usize),
        keys: Keys,
        (key, value): (&K, &R),
        block: &mut Block<*const u8>,
    ) {
        let (g, w) = (self.maximum.len(), self.accumulator.len() / self.maximum.len());
        let (per_key, per_value) = (key.coefficients(w), value.coefficients(w));
        let (mut key_walk, mut value_walk) = (key.walk(first), value.walk(first));
        // Each key's rows, and its value's, are asked for PREFETCH_KEYS keys
        // before it is scored and its value row resolved.
        let (mut key_ahead, mut value_ahead) = (key.walk(first), value.walk(first));
        for _ in 0..PREFETCH_KEYS.min(count) {
            key.prefetch(&mut key_ahead, w);
            value.prefetch(&mut value_ahead, w);
        }
        for ahead in PREFETCH_KEYS..PREFETCH_KEYS + count {
            if ahead < count {
                key.prefetch(&mut key_ahead, w);
                value.prefetch(&mut value_ahead, w);
            }
            // SAFETY: the caller's tier.
            let row = unsafe { key.next(&mut key_walk, w, &mut key_coefficients[..per_key]) };
            let mut head = 0;
            while head < g {
                // SAFETY: the caller's tier; `row` holds w columns.
                tile!(g - head, T => unsafe {
                    score::<K, T>(queries, scale, scores, row, key_coefficients, head, block.n, w);
                    head += T;
                });
            }
            // SAFETY: the caller's tier.
            let value_row = unsafe { value.next(&mut value_walk, w, &mut value_coefficients[block.n * per..][..per_value]) };
            block.push(keys, value_row);
            if block.n == BLOCK {
                // SAFETY: the caller's tier.
                unsafe { self.flush_vectors::<H, F>(scores, value_coefficients, per, block) };
            }
        }
    }

    /// Absorbs the vector body's `block`: the online-softmax step, then its
    /// value rows into every accumulator element in key order, eight columns
    /// of up to eight heads held in registers across the block (its history
    /// rows, of `H`, then its fresh rows, of `F`, the same updates in order).
    ///
    /// # Safety
    /// The tier runs [`F8`]; every row holds w columns, its coefficients at
    /// `per` floats a key in `coefficients`.
    #[inline(always)]
    unsafe fn flush_vectors<H: Rows, F: Rows>(
        &mut self,
        scores: &mut [f32],
        coefficients: &[f32],
        per: usize,
        block: &mut Block<*const u8>,
    ) {
        self.rescale(scores, block.n);
        let (g, w) = (self.maximum.len(), self.accumulator.len() / self.maximum.len());
        let (split, n) = (block.split, block.n);
        let mut head = 0;
        while head < g {
            // SAFETY: the caller's tier; every row holds w columns, and the
            // accumulator G * W.
            tile!(g - head, T => unsafe {
                values::<H, T>(self.accumulator, scores, &block.values, 0..split, coefficients, per, head, w);
                values::<F, T>(self.accumulator, scores, &block.values, split..n, coefficients, per, head, w);
                head += T;
            });
        }
        block.clear();
    }

}


/// The keys of a block scored but not yet absorbed: `n` of them, the
/// first `split` history keys and the rest fresh (a row's history keys
/// precede its fresh ones), each with its value `V`: its row's storage (the
/// vector body) or index (the portable body).
struct Block<V> {
    n: usize,
    split: usize,
    values: [V; BLOCK],
}

impl<V: Copy> Block<V> {
    #[inline(always)]
    fn new(empty: V) -> Self {
        Self { n: 0, split: 0, values: [empty; BLOCK] }
    }

    /// Appends a key of `keys` with its `value`.
    #[inline(always)]
    fn push(&mut self, keys: Keys, value: V) {
        self.values[self.n] = value;
        self.n += 1;
        if keys == Keys::History {
            assert_eq!(self.split + 1, self.n, "a row's history keys precede its fresh ones");
            self.split = self.n;
        }
    }

    #[inline(always)]
    fn clear(&mut self) {
        (self.n, self.split) = (0, 0);
    }
}

/// Scores key `row` (with its widened `coefficients`) against the T query
/// heads from `head`: `reduce::dot` lane by lane, into column `j` of their
/// scores.
///
/// # Safety
/// The tier runs [`F8`]; `row` holds w columns, `queries` (head + T) * w.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn score<K: Rows, const T: usize>(
    queries: &[f32],
    scale: f32,
    scores: &mut [f32],
    row: *const u8,
    coefficients: &[f32],
    head: usize,
    j: usize,
    w: usize,
) {
    unsafe {
        let mut lanes = [F8::zero(); T];
        let chunks = w / 8;
        for from in (0..chunks).step_by(K::GROUP_CHUNKS) {
            let scale = K::scale(coefficients, from / K::GROUP_CHUNKS);
            for c in from..chunks.min(from.saturating_add(K::GROUP_CHUNKS)) {
                let key = K::chunk(row, scale, c);
                for (t, lane) in lanes.iter_mut().enumerate() {
                    *lane = lane.fma(F8::load(queries.as_ptr().add((head + t) * w + 8 * c)), key);
                }
            }
        }
        for (t, lane) in lanes.into_iter().enumerate() {
            scores[(head + t) * BLOCK + j] = lane.sum() * scale;
        }
    }
}

/// Absorbs the block's value `rows` `keys` (their widened `coefficients`,
/// `per` a row) into the output of the T query heads from `head`, each
/// weighted by its score: eight columns at a time, fused in key order.
///
/// # Safety
/// The tier runs [`F8`]; every row holds w columns, `accumulator`
/// (head + T) * w.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn values<R: Rows, const T: usize>(
    accumulator: &mut [f32],
    scores: &[f32],
    rows: &[*const u8],
    keys: Range<usize>,
    coefficients: &[f32],
    per: usize,
    head: usize,
    w: usize,
) {
    if keys.is_empty() {
        return;
    }
    let output = accumulator.as_mut_ptr();
    for c in 0..w / 8 {
        unsafe {
            let mut lanes = [F8::zero(); T];
            for (t, lane) in lanes.iter_mut().enumerate() {
                *lane = F8::load(output.add((head + t) * w + 8 * c));
            }
            for j in keys.clone() {
                let value = R::chunk(rows[j], R::scale(&coefficients[j * per..], c / R::GROUP_CHUNKS), c);
                for (t, lane) in lanes.iter_mut().enumerate() {
                    *lane = lane.fma(F8::splat(scores[(head + t) * BLOCK + j]), value);
                }
            }
            for (t, lane) in lanes.into_iter().enumerate() {
                lane.store(output.add((head + t) * w + 8 * c));
            }
        }
    }
}

/// Whether tier `L` runs [`F8`]: NEON, or an x86 tier of 256-bit vectors
/// (AVX2, FMA and F16C).
#[inline(always)]
fn vectors<L: Isa>() -> bool {
    cfg!(target_arch = "aarch64") || L::TIER.vector_bytes() >= 32
}

/// Floats of an absorb's `work` at head width `w` over `affine` history or
/// not: a row for the portable body; a key's and a block of values'
/// widened coefficients (w / 16 + BLOCK * w / 16) for the vector body over
/// affine rows.
#[inline(always)]
pub const fn work(w: usize, affine: bool) -> usize {
    if affine {
        3 * w
    } else {
        w
    }
}

/// The rows an absorb reads, row `index` of a run: as F32 (`load`, the
/// portable body), or on [`F8`] tiers eight columns at a time from storage
/// (`chunk`), each value exactly as `load` gives it.
pub trait Rows {
    /// The widened coefficients a row of `w` columns carries.
    fn coefficients(&self, w: usize) -> usize;

    /// Row `index` as F32 in `out` (its W columns).
    fn load(&self, index: usize, out: &mut [f32]);

    /// A walk over consecutive rows ([`Rows::next`]).
    type Walk;

    /// The walk from row `first`.
    fn walk(&self, first: usize) -> Self::Walk;

    /// The storage of `walk`'s row of `w` columns, its coefficients widened
    /// into `coefficients`; the walk steps to the next row.
    ///
    /// # Safety
    /// The tier runs [`F8`].
    unsafe fn next(&self, walk: &mut Self::Walk, w: usize, coefficients: &mut [f32]) -> *const u8;

    /// Asks for the cache lines of `walk`'s row of `w` columns ahead of
    /// its [`Rows::next`]; the walk steps to the next row.
    fn prefetch(&self, walk: &mut Self::Walk, w: usize);

    /// The 8-column chunks one [`Rows::Scale`] covers.
    const GROUP_CHUNKS: usize;

    /// What [`Rows::chunk`] takes of a row's widened `coefficients` for
    /// the chunks of group `group`, formed once a group.
    type Scale: Copy;

    /// # Safety
    /// The tier runs [`F8`]; `coefficients` are a [`Rows::next`] row's, of
    /// more than `group` groups.
    unsafe fn scale(coefficients: &[f32], group: usize) -> Self::Scale;

    /// Columns 8c..8c + 8 of a `row` with the `scale` of their group.
    ///
    /// # Safety
    /// The tier runs [`F8`]; `row` is a [`Rows::next`] row of more than 8c
    /// columns, `scale` its group's.
    unsafe fn chunk(row: *const u8, scale: Self::Scale, c: usize) -> F8;
}

/// Rows of element `E`, row `index` at `rows(index)`.
pub struct DenseRows<E, F>(F, PhantomData<E>);

impl<E, F> DenseRows<E, F> {
    #[inline(always)]
    pub fn new(rows: F) -> Self {
        Self(rows, PhantomData)
    }
}

impl<'a, E: Dense, F: Fn(usize) -> &'a [E::Storage]> Rows for DenseRows<E, F>
where
    E::Storage: 'a,
{
    #[inline(always)]
    fn coefficients(&self, _w: usize) -> usize {
        0
    }

    #[inline(always)]
    fn load(&self, index: usize, out: &mut [f32]) {
        activation::widen::<E>(&(self.0)(index)[..out.len()], out);
    }

    type Walk = usize;

    #[inline(always)]
    fn walk(&self, first: usize) -> usize {
        first
    }

    #[inline(always)]
    unsafe fn next(&self, walk: &mut usize, w: usize, _coefficients: &mut [f32]) -> *const u8 {
        let row = (self.0)(*walk);
        assert!(row.len() >= w);
        *walk += 1;
        row.as_ptr().cast()
    }

    #[inline(always)]
    fn prefetch(&self, walk: &mut usize, w: usize) {
        lines(&(self.0)(*walk)[..w]);
        *walk += 1;
    }

    /// Dense rows carry no coefficients: one empty scale covers them.
    const GROUP_CHUNKS: usize = usize::MAX;

    type Scale = ();

    #[inline(always)]
    unsafe fn scale(_coefficients: &[f32], _group: usize) {}

    #[inline(always)]
    unsafe fn chunk(row: *const u8, _scale: (), c: usize) -> F8 {
        unsafe { F8::widen::<E>(row.cast::<E::Storage>().add(8 * c)) }
    }
}

/// The rows of kv head `head` in slab-stored history `plane`.
#[inline(always)]
pub fn history_rows<'a, A: Dense>(
    plane: &SlabTensor<'a, A, 3>,
    head: usize,
) -> DenseRows<A, impl Fn(usize) -> &'a [A::Storage]>
where
    A::Storage: 'a,
{
    let plane = *plane;
    DenseRows::new(move |token| plane.row([token, head, 0]))
}

/// The rows of kv head `head` in prepared batch rows `rows` ([M][KV][W]
/// F32).
#[inline(always)]
pub fn batch_rows<'a>(rows: &'a [f32], kv: usize, head: usize, w: usize) -> DenseRows<F32, impl Fn(usize) -> &'a [f32]> {
    DenseRows::new(move |token| &rows[(token * kv + head) * w..][..w])
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
/// encoded into affine history row `destination`: keys at `key_bits` bits,
/// values at `value_bits`.
///
/// # Safety
/// `destination` is a freshly reserved row no other work item of the call
/// reads or writes.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn store_affine_head<C: Element<Storage = u16>>(
    key: &[f32],
    value: &[f32],
    (key_bits, value_bits): (u32, u32),
    (key_codes, key_coefficients): (&SlabTensor<'_, U32, 3>, &SlabTensor<'_, C, 3>),
    (value_codes, value_coefficients): (&SlabTensor<'_, U32, 3>, &SlabTensor<'_, C, 3>),
    destination: usize,
    head: usize,
) {
    // SAFETY: the caller's.
    unsafe {
        affine_encode(
            key,
            key_bits,
            key_codes.row_mut([destination, head, 0]),
            key_coefficients.row_mut([destination, head, 0]),
        );
        affine_encode(
            value,
            value_bits,
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

/// Prefetches every cache line of `row`.
#[inline(always)]
fn lines<T>(row: &[T]) {
    let (at, bytes) = (row.as_ptr().cast::<u8>(), std::mem::size_of_val(row));
    for offset in (0..bytes).step_by(64) {
        prefetch(at.wrapping_add(offset));
    }
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

pub use lanes::{prefetch, F8};

/// Eight F32 lanes in one AVX2 register. Every operation needs a tier of
/// 256-bit vectors (`vectors`).
#[cfg(target_arch = "x86_64")]
mod lanes {
    use seismic::cpu::Dense;
    use std::arch::x86_64::*;

    /// Asks for the cache line holding `at` ahead of its use; never faults.
    #[inline(always)]
    pub fn prefetch(at: *const u8) {
        // SAFETY: a prefetch reads nothing architecturally.
        unsafe { _mm_prefetch::<_MM_HINT_T0>(at.cast()) }
    }

    #[derive(Clone, Copy)]
    pub struct F8(__m256);

    impl F8 {
        #[inline(always)]
        pub unsafe fn zero() -> Self {
            unsafe { Self(_mm256_setzero_ps()) }
        }

        #[inline(always)]
        pub unsafe fn splat(value: f32) -> Self {
            unsafe { Self(_mm256_set1_ps(value)) }
        }

        #[inline(always)]
        pub unsafe fn load(at: *const f32) -> Self {
            unsafe { Self(_mm256_loadu_ps(at)) }
        }

        #[inline(always)]
        pub unsafe fn store(self, at: *mut f32) {
            unsafe { _mm256_storeu_ps(at, self.0) }
        }

        /// `self + a * b` lane by lane, fused.
        #[inline(always)]
        pub unsafe fn fma(self, a: Self, b: Self) -> Self {
            unsafe { Self(_mm256_fmadd_ps(a.0, b.0, self.0)) }
        }

        /// The lanes combined by `reduce::combine`'s tree:
        /// ((l0 + l4) + (l2 + l6)) + ((l1 + l5) + (l3 + l7)).
        #[inline(always)]
        pub unsafe fn sum(self) -> f32 {
            unsafe {
                let pairs = _mm_add_ps(_mm256_castps256_ps128(self.0), _mm256_extractf128_ps::<1>(self.0));
                let halves = _mm_add_ps(pairs, _mm_movehl_ps(pairs, pairs));
                _mm_cvtss_f32(_mm_add_ss(halves, _mm_shuffle_ps::<1>(halves, halves)))
            }
        }

        /// Eight `E` values widened.
        #[inline(always)]
        pub unsafe fn widen<E: Dense>(at: *const E::Storage) -> Self {
            unsafe { Self(E::widen_x86(at)) }
        }

        /// The eight 8-bit codes at `at`.
        #[inline(always)]
        pub unsafe fn bytes(at: *const u8) -> Self {
            unsafe { Self(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(at.cast())))) }
        }

        /// The eight 4-bit codes of `word`, code k at bits 4k.
        #[inline(always)]
        pub unsafe fn nibbles(word: u32) -> Self {
            unsafe {
                let shifted = _mm256_srlv_epi32(_mm256_set1_epi32(word as i32), _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28));
                Self(_mm256_cvtepi32_ps(_mm256_and_si256(shifted, _mm256_set1_epi32(15))))
            }
        }
    }
}

/// Eight F32 lanes in two NEON registers, lanes 0..4 and 4..8.
#[cfg(target_arch = "aarch64")]
mod lanes {
    use seismic::cpu::Dense;
    use std::arch::aarch64::*;

    /// Asks for the cache line holding `at` ahead of its use; never faults.
    #[inline(always)]
    pub fn prefetch(at: *const u8) {
        // SAFETY: a prefetch reads nothing architecturally.
        unsafe { std::arch::asm!("prfm pldl1keep, [{at}]", at = in(reg) at, options(nostack, preserves_flags, readonly)) }
    }

    #[derive(Clone, Copy)]
    pub struct F8([float32x4_t; 2]);

    impl F8 {
        #[inline(always)]
        pub unsafe fn zero() -> Self {
            unsafe { Self::splat(0.0) }
        }

        #[inline(always)]
        pub unsafe fn splat(value: f32) -> Self {
            unsafe { Self([vdupq_n_f32(value); 2]) }
        }

        #[inline(always)]
        pub unsafe fn load(at: *const f32) -> Self {
            unsafe { Self([vld1q_f32(at), vld1q_f32(at.add(4))]) }
        }

        #[inline(always)]
        pub unsafe fn store(self, at: *mut f32) {
            unsafe {
                vst1q_f32(at, self.0[0]);
                vst1q_f32(at.add(4), self.0[1]);
            }
        }

        /// `self + a * b` lane by lane, fused.
        #[inline(always)]
        pub unsafe fn fma(self, a: Self, b: Self) -> Self {
            unsafe { Self([vfmaq_f32(self.0[0], a.0[0], b.0[0]), vfmaq_f32(self.0[1], a.0[1], b.0[1])]) }
        }

        /// The lanes combined by `reduce::combine`'s tree:
        /// ((l0 + l4) + (l2 + l6)) + ((l1 + l5) + (l3 + l7)).
        #[inline(always)]
        pub unsafe fn sum(self) -> f32 {
            unsafe {
                let pairs = vaddq_f32(self.0[0], self.0[1]);
                (vgetq_lane_f32::<0>(pairs) + vgetq_lane_f32::<2>(pairs))
                    + (vgetq_lane_f32::<1>(pairs) + vgetq_lane_f32::<3>(pairs))
            }
        }

        /// Eight `E` values widened.
        #[inline(always)]
        pub unsafe fn widen<E: Dense>(at: *const E::Storage) -> Self {
            unsafe { Self(E::widen_neon(at)) }
        }

        /// The eight 8-bit codes at `at`.
        #[inline(always)]
        pub unsafe fn bytes(at: *const u8) -> Self {
            unsafe { Self::codes(vld1_u8(at)) }
        }

        /// The eight 4-bit codes of `word`, code k at bits 4k.
        #[inline(always)]
        pub unsafe fn nibbles(word: u32) -> Self {
            unsafe {
                let bytes = vcreate_u8(u64::from(word));
                Self::codes(vzip1_u8(vand_u8(bytes, vdup_n_u8(15)), vshr_n_u8::<4>(bytes)))
            }
        }

        #[inline(always)]
        unsafe fn codes(codes: uint8x8_t) -> Self {
            unsafe {
                let wide = vmovl_u8(codes);
                Self([vcvtq_f32_u32(vmovl_u16(vget_low_u16(wide))), vcvtq_f32_u32(vmovl_high_u16(wide))])
            }
        }
    }
}

/// Absorbs the part `range` of a row's key sequence (its admitted history
/// spans, then its `fresh` span) whose keys `selected` reads into `state`
/// on tier `l`: history rows from `history` (keys, values), fresh rows
/// from `batch`. The keys absorb in blocks of BLOCK consecutive ones from
/// the range's start, history and fresh alike: a key's block, and so the
/// row's reduction order, depends on its position alone, never on which
/// keys are still fresh (how many rows share the launch) or where its spans
/// fall. On a tier with [`F8`] vectors and a head width of whole 8-column
/// chunks by the vector body, otherwise by the portable body over the rows'
/// `load`, with the same result bits. `scores` holds G * BLOCK values,
/// `work` [`work`] floats.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn attend<L: Isa, HK: Rows, HV: Rows, BK: Rows, BV: Rows>(
    _l: L,
    state: &mut Online<'_>,
    queries: &[f32],
    scale: f32,
    scores: &mut [f32],
    work: &mut [f32],
    spans: impl Iterator<Item = (i32, i32)>,
    fresh: (i32, i32),
    range: Range<usize>,
    selected: impl Fn(usize) -> bool,
    history: (&HK, &HV),
    batch: (&BK, &BV),
) {
    let (g, w) = (state.maximum.len(), state.accumulator.len() / state.maximum.len());
    assert!(queries.len() >= g * w && scores.len() >= g * BLOCK);
    if vectors::<L>() && w % 8 == 0 {
        let per_key = history.0.coefficients(w).max(batch.0.coefficients(w));
        let per = history.1.coefficients(w).max(batch.1.coefficients(w));
        assert!(work.len() >= per_key + BLOCK * per);
        let (key_coefficients, value_coefficients) = work.split_at_mut(per_key);
        let mut block = Block::new(std::ptr::null());
        for (keys, first, count) in walk(spans, fresh, range, selected) {
            let work = (&mut *key_coefficients, &mut *value_coefficients, per);
            // SAFETY: the tier runs F8 (`vectors`), over whole chunks.
            unsafe {
                match keys {
                    Keys::History => state.take_vectors::<_, _, HV, BV>(queries, scale, first, count, scores, work, keys, history, &mut block),
                    Keys::Fresh => state.take_vectors::<_, _, HV, BV>(queries, scale, first, count, scores, work, keys, batch, &mut block),
                }
            }
        }
        if block.n > 0 {
            // SAFETY: as above.
            unsafe { state.flush_vectors::<HV, BV>(scores, value_coefficients, per, &mut block) };
        }
    } else {
        let row = &mut work[..w];
        let mut block = Block::new(0);
        for (keys, first, count) in walk(spans, fresh, range, selected) {
            match keys {
                Keys::History => {
                    state.take_portable(queries, scale, first, count, scores, row, keys, history.0, &mut block, (history.1, batch.1))
                }
                Keys::Fresh => {
                    state.take_portable(queries, scale, first, count, scores, row, keys, batch.0, &mut block, (history.1, batch.1))
                }
            }
        }
        if block.n > 0 {
            state.flush_portable(scores, row, &mut block, (history.1, batch.1));
        }
    }
}

/// Absorbs a row's index-list picks (`picks`, -1 for none), each the
/// source's row `load(pick)` of element `E` read as key and value, into
/// `state` as [`attend`] absorbs its keys, in blocks from the start of each
/// run of picks (the portable `attention_rows`' index-list source).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn attend_picks<'p, L: Isa, E: Dense>(
    l: L,
    state: &mut Online<'_>,
    queries: &[f32],
    scale: f32,
    scores: &mut [f32],
    work: &mut [f32],
    picks: &[i32],
    load: impl Fn(usize) -> &'p [E::Storage],
) where
    E::Storage: 'p,
{
    let mut first = 0;
    while first < picks.len() {
        if picks[first] < 0 {
            first += 1;
            continue;
        }
        let end = picks[first..].iter().position(|pick| *pick < 0).map_or(picks.len(), |run| first + run);
        let run = &picks[first..end];
        // A pick is its own value; the run is one span of keys.
        let rows = DenseRows::<E, _>::new(|j: usize| load(run[j] as usize));
        let keys = (0, run.len() as i32);
        attend(l, state, queries, scale, scores, work, std::iter::empty(), keys, 0..run.len(), |_| true, (&rows, &rows), (&rows, &rows));
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

/// Merges the partial states of one query head over its `parts` partitions
/// in partition order: partition `part` has output `partial(part)`, maximum
/// `maximum(part)` and denominator `denominator(part)`; an empty partition
/// (maximum -inf) contributes nothing. Writes the unnormalized output and
/// returns the denominator.
#[inline(always)]
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
