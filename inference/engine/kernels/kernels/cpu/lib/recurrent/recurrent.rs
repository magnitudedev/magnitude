// The delta-rule library of the CPU entries (`gated_delta_step`,
// `gated_delta_chunk`, `kimi_delta_step`, `kimi_delta_chunk`; contracts and
// the portable `gated_delta_rows` and `kimi_delta_rows` in recurrent.seismic):
// slot and version lookup, the causal convolution with SiLU and the q/k
// normalization, the successor window publication, and, for any `Rule`, every
// entry's launches: the staging, the row-sequential advance and the gating.
// An entry binds its operands (`recurrence!`, `kimi::kimi_rule!`) and calls
// one launch. Gated
// DeltaNet's rule is `Recurrence` itself; Kimi Delta Attention's is
// `kimi::Kimi`. The counterpart of `metal/lib/recurrent/recurrent.h`,
// `cuda/lib/recurrent/recurrent.cuh` and `vulkan/lib/recurrent/recurrent.glsl`.
//
// Decomposition: a work item owns (value head, block of state rows, slot) and
// keeps its state rows private for the whole slot; the rows of the slot
// advance in order. Its per-row inputs (the key head's normalized q and k,
// its value channels and the head's gates) come from a `Prologue`: computed
// in the work item (`Fused`, the step) or read from rows another launch
// staged once (`Staged`, the chunk). Both compute every value with the same
// functions, so the entries give the same bits, and a work item's state-row
// block never changes them.
//
// Numerics: the portable bodies' arithmetic in F32, with the dot products of
// the state rows (`S k`, `S q`) and the q/k squares summed in the defined
// order of `reduce`. The decayed state `S diag(d)` is rounded before the dot
// and the update `fma(residual, k, S diag(d))`, as the bodies compute it.
// The raw outputs `S q` go to the `mixed` scratch (F32 values rounded to A,
// [M, NV, W]); a `gate` launch then forms each row's gated result.

use super::super::core::{activation, functions, reduce};
use seismic::cpu::{math, Dense, Element, Scalars, Scratch, Tensor, F32};
use seismic::cpu::slab::SlabTensor;
use std::ops::Range;

/// The operands every delta rule reads: the projection rows (q | k | v
/// channels first), the convolution, the state arenas and the geometry.
pub struct Recurrence<'a, A: Dense> {
    pub projection: Tensor<'a, A, 2>,
    pub convolution: Tensor<'a, F32, 2>,
    pub rate: Tensor<'a, F32, 1>,
    pub time_bias: Tensor<'a, F32, 1>,
    pub segments: Scalars<'a, i32, 2>,
    pub stop: Scalars<'a, i32, 1>,
    pub previous_bank: Scalars<'a, i32, 1>,
    pub previous_tape: Scalars<'a, i32, 1>,
    pub following_bank: Scalars<'a, i32, 1>,
    pub window: SlabTensor<'a, A, 3>,
    pub delta: SlabTensor<'a, F32, 4>,
    pub tape: SlabTensor<'a, F32, 3>,
    /// The raw outputs, [M, NV, W] F32 values rounded to A.
    pub mixed: Scratch<'a>,
    /// M.
    pub rows: usize,
    /// B.
    pub slots: usize,
    /// NK.
    pub key_heads: usize,
    /// NV.
    pub value_heads: usize,
    /// W.
    pub width: usize,
    /// C.
    pub taps: usize,
    /// T.
    pub tape_rows: usize,
    pub epsilon: f32,
    pub grouped: bool,
}

/// The `Recurrence` an entry binds from its own parameters, which every
/// delta-rule entry declares alike (`cx`, its generated `Context`, has no
/// type the library can name). Expands where `recurrent` names this module.
macro_rules! recurrence {
    ($cx:expr) => {
        recurrent::Recurrence {
            projection: $cx.arg_projection(),
            convolution: $cx.arg_convolution(),
            rate: $cx.arg_rate(),
            time_bias: $cx.arg_time_bias(),
            segments: $cx.arg_segments(),
            stop: $cx.arg_stop(),
            previous_bank: $cx.arg_previous_bank(),
            previous_tape: $cx.arg_previous_tape(),
            following_bank: $cx.arg_following_bank(),
            window: seismic::cpu::slab::SlabTensor::from_bound($cx.arg_window(), $cx.arg_slab_banks() as usize),
            delta: seismic::cpu::slab::SlabTensor::from_bound($cx.arg_delta(), $cx.arg_slab_banks() as usize),
            tape: seismic::cpu::slab::SlabTensor::from_bound($cx.arg_tape(), $cx.arg_slab_banks() as usize),
            mixed: $cx.scratch_mixed(),
            rows: $cx.dim_m() as usize,
            slots: $cx.dim_b() as usize,
            key_heads: $cx.dim_nk() as usize,
            value_heads: $cx.dim_nv() as usize,
            width: $cx.dim_w() as usize,
            taps: $cx.dim_c() as usize,
            tape_rows: $cx.dim_t() as usize,
            epsilon: $cx.arg_norm_epsilon(),
            grouped: $cx.arg_grouped(),
        }
    };
}
pub(crate) use recurrence;

/// One slot: rows [lo, hi), its publication row count `stop`, the version it
/// reads (bank `source` advanced by its first `taped` tape rows) and its
/// successor bank `target`.
#[derive(Clone, Copy, Debug)]
pub struct Slot {
    pub lo: usize,
    pub hi: usize,
    pub stop: usize,
    pub source: usize,
    pub taped: usize,
    pub target: usize,
}

impl Slot {
    /// The row after which the successor's state is published.
    #[inline(always)]
    pub fn publish(&self) -> usize {
        self.lo + self.stop
    }
}

/// The state rows of work item `item` when every item owns `rows` of `width`
/// state rows (the last item owns the remainder).
#[inline(always)]
pub fn state_rows(item: u64, rows: u64, width: usize) -> Range<usize> {
    let first = (item * rows) as usize;
    first.min(width)..(first + rows as usize).min(width)
}

/// The inputs of one row for one value head and block of state rows: the
/// key head's L2-normalized q (scaled by W^-1/2) and k, the value channels of
/// the block's state rows, beta and the head's decay factors (`Rule::decays`).
pub struct Inputs<'b> {
    pub query: &'b [f32],
    pub key: &'b [f32],
    pub value: &'b [f32],
    pub beta: f32,
    pub decay: &'b [f32],
}

/// Where a work item's per-row inputs come from.
pub trait Prologue {
    fn row(&mut self, row: usize) -> Inputs<'_>;
}

/// `values` (a state row's columns) scaled by a row's `decays` decay factors:
/// the head's one for every column, or one each. A rule's `decays` is its
/// constant, so the branch resolves at compile time.
#[inline(always)]
fn scale_columns(values: &mut [f32], decay: &[f32], decays: usize) {
    if decays == 1 {
        let factor = decay[0];
        for value in values.iter_mut() {
            *value *= factor;
        }
    } else {
        for (value, factor) in values.iter_mut().zip(decay) {
            *value *= factor;
        }
    }
}

/// A tape entry's update of a state row: value * factor + innovation * key
/// for each column, the factors as in `scale_columns`.
#[inline(always)]
fn replay_columns(values: &mut [f32], key: &[f32], innovation: f32, decay: &[f32], decays: usize) {
    if decays == 1 {
        let factor = decay[0];
        for (value, key) in values.iter_mut().zip(key) {
            *value = innovation.mul_add(*key, *value * factor);
        }
    } else {
        for ((value, key), factor) in values.iter_mut().zip(key).zip(decay) {
            *value = innovation.mul_add(*key, *value * factor);
        }
    }
}

impl<'a, A: Dense> Recurrence<'a, A> {
    /// Channels of a raw input row: q heads, k heads, v heads.
    #[inline(always)]
    pub fn channels(&self) -> usize {
        (2 * self.key_heads + self.value_heads) * self.width
    }

    #[inline(always)]
    pub fn slot(&self, index: usize) -> Slot {
        let lo = self.segments.get([index, 0]) as usize;
        Slot {
            lo,
            hi: self.segments.get([index, 1]) as usize,
            stop: self.stop.get([index]) as usize,
            source: self.previous_bank.get([index]) as usize,
            taped: self.previous_tape.get([index]) as usize,
            target: self.following_bank.get([index]) as usize,
        }
    }

    /// The slot holding `row`, if any.
    #[inline(always)]
    pub fn slot_of_row(&self, row: usize) -> Option<Slot> {
        (0..self.slots).map(|index| self.slot(index)).find(|slot| slot.lo <= row && row < slot.hi)
    }

    /// The end of the rows the slots cover (they partition a prefix).
    #[inline(always)]
    pub fn covered_end(&self) -> usize {
        if self.slots == 0 {
            0
        } else {
            self.segments.get([self.slots - 1, 1]) as usize
        }
    }

    /// Tape rows the slot records in its successor: those after its stop
    /// row, at most T.
    #[inline(always)]
    pub fn recorded(&self, slot: &Slot) -> usize {
        self.tape_rows.min(slot.hi - slot.lo - slot.stop)
    }

    /// The key head of value head `head`.
    #[inline(always)]
    pub fn key_head(&self, head: usize) -> usize {
        if self.grouped {
            head * self.key_heads / self.value_heads
        } else {
            head % self.key_heads
        }
    }

    /// Whether `head` is the lowest value head reading its key head: the one
    /// that records the key in a tape row.
    #[inline(always)]
    pub fn records_key(&self, head: usize) -> bool {
        if self.grouped {
            head == 0 || self.key_head(head - 1) != self.key_head(head)
        } else {
            head < self.key_heads
        }
    }

    /// Tape offsets: innovations u [NV, W], keys k [NK, W], then `decays`
    /// decay factors per value head.
    #[inline(always)]
    fn tape_key(&self, key_head: usize) -> usize {
        (self.value_heads + key_head) * self.width
    }

    #[inline(always)]
    fn tape_decay(&self, head: usize, decays: usize) -> usize {
        (self.value_heads + self.key_heads) * self.width + head * decays
    }


    /// The raw input row at slot-local `position`: the source version's
    /// window rows before the slot, the projection after.
    #[inline(always)]
    fn raw_row(&self, slot: &Slot, position: isize) -> &'a [A::Storage] {
        if position < 0 {
            // Positions reach back at most C - 1 rows.
            self.window.row([slot.source, slot.taped + self.taps - 1 - position.unsigned_abs(), 0])
        } else {
            self.projection.row([slot.lo + position as usize, 0])
        }
    }

    /// SiLU of the causal depthwise convolution of `channels` of slot-local
    /// row `local`, in the body's order: the current row's tap first, then
    /// taps 0..C - 1 fused in turn.
    #[inline(always)]
    pub fn convolve(&self, slot: &Slot, local: usize, channels: Range<usize>, out: &mut [f32]) {
        let out = &mut out[..channels.len()];
        let current = &self.raw_row(slot, local as isize)[channels.clone()];
        let last = self.taps - 1;
        for ((target, channel), value) in out.iter_mut().zip(channels.clone()).zip(current) {
            *target = self.convolution.get([channel, last]) * A::widen(*value);
        }
        for tap in 0..last {
            let previous = &self.raw_row(slot, local as isize + tap as isize - last as isize)[channels.clone()];
            for ((target, channel), value) in out.iter_mut().zip(channels.clone()).zip(previous) {
                *target = self.convolution.get([channel, tap]).mul_add(A::widen(*value), *target);
            }
        }
        for value in out.iter_mut() {
            *value = activation::silu(*value);
        }
    }

    /// L2-normalizes a convolved q (`query`) or k head row.
    #[inline(always)]
    fn normalize(&self, values: &mut [f32], query: bool) {
        let mut inverse = math::rsqrt(reduce::sum_squares(values) + self.epsilon);
        if query {
            inverse *= math::rsqrt(self.width as f32);
        }
        for value in values.iter_mut() {
            *value *= inverse;
        }
    }

    /// The key head's prepared q and k of slot-local row `local`.
    #[inline(always)]
    pub fn prepare_key_head(&self, slot: &Slot, local: usize, key_head: usize, query: &mut [f32], key: &mut [f32]) {
        let w = self.width;
        self.convolve(slot, local, key_head * w..(key_head + 1) * w, query);
        self.normalize(&mut query[..w], true);
        let key_channel = (self.key_heads + key_head) * w;
        self.convolve(slot, local, key_channel..key_channel + w, key);
        self.normalize(&mut key[..w], false);
    }


    /// Publishes value head `head`'s share of the slot's successor window
    /// (the C - 1 raw rows before its publication row, then the raw rows of
    /// its tape): its value channels and the q/k channels of the key heads
    /// congruent to it modulo NV. A bit-exact copy.
    #[inline(always)]
    pub fn publish_window(&self, slot: &Slot, head: usize) {
        let (w, nk, nv) = (self.width, self.key_heads, self.value_heads);
        let last = self.taps - 1;
        let key_channels = (head..nk).step_by(nv).flat_map(|key_head| [key_head * w, (nk + key_head) * w]);
        let channels = std::iter::once((2 * nk + head) * w).chain(key_channels);
        for tap in 0..last + self.recorded(slot) {
            let source = self.raw_row(slot, (slot.stop + tap) as isize - last as isize);
            for channel in channels.clone() {
                // SAFETY: each value head copies its own channels of the
                // successor bank's rows; no slot reads a successor bank.
                let target = unsafe { self.window.span_mut([slot.target, tap, channel], w) };
                target.copy_from_slice(&source[channel..channel + w]);
            }
        }
    }

    /// The raw outputs of state rows `rows` of value head `head` at `row`.
    ///
    /// # Safety
    /// Each work item writes only its own state rows of its head.
    #[allow(clippy::mut_from_ref)]
    #[inline(always)]
    unsafe fn raw_mut(&self, row: usize, head: usize, rows: &Range<usize>) -> &'a mut [f32] {
        let at = (row * self.value_heads + head) * self.width + rows.start;
        unsafe { self.mixed.slice_mut::<f32>(4 * at, rows.len()) }
    }

    /// Zeroes state rows `rows` of value head `head` in the raw rows no slot
    /// covers.
    #[inline(always)]
    pub fn zero_uncovered(&self, head: usize, rows: Range<usize>) {
        for row in self.covered_end()..self.rows {
            // SAFETY: each work item writes its own state rows of the head.
            unsafe { self.raw_mut(row, head, &rows) }.fill(0.0);
        }
    }

    /// Stores state rows `rows` of `head` to the successor bank.
    #[inline(always)]
    fn publish_state(&self, slot: &Slot, head: usize, rows: &Range<usize>, state: &[f32]) {
        let w = self.width;
        for (index, row) in rows.clone().enumerate() {
            // SAFETY: each work item publishes its own state rows of the head.
            let target = unsafe { self.delta.row_mut([slot.target, head, row, 0]) };
            target.copy_from_slice(&state[index * w..(index + 1) * w]);
        }
    }
}

/// A delta rule over the gated-delta operands: its gates and its output
/// gate's input z, which the staging, the row-sequential advance and the
/// gating (its provided methods) read. Gated DeltaNet's (`Recurrence`) are
/// columns of the projection row, with one decay per value head; Kimi Delta
/// Attention's (`kimi::Kimi`) are their own inputs, with one decay per key
/// column.
pub trait Rule<'a> {
    /// The activation element A.
    type Act: Dense;

    /// The operands every rule reads.
    fn recurrence(&self) -> &Recurrence<'a, Self::Act>;

    /// Decay factors of a value head per row: 1 (the head's one decay scales
    /// every state column) or W (one per column). A tape row records as many
    /// per value head after its keys.
    fn decays(&self) -> usize;

    /// beta of value head `head` at `row`, with the head's decay factors in
    /// `decay` (`decays` floats).
    fn gates(&self, row: usize, head: usize, decay: &mut [f32]) -> f32;

    /// z of value head `head` at `row`: W values in A.
    fn gate_input(&self, row: usize, head: usize) -> &'a [<Self::Act as Element>::Storage];

    /// Floats of one staged row (`stage_row`): the prepared channels, then
    /// beta of every value head, then their decay factors.
    #[inline(always)]
    fn staged_width(&self) -> usize {
        let r = self.recurrence();
        r.channels() + r.value_heads * (1 + self.decays())
    }

    /// Floats of a work item's `Fused` buffers over `rows` state rows: q, k,
    /// the value channels and the decay factors.
    #[inline(always)]
    fn fused_floats(&self, rows: usize) -> usize {
        2 * self.recurrence().width + rows + self.decays()
    }

    /// Every input of slot-local row `local` into `out` (`staged_width`
    /// floats): the prepared q, k and v channels, then beta of each value
    /// head, then their decay factors.
    #[inline(always)]
    fn stage_row(&self, slot: &Slot, local: usize, out: &mut [f32]) {
        let r = self.recurrence();
        let (w, nk, nv) = (r.width, r.key_heads, r.value_heads);
        let (channels, decays) = (r.channels(), self.decays());
        r.convolve(slot, local, 0..channels, out);
        for head in 0..2 * nk {
            r.normalize(&mut out[head * w..(head + 1) * w], head < nk);
        }
        let (betas, factors) = out[channels..].split_at_mut(nv);
        for (head, beta) in betas.iter_mut().enumerate() {
            *beta = self.gates(slot.lo + local, head, &mut factors[head * decays..(head + 1) * decays]);
        }
    }

    /// The `gate` launch of every entry: row `row`'s raw outputs, per value
    /// head RMS-normalized over W with `epsilon` and scaled by the recurrent
    /// norm (`norm`, decoded; A), times the output gate of z (A; code
    /// `gate_function`), published to A in `gated`. `values` holds W floats.
    #[inline(always)]
    fn gate_row(
        &self,
        row: usize,
        norm: &[f32],
        epsilon: f32,
        gate_function: i32,
        gated: &Tensor<'a, Self::Act, 3>,
        values: &mut [f32],
    ) {
        let r = self.recurrence();
        let (w, nv) = (r.width, r.value_heads);
        // SAFETY: the advance launch wrote every raw row before this launch;
        // this one only reads them.
        let raw = unsafe { r.mixed.slice::<f32>(4 * row * nv * w, nv * w) };
        for head in 0..nv {
            values.copy_from_slice(&raw[head * w..(head + 1) * w]);
            let inverse = reduce::rms_inverse(values, epsilon);
            let gates = self.gate_input(row, head);
            // SAFETY: each work item writes its own row.
            let out = unsafe { gated.span_mut([row, head, 0], w) };
            for (((target, value), weight), gate) in out.iter_mut().zip(values.iter()).zip(norm).zip(gates) {
                let normalized = activation::publish::<Self::Act>(value * inverse * weight);
                let activated = activation::publish::<Self::Act>(functions::output_gate(
                    gate_function,
                    <Self::Act as Dense>::widen(*gate),
                ));
                *target = <Self::Act as Dense>::narrow(normalized * activated);
            }
        }
    }

    /// The step entry's launch (`*_step`): work item `group` = (value head,
    /// state-row block of `block_rows` (`ROWS`), slot) advances its rows with
    /// the inputs computed in place (`Fused`); grid z = B zeroes the raw rows
    /// no slot covers, and each head's first work item publishes its share of
    /// the successor window.
    #[inline(always)]
    fn step(&self, group: [u64; 3], block_rows: u64, shared: &mut [u8])
    where
        Self: Sized,
    {
        let r = self.recurrence();
        let head = group[0] as usize;
        let rows = state_rows(group[1], block_rows, r.width);
        if group[2] as usize == r.slots {
            r.zero_uncovered(head, rows);
            return;
        }
        let slot = r.slot(group[2] as usize);
        if rows.start == 0 {
            r.publish_window(&slot, head);
        }
        let floats = seismic::cpu::tensor::floats(shared, rows.len() * r.width + self.fused_floats(rows.len()));
        let (state, buffers) = floats.split_at_mut(rows.len() * r.width);
        let mut prologue = Fused::new(self, slot, head, rows.clone(), buffers);
        self.advance(&slot, head, rows, state, &mut prologue);
    }

    /// The chunk entry's `inputs` launch (`*_chunk_inputs`): row `row` of
    /// every slot staged once into `inputs` (`stage_row`); rows after the
    /// slots are no slot's input.
    #[inline(always)]
    fn chunk_inputs(&self, row: usize, inputs: Scratch<'a>) {
        let Some(slot) = self.recurrence().slot_of_row(row) else { return };
        let stride = self.staged_width();
        // SAFETY: each work item writes its own row of the scratch.
        let staged = unsafe { inputs.slice_mut::<f32>(4 * row * stride, stride) };
        self.stage_row(&slot, row - slot.lo, staged);
    }

    /// The chunk entry's `scan` launch (`*_chunk_scan`): `step`'s work items
    /// over the rows the `inputs` launch staged (`Staged`), so every slot
    /// gets the step's bits (the CPU needs no WY form: the row-sequential
    /// rule is its cheapest schedule).
    #[inline(always)]
    fn chunk_scan(&self, group: [u64; 3], block_rows: u64, inputs: Scratch<'a>, shared: &mut [u8])
    where
        Self: Sized,
    {
        let r = self.recurrence();
        let head = group[0] as usize;
        let rows = state_rows(group[1], block_rows, r.width);
        if group[2] as usize == r.slots {
            r.zero_uncovered(head, rows);
            return;
        }
        let slot = r.slot(group[2] as usize);
        if rows.start == 0 {
            r.publish_window(&slot, head);
        }
        // SAFETY: the inputs launch wrote the rows of every slot before this
        // launch; this one only reads them.
        let staged = unsafe { inputs.slice::<f32>(0, r.rows * self.staged_width()) };
        let state = seismic::cpu::tensor::floats(shared, rows.len() * r.width);
        let mut prologue = Staged::new(self, staged, head, rows.clone());
        self.advance(&slot, head, rows, state, &mut prologue);
    }

    /// Every entry's `gate` launch (`*_gate`): row `row` gated as `gate_row`
    /// gates it, the recurrent norm decoded by `decode_norm` into W floats.
    #[inline(always)]
    fn gate(
        &self,
        row: usize,
        decode_norm: impl FnOnce(&mut [f32]),
        epsilon: f32,
        gate_function: i32,
        gated: &Tensor<'a, Self::Act, 3>,
        shared: &mut [u8],
    ) {
        let w = self.recurrence().width;
        let (norm, values) = seismic::cpu::tensor::floats(shared, 2 * w).split_at_mut(w);
        decode_norm(norm);
        self.gate_row(row, norm, epsilon, gate_function, gated, values);
    }

    /// The row-sequential delta rule of the slot for state rows `rows` of
    /// value head `head`, held in `state` (`rows.len() * W` floats): read
    /// from the source version (the bank's state advanced by its tape rows),
    /// advanced over every row of the slot with the inputs `prologue` gives
    /// (S <- S diag(d) + beta (v - S diag(d) k) k^T, output S q), published
    /// to the successor after the slot's first `stop` rows, and the rows
    /// after it recorded in the successor's tape. The work item of the
    /// head's first state rows records the decay factors and, when the head
    /// is its key head's recorder, the key.
    #[inline(always)]
    fn advance<P: Prologue>(&self, slot: &Slot, head: usize, rows: Range<usize>, state: &mut [f32], prologue: &mut P) {
        let r = self.recurrence();
        let (w, decays) = (r.width, self.decays());
        let key_head = r.key_head(head);
        let state = &mut state[..rows.len() * w];
        for (index, row) in rows.clone().enumerate() {
            state[index * w..(index + 1) * w].copy_from_slice(r.delta.row([slot.source, head, row, 0]));
        }
        for entry in 0..slot.taped {
            let tape = r.tape.row([slot.source, entry, 0]);
            let decay = &tape[r.tape_decay(head, decays)..][..decays];
            let key = &tape[r.tape_key(key_head)..][..w];
            for (index, row) in rows.clone().enumerate() {
                let innovation = tape[head * w + row];
                replay_columns(&mut state[index * w..(index + 1) * w], key, innovation, decay, decays);
            }
        }
        let publish = slot.publish();
        if publish == slot.lo {
            r.publish_state(slot, head, &rows, state);
        }
        let recorded = r.recorded(slot);
        let records = rows.start == 0;
        let records_key = records && r.records_key(head);
        for row in slot.lo..slot.hi {
            let inputs = prologue.row(row);
            let entry = (row >= publish && row - publish < recorded).then(|| row - publish);
            if let Some(entry) = entry {
                // SAFETY: the successor's tape row is written by this head's
                // first work item (decays, key) and by each work item for its
                // own state rows (innovations); no slot reads it.
                unsafe {
                    if records {
                        r.tape.span_mut([slot.target, entry, r.tape_decay(head, decays)], decays).copy_from_slice(inputs.decay);
                    }
                    if records_key {
                        r.tape.span_mut([slot.target, entry, r.tape_key(key_head)], w).copy_from_slice(inputs.key);
                    }
                }
            }
            // SAFETY: each work item writes its own state rows of the head.
            let mixed = unsafe { r.raw_mut(row, head, &rows) };
            // SAFETY: as above, this work item's innovations of the entry.
            let mut innovations =
                entry.map(|entry| unsafe { r.tape.span_mut([slot.target, entry, head * w + rows.start], rows.len()) });
            for index in 0..rows.len() {
                let values = &mut state[index * w..(index + 1) * w];
                scale_columns(values, inputs.decay, decays);
                let remembered = reduce::dot(values, inputs.key);
                let residual = (inputs.value[index] - remembered) * inputs.beta;
                for (value, key) in values.iter_mut().zip(inputs.key) {
                    *value = residual.mul_add(*key, *value);
                }
                mixed[index] = activation::publish::<Self::Act>(reduce::dot(values, inputs.query));
                if let Some(innovations) = &mut innovations {
                    innovations[index] = residual;
                }
            }
            if row + 1 == publish {
                r.publish_state(slot, head, &rows, state);
            }
        }
    }
}

/// Gated DeltaNet: beta and the head's one decay from columns of the
/// projection row, z its columns from CH.
impl<'a, A: Dense> Rule<'a> for Recurrence<'a, A> {
    type Act = A;

    #[inline(always)]
    fn recurrence(&self) -> &Recurrence<'a, A> {
        self
    }

    #[inline(always)]
    fn decays(&self) -> usize {
        1
    }

    /// beta = sigmoid(b) and decay = exp(rate * softplus(alpha + time_bias))
    /// of value head `head` at `row`.
    #[inline(always)]
    fn gates(&self, row: usize, head: usize, decay: &mut [f32]) -> f32 {
        let alpha_column = self.channels() + self.value_heads * self.width + head;
        let alpha = self.projection.get([row, alpha_column]);
        let beta_input = self.projection.get([row, alpha_column + self.value_heads]);
        decay[0] = math::exp(self.rate.get([head]) * math::softplus(alpha + self.time_bias.get([head])));
        math::sigmoid(beta_input)
    }

    #[inline(always)]
    fn gate_input(&self, row: usize, head: usize) -> &'a [A::Storage] {
        self.projection.span([row, self.channels() + head * self.width], self.width)
    }
}

/// The step's prologue: each row's inputs computed in the work item.
pub struct Fused<'r, R> {
    rule: &'r R,
    slot: Slot,
    head: usize,
    rows: Range<usize>,
    query: &'r mut [f32],
    key: &'r mut [f32],
    value: &'r mut [f32],
    decay: &'r mut [f32],
}

impl<'r, R> Fused<'r, R> {
    /// Over `buffers` of at least `Rule::fused_floats` floats.
    #[inline(always)]
    pub fn new<'a>(rule: &'r R, slot: Slot, head: usize, rows: Range<usize>, buffers: &'r mut [f32]) -> Self
    where
        R: Rule<'a>,
    {
        let w = rule.recurrence().width;
        let (query, rest) = buffers.split_at_mut(w);
        let (key, rest) = rest.split_at_mut(w);
        let (value, rest) = rest.split_at_mut(rows.len());
        let decay = &mut rest[..rule.decays()];
        Self { rule, slot, head, rows, query, key, value, decay }
    }
}

impl<'a, R: Rule<'a>> Prologue for Fused<'_, R> {
    #[inline(always)]
    fn row(&mut self, row: usize) -> Inputs<'_> {
        let r = self.rule.recurrence();
        let local = row - self.slot.lo;
        r.prepare_key_head(&self.slot, local, r.key_head(self.head), self.query, self.key);
        let value_channel = (2 * r.key_heads + self.head) * r.width;
        r.convolve(&self.slot, local, value_channel + self.rows.start..value_channel + self.rows.end, self.value);
        let beta = self.rule.gates(row, self.head, self.decay);
        Inputs { query: self.query, key: self.key, value: self.value, beta, decay: self.decay }
    }
}

/// The chunk's prologue: each row's inputs read from the rows `stage_row`
/// staged (`staged_width` floats per row).
pub struct Staged<'s> {
    staged: &'s [f32],
    stride: usize,
    width: usize,
    key_heads: usize,
    value_heads: usize,
    decays: usize,
    head: usize,
    key_head: usize,
    rows: Range<usize>,
}

impl<'s> Staged<'s> {
    #[inline(always)]
    pub fn new<'a, R: Rule<'a>>(rule: &R, staged: &'s [f32], head: usize, rows: Range<usize>) -> Self {
        let r = rule.recurrence();
        Self {
            staged,
            stride: rule.staged_width(),
            width: r.width,
            key_heads: r.key_heads,
            value_heads: r.value_heads,
            decays: rule.decays(),
            head,
            key_head: r.key_head(head),
            rows,
        }
    }
}

impl Prologue for Staged<'_> {
    #[inline(always)]
    fn row(&mut self, row: usize) -> Inputs<'_> {
        let (w, nk, nv) = (self.width, self.key_heads, self.value_heads);
        let staged = &self.staged[row * self.stride..(row + 1) * self.stride];
        let channels = (2 * nk + nv) * w;
        let value = (2 * nk + self.head) * w;
        Inputs {
            query: &staged[self.key_head * w..][..w],
            key: &staged[(nk + self.key_head) * w..][..w],
            value: &staged[value + self.rows.start..value + self.rows.end],
            beta: staged[channels + self.head],
            decay: &staged[channels + nv + self.head * self.decays..][..self.decays],
        }
    }
}
