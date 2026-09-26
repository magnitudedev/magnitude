// The gated-delta library of the CPU entries (`gated_delta_step`,
// `gated_delta_chunk`; contracts and the portable `gated_delta_rows` in
// recurrent.seismic): slot and version lookup, the causal convolution with
// SiLU and the q/k normalization, the gates, the successor window
// publication, and the row-sequential advance both entries run. The
// counterpart of `metal/lib/recurrent/recurrent.h`,
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
// Numerics: the portable body's arithmetic in F32, with the dot products of
// the state rows (`S k`, `S q`) and the q/k squares summed in the defined
// order of `reduce`. The decayed state `S * decay` is rounded before the
// dot and the update `fma(residual, k, S * decay)`, as the body computes it.

use super::super::core::{activation, reduce};
use seismic::cpu::{math, Dense, Scalars, Tensor, F32};
use std::ops::Range;

/// The operands of one gated-delta call, shared by both entries.
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
    pub window: Tensor<'a, A, 3>,
    pub delta: Tensor<'a, F32, 4>,
    pub tape: Tensor<'a, F32, 3>,
    pub mixed: Tensor<'a, A, 3>,
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
/// the block's state rows, beta and the decay.
pub struct Inputs<'b> {
    pub query: &'b [f32],
    pub key: &'b [f32],
    pub value: &'b [f32],
    pub beta: f32,
    pub decay: f32,
}

/// Where a work item's per-row inputs come from.
pub trait Prologue {
    fn row(&mut self, row: usize) -> Inputs<'_>;
}

impl<'a, A: Dense> Recurrence<'a, A> {
    /// Channels of a raw input row: q heads, k heads, v heads.
    #[inline(always)]
    pub fn channels(&self) -> usize {
        (2 * self.key_heads + self.value_heads) * self.width
    }

    /// Floats of one staged row (`stage_row`): the prepared channels, then
    /// beta and the decay of every value head.
    #[inline(always)]
    pub fn staged_width(&self) -> usize {
        self.channels() + 2 * self.value_heads
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

    /// Tape offsets: innovations u [NV, W], keys k [NK, W], decays d [NV].
    #[inline(always)]
    fn tape_key(&self, key_head: usize) -> usize {
        (self.value_heads + key_head) * self.width
    }

    #[inline(always)]
    fn tape_decay(&self, head: usize) -> usize {
        (self.value_heads + self.key_heads) * self.width + head
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

    /// beta = sigmoid(b) and decay = exp(rate * softplus(alpha + time_bias))
    /// of value head `head` at `row`.
    #[inline(always)]
    pub fn gates(&self, row: usize, head: usize) -> (f32, f32) {
        let alpha_column = self.channels() + self.value_heads * self.width + head;
        let alpha = self.projection.get([row, alpha_column]);
        let beta_input = self.projection.get([row, alpha_column + self.value_heads]);
        let beta = math::sigmoid(beta_input);
        let decay = math::exp(self.rate.get([head]) * math::softplus(alpha + self.time_bias.get([head])));
        (beta, decay)
    }

    /// Every input of slot-local row `local` into `out` (`staged_width`
    /// floats): the prepared q, k and v channels, then beta and the decay of
    /// each value head.
    #[inline(always)]
    pub fn stage_row(&self, slot: &Slot, local: usize, out: &mut [f32]) {
        let (w, nk, nv) = (self.width, self.key_heads, self.value_heads);
        let channels = self.channels();
        self.convolve(slot, local, 0..channels, out);
        for head in 0..2 * nk {
            self.normalize(&mut out[head * w..(head + 1) * w], head < nk);
        }
        for head in 0..nv {
            let (beta, decay) = self.gates(slot.lo + local, head);
            out[channels + head] = beta;
            out[channels + nv + head] = decay;
        }
    }

    /// Publishes value head `head`'s share of the slot's successor window
    /// (the C - 1 raw rows before its publication row, then the raw rows of
    /// its tape): its value channels and the q/k channels of the key heads
    /// congruent to it modulo NV. A bit-exact copy.
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

    /// Zeroes state rows `rows` of value head `head` in the mixed rows no
    /// slot covers.
    pub fn zero_uncovered(&self, head: usize, rows: Range<usize>) {
        for row in self.covered_end()..self.rows {
            // SAFETY: each work item writes its own state rows of the head.
            let out = unsafe { self.mixed.span_mut([row, head, rows.start], rows.len()) };
            out.fill(A::narrow(0.0));
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

    /// The row-sequential gated delta rule of the slot for state rows `rows`
    /// of value head `head`, held in `state` (`rows.len() * W` floats): read
    /// from the source version (the bank's state advanced by its tape rows),
    /// advanced over every row of the slot with the inputs `prologue` gives,
    /// published to the successor after the slot's first `stop` rows, and
    /// the rows after it recorded in the successor's tape. The work item of
    /// the head's first state rows records the decay and, when the head is
    /// its key head's recorder, the key.
    pub fn advance<P: Prologue>(&self, slot: &Slot, head: usize, rows: Range<usize>, state: &mut [f32], prologue: &mut P) {
        let w = self.width;
        let key_head = self.key_head(head);
        let state = &mut state[..rows.len() * w];
        for (index, row) in rows.clone().enumerate() {
            state[index * w..(index + 1) * w].copy_from_slice(self.delta.row([slot.source, head, row, 0]));
        }
        for entry in 0..slot.taped {
            let tape = self.tape.row([slot.source, entry, 0]);
            let factor = tape[self.tape_decay(head)];
            let key = &tape[self.tape_key(key_head)..][..w];
            for (index, row) in rows.clone().enumerate() {
                let innovation = tape[head * w + row];
                for (value, key) in state[index * w..(index + 1) * w].iter_mut().zip(key) {
                    *value = innovation.mul_add(*key, *value * factor);
                }
            }
        }
        let publish = slot.publish();
        if publish == slot.lo {
            self.publish_state(slot, head, &rows, state);
        }
        let recorded = self.recorded(slot);
        let records = rows.start == 0;
        let records_key = records && self.records_key(head);
        for row in slot.lo..slot.hi {
            let inputs = prologue.row(row);
            let entry = (row >= publish && row - publish < recorded).then(|| row - publish);
            if let Some(entry) = entry {
                // SAFETY: the successor's tape row is written by this head's
                // first work item (decay, key) and by each work item for its
                // own state rows (innovations); no slot reads it.
                unsafe {
                    if records {
                        self.tape.set([slot.target, entry, self.tape_decay(head)], inputs.decay);
                    }
                    if records_key {
                        self.tape.span_mut([slot.target, entry, self.tape_key(key_head)], w).copy_from_slice(inputs.key);
                    }
                }
            }
            // SAFETY: each work item writes its own state rows of the head.
            let mixed = unsafe { self.mixed.span_mut([row, head, rows.start], rows.len()) };
            // SAFETY: as above, this work item's innovations of the entry.
            let mut innovations =
                entry.map(|entry| unsafe { self.tape.span_mut([slot.target, entry, head * w + rows.start], rows.len()) });
            for index in 0..rows.len() {
                let values = &mut state[index * w..(index + 1) * w];
                for value in values.iter_mut() {
                    *value *= inputs.decay;
                }
                let remembered = reduce::dot(values, inputs.key);
                let residual = (inputs.value[index] - remembered) * inputs.beta;
                for (value, key) in values.iter_mut().zip(inputs.key) {
                    *value = residual.mul_add(*key, *value);
                }
                mixed[index] = A::narrow(reduce::dot(values, inputs.query));
                if let Some(innovations) = &mut innovations {
                    innovations[index] = residual;
                }
            }
            if row + 1 == publish {
                self.publish_state(slot, head, &rows, state);
            }
        }
    }
}

/// The step's prologue: each row's inputs computed in the work item.
pub struct Fused<'r, 'a, A: Dense> {
    recurrence: &'r Recurrence<'a, A>,
    slot: Slot,
    head: usize,
    rows: Range<usize>,
    query: &'r mut [f32],
    key: &'r mut [f32],
    value: &'r mut [f32],
}

impl<'r, 'a, A: Dense> Fused<'r, 'a, A> {
    /// Over `buffers` of at least `2 * W + rows.len()` floats.
    pub fn new(recurrence: &'r Recurrence<'a, A>, slot: Slot, head: usize, rows: Range<usize>, buffers: &'r mut [f32]) -> Self {
        let w = recurrence.width;
        let (query, rest) = buffers.split_at_mut(w);
        let (key, rest) = rest.split_at_mut(w);
        let value = &mut rest[..rows.len()];
        Self { recurrence, slot, head, rows, query, key, value }
    }
}

impl<A: Dense> Prologue for Fused<'_, '_, A> {
    #[inline(always)]
    fn row(&mut self, row: usize) -> Inputs<'_> {
        let r = self.recurrence;
        let local = row - self.slot.lo;
        r.prepare_key_head(&self.slot, local, r.key_head(self.head), self.query, self.key);
        let value_channel = (2 * r.key_heads + self.head) * r.width;
        r.convolve(&self.slot, local, value_channel + self.rows.start..value_channel + self.rows.end, self.value);
        let (beta, decay) = r.gates(row, self.head);
        Inputs { query: self.query, key: self.key, value: self.value, beta, decay }
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
    head: usize,
    key_head: usize,
    rows: Range<usize>,
}

impl<'s> Staged<'s> {
    pub fn new<A: Dense>(recurrence: &Recurrence<'_, A>, staged: &'s [f32], head: usize, rows: Range<usize>) -> Self {
        Self {
            staged,
            stride: recurrence.staged_width(),
            width: recurrence.width,
            key_heads: recurrence.key_heads,
            value_heads: recurrence.value_heads,
            head,
            key_head: recurrence.key_head(head),
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
            decay: staged[channels + nv + self.head],
        }
    }
}
