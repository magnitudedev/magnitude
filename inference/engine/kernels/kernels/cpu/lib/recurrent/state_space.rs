// The state-space (Mamba-2) library of the CPU entries (`state_space_step`,
// `state_space_chunk`; contracts and the portable `state_space_rows` in
// state_space.seismic): operand addressing, the causal convolution with its
// bias and SiLU, the step size and decay, the successor window publication,
// and the row-sequential advance both entries run. The counterpart of
// `metal/lib/recurrent/state_space.h`, `cuda/lib/recurrent/state_space.cuh`
// and `vulkan/lib/recurrent/state_space.glsl`.
//
// Decomposition: a work item owns (head, block of state rows, slot) and keeps
// its state rows private for the whole slot; the rows of the slot advance in
// order. Its per-row inputs (the group's convolved B and C, the x channels of
// its state rows, the step size and decay) come from a `Prologue`: computed in
// the work item (`Fused`, the step) or read from rows another launch staged
// once (`Staged`, the chunk). Both compute every value with the same
// functions, so the entries give the same bits, and a work item's state-row
// block never changes them.
//
// Numerics: the portable body's arithmetic in F32, in its order: the decayed
// state `S * decay` is rounded before the update `fma(u, B, S * decay)`, and
// the output `S C` is the body's ascending FMA chain over the state columns.

use super::super::core::activation;
use super::versions::{Slot, Slots};
use seismic::cpu::slab::SlabTensor;
use seismic::cpu::{math, Dense, Tensor, F32};
use std::ops::Range;

/// The operands of one state-space call, shared by both entries.
pub struct StateSpace<'a, A: Dense> {
    pub projection: Tensor<'a, A, 2>,
    pub convolution: Tensor<'a, F32, 2>,
    pub convolution_bias: Tensor<'a, F32, 1>,
    pub rate: Tensor<'a, F32, 1>,
    pub time_bias: Tensor<'a, F32, 1>,
    pub skip: Tensor<'a, F32, 1>,
    pub slots: Slots<'a>,
    pub window: SlabTensor<'a, A, 3>,
    pub state: SlabTensor<'a, F32, 4>,
    pub tape: SlabTensor<'a, F32, 3>,
    pub mixed: Tensor<'a, A, 3>,
    /// M.
    pub rows: usize,
    /// NH.
    pub heads: usize,
    /// P.
    pub head_width: usize,
    /// G.
    pub groups: usize,
    /// N.
    pub state_width: usize,
    /// C.
    pub taps: usize,
    /// T.
    pub tape_rows: usize,
}

/// The inputs of one row for one head and block of state rows: the group's
/// convolved B and C, the convolved x channels of the block's state rows, the
/// step size and the decay.
pub struct Inputs<'b> {
    pub b: &'b [f32],
    pub c: &'b [f32],
    pub x: &'b [f32],
    pub delta: f32,
    pub decay: f32,
}

/// Where a work item's per-row inputs come from.
pub trait Prologue {
    fn row(&mut self, row: usize) -> Inputs<'_>;
}

/// The state rows of work item `item` when every item owns `rows` of `width`
/// state rows (the last item owns the remainder).
#[inline(always)]
pub fn state_rows(item: u64, rows: u64, width: usize) -> Range<usize> {
    let first = (item * rows) as usize;
    first.min(width)..(first + rows as usize).min(width)
}

impl<'a, A: Dense> StateSpace<'a, A> {
    /// Convolved channels x | B | C.
    #[inline(always)]
    pub fn channels(&self) -> usize {
        self.heads * self.head_width + 2 * self.groups * self.state_width
    }

    /// Floats of one staged row (`stage_row`): the convolved channels, then
    /// each head's step size, then its log decay.
    #[inline(always)]
    pub fn staged_width(&self) -> usize {
        self.channels() + 2 * self.heads
    }

    #[inline(always)]
    fn b_channel(&self, group: usize) -> usize {
        self.heads * self.head_width + group * self.state_width
    }

    #[inline(always)]
    fn c_channel(&self, group: usize) -> usize {
        self.heads * self.head_width + (self.groups + group) * self.state_width
    }

    /// The group whose B and C head `head` reads.
    #[inline(always)]
    pub fn group_of(&self, head: usize) -> usize {
        head * self.groups / self.heads
    }

    /// Whether `head` is its group's first head: the one that publishes and
    /// records the group's B and C.
    #[inline(always)]
    pub fn leads_group(&self, head: usize) -> bool {
        head == 0 || self.group_of(head - 1) != self.group_of(head)
    }

    /// Tape offsets: inputs u [NH, P], B [G, N], decays d [NH].
    #[inline(always)]
    fn tape_b(&self, group: usize) -> usize {
        self.heads * self.head_width + group * self.state_width
    }

    #[inline(always)]
    fn tape_decay(&self, head: usize) -> usize {
        self.heads * self.head_width + self.groups * self.state_width + head
    }

    /// The raw convolution input row (x | B | C channels) at slot-local
    /// `position`: the source version's window rows before the slot, the
    /// projection after.
    #[inline(always)]
    fn raw_row(&self, slot: &Slot, position: isize) -> &'a [A::Storage] {
        if position < 0 {
            // Positions reach back at most C - 1 rows.
            self.window.row([slot.source, slot.taped + self.taps - 1 - position.unsigned_abs(), 0])
        } else {
            let first = self.heads * self.head_width;
            &self.projection.row([slot.lo + position as usize, 0])[first..first + self.channels()]
        }
    }

    /// SiLU of the causal depthwise convolution of `channels` at slot-local
    /// row `local` plus the channel bias, in the body's order: the current
    /// row's tap first, then taps 0..C - 1 fused in turn, then the bias.
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
        for (target, channel) in out.iter_mut().zip(channels) {
            *target = activation::silu(*target + self.convolution_bias.get([channel]));
        }
    }

    /// The step size delta = softplus(dt + time_bias) and the log decay
    /// rate * delta of head `head` at `row`.
    #[inline(always)]
    pub fn step(&self, row: usize, head: usize) -> (f32, f32) {
        let dt = 2 * self.heads * self.head_width + 2 * self.groups * self.state_width + head;
        let delta = math::softplus(self.projection.get([row, dt]) + self.time_bias.get([head]));
        (delta, self.rate.get([head]) * delta)
    }

    /// Every input of slot-local row `local` into `out` (`staged_width`
    /// floats): the convolved channels, then each head's step size, then its
    /// log decay.
    #[inline(always)]
    pub fn stage_row(&self, slot: &Slot, local: usize, out: &mut [f32]) {
        let (channels, heads) = (self.channels(), self.heads);
        self.convolve(slot, local, 0..channels, out);
        for head in 0..heads {
            let (delta, log_decay) = self.step(slot.lo + local, head);
            out[channels + head] = delta;
            out[channels + heads + head] = log_decay;
        }
    }

    /// Publishes head `head`'s share of the slot's successor window (the
    /// C - 1 raw rows before its publication row, then the raw rows of its
    /// tape): its x channels, and its group's B and C channels when it leads
    /// the group. A bit-exact copy.
    pub fn publish_window(&self, slot: &Slot, head: usize) {
        let (p, n) = (self.head_width, self.state_width);
        let group = self.group_of(head);
        let (b, c) = (self.b_channel(group), self.c_channel(group));
        let spans = [head * p..(head + 1) * p, b..b + n, c..c + n];
        let count = if self.leads_group(head) { 3 } else { 1 };
        let last = self.taps - 1;
        for tap in 0..last + slot.recorded(self.tape_rows) {
            let source = self.raw_row(slot, (slot.stop + tap) as isize - last as isize);
            for span in &spans[..count] {
                // SAFETY: each head copies its own channels of the successor
                // bank's rows; no slot reads a successor bank.
                let target = unsafe { self.window.span_mut([slot.target, tap, span.start], span.len()) };
                target.copy_from_slice(&source[span.clone()]);
            }
        }
    }

    /// Zeroes state rows `rows` of head `head` in the mixed rows no slot
    /// covers.
    pub fn zero_uncovered(&self, head: usize, rows: Range<usize>) {
        for row in self.slots.covered_end()..self.rows {
            // SAFETY: each work item writes its own state rows of the head.
            let out = unsafe { self.mixed.span_mut([row, head, rows.start], rows.len()) };
            out.fill(A::narrow(0.0));
        }
    }

    /// Stores state rows `rows` of `head` to the successor bank.
    #[inline(always)]
    fn publish_state(&self, slot: &Slot, head: usize, rows: &Range<usize>, state: &[f32]) {
        let n = self.state_width;
        for (index, row) in rows.clone().enumerate() {
            // SAFETY: each work item publishes its own state rows of the head.
            let target = unsafe { self.state.row_mut([slot.target, head, row, 0]) };
            target.copy_from_slice(&state[index * n..(index + 1) * n]);
        }
    }

    /// The row-sequential state-space rule of the slot for state rows `rows`
    /// of head `head`, held in `state` (`rows.len() * N` floats): read from
    /// the source version (the bank's state advanced by its tape rows),
    /// advanced over every row of the slot with the inputs `prologue` gives,
    /// published to the successor after the slot's first `stop` rows, and
    /// the rows after it recorded in the successor's tape. The work item of
    /// the head's first state rows records the decay and, when the head leads
    /// its group, B.
    pub fn advance<P: Prologue>(&self, slot: &Slot, head: usize, rows: Range<usize>, state: &mut [f32], prologue: &mut P) {
        let n = self.state_width;
        let group = self.group_of(head);
        let state = &mut state[..rows.len() * n];
        for (index, row) in rows.clone().enumerate() {
            state[index * n..(index + 1) * n].copy_from_slice(self.state.row([slot.source, head, row, 0]));
        }
        for entry in 0..slot.taped {
            let tape = self.tape.row([slot.source, entry, 0]);
            let decay = tape[self.tape_decay(head)];
            let b = &tape[self.tape_b(group)..][..n];
            for (index, row) in rows.clone().enumerate() {
                let input = tape[head * self.head_width + row];
                for (value, b) in state[index * n..(index + 1) * n].iter_mut().zip(b) {
                    *value = input.mul_add(*b, *value * decay);
                }
            }
        }
        let publish = slot.publish();
        if publish == slot.lo {
            self.publish_state(slot, head, &rows, state);
        }
        let recorded = slot.recorded(self.tape_rows);
        let records = rows.start == 0;
        let records_b = records && self.leads_group(head);
        let skip = self.skip.get([head]);
        for row in slot.lo..slot.hi {
            let inputs = prologue.row(row);
            let entry = (row >= publish && row - publish < recorded).then(|| row - publish);
            if let Some(entry) = entry {
                // SAFETY: the successor's tape row is written by this head's
                // first work item (decay, B) and by each work item for its own
                // state rows (inputs); no slot reads it.
                unsafe {
                    if records {
                        self.tape.set([slot.target, entry, self.tape_decay(head)], inputs.decay);
                    }
                    if records_b {
                        self.tape.span_mut([slot.target, entry, self.tape_b(group)], n).copy_from_slice(inputs.b);
                    }
                }
            }
            // SAFETY: each work item writes its own state rows of the head.
            let mixed = unsafe { self.mixed.span_mut([row, head, rows.start], rows.len()) };
            // SAFETY: as above, this work item's inputs of the entry.
            let mut recorded_inputs = entry.map(|entry| unsafe {
                self.tape.span_mut([slot.target, entry, head * self.head_width + rows.start], rows.len())
            });
            for index in 0..rows.len() {
                let value = inputs.x[index];
                let input = inputs.delta * value;
                let mut output = 0.0f32;
                for ((s, b), c) in state[index * n..(index + 1) * n].iter_mut().zip(inputs.b).zip(inputs.c) {
                    *s = input.mul_add(*b, *s * inputs.decay);
                    output = s.mul_add(*c, output);
                }
                mixed[index] = A::narrow(skip.mul_add(value, output));
                if let Some(recorded_inputs) = &mut recorded_inputs {
                    recorded_inputs[index] = input;
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
    space: &'r StateSpace<'a, A>,
    slot: Slot,
    head: usize,
    rows: Range<usize>,
    b: &'r mut [f32],
    c: &'r mut [f32],
    x: &'r mut [f32],
}

impl<'r, 'a, A: Dense> Fused<'r, 'a, A> {
    /// Over `buffers` of at least `2 * N + rows.len()` floats.
    pub fn new(space: &'r StateSpace<'a, A>, slot: Slot, head: usize, rows: Range<usize>, buffers: &'r mut [f32]) -> Self {
        let n = space.state_width;
        let (b, rest) = buffers.split_at_mut(n);
        let (c, rest) = rest.split_at_mut(n);
        let x = &mut rest[..rows.len()];
        Self { space, slot, head, rows, b, c, x }
    }
}

impl<A: Dense> Prologue for Fused<'_, '_, A> {
    #[inline(always)]
    fn row(&mut self, row: usize) -> Inputs<'_> {
        let s = self.space;
        let local = row - self.slot.lo;
        let group = s.group_of(self.head);
        let n = s.state_width;
        let b = s.b_channel(group);
        s.convolve(&self.slot, local, b..b + n, self.b);
        let c = s.c_channel(group);
        s.convolve(&self.slot, local, c..c + n, self.c);
        let x = self.head * s.head_width;
        s.convolve(&self.slot, local, x + self.rows.start..x + self.rows.end, self.x);
        let (delta, log_decay) = s.step(row, self.head);
        Inputs { b: self.b, c: self.c, x: self.x, delta, decay: math::exp(log_decay) }
    }
}

/// The chunk's prologue: each row's inputs read from the rows `stage_row`
/// staged (`staged_width` floats per row).
pub struct Staged<'s> {
    staged: &'s [f32],
    stride: usize,
    b: usize,
    c: usize,
    x: Range<usize>,
    state_width: usize,
    delta: usize,
    log_decay: usize,
}

impl<'s> Staged<'s> {
    pub fn new<A: Dense>(space: &StateSpace<'_, A>, staged: &'s [f32], head: usize, rows: Range<usize>) -> Self {
        let group = space.group_of(head);
        let x = head * space.head_width;
        Self {
            staged,
            stride: space.staged_width(),
            b: space.b_channel(group),
            c: space.c_channel(group),
            x: x + rows.start..x + rows.end,
            state_width: space.state_width,
            delta: space.channels() + head,
            log_decay: space.channels() + space.heads + head,
        }
    }
}

impl Prologue for Staged<'_> {
    #[inline(always)]
    fn row(&mut self, row: usize) -> Inputs<'_> {
        let staged = &self.staged[row * self.stride..(row + 1) * self.stride];
        let n = self.state_width;
        Inputs {
            b: &staged[self.b..self.b + n],
            c: &staged[self.c..self.c + n],
            x: &staged[self.x.clone()],
            delta: staged[self.delta],
            decay: math::exp(staged[self.log_decay]),
        }
    }
}
