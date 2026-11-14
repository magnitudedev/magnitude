// Recurrent state entries against their semantic oracle, on Metal (when
// present) and the CPU.
//
// `gated_delta_step` and `gated_delta_chunk` share one portable body
// (`gated_delta_rows`), and Kimi Delta Attention's `kimi_delta_step` and
// `kimi_delta_chunk` theirs (`kimi_delta_rows`); a case is one form's
// (`Case::per_channel`). Small geometries run that body in the Seismic
// interpreter; the real 4B geometry uses an independent f64 host model of the
// same contract. Every case checks the gated outputs, the published window
// (bit-exact: it is a copy) and state, and that no bank other than each
// slot's successor changes: accepted banks, the zero seed, and unrelated
// banks keep their exact bytes.

mod common;

use common::{
    bound::Float,
    recurrent_bound::{
        Bounds, Native, RowTrace, Shape, SlotTrace, StateRow, Trace, CUDA_CHUNK_PIECE, METAL_CHUNK_PIECE,
        SEQUENTIAL_ROWS,
    },
};

use magnitude_kernels::{gated_delta_chunk, gated_delta_step, kimi_delta_chunk, kimi_delta_step};
use seismic::{
    BackendName, Device, Element, NativeSpecialization, SlabRegion, SlabTensor,
    Tensor,
};

const SENTINEL: f32 = 7.25;

#[derive(Clone, Copy)]
struct Geometry {
    key_heads: usize,
    value_heads: usize,
    width: usize,
    /// Decay inputs per value head (DW): 1, or `width` (one per key channel).
    decay_width: usize,
    convolution: usize,
    banks: usize,
    /// Tape rows per bank (T).
    tape: usize,
}

impl Geometry {
    fn channels(self) -> usize {
        (2 * self.key_heads + self.value_heads) * self.width
    }
    /// A case's input row: qkv | z | a (NV * DW) | b, Gated DeltaNet's
    /// projection row; Kimi Delta Attention's entries take it as four
    /// operands ([`Case::operands`]).
    fn projection_width(self) -> usize {
        self.channels() + self.value_heads * self.width + self.decays() + self.value_heads
    }
    /// Decay inputs of a row (NV * DW).
    fn decays(self) -> usize {
        self.value_heads * self.decay_width
    }
    fn window_rows(self) -> usize {
        self.convolution - 1 + self.tape
    }
    fn window_bank(self) -> usize {
        self.window_rows() * self.channels()
    }
    fn delta_bank(self) -> usize {
        self.value_heads * self.width * self.width
    }
    /// Floats of one tape row: u [NV, W] | k [NK, W] | d [NV, DW].
    fn tape_row(self) -> usize {
        (self.value_heads + self.key_heads) * self.width + self.decays()
    }
    fn tape_bank(self) -> usize {
        self.tape * self.tape_row()
    }
}

/// One slot: its rows, published prefix, the version it reads (accepted bank
/// and its tape rows) and its successor bank.
#[derive(Clone, Copy)]
struct SlotCase {
    rows: usize,
    stop: usize,
    previous: usize,
    following: usize,
    taped: usize,
}

/// A slot reading bank `previous` with no tape rows.
fn slot(rows: usize, stop: usize, previous: usize, following: usize) -> SlotCase {
    SlotCase {
        rows,
        stop,
        previous,
        following,
        taped: 0,
    }
}

struct Case {
    geometry: Geometry,
    bf16: bool,
    rows: usize,
    slots: Vec<SlotCase>,
    grouped: bool,
    /// The output gate's code: 0 SiLU, 1 sigmoid.
    gate_function: i32,
    epsilon: f32,
    /// The gating's RMS epsilon.
    rms_epsilon: f32,
    /// Input rows qkv | z | a | b ([`Geometry::projection_width`]).
    projection: Vec<f32>,
    convolution: Vec<f32>,
    rate: Vec<f32>,
    /// One per decay input (NV * DW).
    time_bias: Vec<f32>,
    /// The bounded-sigmoid decay's lower bound; the softplus decay without.
    decay_lower: Option<f32>,
    /// The recurrent norm weights (W).
    norm: Vec<f32>,
    window: Vec<f32>,
    delta: Vec<f32>,
    tape: Vec<f32>,
    /// The f64 host's trajectory, once a check asks for it.
    trace: std::cell::OnceCell<Trace>,
    /// Each form's bounds on this case, derived once.
    bounds: std::cell::RefCell<std::collections::HashMap<Native, std::rc::Rc<Bounds>>>,
}

struct Outcome {
    gated: Vec<f32>,
    window: Vec<f32>,
    delta: Vec<f32>,
    tape: Vec<f32>,
}

/// Device tensors of one case; each call mutates its window, delta and tape.
struct Tensors {
    inputs: Inputs,
    convolution: Tensor,
    rate: Tensor,
    time_bias: Tensor,
    norm: Tensor,
    segments: Tensor,
    stop: Tensor,
    previous: Tensor,
    previous_tape: Tensor,
    following: Tensor,
    _slabs: SlabTensor,
    window: Tensor,
    delta: Tensor,
    tape: Tensor,
    slab_banks: u32,
}

/// A case's rows as its form's entries take them.
enum Inputs {
    /// Gated DeltaNet: the projection row qkv | z | alpha | beta.
    Gated { projection: Tensor },
    /// Kimi Delta Attention: q | k | v, beta, z and the decay inputs, and
    /// the decay's lower bound.
    Kimi {
        projection: Tensor,
        beta: Tensor,
        gate: Tensor,
        decay: Tensor,
        lower: f32,
    },
}

/// The state entries' common arguments of `$tensors` for `$case`, beside
/// the form's inputs `$inputs`.
macro_rules! state_args {
    ($module:ident, $tensors:expr, $case:expr, { $($inputs:tt)* }) => {
        $module::Args {
            $($inputs)*
            convolution: &$tensors.convolution,
            rate: &$tensors.rate,
            time_bias: &$tensors.time_bias,
            recurrent_norm: &$tensors.norm,
            segments: &$tensors.segments,
            stop: &$tensors.stop,
            previous_bank: &$tensors.previous,
            previous_tape: &$tensors.previous_tape,
            following_bank: &$tensors.following,
            window: &mut $tensors.window,
            delta: &mut $tensors.delta,
            tape: &mut $tensors.tape,
            norm_epsilon: $case.epsilon,
            epsilon: $case.rms_epsilon,
            grouped: $case.grouped,
            gate_function: $case.gate_function,
            slab_banks: $tensors.slab_banks,
        }
    };
}

impl Tensors {
    /// A Gated DeltaNet case's projection row.
    fn projection(&self) -> &Tensor {
        match &self.inputs {
            Inputs::Gated { projection } => projection,
            Inputs::Kimi { .. } => panic!("a Kimi Delta Attention case has no projection row"),
        }
    }

    fn step_args<'a>(&'a mut self, case: &Case) -> gated_delta_step::Args<'a> {
        let Inputs::Gated { projection } = &self.inputs else {
            panic!("`gated_delta_step` runs Gated DeltaNet cases")
        };
        state_args!(gated_delta_step, self, case, { projection, })
    }

    fn chunk_args<'a>(&'a mut self, case: &Case) -> gated_delta_chunk::Args<'a> {
        let Inputs::Gated { projection } = &self.inputs else {
            panic!("`gated_delta_chunk` runs Gated DeltaNet cases")
        };
        state_args!(gated_delta_chunk, self, case, { projection, })
    }

    /// The case's form's step (`chunk` false) or chunk on `device` at
    /// `specialization`, its gated rows.
    fn run(
        &mut self,
        device: &Device,
        case: &Case,
        activation: Element,
        chunk: bool,
        specialization: &NativeSpecialization,
    ) -> Tensor {
        macro_rules! call {
            ($module:ident, $args:expr) => {
                $module::native_for_device_with(
                    device,
                    $module::Elements {
                        RN: Element::f32(),
                        A: activation,
                    },
                    specialization,
                )
                .unwrap()
                .call($args)
                .unwrap()
                .value
            };
        }
        match (&self.inputs, chunk) {
            (Inputs::Gated { .. }, false) => call!(gated_delta_step, self.step_args(case)),
            (Inputs::Gated { .. }, true) => call!(gated_delta_chunk, self.chunk_args(case)),
            (&Inputs::Kimi { ref projection, ref beta, ref gate, ref decay, lower }, false) => call!(
                kimi_delta_step,
                state_args!(kimi_delta_step, self, case, { projection, beta, gate, decay, decay_lower: lower, })
            ),
            (&Inputs::Kimi { ref projection, ref beta, ref gate, ref decay, lower }, true) => call!(
                kimi_delta_chunk,
                state_args!(kimi_delta_chunk, self, case, { projection, beta, gate, decay, decay_lower: lower, })
            ),
        }
    }
}

struct Random(u64);

impl Random {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn bf16_round(value: f32) -> f32 {
    let bits = value.to_bits();
    f32::from_bits((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff_0000)
}

impl Case {
    /// Rows are packed after the slots, as the batch packer pads them.
    fn new(
        geometry: Geometry,
        rows: usize,
        slots: Vec<SlotCase>,
        grouped: bool,
        seed: u64,
    ) -> Self {
        let mut random = Random(seed);
        let width = geometry.projection_width();
        let mut projection = (0..rows * width).map(|_| random.next()).collect::<Vec<_>>();
        let gates = geometry.channels() + geometry.value_heads * geometry.width;
        for row in 0..rows {
            // a: a spread of decays; beta input: both gate regimes.
            for input in 0..geometry.decays() {
                projection[row * width + gates + input] = random.next() * 3.0;
            }
            for head in 0..geometry.value_heads {
                projection[row * width + gates + geometry.decays() + head] = random.next() * 4.0;
            }
        }
        let convolution = (0..geometry.channels() * geometry.convolution)
            .map(|_| random.next() * 0.6)
            .collect();
        let rate = (0..geometry.value_heads)
            .map(|_| -(0.05 + 0.5 * (random.next() + 1.0)))
            .collect();
        let time_bias = (0..geometry.decays())
            .map(|_| random.next() * 0.5)
            .collect();
        let norm = (0..geometry.width)
            .map(|_| 1.0 + 0.5 * random.next())
            .collect();
        let accepted = slots.iter().map(|slot| slot.previous).collect::<Vec<_>>();
        let mut window = vec![SENTINEL; geometry.banks * geometry.window_bank()];
        let mut delta = vec![SENTINEL; geometry.banks * geometry.delta_bank()];
        let mut tape = vec![SENTINEL; geometry.banks * geometry.tape_bank()];
        window[..geometry.window_bank()].fill(0.0);
        delta[..geometry.delta_bank()].fill(0.0);
        tape[..geometry.tape_bank()].fill(0.0);
        for bank in 1..geometry.banks {
            if accepted.contains(&bank) {
                for value in &mut window[bank * geometry.window_bank()..][..geometry.window_bank()]
                {
                    *value = random.next();
                }
                for value in &mut delta[bank * geometry.delta_bank()..][..geometry.delta_bank()] {
                    *value = random.next() * 0.3;
                }
                // Tape rows: innovations, keys and decays in (0.5, 1).
                let nvw = geometry.value_heads * geometry.width;
                let nkw = geometry.key_heads * geometry.width;
                for entry in 0..geometry.tape {
                    let row = &mut tape
                        [bank * geometry.tape_bank() + entry * geometry.tape_row()..]
                        [..geometry.tape_row()];
                    for (index, value) in row.iter_mut().enumerate() {
                        *value = if index < nvw {
                            random.next() * 0.3
                        } else if index < nvw + nkw {
                            random.next() * 0.2
                        } else {
                            0.75 + 0.25 * random.next()
                        };
                    }
                }
            }
        }
        Self {
            geometry,
            bf16: false,
            rows,
            slots,
            grouped,
            gate_function: 0,
            epsilon: 1.0e-6 * geometry.width as f32,
            rms_epsilon: 1.0e-6,
            projection,
            convolution,
            rate,
            time_bias,
            decay_lower: None,
            norm,
            window,
            delta,
            tape,
            trace: Default::default(),
            bounds: Default::default(),
        }
    }

    /// Kimi Delta Attention's decay: log d = lower * sigmoid(-rate * (a +
    /// bias)) with lower = -5, its inputs drawn so the log decays span
    /// (-5, 0) and reach both ends.
    fn with_bounded_decay(mut self) -> Self {
        let g = self.geometry;
        let width = g.projection_width();
        let first = g.channels() + g.value_heads * g.width;
        let mut random = Random(0x6b64 + self.rows as u64);
        for row in 0..self.rows {
            for input in 0..g.decays() {
                self.projection[row * width + first + input] = random.next() * 9.0;
            }
        }
        if self.bf16 {
            for value in &mut self.projection {
                *value = bf16_round(*value);
            }
        }
        self.decay_lower = Some(-5.0);
        self
    }

    /// Rows [first, first + rows) of this case as one slot `slot`, over the
    /// arenas `arenas` left (this case's own without).
    fn part(&self, first: usize, rows: usize, slot: SlotCase, arenas: Option<&Outcome>) -> Self {
        let width = self.geometry.projection_width();
        let (window, delta, tape) = match arenas {
            Some(outcome) => (outcome.window.clone(), outcome.delta.clone(), outcome.tape.clone()),
            None => (self.window.clone(), self.delta.clone(), self.tape.clone()),
        };
        Self {
            geometry: self.geometry,
            bf16: self.bf16,
            rows,
            slots: vec![slot],
            grouped: self.grouped,
            gate_function: self.gate_function,
            epsilon: self.epsilon,
            rms_epsilon: self.rms_epsilon,
            projection: self.projection[first * width..(first + rows) * width].to_vec(),
            convolution: self.convolution.clone(),
            rate: self.rate.clone(),
            time_bias: self.time_bias.clone(),
            decay_lower: self.decay_lower,
            norm: self.norm.clone(),
            window,
            delta,
            tape,
            trace: Default::default(),
            bounds: Default::default(),
        }
    }

    /// Whether the case is Kimi Delta Attention's (one bounded decay per key
    /// channel) rather than Gated DeltaNet's (one softplus decay per head).
    fn per_channel(&self) -> bool {
        let g = self.geometry;
        match (g.decay_width, self.decay_lower) {
            (1, None) => false,
            (width, Some(_)) if width == g.width => true,
            _ => panic!("a case has one softplus decay per head or one bounded decay per channel"),
        }
    }

    /// Kimi Delta Attention's four input operands: q | k | v, beta, z and
    /// the decay inputs a.
    fn operands(&self) -> [Vec<f32>; 4] {
        let g = self.geometry;
        let (channels, z, decays) = (g.channels(), g.value_heads * g.width, g.decays());
        let mut operands = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for row in self.projection.chunks_exact(g.projection_width()) {
            operands[0].extend_from_slice(&row[..channels]);
            operands[1].extend_from_slice(&row[channels + z + decays..]);
            operands[2].extend_from_slice(&row[channels..channels + z]);
            operands[3].extend_from_slice(&row[channels + z..channels + z + decays]);
        }
        operands
    }

    /// The log decay of decay input `input` (of value head `head`) at `row`.
    /// The decay input `input` (of NV * DW) of `row`: `a + time_bias`.
    fn shifted(&self, row: usize, input: usize) -> f64 {
        let g = self.geometry;
        let first = g.channels() + g.value_heads * g.width;
        let a = self.projection[row * g.projection_width() + first + input] as f64;
        a + self.time_bias[input] as f64
    }

    fn log_decay(&self, row: usize, head: usize, input: usize) -> f64 {
        let shifted = self.shifted(row, input);
        let rate = self.rate[head] as f64;
        match self.decay_lower {
            None => rate * (shifted.max(0.0) + (1.0 + (-shifted.abs()).exp()).ln()),
            Some(lower) => lower as f64 / (1.0 + (rate * shifted).exp()),
        }
    }

    /// The next advance of the same layer after `outcome`: new projection rows
    /// for `slots` (reading the versions they name), the same weights, and the
    /// arenas `outcome` left.
    fn continued(&self, outcome: &Outcome, rows: usize, slots: Vec<SlotCase>, seed: u64) -> Self {
        let mut fresh = Case::new(self.geometry, rows, slots, self.grouped, seed);
        if self.decay_lower.is_some() {
            fresh = fresh.with_bounded_decay();
        }
        let round = |values: &[f32]| {
            if self.bf16 {
                values.iter().map(|value| bf16_round(*value)).collect()
            } else {
                values.to_vec()
            }
        };
        Self {
            projection: round(&fresh.projection),
            convolution: self.convolution.clone(),
            rate: self.rate.clone(),
            time_bias: self.time_bias.clone(),
            decay_lower: self.decay_lower,
            norm: self.norm.clone(),
            window: outcome.window.clone(),
            delta: outcome.delta.clone(),
            tape: outcome.tape.clone(),
            bf16: self.bf16,
            gate_function: self.gate_function,
            ..fresh
        }
    }

    /// Gate the outputs with the sigmoid of z (code 1) in place of SiLU.
    fn with_sigmoid_gate(mut self) -> Self {
        self.gate_function = 1;
        self
    }

    /// BF16 activations: projection and window values are BF16 numbers.
    fn with_bf16_activations(mut self) -> Self {
        for value in self.projection.iter_mut().chain(self.window.iter_mut()) {
            *value = bf16_round(*value);
        }
        self.bf16 = true;
        self
    }

    /// Force near-total decay on some rows: a recurrence reset.
    fn with_resets(mut self, rows: &[usize]) -> Self {
        let width = self.geometry.projection_width();
        let alpha = self.geometry.channels() + self.geometry.value_heads * self.geometry.width;
        for &row in rows {
            for input in 0..self.geometry.decays() {
                self.projection[row * width + alpha + input] = 250.0;
            }
        }
        self
    }

    fn segments(&self) -> Vec<i32> {
        let mut segments = Vec::new();
        let mut row = 0;
        for slot in &self.slots {
            segments.extend([row as i32, (row + slot.rows) as i32]);
            row += slot.rows;
        }
        segments.extend([self.rows as i32, self.rows as i32]);
        segments
    }

    /// The f64 host's trajectory, which every bound of this case derives from.
    fn trace(&self) -> &Trace {
        self.trace.get_or_init(|| self.host_traced().1)
    }

    /// `native`'s bounds on this case.
    fn bounds(&self, native: Native) -> std::rc::Rc<Bounds> {
        self.bounds
            .borrow_mut()
            .entry(native)
            .or_insert_with(|| std::rc::Rc::new(self.trace().bounds(native)))
            .clone()
    }

    /// The f64 host model of the entry contract.
    fn host(&self) -> Outcome {
        self.host_traced().0
    }

    /// The f64 host model of the entry contract, with the trajectory its
    /// bounds are derived from ([`recurrent_bound::Trace`]).
    fn host_traced(&self) -> (Outcome, Trace) {
        let g = self.geometry;
        let (nk, nv, w, c) = (g.key_heads, g.value_heads, g.width, g.convolution);
        let dw = g.decay_width;
        let channels = g.channels();
        let width = g.projection_width();
        let key_of = |head: usize| {
            if self.grouped {
                head * nk / nv
            } else {
                head % nk
            }
        };
        let norm_of = |values: &[f64]| values.iter().map(|v| v * v).sum::<f64>().sqrt();
        let mut mixed = vec![0.0f32; self.rows * nv * w];
        let mut window = self.window.clone();
        let mut delta = self.delta.clone();
        let mut tape = self.tape.clone();
        let mut rows: Vec<RowTrace> = Vec::with_capacity(self.rows);
        let mut slots: Vec<SlotTrace> = Vec::new();
        let mut first = 0;
        for slot in &self.slots {
            let mut state = delta[slot.previous * g.delta_bank()..][..g.delta_bank()]
                .iter()
                .map(|value| *value as f64)
                .collect::<Vec<_>>();
            let mut replay = Vec::with_capacity(slot.taped * nv * w);
            // The version: the bank's state advanced by its first tape rows.
            for entry in 0..slot.taped {
                let row = &self.tape[slot.previous * g.tape_bank() + entry * g.tape_row()..]
                    [..g.tape_row()];
                for head in 0..nv {
                    let decay = &row[(nv + nk) * w + head * dw..][..dw];
                    let key = &row[nv * w + key_of(head) * w..][..w];
                    let largest = decay.iter().fold(0.0f32, |a, b| a.max(*b));
                    for state_row in 0..w {
                        let innovation = row[head * w + state_row] as f64;
                        let s = &mut state[(head * w + state_row) * w..][..w];
                        for column in 0..w {
                            s[column] *= decay[column % dw] as f64;
                        }
                        let decayed = norm_of(s);
                        for column in 0..w {
                            s[column] += innovation * key[column] as f64;
                        }
                        replay.push([largest, decayed as f32, norm_of(s) as f32]);
                    }
                }
            }
            // `replay` was pushed entry-major, head, state row: the trace's
            // layout.
            slots.push(SlotTrace {
                first,
                rows: slot.rows,
                stop: slot.stop,
                previous: slot.previous,
                following: slot.following,
                taped: slot.taped,
                replay,
            });
            let publish = |state: &[f64], delta: &mut Vec<f32>| {
                for (index, value) in state.iter().enumerate() {
                    delta[slot.following * g.delta_bank() + index] = *value as f32;
                }
            };
            if slot.stop == 0 {
                publish(&state, &mut delta);
            }
            // Raw input `position` (slot-local) of a channel.
            let raw = |position: isize, channel: usize| -> f32 {
                if position < 0 {
                    self.window[slot.previous * g.window_bank()
                        + (slot.taped as isize + c as isize - 1 + position) as usize * channels
                        + channel]
                } else {
                    self.projection[(first + position as usize) * width + channel]
                }
            };
            let input = |_row: usize, local: usize, tap: usize, channel: usize| -> f64 {
                raw(local as isize + tap as isize - (c as isize - 1), channel) as f64
            };
            let taped = g.tape.min(slot.rows - slot.stop);
            for local in 0..slot.rows {
                let row = first + local;
                let mut prepared = vec![0.0f64; channels];
                let mut trace = RowTrace {
                    abs_convolution: vec![0.0; channels],
                    convolved: vec![0.0; channels],
                    activated: vec![0.0; channels],
                    prepared: Vec::new(),
                    beta_input: vec![0.0; nv],
                    beta: vec![0.0; nv],
                    shifted: vec![0.0; nv * dw],
                    log_decay: vec![0.0; nv * dw],
                    gate: (0..nv * w)
                        .map(|index| self.projection[row * width + channels + index])
                        .collect(),
                    state: vec![StateRow::default(); nv * w],
                };
                for head in 0..2 * nk + nv {
                    let mut squares = 0.0;
                    for column in 0..w {
                        let channel = head * w + column;
                        let mut sum = self.convolution[channel * c + c - 1] as f64
                            * self.projection[row * width + channel] as f64;
                        let mut absolute = sum.abs();
                        for tap in 0..c - 1 {
                            let product = self.convolution[channel * c + tap] as f64
                                * input(row, local, tap, channel);
                            sum += product;
                            absolute += product.abs();
                        }
                        let value = sum / (1.0 + (-sum).exp());
                        trace.abs_convolution[channel] = absolute as f32;
                        trace.convolved[channel] = sum as f32;
                        trace.activated[channel] = value as f32;
                        prepared[channel] = value;
                        squares += value * value;
                    }
                    if head < 2 * nk {
                        let mut inverse = 1.0 / (squares + self.epsilon as f64).sqrt();
                        if head < nk {
                            inverse /= (w as f64).sqrt();
                        }
                        for column in 0..w {
                            prepared[head * w + column] *= inverse;
                        }
                    }
                }
                trace.prepared = prepared.iter().map(|v| *v as f32).collect();
                for value_head in 0..nv {
                    let key_head = key_of(value_head);
                    let beta_input = self.projection
                        [row * width + channels + nv * w + g.decays() + value_head]
                        as f64;
                    let beta = 1.0 / (1.0 + (-beta_input).exp());
                    trace.beta_input[value_head] = beta_input;
                    trace.beta[value_head] = beta;
                    for input in 0..dw {
                        let index = value_head * dw + input;
                        trace.shifted[index] = self.shifted(row, index);
                        trace.log_decay[index] = self.log_decay(row, value_head, index);
                    }
                    let factors = (0..dw)
                        .map(|input| trace.log_decay[value_head * dw + input].exp())
                        .collect::<Vec<_>>();
                    let query = &prepared[key_head * w..][..w];
                    let key = &prepared[(nk + key_head) * w..][..w];
                    let entry = (local >= slot.stop && local - slot.stop < taped).then(|| {
                        slot.following * g.tape_bank() + (local - slot.stop) * g.tape_row()
                    });
                    if let Some(entry) = entry {
                        for (input, factor) in factors.iter().enumerate() {
                            tape[entry + (nv + nk) * w + value_head * dw + input] = *factor as f32;
                        }
                        for column in 0..w {
                            tape[entry + nv * w + key_head * w + column] = key[column] as f32;
                        }
                    }
                    for state_row in 0..w {
                        let s = &mut state[(value_head * w + state_row) * w..][..w];
                        let before = norm_of(s);
                        let mut remembered = 0.0;
                        let mut absolute_key = 0.0;
                        for column in 0..w {
                            s[column] *= factors[column % dw];
                            remembered += s[column] * key[column];
                            absolute_key += (s[column] * key[column]).abs();
                        }
                        let decayed = norm_of(s);
                        let difference = prepared[(2 * nk + value_head) * w + state_row] - remembered;
                        let residual = difference * beta;
                        if let Some(entry) = entry {
                            tape[entry + value_head * w + state_row] = residual as f32;
                        }
                        let mut output = 0.0;
                        let mut absolute_query = 0.0;
                        for column in 0..w {
                            s[column] += residual * key[column];
                            output += s[column] * query[column];
                            absolute_query += (s[column] * query[column]).abs();
                        }
                        mixed[(row * nv + value_head) * w + state_row] = output as f32;
                        trace.state[value_head * w + state_row] = StateRow {
                            before: before as f32,
                            decayed: decayed as f32,
                            abs_k: absolute_key as f32,
                            residual: difference.abs() as f32,
                            after: norm_of(s) as f32,
                            abs_q: absolute_query as f32,
                            output: output.abs() as f32,
                        };
                    }
                }
                rows.push(trace);
                if local + 1 == slot.stop {
                    publish(&state, &mut delta);
                }
            }
            for row in 0..c - 1 + taped {
                for channel in 0..channels {
                    window[slot.following * g.window_bank() + row * channels + channel] = raw(
                        slot.stop as isize + row as isize - (c as isize - 1),
                        channel,
                    );
                }
            }
            first += slot.rows;
        }
        let trace = Trace {
            shape: Shape {
                key_heads: nk,
                value_heads: nv,
                width: w,
                decay_width: dw,
                convolution: c,
                tape: g.tape,
                banks: g.banks,
                grouped: self.grouped,
                activation: if self.bf16 { Float::Bf16 } else { Float::F32 },
                epsilon: self.epsilon as f64,
                rms_epsilon: self.rms_epsilon as f64,
                sigmoid_gate: self.gate_function == 1,
            },
            decay_lower: self.decay_lower.map(f64::from),
            rate: self.rate.iter().map(|r| *r as f64).collect(),
            norm: self.norm.iter().map(|n| *n as f64).collect(),
            slots,
            rows,
            mixed: mixed.clone(),
        };
        let outcome = Outcome {
            gated: self.gate(&mixed),
            window,
            delta,
            tape,
        };
        (outcome, trace)
    }

    /// The contract's gating of raw outputs `mixed` [M, NV, W] (f64 math,
    /// each rounding point of the body rounded to the case's activation).
    fn gate(&self, mixed: &[f32]) -> Vec<f32> {
        let g = self.geometry;
        let (nv, w) = (g.value_heads, g.width);
        let round = |value: f64| {
            if self.bf16 {
                bf16_round(value as f32)
            } else {
                value as f32
            }
        };
        let z = g.channels();
        let mut gated = vec![0.0f32; mixed.len()];
        for row in 0..self.rows {
            for head in 0..nv {
                let raw = &mixed[(row * nv + head) * w..][..w];
                let raw = raw
                    .iter()
                    .map(|value| round(*value as f64) as f64)
                    .collect::<Vec<_>>();
                let squares = raw.iter().map(|value| value * value).sum::<f64>();
                let inverse = 1.0 / (squares / w as f64 + self.rms_epsilon as f64).sqrt();
                for column in 0..w {
                    let gate =
                        self.projection[row * g.projection_width() + z + head * w + column] as f64;
                    let normalized = round(raw[column] * inverse * self.norm[column] as f64) as f64;
                    let numerator = if self.gate_function == 1 { 1.0 } else { gate };
                    let activated = round(numerator / (1.0 + (-gate).exp())) as f64;
                    gated[(row * nv + head) * w + column] = round(normalized * activated);
                }
            }
        }
        gated
    }

    /// The portable body executed by the Seismic interpreter.
    fn oracle(&self) -> Outcome {
        use seismic_lang::{
            checked::{SourceFile},
            entry::ElementBindings,
            failure::SourceTermination,
            interp::{Arg, Interpreter, OutcomeValue, TensorData},
            reference_math::ReferenceScalar,
            registry,
            types::DType,
        };
        let mut sources = seismic_std::sources();
        sources.push(SourceFile {
            path: "recurrent.seismic".into(),
            text: include_str!("../kernels/recurrent.seismic").into(),
        });
        sources.push(SourceFile {
            path: "functions.seismic".into(),
            text: include_str!("../kernels/functions.seismic").into(),
        });
        let module = crate::common::check(sources);
        let elements = ElementBindings::new()
            .bind("A", registry::dense(DType::F32))
            .bind("RN", registry::dense(DType::F32));
        let per_channel = self.per_channel();
        let entry = if per_channel { "kimi_delta_step" } else { "gated_delta_step" };
        let logical = module
            .entry(module.entry_named(entry).unwrap(), &elements)
            .unwrap();
        let g = self.geometry;
        let floats = |shape: Vec<usize>, values: &[f32]| {
            TensorData::dense(
                DType::F32,
                shape,
                values.iter().map(|value| f64::from(*value)).collect(),
            )
        };
        let ints = |values: Vec<i32>| {
            TensorData::dense(
                DType::I32,
                vec![values.len()],
                values.into_iter().map(f64::from).collect(),
            )
        };
        let mut interpreter = Interpreter::new(&logical);
        let mut tensors = if per_channel {
            let [projection, beta, gate, decay] = self.operands();
            vec![
                floats(vec![self.rows, g.channels()], &projection),
                floats(vec![self.rows, g.value_heads], &beta),
                floats(vec![self.rows, g.value_heads * g.width], &gate),
                floats(vec![self.rows, g.decays()], &decay),
            ]
        } else {
            vec![floats(vec![self.rows, g.projection_width()], &self.projection)]
        };
        // The window, delta and tape arguments' ordinals.
        let state = tensors.len() + 9;
        tensors.extend([
            floats(vec![g.channels(), g.convolution], &self.convolution),
            floats(vec![g.value_heads], &self.rate),
            floats(vec![g.decays()], &self.time_bias),
            floats(vec![g.width], &self.norm),
            TensorData::dense(
                DType::I32,
                vec![self.slots.len() + 1, 2],
                self.segments().into_iter().map(f64::from).collect(),
            ),
            ints(self.slots.iter().map(|slot| slot.stop as i32).collect()),
            ints(self.slots.iter().map(|slot| slot.previous as i32).collect()),
            ints(self.slots.iter().map(|slot| slot.taped as i32).collect()),
            ints(
                self.slots
                    .iter()
                    .map(|slot| slot.following as i32)
                    .collect(),
            ),
            floats(vec![g.banks, g.window_rows(), g.channels()], &self.window),
            floats(vec![g.banks, g.value_heads, g.width, g.width], &self.delta),
            floats(vec![g.banks, g.tape, g.tape_row()], &self.tape),
        ]);
        let mut arguments = tensors
            .into_iter()
            .map(|tensor| Arg::Tensor(interpreter.add_tensor(tensor)))
            .collect::<Vec<_>>();
        arguments.push(Arg::Scalar(ReferenceScalar::F32(self.epsilon.to_bits())));
        arguments.push(Arg::Scalar(ReferenceScalar::F32(
            self.rms_epsilon.to_bits(),
        )));
        if let Some(lower) = self.decay_lower {
            arguments.push(Arg::Scalar(ReferenceScalar::F32(lower.to_bits())));
        }
        arguments.push(Arg::Scalar(ReferenceScalar::Bool(self.grouped)));
        arguments.push(Arg::Scalar(ReferenceScalar::I32(self.gate_function)));
        arguments.push(Arg::Scalar(ReferenceScalar::U32(2)));
        let outcome = interpreter.run_bounded(&arguments, u64::MAX).unwrap();
        if let SourceTermination::Failed(failure) = outcome.termination() {
            panic!("portable recurrent body failed: {failure}");
        }
        let read = |reader: seismic_lang::interp::TensorReader<'_>| {
            (0..reader.element_count())
                .map(|index| reader.read(index).unwrap() as f32)
                .collect::<Vec<_>>()
        };
        let gated = match outcome.results().next().unwrap().value() {
            OutcomeValue::Tensor(reader) => read(reader),
            _ => panic!("gated output is a tensor"),
        };
        let mut window = None;
        let mut delta = None;
        let mut tape = None;
        for input in outcome.inputs() {
            match input.ordinal() {
                ordinal if ordinal == state => window = Some(read(input.tensor())),
                ordinal if ordinal == state + 1 => delta = Some(read(input.tensor())),
                ordinal if ordinal == state + 2 => tape = Some(read(input.tensor())),
                _ => {}
            }
        }
        Outcome {
            gated,
            window: window.expect("window is a mutable input"),
            delta: delta.expect("delta is a mutable input"),
            tape: tape.expect("tape is a mutable input"),
        }
    }

    /// Device tensors of the case in `activation`.
    fn tensors(&self, device: &Device, activation: Element) -> Tensors {
        self.tensors_with_reused_slab(device, activation, false)
    }

    fn tensors_with_reused_slab(
        &self,
        device: &Device,
        activation: Element,
        reuse: bool,
    ) -> Tensors {
        let g = self.geometry;
        let from_f32 = |shape: &[u64], values: &[f32]| {
            Tensor::from_host(
                device,
                Element::f32(),
                shape,
                &values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap()
        };
        let from_activation = |shape: &[u64], values: &[f32]| {
            let bytes = if activation == Element::bf16() {
                values
                    .iter()
                    .flat_map(|value| ((bf16_round(*value).to_bits() >> 16) as u16).to_le_bytes())
                    .collect::<Vec<_>>()
            } else {
                values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect()
            };
            Tensor::from_host(device, activation, shape, &bytes).unwrap()
        };
        let ints = |shape: &[u64], values: Vec<i32>| {
            Tensor::from_host(
                device,
                Element::i32(),
                shape,
                &values
                    .into_iter()
                    .flat_map(i32::to_le_bytes)
                    .collect::<Vec<_>>(),
            )
            .unwrap()
        };
        let slots = self.slots.len() as u64;
        let activation_bytes = |values: &[f32]| {
            if activation == Element::bf16() {
                values
                    .iter()
                    .flat_map(|value| ((bf16_round(*value).to_bits() >> 16) as u16).to_le_bytes())
                    .collect::<Vec<_>>()
            } else {
                values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>()
            }
        };
        let mut regions = vec![
            SlabRegion {
                element: activation,
                row_shape: vec![g.window_rows() as u64, g.channels() as u64],
            },
            SlabRegion {
                element: Element::f32(),
                row_shape: vec![g.value_heads as u64, g.width as u64, g.width as u64],
            },
        ];
        if g.tape > 0 {
            regions.push(SlabRegion {
                element: Element::f32(),
                row_shape: vec![g.tape as u64, g.tape_row() as u64],
            });
        }
        let slab_banks = 2u64;
        let mut slabs = SlabTensor::new(device, slab_banks, g.banks as u64, regions).unwrap();
        let mut contents = vec![
            activation_bytes(&self.window),
            self.delta
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
        ];
        if g.tape > 0 {
            contents.push(
                self.tape
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>(),
            );
        }
        let count = (g.banks as u64).div_ceil(slab_banks);
        for slab in 0..count {
            assert_eq!(slabs.add_slab().unwrap(), slab as usize);
        }
        if reuse {
            assert!(count >= 3);
            slabs.free_slab(1).unwrap();
            assert_eq!(slabs.add_slab().unwrap(), 1);
        }
        for slab in 0..count {
            let start = slab * slab_banks;
            let count = slab_banks.min(g.banks as u64 - start);
            for (region, bytes) in contents.iter().enumerate() {
                let row_bytes = bytes.len() / g.banks;
                slabs
                    .region_rows(region, start, count)
                    .unwrap()
                    .write_from_host(
                        &bytes[start as usize * row_bytes..(start + count) as usize * row_bytes],
                    )
                    .unwrap();
            }
        }
        let window = slabs.logical_region(0).unwrap();
        let delta = slabs.logical_region(1).unwrap();
        let tape = if g.tape > 0 {
            slabs.logical_region(2).unwrap()
        } else {
            from_f32(&[g.banks as u64, 0, g.tape_row() as u64], &[])
        };
        let rows = self.rows as u64;
        let inputs = match self.decay_lower {
            Some(lower) if self.per_channel() => {
                let [projection, beta, gate, decay] = self.operands();
                Inputs::Kimi {
                    projection: from_activation(&[rows, g.channels() as u64], &projection),
                    beta: from_activation(&[rows, g.value_heads as u64], &beta),
                    gate: from_activation(&[rows, (g.value_heads * g.width) as u64], &gate),
                    decay: from_activation(&[rows, g.decays() as u64], &decay),
                    lower,
                }
            }
            _ => Inputs::Gated {
                projection: from_activation(&[rows, g.projection_width() as u64], &self.projection),
            },
        };
        Tensors {
            inputs,
            convolution: from_f32(
                &[g.channels() as u64, g.convolution as u64],
                &self.convolution,
            ),
            rate: from_f32(&[g.value_heads as u64], &self.rate),
            time_bias: from_f32(&[g.decays() as u64], &self.time_bias),
            norm: from_f32(&[g.width as u64], &self.norm),
            segments: ints(&[slots + 1, 2], self.segments()),
            stop: ints(
                &[slots],
                self.slots.iter().map(|slot| slot.stop as i32).collect(),
            ),
            previous: ints(
                &[slots],
                self.slots.iter().map(|slot| slot.previous as i32).collect(),
            ),
            previous_tape: ints(
                &[slots],
                self.slots.iter().map(|slot| slot.taped as i32).collect(),
            ),
            following: ints(
                &[slots],
                self.slots
                    .iter()
                    .map(|slot| slot.following as i32)
                    .collect(),
            ),
            _slabs: slabs,
            window,
            delta,
            tape,
            slab_banks: slab_banks as u32,
        }
    }

    /// The specialization with `ROWS` state rows per threadgroup (Metal) or
    /// work item (CPU, whose implementations have no static dimensions).
    fn specialization(&self, device: &Device, rows: u64) -> NativeSpecialization {
        if is_cpu(device) {
            return NativeSpecialization::new().with_param("ROWS", rows);
        }
        let g = self.geometry;
        NativeSpecialization::new()
            .with_static("NK", g.key_heads as u64)
            .with_static("NV", g.value_heads as u64)
            .with_static("W", g.width as u64)
            .with_static("C", g.convolution as u64)
            .with_param("ROWS", rows)
    }

    /// The step with `ROWS` state rows per threadgroup or work item.
    fn native_step(
        &self,
        device: &Device,
        activation: Element,
        rows: u64,
    ) -> seismic::NativeKernel<gated_delta_step::Entry> {
        gated_delta_step::native_for_device_with(
            device,
            gated_delta_step::Elements {
                RN: Element::f32(),
                A: activation,
            },
            &self.specialization(device, rows),
        )
        .unwrap()
    }

    /// The step (`None`) or the chunk with `ROWS` (`Some`).
    fn native(&self, device: &Device, activation: Element, chunk: Option<u64>) -> Outcome {
        self.native_with_reused_slab(device, activation, chunk, false)
    }

    fn native_with_reused_slab(
        &self,
        device: &Device,
        activation: Element,
        chunk: Option<u64>,
        reuse: bool,
    ) -> Outcome {
        let mut t = self.tensors_with_reused_slab(device, activation, reuse);
        let rows = match chunk {
            // Per-channel decay at ROWS == W exceeds the step's
            // threadgroup memory at the small widths.
            None if self.per_channel() => 32.min(self.geometry.width as u64 / 2),
            None => 32.min(self.geometry.width as u64),
            Some(mapping) => mapping,
        };
        let gated = t.run(device, self, activation, chunk.is_some(), &self.specialization(device, rows));
        Outcome {
            gated: read(&gated),
            window: read(&t.window),
            delta: read(&t.delta),
            tape: read(&t.tape),
        }
    }
}

fn read(tensor: &Tensor) -> Vec<f32> {
    let bytes = tensor.read_to_host().unwrap();
    if tensor.element() == Element::bf16() {
        bytes
            .chunks_exact(2)
            .map(|word| {
                f32::from_bits(u32::from(u16::from_le_bytes(word.try_into().unwrap())) << 16)
            })
            .collect()
    } else {
        bytes
            .chunks_exact(4)
            .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
            .collect()
    }
}

/// The entry whose output is `actual`, and the one whose output `expected`
/// is (`None`: the f64 host's), for the bounds of their difference.
#[derive(Clone, Copy)]
struct Pair {
    actual: Option<Native>,
    expected: Option<Native>,
}

/// The output of `actual`, compared with the f64 host model.
fn against_host(actual: Native) -> Pair {
    Pair { actual: Some(actual), expected: None }
}

/// The outputs of two entries, compared with each other.
fn between(actual: Native, expected: Native) -> Pair {
    Pair { actual: Some(actual), expected: Some(expected) }
}

/// The f64 host model's output, compared with the portable body's.
fn host_against_body() -> Pair {
    Pair { actual: None, expected: Some(ORACLE) }
}

/// `device`'s step entry and its chunk entry.
fn step_of(device: &Device) -> Native {
    Native::Step(device.backend())
}

fn chunk_of(device: &Device) -> Native {
    Native::Chunk(device.backend())
}

/// The portable body, executed by the interpreter in F32 on the CPU's
/// library.
const ORACLE: Native = Native::Step(BackendName::Cpu);

/// Compare one outcome with the reference, each element within the sum of
/// the two entries' bounds from the f64 host ([`recurrent_bound`]): the gated
/// outputs, the successor's state and recorded tape rows. Windows are copies
/// and must be bit-exact; every non-successor bank must be untouched.
fn check(label: &str, case: &Case, actual: &Outcome, expected: &Outcome, pair: Pair) {
    let g = case.geometry;
    let bounds = match (pair.actual.map(|n| case.bounds(n)), pair.expected.map(|n| case.bounds(n))) {
        (Some(actual), Some(expected)) => std::rc::Rc::new(actual.plus(&expected)),
        (Some(bounds), None) | (None, Some(bounds)) => bounds,
        (None, None) => panic!("two host outputs have no bound to compare"),
    };
    let within = |what: &str, actual: &[f32], expected: &[f32], bounds: &dyn Fn(usize) -> f64| {
        assert_eq!(actual.len(), expected.len());
        let mut used = 0.0f64;
        for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!(a.is_finite(), "{label} {what}[{index}] is not finite");
            let (difference, bound) = ((*a as f64 - *e as f64).abs(), bounds(index));
            used = used.max(difference / bound.max(f64::MIN_POSITIVE));
            assert!(
                difference <= bound,
                "{label} {what}[{index}]: {a} vs {e} differ by {difference:e}, bound {bound:e}"
            );
        }
        println!("{label} {what}: largest difference is {used:.3} of its bound");
    };
    within("gated", &actual.gated, &expected.gated, &|index| bounds.gated[index]);
    for bank in 0..g.banks {
        let window = &actual.window[bank * g.window_bank()..][..g.window_bank()];
        let delta = &actual.delta[bank * g.delta_bank()..][..g.delta_bank()];
        let expected_window = &expected.window[bank * g.window_bank()..][..g.window_bank()];
        let expected_delta = &expected.delta[bank * g.delta_bank()..][..g.delta_bank()];
        assert!(
            window
                .iter()
                .zip(expected_window)
                .all(|(a, e)| a.to_bits() == e.to_bits()),
            "{label} window bank {bank} differs"
        );
        let tape = &actual.tape[bank * g.tape_bank()..][..g.tape_bank()];
        let original_tape = &case.tape[bank * g.tape_bank()..][..g.tape_bank()];
        // Tape rows the successor records: after its stop row, at most T.
        let recorded = case
            .slots
            .iter()
            .find(|slot| slot.following == bank)
            .map(|slot| g.tape.min(slot.rows - slot.stop) * g.tape_row());
        if let Some(recorded) = recorded {
            let state = bounds.delta[bank].as_ref().expect("a successor's state bound");
            within(&format!("delta bank {bank}"), delta, expected_delta, &|index| {
                state[index / g.width]
            });
            if recorded > 0 {
                let recorded_bounds = bounds.tape[bank].as_ref().expect("a successor's tape bound");
                within(
                    &format!("tape bank {bank}"),
                    &tape[..recorded],
                    &expected.tape[bank * g.tape_bank()..][..recorded],
                    &|index| recorded_bounds[index],
                );
            }
            assert!(
                tape[recorded..]
                    .iter()
                    .zip(&original_tape[recorded..])
                    .all(|(a, e)| a.to_bits() == e.to_bits()),
                "{label} wrote tape rows of bank {bank} past its recorded rows"
            );
        } else {
            let original = &case.delta[bank * g.delta_bank()..][..g.delta_bank()];
            assert!(
                delta
                    .iter()
                    .zip(original)
                    .all(|(a, e)| a.to_bits() == e.to_bits())
                    && tape
                        .iter()
                        .zip(original_tape)
                        .all(|(a, e)| a.to_bits() == e.to_bits()),
                "{label} wrote bank {bank}, which is not a successor"
            );
        }
    }
}

/// What a CPU model of one form leaves: the raw outputs [M, NV, W] and each
/// slot's published state.
struct Modelled {
    raw: Vec<f32>,
    published: Vec<Option<Vec<f32>>>,
}

/// A CPU model of `native`'s arithmetic in F32 (the WY form's pieces of
/// `METAL_CHUNK_PIECE` rows for Metal, of `CUDA_CHUNK_PIECE` with f16 operands
/// for CUDA), run on the f64 host's own
/// prepared inputs, rounded to F32: where a form's bound
/// ([`recurrent_bound`]) is most likely to be wrong, in the algebra of its
/// pieces, it is checked on the CPU before any device runs.
fn model(case: &Case, native: Native) -> Modelled {
    use common::bound::Float;
    let g = case.geometry;
    let (nk, nv, w, dw) = (g.key_heads, g.value_heads, g.width, g.decay_width);
    let trace = case.trace();
    let (piece, f16) = match native {
        Native::Chunk(BackendName::Metal) => (METAL_CHUNK_PIECE, false),
        Native::Chunk(BackendName::Cuda) => (CUDA_CHUNK_PIECE, true),
        _ => (0, false),
    };
    let single = |x: f32| if f16 { Float::F16.round(x as f64) as f32 } else { x };
    let split = |x: f32| {
        if f16 {
            let high = single(x);
            high + single(x - high)
        } else {
            x
        }
    };
    let mut raw = vec![0.0f32; case.rows * nv * w];
    let mut published = vec![None; g.banks];
    let mut first = 0;
    for (slot, slot_trace) in case.slots.iter().zip(&trace.slots) {
        let chunked = if piece > 0 && slot.rows > SEQUENTIAL_ROWS && slot.stop > 0 { slot.stop } else { 0 };
        let mut bank = vec![0.0f32; nv * w * w];
        for head in 0..nv {
            let key_head = if case.grouped { head * nk / nv } else { head % nk };
            let prepared = |row: usize| &trace.rows[first + row].prepared;
            let q = |row: usize, c: usize| prepared(row)[key_head * w + c];
            let k = |row: usize, c: usize| prepared(row)[(nk + key_head) * w + c];
            let beta = |row: usize| trace.rows[first + row].beta[head] as f32;
            let log_decay = |row: usize, j: usize| trace.rows[first + row].log_decay[head * dw + j] as f32;
            let mut states: Vec<Vec<f32>> = (0..w)
                .map(|i| case.delta[slot.previous * g.delta_bank() + (head * w + i) * w..][..w].to_vec())
                .collect();
            for entry in 0..slot.taped {
                let row = &case.tape[slot.previous * g.tape_bank() + entry * g.tape_row()..][..g.tape_row()];
                let decay = &row[(nv + nk) * w + head * dw..][..dw];
                let key = &row[nv * w + key_head * w..][..w];
                for (i, s) in states.iter_mut().enumerate() {
                    for c in 0..w {
                        s[c] = row[head * w + i].mul_add(key[c], s[c] * decay[c % dw]);
                    }
                }
            }
            if slot.stop == 0 {
                for (i, s) in states.iter().enumerate() {
                    bank[(head * w + i) * w..][..w].copy_from_slice(s);
                }
            }
            // The WY pieces of the chunked rows.
            let mut start = 0;
            while start < chunked {
                let n = piece.min(chunked - start);
                let mut cumulative = vec![vec![0.0f32; dw]; n];
                for t in 0..n {
                    for j in 0..dw {
                        let before = if t == 0 { 0.0 } else { cumulative[t - 1][j] };
                        cumulative[t][j] = before + log_decay(start + t, j);
                    }
                }
                let last = |j: usize| cumulative[n - 1][j];
                let pair = |t: usize, s: usize, c: usize| (cumulative[t][c % dw] - cumulative[s][c % dw]).exp();
                let mut a = vec![vec![0.0f32; n]; n];
                let mut d = vec![vec![0.0f32; n]; n];
                for t in 0..n {
                    for s in 0..=t {
                        let (mut kk, mut qk) = (0.0f32, 0.0f32);
                        for c in 0..w {
                            let factor = pair(t, s, c);
                            kk = (k(start + t, c) * k(start + s, c)).mul_add(factor, kk);
                            qk = (q(start + t, c) * k(start + s, c)).mul_add(factor, qk);
                        }
                        d[t][s] = single(qk);
                        if s < t {
                            a[t][s] = beta(start + t) * kk;
                        }
                    }
                }
                // T = (I + A)^-1 by forward substitution, column by column;
                // Tb = T diag(beta).
                let mut inverse = vec![vec![0.0f32; n]; n];
                for column in 0..n {
                    for row in column..n {
                        let mut value = if row == column { 1.0f32 } else { 0.0 };
                        for k_index in column..row {
                            value -= a[row][k_index] * inverse[k_index][column];
                        }
                        inverse[row][column] = value;
                    }
                }
                let tb: Vec<Vec<f32>> = (0..n)
                    .map(|row| (0..n).map(|column| inverse[row][column] * beta(start + column)).collect())
                    .collect();
                let gamma_of = |t: usize, c: usize| cumulative[t][c % dw].exp();
                let tail_of = |t: usize, c: usize| (last(c % dw) - cumulative[t][c % dw]).exp();
                for i in 0..w {
                    let s = &mut states[i];
                    let mut z = vec![0.0f32; n];
                    let mut y = vec![0.0f32; n];
                    for t in 0..n {
                        let (mut x, mut yy) = (0.0f32, 0.0f32);
                        for c in 0..w {
                            let scale = if dw == 1 { 1.0 } else { gamma_of(t, c) };
                            x = (split(s[c]) * split(k(start + t, c) * scale)).mul_add(1.0, x);
                            yy = (single(s[c]) * single(q(start + t, c) * scale)).mul_add(1.0, yy);
                        }
                        let outer = if dw == 1 { gamma_of(t, 0) } else { 1.0 };
                        z[t] = single(prepared(start + t)[2 * nk * w + head * w + i]) - x * outer;
                        y[t] = yy * outer;
                    }
                    let u: Vec<f32> = (0..n)
                        .map(|t| (0..=t).fold(0.0f32, |sum, r| split(z[r]).mul_add(single(tb[t][r]), sum)))
                        .collect();
                    for t in 0..n {
                        let out = (0..=t).fold(y[t], |sum, r| single(d[t][r]).mul_add(single(u[r]), sum));
                        raw[((first + start + t) * nv + head) * w + i] = out;
                    }
                    for c in 0..w {
                        let mut next = s[c] * last(c % dw).exp();
                        for r in 0..n {
                            let tail = tail_of(r, c);
                            let (weight, key) = if dw == 1 {
                                (split(u[r] * tail), split(k(start + r, c)))
                            } else {
                                (split(u[r]), split(k(start + r, c) * tail))
                            };
                            next = weight.mul_add(key, next);
                        }
                        s[c] = next;
                    }
                }
                start += n;
            }
            if chunked > 0 && chunked == slot.stop {
                for (i, s) in states.iter().enumerate() {
                    bank[(head * w + i) * w..][..w].copy_from_slice(s);
                }
            }
            // The row-sequential rows.
            for local in chunked..slot.rows {
                let factors: Vec<f32> = (0..dw).map(|j| log_decay(local, j).exp()).collect();
                for (i, s) in states.iter_mut().enumerate() {
                    let mut remembered = 0.0f32;
                    for c in 0..w {
                        s[c] *= factors[c % dw];
                        remembered = s[c].mul_add(k(local, c), remembered);
                    }
                    let residual = (prepared(local)[2 * nk * w + head * w + i] - remembered) * beta(local);
                    let mut output = 0.0f32;
                    for c in 0..w {
                        s[c] = residual.mul_add(k(local, c), s[c]);
                        output = q(local, c).mul_add(s[c], output);
                    }
                    raw[((first + local) * nv + head) * w + i] = output;
                }
                if local + 1 == slot.stop {
                    for (i, s) in states.iter().enumerate() {
                        bank[(head * w + i) * w..][..w].copy_from_slice(s);
                    }
                }
            }
        }
        published[slot_trace.following] = Some(bank);
        first += slot.rows;
    }
    Modelled { raw, published }
}

/// Every form's bound holds for a CPU model of its arithmetic on a spread of
/// cases: the step's, Metal's F32 WY form's and CUDA's f16-operand form's,
/// raw outputs and published state, against the f64 host. Prints how much of
/// each bound the model uses.
#[test]
fn bounds_hold_for_a_cpu_model_of_each_form() {
    let kimi = Geometry { decay_width: SMALL.width, banks: 7, tape: 3, ..SMALL };
    let version = |rows, stop, previous, following, taped| SlotCase { rows, stop, previous, following, taped };
    let mut cases: Vec<(String, Case)> = small_cases()
        .into_iter()
        .chain(tape_cases())
        .map(|(label, case)| (label.to_string(), case))
        .collect();
    cases.push((
        "per-channel decay, chunked slots from tape versions".to_string(),
        Case::new(
            kimi,
            64,
            vec![version(5, 2, 1, 4, 2), version(43, 37, 2, 5, 3), version(9, 9, 0, 6, 0)],
            false,
            61,
        )
        .with_bounded_decay(),
    ));
    for (label, case) in &cases {
        let host = case.host();
        let g = case.geometry;
        for native in [
            Native::Step(BackendName::Cpu),
            Native::Chunk(BackendName::Metal),
            Native::Chunk(BackendName::Cuda),
        ] {
            let modelled = model(case, native);
            let bounds = case.bounds(native);
            let (mut raw_used, mut state_used) = (0.0f64, 0.0f64);
            // The state's worst error against its RMS, the measure the
            // earlier hand bounds were set in, and the bound's worst.
            let (mut worst, mut allowed, mut squares, mut count) = (0.0f64, 0.0f64, 0.0f64, 0usize);
            for (index, (a, e)) in modelled.raw.iter().zip(&case.trace().mixed).enumerate() {
                let difference = (*a as f64 - *e as f64).abs();
                raw_used = raw_used.max(difference / bounds.raw[index].max(f64::MIN_POSITIVE));
                assert!(
                    difference <= bounds.raw[index],
                    "{label} {native:?}: raw[{index}] {a} vs {e}, bound {:e}",
                    bounds.raw[index]
                );
            }
            for (bank, state) in modelled.published.iter().enumerate() {
                let Some(state) = state else { continue };
                let expected = &host.delta[bank * g.delta_bank()..][..g.delta_bank()];
                let limit = bounds.delta[bank].as_ref().unwrap();
                for (index, (a, e)) in state.iter().zip(expected).enumerate() {
                    let difference = (*a as f64 - *e as f64).abs();
                    state_used = state_used.max(difference / limit[index / g.width].max(f64::MIN_POSITIVE));
                    worst = worst.max(difference);
                    allowed = allowed.max(limit[index / g.width]);
                    squares += (*e as f64).powi(2);
                    count += 1;
                    assert!(
                        difference <= limit[index / g.width],
                        "{label} {native:?}: state[{index}] of bank {bank} {a} vs {e}, bound {:e}",
                        limit[index / g.width]
                    );
                }
            }
            let rms = (squares / count.max(1) as f64).sqrt();
            println!(
                "{label} {native:?}: raw uses {raw_used:.3} of its bound, state {state_used:.3}; \
                 state error / state rms {:.2e}, bound / state rms {:.2e}",
                worst / rms,
                allowed / rms
            );
        }
    }
}

fn metal() -> Option<Device> {
    seismic::Devices::discover(&[])
        .ok()
        .and_then(|catalog| catalog.open_backend(BackendName::Metal).ok())
}

/// Metal when present and the CPU device. (Vulkan's mappings are checked in
/// `vulkan_recurrent.rs`, CUDA's in `cuda_recurrent.rs`.)
fn devices() -> Vec<Device> {
    let catalog = seismic::Devices::discover(&[]).unwrap();
    catalog
        .open_backend(BackendName::Metal)
        .ok()
        .into_iter()
        .chain(std::iter::once(
            catalog.open_backend(BackendName::Cpu).unwrap(),
        ))
        .collect()
}

fn is_cpu(device: &Device) -> bool {
    device.backend() == BackendName::Cpu
}

/// The chunk's `ROWS` mappings on Metal and on the CPU.
const CHUNK_ROWS: [u64; 3] = [128, 64, 32];

const SMALL: Geometry = Geometry {
    key_heads: 2,
    value_heads: 4,
    width: 32,
    decay_width: 1,
    convolution: 4,
    banks: 6,
    tape: 0,
};

fn small_cases() -> Vec<(&'static str, Case)> {
    vec![
        (
            "one row from the zero seed",
            Case::new(SMALL, 1, vec![slot(1, 1, 0, 3)], false, 1),
        ),
        (
            "short slots shorter than the window, padded rows",
            Case::new(
                SMALL,
                8,
                vec![slot(2, 2, 1, 4), slot(1, 1, 2, 5), slot(3, 3, 0, 3)],
                true,
                2,
            ),
        ),
        (
            "interior stop rows and a zero stop",
            Case::new(
                SMALL,
                16,
                vec![slot(7, 4, 1, 3), slot(5, 0, 2, 4), slot(4, 1, 1, 5)],
                false,
                3,
            ),
        ),
        (
            "pieces across slots with a stop inside a piece and resets",
            Case::new(
                SMALL,
                128,
                vec![slot(70, 33, 1, 4), slot(45, 45, 2, 3)],
                true,
                4,
            )
            .with_resets(&[5, 40, 71, 100]),
        ),
        (
            "sigmoid output gate over short and chunked slots",
            Case::new(
                SMALL,
                64,
                vec![slot(3, 3, 1, 4), slot(40, 33, 2, 5)],
                true,
                5,
            )
            .with_sigmoid_gate(),
        ),
    ]
}

#[test]
fn step_and_chunk_match_the_portable_body() {
    for device in devices() {
        step_and_chunk_match_the_portable_body_on(&device);
    }
}

fn step_and_chunk_match_the_portable_body_on(device: &Device) {
    let backend = device.backend().as_str();
    for (label, case) in small_cases() {
        let label = format!("{backend} {label}");
        let oracle = case.oracle();
        let host = case.host();
        // The f64 host model and the interpreted body agree closely; this
        // pins the host model used for the real geometry below.
        check(
            &format!("{label}: host vs body"),
            &case,
            &host,
            &oracle,
            host_against_body(),
        );
        let step = case.native(device, Element::f32(), None);
        check(
            &format!("{label}: step"),
            &case,
            &step,
            &oracle,
            between(step_of(device), ORACLE),
        );
        // One mapping per distinct kernel path: ROWS never changes result bits
        // (`step_row_block_never_changes_bits_and_stop_equals_a_shorter_run`).
        for rows in CHUNK_ROWS.into_iter().take(1) {
            let rows = rows.min(case.geometry.width as u64);
            let chunked = case.native(device, Element::f32(), Some(rows));
            check(
                &format!("{label}: chunk ROWS {rows}"),
                &case,
                &chunked,
                &oracle,
                between(chunk_of(device), ORACLE),
            );
        }
    }
}

/// Small cases over tape versions (T = 3): slots that start from a version
/// with tape rows, record the rows after their stop row (fewer, exactly, or
/// more than T), in short and chunked slots, grouped or not.
fn tape_cases() -> Vec<(&'static str, Case)> {
    let geometry = Geometry {
        banks: 7,
        tape: 3,
        ..SMALL
    };
    let version = |rows, stop, previous, following, taped| SlotCase {
        rows,
        stop,
        previous,
        following,
        taped,
    };
    vec![
        (
            "verify slots from tape versions",
            Case::new(
                geometry,
                16,
                vec![
                    version(4, 1, 1, 4, 2),
                    version(6, 2, 2, 5, 0),
                    version(3, 3, 3, 6, 3),
                ],
                false,
                51,
            ),
        ),
        (
            "chunked slots with tails, from tape versions",
            Case::new(
                geometry,
                64,
                vec![version(40, 35, 1, 4, 1), version(20, 20, 2, 5, 3)],
                true,
                52,
            )
            .with_resets(&[7, 45]),
        ),
    ]
}

#[test]
fn slab_tape_bank_crossing_matches_host_model() {
    for device in devices() {
        check_slab_tape_bank_crossing(&device);
    }
}

fn check_slab_tape_bank_crossing(device: &Device) {
    let (_, case) = tape_cases().remove(0);
    let host = case.host();
    let backend = device.backend().as_str();
    let step = case.native(device, Element::f32(), None);
    check(
        &format!("{backend} slab tape step"),
        &case,
        &step,
        &host,
        against_host(step_of(device)),
    );
    let chunk = case.native(device, Element::f32(), Some(32));
    check(
        &format!("{backend} slab tape chunk"),
        &case,
        &chunk,
        &host,
        against_host(chunk_of(device)),
    );
}

#[test]
fn reused_interior_bank_slab_matches_host_model() {
    for device in devices() {
        check_reused_interior_bank_slab(&device);
    }
}

fn check_reused_interior_bank_slab(device: &Device) {
    let (_, case) = tape_cases().remove(0);
    let host = case.host();
    let backend = device.backend().as_str();
    let step = case.native_with_reused_slab(device, Element::f32(), None, true);
    check(
        &format!("{backend} reused bank step"),
        &case,
        &step,
        &host,
        against_host(step_of(device)),
    );
    let chunk = case.native_with_reused_slab(device, Element::f32(), Some(32), true);
    check(
        &format!("{backend} reused bank chunk"),
        &case,
        &chunk,
        &host,
        against_host(chunk_of(device)),
    );
}

/// A committed tape version (bank, j) equals a run that published after those
/// rows: the next advance from either has the same bits. `form` shapes each
/// case's inputs (its activations and decay), `run` executes a case on one
/// entry; the tentative advance is a 5-row verify slot (stop 1).
fn tape_versions_equal_stopped_runs(
    label: &str,
    geometry: Geometry,
    form: &dyn Fn(Case) -> Case,
    native: Native,
    run: &dyn Fn(&Case) -> Outcome,
) {
    let verify = 5;
    let g = Geometry {
        banks: 4,
        tape: verify - 1,
        ..geometry
    };
    for accepted in 0..verify {
        let tentative = form(Case::new(g, verify, vec![slot(verify, 1, 1, 2)], true, 41));
        let first = run(&tentative);
        let next = SlotCase {
            rows: 3,
            stop: 3,
            previous: 2,
            following: 3,
            taped: accepted,
        };
        let continued = tentative.continued(&first, 3, vec![next], 42);
        let from_tape = run(&continued);
        let stopped = form(Case::new(g, verify, vec![slot(verify, 1 + accepted, 1, 2)], true, 41));
        let reference_first = run(&stopped);
        let reference = stopped.continued(&reference_first, 3, vec![slot(3, 3, 2, 3)], 42);
        let expected = run(&reference);
        let bank = |values: &[f32], size: usize| values[3 * size..4 * size].to_vec();
        assert!(
            from_tape
                .gated
                .iter()
                .zip(&expected.gated)
                .all(|(a, b)| a.to_bits() == b.to_bits())
                && bank(&from_tape.delta, g.delta_bank()) == bank(&expected.delta, g.delta_bank())
                && bank(&from_tape.window, g.window_bank())
                    == bank(&expected.window, g.window_bank()),
            "{label}: version (bank, {accepted}) differs from a run that stopped after {} rows",
            1 + accepted
        );
        check(
            &format!("{label}: from version (bank, {accepted})"),
            &continued,
            &from_tape,
            &continued.host(),
            against_host(native),
        );
    }
}

#[test]
fn tape_cases_match_the_portable_body() {
    for device in devices() {
        tape_cases_match_the_portable_body_on(&device);
    }
}

fn tape_cases_match_the_portable_body_on(device: &Device) {
    let backend = device.backend().as_str();
    for (label, case) in tape_cases() {
        let label = format!("{backend} {label}");
        let oracle = case.oracle();
        check(
            &format!("{label}: host vs body"),
            &case,
            &case.host(),
            &oracle,
            host_against_body(),
        );
        check(
            &format!("{label}: step"),
            &case,
            &case.native(device, Element::f32(), None),
            &oracle,
            between(step_of(device), ORACLE),
        );
        // One mapping per distinct kernel path: ROWS never changes result bits
        // (`step_row_block_never_changes_bits_and_stop_equals_a_shorter_run`).
        for rows in CHUNK_ROWS.into_iter().take(1) {
            let rows = rows.min(case.geometry.width as u64);
            let chunked = case.native(device, Element::f32(), Some(rows));
            check(
                &format!("{label}: chunk ROWS {rows}"),
                &case,
                &chunked,
                &oracle,
                between(chunk_of(device), ORACLE),
            );
        }
    }
}

#[test]
fn tape_versions_equal_runs_that_stopped_there() {
    for device in devices() {
        tape_versions_equal_runs_that_stopped_there_on(&device);
    }
}

fn tape_versions_equal_runs_that_stopped_there_on(device: &Device) {
    let qwen = Geometry {
        key_heads: 16,
        value_heads: 32,
        width: 128,
        decay_width: 1,
        convolution: 4,
        banks: 4,
        tape: 0,
    };
    let backend = device.backend().as_str();
    let per_head = |case: Case| case.with_bf16_activations();
    // Kimi Delta Attention: a decay per key channel, replayed from the tape.
    let per_channel = |case: Case| case.with_bf16_activations().with_bounded_decay();
    let forms: [(&str, Geometry, &dyn Fn(Case) -> Case); 3] =
        [("", SMALL, &per_head), ("", qwen, &per_head), (" KDA", kda(4, 0), &per_channel)];
    for (name, geometry, form) in forms {
        tape_versions_equal_stopped_runs(&format!("{backend}{name} step"), geometry, form, step_of(device), &|case| {
            case.native(device, Element::bf16(), None)
        });
        tape_versions_equal_stopped_runs(&format!("{backend}{name} chunk"), geometry, form, chunk_of(device), &|case| {
            case.native(device, Element::bf16(), Some(32))
        });
    }
}

#[test]
fn step_row_block_never_changes_bits_and_stop_equals_a_shorter_run() {
    for device in devices() {
        step_row_block_never_changes_bits_and_stop_equals_a_shorter_run_on(&device);
    }
}

fn step_row_block_never_changes_bits_and_stop_equals_a_shorter_run_on(device: &Device) {
    let slot = |rows, stop| slot(rows, stop, 1, 2);
    let full = Case::new(SMALL, 6, vec![slot(6, 3)], false, 11);
    let mut reference = None;
    for rows in [16u64, 32] {
        let mut t = full.tensors(device, Element::f32());
        let gated = full
            .native_step(device, Element::f32(), rows)
            .call(t.step_args(&full))
            .unwrap()
            .value;
        let outcome = (read(&gated), read(&t.window), read(&t.delta));
        match &reference {
            None => reference = Some(outcome),
            Some(reference) => assert!(
                reference
                    .0
                    .iter()
                    .zip(&outcome.0)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
                    && reference.2 == outcome.2,
                "ROWS={rows} changed result bits"
            ),
        }
    }
    // Publishing after row 3 of 6 equals running only the first 3 rows.
    let (_, full_window, full_delta) = reference.unwrap();
    let mut prefix = Case::new(SMALL, 6, vec![slot(6, 3)], false, 11);
    prefix.rows = 3;
    prefix.slots = vec![slot(3, 3)];
    prefix.projection.truncate(3 * SMALL.projection_width());
    let short = prefix.native(device, Element::f32(), None);
    assert!(
        short
            .delta
            .iter()
            .zip(&full_delta)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "stop-row state differs from the state of a shorter run"
    );
    assert!(
        short
            .window
            .iter()
            .zip(&full_window)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "stop-row window differs from the window of a shorter run"
    );
}

#[test]
fn real_4b_geometry_step_and_chunk_agree_with_the_host_model() {
    for device in devices() {
        real_4b_geometry_step_and_chunk_agree_with_the_host_model_on(&device);
    }
}

fn real_4b_geometry_step_and_chunk_agree_with_the_host_model_on(device: &Device) {
    let backend = device.backend().as_str();
    let geometry = Geometry {
        key_heads: 16,
        value_heads: 32,
        width: 128,
        decay_width: 1,
        convolution: 4,
        banks: 9,
        tape: 0,
    };
    for (label, rows, slots) in [
        ("decode, one slot", 1, vec![slot(1, 1, 1, 3)]),
        (
            "decode, four one-row slots",
            4,
            vec![
                slot(1, 1, 1, 5),
                slot(1, 1, 2, 6),
                slot(1, 1, 3, 7),
                slot(1, 1, 4, 8),
            ],
        ),
        (
            "verify, two slots",
            8,
            vec![slot(4, 2, 1, 3), slot(3, 3, 0, 4)],
        ),
        (
            "eight rows, three slots and a padded row",
            8,
            vec![slot(3, 1, 1, 5), slot(2, 2, 2, 6), slot(2, 0, 0, 7)],
        ),
        ("prefill 128", 128, vec![slot(128, 128, 1, 3)]),
        (
            "prefill 512, two slots",
            512,
            vec![slot(300, 211, 1, 3), slot(212, 212, 2, 4)],
        ),
    ] {
        let label = format!("{backend} {label}");
        let case = Case::new(geometry, rows, slots, false, 21).with_bf16_activations();
        let host = case.host();
        let step = case.native(device, Element::bf16(), None);
        check(
            &format!("4B {label}: step"),
            &case,
            &step,
            &host,
            against_host(step_of(device)),
        );
        if rows > SEQUENTIAL_ROWS {
            let mut reference: Option<Outcome> = None;
            for rows in CHUNK_ROWS {
                let chunked = case.native(device, Element::bf16(), Some(rows));
                check(
                    &format!("4B {label}: chunk ROWS {rows}"),
                    &case,
                    &chunked,
                    &host,
                    against_host(chunk_of(device)),
                );
                check(
                    &format!("4B {label}: chunk ROWS {rows} vs step"),
                    &case,
                    &chunked,
                    &step,
                    between(chunk_of(device), step_of(device)),
                );
                // ROWS never changes result bits.
                match &reference {
                    Some(reference) => assert!(
                        reference
                            .gated
                            .iter()
                            .zip(&chunked.gated)
                            .all(|(a, b)| a.to_bits() == b.to_bits())
                            && reference
                                .delta
                                .iter()
                                .zip(&chunked.delta)
                                .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "4B {label}: chunk ROWS {rows} changed result bits"
                    ),
                    None => reference = Some(chunked),
                }
            }
        }
    }
}

/// MTP verify: a slot of at most 16 rows gets the step's state bits from the
/// chunk entry too, whatever its peers (here a 40-row slot on the chunked
/// path), so a request's verify rows never depend on the class. The gating
/// reduces each head's squares in its class's order (Metal: the step's lane
/// butterfly, the chunk's strided simd sum; the CPU: one order), so the gated
/// rows agree to two BF16 ulps, and bit for bit where the orders agree.
#[test]
fn chunk_short_slots_get_the_step_bits() {
    for device in devices() {
        chunk_short_slots_get_the_step_bits_on(&device);
    }
}

fn chunk_short_slots_get_the_step_bits_on(device: &Device) {
    let backend = device.backend().as_str();
    let geometry = Geometry {
        key_heads: 16,
        value_heads: 32,
        width: 128,
        decay_width: 1,
        convolution: 4,
        banks: 11,
        tape: 0,
    };
    let slots = vec![
        slot(4, 1, 1, 6),
        slot(SEQUENTIAL_ROWS, 9, 2, 7),
        slot(40, 40, 3, 8),
        slot(1, 1, 4, 9),
        slot(7, 0, 5, 10),
    ];
    let case = Case::new(geometry, 70, slots, false, 31).with_bf16_activations();
    let step = case.native(device, Element::bf16(), None);
    let host = case.host();
    let row_elements = geometry.value_heads * geometry.width;
    // The two classes' gating orders each lie within their bound of the host.
    let differences = case.bounds(step_of(device)).plus(&case.bounds(chunk_of(device)));
    for rows in CHUNK_ROWS {
        let chunk = case.native(device, Element::bf16(), Some(rows));
        let mut first = 0;
        for s in &case.slots {
            let range = first * row_elements..(first + s.rows) * row_elements;
            first += s.rows;
            if s.rows > SEQUENTIAL_ROWS {
                continue;
            }
            let bank =
                s.following * geometry.delta_bank()..(s.following + 1) * geometry.delta_bank();
            assert!(
                step.delta[bank.clone()]
                    .iter()
                    .zip(&chunk.delta[bank])
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{backend} chunk ROWS {rows}: a {}-row slot's state differs from the step",
                s.rows
            );
            let start = range.start;
            for (index, (a, b)) in step.gated[range.clone()]
                .iter()
                .zip(&chunk.gated[range])
                .enumerate()
            {
                // Beyond the step's GEMV gating (at most 8 rows) both gate the
                // same raw rows in one order; elsewhere the two orders of
                // the RMS sum differ within their bounds from the host.
                let allowed = if is_cpu(device) {
                    0.0
                } else {
                    differences.gated[start + index]
                };
                assert!(
                    (*a as f64 - *b as f64).abs() <= allowed,
                    "{backend} chunk ROWS {rows}: a {}-row slot's gated[{index}] {b} differs from the step's {a}",
                    s.rows
                );
            }
        }
        check(
            &format!("{backend} 4B verify mix: chunk ROWS {rows}"),
            &case,
            &chunk,
            &host,
            against_host(chunk_of(device)),
        );
    }
}

/// BF16 values of `count` pseudo-random numbers in [-scale, scale].
fn bf16_values(count: usize, scale: f32, seed: u64) -> Vec<f32> {
    let mut random = Random(seed);
    (0..count)
        .map(|_| bf16_round(random.next() * scale))
        .collect()
}

fn bf16_tensor(device: &Device, shape: &[u64], values: &[f32]) -> Tensor {
    let bytes = values
        .iter()
        .flat_map(|value| ((bf16_round(*value).to_bits() >> 16) as u16).to_le_bytes())
        .collect::<Vec<_>>();
    Tensor::from_host(device, Element::bf16(), shape, &bytes).unwrap()
}

/// The inputs of the recurrent projection of a case: F32 hidden rows, BF16
/// norm and qkv | z | alpha | beta weights.
struct ProjectionInputs {
    hidden: Tensor,
    norm: Tensor,
    weights: [Tensor; 4],
}

impl ProjectionInputs {
    fn new(device: &Device, case: &Case, hidden: usize, seed: u64) -> Self {
        let g = case.geometry;
        let mut random = Random(seed);
        let values = (0..case.rows * hidden)
            .map(|_| random.next() * 2.0)
            .collect::<Vec<_>>();
        let norm = (0..hidden)
            .map(|_| 1.0 + 0.5 * random.next())
            .collect::<Vec<_>>();
        let rows = [
            g.channels(),
            g.value_heads * g.width,
            g.value_heads,
            g.value_heads,
        ];
        let scale = 2.0 / (hidden as f32).sqrt();
        let weights = std::array::from_fn(|segment| {
            bf16_tensor(
                device,
                &[rows[segment] as u64, hidden as u64],
                &bf16_values(rows[segment] * hidden, scale, seed + 1 + segment as u64),
            )
        });
        Self {
            hidden: Tensor::from_host(
                device,
                Element::f32(),
                &[case.rows as u64, hidden as u64],
                &values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
            norm: bf16_tensor(device, &[hidden as u64], &norm),
            weights,
        }
    }
}

/// One projection mapping of the GEMV and batched GEMV launches.
#[derive(Clone, Copy, Debug)]
struct ProjectionMapping {
    batch_from: u64,
    simdgroups: u64,
    rows: u64,
    lanes: u64,
    batch_simdgroups: u64,
    batch_rows: u64,
}

/// The convolved step form (`gated_delta_project_convolved`, then
/// `gated_delta_step_convolved`) against `gated_delta_project`, then
/// `gated_delta_step`, at equal mappings on Metal: the projection, the gated
/// rows and every bank's window, state and tape are bit-identical. Cases
/// cover 1..16 rows in the GEMV and batched classes, slots shorter and longer
/// than the window, interior stop rows, tape versions, grouped heads, and
/// the 35B geometry.
#[test]
fn convolved_step_form_gives_the_step_bits() {
    let Some(device) = metal() else {
        return;
    };
    let big = Geometry {
        key_heads: 16,
        value_heads: 32,
        width: 128,
        decay_width: 1,
        convolution: 4,
        banks: 11,
        tape: 3,
    };
    let q27 = Geometry {
        key_heads: 16,
        value_heads: 48,
        width: 128,
        decay_width: 1,
        convolution: 4,
        banks: 11,
        tape: 3,
    };
    let small = Geometry {
        banks: 11,
        tape: 3,
        ..SMALL
    };
    let version = |rows, stop, previous, following, taped| SlotCase {
        rows,
        stop,
        previous,
        following,
        taped,
    };
    let cases = vec![
        ("one row", small, 1, vec![slot(1, 1, 1, 6)], false, 512),
        (
            "two rows, one slot",
            small,
            2,
            vec![version(2, 1, 2, 6, 1)],
            true,
            512,
        ),
        (
            "three one-row slots",
            small,
            3,
            (0..3).map(|s| slot(1, 1, 1 + s, 6 + s)).collect(),
            false,
            512,
        ),
        (
            "four rows from a tape version, stop inside",
            small,
            4,
            vec![version(4, 1, 3, 7, 3)],
            true,
            512,
        ),
        (
            "five rows over two slots",
            small,
            5,
            vec![version(2, 2, 1, 6, 0), version(3, 1, 2, 7, 2)],
            false,
            512,
        ),
        (
            "eight rows, padded",
            small,
            8,
            vec![version(3, 3, 1, 6, 1), version(4, 2, 2, 8, 0)],
            true,
            512,
        ),
        (
            "nine rows, one slot",
            small,
            9,
            vec![version(9, 5, 4, 9, 2)],
            false,
            512,
        ),
        (
            "sixteen rows over five slots",
            small,
            16,
            vec![
                version(4, 1, 1, 6, 1),
                version(1, 1, 2, 7, 0),
                version(6, 6, 3, 8, 3),
                version(3, 2, 4, 9, 0),
                version(2, 1, 5, 10, 2),
            ],
            true,
            512,
        ),
        ("27B one row", q27, 1, vec![slot(1, 1, 1, 6)], true, 5120),
        (
            "27B four-row verify",
            q27,
            4,
            vec![version(4, 1, 2, 6, 2)],
            true,
            5120,
        ),
        (
            "27B eight-row verify",
            q27,
            8,
            vec![version(8, 1, 2, 6, 2)],
            true,
            5120,
        ),
        (
            "27B two slots of four",
            q27,
            8,
            vec![version(4, 1, 1, 6, 1), version(4, 2, 2, 7, 0)],
            true,
            5120,
        ),
        ("35B one row", big, 1, vec![slot(1, 1, 1, 6)], true, 2048),
        (
            "35B four-row verify",
            big,
            4,
            vec![version(4, 1, 2, 6, 2)],
            true,
            2048,
        ),
        (
            "35B five one-row slots, padded to eight rows",
            big,
            8,
            (0..5).map(|s| slot(1, 1, 1 + s, 6 + s)).collect(),
            true,
            2048,
        ),
        (
            "35B sixteen rows over two slots",
            big,
            16,
            vec![version(8, 3, 1, 6, 1), version(7, 7, 2, 7, 0)],
            true,
            2048,
        ),
    ];
    let mappings = [
        ProjectionMapping {
            batch_from: 5,
            simdgroups: 16,
            rows: 1,
            lanes: 16,
            batch_simdgroups: 8,
            batch_rows: 2,
        },
        ProjectionMapping {
            batch_from: 3,
            simdgroups: 4,
            rows: 4,
            lanes: 32,
            batch_simdgroups: 4,
            batch_rows: 1,
        },
        ProjectionMapping {
            batch_from: 9,
            simdgroups: 2,
            rows: 2,
            lanes: 16,
            batch_simdgroups: 16,
            batch_rows: 4,
        },
    ];
    for (index, (label, geometry, rows, slots, grouped, hidden)) in cases.into_iter().enumerate() {
        let case =
            Case::new(geometry, rows, slots, grouped, 70 + index as u64).with_bf16_activations();
        let inputs = ProjectionInputs::new(&device, &case, hidden, 90 + index as u64);
        for mapping in mappings {
            let (old_projection, old) = step_form(&device, &case, &inputs, hidden, mapping);
            let (new_projection, new) = convolved_form(&device, &case, &inputs, hidden, mapping);
            let label = format!("{label} {mapping:?}");
            let same = |what: &str, a: &[f32], b: &[f32]| {
                assert_eq!(a.len(), b.len(), "{label}: {what} length");
                if let Some(index) = a
                    .iter()
                    .zip(b)
                    .position(|(a, b)| a.to_bits() != b.to_bits())
                {
                    panic!(
                        "{label}: {what}[{index}] {} differs from {}",
                        b[index], a[index]
                    );
                }
            };
            same("projection", &old_projection, &new_projection);
            same("gated", &old.gated, &new.gated);
            same("window", &old.window, &new.window);
            same("delta", &old.delta, &new.delta);
            same("tape", &old.tape, &new.tape);
        }
    }
}

/// `gated_delta_project`, then `gated_delta_step` (ROWS 32): the projection
/// and the outcome.
fn step_form(
    device: &Device,
    case: &Case,
    inputs: &ProjectionInputs,
    hidden: usize,
    mapping: ProjectionMapping,
) -> (Vec<f32>, Outcome) {
    use magnitude_kernels::gated_delta_project;
    let g = case.geometry;
    // gated_delta_project's launches: stage, gemv, batch, gemm_small, gemm,
    // stage_tall, tall, then the PACK form's four (its tile launch the third
    // of them).
    let specialization = NativeSpecialization::new()
        .with_static("H", hidden as u64)
        .with_static("NK", g.key_heads as u64)
        .with_static("NV", g.value_heads as u64)
        .with_static("W", g.width as u64)
        .with_static("QB", 0)
        .with_static("ZB", 0)
        .with_static("PT", 1)
        .with_static("NM", hidden as u64)
        .with_param("BATCH_FROM", mapping.batch_from)
        .with_param("TALL", 0)
        .with_param("PACK", 0)
        .with_param("RESIDUE", 1)
        .with_launch_param(9, "PACK_TOKENS", 4)
        .with_launch_param(9, "WEIGHTS_AHEAD", 0)
        .with_launch_param(1, "SIMDGROUPS", mapping.simdgroups)
        .with_launch_param(1, "ROWS", mapping.rows)
        .with_launch_param(1, "LANES", mapping.lanes)
        .with_launch_param(2, "BATCH_SIMDGROUPS", mapping.batch_simdgroups)
        .with_launch_param(2, "BATCH_ROWS", mapping.batch_rows)
        .with_launch_param(4, "TILE_M", 64)
        .with_launch_param(4, "TILE_N", 64)
        .with_launch_param(6, "TALL_M", 128)
        .with_launch_param(6, "TALL_K", 64)
        .with_launch_param(6, "STAGERS", 2);
    let projection = gated_delta_project::native_for_device_with(
        device,
        gated_delta_project::Elements {
            NW: Element::bf16(),
            QW: Element::bf16(),
            GW: Element::bf16(),
            AW: Element::bf16(),
            BW: Element::bf16(),
            A: Element::bf16(),
        },
        &specialization,
    )
    .unwrap()
    .call(gated_delta_project::Args {
        hidden: &inputs.hidden,
        input_norm: &inputs.norm,
        qkv_weight: &inputs.weights[0],
        gate_weight: &inputs.weights[1],
        alpha_weight: &inputs.weights[2],
        beta_weight: &inputs.weights[3],
        epsilon: 1e-6,
        qkv_bound: &Tensor::from_host(device, Element::f32(), &[0], &[]).unwrap(),
        gate_bound: &Tensor::from_host(device, Element::f32(), &[0], &[]).unwrap(),
    })
    .unwrap()
    .value;
    let mut t = case.tensors(device, Element::bf16());
    t.inputs = Inputs::Gated { projection };
    let gated = case
        .native_step(device, Element::bf16(), 32)
        .call(t.step_args(case))
        .unwrap()
        .value;
    (
        read(t.projection()),
        Outcome {
            gated: read(&gated),
            window: read(&t.window),
            delta: read(&t.delta),
            tape: read(&t.tape),
        },
    )
}

/// `gated_delta_project_convolved`, then `gated_delta_step_convolved` (ROWS
/// 32): the projection and the outcome.
fn convolved_form(
    device: &Device,
    case: &Case,
    inputs: &ProjectionInputs,
    hidden: usize,
    mapping: ProjectionMapping,
) -> (Vec<f32>, Outcome) {
    use magnitude_kernels::{gated_delta_project_convolved, gated_delta_step_convolved};
    let g = case.geometry;
    // Launches: normalize, gemv (one row), batch, gemv_rows (three rows up to
    // BATCH_FROM), gemv_pair (two rows). Every GEMV launch takes the mapping, as `gated_delta_project`'s one
    // GEMV launch does for every row count.
    let specialization = NativeSpecialization::new()
        .with_static("H", hidden as u64)
        .with_static("NK", g.key_heads as u64)
        .with_static("NV", g.value_heads as u64)
        .with_static("W", g.width as u64)
        .with_static("C", g.convolution as u64)
        .with_static("PT", 1)
        .with_static("NM", hidden as u64)
        .with_param("BATCH_FROM", mapping.batch_from)
        .with_launch_param(1, "SIMDGROUPS", mapping.simdgroups)
        .with_launch_param(1, "ROWS", mapping.rows)
        .with_launch_param(1, "LANES", mapping.lanes)
        .with_launch_param(2, "BATCH_SIMDGROUPS", mapping.batch_simdgroups)
        .with_launch_param(2, "BATCH_ROWS", mapping.batch_rows)
        .with_launch_param(2, "BATCH_PARTS", 2)
        // Dense weights have no matrix GEMV.
        .with_launch_param(2, "MATRIX", 0)
        .with_launch_param(3, "SIMDGROUPS", mapping.simdgroups)
        .with_launch_param(3, "ROWS", mapping.rows)
        .with_launch_param(3, "LANES", mapping.lanes)
        .with_launch_param(4, "SIMDGROUPS", mapping.simdgroups)
        .with_launch_param(4, "ROWS", mapping.rows)
        .with_launch_param(4, "LANES", mapping.lanes);
    let mut t = case.tensors(device, Element::bf16());
    let projected = gated_delta_project_convolved::native_for_device_with(
        device,
        gated_delta_project_convolved::Elements {
            NW: Element::bf16(),
            QW: Element::bf16(),
            GW: Element::bf16(),
            AW: Element::bf16(),
            BW: Element::bf16(),
            A: Element::bf16(),
        },
        &specialization,
    )
    .unwrap()
    .call(gated_delta_project_convolved::Args {
        hidden: &inputs.hidden,
        input_norm: &inputs.norm,
        qkv_weight: &inputs.weights[0],
        gate_weight: &inputs.weights[1],
        alpha_weight: &inputs.weights[2],
        beta_weight: &inputs.weights[3],
        convolution: &t.convolution,
        segments: &t.segments,
        stop: &t.stop,
        previous_bank: &t.previous,
        previous_tape: &t.previous_tape,
        following_bank: &t.following,
        window: &mut t.window,
        epsilon: 1e-6,
        slab_banks: t.slab_banks,
    })
    .unwrap();
    let step = NativeSpecialization::new()
        .with_static("NK", g.key_heads as u64)
        .with_static("NV", g.value_heads as u64)
        .with_static("W", g.width as u64)
        .with_param("ROWS", 32);
    let gated = gated_delta_step_convolved::native_for_device_with(
        device,
        gated_delta_step_convolved::Elements {
            RN: Element::f32(),
            A: Element::bf16(),
        },
        &step,
    )
    .unwrap()
    .call(gated_delta_step_convolved::Args {
        projection: &projected.r0,
        convolved: &projected.r1,
        rate: &t.rate,
        time_bias: &t.time_bias,
        recurrent_norm: &t.norm,
        segments: &t.segments,
        stop: &t.stop,
        previous_bank: &t.previous,
        previous_tape: &t.previous_tape,
        following_bank: &t.following,
        delta: &mut t.delta,
        tape: &mut t.tape,
        norm_epsilon: case.epsilon,
        epsilon: case.rms_epsilon,
        grouped: case.grouped,
        gate_function: case.gate_function,
        slab_banks: t.slab_banks,
    })
    .unwrap()
    .value;
    (
        read(&projected.r0),
        Outcome {
            gated: read(&gated),
            window: read(&t.window),
            delta: read(&t.delta),
            tape: read(&t.tape),
        },
    )
}

/// The normalized staging of the batch fallback retains its prior projection
/// and all recurrent state bits, including a partial second row fragment.
#[test]
fn convolved_batch_fallback_keeps_state_bits() {
    let Some(device) = metal() else {
        return;
    };
    let case = Case::new(
        Geometry {
            banks: 5,
            tape: 3,
            ..SMALL
        },
        13,
        vec![
            SlotCase {
                rows: 2,
                stop: 1,
                previous: 1,
                following: 3,
                taped: 2,
            },
            SlotCase {
                rows: 10,
                stop: 8,
                previous: 2,
                following: 4,
                taped: 1,
            },
        ],
        true,
        172,
    )
    .with_bf16_activations();
    let mapping = ProjectionMapping {
        batch_from: 3,
        simdgroups: 8,
        rows: 2,
        lanes: 16,
        batch_simdgroups: 8,
        batch_rows: 2,
    };
    let inputs = ProjectionInputs::new(&device, &case, 512, 173);
    let (before_projection, before) = step_form(&device, &case, &inputs, 512, mapping);
    let (after_projection, after) = convolved_form(&device, &case, &inputs, 512, mapping);
    for (label, a, b) in [
        ("projection", before_projection, after_projection),
        ("gated", before.gated, after.gated),
        ("window", before.window, after.window),
        ("delta", before.delta, after.delta),
        ("tape", before.tape, after.tape),
    ] {
        assert_eq!(
            a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "{label}"
        );
    }
}

/// GLM-5.3-Flash's linear attention: 64 heads of 128, one decay per key
/// channel (Kimi Delta Attention's bounded sigmoid, log decays in (-5, 0)),
/// convolution 4.
fn kda(banks: usize, tape: usize) -> Geometry {
    Geometry {
        key_heads: 64,
        value_heads: 64,
        width: 128,
        decay_width: 128,
        convolution: 4,
        banks,
        tape,
    }
}

/// One backend's state entries for the per-channel checks: the step and each
/// chunk mapping (a label and its run of a case in an activation element).
struct PerChannelEntries<'a> {
    backend: &'a str,
    kind: BackendName,
    step: Box<dyn Fn(&Case, Element) -> Outcome + 'a>,
    chunks: Vec<(String, Box<dyn Fn(&Case, Element) -> Outcome + 'a>)>,
}

impl PerChannelEntries<'_> {
    fn step_form(&self) -> Native {
        Native::Step(self.kind)
    }

    fn chunk_form(&self) -> Native {
        Native::Chunk(self.kind)
    }
}

impl<'a> PerChannelEntries<'a> {
    /// The step and the chunk (one `CHUNK_ROWS` mapping: ROWS never changes bits) on
    /// `device` through the portable entries.
    fn native(device: &'a Device) -> Self {
        Self {
            backend: device.backend().as_str(),
            kind: device.backend(),
            step: Box::new(move |case: &Case, activation| case.native(device, activation, None)),
            chunks: CHUNK_ROWS
                .into_iter()
                .take(1)
                .map(|rows| {
                    let run: Box<dyn Fn(&Case, Element) -> Outcome + 'a> =
                        Box::new(move |case: &Case, activation| {
                            case.native(device, activation, Some(rows.min(case.geometry.width as u64)))
                        });
                    (format!("chunk ROWS {rows}"), run)
                })
                .collect(),
        }
    }
}

/// Per-channel decays in the portable body's small geometry: the step and
/// the chunk against the interpreted body, from the zero seed, from tape
/// versions (whose decays replay per channel) and over slots whose chunked
/// pieces end inside them; the host model against the body.
#[test]
fn per_channel_decay_matches_the_portable_body() {
    for device in devices() {
        per_channel_decay_matches_the_portable_body_on(&PerChannelEntries::native(&device));
    }
}

fn per_channel_decay_matches_the_portable_body_on(entries: &PerChannelEntries) {
    let geometry = Geometry {
        decay_width: SMALL.width,
        banks: 7,
        tape: 3,
        ..SMALL
    };
    let version = |rows, stop, previous, following, taped| SlotCase {
        rows,
        stop,
        previous,
        following,
        taped,
    };
    let cases = [
        (
            "short and chunked slots from tape versions",
            Case::new(
                geometry,
                64,
                vec![version(5, 2, 1, 4, 2), version(43, 37, 2, 5, 3), version(9, 9, 0, 6, 0)],
                false,
                61,
            )
            .with_bounded_decay(),
        ),
        (
            "grouped heads, sigmoid gate, a zero stop",
            Case::new(
                geometry,
                40,
                vec![version(21, 0, 1, 4, 1), version(19, 19, 3, 5, 0)],
                true,
                62,
            )
            .with_bounded_decay()
            .with_sigmoid_gate(),
        ),
    ];
    for (label, case) in &cases {
        let label = format!("{} per-channel {label}", entries.backend);
        let oracle = case.oracle();
        check(&format!("{label}: host vs body"), case, &case.host(), &oracle, host_against_body());
        check(
            &format!("{label}: step"),
            case,
            &(entries.step)(case, Element::f32()),
            &oracle,
            between(entries.step_form(), ORACLE),
        );
        for (mapping, chunk) in &entries.chunks {
            check(
                &format!("{label}: {mapping}"),
                case,
                &chunk(case, Element::f32()),
                &oracle,
                between(entries.chunk_form(), ORACLE),
            );
        }
    }
}

/// GLM-5.3-Flash's geometry: decode, verify and prefill on the step and the
/// chunk, against the f64 host model of the recurrence, with BF16
/// activations. The prefills are the fewest rows that reach every chunk path
/// on every backend (`METAL_CHUNK_PIECE` on Metal, `CUDA_CHUNK_PIECE` on CUDA): a chunked slot over
/// 16 rows whose last piece is partial and ends at its stop row, inside a
/// piece, with rows after the stop row advancing sequentially; and a chunked
/// slot beside a short (sequential) one. Decays reach -5 a row, so a 16-row
/// piece already spans cumulative log decays to -80.
#[test]
fn kda_geometry_step_and_chunk_agree_with_the_host_model() {
    for device in devices() {
        kda_geometry_step_and_chunk_agree_with_the_host_model_on(&PerChannelEntries::native(&device));
    }
}

fn kda_geometry_step_and_chunk_agree_with_the_host_model_on(entries: &PerChannelEntries) {
    let geometry = kda(9, 0);
    let cases = [
        ("decode, one slot", 1, vec![slot(1, 1, 1, 3)]),
        ("verify, two slots", 8, vec![slot(4, 2, 1, 3), slot(4, 4, 2, 4)]),
        ("prefill 21, stop inside the second piece", 21, vec![slot(21, 18, 1, 3)]),
        (
            "prefill 24, a chunked and a short slot",
            24,
            vec![slot(19, 19, 1, 3), slot(5, 2, 0, 4)],
        ),
    ];
    for (label, rows, slots) in cases {
        let label = format!("{} KDA {label}", entries.backend);
        let case = Case::new(geometry, rows, slots, false, 63)
            .with_bf16_activations()
            .with_bounded_decay();
        let host = case.host();
        check(
            &format!("{label}: step"),
            &case,
            &(entries.step)(&case, Element::bf16()),
            &host,
            against_host(entries.step_form()),
        );
        if rows > SEQUENTIAL_ROWS {
            for (mapping, chunk) in &entries.chunks {
                check(
                    &format!("{label}: {mapping}"),
                    &case,
                    &chunk(&case, Element::bf16()),
                    &host,
                    against_host(entries.chunk_form()),
                );
            }
        }
    }
}

/// A prefill of n rows on the chunk, then m one-row decode steps from the
/// state it published (carried in its stored form), equals one step advance
/// of all n + m rows: the decode rows' outputs and the final state, within
/// the sum of the two runs' bounds from the host model (`Native::Prefill`
/// for the first, whose rows after the prefill advance row by row).
#[test]
fn kda_prefill_then_decode_equals_decode_alone() {
    for device in devices() {
        kda_prefill_then_decode_equals_decode_alone_on(&PerChannelEntries::native(&device));
    }
}

fn kda_prefill_then_decode_equals_decode_alone_on(entries: &PerChannelEntries) {
    // One full piece of either backend's chunk and a partial one, then three
    // decode steps.
    let (n, m) = (20, 3);
    let geometry = kda(5, 0);
    let whole = Case::new(geometry, n + m, vec![slot(n + m, n + m, 1, 2)], false, 64)
        .with_bf16_activations()
        .with_bounded_decay();
    let row = geometry.value_heads * geometry.width;
    let bank = geometry.delta_bank();
    let backend = entries.backend;
    let alone = (entries.step)(&whole, Element::bf16());
    let decode = whole.bounds(entries.step_form());
    let prefill = whole.bounds(Native::Prefill(entries.kind, n));
    for (mapping, chunk) in &entries.chunks {
        let mut outcome = chunk(&whole.part(0, n, slot(n, n, 1, 2), None), Element::bf16());
        let mut decoded = Vec::new();
        let mut source = 2;
        for step in 0..m {
            let following = [3, 4, 1][step];
            outcome = (entries.step)(
                &whole.part(n + step, 1, slot(1, 1, source, following), Some(&outcome)),
                Element::bf16(),
            );
            decoded.extend_from_slice(&outcome.gated[..row]);
            source = following;
        }
        for (index, (a, b)) in decoded.iter().zip(&alone.gated[n * row..]).enumerate() {
            let allowed = decode.gated[n * row + index] + prefill.gated[n * row + index];
            assert!(
                (*a as f64 - *b as f64).abs() <= allowed,
                "{backend} {mapping}: decode output [{index}] after a prefill {a} vs decode alone {b}, bound {allowed:e}"
            );
        }
        let after = outcome.delta[source * bank..][..bank].iter();
        let alone_state = alone.delta[2 * bank..][..bank].iter();
        let (decode_state, prefill_state) = (decode.delta[2].as_ref().unwrap(), prefill.delta[2].as_ref().unwrap());
        for (index, (a, b)) in after.zip(alone_state).enumerate() {
            let allowed = decode_state[index / geometry.width] + prefill_state[index / geometry.width];
            assert!(
                (*a as f64 - *b as f64).abs() <= allowed,
                "{backend} {mapping}: state[{index}] after a prefill and decode {a} vs decode alone {b}, bound {allowed:e}"
            );
        }
    }
}
