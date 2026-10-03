// Recurrent state entries against their semantic oracle, on Metal (when
// present) and the CPU.
//
// `gated_delta_step` and `gated_delta_chunk` share one portable body
// (`gated_delta_rows`). Small geometries run that body in the Seismic
// interpreter; the real 4B geometry uses an independent f64 host model of the
// same contract. Every case checks the gated outputs, the published window
// (bit-exact: it is a copy) and state, and that no bank other than each
// slot's successor changes: accepted banks, the zero seed, and unrelated
// banks keep their exact bytes.

use magnitude_kernels::{gated_delta_chunk, gated_delta_step};
use seismic::{
    BackendName, Device, DeviceCatalog, Element, NativeSpecialization, SlabRegion, SlabTensor,
    Tensor,
};

const SENTINEL: f32 = 7.25;

#[derive(Clone, Copy)]
struct Geometry {
    key_heads: usize,
    value_heads: usize,
    width: usize,
    convolution: usize,
    banks: usize,
    /// Tape rows per bank (T).
    tape: usize,
}

impl Geometry {
    fn channels(self) -> usize {
        (2 * self.key_heads + self.value_heads) * self.width
    }
    fn projection_width(self) -> usize {
        self.channels() + self.value_heads * self.width + 2 * self.value_heads
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
    /// Floats of one tape row: u [NV, W] | k [NK, W] | d [NV].
    fn tape_row(self) -> usize {
        (self.value_heads + self.key_heads) * self.width + self.value_heads
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
    epsilon: f32,
    /// The gating's RMS epsilon.
    rms_epsilon: f32,
    projection: Vec<f32>,
    convolution: Vec<f32>,
    rate: Vec<f32>,
    time_bias: Vec<f32>,
    /// The recurrent norm weights (W).
    norm: Vec<f32>,
    window: Vec<f32>,
    delta: Vec<f32>,
    tape: Vec<f32>,
}

struct Outcome {
    gated: Vec<f32>,
    window: Vec<f32>,
    delta: Vec<f32>,
    tape: Vec<f32>,
}

/// Device tensors of one case; each call mutates its window, delta and tape.
struct Tensors {
    projection: Tensor,
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

impl Tensors {
    fn step_args<'a>(&'a mut self, case: &Case) -> gated_delta_step::Args<'a> {
        gated_delta_step::Args {
            projection: &self.projection,
            convolution: &self.convolution,
            rate: &self.rate,
            time_bias: &self.time_bias,
            recurrent_norm: &self.norm,
            segments: &self.segments,
            stop: &self.stop,
            previous_bank: &self.previous,
            previous_tape: &self.previous_tape,
            following_bank: &self.following,
            window: &mut self.window,
            delta: &mut self.delta,
            tape: &mut self.tape,
            norm_epsilon: case.epsilon,
            epsilon: case.rms_epsilon,
            grouped: case.grouped,
            slab_banks: self.slab_banks,
        }
    }

    fn chunk_args<'a>(&'a mut self, case: &Case) -> gated_delta_chunk::Args<'a> {
        gated_delta_chunk::Args {
            projection: &self.projection,
            convolution: &self.convolution,
            rate: &self.rate,
            time_bias: &self.time_bias,
            recurrent_norm: &self.norm,
            segments: &self.segments,
            stop: &self.stop,
            previous_bank: &self.previous,
            previous_tape: &self.previous_tape,
            following_bank: &self.following,
            window: &mut self.window,
            delta: &mut self.delta,
            tape: &mut self.tape,
            norm_epsilon: case.epsilon,
            epsilon: case.rms_epsilon,
            grouped: case.grouped,
            slab_banks: self.slab_banks,
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
            for head in 0..geometry.value_heads {
                // alpha: a spread of decays; beta input: both gate regimes.
                projection[row * width + gates + head] = random.next() * 3.0;
                projection[row * width + gates + geometry.value_heads + head] = random.next() * 4.0;
            }
        }
        let convolution = (0..geometry.channels() * geometry.convolution)
            .map(|_| random.next() * 0.6)
            .collect();
        let rate = (0..geometry.value_heads)
            .map(|_| -(0.05 + 0.5 * (random.next() + 1.0)))
            .collect();
        let time_bias = (0..geometry.value_heads)
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
            epsilon: 1.0e-6 * geometry.width as f32,
            rms_epsilon: 1.0e-6,
            projection,
            convolution,
            rate,
            time_bias,
            norm,
            window,
            delta,
            tape,
        }
    }

    /// The next advance of the same layer after `outcome`: new projection rows
    /// for `slots` (reading the versions they name), the same weights, and the
    /// arenas `outcome` left.
    fn continued(&self, outcome: &Outcome, rows: usize, slots: Vec<SlotCase>, seed: u64) -> Self {
        let fresh = Case::new(self.geometry, rows, slots, self.grouped, seed);
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
            norm: self.norm.clone(),
            window: outcome.window.clone(),
            delta: outcome.delta.clone(),
            tape: outcome.tape.clone(),
            bf16: self.bf16,
            ..fresh
        }
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
            for head in 0..self.geometry.value_heads {
                self.projection[row * width + alpha + head] = 250.0;
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

    /// The f64 host model of the entry contract.
    fn host(&self) -> Outcome {
        let g = self.geometry;
        let (nk, nv, w, c) = (g.key_heads, g.value_heads, g.width, g.convolution);
        let channels = g.channels();
        let width = g.projection_width();
        let key_of = |head: usize| {
            if self.grouped {
                head * nk / nv
            } else {
                head % nk
            }
        };
        let mut mixed = vec![0.0f32; self.rows * nv * w];
        let mut window = self.window.clone();
        let mut delta = self.delta.clone();
        let mut tape = self.tape.clone();
        let mut first = 0;
        for slot in &self.slots {
            let mut state = delta[slot.previous * g.delta_bank()..][..g.delta_bank()]
                .iter()
                .map(|value| *value as f64)
                .collect::<Vec<_>>();
            // The version: the bank's state advanced by its first tape rows.
            for entry in 0..slot.taped {
                let row = &self.tape[slot.previous * g.tape_bank() + entry * g.tape_row()..]
                    [..g.tape_row()];
                for head in 0..nv {
                    let decay = row[(nv + nk) * w + head] as f64;
                    let key = &row[nv * w + key_of(head) * w..][..w];
                    for state_row in 0..w {
                        let innovation = row[head * w + state_row] as f64;
                        let s = &mut state[(head * w + state_row) * w..][..w];
                        for column in 0..w {
                            s[column] = s[column] * decay + innovation * key[column] as f64;
                        }
                    }
                }
            }
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
                for head in 0..2 * nk + nv {
                    let mut squares = 0.0;
                    for column in 0..w {
                        let channel = head * w + column;
                        let mut sum = self.convolution[channel * c + c - 1] as f64
                            * self.projection[row * width + channel] as f64;
                        for tap in 0..c - 1 {
                            sum += self.convolution[channel * c + tap] as f64
                                * input(row, local, tap, channel);
                        }
                        let value = sum / (1.0 + (-sum).exp());
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
                for value_head in 0..nv {
                    let key_head = if self.grouped {
                        value_head * nk / nv
                    } else {
                        value_head % nk
                    };
                    let alpha =
                        self.projection[row * width + channels + nv * w + value_head] as f64;
                    let beta_input =
                        self.projection[row * width + channels + nv * w + nv + value_head] as f64;
                    let beta = 1.0 / (1.0 + (-beta_input).exp());
                    let shifted = alpha + self.time_bias[value_head] as f64;
                    let softplus = shifted.max(0.0) + (1.0 + (-shifted.abs()).exp()).ln();
                    let factor = (self.rate[value_head] as f64 * softplus).exp();
                    let query = &prepared[key_head * w..][..w];
                    let key = &prepared[(nk + key_head) * w..][..w];
                    let entry = (local >= slot.stop && local - slot.stop < taped).then(|| {
                        slot.following * g.tape_bank() + (local - slot.stop) * g.tape_row()
                    });
                    if let Some(entry) = entry {
                        tape[entry + (nv + nk) * w + value_head] = factor as f32;
                        for column in 0..w {
                            tape[entry + nv * w + key_head * w + column] = key[column] as f32;
                        }
                    }
                    for state_row in 0..w {
                        let s = &mut state[(value_head * w + state_row) * w..][..w];
                        let mut remembered = 0.0;
                        for column in 0..w {
                            s[column] *= factor;
                            remembered += s[column] * key[column];
                        }
                        let residual =
                            (prepared[(2 * nk + value_head) * w + state_row] - remembered) * beta;
                        if let Some(entry) = entry {
                            tape[entry + value_head * w + state_row] = residual as f32;
                        }
                        let mut output = 0.0;
                        for column in 0..w {
                            s[column] += residual * key[column];
                            output += s[column] * query[column];
                        }
                        mixed[(row * nv + value_head) * w + state_row] = output as f32;
                    }
                }
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
        Outcome {
            gated: self.gate(&mixed),
            window,
            delta,
            tape,
        }
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
                    let activated = round(gate / (1.0 + (-gate).exp())) as f64;
                    gated[(row * nv + head) * w + column] = round(normalized * activated);
                }
            }
        }
        gated
    }

    /// The portable body executed by the Seismic interpreter.
    fn oracle(&self) -> Outcome {
        use seismic_lang::{
            checked::{check_source, SourceFile},
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
        let module = check_source(sources).unwrap();
        let elements = ElementBindings::new()
            .bind("A", registry::dense(DType::F32))
            .bind("RN", registry::dense(DType::F32));
        let logical = module
            .entry(module.entry_named("gated_delta_step").unwrap(), &elements)
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
        let tensors = vec![
            floats(vec![self.rows, g.projection_width()], &self.projection),
            floats(vec![g.channels(), g.convolution], &self.convolution),
            floats(vec![g.value_heads], &self.rate),
            floats(vec![g.value_heads], &self.time_bias),
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
        ];
        let mut arguments = tensors
            .into_iter()
            .map(|tensor| Arg::Tensor(interpreter.add_tensor(tensor)))
            .collect::<Vec<_>>();
        arguments.push(Arg::Scalar(ReferenceScalar::F32(self.epsilon.to_bits())));
        arguments.push(Arg::Scalar(ReferenceScalar::F32(
            self.rms_epsilon.to_bits(),
        )));
        arguments.push(Arg::Scalar(ReferenceScalar::Bool(self.grouped)));
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
                10 => window = Some(read(input.tensor())),
                11 => delta = Some(read(input.tensor())),
                12 => tape = Some(read(input.tensor())),
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
        Tensors {
            projection: from_activation(
                &[self.rows as u64, g.projection_width() as u64],
                &self.projection,
            ),
            convolution: from_f32(
                &[g.channels() as u64, g.convolution as u64],
                &self.convolution,
            ),
            rate: from_f32(&[g.value_heads as u64], &self.rate),
            time_bias: from_f32(&[g.value_heads as u64], &self.time_bias),
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

    /// The chunk with `ROWS` state rows per threadgroup or work item.
    fn native_chunk(
        &self,
        device: &Device,
        activation: Element,
        rows: u64,
    ) -> seismic::NativeKernel<gated_delta_chunk::Entry> {
        gated_delta_chunk::native_for_device_with(
            device,
            gated_delta_chunk::Elements {
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
        let gated = match chunk {
            None => {
                self.native_step(device, activation, 32.min(self.geometry.width as u64))
                    .call(t.step_args(self))
                    .unwrap()
                    .value
            }
            Some(mapping) => {
                self.native_chunk(device, activation, mapping)
                    .call(t.chunk_args(self))
                    .unwrap()
                    .value
            }
        };
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

/// Max absolute error relative to the reference's RMS, and the plain RMS of
/// the difference relative to it: the per-layer error measures of the D4 gate.
fn errors(actual: &[f32], expected: &[f32]) -> (f64, f64) {
    assert_eq!(actual.len(), expected.len());
    let scale = (expected.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / expected.len() as f64)
        .sqrt()
        .max(1e-12);
    let mut max = 0.0f64;
    let mut squares = 0.0f64;
    for (a, e) in actual.iter().zip(expected) {
        assert!(a.is_finite(), "non-finite output");
        let difference = (*a as f64 - *e as f64).abs();
        max = max.max(difference);
        squares += difference * difference;
    }
    (max / scale, (squares / actual.len() as f64).sqrt() / scale)
}

/// Compare one outcome with the reference. Gated outputs and successor state
/// are held to `(max, rms)` relative tolerances; windows are copies and must
/// be bit-exact; every non-successor bank must be untouched. BF16 gated
/// outputs are compared with the reference (whose gating rounds to BF16 where
/// the body does), each within two BF16 units in the last place of its
/// magnitude (plus the relative bound): a raw output within tolerance can
/// still round its normalized value, and then the product, one unit apart.
fn check(label: &str, case: &Case, actual: &Outcome, expected: &Outcome, tolerance: (f64, f64)) {
    let g = case.geometry;
    let (max, rms) = errors(&actual.gated, &expected.gated);
    println!("{label} gated: max/rms {max:.3e} rms/rms {rms:.3e}");
    if case.bf16 {
        let scale = (expected
            .gated
            .iter()
            .map(|v| (*v as f64).powi(2))
            .sum::<f64>()
            / expected.gated.len() as f64)
            .sqrt();
        for (index, (a, e)) in actual.gated.iter().zip(&expected.gated).enumerate() {
            let rounded = bf16_round(*e) as f64;
            let allowed = rounded.abs() * 2f64.powi(-6) + tolerance.0 * scale;
            assert!(
                (*a as f64 - rounded).abs() <= allowed,
                "{label} gated[{index}] {a} vs {rounded}"
            );
        }
        assert!(rms <= tolerance.1, "{label} gated rms error {rms}");
    } else {
        assert!(
            max <= tolerance.0 && rms <= tolerance.1,
            "{label} gated error {max} {rms}"
        );
    }
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
            let (max, rms) = errors(delta, expected_delta);
            println!("{label} delta bank {bank}: max/rms {max:.3e} rms/rms {rms:.3e}");
            assert!(
                max <= tolerance.0 && rms <= tolerance.1,
                "{label} state error {max} {rms}"
            );
            if recorded > 0 {
                let expected_tape = &expected.tape[bank * g.tape_bank()..][..recorded];
                let (max, rms) = errors(&tape[..recorded], expected_tape);
                println!("{label} tape bank {bank}: max/rms {max:.3e} rms/rms {rms:.3e}");
                assert!(
                    max <= tolerance.0 && rms <= tolerance.1,
                    "{label} tape error {max} {rms}"
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

fn metal() -> Option<Device> {
    DeviceCatalog::discover()
        .ok()
        .and_then(|catalog| catalog.open_backend(BackendName::Metal).ok())
}

/// Metal when present and the CPU device. (Vulkan's mappings are checked in
/// `vulkan_recurrent.rs`, CUDA's in `cuda_recurrent.rs`.)
fn devices() -> Vec<Device> {
    let catalog = DeviceCatalog::discover().unwrap();
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
            (1e-4, 1e-5),
        );
        let step = case.native(device, Element::f32(), None);
        check(
            &format!("{label}: step"),
            &case,
            &step,
            &oracle,
            (2e-5, 2e-6),
        );
        for rows in CHUNK_ROWS {
            let rows = rows.min(case.geometry.width as u64);
            let chunked = case.native(device, Element::f32(), Some(rows));
            check(
                &format!("{label}: chunk ROWS {rows}"),
                &case,
                &chunked,
                &oracle,
                (5e-4, 2e-5),
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
        (5e-4, 2e-5),
    );
    let chunk = case.native(device, Element::f32(), Some(32));
    check(
        &format!("{backend} slab tape chunk"),
        &case,
        &chunk,
        &host,
        (5e-4, 2e-5),
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
        (5e-4, 2e-5),
    );
    let chunk = case.native_with_reused_slab(device, Element::f32(), Some(32), true);
    check(
        &format!("{backend} reused bank chunk"),
        &case,
        &chunk,
        &host,
        (5e-4, 2e-5),
    );
}

/// A committed tape version (bank, j) equals a run that published after those
/// rows: the next advance from either has the same bits. `run` executes a
/// case on one entry; the tentative advance is a 5-row verify slot (stop 1).
fn tape_versions_equal_stopped_runs(
    label: &str,
    geometry: Geometry,
    run: &dyn Fn(&Case) -> Outcome,
) {
    let verify = 5;
    let g = Geometry {
        banks: 4,
        tape: verify - 1,
        ..geometry
    };
    for accepted in 0..verify {
        let tentative =
            Case::new(g, verify, vec![slot(verify, 1, 1, 2)], true, 41).with_bf16_activations();
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
        let stopped = Case::new(g, verify, vec![slot(verify, 1 + accepted, 1, 2)], true, 41)
            .with_bf16_activations();
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
            (1.5e-2, 3e-3),
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
            (1e-4, 1e-5),
        );
        check(
            &format!("{label}: step"),
            &case,
            &case.native(device, Element::f32(), None),
            &oracle,
            (2e-5, 2e-6),
        );
        for rows in CHUNK_ROWS {
            let rows = rows.min(case.geometry.width as u64);
            let chunked = case.native(device, Element::f32(), Some(rows));
            check(
                &format!("{label}: chunk ROWS {rows}"),
                &case,
                &chunked,
                &oracle,
                (5e-4, 2e-5),
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
        convolution: 4,
        banks: 4,
        tape: 0,
    };
    let backend = device.backend().as_str();
    for geometry in [SMALL, qwen] {
        tape_versions_equal_stopped_runs(&format!("{backend} step"), geometry, &|case| {
            case.native(device, Element::bf16(), None)
        });
        tape_versions_equal_stopped_runs(&format!("{backend} chunk"), geometry, &|case| {
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
            (1e-4, 3e-3),
        );
        if rows >= 16 {
            let mut reference: Option<Outcome> = None;
            for rows in CHUNK_ROWS {
                let chunked = case.native(device, Element::bf16(), Some(rows));
                check(
                    &format!("4B {label}: chunk ROWS {rows}"),
                    &case,
                    &chunked,
                    &host,
                    (1e-3, 3e-3),
                );
                let (max, rms) = errors(&chunked.gated, &step.gated);
                println!("4B {label}: chunk ROWS {rows} vs step: max {max:.3e} rms {rms:.3e}");
                // Both are within two BF16 ulps of the host model per element
                // plus their relative bounds (checked above), so they differ
                // by at most four ulps of the host value plus both bounds.
                let scale = (host.gated.iter().map(|v| (*v as f64).powi(2)).sum::<f64>()
                    / host.gated.len() as f64)
                    .sqrt();
                for (index, ((c, s), e)) in chunked
                    .gated
                    .iter()
                    .zip(&step.gated)
                    .zip(&host.gated)
                    .enumerate()
                {
                    let allowed = (*e as f64).abs() * 2f64.powi(-5) + (1e-3 + 1e-4) * scale;
                    assert!(
                        (*c as f64 - *s as f64).abs() <= allowed,
                        "4B {label}: chunk ROWS {rows} gated[{index}] {c} vs step {s}"
                    );
                }
                assert!(rms <= 3e-3);
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
        convolution: 4,
        banks: 11,
        tape: 0,
    };
    let slots = vec![
        slot(4, 1, 1, 6),
        slot(16, 9, 2, 7),
        slot(40, 40, 3, 8),
        slot(1, 1, 4, 9),
        slot(7, 0, 5, 10),
    ];
    let case = Case::new(geometry, 70, slots, false, 31).with_bf16_activations();
    let step = case.native(device, Element::bf16(), None);
    let host = case.host();
    let row_elements = geometry.value_heads * geometry.width;
    for rows in CHUNK_ROWS {
        let chunk = case.native(device, Element::bf16(), Some(rows));
        let mut first = 0;
        for s in &case.slots {
            let range = first * row_elements..(first + s.rows) * row_elements;
            first += s.rows;
            if s.rows > 16 {
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
            for (index, (a, b)) in step.gated[range.clone()]
                .iter()
                .zip(&chunk.gated[range])
                .enumerate()
            {
                let allowed = if is_cpu(device) {
                    0.0
                } else {
                    (*a as f64).abs() * 2f64.powi(-6)
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
            (1e-3, 3e-3),
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
    for (index, (label, geometry, rows, slots, grouped, hidden)) in cases.into_iter().enumerate()
    {
        let case = Case::new(geometry, rows, slots, grouped, 70 + index as u64)
            .with_bf16_activations();
        let inputs = ProjectionInputs::new(&device, &case, hidden, 90 + index as u64);
        for mapping in mappings {
            let (old_projection, old) = step_form(&device, &case, &inputs, hidden, mapping);
            let (new_projection, new) = convolved_form(&device, &case, &inputs, hidden, mapping);
            let label = format!("{label} {mapping:?}");
            let same = |what: &str, a: &[f32], b: &[f32]| {
                assert_eq!(a.len(), b.len(), "{label}: {what} length");
                if let Some(index) = a.iter().zip(b).position(|(a, b)| a.to_bits() != b.to_bits()) {
                    panic!("{label}: {what}[{index}] {} differs from {}", b[index], a[index]);
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
    // gated_delta_project's launches: stage, gemv, batch, gemm_small, gemm.
    let specialization = NativeSpecialization::new()
        .with_static("H", hidden as u64)
        .with_static("NK", g.key_heads as u64)
        .with_static("NV", g.value_heads as u64)
        .with_static("W", g.width as u64)
        .with_param("BATCH_FROM", mapping.batch_from)
        .with_launch_param(1, "SIMDGROUPS", mapping.simdgroups)
        .with_launch_param(1, "ROWS", mapping.rows)
        .with_launch_param(1, "LANES", mapping.lanes)
        .with_launch_param(2, "BATCH_SIMDGROUPS", mapping.batch_simdgroups)
        .with_launch_param(2, "BATCH_ROWS", mapping.batch_rows)
        .with_launch_param(4, "TILE_M", 64)
        .with_launch_param(4, "TILE_N", 64);
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
    })
    .unwrap()
    .value;
    let mut t = case.tensors(device, Element::bf16());
    t.projection = projection;
    let gated = case
        .native_step(device, Element::bf16(), 32)
        .call(t.step_args(case))
        .unwrap()
        .value;
    (
        read(&t.projection),
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
    // Launches: gemv, batch.
    let specialization = NativeSpecialization::new()
        .with_static("H", hidden as u64)
        .with_static("NK", g.key_heads as u64)
        .with_static("NV", g.value_heads as u64)
        .with_static("W", g.width as u64)
        .with_static("C", g.convolution as u64)
        .with_param("BATCH_FROM", mapping.batch_from)
        .with_launch_param(0, "SIMDGROUPS", mapping.simdgroups)
        .with_launch_param(0, "ROWS", mapping.rows)
        .with_launch_param(0, "LANES", mapping.lanes)
        .with_launch_param(1, "BATCH_SIMDGROUPS", mapping.batch_simdgroups)
        .with_launch_param(1, "BATCH_ROWS", mapping.batch_rows);
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

/// The median device time (µs) of each launch of the calls `run` makes, each
/// launch in its own timed unit, joined as "a + b".
fn launch_medians(device: &Device, run: impl FnOnce()) -> String {
    let trace = device
        .trace_submissions(seismic::TraceDetail::Launches)
        .unwrap();
    run();
    let mut launches: Vec<Vec<f64>> = Vec::new();
    for submission in trace.collect().unwrap() {
        for launch in submission.launches {
            if let Some((start, end)) = launch.device {
                if launches.len() <= launch.launch {
                    launches.resize(launch.launch + 1, Vec::new());
                }
                launches[launch.launch].push((end - start) * 1e6);
            }
        }
    }
    launches
        .iter_mut()
        .map(|samples| {
            samples.sort_by(f64::total_cmp);
            format!("{:.1}", samples[samples.len() / 2])
        })
        .collect::<Vec<_>>()
        .join(" + ")
}

/// Device time per call at the 4B geometry (one layer, bf16 activations): the
/// step at 1 row, one row in each of 8 slots, and 8 and 16 rows; the chunk at
/// 8, 16, 128 and 512 rows, with each launch's time. Calls rotate over 16
/// argument sets (≥ 96 MiB of state arenas), so state traffic comes from DRAM
/// as it does across a model's layers.
#[test]
#[ignore = "timing; run explicitly on the measurement host"]
fn metal_recurrent_timings() {
    let Some(device) = metal() else {
        return;
    };
    let options = seismic::MeasureOptions {
        samples: 15,
        min_sample_seconds: 0.01,
    };
    let slot = |rows, previous, following| slot(rows, rows, previous, following);
    for (label, rows, slots) in [
        ("step 1 row", 1usize, vec![slot(1, 1, 2)]),
        (
            "step 8 slots x 1 row",
            8,
            (0..8).map(|s| slot(1, 1 + s, 9 + s)).collect::<Vec<_>>(),
        ),
        ("step 4 rows", 4, vec![slot(4, 1, 2)]),
        ("step 8 rows", 8, vec![slot(8, 1, 2)]),
        ("step 16 rows", 16, vec![slot(16, 1, 2)]),
        ("chunk 1 row", 1, vec![slot(1, 1, 2)]),
        ("chunk 4 rows", 4, vec![slot(4, 1, 2)]),
        ("chunk 8 rows", 8, vec![slot(8, 1, 2)]),
        ("chunk 16 rows", 16, vec![slot(16, 1, 2)]),
        ("chunk 128 rows", 128, vec![slot(128, 1, 2)]),
        ("chunk 512 rows", 512, vec![slot(512, 1, 2)]),
    ] {
        let banks = slots.iter().map(|s| s.following).max().unwrap() + 1;
        let geometry = Geometry {
            key_heads: 16,
            value_heads: 32,
            width: 128,
            convolution: 4,
            banks,
            tape: 0,
        };
        let case = Case::new(geometry, rows, slots, false, 5).with_bf16_activations();
        let mut rotation = (0..16)
            .map(|_| case.tensors(&device, Element::bf16()))
            .collect::<Vec<_>>();
        if label.starts_with("step") {
            for step_rows in [32u64, 16, 64] {
                let kernel = case.native_step(&device, Element::bf16(), step_rows);
                let args = rotation.iter_mut().map(|t| t.step_args(&case)).collect();
                let measured = kernel.measure(args, &options).unwrap();
                println!("{label} ROWS {step_rows}: {:.1} us", measured.median * 1e6);
            }
        } else {
            for chunk_rows in CHUNK_ROWS {
                let kernel = case.native_chunk(&device, Element::bf16(), chunk_rows);
                let args = rotation.iter_mut().map(|t| t.chunk_args(&case)).collect();
                let measured = kernel.measure(args, &options).unwrap();
                let launches = launch_medians(&device, || {
                    for t in rotation.iter_mut() {
                        kernel.call(t.chunk_args(&case)).unwrap();
                    }
                });
                println!(
                    "{label} ROWS {chunk_rows}: {:.1} us (launches {} us)",
                    measured.median * 1e6,
                    launches
                );
            }
        }
    }
}
