// Recurrent state entries against their semantic oracle.
//
// `qwen_recurrent_step` and `qwen_recurrent_chunk` share one portable body
// (`gated_delta_rows`). Small geometries run that body in the Seismic
// interpreter; the real 4B geometry uses an independent f64 host model of the
// same contract. Every case checks the mixed outputs, the published window
// (bit-exact: it is a copy) and state, and that no bank other than each
// slot's successor changes: accepted banks, the zero seed, and unrelated
// banks keep their exact bytes.

use magnitude_model_kernels::{qwen_recurrent_chunk, qwen_recurrent_step};
use seismic::{BackendName, Device, DeviceCatalog, Element, NativeSpecialization, Tensor};

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
    SlotCase { rows, stop, previous, following, taped: 0 }
}

struct Case {
    geometry: Geometry,
    bf16: bool,
    rows: usize,
    slots: Vec<SlotCase>,
    grouped: bool,
    epsilon: f32,
    projection: Vec<f32>,
    convolution: Vec<f32>,
    rate: Vec<f32>,
    time_bias: Vec<f32>,
    window: Vec<f32>,
    delta: Vec<f32>,
    tape: Vec<f32>,
}

struct Outcome {
    mixed: Vec<f32>,
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
    segments: Tensor,
    stop: Tensor,
    previous: Tensor,
    previous_tape: Tensor,
    following: Tensor,
    window: Tensor,
    delta: Tensor,
    tape: Tensor,
}

impl Tensors {
    fn step_args<'a>(&'a mut self, case: &Case) -> qwen_recurrent_step::Args<'a> {
        qwen_recurrent_step::Args {
            projection: &self.projection,
            convolution: &self.convolution,
            rate: &self.rate,
            time_bias: &self.time_bias,
            segments: &self.segments,
            stop: &self.stop,
            previous_bank: &self.previous,
            previous_tape: &self.previous_tape,
            following_bank: &self.following,
            window: &mut self.window,
            delta: &mut self.delta,
            tape: &mut self.tape,
            norm_epsilon: case.epsilon,
            grouped: case.grouped,
        }
    }

    fn chunk_args<'a>(&'a mut self, case: &Case) -> qwen_recurrent_chunk::Args<'a> {
        qwen_recurrent_chunk::Args {
            projection: &self.projection,
            convolution: &self.convolution,
            rate: &self.rate,
            time_bias: &self.time_bias,
            segments: &self.segments,
            stop: &self.stop,
            previous_bank: &self.previous,
            previous_tape: &self.previous_tape,
            following_bank: &self.following,
            window: &mut self.window,
            delta: &mut self.delta,
            tape: &mut self.tape,
            norm_epsilon: case.epsilon,
            grouped: case.grouped,
        }
    }
}

struct Random(u64);

impl Random {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn bf16_round(value: f32) -> f32 {
    let bits = value.to_bits();
    f32::from_bits((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff_0000)
}

impl Case {
    /// Rows are packed after the slots, as the batch packer pads them.
    fn new(geometry: Geometry, rows: usize, slots: Vec<SlotCase>, grouped: bool, seed: u64) -> Self {
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
        let time_bias = (0..geometry.value_heads).map(|_| random.next() * 0.5).collect();
        let accepted = slots.iter().map(|slot| slot.previous).collect::<Vec<_>>();
        let mut window = vec![SENTINEL; geometry.banks * geometry.window_bank()];
        let mut delta = vec![SENTINEL; geometry.banks * geometry.delta_bank()];
        let mut tape = vec![SENTINEL; geometry.banks * geometry.tape_bank()];
        window[..geometry.window_bank()].fill(0.0);
        delta[..geometry.delta_bank()].fill(0.0);
        tape[..geometry.tape_bank()].fill(0.0);
        for bank in 1..geometry.banks {
            if accepted.contains(&bank) {
                for value in &mut window[bank * geometry.window_bank()..][..geometry.window_bank()] {
                    *value = random.next();
                }
                for value in &mut delta[bank * geometry.delta_bank()..][..geometry.delta_bank()] {
                    *value = random.next() * 0.3;
                }
                // Tape rows: innovations, keys and decays in (0.5, 1).
                let nvw = geometry.value_heads * geometry.width;
                let nkw = geometry.key_heads * geometry.width;
                for entry in 0..geometry.tape {
                    let row = &mut tape[bank * geometry.tape_bank() + entry * geometry.tape_row()..]
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
            projection,
            convolution,
            rate,
            time_bias,
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
        let key_of = |head: usize| if self.grouped { head * nk / nv } else { head % nk };
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
                let row = &self.tape[slot.previous * g.tape_bank() + entry * g.tape_row()..][..g.tape_row()];
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
                    let alpha = self.projection[row * width + channels + nv * w + value_head] as f64;
                    let beta_input =
                        self.projection[row * width + channels + nv * w + nv + value_head] as f64;
                    let beta = 1.0 / (1.0 + (-beta_input).exp());
                    let shifted = alpha + self.time_bias[value_head] as f64;
                    let softplus = shifted.max(0.0) + (1.0 + (-shifted.abs()).exp()).ln();
                    let factor = (self.rate[value_head] as f64 * softplus).exp();
                    let query = &prepared[key_head * w..][..w];
                    let key = &prepared[(nk + key_head) * w..][..w];
                    let entry = (local >= slot.stop && local - slot.stop < taped)
                        .then(|| slot.following * g.tape_bank() + (local - slot.stop) * g.tape_row());
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
                    window[slot.following * g.window_bank() + row * channels + channel] =
                        raw(slot.stop as isize + row as isize - (c as isize - 1), channel);
                }
            }
            first += slot.rows;
        }
        Outcome {
            mixed,
            window,
            delta,
            tape,
        }
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
        let elements = ElementBindings::new().bind("A", registry::dense(DType::F32));
        let logical = module
            .entry(module.entry_named("qwen_recurrent_step").unwrap(), &elements)
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
            TensorData::dense(
                DType::I32,
                vec![self.slots.len() + 1, 2],
                self.segments().into_iter().map(f64::from).collect(),
            ),
            ints(self.slots.iter().map(|slot| slot.stop as i32).collect()),
            ints(self.slots.iter().map(|slot| slot.previous as i32).collect()),
            ints(self.slots.iter().map(|slot| slot.taped as i32).collect()),
            ints(self.slots.iter().map(|slot| slot.following as i32).collect()),
            floats(vec![g.banks, g.window_rows(), g.channels()], &self.window),
            floats(vec![g.banks, g.value_heads, g.width, g.width], &self.delta),
            floats(vec![g.banks, g.tape, g.tape_row()], &self.tape),
        ];
        let mut arguments = tensors
            .into_iter()
            .map(|tensor| Arg::Tensor(interpreter.add_tensor(tensor)))
            .collect::<Vec<_>>();
        arguments.push(Arg::Scalar(ReferenceScalar::F32(self.epsilon.to_bits())));
        arguments.push(Arg::Scalar(ReferenceScalar::Bool(self.grouped)));
        let outcome = interpreter.run_bounded(&arguments, u64::MAX).unwrap();
        if let SourceTermination::Failed(failure) = outcome.termination() {
            panic!("portable recurrent body failed: {failure}");
        }
        let read = |reader: seismic_lang::interp::TensorReader<'_>| {
            (0..reader.element_count())
                .map(|index| reader.read(index).unwrap() as f32)
                .collect::<Vec<_>>()
        };
        let mixed = match outcome.results().next().unwrap().value() {
            OutcomeValue::Tensor(reader) => read(reader),
            _ => panic!("mixed output is a tensor"),
        };
        let mut window = None;
        let mut delta = None;
        let mut tape = None;
        for input in outcome.inputs() {
            match input.ordinal() {
                9 => window = Some(read(input.tensor())),
                10 => delta = Some(read(input.tensor())),
                11 => tape = Some(read(input.tensor())),
                _ => {}
            }
        }
        Outcome {
            mixed,
            window: window.expect("window is a mutable input"),
            delta: delta.expect("delta is a mutable input"),
            tape: tape.expect("tape is a mutable input"),
        }
    }

    /// Device tensors of the case in `activation`.
    fn tensors(&self, device: &Device, activation: Element) -> Tensors {
        let g = self.geometry;
        let from_f32 = |shape: &[u64], values: &[f32]| {
            Tensor::from_host(
                device,
                Element::f32(),
                shape,
                &values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>(),
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
                values.iter().flat_map(|value| value.to_le_bytes()).collect()
            };
            Tensor::from_host(device, activation, shape, &bytes).unwrap()
        };
        let ints = |shape: &[u64], values: Vec<i32>| {
            Tensor::from_host(
                device,
                Element::i32(),
                shape,
                &values.into_iter().flat_map(i32::to_le_bytes).collect::<Vec<_>>(),
            )
            .unwrap()
        };
        let slots = self.slots.len() as u64;
        Tensors {
            projection: from_activation(
                &[self.rows as u64, g.projection_width() as u64],
                &self.projection,
            ),
            convolution: from_f32(&[g.channels() as u64, g.convolution as u64], &self.convolution),
            rate: from_f32(&[g.value_heads as u64], &self.rate),
            time_bias: from_f32(&[g.value_heads as u64], &self.time_bias),
            segments: ints(&[slots + 1, 2], self.segments()),
            stop: ints(&[slots], self.slots.iter().map(|slot| slot.stop as i32).collect()),
            previous: ints(&[slots], self.slots.iter().map(|slot| slot.previous as i32).collect()),
            previous_tape: ints(&[slots], self.slots.iter().map(|slot| slot.taped as i32).collect()),
            following: ints(&[slots], self.slots.iter().map(|slot| slot.following as i32).collect()),
            window: from_activation(
                &[g.banks as u64, g.window_rows() as u64, g.channels() as u64],
                &self.window,
            ),
            delta: from_f32(
                &[g.banks as u64, g.value_heads as u64, g.width as u64, g.width as u64],
                &self.delta,
            ),
            tape: from_f32(&[g.banks as u64, g.tape as u64, g.tape_row() as u64], &self.tape),
        }
    }

    fn statics(&self) -> NativeSpecialization {
        let g = self.geometry;
        NativeSpecialization::new()
            .with_static("NK", g.key_heads as u64)
            .with_static("NV", g.value_heads as u64)
            .with_static("W", g.width as u64)
            .with_static("C", g.convolution as u64)
    }

    /// The Metal step with `ROWS` state rows per threadgroup.
    fn metal_step(&self, device: &Device, activation: Element, rows: u64)
        -> seismic::NativeKernel<qwen_recurrent_step::Entry> {
        qwen_recurrent_step::native_for_device_with(
            device,
            qwen_recurrent_step::Elements { A: activation },
            &self.statics().with_param("ROWS", rows),
        )
        .unwrap()
    }

    /// The Metal chunk with `ROWS` state rows per threadgroup.
    fn metal_chunk(&self, device: &Device, activation: Element, rows: u64)
        -> seismic::NativeKernel<qwen_recurrent_chunk::Entry> {
        qwen_recurrent_chunk::native_for_device_with(
            device,
            qwen_recurrent_chunk::Elements { A: activation },
            &self.statics().with_param("ROWS", rows),
        )
        .unwrap()
    }

    /// The Metal step (`None`) or the chunk with `ROWS` (`Some`).
    fn native(&self, device: &Device, activation: Element, chunk: Option<u64>) -> Outcome {
        let mut t = self.tensors(device, activation);
        let mixed = match chunk {
            None => self
                .metal_step(device, activation, 32.min(self.geometry.width as u64))
                .call(t.step_args(self))
                .unwrap()
                .value,
            Some(mapping) => self
                .metal_chunk(device, activation, mapping)
                .call(t.chunk_args(self))
                .unwrap()
                .value,
        };
        Outcome {
            mixed: read(&mixed),
            window: read(&t.window),
            delta: read(&t.delta),
            tape: read(&t.tape),
        }
    }

    fn successors(&self) -> Vec<usize> {
        self.slots.iter().map(|slot| slot.following).collect()
    }
}

fn read(tensor: &Tensor) -> Vec<f32> {
    let bytes = tensor.read_to_host().unwrap();
    if tensor.element() == Element::bf16() {
        bytes
            .chunks_exact(2)
            .map(|word| f32::from_bits(u32::from(u16::from_le_bytes(word.try_into().unwrap())) << 16))
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
    let scale = (expected.iter().map(|v| (*v as f64).powi(2)).sum::<f64>()
        / expected.len() as f64)
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

/// Compare one outcome with the reference. Mixed outputs and successor state
/// are held to `(max, rms)` relative tolerances; windows are copies and must
/// be bit-exact; every non-successor bank must be untouched. BF16 mixed
/// outputs are compared with the reference rounded to BF16, each within one
/// BF16 unit in the last place of its magnitude (plus the relative bound).
fn check(label: &str, case: &Case, actual: &Outcome, expected: &Outcome, tolerance: (f64, f64)) {
    let g = case.geometry;
    let (max, rms) = errors(&actual.mixed, &expected.mixed);
    println!("{label} mixed: max/rms {max:.3e} rms/rms {rms:.3e}");
    if case.bf16 {
        let scale = (expected.mixed.iter().map(|v| (*v as f64).powi(2)).sum::<f64>()
            / expected.mixed.len() as f64)
            .sqrt();
        for (index, (a, e)) in actual.mixed.iter().zip(&expected.mixed).enumerate() {
            let rounded = bf16_round(*e) as f64;
            let allowed = rounded.abs() * 2f64.powi(-7) + tolerance.0 * scale;
            assert!(
                (*a as f64 - rounded).abs() <= allowed,
                "{label} mixed[{index}] {a} vs {rounded}"
            );
        }
        assert!(rms <= tolerance.1, "{label} mixed rms error {rms}");
    } else {
        assert!(max <= tolerance.0 && rms <= tolerance.1, "{label} mixed error {max} {rms}");
    }
    for bank in 0..g.banks {
        let window = &actual.window[bank * g.window_bank()..][..g.window_bank()];
        let delta = &actual.delta[bank * g.delta_bank()..][..g.delta_bank()];
        let expected_window = &expected.window[bank * g.window_bank()..][..g.window_bank()];
        let expected_delta = &expected.delta[bank * g.delta_bank()..][..g.delta_bank()];
        assert!(
            window.iter().zip(expected_window).all(|(a, e)| a.to_bits() == e.to_bits()),
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
            assert!(max <= tolerance.0 && rms <= tolerance.1, "{label} state error {max} {rms}");
            if recorded > 0 {
                let expected_tape = &expected.tape[bank * g.tape_bank()..][..recorded];
                let (max, rms) = errors(&tape[..recorded], expected_tape);
                println!("{label} tape bank {bank}: max/rms {max:.3e} rms/rms {rms:.3e}");
                assert!(max <= tolerance.0 && rms <= tolerance.1, "{label} tape error {max} {rms}");
            }
            assert!(
                tape[recorded..].iter().zip(&original_tape[recorded..]).all(|(a, e)| a.to_bits() == e.to_bits()),
                "{label} wrote tape rows of bank {bank} past its recorded rows"
            );
        } else {
            let original = &case.delta[bank * g.delta_bank()..][..g.delta_bank()];
            assert!(
                delta.iter().zip(original).all(|(a, e)| a.to_bits() == e.to_bits())
                    && tape.iter().zip(original_tape).all(|(a, e)| a.to_bits() == e.to_bits()),
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

/// The Metal chunk's `ROWS` mappings.
const METAL_CHUNK_ROWS: [u64; 3] = [128, 64, 32];

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
            Case::new(SMALL, 8, vec![slot(2, 2, 1, 4), slot(1, 1, 2, 5), slot(3, 3, 0, 3)], true, 2),
        ),
        (
            "interior stop rows and a zero stop",
            Case::new(SMALL, 16, vec![slot(7, 4, 1, 3), slot(5, 0, 2, 4), slot(4, 1, 1, 5)], false, 3),
        ),
        (
            "pieces across slots with a stop inside a piece and resets",
            Case::new(SMALL, 128, vec![slot(70, 33, 1, 4), slot(45, 45, 2, 3)], true, 4)
                .with_resets(&[5, 40, 71, 100]),
        ),
    ]
}

#[test]
fn step_and_chunk_match_the_portable_body() {
    let Some(device) = metal() else {
        return;
    };
    for (label, case) in small_cases() {
        let oracle = case.oracle();
        let host = case.host();
        // The f64 host model and the interpreted body agree closely; this
        // pins the host model used for the real geometry below.
        check(&format!("{label}: host vs body"), &case, &host, &oracle, (1e-4, 1e-5));
        let step = case.native(&device, Element::f32(), None);
        check(&format!("{label}: step"), &case, &step, &oracle, (2e-5, 2e-6));
        for rows in METAL_CHUNK_ROWS {
            let rows = rows.min(case.geometry.width as u64);
            let chunked = case.native(&device, Element::f32(), Some(rows));
            check(&format!("{label}: chunk ROWS {rows}"), &case, &chunked, &oracle, (5e-4, 2e-5));
        }
    }
}

/// Small cases over tape versions (T = 3): slots that start from a version
/// with tape rows, record the rows after their stop row (fewer, exactly, or
/// more than T), in short and chunked slots, grouped or not.
fn tape_cases() -> Vec<(&'static str, Case)> {
    let geometry = Geometry { banks: 7, tape: 3, ..SMALL };
    let version = |rows, stop, previous, following, taped| SlotCase { rows, stop, previous, following, taped };
    vec![
        (
            "verify slots from tape versions",
            Case::new(
                geometry,
                16,
                vec![version(4, 1, 1, 4, 2), version(6, 2, 2, 5, 0), version(3, 3, 3, 6, 3)],
                false,
                51,
            ),
        ),
        (
            "chunked slots with tails, from tape versions",
            Case::new(geometry, 64, vec![version(40, 35, 1, 4, 1), version(20, 20, 2, 5, 3)], true, 52)
                .with_resets(&[7, 45]),
        ),
    ]
}

/// A committed tape version (bank, j) equals a run that published after those
/// rows: the next advance from either has the same bits. `run` executes a
/// case on one entry; the tentative advance is a 5-row verify slot (stop 1).
fn tape_versions_equal_stopped_runs(label: &str, geometry: Geometry, run: &dyn Fn(&Case) -> Outcome) {
    let verify = 5;
    let g = Geometry { banks: 4, tape: verify - 1, ..geometry };
    for accepted in 0..verify {
        let tentative = Case::new(g, verify, vec![slot(verify, 1, 1, 2)], true, 41).with_bf16_activations();
        let first = run(&tentative);
        let next = SlotCase { rows: 3, stop: 3, previous: 2, following: 3, taped: accepted };
        let continued = tentative.continued(&first, 3, vec![next], 42);
        let from_tape = run(&continued);
        let stopped = Case::new(g, verify, vec![slot(verify, 1 + accepted, 1, 2)], true, 41).with_bf16_activations();
        let reference_first = run(&stopped);
        let reference = stopped.continued(&reference_first, 3, vec![slot(3, 3, 2, 3)], 42);
        let expected = run(&reference);
        let bank = |values: &[f32], size: usize| values[3 * size..4 * size].to_vec();
        assert!(
            from_tape.mixed.iter().zip(&expected.mixed).all(|(a, b)| a.to_bits() == b.to_bits())
                && bank(&from_tape.delta, g.delta_bank()) == bank(&expected.delta, g.delta_bank())
                && bank(&from_tape.window, g.window_bank()) == bank(&expected.window, g.window_bank()),
            "{label}: version (bank, {accepted}) differs from a run that stopped after {} rows",
            1 + accepted
        );
        check(&format!("{label}: from version (bank, {accepted})"), &continued, &from_tape, &continued.host(), (1.5e-2, 3e-3));
    }
}

#[test]
fn tape_cases_match_the_portable_body() {
    let Some(device) = metal() else {
        return;
    };
    for (label, case) in tape_cases() {
        let oracle = case.oracle();
        check(&format!("{label}: host vs body"), &case, &case.host(), &oracle, (1e-4, 1e-5));
        check(&format!("{label}: step"), &case, &case.native(&device, Element::f32(), None), &oracle, (2e-5, 2e-6));
        for rows in METAL_CHUNK_ROWS {
            let rows = rows.min(case.geometry.width as u64);
            let chunked = case.native(&device, Element::f32(), Some(rows));
            check(&format!("{label}: chunk ROWS {rows}"), &case, &chunked, &oracle, (5e-4, 2e-5));
        }
    }
}

#[test]
fn tape_versions_equal_runs_that_stopped_there() {
    let Some(device) = metal() else {
        return;
    };
    let qwen = Geometry { key_heads: 16, value_heads: 32, width: 128, convolution: 4, banks: 4, tape: 0 };
    for geometry in [SMALL, qwen] {
        tape_versions_equal_stopped_runs("step", geometry, &|case| case.native(&device, Element::bf16(), None));
        tape_versions_equal_stopped_runs("chunk", geometry, &|case| case.native(&device, Element::bf16(), Some(32)));
    }
}

#[test]
fn step_row_block_never_changes_bits_and_stop_equals_a_shorter_run() {
    let Some(device) = metal() else {
        return;
    };
    let slot = |rows, stop| slot(rows, stop, 1, 2);
    let full = Case::new(SMALL, 6, vec![slot(6, 3)], false, 11);
    let mut reference = None;
    for rows in [16u64, 32] {
        let mut t = full.tensors(&device, Element::f32());
        let mixed = full.metal_step(&device, Element::f32(), rows).call(t.step_args(&full)).unwrap().value;
        let outcome = (read(&mixed), read(&t.window), read(&t.delta));
        match &reference {
            None => reference = Some(outcome),
            Some(reference) => assert!(
                reference.0.iter().zip(&outcome.0).all(|(a, b)| a.to_bits() == b.to_bits())
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
    let short = prefix.native(&device, Element::f32(), None);
    assert!(
        short.delta.iter().zip(&full_delta).all(|(a, b)| a.to_bits() == b.to_bits()),
        "stop-row state differs from the state of a shorter run"
    );
    assert!(
        short.window.iter().zip(&full_window).all(|(a, b)| a.to_bits() == b.to_bits()),
        "stop-row window differs from the window of a shorter run"
    );
}

#[test]
fn real_4b_geometry_step_and_chunk_agree_with_the_host_model() {
    let Some(device) = metal() else {
        return;
    };
    let geometry = Geometry {
        key_heads: 16,
        value_heads: 32,
        width: 128,
        convolution: 4,
        banks: 5,
        tape: 0,
    };
    for (label, rows, slots) in [
        ("decode, one slot", 1, vec![slot(1, 1, 1, 3)]),
        ("verify, two slots", 8, vec![slot(4, 2, 1, 3), slot(3, 3, 0, 4)]),
        ("prefill 128", 128, vec![slot(128, 128, 1, 3)]),
        ("prefill 512, two slots", 512, vec![slot(300, 211, 1, 3), slot(212, 212, 2, 4)]),
    ] {
        let case = Case::new(geometry, rows, slots, false, 21).with_bf16_activations();
        let host = case.host();
        let step = case.native(&device, Element::bf16(), None);
        check(&format!("4B {label}: step"), &case, &step, &host, (1e-4, 3e-3));
        if rows >= 16 {
            let mut reference: Option<Outcome> = None;
            for rows in METAL_CHUNK_ROWS {
                let chunked = case.native(&device, Element::bf16(), Some(rows));
                check(&format!("4B {label}: chunk ROWS {rows}"), &case, &chunked, &host, (1e-3, 3e-3));
                let (max, rms) = errors(&chunked.mixed, &step.mixed);
                println!("4B {label}: chunk ROWS {rows} vs step: max {max:.3e} rms {rms:.3e}");
                // Both are within one BF16 ulp of the host model per element
                // (checked above), so they may differ by two ulps of the
                // largest output (~8x the RMS at 512 rows).
                assert!(max <= 4e-2 && rms <= 3e-3);
                // ROWS never changes result bits.
                match &reference {
                    Some(reference) => assert!(
                        reference.mixed.iter().zip(&chunked.mixed).all(|(a, b)| a.to_bits() == b.to_bits())
                            && reference.delta.iter().zip(&chunked.delta).all(|(a, b)| a.to_bits() == b.to_bits()),
                        "4B {label}: chunk ROWS {rows} changed result bits"
                    ),
                    None => reference = Some(chunked),
                }
            }
        }
    }
}

/// MTP verify: a slot of at most 16 rows gets the step's bits from the chunk
/// entry too, whatever its peers (here a 40-row slot on the chunked path), so
/// a request's verify rows never depend on the class.
#[test]
fn chunk_short_slots_get_the_step_bits() {
    let Some(device) = metal() else {
        return;
    };
    let geometry = Geometry { key_heads: 16, value_heads: 32, width: 128, convolution: 4, banks: 11, tape: 0 };
    let slots = vec![slot(4, 1, 1, 6), slot(16, 9, 2, 7), slot(40, 40, 3, 8), slot(1, 1, 4, 9), slot(7, 0, 5, 10)];
    let case = Case::new(geometry, 70, slots, false, 31).with_bf16_activations();
    let step = case.native(&device, Element::bf16(), None);
    let host = case.host();
    let row_elements = geometry.value_heads * geometry.width;
    for rows in METAL_CHUNK_ROWS {
        let chunk = case.native(&device, Element::bf16(), Some(rows));
        let mut first = 0;
        for s in &case.slots {
            let range = first * row_elements..(first + s.rows) * row_elements;
            first += s.rows;
            if s.rows > 16 {
                continue;
            }
            let bank = s.following * geometry.delta_bank()..(s.following + 1) * geometry.delta_bank();
            assert!(
                step.mixed[range.clone()].iter().zip(&chunk.mixed[range]).all(|(a, b)| a.to_bits() == b.to_bits())
                    && step.delta[bank.clone()].iter().zip(&chunk.delta[bank]).all(|(a, b)| a.to_bits() == b.to_bits()),
                "chunk ROWS {rows}: a {}-row slot differs from the step",
                s.rows
            );
        }
        check(&format!("4B verify mix: chunk ROWS {rows}"), &case, &chunk, &host, (1e-3, 3e-3));
    }
}

/// The median device time (µs) of each launch of the calls `run` makes, each
/// launch in its own timed unit, joined as "a + b".
fn launch_medians(device: &Device, run: impl FnOnce()) -> String {
    let trace = device.trace_submissions(seismic::TraceDetail::Launches).unwrap();
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
        ("step 8 slots x 1 row", 8, (0..8).map(|s| slot(1, 1 + s, 9 + s)).collect::<Vec<_>>()),
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
                let kernel = case.metal_step(&device, Element::bf16(), step_rows);
                let args = rotation.iter_mut().map(|t| t.step_args(&case)).collect();
                let measured = kernel.measure(args, &options).unwrap();
                println!("{label} ROWS {step_rows}: {:.1} us", measured.median * 1e6);
            }
        } else {
            for chunk_rows in METAL_CHUNK_ROWS {
                let kernel = case.metal_chunk(&device, Element::bf16(), chunk_rows);
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
