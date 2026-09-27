// `state_space_step` and `state_space_chunk` against their shared portable
// body (`state_space_rows`), on every backend this host opens.
//
// Small geometries run the body in the Seismic interpreter; the Nemotron-H
// geometries (Lightning, Super, Ultra) use an independent f64 host model of
// the same contract, pinned to the interpreter on the small cases. Every case
// checks the mixed outputs, the published state and tape, the published
// window (bit-exact: it is a copy), and that no bank other than each slot's
// successor changes: accepted banks, the zero seed, and unrelated banks keep
// their exact bytes.

mod versions_common;

use magnitude_kernels::{state_space_chunk, state_space_gate, state_space_step};
use seismic::{Device, Element, NativeSpecialization, Tensor};
use versions_common::*;

#[derive(Clone, Copy, Debug)]
struct Geometry {
    heads: usize,
    head_width: usize,
    groups: usize,
    state_width: usize,
    convolution: usize,
    banks: usize,
    /// Tape rows per bank (T).
    tape: usize,
}

impl Geometry {
    /// Convolved channels x | B | C.
    fn channels(self) -> usize {
        self.heads * self.head_width + 2 * self.groups * self.state_width
    }
    fn projection_width(self) -> usize {
        2 * self.heads * self.head_width + 2 * self.groups * self.state_width + self.heads
    }
    fn window_rows(self) -> usize {
        self.convolution - 1 + self.tape
    }
    fn window_bank(self) -> usize {
        self.window_rows() * self.channels()
    }
    fn state_bank(self) -> usize {
        self.heads * self.head_width * self.state_width
    }
    /// Floats of one tape row: u [NH, P] | B [G, N] | d [NH].
    fn tape_row(self) -> usize {
        self.heads * self.head_width + self.groups * self.state_width + self.heads
    }
    fn tape_bank(self) -> usize {
        self.tape * self.tape_row()
    }
    fn group_of(self, head: usize) -> usize {
        head * self.groups / self.heads
    }
}

#[derive(Clone)]
struct Case {
    geometry: Geometry,
    bf16: bool,
    rows: usize,
    slots: Vec<SlotCase>,
    projection: Vec<f32>,
    convolution: Vec<f32>,
    bias: Vec<f32>,
    rate: Vec<f32>,
    time_bias: Vec<f32>,
    skip: Vec<f32>,
    window: Vec<f32>,
    state: Vec<f32>,
    tape: Vec<f32>,
    slab_banks: u64,
}

struct Outcome {
    mixed: Vec<f32>,
    window: Vec<f32>,
    state: Vec<f32>,
    tape: Vec<f32>,
}

impl Case {
    /// Rows are packed after the slots, as the batch packer pads them.
    fn new(geometry: Geometry, rows: usize, slots: Vec<SlotCase>, seed: u64) -> Self {
        let g = geometry;
        let mut random = Random(seed);
        let width = g.projection_width();
        let mut projection = (0..rows * width).map(|_| random.next()).collect::<Vec<_>>();
        let dt = 2 * g.heads * g.head_width + 2 * g.groups * g.state_width;
        for row in 0..rows {
            for head in 0..g.heads {
                projection[row * width + dt + head] = random.next() * 3.0;
            }
        }
        let convolution = (0..g.channels() * g.convolution).map(|_| random.next() * 0.6).collect();
        let bias = (0..g.channels()).map(|_| random.next() * 0.3).collect();
        // ssm_a = -exp(A_log): decays from long to short memory.
        let rate = (0..g.heads).map(|_| -(0.3 + 4.0 * (random.next() + 1.0))).collect();
        // dt_bias: step sizes from about 0.02 to 2.
        let time_bias = (0..g.heads).map(|_| -1.5 + 1.5 * random.next()).collect();
        let skip = (0..g.heads).map(|_| random.next()).collect();
        let accepted = slots.iter().map(|slot| slot.previous).collect::<Vec<_>>();
        let mut window = vec![SENTINEL; g.banks * g.window_bank()];
        let mut state = vec![SENTINEL; g.banks * g.state_bank()];
        let mut tape = vec![SENTINEL; g.banks * g.tape_bank()];
        window[..g.window_bank()].fill(0.0);
        state[..g.state_bank()].fill(0.0);
        tape[..g.tape_bank()].fill(0.0);
        for bank in 1..g.banks {
            if accepted.contains(&bank) {
                for value in &mut window[bank * g.window_bank()..][..g.window_bank()] {
                    *value = random.next();
                }
                for value in &mut state[bank * g.state_bank()..][..g.state_bank()] {
                    *value = random.next() * 0.5;
                }
                // Tape rows: inputs, B and decays in (0.5, 1).
                let (u, b) = (g.heads * g.head_width, g.groups * g.state_width);
                for entry in 0..g.tape {
                    let row = &mut tape[bank * g.tape_bank() + entry * g.tape_row()..][..g.tape_row()];
                    for (index, value) in row.iter_mut().enumerate() {
                        *value = if index < u {
                            random.next() * 0.3
                        } else if index < u + b {
                            random.next()
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
            projection,
            convolution,
            bias,
            rate,
            time_bias,
            skip,
            window,
            state,
            tape,
            slab_banks: 2,
        }
    }

    /// The next advance of the same layer after `outcome`: new projection
    /// rows for `slots` (reading the versions they name), the same weights,
    /// and the arenas `outcome` left.
    fn continued(&self, outcome: &Outcome, rows: usize, slots: Vec<SlotCase>, seed: u64) -> Self {
        let fresh = Case::new(self.geometry, rows, slots, seed);
        let projection = if self.bf16 {
            fresh.projection.iter().map(|value| bf16_round(*value)).collect()
        } else {
            fresh.projection.clone()
        };
        Self {
            projection,
            convolution: self.convolution.clone(),
            bias: self.bias.clone(),
            rate: self.rate.clone(),
            time_bias: self.time_bias.clone(),
            skip: self.skip.clone(),
            window: outcome.window.clone(),
            state: outcome.state.clone(),
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

    fn activation(&self) -> Element {
        if self.bf16 {
            Element::bf16()
        } else {
            Element::f32()
        }
    }

    /// The f64 host model of the entry contract.
    fn host(&self) -> Outcome {
        let g = self.geometry;
        let (nh, p, n, c) = (g.heads, g.head_width, g.state_width, g.convolution);
        let (channels, width) = (g.channels(), g.projection_width());
        let x_column = nh * p;
        let dt = 2 * nh * p + 2 * g.groups * n;
        let mut mixed = vec![0.0f32; self.rows * nh * p];
        let mut window = self.window.clone();
        let mut state = self.state.clone();
        let mut tape = self.tape.clone();
        for (slot, first) in self.slots.iter().zip(firsts(&self.slots)) {
            let mut current = state[slot.previous * g.state_bank()..][..g.state_bank()]
                .iter()
                .map(|value| *value as f64)
                .collect::<Vec<_>>();
            for entry in 0..slot.taped {
                let row = &self.tape[slot.previous * g.tape_bank() + entry * g.tape_row()..][..g.tape_row()];
                for head in 0..nh {
                    let decay = row[nh * p + g.groups * n + head] as f64;
                    let b = &row[nh * p + g.group_of(head) * n..][..n];
                    for channel in 0..p {
                        let input = row[head * p + channel] as f64;
                        let s = &mut current[(head * p + channel) * n..][..n];
                        for column in 0..n {
                            s[column] = s[column] * decay + input * b[column] as f64;
                        }
                    }
                }
            }
            let publish = |current: &[f64], state: &mut Vec<f32>| {
                for (index, value) in current.iter().enumerate() {
                    state[slot.following * g.state_bank() + index] = *value as f32;
                }
            };
            if slot.stop == 0 {
                publish(&current, &mut state);
            }
            let raw = |position: isize, channel: usize| -> f32 {
                if position < 0 {
                    self.window[slot.previous * g.window_bank()
                        + (slot.taped as isize + c as isize - 1 + position) as usize * channels
                        + channel]
                } else {
                    self.projection[(first + position as usize) * width + x_column + channel]
                }
            };
            let recorded = g.tape.min(slot.rows - slot.stop);
            for local in 0..slot.rows {
                let row = first + local;
                let prepared = (0..channels)
                    .map(|channel| {
                        let mut sum = self.bias[channel] as f64;
                        for tap in 0..c {
                            sum += self.convolution[channel * c + tap] as f64
                                * raw(local as isize + tap as isize - (c as isize - 1), channel) as f64;
                        }
                        sum / (1.0 + (-sum).exp())
                    })
                    .collect::<Vec<_>>();
                let entry = (local >= slot.stop && local - slot.stop < recorded)
                    .then(|| slot.following * g.tape_bank() + (local - slot.stop) * g.tape_row());
                for head in 0..nh {
                    let group = g.group_of(head);
                    let b = &prepared[nh * p + group * n..][..n];
                    let cc = &prepared[nh * p + g.groups * n + group * n..][..n];
                    let shifted = self.projection[row * width + dt + head] as f64 + self.time_bias[head] as f64;
                    let delta = shifted.max(0.0) + (1.0 + (-shifted.abs()).exp()).ln();
                    let decay = (self.rate[head] as f64 * delta).exp();
                    if let Some(entry) = entry {
                        tape[entry + nh * p + g.groups * n + head] = decay as f32;
                    }
                    for channel in 0..p {
                        let value = prepared[head * p + channel];
                        let input = delta * value;
                        if let Some(entry) = entry {
                            tape[entry + head * p + channel] = input as f32;
                        }
                        let s = &mut current[(head * p + channel) * n..][..n];
                        let mut output = 0.0;
                        for column in 0..n {
                            s[column] = s[column] * decay + input * b[column];
                            output += s[column] * cc[column];
                        }
                        mixed[(row * nh + head) * p + channel] = (output + self.skip[head] as f64 * value) as f32;
                    }
                }
                if let Some(entry) = entry {
                    for column in 0..g.groups * n {
                        tape[entry + nh * p + column] = prepared[nh * p + column] as f32;
                    }
                }
                if local + 1 == slot.stop {
                    publish(&current, &mut state);
                }
            }
            for row in 0..c - 1 + recorded {
                for channel in 0..channels {
                    window[slot.following * g.window_bank() + row * channels + channel] =
                        raw(slot.stop as isize + row as isize - (c as isize - 1), channel);
                }
            }
        }
        Outcome {
            mixed,
            window,
            state,
            tape,
        }
    }

    /// The portable body executed by the Seismic interpreter (F32
    /// activations).
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
        assert!(!self.bf16);
        let mut sources = seismic_std::sources();
        sources.push(SourceFile {
            path: "state_space.seismic".into(),
            text: include_str!("../kernels/state_space.seismic").into(),
        });
        let module = check_source(sources).unwrap();
        let elements = ElementBindings::new().bind("A", registry::dense(DType::F32));
        let logical = module
            .entry(module.entry_named("state_space_step").unwrap(), &elements)
            .unwrap();
        let g = self.geometry;
        let floats = |shape: Vec<usize>, values: &[f32]| {
            TensorData::dense(DType::F32, shape, values.iter().map(|value| f64::from(*value)).collect())
        };
        let ints = |shape: Vec<usize>, values: Vec<i32>| {
            TensorData::dense(DType::I32, shape, values.into_iter().map(f64::from).collect())
        };
        let each = |f: fn(&SlotCase) -> usize| self.slots.iter().map(|slot| f(slot) as i32).collect::<Vec<_>>();
        let count = self.slots.len();
        let mut interpreter = Interpreter::new(&logical);
        let tensors = vec![
            floats(vec![self.rows, g.projection_width()], &self.projection),
            floats(vec![g.channels(), g.convolution], &self.convolution),
            floats(vec![g.channels()], &self.bias),
            floats(vec![g.heads], &self.rate),
            floats(vec![g.heads], &self.time_bias),
            floats(vec![g.heads], &self.skip),
            ints(vec![count + 1, 2], segments(&self.slots, self.rows)),
            ints(vec![count], each(|slot| slot.stop)),
            ints(vec![count], each(|slot| slot.previous)),
            ints(vec![count], each(|slot| slot.taped)),
            ints(vec![count], each(|slot| slot.following)),
            floats(vec![g.banks, g.window_rows(), g.channels()], &self.window),
            floats(vec![g.banks, g.heads, g.head_width, g.state_width], &self.state),
            floats(vec![g.banks, g.tape, g.tape_row()], &self.tape),
        ];
        let mut arguments = tensors
            .into_iter()
            .map(|tensor| Arg::Tensor(interpreter.add_tensor(tensor)))
            .collect::<Vec<_>>();
        arguments.push(Arg::Scalar(ReferenceScalar::U32(self.slab_banks as u32)));
        let outcome = interpreter.run_bounded(&arguments, u64::MAX).unwrap();
        if let SourceTermination::Failed(failure) = outcome.termination() {
            panic!("portable state-space body failed: {failure}");
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
        let input = |ordinal| {
            outcome
                .inputs()
                .find(|input| input.ordinal() == ordinal)
                .map(|input| read(input.tensor()))
                .expect("a mutable input")
        };
        Outcome {
            mixed,
            window: input(11),
            state: input(12),
            tape: input(13),
        }
    }

    /// The specialization with `ROWS` state rows per threadgroup (CPU entries
    /// have no static dimensions).
    fn specialization(&self, device: &Device, rows: u64) -> NativeSpecialization {
        if is_cpu(device) {
            return NativeSpecialization::new().with_param("ROWS", rows);
        }
        let g = self.geometry;
        NativeSpecialization::new()
            .with_static("NH", g.heads as u64)
            .with_static("P", g.head_width as u64)
            .with_static("G", g.groups as u64)
            .with_static("N", g.state_width as u64)
            .with_static("C", g.convolution as u64)
            .with_param("ROWS", rows)
    }

    /// The step (`chunk` false) or the chunk with `ROWS` state rows per
    /// threadgroup; `reuse` places the banks with an interior slab freed and
    /// reused.
    fn native(&self, device: &Device, chunk: bool, rows: u64, reuse: bool) -> Outcome {
        let mut bound = self.bind(device, reuse);
        let specialization = self.specialization(device, rows);
        let activation = self.activation();
        let mixed: Tensor = if chunk {
            state_space_chunk::native_for_device_with(
                device,
                state_space_chunk::Elements { A: activation },
                &specialization,
            )
            .unwrap()
            .call(bound.chunk_args(self.slab_banks))
            .unwrap()
            .value
        } else {
            state_space_step::native_for_device_with(
                device,
                state_space_step::Elements { A: activation },
                &specialization,
            )
            .unwrap()
            .call(bound.step_args(self.slab_banks))
            .unwrap()
            .value
        };
        Outcome {
            mixed: read(&mixed),
            window: read(&bound.window),
            state: read(&bound.state),
            tape: read(&bound.tape),
        }
    }

    /// The case's device tensors; `reuse` places the banks with an interior
    /// slab freed and reused.
    fn bind(&self, device: &Device, reuse: bool) -> Bound {
        let g = self.geometry;
        let activation = self.activation();
        let mut arenas = vec![
            Arena {
                element: activation,
                bank_shape: vec![g.window_rows() as u64, g.channels() as u64],
                values: &self.window,
            },
            Arena {
                element: Element::f32(),
                bank_shape: vec![g.heads as u64, g.head_width as u64, g.state_width as u64],
                values: &self.state,
            },
        ];
        if g.tape > 0 {
            arenas.push(Arena {
                element: Element::f32(),
                bank_shape: vec![g.tape as u64, g.tape_row() as u64],
                values: &self.tape,
            });
        }
        let (slabs, mut logical) = slab_arenas(device, g.banks, self.slab_banks, &arenas, reuse);
        let tape = if g.tape > 0 {
            logical.pop().unwrap()
        } else {
            tensor(device, Element::f32(), &[g.banks as u64, 0, g.tape_row() as u64], &[])
        };
        let state = logical.pop().unwrap();
        let window = logical.pop().unwrap();
        let heads = g.heads as u64;
        Bound {
            _slabs: slabs,
            projection: tensor(device, activation, &[self.rows as u64, g.projection_width() as u64], &self.projection),
            convolution: tensor(device, Element::f32(), &[g.channels() as u64, g.convolution as u64], &self.convolution),
            bias: tensor(device, Element::f32(), &[g.channels() as u64], &self.bias),
            rate: tensor(device, Element::f32(), &[heads], &self.rate),
            time_bias: tensor(device, Element::f32(), &[heads], &self.time_bias),
            skip: tensor(device, Element::f32(), &[heads], &self.skip),
            slots: SlotTensors::new(device, &self.slots, self.rows),
            window,
            state,
            tape,
        }
    }
}

/// Device tensors of one case; each call mutates its bank arenas.
struct Bound {
    _slabs: seismic::SlabTensor,
    projection: Tensor,
    convolution: Tensor,
    bias: Tensor,
    rate: Tensor,
    time_bias: Tensor,
    skip: Tensor,
    slots: SlotTensors,
    window: Tensor,
    state: Tensor,
    tape: Tensor,
}

impl Bound {
    fn step_args(&mut self, slab_banks: u64) -> state_space_step::Args<'_> {
        state_space_step::Args {
            projection: &self.projection,
            convolution: &self.convolution,
            convolution_bias: &self.bias,
            rate: &self.rate,
            time_bias: &self.time_bias,
            skip: &self.skip,
            segments: &self.slots.segments,
            stop: &self.slots.stop,
            previous_bank: &self.slots.previous,
            previous_tape: &self.slots.previous_tape,
            following_bank: &self.slots.following,
            window: &mut self.window,
            state: &mut self.state,
            tape: &mut self.tape,
            slab_banks: slab_banks as u32,
        }
    }

    fn chunk_args(&mut self, slab_banks: u64) -> state_space_chunk::Args<'_> {
        state_space_chunk::Args {
            projection: &self.projection,
            convolution: &self.convolution,
            convolution_bias: &self.bias,
            rate: &self.rate,
            time_bias: &self.time_bias,
            skip: &self.skip,
            segments: &self.slots.segments,
            stop: &self.slots.stop,
            previous_bank: &self.slots.previous,
            previous_tape: &self.slots.previous_tape,
            following_bank: &self.slots.following,
            window: &mut self.window,
            state: &mut self.state,
            tape: &mut self.tape,
            slab_banks: slab_banks as u32,
        }
    }
}

/// Max absolute error relative to the reference's RMS, and the plain RMS of
/// the difference relative to it.
fn errors(actual: &[f32], expected: &[f32]) -> (f64, f64) {
    assert_eq!(actual.len(), expected.len());
    let scale = (expected.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / expected.len().max(1) as f64)
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
    (max / scale, (squares / actual.len().max(1) as f64).sqrt() / scale)
}

/// Mixed outputs within `(max, rms)` relative tolerances; BF16 outputs are
/// compared with the reference rounded to BF16, each within one BF16 unit of
/// its magnitude plus the relative bound.
fn check_mixed(label: &str, bf16: bool, actual: &[f32], expected: &[f32], tolerance: (f64, f64)) {
    let (max, rms) = errors(actual, expected);
    println!("{label} mixed: max/rms {max:.3e} rms/rms {rms:.3e}");
    if bf16 {
        let scale = (expected.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / expected.len() as f64).sqrt();
        for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
            let rounded = bf16_round(*e) as f64;
            let allowed = rounded.abs() * 2f64.powi(-7) + tolerance.0 * scale;
            assert!((*a as f64 - rounded).abs() <= allowed, "{label} mixed[{index}] {a} vs {rounded}");
        }
        assert!(rms <= tolerance.1, "{label} mixed rms error {rms}");
    } else {
        assert!(max <= tolerance.0 && rms <= tolerance.1, "{label} mixed error {max} {rms}");
    }
}

/// Compare one outcome with the reference. Mixed outputs and successor state
/// and tape are held to `(max, rms)` relative tolerances (BF16 outputs also
/// to one BF16 unit of their magnitude); windows are copies and must be
/// bit-exact; every non-successor bank must be untouched.
fn check(label: &str, case: &Case, actual: &Outcome, expected: &Outcome, tolerance: (f64, f64)) {
    let g = case.geometry;
    check_mixed(label, case.bf16, &actual.mixed, &expected.mixed, tolerance);
    for bank in 0..g.banks {
        let window = &actual.window[bank * g.window_bank()..][..g.window_bank()];
        let expected_window = &expected.window[bank * g.window_bank()..][..g.window_bank()];
        assert!(same_bits(window, expected_window), "{label} window bank {bank} differs");
        let state = &actual.state[bank * g.state_bank()..][..g.state_bank()];
        let tape = &actual.tape[bank * g.tape_bank()..][..g.tape_bank()];
        let original_tape = &case.tape[bank * g.tape_bank()..][..g.tape_bank()];
        let successor = case.slots.iter().find(|slot| slot.following == bank);
        if let Some(slot) = successor {
            let expected_state = &expected.state[bank * g.state_bank()..][..g.state_bank()];
            let (max, rms) = errors(state, expected_state);
            println!("{label} state bank {bank}: max/rms {max:.3e} rms/rms {rms:.3e}");
            assert!(max <= tolerance.0 && rms <= tolerance.1, "{label} state error {max} {rms}");
            let recorded = g.tape.min(slot.rows - slot.stop) * g.tape_row();
            if recorded > 0 {
                let (max, rms) = errors(&tape[..recorded], &expected.tape[bank * g.tape_bank()..][..recorded]);
                println!("{label} tape bank {bank}: max/rms {max:.3e} rms/rms {rms:.3e}");
                assert!(max <= tolerance.0 && rms <= tolerance.1, "{label} tape error {max} {rms}");
            }
            assert!(
                same_bits(&tape[recorded..], &original_tape[recorded..]),
                "{label} wrote tape rows of bank {bank} past its recorded rows"
            );
        } else {
            let original = &case.state[bank * g.state_bank()..][..g.state_bank()];
            assert!(
                same_bits(state, original) && same_bits(tape, original_tape),
                "{label} wrote bank {bank}, which is not a successor"
            );
        }
    }
}

/// State rows per threadgroup of every mapping a geometry admits.
fn mappings(g: Geometry) -> Vec<u64> {
    [32u64, 16, 64].into_iter().filter(|rows| g.head_width as u64 % rows == 0).collect()
}

const SMALL: Geometry = Geometry {
    heads: 4,
    head_width: 32,
    groups: 2,
    state_width: 32,
    convolution: 4,
    banks: 7,
    tape: 3,
};

fn small_cases() -> Vec<(&'static str, Case)> {
    vec![
        ("one row from the zero seed", Case::new(SMALL, 1, vec![slot(1, 1, 0, 3)], 1)),
        (
            "short slots shorter than the window, padded rows",
            Case::new(SMALL, 8, vec![slot(2, 2, 1, 4), slot(1, 1, 2, 5), slot(3, 3, 0, 3)], 2),
        ),
        (
            "verify slots from tape versions",
            Case::new(
                SMALL,
                16,
                vec![version(4, 1, 1, 4, 2), version(6, 2, 2, 5, 0), version(3, 3, 3, 6, 3)],
                3,
            ),
        ),
        (
            "pieces across slots, interior and zero stops",
            Case::new(
                SMALL,
                128,
                vec![slot(70, 45, 1, 4), version(37, 37, 2, 5, 2), slot(20, 0, 3, 6)],
                4,
            ),
        ),
    ]
}

#[test]
fn step_and_chunk_match_the_portable_body() {
    for (label, case) in small_cases() {
        let oracle = case.oracle();
        check(&format!("{label}: host vs body"), &case, &case.host(), &oracle, (1e-4, 1e-5));
        for device in devices() {
            let backend = device.backend().as_str();
            for rows in mappings(case.geometry) {
                for reuse in [false, true] {
                    let label = format!("{backend} {label} ROWS {rows} reused slab {reuse}");
                    check(&format!("{label}: step"), &case, &case.native(&device, false, rows, reuse), &oracle, (2e-5, 2e-6));
                    check(&format!("{label}: chunk"), &case, &case.native(&device, true, rows, reuse), &oracle, (2e-4, 2e-5));
                }
            }
        }
    }
}

/// Nemotron-H Mamba-2 geometries (plan §4.6, brief §2): heads NH, head width
/// P = 64, groups G = 8, state N = 128, conv 4.
fn nemotron() -> [(&'static str, Geometry); 3] {
    let geometry = |heads| Geometry {
        heads,
        head_width: 64,
        groups: 8,
        state_width: 128,
        convolution: 4,
        banks: 6,
        tape: 15,
    };
    [("Lightning", geometry(64)), ("Super", geometry(128)), ("Ultra", geometry(256))]
}

/// Decode, a 16-row verify from a tape version recording its tape, and a
/// 160-row prefill over two slots (pieces, an interior stop), BF16
/// activations, step and chunk, against the host model.
#[test]
fn nemotron_geometries_agree_with_the_host_model() {
    for (name, geometry) in nemotron() {
        for (label, rows, slots) in [
            ("decode", 1, vec![slot(1, 1, 1, 3)]),
            ("batched decode", 3, vec![slot(1, 1, 1, 3), slot(1, 1, 0, 4), slot(1, 1, 2, 5)]),
            ("verify 16", 16, vec![version(16, 1, 1, 3, 4)]),
            ("prefill 160, two slots", 160, vec![slot(100, 77, 1, 3), slot(60, 60, 2, 4)]),
        ] {
            let case = Case::new(geometry, rows, slots, 21).with_bf16_activations();
            let host = case.host();
            for device in devices() {
                let backend = device.backend().as_str();
                let label = format!("{backend} {name} {label}");
                let step = case.native(&device, false, 32, false);
                check(&format!("{label}: step"), &case, &step, &host, (1e-4, 3e-3));
                let chunk = case.native(&device, true, 32, false);
                check(&format!("{label}: chunk"), &case, &chunk, &host, (1e-3, 3e-3));
            }
        }
    }
}

/// A slot of at most 16 rows gets the step's bits from the chunk entry too,
/// whatever its peers (here a 40-row slot on the chunked path), so a
/// request's verify rows never depend on the class; `ROWS` never changes bits.
#[test]
fn chunk_short_slots_get_the_step_bits_and_rows_never_change_bits() {
    let (_, lightning) = nemotron()[0];
    let geometry = Geometry {
        banks: 11,
        tape: 0,
        ..lightning
    };
    let slots = vec![
        slot(4, 1, 1, 6),
        slot(16, 9, 2, 7),
        slot(40, 40, 3, 8),
        slot(1, 1, 4, 9),
        slot(7, 0, 5, 10),
    ];
    let case = Case::new(geometry, 70, slots, 31).with_bf16_activations();
    let row_elements = geometry.heads * geometry.head_width;
    for device in devices() {
        let backend = device.backend().as_str();
        let step = case.native(&device, false, 32, false);
        for rows in mappings(geometry) {
            let other = case.native(&device, false, rows, false);
            assert!(
                same_bits(&other.mixed, &step.mixed) && same_bits(&other.state, &step.state),
                "{backend} step ROWS {rows} changed result bits"
            );
            let chunk = case.native(&device, true, rows, false);
            let mut first = 0;
            for s in &case.slots {
                let range = first * row_elements..(first + s.rows) * row_elements;
                first += s.rows;
                if s.rows > 16 {
                    continue;
                }
                let bank = s.following * geometry.state_bank()..(s.following + 1) * geometry.state_bank();
                assert!(
                    same_bits(&step.mixed[range.clone()], &chunk.mixed[range])
                        && same_bits(&step.state[bank.clone()], &chunk.state[bank]),
                    "{backend} chunk ROWS {rows}: a {}-row slot differs from the step",
                    s.rows
                );
            }
        }
    }
}

/// A committed tape version (bank, j) equals a run that published after those
/// rows: the next advance from either has the same bits. The tentative
/// advance is a 5-row verify slot (stop 1).
#[test]
fn tape_versions_equal_runs_that_stopped_there() {
    let verify = 5;
    let (_, lightning) = nemotron()[0];
    let geometry = Geometry {
        banks: 4,
        tape: verify - 1,
        ..lightning
    };
    for device in devices() {
        let backend = device.backend().as_str();
        for chunk in [false, true] {
            let run = |case: &Case| case.native(&device, chunk, 32, false);
            for accepted in 0..verify {
                let tentative = Case::new(geometry, verify, vec![slot(verify, 1, 1, 2)], 41).with_bf16_activations();
                let first = run(&tentative);
                let continued = tentative.continued(&first, 3, vec![version(3, 3, 2, 3, accepted)], 42);
                let from_tape = run(&continued);
                let stopped =
                    Case::new(geometry, verify, vec![slot(verify, 1 + accepted, 1, 2)], 41).with_bf16_activations();
                let reference_first = run(&stopped);
                let expected = run(&stopped.continued(&reference_first, 3, vec![slot(3, 3, 2, 3)], 42));
                let bank = |values: &[f32], size: usize| values[3 * size..4 * size].to_vec();
                assert!(
                    same_bits(&from_tape.mixed, &expected.mixed)
                        && same_bits(&bank(&from_tape.state, geometry.state_bank()), &bank(&expected.state, geometry.state_bank()))
                        && same_bits(&bank(&from_tape.window, geometry.window_bank()), &bank(&expected.window, geometry.window_bank())),
                    "{backend} chunk {chunk}: version (bank, {accepted}) differs from a run that stopped after {} rows",
                    1 + accepted
                );
                check(
                    &format!("{backend} chunk {chunk}: from version (bank, {accepted})"),
                    &continued,
                    &from_tape,
                    &continued.host(),
                    (1.5e-2, 3e-3),
                );
            }
        }
    }
}

/// A prefill split into chunks, each continuing from the bank the previous
/// published, agrees with the host model of one run (the dual form's pieces
/// start at each call's slot, so the bits may differ from one call).
#[test]
fn chunk_boundaries_continue_the_state() {
    let (_, lightning) = nemotron()[0];
    let geometry = Geometry {
        banks: 6,
        tape: 0,
        ..lightning
    };
    let whole = Case::new(geometry, 200, vec![slot(200, 200, 1, 2)], 23).with_bf16_activations();
    let expected = whole.host();
    let width = geometry.projection_width();
    for device in devices() {
        let backend = device.backend().as_str();
        let mut mixed = Vec::new();
        let mut state: Option<(Case, Outcome)> = None;
        for (chunk, (first, rows)) in [(0usize, 77usize), (77, 1), (78, 122)].into_iter().enumerate() {
            let (previous, following) = if chunk == 0 { (1, 3) } else { (2 + chunk, 3 + chunk) };
            let slots = vec![slot(rows, rows, previous, following)];
            let mut case = match &state {
                None => Case {
                    rows,
                    slots,
                    ..whole.clone()
                },
                Some((case, outcome)) => case.continued(outcome, rows, slots, 0),
            };
            case.projection = whole.projection[first * width..(first + rows) * width].to_vec();
            let outcome = case.native(&device, true, 32, false);
            mixed.extend_from_slice(&outcome.mixed);
            state = Some((case, outcome));
        }
        let (_, last) = state.unwrap();
        check_mixed(&format!("{backend} chunked prefill"), true, &mixed, &expected.mixed, (1e-3, 3e-3));
        let bank = |values: &[f32], bank: usize| values[bank * geometry.state_bank()..][..geometry.state_bank()].to_vec();
        let (max, rms) = errors(&bank(&last.state, 5), &bank(&expected.state, 2));
        println!("{backend} chunked prefill state: max/rms {max:.3e} rms/rms {rms:.3e}");
        assert!(max <= 1e-3 && rms <= 1e-4, "{backend}: chunked prefill state differs: {max} {rms}");
        let window = |values: &[f32], bank: usize| values[bank * geometry.window_bank()..][..geometry.window_bank()].to_vec();
        assert!(same_bits(&window(&last.window, 5), &window(&expected.window, 2)), "{backend}: chunked window differs");
    }
}

/// One `state_space_gate` call: `rows` rows of G groups of U heads of width P,
/// the projection's z segment first (its other columns unread).
struct GateCase {
    groups: usize,
    heads: usize,
    width: usize,
    rows: usize,
    columns: usize,
    mixed: Vec<f32>,
    projection: Vec<f32>,
    norm: Vec<f32>,
    epsilon: f32,
}

impl GateCase {
    fn new(groups: usize, heads: usize, width: usize, rows: usize, seed: u64) -> Self {
        let mut random = Random(seed);
        let channels = groups * heads * width;
        let columns = 2 * channels + 17;
        Self {
            groups,
            heads,
            width,
            rows,
            columns,
            mixed: (0..rows * channels).map(|_| bf16_round(random.next() * 2.0)).collect(),
            projection: (0..rows * columns).map(|_| bf16_round(random.next() * 3.0)).collect(),
            norm: (0..channels).map(|_| 0.5 + random.next()).collect(),
            epsilon: 1e-5,
        }
    }

    /// The f64 host model.
    fn host(&self) -> Vec<f32> {
        let channels = self.groups * self.heads * self.width;
        let span = self.heads * self.width;
        let mut out = vec![0.0f32; self.rows * channels];
        for row in 0..self.rows {
            for group in 0..self.groups {
                let gated = (0..span)
                    .map(|column| {
                        let channel = group * span + column;
                        let gate = self.projection[row * self.columns + channel] as f64;
                        self.mixed[row * channels + channel] as f64 * gate / (1.0 + (-gate).exp())
                    })
                    .collect::<Vec<_>>();
                let inverse =
                    1.0 / (gated.iter().map(|v| v * v).sum::<f64>() / span as f64 + self.epsilon as f64).sqrt();
                for (column, value) in gated.iter().enumerate() {
                    let channel = group * span + column;
                    out[row * channels + channel] = (value * inverse * self.norm[channel] as f64) as f32;
                }
            }
        }
        out
    }

    /// The portable body executed by the Seismic interpreter (BF16
    /// activations).
    fn oracle(&self) -> Vec<f32> {
        use seismic_lang::{
            checked::{check_source, SourceFile},
            entry::ElementBindings,
            interp::{Arg, Interpreter, OutcomeValue, TensorData},
            reference_math::ReferenceScalar,
            registry,
            types::DType,
        };
        let mut sources = seismic_std::sources();
        sources.push(SourceFile {
            path: "state_space.seismic".into(),
            text: include_str!("../kernels/state_space.seismic").into(),
        });
        let module = check_source(sources).unwrap();
        let elements = ElementBindings::new().bind("A", registry::dense(DType::BF16));
        let logical = module
            .entry(module.entry_named("state_space_gate").unwrap(), &elements)
            .unwrap();
        let dense = |dtype, shape: Vec<usize>, values: &[f32]| {
            TensorData::dense(dtype, shape, values.iter().map(|value| f64::from(*value)).collect())
        };
        let mut interpreter = Interpreter::new(&logical);
        let tensors = vec![
            dense(DType::BF16, vec![self.rows, self.groups * self.heads, self.width], &self.mixed),
            dense(DType::BF16, vec![self.rows, self.columns], &self.projection),
            dense(DType::F32, vec![self.groups, self.heads, self.width], &self.norm),
        ];
        let mut arguments = tensors
            .into_iter()
            .map(|tensor| Arg::Tensor(interpreter.add_tensor(tensor)))
            .collect::<Vec<_>>();
        arguments.push(Arg::Scalar(ReferenceScalar::F32(self.epsilon.to_bits())));
        let outcome = interpreter.run_bounded(&arguments, u64::MAX).unwrap();
        let values = match outcome.results().next().unwrap().value() {
            OutcomeValue::Tensor(reader) => {
                (0..reader.element_count()).map(|index| reader.read(index).unwrap() as f32).collect()
            }
            _ => panic!("normalized rows are a tensor"),
        };
        values
    }

    fn native(&self, device: &Device) -> Vec<f32> {
        let heads = (self.groups * self.heads) as u64;
        let mixed = tensor(device, Element::bf16(), &[self.rows as u64, heads, self.width as u64], &self.mixed);
        let projection = tensor(device, Element::bf16(), &[self.rows as u64, self.columns as u64], &self.projection);
        let norm = tensor(device, Element::f32(), &[self.groups as u64, self.heads as u64, self.width as u64], &self.norm);
        let out = state_space_gate::native_for_device_with(
            device,
            state_space_gate::Elements { A: Element::bf16() },
            &NativeSpecialization::new(),
        )
        .unwrap()
        .call(state_space_gate::Args {
            mixed: &mixed,
            projection: &projection,
            state_norm: &norm,
            epsilon: self.epsilon,
        })
        .unwrap()
        .value;
        read(&out)
    }
}

/// The gate against the body on a small case, and against the host model at
/// the Nemotron-H groups (8 groups of 8 / 16 / 32 heads of 64 channels).
#[test]
fn gate_matches_the_portable_body_and_host_model() {
    let small = GateCase::new(2, 3, 16, 5, 61);
    let oracle = small.oracle();
    check_mixed("gate: host vs body", true, &oracle, &small.host(), (1e-6, 3e-3));
    for device in devices() {
        let backend = device.backend().as_str();
        check_mixed(&format!("{backend} gate small"), true, &small.native(&device), &small.host(), (1e-6, 3e-3));
        for (name, heads) in [("Lightning", 8), ("Super", 16), ("Ultra", 32)] {
            for rows in [1, 16, 70] {
                let case = GateCase::new(8, heads, 64, rows, 62);
                check_mixed(&format!("{backend} gate {name} {rows} rows"), true, &case.native(&device), &case.host(), (1e-6, 3e-3));
            }
        }
    }
}

/// The median device time (us) of each launch of the calls `run` makes, each
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

/// Device time per call at the Nemotron-H Lightning and Super geometries
/// (BF16 activations): the step at batched decode (B slots of one row), a
/// 16-row verify slot and 4 slots of 4 rows, and the chunk at 512 and 2048
/// prefill rows; the decode lines report the SSM state traffic (each slot
/// reads and writes its whole state). Calls rotate over argument sets of at
/// least 512 MiB of banks, so state comes from DRAM as it does across a
/// model's layers.
#[test]
#[ignore = "timing; run explicitly on the measurement host"]
fn state_space_timings() {
    let options = seismic::MeasureOptions {
        samples: 11,
        min_sample_seconds: 0.01,
    };
    for device in devices() {
        let backend = device.backend().as_str();
        for (name, geometry) in &nemotron()[..2] {
            let bank_bytes = geometry.state_bank() * 4;
            let mut cases: Vec<(String, bool, usize, Vec<SlotCase>)> = [1usize, 8, 32]
                .into_iter()
                .map(|b| (format!("step decode B={b}"), false, b, (0..b).map(|s| slot(1, 1, 1 + s, 1 + b + s)).collect()))
                .collect();
            cases.push(("step verify 16".into(), false, 16, vec![slot(16, 16, 1, 2)]));
            cases.push(("step 4 x 4".into(), false, 16, (0..4).map(|s| slot(4, 4, 1 + s, 5 + s)).collect()));
            for rows in [512usize, 2048] {
                cases.push((format!("chunk prefill {rows}"), true, rows, vec![slot(rows, rows, 1, 2)]));
            }
            for (label, chunk, rows, slots) in cases {
                let banks = slots.iter().map(|s| s.following).max().unwrap() + 1;
                let g = Geometry { banks, tape: 0, ..*geometry };
                let case = Case::new(g, rows, slots.clone(), 5).with_bf16_activations();
                let sets = (512usize << 20).div_ceil(banks * bank_bytes).max(2);
                let mut rotation = (0..sets).map(|_| case.bind(&device, false)).collect::<Vec<_>>();
                for mapping in mappings(g) {
                    let specialization = case.specialization(&device, mapping);
                    let (seconds, launches) = if chunk {
                        let kernel = state_space_chunk::native_for_device_with(
                            &device,
                            state_space_chunk::Elements { A: Element::bf16() },
                            &specialization,
                        )
                        .unwrap();
                        let args = rotation.iter_mut().map(|bound| bound.chunk_args(case.slab_banks)).collect();
                        let seconds = kernel.measure(args, &options).unwrap().median;
                        let launches = launch_medians(&device, || {
                            for bound in rotation.iter_mut() {
                                kernel.call(bound.chunk_args(case.slab_banks)).unwrap();
                            }
                        });
                        (seconds, format!(" (inputs + products + scan {launches} us)"))
                    } else {
                        let kernel = state_space_step::native_for_device_with(
                            &device,
                            state_space_step::Elements { A: Element::bf16() },
                            &specialization,
                        )
                        .unwrap();
                        let args = rotation.iter_mut().map(|bound| bound.step_args(case.slab_banks)).collect();
                        (kernel.measure(args, &options).unwrap().median, String::new())
                    };
                    let traffic = 2.0 * (slots.len() * bank_bytes) as f64;
                    println!(
                        "{backend} {name} {label} ROWS {mapping}: {:.1} us, {:.2} us/row, state traffic {:.0} GB/s{launches}",
                        seconds * 1e6,
                        seconds * 1e6 / rows as f64,
                        traffic / seconds / 1e9
                    );
                }
            }
        }
    }
}
