// `short_conv_rows` against its portable body, on every backend this host
// opens.
//
// The entry's arithmetic is fixed by the body (a tap-ascending F32 FMA chain
// times the gate, rounded to A), so every native must reproduce it bit for
// bit. Small cases run the body in the Seismic interpreter; an F32 host
// replica of the same recipe, pinned to the interpreter bit for bit, checks
// the LFM2 geometry (2048 channels, 3 taps: LFM2.5 2.6B and 8B-A1B share it).
// Every case checks the outputs, the published windows (copies), and that
// no bank other than each slot's successor changes.

mod versions_common;

use magnitude_kernels::short_conv_rows;
use seismic::{Device, Element, NativeSpecialization, Tensor};
use versions_common::*;

#[derive(Clone, Copy)]
struct Geometry {
    channels: usize,
    /// Convolution taps (C).
    taps: usize,
    banks: usize,
    /// Tape rows per bank (T).
    tape: usize,
}

impl Geometry {
    fn window_rows(self) -> usize {
        self.taps - 1 + self.tape
    }
    fn window_bank(self) -> usize {
        self.window_rows() * self.channels
    }
}

#[derive(Clone)]
struct Case {
    geometry: Geometry,
    rows: usize,
    slots: Vec<SlotCase>,
    /// [rows, 2 CH]: u | c.
    projection: Vec<f32>,
    /// [CH, C].
    convolution: Vec<f32>,
    /// [banks, C - 1 + T, CH].
    window: Vec<f32>,
    slab_banks: u64,
}

struct Outcome {
    gated: Vec<f32>,
    window: Vec<f32>,
}

impl Case {
    fn new(geometry: Geometry, rows: usize, slots: Vec<SlotCase>, seed: u64) -> Self {
        let mut random = Random(seed);
        let projection = (0..rows * 2 * geometry.channels).map(|_| random.next() * 2.0).collect();
        let convolution = (0..geometry.channels * geometry.taps).map(|_| random.next() * 0.7).collect();
        let mut window = vec![SENTINEL; geometry.banks * geometry.window_bank()];
        window[..geometry.window_bank()].fill(0.0);
        for slot in &slots {
            if slot.previous != 0 {
                for value in &mut window[slot.previous * geometry.window_bank()..][..geometry.window_bank()] {
                    *value = random.next() * 2.0;
                }
            }
        }
        Self {
            geometry,
            rows,
            slots,
            projection,
            convolution,
            window,
            slab_banks: 2,
        }
    }

    /// The next advance of the same layer after `outcome`: new projection
    /// rows for `slots` and the window arena `outcome` left.
    fn continued(&self, outcome: &Outcome, rows: usize, slots: Vec<SlotCase>, seed: u64) -> Self {
        let fresh = Case::new(self.geometry, rows, slots, seed);
        Self {
            convolution: self.convolution.clone(),
            window: outcome.window.clone(),
            ..fresh
        }
    }

    /// The body's recipe on the host in F32, rounded to `activation`.
    fn host(&self, activation: Element) -> Outcome {
        let g = self.geometry;
        let ch = g.channels;
        let last = g.taps - 1;
        let round = |value: f32| if activation == Element::bf16() { bf16_round(value) } else { value };
        let mut gated = vec![0.0f32; self.rows * ch];
        let mut window = self.window.clone();
        for (slot, first) in self.slots.iter().zip(firsts(&self.slots)) {
            let input = |position: isize, channel: usize| -> f32 {
                if position < 0 {
                    self.window[slot.previous * g.window_bank()
                        + (slot.taped as isize + last as isize + position) as usize * ch
                        + channel]
                } else {
                    self.projection[(first + position as usize) * 2 * ch + channel]
                }
            };
            for local in 0..slot.rows {
                let row = first + local;
                for channel in 0..ch {
                    let mut sum = 0.0f32;
                    for tap in 0..g.taps {
                        let value = input(local as isize + tap as isize - last as isize, channel);
                        sum = self.convolution[channel * g.taps + tap].mul_add(value, sum);
                    }
                    gated[row * ch + channel] = round(self.projection[row * 2 * ch + ch + channel] * sum);
                }
            }
            let published = last + g.tape.min(slot.rows - slot.stop);
            for tap in 0..published {
                for channel in 0..ch {
                    window[slot.following * g.window_bank() + tap * ch + channel] =
                        input(slot.stop as isize + tap as isize - last as isize, channel);
                }
            }
        }
        Outcome { gated, window }
    }

    /// The portable body executed by the Seismic interpreter.
    fn oracle(&self, activation: Element) -> Outcome {
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
            path: "short_conv.seismic".into(),
            text: include_str!("../kernels/short_conv.seismic").into(),
        });
        let module = check_source(sources).unwrap();
        let dtype = if activation == Element::bf16() { DType::BF16 } else { DType::F32 };
        let elements = ElementBindings::new().bind("A", registry::dense(dtype));
        let logical = module
            .entry(module.entry_named("short_conv_rows").unwrap(), &elements)
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
            floats(vec![self.rows, 2 * g.channels], &self.projection),
            floats(vec![g.channels, g.taps], &self.convolution),
            ints(vec![count + 1, 2], segments(&self.slots, self.rows)),
            ints(vec![count], each(|slot| slot.stop)),
            ints(vec![count], each(|slot| slot.previous)),
            ints(vec![count], each(|slot| slot.taped)),
            ints(vec![count], each(|slot| slot.following)),
            floats(vec![g.banks, g.window_rows(), g.channels], &self.window),
        ];
        let mut arguments = tensors
            .into_iter()
            .map(|tensor| Arg::Tensor(interpreter.add_tensor(tensor)))
            .collect::<Vec<_>>();
        arguments.push(Arg::Scalar(ReferenceScalar::U32(self.slab_banks as u32)));
        let outcome = interpreter.run_bounded(&arguments, u64::MAX).unwrap();
        if let SourceTermination::Failed(failure) = outcome.termination() {
            panic!("portable short convolution body failed: {failure}");
        }
        let read = |reader: seismic_lang::interp::TensorReader<'_>| {
            (0..reader.element_count())
                .map(|index| reader.read(index).unwrap() as f32)
                .collect::<Vec<_>>()
        };
        let gated = match outcome.results().next().unwrap().value() {
            OutcomeValue::Tensor(reader) => read(reader),
            _ => panic!("the gated output is a tensor"),
        };
        let window = outcome
            .inputs()
            .find(|input| input.ordinal() == 7)
            .map(|input| read(input.tensor()))
            .expect("window is a mutable input");
        Outcome { gated, window }
    }

    /// The native entry on `device` in `activation`; `reuse` places the banks
    /// with an interior slab freed and reused.
    fn native(&self, device: &Device, activation: Element, reuse: bool) -> Outcome {
        let mut bound = self.bind(device, reuse);
        let gated = kernel(device, activation).call(bound.args(self.slab_banks)).unwrap().value;
        Outcome {
            gated: read(&gated),
            window: read(&bound.window),
        }
    }

    /// The case's device tensors; `reuse` places the banks with an interior
    /// slab freed and reused.
    fn bind(&self, device: &Device, reuse: bool) -> Bound {
        let g = self.geometry;
        let (slabs, mut arenas) = slab_arenas(
            device,
            g.banks,
            self.slab_banks,
            &[Arena {
                element: Element::f32(),
                bank_shape: vec![g.window_rows() as u64, g.channels as u64],
                values: &self.window,
            }],
            reuse,
        );
        Bound {
            _slabs: slabs,
            projection: tensor(device, Element::f32(), &[self.rows as u64, 2 * g.channels as u64], &self.projection),
            convolution: tensor(device, Element::f32(), &[g.channels as u64, g.taps as u64], &self.convolution),
            slots: SlotTensors::new(device, &self.slots, self.rows),
            window: arenas.remove(0),
        }
    }
}

fn kernel(device: &Device, activation: Element) -> seismic::NativeKernel<short_conv_rows::Entry> {
    short_conv_rows::native_for_device_with(
        device,
        short_conv_rows::Elements { A: activation },
        &NativeSpecialization::new(),
    )
    .unwrap()
}

/// Device tensors of one case; each call mutates its window arena.
struct Bound {
    _slabs: seismic::SlabTensor,
    projection: Tensor,
    convolution: Tensor,
    slots: SlotTensors,
    window: Tensor,
}

impl Bound {
    fn args(&mut self, slab_banks: u64) -> short_conv_rows::Args<'_> {
        short_conv_rows::Args {
            projection: &self.projection,
            convolution: &self.convolution,
            segments: &self.slots.segments,
            stop: &self.slots.stop,
            previous_bank: &self.slots.previous,
            previous_tape: &self.slots.previous_tape,
            following_bank: &self.slots.following,
            window: &mut self.window,
            slab_banks: slab_banks as u32,
        }
    }
}

/// Outputs and windows equal bit for bit.
fn check_same(label: &str, actual: &Outcome, expected: &Outcome) {
    assert!(
        same_bits(&actual.gated, &expected.gated),
        "{label}: gated output differs at {:?}",
        first_difference(&actual.gated, &expected.gated)
    );
    assert!(
        same_bits(&actual.window, &expected.window),
        "{label}: window arena differs at {:?}",
        first_difference(&actual.window, &expected.window)
    );
}

const SMALL: Geometry = Geometry {
    channels: 72,
    taps: 3,
    banks: 7,
    tape: 3,
};

fn small_cases() -> Vec<(&'static str, Case)> {
    vec![
        ("decode from the zero seed", Case::new(SMALL, 1, vec![slot(1, 1, 0, 3)], 1)),
        (
            "batched decode with padded rows",
            Case::new(SMALL, 5, vec![slot(1, 1, 1, 4), slot(1, 1, 2, 5), slot(1, 1, 0, 6)], 2),
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
            "prefill slots with an interior stop and a zero stop",
            Case::new(SMALL, 90, vec![slot(60, 41, 1, 4), slot(20, 0, 2, 5), slot(1, 1, 3, 6)], 4),
        ),
    ]
}

#[test]
fn natives_match_the_portable_body() {
    for (label, case) in small_cases() {
        for activation in [Element::f32(), Element::bf16()] {
            let oracle = case.oracle(activation);
            let host = case.host(activation);
            check_same(&format!("{label} {}: host replica vs body", activation.name()), &host, &oracle);
            for device in devices() {
                let backend = device.backend().as_str();
                for reuse in [false, true] {
                    let native = case.native(&device, activation, reuse);
                    check_same(
                        &format!("{backend} {label} {} (reused slab {reuse})", activation.name()),
                        &native,
                        &oracle,
                    );
                }
            }
        }
    }
}

/// The LFM2 conv geometry (hidden 2048, `shortconv.l_cache` 3; the 2.6B and
/// the 8B-A1B share it): decode, batched decode, a 16-row verify with its
/// tape, and prefill chunks, checked against the host replica.
#[test]
fn lfm2_geometry_matches_the_host_replica() {
    let geometry = Geometry {
        channels: 2048,
        taps: 3,
        banks: 9,
        tape: 15,
    };
    let cases = [
        ("decode", 1, vec![slot(1, 1, 1, 5)]),
        ("batched decode", 4, (0..4).map(|b| slot(1, 1, b, 5 + b)).collect::<Vec<_>>()),
        ("verify 16 rows", 16, vec![version(16, 1, 1, 5, 7)]),
        ("prefill 512, two slots", 512, vec![slot(300, 211, 1, 5), slot(212, 212, 2, 6)]),
    ];
    for device in devices() {
        let backend = device.backend().as_str();
        for (label, rows, slots) in cases.clone() {
            let case = Case::new(geometry, rows, slots, 17);
            for activation in [Element::bf16(), Element::f32()] {
                check_same(
                    &format!("{backend} LFM2 {label} {}", activation.name()),
                    &case.native(&device, activation, false),
                    &case.host(activation),
                );
            }
        }
    }
}

/// A prefill split into chunks, each continuing from the bank the previous
/// published, gives the bits of one run over all its rows.
#[test]
fn chunk_boundaries_equal_one_run() {
    let geometry = Geometry {
        channels: 2048,
        taps: 3,
        banks: 6,
        tape: 0,
    };
    for device in devices() {
        let backend = device.backend().as_str();
        let whole = Case::new(geometry, 384, vec![slot(384, 384, 1, 2)], 23);
        let expected = whole.native(&device, Element::bf16(), false);
        let mut gated = Vec::new();
        let mut state: Option<(Case, Outcome)> = None;
        for (chunk, (first, rows)) in [(0usize, 128usize), (128, 1), (129, 255)].into_iter().enumerate() {
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
            case.projection = whole.projection[first * 4096..(first + rows) * 4096].to_vec();
            let outcome = case.native(&device, Element::bf16(), false);
            gated.extend_from_slice(&outcome.gated);
            state = Some((case, outcome));
        }
        let (_, last) = state.unwrap();
        assert!(same_bits(&gated, &expected.gated), "{backend}: chunked outputs differ from one run");
        let bank = |values: &[f32], bank: usize| values[bank * geometry.window_bank()..][..geometry.window_bank()].to_vec();
        assert!(
            same_bits(&bank(&last.window, 5), &bank(&expected.window, 2)),
            "{backend}: the chunked run's window differs from one run's"
        );
    }
}

/// A committed tape version (bank, j) equals a run that published after those
/// rows: the next advance from either has the same bits. The tentative
/// advance is a 5-row verify slot (stop 1).
#[test]
fn tape_versions_equal_runs_that_stopped_there() {
    let verify = 5;
    let geometry = Geometry {
        channels: 2048,
        taps: 3,
        banks: 4,
        tape: verify - 1,
    };
    for device in devices() {
        let backend = device.backend().as_str();
        let run = |case: &Case| case.native(&device, Element::bf16(), false);
        for accepted in 0..verify {
            let tentative = Case::new(geometry, verify, vec![slot(verify, 1, 1, 2)], 41);
            let first = run(&tentative);
            let next = version(3, 3, 2, 3, accepted);
            let from_tape = run(&tentative.continued(&first, 3, vec![next], 42));
            let stopped = Case::new(geometry, verify, vec![slot(verify, 1 + accepted, 1, 2)], 41);
            let reference_first = run(&stopped);
            let expected = run(&stopped.continued(&reference_first, 3, vec![slot(3, 3, 2, 3)], 42));
            let bank = |values: &[f32]| values[3 * geometry.window_bank()..4 * geometry.window_bank()].to_vec();
            assert!(
                same_bits(&from_tape.gated, &expected.gated)
                    && same_bits(&bank(&from_tape.window), &bank(&expected.window)),
                "{backend}: version (bank, {accepted}) differs from a run that stopped after {} rows",
                1 + accepted
            );
        }
    }
}

/// Device time per call at the LFM2 geometry (BF16 output): decode at 1 and
/// 8 slots, a 16-row verify, and 512 and 2048 prefill rows; also the bytes
/// the call moves (u | c rows, the window reads and the published window).
#[test]
#[ignore = "timing; run explicitly on the measurement host"]
fn short_conv_timings() {
    let options = seismic::MeasureOptions {
        samples: 11,
        min_sample_seconds: 0.01,
    };
    let geometry = Geometry {
        channels: 2048,
        taps: 3,
        banks: 0,
        tape: 0,
    };
    for device in devices() {
        let backend = device.backend().as_str();
        let decode = |b: usize| (0..b).map(|s| slot(1, 1, 1 + s, 1 + b + s)).collect::<Vec<_>>();
        for (label, rows, slots) in [
            ("decode B=1", 1usize, decode(1)),
            ("decode B=8", 8, decode(8)),
            ("verify 16", 16, vec![slot(16, 1, 1, 2)]),
            ("prefill 512", 512, vec![slot(512, 512, 1, 2)]),
            ("prefill 2048", 2048, vec![slot(2048, 2048, 1, 2)]),
        ] {
            let banks = slots.iter().map(|s| s.following).max().unwrap() + 1;
            let case = Case::new(Geometry { banks, ..geometry }, rows, slots.clone(), 3);
            let mut rotation = (0..8).map(|_| case.bind(&device, false)).collect::<Vec<_>>();
            let args = rotation.iter_mut().map(|bound| bound.args(case.slab_banks)).collect();
            let seconds = kernel(&device, Element::bf16()).measure(args, &options).unwrap().median;
            let bytes = rows * 2048 * (8 + 2) + slots.len() * 2 * 2 * 2048 * 4;
            println!(
                "{backend} short conv {label}: {:.1} us ({:.0} GB/s effective)",
                seconds * 1e6,
                bytes as f64 / seconds / 1e9
            );
        }
    }
}
