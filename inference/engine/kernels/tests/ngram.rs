// `ngram_rows`, `ngram_gate` and `ngram_inject` against their portable
// bodies, on every backend this host opens.
//
// `ngram_rows` is exact: its rows and token windows equal the body's and an
// independent statement of the row rule, across chunk boundaries and tape
// versions. The gate reduces in each backend's order and both other entries
// evaluate `exp` natively, so outputs agree with the body to a tolerance;
// the published windows hold u values every backend forms in the body's
// order, and agree bit for bit. A chunked prefill and a committed tape
// version reproduce one run's bits on each backend.

mod versions_common;

use magnitude_kernels::{ngram_gate, ngram_inject, ngram_rows};
use seismic::{Device, Element, NativeSpecialization, Tensor};
use versions_common::*;

const EPSILON: f32 = 1e-6;
/// Agreement with the body, relative to the reference RMS.
const TOLERANCE: f64 = 2e-5;

#[derive(Clone, Copy)]
struct Geometry {
    streams: usize,
    hidden: usize,
    taps: usize,
    dilation: usize,
    banks: usize,
    tape: usize,
}

impl Geometry {
    fn channels(self) -> usize {
        self.streams * self.hidden
    }
    fn reach(self) -> usize {
        self.dilation * (self.taps - 1)
    }
    fn window_rows(self) -> usize {
        self.reach() + self.tape
    }
    fn window_bank(self) -> usize {
        self.window_rows() * self.channels()
    }
    fn statics(self) -> [(&'static str, u64); 3] {
        [
            ("H", self.hidden as u64),
            ("S", self.streams as u64),
            ("K", self.taps as u64),
        ]
    }
}

/// The declared defaults of `E` at `statics` (none on the CPU).
fn defaults<E: seismic::Entry>(device: &Device, statics: &[(&str, u64)]) -> NativeSpecialization {
    let statics = if is_cpu(device) {
        NativeSpecialization::new()
    } else {
        statics
            .iter()
            .fold(NativeSpecialization::new(), |spec, (name, value)| {
                spec.with_static(*name, *value)
            })
    };
    seismic::generated::native_implementation::<E>(device)
        .unwrap()
        .unwrap()
        .default_specialization(&statics)
        .unwrap()
}

fn values(random: &mut Random, count: usize, scale: f32) -> Vec<f32> {
    (0..count).map(|_| random.next() * scale).collect()
}

fn interpret(
    source: &str,
    entry: &str,
    tensors: Vec<seismic_lang::interp::TensorData>,
    scalars: Vec<seismic_lang::reference_math::ReferenceScalar>,
    mutable: Option<usize>,
) -> (Vec<f32>, Option<Vec<f32>>) {
    use seismic_lang::{
        checked::{SourceFile, check_source},
        entry::ElementBindings,
        failure::SourceTermination,
        interp::{Arg, Interpreter, OutcomeValue},
        registry,
        types::DType,
    };
    let mut sources = seismic_std::sources();
    sources.push(SourceFile {
        path: "ngram.seismic".into(),
        text: source.into(),
    });
    let module = check_source(sources).unwrap();
    let elements = ElementBindings::new().bind("NW", registry::dense(DType::F32));
    let logical = module
        .entry(module.entry_named(entry).unwrap(), &elements)
        .unwrap();
    let mut interpreter = Interpreter::new(&logical);
    let mut arguments = tensors
        .into_iter()
        .map(|tensor| Arg::Tensor(interpreter.add_tensor(tensor)))
        .collect::<Vec<_>>();
    arguments.extend(scalars.into_iter().map(Arg::Scalar));
    let outcome = interpreter.run_bounded(&arguments, u64::MAX).unwrap();
    if let SourceTermination::Failed(failure) = outcome.termination() {
        panic!("portable {entry} body failed: {failure}");
    }
    let read = |reader: seismic_lang::interp::TensorReader<'_>| {
        (0..reader.element_count())
            .map(|index| reader.read(index).unwrap() as f32)
            .collect::<Vec<_>>()
    };
    let result = match outcome.results().next().unwrap().value() {
        OutcomeValue::Tensor(reader) => read(reader),
        _ => panic!("the result is a tensor"),
    };
    let window = mutable.map(|ordinal| {
        outcome
            .inputs()
            .find(|input| input.ordinal() == ordinal)
            .map(|input| read(input.tensor()))
            .expect("the window is a mutable input")
    });
    (result, window)
}

fn floats(shape: Vec<usize>, values: &[f32]) -> seismic_lang::interp::TensorData {
    seismic_lang::interp::TensorData::dense(
        seismic_lang::types::DType::F32,
        shape,
        values.iter().map(|value| f64::from(*value)).collect(),
    )
}

fn integers(shape: Vec<usize>, values: Vec<i32>) -> seismic_lang::interp::TensorData {
    seismic_lang::interp::TensorData::dense(
        seismic_lang::types::DType::I32,
        shape,
        values.into_iter().map(f64::from).collect(),
    )
}

fn check_close(label: &str, actual: &[f32], expected: &[f32]) {
    let error = relative_max(actual, expected);
    assert!(
        error <= TOLERANCE,
        "{label}: relative error {error:e} exceeds {TOLERANCE:e}"
    );
}

// ---- ngram_gate -------------------------------------------------------------------------------

struct GateCase {
    rows: usize,
    streams: usize,
    hidden: usize,
    x: Vec<f32>,
    key: Vec<f32>,
    value: Vec<f32>,
    key_norm: Vec<f32>,
    query_norm: Vec<f32>,
}

impl GateCase {
    fn new(rows: usize, streams: usize, hidden: usize, seed: u64) -> Self {
        let mut random = Random(seed);
        let wide = rows * streams * hidden;
        Self {
            rows,
            streams,
            hidden,
            x: values(&mut random, wide, 3.0),
            key: values(&mut random, wide, 2.0),
            value: values(&mut random, rows * hidden, 1.5),
            key_norm: values(&mut random, streams * hidden, 1.0),
            query_norm: values(&mut random, streams * hidden, 1.0),
        }
    }

    fn oracle(&self) -> Vec<f32> {
        let (m, s, h) = (self.rows, self.streams, self.hidden);
        interpret(
            include_str!("../kernels/ngram.seismic"),
            "ngram_gate",
            vec![
                floats(vec![m, s, h], &self.x),
                floats(vec![m, s, h], &self.key),
                floats(vec![m, h], &self.value),
                floats(vec![s, h], &self.key_norm),
                floats(vec![s, h], &self.query_norm),
            ],
            vec![seismic_lang::reference_math::ReferenceScalar::F32(
                EPSILON.to_bits(),
            )],
            None,
        )
        .0
    }

    fn native(&self, device: &Device) -> Vec<f32> {
        let (m, s, h) = (self.rows as u64, self.streams as u64, self.hidden as u64);
        let kernel = ngram_gate::native_for_device_with(
            device,
            ngram_gate::Elements { NW: Element::f32() },
            &defaults::<ngram_gate::Entry>(device, &[("H", h), ("S", s)]),
        )
        .unwrap();
        let result = kernel
            .call(ngram_gate::Args {
                hidden: &tensor(device, Element::f32(), &[m, s, h], &self.x),
                key: &tensor(device, Element::f32(), &[m, s, h], &self.key),
                value: &tensor(device, Element::f32(), &[m, h], &self.value),
                key_norm: &tensor(device, Element::f32(), &[s, h], &self.key_norm),
                query_norm: &tensor(device, Element::f32(), &[s, h], &self.query_norm),
                epsilon: EPSILON,
            })
            .unwrap()
            .value;
        read(&result)
    }
}

#[test]
fn gate_natives_match_the_portable_body() {
    for (rows, streams, hidden, seed) in [(1, 4, 64, 1), (5, 4, 160, 2), (17, 2, 300, 3)] {
        let case = GateCase::new(rows, streams, hidden, seed);
        let oracle = case.oracle();
        for device in devices() {
            check_close(
                &format!(
                    "{} gate {rows}x{streams}x{hidden}",
                    device.backend().as_str()
                ),
                &case.native(&device),
                &oracle,
            );
        }
    }
}

// ---- ngram_inject -----------------------------------------------------------------------------

#[derive(Clone)]
struct Case {
    geometry: Geometry,
    rows: usize,
    slots: Vec<SlotCase>,
    /// [rows, S * H].
    hidden: Vec<f32>,
    /// [rows, 2 S]: gates | inverses.
    scales: Vec<f32>,
    /// [rows, H].
    value: Vec<f32>,
    /// [S, H].
    norm: Vec<f32>,
    /// [S * H, K].
    convolution: Vec<f32>,
    /// [banks, D (K - 1) + T, S * H].
    window: Vec<f32>,
    slab_banks: u64,
}

struct Outcome {
    result: Vec<f32>,
    window: Vec<f32>,
}

impl Case {
    fn new(geometry: Geometry, rows: usize, slots: Vec<SlotCase>, seed: u64) -> Self {
        let mut random = Random(seed);
        let (s, h) = (geometry.streams, geometry.hidden);
        let mut window = vec![SENTINEL; geometry.banks * geometry.window_bank()];
        window[..geometry.window_bank()].fill(0.0);
        let mut seeded = Random(seed ^ 0x5eed);
        for slot in &slots {
            if slot.previous != 0 {
                for value in
                    &mut window[slot.previous * geometry.window_bank()..][..geometry.window_bank()]
                {
                    *value = seeded.next();
                }
            }
        }
        let scales = (0..rows)
            .flat_map(|_| {
                let gates = (0..s)
                    .map(|_| 0.5 + 0.5 * random.next())
                    .collect::<Vec<_>>();
                let inverses = (0..s)
                    .map(|_| 1.0 + random.next() * 0.5)
                    .collect::<Vec<_>>();
                gates.into_iter().chain(inverses)
            })
            .collect();
        Self {
            geometry,
            rows,
            slots,
            hidden: values(&mut random, rows * s * h, 2.0),
            scales,
            value: values(&mut random, rows * h, 1.5),
            norm: values(&mut random, s * h, 1.0),
            convolution: values(&mut random, s * h * geometry.taps, 0.7),
            window,
            slab_banks: 2,
        }
    }

    /// The next advance of the same layer after `outcome`.
    fn continued(&self, outcome: &Outcome, rows: usize, slots: Vec<SlotCase>, seed: u64) -> Self {
        let fresh = Case::new(self.geometry, rows, slots, seed);
        Self {
            norm: self.norm.clone(),
            convolution: self.convolution.clone(),
            window: outcome.window.clone(),
            ..fresh
        }
    }

    fn oracle(&self) -> Outcome {
        let g = self.geometry;
        let (s, h, ch) = (g.streams, g.hidden, g.channels());
        let each = |f: fn(&SlotCase) -> usize| {
            self.slots
                .iter()
                .map(|slot| f(slot) as i32)
                .collect::<Vec<_>>()
        };
        let count = self.slots.len();
        let (result, window) = interpret(
            include_str!("../kernels/ngram.seismic"),
            "ngram_inject",
            vec![
                floats(vec![self.rows, ch], &self.hidden),
                floats(vec![self.rows, 2 * s], &self.scales),
                floats(vec![self.rows, h], &self.value),
                floats(vec![s, h], &self.norm),
                floats(vec![ch, g.taps], &self.convolution),
                integers(vec![count + 1, 2], segments(&self.slots, self.rows)),
                integers(vec![count], each(|slot| slot.stop)),
                integers(vec![count], each(|slot| slot.previous)),
                integers(vec![count], each(|slot| slot.taped)),
                integers(vec![count], each(|slot| slot.following)),
                floats(vec![g.banks, g.window_rows(), ch], &self.window),
            ],
            vec![
                seismic_lang::reference_math::ReferenceScalar::I32(g.dilation as i32),
                seismic_lang::reference_math::ReferenceScalar::U32(self.slab_banks as u32),
            ],
            Some(10),
        );
        Outcome {
            result,
            window: window.unwrap(),
        }
    }

    fn native(&self, device: &Device, reuse: bool) -> Outcome {
        let mut bound = self.bind(device, reuse);
        let kernel = ngram_inject::native_for_device_with(
            device,
            ngram_inject::Elements { NW: Element::f32() },
            &defaults::<ngram_inject::Entry>(device, &self.geometry.statics()),
        )
        .unwrap();
        let result = kernel.call(bound.args(self.slab_banks)).unwrap().value;
        Outcome {
            result: read(&result),
            window: read(&bound.window),
        }
    }

    fn bind(&self, device: &Device, reuse: bool) -> Bound {
        let g = self.geometry;
        let (s, h, ch) = (g.streams as u64, g.hidden as u64, g.channels() as u64);
        let rows = self.rows as u64;
        let (slabs, mut arenas) = slab_arenas(
            device,
            g.banks,
            self.slab_banks,
            &[Arena {
                element: Element::f32(),
                bank_shape: vec![g.window_rows() as u64, ch],
                values: &self.window,
            }],
            reuse,
        );
        Bound {
            _slabs: slabs,
            hidden: tensor(device, Element::f32(), &[rows, ch], &self.hidden),
            scales: tensor(device, Element::f32(), &[rows, 2 * s], &self.scales),
            value: tensor(device, Element::f32(), &[rows, h], &self.value),
            norm: tensor(device, Element::f32(), &[s, h], &self.norm),
            convolution: tensor(
                device,
                Element::f32(),
                &[ch, g.taps as u64],
                &self.convolution,
            ),
            slots: SlotTensors::new(device, &self.slots, self.rows),
            window: arenas.remove(0),
            dilation: g.dilation as i32,
        }
    }
}

struct Bound {
    _slabs: seismic::SlabTensor,
    hidden: Tensor,
    scales: Tensor,
    value: Tensor,
    norm: Tensor,
    convolution: Tensor,
    slots: SlotTensors,
    window: Tensor,
    dilation: i32,
}

impl Bound {
    fn args(&mut self, slab_banks: u64) -> ngram_inject::Args<'_> {
        ngram_inject::Args {
            hidden: &self.hidden,
            scales: &self.scales,
            value: &self.value,
            norm: &self.norm,
            convolution: &self.convolution,
            segments: &self.slots.segments,
            stop: &self.slots.stop,
            previous_bank: &self.slots.previous,
            previous_tape: &self.slots.previous_tape,
            following_bank: &self.slots.following,
            window: &mut self.window,
            dilation: self.dilation,
            slab_banks: slab_banks as u32,
        }
    }
}

const SMALL: Geometry = Geometry {
    streams: 2,
    hidden: 40,
    taps: 4,
    dilation: 3,
    banks: 7,
    tape: 3,
};

fn small_cases() -> Vec<(&'static str, Case)> {
    vec![
        (
            "decode from the zero seed",
            Case::new(SMALL, 1, vec![slot(1, 1, 0, 3)], 1),
        ),
        (
            "batched decode with padded rows",
            Case::new(
                SMALL,
                5,
                vec![slot(1, 1, 1, 4), slot(1, 1, 2, 5), slot(1, 1, 0, 6)],
                2,
            ),
        ),
        (
            "verify slots from tape versions",
            Case::new(
                SMALL,
                16,
                vec![
                    version(4, 1, 1, 4, 2),
                    version(6, 2, 2, 5, 0),
                    version(3, 3, 3, 6, 3),
                ],
                3,
            ),
        ),
        (
            "prefill slots with an interior stop and a zero stop",
            Case::new(
                SMALL,
                40,
                vec![slot(25, 17, 1, 4), slot(12, 0, 2, 5), slot(1, 1, 3, 6)],
                4,
            ),
        ),
    ]
}

#[test]
fn inject_natives_match_the_portable_body() {
    for (label, case) in small_cases() {
        let oracle = case.oracle();
        for device in devices() {
            let backend = device.backend().as_str();
            for reuse in [false, true] {
                let native = case.native(&device, reuse);
                let label = format!("{backend} {label} (reused slab {reuse})");
                check_close(&label, &native.result, &oracle.result);
                assert!(
                    same_bits(&native.window, &oracle.window),
                    "{label}: window arena differs at {:?}",
                    first_difference(&native.window, &oracle.window)
                );
            }
        }
    }
}

/// qwen4exp's geometry (4 streams of 2560, 4 taps at dilation 3): a prefill
/// split into chunks, each continuing from the bank the previous published,
/// gives the bits of one run over all its rows.
#[test]
fn chunk_boundaries_equal_one_run() {
    let geometry = Geometry {
        streams: 4,
        hidden: 2560,
        taps: 4,
        dilation: 3,
        banks: 8,
        tape: 0,
    };
    let row = geometry.channels();
    for device in devices() {
        let backend = device.backend().as_str();
        let whole = Case::new(geometry, 96, vec![slot(96, 96, 1, 2)], 23);
        let expected = whole.native(&device, false);
        let mut result = Vec::new();
        let mut state: Option<(Case, Outcome)> = None;
        for (chunk, (first, rows)) in [(0usize, 40usize), (40, 1), (41, 4), (45, 51)]
            .into_iter()
            .enumerate()
        {
            let (previous, following) = if chunk == 0 {
                (1, 3)
            } else {
                (2 + chunk, 3 + chunk)
            };
            let slots = vec![slot(rows, rows, previous, following)];
            let mut case = match &state {
                None => Case {
                    rows,
                    slots,
                    ..whole.clone()
                },
                Some((case, outcome)) => case.continued(outcome, rows, slots, 0),
            };
            case.hidden = whole.hidden[first * row..(first + rows) * row].to_vec();
            let s = 2 * geometry.streams;
            case.scales = whole.scales[first * s..(first + rows) * s].to_vec();
            case.value =
                whole.value[first * geometry.hidden..(first + rows) * geometry.hidden].to_vec();
            let outcome = case.native(&device, false);
            result.extend_from_slice(&outcome.result);
            state = Some((case, outcome));
        }
        let (_, last) = state.unwrap();
        assert!(
            same_bits(&result, &expected.result),
            "{backend}: chunked outputs differ from one run"
        );
        let bank = |values: &[f32], bank: usize| {
            values[bank * geometry.window_bank()..][..geometry.window_bank()].to_vec()
        };
        assert!(
            same_bits(&bank(&last.window, 6), &bank(&expected.window, 2)),
            "{backend}: the chunked run's window differs from one run's"
        );
    }
}

/// A committed tape version (bank, j) equals a run that published after those
/// rows: the next advance from either has the same bits.
#[test]
fn tape_versions_equal_runs_that_stopped_there() {
    let verify = 5;
    let geometry = Geometry {
        streams: 4,
        hidden: 256,
        taps: 4,
        dilation: 3,
        banks: 4,
        tape: verify - 1,
    };
    for device in devices() {
        let backend = device.backend().as_str();
        let run = |case: &Case| case.native(&device, false);
        for accepted in 0..verify {
            let tentative = Case::new(geometry, verify, vec![slot(verify, 1, 1, 2)], 41);
            let first = run(&tentative);
            let from_tape =
                run(&tentative.continued(&first, 3, vec![version(3, 3, 2, 3, accepted)], 42));
            let stopped = Case::new(geometry, verify, vec![slot(verify, 1 + accepted, 1, 2)], 41);
            let reference_first = run(&stopped);
            let expected = run(&stopped.continued(&reference_first, 3, vec![slot(3, 3, 2, 3)], 42));
            let bank = |values: &[f32]| {
                values[3 * geometry.window_bank()..4 * geometry.window_bank()].to_vec()
            };
            assert!(
                same_bits(&from_tape.result, &expected.result)
                    && same_bits(&bank(&from_tape.window), &bank(&expected.window)),
                "{backend}: version (bank, {accepted}) differs from a run that stopped after {} rows",
                1 + accepted
            );
        }
    }
}

// ---- ngram_rows -------------------------------------------------------------------------------

/// An n-gram row rule: two predecessors (orders 2 and 3) of three heads each,
/// with multipliers past 2^32 so its products need 64-bit words.
#[derive(Clone)]
struct RowRule {
    context: usize,
    per_order: usize,
    multipliers: Vec<u64>,
    sizes: Vec<u64>,
    offsets: Vec<u64>,
    reset: i32,
}

impl RowRule {
    fn sample() -> Self {
        let sizes = vec![101, 103, 1_000_003, 107, 109, 50_000_017];
        let offsets = sizes
            .iter()
            .scan(0, |end, size| {
                let start = *end;
                *end += size;
                Some(start)
            })
            .collect();
        Self {
            context: 2,
            per_order: 3,
            multipliers: vec![(1 << 40) + 3, (1 << 41) + 17, 999_983 << 20],
            sizes,
            offsets,
            reset: 7,
        }
    }

    fn heads(&self) -> usize {
        self.context * self.per_order
    }

    /// The rows of every position of `tokens`, the sequence's first: the
    /// rule stated as the reference states it (after `context` reset tokens,
    /// a shift valid while the position's segment since the latest reset
    /// strictly before it reaches back that far).
    fn reference(&self, tokens: &[i32]) -> Vec<i32> {
        let history = std::iter::repeat_n(self.reset, self.context)
            .chain(tokens.iter().copied())
            .collect::<Vec<_>>();
        let mut rows = Vec::new();
        for at in self.context..history.len() {
            let previous = history[..at]
                .iter()
                .rposition(|token| *token == self.reset)
                .map_or(-1, |index| index as i64);
            let in_segment = at as i64 - (previous + 1);
            let shifted = (0..=self.context)
                .map(|shift| match in_segment >= shift as i64 {
                    true => history[at - shift] as u64,
                    false => self.reset as u64,
                })
                .collect::<Vec<_>>();
            for order in 2..=self.context + 1 {
                let mixed = (0..order).fold(0u64, |mixed, distance| {
                    mixed ^ shifted[distance].wrapping_mul(self.multipliers[distance])
                });
                let first = (order - 2) * self.per_order;
                for head in first..first + self.per_order {
                    rows.push((self.offsets[head] + mixed % self.sizes[head]) as i32);
                }
            }
        }
        rows
    }
}

/// Tokens of a sequence: resets at `resets`, others below 50.
fn sequence(length: usize, resets: &[usize], seed: u64) -> Vec<i32> {
    let mut state = seed;
    (0..length)
        .map(|index| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            match resets.contains(&index) {
                true => 7,
                false => ((state >> 33) % 50) as i32,
            }
        })
        .collect()
}

#[derive(Clone)]
struct RowsCase {
    rule: RowRule,
    tape: usize,
    banks: usize,
    slots: Vec<SlotCase>,
    /// [rows, 2]: (token, status).
    tokens: Vec<i32>,
    /// [banks, C + T]: token + 1, 0 for none.
    window: Vec<u32>,
    slab_banks: u64,
}

struct RowsOutcome {
    /// [rows * heads, 2]: (row, status).
    rows: Vec<i32>,
    window: Vec<u32>,
}

impl RowsCase {
    fn window_rows(&self) -> usize {
        self.rule.context + self.tape
    }

    fn rows(&self) -> usize {
        self.tokens.len() / 2
    }

    /// A launch of `tokens` (status 0) over `slots`, its windows the zero
    /// seed in bank 0 and seeded tokens in every slot's source bank.
    fn new(rule: RowRule, tape: usize, slots: Vec<SlotCase>, tokens: &[i32], seed: u64) -> Self {
        let (banks, window_rows) = (8, rule.context + tape);
        let mut window = vec![0xdead_u32; banks * window_rows];
        window[..window_rows].fill(0);
        let mut state = seed;
        for slot in &slots {
            if slot.previous != 0 {
                for value in &mut window[slot.previous * window_rows..][..window_rows] {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    // Stored tokens, the reset's among them, and no token.
                    *value = match (state >> 33) % 6 {
                        0 => 0,
                        1 => rule.reset as u32 + 1,
                        _ => ((state >> 40) % 50) as u32 + 1,
                    };
                }
            }
        }
        Self {
            rule,
            tape,
            banks,
            slots,
            tokens: tokens.iter().flat_map(|token| [*token, 0]).collect(),
            window,
            slab_banks: 2,
        }
    }

    /// The next launch of the same layer after `outcome`.
    fn continued(&self, outcome: &RowsOutcome, slots: Vec<SlotCase>, tokens: &[i32]) -> Self {
        Self {
            slots,
            tokens: tokens.iter().flat_map(|token| [*token, 0]).collect(),
            window: outcome.window.clone(),
            ..self.clone()
        }
    }

    fn oracle(&self) -> RowsOutcome {
        use seismic_lang::{
            checked::{SourceFile, check_source},
            entry::ElementBindings,
            failure::SourceTermination,
            interp::{Arg, Interpreter, OutcomeValue, TensorData},
            reference_math::ReferenceScalar,
            types::DType,
        };
        let mut sources = seismic_std::sources();
        sources.push(SourceFile {
            path: "ngram.seismic".into(),
            text: include_str!("../kernels/ngram.seismic").into(),
        });
        let module = check_source(sources).unwrap();
        let logical = module
            .entry(module.entry_named("ngram_rows").unwrap(), &ElementBindings::new())
            .unwrap();
        let mut interpreter = Interpreter::new(&logical);
        let words = |values: &[u64]| {
            TensorData::dense_from_bytes(
                DType::U64,
                vec![values.len()],
                values.iter().flat_map(|value| value.to_le_bytes()).collect(),
            )
            .unwrap()
        };
        let each = |f: fn(&SlotCase) -> usize| {
            self.slots
                .iter()
                .map(|slot| f(slot) as i32)
                .collect::<Vec<_>>()
        };
        let count = self.slots.len();
        let tensors = vec![
            integers(vec![self.rows(), 2], self.tokens.clone()),
            words(&self.rule.multipliers),
            words(&self.rule.sizes),
            words(&self.rule.offsets),
            integers(vec![count + 1, 2], segments(&self.slots, self.rows())),
            integers(vec![count], each(|slot| slot.stop)),
            integers(vec![count], each(|slot| slot.previous)),
            integers(vec![count], each(|slot| slot.taped)),
            integers(vec![count], each(|slot| slot.following)),
            TensorData::dense_from_bytes(
                DType::U32,
                vec![self.banks, self.window_rows(), 1],
                self.window.iter().flat_map(|value| value.to_le_bytes()).collect(),
            )
            .unwrap(),
        ];
        let mut arguments = tensors
            .into_iter()
            .map(|tensor| Arg::Tensor(interpreter.add_tensor(tensor)))
            .collect::<Vec<_>>();
        arguments.push(Arg::Scalar(ReferenceScalar::I32(self.rule.reset)));
        arguments.push(Arg::Scalar(ReferenceScalar::U32(self.slab_banks as u32)));
        let outcome = interpreter.run_bounded(&arguments, u64::MAX).unwrap();
        if let SourceTermination::Failed(failure) = outcome.termination() {
            panic!("portable ngram_rows body failed: {failure}");
        }
        let read = |reader: seismic_lang::interp::TensorReader<'_>| {
            (0..reader.element_count())
                .map(|index| reader.read(index).unwrap() as i64)
                .collect::<Vec<_>>()
        };
        let rows = match outcome.results().next().unwrap().value() {
            OutcomeValue::Tensor(reader) => read(reader),
            _ => panic!("the result is a tensor"),
        };
        let window = outcome
            .inputs()
            .find(|input| input.ordinal() == 9)
            .map(|input| read(input.tensor()))
            .expect("the window is a mutable input");
        RowsOutcome {
            rows: rows.into_iter().map(|row| row as i32).collect(),
            window: window.into_iter().map(|value| value as u32).collect(),
        }
    }

    fn native(&self, device: &Device, reuse: bool) -> RowsOutcome {
        let rows = self.rows() as u64;
        let words = |values: &[u64]| {
            Tensor::from_host(
                device,
                Element::u64(),
                &[values.len() as u64],
                &values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>(),
            )
            .unwrap()
        };
        let stored = self
            .window
            .iter()
            .map(|value| f32::from_bits(*value))
            .collect::<Vec<_>>();
        let (_slabs, mut arenas) = slab_arenas(
            device,
            self.banks,
            self.slab_banks,
            &[Arena {
                element: Element::u32(),
                bank_shape: vec![self.window_rows() as u64, 1],
                values: &stored,
            }],
            reuse,
        );
        let mut window = arenas.remove(0);
        let slots = SlotTensors::new(device, &self.slots, self.rows());
        let kernel = ngram_rows::native_for_device(
            device,
            &defaults::<ngram_rows::Entry>(
                device,
                &[
                    ("C", self.rule.context as u64),
                    ("P", self.rule.per_order as u64),
                ],
            ),
        )
        .unwrap();
        let result = kernel
            .call(ngram_rows::Args {
                tokens: &ints(device, &[rows, 2], &self.tokens),
                multipliers: &words(&self.rule.multipliers),
                sizes: &words(&self.rule.sizes),
                offsets: &words(&self.rule.offsets),
                segments: &slots.segments,
                stop: &slots.stop,
                previous_bank: &slots.previous,
                previous_tape: &slots.previous_tape,
                following_bank: &slots.following,
                window: &mut window,
                reset: self.rule.reset,
                slab_banks: self.slab_banks as u32,
            })
            .unwrap()
            .value;
        let words32 = |tensor: &Tensor| {
            tensor
                .read_to_host()
                .unwrap()
                .chunks_exact(4)
                .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        RowsOutcome {
            rows: words32(&result).into_iter().map(|word| word as i32).collect(),
            window: words32(&window),
        }
    }
}

/// The row ids of `outcome`'s rows, without their statuses.
fn row_ids(outcome: &RowsOutcome) -> Vec<i32> {
    outcome.rows.chunks_exact(2).map(|pair| pair[0]).collect()
}

fn rows_cases() -> Vec<(&'static str, RowsCase)> {
    let rule = RowRule::sample();
    let tokens = |length, seed| sequence(length, &[3, 4, 11], seed);
    vec![
        (
            "decode from the zero seed",
            RowsCase::new(rule.clone(), 3, vec![slot(1, 1, 0, 3)], &tokens(1, 1), 1),
        ),
        (
            "batched decode with padded rows and a failed selection",
            {
                let mut case = RowsCase::new(
                    rule.clone(),
                    3,
                    vec![slot(1, 1, 1, 4), slot(1, 1, 2, 5), slot(1, 1, 0, 6)],
                    &[7, 12, 30, 0, 0],
                    2,
                );
                case.tokens[3] = -1;
                case
            },
        ),
        (
            "verify slots from tape versions",
            RowsCase::new(
                rule.clone(),
                3,
                vec![
                    version(4, 1, 1, 4, 2),
                    version(6, 2, 2, 5, 0),
                    version(3, 3, 3, 6, 3),
                ],
                &tokens(16, 3),
                3,
            ),
        ),
        (
            "prefill slots with an interior stop and a zero stop",
            RowsCase::new(
                rule,
                3,
                vec![slot(25, 17, 1, 4), slot(12, 0, 2, 5), slot(1, 1, 3, 6)],
                &tokens(40, 4),
                4,
            ),
        ),
    ]
}

#[test]
fn rows_natives_match_the_portable_body() {
    for (label, case) in rows_cases() {
        let oracle = case.oracle();
        for device in devices() {
            let backend = device.backend().as_str();
            for reuse in [false, true] {
                let native = case.native(&device, reuse);
                let label = format!("{backend} {label} (reused slab {reuse})");
                assert_eq!(native.rows, oracle.rows, "{label}: rows differ");
                assert_eq!(native.window, oracle.window, "{label}: token windows differ");
            }
        }
    }
}

/// A sequence fed in chunks, each continuing from the bank the previous
/// published, names the rows of the rule over the whole sequence: on every
/// backend and in the portable body.
#[test]
fn rows_follow_the_rule_across_chunks() {
    let rule = RowRule::sample();
    // Resets at the start, adjacent and mid-sequence.
    let tokens = sequence(64, &[0, 9, 10, 33], 11);
    let expected = rule.reference(&tokens);
    let chunks = [(0usize, 1usize), (1, 8), (9, 1), (10, 1), (11, 23), (34, 30)];
    let run = |launch: &dyn Fn(&RowsCase) -> RowsOutcome| {
        let mut ids = Vec::new();
        let mut state: Option<(RowsCase, RowsOutcome)> = None;
        for (chunk, (first, length)) in chunks.into_iter().enumerate() {
            // Bank 0 is the zero seed; chunk k publishes to bank k + 1.
            let slots = vec![slot(length, length, chunk, chunk + 1)];
            let tokens = &tokens[first..first + length];
            let case = match &state {
                None => RowsCase::new(rule.clone(), 0, slots, tokens, 0),
                Some((case, outcome)) => case.continued(outcome, slots, tokens),
            };
            let outcome = launch(&case);
            ids.extend(row_ids(&outcome));
            state = Some((case, outcome));
        }
        ids
    };
    assert_eq!(run(&|case| case.oracle()), expected, "portable body");
    for device in devices() {
        let backend = device.backend().as_str();
        assert_eq!(run(&|case| case.native(&device, false)), expected, "{backend}");
    }
}

/// A committed tape version (bank, j) continues the sequence that accepted
/// those j rows: the next launch from it names the rule's rows.
#[test]
fn rows_continue_tape_versions() {
    let rule = RowRule::sample();
    let prefix = sequence(6, &[2], 21);
    let verify = [7, 13, 21, 7, 9];
    let next = [4, 7, 8];
    for device in devices() {
        let backend = device.backend().as_str();
        let first = RowsCase::new(rule.clone(), 4, vec![slot(6, 6, 0, 1)], &prefix, 0);
        let first_outcome = first.native(&device, false);
        let tentative = first.continued(&first_outcome, vec![slot(5, 1, 1, 2)], &verify);
        let verified = tentative.native(&device, false);
        for accepted in 0..verify.len() {
            let from_tape = tentative.continued(&verified, vec![version(3, 3, 2, 3, accepted)], &next);
            let sequence = prefix
                .iter()
                .chain(&verify[..1 + accepted])
                .chain(&next)
                .copied()
                .collect::<Vec<_>>();
            let expected = rule.reference(&sequence);
            assert_eq!(
                row_ids(&from_tape.native(&device, false)),
                expected[expected.len() - next.len() * rule.heads()..],
                "{backend}: version (bank, {accepted})"
            );
        }
    }
}

/// Dense `u64` tensors (the row rule's words) reach every backend's storage
/// and come back unchanged, full width.
#[test]
fn u64_tensors_round_trip() {
    let values = [0, 1, u64::from(u32::MAX) + 1, u64::MAX - 1, u64::MAX, 0x0123_4567_89ab_cdef];
    let bytes = values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
    for device in devices() {
        let tensor = Tensor::from_host(&device, Element::u64(), &[values.len() as u64], &bytes).unwrap();
        assert_eq!(tensor.read_to_host().unwrap(), bytes, "{}", device.backend().as_str());
    }
}
