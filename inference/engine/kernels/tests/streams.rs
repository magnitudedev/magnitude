//! Hyper-connection stream entries (`streams.seismic`): the portable bodies
//! define the contract; native implementations are checked against them on
//! the host's GPU backend and on the CPU, at every row class of their
//! launches, at a small geometry and at Qwen3.8 Flash Next's (H 2560, four
//! streams, rank 320).

use magnitude_kernels::{repeat_rows, stream_mix_down, stream_mix_up};
use seismic_lang::{
    checked::{CheckedModule, SourceFile, check_source},
    entry::ElementBindings,
    failure::SourceTermination,
    interp::{Arg, Interpreter, OracleOutcome, OutcomeValue, TensorData},
    reference_math::ReferenceScalar,
    registry,
    types::DType,
};

fn module() -> CheckedModule {
    let mut sources = seismic_std::sources();
    sources.push(SourceFile {
        path: "streams.seismic".into(),
        text: include_str!("../kernels/streams.seismic").into(),
    });
    check_source(sources).unwrap()
}

enum Input {
    Tensor(DType, Vec<usize>, Vec<f32>),
    F32(f32),
}

/// Every result of the portable body of `name`, flattened.
fn interpret(
    module: &CheckedModule,
    name: &str,
    bindings: &[(&str, DType)],
    inputs: Vec<Input>,
) -> Vec<Vec<f64>> {
    let elements = bindings
        .iter()
        .fold(ElementBindings::new(), |elements, (name, dtype)| {
            elements.bind(name, registry::dense(*dtype))
        });
    let logical = module
        .entry(module.entry_named(name).unwrap(), &elements)
        .unwrap();
    let mut interpreter = Interpreter::new(&logical);
    let arguments = inputs
        .into_iter()
        .map(|input| match input {
            Input::Tensor(dtype, shape, values) => Arg::Tensor(interpreter.add_tensor(
                TensorData::dense(dtype, shape, values.iter().map(|v| f64::from(*v)).collect()),
            )),
            Input::F32(value) => Arg::Scalar(ReferenceScalar::F32(value.to_bits())),
        })
        .collect::<Vec<_>>();
    let outcome: OracleOutcome = interpreter
        .run(&arguments)
        .unwrap_or_else(|error| panic!("{name} interpreter error: {error}"));
    if let SourceTermination::Failed(failure) = outcome.termination() {
        panic!("{name} failed: {failure}");
    }
    outcome
        .results()
        .map(|result| {
            let OutcomeValue::Tensor(tensor) = result.value() else {
                panic!("tensor result")
            };
            (0..tensor.element_count())
                .map(|i| tensor.read(i).unwrap())
                .collect()
        })
        .collect()
}

/// Deterministic values in [-scale, scale).
fn pattern(count: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            ((state >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

fn bf16(values: &[f32]) -> Vec<f32> {
    values.iter().map(|v| registry::bf16_round(*v)).collect()
}

fn tensor(
    device: &seismic::Device,
    dtype: DType,
    shape: &[usize],
    values: &[f32],
) -> seismic::Tensor {
    let (element, bytes) = match dtype {
        DType::F32 => (
            seismic::Element::f32(),
            values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        ),
        DType::BF16 => (
            seismic::Element::bf16(),
            values
                .iter()
                .flat_map(|v| ((registry::bf16_round(*v).to_bits() >> 16) as u16).to_le_bytes())
                .collect(),
        ),
        other => panic!("unsupported element {other:?}"),
    };
    let shape = shape
        .iter()
        .map(|extent| *extent as u64)
        .collect::<Vec<_>>();
    seismic::Tensor::from_host(device, element, &shape, &bytes).unwrap()
}

fn read(tensor: &seismic::Tensor, dtype: DType) -> Vec<f32> {
    let bytes = tensor.read_to_host().unwrap();
    match dtype {
        DType::F32 => bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect(),
        DType::BF16 => bytes
            .chunks_exact(2)
            .map(|b| f32::from_bits(u32::from(u16::from_le_bytes(b.try_into().unwrap())) << 16))
            .collect(),
        other => panic!("unsupported element {other:?}"),
    }
}

/// CUDA, else Metal; `SEISMIC_TEST_BACKEND=vulkan` selects Vulkan (a CUDA
/// host also has a Vulkan device). Then the CPU.
fn devices() -> Vec<seismic::Device> {
    let catalog = seismic::Devices::discover(&[]).unwrap();
    let gpu = match std::env::var("SEISMIC_TEST_BACKEND").ok().as_deref() {
        Some("vulkan") => Some(
            catalog
                .open_backend(seismic::BackendName::Vulkan)
                .expect("the Vulkan device opens"),
        ),
        Some("cuda") | None => [seismic::BackendName::Cuda, seismic::BackendName::Metal]
            .into_iter()
            .find_map(|backend| catalog.open_backend(backend).ok()),
        Some(other) => panic!("SEISMIC_TEST_BACKEND={other}: only cuda and vulkan are selectable"),
    };
    gpu.into_iter()
        .chain(std::iter::once(
            catalog.open_backend(seismic::BackendName::Cpu).unwrap(),
        ))
        .collect()
}

/// The declared defaults of `E` at `statics` (none on the CPU).
fn defaults<E: seismic::Entry>(
    device: &seismic::Device,
    statics: &[(&str, u64)],
) -> seismic::NativeSpecialization {
    let statics = if device.backend() == seismic::BackendName::Cpu {
        seismic::NativeSpecialization::new()
    } else {
        statics.iter().fold(
            seismic::NativeSpecialization::new(),
            |spec, (name, value)| spec.with_static(*name, *value),
        )
    };
    seismic::generated::native_implementation::<E>(device)
        .unwrap()
        .unwrap()
        .default_specialization(&statics)
        .unwrap()
}

/// Every value within `absolute + relative * |portable|` of the portable
/// value: the projections reassociate their sums.
fn assert_close(name: &str, actual: &[f32], expected: &[f64], absolute: f64, relative: f64) {
    assert_eq!(actual.len(), expected.len(), "{name} length");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let error = (f64::from(*actual) - expected).abs();
        assert!(
            error <= absolute + relative * expected.abs(),
            "{name}[{index}]: native {actual}, portable {expected}"
        );
    }
}

/// Small and Qwen3.8 Flash Next geometries (H, S, R), and the row counts of
/// every launch class: GEMV (1, 2, 5), batched or 16-row GEMV (12), small
/// GEMM (17) and GEMM (70).
const GEOMETRIES: [(usize, usize, usize); 2] = [(256, 4, 64), (2560, 4, 320)];
const ROWS: [usize; 6] = [1, 2, 5, 12, 17, 70];
const EPSILON: f32 = 1e-6;

/// One case's inputs: streams of O(10) with per-stream scales, an F32 norm
/// near 1, bf16 low-rank weights and (C = 1) F32 inject weights.
struct Case {
    h: usize,
    s: usize,
    r: usize,
    c: usize,
    rows: usize,
    hidden: Vec<f32>,
    norm: Vec<f32>,
    down: Vec<f32>,
    up: Vec<f32>,
    inject: Vec<f32>,
}

impl Case {
    fn new((h, s, r): (usize, usize, usize), c: usize, rows: usize) -> Self {
        let width = s * h;
        let mut hidden = pattern(rows * width, 11 + rows as u32, 10.0);
        for (index, value) in hidden.iter_mut().enumerate() {
            *value *= 1.0 + (index % width / h) as f32;
        }
        Self {
            h,
            s,
            r,
            c,
            rows,
            hidden,
            norm: pattern(width, 3, 0.5).iter().map(|v| 1.0 + v).collect(),
            down: bf16(&pattern(r * width, 5, 0.04)),
            up: bf16(&pattern(width * r, 7, 0.2)),
            inject: pattern(c * s * width, 9, 0.03),
        }
    }

    fn width(&self) -> usize {
        self.s * self.h
    }

    /// Per inject value, the most a projection that rounds its F32 inject
    /// weights to the activation element (bf16, a relative 2^-8) moves it:
    /// `2 sigmoid(v / S)` changes by at most half of `v`'s change over S.
    fn inject_bounds(&self) -> Vec<f64> {
        let (s, h, width) = (self.s, self.h, self.width());
        let mut bounds = Vec::new();
        for row in 0..self.rows {
            let x = &self.hidden[row * width..(row + 1) * width];
            let scales = (0..s)
                .map(|stream| {
                    let squares: f64 = x[stream * h..(stream + 1) * h]
                        .iter()
                        .map(|v| f64::from(*v).powi(2))
                        .sum();
                    1.0 / (squares / h as f64 + f64::from(EPSILON)).sqrt()
                })
                .collect::<Vec<_>>();
            for output in 0..self.c * s {
                let magnitude: f64 = (0..width)
                    .map(|at| {
                        (f64::from(x[at]) * scales[at / h] * f64::from(self.norm[at])
                            * f64::from(self.inject[output * width + at]))
                        .abs()
                    })
                    .sum();
                bounds.push(0.5 * magnitude * 2f64.powi(-8) / s as f64);
            }
        }
        bounds
    }

    /// The portable bodies in the interpreter: (low, inject, inverse) of
    /// `stream_mix_down`, then the partials of `stream_mix_up` over them.
    fn portable(&self, module: &CheckedModule) -> [Vec<f64>; 4] {
        let (s, h, r, rows, width) = (self.s, self.h, self.r, self.rows, self.width());
        let down = interpret(
            module,
            "stream_mix_down",
            &[
                ("NW", DType::F32),
                ("DW", DType::BF16),
                ("IW", DType::F32),
                ("A", DType::BF16),
            ],
            vec![
                Input::Tensor(DType::F32, vec![rows, s, h], self.hidden.clone()),
                Input::Tensor(DType::F32, vec![s, h], self.norm.clone()),
                Input::Tensor(DType::BF16, vec![r, width], self.down.clone()),
                Input::Tensor(DType::F32, vec![self.c * s, width], self.inject.clone()),
                Input::F32(EPSILON),
            ],
        );
        let low = down[0].iter().map(|v| *v as f32).collect::<Vec<_>>();
        let inverse = down[2].iter().map(|v| *v as f32).collect::<Vec<_>>();
        let up = interpret(
            module,
            "stream_mix_up",
            &[("UW", DType::BF16), ("NW", DType::F32), ("A", DType::BF16)],
            vec![
                Input::Tensor(DType::BF16, vec![rows, r], low),
                Input::Tensor(DType::BF16, vec![width, r], self.up.clone()),
                Input::Tensor(DType::F32, vec![rows, s, h], self.hidden.clone()),
                Input::Tensor(DType::F32, vec![rows, s], inverse),
                Input::Tensor(DType::F32, vec![s, h], self.norm.clone()),
            ],
        );
        let [low, inject, inverse] = <[Vec<f64>; 3]>::try_from(down).unwrap();
        let [partials] = <[Vec<f64>; 1]>::try_from(up).unwrap();
        [low, inject, inverse, partials]
    }

    /// The bodies' recipe on the host, sums in F64: the reference at
    /// geometries the interpreter is too slow for. Its roundings to bf16 are
    /// the bodies' (the normalized streams, the low rows).
    fn host(&self) -> [Vec<f64>; 4] {
        let (s, h, r, rows, width) = (self.s, self.h, self.r, self.rows, self.width());
        let sigmoid = |v: f64| 1.0 / (1.0 + (-v).exp());
        let round = |v: f64| f64::from(registry::bf16_round(v as f32));
        let (mut low, mut inject, mut inverse, mut partials) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for row in 0..rows {
            let x = &self.hidden[row * width..(row + 1) * width];
            let scales = (0..s)
                .map(|stream| {
                    let squares: f64 = x[stream * h..(stream + 1) * h]
                        .iter()
                        .map(|v| f64::from(*v).powi(2))
                        .sum();
                    1.0 / (squares / h as f64 + f64::from(EPSILON)).sqrt()
                })
                .collect::<Vec<_>>();
            inverse.extend(&scales);
            let normalized = (0..width)
                .map(|at| round(f64::from(x[at]) * scales[at / h] * f64::from(self.norm[at])))
                .collect::<Vec<_>>();
            let project = |weight: &[f32], output: usize| -> f64 {
                (0..width)
                    .map(|at| normalized[at] * f64::from(weight[output * width + at]))
                    .sum()
            };
            let row_low = (0..r)
                .map(|output| {
                    let v = project(&self.down, output) / s as f64;
                    round(v * sigmoid(v))
                })
                .collect::<Vec<_>>();
            for output in 0..self.c * s {
                inject.push(2.0 * sigmoid(project(&self.inject, output) / s as f64));
            }
            for at in 0..width {
                let gate: f64 = (0..r)
                    .map(|k| row_low[k] * f64::from(self.up[at * r + k]))
                    .sum();
                let normalized = f64::from(x[at]) * scales[at / h] * f64::from(self.norm[at]);
                partials.push(sigmoid(gate) * normalized / s as f64);
            }
            low.extend(row_low);
        }
        [low, inject, inverse, partials]
    }

    /// The native entries on `device`: (low, inject, inverse) of the first
    /// stage, and the second stage's partials over the expected first
    /// stage's low rows and inverses.
    fn native(&self, device: &seismic::Device, expected: &[Vec<f64>; 4]) -> [Vec<f32>; 4] {
        let (s, h, r, rows, width) = (self.s, self.h, self.r, self.rows, self.width());
        let statics = [
            ("H", h as u64),
            ("S", s as u64),
            ("LR", r as u64),
            ("C", self.c as u64),
        ];
        let mixed = stream_mix_down::native_for_device_with(
            device,
            stream_mix_down::Elements {
                NW: seismic::Element::f32(),
                DW: seismic::Element::bf16(),
                IW: seismic::Element::f32(),
                A: seismic::Element::bf16(),
            },
            &defaults::<stream_mix_down::Entry>(device, &statics),
        )
        .unwrap()
        .call(stream_mix_down::Args {
            hidden: &tensor(device, DType::F32, &[rows, s, h], &self.hidden),
            norm: &tensor(device, DType::F32, &[s, h], &self.norm),
            down_weight: &tensor(device, DType::BF16, &[r, width], &self.down),
            inject_weight: &tensor(device, DType::F32, &[self.c * s, width], &self.inject),
            epsilon: EPSILON,
        })
        .unwrap();
        let low = expected[0].iter().map(|v| *v as f32).collect::<Vec<_>>();
        let inverse = expected[2].iter().map(|v| *v as f32).collect::<Vec<_>>();
        let gated = stream_mix_up::native_for_device_with(
            device,
            stream_mix_up::Elements {
                UW: seismic::Element::bf16(),
                NW: seismic::Element::f32(),
                A: seismic::Element::bf16(),
            },
            &defaults::<stream_mix_up::Entry>(device, &statics[..3]),
        )
        .unwrap()
        .call(stream_mix_up::Args {
            low: &tensor(device, DType::BF16, &[rows, r], &low),
            up_weight: &tensor(device, DType::BF16, &[width, r], &self.up),
            hidden: &tensor(device, DType::F32, &[rows, s, h], &self.hidden),
            inverse: &tensor(device, DType::F32, &[rows, s], &inverse),
            norm: &tensor(device, DType::F32, &[s, h], &self.norm),
        })
        .unwrap();
        [
            read(&mixed.r0, DType::BF16),
            read(&mixed.r1, DType::F32),
            read(&mixed.r2, DType::F32),
            read(&gated.value, DType::F32),
        ]
    }
}

/// Every stage within its tolerance of `expected`: one bf16 rounding of a
/// low row may land on either side; the projections reassociate their sums.
fn assert_stages(
    label: &str,
    actual: &[Vec<f32>; 4],
    expected: &[Vec<f64>; 4],
    inject_bounds: &[f64],
) {
    let tolerances = [
        ("low", 1e-4, 1e-2),
        ("inverse", 0.0, 1e-5),
        ("partials", 1e-6, 1e-4),
    ];
    for ((name, absolute, relative), stage) in tolerances.into_iter().zip([0, 2, 3]) {
        assert_close(
            &format!("{label} {name}"),
            &actual[stage],
            &expected[stage],
            absolute,
            relative,
        );
    }
    assert_eq!(actual[1].len(), expected[1].len(), "{label} inject length");
    for (index, ((actual, expected), bound)) in
        actual[1].iter().zip(&expected[1]).zip(inject_bounds).enumerate()
    {
        assert!(
            (f64::from(*actual) - expected).abs() <= 1e-6 + bound,
            "{label} inject[{index}]: native {actual}, expected {expected}, bound {bound}"
        );
    }
}

/// The host replica agrees with the portable bodies at the small geometry,
/// at every launch's row count, with and without inject values.
#[test]
fn host_replica_matches_portable_bodies() {
    let module = module();
    for c in [1, 0] {
        for rows in ROWS {
            let case = Case::new(GEOMETRIES[0], c, rows);
            let portable = case.portable(&module);
            let host = case.host();
            let label = format!("host vs portable C {c} M {rows}");
            for (stage, tolerance) in [(0, 1e-2), (1, 1e-6), (2, 1e-6), (3, 1e-5)] {
                assert_close(
                    &format!("{label} stage {stage}"),
                    &host[stage].iter().map(|v| *v as f32).collect::<Vec<_>>(),
                    &portable[stage],
                    1e-5,
                    tolerance,
                );
            }
        }
    }
}

/// The natives against the portable bodies at the small geometry and
/// against the host replica at Qwen3.8 Flash Next's.
#[test]
fn native_stream_mix_matches_portable_bodies() {
    let module = module();
    let devices = devices();
    for geometry in GEOMETRIES {
        for c in [1, 0] {
            for rows in ROWS {
                let case = Case::new(geometry, c, rows);
                let expected = if geometry == GEOMETRIES[0] {
                    case.portable(&module)
                } else {
                    case.host()
                };
                let bounds = case.inject_bounds();
                for device in &devices {
                    let (h, s, r) = geometry;
                    let label = format!("{:?} H {h} S {s} R {r} C {c} M {rows}", device.backend());
                    assert_stages(&label, &case.native(device, &expected), &expected, &bounds);
                }
            }
        }
    }
}

/// `repeat_rows`: S scaled copies of each row, exactly the portable body's
/// (one F32 product each), at scale 1 (the streams' entry) and others.
#[test]
fn native_repeat_rows_matches_portable_body() {
    let module = module();
    for device in devices() {
        for (h, rows, scales) in [
            (256usize, 1usize, vec![1.0f32; 4]),
            (2560, 70, vec![1.0, -0.5, 3.25, 0.0]),
        ] {
            let s = scales.len();
            let source = pattern(rows * h, 13, 30.0);
            let expected = interpret(
                &module,
                "repeat_rows",
                &[],
                vec![
                    Input::Tensor(DType::F32, vec![rows, h], source.clone()),
                    Input::Tensor(DType::F32, vec![s], scales.clone()),
                ],
            );
            let repeated = repeat_rows::native_for_device(
                &device,
                &defaults::<repeat_rows::Entry>(&device, &[]),
            )
            .unwrap()
            .call(repeat_rows::Args {
                rows: &tensor(&device, DType::F32, &[rows, h], &source),
                scales: &tensor(&device, DType::F32, &[s], &scales),
            })
            .unwrap();
            let actual = read(&repeated.value, DType::F32);
            assert_eq!(actual.len(), rows * s * h);
            for (index, (actual, expected)) in actual.iter().zip(&expected[0]).enumerate() {
                assert_eq!(
                    f64::from(*actual),
                    *expected,
                    "{:?} repeat_rows[{index}]",
                    device.backend()
                );
            }
        }
    }
}
