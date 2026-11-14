//! Hyper-connection stream entries (`streams.seismic`): the portable bodies
//! define the contract; native implementations are checked against them on
//! the host's GPU backend and on the CPU, at every row class of their
//! launches, at a small geometry and at Qwen3.8 Flash Next's (H 2560, four
//! streams, rank 320). The Sinkhorn site and its comb join are checked
//! against the formulas of DeepSeek V4.1 Flash's single pass, written here,
//! at its geometry (H 5120, four streams, 20 iterations); the site mixing by
//! its own pre and the streams' mean exit against GLM-5.3-Flash's
//! (transformers `Glm5NextTextHyperConnection`, `Glm5NextTextHyperHead`) at
//! its geometry (H 4096, four streams, 20 iterations).

mod common;
mod declared;

use common::bound::gamma;

use magnitude_kernels::{
    attention_output, repeat_rows, stream_mix_down, stream_mix_sinkhorn, stream_mix_up,
};
use seismic_lang::{
    checked::{CheckedModule, SourceFile},
    entry::ElementBindings,
    failure::SourceTermination,
    interp::{Arg, Interpreter, OracleOutcome, OutcomeValue, TensorData},
    reference_math::ReferenceScalar,
    registry,
    types::DType,
};

fn module() -> CheckedModule {
    let mut sources = seismic_std::sources();
    // `stream_join` joins through `scatter`.
    sources.push(SourceFile {
        path: "functions.seismic".into(),
        text: include_str!("../kernels/functions.seismic").into(),
    });
    sources.push(SourceFile {
        path: "streams.seismic".into(),
        text: include_str!("../kernels/streams.seismic").into(),
    });
    // The Sinkhorn sites split at `DECODE_ROWS`, which `routed.seismic`
    // declares, beside the words it reads.
    sources.push(SourceFile {
        path: "routed.seismic".into(),
        text: include_str!("../kernels/routed.seismic").into(),
    });
    sources.push(SourceFile {
        path: "streaming_words.seismic".into(),
        text: include_str!("../kernels/streaming_words.seismic").into(),
    });
    crate::common::check(sources)
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

/// CUDA, else Metal, then the CPU; a requested backend (Vulkan among them)
/// alone ([`seismic_native_tests::first_and_cpu`]).
fn devices() -> Vec<seismic::Device> {
    use seismic::BackendName::{Cpu, Cuda, Metal, Vulkan};
    seismic_native_tests::first_and_cpu(&[Cuda, Metal], &[Cuda, Metal, Vulkan, Cpu])
}

/// `entry` prepared at its declared defaults as its bound elements form it,
/// each static it declares taken from `statics` (the CPU declares none).
fn at_defaults<E: seismic::Entry>(
    device: &seismic::Device,
    entry: seismic::BoundEntry<E>,
    statics: &[(&str, u64)],
) -> Result<seismic::NativeKernel<E>, seismic::LoadError> {
    let specialization = declared::declared(device, &entry, |name| {
        statics
            .iter()
            .find(|(given, _)| *given == name)
            .unwrap_or_else(|| panic!("static {name} is given"))
            .1
    }, &[]).unwrap();
    entry.prepare(device, &specialization)
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
        let mixed = at_defaults(
            device,
            stream_mix_down::native_entry_with(stream_mix_down::Elements {
                NW: seismic::Element::f32(),
                DW: seismic::Element::bf16(),
                IW: seismic::Element::f32(),
                A: seismic::Element::bf16(),
            }),
            &statics,
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
        let gated = at_defaults(
            device,
            stream_mix_up::native_entry_with(stream_mix_up::Elements {
                UW: seismic::Element::bf16(),
                NW: seismic::Element::f32(),
                A: seismic::Element::bf16(),
            }),
            &statics[..3],
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
                let (h, s, r) = geometry;
                seismic_native_tests::progress(format_args!(
                    "stream mix reference H {h} S {s} R {r} C {c} M {rows}"
                ));
                let expected = if geometry == GEOMETRIES[0] {
                    case.portable(&module)
                } else {
                    case.host()
                };
                let bounds = case.inject_bounds();
                for device in &devices {
                    let label = format!("{:?} H {h} S {s} R {r} C {c} M {rows}", device.backend());
                    seismic_native_tests::progress(format_args!("stream mix {label}"));
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
            let repeated = at_defaults(
                &device,
                repeat_rows::native_entry(),
                &[],
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

/// DeepSeek V4.1 Flash's mix sites: H, S, Sinkhorn iterations, and the
/// mixing's epsilon and RMS statistic's (`hc_eps`, `norm_eps`).
const V41: (usize, usize, usize) = (5120, 4, 20);
const V41_EPSILON: f32 = 1e-6;
const V41_NORM_EPSILON: f32 = 1e-20;

/// GLM-5.3-Flash's mix sites: H, S and Sinkhorn iterations (`hc_mult`,
/// `hc_sinkhorn_iters`), and the mixing's epsilon and its RMS statistic's
/// (`hc_eps`, `rms_norm_eps`).
const GLM: (usize, usize, usize) = (4096, 4, 20);
const GLM_EPSILONS: (f32, f32) = (1e-6, 1e-5);

/// One Sinkhorn site's weights: `function [(2 + S) S, S H]`, `base`,
/// `scale [3]`, at magnitudes that put every mix at O(1), and its mixing's
/// epsilon and RMS statistic's epsilon.
struct Sinkhorn {
    function: Vec<f32>,
    base: Vec<f32>,
    scale: Vec<f32>,
    epsilons: (f32, f32),
}

impl Sinkhorn {
    fn new(h: usize, s: usize, seed: u32) -> Self {
        Self::with_epsilons(h, s, seed, (V41_EPSILON, V41_NORM_EPSILON))
    }

    fn with_epsilons(h: usize, s: usize, seed: u32, epsilons: (f32, f32)) -> Self {
        let mixes = (2 + s) * s;
        Self {
            function: pattern(mixes * s * h, seed, 4.0 / ((s * h) as f32).sqrt()),
            base: pattern(mixes, seed + 1, 1.0),
            scale: vec![0.6, 0.9, 1.7],
            epsilons,
        }
    }

    /// The site of the streams `x [rows, S H]` mixing its input by the
    /// previous site's `previous [rows, S]` (V4.1), or by its own pre
    /// (none: GLM's `collapsed = Σ pre·X` with this site's pre), by the
    /// map's formulas (engine-map §6) in F64: the sublayer input
    /// `[rows, H]`, pre `[rows, S]` and the join row post | comb
    /// `[rows, S + S²]`.
    fn reference(
        &self,
        x: &[f64],
        previous: Option<&[f64]>,
        h: usize,
        s: usize,
        iterations: usize,
    ) -> [Vec<f64>; 3] {
        let (width, mixes) = (s * h, (2 + s) * s);
        let (epsilon, norm_epsilon) = (f64::from(self.epsilons.0), f64::from(self.epsilons.1));
        let sigmoid = |v: f64| 1.0 / (1.0 + (-v).exp());
        let (mut input, mut pre, mut join) = (Vec::new(), Vec::new(), Vec::new());
        for (row, x) in x.chunks_exact(width).enumerate() {
            let rms =
                1.0 / (x.iter().map(|v| v * v).sum::<f64>() / width as f64 + norm_epsilon).sqrt();
            let m = (0..mixes)
                .map(|j| {
                    let weights = &self.function[j * width..(j + 1) * width];
                    weights
                        .iter()
                        .zip(x)
                        .map(|(w, v)| f64::from(*w) * v)
                        .sum::<f64>()
                        * rms
                })
                .collect::<Vec<_>>();
            let (scale, base) = (
                |i: usize| f64::from(self.scale[i]),
                |i: usize| f64::from(self.base[i]),
            );
            pre.extend((0..s).map(|i| sigmoid(m[i] * scale(0) + base(i)) + epsilon));
            join.extend((0..s).map(|i| 2.0 * sigmoid(m[s + i] * scale(1) + base(s + i))));
            let mut comb = vec![vec![0.0f64; s]; s];
            for (source, values) in comb.iter_mut().enumerate() {
                for (target, value) in values.iter_mut().enumerate() {
                    let at = 2 * s + source * s + target;
                    *value = m[at] * scale(2) + base(at);
                }
                let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let total: f64 = values.iter().map(|v| (v - maximum).exp()).sum();
                for value in values.iter_mut() {
                    *value = (*value - maximum).exp() / total + epsilon;
                }
            }
            let columns = |comb: &mut Vec<Vec<f64>>| {
                for target in 0..s {
                    let sum: f64 = comb.iter().map(|values| values[target]).sum();
                    for values in comb.iter_mut() {
                        values[target] /= sum + epsilon;
                    }
                }
            };
            for sweep in 0..iterations {
                if sweep > 0 {
                    for values in comb.iter_mut() {
                        let sum: f64 = values.iter().sum();
                        for value in values.iter_mut() {
                            *value /= sum + epsilon;
                        }
                    }
                }
                columns(&mut comb);
            }
            join.extend(comb.iter().flatten());
            let carried = match previous {
                Some(previous) => &previous[row * s..(row + 1) * s],
                None => &pre[row * s..(row + 1) * s],
            };
            input.extend((0..h).map(|column| {
                (0..s)
                    .map(|stream| carried[stream] * x[stream * h + column])
                    .sum::<f64>()
            }));
        }
        [input, pre, join]
    }

    /// `stream_mix_sinkhorn` on `device`, mixing by `previous` (P = 1) or
    /// by the site's own pre (none, P = 0): input, pre (empty under P = 0)
    /// and the join row.
    fn native(
        &self,
        device: &seismic::Device,
        x: &[f32],
        previous: Option<&[f32]>,
        rows: usize,
        (h, s, n): (usize, usize, usize),
    ) -> [Vec<f32>; 3] {
        let mixes = (2 + s) * s;
        let carried = usize::from(previous.is_some());
        let site = at_defaults(
            device,
            stream_mix_sinkhorn::native_entry(),
            &[("H", h as u64), ("S", s as u64), ("P", carried as u64)],
        )
        .unwrap()
        .call(stream_mix_sinkhorn::Args {
            hidden: &tensor(device, DType::F32, &[rows, s, h], x),
            previous: &tensor(device, DType::F32, &[rows, carried * s], previous.unwrap_or(&[])),
            function: &tensor(device, DType::F32, &[mixes, s * h], &self.function),
            base: &tensor(device, DType::F32, &[mixes], &self.base),
            scale: &tensor(device, DType::F32, &[3], &self.scale),
            epsilon: self.epsilons.0,
            norm_epsilon: self.epsilons.1,
            iterations: u32::try_from(n).unwrap(),
        })
        .unwrap();
        [
            read(&site.r0, DType::F32),
            read(&site.r1, DType::F32),
            read(&site.r2, DType::F32),
        ]
    }
}

/// The sublayer's projection that the join scatters: `attention_output`'s
/// gated rows `[rows, Q, W]` and bf16 output weight `[H, Q W]`, with `y`
/// its F64 product rounded to bf16 as the entry publishes it (A = bf16).
struct Projection {
    q: usize,
    w: usize,
    gated: Vec<f32>,
    weight: Vec<f32>,
}

impl Projection {
    fn new(h: usize, rows: usize, seed: u32) -> Self {
        let (q, w) = (2, 64);
        Self {
            q,
            w,
            gated: bf16(&pattern(rows * q * w, seed, 1.0)),
            weight: bf16(&pattern(h * q * w, seed + 1, 0.3)),
        }
    }

    fn y(&self, h: usize, rows: usize) -> Vec<f64> {
        let k = self.q * self.w;
        (0..rows * h)
            .map(|at| {
                let (row, column) = (at / h, at % h);
                let sum: f64 = (0..k)
                    .map(|i| {
                        f64::from(self.gated[row * k + i]) * f64::from(self.weight[column * k + i])
                    })
                    .sum();
                f64::from(registry::bf16_round(sum as f32))
            })
            .collect()
    }

    /// The streams `x` joined with `y` through `join` (post | comb):
    /// `X'[d] = sum_s comb[s, d] X[s] + post[d] y`, in F64.
    fn reference(x: &[f64], join: &[f64], y: &[f64], h: usize, s: usize) -> Vec<f64> {
        let (width, row_join) = (s * h, s + s * s);
        let mut joined = Vec::with_capacity(x.len());
        for (row, x) in x.chunks_exact(width).enumerate() {
            let join = &join[row * row_join..(row + 1) * row_join];
            for target in 0..s {
                for column in 0..h {
                    let mixed: f64 = (0..s)
                        .map(|source| join[s + source * s + target] * x[source * h + column])
                        .sum();
                    joined.push(mixed + join[target] * y[row * h + column]);
                }
            }
        }
        joined
    }

    /// `attention_output` on `device` joining into the streams `x` with
    /// the join row `join` (IN = S + S²).
    fn native(
        &self,
        device: &seismic::Device,
        x: &[f32],
        join: &[f32],
        rows: usize,
        h: usize,
        s: usize,
    ) -> Vec<f32> {
        let statics = [
            ("D", h as u64),
            ("SN", s as u64),
            ("IN", (s + s * s) as u64),
            ("Q", self.q as u64),
            ("W", self.w as u64),
            ("OS", 0),
            ("OB", 0),
        ];
        // The statics this backend's native declares.
        let declared = seismic::generated::native_implementation::<attention_output::Entry>(device)
            .unwrap()
            .expect("attention_output has a native on this backend")
            .statics
            .clone();
        let statics = statics
            .iter()
            .filter(|(name, _)| declared.iter().any(|declared| declared == name))
            .fold(
                seismic::NativeSpecialization::new(),
                |spec, (name, value)| spec.with_static(*name, *value),
            );
        let elements = attention_output::Elements {
            OW: seismic::Element::bf16(),
            A: seismic::Element::bf16(),
        };
        let specialization =
            seismic::generated::native_default_specialization::<attention_output::Entry>(
                device,
                &statics,
                &[
                    ("OW", seismic::Element::bf16()),
                    ("A", seismic::Element::bf16()),
                ],
            )
            .unwrap()
            .expect("attention_output joins through the comb on this backend");
        let joined = attention_output::native_for_device_with(device, elements, &specialization)
            .unwrap()
            .call(attention_output::Args {
                hidden: &tensor(device, DType::F32, &[rows, s * h], x),
                inject: &tensor(device, DType::F32, &[rows, s + s * s], join),
                gated: &tensor(device, DType::BF16, &[rows, self.q, self.w], &self.gated),
                output_weight: &tensor(device, DType::BF16, &[h, self.q * self.w], &self.weight),
                output_scale: &tensor(device, DType::F32, &[0], &[]),
                output_bound: &tensor(device, DType::F32, &[0], &[]),
            })
            .unwrap();
        read(&joined.value, DType::F32)
    }
}

fn narrow(values: &[f64]) -> Vec<f32> {
    values.iter().map(|v| *v as f32).collect()
}

/// The previous site's pre before the first site: one-hot on stream 0.
fn entry_pre(rows: usize, s: usize) -> Vec<f64> {
    (0..rows * s)
        .map(|at| if at % s == 0 { 1.0 } else { 0.0 })
        .collect()
}

/// The portable `stream_mix_sinkhorn` gives the formulas' site at a small
/// geometry over two consecutive sites: mixing by the previous site's pre,
/// the second reading the first's (P = 1), and each mixing by its own,
/// carrying none (P = 0).
#[test]
fn portable_sinkhorn_site_matches_the_single_pass_formulas() {
    let module = module();
    let (h, s, n, rows) = (64, 4, V41.2, 3);
    let width = s * h;
    let x = pattern(rows * width, 21, 3.0);
    for carries in [true, false] {
        let mut carried = carries.then(|| narrow(&entry_pre(rows, s)));
        for seed in [31, 41] {
            let site = Sinkhorn::new(h, s, seed);
            let mixes = (2 + s) * s;
            let previous = carried.clone().unwrap_or_default();
            let elements = ElementBindings::new();
            let logical = module
                .entry(
                    module.entry_named("stream_mix_sinkhorn").unwrap(),
                    &elements,
                )
                .unwrap();
            let mut interpreter = Interpreter::new(&logical);
            let tensors = [
                (vec![rows, s, h], x.clone()),
                (vec![rows, previous.len() / rows], previous),
                (vec![mixes, width], site.function.clone()),
                (vec![mixes], site.base.clone()),
                (vec![3], site.scale.clone()),
            ];
            let mut arguments = tensors
                .into_iter()
                .map(|(shape, values)| {
                    Arg::Tensor(interpreter.add_tensor(TensorData::dense(
                        DType::F32,
                        shape,
                        values.iter().map(|v| f64::from(*v)).collect(),
                    )))
                })
                .collect::<Vec<_>>();
            arguments.push(Arg::Scalar(ReferenceScalar::F32(V41_EPSILON.to_bits())));
            arguments.push(Arg::Scalar(ReferenceScalar::F32(
                V41_NORM_EPSILON.to_bits(),
            )));
            arguments.push(Arg::Scalar(ReferenceScalar::U32(u32::try_from(n).unwrap())));
            let outcome = interpreter.run(&arguments).unwrap();
            if let SourceTermination::Failed(failure) = outcome.termination() {
                panic!("stream_mix_sinkhorn failed: {failure}");
            }
            let portable = outcome
                .results()
                .map(|result| {
                    let OutcomeValue::Tensor(tensor) = result.value() else {
                        panic!("tensor result")
                    };
                    (0..tensor.element_count())
                        .map(|i| tensor.read(i).unwrap() as f32)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let x64 = x.iter().map(|v| f64::from(*v)).collect::<Vec<_>>();
            let carried64 = carried
                .as_ref()
                .map(|carried| carried.iter().map(|v| f64::from(*v)).collect::<Vec<_>>());
            let [input, pre, join] = site.reference(&x64, carried64.as_deref(), h, s, n);
            // The site publishes its pre only for a next site to read.
            let pre = if carries { pre } else { Vec::new() };
            for (name, (actual, expected)) in ["input", "pre", "join"]
                .into_iter()
                .zip(portable.iter().zip([&input, &pre, &join]))
            {
                assert_close(
                    &format!("portable carries {carries} seed {seed} {name}"),
                    actual,
                    expected,
                    1e-6,
                    1e-5,
                );
            }
            carried = carries.then(|| portable[1].clone());
        }
    }
}

/// The natives at V4.1's geometry, for decode rows (split across
/// threadgroups, to the last decode row) and prefill rows, over two
/// consecutive sites joined by the comb scatter, the second site reading
/// the streams and pre the first left: each site's input, pre and join row,
/// the joined streams, and the exit's collapse by the last pre, against the
/// formulas over the natives' inputs.
#[test]
fn native_sinkhorn_sites_and_comb_join_match_the_single_pass_formulas() {
    check_sinkhorn_sites(V41, (V41_EPSILON, V41_NORM_EPSILON), true, &[1, 2, 7, 8, 20]);
}

/// Prefill rows in the GPU natives' row tiles (16 rows, two a simdgroup): 9
/// rows, the fewest past the decode rows, one partial tile whose second rows
/// only its first simdgroup holds; 17, a whole tile and a tile of one row.
#[test]
fn native_sinkhorn_prefill_tiles_match_the_single_pass_formulas() {
    check_sinkhorn_sites(V41, (V41_EPSILON, V41_NORM_EPSILON), true, &[9, 17]);
}

/// The natives at GLM-5.3-Flash's geometry, each site mixing by its own pre
/// and carrying none (`SinkhornInput::OwnSite`), and the exit the streams'
/// mean (`ExitSite::Mean`), as
/// [`native_sinkhorn_sites_and_comb_join_match_the_single_pass_formulas`]:
/// one decode row, the last decode row, and the two prefill tilings.
#[test]
fn native_own_site_sinkhorn_sites_and_mean_exit_match_the_formulas() {
    check_sinkhorn_sites(GLM, GLM_EPSILONS, false, &[1, 8, 9, 17]);
}

/// The mean exit at a stream count that is not a power of two, where
/// `1 / S` is no F32 value and a scale by it would round twice: only the
/// summed-then-divided form (`stream_collapse` at P = 0) collapses bit
/// for bit, which this asserts at three streams, over one decode row and
/// both prefill tilings. Its sites also exercise the Sinkhorn comb at an
/// odd S, whose `[S, S]` sweeps no native can tile by powers of two.
#[test]
fn native_mean_exit_collapses_three_streams_exactly() {
    let (h, _, iterations) = GLM;
    check_sinkhorn_sites((h, 3, iterations), GLM_EPSILONS, false, &[1, 9, 17]);
}

/// Two consecutive Sinkhorn sites at `geometry` (H, S, iterations) under
/// `epsilons`, at each of `row_counts`, each joined by the comb scatter and
/// the second reading the streams the first left: with `carries`, each
/// mixing by the previous site's pre (the first by the one-hot entry pre)
/// and the exit collapsing by the last site's; else each mixing by its own,
/// publishing none, and the exit the streams' mean (`stream_collapse` at
/// P = 0, reading no pre). Each site's input, pre and join row and the joined streams
/// are checked against the formulas in F64 over the natives' inputs; the
/// last pre's collapse within the F32 sum's bound, and the mean exactly
/// as F32 sums the streams in order and divides by S.
fn check_sinkhorn_sites(
    geometry: (usize, usize, usize),
    epsilons: (f32, f32),
    carries: bool,
    row_counts: &[usize],
) {
    let (h, s, n) = geometry;
    let width = s * h;
    let sites = [51, 61].map(|seed| Sinkhorn::with_epsilons(h, s, seed, epsilons));
    for &rows in row_counts {
        let mut x = pattern(rows * width, 71 + rows as u32, 2.0);
        for (index, value) in x.iter_mut().enumerate() {
            *value *= 1.0 + (index % width / h) as f32;
        }
        let projections = [Projection::new(h, rows, 81), Projection::new(h, rows, 91)];
        for device in devices() {
            let label = |stage: &str| format!("{:?} M {rows} {stage}", device.backend());
            let (mut streams, mut pre) = (x.clone(), carries.then(|| narrow(&entry_pre(rows, s))));
            for (index, (site, projection)) in sites.iter().zip(&projections).enumerate() {
                let wide =
                    |values: &[f32]| values.iter().map(|v| f64::from(*v)).collect::<Vec<_>>();
                let mut expected =
                    site.reference(&wide(&streams), pre.as_deref().map(wide).as_deref(), h, s, n);
                let actual = site.native(&device, &streams, pre.as_deref(), rows, geometry);
                if !carries {
                    // The site publishes no pre for a next site.
                    expected[1].clear();
                }
                for (name, (actual, expected)) in ["input", "pre", "join"]
                    .into_iter()
                    .zip(actual.iter().zip(&expected))
                {
                    assert_close(
                        &label(&format!("site {index} {name}")),
                        actual,
                        expected,
                        1e-5,
                        1e-5,
                    );
                }
                let y = projection.y(h, rows);
                let joined = Projection::reference(&wide(&streams), &expected[2], &y, h, s);
                let actual_joined = projection.native(&device, &streams, &actual[2], rows, h, s);
                assert_eq!(actual_joined.len(), joined.len());
                for (at, (actual, expected)) in actual_joined.iter().zip(&joined).enumerate() {
                    // The comb's error over the streams, and one bf16
                    // rounding of y landing on its other side: post (< 2)
                    // times an ulp of y (at most 2^-7 |y|).
                    let row = at / width;
                    let streams = &streams[row * width..(row + 1) * width];
                    let magnitude: f64 = (0..s)
                        .map(|source| f64::from(streams[source * h + at % h].abs()))
                        .sum();
                    let bound = 1e-5 * magnitude + 2.0 * y[row * h + at % h].abs() * 2f64.powi(-7);
                    assert!(
                        (f64::from(*actual) - expected).abs() <= 1e-6 + bound,
                        "{}[{at}]: native {actual}, expected {expected}, bound {bound}",
                        label(&format!("site {index} joined"))
                    );
                }
                streams = actual_joined;
                pre = carries.then(|| actual[1].clone());
            }
            let collapsed = collapse_native(&device, &streams, pre.as_deref(), rows, h, s);
            let stream = |row: usize, stream: usize, column: usize| streams[row * width + stream * h + column];
            for (at, actual) in collapsed.iter().enumerate() {
                let (row, column) = (at / h, at % h);
                if let Some(weights) = &pre {
                    let terms = (0..s)
                        .map(|from| f64::from(weights[row * s + from]) * f64::from(stream(row, from, column)))
                        .collect::<Vec<_>>();
                    let (expected, bound) =
                        (terms.iter().sum::<f64>(), gamma(s) * terms.iter().map(|term| term.abs()).sum::<f64>());
                    assert!(
                        (f64::from(*actual) - expected).abs() <= bound,
                        "{}[{at}]: native {actual}, expected {expected}, bound {bound}",
                        label("exit by the last pre")
                    );
                } else {
                    let mean = (0..s).fold(0.0f32, |sum, from| sum + stream(row, from, column)) / s as f32;
                    assert_eq!(actual.to_bits(), mean.to_bits(), "{}[{at}]", label("mean exit"));
                }
            }
        }
    }
}

/// `stream_collapse` on `device`: the streams `[rows, S, H]` collapsed by
/// the last site's `pre [rows, S]` (P = 1), or into their mean without one
/// (P = 0).
fn collapse_native(
    device: &seismic::Device,
    streams: &[f32],
    pre: Option<&[f32]>,
    rows: usize,
    h: usize,
    s: usize,
) -> Vec<f32> {
    let previous = usize::from(pre.is_some());
    let collapsed = at_defaults(
        device,
        magnitude_kernels::stream_collapse::native_entry(),
        &[("H", h as u64), ("S", s as u64), ("P", previous as u64)],
    )
    .unwrap()
    .call(magnitude_kernels::stream_collapse::Args {
        hidden: &tensor(device, DType::F32, &[rows, s, h], streams),
        pre: &tensor(device, DType::F32, &[rows, previous * s], pre.unwrap_or(&[])),
    })
    .unwrap();
    read(&collapsed.value, DType::F32)
}

/// Each value of `actual` against the F64 `expected`: a NaN where it is a
/// NaN, the same infinity where it is one, else within `bound[i]`.
fn assert_ieee_close(name: &str, actual: &[f32], expected: &[f64], bound: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "{name} length");
    for (index, ((actual, expected), bound)) in actual.iter().zip(expected).zip(bound).enumerate() {
        let actual = f64::from(*actual);
        let agrees = if expected.is_nan() {
            actual.is_nan()
        } else if expected.is_infinite() {
            actual == *expected
        } else {
            (actual - expected).abs() <= *bound
        };
        assert!(agrees, "{name}[{index}]: native {actual}, expected {expected}, bound {bound}");
    }
}

/// `stream_join`'s formula in F64 (`scatter`): for each stream d of each
/// row, `X'[d] = (IN > SN: sum_s comb[s, d] X[s], else X[d]) + y (IN = 0)
/// or y inject[d]`, and its terms' magnitude sum.
fn join_reference(
    residual: &[f32],
    y: &[f32],
    inject: &[f32],
    (h, sn, width): (usize, usize, usize),
) -> (Vec<f64>, Vec<f64>) {
    let rows = y.len() / h;
    let (mut joined, mut magnitude) = (Vec::new(), Vec::new());
    for row in 0..rows {
        let x = |s: usize, column: usize| f64::from(residual[row * sn * h + s * h + column]);
        let at = |i: usize| f64::from(inject[row * width + i]);
        for stream in 0..sn {
            for column in 0..h {
                let mut terms = Vec::new();
                if width > sn {
                    terms.extend((0..sn).map(|source| at(sn + source * sn + stream) * x(source, column)));
                } else {
                    terms.push(x(stream, column));
                }
                let value = f64::from(y[row * h + column]);
                terms.push(if width == 0 { value } else { value * at(stream) });
                joined.push(terms.iter().sum());
                magnitude.push(terms.iter().map(|term| term.abs()).sum());
            }
        }
    }
    (joined, magnitude)
}

/// `stream_join` on the host's GPU backend and the CPU against its F64 formula, in every join
/// form (`IN` = 0 of one stream, `IN = SN` and the comb's `IN = SN + SN²`, at
/// one and four streams) and at decode and prefill rows, over adversarial
/// values: sums that cancel to near zero (each row's last column is set so
/// its F64 join is about zero, every term large), NaN and both infinities in
/// the streams, the output and the inject row (a comb of zero times an
/// infinite stream gives NaN, as the formula does), and ordinary values.
/// Finite values agree within the F32 sum's bound ([`gamma`]).
#[test]
fn native_stream_join_matches_its_formula_on_adversarial_values() {
    let h = 256;
    for (sn, width) in [(1, 0), (1, 1), (1, 2), (4, 4), (4, 20)] {
        for rows in [1usize, 3, 70] {
            let mut residual = pattern(rows * sn * h, 101 + sn as u32, 8.0);
            let mut y = pattern(rows * h, 103 + width as u32, 8.0);
            let mut inject = pattern(rows * width, 107, 2.0);
            // The near-zero sums: the last column of each row's stream 0
            // made to cancel, every term of magnitude near 8.
            for row in 0..rows {
                let column = h - 1;
                let x = |s: usize| f64::from(residual[row * sn * h + s * h + column]);
                let at = |i: usize| f64::from(inject[row * width + i]);
                let mixed = if width > sn {
                    (0..sn).map(|source| at(sn + source * sn) * x(source)).sum::<f64>()
                } else {
                    x(0)
                };
                let factor = if width == 0 { 1.0 } else { at(0) };
                if factor.abs() > 1e-3 {
                    y[row * h + column] = (-mixed / factor) as f32;
                }
            }
            // Non-finite values, in the streams, the output and the inject
            // row, on separate columns of row 0.
            residual[3] = f32::NAN;
            residual[5] = f32::INFINITY;
            residual[(sn - 1) * h + 7] = f32::NEG_INFINITY;
            y[9] = f32::INFINITY;
            y[11] = f32::NAN;
            if width > 0 {
                inject[0] = if width > sn { 0.75 } else { f32::INFINITY };
                inject[width - 1] = f32::NAN;
            }
            if width > sn {
                // A zero comb entry against an infinite stream: NaN.
                inject[sn] = 0.0;
            }
            let (expected, magnitude) = join_reference(&residual, &y, &inject, (h, sn, width));
            let bound = magnitude.iter().map(|m| gamma(sn + 1) * m).collect::<Vec<_>>();
            for device in devices() {
                let statics = [("H", h as u64), ("SN", sn as u64), ("IN", width as u64)];
                let joined = at_defaults(
                    &device,
                    magnitude_kernels::stream_join::native_entry(),
                    &statics,
                )
                .unwrap()
                .call(magnitude_kernels::stream_join::Args {
                    residual: &tensor(&device, DType::F32, &[rows, sn * h], &residual),
                    y: &tensor(&device, DType::F32, &[rows, h], &y),
                    inject: &tensor(&device, DType::F32, &[rows, width], &inject),
                })
                .unwrap();
                assert_ieee_close(
                    &format!("{:?} stream_join SN {sn} IN {width} M {rows}", device.backend()),
                    &read(&joined.value, DType::F32),
                    &expected,
                    &bound,
                );
            }
        }
    }
}

/// `stream_collapse` (the exit over Sinkhorn streams) on the host's GPU
/// backend and the CPU against `sum_s pre[s] X[s]` in F64 at V4.1's width, within the F32 sum's
/// bound, at decode and prefill rows.
#[test]
fn native_stream_collapse_matches_its_formula() {
    let (h, s) = (V41.0, V41.1);
    for rows in [1usize, 2, 20] {
        let x = pattern(rows * s * h, 113 + rows as u32, 4.0);
        let pre = pattern(rows * s, 127, 1.0).iter().map(|v| v.abs() + 1e-6).collect::<Vec<_>>();
        let (mut expected, mut bound) = (Vec::new(), Vec::new());
        for row in 0..rows {
            for column in 0..h {
                let terms = (0..s)
                    .map(|stream| f64::from(pre[row * s + stream]) * f64::from(x[row * s * h + stream * h + column]))
                    .collect::<Vec<_>>();
                expected.push(terms.iter().sum::<f64>());
                bound.push(gamma(s) * terms.iter().map(|term| term.abs()).sum::<f64>());
            }
        }
        for device in devices() {
            assert_ieee_close(
                &format!("{:?} stream_collapse M {rows}", device.backend()),
                &collapse_native(&device, &x, Some(&pre), rows, h, s),
                &expected,
                &bound,
            );
        }
    }
}

/// The exit's mean (`stream_collapse` at P = 0) is exactly `Σ_s X[s] / S`
/// as F32 sums the streams in order and divides by S, bit for bit, for
/// stream counts that are powers of two and those that are not (where 1/S
/// is inexact, so a collapse weighted by 1/S would differ).
#[test]
fn native_stream_mean_is_exact_for_any_stream_count() {
    let h = 257;
    for s in [1usize, 2, 3, 4, 5, 6, 7, 8, 12] {
        for rows in [1usize, 3] {
            let x = pattern(rows * s * h, 211 + (rows * s) as u32, 4.0);
            for device in devices() {
                let mean = collapse_native(&device, &x, None, rows, h, s);
                for (at, actual) in mean.iter().enumerate() {
                    let (row, column) = (at / h, at % h);
                    let expected = (0..s).fold(0.0f32, |sum, from| sum + x[row * s * h + from * h + column]) / s as f32;
                    assert_eq!(
                        actual.to_bits(),
                        expected.to_bits(),
                        "{:?} mean S {s} M {rows}[{at}]",
                        device.backend()
                    );
                }
            }
        }
    }
}

/// Metal's or CUDA's `stream_mix_sinkhorn` at V4.1's geometry (S 4, H 5120)
/// at decode rows: one row at 20 Sinkhorn iterations and at none (their
/// difference the normalizations' serial cost), and the last decode row
/// count at 20; and at V4.1's prefill shape, a 2048-token prompt's rows at
/// 20. Each call reads one of 32 function matrices (63 MB), so they stream
/// from memory as a step's layers do. Interleaved round by round, best batch
/// of each.
#[test]
#[ignore = "timing"]
fn native_sinkhorn_split_timing() {
    use seismic::BackendName::{Cuda, Metal};
    let Some(device) = seismic_native_tests::devices(&[Cuda, Metal]).into_iter().next() else {
        return;
    };
    let (h, s, n) = V41;
    let mixes = (2 + s) * s;
    let functions = (0..32u32)
        .map(|copy| tensor(&device, DType::F32, &[mixes, s * h], &pattern(mixes * s * h, 300 + copy, 0.01)))
        .collect::<Vec<_>>();
    let base = tensor(&device, DType::F32, &[mixes], &pattern(mixes, 52, 1.0));
    let scale = tensor(&device, DType::F32, &[3], &[0.6, 0.9, 1.7]);
    let specialization = seismic::generated::native_implementation::<stream_mix_sinkhorn::Entry>(&device)
        .unwrap()
        .unwrap()
        .default_specialization(
            &seismic::NativeSpecialization::new()
                .with_static("H", h as u64)
                .with_static("S", s as u64)
                .with_static("P", 1),
        )
        .unwrap();
    let kernel = stream_mix_sinkhorn::native_for_device(&device, &specialization).unwrap();
    // (rows, iterations)
    let cases = [(1usize, n), (1, 0), (8, n), (2048, n)];
    let rows_of = |rows: usize| {
        (
            tensor(&device, DType::F32, &[rows, s, h], &pattern(rows * s * h, 71, 2.0)),
            tensor(&device, DType::F32, &[rows, s], &narrow(&entry_pre(rows, s))),
            seismic::Tensor::zeros(&device, seismic::Element::f32(), &[rows as u64, h as u64]).unwrap(),
            seismic::Tensor::zeros(&device, seismic::Element::f32(), &[rows as u64, s as u64]).unwrap(),
            seismic::Tensor::zeros(&device, seismic::Element::f32(), &[rows as u64, (s + s * s) as u64]).unwrap(),
        )
    };
    let mut tensors = cases.iter().map(|&(rows, _)| rows_of(rows)).collect::<Vec<_>>();
    const CALLS: u32 = 64;
    let mut best = vec![std::time::Duration::MAX; cases.len()];
    for _ in 0..10 {
        for ((&(_, iterations), (hidden, previous, input, pre, join)), best) in
            cases.iter().zip(&mut tensors).zip(&mut best)
        {
            let mut batch = seismic::NativeTensorBatch::new(&device);
            for call in 0..CALLS as usize {
                batch
                    .push(
                        &kernel,
                        stream_mix_sinkhorn::Args {
                            hidden: &*hidden,
                            previous: &*previous,
                            function: &functions[call % functions.len()],
                            base: &base,
                            scale: &scale,
                            epsilon: V41_EPSILON,
                            norm_epsilon: V41_NORM_EPSILON,
                            iterations: u32::try_from(iterations).unwrap(),
                        },
                        stream_mix_sinkhorn::OutputArgs { r0: &mut *input, r1: &mut *pre, r2: &mut *join },
                    )
                    .unwrap();
            }
            let started = std::time::Instant::now();
            batch.submit().unwrap().wait().unwrap();
            *best = (*best).min(started.elapsed() / CALLS);
        }
    }
    let function_bytes = (mixes * s * h * 4) as f64;
    for ((rows, iterations), best) in cases.iter().zip(&best) {
        println!(
            "{:?} M {rows} iterations {iterations:>2}: {:>7.1} us, {:>6.1} GB/s of the function matrix",
            device.backend(),
            best.as_secs_f64() * 1e6,
            function_bytes / best.as_secs_f64() / 1e9
        );
    }
}
