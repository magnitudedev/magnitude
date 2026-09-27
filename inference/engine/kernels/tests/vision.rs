//! The vision tower entries (`vision_*` in vision.seismic) on the host's
//! accelerator (Metal on macOS, else CUDA; `SEISMIC_TEST_BACKEND=vulkan`
//! selects Vulkan) and on the CPU, checked against their portable bodies run
//! by the Seismic interpreter at small shapes. Weights are f16, bias and norm
//! vectors f32 (a projector file's elements, kept resident as stored); the
//! activation A is f16 and the residual stream F32. The GPU forms run the
//! forms their `where` limits admit; the CPU runs every form.
//!
//! The native entries reassociate the F32 sums, round the stem's pixels and
//! the attention probabilities to half for the matrix units, and may land on
//! the other side of an A rounding than the portable body; each such flip
//! moves a published intermediate by one A ulp. The bounds are relative to
//! the result's RMS.

use magnitude_kernels::{
    vision_attention, vision_clamp, vision_linear, vision_norm, vision_patch_stem, vision_pool,
    vision_position,
};
use seismic::{Device, Element, NativeSpecialization, Tensor};
use seismic_lang::{
    checked::{check_source, CheckedModule, SourceFile},
    entry::ElementBindings,
    failure::SourceTermination,
    interp::{Arg, Interpreter, OracleOutcome, OutcomeValue, TensorData},
    reference_math::ReferenceScalar,
    registry,
    types::DType,
};

// ---------------------------------------------------------------------------
// Host values.

fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits >> 15 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let mantissa = f32::from(bits & 0x3ff);
    sign * if exponent == 0 {
        mantissa * 2f32.powi(-24)
    } else {
        (1.0 + mantissa / 1024.0) * 2f32.powi(exponent - 15)
    }
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(
            seed.wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407),
        )
    }
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }
    fn symmetric(&mut self) -> f32 {
        ((self.next() >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
    fn values(&mut self, count: usize, scale: f32) -> Vec<f32> {
        (0..count).map(|_| self.symmetric() * scale).collect()
    }
}

/// A host tensor: its element, shape and logical values (exact in the
/// element).
struct Host {
    dtype: DType,
    shape: Vec<usize>,
    values: Vec<f32>,
}

impl Host {
    fn new(dtype: DType, shape: &[usize], values: Vec<f32>) -> Self {
        assert_eq!(shape.iter().product::<usize>(), values.len());
        let values = values
            .into_iter()
            .map(|value| {
                if dtype == DType::F16 {
                    f16_to_f32(registry::f16_bits(value))
                } else {
                    value
                }
            })
            .collect();
        Self {
            dtype,
            shape: shape.to_vec(),
            values,
        }
    }
    fn random(dtype: DType, shape: &[usize], scale: f32, rng: &mut Rng) -> Self {
        let values = rng.values(shape.iter().product(), scale);
        Self::new(dtype, shape, values)
    }
    fn ints(shape: &[usize], values: impl IntoIterator<Item = usize>) -> Self {
        Self::new(DType::I32, shape, values.into_iter().map(|v| v as f32).collect())
    }
    /// An optional operand of `count` rows (0 or 1) of `row` shape.
    fn rows(dtype: DType, count: usize, row: &[usize], scale: f32, offset: f32, rng: &mut Rng) -> Self {
        let shape = std::iter::once(count).chain(row.iter().copied()).collect::<Vec<_>>();
        let values = rng
            .values(shape.iter().product(), scale)
            .into_iter()
            .map(|value| value + offset)
            .collect();
        Self::new(dtype, &shape, values)
    }
    fn element(&self) -> Element {
        Element::dense(self.dtype)
    }
    fn tensor(&self, device: &Device) -> Tensor {
        let bytes = self
            .values
            .iter()
            .flat_map(|value| match self.dtype {
                DType::F16 => registry::f16_bits(*value).to_le_bytes().to_vec(),
                DType::F32 => value.to_le_bytes().to_vec(),
                DType::I32 => (*value as i32).to_le_bytes().to_vec(),
                other => panic!("unsupported host dtype {other:?}"),
            })
            .collect::<Vec<_>>();
        let shape = self
            .shape
            .iter()
            .map(|extent| *extent as u64)
            .collect::<Vec<_>>();
        Tensor::from_host(device, Element::dense(self.dtype), &shape, &bytes).unwrap()
    }
    fn oracle(&self) -> TensorData {
        TensorData::dense(
            self.dtype,
            self.shape.clone(),
            self.values.iter().map(|v| f64::from(*v)).collect(),
        )
    }
}

/// The values of an f32 or f16 result.
fn read_values(tensor: &Tensor) -> Vec<f32> {
    let bytes = tensor.read_to_host().unwrap();
    if tensor.element() == Element::f16() {
        bytes
            .chunks_exact(2)
            .map(|b| f16_to_f32(u16::from_le_bytes(b.try_into().unwrap())))
            .collect()
    } else {
        bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Portable bodies.

fn module() -> CheckedModule {
    let mut sources = seismic_std::sources();
    for (path, text) in [
        ("vision.seismic", include_str!("../kernels/vision.seismic")),
        ("attention.seismic", include_str!("../kernels/attention.seismic")),
    ] {
        sources.push(SourceFile {
            path: path.into(),
            text: text.into(),
        });
    }
    check_source(sources).unwrap()
}

enum Input<'a> {
    Tensor(&'a Host),
    F32(f32),
    I32(i32),
}

fn interpret(
    module: &CheckedModule,
    name: &str,
    bindings: &[(&str, DType)],
    inputs: &[Input<'_>],
) -> Vec<f32> {
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
        .iter()
        .map(|input| match input {
            Input::Tensor(host) => Arg::Tensor(interpreter.add_tensor(host.oracle())),
            Input::F32(value) => Arg::Scalar(ReferenceScalar::F32(value.to_bits())),
            Input::I32(value) => Arg::Scalar(ReferenceScalar::I32(*value)),
        })
        .collect::<Vec<_>>();
    let outcome: OracleOutcome = interpreter
        .run(&arguments)
        .unwrap_or_else(|error| panic!("{name}: {error}"));
    if let SourceTermination::Failed(failure) = outcome.termination() {
        panic!("{name} failed: {failure}");
    }
    let result = outcome.results().next().unwrap();
    let OutcomeValue::Tensor(tensor) = result.value() else {
        panic!("tensor result")
    };
    (0..tensor.element_count())
        .map(|i| tensor.read(i).unwrap() as f32)
        .collect()
}

/// Error of `actual` against `expected` relative to the RMS of `expected`:
/// the maximum and the mean must stay within the bounds.
fn relative_error(label: &str, actual: &[f32], expected: &[f32], maximum: f32, mean: f32) {
    assert_eq!(actual.len(), expected.len(), "{label}: result length");
    let rms = (expected.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / expected.len() as f64)
        .sqrt();
    let errors = actual
        .iter()
        .zip(expected)
        .map(|(a, e)| f64::from((a - e).abs()) / rms)
        .collect::<Vec<_>>();
    let worst = errors.iter().copied().fold(0.0, f64::max);
    let average = errors.iter().sum::<f64>() / errors.len() as f64;
    eprintln!("{label}: rms {rms:.4}, max error {worst:.2e} rms, mean {average:.2e} rms");
    assert!(
        actual.iter().all(|value| value.is_finite()),
        "{label}: non-finite result"
    );
    assert!(
        worst <= f64::from(maximum) && average <= f64::from(mean),
        "{label}: max {worst:.2e} (bound {maximum:.0e}), mean {average:.2e} (bound {mean:.0e})"
    );
}

/// The host's accelerator (Metal on macOS, else CUDA;
/// `SEISMIC_TEST_BACKEND=vulkan` selects Vulkan; `cpu` runs the CPU only).
fn accelerator(catalog: &seismic::DeviceCatalog) -> Option<Device> {
    let backend = match std::env::var("SEISMIC_TEST_BACKEND").ok().as_deref() {
        Some("vulkan") => seismic::BackendName::Vulkan,
        Some("cpu") => return None,
        Some(other) => panic!("SEISMIC_TEST_BACKEND={other}: only vulkan or cpu is selectable"),
        None if cfg!(target_os = "macos") => seismic::BackendName::Metal,
        None => seismic::BackendName::Cuda,
    };
    Some(catalog.open_backend(backend).unwrap())
}

/// The accelerator and the CPU.
fn devices() -> Vec<Device> {
    let catalog = seismic::DeviceCatalog::discover().unwrap();
    accelerator(&catalog)
        .into_iter()
        .chain([catalog.open_backend(seismic::BackendName::Cpu).unwrap()])
        .collect()
}

fn is_cpu(device: &Device) -> bool {
    device.backend() == seismic::BackendName::Cpu
}

/// The static dimensions of a GPU form; the CPU forms declare none.
fn specialization_on(device: &Device, statics: &[(&str, usize)]) -> NativeSpecialization {
    if is_cpu(device) {
        return NativeSpecialization::new();
    }
    statics.iter().fold(
        NativeSpecialization::new(),
        |specialization, (name, value)| specialization.with_static(*name, *value as u64),
    )
}

const EPSILON: f32 = 1e-6;

// ---------------------------------------------------------------------------
// Tests.

/// The two-frame biased stem (the Qwen form) and single-frame unbiased ones
/// with F32 weights (Gemma 4's 16-pixel patches, Muse Glimmer's 14-pixel
/// ones, whose depth 588 is no whole GEMM step).
#[test]
fn patch_stem_matches_its_portable_body() {
    let module = module();
    let (rows, channels, hidden, table_rows) = (5usize, 3usize, 64usize, 49usize);
    for device in devices() {
        for (frames, biased, patch, weights) in
            [(2usize, true, 16usize, DType::F16), (1, false, 16, DType::F32), (1, false, 14, DType::F32)]
        {
            let (s, nb) = (frames - 1, usize::from(biased));
            let mut rng = Rng::new(21);
            let pixels = Host::random(DType::F32, &[rows, channels, frames, patch, patch], 1.0, &mut rng);
            let frame = Host::random(weights, &[hidden, channels, patch, patch], 0.05, &mut rng);
            let next = Host::random(weights, &[s, hidden, channels, patch, patch], 0.05, &mut rng);
            let bias = Host::rows(DType::F32, nb, &[hidden], 0.1, 0.0, &mut rng);
            let table = Host::random(DType::F32, &[table_rows, hidden], 0.5, &mut rng);
            let indices = Host::ints(&[rows, 4], (0..rows * 4).map(|i| i * 5 % table_rows));
            let coefficients = Host::new(
                DType::F32,
                &[rows, 4],
                (0..rows * 4).map(|_| (rng.symmetric() + 1.0) / 2.0).collect(),
            );
            let expected = interpret(
                &module,
                "vision_patch_stem",
                &[("W0", weights), ("W1", weights), ("B", DType::F32), ("PE", DType::F32)],
                &[
                    Input::Tensor(&pixels),
                    Input::Tensor(&frame),
                    Input::Tensor(&next),
                    Input::Tensor(&bias),
                    Input::Tensor(&table),
                    Input::Tensor(&indices),
                    Input::Tensor(&coefficients),
                ],
            );
            let kernel = vision_patch_stem::native_for_device_with(
                &device,
                vision_patch_stem::Elements {
                    W0: frame.element(),
                    W1: next.element(),
                    B: bias.element(),
                    PE: table.element(),
                },
                &specialization_on(
                    &device,
                    &[("C", channels), ("S", s), ("P", patch), ("H", hidden), ("NB", nb)],
                ),
            )
            .unwrap();
            let t = |host: &Host| host.tensor(&device);
            let result = kernel
                .call(vision_patch_stem::Args {
                    pixels: &t(&pixels),
                    frame_weight: &t(&frame),
                    next_frame_weight: &t(&next),
                    bias: &t(&bias),
                    table: &t(&table),
                    indices: &t(&indices),
                    coefficients: &t(&coefficients),
                })
                .unwrap()
                .value;
            relative_error(
                &format!("vision_patch_stem {:?} S{s} NB{nb} P{patch} {weights:?}", device.backend()),
                &read_values(&result),
                &expected,
                2e-3,
                2e-4,
            );
        }
    }
}

/// Layer norms and RMS norms, weighted or not, of single rows and of cells
/// of four rows laid side by side or interleaved, gathered or in order.
#[test]
fn norm_matches_its_portable_body() {
    let module = module();
    let (cells, h) = (3usize, 96usize);
    // (members, centered, weighted, biased, gathered, interleave, output)
    let forms = [
        (1usize, true, true, true, false, false, DType::F16),
        (1, false, true, false, false, false, DType::F32),
        (1, false, false, false, false, false, DType::F32),
        (4, true, true, true, false, false, DType::F16),
        (4, false, true, false, true, true, DType::F16),
    ];
    for device in devices() {
        for (g, centered, weighted, biased, gathered, interleave, output) in forms {
            let (nw, nb, no) = (usize::from(weighted), usize::from(biased), usize::from(gathered));
            let mut rng = Rng::new(22);
            let source = Host::random(DType::F32, &[cells, g, h], 2.0, &mut rng);
            let weight = Host::rows(DType::F32, nw, &[h], 0.5, 1.0, &mut rng);
            let bias = Host::rows(DType::F32, nb, &[h], 0.1, 0.0, &mut rng);
            let rows = cells * g;
            let order = Host::ints(&[no, rows], (0..no * rows).map(|r| (r * 7 + 3) % rows));
            let expected = interpret(
                &module,
                "vision_norm",
                &[("NWE", DType::F32), ("NBE", DType::F32), ("Y", output)],
                &[
                    Input::Tensor(&source),
                    Input::Tensor(&weight),
                    Input::Tensor(&bias),
                    Input::Tensor(&order),
                    Input::F32(EPSILON),
                    Input::I32(i32::from(centered)),
                    Input::I32(i32::from(interleave)),
                ],
            );
            let kernel = vision_norm::native_for_device_with(
                &device,
                vision_norm::Elements {
                    NWE: weight.element(),
                    NBE: bias.element(),
                    Y: Element::dense(output),
                },
                &specialization_on(&device, &[("G", g), ("H", h), ("NW", nw), ("NB", nb), ("NO", no)]),
            )
            .unwrap();
            let t = |host: &Host| host.tensor(&device);
            let result = kernel
                .call(vision_norm::Args {
                    source: &t(&source),
                    weight: &t(&weight),
                    bias: &t(&bias),
                    order: &t(&order),
                    epsilon: EPSILON,
                    centered: i32::from(centered),
                    interleave: i32::from(interleave),
                })
                .unwrap()
                .value;
            relative_error(
                &format!("vision_norm {:?} G{g} centered {centered} NW{nw} NB{nb} NO{no}", device.backend()),
                &read_values(&result),
                &expected,
                2e-3,
                2e-4,
            );
        }
    }
}

/// Projections with their epilogues: the residual form (F32), the three
/// activations (A), the gated and the clamped forms, and F32 weights over a
/// depth that is no whole GEMM step.
#[test]
fn linear_matches_its_portable_body() {
    let module = module();
    let (m, n) = (70usize, 96usize);
    // (biased, residual, gated, clamped, activation, output, weights, depth)
    let forms = [
        (true, true, false, false, 0, DType::F32, DType::F16, 128usize),
        (true, false, false, false, 0, DType::F16, DType::F16, 128),
        (true, false, false, false, 1, DType::F16, DType::F16, 128),
        (true, false, false, false, 2, DType::F16, DType::F16, 128),
        (false, false, false, false, 3, DType::F32, DType::F16, 128),
        (false, false, true, false, 0, DType::F16, DType::F16, 128),
        (true, true, false, true, 0, DType::F32, DType::F16, 128),
        (true, false, false, false, 0, DType::F16, DType::F32, 136),
    ];
    for device in devices() {
        for (biased, residual, gated, clamped, activation, output, weights, k) in forms {
            let (nb, nr, ng, nc) = (
                usize::from(biased),
                usize::from(residual),
                usize::from(gated),
                usize::from(clamped),
            );
            let mut rng = Rng::new(23);
            let x = Host::random(DType::F16, &[m, k], 1.0, &mut rng);
            let weight = Host::random(weights, &[n, k], 2.0 * 1.7 / (k as f32).sqrt(), &mut rng);
            let bias = Host::rows(DType::F32, nb, &[n], 0.2, 0.0, &mut rng);
            let residuals = Host::rows(DType::F32, nr, &[m, n], 1.0, 0.0, &mut rng);
            let gate = Host::rows(DType::F16, ng, &[m, n], 1.0, 0.0, &mut rng);
            let minimum = Host::new(DType::F32, &[nc], vec![-1.5; nc]);
            let maximum = Host::new(DType::F32, &[nc], vec![1.0; nc]);
            let expected = interpret(
                &module,
                "vision_linear",
                &[("A", DType::F16), ("W", weights), ("B", DType::F32), ("Y", output)],
                &[
                    Input::Tensor(&x),
                    Input::Tensor(&weight),
                    Input::Tensor(&bias),
                    Input::Tensor(&residuals),
                    Input::Tensor(&gate),
                    Input::Tensor(&minimum),
                    Input::Tensor(&maximum),
                    Input::I32(activation),
                ],
            );
            let kernel = vision_linear::native_for_device_with(
                &device,
                vision_linear::Elements {
                    A: Element::f16(),
                    W: weight.element(),
                    B: bias.element(),
                    Y: Element::dense(output),
                },
                &specialization_on(
                    &device,
                    &[("N", n), ("K", k), ("NB", nb), ("NR", nr), ("NG", ng), ("NC", nc)],
                ),
            )
            .unwrap();
            let t = |host: &Host| host.tensor(&device);
            let result = kernel
                .call(vision_linear::Args {
                    x: &t(&x),
                    weight: &t(&weight),
                    bias: &t(&bias),
                    residual: &t(&residuals),
                    gate: &t(&gate),
                    minimum: &t(&minimum),
                    maximum: &t(&maximum),
                    activation,
                })
                .unwrap()
                .value;
            relative_error(
                &format!(
                    "vision_linear {:?} NB{nb} NR{nr} NG{ng} NC{nc} activation {activation} {weights:?} K{k}",
                    device.backend()
                ),
                &read_values(&result),
                &expected,
                3e-3,
                3e-4,
            );
        }
    }
}

#[test]
fn clamp_matches_its_portable_body() {
    let module = module();
    let (m, n) = (5usize, 300usize);
    for device in devices() {
        let mut rng = Rng::new(24);
        let x = Host::random(DType::F16, &[m, n], 3.0, &mut rng);
        let minimum = Host::new(DType::F32, &[1], vec![-2.0]);
        let maximum = Host::new(DType::F32, &[1], vec![1.25]);
        let expected = interpret(
            &module,
            "vision_clamp",
            &[("A", DType::F16)],
            &[Input::Tensor(&x), Input::Tensor(&minimum), Input::Tensor(&maximum)],
        );
        let kernel = vision_clamp::native_for_device_with(
            &device,
            vision_clamp::Elements { A: Element::f16() },
            &NativeSpecialization::new(),
        )
        .unwrap();
        let t = |host: &Host| host.tensor(&device);
        let result = kernel
            .call(vision_clamp::Args {
                x: &t(&x),
                minimum: &t(&minimum),
                maximum: &t(&maximum),
            })
            .unwrap()
            .value;
        assert_eq!(read_values(&result), expected, "vision_clamp {:?}", device.backend());
    }
}

/// The rotary attention of 100 rows (a partial query tile, a partial key
/// tile): the Qwen form; head norms, a weightless value norm and the unit
/// scale at a head width of 72 (Gemma 4's large towers); windows of 48 rows,
/// so a query tile spans two windows and starts its walk past row 0, at head
/// widths 64 and 96 (Muse Glimmer).
#[test]
fn attention_matches_its_portable_body() {
    let module = module();
    let rows = 100usize;
    // (heads, quarter, head norms, value norm, windows, unit scale)
    let forms = [
        (2usize, 16usize, false, false, false, false),
        (2, 18, true, true, false, true),
        (1, 16, false, false, true, false),
        (1, 24, false, false, true, false),
    ];
    for device in devices() {
        for (heads, quarter, head_norms, value_norm, windows, unit_scale) in forms {
            let width = 4 * quarter;
            let (nq, nv, ws) = (usize::from(head_norms), usize::from(value_norm), usize::from(windows));
            let mut rng = Rng::new(25);
            let shape = [rows, heads, width];
            let query = Host::random(DType::F16, &shape, 2.0, &mut rng);
            let key = Host::random(DType::F16, &shape, 2.0, &mut rng);
            let value = Host::random(DType::F16, &shape, 1.0, &mut rng);
            let query_norm = Host::rows(DType::F32, nq, &[width], 0.3, 1.0, &mut rng);
            let key_norm = Host::rows(DType::F32, nq, &[width], 0.3, 1.0, &mut rng);
            let value_norm = Host::new(DType::F32, &[nv, width], vec![1.0; nv * width]);
            let side = 7;
            let coordinates = Host::ints(&[rows, 2], (0..rows).flat_map(|row| [row / side + 1, row % side + 1]));
            // Windows of 48 rows (the last one partial).
            let spans = Host::ints(
                &[ws, rows, 2],
                (0..ws * rows).flat_map(|row| [row / 48 * 48, (row / 48 * 48 + 48).min(rows)]),
            );
            let log_base = 100f32.ln();
            let expected = interpret(
                &module,
                "vision_attention",
                &[("A", DType::F16)],
                &[
                    Input::Tensor(&query),
                    Input::Tensor(&key),
                    Input::Tensor(&value),
                    Input::Tensor(&query_norm),
                    Input::Tensor(&key_norm),
                    Input::Tensor(&value_norm),
                    Input::Tensor(&coordinates),
                    Input::Tensor(&spans),
                    Input::F32(log_base),
                    Input::F32(EPSILON),
                    Input::I32(i32::from(unit_scale)),
                ],
            );
            let kernel = vision_attention::native_for_device_with(
                &device,
                vision_attention::Elements { A: Element::f16() },
                &specialization_on(
                    &device,
                    &[("H", heads), ("P", quarter), ("NQ", nq), ("NV", nv), ("WS", ws)],
                ),
            )
            .unwrap();
            let t = |host: &Host| host.tensor(&device);
            let result = kernel
                .call(vision_attention::Args {
                    query: &t(&query),
                    key: &t(&key),
                    value: &t(&value),
                    query_norm: &t(&query_norm),
                    key_norm: &t(&key_norm),
                    value_norm: &t(&value_norm),
                    coordinates: &t(&coordinates),
                    spans: &t(&spans),
                    log_base,
                    epsilon: EPSILON,
                    unit_scale: i32::from(unit_scale),
                })
                .unwrap()
                .value;
            relative_error(
                &format!(
                    "vision_attention {:?} W{width} NQ{nq} NV{nv} WS{ws} unit {unit_scale}",
                    device.backend()
                ),
                &read_values(&result),
                &expected,
                5e-3,
                5e-4,
            );
        }
    }
}

#[test]
fn pool_matches_its_portable_body() {
    let module = module();
    let (cells, g, h) = (4usize, 9usize, 80usize);
    for device in devices() {
        for ns in [0usize, 1] {
            let mut rng = Rng::new(26);
            let source = Host::random(DType::F32, &[cells, g, h], 2.0, &mut rng);
            let standard_bias = Host::rows(DType::F32, ns, &[h], 0.2, 0.0, &mut rng);
            let standard_scale = Host::rows(DType::F32, ns, &[h], 0.2, 1.0, &mut rng);
            let (weight, scale) = (1.0 / g as f32, 8.0f32);
            let expected = interpret(
                &module,
                "vision_pool",
                &[("SB", DType::F32), ("SS", DType::F32)],
                &[
                    Input::Tensor(&source),
                    Input::Tensor(&standard_bias),
                    Input::Tensor(&standard_scale),
                    Input::F32(weight),
                    Input::F32(scale),
                ],
            );
            let kernel = vision_pool::native_for_device_with(
                &device,
                vision_pool::Elements {
                    SB: Element::f32(),
                    SS: Element::f32(),
                },
                &specialization_on(&device, &[("G", g), ("H", h), ("NS", ns)]),
            )
            .unwrap();
            let t = |host: &Host| host.tensor(&device);
            let result = kernel
                .call(vision_pool::Args {
                    source: &t(&source),
                    standard_bias: &t(&standard_bias),
                    standard_scale: &t(&standard_scale),
                    weight,
                    scale,
                })
                .unwrap()
                .value;
            relative_error(
                &format!("vision_pool {:?} NS{ns}", device.backend()),
                &read_values(&result),
                &expected,
                1e-6,
                1e-7,
            );
        }
    }
}

#[test]
fn position_matches_its_portable_body() {
    let module = module();
    let (rows, h, table_rows) = (6usize, 80usize, 25usize);
    for device in devices() {
        let mut rng = Rng::new(27);
        let source = Host::random(DType::F32, &[rows, h], 2.0, &mut rng);
        let table = Host::random(DType::F16, &[table_rows, h], 0.5, &mut rng);
        let indices = Host::ints(&[rows, 4], (0..rows * 4).map(|i| i * 3 % table_rows));
        let coefficients = Host::new(
            DType::F32,
            &[rows, 4],
            (0..rows * 4).map(|_| (rng.symmetric() + 1.0) / 2.0).collect(),
        );
        let expected = interpret(
            &module,
            "vision_position",
            &[("PE", DType::F16)],
            &[
                Input::Tensor(&source),
                Input::Tensor(&table),
                Input::Tensor(&indices),
                Input::Tensor(&coefficients),
            ],
        );
        let kernel = vision_position::native_for_device_with(
            &device,
            vision_position::Elements { PE: table.element() },
            &specialization_on(&device, &[("H", h)]),
        )
        .unwrap();
        let t = |host: &Host| host.tensor(&device);
        let result = kernel
            .call(vision_position::Args {
                source: &t(&source),
                table: &t(&table),
                indices: &t(&indices),
                coefficients: &t(&coefficients),
            })
            .unwrap()
            .value;
        relative_error(
            &format!("vision_position {:?}", device.backend()),
            &read_values(&result),
            &expected,
            1e-6,
            1e-7,
        );
    }
}
