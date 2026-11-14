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

mod common;
mod random;

use random::Rng;

use magnitude_kernels::{
    vision_attention, vision_clamp, vision_gated_convolution, vision_linear, vision_norm,
    vision_patch_stem, vision_pool, vision_position,
};
use seismic::{Device, Element, NativeSpecialization, Tensor};
use seismic_lang::{
    checked::{CheckedModule, SourceFile},
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

impl Rng {
    fn values(&mut self, count: usize, scale: f32) -> Vec<f32> {
        (0..count).map(|_| self.symmetric() * scale).collect()
    }
}

/// A host tensor: its element (F32, F16, BF16 or I32), shape and logical
/// values (exact in the element).
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
                match dtype {
                    DType::F16 => f16_to_f32(registry::f16_bits(value)),
                    DType::BF16 => registry::bf16_round(value),
                    _ => value,
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
        Self::new(
            DType::I32,
            shape,
            values.into_iter().map(|v| v as f32).collect(),
        )
    }
    /// An optional operand of `count` rows (0 or 1) of `row` shape.
    fn rows(
        dtype: DType,
        count: usize,
        row: &[usize],
        scale: f32,
        offset: f32,
        rng: &mut Rng,
    ) -> Self {
        let shape = std::iter::once(count)
            .chain(row.iter().copied())
            .collect::<Vec<_>>();
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
                // Exact: the value is already BF16's (`Host::new`).
                DType::BF16 => ((value.to_bits() >> 16) as u16).to_le_bytes().to_vec(),
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
        (
            "attention.seismic",
            include_str!("../kernels/attention.seismic"),
        ),
        (
            "functions.seismic",
            include_str!("../kernels/functions.seismic"),
        ),
        ("rounding.seismic", include_str!("../kernels/rounding.seismic")),
    ] {
        sources.push(SourceFile {
            path: path.into(),
            text: text.into(),
        });
    }
    crate::common::check(sources)
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

/// The host's accelerator (Metal on macOS, else CUDA) and the CPU; a
/// requested backend (Vulkan or the CPU among them) alone
/// ([`seismic_native_tests::first_and_cpu`]).
fn devices() -> Vec<Device> {
    use seismic::BackendName::{Cpu, Cuda, Metal, Vulkan};
    let accelerator = if cfg!(target_os = "macos") { Metal } else { Cuda };
    seismic_native_tests::first_and_cpu(&[accelerator], &[Metal, Cuda, Vulkan, Cpu])
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

/// The two-frame biased stem (the Qwen form), single-frame unbiased ones
/// with F32 weights (Gemma 4's 16-pixel patches, Muse Glimmer's 14-pixel
/// ones, whose depth 588 is no whole GEMM step), the two-frame unbiased
/// stem without a position table (LFM3-VL), and the two-frame biased
/// 14-pixel stem with BF16 weights (GLM-5.3 Flash), whose 16-bit rows of
/// 588 leave 16-byte alignment past the first, at a width of two of CUDA's
/// 128-column tiles.
#[test]
fn patch_stem_matches_its_portable_body() {
    let module = module();
    let (rows, channels, table_rows) = (5usize, 3usize, 49usize);
    for device in devices() {
        for (frames, biased, patch, weights, positioned, hidden) in [
            (2usize, true, 16usize, DType::F16, true, 64usize),
            (1, false, 16, DType::F32, true, 64),
            (1, false, 14, DType::F32, true, 64),
            (2, false, 16, DType::F16, false, 64),
            (2, true, 14, DType::BF16, true, 256),
        ] {
            let (s, nb, nt) = (frames - 1, usize::from(biased), usize::from(positioned));
            let table_rows = table_rows * nt;
            let mut rng = Rng::new(21);
            let pixels = Host::random(
                DType::F32,
                &[rows, channels, frames, patch, patch],
                1.0,
                &mut rng,
            );
            let frame = Host::random(weights, &[hidden, channels, patch, patch], 0.05, &mut rng);
            let next = Host::random(
                weights,
                &[s, hidden, channels, patch, patch],
                0.05,
                &mut rng,
            );
            let bias = Host::rows(DType::F32, nb, &[hidden], 0.1, 0.0, &mut rng);
            let table = Host::random(DType::F32, &[table_rows, hidden], 0.5, &mut rng);
            let indices = Host::ints(
                &[nt, rows, 4],
                (0..nt * rows * 4).map(|i| i * 5 % table_rows.max(1)),
            );
            let coefficients = Host::new(
                DType::F32,
                &[nt, rows, 4],
                (0..nt * rows * 4)
                    .map(|_| rng.fraction())
                    .collect(),
            );
            let expected = interpret(
                &module,
                "vision_patch_stem",
                &[
                    ("W0", weights),
                    ("W1", weights),
                    ("B", DType::F32),
                    ("PE", DType::F32),
                ],
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
                    &[
                        ("C", channels),
                        ("S", s),
                        ("P", patch),
                        ("H", hidden),
                        ("NB", nb),
                        ("NT", nt),
                    ],
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
                &format!(
                    "vision_patch_stem {:?} S{s} NB{nb} NT{nt} P{patch} {weights:?}",
                    device.backend()
                ),
                &read_values(&result),
                &expected,
                2e-3,
                2e-4,
            );
        }
    }
}

/// Layer norms and RMS norms, weighted or not, of single rows and of cells
/// of four rows laid side by side or interleaved, gathered or in order; the
/// interleaved RMS cell also at GLM-5.3 Flash's tower width (1024), whose
/// rows span many threads' strides.
#[test]
fn norm_matches_its_portable_body() {
    let module = module();
    let cells = 3usize;
    // (members, centered, weighted, biased, gathered, interleave, activation, output, width)
    let forms = [
        (1usize, true, true, true, false, false, 0, DType::F16, 96usize),
        (1, false, true, false, false, false, 0, DType::F32, 96),
        (1, false, false, false, false, false, 0, DType::F32, 96),
        (4, true, true, true, false, false, 0, DType::F16, 96),
        (4, false, true, false, true, true, 0, DType::F16, 96),
        (4, false, true, false, false, true, 0, DType::F16, 1024),
        // A layer norm followed by the erf GELU (a merger stage's), and by SiLU.
        (1, true, true, true, false, false, 2, DType::F16, 96),
        (1, true, true, true, false, false, 5, DType::F32, 96),
    ];
    for device in devices() {
        for (g, centered, weighted, biased, gathered, interleave, activation, output, h) in forms {
            let (nw, nb, no) = (
                usize::from(weighted),
                usize::from(biased),
                usize::from(gathered),
            );
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
                    Input::I32(activation),
                ],
            );
            let kernel = vision_norm::native_for_device_with(
                &device,
                vision_norm::Elements {
                    NWE: weight.element(),
                    NBE: bias.element(),
                    Y: Element::dense(output),
                },
                &specialization_on(
                    &device,
                    &[("G", g), ("H", h), ("NW", nw), ("NB", nb), ("NO", no)],
                ),
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
                    activation,
                })
                .unwrap()
                .value;
            relative_error(
                &format!(
                    "vision_norm {:?} G{g} centered {centered} NW{nw} NB{nb} NO{no} activation {activation}",
                    device.backend()
                ),
                &read_values(&result),
                &expected,
                2e-3,
                2e-4,
            );
        }
    }
}

/// Per-element rounding bound of `vision_linear` on `backend` (`x` of A = F16)
/// against its portable body's `expected`: [`common::bound::Form::matrix`]
/// bounds the native and portable F32 dot products (`vision_linear` runs the
/// matrix-unit GEMM at every row count), followed by the F32 bias, clamp,
/// epilogue (`activation` 1..=5, `gate`, or `residual`) and publication to
/// `output`.
#[allow(clippy::too_many_arguments)]
fn linear_bounds(
    backend: seismic::BackendName,
    x: &Host,
    weight: &Host,
    bias: &Host,
    residuals: &Host,
    gate: &Host,
    minimum: &Host,
    maximum: &Host,
    activation: i32,
    output: DType,
    expected: &[f32],
) -> Vec<f64> {
    use common::bound::{
        Float, Form, Library, Products, SILU_SLOPE, gamma, silu_evaluation, silu_f64,
    };
    let (m, k, n) = (x.shape[0], x.shape[1], weight.shape[0]);
    let form = Form::matrix(backend, Float::F16, weight.element());
    let out = Float::dense(Element::dense(output)).unwrap();
    let u = Float::F32.unit();
    let (native, host) = (Library::of(backend), Library::of(seismic::BackendName::Cpu));
    let mut bounds = Vec::with_capacity(m * n);
    for row in 0..m {
        let x_row = &x.values[row * k..(row + 1) * k];
        for col in 0..n {
            let at = row * n + col;
            let w_row = &weight.values[col * k..(col + 1) * k];
            let products = Products::of(x_row, w_row);
            let mut projected: f64 = x_row
                .iter()
                .zip(w_row)
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum();
            let mut delta = form.sums(&products);
            if bias.shape[0] == 1 {
                let b = f64::from(bias.values[col]);
                delta = (1.0 + u) * delta + 2.0 * u * (projected.abs() + delta + b.abs());
                projected += b;
            }
            if minimum.shape[0] == 1 {
                projected = projected
                    .clamp(f64::from(minimum.values[0]), f64::from(maximum.values[0]));
            }
            let reach = projected.abs() + delta;
            let published_delta = match activation {
                0 if gate.shape[0] == 1 => {
                    let g = f64::from(gate.values[at]).abs();
                    delta * g + 2.0 * u * reach * g
                }
                0 if residuals.shape[0] == 1 => {
                    let r = f64::from(residuals.values[at]).abs();
                    (1.0 + u) * delta + 2.0 * u * (r + reach)
                }
                0 => delta,
                // 0.5 a (1 + (2 / (1 + exp(-2u)) - 1)), u = c0 (a + c1 a³);
                // |d/da| <= 1.13, |d/dt 2 / (1 + e^-t)| <= 0.5.
                1 => {
                    let arg = 0.7978845608028654 * (reach + 0.044715 * reach.powi(3));
                    let hyp = 0.5 * (2.0 * gamma(5) * arg + native.exp + host.exp)
                        + 2.0 * (native.divide + host.divide + gamma(4));
                    1.13 * delta + reach * (hyp + 2.0 * gamma(3))
                }
                // 0.5 a (1 + erf(a / √2)); |d/da| <= 1.13, Numerical Recipes
                // `erfcc` on Metal/Vulkan within 1.2e-7 plus Horner steps and
                // `exp` on [0, 1].
                2 => {
                    let erf_eval = 1.2e-7
                        + native.exp
                        + host.exp
                        + native.divide
                        + host.divide
                        + 2.0 * gamma(12);
                    1.13 * delta + 0.5 * reach * (erf_eval + 2.0 * gamma(3))
                }
                // a / (1 + exp(-1.702 a)): slope <= SILU_SLOPE, plus the F32
                // product `-1.702 * a` before `exp`.
                3 => {
                    let eval = silu_evaluation(&native)
                        + silu_evaluation(&host)
                        + 2.0 * u * (1.0 + 1.702 * reach);
                    SILU_SLOPE * delta
                        + eval * (silu_f64(1.702 * projected).abs() / 1.702 + SILU_SLOPE * delta)
                }
                // max(a, 0)².
                4 => {
                    let pos = projected.max(0.0);
                    delta * (2.0 * pos + delta) + 2.0 * u * (pos + delta).powi(2)
                }
                // SiLU: a / (1 + exp(-a)).
                5 => {
                    let eval = silu_evaluation(&native) + silu_evaluation(&host);
                    SILU_SLOPE * delta + eval * (silu_f64(projected).abs() + SILU_SLOPE * delta)
                }
                other => panic!("unexpected vision activation code {other}"),
            };
            bounds.push(out.rounded(published_delta, expected[at]));
        }
    }
    bounds
}

/// Projections with their epilogues: the residual form (F32), the five
/// activations (A), the gated and the clamped forms, and F32 and BF16 weights
/// over depths and tiles past one GEMM step, held element by element to the
/// rounding bound of each backend's matrix-unit form ([`linear_bounds`]).
#[test]
fn linear_matches_its_portable_body() {
    let module = module();
    let (tiny_m, tiny_n, tiny_k) = (2usize, 2usize, 8usize);
    // (biased, residual, gated, clamped, activation, output, weights, m, n, depth)
    let forms = [
        (
            true,
            true,
            false,
            false,
            0,
            DType::F32,
            DType::F16,
            tiny_m,
            tiny_n,
            tiny_k,
        ),
        (true, false, false, false, 0, DType::F16, DType::F16, tiny_m, tiny_n, tiny_k),
        (true, false, false, false, 1, DType::F16, DType::F16, tiny_m, tiny_n, tiny_k),
        (true, false, false, false, 2, DType::F16, DType::F16, tiny_m, tiny_n, tiny_k),
        (false, false, false, false, 3, DType::F32, DType::F16, tiny_m, tiny_n, tiny_k),
        (false, false, false, false, 4, DType::F16, DType::F16, tiny_m, tiny_n, tiny_k),
        (false, false, false, false, 5, DType::F16, DType::F16, tiny_m, tiny_n, tiny_k),
        (false, false, true, false, 0, DType::F16, DType::F16, tiny_m, 65, 33),
        (true, true, false, true, 0, DType::F32, DType::F16, tiny_m, 65, 33),
        // One past a whole M and N tile and a whole K step (64, 64 and 32).
        (true, false, false, false, 0, DType::F16, DType::F32, 65, 65, 33),
        // Past CUDA's 128 x 128 tiles in M and N, two K steps, BF16 weights
        // (GLM-5.3 Flash's projector) under a residual.
        (true, true, false, false, 0, DType::F32, DType::BF16, 200, 300, 64),
    ];
    for device in devices() {
        for (biased, residual, gated, clamped, activation, output, weights, m, n, k) in forms {
            // CUDA's form stages its activation rows in whole 16-byte chunks
            // and stores output pairs (its `K % 8 == 0` and `N % 2 == 0`): a
            // ragged depth or width is there the next whole one.
            let cuda = device.backend() == seismic::BackendName::Cuda;
            let (n, k) = if cuda { (n.next_multiple_of(2), k.next_multiple_of(8)) } else { (n, k) };
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
                &[
                    ("A", DType::F16),
                    ("W", weights),
                    ("B", DType::F32),
                    ("Y", output),
                ],
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
                    &[
                        ("N", n),
                        ("K", k),
                        ("NB", nb),
                        ("NR", nr),
                        ("NG", ng),
                        ("NC", nc),
                    ],
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
            let bounds = linear_bounds(
                device.backend(),
                &x,
                &weight,
                &bias,
                &residuals,
                &gate,
                &minimum,
                &maximum,
                activation,
                output,
                &expected,
            );
            common::bound::assert_within(
                &format!(
                    "vision_linear {:?} NB{nb} NR{nr} NG{ng} NC{nc} activation {activation} {weights:?} K{k}",
                    device.backend()
                ),
                &read_values(&result),
                &expected,
                &bounds,
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
            &[
                Input::Tensor(&x),
                Input::Tensor(&minimum),
                Input::Tensor(&maximum),
            ],
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
        assert_eq!(
            read_values(&result),
            expected,
            "vision_clamp {:?}",
            device.backend()
        );
    }
}

/// The rotary attention (the Qwen form, small: the oracle's per-element
/// cost does not need a boundary shape for a path that isn't about a tile
/// boundary); head norms, a weightless value norm and the unit scale at a
/// head width of 72 (Gemma 4's large towers), also small; windows of 48
/// rows at head widths 64 and 96 (Muse Glimmer), with just enough rows (49)
/// for a query tile to span two windows, the last one partial, since the
/// window size is this test's own 48 (below), not a kernel constant.
#[test]
fn attention_matches_its_portable_body() {
    check_attention_forms(&[
        (2usize, 1usize, 16usize, false, false, false, false, true, 5),
        (2, 1, 18, true, true, false, true, true, 5),
        (1, 1, 16, false, false, true, false, true, 49),
        (1, 1, 24, false, false, true, false, true, 49),
        // LFM3-VL: two query heads per key and value head.
        (2, 2, 16, true, false, false, false, true, 5),
        // GLM-5.3 Flash: head norms and rotary at width 64, full attention
        // over rows past several 64-row query blocks and key tiles.
        (2, 1, 16, true, false, false, false, true, 200),
    ]);
}

#[test]
fn learned_position_attention_matches_its_portable_body() {
    // SigLIP: 72-column heads, learned positions, no rotary or head norm.
    check_attention_forms(&[(2, 1, 18, false, false, false, false, false, 5)]);
}

/// Forms (key/value heads, query heads per key/value head, P, head norms,
/// value norm, windows, unit scale, rotary, rows). `rows` is the smallest
/// that still exercises the form's path: just above one 48-row window
/// (this test's own window size, not a kernel constant) where `windows`,
/// otherwise small, since the oracle's cost is per element and the path
/// through it is chosen by the flags, not by which row is read.
#[allow(clippy::type_complexity)]
fn check_attention_forms(forms: &[(usize, usize, usize, bool, bool, bool, bool, bool, usize)]) {
    let module = module();
    for device in devices() {
        for &(kv_heads, group, quarter, head_norms, value_norm, windows, unit_scale, rotary, rows) in
            forms
        {
            let heads = kv_heads * group;
            let width = 4 * quarter;
            let (nq, nv, ws) = (
                usize::from(head_norms),
                usize::from(value_norm),
                usize::from(windows),
            );
            let mut rng = Rng::new(25);
            let shared = [rows, kv_heads, width];
            let query = Host::random(DType::F16, &[rows, heads, width], 2.0, &mut rng);
            let key = Host::random(DType::F16, &shared, 2.0, &mut rng);
            let value = Host::random(DType::F16, &shared, 1.0, &mut rng);
            let query_norm = Host::rows(DType::F32, nq, &[width], 0.3, 1.0, &mut rng);
            let key_norm = Host::rows(DType::F32, nq, &[width], 0.3, 1.0, &mut rng);
            let value_norm = Host::new(DType::F32, &[nv, width], vec![1.0; nv * width]);
            let side = 7;
            let coordinates = Host::ints(
                &[rows, 2],
                (0..rows).flat_map(|row| [row / side + 1, row % side + 1]),
            );
            // Windows of 48 rows (the last one partial).
            let spans = Host::ints(
                &[ws, rows, 2],
                (0..ws * rows).flat_map(|row| [row / 48 * 48, (row / 48 * 48 + 48).min(rows)]),
            );
            let log_base = if rotary { 100f32.ln() } else { 0.0 };
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
                    &[
                        ("KV", kv_heads),
                        ("G", group),
                        ("P", quarter),
                        ("NQ", nq),
                        ("NV", nv),
                        ("WS", ws),
                    ],
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
                    "vision_attention {:?} KV{kv_heads} G{group} W{width} NQ{nq} NV{nv} WS{ws} unit {unit_scale}",
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

/// The gated 3 × 3 convolution over a 6 × 8 grid held in 2 × 2 cell order
/// (LFM3-VL), with F32 and f16 kernels; padding rows past the grid read no
/// neighbour.
#[test]
fn gated_convolution_matches_its_portable_body() {
    let module = module();
    let (height, width, padding, hidden, side) = (6usize, 8usize, 4usize, 96usize, 3usize);
    let (rows, taps) = (height * width + padding, side * side);
    // Tower row of each grid cell, cell by cell.
    let mut grid = vec![0usize; height * width];
    let mut row = 0;
    for cell_y in 0..height / 2 {
        for cell_x in 0..width / 2 {
            for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                grid[(2 * cell_y + dy) * width + 2 * cell_x + dx] = row;
                row += 1;
            }
        }
    }
    let mut neighbors = vec![-1i64; rows * taps];
    for y in 0..height {
        for x in 0..width {
            for tap in 0..taps {
                let (ny, nx) = ((y + tap / side).checked_sub(1), (x + tap % side).checked_sub(1));
                if let (Some(ny), Some(nx)) = (ny, nx) {
                    if ny < height && nx < width {
                        neighbors[grid[y * width + x] * taps + tap] = grid[ny * width + nx] as i64;
                    }
                }
            }
        }
    }
    let neighbors = Host::new(
        DType::I32,
        &[rows, taps],
        neighbors.into_iter().map(|value| value as f32).collect(),
    );
    for device in devices() {
        for weights in [DType::F32, DType::F16] {
            let mut rng = Rng::new(31);
            let value = Host::random(DType::F16, &[rows, hidden], 1.0, &mut rng);
            let gate = Host::random(DType::F16, &[rows, hidden], 1.0, &mut rng);
            let kernel_weights = Host::random(weights, &[hidden, taps], 0.5, &mut rng);
            let expected = interpret(
                &module,
                "vision_gated_convolution",
                &[("A", DType::F16), ("KE", weights)],
                &[
                    Input::Tensor(&value),
                    Input::Tensor(&gate),
                    Input::Tensor(&kernel_weights),
                    Input::Tensor(&neighbors),
                ],
            );
            let kernel = vision_gated_convolution::native_for_device_with(
                &device,
                vision_gated_convolution::Elements {
                    A: Element::f16(),
                    KE: kernel_weights.element(),
                },
                &specialization_on(&device, &[("H", hidden), ("T", taps)]),
            )
            .unwrap();
            let t = |host: &Host| host.tensor(&device);
            let result = kernel
                .call(vision_gated_convolution::Args {
                    value: &t(&value),
                    gate: &t(&gate),
                    weight: &t(&kernel_weights),
                    neighbors: &t(&neighbors),
                })
                .unwrap()
                .value;
            relative_error(
                &format!("vision_gated_convolution {:?} {weights:?}", device.backend()),
                &read_values(&result),
                &expected,
                2e-3,
                2e-4,
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
            (0..rows * 4)
                .map(|_| rng.fraction())
                .collect(),
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
            vision_position::Elements {
                PE: table.element(),
            },
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
