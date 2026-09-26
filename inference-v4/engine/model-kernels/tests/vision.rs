//! The vision entries (`qwen_vision_stem`, `qwen_vision_block`,
//! `qwen_vision_merger`) on the host's accelerator (Metal on macOS, else
//! CUDA) and on the CPU, checked against their portable bodies run by the Seismic
//! interpreter at small shapes, and against a host model of those bodies at
//! the projector's geometry. Weights are f16 and bias/norm vectors f32 (the
//! projector file's elements, kept resident as stored); the activation A is
//! f16 (the Qwen35 projector's) and the residual stream F32.
//!
//! The native entries reassociate the F32 sums, round the stem's pixels and
//! the attention probabilities to half for the matrix units, and may land on
//! the other side of an A rounding than the portable body; each such flip
//! moves a published intermediate by one A ulp. The bounds are relative to
//! the result's RMS.

use magnitude_model_kernels::{qwen_vision_block, qwen_vision_merger, qwen_vision_stem};
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
use std::time::Instant;

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

/// The activation rounding (A = f16).
fn round_a(value: f32) -> f32 {
    f16_to_f32(registry::f16_bits(value))
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
                    round_a(value)
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
    /// A matrix [N, K] whose rows have dot products of order `scale` with
    /// unit activations.
    fn weight(rows: usize, k: usize, scale: f32, rng: &mut Rng) -> Self {
        Self::random(DType::F16, &[rows, k], scale * 1.7 / (k as f32).sqrt(), rng)
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

fn read_f32(tensor: &Tensor) -> Vec<f32> {
    tensor
        .read_to_host()
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

// ---------------------------------------------------------------------------
// Portable bodies.

fn module() -> CheckedModule {
    let mut sources = seismic_std::sources();
    sources.push(SourceFile {
        path: "vision.seismic".into(),
        text: include_str!("../kernels/vision.seismic").into(),
    });
    check_source(sources).unwrap()
}

enum Input<'a> {
    Tensor(&'a Host),
    F32(f32),
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
        })
        .collect::<Vec<_>>();
    let started = Instant::now();
    let outcome: OracleOutcome = interpreter
        .run(&arguments)
        .unwrap_or_else(|error| panic!("{name}: {error}"));
    if let SourceTermination::Failed(failure) = outcome.termination() {
        panic!("{name} failed: {failure}");
    }
    eprintln!(
        "{name}: portable body in {:.1} s",
        started.elapsed().as_secs_f64()
    );
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
/// `SEISMIC_TEST_BACKEND=vulkan` selects Vulkan).
fn device() -> Device {
    let catalog = seismic::DeviceCatalog::discover().unwrap();
    let backend = match std::env::var("SEISMIC_TEST_BACKEND").ok().as_deref() {
        Some("vulkan") => seismic::BackendName::Vulkan,
        Some(other) => panic!("SEISMIC_TEST_BACKEND={other}: only vulkan is selectable"),
        None if cfg!(target_os = "macos") => seismic::BackendName::Metal,
        None => seismic::BackendName::Cuda,
    };
    catalog.open_backend(backend).unwrap()
}

/// The accelerator and the CPU.
fn devices() -> Vec<Device> {
    let catalog = seismic::DeviceCatalog::discover().unwrap();
    vec![
        device(),
        catalog.open_backend(seismic::BackendName::Cpu).unwrap(),
    ]
}

fn is_cpu(device: &Device) -> bool {
    device.backend() == seismic::BackendName::Cpu
}

/// The specialization of a vision entry on `device`: the GPU forms' static
/// dimensions, or the CPU form's weight rows per work item.
fn specialization_on(device: &Device, statics: &[(&str, usize)]) -> NativeSpecialization {
    if is_cpu(device) {
        return NativeSpecialization::new().with_param("ROWS", 8);
    }
    statics.iter().fold(
        NativeSpecialization::new(),
        |specialization, (name, value)| specialization.with_static(*name, *value as u64),
    )
}

// ---------------------------------------------------------------------------
// The block fixture and its host model.

struct Block {
    heads: usize,
    quarter: usize,
    intermediate: usize,
    hidden: Host,
    coordinates: Host,
    norm1_weight: Host,
    norm1_bias: Host,
    qkv_weight: Host,
    qkv_bias: Host,
    projection_weight: Host,
    projection_bias: Host,
    norm2_weight: Host,
    norm2_bias: Host,
    up_weight: Host,
    up_bias: Host,
    down_weight: Host,
    down_bias: Host,
}

const EPSILON: f32 = 1e-6;

fn host_layer_norm(x: &[f32], width: usize, weight: &[f32], bias: &[f32]) -> Vec<f32> {
    x.chunks_exact(width)
        .flat_map(|row| {
            let mean = row.iter().sum::<f32>() / width as f32;
            let variance = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / width as f32;
            let inverse = 1.0 / (variance + EPSILON).sqrt();
            row.iter()
                .zip(weight.iter().zip(bias))
                .map(move |(v, (w, b))| round_a((v - mean) * inverse * w + b))
                .collect::<Vec<_>>()
        })
        .collect()
}

/// x . transpose(weight) + bias in F32.
fn host_linear(x: &[f32], k: usize, weight: &Host, bias: &Host) -> Vec<f32> {
    let n = weight.shape[0];
    x.chunks_exact(k)
        .flat_map(|row| {
            (0..n).map(move |output| {
                let w = &weight.values[output * k..(output + 1) * k];
                row.iter()
                    .zip(w)
                    .fold(0.0f32, |acc, (a, b)| a.mul_add(*b, acc))
                    + bias.values[output]
            })
        })
        .collect()
}

fn host_gelu_tanh(value: f32) -> f32 {
    let argument = 0.797_884_6 * (value + 0.044715 * value * value * value);
    0.5 * value * (1.0 + (2.0 / (1.0 + (-2.0 * argument).exp()) - 1.0))
}

impl Block {
    fn new(rows: usize, heads: usize, quarter: usize, intermediate: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let d = heads * 4 * quarter;
        let side = (rows as f64).sqrt().ceil() as usize;
        let coordinates = (0..rows)
            .flat_map(|row| [(row / side) as f32, (row % side) as f32])
            .collect();
        let norm = |rng: &mut Rng| {
            Host::new(
                DType::F32,
                &[d],
                rng.values(d, 0.2).iter().map(|v| 1.0 + v).collect(),
            )
        };
        Self {
            heads,
            quarter,
            intermediate,
            hidden: Host::random(DType::F32, &[rows, heads, 4, quarter], 2.0, &mut rng),
            coordinates: Host::new(DType::I32, &[rows, 2], coordinates),
            norm1_weight: norm(&mut rng),
            norm1_bias: Host::random(DType::F32, &[d], 0.1, &mut rng),
            qkv_weight: Host::weight(3 * d, d, 1.5, &mut rng),
            qkv_bias: Host::random(DType::F32, &[3 * d], 0.2, &mut rng),
            projection_weight: Host::weight(d, d, 1.0, &mut rng),
            projection_bias: Host::random(DType::F32, &[d], 0.1, &mut rng),
            norm2_weight: norm(&mut rng),
            norm2_bias: Host::random(DType::F32, &[d], 0.1, &mut rng),
            up_weight: Host::weight(intermediate, d, 2.0, &mut rng),
            up_bias: Host::random(DType::F32, &[intermediate], 0.2, &mut rng),
            down_weight: Host::weight(d, intermediate, 1.0, &mut rng),
            down_bias: Host::random(DType::F32, &[d], 0.1, &mut rng),
        }
    }

    fn inputs(&self) -> Vec<Input<'_>> {
        vec![
            Input::Tensor(&self.hidden),
            Input::Tensor(&self.coordinates),
            Input::Tensor(&self.norm1_weight),
            Input::Tensor(&self.norm1_bias),
            Input::Tensor(&self.qkv_weight),
            Input::Tensor(&self.qkv_bias),
            Input::Tensor(&self.projection_weight),
            Input::Tensor(&self.projection_bias),
            Input::Tensor(&self.norm2_weight),
            Input::Tensor(&self.norm2_bias),
            Input::Tensor(&self.up_weight),
            Input::Tensor(&self.up_bias),
            Input::Tensor(&self.down_weight),
            Input::Tensor(&self.down_bias),
            Input::F32(EPSILON),
        ]
    }

    fn bindings(&self) -> Vec<(&'static str, DType)> {
        vec![
            ("A", DType::F16),
            ("N1W", DType::F32),
            ("N1B", DType::F32),
            ("QW", DType::F16),
            ("QB", DType::F32),
            ("PW", DType::F16),
            ("PB", DType::F32),
            ("N2W", DType::F32),
            ("N2B", DType::F32),
            ("UW", DType::F16),
            ("UB", DType::F32),
            ("DW", DType::F16),
            ("DB", DType::F32),
        ]
    }

    fn kernel(&self, device: &Device) -> seismic::NativeKernel<qwen_vision_block::Entry> {
        qwen_vision_block::native_for_device_with(
            device,
            qwen_vision_block::Elements {
                A: Element::f16(),
                N1W: self.norm1_weight.element(),
                N1B: self.norm1_bias.element(),
                QW: self.qkv_weight.element(),
                QB: self.qkv_bias.element(),
                PW: self.projection_weight.element(),
                PB: self.projection_bias.element(),
                N2W: self.norm2_weight.element(),
                N2B: self.norm2_bias.element(),
                UW: self.up_weight.element(),
                UB: self.up_bias.element(),
                DW: self.down_weight.element(),
                DB: self.down_bias.element(),
            },
            &specialization_on(
                device,
                &[
                    ("H", self.heads),
                    ("P", self.quarter),
                    ("F", self.intermediate),
                ],
            ),
        )
        .unwrap()
    }

    fn run(
        &self,
        device: &Device,
        kernel: &seismic::NativeKernel<qwen_vision_block::Entry>,
    ) -> Tensor {
        let t = |host: &Host| host.tensor(device);
        kernel
            .call(qwen_vision_block::Args {
                hidden: &t(&self.hidden),
                coordinates: &t(&self.coordinates),
                norm1_weight: &t(&self.norm1_weight),
                norm1_bias: &t(&self.norm1_bias),
                qkv_weight: &t(&self.qkv_weight),
                qkv_bias: &t(&self.qkv_bias),
                projection_weight: &t(&self.projection_weight),
                projection_bias: &t(&self.projection_bias),
                norm2_weight: &t(&self.norm2_weight),
                norm2_bias: &t(&self.norm2_bias),
                up_weight: &t(&self.up_weight),
                up_bias: &t(&self.up_bias),
                down_weight: &t(&self.down_weight),
                down_bias: &t(&self.down_bias),
                epsilon: EPSILON,
            })
            .unwrap()
            .value
    }

    /// The portable body's F32 arithmetic and A roundings.
    fn host(&self) -> Vec<f32> {
        let rows = self.hidden.shape[0];
        let w = 4 * self.quarter;
        let d = self.heads * w;
        let p = self.quarter;
        let hidden = &self.hidden.values;
        let normalized = host_layer_norm(
            hidden,
            d,
            &self.norm1_weight.values,
            &self.norm1_bias.values,
        );
        let mut projected = host_linear(&normalized, d, &self.qkv_weight, &self.qkv_bias)
            .into_iter()
            .map(round_a)
            .collect::<Vec<_>>();
        for row in 0..rows {
            let coordinates = [
                self.coordinates.values[row * 2],
                self.coordinates.values[row * 2 + 1],
            ];
            for part in 0..2 {
                for head in 0..self.heads {
                    let at = row * 3 * d + part * d + head * w;
                    let source = projected[at..at + w].to_vec();
                    for i in 0..w {
                        let pair = i % (2 * p);
                        let angle = coordinates[pair / p]
                            * (-(10000f32.ln()) * (pair % p) as f32 / p as f32).exp();
                        let (s, c) = angle.sin_cos();
                        projected[at + i] = round_a(if i < 2 * p {
                            source[i] * c - source[i + 2 * p] * s
                        } else {
                            source[i] * c + source[i - 2 * p] * s
                        });
                    }
                }
            }
        }
        let scale = 1.0 / (w as f32).sqrt();
        let mut attended = vec![0.0f32; rows * d];
        for head in 0..self.heads {
            for row in 0..rows {
                let q = &projected[row * 3 * d + head * w..][..w];
                let scores = (0..rows)
                    .map(|key| {
                        let k = &projected[key * 3 * d + d + head * w..][..w];
                        q.iter()
                            .zip(k)
                            .fold(0.0f32, |acc, (a, b)| a.mul_add(*b, acc))
                            * scale
                    })
                    .collect::<Vec<_>>();
                let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let weights = scores
                    .iter()
                    .map(|s| (s - maximum).exp())
                    .collect::<Vec<_>>();
                let denominator = weights.iter().sum::<f32>();
                for j in 0..w {
                    let acc = (0..rows).fold(0.0f32, |acc, key| {
                        weights[key].mul_add(projected[key * 3 * d + 2 * d + head * w + j], acc)
                    });
                    attended[row * d + head * w + j] = round_a(acc / denominator);
                }
            }
        }
        let mixed = host_linear(&attended, d, &self.projection_weight, &self.projection_bias);
        let residual = hidden
            .iter()
            .zip(&mixed)
            .map(|(h, m)| h + m)
            .collect::<Vec<_>>();
        let normalized = host_layer_norm(
            &residual,
            d,
            &self.norm2_weight.values,
            &self.norm2_bias.values,
        );
        let activated = host_linear(&normalized, d, &self.up_weight, &self.up_bias)
            .into_iter()
            .map(|up| round_a(host_gelu_tanh(round_a(up))))
            .collect::<Vec<_>>();
        let down = host_linear(
            &activated,
            self.intermediate,
            &self.down_weight,
            &self.down_bias,
        );
        residual.iter().zip(&down).map(|(r, d)| r + d).collect()
    }
}

// ---------------------------------------------------------------------------
// Tests.

#[test]
fn stem_matches_its_portable_body() {
    for device in devices() {
        stem_matches_its_portable_body_on(&device);
    }
}

fn stem_matches_its_portable_body_on(device: &Device) {
    let module = module();
    let (rows, channels, patch, hidden, table_rows) = (8, 3, 16, 64, 16);
    let mut rng = Rng::new(11);
    let pixels = Host::random(
        DType::F32,
        &[rows, channels, 2, patch, patch],
        1.0,
        &mut rng,
    );
    let weight_0 = Host::random(
        DType::F16,
        &[hidden, channels, patch, patch],
        0.04,
        &mut rng,
    );
    let weight_1 = Host::random(
        DType::F16,
        &[hidden, channels, patch, patch],
        0.04,
        &mut rng,
    );
    let bias = Host::random(DType::F32, &[hidden], 0.3, &mut rng);
    let table = Host::random(DType::F32, &[table_rows, hidden], 1.0, &mut rng);
    let indices = Host::new(
        DType::I32,
        &[rows, 4],
        (0..rows * 4)
            .map(|_| (rng.next() % table_rows as u32) as f32)
            .collect(),
    );
    let coefficients = Host::new(
        DType::F32,
        &[rows, 4],
        (0..rows * 4)
            .map(|_| (rng.symmetric() + 1.0) / 2.0)
            .collect(),
    );
    let expected = interpret(
        &module,
        "qwen_vision_stem",
        &[
            ("W0", DType::F16),
            ("W1", DType::F16),
            ("B", DType::F32),
            ("PE", DType::F32),
        ],
        &[
            Input::Tensor(&pixels),
            Input::Tensor(&weight_0),
            Input::Tensor(&weight_1),
            Input::Tensor(&bias),
            Input::Tensor(&table),
            Input::Tensor(&indices),
            Input::Tensor(&coefficients),
        ],
    );
    let kernel = qwen_vision_stem::native_for_device_with(
        device,
        qwen_vision_stem::Elements {
            W0: weight_0.element(),
            W1: weight_1.element(),
            B: bias.element(),
            PE: table.element(),
        },
        &specialization_on(device, &[("C", channels), ("P", patch), ("H", hidden)]),
    )
    .unwrap();
    let t = |host: &Host| host.tensor(device);
    let result = kernel
        .call(qwen_vision_stem::Args {
            pixels: &t(&pixels),
            temporal_weight_0: &t(&weight_0),
            temporal_weight_1: &t(&weight_1),
            bias: &t(&bias),
            table: &t(&table),
            indices: &t(&indices),
            coefficients: &t(&coefficients),
        })
        .unwrap()
        .value;
    relative_error(
        "qwen_vision_stem",
        &read_f32(&result),
        &expected,
        2e-3,
        2e-4,
    );
}

/// Two heads of width 64 over 40 rows (a partial query tile, a partial key
/// tile) against the portable body; the host model is held to the portable
/// body too, so the larger case below can use it.
#[test]
fn block_matches_its_portable_body() {
    for device in devices() {
        block_matches_its_portable_body_on(&device);
    }
}

fn block_matches_its_portable_body_on(device: &Device) {
    let module = module();
    let block = Block::new(40, 2, 16, 128, 21);
    let expected = interpret(
        &module,
        "qwen_vision_block",
        &block.bindings(),
        &block.inputs(),
    );
    relative_error(
        "qwen_vision_block host model",
        &block.host(),
        &expected,
        1e-3,
        1e-4,
    );
    let result = block.run(device, &block.kernel(device));
    relative_error(
        "qwen_vision_block",
        &read_f32(&result),
        &expected,
        5e-3,
        5e-4,
    );
}

/// The projector's geometry (16 heads of 64, F 4096) over 200 rows (several
/// query and key tiles, partial GEMM tiles) against the host model.
#[test]
fn block_matches_the_host_model_at_projector_geometry() {
    for device in devices() {
        block_matches_the_host_model_at_projector_geometry_on(&device);
    }
}

fn block_matches_the_host_model_at_projector_geometry_on(device: &Device) {
    let block = Block::new(200, 16, 16, 4096, 31);
    let expected = block.host();
    let result = block.run(device, &block.kernel(device));
    relative_error(
        "qwen_vision_block (projector geometry)",
        &read_f32(&result),
        &expected,
        5e-3,
        5e-4,
    );
}

#[test]
fn merger_matches_its_portable_body() {
    for device in devices() {
        merger_matches_its_portable_body_on(&device);
    }
}

fn merger_matches_its_portable_body_on(device: &Device) {
    let module = module();
    let (rows, group, hidden, output) = (6, 4, 64, 96);
    let mut rng = Rng::new(41);
    let width = group * hidden;
    let rows_in = Host::random(DType::F32, &[rows * group, hidden], 2.0, &mut rng);
    let norm_weight = Host::new(
        DType::F32,
        &[hidden],
        rng.values(hidden, 0.2).iter().map(|v| 1.0 + v).collect(),
    );
    let norm_bias = Host::random(DType::F32, &[hidden], 0.1, &mut rng);
    let up_weight = Host::weight(width, width, 2.0, &mut rng);
    let up_bias = Host::random(DType::F32, &[width], 0.2, &mut rng);
    let down_weight = Host::weight(output, width, 1.0, &mut rng);
    let down_bias = Host::random(DType::F32, &[output], 0.1, &mut rng);
    let expected = interpret(
        &module,
        "qwen_vision_merger",
        &[
            ("A", DType::F16),
            ("NW", DType::F32),
            ("NB", DType::F32),
            ("UW", DType::F16),
            ("UB", DType::F32),
            ("DW", DType::F16),
            ("DB", DType::F32),
        ],
        &[
            Input::Tensor(&rows_in),
            Input::Tensor(&norm_weight),
            Input::Tensor(&norm_bias),
            Input::Tensor(&up_weight),
            Input::Tensor(&up_bias),
            Input::Tensor(&down_weight),
            Input::Tensor(&down_bias),
            Input::F32(EPSILON),
        ],
    );
    let merger = qwen_vision_merger::native_for_device_with(
        device,
        qwen_vision_merger::Elements {
            A: Element::f16(),
            NW: norm_weight.element(),
            NB: norm_bias.element(),
            UW: up_weight.element(),
            UB: up_bias.element(),
            DW: down_weight.element(),
            DB: down_bias.element(),
        },
        &specialization_on(device, &[("G", group), ("H", hidden), ("D", output)]),
    )
    .unwrap();
    let t = |host: &Host| host.tensor(device);
    let merged = merger
        .call(qwen_vision_merger::Args {
            hidden: &t(&rows_in),
            norm_weight: &t(&norm_weight),
            norm_bias: &t(&norm_bias),
            up_weight: &t(&up_weight),
            up_bias: &t(&up_bias),
            down_weight: &t(&down_weight),
            down_bias: &t(&down_bias),
            epsilon: EPSILON,
        })
        .unwrap()
        .value;
    relative_error(
        "qwen_vision_merger",
        &read_f32(&merged),
        &expected,
        5e-3,
        5e-4,
    );
}

/// A real projector forward: the whole tower (stem, every block, merger) on
/// one image's prepared inputs, against an F64 forward of the same inputs and
/// the projector's stored weights. `VISION_FIXTURE` names a directory with
/// `weights.bin` + `weights.txt` (name, dtype, byte offset, shape per line),
/// `pixels.bin` [M, 3, 2, 16, 16] f32, `coordinates.bin` [M, 2] i32,
/// `indices.bin`, `coefficients.bin` [M, 4], `reference.bin` (the F64 forward)
/// and `llama_policy.bin` (f16 matrix inputs, F32 elsewhere: llama.cpp's
/// numerics), both [M / 4, D] f32. `cargo test --release -- --ignored`.
#[test]
#[ignore]
fn projector_forward_matches_the_f64_reference() {
    let directory =
        std::path::PathBuf::from(std::env::var("VISION_FIXTURE").expect("VISION_FIXTURE"));
    let device = device();
    let read = |name: &str| std::fs::read(directory.join(name)).unwrap();
    let f32s = |bytes: &[u8]| {
        bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>()
    };
    let blob = read("weights.bin");
    let index = String::from_utf8(read("weights.txt")).unwrap();
    let weights = index
        .lines()
        .map(|line| {
            let fields = line.split_whitespace().collect::<Vec<_>>();
            let (element, bytes) = match fields[1] {
                "f16" => (Element::f16(), 2),
                "f32" => (Element::f32(), 4),
                other => panic!("unexpected dtype {other}"),
            };
            let offset = fields[2].parse::<usize>().unwrap();
            let shape = fields[3..]
                .iter()
                .map(|f| f.parse::<u64>().unwrap())
                .collect::<Vec<_>>();
            let length = shape.iter().product::<u64>() as usize * bytes;
            let tensor =
                Tensor::from_host(&device, element, &shape, &blob[offset..offset + length])
                    .unwrap();
            (fields[0].to_owned(), tensor)
        })
        .collect::<std::collections::HashMap<_, _>>();
    let w = |name: &str| &weights[name];
    let rows = read("pixels.bin").len() / (3 * 2 * 16 * 16 * 4);
    let (hidden, heads, quarter, intermediate, output) = (1024u64, 16u64, 16u64, 4096u64, 2560u64);
    let tensor = |name: &str, element: Element, shape: &[u64]| {
        Tensor::from_host(&device, element, shape, &read(name)).unwrap()
    };
    let pixels = tensor("pixels.bin", Element::f32(), &[rows as u64, 3, 2, 16, 16]);
    let coordinates = tensor("coordinates.bin", Element::i32(), &[rows as u64, 2]);
    let indices = tensor("indices.bin", Element::i32(), &[rows as u64, 4]);
    let coefficients = tensor("coefficients.bin", Element::f32(), &[rows as u64, 4]);
    let stem = qwen_vision_stem::native_for_device_with(
        &device,
        qwen_vision_stem::Elements {
            W0: Element::f16(),
            W1: Element::f16(),
            B: Element::f32(),
            PE: Element::f32(),
        },
        &NativeSpecialization::new()
            .with_static("C", 3)
            .with_static("P", 16)
            .with_static("H", hidden),
    )
    .unwrap();
    let block = qwen_vision_block::native_for_device_with(
        &device,
        qwen_vision_block::Elements {
            A: Element::f16(),
            N1W: Element::f32(),
            N1B: Element::f32(),
            QW: Element::f16(),
            QB: Element::f32(),
            PW: Element::f16(),
            PB: Element::f32(),
            N2W: Element::f32(),
            N2B: Element::f32(),
            UW: Element::f16(),
            UB: Element::f32(),
            DW: Element::f16(),
            DB: Element::f32(),
        },
        &NativeSpecialization::new()
            .with_static("H", heads)
            .with_static("P", quarter)
            .with_static("F", intermediate),
    )
    .unwrap();
    let merger = qwen_vision_merger::native_for_device_with(
        &device,
        qwen_vision_merger::Elements {
            A: Element::f16(),
            NW: Element::f32(),
            NB: Element::f32(),
            UW: Element::f16(),
            UB: Element::f32(),
            DW: Element::f16(),
            DB: Element::f32(),
        },
        &NativeSpecialization::new()
            .with_static("G", 4)
            .with_static("H", hidden)
            .with_static("D", output),
    )
    .unwrap();
    let forward = || {
        let mut state = stem
            .call(qwen_vision_stem::Args {
                pixels: &pixels,
                temporal_weight_0: w("v.patch_embd.weight"),
                temporal_weight_1: w("v.patch_embd.weight.1"),
                bias: w("v.patch_embd.bias"),
                table: w("v.position_embd.weight"),
                indices: &indices,
                coefficients: &coefficients,
            })
            .unwrap()
            .value;
        for layer in 0..24 {
            let p = |name: &str| w(&format!("v.blk.{layer}.{name}"));
            state = block
                .call(qwen_vision_block::Args {
                    hidden: &state.reshape(&[rows as u64, heads, 4, quarter]).unwrap(),
                    coordinates: &coordinates,
                    norm1_weight: p("ln1.weight"),
                    norm1_bias: p("ln1.bias"),
                    qkv_weight: p("attn_qkv.weight"),
                    qkv_bias: p("attn_qkv.bias"),
                    projection_weight: p("attn_out.weight"),
                    projection_bias: p("attn_out.bias"),
                    norm2_weight: p("ln2.weight"),
                    norm2_bias: p("ln2.bias"),
                    up_weight: p("ffn_up.weight"),
                    up_bias: p("ffn_up.bias"),
                    down_weight: p("ffn_down.weight"),
                    down_bias: p("ffn_down.bias"),
                    epsilon: EPSILON,
                })
                .unwrap()
                .value;
        }
        merger
            .call(qwen_vision_merger::Args {
                hidden: &state.reshape(&[rows as u64, hidden]).unwrap(),
                norm_weight: w("v.post_ln.weight"),
                norm_bias: w("v.post_ln.bias"),
                up_weight: w("mm.0.weight"),
                up_bias: w("mm.0.bias"),
                down_weight: w("mm.2.weight"),
                down_bias: w("mm.2.bias"),
                epsilon: EPSILON,
            })
            .unwrap()
            .value
    };
    let features = read_f32(&forward());
    let mut times = (0..5)
        .map(|_| {
            let started = Instant::now();
            forward().read_to_host().unwrap();
            started.elapsed().as_secs_f64() * 1e3
        })
        .collect::<Vec<_>>();
    times.sort_by(f64::total_cmp);
    eprintln!(
        "projector forward M {rows}: median {:.1} ms (min {:.1}), standalone calls",
        times[2], times[0]
    );
    let reference = f32s(&read("reference.bin"));
    let llama = f32s(&read("llama_policy.bin"));
    let report = |label: &str, actual: &[f32]| {
        let rms = (reference.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>()
            / reference.len() as f64)
            .sqrt();
        let error = (actual
            .iter()
            .zip(&reference)
            .map(|(a, e)| f64::from(a - e).powi(2))
            .sum::<f64>()
            / reference.len() as f64)
            .sqrt()
            / rms;
        let d = output as usize;
        let cosine = actual
            .chunks_exact(d)
            .zip(reference.chunks_exact(d))
            .map(|(a, e)| {
                let dot = a
                    .iter()
                    .zip(e)
                    .map(|(x, y)| f64::from(*x) * f64::from(*y))
                    .sum::<f64>();
                let na = a.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
                let ne = e.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
                dot / (na * ne)
            })
            .fold(1.0, f64::min);
        eprintln!("{label}: relative RMS error {error:.3e}, worst token cosine {cosine:.5}");
        (error, cosine)
    };
    let (ours, ours_cosine) = report("V4 native", &features);
    let (theirs, their_cosine) = report("llama.cpp policy (host)", &llama);
    assert!(
        ours <= 1.25 * theirs && ours_cosine >= their_cosine - 0.02,
        "features {ours:.3e} (cosine {ours_cosine:.4}) vs llama.cpp policy {theirs:.3e} ({their_cosine:.4})"
    );
}

/// Indicative time of one projector block at a real image's patch count
/// (1,200 rows: a 640 x 480 image); `cargo test --release -- --ignored`.
#[test]
#[ignore]
fn block_time_at_projector_geometry() {
    let device = device();
    let block = Block::new(1200, 16, 16, 4096, 51);
    let kernel = block.kernel(&device);
    let t = |host: &Host| host.tensor(&device);
    let (hidden, coordinates) = (t(&block.hidden), t(&block.coordinates));
    let weights = [
        &block.norm1_weight,
        &block.norm1_bias,
        &block.qkv_weight,
        &block.qkv_bias,
        &block.projection_weight,
        &block.projection_bias,
        &block.norm2_weight,
        &block.norm2_bias,
        &block.up_weight,
        &block.up_bias,
        &block.down_weight,
        &block.down_bias,
    ]
    .map(t);
    let call = || {
        kernel
            .call(qwen_vision_block::Args {
                hidden: &hidden,
                coordinates: &coordinates,
                norm1_weight: &weights[0],
                norm1_bias: &weights[1],
                qkv_weight: &weights[2],
                qkv_bias: &weights[3],
                projection_weight: &weights[4],
                projection_bias: &weights[5],
                norm2_weight: &weights[6],
                norm2_bias: &weights[7],
                up_weight: &weights[8],
                up_bias: &weights[9],
                down_weight: &weights[10],
                down_bias: &weights[11],
                epsilon: EPSILON,
            })
            .unwrap()
            .value
            .read_to_host()
            .unwrap()
    };
    call();
    let mut times = (0..10)
        .map(|_| {
            let started = Instant::now();
            call();
            started.elapsed().as_secs_f64() * 1e3
        })
        .collect::<Vec<_>>();
    times.sort_by(f64::total_cmp);
    eprintln!(
        "qwen_vision_block M 1200: median {:.2} ms (min {:.2}) per call incl. readback",
        times[5], times[0]
    );
}
