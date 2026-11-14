//! The general dense projection entries (`project_rows`, `dense_up`,
//! `short_conv_project`, `per_layer_gate`): a host reference is checked
//! against the portable bodies at reduced widths, and every native (each row
//! class and mapping) against the host reference at the catalog's real
//! widths (Gemma 4 E2B/E4B/26B, LFM2, Nemotron-H Lightning and Super).

mod common;
mod mapped;

use common::bound::{Float, Form, Library, Products, assert_within, gamma};

use magnitude_kernels::{dense_up, per_layer_gate, project_rows, short_conv_project};
use mapped::Mapped;
use seismic::BackendName;
use seismic_lang::{
    checked::{CheckedModule, SourceFile},
    entry::ElementBindings,
    failure::SourceTermination,
    interp::{Arg, Interpreter, OutcomeValue, TensorData},
    reference_math::ReferenceScalar,
    registry,
    types::DType,
};

fn module() -> CheckedModule {
    let mut sources = seismic_std::sources();
    for (path, text) in [
        (
            "feed_forward.seismic",
            include_str!("../kernels/feed_forward.seismic"),
        ),
        (
            "dense.seismic",
            include_str!("../kernels/dense.seismic"),
        ),
        (
            "short_conv.seismic",
            include_str!("../kernels/short_conv.seismic"),
        ),
        (
            "per_layer_gate.seismic",
            include_str!("../kernels/per_layer_gate.seismic"),
        ),
        (
            "functions.seismic",
            include_str!("../kernels/functions.seismic"),
        ),
    ] {
        sources.push(SourceFile {
            path: path.into(),
            text: text.into(),
        });
    }
    crate::common::check(sources)
}

enum Input {
    Tensor(DType, Vec<usize>, Vec<f64>),
    F32(f32),
    I32(i32),
}

fn floats(dtype: DType, shape: &[usize], values: &[f32]) -> Input {
    Input::Tensor(
        dtype,
        shape.to_vec(),
        values.iter().map(|v| f64::from(*v)).collect(),
    )
}

fn ints(shape: &[usize], values: &[i32]) -> Input {
    Input::Tensor(
        DType::I32,
        shape.to_vec(),
        values.iter().map(|v| f64::from(*v)).collect(),
    )
}

/// The first result of the portable body `name`.
fn interpret(
    module: &CheckedModule,
    name: &str,
    bindings: &[(&str, DType)],
    inputs: Vec<Input>,
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
        .into_iter()
        .map(|input| match input {
            Input::Tensor(dtype, shape, values) => {
                Arg::Tensor(interpreter.add_tensor(TensorData::dense(dtype, shape, values)))
            }
            Input::F32(value) => Arg::Scalar(ReferenceScalar::F32(value.to_bits())),
            Input::I32(value) => Arg::Scalar(ReferenceScalar::I32(value)),
        })
        .collect::<Vec<_>>();
    let outcome = interpreter
        .run(&arguments)
        .unwrap_or_else(|error| panic!("{name} interpreter error: {error}"));
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

fn bf16s(values: Vec<f32>) -> Vec<f32> {
    values.into_iter().map(registry::bf16_round).collect()
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
    let shape = shape.iter().map(|d| *d as u64).collect::<Vec<_>>();
    seismic::Tensor::from_host(device, element, &shape, &bytes).unwrap()
}

fn i32_tensor(device: &seismic::Device, shape: &[usize], values: &[i32]) -> seismic::Tensor {
    let bytes = values
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    let shape = shape.iter().map(|d| *d as u64).collect::<Vec<_>>();
    seismic::Tensor::from_host(device, seismic::Element::i32(), &shape, &bytes).unwrap()
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
            .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
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

fn is_cpu(device: &seismic::Device) -> bool {
    device.backend() == seismic::BackendName::Cpu
}

// ---------------------------------------------------------------------------
// Mappings.

/// One mapping parameter: entry-level, or owned by the declared launch at
/// `launch` (Metal and CUDA tune their GEMV/GEMM launches separately).
#[derive(Clone, Copy, Debug)]
struct Param {
    name: &'static str,
    launch: Option<usize>,
    value: u64,
}

const fn entry(name: &'static str, value: u64) -> Param {
    Param {
        name,
        launch: None,
        value,
    }
}

const fn scoped(name: &'static str, launch: usize, value: u64) -> Param {
    Param {
        name,
        launch: Some(launch),
        value,
    }
}

/// The declaration indices of an entry's tuned launches: Metal's GEMV,
/// batched GEMV and GEMM, CUDA's two GEMV bands (`None`: the entry declares
/// its KSPLIT for every launch) and large-band GEMM; `split` for a SPLIT
/// parameter.
#[derive(Clone, Copy)]
struct Launches {
    metal_gemv: usize,
    metal_batch: usize,
    metal_gemm: usize,
    cuda_gemv: Option<[usize; 2]>,
    cuda_gemm: usize,
    split: bool,
}

/// A mapping on one backend and whether it takes the INT8 candidate.
struct Mapping {
    params: Vec<Param>,
    int8: bool,
}

/// The mappings on `device` that `admits` holds for ([`Mapped::admits`]).
fn mappings(
    device: &seismic::Device,
    launches: Launches,
    admits: impl Fn(&Mapping) -> bool,
) -> Vec<Mapping> {
    let split = |value| launches.split.then_some(entry("SPLIT", value));
    let mappings: Vec<Mapping> = match device.backend() {
        seismic::BackendName::Metal => [
            (5, 4, 2, 32, 8, 2, 64, 64, 2),
            (9, 16, 1, 16, 4, 4, 32, 128, 1),
        ]
        .into_iter()
        .map(
            |(
                from,
                simdgroups,
                rows,
                lanes,
                batch_simdgroups,
                batch_rows,
                tile_m,
                tile_n,
                parts,
            )| Mapping {
                params: [
                    entry("BATCH_FROM", from),
                    scoped("SIMDGROUPS", launches.metal_gemv, simdgroups),
                    scoped("ROWS", launches.metal_gemv, rows),
                    scoped("LANES", launches.metal_gemv, lanes),
                    scoped("BATCH_SIMDGROUPS", launches.metal_batch, batch_simdgroups),
                    scoped("BATCH_ROWS", launches.metal_batch, batch_rows),
                    scoped("TILE_M", launches.metal_gemm, tile_m),
                    scoped("TILE_N", launches.metal_gemm, tile_n),
                ]
                .into_iter()
                .chain(split(parts))
                .collect(),
                int8: false,
            },
        )
        .collect(),
        seismic::BackendName::Cuda => [(0, 4, 2, 1), (1, 2, 1, 2), (0, 4, 4, 1)]
            .into_iter()
            .map(|(int8, ksplit, rotate, residency)| Mapping {
                params: [
                    entry("INT8", int8),
                    scoped("ROTATE", launches.cuda_gemm, rotate),
                ]
                .into_iter()
                .chain(match launches.cuda_gemv {
                    Some([first, second]) => {
                        vec![
                            scoped("KSPLIT", first, ksplit),
                            scoped("KSPLIT", second, ksplit),
                        ]
                    }
                    // The dense pair: its 1..8-row GEMV is launch 1.
                    None => vec![entry("KSPLIT", ksplit), scoped("RESIDENCY", 1, residency)],
                })
                .collect(),
                int8: int8 == 1,
            })
            .collect(),
        seismic::BackendName::Vulkan => [
            (4, 2, 32, 64, 64, 32, 32, 2),
            (16, 1, 16, 32, 128, 32, 64, 1),
        ]
        .into_iter()
        .map(
            |(simdgroups, rows, lanes, tile_m, tile_n, sub_m, sub_n, parts)| Mapping {
                params: [
                    entry("SIMDGROUPS", simdgroups),
                    entry("ROWS", rows),
                    entry("LANES", lanes),
                    entry("TILE_M", tile_m),
                    entry("TILE_N", tile_n),
                    entry("SUB_M", sub_m),
                    entry("SUB_N", sub_n),
                ]
                .into_iter()
                .chain(split(parts))
                .collect(),
                int8: false,
            },
        )
        .collect(),
        seismic::BackendName::Cpu => [(8, 0), (1, 1)]
            .into_iter()
            .map(|(rows, int8)| Mapping {
                params: vec![entry("ROWS", rows), entry("INT8", int8)],
                int8: int8 == 1,
            })
            .collect(),
    };
    mappings
        .into_iter()
        .filter(|mapping| admits(mapping))
        .collect()
}

/// The statics the device's implementation declares: backends fix different
/// subsets of the entry dimensions (the CPU forms read most of them at run time).
fn declared(
    implementation: Result<Option<seismic::OnDevice>, seismic::CheckedBundleError>,
) -> Vec<String> {
    implementation
        .unwrap()
        .expect("every test device has a native implementation")
        .statics
        .clone()
}

/// `mapping`'s specialization with those of `statics` that `declared` names.
fn specialization(
    device: &seismic::Device,
    declared: &[String],
    statics: &[(&str, usize)],
    mapping: &Mapping,
) -> seismic::NativeSpecialization {
    let specialization = statics
        .iter()
        .filter(|(name, _)| declared.iter().any(|declared| declared == name))
        .fold(
            seismic::NativeSpecialization::new(),
            |spec, (name, value)| spec.with_static(*name, *value as u64),
        );
    let specialization = if device.backend() == seismic::BackendName::Cuda {
        specialization.with_param("OPERAND", 0)
    } else {
        specialization
    };
    mapping
        .params
        .iter()
        .fold(specialization, |spec, param| match param.launch {
            Some(launch) => spec.with_launch_param(launch, param.name, param.value),
            None => spec.with_param(param.name, param.value),
        })
}

/// `project_rows`' specialization: on Metal the staged tiles of `mapping`
/// (TALL and PACK off, their launches at their defaults); the tall and PACK
/// forms are compared with them in `projection.rs`.
fn project_rows_specialization(
    device: &seismic::Device,
    statics: &[(&str, usize)],
    mapping: &Mapping,
) -> seismic::NativeSpecialization {
    let declared = declared(project_rows::native_implementation(device));
    let specialization = specialization(device, &declared, statics, mapping);
    if device.backend() != seismic::BackendName::Metal {
        return specialization;
    }
    specialization
        .with_static("WB", 0)
        .with_param("TALL", 0)
        .with_param("PACK", 0)
        .with_param("RESIDUE", 1)
        .with_launch_param(2, "BATCH_PARTS", 2)
        // Dense weights have no matrix GEMV.
        .with_launch_param(2, "MATRIX", 0)
        .with_launch_param(8, "TALL_M", 128)
        .with_launch_param(8, "TALL_K", 64)
        .with_launch_param(8, "STAGERS", 2)
        .with_launch_param(11, "PACK_TOKENS", 4)
        .with_launch_param(11, "WEIGHTS_AHEAD", 0)
}

/// Row classes: Metal GEMV / batched GEMV / small GEMM / GEMM, CUDA GEMV /
/// GEMV16 / small (split) GEMM / GEMM, Vulkan GEMV / GEMM.
const ROW_CLASSES: [usize; 6] = [1, 3, 5, 12, 40, 80];

/// Whether `rows` runs the INT8 candidate under `mapping` (CUDA: only the
/// small GEMM band takes it; the CPU: every class).
fn quantized(device: &seismic::Device, mapping: &Mapping, rows: usize) -> bool {
    mapping.int8 && (is_cpu(device) || (16 < rows))
}

// ---------------------------------------------------------------------------
// Host references.

/// Rows [M, K] through weight rows [N, K]: per output, the F64 projection
/// and its products as the bound reads them, over `envelope`, a bound on
/// each activation's magnitude as the native may hold it (threads over
/// rows).
fn project(
    x: &[f32],
    envelope: &[f64],
    weight: &[f32],
    k: usize,
    n: usize,
) -> (Vec<f64>, Vec<Products>) {
    let m = x.len() / k;
    let mut values = vec![0.0; m * n];
    let mut products = vec![Products::of::<f32, f32>(&[], &[]); m * n];
    std::thread::scope(|scope| {
        for ((row, values), products) in
            values.chunks_mut(n).enumerate().zip(products.chunks_mut(n))
        {
            let (x, envelope) = (
                &x[row * k..(row + 1) * k],
                &envelope[row * k..(row + 1) * k],
            );
            scope.spawn(move || {
                for (column, (value, product)) in
                    values.iter_mut().zip(products.iter_mut()).enumerate()
                {
                    let w = &weight[column * k..(column + 1) * k];
                    *value = x
                        .iter()
                        .zip(w)
                        .map(|(x, w)| f64::from(*x) * f64::from(*w))
                        .sum();
                    *product = Products::of(envelope, w);
                }
            });
        }
    });
    (values, products)
}

/// Each activation's magnitude, held exactly.
fn magnitudes(x: &[f32]) -> Vec<f64> {
    x.iter().map(|x| f64::from(x.abs())).collect()
}

/// Per output of [M, N], `Σ_k slack[m, k] |w[n, k]|`: how far the products
/// move when each input may move by its slack.
fn slack_terms(slack: &[f64], weight: &[f32], k: usize, n: usize) -> Vec<f64> {
    slack
        .chunks(k)
        .flat_map(|slack| {
            (0..n).map(move |column| {
                let w = &weight[column * k..(column + 1) * k];
                slack
                    .iter()
                    .zip(w)
                    .map(|(s, w)| s * f64::from(w.abs()))
                    .sum::<f64>()
            })
        })
        .collect()
}

/// The A-rounded RMS rows of `residual` rows `rows` with `norm`, and each
/// value's rounding flip: the native prologue on `backend` lands within
/// [`Library::rms`] of the exact value, this F64 host within its own F64
/// operations and its F32 cast, and both then round to A.
fn rms_rows(
    backend: BackendName,
    residual: &[f32],
    rows: &[usize],
    norm: &[f32],
    epsilon: f32,
) -> (Vec<f32>, Vec<f64>) {
    let h = norm.len();
    let unit64 = 2f64.powi(-53);
    let host = (1.0 + Float::F32.unit())
        * (1.0 + (h + 5) as f64 * unit64 / (1.0 - (h + 5) as f64 * unit64))
        - 1.0;
    let window = Library::of(backend).rms(h).max(host);
    rows.iter()
        .flat_map(|row| {
            let x = &residual[row * h..(row + 1) * h];
            let inverse = 1.0
                / (x.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / h as f64
                    + f64::from(epsilon))
                .sqrt();
            x.iter().zip(norm).map(move |(x, w)| {
                let exact = f64::from(*x) * inverse * f64::from(*w);
                (
                    registry::bf16_round(exact as f32),
                    Float::Bf16.flip(exact, window),
                )
            })
        })
        .unzip()
}

/// The activations' envelope: `|x| + slack`.
fn envelope(x: &[f32], slack: &[f64]) -> Vec<f64> {
    x.iter()
        .zip(slack)
        .map(|(x, s)| f64::from(x.abs()) + s)
        .collect()
}

fn activate(function: i32, a: f32) -> f32 {
    match function {
        0 => a / (1.0 + (-a).exp()),
        1 => {
            let u = 0.797_884_6 * (a + 0.044715 * (a * a * a));
            a / (1.0 + (-2.0 * u).exp())
        }
        2 => a.max(0.0) * a.max(0.0),
        other => panic!("activation {other}"),
    }
}

/// The GELU-tanh's Lipschitz bound: `f'(x) = (1 + tanh u) / 2 + x u'(x)
/// sech²(u) / 2` with `x u' <= 3u` (`u = √(2/π)(x + 0.044715 x³)`), and
/// `t sech²(t)` peaks at 0.447743 (t = 0.7717, where `2t tanh t = 1`), so
/// `|f'| <= 1 + 1.5 · 0.4478`.
const GELU_LIPSCHITZ: f64 = 1.0 + 1.5 * 0.4478;

/// The relative error of an F32 GELU-tanh `x / (1 + exp(-2u))` at an input
/// of magnitude at most `a` under `library`: `u`'s five roundings over
/// same-signed terms (`γ_5 |u|`), which `exp(-2u)` turns into a relative
/// `e^{2γ_5|u|} - 1`, the exponential's own error, the sum `1 + E`
/// (`E / (1 + E) <= 1`, one rounding) and the division.
fn gelu_error(library: &Library, a: f64) -> f64 {
    let u = 0.797_884_560_802_865_4 * (a + 0.044715 * a.powi(3));
    let exponential = (1.0 + library.exp) * (2.0 * gamma(5) * u).exp() - 1.0;
    let sum = (1.0 + exponential) * (1.0 + Float::F32.unit()) - 1.0;
    (1.0 + library.divide) / (1.0 - sum) - 1.0
}

/// How far an F32 value of the host's, `fl(v)`, may be from its exact `v`.
fn cast(v: f64) -> f64 {
    Float::F32.unit() * v.abs() + Float::F32.underflow()
}

/// `project_rows`' bound per output over exact sums `sums`: the form's
/// error, the second-level `scale` on the accumulator (one F32 rounding)
/// where the entry takes one, and the rounding to the published element.
fn published_bounds(
    published: DType,
    form: Form,
    sums: &[f64],
    products: &[Products],
    scale: Option<f32>,
) -> Vec<f64> {
    sums.iter()
        .zip(products)
        .map(|(sum, products)| {
            let (exact, bound) = match scale {
                Some(scale) => {
                    let exact = f64::from(scale) * sum;
                    let bound = form.exact(products) * f64::from(scale.abs());
                    (exact, Float::F32.rounded_from_exact(bound, exact))
                }
                None => (*sum, form.exact(products)),
            };
            match published {
                DType::BF16 => Float::Bf16.rounded_from_exact(bound, exact),
                _ => bound,
            }
        })
        .collect()
}

/// The INT8 candidate: the tuner's relative output-norm defect guard.
fn assert_norm(label: &str, actual: &[f32], expected: &[f32]) {
    let error = actual
        .iter()
        .zip(expected)
        .map(|(a, e)| f64::from(a - e).powi(2))
        .sum::<f64>();
    let scale = expected.iter().map(|e| f64::from(*e).powi(2)).sum::<f64>();
    assert!(
        error <= 0.05f64.powi(2) * scale,
        "{label}: relative error {}",
        (error / scale).sqrt()
    );
}

// ---------------------------------------------------------------------------
// Cases.

/// A projection of A rows (post-norm projections, the per-layer projection
/// back to D, Nemotron Super's latent down in A).
struct ProjectShape {
    name: &'static str,
    k: usize,
    n: usize,
    published: DType,
}

const PROJECT_SHAPES: [ProjectShape; 5] = [
    ProjectShape {
        name: "gemma4-e2b sliding wo",
        k: 2048,
        n: 1536,
        published: DType::F32,
    },
    ProjectShape {
        name: "gemma4-e2b full wo",
        k: 4096,
        n: 1536,
        published: DType::F32,
    },
    ProjectShape {
        name: "gemma4-26b full wo",
        k: 4096,
        n: 2816,
        published: DType::F32,
    },
    ProjectShape {
        name: "gemma4-e4b per-layer projection",
        k: 256,
        n: 2560,
        published: DType::F32,
    },
    ProjectShape {
        name: "nemotron-super latent down",
        k: 4096,
        n: 1024,
        published: DType::BF16,
    },
];

/// (name, H, F): Nemotron-H's up-only ReLU² shared experts.
const UP_SHAPES: [(&str, usize, usize); 2] = [
    ("nemotron-lightning shared", 2688, 3712),
    ("nemotron-super shared", 4096, 5376),
];

/// (name, H, CH): LFM2's short-convolution input projection.
const CONV_SHAPES: [(&str, usize, usize); 1] = [("lfm2", 2048, 2048)];

/// (name, D, L, P): Gemma 4's per-layer input gate.
const GATE_SHAPES: [(&str, usize, usize, usize); 2] =
    [("gemma4-e2b", 1536, 35, 256), ("gemma4-e4b", 2560, 42, 256)];

const EPSILON: f32 = 1e-6;

fn residual(rows: usize, h: usize, seed: u32) -> Vec<f32> {
    pattern(rows * h, seed, 2.0)
}

fn norm(h: usize) -> Vec<f32> {
    pattern(h, 5, 0.5).iter().map(|v| v + 1.0).collect()
}

fn weights(rows: usize, k: usize, seed: u32) -> Vec<f32> {
    bf16s(pattern(rows * k, seed, 0.05))
}

/// The row gathered for output o of `rows` outputs: the last rows first, so
/// the gather is not the identity.
fn out_rows(rows: usize) -> Vec<usize> {
    (0..rows).map(|o| (2 * rows - 1 - o) % (rows + 2)).collect()
}

/// A second-level scale at the magnitude Lightning's shared up weight carries.
const UP_SCALE: f32 = 0.412_5;

/// `dense_up`'s A-published product `relu²(A(scale · Σ x w))` over the
/// RMS rows, and each output's bound on `backend` when its projection runs
/// in `form`: the sum's form error and the rows' rounding flips, the scale
/// (one F32 rounding of each), the rounding to A, ReLU² (`|r̂² - r²| <=
/// Δ(2|r| + Δ)`, one F32 rounding of each), and the rounding to A.
fn dense_up_reference(
    backend: BackendName,
    form: Form,
    residual: &[f32],
    rows: &[usize],
    norm: &[f32],
    up: &[f32],
    f: usize,
    scale: f32,
) -> (Vec<f32>, Vec<f64>) {
    let h = norm.len();
    let (normalized, slack) = rms_rows(backend, residual, rows, norm, EPSILON);
    let (values, products) = project(&normalized, &envelope(&normalized, &slack), up, h, f);
    let slack = slack_terms(&slack, up, h, f);
    let factor = f64::from(scale.abs());
    values
        .iter()
        .zip(&products)
        .zip(&slack)
        .map(|((v, p), slack)| {
            let scaled = f64::from(scale) * v;
            let native = Float::F32.rounded_from_exact(factor * (form.exact(p) + slack), scaled);
            let host = Float::F32.rounded_from_exact(factor * cast(*v), scaled);
            let gate = registry::bf16_round(scale * *v as f32);
            let gate_error = Float::Bf16.rounded(native + host, gate);
            let r = f64::from(gate.max(0.0));
            let square = activate(2, gate);
            let value = registry::bf16_round(square);
            let error = Float::F32.rounded(gate_error * (2.0 * r + gate_error), square);
            (value, Float::Bf16.rounded(error, value))
        })
        .unzip()
}

/// `short_conv_project`'s F32 rows (u = b · x | c) [M, 2 CH] in F64, and
/// each one's bound on `backend` when its projections run in `form`: each
/// sum's form error and the rows' rounding flips; the product `b · x`
/// moves by `Δb |x| + Δx |b| + Δb Δx` and rounds once to F32.
fn short_conv_reference(
    backend: BackendName,
    form: Form,
    residual: &[f32],
    norm: &[f32],
    [b, c, x]: [&[f32]; 3],
    ch: usize,
) -> (Vec<f64>, Vec<f64>) {
    let h = norm.len();
    let rows = (0..residual.len() / h).collect::<Vec<_>>();
    let (normalized, slack) = rms_rows(backend, residual, &rows, norm, EPSILON);
    let held = envelope(&normalized, &slack);
    let sums = |weight: &[f32]| {
        let (values, products) = project(&normalized, &held, weight, h, ch);
        let slack = slack_terms(&slack, weight, h, ch);
        let bounds = products
            .iter()
            .zip(&slack)
            .map(|(p, s)| form.exact(p) + s)
            .collect::<Vec<_>>();
        (values, bounds)
    };
    let ((b, db), (c, dc), (x, dx)) = (sums(b), sums(c), sums(x));
    let mut values = Vec::with_capacity(rows.len() * 2 * ch);
    let mut bounds = Vec::with_capacity(rows.len() * 2 * ch);
    for row in rows {
        let at = row * ch..(row + 1) * ch;
        for i in at.clone() {
            values.push(b[i] * x[i]);
            let moved = db[i] * x[i].abs() + dx[i] * b[i].abs() + db[i] * dx[i];
            bounds.push(Float::F32.rounded_from_exact(moved, b[i] * x[i]));
        }
        values.extend_from_slice(&c[at.clone()]);
        bounds.extend_from_slice(&dc[at]);
    }
    (values, bounds)
}

/// `per_layer_gate`'s A-published `A(A(gelu(A(Σ x w))) · input)` over the
/// A-rounded hidden rows (exact on both), and each output's bound on
/// `backend` when its projection runs in `form`: the sum's form error
/// against the host's F32 sum, the rounding to A, the GELU's Lipschitz
/// bound and each side's evaluation error ([`gelu_error`]: `backend`'s
/// library, the host's CPU one), the rounding to A, the product with the
/// input (one F32 rounding of each) and the rounding to A.
fn per_layer_gate_reference(
    backend: BackendName,
    form: Form,
    hidden: &[f32],
    gate: &[f32],
    inputs: &[f32],
    d: usize,
    l: usize,
    p: usize,
    layer: usize,
) -> (Vec<f32>, Vec<f64>) {
    let rounded = bf16s(hidden.to_vec());
    let (values, products) = project(&rounded, &magnitudes(&rounded), gate, d, p);
    let (native, host) = (Library::of(backend), Library::of(BackendName::Cpu));
    values
        .iter()
        .zip(&products)
        .enumerate()
        .map(|(flat, (v, products))| {
            let (row, column) = (flat / p, flat % p);
            let input = inputs[(row * l + layer) * p + column];
            let gate = registry::bf16_round(*v as f32);
            let gate_error = Float::Bf16.rounded(form.exact(products) + cast(*v), gate);
            let reach = f64::from(gate.abs()) + gate_error;
            let host_error = gelu_error(&host, f64::from(gate.abs()));
            let exact = f64::from(activate(1, gate).abs()) / (1.0 - host_error);
            let activated_error = GELU_LIPSCHITZ * gate_error
                + gelu_error(&native, reach) * (exact + GELU_LIPSCHITZ * gate_error)
                + host_error * exact;
            let activated = registry::bf16_round(activate(1, gate));
            let activated_error = Float::Bf16.rounded(activated_error, activated);
            let product = activated * input;
            let value = registry::bf16_round(product);
            let error = Float::F32.rounded(activated_error * f64::from(input.abs()), product);
            (value, Float::Bf16.rounded(error, value))
        })
        .unzip()
}

// ---------------------------------------------------------------------------
// The host reference is the contract: portable bodies at reduced widths.

#[test]
fn portable_entries_match_host_reference() {
    let module = module();
    let (m, k, n) = (3, 96, 40);
    let x = bf16s(pattern(m * k, 11, 1.5));
    let w = weights(n, k, 13);
    for published in [DType::F32, DType::BF16] {
        let portable = interpret(
            &module,
            "project_rows",
            &[("A", DType::BF16), ("W", DType::BF16), ("Y", published)],
            vec![
                floats(DType::BF16, &[m, k], &x),
                floats(DType::BF16, &[n, k], &w),
                floats(DType::F32, &[0], &[]),
                // No packing bound.
                floats(DType::F32, &[0], &[]),
            ],
        );
        let (expected, products) = project(&x, &magnitudes(&x), &w, k, n);
        let bound = published_bounds(published, Form::F32, &expected, &products, None);
        assert_within(
            &format!("portable project_rows {published:?}"),
            &portable,
            &expected,
            &bound,
        );
    }

    let (h, f) = (64, 48);
    let rows = out_rows(m);
    let (x, norm, up) = (residual(m + 2, h, 17), norm(h), weights(f, h, 19));
    let portable = interpret(
        &module,
        "dense_up",
        &[("NW", DType::F32), ("UW", DType::BF16), ("A", DType::BF16)],
        vec![
            floats(DType::F32, &[m + 2, h], &x),
            floats(DType::F32, &[h], &norm),
            floats(DType::BF16, &[f, h], &up),
            ints(&[m], &rows.iter().map(|r| *r as i32).collect::<Vec<_>>()),
            Input::F32(EPSILON),
            Input::I32(2),
            floats(DType::F32, &[1], &[UP_SCALE]),
        ],
    );
    let (expected, bound) = dense_up_reference(
        BackendName::Cpu,
        Form::F32,
        &x,
        &rows,
        &norm,
        &up,
        f,
        UP_SCALE,
    );
    assert_within("portable dense_up", &portable, &expected, &bound);

    let ch = 40;
    let (b, c, xw) = (weights(ch, h, 23), weights(ch, h, 29), weights(ch, h, 31));
    let x = residual(m, h, 37);
    let portable = interpret(
        &module,
        "short_conv_project",
        &[
            ("NW", DType::F32),
            ("BW", DType::BF16),
            ("CW", DType::BF16),
            ("XW", DType::BF16),
            ("A", DType::BF16),
        ],
        vec![
            floats(DType::F32, &[m, h], &x),
            floats(DType::F32, &[h], &norm),
            floats(DType::BF16, &[ch, h], &b),
            floats(DType::BF16, &[ch, h], &c),
            floats(DType::BF16, &[ch, h], &xw),
            Input::F32(EPSILON),
            floats(DType::F32, &[0], &[]),
            floats(DType::F32, &[0], &[]),
            floats(DType::F32, &[0], &[]),
        ],
    );
    let (expected, bound) =
        short_conv_reference(BackendName::Cpu, Form::F32, &x, &norm, [&b, &c, &xw], ch);
    assert_within("portable short_conv_project", &portable, &expected, &bound);

    let (d, l, p, layer) = (64, 3, 32, 2);
    let (hidden, gate, inputs) = (
        residual(m, d, 41),
        weights(p, d, 43),
        pattern(m * l * p, 47, 1.0),
    );
    let portable = interpret(
        &module,
        "per_layer_gate",
        &[("GW", DType::BF16), ("A", DType::BF16)],
        vec![
            floats(DType::F32, &[m, d], &hidden),
            floats(DType::BF16, &[p, d], &gate),
            floats(DType::F32, &[m, l, p], &inputs),
            Input::I32(layer as i32),
            Input::I32(1),
            floats(DType::F32, &[0], &[]),
        ],
    );
    let (expected, bound) = per_layer_gate_reference(
        BackendName::Cpu,
        Form::F32,
        &hidden,
        &gate,
        &inputs,
        d,
        l,
        p,
        layer,
    );
    assert_within("portable per_layer_gate", &portable, &expected, &bound);
}

// ---------------------------------------------------------------------------
// Natives at the catalog's widths.

#[test]
fn native_project_rows_matches_host_reference() {
    let launches = Launches {
        metal_gemv: 0,
        metal_batch: 2,
        metal_gemm: 5,
        cuda_gemv: Some([1, 2]),
        cuda_gemm: 4,
        split: true,
    };
    let devices = devices();
    for shape in PROJECT_SHAPES {
        let (k, n) = (shape.k, shape.n);
        let w = weights(n, k, 3);
        for rows in ROW_CLASSES {
            let x = bf16s(pattern(rows * k, 7 + rows as u32, 1.5));
            let (unscaled, products) = project(&x, &magnitudes(&x), &w, k, n);
            // No scale port (WS = 0), and a weight's second-level scale on
            // the accumulator (WS = 1): the reference scales the exact
            // projection.
            for scale in [None, Some(UP_SCALE)] {
                let factor = f64::from(scale.unwrap_or(1.0));
                let expected = unscaled
                    .iter()
                    .map(|value| value * factor)
                    .collect::<Vec<_>>();
                let extent = usize::from(scale.is_some());
                let projection = Mapped::new(
                    project_rows::native_entry_with(project_rows::Elements {
                        A: seismic::Element::bf16(),
                        W: seismic::Element::bf16(),
                        Y: if shape.published == DType::F32 {
                            seismic::Element::f32()
                        } else {
                            seismic::Element::bf16()
                        },
                    }),
                    |device: &seismic::Device, mapping: &Mapping| {
                        project_rows_specialization(
                            device,
                            &[("G", 1), ("K", k), ("N", n)],
                            mapping,
                        )
                        .with_static("WS", extent as u64)
                    },
                );
                for device in &devices {
                    let form = Form::projection(
                        device.backend(),
                        rows,
                        Float::Bf16,
                        seismic::Element::bf16(),
                    );
                    let bound =
                        published_bounds(shape.published, form, &unscaled, &products, scale);
                    let (source, weight) = (
                        tensor(device, DType::BF16, &[rows, k], &x),
                        tensor(device, DType::BF16, &[n, k], &w),
                    );
                    let weight_scale = tensor(
                        device,
                        DType::F32,
                        &[extent],
                        &scale.into_iter().collect::<Vec<_>>(),
                    );
                    for mapping in mappings(device, launches, |mapping| {
                        projection.admits(device, mapping)
                    }) {
                        let label = format!(
                            "{:?} {} rows {rows} scale {scale:?} {:?}",
                            device.backend(),
                            shape.name,
                            mapping.params
                        );
                        let result = projection
                            .kernel(device, &mapping)
                            .call(project_rows::Args {
                                source: &source,
                                weight: &weight,
                                weight_scale: &weight_scale,
                                weight_bound: &tensor(device, DType::F32, &[0], &[]),
                            })
                            .unwrap()
                            .value;
                        let actual = read(&result, shape.published);
                        if quantized(device, &mapping, rows) {
                            let expected = expected.iter().map(|v| *v as f32).collect::<Vec<_>>();
                            assert_norm(&label, &actual, &expected);
                        } else {
                            assert_within(&label, &actual, &expected, &bound);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn native_dense_up_matches_host_reference() {
    let launches = Launches {
        metal_gemv: 1,
        metal_batch: 2,
        metal_gemm: 4,
        cuda_gemv: None,
        cuda_gemm: 4,
        split: false,
    };
    let devices = devices();
    for (name, h, f) in UP_SHAPES {
        let (norm, up) = (norm(h), weights(f, h, 3));
        let expansion = Mapped::new(
            dense_up::native_entry_with(dense_up::Elements {
                NW: seismic::Element::f32(),
                UW: seismic::Element::bf16(),
                A: seismic::Element::bf16(),
            }),
            |device: &seismic::Device, mapping: &Mapping| {
                specialization(
                    device,
                    &declared(dense_up::native_implementation(device)),
                    &[("H", h), ("F", f)],
                    mapping,
                )
                .with_static("US", 1)
            },
        );
        // Unscaled (dense weights), and an NVFP4 weight's second-level scales.
        for (outputs, scale) in ROW_CLASSES
            .iter()
            .flat_map(|rows| [(*rows, 1.0), (*rows, UP_SCALE)])
        {
            let (x, rows) = (
                residual(outputs + 2, h, 9 + outputs as u32),
                out_rows(outputs),
            );
            for device in devices.iter() {
                let form = Form::projection(
                    device.backend(),
                    outputs,
                    Float::Bf16,
                    seismic::Element::bf16(),
                );
                let (expected, bound) =
                    dense_up_reference(device.backend(), form, &x, &rows, &norm, &up, f, scale);
                let (residual, norm_tensor, up_tensor) = (
                    tensor(device, DType::F32, &[outputs + 2, h], &x),
                    tensor(device, DType::F32, &[h], &norm),
                    tensor(device, DType::BF16, &[f, h], &up),
                );
                let out_rows = i32_tensor(
                    device,
                    &[outputs],
                    &rows.iter().map(|r| *r as i32).collect::<Vec<_>>(),
                );
                for mapping in mappings(device, launches, |mapping| {
                    expansion.admits(device, mapping)
                }) {
                    let label = format!(
                        "{:?} {name} rows {outputs} scale {scale} {:?}",
                        device.backend(),
                        mapping.params
                    );
                    let result = expansion
                        .kernel(device, &mapping)
                        .call(dense_up::Args {
                            residual: &residual,
                            norm: &norm_tensor,
                            up_weight: &up_tensor,
                            out_rows: &out_rows,
                            eps: EPSILON,
                            activation: 2,
                            up_scale: &tensor(device, DType::F32, &[1], &[scale]),
                        })
                        .unwrap()
                        .value;
                    let actual = read(&result, DType::BF16);
                    if quantized(device, &mapping, outputs) {
                        assert_norm(&label, &actual, &expected);
                    } else {
                        assert_within(&label, &actual, &expected, &bound);
                    }
                }
            }
        }
    }
}

#[test]
fn native_short_conv_project_matches_host_reference() {
    let launches = Launches {
        metal_gemv: 1,
        metal_batch: 2,
        metal_gemm: 4,
        cuda_gemv: Some([1, 2]),
        cuda_gemm: 4,
        split: false,
    };
    let devices = devices();
    for (name, h, ch) in CONV_SHAPES {
        let (norm, b, c, xw) = (
            norm(h),
            weights(ch, h, 3),
            weights(ch, h, 5),
            weights(ch, h, 7),
        );
        let projection = Mapped::new(
            short_conv_project::native_entry_with(short_conv_project::Elements {
                NW: seismic::Element::f32(),
                BW: seismic::Element::bf16(),
                CW: seismic::Element::bf16(),
                XW: seismic::Element::bf16(),
                A: seismic::Element::bf16(),
            }),
            |device: &seismic::Device, mapping: &Mapping| {
                specialization(
                    device,
                    &declared(short_conv_project::native_implementation(device)),
                    &[("H", h), ("CH", ch)],
                    mapping,
                )
                .with_static("BS", 0)
                .with_static("CS", 0)
                .with_static("XS", 0)
            },
        );
        for rows in ROW_CLASSES {
            let x = residual(rows, h, 11 + rows as u32);
            for device in devices.iter() {
                let form = Form::projection(
                    device.backend(),
                    rows,
                    Float::Bf16,
                    seismic::Element::bf16(),
                );
                let (expected, bound) =
                    short_conv_reference(device.backend(), form, &x, &norm, [&b, &c, &xw], ch);
                let bf16 = |values: &[f32]| tensor(device, DType::BF16, &[ch, h], values);
                let (b_tensor, c_tensor, x_tensor) = (bf16(&b), bf16(&c), bf16(&xw));
                let (residual, norm_tensor) = (
                    tensor(device, DType::F32, &[rows, h], &x),
                    tensor(device, DType::F32, &[h], &norm),
                );
                for mapping in mappings(device, launches, |mapping| {
                    projection.admits(device, mapping)
                }) {
                    let label = format!(
                        "{:?} {name} rows {rows} {:?}",
                        device.backend(),
                        mapping.params
                    );
                    let result = projection
                        .kernel(device, &mapping)
                        .call(short_conv_project::Args {
                            residual: &residual,
                            norm: &norm_tensor,
                            b_weight: &b_tensor,
                            c_weight: &c_tensor,
                            x_weight: &x_tensor,
                            eps: EPSILON,
                            b_scale: &tensor(device, DType::F32, &[0], &[]),
                            c_scale: &tensor(device, DType::F32, &[0], &[]),
                            x_scale: &tensor(device, DType::F32, &[0], &[]),
                        })
                        .unwrap()
                        .value;
                    let actual = read(&result, DType::F32);
                    if quantized(device, &mapping, rows) {
                        let expected = expected.iter().map(|v| *v as f32).collect::<Vec<_>>();
                        assert_norm(&label, &actual, &expected);
                    } else {
                        assert_within(&label, &actual, &expected, &bound);
                    }
                }
            }
        }
    }
}

#[test]
fn native_per_layer_gate_matches_host_reference() {
    let launches = Launches {
        metal_gemv: 1,
        metal_batch: 2,
        metal_gemm: 4,
        cuda_gemv: Some([1, 2]),
        cuda_gemm: 4,
        split: false,
    };
    let devices = devices();
    for (name, d, l, p) in GATE_SHAPES {
        let gate = weights(p, d, 3);
        let layer = l - 2;
        let gating = Mapped::new(
            per_layer_gate::native_entry_with(per_layer_gate::Elements {
                GW: seismic::Element::bf16(),
                A: seismic::Element::bf16(),
            }),
            |device: &seismic::Device, mapping: &Mapping| {
                specialization(
                    device,
                    &declared(per_layer_gate::native_implementation(device)),
                    &[("D", d), ("L", l), ("P", p)],
                    mapping,
                )
                .with_static("GS", 0)
            },
        );
        for rows in ROW_CLASSES {
            let (hidden, inputs) = (
                residual(rows, d, 13 + rows as u32),
                pattern(rows * l * p, 17, 1.0),
            );
            for device in devices.iter() {
                let form = Form::projection(
                    device.backend(),
                    rows,
                    Float::Bf16,
                    seismic::Element::bf16(),
                );
                let (expected, bound) = per_layer_gate_reference(
                    device.backend(),
                    form,
                    &hidden,
                    &gate,
                    &inputs,
                    d,
                    l,
                    p,
                    layer,
                );
                let (hidden_tensor, gate_tensor, inputs_tensor) = (
                    tensor(device, DType::F32, &[rows, d], &hidden),
                    tensor(device, DType::BF16, &[p, d], &gate),
                    tensor(device, DType::F32, &[rows, l, p], &inputs),
                );
                for mapping in mappings(device, launches, |mapping| gating.admits(device, mapping))
                {
                    let label = format!(
                        "{:?} {name} rows {rows} {:?}",
                        device.backend(),
                        mapping.params
                    );
                    let result = gating
                        .kernel(device, &mapping)
                        .call(per_layer_gate::Args {
                            hidden: &hidden_tensor,
                            gate_weight: &gate_tensor,
                            inputs: &inputs_tensor,
                            layer: layer as i32,
                            activation: 1,
                            gate_scale: &tensor(device, DType::F32, &[0], &[]),
                        })
                        .unwrap()
                        .value;
                    let actual = read(&result, DType::BF16);
                    if quantized(device, &mapping, rows) {
                        assert_norm(&label, &actual, &expected);
                    } else {
                        assert_within(&label, &actual, &expected, &bound);
                    }
                }
            }
        }
    }
}
