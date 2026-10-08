//! The general dense projection entries (`project_rows`, `dense_up`,
//! `short_conv_project`, `per_layer_gate`): a host reference is checked
//! against the portable bodies at reduced widths, and every native (each row
//! class and mapping) against the host reference at the catalog's real
//! widths (Gemma 4 E2B/E4B/26B, LFM2, Nemotron-H Lightning and Super).

use magnitude_kernels::{dense_up, per_layer_gate, project_rows, short_conv_project};
use seismic_lang::{
    checked::{check_source, CheckedModule, SourceFile},
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
            "dense_rows.seismic",
            include_str!("../kernels/dense_rows.seismic"),
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
    check_source(sources).unwrap()
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

/// CUDA, else Metal; `SEISMIC_TEST_BACKEND=vulkan` selects Vulkan. Then the
/// CPU.
fn devices() -> Vec<seismic::Device> {
    let catalog = seismic::DeviceCatalog::discover().unwrap();
    let gpu = match std::env::var("SEISMIC_TEST_BACKEND").ok().as_deref() {
        Some("vulkan") => Some(
            catalog
                .open_backend(seismic::BackendName::Vulkan)
                .expect("the Vulkan device opens"),
        ),
        Some(other) => panic!("SEISMIC_TEST_BACKEND={other}: only vulkan is selectable"),
        None => [seismic::BackendName::Cuda, seismic::BackendName::Metal]
            .into_iter()
            .find_map(|backend| catalog.open_backend(backend).ok()),
    };
    gpu.into_iter()
        .chain(std::iter::once(
            catalog.open_backend(seismic::BackendName::Cpu).unwrap(),
        ))
        .collect()
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
/// batched GEMV and GEMM, CUDA's two GEMV bands and large-band GEMM; `split`
/// for a SPLIT parameter.
#[derive(Clone, Copy)]
struct Launches {
    metal_gemv: usize,
    metal_batch: usize,
    metal_gemm: usize,
    cuda_gemv: [usize; 2],
    cuda_gemm: usize,
    split: bool,
}

/// A mapping on one backend and whether it takes the INT8 candidate.
struct Mapping {
    params: Vec<Param>,
    int8: bool,
}

fn mappings(device: &seismic::Device, launches: Launches) -> Vec<Mapping> {
    let split = |value| launches.split.then_some(entry("SPLIT", value));
    match device.backend() {
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
        seismic::BackendName::Cuda => [(0, 4, 2), (1, 2, 1), (0, 4, 4)]
            .into_iter()
            .map(|(int8, ksplit, rotate)| Mapping {
                params: vec![
                    entry("INT8", int8),
                    scoped("KSPLIT", launches.cuda_gemv[0], ksplit),
                    scoped("KSPLIT", launches.cuda_gemv[1], ksplit),
                    scoped("ROTATE", launches.cuda_gemm, rotate),
                ],
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
    }
}

fn specialization(
    device: &seismic::Device,
    statics: &[(&str, usize)],
    mapping: &Mapping,
) -> seismic::NativeSpecialization {
    let statics = if is_cpu(device) { &[][..] } else { statics };
    let specialization = statics.iter().fold(
        seismic::NativeSpecialization::new(),
        |spec, (name, value)| spec.with_static(*name, *value as u64),
    );
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
    let specialization = specialization(device, statics, mapping);
    if device.backend() != seismic::BackendName::Metal {
        return specialization;
    }
    specialization
        .with_param("TALL", 0)
        .with_param("PACK", 0)
        .with_launch_param(7, "TALL_M", 128)
        .with_launch_param(7, "TALL_K", 64)
        .with_launch_param(7, "STAGERS", 2)
        .with_launch_param(10, "PACK_TOKENS", 4)
        .with_launch_param(10, "WEIGHTS_AHEAD", 0)
}

/// Row classes: Metal GEMV / batched GEMV / small GEMM / GEMM, CUDA GEMV /
/// GEMV16 / small (split) GEMM / GEMM, Vulkan GEMV / GEMM.
const ROW_CLASSES: [usize; 6] = [1, 3, 5, 12, 40, 80];

/// Whether `rows` runs the INT8 candidate under `mapping` (CUDA: only the
/// small GEMM band takes it; the CPU: every class).
fn quantized(device: &seismic::Device, mapping: &Mapping, rows: usize) -> bool {
    mapping.int8 && (is_cpu(device) || (16 < rows && rows <= 64))
}

// ---------------------------------------------------------------------------
// Host references.

/// Rows [M, K] through weight rows [N, K]: per output, the F64 projection and
/// the magnitude sum |x| |w| (threads over rows).
fn project(x: &[f32], weight: &[f32], k: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    let m = x.len() / k;
    let mut values = vec![0.0; m * n];
    let mut magnitudes = vec![0.0; m * n];
    std::thread::scope(|scope| {
        for ((row, values), magnitudes) in values
            .chunks_mut(n)
            .enumerate()
            .zip(magnitudes.chunks_mut(n))
        {
            let x = &x[row * k..(row + 1) * k];
            scope.spawn(move || {
                for (column, (value, magnitude)) in
                    values.iter_mut().zip(magnitudes.iter_mut()).enumerate()
                {
                    let w = &weight[column * k..(column + 1) * k];
                    *value = x
                        .iter()
                        .zip(w)
                        .map(|(x, w)| f64::from(*x) * f64::from(*w))
                        .sum();
                    *magnitude = x
                        .iter()
                        .zip(w)
                        .map(|(x, w)| (f64::from(*x) * f64::from(*w)).abs())
                        .sum();
                }
            });
        }
    });
    (values, magnitudes)
}

/// The A-rounded RMS rows of `residual` rows `rows` with `norm`.
fn rms_rows(residual: &[f32], rows: &[usize], norm: &[f32], epsilon: f32) -> Vec<f32> {
    let h = norm.len();
    rows.iter()
        .flat_map(|row| {
            let x = &residual[row * h..(row + 1) * h];
            let inverse = 1.0
                / (x.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / h as f64
                    + f64::from(epsilon))
                .sqrt();
            x.iter().zip(norm).map(move |(x, w)| {
                registry::bf16_round((f64::from(*x) * inverse * f64::from(*w)) as f32)
            })
        })
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

/// Every F32 output within `tolerance` of its terms' magnitude sum (plus
/// `rounding` of the reference, for a published element).
fn assert_terms(
    label: &str,
    actual: &[f32],
    expected: &[f64],
    magnitudes: &[f64],
    tolerance: f64,
    rounding: f64,
) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    for (index, ((actual, expected), magnitude)) in
        actual.iter().zip(expected).zip(magnitudes).enumerate()
    {
        let bound = tolerance * magnitude + rounding * expected.abs() + 1e-30;
        assert!(
            (f64::from(*actual) - expected).abs() <= bound,
            "{label}[{index}]: native {actual}, reference {expected} (bound {bound})"
        );
    }
}

/// A-published results agree within `relative * |reference| +
/// absolute_fraction * max |reference|`.
fn assert_near(
    label: &str,
    actual: &[f32],
    expected: &[f32],
    relative: f32,
    absolute_fraction: f32,
) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    let scale = expected
        .iter()
        .fold(0.0f32, |max, value| max.max(value.abs()));
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(actual.is_finite(), "{label}[{index}]: {actual}");
        assert!(
            (actual - expected).abs() <= relative * expected.abs() + absolute_fraction * scale,
            "{label}[{index}]: native {actual}, reference {expected} (scale {scale})"
        );
    }
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

fn dense_up_reference(
    residual: &[f32],
    rows: &[usize],
    norm: &[f32],
    up: &[f32],
    f: usize,
    scale: f32,
) -> Vec<f32> {
    let normalized = rms_rows(residual, rows, norm, EPSILON);
    let (values, _) = project(&normalized, up, norm.len(), f);
    values
        .iter()
        .map(|v| registry::bf16_round(activate(2, registry::bf16_round(scale * *v as f32))))
        .collect()
}

/// (u | c rows [M, 2 CH], their magnitude bounds).
fn short_conv_reference(
    residual: &[f32],
    norm: &[f32],
    [b, c, x]: [&[f32]; 3],
    ch: usize,
) -> (Vec<f64>, Vec<f64>) {
    let h = norm.len();
    let rows = (0..residual.len() / h).collect::<Vec<_>>();
    let normalized = rms_rows(residual, &rows, norm, EPSILON);
    let ((b, b_terms), (c, c_terms), (x, x_terms)) = (
        project(&normalized, b, h, ch),
        project(&normalized, c, h, ch),
        project(&normalized, x, h, ch),
    );
    let mut values = Vec::with_capacity(rows.len() * 2 * ch);
    let mut magnitudes = Vec::with_capacity(rows.len() * 2 * ch);
    for row in rows {
        let at = row * ch..(row + 1) * ch;
        for i in at.clone() {
            values.push(b[i] * x[i]);
            // The product of two perturbed sums: |b| dx + |x| db (+ db dx).
            magnitudes
                .push(b_terms[i] * x[i].abs() + x_terms[i] * b[i].abs() + b_terms[i] * x_terms[i]);
        }
        values.extend_from_slice(&c[at.clone()]);
        magnitudes.extend_from_slice(&c_terms[at]);
    }
    (values, magnitudes)
}

fn per_layer_gate_reference(
    hidden: &[f32],
    gate: &[f32],
    inputs: &[f32],
    d: usize,
    l: usize,
    p: usize,
    layer: usize,
) -> Vec<f32> {
    let rounded = bf16s(hidden.to_vec());
    let (values, _) = project(&rounded, gate, d, p);
    values
        .iter()
        .enumerate()
        .map(|(flat, v)| {
            let (row, column) = (flat / p, flat % p);
            let activated = registry::bf16_round(activate(1, registry::bf16_round(*v as f32)));
            registry::bf16_round(activated * inputs[(row * l + layer) * p + column])
        })
        .collect()
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
            ],
        );
        let (expected, magnitudes) = project(&x, &w, k, n);
        let rounding = if published == DType::F32 {
            0.0
        } else {
            1.0 / 256.0
        };
        assert_terms(
            &format!("portable project_rows {published:?}"),
            &portable,
            &expected,
            &magnitudes,
            1e-6,
            rounding,
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
    let expected = dense_up_reference(&x, &rows, &norm, &up, f, UP_SCALE);
    assert_near("portable dense_up", &portable, &expected, 1e-2, 1e-3);

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
    let (expected, magnitudes) = short_conv_reference(&x, &norm, [&b, &c, &xw], ch);
    assert_terms(
        "portable short_conv_project",
        &portable,
        &expected,
        &magnitudes,
        1e-5,
        0.0,
    );

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
    let expected = per_layer_gate_reference(&hidden, &gate, &inputs, d, l, p, layer);
    assert_near("portable per_layer_gate", &portable, &expected, 1e-2, 1e-3);
}

// ---------------------------------------------------------------------------
// Natives at the catalog's widths.

#[test]
fn native_project_rows_matches_host_reference() {
    let launches = Launches {
        metal_gemv: 0,
        metal_batch: 1,
        metal_gemm: 4,
        cuda_gemv: [1, 2],
        cuda_gemm: 4,
        split: true,
    };
    let devices = devices();
    for shape in PROJECT_SHAPES {
        let (k, n) = (shape.k, shape.n);
        let w = weights(n, k, 3);
        let rounding = if shape.published == DType::F32 {
            0.0
        } else {
            1.0 / 256.0
        };
        for rows in ROW_CLASSES {
            let x = bf16s(pattern(rows * k, 7 + rows as u32, 1.5));
            let (unscaled, unscaled_magnitudes) = project(&x, &w, k, n);
            // No scale port (WS = 0), and a weight's second-level scale on
            // the accumulator (WS = 1): the reference scales the exact
            // projection.
            for scale in [None, Some(UP_SCALE)] {
                let factor = f64::from(scale.unwrap_or(1.0));
                let expected = unscaled
                    .iter()
                    .map(|value| value * factor)
                    .collect::<Vec<_>>();
                let magnitudes = unscaled_magnitudes
                    .iter()
                    .map(|value| value * factor)
                    .collect::<Vec<_>>();
                let extent = usize::from(scale.is_some());
                for device in &devices {
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
                    for mapping in mappings(device, launches) {
                        let label = format!(
                            "{:?} {} rows {rows} scale {scale:?} {:?}",
                            device.backend(),
                            shape.name,
                            mapping.params
                        );
                        let result = project_rows::native_for_device_with(
                            device,
                            project_rows::Elements {
                                A: seismic::Element::bf16(),
                                W: seismic::Element::bf16(),
                                Y: if shape.published == DType::F32 {
                                    seismic::Element::f32()
                                } else {
                                    seismic::Element::bf16()
                                },
                            },
                            &project_rows_specialization(device, &[("K", k), ("N", n)], &mapping)
                                .with_static("WS", extent as u64),
                        )
                        .unwrap()
                        .call(project_rows::Args {
                            source: &source,
                            weight: &weight,
                            weight_scale: &weight_scale,
                        })
                        .unwrap()
                        .value;
                        let actual = read(&result, shape.published);
                        if quantized(device, &mapping, rows) {
                            let expected = expected.iter().map(|v| *v as f32).collect::<Vec<_>>();
                            assert_norm(&label, &actual, &expected);
                        } else {
                            assert_terms(&label, &actual, &expected, &magnitudes, 1e-3, rounding);
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
        cuda_gemv: [2, 3],
        cuda_gemm: 5,
        split: false,
    };
    let devices = devices();
    for (name, h, f) in UP_SHAPES {
        let (norm, up) = (norm(h), weights(f, h, 3));
        // Unscaled (dense weights), and an NVFP4 weight's second-level scales.
        for (outputs, scale) in ROW_CLASSES
            .iter()
            .flat_map(|rows| [(*rows, 1.0), (*rows, UP_SCALE)])
        {
            let (x, rows) = (
                residual(outputs + 2, h, 9 + outputs as u32),
                out_rows(outputs),
            );
            let expected = dense_up_reference(&x, &rows, &norm, &up, f, scale);
            for device in devices.iter() {
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
                for mapping in mappings(device, launches) {
                    let label = format!(
                        "{:?} {name} rows {outputs} scale {scale} {:?}",
                        device.backend(),
                        mapping.params
                    );
                    let result = dense_up::native_for_device_with(
                        device,
                        dense_up::Elements {
                            NW: seismic::Element::f32(),
                            UW: seismic::Element::bf16(),
                            A: seismic::Element::bf16(),
                        },
                        &specialization(device, &[("H", h), ("F", f)], &mapping)
                            .with_static("US", 1),
                    )
                    .unwrap()
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
                        assert_near(&label, &actual, &expected, 2e-2, 2e-3);
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
        cuda_gemv: [2, 3],
        cuda_gemm: 5,
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
        for rows in ROW_CLASSES {
            let x = residual(rows, h, 11 + rows as u32);
            let (expected, magnitudes) = short_conv_reference(&x, &norm, [&b, &c, &xw], ch);
            for device in devices.iter() {
                let bf16 = |values: &[f32]| tensor(device, DType::BF16, &[ch, h], values);
                let (b_tensor, c_tensor, x_tensor) = (bf16(&b), bf16(&c), bf16(&xw));
                let (residual, norm_tensor) = (
                    tensor(device, DType::F32, &[rows, h], &x),
                    tensor(device, DType::F32, &[h], &norm),
                );
                for mapping in mappings(device, launches) {
                    let label = format!(
                        "{:?} {name} rows {rows} {:?}",
                        device.backend(),
                        mapping.params
                    );
                    let result = short_conv_project::native_for_device_with(
                        device,
                        short_conv_project::Elements {
                            NW: seismic::Element::f32(),
                            BW: seismic::Element::bf16(),
                            CW: seismic::Element::bf16(),
                            XW: seismic::Element::bf16(),
                            A: seismic::Element::bf16(),
                        },
                        &specialization(device, &[("H", h), ("CH", ch)], &mapping)
                            .with_static("BS", 0)
                            .with_static("CS", 0)
                            .with_static("XS", 0),
                    )
                    .unwrap()
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
                        assert_terms(&label, &actual, &expected, &magnitudes, 1e-3, 0.0);
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
        cuda_gemv: [2, 3],
        cuda_gemm: 5,
        split: false,
    };
    let devices = devices();
    for (name, d, l, p) in GATE_SHAPES {
        let gate = weights(p, d, 3);
        let layer = l - 2;
        for rows in ROW_CLASSES {
            let (hidden, inputs) = (
                residual(rows, d, 13 + rows as u32),
                pattern(rows * l * p, 17, 1.0),
            );
            let expected = per_layer_gate_reference(&hidden, &gate, &inputs, d, l, p, layer);
            for device in devices.iter() {
                let (hidden_tensor, gate_tensor, inputs_tensor) = (
                    tensor(device, DType::F32, &[rows, d], &hidden),
                    tensor(device, DType::BF16, &[p, d], &gate),
                    tensor(device, DType::F32, &[rows, l, p], &inputs),
                );
                for mapping in mappings(device, launches) {
                    let label = format!(
                        "{:?} {name} rows {rows} {:?}",
                        device.backend(),
                        mapping.params
                    );
                    let result = per_layer_gate::native_for_device_with(
                        device,
                        per_layer_gate::Elements {
                            GW: seismic::Element::bf16(),
                            A: seismic::Element::bf16(),
                        },
                        &specialization(device, &[("D", d), ("L", l), ("P", p)], &mapping)
                            .with_static("GS", 0),
                    )
                    .unwrap()
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
                        assert_near(&label, &actual, &expected, 2e-2, 2e-3);
                    }
                }
            }
        }
    }
}
