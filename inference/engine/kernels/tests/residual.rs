//! Residual row ops (`residual.seismic`): the portable bodies define the
//! contract; native implementations are checked against them on the host's
//! GPU backend and on the CPU, at the catalog's real row widths.

use magnitude_kernels::{moe_tail, per_layer_inputs, post_norm_residual};
use seismic_lang::{
    checked::{check_source, CheckedModule, SourceFile},
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
        path: "residual.seismic".into(),
        text: include_str!("../kernels/residual.seismic").into(),
    });
    check_source(sources).unwrap()
}

enum Input {
    Tensor(DType, Vec<usize>, Vec<f64>),
    F32(f32),
}

fn floats(dtype: DType, shape: &[usize], values: &[f32]) -> Input {
    Input::Tensor(dtype, shape.to_vec(), values.iter().map(|v| f64::from(*v)).collect())
}

fn ints(shape: &[usize], values: &[i32]) -> Input {
    Input::Tensor(DType::I32, shape.to_vec(), values.iter().map(|v| f64::from(*v)).collect())
}

fn interpret(module: &CheckedModule, name: &str, bindings: &[(&str, DType)], inputs: Vec<Input>) -> Vec<f64> {
    let elements = bindings
        .iter()
        .fold(ElementBindings::new(), |elements, (name, dtype)| elements.bind(name, registry::dense(*dtype)));
    let logical = module.entry(module.entry_named(name).unwrap(), &elements).unwrap();
    let mut interpreter = Interpreter::new(&logical);
    let arguments = inputs
        .into_iter()
        .map(|input| match input {
            Input::Tensor(dtype, shape, values) => {
                Arg::Tensor(interpreter.add_tensor(TensorData::dense(dtype, shape, values)))
            }
            Input::F32(value) => Arg::Scalar(ReferenceScalar::F32(value.to_bits())),
        })
        .collect::<Vec<_>>();
    let outcome: OracleOutcome =
        interpreter.run(&arguments).unwrap_or_else(|error| panic!("{name} interpreter error: {error}"));
    if let SourceTermination::Failed(failure) = outcome.termination() {
        panic!("{name} failed: {failure}");
    }
    let result = outcome.results().next().unwrap();
    let OutcomeValue::Tensor(tensor) = result.value() else { panic!("tensor result") };
    (0..tensor.element_count()).map(|i| tensor.read(i).unwrap()).collect()
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

/// Values as stored in `dtype` (bf16 rounds to nearest even).
fn stored(dtype: DType, values: &[f32]) -> Vec<f32> {
    match dtype {
        DType::F32 => values.to_vec(),
        DType::BF16 => values.iter().map(|v| registry::bf16_round(*v)).collect(),
        other => panic!("unsupported element {other:?}"),
    }
}

fn element(dtype: DType) -> seismic::Element {
    match dtype {
        DType::F32 => seismic::Element::f32(),
        DType::BF16 => seismic::Element::bf16(),
        other => panic!("unsupported element {other:?}"),
    }
}

fn tensor(device: &seismic::Device, dtype: DType, shape: &[u64], values: &[f32]) -> seismic::Tensor {
    let bytes = match dtype {
        DType::F32 => values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
        DType::BF16 => values
            .iter()
            .flat_map(|v| ((registry::bf16_round(*v).to_bits() >> 16) as u16).to_le_bytes())
            .collect(),
        other => panic!("unsupported element {other:?}"),
    };
    seismic::Tensor::from_host(device, element(dtype), shape, &bytes).unwrap()
}

fn i32_tensor(device: &seismic::Device, shape: &[u64], values: &[i32]) -> seismic::Tensor {
    let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
    seismic::Tensor::from_host(device, seismic::Element::i32(), shape, &bytes).unwrap()
}

fn read_f32(tensor: &seismic::Tensor) -> Vec<f32> {
    tensor
        .read_to_host()
        .unwrap()
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}

/// CUDA, else Metal; `SEISMIC_TEST_BACKEND=vulkan` selects Vulkan (a CUDA
/// host also has a Vulkan device). Then the CPU.
fn devices() -> Vec<seismic::Device> {
    let catalog = seismic::DeviceCatalog::discover().unwrap();
    let gpu = match std::env::var("SEISMIC_TEST_BACKEND").ok().as_deref() {
        Some("vulkan") => Some(catalog.open_backend(seismic::BackendName::Vulkan).expect("the Vulkan device opens")),
        Some(other) => panic!("SEISMIC_TEST_BACKEND={other}: only vulkan is selectable"),
        None => [seismic::BackendName::Cuda, seismic::BackendName::Metal]
            .into_iter()
            .find_map(|backend| catalog.open_backend(backend).ok()),
    };
    gpu.into_iter()
        .chain(std::iter::once(catalog.open_backend(seismic::BackendName::Cpu).unwrap()))
        .collect()
}

fn is_cpu(device: &seismic::Device) -> bool {
    device.backend() == seismic::BackendName::Cpu
}

/// The specialization of an entry on `device`: CPU implementations have no
/// static dimensions.
fn statics_on(device: &seismic::Device, statics: &[(&str, u64)]) -> seismic::NativeSpecialization {
    let mut specialization = seismic::NativeSpecialization::new();
    if !is_cpu(device) {
        for (name, value) in statics {
            specialization = specialization.with_static(*name, *value);
        }
    }
    specialization
}

/// Every value within `tolerance` of the portable value, relative to the
/// magnitude of the terms it sums (`terms[i]`): the row ops' square sums
/// reassociate, and their additions cancel.
fn assert_within_terms(name: &str, actual: &[f32], expected: &[f64], terms: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len(), "{name} length");
    for (index, ((actual, expected), terms)) in actual.iter().zip(expected).zip(terms).enumerate() {
        let error = (f64::from(*actual) - expected).abs();
        assert!(
            error <= tolerance * terms.max(1e-30),
            "{name}[{index}]: native {actual}, portable {expected} (terms {terms})"
        );
    }
}

/// `1 / sqrt(mean(x^2) + epsilon)` in F64.
fn inverse(x: &[f32], epsilon: f32) -> f64 {
    let squares = x.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>();
    1.0 / (squares / x.len() as f64 + f64::from(epsilon)).sqrt()
}

/// Output rows gathered from the residual in a scrambled order.
fn out_rows(rows: usize, outputs: usize) -> Vec<i32> {
    (0..outputs).map(|o| ((o * 7 + 3) % rows) as i32).collect()
}

// Gemma 4 E2B / 26B / 31B and Muse Glimmer model widths.
const WIDTHS: [usize; 4] = [1536, 2816, 5376, 6656];

#[test]
fn native_post_norm_residual_matches_portable_body() {
    let module = module();
    for device in devices() {
        for (width, (rows, outputs), norm_dtype) in [
            (WIDTHS[0], (1, 1), DType::F32),
            (WIDTHS[1], (9, 5), DType::BF16),
            (WIDTHS[2], (16, 16), DType::F32),
            (WIDTHS[3], (40, 17), DType::F32),
        ] {
            let residual = pattern(rows * width, 3, 40.0);
            // Projection outputs of O(100) with a few large channels.
            let mut projected = pattern(outputs * width, 5, 120.0);
            projected[width / 3] = 3.0e4;
            let norm = stored(norm_dtype, &pattern(width, 7, 2.0));
            let selected = out_rows(rows, outputs);
            for (epsilon, scale) in [(1e-8f32, 1.0f32), (1e-6, 0.0209)] {
                let expected = interpret(
                    &module,
                    "post_norm_residual",
                    &[("NW", norm_dtype)],
                    vec![
                        floats(DType::F32, &[rows, width], &residual),
                        floats(DType::F32, &[outputs, width], &projected),
                        floats(norm_dtype, &[width], &norm),
                        ints(&[outputs], &selected),
                        Input::F32(epsilon),
                        Input::F32(scale),
                    ],
                );
                let kernel = post_norm_residual::native_for_device_with(
                    &device,
                    post_norm_residual::Elements { NW: element(norm_dtype) },
                    &seismic::NativeSpecialization::new(),
                )
                .unwrap();
                let outcome = kernel
                    .call(post_norm_residual::Args {
                        residual: &tensor(&device, DType::F32, &[rows as u64, width as u64], &residual),
                        projected: &tensor(&device, DType::F32, &[outputs as u64, width as u64], &projected),
                        norm: &tensor(&device, norm_dtype, &[width as u64], &norm),
                        out_rows: &i32_tensor(&device, &[outputs as u64], &selected),
                        epsilon,
                        scale,
                    })
                    .unwrap();
                let inverses = projected.chunks_exact(width).map(|row| inverse(row, epsilon)).collect::<Vec<_>>();
                let terms = (0..outputs * width)
                    .map(|flat| {
                        let (o, i) = (flat / width, flat % width);
                        let normalized = f64::from(projected[flat]).abs() * inverses[o] * f64::from(norm[i]).abs();
                        f64::from(scale) * (f64::from(residual[selected[o] as usize * width + i]).abs() + normalized)
                    })
                    .collect::<Vec<_>>();
                assert_within_terms(
                    &format!("{:?} post_norm_residual D {width} O {outputs} eps {epsilon}", device.backend()),
                    &read_f32(&outcome.value),
                    &expected,
                    &terms,
                    4e-6,
                );
            }
        }
    }
}

#[test]
fn native_moe_tail_matches_portable_body() {
    let module = module();
    for device in devices() {
        // Gemma 4 26B-A4B (D 2816), and a wider row.
        for (width, rows) in [(2816usize, 1usize), (2816, 9), (2816, 40), (5376, 3)] {
            let residual = pattern(rows * width, 11, 30.0);
            let dense = pattern(rows * width, 13, 200.0);
            let routed = pattern(rows * width, 17, 5.0);
            // Post-norm weights of the real model range up to O(100).
            let dense_norm = pattern(width, 19, 135.0);
            let routed_norm = pattern(width, 23, 2.4);
            let norm = pattern(width, 29, 30.0);
            let expected = interpret(
                &module,
                "moe_tail",
                &[("NW", DType::F32)],
                vec![
                    floats(DType::F32, &[rows, width], &residual),
                    floats(DType::F32, &[rows, width], &dense),
                    floats(DType::F32, &[rows, width], &routed),
                    floats(DType::F32, &[width], &dense_norm),
                    floats(DType::F32, &[width], &routed_norm),
                    floats(DType::F32, &[width], &norm),
                    Input::F32(1e-6),
                    Input::F32(0.0986),
                ],
            );
            let shape = [rows as u64, width as u64];
            let outcome = moe_tail::native_for_device_with(
                &device,
                moe_tail::Elements { NW: seismic::Element::f32() },
                &seismic::NativeSpecialization::new(),
            )
            .unwrap()
            .call(moe_tail::Args {
                residual: &tensor(&device, DType::F32, &shape, &residual),
                dense: &tensor(&device, DType::F32, &shape, &dense),
                routed: &tensor(&device, DType::F32, &shape, &routed),
                dense_norm: &tensor(&device, DType::F32, &[width as u64], &dense_norm),
                routed_norm: &tensor(&device, DType::F32, &[width as u64], &routed_norm),
                norm: &tensor(&device, DType::F32, &[width as u64], &norm),
                epsilon: 1e-6,
                scale: 0.0986,
            })
            .unwrap();
            // The combined row's terms bound its error; its norm scales it.
            let mut terms = Vec::with_capacity(rows * width);
            for m in 0..rows {
                let (d, r) = (&dense[m * width..(m + 1) * width], &routed[m * width..(m + 1) * width]);
                let (di, ri) = (inverse(d, 1e-6), inverse(r, 1e-6));
                let combined = (0..width)
                    .map(|j| {
                        (f64::from(d[j]) * di * f64::from(dense_norm[j])
                            + f64::from(r[j]) * ri * f64::from(routed_norm[j])) as f32
                    })
                    .collect::<Vec<_>>();
                let fi = inverse(&combined, 1e-6);
                for i in 0..width {
                    let spread = f64::from(d[i]).abs() * di * f64::from(dense_norm[i]).abs()
                        + f64::from(r[i]).abs() * ri * f64::from(routed_norm[i]).abs();
                    let normalized = spread * fi * f64::from(norm[i]).abs();
                    terms.push(0.0986 * (f64::from(residual[m * width + i]).abs() + normalized));
                }
            }
            assert_within_terms(
                &format!("{:?} moe_tail D {width} M {rows}", device.backend()),
                &read_f32(&outcome.value),
                &expected,
                &terms,
                8e-6,
            );
        }
    }
}

#[test]
fn native_per_layer_inputs_match_portable_body() {
    let module = module();
    for device in devices() {
        // Gemma 4 E2B (35 layers) and E4B (42 layers), P = 256; the table rows
        // as F32 and as BF16.
        for (layers, rows, table) in [(35usize, 1usize, DType::F32), (35, 5, DType::BF16), (42, 17, DType::F32)] {
            let chunk = 256usize;
            let width = layers * chunk;
            let gathered = stored(table, &pattern(rows * width, 31, 0.1));
            let projected = pattern(rows * width, 37, 40.0);
            let norm = pattern(chunk, 41, 1.5);
            let (epsilon, gathered_scale, projected_scale, scale) =
                (1e-6f32, 16.0f32, 1.0 / 1536f32.sqrt(), std::f32::consts::FRAC_1_SQRT_2);
            let expected = interpret(
                &module,
                "per_layer_inputs",
                &[("TW", table), ("NW", DType::F32)],
                vec![
                    floats(table, &[rows, width], &gathered),
                    floats(DType::F32, &[rows, width], &projected),
                    floats(DType::F32, &[chunk], &norm),
                    Input::F32(epsilon),
                    Input::F32(gathered_scale),
                    Input::F32(projected_scale),
                    Input::F32(scale),
                ],
            );
            let shape = [rows as u64, width as u64];
            let outcome = per_layer_inputs::native_for_device_with(
                &device,
                per_layer_inputs::Elements { TW: element(table), NW: seismic::Element::f32() },
                &statics_on(&device, &[("L", layers as u64), ("P", chunk as u64)]),
            )
            .unwrap()
            .call(per_layer_inputs::Args {
                gathered: &tensor(&device, table, &shape, &gathered),
                projected: &tensor(&device, DType::F32, &shape, &projected),
                norm: &tensor(&device, DType::F32, &[chunk as u64], &norm),
                epsilon,
                gathered_scale,
                projected_scale,
                scale,
            })
            .unwrap();
            let scaled = projected.iter().map(|v| v * projected_scale).collect::<Vec<_>>();
            let inverses = scaled.chunks_exact(chunk).map(|p| inverse(p, epsilon)).collect::<Vec<_>>();
            let terms = (0..rows * width)
                .map(|flat| {
                    let normalized =
                        f64::from(scaled[flat]).abs() * inverses[flat / chunk] * f64::from(norm[flat % chunk]).abs();
                    f64::from(scale) * (normalized + f64::from(gathered[flat]).abs() * f64::from(gathered_scale))
                })
                .collect::<Vec<_>>();
            assert_within_terms(
                &format!("{:?} per_layer_inputs L {layers} M {rows} {table:?}", device.backend()),
                &read_f32(&outcome.value),
                &expected,
                &terms,
                4e-6,
            );
        }
    }
}
