//! CUDA recurrent and attention projection entries
//! (`kernels/cuda_projection_blocks.seismic`) against host models over mma16
//! weights; the gated recurrent-output host model is pinned to C's portable
//! body by the reference interpreter. Tests return early without CUDA.

mod cuda_common;

use cuda_common::*;
use magnitude_model_kernels::{
    qwen_attention_output, qwen_attention_project, qwen_recurrent_output, qwen_recurrent_project,
};
use seismic::{Element, Layout, NativeSpecialization, Tensor};
use seismic_lang::registry::bf16_round;

const ROWS: [usize; 6] = [1, 3, 8, 16, 40, 128];

/// The specialization of an entry; `split` for entries declaring SPLIT.
fn specialization(statics: &[(&str, usize)], mapping: Mapping, split: bool) -> NativeSpecialization {
    let spec = statics
        .iter()
        .fold(NativeSpecialization::new(), |spec, (name, value)| spec.with_static(*name, *value as u64));
    mapping.params(spec, split)
}

/// One segment's expected A-rounded projection and tolerance over operand
/// rows `x` with per-value `slack`.
fn rounded_segment(
    x: &[f32],
    slack: &[f32],
    weight: &Weight,
    m: usize,
    n: usize,
    k: usize,
    mapping: Mapping,
) -> (Vec<f64>, Vec<f64>) {
    let (values, magnitude) = project(x, &weight.values, m, n, k);
    let bound = slack_bound(slack, &weight.values, m, n, k);
    let dequant = dequant_bound(x, &weight.values, m, n, k, mapping);
    let tolerance = (0..values.len())
        .map(|i| values[i].abs() / 256.0 + magnitude[i] * 2.5e-4 + bound[i] + dequant[i] + 1e-6)
        .collect();
    (values, tolerance)
}

/// Columns [offset, offset + n) of every row of a [m, width] expectation.
fn place(target: &mut [f64], width: usize, offset: usize, source: &[f64], m: usize, n: usize) {
    for row in 0..m {
        target[row * width + offset..row * width + offset + n].copy_from_slice(&source[row * n..(row + 1) * n]);
    }
}

#[test]
fn cuda_recurrent_project_matches_host_model() {
    let Some(device) = cuda() else { return };
    let mut rng = Rng(41);
    let (h, nk, nv, w) = (512usize, 2usize, 4usize, 64usize);
    let (qkv, z) = ((2 * nk + nv) * w, nv * w);
    let width = qkv + z + 2 * nv;
    // The 4B mix (qkv q5k, z q4k, alpha/beta q8) and a uniform q6k set.
    for formats in [[Format::Q5K, Format::Q4K, Format::Q8, Format::Q8], [Format::Q6K; 4]] {
        let weights = [
            weight(&device, formats[0], qkv, h, &mut rng),
            weight(&device, formats[1], z, h, &mut rng),
            weight(&device, formats[2], nv, h, &mut rng),
            weight(&device, formats[3], nv, h, &mut rng),
        ];
        for m in ROWS {
            let hidden_values: Vec<f32> = (0..m * h).map(|_| rng.uniform(-3.0, 3.0)).collect();
            let norm_values: Vec<f32> = (0..h).map(|_| rng.uniform(0.25, 1.25)).collect();
            let normed = normalized(&hidden_values, &norm_values, m, h);
            let hidden = f32_tensor(&device, &[m as u64, h as u64], &hidden_values);
            let norm = f32_tensor(&device, &[h as u64], &norm_values);
            for &mapping in mappings(m) {
                let (x, slack) = operand_rows(&normed, h, mapping);
                let mut expected = vec![0.0; m * width];
                let mut tolerance = vec![0.0; m * width];
                for (weight, (offset, n)) in weights.iter().zip([(0, qkv), (qkv, z), (qkv + z, nv), (qkv + z + nv, nv)]) {
                    let (values, bound) = rounded_segment(&x, &slack, weight, m, n, h, mapping);
                    place(&mut expected, width, offset, &values, m, n);
                    place(&mut tolerance, width, offset, &bound, m, n);
                }
                let projection = qwen_recurrent_project::native_for_device_with(
                    &device,
                    qwen_recurrent_project::Elements {
                        NW: Element::f32(),
                        QW: formats[0].resident(),
                        GW: formats[1].resident(),
                        AW: formats[2].resident(),
                        BW: formats[3].resident(),
                        A: Element::bf16(),
                    },
                    &specialization(&[("H", h), ("NK", nk), ("NV", nv), ("W", w)], mapping, false),
                )
                .unwrap()
                .call(qwen_recurrent_project::Args {
                    hidden: &hidden,
                    input_norm: &norm,
                    qkv_weight: &weights[0].tensor,
                    gate_weight: &weights[1].tensor,
                    alpha_weight: &weights[2].tensor,
                    beta_weight: &weights[3].tensor,
                    epsilon: EPSILON,
                })
                .unwrap()
                .value;
                check(
                    &format!("recurrent project {formats:?} M={m} {mapping:?}"),
                    &read_bf16(&projection),
                    &expected,
                    &tolerance,
                );
            }
        }
    }
}

/// C's gated prologue: round(round(v * inv * norm) * round(silu(z))).
fn gated(mixed: &[f32], z: &[f32], norm: &[f32], m: usize, nv: usize, w: usize) -> Vec<f32> {
    let mut out = vec![0.0; m * nv * w];
    for row in 0..m {
        for head in 0..nv {
            let base = row * nv * w + head * w;
            let squares = mixed[base..base + w].iter().fold(0.0f32, |acc, v| v.mul_add(*v, acc));
            let inverse = 1.0 / (squares / w as f32 + EPSILON).sqrt();
            for c in 0..w {
                let normalized = bf16_round(mixed[base + c] * inverse * norm[c]);
                let g = z[base + c];
                let activated = bf16_round(g / (1.0 + (-g).exp()));
                out[base + c] = bf16_round(normalized * activated);
            }
        }
    }
    out
}

struct RecurrentOutputCase {
    m: usize,
    h: usize,
    nk: usize,
    nv: usize,
    w: usize,
    hidden: Vec<f32>,
    mixed: Vec<f32>,
    projection: Vec<f32>,
    norm: Vec<f32>,
}

impl RecurrentOutputCase {
    fn new(m: usize, rng: &mut Rng) -> Self {
        let (h, nk, nv, w) = (272usize, 2usize, 4usize, 128usize);
        let width = (2 * nk + nv) * w + nv * w + 2 * nv;
        Self {
            m,
            h,
            nk,
            nv,
            w,
            hidden: (0..m * h).map(|_| rng.uniform(-2.0, 2.0)).collect(),
            mixed: (0..m * nv * w).map(|_| bf16_round(rng.uniform(-1.5, 1.5))).collect(),
            projection: (0..m * width).map(|_| bf16_round(rng.uniform(-4.0, 4.0))).collect(),
            norm: (0..w).map(|_| rng.uniform(0.5, 1.5)).collect(),
        }
    }
    fn width(&self) -> usize {
        (2 * self.nk + self.nv) * self.w + self.nv * self.w + 2 * self.nv
    }
    fn z(&self) -> Vec<f32> {
        let offset = (2 * self.nk + self.nv) * self.w;
        (0..self.m)
            .flat_map(|row| {
                self.projection[row * self.width() + offset..row * self.width() + offset + self.nv * self.w]
                    .iter()
                    .copied()
            })
            .collect()
    }
    fn expected(&self, weight: &Weight, mapping: Mapping) -> (Vec<f64>, Vec<f64>) {
        let k = self.nv * self.w;
        let (x, slack) = operand_rows(&gated(&self.mixed, &self.z(), &self.norm, self.m, self.nv, self.w), k, mapping);
        let (projected, magnitude) = project(&x, &weight.values, self.m, self.h, k);
        let bound = slack_bound(&slack, &weight.values, self.m, self.h, k);
        let dequant = dequant_bound(&x, &weight.values, self.m, self.h, k, mapping);
        let expected = (0..self.m * self.h).map(|i| f64::from(self.hidden[i]) + projected[i]).collect();
        let tolerance = (0..self.m * self.h)
            .map(|i| projected[i].abs() / 256.0 + magnitude[i] * 2.5e-4 + bound[i] + dequant[i] + 1e-6)
            .collect();
        (expected, tolerance)
    }
}

#[test]
fn cuda_recurrent_output_matches_host_model() {
    let Some(device) = cuda() else { return };
    let mut rng = Rng(43);
    for format in Format::ALL {
        for m in ROWS {
            let case = RecurrentOutputCase::new(m, &mut rng);
            let output = weight(&device, format, case.h, case.nv * case.w, &mut rng);
            let hidden = f32_tensor(&device, &[m as u64, case.h as u64], &case.hidden);
            let mixed = bf16_tensor(&device, &[m as u64, case.nv as u64, case.w as u64], &case.mixed);
            let projection = bf16_tensor(&device, &[m as u64, case.width() as u64], &case.projection);
            let norm = f32_tensor(&device, &[case.w as u64], &case.norm);
            for &mapping in mappings(m) {
                let (expected, tolerance) = case.expected(&output, mapping);
                let result = qwen_recurrent_output::native_for_device_with(
                    &device,
                    qwen_recurrent_output::Elements { A: Element::bf16(), RN: Element::f32(), OW: format.resident() },
                    &specialization(&[("H", case.h), ("NK", case.nk), ("NV", case.nv), ("W", case.w)], mapping, true),
                )
                .unwrap()
                .call(qwen_recurrent_output::Args {
                    hidden: &hidden,
                    mixed: &mixed,
                    projection: &projection,
                    recurrent_norm: &norm,
                    output_weight: &output.tensor,
                    epsilon: EPSILON,
                })
                .unwrap()
                .value;
                check(
                    &format!("recurrent output {format:?} M={m} {mapping:?}"),
                    &read_f32(&result),
                    &expected,
                    &tolerance,
                );
            }
        }
    }
}

/// The gated host model is C's portable body: the reference interpreter over
/// the same mma16 bytes agrees with it.
#[test]
fn recurrent_output_host_model_matches_portable_body() {
    use seismic_lang::{
        checked::{check_source, SourceFile},
        entry::ElementBindings,
        failure::SourceTermination,
        interp::{Arg, Interpreter, OutcomeValue, TensorData},
        reference_math::ReferenceScalar,
        registry,
        types::DType,
    };
    let Some(device) = cuda() else { return };
    let mut rng = Rng(47);
    let mut sources = seismic_std::sources();
    sources.push(SourceFile {
        path: "recurrent_stages.seismic".into(),
        text: include_str!("../kernels/recurrent_stages.seismic").into(),
    });
    let module = check_source(sources).unwrap();
    for format in Format::ALL {
        let case = RecurrentOutputCase::new(2, &mut rng);
        let output = weight(&device, format, case.h, case.nv * case.w, &mut rng);
        let (expected, tolerance) = case.expected(&output, GEMV_MAPPINGS[0]);
        let storage = registry::storage(format.representation(), Layout::Mma16).unwrap();
        let logical = module
            .entry(
                module.entry_named("qwen_recurrent_output").unwrap(),
                &ElementBindings::new()
                    .bind("A", registry::dense(DType::BF16))
                    .bind("RN", registry::dense(DType::F32))
                    .bind("OW", storage),
            )
            .unwrap();
        let mut interpreter = Interpreter::new(&logical);
        let floats = |values: &[f32]| values.iter().map(|x| f64::from(*x)).collect::<Vec<_>>();
        let args = vec![
            Arg::Tensor(interpreter.add_tensor(TensorData::dense(DType::F32, vec![case.m, case.h], floats(&case.hidden)))),
            Arg::Tensor(interpreter.add_tensor(TensorData::dense(
                DType::BF16,
                vec![case.m, case.nv, case.w],
                floats(&case.mixed),
            ))),
            Arg::Tensor(interpreter.add_tensor(TensorData::dense(
                DType::BF16,
                vec![case.m, case.width()],
                floats(&case.projection),
            ))),
            Arg::Tensor(interpreter.add_tensor(TensorData::dense(DType::F32, vec![case.w], floats(&case.norm)))),
            Arg::Tensor(interpreter.add_tensor(
                TensorData::encoded(storage, vec![case.h, case.nv * case.w], output.bytes.clone()).unwrap(),
            )),
            Arg::Scalar(ReferenceScalar::F32(EPSILON.to_bits())),
        ];
        let outcome = interpreter.run(&args).unwrap();
        if let SourceTermination::Failed(failure) = outcome.termination() {
            panic!("qwen_recurrent_output portable body failed: {failure}");
        }
        let result = outcome.results().next().unwrap();
        let OutcomeValue::Tensor(values) = result.value() else { panic!("tensor result") };
        let actual: Vec<f32> = (0..case.m * case.h).map(|i| values.read(i).unwrap() as f32).collect();
        check(&format!("portable recurrent output {format:?}"), &actual, &expected, &tolerance);
    }
}

#[test]
fn cuda_attention_project_matches_host_model() {
    let Some(device) = cuda() else { return };
    let mut rng = Rng(53);
    let (d, kv, g, w) = (512usize, 2usize, 2usize, 64usize);
    let (query, keys) = (kv * g * 2 * w, kv * w);
    // The 4B mix (q+gate q4k, k q4k, v q6k) and a uniform q5k set.
    for formats in [[Format::Q4K, Format::Q4K, Format::Q6K], [Format::Q5K; 3]] {
        let weights = [
            weight(&device, formats[0], query, d, &mut rng),
            weight(&device, formats[1], keys, d, &mut rng),
            weight(&device, formats[2], keys, d, &mut rng),
        ];
        for m in ROWS {
            let hidden_values: Vec<f32> = (0..m * d).map(|_| rng.uniform(-3.0, 3.0)).collect();
            let norm_values: Vec<f32> = (0..d).map(|_| rng.uniform(0.25, 1.25)).collect();
            let normed = normalized(&hidden_values, &norm_values, m, d);
            let hidden = f32_tensor(&device, &[m as u64, d as u64], &hidden_values);
            let norm = f32_tensor(&device, &[d as u64], &norm_values);
            let query_norm = f32_tensor(&device, &[w as u64], &vec![1.0; w]);
            for &mapping in mappings(m) {
                let (x, slack) = operand_rows(&normed, d, mapping);
                let expected: Vec<(Vec<f64>, Vec<f64>)> = weights
                    .iter()
                    .zip([query, keys, keys])
                    .map(|(weight, n)| rounded_segment(&x, &slack, weight, m, n, d, mapping))
                    .collect();
                let results = qwen_attention_project::native_for_device_with(
                    &device,
                    qwen_attention_project::Elements {
                        NW: Element::f32(),
                        QW: formats[0].resident(),
                        KW: formats[1].resident(),
                        VW: formats[2].resident(),
                        A: Element::bf16(),
                    },
                    &specialization(&[("D", d), ("KV", kv), ("G", g), ("W", w)], mapping, false),
                )
                .unwrap()
                .call(qwen_attention_project::Args {
                    hidden: &hidden,
                    input_norm: &norm,
                    query_norm: &query_norm,
                    query_gate_weight: &weights[0].tensor,
                    key_weight: &weights[1].tensor,
                    value_weight: &weights[2].tensor,
                    epsilon: EPSILON,
                })
                .unwrap();
                for (name, tensor, (values, tolerance)) in
                    [("query_gate", &results.r0, &expected[0]), ("key", &results.r1, &expected[1]), ("value", &results.r2, &expected[2])]
                {
                    check(
                        &format!("attention project {name} {formats:?} M={m} {mapping:?}"),
                        &read_bf16(tensor),
                        values,
                        tolerance,
                    );
                }
            }
        }
    }
}

#[test]
fn cuda_attention_output_matches_host_model() {
    let Some(device) = cuda() else { return };
    let mut rng = Rng(59);
    let (d, q, w) = (272usize, 4usize, 128usize);
    for format in Format::ALL {
        let output = weight(&device, format, d, q * w, &mut rng);
        for m in ROWS {
            let gated_values: Vec<f32> = (0..m * q * w).map(|_| bf16_round(rng.uniform(-1.0, 1.0))).collect();
            let hidden_values: Vec<f32> = (0..m * d).map(|_| rng.uniform(-2.0, 2.0)).collect();
            let gated_tensor = bf16_tensor(&device, &[m as u64, q as u64, w as u64], &gated_values);
            let hidden = f32_tensor(&device, &[m as u64, d as u64], &hidden_values);
            for &mapping in mappings(m) {
                // The heads are exact input, so the INT8 emulation is exact too.
                let (x, _) = operand_rows(&gated_values, q * w, mapping);
                let (projected, magnitude) = project(&x, &output.values, m, d, q * w);
                let expected: Vec<f64> = (0..m * d).map(|i| f64::from(hidden_values[i]) + projected[i]).collect();
                let dequant = dequant_bound(&x, &output.values, m, d, q * w, mapping);
                let tolerance: Vec<f64> = (0..m * d)
                    .map(|i| projected[i].abs() / 256.0 + magnitude[i] * 2e-5 + dequant[i] + 1e-6)
                    .collect();
                let result = qwen_attention_output::native_for_device_with(
                    &device,
                    qwen_attention_output::Elements { A: Element::bf16(), OW: format.resident() },
                    &specialization(&[("D", d), ("Q", q), ("W", w)], mapping, true),
                )
                .unwrap()
                .call(qwen_attention_output::Args { hidden: &hidden, gated: &gated_tensor, output_weight: &output.tensor })
                .unwrap()
                .value;
                check(
                    &format!("attention output {format:?} M={m} {mapping:?}"),
                    &read_f32(&result),
                    &expected,
                    &tolerance,
                );
            }
        }
    }
}

/// Device time per call on the 4B geometry over zero-filled weights; the
/// rotation exceeds the 24 MiB L2.
#[test]
#[ignore = "timing; run explicitly on the measurement host"]
fn cuda_projection_block_timings() {
    let Some(device) = cuda() else { return };
    let options = seismic::MeasureOptions { samples: 15, min_sample_seconds: 0.002 };
    let (h, nk, nv, w) = (2560usize, 16usize, 32usize, 128usize);
    let (qkv, z) = ((2 * nk + nv) * w, nv * w);
    let rotation = 4;
    let project_weights: Vec<[Tensor; 4]> = (0..rotation)
        .map(|_| {
            [
                timing_weight(&device, Format::Q5K, qkv, h),
                timing_weight(&device, Format::Q4K, z, h),
                timing_weight(&device, Format::Q8, nv, h),
                timing_weight(&device, Format::Q8, nv, h),
            ]
        })
        .collect();
    let output_weights: Vec<Tensor> = (0..rotation * 2).map(|_| timing_weight(&device, Format::Q5K, h, z)).collect();
    let project_bytes: f64 = project_weights[0].iter().map(|t| t.byte_len() as f64).sum();
    let output_bytes = output_weights[0].byte_len() as f64;
    for m in [1usize, 8, 128] {
        let hidden = f32_tensor(&device, &[m as u64, h as u64], &vec![0.5; m * h]);
        let norm = f32_tensor(&device, &[h as u64], &vec![1.0; h]);
        let mixed = bf16_tensor(&device, &[m as u64, nv as u64, w as u64], &vec![0.25; m * z]);
        let projection = bf16_tensor(&device, &[m as u64, (qkv + z + 2 * nv) as u64], &vec![0.5; m * (qkv + z + 2 * nv)]);
        let recurrent_norm = f32_tensor(&device, &[w as u64], &vec![1.0; w]);
        for &mapping in mappings(m) {
            let kernel = qwen_recurrent_project::native_for_device_with(
                &device,
                qwen_recurrent_project::Elements {
                    NW: Element::f32(),
                    QW: Format::Q5K.resident(),
                    GW: Format::Q4K.resident(),
                    AW: Format::Q8.resident(),
                    BW: Format::Q8.resident(),
                    A: Element::bf16(),
                },
                &specialization(&[("H", h), ("NK", nk), ("NV", nv), ("W", w)], mapping, false),
            )
            .unwrap();
            let args = project_weights
                .iter()
                .map(|weights| qwen_recurrent_project::Args {
                    hidden: &hidden,
                    input_norm: &norm,
                    qkv_weight: &weights[0],
                    gate_weight: &weights[1],
                    alpha_weight: &weights[2],
                    beta_weight: &weights[3],
                    epsilon: EPSILON,
                })
                .collect();
            let measured = kernel.measure(args, &options).unwrap();
            println!(
                "recurrent project M={m} K={h} N={} {mapping:?}: {:.1} us, {:.0} GB/s, {:.2} TFLOP/s",
                qkv + z + 2 * nv,
                measured.median * 1e6,
                project_bytes / measured.median / 1e9,
                2.0 * (m * h * (qkv + z + 2 * nv)) as f64 / measured.median / 1e12
            );
            let kernel = qwen_recurrent_output::native_for_device_with(
                &device,
                qwen_recurrent_output::Elements { A: Element::bf16(), RN: Element::f32(), OW: Format::Q5K.resident() },
                &specialization(&[("H", h), ("NK", nk), ("NV", nv), ("W", w)], mapping, true),
            )
            .unwrap();
            let args = output_weights
                .iter()
                .map(|weight| qwen_recurrent_output::Args {
                    hidden: &hidden,
                    mixed: &mixed,
                    projection: &projection,
                    recurrent_norm: &recurrent_norm,
                    output_weight: weight,
                    epsilon: EPSILON,
                })
                .collect();
            let measured = kernel.measure(args, &options).unwrap();
            println!(
                "recurrent output M={m} K={z} N={h} {mapping:?}: {:.1} us, {:.0} GB/s, {:.2} TFLOP/s",
                measured.median * 1e6,
                output_bytes / measured.median / 1e9,
                2.0 * (m * h * z) as f64 / measured.median / 1e12
            );
        }
    }
}
