//! CUDA recurrent and attention projection entries
//! (`kernels/{attention,recurrent}.seismic`) against host models over mma16
//! weights. The recurrent output projection is `attention_output` over the
//! gated rows the state entries publish. Tests return early without CUDA.

mod cuda_common;

use cuda_common::*;
use magnitude_kernels::{attention_output, attention_project, gated_delta_project};
use seismic::{Element, NativeSpecialization, Tensor};
use seismic_lang::registry::bf16_round;

const ROWS: [usize; 8] = [1, 3, 8, 12, 16, 17, 40, 128];

/// The specialization of an entry.
fn statics_specialization(statics: &[(&str, usize)]) -> NativeSpecialization {
    statics
        .iter()
        .fold(NativeSpecialization::new(), |spec, (name, value)| {
            spec.with_static(*name, *value as u64)
        })
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
        target[row * width + offset..row * width + offset + n]
            .copy_from_slice(&source[row * n..(row + 1) * n]);
    }
}

#[test]
fn cuda_recurrent_project_matches_host_model() {
    // The 4B mix (qkv q5k, z q4k, alpha/beta q8), a uniform q6k set, and the
    // 4B MTP GGUF's mix (qkv q6k, alpha/beta F32 in the file: dense, resident
    // bf16). `None` is a dense bf16 weight.
    recurrent_project_cases(
        &[
            [
                Some(Format::Q5K),
                Some(Format::Q4K),
                Some(Format::Q8),
                Some(Format::Q8),
            ],
            [Some(Format::Q6K); 4],
            [Some(Format::Q6K), Some(Format::Q4K), None, None],
        ],
        41,
    );
}

/// IQ4_XS / IQ4_NL weights are resident as iq4g32.
#[test]
fn cuda_recurrent_project_iq4_matches_host_model() {
    recurrent_project_cases(&[[Some(Format::Iq4); 4]], 43);
}

fn recurrent_project_cases(mixes: &[[Option<Format>; 4]], seed: u64) {
    let Some(device) = cuda() else { return };
    let mut rng = Rng(seed);
    let (h, nk, nv, w) = (512usize, 2usize, 4usize, 64usize);
    let (qkv, z) = ((2 * nk + nv) * w, nv * w);
    let width = qkv + z + 2 * nv;
    for &formats in mixes {
        let make = |format: Option<Format>, rows: usize, rng: &mut Rng| match format {
            Some(format) => weight(&device, format, rows, h, rng),
            None => dense_weight(&device, rows, h, rng),
        };
        let element = |format: Option<Format>| format.map_or(Element::bf16(), Format::resident);
        let weights = [
            make(formats[0], qkv, &mut rng),
            make(formats[1], z, &mut rng),
            make(formats[2], nv, &mut rng),
            make(formats[3], nv, &mut rng),
        ];
        for m in ROWS {
            let hidden_values: Vec<f32> = (0..m * h).map(|_| rng.uniform(-3.0, 3.0)).collect();
            let norm_values: Vec<f32> = (0..h).map(|_| rng.uniform(0.25, 1.25)).collect();
            let normed = normalized(&hidden_values, &norm_values, m, h);
            let hidden = f32_tensor(&device, &[m as u64, h as u64], &hidden_values);
            let norm = f32_tensor(&device, &[h as u64], &norm_values);
            for &mapping in mappings(m) {
                // INT8 has no effect when any weight is dense.
                let path = if formats.contains(&None) {
                    Mapping { int8: 0, ..mapping }
                } else {
                    mapping
                };
                let (x, slack) = operand_rows(&normed, h, path);
                let mut expected = vec![0.0; m * width];
                let mut tolerance = vec![0.0; m * width];
                for (weight, (offset, n)) in
                    weights
                        .iter()
                        .zip([(0, qkv), (qkv, z), (qkv + z, nv), (qkv + z + nv, nv)])
                {
                    let (values, bound) = rounded_segment(&x, &slack, weight, m, n, h, path);
                    place(&mut expected, width, offset, &values, m, n);
                    place(&mut tolerance, width, offset, &bound, m, n);
                }
                let projection = gated_delta_project::native_for_device_with(
                    &device,
                    gated_delta_project::Elements {
                        NW: Element::f32(),
                        QW: element(formats[0]),
                        GW: element(formats[1]),
                        AW: element(formats[2]),
                        BW: element(formats[3]),
                        A: Element::bf16(),
                    },
                    &mapping.recurrent_params(statics_specialization(&[
                        ("H", h),
                        ("NK", nk),
                        ("NV", nv),
                        ("W", w),
                    ])),
                )
                .unwrap()
                .call(gated_delta_project::Args {
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

#[test]
fn cuda_attention_project_matches_host_model() {
    // The 4B mix (q+gate q4k, k q4k, v q6k) and a uniform q5k set.
    attention_project_cases(
        &[[Format::Q4K, Format::Q4K, Format::Q6K], [Format::Q5K; 3]],
        53,
    );
}

/// IQ4_XS / IQ4_NL weights are resident as iq4g32.
#[test]
fn cuda_attention_project_iq4_matches_host_model() {
    attention_project_cases(&[[Format::Iq4; 3]], 59);
}

fn attention_project_cases(sets: &[[Format; 3]], seed: u64) {
    let Some(device) = cuda() else { return };
    let mut rng = Rng(seed);
    let (d, kv, g, w) = (512usize, 2usize, 2usize, 64usize);
    let (query, keys) = (kv * g * 2 * w, kv * w);
    for &formats in sets {
        let weights = [
            weight(&device, formats[0], query, d, &mut rng),
            weight(&device, formats[1], keys, d, &mut rng),
            weight(&device, formats[2], keys, d, &mut rng),
        ];
        // Qwen's form: the gate is interleaved in the query segment, so the
        // gate segment is zero rows of the query weight.
        let gate = weights[0].tensor.slice_leading(0, 0).unwrap();
        for m in ROWS {
            let hidden_values: Vec<f32> = (0..m * d).map(|_| rng.uniform(-3.0, 3.0)).collect();
            let norm_values: Vec<f32> = (0..d).map(|_| rng.uniform(0.25, 1.25)).collect();
            let normed = normalized(&hidden_values, &norm_values, m, d);
            let hidden = f32_tensor(&device, &[m as u64, d as u64], &hidden_values);
            let norm = f32_tensor(&device, &[d as u64], &norm_values);
            for &mapping in mappings(m) {
                let (x, slack) = operand_rows(&normed, d, mapping);
                let expected: Vec<(Vec<f64>, Vec<f64>)> = weights
                    .iter()
                    .zip([query, keys, keys])
                    .map(|(weight, n)| rounded_segment(&x, &slack, weight, m, n, d, mapping))
                    .collect();
                let results = attention_project::native_for_device_with(
                    &device,
                    attention_project::Elements {
                        NW: Element::f32(),
                        QW: formats[0].resident(),
                        GW: formats[0].resident(),
                        KW: formats[1].resident(),
                        VW: formats[2].resident(),
                        A: Element::bf16(),
                    },
                    &mapping.attention_project_params(statics_specialization(&[
                        ("D", d),
                        ("Q", query),
                        ("GR", 0),
                        ("K", keys),
                        ("V", keys),
                    ])),
                )
                .unwrap()
                .call(attention_project::Args {
                    hidden: &hidden,
                    input_norm: &norm,
                    query_weight: &weights[0].tensor,
                    gate_weight: &gate,
                    key_weight: &weights[1].tensor,
                    value_weight: &weights[2].tensor,
                    epsilon: EPSILON,
                    project_mode: 0,
                })
                .unwrap();
                for (name, tensor, (values, tolerance)) in [
                    ("query", &results.r0, &expected[0]),
                    ("key", &results.r2, &expected[1]),
                    ("value", &results.r3, &expected[2]),
                ] {
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
            let gated_values: Vec<f32> = (0..m * q * w)
                .map(|_| bf16_round(rng.uniform(-1.0, 1.0)))
                .collect();
            let hidden_values: Vec<f32> = (0..m * d).map(|_| rng.uniform(-2.0, 2.0)).collect();
            let gated_tensor = bf16_tensor(&device, &[m as u64, q as u64, w as u64], &gated_values);
            let hidden = f32_tensor(&device, &[m as u64, d as u64], &hidden_values);
            for &mapping in mappings(m) {
                // The heads are exact input, so the INT8 emulation is exact too.
                let (x, _) = operand_rows(&gated_values, q * w, mapping);
                let (projected, magnitude) = project(&x, &output.values, m, d, q * w);
                let expected: Vec<f64> = (0..m * d)
                    .map(|i| f64::from(hidden_values[i]) + projected[i])
                    .collect();
                let dequant = dequant_bound(&x, &output.values, m, d, q * w, mapping);
                let tolerance: Vec<f64> = (0..m * d)
                    .map(|i| projected[i].abs() / 256.0 + magnitude[i] * 2e-5 + dequant[i] + 1e-6)
                    .collect();
                let result = attention_output::native_for_device_with(
                    &device,
                    attention_output::Elements {
                        A: Element::bf16(),
                        OW: format.resident(),
                    },
                    &mapping.attention_output_params(statics_specialization(&[
                        ("D", d),
                        ("Q", q),
                        ("W", w),
                    ])),
                )
                .unwrap()
                .call(attention_output::Args {
                    hidden: &hidden,
                    gated: &gated_tensor,
                    output_weight: &output.tensor,
                })
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
    let options = seismic::MeasureOptions {
        samples: 15,
        min_sample_seconds: 0.002,
    };
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
    let output_weights: Vec<Tensor> = (0..rotation * 2)
        .map(|_| timing_weight(&device, Format::Q5K, h, z))
        .collect();
    let project_bytes: f64 = project_weights[0].iter().map(|t| t.byte_len() as f64).sum();
    let output_bytes = output_weights[0].byte_len() as f64;
    for m in timing_rows(&[1, 8, 32, 128, 512]) {
        let hidden = f32_tensor(&device, &[m as u64, h as u64], &vec![0.5; m * h]);
        let norm = f32_tensor(&device, &[h as u64], &vec![1.0; h]);
        let gated = bf16_tensor(
            &device,
            &[m as u64, nv as u64, w as u64],
            &vec![0.25; m * z],
        );
        for &mapping in mappings(m) {
            let kernel = gated_delta_project::native_for_device_with(
                &device,
                gated_delta_project::Elements {
                    NW: Element::f32(),
                    QW: Format::Q5K.resident(),
                    GW: Format::Q4K.resident(),
                    AW: Format::Q8.resident(),
                    BW: Format::Q8.resident(),
                    A: Element::bf16(),
                },
                &mapping.recurrent_params(statics_specialization(&[
                    ("H", h),
                    ("NK", nk),
                    ("NV", nv),
                    ("W", w),
                ])),
            )
            .unwrap();
            let args = project_weights
                .iter()
                .map(|weights| gated_delta_project::Args {
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
            let kernel = attention_output::native_for_device_with(
                &device,
                attention_output::Elements {
                    A: Element::bf16(),
                    OW: Format::Q5K.resident(),
                },
                &mapping.attention_output_params(statics_specialization(&[
                    ("D", h),
                    ("Q", nv),
                    ("W", w),
                ])),
            )
            .unwrap();
            let args = output_weights
                .iter()
                .map(|weight| attention_output::Args {
                    hidden: &hidden,
                    gated: &gated,
                    output_weight: weight,
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
