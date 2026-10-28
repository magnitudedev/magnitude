//! CUDA recurrent and attention projection entries
//! (`kernels/{attention,recurrent}.seismic`) against host models over mma16
//! weights. The recurrent output projection is `attention_output` over the
//! gated rows the state entries publish. Tests return early without CUDA.

mod cuda_common;

use cuda_common::*;
use magnitude_kernels::{attention_output, attention_project, gated_delta_project};
use seismic::{Element, NativeSpecialization};
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

/// IQ4_XS weights are resident as iq4xs.
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
                let (x, slack) = operand_rows(&normed, h, path, formats[0]);
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
                    &mapping.projection_params(statics_specialization(&[
                        ("H", h),
                        ("PT", 1),
                        ("NM", h),
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
                    qkv_bound: &f32_tensor(&device, &[0], &[]),
                    gate_bound: &f32_tensor(&device, &[0], &[]),
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

/// IQ4_XS weights are resident as iq4xs.
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
                let (x, slack) = operand_rows(&normed, d, mapping, Some(formats[0]));
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
                    &mapping.projection_params(statics_specialization(&[
                        ("D", d),
                        ("PT", 1),
                        ("NM", d),
                        ("Q", query),
                        ("GR", 0),
                        ("K", keys),
                        ("V", keys),
                        ("QS", 0),
                        ("GS", 0),
                        ("KS", 0),
                        ("VS", 0),
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
                    query_scale: &f32_tensor(&device, &[0], &[]),
                    gate_scale: &f32_tensor(&device, &[0], &[]),
                    key_scale: &f32_tensor(&device, &[0], &[]),
                    value_scale: &f32_tensor(&device, &[0], &[]),
                    query_bound: &f32_tensor(&device, &[0], &[]),
                    gate_bound: &f32_tensor(&device, &[0], &[]),
                    key_bound: &f32_tensor(&device, &[0], &[]),
                    value_bound: &f32_tensor(&device, &[0], &[]),
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
                let (x, _) = operand_rows(&gated_values, q * w, mapping, Some(format));
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
                    &mapping.projection_params(statics_specialization(&[
                        ("D", d),
                        ("SN", 1),
                        ("IN", 0),
                        ("Q", q),
                        ("W", w),
                        ("OS", 0),
                    ])),
                )
                .unwrap()
                .call(attention_output::Args {
                    hidden: &hidden,
                    inject: &no_inject(&device, hidden.extents()[0]),
                    gated: &gated_tensor,
                    output_weight: &output.tensor,
                    output_scale: &f32_tensor(&device, &[0], &[]),
                    output_bound: &f32_tensor(&device, &[0], &[]),
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

/// Projections under their weights' second-level scales (Nemotron's NVFP4
/// `.scale`s): every segment of `attention_project` (QS = GS = KS = VS = 1,
/// one scale each) and `attention_output` (OS = 1) multiply their F32
/// accumulators by their scale before rounding.
#[test]
fn cuda_scaled_projections_match_host_model() {
    let Some(device) = cuda() else { return };
    /// The segments' second-level scales, query, gate, key, value: none a
    /// power of two, each distinct, so a scale bound to the wrong segment
    /// fails.
    const SCALES: [f32; 4] = [0.412_5, 0.731_2, 1.368_8, 0.207_5];
    /// The output projection's.
    const SCALE: f32 = 0.412_5;
    let factor = f64::from(SCALE);
    let mut rng = Rng(61);
    let (d, heads, width) = (512usize, 4usize, 128usize);
    let segments = [192usize, 64, 128, 128];
    let k = heads * width;
    let none = f32_tensor(&device, &[0], &[]);
    let scales = SCALES.map(|scale| f32_tensor(&device, &[1], &[scale]));
    let scale = f32_tensor(&device, &[1], &[SCALE]);
    for format in [Format::Nvfp4, Format::Nvfp4A4, Format::Q4K] {
        let projections = segments.map(|rows| weight(&device, format, rows, d, &mut rng));
        let output = weight(&device, format, d, k, &mut rng);
        for m in ROWS {
            let hidden_values: Vec<f32> = (0..m * d).map(|_| rng.uniform(-3.0, 3.0)).collect();
            let norm_values: Vec<f32> = (0..d).map(|_| rng.uniform(0.25, 1.25)).collect();
            let gated_values: Vec<f32> =
                (0..m * k).map(|_| bf16_round(rng.uniform(-1.0, 1.0))).collect();
            let normed = normalized(&hidden_values, &norm_values, m, d);
            let hidden = f32_tensor(&device, &[m as u64, d as u64], &hidden_values);
            let norm = f32_tensor(&device, &[d as u64], &norm_values);
            let gated = bf16_tensor(&device, &[m as u64, heads as u64, width as u64], &gated_values);
            for &mapping in mappings(m) {
                let (x, slack) = operand_rows(&normed, d, mapping, Some(format));
                let projected = attention_project::native_for_device_with(
                    &device,
                    attention_project::Elements {
                        NW: Element::f32(),
                        QW: format.resident(),
                        GW: format.resident(),
                        KW: format.resident(),
                        VW: format.resident(),
                        A: Element::bf16(),
                    },
                    &mapping.projection_params(statics_specialization(&[
                        ("D", d),
                        ("PT", 1),
                        ("NM", d),
                        ("Q", segments[0]),
                        ("GR", segments[1]),
                        ("K", segments[2]),
                        ("V", segments[3]),
                        ("QS", 1),
                        ("GS", 1),
                        ("KS", 1),
                        ("VS", 1),
                    ])),
                )
                .unwrap()
                .call(attention_project::Args {
                    hidden: &hidden,
                    input_norm: &norm,
                    query_weight: &projections[0].tensor,
                    gate_weight: &projections[1].tensor,
                    key_weight: &projections[2].tensor,
                    value_weight: &projections[3].tensor,
                    epsilon: EPSILON,
                    project_mode: 0,
                    query_scale: &scales[0],
                    gate_scale: &scales[1],
                    key_scale: &scales[2],
                    value_scale: &scales[3],
                    query_bound: &none,
                    gate_bound: &none,
                    key_bound: &none,
                    value_bound: &none,
                })
                .unwrap();
                let outputs = [&projected.r0, &projected.r1, &projected.r2, &projected.r3];
                for (segment, projected) in outputs.into_iter().enumerate() {
                    let segment_factor = f64::from(SCALES[segment]);
                    let (values, tolerance) = rounded_segment(
                        &x,
                        &slack,
                        &projections[segment],
                        m,
                        segments[segment],
                        d,
                        mapping,
                    );
                    check(
                        &format!(
                            "scaled attention project {format:?} M={m} {mapping:?} segment {segment}"
                        ),
                        &read_bf16(projected),
                        &values.iter().map(|value| segment_factor * value).collect::<Vec<_>>(),
                        &tolerance
                            .iter()
                            .map(|tolerance| segment_factor * tolerance + 1e-6)
                            .collect::<Vec<_>>(),
                    );
                }

                let (x, _) = operand_rows(&gated_values, k, mapping, Some(format));
                let (projected, magnitude) = project(&x, &output.values, m, d, k);
                let dequant = dequant_bound(&x, &output.values, m, d, k, mapping);
                let expected: Vec<f64> = (0..m * d)
                    .map(|i| f64::from(hidden_values[i]) + factor * projected[i])
                    .collect();
                let tolerance: Vec<f64> = (0..m * d)
                    .map(|i| {
                        factor * (projected[i].abs() / 256.0 + magnitude[i] * 2e-5 + dequant[i])
                            + 1e-6
                    })
                    .collect();
                let result = attention_output::native_for_device_with(
                    &device,
                    attention_output::Elements {
                        A: Element::bf16(),
                        OW: format.resident(),
                    },
                    &mapping.projection_params(statics_specialization(&[
                        ("D", d),
                        ("SN", 1),
                        ("IN", 0),
                        ("Q", heads),
                        ("W", width),
                        ("OS", 1),
                    ])),
                )
                .unwrap()
                .call(attention_output::Args {
                    hidden: &hidden,
                    inject: &no_inject(&device, hidden.extents()[0]),
                    gated: &gated,
                    output_weight: &output.tensor,
                    output_scale: &scale,
                    output_bound: &none,
                })
                .unwrap()
                .value;
                check(
                    &format!("scaled attention output {format:?} M={m} {mapping:?}"),
                    &read_f32(&result),
                    &expected,
                    &tolerance,
                );
            }
        }
    }
}

/// The inject port of a single residual stream: no columns.
fn no_inject(device: &seismic::Device, rows: u64) -> seismic::Tensor {
    seismic::Tensor::zeros(device, seismic::Element::f32(), &[rows, 0]).unwrap()
}
