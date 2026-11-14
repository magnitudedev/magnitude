//! CUDA recurrent and attention projection entries
//! (`kernels/{attention,recurrent}.seismic`) against host models over mma16
//! weights. The recurrent output projection is `attention_output` over the
//! gated rows the state entries publish. Tests return early without CUDA.

mod cuda_common;

use cuda_common::*;
use magnitude_kernels::{attention_output, attention_project, gated_delta_project};
use seismic::{Device, Element, NativeSpecialization};
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

/// [`project_with_slack`] published in A: each value and its tolerance.
fn rounded_segment(
    x: &[f32],
    slack: &[f64],
    weight: &Weight,
    m: usize,
    n: usize,
    k: usize,
    mapping: Mapping,
) -> (Vec<f64>, Vec<f64>) {
    let (values, sums) = project_with_slack(x, slack, weight, m, n, k, mapping);
    let tolerance = values.iter().zip(&sums).map(|(y, d)| published_bound(*y, *d)).collect();
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

/// IQ4_XS weights (iq4xs).
#[test]
fn cuda_recurrent_project_iq4_matches_host_model() {
    recurrent_project_cases(&[[Some(Format::IQ4_XS); 4]], 43);
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
        let projection = Mapped::new(
            gated_delta_project::native_entry_with(gated_delta_project::Elements {
                NW: Element::f32(),
                QW: element(formats[0]),
                GW: element(formats[1]),
                AW: element(formats[2]),
                BW: element(formats[3]),
                A: Element::bf16(),
            }),
            |_: &Device, mapping: Mapping| {
                mapping.projection_params(statics_specialization(&[
                    ("H", h),
                    ("PT", 1),
                    ("NM", h),
                    ("NK", nk),
                    ("NV", nv),
                    ("W", w),
                ]))
            },
        );
        for m in ROWS {
            let hidden_values: Vec<f32> = (0..m * h).map(|_| rng.uniform(-3.0, 3.0)).collect();
            let norm_values: Vec<f32> = (0..h).map(|_| rng.uniform(0.25, 1.25)).collect();
            let (normed, flips) = normalized(&hidden_values, &norm_values, m, h);
            let hidden = f32_tensor(&device, &[m as u64, h as u64], &hidden_values);
            let norm = f32_tensor(&device, &[h as u64], &norm_values);
            for mapping in mappings(m, |mapping| projection.admits(&device, mapping)) {
                let (x, operand) = operand_rows(&normed, h, mapping, formats[0]);
                let slack = sum_slack(&flips, &operand);
                let mut expected = vec![0.0; m * width];
                let mut tolerance = vec![0.0; m * width];
                for (weight, (offset, n)) in
                    weights
                        .iter()
                        .zip([(0, qkv), (qkv, z), (qkv + z, nv), (qkv + z + nv, nv)])
                {
                    let (values, bound) = rounded_segment(&x, &slack, weight, m, n, h, mapping);
                    place(&mut expected, width, offset, &values, m, n);
                    place(&mut tolerance, width, offset, &bound, m, n);
                }
                let projected = projection
                    .kernel(&device, mapping)
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
                assert_within(
                    &format!("recurrent project {formats:?} M={m} {mapping:?}"),
                    &read_bf16(&projected),
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

/// IQ4_XS weights (iq4xs).
#[test]
fn cuda_attention_project_iq4_matches_host_model() {
    attention_project_cases(&[[Format::IQ4_XS; 3]], 59);
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
        let projection = Mapped::new(
            attention_project::native_entry_with(attention_project::Elements {
                NW: Element::f32(),
                QW: formats[0].resident(),
                GW: formats[0].resident(),
                KW: formats[1].resident(),
                VW: formats[2].resident(),
                A: Element::bf16(),
                Y: Element::bf16(),
            }),
            |_: &Device, mapping: Mapping| {
                mapping.projection_params(statics_specialization(&[
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
                ]))
            },
        );
        for m in ROWS {
            let hidden_values: Vec<f32> = (0..m * d).map(|_| rng.uniform(-3.0, 3.0)).collect();
            let norm_values: Vec<f32> = (0..d).map(|_| rng.uniform(0.25, 1.25)).collect();
            let (normed, flips) = normalized(&hidden_values, &norm_values, m, d);
            let hidden = f32_tensor(&device, &[m as u64, d as u64], &hidden_values);
            let norm = f32_tensor(&device, &[d as u64], &norm_values);
            for mapping in mappings(m, |mapping| projection.admits(&device, mapping)) {
                seismic_native_tests::progress(format_args!(
                    "attention project {formats:?} M={m} {mapping:?}"
                ));
                let (x, operand) = operand_rows(&normed, d, mapping, Some(formats[0]));
                let slack = sum_slack(&flips, &operand);
                let expected: Vec<(Vec<f64>, Vec<f64>)> = weights
                    .iter()
                    .zip([query, keys, keys])
                    .map(|(weight, n)| rounded_segment(&x, &slack, weight, m, n, d, mapping))
                    .collect();
                let project = |project_mode| {
                    projection
                        .kernel(&device, mapping)
                        .call(attention_project::Args {
                            hidden: &hidden,
                            input_norm: &norm,
                            query_weight: &weights[0].tensor,
                            gate_weight: &gate,
                            key_weight: &weights[1].tensor,
                            value_weight: &weights[2].tensor,
                            epsilon: EPSILON,
                            project_mode,
                            query_scale: &f32_tensor(&device, &[0], &[]),
                            gate_scale: &f32_tensor(&device, &[0], &[]),
                            key_scale: &f32_tensor(&device, &[0], &[]),
                            value_scale: &f32_tensor(&device, &[0], &[]),
                            query_bound: &f32_tensor(&device, &[0], &[]),
                            gate_bound: &f32_tensor(&device, &[0], &[]),
                            key_bound: &f32_tensor(&device, &[0], &[]),
                            value_bound: &f32_tensor(&device, &[0], &[]),
                        })
                        .unwrap()
                };
                let results = project(0);
                let zeroed = project(1);
                for (name, tensor, (values, tolerance)) in [
                    ("query", &results.r0, &expected[0]),
                    ("key", &results.r2, &expected[1]),
                    ("value", &results.r3, &expected[2]),
                    ("key beside zeroed segments", &zeroed.r2, &expected[1]),
                    ("value beside zeroed segments", &zeroed.r3, &expected[2]),
                ] {
                    assert_within(
                        &format!("attention project {name} {formats:?} M={m} {mapping:?}"),
                        &read_bf16(tensor),
                        values,
                        tolerance,
                    );
                }
                for (name, tensor) in [("query", &zeroed.r0), ("gate", &zeroed.r1)] {
                    assert!(
                        read_bf16(tensor).iter().all(|&value| value == 0.0),
                        "zeroed {name} {formats:?} M={m} {mapping:?}"
                    );
                }
            }
        }
    }
}

/// `attention_output` of `q` heads of `w` onto `d` rows through a `weight`,
/// under second-level scales or not (`os`).
fn attention_output_projection(
    weight: Element,
    d: usize,
    q: usize,
    w: usize,
    os: usize,
) -> Mapped<attention_output::Entry, impl Fn(&Device, Mapping) -> NativeSpecialization> {
    Mapped::new(
        attention_output::native_entry_with(attention_output::Elements {
            A: Element::bf16(),
            OW: weight,
        }),
        move |_: &Device, mapping: Mapping| {
            mapping.projection_params(statics_specialization(&[
                ("D", d),
                ("SN", 1),
                ("IN", 0),
                ("Q", q),
                ("W", w),
                ("OS", os),
            ]))
        },
    )
}

fn attention_output_matches_host_model(format: Format) {
    let Some(device) = cuda() else { return };
    let mut rng = Rng(59);
    let (d, q, w) = (272usize, 4usize, 128usize);
    let output = weight(&device, format, d, q * w, &mut rng);
    let projection = attention_output_projection(format.resident(), d, q, w, 0);
    for m in ROWS {
        let gated_values: Vec<f32> = (0..m * q * w)
            .map(|_| bf16_round(rng.uniform(-1.0, 1.0)))
            .collect();
        let hidden_values: Vec<f32> = (0..m * d).map(|_| rng.uniform(-2.0, 2.0)).collect();
        let gated_tensor = bf16_tensor(&device, &[m as u64, q as u64, w as u64], &gated_values);
        let hidden = f32_tensor(&device, &[m as u64, d as u64], &hidden_values);
        for mapping in mappings(m, |mapping| projection.admits(&device, mapping)) {
            // The heads are exact input, so the INT8 emulation is exact too.
            let (x, _) = operand_rows(&gated_values, q * w, mapping, Some(format));
            let (projected, sums) =
                project(&x, &output.values, format.resident(), m, d, q * w, mapping);
            let (expected, tolerance): (Vec<f64>, Vec<f64>) = (0..m * d)
                .map(|i| {
                    let residual = f64::from(hidden_values[i]);
                    (residual + projected[i], residual_bound(residual, projected[i], sums[i]))
                })
                .unzip();
            let result = projection
                .kernel(&device, mapping)
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
            assert_within(
                &format!("attention output {format:?} M={m} {mapping:?}"),
                &read_f32(&result),
                &expected,
                &tolerance,
            );
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
    for format in [Format::NVFP4, Format::NVFP4_A4, Format::Q4K] {
        let projections = segments.map(|rows| weight(&device, format, rows, d, &mut rng));
        let output = weight(&device, format, d, k, &mut rng);
        let projection = Mapped::new(
            attention_project::native_entry_with(attention_project::Elements {
                NW: Element::f32(),
                QW: format.resident(),
                GW: format.resident(),
                KW: format.resident(),
                VW: format.resident(),
                A: Element::bf16(),
                Y: Element::bf16(),
            }),
            |_: &Device, mapping: Mapping| {
                mapping.projection_params(statics_specialization(&[
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
                ]))
            },
        );
        let output_projection = attention_output_projection(format.resident(), d, heads, width, 1);
        for m in ROWS {
            let hidden_values: Vec<f32> = (0..m * d).map(|_| rng.uniform(-3.0, 3.0)).collect();
            let norm_values: Vec<f32> = (0..d).map(|_| rng.uniform(0.25, 1.25)).collect();
            let gated_values: Vec<f32> = (0..m * k)
                .map(|_| bf16_round(rng.uniform(-1.0, 1.0)))
                .collect();
            let (normed, flips) = normalized(&hidden_values, &norm_values, m, d);
            let hidden = f32_tensor(&device, &[m as u64, d as u64], &hidden_values);
            let norm = f32_tensor(&device, &[d as u64], &norm_values);
            let gated = bf16_tensor(
                &device,
                &[m as u64, heads as u64, width as u64],
                &gated_values,
            );
            for mapping in mappings(m, |mapping| {
                projection.admits(&device, mapping) && output_projection.admits(&device, mapping)
            }) {
                let (x, operand) = operand_rows(&normed, d, mapping, Some(format));
                let slack = sum_slack(&flips, &operand);
                let projected = projection
                    .kernel(&device, mapping)
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
                    let (values, sums) = project_with_slack(
                        &x,
                        &slack,
                        &projections[segment],
                        m,
                        segments[segment],
                        d,
                        mapping,
                    );
                    // The F32 sum times the scale, rounded to F32, then to A.
                    let (scaled, tolerance): (Vec<f64>, Vec<f64>) = values
                        .iter()
                        .zip(&sums)
                        .map(|(value, sums)| {
                            let scaled = segment_factor * value;
                            let product =
                                Float::F32.rounded_from_exact(segment_factor.abs() * sums, scaled);
                            (scaled, published_bound(scaled, product))
                        })
                        .unzip();
                    assert_within(
                        &format!(
                            "scaled attention project {format:?} M={m} {mapping:?} segment {segment}"
                        ),
                        &read_bf16(projected),
                        &scaled,
                        &tolerance,
                    );
                }

                let (x, _) = operand_rows(&gated_values, k, mapping, Some(format));
                let (projected, sums) =
                    project(&x, &output.values, format.resident(), m, d, k, mapping);
                // residual + round_A(fl(scale · ŝ)), in F32.
                let (expected, tolerance): (Vec<f64>, Vec<f64>) = (0..m * d)
                    .map(|i| {
                        let residual = f64::from(hidden_values[i]);
                        let scaled = factor * projected[i];
                        let product = Float::F32.rounded_from_exact(factor.abs() * sums[i], scaled);
                        (residual + scaled, residual_bound(residual, scaled, product))
                    })
                    .unzip();
                let result = output_projection
                    .kernel(&device, mapping)
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
                assert_within(
                    &format!("scaled attention output {format:?} M={m} {mapping:?}"),
                    &read_f32(&result),
                    &expected,
                    &tolerance,
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GGUF formats: a module of tests a format (`gguf::<format>::`), generated
// from the registry.

gguf_format_tests! {
    mma16;
    every: [attention_output_matches_host_model];
    partial_block: [];
    own_storage: [];
}

/// The inject port of a single residual stream: no columns.
fn no_inject(device: &seismic::Device, rows: u64) -> seismic::Tensor {
    seismic::Tensor::zeros(device, seismic::Element::f32(), &[rows, 0]).unwrap()
}
