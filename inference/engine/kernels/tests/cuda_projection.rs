//! CUDA K1 / K2 / K5 entries (`kernels/cuda_{dense,target,sampling}.seismic`)
//! against host models, over mma16 weights produced by the registry's host
//! conversion from random GGUF blocks. The dense host model is pinned to the
//! portable body by the reference interpreter. Tests return early without a
//! CUDA device.

use magnitude_kernels::{
    dense_expand, dense_output, embedding_rows, head_logits_rows, readout_features_rows,
    readout_head_rows, readout_selected_rows, sample_rows, shape_rows,
};
mod cuda_common;

use cuda_common::*;
use seismic::{Device, Element, Layout, NativeSpecialization, Tensor};
use seismic_lang::registry::bf16_round;

/// O output rows gathered through `out_rows` from M = O + 2 residual rows.
struct DenseCase {
    o: usize,
    m: usize,
    h: usize,
    f: usize,
    out_rows: Vec<i32>,
    residual: Vec<f32>,
    norm: Vec<f32>,
}

impl DenseCase {
    fn new(o: usize, h: usize, f: usize, rng: &mut Rng) -> Self {
        let m = o + 2;
        let out_rows = (0..o).map(|i| ((i * 5 + 3) % m) as i32).collect::<Vec<_>>();
        let residual = (0..m * h).map(|_| rng.uniform(-3.0, 3.0)).collect();
        let norm = (0..h).map(|_| rng.uniform(0.25, 1.25)).collect();
        Self {
            o,
            m,
            h,
            f,
            out_rows,
            residual,
            norm,
        }
    }
    /// The residual rows the entry reads, in output-row order.
    fn selected(&self) -> Vec<f32> {
        self.out_rows
            .iter()
            .flat_map(|row| {
                self.residual[*row as usize * self.h..(*row as usize + 1) * self.h]
                    .iter()
                    .copied()
            })
            .collect()
    }
    /// Expected SiLU(gate) * up on `mapping`'s operand path and the
    /// per-element tolerance.
    fn expand(&self, gate: &Weight, up: &Weight, mapping: Mapping) -> (Vec<f64>, Vec<f64>) {
        let (x, slack) = operand_rows(
            &normalized(&self.selected(), &self.norm, self.o, self.h),
            self.h,
            mapping,
        );
        let (g, gm) = project(&x, &gate.values, self.o, self.f, self.h);
        let (u, um) = project(&x, &up.values, self.o, self.f, self.h);
        let with_dequant = |slack: Vec<f64>, weight: &Weight| {
            let bound = dequant_bound(&x, &weight.values, self.o, self.f, self.h, mapping);
            slack
                .iter()
                .zip(&bound)
                .map(|(s, b)| s + b)
                .collect::<Vec<_>>()
        };
        let gs = with_dequant(
            slack_bound(&slack, &gate.values, self.o, self.f, self.h),
            gate,
        );
        let us = with_dequant(slack_bound(&slack, &up.values, self.o, self.f, self.h), up);
        let mut expected = vec![0.0; self.o * self.f];
        let mut tolerance = vec![0.0; self.o * self.f];
        for i in 0..expected.len() {
            expected[i] = silu(g[i]) * u[i];
            // Three bf16 roundings of the epilogue, plus the prologue's own
            // bf16 rounding of x, whose ties can resolve differently under
            // the device's reduction order and approximate rsqrt (bounded by
            // 2^-12 of the absolute products, scaled by the other factor),
            // plus the operand path's slack.
            tolerance[i] = expected[i].abs() * 4.0 / 256.0
                + ((gm[i] * 2.5e-4 + gs[i]) * u[i].abs() + (um[i] * 2.5e-4 + us[i]) * g[i].abs())
                + gs[i] * us[i]
                + 1e-6;
        }
        (expected, tolerance)
    }
}

fn expand_kernel(
    device: &Device,
    format: Format,
    case: &DenseCase,
    mapping: Mapping,
) -> seismic::NativeKernel<dense_expand::Entry> {
    dense_expand::native_for_device_with(
        device,
        dense_expand::Elements {
            NW: Element::f32(),
            GW: format.resident(),
            UW: format.resident(),
            A: Element::bf16(),
        },
        &mapping.dense_expand_params(
            NativeSpecialization::new()
                .with_static("H", case.h as u64)
                .with_static("F", case.f as u64)
                .with_static("GS", 0)
                .with_static("US", 0),
        ),
    )
    .unwrap()
}

fn output_kernel(
    device: &Device,
    format: Format,
    h: usize,
    f: usize,
    mapping: Mapping,
) -> seismic::NativeKernel<dense_output::Entry> {
    dense_output::native_for_device_with(
        device,
        dense_output::Elements {
            DW: format.resident(),
            A: Element::bf16(),
        },
        &mapping.dense_output_params(
            NativeSpecialization::new()
                .with_static("H", h as u64)
                .with_static("F", f as u64)
                .with_static("DS", 0),
        ),
    )
    .unwrap()
}

// GEMV (1..8), two-block GEMV (12, 16), GEMM (17 and beyond).
const ROWS: [usize; 10] = [1, 2, 3, 8, 12, 16, 17, 40, 64, 128];

#[test]
fn cuda_dense_expand_matches_host_model() {
    dense_expand_cases(&[Format::Q4K, Format::Q5K, Format::Q6K, Format::Q8], 7);
}

/// IQ4_XS / IQ4_NL weights are resident as iq4g32.
#[test]
fn cuda_dense_expand_iq4_matches_host_model() {
    dense_expand_cases(&[Format::Iq4], 9);
}

/// Q4_0, Q5_0, Q5_1, MXFP4 and NVFP4 gate/up weights (the scaled GEMM for
/// all but Q5_1).
#[test]
fn cuda_dense_expand_coded_matches_host_model() {
    dense_expand_cases(
        &[
            Format::Q4_0,
            Format::Q5_0,
            Format::Q5_1,
            Format::Mxfp4,
            Format::Nvfp4,
        ],
        17,
    );
}

fn dense_expand_cases(formats: &[Format], seed: u64) {
    let Some(device) = cuda() else { return };
    let mut rng = Rng(seed);
    // 17 row tiles: a GEMV group tail and a GEMM block-column tail.
    let (h, f) = (512, 272);
    for &format in formats {
        let gate = weight(&device, format, f, h, &mut rng);
        let up = weight(&device, format, f, h, &mut rng);
        for o in ROWS {
            let case = DenseCase::new(o, h, f, &mut rng);
            let residual = f32_tensor(&device, &[case.m as u64, h as u64], &case.residual);
            let norm = f32_tensor(&device, &[h as u64], &case.norm);
            let out_rows = i32_tensor(&device, &[o as u64], &case.out_rows);
            for &mapping in mappings(o) {
                let (expected, tolerance) = case.expand(&gate, &up, mapping);
                let product = expand_kernel(&device, format, &case, mapping)
                    .call(dense_expand::Args {
                        residual: &residual,
                        norm: &norm,
                        gate_weight: &gate.tensor,
                        up_weight: &up.tensor,
                        out_rows: &out_rows,
                        eps: EPSILON,
                        activation: 0,
                        gate_scale: &f32_tensor(&device, &[0], &[]),
                        up_scale: &f32_tensor(&device, &[0], &[]),
                    })
                    .unwrap()
                    .value;
                check(
                    &format!("expand {format:?} O={o} {mapping:?}"),
                    &read_bf16(&product),
                    &expected,
                    &tolerance,
                );
            }
        }
    }
}

#[test]
fn cuda_dense_output_matches_host_model() {
    let Some(device) = cuda() else { return };
    let mut rng = Rng(11);
    dense_output_cases(&device, &Format::ALL, 272, 512, &mut rng);
}

/// Rows whose K is not a multiple of 256 (a partial last superblock):
/// Gemma's 704 and 2112, Nemotron's 2688.
#[test]
fn cuda_dense_output_32_granular_k_matches_host_model() {
    let Some(device) = cuda() else { return };
    let mut rng = Rng(13);
    let formats = [
        Format::Q8,
        Format::Q4_0,
        Format::Q5_0,
        Format::Q5_1,
        Format::Mxfp4,
        Format::Nvfp4,
    ];
    for f in [704, 2112, 2688] {
        dense_output_cases(&device, &formats, 48, f, &mut rng);
    }
}

fn dense_output_cases(device: &Device, formats: &[Format], h: usize, f: usize, rng: &mut Rng) {
    let device = device.clone();
    for &format in formats {
        let down = weight(&device, format, h, f, rng);
        for o in ROWS {
            let m = o + 2;
            let out_rows_values: Vec<i32> = (0..o).map(|i| ((i * 5 + 3) % m) as i32).collect();
            let product_values: Vec<f32> = (0..o * f)
                .map(|_| bf16_round(rng.uniform(-1.0, 1.0)))
                .collect();
            let residual_values: Vec<f32> = (0..m * h).map(|_| rng.uniform(-2.0, 2.0)).collect();
            let product = bf16_tensor(&device, &[o as u64, f as u64], &product_values);
            let residual = f32_tensor(&device, &[m as u64, h as u64], &residual_values);
            let out_rows = i32_tensor(&device, &[o as u64], &out_rows_values);
            for &mapping in mappings(o) {
                // The product is exact input, so the INT8 emulation is exact too.
                let (x, _) = operand_rows(&product_values, f, mapping);
                let (projected, magnitude) = project(&x, &down.values, o, h, f);
                let expected: Vec<f64> = (0..o * h)
                    .map(|i| {
                        f64::from(residual_values[out_rows_values[i / h] as usize * h + i % h])
                            + projected[i]
                    })
                    .collect();
                // One A rounding of the projection, F32 accumulation and the
                // GEMM's weight dequantization.
                // A scaled-GEMM format's weights enter exactly: no
                // dequantization slack.
                let dequant = if format.scaled_gemm() {
                    vec![0.0; o * h]
                } else {
                    dequant_bound(&x, &down.values, o, h, f, mapping)
                };
                let tolerance: Vec<f64> = (0..o * h)
                    .map(|i| projected[i].abs() / 256.0 + magnitude[i] * 2e-5 + dequant[i] + 1e-6)
                    .collect();
                let result = output_kernel(&device, format, h, f, mapping)
                    .call(dense_output::Args {
                        residual: &residual,
                        product: &product,
                        down_weight: &down.tensor,
                        out_rows: &out_rows,
                        down_scale: &f32_tensor(&device, &[0], &[]),
                    })
                    .unwrap()
                    .value;
                check(
                    &format!("output {format:?} O={o} {mapping:?}"),
                    &read_f32(&result),
                    &expected,
                    &tolerance,
                );
            }
        }
    }
}

/// FNV-1a of an output's bytes: GEMM restructurings that keep the arithmetic
/// print the same digests (compared across builds).
fn digest(tensor: &Tensor) -> u64 {
    tensor
        .read_to_host()
        .unwrap()
        .iter()
        .fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3)
        })
}

/// The large GEMM over several row bands (ordered block walk, a partial last
/// band) and, for the output, a K with a partial last superblock, under every
/// loader rotation: each result matches the host model, the rotations agree
/// bit for bit, and each result's digest is printed (to compare across
/// builds).
#[test]
fn cuda_gemm_row_bands_match_host_model() {
    let Some(device) = cuda() else { return };
    let mut rng = Rng(23);
    let (h, f) = (512, 272);
    let rotations = [2, 1, 4].map(|rotate| Mapping {
        rotate,
        ..GEMM_MAPPINGS[0]
    });
    let mapping = rotations[0];
    for format in Format::ALL {
        // The output projects K codes onto f rows: 704 (11 k-blocks, a
        // partial last superblock) where the format's blocks allow it.
        let k = if format.block_values() == 256 {
            1280
        } else {
            704
        };
        let gate = weight(&device, format, f, h, &mut rng);
        let up = weight(&device, format, f, h, &mut rng);
        let down = weight(&device, format, f, k, &mut rng);
        // 100: one band, the output split over K; 300 and 1100: several
        // bands, the last partial.
        for o in [100usize, 300, 1100] {
            let case = DenseCase::new(o, h, f, &mut rng);
            let residual = f32_tensor(&device, &[case.m as u64, h as u64], &case.residual);
            let norm = f32_tensor(&device, &[h as u64], &case.norm);
            let out_rows = i32_tensor(&device, &[o as u64], &case.out_rows);
            let (expected, tolerance) = case.expand(&gate, &up, mapping);
            let label = format!("expand {format:?} O={o}");
            let digests = rotations.map(|rotation| {
                let product = expand_kernel(&device, format, &case, rotation)
                    .call(dense_expand::Args {
                        residual: &residual,
                        norm: &norm,
                        gate_weight: &gate.tensor,
                        up_weight: &up.tensor,
                        out_rows: &out_rows,
                        eps: EPSILON,
                        activation: 0,
                        gate_scale: &f32_tensor(&device, &[0], &[]),
                        up_scale: &f32_tensor(&device, &[0], &[]),
                    })
                    .unwrap()
                    .value;
                check(&label, &read_bf16(&product), &expected, &tolerance);
                digest(&product)
            });
            assert!(
                digests.iter().all(|value| *value == digests[0]),
                "{label}: rotations differ {digests:x?}"
            );
            println!("BITS {label} {:016x}", digests[0]);

            let m = o + 2;
            let rows: Vec<i32> = (0..o).map(|i| ((i * 5 + 3) % m) as i32).collect();
            let values: Vec<f32> = (0..o * k)
                .map(|_| bf16_round(rng.uniform(-1.0, 1.0)))
                .collect();
            let base: Vec<f32> = (0..m * f).map(|_| rng.uniform(-2.0, 2.0)).collect();
            let (x, _) = operand_rows(&values, k, mapping);
            let (projected, magnitude) = project(&x, &down.values, o, f, k);
            let dequant = if format.scaled_gemm() {
                vec![0.0; o * f]
            } else {
                dequant_bound(&x, &down.values, o, f, k, mapping)
            };
            let expected: Vec<f64> = (0..o * f)
                .map(|i| f64::from(base[rows[i / f] as usize * f + i % f]) + projected[i])
                .collect();
            let tolerance: Vec<f64> = (0..o * f)
                .map(|i| projected[i].abs() / 256.0 + magnitude[i] * 2e-5 + dequant[i] + 1e-6)
                .collect();
            let residual = f32_tensor(&device, &[m as u64, f as u64], &base);
            let product = bf16_tensor(&device, &[o as u64, k as u64], &values);
            let out_rows = i32_tensor(&device, &[o as u64], &rows);
            let label = format!("output {format:?} O={o}");
            let digests = rotations.map(|rotation| {
                let result = output_kernel(&device, format, f, k, rotation)
                    .call(dense_output::Args {
                        residual: &residual,
                        product: &product,
                        down_weight: &down.tensor,
                        out_rows: &out_rows,
                        down_scale: &f32_tensor(&device, &[0], &[]),
                    })
                    .unwrap()
                    .value;
                check(&label, &read_f32(&result), &expected, &tolerance);
                digest(&result)
            });
            assert!(
                digests.iter().all(|value| *value == digests[0]),
                "{label}: rotations differ {digests:x?}"
            );
            println!("BITS {label} {:016x}", digests[0]);
        }
    }
}

#[test]
#[ignore]
fn cuda_dense_expand_scoped_tuning_is_factored() {
    let Some(device) = cuda() else { return };
    let absent_scale = f32_tensor(&device, &[0], &[]);
    let mut rng = Rng(731);
    let (h, f) = (512usize, 272usize);
    let gate = weight(&device, Format::Q4K, f, h, &mut rng);
    let up = weight(&device, Format::Q4K, f, h, &mut rng);
    let norm_values = (0..h).map(|_| rng.uniform(0.25, 1.25)).collect::<Vec<_>>();
    let norm = f32_tensor(&device, &[h as u64], &norm_values);
    let rows = [1usize, 12, 32, 128];
    let inputs = rows
        .iter()
        .map(|&o| {
            let case = DenseCase::new(o, h, f, &mut rng);
            (
                f32_tensor(&device, &[case.m as u64, h as u64], &case.residual),
                i32_tensor(&device, &[o as u64], &case.out_rows),
            )
        })
        .collect::<Vec<_>>();
    let points: Vec<_> = rows
        .iter()
        .zip(&inputs)
        .map(|(rows, (residual, out_rows))| seismic::TuningPoint {
            label: format!("rows{rows}"),
            weight: 1.0,
            class: None,
            cost: 1.0,
            required: false,
            rotation: vec![dense_expand::Args {
                residual,
                norm: &norm,
                gate_weight: &gate.tensor,
                up_weight: &up.tensor,
                out_rows,
                eps: EPSILON,
                activation: 0,
                gate_scale: &absent_scale,
                up_scale: &absent_scale,
            }],
            initialize: None,
            written: Default::default(),
        })
        .collect();
    let search = seismic::Strategy::Search(seismic::SearchPlan {
        allowance: std::time::Duration::from_secs(600),
        admission: std::time::Duration::from_secs(600),
        required: std::time::Duration::from_secs(600),
        settings: seismic::SearchSettings {
            improvement: 0.01,
            restarts: 2,
            confirmed: 3,
            default_margin: 0.02,
            samples: 5,
            confirmation_samples: 5,
        },
        min_sample_seconds: 0.001,
        start: Vec::new(),
    });
    let statics = NativeSpecialization::new()
        .with_static("H", h as u64)
        .with_static("F", f as u64)
        .with_static("GS", 0)
        .with_static("US", 0);
    let result = dense_expand::native_tune_with(
        &device,
        dense_expand::Elements {
            NW: Element::f32(),
            GW: Format::Q4K.resident(),
            UW: Format::Q4K.resident(),
            A: Element::bf16(),
        },
        &statics,
        points,
        seismic::PrecisionPolicy::bounded(seismic::precision::Tolerance {
            absolute: seismic::precision::Limit::new(0.01).unwrap(),
            relative: seismic::precision::Limit::new(0.05).unwrap(),
            relative_floor: seismic::precision::Limit::ZERO,
            ulps: None,
        }),
        search,
        seismic::TuningReference::Portable,
    )
    .unwrap();
    assert!(matches!(
        result.method,
        seismic::TuningMethod::Factored { complete: true, .. }
    ));
    assert_eq!(result.rejections().count(), 0);
    println!(
        "dense_expand CUDA BF16: choice {:?}, method {:?}, time {:?}, records {}",
        result.overall.launches,
        result.method,
        result.time,
        result.configurations.len()
    );
}

#[test]
#[ignore]
fn cuda_dense_output_scoped_tuning_is_factored() {
    let Some(device) = cuda() else { return };
    let absent_scale = f32_tensor(&device, &[0], &[]);
    let mut rng = Rng(741);
    let (h, f) = (272usize, 512usize);
    let down = weight(&device, Format::Q5K, h, f, &mut rng);
    let rows = [1usize, 12, 32, 128];
    let inputs = rows
        .iter()
        .map(|&m| {
            let product = (0..m * f)
                .map(|_| bf16_round(rng.uniform(-1.0, 1.0)))
                .collect::<Vec<_>>();
            let residual = (0..m * h)
                .map(|_| rng.uniform(-2.0, 2.0))
                .collect::<Vec<_>>();
            (
                f32_tensor(&device, &[m as u64, h as u64], &residual),
                bf16_tensor(&device, &[m as u64, f as u64], &product),
                i32_tensor(&device, &[m as u64], &(0..m as i32).collect::<Vec<_>>()),
            )
        })
        .collect::<Vec<_>>();
    let points: Vec<_> = rows
        .iter()
        .zip(&inputs)
        .map(
            |(rows, (residual, product, out_rows))| seismic::TuningPoint {
                label: format!("rows{rows}"),
                weight: 1.0,
                class: None,
                cost: 1.0,
                required: false,
                rotation: vec![dense_output::Args {
                    residual,
                    product,
                    down_weight: &down.tensor,
                    out_rows,
                    down_scale: &absent_scale,
                }],
                initialize: None,
                written: Default::default(),
            },
        )
        .collect();
    let search = seismic::Strategy::Search(seismic::SearchPlan {
        allowance: std::time::Duration::from_secs(600),
        admission: std::time::Duration::from_secs(600),
        required: std::time::Duration::from_secs(600),
        settings: seismic::SearchSettings {
            improvement: 0.01,
            restarts: 2,
            confirmed: 3,
            default_margin: 0.02,
            samples: 5,
            confirmation_samples: 5,
        },
        min_sample_seconds: 0.001,
        start: Vec::new(),
    });
    let statics = NativeSpecialization::new()
        .with_static("H", h as u64)
        .with_static("F", f as u64)
        .with_static("DS", 0);
    let result = dense_output::native_tune_with(
        &device,
        dense_output::Elements {
            DW: Format::Q5K.resident(),
            A: Element::bf16(),
        },
        &statics,
        points,
        seismic::PrecisionPolicy::bounded(seismic::precision::Tolerance {
            absolute: seismic::precision::Limit::new(0.01).unwrap(),
            relative: seismic::precision::Limit::new(0.05).unwrap(),
            relative_floor: seismic::precision::Limit::ZERO,
            ulps: None,
        }),
        search,
        seismic::TuningReference::Portable,
    )
    .unwrap();
    assert!(matches!(
        result.method,
        seismic::TuningMethod::Factored { complete: true, .. }
    ));
    assert_eq!(result.rejections().count(), 0);
    println!(
        "dense_output CUDA BF16: choice {:?}, method {:?}, time {:?}, records {}",
        result.overall.launches,
        result.method,
        result.time,
        result.configurations.len()
    );
}

/// Dense bf16 weights (row-major, unpadded) bind like packed ones on every
/// path; INT8 has no effect with them (the INT8 GEMM needs packed weights).
#[test]
fn cuda_dense_weights_match_host_model() {
    let Some(device) = cuda() else { return };
    let mut rng = Rng(23);
    let (h, f) = (512, 320);
    let gate = dense_weight(&device, f, h, &mut rng);
    let up = dense_weight(&device, f, h, &mut rng);
    let down = dense_weight(&device, h, f, &mut rng);
    let bf16 = Element::bf16();
    for o in ROWS {
        let case = DenseCase::new(o, h, f, &mut rng);
        let residual = f32_tensor(&device, &[case.m as u64, h as u64], &case.residual);
        let norm = f32_tensor(&device, &[h as u64], &case.norm);
        let out_rows = i32_tensor(&device, &[o as u64], &case.out_rows);
        for &mapping in mappings(o) {
            let statics = NativeSpecialization::new()
                .with_static("H", h as u64)
                .with_static("F", f as u64);
            // INT8 has no effect with dense weights: every mapping runs the
            // 16-bit path.
            let (expected, tolerance) = case.expand(&gate, &up, Mapping { int8: 0, ..mapping });
            let product = dense_expand::native_for_device_with(
                &device,
                dense_expand::Elements {
                    NW: Element::f32(),
                    GW: bf16,
                    UW: bf16,
                    A: bf16,
                },
                &mapping
                    .dense_expand_params(statics.clone().with_static("GS", 0).with_static("US", 0)),
            )
            .unwrap()
            .call(dense_expand::Args {
                residual: &residual,
                norm: &norm,
                gate_weight: &gate.tensor,
                up_weight: &up.tensor,
                out_rows: &out_rows,
                eps: EPSILON,
                activation: 0,
                gate_scale: &f32_tensor(&device, &[0], &[]),
                up_scale: &f32_tensor(&device, &[0], &[]),
            })
            .unwrap()
            .value;
            check(
                &format!("dense expand O={o} {mapping:?}"),
                &read_bf16(&product),
                &expected,
                &tolerance,
            );

            let product_values: Vec<f32> = (0..o * f)
                .map(|_| bf16_round(rng.uniform(-1.0, 1.0)))
                .collect();
            let product = bf16_tensor(&device, &[o as u64, f as u64], &product_values);
            let (projected, magnitude) = project(&product_values, &down.values, o, h, f);
            let expected: Vec<f64> = (0..o * h)
                .map(|i| {
                    f64::from(case.residual[case.out_rows[i / h] as usize * h + i % h])
                        + projected[i]
                })
                .collect();
            let tolerance: Vec<f64> = (0..o * h)
                .map(|i| projected[i].abs() / 256.0 + magnitude[i] * 2e-5 + 1e-6)
                .collect();
            let result = dense_output::native_for_device_with(
                &device,
                dense_output::Elements { DW: bf16, A: bf16 },
                &mapping.dense_output_params(statics.with_static("DS", 0)),
            )
            .unwrap()
            .call(dense_output::Args {
                residual: &residual,
                product: &product,
                down_weight: &down.tensor,
                out_rows: &out_rows,
                down_scale: &f32_tensor(&device, &[0], &[]),
            })
            .unwrap()
            .value;
            check(
                &format!("dense output O={o} {mapping:?}"),
                &read_f32(&result),
                &expected,
                &tolerance,
            );
        }
    }
}

/// Every row of a GEMV class (2..=GEMV_ROWS rows, staged and two-block paths)
/// equals the same row projected alone (M = 1, formed in the GEMV block) bit
/// for bit: a verification row does not depend on its peers or on M.
#[test]
fn cuda_gemv_rows_match_single_row_bits() {
    let Some(device) = cuda() else { return };
    let mut rng = Rng(19);
    let (h, f) = (512, 768);
    for format in Format::ALL {
        let gate = weight(&device, format, f, h, &mut rng);
        let up = weight(&device, format, f, h, &mut rng);
        let down = weight(&device, format, h, f, &mut rng);
        let rows = GEMV_ROWS;
        let residual_values: Vec<f32> = (0..rows * h).map(|_| rng.uniform(-3.0, 3.0)).collect();
        let norm = f32_tensor(
            &device,
            &[h as u64],
            &(0..h).map(|_| rng.uniform(0.25, 1.25)).collect::<Vec<_>>(),
        );
        let product_values: Vec<f32> = (0..rows * f)
            .map(|_| bf16_round(rng.uniform(-1.0, 1.0)))
            .collect();
        for &mapping in &GEMV_MAPPINGS {
            let expand = dense_expand::native_for_device_with(
                &device,
                dense_expand::Elements {
                    NW: Element::f32(),
                    GW: format.resident(),
                    UW: format.resident(),
                    A: Element::bf16(),
                },
                &mapping.dense_expand_params(
                    NativeSpecialization::new()
                        .with_static("H", h as u64)
                        .with_static("F", f as u64)
                        .with_static("GS", 0)
                        .with_static("US", 0),
                ),
            )
            .unwrap();
            let output = output_kernel(&device, format, h, f, mapping);
            let run = |o: usize, first: usize| {
                let residual = f32_tensor(
                    &device,
                    &[o as u64, h as u64],
                    &residual_values[first * h..(first + o) * h],
                );
                let product = bf16_tensor(
                    &device,
                    &[o as u64, f as u64],
                    &product_values[first * f..(first + o) * f],
                );
                let out_rows = i32_tensor(&device, &[o as u64], &(0..o as i32).collect::<Vec<_>>());
                let expanded = expand
                    .call(dense_expand::Args {
                        residual: &residual,
                        norm: &norm,
                        gate_weight: &gate.tensor,
                        up_weight: &up.tensor,
                        out_rows: &out_rows,
                        eps: EPSILON,
                        activation: 0,
                        gate_scale: &f32_tensor(&device, &[0], &[]),
                        up_scale: &f32_tensor(&device, &[0], &[]),
                    })
                    .unwrap()
                    .value;
                let residual_rows = f32_tensor(&device, &[o as u64, h as u64], &vec![0.0; o * h]);
                let projected = output
                    .call(dense_output::Args {
                        residual: &residual_rows,
                        product: &product,
                        down_weight: &down.tensor,
                        out_rows: &out_rows,
                        down_scale: &f32_tensor(&device, &[0], &[]),
                    })
                    .unwrap()
                    .value;
                (read_bf16(&expanded), read_f32(&projected))
            };
            let singles: Vec<(Vec<f32>, Vec<f32>)> = (0..rows).map(|row| run(1, row)).collect();
            for o in [2usize, 5, 8, 9, 12, 16] {
                let (expanded, projected) = run(o, 0);
                for row in 0..o {
                    let label = format!("{format:?} {mapping:?} O={o} row {row}");
                    let bits =
                        |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                    assert_eq!(
                        bits(&expanded[row * f..(row + 1) * f]),
                        bits(&singles[row].0),
                        "expand {label}"
                    );
                    assert_eq!(
                        bits(&projected[row * h..(row + 1) * h]),
                        bits(&singles[row].1),
                        "output {label}"
                    );
                }
            }
        }
    }
}

/// Expected head logits of a case (features of the `out_rows` rows against
/// every vocabulary row) and their tolerance.
fn head_expected(
    case: &DenseCase,
    weight: &Weight,
    v: usize,
    mapping: Mapping,
) -> (Vec<f64>, Vec<f64>) {
    let (x, slack) = operand_rows(
        &normalized(&case.selected(), &case.norm, case.o, case.h),
        case.h,
        mapping,
    );
    let (logits, magnitude) = project(&x, &weight.values, case.o, v, case.h);
    let bound = slack_bound(&slack, &weight.values, case.o, v, case.h);
    let dequant = dequant_bound(&x, &weight.values, case.o, v, case.h, mapping);
    // F32 accumulation plus the prologue's bf16 ties (as for the expand), the
    // operand path's slack and the GEMM's weight dequantization.
    let tolerance = (0..logits.len())
        .map(|i| magnitude[i] * 2.5e-4 + bound[i] + dequant[i] + 1e-6)
        .collect();
    (logits, tolerance)
}

#[test]
fn cuda_head_rows_match_host_model() {
    let Some(device) = cuda() else { return };
    let mut rng = Rng(13);
    let (d, v) = (512, 272);
    for format in Format::ALL {
        let head = weight(&device, format, v, d, &mut rng);
        for o in ROWS {
            let case = DenseCase::new(o, d, v, &mut rng);
            let hidden = f32_tensor(&device, &[case.m as u64, d as u64], &case.residual);
            let norm = f32_tensor(&device, &[d as u64], &case.norm);
            let out_rows = i32_tensor(&device, &[o as u64], &case.out_rows);
            for &mapping in mappings(o) {
                // The head's GEMMs are the 16-bit path.
                let (expected, tolerance) =
                    head_expected(&case, &head, v, Mapping { int8: 0, ..mapping });
                let logits = readout_head_rows::native_for_device_with(
                    &device,
                    readout_head_rows::Elements {
                        NW: Element::f32(),
                        OW: format.resident(),
                        A: Element::bf16(),
                    },
                    &mapping.head_params(
                        NativeSpecialization::new()
                            .with_static("V", v as u64)
                            .with_static("D", d as u64)
                            .with_static("WS", 0),
                    ),
                )
                .unwrap()
                .call(readout_head_rows::Args {
                    hidden: &hidden,
                    norm: &norm,
                    weight: &head.tensor,
                    out_rows: &out_rows,
                    epsilon: EPSILON,
                    softcap: 0.0,
                    weight_scale: &f32_tensor(&device, &[0], &[]),
                })
                .unwrap()
                .value;
                check(
                    &format!("head {format:?} O={o} {mapping:?}"),
                    &read_f32(&logits),
                    &expected,
                    &tolerance,
                );
            }
        }
    }
}

#[test]
fn cuda_head_logits_scoped_launches_match_host_model() {
    let Some(device) = cuda() else { return };
    let mut rng = Rng(83);
    let (d, v) = (256, 256);
    let head = weight(&device, Format::Q6K, v, d, &mut rng);
    for rows in [1, 8, 32, 128] {
        let features = (0..rows * d)
            .map(|_| bf16_round(rng.uniform(-1.0, 1.0)))
            .collect::<Vec<_>>();
        let (expected, magnitude) = project(&features, &head.values, rows, v, d);
        let tolerance = magnitude
            .iter()
            .map(|value| value * 0.001 + 1e-5)
            .collect::<Vec<_>>();
        for ksplit in [2, 4] {
            let specialization = NativeSpecialization::new()
                .with_static("V", v as u64)
                .with_static("D", d as u64)
                .with_launch_param(0, "KSPLIT", ksplit)
                .with_launch_param(1, "KSPLIT", ksplit)
                .with_launch_param(3, "ROTATE", 2);
            let kernel = head_logits_rows::native_for_device_with(
                &device,
                head_logits_rows::Elements {
                    A: Element::bf16(),
                    OW: Format::Q6K.resident(),
                },
                &specialization,
            )
            .unwrap();
            let logits = kernel
                .call(head_logits_rows::Args {
                    features: &bf16_tensor(&device, &[rows as u64, d as u64], &features),
                    weight: &head.tensor,
                })
                .unwrap()
                .value;
            check(
                &format!("head_logits rows={rows} ksplit={ksplit}"),
                &read_f32(&logits),
                &expected,
                &tolerance,
            );
        }
    }
}

#[test]
fn cuda_features_and_selected_rows_match_host_model() {
    let Some(device) = cuda() else { return };
    let mut rng = Rng(17);
    let (d, v) = (512, 272);
    let selected_values = [0i32, 271, 16, 15, 100, 100, 7];
    let sv = selected_values.len();
    for format in Format::ALL {
        let head = weight(&device, format, v, d, &mut rng);
        for o in [1usize, 3, 9] {
            let case = DenseCase::new(o, d, v, &mut rng);
            let (logits, tolerance) = head_expected(&case, &head, v, GEMV_MAPPINGS[0]);
            let hidden = f32_tensor(&device, &[case.m as u64, d as u64], &case.residual);
            let norm = f32_tensor(&device, &[d as u64], &case.norm);
            let out_rows = i32_tensor(&device, &[o as u64], &case.out_rows);
            let features = readout_features_rows::native_for_device_with(
                &device,
                readout_features_rows::Elements {
                    NW: Element::f32(),
                    A: Element::bf16(),
                },
                &NativeSpecialization::new().with_static("D", d as u64),
            )
            .unwrap()
            .call(readout_features_rows::Args {
                hidden: &hidden,
                norm: &norm,
                out_rows: &out_rows,
                epsilon: EPSILON,
            })
            .unwrap()
            .value;
            // The device and host differ only in the order of the square sum
            // and in rsqrt: at most one bf16 step, and only rarely.
            let expected_features = normalized(&case.selected(), &case.norm, o, d);
            let actual_features = read_bf16(&features);
            let mut off = 0;
            for (a, e) in actual_features.iter().zip(&expected_features) {
                let step = (a.to_bits() as i64 - e.to_bits() as i64).abs() >> 16;
                assert!(step <= 1, "features {format:?} O={o}: {a} vs {e}");
                off += step;
            }
            assert!(
                off * 100 <= actual_features.len() as i64,
                "features {format:?} O={o}: {off} values one step off"
            );
            let selected = i32_tensor(&device, &[sv as u64], &selected_values);
            let actual = readout_selected_rows::native_for_device_with(
                &device,
                readout_selected_rows::Elements {
                    NW: Element::f32(),
                    OW: format.resident(),
                    A: Element::bf16(),
                },
                &NativeSpecialization::new()
                    .with_static("V", v as u64)
                    .with_static("D", d as u64),
            )
            .unwrap()
            .call(readout_selected_rows::Args {
                hidden: &hidden,
                norm: &norm,
                weight: &head.tensor,
                out_rows: &out_rows,
                selected: &selected,
                epsilon: EPSILON,
                softcap: 0.0,
            })
            .unwrap()
            .value;
            let expected: Vec<f64> = (0..o * sv)
                .map(|i| logits[(i / sv) * v + selected_values[i % sv] as usize])
                .collect();
            let tolerance: Vec<f64> = (0..o * sv)
                .map(|i| tolerance[(i / sv) * v + selected_values[i % sv] as usize])
                .collect();
            check(
                &format!("selected {format:?} O={o}"),
                &read_f32(&actual),
                &expected,
                &tolerance,
            );
        }
    }
}

/// The dense host model is the portable body's meaning: the reference
/// interpreter over the same mma16 bytes agrees with it.
#[test]
fn dense_expand_host_model_matches_portable_body() {
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
    let mut rng = Rng(3);
    let (o, h, f) = (2, 256, 32);
    let mut sources = seismic_std::sources();
    sources.push(SourceFile {
        path: "dense_rows.seismic".into(),
        text: include_str!("../kernels/dense_rows.seismic").into(),
    });
    sources.push(SourceFile {
        path: "functions.seismic".into(),
        text: include_str!("../kernels/functions.seismic").into(),
    });
    let module = check_source(sources).unwrap();
    for format in Format::ALL {
        let gate = weight(&device, format, f, h, &mut rng);
        let up = weight(&device, format, f, h, &mut rng);
        let case = DenseCase::new(o, h, f, &mut rng);
        let (expected, tolerance) = case.expand(&gate, &up, GEMV_MAPPINGS[0]);
        let storage = registry::storage(format.representation(), Layout::Mma16).unwrap();
        let logical = module
            .entry(
                module.entry_named("dense_expand").unwrap(),
                &ElementBindings::new()
                    .bind("NW", registry::dense(DType::F32))
                    .bind("GW", storage)
                    .bind("UW", storage)
                    .bind("A", registry::dense(DType::BF16)),
            )
            .unwrap();
        let mut interpreter = Interpreter::new(&logical);
        let floats = |values: &[f32]| values.iter().map(|x| f64::from(*x)).collect::<Vec<_>>();
        let rows = case
            .out_rows
            .iter()
            .map(|x| f64::from(*x))
            .collect::<Vec<_>>();
        let args =
            vec![
                Arg::Tensor(interpreter.add_tensor(TensorData::dense(
                    DType::F32,
                    vec![case.m, h],
                    floats(&case.residual),
                ))),
                Arg::Tensor(interpreter.add_tensor(TensorData::dense(
                    DType::F32,
                    vec![h],
                    floats(&case.norm),
                ))),
                Arg::Tensor(interpreter.add_tensor(
                    TensorData::encoded(storage, vec![f, h], gate.bytes.clone()).unwrap(),
                )),
                Arg::Tensor(interpreter.add_tensor(
                    TensorData::encoded(storage, vec![f, h], up.bytes.clone()).unwrap(),
                )),
                Arg::Tensor(interpreter.add_tensor(TensorData::dense(DType::I32, vec![o], rows))),
                Arg::Scalar(ReferenceScalar::F32(EPSILON.to_bits())),
                Arg::Scalar(ReferenceScalar::I32(0)),
                Arg::Tensor(interpreter.add_tensor(TensorData::dense(DType::F32, vec![0], vec![]))),
                Arg::Tensor(interpreter.add_tensor(TensorData::dense(DType::F32, vec![0], vec![]))),
            ];
        let outcome = interpreter.run(&args).unwrap();
        if let SourceTermination::Failed(failure) = outcome.termination() {
            panic!("dense_expand portable body failed: {failure}");
        }
        let result = outcome.results().next().unwrap();
        let OutcomeValue::Tensor(product) = result.value() else {
            panic!("tensor result")
        };
        let actual: Vec<f32> = (0..o * f)
            .map(|i| product.read(i).unwrap() as f32)
            .collect();
        check(
            &format!("portable expand {format:?}"),
            &actual,
            &expected,
            &tolerance,
        );
    }
}

#[test]
fn cuda_embedding_rows_decode_table_rows() {
    let Some(device) = cuda() else { return };
    let mut rng = Rng(5);
    let (v, d) = (40, 512);
    for format in Format::ALL {
        let table = weight(&device, format, v, d, &mut rng);
        let tokens_values = [0, 39, 17, 16, 15, 3];
        // (token, status) rows, the `sample_rows` result layout.
        let rows = tokens_values
            .iter()
            .flat_map(|token| [*token, 0])
            .collect::<Vec<_>>();
        let tokens = i32_tensor(&device, &[tokens_values.len() as u64, 2], &rows);
        let result = embedding_rows::native_for_device_with(
            &device,
            embedding_rows::Elements {
                EW: format.resident(),
                A: Element::bf16(),
            },
            &NativeSpecialization::new().with_static("D", d as u64),
        )
        .unwrap()
        .call(embedding_rows::Args {
            table: &table.tensor,
            tokens: &tokens,
            scale: 1.0,
            normalize: 0,
            epsilon: 0.0,
        })
        .unwrap();
        let rounded = read_bf16(&result.r0);
        let wide = read_f32(&result.r1);
        for (row, token) in tokens_values.iter().enumerate() {
            for column in 0..d {
                let expected = bf16_round(table.values[*token as usize * d + column] as f32);
                assert_eq!(
                    rounded[row * d + column].to_bits(),
                    expected.to_bits(),
                    "{format:?} row {row} col {column}"
                );
                assert_eq!(
                    wide[row * d + column].to_bits(),
                    expected.to_bits(),
                    "{format:?} row {row} col {column}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Sampling.

/// The portable body's Philox4x32-10 Gumbel score.
fn gumbel(value: f32, token: u32, draw: &[u32; 6]) -> f32 {
    if draw[0] != 1 {
        return value;
    }
    let (mut c0, mut c1, mut c2, mut c3) = (token, draw[3], draw[4], draw[5]);
    let (mut k0, mut k1) = (draw[1], draw[2]);
    for _ in 0..10 {
        let p0 = u64::from(3528531795u32) * u64::from(c0);
        let p1 = u64::from(3449720151u32) * u64::from(c2);
        let next0 = (p1 >> 32) as u32 ^ c1 ^ k0;
        let next2 = (p0 >> 32) as u32 ^ c3 ^ k1;
        c0 = next0;
        c1 = p1 as u32;
        c2 = next2;
        c3 = p0 as u32;
        k0 = k0.wrapping_add(2654435769);
        k1 = k1.wrapping_add(3144134277);
    }
    let uniform = ((c0 >> 9) as f32 + 0.5) * 0.00000011920928955078125;
    value - (-uniform.ln()).ln()
}

fn sample_host(logits: &[f32], mask: &[u32], constrained: bool, draw: &[u32; 6]) -> (i32, i32) {
    let mut bad = false;
    let mut best: Option<(f32, usize)> = None;
    for (token, value) in logits.iter().enumerate() {
        bad |= value.is_nan() || *value == f32::INFINITY;
        if (constrained && (mask[token / 32] >> (token % 32)) & 1 == 0) || !value.is_finite() {
            continue;
        }
        let score = gumbel(*value, token as u32, draw);
        if best.is_none_or(|(s, _)| score > s) {
            best = Some((score, token));
        }
    }
    match (bad, best) {
        (true, _) => (-1, 2),
        (false, None) => (-1, 1),
        (false, Some((_, token))) => (token as i32, 0),
    }
}

#[test]
fn cuda_sample_rows_match_portable_semantics_for_every_partition() {
    let Some(device) = cuda() else { return };
    let mut rng = Rng(19);
    let v = 5000usize;
    let words = v.div_ceil(32);
    let m = 5usize;
    let mut logits: Vec<f32> = (0..m * v).map(|_| rng.uniform(-8.0, 8.0)).collect();
    let mut mask: Vec<u32> = (0..m * words).map(|_| rng.next() | rng.next()).collect();
    // Row 2: everything masked (empty); row 3: a NaN (nonfinite source);
    // row 4: ties at the maximum (lowest index wins).
    mask[2 * words..3 * words].fill(0);
    logits[3 * v + 77] = f32::NAN;
    logits[4 * v + 100] = 50.0;
    logits[4 * v + 4000] = 50.0;
    mask[4 * words + 100 / 32] |= 1 << (100 % 32);
    mask[4 * words + 4000 / 32] |= 1 << (4000 % 32);
    let draws: Vec<[u32; 6]> = (0..m)
        .map(|row| {
            [
                u32::from(row % 2 == 0),
                rng.next(),
                rng.next(),
                17,
                row as u32,
                0,
            ]
        })
        .map(|mut d| {
            if d[4] == 4 {
                d[0] = 0;
            }
            d
        })
        .collect();
    // Row 0 is unconstrained: its random mask is ignored.
    let constrained = [0_i32, 1, 1, 1, 1];
    let expected: Vec<(i32, i32)> = (0..m)
        .map(|row| {
            sample_host(
                &logits[row * v..(row + 1) * v],
                &mask[row * words..(row + 1) * words],
                constrained[row] != 0,
                &draws[row],
            )
        })
        .collect();
    let logits_tensor = f32_tensor(&device, &[m as u64, v as u64], &logits);
    let mask_tensor = u32_tensor(&device, &[m as u64, words as u64], &mask);
    let constrained_tensor = i32_tensor(&device, &[m as u64], &constrained);
    let draws_tensor = u32_tensor(&device, &[m as u64, 6], &draws.concat());
    for parts in [16u64, 32, 64, 128] {
        for width in [256u64, 512] {
            let mut result = i32_tensor(&device, &[m as u64, 2], &vec![7; m * 2]);
            sample_rows::native_for_device(
                &device,
                &NativeSpecialization::new()
                    .with_static("V", v as u64)
                    .with_param("PARTS", parts)
                    .with_param("WIDTH", width),
            )
            .unwrap()
            .call(sample_rows::Args {
                logits: &logits_tensor,
                mask: &mask_tensor,
                constrained: &constrained_tensor,
                draws: &draws_tensor,
                result: &mut result,
            })
            .unwrap();
            let actual = read_i32(&result);
            for row in 0..m {
                assert_eq!(
                    (actual[row * 2], actual[row * 2 + 1]),
                    expected[row],
                    "row {row} PARTS={parts} WIDTH={width}"
                );
            }
        }
    }
}

/// The portable `shape_rows` body, row by row.
fn shape_host(logits: &[f32], params: &[f32; 8], history: &[i32]) -> Vec<f32> {
    let [temperature, top_k, top_p, min_p, repetition, presence, frequency, _] = *params;
    let top_k = top_k as i32;
    let penalized: Vec<f32> = logits
        .iter()
        .enumerate()
        .map(|(token, value)| {
            let count = history.iter().filter(|h| **h == token as i32).count();
            let mut value = *value;
            if count > 0 {
                value = if value < 0.0 {
                    value * repetition
                } else {
                    value / repetition
                };
                value = value - presence - frequency * count as f32;
            }
            value
        })
        .collect();
    if temperature == 0.0 {
        return penalized;
    }
    let tempered: Vec<f32> = penalized.iter().map(|v| v / temperature).collect();
    let mut sorted = tempered.clone();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let topk: Vec<f32> = tempered
        .iter()
        .map(|v| {
            let greater = sorted.partition_point(|s| s > v);
            if top_k > 0 && greater >= top_k as usize {
                f32::NEG_INFINITY
            } else {
                *v
            }
        })
        .collect();
    let maximum = topk.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let minp: Vec<f32> = topk
        .iter()
        .map(|v| {
            if min_p > 0.0 && (v - maximum).exp() < min_p {
                f32::NEG_INFINITY
            } else {
                *v
            }
        })
        .collect();
    if top_p >= 1.0 {
        return minp;
    }
    let shifted: Vec<f64> = minp
        .iter()
        .map(|v| f64::from((v - maximum).exp()))
        .collect();
    let total: f64 = shifted.iter().sum();
    let mut order: Vec<usize> = (0..minp.len()).collect();
    order.sort_by(|a, b| minp[*b].partial_cmp(&minp[*a]).unwrap().then(a.cmp(b)));
    let mut out = minp.clone();
    let mut preceding = 0.0;
    for token in order {
        if preceding / total >= f64::from(top_p) {
            out[token] = f32::NEG_INFINITY;
        }
        preceding += shifted[token];
    }
    out
}

#[test]
fn cuda_shape_rows_match_portable_semantics() {
    let Some(device) = cuda() else { return };
    let mut rng = Rng(23);
    let (v, hn) = (3000usize, 16usize);
    let rows: Vec<[f32; 8]> = vec![
        [0.0, 0.0, 1.0, 0.0, 1.3, 0.2, 0.1, 0.0], // greedy: penalties only
        [0.7, 40.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0], // top-k
        [1.0, 0.0, 0.9, 0.0, 1.1, 0.0, 0.05, 0.0], // top-p
        [0.8, 0.0, 1.0, 0.05, 1.0, 0.0, 0.0, 0.0], // min-p
        [0.6, 64.0, 0.8, 0.02, 1.2, 0.3, 0.1, 0.0], // everything
    ];
    let sx = rows.len();
    let mut logits: Vec<f32> = (0..sx * v).map(|_| rng.uniform(-6.0, 6.0)).collect();
    // Ties at the top-k boundary region.
    for row in 0..sx {
        for token in [5usize, 6, 7, 2999] {
            logits[row * v + token] = 5.5;
        }
    }
    let history: Vec<i32> = (0..sx * hn)
        .map(|i| {
            if i % 7 == 0 {
                -1
            } else {
                (rng.next() % v as u32) as i32
            }
        })
        .collect();
    let params_tensor = f32_tensor(&device, &[sx as u64, 8], &rows.concat());
    let history_tensor = i32_tensor(&device, &[sx as u64, hn as u64], &history);
    // Unconstrained rows: their empty masks are never read.
    let mask_tensor = u32_tensor(
        &device,
        &[sx as u64, v.div_ceil(32) as u64],
        &vec![0; sx * v.div_ceil(32)],
    );
    let constrained_tensor = i32_tensor(&device, &[sx as u64], &vec![0; sx]);
    for parts in [32u64, 64, 128] {
        for width in [256u64, 512] {
            // Shaped in place: every configuration shapes fresh logits.
            let mut shaped = f32_tensor(&device, &[sx as u64, v as u64], &logits);
            shape_rows::native_for_device(
                &device,
                &NativeSpecialization::new()
                    .with_static("V", v as u64)
                    .with_static("Hn", hn as u64)
                    .with_param("PARTS", parts)
                    .with_param("WIDTH", width),
            )
            .unwrap()
            .call(shape_rows::Args {
                logits: &mut shaped,
                mask: &mask_tensor,
                constrained: &constrained_tensor,
                params: &params_tensor,
                history: &history_tensor,
            })
            .unwrap();
            let actual = read_f32(&shaped);
            for (row, params) in rows.iter().enumerate() {
                let expected = shape_host(
                    &logits[row * v..(row + 1) * v],
                    params,
                    &history[row * hn..(row + 1) * hn],
                );
                for token in 0..v {
                    let (a, e) = (actual[row * v + token], expected[token]);
                    assert!(
                        a.to_bits() == e.to_bits(),
                        "row {row} token {token} PARTS={parts} WIDTH={width}: {a} vs {e}"
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Timings on the 4B geometry (run explicitly on the measurement host), over
// zero-filled resident weights (`timing_weight`).

/// The GEMM band (16-bit path) of the dense entries over prefill rows at real
/// geometries: Qwen3.5-4B (H 2560, F 9216), MiniCPM5-2B (H 2048, F 6144) and
/// the Qwen3.6-35B-A3B shared expert (H 2048, F 512), under the loader
/// rotations `CUDA_TIMING_ROTATE` (comma-separated, default 2). Each (entry,
/// rows, rotation) is measured `CUDA_TIMING_ROUNDS` times (default 3),
/// reporting the fastest and the median round, which separates the kernel
/// from host and device noise.
#[test]
#[ignore = "timing; run explicitly on the measurement host"]
fn cuda_gemm_timings() {
    let Some(device) = cuda() else { return };
    let absent_scale = f32_tensor(&device, &[0], &[]);
    let mut rng = Rng(37);
    let options = seismic::MeasureOptions {
        samples: 15,
        min_sample_seconds: 0.002,
    };
    let rounds: usize =
        std::env::var("CUDA_TIMING_ROUNDS").map_or(3, |value| value.parse().unwrap());
    let shapes: Vec<(&str, usize, usize)> = match std::env::var("CUDA_TIMING_SHAPES") {
        Ok(list) => list
            .split(',')
            .map(|name| match name.trim() {
                "4b" => ("4b", 2560, 9216),
                "2b" => ("2b", 2048, 6144),
                "35b" => ("35b", 2048, 512),
                other => panic!("CUDA_TIMING_SHAPES: unknown shape {other}"),
            })
            .collect(),
        Err(_) => vec![("4b", 2560, 9216)],
    };
    let rotates: Vec<u64> = std::env::var("CUDA_TIMING_ROTATE").map_or(vec![2], |list| {
        list.split(',')
            .map(|rotate| {
                rotate
                    .trim()
                    .parse()
                    .expect("CUDA_TIMING_ROTATE: loader rotations")
            })
            .collect()
    });
    let summary = |mut times: Vec<f64>| {
        times.sort_by(f64::total_cmp);
        (times[0], times[times.len() / 2])
    };
    for (name, h, f) in shapes {
        // Rotations exceed the 24 MiB L2 at the 4B geometry.
        let rotation = 4;
        for format in timing_formats() {
            let gates: Vec<Tensor> = (0..rotation)
                .map(|_| timing_weight(&device, format, f, h))
                .collect();
            let ups: Vec<Tensor> = (0..rotation)
                .map(|_| timing_weight(&device, format, f, h))
                .collect();
            let downs: Vec<Tensor> = (0..rotation)
                .map(|_| timing_weight(&device, format, h, f))
                .collect();
            for m in timing_rows(&[128, 512, 2048]) {
                let case = DenseCase::new(m, h, f, &mut rng);
                let residual = f32_tensor(&device, &[case.m as u64, h as u64], &case.residual);
                let norm = f32_tensor(&device, &[h as u64], &case.norm);
                let out_rows = i32_tensor(&device, &[m as u64], &case.out_rows);
                let product = bf16_tensor(&device, &[m as u64, f as u64], &vec![0.5; m * f]);
                for &rotate in &rotates {
                    let mapping = Mapping {
                        rotate,
                        ..GEMM_MAPPINGS[0]
                    };
                    let expand = expand_kernel(&device, format, &case, mapping);
                    let output = output_kernel(&device, format, h, f, mapping);
                    let mut expand_times = Vec::new();
                    let mut output_times = Vec::new();
                    for _ in 0..rounds {
                        let args = (0..rotation)
                            .map(|r| dense_expand::Args {
                                residual: &residual,
                                norm: &norm,
                                gate_weight: &gates[r],
                                up_weight: &ups[r],
                                out_rows: &out_rows,
                                eps: EPSILON,
                                activation: 0,
                                gate_scale: &absent_scale,
                                up_scale: &absent_scale,
                            })
                            .collect();
                        expand_times.push(expand.measure(args, &options).unwrap().median);
                        let args = (0..rotation)
                            .map(|r| dense_output::Args {
                                residual: &residual,
                                product: &product,
                                down_weight: &downs[r],
                                out_rows: &out_rows,
                                down_scale: &absent_scale,
                            })
                            .collect();
                        output_times.push(output.measure(args, &options).unwrap().median);
                    }
                    let (best, median) = summary(expand_times);
                    println!(
                        "gemm {name} expand {format:?} M={m} R={rotate}: {:.1} us ({:.1} median), {:.2} TFLOP/s",
                        best * 1e6,
                        median * 1e6,
                        4.0 * (m * h * f) as f64 / best / 1e12
                    );
                    let (best, median) = summary(output_times);
                    println!(
                        "gemm {name} output {format:?} M={m} R={rotate}: {:.1} us ({:.1} median), {:.2} TFLOP/s",
                        best * 1e6,
                        median * 1e6,
                        2.0 * (m * h * f) as f64 / best / 1e12
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "timing; run explicitly on the measurement host"]
fn cuda_projection_timings() {
    let Some(device) = cuda() else { return };
    let absent_scale = f32_tensor(&device, &[0], &[]);
    let mut rng = Rng(29);
    let options = seismic::MeasureOptions {
        samples: 15,
        min_sample_seconds: 0.002,
    };
    let (h, f) = (2560usize, 9216usize);
    // Rotations exceed the 24 MiB L2.
    let rotation = 4;
    for format in timing_formats() {
        let gates: Vec<Tensor> = (0..rotation)
            .map(|_| timing_weight(&device, format, f, h))
            .collect();
        let ups: Vec<Tensor> = (0..rotation)
            .map(|_| timing_weight(&device, format, f, h))
            .collect();
        let downs: Vec<Tensor> = (0..rotation)
            .map(|_| timing_weight(&device, format, h, f))
            .collect();
        let expand_bytes = gates[0].byte_len() as f64 * 2.0;
        let output_bytes = downs[0].byte_len() as f64;
        // Decode, the verification / concurrency curve (MTP), prefill.
        for m in timing_rows(&[1, 2, 4, 8, 12, 16, 32, 128, 256, 512]) {
            let case = DenseCase::new(m, h, f, &mut rng);
            let residual = f32_tensor(&device, &[case.m as u64, h as u64], &case.residual);
            let norm = f32_tensor(&device, &[h as u64], &case.norm);
            let out_rows = i32_tensor(&device, &[m as u64], &case.out_rows);
            let product = bf16_tensor(&device, &[m as u64, f as u64], &vec![0.5; m * f]);
            for &mapping in mappings(m) {
                let expand = expand_kernel(&device, format, &case, mapping);
                let args = (0..rotation)
                    .map(|r| dense_expand::Args {
                        residual: &residual,
                        norm: &norm,
                        gate_weight: &gates[r],
                        up_weight: &ups[r],
                        out_rows: &out_rows,
                        eps: EPSILON,
                        activation: 0,
                        gate_scale: &absent_scale,
                        up_scale: &absent_scale,
                    })
                    .collect();
                let measured = expand.measure(args, &options).unwrap();
                println!(
                    "expand {format:?} M={m} K={h} N=2x{f} {mapping:?}: {:.1} us, {:.0} GB/s, {:.2} TFLOP/s",
                    measured.median * 1e6,
                    expand_bytes / measured.median / 1e9,
                    4.0 * (m * h * f) as f64 / measured.median / 1e12
                );
                let output = output_kernel(&device, format, h, f, mapping);
                let args = (0..rotation)
                    .map(|r| dense_output::Args {
                        residual: &residual,
                        product: &product,
                        down_weight: &downs[r],
                        out_rows: &out_rows,
                        down_scale: &absent_scale,
                    })
                    .collect();
                let measured = output.measure(args, &options).unwrap();
                println!(
                    "output {format:?} M={m} K={f} N={h} {mapping:?}: {:.1} us, {:.0} GB/s, {:.2} TFLOP/s",
                    measured.median * 1e6,
                    output_bytes / measured.median / 1e9,
                    2.0 * (m * h * f) as f64 / measured.median / 1e12
                );
            }
        }
    }
}

/// Head (Q6_K, D = 2560) over 65536 vocabulary rows per weight (a quarter of
/// the 4B vocabulary; two rotations exceed the L2).
#[test]
#[ignore = "timing; run explicitly on the measurement host"]
fn cuda_head_timings() {
    let Some(device) = cuda() else { return };
    let mut rng = Rng(31);
    let options = seismic::MeasureOptions {
        samples: 15,
        min_sample_seconds: 0.002,
    };
    let (d, v) = (2560usize, 65536usize);
    let heads: Vec<Tensor> = (0..2)
        .map(|_| timing_weight(&device, Format::Q6K, v, d))
        .collect();
    let bytes = heads[0].byte_len() as f64;
    for o in [1usize, 2, 4, 8, 12, 16, 32, 128] {
        let case = DenseCase::new(o, d, v, &mut rng);
        let hidden = f32_tensor(&device, &[case.m as u64, d as u64], &case.residual);
        let norm = f32_tensor(&device, &[d as u64], &case.norm);
        let out_rows = i32_tensor(&device, &[o as u64], &case.out_rows);
        for &mapping in mappings(o) {
            let kernel = readout_head_rows::native_for_device_with(
                &device,
                readout_head_rows::Elements {
                    NW: Element::f32(),
                    OW: Format::Q6K.resident(),
                    A: Element::bf16(),
                },
                &mapping.head_params(
                    NativeSpecialization::new()
                        .with_static("V", v as u64)
                        .with_static("D", d as u64)
                        .with_static("WS", 0),
                ),
            )
            .unwrap();
            let absent_scale = f32_tensor(&device, &[0], &[]);
            let args = heads
                .iter()
                .map(|head| readout_head_rows::Args {
                    hidden: &hidden,
                    norm: &norm,
                    weight: head,
                    out_rows: &out_rows,
                    epsilon: EPSILON,
                    softcap: 0.0,
                    weight_scale: &absent_scale,
                })
                .collect();
            let measured = kernel.measure(args, &options).unwrap();
            println!(
                "head Q6K O={o} K={d} N={v} {mapping:?}: {:.1} us, {:.0} GB/s, {:.2} TFLOP/s",
                measured.median * 1e6,
                bytes / measured.median / 1e9,
                2.0 * (o * d * v) as f64 / measured.median / 1e12
            );
        }
    }
}
