//! Semantic precision tests; no native-selection or performance claim.
#[path = "support/reference.rs"]
mod reference;
use reference::{Arg, TensorData};
use seismic_lang::types::DType;
use std::collections::HashMap;
#[test]
fn layer_norm_centers_in_f32_then_publishes_bf16() {
    {
        let program = seismic_std::program().unwrap();
        let mut vm = reference::interpreter(&program);
        let source = vec![1001., 1002., 1003., 1004., 7., 7., 7., 7.];
        let scales = vec![1., 2., -1., 0.5];
        let shifts = vec![0.25, -0.5, 1., -2.];
        let x = vm.add_tensor(TensorData::dense(DType::F32, vec![2, 4], source.clone()));
        let weight = vm.add_tensor(TensorData::dense(DType::BF16, vec![4], scales.clone()));
        let bias = vm.add_tensor(TensorData::dense(DType::F32, vec![4], shifts.clone()));
        let out = vm.add_tensor(TensorData::dense(DType::BF16, vec![2, 4], vec![0.; 8]));
        reference::run(
            &mut vm,
            "layer_norm",
            &[
                Arg::Tensor(x),
                Arg::Tensor(weight),
                Arg::Tensor(bias),
                Arg::Tensor(out),
                Arg::Scalar(1e-6),
            ],
            &HashMap::from([("R".into(), 2), ("W".into(), 4)]),
        );
        for column in 0..4 {
            let value = ((source[column] as f32 - 1002.5) / (1.25f32 + 1e-6).sqrt())
                * scales[column] as f32
                + shifts[column] as f32;
            assert_eq!(
                vm.tensors[out].get(column),
                seismic_lang::numeric::bf16_round(value) as f64
            );
            assert_eq!(vm.tensors[out].get(4 + column), shifts[column]);
        }
    }
}
#[test]
fn affine_bias_precedes_compact_publication() {
    {
        let program = seismic_std::program().unwrap();
        let mut vm = reference::interpreter(&program);
        let x = vm.add_tensor(TensorData::dense(DType::BF16, vec![1, 2], vec![1., 1.]));
        let weight = vm.add_tensor(TensorData::dense(
            DType::F32,
            vec![1, 2],
            vec![1., 0.00390625],
        ));
        let bias = vm.add_tensor(TensorData::dense(
            DType::F32,
            vec![1],
            vec![0.001f32 as f64],
        ));
        let out = vm.add_tensor(TensorData::dense(DType::BF16, vec![1, 1], vec![0.]));
        reference::run(
            &mut vm,
            "linear_bias",
            &[
                Arg::Tensor(x),
                Arg::Tensor(weight),
                Arg::Tensor(bias),
                Arg::Tensor(out),
            ],
            &HashMap::from([("M".into(), 1), ("N".into(), 1), ("K".into(), 2)]),
        );
        assert_eq!(vm.tensors[out].get(0), 1.0078125);
        let premature = seismic_lang::numeric::bf16_round(
            seismic_lang::numeric::bf16_round(1.00390625) + 0.001,
        );
        assert_ne!(vm.tensors[out].get(0), premature as f64);
    }
}

#[test]
fn vision_patch_order_and_position_add_preserve_compact_publications() {
    {
        let program = seismic_engine::models::qwen35::program::program().unwrap();
        let mut vm = reference::interpreter(&program);
        let values = (0..24).map(|i| i as f64 + 0.03125).collect::<Vec<_>>();
        let pixels = vm.add_tensor(TensorData::dense(
            DType::F32,
            vec![1, 3, 2, 2, 2],
            values.clone(),
        ));
        let ordered = vm.add_tensor(TensorData::dense(
            DType::BF16,
            vec![1, 2, 2, 2, 3],
            vec![0.; 24],
        ));
        reference::run(
            &mut vm,
            "qwen_vision_patch_order",
            &[Arg::Tensor(pixels), Arg::Tensor(ordered)],
            &HashMap::from([
                ("M".into(), 1),
                ("C".into(), 3),
                ("T".into(), 2),
                ("P".into(), 2),
            ]),
        );
        for t in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    for c in 0..3 {
                        assert_eq!(
                            vm.tensors[ordered].get(((t * 2 + y) * 2 + x) * 3 + c),
                            seismic_lang::numeric::bf16_round(
                                values[((c * 2 + t) * 2 + y) * 2 + x] as f32
                            ) as f64
                        );
                    }
                }
            }
        }
        let bf = seismic_lang::numeric::bf16_round;
        let weights = [0.13f32, 0.27, 0.19, 0.41];
        let table_values = [1.125f32, -0.375, 2.25, 0.6875, 0.5625, -1.25, 3.5, 0.15625];
        let projected = vm.add_tensor(TensorData::dense(DType::BF16, vec![1, 2], vec![0.25, -0.5]));
        let table = vm.add_tensor(TensorData::dense(
            DType::BF16,
            vec![4, 2],
            table_values.iter().map(|&v| v as f64).collect(),
        ));
        let indices = vm.add_tensor(TensorData::dense(
            DType::I32,
            vec![1, 4],
            vec![2., 0., 3., 1.],
        ));
        let coefficients = vm.add_tensor(TensorData::dense(
            DType::F32,
            vec![1, 4],
            weights.iter().map(|&v| v as f64).collect(),
        ));
        let out = vm.add_tensor(TensorData::dense(DType::BF16, vec![1, 2], vec![0.; 2]));
        reference::run(
            &mut vm,
            "qwen_vision_position_add",
            &[
                Arg::Tensor(projected),
                Arg::Tensor(table),
                Arg::Tensor(indices),
                Arg::Tensor(coefficients),
                Arg::Tensor(out),
            ],
            &HashMap::from([("M".into(), 1), ("H".into(), 2), ("L".into(), 4)]),
        );
        for column in 0..2 {
            let parts: Vec<f32> = [2, 0, 3, 1]
                .iter()
                .zip(weights)
                .map(|(&index, coefficient)| bf(table_values[index * 2 + column] * bf(coefficient)))
                .collect();
            let sum = bf(bf(bf(parts[0] + parts[1]) + parts[2]) + parts[3]);
            assert_eq!(
                vm.tensors[out].get(column),
                bf([0.25, -0.5][column] + sum) as f64
            );
        }
    }
}

#[test]
fn vision_rotary_preserves_heads_axes_and_unrotated_values() {
    {
        let program = seismic_engine::models::qwen35::program::program().unwrap();
        let mut vm = reference::interpreter(&program);
        let (rows, heads, width) = (6usize, 3usize, 12usize);
        let positions = [[0i32, 0], [0, 1], [1, 0], [1, 1], [7, 19], [19, 7]];
        let values = (0..rows * 3 * heads * width)
            .map(|i| ((i * 17 % 101) as f32 - 50.) / 32.)
            .collect::<Vec<_>>();
        let projected = vm.add_tensor(TensorData::dense(
            DType::F32,
            vec![rows, 3, heads, width],
            values.iter().map(|&v| v as f64).collect(),
        ));
        let coords = vm.add_tensor(TensorData::dense(
            DType::I32,
            vec![rows, 2],
            positions.iter().flatten().map(|&v| v as f64).collect(),
        ));
        let outputs: Vec<_> = (0..3)
            .map(|_| {
                vm.add_tensor(TensorData::dense(
                    DType::F32,
                    vec![rows, heads, width],
                    vec![0.; rows * heads * width],
                ))
            })
            .collect();
        reference::run(
            &mut vm,
            "qwen_vision_qkv",
            &[
                Arg::Tensor(projected),
                Arg::Tensor(coords),
                Arg::Tensor(outputs[0]),
                Arg::Tensor(outputs[1]),
                Arg::Tensor(outputs[2]),
            ],
            &HashMap::from([
                ("M".into(), rows as i64),
                ("H".into(), heads as i64),
                ("P".into(), 3),
            ]),
        );
        for row in 0..rows {
            for head in 0..heads {
                for channel in 0..width {
                    let output = (row * heads + head) * width + channel;
                    for kind in 0..3 {
                        let base = ((row * 3 + kind) * heads + head) * width;
                        let expected = if kind == 2 {
                            values[base + channel] as f64
                        } else {
                            let pair = channel % 6;
                            let angle = positions[row][pair / 3] as f64
                                * 10000f64.powf(-((pair % 3) as f64) / 3.);
                            let a = values[base + channel] as f64;
                            let b = values[base
                                + if channel < 6 {
                                    channel + 6
                                } else {
                                    channel - 6
                                }] as f64;
                            if channel < 6 {
                                a * angle.cos() - b * angle.sin()
                            } else {
                                a * angle.cos() + b * angle.sin()
                            }
                        };
                        assert!(
                            (vm.tensors[outputs[kind]].get(output) - expected).abs() < 3e-6,
                            "row={row}, head={head}, channel={channel}, kind={kind}"
                        );
                    }
                }
            }
        }
    }
}
#[test]
fn vision_attention_includes_future_rows_and_keeps_heads_independent() {
    {
        let program = seismic_engine::models::qwen35::program::program().unwrap();
        for nonzero in [false, true] {
            let mut vm = reference::interpreter(&program);
            let (rows, heads, width) = (3usize, 2usize, 4usize);
            let queries = (0..24)
                .map(|i| if nonzero { (i % 7) as f32 / 8. } else { 0. })
                .collect::<Vec<_>>();
            let keys = (0..24).map(|i| (i % 11) as f32 / 9.).collect::<Vec<_>>();
            let values = (0..24).map(|i| i as f32 / 4.).collect::<Vec<_>>();
            let args: Vec<_> = [&queries, &keys, &values]
                .into_iter()
                .map(|data| {
                    vm.add_tensor(TensorData::dense(
                        DType::F32,
                        vec![rows, heads, width],
                        data.iter().map(|&v| v as f64).collect(),
                    ))
                })
                .collect();
            let out = vm.add_tensor(TensorData::dense(
                DType::F32,
                vec![rows, heads, width],
                vec![0.; 24],
            ));
            reference::run(
                &mut vm,
                "qwen_vision_full_attention",
                &[
                    Arg::Tensor(args[0]),
                    Arg::Tensor(args[1]),
                    Arg::Tensor(args[2]),
                    Arg::Tensor(out),
                ],
                &HashMap::from([("M".into(), 3), ("H".into(), 2), ("W".into(), 4)]),
            );
            for row in 0..rows {
                for head in 0..heads {
                    let scores: Vec<f64> = (0..rows)
                        .map(|key| {
                            (0..width)
                                .map(|i| {
                                    queries[(row * heads + head) * width + i] as f64
                                        * keys[(key * heads + head) * width + i] as f64
                                })
                                .sum::<f64>()
                                / 2.
                        })
                        .collect();
                    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let exp: Vec<f64> = scores.iter().map(|score| (score - max).exp()).collect();
                    let total: f64 = exp.iter().sum();
                    for channel in 0..width {
                        let expected: f64 = (0..rows)
                            .map(|key| {
                                exp[key] / total
                                    * values[(key * heads + head) * width + channel] as f64
                            })
                            .sum();
                        assert!(
                            (vm.tensors[out].get((row * heads + head) * width + channel)
                                - expected)
                                .abs()
                                < 2e-6
                        );
                    }
                }
            }
            if !nonzero {
                assert_eq!(vm.tensors[out].get(0), 2.);
            }
        }
    }
}

#[test]
fn tanh_gelu_matches_declared_variant_across_tails_and_zero() {
    {
        let program = seismic_std::program().unwrap();
        let mut vm = reference::interpreter(&program);
        let values: Vec<f64> = (-160..=160)
            .map(|i| i as f64 / 16.)
            .chain([-100., -0.00001, 0.00001, 100.])
            .collect();
        let x = vm.add_tensor(TensorData::dense(
            DType::F32,
            vec![1, values.len()],
            values.clone(),
        ));
        let out = vm.add_tensor(TensorData::dense(
            DType::F32,
            vec![1, values.len()],
            vec![0.; values.len()],
        ));
        reference::run(
            &mut vm,
            "gelu_tanh",
            &[Arg::Tensor(x), Arg::Tensor(out)],
            &HashMap::from([("M".into(), 1), ("N".into(), values.len() as i64)]),
        );
        for (index, &x) in values.iter().enumerate() {
            let expected = 0.5
                * x
                * (1. + ((2. / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh());
            let actual = vm.tensors[out].get(index);
            assert!(
                (actual - expected).abs() < 1e-6,
                "x={x}: {actual} vs {expected}"
            );
        }
    }
}

#[test]
fn full_vision_block_matches_independent_v3_equation_fixture() {
    check_vision_reference(
        "qwen_vision_block",
        include_str!("../../validation/results/fixtures/qwen-vision-block-reference.json"),
    );
}

#[test]
fn full_vision_merger_matches_independent_v3_equation_fixture() {
    check_vision_reference(
        "qwen_vision_merger",
        include_str!("../../validation/results/fixtures/qwen-vision-merger-reference.json"),
    );
}

fn check_vision_reference(entry: &str, fixture: &str) {
    {
        use seismic_lang::types::{Elem, ValueType};
        let reference: serde_json::Value = serde_json::from_str(fixture).unwrap();
        let dimensions: HashMap<String, i64> = reference["dimensions"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), v.as_i64().unwrap()))
            .collect();
        let program = seismic_engine::models::qwen35::program::program().unwrap();
        let mut vm = reference::interpreter(&program);
        let mut args = Vec::new();
        let mut output = None;
        for param in &reference::entry(&program, entry).params {
            let name = &param.name;
            match &param.ty {
                ValueType::Tensor(t) => {
                    let shape = reference::extents(t, &dimensions);
                    let dtype = match t.elem {
                        Elem::Dtype(d) => d,
                        _ => DType::F32,
                    };
                    let values = reference["inputs"]
                        .get(name)
                        .map(|v| {
                            v.as_array()
                                .unwrap()
                                .iter()
                                .map(|v| v.as_f64().unwrap())
                                .collect()
                        })
                        .unwrap_or_else(|| vec![0.; shape.iter().product()]);
                    let id = vm.add_tensor(TensorData::dense(dtype, shape, values));
                    if name == "out" {
                        output = Some(id);
                    }
                    args.push(Arg::Tensor(id));
                }
                ValueType::Scalar(_) => {
                    assert_eq!(name, "epsilon");
                    args.push(Arg::Scalar(1e-6));
                }
                _ => panic!("unexpected vision parameter"),
            }
        }
        reference::run(&mut vm, entry, &args, &dimensions);
        for (i, expected) in reference["output"].as_array().unwrap().iter().enumerate() {
            let actual = vm.tensors[output.unwrap()].get(i);
            assert!(
                (actual - expected.as_f64().unwrap()).abs() < 2e-6,
                "{i}: {actual} vs {expected}"
            );
        }
    }
}

#[test]
fn merger_normalizes_patches_before_grouping_and_publishes_before_projection() {
    {
        let program = seismic_engine::models::qwen35::program::program().unwrap();
        for dtype in [DType::F32, DType::BF16] {
            let round = |x: f32| match dtype {
                DType::BF16 => seismic_lang::numeric::bf16_round(x),
                _ => x,
            };
            let (groups, patches, width) = (2, 4, 3);
            let merged = patches * width;
            // Different offsets/scales make normalization across a group incorrect.
            let source: Vec<f32> = (0..groups * patches)
                .flat_map(|r| [100. * r as f32, 100. * r as f32 + 2., 100. * r as f32 + 7.])
                .map(round)
                .collect();
            let scales = [0.75f32, -1.5, 2.];
            let shifts = [0.125f32, 0.25, -0.5];
            let weights: Vec<f32> = (0..merged * merged)
                .map(|i| round(((i * 7 % 23) as f32 - 11.) / 16.))
                .collect();
            let biases: Vec<f32> = (0..merged).map(|i| round(i as f32 / 32.)).collect();
            let mut vm = reference::interpreter(&program);
            let mut tensor = |shape, values: &[f32]| {
                vm.add_tensor(TensorData::dense(
                    dtype,
                    shape,
                    values.iter().map(|&v| v as f64).collect(),
                ))
            };
            let hidden = tensor(vec![groups * patches, width], &source);
            let nw = tensor(vec![width], &scales);
            let nb = tensor(vec![width], &shifts);
            let uw = tensor(vec![merged, merged], &weights);
            let ub = tensor(vec![merged], &biases);
            let normalized = tensor(vec![groups * patches, width], &vec![0.; source.len()]);
            let up = tensor(vec![groups, merged], &vec![0.; groups * merged]);
            reference::run(
                &mut vm,
                "qwen_vision_merger_up",
                &[
                    Arg::Tensor(hidden),
                    Arg::Tensor(nw),
                    Arg::Tensor(nb),
                    Arg::Tensor(uw),
                    Arg::Tensor(ub),
                    Arg::Tensor(normalized),
                    Arg::Tensor(up),
                    Arg::Scalar(1e-6),
                ],
                &HashMap::from([
                    ("M".into(), groups as i64),
                    ("G".into(), patches as i64),
                    ("H".into(), width as i64),
                ]),
            );
            let expected: Vec<f32> = source
                .chunks_exact(width)
                .flat_map(|row| {
                    let mean = row.iter().sum::<f32>() / width as f32;
                    let variance =
                        row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / width as f32;
                    let inverse = 1. / (variance + 1e-6).sqrt();
                    (0..width)
                        .map(move |i| round((row[i] - mean) * inverse * scales[i] + shifts[i]))
                })
                .collect();
            for (i, &expected) in expected.iter().enumerate() {
                assert!((vm.tensors[normalized].get(i) - expected as f64).abs() < 1e-6);
            }
            for group in 0..groups {
                for output in 0..merged {
                    let sum = (0..merged).fold(0f32, |sum, input| {
                        sum + expected[group * merged + input] * weights[output * merged + input]
                    });
                    let expected = round(sum + biases[output]);
                    assert!(
                        (vm.tensors[up].get(group * merged + output) - expected as f64).abs()
                            < 1e-6
                    );
                }
            }
        }
    }
}

#[test]
fn erf_gelu_matches_python_oracle_at_boundaries_small_values_and_tails() {
    {
        use seismic_lang::program::{compile, SourceFile};
        let reference: Vec<[f64; 3]> = serde_json::from_str(include_str!(
            "../../validation/results/fixtures/erf-gelu-reference.json"
        ))
        .unwrap();
        let mut sources = seismic_std::sources();
        sources.push(SourceFile {
            path: "erf-test.seismic".into(),
            text: "fn erf_test[M, N](x: &tensor[M, N] f32, out: &mut tensor[M, N] f32):\n    parallel for row in 0..M:\n        out[row] = erf_values(to_owned(f32(x[row])))\n".into(),
        });
        let program = compile(&sources).unwrap_or_else(|errors| {
            panic!(
                "{}",
                errors
                    .iter()
                    .map(|e| e.render())
                    .collect::<Vec<_>>()
                    .join("\n")
            )
        });
        let mut values: Vec<f64> = reference.iter().map(|row| row[0]).collect();
        values.extend([-0.0, 0.0, f64::NEG_INFINITY, f64::INFINITY, f64::NAN]);
        let n = values.len();
        let mut vm = reference::interpreter(&program);
        let x = vm.add_tensor(TensorData::dense(DType::F32, vec![1, n], values));
        let erf = vm.add_tensor(TensorData::dense(DType::F32, vec![1, n], vec![0.; n]));
        let gelu = vm.add_tensor(TensorData::dense(DType::F32, vec![1, n], vec![0.; n]));
        let shape = HashMap::from([("M".into(), 1), ("N".into(), n as i64)]);
        reference::run(
            &mut vm,
            "erf_test",
            &[Arg::Tensor(x), Arg::Tensor(erf)],
            &shape,
        );
        reference::run(
            &mut vm,
            "gelu",
            &[Arg::Tensor(x), Arg::Tensor(gelu)],
            &shape,
        );
        for (i, &[x, expected_erf, expected_gelu]) in reference.iter().enumerate() {
            let actual = vm.tensors[erf].get(i) as f32;
            let expected = expected_erf as f32;
            assert!(
                actual.to_bits().abs_diff(expected.to_bits()) <= 2,
                "erf({x}): {actual} vs {expected}"
            );
            let actual = vm.tensors[gelu].get(i);
            assert!(
                (actual - expected_gelu).abs() <= 3e-7 * expected_gelu.abs().max(1.),
                "gelu({x}): {actual} vs {expected_gelu}"
            );
        }
        let special = reference.len();
        assert_eq!(
            (vm.tensors[erf].get(special) as f32).to_bits(),
            (-0.0f32).to_bits()
        );
        assert_eq!(vm.tensors[erf].get(special + 1), 0.);
        assert_eq!(vm.tensors[erf].get(special + 2), -1.);
        assert_eq!(vm.tensors[erf].get(special + 3), 1.);
        assert!(vm.tensors[erf].get(special + 4).is_nan());
    }
}
