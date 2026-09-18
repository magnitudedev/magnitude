//! Semantic model qualification only. The interpreter is not a native benchmark
//! and supplies no implementation choices to the runtime compiler.
use seismic_lang::{
    interp::{Arg, Interpreter, TensorData},
    program::Program,
    types::{DType, Elem, Ty},
};
use serde_json::Value;
use std::collections::HashMap;

fn tensor(vm: &mut Interpreter<'_>, dtype: DType, shape: Vec<usize>, values: Vec<f64>) -> usize {
    vm.add_tensor(TensorData::dense(dtype, shape, values))
}

#[test]
fn packed_embedding_publishes_bf16_before_residual_cast() {
    let p = seismic_engine::models::qwen35::program::program().unwrap();
    let mut vm = Interpreter::new(&p);
    // Three distinct rows, one affine group each. Values require BF16 rounding.
    let scales = [0x3d81u16, 0x3e03, 0x3e85];
    let biases = [0xbf00u16, 0xbe80, 0x3e00];
    let code = |row: usize, column: usize| ((column + row * 3) % 16) as u8;
    let codes = (0..3)
        .flat_map(|row| (0..32).map(move |i| code(row, 2 * i) | (code(row, 2 * i + 1) << 4)))
        .collect();
    let table = vm.add_tensor(TensorData::Packed {
        repr: seismic_lang::repr::lookup("q4g64").unwrap(),
        shape: vec![3, 64],
        planes: vec![
            codes,
            scales.iter().flat_map(|v| v.to_le_bytes()).collect(),
            biases.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ],
    });
    let tokens = tensor(&mut vm, DType::I32, vec![3], vec![2., 0., 2.]);
    let embedded = tensor(&mut vm, DType::BF16, vec![3, 64], vec![0.; 192]);
    let out = tensor(&mut vm, DType::F32, vec![3, 64], vec![0.; 192]);
    call(
        &mut vm,
        &p,
        "qwen_embedding_rows",
        &[("M", 3), ("V", 3), ("D", 64)],
        &[
            ("table", table),
            ("tokens", tokens),
            ("embedded", embedded),
            ("out", out),
        ],
        &[],
    );
    let mut rounded = false;
    for (row, token) in [2, 0, 2].into_iter().enumerate() {
        for column in 0..64 {
            let scale = f32::from_bits(u32::from(scales[token]) << 16);
            let bias = f32::from_bits(u32::from(biases[token]) << 16);
            let decoded = scale * f32::from(code(token, column)) + bias;
            let expected = seismic_lang::numeric::bf16_round(decoded) as f64;
            rounded |= expected != f64::from(decoded);
            assert_eq!(vm.tensors[embedded].get(row * 64 + column), expected);
            assert_eq!(vm.tensors[out].get(row * 64 + column), expected);
        }
    }
    assert!(
        rounded,
        "fixture must distinguish BF16 publication from an F32 bypass"
    );
}
fn call(
    vm: &mut Interpreter<'_>,
    p: &Program,
    entry: &str,
    dims: &[(&str, i64)],
    bindings: &[(&str, usize)],
    scalars: &[(&str, f64)],
) {
    let shapes: HashMap<_, _> = dims.iter().map(|(k, v)| (k.to_string(), *v)).collect();
    let mut args = Vec::new();
    for (name, ty) in &p.functions.iter().find(|f| f.name == entry).unwrap().params {
        match ty {
            Ty::Tensor(t) => {
                let id = if let Some((_, id)) = bindings.iter().find(|(n, _)| *n == name) {
                    *id
                } else {
                    let shape = t
                        .shape
                        .iter()
                        .map(|d| d.eval(&|n| shapes.get(n).copied()).unwrap() as usize)
                        .collect::<Vec<_>>();
                    let size = shape.iter().product();
                    let dtype = match t.elem {
                        Elem::Dtype(d) => d,
                        _ => DType::BF16,
                    };
                    tensor(vm, dtype, shape, vec![0.; size])
                };
                args.push(Arg::Tensor(id));
            }
            Ty::Scalar(_) => args.push(Arg::Scalar(
                scalars
                    .iter()
                    .find(|(n, _)| *n == name)
                    .unwrap_or_else(|| panic!("missing {entry}.{name}"))
                    .1,
            )),
            _ => panic!("unexpected parameter"),
        }
    }
    vm.run(entry, &args, &shapes)
        .unwrap_or_else(|e| panic!("{entry}: {e}"));
}
fn check_logits(vm: &Interpreter<'_>, id: usize, step: &Value) {
    for (i, v) in step["logits"].as_array().unwrap().iter().enumerate() {
        let expected = v.as_f64().unwrap();
        let actual = vm.tensors[id].get(i);
        assert!(
            (actual - expected).abs() <= 2e-4 + 0.002 * expected.abs(),
            "logit {i}: {actual} != {expected}"
        );
    }
}
#[test]
fn multirow_prefill_and_continuation_match_v3_equations() {
    let p = seismic_engine::models::qwen35::program::program().unwrap();
    let fixture: Value = serde_json::from_str(include_str!(
        "../../validation/results/fixtures/qwen-decoder-reference.json"
    ))
    .unwrap();
    // Whole prompt, split prefill followed by decode, and one-row continuation.
    for fragmented in [false, true] {
        let placement = if fragmented { [5, 1, 7] } else { [0, 1, 2] };
        for chunks in [vec![3], vec![2, 1], vec![1, 2], vec![1, 1, 1]] {
            let mut vm = Interpreter::new(&p);
            let mut weights = HashMap::new();
            for (name, w) in fixture["weights"].as_object().unwrap() {
                let f32_weight = ["convolution", "rate", "time_bias", "query_norm", "key_norm"]
                    .iter()
                    .any(|s| name.ends_with(s));
                let id = tensor(
                    &mut vm,
                    if f32_weight { DType::F32 } else { DType::BF16 },
                    w["shape"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_u64().unwrap() as usize)
                        .collect(),
                    w["values"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_f64().unwrap())
                        .collect(),
                );
                weights.insert(name.clone(), id);
            }
            let mut states = Vec::new();
            for i in 0..4 {
                let (a, b) = if i % 2 == 0 {
                    (
                        tensor(&mut vm, DType::BF16, vec![3, 32], vec![0.; 96]),
                        tensor(&mut vm, DType::F32, vec![4, 4, 4], vec![0.; 64]),
                    )
                } else {
                    (
                        tensor(&mut vm, DType::BF16, vec![8, 2, 16], vec![100.; 256]),
                        tensor(&mut vm, DType::BF16, vec![8, 2, 16], vec![100.; 256]),
                    )
                };
                states.push((a, b));
            }
            let mut position = 0;
            for m in chunks {
                let tokens = fixture["steps"].as_array().unwrap()[position..position + m]
                    .iter()
                    .map(|s| s["token"].as_f64().unwrap())
                    .collect();
                let tokens = tensor(&mut vm, DType::I32, vec![m], tokens);
                let hidden = tensor(&mut vm, DType::F32, vec![m, 8], vec![0.; m * 8]);
                call(
                    &mut vm,
                    &p,
                    "qwen_embedding_rows",
                    &[("M", m as i64), ("V", 32), ("D", 8)],
                    &[
                        ("table", weights["embedding"]),
                        ("tokens", tokens),
                        ("out", hidden),
                    ],
                    &[],
                );
                for i in 0..4 {
                    let w = |name: &str| weights[&format!("b{i}.{name}")];
                    if i % 2 == 0 {
                        let next_window = tensor(&mut vm, DType::BF16, vec![3, 32], vec![0.; 96]);
                        let next_delta = tensor(&mut vm, DType::F32, vec![4, 4, 4], vec![0.; 64]);
                        call(
                            &mut vm,
                            &p,
                            "qwen_recurrent_sequence",
                            &[
                                ("M", m as i64),
                                ("H", 8),
                                ("NK", 2),
                                ("GV", 2),
                                ("W", 4),
                                ("C", 4),
                            ],
                            &[
                                ("hidden", hidden),
                                ("out", hidden),
                                ("input_norm", w("input_norm")),
                                ("qkv_weight", w("qkv")),
                                ("gate_weight", w("gate")),
                                ("alpha_weight", w("alpha")),
                                ("beta_weight", w("beta")),
                                ("convolution", w("convolution")),
                                ("rate", w("rate")),
                                ("time_bias", w("time_bias")),
                                ("recurrent_norm", w("recurrent_norm")),
                                ("output_weight", w("output")),
                                ("window", states[i].0),
                                ("delta", states[i].1),
                                ("next_window", next_window),
                                ("next_delta", next_delta),
                            ],
                            &[
                                ("epsilon", 1e-6),
                                ("preparation_epsilon", 4e-6),
                                ("grouped", 1.),
                            ],
                        );
                        states[i] = (next_window, next_delta);
                    } else {
                        let coords = tensor(
                            &mut vm,
                            DType::I32,
                            vec![m, 4],
                            (position..position + m)
                                .flat_map(|i| [i as f64; 4])
                                .collect(),
                        );
                        // Fragmented history has nonmonotonic physical placement,
                        // unrelated rows in gaps, and empty spans at both ends.
                        let spans: Vec<_> = if fragmented {
                            std::iter::once([0., 0.])
                                .chain(
                                    (0..position)
                                        .map(|i| [placement[i] as f64, (placement[i] + 1) as f64]),
                                )
                                .chain(std::iter::once([8., 8.]))
                                .collect()
                        } else {
                            vec![[0., position as f64]]
                        };
                        let visible = tensor(
                            &mut vm,
                            DType::I32,
                            vec![m, spans.len(), 2],
                            (0..m)
                                .flat_map(|_| spans.iter().flatten().copied())
                                .collect(),
                        );
                        let destinations = tensor(
                            &mut vm,
                            DType::I32,
                            vec![m],
                            (position..position + m)
                                .map(|i| placement[i] as f64)
                                .collect(),
                        );
                        call(
                            &mut vm,
                            &p,
                            "qwen_attention_sequence",
                            &[
                                ("M", m as i64),
                                ("D", 8),
                                ("T", 8),
                                ("R", spans.len() as i64),
                                ("H", 4),
                                ("KV", 2),
                                ("P", 6),
                                ("S", 4),
                                ("SH", 2),
                                ("SW", 1),
                            ],
                            &[
                                ("hidden", hidden),
                                ("out", hidden),
                                ("input_norm", w("input_norm")),
                                ("query_gate_weight", w("query_gate")),
                                ("key_weight", w("key")),
                                ("value_weight", w("value")),
                                ("query_norm", w("query_norm")),
                                ("key_norm", w("key_norm")),
                                ("output_weight", w("output")),
                                ("coordinates", coords),
                                ("visible", visible),
                                ("destinations", destinations),
                                ("history_key", states[i].0),
                                ("history_value", states[i].1),
                            ],
                            &[("base", 1e6), ("epsilon", 1e-6), ("scale", 0.25)],
                        );
                    }
                    call(
                        &mut vm,
                        &p,
                        "qwen_dense_suffix",
                        &[("M", m as i64), ("H", 8), ("F", 12)],
                        &[
                            ("residual", hidden),
                            ("out", hidden),
                            ("norm", w("ff_norm")),
                            ("gate_weight", w("ff_gate")),
                            ("up_weight", w("ff_up")),
                            ("down_weight", w("ff_down")),
                        ],
                        &[("eps", 1e-6)],
                    );
                    for row in 0..m {
                        for (j, v) in fixture["steps"][position + row]["block_outputs"][i]
                            .as_array()
                            .unwrap()
                            .iter()
                            .enumerate()
                        {
                            let expected = v.as_f64().unwrap();
                            let actual = vm.tensors[hidden].get(row * 8 + j);
                            assert!(
                                (actual - expected).abs() < 2e-4 + 0.002 * expected.abs(),
                                "row {} block {i} col {j}: {actual} != {expected}",
                                position + row
                            );
                        }
                    }
                }
                let logits = tensor(&mut vm, DType::F32, vec![1, 32], vec![0.; 32]);
                call(
                    &mut vm,
                    &p,
                    "qwen_readout_rows",
                    &[("M", m as i64), ("V", 32), ("D", 8)],
                    &[
                        ("hidden", hidden),
                        ("norm", weights["output_norm"]),
                        ("weight", weights["embedding"]),
                        ("logits", logits),
                    ],
                    &[("epsilon", 1e-6)],
                );
                position += m;
                check_logits(&vm, logits, &fixture["steps"][position - 1]);
            }
        }
    }
}

#[test]
fn selected_readout_matches_full_projection_with_order_duplicates_and_packed_weights() {
    let p = seismic_engine::models::qwen35::program::program().unwrap();
    for packed in [false, true] {
        let mut vm = Interpreter::new(&p);
        let hidden = tensor(
            &mut vm,
            DType::F32,
            vec![2, 64],
            (0..128)
                .map(|i| ((i * 13 % 29) as f64 - 11.) / 7.)
                .collect(),
        );
        let norm = tensor(
            &mut vm,
            DType::BF16,
            vec![64],
            (0..64).map(|i| 0.75 + (i % 5) as f64 / 8.).collect(),
        );
        let weight = if packed {
            vm.add_tensor(TensorData::Packed {
                repr: seismic_lang::repr::lookup("q4g64").unwrap(),
                shape: vec![4, 64],
                planes: vec![
                    (0..128).map(|i| ((i * 19 + 3) % 256) as u8).collect(),
                    [0x3d81u16, 0x3e03, 0x3e85, 0x3d05]
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect(),
                    [0xbf00u16, 0xbe80, 0x3e00, 0xbe00]
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect(),
                ],
            })
        } else {
            tensor(
                &mut vm,
                DType::BF16,
                vec![4, 64],
                (0..256)
                    .map(|i| {
                        seismic_lang::numeric::bf16_round(((i * 11 % 31) as f32 - 15.) / 9.) as f64
                    })
                    .collect(),
            )
        };
        let full = tensor(&mut vm, DType::F32, vec![1, 4], vec![0.; 4]);
        call(
            &mut vm,
            &p,
            "qwen_readout_rows",
            &[("M", 2), ("V", 4), ("D", 64)],
            &[
                ("hidden", hidden),
                ("norm", norm),
                ("weight", weight),
                ("logits", full),
            ],
            &[("epsilon", 1e-6)],
        );
        for ids in [vec![3, 0, 3, 1], vec![2], vec![0, 1, 2, 3]] {
            let selected = tensor(
                &mut vm,
                DType::I32,
                vec![ids.len()],
                ids.iter().map(|&id| id as f64).collect(),
            );
            let output = tensor(&mut vm, DType::F32, vec![1, ids.len()], vec![0.; ids.len()]);
            call(
                &mut vm,
                &p,
                "qwen_readout_selected",
                &[("M", 2), ("V", 4), ("D", 64), ("S", ids.len() as i64)],
                &[
                    ("hidden", hidden),
                    ("norm", norm),
                    ("weight", weight),
                    ("selected", selected),
                    ("logits", output),
                ],
                &[("epsilon", 1e-6)],
            );
            for (position, id) in ids.into_iter().enumerate() {
                assert_eq!(
                    vm.tensors[output].get(position),
                    vm.tensors[full].get(id),
                    "packed={packed}, position={position}"
                );
            }
        }
    }
}
