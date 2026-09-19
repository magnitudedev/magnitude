#[path = "support/reference.rs"]
mod reference;
use reference::{allocate, fill, Backend, WIDTHS};
use seismic_lang::types::DType;
use serde_json::Value;
use std::collections::HashMap;
fn values(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}
fn doubles(value: &Value) -> impl Iterator<Item = f64> + '_ {
    value.as_array().unwrap().iter().map(|x| x.as_f64().unwrap())
}
fn exercise(backend: &mut Backend<'_>) {
    let reference: Value = serde_json::from_str(include_str!(
        "../../validation/results/fixtures/qwen-attention-reference.json"
    ))
    .unwrap();
    let shapes: HashMap<String, i64> = reference["shapes"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.clone(), v.as_i64().unwrap()))
        .collect();
    let mut scalars: HashMap<String, f64> = reference["scalars"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.clone(), v.as_f64().unwrap()))
        .collect();
    let mut tensors = allocate(backend.program(), "qwen_attention_step", &shapes, |_| DType::BF16);
    for (name, value) in reference["weights"].as_object().unwrap() {
        fill(tensors.get_mut(name).unwrap(), doubles(value));
    }
    let mut failures = Vec::new();
    for (step_id, case) in reference["cases"].as_array().unwrap().iter().enumerate() {
        scalars.insert("destination".into(), case["destination"].as_f64().unwrap());
        for name in ["history_key", "history_value"] {
            fill(tensors.get_mut(name).unwrap(), doubles(&reference[format!("initial_{name}")]));
        }
        for (name, value) in case["inputs"].as_object().unwrap() {
            fill(tensors.get_mut(name).unwrap(), doubles(value));
        }
        backend.run("qwen_attention_step", &shapes, &mut tensors, &scalars);
        let projected = reference::values(&tensors["projected"]);
        let residual = reference::values(&tensors["out"]);
        for (i, (projected, out)) in projected.iter().zip(&residual).enumerate() {
            let hidden = case["inputs"]["hidden"][i].as_f64().unwrap() as f32;
            assert_eq!(*out, hidden + projected);
        }

        for (name, value) in case["outputs"].as_object().unwrap() {
            let expected = values(value);
            let reference::TensorData::Dense { dtype, .. } = &tensors[name] else { panic!() };
            let dtype = *dtype;
            let actual = reference::values(&tensors[name]);
            assert_eq!(actual.len(), expected.len());
            let mut maximum = 0f32;
            for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
                maximum = maximum.max((a - e).abs());
                // The residual is F32, but its projected operand has already
                // passed through compact publications. Check that publication's
                // error budget, plus a separate exact residual identity below.
                let tolerance = if name == "out" {
                    let projected = case["outputs"]["projected"][i].as_f64().unwrap() as f32;
                    2e-5 + 0.008 * projected.abs()
                } else if name.starts_with("history_") {
                    0.0
                } else if dtype == DType::F32 {
                    3e-6 + 3e-5 * e.abs()
                } else {
                    2e-5 + 0.008 * e.abs()
                };
                if (a - e).abs() > tolerance {
                    failures.push(format!(
                        "attention step {step_id} {name}[{i}] {a} != {e} tolerance {tolerance}"
                    ));
                }
            }
            eprintln!("attention step {step_id} {name}: max_abs={maximum:e}");
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
#[test]
fn reference_attention_step() {
    let program = seismic_engine::models::qwen35::program::program().unwrap();
    for width in WIDTHS {
        exercise(&mut Backend::Interpreter(&program, width));
    }
}
#[test]
#[ignore = "requires a Metal device"]
fn metal_attention_step() {
    use seismic_runtime::{plan::{PlanCompiler, Settings}, Device};
    let program = seismic_engine::models::qwen35::program::program().unwrap();
    let device = Device::metal().unwrap();
    exercise(&mut Backend::Metal(PlanCompiler::new(&device, &program, Settings::default())));
}
