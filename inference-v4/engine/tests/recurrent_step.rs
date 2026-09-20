#[path = "support/reference.rs"]
mod reference;
use reference::{allocate, fill, Backend, TensorData, WIDTHS};
use seismic_lang::types::DType;
use serde_json::Value;
use std::collections::HashMap;
fn doubles(value: &Value) -> impl Iterator<Item = f64> + '_ {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
}
fn exercise(backend: &mut Backend<'_>) {
    let reference: Value = serde_json::from_str(include_str!(
        "../../validation/results/fixtures/qwen-recurrent-reference.json"
    ))
    .unwrap();
    let shapes: HashMap<String, i64> = reference["shapes"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.clone(), v.as_i64().unwrap()))
        .collect();
    let mut tensors = allocate(backend.program(), "qwen_recurrent_step", &shapes, |_| {
        DType::BF16
    });
    for (name, value) in reference["weights"].as_object().unwrap() {
        fill(tensors.get_mut(name).unwrap(), doubles(value));
    }
    for case in reference["cases"].as_array().unwrap() {
        let scalars = HashMap::from([
            ("epsilon".into(), 1e-6),
            ("preparation_epsilon".into(), 4e-6),
            ("grouped".into(), f64::from(case["mapping"] == "grouped")),
        ]);
        for name in ["window", "delta"] {
            fill(
                tensors.get_mut(name).unwrap(),
                doubles(&reference[format!("initial_{name}")]),
            );
        }
        for (step_id, step) in case["steps"].as_array().unwrap().iter().enumerate() {
            fill(tensors.get_mut("hidden").unwrap(), doubles(&step["hidden"]));
            let before = ["window", "delta"].map(|name| reference::values(&tensors[name]));
            backend.run("qwen_recurrent_step", &shapes, &mut tensors, &scalars);
            for (name, before) in ["window", "delta"].into_iter().zip(before) {
                assert_eq!(
                    reference::values(&tensors[name]),
                    before,
                    "accepted {name} must remain unchanged"
                );
            }
            for (name, value) in step.as_object().unwrap() {
                if name == "hidden" {
                    continue;
                }
                let expected = doubles(value).map(|v| v as f32).collect::<Vec<_>>();
                let published = if name == "window" || name == "delta" {
                    format!("next_{name}")
                } else {
                    name.clone()
                };
                let TensorData::Dense { dtype, .. } = &tensors[&published] else {
                    panic!()
                };
                let actual = reference::values(&tensors[&published]);
                assert_eq!(actual.len(), expected.len());
                let mut maximum = 0f32;
                for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
                    maximum = maximum.max((a - e).abs());
                    let tolerance = if *dtype == DType::F32 {
                        3e-6 + 3e-5 * e.abs()
                    } else {
                        2e-5 + 0.008 * e.abs()
                    };
                    assert!(
                        (a - e).abs() <= tolerance,
                        "{} step {step_id} {name}[{i}] {a} != {e} tolerance {tolerance}",
                        case["mapping"]
                    );
                }
                eprintln!(
                    "{} step {step_id} {name}: max_abs={maximum:e}",
                    case["mapping"]
                );
            }
            // Accept the step: the published successor becomes the recurrent state.
            for name in ["window", "delta"] {
                let next = tensors[&format!("next_{name}")].clone();
                tensors.insert(name.into(), next);
            }
        }
    }
}
#[test]
fn reference_recurrent_step() {
    let program = seismic_engine::models::qwen35::program::program().unwrap();
    for width in WIDTHS {
        exercise(&mut Backend::Interpreter(&program, width));
    }
}
#[test]
#[ignore = "requires a Metal device"]
fn metal_recurrent_step() {
    use seismic_runtime::{
        plan::{PlanCompiler, Settings},
        Device,
    };
    let program = seismic_engine::models::qwen35::program::program().unwrap();
    let device = Device::metal().unwrap();
    exercise(&mut Backend::Metal(PlanCompiler::new(
        &device,
        &program,
        Settings::default(),
    )));
}
