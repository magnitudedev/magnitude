#[path = "support/reference.rs"]
mod reference;
use reference::{allocate, fill, Backend, WIDTHS};
use seismic_lang::types::{DType, Ty};
use serde_json::Value;
use std::collections::HashMap;
fn exercise(backend: &mut Backend<'_>) {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../validation/results/fixtures/qwen-rotary-reference.json"
    ))
    .unwrap();
    let shapes: HashMap<String, i64> = fixture["shapes"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.clone(), v.as_i64().unwrap()))
        .collect();
    let mut tensors = allocate(backend.program(), "rotary_prepare", &shapes, |_| DType::BF16);
    for (name, tensor) in tensors.iter_mut() {
        if let Some(values) = fixture["inputs"][name].as_array() {
            fill(tensor, values.iter().map(|v| v.as_f64().unwrap()));
        }
    }
    let scalars = reference::entry(backend.program(), "rotary_prepare")
        .params
        .iter()
        .filter(|p| !matches!(p.ty, Ty::Tensor(_)))
        .map(|p| (p.name.clone(), fixture["scalars"][&p.name].as_f64().unwrap()))
        .collect();
    backend.run("rotary_prepare", &shapes, &mut tensors, &scalars);
    for (name, expected) in fixture["outputs"].as_object().unwrap() {
        let actual = reference::values(&tensors[name]);
        let expected = expected.as_array().unwrap();
        assert_eq!(actual.len(), expected.len());
        let mut maximum = 0f32;
        for (i, (actual, e)) in actual.into_iter().zip(expected).enumerate() {
            let expected = e.as_f64().unwrap() as f32;
            maximum = maximum.max((actual - expected).abs());
            assert!(
                (actual - expected).abs() <= 2e-5 + 0.008 * expected.abs(),
                "{name}[{i}]: {actual} != {expected}"
            );
        }
        eprintln!("{name}: max_abs={maximum:e}");
    }
}
#[test]
fn reference_rotary_prepare() {
    let program = seismic_std::program().unwrap();
    for width in WIDTHS {
        exercise(&mut Backend::Interpreter(&program, width));
    }
}
#[test]
#[ignore = "requires a Metal device"]
fn metal_rotary_prepare() {
    use seismic_runtime::{plan::{PlanCompiler, Settings}, Device};
    let program = seismic_std::program().unwrap();
    let device = Device::metal().unwrap();
    exercise(&mut Backend::Metal(PlanCompiler::new(&device, &program, Settings::default())));
}
