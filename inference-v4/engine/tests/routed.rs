#[path = "support/reference.rs"]
mod reference;
use reference::{allocate, fill, Backend, WIDTHS};
use seismic_engine::models::qwen35::program::program;
use seismic_lang::types::DType;
use serde_json::Value;
use std::collections::HashMap;
fn doubles(v: &Value) -> impl Iterator<Item = f64> + '_ {
    v.as_array().unwrap().iter().map(|v| v.as_f64().unwrap())
}
fn close(actual: &[f32], expected: &Value, tolerance: f32) {
    let expected = doubles(expected).map(|v| v as f32).collect::<Vec<_>>();
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
        assert!((a - e).abs() <= tolerance, "element {i}: {a} != {e}");
    }
}
fn exercise(backend: &mut Backend<'_>) {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../validation/results/fixtures/qwen-routed-reference.json"
    ))
    .unwrap();
    for case in fixture["routing"].as_array().unwrap() {
        let shapes = ["M", "E", "K"]
            .map(|n| (n.to_string(), case[n].as_i64().unwrap()))
            .into_iter()
            .collect();
        let mut tensors = allocate(backend.program(), "route_topk", &shapes, |_| DType::F32);
        fill(tensors.get_mut("logits").unwrap(), doubles(&case["logits"]));
        let scalars = HashMap::from([("normalize".into(), f64::from(case["normalize"].as_bool().unwrap()))]);
        backend.run("route_topk", &shapes, &mut tensors, &scalars);
        assert_eq!(
            reference::values(&tensors["routes"]).into_iter().map(|v| v as i64).collect::<Vec<_>>(),
            case["routes"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap())
                .collect::<Vec<_>>()
        );
        close(&reference::values(&tensors["scores"]), &case["scores"], 2e-7);
    }
    let shapes = fixture["shape"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(n, v)| (n.clone(), v.as_i64().unwrap()))
        .collect();
    let mut tensors = allocate(backend.program(), "qwen_routed_suffix", &shapes, |parameter| {
        if parameter == "SRW" { DType::F32 } else { DType::BF16 }
    });
    for (name, record) in fixture["weights"].as_object().unwrap() {
        fill(tensors.get_mut(name).unwrap(), doubles(&record["values"]));
    }
    fill(tensors.get_mut("residual").unwrap(), doubles(&fixture["residual"]));
    for case in fixture["cases"].as_array().unwrap() {
        let scalars = HashMap::from([
            ("eps".into(), 1e-6),
            ("normalize".into(), f64::from(case["normalize"].as_bool().unwrap())),
        ]);
        backend.run("qwen_routed_suffix", &shapes, &mut tensors, &scalars);
        close(&reference::values(&tensors["out"]), &case["out"], 2e-6);
    }
}
#[test]
fn reference_routed() {
    let program = program().unwrap();
    for width in WIDTHS {
        exercise(&mut Backend::Interpreter(&program, width));
    }
}
#[test]
#[ignore = "requires a Metal device"]
fn metal_routed() {
    use seismic_runtime::{plan::{PlanCompiler, Settings}, Device};
    let program = program().unwrap();
    let device = Device::metal().unwrap();
    exercise(&mut Backend::Metal(PlanCompiler::new(&device, &program, Settings::default())));
}
