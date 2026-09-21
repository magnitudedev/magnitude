#[path = "support/reference.rs"]
mod reference;
use reference::{allocate, fill, Backend};
use magnitude_engine::models::qwen35::program::program;
use seismic_lang::types::{DType, Elem};
use serde_json::Value;
use std::collections::HashMap;
fn doubles(v: &Value) -> impl Iterator<Item = f64> + '_ {
    v.as_array().unwrap().iter().map(|v| v.as_f64().unwrap())
}
fn close_bf16(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        // BF16 publication: agreement is at the published BF16 value.
        assert_eq!(a, e, "element {i}: {a} != {e}");
    }
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
        let scalars = HashMap::from([(
            "normalize".into(),
            f64::from(case["normalize"].as_bool().unwrap()),
        )]);
        backend.run("route_topk", &shapes, &mut tensors, &scalars);
        assert_eq!(
            reference::values(&tensors["routes"])
                .into_iter()
                .map(|v| v as i64)
                .collect::<Vec<_>>(),
            case["routes"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap())
                .collect::<Vec<_>>()
        );
        close(
            &reference::values(&tensors["scores"]),
            &case["scores"],
            2e-7,
        );
    }
    let shapes = fixture["shape"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(n, v)| (n.clone(), v.as_i64().unwrap()))
        .collect();
    let mut tensors = allocate(
        backend.program(),
        "qwen_routed_suffix",
        &shapes,
        |parameter| {
            if parameter == "SRW" {
                DType::F32
            } else {
                DType::BF16
            }
        },
    );
    for (name, record) in fixture["weights"].as_object().unwrap() {
        fill(tensors.get_mut(name).unwrap(), doubles(&record["values"]));
    }
    fill(
        tensors.get_mut("residual").unwrap(),
        doubles(&fixture["residual"]),
    );
    // Normative expected values: the current reference interpreter over the
    // same inputs (the registry's f32 reduction accumulator governs both
    // sides; the V3 fixture only supplies the inputs).
    let expected: Vec<(bool, Vec<f32>)> = {
        let program = match backend {
            Backend::Interpreter(program, _) => *program,
            Backend::Metal(compiler) => compiler.program(),
        };
        let mut reference = Backend::Interpreter(
            program,
            HashMap::from([("A".into(), Elem::Dtype(DType::BF16))]),
        );
        fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .map(|case| {
                let normalize = case["normalize"].as_bool().unwrap();
                let scalars = HashMap::from([
                    ("eps".into(), 1e-6),
                    ("normalize".into(), f64::from(normalize)),
                ]);
                let mut run = tensors.clone();
                reference.run("qwen_routed_suffix", &shapes, &mut run, &scalars);
                // The residual out is result leaf 14.
                (
                    normalize,
                    reference::values(&run["result14"])
                        .into_iter()
                        .map(|v| v as f32)
                        .collect(),
                )
            })
            .collect()
    };
    let cases = fixture["cases"].as_array().unwrap();
    for (case, (normalize, expected)) in cases.iter().zip(&expected) {
        let scalars = HashMap::from([
            ("eps".into(), 1e-6),
            ("normalize".into(), f64::from(*normalize)),
        ]);
        backend.run("qwen_routed_suffix", &shapes, &mut tensors, &scalars);
        // The backend must agree with the reference bit for bit at the
        // published BF16 value.
        close_bf16(&reference::values(&tensors["result14"]), expected);
    }
}
#[test]
fn reference_routed() {
    let program = program().unwrap();
    exercise(&mut Backend::Interpreter(
        &program,
        HashMap::from([("A".into(), Elem::Dtype(DType::BF16))]),
    ));
}
#[test]
#[ignore = "requires a Metal device"]
fn metal_routed() {
    use seismic_runtime::{
        plan::{PlanCompiler, Settings},
        Device,
    };
    let program = program().unwrap();
    let device = Device::metal().unwrap();
    exercise(&mut Backend::Metal(PlanCompiler::new(
        &device,
        &program,
        Settings::default(),
    )));
}
