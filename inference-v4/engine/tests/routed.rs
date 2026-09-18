use seismic_engine::{
    execution::{Composition, CompositionSpec},
    models::qwen35::program::program,
};
use seismic_lang::{
    lower::{Options, lower_specialized},
    numeric::bf16_round,
    types::{DType, Elem},
};
use seismic_runtime::{Buffer, Candidate, Device, plan::PlanCompiler};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
fn f32s(v: &Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}
fn upload(device: &Device, values: &[f32], compact: bool) -> Buffer {
    let bytes = if compact {
        values
            .iter()
            .flat_map(|v| ((bf16_round(*v).to_bits() >> 16) as u16).to_le_bytes())
            .collect::<Vec<_>>()
    } else {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    };
    device.buffer_from(&bytes).unwrap()
}
fn read(buffer: &Buffer, len: usize) -> Vec<f32> {
    let mut bytes = vec![0; len * 4];
    buffer.read(&mut bytes).unwrap();
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn close(actual: &[f32], expected: &[f32], tolerance: f32) {
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!((a - e).abs() <= tolerance, "element {i}: {a} != {e}");
    }
}
fn exercise(device: Device, candidate: Candidate) {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../validation/fixtures/qwen-routed-reference.json"
    ))
    .unwrap();
    let program = program().unwrap();
    for case in fixture["routing"].as_array().unwrap() {
        let shapes = ["M", "E", "K"]
            .map(|n| (n.to_string(), case[n].as_i64().unwrap()))
            .into_iter()
            .collect();
        let lowered = lower_specialized(
            &program,
            "route_topk",
            device.backend(),
            &shapes,
            &HashMap::new(),
            &Options::default(),
        )
        .unwrap();
        let mut kernel = device.compile(&lowered, candidate.clone()).unwrap();
        let count = (case["M"].as_u64().unwrap() * case["K"].as_u64().unwrap()) as usize;
        let logits = upload(&device, &f32s(&case["logits"]), false);
        let routes = device.buffer(count * 4).unwrap();
        let scores = device.buffer(count * 4).unwrap();
        let buffers = kernel
            .buffers()
            .iter()
            .map(|b| match b.parameter.as_str() {
                "logits" => logits.clone(),
                "routes" => routes.clone(),
                "scores" => scores.clone(),
                _ => panic!(),
            })
            .collect::<Vec<_>>();
        kernel
            .execute(&buffers, &[f64::from(case["normalize"].as_bool().unwrap())])
            .unwrap();
        let mut bytes = vec![0; count * 4];
        routes.read(&mut bytes).unwrap();
        let actual = bytes
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()) as i64)
            .collect::<Vec<_>>();
        assert_eq!(
            actual,
            case["routes"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap())
                .collect::<Vec<_>>()
        );
        close(&read(&scores, count), &f32s(&case["scores"]), 2e-7);
    }
    let mut compiler = PlanCompiler::diagnostic(&device, &program, Options::default(), candidate);
    let mut tensors = HashMap::new();
    for (name, record) in fixture["weights"].as_object().unwrap() {
        tensors.insert(
            name.clone(),
            upload(&device, &f32s(&record["values"]), name != "shared_router"),
        );
    }
    tensors.insert(
        "residual".into(),
        upload(&device, &f32s(&fixture["residual"]), false),
    );
    let count = f32s(&fixture["residual"]).len();
    let output = device.buffer(count * 4).unwrap();
    tensors.insert("out".into(), output.clone());
    let intermediates = [
        "normalized",
        "logits",
        "routes",
        "scores",
        "gate",
        "up",
        "product",
        "projected",
        "sg",
        "su",
        "sa",
        "sp",
        "shared",
        "coefficient",
    ];
    for case in fixture["cases"].as_array().unwrap() {
        let spec = CompositionSpec {
            entry: "qwen_routed_suffix".into(),
            shapes: fixture["shape"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(n, v)| (n.clone(), v.as_i64().unwrap()))
                .collect(),
            elements: [
                "A", "NW", "RW", "SRW", "EGW", "EUW", "EDW", "SGW", "SUW", "SDW",
            ]
            .map(|n| {
                (
                    n.into(),
                    Elem::Dtype(if n == "SRW" { DType::F32 } else { DType::BF16 }),
                )
            })
            .into_iter()
            .collect(),
            weights: HashMap::new(),
            external: tensors.keys().cloned().collect::<HashSet<_>>(),
            intermediates: intermediates.iter().map(|n| n.to_string()).collect(),
            scalars: HashMap::from([
                ("eps".into(), 1e-6),
                (
                    "normalize".into(),
                    f64::from(case["normalize"].as_bool().unwrap()),
                ),
            ]),
        };
        let mut composition = Composition::compile(&mut compiler, spec).unwrap();
        composition.execute(&tensors, &HashMap::new()).unwrap();
        close(&read(&output, count), &f32s(&case["out"]), 2e-6);
    }
}
#[test]
fn cpu_routed() {
    exercise(
        Device::cpu(),
        Candidate::Cpu {
            loads: seismic_realization::LoadStrategy::BorrowProvenReadOnly,
        },
    );
}
#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn metal_routed() {
    exercise(
        Device::metal().unwrap(),
        Candidate::Metal(Default::default()),
    );
}
#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_routed() {
    exercise(
        Device::cuda(0).unwrap(),
        Candidate::Cuda {
            options: seismic_realization::ScalarOptions {
                dispatch: seismic_realization::Dispatch::ParallelRoot,
                loads: seismic_realization::LoadStrategy::BorrowProvenReadOnly,
            },
            threads_per_block: 32,
        },
    );
}
