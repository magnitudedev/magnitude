use seismic_lang::{
    lower::Options,
    plan::plan_specialized,
    types::{DType, Elem, Ty},
};
use seismic_runtime::{
    plan::{Bindings, CompiledPlan},
    Buffer, Candidate, Device,
};
use serde_json::Value;
use std::collections::HashMap;
struct Bound {
    buffers: HashMap<String, Buffer>,
    scalars: HashMap<String, f64>,
}
impl Bindings for Bound {
    fn buffer(&self, name: &str, plane: &str) -> Option<&Buffer> {
        assert!(plane.is_empty());
        self.buffers.get(name)
    }
    fn scalar(&self, name: &str) -> Option<f64> {
        self.scalars.get(name).copied()
    }
}
fn values(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}
fn encode(xs: &[f32], dtype: DType) -> Vec<u8> {
    if dtype == DType::I32 {
        xs.iter().flat_map(|x| (*x as i32).to_le_bytes()).collect()
    } else if dtype == DType::F32 {
        xs.iter().flat_map(|x| x.to_le_bytes()).collect()
    } else {
        xs.iter()
            .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }
}
fn exercise(device: Device, candidate: Candidate) {
    let reference: Value = serde_json::from_str(include_str!(
        "../../validation/fixtures/qwen-attention-reference.json"
    ))
    .unwrap();
    let program = seismic_engine::models::qwen35::program::program().unwrap();
    let shapes = reference["shapes"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.clone(), v.as_i64().unwrap()))
        .collect();
    let function = program
        .functions
        .iter()
        .find(|f| f.name == "qwen_attention_step")
        .unwrap();
    let elements = function
        .elem_params
        .iter()
        .map(|p| (p.clone(), Elem::Dtype(DType::BF16)))
        .collect();
    let plan = plan_specialized(&program, "qwen_attention_step", &shapes, &elements).unwrap();
    assert_eq!(plan.steps.len(), 11);
    let mut compiled =
        CompiledPlan::compile_diagnostic(&device, &program, &plan, &Options::default(), candidate).unwrap();
    let mut dtypes = HashMap::new();
    let mut bound = Bound {
        buffers: HashMap::new(),
        scalars: reference["scalars"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), v.as_f64().unwrap()))
            .collect(),
    };
    for (name, ty) in &function.params {
        if let Ty::Tensor(t) = ty {
            let dtype = match &t.elem {
                Elem::Dtype(d) => *d,
                Elem::Param(_) => DType::BF16,
                _ => panic!(),
            };
            let count = t
                .shape
                .iter()
                .map(|s| s.eval(&|p| shapes.get(p).copied()).unwrap() as usize)
                .product::<usize>();
            bound.buffers.insert(
                name.clone(),
                device.buffer(count * dtype.bytes() as usize).unwrap(),
            );
            dtypes.insert(name.clone(), dtype);
        }
    }
    for (name, value) in reference["weights"].as_object().unwrap() {
        bound.buffers[name]
            .write(&encode(&values(value), dtypes[name]))
            .unwrap();
    }
    let mut failures = Vec::new();
    for (step_id, case) in reference["cases"].as_array().unwrap().iter().enumerate() {
        bound
            .scalars
            .insert("destination".into(), case["destination"].as_f64().unwrap());
        for name in ["history_key", "history_value"] {
            bound.buffers[name]
                .write(&encode(
                    &values(&reference[format!("initial_{name}")]),
                    DType::BF16,
                ))
                .unwrap();
        }
        for (name, value) in case["inputs"].as_object().unwrap() {
            bound.buffers[name]
                .write(&encode(&values(value), dtypes[name]))
                .unwrap();
        }
        compiled.execute(&bound).unwrap();
        let mut projected_bytes = vec![0; values(&case["outputs"]["projected"]).len() * 2];
        bound.buffers["projected"]
            .read(&mut projected_bytes)
            .unwrap();
        let mut residual_bytes = vec![0; projected_bytes.len() * 2];
        bound.buffers["out"].read(&mut residual_bytes).unwrap();
        for (i, (p, out)) in projected_bytes
            .chunks_exact(2)
            .zip(residual_bytes.chunks_exact(4))
            .enumerate()
        {
            let projected =
                f32::from_bits(u32::from(u16::from_le_bytes(p.try_into().unwrap())) << 16);
            let hidden = case["inputs"]["hidden"][i].as_f64().unwrap() as f32;
            assert_eq!(
                f32::from_le_bytes(out.try_into().unwrap()),
                hidden + projected
            );
        }

        for (name, value) in case["outputs"].as_object().unwrap() {
            let expected = values(value);
            let dtype = dtypes[name];
            let mut bytes = vec![0; expected.len() * dtype.bytes() as usize];
            bound.buffers[name].read(&mut bytes).unwrap();
            let actual = if dtype == DType::F32 {
                bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect::<Vec<_>>()
            } else {
                bytes
                    .chunks_exact(2)
                    .map(|b| {
                        f32::from_bits(u32::from(u16::from_le_bytes(b.try_into().unwrap())) << 16)
                    })
                    .collect()
            };
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
fn cpu_attention_step() {
    exercise(
        Device::cpu(),
        Candidate::Cpu {
            loads: seismic_realization::LoadStrategy::Materialize,
        },
    );
}
#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn metal_attention_step() {
    exercise(
        Device::metal().unwrap(),
        Candidate::Metal(Default::default()),
    );
}
#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_attention_step() {
    exercise(
        Device::cuda(0).unwrap(),
        Candidate::Cuda {
            options: seismic_realization::ScalarOptions {
                dispatch: seismic_realization::Dispatch::ParallelRoot,
                loads: seismic_realization::LoadStrategy::Materialize,
            },
            threads_per_block: 32,
        },
    );
}
