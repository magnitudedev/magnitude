use seismic_engine::state::{ComponentSpec, StateStore};
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
use std::rc::Rc;
struct Bound {
    buffers: HashMap<String, Buffer>,
    grouped: bool,
}
impl Bindings for Bound {
    fn buffer(&self, name: &str, plane: &str) -> Option<&Buffer> {
        assert!(plane.is_empty());
        self.buffers.get(name)
    }
    fn scalar(&self, name: &str) -> Option<f64> {
        match name {
            "epsilon" => Some(1e-6),
            "preparation_epsilon" => Some(4e-6),
            "grouped" => Some(f64::from(self.grouped)),
            _ => None,
        }
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
    if dtype == DType::F32 {
        xs.iter().flat_map(|x| x.to_le_bytes()).collect()
    } else {
        xs.iter()
            .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }
}
fn exercise(device: Device, candidate: Candidate) {
    let device = Rc::new(device);
    let reference: Value = serde_json::from_str(include_str!(
        "../../validation/fixtures/qwen-recurrent-reference.json"
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
        .find(|f| f.name == "qwen_recurrent_step")
        .unwrap();
    let elements = function
        .elem_params
        .iter()
        .map(|p| (p.clone(), Elem::Dtype(DType::BF16)))
        .collect();
    let plan = plan_specialized(&program, "qwen_recurrent_step", &shapes, &elements).unwrap();
    assert_eq!(plan.steps.len(), 12);
    let mut compiled =
        CompiledPlan::compile_diagnostic(&device, &program, &plan, &Options::default(), candidate).unwrap();
    let mut dtypes = HashMap::new();
    let mut bound = Bound {
        buffers: HashMap::new(),
        grouped: true,
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
    let components = ["window", "delta"]
        .iter()
        .map(|name| {
            let (_, Ty::Tensor(t)) = function.params.iter().find(|(n, _)| n == name).unwrap()
            else {
                panic!()
            };
            ComponentSpec {
                shape: t
                    .shape
                    .iter()
                    .map(|s| s.eval(&|p| shapes.get(p).copied()).unwrap() as usize)
                    .collect(),
                dtype: dtypes[*name],
            }
        })
        .collect();
    let store = StateStore::new(device.clone(), 3, 3, vec![], components).unwrap();
    for case in reference["cases"].as_array().unwrap() {
        let mut state = store.create().unwrap();
        for (i, name) in ["window", "delta"].iter().enumerate() {
            bound
                .buffers
                .insert((*name).into(), state.values()[i].clone());
        }
        bound.grouped = case["mapping"] == "grouped";
        bound.buffers["window"]
            .write(&encode(&values(&reference["initial_window"]), DType::BF16))
            .unwrap();
        bound.buffers["delta"]
            .write(&encode(&values(&reference["initial_delta"]), DType::F32))
            .unwrap();
        for (step_id, step) in case["steps"].as_array().unwrap().iter().enumerate() {
            bound.buffers["hidden"]
                .write(&encode(&values(&step["hidden"]), DType::F32))
                .unwrap();
            let before = ["window", "delta"].map(|name| {
                let count = values(&reference[format!("initial_{name}")]).len();
                let mut bytes = vec![0; count * dtypes[name].bytes() as usize];
                bound.buffers[name].read(&mut bytes).unwrap();
                bytes
            });
            let mut advance = state.begin(1).unwrap();
            advance
                .execute(|transition| {
                    for (i, name) in ["window", "delta"].iter().enumerate() {
                        bound
                            .buffers
                            .insert((*name).into(), transition.previous[i].clone());
                        bound
                            .buffers
                            .insert(format!("next_{name}"), transition.following[i].clone());
                    }
                    compiled.execute(&bound)
                })
                .unwrap();
            for (name, before) in ["window", "delta"].into_iter().zip(before) {
                let mut after = vec![0; before.len()];
                bound.buffers[name].read(&mut after).unwrap();
                assert_eq!(after, before, "accepted {name} must remain unchanged");
            }
            for (name, value) in step.as_object().unwrap() {
                if name == "hidden" {
                    continue;
                }
                let expected = values(value);
                let dtype = dtypes[name];
                let mut bytes = vec![0; expected.len() * dtype.bytes() as usize];
                let buffer_name = if name == "window" || name == "delta" {
                    format!("next_{name}")
                } else {
                    name.clone()
                };
                bound.buffers[&buffer_name].read(&mut bytes).unwrap();
                let actual = if dtype == DType::F32 {
                    bytes
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                        .collect::<Vec<_>>()
                } else {
                    bytes
                        .chunks_exact(2)
                        .map(|b| {
                            f32::from_bits(
                                u32::from(u16::from_le_bytes(b.try_into().unwrap())) << 16,
                            )
                        })
                        .collect()
                };
                let mut maximum = 0f32;
                for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
                    maximum = maximum.max((a - e).abs());
                    let tolerance = if dtype == DType::F32 {
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
            advance.commit().unwrap();
            assert_eq!(state.position(), step_id + 1);
            for (i, name) in ["window", "delta"].iter().enumerate() {
                bound
                    .buffers
                    .insert((*name).into(), state.values()[i].clone());
            }
        }
    }
}
#[test]
fn cpu_recurrent_step() {
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
fn metal_recurrent_step() {
    exercise(
        Device::metal().unwrap(),
        Candidate::Metal(Default::default()),
    );
}
#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_recurrent_step() {
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
