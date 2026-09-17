use seismic_lang::{
    lower::{lower_specialized, Options},
    types::{DType, Elem, Ty},
};
use seismic_runtime::{Candidate, Device};
use serde_json::Value;
use std::collections::HashMap;
fn exercise(device: Device, candidate: Candidate) {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../validation/fixtures/qwen-rotary-reference.json"
    ))
    .unwrap();
    let program = seismic_std::program().unwrap();
    let shapes = fixture["shapes"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.clone(), v.as_i64().unwrap()))
        .collect();
    let lowered = lower_specialized(
        &program,
        "rotary_prepare",
        device.backend(),
        &shapes,
        &HashMap::from([("A".into(), Elem::Dtype(DType::BF16))]),
        &Options::default(),
    )
    .unwrap();
    let mut kernel = device.compile(&lowered, candidate).unwrap();
    let function = program
        .functions
        .iter()
        .find(|f| f.name == "rotary_prepare")
        .unwrap();
    let mut buffers = vec![];
    let mut names = vec![];
    let mut scalars = vec![];
    for (name, ty) in &function.params {
        match ty {
            Ty::Tensor(t) => {
                let dtype = match t.elem {
                    Elem::Dtype(d) => d,
                    _ => DType::BF16,
                };
                let count = t
                    .shape
                    .iter()
                    .map(|s| s.eval(&|p| shapes.get(p).copied()).unwrap() as usize)
                    .product::<usize>();
                let buffer = device.buffer(count * dtype.bytes() as usize).unwrap();
                if let Some(values) = fixture["inputs"][name].as_array() {
                    let bytes = values
                        .iter()
                        .flat_map(|v| match dtype {
                            DType::F32 => (v.as_f64().unwrap() as f32).to_le_bytes().to_vec(),
                            DType::I32 => (v.as_i64().unwrap() as i32).to_le_bytes().to_vec(),
                            _ => (((v.as_f64().unwrap() as f32).to_bits() >> 16) as u16)
                                .to_le_bytes()
                                .to_vec(),
                        })
                        .collect::<Vec<_>>();
                    buffer.write(&bytes).unwrap();
                }
                names.push(name.clone());
                buffers.push(buffer);
            }
            Ty::Scalar(_) => scalars.push(fixture["scalars"][name].as_f64().unwrap()),
            _ => panic!(),
        }
    }
    kernel.execute(&buffers, &scalars).unwrap();
    for (name, values) in fixture["outputs"].as_object().unwrap() {
        let expected = values.as_array().unwrap();
        let mut bytes = vec![0; expected.len() * 2];
        buffers[names.iter().position(|n| n == name).unwrap()]
            .read(&mut bytes)
            .unwrap();
        let mut maximum = 0f32;
        for (i, (b, e)) in bytes.chunks_exact(2).zip(expected).enumerate() {
            let actual = f32::from_bits(u32::from(u16::from_le_bytes(b.try_into().unwrap())) << 16);
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
fn cpu_rotary_prepare() {
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
fn metal_rotary_prepare() {
    exercise(
        Device::metal().unwrap(),
        Candidate::Metal(Default::default()),
    );
}
#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_rotary_prepare() {
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
