use seismic_lang::{
    lower::lower,
    program::{compile, SourceFile},
    Scope,
};
use seismic_runtime::{Candidate, Device};
use std::collections::HashMap;
const SOURCE: &str = "
fn truth(a: bool, b: bool, out: tensor[2] f32):
  for row in parallel:
    y = tile[2] f32
    for i in owned(y):
      if i == 0:
        if a and b: y[i] = 1.0
        else: y[i] = 0.0
      else:
        if a or b: y[i] = 1.0
        else: y[i] = 0.0
    store(y,out[row*2:(row+1)*2])
fn eager_and(a: bool, source: tensor[1] f32, which: i32, out: tensor[1] f32):
  for row in parallel:
    y = tile[1] f32
    for i in owned(y):
      if a and source[which] > 0.0: y[i] = 1.0
      else: y[i] = 0.0
    store(y,out[row:row+1])
fn eager_or(a: bool, source: tensor[1] f32, which: i32, out: tensor[1] f32):
  for row in parallel:
    y = tile[1] f32
    for i in owned(y):
      if a or source[which] > 0.0: y[i] = 1.0
      else: y[i] = 0.0
    store(y,out[row:row+1])
";
fn exercise(device: Device, candidate: Candidate) {
    let program = compile(
        &[SourceFile {
            path: "boolean.seismic.portable".into(),
            scope: Scope::Portable,
            text: SOURCE.into(),
        }],
        &[],
    )
    .unwrap();
    let mut kernel = device
        .compile(
            &lower(&program, "truth", device.backend(), &HashMap::new()).unwrap(),
            candidate.clone(),
        )
        .unwrap();
    let out = device.buffer(8).unwrap();
    for a in [false, true] {
        for b in [false, true] {
            kernel
                .execute(std::slice::from_ref(&out), &[f64::from(a), f64::from(b)])
                .unwrap();
            let mut bytes = [0; 8];
            out.read(&mut bytes).unwrap();
            assert_eq!(
                f32::from_le_bytes(bytes[0..4].try_into().unwrap()),
                if a && b { 1. } else { 0. }
            );
            assert_eq!(
                f32::from_le_bytes(bytes[4..8].try_into().unwrap()),
                if a || b { 1. } else { 0. }
            );
        }
    }
    let source = device.buffer_from(&1f32.to_le_bytes()).unwrap();
    for (name, a) in [("eager_and", false), ("eager_or", true)] {
        let mut kernel = device
            .compile(
                &lower(&program, name, device.backend(), &HashMap::new()).unwrap(),
                candidate.clone(),
            )
            .unwrap();
        for which in [-1, 1, 0] {
            let result = kernel.execute(
                &[source.clone(), out.clone()],
                &[f64::from(a), f64::from(which)],
            );
            assert_eq!(result.is_ok(), which == 0, "{name}: {which}: {result:?}");
        }
    }
}
#[test]
fn cpu_boolean_logic() {
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
fn metal_boolean_logic() {
    exercise(
        Device::metal().unwrap(),
        Candidate::Metal(Default::default()),
    );
}
#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_boolean_logic() {
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
