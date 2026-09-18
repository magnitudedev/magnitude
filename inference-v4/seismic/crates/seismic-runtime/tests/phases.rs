use seismic_lang::{
    program::{compile, SourceFile},
    Scope,
};
use seismic_runtime::{Candidate, Device};
use std::collections::HashMap;
fn exercise(device: Device, candidate: Candidate) {
    let source="fn phases[N](x: tensor[N] f32, intermediate: tensor[N] f32, out: tensor[N] f32):\n  for row in parallel:\n    t = load(x[row:row+1])\n    y = tile[1] f32\n    for i in owned(y): y[i] = t[i] + 1.0\n    store(y,intermediate[row:row+1])\n  for row in parallel:\n    t = load(intermediate[N - row - 1:N - row])\n    y = tile[1] f32\n    for i in owned(y): y[i] = t[i] * 2.0\n    store(y,out[row:row+1])\n";
    let program = compile(
        &[SourceFile {
            path: "phases.seismic.portable".into(),
            scope: Scope::Portable,
            text: source.into(),
        }],
        &[],
    )
    .unwrap();
    let lowered = seismic_lang::lower::lower(
        &program,
        "phases",
        device.backend(),
        &HashMap::from([("N".into(), 67)]),
    )
    .unwrap();
    let execution = seismic_runtime::execution::Execution::prepare(
        &lowered, candidate, &device.facts(),
    ).unwrap();
    // Account and compile exactly the same prepared execution across backends.
    let _account = execution.account().unwrap();
    let mut kernel = device.compile_execution(execution).unwrap();
    assert_eq!(
        kernel.phase_count(),
        if device.backend() == "cpu" { 1 } else { 2 }
    );
    let input = (0..67)
        .map(|i| i as f32)
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let x = device.buffer_from(&input).unwrap();
    let intermediate = device.buffer(input.len()).unwrap();
    let out = device.buffer(input.len()).unwrap();
    let observation = kernel
        .execute_observed(&[x, intermediate, out.clone()], &[])
        .unwrap();
    let mut actual = vec![0; input.len()];
    out.read(&mut actual).unwrap();
    let expected = (0..67)
        .rev()
        .map(|i| (i as f32 + 1.0) * 2.0)
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
    assert!(observation.host_seconds > 0.0);
}
#[test]
fn cpu_ordered_phases() {
    exercise(
        Device::cpu(),
        Candidate::Cpu {
            loads: seismic_realization::LoadStrategy::Materialize,
        },
    )
}
#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_ordered_phases() {
    exercise(
        Device::cuda(0).unwrap(),
        Candidate::Cuda {
            options: seismic_realization::ScalarOptions {
                dispatch: seismic_realization::Dispatch::ParallelRoot,
                loads: seismic_realization::LoadStrategy::Materialize,
            },
            threads_per_block: 32,
        },
    )
}
#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn metal_ordered_phases() {
    exercise(
        Device::metal().unwrap(),
        Candidate::Metal(Default::default()),
    )
}
