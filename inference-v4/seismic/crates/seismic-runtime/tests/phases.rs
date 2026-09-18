use seismic_lang::{
    program::{compile, SourceFile},
    Scope,
};
use seismic_accounting::{selection::Budget, workload::DerivationLimits};
use seismic_runtime::{Device, tuner::{self, Form, Hardware, Input, Outcome, Request}};
#[path = "support/automatic_hardware.rs"]
mod automatic_hardware;
use std::collections::HashMap;
fn exercise(device: Device) {
    let source = "fn phases[N](x: tensor[N] f32, intermediate: tensor[N] f32, out: tensor[N] f32):\n  for row in parallel:\n    t = load(x[row:row+1])\n    y = tile[1] f32\n    for i in owned(y): y[i] = t[i] + 1.0\n    store(y,intermediate[row:row+1])\n  for row in parallel:\n    t = load(intermediate[N - row - 1:N - row])\n    y = tile[1] f32\n    for i in owned(y): y[i] = t[i] * 2.0\n    store(y,out[row:row+1])\n";
    exercise_case(&device, source);
    // The same dependency now crosses an ordinary serial scalar and dense tile
    // declaration. Their private publication storage must stay outside the
    // caller ABI and survive both ordered GPU launches.
    let retained = source.replacen("  for row in parallel:",
        "  bias = 1.0\n  factors = tile[1] f32\n  for i in owned(factors): factors[i] = 2.0\n  for row in parallel:", 1)
        .replace("t[i] + 1.0", "t[i] + bias")
        .replace("t[i] * 2.0", "t[i] * factors[0]");
    exercise_case(&device, &retained);
}
fn exercise_case(device: &Device, source: &str) {
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
    let input = (0..67)
        .map(|i| i as f32)
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let x = device.buffer_from(&input).unwrap();
    let intermediate = device.buffer(input.len()).unwrap();
    let out = device.buffer(input.len()).unwrap();
    let facts = device.facts();
    let (form, hardware) = match &facts {
        seismic_runtime::DeviceFacts::Cpu { .. } => (Form::CpuScalar, Hardware::Cpu(automatic_hardware::cpu(&lowered))),
        seismic_runtime::DeviceFacts::Cuda(_) => (Form::CudaScalar, automatic_hardware::cuda(&lowered, &facts)),
        #[cfg(target_os = "macos")]
        seismic_runtime::DeviceFacts::Metal(_) => (Form::Metal, automatic_hardware::metal(&lowered)),
    };
    let bindings = [x, intermediate, out.clone()];
    let workload = tuner::workload("ordered phases with a partial final group", &bindings, &[], &[]).unwrap();
    let shapes = HashMap::from([("N".into(), 67)]);
    let elements = HashMap::new();
    let options = seismic_lang::lower::Options::default();
    let request = Request {
        input: Input::Portable { program: &program, entry: "phases", shapes: &shapes, elements: &elements, options: &options },
        device: &facts, form, hardware: &hardware, workload: &workload,
        derivation_limits: DerivationLimits { instructions: 1_000_000, operations: 1_000_000 },
    };
    let Outcome::Optimal(tuned) = tuner::tune(&request, Budget { nodes: 20_000, schedule_assignments: 1_000_000 }).unwrap() else {
        panic!("ordered phases require a completed automatic selection");
    };
    let objective = tuned.objective().clone();
    let selected_phases = match tuned.execution() {
        seismic_runtime::execution::Execution::Cpu(_) => 1,
        seismic_runtime::execution::Execution::Cuda(phases) => phases.len(),
        #[cfg(target_os = "macos")]
        seismic_runtime::execution::Execution::Metal(execution) => execution.phases().len(),
    };
    let mut kernel = device.compile_tuned(tuned).unwrap();
    assert_eq!(kernel.buffers().len(), bindings.len(), "phase publications must remain private");
    assert_eq!(kernel.phase_count(), selected_phases);
    assert_eq!(kernel.tuning().objective(), &objective);
    let observation = kernel
        .execute_observed(&bindings, &[])
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
fn cpu_ordered_phases() { exercise(Device::cpu()) }
#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_ordered_phases() { exercise(Device::cuda(0).unwrap()) }
#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn metal_ordered_phases() { exercise(Device::metal().unwrap()) }
