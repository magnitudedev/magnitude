use seismic_accounting::{workload::DerivationLimits};
use seismic_lang::{
    program::{compile, SourceFile},
    Scope,
};
use seismic_runtime::{
    tuner::{self, Form, Hardware, Input, Outcome, Request},
    Buffer, Device,
};
#[path = "support/automatic_hardware.rs"]
mod automatic_hardware;
fn exercise(device: Device) {
    let p=compile(&[SourceFile{path:"native.seismic.portable".into(),scope:Scope::Portable,text:"fn transform[N](x: tensor[N] f32, out: tensor[N] f32, gain: f32):\n  for row in parallel:\n    t = load(x[row:row+1])\n    y = tile[1] f32\n    for i in owned(y): y[i] = t[i] * gain\n    store(y,out[row:row+1])\n".into()}],&[]).unwrap();
    let l = seismic_lang::lower::lower(
        &p,
        "transform",
        device.backend(),
        &std::collections::HashMap::from([("N".into(), 4)]),
    )
    .unwrap();
    let input = device
        .buffer_from(
            &[9f32, 1., 2., 3., 4., 9.]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap()
        .view(4..20)
        .unwrap();
    let backing = device.buffer_from(&[0xa5; 24]).unwrap();
    let out = backing.view(4..20).unwrap();
    assert!(out.shares_allocation(&backing));
    assert!(!out.shares_allocation(&input));
    assert_eq!(Buffer::reclaimable_bytes([&backing]).unwrap(), 0);
    assert_eq!(
        Buffer::reclaimable_bytes([&backing, &out, &out]).unwrap(),
        24
    );
    assert_eq!(Buffer::reclaimable_bytes([&input]).unwrap(), 24);
    let pin = out.clone();
    assert_eq!(Buffer::reclaimable_bytes([&backing, &out]).unwrap(), 0);
    drop(pin);

    let schema = vec![seismic_lang::abi::ScalarParameter::plain(
        "gain",
        seismic_lang::types::DType::F32,
    )];
    let bindings = [input.clone(), out.clone()];
    let invocation = tuner::workload("native automatic views", &bindings, &schema, &[2.0]).unwrap();
    let facts = device.facts();
    let (form, hardware) = match &facts {
        seismic_runtime::DeviceFacts::Cpu { .. } => {
            (Form::CpuScalar, Hardware::Cpu(automatic_hardware::cpu(&l)))
        }
        seismic_runtime::DeviceFacts::Cuda(_) => {
            (Form::CudaScalar, automatic_hardware::cuda(&l, &facts))
        }
        #[cfg(target_os = "macos")]
        seismic_runtime::DeviceFacts::Metal(_) => (Form::Metal, automatic_hardware::metal(&l)),
    };
    let shapes = std::collections::HashMap::from([("N".into(), 4)]);
    let elements = std::collections::HashMap::new();
    let options = seismic_lang::lower::Options::default();
    let request = Request {
        input: Input::Portable {
            program: &p,
            entry: "transform",
            shapes: &shapes,
            elements: &elements,
            options: &options,
        },
        device: &facts,
        form,
        hardware: &hardware,
        workload: &invocation,
        derivation_limits: DerivationLimits {
            instructions: 100_000,
            operations: 100_000,
        },
    };
    let Outcome::Optimal(selected) = tuner::tune(
        &request,
        seismic_runtime::tuner::Settings { limits: seismic_runtime::tuner::Limits { work: 100_000, ..Default::default() }, ..Default::default() },
    )
    .unwrap() else {
        panic!("native views require completed automatic selection")
    };
    let mut kernel = device.compile_tuned(selected).unwrap();

    // Native allocation facts are derived from retained roots, including views.
    // GPU address alignment is checked separately from CPU mapping alignment.
    let workload = seismic_runtime::tuner::workload(
        "native views",
        &[input.clone(), out.clone(), backing.clone()],
        kernel.scalars(),
        &[2.0],
    )
    .unwrap();
    assert_eq!(workload.allocations.len(), 2);
    assert!(workload
        .allocations
        .iter()
        .all(|a| a.alignment.is_power_of_two()));
    assert_eq!(
        workload.buffers[1].allocation,
        workload.buffers[2].allocation
    );
    assert_eq!(workload.buffers[1].offset, 4);
    assert_eq!(workload.buffers[2].offset, 0);

    let gpu = device.backend() != "cpu";
    drop(device);
    let observation = kernel
        .execute_observed(&[input, out.clone()], &[2.0])
        .unwrap();
    assert!(observation.host_seconds.is_finite() && observation.host_seconds > 0.0);
    assert_eq!(observation.device_seconds.is_some(), gpu);
    if let Some(seconds) = observation.device_seconds {
        assert!(seconds.is_finite() && seconds >= 0.0);
    }
    let mut got = vec![0; 16];
    out.read(&mut got).unwrap();
    assert_eq!(
        got,
        [2f32, 4., 6., 8.]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>()
    );
    let mut all = vec![0; 24];
    backing.read(&mut all).unwrap();
    assert_eq!(&all[..4], &[0xa5; 4]);
    assert_eq!(&all[20..], &[0xa5; 4]);
    assert!(out.read(&mut [0; 17]).is_err());
    assert!(out.write(&[0; 17]).is_err());
    assert!(kernel.execute(&[], &[2.0]).is_err());
}
#[test]
fn cpu_resident_invocation() {
    exercise(Device::cpu());
}
#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_resident_invocation() {
    exercise(Device::cuda(0).unwrap());
}
#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn metal_resident_invocation() {
    exercise(Device::metal().unwrap());
}
