use seismic_lang::{
    program::{compile, SourceFile},
    Scope,
};
use seismic_runtime::{Buffer, Candidate, Device};
fn exercise(device: Device, candidate: Candidate) {
    let p=compile(&[SourceFile{path:"native.seismic.portable".into(),scope:Scope::Portable,text:"fn transform[N](x: tensor[N] f32, out: tensor[N] f32, gain: f32):\n  for row in parallel:\n    t = load(x[row:row+1])\n    y = tile[1] f32\n    for i in owned(y): y[i] = t[i] * gain\n    store(y,out[row:row+1])\n".into()}],&[]).unwrap();
    let l = seismic_lang::lower::lower(
        &p,
        "transform",
        device.backend(),
        &std::collections::HashMap::from([("N".into(), 4)]),
    )
    .unwrap();
    let mut kernel = device.compile(&l, candidate).unwrap();
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
    assert_eq!(Buffer::reclaimable_bytes([&backing, &out, &out]).unwrap(), 24);
    assert_eq!(Buffer::reclaimable_bytes([&input]).unwrap(), 24);
    let pin = out.clone();
    assert_eq!(Buffer::reclaimable_bytes([&backing, &out]).unwrap(), 0);
    drop(pin);

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
    exercise(
        Device::cpu(),
        Candidate::Cpu {
            loads: seismic_realization::LoadStrategy::Materialize,
        },
    );
}
#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_resident_invocation() {
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
#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn metal_resident_invocation() {
    exercise(
        Device::metal().unwrap(),
        Candidate::Metal(Default::default()),
    );
}
