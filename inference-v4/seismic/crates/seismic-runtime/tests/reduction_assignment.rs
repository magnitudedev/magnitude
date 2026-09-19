use seismic_lang::{
    lower::Options,
    program::{compile, SourceFile},
    Scope,
};
use seismic_runtime::Device;
#[path = "support/automatic_hardware.rs"]
mod automatic_hardware;
use std::collections::HashMap;
fn exercise(device: Device) {
    let src = "fn reductions[M,N](x: tensor[M,N] f32, out: tensor[M] f32, enable: bool):\n  a = load(x)\n  result = tile[M] f32\n  for i in owned(result): result[i] = 7.0\n  if enable: result = reduce(a,1,sum)\n  store(result,out)\nfn scalar[N](x: tensor[N] f32, out: tensor[1] f32, enable: bool):\n  a = load(x)\n  result = 7.0\n  winner = reduce(a,0,argmax)\n  if enable:\n    result = reduce(a,0,sum)\n  y = tile[1] f32\n  for i in owned(y): y[i] = result + f32(winner)\n  store(y,out)\n";
    let program = compile(
        &[SourceFile {
            path: "reassign.seismic.portable".into(),
            scope: Scope::Portable,
            text: src.into(),
        }],
        &[],
    )
    .unwrap();
    let values = (0..96).map(|i| (i % 32) as f32).collect::<Vec<_>>();
    for entry in ["reductions", "scalar"] {
        let shapes = HashMap::from([("M".into(), 3), ("N".into(), 32)]);
        let invocation = seismic_runtime::tuner::Input::Portable {
            program: &program,
            entry,
            shapes: &shapes,
            elements: &HashMap::new(),
            options: &Options::default(),
        };
        let x = device
            .buffer_from(
                &values
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let count = if entry == "scalar" { 1 } else { 3 };
        let out = device.buffer(count * 4).unwrap();
        for enable in [0., 1., 0.] {
            let mut kernel = automatic_hardware::compile(
                &device,
                invocation,
                &[x.clone(), out.clone()],
                &[enable],
            )
            .unwrap();
            kernel
                .execute(&[x.clone(), out.clone()], &[enable])
                .unwrap();
            let mut b = vec![0; count * 4];
            out.read(&mut b).unwrap();
            for v in b.chunks_exact(4) {
                assert_eq!(
                    f32::from_le_bytes(v.try_into().unwrap()),
                    (if enable == 0. { 7. } else { 496. })
                        + if entry == "scalar" { 31. } else { 0. }
                );
            }
        }
    }
}
#[test]
fn cpu_reduction_assignment() {
    exercise(Device::cpu());
}
#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn metal_reduction_assignment() {
    exercise(Device::metal().unwrap());
}
#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_reduction_assignment() {
    exercise(Device::cuda(0).unwrap());
}
