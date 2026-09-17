use seismic_lang::{
    Scope,
    lower::{Options, lower_specialized},
    program::{SourceFile, compile},
};
use seismic_runtime::{Candidate, Device};
use std::collections::HashMap;
fn exercise(device: Device, candidate: Candidate) {
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
        let lowered = lower_specialized(
            &program,
            entry,
            device.backend(),
            &shapes,
            &HashMap::new(),
            &Options::default(),
        )
        .unwrap();
        let mut kernel = device.compile(&lowered, candidate.clone()).unwrap();
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
fn metal_reduction_assignment() {
    exercise(
        Device::metal().unwrap(),
        Candidate::Metal(Default::default()),
    );
}
#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_reduction_assignment() {
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
