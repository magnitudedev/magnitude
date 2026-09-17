use seismic_lang::{
    lower::lower,
    program::{compile, SourceFile},
    Scope,
};
use seismic_runtime::{Candidate, Device};
use std::collections::HashMap;
fn exercise(device: Device, candidate: Candidate) {
    let source="fn transpose_snapshot(x: tensor[1,2,64] f32, out: tensor[1,64,2] f32):\n  for row in parallel:\n    before = load(x[row])\n    zero = tile[2,64] f32\n    for i,j in owned(zero): zero[i,j]=0.0\n    store(zero,x[row])\n    result = tile[64,2] f32\n    for i,j in owned(result): result[i,j]=before.T[i,j]\n    store(result,out[row])\n";
    let program = compile(
        &[SourceFile {
            path: "transpose.seismic.portable".into(),
            scope: Scope::Portable,
            text: source.into(),
        }],
        &[],
    )
    .unwrap();
    let lowered = lower(
        &program,
        "transpose_snapshot",
        device.backend(),
        &HashMap::new(),
    )
    .unwrap();
    let mut kernel = device.compile(&lowered, candidate).unwrap();
    let bytes = (0..128)
        .flat_map(|i| (i as f32 + 0.5).to_le_bytes())
        .collect::<Vec<_>>();
    let input = device.buffer_from(&bytes).unwrap();
    let out = device.buffer(bytes.len()).unwrap();
    kernel.execute(&[input.clone(), out.clone()], &[]).unwrap();
    let mut actual = vec![0; bytes.len()];
    out.read(&mut actual).unwrap();
    let expected = (0..64)
        .flat_map(|i| [i, i + 64])
        .flat_map(|i| (i as f32 + 0.5).to_le_bytes())
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
    input.read(&mut actual).unwrap();
    assert_eq!(actual, vec![0; bytes.len()]);
}
#[test]
fn cpu_transpose_snapshot() {
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
fn metal_transpose_snapshot() {
    exercise(
        Device::metal().unwrap(),
        Candidate::Metal(Default::default()),
    );
}
#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_transpose_snapshot() {
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
