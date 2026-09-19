use seismic_lang::{
    program::{compile, SourceFile},
    Scope,
};
use seismic_runtime::Device;
#[path = "support/automatic_hardware.rs"]
mod automatic_hardware;
fn exercise(device: Device) {
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
    let invocation = seismic_runtime::tuner::Input::Portable {
        program: &program,
        entry: "transpose_snapshot",
        shapes: &Default::default(),
        elements: &Default::default(),
        options: &Default::default(),
    };
    let bytes = (0..128)
        .flat_map(|i| (i as f32 + 0.5).to_le_bytes())
        .collect::<Vec<_>>();
    let input = device.buffer_from(&bytes).unwrap();
    let out = device.buffer(bytes.len()).unwrap();
    let mut kernel =
        automatic_hardware::compile(&device, invocation, &[input.clone(), out.clone()], &[])
            .unwrap();
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
    exercise(Device::cpu());
}
#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn metal_transpose_snapshot() {
    exercise(Device::metal().unwrap());
}
#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_transpose_snapshot() {
    exercise(Device::cuda(0).unwrap());
}
