use seismic_lang::{
    lower::lower,
    program::{compile, SourceFile},
    Scope,
};
use seismic_runtime::{Candidate, Device};

fn exercise(device: Device, candidate: Candidate) {
    let scoped = "fn evaluate(x: tensor[4] f32, out: tensor[2] f32, enabled: bool):\n  a = load(x)\n  acc = 10.0\n  if enabled:\n    acc += reduce(a,0,sum)\n    for i in owned(a): a[i] = a[i] + 1.0\n    acc += reduce(a,0,sum) * 2.0\n  y = tile[2] f32\n  for i in owned(y):\n    if i == 0: y[i] = acc\n    else: y[i] = reduce(a,0,sum)\n  store(y,out)\n";
    let nested = "fn evaluate(x: tensor[2,3] f32, out: tensor[1] f32):\n  a = load(x)\n  y = tile[1] f32\n  for i in owned(y): y[i] = reduce(reduce(a,1,sum),0,sum)\n  store(y,out)\n";
    let inline_load =
        "fn evaluate(x: tensor[4] f32, out: tensor[2] f32):\n  store(load(x[0:2]),out)\n";
    let repeated_load = "fn evaluate(x: tensor[4] f32, out: tensor[2] f32):\n  a = load(x[0:2])\n  store(a,out)\n  a = load(x[2:4])\n  store(a,out)\n";
    let inline_reduction = "fn evaluate(x: tensor[6] f32, out: tensor[1] f32):\n  y = tile[1] f32\n  for i in owned(y): y[i] = reduce(load(x),0,sum)\n  store(y,out)\n";
    for (text, input_count, expected, scalars) in [
        (scoped, 4, vec![10., 10.], vec![0.]),
        (scoped, 4, vec![48., 14.], vec![1.]),
        (nested, 6, vec![21.], vec![]),
        (inline_load, 4, vec![1., 2.], vec![]),
        (repeated_load, 4, vec![3., 4.], vec![]),
        (inline_reduction, 6, vec![21.], vec![]),
    ] {
        let program = compile(
            &[SourceFile {
                path: "bindings.seismic.portable".into(),
                scope: Scope::Portable,
                text: text.into(),
            }],
            &[],
        )
        .unwrap();
        let lowered = lower(&program, "evaluate", device.backend(), &Default::default()).unwrap();
        let mut kernel = device.compile(&lowered, candidate.clone()).unwrap();
        let input = device
            .buffer_from(
                &(1..=input_count)
                    .flat_map(|v| (v as f32).to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let output = device.buffer(expected.len() * 4).unwrap();
        kernel.execute(&[input, output.clone()], &scalars).unwrap();
        let mut bytes = vec![0; expected.len() * 4];
        output.read(&mut bytes).unwrap();
        let actual: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(actual, expected);
    }
}

#[test]
fn cpu_reduction_expressions() {
    for loads in [
        seismic_realization::LoadStrategy::Materialize,
        seismic_realization::LoadStrategy::BorrowProvenReadOnly,
    ] {
        exercise(Device::cpu(), Candidate::Cpu { loads });
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn metal_reduction_expressions() {
    for loads in [
        seismic_realization::LoadStrategy::Materialize,
        seismic_realization::LoadStrategy::BorrowProvenReadOnly,
    ] {
        exercise(
            Device::metal().unwrap(),
            Candidate::Metal(seismic_metal::execution::Config {
                loads,
                ..Default::default()
            }),
        );
    }
}

#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_reduction_expressions() {
    for loads in [
        seismic_realization::LoadStrategy::Materialize,
        seismic_realization::LoadStrategy::BorrowProvenReadOnly,
    ] {
        exercise(
            Device::cuda(0).unwrap(),
            Candidate::Cuda {
                options: seismic_realization::ScalarOptions {
                    dispatch: seismic_realization::Dispatch::Sequential,
                    loads,
                },
                threads_per_block: 32,
            },
        );
    }
}
