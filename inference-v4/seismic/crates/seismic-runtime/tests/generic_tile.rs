use seismic_lang::{
    Scope,
    interp::{Arg, Interpreter, TensorData},
    lower::{Options, lower_specialized},
    numeric::{bf16_round, f16_bits, f16_round},
    program::{SourceFile, compile},
    types::{DType, Elem},
};
use seismic_runtime::{Candidate, Device};
use std::collections::HashMap;
fn exercise(device: Device, candidate: Candidate) {
    let text = "fn local[N](x: tensor[N] ACTIVATION, out: tensor[N] f32):\n  a = load(x)\n  compact = tile[N] ACTIVATION\n  for i in owned(compact): compact[i] = f32(a[i]) * 1.003\n  y = tile[N] f32\n  for i in owned(y): y[i] = f32(compact[i]) * 1031.0\n  store(y, out)\nfn wrapper[N](x: tensor[N] INPUT, out: tensor[N] f32):\n  local(x,out)\n";
    let program = compile(
        &[SourceFile {
            path: "generic.seismic.portable".into(),
            text: text.into(),
            scope: Scope::Portable,
        }],
        &[],
    )
    .unwrap();
    let shapes = HashMap::from([("N".into(), 4)]);
    for dtype in [DType::BF16, DType::F16] {
        let round = if dtype == DType::BF16 {
            bf16_round
        } else {
            f16_round
        };
        let input = [1.0f32, -1.0, 0.125, 3.5].map(round);
        let expected = input.map(|x| round(x * 1.003) * 1031.0);
        for entry in ["local", "wrapper"] {
            let mut interpreter = Interpreter::new(&program);
            let x = interpreter.add_tensor(TensorData::dense(
                dtype,
                vec![4],
                input.iter().map(|v| *v as f64).collect(),
            ));
            let o = interpreter.add_tensor(TensorData::dense(DType::F32, vec![4], vec![0.; 4]));
            interpreter
                .run(entry, &[Arg::Tensor(x), Arg::Tensor(o)], &shapes)
                .unwrap();
            for (i, e) in expected.iter().enumerate() {
                assert_eq!(interpreter.tensors[o].get(i) as f32, *e);
            }
        }
        let lowered = lower_specialized(
            &program,
            "local",
            device.backend(),
            &shapes,
            &HashMap::from([("ACTIVATION".into(), Elem::Dtype(dtype))]),
            &Options::default(),
        )
        .unwrap();
        let mut kernel = device.compile(&lowered, candidate.clone()).unwrap();
        let inputbytes = input
            .iter()
            .flat_map(|v| {
                if dtype == DType::BF16 {
                    ((v.to_bits() >> 16) as u16).to_le_bytes()
                } else {
                    f16_bits(*v).to_le_bytes()
                }
            })
            .collect::<Vec<_>>();
        let x = device.buffer_from(&inputbytes).unwrap();
        let out = device.buffer(16).unwrap();
        kernel.execute(&[x, out.clone()], &[]).unwrap();
        let mut bytes = [0; 16];
        out.read(&mut bytes).unwrap();
        for (b, e) in bytes.chunks_exact(4).zip(expected) {
            assert_eq!(f32::from_le_bytes(b.try_into().unwrap()), e);
        }
    }
    let invalid = lower_specialized(
        &program,
        "local",
        device.backend(),
        &shapes,
        &HashMap::from([("ACTIVATION".into(), Elem::Repr("q4g32".into()))]),
        &Options::default(),
    );
    assert!(invalid.unwrap_err().contains("dense dtype"));
}
#[test]
fn cpu_generic_tile() {
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
fn metal_generic_tile() {
    exercise(
        Device::metal().unwrap(),
        Candidate::Metal(Default::default()),
    );
}
#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_generic_tile() {
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
