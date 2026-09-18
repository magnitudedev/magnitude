use seismic_lang::{
    interp::{TensorData, Rng},
    lower::{lower_specialized, Options},
    program::{compile, SourceFile},
    repr,
    types::Elem,
    Scope,
};
use seismic_runtime::{Candidate, Device};
use std::collections::HashMap;
fn exercise(device: Device, candidate: Candidate) {
    let mut sources = seismic_std::sources();
    sources.push(SourceFile{path:"whole_matrix.seismic.portable".into(),scope:Scope::Portable,text:"fn whole[M,N,K](x: tensor[M,K] f32, weight: tensor[N,K] W, out: tensor[M,N] f32):\n  a = load(x)\n  b = load(weight)\n  c = tile[M,N] f32\n  for i,j in owned(c): c[i,j] = 0.0\n  matmul(a,b,c)\n  store(c,out)\n".into()});
    let program = compile(&sources, &["cpu".into(), "cuda".into(), "metal".into()]).unwrap();
    for rep in ["q4g32", "q4k", "q5k", "q6k", "q8g32s", "iq4g32"] {
        let rep = repr::lookup(rep).unwrap();
        for m in [1, 3] {
            let (n, k) = (5usize, 256usize);
            let weight = TensorData::random_packed(&mut Rng(0x123456789), rep, vec![n, k]);
            let x = (0..m * k)
                .map(|i| ((i % 19) as f32 - 9.) / 32.)
                .collect::<Vec<_>>();
            let expected = (0..m * n)
                .map(|i| {
                    (0..k).fold(0f32, |a, j| {
                        x[(i / n) * k + j].mul_add(weight.get((i % n) * k + j) as f32, a)
                    })
                })
                .collect::<Vec<_>>();
            let lowered = lower_specialized(
                &program,
                "whole",
                device.backend(),
                &HashMap::from([
                    ("M".into(), m as i64),
                    ("N".into(), n as i64),
                    ("K".into(), k as i64),
                ]),
                &HashMap::from([("W".into(), Elem::Repr(rep.name.into()))]),
                &Options::default(),
            )
            .unwrap();
            let mut kernel = device.compile(&lowered, candidate.clone()).unwrap();
            let xb = device
                .buffer_from(&x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
                .unwrap();
            let planes = weight
                .device_bytes()
                .iter()
                .map(|b| device.buffer_from(b).unwrap())
                .collect::<Vec<_>>();
            let out = device.buffer(m * n * 4).unwrap();
            let buffers = kernel
                .buffers()
                .iter()
                .map(|slot| match slot.parameter.as_str() {
                    "x" => xb.clone(),
                    "out" => out.clone(),
                    _ => planes[rep.plane_index(&slot.plane).unwrap()]
                    .clone(),
                })
                .collect::<Vec<_>>();
            kernel.execute(&buffers, &[]).unwrap();
            let mut bytes = vec![0; m * n * 4];
            out.read(&mut bytes).unwrap();
            for (b, e) in bytes.chunks_exact(4).zip(expected) {
                let a = f32::from_le_bytes(b.try_into().unwrap());
                assert!(
                    (a - e).abs() < 1e-5 + e.abs() * 1e-6,
                    "{} M{m}: {a} != {e}",
                    rep.name
                );
            }
        }
    }
}
#[test]
fn cpu_packed_matrix() {
    exercise(
        Device::cpu(),
        Candidate::Cpu {
            loads: seismic_realization::LoadStrategy::BorrowProvenReadOnly,
        },
    );
}
#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn metal_packed_matrix() {
    exercise(
        Device::metal().unwrap(),
        Candidate::Metal(Default::default()),
    );
}
#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_packed_matrix() {
    exercise(
        Device::cuda(0).unwrap(),
        Candidate::Cuda {
            options: seismic_realization::ScalarOptions {
                dispatch: seismic_realization::Dispatch::ParallelRoot,
                loads: seismic_realization::LoadStrategy::BorrowProvenReadOnly,
            },
            threads_per_block: 32,
        },
    );
}
