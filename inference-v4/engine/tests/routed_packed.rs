use seismic_engine::models::qwen35::program::program;
use seismic_lang::{
    interp::TensorData,
    lower::{Options, lower_specialized},
    numeric::bf16_round,
    repr,
    types::{DType, Elem},
};
use seismic_runtime::{Candidate, Device};
use std::collections::HashMap;
fn exercise(device: Device, candidate: Candidate) {
    let program = program().unwrap();
    let routes = [2i32, 0, 1, 2];
    for repname in ["q4g32", "q8g32a", "q8g16z32", "q8g32s", "iq4g32"] {
        let rep = repr::lookup(repname).unwrap();
        let (e, m, k, n, d) = (3usize, 2usize, 2usize, 5usize, 256usize);
        let weight = TensorData::Packed {
            repr: rep,
            shape: vec![e, n, d],
            words: (0..e * n * d / rep.codes_per_word() as usize)
                .map(|i| 0x17491b37u32.wrapping_mul(i as u32 + 1))
                .collect(),
            scale: vec![0.015625; e * n * d / rep.group as usize],
            bias: if rep.has_bias {
                vec![0.0625; e * n * d / rep.group as usize]
            } else {
                vec![]
            },
        };
        let weightplanes = weight
            .device_bytes()
            .iter()
            .map(|b| device.buffer_from(b).unwrap())
            .collect::<Vec<_>>();
        for entry in ["routed_input", "routed_output"] {
            let xcount = if entry == "routed_input" {
                m * d
            } else {
                m * k * d
            };
            let values = (0..xcount)
                .map(|i| bf16_round(((i % 31) as f32 - 15.) / 32.))
                .collect::<Vec<_>>();
            let x = device
                .buffer_from(
                    &values
                        .iter()
                        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let route = device
                .buffer_from(
                    &routes
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let out = device.buffer(m * k * n * 2).unwrap();
            let shapes = HashMap::from([
                ("M".into(), m as i64),
                ("E".into(), e as i64),
                ("K".into(), k as i64),
                (
                    "H".into(),
                    if entry == "routed_input" { d } else { n } as i64,
                ),
                (
                    "F".into(),
                    if entry == "routed_input" { n } else { d } as i64,
                ),
            ]);
            let lowered = lower_specialized(
                &program,
                entry,
                device.backend(),
                &shapes,
                &HashMap::from([
                    ("A".into(), Elem::Dtype(DType::BF16)),
                    ("W".into(), Elem::Repr(repname.into())),
                ]),
                &Options::default(),
            )
            .unwrap();
            let mut kernel = device.compile(&lowered, candidate.clone()).unwrap();
            let buffers = kernel
                .buffers()
                .iter()
                .map(|s| match s.parameter.as_str() {
                    "x" => x.clone(),
                    "routes" => route.clone(),
                    "out" => out.clone(),
                    "weight" => weightplanes[match s.plane.as_str() {
                        "words" => 0,
                        "scale" => 1,
                        "bias" => 2,
                        _ => panic!(),
                    }]
                    .clone(),
                    _ => panic!(),
                })
                .collect::<Vec<_>>();
            kernel.execute(&buffers, &[]).unwrap();
            let mut bytes = vec![0; m * k * n * 2];
            out.read(&mut bytes).unwrap();
            for (i, b) in bytes.chunks_exact(2).enumerate() {
                let row = i / (k * n);
                let choice = (i / n) % k;
                let column = i % n;
                let expert = routes[row * k + choice] as usize;
                let xrow = if entry == "routed_input" {
                    row
                } else {
                    row * k + choice
                };
                let expected = bf16_round((0..d).fold(0f32, |a, j| {
                    values[xrow * d + j]
                        .mul_add(weight.get((expert * n + column) * d + j) as f32, a)
                }));
                let actual =
                    f32::from_bits((u16::from_le_bytes(b.try_into().unwrap()) as u32) << 16);
                assert_eq!(actual, expected, "{repname} {entry} element{i}");
            }
            route
                .write(
                    &[e as i32, 0, 1, 2]
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            assert!(
                kernel.execute(&buffers, &[]).is_err(),
                "invalid route must fail"
            );
        }
    }
}
#[test]
fn cpu_routed_packed() {
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
fn metal_routed_packed() {
    exercise(
        Device::metal().unwrap(),
        Candidate::Metal(Default::default()),
    );
}
#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_routed_packed() {
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
