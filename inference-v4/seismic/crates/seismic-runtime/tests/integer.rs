use seismic_lang::{
    program::{compile, SourceFile},
    Scope,
};
use seismic_runtime::{Candidate, Device};
use std::collections::HashMap;
fn exercise(device: Device, candidate: Candidate) {
    let source="fn signed[N](a: tensor[N] i32, b: tensor[N] i32, q: tensor[N] i32, r: tensor[N] i32):\n  for row in parallel:\n    at = load(a[row:row+1])\n    bt = load(b[row:row+1])\n    qt = tile[1] i32\n    rt = tile[1] i32\n    for i in owned(qt): qt[i] = at[i] / bt[i]; rt[i] = at[i] % bt[i]\n    store(qt,q[row:row+1])\n    store(rt,r[row:row+1])\n\nfn unsigned[N](a: tensor[N] u32, b: tensor[N] u32, q: tensor[N] u32, r: tensor[N] u32):\n  for row in parallel:\n    at = load(a[row:row+1])\n    bt = load(b[row:row+1])\n    qt = tile[1] u32\n    rt = tile[1] u32\n    for i in owned(qt): qt[i] = at[i] / bt[i]; rt[i] = at[i] % bt[i]\n    store(qt,q[row:row+1])\n    store(rt,r[row:row+1])\n";
    let program = compile(
        &[SourceFile {
            path: "integer.seismic.portable".into(),
            scope: Scope::Portable,
            text: source.into(),
        }],
        &[],
    )
    .unwrap();
    let values = [-17i32, -3, -1, 0, 1, 3, 17, i32::MIN, i32::MAX];
    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut quotient = Vec::new();
    let mut remainder = Vec::new();
    for x in values {
        for y in values {
            if y == 0 || (x == i32::MIN && y == -1) {
                continue;
            }
            a.push(x);
            b.push(y);
            quotient.push(x.div_euclid(y));
            remainder.push(x.rem_euclid(y));
        }
    }
    let encode = |values: &[i32]| {
        values
            .iter()
            .flat_map(|n| n.to_le_bytes())
            .collect::<Vec<_>>()
    };
    let lowered = seismic_lang::lower::lower(
        &program,
        "signed",
        device.backend(),
        &HashMap::from([("N".into(), a.len() as i64)]),
    )
    .unwrap();
    let mut kernel = device.compile(&lowered, candidate.clone()).unwrap();
    let aa = device.buffer_from(&encode(&a)).unwrap();
    let bb = device.buffer_from(&encode(&b)).unwrap();
    let q = device.buffer(a.len() * 4).unwrap();
    let r = device.buffer(a.len() * 4).unwrap();
    let buffers = [aa.clone(), bb.clone(), q.clone(), r.clone()];
    kernel.execute(&buffers, &[]).unwrap();
    let mut got = vec![0; a.len() * 4];
    q.read(&mut got).unwrap();
    assert_eq!(got, encode(&quotient));
    r.read(&mut got).unwrap();
    assert_eq!(got, encode(&remainder));
    b[0] = 0;
    bb.write(&encode(&b)).unwrap();
    assert!(kernel.execute(&buffers, &[]).is_err());
    a[0] = i32::MIN;
    b[0] = -1;
    aa.write(&encode(&a)).unwrap();
    bb.write(&encode(&b)).unwrap();
    assert!(kernel.execute(&buffers, &[]).is_err());
    a[0] = -17;
    b[0] = 3;
    aa.write(&encode(&a)).unwrap();
    bb.write(&encode(&b)).unwrap();
    kernel.execute(&buffers, &[]).unwrap();
    let aa = [0u32, 1, u32::MAX, 1 << 31, 17, 77];
    let bb = [1u32, 3, 2, u32::MAX, 5, 9];
    let encode = |values: &[u32]| {
        values
            .iter()
            .flat_map(|n| n.to_le_bytes())
            .collect::<Vec<_>>()
    };
    let lowered = seismic_lang::lower::lower(
        &program,
        "unsigned",
        device.backend(),
        &HashMap::from([("N".into(), aa.len() as i64)]),
    )
    .unwrap();
    let mut kernel = device.compile(&lowered, candidate).unwrap();
    let q = device.buffer(24).unwrap();
    let r = device.buffer(24).unwrap();
    kernel
        .execute(
            &[
                device.buffer_from(&encode(&aa)).unwrap(),
                device.buffer_from(&encode(&bb)).unwrap(),
                q.clone(),
                r.clone(),
            ],
            &[],
        )
        .unwrap();
    let mut got = vec![0; 24];
    q.read(&mut got).unwrap();
    assert_eq!(
        got,
        encode(&aa.iter().zip(bb).map(|(a, b)| a / b).collect::<Vec<_>>())
    );
    r.read(&mut got).unwrap();
    assert_eq!(
        got,
        encode(&aa.iter().zip(bb).map(|(a, b)| a % b).collect::<Vec<_>>())
    );
}
#[test]
fn cpu_integer_division() {
    exercise(
        Device::cpu(),
        Candidate::Cpu {
            loads: seismic_realization::LoadStrategy::Materialize,
        },
    )
}
#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_integer_division() {
    exercise(
        Device::cuda(0).unwrap(),
        Candidate::Cuda {
            options: seismic_realization::ScalarOptions {
                dispatch: seismic_realization::Dispatch::ParallelRoot,
                loads: seismic_realization::LoadStrategy::Materialize,
            },
            threads_per_block: 32,
        },
    )
}
#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn metal_integer_division() {
    exercise(
        Device::metal().unwrap(),
        Candidate::Metal(Default::default()),
    )
}
