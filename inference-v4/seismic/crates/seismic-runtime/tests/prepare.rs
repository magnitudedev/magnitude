use seismic_accounting::selection::IntegerRange;
use seismic_compiler::tuner::Preparation;
use seismic_cuda::tuning::{self, BlockChoice};
use seismic_lang::{
    program::{compile, SourceFile},
    Scope,
};
use seismic_runtime::DeviceFacts;
use std::collections::HashMap;

fn phases() -> seismic_lang::lowered_ir::LoweredIr {
    let program = compile(&[SourceFile {
        path: "prepare.seismic.portable".into(),
        scope: Scope::Portable,
        text: "fn phases(x: tensor[5] f32, y: tensor[67] f32):\n  for row in parallel:\n    t = tile[1] f32\n    for i in owned(t): t[i] = 2.0\n    store(t,x[row:row+1])\n  for row in parallel:\n    t = tile[1] f32\n    for i in owned(t): t[i] = x[0]\n    store(t,y[row:row+1])\n".into(),
    }], &[]).unwrap();
    seismic_lang::lower::lower(&program, "phases", "cuda", &HashMap::new()).unwrap()
}

fn facts() -> DeviceFacts {
    // Pure preparation requires no driver. These explicit capacity facts make the
    // second phase's minimum block width different from the first phase's.
    DeviceFacts::Cuda(seismic_cuda::DeviceInfo {
        name: "preparation fixture".into(),
        compute_capability: (8, 0),
        driver_version: 0,
        max_threads_per_block: 64,
        max_grid_x: 4,
        warp_size: 32,
        multiprocessors: 2,
        global_memory_bytes: 1 << 20,
        l2_cache_bytes: 0,
        max_threads_per_multiprocessor: 128,
        registers_32bit_per_multiprocessor: 65536,
        shared_bytes_per_multiprocessor: 65536,
    })
}

#[test]
fn cuda_phase_family_establishes_each_dispatch_domain_before_selection() {
    let lowered = phases();
    let DeviceFacts::Cuda(device) = facts() else {
        unreachable!()
    };
    // Inspect the parallel execution family that automatic selection consumes.
    // This test constructs no executable and does not bypass the runtime gate.
    let Preparation::Choice { alternatives, .. } =
        tuning::prepare(&lowered, &device, &[1]).unwrap()
    else {
        panic!("expected unresolved phase dimensions")
    };
    let family = alternatives
        .owner::<IntegerRange<BlockChoice>>()
        .unwrap()
        .decision
        .family();
    assert_eq!(family.phase_count(), 2);
    assert_eq!(family.block_interval(0).unwrap(), 2..=64);
    assert_eq!(family.block_interval(1).unwrap(), 17..=64);
    assert!(!family.block_interval(1).unwrap().contains(&16));
    let Preparation::Execution(phases) =
        tuning::prepare(&lowered, &device, &[1, 32 - 2, 32 - 17]).unwrap()
    else {
        panic!("expected fully selected phase geometry")
    };
    assert_eq!(
        phases
            .iter()
            .map(|p| p.dispatch().threads_per_group)
            .collect::<Vec<_>>(),
        [32, 32]
    );
    assert_eq!(
        phases
            .iter()
            .map(|p| p.dispatch().groups)
            .collect::<Vec<_>>(),
        [1, 3]
    );
    assert!(tuning::prepare(&lowered, &device, &[1, 65 - 2]).is_err());
    let mut invalid = device;
    invalid.max_threads_per_block = 0;
    assert!(tuning::prepare(&lowered, &invalid, &[]).is_err());
    let mut wrong_backend = lowered;
    wrong_backend.backend = "cpu".into();
    assert!(tuning::prepare(&wrong_backend, &invalid, &[]).is_err());
}
