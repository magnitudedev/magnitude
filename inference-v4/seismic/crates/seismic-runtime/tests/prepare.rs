use seismic_lang::{program::{compile, SourceFile}, Scope};
use seismic_realization::{Dispatch, LoadStrategy, ScalarOptions};
use seismic_runtime::{Candidate, DeviceFacts, execution::Execution};
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
        name: "preparation fixture".into(), compute_capability: (8, 0), driver_version: 0,
        max_threads_per_block: 64, max_grid_x: 4, warp_size: 32, multiprocessors: 2,
        global_memory_bytes: 1 << 20, l2_cache_bytes: 0,
        max_threads_per_multiprocessor: 128, registers_32bit_per_multiprocessor: 65536,
        shared_bytes_per_multiprocessor: 65536,
    })
}

fn candidate(threads_per_block: u32) -> Candidate {
    Candidate::Cuda {
        options: ScalarOptions { dispatch: Dispatch::ParallelRoot, loads: LoadStrategy::Materialize },
        threads_per_block,
    }
}

#[test]
fn explicit_cuda_candidate_obeys_every_phase_choice_domain() {
    let lowered = phases();
    let facts = facts();
    let Execution::Cuda(phases) = Execution::prepare(&lowered, candidate(32), &facts).unwrap() else { panic!("expected CUDA phases") };
    assert_eq!(phases.len(), 2);
    assert_eq!(phases.iter().map(|p| p.dispatch().threads_per_group).collect::<Vec<_>>(), [32, 32]);
    assert_eq!(phases.iter().map(|p| p.dispatch().groups).collect::<Vec<_>>(), [1, 3]);
    assert!(matches!(Execution::prepare(&lowered, candidate(16), &facts), Err(e) if e.contains("phase 1")));
    assert!(Execution::prepare(&lowered, candidate(0), &facts).is_err());
    assert!(Execution::prepare(&lowered, candidate(65), &facts).is_err());
    let wrong = DeviceFacts::Cpu { architecture: std::env::consts::ARCH, operating_system: std::env::consts::OS };
    assert!(Execution::prepare(&lowered, candidate(32), &wrong).is_err());
}
