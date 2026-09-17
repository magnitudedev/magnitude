use cranelift_codegen::isa::CallConv;
use seismic_accounting::{
    quantity::Count,
    realization::{scalar, ScalarAccount},
};
use seismic_lang::{
    program::{compile, SourceFile},
    Scope,
};
use seismic_realization::{execution::MemoryObject, Dispatch};
use std::collections::HashMap;
fn account(text: &str, name: &str, n: i64, dispatch: Dispatch) -> ScalarAccount {
    let p = compile(
        &[SourceFile {
            path: "case.seismic.portable".into(),
            scope: Scope::Portable,
            text: text.into(),
        }],
        &[],
    )
    .unwrap();
    let l = seismic_lang::lower::lower(&p, name, "cpu", &HashMap::from([("N".into(), n)])).unwrap();
    scalar(&seismic_compiler::scalar_with(&l, CallConv::SystemV, dispatch).unwrap())
}
fn buffer(name: &str) -> MemoryObject {
    MemoryObject::Buffer {
        parameter: name.into(),
        plane: String::new(),
    }
}
const COPY:&str="fn copy[N](x: tensor[N] f32, out: tensor[N] f32):\n  for row in parallel:\n    t = load(x[row : row + 1])\n    store(t, out[row : row + 1])\n";
#[test]
fn native_domains_count_work_and_abi_at_their_actual_frequency() {
    for dispatch in [Dispatch::Sequential, Dispatch::ParallelRoot] {
        let a = account(COPY, "copy", 1_000_000, dispatch);
        assert_eq!(a.traffic[&buffer("x")].reads, Count::Exact(4_000_000));
        assert_eq!(a.traffic[&buffer("out")].writes, Count::Exact(4_000_000));
        let invocations = if dispatch == Dispatch::Sequential {
            1
        } else {
            1_000_000
        };
        assert_eq!(a.invocation_count, invocations);
        assert_eq!(
            a.traffic[&MemoryObject::BufferTable].reads,
            Count::Exact(16 * invocations)
        );
        assert_eq!(
            a.traffic[&MemoryObject::PrivateScratch].writes,
            Count::Exact(4_000_000)
        );
        assert_eq!(
            a.traffic[&MemoryObject::PrivateScratch].reads,
            Count::Exact(4_000_000)
        );
        assert!(a.unavailable.is_empty(), "{:?}", a.unavailable);
        assert!(
            a.instructions.len() < 200,
            "analysis should scale with emitted program, not domain extent"
        );
    }
}
#[test]
fn zero_domain_has_no_parallel_invocations_but_sequential_abi_still_runs() {
    let parallel = account(COPY, "copy", 0, Dispatch::ParallelRoot);
    assert_eq!(parallel.invocation_count, 0);
    assert!(parallel
        .instructions
        .iter()
        .all(|i| i.count == Count::Exact(0)));
    let sequential = account(COPY, "copy", 0, Dispatch::Sequential);
    assert_eq!(
        sequential.traffic[&MemoryObject::BufferTable].reads,
        Count::Exact(16)
    );
    assert!(!sequential.traffic.contains_key(&buffer("x")));
}
#[test]
fn conditional_stores_are_bounded_not_assumed_to_execute() {
    let text="fn branch[N](out: tensor[1] f32, choose: bool):\n  y = tile[1] f32\n  for i in owned(y): y[i] = 7.0\n  if choose:\n    store(y, out)\n";
    let a = account(text, "branch", 1, Dispatch::Sequential);
    assert_eq!(
        a.traffic[&buffer("out")].writes,
        Count::interval(0, 4).unwrap()
    );
    assert_eq!(
        a.traffic[&MemoryObject::ScalarArguments].reads,
        Count::Exact(1)
    );
}
#[test]
fn product_preserves_unknown_zero_and_overflow() {
    assert_eq!(
        Count::interval(2, 3)
            .unwrap()
            .multiply(&Count::interval(4, 5).unwrap()),
        Count::interval(8, 15).unwrap()
    );
    assert_eq!(
        Count::unknown("not executed").multiply(&Count::Exact(0)),
        Count::Exact(0)
    );
    assert!(Count::Exact(u64::MAX)
        .multiply(&Count::Exact(2))
        .bounds()
        .is_none());
}

#[test]
fn borrowing_streams_changes_consumption_without_changing_the_algorithm() {
    use seismic_lang::program::collect_files;
    let root =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../seismic-std/lib");
    let p = compile(
        &collect_files(&[root]).unwrap(),
        &["cpu".into(), "metal".into(), "cuda".into()],
    )
    .unwrap();
    let l = seismic_lang::lower::lower(
        &p,
        "projection",
        "cuda",
        &HashMap::from([("N".into(), 128), ("K".into(), 256)]),
    )
    .unwrap();
    let realize = |loads| {
        scalar(
            &seismic_compiler::scalar_candidate(
                &l,
                CallConv::SystemV,
                seismic_realization::ScalarOptions {
                    dispatch: Dispatch::ParallelRoot,
                    loads,
                },
            )
            .unwrap(),
        )
    };
    let eager = realize(seismic_realization::LoadStrategy::Materialize);
    let borrowed = realize(seismic_realization::LoadStrategy::BorrowProvenReadOnly);
    assert_eq!(eager.scratch_bytes_per_dispatch, Count::Exact(197632));
    assert_eq!(borrowed.scratch_bytes_per_dispatch, Count::Exact(1024));
    assert_eq!(
        borrowed.traffic[&buffer("x")].reads,
        eager.traffic[&buffer("x")].reads
    );
    assert_eq!(
        borrowed.traffic[&MemoryObject::PrivateScratch].writes,
        Count::Exact(131584)
    );
    assert_eq!(
        eager.traffic[&MemoryObject::PrivateScratch].writes,
        Count::Exact(328192)
    );
    assert!(borrowed.unavailable.is_empty());
    // This establishes concrete consumption, not an assumption that less storage
    // always predicts shorter runtime on any device.
}

#[test]
fn runtime_stream_traffic_is_unknown_without_runtime_domain_evidence() {
    let text="fn stream[N](x: tensor[N] f32, visible: tensor[2] i32, out: tensor[1] f32):\n  acc = tile[1] f32\n  for i in owned(acc): acc[i] = 0.0\n  for t in load(x[visible[0]:visible[1]], over=0):\n    acc[0] += reduce(t, 0, sum)\n  store(acc, out)\n";
    let a = account(text, "stream", 137, Dispatch::Sequential);
    assert!(matches!(
        a.traffic[&buffer("x")].reads,
        Count::Unknown { .. }
    ));
    assert_ne!(a.traffic[&buffer("x")].reads, Count::Exact(0));
    assert_eq!(a.traffic[&buffer("out")].writes, Count::Exact(4));
}

#[test]
fn ordered_phases_keep_domains_and_completion_edges() {
    use seismic_lang::{
        program::{compile, SourceFile},
        Scope,
    };
    let program=compile(&[SourceFile{path:"phases.seismic.portable".into(),scope:Scope::Portable,text:"fn phases(x: tensor[3] f32, y: tensor[7] f32):\n  for row in parallel:\n    a = tile[1] f32\n    for i in owned(a): a[i] = 1.0\n    store(a,x[row:row+1])\n  for row in parallel:\n    b = tile[1] f32\n    for i in owned(b): b[i] = x[0]\n    store(b,y[row:row+1])\n".into()}],&[]).unwrap();
    let lowered = seismic_lang::lower::lower(
        &program,
        "phases",
        "cuda",
        &std::collections::HashMap::new(),
    )
    .unwrap();
    let sequence = seismic_compiler::scalar_sequence(
        &lowered,
        seismic_realization::CallConv::SystemV,
        seismic_realization::ScalarOptions {
            dispatch: seismic_realization::Dispatch::ParallelRoot,
            loads: seismic_realization::LoadStrategy::Materialize,
        },
    )
    .unwrap();
    let account = seismic_accounting::realization::sequence(&sequence);
    assert_eq!(account.completion_edges, vec![(0, 1)]);
    assert_eq!(account.phases[0].source_statement, 0);
    assert_eq!(account.phases[1].source_statement, 1);
    assert_eq!(account.phases[0].account.invocation_count, 3);
    assert_eq!(account.phases[1].account.invocation_count, 7);
    assert_eq!(
        account.retained_scratch_bytes,
        account.phases[0]
            .account
            .scratch_bytes_per_dispatch
            .add(&account.phases[1].account.scratch_bytes_per_dispatch)
    );
}
