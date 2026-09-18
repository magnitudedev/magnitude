use seismic_lang::{
    lower::lower,
    program::{compile, SourceFile},
    Scope,
};
use seismic_metal::{
    execution::{prepare_storage_selected, Config},
    msl::emit_execution,
};
use seismic_realization::{dispatch::TilePlacement, LoadStrategy};

fn ir(text: &str) -> seismic_lang::lowered_ir::LoweredIr {
    let p = compile(
        &[SourceFile {
            path: "allocations.seismic.portable".into(),
            scope: Scope::Portable,
            text: text.into(),
        }],
        &[],
    )
    .unwrap();
    lower(&p, "evaluate", "metal", &Default::default()).unwrap()
}
#[test]
fn repeated_definitions_have_distinct_allocation_sites() {
    let source = ir("fn evaluate(x: tensor[6] f32, out: tensor[6] f32):\n  a = load(x)\n  a = load(x)\n  store(a,out)\n");
    let execution = prepare_storage_selected(
        &source,
        Config {
            loads: LoadStrategy::Materialize,
            ..Default::default()
        },
        &mut |_| Ok(TilePlacement::Replicated),
    )
    .unwrap();
    let launch = &execution.memory().launches()[0];
    assert_eq!(launch.arrays.len(), 2);
    assert_eq!(launch.arrays[0].id.variable, launch.arrays[1].id.variable);
    assert_ne!(launch.arrays[0].id, launch.arrays[1].id);
    assert_eq!(launch.declared_private_bytes_per_lane, 48);
    assert_eq!(launch.shared_bytes_per_group, 0);
    assert!(launch.fragments.is_empty());
    emit_execution(&execution).unwrap();
}
#[test]
fn branch_scopes_and_shared_capacity_are_known_before_emission() {
    let source = ir("fn evaluate(x: tensor[6] f32, out: tensor[6] f32, enabled: bool):\n  if enabled:\n    a = load(x)\n    store(a,out)\n  else:\n    b = load(x)\n    store(b,out)\n");
    for placement in [TilePlacement::Replicated, TilePlacement::GroupShared] {
        let execution = prepare_storage_selected(
            &source,
            Config {
                loads: LoadStrategy::Materialize,
                max_threadgroup_bytes: 192,
                ..Default::default()
            },
            &mut |_| Ok(placement.clone()),
        )
        .unwrap();
        let launch = &execution.memory().launches()[0];
        assert_eq!(launch.arrays.len(), 2);
        if placement == TilePlacement::GroupShared {
            assert_eq!(launch.shared_bytes_per_group, 192);
            assert_eq!(launch.arrays[0].scope, launch.arrays[1].scope);
        } else {
            assert_eq!(launch.declared_private_bytes_per_lane, 48);
            assert_ne!(launch.arrays[0].scope, launch.arrays[1].scope);
        }
        emit_execution(&execution).unwrap();
    }
    let rejected = prepare_storage_selected(
        &source,
        Config {
            loads: LoadStrategy::Materialize,
            max_threadgroup_bytes: 191,
            ..Default::default()
        },
        &mut |_| Ok(TilePlacement::GroupShared),
    );
    assert!(rejected
        .err()
        .unwrap()
        .contains("192 bytes of threadgroup memory"));
}
#[test]
fn empty_ranges_are_removed_before_allocations_are_selected() {
    let source = ir("fn evaluate(x: tensor[6] f32, out: tensor[6] f32):\n  for k in range(0):\n    unused = tile[4096] f32\n    for i in owned(unused): unused[i] = 0.0\n  a = load(x)\n  store(a,out)\n");
    let execution = prepare_storage_selected(
        &source,
        Config {
            loads: LoadStrategy::Materialize,
            max_threadgroup_bytes: 96,
            ..Default::default()
        },
        &mut |_| Ok(TilePlacement::GroupShared),
    )
    .unwrap();
    let launch = &execution.memory().launches()[0];
    assert_eq!(launch.arrays.len(), 1);
    assert_eq!(launch.shared_bytes_per_group, 96);
    emit_execution(&execution).unwrap();
}
