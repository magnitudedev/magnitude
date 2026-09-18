//! Cross-crate checks of structured IR -> selected memory plan -> accounting.
//! No emitted source, native compilation, or device is needed for these counts.
use seismic_accounting::quantity::Count;
use seismic_lang::{
    Scope,
    ir::{ExprKind, Stmt, StmtKind, VarId},
    lower::{Options, lower_with},
    lowered_ir::LoweredIr,
    program::{SourceFile, compile},
};
use seismic_metal::execution::{Config, prepare_storage_selected};
use seismic_metal::model as storage;
use seismic_realization::{LoadStrategy, dispatch::TilePlacement};
use std::collections::HashSet;

fn account(
    text: &str,
    piece: Option<i64>,
    shared: impl FnOnce(&LoweredIr) -> HashSet<VarId>,
) -> storage::StorageAccount {
    let program = compile(
        &[SourceFile {
            path: "counts.seismic.portable".into(),
            scope: Scope::Portable,
            text: text.into(),
        }],
        &[],
    )
    .unwrap();
    let lowered = lower_with(
        &program,
        "evaluate",
        "metal",
        &Default::default(),
        &Options {
            piece,
            ..Default::default()
        },
    )
    .unwrap();
    let shared = shared(&lowered);
    assert!(!shared.is_empty());
    let execution = prepare_storage_selected(
        &lowered,
        Config {
            loads: LoadStrategy::Materialize,
            ..Default::default()
        },
        &mut |decision| {
            Ok(if shared.contains(&decision.variable) {
                TilePlacement::GroupShared
            } else {
                TilePlacement::Replicated
            })
        },
    )
    .unwrap();
    assert!(execution.memory().launches().iter().all(|launch| {
        launch
            .barriers
            .keys()
            .all(|site| shared.contains(&site.variable))
    }));
    let dispatches = execution
        .phases()
        .iter()
        .flat_map(|p| std::iter::once(p.dispatch.clone()).chain(p.merge_dispatch.clone()))
        .collect::<Vec<_>>();
    storage::derive(execution.memory(), &dispatches).unwrap()
}

#[test]
fn barrier_counts_follow_ranges_and_runtime_branches() {
    for (condition, expected) in [
        ("true", Count::Exact(6)),
        ("false", Count::Exact(0)),
        ("enabled", Count::interval(0, 6).unwrap()),
    ] {
        let text = format!(
            "fn evaluate(x: tensor[2,65] f32, out: tensor[2,65] f32, enabled: bool):\n  for row in parallel:\n    acc = tile[65] f32\n    for i in owned(acc): acc[i] = 0.0\n    if {condition}:\n      for k in range(3):\n        t = load(x[row])\n        for i in owned(acc): acc[i] += t[(i+1)%65]\n    store(acc,out[row])\n"
        );
        let account = account(&text, None, |function| {
            function
                .vars
                .iter()
                .enumerate()
                .filter_map(|(id, var)| (var.name == "t").then_some(id))
                .collect()
        });
        assert_eq!(account.launches[0].static_barrier_sites, Count::Exact(1));
        assert_eq!(account.launches[0].barrier_executions, expected);
        assert!(
            account.launches[0]
                .native_private_bytes_per_lane
                .bounds()
                .is_none()
        );
    }
}

#[test]
fn streamed_tail_counts_chunks_rather_than_capacity_or_site_count() {
    let text = "fn evaluate(x: tensor[2,65] f32, out: tensor[2] f32):\n  for row in parallel:\n    acc = tile[1] f32\n    for i in owned(acc): acc[i] = 0.0\n    t = load(x[row])\n    acc[0] = reduce(t,0,sum,ordered=true)\n    store(acc,out[row:row+1])\n";
    // Put the retained reduction inputs in shared memory and keep the reduction's
    // state and implementation temporaries private. Each input publication then
    // contributes one barrier per logical chunk. The compiler gives full chunks
    // and a nonempty remainder separate static sites; the remainder executes once.
    fn inputs(body: &[Stmt], shared: &mut HashSet<VarId>) {
        for statement in body {
            match &statement.kind {
                StmtKind::Reduction(reduction) => {
                    for input in &reduction.inputs {
                        let ExprKind::Var(variable) = input.kind else {
                            panic!("fixture reduction input must have an explicit binding")
                        };
                        shared.insert(variable);
                    }
                }
                StmtKind::Parallel { body, .. }
                | StmtKind::Range { body, .. }
                | StmtKind::Owned { body, .. }
                | StmtKind::LoadLoop { body, .. }
                | StmtKind::Lanes { body, .. } => inputs(body, shared),
                StmtKind::If { then, els, .. } => {
                    inputs(then, shared);
                    inputs(els, shared);
                }
                _ => {}
            }
        }
    }
    for (piece, static_sites, expected) in [
        (None, 1, 2),
        (Some(1), 1, 130),
        (Some(13), 1, 10),
        (Some(16), 2, 10),
        (Some(32), 2, 6),
        (Some(65), 1, 2),
        (Some(80), 1, 2),
    ] {
        let account = account(text, piece, |function| {
            let mut shared = HashSet::new();
            inputs(&function.body, &mut shared);
            shared
        });
        assert_eq!(
            account.launches[0].static_barrier_sites,
            Count::Exact(static_sites)
        );
        assert_eq!(
            account.launches[0].barrier_executions,
            Count::Exact(expected),
            "piece={piece:?}"
        );
    }
}
