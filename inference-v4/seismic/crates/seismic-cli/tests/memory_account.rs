//! Cross-crate checks of structured IR -> selected memory plan -> accounting.
//! No emitted source, native compilation, or device is needed for these counts.
use seismic_accounting::quantity::Count;
use seismic_metal::model as storage;
use seismic_lang::{
    lower::{lower_with, Options},
    program::{compile, SourceFile},
    Scope,
};
use seismic_metal::execution::{prepare_storage_selected, Config};
use seismic_realization::{dispatch::TilePlacement, LoadStrategy};

fn account(text: &str, piece: Option<i64>) -> storage::StorageAccount {
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
        &Options { piece },
    )
    .unwrap();
    let execution = prepare_storage_selected(
        &lowered,
        Config {
            loads: LoadStrategy::Materialize,
            ..Default::default()
        },
        &mut |decision| {
            Ok(if decision.name == "acc" {
                TilePlacement::Replicated
            } else {
                TilePlacement::GroupShared
            })
        },
    )
    .unwrap();
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
        let text = format!("fn evaluate(x: tensor[2,65] f32, out: tensor[2,65] f32, enabled: bool):\n  for row in parallel:\n    acc = tile[65] f32\n    for i in owned(acc): acc[i] = 0.0\n    if {condition}:\n      for k in range(3):\n        t = load(x[row])\n        for i in owned(acc): acc[i] += t[(i+1)%65]\n    store(acc,out[row])\n");
        let account = account(&text, None);
        assert_eq!(account.launches[0].static_barrier_sites, Count::Exact(1));
        assert_eq!(account.launches[0].barrier_executions, expected);
        assert!(account.launches[0]
            .native_private_bytes_per_lane
            .bounds()
            .is_none());
    }
}

#[test]
fn streamed_tail_counts_chunks_rather_than_capacity_or_site_count() {
    let text = "fn evaluate(x: tensor[2,65] f32, out: tensor[2] f32):\n  for row in parallel:\n    acc = tile[1] f32\n    for i in owned(acc): acc[i] = 0.0\n    for t in load(x[row],over=0):\n      acc[0] += reduce(t,0,sum)\n    store(acc,out[row:row+1])\n";
    for (piece, expected) in [
        (None, 2),
        (Some(16), 10),
        (Some(32), 6),
        (Some(65), 2),
        (Some(80), 2),
    ] {
        let account = account(text, piece);
        assert_eq!(account.launches[0].static_barrier_sites, Count::Exact(1));
        assert_eq!(
            account.launches[0].barrier_executions,
            Count::Exact(expected),
            "piece={piece:?}"
        );
    }
}
