mod support;
use seismic_lang::{
    Scope,
    ir::LoadMode,
    lowered_ir::LoweredIr,
    program::{SourceFile, compile},
};
use seismic_metal::{
    execution::{self, Config, Execution},
    family::GroupFamily,
    msl,
};
use seismic_realization::{LoadStrategy, dispatch::TilePlacement};
use std::sync::Arc;
const TWO: &str = "fn evaluate(x: tensor[5,64] f32, middle: tensor[5,64] f32, out: tensor[5] f32):\n  for row in parallel:\n    a = load(x[row])\n    y = tile[64] f32\n    for i in owned(y): y[i] = a[(i+1)%64]\n    store(y,middle[row])\n  for row in parallel:\n    y = tile[1] f32\n    for i in owned(y): y[i] = middle[row,0]\n    store(y,out[row:row+1])\n";
const SPLIT: &str = "fn evaluate(x: tensor[2,65] f32, middle: tensor[2] f32, out: tensor[2] f32):\n  for row in parallel:\n    acc = tile[1] f32\n    for i in owned(acc): acc[i] = 0.0\n    chunk = load(x[row,0:65])\n    acc[0] += reduce(chunk,0,sum)\n    store(acc,middle[row:row+1])\n  for row in parallel:\n    y = tile[1] f32\n    for i in owned(y): y[i] = middle[row] * 2.0 + 1.0\n    store(y,out[row:row+1])\n";

fn lowered(text: &str) -> LoweredIr {
    let program = compile(
        &[SourceFile {
            path: "refinement.seismic.portable".into(),
            scope: Scope::Portable,
            text: text.into(),
        }],
        &[],
    )
    .unwrap();
    seismic_lang::lower::lower(&program, "evaluate", "metal", &Default::default()).unwrap()
}
fn config() -> Config {
    Config {
        loads: LoadStrategy::Materialize,
        sg_per_tg: 1,
        max_threads_per_threadgroup: 96,
        max_threadgroup_bytes: 1024,
        ..Default::default()
    }
}
fn family(function: &LoweredIr, config: Config) -> Arc<GroupFamily> {
    Arc::new(
        GroupFamily::derive(
            execution::prepare_with_choices(
                function,
                config,
                &mut |_, _| Ok(LoadMode::Materialize),
                &mut |_| Ok(TilePlacement::GroupShared),
                &mut |decision| Ok(decision.diagnostic()),
            )
            .unwrap(),
        )
        .unwrap(),
    )
}
fn refine(family: &Arc<GroupFamily>, values: &[u64]) -> Execution {
    family.select_launches(values).unwrap()
}
#[test]
fn split_merge_refinement_preserves_allocation_identity_and_launch_handoffs() {
    let function = support::streamed(SPLIT, 17, &["chunk"]);
    let family = family(
        &function,
        Config {
            split: 3,
            ..config()
        },
    );
    assert_eq!(family.execution().memory().launches().len(), 3);
    for groups in [[1, 3, 2], [3, 1, 2], [2, 2, 3]] {
        let selected = refine(&family, &groups);
        let expected = family.select_launches(&groups).unwrap();
        assert_eq!(selected.memory(), expected.memory());
        assert_eq!(selected.storage(), family.execution().storage());
        assert_eq!(selected.reductions(), family.execution().reductions());
        assert_eq!(
            selected.memory().scratch(),
            family.execution().memory().scratch()
        );
        for (actual, baseline) in selected
            .memory()
            .launches()
            .iter()
            .zip(family.execution().memory().launches())
        {
            assert_eq!(
                actual
                    .arrays
                    .iter()
                    .map(|a| (a.id, a.slot))
                    .collect::<Vec<_>>(),
                baseline
                    .arrays
                    .iter()
                    .map(|a| (a.id, a.slot))
                    .collect::<Vec<_>>()
            );
            assert_eq!(actual.barriers, baseline.barriers);
            assert_eq!(actual.predecessor, baseline.predecessor);
        }
        let emitted = msl::emit_execution(&selected).unwrap();
        assert_eq!(emitted, msl::emit_execution(&expected).unwrap());
        assert_eq!(
            emitted
                .launches
                .iter()
                .map(|l| l.kernel.as_str())
                .collect::<Vec<_>>(),
            ["evaluate_0", "evaluate_0_merge", "evaluate_1"]
        );
        assert_eq!(
            emitted
                .launches
                .iter()
                .map(|l| l.dispatch.as_ref().unwrap().items_per_group)
                .collect::<Vec<_>>(),
            groups
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn native_refined_split_launches_keep_independent_groupings() {
    let device = seismic_metal::runtime::Device::open().unwrap();
    let function = support::streamed(SPLIT, 17, &["chunk"]);
    let family = family(
        &function,
        Config {
            split: 3,
            ..config()
        },
    );
    let input = (0..130).map(|i| i as f32 * 0.25 - 7.0).collect::<Vec<_>>();
    let bytes = input
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    let expected = input
        .chunks(65)
        .map(|row| row.iter().sum::<f32>() * 2.0 + 1.0)
        .collect::<Vec<_>>();
    for groups in [[1, 3, 2], [3, 1, 2], [2, 2, 3]] {
        let selected = refine(&family, &groups);
        let kernel = device
            .compile(msl::emit_execution(&selected).unwrap())
            .unwrap();
        let x = device.buffer_from(&bytes).unwrap();
        let middle = device.buffer(8).unwrap();
        let out = device.buffer(8).unwrap();
        device.run(&kernel, &[&x, &middle, &out], &[], 1).unwrap();
        let actual = out
            .read(8)
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected, "groupings={groups:?}");
    }
}

#[test]
fn symbolic_grouping_preserves_independent_launch_domains_and_shared_storage() {
    use magnitude_solver::{
        Limits, Options, Outcome, Search,
        model::{Constraint, Domain, ModelBuilder},
    };
    let family = family(&lowered(TWO), config());
    for first in [1, 2] {
        for second in [1, 2, 3] {
            let mut builder = ModelBuilder::new();
            let symbolic = family.append_dispatch(&mut builder, "Metal", &[]).unwrap();
            for (launch, items) in symbolic.launches.iter().zip([first, second]) {
                builder.constraint(Constraint::InDomain {
                    variable: launch.items_per_group.id(),
                    domain: Domain::singleton(items),
                });
            }
            let mut search =
                Search::new(Arc::new(builder.build().unwrap()), Options::default()).unwrap();
            let Outcome::Optimal(solution) = search.advance(Limits::default()).unwrap() else {
                panic!("every original grouping must retain a symbolic witness")
            };
            let selected = symbolic.reconstruct(solution.values()).unwrap();
            let expected = refine(&family, &[first as u64, second as u64]);
            assert_eq!(selected.function(), expected.function());
            assert_eq!(
                msl::emit_execution(&selected).unwrap(),
                msl::emit_execution(&expected).unwrap()
            );
        }
    }
}
