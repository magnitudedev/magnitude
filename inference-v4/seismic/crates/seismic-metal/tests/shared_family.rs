use magnitude_solver::{
    Limits, Options, Outcome, Search,
    model::{Constraint, Domain, ModelBuilder},
};
use seismic_accounting::algebra::{Algebra, Symbolic};
use seismic_lang::{
    Scope,
    ir::LoadMode,
    program::{SourceFile, compile},
};
use seismic_metal::{
    execution::{self, Config},
    family::{GroupFamily, decomposition},
    storage::StorageFamily,
    tuning::{Capacities, Form},
};
use seismic_realization::dispatch::{GroupDispatch, TilePlacement};
use std::sync::Arc;

fn source() -> seismic_lang::lowered_ir::LoweredIr {
    let program = compile(&[SourceFile { path: "family.seismic.portable".into(), scope: Scope::Portable,
        text: "fn evaluate(x: tensor[3,5] f32, out: tensor[3,5] f32):\n  for row in parallel:\n    a = load(x[row])\n    y = tile[5] f32\n    for i in owned(y): y[i] = a[i] + 1.0\n    store(y,out[row])\n".into() }], &[]).unwrap();
    seismic_lang::lower::lower(&program, "evaluate", "metal", &Default::default()).unwrap()
}
#[test]
fn partition_mapping_and_grouping_share_original_numeric_parameters() {
    let source = source();
    let mut builder = ModelBuilder::new();
    let binding = decomposition::Binding::append(
        &mut builder,
        "metal",
        &source,
        &Form::Automatic,
        &Capacities {
            max_threads_per_threadgroup: 64,
            max_threadgroup_bytes: 4096,
        },
    )
    .unwrap();
    assert_eq!(binding.partition.bounds(), (0, 5));
    assert_eq!(binding.phases[0].axes[0].step.bounds(), (1, 3));
    for (variable, value) in [
        (binding.partition.id(), 2),
        (binding.phases[0].axes[0].step.id(), 2),
        (binding.phases[0].main.items_per_group.id(), 2),
    ] {
        builder.constraint(Constraint::InDomain {
            variable,
            domain: Domain::singleton(value),
        });
    }
    let model = Arc::new(builder.build().unwrap());
    let mut search = Search::new(model.clone(), Options::default()).unwrap();
    let Outcome::Optimal(witness) = search.advance(Limits::default()).unwrap() else {
        panic!("small geometry family has a feasible witness")
    };
    let selected = binding.reconstruct(witness.values()).unwrap();
    assert_eq!(selected.decomposition.tile_piece, Some(2));
    assert_eq!(selected.mappings[0].axes()[0].step, 2);
    assert_eq!(selected.mappings[0].axes()[1].logical_extent, 3);
    let execution = execution::prepare_with_mappings(
        &source,
        Config {
            tile_piece: Some(2),
            per_item: 1,
            sg_per_tg: 1,
            max_threads_per_threadgroup: 64,
            max_threadgroup_bytes: 4096,
            ..Default::default()
        },
        Some(&selected.mappings),
        &mut |_, _| Ok(LoadMode::Materialize),
        &mut |_| Ok(TilePlacement::GroupShared),
        &mut |d| Ok(d.diagnostic()),
        &mut |d| Ok(d.new_slot),
    )
    .unwrap();
    let execution = GroupFamily::derive(execution)
        .unwrap()
        .select_launches(&selected.groups)
        .unwrap();
    assert_eq!(
        execution.phases()[0].dispatch,
        binding.phases[0]
            .main
            .reconstruct(witness.values())
            .unwrap()
    );
    let mut changed = witness.values().to_vec();
    changed[binding.partition.id().0] = 4;
    assert!(model.validate_assignment(&changed).unwrap().infeasible);
}

#[test]
fn joint_storage_constraints_match_concrete_ownership_for_every_local_pair() {
    let execution = execution::prepare_with_choices(
        &source(),
        Config {
            loads: seismic_realization::LoadStrategy::Materialize,
            ..Default::default()
        },
        &mut |_, _| Ok(LoadMode::Materialize),
        &mut |_| Ok(TilePlacement::GroupShared),
        &mut |d| Ok(d.diagnostic()),
    )
    .unwrap();
    let family = Arc::new(
        StorageFamily::derive(&execution.function().vars, &execution.function().body, &[]).unwrap(),
    );
    assert_eq!(family.decisions().len(), 2);
    for left in 0..family.decisions()[0].alternatives.len() {
        for right in 0..family.decisions()[1].alternatives.len() {
            let assignments = [
                (family.decisions()[0].variable, left),
                (family.decisions()[1].variable, right),
            ];
            let concrete = family.select(&mut |decision| {
                let ordinal = assignments
                    .iter()
                    .find(|(variable, _)| *variable == decision.variable)
                    .unwrap()
                    .1;
                let original = family
                    .decisions()
                    .iter()
                    .find(|original| original.variable == decision.variable)
                    .unwrap();
                Ok(original.alternatives[ordinal].clone())
            });
            let mut builder = ModelBuilder::new();
            let mut algebra = Symbolic::new(&mut builder, "dispatch");
            let lanes = algebra.constant(32).unwrap();
            let items = algebra.constant(2).unwrap();
            let binding = family
                .append(&mut builder, "storage", lanes, items)
                .unwrap();
            for (variable, ordinal) in assignments {
                builder.constraint(Constraint::InDomain {
                    variable: binding.placements[&variable].ordinal,
                    domain: Domain::singleton(ordinal as i64),
                });
            }
            let mut search =
                Search::new(Arc::new(builder.build().unwrap()), Options::default()).unwrap();
            match (concrete, search.advance(Limits::default()).unwrap()) {
                (Ok(expected), Outcome::Optimal(witness)) => {
                    let actual = binding
                        .reconstruct(witness.values(), &GroupDispatch::new(3, 32, 2).unwrap())
                        .unwrap();
                    assert_eq!(actual, expected);
                }
                (Err(_), Outcome::Infeasible) => {}
                pair => panic!(
                    "storage family and concrete ownership disagree: {}",
                    match pair.0 {
                        Ok(_) => "expected feasible",
                        Err(_) => "expected infeasible",
                    }
                ),
            }
        }
    }
}
