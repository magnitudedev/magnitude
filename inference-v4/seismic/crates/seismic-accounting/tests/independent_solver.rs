use magnitude_solver::{Limits, Options};
use seismic_accounting::schedule::{
    self,
    independent::{IndependentOutcome, IndependentSearch},
};
use std::sync::Arc;

fn operation(index: usize, latency: u64, resource: usize) -> schedule::Operation {
    schedule::Operation {
        name: format!("operation-{index}"),
        latency,
        predecessors: Vec::new(),
        start_predecessors: Vec::new(),
        reservations: vec![schedule::Reservation {
            resource,
            offset: 0,
            duration: latency,
            units: 1,
        }],
    }
}

fn model(operations: Vec<schedule::Operation>) -> schedule::Model {
    schedule::Model {
        relationship: seismic_accounting::authority::ModelRelationship::hypothetical_execution(),
        identity: "independent solver differential fixture".into(),
        timebase: schedule::Timebase {
            seconds_numerator: 1,
            seconds_denominator: 1,
        },
        resources: vec![
            schedule::Resource {
                name: "a".into(),
                capacity: 1,
                unit: schedule::CapacityUnit::Slots,
            },
            schedule::Resource {
                name: "b".into(),
                capacity: 1,
                unit: schedule::CapacityUnit::Slots,
            },
        ],
        operations,
        lifetimes: Vec::new(),
        static_orders: Vec::new(),
        unmapped: Vec::new(),
    }
}

fn serial(model: &schedule::Model) -> schedule::Schedule {
    let mut completion = 0;
    let starts = model
        .operations
        .iter()
        .map(|operation| {
            let start = completion;
            completion += operation.latency;
            start
        })
        .collect();
    schedule::Schedule { starts, completion }
}

fn solve(model: schedule::Model) -> schedule::Solution {
    let upper = serial(&model);
    model.check_schedule(&upper).unwrap();
    let mut search = IndependentSearch::new(Arc::new(model), upper, Options::default()).unwrap();
    match search
        .advance(Limits {
            work: 2_000_000,
            ..Limits::default()
        })
        .unwrap()
    {
        IndependentOutcome::Optimal(solution) => solution,
        IndependentOutcome::Infeasible => panic!("validated upper must remain feasible"),
        IndependentOutcome::Incomplete { stats, .. } => {
            panic!("tiny adapter case incomplete after {} work", stats.work)
        }
    }
}

fn exhaustive(model: &schedule::Model, horizon: u64) -> u64 {
    let mut starts = vec![0; model.operations.len()];
    let mut best = u64::MAX;
    loop {
        let completion = model
            .operations
            .iter()
            .zip(&starts)
            .map(|(op, start)| start + op.latency)
            .max()
            .unwrap_or(0);
        if completion <= horizon && completion < best {
            let schedule = schedule::Schedule {
                starts: starts.clone(),
                completion,
            };
            if model.check_schedule(&schedule).is_ok() {
                best = completion;
            }
        }
        let mut cursor = 0;
        while cursor < starts.len() && starts[cursor] == horizon {
            starts[cursor] = 0;
            cursor += 1;
        }
        if cursor == starts.len() {
            return best;
        }
        starts[cursor] += 1;
    }
}

#[test]
fn translated_schedules_agree_with_public_search_and_bounded_exhaustive_oracle() {
    for seed in 0..6u64 {
        let mut model = model(vec![
            operation(0, 1 + seed % 2, 0),
            operation(1, 1 + (seed / 2) % 2, 1),
            operation(2, 1, seed as usize % 2),
        ]);
        if seed % 2 == 0 {
            model.operations[2].predecessors.push(0);
        }
        if seed % 3 == 0 {
            model.operations[1].start_predecessors.push(0);
        }
        let expected = exhaustive(&model, serial(&model).completion);
        let old = model.solve(100_000).unwrap();
        assert!(old.is_optimal());
        assert_eq!(old.schedule().completion, expected);
        let new = solve(model.clone());
        assert!(new.is_optimal());
        assert_eq!(new.schedule().completion, expected);
        model.check_schedule(new.schedule()).unwrap();
    }
}

#[test]
fn adapter_keeps_reservation_offsets_and_event_lifetimes() {
    let mut model = model(vec![
        operation(0, 2, 0),
        operation(1, 2, 0),
        operation(2, 2, 1),
    ]);
    model.operations[0].reservations[0].offset = 1;
    model.operations[0].reservations[0].duration = 1;
    model.operations[1].predecessors = vec![0];
    model.lifetimes.push(schedule::Lifetime {
        resource: 1,
        units: 1,
        begin: schedule::Event {
            operation: 0,
            point: schedule::Point::Start,
        },
        end: schedule::Event {
            operation: 1,
            point: schedule::Point::Completion,
        },
    });
    let expected = exhaustive(&model, 6);
    let actual = solve(model.clone());
    assert_eq!(actual.schedule().completion, expected);
    model.check_schedule(actual.schedule()).unwrap();
}

#[test]
fn one_static_instruction_order_is_shared_by_all_visits() {
    use cranelift_codegen::ir::{Block, Inst};
    let mut model = model(vec![
        operation(0, 1, 0),
        operation(1, 1, 1),
        operation(2, 1, 0),
        operation(3, 1, 1),
    ]);
    model.operations[1].predecessors.push(0);
    model.operations[2].predecessors.push(1);
    model
        .static_orders
        .push(schedule::static_order::Constraint {
            block: Block::from_u32(0),
            instructions: vec![Inst::from_u32(0), Inst::from_u32(1)],
            predecessors: Vec::new(),
            visits: vec![
                schedule::static_order::Visit {
                    invocation: 0,
                    occurrence: 0,
                    roots: vec![vec![0], vec![1]],
                },
                schedule::static_order::Visit {
                    invocation: 0,
                    occurrence: 1,
                    roots: vec![vec![2], vec![3]],
                },
            ],
        });
    let expected = exhaustive(&model, 4);
    let actual = solve(model.clone());
    assert_eq!(actual.schedule().completion, expected);
    assert_eq!(
        schedule::static_order::orders(&model, actual.schedule())
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn pause_preserves_original_feasible_upper_without_claiming_optimality() {
    let model = Arc::new(model(vec![operation(0, 2, 0), operation(1, 2, 1)]));
    let upper = serial(&model);
    let mut search =
        IndependentSearch::new(model.clone(), upper.clone(), Options::default()).unwrap();
    let IndependentOutcome::Incomplete {
        incumbent,
        lower_bound,
        ..
    } = search
        .advance(Limits {
            work: 0,
            ..Limits::default()
        })
        .unwrap()
    else {
        panic!("zero work must remain incomplete");
    };
    let incumbent = incumbent.expect("original upper retained");
    assert_eq!(incumbent.schedule(), &upper);
    assert!(lower_bound <= 2);
    assert!(!incumbent.is_optimal());
    model.check_schedule(incumbent.schedule()).unwrap();
    let IndependentOutcome::Optimal(solution) = search.advance(Limits::default()).unwrap() else {
        panic!("tiny schedule should complete");
    };
    assert_eq!(solution.schedule().completion, 2);
}

#[test]
fn invalid_upper_and_incomplete_mapping_are_not_silently_accepted() {
    let model = model(vec![operation(0, 1, 0)]);
    let invalid = schedule::Schedule {
        starts: vec![0],
        completion: 0,
    };
    assert!(IndependentSearch::new(Arc::new(model.clone()), invalid, Options::default()).is_err());
    let mut missing = model;
    missing.unmapped.push("unmapped operation".into());
    assert!(
        IndependentSearch::new(
            Arc::new(missing.clone()),
            serial(&missing),
            Options::default()
        )
        .is_err()
    );
}

#[test]
fn relaxation_authority_is_not_promoted_by_exact_solving() {
    let mut model = model(vec![operation(0, 1, 0)]);
    model.relationship = seismic_accounting::authority::ModelRelationship::OptimisticRelaxation;
    let solution = solve(model);
    assert!(solution.is_optimal());
    assert!(
        solution
            .model()
            .check_execution_upper(solution.schedule())
            .is_err()
    );
}

#[test]
fn public_path_uses_capacity_incompatibility_for_the_nine_tick_case() {
    let mut m = model(vec![
        operation(0, 2, 0),
        operation(1, 3, 0),
        operation(2, 4, 0),
    ]);
    m.resources[0].capacity = 3;
    for operation in &mut m.operations {
        operation.reservations[0].units = 2;
    }
    let result = m.solve(10_000).unwrap();
    assert!(result.is_optimal());
    assert_eq!(result.schedule().completion, 9);
    m.check_schedule(result.schedule()).unwrap();
}

#[test]
fn solver_integer_width_is_an_analysis_obligation_not_infeasibility() {
    let mut m = model(vec![operation(0, i64::MAX as u64 + 1, 0)]);
    m.operations[0].reservations.clear();
    let mut search = m.start_search().unwrap();
    assert!(matches!(
        search.advance(100).unwrap(),
        schedule::SearchOutcome::Incomplete { .. }
    ));
    assert_eq!(
        search.obligations()[0].kind,
        magnitude_solver::model::ObligationKind::Analysis
    );
    assert!(search.obligations()[0].reason.contains("i64"));
}

#[test]
fn earliest_seed_tightens_horizon_only_after_full_resource_validation() {
    for (second_resource, expected_horizon) in [(1, 2), (0, 4)] {
        let original = model(vec![operation(0, 2, 0), operation(1, 2, second_resource)]);
        let mut search =
            IndependentSearch::from_model(Arc::new(original), Options::default()).unwrap();
        let completion = search
            .translated_model()
            .variables()
            .iter()
            .find(|v| v.name == "schedule.completion")
            .unwrap();
        assert_eq!(completion.domain.max(), Some(expected_horizon));
        let IndependentOutcome::Optimal(solution) = search.advance(Limits::default()).unwrap()
        else {
            panic!("tiny seed correspondence must complete")
        };
        assert_eq!(solution.schedule().completion, expected_horizon as u64);
        solution
            .model()
            .check_schedule(solution.schedule())
            .unwrap();
    }
}

#[test]
fn mandatory_static_precedence_does_not_create_redundant_pair_choices() {
    use cranelift_codegen::ir::{Block, Inst};
    for mask in 0..8u8 {
        let edges = [(0, 1), (0, 2), (1, 2)]
            .into_iter()
            .enumerate()
            .filter_map(|(bit, edge)| (mask & (1 << bit) != 0).then_some(edge))
            .collect::<Vec<_>>();
        let mut original = model(vec![
            operation(0, 1, 0),
            operation(1, 2, 1),
            operation(2, 1, 0),
        ]);
        original
            .static_orders
            .push(schedule::static_order::Constraint {
                block: Block::from_u32(0),
                instructions: (0..3).map(Inst::from_u32).collect(),
                predecessors: edges,
                visits: vec![schedule::static_order::Visit {
                    invocation: 0,
                    occurrence: 0,
                    roots: vec![vec![0], vec![1], vec![2]],
                }],
            });
        let expected = exhaustive(&original, 4);
        let mut search =
            IndependentSearch::from_model(Arc::new(original.clone()), Options::default()).unwrap();
        let known_02 = mask & 2 != 0 || (mask & 1 != 0 && mask & 4 != 0);
        let ordered_pairs =
            usize::from(mask & 1 != 0) + usize::from(mask & 4 != 0) + usize::from(known_02);
        assert_eq!(
            search
                .translated_model()
                .variables()
                .iter()
                .filter(|v| v.name.contains("_pair_"))
                .count(),
            3 - ordered_pairs
        );
        let IndependentOutcome::Optimal(solution) = search.advance(Limits::default()).unwrap()
        else {
            panic!("small static-order family must complete")
        };
        assert_eq!(solution.schedule().completion, expected);
        original.check_schedule(solution.schedule()).unwrap();
    }
}

#[test]
fn guarded_fragments_share_resources_and_reconstruct_only_the_selected_execution() {
    use magnitude_solver::{
        Outcome, Search,
        model::{Constraint, Cost, Domain, LinearTerm, ModelBuilder},
    };
    use schedule::{independent::Fragment, symbolic::Encoding};
    let mut builder = ModelBuilder::new();
    let short = builder.variable("short", Domain::boolean());
    let long = builder.variable("long", Domain::boolean());
    builder.constraint(Constraint::ExactlyOne {
        variables: vec![short, long],
    });
    let mut short_model = model(vec![operation(0, 1, 0)]);
    short_model.resources[0].capacity = 2;
    short_model.operations[0].reservations[0].units = 2;
    let mut long_model = short_model.clone();
    long_model.operations = vec![operation(0, 2, 0)];
    let neighbor_model = long_model.clone();
    let mut encoding = Encoding::new(&mut builder, "family", &short_model.resources, 5).unwrap();
    let short_fragment = Fragment::append(
        &mut builder,
        &mut encoding,
        Arc::new(short_model),
        Some(short),
    )
    .unwrap();
    let long_fragment = Fragment::append(
        &mut builder,
        &mut encoding,
        Arc::new(long_model),
        Some(long),
    )
    .unwrap();
    let neighbor =
        Fragment::append(&mut builder, &mut encoding, Arc::new(neighbor_model), None).unwrap();
    let completion = encoding.finish(&mut builder).unwrap();
    builder.cost(Cost::Linear {
        constant: 0,
        terms: vec![LinearTerm::new(completion, 1)],
    });
    let translated = Arc::new(builder.build().unwrap());
    let mut search = Search::new(translated.clone(), Options::default()).unwrap();
    let Outcome::Optimal(solution) = search.advance(Limits::default()).unwrap() else {
        panic!("joint family must complete")
    };
    assert_eq!(solution.cost(), 2);
    assert!(
        short_fragment
            .reconstruct(solution.values())
            .unwrap()
            .is_none()
    );
    assert_eq!(
        long_fragment
            .reconstruct(solution.values())
            .unwrap()
            .unwrap()
            .completion,
        2
    );
    assert_eq!(
        neighbor
            .reconstruct(solution.values())
            .unwrap()
            .unwrap()
            .completion,
        2
    );
    // Private inactive assignments remain canonical, rather than introducing
    // arbitrary schedule branches for implementations that were not selected.
    let mut changed = solution.values().to_vec();
    changed[short_fragment.starts()[0].0] = 1;
    assert!(translated.validate_assignment(&changed).unwrap().infeasible);
}

#[test]
fn fragment_guards_cover_partial_services_lifetimes_and_static_orders() {
    use magnitude_solver::{
        Outcome, Search,
        model::{Cost, Domain, LinearTerm, ModelBuilder},
    };
    use schedule::{independent::Fragment, symbolic::Encoding};
    for present in [0, 1] {
        let mut original = model(vec![operation(0, 3, 0), operation(1, 2, 0)]);
        original.operations[0].reservations[0].offset = 1;
        original.operations[0].reservations[0].duration = 1;
        original.operations[1].predecessors = vec![0];
        original
            .static_orders
            .push(schedule::static_order::Constraint {
                block: cranelift_codegen::ir::Block::from_u32(0),
                instructions: vec![
                    cranelift_codegen::ir::Inst::from_u32(0),
                    cranelift_codegen::ir::Inst::from_u32(1),
                ],
                predecessors: Vec::new(),
                visits: vec![schedule::static_order::Visit {
                    invocation: 0,
                    occurrence: 0,
                    roots: vec![vec![0], vec![1]],
                }],
            });
        original.lifetimes.push(schedule::Lifetime {
            resource: 1,
            units: 1,
            begin: schedule::Event {
                operation: 0,
                point: schedule::Point::Start,
            },
            end: schedule::Event {
                operation: 1,
                point: schedule::Point::Completion,
            },
        });
        let mut builder = ModelBuilder::new();
        let presence = builder.variable("present", Domain::singleton(present));
        let mut encoding = Encoding::new(&mut builder, "guarded", &original.resources, 5).unwrap();
        let fragment = Fragment::append(
            &mut builder,
            &mut encoding,
            Arc::new(original),
            Some(presence),
        )
        .unwrap();
        let completion = encoding.finish(&mut builder).unwrap();
        builder.cost(Cost::Linear {
            constant: 0,
            terms: vec![LinearTerm::new(completion, 1)],
        });
        let mut search =
            Search::new(Arc::new(builder.build().unwrap()), Options::default()).unwrap();
        let Outcome::Optimal(solution) = search.advance(Limits::default()).unwrap() else {
            panic!("guarded schedule must complete")
        };
        assert_eq!(solution.cost(), if present == 0 { 0 } else { 5 });
        let reconstructed = fragment.reconstruct(solution.values()).unwrap();
        assert_eq!(reconstructed.is_some(), present == 1);
        if let Some(schedule) = reconstructed {
            assert_eq!(schedule.starts, vec![0, 3]);
        }
    }
}

#[test]
fn an_inactive_fragment_may_exceed_the_improvement_horizon() {
    use magnitude_solver::{
        Outcome, Search,
        model::{Constraint, Cost, Domain, LinearTerm, ModelBuilder},
    };
    use schedule::{independent::Fragment, symbolic::Encoding};
    let mut builder = ModelBuilder::new();
    let too_long = builder.variable("too long", Domain::boolean());
    let fits = builder.variable("fits", Domain::boolean());
    builder.constraint(Constraint::ExactlyOne {
        variables: vec![too_long, fits],
    });
    let original = model(vec![operation(0, 7, 0)]);
    let mut encoding = Encoding::new(&mut builder, "improvement", &original.resources, 2).unwrap();
    let excluded = Fragment::append(
        &mut builder,
        &mut encoding,
        Arc::new(original),
        Some(too_long),
    )
    .unwrap();
    let admitted = Fragment::append(
        &mut builder,
        &mut encoding,
        Arc::new(model(vec![operation(0, 2, 0)])),
        Some(fits),
    )
    .unwrap();
    let completion = encoding.finish(&mut builder).unwrap();
    builder.cost(Cost::Linear {
        constant: 0,
        terms: vec![LinearTerm::new(completion, 1)],
    });
    let mut search = Search::new(Arc::new(builder.build().unwrap()), Options::default()).unwrap();
    let Outcome::Optimal(solution) = search.advance(Limits::default()).unwrap() else {
        panic!("guarded horizon must complete")
    };
    assert_eq!(solution.cost(), 2);
    assert!(excluded.reconstruct(solution.values()).unwrap().is_none());
    assert!(admitted.reconstruct(solution.values()).unwrap().is_some());
}
