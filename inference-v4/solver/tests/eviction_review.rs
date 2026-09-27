//! Adversarial cache compaction checks. Tiny memory limits deliberately trigger
//! compaction; subsequent unrestricted slices must preserve the original problem.
use magnitude_solver::{
    model::{Constraint, Cost, Fragment, LinearTerm, Literal, Model, ObligationKind},
    result::{FeasibleSolution, StopReason},
    scheduling::{Activity, Demand, Interval, Reservation, SchedulingConstraint},
    Domain, Limits, ModelBuilder, Options, Outcome, Policy, Search,
};
use std::sync::Arc;

fn checked_witness(model: &Model, witness: &FeasibleSolution) {
    let assessment = model.validate_assignment(witness.values()).unwrap();
    assert!(!assessment.infeasible);
    assert_eq!(assessment.exact_cost, Some(witness.cost()));
}

// The oracle deliberately uses plain integer arithmetic, without model factor
// evaluation, decomposition, search bounds, or the production memo.
fn chain_cost(seed: i64, index: usize, value: i64) -> u64 {
    ((seed * 11 + index as i64 * 7 + value * 13) % 19) as u64
}

fn chains(seed: i64, impossible: bool) -> (Arc<Model>, Option<u64>) {
    let mut b = ModelBuilder::new();
    let vars: Vec<_> = (0..6)
        .map(|i| b.variable(format!("choice {i}"), Domain::interval(0, 2).unwrap()))
        .collect();
    for (i, &v) in vars.iter().enumerate() {
        b.cost(Cost::Table {
            variables: vec![v],
            entries: (0..3).map(|x| (vec![x], chain_cost(seed, i, x))).collect(),
        });
    }
    for group in vars.chunks(3) {
        for pair in group.windows(2) {
            b.constraint(Constraint::NotEqual {
                left: pair[0],
                right: pair[1],
            });
        }
        if impossible {
            // A two-color odd cycle is arc-consistent before branching, but
            // globally infeasible. This exercises proved losing OR branches.
            for &variable in group {
                b.constraint(Constraint::InDomain {
                    variable,
                    domain: Domain::boolean(),
                });
            }
            b.constraint(Constraint::NotEqual {
                left: group[0],
                right: group[2],
            });
        }
    }
    let expected = (0..729)
        .filter_map(|mut encoding| {
            let values: Vec<_> = (0..6)
                .map(|_| {
                    let value = encoding % 3;
                    encoding /= 3;
                    value
                })
                .collect();
            let feasible = values.chunks(3).all(|group| {
                group[0] != group[1]
                    && group[1] != group[2]
                    && (!impossible || (group.iter().all(|&v| v < 2) && group[0] != group[2]))
            });
            feasible.then(|| {
                values
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| chain_cost(seed, i, v))
                    .sum()
            })
        })
        .min();
    (Arc::new(b.build().unwrap()), expected)
}

fn interrupted_with_eviction(model: Arc<Model>, expected: Option<u64>, options: Options) -> u64 {
    let mut search = Search::new(model.clone(), options).unwrap();
    let mut last_lower = 0;
    let mut last_incumbent = None;
    for slice in 0..50_000 {
        let force_compaction = slice % 2 == 1;
        let before_work = search.stats().work;
        let outcome = search
            .advance(Limits {
                work: if force_compaction { 1 } else { 3 },
                memory_bytes: force_compaction.then_some(1),
                ..Default::default()
            })
            .unwrap();
        match outcome {
            Outcome::Incomplete(progress) => {
                if force_compaction {
                    assert_eq!(progress.reason, StopReason::Memory);
                    assert_eq!(search.stats().work, before_work);
                }
                assert!(
                    progress.lower_bound >= last_lower,
                    "eviction lost a proved bound"
                );
                last_lower = progress.lower_bound;
                if let Some(optimum) = expected {
                    assert!(progress.lower_bound <= optimum);
                }
                if let Some(witness) = progress.incumbent {
                    checked_witness(&model, &witness);
                    assert!(witness.cost() >= expected.expect("infeasible model has a witness"));
                    if let Some(previous) = last_incumbent {
                        assert!(witness.cost() <= previous, "eviction lost an incumbent");
                    }
                    last_incumbent = Some(witness.cost());
                } else {
                    assert!(last_incumbent.is_none(), "eviction forgot the incumbent");
                }
            }
            Outcome::Optimal(solution) => {
                assert_eq!(Some(solution.cost()), expected);
                checked_witness(&model, solution.feasible());
                return search.stats().evictions;
            }
            Outcome::Infeasible => {
                assert_eq!(expected, None);
                return search.stats().evictions;
            }
        }
    }
    panic!("tiny finite search did not resume to completion after compaction");
}

#[test]
fn repeated_compaction_preserves_bounds_witnesses_and_finite_coverage() {
    let mut evictions = 0;
    for seed in [0, 3] {
        for impossible in [false, true] {
            let (model, expected) = chains(seed, impossible);
            for policy in [Policy::DepthFirst, Policy::BestFirst] {
                for memoize in [false, true] {
                    for decompose in [false, true] {
                        evictions += interrupted_with_eviction(
                            model.clone(),
                            expected,
                            Options {
                                policy,
                                memoize,
                                decompose,
                                bounds: false,
                                ..Default::default()
                            },
                        );
                    }
                }
            }
        }
    }
    assert!(
        evictions > 0,
        "fixture did not actually reclaim cached regions"
    );
}

#[test]
fn lazy_guarded_occurrences_survive_node_renumbering_and_reconstruction() {
    let mut leaf = ModelBuilder::new();
    let boundary = leaf.variable("boundary", Domain::boolean());
    let left = leaf.variable("left", Domain::interval(0, 2).unwrap());
    let right = leaf.variable("right", Domain::interval(0, 2).unwrap());
    leaf.constraint(Constraint::NotEqual { left, right });
    for (index, variable) in [left, right].into_iter().enumerate() {
        leaf.cost(Cost::Table {
            variables: vec![boundary, variable],
            entries: (0..2)
                .flat_map(|boundary| {
                    (0..3).map(move |x| (vec![boundary, x], chain_cost(boundary, index, x)))
                })
                .collect(),
        });
    }
    let leaf = Fragment::new(leaf.build().unwrap(), vec![boundary]).unwrap();
    let mut wrapper = ModelBuilder::new();
    let boundary = wrapper.variable("boundary", Domain::boolean());
    wrapper
        .instantiate_lazy("first body", &leaf, &[boundary], vec![])
        .unwrap();
    wrapper
        .instantiate_lazy("second body", &leaf, &[boundary], vec![])
        .unwrap();
    let wrapper = Fragment::new(wrapper.build().unwrap(), vec![boundary]).unwrap();
    for active in [0, 1] {
        let mut b = ModelBuilder::new();
        let gate = b.variable("gate", Domain::singleton(active));
        let format = b.variable("format", Domain::boolean());
        b.instantiate_lazy("outer", &wrapper, &[format], vec![Literal::new(gate, 1)])
            .unwrap();
        let expected = if active == 0 {
            0
        } else {
            2 * (0..2)
                .flat_map(|f| {
                    (0..3).flat_map(move |a| {
                        (0..3).filter_map(move |c| {
                            (a != c).then(|| chain_cost(f, 0, a) + chain_cost(f, 1, c))
                        })
                    })
                })
                .min()
                .unwrap()
        };
        let model = Arc::new(b.build().unwrap());
        for policy in [Policy::DepthFirst, Policy::BestFirst] {
            interrupted_with_eviction(
                model.clone(),
                Some(expected),
                Options {
                    policy,
                    bounds: false,
                    ..Default::default()
                },
            );
        }
    }
}

#[test]
fn compaction_cannot_discard_unresolved_coverage_or_turn_it_into_infeasibility() {
    for policy in [Policy::DepthFirst, Policy::BestFirst] {
        let (known, _) = chains(1, false);
        let mut b = ModelBuilder::new();
        let choice = b.variable("known or unresolved", Domain::boolean());
        b.guarded_cost(vec![Literal::new(choice, 1)], Cost::Constant(1_000));
        b.obligation(
            vec![Literal::new(choice, 0)],
            ObligationKind::Analysis,
            "unresolved region",
        );
        let fragment = Fragment::new((*known).clone(), vec![]).unwrap();
        b.instantiate_lazy("known work", &fragment, &[], vec![])
            .unwrap();
        let model = Arc::new(b.build().unwrap());
        let mut search = Search::new(
            model.clone(),
            Options {
                policy,
                ..Default::default()
            },
        )
        .unwrap();
        let Outcome::Incomplete(before) = search.advance(Limits::default()).unwrap() else {
            panic!("unresolved competitive region cannot be completed")
        };
        assert!(matches!(before.reason, StopReason::Coverage(_)));
        let witness = before
            .incumbent
            .as_ref()
            .expect("known branch remains feasible");
        checked_witness(&model, witness);
        let Outcome::Incomplete(pressure) = search
            .advance(Limits {
                memory_bytes: Some(1),
                ..Default::default()
            })
            .unwrap()
        else {
            panic!("memory pressure cannot resolve a coverage obligation")
        };
        assert_eq!(pressure.reason, StopReason::Memory);
        assert!(search.stats().evictions > 0);
        let Outcome::Incomplete(after) = search.advance(Limits::default()).unwrap() else {
            panic!("compaction discarded required coverage")
        };
        assert_eq!(after.reason, before.reason);
        assert_eq!(after.lower_bound, before.lower_bound);
        assert_eq!(after.incumbent.as_ref().unwrap().cost(), witness.cost());
        checked_witness(&model, after.incumbent.as_ref().unwrap());
    }
}

#[test]
fn schedule_resource_context_survives_compaction() {
    let mut b = ModelBuilder::new();
    let mut reservations = Vec::new();
    let completion = b.variable("completion", Domain::interval(0, 9).unwrap());
    for index in 0..2 {
        let start = b.variable(format!("start {index}"), Domain::interval(0, 7).unwrap());
        let end = b.variable(format!("end {index}"), Domain::interval(0, 9).unwrap());
        let duration = b.variable(format!("duration {index}"), Domain::singleton(2));
        b.constraint(Constraint::Schedule(SchedulingConstraint::Activity(
            Activity {
                start,
                duration,
                end,
                presence: None,
            },
        )));
        b.constraint(Constraint::LinearLe {
            terms: vec![LinearTerm::new(end, 1), LinearTerm::new(completion, -1)],
            rhs: 0,
        });
        reservations.push(Reservation {
            interval: Interval::mandatory(start, end),
            demand: Demand::Constant(1),
        });
    }
    let busy_start = b.variable("outside busy start", Domain::singleton(1));
    let busy_end = b.variable("outside busy end", Domain::singleton(3));
    reservations.push(Reservation {
        interval: Interval::mandatory(busy_start, busy_end),
        demand: Demand::Constant(1),
    });
    b.constraint(Constraint::Schedule(SchedulingConstraint::Cumulative {
        capacity: 1,
        reservations,
    }));
    b.cost(Cost::Linear {
        constant: 0,
        terms: vec![LinearTerm::new(completion, 1)],
    });
    let model = Arc::new(b.build().unwrap());
    // Neither duration-2 activity fits before [1, 3). Both must execute after
    // time 3, so completion is at least 7; [3, 5), [5, 7) attain that bound.
    for policy in [Policy::DepthFirst, Policy::BestFirst] {
        interrupted_with_eviction(
            model.clone(),
            Some(7),
            Options {
                policy,
                bounds: false,
                ..Default::default()
            },
        );
    }
}
