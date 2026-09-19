//! Independent small-context checks of composition, cache boundaries and resumes.
use magnitude_solver::model::{Constraint, Cost, Fragment, LinearTerm, Literal, Model};
use magnitude_solver::scheduling::{Interval, SchedulingConstraint};
use magnitude_solver::{Domain, Limits, ModelBuilder, Options, Outcome, Policy, Search};
use std::sync::Arc;

fn complete(model: Model, options: Options, quantum: u64) -> (u64, Vec<i64>) {
    let mut search = Search::new(Arc::new(model), options).unwrap();
    for _ in 0..100_000 {
        match search
            .advance(Limits {
                work: quantum,
                ..Default::default()
            })
            .unwrap()
        {
            Outcome::Optimal(s) => return (s.cost(), s.values().to_vec()),
            Outcome::Incomplete(_) => (),
            Outcome::Infeasible => panic!("fixture has a known feasible assignment"),
        }
    }
    panic!("finite tiny fixture failed to finish");
}
fn contexts(format: Option<i64>, capacity: Option<i64>) -> Model {
    let mut b = ModelBuilder::new();
    let f = b.variable(
        "boundary representation",
        format.map_or_else(|| Domain::interval(0, 2).unwrap(), Domain::singleton),
    );
    let cap = b.variable(
        "shared resource capacity",
        capacity.map_or_else(|| Domain::interval(2, 4).unwrap(), Domain::singleton),
    );
    let left = b.variable("left private mode", Domain::interval(0, 2).unwrap());
    let right = b.variable("right private mode", Domain::interval(0, 2).unwrap());
    b.cost(Cost::Table {
        variables: vec![f, left],
        entries: (0..3)
            .flat_map(|f| (0..3).map(move |x| (vec![f, x], left_cost(f, x))))
            .collect(),
    });
    b.cost(Cost::Table {
        variables: vec![f, right],
        entries: (0..3)
            .flat_map(|f| (0..3).map(move |x| (vec![f, x], right_cost(f, x))))
            .collect(),
    });
    b.cost(Cost::Linear {
        constant: 0,
        terms: vec![LinearTerm::new(cap, 2)],
    });
    b.constraint(Constraint::LinearLe {
        terms: vec![
            LinearTerm::new(left, 1),
            LinearTerm::new(right, 1),
            LinearTerm::new(cap, -1),
        ],
        rhs: -2,
    });
    b.build().unwrap()
}
fn left_cost(format: i64, mode: i64) -> u64 {
    ((format * 7 + mode * 5 + 3) % 13) as u64
}
fn right_cost(format: i64, mode: i64) -> u64 {
    ((format * 3 + mode * 11 + 8) % 17) as u64
}
fn reference(format: i64, capacity: i64) -> u64 {
    (0..3)
        .flat_map(|a| (0..3).map(move |b| (a, b)))
        .filter(|(a, b)| a + b + 2 <= capacity)
        .map(|(a, b)| left_cost(format, a) + right_cost(format, b) + 2 * capacity as u64)
        .min()
        .unwrap()
}
#[test]
fn every_boundary_context_matches_joint_enumeration() {
    for format in 0..3 {
        for capacity in 2..=4 {
            let expected = reference(format, capacity);
            for policy in [Policy::DepthFirst, Policy::BestFirst] {
                for decompose in [false, true] {
                    let actual = complete(
                        contexts(Some(format), Some(capacity)),
                        Options {
                            policy,
                            decompose,
                            ..Default::default()
                        },
                        3,
                    );
                    assert_eq!(actual.0, expected, "format={format}, capacity={capacity}");
                }
            }
        }
    }
    let expected = (0..3)
        .flat_map(|f| (2..=4).map(move |c| reference(f, c)))
        .min()
        .unwrap();
    for memoize in [false, true] {
        for policy in [Policy::DepthFirst, Policy::BestFirst] {
            assert_eq!(
                complete(
                    contexts(None, None),
                    Options {
                        policy,
                        memoize,
                        ..Default::default()
                    },
                    1
                )
                .0,
                expected
            );
        }
    }
}

#[test]
fn equal_duration_resource_profiles_are_not_interchangeable() {
    for busy_start in [0, 2] {
        for decompose in [false, true] {
            let mut b = ModelBuilder::new();
            let mode = b.variable("profile", Domain::boolean());
            let start = b.variable("private start", Domain::set([0, 2]));
            let end = b.variable("private end", Domain::set([2, 4]));
            let busy = b.variable("boundary busy start", Domain::singleton(busy_start));
            let released = b.variable("boundary busy end", Domain::singleton(busy_start + 2));
            b.constraint(Constraint::Table {
                variables: vec![mode, start, end],
                tuples: vec![vec![0, 0, 2], vec![1, 2, 4]],
            });
            b.constraint(Constraint::Schedule(SchedulingConstraint::NoOverlap {
                intervals: vec![
                    Interval {
                        start,
                        end,
                        presence: vec![],
                    },
                    Interval {
                        start: busy,
                        end: released,
                        presence: vec![],
                    },
                ],
            }));
            b.cost(Cost::Constant(2));
            let (_, values) = complete(
                b.build().unwrap(),
                Options {
                    decompose,
                    ..Default::default()
                },
                1,
            );
            assert_eq!(values[mode.0], if busy_start == 0 { 1 } else { 0 });
        }
    }
}

#[test]
fn lazy_and_eager_guarded_interfaces_preserve_all_contexts() {
    let mut definition = ModelBuilder::new();
    let boundary = definition.variable("boundary", Domain::interval(0, 2).unwrap());
    let private = definition.variable("private", Domain::boolean());
    definition.cost(Cost::Table {
        variables: vec![boundary, private],
        entries: (0..3)
            .flat_map(|f| (0..2).map(move |x| (vec![f, x], left_cost(f, x))))
            .collect(),
    });
    let fragment = Fragment::new(definition.build().unwrap(), vec![boundary]).unwrap();
    for format in 0..3 {
        for active in 0..=1 {
            let expected = if active == 0 {
                0
            } else {
                left_cost(format, 0).min(left_cost(format, 1))
            };
            for lazy in [false, true] {
                let mut b = ModelBuilder::new();
                let guard = b.variable("active", Domain::singleton(active));
                let value = b.variable("bound format", Domain::singleton(format));
                if lazy {
                    b.instantiate_lazy(
                        "realization",
                        &fragment,
                        &[value],
                        vec![Literal::new(guard, 1)],
                    )
                    .unwrap();
                } else {
                    b.instantiate(
                        "realization",
                        &fragment,
                        &[value],
                        vec![Literal::new(guard, 1)],
                    )
                    .unwrap();
                }
                assert_eq!(
                    complete(b.build().unwrap(), Options::default(), 1).0,
                    expected
                );
            }
        }
    }
}

#[test]
fn table_and_equality_fixpoint_makes_semantic_progress() {
    let mut b = ModelBuilder::new();
    let a = b.variable("table domain", Domain::set([0, 2, 4, 5, 6]));
    let z = b.variable(
        "intersection domain",
        Domain::interval(0, 6).unwrap().without(1).without(3),
    );
    let selector = b.variable("coupled selector", Domain::boolean());
    b.constraint(Constraint::Equal { left: a, right: z });
    b.constraint(Constraint::Table {
        variables: vec![a, selector],
        tuples: vec![vec![0, 0], vec![2, 1], vec![4, 0], vec![5, 1], vec![6, 0]],
    });
    b.cost(Cost::Linear {
        constant: 0,
        terms: vec![LinearTerm::new(z, 1)],
    });
    assert_eq!(complete(b.build().unwrap(), Options::default(), 1).0, 0);
}

#[cfg(feature = "serde")]
#[test]
fn malformed_serialized_models_never_reach_unchecked_search() {
    let model = contexts(Some(0), Some(2));
    let mut json = serde_json::to_value(model).unwrap();
    json["factors"][0]["kind"]["Cost"]["Table"]["entries"][0][0] = serde_json::json!([]);
    assert!(serde_json::from_value::<Model>(json).is_err());
    let malformed = serde_json::json!({"runs":[{"first":0,"last":1,"step":0}]});
    assert!(serde_json::from_value::<Domain>(malformed).is_err());
}

#[test]
fn nested_materialization_bounds_count_each_occurrence_once() {
    let mut leaf = ModelBuilder::new();
    let representation = leaf.variable("representation", Domain::boolean());
    leaf.cost(Cost::Table {
        variables: vec![representation],
        entries: vec![(vec![0], 3), (vec![1], 5)],
    });
    leaf.cost(Cost::Constant(2));
    let leaf = Fragment::new(leaf.build().unwrap(), vec![representation]).unwrap();
    let mut middle = ModelBuilder::new();
    let format = middle.variable("format", Domain::boolean());
    middle.cost(Cost::Constant(7));
    middle
        .instantiate_lazy("first", &leaf, &[format], vec![])
        .unwrap();
    middle
        .instantiate_lazy("second", &leaf, &[format], vec![])
        .unwrap();
    let middle = Fragment::new(middle.build().unwrap(), vec![format]).unwrap();
    for format in 0..=1 {
        for policy in [Policy::DepthFirst, Policy::BestFirst] {
            let mut b = ModelBuilder::new();
            let boundary = b.variable("boundary", Domain::singleton(format));
            b.instantiate_lazy("outer", &middle, &[boundary], vec![])
                .unwrap();
            let expected = 7 + 2 * (if format == 0 { 3 } else { 5 } + 2);
            let mut search = Search::new(
                Arc::new(b.build().unwrap()),
                Options {
                    policy,
                    ..Default::default()
                },
            )
            .unwrap();
            let mut previous = 0;
            let mut completed = false;
            for _ in 0..10_000 {
                match search
                    .advance(Limits {
                        work: 1,
                        ..Default::default()
                    })
                    .unwrap()
                {
                    Outcome::Incomplete(progress) => {
                        assert!(progress.lower_bound <= expected);
                        assert!(progress.lower_bound >= previous);
                        previous = progress.lower_bound;
                    }
                    Outcome::Optimal(solution) => {
                        assert_eq!(solution.cost(), expected);
                        completed = true;
                        break;
                    }
                    Outcome::Infeasible => panic!("known feasible nested fragments"),
                }
            }
            assert!(completed);
        }
    }
}

#[test]
fn partially_opened_losing_fragment_preserves_unopened_coverage() {
    let mut leaf = ModelBuilder::new();
    for _ in 0..50 {
        leaf.cost(Cost::Constant(1));
    }
    let leaf = Fragment::new(leaf.build().unwrap(), vec![]).unwrap();
    let mut middle = ModelBuilder::new();
    middle.cost(Cost::Constant(3));
    middle
        .instantiate_lazy("nested", &leaf, &[], vec![])
        .unwrap();
    let middle = Fragment::new(middle.build().unwrap(), vec![]).unwrap();
    for policy in [Policy::DepthFirst, Policy::BestFirst] {
        let mut b = ModelBuilder::new();
        let choice = b.variable("alternative", Domain::boolean());
        b.guarded_cost(vec![Literal::new(choice, 0)], Cost::Constant(10));
        b.instantiate_lazy("expensive", &middle, &[], vec![Literal::new(choice, 1)])
            .unwrap();
        let mut search = Search::new(
            Arc::new(b.build().unwrap()),
            Options {
                policy,
                ..Default::default()
            },
        )
        .unwrap();
        let mut completed = false;
        for _ in 0..10_000 {
            match search
                .advance(Limits {
                    work: 1,
                    ..Default::default()
                })
                .unwrap()
            {
                Outcome::Incomplete(progress) => assert!(progress.lower_bound <= 10),
                Outcome::Optimal(solution) => {
                    assert_eq!(solution.cost(), 10);
                    completed = true;
                    break;
                }
                Outcome::Infeasible => panic!("known feasible alternative"),
            }
        }
        assert!(completed);
        assert!(
            search.stats().materializations < 52,
            "loser should be excluded while its body remains partly unopened"
        );
    }
}
