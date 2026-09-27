use magnitude_solver::{
    composition::{Aggregation, CompositeOutcome, CompositeSearch, Composition},
    model::{Constraint, Cost, Literal, ObligationKind},
    result::StopReason,
    Domain, Error, Limits, Model, ModelBuilder, Options, Outcome, Search,
};
use std::{sync::Arc, time::Duration};

fn constant(cost: u64) -> Arc<Model> {
    let mut b = ModelBuilder::new();
    b.cost(Cost::Constant(cost));
    Arc::new(b.build().unwrap())
}

fn choices(costs: &[u64]) -> Arc<Model> {
    let mut b = ModelBuilder::new();
    let choice = b.variable(
        "choice",
        Domain::interval(0, costs.len() as i64 - 1).unwrap(),
    );
    b.cost(Cost::Table {
        variables: vec![choice],
        entries: costs
            .iter()
            .enumerate()
            .map(|(i, &cost)| (vec![i as i64], cost))
            .collect(),
    });
    Arc::new(b.build().unwrap())
}

fn search(aggregation: Aggregation, models: Vec<Arc<Model>>) -> CompositeSearch {
    CompositeSearch::new(
        Arc::new(Composition::new(aggregation, models).unwrap()),
        Options::default(),
    )
    .unwrap()
}

#[test]
fn independent_products_match_direct_cartesian_enumeration() {
    for a in [[9, 2, 7], [0, 8, 1], [6, 6, 6]] {
        for b in [[4, 1, 5], [7, 9, 2], [0, 0, 0]] {
            for aggregation in [Aggregation::Sum, Aggregation::Maximum] {
                let expected = a
                    .iter()
                    .flat_map(|&x| {
                        b.iter().map(move |&y| match aggregation {
                            Aggregation::Sum => x + y,
                            Aggregation::Maximum => x.max(y),
                        })
                    })
                    .min()
                    .unwrap();
                let mut solver = search(aggregation, vec![choices(&a), choices(&b)]);
                let CompositeOutcome::Optimal(solution) =
                    solver.advance(Limits::default()).unwrap()
                else {
                    panic!("finite product must complete");
                };
                assert_eq!(solution.cost(), expected);
                for (model, witness) in solution
                    .feasible()
                    .model()
                    .components()
                    .iter()
                    .zip(solution.feasible().components())
                {
                    assert_eq!(
                        model
                            .validate_assignment(witness.values())
                            .unwrap()
                            .exact_cost,
                        Some(witness.cost())
                    );
                }
            }
        }
    }
}

#[test]
fn repeated_model_identity_still_represents_distinct_occurrences() {
    let shared = choices(&[7, 3]);
    let mut solver = search(Aggregation::Sum, vec![shared.clone(), shared]);
    let CompositeOutcome::Optimal(solution) = solver.advance(Limits::default()).unwrap() else {
        panic!()
    };
    assert_eq!(solution.cost(), 6);
    assert_eq!(solution.feasible().components().len(), 2);
}

#[test]
fn empty_product_has_zero_objective_without_work() {
    for aggregation in [Aggregation::Sum, Aggregation::Maximum] {
        let mut solver = search(aggregation, vec![]);
        let CompositeOutcome::Optimal(solution) = solver
            .advance(Limits {
                work: 0,
                ..Limits::default()
            })
            .unwrap()
        else {
            panic!()
        };
        assert_eq!(solution.cost(), 0);
        assert_eq!(solver.work(), 0);
    }
}

#[test]
fn one_infeasible_component_invalidates_both_products() {
    let mut b = ModelBuilder::new();
    let x = b.variable("x", Domain::singleton(0));
    b.constraint(Constraint::NotEqual { left: x, right: x });
    let impossible = Arc::new(b.build().unwrap());
    for aggregation in [Aggregation::Sum, Aggregation::Maximum] {
        let mut solver = search(aggregation, vec![constant(100), impossible.clone()]);
        assert!(matches!(
            solver.advance(Limits::default()).unwrap(),
            CompositeOutcome::Infeasible
        ));
    }
}

#[test]
fn guarded_component_constraints_are_not_lost_during_composition() {
    let mut b = ModelBuilder::new();
    let x = b.variable("x", Domain::set([0, 1]));
    b.guarded_constraint(
        vec![Literal::new(x, 0)],
        Constraint::NotEqual { left: x, right: x },
    );
    b.cost(Cost::Table {
        variables: vec![x],
        entries: vec![(vec![0], 0), (vec![1], 7)],
    });
    for aggregation in [Aggregation::Sum, Aggregation::Maximum] {
        let mut solver = search(
            aggregation,
            vec![constant(2), Arc::new(b.clone().build().unwrap())],
        );
        let CompositeOutcome::Optimal(solution) = solver.advance(Limits::default()).unwrap() else {
            panic!()
        };
        assert_eq!(
            solution.cost(),
            if aggregation == Aggregation::Sum {
                9
            } else {
                7
            }
        );
        assert_eq!(solution.feasible().components()[1].values(), &[1]);
    }
}

#[test]
fn unresolved_component_cannot_be_hidden_under_a_large_maximum() {
    let mut b = ModelBuilder::new();
    b.obligation(
        vec![],
        ObligationKind::Analysis,
        "no feasible witness available",
    );
    for aggregation in [Aggregation::Sum, Aggregation::Maximum] {
        let mut solver = search(
            aggregation,
            vec![constant(100), Arc::new(b.clone().build().unwrap())],
        );
        for _ in 0..2 {
            let CompositeOutcome::Incomplete(progress) = solver.advance(Limits::default()).unwrap()
            else {
                panic!()
            };
            assert_eq!(progress.lower_bound, 100);
            assert!(progress.incumbent.is_none());
            let StopReason::Coverage(obligations) = progress.reason else {
                panic!()
            };
            assert_eq!(obligations.len(), 1);
            assert_eq!(obligations[0].kind, ObligationKind::Analysis);
        }
    }
}

#[test]
fn budgets_resume_and_bounds_remain_sound() {
    for aggregation in [Aggregation::Sum, Aggregation::Maximum] {
        let optimum = if aggregation == Aggregation::Sum {
            5
        } else {
            3
        };
        for (limits, expected_reason) in [
            (
                Limits {
                    work: 0,
                    ..Limits::default()
                },
                StopReason::Work,
            ),
            (
                Limits {
                    time: Some(Duration::ZERO),
                    ..Limits::default()
                },
                StopReason::Time,
            ),
            (
                Limits {
                    memory_bytes: Some(1),
                    ..Limits::default()
                },
                StopReason::Memory,
            ),
        ] {
            let mut solver = search(aggregation, vec![choices(&[8, 2, 5]), choices(&[9, 3, 7])]);
            let CompositeOutcome::Incomplete(progress) = solver.advance(limits).unwrap() else {
                panic!()
            };
            assert_eq!(progress.reason, expected_reason);
            let mut complete = false;
            for _ in 0..1000 {
                let before = solver.work();
                match solver
                    .advance(Limits {
                        work: 1,
                        ..Limits::default()
                    })
                    .unwrap()
                {
                    CompositeOutcome::Incomplete(progress) => {
                        assert!(progress.lower_bound <= optimum);
                        if let Some(witness) = progress.incumbent {
                            assert!(witness.cost() >= optimum);
                        }
                    }
                    CompositeOutcome::Optimal(solution) => {
                        assert_eq!(solution.cost(), optimum);
                        complete = true;
                    }
                    CompositeOutcome::Infeasible => panic!("known feasible"),
                }
                assert!(solver.work() - before <= 1);
                if complete {
                    break;
                }
            }
            assert!(complete, "one-unit resumption must finish");
        }
    }
}

#[test]
fn maximum_can_close_with_a_component_whose_own_optimum_is_unresolved() {
    let mut b = ModelBuilder::new();
    let x = b.variable("method", Domain::set([0, 1]));
    b.guarded_cost(vec![Literal::new(x, 1)], Cost::Constant(7));
    b.obligation(
        vec![Literal::new(x, 0)],
        ObligationKind::Analysis,
        "cheaper branch is not analyzed",
    );
    let component = Arc::new(b.build().unwrap());
    let mut individual = Search::new(component.clone(), Options::default()).unwrap();
    let Outcome::Incomplete(progress) = individual.advance(Limits::default()).unwrap() else {
        panic!("individual optimality remains unresolved")
    };
    assert_eq!(progress.lower_bound, 0);
    assert_eq!(progress.incumbent.unwrap().cost(), 7);
    assert!(matches!(progress.reason, StopReason::Coverage(_)));
    let mut solver = search(Aggregation::Maximum, vec![constant(100), component.clone()]);
    let CompositeOutcome::Optimal(solution) = solver.advance(Limits::default()).unwrap() else {
        panic!("a cheaper component cannot improve the maximum of 100")
    };
    assert_eq!(solution.cost(), 100);
    assert_eq!(solution.feasible().components()[1].cost(), 7);
    // The same argument is invalid for sum: a cheaper component would improve it.
    let mut sum = search(Aggregation::Sum, vec![constant(100), component]);
    let CompositeOutcome::Incomplete(progress) = sum.advance(Limits::default()).unwrap() else {
        panic!("sum must retain the coverage obligation")
    };
    assert_eq!(progress.lower_bound, 100);
    assert_eq!(progress.incumbent.unwrap().cost(), 107);
    assert!(matches!(progress.reason, StopReason::Coverage(_)));
}

#[test]
fn objective_units_must_match_and_sum_overflow_is_sticky() {
    let mut b = ModelBuilder::new();
    b.units("nanoseconds");
    assert!(matches!(
        Composition::new(
            Aggregation::Sum,
            vec![constant(0), Arc::new(b.build().unwrap())]
        ),
        Err(Error::InvalidModel(_))
    ));
    let mut solver = search(Aggregation::Sum, vec![constant(u64::MAX), constant(1)]);
    let error = solver.advance(Limits::default()).unwrap_err();
    assert!(matches!(error, Error::Overflow(_)));
    assert_eq!(solver.advance(Limits::default()).unwrap_err(), error);
    let mut maximum = search(Aggregation::Maximum, vec![constant(u64::MAX), constant(1)]);
    let CompositeOutcome::Optimal(solution) = maximum.advance(Limits::default()).unwrap() else {
        panic!()
    };
    assert_eq!(solution.cost(), u64::MAX);
}
