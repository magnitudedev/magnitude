use magnitude_solver::{
    Domain, Limits, Model, ModelBuilder, Options, Outcome, Search,
    model::{Constraint, Cost},
    result::StopReason,
};
use std::{sync::Arc, time::Duration};

fn instance() -> Arc<Model> {
    let mut b = ModelBuilder::new();
    let x = b.variable("x", Domain::interval(0, 7).unwrap());
    let y = b.variable("y", Domain::interval(0, 7).unwrap());
    b.constraint(Constraint::NotEqual { left: x, right: y });
    for v in [x, y] {
        b.cost(Cost::Table {
            variables: vec![v],
            entries: (0..8).map(|i| (vec![i], (7 - i) as u64)).collect(),
        });
    }
    Arc::new(b.build().unwrap())
}

#[test]
fn every_budget_preserves_coverage_and_can_resume() {
    for (limits, expected) in [
        (
            Limits {
                work: 0,
                ..Default::default()
            },
            StopReason::Work,
        ),
        (
            Limits {
                time: Some(Duration::ZERO),
                ..Default::default()
            },
            StopReason::Time,
        ),
        (
            Limits {
                memory_bytes: Some(1),
                ..Default::default()
            },
            StopReason::Memory,
        ),
    ] {
        let mut search = Search::new(instance(), Options::default()).unwrap();
        let Outcome::Incomplete(progress) = search.advance(limits).unwrap() else {
            panic!("budget must interrupt")
        };
        assert_eq!(progress.reason, expected);
        assert!(progress.incumbent.is_none());
        assert_eq!(search.stats().work, 0);
        let Outcome::Optimal(solution) = search.advance(Limits::default()).unwrap() else {
            panic!("must resume")
        };
        assert_eq!(solution.cost(), 1);
        assert!(search.stats().retained_bytes > 0);
    }
}

#[test]
fn one_work_unit_resumption_matches_uninterrupted_trace() {
    let mut one = Search::new(instance(), Options::default()).unwrap();
    let mut all = Search::new(instance(), Options::default()).unwrap();
    let Outcome::Optimal(expected) = all.advance(Limits::default()).unwrap() else {
        panic!("must complete")
    };
    loop {
        match one
            .advance(Limits {
                work: 1,
                ..Default::default()
            })
            .unwrap()
        {
            Outcome::Incomplete(progress) => {
                assert!(progress.lower_bound <= expected.cost());
                if let Some(incumbent) = progress.incumbent {
                    assert!(incumbent.cost() >= expected.cost());
                }
            }
            Outcome::Optimal(actual) => {
                assert_eq!(actual.cost(), expected.cost());
                assert_eq!(actual.values(), expected.values());
                assert_eq!(one.stats().work, all.stats().work);
                assert_eq!(one.stats().nodes, all.stats().nodes);
                break;
            }
            Outcome::Infeasible => panic!("known feasible"),
        }
    }
}

#[test]
fn insufficient_pending_storage_requires_a_larger_memory_allowance() {
    let mut probe = Search::new(instance(), Options::default()).unwrap();
    let Outcome::Incomplete(progress) = probe
        .advance(Limits {
            work: 1,
            ..Default::default()
        })
        .unwrap()
    else {
        panic!("probe must still be pending")
    };
    let cap = progress.stats.retained_bytes + 2_048;

    let mut search = Search::new(instance(), Options::default()).unwrap();
    let mut stalled = None;
    for _ in 0..4 {
        match search
            .advance(Limits {
                work: 1_000,
                memory_bytes: Some(cap),
                ..Default::default()
            })
            .unwrap()
        {
            Outcome::Optimal(_) => panic!("fixture must require more pending storage"),
            Outcome::Incomplete(progress) => {
                assert_eq!(progress.reason, StopReason::Memory);
                if let Some(work) = stalled {
                    assert_eq!(progress.stats.work, work);
                }
                stalled = Some(progress.stats.work);
            }
            Outcome::Infeasible => panic!("known feasible"),
        }
    }
    let Outcome::Optimal(result) = search.advance(Limits::default()).unwrap() else {
        panic!("search must resume after lifting the insufficient memory cap");
    };
    assert_eq!(result.cost(), 1);
}

#[test]
fn arithmetic_failure_does_not_turn_into_coverage_or_infeasibility_on_retry() {
    let mut b = ModelBuilder::new();
    b.cost(Cost::Constant(u64::MAX));
    b.cost(Cost::Constant(1));
    let mut search = Search::new(Arc::new(b.build().unwrap()), Options::default()).unwrap();
    let first = search.advance(Limits::default()).unwrap_err();
    assert!(matches!(first, magnitude_solver::Error::Overflow(_)));
    assert_eq!(search.advance(Limits::default()).unwrap_err(), first);
}

#[test]
fn missing_construction_analysis_and_refinement_need_more_than_search_budget() {
    use magnitude_solver::model::ObligationKind;
    for kind in [
        ObligationKind::Construction,
        ObligationKind::Analysis,
        ObligationKind::Refinement,
    ] {
        let mut builder = ModelBuilder::new();
        builder.obligation(vec![], kind, "required relation unavailable");
        let mut search =
            Search::new(Arc::new(builder.build().unwrap()), Options::default()).unwrap();
        let Outcome::Incomplete(progress) = search.advance(Limits::default()).unwrap() else {
            panic!("coverage must remain unresolved")
        };
        let StopReason::Coverage(reasons) = progress.reason else {
            panic!("not a budget stop")
        };
        assert_eq!(reasons.len(), 1);
        assert_eq!(reasons[0].kind, kind);
        let work = search.stats().work;
        let Outcome::Incomplete(again) = search
            .advance(Limits {
                work: 1_000_000,
                ..Default::default()
            })
            .unwrap()
        else {
            panic!("budget cannot invent a missing relation")
        };
        assert_eq!(again.reason, StopReason::Coverage(reasons));
        assert_eq!(search.stats().work, work);
    }
}
