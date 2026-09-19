use magnitude_solver::{
    model::{Constraint, Cost, LinearTerm, Literal, ModelBuilder},
    Domain, Limits, Options, Outcome, Search,
};
use std::sync::Arc;
fn solve(b: ModelBuilder) -> u64 {
    let mut s = Search::new(Arc::new(b.build().unwrap()), Options::default()).unwrap();
    match s
        .advance(Limits {
            work: 1_000_000,
            ..Default::default()
        })
        .unwrap()
    {
        Outcome::Optimal(x) => x.cost(),
        x => panic!("{x:?}"),
    }
}
#[test]
fn representation_is_chosen_for_complete_composition() {
    let mut b = ModelBuilder::new();
    let v = b.variable("format", Domain::boolean());
    for costs in [[2, 8], [9, 3], [9, 4]] {
        b.cost(Cost::Table {
            variables: vec![v],
            entries: vec![(vec![0], costs[0]), (vec![1], costs[1])],
        });
    }
    assert_eq!(solve(b), 15);
}
#[test]
fn physical_production_is_distinct_from_analysis_reuse() {
    for (shared, expected) in [(true, 8), (false, 14)] {
        let mut b = ModelBuilder::new();
        for _ in 0..if shared { 1 } else { 2 } {
            b.cost(Cost::Constant(6));
        }
        b.cost(Cost::Constant(2));
        assert_eq!(solve(b), expected);
    }
}
#[test]
fn all_policies_and_resumption_match_exhaustive_integer_problem() {
    for seed in 0..20 {
        let mut b = ModelBuilder::new();
        let vars: Vec<_> = (0..5)
            .map(|i| b.variable(format!("x{i}"), Domain::interval(0, 3).unwrap()))
            .collect();
        for (i, &v) in vars.iter().enumerate() {
            b.cost(Cost::Table {
                variables: vec![v],
                entries: (0..4)
                    .map(|x| (vec![x], ((seed + i as i64 * 3 + x * 7) % 11) as u64))
                    .collect(),
            });
        }
        for w in vars.windows(2) {
            b.constraint(Constraint::NotEqual {
                left: w[0],
                right: w[1],
            });
        }
        let m = Arc::new(b.build().unwrap());
        let mut best = None;
        for mask in 0..1024 {
            let mut n = mask;
            let values: Vec<_> = (0..5)
                .map(|_| {
                    let x = n % 4;
                    n /= 4;
                    x
                })
                .collect();
            let a = m.validate_assignment(&values).unwrap();
            if !a.infeasible {
                if let Some(c) = a.exact_cost {
                    best = Some(best.map_or(c, |old: u64| old.min(c)));
                }
            }
        }
        for policy in [
            magnitude_solver::Policy::DepthFirst,
            magnitude_solver::Policy::BestFirst,
        ] {
            for enabled in [false, true] {
                let mut search = Search::new(
                    m.clone(),
                    Options {
                        policy,
                        memoize: enabled,
                        decompose: enabled,
                        bounds: enabled,
                        ..Default::default()
                    },
                )
                .unwrap();
                let mut iterations = 0;
                loop {
                    iterations += 1;
                    assert!(iterations < 100_000);
                    match search
                        .advance(Limits {
                            work: 7,
                            ..Default::default()
                        })
                        .unwrap()
                    {
                        Outcome::Optimal(s) => {
                            assert_eq!(Some(s.cost()), best);
                            break;
                        }
                        Outcome::Incomplete(p) => {
                            assert!(p.lower_bound <= best.unwrap());
                            if let Some(i) = p.incumbent {
                                assert!(i.cost() >= best.unwrap());
                            }
                        }
                        x => panic!("{x:?}"),
                    }
                }
            }
        }
    }
}
#[test]
fn unknown_coverage_cannot_be_claimed_optimal_but_can_be_excluded() {
    for unknown_cost in [0, 20] {
        let mut b = ModelBuilder::new();
        let x = b.variable("choice", Domain::boolean());
        b.guarded_cost(vec![Literal::new(x, 0)], Cost::Constant(10));
        b.guarded_cost(vec![Literal::new(x, 1)], Cost::Constant(unknown_cost));
        b.unresolved(vec![Literal::new(x, 1)], "missing realization");
        let mut s = Search::new(Arc::new(b.build().unwrap()), Options::default()).unwrap();
        let out = s
            .advance(Limits {
                work: 10000,
                ..Default::default()
            })
            .unwrap();
        if unknown_cost == 0 {
            assert!(matches!(out, Outcome::Incomplete(_)));
        } else {
            assert!(matches!(out,Outcome::Optimal(ref x) if x.cost()==10));
        }
    }
}
#[test]
fn large_domain_affine_bound_finishes_without_enumerating_values() {
    let mut b = ModelBuilder::new();
    let x = b.variable("million", Domain::interval(0, 1_000_000).unwrap());
    b.constraint(Constraint::LinearLe {
        terms: vec![LinearTerm::new(x, -1)],
        rhs: -900_000,
    });
    b.cost(Cost::Linear {
        constant: 0,
        terms: vec![LinearTerm::new(x, 1)],
    });
    assert_eq!(solve(b), 900_000);
}
#[test]
fn infeasible_does_not_mean_arithmetic_or_model_error() {
    let mut b = ModelBuilder::new();
    let x = b.variable("x", Domain::singleton(1));
    b.constraint(Constraint::NotEqual { left: x, right: x });
    let mut s = Search::new(Arc::new(b.build().unwrap()), Options::default()).unwrap();
    assert!(matches!(
        s.advance(Limits::default()).unwrap(),
        Outcome::Infeasible
    ));
}

#[test]
fn discharged_constraints_retain_narrowed_unfixed_values() {
    let mut b = ModelBuilder::new();
    let x = b.variable("narrowed", Domain::interval(0, 10).unwrap());
    let y = b.variable("remaining", Domain::interval(0, 5).unwrap());
    b.constraint(Constraint::LinearLe {
        terms: vec![LinearTerm::new(x, -1)],
        rhs: -3,
    });
    b.cost(Cost::Linear {
        constant: 0,
        terms: vec![LinearTerm::new(y, 1)],
    });
    let mut search = Search::new(Arc::new(b.build().unwrap()), Options::default()).unwrap();
    let Outcome::Optimal(solution) = search.advance(Limits::default()).unwrap() else {
        panic!("expected optimum")
    };
    assert!(solution.values()[x.0] >= 3);
    assert_eq!(solution.cost(), 0);
}
