use magnitude_solver::{
    model::{Cost, Fragment, Literal, ModelBuilder},
    Domain, Limits, Options, Outcome, Search,
};
use std::sync::Arc;

fn conditional(lazy: bool) -> magnitude_solver::Model {
    let mut body = ModelBuilder::new();
    let x = body.variable("private", Domain::interval(1, 8).unwrap());
    body.cost(Cost::Table {
        variables: vec![x],
        entries: (1..=8).map(|v| (vec![v], (9 - v) as u64)).collect(),
    });
    let fragment = Fragment::new(body.build().unwrap(), vec![]).unwrap();
    let mut b = ModelBuilder::new();
    let active = b.variable("realization", Domain::boolean());
    for value in 0..=1 {
        if lazy {
            b.instantiate_lazy(
                format!("body{value}"),
                &fragment,
                &[],
                vec![Literal::new(active, value)],
            )
            .unwrap();
        } else {
            b.instantiate(
                format!("body{value}"),
                &fragment,
                &[],
                vec![Literal::new(active, value)],
            )
            .unwrap();
        }
    }
    b.build().unwrap()
}

#[test]
fn lazy_eager_cached_and_uncached_preserve_the_same_optimum() {
    for lazy in [false, true] {
        for cache in [false, true] {
            let mut search = Search::new(
                Arc::new(conditional(lazy)),
                Options {
                    memoize: cache,
                    ..Default::default()
                },
            )
            .unwrap();
            for _ in 0..100_000 {
                match search
                    .advance(Limits {
                        work: 1,
                        ..Default::default()
                    })
                    .unwrap()
                {
                    Outcome::Optimal(solution) => {
                        assert_eq!(solution.cost(), 1);
                        break;
                    }
                    Outcome::Incomplete(_) => continue,
                    Outcome::Infeasible => panic!("feasible fragments"),
                }
            }
            assert!(matches!(
                search.advance(Limits::default()).unwrap(),
                Outcome::Optimal(_)
            ));
            if lazy {
                assert!(search.stats().materializations > 0);
            }
        }
    }
}

#[test]
fn inactive_fragment_is_never_materialized() {
    let mut body = ModelBuilder::new();
    for _ in 0..10_000 {
        body.cost(Cost::Constant(1));
    }
    let fragment = Fragment::new(body.build().unwrap(), vec![]).unwrap();
    let mut b = ModelBuilder::new();
    let off = b.variable("off", Domain::singleton(0));
    b.instantiate_lazy("unused", &fragment, &[], vec![Literal::new(off, 1)])
        .unwrap();
    let mut search = Search::new(Arc::new(b.build().unwrap()), Options::default()).unwrap();
    assert!(
        matches!(search.advance(Limits::default()).unwrap(),Outcome::Optimal(ref x) if x.cost()==0)
    );
    assert_eq!(search.stats().materializations, 0);
}

#[test]
fn active_fragment_construction_is_resumable_one_child_at_a_time() {
    let mut body = ModelBuilder::new();
    for _ in 0..100 {
        body.cost(Cost::Constant(1));
    }
    let fragment = Fragment::new(body.build().unwrap(), vec![]).unwrap();
    let mut b = ModelBuilder::new();
    b.instantiate_lazy("body", &fragment, &[], vec![]).unwrap();
    let mut search = Search::new(Arc::new(b.build().unwrap()), Options::default()).unwrap();
    for count in 1..=100 {
        assert!(matches!(
            search
                .advance(Limits {
                    work: 1,
                    ..Default::default()
                })
                .unwrap(),
            Outcome::Incomplete(_)
        ));
        assert_eq!(search.stats().materializations, count);
    }
    assert!(
        matches!(search.advance(Limits::default()).unwrap(),Outcome::Optimal(ref x) if x.cost()==100)
    );
}

#[test]
fn losing_body_is_excluded_before_full_materialization() {
    let mut body = ModelBuilder::new();
    for _ in 0..10_000 {
        body.cost(Cost::Constant(1));
    }
    let body = Fragment::new(body.build().unwrap(), vec![]).unwrap();
    let mut b = ModelBuilder::new();
    let choice = b.variable("choice", Domain::boolean());
    b.instantiate_lazy("expensive", &body, &[], vec![Literal::new(choice, 1)])
        .unwrap();
    b.guarded_cost(vec![Literal::new(choice, 0)], Cost::Constant(10));
    b.guarded_cost(vec![Literal::new(choice, 1)], Cost::Constant(20));
    let mut search = Search::new(Arc::new(b.build().unwrap()), Options::default()).unwrap();
    assert!(
        matches!(search.advance(Limits::default()).unwrap(),Outcome::Optimal(ref x) if x.cost()==10)
    );
    assert_eq!(search.stats().materializations, 0);
}
