//! Structural reuse must preserve physical occurrence costs and full contexts.
use magnitude_solver::{
    model::{Constraint, Cost, Domain, LinearTerm, Literal, ModelBuilder},
    Limits, Options, Outcome, Search,
};
use std::sync::Arc;

fn repeated(copies: usize, changed: bool) -> Arc<magnitude_solver::Model> {
    let mut b = ModelBuilder::new();
    for copy in 0..copies {
        let x = b.variable(format!("x{copy}"), Domain::boolean());
        let y = b.variable(format!("y{copy}"), Domain::boolean());
        b.constraint(Constraint::ExactlyOne {
            variables: vec![x, y],
        });
        b.cost(Cost::Linear {
            constant: 0,
            terms: vec![
                LinearTerm::new(x, 3),
                LinearTerm::new(y, if changed && copy == copies - 1 { 7 } else { 2 }),
            ],
        });
    }
    Arc::new(b.build().unwrap())
}
#[test]
fn renamed_components_reuse_proofs_but_each_occurrence_is_charged() {
    let model = repeated(32, false);
    let mut s = Search::new(model.clone(), Options::default()).unwrap();
    let Outcome::Optimal(solution) = s.advance(Limits::default()).unwrap() else {
        panic!("small finite model must finish")
    };
    assert_eq!(solution.cost(), 64);
    assert!(s.stats().structural_hits >= 31, "{:?}", s.stats());
    assert_eq!(
        model
            .validate_assignment(solution.values())
            .unwrap()
            .exact_cost,
        Some(64)
    );
}
#[test]
fn changed_factor_prevents_stale_reuse() {
    let model = repeated(8, true);
    let mut s = Search::new(model.clone(), Options::default()).unwrap();
    let Outcome::Optimal(solution) = s.advance(Limits::default()).unwrap() else {
        panic!()
    };
    assert_eq!(solution.cost(), 17);
    assert_eq!(
        model
            .validate_assignment(solution.values())
            .unwrap()
            .exact_cost,
        Some(17)
    );
}
#[test]
fn shared_resource_keeps_otherwise_equivalent_choices_coupled() {
    let mut b = ModelBuilder::new();
    let mut terms = vec![];
    for i in 0..8 {
        let x = b.variable(format!("choice{i}"), Domain::boolean());
        terms.push(LinearTerm::new(x, 1));
        b.cost(Cost::Table {
            variables: vec![x],
            entries: vec![(vec![0], 3), (vec![1], 1)],
        });
    }
    b.constraint(Constraint::LinearLe { terms, rhs: 3 });
    let model = Arc::new(b.build().unwrap());
    let mut s = Search::new(model, Options::default()).unwrap();
    let Outcome::Optimal(solution) = s
        .advance(Limits {
            work: 1_000_000,
            ..Default::default()
        })
        .unwrap()
    else {
        panic!()
    };
    assert_eq!(solution.cost(), 18);
}
#[test]
fn guard_and_fixed_boundary_changes_remain_in_reuse_identity() {
    let mut b = ModelBuilder::new();
    for (i, active) in [0, 1].into_iter().enumerate() {
        let g = b.variable(format!("guard{i}"), Domain::singleton(active));
        let x = b.variable(format!("value{i}"), Domain::boolean());
        b.guarded_constraint(
            vec![Literal::new(g, 1)],
            Constraint::InDomain {
                variable: x,
                domain: Domain::singleton(1),
            },
        );
        b.cost(Cost::Table {
            variables: vec![x],
            entries: vec![(vec![0], 1), (vec![1], 5)],
        });
    }
    let mut s = Search::new(Arc::new(b.build().unwrap()), Options::default()).unwrap();
    let Outcome::Optimal(solution) = s.advance(Limits::default()).unwrap() else {
        panic!()
    };
    assert_eq!(solution.cost(), 6);
}
