//! Budgeted one-solve API: incumbent installation and feasibility-before-budget ordering.
use magnitude_solver::{
    model::{Constraint, Cost, ModelBuilder},
    Algorithm, Budget, Budgeted, Domain, FeasibleAssignment, FeasibleSolution, Limits, Options,
    Outcome, Search,
};
use std::sync::Arc;

/// `choice` costs 10 at 0 and 5 at 1; the eight `f` variables are feasible
/// only at all-ones. The domain-minimum assignment is infeasible, the
/// constructive assignment (choice 0, all-ones) costs 10, the optimum is 5.
fn constrained_choice() -> Arc<magnitude_solver::Model> {
    let mut b = ModelBuilder::new();
    let choice = b.variable("choice", Domain::boolean());
    let forced: Vec<_> = (0..8)
        .map(|i| b.variable(format!("f{i}"), Domain::boolean()))
        .collect();
    b.constraint(Constraint::Table {
        variables: forced.clone(),
        tuples: vec![vec![1; forced.len()]],
    });
    b.cost(Cost::Table {
        variables: vec![choice],
        entries: vec![(vec![0], 10), (vec![1], 5)],
    });
    Arc::new(b.build().unwrap())
}

fn constructive(model: &magnitude_solver::Model) -> Vec<i64> {
    let mut values: Vec<_> = model
        .variables()
        .iter()
        .map(|v| v.domain.min().unwrap())
        .collect();
    // The compiler's constructive assignment fixes every forced variable.
    for (i, value) in values.iter_mut().enumerate().skip(1) {
        if model.variables()[i].domain.contains(1) {
            *value = 1;
        }
    }
    values[0] = 0;
    values
}

#[test]
fn witnesses_are_constructed_only_from_feasible_assignments() {
    let model = constrained_choice();
    let solution = FeasibleSolution::from_assignment(model.clone(), &constructive(&model)).unwrap();
    assert_eq!(solution.cost(), 10);
    assert!(matches!(
        FeasibleSolution::from_assignment(model.clone(), &[0; 9]),
        Err(magnitude_solver::model::Error::InvalidAssignment(_))
    ));
    assert!(matches!(
        FeasibleSolution::from_assignment(model.clone(), &[7, 1, 1, 1, 1, 1, 1, 1, 1]),
        Err(magnitude_solver::model::Error::InvalidAssignment(_))
    ));

    let mut b = ModelBuilder::new();
    b.variable("x", Domain::boolean());
    b.unresolved(vec![], "missing typed construction");
    let uncovered = b.build().unwrap();
    assert!(matches!(
        FeasibleSolution::from_assignment(Arc::new(uncovered), &[0]),
        Err(magnitude_solver::model::Error::InvalidAssignment(_))
    ));
}

#[test]
fn installed_incumbent_survives_a_tiny_optimization_budget() {
    let model = constrained_choice();
    let incumbent =
        FeasibleSolution::from_assignment(model.clone(), &constructive(&model)).unwrap();
    let mut search = Search::new(model.clone(), Options::default()).unwrap();
    assert!(search.install_incumbent(incumbent).unwrap());
    // The budget limits optimization only; it cannot discard the incumbent.
    let result = search
        .advance_budgeted(Budget {
            optimization: Limits {
                work: 1,
                ..Default::default()
            },
        })
        .unwrap();
    let Budgeted::Incumbent {
        solution,
        lower_bound,
    } = result
    else {
        panic!("installed incumbent must be returned under a tiny budget");
    };
    assert_eq!(solution.cost(), 10);
    assert!(lower_bound <= 10);
    // Optimization continues from the installed incumbent without restart.
    let Budgeted::Optimal(best) = search.advance_budgeted(Budget::default()).unwrap() else {
        panic!("exact search must prove the optimum");
    };
    assert_eq!(best.cost(), 5);
}

#[test]
fn a_weaker_incumbent_is_retained_but_not_installed() {
    let model = constrained_choice();
    let mut search = Search::new(model.clone(), Options::default()).unwrap();
    let mut best = constructive(&model);
    best[0] = 1;
    let winner = FeasibleSolution::from_assignment(model.clone(), &best).unwrap();
    assert!(search.install_incumbent(winner).unwrap());
    let weaker = FeasibleSolution::from_assignment(model.clone(), &constructive(&model)).unwrap();
    assert!(!search.install_incumbent(weaker).unwrap());
    let Budgeted::Optimal(solution) = search.advance_budgeted(Budget::default()).unwrap() else {
        panic!()
    };
    assert_eq!(solution.cost(), 5);
}

#[test]
fn feasibility_is_decided_before_any_budget_is_spent() {
    let model = constrained_choice();
    // The unrestricted domain-minimum proposal is infeasible, and one work
    // unit of ordinary advance cannot decide feasibility.
    let mut plain = Search::new(model.clone(), Options::default()).unwrap();
    let Outcome::Incomplete(progress) = plain
        .advance(Limits {
            work: 1,
            ..Default::default()
        })
        .unwrap()
    else {
        panic!("one work unit must suspend ordinary search");
    };
    assert!(progress.incumbent.is_none());
    // The same budget under budgeted semantics still decides feasibility.
    let mut budgeted = Search::new(model, Options::default()).unwrap();
    let result = budgeted
        .advance_budgeted(Budget {
            optimization: Limits {
                work: 1,
                ..Default::default()
            },
        })
        .unwrap();
    match result {
        Budgeted::Incumbent { solution, .. } => assert!(solution.cost() <= 10),
        Budgeted::Optimal(solution) => assert!(solution.cost() <= 10),
        _ => panic!("work budget must never suspend before feasibility is decided"),
    }
}

#[test]
fn proved_infeasibility_ignores_the_optimization_budget() {
    let mut b = ModelBuilder::new();
    let x = b.variable("x", Domain::boolean());
    b.constraint(Constraint::Table {
        variables: vec![x],
        tuples: vec![],
    });
    let model = Arc::new(b.build().unwrap());
    let mut search = Search::new(model, Options::default()).unwrap();
    let result = search
        .advance_budgeted(Budget {
            optimization: Limits {
                work: 1,
                ..Default::default()
            },
        })
        .unwrap();
    assert!(matches!(result, Budgeted::Infeasible));
}

#[test]
fn neighborhood_policy_installs_incumbents_without_restart() {
    let model = constrained_choice();
    let incumbent =
        FeasibleSolution::from_assignment(model.clone(), &constructive(&model)).unwrap();
    let mut search = Search::new(
        model,
        Options {
            algorithm: Algorithm::Neighborhood(Default::default()),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(search.install_incumbent(incumbent).unwrap());
    let result = search
        .advance_budgeted(Budget {
            optimization: Limits {
                work: 1,
                ..Default::default()
            },
        })
        .unwrap();
    let Budgeted::Incumbent { solution, .. } = result else {
        panic!("neighborhood policy must retain the installed incumbent");
    };
    assert_eq!(solution.cost(), 10);
    match search.advance_budgeted(Budget::default()).unwrap() {
        Budgeted::Incumbent { solution, .. } => assert!(solution.cost() <= 10),
        Budgeted::Optimal(solution) => assert!(solution.cost() <= 10),
        _ => panic!(),
    }
}

#[test]
fn preferred_assignments_guide_search_but_never_validate() {
    let model = constrained_choice();
    let mut search = Search::new(model.clone(), Options::default()).unwrap();
    // Out-of-domain preferences are rejected.
    assert!(search.prefer(&[7, 1, 1, 1, 1, 1, 1, 1, 1]).is_err());
    // An infeasible in-domain preference still guides a complete feasibility
    // search to the optimum.
    search.prefer(&[0; 9]).unwrap();
    let Budgeted::Optimal(solution) = search.advance_budgeted(Budget::default()).unwrap() else {
        panic!("preference is not a feasibility claim");
    };
    assert_eq!(solution.cost(), 5);
}

#[test]
fn incumbents_belong_to_the_exact_model() {
    let model = constrained_choice();
    let incumbent =
        FeasibleSolution::from_assignment(model.clone(), &constructive(&model)).unwrap();
    let mut other = ModelBuilder::new();
    other.variable("y", Domain::boolean());
    let mut search = Search::new(Arc::new(other.build().unwrap()), Options::default()).unwrap();
    assert!(search.install_incumbent(incumbent).is_err());
}

#[test]
fn feasible_assignments_export_only_through_validated_solutions() {
    let model = constrained_choice();
    let solution = FeasibleSolution::from_assignment(model.clone(), &constructive(&model)).unwrap();
    let exported = FeasibleAssignment::from(solution);
    assert_eq!(exported.values(), &constructive(&model)[..]);
    assert_eq!(exported.cost(), 10);
    assert_eq!(exported.model(), model.as_ref());
    assert_eq!(exported.solution().cost(), 10);
}
