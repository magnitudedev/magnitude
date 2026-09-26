use magnitude_solver::model::{Constraint, Cost, Domain, LinearTerm, ModelBuilder};
use magnitude_solver::scheduling::repetition::{IndependentRepetition, RepetitionOutcome};
use magnitude_solver::scheduling::{Activity, Interval, SchedulingConstraint};
use magnitude_solver::{Limits, Options, Outcome, Search};
use std::sync::Arc;

fn body() -> Arc<magnitude_solver::Model> {
    let mut builder = ModelBuilder::new();
    let choice = builder.variable("method", Domain::set([0, 1, 2]));
    builder.cost(Cost::Table {
        variables: vec![choice],
        entries: vec![(vec![0], 8), (vec![1], 3), (vec![2], 5)],
    });
    Arc::new(builder.build().unwrap())
}

#[test]
fn million_independent_occurrences_solve_one_body_and_count_every_copy() {
    let mut one = IndependentRepetition::new(body(), 1, Options::default()).unwrap();
    let mut many = IndependentRepetition::new(body(), 1_000_000, Options::default()).unwrap();
    let RepetitionOutcome::Optimal(one_solution) = one.advance(Limits::default()).unwrap() else {
        panic!("body should complete");
    };
    let RepetitionOutcome::Optimal(many_solution) = many.advance(Limits::default()).unwrap() else {
        panic!("repetition should complete");
    };
    assert_eq!(one_solution.cost(), 3);
    assert_eq!(many_solution.cost(), 3_000_000);
    assert_eq!(many_solution.count(), 1_000_000);
    assert_eq!(
        one.body_search().stats().work,
        many.body_search().stats().work
    );
    assert_eq!(many_solution.body().unwrap().values(), &[1]);
}

#[test]
fn repetition_resumes_and_does_not_turn_partial_progress_into_optimality() {
    let mut search = IndependentRepetition::new(body(), 17, Options::default()).unwrap();
    let RepetitionOutcome::Incomplete(progress) = search
        .advance(Limits {
            work: 0,
            ..Limits::default()
        })
        .unwrap()
    else {
        panic!("zero budget cannot prove this body");
    };
    assert!(progress.lower_bound <= 51);
    let RepetitionOutcome::Optimal(solution) = search.advance(Limits::default()).unwrap() else {
        panic!("body should complete");
    };
    assert_eq!(solution.cost(), 51);
}

#[test]
fn repeated_cost_overflow_is_an_error_not_a_valid_bound() {
    let mut builder = ModelBuilder::new();
    builder.cost(Cost::Constant(u64::MAX));
    let mut search =
        IndependentRepetition::new(Arc::new(builder.build().unwrap()), 2, Options::default())
            .unwrap();
    assert!(matches!(
        search.advance(Limits::default()),
        Err(magnitude_solver::Error::Overflow(_))
    ));
}

#[test]
fn empty_repetition_has_no_feasibility_obligation_for_unexecuted_body() {
    let mut builder = ModelBuilder::new();
    let v = builder.variable("v", Domain::singleton(0));
    builder.constraint(Constraint::NotEqual { left: v, right: v });
    let mut search =
        IndependentRepetition::new(Arc::new(builder.build().unwrap()), 0, Options::default())
            .unwrap();
    let RepetitionOutcome::Optimal(solution) = search
        .advance(Limits {
            work: 0,
            ..Limits::default()
        })
        .unwrap()
    else {
        panic!("empty repetition costs zero");
    };
    assert_eq!(solution.cost(), 0);
    assert!(solution.body().is_none());
}

#[test]
fn parallel_resource_coupled_occurrences_are_solved_jointly() {
    let mut builder = ModelBuilder::new();
    let duration = builder.variable("duration", Domain::singleton(2));
    let completion = builder.variable("completion", Domain::interval(0, 4).unwrap());
    let mut intervals = Vec::new();
    for i in 0..2 {
        let start = builder.variable(format!("start_{i}"), Domain::interval(0, 2).unwrap());
        let end = builder.variable(format!("end_{i}"), Domain::interval(2, 4).unwrap());
        builder.constraint(Constraint::Schedule(SchedulingConstraint::Activity(
            Activity {
                start,
                duration,
                end,
                presence: None,
            },
        )));
        builder.constraint(Constraint::LinearLe {
            terms: vec![LinearTerm::new(end, 1), LinearTerm::new(completion, -1)],
            rhs: 0,
        });
        intervals.push(Interval::mandatory(start, end));
    }
    builder.constraint(Constraint::Schedule(SchedulingConstraint::NoOverlap {
        intervals,
    }));
    builder.cost(Cost::Linear {
        constant: 0,
        terms: vec![LinearTerm::new(completion, 1)],
    });
    let mut search = Search::new(Arc::new(builder.build().unwrap()), Options::default()).unwrap();
    let Outcome::Optimal(solution) = search.advance(Limits::default()).unwrap() else {
        panic!("tiny joint schedule should complete");
    };
    assert_eq!(solution.cost(), 4);
    assert_ne!(solution.values()[2], solution.values()[4]);
}
