use magnitude_solver_lab::{
    generators::{generate, Parameters},
    reference::{self, ReferenceOutcome},
};
use std::time::Duration;

#[test]
fn kernel_capacity_dp_agrees_with_independent_complete_enumeration() {
    for family in ["kernel-contraction", "kernel-shared"] {
        for seed in 0..4 {
            for capacity in [0, 2, 8] {
                let instance = generate(
                    family,
                    Parameters {
                        n: 2,
                        d: 2,
                        capacity,
                        ..Parameters::default()
                    },
                    seed,
                )
                .unwrap();
                let exact =
                    reference::exhaustive(&instance, 100_000, Duration::from_secs(3)).unwrap();
                assert_ne!(exact.outcome, ReferenceOutcome::Incomplete);
                assert_eq!(
                    reference::analytic(&instance).unwrap().unwrap().outcome,
                    exact.outcome,
                    "{family} seed {seed} capacity {capacity}"
                );
            }
        }
    }
}

#[test]
fn scheduled_modes_have_independently_enumerated_optima() {
    for seed in 0..4 {
        let instance = generate(
            "kernel-schedule",
            Parameters {
                n: 2,
                horizon: 8,
                capacity: 2,
                ..Parameters::default()
            },
            seed,
        )
        .unwrap();
        let result = reference::exhaustive(&instance, 100_000, Duration::from_secs(3)).unwrap();
        assert!(matches!(result.outcome, ReferenceOutcome::Optimal(_)));
        assert_eq!(result.assignments, 4 * 9 * 9);
        assert_eq!(
            reference::exhaustive(&instance, 1, Duration::from_secs(3))
                .unwrap()
                .outcome,
            ReferenceOutcome::Incomplete
        );
    }
}

#[test]
fn schedule_oracle_separates_bad_schedule_from_implementation_choice() {
    use magnitude_solver_lab::generators::Reference;
    let instance = generate(
        "kernel-schedule",
        Parameters {
            n: 2,
            horizon: 12,
            capacity: 2,
            ..Parameters::default()
        },
        7,
    )
    .unwrap();
    let Some(Reference::KernelSchedule {
        modes,
        starts,
        ends,
        durations,
        demands,
        completion,
        ..
    }) = &instance.reference
    else {
        panic!()
    };
    let mut values: Vec<_> = instance
        .model
        .variables()
        .iter()
        .map(|v| v.domain.min().unwrap())
        .collect();
    let mut finish = 0;
    for i in 0..modes.len() {
        values[modes[i].0] = 0;
        values[starts[i].0] = finish;
        values[durations[i].0 .0] = durations[i].1[0] as i64;
        values[demands[i].0 .0] = demands[i].1[0] as i64;
        finish += durations[i].1[0] as i64;
        values[ends[i].0] = finish;
    }
    values[completion.0] = finish;
    assert!(
        reference::evaluate(&instance.model, &values)
            .unwrap()
            .feasible
    );
    let selected =
        reference::implementation_schedule(&instance, Some(&values), 10000, Duration::from_secs(2))
            .unwrap();
    let expected = durations.iter().map(|(_, d)| d[0]).max().unwrap();
    assert_eq!(selected.outcome, ReferenceOutcome::Optimal(expected));
    assert!(expected < finish as u64);
}
