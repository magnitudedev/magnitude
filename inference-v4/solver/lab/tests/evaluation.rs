use magnitude_solver_lab::{
    generators::{self, Parameters},
    reference::{self, ReferenceOutcome},
    runner::{self, RunSettings},
};
use std::time::Duration;

#[test]
fn every_search_policy_matches_the_reference() {
    for family in [
        "independent",
        "chain",
        "separator",
        "shared-producer",
        "process-plan",
        "repeated",
        "coverage-gap",
    ] {
        let instance = generators::generate(
            family,
            Parameters {
                n: 2,
                depth: 1,
                ..Parameters::default()
            },
            3,
        )
        .unwrap();
        let reference = reference::exhaustive(&instance, 10_000, Duration::from_secs(2)).unwrap();
        for policy in [
            "dfs",
            "best-first",
            "no-cache",
            "no-decompose",
            "weak-bounds",
        ] {
            let result = runner::solve(
                &instance,
                &RunSettings {
                    policy: policy.into(),
                    seconds: 5.0,
                    sample_work: 7,
                    ..RunSettings::default()
                },
            )
            .unwrap();
            match reference.outcome {
                ReferenceOutcome::Optimal(cost) => {
                    assert_eq!(
                        result.status, "optimal",
                        "{family} {policy}: {:?}",
                        result.reason
                    );
                    assert_eq!(result.cost, Some(cost));
                }
                ReferenceOutcome::Infeasible => assert_eq!(result.status, "infeasible"),
                ReferenceOutcome::Incomplete => assert_eq!(result.status, "incomplete"),
            }
        }
    }
}

#[test]
fn saved_instance_preserves_model_identity_and_optimum() {
    let instance = generators::conditional_fixture().unwrap();
    let json = serde_json::to_vec(&instance).unwrap();
    let read: generators::Instance = serde_json::from_slice(&json).unwrap();
    assert_eq!(instance.fingerprint().unwrap(), read.fingerprint().unwrap());
    assert_eq!(
        runner::solve(&read, &RunSettings::default()).unwrap().cost,
        Some(15)
    );
}

#[test]
fn schedules_and_pipeline_have_small_exact_references() {
    for family in ["schedule", "pipeline"] {
        let instance = generators::generate(
            family,
            Parameters {
                n: 1,
                d: 1,
                horizon: 3,
                outputs: 1,
                input_length: 1,
                overhead: 0,
                ..Parameters::default()
            },
            0,
        )
        .unwrap();
        let reference =
            reference::exhaustive(&instance, 1_000_000, Duration::from_secs(5)).unwrap();
        let result = runner::solve(
            &instance,
            &RunSettings {
                seconds: 5.0,
                ..RunSettings::default()
            },
        )
        .unwrap();
        let ReferenceOutcome::Optimal(expected) = reference.outcome else {
            panic!("{family}: {reference:?}");
        };
        assert_eq!(result.status, "optimal", "{family}: {:?}", result.reason);
        assert_eq!(result.cost, Some(expected));
    }
}

#[test]
fn initially_open_coupled_fixtures_complete_exactly() {
    for (family, expected) in [("packing-gap", 9), ("repeated-coupled", 8)] {
        let instance = generators::generate(family, Parameters::default(), 0).unwrap();
        assert_eq!(
            reference::exhaustive(&instance, 1_000_000, Duration::from_secs(5))
                .unwrap()
                .outcome,
            ReferenceOutcome::Optimal(expected)
        );
        let result = runner::solve(
            &instance,
            &RunSettings {
                seconds: 5.0,
                sample_work: 100,
                ..RunSettings::default()
            },
        )
        .unwrap();
        assert_eq!(result.status, "optimal", "{family}: {:?}", result.reason);
        assert_eq!(result.cost, Some(expected));
    }
}
