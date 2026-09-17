use seismic_accounting::{prediction::*, quantity::Count, resource::*};
fn evidence() -> Evidence {
    Evidence {
        source: "independent analytic test".into(),
        conditions: "test schedule and units".into(),
    }
}
fn latency(seconds: f64) -> Latency {
    Latency {
        seconds: Range::exact(seconds).unwrap(),
        evidence: evidence(),
    }
}
fn service(pool: &str) -> Service {
    Service {
        pool: pool.into(),
        unit: Unit::Bytes,
    }
}
fn demand(pool: &str, count: Count, outstanding: Count) -> PlannedService {
    PlannedService {
        demand: Demand {
            service: service(pool),
            count,
            meaning: DemandMeaning::Realization,
            origin: "test transfer".into(),
        },
        mapping: evidence(),
        outstanding: Some((outstanding, evidence())),
    }
}
fn behavior(pool: &str, rate: f64, delay: f64) -> Behavior {
    Behavior {
        saturated: Rate::new(
            service(pool),
            rate,
            RateEvidence::Calibration {
                probe: "test-service".into(),
                environment: "test-state".into(),
            },
        )
        .unwrap(),
        latency: Some(latency(delay)),
    }
}
fn plan(services: Vec<PlannedService>) -> Plan {
    Plan {
        identity: "candidate-a".into(),
        profile_identity: "profile-a".into(),
        workload_identity: "workload-a".into(),
        boundary_identity: "synchronous-kernel".into(),
        stages: vec![Stage {
            name: "work".into(),
            services,
            dependency: latency(0.0),
            overlap: evidence(),
        }],
        sequencing: evidence(),
        submission: latency(0.25),
        unavailable: vec![],
    }
}
fn profile() -> Profile {
    Profile {
        identity: "profile-a".into(),
        behaviors: vec![behavior("memory", 100., 2.), behavior("other", 50., 1.)],
    }
}
#[test]
fn concurrency_saturation_shared_service_and_dependencies() {
    // 20 bytes in flight / 2 sec = 10 bytes/sec; 100 bytes takes 10 sec.
    let mut p = plan(vec![demand("memory", Count::Exact(100), Count::Exact(20))]);
    assert_eq!(
        predict(&p, &profile())
            .unwrap()
            .seconds
            .unwrap()
            .endpoints(),
        (10.25, 10.25)
    );
    // Concurrency beyond saturation cannot make the calibrated pool faster.
    p.stages[0].services[0].outstanding = Some((Count::Exact(400), evidence()));
    assert_eq!(
        predict(&p, &profile())
            .unwrap()
            .seconds
            .unwrap()
            .endpoints(),
        (1.25, 1.25)
    );
    // Same-pool demands add before independent pools overlap.
    p.stages[0]
        .services
        .push(demand("memory", Count::Exact(200), Count::Exact(400)));
    p.stages[0]
        .services
        .push(demand("other", Count::Exact(100), Count::Exact(100)));
    assert_eq!(
        predict(&p, &profile())
            .unwrap()
            .seconds
            .unwrap()
            .endpoints(),
        (3.25, 3.25)
    );
    p.stages[0].dependency = latency(4.);
    p.stages.push(p.stages[0].clone());
    assert_eq!(
        predict(&p, &profile())
            .unwrap()
            .seconds
            .unwrap()
            .endpoints(),
        (8.25, 8.25)
    );
}
#[test]
fn uncertainty_is_propagated_and_never_silently_ranked() {
    let p = plan(vec![demand(
        "memory",
        Count::interval(100, 200).unwrap(),
        Count::interval(20, 40).unwrap(),
    )]);
    let prediction = predict(&p, &profile()).unwrap();
    assert_eq!(prediction.seconds.unwrap().endpoints(), (5.25, 20.25));
    let mut faster = plan(vec![demand("memory", Count::Exact(100), Count::Exact(400))]);
    faster.identity = "candidate-b".into();
    let fast = predict(&faster, &profile()).unwrap();
    assert_eq!(
        unambiguous_choice(&[prediction.clone(), fast]).unwrap(),
        Some(1)
    );
    assert_eq!(
        unambiguous_choice(&[prediction.clone(), prediction]).unwrap(),
        None
    );
    faster
        .unavailable
        .push("post-compiler spilling is unmodeled".into());
    let missing = predict(&faster, &profile()).unwrap();
    assert!(missing.seconds.is_none());
    assert_eq!(unambiguous_choice(&[missing]).unwrap(), None);
}
#[test]
fn unknowns_missing_units_and_capacity_only_profiles_stay_unavailable() {
    for d in [
        demand(
            "memory",
            Count::unknown("runtime domain"),
            Count::Exact(100),
        ),
        demand("missing", Count::Exact(100), Count::Exact(100)),
        demand("memory", Count::Exact(100), Count::Exact(0)),
    ] {
        let prediction = predict(&plan(vec![d]), &profile()).unwrap();
        assert!(prediction.seconds.is_none());
        assert!(!prediction.unavailable.is_empty());
    }
    let mut p = profile();
    p.behaviors[0].saturated = Rate::new(
        service("memory"),
        100.,
        RateEvidence::CapacityBound {
            source: "hardware maximum".into(),
            conditions: "test".into(),
        },
    )
    .unwrap();
    assert!(predict(
        &plan(vec![demand("memory", Count::Exact(100), Count::Exact(100))]),
        &p
    )
    .unwrap()
    .seconds
    .is_none());
}
#[test]
fn rejects_incompatible_comparisons_and_duplicate_service_definitions() {
    let p = plan(vec![demand("memory", Count::Exact(100), Count::Exact(100))]);
    let a = predict(&p, &profile()).unwrap();
    let mut other = p.clone();
    other.workload_identity = "different-work".into();
    let b = predict(&other, &profile()).unwrap();
    assert!(unambiguous_choice(&[a, b]).is_err());
    let mut duplicate = profile();
    duplicate.behaviors.push(duplicate.behaviors[0].clone());
    assert!(predict(&p, &duplicate).is_err());
    let mut mismatch = profile();
    mismatch.identity = "different-driver".into();
    assert!(predict(&p, &mismatch).is_err());
    for (lo, hi) in [(f64::NAN, 1.), (-1., 1.), (2., 1.), (1., f64::INFINITY)] {
        assert!(Range::new(lo, hi).is_err());
    }
}
