use seismic_estimator::*;
use seismic_lang::expr::{Assignment, ExprArena};

struct Model(ServiceDefinition);
impl ServiceModel for Model {
    fn service(&self, class: ServiceClassId) -> &ServiceDefinition {
        assert_eq!(class, self.0.class);
        &self.0
    }
    fn maximum_relative_error_basis_points(&self) -> u16 {
        1000
    }
}
fn model() -> Model {
    Model(ServiceDefinition {
        class: ServiceClassId::new("fixture.compute"),
        correlation: ServiceCorrelationId::new("fixture.shared-coefficient"),
        qualification: ServiceQualificationDomain {
            minimum_units: 1,
            maximum_units: 16,
            maximum_concurrent_uses: 1,
        },
        accuracy: ServiceAccuracyClass::Compute,
        topology: ResourceTopology {
            resources: 2,
            max_concurrency: 2,
        },
        dependency_latency: DurationInterval::new(7, 7, 1),
        saturated_capacity: ServiceCurve {
            setup: DurationInterval::new(3, 3, 1),
            regimes: vec![
                ServiceCurveRegime {
                    max_units: Some(8),
                    per_unit: DurationInterval::new(2, 2, 1),
                },
                ServiceCurveRegime {
                    max_units: None,
                    per_unit: DurationInterval::new(5, 5, 1),
                },
            ],
        },
        provenance: FactProvenance::Derived {
            rule: "exact synthetic fixture",
            inputs: Box::new([]),
        },
    })
}

#[test]
fn symbolic_cost_matches_independent_oracle_across_regime_and_capacity_boundaries() {
    let model = model();
    let service = ServiceClassId::new("fixture.compute");
    for units in 0_u64..=24 {
        for mode in [DemandMode::DependencyLatency, DemandMode::SaturatedCapacity] {
            let mut arena = ExprArena::default();
            let demand = arena.nat(units);
            let contribution = service_contribution(
                &model,
                &mut arena,
                service,
                demand,
                mode,
                InvocationProvenance::KernelLaunch { ordinal: 2 },
            )
            .unwrap();
            let estimate = arena
                .eval_duration(contribution.duration(), &Assignment::new())
                .unwrap();
            let expected = match mode {
                DemandMode::DependencyLatency => units * 7,
                DemandMode::SaturatedCapacity => {
                    3 + units.div_ceil(4) * if units <= 8 { 2 } else { 5 }
                }
            } as u128;
            assert_eq!(
                estimate.lower().numerator() * 10,
                expected * 9 * u128::from(estimate.lower().denominator())
            );
            assert_eq!(
                estimate.upper().numerator() * 10,
                expected * 11 * u128::from(estimate.upper().denominator())
            );
            assert_eq!(contribution.correlation(), model.0.correlation);
            assert_eq!(
                contribution.region().invocation(),
                InvocationProvenance::KernelLaunch { ordinal: 2 }
            );
            assert!(contribution.region().guards().is_empty());
            assert_eq!(contribution.evidence(), &model.0.provenance);
        }
    }
}

#[test]
fn service_contribution_reports_invalid_envelope_overflow_and_zero_capacity() {
    struct CustomModel {
        service: ServiceDefinition,
        envelope_bps: u16,
    }
    impl ServiceModel for CustomModel {
        fn service(&self, class: ServiceClassId) -> &ServiceDefinition {
            assert_eq!(class, self.service.class);
            &self.service
        }
        fn maximum_relative_error_basis_points(&self) -> u16 {
            self.envelope_bps
        }
    }

    let service_id = ServiceClassId::new("fixture.compute");
    let mut arena = ExprArena::default();
    let demand = arena.nat(4);

    // 1. Envelope > 10_000 bps must return Err instead of underflowing.
    let bad_envelope = CustomModel {
        service: model().0,
        envelope_bps: 10_001,
    };
    assert_eq!(
        service_contribution(
            &bad_envelope,
            &mut arena,
            service_id,
            demand,
            DemandMode::DependencyLatency,
            InvocationProvenance::HostSchedule,
        )
        .unwrap_err(),
        ModelLimitation::InvalidErrorEnvelope {
            basis_points: 10_001,
        }
    );

    // 2. Denominator overflow in checked_mul(10_000) must return Err instead of panicking.
    let mut overflow_def = model().0;
    overflow_def.dependency_latency = DurationInterval::new(1, 1, u64::MAX);
    let overflow_model = CustomModel {
        service: overflow_def,
        envelope_bps: 100,
    };
    assert_eq!(
        service_contribution(
            &overflow_model,
            &mut arena,
            service_id,
            demand,
            DemandMode::DependencyLatency,
            InvocationProvenance::HostSchedule,
        )
        .unwrap_err(),
        ModelLimitation::DurationArithmeticOverflow {
            service: service_id,
        }
    );

    // 3. Zero capacity (resources == 0 or max_concurrency == 0) must return Err.
    let mut zero_cap_def = model().0;
    zero_cap_def.topology = ResourceTopology {
        resources: 0,
        max_concurrency: 2,
    };
    let zero_cap_model = CustomModel {
        service: zero_cap_def,
        envelope_bps: 100,
    };
    assert_eq!(
        service_contribution(
            &zero_cap_model,
            &mut arena,
            service_id,
            demand,
            DemandMode::SaturatedCapacity,
            InvocationProvenance::HostSchedule,
        )
        .unwrap_err(),
        ModelLimitation::ZeroServiceCapacity {
            service: service_id,
        }
    );
}
