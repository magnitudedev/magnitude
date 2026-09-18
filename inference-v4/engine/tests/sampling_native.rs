//! Functional automatic-selection test on an explicitly hypothetical machine.
//! Its unit service costs are never a production profile or performance evidence.
use seismic_accounting::{
    execution_model::*, schedule::*, selection::Budget, workload::DerivationLimits,
};
use seismic_engine::{
    generation::{
        sampling::{Sampler, Selection},
        Sampling,
    },
    inputs::TokenId,
};
use seismic_realization::LoadStrategy;
use seismic_runtime::{
    plan::Settings,
    tuner::{Form, Hardware},
    Device,
};
use std::collections::HashMap;

#[test]
#[ignore = "automatic CPU accounting cannot yet model data-dependent sampling branches; see bugs/26-09-18/seismic-automatic-sampling-control.md"]
fn automatic_cpu_selection_executes_device_sampling() {
    let program = seismic_std::program().unwrap();
    let lowered = seismic_lang::lower::lower(
        &program,
        "sample_rows",
        "cpu",
        &HashMap::from([("M".into(), 1), ("V".into(), 3)]),
    )
    .unwrap();
    let mut patterns = Vec::new();
    for loads in [
        LoadStrategy::Materialize,
        LoadStrategy::BorrowProvenReadOnly,
    ] {
        let prepared = seismic_cpu::prepare(&lowered, loads).unwrap();
        for primitive in requirements(&prepared).unwrap() {
            // Opaque math helpers are not hardware instructions. Leave their
            // implementation accounting unresolved instead of inventing costs.
            if matches!(primitive.kind, PrimitiveKind::Math(_)) {
                continue;
            }
            let pattern = primitive.signature();
            if !patterns.contains(&pattern) {
                patterns.push(pattern);
            }
        }
    }
    let hardware = ScalarHardware {
        identity: "hypothetical sampling fixture".into(),
        scope: Scope::HypotheticalDirectScalarV1,
        timebase: Timebase {
            seconds_numerator: 1,
            seconds_denominator: 1,
        },
        resources: vec![Resource {
            name: "service".into(),
            capacity: 1,
            unit: CapacityUnit::Slots,
        }],
        timings: patterns
            .into_iter()
            .map(|primitive| PrimitiveTiming {
                primitive,
                latency: 1,
                services: vec![Reservation {
                    resource: 0,
                    offset: 0,
                    duration: 1,
                    units: 1,
                }],
            })
            .collect(),
    };
    let device = Device::cpu();
    let mut sampler = Sampler::compile(
        &device,
        3,
        Settings {
            hardware: Hardware::Cpu(hardware),
            form: Form::CpuScalar,
            derivation_limits: DerivationLimits {
                instructions: 100_000,
                operations: 100_000,
            },
            search: Budget {
                nodes: 100,
                schedule_assignments: 100_000,
            },
        },
    )
    .unwrap();
    let logits = device
        .buffer_from(
            &[1f32, 3., 2.]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    assert_eq!(
        sampler
            .sample(&logits, None, Sampling::Greedy, 42, 0)
            .unwrap(),
        Selection::Token(TokenId(1))
    );
    assert_eq!(
        sampler
            .sample(&logits, Some(&[5]), Sampling::Greedy, 42, 0)
            .unwrap(),
        Selection::Token(TokenId(2))
    );
}
