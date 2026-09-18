//! The fixture machine is intentionally hypothetical. These tests verify the
//! source -> choices -> derived model -> checked selection -> native artifact
//! connection and applicability checks; they do not qualify physical CPU timing.
use seismic_accounting::{
    execution_model::*, schedule::*, selection::Budget, workload::DerivationLimits,
};
use seismic_lang::{
    program::{compile, SourceFile},
    Scope as LanguageScope,
};
use seismic_realization::LoadStrategy;
use seismic_runtime::{
    choices::Form,
    tuner::{self, Input, Outcome, Request},
    Device,
};
use std::collections::HashMap;

fn contract(lowered: &seismic_lang::lowered_ir::LoweredIr) -> ScalarHardware {
    let mut patterns = Vec::new();
    for loads in [
        LoadStrategy::Materialize,
        LoadStrategy::BorrowProvenReadOnly,
    ] {
        let p = seismic_cpu::prepare(lowered, loads).unwrap();
        for primitive in requirements(&p).unwrap() {
            let pattern = primitive.signature();
            if !patterns.contains(&pattern) {
                patterns.push(pattern);
            }
        }
    }
    ScalarHardware {
        identity: "hypothetical one-service scalar machine".into(),
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
    }
}
fn budget() -> Budget {
    Budget {
        nodes: 100,
        schedule_assignments: 100_000,
    }
}
#[test]
fn source_to_native_keeps_checked_model_and_enforces_workload_conditions() {
    let program = compile(&[SourceFile { path: "tuner.seismic.portable".into(), scope: LanguageScope::Portable,
        text: "fn copy(x: tensor[7] f32, out: tensor[7] f32, enabled: bool):\n  if enabled:\n    a = load(x)\n    b = a\n    store(b, out)\n".into() }], &[]).unwrap();
    let shapes = HashMap::new();
    let lowered = seismic_lang::lower::lower(&program, "copy", "cpu", &shapes).unwrap();
    let model = contract(&lowered);
    let device = Device::cpu();
    let hardware = device.facts();
    let input: Vec<_> = (0..7).flat_map(|n| (n as f32).to_le_bytes()).collect();
    let source = device.buffer_from(&input).unwrap();
    let target = device.buffer(input.len()).unwrap();
    let schema = seismic_cpu::prepare(&lowered, LoadStrategy::Materialize)
        .unwrap()
        .scalars;
    let bindings = vec![source.clone(), target.clone()];
    let workload = tuner::workload(
        "fixed scalars and allocation geometry",
        &bindings,
        &schema,
        &[1.0],
    )
    .unwrap();
    let elements = HashMap::new();
    let options = seismic_lang::lower::Options::default();
    let request = Request {
        input: Input::Portable {
            program: &program,
            entry: "copy",
            shapes: &shapes,
            elements: &elements,
            options: &options,
        },
        device: &hardware,
        form: Form::CpuScalar,
        hardware: &model,
        workload: &workload,
        derivation_limits: DerivationLimits {
            instructions: 10_000,
            operations: 30_000,
        },
    };
    let Outcome::Optimal(tuned) = tuner::tune(&request, budget()).unwrap() else {
        panic!("expected exact conditional optimum")
    };
    assert_eq!(
        tuned.conditions().hardware.scope,
        Scope::HypotheticalDirectScalarV1
    );
    assert!(tuned.modeled_cost().is_exact());
    let selected = tuned.execution();
    assert!(selected
        .loads
        .iter()
        .all(|d| d.mode == seismic_lang::ir::LoadMode::Borrow));
    let emitted_order = seismic_realization::scheduling::Order::current(selected);
    assert_eq!(
        emitted_order.blocks,
        static_order::orders(tuned.model(), tuned.schedule()).unwrap()
    );
    let selected_analysis = tuned.model().clone();
    let selected_schedule = tuned.schedule().clone();
    let mut kernel = device.compile_tuned(tuned).unwrap();
    let retained = kernel.tuning().unwrap();
    assert_eq!(retained.model(), &selected_analysis);
    assert_eq!(retained.schedule(), &selected_schedule);
    assert!(retained.modeled_cost().is_exact());
    kernel.execute(&bindings, &[1.0]).unwrap();
    let mut output = vec![0; input.len()];
    target.read(&mut output).unwrap();
    assert_eq!(output, input);
    assert!(kernel
        .execute(&bindings, &[0.0])
        .unwrap_err()
        .contains("scalar bindings"));
    assert!(kernel
        .execute(&[source.clone(), source], &[1.0])
        .unwrap_err()
        .contains("alias relationships"));
    let larger = device
        .buffer(input.len() + 4)
        .unwrap()
        .view(4..input.len() + 4)
        .unwrap();
    assert!(kernel
        .execute(&[larger, target], &[1.0])
        .unwrap_err()
        .contains("allocation/view"));
}
#[test]
fn nested_views_derive_canonical_alias_geometry_and_resume_binds_all_inputs() {
    let program = compile(
        &[SourceFile {
            path: "copy.seismic.portable".into(),
            scope: LanguageScope::Portable,
            text:
                "fn copy(x: tensor[2] f32, out: tensor[2] f32):\n  a = load(x)\n  store(a, out)\n"
                    .into(),
        }],
        &[],
    )
    .unwrap();
    let lowered = seismic_lang::lower::lower(&program, "copy", "cpu", &HashMap::new()).unwrap();
    let model = contract(&lowered);
    let device = Device::cpu();
    let hardware = device.facts();
    let allocation = device.buffer(32).unwrap();
    let a = allocation.view(4..24).unwrap().view(4..12).unwrap();
    let b = allocation.view(20..28).unwrap();
    assert_eq!(a.allocation_offset(), 8);
    let workload = tuner::workload("views", &[a, b], &[], &[]).unwrap();
    assert_eq!(workload.allocations.len(), 1);
    assert_eq!(
        workload.buffers[0].allocation,
        workload.buffers[1].allocation
    );
    assert_eq!(workload.buffers[0].offset, 8);
    let request = Request {
        input: Input::Lowered(&lowered),
        device: &hardware,
        form: Form::CpuScalar,
        hardware: &model,
        workload: &workload,
        derivation_limits: DerivationLimits {
            instructions: 10_000,
            operations: 30_000,
        },
    };
    let mut limited = budget();
    limited.nodes = 0;
    let Outcome::Incomplete(progress) = tuner::tune(&request, limited).unwrap() else {
        panic!()
    };
    assert_eq!(
        progress.frontier().iter().map(|r| r.len()).sum::<usize>(),
        1
    );
    let Outcome::Optimal(tuned) = tuner::resume(&request, progress, budget()).unwrap() else {
        panic!()
    };
    let Outcome::Incomplete(second) = tuner::tune(&request, limited).unwrap() else {
        panic!()
    };
    let mut changed = model.clone();
    changed.resources[0].capacity = 2;
    let altered = Request {
        hardware: &changed,
        ..request
    };
    assert!(
        matches!(tuner::resume(&altered,second,budget()),Err(error) if error.contains("inputs changed"))
    );
    assert!(tuned.modeled_cost().is_exact());
}
