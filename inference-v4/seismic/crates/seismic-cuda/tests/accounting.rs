use cranelift_codegen::isa::CallConv;
use seismic_accounting::{schedule, workload};
use seismic_cuda::{execution, model};
use seismic_lang::program::{SourceFile, compile};
use std::collections::HashMap;

fn execution(expression: &str) -> execution::Execution {
    let source = format!(
        "fn kernel(x: tensor[4] f32, y: tensor[4] f32):\n  for i in parallel:\n    a = tile[1] f32\n    for j in owned(a): a[j] = {expression}\n    store(a,y[i:i+1])\n"
    );
    let program = compile(
        &[SourceFile {
            path: "accounting.seismic.portable".into(),
            scope: seismic_lang::Scope::Portable,
            text: source,
        }],
        &[],
    )
    .unwrap();
    let lowered = seismic_lang::lower::lower(&program, "kernel", "cuda", &HashMap::new()).unwrap();
    let scalar = seismic_compiler::scalar_with(
        &lowered,
        CallConv::SystemV,
        seismic_realization::Dispatch::ParallelRoot,
    )
    .unwrap();
    execution::Execution::new(
        scalar,
        32,
        execution::Limits {
            max_threads_per_block: 1024,
            max_grid_x: u32::MAX,
        },
    )
    .unwrap()
}

fn hardware(execution: &execution::Execution) -> model::CudaHardware {
    model::CudaHardware {
        identity: "synthetic PTX services".into(),
        scope: model::Scope::HypotheticalInstructionPreservingPtxV1,
        timebase: schedule::Timebase {
            seconds_numerator: 1,
            seconds_denominator: 1_000_000_000,
        },
        execution_units: 1,
        warp_width: 32,
        cohorts: model::CohortPolicy::LowestPosition,
        internal_alignment: 256,
        resources: vec![model::Resource {
            name: "issue".into(),
            scope: model::ResourceScope::Device,
            capacity: 1,
            unit: schedule::CapacityUnit::Slots,
        }],
        timings: model::requirements(execution)
            .into_iter()
            .filter_map(|requirement| {
                let model::Requirement::Instruction(primitive) = requirement else {
                    return None;
                };
                Some(model::PrimitiveTiming {
                    primitive,
                    latency: model::Ticks::Fixed(1),
                    reservations: vec![model::Reservation {
                        resource: 0,
                        offset: 0,
                        duration: model::Ticks::Fixed(1),
                        units: model::Amount::fixed(1),
                    }],
                })
            })
            .collect(),
        block_residency: vec![],
        per_unit_residency: vec![model::UnitResidency {
            name: "resident blocks".into(),
            capacity: 2,
            units_per_block: model::Amount::fixed(1),
        }],
    }
}

fn workload(execution: &execution::Execution) -> workload::ScalarWorkload {
    workload::ScalarWorkload { integer_domains: Vec::new(),
        identity: "four unknown f32 inputs".into(),
        allocations: execution
            .program()
            .buffers
            .iter()
            .enumerate()
            .map(|(id, b)| workload::Allocation {
                id: id as u64,
                bytes: b.bytes as u64,
                alignment: 256,
                known_bytes: Default::default(),
            })
            .collect(),
        buffers: execution
            .program()
            .buffers
            .iter()
            .enumerate()
            .map(|(id, b)| workload::BufferBinding {
                allocation: id as u64,
                offset: 0,
                bytes: b.bytes as u64,
            })
            .collect(),
        scalars: vec![],
    }
}

const LIMITS: workload::DerivationLimits = workload::DerivationLimits {
    instructions: 100_000,
    operations: 100_000,
};

#[test]
fn hardware_services_do_not_author_the_execution_graph() {
    let execution = execution("x[i] + 1.0");
    let hardware = hardware(&execution);
    let workload = workload(&execution);
    let derived = model::derive_cuda(
        &execution,
        &hardware,
        &workload,
        &model::Placement::HomogeneousResidentSlots,
        LIMITS,
    )
    .unwrap();
    assert!(std::ptr::eq(derived.implementation(), &execution));
    assert!(std::ptr::eq(derived.hardware(), &hardware));
    assert!(std::ptr::eq(derived.workload(), &workload));
    assert_eq!(derived.events.len(), derived.model.operations.len());
    assert!(
        derived
            .events
            .iter()
            .any(|event| !event.accesses.is_empty())
    );
    for (event, operation) in derived.events.iter().zip(&derived.model.operations) {
        assert_eq!(operation.predecessors, event.predecessors);
        assert_eq!(operation.start_predecessors, event.start_predecessors);
        if let model::Requirement::Lifecycle(_) = event.requirement {
            assert_eq!(operation.latency, 0);
            assert!(operation.reservations.is_empty());
        }
    }
    let mut slower = hardware.clone();
    for timing in &mut slower.timings {
        timing.latency = model::Ticks::Fixed(3);
    }
    let changed = model::derive_cuda(
        &execution,
        &slower,
        &workload,
        &model::Placement::HomogeneousResidentSlots,
        LIMITS,
    )
    .unwrap();
    assert_eq!(derived.events, changed.events);
    assert_eq!(derived.model.lifetimes, changed.model.lifetimes);
    assert!(changed.model.lower_bound().unwrap() > derived.model.lower_bound().unwrap());
}

#[test]
fn opaque_helper_cannot_be_replaced_with_an_authored_cost() {
    let execution = execution("exp(x[i])");
    let hardware = hardware(&execution);
    let workload = workload(&execution);
    let before = seismic_cuda::ptx::print(execution.target_plan());
    let error = match model::derive_cuda(
        &execution,
        &hardware,
        &workload,
        &model::Placement::HomogeneousResidentSlots,
        LIMITS,
    ) {
        Ok(_) => panic!("unexpanded math helper acquired an execution cost"),
        Err(error) => error,
    };
    assert!(
        error.to_string().contains("expanded into PTX operations"),
        "{error}"
    );
    assert_eq!(before, seismic_cuda::ptx::print(execution.target_plan()));
}

#[test]
fn missing_hardware_service_is_an_explicitly_unsupported_analysis() {
    let execution = execution("x[i] + 1.0");
    let mut hardware = hardware(&execution);
    hardware.timings.pop().unwrap();
    let workload = workload(&execution);
    let error = match model::derive_cuda(
        &execution,
        &hardware,
        &workload,
        &model::Placement::HomogeneousResidentSlots,
        LIMITS,
    ) {
        Ok(_) => panic!("missing primitive timing silently disappeared"),
        Err(error) => error,
    };
    assert!(matches!(error, workload::DerivationError::Unsupported(ref reason)
        if reason.contains("missing CUDA hardware timing")), "{error}");
}

#[test]
fn construction_limits_are_typed_separately_from_analysis_errors() {
    let execution = execution("x[i] + 1.0");
    let hardware = hardware(&execution);
    let workload = workload(&execution);
    for (limits, expected) in [
        (
            workload::DerivationLimits {
                instructions: 1,
                operations: 100_000,
            },
            workload::DerivationLimit::Instructions(1),
        ),
        (
            workload::DerivationLimits {
                instructions: 100_000,
                operations: 1,
            },
            workload::DerivationLimit::Operations(1),
        ),
    ] {
        let Err(error) = model::derive_cuda(
            &execution,
            &hardware,
            &workload,
            &model::Placement::HomogeneousResidentSlots,
            limits,
        ) else {
            panic!("construction unexpectedly fit the limit");
        };
        assert_eq!(error, workload::DerivationError::Exhausted(expected));
    }
}
