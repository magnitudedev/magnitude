//! The fixture machine is intentionally hypothetical. These tests verify the
//! source -> choices -> derived model -> checked selection -> native artifact
//! connection and applicability checks; they do not qualify physical machine timing.
use seismic_accounting::{
    execution_model::*, schedule::*, selection::Budget, workload::DerivationLimits,
};
use seismic_lang::{
    Scope as LanguageScope,
    program::{SourceFile, compile},
};
use seismic_realization::LoadStrategy;
use seismic_runtime::{
    Device,
    execution::Execution,
    tuner::{self, Form, Hardware, ImplementationConditions, Input, Outcome, Request},
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
fn model_construction_limits_are_resumable_without_changing_execution_identity() {
    let program = compile(&[SourceFile {
        path: "budget.seismic.portable".into(), scope: LanguageScope::Portable,
        text: "fn copy(x:tensor[2] f32,out:tensor[2] f32):\n  values = load(x)\n  store(values,out)\n".into(),
    }], &[]).unwrap();
    let lowered = seismic_lang::lower::lower(&program, "copy", "cpu", &HashMap::new()).unwrap();
    let hardware = Hardware::Cpu(contract(&lowered));
    let device = Device::cpu();
    let facts = device.facts();
    let input = device
        .buffer_from(&[1f32.to_le_bytes(), 2f32.to_le_bytes()].concat())
        .unwrap();
    let output = device.buffer(8).unwrap();
    let workload = tuner::workload("copy", &[input, output], &[], &[]).unwrap();
    for limits in [
        DerivationLimits {
            instructions: 1,
            operations: 10_000,
        },
        DerivationLimits {
            instructions: 10_000,
            operations: 1,
        },
    ] {
        let mut request = Request {
            input: Input::Lowered(&lowered),
            device: &facts,
            form: Form::CpuScalar,
            hardware: &hardware,
            workload: &workload,
            derivation_limits: limits,
        };
        let Outcome::Incomplete(progress) = tuner::tune(&request, budget()).unwrap() else {
            panic!("model construction budget cannot establish an optimum or infeasibility");
        };
        assert!(progress.frontier().is_empty());
        assert!(progress.feasible_upper().is_none());
        assert!(progress.unresolved().derivations > 0);
        assert_eq!(
            progress.exhausted_derivations().count(),
            progress.unresolved().derivations
        );
        request.derivation_limits = DerivationLimits {
            instructions: 10_000,
            operations: 10_000,
        };
        let Outcome::Optimal(resumed) = tuner::resume(&request, progress, budget()).unwrap() else {
            panic!("larger construction limits should resolve the retained executions");
        };
        let Outcome::Optimal(uninterrupted) = tuner::tune(&request, budget()).unwrap() else {
            panic!()
        };
        assert_eq!(resumed.modeled_cost(), uninterrupted.modeled_cost());
        assert_eq!(resumed.selected_path(), uninterrupted.selected_path());
        assert_eq!(resumed.model(), uninterrupted.model());
    }
}

#[test]
fn source_to_native_keeps_checked_model_and_enforces_workload_conditions() {
    let program = compile(&[SourceFile { path: "tuner.seismic.portable".into(), scope: LanguageScope::Portable,
        text: "fn copy(x: tensor[7] f32, out: tensor[7] f32, enabled: bool):\n  if enabled:\n    a = load(x)\n    b = a\n    store(b, out)\n".into() }], &[]).unwrap();
    let shapes = HashMap::new();
    let lowered = seismic_lang::lower::lower(&program, "copy", "cpu", &shapes).unwrap();
    let model = Hardware::Cpu(contract(&lowered));
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
    let ImplementationConditions::Cpu(conditions) = tuned.conditions().implementation() else {
        panic!("CPU conditions")
    };
    assert_eq!(conditions.hardware.scope, Scope::HypotheticalDirectScalarV1);
    assert!(tuned.modeled_cost().is_exact());
    let Execution::Cpu(selected) = tuned.execution() else {
        panic!("CPU execution")
    };
    assert!(
        selected
            .loads
            .iter()
            .all(|d| d.mode == seismic_lang::ir::LoadMode::Borrow)
    );
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
    assert!(
        kernel
            .execute(&bindings, &[0.0])
            .unwrap_err()
            .contains("scalar bindings")
    );
    assert!(
        kernel
            .execute(&[source.clone(), source], &[1.0])
            .unwrap_err()
            .contains("alias relationships")
    );
    let larger = device
        .buffer(input.len() + 4)
        .unwrap()
        .view(4..input.len() + 4)
        .unwrap();
    assert!(
        kernel
            .execute(&[larger, target], &[1.0])
            .unwrap_err()
            .contains("allocation/view")
    );
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
    let model = Hardware::Cpu(contract(&lowered));
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
    let Hardware::Cpu(hardware) = &mut changed else {
        unreachable!()
    };
    hardware.resources[0].capacity = 2;
    let altered = Request {
        hardware: &changed,
        ..request
    };
    assert!(
        matches!(tuner::resume(&altered,second,budget()),Err(error) if error.contains("inputs changed"))
    );
    assert!(tuned.modeled_cost().is_exact());
}

fn gpu_program() -> seismic_lang::program::Program {
    compile(&[SourceFile { path: "gpu-tuner.seismic.portable".into(), scope: LanguageScope::Portable,
        text: "fn copy(x: tensor[1] f32, out: tensor[1] f32, enabled: bool):\n  if enabled:\n    value = load(x)\n    store(value, out)\n".into() }], &[]).unwrap()
}
fn cuda_hardware(
    lowered: &seismic_lang::lowered_ir::LoweredIr,
    facts: &seismic_runtime::DeviceFacts,
) -> Hardware {
    use seismic_cuda::model as cuda;
    let seismic_runtime::DeviceFacts::Cuda(device) = facts else {
        panic!("CUDA device")
    };
    let mut primitives = Vec::new();
    for loads in [
        LoadStrategy::Materialize,
        LoadStrategy::BorrowProvenReadOnly,
    ] {
        for dispatch in [
            seismic_realization::Dispatch::Sequential,
            seismic_realization::Dispatch::ParallelRoot,
        ] {
            let Execution::Cuda(phases) = Execution::prepare(
                lowered,
                seismic_runtime::Candidate::Cuda {
                    options: seismic_realization::ScalarOptions { dispatch, loads },
                    threads_per_block: 1,
                },
                facts,
            )
            .unwrap() else {
                panic!("CUDA execution")
            };
            for phase in &phases {
                for required in cuda::requirements(phase) {
                    if let cuda::Requirement::Instruction(primitive) = required {
                        if !primitives.contains(&primitive) {
                            primitives.push(primitive);
                        }
                    }
                }
            }
        }
    }
    Hardware::Cuda(cuda::CudaHardware {
        identity: "hypothetical terminal instruction service".into(),
        scope: cuda::Scope::HypotheticalInstructionPreservingPtxV1,
        timebase: Timebase {
            seconds_numerator: 1,
            seconds_denominator: 1,
        },
        execution_units: device.multiprocessors as usize,
        warp_width: device.warp_size,
        cohorts: cuda::CohortPolicy::LowestPosition,
        internal_alignment: 256,
        resources: vec![cuda::Resource {
            name: "issue".into(),
            scope: cuda::ResourceScope::Device,
            capacity: 1,
            unit: CapacityUnit::Slots,
        }],
        timings: primitives
            .into_iter()
            .map(|primitive| cuda::PrimitiveTiming {
                primitive,
                latency: cuda::Ticks::Fixed(1),
                reservations: vec![cuda::Reservation {
                    resource: 0,
                    offset: 0,
                    duration: cuda::Ticks::Fixed(1),
                    units: cuda::Amount::fixed(1),
                }],
            })
            .collect(),
        block_residency: vec![],
        per_unit_residency: vec![cuda::UnitResidency {
            name: "resident blocks".into(),
            capacity: 1,
            units_per_block: cuda::Amount::fixed(1),
        }],
    })
}
#[test]
fn cuda_uses_compiler_selection_and_rejects_a_different_native_device() {
    let program = gpu_program();
    let lowered = seismic_lang::lower::lower(&program, "copy", "cuda", &HashMap::new()).unwrap();
    let facts = seismic_runtime::DeviceFacts::Cuda(seismic_cuda::DeviceInfo {
        name: "one-thread hypothetical device".into(),
        compute_capability: (8, 0),
        driver_version: 0,
        max_threads_per_block: 1,
        max_grid_x: 1,
        warp_size: 32,
        multiprocessors: 1,
        global_memory_bytes: 1024,
        l2_cache_bytes: 0,
        max_threads_per_multiprocessor: 32,
        registers_32bit_per_multiprocessor: 65536,
        shared_bytes_per_multiprocessor: 65536,
    });
    let hardware = cuda_hardware(&lowered, &facts);
    // Only geometry is used by the IR-only compiler: it needs no GPU or native
    // candidate. Actual device compatibility is established at native binding.
    let host = Device::cpu();
    let schema = vec![seismic_lang::abi::ScalarParameter::plain(
        "enabled",
        seismic_lang::types::DType::Bool,
    )];
    let workload = tuner::workload(
        "two separate words",
        &[host.buffer(4).unwrap(), host.buffer(4).unwrap()],
        &schema,
        &[1.0],
    )
    .unwrap();
    let request = Request {
        input: Input::Lowered(&lowered),
        device: &facts,
        form: Form::CudaScalar,
        hardware: &hardware,
        workload: &workload,
        derivation_limits: DerivationLimits {
            instructions: 100_000,
            operations: 100_000,
        },
    };
    let Outcome::Optimal(tuned) = tuner::tune(&request, budget()).unwrap() else {
        panic!("CUDA conditional optimum")
    };
    let Execution::Cuda(phases) = tuned.execution() else {
        panic!("selected CUDA execution")
    };
    assert_eq!(phases.len(), 1);
    assert_eq!(tuned.conditions().device(), &facts);
    assert!(matches!(
        tuned.conditions().implementation(),
        ImplementationConditions::Cuda(_)
    ));
    assert!(tuned.model().unmapped.is_empty());
    assert!(tuned.modeled_cost().is_exact());
    assert!(matches!(host.compile_tuned(tuned), Err(e) if e.contains("device differs")));
}
#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_tuned_native_artifact_retains_conditions_and_enforces_bindings() {
    let device = Device::cuda(0).unwrap();
    let facts = device.facts();
    let program = gpu_program();
    let lowered = seismic_lang::lower::lower(&program, "copy", "cuda", &HashMap::new()).unwrap();
    let hardware = cuda_hardware(&lowered, &facts);
    let schema = vec![seismic_lang::abi::ScalarParameter::plain(
        "enabled",
        seismic_lang::types::DType::Bool,
    )];
    let source = device.buffer_from(&7f32.to_le_bytes()).unwrap();
    let target = device.buffer(4).unwrap();
    let bindings = vec![source.clone(), target.clone()];
    let workload = tuner::workload("native CUDA copy", &bindings, &schema, &[1.0]).unwrap();
    let request = Request {
        input: Input::Lowered(&lowered),
        device: &facts,
        form: Form::CudaScalar,
        hardware: &hardware,
        workload: &workload,
        derivation_limits: DerivationLimits {
            instructions: 1_000_000,
            operations: 1_000_000,
        },
    };
    let Outcome::Optimal(tuned) = tuner::tune(
        &request,
        Budget {
            nodes: 20_000,
            schedule_assignments: 100_000,
        },
    )
    .unwrap() else {
        panic!("CUDA conditional optimum")
    };
    let selected_model = tuned.model().clone();
    let mut kernel = device.compile_tuned(tuned).unwrap();
    assert_eq!(kernel.tuning().unwrap().model(), &selected_model);
    kernel.execute(&bindings, &[1.0]).unwrap();
    let mut result = [0; 4];
    target.read(&mut result).unwrap();
    assert_eq!(result, 7f32.to_le_bytes());
    assert!(
        kernel
            .execute(&bindings, &[0.0])
            .unwrap_err()
            .contains("scalar bindings")
    );
    assert!(
        kernel
            .execute(&[source.clone(), source], &[1.0])
            .unwrap_err()
            .contains("alias relationships")
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn metal_tuned_native_artifact_preserves_publication_and_binding_conditions() {
    use seismic_metal::{execution, model, tuning as metal};
    let program = compile(&[SourceFile { path: "metal-tuner.seismic.portable".into(), scope: LanguageScope::Portable,
        text: "fn evaluate(out: tensor[2] f32):\n  y = tile[2] f32\n  for i in owned(y): y[i] = 3.0\n  store(y,out)\n".into() }], &[]).unwrap();
    let lowered =
        seismic_lang::lower::lower(&program, "evaluate", "metal", &HashMap::new()).unwrap();
    let device = Device::metal().unwrap();
    let facts = device.facts();
    let mut keys = Vec::new();
    // A synthetic reusable instruction service, independent of native feedback.
    // Inventory each storage implementation because its operations really differ.
    for placement in [
        seismic_realization::dispatch::TilePlacement::Replicated,
        seismic_realization::dispatch::TilePlacement::Distributed,
        seismic_realization::dispatch::TilePlacement::GroupShared,
    ] {
        let e = execution::prepare_storage_selected(
            &lowered,
            execution::Config::default(),
            &mut |_| Ok(placement.clone()),
        )
        .unwrap();
        let required = model::requirements(&e).unwrap();
        assert!(required.unmapped.is_empty(), "{:?}", required.unmapped);
        for key in required.primitives {
            if !keys.contains(&key) {
                keys.push(key);
            }
        }
    }
    let hardware = Hardware::Metal(model::Hardware {
        identity: "test pooled MSL service; native timing unqualified".into(),
        timebase: Timebase {
            seconds_numerator: 1,
            seconds_denominator: 1,
        },
        resources: vec![Resource {
            name: "service".into(),
            capacity: 1024,
            unit: CapacityUnit::Slots,
        }],
        resident_groups: 2,
        resident_shared_bytes: 65536,
        timings: keys
            .into_iter()
            .map(|primitive| model::Timing {
                primitive,
                latency: 1,
                services: vec![model::Service {
                    resource: 0,
                    offset: 0,
                    duration: 1,
                    units: model::Units::PerLane(1),
                }],
            })
            .collect(),
    });
    let backing = device.buffer_from(&[0xa5; 16]).unwrap();
    let out = backing.view(4..12).unwrap();
    let workload = tuner::workload("native Metal output", &[out.clone()], &[], &[]).unwrap();
    let request = Request {
        input: Input::Lowered(&lowered),
        device: &facts,
        form: Form::Metal(metal::Form::default()),
        hardware: &hardware,
        workload: &workload,
        derivation_limits: DerivationLimits {
            instructions: 1_000_000,
            operations: 1_000_000,
        },
    };
    let Outcome::Optimal(tuned) = tuner::tune(
        &request,
        Budget {
            nodes: 20_000,
            schedule_assignments: 100_000,
        },
    )
    .unwrap() else {
        panic!("Metal conditional optimum")
    };
    assert!(matches!(tuned.execution(), Execution::Metal(_)));
    let selected_model = tuned.model().clone();
    let mut kernel = device.compile_tuned(tuned).unwrap();
    assert_eq!(kernel.tuning().unwrap().model(), &selected_model);
    assert!(matches!(
        kernel.tuning().unwrap().conditions().implementation(),
        ImplementationConditions::Metal(_)
    ));
    kernel.execute(&[out], &[]).unwrap();
    let mut result = [0; 16];
    backing.read(&mut result).unwrap();
    assert_eq!(&result[..4], &[0xa5; 4]);
    assert_eq!(
        &result[4..12],
        &[3f32.to_le_bytes(), 3f32.to_le_bytes()].concat()
    );
    assert_eq!(&result[12..], &[0xa5; 4]);
    assert!(
        kernel
            .execute(&[backing.view(0..8).unwrap()], &[])
            .unwrap_err()
            .contains("allocation/view")
    );
}
