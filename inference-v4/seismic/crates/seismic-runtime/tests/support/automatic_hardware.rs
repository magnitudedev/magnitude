//! Explicit hypothetical functional machines. Inventories come from typed
//! backend IR; no native candidate execution or measured timing supplies costs.
use seismic_accounting::{execution_model::*, schedule::*};
use seismic_realization::LoadStrategy;
use seismic_runtime::tuner::Hardware;

pub fn cpu(lowered: &seismic_lang::lowered_ir::LoweredIr) -> ScalarHardware {
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

pub fn cuda(
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
            // Populate the fixture hardware contract from backend IR only.
            // This does not compile or execute a candidate.
            let sequence = seismic_compiler::scalar_sequence(
                lowered,
                seismic_realization::CallConv::SystemV,
                seismic_realization::ScalarOptions { dispatch, loads },
            )
            .unwrap();
            let phases = sequence
                .phases
                .into_iter()
                .map(|phase| {
                    seismic_cuda::execution::Execution::new(
                        phase.program,
                        1,
                        seismic_cuda::execution::Limits {
                            max_threads_per_block: device.max_threads_per_block,
                            max_grid_x: device.max_grid_x,
                        },
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>();
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

#[cfg(target_os = "macos")]
pub fn metal(lowered: &seismic_lang::lowered_ir::LoweredIr) -> Hardware {
    use seismic_metal::{execution, model};
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
    Hardware::Metal(model::Hardware {
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
    })
}

/// Functional native tests use the same automatic-selection gate as callers.
/// Bindings precede selection; no executable is built from an incomplete result.
#[allow(dead_code)]
pub fn compile(
    device: &seismic_runtime::Device,
    input: seismic_runtime::tuner::Input<'_>,
    buffers: &[seismic_runtime::Buffer],
    scalars: &[f64],
) -> Result<seismic_runtime::Kernel, String> {
    compile_with_controls(device, input, buffers, scalars, &[])
}

/// Content bindings are explicit control inputs of the test case, never inferred
/// from a tensor's size, name or current values.
#[allow(dead_code)]
pub fn compile_with_controls(
    device: &seismic_runtime::Device,
    input: seismic_runtime::tuner::Input<'_>,
    buffers: &[seismic_runtime::Buffer],
    scalars: &[f64],
    controls: &[usize],
) -> Result<seismic_runtime::Kernel, String> {
    use seismic_runtime::tuner::{self, Input, Outcome, Request};
    let Input::Portable {
        program,
        entry,
        shapes,
        elements,
        options,
    } = input
    else {
        return Err("native test requires an unresolved portable source family".into());
    };
    // Only the fixture's instruction inventory and ABI use diagnostic lowering.
    // The complete source request, with all admitted choices, goes to selection.
    let lowered = seismic_lang::lower::lower_specialized(
        program,
        entry,
        device.backend(),
        shapes,
        elements,
        options,
    )?;
    let facts = device.facts();
    let settings = settings(device, &lowered);
    let parameters = lowered
        .params
        .iter()
        .filter_map(|(name, ty)| match ty {
            seismic_lang::types::Ty::Scalar(dtype) => Some(
                seismic_lang::abi::ScalarParameter::from_lowered(&lowered, name, *dtype),
            ),
            _ => None,
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut workload = tuner::workload(
        format!("automatic native {}", lowered.name),
        buffers,
        &parameters,
        scalars,
    )?;
    tuner::capture_contents(&mut workload, buffers, controls)?;
    let request = Request {
        input,
        device: &facts,
        form: settings.form,
        hardware: &settings.hardware,
        workload: &workload,
        derivation_limits: settings.derivation_limits,
    };
    match tuner::tune(&request, settings.search)? {
        Outcome::Optimal(selected) => device.compile_tuned(selected),
        Outcome::Incomplete(_) => Err(format!(
            "automatic native selection remains incomplete for {}",
            lowered.name
        )),
        Outcome::Infeasible => Err(format!(
            "automatic native selection found no feasible execution for {}",
            lowered.name
        )),
    }
}

#[allow(dead_code)]
pub fn settings(
    device: &seismic_runtime::Device,
    lowered: &seismic_lang::lowered_ir::LoweredIr,
) -> seismic_runtime::plan::Settings {
    use seismic_runtime::tuner::Form;
    let facts = device.facts();
    let (form, hardware) = match &facts {
        seismic_runtime::DeviceFacts::Cpu { .. } => (Form::CpuScalar, Hardware::Cpu(cpu(lowered))),
        seismic_runtime::DeviceFacts::Cuda(_) => (Form::CudaScalar, cuda(lowered, &facts)),
        #[cfg(target_os = "macos")]
        seismic_runtime::DeviceFacts::Metal(_) => (Form::Metal, metal(lowered)),
    };
    seismic_runtime::plan::Settings {
        form,
        hardware,
        derivation_limits: seismic_accounting::workload::DerivationLimits {
            instructions: 1_000_000,
            operations: 1_000_000,
        },
        search: seismic_runtime::tuner::Settings { limits: seismic_runtime::tuner::Limits { work: 100_000, ..Default::default() }, ..Default::default() },
    }
}
