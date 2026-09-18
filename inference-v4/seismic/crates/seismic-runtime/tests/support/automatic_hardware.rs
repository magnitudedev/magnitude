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
                lowered, seismic_realization::CallConv::SystemV,
                seismic_realization::ScalarOptions { dispatch, loads },
            ).unwrap();
            let phases = sequence.phases.into_iter().map(|phase| {
                seismic_cuda::execution::Execution::new(phase.program, 1,
                    seismic_cuda::execution::Limits {
                        max_threads_per_block: device.max_threads_per_block,
                        max_grid_x: device.max_grid_x,
                    }).unwrap()
            }).collect::<Vec<_>>();
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
