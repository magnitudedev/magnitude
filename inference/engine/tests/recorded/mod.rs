//! Assessment setups on recorded machines (`seismic::fixtures::recorded`).

use magnitude_engine::{
    assessment::{AssessmentSetup, AssessmentStructure, Measurements},
    options::ModelPolicy,
};
use magnitude_executor::platform::{DeviceRequest, MemoryReserves};
use magnitude_scheduler::ServiceLimits;
use seismic::BackendName;
use std::sync::Arc;

/// The recorded M4 Pro (48 GB) under `reserves`, its Metal executor
/// selected, before any measurement. Metal's recorded schedules hold no
/// device memory.
pub fn m4_pro_setup(reserves: MemoryReserves, policy: ModelPolicy, service: ServiceLimits) -> AssessmentSetup {
    let topology = seismic::fixtures::recorded::m4_pro();
    recorded_setup(topology, BackendName::Metal, seismic::GraphCost::None, reserves, policy, service)
}

/// What graphs recorded on the RTX 3090 hold, by backend, as each device
/// measured it: CUDA's pooled driver memory (2026-11-02,
/// `seismic-cuda` `the_measured_cost_prices_a_class_of_graphs`), and
/// Vulkan's command memory, which the driver takes none of, and argument
/// chunks (2026-11-03, `seismic-native-tests`
/// `the_device_states_its_graph_cost`).
fn rtx3090_graph_cost(backend: BackendName) -> seismic::GraphCost {
    match backend {
        BackendName::Cuda => seismic::GraphCost::Pooled { graph: 92_663, launch: 8_102, granule: 2 << 20 },
        BackendName::Vulkan => seismic::GraphCost::Chunked { launch: 0, chunk: 256 << 10 },
        BackendName::Metal | BackendName::Cpu => panic!("the RTX 3090 host's graphs are CUDA's or Vulkan's"),
    }
}

/// The recorded RTX 3090 host under `reserves`, its `backend` executor
/// (CUDA or Vulkan) selected, before any measurement, its graphs priced as
/// the device measured them.
#[allow(dead_code)]
pub fn rtx3090_setup(backend: BackendName, reserves: MemoryReserves, policy: ModelPolicy, service: ServiceLimits) -> AssessmentSetup {
    let topology = seismic::fixtures::recorded::rtx3090();
    recorded_setup(topology, backend, rtx3090_graph_cost(backend), reserves, policy, service)
}

/// The recorded machine `topology` under `reserves`, its `backend`
/// executor selected with `graph_cost`, before any measurement.
fn recorded_setup(
    topology: seismic::DeviceTopology,
    backend: BackendName,
    graph_cost: seismic::GraphCost,
    reserves: MemoryReserves,
    policy: ModelPolicy,
    service: ServiceLimits,
) -> AssessmentSetup {
    let topology = Arc::new(topology);
    // Stable fit reads no headroom.
    let status = seismic::Status {
        sample: seismic::Sample {
            taken: std::time::SystemTime::now(),
            host_ram: Ok(seismic::HostStatus {
                allocatable: seismic::Reported::new(0, seismic::ReportedBy::MachAllocatablePages),
                obtainable: seismic::Reported::new(0, seismic::ReportedBy::MachMovablePages),
                limits: Vec::new(),
                limit_visibility: seismic::LimitVisibility::Complete,
                displacement: None,
                kernel_pressure: None,
                held: None,
                displaced_bytes: None,
            }),
            volumes: Vec::new(),
            executors: topology.executors().iter().map(|executor| executor.id()).collect(),
            grants: Vec::new(),
        },
        runtime: Default::default(),
    };
    let info = topology
        .executors()
        .iter()
        .find(|executor| executor.backend() == backend)
        .expect("the recorded machine has the backend's executor")
        .clone();
    let selected = magnitude_executor::platform::selected(&topology, &status, &reserves, info, magnitude_executor::platform::GraphPricing::Measured(graph_cost)).unwrap();
    AssessmentSetup {
        structure: Arc::new(AssessmentStructure {
            topology,
            status,
            device: DeviceRequest::Backend(backend),
            selected,
            reserves,
            policy,
            service,
        }),
        measurements: Measurements::Pending,
    }
}
