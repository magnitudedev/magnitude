//! Assessment setups on recorded machines (`seismic::fixtures::recorded`).

use magnitude_engine::{
    assessment::{AssessmentSetup, AssessmentStructure, Measurements},
    options::ModelPolicy,
};
use magnitude_executor::platform::{DeviceRequest, ReleaseRate};
use magnitude_scheduler::ServiceLimits;
use seismic::BackendName;
use std::sync::Arc;

/// The recorded M4 Pro (48 GB) under `release`, its Metal executor
/// selected, before any measurement. Metal's recorded schedules hold no
/// device memory.
pub fn m4_pro_setup(release: ReleaseRate, policy: ModelPolicy, service: ServiceLimits) -> AssessmentSetup {
    let topology = seismic::fixtures::recorded::m4_pro();
    recorded_setup(topology, BackendName::Metal, seismic::GraphCost::None, release, policy, service)
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

/// The recorded RTX 3090 host under `release`, its `backend` executor
/// (CUDA or Vulkan) selected, before any measurement, its graphs priced as
/// the device measured them.
#[allow(dead_code)]
pub fn rtx3090_setup(backend: BackendName, release: ReleaseRate, policy: ModelPolicy, service: ServiceLimits) -> AssessmentSetup {
    let topology = seismic::fixtures::recorded::rtx3090();
    recorded_setup(topology, backend, rtx3090_graph_cost(backend), release, policy, service)
}

/// [`rtx3090_setup`] of the RTX 3090 granting `backend`'s device storage
/// only `grant` bytes (`seismic::fixtures::rtx3090_granting`), the grant
/// stable fit plans against.
#[allow(dead_code)]
pub fn rtx3090_granting_setup(
    backend: BackendName,
    grant: u64,
    release: ReleaseRate,
    policy: ModelPolicy,
    service: ServiceLimits,
) -> AssessmentSetup {
    let topology = seismic::fixtures::topology(&seismic::fixtures::rtx3090_granting(backend, grant));
    recorded_setup(topology, backend, rtx3090_graph_cost(backend), release, policy, service)
}

/// The recorded machine `topology` under `release`, its `backend`
/// executor selected with `graph_cost`, before any measurement.
fn recorded_setup(
    topology: seismic::DeviceTopology,
    backend: BackendName,
    graph_cost: seismic::GraphCost,
    release: ReleaseRate,
    policy: ModelPolicy,
    service: ServiceLimits,
) -> AssessmentSetup {
    let topology = Arc::new(topology);
    // Stable fit reads no headroom: host RAM's binding scope is the whole
    // host, and the OS supplies nothing.
    let at_once = std::time::Duration::ZERO;
    let at = std::time::Instant::now();
    let host = seismic::HostSample {
        at,
        page_bytes: 4096,
        total: topology.host_ram().capacity().value,
        supply: seismic::Supply { free_beyond_target: 0, clean_cache: 0, cold_reclaimable: 0 },
        counted: seismic::Counted { at, lag: at_once },
        cold: seismic::ColdPages::default(),
        own_reclaimed: Some(seismic::OwnReclaimed::NONE),
        wired: None,
        harm: Some(seismic::Harm { machine: 0, own: 0 }),
        pressure: Some(seismic::PressureLevel::Normal),
        lag: seismic::Lag { release: at_once, growth: at_once },
    };
    let status = seismic::Status {
        sample: seismic::Sample {
            taken: std::time::SystemTime::now(),
            host,
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
    let selected = magnitude_executor::platform::selected(&topology, &status, &release, info, magnitude_executor::platform::GraphPricing::Measured(graph_cost));
    AssessmentSetup {
        structure: Arc::new(AssessmentStructure {
            topology,
            status,
            device: DeviceRequest::Backend(backend),
            selected,
            release,
            policy,
            service,
            served_context: None,
        }),
        measurements: Measurements::Pending(magnitude_executor::assessment::PendingPrices::standing(
            magnitude_executor::assessment::PointRequest::Costs(magnitude_executor::assessment::CostWalk::Required),
        )),
    }
}
