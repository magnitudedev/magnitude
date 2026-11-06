use super::policy::{DomainRole, Fit, HostRoom, LoadStage, MemoryPolicyError, MemoryReserves, refresh_device_ceiling};
use super::selection::{DeviceRequest, SelectionError, select};
use crate::planning::HostTier;
use crate::{CatalogError, ExecutionPath};
use seismic::{
    ArtifactStore, Device, DeviceOptions, Devices, Executor, ExecutorSelector, OpenError,
    ResolveError, Contract,
};
use std::fmt;
use std::sync::Arc;

#[derive(Clone)]
pub struct PlatformConfig {
    pub path: ExecutionPath,
    /// Where the device keeps formed kernels between loads.
    pub artifacts: Option<Arc<dyn ArtifactStore>>,
    /// What its graphs hold, as the selection read it; none to measure it
    /// on opening.
    pub graph_cost: Option<seismic::GraphCost>,
    /// The host's threshold policy the opened device's ceiling keeps.
    pub reserves: MemoryReserves,
}

/// The automatically selected device and its stable assessment capacity.
/// Planning input only: it reserves nothing and guarantees no later load.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectedDevice {
    /// The executor, whose reported capabilities graphs are planned and
    /// assessed for (`Executor::forms_tensor_operations`).
    pub info: Executor,
    /// The executor's identity in the topology it was selected from, which
    /// keys what is measured of it.
    pub identity: seismic::Identity,
    /// Device storage's stable fit: what its placement holds across its
    /// stores, each within process limits and grants, less its planning
    /// reserve (`Fit::placement`).
    pub assessment_capacity_bytes: u64,
    /// How an expert bank's host tier is reserved on this device, where it
    /// has one, or why it has none ([`host_tier`]).
    pub host_tier: Result<Option<HostTier>, String>,
    /// What the selection's fit questions read, so a plan asks `Fit` of a
    /// demand again: whether a linked host tier's size fits beside the
    /// load's other host RAM demand (`ExecutionPlanDraft::with_host_tier`).
    pub reading: FitReading,
    /// What a plan prices the executor's recorded graphs at
    /// ([`GraphPricing`]).
    pub graph_cost: GraphPricing,
}

/// What a selected executor's recorded graphs are priced at for their
/// census: the device's own cost, or its backend's bound until that is
/// measured. It serializes as which of the two it is and the cost, which is
/// how it enters an identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "pricing", content = "cost", rename_all = "snake_case")]
pub enum GraphPricing {
    /// As its backend measured it (`seismic::Devices::graph_cost`), on
    /// opening the device or by the measurement job
    /// (`assessment::GraphMemory`).
    Measured(seismic::GraphCost),
    /// Its backend's bound (`seismic::unmeasured_graph_cost`), until the
    /// measurement job stores the device's own: a load requests that of the
    /// job (`assessment::PointRequest::GraphCost`), and later loads read it.
    Bound(seismic::GraphCost),
}

impl GraphPricing {
    /// The cost a plan charges its graphs at.
    pub fn cost(self) -> seismic::GraphCost {
        match self {
            Self::Measured(cost) | Self::Bound(cost) => cost,
        }
    }
}

/// The stores, grants and reserves a fit question reads, as one reading
/// of them found: `Fit`'s inputs, which alone interprets them, kept for a
/// later question of the same reading.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FitReading {
    pub topology: seismic::DeviceTopology,
    pub status: seismic::Status,
    pub reserves: MemoryReserves,
}

/// How an expert bank's host tier is held on `executor`: where reserved
/// `Shared` ranges lie in the store of its reserved `DeviceOnly` ranges
/// (unified memory), the tier is the bank itself; elsewhere host RAM is a
/// tier of its own across the link, reserved `Shared`; none, or why not,
/// where `Shared` reserves nothing.
/// Every backend's decode entries read tier records in place; only a
/// linked tier's are promoted ([`HostTier::Linked`]). The host lands and
/// converts file misses into the tier's records in place.
pub fn host_tier(
    topology: &seismic::DeviceTopology,
    executor: &Executor,
) -> Result<Option<HostTier>, String> {
    let reservation = executor.reserving(seismic::ReservedContract::Shared).map_err(ToString::to_string)?;
    Ok(reservation.map(|reservation| match topology.shared_ranges_lie_in_device_memory(executor) {
        true => HostTier::Unified,
        false => HostTier::Linked(reservation),
    }))
}

/// The selected device, resolved and opened in the executing process,
/// admitted against its fresh observations.
pub struct OpenedPlatform {
    selector: ExecutorSelector,
    device: Device,
    /// What the engine might still claim in each of the device's memory
    /// domains when it was opened.
    room: crate::TuningRoom,
}

impl OpenedPlatform {
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// What a search on this device is held to: what the platform let the
    /// engine claim in each memory domain when the device was opened, its
    /// observed availability above the planning reserve.
    pub fn tuning_room(&self) -> crate::TuningRoom {
        self.room
    }

    pub fn selector(&self) -> ExecutorSelector {
        self.selector
    }

    pub fn into_device(self) -> Device {
        self.device
    }
}

#[derive(Debug)]
pub enum PlatformError {
    Selection(SelectionError),
    Memory(MemoryPolicyError),
    Resolve(ResolveError),
    Open(OpenError),
    Policy {
        path: ExecutionPath,
        outcome: String,
    },
    Qualification(CatalogError),
    Records(crate::assessment::ExecutorCostsError),
}

impl fmt::Display for PlatformError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Selection(error) => write!(formatter, "{error}"),
            Self::Memory(error) => write!(formatter, "{error}"),
            Self::Resolve(error) => write!(formatter, "{error}"),
            Self::Open(error) => write!(formatter, "{error}"),
            Self::Policy { path, outcome } => {
                write!(formatter, "platform rejected path {path}: {outcome}")
            }
            Self::Qualification(error) => write!(formatter, "{error}"),
            Self::Records(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for PlatformError {}

/// Automatic selection and stable assessment against the current topology,
/// its graph cost measured by opening the device: what a tool selects by. A
/// load and an assessment select by [`select_recorded`], which measures
/// nothing.
pub fn select_device(
    catalog: &Devices,
    path: ExecutionPath,
    request: DeviceRequest,
    reserves: &MemoryReserves,
) -> Result<SelectedDevice, PlatformError> {
    enforce_phase_one(path)?;
    let topology = catalog.topology();
    let info = select(&topology, path, request).map_err(PlatformError::Selection)?;
    let status = catalog.status(&topology, info.id());
    // Read after the status, whose fit the opening the measurement takes
    // must not lower.
    let graph_cost = catalog.graph_cost(info.id()).map_err(PlatformError::Open)?;
    selected(&topology, &status, reserves, info, GraphPricing::Measured(graph_cost)).map_err(PlatformError::Memory)
}

/// [`select_device`] with no device opened and nothing measured, what a
/// load and an assessment select by: its graph cost as the kernel cache at
/// `root` records it (`assessment::GraphMemory`), or, without a cache or
/// until the measurement job stored it there, its backend's bound
/// (`seismic::unmeasured_graph_cost`), so a load always plans, charges and
/// records its graphs.
pub fn select_recorded(
    catalog: &Devices,
    path: ExecutionPath,
    request: DeviceRequest,
    reserves: &MemoryReserves,
    root: Option<&std::path::Path>,
) -> Result<SelectedDevice, PlatformError> {
    enforce_phase_one(path)?;
    let topology = catalog.topology();
    let info = select(&topology, path, request).map_err(PlatformError::Selection)?;
    let status = catalog.status(&topology, info.id());
    let graph_cost = graph_pricing(&topology, &info, root)?;
    selected(&topology, &status, reserves, info, graph_cost).map_err(PlatformError::Memory)
}

/// What `info`'s graphs are priced at from the kernel cache at `root`: the
/// device's own cost once the measurement job recorded it there, its
/// backend's bound until then and without a cache.
pub fn graph_pricing(
    topology: &seismic::DeviceTopology,
    info: &Executor,
    root: Option<&std::path::Path>,
) -> Result<GraphPricing, PlatformError> {
    let recorded = match root {
        Some(root) => {
            let records = crate::KernelCache::measurement_records(root);
            crate::assessment::GraphMemory::stored(topology, info, &records).map_err(PlatformError::Records)?
        }
        None => None,
    };
    Ok(match recorded {
        Some(recorded) => GraphPricing::Measured(recorded),
        None => GraphPricing::Bound(seismic::unmeasured_graph_cost(info.backend())),
    })
}

/// `info`, of `topology`, as selected under `status` and `reserves`: its
/// stable device capacity, its host tier, the reading its fit questions
/// read, and its graph cost.
pub fn selected(
    topology: &seismic::DeviceTopology,
    status: &seismic::Status,
    reserves: &MemoryReserves,
    info: Executor,
    graph_cost: GraphPricing,
) -> Result<SelectedDevice, MemoryPolicyError> {
    let fit = Fit::new(topology, status, reserves, HostRoom::Allocatable);
    let assessment_capacity_bytes = fit.placement(&info, Contract::DeviceOnly)?.stable_bytes;
    Ok(SelectedDevice {
        graph_cost,
        host_tier: host_tier(topology, &info),
        identity: topology
            .executor_identity(info.id())
            .expect("the selected executor is of the topology it was selected from"),
        info,
        assessment_capacity_bytes,
        reading: FitReading { topology: topology.clone(), status: status.clone(), reserves: *reserves },
    })
}

/// Resolve the selected identity in this process's catalog, open it, set a
/// ceiling from fresh scoped availability above the planning reserve, and
/// enforce it in Seismic.
/// Never substitutes another device.
pub fn open_selected(
    catalog: &Devices,
    selector: ExecutorSelector,
    config: PlatformConfig,
) -> Result<OpenedPlatform, PlatformError> {
    enforce_phase_one(config.path)?;
    let id = catalog.resolve(selector).map_err(PlatformError::Resolve)?;
    let device = catalog
        .open_with(
            id,
            DeviceOptions {
                artifacts: config.artifacts,
                graph_cost: config.graph_cost,
            },
        )
        .map_err(PlatformError::Open)?;
    let status = catalog.status(device.topology(), device.info().id());
    // A load opens the device: its first allocations are a load's, which
    // may take what the OS can obtain (`DeviceHeap::open` holds the device
    // to the same room until the load ends).
    refresh_device_ceiling(&device, &status, &config.reserves, HostRoom::Obtainable(LoadStage::Loading))
        .map_err(PlatformError::Memory)?;
    let topology = device.topology();
    let host_ram = topology.host_ram().id();
    let fit = Fit::new(topology, &status, &config.reserves, HostRoom::Obtainable(LoadStage::Loading));
    let room = crate::TuningRoom {
        device_bytes: fit
            .placement(device.info(), Contract::DeviceOnly)
            .map_err(PlatformError::Memory)?
            .ceiling_bytes,
        host_bytes: fit
            .store(host_ram)
            .map_err(PlatformError::Memory)?
            .ceiling_bytes,
        host_shared: DomainRole::of(topology, device.info(), host_ram) == DomainRole::Allocation,
    };
    Ok(OpenedPlatform {
        selector,
        device,
        room,
    })
}

fn enforce_phase_one(path: ExecutionPath) -> Result<(), PlatformError> {
    match path {
        ExecutionPath::Native => Ok(()),
        ExecutionPath::Planned => Err(PlatformError::Policy {
            path,
            outcome: "Planned remains unavailable until compiler convergence gate G6".into(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_one_rejects_planned_without_qualification() {
        assert!(matches!(
            enforce_phase_one(ExecutionPath::Planned),
            Err(PlatformError::Policy {
                path: ExecutionPath::Planned,
                ..
            })
        ));
        assert!(enforce_phase_one(ExecutionPath::Native).is_ok());
    }

    #[test]
    fn opening_an_unknown_selector_never_substitutes_a_device() {
        let catalog = Devices::discover(&[]).unwrap();
        let missing = ExecutorSelector::Metal { registry_id: 0 };
        assert!(matches!(
            open_selected(
                &catalog,
                missing,
                PlatformConfig {
                    path: ExecutionPath::Native,
                    artifacts: None,
                    graph_cost: None,
                    reserves: MemoryReserves::standard(),
                },
            ),
            Err(PlatformError::Resolve(ResolveError::Missing(selector))) if selector == missing
        ));
    }
}
