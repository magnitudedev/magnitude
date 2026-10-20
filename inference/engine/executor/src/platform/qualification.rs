use super::policy::{DomainRole, Fit, MemoryPolicyError, MemoryReserves, refresh_device_ceiling};
use super::selection::{DeviceRequest, SelectionError, select};
use crate::{CatalogError, ExecutionPath};
use seismic::{
    ArtifactStore, Device, DeviceOptions, Devices, Executor, ExecutorSelector, OpenError,
    ResolveError, StorageKind,
};
use std::fmt;
use std::sync::Arc;

#[derive(Clone)]
pub struct PlatformConfig {
    pub path: ExecutionPath,
    /// Where the device keeps formed kernels between loads.
    pub artifacts: Option<Arc<dyn ArtifactStore>>,
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
    /// Device storage's stable fit: what its placement holds across its
    /// stores, each within process limits and grants, less its planning
    /// reserve (`Fit::placement`).
    pub assessment_capacity_bytes: u64,
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
        }
    }
}

impl std::error::Error for PlatformError {}

/// Automatic selection and stable assessment against the current topology.
pub fn select_device(
    catalog: &Devices,
    path: ExecutionPath,
    request: DeviceRequest,
    reserves: &MemoryReserves,
) -> Result<SelectedDevice, PlatformError> {
    enforce_phase_one(path)?;
    let topology = catalog.topology();
    let info = select(&topology, path, request).map_err(PlatformError::Selection)?;
    let status = catalog.status();
    let assessment_capacity_bytes = Fit::new(&topology, &status, reserves)
        .placement(&info, StorageKind::Device)
        .map_err(PlatformError::Memory)?
        .stable_bytes;
    Ok(SelectedDevice {
        info,
        assessment_capacity_bytes,
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
            },
        )
        .map_err(PlatformError::Open)?;
    let status = catalog.status();
    refresh_device_ceiling(&device, &status, &config.reserves).map_err(PlatformError::Memory)?;
    let topology = device.topology();
    let host_ram = topology.host_ram().id();
    let fit = Fit::new(topology, &status, &config.reserves);
    let room = crate::TuningRoom {
        device_bytes: fit
            .placement(device.info(), StorageKind::Device)
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
                    reserves: MemoryReserves::standard(),
                },
            ),
            Err(PlatformError::Resolve(ResolveError::Missing(selector))) if selector == missing
        ));
    }
}
