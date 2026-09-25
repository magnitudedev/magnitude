use super::policy::{assessment_capacity, refresh_allocation_ceiling, MemoryPolicyError};
use super::selection::{select, DeviceRequest, SelectionError};
use crate::{CatalogError, ExecutionPath};
use seismic::{
    ArtifactStore, Device, DeviceCatalog, DeviceInfo, DeviceOptions, DeviceSelector,
    ObservationError, OpenError, ResolveError,
};
use std::fmt;
use std::sync::Arc;

#[derive(Clone)]
pub struct PlatformConfig {
    pub path: ExecutionPath,
    /// Where the device keeps formed kernels between loads.
    pub artifacts: Option<Arc<dyn ArtifactStore>>,
}

/// The automatically selected device and its stable assessment capacity.
/// Planning input only: it reserves nothing and guarantees no later load.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectedDevice {
    pub info: DeviceInfo,
    pub assessment_capacity_bytes: u64,
}

/// The selected device, resolved and opened in the executing process,
/// admitted against its fresh observations.
pub struct OpenedPlatform {
    selector: DeviceSelector,
    device: Device,
}

impl OpenedPlatform {
    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn selector(&self) -> DeviceSelector {
        self.selector
    }

    pub fn into_device(self) -> Device {
        self.device
    }
}

#[derive(Debug)]
pub enum PlatformError {
    Selection(SelectionError),
    Observation(ObservationError),
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
            Self::Observation(error) => write!(formatter, "{error}"),
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
    catalog: &DeviceCatalog,
    path: ExecutionPath,
    request: DeviceRequest,
) -> Result<SelectedDevice, PlatformError> {
    enforce_phase_one(path)?;
    let topology = catalog.topology();
    let info = select(&topology, path, request).map_err(PlatformError::Selection)?;
    let host = catalog
        .host_memory_status()
        .map_err(PlatformError::Observation)?;
    let assessment_capacity_bytes =
        assessment_capacity(&topology, &info, &host).map_err(PlatformError::Memory)?;
    Ok(SelectedDevice {
        info,
        assessment_capacity_bytes,
    })
}

/// Resolve the selected identity in this process's catalog, open it, set a
/// ceiling from fresh scoped availability, and enforce it in Seismic.
/// Never substitutes another device.
pub fn open_selected(
    catalog: &DeviceCatalog,
    selector: DeviceSelector,
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
    refresh_allocation_ceiling(catalog, &device).map_err(PlatformError::Memory)?;
    Ok(OpenedPlatform { selector, device })
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
        let catalog = DeviceCatalog::discover().unwrap();
        let missing = DeviceSelector::Metal { registry_id: 0 };
        assert!(matches!(
            open_selected(
                &catalog,
                missing,
                PlatformConfig {
                    path: ExecutionPath::Native,
                    artifacts: None,
                },
            ),
            Err(PlatformError::Resolve(ResolveError::Missing(selector))) if selector == missing
        ));
    }
}
