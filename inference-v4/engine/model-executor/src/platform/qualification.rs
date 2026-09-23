use super::policy::{admit_growth, assessment_capacity, MemoryPolicyError};
use super::selection::{select, SelectionError};
use crate::{AttestedPrograms, CatalogError, ExecutionPath};
use seismic::{
    BackendName, Device, DeviceCatalog, DeviceInfo, DeviceSelector, MemoryLimitError,
    ObservationError, OpenError, ResolveError,
};
use std::fmt;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryRequirements {
    pub immutable_weights: u64,
    pub state: u64,
    pub scratch: u64,
    pub safety_margin: u64,
}

impl MemoryRequirements {
    pub fn total(self) -> Result<u64, BudgetError> {
        self.immutable_weights
            .checked_add(self.state)
            .and_then(|value| value.checked_add(self.scratch))
            .and_then(|value| value.checked_add(self.safety_margin))
            .ok_or(BudgetError::Overflow)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BudgetError {
    Overflow,
    ZeroBudget,
    Insufficient {
        required: u64,
        storage_bytes: u64,
        immutable_weights: u64,
        state: u64,
        scratch: u64,
        safety_margin: u64,
    },
}

impl fmt::Display for BudgetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Overflow => formatter.write_str("device budget arithmetic overflow"),
            Self::ZeroBudget => formatter.write_str("device storage budget is zero"),
            Self::Insufficient {
                required,
                storage_bytes,
                immutable_weights,
                state,
                scratch,
                safety_margin,
            } => write!(
                formatter,
                "device budget {storage_bytes} is below required {required} bytes (weights {immutable_weights}, state {state}, scratch {scratch}, safety {safety_margin})"
            ),
        }
    }
}

impl std::error::Error for BudgetError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlatformConfig {
    pub path: ExecutionPath,
    /// The engine's requested Seismic allocation budget for the device.
    pub storage_bytes: u64,
}

/// The automatically selected device and its stable assessment capacity.
/// Planning input only: it reserves nothing and guarantees no later load.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectedDevice {
    pub info: DeviceInfo,
    pub assessment_capacity_bytes: u64,
}

/// The selected device, resolved and opened in the executing process,
/// admitted against its fresh observations and bounded by the budget.
/// Final requirements are admitted after Seismic derives graph storage.
pub struct OpenedPlatform {
    selector: DeviceSelector,
    storage_bytes: u64,
    device: Device,
}

pub struct QualifiedPlatform {
    pub selector: DeviceSelector,
    pub storage_bytes: u64,
    pub requirements: MemoryRequirements,
    pub device: Device,
    pub programs: AttestedPrograms,
}

impl OpenedPlatform {
    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn admit(
        self,
        requirements: MemoryRequirements,
        programs: AttestedPrograms,
    ) -> Result<QualifiedPlatform, PlatformError> {
        let required = requirements.total().map_err(PlatformError::Budget)?;
        if required > self.storage_bytes {
            return Err(PlatformError::Budget(BudgetError::Insufficient {
                required,
                storage_bytes: self.storage_bytes,
                immutable_weights: requirements.immutable_weights,
                state: requirements.state,
                scratch: requirements.scratch,
                safety_margin: requirements.safety_margin,
            }));
        }
        Ok(QualifiedPlatform {
            selector: self.selector,
            storage_bytes: self.storage_bytes,
            requirements,
            device: self.device,
            programs,
        })
    }
}

impl fmt::Debug for QualifiedPlatform {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("QualifiedPlatform")
            .field("selector", &self.selector)
            .field("storage_bytes", &self.storage_bytes)
            .field("requirements", &self.requirements)
            .field("device", &self.device)
            .field("program_path", &self.programs.path())
            .finish()
    }
}

#[derive(Debug)]
pub enum PlatformError {
    Selection(SelectionError),
    Observation(ObservationError),
    Memory(MemoryPolicyError),
    Resolve(ResolveError),
    Open(OpenError),
    Budget(BudgetError),
    Policy {
        path: ExecutionPath,
        backend: BackendName,
        outcome: String,
    },
    MemoryLimit {
        path: ExecutionPath,
        selector: DeviceSelector,
        storage_bytes: u64,
        error: MemoryLimitError,
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
            Self::Budget(error) => write!(formatter, "{error}"),
            Self::Policy {
                path,
                backend,
                outcome,
            } => write!(
                formatter,
                "platform rejected path {path} on {}: {outcome}",
                backend.as_str()
            ),
            Self::MemoryLimit {
                path,
                selector,
                storage_bytes,
                error,
            } => write!(
                formatter,
                "failed to set {storage_bytes}-byte limit for {path} on {selector}: {error}"
            ),
            Self::Qualification(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for PlatformError {}

/// Automatic selection and stable assessment against the current topology.
pub fn select_device(
    catalog: &DeviceCatalog,
    path: ExecutionPath,
) -> Result<SelectedDevice, PlatformError> {
    enforce_phase_one(path)?;
    let topology = catalog.topology();
    let info = select(&topology, path).map_err(PlatformError::Selection)?;
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

/// Resolve the selected identity in this process's catalog, open it, admit
/// the budget against fresh scoped observations, and enforce it in Seismic.
/// Never substitutes another device.
pub fn open_selected(
    catalog: &DeviceCatalog,
    selector: DeviceSelector,
    config: PlatformConfig,
) -> Result<OpenedPlatform, PlatformError> {
    enforce_phase_one(config.path)?;
    if config.storage_bytes == 0 {
        return Err(PlatformError::Budget(BudgetError::ZeroBudget));
    }
    let id = catalog.resolve(selector).map_err(PlatformError::Resolve)?;
    let device = catalog.open(id).map_err(PlatformError::Open)?;
    if config.path == ExecutionPath::NativeMetal && device.backend() != BackendName::Metal {
        return Err(PlatformError::Policy {
            path: config.path,
            backend: device.backend(),
            outcome: "the native Metal path requires a Metal device".into(),
        });
    }
    admit_growth(catalog, &device, config.storage_bytes).map_err(PlatformError::Memory)?;
    device
        .set_memory_limit(Some(config.storage_bytes))
        .map_err(|error| PlatformError::MemoryLimit {
            path: config.path,
            selector,
            storage_bytes: config.storage_bytes,
            error,
        })?;
    Ok(OpenedPlatform {
        selector,
        storage_bytes: config.storage_bytes,
        device,
    })
}

fn enforce_phase_one(path: ExecutionPath) -> Result<(), PlatformError> {
    if path != ExecutionPath::NativeMetal {
        return Err(PlatformError::Policy {
            path,
            backend: BackendName::Metal,
            outcome: "Planned remains unavailable until compiler convergence gate G6".into(),
        });
    }
    Ok(())
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
        assert!(enforce_phase_one(ExecutionPath::NativeMetal).is_ok());
    }

    #[test]
    fn requirement_accounting_is_exact_and_checked() {
        let requirements = MemoryRequirements {
            immutable_weights: 100,
            state: 200,
            scratch: 300,
            safety_margin: 100,
        };
        assert_eq!(requirements.total(), Ok(700));
        assert!(matches!(
            MemoryRequirements {
                immutable_weights: u64::MAX,
                state: 1,
                ..Default::default()
            }
            .total(),
            Err(BudgetError::Overflow)
        ));
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
                    path: ExecutionPath::NativeMetal,
                    storage_bytes: 1,
                },
            ),
            Err(PlatformError::Resolve(ResolveError::Missing(selector))) if selector == missing
        ));
    }
}
