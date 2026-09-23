use super::{
    budget, select, BudgetError, Discovery, DiscoveryError, Endpoint, MemoryRequirements, Plan,
    SelectionError,
};
use crate::{AttestedPrograms, CatalogError, ExecutionPath, ExecutionPlan};
use seismic::{BackendName, Device, MemoryLimitError};
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlatformConfig {
    pub path: ExecutionPath,
    pub requested_backend: Option<BackendName>,
    pub storage_bytes: Option<u64>,
    pub requirements: MemoryRequirements,
}

pub struct QualifiedPlatform {
    pub plan: Plan,
    pub device: Device,
    pub programs: AttestedPrograms,
}

/// The selected device is opened and bounded before model graph preparation.
/// Its final memory requirements are admitted after Seismic has derived the
/// graph storage footprint from checked entry contracts.
pub struct OpenedPlatform {
    endpoint: Endpoint,
    storage_bytes: u64,
    device: Device,
}

impl OpenedPlatform {
    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn admit(
        self,
        discovery: &Discovery,
        requirements: MemoryRequirements,
        programs: AttestedPrograms,
    ) -> Result<QualifiedPlatform, PlatformError> {
        let plan = budget(
            discovery.topology(),
            self.endpoint,
            Some(self.storage_bytes),
            requirements,
        )
        .map_err(PlatformError::Budget)?;
        Ok(QualifiedPlatform {
            plan,
            device: self.device,
            programs,
        })
    }
}

impl fmt::Debug for QualifiedPlatform {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("QualifiedPlatform")
            .field("plan", &self.plan)
            .field("device", &self.device)
            .field("program_path", &self.programs.path())
            .finish()
    }
}

#[derive(Debug)]
pub enum PlatformError {
    Discovery(DiscoveryError),
    Selection(SelectionError),
    Budget(BudgetError),
    Policy {
        path: ExecutionPath,
        backend: Option<BackendName>,
        outcome: String,
    },
    MemoryLimit {
        path: ExecutionPath,
        backend: BackendName,
        storage_bytes: u64,
        outcome: String,
    },
    Qualification(CatalogError),
}

impl fmt::Display for PlatformError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Discovery(error) => write!(formatter, "{error}"),
            Self::Selection(error) => write!(formatter, "{error}"),
            Self::Budget(error) => write!(formatter, "{error}"),
            Self::Policy {
                path,
                backend,
                outcome,
            } => write!(
                formatter,
                "platform rejected path {path} on {}: {outcome}",
                backend
                    .map(BackendName::as_str)
                    .unwrap_or("automatic backend")
            ),
            Self::MemoryLimit {
                path,
                backend,
                storage_bytes,
                outcome,
            } => write!(
                formatter,
                "failed to set {storage_bytes}-byte limit for {path} on {}: {outcome}",
                backend.as_str()
            ),
            Self::Qualification(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for PlatformError {}

pub fn prepare(
    discovery: &Discovery,
    config: PlatformConfig,
    execution: &ExecutionPlan,
) -> Result<QualifiedPlatform, PlatformError> {
    enforce_phase_one(config.path, config.requested_backend)?;
    let endpoint =
        select(discovery.topology(), Some(BackendName::Metal)).map_err(PlatformError::Selection)?;
    prepare_selected(discovery, endpoint, config, execution)
}

/// Prepare exactly the device selected during allocation-free execution
/// planning. Native qualification may not silently choose another endpoint.
pub fn prepare_selected(
    discovery: &Discovery,
    endpoint: Endpoint,
    config: PlatformConfig,
    execution: &ExecutionPlan,
) -> Result<QualifiedPlatform, PlatformError> {
    let requirements = config.requirements;
    let opened = open_selected(discovery, endpoint, config)?;
    let programs = AttestedPrograms::prepare(execution, opened.device())
        .map_err(PlatformError::Qualification)?;
    opened.admit(discovery, requirements, programs)
}

pub fn open_selected(
    discovery: &Discovery,
    endpoint: Endpoint,
    config: PlatformConfig,
) -> Result<OpenedPlatform, PlatformError> {
    enforce_phase_one(config.path, config.requested_backend)?;
    if endpoint.backend != BackendName::Metal
        || config
            .requested_backend
            .is_some_and(|backend| backend != endpoint.backend)
        || !endpoint.is_available()
    {
        return Err(PlatformError::Policy {
            path: config.path,
            backend: Some(endpoint.backend),
            outcome: "selected endpoint is unavailable or incompatible with the execution path"
                .into(),
        });
    }
    let plan = budget(
        discovery.topology(),
        endpoint,
        config.storage_bytes,
        MemoryRequirements::default(),
    )
    .map_err(PlatformError::Budget)?;
    let device = discovery
        .open(&plan.endpoint)
        .map_err(PlatformError::Discovery)?;
    device
        .set_memory_limit(Some(plan.storage_bytes))
        .map_err(|error| memory_limit_error(config.path, &plan, error))?;
    Ok(OpenedPlatform {
        endpoint: plan.endpoint,
        storage_bytes: plan.storage_bytes,
        device,
    })
}

fn enforce_phase_one(
    path: ExecutionPath,
    requested_backend: Option<BackendName>,
) -> Result<(), PlatformError> {
    if path != ExecutionPath::NativeMetal {
        return Err(PlatformError::Policy {
            path,
            backend: requested_backend,
            outcome: "Planned remains unavailable until compiler convergence gate G6".into(),
        });
    }
    if requested_backend.is_some_and(|backend| backend != BackendName::Metal) {
        return Err(PlatformError::Policy {
            path,
            backend: requested_backend,
            outcome: "phase one admits Metal only".into(),
        });
    }
    Ok(())
}

fn memory_limit_error(path: ExecutionPath, plan: &Plan, error: MemoryLimitError) -> PlatformError {
    PlatformError::MemoryLimit {
        path,
        backend: plan.endpoint.backend,
        storage_bytes: plan.storage_bytes,
        outcome: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_one_rejects_planned_and_non_metal_without_qualification() {
        assert!(matches!(
            enforce_phase_one(ExecutionPath::Planned, Some(BackendName::Metal)),
            Err(PlatformError::Policy {
                path: ExecutionPath::Planned,
                ..
            })
        ));
        assert!(matches!(
            enforce_phase_one(ExecutionPath::NativeMetal, Some(BackendName::Cuda)),
            Err(PlatformError::Policy {
                backend: Some(BackendName::Cuda),
                ..
            })
        ));
    }
}
