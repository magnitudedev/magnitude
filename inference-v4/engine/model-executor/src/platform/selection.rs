use super::{Endpoint, Topology};
use seismic::BackendName;
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

#[derive(Clone, Debug)]
pub struct Plan {
    pub endpoint: Endpoint,
    pub storage_bytes: u64,
    pub requirements: MemoryRequirements,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SelectionError {
    NoAvailableDevice,
    RequestedUnavailable {
        backend: BackendName,
        evidence: Vec<String>,
    },
}

impl fmt::Display for SelectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoAvailableDevice => formatter.write_str("no available device was discovered"),
            Self::RequestedUnavailable { backend, evidence } => write!(
                formatter,
                "requested {} backend is unavailable: {}",
                backend.as_str(),
                evidence.join("; ")
            ),
        }
    }
}

impl std::error::Error for SelectionError {}

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

pub fn select(
    topology: &Topology,
    requested: Option<BackendName>,
) -> Result<Endpoint, SelectionError> {
    let mut candidates = topology
        .endpoints
        .iter()
        .filter(|endpoint| requested.is_none_or(|backend| endpoint.backend == backend))
        .filter(|endpoint| endpoint.is_available())
        .enumerate()
        .collect::<Vec<_>>();
    candidates.sort_by_key(|(order, endpoint)| {
        (
            endpoint.backend == BackendName::Cpu,
            endpoint.ordinal,
            *order,
        )
    });
    if let Some((_, endpoint)) = candidates.first() {
        return Ok((*endpoint).clone());
    }
    if let Some(backend) = requested {
        let evidence = topology
            .endpoints
            .iter()
            .filter(|endpoint| endpoint.backend == backend)
            .flat_map(|endpoint| endpoint.unavailable.clone())
            .collect::<Vec<_>>();
        return Err(SelectionError::RequestedUnavailable {
            backend,
            evidence: if evidence.is_empty() {
                vec!["backend was not present in the discovered topology".into()]
            } else {
                evidence
            },
        });
    }
    Err(SelectionError::NoAvailableDevice)
}

pub fn budget(
    topology: &Topology,
    endpoint: Endpoint,
    configured_storage_bytes: Option<u64>,
    requirements: MemoryRequirements,
) -> Result<Plan, BudgetError> {
    let storage_bytes = match configured_storage_bytes {
        Some(bytes) => bytes,
        None if endpoint.backend == BackendName::Cuda => {
            endpoint
                .facts
                .memory_bytes
                .checked_mul(85)
                .ok_or(BudgetError::Overflow)?
                / 100
        }
        None => {
            topology
                .host_memory_bytes
                .checked_mul(70)
                .ok_or(BudgetError::Overflow)?
                / 100
        }
    };
    if storage_bytes == 0 {
        return Err(BudgetError::ZeroBudget);
    }
    let required = requirements.total()?;
    if required > storage_bytes {
        return Err(BudgetError::Insufficient {
            required,
            storage_bytes,
            immutable_weights: requirements.immutable_weights,
            state: requirements.state,
            scratch: requirements.scratch,
            safety_margin: requirements.safety_margin,
        });
    }
    Ok(Plan {
        endpoint,
        storage_bytes,
        requirements,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::DeviceFacts;

    fn endpoint(backend: BackendName, ordinal: u32, available: bool) -> Endpoint {
        Endpoint {
            backend,
            ordinal,
            name: format!("{}:{ordinal}", backend.as_str()),
            facts: DeviceFacts {
                memory_bytes: 1_000,
                unified_memory: backend != BackendName::Cuda,
            },
            unavailable: (!available)
                .then_some(vec!["controlled unavailable".into()])
                .unwrap_or_default(),
            id: None,
        }
    }

    #[test]
    fn selection_prefers_accelerators_then_lowest_ordinal_and_preserves_evidence() {
        let topology = Topology {
            endpoints: vec![
                endpoint(BackendName::Cpu, 0, true),
                endpoint(BackendName::Cuda, 1, true),
                endpoint(BackendName::Cuda, 0, true),
                endpoint(BackendName::Metal, 0, false),
            ],
            host_memory_bytes: 2_000,
        };
        let selected = select(&topology, None).unwrap();
        assert_eq!((selected.backend, selected.ordinal), (BackendName::Cuda, 0));
        let error = select(&topology, Some(BackendName::Metal)).unwrap_err();
        assert!(matches!(
            error,
            SelectionError::RequestedUnavailable { evidence, .. }
                if evidence == ["controlled unavailable"]
        ));
    }

    #[test]
    fn budget_defaults_and_checked_requirement_accounting_are_exact() {
        let topology = Topology {
            endpoints: Vec::new(),
            host_memory_bytes: 1_000,
        };
        let metal = endpoint(BackendName::Metal, 0, true);
        let requirements = MemoryRequirements {
            immutable_weights: 100,
            state: 200,
            scratch: 300,
            safety_margin: 100,
        };
        assert_eq!(
            budget(&topology, metal.clone(), None, requirements)
                .unwrap()
                .storage_bytes,
            700
        );
        assert!(matches!(
            budget(&topology, metal, Some(699), requirements),
            Err(BudgetError::Insufficient { required: 700, .. })
        ));
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
}
