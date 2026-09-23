//! Automatic device selection over the Seismic topology. Callers supply no
//! backend order or ordinal; candidates must be executable on the path and
//! have established memory backing. Ranking between several fitting devices
//! is an open policy, so more than one candidate is an explicit result
//! rather than an arbitrary choice.

use crate::ExecutionPath;
use seismic::{Availability, BackendName, DeviceInfo, DeviceMemory, DeviceSelector, DeviceTopology};
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SelectionError {
    NoCandidate {
        path: ExecutionPath,
        evidence: Vec<String>,
    },
    Ambiguous {
        path: ExecutionPath,
        candidates: Vec<DeviceSelector>,
    },
}

impl fmt::Display for SelectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoCandidate { path, evidence } => write!(
                formatter,
                "no device can execute {path}: {}",
                evidence.join("; ")
            ),
            Self::Ambiguous { path, candidates } => write!(
                formatter,
                "{} devices can execute {path} and no ranking policy is defined: {}",
                candidates.len(),
                candidates
                    .iter()
                    .map(DeviceSelector::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }
}

impl std::error::Error for SelectionError {}

fn executes(path: ExecutionPath, backend: BackendName) -> bool {
    match path {
        ExecutionPath::NativeMetal => backend == BackendName::Metal,
        ExecutionPath::Planned => true,
    }
}

pub fn select(topology: &DeviceTopology, path: ExecutionPath) -> Result<DeviceInfo, SelectionError> {
    let mut candidates = Vec::new();
    let mut evidence = Vec::new();
    for device in topology.devices() {
        if !executes(path, device.backend) {
            continue;
        }
        match (&device.availability, &device.memory) {
            (Availability::Unavailable { reason }, _) => {
                evidence.push(format!("{} ({}): {reason}", device.selector, device.name))
            }
            (Availability::Available, DeviceMemory::Unsupported { reason }) => {
                evidence.push(format!("{} ({}): {reason}", device.selector, device.name))
            }
            (Availability::Available, DeviceMemory::Established(_)) => candidates.push(device),
        }
    }
    match candidates.as_slice() {
        [device] => Ok((*device).clone()),
        [] => {
            evidence.extend(
                topology
                    .diagnostics()
                    .iter()
                    .filter(|diagnostic| executes(path, diagnostic.backend))
                    .map(|diagnostic| format!("{}: {}", diagnostic.backend.as_str(), diagnostic.message)),
            );
            if evidence.is_empty() {
                evidence.push("no device of a compatible backend was discovered".into());
            }
            Err(SelectionError::NoCandidate { path, evidence })
        }
        several => Err(SelectionError::Ambiguous {
            path,
            candidates: several.iter().map(|device| device.selector).collect(),
        }),
    }
}
