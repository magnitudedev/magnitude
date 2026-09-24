//! Device selection over the Seismic topology. A host either names the
//! device (a backend or an exact selector) or asks for automatic selection.
//! Automatic selection considers accelerators only (Metal, CUDA, and Vulkan
//! GPUs that CUDA does not also expose: NVIDIA reports one UUID to both
//! APIs, and such a GPU is left to CUDA). The CPU native route and software
//! Vulkan devices (lavapipe) exist for functional coverage and are used only
//! when requested explicitly. Candidates must be executable on the path and
//! have established memory backing. Ranking between several fitting devices
//! is an open policy, so more than one candidate is an explicit result
//! rather than an arbitrary choice; the host then names one.

use crate::ExecutionPath;
use seismic::{
    Availability, BackendName, DeviceInfo, DeviceKind, DeviceMemory, DeviceSelector,
    DeviceTopology,
};
use std::{fmt, str::FromStr};

/// Which device the host asks the engine to execute on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeviceRequest {
    /// The single available accelerator.
    Automatic,
    /// The single available device of this backend.
    Backend(BackendName),
    /// Exactly this device.
    Selector(DeviceSelector),
}

impl fmt::Display for DeviceRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Automatic => formatter.write_str("auto"),
            Self::Backend(backend) => formatter.write_str(backend.as_str()),
            Self::Selector(selector) => write!(formatter, "{selector}"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceRequestParseError(String);

impl fmt::Display for DeviceRequestParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid device `{}`: expected auto, metal, cuda, vulkan, cpu, or an exact selector (host-cpu, metal:<id>, cuda:<uuid>, vulkan:<uuid>)",
            self.0
        )
    }
}

impl std::error::Error for DeviceRequestParseError {}

impl FromStr for DeviceRequest {
    type Err = DeviceRequestParseError;

    /// Parses exactly the [`fmt::Display`] form.
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        if text == "auto" {
            return Ok(Self::Automatic);
        }
        if let Some(backend) = BackendName::parse(text) {
            return Ok(Self::Backend(backend));
        }
        text.parse::<DeviceSelector>()
            .map(Self::Selector)
            .map_err(|_| DeviceRequestParseError(text.to_owned()))
    }
}

impl DeviceRequest {
    fn admits(self, device: &DeviceInfo, topology: &DeviceTopology) -> bool {
        match self {
            Self::Automatic => match device.backend {
                BackendName::Metal | BackendName::Cuda => true,
                BackendName::Vulkan => {
                    device.kind == DeviceKind::Gpu && !exposed_through_cuda(device, topology)
                }
                BackendName::Cpu => false,
            },
            Self::Backend(backend) => device.backend == backend,
            Self::Selector(selector) => device.selector == selector,
        }
    }
}

/// Whether the topology holds a CUDA device with this Vulkan device's UUID.
fn exposed_through_cuda(device: &DeviceInfo, topology: &DeviceTopology) -> bool {
    let DeviceSelector::Vulkan { uuid } = device.selector else {
        return false;
    };
    topology
        .devices()
        .iter()
        .any(|other| other.selector == DeviceSelector::Cuda { uuid })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SelectionError {
    NoCandidate {
        path: ExecutionPath,
        request: DeviceRequest,
        evidence: Vec<String>,
    },
    Ambiguous {
        path: ExecutionPath,
        request: DeviceRequest,
        candidates: Vec<DeviceSelector>,
    },
}

impl fmt::Display for SelectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoCandidate {
                path,
                request,
                evidence,
            } => write!(
                formatter,
                "no device matching `{request}` can execute {path}: {}",
                evidence.join("; ")
            ),
            Self::Ambiguous {
                path,
                request,
                candidates,
            } => write!(
                formatter,
                "{} devices match `{request}` for {path} and no ranking policy is defined; name one with --device: {}",
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

pub fn select(
    topology: &DeviceTopology,
    path: ExecutionPath,
    request: DeviceRequest,
) -> Result<DeviceInfo, SelectionError> {
    let mut candidates = Vec::new();
    let mut evidence = Vec::new();
    for device in topology.devices() {
        if !request.admits(device, topology) {
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
                    .filter(|diagnostic| match request {
                        DeviceRequest::Automatic => diagnostic.backend != BackendName::Cpu,
                        DeviceRequest::Backend(backend) => diagnostic.backend == backend,
                        DeviceRequest::Selector(_) => true,
                    })
                    .map(|diagnostic| {
                        format!("{}: {}", diagnostic.backend.as_str(), diagnostic.message)
                    }),
            );
            if evidence.is_empty() {
                evidence.push("no matching device was discovered".into());
            }
            Err(SelectionError::NoCandidate {
                path,
                request,
                evidence,
            })
        }
        several => Err(SelectionError::Ambiguous {
            path,
            request,
            candidates: several.iter().map(|device| device.selector).collect(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_requests_round_trip_through_their_text_form() {
        for text in [
            "auto",
            "metal",
            "cuda",
            "cpu",
            "host-cpu",
            "metal:00000001000004a5",
            "cuda:00112233-4455-6677-8899-aabbccddeeff",
            "vulkan",
            "vulkan:00112233-4455-6677-8899-aabbccddeeff",
        ] {
            let request: DeviceRequest = text.parse().unwrap();
            assert_eq!(request.to_string(), text);
        }
        assert_eq!(
            "metal".parse::<DeviceRequest>(),
            Ok(DeviceRequest::Backend(BackendName::Metal))
        );
        assert_eq!(
            "host-cpu".parse::<DeviceRequest>(),
            Ok(DeviceRequest::Selector(DeviceSelector::HostCpu))
        );
        assert!("gpu".parse::<DeviceRequest>().is_err());
    }

    #[test]
    fn automatic_selection_considers_accelerators_only() {
        let catalog = seismic::DeviceCatalog::discover().unwrap();
        let topology = catalog.topology();
        let accelerators = topology
            .devices()
            .iter()
            .filter(|device| match device.backend {
                BackendName::Cpu => false,
                BackendName::Metal | BackendName::Cuda => true,
                BackendName::Vulkan => {
                    device.kind == DeviceKind::Gpu && !exposed_through_cuda(device, &topology)
                }
            })
            .filter(|device| {
                device.availability == Availability::Available
                    && matches!(device.memory, DeviceMemory::Established(_))
            })
            .count();
        match select(&topology, ExecutionPath::Native, DeviceRequest::Automatic) {
            Ok(device) => {
                assert_eq!(accelerators, 1);
                assert_ne!(device.backend, BackendName::Cpu);
            }
            Err(SelectionError::NoCandidate { .. }) => assert_eq!(accelerators, 0),
            Err(SelectionError::Ambiguous { candidates, .. }) => {
                assert_eq!(candidates.len(), accelerators)
            }
        }
    }

    /// A Vulkan device CUDA also exposes (same UUID) is never an automatic
    /// candidate; a software Vulkan device neither.
    #[test]
    fn automatic_selection_leaves_shadowed_and_software_vulkan_devices_out() {
        let catalog = seismic::DeviceCatalog::discover().unwrap();
        let topology = catalog.topology();
        for device in topology.devices().iter().filter(|device| device.backend == BackendName::Vulkan) {
            let DeviceSelector::Vulkan { uuid } = device.selector else {
                panic!("a Vulkan device has a Vulkan selector")
            };
            let shadowed = topology
                .devices()
                .iter()
                .any(|other| other.selector == DeviceSelector::Cuda { uuid });
            assert_eq!(
                DeviceRequest::Automatic.admits(device, &topology),
                device.kind == DeviceKind::Gpu && !shadowed,
                "{}",
                device.selector
            );
        }
    }

    /// Vulkan is not built on macOS.
    #[cfg(target_os = "macos")]
    #[test]
    fn a_vulkan_request_is_refused_with_the_missing_runtime() {
        let catalog = seismic::DeviceCatalog::discover().unwrap();
        let request: DeviceRequest = "vulkan".parse().unwrap();
        assert_eq!(
            select(&catalog.topology(), ExecutionPath::Native, request),
            Err(SelectionError::NoCandidate {
                path: ExecutionPath::Native,
                request: DeviceRequest::Backend(BackendName::Vulkan),
                evidence: vec!["vulkan: this build has no Vulkan runtime".into()],
            })
        );
    }

    #[test]
    fn explicit_backend_and_selector_requests_select_exactly_that_device() {
        let catalog = seismic::DeviceCatalog::discover().unwrap();
        let topology = catalog.topology();
        for device in topology.devices().iter().filter(|device| {
            device.availability == Availability::Available
                && matches!(device.memory, DeviceMemory::Established(_))
        }) {
            let selected = select(
                &topology,
                ExecutionPath::Native,
                DeviceRequest::Selector(device.selector),
            )
            .unwrap();
            assert_eq!(selected.selector, device.selector);
            if let Ok(by_backend) = select(
                &topology,
                ExecutionPath::Native,
                DeviceRequest::Backend(device.backend),
            ) {
                assert_eq!(by_backend.backend, device.backend);
            }
        }
        let missing = DeviceSelector::Metal { registry_id: 0 };
        assert!(matches!(
            select(
                &topology,
                ExecutionPath::Native,
                DeviceRequest::Selector(missing)
            ),
            Err(SelectionError::NoCandidate { .. })
        ));
    }
}
