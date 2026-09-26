//! Device selection over the Seismic topology (integration spec §5.5).
//!
//! A host either names the device (a backend or an exact selector) or asks
//! for automatic selection. Automatic selection takes backends in the fixed
//! order Metal > CUDA > Vulkan > CPU, uses the first backend that has a usable
//! device, and within it the first device in Seismic's enumeration order. A
//! device is usable when Seismic reports it available (its driver or loader
//! is present and it meets its backend's floor; Seismic discovery is the
//! authority) and its memory backing is established. Automatic Vulkan
//! candidates are GPUs: a software Vulkan device (lavapipe) is never an
//! accelerator, so the native CPU backend is automatic when no accelerator
//! is usable. Selection never ranks devices by fit or speed. Preview,
//! assessment and load all select through here, so the assessed device is
//! the device a load uses.

use crate::ExecutionPath;
use seismic::{
    Availability, BackendName, DeviceInfo, DeviceKind, DeviceMemory, DeviceSelector, DeviceTopology,
};
use std::{fmt, str::FromStr};

/// Which device the host asks the engine to execute on. Its text form is
/// its stable, cross-process encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeviceRequest {
    /// The automatic rule: first usable device of the first usable backend.
    Automatic,
    /// The first usable device of this backend.
    Backend(BackendName),
    /// Exactly this device.
    Selector(DeviceSelector),
}

/// The automatic backend order.
pub const AUTOMATIC_BACKEND_ORDER: [BackendName; 4] = [
    BackendName::Metal,
    BackendName::Cuda,
    BackendName::Vulkan,
    BackendName::Cpu,
];

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

impl serde::Serialize for DeviceRequest {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> serde::Deserialize<'de> for DeviceRequest {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = <String as serde::Deserialize>::deserialize(deserializer)?;
        text.parse().map_err(serde::de::Error::custom)
    }
}

/// Why a discovered device cannot be selected, when it cannot. `None` means
/// the device is usable. This is Seismic's own reason (for example a driver
/// below the backend floor), surfaced unchanged.
pub fn unusable_reason(device: &DeviceInfo) -> Option<&str> {
    match (&device.availability, &device.memory) {
        (Availability::Unavailable { reason }, _) => Some(reason),
        (Availability::Available, DeviceMemory::Unsupported { reason }) => Some(reason),
        (Availability::Available, DeviceMemory::Established(_)) => None,
    }
}

fn automatic_candidate(device: &DeviceInfo) -> bool {
    device.backend != BackendName::Vulkan || device.kind == DeviceKind::Gpu
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SelectionError {
    /// No usable device matches the request. `evidence` names each matching
    /// device's unusable reason and each relevant backend's discovery
    /// diagnostic.
    NoCandidate {
        path: ExecutionPath,
        request: DeviceRequest,
        evidence: Vec<String>,
    },
    /// An exact selector matches several devices in this process's catalog:
    /// the identity no longer names one device.
    Stale {
        selector: DeviceSelector,
        matches: usize,
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
            Self::Stale { selector, matches } => write!(
                formatter,
                "device {selector} matches {matches} discovered devices"
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
    let backends: &[BackendName] = match request {
        DeviceRequest::Automatic => &AUTOMATIC_BACKEND_ORDER,
        DeviceRequest::Backend(backend) => &[backend],
        DeviceRequest::Selector(selector) => {
            let matching = topology
                .devices()
                .iter()
                .filter(|device| device.selector == selector)
                .collect::<Vec<_>>();
            if matching.len() > 1 {
                return Err(SelectionError::Stale {
                    selector,
                    matches: matching.len(),
                });
            }
            return match matching.first().copied() {
                Some(device) => match unusable_reason(device) {
                    None => Ok(device.clone()),
                    Some(reason) => Err(SelectionError::NoCandidate {
                        path,
                        request,
                        evidence: vec![format!("{} ({}): {reason}", device.selector, device.name)],
                    }),
                },
                None => Err(SelectionError::NoCandidate {
                    path,
                    request,
                    evidence: vec![format!("device {selector} was not discovered")],
                }),
            };
        }
    };
    let mut evidence = Vec::new();
    for &backend in backends {
        for device in topology.devices().iter().filter(|device| {
            device.backend == backend
                && (request != DeviceRequest::Automatic || automatic_candidate(device))
        }) {
            match unusable_reason(device) {
                None => return Ok(device.clone()),
                Some(reason) => {
                    evidence.push(format!("{} ({}): {reason}", device.selector, device.name))
                }
            }
        }
        evidence.extend(
            topology
                .diagnostics()
                .iter()
                .filter(|diagnostic| diagnostic.backend == backend)
                .map(|diagnostic| format!("{}: {}", backend.as_str(), diagnostic.message)),
        );
    }
    if evidence.is_empty() {
        evidence.push("no matching device was discovered".into());
    }
    Err(SelectionError::NoCandidate {
        path,
        request,
        evidence,
    })
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
            let encoded = serde_json::to_string(&request).unwrap();
            assert_eq!(serde_json::from_str::<DeviceRequest>(&encoded).unwrap(), request);
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

    /// The automatic rule over the live topology: the first usable device of
    /// the first backend (in the fixed order) that has one. The host CPU is
    /// always usable, so automatic selection always succeeds.
    #[test]
    fn automatic_selection_takes_the_first_usable_device_of_the_first_usable_backend() {
        let catalog = seismic::DeviceCatalog::discover().unwrap();
        let topology = catalog.topology();
        let expected = AUTOMATIC_BACKEND_ORDER
            .iter()
            .find_map(|&backend| {
                topology.devices().iter().find(|device| {
                    device.backend == backend
                        && automatic_candidate(device)
                        && unusable_reason(device).is_none()
                })
            })
            .expect("the host CPU is usable");
        let selected = select(&topology, ExecutionPath::Native, DeviceRequest::Automatic).unwrap();
        assert_eq!(selected.selector, expected.selector);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn automatic_selection_prefers_metal_on_apple_silicon() {
        let catalog = seismic::DeviceCatalog::discover().unwrap();
        let selected =
            select(&catalog.topology(), ExecutionPath::Native, DeviceRequest::Automatic).unwrap();
        assert_eq!(selected.backend, BackendName::Metal);
    }

    /// A software Vulkan device is never an automatic candidate.
    #[test]
    fn automatic_selection_leaves_software_vulkan_devices_out() {
        let catalog = seismic::DeviceCatalog::discover().unwrap();
        let topology = catalog.topology();
        for device in topology
            .devices()
            .iter()
            .filter(|device| device.backend == BackendName::Vulkan)
        {
            assert_eq!(automatic_candidate(device), device.kind == DeviceKind::Gpu);
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
        for device in topology
            .devices()
            .iter()
            .filter(|device| unusable_reason(device).is_none())
        {
            let selected = select(
                &topology,
                ExecutionPath::Native,
                DeviceRequest::Selector(device.selector),
            )
            .unwrap();
            assert_eq!(selected.selector, device.selector);
            let by_backend = select(
                &topology,
                ExecutionPath::Native,
                DeviceRequest::Backend(device.backend),
            )
            .unwrap();
            assert_eq!(by_backend.backend, device.backend);
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
