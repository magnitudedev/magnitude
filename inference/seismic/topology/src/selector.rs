//! Stable native executor identities for selection across processes and revisions.

use crate::BackendName;
use serde::{Deserialize, Serialize};
use std::{fmt, str::FromStr};

/// The identity an executor's own API uses to open it. Physical-device
/// grouping does not change this backend-specific selection identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExecutorSelector {
    /// The host CPU.
    HostCpu,
    /// The Metal device of `MTLDevice.registryID`.
    Metal { registry_id: u64 },
    /// The CUDA device of this device UUID.
    Cuda { uuid: [u8; 16] },
    /// The Vulkan device of this device UUID.
    Vulkan { uuid: [u8; 16] },
}

impl ExecutorSelector {
    /// The backend that opens the executor.
    pub fn backend(self) -> BackendName {
        match self {
            Self::HostCpu => BackendName::Cpu,
            Self::Metal { .. } => BackendName::Metal,
            Self::Cuda { .. } => BackendName::Cuda,
            Self::Vulkan { .. } => BackendName::Vulkan,
        }
    }
}

impl fmt::Display for ExecutorSelector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (backend, uuid) = match self {
            Self::HostCpu => return f.write_str("host-cpu"),
            Self::Metal { registry_id } => return write!(f, "metal:{registry_id:016x}"),
            Self::Cuda { uuid } => ("cuda", uuid),
            Self::Vulkan { uuid } => ("vulkan", uuid),
        };
        write!(f, "{backend}:")?;
        for (index, byte) in uuid.iter().enumerate() {
            if matches!(index, 4 | 6 | 8 | 10) {
                f.write_str("-")?;
            }
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// Text that names no executor selector, as given.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectorParseError(String);

impl fmt::Display for SelectorParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid executor selector `{}`", self.0)
    }
}

impl std::error::Error for SelectorParseError {}

impl FromStr for ExecutorSelector {
    type Err = SelectorParseError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let invalid = || SelectorParseError(text.to_owned());
        if text == "host-cpu" {
            return Ok(Self::HostCpu);
        }
        if let Some(id) = text.strip_prefix("metal:") {
            return u64::from_str_radix(id, 16)
                .map(|registry_id| Self::Metal { registry_id })
                .map_err(|_| invalid());
        }
        let (backend, id) = text.split_once(':').ok_or_else(invalid)?;
        let hex = id.replace('-', "");
        if hex.len() != 32 || !hex.is_ascii() {
            return Err(invalid());
        }
        let mut uuid = [0; 16];
        for (index, byte) in uuid.iter_mut().enumerate() {
            *byte =
                u8::from_str_radix(&hex[index * 2..index * 2 + 2], 16).map_err(|_| invalid())?;
        }
        match backend {
            "cuda" => Ok(Self::Cuda { uuid }),
            "vulkan" => Ok(Self::Vulkan { uuid }),
            _ => Err(invalid()),
        }
    }
}
