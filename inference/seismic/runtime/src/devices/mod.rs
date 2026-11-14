//! Opened execution resources over the machine topology.
mod catalog;
pub use catalog::Devices;
use seismic_compiler::errors::TargetError;
use seismic_topology::{BackendName, ExecutorId, ExecutorSelector};
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OpenError {
    /// The identifier belongs to an earlier topology revision.
    Stale(ExecutorId),
    /// No device of `backend` was discovered; `diagnostics` are that
    /// backend's discovery diagnostics (why it has none, when it said).
    NoDevice {
        backend: BackendName,
        diagnostics: Vec<String>,
    },
    Unavailable {
        selector: ExecutorSelector,
        reason: String,
    },
    /// The native device at this descriptor no longer has the discovered
    /// identity.
    IdentityChanged(ExecutorSelector),
    Backend(TargetError),
    /// The device is already open with a different artifact store.
    ArtifactStoreConflict(ExecutorSelector),
    /// The device is already open charging other ledgers' stores.
    StoreConflict(ExecutorSelector),
}

impl fmt::Display for OpenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Stale(id) => write!(
                f,
                "device id {id:?} belongs to an earlier topology revision"
            ),
            Self::NoDevice {
                backend,
                diagnostics,
            } => {
                write!(f, "no {} device was discovered", backend.as_str())?;
                if !diagnostics.is_empty() {
                    write!(f, ": {}", diagnostics.join("; "))?;
                }
                Ok(())
            }
            Self::Unavailable { selector, reason } => {
                write!(f, "device {selector} is unavailable: {reason}")
            }
            Self::IdentityChanged(selector) => {
                write!(f, "device {selector} changed identity since discovery")
            }
            Self::Backend(error) => write!(f, "{error}"),
            Self::ArtifactStoreConflict(selector) => {
                write!(
                    f,
                    "device {selector} is already open with a different artifact store"
                )
            }
            Self::StoreConflict(selector) => {
                write!(f, "device {selector} is already open charging other ledgers' stores")
            }
        }
    }
}
impl std::error::Error for OpenError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResolveError {
    Missing(ExecutorSelector),
}

impl fmt::Display for ResolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing(selector) => write!(f, "device {selector} is not present"),
        }
    }
}
impl std::error::Error for ResolveError {}
