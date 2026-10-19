//! Opened execution resources over the machine topology.
mod catalog;
pub use crate::memory::MemoryUsage;
pub use catalog::Devices;
use seismic_compiler::errors::TargetError;
use seismic_topology::{AccessPathId, BackendName, ExecutorId, ExecutorSelector, StoreId};
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
        }
    }
}
impl std::error::Error for OpenError {}

/// A limit asked of an opened device on an identifier that is not its own.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LimitError {
    /// The store belongs to another topology revision than the device's.
    StaleStore(StoreId),
    /// The access path is not one of the device's.
    ForeignPath(AccessPathId),
}

impl fmt::Display for LimitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StaleStore(store) => write!(
                f,
                "store {store:?} belongs to another topology revision than the device's"
            ),
            Self::ForeignPath(path) => {
                write!(f, "access path {path:?} is not one of the device's")
            }
        }
    }
}
impl std::error::Error for LimitError {}

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
