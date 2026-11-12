//! Opened execution resources over the machine topology.
mod catalog;
pub use crate::memory::DeviceLedgerView;
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

/// A level of the ledger an opened device bounds, through the one right to
/// bound it (`Device::bounds`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ledger {
    /// The account of one of the device's access paths: every open of its
    /// executor.
    Path(AccessPathId),
    /// The device's own ledger on a store it reaches.
    Own(StoreId),
    /// A store's ledger: every executor's of the scope that reaches it.
    Store(StoreId),
}

/// A store's ledger as an opened device reads it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StoreLedgerView {
    pub store: StoreId,
    /// What every executor of the process claims on the store.
    pub claim: u64,
    pub bound: Option<u64>,
}

/// The account of one of an opened device's access paths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PathLedgerView {
    pub path: AccessPathId,
    /// What every open of the executor claims along the path.
    pub claim: u64,
    pub bound: Option<u64>,
}

/// The ledgers an opened device reaches: each store's, each of its paths'
/// accounts, and its own on each store.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ledgers {
    pub stores: Vec<StoreLedgerView>,
    pub paths: Vec<PathLedgerView>,
    pub own: Vec<(StoreId, DeviceLedgerView)>,
}

impl Ledgers {
    /// What the device claims on each store it reaches, in store order.
    pub fn own_claims(&self) -> Vec<(StoreId, u64)> {
        let mut claims = self.own.iter().map(|(store, view)| (*store, view.claim)).collect::<Vec<_>>();
        claims.sort();
        claims
    }

    /// The account of the device's path `path`.
    pub fn path(&self, path: AccessPathId) -> Option<PathLedgerView> {
        self.paths.iter().find(|view| view.path == path).copied()
    }
}

/// What a backend holds for a storage kind beyond its allocations' charges,
/// as the backend reports it: a fact of the backend, not a ledger.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BackendHeld {
    /// The free space of the blocks it sub-allocates from: memory its stores
    /// already lost to the device that an allocation takes without the
    /// device holding more.
    pub unallocated: u64,
    /// Everything it holds beyond its allocations: that free space, its own
    /// ranges in its blocks, and its driver objects (such as recorded
    /// graphs).
    pub beyond: u64,
}

/// A bound asked of an opened device on a ledger that is not its own.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LimitError {
    /// The store belongs to another topology revision than the device's.
    StaleStore(StoreId),
    /// The access path is not one of the device's.
    ForeignPath(AccessPathId),
    /// The device reaches no such store.
    ForeignStore(StoreId),
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
            Self::ForeignStore(store) => write!(f, "the device reaches no store {store:?}"),
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
