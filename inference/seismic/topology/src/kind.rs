//! Storage kinds: each
//! executor's ways of obtaining memory, as facts.
//!
//! A kind is one way a backend obtains memory: a CUDA allocation call and
//! location, a Vulkan memory type, a Metal storage mode, host pages. Which
//! kinds exist, what each is called and how it is allocated are the
//! backend's, so here a kind is opaque ([`NativeKind`]) with the facts its
//! backend reports of it. A kind holds no measured figure: what running on
//! it costs is measured by whoever prices it.

use crate::model::{AccessPathId, ExecutorId, KindId};
use seismic_lang::registry::BackendName;
use serde::{Deserialize, Serialize};

/// A kind as its backend names it: stable across runs and revisions, so it
/// keys what is measured of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct NativeKind {
    /// The backend that allocates it.
    pub backend: BackendName,
    /// The backend's own identifier of the kind, stable across runs.
    pub id: u32,
}

/// Who reaches a kind's bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Reach {
    /// The device alone: the host cannot map it.
    Device,
    /// The device, and the host through a coherent mapping.
    Mapped { reads: HostReads },
}

/// How the host reads a mapped kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HostReads {
    /// Through its caches.
    Cached,
    /// Uncached: write-combined memory, a BAR. Its stores are combined, its
    /// loads slow.
    Uncached,
}

/// Which memory a kind's bytes lie in on a device whose memory is its own;
/// on a device whose memory is the host's every kind is charged alike, along
/// its executor's placement.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KindMemory {
    /// The device's own memory.
    DeviceMemory,
    /// Host RAM.
    HostRam,
    /// One heap of the device's memory, by its backend's index of it: the
    /// device's own memory when the heap is device-local, else host RAM;
    /// `bytes` is its size.
    Heap { index: usize, device_local: bool, bytes: u64 },
}

/// What a backend reports of one of its kinds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KindReport {
    /// The backend's identifier of the kind, stable across runs.
    pub id: u32,
    /// The backend's name of the kind, for people.
    pub name: String,
    pub reach: Reach,
    pub memory: KindMemory,
}

/// One kind of one executor in a revision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Kind {
    pub(crate) id: KindId,
    pub(crate) executor: ExecutorId,
    pub(crate) native: NativeKind,
    pub(crate) name: String,
    pub(crate) reach: Reach,
    pub(crate) paths: Vec<AccessPathId>,
}

impl Kind {
    /// The kind's identifier in its revision.
    pub fn id(&self) -> KindId {
        self.id
    }
    /// The executor that allocates it.
    pub fn executor(&self) -> ExecutorId {
        self.executor
    }
    /// The kind as its backend names it.
    pub fn native(&self) -> NativeKind {
        self.native
    }
    /// The backend's name of the kind.
    pub fn name(&self) -> &str {
        &self.name
    }
    /// Who reaches its bytes.
    pub fn reach(&self) -> Reach {
        self.reach
    }
    /// The access paths its allocations are charged along, in order.
    pub fn paths(&self) -> &[AccessPathId] {
        &self.paths
    }
}
