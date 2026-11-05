//! Storage kinds: each
//! executor's ways of obtaining memory, as facts.
//!
//! A kind is one way a backend obtains memory: a CUDA allocation call and
//! location, a Vulkan memory type, a Metal storage mode, host pages. Which
//! kinds exist, what each is called and how it is allocated are the
//! backend's, so here a kind is opaque ([`NativeKind`]) with the facts its
//! backend reports of it. A kind holds no measured figure: what running on
//! it costs is measured by whoever prices it.

use crate::model::{AccessPathId, ExecutorId, KindId, Reservation};
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

/// How a kind's memory is obtained.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Obtained {
    /// Through the device's own allocator.
    Allocated,
    /// Host pages the process obtained itself, wrapped for the device
    /// (`cuMemHostRegister`, `VK_EXT_external_memory_host`).
    ImportedHostPages,
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

/// How a kind reserves address ranges: none where it does not, and why it
/// cannot where it would but its device or host refuses.
pub type Reserving = Result<Option<Reservation>, Unreservable>;

/// Why a kind that would reserve address ranges cannot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Unreservable {
    /// What decides it could not be read from the driver or host: a
    /// granule, or the host page.
    Unread(String),
    /// Device memory chunks are zeroed on a second queue of the work queue
    /// family, beside work parked on the first, and it has `queues`.
    OneQueue { queues: u32 },
    /// Host pages' chunks are imported host memory, and it imports none.
    NoHostImport,
    /// It imports host memory at `alignment` bytes, beyond the `page`-byte
    /// host page a reserved range's chunks start on.
    HostImportAlignment { alignment: u64, page: u64 },
    /// Chunks mapped in place need a placed mapping, and it maps none.
    NoPlacedMapping,
    /// It maps memory in place at `alignment` bytes, beyond the
    /// `least`-byte least chunk.
    PlacedMappingAlignment { alignment: u64, least: u64 },
    /// A recorded report records neither the queue count nor the host
    /// import that decide it.
    Unrecorded,
}

impl std::fmt::Display for Unreservable {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unread(why) => write!(formatter, "what decides its reservation could not be read: {why}"),
            Self::OneQueue { queues } => write!(
                formatter,
                "its compute queue family has {queues} queue, and a reservation's growth is zeroed on a second \
                 beside work parked on the first"
            ),
            Self::NoHostImport => formatter.write_str(
                "it imports no host memory (VK_EXT_external_memory_host), which host pages' chunks are",
            ),
            Self::HostImportAlignment { alignment, page } => write!(
                formatter,
                "it imports host memory at {alignment} bytes, beyond the {page}-byte host page a reserved \
                 range's chunks start on"
            ),
            Self::NoPlacedMapping => formatter.write_str("it maps no memory at a placed address"),
            Self::PlacedMappingAlignment { alignment, least } => write!(
                formatter,
                "it maps memory in place at {alignment} bytes, beyond the {least}-byte least chunk"
            ),
            Self::Unrecorded => formatter.write_str(
                "its report records no queue count or host import, which decide how it reserves",
            ),
        }
    }
}

impl std::error::Error for Unreservable {}

/// What a backend reports of one of its kinds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KindReport {
    /// The backend's identifier of the kind, stable across runs.
    pub id: u32,
    /// The backend's name of the kind, for people.
    pub name: String,
    pub reach: Reach,
    pub obtained: Obtained,
    pub memory: KindMemory,
    pub reservation: Reserving,
}

/// One kind of one executor in a revision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Kind {
    pub(crate) id: KindId,
    pub(crate) executor: ExecutorId,
    pub(crate) native: NativeKind,
    pub(crate) name: String,
    pub(crate) reach: Reach,
    pub(crate) obtained: Obtained,
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
    /// How its memory is obtained.
    pub fn obtained(&self) -> Obtained {
        self.obtained
    }
    /// The access paths its allocations are charged along, in order.
    pub fn paths(&self) -> &[AccessPathId] {
        &self.paths
    }
}
