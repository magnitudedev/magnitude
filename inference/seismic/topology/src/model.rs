//! The graph: physical devices, executors and stores, the
//! access and transfer paths between them, and each executor's placements.
//!
//! A [`DeviceTopology`] is one immutable revision. Its identifiers carry that
//! revision, are issued when the revision is built and are never renumbered:
//! a lookup with an identifier of another revision finds nothing. Only the
//! [`Builder`] constructs a topology, and it states the invariants by its
//! shape: every path is created between nodes it issued, and an executor's
//! placements name only the access paths created for that executor.

use crate::kind::{Kind, KindReport, NativeKind, Reserving, Unreservable};
use crate::placement::{Contract, Placing, ReservedContract};
use crate::selector::ExecutorSelector;
use seismic_lang::registry::BackendName;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::num::NonZeroU64;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// One topology revision. Revisions are unique within the process, so an
/// identifier of one topology never resolves in another.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Revision(u64);

impl Revision {
    /// A revision no topology of this process has had.
    pub fn next() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        Self(NEXT.fetch_add(1, Ordering::Relaxed))
    }
}

macro_rules! identifier {
    ($(#[$doc:meta])* $name:ident) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name {
            revision: Revision,
            index: u32,
        }

        impl $name {
            /// The revision that issued this identifier.
            pub fn revision(self) -> Revision {
                self.revision
            }
        }
    };
}

identifier!(
    /// A physical device of one revision.
    PhysicalDeviceId
);
identifier!(
    /// An executor of one revision.
    ExecutorId
);
identifier!(
    /// A store of one revision.
    StoreId
);
identifier!(
    /// An access path of one revision.
    AccessPathId
);
identifier!(
    /// A transfer path of one revision.
    TransferPathId
);
identifier!(
    /// A storage kind of one revision.
    KindId
);

fn index(position: usize) -> u32 {
    u32::try_from(position).expect("a topology holds fewer than 2^32 nodes of a kind")
}

/// A value an API reports, with the API that reported it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Reported<T> {
    /// The value, as the API reports it.
    pub value: T,
    /// The API, and the figure of it, that reported the value.
    pub by: ReportedBy,
}

impl<T> Reported<T> {
    /// `value`, as `by` reports it.
    pub fn new(value: T, by: ReportedBy) -> Self {
        Self { value, by }
    }
}

/// The interface, and the figure of it, a reported value comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum ReportedBy {
    /// Linux `/proc/meminfo` `MemTotal`: RAM the OS can use.
    LinuxMemTotal,
    /// macOS `sysctl hw.memsize`: installed RAM.
    MacOsMemsize,
    /// Windows `GlobalMemoryStatusEx` `ullTotalPhys`: RAM the OS can use.
    WindowsTotalPhys,
    /// Linux `/proc/meminfo` `MemAvailable`.
    LinuxMemAvailable,
    /// macOS Mach `HOST_VM_INFO64`: free, active and inactive pages, every
    /// page neither wired nor held by the compressor: what the OS can make
    /// available, compressing other programs' data if it must.
    MachMovablePages,
    /// macOS Mach `HOST_VM_INFO64`: free (less speculative), file-backed and
    /// purgeable pages, what can be taken without compressing or swapping a
    /// program's data.
    MachAllocatablePages,
    /// Windows `GlobalMemoryStatusEx`: the lesser of available physical
    /// memory and available commit.
    WindowsPhysicalAndCommit,
    /// The CPU backend's largest allocation, the host's address-size limit.
    CpuAddressSpace,
    /// `MTLDevice.maxBufferLength`.
    MetalMaxBufferLength,
    /// `MTLDevice.recommendedMaxWorkingSetSize`.
    MetalRecommendedWorkingSet,
    /// `recommendedMaxWorkingSetSize` less `currentAllocatedSize`.
    MetalWorkingSetUnallocated,
    /// `cuDeviceTotalMem`: memory of the exposed device or partition.
    CudaTotalMemory,
    /// The memory a CUDA device lets this process use
    /// (`seismic_cuda::usable_memory_bytes`).
    CudaUsableMemory,
    /// `cuMemGetInfo` free memory of an opened discrete device.
    CudaFree,
    /// The size of one Vulkan memory heap.
    VulkanHeap,
    /// `VK_EXT_memory_budget`: the heaps' budgets less their usage, attributed
    /// along the executor's placement.
    VulkanHeapBudget,
    /// The sizes of a Vulkan device's memory heaps together.
    VulkanHeaps,
    /// The sizes of a Vulkan device's heaps together, less the dedicated
    /// memory the OS reports for its adapter.
    VulkanHeapsBeyondDedicated,
    /// `min(maxMemoryAllocationSize, maxBufferSize)`.
    VulkanMaxAllocation,
    /// CUDA's device total less the dedicated memory the OS reports for
    /// its adapter.
    CudaTotalBeyondDedicated,
    /// Metal's recommended working set less the dedicated memory the OS
    /// reports for its adapter.
    MetalWorkingSetBeyondDedicated,
    /// DXGI `DXGI_ADAPTER_DESC1::DedicatedVideoMemory` and
    /// `DedicatedSystemMemory` together: the memory the OS does not manage
    /// that it reports as the adapter's.
    DxgiDedicatedMemory,
    /// Linux cacheinfo (`/sys/devices/system/cpu/cpu*/cache/index*`): the
    /// highest-level data or unified caches, summed over their instances.
    LinuxSysfsCache,
    /// macOS `sysctl hw.perflevel*.l{2,3}cachesize` over
    /// `hw.perflevel*.cpusperl{2,3}`: each core type's last-level caches,
    /// summed over their instances.
    MacOsSysctlCache,
    /// Windows `GetLogicalProcessorInformationEx(RelationCache)`: the
    /// highest-level data or unified caches, summed over their records.
    WindowsProcessorCache,
    /// `CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE`.
    CudaL2Cache,
    /// The filesystem's size (`statvfs` blocks, `GetDiskFreeSpaceExW`
    /// total).
    FilesystemSize,
    /// The filesystem's space available to this process (`statvfs`
    /// available blocks, `GetDiskFreeSpaceExW` free to the caller).
    FilesystemAvailable,
}

/// One piece of hardware, by the identity its drivers share.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhysicalDevice {
    id: PhysicalDeviceId,
    identity: PhysicalIdentity,
    memory: DeviceMemory,
    cache: Option<Reported<u64>>,
}

impl PhysicalDevice {
    /// The device's identifier in its revision.
    pub fn id(&self) -> PhysicalDeviceId {
        self.id
    }
    /// The identity the device's drivers share, across revisions.
    pub fn identity(&self) -> &PhysicalIdentity {
        &self.identity
    }
    /// Whose memory the device's is, as discovery decided it once for the
    /// device from its backends' reports.
    pub fn memory(&self) -> DeviceMemory {
        self.memory
    }
    /// The device's last-level cache as an API reports it: the host CPU's
    /// summed over its instances, a CUDA device's L2. None where no API
    /// reports one (Metal, Vulkan, and every memory-side cache).
    pub fn cache(&self) -> Option<Reported<u64>> {
        self.cache
    }
}

/// Whose memory a physical device's is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum DeviceMemory {
    /// The host's, which the OS manages: the host CPU's, and a GPU's whose
    /// backends report it so (Vulkan's device type, CUDA's `integrated` or
    /// NUMA node, Metal's `hasUnifiedMemory`). Its executors place on the
    /// memory the OS reports as dedicated to its adapter first, then on
    /// host RAM.
    Host,
    /// Its own, which the OS does not manage.
    Own,
    /// Its backends report it differently, or none states it, so no rule
    /// places on it: its executors are unavailable.
    Undecided,
}

/// The identity a physical device's drivers share.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PhysicalIdentity {
    /// The host: its CPU as the OS reports it (model, instruction set,
    /// caches) and its RAM, whose change invalidates what was measured on
    /// it.
    Host { machine: String },
    /// A GPU's device UUID, which CUDA (`cuDeviceGetUuid_v2`) and Vulkan
    /// (`VkPhysicalDeviceIDProperties::deviceUUID`) report alike, so their
    /// views of one GPU share it. It is the device's across boots, unlike
    /// the OS adapter LUID (WDDM issues a new one each boot), which only
    /// matches the dedicated memory the OS reports to its adapter.
    Uuid([u8; 16]),
    /// `MTLDevice.registryID`.
    Registry(u64),
}

impl fmt::Display for PhysicalIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Host { .. } => f.write_str("host"),
            Self::Registry(id) => write!(f, "registry:{id:016x}"),
            Self::Uuid(uuid) => {
                f.write_str("uuid:")?;
                for byte in uuid {
                    write!(f, "{byte:02x}")?;
                }
                Ok(())
            }
        }
    }
}

/// An executor's hardware and driver across revisions: the physical
/// device, the backend and the driver as the backend reports it. What a
/// measurement of work on the executor is keyed by, so it is taken again
/// when the hardware or driver under it changes.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Identity {
    /// The physical device's identity.
    pub physical: PhysicalIdentity,
    /// The backend.
    pub backend: BackendName,
    /// The driver, as the backend reports it.
    pub driver: String,
}

/// A store across revisions.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum StoreIdentity {
    /// The host's RAM.
    HostRam,
    /// The own memory of the physical device of this identity.
    DeviceMemory(PhysicalIdentity),
    /// The volume of this identity.
    Volume(VolumeId),
}

impl fmt::Display for StoreIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HostRam => f.write_str("host-ram"),
            Self::DeviceMemory(device) => write!(f, "device-memory:{device}"),
            Self::Volume(volume) => {
                write!(f, "volume:{:x}", volume.device)?;
                if let Some(uuid) = &volume.uuid {
                    write!(f, ":{uuid}")?;
                }
                Ok(())
            }
        }
    }
}

/// Whether an executor can run, with the reason when it cannot.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub enum Availability {
    /// The executor can run.
    Available,
    /// The executor cannot run, for `reason`.
    Unavailable { reason: String },
}

/// What an executor's API reports of single allocations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Limits {
    /// The largest single allocation; neither capacity nor availability.
    pub max_allocation: Reported<u64>,
}

/// What a reserved range's records are: each `record` bytes, and the most
/// bytes of any one view laid whole in them beside their own records
/// (`carries`, a packing of another form's records in them; none for a range
/// of its records alone). On chunked storage each must lie in one chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReservedRecords {
    pub record: NonZeroU64,
    pub carries: Option<NonZeroU64>,
}

impl ReservedRecords {
    /// A range of `record`-byte records, holding nothing else.
    pub const fn of(record: NonZeroU64) -> Self {
        Self {
            record,
            carries: None,
        }
    }

    /// The range also holding views of up to `view` bytes whole.
    pub fn carrying(self, view: NonZeroU64) -> Self {
        Self {
            carries: Some(self.carries.map_or(view, |carries| carries.max(view))),
            ..self
        }
    }
}

/// How a backend reserves a tensor's address range and backs its committed
/// bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reservation {
    /// The granule in which it backs the committed bytes, and charges them.
    pub granule: u64,
    /// Whether the host addresses the backed bytes in place, so a read can
    /// land in a pinned region of them: a property of the backend's reserved
    /// storage, which each backend declares once per kind
    /// (host ranges for the CPU's and Metal's, CUDA's host-node memory and
    /// registered pages, Vulkan's imported pages and placed mappings) and
    /// its storage tests hold it to. A reserved `Shared` buffer requires it.
    pub host_addressed: bool,
    /// Where device work reaches the backed bytes through an address table
    /// of chunks, each its own native allocation (Metal's and Vulkan's
    /// buffers), rather
    /// than at one base address: the bytes every chunk is a whole number of
    /// (where it starts and ends; Metal's, a host page). A record must then
    /// lie in one chunk ([`Reservation::of_records`]). Declared once by each
    /// backend (`RESERVATION_CHUNKS`).
    pub chunks: Option<ReservationChunks>,
}

impl Reservation {
    /// Whether device work reaches the backed bytes through a table of
    /// chunks ([`Self::chunks`]).
    pub fn chunked(&self) -> bool {
        self.chunks.is_some()
    }

    /// What backing a reserved range's leading `bytes` holds, and is
    /// charged: the whole granules covering them; `None` past the bytes a
    /// `u64` counts.
    pub fn backing(&self, bytes: u64) -> Option<u64> {
        bytes.div_ceil(self.granule).checked_mul(self.granule)
    }

    /// The reservation of a tensor of `records`. Chunked storage backs,
    /// charges and addresses chunks of whole records and whole chunk units,
    /// the least such chunk no smaller than the granule nor than the largest
    /// view the records carry, so no record and no carried view crosses a
    /// chunk and a chunk holds as little past the committed records as its
    /// rules admit; other storage keeps its granule.
    pub fn of_records(&self, records: ReservedRecords) -> Self {
        let granule = match self.chunks {
            Some(ReservationChunks { unit, .. }) => {
                let record = records.record.get();
                let whole = unit / gcd(unit, record) * record;
                let least = records
                    .carries
                    .map_or(self.granule, |carries| self.granule.max(carries.get()));
                least.div_ceil(whole) * whole
            }
            None => self.granule,
        };
        Self { granule, ..*self }
    }

    /// The largest single native allocation of a reserved range's `backing`
    /// ([`Self::backing`]), which the device's single-allocation limit
    /// bounds: a chunk of chunked storage, otherwise the whole backing.
    pub fn largest_allocation(&self, backing: u64) -> u64 {
        match self.chunked() {
            true => self.granule.min(backing),
            false => backing,
        }
    }

    /// How views of `record`-byte records lie in a reserved range
    /// ([`Packing`]); `None` for a record of no bytes, or on chunked storage
    /// one larger than a chunk, which no chunk holds whole.
    pub fn packing(&self, record: u64) -> Option<Packing> {
        let record = NonZeroU64::new(record)?;
        match self.chunks {
            Some(_) if record.get() > self.granule => None,
            Some(_) => Some(Packing {
                record,
                chunk: Some(self.granule),
            }),
            None => Some(Packing::contiguous(record)),
        }
    }

    /// The bytes of the address table device work reaches `records` records
    /// of a reserved range through: a record table
    /// (`seismic_lang::registry::record_table_bytes`) on chunked storage,
    /// none on other storage; `None` past the bytes a `u64` counts.
    pub fn table_bytes(&self, records: u64) -> Option<u64> {
        match self.chunked() {
            true => seismic_lang::registry::record_table_bytes(records),
            false => Some(0),
        }
    }

    /// What a reserved range of `records` records of `held` is charged with
    /// its leading `bytes` backed: their backing in its records'
    /// reservation ([`Self::of_records`]) and its record table; `None` past
    /// the bytes a `u64` counts.
    pub fn charge(&self, held: ReservedRecords, records: u64, bytes: u64) -> Option<u64> {
        let reservation = self.of_records(held);
        reservation
            .backing(bytes)?
            .checked_add(reservation.table_bytes(records)?)
    }

    /// What a reservation under `contract` is charged ([`Self::charge`]),
    /// its backing and its record table apart: a table under the backing's
    /// own contract is charged with it, one under another on its own.
    pub fn charges(
        &self,
        contract: ReservedContract,
        held: ReservedRecords,
        records: u64,
        bytes: u64,
    ) -> Option<ReservationCharges> {
        let reservation = self.of_records(held);
        let table = match reservation.chunks {
            Some(chunks) => {
                let bytes = reservation.table_bytes(records)?;
                let placement = chunks.table;
                Some(if placement == contract.contract() {
                    TableCharge::WithBacking { placement, bytes }
                } else {
                    TableCharge::Separate { placement, bytes }
                })
            }
            None => None,
        };
        Some(ReservationCharges {
            contract,
            backing: reservation.backing(bytes)?,
            table,
        })
    }

    /// The most bytes a reserved range backs with each of its native
    /// allocations within `max_allocation`.
    pub fn most_backed(&self, max_allocation: u64) -> u64 {
        match (self.chunked(), self.granule <= max_allocation) {
            (true, true) => u64::MAX,
            (_, false) => 0,
            (false, true) => max_allocation / self.granule * self.granule,
        }
    }
}

/// Chunk geometry and the contract of its record table: ordinary allocated
/// storage the backend names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReservationChunks {
    pub unit: u64,
    pub table: Contract,
}

/// The physical holdings of a reservation under `contract`
/// ([`Reservation::charges`]). Only backing changes on resize.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReservationCharges {
    pub contract: ReservedContract,
    pub backing: u64,
    pub table: Option<TableCharge>,
}

/// A chunked reservation's record table charge, in its placement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TableCharge {
    /// Under the backing's own contract: charged with the backing.
    WithBacking { placement: Contract, bytes: u64 },
    /// Under another: charged on its own.
    Separate { placement: Contract, bytes: u64 },
}

impl TableCharge {
    pub const fn placement(self) -> Contract {
        match self {
            Self::WithBacking { placement, .. } | Self::Separate { placement, .. } => placement,
        }
    }

    pub const fn bytes(self) -> u64 {
        match self {
            Self::WithBacking { bytes, .. } | Self::Separate { bytes, .. } => bytes,
        }
    }
}

impl ReservationCharges {
    pub fn total(self) -> Option<u64> {
        self.backing
            .checked_add(self.table.map_or(0, TableCharge::bytes))
    }
}

/// How views of records of one size lie in storage, the one rule for every
/// view of a range ([`Reservation::packing`]). On chunked storage a view's
/// first record is at the first place at or after its start where a whole
/// record fits in one chunk, and each next record follows the last, moving
/// to the next chunk boundary when it would cross one, so no record crosses
/// a chunk and views of different record sizes may follow one another
/// ([`Self::end`]); a view of records that fill chunks exactly (the range's
/// own, [`Reservation::of_records`]) is contiguous from a record-aligned
/// start. Other storage has no chunk to cross: its records follow one
/// another.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Packing {
    record: NonZeroU64,
    /// The bytes of a chunk, no fewer than a record's, on chunked storage.
    chunk: Option<u64>,
}

impl Packing {
    /// Records of `record` bytes one after another, in storage without
    /// chunks.
    pub fn contiguous(record: NonZeroU64) -> Self {
        Self {
            record,
            chunk: None,
        }
    }

    /// The bytes of one record.
    pub fn record(&self) -> u64 {
        self.record.get()
    }

    /// Whether records are packed in chunks.
    pub fn chunked(&self) -> bool {
        self.chunk.is_some()
    }

    /// Where record `index` of a view from byte `start` lies, in bytes from
    /// the storage's first.
    pub fn at(&self, start: u64, index: u64) -> u64 {
        let record = self.record();
        let Some(chunk) = self.chunk else {
            return start + index * record;
        };
        let first = match start % chunk + record <= chunk {
            true => start,
            false => start.next_multiple_of(chunk),
        };
        // Whole records left in the first record's chunk, then whole chunks.
        let leading = (chunk - first % chunk) / record;
        match index.checked_sub(leading) {
            None => first + index * record,
            Some(rest) => {
                let per = chunk / record;
                (first / chunk + 1 + rest / per) * chunk + rest % per * record
            }
        }
    }

    /// The byte just past the last of `records` records of a view from
    /// `start` ([`Self::at`]), `start` for none: where the next view may
    /// start.
    pub fn end(&self, start: u64, records: u64) -> u64 {
        records
            .checked_sub(1)
            .map_or(start, |last| self.at(start, last) + self.record())
    }
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// The kinds a buffer of one placing is allocated from, in order,
/// and their access paths, each once.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Placed {
    kinds: Vec<KindId>,
    paths: Vec<AccessPathId>,
}

/// One backend's handle on a physical device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Executor {
    id: ExecutorId,
    physical: PhysicalDeviceId,
    selector: ExecutorSelector,
    kind: ExecutorKind,
    name: String,
    availability: Availability,
    driver: String,
    limits: Limits,
    tensor_operations: bool,
    /// Each placing's placement, by [`Placing::index`].
    placed: [Placed; Placing::COUNT],
    /// How each reserved contract is reserved (`placement::reserving`).
    reserving: [Reserving; ReservedContract::COUNT],
}

impl Executor {
    /// The executor's identifier in its revision.
    pub fn id(&self) -> ExecutorId {
        self.id
    }
    /// The physical device the executor is a handle on.
    pub fn physical(&self) -> PhysicalDeviceId {
        self.physical
    }
    /// The backend the executor is of.
    pub fn backend(&self) -> BackendName {
        self.selector.backend()
    }
    /// The identity the executor's own API opens it by.
    pub fn selector(&self) -> ExecutorSelector {
        self.selector
    }
    /// What the executor runs on, as its API describes it.
    pub fn kind(&self) -> ExecutorKind {
        self.kind
    }
    /// The device's name as the backend reports it.
    pub fn name(&self) -> &str {
        &self.name
    }
    /// Whether the executor can run.
    pub fn availability(&self) -> &Availability {
        &self.availability
    }
    /// The driver this executor runs on, as its backend reports it without
    /// opening the executor (the OS build for the CPU and Metal).
    pub fn driver(&self) -> &str {
        &self.driver
    }
    /// What the executor's API reports it cannot exceed.
    pub fn limits(&self) -> &Limits {
        &self.limits
    }
    /// Whether the executor forms Metal tensor operations: Apple GPU family
    /// 10 or later with a compiler that forms the tensor probe
    /// (`seismic_metal::profile::forms_tensor_operations`), the fact the
    /// opened device's native sources are formed under. Always false on the
    /// other backends, whose kernels have no such form.
    pub fn forms_tensor_operations(&self) -> bool {
        self.tensor_operations
    }
    /// The kinds a buffer placed as `placing` is allocated from, in order:
    /// an allocation lands on the first with room along its paths
    /// (`placement::place`). Empty when no kind is admitted, and for an
    /// unavailable executor.
    pub fn placement(&self, placing: Placing) -> &[KindId] {
        &self.placed[placing.index()].kinds
    }
    /// The access paths of [`Self::placement`], in order, each once.
    pub fn paths(&self, placing: Placing) -> &[AccessPathId] {
        &self.placed[placing.index()].paths
    }
    /// How a buffer of `contract` is reserved: none where no kind reserves
    /// it, and why not where one would but cannot.
    pub fn reserving(&self, contract: ReservedContract) -> Result<Option<Reservation>, &Unreservable> {
        self.reserving[contract as usize].as_ref().copied()
    }
}

/// What an executor runs on, as its API describes it (Vulkan's device type,
/// CUDA's `integrated`, Metal's `hasUnifiedMemory`). Descriptive only:
/// whose memory a device's is, discovery decides once per physical device
/// ([`PhysicalDevice::memory`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum ExecutorKind {
    /// A CPU.
    Cpu,
    /// A GPU its API describes as integrated.
    IntegratedGpu,
    /// A GPU its API describes as discrete.
    DiscreteGpu,
    /// A device type that states neither.
    Other,
}

/// A place bytes live, with one ledger per process.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Store {
    id: StoreId,
    kind: StoreKind,
    capacity: Reported<u64>,
}

impl Store {
    /// The store's identifier in its revision.
    pub fn id(&self) -> StoreId {
        self.id
    }
    /// What the store is.
    pub fn kind(&self) -> &StoreKind {
        &self.kind
    }
    /// The store's size, as an API reports it.
    pub fn capacity(&self) -> Reported<u64> {
        self.capacity
    }
}

/// What a store is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StoreKind {
    /// Memory the OS manages.
    HostRam,
    /// Memory the OS does not manage, owned by one physical device.
    DeviceMemory(PhysicalDeviceId),
    /// A filesystem on a block device.
    Volume(VolumeId),
}

/// A volume by the OS's identity of it: the device number its files report
/// (`st_dev`; the volume serial number on Windows) and, where the OS gives
/// one, the volume's UUID.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct VolumeId {
    /// The device number the volume's files report.
    pub device: u64,
    /// The volume's UUID, where the OS gives one.
    pub uuid: Option<String>,
}

/// An executor reading and writing a store directly, through one grant:
/// what the driver lets this process charge there. The kinds the executor
/// allocates are charged along their paths ([`Kind::paths`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AccessPath {
    id: AccessPathId,
    executor: ExecutorId,
    store: StoreId,
    grant: Option<Grant>,
}

/// A path's grant, and where its live status is read without an opened
/// executor, where it can be.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Grant {
    reported: Reported<u64>,
    source: Option<GrantSource>,
}

impl AccessPath {
    /// The path's identifier in its revision.
    pub fn id(&self) -> AccessPathId {
        self.id
    }
    /// The executor reading and writing along the path.
    pub fn executor(&self) -> ExecutorId {
        self.executor
    }
    /// The store the path reaches.
    pub fn store(&self) -> StoreId {
        self.store
    }
    /// What the driver lets this process charge along the path; none for a
    /// path no driver grants, which only the store's own headroom bounds.
    pub fn grant(&self) -> Option<Reported<u64>> {
        self.grant.as_ref().map(|grant| grant.reported)
    }
    /// Where the grant's live status is read without an opened executor.
    pub(crate) fn grant_source(&self) -> Option<(u64, &GrantSource)> {
        let grant = self.grant.as_ref()?;
        Some((grant.reported.value, grant.source.as_ref()?))
    }
}

/// Bytes moved from one store to another by an executor's copy mechanism.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransferPath {
    id: TransferPathId,
    from: StoreId,
    to: StoreId,
    by: ExecutorId,
}

impl TransferPath {
    /// The path's identifier in its revision.
    pub fn id(&self) -> TransferPathId {
        self.id
    }
    /// The store the bytes are copied from.
    pub fn from(&self) -> StoreId {
        self.from
    }
    /// The store the bytes are copied to.
    pub fn to(&self) -> StoreId {
        self.to
    }
    /// The executor whose copy mechanism moves them.
    pub fn by(&self) -> ExecutorId {
        self.by
    }
}

/// Who a discovery diagnostic is about.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiagnosticSource {
    /// A backend's enumeration.
    Backend(BackendName),
    /// The OS's adapter interface (DXGI).
    Adapters,
}

/// An enumeration problem retained from discovery: an absent driver, a
/// device that could not be described.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagnostic {
    /// Who the problem is about.
    pub source: DiagnosticSource,
    /// The problem, as its source reported it.
    pub message: String,
}

impl Diagnostic {
    /// The diagnostic of a backend whose enumeration succeeded with no
    /// device: every backend yields devices or a diagnostic.
    pub fn no_device(backend: BackendName) -> Self {
        Self {
            source: DiagnosticSource::Backend(backend),
            message: format!("{} reports no device", backend.as_str()),
        }
    }
}

/// One immutable revision of the graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceTopology {
    revision: Revision,
    physical: Vec<PhysicalDevice>,
    executors: Vec<Executor>,
    stores: Vec<Store>,
    access: Vec<AccessPath>,
    kinds: Vec<Kind>,
    transfer: Vec<TransferPath>,
    /// Each path the inventory serves and the volume it is on.
    volumes: Vec<(PathBuf, StoreId)>,
    diagnostics: Vec<Diagnostic>,
}

/// Where a grant's live status is read without an opened executor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GrantSource {
    /// The `VK_EXT_memory_budget` of the Vulkan device `uuid`'s `heaps`:
    /// their budgets less usage, of which the path's grant takes what is
    /// left once `before` bytes of usage are attributed to the paths placed
    /// ahead of it, up to the grant.
    VulkanHeaps {
        uuid: [u8; 16],
        heaps: Vec<usize>,
        before: u64,
    },
    /// The Metal device's recommended working set less what this process
    /// has allocated on it, attributed as `VulkanHeaps` is.
    MetalWorkingSet { registry_id: u64, before: u64 },
}

impl DeviceTopology {
    /// The revision this topology is.
    pub fn revision(&self) -> Revision {
        self.revision
    }
    /// The persistent identity of an executor of this revision; none for an
    /// executor of another.
    pub fn executor_identity(&self, executor: ExecutorId) -> Option<Identity> {
        let executor = self.executor(executor)?;
        Some(Identity {
            physical: self
                .physical_device(executor.physical())
                .expect("an executor's device is of its revision")
                .identity()
                .clone(),
            backend: executor.backend(),
            driver: executor.driver().to_owned(),
        })
    }
    /// The persistent identity of a store of this revision; none for a
    /// store of another.
    pub fn store_identity(&self, id: StoreId) -> Option<StoreIdentity> {
        Some(match self.store(id)?.kind() {
            StoreKind::HostRam => StoreIdentity::HostRam,
            StoreKind::DeviceMemory(device) => StoreIdentity::DeviceMemory(
                self.physical_device(*device)
                    .expect("store device belongs to its revision")
                    .identity()
                    .clone(),
            ),
            StoreKind::Volume(volume) => StoreIdentity::Volume(volume.clone()),
        })
    }
    /// Every physical device, the host's first.
    pub fn physical_devices(&self) -> &[PhysicalDevice] {
        &self.physical
    }
    /// Every executor, available or not.
    pub fn executors(&self) -> &[Executor] {
        &self.executors
    }
    /// The executor `selector` opens, when this revision has it.
    pub fn resolve(&self, selector: ExecutorSelector) -> Option<&Executor> {
        self.executors
            .iter()
            .find(|executor| executor.selector == selector)
    }
    /// Every store, host RAM first.
    pub fn stores(&self) -> &[Store] {
        &self.stores
    }
    /// Every access path.
    pub fn access_paths(&self) -> &[AccessPath] {
        &self.access
    }

    /// Every executor's storage kinds.
    pub fn kinds(&self) -> &[Kind] {
        &self.kinds
    }
    /// Every transfer path.
    pub fn transfer_paths(&self) -> &[TransferPath] {
        &self.transfer
    }
    /// The enumeration problems discovery retained.
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }

    /// The physical device `id`; none for an identifier of another revision.
    pub fn physical_device(&self, id: PhysicalDeviceId) -> Option<&PhysicalDevice> {
        (id.revision == self.revision).then(|| &self.physical[id.index as usize])
    }
    /// The executor `id`; none for an identifier of another revision.
    pub fn executor(&self, id: ExecutorId) -> Option<&Executor> {
        (id.revision == self.revision).then(|| &self.executors[id.index as usize])
    }
    /// The store `id`; none for an identifier of another revision.
    pub fn store(&self, id: StoreId) -> Option<&Store> {
        (id.revision == self.revision).then(|| &self.stores[id.index as usize])
    }
    /// The access path `id`; none for an identifier of another revision.
    pub fn access_path(&self, id: AccessPathId) -> Option<&AccessPath> {
        (id.revision == self.revision).then(|| &self.access[id.index as usize])
    }

    pub fn kind(&self, id: KindId) -> Option<&Kind> {
        (id.revision == self.revision).then(|| &self.kinds[id.index as usize])
    }
    /// The transfer path `id`; none for an identifier of another revision.
    /// The store a buffer placed as `placing` on `executor` lands in first.
    pub fn first_store(&self, executor: &Executor, placing: Placing) -> Option<StoreId> {
        executor
            .paths(placing)
            .first()
            .and_then(|path| self.access_path(*path))
            .map(AccessPath::store)
    }
    /// Whether a range the host and `executor` share lies where the device's
    /// own ranges do: the store of reserved `Shared` is that of reserved
    /// `DeviceOnly`, so a reserved buffer the host writes in place reads as
    /// device memory, with no copy across a link.
    pub fn shared_ranges_lie_in_device_memory(&self, executor: &Executor) -> bool {
        let store = |contract| self.first_store(executor, Placing::Reserved(contract));
        store(ReservedContract::Shared).is_some() && store(ReservedContract::Shared) == store(ReservedContract::DeviceOnly)
    }
    pub fn transfer_path(&self, id: TransferPathId) -> Option<&TransferPath> {
        (id.revision == self.revision).then(|| &self.transfer[id.index as usize])
    }

    /// The host RAM store; every topology has exactly one.
    pub fn host_ram(&self) -> &Store {
        &self.stores[HOST_RAM]
    }

    /// The paths the inventory serves on volume `store`.
    pub fn volume_paths(&self, store: StoreId) -> impl Iterator<Item = &Path> {
        self.volumes
            .iter()
            .filter(move |(_, on)| *on == store)
            .map(|(path, _)| path.as_path())
    }

    /// Each grant whose status is read without an opened executor: its
    /// path, its size and where it is read.
    pub fn grant_sources(&self) -> impl Iterator<Item = (&AccessPath, u64, &GrantSource)> {
        self.access.iter().filter_map(|path| {
            let (grant, source) = path.grant_source()?;
            Some((path, grant, source))
        })
    }
}

/// The host RAM store's and the host CPU's index in every revision.
const HOST_RAM: usize = 0;
const HOST: usize = 0;

/// Builds one revision. Nodes are added before the paths between them, and
/// each executor's paths and placements through its [`ExecutorBuilder`].
pub(crate) struct Builder {
    topology: DeviceTopology,
}

impl Builder {
    /// A revision holding the host of identity `host`, whose CPU's
    /// last-level cache is `cache`, and host RAM of `capacity`.
    pub(crate) fn new(
        revision: Revision,
        host: PhysicalIdentity,
        cache: Option<Reported<u64>>,
        capacity: Reported<u64>,
    ) -> Self {
        let mut builder = Self {
            topology: DeviceTopology {
                revision,
                physical: Vec::new(),
                executors: Vec::new(),
                stores: Vec::new(),
                access: Vec::new(),
                kinds: Vec::new(),
                transfer: Vec::new(),
                volumes: Vec::new(),
                diagnostics: Vec::new(),
            },
        };
        builder.physical(host, DeviceMemory::Host, cache);
        builder.store(StoreKind::HostRam, capacity);
        builder
    }

    pub(crate) fn host(&self) -> PhysicalDeviceId {
        self.topology.physical[HOST].id
    }

    pub(crate) fn host_ram(&self) -> StoreId {
        self.topology.stores[HOST_RAM].id
    }

    /// The physical device of `identity`, already added.
    pub(crate) fn device(&self, identity: &PhysicalIdentity) -> PhysicalDeviceId {
        self.topology
            .physical
            .iter()
            .find(|device| device.identity == *identity)
            .expect("physical devices are added before their executors")
            .id
    }

    /// The physical device of `identity`, whose memory is `memory` and whose
    /// reported cache is `cache`, added when it is new. Both are decided
    /// once for the device.
    pub(crate) fn physical(
        &mut self,
        identity: PhysicalIdentity,
        memory: DeviceMemory,
        cache: Option<Reported<u64>>,
    ) -> PhysicalDeviceId {
        if let Some(device) = self
            .topology
            .physical
            .iter()
            .find(|device| device.identity == identity)
        {
            assert_eq!(device.memory, memory, "a device's memory is decided once");
            assert_eq!(device.cache, cache, "a device's cache is decided once");
            return device.id;
        }
        let id = PhysicalDeviceId {
            revision: self.topology.revision,
            index: index(self.topology.physical.len()),
        };
        self.topology.physical.push(PhysicalDevice {
            id,
            identity,
            memory,
            cache,
        });
        id
    }

    pub(crate) fn store(&mut self, kind: StoreKind, capacity: Reported<u64>) -> StoreId {
        let id = StoreId {
            revision: self.topology.revision,
            index: index(self.topology.stores.len()),
        };
        self.topology.stores.push(Store { id, kind, capacity });
        id
    }

    /// The store of `kind`, when one was added.
    pub(crate) fn find_store(&self, kind: &StoreKind) -> Option<StoreId> {
        self.topology
            .stores
            .iter()
            .find(|store| store.kind == *kind)
            .map(|store| store.id)
    }

    /// Records that the inventory serves `path`, on volume `store`.
    pub(crate) fn serve(&mut self, path: PathBuf, store: StoreId) {
        self.topology.volumes.push((path, store));
    }

    pub(crate) fn executor(&mut self, executor: ExecutorReport) -> ExecutorBuilder<'_> {
        let id = ExecutorId {
            revision: self.topology.revision,
            index: index(self.topology.executors.len()),
        };
        self.topology.executors.push(Executor {
            id,
            physical: executor.physical,
            selector: executor.selector,
            kind: executor.kind,
            name: executor.name,
            availability: executor.availability,
            driver: executor.driver,
            limits: executor.limits,
            tensor_operations: executor.tensor_operations,
            placed: Default::default(),
            reserving: std::array::from_fn(|_| Ok(None)),
        });
        ExecutorBuilder {
            builder: self,
            executor: id,
            paths: Vec::new(),
            kinds: Vec::new(),
        }
    }

    pub(crate) fn transfer(&mut self, from: StoreId, to: StoreId, by: ExecutorId) {
        let id = TransferPathId {
            revision: self.topology.revision,
            index: index(self.topology.transfer.len()),
        };
        self.topology
            .transfer
            .push(TransferPath { id, from, to, by });
    }

    pub(crate) fn diagnostic(&mut self, diagnostic: Diagnostic) {
        self.topology.diagnostics.push(diagnostic);
    }

    pub(crate) fn topology(&self) -> &DeviceTopology {
        &self.topology
    }

    pub(crate) fn finish(self) -> DeviceTopology {
        self.topology
    }
}

/// An executor's node fields, before its paths.
pub(crate) struct ExecutorReport {
    pub(crate) physical: PhysicalDeviceId,
    pub(crate) selector: ExecutorSelector,
    pub(crate) kind: ExecutorKind,
    pub(crate) name: String,
    pub(crate) availability: Availability,
    pub(crate) driver: String,
    pub(crate) limits: Limits,
    /// Metal tensor operations; false for every other backend.
    pub(crate) tensor_operations: bool,
}

/// Adds one executor's access paths and kinds, and places its storage in
/// them.
pub(crate) struct ExecutorBuilder<'a> {
    builder: &'a mut Builder,
    executor: ExecutorId,
    /// The access paths created for this executor.
    paths: Vec<AccessPathId>,
    /// The kinds created for this executor.
    kinds: Vec<KindId>,
}

/// One of the executor's own access paths, by its position among them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct OwnPath(usize);

/// One of the executor's own kinds, by its position among them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct OwnKind(usize);

impl ExecutorBuilder<'_> {
    /// The access path to `store` that no driver grants: one per store.
    pub(crate) fn ungranted(&mut self, store: StoreId) -> OwnPath {
        let topology = &self.builder.topology;
        if let Some(position) = self.paths.iter().position(|id| {
            let path = &topology.access[id.index as usize];
            path.store == store && path.grant.is_none()
        }) {
            return OwnPath(position);
        }
        self.access(store, None)
    }

    /// A new access path to `store` under `grant`, whose live status is
    /// read from `source` where it needs no opened executor.
    pub(crate) fn granted(
        &mut self,
        store: StoreId,
        grant: Reported<u64>,
        source: Option<GrantSource>,
    ) -> OwnPath {
        let grant = Grant {
            reported: grant,
            source,
        };
        self.access(store, Some(grant))
    }

    fn access(&mut self, store: StoreId, grant: Option<Grant>) -> OwnPath {
        let topology = &mut self.builder.topology;
        let id = AccessPathId {
            revision: topology.revision,
            index: index(topology.access.len()),
        };
        topology.access.push(AccessPath {
            id,
            executor: self.executor,
            store,
            grant,
        });
        self.paths.push(id);
        OwnPath(self.paths.len() - 1)
    }

    /// The kind `report` describes, charged along `paths` in order. A
    /// backend reports each of its kinds once.
    pub(crate) fn kind(&mut self, report: &KindReport, paths: &[OwnPath]) -> OwnKind {
        let backend = self.builder.topology.executors[self.executor.index as usize].backend();
        let native = NativeKind {
            backend,
            id: report.id,
        };
        let topology = &mut self.builder.topology;
        assert!(
            !self
                .kinds
                .iter()
                .any(|id| topology.kinds[id.index as usize].native == native),
            "an executor reports each of its kinds once"
        );
        assert!(
            !paths.is_empty(),
            "a kind is charged along at least one path"
        );
        let id = KindId {
            revision: topology.revision,
            index: index(topology.kinds.len()),
        };
        topology.kinds.push(Kind {
            id,
            executor: self.executor,
            native,
            name: report.name.clone(),
            reach: report.reach,
            obtained: report.obtained,
            paths: paths.iter().map(|path| self.paths[path.0]).collect(),
        });
        self.kinds.push(id);
        OwnKind(self.kinds.len() - 1)
    }

    /// A buffer placed as `placing` is allocated from `kinds`, in order, and
    /// so along their paths, each once.
    pub(crate) fn place(&mut self, placing: Placing, kinds: &[OwnKind]) {
        let topology = &mut self.builder.topology;
        let kinds = kinds
            .iter()
            .map(|kind| self.kinds[kind.0])
            .collect::<Vec<_>>();
        let mut paths = Vec::new();
        for kind in &kinds {
            for path in &topology.kinds[kind.index as usize].paths {
                if !paths.contains(path) {
                    paths.push(*path);
                }
            }
        }
        topology.executors[self.executor.index as usize].placed[placing.index()] = Placed { kinds, paths };
    }

    /// A buffer of `contract` is reserved as `reserving` says.
    pub(crate) fn reserves(&mut self, contract: ReservedContract, reserving: Reserving) {
        self.builder.topology.executors[self.executor.index as usize].reserving
            [contract as usize] = reserving;
    }

    pub(crate) fn id(&self) -> ExecutorId {
        self.executor
    }
}
