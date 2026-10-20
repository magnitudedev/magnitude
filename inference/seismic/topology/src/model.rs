//! The graph (spec §2.1–2.3): physical devices, executors and stores, the
//! access and transfer paths between them, and each executor's placements.
//!
//! A [`DeviceTopology`] is one immutable revision. Its identifiers carry that
//! revision, are issued when the revision is built and are never renumbered:
//! a lookup with an identifier of another revision finds nothing. Only the
//! [`Builder`] constructs a topology, and it states the invariants by its
//! shape: every path is created between nodes it issued, and an executor's
//! placements name only the access paths created for that executor.

use crate::selector::ExecutorSelector;
use seismic_lang::registry::BackendName;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// One topology revision. Revisions are unique within the process, so an
/// identifier of one topology never resolves in another.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Revision(u64);

impl Revision {
    /// A revision no topology of this process has had.
    pub(crate) fn next() -> Self {
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
    /// page neither wired nor held by the compressor.
    MachMovablePages,
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
    /// `CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE` (double data rate) times
    /// `CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH`: a discrete device's
    /// peak memory bandwidth.
    CudaMemoryClockBusWidth,
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
}

/// Whose memory a physical device's is (spec §2.3).
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
    /// The host: its CPU as the OS reports it (model and instruction set)
    /// and its RAM.
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

/// What an executor's API reports it cannot exceed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    /// The largest single allocation; neither capacity nor availability.
    pub max_allocation: Reported<u64>,
    /// The peak bandwidth of the memory it allocates from, where its API
    /// reports one: none but CUDA's for a discrete device.
    pub memory_bandwidth: Option<Reported<u64>>,
}

/// The kinds of storage an executor allocates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StorageKind {
    /// Storage device work reads and writes.
    Device,
    /// Storage the host writes in place while device work reads it
    /// (`Tensor::shared`).
    Upload,
}

impl StorageKind {
    /// How many kinds there are.
    pub const COUNT: usize = 2;
    /// Every kind, in declaration order.
    pub const ALL: [Self; Self::COUNT] = [Self::Device, Self::Upload];
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
    placement: [Vec<AccessPathId>; StorageKind::COUNT],
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
    /// The access paths `kind` storage is allocated from, in order: an
    /// allocation lands on the first with room in its grant. Empty for an
    /// unavailable executor.
    pub fn placement(&self, kind: StorageKind) -> &[AccessPathId] {
        &self.placement[kind as usize]
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

/// What a store is (spec §2.1).
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

/// Where an executor allocates along an access path: the memory its backend
/// takes, and with the path's store the account the runtime charges the
/// allocation to. An executor's paths along one route allocate the same
/// memory and differ only in the store they charge, as an integrated Vulkan
/// device's own memory and host RAM do (§2.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AccessRoute {
    /// The executor's native device storage.
    Device,
    /// Host-visible upload storage, distinct from device storage.
    Upload,
    /// One Vulkan heap, which the backend allocates from directly.
    VulkanHeap(u32),
}

/// An executor reading and writing a store directly, through one grant.
/// Its route is the memory the executor allocates along it; its grant is
/// what the driver lets this process charge there.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AccessPath {
    id: AccessPathId,
    executor: ExecutorId,
    store: StoreId,
    grant: Option<Grant>,
    route: AccessRoute,
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
    /// Where the executor allocates along this path.
    pub fn route(&self) -> AccessRoute {
        self.route
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

/// One immutable revision of the graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceTopology {
    revision: Revision,
    physical: Vec<PhysicalDevice>,
    executors: Vec<Executor>,
    stores: Vec<Store>,
    access: Vec<AccessPath>,
    transfer: Vec<TransferPath>,
    /// Each path the inventory serves and the volume it is on.
    volumes: Vec<(PathBuf, StoreId)>,
    diagnostics: Vec<Diagnostic>,
}

/// Where a grant's live status is read without an opened executor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum GrantSource {
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
    pub(crate) fn grant_sources(&self) -> impl Iterator<Item = (AccessPathId, u64, &GrantSource)> {
        self.access.iter().filter_map(|path| {
            let (grant, source) = path.grant_source()?;
            Some((path.id, grant, source))
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
    /// A revision holding the host of identity `host` and host RAM of
    /// `capacity`.
    pub(crate) fn new(revision: Revision, host: PhysicalIdentity, capacity: Reported<u64>) -> Self {
        let mut builder = Self {
            topology: DeviceTopology {
                revision,
                physical: Vec::new(),
                executors: Vec::new(),
                stores: Vec::new(),
                access: Vec::new(),
                transfer: Vec::new(),
                volumes: Vec::new(),
                diagnostics: Vec::new(),
            },
        };
        builder.physical(host, DeviceMemory::Host);
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

    /// The physical device of `identity`, whose memory is `memory`, added
    /// when it is new. Its memory is decided once for the device.
    pub(crate) fn physical(
        &mut self,
        identity: PhysicalIdentity,
        memory: DeviceMemory,
    ) -> PhysicalDeviceId {
        if let Some(device) = self
            .topology
            .physical
            .iter()
            .find(|device| device.identity == identity)
        {
            assert_eq!(device.memory, memory, "a device's memory is decided once");
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
            placement: Default::default(),
        });
        ExecutorBuilder {
            builder: self,
            executor: id,
            paths: Vec::new(),
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

/// Adds one executor's access paths and places its storage on them.
pub(crate) struct ExecutorBuilder<'a> {
    builder: &'a mut Builder,
    executor: ExecutorId,
    /// The access paths created for this executor.
    paths: Vec<AccessPathId>,
}

/// One of the executor's own access paths, by its position among them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct OwnPath(usize);

impl ExecutorBuilder<'_> {
    /// A new access path to `store` along `route` that no driver grants.
    pub(crate) fn ungranted(&mut self, store: StoreId, route: AccessRoute) -> OwnPath {
        self.access(store, None, route)
    }

    /// A new access path to `store` along `route` under `grant`, whose live
    /// status is read from `source` where it needs no opened executor.
    pub(crate) fn granted(
        &mut self,
        store: StoreId,
        grant: Reported<u64>,
        source: Option<GrantSource>,
        route: AccessRoute,
    ) -> OwnPath {
        let grant = Grant {
            reported: grant,
            source,
        };
        self.access(store, Some(grant), route)
    }

    /// A new access path to `store` along `route`. An executor has at most
    /// one path per store and route: the runtime charges a path's
    /// allocations to the account of its store and route, which a second
    /// such path would share.
    fn access(&mut self, store: StoreId, grant: Option<Grant>, route: AccessRoute) -> OwnPath {
        let topology = &mut self.builder.topology;
        assert!(
            !self.paths.iter().any(|id| {
                let path = &topology.access[id.index as usize];
                path.store == store && path.route == route
            }),
            "an executor has one access path per store and route"
        );
        let id = AccessPathId {
            revision: topology.revision,
            index: index(topology.access.len()),
        };
        topology.access.push(AccessPath {
            id,
            executor: self.executor,
            store,
            grant,
            route,
        });
        self.paths.push(id);
        OwnPath(self.paths.len() - 1)
    }

    /// `kind` storage is allocated from `paths`, in order.
    pub(crate) fn place(&mut self, kind: StorageKind, paths: &[OwnPath]) {
        let placement = paths.iter().map(|path| self.paths[path.0]).collect();
        self.builder.topology.executors[self.executor.index as usize].placement[kind as usize] =
            placement;
    }

    pub(crate) fn id(&self) -> ExecutorId {
        self.executor
    }
}
