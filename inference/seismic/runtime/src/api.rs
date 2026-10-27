//! Public-runtime internals composed by the `seismic` crate. Backends form a
//! private closed sum; callers see devices, tensor descriptors and prepared
//! kernels, never compiler plans or raw buffers.

use seismic_compiler::errors::{ExecutionError, InvocationError};
use seismic_lang::registry::BackendName;
use std::fmt;

/// A stable, immutable host mapping lent to a device. The owner is retained
/// by backend storage through every submission that reads the mapping.
#[derive(Clone)]
pub struct HostRegion {
    pointer: std::ptr::NonNull<u8>,
    len: usize,
    owner: std::sync::Arc<dyn std::any::Any + Send + Sync>,
}

unsafe impl Send for HostRegion {}
unsafe impl Sync for HostRegion {}

impl HostRegion {
    /// # Safety
    /// The pointer must name `len` readable, unchanging bytes for the whole
    /// lifetime of `owner`. The range must be page-aligned for device import.
    pub unsafe fn new(
        pointer: std::ptr::NonNull<u8>,
        len: usize,
        owner: std::sync::Arc<dyn std::any::Any + Send + Sync>,
    ) -> Self {
        Self {
            pointer,
            len,
            owner,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub(crate) fn pointer(&self) -> std::ptr::NonNull<u8> {
        self.pointer
    }
    pub(crate) fn owner(&self) -> std::sync::Arc<dyn std::any::Any + Send + Sync> {
        self.owner.clone()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CallError {
    Invocation(InvocationError),
    Output(OutputError),
    Execution(ExecutionError),
    Workflow(WorkflowError),
}

/// A caller-supplied native result does not match the checked entry contract.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OutputError {
    Count { expected: usize, actual: usize },
    WrongDevice { result: usize },
    WrongRepresentation { result: usize },
    ShapeMismatch { result: usize, axis: usize },
    NoncanonicalLayout { result: usize },
    IllegalAliasing { result: usize },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WorkflowError {
    Empty,
    CrossWorkflowResult,
    MissingProducerResult,
    HostBoundaryRequired,
    ReadOnlyTensor {
        parameter: String,
    },
    TensorView(TensorError),
    NativePortUnbound {
        port: usize,
    },
    NativePortMismatch {
        port: usize,
    },
    NativeGraphArgumentMismatch {
        parameter: usize,
    },
    NativePortAlreadyBound {
        port: usize,
    },
    /// A run's timeline node was not bound to a timeline.
    NativeTimelineUnbound {
        node: usize,
    },
    NativeTimelineAlreadyBound {
        node: usize,
    },
    /// A timeline node was bound to another device's timeline.
    NativeTimelineMismatch {
        node: usize,
    },
    NativeGraphSlotMismatch,
    NativeGraphLayoutMismatch,
    NativeOutputLeaseConsumed,
    NativeExportStillLive,
    /// Every upload region a family slot was created with is still read by
    /// a submission in flight; a slot never grows its regions after sealing.
    UploadRegionsExhausted {
        regions: usize,
    },
    /// A family's workspace exceeds the shared execution arena it binds.
    NativeArenaTooSmall {
        required: u64,
        capacity: u64,
    },
    /// A family that keeps a prewritten port in workspace between runs
    /// cannot bind a shared execution arena.
    NativeArenaPrewrittenPort,
    /// A graph with timeline nodes was given to be measured: a run that
    /// waits on or signals a timeline is not repeatable as the same work.
    NativeGraphTimelineMeasured,
    /// The device's backend is native-only (Vulkan): planned workflows are
    /// refused.
    PlannedRouteUnavailable {
        backend: seismic_lang::registry::BackendName,
    },
}
impl fmt::Display for WorkflowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", CallError::Workflow(self.clone()))
    }
}
impl std::error::Error for WorkflowError {}
impl fmt::Display for CallError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invocation(e) => write!(f, "{e}"),
            Self::Output(e) => write!(f, "{e}"),
            Self::Execution(e) => write!(f, "{e}"),
            Self::Workflow(WorkflowError::CrossWorkflowResult) => {
                f.write_str("workflow result belongs to another workflow")
            }
            Self::Workflow(WorkflowError::Empty) => f.write_str("workflow has no nodes"),
            Self::Workflow(WorkflowError::MissingProducerResult) => {
                f.write_str("workflow result producer is absent or not ordered before its use")
            }
            Self::Workflow(WorkflowError::HostBoundaryRequired) => {
                f.write_str("a scalar workflow dependency requires an explicit host boundary")
            }
            Self::Workflow(WorkflowError::ReadOnlyTensor { parameter }) => {
                write!(f, "parameter `{parameter}` cannot write a mapped tensor")
            }
            Self::Workflow(WorkflowError::TensorView(error)) => write!(f, "{error}"),
            Self::Workflow(WorkflowError::NativePortUnbound { port }) => {
                write!(f, "native graph port {port} is unbound")
            }
            Self::Workflow(WorkflowError::NativePortMismatch { port }) => {
                write!(f, "native graph port {port} has an incompatible tensor")
            }
            Self::Workflow(WorkflowError::NativeGraphArgumentMismatch { parameter }) => {
                write!(
                    f,
                    "native graph argument {parameter} differs from its checked entry"
                )
            }
            Self::Workflow(WorkflowError::NativePortAlreadyBound { port }) => {
                write!(f, "native graph port {port} was bound twice")
            }
            Self::Workflow(WorkflowError::NativeTimelineUnbound { node }) => {
                write!(f, "native graph timeline node {node} is unbound")
            }
            Self::Workflow(WorkflowError::NativeTimelineAlreadyBound { node }) => {
                write!(f, "native graph timeline node {node} was bound twice")
            }
            Self::Workflow(WorkflowError::NativeTimelineMismatch { node }) => {
                write!(f, "native graph timeline node {node} names another device's timeline")
            }
            Self::Workflow(WorkflowError::NativeGraphSlotMismatch) => {
                f.write_str("native graph slot belongs to another graph")
            }
            Self::Workflow(WorkflowError::NativeGraphLayoutMismatch) => {
                f.write_str("native graph shape or topology exceeds its certified layout")
            }
            Self::Workflow(WorkflowError::NativeOutputLeaseConsumed) => {
                f.write_str("native graph output lease was already used")
            }
            Self::Workflow(WorkflowError::NativeExportStillLive) => {
                f.write_str("native graph export is still retained")
            }
            Self::Workflow(WorkflowError::UploadRegionsExhausted { regions }) => write!(
                f,
                "all {regions} upload regions of the native graph slot are in flight"
            ),
            Self::Workflow(WorkflowError::NativeArenaTooSmall { required, capacity }) => write!(
                f,
                "native graph family needs {required} workspace bytes; its execution arena holds {capacity}"
            ),
            Self::Workflow(WorkflowError::NativeArenaPrewrittenPort) => f.write_str(
                "a native graph family with a prewritten port cannot bind a shared execution arena",
            ),
            Self::Workflow(WorkflowError::NativeGraphTimelineMeasured) => {
                f.write_str("a native graph with timeline nodes cannot be measured")
            }
            Self::Workflow(WorkflowError::PlannedRouteUnavailable { backend }) => write!(
                f,
                "`{}` is a native-only backend: planned workflows are unavailable",
                backend.as_str()
            ),
        }
    }
}
impl std::error::Error for CallError {}

impl fmt::Display for OutputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Count { expected, actual } => {
                write!(f, "expected {expected} tensor results, received {actual}")
            }
            Self::WrongDevice { result } => write!(f, "result {result} is on the wrong device"),
            Self::WrongRepresentation { result } => {
                write!(f, "result {result} has the wrong representation")
            }
            Self::ShapeMismatch { result, axis } => {
                write!(f, "result {result} has the wrong extent on axis {axis}")
            }
            Self::NoncanonicalLayout { result } => {
                write!(f, "result {result} does not have canonical storage layout")
            }
            Self::IllegalAliasing { result } => {
                write!(f, "result {result} overlaps another call tensor")
            }
        }
    }
}
impl std::error::Error for OutputError {}

/// Real failures of public tensor construction/view creation. These cannot
/// carry a schema parameter id and therefore are not invocation failures.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TensorError {
    SlabLayout(String),
    UnsupportedRepresentation {
        backend: BackendName,
        representation: &'static str,
    },
    HostByteLength {
        expected: u64,
        actual: u64,
    },
    HostIo(String),
    HostRegion(String),
    SliceOutOfBounds {
        extent: u64,
        start: u64,
        end: u64,
    },
    UnalignedPacketSlice {
        group: u32,
        start: u64,
        end: u64,
    },
    /// A row-layout view may select whole rows only: never its packing axis,
    /// and on an `mma16` row axis only whole 16-row tiles.
    UnalignedRowSlice {
        representation: &'static str,
        start: u64,
        end: u64,
    },
    /// A row-layout reshape must keep the packing axis (and the `mma16` row
    /// axis) whose extents define its row geometry.
    RowLayoutReshape {
        representation: &'static str,
        extents: Vec<u64>,
    },
    ReshapeStorage {
        current_bytes: u64,
        requested_bytes: u64,
    },
    /// A reshape of a view whose storage is not contiguous row-major.
    ReshapeLayout {
        extents: Vec<u64>,
        strides: Vec<u64>,
    },
    /// Host access or a commit beyond a reserved tensor's backed rows, or a
    /// recommit of a tensor that is not a whole reserved tensor.
    Uncommitted {
        rows: u64,
        committed: u64,
    },
    /// A view's store charges cannot be read.
    ViewCharge(ViewChargeError),
    /// A typed region of a view's bytes cannot be formed.
    Region(RegionError),
    Execution(ExecutionError),
}

/// Why a typed region of a view's bytes cannot be formed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegionError {
    /// The view is no one contiguous byte range of a backed allocation: it
    /// is reserved, slabbed or strided.
    NotContiguous,
    /// The region's `bytes` from `offset` reach past the view's.
    Outside {
        offset: u64,
        bytes: u64,
        view_bytes: u64,
    },
    /// The region would start at `offset` in its allocation, off its
    /// layout's `alignment`.
    Unaligned { offset: u64, alignment: u64 },
}

impl fmt::Display for RegionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotContiguous => {
                f.write_str("a region needs a contiguous view of a backed allocation")
            }
            Self::Outside {
                offset,
                bytes,
                view_bytes,
            } => write!(
                f,
                "a region of {bytes} bytes at {offset} reaches past its view's {view_bytes}"
            ),
            Self::Unaligned { offset, alignment } => write!(
                f,
                "a region at allocation offset {offset} is off its layout's {alignment}-byte alignment"
            ),
        }
    }
}

/// Why a view's store charges cannot be read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewChargeError {
    /// The view is not contiguous, so it covers no one byte range.
    NotContiguous,
    /// Its allocation is charged to no store (a mapped host region).
    Uncharged,
    /// The view reaches past what its allocation is charged for.
    BeyondCharge,
}

impl fmt::Display for ViewChargeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::NotContiguous => "store charges require a contiguous tensor view",
            Self::Uncharged => "tensor has no owned allocation charge",
            Self::BeyondCharge => "tensor view exceeds its allocation charge",
        })
    }
}
impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SlabLayout(reason) => write!(f, "slab layout: {reason}"),
            Self::UnsupportedRepresentation {
                backend,
                representation,
            } => write!(
                f,
                "{backend:?} does not support tensor representation `{representation}`"
            ),
            Self::HostByteLength { expected, actual } => write!(
                f,
                "host data has {actual} bytes; tensor requires {expected}"
            ),
            Self::HostIo(reason) => write!(f, "host input: {reason}"),
            Self::HostRegion(reason) => write!(f, "host mapping: {reason}"),
            Self::SliceOutOfBounds { extent, start, end } => {
                write!(f, "slice {start}..{end} is outside leading extent {extent}")
            }
            Self::UnalignedPacketSlice { group, start, end } => write!(
                f,
                "packed slice {start}..{end} is not aligned to packet group {group}"
            ),
            Self::UnalignedRowSlice {
                representation,
                start,
                end,
            } => write!(
                f,
                "`{representation}` slice {start}..{end} does not select whole rows or row tiles"
            ),
            Self::RowLayoutReshape {
                representation,
                extents,
            } => write!(
                f,
                "reshape to {extents:?} changes the row geometry of `{representation}` storage"
            ),
            Self::ReshapeStorage {
                current_bytes,
                requested_bytes,
            } => write!(
                f,
                "reshape requires {requested_bytes} bytes but the tensor view contains {current_bytes}"
            ),
            Self::ReshapeLayout { extents, strides } => write!(
                f,
                "reshape requires contiguous storage, but the view of extents {extents:?} has strides {strides:?}"
            ),
            Self::Uncommitted { rows, committed } => write!(
                f,
                "access to {rows} leading rows of a reserved tensor with {committed} committed rows"
            ),
            Self::ViewCharge(error) => write!(f, "{error}"),
            Self::Region(error) => write!(f, "{error}"),
            Self::Execution(error) => write!(f, "{error}"),
        }
    }
}
impl std::error::Error for TensorError {}
impl From<ExecutionError> for TensorError {
    fn from(value: ExecutionError) -> Self {
        Self::Execution(value)
    }
}

pub mod device {
    use crate::backends::OpenedKind;
    use crate::devices::{LimitError, MemoryUsage};
    use crate::driver::Allocation;
    use seismic_compiler::errors::ExecutionError;
    use seismic_topology::{AccessPathId, DeviceTopology, Executor, StorageKind, StoreId};
    use std::sync::Arc;

    /// One opened executor: its immutable description, execution resources
    /// and accounting scopes.
    pub struct DeviceInner {
        pub(crate) info: Executor,
        pub(crate) topology: Arc<DeviceTopology>,
        pub(crate) memory: Arc<crate::memory::ExecutorMemory>,
        pub(crate) capabilities: std::sync::OnceLock<Vec<String>>,
        pub(crate) kind: OpenedKind,
        /// Forms this device's native programs.
        pub(crate) programs: crate::backends::DevicePrograms,
        /// The active submission trace, if any (`native::trace`).
        pub(crate) trace: std::sync::Mutex<Option<Arc<crate::native::trace::TraceSink>>>,
        /// The artifact store this device was opened with; its program
        /// former keeps artifacts there. A device is shared only with callers
        /// asking for the same store.
        pub(crate) artifacts: Option<Arc<dyn crate::artifacts::ArtifactStore>>,
        /// Native submission order and CUDA graph replays.
        pub(crate) native: crate::native::NativeQueue,
        /// Kernels tuning chose and formed, handed to the serving
        /// preparation of the same specialization so nothing is formed twice
        /// (`NativePrepared::hand_off`). Taken on first use.
        pub(crate) chosen: std::sync::Mutex<
            std::collections::HashMap<
                crate::native::NativeHandoffKey,
                crate::native::NativeHandoff,
            >,
        >,
    }
    impl DeviceInner {
        pub(crate) fn active_trace(&self) -> Option<Arc<crate::native::trace::TraceSink>> {
            self.trace
                .lock()
                .expect("trace slot lock is never poisoned")
                .clone()
        }
        pub fn info(&self) -> &Executor {
            &self.info
        }
        /// Graphs recorded for this device's submissions so far: a
        /// submission during which the count grew recorded one on the host,
        /// so its duration is not its launches' alone.
        pub fn recorded_graphs(&self) -> u64 {
            self.native.recorded_graphs()
        }
        /// The topology revision this device was opened in; its identifiers
        /// are the ones this device's charges and limits name.
        pub fn topology(&self) -> &Arc<DeviceTopology> {
            &self.topology
        }
        /// How many native programs this device has formed.
        pub fn programs_formed(&self) -> usize {
            self.programs.formed()
        }
        pub fn capabilities(&self) -> &[String] {
            self.capabilities
                .get_or_init(|| self.kind.capabilities())
                .as_slice()
        }
        pub(crate) fn allocate(
            self: &Arc<Self>,
            bytes: u64,
            alignment: u64,
        ) -> Result<Arc<Allocation>, ExecutionError> {
            self.kind.allocate(bytes, alignment)
        }
        /// Storage the host fills before each submission; see
        /// `OpenedKind::allocate_upload`.
        pub(crate) fn allocate_upload(
            self: &Arc<Self>,
            bytes: u64,
            alignment: u64,
        ) -> Result<Arc<Allocation>, ExecutionError> {
            self.kind.allocate_upload(bytes, alignment)
        }
        pub(crate) fn map_read_only_host_region(
            self: &Arc<Self>,
            region: super::HostRegion,
            alignment: u64,
        ) -> Result<Arc<Allocation>, super::TensorError> {
            self.kind
                .map_read_only_host_region(region, alignment)
                .map_err(super::TensorError::Execution)
        }
        /// Zero-filled storage of a reserved tensor; see
        /// `OpenedKind::allocate_reserved_tensor`.
        pub(crate) fn allocate_reserved_tensor(
            self: &Arc<Self>,
            committed: u64,
            reserved: u64,
            alignment: u64,
        ) -> Result<Arc<Allocation>, ExecutionError> {
            self.kind
                .allocate_reserved_tensor(committed, reserved, alignment)
        }
        pub(crate) fn reserves_address(&self, allocation: &Arc<Allocation>) -> bool {
            self.kind.reserves_address(allocation)
        }
        pub(crate) fn allocation_address(&self, allocation: &Arc<Allocation>) -> u64 {
            self.kind.allocation_address(allocation)
        }
        /// A reserved tensor's storage resized in place; see
        /// `OpenedKind::recommit_in_place`.
        pub(crate) fn recommit_in_place(
            self: &Arc<Self>,
            allocation: &Arc<Allocation>,
            committed: u64,
            alignment: u64,
            exclusive_view: bool,
        ) -> Result<Option<Arc<Allocation>>, ExecutionError> {
            self.kind
                .recommit_in_place(allocation, committed, alignment, exclusive_view)
        }
        /// Stable key of this device's model and configuration for native
        /// tuning records: its name and the backend facts that shape
        /// performance.
        /// What decides how a configuration measures on this device: the
        /// device, its backend's identity, and the bytes a cold pass of its
        /// timing reads where it times cold passes
        /// (`native::point::cold_bytes`).
        pub fn tuning_identity(&self) -> String {
            let identity = format!("{};{}", self.info.name(), self.kind.tuning_identity());
            match crate::native::point::cold_bytes(self) {
                Some(cold) => format!("{identity};cold passes {cold} bytes"),
                None => identity,
            }
        }
        /// The device's last-level cache, where its backend reports it.
        pub(crate) fn last_level_cache_bytes(&self) -> Option<u64> {
            self.kind.last_level_cache_bytes()
        }
        /// Lend the device `workers` for the programs it forms off the
        /// calling thread, or take them back (none). The device starts no
        /// thread of its own: without workers it forms in turn on the
        /// thread that asks.
        pub fn staff(&self, workers: Option<Arc<seismic_native_target::workers::Workers>>) {
            self.programs.staff(workers);
        }
        /// Whether this device forms Metal tensor operations: what its
        /// executor reports ([`seismic_topology::Executor::forms_tensor_operations`]),
        /// false on every other backend.
        pub fn forms_tensor_operations(&self) -> bool {
            self.info().forms_tensor_operations()
        }
        pub(crate) fn supports_representation(
            &self,
            representation: seismic_lang::ids::RepresentationId,
        ) -> bool {
            self.kind.supports_representation(representation)
        }
        /// The bytes this device charges to each store it reaches: its
        /// paths' claims, and its staging on host RAM.
        pub fn store_charges(&self) -> Vec<(StoreId, u64)> {
            let mut charges = crate::memory::store_charges(&self.topology, self.path_charges());
            *charges.entry(self.topology.host_ram().id()).or_default() +=
                self.memory.staging.scope_charge();
            charges.into_iter().collect()
        }
        /// The claim this device holds along each of its access paths.
        pub fn path_charges(&self) -> Vec<(AccessPathId, u64)> {
            self.memory
                .paths
                .iter()
                .map(|path| (path.id, path.scope.charged()))
                .collect()
        }
        /// CPU allocations are ordinary process memory; Metal resources enter
        /// the task's physical-footprint ledger. CUDA and Vulkan driver allocations
        /// are accounted separately from the process's anonymous footprint.
        pub fn allocations_in_process_footprint(&self) -> bool {
            match &self.kind {
                OpenedKind::Cpu(_) => true,
                #[cfg(target_os = "macos")]
                OpenedKind::Metal(_) => true,
                OpenedKind::Cuda(_) => false,
                #[cfg(not(target_os = "macos"))]
                OpenedKind::Vulkan(_) => false,
            }
        }
        /// Seismic-owned charges and limits of `kind` storage on this
        /// device: its allocations' charges, what its backend holds for it
        /// beyond them, and the charges to the stores its placement reaches.
        pub fn storage_usage(&self, kind: StorageKind) -> MemoryUsage {
            self.kind.usage(kind)
        }
        /// What `kind` storage can still take on this device: what its
        /// placement's paths, their grants and stores, and the device's
        /// limit leave to both its allocations' charges and what the backend
        /// holds natively. An allocation needing more is refused as a
        /// capacity shortfall.
        pub fn storage_available(&self, kind: StorageKind) -> u64 {
            let domain = match kind {
                StorageKind::Device => &self.memory.device,
                StorageKind::Upload => &self.memory.upload,
            };
            domain.available().min(domain.physical().available())
        }
        /// Holds the charges to `store`, from every executor of the process
        /// that reaches it, to `limit` on top of its capacity.
        pub fn set_store_limit(
            &self,
            store: StoreId,
            limit: Option<u64>,
        ) -> Result<(), LimitError> {
            self.topology
                .store(store)
                .ok_or(LimitError::StaleStore(store))?;
            crate::memory::store_ledger(&self.topology, store).set_limit(limit);
            Ok(())
        }
        /// Holds this device's claim along its access path `path` to `limit`
        /// on top of the path's grant.
        pub fn set_path_limit(
            &self,
            path: AccessPathId,
            limit: Option<u64>,
        ) -> Result<(), LimitError> {
            let path = self
                .memory
                .path(path)
                .ok_or(LimitError::ForeignPath(path))?;
            path.scope.set_limit(limit);
            Ok(())
        }
        /// Bounds `kind` storage allocated through this device. Charges
        /// already made count against the limit; they are never charged
        /// twice.
        pub fn set_storage_limit(&self, kind: StorageKind, limit: Option<u64>) {
            self.kind.set_limit(kind, limit)
        }
        /// Start device storage's [`MemoryUsage::high_water`] again from its
        /// current charge.
        pub fn restart_memory_high_water(&self) {
            self.kind.restart_memory_high_water()
        }
    }
}

pub mod tensor {
    use super::{device::DeviceInner, HostRegion, RegionError, TensorError, ViewChargeError};
    use crate::driver::{write_zeros, Allocation};
    use crate::layout;
    use seismic_compiler::errors::ExecutionError;
    use seismic_compiler::prepared::TensorDescriptor;
    use seismic_lang::ids::RepresentationId;
    use seismic_lang::registry::representation_info;
    use std::io::Read;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Weak,
    };

    /// A non-owning observation of one physical tensor allocation. It keeps
    /// no storage alive and reads the same charge as the device ledger.
    pub struct TensorStorageObserver {
        allocation: Weak<Allocation>,
        identity: u64,
        topology: Arc<seismic_topology::DeviceTopology>,
        /// The accounting of the executor the allocation was made through.
        memory: Arc<crate::memory::ExecutorMemory>,
    }

    impl TensorStorageObserver {
        /// The bytes the allocation charges to each store, while it lives
        /// and is charged.
        pub fn store_charges(&self) -> Option<Vec<(seismic_topology::StoreId, u64)>> {
            let paths = self.path_charges()?;
            Some(
                crate::memory::store_charges(&self.topology, paths)
                    .into_iter()
                    .collect(),
            )
        }
        /// The bytes the allocation charges along each access path, while
        /// it lives and is charged.
        pub fn path_charges(&self) -> Option<Vec<(seismic_topology::AccessPathId, u64)>> {
            self.allocation.upgrade()?.path_charges(&self.memory)
        }
        pub fn identity(&self) -> u64 {
            self.identity
        }

        pub fn charged_bytes(&self) -> Option<u64> {
            self.allocation
                .upgrade()
                .and_then(|allocation| allocation.charged_bytes())
        }
    }

    /// One charged device view of an immutable host window. Tensor views
    /// share its Metal buffer and retain it through submitted device work.
    pub struct MappedRegionInner {
        device: Arc<DeviceInner>,
        allocation: Arc<Allocation>,
    }

    impl MappedRegionInner {
        pub fn new(device: &Arc<DeviceInner>, region: HostRegion) -> Result<Self, TensorError> {
            let allocation = device.map_read_only_host_region(region, 256)?;
            Ok(Self {
                device: device.clone(),
                allocation,
            })
        }

        /// Whether a canonical-layout tensor of `representation` and
        /// `extents` over a region's bytes from `byte_offset` keeps the
        /// layout's alignment, which a kernel reading it assumes: a region
        /// starts at a host page, so its bytes from an offset are aligned as
        /// the offset is (to the page).
        pub fn keeps_alignment(
            representation: RepresentationId,
            extents: &[u64],
            byte_offset: u64,
        ) -> Result<bool, TensorError> {
            let layout = layout::canonical(representation, extents)?;
            Ok(byte_offset.is_multiple_of(layout.alignment))
        }

        /// A canonical-layout tensor of `representation` and `extents` over
        /// the region's bytes from `byte_offset`, which must keep the
        /// layout's alignment ([`Self::keeps_alignment`]).
        pub fn tensor(
            &self,
            representation: RepresentationId,
            extents: &[u64],
            byte_offset: u64,
        ) -> Result<TensorInner, TensorError> {
            if !self.device.supports_representation(representation) {
                return Err(TensorError::UnsupportedRepresentation {
                    backend: self.device.info().backend(),
                    representation: representation_info(representation).name,
                });
            }
            let layout = layout::canonical(representation, extents)?;
            let end = byte_offset
                .checked_add(layout.byte_len)
                .ok_or_else(|| TensorError::HostRegion("view offset overflows".into()))?;
            if end > self.allocation.bytes() {
                return Err(TensorError::HostRegion(
                    "tensor view exceeds mapped region".into(),
                ));
            }
            if !Self::keeps_alignment(representation, extents, byte_offset)? {
                return Err(TensorError::Region(RegionError::Unaligned {
                    offset: byte_offset,
                    alignment: layout.alignment,
                }));
            }
            Ok(TensorInner::new_view(
                self.device.clone(),
                self.allocation.clone(),
                byte_offset,
                layout.byte_len,
                representation,
                extents.to_vec(),
                layout.strides,
            ))
        }
    }

    pub struct TensorInner {
        device: Arc<DeviceInner>,
        allocation: Arc<Allocation>,
        byte_offset: u64,
        byte_len: u64,
        representation: RepresentationId,
        extents: Vec<u64>,
        strides: Vec<u64>,
        /// A reserved tensor's logical bytes reach past its allocation: only
        /// its leading committed rows are backed, and no access, host or
        /// device, may touch the rest. Its owner guarantees the device side.
        reserved: bool,
        slabs: Option<SlabView>,
        alias_identity: Option<u64>,
        slab_lease: Option<Arc<AtomicUsize>>,
    }

    impl Drop for TensorInner {
        fn drop(&mut self) {
            if let Some(lease) = &self.slab_lease {
                lease.fetch_sub(1, Ordering::AcqRel);
            }
        }
    }

    struct SlabView {
        rows_per_slab: u64,
        regions: Vec<Option<Arc<TensorInner>>>,
    }

    /// A tensor's elements on the host ([`TensorInner::image`]).
    pub(crate) enum TensorImage {
        Whole(Vec<u8>),
        Committed(Vec<u8>),
        Slabs(Vec<Option<Vec<u8>>>),
    }

    /// Bytes of one leading-axis row of a canonical layout.
    fn leading_row_bytes(extents: &[u64], byte_len: u64) -> Result<u64, TensorError> {
        match extents.first() {
            Some(&rows) if rows != 0 && byte_len % rows == 0 => Ok(byte_len / rows),
            _ => Err(TensorError::Uncommitted {
                rows: extents.first().copied().unwrap_or(0),
                committed: 0,
            }),
        }
    }

    impl TensorInner {
        /// A logical tensor whose native binding is a slab address table.
        /// Each present region view retains its slab allocation; a launch
        /// snapshots this list and tracks every allocation it can address.
        pub fn slabbed(
            table: &Arc<TensorInner>,
            representation: RepresentationId,
            rows_per_slab: u64,
            logical_rows: u64,
            alias_identity: u64,
            regions: Vec<Option<Arc<TensorInner>>>,
            lease: Arc<AtomicUsize>,
        ) -> Result<Self, TensorError> {
            if regions.is_empty() || rows_per_slab == 0 {
                return Err(TensorError::SlabLayout("empty slab tensor".into()));
            }
            if table.representation
                != seismic_lang::registry::dense(seismic_lang::types::DType::U32)
                || table.extents != [regions.len() as u64, 2]
                || table.reserved
            {
                return Err(TensorError::SlabLayout(
                    "slab address table has the wrong representation or shape".into(),
                ));
            }
            let example = regions
                .iter()
                .flatten()
                .next()
                .ok_or_else(|| TensorError::SlabLayout("no backed slab".into()))?;
            if example.representation != representation
                || example.extents.first().copied() != Some(rows_per_slab)
                || regions.iter().flatten().any(|region| {
                    !Arc::ptr_eq(&region.device, &table.device)
                        || region.representation != representation
                        || region.extents != example.extents
                        || !region.allocation.shares_access_with(&table.allocation)
                })
            {
                return Err(TensorError::SlabLayout(
                    "slab region shapes or devices differ".into(),
                ));
            }
            let capacity_rows = u64::try_from(regions.len())
                .ok()
                .and_then(|count| count.checked_mul(rows_per_slab))
                .ok_or_else(|| TensorError::SlabLayout("logical slab extent overflows".into()))?;
            if logical_rows == 0
                || logical_rows > capacity_rows
                || logical_rows <= capacity_rows - rows_per_slab
            {
                return Err(TensorError::SlabLayout(
                    "logical extent does not match slab capacity".into(),
                ));
            }
            let mut extents = example.extents.clone();
            extents[0] = logical_rows;
            let layout = layout::canonical(representation, &extents)?;
            lease.fetch_add(1, Ordering::AcqRel);
            Ok(Self {
                device: table.device.clone(),
                allocation: table.allocation.clone(),
                byte_offset: table.byte_offset,
                byte_len: layout.byte_len,
                representation,
                extents,
                strides: layout.strides,
                reserved: false,
                slabs: Some(SlabView {
                    rows_per_slab,
                    regions,
                }),
                alias_identity: Some(alias_identity),
                slab_lease: Some(lease),
            })
        }

        pub fn new_slab_alias() -> u64 {
            crate::driver::fresh_allocation_identity()
        }

        pub(crate) fn bound_allocations(&self) -> impl Iterator<Item = Arc<Allocation>> + '_ {
            // Every slab region shares the address table's hazard group.
            // One fence orders host access and release for all backed slots.
            std::iter::once(self.allocation.clone())
        }

        pub(crate) fn is_slabbed(&self) -> bool {
            self.slabs.is_some()
        }

        /// The device bytes the tensor's elements occupy: a slab view's
        /// backed slabs, a reserved tensor's committed rows, otherwise its
        /// byte length.
        pub(crate) fn occupied_bytes(&self) -> u64 {
            match &self.slabs {
                Some(slabs) => slabs.regions.iter().flatten().map(|r| r.byte_len).sum(),
                None if self.reserved => self.committed_rows() * (self.byte_len / self.extents[0]),
                None => self.byte_len,
            }
        }

        /// The bytes a device copy of the tensor is written from
        /// ([`TensorInner::copy_of`]), in canonical layout: all of them, a
        /// reserved tensor's committed rows (only they are backed), or each
        /// backed slab's region of a slab view.
        pub(crate) fn image(&self) -> Result<TensorImage, TensorError> {
            if let Some(slabs) = &self.slabs {
                return Ok(TensorImage::Slabs(
                    slabs
                        .regions
                        .iter()
                        .map(|region| region.as_ref().map(|r| r.read_to_host()).transpose())
                        .collect::<Result<_, _>>()?,
                ));
            }
            if self.reserved {
                return Ok(TensorImage::Committed(
                    self.slice_leading(0, self.committed_rows())?
                        .read_to_host()?,
                ));
            }
            Ok(TensorImage::Whole(self.read_to_host()?))
        }

        /// A tensor of this one's shape over storage of its own holding
        /// `image` (this tensor's [`TensorInner::image`]): a call reads the
        /// same values from it and shares no byte of storage with a call of
        /// this one. A slab view's copy has its own slabs and address table.
        pub(crate) fn copy_of(&self, image: &TensorImage) -> Result<Self, TensorError> {
            let device = &self.device;
            match (image, &self.slabs) {
                (TensorImage::Whole(bytes), None) if !self.reserved => {
                    Self::from_host(device, self.representation, &self.extents, bytes)
                }
                (TensorImage::Committed(bytes), None) if self.reserved => {
                    let committed = self.committed_rows();
                    let copy =
                        Self::reserved(device, self.representation, &self.extents, committed)?;
                    copy.slice_leading(0, committed)?.write_from_host(bytes)?;
                    Ok(copy)
                }
                (TensorImage::Slabs(images), Some(slabs))
                    if images.len() == slabs.regions.len() =>
                {
                    let dense_u32 = seismic_lang::registry::dense(seismic_lang::types::DType::U32);
                    let table = Arc::new(Self::zeros(
                        device,
                        dense_u32,
                        &[slabs.regions.len() as u64, 2],
                    )?);
                    let regions = slabs
                        .regions
                        .iter()
                        .zip(images)
                        .map(|(region, bytes)| match (region, bytes) {
                            (Some(region), Some(bytes)) => {
                                let copy = Self::from_host(
                                    device,
                                    region.representation,
                                    &region.extents,
                                    bytes,
                                )?;
                                copy.share_slab_access_with(&table);
                                Ok(Some(Arc::new(copy)))
                            }
                            _ => Ok(None),
                        })
                        .collect::<Result<Vec<_>, TensorError>>()?;
                    // An unbacked slot names the first backed slab, as a
                    // slab store's table does.
                    let stand_in = regions
                        .iter()
                        .flatten()
                        .next()
                        .map(|region| region.device_address())
                        .unwrap_or(0);
                    let words = regions
                        .iter()
                        .flat_map(|region| {
                            region
                                .as_ref()
                                .map_or(stand_in, |region| region.device_address())
                                .to_le_bytes()
                        })
                        .collect::<Vec<_>>();
                    table.write_from_host(&words)?;
                    Self::slabbed(
                        &table,
                        self.representation,
                        slabs.rows_per_slab,
                        self.extents[0],
                        Self::new_slab_alias(),
                        regions,
                        Arc::new(AtomicUsize::new(0)),
                    )
                }
                _ => Err(TensorError::SlabLayout(
                    "a tensor image does not match the tensor".into(),
                )),
            }
        }

        pub fn canonical_layout(
            representation: RepresentationId,
            extents: &[u64],
        ) -> Result<(u64, u64), TensorError> {
            let layout = layout::canonical(representation, extents)?;
            Ok((layout.byte_len, layout.alignment))
        }

        /// A canonical-layout tensor of `representation` and `extents` over
        /// this view's bytes from `offset`: its allocation remains shared
        /// with this view and every sibling region, so each view carries the
        /// allocation's lifetime into a launch. This view must be one
        /// contiguous byte range of a backed allocation (not reserved, not
        /// slabbed), and the region must lie within it, starting at a
        /// multiple of its layout's alignment in the allocation.
        pub fn region(
            &self,
            representation: RepresentationId,
            extents: &[u64],
            offset: u64,
        ) -> Result<Self, TensorError> {
            if !self.device.supports_representation(representation) {
                return Err(TensorError::UnsupportedRepresentation {
                    backend: self.device.info().backend(),
                    representation: representation_info(representation).name,
                });
            }
            if self.reserved
                || self.slabs.is_some()
                || layout::canonical(self.representation, &self.extents)?.strides != self.strides
            {
                return Err(TensorError::Region(RegionError::NotContiguous));
            }
            let layout = layout::canonical(representation, extents)?;
            let outside = TensorError::Region(RegionError::Outside {
                offset,
                bytes: layout.byte_len,
                view_bytes: self.byte_len,
            });
            let end = offset.checked_add(layout.byte_len).ok_or(outside.clone())?;
            if end > self.byte_len {
                return Err(outside);
            }
            let start = self.byte_offset + offset;
            if start % layout.alignment != 0 {
                return Err(TensorError::Region(RegionError::Unaligned {
                    offset: start,
                    alignment: layout.alignment,
                }));
            }
            Ok(Self::new_view(
                self.device.clone(),
                self.allocation.clone(),
                start,
                layout.byte_len,
                representation,
                extents.to_vec(),
                layout.strides,
            ))
        }

        pub fn device_address(&self) -> u64 {
            self.device.allocation_address(&self.allocation) + self.byte_offset
        }

        /// Join `other`'s hazard domain: a new slab its address table's, or a
        /// new table that of the allocation its slabs view.
        pub fn share_slab_access_with(&self, other: &Self) {
            self.allocation.share_access_with(&other.allocation);
        }

        /// A zero-filled tensor over `extents` whose leading rows
        /// `[0, committed)` are physically backed. The remaining rows are an
        /// address reservation: binding the tensor is legal only for device
        /// work that stays within the committed rows, and host access to them
        /// is refused.
        pub fn reserved(
            device: &Arc<DeviceInner>,
            representation: RepresentationId,
            extents: &[u64],
            committed: u64,
        ) -> Result<Self, TensorError> {
            if !device.supports_representation(representation) {
                return Err(TensorError::UnsupportedRepresentation {
                    backend: device.info().backend(),
                    representation: representation_info(representation).name,
                });
            }
            let layout = layout::canonical(representation, extents)?;
            let row = leading_row_bytes(extents, layout.byte_len)?;
            if committed == 0 || committed > extents[0] {
                return Err(TensorError::Uncommitted {
                    rows: committed,
                    committed: extents[0],
                });
            }
            let bytes = row * committed;
            let allocation =
                device.allocate_reserved_tensor(bytes, layout.byte_len, layout.alignment)?;
            Ok(Self {
                device: device.clone(),
                allocation,
                byte_offset: 0,
                byte_len: layout.byte_len,
                representation,
                extents: extents.to_vec(),
                strides: layout.strides,
                reserved: bytes < layout.byte_len,
                slabs: None,
                alias_identity: None,
                slab_lease: None,
            })
        }

        /// Leading rows physically backed: all of them unless reserved.
        pub fn committed_rows(&self) -> u64 {
            match (
                self.reserved,
                leading_row_bytes(&self.extents, self.byte_len),
            ) {
                (true, Ok(row)) => (self.allocation.bytes().saturating_sub(self.byte_offset) / row)
                    .min(self.extents[0]),
                _ => self.extents.first().copied().unwrap_or(1),
            }
        }

        /// The same logical tensor with `committed` leading rows backed by a
        /// new allocation. The first `min(committed, current)` rows keep their
        /// contents, as seen after every submitted device write of this
        /// tensor completes; new rows are zero. Where the backend reserved the
        /// address range (CUDA virtual memory management) the backing is
        /// resized in place after every device use of this tensor completes,
        /// and the address is unchanged; elsewhere the rows are copied into
        /// new storage. Device work bound to this tensor must not be
        /// submitted after the new one is used.
        pub fn recommitted(
            &self,
            committed: u64,
            exclusive_view: bool,
        ) -> Result<Self, TensorError> {
            let layout = layout::canonical(self.representation, &self.extents)?;
            let row = leading_row_bytes(&self.extents, layout.byte_len)?;
            if self.byte_offset != 0
                || self.byte_len != layout.byte_len
                || committed == 0
                || committed > self.extents[0]
            {
                return Err(TensorError::Uncommitted {
                    rows: committed,
                    committed: self.committed_rows(),
                });
            }
            let bytes = row * committed;
            // Caller-held views keep the old allocation alive. A CUDA VMM
            // reservation cannot be shrunk beneath such a view's backed
            // range, so the backend will return None and use a new backing.
            let allocation = match self.device.recommit_in_place(
                &self.allocation,
                bytes,
                layout.alignment,
                exclusive_view,
            )? {
                Some(allocation) => allocation,
                None => self.reallocated(bytes, layout.alignment)?,
            };
            Ok(Self {
                device: self.device.clone(),
                allocation,
                byte_offset: 0,
                byte_len: layout.byte_len,
                representation: self.representation,
                extents: self.extents.clone(),
                strides: layout.strides,
                reserved: bytes < layout.byte_len,
                slabs: None,
                alias_identity: None,
                slab_lease: None,
            })
        }

        /// A new zero-filled allocation of `bytes` holding this tensor's
        /// leading bytes, copied after every submitted device write of it
        /// completes: recommitting where the backend keeps no reservation.
        fn reallocated(&self, bytes: u64, alignment: u64) -> Result<Arc<Allocation>, TensorError> {
            let allocation =
                self.device
                    .allocate_reserved_tensor(bytes, self.byte_len, alignment)?;
            let kept = bytes.min(self.allocation.bytes());
            let access = self.allocation.acquire(false);
            const CHUNK: u64 = 1 << 24;
            let mut scratch = vec![0u8; usize::try_from(kept.min(CHUNK)).unwrap_or(1 << 24)];
            let mut offset = 0u64;
            while offset < kept {
                let length = usize::try_from((kept - offset).min(CHUNK)).unwrap_or(1 << 24);
                access.read(&self.allocation, offset, &mut scratch[..length])?;
                allocation.storage().write(offset, &scratch[..length])?;
                offset += length as u64;
            }
            Ok(allocation)
        }

        /// Whether [`Self::recommitted`] keeps this tensor's address: the
        /// backend reserved its address range and resizes the backing in place.
        pub fn resizes_in_place(&self) -> bool {
            self.byte_offset == 0 && self.device.reserves_address(&self.allocation)
        }

        /// This exact tensor can replace its backing in place now. Any
        /// external view or unpublished predecessor requires new storage.
        pub fn can_recommit_in_place(&self) -> bool {
            self.resizes_in_place()
                && Arc::strong_count(&self.allocation) == 1
                && !self.allocation.has_live_predecessor()
        }

        /// The same logical tensor with `committed` leading rows backed by a
        /// new allocation that holds rows moved from this one: each
        /// `(from, to, rows)` copies rows `[from, from + rows)`, as seen after
        /// every submitted device write of this tensor completes, to rows
        /// `[to, to + rows)`. Every other row is zero. The address changes on
        /// every backend; device work bound to this tensor must not be
        /// submitted after the new one is used.
        pub fn relocated(
            &self,
            committed: u64,
            moves: &[(u64, u64, u64)],
        ) -> Result<Self, TensorError> {
            let layout = layout::canonical(self.representation, &self.extents)?;
            let row = leading_row_bytes(&self.extents, layout.byte_len)?;
            let current = self.committed_rows();
            let within = |start: u64, rows: u64, limit: u64| {
                start.checked_add(rows).is_some_and(|end| end <= limit)
            };
            if self.byte_offset != 0
                || self.byte_len != layout.byte_len
                || committed == 0
                || committed > self.extents[0]
                || moves.iter().any(|&(from, to, rows)| {
                    !within(from, rows, current) || !within(to, rows, committed)
                })
            {
                return Err(TensorError::Uncommitted {
                    rows: committed,
                    committed: current,
                });
            }
            let bytes = row * committed;
            let allocation =
                self.device
                    .allocate_reserved_tensor(bytes, layout.byte_len, layout.alignment)?;
            let access = self.allocation.acquire(false);
            const CHUNK: u64 = 1 << 24;
            let mut scratch = Vec::new();
            for &(from, to, rows) in moves {
                let (mut offset, end) = (from * row, (from + rows) * row);
                while offset < end {
                    let length = usize::try_from((end - offset).min(CHUNK)).unwrap_or(1 << 24);
                    scratch.resize(length, 0);
                    access.read(&self.allocation, offset, &mut scratch)?;
                    allocation
                        .storage()
                        .write(offset - from * row + to * row, &scratch)?;
                    offset += length as u64;
                }
            }
            Ok(Self {
                device: self.device.clone(),
                allocation,
                byte_offset: 0,
                byte_len: layout.byte_len,
                representation: self.representation,
                extents: self.extents.clone(),
                strides: layout.strides,
                reserved: bytes < layout.byte_len,
                slabs: None,
                alias_identity: None,
                slab_lease: None,
            })
        }

        /// Refuse host access reaching past the committed rows.
        fn backed(&self) -> Result<(), TensorError> {
            if self.reserved {
                return Err(TensorError::Uncommitted {
                    rows: self.extents.first().copied().unwrap_or(0),
                    committed: self.committed_rows(),
                });
            }
            Ok(())
        }

        /// Exact canonical storage charge for a representation and logical
        /// shape, without allocating a device tensor.
        pub fn canonical_byte_len(
            representation: RepresentationId,
            extents: &[u64],
        ) -> Result<u64, TensorError> {
            Ok(layout::canonical(representation, extents)?.byte_len)
        }

        /// A zero-filled tensor in storage the host writes and reads in
        /// place while device work that uses it is queued or running
        /// (`OpenedKind::allocate_upload`: host-visible on every backend,
        /// mapped pinned host memory on CUDA).
        pub fn shared(
            device: &Arc<DeviceInner>,
            representation: RepresentationId,
            extents: &[u64],
        ) -> Result<Self, TensorError> {
            if !device.supports_representation(representation) {
                return Err(TensorError::UnsupportedRepresentation {
                    backend: device.info().backend(),
                    representation: representation_info(representation).name,
                });
            }
            let layout = layout::canonical(representation, extents)?;
            let allocation = device.allocate_upload(layout.byte_len, layout.alignment)?;
            write_zeros(allocation.storage(), layout.byte_len)?;
            Ok(Self::new_view(
                device.clone(),
                allocation,
                0,
                layout.byte_len,
                representation,
                extents.to_vec(),
                layout.strides,
            ))
        }

        /// Write `bytes` at `offset` bytes into this view without waiting
        /// for device work that uses its storage.
        ///
        /// # Safety
        /// The tensor was made by [`Self::shared`] (or is a view of one),
        /// and no device work in flight reads or writes these bytes.
        pub unsafe fn write_shared(&self, offset: u64, bytes: &[u8]) -> Result<(), TensorError> {
            offset
                .checked_add(bytes.len() as u64)
                .filter(|end| *end <= self.byte_len)
                .ok_or(TensorError::HostByteLength {
                    expected: self.byte_len,
                    actual: offset.saturating_add(bytes.len() as u64),
                })?;
            self.allocation
                .storage()
                .write(self.byte_offset + offset, bytes)
                .map_err(TensorError::Execution)
        }

        /// Read `into.len()` bytes at `offset` bytes into this view without
        /// waiting for device work that uses its storage.
        ///
        /// # Safety
        /// The tensor was made by [`Self::shared`] (or is a view of one),
        /// and no device work in flight writes these bytes.
        pub unsafe fn read_shared(&self, offset: u64, into: &mut [u8]) -> Result<(), TensorError> {
            offset
                .checked_add(into.len() as u64)
                .filter(|end| *end <= self.byte_len)
                .ok_or(TensorError::HostByteLength {
                    expected: self.byte_len,
                    actual: offset.saturating_add(into.len() as u64),
                })?;
            self.allocation
                .storage()
                .read(self.byte_offset + offset, into)
                .map_err(TensorError::Execution)
        }

        pub fn zeros(
            device: &Arc<DeviceInner>,
            representation: RepresentationId,
            extents: &[u64],
        ) -> Result<Self, TensorError> {
            // SAFETY: every byte is initialized before this tensor escapes.
            let tensor = unsafe { Self::uninitialized(device, representation, extents)? };
            write_zeros(tensor.allocation.storage(), tensor.byte_len)?;
            Ok(tensor)
        }

        /// Allocate a canonical tensor without writing its storage. The
        /// caller must fully initialize all bytes, including representation
        /// padding, before any host read or device input use.
        pub unsafe fn uninitialized(
            device: &Arc<DeviceInner>,
            representation: RepresentationId,
            extents: &[u64],
        ) -> Result<Self, TensorError> {
            if !device.supports_representation(representation) {
                return Err(TensorError::UnsupportedRepresentation {
                    backend: device.info().backend(),
                    representation: representation_info(representation).name,
                });
            }
            let layout = layout::canonical(representation, extents)?;
            let allocation = device.allocate(layout.byte_len, layout.alignment)?;
            Ok(Self::new_view(
                device.clone(),
                allocation,
                0,
                layout.byte_len,
                representation,
                extents.to_vec(),
                layout.strides,
            ))
        }
        pub fn from_host(
            device: &Arc<DeviceInner>,
            representation: RepresentationId,
            extents: &[u64],
            bytes: &[u8],
        ) -> Result<Self, TensorError> {
            if !device.supports_representation(representation) {
                return Err(TensorError::UnsupportedRepresentation {
                    backend: device.info().backend(),
                    representation: representation_info(representation).name,
                });
            }
            let layout = layout::canonical(representation, extents)?;
            let actual = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
            if actual != layout.byte_len {
                return Err(TensorError::HostByteLength {
                    expected: layout.byte_len,
                    actual,
                });
            }
            let allocation = device.allocate(layout.byte_len, layout.alignment)?;
            allocation.storage().write(0, bytes)?;
            Ok(Self::new_view(
                device.clone(),
                allocation,
                0,
                layout.byte_len,
                representation,
                extents.to_vec(),
                layout.strides,
            ))
        }

        pub(crate) fn new_view(
            device: Arc<DeviceInner>,
            allocation: Arc<Allocation>,
            byte_offset: u64,
            byte_len: u64,
            representation: RepresentationId,
            extents: Vec<u64>,
            strides: Vec<u64>,
        ) -> Self {
            assert_eq!(
                extents.len(),
                strides.len(),
                "TensorInner view rank differs from stride count"
            );
            let end = byte_offset
                .checked_add(byte_len)
                .expect("TensorInner view byte range overflowed");
            assert!(
                end <= allocation.bytes(),
                "TensorInner view exceeds its physical allocation"
            );
            Self {
                device,
                allocation,
                byte_offset,
                byte_len,
                representation,
                extents,
                strides,
                reserved: false,
                slabs: None,
                alias_identity: None,
                slab_lease: None,
            }
        }
        pub fn read_to_host(&self) -> Result<Vec<u8>, ExecutionError> {
            if let Some(slabs) = &self.slabs {
                let length = usize::try_from(self.byte_len).map_err(|_| {
                    ExecutionError::AllocationFailed(
                        "slab tensor is too large for a host byte vector".into(),
                    )
                })?;
                let row_bytes = self.byte_len / self.extents[0];
                let slab_bytes =
                    usize::try_from(row_bytes * slabs.rows_per_slab).map_err(|_| {
                        ExecutionError::AllocationFailed("slab exceeds host byte domain".into())
                    })?;
                let mut bytes = vec![0u8; length];
                for (index, region) in slabs.regions.iter().enumerate() {
                    if let Some(region) = region {
                        let start = index * slab_bytes;
                        let keep = slab_bytes.min(length - start);
                        bytes[start..start + keep].copy_from_slice(&region.read_to_host()?[..keep]);
                    }
                }
                return Ok(bytes);
            }
            let access = self.allocation.acquire(false);
            self.read_with(&access)
        }
        pub(crate) fn read_with(
            &self,
            access: &crate::driver::AllocationPermit,
        ) -> Result<Vec<u8>, ExecutionError> {
            self.backed()
                .map_err(|error| ExecutionError::AllocationFailed(error.to_string()))?;
            let canonical = crate::layout::canonical(self.representation, &self.extents)?;
            let length = usize::try_from(canonical.byte_len).map_err(|_| {
                ExecutionError::AllocationFailed(
                    "tensor is too large for a host byte vector".to_owned(),
                )
            })?;
            let mut bytes = vec![0u8; length];
            crate::layout::transfer_ranges(
                self.representation,
                &self.extents,
                &self.strides,
                |offset, host| {
                    let offset = self.byte_offset.checked_add(offset).ok_or_else(|| {
                        ExecutionError::AllocationFailed(
                            "tensor host transfer address overflow".to_owned(),
                        )
                    })?;
                    access.read(&self.allocation, offset, &mut bytes[host])
                },
            )?;
            Ok(bytes)
        }
        pub fn write_from_host(&self, bytes: &[u8]) -> Result<(), TensorError> {
            if let Some(slabs) = &self.slabs {
                let actual = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
                if actual != self.byte_len {
                    return Err(TensorError::HostByteLength {
                        expected: self.byte_len,
                        actual,
                    });
                }
                if slabs.regions.iter().any(Option::is_none) {
                    return Err(TensorError::SlabLayout(
                        "host write reaches a free slab".into(),
                    ));
                }
                let slab_bytes =
                    usize::try_from(self.byte_len / self.extents[0] * slabs.rows_per_slab)
                        .map_err(|_| {
                            TensorError::SlabLayout("slab exceeds host byte domain".into())
                        })?;
                for (index, region) in slabs.regions.iter().enumerate() {
                    let start = index * slab_bytes;
                    let keep = slab_bytes.min(bytes.len() - start);
                    let region = region.as_ref().expect("checked backed slab");
                    if keep == slab_bytes {
                        region.write_from_host(&bytes[start..start + keep])?;
                    } else {
                        let rows = u64::try_from(keep)
                            .ok()
                            .and_then(|count| count.checked_div(self.byte_len / self.extents[0]))
                            .ok_or_else(|| {
                                TensorError::SlabLayout("partial slab row count overflows".into())
                            })?;
                        region
                            .slice_leading(0, rows)?
                            .write_from_host(&bytes[start..start + keep])?;
                    }
                }
                return Ok(());
            }
            self.backed()?;
            if self.allocation.storage().read_only() {
                return Err(TensorError::HostRegion("mapped tensor is read-only".into()));
            }
            let canonical = crate::layout::canonical(self.representation, &self.extents)?;
            let actual = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
            if actual != canonical.byte_len {
                return Err(TensorError::HostByteLength {
                    expected: canonical.byte_len,
                    actual,
                });
            }
            let _guard = self.allocation.acquire(true);
            crate::layout::transfer_ranges(
                self.representation,
                &self.extents,
                &self.strides,
                |offset, host| {
                    let offset = self.byte_offset.checked_add(offset).ok_or_else(|| {
                        ExecutionError::AllocationFailed(
                            "tensor host transfer address overflow".to_owned(),
                        )
                    })?;
                    self.allocation.storage().write(offset, &bytes[host])
                },
            )?;
            Ok(())
        }

        /// Fill the physical bytes of a canonical, contiguous tensor from a
        /// reader using bounded scratch. Intended for complete upload inputs;
        /// an error leaves the tensor only partly initialized.
        pub fn write_from_reader(&self, reader: &mut dyn Read) -> Result<(), TensorError> {
            if self.slabs.is_some() {
                return Err(TensorError::SlabLayout(
                    "reader upload requires one contiguous allocation".into(),
                ));
            }
            self.backed()?;
            if self.allocation.storage().read_only() {
                return Err(TensorError::HostRegion("mapped tensor is read-only".into()));
            }
            let canonical = crate::layout::canonical(self.representation, &self.extents)?;
            if self.strides != canonical.strides || self.byte_len != canonical.byte_len {
                return Err(TensorError::HostRegion(
                    "reader requires a canonical contiguous tensor".into(),
                ));
            }
            let _guard = self.allocation.acquire(true);
            // On the heap: a 1 MiB array would fill a whole Windows main-thread stack.
            let mut scratch = vec![0u8; 1024 * 1024];
            let mut written = 0u64;
            while written < self.byte_len {
                let count = usize::try_from((self.byte_len - written).min(scratch.len() as u64))
                    .expect("chunk fits usize");
                reader
                    .read_exact(&mut scratch[..count])
                    .map_err(|error| TensorError::HostIo(error.to_string()))?;
                let offset = self.byte_offset.checked_add(written).ok_or_else(|| {
                    TensorError::HostRegion("reader write offset overflows".into())
                })?;
                self.allocation.storage().write(offset, &scratch[..count])?;
                written += count as u64;
            }
            Ok(())
        }
        pub fn device(&self) -> &Arc<DeviceInner> {
            &self.device
        }
        pub fn representation(&self) -> RepresentationId {
            self.representation
        }
        pub fn extents(&self) -> &[u64] {
            &self.extents
        }
        pub fn strides(&self) -> &[u64] {
            &self.strides
        }
        pub fn byte_len(&self) -> u64 {
            self.byte_len
        }
        pub(crate) fn byte_offset(&self) -> u64 {
            self.byte_offset
        }
        pub(crate) fn allocation(&self) -> &Arc<Allocation> {
            &self.allocation
        }
        pub(crate) fn alias_identity(&self) -> u64 {
            self.alias_identity
                .unwrap_or_else(|| self.allocation.identity())
        }
        #[doc(hidden)]
        pub fn shares_allocation(&self, other: &Self) -> bool {
            Arc::ptr_eq(&self.allocation, &other.allocation)
        }
        #[doc(hidden)]
        pub fn storage_bytes(&self) -> u64 {
            self.allocation.bytes()
        }
        /// A non-owning observation of this tensor's allocation.
        pub fn observe_storage(&self) -> TensorStorageObserver {
            TensorStorageObserver {
                allocation: Arc::downgrade(&self.allocation),
                identity: self.allocation.identity(),
                topology: self.device.topology.clone(),
                memory: self.device.memory.clone(),
            }
        }
        /// The bytes of this contiguous view charged to each store: the
        /// share of its allocation's path charges its byte range covers.
        pub fn view_store_charges(
            &self,
        ) -> Result<Vec<(seismic_topology::StoreId, u64)>, TensorError> {
            if layout::canonical(self.representation, &self.extents)?.strides != self.strides {
                return Err(TensorError::ViewCharge(ViewChargeError::NotContiguous));
            }
            let paths = self
                .allocation
                .path_charges(&self.device.memory)
                .ok_or(TensorError::ViewCharge(ViewChargeError::Uncharged))?;
            let mut skip = self.byte_offset;
            let mut remaining = self.byte_len;
            let mut held = Vec::new();
            for (path, bytes) in paths {
                let ignored = skip.min(bytes);
                skip -= ignored;
                let covered = remaining.min(bytes - ignored);
                remaining -= covered;
                if covered != 0 {
                    held.push((path, covered));
                }
            }
            if remaining != 0 {
                return Err(TensorError::ViewCharge(ViewChargeError::BeyondCharge));
            }
            Ok(crate::memory::store_charges(&self.device.topology, held)
                .into_iter()
                .collect())
        }
        #[doc(hidden)]
        pub fn reclaimable_bytes<'a>(
            tensors: impl IntoIterator<Item = &'a Arc<Self>>,
        ) -> Result<u64, TensorError> {
            use std::collections::{HashMap, HashSet};

            let mut handles = HashSet::new();
            let mut inners: HashMap<*const Self, (usize, &Arc<Self>)> = HashMap::new();
            for tensor in tensors {
                if !handles.insert(tensor as *const Arc<Self>) {
                    continue;
                }
                let entry = inners.entry(Arc::as_ptr(tensor)).or_insert((0, tensor));
                entry.0 += 1;
            }

            let mut allocations: HashMap<*const Allocation, (usize, &Arc<Allocation>)> =
                HashMap::new();
            for (selected, inner) in inners.values() {
                if *selected != Arc::strong_count(inner) {
                    continue;
                }
                let allocation = &inner.allocation;
                let entry = allocations
                    .entry(Arc::as_ptr(allocation))
                    .or_insert((0, allocation));
                entry.0 += 1;
            }
            allocations
                .values()
                .try_fold(0u64, |total, (selected, allocation)| {
                    let bytes = if *selected == Arc::strong_count(allocation) {
                        allocation.bytes()
                    } else {
                        0
                    };
                    total.checked_add(bytes).ok_or_else(|| {
                        TensorError::Execution(ExecutionError::AllocationFailed(
                            "reclaimable allocation total overflow".to_owned(),
                        ))
                    })
                })
        }
        pub fn slice_leading(&self, start: u64, end: u64) -> Result<Self, TensorError> {
            if let Some(slabs) = &self.slabs {
                if start >= end || end > self.extents[0] {
                    return Err(TensorError::SliceOutOfBounds {
                        extent: self.extents[0],
                        start,
                        end,
                    });
                }
                if start / slabs.rows_per_slab != (end - 1) / slabs.rows_per_slab {
                    return Err(TensorError::SlabLayout(
                        "slice crosses a slab boundary".into(),
                    ));
                }
                let slab_index = usize::try_from(start / slabs.rows_per_slab).map_err(|_| {
                    TensorError::SlabLayout("slab index exceeds host domain".into())
                })?;
                let region = slabs.regions[slab_index]
                    .as_ref()
                    .ok_or_else(|| TensorError::SlabLayout("slice refers to a free slab".into()))?;
                let local_start = start % slabs.rows_per_slab;
                return region.slice_leading(local_start, local_start + end - start);
            }
            self.view(&crate::api::kernel::ViewOperation::LeadingSlice { start, end })
        }
        pub fn reshape(&self, extents: &[u64]) -> Result<Self, TensorError> {
            if let Some(slabs) = &self.slabs {
                if extents.first() != self.extents.first() {
                    return Err(TensorError::SlabLayout(
                        "slab reshape must preserve its leading row axis".into(),
                    ));
                }
                let view = layout::apply_view(
                    self.representation,
                    layout::ViewGeometry {
                        extents: self.extents.clone(),
                        strides: self.strides.clone(),
                        byte_offset: 0,
                        byte_len: self.byte_len,
                    },
                    &crate::api::kernel::ViewOperation::Reshape {
                        extents: extents.to_vec(),
                    },
                )?;
                let region_extents = std::iter::once(slabs.rows_per_slab)
                    .chain(extents.iter().skip(1).copied())
                    .collect::<Vec<_>>();
                let regions = slabs
                    .regions
                    .iter()
                    .map(|region| {
                        region
                            .as_ref()
                            .map(|region| region.reshape(&region_extents).map(Arc::new))
                            .transpose()
                    })
                    .collect::<Result<Vec<_>, TensorError>>()?;
                let lease = self
                    .slab_lease
                    .as_ref()
                    .expect("slab view has a binding lease")
                    .clone();
                lease.fetch_add(1, Ordering::AcqRel);
                return Ok(Self {
                    device: self.device.clone(),
                    allocation: self.allocation.clone(),
                    byte_offset: self.byte_offset,
                    byte_len: view.byte_len,
                    representation: self.representation,
                    extents: view.extents,
                    strides: view.strides,
                    reserved: false,
                    slabs: Some(SlabView {
                        rows_per_slab: slabs.rows_per_slab,
                        regions,
                    }),
                    alias_identity: self.alias_identity,
                    slab_lease: Some(lease),
                });
            }
            self.view(&crate::api::kernel::ViewOperation::Reshape {
                extents: extents.to_vec(),
            })
        }
        /// This tensor seen through one view operation (`layout::apply_view`).
        fn view(&self, operation: &crate::api::kernel::ViewOperation) -> Result<Self, TensorError> {
            let view = layout::apply_view(
                self.representation,
                layout::ViewGeometry {
                    extents: self.extents.clone(),
                    strides: self.strides.clone(),
                    byte_offset: self.byte_offset,
                    byte_len: self.byte_len,
                },
                operation,
            )?;
            if self.reserved && view.byte_offset + view.byte_len > self.allocation.bytes() {
                // A view of a reserved tensor reaching past its committed
                // rows stays reserved; one within them is an ordinary view.
                return Ok(Self {
                    device: self.device.clone(),
                    allocation: self.allocation.clone(),
                    byte_offset: view.byte_offset,
                    byte_len: view.byte_len,
                    representation: self.representation,
                    extents: view.extents,
                    strides: view.strides,
                    reserved: true,
                    slabs: None,
                    alias_identity: None,
                    slab_lease: None,
                });
            }
            Ok(Self::new_view(
                self.device.clone(),
                self.allocation.clone(),
                view.byte_offset,
                view.byte_len,
                self.representation,
                view.extents,
                view.strides,
            ))
        }
        pub fn descriptor(&self) -> TensorDescriptor {
            TensorDescriptor {
                device: self.device.kind.identity(),
                representation: self.representation,
                extents: self.extents.clone(),
                strides: self.strides.clone(),
                allocation: self.alias_identity(),
                byte_offset: self.byte_offset,
                byte_len: self.byte_len,
            }
        }
    }

    #[cfg(test)]
    mod reader_tests {
        use super::TensorInner;
        use seismic_lang::registry;
        use seismic_lang::types::DType;

        /// A reader upload runs on a Windows main thread (1 MiB of stack) in a
        /// model's Prepare; it must not need a stack of its own size.
        #[test]
        fn reader_upload_fits_a_small_thread_stack() {
            std::thread::Builder::new()
                .stack_size(256 * 1024)
                .spawn(|| {
                    let devices = crate::devices::Devices::discover(&[]).unwrap();
                    let device = devices.open_backend(registry::BackendName::Cpu).unwrap();
                    let count = 3 * 1024 * 1024 / 4 + 5;
                    let tensor = TensorInner::from_host(
                        &device,
                        registry::dense(DType::F32),
                        &[count],
                        &vec![0; count as usize * 4],
                    )
                    .unwrap();
                    let bytes = (0..count * 4).map(|i| i as u8).collect::<Vec<_>>();
                    tensor.write_from_reader(&mut bytes.as_slice()).unwrap();
                    assert_eq!(tensor.read_to_host().unwrap(), bytes);
                })
                .unwrap()
                .join()
                .unwrap();
        }

        #[test]
        fn a_view_reports_only_its_owned_byte_interval() {
            let device = crate::devices::Devices::discover(&[])
                .unwrap()
                .open_backend(registry::BackendName::Cpu)
                .unwrap();
            let tensor = TensorInner::zeros(&device, registry::dense(DType::U32), &[4, 8]).unwrap();
            let view = tensor.slice_leading(1, 3).unwrap();
            assert_eq!(
                view.view_store_charges().unwrap(),
                vec![(device.topology().host_ram().id(), 64)]
            );
            assert_eq!(view.observe_storage().charged_bytes(), Some(128));
        }
    }
}

pub mod kernel {
    use super::{device::DeviceInner, tensor::TensorInner, CallError};
    use seismic_compiler::errors::PreparationError;
    use seismic_compiler::prepared::ArgumentValue;
    use seismic_lang::checked::{CheckedModule, SourceError};
    use seismic_lang::entry::ElementBindings;
    use seismic_lang::ids::EntryId;
    use std::collections::VecDeque;
    use std::sync::Arc;

    #[derive(Debug)]
    pub enum PrepareError {
        Source(SourceError),
        Preparation(PreparationError),
    }

    /// An explicitly selected, formed native implementation.
    pub struct NativePreparedAny {
        pub(crate) inner: Arc<crate::native::NativePrepared>,
    }

    impl NativePreparedAny {
        pub fn invocation_workspace_bytes(&self) -> u64 {
            self.inner.invocation_workspace_bytes()
        }
        pub fn artifact(&self) -> &crate::native::NativeArtifactIdentity {
            self.inner.artifact()
        }
        pub fn specialization(&self) -> &seismic_lang::checked::NativeSpecialization {
            self.inner.specialization()
        }
        pub fn implementation(&self) -> &seismic_lang::checked::NativeImplementation {
            self.inner.implementation()
        }
        /// Check the launches of a call whose dimensions are `dimensions`
        /// against the formed functions and the device limits.
        pub fn admits(
            &self,
            dimensions: &std::collections::BTreeMap<String, u64>,
        ) -> Result<(), super::CallError> {
            self.inner.admits(dimensions)
        }
        /// Calls cycling through `rotation`, each argument set validated
        /// and placed once, as `seismic-measure` times them.
        pub fn point(
            &self,
            rotation: Vec<EncodedArgs>,
        ) -> Result<crate::native::KernelPoint, super::CallError> {
            crate::native::KernelPoint::place(&self.inner, rotation, Vec::new(), None)
        }
    }

    #[derive(Clone)]
    enum EncodedArgument {
        Tensor(Arc<TensorInner>),
        Scalar(EncodedScalar),
    }

    /// Opaque result edge inside one workflow draft. The workflow identity
    /// prevents a result from being wired into another draft.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct WorkflowResultRef {
        pub(crate) workflow: u64,
        pub(crate) node: u32,
        pub(crate) result: u32,
    }

    #[derive(Clone)]
    pub(crate) enum EncodedWorkflowArgument {
        Tensor(WorkflowTensorArgument),
        Scalar(EncodedScalar),
        ScalarResult(WorkflowResultRef),
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    pub enum ViewOperation {
        LeadingSlice { start: u64, end: u64 },
        Reshape { extents: Vec<u64> },
    }

    #[derive(Clone)]
    pub(crate) enum WorkflowTensorArgument {
        External(Arc<TensorInner>),
        Result(WorkflowResultRef),
        ResultView {
            result: WorkflowResultRef,
            operations: Vec<ViewOperation>,
        },
    }

    /// Generated workflow arguments retain symbolic producer edges until
    /// whole-workflow binding resolves them. They cannot be passed to the
    /// one-node call path.
    #[derive(Clone)]
    pub struct EncodedWorkflowArgs {
        arguments: Vec<EncodedWorkflowArgument>,
    }

    impl EncodedWorkflowArgs {
        #[doc(hidden)]
        pub fn new() -> Self {
            Self {
                arguments: Vec::new(),
            }
        }
        #[doc(hidden)]
        pub fn push_external_tensor(&mut self, tensor: Arc<TensorInner>) {
            self.arguments.push(EncodedWorkflowArgument::Tensor(
                WorkflowTensorArgument::External(tensor),
            ));
        }
        #[doc(hidden)]
        pub fn push_result_tensor(&mut self, result: WorkflowResultRef) {
            self.arguments.push(EncodedWorkflowArgument::Tensor(
                WorkflowTensorArgument::Result(result),
            ));
        }
        #[doc(hidden)]
        pub fn push_result_tensor_view(
            &mut self,
            result: WorkflowResultRef,
            operations: Vec<ViewOperation>,
        ) {
            self.arguments.push(EncodedWorkflowArgument::Tensor(
                WorkflowTensorArgument::ResultView { result, operations },
            ));
        }
        #[doc(hidden)]
        pub fn push_scalar(&mut self, value: EncodedScalar) {
            self.arguments.push(EncodedWorkflowArgument::Scalar(value));
        }
        #[doc(hidden)]
        pub fn push_result_scalar(&mut self, result: WorkflowResultRef) {
            self.arguments
                .push(EncodedWorkflowArgument::ScalarResult(result));
        }
        pub(crate) fn into_arguments(self) -> Vec<EncodedWorkflowArgument> {
            self.arguments
        }
        pub(crate) fn arguments(&self) -> &[EncodedWorkflowArgument] {
            &self.arguments
        }
    }

    pub struct PendingWorkflowResults {
        workflow: u64,
        node: u32,
        next: u32,
        count: u32,
    }

    impl PendingWorkflowResults {
        pub(crate) fn new(workflow: u64, node: u32, count: u32) -> Self {
            Self {
                workflow,
                node,
                next: 0,
                count,
            }
        }

        #[doc(hidden)]
        pub fn take(&mut self) -> WorkflowResultRef {
            assert!(
                self.next < self.count,
                "generated workflow result schema over-read"
            );
            let result = WorkflowResultRef {
                workflow: self.workflow,
                node: self.node,
                result: self.next,
            };
            self.next += 1;
            result
        }
    }

    #[doc(hidden)]
    pub use crate::driver::workflow::ScalarValue as EncodedScalar;
    impl EncodedScalar {
        pub(crate) fn value(&self) -> ArgumentValue {
            match self.clone() {
                Self::F32Bits(bits) => ArgumentValue::F32(f32::from_bits(bits)),
                Self::F16(value) => ArgumentValue::F16(value),
                Self::BF16(value) => ArgumentValue::BF16(value),
                Self::I32(value) => ArgumentValue::I32(value),
                Self::U32(value) => ArgumentValue::U32(value),
                Self::U64(value) => ArgumentValue::U64(value),
                Self::Bool(value) => ArgumentValue::Bool(value),
                Self::Index(value) => ArgumentValue::Index(value),
                Self::Range { start, end } => ArgumentValue::Range { start, end },
            }
        }
    }

    /// Generated call arguments. Each tensor descriptor and the allocation
    /// keepalive it describes are one value, never parallel vectors joined by
    /// ordinal at runtime.
    #[derive(Clone)]
    pub struct EncodedArgs {
        arguments: Vec<EncodedArgument>,
    }
    impl EncodedArgs {
        #[doc(hidden)]
        pub fn new() -> Self {
            Self {
                arguments: Vec::new(),
            }
        }
        pub(crate) fn into_workflow(self) -> EncodedWorkflowArgs {
            let arguments = self
                .arguments
                .into_iter()
                .map(|argument| match argument {
                    EncodedArgument::Tensor(tensor) => {
                        EncodedWorkflowArgument::Tensor(WorkflowTensorArgument::External(tensor))
                    }
                    EncodedArgument::Scalar(value) => EncodedWorkflowArgument::Scalar(value),
                })
                .collect();
            EncodedWorkflowArgs { arguments }
        }
        #[doc(hidden)]
        pub fn push_tensor(&mut self, tensor: Arc<TensorInner>) {
            self.arguments.push(EncodedArgument::Tensor(tensor));
        }
        #[doc(hidden)]
        pub fn push_scalar(&mut self, value: EncodedScalar) {
            self.arguments.push(EncodedArgument::Scalar(value));
        }
        pub(crate) fn values(&self) -> Vec<ArgumentValue> {
            self.arguments
                .iter()
                .map(|argument| match argument {
                    EncodedArgument::Tensor(tensor) => ArgumentValue::Tensor(tensor.descriptor()),
                    EncodedArgument::Scalar(value) => value.value(),
                })
                .collect()
        }
        pub(crate) fn tensor(&self, ordinal: usize) -> Option<&Arc<TensorInner>> {
            match self.arguments.get(ordinal) {
                Some(EncodedArgument::Tensor(tensor)) => Some(tensor),
                _ => None,
            }
        }
        pub(crate) fn tensors(&self) -> impl Iterator<Item = Option<&Arc<TensorInner>>> {
            self.arguments.iter().map(|argument| match argument {
                EncodedArgument::Tensor(tensor) => Some(tensor),
                EncodedArgument::Scalar(_) => None,
            })
        }
        /// The same arguments with every tensor replaced by `map`'s.
        pub(crate) fn map_tensors<E>(
            &self,
            mut map: impl FnMut(&Arc<TensorInner>) -> Result<Arc<TensorInner>, E>,
        ) -> Result<Self, E> {
            let arguments = self
                .arguments
                .iter()
                .map(|argument| {
                    Ok(match argument {
                        EncodedArgument::Tensor(tensor) => EncodedArgument::Tensor(map(tensor)?),
                        EncodedArgument::Scalar(value) => EncodedArgument::Scalar(value.clone()),
                    })
                })
                .collect::<Result<_, E>>()?;
            Ok(Self { arguments })
        }
    }

    /// Tensor result storage supplied by generated native entry bindings.
    pub struct EncodedOutputs {
        tensors: Vec<Arc<TensorInner>>,
    }
    impl EncodedOutputs {
        #[doc(hidden)]
        pub fn new() -> Self {
            Self {
                tensors: Vec::new(),
            }
        }
        #[doc(hidden)]
        pub fn push_tensor(&mut self, tensor: Arc<TensorInner>) {
            self.tensors.push(tensor);
        }
        pub(crate) fn into_tensors(self) -> Vec<Arc<TensorInner>> {
            self.tensors
        }
    }

    #[derive(Clone)]
    pub enum DecodedValue {
        Tensor(Arc<TensorInner>),
        Scalar(ArgumentValue),
    }
    pub struct DecodedResults {
        values: VecDeque<DecodedValue>,
    }
    impl DecodedResults {
        pub fn into_values(self) -> Vec<DecodedValue> {
            self.values.into_iter().collect()
        }
        pub(crate) fn new(values: Vec<DecodedValue>) -> Self {
            Self {
                values: values.into(),
            }
        }
        #[doc(hidden)]
        pub fn take_tensor(&mut self) -> Arc<TensorInner> {
            match self.values.pop_front() {
                Some(DecodedValue::Tensor(tensor)) => tensor,
                _ => panic!("unsafe generated Entry contract decoded a non-tensor as a tensor"),
            }
        }
        #[doc(hidden)]
        pub fn take_scalar(&mut self) -> ArgumentValue {
            match self.values.pop_front() {
                Some(DecodedValue::Scalar(value)) => value,
                _ => panic!("unsafe generated Entry contract decoded a non-scalar as a scalar"),
            }
        }
    }

    type WorkflowOutcome = seismic_lang::failure::SourceTermination<
        Vec<Vec<DecodedValue>>,
        seismic_compiler::errors::CheckFailure,
    >;

    type PendingCompletion = Box<dyn FnOnce() -> Result<WorkflowOutcome, super::CallError>>;

    enum CompletionState {
        Pending(PendingCompletion),
        Running,
        Complete(Result<WorkflowOutcome, super::CallError>),
        Panicked,
    }

    /// Submitted workflow owner. It retains native execution, reservations,
    /// buffers, and permits until the first resolution completes the device
    /// work; completed values are then cached for independent typed resolves.
    pub struct WorkflowCompletionAny {
        workflow: u64,
        state: std::sync::Mutex<CompletionState>,
        changed: std::sync::Condvar,
    }

    impl WorkflowCompletionAny {
        pub(crate) fn pending(
            workflow: u64,
            completion: impl FnOnce() -> Result<WorkflowOutcome, super::CallError> + 'static,
        ) -> Self {
            Self {
                workflow,
                state: std::sync::Mutex::new(CompletionState::Pending(Box::new(completion))),
                changed: std::sync::Condvar::new(),
            }
        }

        /// Complete native work exactly once and inspect its typed source
        /// termination. A source stop has no returned product; device errors
        /// remain errors. Caller-owned input storage already contains the
        /// terminal prefix state when this method returns.
        pub fn outcome(&self) -> Result<WorkflowOutcome, super::CallError> {
            loop {
                let mut state = self
                    .state
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                match &*state {
                    CompletionState::Complete(result) => return result.clone(),
                    CompletionState::Running => {
                        drop(
                            self.changed
                                .wait(state)
                                .unwrap_or_else(std::sync::PoisonError::into_inner),
                        );
                    }
                    CompletionState::Panicked => {
                        panic!("workflow completion previously panicked")
                    }
                    CompletionState::Pending(_) => {
                        let CompletionState::Pending(completion) =
                            std::mem::replace(&mut *state, CompletionState::Running)
                        else {
                            unreachable!("pending workflow completion changed under its lock")
                        };
                        drop(state);

                        // Device completion must not run under a host mutex. Publish the
                        // terminal state afterward so concurrent resolvers wait without
                        // duplicating submission or synchronization.
                        let completed =
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(completion));
                        let mut state = self
                            .state
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        match completed {
                            Ok(result) => {
                                *state = CompletionState::Complete(result.clone());
                                self.changed.notify_all();
                                return result;
                            }
                            Err(payload) => {
                                *state = CompletionState::Panicked;
                                self.changed.notify_all();
                                drop(state);
                                std::panic::resume_unwind(payload);
                            }
                        }
                    }
                }
            }
        }

        pub fn resolve(
            &self,
            outputs: Vec<WorkflowResultRef>,
        ) -> Result<DecodedResults, super::CallError> {
            let values = match self.outcome()? {
                seismic_lang::failure::SourceTermination::Returned(values) => values,
                seismic_lang::failure::SourceTermination::Failed(failure) => {
                    return Err(super::CallError::Execution(
                        seismic_compiler::errors::ExecutionError::DataCheckFailed(failure),
                    ));
                }
            };
            let values = outputs
                .into_iter()
                .map(|output| {
                    if output.workflow != self.workflow {
                        return Err(super::CallError::Workflow(
                            super::WorkflowError::CrossWorkflowResult,
                        ));
                    }
                    values
                        .get(output.node as usize)
                        .and_then(|results| results.get(output.result as usize))
                        .cloned()
                        .ok_or(super::CallError::Workflow(
                            super::WorkflowError::MissingProducerResult,
                        ))
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(DecodedResults::new(values))
        }
    }

    impl Drop for WorkflowCompletionAny {
        fn drop(&mut self) {
            // Abandoning a submitted workflow must still synchronize before its
            // captured reservations, access permits, buffers, and native objects
            // are released. Completion errors cannot be reported from Drop.
            let state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let unfinished = matches!(
                &*state,
                CompletionState::Pending(_) | CompletionState::Running
            );
            drop(state);
            if unfinished {
                // A destructor cannot surface either a device error or a
                // completion panic. The completion path has already
                // published `Panicked` and notified waiters before resuming
                // that panic, so swallowing it here cannot strand another
                // resolver or release resources before synchronization ran.
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _ = self.outcome();
                }));
            }
        }
    }

    pub struct PreparedAny {
        pub(crate) inner: crate::backends::PreparedKind,
    }

    impl PreparedAny {
        pub fn feedback_report(&self) -> Option<&seismic_compiler::feedback::FeedbackReport> {
            self.inner.feedback_report()
        }
    }

    pub struct WorkflowDraftAny {
        inner: crate::backends::WorkflowDraftKind,
    }

    /// Workflow references and the first invocation are bound without acquiring
    /// resources. Dependent invocations bind after producer results complete.
    pub struct BoundWorkflowAny {
        inner: crate::backends::BoundWorkflowKind,
    }

    impl BoundWorkflowAny {
        /// Bounds actual newly acquired backing throughout this run, including
        /// storage reached after producer completion or inside source control.
        pub fn with_allocation_limit(mut self, bytes: u64) -> Self {
            self.inner.set_allocation_limit(bytes);
            self
        }
        /// Initial non-external reservation. Reached private allocations are charged during execution.
        pub fn initial_allocation_bytes(&self) -> u64 {
            self.inner.initial_allocation_bytes()
        }
    }

    /// Initially admitted workflow. It owns initial capacity, allocations,
    /// persistent leases, access permits, and a backend submission. Planned
    /// private acquisitions may still fail after source execution begins.
    pub struct AdmittedWorkflowAny {
        inner: crate::backends::AdmittedWorkflowKind,
    }

    pub fn workflow(device: &Arc<DeviceInner>) -> Result<WorkflowDraftAny, super::WorkflowError> {
        device
            .kind
            .workflow()
            .map(|inner| WorkflowDraftAny { inner })
    }

    pub fn enqueue(
        workflow: &mut WorkflowDraftAny,
        kernel: &Arc<PreparedAny>,
        args: EncodedWorkflowArgs,
    ) -> Result<PendingWorkflowResults, super::WorkflowError> {
        workflow.inner.enqueue(&kernel.inner, args)
    }

    pub fn bind_workflow(workflow: WorkflowDraftAny) -> Result<BoundWorkflowAny, CallError> {
        workflow
            .inner
            .bind()
            .map(|inner| BoundWorkflowAny { inner })
    }

    pub fn admit_workflow(workflow: BoundWorkflowAny) -> Result<AdmittedWorkflowAny, CallError> {
        workflow
            .inner
            .admit()
            .map(|inner| AdmittedWorkflowAny { inner })
    }

    pub fn submit_workflow(
        workflow: AdmittedWorkflowAny,
    ) -> Result<WorkflowCompletionAny, CallError> {
        workflow.inner.submit()
    }
    pub struct FeedbackPreparation<'a> {
        inner: crate::backends::FeedbackKind<'a>,
    }
    impl FeedbackPreparation<'_> {
        pub fn continue_for(
            &mut self,
            additional: std::time::Duration,
        ) -> Result<PreparedAny, PrepareError> {
            self.inner
                .continue_for(additional)
                .map(|inner| PreparedAny { inner })
        }
        pub fn report(&self) -> &seismic_compiler::feedback::FeedbackReport {
            self.inner.report()
        }
    }
    pub fn start_feedback<'a>(
        module: &CheckedModule,
        entry: EntryId,
        bindings: ElementBindings,
        device: &'a Arc<DeviceInner>,
        precision: seismic_lang::precision::PrecisionPolicy,
        options: seismic_compiler::feedback::FeedbackOptions,
    ) -> Result<(FeedbackPreparation<'a>, PreparedAny), PrepareError> {
        device
            .kind
            .start_feedback(module, entry, bindings, device, precision, options)
            .map(|(inner, kernel)| (FeedbackPreparation { inner }, PreparedAny { inner: kernel }))
    }

    pub fn prepare(
        module: &CheckedModule,
        entry: EntryId,
        bindings: ElementBindings,
        device: &Arc<DeviceInner>,
        options: seismic_compiler::feedback::PreparationOptions,
    ) -> Result<PreparedAny, PrepareError> {
        let logical = module
            .entry(entry, &bindings)
            .map_err(PrepareError::Source)?;
        prepare_logical(logical, bindings, device, options)
    }

    /// Prepare an already checked computation through the ordinary compiler,
    /// cache and device-admission path. Bound numerical recipes enter its
    /// logical identity; they do not add invocation arguments or another cache.
    pub(crate) fn prepare_logical(
        logical: seismic_lang::entry::LogicalEntry,
        bindings: ElementBindings,
        device: &Arc<DeviceInner>,
        options: seismic_compiler::feedback::PreparationOptions,
    ) -> Result<PreparedAny, PrepareError> {
        device
            .kind
            .prepare(logical, bindings, device, options)
            .map(|inner| PreparedAny { inner })
    }
    pub fn call(kernel: &Arc<PreparedAny>, args: EncodedArgs) -> Result<DecodedResults, CallError> {
        kernel.inner.call(args)
    }

    pub fn prepare_native(
        module: &CheckedModule,
        entry: EntryId,
        bindings: ElementBindings,
        device: &Arc<DeviceInner>,
        specialization: seismic_lang::checked::NativeSpecialization,
        cpu: Option<&'static crate::native::CpuNativeKernels>,
    ) -> Result<NativePreparedAny, PrepareError> {
        crate::native::NativePrepared::prepare(device, module, entry, bindings, specialization, cpu)
            .map(|inner| NativePreparedAny { inner })
    }

    pub fn call_native(
        kernel: &Arc<NativePreparedAny>,
        args: EncodedArgs,
    ) -> Result<DecodedResults, CallError> {
        kernel.inner.call(args)
    }

    pub fn call_native_into(
        kernel: &Arc<NativePreparedAny>,
        args: EncodedArgs,
        outputs: EncodedOutputs,
    ) -> Result<DecodedResults, CallError> {
        kernel.inner.call_into(args, outputs)
    }

    /// Tensor-result calls from different native entries in one ordered
    /// device submission. The caller retains output tensors until completion.
    pub struct NativeTensorBatchAny {
        inner: crate::native::NativeTensorBatch,
    }

    pub struct NativeTensorBatchCompletionAny {
        inner: crate::native::NativeTensorBatchCompletion,
    }

    impl NativeTensorBatchAny {
        pub fn new(device: &Arc<DeviceInner>) -> Self {
            Self {
                inner: crate::native::NativeTensorBatch::new(device.clone()),
            }
        }

        pub fn push(
            &mut self,
            kernel: &Arc<NativePreparedAny>,
            args: EncodedArgs,
            outputs: EncodedOutputs,
        ) -> Result<(), CallError> {
            self.inner.push(&kernel.inner, args, outputs)
        }

        pub fn submit(self) -> Result<NativeTensorBatchCompletionAny, CallError> {
            self.inner
                .submit()
                .map(|inner| NativeTensorBatchCompletionAny { inner })
        }
    }

    impl NativeTensorBatchCompletionAny {
        pub fn wait(self) -> Result<(), CallError> {
            self.inner.wait()
        }
    }

    pub fn call_native_with_commit(
        kernel: &Arc<NativePreparedAny>,
        args: EncodedArgs,
        commit: impl FnOnce(),
    ) -> Result<DecodedResults, CallError> {
        kernel.inner.call_with_commit(args, commit)
    }

    #[cfg(test)]
    mod completion_tests {
        use super::*;
        use std::sync::atomic::{AtomicUsize, Ordering};

        #[test]
        fn device_completion_failure_never_becomes_a_source_stop() {
            let completion = WorkflowCompletionAny::pending(3, || {
                Err(super::super::CallError::Execution(
                    seismic_compiler::errors::ExecutionError::DeviceLost(
                        "fixture device failure".into(),
                    ),
                ))
            });
            assert!(matches!(
                completion.outcome(),
                Err(super::super::CallError::Execution(
                    seismic_compiler::errors::ExecutionError::DeviceLost(_),
                ))
            ));
            assert!(matches!(
                completion.resolve(Vec::new()),
                Err(super::super::CallError::Execution(
                    seismic_compiler::errors::ExecutionError::DeviceLost(_),
                ))
            ));
        }

        #[test]
        fn completion_runs_once_and_caches() {
            let runs = Arc::new(AtomicUsize::new(0));
            let observed = runs.clone();
            let completion = WorkflowCompletionAny::pending(7, move || {
                observed.fetch_add(1, Ordering::SeqCst);
                Ok(seismic_lang::failure::SourceTermination::Returned(
                    Vec::new(),
                ))
            });
            completion.resolve(Vec::new()).unwrap();
            completion.resolve(Vec::new()).unwrap();
            assert_eq!(runs.load(Ordering::SeqCst), 1);
        }

        #[test]
        fn dropping_unresolved_completion_synchronizes() {
            let runs = Arc::new(AtomicUsize::new(0));
            let observed = runs.clone();
            let completion = WorkflowCompletionAny::pending(9, move || {
                observed.fetch_add(1, Ordering::SeqCst);
                Ok(seismic_lang::failure::SourceTermination::Returned(
                    Vec::new(),
                ))
            });
            drop(completion);
            assert_eq!(runs.load(Ordering::SeqCst), 1);
        }

        #[test]
        fn dropping_unresolved_panicking_completion_is_non_panicking() {
            let dropped = std::panic::catch_unwind(|| {
                let completion =
                    WorkflowCompletionAny::pending(11, move || panic!("completion fixture panic"));
                drop(completion);
            });
            assert!(dropped.is_ok());
        }
    }
}
