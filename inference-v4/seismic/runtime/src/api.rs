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
    NativeGraphSlotMismatch,
    NativeOutputLeaseConsumed,
    NativeExportStillLive,
    /// Every upload region a family slot was created with is still read by
    /// a submission in flight; a slot never grows its regions after sealing.
    UploadRegionsExhausted {
        regions: usize,
    },
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
            Self::Workflow(WorkflowError::NativeGraphSlotMismatch) => {
                f.write_str("native graph slot belongs to another graph")
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
    Execution(ExecutionError),
}
impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
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
    use crate::devices::{DeviceInfo, DeviceMemoryStatus, MemoryUsage, ObservationError};
    use crate::driver::Allocation;
    use seismic_compiler::errors::ExecutionError;
    use std::sync::Arc;

    /// One opened catalog device: its immutable description, execution
    /// resources and accounting scope.
    pub struct DeviceInner {
        pub(crate) info: DeviceInfo,
        pub(crate) capabilities: std::sync::OnceLock<Vec<String>>,
        pub(crate) kind: OpenedKind,
        /// The active submission trace, if any (`native::trace`).
        pub(crate) trace: std::sync::Mutex<Option<Arc<crate::native::trace::TraceSink>>>,
        /// Where formed native artifacts are looked up and kept.
        pub(crate) artifacts: Option<Arc<dyn crate::artifacts::ArtifactStore>>,
        /// Native submission order and CUDA graph replays.
        pub(crate) native: crate::native::NativeQueue,
    }
    impl DeviceInner {
        pub(crate) fn active_trace(&self) -> Option<Arc<crate::native::trace::TraceSink>> {
            self.trace
                .lock()
                .expect("trace slot lock is never poisoned")
                .clone()
        }
        pub fn info(&self) -> &DeviceInfo {
            &self.info
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
        pub fn tuning_identity(&self) -> String {
            format!("{};{}", self.info.name, self.kind.tuning_identity())
        }
        pub(crate) fn supports_representation(
            &self,
            representation: seismic_lang::ids::RepresentationId,
        ) -> bool {
            self.kind.supports_representation(representation)
        }
        pub fn memory_usage(&self) -> MemoryUsage {
            self.kind.memory_usage()
        }
        /// Bounds allocations made through this device. Charges already made
        /// count against the limit; they are never charged twice.
        pub fn set_memory_limit(&self, limit: Option<u64>) {
            self.kind.set_memory_limit(limit)
        }
        pub fn memory_status(&self) -> Result<DeviceMemoryStatus, ObservationError> {
            self.kind.memory_status()
        }
    }
}

pub mod tensor {
    use super::{device::DeviceInner, HostRegion, TensorError};
    use crate::driver::{write_zeros, Allocation};
    use crate::layout;
    use seismic_compiler::errors::ExecutionError;
    use seismic_compiler::prepared::TensorDescriptor;
    use seismic_lang::ids::RepresentationId;
    use seismic_lang::registry::representation_info;
    use std::io::Read;
    use std::sync::{Arc, Weak};

    /// A non-owning observation of one physical tensor allocation. It keeps
    /// no storage alive and reads the same charge as the device ledger.
    pub struct TensorStorageObserver {
        allocation: Weak<Allocation>,
        identity: u64,
    }

    impl TensorStorageObserver {
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

        pub fn tensor(
            &self,
            representation: RepresentationId,
            extents: &[u64],
            byte_offset: u64,
        ) -> Result<TensorInner, TensorError> {
            if !self.device.supports_representation(representation) {
                return Err(TensorError::UnsupportedRepresentation {
                    backend: self.device.info().backend,
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
                    backend: device.info().backend,
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

        pub fn copy_committed_leading_row(&self, from: u64, to: u64) -> Result<(), TensorError> {
            let layout = layout::canonical(self.representation, &self.extents)?;
            let row = leading_row_bytes(&self.extents, layout.byte_len)?;
            let committed = self.committed_rows();
            if self.byte_offset != 0
                || self.byte_len != layout.byte_len
                || from >= committed
                || to >= committed
            {
                return Err(TensorError::Uncommitted {
                    rows: from.max(to) + 1,
                    committed,
                });
            }
            if from == to {
                return Ok(());
            }
            let _access = self.allocation.acquire(true);
            const CHUNK: u64 = 1 << 16;
            let mut scratch = vec![0u8; usize::try_from(row.min(CHUNK)).unwrap_or(CHUNK as usize)];
            let (source, destination) = (from * row, to * row);
            let mut offset = 0u64;
            while offset < row {
                let length = usize::try_from((row - offset).min(CHUNK)).unwrap_or(CHUNK as usize);
                self.allocation
                    .storage()
                    .read(source + offset, &mut scratch[..length])?;
                self.allocation
                    .storage()
                    .write(destination + offset, &scratch[..length])?;
                offset += length as u64;
            }
            Ok(())
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
                    backend: device.info().backend,
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
                    backend: device.info().backend,
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
            }
        }
        pub fn read_to_host(&self) -> Result<Vec<u8>, ExecutionError> {
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
            let mut scratch = [0u8; 1024 * 1024];
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
        #[doc(hidden)]
        pub fn shares_allocation(&self, other: &Self) -> bool {
            Arc::ptr_eq(&self.allocation, &other.allocation)
        }
        #[doc(hidden)]
        pub fn storage_bytes(&self) -> u64 {
            self.allocation.bytes()
        }
        pub fn observe_storage(&self) -> TensorStorageObserver {
            TensorStorageObserver {
                allocation: Arc::downgrade(&self.allocation),
                identity: self.allocation.identity(),
            }
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
            self.view(&crate::api::kernel::ViewOperation::LeadingSlice { start, end })
        }
        pub fn reshape(&self, extents: &[u64]) -> Result<Self, TensorError> {
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
                allocation: self.allocation.identity(),
                byte_offset: self.byte_offset,
                byte_len: self.byte_len,
            }
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
        /// Time calls cycling through `rotation` on the device.
        pub fn measure(
            &self,
            rotation: Vec<EncodedArgs>,
            options: &crate::native::MeasureOptions,
        ) -> Result<crate::native::Measurement, super::CallError> {
            self.inner.measure(rotation, options)
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
                    ))
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
        device
            .kind
            .prepare(module, entry, bindings, device, options)
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
