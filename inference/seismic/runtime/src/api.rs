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
    /// Issuing a submission failed after earlier work of it was issued, and
    /// fencing that work failed as well. Both are kept: either may decide
    /// recovery.
    Unfenced {
        issue: Box<CallError>,
        fence: ExecutionError,
    },
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

/// How a native graph's exact class fails the layout certified for its
/// topology.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LayoutMismatch {
    /// Its ports, launches, their results and scratch, exports or stores
    /// differ from the layout's topology.
    Topology,
    /// Its `buffer` takes `bytes`, more than the layout's `capacity` for it.
    Exceeds {
        buffer: GraphBuffer,
        bytes: u64,
        capacity: u64,
    },
}

/// A buffer of a native graph's storage: a port of the given extents, a
/// launch's result of the given extents, or a launch's scratch buffer.
/// Launches count from 0 in the graph's order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GraphBuffer {
    Port {
        port: usize,
        extents: Vec<u64>,
    },
    Result {
        launch: usize,
        result: usize,
        extents: Vec<u64>,
    },
    Scratch {
        launch: usize,
        scratch: usize,
    },
}

impl std::fmt::Display for LayoutMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Topology => f.write_str("native graph topology differs from its certified layout"),
            Self::Exceeds {
                buffer,
                bytes,
                capacity,
            } => {
                f.write_str("native graph ")?;
                match buffer {
                    GraphBuffer::Port { port, extents } => write!(f, "port {port} {extents:?}")?,
                    GraphBuffer::Result {
                        launch,
                        result,
                        extents,
                    } => write!(f, "launch {launch}'s result {result} {extents:?}")?,
                    GraphBuffer::Scratch { launch, scratch } => {
                        write!(f, "launch {launch}'s scratch {scratch}")?
                    }
                }
                write!(
                    f,
                    " takes {bytes} bytes, beyond the {capacity} of its certified layout"
                )
            }
        }
    }
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
    /// A tensor bound to a sealed graph's port that differs from the port's
    /// representation, extents, strides or length, or a read-only tensor
    /// bound where the graph writes.
    NativePortBinding {
        port: usize,
        expected: String,
        bound: String,
    },
    NativeGraphArgumentMismatch {
        parameter: usize,
    },
    NativePortAlreadyBound {
        port: usize,
    },
    /// A slab view bound to a port a launch reads its grid from: a grid is
    /// read whole from its binding, which a slab view's is not.
    NativePortSlabGrid {
        port: usize,
    },
    /// Chunked reserved storage, which has no one buffer, bound where device
    /// work would bind it directly: a device tensor port (parameter or
    /// result) not marked `records`, or a language kernel's argument. A
    /// record port reaches it through a table and a host port, never bound,
    /// orders by its packed span.
    ChunkedBinding {
        parameter: String,
    },
    /// A tensor with elements other than a slab view bound to a slab port
    /// (`&slab`): its natives read its binding as a slab address table.
    PlainSlabBinding {
        parameter: String,
    },
    /// A graph's record port bound other than to whole records of a
    /// canonical view of a tensor bound per run or of the graph's
    /// workspace: its table would be formed per run.
    RecordPortBinding {
        parameter: String,
    },
    /// A call that parks (an awaited port, `&await`) without a park node
    /// just before it, or a park node before anything but such a call
    /// (`TimelineNode::Park`); `node` is the call's ordinal, or the node
    /// count where a park ends the graph.
    ParkUnpaired {
        node: usize,
    },
    /// A publication (`TimelineNode::Publish`) before any call: it
    /// publishes the call before it.
    PublishUnpreceded,
    /// A launch condition of node `node` naming no word of an i32 tensor
    /// the host writes: a node condition (`NativeGraphDraft::enqueue_under`)
    /// of no external or owned input i32 port of its graph, or a condition
    /// its kernel declares (`NativeLaunch::conditions`) bound to a node
    /// result, a local or a record port.
    ConditionWord {
        node: usize,
    },
    /// A launch condition's word bound to storage the host does not
    /// address, refused as its call or graph's bindings form: `parameter`
    /// names the call's parameter or the graph's port.
    UnaddressedCondition {
        parameter: String,
    },
    /// Node `node`'s shared result (`NativeGraphDraft::share`) is not one a
    /// graph may place where the result it shares lies: of another
    /// description, read, exported or stored, sharing a stored or shared
    /// result, or of a node whose condition is not exclusive of the other's.
    SharedResult {
        node: usize,
    },
    /// Standalone calls have no host-service boundary; this backend requires
    /// an explicit graph park for an awaited entry.
    ParkRequiresGraph,
    /// A publication bound answered (`set_answered`): only a park waits
    /// for an answer.
    NotPark {
        node: usize,
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
    NativeGraphLayoutMismatch(LayoutMismatch),
    NativeOutputLeaseConsumed,
    NativeExportStillLive,
    /// No upload region a family slot was created with holds the run's
    /// `bytes` of input: the largest holds `largest`.
    UploadRegionTooSmall {
        bytes: u64,
        largest: u64,
    },
    /// Tensors written together into one upload region exceed its byte
    /// domain.
    UploadBytesOverflow,
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
    /// A run was sampled for its spans' device time on a backend that times
    /// none (`native::spans`).
    SpanTimingUnsupported {
        backend: seismic_lang::registry::BackendName,
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
            Self::Unfenced { issue, fence } => {
                write!(f, "{issue}; fencing the work issued before it also failed ({fence})")
            }
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
            Self::Workflow(WorkflowError::NativePortBinding { port, expected, bound }) => {
                write!(f, "native graph port {port} takes {expected}, bound {bound}")
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
            Self::Workflow(WorkflowError::NativePortSlabGrid { port }) => {
                write!(f, "native graph port {port} is a launch grid, which binds no slab view")
            }
            Self::Workflow(WorkflowError::PlainSlabBinding { parameter }) => write!(
                f,
                "`{parameter}` is a slab port, which binds a slab view or no elements"
            ),
            Self::Workflow(WorkflowError::ChunkedBinding { parameter }) => write!(
                f,
                "`{parameter}`: chunked reserved storage binds only to a native record port"
            ),
            Self::Workflow(WorkflowError::RecordPortBinding { parameter }) => write!(
                f,
                "`{parameter}`: a native graph's record port binds only whole records of a \
                 canonical view of a tensor bound per run or of the graph's workspace"
            ),
            Self::Workflow(WorkflowError::ParkUnpaired { node }) => write!(
                f,
                "native graph node {node}: a call that parks follows exactly one park node, and a park node \
                 precedes only such a call"
            ),
            Self::Workflow(WorkflowError::PublishUnpreceded) => {
                write!(f, "native graph: a publication precedes every call, so it publishes none")
            }
            Self::Workflow(WorkflowError::SharedResult { node }) => write!(
                f,
                "native graph node {node}'s shared result is not of the result it shares' description, is used, \
                 or its node's condition is not exclusive of that result's node's"
            ),
            Self::Workflow(WorkflowError::ConditionWord { node }) => write!(
                f,
                "native graph node {node}'s launch condition names no word of an i32 tensor the host writes"
            ),
            Self::Workflow(WorkflowError::UnaddressedCondition { parameter }) => {
                write!(f, "`{parameter}`: a launch condition's word is bound to storage the host does not address")
            }
            Self::Workflow(WorkflowError::ParkRequiresGraph) => {
                f.write_str("an awaited native call requires a graph park on this backend")
            }
            Self::Workflow(WorkflowError::NotPark { node }) => {
                write!(f, "native graph timeline node {node} is bound answered but is no park")
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
            Self::Workflow(WorkflowError::NativeGraphLayoutMismatch(mismatch)) => {
                write!(f, "{mismatch}")
            }
            Self::Workflow(WorkflowError::NativeOutputLeaseConsumed) => {
                f.write_str("native graph output lease was already used")
            }
            Self::Workflow(WorkflowError::NativeExportStillLive) => {
                f.write_str("native graph export is still retained")
            }
            Self::Workflow(WorkflowError::UploadRegionTooSmall { bytes, largest }) => write!(
                f,
                "no upload region of the native graph slot holds the run's {bytes} bytes of input; the largest holds {largest}"
            ),
            Self::Workflow(WorkflowError::UploadBytesOverflow) => {
                f.write_str("tensors uploaded together exceed the upload byte domain")
            }
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
            Self::Workflow(WorkflowError::SpanTimingUnsupported { backend }) => {
                write!(f, "`{}` times no span of a graph run", backend.as_str())
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
    /// and on an `mma16` row axis only whole 16-row tiles; a view of a planar
    /// source (rows sharing their matrix's grid) whole matrices only.
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
    /// Host access to `rows` leading rows of a reserved tensor with only
    /// `committed` backed, or a recommit of a tensor that is not a whole
    /// reserved tensor.
    Uncommitted {
        rows: u64,
        committed: u64,
    },
    /// A commitment of `committed` rows of a reserved tensor of `rows`: it
    /// commits at least one and at most all of them.
    Uncommittable {
        committed: u64,
        rows: u64,
    },
    /// The backend reserves no address range for a reserved tensor under
    /// its contract: none of its kinds would (`why` is `None`), or the one
    /// that would cannot, for `why`.
    Unreservable {
        backend: BackendName,
        why: Option<seismic_topology::Unreservable>,
    },
    /// A view's store charges cannot be read.
    ViewCharge(ViewChargeError),
    /// A typed region of a view's bytes cannot be formed.
    Region(RegionError),
    /// A host-shared word of a view cannot be accessed.
    SharedWord(SharedWordError),
    Execution(ExecutionError),
}

/// Why a 32-bit word of a view cannot be accessed in place by the host
/// (`load_acquire`, `store_release`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SharedWordError {
    /// The view's storage is not one allocation the host addresses in place
    /// (device memory, a reserved or slabbed tensor, a read-only mapping).
    NotHostAddressed,
    /// The word at `offset` reaches past the view's `view_bytes`.
    Outside { offset: u64, view_bytes: u64 },
    /// The word at `offset` is not aligned to its width in host memory.
    Unaligned { offset: u64 },
}

impl fmt::Display for SharedWordError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotHostAddressed => {
                f.write_str("a shared word needs a view of one host-addressed allocation")
            }
            Self::Outside { offset, view_bytes } => write!(
                f,
                "a shared word at {offset} reaches past its view's {view_bytes} bytes"
            ),
            Self::Unaligned { offset } => {
                write!(f, "a shared word at {offset} is not aligned to its width")
            }
        }
    }
}

/// Why a typed region of a view's bytes cannot be formed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegionError {
    /// The view is no one contiguous byte range of an allocation: it is
    /// slabbed or strided.
    NotContiguous,
    /// A region of a reserved tensor would end at `end`, past its backed
    /// `bytes`.
    Unbacked { end: u64, bytes: u64 },
    /// A recommit would back `bytes`, fewer than a live region of the
    /// reserved tensor pins (`pinned`).
    Pinned { pinned: u64, bytes: u64 },
    /// The region's `bytes` from `offset` reach past the view's.
    Outside {
        offset: u64,
        bytes: u64,
        view_bytes: u64,
    },
    /// The region would start at `offset` in its allocation, off its
    /// layout's `alignment`.
    Unaligned { offset: u64, alignment: u64 },
    /// On chunked storage, a `record`-byte record at `offset` would cross a
    /// chunk (or be larger than one): a region starts where its records are
    /// packed.
    Straddles { offset: u64, record: u64 },
    /// On chunked storage, a region at `offset` whose bytes are no leading
    /// rows each in bytes of its own: chunked storage holds only records.
    Unrecorded { offset: u64 },
}

impl fmt::Display for RegionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotContiguous => {
                f.write_str("a region needs a contiguous view of an allocation")
            }
            Self::Unbacked { end, bytes } => write!(
                f,
                "a region of a reserved tensor ending at byte {end} reaches past its {bytes} backed bytes"
            ),
            Self::Pinned { pinned, bytes } => write!(
                f,
                "a live region of a reserved tensor pins its first {pinned} bytes; a recommit cannot back {bytes}"
            ),
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
            Self::Straddles { offset, record } => write!(
                f,
                "a {record}-byte record at allocation offset {offset} would cross a chunk"
            ),
            Self::Unrecorded { offset } => write!(
                f,
                "a region at allocation offset {offset} of chunked storage holds no records"
            ),
        }
    }
}

/// Why a view's store charges cannot be read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewChargeError {
    /// The view is not contiguous, so it covers no one byte range.
    NotContiguous,
    /// The view reaches past what its allocation is charged for.
    BeyondCharge,
}

impl fmt::Display for ViewChargeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::NotContiguous => "store charges require a contiguous tensor view",
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
                "`{representation}` slice {start}..{end} does not select whole rows, row tiles or matrices"
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
            Self::Uncommittable { committed, rows } => write!(
                f,
                "a reserved tensor of {rows} rows commits between 1 and {rows} of them, not {committed}"
            ),
            Self::Unreservable { backend, why: None } => {
                write!(f, "{backend:?} reserves no address range for a reserved tensor")
            }
            Self::Unreservable { backend, why: Some(why) } => {
                write!(f, "{backend:?} cannot reserve an address range for a reserved tensor: {why}")
            }
            Self::ViewCharge(error) => write!(f, "{error}"),
            Self::Region(error) => write!(f, "{error}"),
            Self::SharedWord(error) => write!(f, "{error}"),
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
    use crate::devices::{BackendHeld, DeviceLedgerView, Ledger, Ledgers, LimitError, PathLedgerView, StoreLedgerView};
    use crate::driver::Allocation;
    use seismic_compiler::errors::ExecutionError;
    use seismic_topology::{Contract, DeviceTopology, Executor, Placing, StoreId};
    use std::sync::Arc;

    /// The right to bound every ledger an opened device reaches
    /// (`DeviceInner::bounds`), released as it drops.
    pub struct Bounds {
        memory: Arc<crate::memory::DeviceLedgers>,
        topology: Arc<DeviceTopology>,
        _bounder: Arc<crate::memory::Bounder>,
    }

    impl Bounds {
        /// Hold `ledger` to `bound`. A lower bound refuses growth; it never
        /// revokes what is charged. A store's ledger lives only while an
        /// open device reaches it, so only a store this device reaches can
        /// be bound.
        pub fn bound(&self, ledger: Ledger, bound: Option<u64>) -> Result<(), LimitError> {
            match ledger {
                Ledger::Path(path) => {
                    self.memory.path(path).ok_or(LimitError::ForeignPath(path))?.share.account().set_bound(bound)
                }
                Ledger::Own(store) => {
                    self.memory.store(store).ok_or(LimitError::ForeignStore(store))?.set_bound(bound)
                }
                Ledger::Store(store) => {
                    self.topology.store(store).ok_or(LimitError::StaleStore(store))?;
                    self.memory.store(store).ok_or(LimitError::ForeignStore(store))?.store().set_bound(bound)
                }
            }
            Ok(())
        }

        /// Hold store `store`'s bound to `source` of the store's claim,
        /// read at each admission against the store from now on, in place of
        /// any value set: its holder's room as the holder stands at the
        /// admission, never as it stood at a reading.
        pub fn bound_by(
            &self,
            store: StoreId,
            source: impl Fn(u64) -> Option<u64> + Send + Sync + 'static,
        ) -> Result<(), LimitError> {
            self.topology.store(store).ok_or(LimitError::StaleStore(store))?;
            self.memory.store(store).ok_or(LimitError::ForeignStore(store))?.store().set_bound_source(Arc::new(source));
            Ok(())
        }
    }

    /// One opened executor: its immutable description, execution resources
    /// and ledgers.
    pub struct DeviceInner {
        pub(crate) info: Executor,
        pub(crate) topology: Arc<DeviceTopology>,
        pub(crate) memory: Arc<crate::memory::DeviceLedgers>,
        pub(crate) capabilities: std::sync::OnceLock<Vec<String>>,
        pub(crate) kind: OpenedKind,
        /// The device's properties, described once by its backend when it
        /// opened.
        pub(crate) properties: seismic_lang::device::DeviceProperties,
        /// Forms this device's native programs.
        pub(crate) programs: crate::backends::DevicePrograms,
        /// The active submission trace, if any (`native::trace`).
        pub(crate) trace: std::sync::Mutex<Option<Arc<crate::native::trace::TraceSink>>>,
        /// The artifact store this device was opened with; its program
        /// former keeps artifacts there. A device is shared only with callers
        /// asking for the same store.
        pub(crate) artifacts: Option<Arc<dyn crate::artifacts::ArtifactStore>>,
        /// Native submission order and CUDA graph replays.
        pub(crate) native: crate::native::NativeState,
        /// What the device's recorded graphs hold for their census,
        /// measured on opening where the driver takes it.
        pub(crate) graph_cost: seismic_compiler::graph_cost::GraphCost,
        /// Set once the load's charge is final: from then on a record table
        /// formed outside a lifted scope ([`DeviceInner::lift_table_seal`])
        /// is memory no claim covers (see `record_table`).
        pub(crate) tables_sealed: std::sync::atomic::AtomicBool,
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
    thread_local! {
        /// The devices whose table seal this thread has lifted, one entry per
        /// open scope ([`DeviceInner::lift_table_seal`]).
        static LIFTED_SEALS: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    impl DeviceInner {
        /// Declare the load's charge final: forming a record table afterwards,
        /// outside a lifted scope, is an invariant violation.
        pub fn seal_record_tables(&self) {
            self.tables_sealed.store(true, std::sync::atomic::Ordering::SeqCst);
        }
        /// Lift the seal until [`Self::restore_table_seal`], for allocations
        /// made under a claim after the load (a component's import).
        ///
        /// The lift belongs to the calling thread: a table another thread
        /// forms meanwhile is outside any claim and stops there as it does
        /// anywhere ([`record_table`]).
        pub fn lift_table_seal(&self) {
            LIFTED_SEALS.with(|lifted| lifted.borrow_mut().push(self.seal_key()));
        }
        pub fn restore_table_seal(&self) {
            LIFTED_SEALS.with(|lifted| {
                let mut lifted = lifted.borrow_mut();
                let at = lifted.iter().rposition(|key| *key == self.seal_key()).expect("a restored seal lift is this thread's");
                lifted.remove(at);
            });
        }
        fn seal_key(&self) -> usize {
            std::ptr::from_ref(self) as usize
        }
        pub(crate) fn seal_lifted_here(&self) -> bool {
            LIFTED_SEALS.with(|lifted| lifted.borrow().contains(&self.seal_key()))
        }
        pub(crate) fn active_trace(&self) -> Option<Arc<crate::native::trace::TraceSink>> {
            self.trace
                .lock()
                .expect("trace slot lock is never poisoned")
                .clone()
        }
        pub fn info(&self) -> &Executor {
            &self.info
        }
        /// The device's properties (`seismic_lang::device`).
        pub fn properties(&self) -> &seismic_lang::device::DeviceProperties {
            &self.properties
        }
        #[cfg(feature = "machine-line")]
        /// Whether a Vulkan device forms the cooperative-matrix path
        /// (`FormationTarget::matrix`: supported and chosen), which a
        /// matrix program is empty without; `None` on every other backend.
        pub fn forms_matrix_path(&self) -> Option<bool> {
            match &self.kind {
                #[cfg(not(target_os = "macos"))]
                OpenedKind::Vulkan(opened) => Some(opened.service().formation().matrix),
                _ => None,
            }
        }
        /// What this device's graphs cost its submissions so far: graphs
        /// recorded (a submission during which that count grew recorded one
        /// on the host, so its duration is not its launches' alone) and
        /// replays that updated a graph's parameters first.
        pub fn graph_replays(&self) -> crate::native::GraphReplays {
            self.native.graph_replays()
        }
        /// What the graphs the device keeps may be made of: what every
        /// live graph slot's runs may record.
        pub fn graph_budget(&self) -> seismic_compiler::graph_cost::GraphCensus {
            self.native.graph_budget()
        }
        /// What the device's recorded graphs hold for their census.
        pub fn graph_cost(&self) -> seismic_compiler::graph_cost::GraphCost {
            self.graph_cost
        }
        /// The topology revision this device was opened in; its identifiers
        /// are the ones this device's charges and limits name.
        pub fn topology(&self) -> &Arc<DeviceTopology> {
            &self.topology
        }
        /// Whether this device's toolchain keeps what it compiles in the
        /// platform's own cache (`seismic_native_target::Load::PlatformCached`),
        /// which the process's normal exit leaves written.
        pub fn platform_cached(&self) -> bool {
            self.programs.platform_cached()
        }
        /// The host threads this device's own execution keeps busy while
        /// its work runs: a CPU device's worker pool's participants, the
        /// submitting thread among them; an accelerator's submitting thread
        /// alone.
        pub fn host_threads(&self) -> usize {
            match &self.kind {
                OpenedKind::Cpu(opened) => opened.executor().workers(),
                #[allow(unreachable_patterns)]
                _ => 1,
            }
        }
        /// The host memory node the device is attached to
        /// (`seismic_discovery::DeviceNode`): a CUDA device's, from its PCI
        /// location; none a device without a bus location reports (a CPU
        /// device, Metal, Vulkan).
        pub fn host_node(&self) -> Result<seismic_discovery::DeviceNode, String> {
            match &self.kind {
                OpenedKind::Cuda(opened) => {
                    let [domain, bus, device] = seismic_cuda::describe(opened.executor().device().ordinal()).map_err(|error| error.to_string())?.pci;
                    seismic_discovery::device_node(seismic_discovery::PciLocation { domain, bus, device })
                }
                #[allow(unreachable_patterns)]
                _ => Ok(seismic_discovery::DeviceNode::Unknown),
            }
        }
        /// The native programs this device has formed, and the time it took.
        pub fn programs_formed(&self) -> crate::formation::FormedPrograms {
            self.programs.programs_formed()
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
        /// `OpenedKind::allocate_mapped`.
        pub(crate) fn allocate_mapped(
            self: &Arc<Self>,
            contract: Contract,
            bytes: u64,
            alignment: u64,
        ) -> Result<Arc<Allocation>, ExecutionError> {
            self.kind.allocate_mapped(contract, bytes, alignment)
        }
        /// Storage that reads zero, backed; see `OpenedKind::allocate_zeroed`.
        pub(crate) fn allocate_zeroed(
            self: &Arc<Self>,
            bytes: u64,
            alignment: u64,
        ) -> Result<Arc<Allocation>, ExecutionError> {
            self.kind.allocate_zeroed(bytes, alignment)
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
        /// Upload storage that reads zero, backed; see
        /// `OpenedKind::allocate_mapped_zeroed`.
        pub(crate) fn allocate_mapped_zeroed(
            self: &Arc<Self>,
            contract: Contract,
            bytes: u64,
            alignment: u64,
        ) -> Result<Arc<Allocation>, seismic_compiler::errors::ExecutionError> {
            self.kind.allocate_mapped_zeroed(contract, bytes, alignment)
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
        /// Zero-filled storage of a reserved tensor of `rows` leading rows of
        /// `row` bytes, `committed` of them backed, as the device reserves
        /// (its executor's limits); see `OpenedKind::allocate_reserved_tensor`.
        pub(crate) fn allocate_reserved_tensor(
            self: &Arc<Self>,
            contract: seismic_topology::ReservedContract,
            (records, rows, committed): (seismic_topology::ReservedRecords, u64, u64),
            alignment: u64,
        ) -> Result<crate::driver::Resizer, super::TensorError> {
            let unreservable = |why| super::TensorError::Unreservable {
                backend: self.info().backend(),
                why,
            };
            let span = self
                .info()
                .reserving(contract)
                .map_err(|why| unreservable(Some(why.clone())))?
                .ok_or_else(|| unreservable(None))?;
            self.kind.allocate_reserved_tensor(
                crate::driver::ReservedRange::new(contract, span, records, rows),
                committed,
                alignment,
            )
        }
        pub(crate) fn allocation_address(&self, allocation: &Arc<Allocation>, offset: u64) -> u64 {
            self.kind.allocation_address(allocation, offset)
        }
        /// What decides how a configuration measures on this device: the
        /// device, its backend's identity, the toolchain that forms its
        /// programs (a compiler, target or option change forms other
        /// programs from the same source), and the bytes a cold pass of its
        /// timing reads where it times cold passes
        /// (`native::point::cold_bytes`).
        pub fn tuning_identity(&self) -> String {
            let mut identity = format!("{};{}", self.info.name(), self.kind.tuning_identity());
            if let Some(toolchain) = self.programs.toolchain() {
                identity = format!("{identity};toolchain {toolchain}");
            }
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
        /// Whether the process's footprint (`ProcessMemory`) holds this
        /// device's allocations. CPU allocations are ordinary process memory;
        /// Metal resources enter the task's physical-footprint ledger; a CUDA
        /// device whose memory is the host's has its driver's count of the
        /// process in the footprint. Other CUDA and Vulkan driver allocations
        /// are accounted separately from it.
        pub fn allocations_in_process_footprint(&self) -> bool {
            match &self.kind {
                OpenedKind::Cpu(_) => true,
                #[cfg(target_os = "macos")]
                OpenedKind::Metal(_) => true,
                OpenedKind::Cuda(opened) => opened.service().allocations_in_process_footprint(),
                #[cfg(not(target_os = "macos"))]
                OpenedKind::Vulkan(_) => false,
            }
        }
        /// The ledgers this device reaches: each store's, its paths'
        /// accounts, and its own on each store.
        pub fn ledgers(&self) -> Ledgers {
            let memory = &self.memory;
            let mut stores = Vec::new();
            for (store, ledger) in &memory.stores {
                let ledger = ledger.store();
                stores.push(StoreLedgerView { store: *store, claim: ledger.claim(), bound: ledger.bound() });
            }
            Ledgers {
                stores,
                paths: memory
                    .paths
                    .iter()
                    .map(|path| PathLedgerView {
                        path: path.id,
                        claim: path.share.account().claim(),
                        bound: path.share.account().bound(),
                    })
                    .collect(),
                own: memory.stores.iter().map(|(store, ledger)| (*store, ledger.view())).collect(),
            }
        }
        /// The store device storage is placed on first: whose ledger is the
        /// device's charge, on a device whose memory is the host's every
        /// charge it makes in that memory.
        pub fn device_store(&self) -> StoreId {
            let path = self
                .info
                .paths(Placing::Fixed(Contract::DeviceOnly))
                .first()
                .expect("an opened executor places device storage");
            self.topology.access_path(*path).expect("a placement's path belongs to its topology").store()
        }
        /// The device's own ledger on its device store
        /// ([`Self::device_store`]).
        pub fn device_ledger(&self) -> DeviceLedgerView {
            self.memory.device_ledger().view()
        }
        /// The right to bound every ledger the device reaches: its paths'
        /// accounts, its own ledgers, and each store's ledger, every
        /// executor's of its scope that reaches the store. One live holder
        /// per ledger; taking it while another holder lives panics, naming
        /// both.
        #[track_caller]
        pub fn bounds(&self) -> Bounds {
            let bounder = Arc::new(crate::memory::Bounder {
                selector: self.info.selector(),
                taken: std::panic::Location::caller(),
            });
            for path in &self.memory.paths {
                path.share.account().right.take(&bounder);
            }
            for (_, ledger) in &self.memory.stores {
                ledger.right.take(&bounder);
                ledger.store().right.take(&bounder);
            }
            Bounds { memory: self.memory.clone(), topology: self.topology.clone(), _bounder: bounder }
        }
        /// Start the high-water mark of each of the device's own ledgers
        /// again from what it charges now, and forget its refusals.
        pub fn restart_high_water(&self) {
            for (_, ledger) in &self.memory.stores {
                ledger.restart_high_water();
            }
        }
        /// What `kind` storage can still take on this device: what its
        /// placement's paths, every level of the ledger above them, leave
        /// to both its allocations' charges and what the backend holds
        /// natively. An allocation needing more is refused as a capacity
        /// shortfall.
        pub fn room(&self, contract: Contract) -> u64 {
            let placement = self.memory.placement(Placing::Fixed(contract));
            placement.available().min(placement.physical().available())
        }
        /// What the backend holds for `kind` storage beyond its
        /// allocations: a fact of the backend.
        pub fn backend_held(&self, kind: Contract) -> BackendHeld {
            self.kind.backend_held(kind)
        }
    }
}

pub mod tensor {
    use super::{
        device::DeviceInner, HostRegion, RegionError, SharedWordError, TensorError,
        ViewChargeError,
    };
    use crate::driver::{Allocation, RegionPin};
    use crate::layout;
    use seismic_compiler::errors::ExecutionError;
    use seismic_compiler::prepared::TensorDescriptor;
    use seismic_lang::ids::RepresentationId;
    use seismic_lang::registry::representation_info;
    use seismic_topology::Contract;
    use std::sync::{
        atomic::{fence, AtomicU32, AtomicU64, AtomicUsize, Ordering},
        Arc, OnceLock, Weak,
    };

    pub use crate::driver::{PreparedRecommit, RecommitRefusal, RecommitRelease, Resizer};

    /// A non-owning observation of one physical tensor allocation. It keeps
    /// no storage alive and reads the same charge as the device ledger.
    pub struct TensorStorageObserver {
        allocation: Weak<Allocation>,
        identity: u64,
        topology: Arc<seismic_topology::DeviceTopology>,
        /// The accounting of the executor the allocation was made through.
        memory: Arc<crate::memory::DeviceLedgers>,
    }

    impl TensorStorageObserver {
        /// The bytes the allocation charges to each store, while it lives.
        pub fn store_charges(&self) -> Option<Vec<(seismic_topology::StoreId, u64)>> {
            let paths = self.path_charges()?;
            Some(
                crate::memory::store_charges(&self.topology, paths)
                    .into_iter()
                    .collect(),
            )
        }
        /// The bytes the allocation charges along each access path, while
        /// it lives.
        pub fn path_charges(&self) -> Option<Vec<(seismic_topology::AccessPathId, u64)>> {
            Some(self.allocation.upgrade()?.path_charges(&self.memory))
        }
        pub fn identity(&self) -> u64 {
            self.identity
        }

        pub fn charged_bytes(&self) -> Option<u64> {
            self.allocation
                .upgrade()
                .map(|allocation| allocation.charged_bytes())
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
        slabs: Option<SlabView>,
        alias_identity: Option<u64>,
        slab_lease: Option<Arc<AtomicUsize>>,
        /// A region of a reserved tensor's backed bytes, and every view of
        /// it, holds them backed ([`TensorInner::region`]).
        pin: Option<Arc<RegionPin>>,
        /// This view's records as a native record port binds them, formed
        /// at its first such binding ([`TensorInner::records`]).
        records: OnceLock<Arc<Records>>,
    }

    /// A view's records as a native record port reaches them
    /// ([`TensorInner::records`]): through a table of their device
    /// addresses, in the storage they lie in.
    pub(crate) struct Records {
        /// The table's allocation, and the byte offset of the view's entry 0.
        pub(crate) table: Arc<Allocation>,
        pub(crate) table_offset: u64,
        /// The view's allocation, which the encodings order by
        /// (`native::hazard`), its records, and where they lie in it.
        pub(crate) storage: Arc<Allocation>,
        pub(crate) count: u64,
        pub(crate) placement: RecordPlacement,
    }

    impl Records {
        /// Where records `[first, first + count)` lie: their bytes' offset
        /// in the storage, and the bytes they span (the encodings order by
        /// them).
        pub(crate) fn span(&self, first: u64, count: u64) -> (u64, u64) {
            let record = self.placement.record();
            self.placement.reach(first * record, count * record)
        }
    }

    /// Where a view's records lie in its storage: records of a nonzero
    /// size from the view's first byte, packed as its storage packs them
    /// ([`seismic_topology::Packing`]). Only a view with records has one
    /// ([`TensorInner::record_placement`]).
    #[derive(Clone, Copy, Debug)]
    pub struct RecordPlacement {
        start: u64,
        packing: seismic_topology::Packing,
    }

    impl RecordPlacement {
        /// The bytes of one record.
        pub fn record(&self) -> u64 {
            self.packing.record()
        }

        /// Where record `index` lies, in bytes from the storage's first.
        fn at(&self, index: u64) -> u64 {
            self.packing.at(self.start, index)
        }

        /// Where record `index` lies, in bytes from the view's first.
        pub fn offset(&self, index: u64) -> u64 {
            self.at(index) - self.start
        }

        /// The byte of the storage just past the view's first `records`
        /// records.
        fn end(&self, records: u64) -> u64 {
            self.packing.end(self.start, records)
        }

        /// Where the view's canonical bytes `[offset, offset + bytes)` lie,
        /// and the bytes they span: on chunked storage the records covering
        /// them, as packed; otherwise those bytes.
        pub(crate) fn reach(&self, offset: u64, bytes: u64) -> (u64, u64) {
            match self.packing.chunked() && bytes != 0 {
                true => {
                    let at = self.at(offset / self.record());
                    (at, self.end((offset + bytes).div_ceil(self.record())) - at)
                }
                false => (self.start + offset, bytes),
            }
        }
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

    /// A record table over `storage` on `device`, entry `i` the device
    /// address of its `i`th byte offset in `offsets`
    /// ([`seismic_lang::registry::record_table`]), in `storage`'s hazard
    /// domain: the one place a record table is formed.
    pub(crate) fn record_table(
        device: &Arc<DeviceInner>,
        storage: &Arc<Allocation>,
        offsets: impl Iterator<Item = u64>,
    ) -> Result<Arc<Allocation>, TensorError> {
        // A table formed after the load is sealed is device memory no claim
        // covers: a state that must not exist, so every build stops where it
        // is formed (every fixed record port's table is formed as its plan is
        // bound, `bind_static`, and a grown slab's is in its allocation's
        // charge).
        assert!(
            !device.tables_sealed.load(std::sync::atomic::Ordering::SeqCst)
                || device.seal_lifted_here(),
            "invariant violated: a record table over {} bytes of storage was formed after the load, outside any claim: {}",
            storage.bytes(),
            std::backtrace::Backtrace::force_capture()
        );
        let bytes = seismic_lang::registry::record_table(
            offsets.map(|offset| device.allocation_address(storage, offset)),
        );
        let table = device.kind.allocate_in(
            bytes.len() as u64,
            seismic_lang::registry::RECORD_TABLE_ENTRY_BYTES,
            Some(storage.access_group().clone()),
        )?;
        table.storage().write(0, &bytes)?;
        Ok(table)
    }

    /// A tensor over ordinary storage, never a reservation's: its device
    /// address is fixed for its life (a reserved tensor's can change when it
    /// is recommitted, on Metal). Only ordinary storage and views of it form
    /// one.
    #[derive(Clone)]
    pub struct Ordinary(Arc<TensorInner>);

    impl Ordinary {
        /// New zeroed storage ordered by the same hazard domain as `other`.
        pub fn zeros_sharing_access(
            other: &Self,
            representation: RepresentationId,
            extents: &[u64],
        ) -> Result<Self, TensorError> {
            TensorInner::canonical_in(&other.device, representation, extents, |device, bytes, alignment| {
                let allocation = device.kind.allocate_in(bytes, alignment, Some(other.allocation.access_group().clone()))?;
                device.kind.zeroed(allocation, bytes)
            }).map(|tensor| Self(Arc::new(tensor)))
        }

        /// New initialized storage ordered by the same hazard domain as `other`.
        pub fn from_host_sharing_access(
            other: &Self,
            representation: RepresentationId,
            extents: &[u64],
            bytes: &[u8],
        ) -> Result<Self, TensorError> {
            TensorInner::from_host_in(&other.device, representation, extents, bytes, Some(other.allocation.access_group().clone()))
                .map(|tensor| Self(Arc::new(tensor)))
        }

        /// `tensor`, unless it views a reservation.
        pub fn of(tensor: Arc<TensorInner>) -> Option<Self> {
            tensor.allocation.reservation().is_none().then_some(Self(tensor))
        }

        /// [`TensorInner::zeros`].
        pub fn zeros(
            device: &Arc<DeviceInner>,
            representation: RepresentationId,
            extents: &[u64],
        ) -> Result<Self, TensorError> {
            TensorInner::zeros(device, representation, extents).map(|tensor| Self(Arc::new(tensor)))
        }

        /// [`TensorInner::from_host`].
        pub fn from_host(
            device: &Arc<DeviceInner>,
            representation: RepresentationId,
            extents: &[u64],
            bytes: &[u8],
        ) -> Result<Self, TensorError> {
            TensorInner::from_host(device, representation, extents, bytes)
                .map(|tensor| Self(Arc::new(tensor)))
        }

        /// [`TensorInner::region`] of this tensor.
        pub fn region(
            &self,
            representation: RepresentationId,
            extents: &[u64],
            offset: u64,
        ) -> Result<Self, TensorError> {
            self.0
                .region(representation, extents, offset)
                .map(|tensor| Self(Arc::new(tensor)))
        }

        /// [`TensorInner::slice_leading`] of this tensor.
        pub fn slice_leading(&self, start: u64, end: u64) -> Result<Self, TensorError> {
            self.0.slice_leading(start, end).map(|tensor| Self(Arc::new(tensor)))
        }

        /// [`TensorInner::reshape`] of this tensor.
        pub fn reshape(&self, extents: &[u64]) -> Result<Self, TensorError> {
            self.0.reshape(extents).map(|tensor| Self(Arc::new(tensor)))
        }

        pub fn tensor(&self) -> &Arc<TensorInner> {
            &self.0
        }
    }

    impl std::ops::Deref for Ordinary {
        type Target = TensorInner;

        fn deref(&self) -> &TensorInner {
            &self.0
        }
    }

    impl TensorInner {
        /// A logical tensor whose native binding is a slab address table.
        /// Each present region view retains its slab allocation; a launch
        /// snapshots this list and tracks every allocation it can address.
        /// The table holds its regions' device addresses, fixed for their
        /// lives: table and regions are ordinary tensors.
        pub fn slabbed(
            table: &Ordinary,
            representation: RepresentationId,
            rows_per_slab: u64,
            logical_rows: u64,
            alias_identity: u64,
            regions: Vec<Option<Ordinary>>,
            lease: Arc<AtomicUsize>,
        ) -> Result<Self, TensorError> {
            if regions.is_empty() || rows_per_slab == 0 {
                return Err(TensorError::SlabLayout("empty slab tensor".into()));
            }
            if table.representation
                != seismic_lang::registry::dense(seismic_lang::types::DType::U32)
                || table.extents != [regions.len() as u64, 2]
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
                slabs: Some(SlabView {
                    rows_per_slab,
                    regions: regions
                        .into_iter()
                        .map(|region| region.map(|region| region.0))
                        .collect(),
                }),
                alias_identity: Some(alias_identity),
                slab_lease: Some(lease),
                pin: None,
                records: OnceLock::new(),
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

        /// Whether the tensor is a slab view: its binding is an address table,
        /// not its elements.
        pub(crate) fn is_slab_view(&self) -> bool {
            self.slabs.is_some()
        }

        /// Whether the tensor may bind a slab port (`&slab`,
        /// [`seismic_lang::entry::PortBinding::Slab`]): a slab view, or a
        /// tensor of no elements, which has no row to address.
        pub(crate) fn binds_slab_port(&self) -> bool {
            self.is_slab_view() || self.extents.contains(&0)
        }

        /// Whether a call bound to this view must hold it until its
        /// submission records its fences: a slab view's placement, or a
        /// pinned region of reserved storage. From then on the allocation
        /// fences defer any physical release to the work's completion.
        pub(crate) fn holds_binding_lease(&self) -> bool {
            self.slabs.is_some() || self.pin.is_some()
        }

        /// What device work bound to a slab view reaches: its address-table
        /// entries (the binding, `[slabs, 2]` u32 at the view's offset) and
        /// each backed slab region; `None` for any other view.
        pub(crate) fn slab_reach(&self) -> Option<crate::native::hazard::SlabReach> {
            self.slabs
                .as_ref()
                .map(|slabs| crate::native::hazard::SlabReach {
                    table_bytes: slabs.regions.len() as u64 * 8,
                    regions: slabs
                        .regions
                        .iter()
                        .flatten()
                        .map(|region| {
                            (region.allocation.clone(), region.byte_offset, region.byte_len)
                        })
                        .collect(),
                })
        }

        /// The device bytes the tensor's elements occupy: a slab view's
        /// backed slabs, a reserved tensor's committed rows, otherwise its
        /// byte length.
        pub(crate) fn occupied_bytes(&self) -> u64 {
            match &self.slabs {
                Some(slabs) => slabs.regions.iter().flatten().map(|r| r.byte_len).sum(),
                None => self
                    .byte_len
                    .min(self.allocation.bytes().saturating_sub(self.byte_offset)),
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
            if self.reaches_uncommitted() {
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
        /// A copy of a view the host addresses in place ([`Self::shared`])
        /// is host-shared too, so the host reads it as it reads this one (a
        /// launch condition's word, `native::condition_holds`).
        pub(crate) fn copy_of(&self, image: &TensorImage) -> Result<Self, TensorError> {
            let device = &self.device;
            match (image, &self.slabs) {
                (TensorImage::Whole(bytes), None) if !self.reaches_uncommitted() => {
                    match self.host_address() {
                        Some(_) => {
                            let copy = Self::shared(device, self.representation, &self.extents)?;
                            copy.write_from_host(bytes)?;
                            Ok(copy)
                        }
                        None => Self::from_host(device, self.representation, &self.extents, bytes),
                    }
                }
                (TensorImage::Committed(bytes), None) if self.reaches_uncommitted() => {
                    let committed = self.committed_rows();
                    // The copy keeps its rows, never resized, in storage
                    // of the kind the call reads this tensor's from.
                    let reservation = self
                        .allocation
                        .reservation()
                        .expect("a view reaching past its backed bytes is of a reservation");
                    let (copy, _) = Self::reserved(
                        device,
                        reservation.contract,
                        self.representation,
                        &self.extents,
                        committed,
                        reservation.carries,
                    )?;
                    copy.slice_leading(0, committed)?.write_from_host(bytes)?;
                    Ok(copy)
                }
                (TensorImage::Slabs(images), Some(slabs))
                    if images.len() == slabs.regions.len() =>
                {
                    let dense_u32 = seismic_lang::registry::dense(seismic_lang::types::DType::U32);
                    let table =
                        Ordinary::zeros(device, dense_u32, &[slabs.regions.len() as u64, 2])?;
                    let regions = slabs
                        .regions
                        .iter()
                        .zip(images)
                        .map(|(region, bytes)| match (region, bytes) {
                            (Some(region), Some(bytes)) => {
                                let copy = Ordinary::from_host_sharing_access(
                                    &table,
                                    region.representation,
                                    &region.extents,
                                    bytes,
                                )?;
                                Ok(Some(copy))
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
        /// contiguous byte range of an allocation (not slabbed), and the
        /// region must lie within it, starting at a multiple of its layout's
        /// alignment in the allocation. A region of a reserved tensor lies
        /// within its backed bytes and pins them: while any view of the
        /// region lives, a recommit backs no fewer. On chunked storage its
        /// records are packed, none crossing a chunk: it starts where its
        /// first record fits in one chunk (a packed start, such as the end
        /// of an earlier packed view) and spans its records as they are
        /// packed ([`TensorInner::record_placement`]).
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
            if self.slabs.is_some()
                || self.packed()
                || layout::canonical(self.representation, &self.extents)?.strides != self.strides
            {
                return Err(TensorError::Region(RegionError::NotContiguous));
            }
            let layout = layout::canonical(representation, extents)?;
            let outside = |bytes| {
                TensorError::Region(RegionError::Outside {
                    offset,
                    bytes,
                    view_bytes: self.byte_len,
                })
            };
            let start = self.byte_offset.checked_add(offset).ok_or(outside(layout.byte_len))?;
            // On chunked storage the region's records are packed from an
            // already packed start ([`seismic_topology::Packing`]).
            let packed = self.allocation.reservation().map(|reservation| reservation.span);
            let span = match packed {
                Some(span) if span.chunked() && layout.byte_len != 0 => {
                    let record = seismic_lang::registry::record_bytes(representation, extents)
                        .ok_or(TensorError::Region(RegionError::Unrecorded { offset: start }))?
                        .get();
                    let packing = span
                        .packing(record)
                        .filter(|packing| packing.at(start, 0) == start)
                        .ok_or(TensorError::Region(RegionError::Straddles {
                            offset: start,
                            record,
                        }))?;
                    packing.end(start, extents[0]) - start
                }
                _ => layout.byte_len,
            };
            if offset.checked_add(span).is_none_or(|end| end > self.byte_len) {
                return Err(outside(span));
            }
            if start % layout.alignment != 0 {
                return Err(TensorError::Region(RegionError::Unaligned {
                    offset: start,
                    alignment: layout.alignment,
                }));
            }
            let pin = match self.allocation.reservation() {
                Some(_) => Some(Arc::new(
                    self.allocation
                        .pin(start + span)
                        .ok_or(TensorError::Region(RegionError::Unbacked {
                            end: start + span,
                            bytes: self.allocation.bytes(),
                        }))?,
                )),
                None => None,
            };
            let mut region = Self::new_view(
                self.device.clone(),
                self.allocation.clone(),
                start,
                layout.byte_len,
                representation,
                extents.to_vec(),
                layout.strides,
            );
            region.pin = pin;
            Ok(region)
        }

        /// Records `start..end` of this view (its leading-axis rows, where
        /// [`Self::record_placement`] lays them) as a region of their own:
        /// the view a [`Self::region`] of its storage at the first of them
        /// forms, which pins a reservation's backed bytes only through the
        /// last of them, where any view derived from this one pins what this
        /// one does. So a holder of a few records of a reserved tensor never
        /// holds a recommit above them.
        pub fn record_region(&self, start: u64, end: u64) -> Result<Self, TensorError> {
            let placement = self
                .record_placement()
                .ok_or(TensorError::Region(RegionError::Unrecorded { offset: self.byte_offset }))?;
            if self.slabs.is_some()
                || layout::canonical(self.representation, &self.extents)?.strides != self.strides
            {
                return Err(TensorError::Region(RegionError::NotContiguous));
            }
            let extent = self.extents[0];
            if start >= end || end > extent {
                return Err(TensorError::SliceOutOfBounds { extent, start, end });
            }
            let extents = std::iter::once(end - start)
                .chain(self.extents[1..].iter().copied())
                .collect::<Vec<_>>();
            let layout = layout::canonical(self.representation, &extents)?;
            let pin = match self.allocation.reservation() {
                Some(_) => Some(Arc::new(self.allocation.pin(placement.end(end)).ok_or(
                    TensorError::Region(RegionError::Unbacked {
                        end: placement.end(end),
                        bytes: self.allocation.bytes(),
                    }),
                )?)),
                None => None,
            };
            let mut region = Self::new_view(
                self.device.clone(),
                self.allocation.clone(),
                placement.at(start),
                layout.byte_len,
                self.representation,
                extents,
                layout.strides,
            );
            region.pin = pin;
            Ok(region)
        }

        pub fn device_address(&self) -> u64 {
            self.device.allocation_address(&self.allocation, self.byte_offset)
        }

        /// Where this canonical view's records (leading-axis rows) lie: on
        /// chunked reserved storage packed from the view's first byte, none
        /// crossing a chunk ([`TensorInner::region`]), otherwise one after
        /// another. None for a view without records of their own
        /// ([`record_bytes`]), or whose records no chunk of its storage
        /// holds whole ([`seismic_topology::Reservation::packing`]).
        pub fn record_placement(&self) -> Option<RecordPlacement> {
            let record = seismic_lang::registry::record_bytes(self.representation, &self.extents)?;
            let packing = match self.allocation.reservation() {
                Some(reservation) => reservation.span.packing(record.get())?,
                None => seismic_topology::Packing::contiguous(record),
            };
            Some(RecordPlacement {
                start: self.byte_offset,
                packing,
            })
        }

        /// The bytes from its first that this view reaches: its records as
        /// packed on chunked storage ([`RecordPlacement::reach`]),
        /// otherwise its bytes.
        pub(crate) fn reached_bytes(&self) -> u64 {
            self.record_placement()
                .map_or(self.byte_len, |placement| placement.reach(0, self.byte_len).1)
        }

        /// Whether this view lies in chunked reserved storage, which device
        /// work reaches only through native record ports (or, never bound,
        /// host ports).
        pub(crate) fn chunked(&self) -> bool {
            self.allocation
                .reservation()
                .is_some_and(|reservation| reservation.span.chunked())
        }

        /// Whether this view's records are packed in chunks with room left
        /// between them: its bytes are no one canonical range.
        fn packed(&self) -> bool {
            self.record_placement().is_some_and(|placement| {
                placement.end(self.extents[0]) != self.byte_offset + self.byte_len
            })
        }

        /// This canonical view's records as a native record port binds them.
        /// It binds a table its storage holds: a view of a reservation's own
        /// records the reservation's table (its entries kept as chunks
        /// change); a view of a fixed allocation the allocation's table of
        /// records of its size and phase, or for a single record any table
        /// the allocation holds with an entry at its address
        /// (`Allocation::view_table`), formed at the first such binding and
        /// charged with the storage. A view
        /// formed per run binds the same table and allocates nothing. A
        /// pinned view of other records of reserved storage forms a table
        /// of its own, kept with the view (its records stay at those
        /// addresses only while its pin holds them) and charged with the
        /// storage while it lives (`Allocation::hold_table`); an unpinned
        /// one is refused. [`Self::hold_records`] forms the table before
        /// the first binding.
        ///
        /// Device work reaches the records by address alone, so they must
        /// lie in storage the device holds resident (on Metal, its
        /// residency set): a read-only host mapping, which the device makes
        /// resident only where it is bound, is refused.
        pub(crate) fn records(&self) -> Result<Arc<Records>, TensorError> {
            match self.records.get() {
                Some(records) => Ok(records.clone()),
                None => {
                    let placement = self
                        .record_placement()
                        .ok_or(TensorError::Region(RegionError::NotContiguous))?;
                    let (table, table_offset) = self.form_record_table(placement)?;
                    let records = Records {
                        table,
                        table_offset,
                        storage: self.allocation.clone(),
                        count: self.extents[0],
                        placement,
                    };
                    Ok(self.records.get_or_init(|| Arc::new(records)).clone())
                }
            }
        }

        /// Form now the table a native record port binds this view through
        /// ([`Self::records`]), where its backend binds record tables, so no
        /// later binding allocates: for a view first bound while its
        /// device's storage may refuse an allocation (memory short).
        pub fn hold_records(&self) -> Result<(), TensorError> {
            match self.device.info().backend().binds_record_tables() {
                true => self.records().map(drop),
                false => Ok(()),
            }
        }

        /// The table a record port binds this view's records, at
        /// `placement`, through, and the byte of its first entry
        /// ([`Self::records`]).
        fn form_record_table(
            &self,
            placement: RecordPlacement,
        ) -> Result<(Arc<Allocation>, u64), TensorError> {
            let not_contiguous = TensorError::Region(RegionError::NotContiguous);
            if self.slabs.is_some()
                || layout::canonical(self.representation, &self.extents)?.strides != self.strides
            {
                return Err(not_contiguous);
            }
            if self.allocation.storage().read_only() {
                return Err(TensorError::HostRegion(
                    "a record port's records cannot lie in a read-only host mapping".into(),
                ));
            }
            let (record, records) = (placement.record(), self.extents[0]);
            match (self.allocation.reservation(), self.allocation.record_table()) {
                (Some(reservation), Some(table))
                    if reservation.record.get() == record && self.byte_offset % record == 0 =>
                {
                    let entry = seismic_lang::registry::record_table_entry(self.byte_offset / record);
                    Ok((table.clone(), entry))
                }
                (Some(_), _) if self.pin.is_none() => Err(not_contiguous),
                // Its records' chunks are backed, at those addresses, only
                // while the view's pin holds them: the table is the view's.
                (Some(_), _) => {
                    let table = record_table(
                        &self.device,
                        &self.allocation,
                        (0..records).map(|index| placement.at(index)),
                    )?;
                    self.allocation.hold_table(&table);
                    Ok((table, 0))
                }
                (None, _) => self.allocation.view_table(
                    record,
                    self.byte_offset,
                    records,
                    |phase, records| {
                        record_table(
                            &self.device,
                            &self.allocation,
                            (0..records).map(|index| phase + index * record),
                        )
                    },
                ),
            }
        }

        /// A zero-filled tensor over `extents` whose leading rows
        /// `[0, committed)` are physically backed, and the resizer that
        /// changes them. The remaining rows are an address reservation:
        /// binding the tensor is legal only for device work that stays within
        /// the committed rows, and host access to them is refused. Its
        /// storage is of `kind`: device storage, or host storage the host
        /// addresses in place, where its backend reserves it
        /// (`seismic_topology::Limits::reservation`). Its records may carry
        /// views of up to `carries` bytes whole (another form's records
        /// packed in them), which no chunk boundary then crosses either.
        pub fn reserved(
            device: &Arc<DeviceInner>,
            kind: seismic_topology::ReservedContract,
            representation: RepresentationId,
            extents: &[u64],
            committed: u64,
            carries: Option<std::num::NonZeroU64>,
        ) -> Result<(Self, Resizer), TensorError> {
            if !device.supports_representation(representation) {
                return Err(TensorError::UnsupportedRepresentation {
                    backend: device.info().backend(),
                    representation: representation_info(representation).name,
                });
            }
            let layout = layout::canonical(representation, extents)?;
            let row = seismic_lang::registry::record_bytes(representation, extents)
                .ok_or(TensorError::Uncommitted {
                    rows: extents.first().copied().unwrap_or(0),
                    committed: 0,
                })?;
            let resizer =
                device.allocate_reserved_tensor(kind, (seismic_topology::ReservedRecords { record: row, carries }, extents[0], committed), layout.alignment)?;
            let tensor = Self::new_view(
                device.clone(),
                resizer.allocation().clone(),
                0,
                layout.byte_len,
                representation,
                extents.to_vec(),
                layout.strides,
            );
            Ok((tensor, resizer))
        }

        /// Whether this view reaches past its allocation's backed bytes: a
        /// view of a reserved tensor beyond its committed rows.
        fn reaches_uncommitted(&self) -> bool {
            self.byte_offset + self.byte_len > self.allocation.bytes()
        }

        /// Leading rows physically backed: all of them unless this view
        /// reaches past its allocation's committed bytes.
        pub fn committed_rows(&self) -> u64 {
            match (
                self.reaches_uncommitted(),
                seismic_lang::registry::record_bytes(self.representation, &self.extents),
            ) {
                (true, Some(row)) => (self.allocation.bytes().saturating_sub(self.byte_offset)
                    / row.get())
                .min(self.extents[0]),
                _ => self.extents.first().copied().unwrap_or(1),
            }
        }

        /// Refuse host access reaching past the committed rows. A resize's
        /// install changes them only under exclusive access, so a caller
        /// holding a permit of the allocation reads them stable.
        fn backed(&self) -> Result<(), TensorError> {
            // A packed view's records are not its canonical bytes: the host
            // addresses them at their `record_placement`.
            if self.packed() {
                return Err(TensorError::Region(RegionError::NotContiguous));
            }
            if self.reaches_uncommitted() {
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
        /// (`Contract::Shared`).
        pub fn shared(
            device: &Arc<DeviceInner>,
            representation: RepresentationId,
            extents: &[u64],
        ) -> Result<Self, TensorError> {
            Self::host_visible(device, representation, extents, |bytes, alignment| {
                device.allocate_mapped_zeroed(Contract::Shared, bytes, alignment)
            })
        }

        /// A zero-filled tensor of coherent words the host and device work
        /// load and store while either is running (`Contract::Words`).
        pub fn words(
            device: &Arc<DeviceInner>,
            representation: RepresentationId,
            extents: &[u64],
        ) -> Result<Self, TensorError> {
            Self::host_visible(device, representation, extents, |bytes, alignment| {
                device.allocate_mapped_zeroed(Contract::Words, bytes, alignment)
            })
        }

        /// A [`Self::words`] tensor whose device accesses the protocol of
        /// the work that uses it orders across device queues, not their
        /// fences (`Allocation::order_by_protocol`).
        pub fn protocol_words(
            device: &Arc<DeviceInner>,
            representation: RepresentationId,
            extents: &[u64],
        ) -> Result<Self, TensorError> {
            let tensor = Self::words(device, representation, extents)?;
            tensor.allocation().order_by_protocol();
            Ok(tensor)
        }

        /// A zero-filled tensor in storage the host only writes in place
        /// while device work reads it (`Contract::Upload`: a
        /// graph run's inputs' storage, which a large BAR may hold).
        pub fn upload(
            device: &Arc<DeviceInner>,
            representation: RepresentationId,
            extents: &[u64],
        ) -> Result<Self, TensorError> {
            Self::host_visible(device, representation, extents, |bytes, alignment| {
                device.allocate_mapped_zeroed(Contract::Upload, bytes, alignment)
            })
        }

        /// A tensor of `representation` and `extents` in host-visible
        /// storage `allocate` allocates zeroed.
        fn host_visible(
            device: &Arc<DeviceInner>,
            representation: RepresentationId,
            extents: &[u64],
            allocate: impl FnOnce(u64, u64) -> Result<Arc<Allocation>, ExecutionError>,
        ) -> Result<Self, TensorError> {
            if !device.supports_representation(representation) {
                return Err(TensorError::UnsupportedRepresentation {
                    backend: device.info().backend(),
                    representation: representation_info(representation).name,
                });
            }
            let layout = layout::canonical(representation, extents)?;
            let allocation = allocate(layout.byte_len, layout.alignment)?;
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

        /// The host address of this view's first byte, where the host
        /// addresses its storage in place: a view of one host-addressed
        /// allocation (see [`Self::shared`]), not slabbed, and of a
        /// reservation only a region, whose pin keeps its bytes backed (a
        /// reservation's host range never moves). Valid while the view
        /// lives.
        pub fn host_address(&self) -> Option<std::ptr::NonNull<u8>> {
            let unpinned = self.allocation.reservation().is_some() && self.pin.is_none();
            if unpinned || self.slabs.is_some() {
                return None;
            }
            let base = self.allocation.storage().host_address()?;
            // In bounds: a backed view's bytes lie inside its allocation.
            Some(unsafe { base.add(self.byte_offset as usize) })
        }

        /// The word of atomic `A` at `offset` bytes into this view, which the
        /// host addresses in place: a view of one host-addressed allocation
        /// (see [`Self::shared`]), the word aligned to its width and inside
        /// the view.
        fn shared_word<A>(&self, offset: u64) -> Result<&A, TensorError> {
            let width = std::mem::size_of::<A>() as u64;
            if offset.checked_add(width).is_none_or(|end| end > self.byte_len) {
                return Err(TensorError::SharedWord(SharedWordError::Outside {
                    offset,
                    view_bytes: self.byte_len,
                }));
            }
            let base = self
                .host_address()
                .ok_or(TensorError::SharedWord(SharedWordError::NotHostAddressed))?;
            // In bounds: the word lies inside the view.
            let address = unsafe { base.as_ptr().add(offset as usize) };
            if address as usize % std::mem::align_of::<A>() != 0 {
                return Err(TensorError::SharedWord(SharedWordError::Unaligned { offset }));
            }
            // Aligned and live while `self` holds the allocation; the host
            // touches shared words only through atomics (`A` is one of
            // `AtomicU32`, `AtomicU64`, whose layout is its integer's).
            Ok(unsafe { &*address.cast::<A>() })
        }

        /// A 64-bit word's acquire load: one access, as [`Self::load_acquire`]
        /// is of a 32-bit one; the word 8-byte aligned.
        pub fn load_acquire_u64(&self, offset: u64) -> Result<u64, TensorError> {
            Ok(self.shared_word::<AtomicU64>(offset)?.load(Ordering::Acquire))
        }

        /// Publish a 64-bit word as one store, as [`Self::store_release`]
        /// publishes a 32-bit one: no reader assembles a value the host never
        /// held. The word 8-byte aligned.
        pub fn store_release_u64(&self, offset: u64, value: u64) -> Result<(), TensorError> {
            self.shared_word::<AtomicU64>(offset)?.store(value, Ordering::Release);
            fence(Ordering::SeqCst);
            Ok(())
        }

        /// Publish `update` of a 64-bit word as one read-modify-write, where
        /// it returns a value for what the word holds (acquire), as
        /// [`Self::store_release_u64`] publishes a store; none leaves the
        /// word as it is. Whether it published. The word 8-byte aligned.
        pub fn update_release_u64(
            &self,
            offset: u64,
            update: impl FnMut(u64) -> Option<u64>,
        ) -> Result<bool, TensorError> {
            let published =
                self.shared_word::<AtomicU64>(offset)?.fetch_update(Ordering::Release, Ordering::Acquire, update).is_ok();
            fence(Ordering::SeqCst);
            Ok(published)
        }

        /// The cross-agent publication contract of host-shared words: the
        /// device publishes a word with a system-scope release (a store after
        /// a system-scope fence); this acquire load then makes what the
        /// device wrote before it visible to the host. Polling takes no full
        /// fence per load: `store_release` ends with the sequentially
        /// consistent fence that orders a host store before its later loads.
        /// It neither waits for nor excludes submitted device work.
        pub fn load_acquire(&self, offset: u64) -> Result<u32, TensorError> {
            Ok(self.shared_word::<AtomicU32>(offset)?.load(Ordering::Acquire))
        }

        /// Publish a word to device work that may be running: every earlier
        /// host write (`write_shared` or a word store) is visible to a device
        /// load of this word under a system-scope acquire (a system-coherent
        /// load, then a system-scope fence). A sequentially consistent fence
        /// follows, so the store is ordered before every later host load.
        /// It neither waits for nor excludes submitted device work.
        pub fn store_release(&self, offset: u64, value: u32) -> Result<(), TensorError> {
            self.shared_word::<AtomicU32>(offset)?.store(value, Ordering::Release);
            fence(Ordering::SeqCst);
            Ok(())
        }

        /// The view's 32-bit words, as the host stores them in place: words
        /// that order nothing, which a device reading them sees in time.
        pub fn shared_words(&self) -> Result<&[AtomicU32], TensorError> {
            let words = usize::try_from(self.byte_len / 4).expect("a host-addressed view fits the host");
            if words == 0 {
                return Ok(&[]);
            }
            self.shared_word::<AtomicU32>(0)?;
            let base = self.host_address().ok_or(TensorError::SharedWord(SharedWordError::NotHostAddressed))?;
            // Aligned (`shared_word`) and within the view, live while `self`
            // holds the allocation, and the host touches them only through
            // atomics.
            Ok(unsafe { std::slice::from_raw_parts(base.as_ptr().cast::<AtomicU32>(), words) })
        }

        pub fn zeros(
            device: &Arc<DeviceInner>,
            representation: RepresentationId,
            extents: &[u64],
        ) -> Result<Self, TensorError> {
            Self::canonical_in(device, representation, extents, DeviceInner::allocate_zeroed)
        }

        /// Allocate a canonical tensor without writing its storage. The
        /// caller must fully initialize all bytes, including representation
        /// padding, before any host read or device input use.
        pub unsafe fn uninitialized(
            device: &Arc<DeviceInner>,
            representation: RepresentationId,
            extents: &[u64],
        ) -> Result<Self, TensorError> {
            Self::canonical_in(device, representation, extents, DeviceInner::allocate)
        }

        /// A canonical tensor over the storage `allocate` makes for it.
        fn canonical_in(
            device: &Arc<DeviceInner>,
            representation: RepresentationId,
            extents: &[u64],
            allocate: impl FnOnce(&Arc<DeviceInner>, u64, u64) -> Result<Arc<Allocation>, ExecutionError>,
        ) -> Result<Self, TensorError> {
            if !device.supports_representation(representation) {
                return Err(TensorError::UnsupportedRepresentation {
                    backend: device.info().backend(),
                    representation: representation_info(representation).name,
                });
            }
            let layout = layout::canonical(representation, extents)?;
            let allocation = allocate(device, layout.byte_len, layout.alignment)?;
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
            Self::from_host_in(device, representation, extents, bytes, None)
        }

        fn from_host_in(
            device: &Arc<DeviceInner>,
            representation: RepresentationId,
            extents: &[u64],
            bytes: &[u8],
            access: Option<Arc<crate::driver::AccessGroup>>,
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
            let allocation = device.kind.allocate_in(layout.byte_len, layout.alignment, access)?;
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
                end <= allocation.span(),
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
                slabs: None,
                alias_identity: None,
                slab_lease: None,
                pin: None,
                records: OnceLock::new(),
            }
        }
        pub fn read_to_host(&self) -> Result<Vec<u8>, TensorError> {
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
            let access = self.allocation.acquire(false)?;
            self.read_with(&access)
        }
        pub(crate) fn read_with(
            &self,
            access: &crate::driver::AllocationPermit,
        ) -> Result<Vec<u8>, TensorError> {
            self.backed()?;
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
            self.write_whole(bytes, false)
        }

        /// Write `bytes` into storage no device work uses: an upload region
        /// its owner took once the work that last read it completed
        /// (`native::graph`). A use still in flight is an invariant
        /// violation: it panics rather than wait.
        pub(crate) fn write_idle(&self, bytes: &[u8]) -> Result<(), TensorError> {
            self.write_whole(bytes, true)
        }

        /// Write `bytes` into an unslabbed view, waiting for every device use
        /// of its storage unless `idle` holds none can be in flight.
        fn write_whole(&self, bytes: &[u8], idle: bool) -> Result<(), TensorError> {
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
            let _guard = match idle {
                true => self
                    .allocation
                    .try_acquire(true)
                    .expect("storage written as idle is used by no device work in flight"),
                false => self.allocation.acquire(true)?,
            };
            self.backed()?;
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

        /// Write a canonical, contiguous tensor's physical bytes in place as
        /// `fill` hands them over, chunk by chunk, each at its byte offset
        /// into the tensor, holding its host access until `fill` returns: a
        /// whole upload filled as a stream lands, never held on the host at
        /// once. A chunk outside the tensor fails `fill`'s write; covering
        /// every byte is `fill`'s to do, and on error the tensor is only
        /// partly initialized.
        pub fn write_chunks(
            &self,
            fill: impl FnOnce(&mut dyn FnMut(u64, &[u8]) -> std::io::Result<()>) -> std::io::Result<()>,
        ) -> Result<(), TensorError> {
            if self.slabs.is_some() {
                return Err(TensorError::SlabLayout(
                    "a chunked upload requires one contiguous allocation".into(),
                ));
            }
            if self.allocation.storage().read_only() {
                return Err(TensorError::HostRegion("mapped tensor is read-only".into()));
            }
            let canonical = crate::layout::canonical(self.representation, &self.extents)?;
            if self.strides != canonical.strides || self.byte_len != canonical.byte_len {
                return Err(TensorError::HostRegion(
                    "a chunked upload requires a canonical contiguous tensor".into(),
                ));
            }
            let _guard = self.allocation.acquire(true)?;
            self.backed()?;
            let mut write = |at: u64, bytes: &[u8]| -> std::io::Result<()> {
                let end = at.checked_add(bytes.len() as u64).filter(|end| *end <= self.byte_len);
                if end.is_none() {
                    return Err(std::io::Error::other(format!(
                        "a chunk of {} bytes at {at} of a tensor of {}",
                        bytes.len(),
                        self.byte_len
                    )));
                }
                self.allocation
                    .storage()
                    .write(self.byte_offset + at, bytes)
                    .map_err(std::io::Error::other)
            };
            fill(&mut write).map_err(|error| TensorError::HostIo(error.to_string()))
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
        /// The allocation the tensor views: its identity and its backed
        /// bytes.
        pub(crate) fn allocation_bytes(&self) -> (u64, u64) {
            (self.allocation.identity(), self.allocation.bytes())
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
        /// What this tensor's allocation is charged, the record tables its
        /// storage holds included.
        pub fn charged_bytes(&self) -> u64 {
            self.allocation.charged_bytes()
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
            let paths = self.allocation.storage_path_charges(&self.device.memory);
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
                    slabs: Some(SlabView {
                        rows_per_slab: slabs.rows_per_slab,
                        regions,
                    }),
                    alias_identity: self.alias_identity,
                    slab_lease: Some(lease),
                    pin: None,
                    records: OnceLock::new(),
                });
            }
            self.view(&crate::api::kernel::ViewOperation::Reshape {
                extents: extents.to_vec(),
            })
        }
        /// This tensor seen through one view operation (`layout::apply_view`).
        fn view(&self, operation: &crate::api::kernel::ViewOperation) -> Result<Self, TensorError> {
            if self.packed() {
                return Err(TensorError::Region(RegionError::NotContiguous));
            }
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
            let mut derived = Self::new_view(
                self.device.clone(),
                self.allocation.clone(),
                view.byte_offset,
                view.byte_len,
                self.representation,
                view.extents,
                view.strides,
            );
            derived.pin = self.pin.clone();
            Ok(derived)
        }
        /// The view's range in its alias space. A slab view's alias space is
        /// its own logical tensor, which starts at 0; its `byte_offset` is
        /// where its address table's entries lie, what it binds, not where
        /// its elements do.
        pub fn descriptor(&self) -> TensorDescriptor {
            TensorDescriptor {
                device: self.device.kind.identity(),
                representation: self.representation,
                extents: self.extents.clone(),
                strides: self.strides.clone(),
                allocation: self.alias_identity(),
                byte_offset: if self.slabs.is_some() { 0 } else { self.byte_offset },
                byte_len: self.byte_len,
            }
        }
    }

    #[cfg(test)]
    mod chunk_tests {
        use super::TensorInner;
        use std::sync::Arc;
        use seismic_lang::registry;
        use seismic_lang::types::DType;

        /// A chunked upload lands each chunk at its offset, whatever order
        /// they come in, and refuses one past the tensor.
        #[test]
        fn a_chunked_upload_lands_each_chunk_at_its_offset() {
            let devices = crate::devices::Devices::discover(&[]).unwrap();
            let device = devices.open_backend(registry::BackendName::Cpu).unwrap();
            let count = 3 * 1024 * 1024 / 4 + 5;
            let tensor = TensorInner::from_host(&device, registry::dense(DType::F32), &[count], &vec![0; count as usize * 4]).unwrap();
            let bytes = (0..count * 4).map(|i| i as u8).collect::<Vec<_>>();
            let half = bytes.len() / 2;
            tensor.write_chunks(|write| {
                write(half as u64, &bytes[half..])?;
                write(0, &bytes[..half])
            })
            .unwrap();
            assert_eq!(tensor.read_to_host().unwrap(), bytes);
            assert!(tensor.write_chunks(|write| write(1, &bytes)).is_err());
        }

        /// Views formed per run of the same records bind the one table their
        /// storage holds, entry `i` record `i`'s address: binding allocates
        /// nothing after the first, and the storage is charged the table once.
        #[test]
        fn views_of_the_same_records_bind_the_table_their_storage_holds() {
            let device = crate::devices::Devices::discover(&[])
                .unwrap()
                .open_backend(registry::BackendName::Cpu)
                .unwrap();
            let tensor = TensorInner::zeros(&device, registry::dense(DType::U32), &[4, 8]).unwrap();
            let charged = tensor.allocation.charged_bytes();
            let first = tensor.slice_leading(1, 3).unwrap().records().unwrap();
            let again = tensor.slice_leading(1, 3).unwrap().records().unwrap();
            let later = tensor.slice_leading(2, 4).unwrap().records().unwrap();
            assert!(Arc::ptr_eq(&first.table, &again.table));
            assert!(Arc::ptr_eq(&first.table, &later.table));
            assert_eq!((first.table_offset, later.table_offset), (8, 16));
            assert_eq!(tensor.allocation.charged_bytes(), charged + 4 * 8);
            let mut entries = [0u8; 32];
            first.table.storage().read(0, &mut entries).unwrap();
            let base = device.allocation_address(&tensor.allocation, 0);
            let addresses = entries
                .chunks_exact(8)
                .map(|entry| u64::from_le_bytes(entry.try_into().unwrap()) - base)
                .collect::<Vec<_>>();
            assert_eq!(addresses, [0, 32, 64, 96]);
        }

        /// A single record binds any table its storage holds with an entry
        /// at its address: once the storage holds the table of its rows, a
        /// view of one record of several rows (a run of a streamed table's
        /// slots) binds into it and allocates nothing; records spaced
        /// otherwise bind a table of their own.
        #[test]
        fn a_single_record_binds_a_held_table_with_its_address() {
            let device = crate::devices::Devices::discover(&[])
                .unwrap()
                .open_backend(registry::BackendName::Cpu)
                .unwrap();
            let tensor = TensorInner::zeros(&device, registry::dense(DType::U32), &[4, 8]).unwrap();
            let rows = tensor.records().unwrap();
            let charged = tensor.allocation.charged_bytes();
            let run = tensor
                .slice_leading(1, 3)
                .unwrap()
                .reshape(&[1, 2, 8])
                .unwrap()
                .records()
                .unwrap();
            assert!(Arc::ptr_eq(&rows.table, &run.table));
            assert_eq!(run.table_offset, 8);
            assert_eq!(tensor.allocation.charged_bytes(), charged);
            let pairs = tensor.reshape(&[2, 2, 8]).unwrap().records().unwrap();
            assert!(!Arc::ptr_eq(&rows.table, &pairs.table));
            assert_eq!(tensor.allocation.charged_bytes(), charged + 2 * 8);
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

        /// A reserved tensor's first commitment and its recommits are
        /// admitted by one rule, and a refusal names the commitment and the
        /// rows the reservation spans.
        #[test]
        fn a_commitment_outside_the_reservation_names_both() {
            let device = crate::devices::Devices::discover(&[])
                .unwrap()
                .open_backend(registry::BackendName::Cpu)
                .unwrap();
            let device_storage = seismic_topology::ReservedContract::DeviceOnly;
            let dense = registry::dense(DType::U32);
            assert!(matches!(
                TensorInner::reserved(&device, device_storage, dense, &[4, 8], 0, None),
                Err(crate::api::TensorError::Uncommittable { committed: 0, rows: 4 })
            ));
            let (_, resizer) = TensorInner::reserved(&device, device_storage, dense, &[4, 8], 2, None).unwrap();
            let Err(refusal) = resizer.recommit(5) else {
                panic!("a commitment past the reservation is refused")
            };
            assert!(matches!(
                refusal.error,
                crate::api::TensorError::Uncommittable { committed: 5, rows: 4 }
            ));
        }

        /// A region reaching past a reserved tensor's backed bytes is
        /// refused as unbacked; a recommit beneath a live region, as pinned.
        #[test]
        fn a_region_past_the_backed_bytes_is_unbacked_and_a_shrink_beneath_one_is_pinned() {
            use crate::api::{RegionError, TensorError};
            let device = crate::devices::Devices::discover(&[])
                .unwrap()
                .open_backend(registry::BackendName::Cpu)
                .unwrap();
            let dense = registry::dense(DType::U32);
            let (rows, n) = (8192u64, 256u64);
            let (reserved, resizer) = TensorInner::reserved(
                &device,
                seismic_topology::ReservedContract::DeviceOnly,
                dense,
                &[rows, n],
                4096,
                None,
            )
            .unwrap();
            assert!(matches!(
                reserved.region(dense, &[4097, n], 0),
                Err(TensorError::Region(RegionError::Unbacked { end, .. })) if end == 4097 * n * 4
            ));
            let region = reserved.region(dense, &[1024, n], 1024 * n * 4).unwrap();
            let Err(refusal) = resizer.recommit(2047) else {
                panic!("a shrink beneath a live region is refused")
            };
            assert!(matches!(
                refusal.error,
                TensorError::Region(RegionError::Pinned { pinned, .. }) if pinned == 2048 * n * 4
            ));
            drop(region);
        }

        /// A view's bytes are attributed from its storage's own bytes alone:
        /// the record tables the storage holds lie elsewhere, so a range
        /// past the storage reaches past its charge however many it holds.
        #[test]
        fn a_view_past_its_storage_reaches_past_the_charge_whatever_tables_it_holds() {
            let device = crate::devices::Devices::discover(&[])
                .unwrap()
                .open_backend(registry::BackendName::Cpu)
                .unwrap();
            let tensor = TensorInner::zeros(&device, registry::dense(DType::U32), &[4, 8]).unwrap();
            tensor.records().unwrap();
            assert_eq!(tensor.allocation.charged_bytes(), 128 + 4 * 8);
            let mut past = tensor.slice_leading(3, 4).unwrap();
            past.byte_offset += 8;
            assert!(matches!(
                past.view_store_charges(),
                Err(crate::api::TensorError::ViewCharge(
                    crate::api::ViewChargeError::BeyondCharge
                ))
            ));
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
        pub fn scalar_result_bytes(&self) -> u64 {
            self.inner.scalar_result_bytes()
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
            /// The operations with their numbers as expressions over a
            /// graph trace's facts (`native::dimension`), each of
            /// `operations` the one at the recording's; none where no fact
            /// reaches any of them, so `operations` are the view at every
            /// model's facts.
            expressions: Option<Arc<[crate::native::dimension::ViewExpr]>>,
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
            expressions: Option<Arc<[crate::native::dimension::ViewExpr]>>,
        ) {
            self.arguments.push(EncodedWorkflowArgument::Tensor(
                WorkflowTensorArgument::ResultView { result, operations, expressions },
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

    /// [`call_native`] of a call whose awaited words already hold their
    /// answers: valid on every backend (`NativePrepared::call_answered`).
    pub fn call_native_answered(
        kernel: &Arc<NativePreparedAny>,
        args: EncodedArgs,
    ) -> Result<DecodedResults, CallError> {
        kernel.inner.call_answered(args)
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
        pub fn is_complete(&self) -> bool {
            self.inner.is_complete()
        }

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
