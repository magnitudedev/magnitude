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
    /// Argument `parameter` of a node of checked entry `entry` is not what
    /// the entry declares: `expected` against the `bound` argument.
    NativeGraphArgumentMismatch {
        entry: &'static str,
        parameter: usize,
        expected: String,
        bound: String,
    },
    NativePortAlreadyBound {
        port: usize,
    },
    /// A slab view bound to a port a launch reads its grid from: a grid is
    /// read whole from its binding, which a slab view's is not.
    NativePortSlabGrid {
        port: usize,
    },
    /// A tensor with elements other than a slab view bound to a slab port
    /// (`&slab`): its natives read its binding as a slab address table. In
    /// a graph, also a view with elements of a port from other than its
    /// first byte: no table of the port's addresses starts there.
    PlainSlabBinding {
        parameter: String,
    },
    /// A slab view with elements bound to a plain tensor port (neither
    /// `&slab` nor `records`): its binding is an address table, not elements.
    SlabPlainBinding {
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
    /// An operand carry declared in (`carry_in`) that no consumer of the
    /// graph reads: none forms the operand of its tensor under its norm and
    /// epsilon. `port` is the carry's rows port.
    UnreadOperandCarry {
        port: usize,
    },
    /// Node `node`, whose result an operand carry is declared out of
    /// (`carry_out`), emits no normed operand of it in both forms a carry
    /// holds, or emits it to a consumer in its own graph, or a carry of it
    /// was declared twice.
    OperandEmitter {
        node: usize,
    },
    /// Node `node` reads a carried operand in forms other than the carry's:
    /// its rows not formed, its Q8 blocks formed where the carry holds none
    /// or not formed where it holds them, its NVFP4 transform not the
    /// carry's, or a formation that cannot be left out.
    OperandForms {
        node: usize,
    },
    /// Node `node` binds an activation element other than its operand's
    /// other end: a pair's producer, or the carry it reads or writes.
    OperandElement {
        node: usize,
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
            Self::Workflow(WorkflowError::NativeGraphArgumentMismatch { entry, parameter, expected, bound }) => {
                write!(
                    f,
                    "native graph argument {parameter} of {entry} takes {expected}, bound {bound}"
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
            Self::Workflow(WorkflowError::SlabPlainBinding { parameter }) => write!(
                f,
                "`{parameter}` is a plain tensor port, which binds elements, not a slab view's address table"
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
            Self::Workflow(WorkflowError::UnreadOperandCarry { port }) => {
                write!(f, "the operand carried in at native graph port {port} has no consumer that reads it")
            }
            Self::Workflow(WorkflowError::OperandEmitter { node }) => write!(
                f,
                "native graph node {node} emits no normed operand in both carried forms, or emits it within its graph"
            ),
            Self::Workflow(WorkflowError::OperandForms { node }) => {
                write!(f, "native graph node {node} forms its operand other than the carry it reads holds it")
            }
            Self::Workflow(WorkflowError::OperandElement { node }) => {
                write!(f, "native graph node {node} binds another activation element than its operand's other end")
            }
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
    /// A view's store charges cannot be read.
    ViewCharge(ViewChargeError),
    /// A typed region of a view's bytes cannot be formed.
    Region(RegionError),
    /// A host-shared word of a view cannot be accessed.
    SharedWord(SharedWordError),
    /// A raw host read (`read_shared`) of a dense floating tensor: its values leave only through
    /// a read that gives every NaN as the registry's one NaN (`read_with`).
    FloatingSharedRead {
        representation: &'static str,
    },
    Execution(ExecutionError),
}

/// Why a 32-bit word of a view cannot be accessed in place by the host
/// (`load_acquire`, `store_release`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SharedWordError {
    /// The view's storage is not one allocation the host addresses in place
    /// (device memory, a slabbed tensor, a read-only mapping).
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
    /// The region's `bytes` from `offset` reach past the view's.
    Outside {
        offset: u64,
        bytes: u64,
        view_bytes: u64,
    },
    /// The region would start at `offset` in its allocation, off its
    /// layout's `alignment`.
    Unaligned { offset: u64, alignment: u64 },
    /// The view at `offset` has no records of its own: no leading rows each
    /// in bytes of their own.
    Unrecorded { offset: u64 },
    /// A region of a dense floating tensor would view its bytes as another
    /// representation, letting a host read skip NaN canonicalization.
    FloatingReinterpret {
        from: &'static str,
        to: &'static str,
    },
}

impl fmt::Display for RegionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotContiguous => {
                f.write_str("a region needs a contiguous view of an allocation")
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
            Self::Unrecorded { offset } => write!(
                f,
                "the view at allocation offset {offset} has no records of its own"
            ),
            Self::FloatingReinterpret { from, to } => write!(
                f,
                "a floating `{from}` view cannot be re-viewed as `{to}`; its values stay `{from}` so every host read canonicalizes its NaNs"
            ),
        }
    }
}

/// Canonicalizes every NaN in `bytes` (whole elements of `representation` when
/// it is a dense floating type) to the registry's one NaN (`layout::canonicalize_in_place`).
pub fn canonicalize_in_place(
    representation: seismic_lang::ids::RepresentationId,
    bytes: &mut [u8],
) {
    crate::layout::canonicalize_in_place(representation, bytes);
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
            Self::ViewCharge(error) => write!(f, "{error}"),
            Self::Region(error) => write!(f, "{error}"),
            Self::SharedWord(error) => write!(f, "{error}"),
            Self::FloatingSharedRead { representation } => write!(
                f,
                "a raw shared read of `{representation}` values; a floating tensor is read only with its NaNs canonical"
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
    use crate::driver::Allocation;
    use seismic_topology::{Contract, DeviceTopology, Executor, StoreId, StoreIdentity};
    use std::sync::Arc;

    /// One opened executor: its immutable description, execution resources
    /// and the stores it charges.
    pub struct DeviceInner {
        /// The tests that opened the device, which its work is recorded for
        /// where the test runner asks (`seismic_lang::registry::TestDevice`).
        pub(crate) test: Option<seismic_lang::registry::TestDevice>,
        pub(crate) info: Executor,
        pub(crate) topology: Arc<DeviceTopology>,
        pub(crate) memory: Arc<crate::memory::DeviceStores>,
        pub(crate) capabilities: std::sync::OnceLock<Vec<String>>,
        pub(crate) kind: OpenedKind,
        /// The device's properties, described once by its backend when it
        /// opened.
        pub(crate) properties: seismic_lang::device::DeviceProperties,
        /// Forms this device's native programs.
        pub(crate) programs: crate::backends::DevicePrograms,
        /// The active submission trace, if any (`native::trace`).
        pub(crate) trace: std::sync::Mutex<Option<Arc<crate::native::trace::TraceSink>>>,
        /// The program store this device was opened with; its program
        /// former forms through it. A device is shared with callers naming
        /// no store or the same one.
        pub(crate) artifacts: Arc<crate::artifacts::ProgramStore>,
        /// Native submission order and CUDA graph replays.
        pub(crate) native: crate::native::NativeState,
        /// What the device's recorded graphs hold for their census,
        /// measured on opening where the driver takes it.
        pub(crate) graph_cost: seismic_compiler::graph_cost::GraphCost,
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
        /// The device did work, a submission or an allocation: its test
        /// ran there (`seismic_lang::registry::TestDevice::ran`).
        pub(crate) fn ran_work(&self) {
            if let Some(test) = &self.test {
                test.ran();
            }
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
        /// Whether a Vulkan device has cooperative matrices to form
        /// (`FormationTarget::matrix_hardware`, whether or not the
        /// projection GEMMs choose the path), which a matrix program is
        /// empty without; `None` on every other backend.
        pub fn forms_matrix_path(&self) -> Option<bool> {
            match &self.kind {
                #[cfg(not(target_os = "macos"))]
                OpenedKind::Vulkan(opened) => Some(opened.service().formation().matrix_hardware),
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
        /// ([`seismic_topology::HostNode`]).
        pub fn host_node(&self) -> seismic_topology::HostNode {
            self.topology
                .physical_device(self.info.physical())
                .expect("an executor's device is of its revision")
                .host_node()
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
        pub(crate) fn allocation_address(&self, allocation: &Arc<Allocation>, offset: u64) -> u64 {
            self.kind.allocation_address(allocation, offset)
        }
        /// What decides how a configuration measures on this device: the
        /// device, its backend's identity, the toolchain material that forms
        /// its programs (a compiler, target or option change forms other
        /// programs from the same source), and the bytes a cold pass of its
        /// timing reads where it times cold passes
        /// (`native::point::cold_bytes`). Not the build of the backend's
        /// formation code, which keys stored programs alone
        /// ([`seismic_native_target::ToolchainIdentity::stored`]).
        pub fn tuning_identity(&self) -> String {
            let mut identity = format!("{};{}", self.info.name(), self.kind.tuning_identity());
            if let Some(toolchain) = self.programs.toolchain() {
                identity = format!("{identity};toolchain {}", toolchain.material);
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
        /// Charges of this device, each store's entry the one `entry_in`
        /// names for it, called once with the identity of each store the
        /// device reaches: how the embedder's ledger mints what its holdings
        /// allocate through ([`crate::memory::Charges`]). Handed no store,
        /// it cannot name a store's own entry.
        ///
        /// # Panics
        /// Where `entry_in` names an entry of another store than it was given.
        pub fn charges(
            self: &Arc<Self>,
            entry_in: impl FnMut(&StoreIdentity) -> Arc<dyn crate::memory::Entry>,
        ) -> crate::memory::Charges {
            crate::memory::Charges::mint(self, entry_in)
        }
        /// The bytes the embedder's ledger counts the process holding in the
        /// store device storage is placed on ([`crate::memory::Store::allocated`]).
        pub fn allocated(&self) -> u64 {
            self.memory.store(Contract::DeviceOnly).allocated()
        }
        /// Bring the driver's own memory up to date before a reading: on
        /// Vulkan the memory its driver took that only asking shows
        /// (`seismic_vulkan::Device::reconcile_driver_memory`), charged to
        /// the device's own entries, and the grant read in that same query;
        /// on Metal the residency set requested resident, which a lapse
        /// while the device idled may have un-wired
        /// (`seismic_metal::MetalDevice::request_residency`), which re-wires
        /// it with no completion the OS reports: the host's wired count, read
        /// after this, may still fall short of our wired buffers. Every other backend
        /// charges its driver's memory as it is taken and reads no grant
        /// here.
        pub fn reconcile_driver_memory(&self) -> Option<crate::memory::Grant> {
            match &self.kind {
                #[cfg(not(target_os = "macos"))]
                OpenedKind::Vulkan(opened) => Some(crate::memory::Grant { left: opened.service().reconcile_driver_memory() }),
                #[cfg(target_os = "macos")]
                OpenedKind::Metal(opened) => {
                    opened.service().request_residency();
                    None
                }
                _ => None,
            }
        }
        /// Ensure the device's command queue can hold at least `count`
        /// uncompleted command buffers at once. On Metal, sizes or grows the
        /// idle `MTLCommandQueue` with
        /// `newCommandQueueWithMaxCommandBufferCount:`; no-op on backends
        /// whose queues have no fixed command-buffer ceiling.
        pub fn ensure_command_buffer_capacity(&self, count: usize) -> Result<(), super::ExecutionError> {
            let _ = count;
            match &self.kind {
                #[cfg(target_os = "macos")]
                OpenedKind::Metal(opened) => opened.service().ensure_command_buffer_capacity(count),
                _ => Ok(()),
            }
        }
        /// The most [`Self::allocated`] has been since the last
        /// [`Self::restart_high_water`].
        pub fn high_water(&self) -> u64 {
            self.memory.store(Contract::DeviceOnly).high_water()
        }
        /// Start [`Self::high_water`] again from what is allocated now.
        pub fn restart_high_water(&self) {
            self.memory.store(Contract::DeviceOnly).restart_high_water()
        }
        /// What an allocation of `contract` storage can take now
        /// ([`crate::memory::Store::room`]): one needing more is refused as a
        /// capacity shortfall.
        pub fn room(&self, contract: Contract) -> u64 {
            self.memory.store(contract).room()
        }
        /// The store device storage is placed on first.
        pub fn device_store(&self) -> StoreId {
            let path = self
                .info
                .paths(Contract::DeviceOnly)
                .first()
                .expect("an opened executor places device storage");
            self.topology.access_path(*path).expect("a placement's path belongs to its topology").store()
        }
    }
}

pub mod tensor {
    use super::{
        device::DeviceInner, HostRegion, RegionError, SharedWordError, TensorError,
        ViewChargeError,
    };
    use crate::driver::Allocation;
    use crate::memory::Charges;
    use crate::pool::{SlabHold, Slots};
    use crate::layout;
    use seismic_compiler::errors::ExecutionError;
    use seismic_compiler::prepared::TensorDescriptor;
    use seismic_lang::ids::RepresentationId;
    use seismic_lang::registry::{representation_info, RepresentationKind};
    use seismic_topology::Contract;
    use std::sync::{
        atomic::{fence, AtomicU32, AtomicU64, Ordering},
        Arc, Mutex, OnceLock, Weak,
    };

    /// A non-owning observation of one physical tensor allocation. It keeps
    /// no storage alive and reads the same charge as the device ledger.
    pub struct TensorStorageObserver {
        allocation: Weak<Allocation>,
        identity: u64,
        topology: Arc<seismic_topology::DeviceTopology>,
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
            Some(self.allocation.upgrade()?.path_charges())
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

    /// How a tensor view is backed: either elements directly in its allocation
    /// (optionally holding the slab granule it lies in), or a logical slab view
    /// addressed through a region table with its own alias identity.
    enum TensorBacking {
        Plain {
            /// The slab this view lies in, held while any view of it lives: its
            /// store's slot, and its granule where it is a pool's
            /// (`crate::pool::SlabHold`).
            hold: Option<Arc<SlabHold>>,
        },
        Slabs {
            view: SlabView,
            alias_identity: u64,
        },
    }

    pub struct TensorInner {
        device: Arc<DeviceInner>,
        allocation: Arc<Allocation>,
        byte_offset: u64,
        byte_len: u64,
        representation: RepresentationId,
        extents: Vec<u64>,
        strides: Vec<u64>,
        backing: TensorBacking,
        /// This view's records as a native record port binds them, formed
        /// at its first such binding ([`TensorInner::forming_records`]).
        records: OnceLock<Arc<Records>>,
    }

    /// A view's records as a native record port reaches them
    /// ([`TensorInner::forming_records`]): through a table of their device
    /// addresses.
    pub(crate) struct Records {
        /// The table's allocation, and the byte offset of the view's entry 0.
        pub(crate) table: Arc<Allocation>,
        pub(crate) table_offset: u64,
        pub(crate) count: u64,
        /// The bytes of one record.
        pub(crate) record: u64,
        /// The allocation the records lie in, one after another from its
        /// byte `.1`, which the encodings order by (`native::hazard`); none
        /// for a slab view's, one record a slab, which its slab reach orders
        /// ([`TensorInner::slab_reach`]).
        #[cfg(any(test, target_os = "macos"))]
        pub(crate) storage: Option<(Arc<Allocation>, u64)>,
    }

    /// Where a view's records lie in its storage: records of a nonzero size
    /// one after another from the view's first byte. Only a view with
    /// records has one ([`TensorInner::record_placement`]).
    #[derive(Clone, Copy, Debug)]
    pub struct RecordPlacement {
        start: u64,
        record: std::num::NonZeroU64,
    }

    impl RecordPlacement {
        /// The bytes of one record.
        pub fn record(&self) -> u64 {
            self.record.get()
        }

        /// Where record `index` lies, in bytes from the view's first.
        pub fn offset(&self, index: u64) -> u64 {
            index * self.record()
        }

        /// Where record `index` lies, in bytes from the storage's first.
        fn at(&self, index: u64) -> u64 {
            self.start + self.offset(index)
        }
    }

    /// What a slab view's binding reaches ([`TensorInner::slab_reach`]): its
    /// address-table entries and, by address, each slot's slab region
    /// (allocation, byte offset, bytes), `None` a free slot's.
    pub(crate) struct SlabReach {
        pub(crate) table_bytes: u64,
        /// The bytes of one slab's rows of the view, as its elements lie
        /// in the view: slot `s` holds the view's bytes from `s` of them.
        pub(crate) slab_bytes: u64,
        pub(crate) regions: Vec<Option<(u64, u64, u64)>>,
        /// Per slot, the backed slots before it, and last all of them: how
        /// many of a run of slots are backed is one difference.
        backed: Vec<usize>,
        /// Distinct `Allocation::access_identity()` values of any lent slab
        /// lenders in this view's store (empty when no slabs are lent).
        #[cfg(any(test, target_os = "macos"))]
        pub(crate) lender_groups: Box<[usize]>,
        /// Globally unique logical allocation identity for this view's
        /// `(Slots, region)` (`Slots::region_identity`), used to summarize
        /// pool-backed slabs in O(1) reaches instead of O(slabs).
        #[cfg(any(test, target_os = "macos"))]
        pub(crate) logical_allocation: u64,
        /// Starting slot index of this view within its `Slots` store.
        #[cfg(any(test, target_os = "macos"))]
        pub(crate) first_slot: usize,
        /// Contiguous physical lender range for this view's leading lent
        /// slots, if any: `(lender_allocation_id, lender_byte_base, lent_bytes)`
        /// where view byte `b < lent_bytes` lies at `lender_byte_base + b` in
        /// `lender_allocation_id`.
        #[cfg(any(test, target_os = "macos"))]
        pub(crate) lent_region: Option<(u64, u64, u64)>,
    }

    /// The slots of `slab_bytes` each that hold bytes `[offset, offset +
    /// bytes)` of a slab view, slot `s` its bytes from `s` slabs' on.
    fn slots_holding(slab_bytes: u64, offset: u64, bytes: u64) -> std::ops::Range<usize> {
        if bytes == 0 {
            return 0..0;
        }
        (offset / slab_bytes) as usize..(offset + bytes).div_ceil(slab_bytes) as usize
    }

    impl SlabReach {
        fn new(
            table_bytes: u64,
            slab_bytes: u64,
            regions: Vec<Option<(u64, u64, u64)>>,
            #[cfg(any(test, target_os = "macos"))] lender_groups: Box<[usize]>,
            #[cfg(any(test, target_os = "macos"))] logical_allocation: u64,
            #[cfg(any(test, target_os = "macos"))] first_slot: usize,
            #[cfg(any(test, target_os = "macos"))] lent_region: Option<(u64, u64, u64)>,
        ) -> Self {
            let backed = std::iter::once(0)
                .chain(regions.iter().scan(0, |backed, region| {
                    *backed += usize::from(region.is_some());
                    Some(*backed)
                }))
                .collect();
            Self {
                table_bytes,
                slab_bytes,
                regions,
                backed,
                #[cfg(any(test, target_os = "macos"))]
                lender_groups,
                #[cfg(any(test, target_os = "macos"))]
                logical_allocation,
                #[cfg(any(test, target_os = "macos"))]
                first_slot,
                #[cfg(any(test, target_os = "macos"))]
                lent_region,
            }
        }

        /// The slots that hold the view's bytes `[offset, offset + bytes)`.
        #[cfg(any(test, target_os = "macos"))]
        pub(crate) fn slots(&self, offset: u64, bytes: u64) -> std::ops::Range<usize> {
            slots_holding(self.slab_bytes, offset, bytes)
        }

        /// Whether any of `slots` is backed.
        pub(crate) fn any_backed(&self, slots: std::ops::Range<usize>) -> bool {
            self.backed[slots.end] > self.backed[slots.start]
        }

        #[cfg(test)]
        pub(crate) fn for_test(
            slab_bytes: u64,
            regions: Vec<Option<(u64, u64, u64)>>,
            logical_allocation: u64,
            first_slot: usize,
            lent_region: Option<(u64, u64, u64)>,
        ) -> Self {
            Self::new(
                seismic_lang::registry::record_table_entry(regions.len() as u64),
                slab_bytes,
                regions,
                Box::new([]),
                logical_allocation,
                first_slot,
                lent_region,
            )
        }

        /// The bytes device work reaches in slot `slot` for the view's bytes
        /// `[offset, offset + bytes)`, which it holds some of: the slab's
        /// bytes that hold them (`Allocation::identity`, range); none for a
        /// free slot.
        fn slot_bytes(&self, slot: usize, offset: u64, bytes: u64) -> Option<(u64, std::ops::Range<u64>)> {
            let (allocation, start, _) = self.regions[slot]?;
            let first = slot as u64 * self.slab_bytes;
            let from = offset.max(first) - first;
            let to = (offset + bytes).min(first + self.slab_bytes) - first;
            Some((allocation, start + from..start + to))
        }

        /// The bytes device work reaches for the view's bytes `[offset,
        /// offset + bytes)`: in each slot holding some, the slab's bytes
        /// that hold them ([`Self::slot_bytes`]); a free slot's none.
        #[cfg(any(test, target_os = "macos"))]
        pub(crate) fn bytes(
            &self,
            offset: u64,
            bytes: u64,
        ) -> impl Iterator<Item = (u64, std::ops::Range<u64>)> + '_ {
            self.slots(offset, bytes).filter_map(move |slot| self.slot_bytes(slot, offset, bytes))
        }
    }

    /// Whether two byte ranges share a byte (an empty range shares none).
    pub(crate) fn meet(first: &std::ops::Range<u64>, second: &std::ops::Range<u64>) -> bool {
        first.start.max(second.start) < first.end.min(second.end)
    }

    /// The pairs `(first, second)`, `first < second`, of a launch's `count`
    /// buffers that may share no bound byte: those it writes either of
    /// (`written`, as the buffers' declared access says,
    /// [`seismic_lang::entry::TensorAccess::writer`]). A launch reads a
    /// buffer it writes nothing through as memory nothing in the launch
    /// writes (Vulkan formation's execution model,
    /// `seismic_vulkan::uniformity`), and orders its launches by what each
    /// reaches, so binding refuses every such pair whose bytes meet: the one
    /// rule each binder holds its bindings to, by the bytes it binds
    /// ([`Reaching::meets`] for a tensor's).
    pub(crate) fn exclusive_pairs(
        count: usize,
        written: impl Fn(usize) -> bool,
    ) -> impl Iterator<Item = (usize, usize)> {
        (0..count)
            .flat_map(move |first| (first + 1..count).map(move |second| (first, second)))
            .filter(move |&(first, second)| written(first) || written(second))
    }

    /// A tensor as binding's no-alias check reaches it ([`Self::meets`]): a
    /// slab view's reach is read at most once, by the first of its pairs
    /// that needs it, however many pairs it is in.
    pub(crate) struct Reaching<'a> {
        tensor: &'a TensorInner,
        reach: std::cell::OnceCell<Arc<SlabReach>>,
    }

    impl<'a> Reaching<'a> {
        pub(crate) fn new(tensor: &'a TensorInner) -> Self {
            Self { tensor, reach: std::cell::OnceCell::new() }
        }

        /// What `slabs`, the tensor's slab view, reaches now.
        fn reach(&self, slabs: &SlabView) -> &SlabReach {
            self.reach.get_or_init(|| slabs.reach())
        }

        /// Whether the bytes device work reaches for the tensor's bytes
        /// `span` and `other`'s bytes `other_span` share one: what binding
        /// refuses of an exclusive pair ([`exclusive_pairs`]). A plain view
        /// reaches its own bytes; a slab view its lent slabs' in their
        /// lender, one range ([`SlabView::lent_bytes`]), and its other
        /// slabs', which only its store holds, each region's apart
        /// (`crate::pool::SlabStore::in_pool`). So each pair costs O(1),
        /// whatever the slots, and reads a slab view's reach only where two
        /// views of one store's region meet in it or a view of one of its
        /// slabs is the other side.
        pub(crate) fn meets(
            &self,
            span: std::ops::Range<u64>,
            other: &Reaching<'_>,
            other_span: std::ops::Range<u64>,
        ) -> bool {
            match (&self.tensor.backing, &other.tensor.backing) {
                (TensorBacking::Plain { .. }, TensorBacking::Plain { .. }) => {
                    let (identity, range) = self.tensor.plain_bytes(span);
                    let (other_identity, other_range) = other.tensor.plain_bytes(other_span);
                    identity == other_identity && meet(&range, &other_range)
                }
                (TensorBacking::Slabs { view, .. }, TensorBacking::Plain { hold }) => {
                    self.meets_plain(view, span, other.tensor, hold.as_ref(), other_span)
                }
                (TensorBacking::Plain { hold }, TensorBacking::Slabs { view, .. }) => {
                    other.meets_plain(view, other_span, self.tensor, hold.as_ref(), span)
                }
                (
                    TensorBacking::Slabs { view, .. },
                    TensorBacking::Slabs {
                        view: other_view, ..
                    },
                ) => self.meets_slabs(view, span, other_view, other_span),
            }
        }

        /// [`Self::meets`] of this slab view, `slabs`, and `plain`, a plain
        /// view: in a lent slab, or in the one slab whose region `plain`
        /// views, where it is one of this view's store's
        /// ([`TensorInner::slab_hold`]).
        fn meets_plain(
            &self,
            slabs: &SlabView,
            span: std::ops::Range<u64>,
            plain: &TensorInner,
            hold: Option<&Arc<SlabHold>>,
            plain_span: std::ops::Range<u64>,
        ) -> bool {
            let (identity, range) = plain.plain_bytes(plain_span);
            if slabs.lent_bytes(&span).is_some_and(|(lender, lent)| lender == identity && meet(&lent, &range)) {
                return true;
            }
            let Some(hold) = hold.filter(|hold| hold.store() == slabs.slots.id()) else {
                return false;
            };
            let bytes = span.end - span.start;
            hold.slot()
                .checked_sub(slabs.first)
                .filter(|slot| slots_holding(slabs.slab_bytes, span.start, bytes).contains(slot))
                .and_then(|slot| self.reach(slabs).slot_bytes(slot, span.start, bytes))
                .is_some_and(|(allocation, bound)| allocation == identity && meet(&bound, &range))
        }

        /// [`Self::meets`] of two slab views, this one's `slabs` and the
        /// other's `other_slabs`: in their lent slabs, by lender, or in
        /// slabs of one store's one region, where each view's bytes lie as
        /// the region's from its first slot's, which meet in a backed slot
        /// of the store's own (read from this view's reach: those slots lie
        /// in both).
        fn meets_slabs(
            &self,
            slabs: &SlabView,
            span: std::ops::Range<u64>,
            other_slabs: &SlabView,
            other_span: std::ops::Range<u64>,
        ) -> bool {
            if let (Some((lender, lent)), Some((other_lender, other_lent))) =
                (slabs.lent_bytes(&span), other_slabs.lent_bytes(&other_span))
            {
                if lender == other_lender && meet(&lent, &other_lent) {
                    return true;
                }
            }
            if !Arc::ptr_eq(&slabs.slots, &other_slabs.slots) || slabs.region != other_slabs.region {
                return false;
            }
            assert_eq!(slabs.slab_bytes, other_slabs.slab_bytes, "invariant violated: views of one slab region differ in its slab bytes");
            let in_region = |view: &SlabView, span: &std::ops::Range<u64>| {
                let first = view.first as u64 * view.slab_bytes;
                first + span.start..first + span.end
            };
            let (mine, theirs) = (in_region(slabs, &span), in_region(other_slabs, &other_span));
            let common = mine.start.max(theirs.start)..mine.end.min(theirs.end);
            if common.is_empty() {
                return false;
            }
            // The store's own slots of those holding the common bytes; the
            // lent ones were compared above.
            let slots = slots_holding(slabs.slab_bytes, common.start, common.end - common.start);
            let own = slots.start.max(slabs.slots.lent())..slots.end;
            !own.is_empty() && self.reach(slabs).any_backed(own.start - slabs.first..own.end - slabs.first)
        }
    }

    /// A slab view's region of its store: its slots as they are when it is
    /// read, never a snapshot, so the view stays valid as slabs are added,
    /// freed and moved.
    struct SlabView {
        slots: Arc<Slots>,
        region: usize,
        /// The store's first slot in this view, and the slots it spans.
        first: usize,
        count: usize,
        rows_per_slab: u64,
        /// One slab's rows of the region, as this view shapes them:
        /// `[rows_per_slab, ...]`.
        slab_extents: Vec<u64>,
        /// The bytes of one slab's rows of the region.
        slab_bytes: u64,
        /// What device work bound to the view reaches, with the store's
        /// slab generation it was read at ([`SlabView::reach`]): formed
        /// again only once a slab is added, freed or moved, so a dispatch
        /// takes it without allocating.
        reach: Mutex<Option<(u64, Arc<SlabReach>)>>,
    }

    impl SlabView {
        fn new(
            slots: Arc<Slots>,
            region: usize,
            first: usize,
            count: usize,
            rows_per_slab: u64,
            slab_extents: Vec<u64>,
            slab_bytes: u64,
        ) -> Self {
            Self {
                slots,
                region,
                first,
                count,
                rows_per_slab,
                slab_extents,
                slab_bytes,
                reach: Mutex::new(None),
            }
        }

        /// What device work bound to the view reaches now: its address-table
        /// entries (`[count, 2]` u32 from `first`) and each backed slab's
        /// region in `first..first + count`. The generation is read before
        /// the slabs, and the store moves it under its slab lock with every
        /// change, so a reach is never kept past a change it does not show.
        fn reach(&self) -> Arc<SlabReach> {
            let mut cached = self.reach.lock().expect("slab reach lock poisoned");
            let generation = self.slots.generation();
            match &*cached {
                Some((read, reach)) if *read == generation => reach.clone(),
                _ => {
                    #[cfg(any(test, target_os = "macos"))]
                    let lender_groups = {
                        let mut groups: Vec<usize> = self
                            .slots
                            .lenders()
                            .map(|lender| lender.access_identity())
                            .collect();
                        groups.sort_unstable();
                        groups.dedup();
                        groups.into_boxed_slice()
                    };
                    #[cfg(any(test, target_os = "macos"))]
                    let lent_region = self.slots.lent_region(self.region).and_then(|(lender, start)| {
                        let lent_slots = self.slots.lent().saturating_sub(self.first).min(self.count);
                        let lent_bytes = lent_slots as u64 * self.slab_bytes;
                        (lent_bytes > 0).then(|| {
                            (
                                lender.identity(),
                                start + self.first as u64 * self.slab_bytes,
                                lent_bytes,
                            )
                        })
                    });
                    let reach = Arc::new(SlabReach::new(
                        seismic_lang::registry::record_table_entry(self.count as u64),
                        self.slab_bytes,
                        self.slots.reach(self.region, self.first, self.count, self.slab_bytes),
                        #[cfg(any(test, target_os = "macos"))]
                        lender_groups,
                        #[cfg(any(test, target_os = "macos"))]
                        self.slots.region_identity(self.region),
                        #[cfg(any(test, target_os = "macos"))]
                        self.first,
                        #[cfg(any(test, target_os = "macos"))]
                        lent_region,
                    ));
                    *cached = Some((generation, reach.clone()));
                    reach
                }
            }
        }

        /// The bytes of the view's `span` that lie in its store's lent slabs,
        /// which never move nor free: one range of the allocation they are
        /// lent from (`Allocation::identity`, range), as each slot's lies a
        /// slab after the one before; none where none do.
        fn lent_bytes(&self, span: &std::ops::Range<u64>) -> Option<(u64, std::ops::Range<u64>)> {
            let (lender, start) = self.slots.lent_region(self.region)?;
            let lent = self.slots.lent().saturating_sub(self.first).min(self.count) as u64 * self.slab_bytes;
            let first = start + self.first as u64 * self.slab_bytes;
            let end = span.end.min(lent);
            (span.start < end).then(|| (lender.identity(), first + span.start..first + end))
        }

        /// The region's rows of each slot's slab in `first..first + count`
        /// now, `None` a free slot's.
        fn regions(
            &self,
            representation: RepresentationId,
        ) -> Result<Vec<Option<Arc<TensorInner>>>, TensorError> {
            self.slots.region_views(
                self.region,
                self.first,
                self.count,
                representation,
                &self.slab_extents,
            )
        }
    }

    /// A tensor's elements on the host ([`TensorInner::image`]).
    pub(crate) enum TensorImage {
        Whole(Vec<u8>),
        Slabs(Vec<Option<Vec<u8>>>),
    }

    /// A record table over `storage`, entry `i` the device address of its
    /// `i`th byte offset in `offsets`
    /// ([`seismic_lang::registry::record_table`]), in `storage`'s hazard
    /// domain, formed through `charges` as device storage
    /// ([`Charges::of`]), in the kind they are confined to where they are:
    /// the one place a record table is formed.
    pub(crate) fn record_table(
        charges: &Charges,
        storage: &Arc<Allocation>,
        offsets: impl Iterator<Item = u64>,
    ) -> Result<Arc<Allocation>, TensorError> {
        let device = charges.device();
        let bytes = seismic_lang::registry::record_table(
            offsets.map(|offset| device.allocation_address(storage, offset)),
        );
        let table = device.allocate_contract(
            charges.of(Contract::DeviceOnly, bytes.len() as u64)?,
            bytes.len() as u64,
            seismic_lang::registry::RECORD_TABLE_ENTRY_BYTES,
            Some(storage.access_group().clone()),
        )?;
        table.storage().write(0, &bytes)?;
        Ok(table)
    }

    impl TensorInner {
        /// The logical tensor of region `region` of a slab store's `slots`:
        /// `logical_rows` rows of `row_shape`, `rows_per_slab` a slab, bound
        /// through the region's address table, which device work reads as it
        /// runs. Every slab of the store shares the table's hazard domain,
        /// save its lent slabs ([`Self::bound_allocations`]).
        pub(crate) fn slabbed(
            slots: &Arc<Slots>,
            region: usize,
            representation: RepresentationId,
            rows_per_slab: u64,
            logical_rows: u64,
            row_shape: &[u64],
            alias_identity: u64,
        ) -> Result<Self, TensorError> {
            if rows_per_slab == 0 {
                return Err(TensorError::SlabLayout("empty slab tensor".into()));
            }
            let capacity = slots.capacity() as u64;
            let capacity_rows = capacity
                .checked_mul(rows_per_slab)
                .ok_or_else(|| TensorError::SlabLayout("logical slab extent overflows".into()))?;
            if logical_rows == 0
                || logical_rows > capacity_rows
                || logical_rows <= capacity_rows - rows_per_slab
            {
                return Err(TensorError::SlabLayout(
                    "logical extent does not match slab capacity".into(),
                ));
            }
            let extents = std::iter::once(logical_rows).chain(row_shape.iter().copied()).collect::<Vec<_>>();
            let layout = layout::canonical(representation, &extents)?;
            let slab_extents = std::iter::once(rows_per_slab).chain(row_shape.iter().copied()).collect::<Vec<_>>();
            let slab_bytes = layout::canonical(representation, &slab_extents)?.byte_len;
            Ok(Self {
                device: slots.device().clone(),
                allocation: slots.table().clone(),
                byte_offset: slots.region_offset(region),
                byte_len: layout.byte_len,
                representation,
                extents,
                strides: layout.strides,
                backing: TensorBacking::Slabs {
                    view: SlabView::new(slots.clone(), region, 0, slots.capacity(), rows_per_slab, slab_extents, slab_bytes),
                    alias_identity,
                },
                records: OnceLock::new(),
            })
        }

        /// A canonical view of `representation` over `extents` at `offset`
        /// of `allocation`, a slab's region, holding its granule where it is
        /// a pool's.
        pub(crate) fn in_slab(
            device: &Arc<DeviceInner>,
            allocation: Arc<Allocation>,
            offset: u64,
            representation: RepresentationId,
            extents: &[u64],
            hold: Option<Arc<SlabHold>>,
        ) -> Result<Self, TensorError> {
            let layout = layout::canonical(representation, extents)?;
            Ok(Self::new_plain(
                device.clone(),
                allocation,
                offset,
                layout.byte_len,
                representation,
                extents.to_vec(),
                layout.strides,
                hold,
            ))
        }

        /// The allocations device work bound to this view reaches, each in
        /// its own hazard domain: its own (a slab view's table, which every
        /// pooled slab of its store shares a domain with), and a slab
        /// view's lent slabs' storage.
        pub(crate) fn bound_allocations(&self) -> impl Iterator<Item = Arc<Allocation>> + '_ {
            let lenders = match &self.backing {
                TensorBacking::Plain { .. } => None,
                TensorBacking::Slabs { view, .. } => Some(view.slots.lenders().cloned()),
            };
            std::iter::once(self.allocation.clone()).chain(lenders.into_iter().flatten())
        }

        /// Whether the tensor is a slab view: its binding is an address table,
        /// not its elements.
        pub(crate) fn is_slab_view(&self) -> bool {
            match &self.backing {
                TensorBacking::Plain { .. } => false,
                TensorBacking::Slabs { .. } => true,
            }
        }

        /// Whether the tensor may bind a slab port (`&slab`,
        /// [`seismic_lang::entry::PortBinding::Slab`]): a slab view, or a
        /// tensor of no elements, which has no row to address.
        pub(crate) fn binds_slab_port(&self) -> bool {
            self.is_slab_view() || self.extents.contains(&0)
        }

        /// Whether a call bound to this view holds it until its submission
        /// records its fences: a slab view, whose store's slots device work
        /// reads through its table as it runs.
        pub(crate) fn holds_binding_lease(&self) -> bool {
            self.is_slab_view()
        }

        /// What device work bound to a slab view reaches now: its address-table
        /// entries (the binding, `[slabs, 2]` u32 at the view's offset) and
        /// each backed slab's region ([`SlabView::reach`]); `None` for any
        /// other view.
        pub(crate) fn slab_reach(&self) -> Option<Arc<SlabReach>> {
            match &self.backing {
                TensorBacking::Plain { .. } => None,
                TensorBacking::Slabs { view, .. } => Some(view.reach()),
            }
        }

        /// The slab a view of a slab's region lies in ([`SlabHold`]): its
        /// store's slot; none for any other view.
        pub(crate) fn slab_hold(&self) -> Option<&Arc<SlabHold>> {
            match &self.backing {
                TensorBacking::Plain { hold } => hold.as_ref(),
                TensorBacking::Slabs { .. } => None,
            }
        }

        /// The bytes device work bound to this plain view's bytes `span`
        /// reaches: its own (`Self::alias_identity`, range).
        fn plain_bytes(&self, span: std::ops::Range<u64>) -> (u64, std::ops::Range<u64>) {
            (self.alias_identity(), self.byte_offset + span.start..self.byte_offset + span.end)
        }

        /// The device bytes the tensor's elements occupy: a slab view's
        /// backed slabs, otherwise its byte length.
        pub(crate) fn occupied_bytes(&self) -> u64 {
            match &self.backing {
                TensorBacking::Plain { .. } => self.byte_len,
                TensorBacking::Slabs { view, .. } => {
                    view.reach().regions.iter().flatten().count() as u64 * view.slab_bytes
                }
            }
        }

        /// The bytes a device copy of the tensor is written from
        /// ([`TensorInner::copy_of`]), in canonical layout: all of them, or
        /// each backed slab's region of a slab view.
        pub(crate) fn image(&self) -> Result<TensorImage, TensorError> {
            match &self.backing {
                TensorBacking::Plain { .. } => Ok(TensorImage::Whole(self.read_to_host()?)),
                TensorBacking::Slabs { view, .. } => Ok(TensorImage::Slabs(
                    view.regions(self.representation)?
                        .iter()
                        .map(|region| region.as_ref().map(|r| r.read_to_host()).transpose())
                        .collect::<Result<_, _>>()?,
                )),
            }
        }

        /// A tensor of this one's shape over storage of its own holding
        /// `image` (this tensor's [`TensorInner::image`]): a call reads the
        /// same values from it and shares no byte of storage with a call of
        /// this one. A slab view's copy has its own slabs and address table.
        /// A copy of a view the host addresses in place ([`Self::shared`])
        /// is host-shared too, so the host reads it as it reads this one (a
        /// launch condition's word, `native::condition_holds`). Allocated
        /// through `charges`.
        pub(crate) fn copy_of(&self, charges: &Charges, image: &TensorImage) -> Result<Self, TensorError> {
            let device = charges;
            match (image, &self.backing) {
                (TensorImage::Whole(bytes), TensorBacking::Plain { .. }) => match self.host_address() {
                    Some(_) => {
                        let copy = Self::shared(device, self.representation, &self.extents)?;
                        copy.write_from_host(bytes)?;
                        Ok(copy)
                    }
                    None => Self::from_host(device, self.representation, &self.extents, bytes),
                },
                (TensorImage::Slabs(images), TensorBacking::Slabs { view: slabs, .. })
                    if images.len() == slabs.count =>
                {
                    if slabs.count == 0 {
                        let slots = Slots::copy(device, self.representation, &slabs.slab_extents, &[None])?;
                        let whole = Self::slabbed(
                            &slots,
                            0,
                            self.representation,
                            slabs.rows_per_slab,
                            slabs.rows_per_slab,
                            &self.extents[1..],
                            crate::driver::fresh_allocation_identity(),
                        )?;
                        return whole.slice_leading(0, 0);
                    }
                    let slots = Slots::copy(device, self.representation, &slabs.slab_extents, images)?;
                    Self::slabbed(
                        &slots,
                        0,
                        self.representation,
                        slabs.rows_per_slab,
                        self.extents[0],
                        &self.extents[1..],
                        crate::driver::fresh_allocation_identity(),
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
        /// alignment in the allocation. A dense floating view is viewed only
        /// as its own representation ([`RegionError::FloatingReinterpret`]).
        pub fn region(
            &self,
            representation: RepresentationId,
            extents: &[u64],
            offset: u64,
        ) -> Result<Self, TensorError> {
            self.region_as(representation, extents, offset, false)
        }

        /// [`Self::region`] of another form's records this view's records
        /// carry whole (an expert bank's lent records, carrying its conveyor
        /// or its scratch): a dense floating view may also be viewed as a
        /// non-dense form, whose records its own lend their bytes to.
        pub fn carried_region(
            &self,
            representation: RepresentationId,
            extents: &[u64],
            offset: u64,
        ) -> Result<Self, TensorError> {
            self.region_as(representation, extents, offset, true)
        }

        fn region_as(
            &self,
            representation: RepresentationId,
            extents: &[u64],
            offset: u64,
            carried: bool,
        ) -> Result<Self, TensorError> {
            if !self.device.supports_representation(representation) {
                return Err(TensorError::UnsupportedRepresentation {
                    backend: self.device.info().backend(),
                    representation: representation_info(representation).name,
                });
            }
            let from = representation_info(self.representation);
            let to = representation_info(representation);
            let carries_packed = carried && !matches!(to.kind, RepresentationKind::Dense(_));
            if matches!(from.kind, RepresentationKind::Dense(dtype) if dtype.is_float())
                && representation != self.representation
                && !carries_packed
            {
                return Err(TensorError::Region(RegionError::FloatingReinterpret {
                    from: from.name,
                    to: to.name,
                }));
            }
            let TensorBacking::Plain { hold } = &self.backing else {
                return Err(TensorError::Region(RegionError::NotContiguous));
            };
            if layout::canonical(self.representation, &self.extents)?.strides != self.strides {
                return Err(TensorError::Region(RegionError::NotContiguous));
            }
            let layout = layout::canonical(representation, extents)?;
            if offset.checked_add(layout.byte_len).is_none_or(|end| end > self.byte_len) {
                return Err(TensorError::Region(RegionError::Outside {
                    offset,
                    bytes: layout.byte_len,
                    view_bytes: self.byte_len,
                }));
            }
            let start = self.byte_offset + offset;
            if start % layout.alignment != 0 {
                return Err(TensorError::Region(RegionError::Unaligned {
                    offset: start,
                    alignment: layout.alignment,
                }));
            }
            Ok(Self::new_plain(
                self.device.clone(),
                self.allocation.clone(),
                start,
                layout.byte_len,
                representation,
                extents.to_vec(),
                layout.strides,
                hold.clone(),
            ))
        }

        /// Records `start..end` of this view (its leading-axis rows, where
        /// [`Self::record_placement`] lays them) as a region of their own:
        /// the view a [`Self::region`] of its storage at the first of them
        /// forms.
        pub fn record_region(&self, start: u64, end: u64) -> Result<Self, TensorError> {
            let placement = self
                .record_placement()
                .ok_or(TensorError::Region(RegionError::Unrecorded { offset: self.byte_offset }))?;
            let TensorBacking::Plain { hold } = &self.backing else {
                return Err(TensorError::Region(RegionError::NotContiguous));
            };
            if layout::canonical(self.representation, &self.extents)?.strides != self.strides {
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
            Ok(Self::new_plain(
                self.device.clone(),
                self.allocation.clone(),
                placement.at(start),
                layout.byte_len,
                self.representation,
                extents,
                layout.strides,
                hold.clone(),
            ))
        }

        pub fn device_address(&self) -> u64 {
            self.device.allocation_address(&self.allocation, self.byte_offset)
        }

        /// Where this canonical view's records (leading-axis rows) lie: one
        /// after another from its first byte. None for a view without
        /// records of their own ([`record_bytes`]).
        ///
        /// [`record_bytes`]: seismic_lang::registry::record_bytes
        pub fn record_placement(&self) -> Option<RecordPlacement> {
            let record = seismic_lang::registry::record_bytes(self.representation, &self.extents)?;
            Some(RecordPlacement {
                start: self.byte_offset,
                record,
            })
        }

        /// This canonical view's records as a native record port binds them:
        /// a slab view of one row a slab binds its region's address table,
        /// entry `i` its slot `i`'s record, as its slabs change. Any other
        /// view binds a table its allocation holds of records of its size
        /// and phase, or for a single record any table the allocation holds
        /// with an entry at its address (`Allocation::view_table`); where it
        /// holds none, one is formed through `charges`, the binder's, and
        /// held with the storage from then on. A record port binds a tensor
        /// only through this, as it is bound (`NativeGraphBindings::set`,
        /// a native call's arguments), so a view bound again, or formed per
        /// run over storage that holds its table, allocates nothing.
        ///
        /// Device work reaches the records by address alone, so they must
        /// lie in storage the device holds resident (on Metal, its
        /// residency set): a read-only host mapping, which the device makes
        /// resident only where it is bound, is refused.
        ///
        /// # Panics
        /// Where `charges` are another device's ([`Charges::of_device`]).
        pub(crate) fn forming_records(&self, charges: &Charges) -> Result<Arc<Records>, TensorError> {
            let charges = charges.of_device(&self.device);
            if let Some(records) = self.records.get() {
                return Ok(records.clone());
            }
            let record = seismic_lang::registry::record_bytes(self.representation, &self.extents)
                .ok_or(TensorError::Region(RegionError::Unrecorded { offset: self.byte_offset }))?
                .get();
            let records = match &self.backing {
                TensorBacking::Slabs { view, .. } if view.rows_per_slab == 1 => Records {
                    table: self.allocation.clone(),
                    table_offset: self.byte_offset,
                    count: self.extents[0],
                    record,
                    #[cfg(any(test, target_os = "macos"))]
                    storage: None,
                },
                TensorBacking::Slabs { .. } => {
                    return Err(TensorError::SlabLayout(
                        "a record port binds a slab view of one row a slab".into(),
                    ))
                }
                TensorBacking::Plain { .. } => {
                    let (table, table_offset) = self.record_table(record, charges)?;
                    Records {
                        table,
                        table_offset,
                        count: self.extents[0],
                        record,
                        #[cfg(any(test, target_os = "macos"))]
                        storage: Some((self.allocation.clone(), self.byte_offset)),
                    }
                }
            };
            Ok(self.records.get_or_init(|| Arc::new(records)).clone())
        }

        /// The table a record port binds this unslabbed view's
        /// `record`-byte records through, and the byte of its first entry
        /// ([`Self::forming_records`]), formed through `charges` where none
        /// is held.
        fn record_table(&self, record: u64, charges: &Charges) -> Result<(Arc<Allocation>, u64), TensorError> {
            if layout::canonical(self.representation, &self.extents)?.strides != self.strides {
                return Err(TensorError::Region(RegionError::NotContiguous));
            }
            if self.allocation.storage().read_only() {
                return Err(TensorError::HostRegion(
                    "a record port's records cannot lie in a read-only host mapping".into(),
                ));
            }
            self.allocation.view_table(
                record,
                self.byte_offset,
                self.extents[0],
                charges,
                |phase, records, charges| {
                    record_table(charges, &self.allocation, (0..records).map(|index| phase + index * record))
                },
            )
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
            charges: &Charges,
            representation: RepresentationId,
            extents: &[u64],
        ) -> Result<Self, TensorError> {
            Self::canonical_in(charges, representation, extents, |charges, bytes, alignment| {
                charges.allocate_mapped_zeroed(Contract::Shared, bytes, alignment)
            })
        }

        /// A zero-filled tensor of coherent words the host and device work
        /// load and store while either is running (`Contract::Words`).
        pub fn words(
            charges: &Charges,
            representation: RepresentationId,
            extents: &[u64],
        ) -> Result<Self, TensorError> {
            Self::canonical_in(charges, representation, extents, |charges, bytes, alignment| {
                charges.allocate_mapped_zeroed(Contract::Words, bytes, alignment)
            })
        }

        /// A [`Self::words`] tensor whose device accesses the protocol of
        /// the work that uses it orders across device queues, not their
        /// fences (`Allocation::order_by_protocol`).
        pub fn protocol_words(
            charges: &Charges,
            representation: RepresentationId,
            extents: &[u64],
        ) -> Result<Self, TensorError> {
            let tensor = Self::words(charges, representation, extents)?;
            tensor.allocation().order_by_protocol();
            Ok(tensor)
        }

        /// A zero-filled tensor in storage the host only writes in place
        /// while device work reads it (`Contract::Upload`: a
        /// graph run's inputs' storage, which a large BAR may hold).
        pub fn upload(
            charges: &Charges,
            representation: RepresentationId,
            extents: &[u64],
        ) -> Result<Self, TensorError> {
            Self::canonical_in(charges, representation, extents, |charges, bytes, alignment| {
                charges.allocate_mapped_zeroed(Contract::Upload, bytes, alignment)
            })
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
        /// waiting for device work that uses its storage. A dense floating
        /// tensor is refused ([`TensorError::FloatingSharedRead`]): its values
        /// leave only through the canonicalizing read (`read_with`).
        ///
        /// # Safety
        /// The tensor was made by [`Self::shared`] (or is a view of one),
        /// and no device work in flight writes these bytes.
        pub unsafe fn read_shared(&self, offset: u64, into: &mut [u8]) -> Result<(), TensorError> {
            let info = representation_info(self.representation);
            if matches!(info.kind, RepresentationKind::Dense(dtype) if dtype.is_float()) {
                return Err(TensorError::FloatingSharedRead {
                    representation: info.name,
                });
            }
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
        /// allocation (see [`Self::shared`]), not slabbed. Valid while the
        /// view lives, for staging writes and in-engine co-processor storage;
        /// floating values read out to a caller leave through `read_with` or
        /// `canonicalize_in_place` so NaN bits are never exposed.
        pub fn host_address(&self) -> Option<std::ptr::NonNull<u8>> {
            match &self.backing {
                TensorBacking::Slabs { .. } => return None,
                TensorBacking::Plain { .. } => {}
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
            let info = representation_info(self.representation);
            if matches!(info.kind, RepresentationKind::Dense(dtype) if dtype.is_float()) {
                return Err(TensorError::FloatingSharedRead {
                    representation: info.name,
                });
            }
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
            let info = representation_info(self.representation);
            if matches!(info.kind, RepresentationKind::Dense(dtype) if dtype.is_float()) {
                return Err(TensorError::FloatingSharedRead {
                    representation: info.name,
                });
            }
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
            charges: &Charges,
            representation: RepresentationId,
            extents: &[u64],
        ) -> Result<Self, TensorError> {
            Self::canonical_in(charges, representation, extents, Charges::allocate_zeroed)
        }

        /// A zero-filled tensor whose records' table is formed with it,
        /// through `charges`, and held with its storage: a record port then
        /// binds a view of a run of its records (a streamed table's landed
        /// slots) through that table, allocating no other
        /// (`Allocation::view_table`).
        pub fn zeros_recorded(
            charges: &Charges,
            representation: RepresentationId,
            extents: &[u64],
        ) -> Result<Self, TensorError> {
            let tensor = Self::zeros(charges, representation, extents)?;
            tensor.forming_records(charges)?;
            Ok(tensor)
        }

        /// Allocate a canonical tensor without writing its storage. The
        /// caller must fully initialize all bytes, including representation
        /// padding, before any host read or device input use.
        pub unsafe fn uninitialized(
            charges: &Charges,
            representation: RepresentationId,
            extents: &[u64],
        ) -> Result<Self, TensorError> {
            Self::canonical_in(charges, representation, extents, Charges::allocate)
        }

        /// A canonical tensor over the storage `allocate` makes for it
        /// through `charges`.
        fn canonical_in(
            charges: &Charges,
            representation: RepresentationId,
            extents: &[u64],
            allocate: impl FnOnce(&Charges, u64, u64) -> Result<Arc<Allocation>, ExecutionError>,
        ) -> Result<Self, TensorError> {
            let device = charges.device();
            if !device.supports_representation(representation) {
                return Err(TensorError::UnsupportedRepresentation {
                    backend: device.info().backend(),
                    representation: representation_info(representation).name,
                });
            }
            let layout = layout::canonical(representation, extents)?;
            let allocation = allocate(charges, layout.byte_len, layout.alignment)?;
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
            charges: &Charges,
            representation: RepresentationId,
            extents: &[u64],
            bytes: &[u8],
        ) -> Result<Self, TensorError> {
            Self::from_host_in(charges, representation, extents, bytes, None)
        }

        pub(crate) fn from_host_in(
            charges: &Charges,
            representation: RepresentationId,
            extents: &[u64],
            bytes: &[u8],
            access: Option<Arc<crate::driver::AccessGroup>>,
        ) -> Result<Self, TensorError> {
            let device = charges.device();
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
            let allocation = charges.allocate_in(layout.byte_len, layout.alignment, access)?;
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
            Self::new_plain(
                device,
                allocation,
                byte_offset,
                byte_len,
                representation,
                extents,
                strides,
                None,
            )
        }

        #[allow(clippy::too_many_arguments)]
        fn new_plain(
            device: Arc<DeviceInner>,
            allocation: Arc<Allocation>,
            byte_offset: u64,
            byte_len: u64,
            representation: RepresentationId,
            extents: Vec<u64>,
            strides: Vec<u64>,
            hold: Option<Arc<SlabHold>>,
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
                backing: TensorBacking::Plain { hold },
                records: OnceLock::new(),
            }
        }
        pub fn read_to_host(&self) -> Result<Vec<u8>, TensorError> {
            match &self.backing {
                TensorBacking::Slabs { view: slabs, .. } => {
                    let length = usize::try_from(self.byte_len).map_err(|_| {
                        ExecutionError::AllocationFailed(
                            "slab tensor is too large for a host byte vector".into(),
                        )
                    })?;
                    let slab_bytes = usize::try_from(slabs.slab_bytes).map_err(|_| {
                        ExecutionError::AllocationFailed("slab exceeds host byte domain".into())
                    })?;
                    let mut bytes = vec![0u8; length];
                    for (index, region) in slabs.regions(self.representation)?.iter().enumerate() {
                        if let Some(region) = region {
                            let start = index * slab_bytes;
                            let keep = slab_bytes.min(length - start);
                            bytes[start..start + keep]
                                .copy_from_slice(&region.read_to_host()?[..keep]);
                        }
                    }
                    Ok(bytes)
                }
                TensorBacking::Plain { .. } => {
                    let access = self.allocation.acquire(false)?;
                    self.read_with(&access)
                }
            }
        }
        pub(crate) fn read_with(
            &self,
            access: &crate::driver::AllocationPermit,
        ) -> Result<Vec<u8>, TensorError> {
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
            // The one way a tensor's bytes leave the engine: every NaN reads out as the registry's
            // one NaN (`layout::canonicalize_in_place`).
            crate::layout::canonicalize_in_place(self.representation, &mut bytes);
            Ok(bytes)
        }
        pub fn write_from_host(&self, bytes: &[u8]) -> Result<(), TensorError> {
            match &self.backing {
                TensorBacking::Slabs { view: slabs, .. } => {
                    let actual = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
                    if actual != self.byte_len {
                        return Err(TensorError::HostByteLength {
                            expected: self.byte_len,
                            actual,
                        });
                    }
                    let regions = slabs.regions(self.representation)?;
                    if regions.iter().any(Option::is_none) {
                        return Err(TensorError::SlabLayout(
                            "host write reaches a free slab".into(),
                        ));
                    }
                    let slab_bytes = usize::try_from(slabs.slab_bytes).map_err(|_| {
                        TensorError::SlabLayout("slab exceeds host byte domain".into())
                    })?;
                    for (index, region) in regions.iter().enumerate() {
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
                    Ok(())
                }
                TensorBacking::Plain { .. } => self.write_whole(bytes, false),
            }
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
            match &self.backing {
                TensorBacking::Slabs { .. } => {
                    return Err(TensorError::SlabLayout(
                        "a chunked upload requires one contiguous allocation".into(),
                    ));
                }
                TensorBacking::Plain { .. } => {}
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
            match &self.backing {
                TensorBacking::Plain { .. } => self.allocation.identity(),
                TensorBacking::Slabs { alias_identity, .. } => *alias_identity,
            }
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
            let paths = self.allocation.storage_path_charges();
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
        fn slab_subview(
            &self,
            slabs: &SlabView,
            alias_identity: u64,
            first: usize,
            count: usize,
            rows: u64,
        ) -> Result<Self, TensorError> {
            let mut extents = self.extents.clone();
            extents[0] = rows;
            let layout = layout::canonical(self.representation, &extents)?;
            Ok(Self {
                device: self.device.clone(),
                allocation: self.allocation.clone(),
                byte_offset: slabs.slots.entry(slabs.region, first),
                byte_len: layout.byte_len,
                representation: self.representation,
                extents,
                strides: layout.strides,
                backing: TensorBacking::Slabs {
                    view: SlabView::new(
                        slabs.slots.clone(),
                        slabs.region,
                        first,
                        count,
                        slabs.rows_per_slab,
                        slabs.slab_extents.clone(),
                        slabs.slab_bytes,
                    ),
                    alias_identity,
                },
                records: OnceLock::new(),
            })
        }

        pub fn slice_leading(&self, start: u64, end: u64) -> Result<Self, TensorError> {
            match &self.backing {
                TensorBacking::Slabs {
                    view: slabs,
                    alias_identity,
                } => {
                    if start > end || end > self.extents[0] {
                        return Err(TensorError::SliceOutOfBounds {
                            extent: self.extents[0],
                            start,
                            end,
                        });
                    }
                    // An empty view names no slab: a zero-slot slab view anchored
                    // at its slot boundary in the region's address table.
                    if start == end {
                        let local = usize::try_from(start / slabs.rows_per_slab)
                            .unwrap_or(slabs.count)
                            .min(slabs.count);
                        return self.slab_subview(
                            slabs,
                            *alias_identity,
                            slabs.first + local,
                            0,
                            0,
                        );
                    }
                    // A span of whole slabs starting on a slab boundary (including
                    // a partial final slab at the view's end) stays a slab view
                    // over those slots' table entries, holding no granule.
                    if start % slabs.rows_per_slab == 0
                        && (end % slabs.rows_per_slab == 0 || end == self.extents[0])
                    {
                        let local = usize::try_from(start / slabs.rows_per_slab).map_err(|_| {
                            TensorError::SlabLayout("slab index exceeds host domain".into())
                        })?;
                        let count = usize::try_from((end - start).div_ceil(slabs.rows_per_slab))
                            .map_err(|_| {
                                TensorError::SlabLayout("slab count exceeds host domain".into())
                            })?;
                        return self.slab_subview(
                            slabs,
                            *alias_identity,
                            slabs.first + local,
                            count,
                            end - start,
                        );
                    }
                    if start / slabs.rows_per_slab != (end - 1) / slabs.rows_per_slab {
                        return Err(TensorError::SlabLayout(
                            "slice crosses a slab boundary".into(),
                        ));
                    }
                    let local = usize::try_from(start / slabs.rows_per_slab).map_err(|_| {
                        TensorError::SlabLayout("slab index exceeds host domain".into())
                    })?;
                    let region = slabs
                        .slots
                        .region_view(
                            slabs.first + local,
                            slabs.region,
                            self.representation,
                            &slabs.slab_extents,
                        )?
                        .ok_or_else(|| TensorError::SlabLayout("slice refers to a free slab".into()))?;
                    let local_start = start % slabs.rows_per_slab;
                    region.slice_leading(local_start, local_start + end - start)
                }
                TensorBacking::Plain { hold } => self.view(
                    &crate::api::kernel::ViewOperation::LeadingSlice { start, end },
                    hold.clone(),
                ),
            }
        }
        pub fn reshape(&self, extents: &[u64]) -> Result<Self, TensorError> {
            match &self.backing {
                TensorBacking::Slabs {
                    view: slabs,
                    alias_identity,
                } => {
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
                    let slab_extents = std::iter::once(slabs.rows_per_slab)
                        .chain(extents.iter().skip(1).copied())
                        .collect::<Vec<_>>();
                    Ok(Self {
                        device: self.device.clone(),
                        allocation: self.allocation.clone(),
                        byte_offset: self.byte_offset,
                        byte_len: view.byte_len,
                        representation: self.representation,
                        extents: view.extents,
                        strides: view.strides,
                        backing: TensorBacking::Slabs {
                            view: SlabView::new(
                                slabs.slots.clone(),
                                slabs.region,
                                slabs.first,
                                slabs.count,
                                slabs.rows_per_slab,
                                slab_extents,
                                slabs.slab_bytes,
                            ),
                            alias_identity: *alias_identity,
                        },
                        records: OnceLock::new(),
                    })
                }
                TensorBacking::Plain { hold } => self.view(
                    &crate::api::kernel::ViewOperation::Reshape {
                        extents: extents.to_vec(),
                    },
                    hold.clone(),
                ),
            }
        }
        /// This plain tensor seen through one view operation (`layout::apply_view`).
        fn view(
            &self,
            operation: &crate::api::kernel::ViewOperation,
            hold: Option<Arc<SlabHold>>,
        ) -> Result<Self, TensorError> {
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
            Ok(Self::new_plain(
                self.device.clone(),
                self.allocation.clone(),
                view.byte_offset,
                view.byte_len,
                self.representation,
                view.extents,
                view.strides,
                hold,
            ))
        }
        /// The view's range in its alias space. A slab view's alias space is
        /// its region's logical tensor, where slot `first` starts at
        /// `first * slab_bytes`; its `byte_offset` is where its address
        /// table's entries lie, what it binds, not where its elements do.
        pub fn descriptor(&self) -> TensorDescriptor {
            let (allocation, byte_offset) = match &self.backing {
                TensorBacking::Plain { .. } => (self.allocation.identity(), self.byte_offset),
                TensorBacking::Slabs {
                    view,
                    alias_identity,
                } => (*alias_identity, view.first as u64 * view.slab_bytes),
            };
            TensorDescriptor {
                device: self.device.kind.identity(),
                representation: self.representation,
                extents: self.extents.clone(),
                strides: self.strides.clone(),
                allocation,
                byte_offset,
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
            let tensor = TensorInner::from_host(&crate::memory::Charges::tally(&device), registry::dense(DType::F32), &[count], &vec![0; count as usize * 4]).unwrap();
            // Finite elements: a NaN reads out as the registry's one NaN.
            let bytes = (0..count * 4).map(|i| if i % 4 == 3 { 0x3f } else { i as u8 }).collect::<Vec<_>>();
            let half = bytes.len() / 2;
            tensor.write_chunks(|write| {
                write(half as u64, &bytes[half..])?;
                write(0, &bytes[..half])
            })
            .unwrap();
            assert_eq!(tensor.read_to_host().unwrap(), bytes);
            assert!(tensor.write_chunks(|write| write(1, &bytes)).is_err());
        }

        /// Every NaN of a dense floating tensor the host gives, of any sign and payload, reads back
        /// as the registry's one NaN: through a whole write, an in-place write and a chunked upload
        /// whose chunks end inside elements.
        #[test]
        fn host_nans_read_back_as_the_registrys_one_nan() {
            let devices = crate::devices::Devices::discover(&[]).unwrap();
            let device = devices.open_backend(registry::BackendName::Cpu).unwrap();
            let words = |dtype: DType, odd: &[u32], one: u32, finite: u32| -> (Vec<u8>, Vec<u8>) {
                let width = if dtype == DType::F32 { 4 } else { 2 };
                let (mut given, mut held) = (Vec::new(), Vec::new());
                for bits in odd.iter().copied().chain([finite]) {
                    given.extend_from_slice(&bits.to_le_bytes()[..width]);
                    held.extend_from_slice(&if bits == finite { bits } else { one }.to_le_bytes()[..width]);
                }
                (given, held)
            };
            for (dtype, odd, one, finite) in [
                (DType::F32, vec![0x7f80_0001, 0xffc0_0000, 0x7fff_ffff, 0xff80_0001, 0x7fc0_0000], 0x7fc0_0000, 0x3f80_0000),
                (DType::BF16, vec![0x7f81, 0xffc0, 0x7fff, 0xff81, 0x7fc0], 0x7fc0, 0x3f80),
                (DType::F16, vec![0x7c01, 0xfe00, 0x7fff, 0xfc01, 0x7e00], 0x7e00, 0x3c00),
            ] {
                let (given, held) = words(dtype, &odd, one, finite);
                let count = odd.len() as u64 + 1;
                let representation = registry::dense(dtype);
                let whole = TensorInner::from_host(&crate::memory::Charges::tally(&device), representation, &[count], &given).unwrap();
                assert_eq!(whole.read_to_host().unwrap(), held, "{dtype:?} from_host");
                let written = TensorInner::zeros(&crate::memory::Charges::tally(&device), representation, &[count]).unwrap();
                written.write_from_host(&given).unwrap();
                assert_eq!(written.read_to_host().unwrap(), held, "{dtype:?} write_from_host");
                for chunk in [1, 3, 5] {
                    let chunked = TensorInner::zeros(&crate::memory::Charges::tally(&device), representation, &[count]).unwrap();
                    chunked
                        .write_chunks(|write| {
                            for (index, bytes) in given.chunks(chunk).enumerate() {
                                write((index * chunk) as u64, bytes)?;
                            }
                            Ok(())
                        })
                        .unwrap();
                    assert_eq!(chunked.read_to_host().unwrap(), held, "{dtype:?} chunks of {chunk}");
                }
            }
        }

        /// A NaN a kernel leaves in a dense floating tensor, whatever its sign and payload, reads back
        /// as the registry's one NaN: the way out of the engine canonicalizes, so a kernel's own NaN
        /// bits (a sign a negation flipped, an operation's default NaN) are unobservable.
        #[test]
        fn device_nans_leave_as_the_registrys_one_nan() {
            let device = crate::devices::Devices::discover(&[])
                .unwrap()
                .open_backend(registry::BackendName::Cpu)
                .unwrap();
            for (dtype, left, one, finite) in [
                (DType::F32, vec![0xffc0_0000u32, 0x7fff_ffff, 0xff80_0001, 0x7f80_0001], 0x7fc0_0000u32, 0x3f80_0000u32),
                (DType::BF16, vec![0xffc0, 0x7fff, 0xff81, 0x7f81], 0x7fc0, 0x3f80),
                (DType::F16, vec![0xfe00, 0x7fff, 0xfc01, 0x7c01], 0x7e00, 0x3c00),
            ] {
                let width = if dtype == DType::F32 { 4 } else { 2 };
                let tensor = TensorInner::zeros(&crate::memory::Charges::tally(&device), registry::dense(dtype), &[left.len() as u64 + 1]).unwrap();
                let raw = left.iter().chain([&finite]).flat_map(|bits| bits.to_le_bytes()[..width].to_vec()).collect::<Vec<_>>();
                tensor.allocation.storage().write(0, &raw).unwrap();
                let held = left.iter().map(|_| one).chain([finite]).flat_map(|bits| bits.to_le_bytes()[..width].to_vec()).collect::<Vec<_>>();
                assert_eq!(tensor.read_to_host().unwrap(), held, "{dtype:?}");
            }
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
            let tensor = TensorInner::zeros(&crate::memory::Charges::tally(&device), registry::dense(DType::U32), &[4, 8]).unwrap();
            let charged = tensor.allocation.charged_bytes();
            let first = tensor.slice_leading(1, 3).unwrap().forming_records(&crate::memory::Charges::tally(&device)).unwrap();
            let again = tensor.slice_leading(1, 3).unwrap().forming_records(&crate::memory::Charges::tally(&device)).unwrap();
            let later = tensor.slice_leading(2, 4).unwrap().forming_records(&crate::memory::Charges::tally(&device)).unwrap();
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
            let tensor = TensorInner::zeros(&crate::memory::Charges::tally(&device), registry::dense(DType::U32), &[4, 8]).unwrap();
            let rows = tensor.forming_records(&crate::memory::Charges::tally(&device)).unwrap();
            let charged = tensor.allocation.charged_bytes();
            let run = tensor
                .slice_leading(1, 3)
                .unwrap()
                .reshape(&[1, 2, 8])
                .unwrap()
                .forming_records(&crate::memory::Charges::tally(&device))
                .unwrap();
            assert!(Arc::ptr_eq(&rows.table, &run.table));
            assert_eq!(run.table_offset, 8);
            assert_eq!(tensor.allocation.charged_bytes(), charged);
            let pairs = tensor.reshape(&[2, 2, 8]).unwrap().forming_records(&crate::memory::Charges::tally(&device)).unwrap();
            assert!(!Arc::ptr_eq(&rows.table, &pairs.table));
            assert_eq!(tensor.allocation.charged_bytes(), charged + 2 * 8);
        }

        #[test]
        fn a_view_reports_only_its_owned_byte_interval() {
            let device = crate::devices::Devices::discover(&[])
                .unwrap()
                .open_backend(registry::BackendName::Cpu)
                .unwrap();
            let tensor = TensorInner::zeros(&crate::memory::Charges::tally(&device), registry::dense(DType::U32), &[4, 8]).unwrap();
            let view = tensor.slice_leading(1, 3).unwrap();
            assert_eq!(
                view.view_store_charges().unwrap(),
                vec![(device.topology().host_ram().id(), 64)]
            );
            assert_eq!(view.observe_storage().charged_bytes(), Some(128));
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
            let tensor = TensorInner::zeros(&crate::memory::Charges::tally(&device), registry::dense(DType::U32), &[4, 8]).unwrap();
            tensor.forming_records(&crate::memory::Charges::tally(&device)).unwrap();
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

        /// A dense floating tensor refuses raw shared reads, word accesses and
        /// re-interpretation as another representation, while admitting
        /// same-representation regions and, as carried regions, records
        /// lent to a non-dense weight form.
        #[test]
        fn floating_views_refuse_shared_word_and_reinterpreted_region_bypasses() {
            use crate::api::{RegionError, TensorError};
            let device = crate::devices::Devices::discover(&[])
                .unwrap()
                .open_backend(registry::BackendName::Cpu)
                .unwrap();
            for dtype in [DType::F32, DType::BF16, DType::F16] {
                let repr = registry::dense(dtype);
                let shared = TensorInner::shared(&crate::memory::Charges::tally(&device), repr, &[8]).unwrap();
                let mut scratch = [0u8; 4];
                assert!(matches!(
                    unsafe { shared.read_shared(0, &mut scratch) },
                    Err(TensorError::FloatingSharedRead { .. })
                ));
                assert!(matches!(
                    shared.load_acquire(0),
                    Err(TensorError::FloatingSharedRead { .. })
                ));
                assert!(matches!(
                    shared.load_acquire_u64(0),
                    Err(TensorError::FloatingSharedRead { .. })
                ));
                assert!(matches!(
                    shared.store_release(0, 1),
                    Err(TensorError::FloatingSharedRead { .. })
                ));
                assert!(matches!(
                    shared.shared_words(),
                    Err(TensorError::FloatingSharedRead { .. })
                ));
                let empty = TensorInner::shared(&crate::memory::Charges::tally(&device), repr, &[0]).unwrap();
                assert!(matches!(
                    empty.shared_words(),
                    Err(TensorError::FloatingSharedRead { .. })
                ));
            }

            let f32_repr = registry::dense(DType::F32);
            let bf16_repr = registry::dense(DType::BF16);
            let i32_repr = registry::dense(DType::I32);
            let bool_repr = registry::dense(DType::Bool);
            let q8_repr = registry::representation("q8g32").unwrap();
            let float_view = TensorInner::shared(&crate::memory::Charges::tally(&device), f32_repr, &[64]).unwrap();
            assert!(float_view.region(f32_repr, &[16], 0).is_ok());
            for bad in [i32_repr, bool_repr, bf16_repr, q8_repr] {
                assert!(
                    matches!(
                        float_view.region(bad, &[32], 0),
                        Err(TensorError::Region(RegionError::FloatingReinterpret { .. }))
                    ),
                    "expected FloatingReinterpret for {}",
                    registry::representation_info(bad).name
                );
            }

            let carrier = TensorInner::zeros(&crate::memory::Charges::tally(&device), bf16_repr, &[4, 2048]).unwrap();
            assert!(matches!(
                carrier.region(q8_repr, &[1, 64], 0),
                Err(TensorError::Region(RegionError::FloatingReinterpret { .. }))
            ));
            assert!(carrier.carried_region(q8_repr, &[1, 64], 0).is_ok());
            assert!(matches!(
                carrier.carried_region(i32_repr, &[32], 0),
                Err(TensorError::Region(RegionError::FloatingReinterpret { .. }))
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
        pub fn records_result(&self, ordinal: usize) -> bool {
            self.inner.records_result(ordinal)
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
        /// What each launch's program uses, in ordinal order
        /// ([`crate::native::NativePrepared::launch_programs`]).
        pub fn launch_programs(&self) -> Vec<Option<seismic_native_target::LaunchProgram>> {
            self.inner.launch_programs()
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
            charges: &crate::memory::Charges,
            rotation: Vec<EncodedArgs>,
        ) -> Result<crate::native::KernelPoint, super::CallError> {
            crate::native::KernelPoint::place(&self.inner, charges, rotation, Vec::new(), None)
        }
        /// The unrounded bytes placed in a standalone call's scratch
        /// regions for `args`: its sync region and its body together.
        pub fn standalone_scratch_bytes(
            &self,
            args: EncodedArgs,
        ) -> Result<u64, super::CallError> {
            self.inner.standalone_scratch_bytes(args)
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
        pub workflow: u64,
        pub node: u32,
        pub result: u32,
    }

    #[derive(Clone)]
    pub enum EncodedWorkflowArgument {
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
    pub enum WorkflowTensorArgument {
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
        pub fn arguments(&self) -> &[EncodedWorkflowArgument] {
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
        #[doc(hidden)]
        pub fn new(workflow: u64, node: u32, count: u32) -> Self {
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
        /// The argument the engine takes in for this host scalar, as given: a NaN's bits are
        /// unobservable inside, and every scalar result leaves through `DecodedValue::scalar`.
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
        Scalar(ScalarResult),
    }
    impl DecodedValue {
        /// A scalar result as it leaves the engine: a floating NaN is the registry's one NaN
        /// (`layout::canonical_scalar`). The only constructor of a `ScalarResult`.
        pub(crate) fn scalar(value: ArgumentValue) -> Self {
            Self::Scalar(ScalarResult(crate::layout::canonical_scalar(value)))
        }
    }
    /// A scalar result out of the engine, as `DecodedValue::scalar` built it.
    #[derive(Clone, Debug)]
    pub struct ScalarResult(ArgumentValue);
    impl ScalarResult {
        pub fn value(&self) -> &ArgumentValue {
            &self.0
        }
        pub fn into_value(self) -> ArgumentValue {
            self.0
        }
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
                Some(DecodedValue::Scalar(value)) => value.into_value(),
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
            let completion = loop {
                let mut state = self
                    .state
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                match std::mem::replace(&mut *state, CompletionState::Running) {
                    CompletionState::Complete(result) => {
                        let output = result.clone();
                        *state = CompletionState::Complete(result);
                        return output;
                    }
                    CompletionState::Running => {
                        drop(
                            self.changed
                                .wait(state)
                                .unwrap_or_else(std::sync::PoisonError::into_inner),
                        );
                    }
                    CompletionState::Panicked => {
                        *state = CompletionState::Panicked;
                        panic!("workflow completion previously panicked");
                    }
                    CompletionState::Pending(completion) => break completion,
                }
            };

            // Device completion must not run under a host mutex. Publish the
            // terminal state afterward so concurrent resolvers wait without
            // duplicating submission or synchronization.
            let completed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(completion));
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            match completed {
                Ok(result) => {
                    *state = CompletionState::Complete(result.clone());
                    self.changed.notify_all();
                    result
                }
                Err(payload) => {
                    *state = CompletionState::Panicked;
                    self.changed.notify_all();
                    drop(state);
                    std::panic::resume_unwind(payload);
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

    /// A workflow on `charges`' device, every allocation its runs make
    /// charged to them.
    pub fn workflow(charges: &crate::memory::Charges) -> Result<WorkflowDraftAny, super::WorkflowError> {
        charges
            .device()
            .kind
            .workflow(charges)
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
    /// Start a feedback preparation on `charges`' device, its trials'
    /// inputs and runs allocated through `charges`.
    pub fn start_feedback<'a>(
        module: &CheckedModule,
        entry: EntryId,
        bindings: ElementBindings,
        charges: &'a crate::memory::Charges,
        precision: seismic_lang::precision::PrecisionPolicy,
        options: seismic_compiler::feedback::FeedbackOptions,
    ) -> Result<(FeedbackPreparation<'a>, PreparedAny), PrepareError> {
        charges
            .device()
            .kind
            .start_feedback(module, entry, bindings, charges, precision, options)
            .map(|(inner, kernel)| (FeedbackPreparation { inner }, PreparedAny { inner: kernel }))
    }

    /// Prepare `entry` on `charges`' device, a feedback preparation's
    /// trials allocated through `charges`.
    pub fn prepare(
        module: &CheckedModule,
        entry: EntryId,
        bindings: ElementBindings,
        charges: &crate::memory::Charges,
        options: seismic_compiler::feedback::PreparationOptions,
    ) -> Result<PreparedAny, PrepareError> {
        let logical = module
            .entry(entry, &bindings)
            .map_err(PrepareError::Source)?;
        prepare_logical(logical, bindings, charges, options)
    }

    /// Prepare an already checked computation through the ordinary compiler,
    /// cache and device-admission path. Bound numerical recipes enter its
    /// logical identity; they do not add invocation arguments or another cache.
    pub(crate) fn prepare_logical(
        logical: seismic_lang::entry::LogicalEntry,
        bindings: ElementBindings,
        charges: &crate::memory::Charges,
        options: seismic_compiler::feedback::PreparationOptions,
    ) -> Result<PreparedAny, PrepareError> {
        charges
            .device()
            .kind
            .prepare(logical, bindings, charges, options)
            .map(|inner| PreparedAny { inner })
    }
    /// A call of `kernel`, every allocation it makes charged to `charges`.
    pub fn call(kernel: &Arc<PreparedAny>, charges: &crate::memory::Charges, args: EncodedArgs) -> Result<DecodedResults, CallError> {
        kernel.inner.call(charges, args)
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
        charges: &crate::memory::Charges,
        args: EncodedArgs,
    ) -> Result<DecodedResults, CallError> {
        kernel.inner.call(charges, args)
    }

    /// [`call_native`] of a call whose awaited words already hold their
    /// answers: valid on every backend (`NativePrepared::call_answered`).
    pub fn call_native_answered(
        kernel: &Arc<NativePreparedAny>,
        charges: &crate::memory::Charges,
        args: EncodedArgs,
    ) -> Result<DecodedResults, CallError> {
        kernel.inner.call_answered(charges, args)
    }

    pub fn call_native_into(
        kernel: &Arc<NativePreparedAny>,
        charges: &crate::memory::Charges,
        args: EncodedArgs,
        outputs: EncodedOutputs,
    ) -> Result<DecodedResults, CallError> {
        kernel.inner.call_into(charges, args, outputs)
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
        /// A batch whose calls' scratch is charged to `charges`.
        pub fn new(charges: &crate::memory::Charges) -> Self {
            Self {
                inner: crate::native::NativeTensorBatch::new(charges.clone()),
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
        charges: &crate::memory::Charges,
        args: EncodedArgs,
        commit: impl FnOnce(),
    ) -> Result<DecodedResults, CallError> {
        kernel.inner.call_with_commit(charges, args, commit)
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

        #[test]
        fn slab_subviews_scope_reach_occupied_bytes_and_copy_of() {
            let catalog = crate::devices::Devices::discover(&[]).unwrap();
            let id = catalog
                .resolve(seismic_topology::ExecutorSelector::HostCpu)
                .unwrap();
            let device = catalog.open(id).unwrap();
            let topology = catalog.topology();
            let info = topology.executor(id).unwrap();
            let store = topology
                .first_store(info, seismic_topology::Contract::DeviceOnly)
                .unwrap();
            let identity = topology.store_identity(store).unwrap();
            let entry: Arc<dyn crate::memory::Entry> =
                Arc::new(crate::memory::Tally::unbounded(identity));
            let pool = crate::pool::Pool::open(
                &crate::pool::PoolCharges::one(&crate::memory::Charges::in_store(&device, seismic_topology::Contract::DeviceOnly, entry)),
                seismic_topology::Contract::DeviceOnly,
                1 << 16,
                1 << 18,
            )
            .unwrap();
            let f32_rep = seismic_lang::registry::dense(seismic_lang::types::DType::F32);
            let geometry = crate::pool::SlabGeometry {
                rows_per_slab: 4,
                logical_rows: 14,
                capacity: 4,
                slab_bytes: crate::pool::BLOCK_ALIGNMENT,
                slab_alignment: crate::pool::BLOCK_ALIGNMENT,
                regions: vec![(f32_rep, vec![2], 0)],
            };
            let slabs = crate::pool::SlabStore::in_pool(&crate::memory::Charges::tally(&device), &pool, geometry, &[]).unwrap();
            for _ in 0..4 {
                slabs.add_slab(crate::pool::SlabFill::Zeroed).unwrap();
            }
            // Free slot 0 and slot 2: slots 1 and 3 stay backed.
            assert!(slabs.free_slab(0).unwrap());
            assert!(slabs.free_slab(2).unwrap());

            let logical = slabs.logical_region(0).unwrap();
            let sub = logical.slice_leading(4, 14).unwrap(); // slots 1..4 (3 slots, 2 backed)
            let reach = sub.slab_reach().unwrap();
            assert_eq!(
                reach.table_bytes,
                seismic_lang::registry::record_table_entry(3)
            );
            assert_eq!(reach.regions.len(), 3);
            assert_eq!(reach.regions.iter().flatten().count(), 2);
            assert_eq!(sub.occupied_bytes(), 2 * (4 * 2 * 4));

            let img = sub.image().unwrap();
            let copy = sub.copy_of(&crate::memory::Charges::tally(&device), &img).unwrap();
            assert_eq!(copy.extents(), &[10, 2]);
            assert_eq!(copy.occupied_bytes(), 2 * (4 * 2 * 4));
            assert_eq!(copy.read_to_host().unwrap(), sub.read_to_host().unwrap());

            // Empty sub-view (`count == 0`): 0 table bytes, 0 regions, round-trips `copy_of`.
            let empty = sub.slice_leading(4, 4).unwrap();
            let empty_reach = empty.slab_reach().unwrap();
            assert_eq!(empty_reach.table_bytes, 0);
            assert!(empty_reach.regions.is_empty());
            assert_eq!(empty.occupied_bytes(), 0);
            let empty_copy = empty.copy_of(&crate::memory::Charges::tally(&device), &empty.image().unwrap()).unwrap();
            assert_eq!(empty_copy.extents(), &[0, 2]);
            assert!(empty_copy.read_to_host().unwrap().is_empty());
        }

        /// Binding compares the bytes device work reaches
        /// (`Reaching::meets`): a slab view's slot meets that slot's own
        /// region view and no other slot's bytes, and a free slot's rows
        /// meet nothing.
        #[test]
        fn reaching_compares_the_slab_bytes_bound() {
            let catalog = crate::devices::Devices::discover(&[]).unwrap();
            let id = catalog
                .resolve(seismic_topology::ExecutorSelector::HostCpu)
                .unwrap();
            let device = catalog.open(id).unwrap();
            let topology = catalog.topology();
            let info = topology.executor(id).unwrap();
            let store = topology
                .first_store(info, seismic_topology::Contract::DeviceOnly)
                .unwrap();
            let identity = topology.store_identity(store).unwrap();
            let entry: Arc<dyn crate::memory::Entry> =
                Arc::new(crate::memory::Tally::unbounded(identity));
            let pool = crate::pool::Pool::open(
                &crate::pool::PoolCharges::one(&crate::memory::Charges::in_store(&device, seismic_topology::Contract::DeviceOnly, entry)),
                seismic_topology::Contract::DeviceOnly,
                1 << 16,
                1 << 18,
            )
            .unwrap();
            let f32_rep = seismic_lang::registry::dense(seismic_lang::types::DType::F32);
            let geometry = crate::pool::SlabGeometry {
                rows_per_slab: 4,
                logical_rows: 16,
                capacity: 4,
                slab_bytes: crate::pool::BLOCK_ALIGNMENT,
                slab_alignment: crate::pool::BLOCK_ALIGNMENT,
                regions: vec![(f32_rep, vec![2], 0)],
            };
            let slabs = crate::pool::SlabStore::in_pool(&crate::memory::Charges::tally(&device), &pool, geometry, &[]).unwrap();
            for _ in 0..4 {
                slabs.add_slab(crate::pool::SlabFill::Zeroed).unwrap();
            }
            assert!(slabs.free_slab(2).unwrap());
            let logical = slabs.logical_region(0).unwrap();
            let region = slabs.slab_region(1, 0).unwrap().unwrap();
            let (whole, one) = (crate::api::tensor::Reaching::new(&logical), crate::api::tensor::Reaching::new(&region));
            // One slot's rows of the view: 4 rows of 2 f32.
            let slot = 4 * 2 * 4;
            let rows = |index: u64| index * slot..(index + 1) * slot;
            assert!(whole.meets(rows(1), &whole, rows(1)));
            assert!(!whole.meets(rows(0), &whole, rows(1)), "distinct slots lie apart");
            assert!(whole.meets(rows(1), &one, 0..region.byte_len()), "a slot meets its region view");
            assert!(!whole.meets(rows(0), &one, 0..region.byte_len()));
            assert!(!whole.meets(rows(3), &one, 0..region.byte_len()));
            // Slot 2 is free: its rows reach nothing, not even the whole view.
            assert!(!whole.meets(rows(2), &whole, 0..4 * slot), "a free slot reaches nothing");
        }

        /// A pool's absent granule, which every free slot names and no slab
        /// holds, is charged to its own entry as the pool opens, and its
        /// blocks to theirs: what the blocks' entry holds is granules a slab
        /// claims, and nothing more.
        #[test]
        fn a_pools_absent_granule_is_charged_apart_from_its_blocks() {
            let catalog = crate::devices::Devices::discover(&[]).unwrap();
            let id = catalog
                .resolve(seismic_topology::ExecutorSelector::HostCpu)
                .unwrap();
            let device = catalog.open(id).unwrap();
            let topology = catalog.topology();
            let info = topology.executor(id).unwrap();
            let store = topology
                .first_store(info, seismic_topology::Contract::DeviceOnly)
                .unwrap();
            let identity = topology.store_identity(store).unwrap();
            let (blocks, absent) = (
                crate::memory::Tally::unbounded(identity.clone()),
                crate::memory::Tally::unbounded(identity),
            );
            let charges = |tally: &crate::memory::Tally| {
                crate::memory::Charges::in_store(&device, seismic_topology::Contract::DeviceOnly, Arc::new(tally.clone()))
            };
            let pool = crate::pool::Pool::open(
                &crate::pool::PoolCharges::apart(charges(&blocks), charges(&absent)),
                seismic_topology::Contract::DeviceOnly,
                1 << 16,
                1 << 18,
            )
            .unwrap();
            assert_eq!((blocks.held(), absent.held()), (0, 1 << 16), "the open allocates the absent granule alone");
            pool.grow().unwrap();
            assert_eq!((blocks.held(), absent.held()), (1 << 18, 1 << 16), "a block is the blocks' entry's");
            assert_eq!(pool.held_bytes(), blocks.held() + absent.held());
        }

        #[test]
        fn cached_slab_reach_does_not_retain_removed_pool_blocks() {
            let catalog = crate::devices::Devices::discover(&[]).unwrap();
            let id = catalog
                .resolve(seismic_topology::ExecutorSelector::HostCpu)
                .unwrap();
            let device = catalog.open(id).unwrap();
            let topology = catalog.topology();
            let info = topology.executor(id).unwrap();
            let store = topology
                .first_store(info, seismic_topology::Contract::DeviceOnly)
                .unwrap();
            let identity = topology.store_identity(store).unwrap();
            let tally = crate::memory::Tally::unbounded(identity);
            let entry: Arc<dyn crate::memory::Entry> = Arc::new(tally.clone());
            let pool = crate::pool::Pool::open(
                &crate::pool::PoolCharges::one(&crate::memory::Charges::in_store(&device, seismic_topology::Contract::DeviceOnly, entry)),
                seismic_topology::Contract::DeviceOnly,
                1 << 16,
                1 << 18,
            )
            .unwrap();
            let absent = tally.held();
            assert_eq!(absent, 1 << 16);
            let f32_rep = seismic_lang::registry::dense(seismic_lang::types::DType::F32);
            let geometry = crate::pool::SlabGeometry {
                rows_per_slab: 4,
                logical_rows: 16,
                capacity: 4,
                slab_bytes: crate::pool::BLOCK_ALIGNMENT,
                slab_alignment: crate::pool::BLOCK_ALIGNMENT,
                regions: vec![(f32_rep, vec![2], 0)],
            };
            let slabs = crate::pool::SlabStore::in_pool(&crate::memory::Charges::tally(&device), &pool, geometry, &[]).unwrap();
            for _ in 0..4 {
                slabs.add_slab(crate::pool::SlabFill::Zeroed).unwrap();
            }
            assert_eq!(tally.held(), absent + (1 << 18));

            // Populate `SlabView.reach` on a live view, then free every slab without
            // refreshing `logical.slab_reach()`: `return_freed` must still drop the
            // idle pool block back to the store (`tally.held() == absent`).
            let logical = slabs.logical_region(0).unwrap();
            assert_eq!(logical.slab_reach().unwrap().regions.len(), 4);
            for slot in 0..4 {
                assert!(slabs.free_slab(slot).unwrap());
            }
            assert_eq!(pool.return_freed(1 << 18).unwrap(), 1 << 18);
            assert_eq!(tally.held(), absent, "stale cached SlabReach must not keep freed pool blocks alive");
            assert_eq!(logical.extents(), &[16, 2]);
        }
    }
}
