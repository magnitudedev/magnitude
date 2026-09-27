//! The public Seismic API (spec §14): devices, tensors, prepared kernels,
//! runtime-discovered functions, and the helpers generated bindings compose.
//! This crate re-exports and
//! composes; it contains no second implementation.
//!
//! Consumers import only this crate and `seismic-build`. Nothing here
//! exposes `PlanSpace`, `FrozenPlan`, `ExecutableVariant`, target domains,
//! solver decisions, raw backend buffers, binding indices, or manual arena
//! allocation (§14.6).
//!
//! W9-B owns device/tensor internals, W9-C owns invocation, W9-A owns the
//! generated-code contract. Dynamic callers use [`dynamic`] without implementing
//! the unsafe generated entry contract.

pub use seismic_compiler::errors::{
    CheckedBundleError, ExecutionError, InvocationError, PreparationError, TargetError,
};
pub use seismic_compiler::feedback::{
    EvaluationMethod, FeedbackOptions, FeedbackReport, InvocationRange, InvocationScope,
    PreparationOptions,
};
pub use seismic_lang::checked::{
    NativeComparison, NativeCondition, NativeImplementation, NativeLaunch, NativeNatExpr,
    NativeParameter, NativeScratch, NativeSpecialization, NativeSpecializationError,
};
pub use seismic_lang::precision::PrecisionPolicy;
/// Numerical comparison helpers used by validation frontends.
pub mod testing {
    pub use seismic_compiler::numerics::{compare_element, ElementComparison};
}
pub use seismic_runtime::artifacts::{ArtifactKey, ArtifactStore, DeviceOptions};
/// Replay of the tuning search against recorded surveys (development).
pub use seismic_runtime::native::replay;
pub use seismic_runtime::native::search::{
    search, Cost, Evaluator, ParameterValues, PointKey, SearchSettings, SearchSpace,
    SearchSpaceError, SearchStop, SearchTrace,
};
pub use seismic_runtime::native::trace::{
    host_seconds, SubmissionTrace, TraceDetail, TraceError, TracedLaunch, TracedSubmission,
};
pub use seismic_runtime::native::tune::{
    Configuration, ConfigurationRecord, DeclaredParameter, Exclusion, Outcome, PointMeasurement,
    PointRecord, ScreeningPoint, SearchPlan, Strategy, SurveyPlan, TuneError, TuningInitializer,
    TuningMethod, TuningResult, TuningTime, Validation,
};
pub use seismic_runtime::native::{MeasureOptions, Measurement, NativeArtifactIdentity};

/// Result of checking one native entry's element bindings without device
/// formation. Acceptance proves only checked source semantics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NativeBindingCheck {
    AcceptedByCheckedEntry,
    Unsupported(String),
}

/// One checked tensor port or result before device or native implementation formation.
/// The byte count uses the registered canonical representation layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeTensorMetadata {
    pub element: Element,
    pub extents: Vec<u64>,
    pub canonical_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NativeTensorParameterCheck {
    Checked(NativeTensorMetadata),
    Unsupported(String),
}

/// Checked result leaves in schema order; `None` is a scalar leaf. Native
/// graph formation still has to reject scalar results and validate the node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NativeTensorResultsCheck {
    Checked(Vec<Option<NativeTensorMetadata>>),
    Unsupported(String),
}

/// The CPU native ABI that generated bindings wrap. Kernel authors use the
/// generated `Context` of their entry instead.
pub mod native_cpu {
    pub use seismic_runtime::native::{
        CpuInvocation, CpuKernelFn, CpuLaunchVariants, CpuNativeKernels, CpuVariant,
    };
}

pub use seismic_lang::expr::{BigInt, BigUint};
pub use seismic_lang::registry::{BackendName, Layout};
pub use seismic_lang::types::DType;
/// The Seismic CPU library every CPU native kernel builds on.
pub use seismic_native_cpu as cpu;
pub use seismic_runtime::api::HostRegion;
pub use seismic_runtime::api::{CallError, OutputError, TensorError, WorkflowError};
mod slab;
pub use seismic_runtime::devices::{
    Availability, CapacityBasis, DeviceId, DeviceInfo, DeviceKind, DeviceMeasurements,
    DeviceMemory, DeviceMemoryInfo, DeviceMemoryStatus, DeviceSelector, DeviceTopology,
    DiscoveryDiagnostic, DiscoveryError, HeadroomBasis, HeadroomEstimate, HostMeasurements,
    HostMemoryStatus, LimitVisibility, MemoryPoolId, MemoryPoolInfo, MemoryPoolKind, MemoryUsage,
    ObservationError, OpenError, ProcessLimitKind, ProcessMemoryLimit, ResolveError,
    SelectorParseError,
};
pub use slab::{Slab, SlabLayout, SlabRegion, SlabTensor};

use seismic_compiler::prepared::ArgumentValue;
use seismic_lang::ids::RepresentationId;
use std::fmt;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Devices
// ---------------------------------------------------------------------------

/// The one device inventory: host RAM and backend devices, their memory
/// pools and identities, scoped observations, and device opening.
pub struct DeviceCatalog {
    inner: seismic_runtime::devices::Catalog,
}

impl DeviceCatalog {
    pub fn discover() -> Result<Self, DiscoveryError> {
        seismic_runtime::devices::Catalog::discover().map(|inner| Self { inner })
    }
    /// The current immutable inventory snapshot.
    pub fn topology(&self) -> Arc<DeviceTopology> {
        self.inner.topology()
    }
    /// Re-enumerates; identifiers stay valid unless the inventory changed.
    pub fn refresh(&self) -> Result<Arc<DeviceTopology>, DiscoveryError> {
        self.inner.refresh()
    }
    /// Resolves a same-machine selector, rejecting missing or ambiguous
    /// matches. Never substitutes another device.
    pub fn resolve(&self, selector: DeviceSelector) -> Result<DeviceId, ResolveError> {
        self.inner.resolve(selector)
    }
    pub fn open(&self, id: DeviceId) -> Result<Device, OpenError> {
        self.inner.open(id).map(|inner| Device { inner })
    }
    /// Open with `options` (for example an artifact store). A device already
    /// open is shared; asking it for a different store is an error.
    pub fn open_with(&self, id: DeviceId, options: DeviceOptions) -> Result<Device, OpenError> {
        self.inner
            .open_with(id, options)
            .map(|inner| Device { inner })
    }
    /// Low-level control: opens the first discovered device of a backend.
    /// Managed callers select by requirements and open by identity.
    pub fn open_backend(&self, backend: BackendName) -> Result<Device, OpenError> {
        self.inner
            .open_backend(backend)
            .map(|inner| Device { inner })
    }
    pub fn host_memory_status(&self) -> Result<HostMemoryStatus, ObservationError> {
        self.inner.host_memory_status()
    }
}

impl fmt::Debug for DeviceCatalog {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceCatalog")
            .field("topology", &self.topology())
            .finish()
    }
}

/// One open device. Cloning shares the device.
#[derive(Clone)]
pub struct Device {
    inner: Arc<seismic_runtime::api::device::DeviceInner>,
}

impl Device {
    pub fn same_device(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
    pub fn info(&self) -> &DeviceInfo {
        self.inner.info()
    }
    /// Complete capability summary acquired from this opened device.
    pub fn capabilities(&self) -> &[String] {
        self.inner.capabilities()
    }
    pub fn backend(&self) -> BackendName {
        self.inner.info().backend
    }
    /// Stable key of this device's model and configuration for native
    /// tuning records.
    pub fn tuning_identity(&self) -> String {
        self.inner.tuning_identity()
    }
    /// Record every native submission on this device until the returned
    /// trace is dropped (measurement only; see `TraceDetail`).
    pub fn trace_submissions(&self, detail: TraceDetail) -> Result<SubmissionTrace, TraceError> {
        SubmissionTrace::start(&self.inner, detail)
    }
    /// Begin a checked direct-native graph on this opened device.
    pub fn native_graph(&self) -> NativeGraph {
        NativeGraph {
            inner: seismic_runtime::native::graph::NativeGraphDraft::new(&self.inner),
        }
    }
    /// Begin an exact graph using a certified Seismic placement.
    pub fn native_graph_with_layout(&self, layout: &NativeGraphLayout) -> NativeGraph {
        NativeGraph {
            inner: seismic_runtime::native::graph::NativeGraphDraft::with_layout(
                &self.inner,
                layout.inner.clone(),
            ),
        }
    }
    /// An empty sequence of graph runs to submit together.
    pub fn native_sequence(&self) -> NativeGraphSequence {
        NativeGraphSequence {
            inner: seismic_runtime::native::graph::NativeGraphSequence::new(&self.inner),
        }
    }
    /// Seismic-owned charges and limits for this device and its pool.
    pub fn memory_usage(&self) -> MemoryUsage {
        self.inner.memory_usage()
    }
    /// Enforce a requested allocation budget on this device's allocations.
    pub fn set_memory_limit(&self, limit: Option<u64>) {
        self.inner.set_memory_limit(limit)
    }
    /// Samples this device's backend memory observation.
    pub fn memory_status(&self) -> Result<DeviceMemoryStatus, ObservationError> {
        self.inner.memory_status()
    }
    /// A planned workflow on this device; refused on a native-only backend.
    pub fn workflow(&self) -> Result<WorkflowDraft, WorkflowError> {
        seismic_runtime::api::kernel::workflow(&self.inner).map(|inner| WorkflowDraft { inner })
    }
    pub(crate) fn inner(&self) -> &Arc<seismic_runtime::api::device::DeviceInner> {
        &self.inner
    }
}

impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Device").field("info", self.info()).finish()
    }
}

// ---------------------------------------------------------------------------
// Elements and tensors
// ---------------------------------------------------------------------------

/// An element representation by registry identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Element(RepresentationId);

impl Element {
    /// The storage registered under a unique storage name (`q4k`,
    /// `q4k@rows16`, `f32`, `gguf_q4_k`).
    pub fn named(name: &str) -> Option<Element> {
        seismic_lang::registry::representation(name).map(Element)
    }
    /// Packed representation `representation` (`q4k`) in `layout`.
    pub fn stored(representation: &str, layout: Layout) -> Option<Element> {
        seismic_lang::registry::storage(representation, layout).map(Element)
    }
    /// Unique storage name.
    pub fn name(&self) -> &'static str {
        seismic_lang::registry::representation_info(self.0).name
    }
    /// The logical representation this storage encodes.
    pub fn representation(&self) -> &'static str {
        seismic_lang::registry::representation_info(self.0).representation
    }
    pub fn layout(&self) -> Layout {
        seismic_lang::registry::representation_info(self.0).layout
    }
    /// Logical values per packet along the packing axis of packed or external
    /// storage; `None` for dense storage.
    pub fn logical_group(&self) -> Option<u64> {
        match &seismic_lang::registry::representation_info(self.0).kind {
            seismic_lang::registry::RepresentationKind::Dense(_) => None,
            seismic_lang::registry::RepresentationKind::Packed(layout) => {
                Some(u64::from(layout.group))
            }
            seismic_lang::registry::RepresentationKind::PackedRows(layout) => {
                Some(u64::from(layout.group()))
            }
            seismic_lang::registry::RepresentationKind::External(layout) => {
                Some(u64::from(layout.logical_group))
            }
        }
    }
    /// Host reference of the registered conversion from `source` into this
    /// storage over a tensor of `shape`. `None` when no conversion is
    /// registered or `bytes` is not the source's canonical byte count.
    pub fn repack_host(self, source: Element, shape: &[u64], bytes: &[u8]) -> Option<Vec<u8>> {
        let conversion = seismic_lang::registry::representation_conversion(source.0, self.0)?;
        let shape = shape
            .iter()
            .map(|extent| usize::try_from(*extent).ok())
            .collect::<Option<Vec<_>>>()?;
        seismic_lang::interp::repack(conversion.id, &shape, bytes)
    }
    /// Host reference decode of canonical packed or dense bytes of a tensor
    /// of `shape` into logical values in row-major order. `None` for external
    /// storage or a wrong byte count.
    pub fn decode_host(self, shape: &[u64], bytes: &[u8]) -> Option<Vec<f64>> {
        let shape = shape
            .iter()
            .map(|extent| usize::try_from(*extent).ok())
            .collect::<Option<Vec<_>>>()?;
        let data = match &seismic_lang::registry::representation_info(self.0).kind {
            seismic_lang::registry::RepresentationKind::Dense(dtype) => {
                seismic_lang::interp::TensorData::dense_from_bytes(*dtype, shape, bytes.to_vec())
            }
            seismic_lang::registry::RepresentationKind::Packed(_)
            | seismic_lang::registry::RepresentationKind::PackedRows(_) => {
                seismic_lang::interp::TensorData::encoded(self.0, shape, bytes.to_vec())
            }
            seismic_lang::registry::RepresentationKind::External(_) => return None,
        }
        .ok()?;
        data.values().ok()
    }
    fn id(&self) -> RepresentationId {
        self.0
    }
    pub fn dense(dtype: DType) -> Element {
        Element(seismic_lang::registry::dense(dtype))
    }
    pub fn dtype(self) -> Option<DType> {
        match &seismic_lang::registry::representation_info(self.0).kind {
            seismic_lang::registry::RepresentationKind::Dense(dtype) => Some(*dtype),
            seismic_lang::registry::RepresentationKind::Packed(_)
            | seismic_lang::registry::RepresentationKind::PackedRows(_)
            | seismic_lang::registry::RepresentationKind::External(_) => None,
        }
    }
    /// Exact bytes of the runtime's canonical tensor layout for these
    /// logical extents, including per-row packets and packet alignment.
    pub fn canonical_byte_len(self, extents: &[u64]) -> Result<u64, TensorError> {
        seismic_runtime::api::tensor::TensorInner::canonical_byte_len(self.id(), extents)
    }
    pub fn f32() -> Element {
        Element(seismic_lang::registry::dense(
            seismic_lang::types::DType::F32,
        ))
    }
    pub fn f16() -> Element {
        Element(seismic_lang::registry::dense(
            seismic_lang::types::DType::F16,
        ))
    }
    pub fn bf16() -> Element {
        Element(seismic_lang::registry::dense(
            seismic_lang::types::DType::BF16,
        ))
    }
    pub fn i32() -> Element {
        Element(seismic_lang::registry::dense(
            seismic_lang::types::DType::I32,
        ))
    }
    pub fn u32() -> Element {
        Element(seismic_lang::registry::dense(
            seismic_lang::types::DType::U32,
        ))
    }
    pub fn bool() -> Element {
        Element(seismic_lang::registry::dense(
            seismic_lang::types::DType::Bool,
        ))
    }
}

impl From<DType> for Element {
    fn from(value: DType) -> Self {
        Self::dense(value)
    }
}

/// IEEE-754 binary16 scalar value, preserved as bits at the Rust boundary.
/// Seismic performs any widening explicitly according to the authored
/// kernel and selected numerical policy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct F16(u16);

impl F16 {
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }
    pub const fn to_bits(self) -> u16 {
        self.0
    }
}

/// bfloat16 scalar value, preserved as bits at the Rust boundary.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct BF16(u16);

impl BF16 {
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }
    pub const fn to_bits(self) -> u16 {
        self.0
    }
}

/// A device tensor: an owned allocation or a view, with device identity,
/// representation, extents and strides (§14.3). Shapes come from here.
/// One read-only host window wrapped by the device. Every tensor view shares
/// one allocation and one charge; submitted work retains that allocation and
/// its host mapping until completion.
pub struct ReadOnlyMappedRegion {
    inner: seismic_runtime::api::tensor::MappedRegionInner,
}

impl ReadOnlyMappedRegion {
    pub fn new(device: &Device, region: HostRegion) -> Result<Self, TensorError> {
        seismic_runtime::api::tensor::MappedRegionInner::new(device.inner(), region)
            .map(|inner| Self { inner })
    }

    pub fn tensor(
        &self,
        element: Element,
        extents: &[u64],
        byte_offset: u64,
    ) -> Result<Tensor, TensorError> {
        self.inner
            .tensor(element.id(), extents, byte_offset)
            .map(|inner| Tensor {
                inner: Arc::new(inner),
            })
    }
}

#[derive(Clone)]
pub struct Tensor {
    inner: Arc<seismic_runtime::api::tensor::TensorInner>,
}

/// A weak view of one tensor's physical storage. It never pins the
/// allocation; callers can classify storage still charged after ownership
/// moves without maintaining a second byte ledger.
pub struct TensorStorageObserver {
    inner: seismic_runtime::api::tensor::TensorStorageObserver,
}

impl TensorStorageObserver {
    pub fn identity(&self) -> u64 {
        self.inner.identity()
    }

    pub fn charged_bytes(&self) -> Option<u64> {
        self.inner.charged_bytes()
    }
}

impl Tensor {
    pub(crate) fn slabbed(
        table: &Tensor,
        element: Element,
        rows_per_slab: u64,
        logical_rows: u64,
        alias_identity: u64,
        regions: Vec<Option<Tensor>>,
        lease: Arc<std::sync::atomic::AtomicUsize>,
    ) -> Result<Tensor, TensorError> {
        seismic_runtime::api::tensor::TensorInner::slabbed(
            &table.inner,
            element.id(),
            rows_per_slab,
            logical_rows,
            alias_identity,
            regions
                .into_iter()
                .map(|region| region.map(|tensor| tensor.inner))
                .collect(),
            lease,
        )
        .map(|inner| Tensor {
            inner: Arc::new(inner),
        })
    }

    pub(crate) fn new_slab_alias() -> u64 {
        seismic_runtime::api::tensor::TensorInner::new_slab_alias()
    }

    /// Canonical storage footprint and required alignment of a tensor shape.
    pub fn canonical_layout(element: Element, extents: &[u64]) -> Result<(u64, u64), TensorError> {
        seismic_runtime::api::tensor::TensorInner::canonical_layout(element.id(), extents)
    }

    /// A typed region inside a byte-addressed tensor allocation. Views share
    /// the allocation and its device-use lifetime.
    pub(crate) fn region(
        &self,
        element: Element,
        extents: &[u64],
        byte_offset: u64,
    ) -> Result<Tensor, TensorError> {
        self.inner
            .region(element.id(), extents, byte_offset)
            .map(|inner| Tensor {
                inner: Arc::new(inner),
            })
    }

    pub(crate) fn device_address(&self) -> u64 {
        self.inner.device_address()
    }

    pub(crate) fn register_slab(&self) {
        self.inner.register_slab()
    }

    pub(crate) fn share_slab_access_with(&self, table: &Tensor) {
        self.inner.share_slab_access_with(&table.inner)
    }

    /// Allocate without initializing storage. A pure-output kernel can use
    /// this to avoid a host zero-fill before writing the result.
    ///
    /// # Safety
    /// Every byte of the tensor's physical layout, including padding, must
    /// be initialized before the tensor is read or passed as a device input.
    pub unsafe fn uninitialized(
        device: &Device,
        element: Element,
        extents: &[u64],
    ) -> Result<Tensor, TensorError> {
        // SAFETY: the caller upholds the complete-initialization contract.
        unsafe {
            seismic_runtime::api::tensor::TensorInner::uninitialized(
                device.inner(),
                element.id(),
                extents,
            )
        }
        .map(|inner| Tensor {
            inner: Arc::new(inner),
        })
    }

    /// A zero-filled owned tensor.
    pub fn zeros(
        device: &Device,
        element: Element,
        extents: &[u64],
    ) -> Result<Tensor, TensorError> {
        seismic_runtime::api::tensor::TensorInner::zeros(device.inner(), element.id(), extents).map(
            |inner| Tensor {
                inner: Arc::new(inner),
            },
        )
    }
    /// An owned tensor initialized from host bytes in the representation's
    /// canonical dense layout.
    pub fn from_host(
        device: &Device,
        element: Element,
        extents: &[u64],
        bytes: &[u8],
    ) -> Result<Tensor, TensorError> {
        seismic_runtime::api::tensor::TensorInner::from_host(
            device.inner(),
            element.id(),
            extents,
            bytes,
        )
        .map(|inner| Tensor {
            inner: Arc::new(inner),
        })
    }
    /// A zero-filled tensor over `extents` whose leading-axis rows
    /// `[0, committed)` are physically backed and charged. The rest of the
    /// leading extent is reserved address space: graphs are sealed over the
    /// whole shape, while device work bound to the tensor must stay within
    /// the committed rows, and host access to the tensor is refused (views
    /// within the committed rows are ordinary tensors).
    pub fn reserved(
        device: &Device,
        element: Element,
        extents: &[u64],
        committed: u64,
    ) -> Result<Tensor, TensorError> {
        seismic_runtime::api::tensor::TensorInner::reserved(
            device.inner(),
            element.id(),
            extents,
            committed,
        )
        .map(|inner| Tensor {
            inner: Arc::new(inner),
        })
    }
    /// Leading rows physically backed: every row unless the tensor is reserved.
    pub fn committed_rows(&self) -> u64 {
        self.inner.committed_rows()
    }
    /// The same reserved shape with `committed` leading rows backed. Rows
    /// below the smaller commitment keep their contents (copied after every
    /// submitted device write of this tensor completes); new rows are zero.
    /// The result is a new allocation: work bound to `self` is not ordered
    /// with work bound to the result, so the owner switches only between
    /// submissions. An in-place growth can be abandoned while the old tensor
    /// is live; dropping the new tensor restores its former backed prefix.
    /// A successful in-place shrink must be published by dropping the old
    /// tensor, since the released tail's contents cannot be restored.
    pub fn recommitted(&self, committed: u64) -> Result<Tensor, TensorError> {
        self.inner
            .recommitted(committed, self.can_recommit_in_place())
            .map(|inner| Tensor {
                inner: Arc::new(inner),
            })
    }
    /// Recommit into separate backing even when this tensor owns a reserved
    /// address. This lets an owner prepare a multi-tensor replacement before
    /// releasing any old rows through an in-place shrink.
    pub fn recommitted_separately(&self, committed: u64) -> Result<Tensor, TensorError> {
        self.inner
            .recommitted(committed, false)
            .map(|inner| Tensor {
                inner: Arc::new(inner),
            })
    }
    /// Whether [`Tensor::recommitted`] keeps this tensor's address: the backend
    /// reserved its address range (CUDA virtual memory management).
    pub fn resizes_in_place(&self) -> bool {
        self.inner.resizes_in_place()
    }
    /// Whether this tensor currently has sole ownership of its reserved
    /// backing, allowing a recommit without a second physical allocation.
    pub fn can_recommit_in_place(&self) -> bool {
        Arc::strong_count(&self.inner) == 1 && self.inner.can_recommit_in_place()
    }
    /// The same reserved shape with `committed` leading rows backed by a new
    /// allocation holding the rows `moves` names: each `(from, to, rows)`
    /// copies rows `[from, from + rows)` (after every submitted device write
    /// of this tensor completes) to rows `[to, to + rows)`; every other row is
    /// zero. The address always changes, so the owner switches only between
    /// submissions.
    pub fn relocated(
        &self,
        committed: u64,
        moves: &[(u64, u64, u64)],
    ) -> Result<Tensor, TensorError> {
        self.inner.relocated(committed, moves).map(|inner| Tensor {
            inner: Arc::new(inner),
        })
    }
    pub fn read_to_host(&self) -> Result<Vec<u8>, ExecutionError> {
        self.inner.read_to_host()
    }
    /// Replaces the bytes of this exact tensor view.  The byte count must
    /// match its canonical representation layout; allocation-level access
    /// exclusion makes this safe even when other views share the allocation.
    pub fn write_from_host(&mut self, bytes: &[u8]) -> Result<(), TensorError> {
        self.inner.write_from_host(bytes)
    }
    /// Fill all physical bytes of a canonical upload tensor from `reader`
    /// with bounded scratch, rather than materializing a second whole tensor.
    /// On error the tensor is only partly initialized and must not be used.
    pub fn write_from_reader(&mut self, reader: &mut dyn std::io::Read) -> Result<(), TensorError> {
        self.inner.write_from_reader(reader)
    }
    pub fn device(&self) -> Device {
        Device {
            inner: self.inner.device().clone(),
        }
    }
    pub fn element(&self) -> Element {
        Element(self.inner.representation())
    }
    pub fn extents(&self) -> &[u64] {
        self.inner.extents()
    }
    pub fn strides(&self) -> &[u64] {
        self.inner.strides()
    }
    /// A view over a contiguous range of the leading axis.
    pub fn slice_leading(&self, start: u64, end: u64) -> Result<Tensor, TensorError> {
        self.inner.slice_leading(start, end).map(|inner| Tensor {
            inner: Arc::new(inner),
        })
    }
    /// A canonical-layout view with different logical extents and identical
    /// physical byte coverage.
    pub fn reshape(&self, extents: &[u64]) -> Result<Tensor, TensorError> {
        self.inner.reshape(extents).map(|inner| Tensor {
            inner: Arc::new(inner),
        })
    }
    pub fn byte_len(&self) -> u64 {
        self.inner.byte_len()
    }
    pub fn storage_bytes(&self) -> u64 {
        self.inner.storage_bytes()
    }
    pub fn observe_storage(&self) -> TensorStorageObserver {
        TensorStorageObserver {
            inner: self.inner.observe_storage(),
        }
    }
    pub fn belongs_to(&self, device: &Device) -> bool {
        Arc::ptr_eq(self.inner.device(), device.inner())
    }
    pub fn shares_allocation(&self, other: &Tensor) -> bool {
        self.inner.shares_allocation(&other.inner)
    }

    /// Physical bytes released if exactly these tensor handles are dropped.
    /// Views and clones outside the supplied set continue to pin storage.
    pub fn reclaimable_bytes<'a>(
        tensors: impl IntoIterator<Item = &'a Tensor>,
    ) -> Result<u64, TensorError> {
        seismic_runtime::api::tensor::TensorInner::reclaimable_bytes(
            tensors.into_iter().map(|tensor| &tensor.inner),
        )
    }
    pub(crate) fn descriptor(&self) -> seismic_compiler::prepared::TensorDescriptor {
        self.inner.descriptor()
    }
    pub(crate) fn inner(&self) -> &Arc<seismic_runtime::api::tensor::TensorInner> {
        &self.inner
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("element", &self.element().name())
            .field("extents", &self.extents())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

/// Failure of `for_device`.
#[derive(Debug)]
pub enum LoadError {
    Bundle(CheckedBundleError),
    Source(SourceLoadError),
    Preparation(PreparationError),
}

impl LoadError {
    fn from_prepare(error: seismic_runtime::api::kernel::PrepareError) -> Self {
        match error {
            seismic_runtime::api::kernel::PrepareError::Source(error) => {
                Self::Source(SourceLoadError::from_internal(error))
            }
            seismic_runtime::api::kernel::PrepareError::Preparation(error) => {
                Self::Preparation(error)
            }
        }
    }
}

/// A checked source/binding error discovered while instantiating a generated
/// entry. The checked compiler object and its internal IDs are deliberately
/// not part of the public API.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceLoadError {
    message: String,
}

impl SourceLoadError {
    fn from_internal(error: seismic_lang::checked::SourceError) -> Self {
        Self {
            message: error.to_string(),
        }
    }
}

impl fmt::Display for SourceLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for SourceLoadError {}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bundle(e) => write!(f, "{e}"),
            Self::Source(e) => write!(f, "{e}"),
            Self::Preparation(e) => write!(f, "{e}"),
        }
    }
}
impl std::error::Error for LoadError {}

/// Symbolic tensor result owned by one workflow draft. It has no ordinary
/// tensor operations and cannot be passed to `Kernel::call`.
#[derive(Clone)]
pub struct WorkflowTensor {
    inner: seismic_runtime::api::kernel::WorkflowResultRef,
}

#[derive(Clone)]
pub struct WorkflowTensorView {
    inner: seismic_runtime::api::kernel::WorkflowResultRef,
    operations: Vec<seismic_runtime::api::kernel::ViewOperation>,
}

impl WorkflowTensor {
    /// A symbolic contiguous view of the leading axis. Bounds and packet
    /// alignment are validated during whole-workflow admission, once the
    /// producer's exact result descriptor exists.
    pub fn slice_leading(&self, start: u64, end: u64) -> WorkflowTensorView {
        WorkflowTensorView {
            inner: self.inner,
            operations: vec![seismic_runtime::api::kernel::ViewOperation::LeadingSlice {
                start,
                end,
            }],
        }
    }
    /// A symbolic canonical reshape of a workflow result. Seismic checks
    /// physical byte coverage and representation before graph execution.
    pub fn reshape(&self, extents: &[u64]) -> WorkflowTensorView {
        WorkflowTensorView {
            inner: self.inner,
            operations: vec![seismic_runtime::api::kernel::ViewOperation::Reshape {
                extents: extents.to_vec(),
            }],
        }
    }
}

impl WorkflowTensorView {
    pub fn slice_leading(&self, start: u64, end: u64) -> Self {
        let mut view = self.clone();
        view.operations
            .push(seismic_runtime::api::kernel::ViewOperation::LeadingSlice { start, end });
        view
    }
    pub fn reshape(&self, extents: &[u64]) -> Self {
        let mut view = self.clone();
        view.operations
            .push(seismic_runtime::api::kernel::ViewOperation::Reshape {
                extents: extents.to_vec(),
            });
        view
    }
}

pub enum WorkflowTensorRef<'a> {
    External(&'a Tensor),
    Result(&'a WorkflowTensor),
    View(&'a WorkflowTensorView),
}

impl<'a> From<&'a Tensor> for WorkflowTensorRef<'a> {
    fn from(value: &'a Tensor) -> Self {
        Self::External(value)
    }
}

impl<'a> From<&'a WorkflowTensor> for WorkflowTensorRef<'a> {
    fn from(value: &'a WorkflowTensor) -> Self {
        Self::Result(value)
    }
}

impl<'a> From<&'a WorkflowTensorView> for WorkflowTensorRef<'a> {
    fn from(value: &'a WorkflowTensorView) -> Self {
        Self::View(value)
    }
}

pub enum WorkflowTensorMut<'a> {
    External(&'a mut Tensor),
    Result(&'a mut WorkflowTensor),
    View(&'a mut WorkflowTensorView),
}

impl<'a> From<&'a mut Tensor> for WorkflowTensorMut<'a> {
    fn from(value: &'a mut Tensor) -> Self {
        Self::External(value)
    }
}

impl<'a> From<&'a mut WorkflowTensor> for WorkflowTensorMut<'a> {
    fn from(value: &'a mut WorkflowTensor) -> Self {
        Self::Result(value)
    }
}

impl<'a> From<&'a mut WorkflowTensorView> for WorkflowTensorMut<'a> {
    fn from(value: &'a mut WorkflowTensorView) -> Self {
        Self::View(value)
    }
}

enum WorkflowTensorOwnedValue {
    External(Tensor),
    Result(WorkflowTensor),
}

pub struct WorkflowTensorOwned<'a> {
    value: WorkflowTensorOwnedValue,
    marker: std::marker::PhantomData<&'a ()>,
}

impl<'a> From<Tensor> for WorkflowTensorOwned<'a> {
    fn from(value: Tensor) -> Self {
        Self {
            value: WorkflowTensorOwnedValue::External(value),
            marker: std::marker::PhantomData,
        }
    }
}

impl<'a> From<WorkflowTensor> for WorkflowTensorOwned<'a> {
    fn from(value: WorkflowTensor) -> Self {
        Self {
            value: WorkflowTensorOwnedValue::Result(value),
            marker: std::marker::PhantomData,
        }
    }
}

#[derive(Clone)]
pub struct WorkflowScalar<T> {
    inner: seismic_runtime::api::kernel::WorkflowResultRef,
    marker: std::marker::PhantomData<fn() -> T>,
}

pub enum WorkflowScalarValue<'a, T> {
    Immediate(T),
    Result(&'a WorkflowScalar<T>),
}

/// The contract a generated entry type satisfies.
///
/// # Safety
///
/// Implementations must be emitted together with the content-addressed
/// checked bundle by `seismic-build`. `Args`, `OutputArgs`, `Results`, their
/// encoders, `decode`, and `resolve` must describe that bundle entry exactly. Safe consumers
/// never implement this trait; the unsafe boundary is what lets the runtime
/// treat a schema mismatch as a compiler/generator invariant rather than a
/// recoverable invocation condition.
pub unsafe trait Entry: 'static {
    /// Typed arguments.
    type Args<'a>;
    /// Typed results.
    type Results;
    /// Caller-owned tensor storage for checked native results.
    type OutputArgs<'a>;
    /// Typed arguments whose tensor/scalar leaves may reference results of
    /// earlier nodes in the same workflow draft.
    type WorkflowArgs<'a>;
    /// Typed symbolic results returned while constructing a workflow.
    type WorkflowResults;
    const NAME: &'static str;
    /// Fixed device bytes for one prepared direct native entry's ABI words
    /// and scalar-result storage, including each buffer's one-byte minimum.
    const NATIVE_INVOCATION_WORKSPACE_BYTES: u64;
    /// The checked module this entry belongs to.
    fn module() -> Result<&'static generated::Module, CheckedBundleError>;
    fn resolve(module: &generated::Module) -> Result<generated::EntryToken, CheckedBundleError>;
    fn encode(args: Self::Args<'_>) -> generated::EncodedArgs;
    fn encode_outputs(outputs: Self::OutputArgs<'_>) -> generated::EncodedOutputs;
    fn decode(results: generated::DecodedResults) -> Self::Results;
    fn encode_workflow(args: Self::WorkflowArgs<'_>) -> generated::EncodedWorkflowArgs;
    fn decode_workflow(results: generated::PendingWorkflowResults) -> Self::WorkflowResults;
    fn workflow_outputs(results: Self::WorkflowResults) -> Vec<generated::WorkflowResultRef>;
}

/// A prepared kernel for one entry on one device under one precision
/// policy. `call` validates, selects, allocates, and executes (§14.4).
pub struct Kernel<E: Entry> {
    inner: Arc<seismic_runtime::api::kernel::PreparedAny>,
    marker: std::marker::PhantomData<E>,
}

/// Mutable feedback search state. Kernels returned by this preparation own
/// their executable resources independently and are never changed by continuation.
pub struct FeedbackPreparation<'device, E: Entry> {
    inner: seismic_runtime::api::kernel::FeedbackPreparation<'device>,
    marker: std::marker::PhantomData<E>,
}

impl<'device, E: Entry> FeedbackPreparation<'device, E> {
    fn start(
        device: &'device Device,
        precision: PrecisionPolicy,
        options: FeedbackOptions,
        bindings: seismic_lang::entry::ElementBindings,
    ) -> Result<(Self, Kernel<E>), LoadError> {
        let module = E::module().map_err(LoadError::Bundle)?;
        let entry = E::resolve(module).map_err(LoadError::Bundle)?;
        let (inner, kernel) = seismic_runtime::api::kernel::start_feedback(
            module.checked(),
            entry.id(),
            bindings,
            device.inner(),
            precision,
            options,
        )
        .map_err(LoadError::from_prepare)?;
        Ok((
            Self {
                inner,
                marker: std::marker::PhantomData,
            },
            Kernel {
                inner: Arc::new(kernel),
                marker: std::marker::PhantomData,
            },
        ))
    }

    pub fn continue_for(
        &mut self,
        additional: std::time::Duration,
    ) -> Result<Kernel<E>, LoadError> {
        self.inner
            .continue_for(additional)
            .map(|inner| Kernel {
                inner: Arc::new(inner),
                marker: std::marker::PhantomData,
            })
            .map_err(LoadError::from_prepare)
    }

    pub fn report(&self) -> &FeedbackReport {
        self.inner.report()
    }
}

/// An explicitly selected, top-level native implementation of one entry.
/// It has the same typed call contract as [`Kernel`]. It can be composed into
/// a direct-native graph, but is not a portable workflow/compiler candidate.
pub struct NativeKernel<E: Entry> {
    inner: Arc<seismic_runtime::api::kernel::NativePreparedAny>,
    marker: std::marker::PhantomData<E>,
}

/// Tensor-result native calls from any prepared entries on one device,
/// encoded in order and completed with one device wait.
pub struct NativeTensorBatch {
    inner: seismic_runtime::api::kernel::NativeTensorBatchAny,
}

pub struct NativeTensorBatchCompletion {
    inner: seismic_runtime::api::kernel::NativeTensorBatchCompletionAny,
}

impl NativeTensorBatch {
    pub fn new(device: &Device) -> Self {
        Self {
            inner: seismic_runtime::api::kernel::NativeTensorBatchAny::new(device.inner()),
        }
    }

    pub fn push<E: Entry>(
        &mut self,
        kernel: &NativeKernel<E>,
        args: E::Args<'_>,
        outputs: E::OutputArgs<'_>,
    ) -> Result<(), CallError> {
        self.inner
            .push(&kernel.inner, E::encode(args), E::encode_outputs(outputs))
    }

    pub fn submit(self) -> Result<NativeTensorBatchCompletion, CallError> {
        self.inner
            .submit()
            .map(|inner| NativeTensorBatchCompletion { inner })
    }
}

impl NativeTensorBatchCompletion {
    /// Observe completion and any device failure. Dropping an unobserved
    /// completion still waits so its mapped inputs stay alive.
    pub fn wait(self) -> Result<(), CallError> {
        self.inner.wait()
    }
}

/// One workload for native tuning: argument sets cycled by measurement, the
/// first also used for validation, and the point's share of the objective.
pub struct TuningPoint<'a, E: Entry> {
    pub label: String,
    pub weight: f64,
    /// Points naming the same class are variants of one workload (the same
    /// rows at different history lengths): they split their summed weight by
    /// the defaults' real time at each. `None`: a class of its own.
    pub class: Option<String>,
    pub rotation: Vec<E::Args<'a>>,
    /// Required when the entry has `&mut` parameters: restores the tensors
    /// they bind in `rotation[0]` before each configuration's validation
    /// run. The tuner never saves or restores state itself.
    pub initialize: Option<TuningInitializer<'a>>,
}

/// A direct-native graph is assembled from generated entry arguments and
/// symbolic result edges. Seismic derives every intermediate tensor from the
/// checked entry contracts and owns its storage plan.
pub struct NativeGraph {
    inner: seismic_runtime::native::graph::NativeGraphDraft,
}

/// Backend-free storage projection of a native graph. The caller follows the
/// same checked topology as a prepared graph. Native scratch uses the largest
/// checked charge across the declaration's finite tuning choices.
pub struct NativeGraphMetadata {
    backend: BackendName,
    inner: seismic_runtime::native::graph::NativeGraphMetadataDraft,
    template: Option<MetadataTemplateRecorder>,
    class_scope: Option<&'static str>,
}

struct MetadataTemplateRecorder {
    ports: Vec<MetadataPortSource>,
    nodes: Vec<MetadataNodeSource>,
}

enum MetadataPortSource {
    Direct {
        element: Element,
        extents: Vec<u64>,
        class_extent: Option<(usize, &'static str)>,
    },
    Checked {
        checked: generated::NativeGraphCheckedEntry,
        entry_name: &'static str,
        class_scope: Option<&'static str>,
        parameter: String,
        dimensions: Vec<(String, u64)>,
    },
}

struct MetadataNodeSource {
    checked: generated::NativeGraphCheckedEntry,
    entry_name: &'static str,
    class_scope: Option<&'static str>,
    dimensions: Vec<(String, u64)>,
}

/// One checked graph topology reused across the numeric shape classes of a
/// structural regime. Certification asks the generated entry contracts for
/// exact shapes and scratch before Seismic places the graph's original edges
/// and lifetimes.
pub struct NativeGraphResourceTemplate {
    inner: seismic_runtime::native::graph::NativeGraphMetadataDraft,
    recorder: MetadataTemplateRecorder,
}

/// One slice of the admitted classes of a structural graph regime: every
/// combination of its listed dimension values. A regime's classes are the
/// union of its slices, so correlated dimensions (slots bounded by rows) are
/// fixed together per slice. Every slice of a regime names the same
/// dimensions.
#[derive(Clone, Debug, Default)]
pub struct NativeGraphClassSlice {
    dimensions: Vec<(&'static str, Vec<u64>)>,
    scoped: Vec<(&'static str, &'static str, Vec<u64>)>,
}

impl NativeGraphClassSlice {
    pub fn new() -> Self {
        Self::default()
    }

    /// The values of a class dimension for every node that declares it.
    pub fn dimension(mut self, name: &'static str, values: impl IntoIterator<Item = u64>) -> Self {
        self.dimensions.push((name, values.into_iter().collect()));
        self
    }

    /// The values of a dimension for nodes of one class scope or entry name.
    /// A class scope takes precedence over an entry name, and both over an
    /// unscoped dimension of the same name.
    pub fn scoped(
        mut self,
        scope: &'static str,
        name: &'static str,
        values: impl IntoIterator<Item = u64>,
    ) -> Self {
        self.scoped
            .push((scope, name, values.into_iter().collect()));
        self
    }

    fn names(&self) -> Vec<(Option<&'static str>, &'static str)> {
        let mut names = self
            .dimensions
            .iter()
            .map(|(name, _)| (None, *name))
            .chain(
                self.scoped
                    .iter()
                    .map(|(scope, name, _)| (Some(*scope), *name)),
            )
            .collect::<Vec<_>>();
        names.sort();
        names
    }

    fn global(&self, name: &str) -> Option<&[u64]> {
        self.dimensions
            .iter()
            .find(|(candidate, _)| *candidate == name)
            .map(|(_, values)| values.as_slice())
    }

    fn values(&self, entry_name: &str, class_scope: Option<&str>, name: &str) -> Option<&[u64]> {
        let scoped = |scope: &str| {
            self.scoped
                .iter()
                .find(|(candidate, axis, _)| *candidate == scope && *axis == name)
                .map(|(_, _, values)| values.as_slice())
        };
        class_scope
            .and_then(scoped)
            .or_else(|| scoped(entry_name))
            .or_else(|| self.global(name))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NativeGraphMetadataError {
    Bundle(CheckedBundleError),
    Unsupported(String),
    Tensor(TensorError),
    Call(CallError),
    Workflow(WorkflowError),
}

impl fmt::Display for NativeGraphMetadataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bundle(error) => write!(f, "{error}"),
            Self::Unsupported(reason) => write!(f, "{reason}"),
            Self::Tensor(error) => write!(f, "{error}"),
            Self::Call(error) => write!(f, "{error}"),
            Self::Workflow(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for NativeGraphMetadataError {}

pub use seismic_runtime::native::graph::NativeGraphStorageBytes;

#[derive(Clone)]
pub struct NativeGraphLayout {
    inner: Arc<seismic_runtime::native::graph::NativeGraphLayout>,
}

impl NativeGraphLayout {
    pub fn storage_bytes(&self) -> NativeGraphStorageBytes {
        self.inner.storage_bytes()
    }
}

fn owned_dimensions(dimensions: &[(&str, u64)]) -> Vec<(String, u64)> {
    dimensions
        .iter()
        .map(|(name, value)| ((*name).to_owned(), *value))
        .collect()
}

/// The largest value `measure` takes over every admitted class. A buffer's
/// size is a function of the dimensions its expression reads, so it is
/// evaluated once per distinct combination of those dimensions across all
/// slices; every other dimension of the node keeps its recorded value.
fn class_maximum(
    slices: &[NativeGraphClassSlice],
    entry_name: &str,
    class_scope: Option<&str>,
    recorded: &[(String, u64)],
    reads: &[&str],
    measure: impl Fn(&[(&str, u64)]) -> Result<u64, String>,
) -> Result<u64, String> {
    // Every slice names the same dimensions, so the class dimensions this
    // buffer reads are the same in each.
    let varying = recorded
        .iter()
        .enumerate()
        .filter(|(_, (name, _))| {
            reads.contains(&name.as_str())
                && slices[0].values(entry_name, class_scope, name).is_some()
        })
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let mut combinations = std::collections::HashSet::<Vec<u64>>::new();
    let mut combination = vec![0; varying.len()];
    for slice in slices {
        let lists = varying
            .iter()
            .map(|&index| {
                let name = &recorded[index].0;
                slice
                    .values(entry_name, class_scope, name)
                    .ok_or_else(|| format!("class slice omits dimension `{name}`"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut digits = vec![0; varying.len()];
        'combinations: loop {
            for ((value, values), &digit) in combination.iter_mut().zip(&lists).zip(&digits) {
                *value = values[digit];
            }
            if !combinations.contains(combination.as_slice()) {
                combinations.insert(combination.clone());
            }
            for (digit, values) in digits.iter_mut().zip(&lists) {
                *digit += 1;
                if *digit < values.len() {
                    continue 'combinations;
                }
                *digit = 0;
            }
            break;
        }
    }
    let mut point = recorded
        .iter()
        .map(|(name, value)| (name.as_str(), *value))
        .collect::<Vec<_>>();
    let mut maximum = 0;
    for combination in &combinations {
        for (&index, &value) in varying.iter().zip(combination) {
            point[index].1 = value;
        }
        maximum = maximum.max(measure(&point)?);
    }
    Ok(maximum)
}

impl NativeGraphResourceTemplate {
    /// Place one layout for every admitted class of this structural regime.
    /// Each port, result and scratch buffer is charged the exact maximum of
    /// its checked size over the classes, evaluated at the combinations of
    /// the dimensions its size expression reads. Seismic places those
    /// capacities once; production seals each exact class into the same
    /// offsets and rejects any class that does not fit.
    pub fn certify(
        &self,
        slices: &[NativeGraphClassSlice],
    ) -> Result<NativeGraphLayout, NativeGraphMetadataError> {
        let unsupported = NativeGraphMetadataError::Unsupported;
        let Some(first) = slices.first() else {
            return Err(unsupported("graph regime has no admitted classes".into()));
        };
        let names = first.names();
        let valid = names.windows(2).all(|pair| pair[0] != pair[1])
            && slices.iter().all(|slice| {
                slice.names() == names
                    && slice
                        .dimensions
                        .iter()
                        .all(|(_, values)| !values.is_empty())
                    && slice.scoped.iter().all(|(_, _, values)| !values.is_empty())
            });
        if !valid {
            return Err(unsupported(
                "graph regime slices must name the same dimensions once, each with values".into(),
            ));
        }
        let ports = self
            .recorder
            .ports
            .iter()
            .map(|source| match source {
                MetadataPortSource::Direct {
                    element,
                    extents,
                    class_extent,
                } => {
                    let Some((axis, name)) = class_extent else {
                        return element
                            .canonical_byte_len(extents)
                            .map_err(NativeGraphMetadataError::Tensor);
                    };
                    let mut extents = extents.clone();
                    let mut maximum = 0;
                    for slice in slices {
                        let values = slice.global(name).ok_or_else(|| {
                            unsupported(format!("graph regime omits class dimension `{name}`"))
                        })?;
                        for &value in values {
                            extents[*axis] = value;
                            maximum = maximum.max(
                                element
                                    .canonical_byte_len(&extents)
                                    .map_err(NativeGraphMetadataError::Tensor)?,
                            );
                        }
                    }
                    Ok(maximum)
                }
                MetadataPortSource::Checked {
                    checked,
                    entry_name,
                    class_scope,
                    parameter,
                    dimensions,
                } => {
                    let reads = checked
                        .parameter_dimensions(parameter)
                        .map_err(unsupported)?;
                    class_maximum(
                        slices,
                        entry_name,
                        *class_scope,
                        dimensions,
                        &reads,
                        |point| {
                            checked
                                .parameter(parameter, point)
                                .map(|metadata| metadata.canonical_bytes)
                        },
                    )
                    .map_err(|error| unsupported(format!("`{entry_name}.{parameter}`: {error}")))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut results = Vec::with_capacity(self.recorder.nodes.len());
        let mut scratch = Vec::with_capacity(self.recorder.nodes.len());
        for source in &self.recorder.nodes {
            let entry = source.entry_name;
            let maximum =
                |reads: &[&str], measure: &dyn Fn(&[(&str, u64)]) -> Result<u64, String>| {
                    class_maximum(
                        slices,
                        entry,
                        source.class_scope,
                        &source.dimensions,
                        reads,
                        measure,
                    )
                    .map_err(|error| unsupported(format!("`{entry}`: {error}")))
                };
            results.push(
                (0..source.checked.result_count())
                    .map(|ordinal| {
                        let reads = source
                            .checked
                            .result_dimensions(ordinal)
                            .map_err(unsupported)?;
                        maximum(&reads, &|point| {
                            source
                                .checked
                                .result(ordinal, point)
                                .map(|metadata| metadata.canonical_bytes)
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            );
            let implementation = source.checked.implementation();
            scratch.push(
                implementation
                    .scratch
                    .iter()
                    .map(|buffer| {
                        let reads = buffer.dimensions();
                        let reads = reads.iter().map(String::as_str).collect::<Vec<_>>();
                        maximum(&reads, &|point| {
                            buffer
                                .maximum_bytes(&implementation.params, &|name| {
                                    point
                                        .iter()
                                        .find(|(candidate, _)| *candidate == name)
                                        .map(|(_, value)| *value)
                                })
                                .map_err(|error| format!("scratch `{}`: {error}", buffer.name))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            );
        }
        self.inner
            .capacity_layout(&ports, &results, scratch)
            .map(|inner| NativeGraphLayout {
                inner: Arc::new(inner),
            })
            .map_err(NativeGraphMetadataError::Workflow)
    }
}

fn checked_native_scratch_bound(
    scratch_buffers: &[NativeScratch],
    tuning_parameters: &[NativeParameter],
    dimensions: &[(&str, u64)],
) -> Result<Vec<u64>, String> {
    scratch_buffers
        .iter()
        .map(|scratch| {
            scratch.maximum_bytes(tuning_parameters, &|name| {
                dimensions
                    .iter()
                    .find(|(candidate, _)| *candidate == name)
                    .map(|(_, value)| *value)
            })
        })
        .collect::<Result<_, _>>()
        .map_err(|error| error.to_string())
}

#[cfg(test)]
mod metadata_scratch_tests {
    use super::*;
    use seismic_lang::checked::NativeParameterRole;

    #[test]
    fn scratch_bound_covers_conditional_tuning_choices() {
        let parameter = NativeParameter {
            name: "tile".into(),
            code: false,
            arithmetic: false,
            values: vec![1, 4],
            role: NativeParameterRole::Declared,
        };
        let scratch = NativeScratch {
            name: "staged".into(),
            bytes: NativeNatExpr::Mul(
                Box::new(NativeNatExpr::Dimension("D".into())),
                Box::new(NativeNatExpr::Parameter("tile".into())),
            ),
            when: Some(NativeCondition::Compare {
                comparison: NativeComparison::Gt,
                left: NativeNatExpr::Parameter("tile".into()),
                right: NativeNatExpr::Constant(1),
            }),
        };
        assert_eq!(
            checked_native_scratch_bound(&[scratch], &[parameter], &[("D", 4)]).unwrap(),
            vec![16]
        );
    }
}

impl NativeGraphMetadata {
    pub fn new(backend: BackendName) -> Self {
        Self {
            backend,
            inner: seismic_runtime::native::graph::NativeGraphMetadataDraft::new(),
            template: None,
            class_scope: None,
        }
    }

    pub fn new_template(backend: BackendName) -> Self {
        Self {
            backend,
            inner: seismic_runtime::native::graph::NativeGraphMetadataDraft::new(),
            template: Some(MetadataTemplateRecorder {
                ports: Vec::new(),
                nodes: Vec::new(),
            }),
            class_scope: None,
        }
    }

    /// Tag following checked ports and nodes with a semantic class extent
    /// scope. A scoped override distinguishes repeated calls to one entry.
    pub fn set_class_scope(&mut self, scope: Option<&'static str>) {
        self.class_scope = scope;
    }

    pub fn port(
        &mut self,
        element: Element,
        extents: &[u64],
    ) -> Result<NativePort, NativeGraphMetadataError> {
        let inner = self
            .inner
            .port(element.id(), extents, false, false)
            .map_err(NativeGraphMetadataError::Tensor)?;
        if let Some(template) = &mut self.template {
            template.ports.push(MetadataPortSource::Direct {
                element,
                extents: extents.to_vec(),
                class_extent: None,
            });
        }
        Ok(NativePort {
            tensor: WorkflowTensor {
                inner: inner.reference(),
            },
            inner,
        })
    }

    /// Mark one direct port extent as a graph-class dimension. Checked entry
    /// ports already carry named dimensions in their declarations.
    pub fn port_with_class_extent(
        &mut self,
        element: Element,
        extents: &[u64],
        extent_axis: usize,
        class_dimension: &'static str,
    ) -> Result<NativePort, NativeGraphMetadataError> {
        let port = self.port(element, extents)?;
        if let Some(template) = &mut self.template {
            let Some(MetadataPortSource::Direct { class_extent, .. }) = template.ports.last_mut()
            else {
                unreachable!("direct port was just recorded")
            };
            if extent_axis >= extents.len() {
                return Err(NativeGraphMetadataError::Unsupported(format!(
                    "graph port extent axis {extent_axis} is absent"
                )));
            }
            *class_extent = Some((extent_axis, class_dimension));
        }
        Ok(port)
    }

    fn bind_checked<E: Entry>(
        &self,
        elements: &[(&str, Element)],
    ) -> Result<generated::NativeGraphCheckedEntry, NativeGraphMetadataError> {
        generated::NativeGraphCheckedEntry::bind::<E>(self.backend, elements)
            .map_err(NativeGraphMetadataError::Bundle)?
            .map_err(NativeGraphMetadataError::Unsupported)
    }

    fn checked_port<E: Entry>(
        &mut self,
        elements: &[(&str, Element)],
        parameter: &str,
        dimensions: &[(&str, u64)],
        owned_input: bool,
    ) -> Result<NativePort, NativeGraphMetadataError> {
        let checked = self.bind_checked::<E>(elements)?;
        let metadata = checked
            .parameter(parameter, dimensions)
            .map_err(NativeGraphMetadataError::Unsupported)?;
        let inner = self
            .inner
            .port(metadata.element.id(), &metadata.extents, true, owned_input)
            .map_err(NativeGraphMetadataError::Tensor)?;
        if let Some(template) = &mut self.template {
            template.ports.push(MetadataPortSource::Checked {
                checked,
                entry_name: E::NAME,
                class_scope: self.class_scope,
                parameter: parameter.to_owned(),
                dimensions: owned_dimensions(dimensions),
            });
        }
        Ok(NativePort {
            tensor: WorkflowTensor {
                inner: inner.reference(),
            },
            inner,
        })
    }

    pub fn input_for<E: Entry>(
        &mut self,
        elements: &[(&str, Element)],
        parameter: &str,
        dimensions: &[(&str, u64)],
    ) -> Result<NativePort, NativeGraphMetadataError> {
        self.checked_port::<E>(elements, parameter, dimensions, true)
    }

    pub fn local_for<E: Entry>(
        &mut self,
        elements: &[(&str, Element)],
        parameter: &str,
        dimensions: &[(&str, u64)],
    ) -> Result<NativePort, NativeGraphMetadataError> {
        self.checked_port::<E>(elements, parameter, dimensions, false)
    }

    pub fn prewrite(&mut self, port: &NativePort) -> Result<(), NativeGraphMetadataError> {
        self.inner
            .prewrite(port.inner)
            .map_err(NativeGraphMetadataError::Workflow)
    }

    pub fn enqueue<E: Entry>(
        &mut self,
        elements: &[(&str, Element)],
        dimensions: &[(&str, u64)],
        args: E::WorkflowArgs<'_>,
    ) -> Result<E::WorkflowResults, NativeGraphMetadataError> {
        let checked = self.bind_checked::<E>(elements)?;
        let (parameters, results) = checked
            .shapes(dimensions)
            .map_err(NativeGraphMetadataError::Unsupported)?;
        let implementation = checked.implementation();
        let scratch = checked_native_scratch_bound(
            &implementation.scratch,
            &implementation.params,
            dimensions,
        )
        .map_err(|reason| {
            NativeGraphMetadataError::Unsupported(format!("`{}` scratch bound: {reason}", E::NAME))
        })?;
        let shapes = results
            .into_iter()
            .map(|result| result.map(|metadata| (metadata.element.id(), metadata.extents)))
            .collect();
        let parameter_shapes = parameters
            .into_iter()
            .map(|parameter| parameter.map(|metadata| (metadata.element.id(), metadata.extents)))
            .collect();
        let pending = self
            .inner
            .enqueue(E::encode_workflow(args), parameter_shapes, shapes, scratch)
            .map_err(NativeGraphMetadataError::Call)?;
        if let Some(template) = &mut self.template {
            template.nodes.push(MetadataNodeSource {
                checked,
                entry_name: E::NAME,
                class_scope: self.class_scope,
                dimensions: owned_dimensions(dimensions),
            });
        }
        Ok(E::decode_workflow(pending))
    }

    pub fn export(&mut self, result: &WorkflowTensor) -> Result<(), NativeGraphMetadataError> {
        self.inner
            .export(result.inner)
            .map_err(NativeGraphMetadataError::Workflow)
    }

    pub fn seal(self) -> Result<NativeGraphStorageBytes, NativeGraphMetadataError> {
        self.inner
            .seal()
            .map_err(NativeGraphMetadataError::Workflow)
    }

    pub fn seal_with_layout(
        self,
        layout: &NativeGraphLayout,
    ) -> Result<NativeGraphStorageBytes, NativeGraphMetadataError> {
        self.inner
            .seal_with_layout(&layout.inner)
            .map_err(NativeGraphMetadataError::Workflow)
    }

    pub fn seal_template(self) -> Result<NativeGraphResourceTemplate, NativeGraphMetadataError> {
        self.inner
            .validate_topology()
            .map_err(NativeGraphMetadataError::Workflow)?;
        let recorder = self.template.ok_or_else(|| {
            NativeGraphMetadataError::Unsupported(
                "graph was not built as a resource template".into(),
            )
        })?;
        Ok(NativeGraphResourceTemplate {
            inner: self.inner,
            recorder,
        })
    }
}

#[derive(Clone)]
pub struct NativePort {
    inner: seismic_runtime::native::graph::NativePort,
    tensor: WorkflowTensor,
}

impl NativePort {
    pub fn tensor(&self) -> &WorkflowTensor {
        &self.tensor
    }
    pub fn tensor_mut(&mut self) -> &mut WorkflowTensor {
        &mut self.tensor
    }
}

impl NativeGraph {
    /// Declare an external tensor descriptor from model metadata. Sealing
    /// checks each use against its generated Seismic entry contract.
    pub fn port(&mut self, element: Element, extents: &[u64]) -> Result<NativePort, TensorError> {
        let inner = self.inner.port(element.id(), extents)?;
        Ok(NativePort {
            tensor: WorkflowTensor {
                inner: inner.reference(),
            },
            inner,
        })
    }

    /// Declare graph-owned mutable storage using a checked tensor parameter.
    /// The entry contract derives its representation and extents from the
    /// supplied dimension values; callers provide no independent shape.
    pub fn local_for<E: Entry>(
        &mut self,
        kernel: &NativeKernel<E>,
        parameter: &str,
        dimensions: &[(&str, u64)],
    ) -> Result<NativePort, CallError> {
        let inner = self.inner.local_for(&kernel.inner, parameter, dimensions)?;
        Ok(NativePort {
            tensor: WorkflowTensor {
                inner: inner.reference(),
            },
            inner,
        })
    }

    /// Declare a Seismic-owned host upload tensor from a checked input
    /// parameter. Its storage remains live from upload through its last use.
    pub fn input_for<E: Entry>(
        &mut self,
        kernel: &NativeKernel<E>,
        parameter: &str,
        dimensions: &[(&str, u64)],
    ) -> Result<NativePort, CallError> {
        let inner = self.inner.input_for(&kernel.inner, parameter, dimensions)?;
        Ok(NativePort {
            tensor: WorkflowTensor {
                inner: inner.reference(),
            },
            inner,
        })
    }

    /// Mark a checked local as filled before graph execution. Seismic keeps
    /// its storage live from the start, including across earlier nodes.
    pub fn prewrite(&mut self, port: &NativePort) -> Result<(), WorkflowError> {
        self.inner.prewrite(port.inner)
    }

    pub fn enqueue<E: Entry>(
        &mut self,
        kernel: &NativeKernel<E>,
        args: E::WorkflowArgs<'_>,
    ) -> Result<E::WorkflowResults, WorkflowError> {
        let pending = self
            .inner
            .enqueue(&kernel.inner, E::encode_workflow(args))?;
        Ok(E::decode_workflow(pending))
    }

    /// Keep a result beyond the graph run. Exported tensors receive separate
    /// storage so releasing a scratch slot does not reclaim a retained output.
    pub fn export(&mut self, result: &WorkflowTensor) -> Result<(), WorkflowError> {
        self.inner.export(result.inner)
    }

    pub fn seal(self) -> Result<NativeGraphPlan, CallError> {
        self.inner.seal().map(|inner| NativeGraphPlan {
            inner: Arc::new(inner),
        })
    }
}

#[derive(Clone)]
pub struct NativeGraphPlan {
    inner: Arc<seismic_runtime::native::graph::NativeGraphPlan>,
}

impl NativeGraphPlan {
    pub fn workspace_bytes(&self) -> u64 {
        self.inner.workspace_bytes()
    }
    pub fn output_bytes(&self) -> u64 {
        self.inner.output_bytes()
    }
    pub fn slot_storage_bytes(&self) -> u64 {
        self.inner.slot_storage_bytes()
    }
    /// Bytes of one submission's host-written inputs.
    pub fn upload_bytes(&self) -> u64 {
        self.inner.upload_bytes()
    }
    pub fn new_slot(&self) -> Result<NativeGraphSlot, TensorError> {
        self.inner.new_slot().map(|inner| NativeGraphSlot { inner })
    }
    pub fn new_outputs(&self) -> Result<NativeGraphOutputs, TensorError> {
        self.inner
            .new_outputs()
            .map(|inner| NativeGraphOutputs { inner })
    }
    pub fn bindings(&self) -> NativeGraphBindings {
        NativeGraphBindings {
            inner: self.inner.bindings(),
        }
    }
    pub fn bind_static(
        &self,
        fixed: &[(&NativePort, &Tensor)],
    ) -> Result<BoundNativeGraphPlan, CallError> {
        let fixed = fixed
            .iter()
            .map(|(port, tensor)| (port.inner, tensor.inner.clone()))
            .collect::<Vec<_>>();
        self.inner
            .bind_static(&fixed)
            .map(|inner| BoundNativeGraphPlan { inner })
    }
}

pub struct NativeGraphFamilyOutputSlot {
    inner: seismic_runtime::native::graph::NativeGraphFamilyOutputSlot,
}

impl NativeGraphFamilyOutputSlot {
    pub fn activate(self, plan: &NativeGraphPlan) -> Result<NativeGraphOutputs, WorkflowError> {
        self.inner
            .activate(&plan.inner)
            .map(|inner| NativeGraphOutputs { inner })
    }
}

pub struct BoundNativeGraphPlan {
    inner: seismic_runtime::native::graph::BoundNativeGraphPlan,
}

impl BoundNativeGraphPlan {
    pub fn bindings(&self) -> NativeGraphBindings {
        NativeGraphBindings {
            inner: self.inner.bindings(),
        }
    }
}

#[derive(Clone)]
pub struct NativeGraphFamily {
    inner: Arc<seismic_runtime::native::graph::NativeGraphFamily>,
}

impl NativeGraphFamily {
    pub fn new(plans: &[NativeGraphPlan]) -> Result<Self, WorkflowError> {
        let members = plans
            .iter()
            .map(|plan| plan.inner.clone())
            .collect::<Vec<_>>();
        seismic_runtime::native::graph::NativeGraphFamily::new(&members).map(|inner| Self {
            inner: Arc::new(inner),
        })
    }
    pub fn workspace_bytes(&self) -> u64 {
        self.inner.workspace_bytes()
    }
    pub fn output_bytes(&self) -> u64 {
        self.inner.output_bytes()
    }
    /// Bytes of one upload region; a slot holds one per run in flight.
    pub fn upload_bytes(&self) -> u64 {
        self.inner.upload_bytes()
    }
    pub fn new_output_slot(&self) -> Result<NativeGraphFamilyOutputSlot, TensorError> {
        self.inner
            .new_output_slot()
            .map(|inner| NativeGraphFamilyOutputSlot { inner })
    }
    /// A slot whose `regions` upload regions (one per run it keeps in flight
    /// at once) are allocated here; activation never allocates.
    pub fn new_slot(&self, regions: usize) -> Result<NativeGraphFamilySlot, TensorError> {
        self.inner
            .new_slot(regions)
            .map(|inner| NativeGraphFamilySlot { inner })
    }
}

pub struct NativeGraphFamilySlot {
    inner: seismic_runtime::native::graph::NativeGraphFamilySlot,
}

impl NativeGraphFamilySlot {
    pub fn activate<'a>(
        &'a mut self,
        plan: &NativeGraphPlan,
    ) -> Result<NativeGraphFamilyActive<'a>, WorkflowError> {
        self.inner
            .activate(&plan.inner)
            .map(|inner| NativeGraphFamilyActive { inner })
    }
}

pub struct NativeGraphFamilyActive<'a> {
    inner: seismic_runtime::native::graph::NativeGraphFamilyActive<'a>,
}

impl NativeGraphFamilyActive<'_> {
    pub fn local(&mut self, port: &NativePort) -> Option<Tensor> {
        self.inner.local(port.inner).map(|inner| Tensor { inner })
    }

    pub fn write_input(&mut self, port: &NativePort, bytes: &[u8]) -> Result<(), TensorError> {
        self.inner.write_input(port.inner, bytes)
    }
    pub fn attach(
        &mut self,
        bindings: NativeGraphBindings,
        outputs: NativeGraphOutputs,
    ) -> Result<ReadyNativeGraphRun<'_>, CallError> {
        self.inner
            .attach(bindings.inner, outputs.inner)
            .map(|inner| ReadyNativeGraphRun { inner })
    }
}

pub struct NativeGraphBindings {
    inner: seismic_runtime::native::graph::NativeGraphBindings,
}

impl NativeGraphBindings {
    pub fn set(&mut self, port: &NativePort, tensor: &Tensor) -> Result<(), WorkflowError> {
        self.inner.set(port.inner, tensor.inner.clone())
    }

    /// Bind an unsubmitted exported output lease as a graph destination.
    /// The tensor is only exposed to the checked graph attachment.
    pub fn set_reserved_export(
        &mut self,
        port: &NativePort,
        outputs: &NativeGraphOutputs,
        result: &WorkflowTensor,
    ) -> Result<(), WorkflowError> {
        self.inner
            .set_reserved_export(port.inner, &outputs.inner, result.inner)
    }
}

pub struct NativeGraphSlot {
    inner: seismic_runtime::native::graph::NativeGraphSlot,
}

impl NativeGraphSlot {
    pub fn write_input(&mut self, port: &NativePort, bytes: &[u8]) -> Result<(), TensorError> {
        self.inner.write_input(port.inner, bytes)
    }
    pub fn attach(
        &mut self,
        bindings: NativeGraphBindings,
        outputs: NativeGraphOutputs,
    ) -> Result<ReadyNativeGraphRun<'_>, CallError> {
        self.inner
            .attach(bindings.inner, outputs.inner)
            .map(|inner| ReadyNativeGraphRun { inner })
    }
}

pub struct NativeGraphOutputs {
    inner: seismic_runtime::native::graph::NativeGraphOutputs,
}

impl NativeGraphOutputs {
    pub fn recycle(self) -> Result<NativeGraphFamilyOutputSlot, WorkflowError> {
        self.inner
            .recycle()
            .map(|inner| NativeGraphFamilyOutputSlot { inner })
    }
    pub fn exported(&self, result: &WorkflowTensor) -> Option<Tensor> {
        self.inner
            .exported_tensor(result.inner)
            .map(|inner| Tensor { inner })
    }
}

pub struct ReadyNativeGraphRun<'a> {
    inner: seismic_runtime::native::graph::ReadyNativeGraphRun<'a>,
}

impl ReadyNativeGraphRun<'_> {
    /// Submit without waiting. The outputs can be bound into later runs at
    /// once; host reads of exported tensors wait for this run. The
    /// completion reports its outcome.
    pub fn submit(self) -> Result<(NativeGraphOutputs, NativeGraphCompletion), CallError> {
        self.inner.submit().map(|(inner, completion)| {
            (
                NativeGraphOutputs { inner },
                NativeGraphCompletion { inner: completion },
            )
        })
    }

    /// Append this run to `sequence` instead of submitting it: every run
    /// queued on a sequence is submitted as one unit of device work, in
    /// queue order. The outputs may be bound into later queued runs at
    /// once; host reads of them are valid only after the sequence is
    /// submitted.
    pub fn queue(
        self,
        sequence: &mut NativeGraphSequence,
    ) -> Result<NativeGraphOutputs, CallError> {
        self.inner
            .queue(&mut sequence.inner)
            .map(|inner| NativeGraphOutputs { inner })
    }
}

/// Graph runs queued for one submission (one Metal command buffer, one
/// CUDA graph launch), executed in queue order as if submitted one by one.
pub struct NativeGraphSequence {
    inner: seismic_runtime::native::graph::NativeGraphSequence,
}

impl NativeGraphSequence {
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
    /// Submit every queued run without waiting; the completion reports all
    /// of them.
    pub fn submit(self) -> Result<NativeGraphCompletion, CallError> {
        self.inner
            .submit()
            .map(|inner| NativeGraphCompletion { inner })
    }
}

/// The outcome of one submitted native graph run.
#[must_use = "a native graph completion reports whether the run succeeded"]
pub struct NativeGraphCompletion {
    inner: seismic_runtime::native::graph::NativeGraphCompletion,
}

impl NativeGraphCompletion {
    pub fn is_complete(&self) -> bool {
        self.inner.is_complete()
    }
    pub fn wait(self) -> Result<(), CallError> {
        self.inner.wait()
    }
}

/// The only workflow phase that accepts nodes. It owns symbolic producer edges
/// and no bound policy or live device resources.
pub struct WorkflowDraft {
    inner: seismic_runtime::api::kernel::WorkflowDraftAny,
}

/// A completely bound workflow. Selection, concrete output descriptors,
/// hazards, lifetimes, and allocation requirements are fixed; no capacity or
/// native resources have been acquired.
pub struct BoundWorkflow {
    inner: seismic_runtime::api::kernel::BoundWorkflowAny,
}

/// A graph-wide admitted workflow. It owns all reservations, allocations,
/// persistent leases, and access permits required for submission.
pub struct AdmittedWorkflow {
    inner: seismic_runtime::api::kernel::AdmittedWorkflowAny,
}

/// Completion owner for a submitted workflow. It retains the native execution
/// and all admitted resources until the first typed resolution completes it;
/// later result groups resolve from the completed table without resubmission.
pub struct WorkflowCompletion {
    inner: seismic_runtime::api::kernel::WorkflowCompletionAny,
}

impl WorkflowDraft {
    pub fn enqueue<E: Entry>(
        &mut self,
        kernel: &Kernel<E>,
        args: E::WorkflowArgs<'_>,
    ) -> Result<E::WorkflowResults, WorkflowError> {
        let encoded = E::encode_workflow(args);
        let pending =
            seismic_runtime::api::kernel::enqueue(&mut self.inner, &kernel.inner, encoded)?;
        Ok(E::decode_workflow(pending))
    }

    pub fn bind(self) -> Result<BoundWorkflow, CallError> {
        seismic_runtime::api::kernel::bind_workflow(self.inner).map(|inner| BoundWorkflow { inner })
    }
}

impl BoundWorkflow {
    pub fn admit(self) -> Result<AdmittedWorkflow, CallError> {
        seismic_runtime::api::kernel::admit_workflow(self.inner)
            .map(|inner| AdmittedWorkflow { inner })
    }
}

impl AdmittedWorkflow {
    pub fn submit(self) -> Result<WorkflowCompletion, CallError> {
        seismic_runtime::api::kernel::submit_workflow(self.inner)
            .map(|inner| WorkflowCompletion { inner })
    }
}

impl WorkflowCompletion {
    pub fn resolve<E: Entry>(&self, outputs: E::WorkflowResults) -> Result<E::Results, CallError> {
        let outputs = E::workflow_outputs(outputs);
        let decoded = self.inner.resolve(outputs)?;
        Ok(E::decode(decoded))
    }
}

impl<E: Entry> Kernel<E> {
    pub fn feedback_report(&self) -> Option<&FeedbackReport> {
        self.inner.feedback_report()
    }

    fn prepare(
        device: &Device,
        options: PreparationOptions,
        bindings: seismic_lang::entry::ElementBindings,
    ) -> Result<Self, LoadError> {
        let module = E::module().map_err(LoadError::Bundle)?;
        let entry = E::resolve(module).map_err(LoadError::Bundle)?;
        seismic_runtime::api::kernel::prepare(
            module.checked(),
            entry.id(),
            bindings,
            device.inner(),
            options,
        )
        .map(|inner| Self {
            inner: Arc::new(inner),
            marker: std::marker::PhantomData,
        })
        .map_err(LoadError::from_prepare)
    }

    pub fn call(&self, args: E::Args<'_>) -> Result<E::Results, CallError> {
        let encoded = E::encode(args);
        let decoded = seismic_runtime::api::kernel::call(&self.inner, encoded)?;
        Ok(E::decode(decoded))
    }
}

impl<E: Entry> NativeKernel<E> {
    /// Identity of the prepared storage shared by clones and by typed slots.
    /// Intended for ownership accounting; it has no cross-process meaning.
    #[doc(hidden)]
    pub fn prepared_storage_identity(&self) -> usize {
        Arc::as_ptr(&self.inner) as usize
    }

    /// Allocation-free planning charge before this checked specialization is
    /// prepared. The value is generated from its checked entry schema.
    pub const fn planned_invocation_workspace_bytes() -> u64 {
        E::NATIVE_INVOCATION_WORKSPACE_BYTES
    }

    /// Fixed device storage charged when this checked native specialization
    /// is prepared. Calls reuse it and do not allocate ABI or scalar buffers.
    pub fn invocation_workspace_bytes(&self) -> u64 {
        self.inner.invocation_workspace_bytes()
    }

    fn prepare(
        device: &Device,
        specialization: NativeSpecialization,
        bindings: seismic_lang::entry::ElementBindings,
        cpu: Option<&'static native_cpu::CpuNativeKernels>,
    ) -> Result<Self, LoadError> {
        let module = E::module().map_err(LoadError::Bundle)?;
        let entry = E::resolve(module).map_err(LoadError::Bundle)?;
        seismic_runtime::api::kernel::prepare_native(
            module.checked(),
            entry.id(),
            bindings,
            device.inner(),
            specialization,
            cpu,
        )
        .map(|inner| {
            debug_assert_eq!(
                inner.invocation_workspace_bytes(),
                E::NATIVE_INVOCATION_WORKSPACE_BYTES
            );
            Self {
                inner: Arc::new(inner),
                marker: std::marker::PhantomData,
            }
        })
        .map_err(LoadError::from_prepare)
    }

    /// The specialization this kernel was formed under.
    pub fn specialization(&self) -> &NativeSpecialization {
        self.inner.specialization()
    }

    /// What was formed: backend, bindings, specialization, source digest
    /// and toolchain.
    pub fn artifact(&self) -> &NativeArtifactIdentity {
        self.inner.artifact()
    }

    /// Device time of calls cycling through `rotation`.
    pub fn measure(
        &self,
        rotation: Vec<E::Args<'_>>,
        options: &MeasureOptions,
    ) -> Result<Measurement, CallError> {
        self.inner
            .measure(rotation.into_iter().map(E::encode).collect(), options)
    }

    pub fn call(&self, args: E::Args<'_>) -> Result<E::Results, CallError> {
        let encoded = E::encode(args);
        let decoded = seismic_runtime::api::kernel::call_native(&self.inner, encoded)?;
        Ok(E::decode(decoded))
    }

    /// Execute into caller-owned tensor results after validating their checked
    /// representation, device, and extents. Scalar results remain returned.
    pub fn call_into(
        &self,
        args: E::Args<'_>,
        outputs: E::OutputArgs<'_>,
    ) -> Result<E::Results, CallError> {
        let encoded = E::encode(args);
        let encoded_outputs = E::encode_outputs(outputs);
        let decoded =
            seismic_runtime::api::kernel::call_native_into(&self.inner, encoded, encoded_outputs)?;
        Ok(E::decode(decoded))
    }
}

impl<E: Entry> Clone for NativeKernel<E> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            marker: std::marker::PhantomData,
        }
    }
}

impl<E: Entry> fmt::Debug for NativeKernel<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NativeKernel")
            .field("entry", &E::NAME)
            .finish()
    }
}

impl<E: Entry> Clone for Kernel<E> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            marker: std::marker::PhantomData,
        }
    }
}

impl<E: Entry> fmt::Debug for Kernel<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Kernel").field("entry", &E::NAME).finish()
    }
}

/// Helpers generated code composes. Not for hand-written use.
#[doc(hidden)]
pub mod generated {
    use super::*;

    pub use std::sync::OnceLock;

    use seismic_runtime::api::kernel::EncodedScalar;
    pub use seismic_runtime::api::kernel::{
        DecodedResults, EncodedArgs, EncodedOutputs, EncodedWorkflowArgs, PendingWorkflowResults,
        WorkflowResultRef,
    };

    /// Opaque checked module token used only by generated bindings. Consumers
    /// can name the type because Rust trait implementations must, but cannot
    /// inspect or construct the compiler artifact it owns.
    pub struct Module {
        checked: seismic_lang::checked::CheckedModule,
        /// A checked entry's semantics depend only on its entry identity and
        /// element bindings. Graph metadata asks for many port shapes under
        /// the same binding, so retain a bounded set of these immutable
        /// monomorphizations instead of rebuilding one per port and node.
        logical_entries: std::sync::Mutex<std::collections::VecDeque<LogicalEntryCacheItem>>,
    }

    struct LogicalEntryCacheItem {
        entry: seismic_lang::ids::EntryId,
        bindings: seismic_lang::entry::ElementBindings,
        logical: Arc<seismic_lang::entry::LogicalEntry>,
        compiled_shapes: Arc<OnceLock<Arc<seismic_lang::entry::CompiledEntryShapes>>>,
    }

    const LOGICAL_ENTRY_CACHE_CAPACITY: usize = 64;

    /// Opaque proof that a generated entry name resolved in its own checked
    /// module. Generated code can pass it back to Seismic, but consumers
    /// never observe or fabricate compiler entry identifiers.
    pub struct EntryToken(seismic_lang::ids::EntryId);

    pub use seismic_compiler::feedback::InvocationParameter;
    pub use seismic_lang::expr::SymbolValue;

    pub fn invocation_scope<E: Entry>() -> Result<InvocationScope, CheckedBundleError> {
        let module = E::module()?;
        let entry = E::resolve(module)?;
        let stable = module
            .checked()
            .entries()
            .iter()
            .find(|candidate| candidate.id == entry.id())
            .expect("resolved entry belongs to module")
            .stable;
        Ok(InvocationScope::for_entry(stable))
    }

    impl EntryToken {
        pub(super) fn id(&self) -> seismic_lang::ids::EntryId {
            self.0
        }
    }

    impl Module {
        fn new(checked: seismic_lang::checked::CheckedModule) -> Self {
            Self {
                checked,
                logical_entries: std::sync::Mutex::new(std::collections::VecDeque::new()),
            }
        }

        pub(crate) fn checked(&self) -> &seismic_lang::checked::CheckedModule {
            &self.checked
        }
        #[doc(hidden)]
        pub fn entry_named(&self, name: &str) -> Option<EntryToken> {
            self.checked.entry_named(name).map(EntryToken)
        }

        fn logical_entry(
            &self,
            entry: seismic_lang::ids::EntryId,
            bindings: &seismic_lang::entry::ElementBindings,
        ) -> Result<Arc<seismic_lang::entry::LogicalEntry>, seismic_lang::checked::SourceError>
        {
            if let Some(logical) = self
                .logical_entries
                .lock()
                .expect("checked-entry cache mutex poisoned")
                .iter()
                .find(|item| item.entry == entry && item.bindings == *bindings)
                .map(|item| item.logical.clone())
            {
                return Ok(logical);
            }
            let logical = Arc::new(self.checked.entry(entry, bindings)?);
            let mut cache = self
                .logical_entries
                .lock()
                .expect("checked-entry cache mutex poisoned");
            if let Some(item) = cache
                .iter()
                .find(|item| item.entry == entry && item.bindings == *bindings)
            {
                return Ok(item.logical.clone());
            }
            if cache.len() == LOGICAL_ENTRY_CACHE_CAPACITY {
                cache.pop_front();
            }
            cache.push_back(LogicalEntryCacheItem {
                entry,
                bindings: bindings.clone(),
                logical: logical.clone(),
                compiled_shapes: Arc::new(OnceLock::new()),
            });
            Ok(logical)
        }

        fn compiled_entry_shapes(
            &self,
            entry: seismic_lang::ids::EntryId,
            bindings: &seismic_lang::entry::ElementBindings,
        ) -> Result<Arc<seismic_lang::entry::CompiledEntryShapes>, seismic_lang::checked::SourceError>
        {
            let logical = self.logical_entry(entry, bindings)?;
            let cell = self
                .logical_entries
                .lock()
                .expect("checked-entry cache mutex poisoned")
                .iter()
                .find(|item| item.entry == entry && item.bindings == *bindings)
                .map(|item| item.compiled_shapes.clone());
            Ok(match cell {
                Some(cell) => cell
                    .get_or_init(|| Arc::new(logical.compile_tensor_shapes()))
                    .clone(),
                None => Arc::new(logical.compile_tensor_shapes()),
            })
        }

        #[cfg(test)]
        fn cached_logical_entries(&self) -> usize {
            self.logical_entries
                .lock()
                .expect("checked-entry cache mutex poisoned")
                .len()
        }
    }

    #[cfg(test)]
    mod logical_entry_cache_tests {
        use super::*;
        use seismic_lang::checked::{check_source, SourceFile, SourceSet};

        #[test]
        fn reuses_one_semantic_entry_per_element_binding() {
            let source = SourceSet::new(vec![SourceFile {
                path: "cache.seismic".into(),
                text:
                    "fn copy[N](input: &tensor[N] A) -> tensor[N] A:\n    return to_owned(input)\n"
                        .into(),
            }]);
            let module = Module::new(check_source(source).unwrap());
            let entry = module.entry_named("copy").unwrap().id();
            let f32_binding =
                seismic_lang::entry::ElementBindings::new().bind("A", Element::f32().id());
            let first = module.logical_entry(entry, &f32_binding).unwrap();
            let again = module.logical_entry(entry, &f32_binding).unwrap();
            assert!(Arc::ptr_eq(&first, &again));
            assert_eq!(module.cached_logical_entries(), 1);
        }
    }

    /// Opaque generated-code encoder. It preserves parameter order without
    /// exposing compiler argument enums or mutable ABI vectors to consumers.
    pub struct ArgsEncoder {
        inner: EncodedArgs,
    }

    impl ArgsEncoder {
        pub fn new() -> Self {
            Self {
                inner: EncodedArgs::new(),
            }
        }
        pub fn tensor(&mut self, tensor: &Tensor) {
            self.inner.push_tensor(tensor.inner().clone());
        }
        pub fn f32(&mut self, value: f32) {
            self.inner.push_scalar(EncodedScalar::from_f32(value));
        }
        pub fn f16(&mut self, value: F16) {
            self.inner.push_scalar(EncodedScalar::F16(value.to_bits()));
        }
        pub fn bf16(&mut self, value: BF16) {
            self.inner.push_scalar(EncodedScalar::BF16(value.to_bits()));
        }
        pub fn i32(&mut self, value: i32) {
            self.inner.push_scalar(EncodedScalar::I32(value));
        }
        pub fn u32(&mut self, value: u32) {
            self.inner.push_scalar(EncodedScalar::U32(value));
        }
        pub fn bool(&mut self, value: bool) {
            self.inner.push_scalar(EncodedScalar::Bool(value));
        }
        pub fn index(&mut self, value: BigUint) {
            self.inner.push_scalar(EncodedScalar::Index(value));
        }
        pub fn range(&mut self, value: (BigUint, BigUint)) {
            self.inner.push_scalar(EncodedScalar::Range {
                start: value.0,
                end: value.1,
            });
        }
        pub fn finish(self) -> EncodedArgs {
            self.inner
        }
    }

    /// Generated-code encoder for caller-supplied native result tensors.
    pub struct OutputArgsEncoder {
        inner: EncodedOutputs,
    }

    impl OutputArgsEncoder {
        pub fn new() -> Self {
            Self {
                inner: EncodedOutputs::new(),
            }
        }
        pub fn tensor(&mut self, tensor: &mut Tensor) {
            self.inner.push_tensor(tensor.inner().clone());
        }
        pub fn finish(self) -> EncodedOutputs {
            self.inner
        }
    }

    pub struct WorkflowArgsEncoder {
        inner: EncodedWorkflowArgs,
    }

    impl WorkflowArgsEncoder {
        pub fn new() -> Self {
            Self {
                inner: EncodedWorkflowArgs::new(),
            }
        }
        pub fn shared_tensor(&mut self, tensor: WorkflowTensorRef<'_>) {
            match tensor {
                WorkflowTensorRef::External(tensor) => {
                    self.inner.push_external_tensor(tensor.inner().clone())
                }
                WorkflowTensorRef::Result(result) => self.inner.push_result_tensor(result.inner),
                WorkflowTensorRef::View(view) => self
                    .inner
                    .push_result_tensor_view(view.inner, view.operations.clone()),
            }
        }
        pub fn mutable_tensor(&mut self, tensor: WorkflowTensorMut<'_>) {
            match tensor {
                WorkflowTensorMut::External(tensor) => {
                    self.inner.push_external_tensor(tensor.inner().clone())
                }
                WorkflowTensorMut::Result(result) => self.inner.push_result_tensor(result.inner),
                WorkflowTensorMut::View(view) => self
                    .inner
                    .push_result_tensor_view(view.inner, view.operations.clone()),
            }
        }
        pub fn owned_tensor(&mut self, tensor: WorkflowTensorOwned<'_>) {
            match tensor.value {
                WorkflowTensorOwnedValue::External(tensor) => {
                    self.inner.push_external_tensor(tensor.inner().clone())
                }
                WorkflowTensorOwnedValue::Result(result) => {
                    self.inner.push_result_tensor(result.inner)
                }
            }
        }
        pub fn f32(&mut self, value: f32) {
            self.inner.push_scalar(EncodedScalar::from_f32(value));
        }
        pub fn f16(&mut self, value: F16) {
            self.inner.push_scalar(EncodedScalar::F16(value.to_bits()));
        }
        pub fn bf16(&mut self, value: BF16) {
            self.inner.push_scalar(EncodedScalar::BF16(value.to_bits()));
        }
        pub fn i32(&mut self, value: i32) {
            self.inner.push_scalar(EncodedScalar::I32(value));
        }
        pub fn u32(&mut self, value: u32) {
            self.inner.push_scalar(EncodedScalar::U32(value));
        }
        pub fn bool(&mut self, value: bool) {
            self.inner.push_scalar(EncodedScalar::Bool(value));
        }
        pub fn index(&mut self, value: BigUint) {
            self.inner.push_scalar(EncodedScalar::Index(value));
        }
        pub fn range(&mut self, value: (BigUint, BigUint)) {
            self.inner.push_scalar(EncodedScalar::Range {
                start: value.0,
                end: value.1,
            });
        }
        pub fn scalar_result<T>(&mut self, value: &WorkflowScalar<T>) {
            self.inner.push_result_scalar(value.inner)
        }
        pub fn finish(self) -> EncodedWorkflowArgs {
            self.inner
        }
    }

    pub fn take_workflow_tensor(results: &mut PendingWorkflowResults) -> WorkflowTensor {
        WorkflowTensor {
            inner: results.take(),
        }
    }

    pub fn take_workflow_scalar<T>(results: &mut PendingWorkflowResults) -> WorkflowScalar<T> {
        WorkflowScalar {
            inner: results.take(),
            marker: std::marker::PhantomData,
        }
    }

    pub fn workflow_tensor_ref(value: WorkflowTensor) -> WorkflowResultRef {
        value.inner
    }
    pub fn workflow_scalar_ref<T>(value: WorkflowScalar<T>) -> WorkflowResultRef {
        value.inner
    }

    pub fn prepare<E: Entry>(
        device: &Device,
        options: PreparationOptions,
        elements: &[(&str, Element)],
    ) -> Result<Kernel<E>, LoadError> {
        let bindings = elements.iter().fold(
            seismic_lang::entry::ElementBindings::new(),
            |bindings, (name, element)| bindings.bind(name, element.id()),
        );
        Kernel::prepare(device, options, bindings)
    }

    pub fn start_feedback<'device, E: Entry>(
        device: &'device Device,
        precision: PrecisionPolicy,
        options: FeedbackOptions,
        elements: &[(&str, Element)],
    ) -> Result<(FeedbackPreparation<'device, E>, Kernel<E>), LoadError> {
        let bindings = elements.iter().fold(
            seismic_lang::entry::ElementBindings::new(),
            |bindings, (name, element)| bindings.bind(name, element.id()),
        );
        FeedbackPreparation::start(device, precision, options, bindings)
    }

    fn element_bindings(elements: &[(&str, Element)]) -> seismic_lang::entry::ElementBindings {
        elements.iter().fold(
            seismic_lang::entry::ElementBindings::new(),
            |bindings, (name, element)| bindings.bind(name, element.id()),
        )
    }

    pub fn prepare_native<E: Entry>(
        device: &Device,
        specialization: &NativeSpecialization,
        elements: &[(&str, Element)],
        cpu: Option<&'static native_cpu::CpuNativeKernels>,
    ) -> Result<NativeKernel<E>, LoadError> {
        NativeKernel::prepare(
            device,
            specialization.clone(),
            element_bindings(elements),
            cpu,
        )
    }

    /// The checked native implementation of an entry for a backend, without
    /// opening a device. Metadata-only assessment can inspect the same
    /// declaration that execution later prepares.
    pub fn native_implementation_for_backend<E: Entry>(
        backend: BackendName,
    ) -> Result<Option<NativeImplementation>, CheckedBundleError> {
        let module = E::module()?;
        let entry = E::resolve(module)?;
        Ok(module
            .checked()
            .native_implementation(entry.id(), backend)
            .cloned())
    }

    /// Resolve one metadata graph call from one checked entry and one set of
    /// element bindings. The graph still validates every argument edge and
    /// Seismic still places its storage when the graph seals.
    pub enum CheckedNativeGraphCall {
        Checked {
            implementation: NativeImplementation,
            parameters: Vec<Option<NativeTensorMetadata>>,
            results: Vec<Option<NativeTensorMetadata>>,
        },
        Unsupported(String),
    }

    /// A checked entry and its backend declaration resolved once for a graph
    /// resource template. Numeric dimensions can then be evaluated without
    /// repeating module resolution or element monomorphization.
    pub struct NativeGraphCheckedEntry {
        shapes: Arc<seismic_lang::entry::CompiledEntryShapes>,
        implementation: NativeImplementation,
    }

    impl NativeGraphCheckedEntry {
        pub fn bind<E: Entry>(
            backend: BackendName,
            elements: &[(&str, Element)],
        ) -> Result<Result<Self, String>, CheckedBundleError> {
            let module = E::module()?;
            let entry = E::resolve(module)?;
            let Some(implementation) = module
                .checked()
                .native_implementation(entry.id(), backend)
                .cloned()
            else {
                return Ok(Err(format!(
                    "`{}` has no native implementation for `{}`",
                    E::NAME,
                    backend.as_str()
                )));
            };
            let shapes = match module.compiled_entry_shapes(entry.id(), &element_bindings(elements))
            {
                Ok(shapes) => shapes,
                Err(error) => return Ok(Err(error.to_string())),
            };
            Ok(Ok(Self {
                shapes,
                implementation,
            }))
        }

        pub fn implementation(&self) -> &NativeImplementation {
            &self.implementation
        }

        /// The entry dimensions a tensor parameter's checked shape reads.
        pub fn parameter_dimensions(&self, name: &str) -> Result<Vec<&str>, String> {
            self.shapes
                .parameter_dimensions(name)
                .map_err(|error| error.to_string())
        }

        /// The entry dimensions a tensor result's checked shape reads.
        pub fn result_dimensions(&self, ordinal: usize) -> Result<Vec<&str>, String> {
            self.shapes
                .result_dimensions(ordinal)
                .map_err(|error| error.to_string())
        }

        pub fn result_count(&self) -> usize {
            self.shapes.result_count()
        }

        pub fn result(
            &self,
            ordinal: usize,
            dimensions: &[(&str, u64)],
        ) -> Result<NativeTensorMetadata, String> {
            self.shapes
                .result_shape(ordinal, dimensions)
                .map_err(|error| error.to_string())
                .and_then(tensor_metadata)
        }

        pub fn parameter(
            &self,
            name: &str,
            dimensions: &[(&str, u64)],
        ) -> Result<NativeTensorMetadata, String> {
            self.shapes
                .parameter_shape(name, dimensions)
                .map_err(|error| error.to_string())
                .and_then(tensor_metadata)
        }

        pub fn shapes(
            &self,
            dimensions: &[(&str, u64)],
        ) -> Result<
            (
                Vec<Option<NativeTensorMetadata>>,
                Vec<Option<NativeTensorMetadata>>,
            ),
            String,
        > {
            let (parameters, results) = self
                .shapes
                .all_shapes(dimensions)
                .map_err(|error| error.to_string())?;
            let parameters = parameters
                .into_iter()
                .map(|shape| shape.map(tensor_metadata).transpose())
                .collect::<Result<Vec<_>, _>>()?;
            let results = results
                .into_iter()
                .map(|shape| shape.map(tensor_metadata).transpose())
                .collect::<Result<Vec<_>, _>>()?;
            Ok((parameters, results))
        }
    }

    fn tensor_metadata(
        shape: seismic_lang::entry::CheckedTensorShape,
    ) -> Result<NativeTensorMetadata, String> {
        let element = Element(shape.representation);
        let canonical_bytes = element
            .canonical_byte_len(&shape.extents)
            .map_err(|error| error.to_string())?;
        Ok(NativeTensorMetadata {
            element,
            extents: shape.extents,
            canonical_bytes,
        })
    }

    pub fn checked_native_graph_call<E: Entry>(
        backend: BackendName,
        elements: &[(&str, Element)],
        dimensions: &[(&str, u64)],
    ) -> Result<CheckedNativeGraphCall, CheckedBundleError> {
        let checked = match NativeGraphCheckedEntry::bind::<E>(backend, elements)? {
            Ok(checked) => checked,
            Err(reason) => return Ok(CheckedNativeGraphCall::Unsupported(reason)),
        };
        let (parameters, results) = match checked.shapes(dimensions) {
            Ok(shapes) => shapes,
            Err(reason) => return Ok(CheckedNativeGraphCall::Unsupported(reason)),
        };
        Ok(CheckedNativeGraphCall::Checked {
            implementation: checked.implementation.clone(),
            parameters,
            results,
        })
    }

    /// Check an entry's exact element bindings with the same checked-module
    /// operation used by native preparation, without opening a device or
    /// forming backend code. A successful check proves the semantic binding,
    /// not that a backend compiler can form the native implementation.
    pub fn checked_native_binding<E: Entry>(
        backend: BackendName,
        elements: &[(&str, Element)],
    ) -> Result<NativeBindingCheck, CheckedBundleError> {
        let module = E::module()?;
        let entry = E::resolve(module)?;
        if module
            .checked()
            .native_implementation(entry.id(), backend)
            .is_none()
        {
            return Ok(NativeBindingCheck::Unsupported(format!(
                "`{}` has no native implementation for `{}`",
                E::NAME,
                backend.as_str()
            )));
        }
        Ok(module
            .logical_entry(entry.id(), &element_bindings(elements))
            .map(|_| NativeBindingCheck::AcceptedByCheckedEntry)
            .unwrap_or_else(|error| NativeBindingCheck::Unsupported(error.to_string())))
    }

    /// Resolve one native graph port from the checked entry and registry,
    /// without opening a device or preparing a native kernel. This checks a
    /// port's semantic shape and canonical storage, not graph liveness or
    /// backend formation.
    pub fn checked_native_tensor_parameter<E: Entry>(
        backend: BackendName,
        elements: &[(&str, Element)],
        parameter: &str,
        dimensions: &[(&str, u64)],
    ) -> Result<NativeTensorParameterCheck, CheckedBundleError> {
        let module = E::module()?;
        let entry = E::resolve(module)?;
        if module
            .checked()
            .native_implementation(entry.id(), backend)
            .is_none()
        {
            return Ok(NativeTensorParameterCheck::Unsupported(format!(
                "`{}` has no native implementation for `{}`",
                E::NAME,
                backend.as_str()
            )));
        }
        let logical = match module.logical_entry(entry.id(), &element_bindings(elements)) {
            Ok(logical) => logical,
            Err(error) => return Ok(NativeTensorParameterCheck::Unsupported(error.to_string())),
        };
        let shape = match logical.tensor_parameter_shape(parameter, dimensions) {
            Ok(shape) => shape,
            Err(error) => return Ok(NativeTensorParameterCheck::Unsupported(error.to_string())),
        };
        let element = Element(shape.representation);
        let canonical_bytes = match element.canonical_byte_len(&shape.extents) {
            Ok(bytes) => bytes,
            Err(error) => return Ok(NativeTensorParameterCheck::Unsupported(error.to_string())),
        };
        Ok(NativeTensorParameterCheck::Checked(NativeTensorMetadata {
            element,
            extents: shape.extents,
            canonical_bytes,
        }))
    }

    /// Resolve checked result storage for a native entry at exact dimensions.
    /// This does not account for the result's graph lifetime or placement.
    pub fn checked_native_tensor_results<E: Entry>(
        backend: BackendName,
        elements: &[(&str, Element)],
        dimensions: &[(&str, u64)],
    ) -> Result<NativeTensorResultsCheck, CheckedBundleError> {
        let module = E::module()?;
        let entry = E::resolve(module)?;
        if module
            .checked()
            .native_implementation(entry.id(), backend)
            .is_none()
        {
            return Ok(NativeTensorResultsCheck::Unsupported(format!(
                "`{}` has no native implementation for `{}`",
                E::NAME,
                backend.as_str()
            )));
        }
        let logical = match module.logical_entry(entry.id(), &element_bindings(elements)) {
            Ok(logical) => logical,
            Err(error) => return Ok(NativeTensorResultsCheck::Unsupported(error.to_string())),
        };
        let shapes = match logical.tensor_result_shapes(dimensions) {
            Ok(shapes) => shapes,
            Err(error) => return Ok(NativeTensorResultsCheck::Unsupported(error.to_string())),
        };
        let mut results = Vec::with_capacity(shapes.len());
        for shape in shapes {
            let Some(shape) = shape else {
                results.push(None);
                continue;
            };
            let element = Element(shape.representation);
            let canonical_bytes = match element.canonical_byte_len(&shape.extents) {
                Ok(bytes) => bytes,
                Err(error) => {
                    return Ok(NativeTensorResultsCheck::Unsupported(error.to_string()));
                }
            };
            results.push(Some(NativeTensorMetadata {
                element,
                extents: shape.extents,
                canonical_bytes,
            }));
        }
        Ok(NativeTensorResultsCheck::Checked(results))
    }

    /// Parameter shapes in checked schema order, for a metadata graph node.
    /// Scalar parameters have no tensor descriptor.
    pub fn checked_native_tensor_parameters<E: Entry>(
        backend: BackendName,
        elements: &[(&str, Element)],
        dimensions: &[(&str, u64)],
    ) -> Result<NativeTensorResultsCheck, CheckedBundleError> {
        use seismic_lang::entry::ParameterKind;

        let module = E::module()?;
        let entry = E::resolve(module)?;
        if module
            .checked()
            .native_implementation(entry.id(), backend)
            .is_none()
        {
            return Ok(NativeTensorResultsCheck::Unsupported(format!(
                "`{}` has no native implementation for `{}`",
                E::NAME,
                backend.as_str()
            )));
        }
        let logical = match module.logical_entry(entry.id(), &element_bindings(elements)) {
            Ok(logical) => logical,
            Err(error) => return Ok(NativeTensorResultsCheck::Unsupported(error.to_string())),
        };
        let mut parameters = Vec::with_capacity(logical.schema().parameters().len());
        for parameter in logical.schema().parameters() {
            if !matches!(&parameter.kind, ParameterKind::Tensor { .. }) {
                parameters.push(None);
                continue;
            }
            let shape = match logical.tensor_parameter_shape(&parameter.name, dimensions) {
                Ok(shape) => shape,
                Err(error) => {
                    return Ok(NativeTensorResultsCheck::Unsupported(error.to_string()));
                }
            };
            let element = Element(shape.representation);
            let canonical_bytes = match element.canonical_byte_len(&shape.extents) {
                Ok(bytes) => bytes,
                Err(error) => {
                    return Ok(NativeTensorResultsCheck::Unsupported(error.to_string()));
                }
            };
            parameters.push(Some(NativeTensorMetadata {
                element,
                extents: shape.extents,
                canonical_bytes,
            }));
        }
        Ok(NativeTensorResultsCheck::Checked(parameters))
    }

    /// The checked native implementation of an entry for a device's backend.
    pub fn native_implementation<E: Entry>(
        device: &Device,
    ) -> Result<Option<NativeImplementation>, CheckedBundleError> {
        native_implementation_for_backend::<E>(device.backend())
    }

    /// Validate a stored native choice against the opened device's complete
    /// parameter domain, including Seismic-owned CPU workers and ISA tier.
    pub fn native_specialization_valid<E: Entry>(
        device: &Device,
        specialization: &NativeSpecialization,
    ) -> Result<bool, CheckedBundleError> {
        let module = E::module()?;
        let entry = E::resolve(module)?;
        Ok(seismic_runtime::native::specialization_valid(
            device.inner(),
            module.checked(),
            entry.id(),
            specialization,
        ))
    }

    /// Digest of the entry's implementation for `device`'s backend at these
    /// bindings and static values, for keying stored tuning results.
    pub fn digest_native<E: Entry>(
        device: &Device,
        statics: &NativeSpecialization,
        elements: &[(&str, Element)],
        cpu: Option<&'static native_cpu::CpuNativeKernels>,
    ) -> Result<String, TuneError> {
        let module = E::module().map_err(|error| TuneError::Declaration(error.to_string()))?;
        let entry =
            E::resolve(module).map_err(|error| TuneError::Declaration(error.to_string()))?;
        seismic_runtime::native::tune::implementation_digest(
            device.inner(),
            module.checked(),
            entry.id(),
            &element_bindings(elements),
            statics,
            cpu,
        )
    }

    pub fn tune_native<E: Entry>(
        device: &Device,
        statics: &NativeSpecialization,
        elements: &[(&str, Element)],
        cpu: Option<&'static native_cpu::CpuNativeKernels>,
        points: Vec<TuningPoint<'_, E>>,
        validation: Validation,
        strategy: Strategy,
    ) -> Result<TuningResult, TuneError> {
        let module = E::module().map_err(|error| TuneError::Declaration(error.to_string()))?;
        let entry =
            E::resolve(module).map_err(|error| TuneError::Declaration(error.to_string()))?;
        seismic_runtime::native::tune::tune(seismic_runtime::native::tune::TuneRequest {
            device: device.inner(),
            module: module.checked(),
            entry: entry.id(),
            bindings: element_bindings(elements),
            statics: statics.clone(),
            cpu,
            points: points
                .into_iter()
                .map(|point| seismic_runtime::native::tune::TuningPoint {
                    label: point.label,
                    weight: point.weight,
                    class: point.class,
                    rotation: point.rotation.into_iter().map(E::encode).collect(),
                    initialize: point.initialize,
                })
                .collect(),
            validation,
            strategy,
        })
    }

    fn tensor_result(inner: Arc<seismic_runtime::api::tensor::TensorInner>) -> Tensor {
        Tensor { inner }
    }

    pub fn take_tensor(results: &mut DecodedResults) -> Tensor {
        let inner = results.take_tensor();
        tensor_result(inner)
    }

    macro_rules! scalar_result {
        ($name:ident, $variant:ident, $ty:ty, $map:expr) => {
            pub fn $name(results: &mut DecodedResults) -> $ty {
                match results.take_scalar() {
                    ArgumentValue::$variant(value) => ($map)(value),
                    _ => panic!("generated result schema disagrees with prepared kernel"),
                }
            }
        };
    }
    scalar_result!(take_f32, F32, f32, |value| value);
    scalar_result!(take_f16, F16, F16, F16::from_bits);
    scalar_result!(take_bf16, BF16, BF16, BF16::from_bits);
    scalar_result!(take_i32, I32, i32, |value| value);
    scalar_result!(take_u32, U32, u32, |value| value);
    scalar_result!(take_bool, Bool, bool, |value| value);
    scalar_result!(take_index, Index, BigUint, |value| value);

    pub fn take_range(results: &mut DecodedResults) -> (BigUint, BigUint) {
        match results.take_scalar() {
            ArgumentValue::Range { start, end } => (start, end),
            _ => panic!("generated result schema disagrees with prepared kernel"),
        }
    }

    pub fn module_from_bundle(
        cell: &'static OnceLock<Result<Module, CheckedBundleError>>,
        bytes: &'static [u8],
    ) -> Result<&'static Module, CheckedBundleError> {
        match cell
            .get_or_init(|| seismic_lang::bundle::decode_checked_bundle(bytes).map(Module::new))
        {
            Ok(module) => Ok(module),
            Err(error) => Err(error.clone()),
        }
    }
}

/// Checked dynamic integration for runtime-discovered functions.
pub mod dynamic;

/// Numerical policy values shared by every host language.
pub mod precision {
    pub use seismic_lang::precision::*;
}
