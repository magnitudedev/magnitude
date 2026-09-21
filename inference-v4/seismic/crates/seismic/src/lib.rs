//! The public Seismic API (spec §14): devices, tensors, prepared kernels,
//! and the helpers generated bindings compose. This crate re-exports and
//! composes; it contains no second implementation.
//!
//! Consumers import only this crate and `seismic-build`. Nothing here
//! exposes `PlanSpace`, `FrozenPlan`, `ExecutableVariant`, target domains,
//! solver decisions, raw backend buffers, binding indices, or manual arena
//! allocation (§14.6).
//!
//! W9-B owns device/tensor internals, W9-C owns invocation, W9-A owns the
//! generated-code contract. The surface below is frozen.

pub use seismic_compiler::errors::{
    CheckedBundleError, ExecutionError, InvocationError, PreparationError, TargetError,
};
pub use seismic_lang::precision::PrecisionPolicy;
pub use seismic_lang::registry::BackendName;
pub use seismic_lang::types::DType;
pub use seismic_runtime::api::{
    CallError, DeviceId, DeviceInfo, MemoryLimitError, MemoryUsage, TensorError, WorkflowError,
};

use seismic_compiler::prepared::ArgumentValue;
use seismic_lang::ids::RepresentationId;
use std::fmt;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Devices
// ---------------------------------------------------------------------------

pub struct DeviceCatalog {
    inner: seismic_runtime::api::catalog::Catalog,
}

impl DeviceCatalog {
    pub fn discover() -> Result<Self, TargetError> {
        seismic_runtime::api::catalog::Catalog::discover().map(|inner| Self { inner })
    }
    pub fn devices(&self) -> &[DeviceInfo] {
        self.inner.devices()
    }
    pub fn open(&self, id: DeviceId) -> Result<Device, TargetError> {
        self.inner.open(id).map(|inner| Device { inner })
    }
    /// Opens the first discovered device for a backend. Discovery order is
    /// stable within one catalog; callers that care about a particular
    /// physical device use `devices()` and `open()` instead.
    pub fn open_backend(&self, backend: BackendName) -> Result<Device, TargetError> {
        let id = self
            .devices()
            .iter()
            .find(|device| device.backend == backend)
            .map(|device| device.id)
            .ok_or_else(|| {
                TargetError::UnsupportedDevice(format!(
                    "no {} device was discovered",
                    backend.as_str()
                ))
            })?;
        self.open(id)
    }
}

impl fmt::Debug for DeviceCatalog {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceCatalog")
            .field("devices", &self.devices())
            .finish()
    }
}

/// One open device. Cloning shares the device.
#[derive(Clone)]
pub struct Device {
    inner: Arc<seismic_runtime::api::device::DeviceInner>,
}

impl Device {
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
    pub fn memory_usage(&self) -> MemoryUsage {
        self.inner.memory_usage()
    }
    pub fn set_memory_limit(&self, limit: Option<u64>) -> Result<(), MemoryLimitError> {
        self.inner.set_memory_limit(limit)
    }
    pub fn workflow(&self) -> Workflow {
        Workflow {
            inner: seismic_runtime::api::kernel::workflow(&self.inner),
        }
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
    pub fn named(name: &str) -> Option<Element> {
        seismic_lang::registry::representation(name).map(Element)
    }
    pub fn name(&self) -> &'static str {
        seismic_lang::registry::representation_info(self.0).name
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
            | seismic_lang::registry::RepresentationKind::External(_) => None,
        }
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
#[derive(Clone)]
pub struct Tensor {
    inner: Arc<seismic_runtime::api::tensor::TensorInner>,
}

impl Tensor {
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
    pub fn read_to_host(&self) -> Result<Vec<u8>, ExecutionError> {
        self.inner.read_to_host()
    }
    /// Replaces the bytes of this exact tensor view.  The byte count must
    /// match its canonical representation layout; allocation-level access
    /// exclusion makes this safe even when other views share the allocation.
    pub fn write_from_host(&mut self, bytes: &[u8]) -> Result<(), TensorError> {
        self.inner.write_from_host(bytes)
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
    start: u64,
    end: u64,
}

impl WorkflowTensor {
    /// A symbolic contiguous view of the leading axis. Bounds and packet
    /// alignment are validated during whole-workflow admission, once the
    /// producer's exact result descriptor exists.
    pub fn slice_leading(&self, start: u64, end: u64) -> WorkflowTensorView {
        WorkflowTensorView {
            inner: self.inner,
            start,
            end,
        }
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
/// checked bundle by `seismic-build`. `Args`, `Results`, `encode`, `decode`
/// and `resolve` must describe that bundle entry exactly. Safe consumers
/// never implement this trait; the unsafe boundary is what lets the runtime
/// treat a schema mismatch as a compiler/generator invariant rather than a
/// recoverable invocation condition.
pub unsafe trait Entry: 'static {
    /// Typed arguments.
    type Args<'a>;
    /// Typed results.
    type Results;
    /// Typed arguments whose tensor/scalar leaves may reference results of
    /// earlier nodes in the same workflow draft.
    type WorkflowArgs<'a>;
    /// Typed symbolic results returned while constructing a workflow.
    type WorkflowResults;
    const NAME: &'static str;
    /// The checked module this entry belongs to.
    fn module() -> Result<&'static generated::Module, CheckedBundleError>;
    fn resolve(module: &generated::Module) -> Result<generated::EntryToken, CheckedBundleError>;
    fn encode(args: Self::Args<'_>) -> generated::EncodedArgs;
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

/// A workflow draft is the only public object that accepts prepared call
/// nodes. Enqueue returns symbolic results that cannot be used by the
/// synchronous kernel API. `run` consumes the draft, admits all nodes, submits
/// once, and materializes only the selected final results.
pub struct Workflow {
    inner: seismic_runtime::api::kernel::WorkflowDraftAny,
}

/// Completion owner for a submitted workflow. It retains the native execution
/// and all admitted resources until the first typed resolution completes it;
/// later result groups resolve from the completed table without resubmission.
pub struct WorkflowCompletion {
    inner: seismic_runtime::api::kernel::WorkflowCompletionAny,
}

impl Workflow {
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

    pub fn run<E: Entry>(self, outputs: E::WorkflowResults) -> Result<E::Results, CallError> {
        let outputs = E::workflow_outputs(outputs);
        let decoded = seismic_runtime::api::kernel::run_workflow(self.inner, outputs)?;
        Ok(E::decode(decoded))
    }

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
    fn prepare(
        device: &Device,
        precision: PrecisionPolicy,
        bindings: seismic_lang::entry::ElementBindings,
    ) -> Result<Self, LoadError> {
        let module = E::module().map_err(LoadError::Bundle)?;
        let entry = E::resolve(module).map_err(LoadError::Bundle)?;
        seismic_runtime::api::kernel::prepare(
            module.checked(),
            entry.id(),
            bindings,
            device.inner(),
            precision,
        )
        .map(|inner| Self {
            inner: Arc::new(inner),
            marker: std::marker::PhantomData,
        })
        .map_err(|error| match error {
            seismic_runtime::api::kernel::PrepareError::Source(error) => {
                LoadError::Source(SourceLoadError::from_internal(error))
            }
            seismic_runtime::api::kernel::PrepareError::Preparation(error) => {
                LoadError::Preparation(error)
            }
        })
    }

    pub fn call(&self, args: E::Args<'_>) -> Result<E::Results, CallError> {
        let encoded = E::encode(args);
        let decoded = seismic_runtime::api::kernel::call(&self.inner, encoded)?;
        Ok(E::decode(decoded))
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
        DecodedResults, EncodedArgs, EncodedWorkflowArgs, PendingWorkflowResults, WorkflowResultRef,
    };

    /// Opaque checked module token used only by generated bindings. Consumers
    /// can name the type because Rust trait implementations must, but cannot
    /// inspect or construct the compiler artifact it owns.
    pub struct Module(seismic_lang::checked::CheckedModule);

    /// Opaque proof that a generated entry name resolved in its own checked
    /// module. Generated code can pass it back to Seismic, but consumers
    /// never observe or fabricate compiler entry identifiers.
    pub struct EntryToken(seismic_lang::ids::EntryId);

    impl EntryToken {
        pub(super) fn id(&self) -> seismic_lang::ids::EntryId {
            self.0
        }
    }

    impl Module {
        pub(crate) fn checked(&self) -> &seismic_lang::checked::CheckedModule {
            &self.0
        }
        #[doc(hidden)]
        pub fn entry_named(&self, name: &str) -> Option<EntryToken> {
            self.0.entry_named(name).map(EntryToken)
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
            self.inner.push_scalar(EncodedScalar::F32(value));
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
        pub fn index(&mut self, value: u64) {
            self.inner.push_scalar(EncodedScalar::Index(value));
        }
        pub fn range(&mut self, value: (u64, u64)) {
            self.inner.push_scalar(EncodedScalar::Range {
                start: value.0,
                end: value.1,
            });
        }
        pub fn finish(self) -> EncodedArgs {
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
                    .push_result_tensor_leading_slice(view.inner, view.start, view.end),
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
                    .push_result_tensor_leading_slice(view.inner, view.start, view.end),
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
            self.inner.push_scalar(EncodedScalar::F32(value));
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
        pub fn index(&mut self, value: u64) {
            self.inner.push_scalar(EncodedScalar::Index(value));
        }
        pub fn range(&mut self, value: (u64, u64)) {
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
        precision: PrecisionPolicy,
        elements: &[(&str, Element)],
    ) -> Result<Kernel<E>, LoadError> {
        let bindings = elements.iter().fold(
            seismic_lang::entry::ElementBindings::new(),
            |bindings, (name, element)| bindings.bind(name, element.id()),
        );
        Kernel::prepare(device, precision, bindings)
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
    scalar_result!(take_index, Index, u64, |value| value);

    pub fn take_range(results: &mut DecodedResults) -> (u64, u64) {
        match results.take_scalar() {
            ArgumentValue::Range { start, end } => (start, end),
            _ => panic!("generated result schema disagrees with prepared kernel"),
        }
    }

    pub fn module_from_bundle(
        cell: &'static OnceLock<Result<Module, CheckedBundleError>>,
        bytes: &'static [u8],
    ) -> Result<&'static Module, CheckedBundleError> {
        match cell.get_or_init(|| seismic_lang::bundle::decode_checked_bundle(bytes).map(Module)) {
            Ok(module) => Ok(module),
            Err(error) => Err(error.clone()),
        }
    }
}
