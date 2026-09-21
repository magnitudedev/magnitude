//! Public-runtime internals composed by the `seismic` crate. Backends form a
//! private closed sum; callers see devices, tensor descriptors and prepared
//! kernels, never compiler plans or raw buffers.

use seismic_compiler::errors::{ExecutionError, InvocationError};
use seismic_lang::registry::BackendName;
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(pub(crate) usize);

#[derive(Clone)]
pub struct DeviceInfo {
    pub id: DeviceId,
    pub backend: BackendName,
    pub name: String,
    pub memory_bytes: u64,
    pub(crate) descriptor: std::sync::Arc<crate::backends::Descriptor>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryUsage {
    pub charged: u64,
    pub limit: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryLimitError {
    pub limit: u64,
    pub charged: u64,
}

impl fmt::Display for MemoryLimitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "memory limit {} is below {} retained bytes",
            self.limit, self.charged
        )
    }
}
impl std::error::Error for MemoryLimitError {}

impl fmt::Debug for DeviceInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceInfo")
            .field("id", &self.id)
            .field("backend", &self.backend)
            .field("name", &self.name)
            .field("memory_bytes", &self.memory_bytes)
            .finish()
    }
}

impl PartialEq for DeviceInfo {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
            && self.backend == other.backend
            && self.name == other.name
            && self.memory_bytes == other.memory_bytes
    }
}
impl Eq for DeviceInfo {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CallError {
    Invocation(InvocationError),
    Execution(ExecutionError),
    Workflow(WorkflowError),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WorkflowError {
    Empty,
    CrossWorkflowResult,
    MissingProducerResult,
    HostBoundaryRequired,
    TensorView(TensorError),
}
impl fmt::Display for CallError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invocation(e) => write!(f, "{e}"),
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
            Self::Workflow(WorkflowError::TensorView(error)) => write!(f, "{error}"),
        }
    }
}
impl std::error::Error for CallError {}

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
    ReshapeStorage {
        current_bytes: u64,
        requested_bytes: u64,
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
            Self::SliceOutOfBounds { extent, start, end } => {
                write!(f, "slice {start}..{end} is outside leading extent {extent}")
            }
            Self::UnalignedPacketSlice { group, start, end } => write!(
                f,
                "packed slice {start}..{end} is not aligned to packet group {group}"
            ),
            Self::ReshapeStorage {
                current_bytes,
                requested_bytes,
            } => write!(
                f,
                "reshape requires {requested_bytes} bytes but the tensor view contains {current_bytes}"
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

pub mod catalog {
    use super::{DeviceId, DeviceInfo};
    use crate::api::device::DeviceInner;
    use crate::backends;
    use seismic_compiler::errors::TargetError;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, Weak};

    pub struct Catalog {
        pub(crate) infos: Vec<DeviceInfo>,
        opened: Mutex<HashMap<DeviceId, Weak<DeviceInner>>>,
    }
    impl Catalog {
        pub fn discover() -> Result<Self, TargetError> {
            Ok(Self {
                infos: backends::discover()?,
                opened: Mutex::new(HashMap::new()),
            })
        }
        pub fn devices(&self) -> &[DeviceInfo] {
            &self.infos
        }
        pub fn open(&self, id: DeviceId) -> Result<Arc<DeviceInner>, TargetError> {
            // Holding this lock through acquisition makes opening atomic: two
            // callers cannot create independent services or profiles for the
            // same catalog descriptor. A dropped device may be opened and
            // profiled again; a live one is always shared.
            let mut opened = self
                .opened
                .lock()
                .expect("device catalog open-cache lock poisoned while acquiring a private device");
            if let Some(device) = opened.get(&id).and_then(Weak::upgrade) {
                return Ok(device);
            }
            let device = backends::open(&self.infos, id)?;
            opened.insert(id, Arc::downgrade(&device));
            Ok(device)
        }
    }
}

pub mod device {
    use super::DeviceInfo;
    use crate::backends::DeviceKind;
    use crate::driver::Allocation;
    use seismic_compiler::errors::ExecutionError;
    use std::sync::Arc;

    pub struct DeviceInner {
        pub(crate) info: DeviceInfo,
        pub(crate) capabilities: Vec<String>,
        pub(crate) kind: DeviceKind,
    }
    impl DeviceInner {
        pub fn info(&self) -> &DeviceInfo {
            &self.info
        }
        pub fn capabilities(&self) -> &[String] {
            &self.capabilities
        }
        pub(crate) fn allocate(
            self: &Arc<Self>,
            bytes: u64,
            alignment: u64,
        ) -> Result<Arc<Allocation>, ExecutionError> {
            self.kind.allocate(bytes, alignment)
        }
        pub(crate) fn supports_representation(
            &self,
            representation: seismic_lang::ids::RepresentationId,
        ) -> bool {
            self.kind.supports_representation(representation)
        }
        #[doc(hidden)]
        pub fn memory_usage(&self) -> super::MemoryUsage {
            let usage = self.kind.memory_usage();
            super::MemoryUsage {
                charged: usage.charged,
                limit: usage.limit,
            }
        }
        #[doc(hidden)]
        pub fn set_memory_limit(&self, limit: Option<u64>) -> Result<(), super::MemoryLimitError> {
            self.kind.set_memory_limit(limit)
        }
    }
}

pub mod tensor {
    use super::{device::DeviceInner, TensorError};
    use crate::driver::{write_zeros, Allocation};
    use crate::layout;
    use seismic_compiler::errors::ExecutionError;
    use seismic_compiler::prepared::TensorDescriptor;
    use seismic_lang::ids::RepresentationId;
    use seismic_lang::registry::{representation_info, RepresentationKind};
    use std::sync::Arc;

    pub struct TensorInner {
        device: Arc<DeviceInner>,
        allocation: Arc<Allocation>,
        byte_offset: u64,
        byte_len: u64,
        representation: RepresentationId,
        extents: Vec<u64>,
        strides: Vec<u64>,
    }

    impl TensorInner {
        pub fn zeros(
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
            }
        }
        pub fn read_to_host(&self) -> Result<Vec<u8>, ExecutionError> {
            let _guard = self.allocation.acquire(false);
            let length = usize::try_from(self.byte_len).map_err(|_| {
                ExecutionError::AllocationFailed(
                    "tensor is too large for a host byte vector".to_owned(),
                )
            })?;
            let mut bytes = vec![0u8; length];
            self.allocation
                .storage()
                .read(self.byte_offset, &mut bytes)?;
            Ok(bytes)
        }
        pub fn write_from_host(&self, bytes: &[u8]) -> Result<(), TensorError> {
            let actual = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
            if actual != self.byte_len {
                return Err(TensorError::HostByteLength {
                    expected: self.byte_len,
                    actual,
                });
            }
            let _guard = self.allocation.acquire(true);
            self.allocation.storage().write(self.byte_offset, bytes)?;
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
            let Some(&leading) = self.extents.first() else {
                return Err(TensorError::SliceOutOfBounds {
                    extent: 0,
                    start,
                    end,
                });
            };
            if start > end || end > leading {
                return Err(TensorError::SliceOutOfBounds {
                    extent: leading,
                    start,
                    end,
                });
            }
            let mut extents = self.extents.clone();
            extents[0] = end - start;
            let (relative, byte_len) = match &representation_info(self.representation).kind {
                RepresentationKind::Dense(dtype) => {
                    let unit = u64::from(dtype.bytes());
                    let row = self.strides[0]
                        .checked_mul(unit)
                        .expect("canonical tensor row bytes overflowed");
                    (
                        start
                            .checked_mul(row)
                            .expect("validated leading slice offset overflowed"),
                        (end - start)
                            .checked_mul(row)
                            .expect("validated leading slice length overflowed"),
                    )
                }
                RepresentationKind::Packed(packet) if self.extents.len() == 1 => {
                    let group = u64::from(packet.group);
                    if start % group != 0 || (end != leading && end % group != 0) {
                        return Err(TensorError::UnalignedPacketSlice {
                            group: packet.group,
                            start,
                            end,
                        });
                    }
                    let first = start / group;
                    let last = end
                        .checked_add(group - 1)
                        .and_then(|v| v.checked_div(group))
                        .expect("validated packed slice packet bound overflowed");
                    let packet_bytes = u64::from(packet.packet_size);
                    (
                        first
                            .checked_mul(packet_bytes)
                            .expect("packed slice offset overflowed"),
                        (last - first)
                            .checked_mul(packet_bytes)
                            .expect("packed slice length overflowed"),
                    )
                }
                RepresentationKind::Packed(packet) => {
                    let row = self.strides[0]
                        .checked_mul(u64::from(packet.packet_size))
                        .expect("canonical packed row bytes overflowed");
                    (
                        start
                            .checked_mul(row)
                            .expect("validated leading slice offset overflowed"),
                        (end - start)
                            .checked_mul(row)
                            .expect("validated leading slice length overflowed"),
                    )
                }
                RepresentationKind::External(packet) if self.extents.len() == 1 => {
                    let group = u64::from(packet.logical_group);
                    if start % group != 0 || (end != leading && end % group != 0) {
                        return Err(TensorError::UnalignedPacketSlice {
                            group: packet.logical_group,
                            start,
                            end,
                        });
                    }
                    let first = start / group;
                    let last = end
                        .checked_add(group - 1)
                        .and_then(|value| value.checked_div(group))
                        .expect("validated external slice packet bound overflowed");
                    let packet_bytes = u64::from(packet.packet_size);
                    (
                        first
                            .checked_mul(packet_bytes)
                            .expect("external slice offset overflowed"),
                        (last - first)
                            .checked_mul(packet_bytes)
                            .expect("external slice length overflowed"),
                    )
                }
                RepresentationKind::External(packet) => {
                    let row = self.strides[0]
                        .checked_mul(u64::from(packet.packet_size))
                        .expect("canonical external row bytes overflowed");
                    (
                        start
                            .checked_mul(row)
                            .expect("validated leading slice offset overflowed"),
                        (end - start)
                            .checked_mul(row)
                            .expect("validated leading slice length overflowed"),
                    )
                }
            };
            let byte_offset = self
                .byte_offset
                .checked_add(relative)
                .expect("validated leading slice base overflowed");
            Ok(Self::new_view(
                self.device.clone(),
                self.allocation.clone(),
                byte_offset,
                byte_len,
                self.representation,
                extents,
                self.strides.clone(),
            ))
        }
        pub fn reshape(&self, extents: &[u64]) -> Result<Self, TensorError> {
            let layout = layout::canonical(self.representation, extents)?;
            if layout.byte_len != self.byte_len {
                return Err(TensorError::ReshapeStorage {
                    current_bytes: self.byte_len,
                    requested_bytes: layout.byte_len,
                });
            }
            Ok(Self::new_view(
                self.device.clone(),
                self.allocation.clone(),
                self.byte_offset,
                self.byte_len,
                self.representation,
                extents.to_vec(),
                layout.strides,
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
    use seismic_lang::precision::PrecisionPolicy;
    use std::collections::VecDeque;
    use std::sync::Arc;

    #[derive(Debug)]
    pub enum PrepareError {
        Source(SourceError),
        Preparation(PreparationError),
    }

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

    pub(crate) enum EncodedWorkflowArgument {
        Tensor(WorkflowTensorArgument),
        Scalar(EncodedScalar),
        ScalarResult(WorkflowResultRef),
    }

    pub(crate) enum WorkflowTensorArgument {
        External(Arc<TensorInner>),
        Result(WorkflowResultRef),
        ResultLeadingSlice {
            result: WorkflowResultRef,
            start: u64,
            end: u64,
        },
    }

    /// Generated workflow arguments retain symbolic producer edges until
    /// whole-workflow binding resolves them. They cannot be passed to the
    /// one-node call path.
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
        pub fn push_result_tensor_leading_slice(
            &mut self,
            result: WorkflowResultRef,
            start: u64,
            end: u64,
        ) {
            self.arguments.push(EncodedWorkflowArgument::Tensor(
                WorkflowTensorArgument::ResultLeadingSlice { result, start, end },
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
    pub enum EncodedScalar {
        F32(f32),
        F16(u16),
        BF16(u16),
        I32(i32),
        U32(u32),
        Bool(bool),
        Index(u64),
        Range { start: u64, end: u64 },
    }
    impl EncodedScalar {
        fn value(&self) -> ArgumentValue {
            match *self {
                Self::F32(value) => ArgumentValue::F32(value),
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
        pub(crate) fn into_tensors(self) -> Vec<Arc<TensorInner>> {
            self.arguments
                .into_iter()
                .filter_map(|argument| match argument {
                    EncodedArgument::Tensor(tensor) => Some(tensor),
                    EncodedArgument::Scalar(_) => None,
                })
                .collect()
        }
    }

    #[derive(Clone)]
    pub(crate) enum DecodedValue {
        Tensor(Arc<TensorInner>),
        Scalar(ArgumentValue),
    }
    pub struct DecodedResults {
        values: VecDeque<DecodedValue>,
    }
    impl DecodedResults {
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

    type PendingCompletion = Box<dyn FnOnce() -> Result<Vec<Vec<DecodedValue>>, super::CallError>>;

    enum CompletionState {
        Pending(PendingCompletion),
        Running,
        Complete(Result<Vec<Vec<DecodedValue>>, super::CallError>),
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
            completion: impl FnOnce() -> Result<Vec<Vec<DecodedValue>>, super::CallError> + 'static,
        ) -> Self {
            Self {
                workflow,
                state: std::sync::Mutex::new(CompletionState::Pending(Box::new(completion))),
                changed: std::sync::Condvar::new(),
            }
        }

        fn complete(&self) -> Result<Vec<Vec<DecodedValue>>, super::CallError> {
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
            let values = self.complete()?;
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
                    let _ = self.complete();
                }));
            }
        }
    }

    pub struct PreparedAny {
        pub(crate) inner: crate::backends::PreparedKind,
    }

    pub struct WorkflowDraftAny {
        inner: crate::backends::WorkflowDraftKind,
    }

    pub fn workflow(device: &Arc<DeviceInner>) -> WorkflowDraftAny {
        WorkflowDraftAny {
            inner: device.kind.workflow(device),
        }
    }

    pub fn enqueue(
        workflow: &mut WorkflowDraftAny,
        kernel: &Arc<PreparedAny>,
        args: EncodedWorkflowArgs,
    ) -> Result<PendingWorkflowResults, super::WorkflowError> {
        workflow.inner.enqueue(&kernel.inner, args)
    }

    pub fn run_workflow(
        workflow: WorkflowDraftAny,
        outputs: Vec<WorkflowResultRef>,
    ) -> Result<DecodedResults, CallError> {
        workflow.inner.run(outputs)
    }

    pub fn submit_workflow(workflow: WorkflowDraftAny) -> Result<WorkflowCompletionAny, CallError> {
        workflow.inner.submit()
    }
    pub fn prepare(
        module: &CheckedModule,
        entry: EntryId,
        bindings: ElementBindings,
        device: &Arc<DeviceInner>,
        precision: PrecisionPolicy,
    ) -> Result<PreparedAny, PrepareError> {
        device
            .kind
            .prepare(module, entry, bindings, device, precision)
            .map(|inner| PreparedAny { inner })
    }
    pub fn call(kernel: &Arc<PreparedAny>, args: EncodedArgs) -> Result<DecodedResults, CallError> {
        kernel.inner.call(args)
    }

    #[cfg(test)]
    mod completion_tests {
        use super::*;
        use std::sync::atomic::{AtomicUsize, Ordering};

        #[test]
        fn completion_runs_once_and_caches() {
            let runs = Arc::new(AtomicUsize::new(0));
            let observed = runs.clone();
            let completion = WorkflowCompletionAny::pending(7, move || {
                observed.fetch_add(1, Ordering::SeqCst);
                Ok(Vec::new())
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
                Ok(Vec::new())
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
