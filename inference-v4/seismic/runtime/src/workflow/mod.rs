//! Descriptor-first workflow planning and its native binder.
//!
//! The graph planner resolves the complete graph before resource admission. It
//! owns no allocator, device, queue, native handle, or resource-state callback;
//! the native binder supplies the bound execution and its derived description.

pub(crate) mod native;

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use seismic_compiler::prepared::DeviceIdentity;
use seismic_lang::ids::RepresentationId;

static NEXT_WORKFLOW: AtomicU64 = AtomicU64::new(1);

fn fresh_workflow_id() -> WorkflowId {
    let identity = NEXT_WORKFLOW
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            current.checked_add(1)
        })
        .unwrap_or_else(|_| panic!("workflow identity space exhausted"));
    WorkflowId(identity)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct WorkflowId(u64);

impl WorkflowId {
    pub(crate) fn fresh() -> Self {
        fresh_workflow_id()
    }
    pub(crate) fn raw(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct OutputRef {
    workflow: WorkflowId,
    node: u32,
    output: u32,
}

impl OutputRef {
    pub(crate) fn from_raw(workflow: u64, node: u32, output: u32) -> Self {
        Self {
            workflow: WorkflowId(workflow),
            node,
            output,
        }
    }
    pub(crate) fn workflow(self) -> WorkflowId {
        self.workflow
    }
    pub(crate) fn node(self) -> u32 {
        self.node
    }
    pub(crate) fn output(self) -> u32 {
        self.output
    }
}

/// Symbolic storage identity, never a native allocation handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum ResourceId {
    External(u64),
    Produced {
        workflow: WorkflowId,
        node: u32,
        output: u32,
    },
    NodeLocal {
        workflow: WorkflowId,
        node: u32,
        slot: u32,
    },
    Persistent {
        owner: u64,
        variant: u32,
        slot: u32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ByteRange {
    pub(crate) offset: u64,
    pub(crate) len: u64,
}

impl ByteRange {
    fn end(self) -> Option<u64> {
        self.offset.checked_add(self.len)
    }
    fn contains(self, relative: Self) -> bool {
        relative.end().is_some_and(|end| end <= self.len)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TensorDescriptor {
    pub(crate) device: DeviceIdentity,
    pub(crate) resource: ResourceId,
    pub(crate) representation: RepresentationId,
    pub(crate) extents: Vec<u64>,
    /// Representation-defined strides, matching the compiler call descriptor.
    pub(crate) strides: Vec<u64>,
    pub(crate) range: ByteRange,
}

/// Canonical scalar identity used by workflow binding.
///
/// Floating-point values are stored as their wire bits so descriptor equality
/// is reflexive (including NaNs) and distinguishes values such as `0.0` and
/// `-0.0`. Public APIs still accept ordinary `f32` values and convert at the
/// boundary.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ScalarValue {
    F32Bits(u32),
    F16(u16),
    BF16(u16),
    I32(i32),
    U32(u32),
    Bool(bool),
    Index(seismic_lang::expr::BigUint),
    Range { start: seismic_lang::expr::BigUint, end: seismic_lang::expr::BigUint },
}

impl ScalarValue {
    pub fn from_f32(value: f32) -> Self {
        Self::F32Bits(value.to_bits())
    }

    #[cfg(test)]
    pub(crate) fn as_f32(self) -> Option<f32> {
        match self {
            Self::F32Bits(bits) => Some(f32::from_bits(bits)),
            _ => None,
        }
    }
}

impl From<f32> for ScalarValue {
    fn from(value: f32) -> Self {
        Self::from_f32(value)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ScalarDescriptor {
    HostReady(ScalarValue),
    DeviceProduced,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ValueDescriptor {
    Tensor(TensorDescriptor),
    PendingTensor {
        device: DeviceIdentity,
        representation: RepresentationId,
        rank: usize,
    },
    Scalar(ScalarDescriptor),
}

/// A checked view relative to a producer tensor's visible byte range.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TensorView {
    pub(crate) relative_range: ByteRange,
    pub(crate) extents: Vec<u64>,
    pub(crate) strides: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ArgumentBinding {
    External(ValueDescriptor),
    Result(OutputRef),
    View {
        result: OutputRef,
        view: TensorView,
    },
    LeadingSlice {
        result: OutputRef,
        start: u64,
        end: u64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AccessMode {
    Read,
    Write,
    ReadWrite,
}

impl AccessMode {
    fn merge(self, other: Self) -> Self {
        match (self, other) {
            (Self::Read, Self::Read) => Self::Read,
            (Self::Write, Self::Write) => Self::Write,
            _ => Self::ReadWrite,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum TensorStorage {
    Fresh { bytes: u64, alignment: u64 },
    Alias { argument: u32, range: ByteRange },
    Private { allocation: u32, range: ByteRange },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum OutputDescription {
    ProducedTensor {
        device: DeviceIdentity,
        representation: RepresentationId,
        rank: usize,
    },
    Tensor {
        device: DeviceIdentity,
        representation: RepresentationId,
        extents: Vec<u64>,
        strides: Vec<u64>,
        storage: TensorStorage,
    },
    Scalar(ScalarDescriptor),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AllocationKind {
    Temporary,
    Persistent { owner: u64, variant: u32, slot: u32 },
}

/// Invocation capacity is known at that node's binding. Reached private backing
/// is acquired by the selected schedule and may become an actual published
/// result after successful producer completion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Acquisition {
    Invocation(u64),
    ReachedPrivate,
}
impl Acquisition {
    pub(crate) fn initial_bytes(self) -> Option<u64> {
        match self { Self::Invocation(bytes) => Some(bytes), Self::ReachedPrivate => None }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AllocationDescription {
    pub(crate) kind: AllocationKind,
    pub(crate) acquisition: Acquisition,
    pub(crate) alignment: u64,
}

/// One bound execution with the arguments and descriptions used to derive it.
/// Constructed only inside workflow binding, then consumed by graph insertion.
pub(crate) struct BoundInvocation<S> {
    arguments: Vec<ValueDescriptor>,
    selection: S,
    argument_access: Vec<Option<AccessMode>>,
    outputs: Vec<OutputDescription>,
    private_allocations: Vec<AllocationDescription>,
}

/// The only capability workflow planning needs from a prepared kernel.
pub(crate) trait PreparedPolicy {
    type Selection;
    type Error;
    fn evaluate(
        &self,
        arguments: Vec<ValueDescriptor>,
    ) -> Result<BoundInvocation<Self::Selection>, Self::Error>;
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum PlanError<E> {
    Empty,
    InvalidReference(OutputRef),
    HostBoundaryRequired(OutputRef),
    ViewOfScalar(OutputRef),
    InvalidView(OutputRef),
    InvalidPolicyDescription(&'static str),
    Policy(E),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ResourceAccess {
    pub(crate) resource: ResourceId,
    pub(crate) range: ByteRange,
    pub(crate) mode: AccessMode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Lifetime {
    pub(crate) first: u32,
    /// Exclusive node boundary through which the resource remains available.
    pub(crate) end: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ResourceRequirement {
    pub(crate) resource: ResourceId,
    pub(crate) acquisition: Acquisition,
    pub(crate) alignment: u64,
    pub(crate) lifetime: Lifetime,
}

#[derive(Clone)]
pub(crate) struct BoundNode<S> {
    selection: S,
    /// Unique producer ordinals, ordered by first argument occurrence.
    dependencies: Vec<u32>,
    arguments: Vec<ValueDescriptor>,
    outputs: Vec<ValueDescriptor>,
    accesses: Vec<ResourceAccess>,
    private_resources: Vec<ResourceRequirement>,
}

pub(crate) struct BoundWorkflow<S> {
    identity: WorkflowId,
    nodes: Vec<BoundNode<S>>,
    lifetimes: BTreeMap<ResourceId, Lifetime>,
    requirements: BTreeMap<ResourceId, ResourceRequirement>,
}

impl<S> BoundWorkflow<S> {
    pub(crate) fn nodes(&self) -> &[BoundNode<S>] {
        &self.nodes
    }
    pub(crate) fn lifetimes(&self) -> &BTreeMap<ResourceId, Lifetime> {
        &self.lifetimes
    }
    pub(crate) fn requirements(&self) -> &BTreeMap<ResourceId, ResourceRequirement> {
        &self.requirements
    }
    pub(crate) fn into_parts(
        self,
    ) -> (
        WorkflowId,
        Vec<BoundNode<S>>,
        BTreeMap<ResourceId, Lifetime>,
        BTreeMap<ResourceId, ResourceRequirement>,
    ) {
        (self.identity, self.nodes, self.lifetimes, self.requirements)
    }
}

pub(crate) struct WorkflowPlanDraft<S, E> {
    error: std::marker::PhantomData<fn() -> E>,
    identity: WorkflowId,
    nodes: Vec<BoundNode<S>>,
    outputs: Vec<Vec<ValueDescriptor>>,
    lifetimes: BTreeMap<ResourceId, Lifetime>,
    requirements: BTreeMap<ResourceId, ResourceRequirement>,
    access_history: BTreeMap<ResourceId, Vec<HistoricalAccess>>,
}

#[derive(Clone, Copy)]
struct HistoricalAccess {
    node: u32,
    range: ByteRange,
    mode: AccessMode,
}

impl<S, E> WorkflowPlanDraft<S, E> {
    pub(crate) fn new() -> Self {
        Self::with_identity(WorkflowId::fresh())
    }

    pub(crate) fn with_identity(identity: WorkflowId) -> Self {
        Self {
            error: std::marker::PhantomData,
            identity,
            nodes: Vec::new(),
            outputs: Vec::new(),
            lifetimes: BTreeMap::new(),
            requirements: BTreeMap::new(),
            access_history: BTreeMap::new(),
        }
    }
    pub(crate) fn output(&self, node: u32, output: u32) -> OutputRef {
        OutputRef {
            workflow: self.identity,
            node,
            output,
        }
    }

    pub(crate) fn push<P: PreparedPolicy<Selection = S, Error = E>>(
        &mut self,
        policy: P,
        bindings: Vec<ArgumentBinding>,
    ) -> Result<Vec<OutputRef>, PlanError<E>> {
        self.bind_and_insert(bindings, |arguments| policy.evaluate(arguments))
    }

    pub(crate) fn bind_and_insert(
        &mut self,
        bindings: Vec<ArgumentBinding>,
        bind: impl FnOnce(Vec<ValueDescriptor>) -> Result<BoundInvocation<S>, E>,
    ) -> Result<Vec<OutputRef>, PlanError<E>> {
        let previous_lifetimes = self.lifetimes.clone();
        let previous_requirements = self.requirements.clone();
        match self.bind_and_insert_inner(bindings, bind) {
            Ok(outputs) => Ok(outputs),
            Err(error) => {
                self.lifetimes = previous_lifetimes;
                self.requirements = previous_requirements;
                Err(error)
            }
        }
    }

    pub(crate) fn bindings_ready(&self, bindings: &[ArgumentBinding]) -> bool {
        bindings.iter().all(|binding| match binding {
            ArgumentBinding::External(_) => true,
            ArgumentBinding::Result(result) | ArgumentBinding::View { result, .. }
            | ArgumentBinding::LeadingSlice { result, .. } => matches!(
                self.resolve_result(*result),
                Ok(ValueDescriptor::Tensor(_)) | Ok(ValueDescriptor::Scalar(ScalarDescriptor::HostReady(_)))
            ),
        })
    }

    /// Pure ready-invocation binding. It does not insert a node or reserve its
    /// storage; the original graph ordinal is preserved until execution reaches
    /// the node. All references still resolve through this planner.
    pub(crate) fn prepare_at(
        &mut self,
        node: u32,
        bindings: &[ArgumentBinding],
        bind: impl FnOnce(Vec<ValueDescriptor>) -> Result<BoundInvocation<S>, E>,
    ) -> Result<BoundInvocation<S>, PlanError<E>> {
        let previous = self.lifetimes.clone();
        let result = bindings.iter().cloned().map(|binding| self.resolve_binding(binding, node))
            .collect::<Result<Vec<_>, _>>().and_then(|arguments| bind(arguments).map_err(PlanError::Policy));
        // Insertion owns graph lifetime changes. Preparing an invocation only
        // proves that its present argument descriptors can select/bind it.
        self.lifetimes = previous;
        result
    }

    fn bind_and_insert_inner(
        &mut self,
        bindings: Vec<ArgumentBinding>,
        bind: impl FnOnce(Vec<ValueDescriptor>) -> Result<BoundInvocation<S>, E>,
    ) -> Result<Vec<OutputRef>, PlanError<E>> {
        let node = u32::try_from(self.nodes.len())
            .map_err(|_| PlanError::InvalidPolicyDescription("node ordinal overflow"))?;
        let mut dependencies = Vec::new();
        for binding in &bindings {
            let producer = match binding {
                ArgumentBinding::Result(result)
                | ArgumentBinding::View { result, .. }
                | ArgumentBinding::LeadingSlice { result, .. } => Some(result.node),
                ArgumentBinding::External(_) => None,
            };
            if let Some(producer) = producer {
                if !dependencies.contains(&producer) {
                    dependencies.push(producer);
                }
            }
        }
        let mut arguments = Vec::with_capacity(bindings.len());
        for binding in bindings {
            arguments.push(self.resolve_binding(binding, node)?);
        }
        let bound = bind(arguments).map_err(PlanError::Policy)?;
        self.insert_bound(node, dependencies, bound)
    }

    fn insert_bound(
        &mut self,
        node: u32,
        mut dependencies: Vec<u32>,
        evaluated: BoundInvocation<S>,
    ) -> Result<Vec<OutputRef>, PlanError<E>> {
        let arguments = evaluated.arguments;
        if evaluated.argument_access.len() != arguments.len() {
            return Err(PlanError::InvalidPolicyDescription(
                "argument access count differs from argument count",
            ));
        }

        let mut accesses = Vec::new();
        for (argument, mode) in arguments.iter().zip(&evaluated.argument_access) {
            match (argument, mode) {
                (ValueDescriptor::Tensor(tensor), Some(mode)) => {
                    coalesce_identical_access(
                        &mut accesses,
                        ResourceAccess {
                            resource: tensor.resource,
                            range: tensor.range,
                            mode: *mode,
                        },
                    );
                    extend_lifetime(&mut self.lifetimes, tensor.resource, node)?;
                }
                (ValueDescriptor::Tensor(_), None) | (ValueDescriptor::Scalar(_), None) => {}
                (ValueDescriptor::Scalar(_), Some(_)) => {
                    return Err(PlanError::InvalidPolicyDescription(
                        "scalar arguments cannot declare storage access",
                    ));
                }
                (ValueDescriptor::PendingTensor { .. }, _) => {
                    return Err(PlanError::InvalidPolicyDescription("pending output cannot bind an invocation"));
                }
            }
        }

        let mut private_resources = Vec::with_capacity(evaluated.private_allocations.len());
        for (index, allocation) in evaluated.private_allocations.into_iter().enumerate() {
            validate_alignment(allocation.alignment)?;
            let index = u32::try_from(index).map_err(|_| {
                PlanError::InvalidPolicyDescription("private allocation ordinal overflow")
            })?;
            let resource = match allocation.kind {
                AllocationKind::Temporary => ResourceId::NodeLocal {
                    workflow: self.identity,
                    node,
                    slot: index,
                },
                AllocationKind::Persistent {
                    owner,
                    variant,
                    slot,
                } => ResourceId::Persistent {
                    owner,
                    variant,
                    slot,
                },
            };
            let lifetime = Lifetime {
                first: node,
                end: next_boundary(node)?,
            };
            private_resources.push(ResourceRequirement {
                resource,
                acquisition: allocation.acquisition,
                alignment: allocation.alignment,
                lifetime,
            });
            if let Some(bytes) = allocation.acquisition.initial_bytes() {
                coalesce_identical_access(&mut accesses, ResourceAccess {
                    resource, range: ByteRange { offset: 0, len: bytes }, mode: AccessMode::ReadWrite,
                });
            } else if !matches!(allocation.kind, AllocationKind::Temporary) {
                return Err(PlanError::InvalidPolicyDescription("reached backing must be invocation-private"));
            }
            // An unpublished node-local resource has no cross-node access edge.
            // Its allocation identity and lifetime still belong to this graph.
            self.lifetimes
                .entry(resource)
                .and_modify(|known| known.end = known.end.max(lifetime.end))
                .or_insert(lifetime);
            self.register_requirement(ResourceRequirement {
                resource,
                acquisition: allocation.acquisition,
                alignment: allocation.alignment,
                lifetime,
            })?;
        }

        let mut outputs = Vec::with_capacity(evaluated.outputs.len());
        let mut output_refs = Vec::with_capacity(evaluated.outputs.len());
        for (index, description) in evaluated.outputs.into_iter().enumerate() {
            let output = u32::try_from(index)
                .map_err(|_| PlanError::InvalidPolicyDescription("output ordinal overflow"))?;
            let reference = self.output(node, output);
            let value = self.materialize_output(
                node,
                output,
                description,
                &arguments,
                &private_resources,
                &mut accesses,
            )?;
            outputs.push(value);
            output_refs.push(reference);
        }

        // Admission consumes these closed edges; it never rediscovers ordering
        // from live resource state. Any overlapping pair with at least one
        // writer orders the current node after the prior accessor.
        for access in &accesses {
            if let Some(history) = self.access_history.get(&access.resource) {
                for prior in history {
                    if ranges_overlap(prior.range, access.range)
                        && access_modes_conflict(prior.mode, access.mode)
                        && !dependencies.contains(&prior.node)
                    {
                        dependencies.push(prior.node);
                    }
                }
            }
        }
        dependencies.sort_unstable();
        for access in &accesses {
            self.access_history
                .entry(access.resource)
                .or_default()
                .push(HistoricalAccess {
                    node,
                    range: access.range,
                    mode: access.mode,
                });
        }

        self.outputs.push(outputs.clone());
        self.nodes.push(BoundNode {
            selection: evaluated.selection,
            dependencies,
            arguments,
            outputs,
            accesses,
            private_resources,
        });
        Ok(output_refs)
    }

    fn materialize_output(
        &mut self,
        node: u32,
        output: u32,
        description: OutputDescription,
        arguments: &[ValueDescriptor],
        private_resources: &[ResourceRequirement],
        accesses: &mut Vec<ResourceAccess>,
    ) -> Result<ValueDescriptor, PlanError<E>> {
        if let OutputDescription::ProducedTensor { device, representation, rank } = description {
            return Ok(ValueDescriptor::PendingTensor { device, representation, rank });
        }
        let OutputDescription::Tensor {
            device,
            representation,
            extents,
            strides,
            storage,
        } = description
        else {
            let OutputDescription::Scalar(value) = description else {
                unreachable!()
            };
            return Ok(ValueDescriptor::Scalar(value));
        };
        if extents.len() != strides.len() {
            return Err(PlanError::InvalidPolicyDescription(
                "tensor output rank differs from stride count",
            ));
        }
        let (resource, range) = match storage {
            TensorStorage::Fresh { bytes, alignment } => {
                validate_alignment(alignment)?;
                let resource = ResourceId::Produced {
                    workflow: self.identity,
                    node,
                    output,
                };
                let range = ByteRange {
                    offset: 0,
                    len: bytes,
                };
                coalesce_identical_access(
                    accesses,
                    ResourceAccess {
                        resource,
                        range,
                        mode: AccessMode::Write,
                    },
                );
                let lifetime = Lifetime {
                    first: node,
                    end: next_boundary(node)?,
                };
                self.lifetimes.insert(resource, lifetime);
                self.register_requirement(ResourceRequirement {
                    resource,
                    acquisition: Acquisition::Invocation(bytes),
                    alignment,
                    lifetime,
                })?;
                (resource, range)
            }
            TensorStorage::Alias { argument, range } => {
                let Some(ValueDescriptor::Tensor(source)) = arguments.get(argument as usize) else {
                    return Err(PlanError::InvalidPolicyDescription(
                        "tensor output aliases a non-tensor argument",
                    ));
                };
                if !source.range.contains(range) {
                    return Err(PlanError::InvalidPolicyDescription(
                        "tensor output alias exceeds its argument",
                    ));
                }
                let offset = source.range.offset.checked_add(range.offset).ok_or(
                    PlanError::InvalidPolicyDescription("tensor output alias range overflow"),
                )?;
                let absolute = ByteRange {
                    offset,
                    len: range.len,
                };
                coalesce_identical_access(
                    accesses,
                    ResourceAccess {
                        resource: source.resource,
                        range: absolute,
                        mode: AccessMode::Write,
                    },
                );
                extend_lifetime(&mut self.lifetimes, source.resource, node)?;
                (source.resource, absolute)
            }
            TensorStorage::Private { allocation, range } => {
                let Some(source) = private_resources.get(allocation as usize) else {
                    return Err(PlanError::InvalidPolicyDescription(
                        "tensor output references an unknown private allocation",
                    ));
                };
                if !(ByteRange {
                    offset: 0,
                    len: source.acquisition.initial_bytes().ok_or(PlanError::InvalidPolicyDescription("reached private backing cannot be published"))?,
                })
                .contains(range)
                {
                    return Err(PlanError::InvalidPolicyDescription(
                        "tensor output exceeds its private allocation",
                    ));
                }
                coalesce_identical_access(
                    accesses,
                    ResourceAccess {
                        resource: source.resource,
                        range,
                        mode: AccessMode::Write,
                    },
                );
                extend_lifetime(&mut self.lifetimes, source.resource, node)?;
                (source.resource, range)
            }
        };
        let descriptor = TensorDescriptor {
            device,
            resource,
            representation,
            extents,
            strides,
            range,
        };
        validate_tensor(&descriptor)?;
        Ok(ValueDescriptor::Tensor(descriptor))
    }

    fn resolve_binding(
        &mut self,
        binding: ArgumentBinding,
        consumer: u32,
    ) -> Result<ValueDescriptor, PlanError<E>> {
        match binding {
            ArgumentBinding::External(value) => {
                validate_value(&value)?;
                Ok(value)
            }
            ArgumentBinding::Result(reference) => {
                let value = self.resolve_result(reference)?.clone();
                if matches!(
                    value,
                    ValueDescriptor::Scalar(ScalarDescriptor::DeviceProduced) | ValueDescriptor::PendingTensor { .. }
                ) {
                    return Err(PlanError::HostBoundaryRequired(reference));
                }
                if let ValueDescriptor::Tensor(tensor) = &value {
                    extend_lifetime(&mut self.lifetimes, tensor.resource, consumer)?;
                }
                Ok(value)
            }
            ArgumentBinding::View { result, view } => {
                let source = self.resolve_result(result)?.clone();
                let ValueDescriptor::Tensor(source) = source else {
                    return Err(PlanError::ViewOfScalar(result));
                };
                if view.extents.len() != view.strides.len()
                    || !source.range.contains(view.relative_range)
                {
                    return Err(PlanError::InvalidView(result));
                }
                let offset = source
                    .range
                    .offset
                    .checked_add(view.relative_range.offset)
                    .ok_or(PlanError::InvalidView(result))?;
                extend_lifetime(&mut self.lifetimes, source.resource, consumer)?;
                let descriptor = TensorDescriptor {
                    device: source.device,
                    resource: source.resource,
                    representation: source.representation,
                    extents: view.extents,
                    strides: view.strides,
                    range: ByteRange {
                        offset,
                        len: view.relative_range.len,
                    },
                };
                validate_tensor(&descriptor)?;
                Ok(ValueDescriptor::Tensor(descriptor))
            }
            ArgumentBinding::LeadingSlice { result, start, end } => {
                let source = self.resolve_result(result)?.clone();
                let ValueDescriptor::Tensor(source) = source else {
                    return Err(PlanError::ViewOfScalar(result));
                };
                let Some(view) = crate::layout::leading_slice(
                    source.representation,
                    &source.extents,
                    &source.strides,
                    start,
                    end,
                ) else {
                    return Err(PlanError::InvalidView(result));
                };
                let offset = source
                    .range
                    .offset
                    .checked_add(view.relative_offset)
                    .ok_or(PlanError::InvalidView(result))?;
                extend_lifetime(&mut self.lifetimes, source.resource, consumer)?;
                let descriptor = TensorDescriptor {
                    device: source.device,
                    resource: source.resource,
                    representation: source.representation,
                    extents: view.extents,
                    strides: view.strides,
                    range: ByteRange {
                        offset,
                        len: view.byte_len,
                    },
                };
                validate_tensor(&descriptor)?;
                Ok(ValueDescriptor::Tensor(descriptor))
            }
        }
    }

    fn resolve_result(&self, reference: OutputRef) -> Result<&ValueDescriptor, PlanError<E>> {
        if reference.workflow != self.identity {
            return Err(PlanError::InvalidReference(reference));
        }
        self.outputs
            .get(reference.node as usize)
            .and_then(|outputs| outputs.get(reference.output as usize))
            .ok_or(PlanError::InvalidReference(reference))
    }

    /// Installs an execution-produced descriptor only after its producer has
    /// completed successfully. The plan retains the same result identity;
    /// subsequent binding sees its actual backing and geometry.
    pub(crate) fn publish_tensor(
        &mut self,
        reference: OutputRef,
        tensor: TensorDescriptor,
    ) -> Result<(), PlanError<E>> {
        let expected = self.resolve_result(reference)?;
        let ValueDescriptor::PendingTensor { device, representation, rank } = expected else {
            return Err(PlanError::InvalidPolicyDescription("tensor result was already published or has another type"));
        };
        if *device != tensor.device || *representation != tensor.representation || *rank != tensor.extents.len() {
            return Err(PlanError::InvalidPolicyDescription("published tensor differs from its result schema"));
        }
        validate_tensor(&tensor)?;
        let lifetime = self.lifetimes.get_mut(&tensor.resource).ok_or(
            PlanError::InvalidPolicyDescription("published backing is not owned by the workflow"),
        )?;
        lifetime.end = lifetime.end.max(next_boundary(reference.node)?);
        let value = ValueDescriptor::Tensor(tensor);
        self.outputs[reference.node as usize][reference.output as usize] = value.clone();
        self.nodes[reference.node as usize].outputs[reference.output as usize] = value;
        Ok(())
    }

    pub(crate) fn bound_node(&self, node: u32) -> Option<&BoundNode<S>> {
        self.nodes.get(node as usize)
    }

    fn register_requirement(
        &mut self,
        requirement: ResourceRequirement,
    ) -> Result<(), PlanError<E>> {
        match self.requirements.get_mut(&requirement.resource) {
            None => {
                self.requirements.insert(requirement.resource, requirement);
            }
            Some(existing) => {
                if !matches!(requirement.resource, ResourceId::Persistent { .. }) {
                    return Err(PlanError::InvalidPolicyDescription(
                        "a non-persistent resource identity was declared more than once",
                    ));
                }
                // Repeated persistent use can legitimately require a larger
                // capacity at a later invocation. All alignments are powers of
                // two, so their maximum satisfies every declaration.
                existing.acquisition = Acquisition::Invocation(
                    existing.acquisition.initial_bytes().expect("persistent backing is initially acquired")
                        .max(requirement.acquisition.initial_bytes().expect("persistent backing is initially acquired")));
                existing.alignment = existing.alignment.max(requirement.alignment);
                existing.lifetime.first = existing.lifetime.first.min(requirement.lifetime.first);
                existing.lifetime.end = existing.lifetime.end.max(requirement.lifetime.end);
            }
        }
        Ok(())
    }

    pub(crate) fn close(mut self) -> Result<BoundWorkflow<S>, PlanError<E>> {
        if self.nodes.is_empty() {
            return Err(PlanError::Empty);
        }
        // Public workflow completion may resolve any output after submission.
        let boundary = u32::try_from(self.nodes.len())
            .map_err(|_| PlanError::InvalidPolicyDescription("node ordinal overflow"))?;
        for outputs in &self.outputs {
            for output in outputs {
                if let ValueDescriptor::Tensor(tensor) = output {
                    self.lifetimes
                        .entry(tensor.resource)
                        .and_modify(|lifetime| lifetime.end = boundary.max(lifetime.end));
                }
            }
        }
        for (resource, requirement) in &mut self.requirements {
            if let Some(lifetime) = self.lifetimes.get(resource) {
                requirement.lifetime = *lifetime;
            }
        }
        // Node-local entries retain the allocation-slot ordering needed by
        // admission, but once the graph is closed they must expose the same
        // aggregate capacity/alignment/lifetime as the authoritative map.
        // Otherwise a `BoundWorkflow` contains two contradictory resource
        // contracts for the same symbolic identity.
        for node in &mut self.nodes {
            for local in &mut node.private_resources {
                if let Some(aggregate) = self.requirements.get(&local.resource) {
                    *local = aggregate.clone();
                }
            }
        }
        Ok(BoundWorkflow {
            identity: self.identity,
            nodes: self.nodes,
            lifetimes: self.lifetimes,
            requirements: self.requirements,
        })
    }
}

fn validate_alignment<E>(alignment: u64) -> Result<(), PlanError<E>> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(PlanError::InvalidPolicyDescription(
            "allocation alignment must be a nonzero power of two",
        ));
    }
    Ok(())
}

fn validate_value<E>(value: &ValueDescriptor) -> Result<(), PlanError<E>> {
    if matches!(value, ValueDescriptor::PendingTensor { .. }) {
        return Err(PlanError::InvalidPolicyDescription("unpublished tensor cannot be an external value"));
    }
    if let ValueDescriptor::Tensor(tensor) = value {
        validate_tensor(tensor)?;
    }
    Ok(())
}

fn validate_tensor<E>(tensor: &TensorDescriptor) -> Result<(), PlanError<E>> {
    if !crate::layout::validates_view(
        tensor.representation,
        &tensor.extents,
        &tensor.strides,
        tensor.range.offset,
        tensor.range.len,
    ) {
        return Err(PlanError::InvalidPolicyDescription(
            "tensor descriptor geometry disagrees with its representation footprint",
        ));
    }
    Ok(())
}

fn next_boundary<E>(node: u32) -> Result<u32, PlanError<E>> {
    node.checked_add(1)
        .ok_or(PlanError::InvalidPolicyDescription(
            "node boundary overflow",
        ))
}

fn extend_lifetime<E>(
    lifetimes: &mut BTreeMap<ResourceId, Lifetime>,
    resource: ResourceId,
    node: u32,
) -> Result<(), PlanError<E>> {
    let end = next_boundary(node)?;
    lifetimes
        .entry(resource)
        .and_modify(|lifetime| lifetime.end = lifetime.end.max(end))
        .or_insert(Lifetime { first: node, end });
    Ok(())
}

/// Coalesce identical intervals only. Overlapping, non-identical intervals
/// remain explicit for admission's interval hazard analysis.
fn coalesce_identical_access(accesses: &mut Vec<ResourceAccess>, next: ResourceAccess) {
    if let Some(existing) = accesses
        .iter_mut()
        .find(|access| access.resource == next.resource && access.range == next.range)
    {
        existing.mode = existing.mode.merge(next.mode);
    } else {
        accesses.push(next);
    }
}

fn ranges_overlap(first: ByteRange, second: ByteRange) -> bool {
    let first_end = first
        .end()
        .expect("bound workflow contains only validated byte ranges");
    let second_end = second
        .end()
        .expect("bound workflow contains only validated byte ranges");
    first.offset < second_end && second.offset < first_end
}

fn access_modes_conflict(first: AccessMode, second: AccessMode) -> bool {
    !matches!((first, second), (AccessMode::Read, AccessMode::Read))
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic_lang::registry;
    use seismic_lang::types::DType;
    use std::cell::Cell;
    use std::rc::Rc;

    #[derive(Clone)]
    struct Policy {
        selection: u32,
        outputs: Vec<OutputDescription>,
        access: Vec<Option<AccessMode>>,
        allocations: Vec<AllocationDescription>,
        evaluations: Rc<Cell<usize>>,
    }

    impl PreparedPolicy for Policy {
        type Selection = u32;
        type Error = ();

        fn evaluate(
            &self,
            arguments: Vec<ValueDescriptor>,
        ) -> Result<BoundInvocation<Self::Selection>, Self::Error> {
            self.evaluations.set(self.evaluations.get() + 1);
            Ok(BoundInvocation {
                arguments,
                selection: self.selection,
                argument_access: self.access.clone(),
                outputs: self.outputs.clone(),
                private_allocations: self.allocations.clone(),
            })
        }
    }

    fn fresh(bytes: u64) -> OutputDescription {
        assert_eq!(bytes % 4, 0);
        OutputDescription::Tensor {
            device: DeviceIdentity(1),
            representation: representation(),
            extents: vec![bytes / 4],
            strides: vec![1],
            storage: TensorStorage::Fresh {
                bytes,
                alignment: 8,
            },
        }
    }

    fn representation() -> RepresentationId {
        registry::dense(DType::U32)
    }

    fn policy(selection: u32, inputs: usize, outputs: Vec<OutputDescription>) -> Policy {
        Policy {
            selection,
            outputs,
            access: vec![Some(AccessMode::Read); inputs],
            allocations: Vec::new(),
            evaluations: Rc::new(Cell::new(0)),
        }
    }

    #[test]
    fn ready_independent_root_binds_without_waiting_for_an_earlier_dependent() {
        let mut draft = WorkflowPlanDraft::new();
        let output = draft.push(policy(0, 0, vec![OutputDescription::ProducedTensor {
            device: DeviceIdentity(1), representation: representation(), rank: 1,
        }]), vec![]).unwrap()[0];
        let dependent = vec![ArgumentBinding::Result(output)];
        assert!(!draft.bindings_ready(&dependent));
        let independent = policy(2, 0, vec![]);
        assert!(draft.bindings_ready(&[]));
        let prepared = draft.prepare_at(2, &[], |arguments| independent.evaluate(arguments)).unwrap();
        assert_eq!(prepared.selection, 2);
        assert_eq!(independent.evaluations.get(), 1);
        assert_eq!(draft.nodes.len(), 1, "prebinding does not change node ordinals or execute the root");
        assert!(draft.requirements.is_empty(), "prebinding reserves no speculative backing");
    }

    #[test]
    fn produced_tensor_binds_dependents_only_after_actual_publication() {
        let mut draft = WorkflowPlanDraft::new();
        let mut producer = policy(0, 0, vec![OutputDescription::ProducedTensor {
            device: DeviceIdentity(1), representation: representation(), rank: 1,
        }]);
        producer.allocations.push(AllocationDescription {
            kind: AllocationKind::Temporary,
            acquisition: Acquisition::ReachedPrivate,
            alignment: 4,
        });
        let output = draft.push(producer, vec![]).unwrap()[0];
        let consumer = policy(1, 1, vec![]);
        assert!(matches!(draft.push(consumer.clone(), vec![ArgumentBinding::Result(output)]),
            Err(PlanError::HostBoundaryRequired(_))));
        assert_eq!(consumer.evaluations.get(), 0);
        let backing = draft.nodes[0].private_resources[0].resource;
        let actual = TensorDescriptor {
            device: DeviceIdentity(1), resource: backing,
            representation: representation(), extents: vec![3], strides: vec![2],
            range: ByteRange { offset: 4, len: 20 },
        };
        draft.publish_tensor(output, actual.clone()).unwrap();
        assert!(draft.publish_tensor(output, actual.clone()).is_err());
        draft.push(consumer.clone(), vec![ArgumentBinding::Result(output)]).unwrap();
        assert_eq!(consumer.evaluations.get(), 1);
        assert_eq!(draft.nodes[1].arguments, vec![ValueDescriptor::Tensor(actual)]);
        assert_eq!(draft.nodes[1].dependencies, vec![0]);
        assert_eq!(draft.requirements[&backing].acquisition, Acquisition::ReachedPrivate);
    }

    #[test]
    fn chains_and_diamonds_resolve_without_an_allocator() {
        let mut draft = WorkflowPlanDraft::new();
        let root = draft.push(policy(0, 0, vec![fresh(64)]), vec![]).unwrap()[0];
        let left = draft
            .push(
                policy(1, 1, vec![fresh(32)]),
                vec![ArgumentBinding::Result(root)],
            )
            .unwrap()[0];
        let right = draft
            .push(
                policy(2, 1, vec![fresh(32)]),
                vec![ArgumentBinding::Result(root)],
            )
            .unwrap()[0];
        draft
            .push(
                policy(3, 2, vec![fresh(16)]),
                vec![
                    ArgumentBinding::Result(left),
                    ArgumentBinding::Result(right),
                ],
            )
            .unwrap();

        let plan = draft.close().unwrap();
        assert_eq!(plan.nodes().len(), 4);
        assert_eq!(plan.nodes()[3].selection, 3);
        let root_resource = match &plan.nodes()[0].outputs[0] {
            ValueDescriptor::Tensor(tensor) => tensor.resource,
            _ => unreachable!(),
        };
        assert_eq!(
            plan.lifetimes()[&root_resource],
            Lifetime { first: 0, end: 4 }
        );
    }

    #[test]
    fn rejects_future_missing_and_cross_workflow_references_before_policy_evaluation() {
        let mut first = WorkflowPlanDraft::new();
        let second: WorkflowPlanDraft<u32, ()> = WorkflowPlanDraft::new();
        let invalid = [second.output(0, 0), first.output(1, 0), first.output(0, 3)];
        for reference in invalid {
            let probe = policy(0, 1, vec![]);
            let evaluations = probe.evaluations.clone();
            assert!(matches!(
                first.push(probe, vec![ArgumentBinding::Result(reference)]),
                Err(PlanError::InvalidReference(found)) if found == reference
            ));
            assert_eq!(evaluations.get(), 0);
        }
        first.push(policy(0, 0, vec![fresh(8)]), vec![]).unwrap();
    }

    #[test]
    fn reached_private_requirement_retains_identity_without_inventing_initial_bytes() {
        let mut draft = WorkflowPlanDraft::new();
        let mut producer = policy(0, 0, vec![]);
        producer.allocations.push(AllocationDescription {
            kind: AllocationKind::Temporary,
            acquisition: Acquisition::ReachedPrivate,
            alignment: 16,
        });
        draft.push(producer, vec![]).unwrap();
        let plan = draft.close().unwrap();
        let requirement = &plan.nodes()[0].private_resources[0];
        assert_eq!(requirement.acquisition.initial_bytes(), None);
        assert_eq!(requirement.lifetime, Lifetime { first: 0, end: 1 });
        assert_eq!(plan.requirements().get(&requirement.resource), Some(requirement));
        assert!(plan.nodes()[0].accesses.is_empty(), "unpublished node-local backing has no cross-node hazard");
    }

    #[test]
    fn reached_private_requirement_cannot_masquerade_as_a_published_result() {
        let mut draft = WorkflowPlanDraft::new();
        let mut producer = policy(0, 0, vec![OutputDescription::Tensor {
            device: DeviceIdentity(1), representation: representation(), extents: vec![1], strides: vec![1],
            storage: TensorStorage::Private { allocation: 0, range: ByteRange { offset: 0, len: 4 } },
        }]);
        producer.allocations.push(AllocationDescription {
            kind: AllocationKind::Temporary, acquisition: Acquisition::ReachedPrivate, alignment: 4,
        });
        assert!(matches!(draft.push(producer, vec![]), Err(PlanError::InvalidPolicyDescription(
            "reached private backing cannot be published"
        ))));
    }

    #[test]
    fn views_and_alias_outputs_preserve_identity_and_compose_ranges() {
        let mut draft = WorkflowPlanDraft::new();
        let source = draft.push(policy(0, 0, vec![fresh(64)]), vec![]).unwrap()[0];
        let alias = OutputDescription::Tensor {
            device: DeviceIdentity(1),
            representation: representation(),
            extents: vec![2],
            strides: vec![1],
            storage: TensorStorage::Alias {
                argument: 0,
                range: ByteRange { offset: 4, len: 8 },
            },
        };
        let result = draft
            .push(
                policy(1, 1, vec![alias]),
                vec![ArgumentBinding::View {
                    result: source,
                    view: TensorView {
                        relative_range: ByteRange {
                            offset: 16,
                            len: 32,
                        },
                        extents: vec![8],
                        strides: vec![1],
                    },
                }],
            )
            .unwrap()[0];
        let plan = draft.close().unwrap();
        let source = match &plan.nodes()[0].outputs[0] {
            ValueDescriptor::Tensor(tensor) => tensor,
            _ => unreachable!(),
        };
        let alias = match &plan.nodes()[1].outputs[0] {
            ValueDescriptor::Tensor(tensor) => tensor,
            _ => unreachable!(),
        };
        assert_eq!(source.resource, alias.resource);
        assert_eq!(alias.range, ByteRange { offset: 20, len: 8 });
        assert_eq!(result.node(), 1);
    }

    #[test]
    fn invalid_views_and_device_scalars_are_explicit_outcomes() {
        let mut draft = WorkflowPlanDraft::new();
        let outputs = draft
            .push(
                policy(
                    0,
                    0,
                    vec![
                        fresh(8),
                        OutputDescription::Scalar(ScalarDescriptor::DeviceProduced),
                    ],
                ),
                vec![],
            )
            .unwrap();
        assert!(matches!(
            draft.push(
                policy(1, 1, vec![]),
                vec![ArgumentBinding::View {
                    result: outputs[0],
                    view: TensorView {
                        relative_range: ByteRange { offset: 4, len: 8 },
                        extents: vec![2],
                        strides: vec![1],
                    },
                }],
            ),
            Err(PlanError::InvalidView(reference)) if reference == outputs[0]
        ));
        assert!(matches!(
            draft.push(
                policy(2, 1, vec![]),
                vec![ArgumentBinding::Result(outputs[1])],
            ),
            Err(PlanError::HostBoundaryRequired(reference)) if reference == outputs[1]
        ));
    }

    #[test]
    fn external_aliases_merge_hazards_and_private_requirements_remain_symbolic() {
        let external = ValueDescriptor::Tensor(TensorDescriptor {
            device: DeviceIdentity(1),
            resource: ResourceId::External(41),
            representation: representation(),
            extents: vec![4],
            strides: vec![1],
            range: ByteRange { offset: 0, len: 16 },
        });
        let mut draft = WorkflowPlanDraft::new();
        let node = Policy {
            selection: 5,
            outputs: vec![],
            access: vec![Some(AccessMode::Read), Some(AccessMode::Write)],
            allocations: vec![
                AllocationDescription {
                    kind: AllocationKind::Temporary,
                    acquisition: Acquisition::Invocation(24),
                    alignment: 8,
                },
                AllocationDescription {
                    kind: AllocationKind::Persistent {
                        owner: 9,
                        variant: 0,
                        slot: 2,
                    },
                    acquisition: Acquisition::Invocation(64),
                    alignment: 16,
                },
            ],
            evaluations: Rc::new(Cell::new(0)),
        };
        draft
            .push(
                node,
                vec![
                    ArgumentBinding::External(external.clone()),
                    ArgumentBinding::External(external),
                ],
            )
            .unwrap();
        let plan = draft.close().unwrap();
        assert_eq!(
            plan.nodes()[0]
                .accesses
                .iter()
                .filter(|access| access.resource == ResourceId::External(41))
                .count(),
            1
        );
        assert_eq!(plan.nodes()[0].accesses[0].mode, AccessMode::ReadWrite);
        assert_eq!(plan.nodes()[0].private_resources.len(), 2);
        assert!(matches!(
            plan.nodes()[0].private_resources[1].resource,
            ResourceId::Persistent {
                owner: 9,
                variant: 0,
                slot: 2
            }
        ));
    }

    #[test]
    fn empty_graph_and_malformed_policy_descriptions_are_rejected() {
        let empty: WorkflowPlanDraft<u32, ()> = WorkflowPlanDraft::new();
        assert!(matches!(empty.close(), Err(PlanError::Empty)));

        let mut draft = WorkflowPlanDraft::new();
        let malformed = policy(0, 1, vec![]);
        assert!(matches!(
            draft.push(malformed, vec![]),
            Err(PlanError::InvalidPolicyDescription(_))
        ));
    }

    #[test]
    fn fresh_output_requirements_preserve_size_alignment_and_boundary_lifetime() {
        let mut draft = WorkflowPlanDraft::new();
        let output = draft.push(policy(0, 0, vec![fresh(96)]), vec![]).unwrap()[0];
        let plan = draft.close().unwrap();
        let resource = ResourceId::Produced {
            workflow: output.workflow(),
            node: output.node(),
            output: output.output(),
        };
        assert_eq!(
            plan.requirements()[&resource],
            ResourceRequirement {
                resource,
                acquisition: Acquisition::Invocation(96),
                alignment: 8,
                lifetime: Lifetime { first: 0, end: 1 },
            }
        );
    }

    #[test]
    fn private_outputs_reuse_the_declared_symbolic_allocation() {
        let mut draft = WorkflowPlanDraft::new();
        let producer = Policy {
            selection: 0,
            outputs: vec![OutputDescription::Tensor {
                device: DeviceIdentity(1),
                representation: representation(),
                extents: vec![8],
                strides: vec![1],
                storage: TensorStorage::Private {
                    allocation: 0,
                    range: ByteRange {
                        offset: 16,
                        len: 32,
                    },
                },
            }],
            access: vec![],
            allocations: vec![AllocationDescription {
                kind: AllocationKind::Temporary,
                acquisition: Acquisition::Invocation(64),
                alignment: 16,
            }],
            evaluations: Rc::new(Cell::new(0)),
        };
        let output = draft.push(producer, vec![]).unwrap()[0];
        draft
            .push(policy(1, 1, vec![]), vec![ArgumentBinding::Result(output)])
            .unwrap();
        let plan = draft.close().unwrap();
        let requirement = &plan.nodes()[0].private_resources[0];
        let ValueDescriptor::Tensor(output) = &plan.nodes()[0].outputs[0] else {
            unreachable!()
        };
        assert_eq!(output.resource, requirement.resource);
        assert_eq!(
            output.range,
            ByteRange {
                offset: 16,
                len: 32
            }
        );
        assert_eq!(requirement.acquisition, Acquisition::Invocation(64));
        assert_eq!(requirement.alignment, 16);
        assert_eq!(requirement.lifetime, Lifetime { first: 0, end: 2 });
        assert_eq!(plan.nodes()[1].dependencies, vec![0]);
    }

    #[test]
    fn persistent_identity_aggregates_capacity_and_alignment() {
        let persistent = |bytes, alignment| Policy {
            selection: 0,
            outputs: vec![],
            access: vec![],
            allocations: vec![AllocationDescription {
                kind: AllocationKind::Persistent {
                    owner: 4,
                    variant: 0,
                    slot: 2,
                },
                acquisition: Acquisition::Invocation(bytes),
                alignment,
            }],
            evaluations: Rc::new(Cell::new(0)),
        };
        let mut draft = WorkflowPlanDraft::new();
        draft.push(persistent(64, 16), vec![]).unwrap();
        draft.push(persistent(128, 32), vec![]).unwrap();
        let plan = draft.close().unwrap();
        assert_eq!(plan.nodes().len(), 2);
        assert_eq!(plan.nodes()[1].dependencies, vec![0]);
        let requirement = &plan.requirements()[&ResourceId::Persistent {
            owner: 4,
            variant: 0,
            slot: 2,
        }];
        assert_eq!(requirement.acquisition, Acquisition::Invocation(128));
        assert_eq!(requirement.alignment, 32);
        assert_eq!(requirement.lifetime, Lifetime { first: 0, end: 2 });
    }

    #[test]
    fn scalar_edges_preserve_dependencies_or_report_a_host_boundary() {
        let mut host = WorkflowPlanDraft::new();
        let value = host
            .push(
                policy(
                    0,
                    0,
                    vec![OutputDescription::Scalar(ScalarDescriptor::HostReady(
                        ScalarValue::U32(3),
                    ))],
                ),
                vec![],
            )
            .unwrap()[0];
        let mut consumer = policy(1, 1, vec![]);
        consumer.access[0] = None;
        host.push(consumer, vec![ArgumentBinding::Result(value)])
            .unwrap();
        let plan = host.close().unwrap();
        assert_eq!(plan.nodes()[1].dependencies, vec![0]);

        let mut device = WorkflowPlanDraft::new();
        let value = device
            .push(
                policy(
                    0,
                    0,
                    vec![OutputDescription::Scalar(ScalarDescriptor::DeviceProduced)],
                ),
                vec![],
            )
            .unwrap()[0];
        let mut consumer = policy(1, 1, vec![]);
        consumer.access[0] = None;
        assert!(matches!(
            device.push(consumer, vec![ArgumentBinding::Result(value)]),
            Err(PlanError::HostBoundaryRequired(found)) if found == value
        ));
    }

    #[test]
    fn overlapping_nonidentical_ranges_remain_explicit_for_hazard_analysis() {
        let tensor = |offset, len| {
            assert_eq!(len % 4, 0);
            ValueDescriptor::Tensor(TensorDescriptor {
                device: DeviceIdentity(1),
                resource: ResourceId::External(77),
                representation: representation(),
                extents: vec![len / 4],
                strides: vec![1],
                range: ByteRange { offset, len },
            })
        };
        let mut draft = WorkflowPlanDraft::new();
        draft
            .push(
                policy(0, 2, vec![]),
                vec![
                    ArgumentBinding::External(tensor(0, 12)),
                    ArgumentBinding::External(tensor(8, 12)),
                ],
            )
            .unwrap();
        let plan = draft.close().unwrap();
        assert_eq!(plan.nodes()[0].accesses.len(), 2);
        assert_eq!(
            plan.nodes()[0].accesses[0].range,
            ByteRange { offset: 0, len: 12 }
        );
        assert_eq!(
            plan.nodes()[0].accesses[1].range,
            ByteRange { offset: 8, len: 12 }
        );
    }

    #[test]
    fn floating_scalar_identity_is_bit_stable() {
        let positive_zero = ScalarValue::from_f32(0.0);
        let negative_zero = ScalarValue::from_f32(-0.0);
        assert_ne!(positive_zero, negative_zero);
        assert_eq!(positive_zero.as_f32().unwrap().to_bits(), 0.0f32.to_bits());
        assert_eq!(
            negative_zero.as_f32().unwrap().to_bits(),
            (-0.0f32).to_bits()
        );

        let nan = ScalarValue::from_f32(f32::from_bits(0x7fc0_1234));
        let same_nan = ScalarValue::from_f32(f32::from_bits(0x7fc0_1234));
        let other_nan = ScalarValue::from_f32(f32::from_bits(0x7fc0_5678));
        assert_eq!(nan, same_nan);
        assert_ne!(nan, other_nan);
    }

    #[test]
    fn alias_outputs_write_and_order_later_overlapping_accesses() {
        let external = ValueDescriptor::Tensor(TensorDescriptor {
            device: DeviceIdentity(1),
            resource: ResourceId::External(90),
            representation: representation(),
            extents: vec![4],
            strides: vec![1],
            range: ByteRange { offset: 0, len: 16 },
        });
        let alias = OutputDescription::Tensor {
            device: DeviceIdentity(1),
            representation: representation(),
            extents: vec![2],
            strides: vec![1],
            storage: TensorStorage::Alias {
                argument: 0,
                range: ByteRange { offset: 4, len: 8 },
            },
        };
        let mut producer = policy(0, 1, vec![alias]);
        producer.access[0] = None;
        let mut draft = WorkflowPlanDraft::new();
        draft
            .push(producer, vec![ArgumentBinding::External(external)])
            .unwrap();
        draft
            .push(
                policy(1, 1, vec![]),
                vec![ArgumentBinding::External(ValueDescriptor::Tensor(
                    TensorDescriptor {
                        device: DeviceIdentity(1),
                        resource: ResourceId::External(90),
                        representation: representation(),
                        extents: vec![2],
                        strides: vec![1],
                        range: ByteRange { offset: 4, len: 8 },
                    },
                ))],
            )
            .unwrap();
        let plan = draft.close().unwrap();
        assert_eq!(plan.nodes()[0].accesses.len(), 1);
        assert_eq!(plan.nodes()[0].accesses[0].mode, AccessMode::Write);
        assert_eq!(plan.nodes()[1].dependencies, vec![0]);
        assert_eq!(
            plan.lifetimes()[&ResourceId::External(90)],
            Lifetime { first: 0, end: 2 }
        );
    }

    #[test]
    fn hazard_edges_cover_writes_but_do_not_serialize_readers() {
        let tensor = |resource, offset, len| {
            ValueDescriptor::Tensor(TensorDescriptor {
                device: DeviceIdentity(1),
                resource: ResourceId::External(resource),
                representation: representation(),
                extents: vec![len / 4],
                strides: vec![1],
                range: ByteRange { offset, len },
            })
        };
        let mut draft = WorkflowPlanDraft::new();
        let mut writer = policy(0, 1, vec![]);
        writer.access[0] = Some(AccessMode::Write);
        draft
            .push(writer, vec![ArgumentBinding::External(tensor(1, 0, 12))])
            .unwrap();
        draft
            .bind_and_insert(
                vec![ArgumentBinding::External(tensor(1, 8, 12))],
                |arguments| policy(1, 1, vec![]).evaluate(arguments),
            )
            .unwrap();
        draft
            .push(
                policy(2, 1, vec![]),
                vec![ArgumentBinding::External(tensor(1, 24, 8))],
            )
            .unwrap();
        draft
            .push(
                policy(3, 1, vec![]),
                vec![ArgumentBinding::External(tensor(2, 0, 8))],
            )
            .unwrap();
        draft
            .bind_and_insert(
                vec![ArgumentBinding::External(tensor(2, 0, 8))],
                |arguments| policy(4, 1, vec![]).evaluate(arguments),
            )
            .unwrap();
        let plan = draft.close().unwrap();
        assert_eq!(plan.nodes()[1].dependencies, vec![0]);
        assert!(plan.nodes()[2].dependencies.is_empty());
        assert!(plan.nodes()[4].dependencies.is_empty());
    }

    #[test]
    fn representation_geometry_is_checked_before_policy_evaluation() {
        let malformed = [
            TensorDescriptor {
                device: DeviceIdentity(1),
                resource: ResourceId::External(1),
                representation: representation(),
                extents: vec![4],
                strides: vec![2],
                range: ByteRange { offset: 0, len: 16 },
            },
            TensorDescriptor {
                device: DeviceIdentity(1),
                resource: ResourceId::External(1),
                representation: representation(),
                extents: vec![4],
                strides: vec![1],
                range: ByteRange { offset: 0, len: 12 },
            },
        ];
        let mut draft = WorkflowPlanDraft::new();
        for tensor in malformed {
            let probe = policy(0, 1, vec![]);
            let evaluations = probe.evaluations.clone();
            assert!(matches!(
                draft.push(
                    probe,
                    vec![ArgumentBinding::External(ValueDescriptor::Tensor(tensor))]
                ),
                Err(PlanError::InvalidPolicyDescription(_))
            ));
            assert_eq!(evaluations.get(), 0);
        }
    }
}
