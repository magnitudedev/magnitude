//! Production adapter from descriptor-first workflow plans to native resources.
use crate::driver::workflow::{
    AccessMode, AllocationDescription, AllocationKind, ArgumentBinding, BoundInvocation, BoundNode,
    ByteRange, OutputDescription, OutputRef, PlanError, PreparedPolicy, ResourceId,
    ScalarDescriptor, ScalarValue, TensorDescriptor, TensorStorage, WorkflowId, WorkflowPlanDraft,
};
use crate::driver::AdmittedCommand;
use crate::driver::*;
use crate::execution::{AdmittedNode, AdmittedOutput, AdmittedRun, AdmittedScalarOutput};
use crate::resources::{
    AdmittedResources, PersistentAvailability, PersistentBinding, PersistentTable,
};

use super::Acquisition;
#[derive(Clone, Copy)]
enum PlannedAllocation {
    Argument { argument: usize },
    Private { slot: usize, executable: usize },
}

struct PlannedNode<T: TargetFamily, E: NativeExecutor<T>> {
    selected: Arc<SelectedExecutable<T, E>>,
    values: InvocationValues,
    allocations: Vec<PlannedAllocation>,
}
impl<T: TargetFamily, E: NativeExecutor<T>> Clone for PlannedNode<T, E> {
    fn clone(&self) -> Self { Self { selected: self.selected.clone(), values: self.values.clone(), allocations: self.allocations.clone() } }
}

enum InvocationTarget<T: TargetFamily, E: NativeExecutor<T>> {
    Policy(Arc<PreparedHandle<T, E>>),
    Candidate {
        device: Arc<Opened<T, E>>,
        selected: Arc<SelectedExecutable<T, E>>,
    },
}

impl<T: TargetFamily, E: NativeExecutor<T>> InvocationTarget<T, E> {
    fn persistent_inventory(&self, keys: &mut BTreeMap<(u64, usize, usize), Arc<PersistentTable>>) {
        let mut include = |owner, variant, executable: &ExecutableVariant<T, E::Handle>, table: &Arc<PersistentTable>| {
            for (slot, allocation) in executable.allocations().iter().enumerate() {
                if matches!(allocation.kind, ExecutableAllocationKind::Global(ExecutableGlobalAllocationKind::Persistent)) {
                    keys.entry((owner, variant, slot)).or_insert_with(|| table.clone());
                }
            }
        };
        match self {
            Self::Policy(kernel) => {
                for (variant, executable) in kernel.prepared.kernel.variants().as_slice().iter().enumerate() {
                    include(kernel.prepared.identity, variant, executable, &kernel.prepared.persistent);
                }
            }
            Self::Candidate { selected, .. } => include(selected.owner, selected.variant, &selected.executable, &selected.persistent),
        }
    }
}

impl<T: TargetFamily, E: NativeExecutor<T>> PreparedPolicy for InvocationTarget<T, E> {
    type Selection = PlannedNode<T, E>;
    type Error = CallError;

    fn evaluate(
        &self,
        arguments: Vec<crate::driver::workflow::ValueDescriptor>,
    ) -> Result<BoundInvocation<Self::Selection>, Self::Error> {
        let (contract, device) = match self {
            Self::Policy(kernel) => (
                kernel.prepared.kernel.invocation_contract(),
                &kernel.prepared.device,
            ),
            Self::Candidate { device, selected } => {
                (selected.executable.invocation_contract(), device)
            }
        };
        let mut allocation_identities = BTreeMap::new();
        for argument in &arguments {
            if let crate::driver::workflow::ValueDescriptor::Tensor(tensor) = argument {
                let next = u64::try_from(allocation_identities.len() + 1)
                    .expect("invocation allocation identity space exhausted");
                allocation_identities.entry(tensor.resource).or_insert(next);
            }
        }
        let values = arguments
            .iter()
            .map(|argument| descriptor_argument(argument, &allocation_identities))
            .collect::<Vec<_>>();
        let invocation = validate_invocation(contract, device.identity(), &values)
            .map_err(CallError::Invocation)?;
        let selected = match self {
            Self::Policy(kernel) => {
                let variant = kernel.prepared.kernel.select(&invocation).as_usize();
                Arc::new(SelectedExecutable {
                    executable: kernel.prepared.kernel.variants().as_slice()[variant].clone(),
                    owner: kernel.prepared.identity,
                    variant,
                    persistent: kernel.prepared.persistent.clone(),
                    output_device: kernel.device.clone(),
                })
            }
            Self::Candidate { selected, .. } => {
                assert_eq!(
                    selected.executable.device_identity(),
                    device.device_description().identity(),
                    "candidate execution belongs to its compilation device"
                );
                if !admitted(selected.executable.guard().evaluate(&invocation)) {
                    return Err(CallError::Invocation(InvocationError::OutsideTargetDomain));
                }
                selected.clone()
            }
        };
        let executable = &selected.executable;
        let schema = executable.invocation_contract().schema();

        let argument_access = schema
            .parameters()
            .iter()
            .map(|parameter| match &parameter.kind {
                ParameterKind::Tensor { access, .. } => Some(match access {
                    TensorAccess::Shared => AccessMode::Read,
                    TensorAccess::Owned | TensorAccess::Mutable => AccessMode::ReadWrite,
                }),
                ParameterKind::Scalar { .. }
                | ParameterKind::Index { .. }
                | ParameterKind::Range { .. } => None,
            })
            .collect::<Vec<_>>();

        let mut allocations = Vec::with_capacity(executable.allocations().len());
        let mut private_allocations = Vec::new();
        for (index, allocation) in executable.allocations().iter().enumerate() {
            match &allocation.kind {
                ExecutableAllocationKind::Global(ExecutableGlobalAllocationKind::Argument(
                    parameter,
                )) => {
                    allocations.push(PlannedAllocation::Argument {
                        argument: schema.parameter_ordinal(*parameter),
                    });
                }
                _ => {
                    let slot = private_allocations.len();
                    let acquisition = match allocation.acquisition {
                        seismic_compiler::executable::AllocationAcquisition::Invocation => Acquisition::Invocation(evaluated_bytes(allocation, &invocation)?),
                        seismic_compiler::executable::AllocationAcquisition::Reached => Acquisition::ReachedPrivate,
                    };
                    let kind = match &allocation.kind {
                        ExecutableAllocationKind::Global(
                            ExecutableGlobalAllocationKind::Persistent,
                        ) => AllocationKind::Persistent {
                            owner: selected.owner,
                            variant: u32::try_from(selected.variant)
                                .expect("prepared variant ordinal space exhausted"),
                            slot: u32::try_from(index)
                                .expect("executable allocation ordinal space exhausted"),
                        },
                        _ => AllocationKind::Temporary,
                    };
                    private_allocations.push(AllocationDescription {
                        kind,
                        acquisition,
                        alignment: allocation.alignment,
                    });
                    allocations.push(PlannedAllocation::Private {
                        slot,
                        executable: index,
                    });
                }
            }
        }

        let mut outputs = Vec::with_capacity(executable.bindings().results.len());
        for (_, result) in &executable.bindings().results {
            outputs.push(match result {
                ExecutableResultBinding::Buffer { representation, rank } =>
                    OutputDescription::ProducedTensor {
                        device: device.identity(),
                        representation: *representation,
                        rank: *rank,
                    },
                ExecutableResultBinding::Scalar { .. }
                | ExecutableResultBinding::Quantity { .. }
                | ExecutableResultBinding::Range { .. } => {
                    OutputDescription::Scalar(ScalarDescriptor::DeviceProduced)
                }
            });
        }

        Ok(BoundInvocation {
            arguments,
            selection: PlannedNode {
                selected,
                values: invocation,
                allocations,
            },
            argument_access,
            outputs,
            private_allocations,
        })
    }
}

fn descriptor_argument(
    descriptor: &crate::driver::workflow::ValueDescriptor,
    allocation_identities: &BTreeMap<ResourceId, u64>,
) -> ArgumentValue {
    match descriptor {
        crate::driver::workflow::ValueDescriptor::Tensor(tensor) => {
            ArgumentValue::Tensor(seismic_compiler::prepared::TensorDescriptor {
                device: tensor.device,
                representation: tensor.representation,
                extents: tensor.extents.clone(),
                strides: tensor.strides.clone(),
                allocation: allocation_identities[&tensor.resource],
                byte_offset: tensor.range.offset,
                byte_len: tensor.range.len,
            })
        }
        crate::driver::workflow::ValueDescriptor::Scalar(ScalarDescriptor::HostReady(value)) => {
            scalar_argument(value.clone())
        }
        crate::driver::workflow::ValueDescriptor::PendingTensor { .. }
        | crate::driver::workflow::ValueDescriptor::Scalar(ScalarDescriptor::DeviceProduced) => {
            unreachable!("device-produced scalars are rejected at the workflow boundary")
        }
    }
}

fn scalar_argument(value: ScalarValue) -> ArgumentValue {
    match value {
        ScalarValue::F32Bits(bits) => ArgumentValue::F32(f32::from_bits(bits)),
        ScalarValue::F16(value) => ArgumentValue::F16(value),
        ScalarValue::BF16(value) => ArgumentValue::BF16(value),
        ScalarValue::I32(value) => ArgumentValue::I32(value),
        ScalarValue::U32(value) => ArgumentValue::U32(value),
        ScalarValue::Bool(value) => ArgumentValue::Bool(value),
        ScalarValue::Index(value) => ArgumentValue::Index(value),
        ScalarValue::Range { start, end } => ArgumentValue::Range { start, end },
    }
}

fn evaluated_bytes(
    allocation: &seismic_compiler::executable::AllocationPlan,
    values: &InvocationValues,
) -> Result<u64, CallError> {
    let candidates = allocation.byte_candidates();
    let mut required = seismic_lang::expr::BigUint::default();
    for bytes in std::iter::once(candidates.first()).chain(candidates.rest().iter()) {
        let bytes = bytes.evaluate(values).map_err(|error| CallError::Execution(
            ExecutionError::ConstructionContradiction(format!("upfront allocation size is undefined: {error:?}"))))?;
        required = required.max(bytes);
    }
    u64::try_from(&required).map_err(|_| CallError::Invocation(InvocationError::AllocationCapacity {
        required, available: u64::MAX,
    }))
}

struct QueuedNode<T: TargetFamily, E: NativeExecutor<T>> {
    kernel: Arc<PreparedHandle<T, E>>,
    args: EncodedWorkflowArgs,
}

pub(crate) struct WorkflowGraphDraft<T: TargetFamily, E: NativeExecutor<T>> {
    identity: WorkflowId,
    device: Arc<Opened<T, E>>,
    nodes: Vec<QueuedNode<T, E>>,
    result_counts: Vec<u32>,
}

/// A graph with validated references and its first invocation bound. Later
/// policy choices consume completed producer descriptors in the same planner.
/// Binding remains pure with respect to device resource state.
pub(crate) struct BoundWorkflowGraph<T: TargetFamily, E: NativeExecutor<T>> {
    continuation: WorkflowContinuation<T, E>,
    keys: BTreeMap<(u64, usize, usize), Arc<PersistentTable>>,
    allocation_limit: u64,
    device: Arc<Opened<T, E>>,
}

pub(crate) struct WorkflowContinuation<T: TargetFamily, E: NativeExecutor<T>> {
    planner: WorkflowPlanDraft<PlannedNode<T, E>, CallError>,
    queued: std::collections::VecDeque<QueuedInvocation<T, E>>,
    physical: BTreeMap<ResourceId, Arc<Allocation>>,
    next_node: u32,
}

struct QueuedInvocation<T: TargetFamily, E: NativeExecutor<T>> {
    target: InvocationTarget<T, E>,
    bindings: Vec<ArgumentBinding>,
    prepared: Option<BoundInvocation<PlannedNode<T, E>>>,
}

impl<T: TargetFamily, E: NativeExecutor<T>> WorkflowContinuation<T, E> {
    fn prebind_ready(&mut self) -> Result<(), CallError> {
        let first = self.planner.nodes.len();
        for (offset, queued) in self.queued.iter_mut().enumerate() {
            if queued.prepared.is_none() && self.planner.bindings_ready(&queued.bindings) {
                queued.prepared = Some(self.planner.prepare_at((first + offset) as u32, &queued.bindings,
                    |arguments| queued.target.evaluate(arguments)).map_err(plan_error)?);
            }
        }
        Ok(())
    }

    fn bind_next(&mut self) -> Result<bool, CallError> {
        self.prebind_ready()?;
        let Some(queued) = self.queued.pop_front() else { return Ok(false); };
        match queued.prepared {
            Some(prepared) => { self.planner.bind_and_insert(queued.bindings, |_| Ok(prepared)).map_err(plan_error)?; }
            None => { self.planner.push(queued.target, queued.bindings).map_err(plan_error)?; }
        }
        Ok(true)
    }
    pub(crate) fn next(&mut self, device: &Arc<Opened<T, E>>, resources: &mut AdmittedResources)
        -> Result<Option<AdmittedNode<T, E>>, CallError> {
        if self.planner.bound_node(self.next_node).is_none() {
            if !self.bind_next()? { return Ok(None); }
        }
        let node = self.planner.bound_node(self.next_node).expect("node just bound").clone();
        let admitted = admit_reached_node(node, &mut self.physical, device, resources)?;
        self.next_node += 1;
        Ok(Some(admitted))
    }

    pub(crate) fn completed(&mut self, ordinal: u32, completed: &AdmittedNode<T, E>, device: seismic_compiler::prepared::DeviceIdentity, resources: &mut AdmittedResources) -> Result<(), CallError> {
        let planned = self.planner.bound_node(ordinal).expect("completed node was bound");
        let mut publications = Vec::new();
        let mut retained_slots = std::collections::BTreeSet::new();
        for ((path, _), output) in planned.selection.selected.executable.bindings().results.iter().zip(&completed.outputs) {
            let AdmittedOutput::Tensor { allocation, byte_offset, byte_len, representation, extents, strides } = output else { continue; };
            retained_slots.insert(resources.slot_for(allocation));
            let index = completed.command.published_allocation(path).ok_or_else(|| CallError::Execution(
                ExecutionError::ConstructionContradiction("tensor result has no executed publication".into()),
            ))?;
            let resource = match planned.selection.allocations[index] {
                PlannedAllocation::Argument { argument } => {
                    let crate::driver::workflow::ValueDescriptor::Tensor(tensor) = &planned.arguments[argument] else {
                        unreachable!("tensor publication references a tensor argument")
                    };
                    tensor.resource
                }
                PlannedAllocation::Private { slot, .. } => planned.private_resources[slot].resource,
            };
            publications.push((allocation.clone(), TensorDescriptor {
                device, resource,
                representation: *representation, extents: extents.clone(), strides: strides.clone(),
                range: ByteRange { offset: *byte_offset, len: *byte_len },
            }));
        }
        for (index, allocation) in planned.selection.allocations.iter().enumerate() {
            let PlannedAllocation::Private { slot, .. } = allocation else { continue; };
            let resource = planned.private_resources[*slot].resource;
            if matches!(resource, ResourceId::Persistent { .. }) { continue; }
            let slot = completed.command.allocation_slot(index);
            if !retained_slots.contains(&slot) {
                // The caller completed this entire native prefix. No output
                // descriptor retains this bank, and no later source use in
                // this invocation exists; its charge can end now.
                resources.retire_private(slot);
                self.physical.remove(&resource);
            }
        }
        let mut tensor_publications = publications.into_iter();
        for (output, value) in completed.outputs.iter().enumerate() {
            if matches!(value, AdmittedOutput::Tensor { .. }) {
                let (backing, descriptor) = tensor_publications.next().expect("one descriptor per tensor output");
                self.physical.insert(descriptor.resource, backing);
                let reference = self.planner.output(ordinal, output as u32);
                self.planner.publish_tensor(reference, descriptor).map_err(plan_error)?;
            }
        }
        Ok(())
    }
}

/// Claim identities, not speculative capacity. The complete inventory is
/// finite in the prepared portfolio, including keys with no backing yet.
/// All waits happen before this run can issue source effects.
fn preclaim_resources<T: TargetFamily, E: NativeExecutor<T>>(
    device: &Arc<Opened<T, E>>,
    keys: &BTreeMap<(u64, usize, usize), Arc<PersistentTable>>,
    external: &BTreeMap<ResourceId, Arc<Allocation>>,
) -> Result<AdmittedResources, CallError> {
    let (snapshots, guard) = loop {
        let guard = device.admission.enter();
        let mut snapshots = BTreeMap::new();
        let mut wait = None;
        for (key, table) in keys {
            match table.preclaim_availability((key.1, key.2)) {
                PersistentAvailability::Grow { old } => { snapshots.insert(*key, old); }
                PersistentAvailability::Wait => { wait = Some((*key, table.clone())); break; }
            }
        }
        if let Some((key, table)) = wait {
            drop(guard);
            table.wait_until_preclaimable((key.1, key.2));
        } else { break (snapshots, guard); }
    };
    let mut requests = BTreeMap::new();
    for allocation in external.values() {
        merge_access_request(&mut requests, allocation.clone(), true, true);
    }
    for old in snapshots.values().filter_map(Option::as_ref) {
        merge_access_request(&mut requests, old.allocation.clone(), true, true);
    }
    let access = requests.into_values()
        .map(|(allocation, write, _)| allocation.acquire(write)).collect();
    let claims = keys.iter().map(|(key, table)| (*key, table.claim_growth((key.1, key.2)))).collect();
    let reservation = device.memory.reserve(0).map_err(|capacity| CallError::Invocation(
        InvocationError::AllocationCapacity { required: capacity.required.into(), available: capacity.available },
    ))?;
    let mut resources = AdmittedResources::new(reservation, access);
    resources.retain_preclaims(claims);
    drop(guard);
    Ok(resources)
}

fn admit_reached_node<T: TargetFamily, E: NativeExecutor<T>>(
    node: BoundNode<PlannedNode<T, E>>,
    physical: &mut BTreeMap<ResourceId, Arc<Allocation>>,
    device: &Arc<Opened<T, E>>,
    resources: &mut AdmittedResources,
) -> Result<AdmittedNode<T, E>, CallError> {
    let mut demands = Vec::new();
    for requirement in &node.private_resources {
        let Some(bytes) = requirement.acquisition.initial_bytes() else { continue; };
        if let ResourceId::Persistent { owner, variant, slot } = requirement.resource {
            let key = (owner, variant as usize, slot as usize);
            if let Some(binding) = resources.preclaimed_binding(key).filter(|binding| binding.capacity >= bytes) {
                physical.insert(requirement.resource, binding.allocation.clone());
                continue;
            }
        } else if physical.contains_key(&requirement.resource) { continue; }
        demands.push((requirement, bytes));
    }
    let exact = demands.iter().fold(seismic_lang::expr::BigUint::default(), |sum, (_, bytes)| sum + *bytes);
    let required = u64::try_from(&exact).map_err(|_| CallError::Invocation(
        InvocationError::AllocationCapacity { required: exact, available: u64::MAX },
    ))?;
    resources.check_reached_capacity(required).map_err(CallError::Execution)?;
    let mut reservation = device.memory.reserve(required).map_err(|capacity| CallError::Invocation(
        InvocationError::AllocationCapacity { required: capacity.required.into(), available: capacity.available },
    ))?;
    let mut replacements = Vec::new();
    for (requirement, bytes) in demands {
        let allocation = device.allocate_reserved(bytes, requirement.alignment, &mut reservation).map_err(CallError::Execution)?;
        if let ResourceId::Persistent { owner, variant, slot } = requirement.resource {
            let key = (owner, variant as usize, slot as usize);
            if let Some(old) = resources.preclaimed_binding(key) {
                copy_between::<T, E>(&*device.service, &typed_buffer::<T, E>(&old.allocation),
                    &typed_buffer::<T, E>(&allocation), old.capacity).map_err(CallError::Execution)?;
            }
            replacements.push((key, PersistentBinding { allocation: allocation.clone(), capacity: bytes }));
        }
        let identity = allocation.identity();
        let permit = allocation.try_acquire(true).expect("fresh reached backing cannot have another access owner");
        resources.declare_private(identity);
        resources.install_private(identity, permit);
        physical.insert(requirement.resource, allocation);
    }
    let selected = node.selection.selected.clone();
    let requirements = node.private_resources.iter().map(|requirement| (requirement.resource, requirement.clone())).collect();
    let mut span = Timed::start("seismic.workflow.reached-admit", Vec::new());
    let (mut staged, outputs) = stage_planned(&selected.executable, node, &requirements, physical, &mut span, 0)
        .map_err(CallError::Execution)?;
    for binding in &mut staged.buffers {
        if let PhysicalBufferBinding::Bound { allocation, .. } = binding {
            let backing = physical.values().find(|backing| backing.identity() == *allocation)
                .expect("staged backing came from the physical resource map");
            *allocation = resources.slot_for(backing);
        }
    }
    // No fallible preparation remains for this node. Before this point an
    // error drops fresh backing while preserving every claimed prior value.
    for (key, binding) in replacements { resources.install_preclaimed(key, binding); }
    Ok(AdmittedNode::new(AdmittedCommand::new(selected, staged), outputs))
}

impl<T: TargetFamily, E: NativeExecutor<T>> WorkflowGraphDraft<T, E> {
    pub(crate) fn new(device: Arc<Opened<T, E>>) -> Self {
        Self {
            identity: WorkflowId::fresh(),
            device,
            nodes: Vec::new(),
            result_counts: Vec::new(),
        }
    }

    pub(crate) fn enqueue(
        &mut self,
        kernel: Arc<PreparedHandle<T, E>>,
        args: EncodedWorkflowArgs,
    ) -> Result<PendingWorkflowResults, crate::api::WorkflowError> {
        if !Arc::ptr_eq(&kernel.prepared.device, &self.device) {
            return Err(crate::api::WorkflowError::CrossWorkflowResult);
        }
        for argument in args.arguments() {
            let reference = match argument {
                EncodedWorkflowArgument::Tensor(WorkflowTensorArgument::Result(reference))
                | EncodedWorkflowArgument::ScalarResult(reference) => Some(reference),
                EncodedWorkflowArgument::Tensor(WorkflowTensorArgument::ResultLeadingSlice {
                    result,
                    ..
                }) => Some(result),
                EncodedWorkflowArgument::Tensor(WorkflowTensorArgument::External(_))
                | EncodedWorkflowArgument::Scalar(_) => None,
            };
            if let Some(reference) = reference {
                if reference.workflow != self.identity.raw()
                    || self
                        .result_counts
                        .get(reference.node as usize)
                        .is_none_or(|count| reference.result >= *count)
                {
                    return Err(crate::api::WorkflowError::CrossWorkflowResult);
                }
            }
        }
        let node = u32::try_from(self.nodes.len()).expect("workflow node ordinal space exhausted");
        let count = u32::try_from(kernel.prepared.kernel.schema().results().len())
            .expect("workflow result ordinal space exhausted");
        self.nodes.push(QueuedNode { kernel, args });
        self.result_counts.push(count);
        Ok(PendingWorkflowResults::new(
            self.identity.raw(),
            node,
            count,
        ))
    }

    pub(crate) fn bind(self) -> Result<BoundWorkflowGraph<T, E>, CallError> {
        if self.nodes.is_empty() {
            return Err(CallError::Workflow(crate::api::WorkflowError::Empty));
        }
        let mut planner = WorkflowPlanDraft::with_identity(self.identity);
        let mut physical = BTreeMap::new();
        let mut pending = std::collections::VecDeque::new();
        let mut keys = BTreeMap::new();
        for queued in self.nodes {
            let mut bindings = Vec::new();
            for argument in queued.args.into_arguments() {
                bindings.push(match argument {
                    EncodedWorkflowArgument::Tensor(WorkflowTensorArgument::External(tensor)) => {
                        let descriptor = tensor_descriptor(&tensor);
                        physical
                            .entry(descriptor.resource)
                            .or_insert_with(|| tensor.allocation().clone());
                        ArgumentBinding::External(crate::driver::workflow::ValueDescriptor::Tensor(
                            descriptor,
                        ))
                    }
                    EncodedWorkflowArgument::Tensor(WorkflowTensorArgument::Result(reference)) => {
                        ArgumentBinding::Result(output_ref(reference))
                    }
                    EncodedWorkflowArgument::Tensor(
                        WorkflowTensorArgument::ResultLeadingSlice { result, start, end },
                    ) => ArgumentBinding::LeadingSlice {
                        result: output_ref(result),
                        start,
                        end,
                    },
                    EncodedWorkflowArgument::Scalar(value) => {
                        ArgumentBinding::External(crate::driver::workflow::ValueDescriptor::Scalar(
                            ScalarDescriptor::HostReady(value),
                        ))
                    }
                    EncodedWorkflowArgument::ScalarResult(reference) => {
                        ArgumentBinding::Result(output_ref(reference))
                    }
                });
            }
            let target = InvocationTarget::Policy(queued.kernel);
            target.persistent_inventory(&mut keys);
            pending.push_back(QueuedInvocation { target, bindings, prepared: None });
        }
        let mut continuation = WorkflowContinuation { planner, queued: pending, physical, next_node: 0 };
        assert!(continuation.bind_next()?);
        Ok(BoundWorkflowGraph {
            continuation,
            keys,
            allocation_limit: u64::MAX,
            device: self.device,
        })
    }
}

impl<T: TargetFamily, E: NativeExecutor<T>> BoundWorkflowGraph<T, E> {
    pub(crate) fn initial_allocation_bytes(&self) -> u64 {
        self.continuation.planner.bound_node(0).expect("bound first node").private_resources.iter()
            .try_fold(0u64, |total, requirement| total.checked_add(requirement.acquisition.initial_bytes().unwrap_or(0)))
            .unwrap_or(u64::MAX)
    }

    pub(crate) fn set_allocation_limit(&mut self, limit: u64) { self.allocation_limit = limit; }

    pub(crate) fn admit(self) -> Result<AdmittedRun<T, E>, CallError> {
        let limit = self.allocation_limit;
        self.admit_with_limit(limit)
    }

    fn admit_with_limit(mut self, limit: u64) -> Result<AdmittedRun<T, E>, CallError> {
        let mut resources = preclaim_resources(&self.device, &self.keys, &self.continuation.physical)?;
        resources.set_reached_budget(limit);
        let submission = self.device.begin_submission().map_err(CallError::Execution)?;
        let first = self.continuation.next(&self.device, &mut resources)?.expect("first node is already bound");
        let identity = self.continuation.planner.identity.raw();
        let mut run = AdmittedRun::new(identity, self.device.service_arc(), self.device,
            vec![first], resources, submission);
        run.set_continuation(self.continuation);
        Ok(run)
    }

}

pub(crate) fn call_one<T: TargetFamily + 'static, E: NativeExecutor<T>>(
    kernel: Arc<PreparedHandle<T, E>>,
    args: EncodedArgs,
) -> Result<(DecodedResults, u64), CallError> {
    let mut workflow = WorkflowGraphDraft::new(kernel.prepared.device.clone());
    workflow
        .enqueue(kernel, args.into_workflow())
        .map_err(CallError::Workflow)?;
    let admitted = workflow.bind()?.admit()?;
    let completed = admitted.submit()?.complete()?;
    let allocated = completed.allocated_bytes();
    let mut values = completed.values()?;
    let values = values
        .pop()
        .expect("one-node workflow completed without its node");
    Ok((DecodedResults::new(values), allocated))
}

/// Admission of a single already-prepared candidate uses the same workflow
/// planner and resource transaction as an ordinary selected variant.
pub(crate) fn admit_trial<T: TargetFamily, E: NativeExecutor<T>>(
    device: &Arc<Opened<T, E>>,
    public_device: &Arc<crate::api::device::DeviceInner>,
    executable: &ExecutableVariant<T, E::Handle>,
    args: EncodedArgs,
    allocation_limit: u64,
) -> Result<AdmittedRun<T, E>, CallError> {
    let mut physical = BTreeMap::new();
    let mut descriptors = Vec::new();
    for argument in args.into_workflow().into_arguments() {
        descriptors.push(match argument {
            EncodedWorkflowArgument::Tensor(WorkflowTensorArgument::External(tensor)) => {
                let descriptor = tensor_descriptor(&tensor);
                physical.insert(descriptor.resource, tensor.allocation().clone());
                crate::driver::workflow::ValueDescriptor::Tensor(descriptor)
            }
            EncodedWorkflowArgument::Scalar(value) => {
                crate::driver::workflow::ValueDescriptor::Scalar(ScalarDescriptor::HostReady(value))
            }
            _ => unreachable!("concrete trial arguments cannot contain workflow result references"),
        });
    }
    let selected = Arc::new(SelectedExecutable {
        executable: executable.clone(),
        owner: NEXT_PREPARED.fetch_add(1, Ordering::Relaxed),
        variant: 0,
        persistent: Arc::new(PersistentTable::new()),
        output_device: public_device.clone(),
    });
    let mut planner = WorkflowPlanDraft::new();
    let bindings = descriptors
        .into_iter()
        .map(ArgumentBinding::External)
        .collect();
    let target = InvocationTarget::Candidate { device: device.clone(), selected };
    let mut keys = BTreeMap::new();
    target.persistent_inventory(&mut keys);
    planner.push(target, bindings).map_err(plan_error)?;
    BoundWorkflowGraph {
        continuation: WorkflowContinuation { planner, queued: std::collections::VecDeque::new(), physical, next_node: 0 },
        keys, allocation_limit, device: device.clone(),
    }.admit_with_limit(allocation_limit)

}

fn output_ref(reference: crate::api::kernel::WorkflowResultRef) -> OutputRef {
    OutputRef::from_raw(reference.workflow, reference.node, reference.result)
}

fn tensor_descriptor(tensor: &TensorInner) -> TensorDescriptor {
    let source = tensor.descriptor();
    TensorDescriptor {
        device: source.device,
        resource: ResourceId::External(tensor.allocation().identity()),
        representation: source.representation,
        extents: source.extents,
        strides: source.strides,
        range: ByteRange {
            offset: source.byte_offset,
            len: source.byte_len,
        },
    }
}

fn plan_error(error: PlanError<CallError>) -> CallError {
    match error {
        PlanError::Empty => CallError::Workflow(crate::api::WorkflowError::Empty),
        PlanError::InvalidReference(_) => {
            CallError::Workflow(crate::api::WorkflowError::MissingProducerResult)
        }
        PlanError::HostBoundaryRequired(_) => {
            CallError::Workflow(crate::api::WorkflowError::HostBoundaryRequired)
        }
        PlanError::ViewOfScalar(_) | PlanError::InvalidView(_) => {
            CallError::Workflow(crate::api::WorkflowError::MissingProducerResult)
        }
        PlanError::InvalidPolicyDescription(message) => CallError::Execution(
            ExecutionError::AllocationFailed(format!("invalid bound workflow: {message}")),
        ),
        PlanError::Policy(error) => error,
    }
}

fn merge_access_request(
    requests: &mut BTreeMap<u64, (Arc<Allocation>, bool, bool)>,
    allocation: Arc<Allocation>,
    write: bool,
    retain: bool,
) {
    requests
        .entry(allocation.identity())
        .and_modify(|(_, known_write, known_retain)| {
            *known_write |= write;
            *known_retain |= retain;
        })
        .or_insert((allocation, write, retain));
}

fn stage_planned<T: TargetFamily, E: NativeExecutor<T>>(
    executable: &ExecutableVariant<T, E::Handle>,
    node: BoundNode<PlannedNode<T, E>>,
    requirements: &BTreeMap<ResourceId, crate::driver::workflow::ResourceRequirement>,
    physical: &BTreeMap<ResourceId, Arc<Allocation>>,
    span: &mut Timed,
    allocated_bytes: u64,
) -> Result<(Staged, Vec<AdmittedOutput>), ExecutionError> {
    let PlannedNode {
        values,
        allocations: planned,
        ..
    } = node.selection;
    let mut buffers = Vec::with_capacity(planned.len());
    for allocation in planned {
        match allocation {
            PlannedAllocation::Argument { argument } => {
                let crate::driver::workflow::ValueDescriptor::Tensor(tensor) =
                    &node.arguments[argument]
                else {
                    panic!("planned tensor allocation references a scalar argument")
                };
                let allocation = physical
                    .get(&tensor.resource)
                    .cloned()
                    .expect("bound external/producer allocation is absent");
                buffers.push(PhysicalBufferBinding::Bound {
                    allocation: allocation.identity(),
                    base_offset: tensor.range.offset,
                    accessible_bytes: tensor.range.len,
                    tensor: Some(seismic_compiler::executable::RuntimeTensorGeometry {
                        representation: tensor.representation, extents: tensor.extents.clone(), strides: tensor.strides.clone(),
                    }),
                });
            }
            PlannedAllocation::Private {
                slot,
                executable: _,
            } => {
                let local = &node.private_resources[slot];
                let requirement = requirements.get(&local.resource).unwrap_or(local);
                if requirement.acquisition == Acquisition::ReachedPrivate {
                    buffers.push(PhysicalBufferBinding::Reached { slot: fresh_allocation_identity(), alignment: requirement.alignment });
                    continue;
                }
                let allocation = physical
                    .get(&requirement.resource)
                    .cloned()
                    .ok_or_else(|| {
                        ExecutionError::AllocationFailed(
                            "admitted private resource has no physical allocation".to_owned(),
                        )
                    })?;
                buffers.push(PhysicalBufferBinding::Bound {
                    allocation: allocation.identity(),
                    base_offset: 0,
                    tensor: None,
                    accessible_bytes: requirement.acquisition.initial_bytes().expect("initial resource requirement"),
                });
            }
        }
    }
    let outputs = close_admitted_outputs::<T, E>(&node.outputs, executable, physical)?;
    span.attribute(key_u64("seismic.allocated_bytes", allocated_bytes));
    let staged = Staged {
        values,
        buffers,
        allocated_bytes,
    };
    Ok((staged, outputs))
}

fn close_admitted_outputs<T: TargetFamily, E: NativeExecutor<T>>(
    descriptors: &[crate::driver::workflow::ValueDescriptor],
    executable: &seismic_compiler::executable::ExecutableVariant<T, E::Handle>,
    physical: &BTreeMap<ResourceId, Arc<Allocation>>,
) -> Result<Vec<AdmittedOutput>, ExecutionError> {
    if descriptors.len() != executable.bindings().results.len() {
        return Err(ExecutionError::AllocationFailed(
            "bound output descriptor count disagrees with executable results".to_owned(),
        ));
    }
    descriptors
        .iter()
        .zip(&executable.bindings().results)
        .map(|(descriptor, (path, binding))| match (descriptor, binding) {
            (
                crate::driver::workflow::ValueDescriptor::PendingTensor { .. },
                ExecutableResultBinding::Buffer { .. },
            ) => Ok(AdmittedOutput::PendingTensor { path: path.clone() }),
            (
                crate::driver::workflow::ValueDescriptor::Tensor(tensor),
                ExecutableResultBinding::Buffer { .. },
            ) => {
                let allocation = physical.get(&tensor.resource).cloned().ok_or_else(|| {
                    ExecutionError::AllocationFailed(
                        "bound output descriptor has no admitted allocation".to_owned(),
                    )
                })?;
                Ok(AdmittedOutput::Tensor {
                    allocation,
                    byte_offset: tensor.range.offset,
                    byte_len: tensor.range.len,
                    representation: tensor.representation,
                    extents: tensor.extents.clone(),
                    strides: tensor.strides.clone(),
                })
            }
            (
                crate::driver::workflow::ValueDescriptor::Scalar(ScalarDescriptor::DeviceProduced),
                ExecutableResultBinding::Scalar { slot },
            ) => Ok(AdmittedOutput::Scalar(match slot.kind() {
                seismic_ir::repr::ScalarKind::Scalar(dtype) => AdmittedScalarOutput::Value {
                    dtype,
                    symbol: slot.symbol(),
                },
                seismic_ir::repr::ScalarKind::Nat64 => AdmittedScalarOutput::Index {
                    symbol: slot.symbol(),
                },
            })),
            (
                crate::driver::workflow::ValueDescriptor::Scalar(ScalarDescriptor::DeviceProduced),
                ExecutableResultBinding::Quantity { slot },
            ) => Ok(AdmittedOutput::Scalar(AdmittedScalarOutput::Index {
                symbol: slot.symbol(),
            })),
            (
                crate::driver::workflow::ValueDescriptor::Scalar(ScalarDescriptor::DeviceProduced),
                ExecutableResultBinding::Range { start, end },
            ) => Ok(AdmittedOutput::Scalar(AdmittedScalarOutput::Range {
                start: start.symbol(),
                end: end.symbol(),
            })),
            _ => Err(ExecutionError::AllocationFailed(
                "bound output descriptor kind disagrees with executable result".to_owned(),
            )),
        })
        .collect()
}

#[cfg(test)]
mod preclaim_tests {
    use super::*;

    #[test]
    fn opposite_graph_key_orders_preclaim_without_deadlock_or_unused_capacity() {
        let catalog = crate::api::catalog::Catalog::discover().unwrap();
        let info = catalog.devices().iter().find(|device| device.backend == registry::BackendName::Cpu).unwrap();
        let device = catalog.open(info.id).unwrap();
        let crate::backends::DeviceKind::Cpu(opened) = &device.kind else { unreachable!() };
        let first = opened.allocate_storage(16, 4).unwrap();
        let second = opened.allocate_storage(16, 4).unwrap();
        let table = Arc::new(PersistentTable::new());
        let mut prior = table.claim_growth((0, 0));
        prior.install_reached(PersistentBinding { allocation: second.clone(), capacity: 16 });
        drop(prior);
        let baseline = opened.memory_usage().charged;
        opened.memory.set_limit(Some(baseline)).unwrap();
        let gate = Arc::new(std::sync::Barrier::new(3));
        let (send, receive) = std::sync::mpsc::channel();
        let mut workers = Vec::new();
        for reverse in [false, true] {
            let opened = opened.clone();
            let table = table.clone();
            let first = first.clone();
            let second = second.clone();
            let gate = gate.clone();
            let send = send.clone();
            workers.push(std::thread::spawn(move || {
                let mut keys = BTreeMap::new();
                for slot in if reverse { [1, 0] } else { [0, 1] } {
                    keys.insert((1, 0, slot), table.clone());
                }
                let mut external = BTreeMap::new();
                let allocations = if reverse { [second, first] } else { [first, second] };
                for (index, allocation) in allocations.into_iter().enumerate() {
                    external.insert(ResourceId::External(index as u64), allocation);
                }
                gate.wait();
                let resources = preclaim_resources(&opened, &keys, &external).unwrap();
                assert_eq!(resources.reached_allocated(), 0);
                assert_eq!(opened.memory_usage().charged, baseline);
                // The absent key belongs to an unselected portfolio arm. No
                // allocation is made even though the memory limit is full.
                assert!(resources.preclaimed_binding((1, 0, 1)).is_none());
                drop(resources);
                send.send(()).unwrap();
            }));
        }
        gate.wait();
        for _ in 0..2 { receive.recv_timeout(std::time::Duration::from_secs(5)).expect("preclaim admission deadlocked"); }
        for worker in workers { worker.join().unwrap(); }
        assert!(matches!(table.preclaim_availability((0, 1)), PersistentAvailability::Grow { old: None }));
        assert_eq!(opened.memory_usage().charged, baseline);
    }
}
