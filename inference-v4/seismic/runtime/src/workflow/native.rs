//! Production adapter from descriptor-first workflow plans to native resources.
use super::*;
use crate::driver::AdmittedCommand;
use crate::execution::{AdmittedNode, AdmittedOutput, AdmittedRun, AdmittedScalarOutput};
use crate::resources::{
    AdmittedResources, PersistentAvailability, PersistentBinding, PersistentGrowth, PersistentTable,
};
use crate::workflow::{
    AccessMode, AllocationDescription, AllocationKind, ArgumentBinding, BoundNode, BoundWorkflow,
    ByteRange, EvaluatedNode, OutputDescription, OutputRef, PlanError, PreparedPolicy, ResourceId,
    ScalarDescriptor, ScalarValue, TensorDescriptor, TensorStorage, WorkflowId, WorkflowPlanDraft,
};

enum PlannedAllocation {
    Argument { argument: usize },
    Private { slot: usize, executable: usize },
}

struct PlannedNode<T: TargetFamily, E: NativeExecutor<T>> {
    kernel: Arc<PreparedHandle<T, E>>,
    values: InvocationValues,
    variant: usize,
    allocations: Vec<PlannedAllocation>,
}

struct WorkflowPolicy<T: TargetFamily, E: NativeExecutor<T>> {
    kernel: Arc<PreparedHandle<T, E>>,
}

impl<T: TargetFamily, E: NativeExecutor<T>> PreparedPolicy for WorkflowPolicy<T, E> {
    type Selection = PlannedNode<T, E>;
    type Error = CallError;

    fn evaluate(
        &self,
        arguments: &[crate::workflow::ValueDescriptor],
    ) -> Result<EvaluatedNode<Self::Selection>, Self::Error> {
        let mut allocation_identities = BTreeMap::new();
        for argument in arguments {
            if let crate::workflow::ValueDescriptor::Tensor(tensor) = argument {
                let next = u64::try_from(allocation_identities.len() + 1)
                    .expect("invocation allocation identity space exhausted");
                allocation_identities.entry(tensor.resource).or_insert(next);
            }
        }
        let values = arguments
            .iter()
            .map(|argument| {
                descriptor_argument(
                    argument,
                    self.kernel.prepared.device.identity(),
                    &allocation_identities,
                )
            })
            .collect::<Vec<_>>();
        let invocation = validate_invocation(
            self.kernel.prepared.kernel.schema(),
            self.kernel.prepared.kernel.invocation_contract(),
            self.kernel.prepared.device.identity(),
            &values,
        )
        .map_err(CallError::Invocation)?;
        let variant = self.kernel.prepared.kernel.select(&invocation).as_usize();
        let executable = &self.kernel.prepared.kernel.variants().as_slice()[variant];

        let argument_access = self
            .kernel
            .prepared
            .kernel
            .schema()
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
                        argument: self
                            .kernel
                            .prepared
                            .kernel
                            .schema()
                            .parameter_ordinal(*parameter),
                    });
                }
                _ => {
                    let slot = private_allocations.len();
                    let bytes = evaluated_bytes(allocation, &invocation);
                    let kind = match &allocation.kind {
                        ExecutableAllocationKind::Global(
                            ExecutableGlobalAllocationKind::Persistent,
                        ) => AllocationKind::Persistent {
                            owner: self.kernel.prepared.identity,
                            variant: u32::try_from(variant)
                                .expect("prepared variant ordinal space exhausted"),
                            slot: u32::try_from(index)
                                .expect("executable allocation ordinal space exhausted"),
                        },
                        _ => AllocationKind::Temporary,
                    };
                    private_allocations.push(AllocationDescription {
                        kind,
                        bytes,
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
                ExecutableResultBinding::Buffer { view, bytes } => {
                    let range = ByteRange {
                        offset: admitted(view.byte_offset.evaluate(&invocation)),
                        len: admitted(bytes.evaluate(&invocation)),
                    };
                    let storage = match allocations
                        .get(view.allocation_index())
                        .expect("result binding references an unknown executable allocation")
                    {
                        PlannedAllocation::Argument { argument } => TensorStorage::Alias {
                            argument: u32::try_from(*argument)
                                .expect("argument ordinal space exhausted"),
                            range,
                        },
                        PlannedAllocation::Private { slot, .. } => TensorStorage::Private {
                            allocation: u32::try_from(*slot)
                                .expect("private allocation ordinal space exhausted"),
                            range,
                        },
                    };
                    OutputDescription::Tensor {
                        representation: view.representation,
                        extents: view
                            .extents
                            .iter()
                            .map(|value| admitted(value.evaluate(&invocation)))
                            .collect(),
                        strides: view
                            .strides
                            .iter()
                            .map(|value| admitted(value.evaluate(&invocation)))
                            .collect(),
                        storage,
                    }
                }
                ExecutableResultBinding::Scalar { .. } | ExecutableResultBinding::Range { .. } => {
                    OutputDescription::Scalar(ScalarDescriptor::DeviceProduced)
                }
            });
        }

        Ok(EvaluatedNode {
            selection: PlannedNode {
                kernel: self.kernel.clone(),
                values: invocation,
                variant,
                allocations,
            },
            argument_access,
            outputs,
            private_allocations,
        })
    }
}

fn descriptor_argument(
    descriptor: &crate::workflow::ValueDescriptor,
    device: DeviceIdentity,
    allocation_identities: &BTreeMap<ResourceId, u64>,
) -> ArgumentValue {
    match descriptor {
        crate::workflow::ValueDescriptor::Tensor(tensor) => {
            ArgumentValue::Tensor(seismic_compiler::prepared::TensorDescriptor {
                device,
                representation: tensor.representation,
                extents: tensor.extents.clone(),
                strides: tensor.strides.clone(),
                allocation: allocation_identities[&tensor.resource],
                byte_offset: tensor.range.offset,
                byte_len: tensor.range.len,
            })
        }
        crate::workflow::ValueDescriptor::Scalar(ScalarDescriptor::HostReady(value)) => {
            scalar_argument(*value)
        }
        crate::workflow::ValueDescriptor::Scalar(ScalarDescriptor::DeviceProduced) => {
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
) -> u64 {
    let candidates = allocation.byte_candidates();
    candidates.rest().iter().fold(
        admitted(candidates.first().evaluate(values)),
        |required, bytes| required.max(admitted(bytes.evaluate(values))),
    )
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

/// A graph whose policy choices, output descriptors, hazards, lifetimes, and
/// allocation requirements have all been evaluated. Binding is pure with
/// respect to device resource state; admission is the only transition that
/// acquires native resources.
pub(crate) struct BoundWorkflowGraph<T: TargetFamily, E: NativeExecutor<T>> {
    plan: BoundWorkflow<PlannedNode<T, E>>,
    physical: BTreeMap<ResourceId, Arc<Allocation>>,
    device: Arc<Opened<T, E>>,
}

struct PersistentIntent {
    table: Arc<PersistentTable>,
    key: (usize, usize),
    bytes: u64,
}

enum PersistentDecision {
    Reuse(PersistentBinding),
    Grow { old: Option<PersistentBinding> },
}

struct PendingGrowth {
    claim: PersistentGrowth,
    binding: PersistentBinding,
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
        for queued in self.nodes {
            let mut bindings = Vec::new();
            for argument in queued.args.into_arguments() {
                bindings.push(match argument {
                    EncodedWorkflowArgument::Tensor(WorkflowTensorArgument::External(tensor)) => {
                        let descriptor = tensor_descriptor(&tensor);
                        physical
                            .entry(descriptor.resource)
                            .or_insert_with(|| tensor.allocation().clone());
                        ArgumentBinding::External(crate::workflow::ValueDescriptor::Tensor(
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
                        ArgumentBinding::External(crate::workflow::ValueDescriptor::Scalar(
                            ScalarDescriptor::HostReady(value),
                        ))
                    }
                    EncodedWorkflowArgument::ScalarResult(reference) => {
                        ArgumentBinding::Result(output_ref(reference))
                    }
                });
            }
            planner
                .push(
                    WorkflowPolicy {
                        kernel: queued.kernel,
                    },
                    bindings,
                )
                .map_err(plan_error)?;
        }
        let plan = planner.close().map_err(plan_error)?;
        Ok(BoundWorkflowGraph {
            plan,
            physical,
            device: self.device,
        })
    }
}

impl<T: TargetFamily, E: NativeExecutor<T>> BoundWorkflowGraph<T, E> {
    pub(crate) fn admit(self) -> Result<AdmittedRun<T, E>, CallError> {
        admit_plan(self.plan, self.physical, self.device)
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
    let allocated = admitted.allocated_bytes();
    let mut values = admitted.submit()?.complete_values()?;
    let values = values
        .pop()
        .expect("one-node workflow completed without its node");
    Ok((DecodedResults::new(values), allocated))
}

fn output_ref(reference: crate::api::kernel::WorkflowResultRef) -> OutputRef {
    OutputRef::from_raw(reference.workflow, reference.node, reference.result)
}

fn tensor_descriptor(tensor: &TensorInner) -> TensorDescriptor {
    TensorDescriptor {
        resource: ResourceId::External(tensor.allocation().identity()),
        representation: tensor.representation(),
        extents: tensor.extents().to_vec(),
        strides: tensor.strides().to_vec(),
        range: ByteRange {
            offset: tensor.byte_offset(),
            len: tensor.byte_len(),
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

fn admit_plan<T: TargetFamily, E: NativeExecutor<T>>(
    plan: BoundWorkflow<PlannedNode<T, E>>,
    mut physical: BTreeMap<ResourceId, Arc<Allocation>>,
    device: Arc<Opened<T, E>>,
) -> Result<AdmittedRun<T, E>, CallError> {
    let (identity, nodes, _, requirements) = plan.into_parts();
    let mut span = Timed::start("seismic.workflow.admit", Vec::new());
    let persistent_intents = persistent_intents(&nodes, &requirements);

    // The retry boundary owns no resource. We inspect all persistent slots
    // under the device admission guard. If any slot requires a wait, the guard
    // is released and the complete graph is re-snapshotted after notification.
    let (decisions, admission_guard) = loop {
        let guard = device.admission.enter();
        let decisions = persistent_intents
            .iter()
            .map(|(resource, intent)| {
                let decision = match intent.table.availability(intent.key, intent.bytes) {
                    PersistentAvailability::Reuse(binding) => PersistentDecision::Reuse(binding),
                    PersistentAvailability::Grow { old } => PersistentDecision::Grow { old },
                    PersistentAvailability::Wait => return Err(intent),
                };
                Ok((*resource, decision))
            })
            .collect::<Result<BTreeMap<_, _>, _>>();
        match decisions {
            Ok(decisions) => break (decisions, guard),
            Err(wait) => {
                let table = wait.table.clone();
                let key = wait.key;
                let bytes = wait.bytes;
                drop(guard);
                table.wait_until_available(key, bytes);
            }
        }
    };

    // Acquire every already-existing allocation in global identity order.
    // No reservation or persistent claim exists yet, so an access wait cannot
    // participate in a hold-and-wait cycle inside admission.
    let mut access_requests: BTreeMap<u64, (Arc<Allocation>, bool, bool)> = BTreeMap::new();
    let mut access_by_resource: BTreeMap<ResourceId, bool> = BTreeMap::new();
    for node in &nodes {
        for access in &node.accesses {
            let write = !matches!(access.mode, AccessMode::Read);
            access_by_resource
                .entry(access.resource)
                .and_modify(|known| *known |= write)
                .or_insert(write);
        }
    }
    for (resource, write) in &access_by_resource {
        if let Some(allocation) = physical.get(resource) {
            merge_access_request(&mut access_requests, allocation.clone(), *write, true);
        }
    }
    for (resource, decision) in &decisions {
        match decision {
            PersistentDecision::Reuse(binding) => merge_access_request(
                &mut access_requests,
                binding.allocation.clone(),
                *access_by_resource.get(resource).unwrap_or(&true),
                true,
            ),
            PersistentDecision::Grow { old: Some(old) } => {
                merge_access_request(&mut access_requests, old.allocation.clone(), false, false)
            }
            PersistentDecision::Grow { old: None } => {}
        }
    }
    let mut acquired = access_requests
        .into_values()
        .map(|(allocation, write, retain)| (allocation.acquire(write), retain))
        .collect::<Vec<_>>();

    // Persistent claims cannot wait: the admission guard excludes competing
    // admissions, and execution can only release leases after the snapshot.
    let mut persistent_leases = Vec::new();
    let mut growth_claims = BTreeMap::new();
    for (resource, decision) in &decisions {
        let intent = &persistent_intents[resource];
        match decision {
            PersistentDecision::Reuse(binding) => {
                persistent_leases.push(intent.table.claim_reuse(intent.key, &binding.allocation));
                physical.insert(*resource, binding.allocation.clone());
            }
            PersistentDecision::Grow { .. } => {
                growth_claims.insert(*resource, intent.table.claim_growth(intent.key));
            }
        }
    }

    let required = requirements
        .iter()
        .try_fold(0u64, |total, (resource, requirement)| {
            let needs_allocation = match decisions.get(resource) {
                Some(PersistentDecision::Reuse(_)) => false,
                Some(PersistentDecision::Grow { .. }) => true,
                None => !physical.contains_key(resource),
            };
            total.checked_add(if needs_allocation {
                requirement.bytes
            } else {
                0
            })
        })
        .ok_or_else(|| {
            CallError::Execution(ExecutionError::AllocationFailed(
                "bound workflow allocation total overflowed u64".to_owned(),
            ))
        })?;
    let mut reservation = device.memory.reserve(required).map_err(|capacity| {
        CallError::Invocation(InvocationError::AllocationCapacity {
            required: capacity.required,
            available: capacity.available,
        })
    })?;

    let mut newly_allocated = BTreeMap::new();
    let mut pending_growth = Vec::new();
    for (resource, requirement) in &requirements {
        if physical.contains_key(resource) {
            continue;
        }
        let allocation = device
            .allocate_reserved(requirement.bytes, requirement.alignment, &mut reservation)
            .map_err(CallError::Execution)?;
        newly_allocated.insert(*resource, requirement.bytes);

        if let Some(claim) = growth_claims.remove(resource) {
            if let Some(old) = claim.old() {
                copy_between::<T, E>(
                    &*device.service,
                    &typed_buffer::<T, E>(&old.allocation),
                    &typed_buffer::<T, E>(&allocation),
                    old.capacity,
                )
                .map_err(CallError::Execution)?;
            }
            pending_growth.push(PendingGrowth {
                claim,
                binding: PersistentBinding {
                    allocation: allocation.clone(),
                    capacity: requirement.bytes,
                },
            });
        }
        physical.insert(*resource, allocation);
    }
    assert!(
        growth_claims.is_empty(),
        "persistent growth lacked an authoritative workflow requirement"
    );

    // Fresh allocations cannot contend because they have not yet been
    // published. Retain their graph access permits through completion.
    for (resource, write) in &access_by_resource {
        let Some(allocation) = physical.get(resource) else {
            return Err(CallError::Execution(ExecutionError::AllocationFailed(
                "bound workflow access has no physical allocation".to_owned(),
            )));
        };
        if newly_allocated.contains_key(resource) {
            let permit = allocation.try_acquire(*write).unwrap_or_else(|| {
                panic!("unpublished workflow allocation unexpectedly had an access owner")
            });
            acquired.push((permit, true));
        }
    }

    let mut allocated_by_node = vec![0u64; nodes.len()];
    for (resource, bytes) in &newly_allocated {
        let first = requirements[resource].lifetime.first as usize;
        allocated_by_node[first] = allocated_by_node[first].saturating_add(*bytes);
    }
    let mut staged_nodes = Vec::with_capacity(nodes.len());
    for (node_index, node) in nodes.into_iter().enumerate() {
        let kernel = node.selection.kernel.clone();
        let (staged, outputs) = stage_planned(
            &kernel.prepared,
            node,
            &requirements,
            &physical,
            &mut span,
            allocated_by_node[node_index],
        )
        .map_err(CallError::Execution)?;
        staged_nodes.push(AdmittedNode::new(
            AdmittedCommand::new(kernel, staged),
            outputs,
        ));
    }
    let submission = device.begin_submission().map_err(CallError::Execution)?;

    // Every fallible operation has completed. Publishing persistent growths
    // while still holding the admission guard is the transaction's
    // linearization point; other admissions can observe only the state before
    // this block or the complete committed graph state after the guard drops.
    for pending in pending_growth {
        persistent_leases.push(pending.claim.commit(pending.binding));
    }
    let access = acquired
        .into_iter()
        .filter_map(|(permit, retain)| retain.then_some(permit))
        .collect();
    drop(admission_guard);

    let execution_device = device.service_arc();
    Ok(AdmittedRun::new(
        identity.raw(),
        execution_device,
        staged_nodes,
        AdmittedResources::new(reservation, persistent_leases, access),
        submission,
    ))
}

fn persistent_intents<T: TargetFamily, E: NativeExecutor<T>>(
    nodes: &[BoundNode<PlannedNode<T, E>>],
    requirements: &BTreeMap<ResourceId, crate::workflow::ResourceRequirement>,
) -> BTreeMap<ResourceId, PersistentIntent> {
    let mut intents = BTreeMap::new();
    for node in nodes {
        for allocation in &node.selection.allocations {
            let PlannedAllocation::Private { slot, executable } = allocation else {
                continue;
            };
            let local = &node.private_resources[*slot];
            let requirement = requirements.get(&local.resource).unwrap_or(local);
            let ResourceId::Persistent {
                owner,
                variant,
                slot: persistent_slot,
            } = requirement.resource
            else {
                continue;
            };
            assert_eq!(owner, node.selection.kernel.prepared.identity);
            assert_eq!(variant as usize, node.selection.variant);
            assert_eq!(persistent_slot as usize, *executable);
            intents
                .entry(requirement.resource)
                .or_insert_with(|| PersistentIntent {
                    table: node.selection.kernel.prepared.persistent.clone(),
                    key: (node.selection.variant, *executable),
                    bytes: requirement.bytes,
                });
        }
    }
    intents
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
    prepared: &Prepared<T, E>,
    node: BoundNode<PlannedNode<T, E>>,
    requirements: &BTreeMap<ResourceId, crate::workflow::ResourceRequirement>,
    physical: &BTreeMap<ResourceId, Arc<Allocation>>,
    span: &mut Timed,
    allocated_bytes: u64,
) -> Result<(Staged<T, E>, Vec<AdmittedOutput>), ExecutionError> {
    let PlannedNode {
        values,
        variant,
        allocations: planned,
        ..
    } = node.selection;
    let executable = &prepared.kernel.variants().as_slice()[variant];
    span.attribute(key_str(
        "seismic.variant.factory",
        executable.identity().implementation.factory.name,
    ));
    let mut buffers = Vec::with_capacity(planned.len());
    let mut allocations = Vec::with_capacity(planned.len());
    for allocation in planned {
        match allocation {
            PlannedAllocation::Argument { argument } => {
                let crate::workflow::ValueDescriptor::Tensor(tensor) = &node.arguments[argument]
                else {
                    panic!("planned tensor allocation references a scalar argument")
                };
                let allocation = physical
                    .get(&tensor.resource)
                    .cloned()
                    .expect("bound external/producer allocation is absent");
                buffers.push(RuntimeBuffer {
                    buffer: typed_buffer::<T, E>(&allocation),
                    base_offset: tensor.range.offset,
                    accessible_bytes: tensor.range.len,
                });
                allocations.push(allocation);
            }
            PlannedAllocation::Private {
                slot,
                executable: _,
            } => {
                let local = &node.private_resources[slot];
                let requirement = requirements.get(&local.resource).unwrap_or(local);
                let allocation = physical
                    .get(&requirement.resource)
                    .cloned()
                    .ok_or_else(|| {
                        ExecutionError::AllocationFailed(
                            "admitted private resource has no physical allocation".to_owned(),
                        )
                    })?;
                buffers.push(RuntimeBuffer {
                    buffer: typed_buffer::<T, E>(&allocation),
                    base_offset: 0,
                    accessible_bytes: requirement.bytes,
                });
                allocations.push(allocation);
            }
        }
    }
    let outputs = close_admitted_outputs::<T, E>(&node.outputs, executable, physical)?;
    span.attribute(key_u64("seismic.allocated_bytes", allocated_bytes));
    let staged = Staged {
        variant,
        values,
        buffers,
        _allocations: allocations,
        allocated_bytes,
    };
    Ok((staged, outputs))
}

fn close_admitted_outputs<T: TargetFamily, E: NativeExecutor<T>>(
    descriptors: &[crate::workflow::ValueDescriptor],
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
        .map(|(descriptor, (_, binding))| match (descriptor, binding) {
            (
                crate::workflow::ValueDescriptor::Tensor(tensor),
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
                crate::workflow::ValueDescriptor::Scalar(ScalarDescriptor::DeviceProduced),
                ExecutableResultBinding::Scalar {
                    slot,
                    kind: ExecutableScalarResultKind::Value(dtype),
                },
            ) => Ok(AdmittedOutput::Scalar(AdmittedScalarOutput::Value {
                dtype: *dtype,
                symbol: slot.symbol(),
            })),
            (
                crate::workflow::ValueDescriptor::Scalar(ScalarDescriptor::DeviceProduced),
                ExecutableResultBinding::Scalar {
                    slot,
                    kind: ExecutableScalarResultKind::Index,
                },
            ) => Ok(AdmittedOutput::Scalar(AdmittedScalarOutput::Index {
                symbol: slot.symbol(),
            })),
            (
                crate::workflow::ValueDescriptor::Scalar(ScalarDescriptor::DeviceProduced),
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
