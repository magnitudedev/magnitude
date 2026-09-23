//! Checked direct-native graph construction and reusable result storage.
//!
//! A graph is sealed from the generated entry contracts before resident model
//! storage is admitted. Ports are the only late bindings. No consumer supplies
//! a name-to-buffer recipe for intermediate results.

#![cfg(target_os = "macos")]

use crate::api::device::DeviceInner;
use crate::api::kernel::{
    EncodedArgs, EncodedWorkflowArgs, EncodedWorkflowArgument, NativePreparedAny,
    PendingWorkflowResults, ViewOperation, WorkflowResultRef, WorkflowTensorArgument,
};
use crate::api::tensor::TensorInner;
use crate::api::{CallError, TensorError, WorkflowError};
use crate::backends::NativeBoundKind;
use crate::driver::{Allocation, NativeTensorSpec, write_zeros};
use crate::layout;
use seismic_compiler::errors::ExecutionError;
use seismic_compiler::prepared::{ArgumentValue, DeviceIdentity, TensorDescriptor};
use seismic_lang::ids::RepresentationId;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};

static NEXT_NATIVE_GRAPH: AtomicU64 = AtomicU64::new(1);
const INPUT_NODE: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NativePort {
    reference: WorkflowResultRef,
}

impl NativePort {
    pub fn reference(&self) -> WorkflowResultRef {
        self.reference
    }
}

#[derive(Clone)]
struct PortSpec {
    representation: RepresentationId,
    extents: Vec<u64>,
    strides: Vec<u64>,
    byte_len: u64,
    alignment: u64,
    local: bool,
    owned_input: bool,
    prewritten: bool,
}

struct NodeDraft {
    kernel: Arc<NativePreparedAny>,
    args: EncodedWorkflowArgs,
}

pub struct NativeGraphDraft {
    identity: u64,
    device: Arc<DeviceInner>,
    ports: Vec<PortSpec>,
    nodes: Vec<NodeDraft>,
    exports: Vec<WorkflowResultRef>,
}

impl NativeGraphDraft {
    pub fn new(device: &Arc<DeviceInner>) -> Self {
        Self {
            identity: NEXT_NATIVE_GRAPH.fetch_add(1, Ordering::Relaxed),
            device: device.clone(),
            ports: Vec::new(),
            nodes: Vec::new(),
            exports: Vec::new(),
        }
    }

    /// Declare one externally owned tensor from artifact/model dimensions.
    /// Every use is checked against the generated entry that consumes it.
    pub fn port(
        &mut self,
        representation: RepresentationId,
        extents: &[u64],
    ) -> Result<NativePort, TensorError> {
        let layout = layout::canonical(representation, extents)?;
        let ordinal = u32::try_from(self.ports.len()).expect("native port ordinal exhausted");
        self.ports.push(PortSpec {
            representation,
            extents: extents.to_vec(),
            strides: layout.strides,
            byte_len: layout.byte_len,
            alignment: layout.alignment,
            local: false,
            owned_input: false,
            prewritten: false,
        });
        Ok(NativePort {
            reference: WorkflowResultRef {
                workflow: self.identity,
                node: INPUT_NODE,
                result: ordinal,
            },
        })
    }

    /// Allocate a graph-local mutable tensor from a checked parameter. The
    /// caller supplies only dimension values; Seismic derives representation,
    /// extents, strides, and bytes from the entry contract.
    pub fn local_for(
        &mut self,
        kernel: &Arc<NativePreparedAny>,
        parameter: &str,
        dimensions: &[(&str, u64)],
    ) -> Result<NativePort, CallError> {
        let spec = kernel.inner.tensor_parameter_spec(parameter, dimensions)?;
        let ordinal = u32::try_from(self.ports.len()).expect("native port ordinal exhausted");
        self.ports.push(PortSpec {
            representation: spec.representation,
            extents: spec.extents,
            strides: spec.strides,
            byte_len: spec.byte_len,
            alignment: spec.alignment,
            local: true,
            owned_input: false,
            prewritten: false,
        });
        Ok(NativePort {
            reference: WorkflowResultRef {
                workflow: self.identity,
                node: INPUT_NODE,
                result: ordinal,
            },
        })
    }

    pub fn input_for(
        &mut self,
        kernel: &Arc<NativePreparedAny>,
        parameter: &str,
        dimensions: &[(&str, u64)],
    ) -> Result<NativePort, CallError> {
        let spec = kernel.inner.tensor_parameter_spec(parameter, dimensions)?;
        let ordinal = u32::try_from(self.ports.len()).expect("native port ordinal exhausted");
        self.ports.push(PortSpec {
            representation: spec.representation,
            extents: spec.extents,
            strides: spec.strides,
            byte_len: spec.byte_len,
            alignment: spec.alignment,
            local: true,
            owned_input: true,
            prewritten: false,
        });
        Ok(NativePort {
            reference: WorkflowResultRef {
                workflow: self.identity,
                node: INPUT_NODE,
                result: ordinal,
            },
        })
    }

    /// Declare that an adjacent graph fills this checked local before the
    /// first node executes. Its storage is then live from graph start.
    pub fn prewrite(&mut self, port: NativePort) -> Result<(), WorkflowError> {
        let ordinal = port.reference.result as usize;
        if port.reference.workflow != self.identity || port.reference.node != INPUT_NODE {
            return Err(WorkflowError::CrossWorkflowResult);
        }
        let spec = self
            .ports
            .get_mut(ordinal)
            .ok_or(WorkflowError::CrossWorkflowResult)?;
        if !spec.local || spec.owned_input {
            return Err(WorkflowError::NativePortMismatch { port: ordinal });
        }
        spec.prewritten = true;
        Ok(())
    }

    pub fn enqueue(
        &mut self,
        kernel: &Arc<NativePreparedAny>,
        args: EncodedWorkflowArgs,
    ) -> Result<PendingWorkflowResults, WorkflowError> {
        let node = u32::try_from(self.nodes.len()).expect("native graph node ordinal exhausted");
        for argument in args.arguments() {
            let reference = argument_reference(argument);
            if reference.is_some_and(|reference| {
                reference.workflow != self.identity
                    || (reference.node != INPUT_NODE && reference.node >= node)
                    || (reference.node == INPUT_NODE
                        && reference.result as usize >= self.ports.len())
            }) {
                return Err(WorkflowError::CrossWorkflowResult);
            }
        }
        // The checked result count is supplied by the native handle; no
        // caller-authored result list is allowed here.
        let count = kernel.inner.result_count();
        self.nodes.push(NodeDraft {
            kernel: kernel.clone(),
            args,
        });
        Ok(PendingWorkflowResults::new(self.identity, node, count))
    }

    pub fn export(&mut self, reference: WorkflowResultRef) -> Result<(), WorkflowError> {
        if reference.workflow != self.identity
            || (reference.node == INPUT_NODE
                && !self
                    .ports
                    .get(reference.result as usize)
                    .is_some_and(|port| port.local))
            || (reference.node != INPUT_NODE && reference.node as usize >= self.nodes.len())
        {
            return Err(WorkflowError::CrossWorkflowResult);
        }
        if !self.exports.contains(&reference) {
            self.exports.push(reference);
        }
        Ok(())
    }

    pub fn seal(self) -> Result<NativeGraphPlan, CallError> {
        if self.nodes.is_empty() {
            return Err(CallError::Workflow(WorkflowError::Empty));
        }
        let mut used_ports = vec![false; self.ports.len()];
        for node in &self.nodes {
            for argument in node.args.arguments() {
                if let Some(reference) = argument_reference(argument) {
                    if reference.node == INPUT_NODE {
                        used_ports[reference.result as usize] = true;
                    }
                }
            }
        }
        for reference in &self.exports {
            if reference.node == INPUT_NODE {
                used_ports[reference.result as usize] = true;
            }
        }
        if let Some(port) = used_ports.iter().position(|used| !used) {
            return Err(CallError::Workflow(WorkflowError::NativePortUnbound {
                port,
            }));
        }
        let identity = self.identity;
        let device_identity = self.device.kind.identity();
        let mut results: Vec<Vec<Option<NativeTensorSpec>>> = Vec::new();
        let mut nodes = Vec::with_capacity(self.nodes.len());
        for node in self.nodes {
            node.kernel.inner.validate_graph_batch(device_identity)?;
            let arguments =
                describe_arguments(&node.args, &self.ports, &results, identity, device_identity)?;
            let described = node.kernel.inner.describe_results(&arguments)?;
            results.push(described);
            nodes.push(PlannedNode {
                kernel: node.kernel,
                args: node.args,
            });
        }
        for reference in &self.exports {
            if reference.node != INPUT_NODE && result_spec(&results, *reference).is_none() {
                return Err(CallError::Workflow(WorkflowError::MissingProducerResult));
            }
        }
        let (
            port_placements,
            placements,
            scratch_bytes,
            scratch_alignment,
            output_bytes,
            output_alignment,
        ) = plan_storage(&self.ports, &nodes, &results, &self.exports);
        Ok(NativeGraphPlan {
            identity,
            device: self.device,
            ports: self.ports,
            nodes,
            results,
            exports: self.exports,
            port_placements,
            placements,
            scratch_bytes,
            scratch_alignment,
            output_bytes,
            output_alignment,
        })
    }
}

fn argument_reference(argument: &EncodedWorkflowArgument) -> Option<WorkflowResultRef> {
    match argument {
        EncodedWorkflowArgument::Tensor(WorkflowTensorArgument::Result(reference))
        | EncodedWorkflowArgument::ScalarResult(reference) => Some(*reference),
        EncodedWorkflowArgument::Tensor(WorkflowTensorArgument::ResultView { result, .. }) => {
            Some(*result)
        }
        _ => None,
    }
}

fn result_spec(
    results: &[Vec<Option<NativeTensorSpec>>],
    reference: WorkflowResultRef,
) -> Option<&NativeTensorSpec> {
    results
        .get(reference.node as usize)?
        .get(reference.result as usize)?
        .as_ref()
}

fn virtual_descriptor(
    spec: &NativeTensorSpec,
    device: DeviceIdentity,
    allocation: u64,
) -> TensorDescriptor {
    TensorDescriptor {
        device,
        representation: spec.representation,
        extents: spec.extents.clone(),
        strides: spec.strides.clone(),
        allocation,
        byte_offset: 0,
        byte_len: spec.byte_len,
    }
}

fn port_descriptor(spec: &PortSpec, device: DeviceIdentity, ordinal: usize) -> TensorDescriptor {
    TensorDescriptor {
        device,
        representation: spec.representation,
        extents: spec.extents.clone(),
        strides: spec.strides.clone(),
        allocation: u64::MAX - ordinal as u64,
        byte_offset: 0,
        byte_len: spec.byte_len,
    }
}

fn describe_arguments(
    args: &EncodedWorkflowArgs,
    ports: &[PortSpec],
    results: &[Vec<Option<NativeTensorSpec>>],
    identity: u64,
    device: DeviceIdentity,
) -> Result<Vec<ArgumentValue>, CallError> {
    args.arguments()
        .iter()
        .map(|argument| match argument {
            EncodedWorkflowArgument::Tensor(WorkflowTensorArgument::External(_)) => {
                Err(CallError::Workflow(WorkflowError::NativePortMismatch {
                    port: usize::MAX,
                }))
            }
            EncodedWorkflowArgument::Tensor(WorkflowTensorArgument::Result(reference)) => {
                describe_reference(*reference, ports, results, identity, device)
                    .map(ArgumentValue::Tensor)
            }
            EncodedWorkflowArgument::Tensor(WorkflowTensorArgument::ResultView {
                result,
                operations,
            }) => {
                let mut descriptor = describe_reference(*result, ports, results, identity, device)?;
                for operation in operations {
                    descriptor = apply_view_descriptor(descriptor, operation)?;
                }
                Ok(ArgumentValue::Tensor(descriptor))
            }
            EncodedWorkflowArgument::Scalar(value) => Ok(value.value()),
            EncodedWorkflowArgument::ScalarResult(_) => {
                Err(CallError::Workflow(WorkflowError::HostBoundaryRequired))
            }
        })
        .collect()
}

fn apply_view_descriptor(
    descriptor: TensorDescriptor,
    operation: &ViewOperation,
) -> Result<TensorDescriptor, CallError> {
    let view = layout::apply_view(
        descriptor.representation,
        layout::ViewGeometry {
            extents: descriptor.extents,
            strides: descriptor.strides,
            byte_offset: descriptor.byte_offset,
            byte_len: descriptor.byte_len,
        },
        operation,
    )
    .map_err(|error| CallError::Workflow(WorkflowError::TensorView(error)))?;
    Ok(TensorDescriptor {
        extents: view.extents,
        strides: view.strides,
        byte_offset: view.byte_offset,
        byte_len: view.byte_len,
        ..descriptor
    })
}

fn describe_reference(
    reference: WorkflowResultRef,
    ports: &[PortSpec],
    results: &[Vec<Option<NativeTensorSpec>>],
    identity: u64,
    device: DeviceIdentity,
) -> Result<TensorDescriptor, CallError> {
    if reference.workflow != identity {
        return Err(CallError::Workflow(WorkflowError::CrossWorkflowResult));
    }
    if reference.node == INPUT_NODE {
        let ordinal = reference.result as usize;
        let spec = ports
            .get(ordinal)
            .ok_or(CallError::Workflow(WorkflowError::MissingProducerResult))?;
        return Ok(port_descriptor(spec, device, ordinal));
    }
    let spec = result_spec(results, reference)
        .ok_or(CallError::Workflow(WorkflowError::MissingProducerResult))?;
    let allocation = ((reference.node as u64) << 32) | reference.result as u64;
    Ok(virtual_descriptor(spec, device, allocation))
}

struct PlannedNode {
    kernel: Arc<NativePreparedAny>,
    args: EncodedWorkflowArgs,
}

#[derive(Clone, Copy)]
enum Placement {
    Scratch(u64),
    Export(u64),
    Scalar,
}

struct ScratchBlock {
    offset: u64,
    capacity: u64,
    live_until: usize,
}

#[derive(Clone, Copy)]
enum StorageKey {
    Port(usize),
    Result(usize, usize),
}

struct Interval {
    key: StorageKey,
    start: usize,
    end: usize,
    bytes: u64,
    alignment: u64,
    exported: bool,
}

fn align_up(value: u64, alignment: u64) -> u64 {
    value.div_ceil(alignment) * alignment
}

fn plan_storage(
    ports: &[PortSpec],
    nodes: &[PlannedNode],
    results: &[Vec<Option<NativeTensorSpec>>],
    exports: &[WorkflowResultRef],
) -> (
    Vec<Option<Placement>>,
    Vec<Vec<Placement>>,
    u64,
    u64,
    u64,
    u64,
) {
    let mut last_use = results
        .iter()
        .enumerate()
        .map(|(node, row)| vec![node * 2 + 1; row.len()])
        .collect::<Vec<_>>();
    let mut port_first = vec![None; ports.len()];
    let mut port_last = vec![0usize; ports.len()];
    for (consumer, node) in nodes.iter().enumerate() {
        for argument in node.args.arguments() {
            if let Some(reference) = argument_reference(argument) {
                if reference.node == INPUT_NODE {
                    let ordinal = reference.result as usize;
                    if ports[ordinal].local {
                        port_first[ordinal].get_or_insert(consumer * 2);
                        port_last[ordinal] = consumer * 2 + 1;
                    }
                } else {
                    last_use[reference.node as usize][reference.result as usize] = consumer * 2 + 1;
                }
            }
        }
    }
    let mut intervals = Vec::new();
    for (ordinal, port) in ports.iter().enumerate() {
        if !port.local {
            continue;
        }
        let exported = exports
            .iter()
            .any(|item| item.node == INPUT_NODE && item.result as usize == ordinal);
        intervals.push(Interval {
            key: StorageKey::Port(ordinal),
            start: if port.owned_input || port.prewritten {
                0
            } else {
                port_first[ordinal].unwrap_or(0)
            },
            end: port_last[ordinal],
            bytes: port.byte_len.max(1),
            alignment: port.alignment,
            exported,
        });
    }
    for (node, row) in results.iter().enumerate() {
        for (ordinal, result) in row.iter().enumerate() {
            let Some(spec) = result else { continue };
            let exported = exports
                .iter()
                .any(|item| item.node as usize == node && item.result as usize == ordinal);
            intervals.push(Interval {
                key: StorageKey::Result(node, ordinal),
                start: node * 2 + 1,
                end: last_use[node][ordinal],
                bytes: spec.byte_len.max(1),
                alignment: spec.alignment,
                exported,
            });
        }
    }
    intervals.sort_by_key(|interval| interval.start);
    let mut blocks: Vec<ScratchBlock> = Vec::new();
    let mut cursor = 0u64;
    let mut alignment = 1u64;
    let mut output_bytes = 0u64;
    let mut output_alignment = 1u64;
    let mut port_placements = vec![None; ports.len()];
    let mut placements = results
        .iter()
        .map(|row| vec![Placement::Scalar; row.len()])
        .collect::<Vec<_>>();
    for interval in intervals {
        let placement = if interval.exported {
            output_alignment = output_alignment.max(interval.alignment);
            let offset = align_up(output_bytes, interval.alignment);
            output_bytes = offset
                .checked_add(interval.bytes)
                .expect("native graph output bytes overflow");
            Placement::Export(offset)
        } else {
            alignment = alignment.max(interval.alignment);
            let reuse = blocks.iter_mut().find(|block| {
                block.live_until < interval.start
                    && block.capacity >= interval.bytes
                    && block.offset % interval.alignment == 0
            });
            let offset = if let Some(block) = reuse {
                block.live_until = interval.end;
                block.offset
            } else {
                let offset = align_up(cursor, interval.alignment);
                cursor = offset
                    .checked_add(interval.bytes)
                    .expect("native graph storage overflow");
                blocks.push(ScratchBlock {
                    offset,
                    capacity: interval.bytes,
                    live_until: interval.end,
                });
                offset
            };
            Placement::Scratch(offset)
        };
        match interval.key {
            StorageKey::Port(ordinal) => port_placements[ordinal] = Some(placement),
            StorageKey::Result(node, ordinal) => placements[node][ordinal] = placement,
        }
    }
    (
        port_placements,
        placements,
        cursor,
        alignment,
        output_bytes,
        output_alignment,
    )
}

pub struct NativeGraphPlan {
    identity: u64,
    device: Arc<DeviceInner>,
    ports: Vec<PortSpec>,
    nodes: Vec<PlannedNode>,
    results: Vec<Vec<Option<NativeTensorSpec>>>,
    exports: Vec<WorkflowResultRef>,
    port_placements: Vec<Option<Placement>>,
    placements: Vec<Vec<Placement>>,
    scratch_bytes: u64,
    scratch_alignment: u64,
    output_bytes: u64,
    output_alignment: u64,
}

impl NativeGraphPlan {
    pub fn workspace_bytes(&self) -> u64 {
        self.scratch_bytes
    }
    pub fn output_bytes(&self) -> u64 {
        self.output_bytes
    }
    pub fn slot_storage_bytes(&self) -> u64 {
        self.scratch_bytes + self.output_bytes
    }
    pub fn bindings(&self) -> NativeGraphBindings {
        NativeGraphBindings {
            identity: self.identity,
            inputs: vec![None; self.ports.len()],
            external: self.ports.iter().map(|port| !port.local).collect(),
        }
    }
    pub fn bind_static(
        self: &Arc<Self>,
        fixed: &[(NativePort, Arc<TensorInner>)],
    ) -> Result<BoundNativeGraphPlan, CallError> {
        let mut bindings = self.bindings();
        for (port, tensor) in fixed {
            bindings
                .set(*port, tensor.clone())
                .map_err(CallError::Workflow)?;
            let ordinal = port.reference.result as usize;
            if !port_matches(&self.device, &self.ports[ordinal], tensor) {
                return Err(CallError::Workflow(WorkflowError::NativePortMismatch {
                    port: ordinal,
                }));
            }
        }
        Ok(BoundNativeGraphPlan {
            _plan: self.clone(),
            fixed: bindings,
        })
    }
    pub fn new_slot(self: &Arc<Self>) -> Result<NativeGraphSlot, TensorError> {
        let scratch = if self.scratch_bytes == 0 {
            None
        } else {
            let allocation = self
                .device
                .allocate(self.scratch_bytes, self.scratch_alignment)?;
            write_zeros(allocation.storage(), self.scratch_bytes)?;
            Some(allocation)
        };
        Ok(self.slot_from_scratch(scratch))
    }

    fn slot_from_scratch(self: &Arc<Self>, scratch: Option<Arc<Allocation>>) -> NativeGraphSlot {
        let mut locals = Vec::with_capacity(self.ports.len());
        for (ordinal, port) in self.ports.iter().enumerate() {
            let tensor = match self.port_placements[ordinal] {
                Some(Placement::Scratch(offset)) => Some(Arc::new(TensorInner::new_view(
                    self.device.clone(),
                    scratch
                        .as_ref()
                        .expect("planned scratch allocation absent")
                        .clone(),
                    offset,
                    port.byte_len,
                    port.representation,
                    port.extents.clone(),
                    port.strides.clone(),
                ))),
                Some(Placement::Export(_)) | None => None,
                Some(Placement::Scalar) => unreachable!(),
            };
            locals.push(tensor);
        }
        let mut outputs = Vec::with_capacity(self.results.len());
        for (node, row) in self.results.iter().enumerate() {
            let mut output_row = Vec::with_capacity(row.len());
            for (ordinal, result) in row.iter().enumerate() {
                let Some(spec) = result else {
                    output_row.push(None);
                    continue;
                };
                let (allocation, offset) = match self.placements[node][ordinal] {
                    Placement::Scratch(offset) => (
                        scratch
                            .as_ref()
                            .expect("planned scratch allocation absent")
                            .clone(),
                        offset,
                    ),
                    Placement::Export(_) => {
                        output_row.push(None);
                        continue;
                    }
                    Placement::Scalar => unreachable!(),
                };
                output_row.push(Some(Arc::new(TensorInner::new_view(
                    self.device.clone(),
                    allocation,
                    offset,
                    spec.byte_len,
                    spec.representation,
                    spec.extents.clone(),
                    spec.strides.clone(),
                ))));
            }
            outputs.push(output_row);
        }
        NativeGraphSlot {
            plan: self.clone(),
            locals,
            outputs,
        }
    }

    /// Exported outputs are a separate owner so their lifetime can outlive
    /// scratch-slot reuse without retaining the scratch allocation.
    pub fn new_outputs(self: &Arc<Self>) -> Result<NativeGraphOutputs, TensorError> {
        let arena = if self.output_bytes == 0 {
            None
        } else {
            Some(
                self.device
                    .allocate(self.output_bytes, self.output_alignment)?,
            )
        };
        Ok(self.outputs_from_arena(arena, None))
    }

    fn outputs_from_arena(
        self: &Arc<Self>,
        arena: Option<Arc<Allocation>>,
        family: Option<Arc<NativeGraphFamily>>,
    ) -> NativeGraphOutputs {
        let mut locals = Vec::with_capacity(self.ports.len());
        for (ordinal, port) in self.ports.iter().enumerate() {
            let tensor = if let Some(Placement::Export(offset)) = self.port_placements[ordinal] {
                Some(Arc::new(TensorInner::new_view(
                    self.device.clone(),
                    arena.as_ref().expect("planned output arena absent").clone(),
                    offset,
                    port.byte_len,
                    port.representation,
                    port.extents.clone(),
                    port.strides.clone(),
                )))
            } else {
                None
            };
            locals.push(tensor);
        }
        let mut outputs = Vec::with_capacity(self.results.len());
        for (node, row) in self.results.iter().enumerate() {
            let mut output_row = Vec::with_capacity(row.len());
            for (ordinal, result) in row.iter().enumerate() {
                let Some(spec) = result else {
                    output_row.push(None);
                    continue;
                };
                let Placement::Export(offset) = self.placements[node][ordinal] else {
                    output_row.push(None);
                    continue;
                };
                output_row.push(Some(Arc::new(TensorInner::new_view(
                    self.device.clone(),
                    arena.as_ref().expect("planned output arena absent").clone(),
                    offset,
                    spec.byte_len,
                    spec.representation,
                    spec.extents.clone(),
                    spec.strides.clone(),
                ))));
            }
            outputs.push(output_row);
        }
        NativeGraphOutputs {
            plan: self.clone(),
            locals,
            outputs,
            submitted: false,
            family,
            arena,
        }
    }
}

pub struct BoundNativeGraphPlan {
    _plan: Arc<NativeGraphPlan>,
    fixed: NativeGraphBindings,
}

impl BoundNativeGraphPlan {
    pub fn bindings(&self) -> NativeGraphBindings {
        self.fixed.clone()
    }
}

fn port_matches(device: &Arc<DeviceInner>, spec: &PortSpec, tensor: &Arc<TensorInner>) -> bool {
    Arc::ptr_eq(tensor.device(), device)
        && tensor.representation() == spec.representation
        && tensor.extents() == spec.extents
        && tensor.strides() == spec.strides
        && tensor.byte_len() == spec.byte_len
}

/// A finite family of sealed exact-shape graphs sharing one reusable scratch
/// arena per execution slot. Activating a class borrows the family slot
/// exclusively; two classes cannot use the same arena concurrently.
pub struct NativeGraphFamily {
    plans: Vec<Arc<NativeGraphPlan>>,
    device: Arc<DeviceInner>,
    workspace_bytes: u64,
    alignment: u64,
    output_bytes: u64,
    output_alignment: u64,
}

impl NativeGraphFamily {
    pub fn new(plans: &[Arc<NativeGraphPlan>]) -> Result<Self, WorkflowError> {
        let first = plans.first().ok_or(WorkflowError::Empty)?;
        if plans
            .iter()
            .any(|plan| !Arc::ptr_eq(&plan.device, &first.device))
        {
            return Err(WorkflowError::NativeGraphSlotMismatch);
        }
        Ok(Self {
            plans: plans.to_vec(),
            device: first.device.clone(),
            workspace_bytes: plans
                .iter()
                .map(|plan| plan.scratch_bytes)
                .max()
                .unwrap_or(0),
            alignment: plans
                .iter()
                .map(|plan| plan.scratch_alignment)
                .max()
                .unwrap_or(1),
            output_bytes: plans
                .iter()
                .map(|plan| plan.output_bytes)
                .max()
                .unwrap_or(0),
            output_alignment: plans
                .iter()
                .map(|plan| plan.output_alignment)
                .max()
                .unwrap_or(1),
        })
    }

    pub fn workspace_bytes(&self) -> u64 {
        self.workspace_bytes
    }

    pub fn output_bytes(&self) -> u64 {
        self.output_bytes
    }

    pub fn new_output_slot(self: &Arc<Self>) -> Result<NativeGraphFamilyOutputSlot, TensorError> {
        let arena = if self.output_bytes == 0 {
            None
        } else {
            Some(
                self.device
                    .allocate(self.output_bytes, self.output_alignment)?,
            )
        };
        Ok(NativeGraphFamilyOutputSlot {
            family: self.clone(),
            arena,
        })
    }

    pub fn new_slot(self: &Arc<Self>) -> Result<NativeGraphFamilySlot, TensorError> {
        let scratch = if self.workspace_bytes == 0 {
            None
        } else {
            let allocation = self.device.allocate(self.workspace_bytes, self.alignment)?;
            write_zeros(allocation.storage(), self.workspace_bytes)?;
            Some(allocation)
        };
        Ok(NativeGraphFamilySlot {
            family: self.clone(),
            scratch,
            lent_locals: Vec::new(),
        })
    }
}

pub struct NativeGraphFamilyOutputSlot {
    family: Arc<NativeGraphFamily>,
    arena: Option<Arc<Allocation>>,
}

impl NativeGraphFamilyOutputSlot {
    pub fn activate(
        self,
        plan: &Arc<NativeGraphPlan>,
    ) -> Result<NativeGraphOutputs, WorkflowError> {
        if !self
            .family
            .plans
            .iter()
            .any(|member| Arc::ptr_eq(member, plan))
        {
            return Err(WorkflowError::NativeGraphSlotMismatch);
        }
        Ok(plan.outputs_from_arena(self.arena, Some(self.family)))
    }
}

pub struct NativeGraphFamilySlot {
    family: Arc<NativeGraphFamily>,
    scratch: Option<Arc<Allocation>>,
    lent_locals: Vec<Weak<TensorInner>>,
}

impl NativeGraphFamilySlot {
    pub fn activate(
        &mut self,
        plan: &Arc<NativeGraphPlan>,
    ) -> Result<NativeGraphFamilyActive<'_>, WorkflowError> {
        if self
            .lent_locals
            .iter()
            .any(|tensor| tensor.strong_count() != 0)
        {
            return Err(WorkflowError::NativeExportStillLive);
        }
        self.lent_locals.clear();
        if !self
            .family
            .plans
            .iter()
            .any(|member| Arc::ptr_eq(member, plan))
        {
            return Err(WorkflowError::NativeGraphSlotMismatch);
        }
        let slot = plan.slot_from_scratch(self.scratch.clone());
        Ok(NativeGraphFamilyActive { slot, _owner: self })
    }
}

pub struct NativeGraphFamilyActive<'a> {
    slot: NativeGraphSlot,
    _owner: &'a mut NativeGraphFamilySlot,
}

impl NativeGraphFamilyActive<'_> {
    /// Lend a checked graph-local scratch tensor to an adjacent graph. The
    /// family slot cannot be reactivated until every such handle is dropped.
    pub fn local(&mut self, port: NativePort) -> Option<Arc<TensorInner>> {
        let tensor = self.slot.local(port)?;
        self._owner.lent_locals.push(Arc::downgrade(&tensor));
        Some(tensor)
    }

    pub fn write_input(&mut self, port: NativePort, bytes: &[u8]) -> Result<(), TensorError> {
        self.slot.write_input(port, bytes)
    }
    pub fn attach(
        &mut self,
        bindings: NativeGraphBindings,
        outputs: NativeGraphOutputs,
    ) -> Result<ReadyNativeGraphRun<'_>, CallError> {
        self.slot.attach(bindings, outputs)
    }
}

pub struct NativeGraphSlot {
    plan: Arc<NativeGraphPlan>,
    locals: Vec<Option<Arc<TensorInner>>>,
    outputs: Vec<Vec<Option<Arc<TensorInner>>>>,
}

impl NativeGraphSlot {
    fn local(&self, port: NativePort) -> Option<Arc<TensorInner>> {
        let ordinal = port.reference.result as usize;
        if port.reference.workflow != self.plan.identity || port.reference.node != INPUT_NODE {
            return None;
        }
        let spec = self.plan.ports.get(ordinal)?;
        if !spec.prewritten {
            return None;
        }
        self.locals.get(ordinal)?.clone()
    }

    pub fn write_input(&mut self, port: NativePort, bytes: &[u8]) -> Result<(), TensorError> {
        let ordinal = port.reference.result as usize;
        if port.reference.workflow != self.plan.identity
            || port.reference.node != INPUT_NODE
            || !self
                .plan
                .ports
                .get(ordinal)
                .is_some_and(|spec| spec.owned_input)
        {
            return Err(TensorError::Execution(ExecutionError::SubmissionFailed(
                "write_input requires an owned native graph input port".to_owned(),
            )));
        }
        self.locals[ordinal]
            .as_ref()
            .expect("planned owned input tensor is absent")
            .write_from_host(bytes)
    }
    pub fn attach(
        &mut self,
        bindings: NativeGraphBindings,
        exports: NativeGraphOutputs,
    ) -> Result<ReadyNativeGraphRun<'_>, CallError> {
        if bindings.identity != self.plan.identity || exports.plan.identity != self.plan.identity {
            return Err(CallError::Workflow(WorkflowError::NativeGraphSlotMismatch));
        }
        if exports.submitted {
            return Err(CallError::Workflow(
                WorkflowError::NativeOutputLeaseConsumed,
            ));
        }
        let combined = self
            .outputs
            .iter()
            .zip(&exports.outputs)
            .map(|(scratch, retained)| {
                scratch
                    .iter()
                    .zip(retained)
                    .map(|(scratch, retained)| scratch.clone().or_else(|| retained.clone()))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let inputs = bindings
            .inputs
            .into_iter()
            .enumerate()
            .map(|(port, tensor)| {
                tensor
                    .or_else(|| self.locals[port].clone())
                    .or_else(|| exports.locals[port].clone())
                    .ok_or(CallError::Workflow(WorkflowError::NativePortUnbound {
                        port,
                    }))
            })
            .collect::<Result<Vec<_>, _>>()?;
        for (ordinal, (tensor, spec)) in inputs.iter().zip(&self.plan.ports).enumerate() {
            if !port_matches(&self.plan.device, spec, tensor) {
                return Err(CallError::Workflow(WorkflowError::NativePortMismatch {
                    port: ordinal,
                }));
            }
        }
        let mut bound = Vec::with_capacity(self.plan.nodes.len());
        for (ordinal, node) in self.plan.nodes.iter().enumerate() {
            let args = resolve_arguments(&node.args, &inputs, &combined, self.plan.identity)?;
            let outputs = combined[ordinal]
                .iter()
                .filter_map(Clone::clone)
                .collect::<Vec<_>>();
            bound.push(node.kernel.inner.bind(args, outputs)?);
        }
        Ok(ReadyNativeGraphRun {
            _slot: self,
            bound,
            exports,
        })
    }
}

pub struct NativeGraphOutputs {
    plan: Arc<NativeGraphPlan>,
    locals: Vec<Option<Arc<TensorInner>>>,
    outputs: Vec<Vec<Option<Arc<TensorInner>>>>,
    submitted: bool,
    family: Option<Arc<NativeGraphFamily>>,
    arena: Option<Arc<Allocation>>,
}

impl NativeGraphOutputs {
    fn reserved_export(
        &self,
        reference: WorkflowResultRef,
    ) -> Result<Arc<TensorInner>, WorkflowError> {
        if self.submitted {
            return Err(WorkflowError::NativeOutputLeaseConsumed);
        }
        if reference.workflow != self.plan.identity || !self.plan.exports.contains(&reference) {
            return Err(WorkflowError::CrossWorkflowResult);
        }
        let tensor = if reference.node == INPUT_NODE {
            self.locals.get(reference.result as usize)
        } else {
            self.outputs
                .get(reference.node as usize)
                .and_then(|row| row.get(reference.result as usize))
        };
        tensor
            .and_then(Option::as_ref)
            .cloned()
            .ok_or(WorkflowError::MissingProducerResult)
    }

    /// Return a family output slot after all exported tensor handles have
    /// been released. A live export prevents reuse of the same physical bytes.
    pub fn recycle(self) -> Result<NativeGraphFamilyOutputSlot, WorkflowError> {
        let family = self
            .family
            .clone()
            .ok_or(WorkflowError::NativeGraphSlotMismatch)?;
        let retained = self
            .locals
            .iter()
            .chain(self.outputs.iter().flatten())
            .filter_map(Option::as_ref)
            .any(|tensor| Arc::strong_count(tensor) != 1);
        if retained {
            return Err(WorkflowError::NativeExportStillLive);
        }
        Ok(NativeGraphFamilyOutputSlot {
            family,
            arena: self.arena.clone(),
        })
    }
    pub fn exported_tensor(&self, reference: WorkflowResultRef) -> Option<Arc<TensorInner>> {
        if !self.submitted
            || reference.workflow != self.plan.identity
            || !self.plan.exports.contains(&reference)
        {
            return None;
        }
        if reference.node == INPUT_NODE {
            return self.locals.get(reference.result as usize)?.clone();
        }
        self.outputs
            .get(reference.node as usize)?
            .get(reference.result as usize)?
            .clone()
    }
}

#[derive(Clone)]
pub struct NativeGraphBindings {
    identity: u64,
    inputs: Vec<Option<Arc<TensorInner>>>,
    external: Vec<bool>,
}

impl NativeGraphBindings {
    /// Attach an exported tensor from a reserved, unsubmitted result lease.
    /// The destination port descriptor and aliases are checked by `attach`.
    pub fn set_reserved_export(
        &mut self,
        port: NativePort,
        outputs: &NativeGraphOutputs,
        reference: WorkflowResultRef,
    ) -> Result<(), WorkflowError> {
        let tensor = outputs.reserved_export(reference)?;
        self.set(port, tensor)
    }

    pub fn set(&mut self, port: NativePort, tensor: Arc<TensorInner>) -> Result<(), WorkflowError> {
        if port.reference.workflow != self.identity || port.reference.node != INPUT_NODE {
            return Err(WorkflowError::CrossWorkflowResult);
        }
        let ordinal = port.reference.result as usize;
        if !self.external.get(ordinal).copied().unwrap_or(false) {
            return Err(WorkflowError::NativePortMismatch { port: ordinal });
        }
        let value = self
            .inputs
            .get_mut(ordinal)
            .ok_or(WorkflowError::CrossWorkflowResult)?;
        if value.is_some() {
            return Err(WorkflowError::NativePortAlreadyBound { port: ordinal });
        }
        *value = Some(tensor);
        Ok(())
    }
}

pub struct ReadyNativeGraphRun<'a> {
    _slot: &'a mut NativeGraphSlot,
    bound: Vec<NativeBoundKind>,
    exports: NativeGraphOutputs,
}

impl ReadyNativeGraphRun<'_> {
    /// Submission consumes the already checked attachments. Missing ports,
    /// wrong extents, and illegal aliases cannot enter this method.
    pub fn run(self) -> Result<NativeGraphOutputs, CallError> {
        NativeBoundKind::run_graph(self.bound)?;
        let mut exports = self.exports;
        exports.submitted = true;
        Ok(exports)
    }
}

fn resolve_arguments(
    args: &EncodedWorkflowArgs,
    inputs: &[Arc<TensorInner>],
    outputs: &[Vec<Option<Arc<TensorInner>>>],
    identity: u64,
) -> Result<EncodedArgs, CallError> {
    let mut resolved = EncodedArgs::new();
    for argument in args.clone().into_arguments() {
        match argument {
            EncodedWorkflowArgument::Tensor(WorkflowTensorArgument::External(_)) => {
                return Err(CallError::Workflow(WorkflowError::NativePortMismatch {
                    port: usize::MAX,
                }));
            }
            EncodedWorkflowArgument::Tensor(WorkflowTensorArgument::Result(reference)) => {
                resolved.push_tensor(resolve_reference(reference, inputs, outputs, identity)?);
            }
            EncodedWorkflowArgument::Tensor(WorkflowTensorArgument::ResultView {
                result,
                operations,
            }) => {
                let tensor = resolve_reference(result, inputs, outputs, identity)?;
                let mut view = tensor;
                for operation in operations {
                    view = Arc::new(
                        match operation {
                            ViewOperation::LeadingSlice { start, end } => {
                                view.slice_leading(start, end)
                            }
                            ViewOperation::Reshape { extents } => view.reshape(&extents),
                        }
                        .map_err(WorkflowError::TensorView)
                        .map_err(CallError::Workflow)?,
                    );
                }
                resolved.push_tensor(view);
            }
            EncodedWorkflowArgument::Scalar(value) => resolved.push_scalar(value),
            EncodedWorkflowArgument::ScalarResult(_) => {
                return Err(CallError::Workflow(WorkflowError::HostBoundaryRequired));
            }
        }
    }
    Ok(resolved)
}

fn resolve_reference(
    reference: WorkflowResultRef,
    inputs: &[Arc<TensorInner>],
    outputs: &[Vec<Option<Arc<TensorInner>>>],
    identity: u64,
) -> Result<Arc<TensorInner>, CallError> {
    if reference.workflow != identity {
        return Err(CallError::Workflow(WorkflowError::CrossWorkflowResult));
    }
    if reference.node == INPUT_NODE {
        return inputs
            .get(reference.result as usize)
            .cloned()
            .ok_or(CallError::Workflow(WorkflowError::MissingProducerResult));
    }
    outputs
        .get(reference.node as usize)
        .and_then(|row| row.get(reference.result as usize))
        .and_then(Clone::clone)
        .ok_or(CallError::Workflow(WorkflowError::MissingProducerResult))
}
