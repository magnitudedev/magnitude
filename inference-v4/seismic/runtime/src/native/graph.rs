//! Checked direct-native graph construction and reusable result storage.
//!
//! A graph is sealed from the generated entry contracts before resident model
//! storage is admitted. Ports are the only late bindings. No consumer supplies
//! a name-to-buffer recipe for intermediate results.
//!
//! Sealing validates every node against its entry contract once and fixes
//! its executable form: argument words, launch geometry, and for every
//! buffer the storage region and offset it lives at. A run supplies only the
//! regions (the slot's workspace, the run's upload region and output arena)
//! and the tensors bound to external ports; attaching checks those bindings
//! alone, so per-run host work does not revalidate nodes.

use super::{
    merge_access, submit, word_bytes, CallLaunches, Dispatch, DispatchList, NativePrepared,
    NativeSubmission, NativeTensorSpec, BUFFER_ALIGNMENT,
};
use crate::api::device::DeviceInner;
use crate::api::kernel::{
    EncodedWorkflowArgs, EncodedWorkflowArgument, NativePreparedAny, PendingWorkflowResults,
    ViewOperation, WorkflowResultRef, WorkflowTensorArgument,
};
use crate::api::tensor::TensorInner;
use crate::api::{CallError, TensorError, WorkflowError};
use crate::driver::{write_zeros, Allocation};
use crate::layout;
use seismic_compiler::errors::{ExecutionError, InvocationError};
use seismic_compiler::prepared::{ArgumentValue, DeviceIdentity, TensorDescriptor};
use seismic_lang::entry::{AliasRule, ParameterKind, TensorAccess};
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
        prewrite_port(self.identity, &mut self.ports, port)
    }

    pub fn enqueue(
        &mut self,
        kernel: &Arc<NativePreparedAny>,
        args: EncodedWorkflowArgs,
    ) -> Result<PendingWorkflowResults, WorkflowError> {
        let node = u32::try_from(self.nodes.len()).expect("native graph node ordinal exhausted");
        validate_node_references(self.identity, self.ports.len(), node, &args)?;
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
        validate_export_reference(self.identity, &self.ports, self.nodes.len(), reference)?;
        if !self.exports.contains(&reference) {
            self.exports.push(reference);
        }
        Ok(())
    }

    pub fn seal(self) -> Result<NativeGraphPlan, CallError> {
        if self.nodes.is_empty() {
            return Err(CallError::Workflow(WorkflowError::Empty));
        }
        validate_used_ports(
            self.ports.len(),
            self.nodes.iter().map(|node| &node.args),
            &self.exports,
        )
        .map_err(CallError::Workflow)?;
        let identity = self.identity;
        let device_identity = self.device.kind.identity();
        let mut results: Vec<Vec<Option<NativeTensorSpec>>> = Vec::new();
        let mut planned = Vec::with_capacity(self.nodes.len());
        let mut shaped = Vec::with_capacity(self.nodes.len());
        for node in self.nodes {
            node.kernel.inner.validate_graph_node(device_identity)?;
            let arguments =
                describe_arguments(&node.args, &self.ports, &results, identity, device_identity)?;
            let shape = node.kernel.inner.shape(&arguments)?;
            results.push(shape.results.clone());
            planned.push(PlannedNode {
                args: node.args,
                scratch: shape.scratch.clone(),
            });
            shaped.push((node.kernel.inner.clone(), arguments, shape));
        }
        for reference in &self.exports {
            if reference.node != INPUT_NODE && result_spec(&results, *reference).is_none() {
                return Err(CallError::Workflow(WorkflowError::MissingProducerResult));
            }
        }
        let storage = plan_storage(&self.ports, &planned, &results, &self.exports);
        let mut executable = Executable {
            nodes: Vec::with_capacity(planned.len()),
            external_writes: vec![false; self.ports.len()],
            upload_written: false,
            disjoint: Vec::new(),
        };
        for (node, ((kernel, arguments, shape), planned)) in
            shaped.into_iter().zip(&planned).enumerate()
        {
            executable.seal_node(kernel, &arguments, shape, &planned.args, &storage, node);
        }
        Ok(NativeGraphPlan {
            identity,
            device: self.device,
            ports: self.ports,
            executable,
            results,
            exports: self.exports,
            port_placements: storage.port_placements,
            placements: storage.placements,
            scratch_bytes: storage.scratch_bytes,
            output_bytes: storage.output_bytes,
            upload_bytes: storage.upload_bytes,
        })
    }
}

/// Storage-only graph construction from checked entry metadata. It uses the
/// same port identities and interval allocator as a prepared native graph,
/// but owns no device, formed kernel, or executable node.
pub struct NativeGraphMetadataDraft {
    identity: u64,
    ports: Vec<PortSpec>,
    nodes: Vec<PlannedNode>,
    results: Vec<Vec<Option<NativeTensorSpec>>>,
    exports: Vec<WorkflowResultRef>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeGraphStorageBytes {
    pub workspace: u64,
    pub output: u64,
    pub upload: u64,
}

impl NativeGraphMetadataDraft {
    pub fn new() -> Self {
        Self {
            identity: NEXT_NATIVE_GRAPH.fetch_add(1, Ordering::Relaxed),
            ports: Vec::new(),
            nodes: Vec::new(),
            results: Vec::new(),
            exports: Vec::new(),
        }
    }

    pub fn port(
        &mut self,
        representation: RepresentationId,
        extents: &[u64],
        local: bool,
        owned_input: bool,
    ) -> Result<NativePort, TensorError> {
        let layout = layout::canonical(representation, extents)?;
        let ordinal = u32::try_from(self.ports.len()).expect("native port ordinal exhausted");
        self.ports.push(PortSpec {
            representation,
            extents: extents.to_vec(),
            strides: layout.strides,
            byte_len: layout.byte_len,
            local,
            owned_input,
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

    pub fn prewrite(&mut self, port: NativePort) -> Result<(), WorkflowError> {
        prewrite_port(self.identity, &mut self.ports, port)
    }

    pub fn enqueue(
        &mut self,
        args: EncodedWorkflowArgs,
        parameter_shapes: Vec<Option<(RepresentationId, Vec<u64>)>>,
        result_shapes: Vec<Option<(RepresentationId, Vec<u64>)>>,
        scratch: Vec<u64>,
    ) -> Result<PendingWorkflowResults, CallError> {
        let node = u32::try_from(self.nodes.len()).expect("native graph node ordinal exhausted");
        validate_node_references(self.identity, self.ports.len(), node, &args)
            .map_err(CallError::Workflow)?;
        if parameter_shapes.len() != args.arguments().len() {
            return Err(CallError::Workflow(
                WorkflowError::NativeGraphArgumentMismatch { parameter: 0 },
            ));
        }
        for (ordinal, (argument, expected)) in args
            .arguments()
            .iter()
            .zip(parameter_shapes.iter())
            .enumerate()
        {
            match (argument, expected) {
                (EncodedWorkflowArgument::Tensor(argument), Some((representation, extents))) => {
                    let actual = metadata_argument_shape(argument, &self.ports, &self.results)?;
                    if actual.0 != *representation || actual.1 != *extents {
                        return Err(CallError::Workflow(
                            WorkflowError::NativeGraphArgumentMismatch { parameter: ordinal },
                        ));
                    }
                }
                (EncodedWorkflowArgument::Scalar(_), None) => {}
                _ => {
                    return Err(CallError::Workflow(
                        WorkflowError::NativeGraphArgumentMismatch { parameter: ordinal },
                    ));
                }
            }
        }
        let results = result_shapes
            .into_iter()
            .map(|shape| {
                let Some((representation, extents)) = shape else {
                    return Err(CallError::Workflow(WorkflowError::HostBoundaryRequired));
                };
                let layout =
                    layout::canonical(representation, &extents).map_err(CallError::Execution)?;
                Ok(Some(NativeTensorSpec {
                    representation,
                    extents,
                    strides: layout.strides,
                    byte_len: layout.byte_len,
                }))
            })
            .collect::<Result<Vec<_>, CallError>>()?;
        let count = u32::try_from(results.len()).expect("native result ordinal exhausted");
        self.nodes.push(PlannedNode { args, scratch });
        self.results.push(results);
        Ok(PendingWorkflowResults::new(self.identity, node, count))
    }

    pub fn export(&mut self, reference: WorkflowResultRef) -> Result<(), WorkflowError> {
        validate_export_reference(self.identity, &self.ports, self.nodes.len(), reference)?;
        if !self.exports.contains(&reference) {
            self.exports.push(reference);
        }
        Ok(())
    }

    pub fn seal(self) -> Result<NativeGraphStorageBytes, WorkflowError> {
        if self.nodes.is_empty() {
            return Err(WorkflowError::Empty);
        }
        validate_used_ports(
            self.ports.len(),
            self.nodes.iter().map(|node| &node.args),
            &self.exports,
        )?;
        for reference in &self.exports {
            if reference.node != INPUT_NODE && result_spec(&self.results, *reference).is_none() {
                return Err(WorkflowError::MissingProducerResult);
            }
        }
        let storage = plan_storage(&self.ports, &self.nodes, &self.results, &self.exports);
        Ok(NativeGraphStorageBytes {
            workspace: storage.scratch_bytes,
            output: storage.output_bytes,
            upload: storage.upload_bytes,
        })
    }
}

fn metadata_argument_shape(
    argument: &WorkflowTensorArgument,
    ports: &[PortSpec],
    results: &[Vec<Option<NativeTensorSpec>>],
) -> Result<(RepresentationId, Vec<u64>), CallError> {
    let (reference, operations) = match argument {
        WorkflowTensorArgument::Result(reference) => (*reference, &[][..]),
        WorkflowTensorArgument::ResultView { result, operations } => {
            (*result, operations.as_slice())
        }
        WorkflowTensorArgument::External(_) => {
            return Err(CallError::Workflow(WorkflowError::NativePortMismatch {
                port: usize::MAX,
            }));
        }
    };
    let (representation, mut geometry) = if reference.node == INPUT_NODE {
        let port = &ports[reference.result as usize];
        (
            port.representation,
            layout::ViewGeometry {
                extents: port.extents.clone(),
                strides: port.strides.clone(),
                byte_offset: 0,
                byte_len: port.byte_len,
            },
        )
    } else {
        let result = result_spec(results, reference)
            .ok_or(CallError::Workflow(WorkflowError::MissingProducerResult))?;
        (
            result.representation,
            layout::ViewGeometry {
                extents: result.extents.clone(),
                strides: result.strides.clone(),
                byte_offset: 0,
                byte_len: result.byte_len,
            },
        )
    };
    for operation in operations {
        geometry = layout::apply_view(representation, geometry, operation)
            .map_err(|error| CallError::Workflow(WorkflowError::TensorView(error)))?;
    }
    Ok((representation, geometry.extents))
}

fn prewrite_port(
    identity: u64,
    ports: &mut [PortSpec],
    port: NativePort,
) -> Result<(), WorkflowError> {
    let ordinal = port.reference.result as usize;
    if port.reference.workflow != identity || port.reference.node != INPUT_NODE {
        return Err(WorkflowError::CrossWorkflowResult);
    }
    let spec = ports
        .get_mut(ordinal)
        .ok_or(WorkflowError::CrossWorkflowResult)?;
    if !spec.local || spec.owned_input {
        return Err(WorkflowError::NativePortMismatch { port: ordinal });
    }
    spec.prewritten = true;
    Ok(())
}

fn validate_node_references(
    identity: u64,
    port_count: usize,
    node: u32,
    args: &EncodedWorkflowArgs,
) -> Result<(), WorkflowError> {
    for argument in args.arguments() {
        let reference = argument_reference(argument);
        if reference.is_some_and(|reference| {
            reference.workflow != identity
                || (reference.node != INPUT_NODE && reference.node >= node)
                || (reference.node == INPUT_NODE && reference.result as usize >= port_count)
        }) {
            return Err(WorkflowError::CrossWorkflowResult);
        }
    }
    Ok(())
}

fn validate_export_reference(
    identity: u64,
    ports: &[PortSpec],
    node_count: usize,
    reference: WorkflowResultRef,
) -> Result<(), WorkflowError> {
    if reference.workflow != identity
        || (reference.node == INPUT_NODE
            && !ports
                .get(reference.result as usize)
                .is_some_and(|port| port.local))
        || (reference.node != INPUT_NODE && reference.node as usize >= node_count)
    {
        return Err(WorkflowError::CrossWorkflowResult);
    }
    Ok(())
}

fn validate_used_ports<'a>(
    port_count: usize,
    nodes: impl Iterator<Item = &'a EncodedWorkflowArgs>,
    exports: &[WorkflowResultRef],
) -> Result<(), WorkflowError> {
    let mut used = vec![false; port_count];
    for node in nodes {
        for argument in node.arguments() {
            if let Some(reference) = argument_reference(argument) {
                if reference.node == INPUT_NODE {
                    used[reference.result as usize] = true;
                }
            }
        }
    }
    for reference in exports {
        if reference.node == INPUT_NODE {
            used[reference.result as usize] = true;
        }
    }
    if let Some(port) = used.iter().position(|used| !used) {
        return Err(WorkflowError::NativePortUnbound { port });
    }
    Ok(())
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

/// A node as storage planning sees it.
struct PlannedNode {
    args: EncodedWorkflowArgs,
    /// Bytes of each of the node's call-private scratch buffers.
    scratch: Vec<u64>,
}

/// Where a node's buffer lives. Only an external port's tensor is bound
/// per run; every other buffer is placed at seal.
#[derive(Clone, Copy, Debug)]
enum Region {
    /// The tensor bound to this external port; the site offset is added to
    /// the tensor's own byte offset.
    External(usize),
    /// The slot's workspace arena: results, graph locals and node scratch.
    Workspace,
    /// The run's upload region: host-written inputs.
    Upload,
    /// The run's output arena: exports.
    Outputs,
}

#[derive(Clone, Copy, Debug)]
struct Site {
    region: Region,
    offset: u64,
}

/// One node in executable form, fixed at seal.
struct SealedNode {
    kernel: Arc<NativePrepared>,
    /// Buffers in ABI order.
    sites: Vec<Site>,
    representations: Vec<&'static str>,
    words: Vec<u64>,
    word_bytes: Vec<u8>,
    /// Per launch ordinal, fixed at seal; `None` for an inactive launch.
    launches: CallLaunches,
}

/// Byte range of an external port's view in one node argument.
#[derive(Clone, Debug)]
struct PortRange {
    port: usize,
    offset: u64,
    bytes: u64,
    /// The entry parameter, for the aliasing error.
    parameter: String,
}

/// The sealed nodes and what a run must check of its external bindings.
struct Executable {
    nodes: Vec<SealedNode>,
    /// Per port: whether any node writes the external tensor bound to it.
    external_writes: Vec<bool>,
    /// Whether any node writes a host-written input.
    upload_written: bool,
    /// Argument pairs on two different external ports that their entry
    /// requires to be disjoint. Pairs on graph storage, or on one port,
    /// were decided at seal.
    disjoint: Vec<(PortRange, PortRange)>,
}

impl Executable {
    fn seal_node(
        &mut self,
        kernel: Arc<NativePrepared>,
        arguments: &[ArgumentValue],
        shape: super::CallShape,
        args: &EncodedWorkflowArgs,
        storage: &StoragePlan,
        node: usize,
    ) {
        let schema = kernel.schema();
        let mut sites = Vec::new();
        let mut representations = Vec::new();
        let mut external = vec![None; arguments.len()];
        for (ordinal, ((parameter, argument), value)) in schema
            .parameters()
            .iter()
            .zip(args.arguments())
            .zip(arguments)
            .enumerate()
        {
            let ParameterKind::Tensor { access, .. } = &parameter.kind else {
                continue;
            };
            let ArgumentValue::Tensor(descriptor) = value else {
                unreachable!("a validated tensor parameter carries a tensor descriptor");
            };
            let reference = argument_reference(argument)
                .expect("a sealed graph tensor argument names a port or a result");
            let (region, base) = if reference.node == INPUT_NODE {
                let port = reference.result as usize;
                match storage.port_placements[port] {
                    Some(placement) => placement.site(),
                    None => {
                        external[ordinal] = Some(PortRange {
                            port,
                            offset: descriptor.byte_offset,
                            bytes: descriptor.byte_len,
                            parameter: parameter.name.clone(),
                        });
                        (Region::External(port), 0)
                    }
                }
            } else {
                storage.placements[reference.node as usize][reference.result as usize].site()
            };
            let write = matches!(access, TensorAccess::Owned | TensorAccess::Mutable);
            match region {
                Region::External(port) => self.external_writes[port] |= write,
                Region::Upload => self.upload_written |= write,
                Region::Workspace | Region::Outputs => {}
            }
            sites.push(Site {
                region,
                offset: base + descriptor.byte_offset,
            });
            representations
                .push(seismic_lang::registry::representation_info(descriptor.representation).name);
        }
        for rule in schema.aliases() {
            let AliasRule::Disjoint(first, second) = *rule else {
                continue;
            };
            let position = |id| {
                schema
                    .parameters()
                    .iter()
                    .position(|parameter| parameter.id == id)
                    .expect("alias rule names a schema parameter")
            };
            if let (Some(first), Some(second)) =
                (&external[position(first)], &external[position(second)])
            {
                if first.port != second.port {
                    self.disjoint.push((first.clone(), second.clone()));
                }
            }
        }
        for (ordinal, result) in shape.results.iter().enumerate() {
            let Some(spec) = result else { continue };
            let (region, offset) = storage.placements[node][ordinal].site();
            sites.push(Site { region, offset });
            representations
                .push(seismic_lang::registry::representation_info(spec.representation).name);
        }
        for offset in &storage.node_scratch[node] {
            sites.push(Site {
                region: Region::Workspace,
                offset: *offset,
            });
            representations.push("bytes");
        }
        self.nodes.push(SealedNode {
            kernel,
            sites,
            representations,
            word_bytes: word_bytes(&shape.words),
            words: shape.words,
            launches: shape.launches,
        });
    }
}

#[derive(Clone, Copy)]
enum Placement {
    Scratch(u64),
    Export(u64),
    /// A host-written input in the per-submission upload region.
    Upload(u64),
    Scalar,
}

impl Placement {
    fn site(self) -> (Region, u64) {
        match self {
            Self::Scratch(offset) => (Region::Workspace, offset),
            Self::Export(offset) => (Region::Outputs, offset),
            Self::Upload(offset) => (Region::Upload, offset),
            Self::Scalar => unreachable!("a scalar result is never a graph buffer"),
        }
    }
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
    NodeScratch(usize, usize),
}

struct StoragePlan {
    port_placements: Vec<Option<Placement>>,
    placements: Vec<Vec<Placement>>,
    node_scratch: Vec<Vec<u64>>,
    scratch_bytes: u64,
    output_bytes: u64,
    upload_bytes: u64,
}

struct Interval {
    key: StorageKey,
    start: usize,
    end: usize,
    bytes: u64,
    exported: bool,
}

fn align_up(value: u64, alignment: u64) -> u64 {
    value.div_ceil(alignment) * alignment
}

/// Place graph storage. Results, graph locals and node scratch share one
/// arena by lifetime (a node's scratch is live across its inputs and
/// results, so it never aliases them); exports get their own arena; inputs
/// the host writes go to a per-submission upload region so a later host
/// write cannot race an earlier submission still reading them. Every buffer
/// starts at a [`BUFFER_ALIGNMENT`] offset of its region.
fn plan_storage(
    ports: &[PortSpec],
    nodes: &[PlannedNode],
    results: &[Vec<Option<NativeTensorSpec>>],
    exports: &[WorkflowResultRef],
) -> StoragePlan {
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
    let mut upload_bytes = 0u64;
    let mut port_placements = vec![None; ports.len()];
    for (ordinal, port) in ports.iter().enumerate() {
        if !port.local {
            continue;
        }
        if port.owned_input {
            let offset = align_up(upload_bytes, BUFFER_ALIGNMENT);
            upload_bytes = offset
                .checked_add(port.byte_len.max(1))
                .expect("native graph upload bytes overflow");
            port_placements[ordinal] = Some(Placement::Upload(offset));
            continue;
        }
        let exported = exports
            .iter()
            .any(|item| item.node == INPUT_NODE && item.result as usize == ordinal);
        intervals.push(Interval {
            key: StorageKey::Port(ordinal),
            start: if port.prewritten {
                0
            } else {
                port_first[ordinal].unwrap_or(0)
            },
            end: port_last[ordinal],
            bytes: port.byte_len.max(1),
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
                exported,
            });
        }
    }
    for (node, planned) in nodes.iter().enumerate() {
        for (ordinal, bytes) in planned.scratch.iter().enumerate() {
            intervals.push(Interval {
                key: StorageKey::NodeScratch(node, ordinal),
                start: node * 2,
                end: node * 2 + 1,
                bytes: *bytes,
                exported: false,
            });
        }
    }
    intervals.sort_by_key(|interval| interval.start);
    let mut blocks: Vec<ScratchBlock> = Vec::new();
    let mut cursor = 0u64;
    let mut output_bytes = 0u64;
    let mut placements = results
        .iter()
        .map(|row| vec![Placement::Scalar; row.len()])
        .collect::<Vec<_>>();
    let mut node_scratch = nodes
        .iter()
        .map(|node| vec![0u64; node.scratch.len()])
        .collect::<Vec<_>>();
    for interval in intervals {
        let placement = if interval.exported {
            let offset = align_up(output_bytes, BUFFER_ALIGNMENT);
            output_bytes = offset
                .checked_add(interval.bytes)
                .expect("native graph output bytes overflow");
            Placement::Export(offset)
        } else {
            let reuse = blocks.iter_mut().find(|block| {
                block.live_until < interval.start && block.capacity >= interval.bytes
            });
            let offset = if let Some(block) = reuse {
                block.live_until = interval.end;
                block.offset
            } else {
                let offset = align_up(cursor, BUFFER_ALIGNMENT);
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
        match (interval.key, placement) {
            (StorageKey::Port(ordinal), _) => port_placements[ordinal] = Some(placement),
            (StorageKey::Result(node, ordinal), _) => placements[node][ordinal] = placement,
            (StorageKey::NodeScratch(node, ordinal), Placement::Scratch(offset)) => {
                node_scratch[node][ordinal] = offset;
            }
            (StorageKey::NodeScratch(..), _) => unreachable!("node scratch is never exported"),
        }
    }
    StoragePlan {
        port_placements,
        placements,
        node_scratch,
        scratch_bytes: cursor,
        output_bytes,
        upload_bytes,
    }
}

pub struct NativeGraphPlan {
    identity: u64,
    device: Arc<DeviceInner>,
    ports: Vec<PortSpec>,
    executable: Executable,
    results: Vec<Vec<Option<NativeTensorSpec>>>,
    exports: Vec<WorkflowResultRef>,
    port_placements: Vec<Option<Placement>>,
    placements: Vec<Vec<Placement>>,
    scratch_bytes: u64,
    output_bytes: u64,
    upload_bytes: u64,
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
    /// Bytes of one submission's host-written inputs.
    pub fn upload_bytes(&self) -> u64 {
        self.upload_bytes
    }
    pub fn bindings(&self) -> NativeGraphBindings {
        NativeGraphBindings {
            identity: self.identity,
            inputs: vec![None; self.ports.len()],
            external: self.ports.iter().map(|port| !port.local).collect(),
            checked: vec![false; self.ports.len()],
        }
    }
    /// Bind the ports that stay fixed across runs, checked once here; a
    /// run's attach checks only the ports set after this.
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
            if !port_matches(&self.device, &self.ports[ordinal], tensor)
                || (self.executable.external_writes[ordinal]
                    && tensor.allocation().storage().read_only())
            {
                return Err(CallError::Workflow(WorkflowError::NativePortMismatch {
                    port: ordinal,
                }));
            }
            bindings.checked[ordinal] = true;
        }
        Ok(BoundNativeGraphPlan {
            _plan: self.clone(),
            fixed: bindings,
        })
    }
    /// A slot of this plan alone. Its upload region is fixed, so host writes
    /// to its inputs wait for the slot's previous submission; a family slot
    /// rotates upload regions instead.
    pub fn new_slot(self: &Arc<Self>) -> Result<NativeGraphSlot, TensorError> {
        let workspace = if self.scratch_bytes == 0 {
            None
        } else {
            let allocation = self.device.allocate(self.scratch_bytes, BUFFER_ALIGNMENT)?;
            write_zeros(allocation.storage(), self.scratch_bytes)?;
            Some(allocation)
        };
        let upload = if self.upload_bytes == 0 {
            None
        } else {
            Some(
                self.device
                    .allocate_upload(self.upload_bytes, BUFFER_ALIGNMENT)?,
            )
        };
        Ok(NativeGraphSlot {
            plan: self.clone(),
            workspace,
            upload,
        })
    }

    /// Exported outputs are a separate owner so their lifetime can outlive
    /// scratch-slot reuse without retaining the scratch allocation.
    pub fn new_outputs(self: &Arc<Self>) -> Result<NativeGraphOutputs, TensorError> {
        let arena = if self.output_bytes == 0 {
            None
        } else {
            Some(self.device.allocate(self.output_bytes, BUFFER_ALIGNMENT)?)
        };
        Ok(self.outputs_from_arena(arena, None))
    }

    fn port_view(
        &self,
        ordinal: usize,
        allocation: &Arc<Allocation>,
        offset: u64,
    ) -> Arc<TensorInner> {
        let port = &self.ports[ordinal];
        Arc::new(TensorInner::new_view(
            self.device.clone(),
            allocation.clone(),
            offset,
            port.byte_len,
            port.representation,
            port.extents.clone(),
            port.strides.clone(),
        ))
    }

    fn outputs_from_arena(
        self: &Arc<Self>,
        arena: Option<Arc<Allocation>>,
        family: Option<Arc<NativeGraphFamily>>,
    ) -> NativeGraphOutputs {
        let mut locals = Vec::with_capacity(self.ports.len());
        for ordinal in 0..self.ports.len() {
            let tensor = if let Some(Placement::Export(offset)) = self.port_placements[ordinal] {
                Some(self.port_view(
                    ordinal,
                    arena.as_ref().expect("planned output arena absent"),
                    offset,
                ))
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
    output_bytes: u64,
    upload_bytes: u64,
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
            output_bytes: plans
                .iter()
                .map(|plan| plan.output_bytes)
                .max()
                .unwrap_or(0),
            upload_bytes: plans
                .iter()
                .map(|plan| plan.upload_bytes)
                .max()
                .unwrap_or(0),
        })
    }

    /// Bytes of one upload region. A slot holds one region per submission
    /// still in flight.
    pub fn upload_bytes(&self) -> u64 {
        self.upload_bytes
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
            Some(self.device.allocate(self.output_bytes, BUFFER_ALIGNMENT)?)
        };
        Ok(NativeGraphFamilyOutputSlot {
            family: self.clone(),
            arena,
        })
    }

    /// A slot with its scratch arena and `regions` upload regions, all
    /// allocated here: one region per graph run the slot's owner keeps in
    /// flight at once. A family whose plans write no input allocates none.
    pub fn new_slot(
        self: &Arc<Self>,
        regions: usize,
    ) -> Result<NativeGraphFamilySlot, TensorError> {
        let scratch = if self.workspace_bytes == 0 {
            None
        } else {
            let allocation = self
                .device
                .allocate(self.workspace_bytes, BUFFER_ALIGNMENT)?;
            write_zeros(allocation.storage(), self.workspace_bytes)?;
            Some(allocation)
        };
        let uploads = if self.upload_bytes == 0 {
            Vec::new()
        } else {
            (0..regions)
                .map(|_| {
                    self.device
                        .allocate_upload(self.upload_bytes, BUFFER_ALIGNMENT)
                })
                .collect::<Result<Vec<_>, _>>()?
        };
        Ok(NativeGraphFamilySlot {
            family: self.clone(),
            scratch,
            uploads,
            next_upload: 0,
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
    /// Upload regions, fixed when the slot was created. Activation takes one
    /// no submission still reads and no tensor view still names.
    uploads: Vec<Arc<Allocation>>,
    /// The region the next activation tries first. Regions are taken in
    /// rotation, so a submission sequence of a fixed length binds the same
    /// region at the same position every time and replays its graphs.
    next_upload: usize,
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
        let upload = self.idle_upload()?;
        let slot = NativeGraphSlot {
            plan: plan.clone(),
            workspace: self.scratch.clone(),
            upload,
        };
        Ok(NativeGraphFamilyActive { slot, _owner: self })
    }

    /// The first idle region in rotation order from `next_upload`.
    fn idle_upload(&mut self) -> Result<Option<Arc<Allocation>>, WorkflowError> {
        if self.family.upload_bytes == 0 {
            return Ok(None);
        }
        let regions = self.uploads.len();
        let index = (0..regions)
            .map(|step| (self.next_upload + step) % regions)
            .find(|index| {
                let upload = &self.uploads[*index];
                Arc::strong_count(upload) == 1 && upload.device_idle()
            })
            .ok_or(WorkflowError::UploadRegionsExhausted { regions })?;
        self.next_upload = (index + 1) % regions;
        Ok(Some(self.uploads[index].clone()))
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
    /// Results, graph locals and node scratch.
    workspace: Option<Arc<Allocation>>,
    /// Host-written inputs of the next run.
    upload: Option<Arc<Allocation>>,
}

impl NativeGraphSlot {
    fn local(&self, port: NativePort) -> Option<Arc<TensorInner>> {
        let ordinal = port.reference.result as usize;
        if port.reference.workflow != self.plan.identity || port.reference.node != INPUT_NODE {
            return None;
        }
        if !self.plan.ports.get(ordinal)?.prewritten {
            return None;
        }
        let Some(Placement::Scratch(offset)) = self.plan.port_placements[ordinal] else {
            return None;
        };
        let workspace = self
            .workspace
            .as_ref()
            .expect("planned graph workspace absent");
        Some(self.plan.port_view(ordinal, workspace, offset))
    }

    pub fn write_input(&mut self, port: NativePort, bytes: &[u8]) -> Result<(), TensorError> {
        let ordinal = port.reference.result as usize;
        let owned = port.reference.workflow == self.plan.identity
            && port.reference.node == INPUT_NODE
            && self
                .plan
                .ports
                .get(ordinal)
                .is_some_and(|spec| spec.owned_input);
        let Some(Placement::Upload(offset)) =
            owned.then(|| self.plan.port_placements[ordinal]).flatten()
        else {
            return Err(TensorError::Execution(ExecutionError::SubmissionFailed(
                "write_input requires an owned native graph input port".to_owned(),
            )));
        };
        let upload = self.upload.as_ref().expect("planned upload region absent");
        self.plan
            .port_view(ordinal, upload, offset)
            .write_from_host(bytes)
    }

    /// Check the run's external bindings and output lease. Nodes were
    /// validated at seal; only bindings set after [`NativeGraphPlan::bind_static`]
    /// are checked against their port here.
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
        let owned = [&self.workspace, &self.upload, &exports.arena]
            .into_iter()
            .flatten()
            .map(|allocation| allocation.identity())
            .collect::<Vec<_>>();
        for (port, spec) in self.plan.ports.iter().enumerate() {
            if spec.local {
                continue;
            }
            let tensor = bindings.inputs[port].as_ref().ok_or(CallError::Workflow(
                WorkflowError::NativePortUnbound { port },
            ))?;
            // An external binding never names this run's own storage: the
            // plan placed every graph buffer apart from external tensors.
            if (!bindings.checked[port] && !port_matches(&self.plan.device, spec, tensor))
                || (self.plan.executable.external_writes[port]
                    && tensor.allocation().storage().read_only())
                || owned.contains(&tensor.allocation().identity())
            {
                return Err(CallError::Workflow(WorkflowError::NativePortMismatch {
                    port,
                }));
            }
        }
        for (first, second) in &self.plan.executable.disjoint {
            let range = |range: &PortRange| {
                let tensor = bindings.inputs[range.port]
                    .as_ref()
                    .expect("every external port was bound above");
                let start = tensor.byte_offset() + range.offset;
                (tensor.allocation().identity(), start, start + range.bytes)
            };
            let (first_allocation, first_start, first_end) = range(first);
            let (second_allocation, second_start, second_end) = range(second);
            if first_allocation == second_allocation
                && first.bytes != 0
                && second.bytes != 0
                && first_start < second_end
                && second_start < first_end
            {
                return Err(CallError::Invocation(InvocationError::IllegalAliasing {
                    first: first.parameter.clone(),
                    second: second.parameter.clone(),
                }));
            }
        }
        Ok(ReadyNativeGraphRun {
            slot: self,
            externals: bindings.inputs,
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
    /// Ports whose binding was checked by `bind_static`.
    checked: Vec<bool>,
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
    slot: &'a mut NativeGraphSlot,
    /// The tensor of every external port (`None` for graph locals).
    externals: Vec<Option<Arc<TensorInner>>>,
    exports: NativeGraphOutputs,
}

impl ReadyNativeGraphRun<'_> {
    /// Submit the already checked attachments without waiting, as a
    /// sequence of this one run. Missing ports, wrong extents, and illegal
    /// aliases cannot enter this method.
    ///
    /// The returned outputs may be bound into later submissions at once;
    /// the device queue orders them. Host reads of exported tensors wait for
    /// this submission. The completion reports its outcome.
    pub fn submit(self) -> Result<(NativeGraphOutputs, NativeGraphCompletion), CallError> {
        let mut sequence = NativeGraphSequence::new(&self.slot.plan.device);
        let outputs = self.queue(&mut sequence)?;
        Ok((outputs, sequence.submit()?))
    }

    /// Append this run to `sequence`, which submits every run queued on it
    /// as one unit of device work, in queue order. The returned outputs may
    /// be bound into runs queued after it at once. The run's storage stays
    /// held by the sequence until it is submitted; host access to it before
    /// then does not wait for the queued work.
    pub fn queue(
        self,
        sequence: &mut NativeGraphSequence,
    ) -> Result<NativeGraphOutputs, CallError> {
        let plan = self.slot.plan.clone();
        if !Arc::ptr_eq(&plan.device, &sequence.device) {
            return Err(CallError::Workflow(WorkflowError::NativeGraphSlotMismatch));
        }
        let externals = self
            .externals
            .iter()
            .map(|tensor| {
                tensor
                    .as_ref()
                    .map(|tensor| (tensor.allocation().clone(), tensor.byte_offset()))
            })
            .collect::<Vec<_>>();
        sequence.access.extend(
            externals
                .iter()
                .zip(&plan.executable.external_writes)
                .filter_map(|(binding, write)| {
                    binding
                        .as_ref()
                        .map(|(allocation, _)| (allocation.clone(), *write))
                })
                .chain(
                    self.slot
                        .workspace
                        .iter()
                        .map(|allocation| (allocation.clone(), true)),
                )
                .chain(
                    self.slot
                        .upload
                        .iter()
                        .map(|allocation| (allocation.clone(), plan.executable.upload_written)),
                )
                .chain(
                    self.exports
                        .arena
                        .iter()
                        .map(|allocation| (allocation.clone(), true)),
                ),
        );
        sequence.runs.push(QueuedRun {
            plan,
            externals,
            workspace: self.slot.workspace.clone(),
            upload: self.slot.upload.clone(),
            outputs: self.exports.arena.clone(),
        });
        let mut exports = self.exports;
        exports.submitted = true;
        Ok(exports)
    }
}

/// Attached graph runs queued for one submission: on Metal one command
/// buffer, on CUDA one stream submission (one graph launch once seen). The
/// device executes the runs in queue order, each after the previous one's
/// writes, exactly as separately submitted runs; the sequence only removes
/// the submission boundaries between them.
///
/// Runs hold their storage (not their tensor handles), so an output lease
/// whose tensors a queued run reads may be recycled and bound again by a
/// later run of the same sequence; the queue orders the reuse. An upload
/// region a queued run reads is not idle until the sequence's work
/// completes, so no later activation takes it.
pub struct NativeGraphSequence {
    device: Arc<DeviceInner>,
    runs: Vec<QueuedRun>,
    /// Every allocation the queued runs touch, with whether they write it.
    access: Vec<(Arc<Allocation>, bool)>,
}

impl NativeGraphSequence {
    pub fn new(device: &Arc<DeviceInner>) -> Self {
        Self {
            device: device.clone(),
            runs: Vec::new(),
            access: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.runs.is_empty()
    }

    /// Submit every queued run without waiting. The completion reports the
    /// outcome of all of them.
    pub fn submit(self) -> Result<NativeGraphCompletion, CallError> {
        let access = merge_access(self.access.iter().cloned());
        let retained: Arc<dyn std::any::Any + Send + Sync> = Arc::new(
            self.runs
                .iter()
                .map(|run| run.plan.clone())
                .collect::<Vec<_>>(),
        );
        let nodes = self
            .runs
            .iter()
            .enumerate()
            .flat_map(|(run, queued)| {
                (0..queued.plan.executable.nodes.len()).map(move |node| (run, node))
            })
            .collect::<Vec<_>>();
        let list = SequenceList {
            runs: &self.runs,
            nodes,
        };
        let submission = submit(&list, &access, retained, 1)?;
        Ok(NativeGraphCompletion { submission })
    }
}

/// One queued run: its plan's sealed nodes over the storage it binds.
struct QueuedRun {
    plan: Arc<NativeGraphPlan>,
    /// Per port, the allocation and byte offset of the bound tensor (`None`
    /// for graph locals).
    externals: Vec<Option<(Arc<Allocation>, u64)>>,
    workspace: Option<Arc<Allocation>>,
    upload: Option<Arc<Allocation>>,
    outputs: Option<Arc<Allocation>>,
}

/// A sequence's runs as the encoder reads them: every node of every run.
struct SequenceList<'a> {
    runs: &'a [QueuedRun],
    /// (run, node) of each dispatch, in order.
    nodes: Vec<(usize, usize)>,
}

impl DispatchList for SequenceList<'_> {
    fn count(&self) -> usize {
        self.nodes.len()
    }

    fn dispatch<'s>(
        &'s self,
        index: usize,
        buffers: &mut Vec<(&'s Allocation, u64)>,
    ) -> Dispatch<'s> {
        let (run, node) = self.nodes[index];
        let run = &self.runs[run];
        let node = &run.plan.executable.nodes[node];
        let region = |region: &'s Option<Arc<Allocation>>| {
            &**region
                .as_ref()
                .expect("a planned graph storage region is present")
        };
        buffers.extend(node.sites.iter().map(|site| match site.region {
            Region::External(port) => {
                let (allocation, offset) = run.externals[port]
                    .as_ref()
                    .expect("attach bound every external port");
                (&**allocation, offset + site.offset)
            }
            Region::Workspace => (region(&run.workspace), site.offset),
            Region::Upload => (region(&run.upload), site.offset),
            Region::Outputs => (region(&run.outputs), site.offset),
        }));
        Dispatch {
            kernel: &node.kernel,
            words: &node.words,
            word_bytes: &node.word_bytes,
            launches: &node.launches,
            representations: &node.representations,
        }
    }

    fn plans(&self) -> Option<Vec<u64>> {
        Some(self.runs.iter().map(|run| run.plan.identity).collect())
    }
}

/// The outcome of one submitted native graph run.
#[must_use = "a native graph completion reports whether the run succeeded"]
pub struct NativeGraphCompletion {
    submission: NativeSubmission,
}

impl NativeGraphCompletion {
    pub fn is_complete(&self) -> bool {
        self.submission.is_complete()
    }
    /// Wait for the run and report its outcome.
    pub fn wait(self) -> Result<(), CallError> {
        self.submission.wait()
    }
}
