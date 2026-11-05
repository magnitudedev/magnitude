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

use super::hazard::{Reach, Through};
use super::point::{GraphPoint, GraphTiming};
use super::spans::{NativeSpans, Stamp};
use super::timeline::{TimelineAction, TimelineInner, TimelineNode, TimelineOperation};
use super::{
    merge_access, submit, CallLaunches, Dispatch, DispatchList, LaunchGeometry, NativeQueue,
    NativePrepared, NativeRoute, NativeSubmission, NativeTensorSpec, ScratchNeed, BUFFER_ALIGNMENT,
    SYNC_SCRATCH_BYTES,
};
use crate::api::device::DeviceInner;
use crate::api::kernel::{
    EncodedWorkflowArgs, EncodedWorkflowArgument, NativePreparedAny, PendingWorkflowResults,
    ViewOperation, WorkflowResultRef, WorkflowTensorArgument,
};
use crate::api::tensor::RecordPlacement;
use crate::api::tensor::{Records, TensorInner};
use crate::api::{CallError, TensorError, WorkflowError};
use crate::driver::Allocation;
use crate::layout;
use seismic_compiler::errors::{ExecutionError, InvocationError};
use seismic_compiler::graph_cost::{argument_bytes, GraphCensus};
use seismic_compiler::prepared::{ArgumentValue, DeviceIdentity, TensorDescriptor};
use seismic_lang::checked::NativeImplementation;
use seismic_lang::entry::{AliasRule, CallSchema, ParameterKind, TensorAccess};
use seismic_lang::ids::RepresentationId;
use seismic_lang::registry::BackendName;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};

static NEXT_NATIVE_GRAPH: AtomicU64 = AtomicU64::new(1);
const INPUT_NODE: u32 = u32::MAX;

/// Host heap a structure holds: the bytes it asked the allocator for and in
/// how many allocations. The allocator's own rounding and bookkeeping of
/// each allocation are not part of `bytes`; `rounding` bounds them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct HostHeap {
    pub bytes: u64,
    pub allocations: u64,
    /// The most the platform's allocator holds for these allocations beyond
    /// `bytes` (`allocator_rounding`, summed). A bound on what a process
    /// grows by beyond its requested bytes, not part of what is planned.
    pub rounding: u64,
}

/// The most the platform's allocator holds for one allocation of `bytes`
/// requested bytes beyond them, from its size classes.
///
/// macOS (libmalloc): a request is served in 16-byte quanta up to 1,008
/// bytes, in 512-byte quanta up to 32 KiB, and in at most 32 KiB quanta
/// beyond (its medium ranges; a larger one is rounded to pages, which are
/// smaller).
#[cfg(target_os = "macos")]
fn allocator_rounding(bytes: u64) -> u64 {
    match bytes {
        0..=1008 => bytes.next_multiple_of(16).max(16) - bytes,
        1009..=32_768 => bytes.next_multiple_of(512) - bytes,
        _ => 32_768 - 1,
    }
}

/// The most the platform's allocator holds for one allocation of `bytes`
/// requested bytes beyond them, from its size classes.
///
/// glibc: a request below the mapping threshold is a chunk of the request
/// and one size word, in 16-byte quanta and at least 32 bytes. From the
/// threshold's least value (128 KiB) a request may be mapped alone: the
/// request and two size words, rounded to a page.
#[cfg(not(target_os = "macos"))]
fn allocator_rounding(bytes: u64) -> u64 {
    const MAPPING_THRESHOLD: u64 = 128 * 1024;
    const PAGE: u64 = 4096;
    let chunk = (bytes + 8).next_multiple_of(16).max(32) - bytes;
    if bytes < MAPPING_THRESHOLD {
        chunk
    } else {
        chunk.max((bytes + 16).next_multiple_of(PAGE) - bytes)
    }
}

impl HostHeap {
    /// One allocation of `elements` elements of `size` bytes; none when it
    /// is empty.
    pub fn hold(&mut self, elements: usize, size: usize) {
        let bytes = elements
            .checked_mul(size)
            .expect("host allocation bytes overflow");
        if bytes != 0 {
            self.bytes += bytes as u64;
            self.allocations += 1;
            self.rounding += allocator_rounding(bytes as u64);
        }
    }

    /// The heap of one allocation of `elements` elements of `size` bytes.
    pub fn allocation(elements: usize, size: usize) -> Self {
        let mut heap = Self::default();
        heap.hold(elements, size);
        heap
    }

    fn hold_vec<T>(&mut self, vector: &Vec<T>) {
        self.hold(vector.capacity(), std::mem::size_of::<T>());
    }

    /// The heap of `count` structures that each hold `self`.
    pub fn times(self, count: u64) -> Option<Self> {
        Some(Self {
            bytes: self.bytes.checked_mul(count)?,
            allocations: self.allocations.checked_mul(count)?,
            rounding: self.rounding.checked_mul(count)?,
        })
    }

    pub fn plus(self, other: Self) -> Option<Self> {
        Some(Self {
            bytes: self.bytes.checked_add(other.bytes)?,
            allocations: self.allocations.checked_add(other.allocations)?,
            rounding: self.rounding.checked_add(other.rounding)?,
        })
    }
}

/// `vector` in an allocation of exactly its length. A vector grown by
/// pushes and then shrunk in place (`shrink_to_fit`) reports its length as
/// its capacity while the allocator may keep the grown block, so a sealed
/// plan copies each grown vector once into a new allocation of its length:
/// what its host heap counts is then what the allocator holds.
fn exact<T>(vector: Vec<T>) -> Vec<T> {
    let mut exact = Vec::with_capacity(vector.len());
    exact.extend(vector);
    exact
}

/// The lengths of one sealed node's vectors. Its entry's call schema and its
/// backend's native declaration fix them, whatever the call's extents and
/// the kernel's tuning: a graph's host heap is known from them before any
/// kernel is formed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SealedNodeShape {
    /// Buffers in ABI order: tensor parameters, tensor results, scratch.
    buffers: usize,
    /// The buffers the kernel binds: all but its host ports.
    arguments: usize,
    /// Argument words.
    words: usize,
    /// Declared launches, active or not.
    launches: usize,
    /// Per pair of parameters the entry requires disjoint: their ordinals.
    disjoint: Vec<[usize; 2]>,
    /// Whether a call parks (`CallSchema::parks`), so a park node precedes
    /// it.
    parks: bool,
    /// What a graph recording the node holds of it: every launch it
    /// declares, active or not, with its arguments in its backend's layout.
    census: GraphCensus,
}

impl SealedNodeShape {
    pub fn of(schema: &CallSchema, implementation: &NativeImplementation) -> Self {
        let parameter = |id| {
            schema
                .parameters()
                .iter()
                .position(|parameter| parameter.id == id)
                .expect("alias rule names a schema parameter")
        };
        let arguments = super::abi::buffer_slots(schema, implementation) - 2;
        let words = super::abi::native_word_count(schema, implementation);
        let launch_arguments = argument_bytes(implementation.backend, arguments as u64, 8 * words as u64);
        let census = implementation
            .launches
            .iter()
            .map(|_| GraphCensus::launch(launch_arguments))
            .try_fold(GraphCensus::default(), GraphCensus::plus)
            .expect("a node's launch census fits u64");
        Self {
            buffers: sealed_buffers(schema, implementation),
            arguments,
            words,
            launches: implementation.launches.len(),
            census,
            disjoint: schema
                .aliases()
                .iter()
                .filter_map(|rule| match *rule {
                    AliasRule::Disjoint(first, second) => {
                        Some([parameter(first), parameter(second)])
                    }
                    _ => None,
                })
                .collect(),
            parks: schema.parks(),
        }
    }
}

/// The buffers a sealed node places: its entry's tensor parameters (host
/// ports included) and results and its declaration's scratch.
fn sealed_buffers(schema: &CallSchema, implementation: &NativeImplementation) -> usize {
    super::abi::buffer_writes(schema, implementation).len()
}

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
    condition: Option<NodeCondition>,
    share: Option<(u32, WorkflowResultRef)>,
}

/// A node condition (`NativeGraphDraft::enqueue_under`): the launch
/// condition every launch of the node carries, beside those its kernel
/// declares (`NativeLaunch::conditions`): word `index` of the i32 tensor of
/// port `port`, which the host writes, read as a run is queued, the node's
/// launches encoded while it is nonzero (`WhileSet`) or zero (`WhileClear`).
/// A graph names it where the kernel has no such port: a streamed layer's
/// resident combine and its site path under one policy word, each kernel
/// binding only what it reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeCondition {
    pub port: NativePort,
    pub index: u64,
    pub encoded: seismic_lang::checked::Encoded,
}

/// A node result a run writes into the tensor bound to an external port
/// (`NativeGraphDraft::store`) rather than into graph storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Store {
    result: WorkflowResultRef,
    port: u32,
}

/// Declare the external or local tensor `spec`, the graph's next port.
fn add_port(identity: u64, ports: &mut Vec<PortSpec>, spec: PortSpec) -> NativePort {
    let result = u32::try_from(ports.len()).expect("native port ordinal exhausted");
    ports.push(spec);
    NativePort {
        reference: WorkflowResultRef {
            workflow: identity,
            node: INPUT_NODE,
            result,
        },
    }
}

/// Add timeline node `node` after the `nodes` enqueued so far.
fn add_timeline(
    identity: u64,
    timelines: &mut Vec<(usize, TimelineNode)>,
    nodes: usize,
    node: TimelineNode,
) -> NativeTimelinePort {
    let ordinal = u32::try_from(timelines.len()).expect("timeline ordinal exhausted");
    timelines.push((nodes, node));
    NativeTimelinePort {
        workflow: identity,
        ordinal,
    }
}

/// Export `reference`, once however often it is asked for.
fn add_export(
    identity: u64,
    ports: &[PortSpec],
    nodes: usize,
    exports: &mut Vec<WorkflowResultRef>,
    reference: WorkflowResultRef,
) -> Result<(), WorkflowError> {
    validate_export_reference(identity, ports, nodes, reference)?;
    if !exports.contains(&reference) {
        exports.push(reference);
    }
    Ok(())
}

/// Record that `result`, a result of one of the `nodes` enqueued so far, is
/// written into the tensor bound to the external port `port`: each such port
/// takes one result, and each result goes to one port.
fn add_store(
    identity: u64,
    ports: &[PortSpec],
    nodes: usize,
    stores: &mut Vec<Store>,
    result: WorkflowResultRef,
    port: NativePort,
) -> Result<(), WorkflowError> {
    let ordinal = port.reference.result;
    if result.workflow != identity
        || result.node == INPUT_NODE
        || result.node as usize >= nodes
        || port.reference.workflow != identity
        || port.reference.node != INPUT_NODE
    {
        return Err(WorkflowError::CrossWorkflowResult);
    }
    if ports.get(ordinal as usize).is_none_or(|spec| spec.local)
        || stores
            .iter()
            .any(|store| store.port == ordinal || store.result == result)
    {
        return Err(WorkflowError::NativePortMismatch {
            port: ordinal as usize,
        });
    }
    stores.push(Store {
        result,
        port: ordinal,
    });
    Ok(())
}

/// Every stored result is the exact tensor its port describes, and none is
/// exported: the run's binding of the port is where it lives.
fn validate_stores(
    stores: &[Store],
    ports: &[PortSpec],
    results: &[Vec<Option<NativeTensorSpec>>],
    exports: &[WorkflowResultRef],
) -> Result<(), WorkflowError> {
    for store in stores {
        let port = &ports[store.port as usize];
        let exact = result_spec(results, store.result).is_some_and(|result| {
            result.representation == port.representation
                && result.extents == port.extents
                && result.strides == port.strides
                && result.byte_len == port.byte_len
        });
        if !exact || exports.contains(&store.result) {
            return Err(WorkflowError::NativePortMismatch {
                port: store.port as usize,
            });
        }
    }
    Ok(())
}

/// That the timeline nodes before node `node` (or the graph's end, `node`
/// the node count) end with one park exactly when the node parks
/// (`TimelineNode::Park`): a park answers the call right after it, and
/// every call that parks is answered by one; and that a publication
/// (`TimelineNode::Publish`) follows a call, which it publishes.
fn validate_park(timelines: &[(usize, TimelineNode)], node: usize, parks: bool) -> Result<(), WorkflowError> {
    let before = timelines.iter().rev().take_while(|&&(position, _)| position == node);
    if node == 0 && before.clone().any(|&(_, kind)| kind == TimelineNode::Publish) {
        return Err(WorkflowError::PublishUnpreceded);
    }
    let parked = before.clone().filter(|&&(_, kind)| kind == TimelineNode::Park).count();
    let last_parks = before.clone().next().is_some_and(|&(_, kind)| kind == TimelineNode::Park);
    if parked == usize::from(parks) && (!parks || last_parks) {
        Ok(())
    } else {
        Err(WorkflowError::ParkUnpaired { node })
    }
}

pub struct NativeGraphDraft {
    identity: u64,
    device: Arc<DeviceInner>,
    ports: Vec<PortSpec>,
    nodes: Vec<NodeDraft>,
    exports: Vec<WorkflowResultRef>,
    stores: Vec<Store>,
    layout: Option<Arc<NativeGraphLayout>>,
    /// Timeline nodes, each before the kernel node of its position.
    timelines: Vec<(usize, TimelineNode)>,
    /// Span marks (`span_mark`), each before the kernel node of its
    /// position.
    marks: Vec<usize>,
}

/// A timeline node of a graph (`timeline`): each run binds it to a
/// timeline and a value (`NativeGraphBindings::set_timeline`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeTimelinePort {
    workflow: u64,
    ordinal: u32,
}

impl NativeGraphDraft {
    pub fn new(device: &Arc<DeviceInner>) -> Self {
        Self {
            identity: NEXT_NATIVE_GRAPH.fetch_add(1, Ordering::Relaxed),
            device: device.clone(),
            ports: Vec::new(),
            nodes: Vec::new(),
            exports: Vec::new(),
            stores: Vec::new(),
            layout: None,
            timelines: Vec::new(),
            marks: Vec::new(),
        }
    }

    /// A span mark after every node enqueued so far, its ordinal returned: a
    /// boundary of the plan's calls where a sampled run stamps the device's
    /// time (`spans`), so the device time between two marks is the time of
    /// the calls between them. It is a node inside the graph its calls
    /// replay (`spans`), dividing nothing. Every plan of one layout marks
    /// alike (its metadata topology's `span_mark`).
    pub fn span_mark(&mut self) -> usize {
        self.marks.push(self.nodes.len());
        self.marks.len() - 1
    }

    /// A timeline node after every node enqueued so far: a host gate the
    /// device waits at (`TimelineNode::Wait`), or a signal the host waits
    /// for (`TimelineNode::Signal`).
    pub fn timeline(&mut self, node: TimelineNode) -> NativeTimelinePort {
        add_timeline(self.identity, &mut self.timelines, self.nodes.len(), node)
    }

    pub fn with_layout(device: &Arc<DeviceInner>, layout: Arc<NativeGraphLayout>) -> Self {
        let mut draft = Self::new(device);
        draft.layout = Some(layout);
        draft
    }

    /// Declare one externally owned tensor from artifact/model dimensions.
    /// Every use is checked against the generated entry that consumes it.
    pub fn port(
        &mut self,
        representation: RepresentationId,
        extents: &[u64],
    ) -> Result<NativePort, TensorError> {
        let layout = layout::canonical(representation, extents)?;
        Ok(add_port(self.identity, &mut self.ports, PortSpec {
            representation,
            extents: extents.to_vec(),
            strides: layout.strides,
            byte_len: layout.byte_len,
            local: false,
            owned_input: false,
            prewritten: false,
        }))
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
        self.checked_port(kernel, parameter, dimensions, true, false)
    }

    pub fn input_for(
        &mut self,
        kernel: &Arc<NativePreparedAny>,
        parameter: &str,
        dimensions: &[(&str, u64)],
    ) -> Result<NativePort, CallError> {
        self.checked_port(kernel, parameter, dimensions, true, true)
    }

    /// Declare an externally owned tensor from a checked parameter: each
    /// run binds it, as a [`Self::port`], in the layout the entry contract
    /// derives from the dimension values.
    pub fn port_for(
        &mut self,
        kernel: &Arc<NativePreparedAny>,
        parameter: &str,
        dimensions: &[(&str, u64)],
    ) -> Result<NativePort, CallError> {
        self.checked_port(kernel, parameter, dimensions, false, false)
    }

    fn checked_port(
        &mut self,
        kernel: &Arc<NativePreparedAny>,
        parameter: &str,
        dimensions: &[(&str, u64)],
        local: bool,
        owned_input: bool,
    ) -> Result<NativePort, CallError> {
        let spec = kernel.inner.tensor_parameter_spec(parameter, dimensions)?;
        Ok(add_port(self.identity, &mut self.ports, PortSpec {
            representation: spec.representation,
            extents: spec.extents,
            strides: spec.strides,
            byte_len: spec.byte_len,
            local,
            owned_input,
            prewritten: false,
        }))
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
        self.enqueue_node(kernel, args, None)
    }

    /// [`Self::enqueue`] under `condition`: the node's launches are encoded
    /// in a run only while its word holds what the condition encodes under
    /// (a park before it then waits only where they are). Its word is one
    /// of an external i32 port's.
    pub fn enqueue_under(
        &mut self,
        kernel: &Arc<NativePreparedAny>,
        args: EncodedWorkflowArgs,
        condition: NodeCondition,
    ) -> Result<PendingWorkflowResults, WorkflowError> {
        condition_port(self.identity, &self.ports, self.nodes.len(), condition)?;
        self.enqueue_node(kernel, args, Some(condition))
    }

    fn enqueue_node(
        &mut self,
        kernel: &Arc<NativePreparedAny>,
        args: EncodedWorkflowArgs,
        condition: Option<NodeCondition>,
    ) -> Result<PendingWorkflowResults, WorkflowError> {
        let node = u32::try_from(self.nodes.len()).expect("native graph node ordinal exhausted");
        validate_node_references(self.identity, self.ports.len(), node, &args)?;
        validate_park(&self.timelines, self.nodes.len(), kernel.inner.schema().parks())?;
        // The checked result count is supplied by the native handle; no
        // caller-authored result list is allowed here.
        let count = kernel.inner.result_count();
        self.nodes.push(NodeDraft {
            kernel: kernel.clone(),
            args,
            condition,
            share: None,
        });
        Ok(PendingWorkflowResults::new(self.identity, node, count))
    }

    pub fn export(&mut self, reference: WorkflowResultRef) -> Result<(), WorkflowError> {
        add_export(self.identity, &self.ports, self.nodes.len(), &mut self.exports, reference)
    }

    /// Place node result `other` where node result `primary` lies, one
    /// storage for both: the nodes' conditions are exclusive (one word,
    /// opposite encodings: `NodeCondition`), so each run has exactly one
    /// writer, and every reader reads `primary`. Sealing proves it
    /// (`validate_shares`): of results of one description, `other` read by
    /// no node, export or store.
    pub fn share(&mut self, other: WorkflowResultRef, primary: WorkflowResultRef) -> Result<(), WorkflowError> {
        let node = (other.workflow == self.identity && primary.workflow == self.identity)
            .then(|| self.nodes.get_mut(other.node as usize))
            .flatten()
            .filter(|node| node.share.is_none())
            .ok_or(WorkflowError::SharedResult { node: other.node as usize })?;
        node.share = Some((other.result, primary));
        Ok(())
    }

    /// Write the node result `result` into the tensor each run binds to the
    /// external port `port`, which describes exactly that result, instead of
    /// into graph storage: a node writes a tensor the graph does not own.
    pub fn store(&mut self, result: WorkflowResultRef, port: NativePort) -> Result<(), WorkflowError> {
        add_store(
            self.identity,
            &self.ports,
            self.nodes.len(),
            &mut self.stores,
            result,
            port,
        )
    }

    /// Seal the graph as a plan whose runs replay graphs: on CUDA graphs a
    /// slot of it instantiates as it is created, between flights.
    pub fn seal(self) -> Result<NativeGraphPlan, CallError> {
        self.sealed(false)
    }

    /// Seal the graph as a launched plan: per-use work whose shapes no load
    /// can enumerate ahead (rows a request brings), so no slot instantiates
    /// a graph of it. On CUDA, which replays only graphs slots instantiated,
    /// its runs are launched call by call in their submission's stream
    /// order, and a slot of it holds no graph, so it may be created with
    /// work in flight; elsewhere its graphs are recorded at a shape's first
    /// use, as every plan's are.
    pub fn seal_launched(self) -> Result<NativeGraphPlan, CallError> {
        self.sealed(true)
    }

    fn sealed(mut self, launched: bool) -> Result<NativeGraphPlan, CallError> {
        if self.nodes.is_empty() {
            return Err(CallError::Workflow(WorkflowError::Empty));
        }
        validate_park(&self.timelines, self.nodes.len(), false).map_err(CallError::Workflow)?;
        validate_used_ports(
            self.ports.len(),
            self.nodes.iter().map(|node| &node.args),
            &self.exports,
            &self.stores,
        )
        .map_err(CallError::Workflow)?;
        // A sealed plan's vectors hold exactly their lengths, so its host
        // heap follows from its topology (`NativeGraphPlan::host_heap`,
        // `NativeGraphMetadataDraft::plan_heap`).
        self.ports = exact(self.ports);
        for port in &mut self.ports {
            port.extents = exact(std::mem::take(&mut port.extents));
            port.strides = exact(std::mem::take(&mut port.strides));
        }
        self.exports = exact(self.exports);
        let identity = self.identity;
        let device_identity = self.device.kind.identity();
        let mut results: Vec<Vec<Option<NativeTensorSpec>>> = Vec::with_capacity(self.nodes.len());
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
                condition: node
                    .condition
                    .map(|condition| (condition.port.reference.result, condition.index, condition.encoded)),
                share: node.share,
            });
            shaped.push((node.kernel.inner.clone(), arguments, shape));
        }
        for reference in &self.exports {
            if reference.node != INPUT_NODE && result_spec(&results, *reference).is_none() {
                return Err(CallError::Workflow(WorkflowError::MissingProducerResult));
            }
        }
        validate_stores(&self.stores, &self.ports, &results, &self.exports)
            .map_err(CallError::Workflow)?;
        validate_shares(&planned, &results, &self.exports, &self.stores).map_err(CallError::Workflow)?;
        let recorded = recorded_census(
            shaped
                .iter()
                .map(|(kernel, _, _)| SealedNodeShape::of(kernel.schema(), kernel.implementation()).census),
            self.timelines.len(),
        );
        let storage = if let Some(layout) = &self.layout {
            layout
                .storage_for(&self.ports, &planned, &results, &self.exports, &self.stores)
                .map_err(CallError::Workflow)?
        } else {
            plan_storage(&self.ports, &planned, &results, &self.exports, &self.stores)
        };
        let mut executable = Executable {
            nodes: Vec::with_capacity(planned.len()),
            external_writes: vec![false; self.ports.len()],
            port_uses: vec![PortUse::default(); self.ports.len()],
            upload_written: false,
            disjoint: Vec::new(),
            timelines: exact(self.timelines),
            marks: exact(self.marks),
            workspace_records: 0,
        };
        for (node, ((kernel, arguments, shape), planned)) in
            shaped.into_iter().zip(&planned).enumerate()
        {
            executable
                .seal_node(kernel, &arguments, shape, planned, &storage, &self.ports, node)
                .map_err(CallError::Workflow)?;
        }
        executable.disjoint = exact(executable.disjoint);
        let shape = PlanShape::of(identity, self.layout.as_deref(), &executable);
        Ok(NativeGraphPlan {
            identity,
            shape,
            device: self.device,
            ports: self.ports,
            executable,
            results,
            exports: self.exports,
            port_placements: storage.port_placements,
            placements: storage.placements,
            scratch_bytes: storage.scratch_bytes,
            scratch_floor_bytes: storage.scratch_floor_bytes,
            output_bytes: storage.output_bytes,
            upload_bytes: storage.upload_bytes,
            recorded,
            launched,
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
    /// Per node, the lengths of its sealed form's vectors.
    shapes: Vec<SealedNodeShape>,
    results: Vec<Vec<Option<NativeTensorSpec>>>,
    exports: Vec<WorkflowResultRef>,
    stores: Vec<Store>,
    timelines: Vec<(usize, TimelineNode)>,
    marks: Vec<usize>,
}

/// A Seismic placement for a dominating graph class. Exact graphs with the
/// same edges and smaller checked buffers can seal directly into its offsets.
pub struct NativeGraphLayout {
    /// What plans sealed into it share as their shape (`NativeGraphPlan::shape`).
    identity: u64,
    storage: StoragePlan,
    signature: StorageSignature,
    port_capacity: Vec<u64>,
    result_capacity: Vec<Vec<Option<u64>>>,
    scratch_capacity: Vec<Vec<ScratchNeed>>,
    /// The host heap of a plan sealed into this layout
    /// (`NativeGraphMetadataDraft::plan_heap` of the layout's topology).
    plan_heap: HostHeap,
    /// The topology's timeline nodes.
    timelines: usize,
    /// What recording one run of a plan sealed into it takes.
    recorded: GraphCensus,
}

impl NativeGraphLayout {
    /// The host heap of each plan sealed into this layout: every vector of
    /// a sealed plan has a length the layout's topology fixes, so its
    /// classes' plans hold the same heap. `NativeGraphPlan::host_heap` is
    /// what a sealed plan holds.
    pub fn plan_heap(&self) -> HostHeap {
        self.plan_heap
    }

    /// The host heap of one static binding of a plan sealed into this
    /// layout (`BoundNativeGraphPlan::host_heap` is what a binding holds):
    /// a binding holds, for each port of the plan, the tensor bound to it
    /// and whether it is external and checked, and for each timeline node
    /// its timeline and value.
    pub fn bound_heap(&self) -> HostHeap {
        bindings_heap(self.port_capacity.len(), self.timelines)
    }

    /// What a plan sealed into this layout holds: every class of its
    /// topology records the same launches ([`SealedNodeShape`]).
    pub fn storage_bytes(&self) -> NativeGraphStorageBytes {
        NativeGraphStorageBytes {
            workspace: self.storage.scratch_bytes,
            output: self.storage.output_bytes,
            upload: self.storage.upload_bytes,
            recorded: self.recorded,
        }
    }

    fn storage_for(
        &self,
        ports: &[PortSpec],
        nodes: &[PlannedNode],
        results: &[Vec<Option<NativeTensorSpec>>],
        exports: &[WorkflowResultRef],
        stores: &[Store],
    ) -> Result<StoragePlan, WorkflowError> {
        let valid = storage_signature(ports, nodes, results, exports, stores) == self.signature
            && ports.len() == self.port_capacity.len()
            && ports
                .iter()
                .zip(&self.port_capacity)
                .all(|(port, capacity)| port.byte_len <= *capacity)
            && results.len() == self.result_capacity.len()
            && results
                .iter()
                .zip(&self.result_capacity)
                .all(|(row, capacities)| {
                    row.len() == capacities.len()
                        && row.iter().zip(capacities).all(|(result, capacity)| {
                            match (result, capacity) {
                                (Some(result), Some(capacity)) => result.byte_len <= *capacity,
                                (None, None) => true,
                                _ => false,
                            }
                        })
                })
            && nodes.len() == self.scratch_capacity.len()
            && nodes
                .iter()
                .zip(&self.scratch_capacity)
                .all(|(node, capacities)| {
                    node.scratch.len() == capacities.len()
                        && node.scratch.iter().zip(capacities).all(|(need, capacity)| {
                            need.sync == capacity.sync && need.bytes <= capacity.bytes
                        })
                });
        if !valid {
            return Err(WorkflowError::NativeGraphLayoutMismatch);
        }
        Ok(self.storage.clone())
    }
}

#[derive(Clone, PartialEq, Eq)]
struct StorageSignature {
    ports: Vec<(RepresentationId, bool, bool, bool)>,
    results: Vec<Vec<Option<RepresentationId>>>,
    edges: Vec<Vec<Option<(u32, u32)>>>,
    exports: Vec<(u32, u32)>,
    /// Each stored result's (node, result, port).
    stores: Vec<(u32, u32, u32)>,
    /// Per node, its condition and its shared result (ordinal, the result it
    /// shares).
    conditions: Vec<(Option<Condition>, Option<(u32, u32, u32)>)>,
}

fn storage_signature(
    ports: &[PortSpec],
    nodes: &[PlannedNode],
    results: &[Vec<Option<NativeTensorSpec>>],
    exports: &[WorkflowResultRef],
    stores: &[Store],
) -> StorageSignature {
    StorageSignature {
        ports: ports
            .iter()
            .map(|port| {
                (
                    port.representation,
                    port.local,
                    port.owned_input,
                    port.prewritten,
                )
            })
            .collect(),
        results: results
            .iter()
            .map(|row| {
                row.iter()
                    .map(|result| result.as_ref().map(|result| result.representation))
                    .collect()
            })
            .collect(),
        edges: nodes
            .iter()
            .map(|node| {
                node.args
                    .arguments()
                    .iter()
                    .map(|argument| {
                        argument_reference(argument)
                            .map(|reference| (reference.node, reference.result))
                    })
                    .collect()
            })
            .collect(),
        exports: exports
            .iter()
            .map(|reference| (reference.node, reference.result))
            .collect(),
        stores: stores
            .iter()
            .map(|store| (store.result.node, store.result.result, store.port))
            .collect(),
        conditions: nodes
            .iter()
            .map(|node| (node.condition, node.share.map(|(ordinal, primary)| (ordinal, primary.node, primary.result))))
            .collect(),
    }
}

/// What a graph holds while it runs and once it is recorded: its storage
/// (workspace, output arena, upload region) and what recording one run of
/// it may take ([`GraphCensus`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NativeGraphStorageBytes {
    pub workspace: u64,
    pub output: u64,
    pub upload: u64,
    pub recorded: GraphCensus,
}

impl NativeGraphStorageBytes {
    /// What one run of either graph holds at most: each term's larger.
    pub fn most(self, other: Self) -> Self {
        Self {
            workspace: self.workspace.max(other.workspace),
            output: self.output.max(other.output),
            upload: self.upload.max(other.upload),
            recorded: self.recorded.most(other.recorded),
        }
    }
}

/// What a plan's graphs are keyed by beside the storage its runs bind
/// (`ReplayKeys`).
///
/// - A CUDA graph's launches are parameters, its function, geometry, words
///   and addresses set in place and its launches enabled or not
///   (`DirectGraph::bind`): its shape is its layout's, which every class
///   and block sealed into the layout shares, with a node for every launch
///   any plan of the layout its slot runs makes (`super::LayoutForms`, of
///   each plan's `structure`).
/// - A Vulkan graph records each launch's pipeline and groups, and leaves
///   out the launches its run does not encode: its shape is its plan's, a
///   graph for each encoding a run of it makes.
/// - A Metal schedule is a submission's, of its plans.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct PlanShape {
    /// The layout's identity, or the plan's where it was sealed alone.
    pub(crate) layout: u64,
    pub(crate) identity: u64,
    /// CUDA: the launches each node makes, as `launch_structure` writes
    /// them; empty elsewhere.
    pub(crate) structure: Vec<u64>,
}

impl PlanShape {
    fn of(identity: u64, layout: Option<&NativeGraphLayout>, executable: &Executable) -> Self {
        let mut structure = Vec::new();
        for node in &executable.nodes {
            if let NativeRoute::Cuda { .. } = &node.kernel.route {
                super::cuda::launch_structure(&node.launches.geometry, &mut structure);
            }
        }
        Self {
            layout: layout.map_or(identity, |layout| layout.identity),
            identity,
            structure,
        }
    }

    /// The first word of its graphs' keys on `backend` (`ReplayKeys::key`),
    /// which the device counts its graphs by (`Replays::plan_counts`): its
    /// layout's on CUDA, its own elsewhere.
    pub(crate) fn keyed(&self, backend: BackendName) -> u64 {
        match backend {
            BackendName::Cuda => self.layout,
            BackendName::Vulkan | BackendName::Metal | BackendName::Cpu => self.identity,
        }
    }
}

/// What recording one run of a graph of `shapes`' nodes and `timelines`
/// timeline nodes takes: every launch its nodes declare, in a graph of the
/// run's own (a run's calls are a graph of their shape, `ReplayKeys`), and a
/// graph more for each timeline node, which ends the graph the calls before
/// it are recorded in whether or not a run binds it (`graph_segments`).
/// Every run of the plan, on any slot or place, replays these graphs.
fn recorded_census(shapes: impl IntoIterator<Item = GraphCensus>, timelines: usize) -> GraphCensus {
    let launches = shapes
        .into_iter()
        .try_fold(GraphCensus::default(), GraphCensus::plus)
        .expect("a graph's launch census fits u64");
    GraphCensus { graphs: 1 + timelines as u64, ..launches }
}

impl NativeGraphMetadataDraft {
    pub fn new() -> Self {
        Self {
            identity: NEXT_NATIVE_GRAPH.fetch_add(1, Ordering::Relaxed),
            ports: Vec::new(),
            nodes: Vec::new(),
            shapes: Vec::new(),
            results: Vec::new(),
            exports: Vec::new(),
            stores: Vec::new(),
            timelines: Vec::new(),
            marks: Vec::new(),
        }
    }

    /// A span mark after every node enqueued so far
    /// (`NativeGraphDraft::span_mark`), its ordinal returned.
    pub fn span_mark(&mut self) -> usize {
        self.marks.push(self.nodes.len());
        self.marks.len() - 1
    }

    /// A timeline node after every node enqueued so far
    /// (`NativeGraphDraft::timeline`).
    pub fn timeline(&mut self, node: TimelineNode) -> NativeTimelinePort {
        add_timeline(self.identity, &mut self.timelines, self.nodes.len(), node)
    }

    /// The host heap of the plan this topology seals to
    /// (`NativeGraphPlan::host_heap`), from the lengths of its vectors: the
    /// plan in its shared allocation, its ports, each node's buffers, words
    /// and launches, the external pairs a run checks disjoint, the result
    /// specs, the exports and the placements.
    pub fn plan_heap(&self) -> HostHeap {
        use std::mem::size_of;
        let mut heap = HostHeap::default();
        heap.hold(1, SHARED_PLAN_BYTES);
        heap.hold(self.ports.len(), size_of::<PortSpec>());
        for port in &self.ports {
            heap.hold(port.extents.len(), size_of::<u64>());
            heap.hold(port.strides.len(), size_of::<u64>());
        }
        heap.hold(self.nodes.len(), size_of::<SealedNode>());
        // `Executable::external_writes` and `port_uses`.
        heap.hold(self.ports.len(), size_of::<bool>());
        heap.hold(self.ports.len(), size_of::<PortUse>());
        heap.hold(self.timelines.len(), size_of::<(usize, TimelineNode)>());
        heap.hold(self.marks.len(), size_of::<usize>());
        let mut disjoint = 0usize;
        for (node, shape) in self.nodes.iter().zip(&self.shapes) {
            heap.hold(shape.buffers, size_of::<Site>());
            heap.hold(shape.arguments, size_of::<&'static str>());
            heap.hold(shape.words, size_of::<u64>());
            heap.hold(shape.launches, size_of::<Option<LaunchGeometry>>());
            // A pair is checked per run when both arguments are external
            // ports, and different ones (`Executable::seal_node`).
            let external = |ordinal: usize| {
                argument_reference(&node.args.arguments()[ordinal])
                    .filter(|reference| {
                        reference.node == INPUT_NODE && !self.ports[reference.result as usize].local
                    })
                    .map(|reference| reference.result)
            };
            for [first, second] in &shape.disjoint {
                if let (Some(first), Some(second)) = (external(*first), external(*second)) {
                    if first != second {
                        disjoint += 1;
                    }
                }
            }
        }
        heap.hold(disjoint, size_of::<(PortRange, PortRange)>());
        heap.hold(
            self.results.len(),
            size_of::<Vec<Option<NativeTensorSpec>>>(),
        );
        heap.hold(self.results.len(), size_of::<Vec<Placement>>());
        for row in &self.results {
            heap.hold(row.len(), size_of::<Option<NativeTensorSpec>>());
            heap.hold(row.len(), size_of::<Placement>());
            for spec in row.iter().flatten() {
                heap.hold(spec.extents.len(), size_of::<u64>());
                heap.hold(spec.strides.len(), size_of::<u64>());
            }
        }
        heap.hold(self.exports.len(), size_of::<WorkflowResultRef>());
        heap.hold(self.ports.len(), size_of::<Option<Placement>>());
        heap
    }

    pub fn port(
        &mut self,
        representation: RepresentationId,
        extents: &[u64],
        local: bool,
        owned_input: bool,
    ) -> Result<NativePort, TensorError> {
        let layout = layout::canonical(representation, extents)?;
        Ok(add_port(self.identity, &mut self.ports, PortSpec {
            representation,
            extents: extents.to_vec(),
            strides: layout.strides,
            byte_len: layout.byte_len,
            local,
            owned_input,
            prewritten: false,
        }))
    }

    pub fn prewrite(&mut self, port: NativePort) -> Result<(), WorkflowError> {
        prewrite_port(self.identity, &mut self.ports, port)
    }

    pub fn enqueue(
        &mut self,
        args: EncodedWorkflowArgs,
        parameter_shapes: Vec<Option<(RepresentationId, Vec<u64>)>>,
        result_shapes: Vec<Option<(RepresentationId, Vec<u64>)>>,
        scratch: Vec<ScratchNeed>,
        shape: SealedNodeShape,
    ) -> Result<PendingWorkflowResults, CallError> {
        let node = u32::try_from(self.nodes.len()).expect("native graph node ordinal exhausted");
        validate_node_references(self.identity, self.ports.len(), node, &args)
            .map_err(CallError::Workflow)?;
        validate_metadata_arguments(&args, &parameter_shapes, &self.ports, &self.results)?;
        validate_park(&self.timelines, self.nodes.len(), shape.parks).map_err(CallError::Workflow)?;
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
        self.nodes.push(PlannedNode { args, scratch, condition: None, share: None });
        self.shapes.push(shape);
        self.results.push(results);
        Ok(PendingWorkflowResults::new(self.identity, node, count))
    }

    pub fn export(&mut self, reference: WorkflowResultRef) -> Result<(), WorkflowError> {
        add_export(self.identity, &self.ports, self.nodes.len(), &mut self.exports, reference)
    }

    /// [`Self::enqueue`] under `condition` (`NativeGraphDraft::enqueue_under`):
    /// planning holds the node as any other, its condition kept for the
    /// proof that results may share storage (`Self::share`).
    pub fn enqueue_under(
        &mut self,
        args: EncodedWorkflowArgs,
        parameter_shapes: Vec<Option<(RepresentationId, Vec<u64>)>>,
        result_shapes: Vec<Option<(RepresentationId, Vec<u64>)>>,
        scratch: Vec<ScratchNeed>,
        shape: SealedNodeShape,
        condition: NodeCondition,
    ) -> Result<PendingWorkflowResults, CallError> {
        let port = condition_port(self.identity, &self.ports, self.nodes.len(), condition).map_err(CallError::Workflow)?;
        let pending = self.enqueue(args, parameter_shapes, result_shapes, scratch, shape)?;
        self.nodes.last_mut().expect("the node just enqueued").condition = Some((port, condition.index, condition.encoded));
        Ok(pending)
    }

    /// `NativeGraphDraft::share`.
    pub fn share(&mut self, other: WorkflowResultRef, primary: WorkflowResultRef) -> Result<(), WorkflowError> {
        let node = (other.workflow == self.identity && primary.workflow == self.identity)
            .then(|| self.nodes.get_mut(other.node as usize))
            .flatten()
            .filter(|node| node.share.is_none())
            .ok_or(WorkflowError::SharedResult { node: other.node as usize })?;
        node.share = Some((other.result, primary));
        Ok(())
    }

    /// `NativeGraphDraft::store`.
    pub fn store(&mut self, result: WorkflowResultRef, port: NativePort) -> Result<(), WorkflowError> {
        add_store(
            self.identity,
            &self.ports,
            self.nodes.len(),
            &mut self.stores,
            result,
            port,
        )
    }

    pub fn seal(self) -> Result<NativeGraphStorageBytes, WorkflowError> {
        self.validate_topology()?;
        let storage = plan_storage(
            &self.ports,
            &self.nodes,
            &self.results,
            &self.exports,
            &self.stores,
        );
        Ok(NativeGraphStorageBytes {
            workspace: storage.scratch_bytes,
            output: storage.output_bytes,
            upload: storage.upload_bytes,
            recorded: self.recorded(),
        })
    }

    /// What recording one run of this graph takes ([`recorded_census`]).
    fn recorded(&self) -> GraphCensus {
        recorded_census(self.shapes.iter().map(|shape| shape.census), self.timelines.len())
    }

    /// Verify an exact checked metadata graph against the storage certificate
    /// used by executable preparation, without placing it again.
    pub fn seal_with_layout(
        self,
        layout: &NativeGraphLayout,
    ) -> Result<NativeGraphStorageBytes, WorkflowError> {
        self.validate_topology()?;
        let storage = layout.storage_for(
            &self.ports,
            &self.nodes,
            &self.results,
            &self.exports,
            &self.stores,
        )?;
        Ok(NativeGraphStorageBytes {
            workspace: storage.scratch_bytes,
            output: storage.output_bytes,
            upload: storage.upload_bytes,
            recorded: self.recorded(),
        })
    }

    /// Validate a graph once before using its ordered edges and exports as a
    /// resource template for other exact shape classes.
    pub fn validate_topology(&self) -> Result<(), WorkflowError> {
        if self.nodes.is_empty() {
            return Err(WorkflowError::Empty);
        }
        validate_used_ports(
            self.ports.len(),
            self.nodes.iter().map(|node| &node.args),
            &self.exports,
            &self.stores,
        )?;
        validate_park(&self.timelines, self.nodes.len(), false)?;
        for reference in &self.exports {
            if reference.node != INPUT_NODE && result_spec(&self.results, *reference).is_none() {
                return Err(WorkflowError::MissingProducerResult);
            }
        }
        validate_stores(&self.stores, &self.ports, &self.results, &self.exports)?;
        validate_shares(&self.nodes, &self.results, &self.exports, &self.stores)
    }

    /// Place this validated topology once with every buffer at its supplied
    /// byte capacity: the largest size it takes in any class the layout is
    /// certified for. Placement reads only sizes, lifetimes and regions, so
    /// any exact class of the same topology whose buffers fit these
    /// capacities seals into the resulting offsets.
    pub fn capacity_layout(
        &self,
        ports: &[u64],
        results: &[Vec<u64>],
        scratch: Vec<Vec<ScratchNeed>>,
    ) -> Result<NativeGraphLayout, WorkflowError> {
        self.validate_topology()?;
        let shaped = ports.len() == self.ports.len()
            && results.len() == self.results.len()
            && results
                .iter()
                .zip(&self.results)
                .all(|(capacities, row)| capacities.len() == row.len())
            && scratch.len() == self.nodes.len()
            && scratch
                .iter()
                .zip(&self.nodes)
                .all(|(capacities, node)| capacities.len() == node.scratch.len());
        if !shaped {
            return Err(WorkflowError::NativeGraphLayoutMismatch);
        }
        let ports = self
            .ports
            .iter()
            .zip(ports)
            .map(|(port, &byte_len)| PortSpec {
                byte_len,
                ..port.clone()
            })
            .collect::<Vec<_>>();
        let results = self
            .results
            .iter()
            .zip(results)
            .map(|(row, capacities)| {
                row.iter()
                    .zip(capacities)
                    .map(|(result, &byte_len)| {
                        result.as_ref().map(|result| NativeTensorSpec {
                            byte_len,
                            ..result.clone()
                        })
                    })
                    .collect()
            })
            .collect::<Vec<_>>();
        let nodes = self
            .nodes
            .iter()
            .zip(scratch)
            .map(|(node, scratch)| PlannedNode {
                args: node.args.clone(),
                scratch,
                condition: node.condition,
                share: node.share,
            })
            .collect::<Vec<_>>();
        Ok(make_layout(
            &ports,
            &nodes,
            &results,
            &self.exports,
            &self.stores,
            self.plan_heap(),
            self.timelines.len(),
            self.recorded(),
        ))
    }
}

#[allow(clippy::too_many_arguments)]
fn make_layout(
    ports: &[PortSpec],
    nodes: &[PlannedNode],
    results: &[Vec<Option<NativeTensorSpec>>],
    exports: &[WorkflowResultRef],
    stores: &[Store],
    plan_heap: HostHeap,
    timelines: usize,
    recorded: GraphCensus,
) -> NativeGraphLayout {
    NativeGraphLayout {
        identity: NEXT_NATIVE_GRAPH.fetch_add(1, Ordering::Relaxed),
        plan_heap,
        timelines,
        recorded,
        storage: plan_storage(ports, nodes, results, exports, stores),
        signature: storage_signature(ports, nodes, results, exports, stores),
        port_capacity: ports.iter().map(|port| port.byte_len).collect(),
        result_capacity: results
            .iter()
            .map(|row| {
                row.iter()
                    .map(|result| result.as_ref().map(|result| result.byte_len))
                    .collect()
            })
            .collect(),
        scratch_capacity: nodes.iter().map(|node| node.scratch.clone()).collect(),
    }
}

fn validate_metadata_arguments(
    args: &EncodedWorkflowArgs,
    parameter_shapes: &[Option<(RepresentationId, Vec<u64>)>],
    ports: &[PortSpec],
    results: &[Vec<Option<NativeTensorSpec>>],
) -> Result<(), CallError> {
    if parameter_shapes.len() != args.arguments().len() {
        return Err(CallError::Workflow(
            WorkflowError::NativeGraphArgumentMismatch { parameter: 0 },
        ));
    }
    for (ordinal, (argument, expected)) in args.arguments().iter().zip(parameter_shapes).enumerate()
    {
        match (argument, expected) {
            (EncodedWorkflowArgument::Tensor(argument), Some((representation, extents))) => {
                let actual = metadata_argument_shape(argument, ports, results)?;
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
    Ok(())
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
    stores: &[Store],
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
    for store in stores {
        used[store.port as usize] = true;
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
    /// Each of the node's call-private scratch buffers.
    scratch: Vec<ScratchNeed>,
    /// Its condition (`NodeCondition`): the external port, the word and what
    /// the node is encoded under.
    condition: Option<Condition>,
    /// Its result that shares an earlier or later node's result storage
    /// (`NativeGraphDraft::share`): its ordinal and that result.
    share: Option<(u32, WorkflowResultRef)>,
}

/// The port of graph `identity`'s `ports` a condition of node `node` names,
/// one of whose i32 words it reads: refused unless the host writes it (an
/// external or owned input port, not a local).
fn condition_port(identity: u64, ports: &[PortSpec], node: usize, condition: NodeCondition) -> Result<u32, WorkflowError> {
    let reference = condition.port.reference;
    let port = (reference.workflow == identity && reference.node == INPUT_NODE)
        .then(|| ports.get(reference.result as usize))
        .flatten()
        .filter(|port| !port.local)
        .ok_or(WorkflowError::ConditionWord { node })?;
    if port.representation != seismic_lang::registry::dense(seismic_lang::types::DType::I32) || condition.index >= port.byte_len / 4 {
        return Err(WorkflowError::ConditionWord { node });
    }
    Ok(reference.result)
}

/// A node condition as a plan holds it: the external port, the word, and
/// what the node is encoded under.
type Condition = (u32, u64, seismic_lang::checked::Encoded);

/// Whether nodes of conditions `a` and `b` are never encoded in one run:
/// both name one word, under opposite encodings (one while it is set, the
/// other while it is clear). The proof that two results may share storage.
fn exclusive(a: Option<Condition>, b: Option<Condition>) -> bool {
    match (a, b) {
        (Some((port, index, encoded)), Some((other_port, other_index, other_encoded))) => {
            port == other_port && index == other_index && encoded != other_encoded
        }
        _ => false,
    }
}

/// Validate each node's shared result (`NativeGraphDraft::share`): it is a
/// result of the node, of exactly the description of the result it shares,
/// a result of another node whose condition is exclusive of its own; nothing
/// reads, exports or stores it (its readers read the result it shares), and
/// the result it shares is neither stored nor itself shared.
fn validate_shares(
    nodes: &[PlannedNode],
    results: &[Vec<Option<NativeTensorSpec>>],
    exports: &[WorkflowResultRef],
    stores: &[Store],
) -> Result<(), WorkflowError> {
    for (node, planned) in nodes.iter().enumerate() {
        let Some((ordinal, primary)) = planned.share else { continue };
        let other = WorkflowResultRef { workflow: primary.workflow, node: node as u32, result: ordinal };
        let spec = |reference: WorkflowResultRef| {
            (reference.node != INPUT_NODE)
                .then(|| results.get(reference.node as usize)?.get(reference.result as usize)?.as_ref())
                .flatten()
        };
        let (Some(own), Some(shared)) = (spec(other), spec(primary)) else {
            return Err(WorkflowError::SharedResult { node });
        };
        let same = own.representation == shared.representation
            && own.extents == shared.extents
            && own.strides == shared.strides
            && own.byte_len == shared.byte_len;
        let primary_node = &nodes[primary.node as usize];
        let read = nodes.iter().any(|node| {
            node.args.arguments().iter().any(|argument| argument_reference(argument) == Some(other))
        });
        let held = exports.contains(&other)
            || stores.iter().any(|store| store.result == other || store.result == primary)
            || primary_node.share.is_some_and(|(shared, _)| shared == primary.result);
        if primary.node as usize == node || !same || read || held || !exclusive(planned.condition, primary_node.condition) {
            return Err(WorkflowError::SharedResult { node });
        }
    }
    Ok(())
}

/// Where a node's buffer lives. Only an external port's tensor is bound
/// per run; every other buffer is placed at seal.
#[derive(Clone, Copy, Debug)]
enum Region {
    /// The tensor bound to this external port (a port ordinal is a `u32`
    /// result reference); the site offset is added to the tensor's own byte
    /// offset.
    External(u32),
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
    /// The bytes the node reaches from `offset`.
    bytes: u64,
    /// For a record port, the records it binds (whole records, checked at
    /// seal), through a table of their addresses: its external port's
    /// tensor's from the record at `offset`, or in the workspace the slot's
    /// workspace record table from byte `table`; zero for any other buffer.
    records: u64,
    /// A workspace record site's entries in the slot's workspace record
    /// table (`NativeGraphPlan::workspace_record_table`), in bytes.
    table: u64,
}

/// How a graph's nodes use one external port.
#[derive(Clone, Copy, Debug, Default)]
struct PortUse {
    /// A launch reads its grid from it, so it binds no slab view.
    grid: bool,
    /// A node binds it other than as a record port (a host port, never
    /// bound, does not), so it binds no chunked storage.
    direct: bool,
    /// A node binds it as a record port: a run forms its tensor's records.
    records: bool,
    /// A launch condition reads a word of it, which the host reads as a
    /// run is queued: it binds host-addressed storage.
    condition: bool,
}

/// A launch condition as a graph seals it: one its node's kernel declares
/// (`NativeLaunch::conditions`) or the node's own (`NodeCondition`), the
/// i32 word at byte `offset` of a tensor the host writes, the launch
/// encoded while it holds `encoded`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LaunchCondition {
    word: HostWord,
    offset: u64,
    encoded: seismic_lang::checked::Encoded,
}

/// The host-written storage a launch condition reads its word of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HostWord {
    /// The tensor bound to this external port, from its view's first byte.
    Port(u32),
    /// The run's upload region, host-mapped on every backend.
    Upload,
}

/// One node in executable form, fixed at seal.
struct SealedNode {
    kernel: Arc<NativePrepared>,
    /// Buffers in ABI order.
    sites: Vec<Site>,
    /// The bound buffers' representations, in argument order.
    representations: Vec<&'static str>,
    words: Vec<u64>,
    /// Per launch ordinal, fixed at seal; `None` for an inactive launch.
    launches: CallLaunches,
    /// Per launch ordinal, the launch conditions it is encoded under, a
    /// conjunction.
    conditions: Vec<Vec<LaunchCondition>>,
}

/// Byte range of an external port's view in one node argument.
#[derive(Clone, Debug)]
struct PortRange {
    port: usize,
    offset: u64,
    bytes: u64,
    /// The node and the ordinal of its entry parameter, which name the
    /// parameter in the aliasing error from the node's own schema.
    node: usize,
    parameter: usize,
}

impl PortRange {
    fn parameter_name(&self, nodes: &[SealedNode]) -> String {
        nodes[self.node].kernel.schema().parameters()[self.parameter]
            .name
            .clone()
    }
}

/// The sealed nodes and what a run must check of its external bindings.
struct Executable {
    nodes: Vec<SealedNode>,
    /// Per port: whether any node writes the external tensor bound to it.
    external_writes: Vec<bool>,
    /// Per port: how the nodes bind it.
    port_uses: Vec<PortUse>,
    /// Whether any node writes a host-written input.
    upload_written: bool,
    /// Argument pairs on two different external ports that their entry
    /// requires to be disjoint. Pairs on graph storage, or on one port,
    /// were decided at seal.
    disjoint: Vec<(PortRange, PortRange)>,
    /// Timeline nodes in order, each with the kernel node it precedes
    /// (`nodes.len()` after the last).
    timelines: Vec<(usize, TimelineNode)>,
    /// Span marks in order, each the kernel node it precedes.
    marks: Vec<usize>,
    /// Entries of the workspace record table: the records of every record
    /// site in the workspace, in node and site order.
    workspace_records: u64,
}

impl Executable {
    /// The node positions its graphs divide at: each timeline node's.
    fn bounds(&self) -> Vec<usize> {
        self.timelines.iter().map(|&(position, _)| position).collect()
    }

    fn seal_node(
        &mut self,
        kernel: Arc<NativePrepared>,
        arguments: &[ArgumentValue],
        shape: super::CallShape,
        planned: &PlannedNode,
        storage: &StoragePlan,
        ports: &[PortSpec],
        node: usize,
    ) -> Result<(), WorkflowError> {
        let args = &planned.args;
        let schema = kernel.schema();
        let buffers = sealed_buffers(schema, kernel.implementation());
        let mut sites = Vec::with_capacity(buffers);
        let mut representations = Vec::with_capacity(buffers);
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
                            node,
                            parameter: ordinal,
                        });
                        (Region::External(reference.result), 0)
                    }
                }
            } else {
                storage.placements[reference.node as usize][reference.result as usize].site()
            };
            let write = matches!(access, TensorAccess::Owned | TensorAccess::Mutable);
            match region {
                Region::External(port) => {
                    self.external_writes[port as usize] |= write;
                    self.port_uses[port as usize].grid |= kernel.reads_grid(ordinal);
                }
                Region::Upload => self.upload_written |= write,
                Region::Workspace | Region::Outputs => {}
            }
            let mut site = Site {
                region,
                offset: base + descriptor.byte_offset,
                bytes: descriptor.byte_len,
                records: 0,
                table: 0,
            };
            let (extents, strides) = (&descriptor.extents, &descriptor.strides);
            let (record, bound) = (kernel.records[sites.len()], kernel.bound[sites.len()]);
            if !self.use_site(&mut site, record, bound, descriptor.representation, extents, strides, ports) {
                return Err(WorkflowError::RecordPortBinding {
                    parameter: parameter.name.clone(),
                });
            }
            sites.push(site);
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
            if let Region::External(port) = region {
                self.external_writes[port as usize] = true;
            }
            let mut site = Site {
                region,
                offset,
                bytes: spec.byte_len,
                records: 0,
                table: 0,
            };
            let record = kernel.records[sites.len()];
            if !self.use_site(&mut site, record, true, spec.representation, &spec.extents, &spec.strides, ports) {
                return Err(WorkflowError::RecordPortBinding {
                    parameter: format!("result {ordinal}"),
                });
            }
            sites.push(site);
            representations
                .push(seismic_lang::registry::representation_info(spec.representation).name);
        }
        for (offset, need) in storage.node_scratch[node].iter().zip(&shape.scratch) {
            sites.push(Site {
                region: Region::Workspace,
                offset: *offset,
                bytes: need.bytes,
                records: 0,
                table: 0,
            });
            representations.push("bytes");
        }
        // The kernel's arguments: its bound buffers', all but its host
        // ports, in an allocation of their count (as `SealedNodeShape`
        // plans them).
        let mut bound = Vec::with_capacity(kernel.bound_buffers());
        kernel.bind(representations, &mut bound);
        // Each launch's conditions: its kernel's, of the words its sites
        // bind, and the node's own on every launch, each a word the host
        // writes (an external port's tensor, or the run's upload region).
        let host_word = |(region, offset): (Region, u64)| match region {
            Region::External(port) => Some((HostWord::Port(port), offset)),
            Region::Upload => Some((HostWord::Upload, offset)),
            Region::Workspace | Region::Outputs => None,
        };
        let own = planned
            .condition
            .map(|(port, index, encoded)| {
                let site = storage.port_placements[port as usize].map_or((Region::External(port), 0), Placement::site);
                let (word, offset) = host_word(site).ok_or(WorkflowError::ConditionWord { node })?;
                Ok(LaunchCondition { word, offset: offset + 4 * index, encoded })
            })
            .transpose()?;
        let conditions = (0..shape.launches.geometry.len())
            .map(|ordinal| {
                kernel
                    .conditions(ordinal)
                    .iter()
                    .map(|&(buffer, index, encoded)| {
                        let site = &sites[buffer];
                        let (word, offset) = (site.records == 0)
                            .then(|| host_word((site.region, site.offset)))
                            .flatten()
                            .ok_or(WorkflowError::ConditionWord { node })?;
                        Ok(LaunchCondition { word, offset: offset + 4 * index, encoded })
                    })
                    .chain(own.map(Ok))
                    .collect::<Result<Vec<_>, WorkflowError>>()
            })
            .collect::<Result<Vec<_>, WorkflowError>>()?;
        for condition in conditions.iter().flatten() {
            if let HostWord::Port(port) = condition.word {
                self.port_uses[port as usize].condition = true;
            }
        }
        self.nodes.push(SealedNode {
            kernel,
            sites,
            representations: bound,
            words: shape.words,
            launches: shape.launches,
            conditions,
        });
        Ok(())
    }

    /// Record how `site`, a view of `extents` and `strides`, uses its
    /// storage, and whether it may bind there. A record port's site
    /// (`record`) binds only whole records of a canonical view (of its
    /// records' bytes): of an external port's tensor from a record boundary,
    /// through the table the tensor keeps ([`TensorInner::records`]), or in
    /// the workspace, through the table each slot forms with its workspace
    /// ([`NativeGraphPlan::workspace_record_table`]). One in a run's upload
    /// region or output arena is refused: its table would be formed per run.
    #[allow(clippy::too_many_arguments)]
    fn use_site(
        &mut self,
        site: &mut Site,
        record: bool,
        bound: bool,
        representation: RepresentationId,
        extents: &[u64],
        strides: &[u64],
        ports: &[PortSpec],
    ) -> bool {
        let canonical = || {
            crate::layout::canonical(representation, extents).is_ok_and(|layout| layout.strides == strides)
        };
        use seismic_lang::registry::record_bytes;
        let records = extents.first().copied().unwrap_or(0);
        match (site.region, record) {
            (Region::External(port), true) => {
                let spec = &ports[port as usize];
                let whole = match (
                    record_bytes(spec.representation, &spec.extents),
                    record_bytes(representation, extents),
                ) {
                    (Some(port), Some(record)) => {
                        canonical() && port == record && site.offset % record.get() == 0
                    }
                    _ => false,
                };
                self.port_uses[port as usize].records |= whole;
                site.records = records;
                whole
            }
            (Region::Workspace, true) => {
                let whole = record_bytes(representation, extents).is_some() && canonical();
                site.records = records;
                site.table = seismic_lang::registry::record_table_entry(self.workspace_records);
                self.workspace_records += records;
                whole
            }
            // A host port is ordered by, never bound: it binds any storage.
            (Region::External(port), false) => {
                self.port_uses[port as usize].direct |= bound;
                true
            }
            (_, record) => !record,
        }
    }
}

#[derive(Clone, Copy)]
enum Placement {
    Scratch(u64),
    Export(u64),
    /// A host-written input in the per-submission upload region.
    Upload(u64),
    /// A stored result: the tensor bound to this external port.
    Port(u32),
    Scalar,
}

impl Placement {
    fn site(self) -> (Region, u64) {
        match self {
            Self::Scratch(offset) => (Region::Workspace, offset),
            Self::Export(offset) => (Region::Outputs, offset),
            Self::Upload(offset) => (Region::Upload, offset),
            Self::Port(port) => (Region::External(port), 0),
            Self::Scalar => unreachable!("a scalar result is never a graph buffer"),
        }
    }
}

#[derive(Clone, Copy)]
enum StorageKey {
    Port(usize),
    Result(usize, usize),
    NodeScratch(usize, usize),
}

#[derive(Clone)]
struct StoragePlan {
    port_placements: Vec<Option<Placement>>,
    placements: Vec<Vec<Placement>>,
    node_scratch: Vec<Vec<u64>>,
    scratch_bytes: u64,
    /// The most workspace bytes live at one step: no placement fits in less.
    scratch_floor_bytes: u64,
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

impl Interval {
    /// Whether the two buffers are live at a common step. Endpoints count:
    /// a node's inputs end and its results start at the same step, so they
    /// never alias.
    fn overlaps(&self, other: &Interval) -> bool {
        self.start <= other.end && other.start <= self.end
    }
}

/// Where tensors of `tensors`' representations and extents lie in one
/// upload region ([`NativeGraphFamilySlot::upload`]): in order, each at a
/// buffer boundary, as a run's own inputs are. Each one's first byte and
/// layout, and the bytes the region needs.
fn upload_placement(
    tensors: &[(RepresentationId, &[u64])],
) -> Result<(Vec<(u64, layout::Layout)>, u64), WorkflowError> {
    let mut end = 0u64;
    let placed = tensors
        .iter()
        .map(|&(representation, extents)| {
            let canonical = layout::canonical(representation, extents)
                .map_err(|error| WorkflowError::TensorView(error.into()))?;
            let offset = end
                .checked_next_multiple_of(BUFFER_ALIGNMENT)
                .and_then(|offset| Some((offset, offset.checked_add(canonical.byte_len)?)));
            let (offset, next) = offset.ok_or(WorkflowError::UploadBytesOverflow)?;
            end = next;
            Ok((offset, canonical))
        })
        .collect::<Result<Vec<_>, WorkflowError>>()?;
    Ok((placed, end))
}

/// The bytes of the upload region [`NativeGraphFamilySlot::upload`] takes
/// for tensors of `tensors`' representations and extents.
pub fn upload_region_bytes(tensors: &[(RepresentationId, &[u64])]) -> Result<u64, WorkflowError> {
    upload_placement(tensors).map(|(_, bytes)| bytes)
}

fn align_up(value: u64, alignment: u64) -> u64 {
    value.div_ceil(alignment) * alignment
}

/// The most bytes of `intervals` live at one step: the size no placement of
/// them can go below.
fn liveness_floor(intervals: &[&Interval]) -> u64 {
    let mut events = intervals
        .iter()
        .flat_map(|interval| {
            [
                (interval.start, interval.bytes as i128),
                (interval.end + 1, -(interval.bytes as i128)),
            ]
        })
        .collect::<Vec<_>>();
    events.sort_unstable();
    let mut live = 0i128;
    let mut floor = 0i128;
    let mut index = 0;
    while index < events.len() {
        let step = events[index].0;
        while index < events.len() && events[index].0 == step {
            live += events[index].1;
            index += 1;
        }
        floor = floor.max(live);
    }
    u64::try_from(floor).expect("live workspace bytes fit u64")
}

/// Offsets of `intervals` in one arena, and the arena's size. Largest first,
/// each buffer takes the lowest aligned offset that overlaps no placed buffer
/// whose lifetime overlaps its own. Ties keep the intervals' order, so the
/// placement is deterministic.
fn place_by_liveness(intervals: &[&Interval]) -> (Vec<u64>, u64) {
    let mut order = (0..intervals.len()).collect::<Vec<_>>();
    order.sort_by_key(|&index| std::cmp::Reverse(intervals[index].bytes));
    let mut offsets = vec![0u64; intervals.len()];
    let mut placed: Vec<usize> = Vec::with_capacity(intervals.len());
    let mut size = 0u64;
    for index in order {
        let interval = intervals[index];
        let mut occupied = placed
            .iter()
            .filter(|&&other| intervals[other].overlaps(interval))
            .map(|&other| (offsets[other], offsets[other] + intervals[other].bytes))
            .collect::<Vec<_>>();
        occupied.sort_unstable();
        let mut offset = 0u64;
        for (start, end) in occupied {
            if offset
                .checked_add(interval.bytes)
                .expect("native graph storage overflow")
                <= start
            {
                break;
            }
            offset = offset.max(align_up(end, BUFFER_ALIGNMENT));
        }
        offsets[index] = offset;
        size = size.max(offset + interval.bytes);
        placed.push(index);
    }
    (offsets, size)
}

/// Place graph storage. Results, graph locals and node scratch share one
/// arena by lifetime (a node's scratch is live across its inputs and
/// results, so it never aliases them), placed by [`place_by_liveness`];
/// exports get their own arena; inputs the host writes go to a
/// per-submission upload region so a later host write cannot race an
/// earlier submission still reading them. Every buffer starts at a
/// [`BUFFER_ALIGNMENT`] offset of its region.
fn plan_storage(
    ports: &[PortSpec],
    nodes: &[PlannedNode],
    results: &[Vec<Option<NativeTensorSpec>>],
    exports: &[WorkflowResultRef],
    stores: &[Store],
) -> StoragePlan {
    let stored = |node: usize, ordinal: usize| {
        stores
            .iter()
            .find(|store| store.result.node as usize == node && store.result.result as usize == ordinal)
            .map(|store| store.port as usize)
    };
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
    // A result another shares (`NativeGraphDraft::share`) lives from the
    // first of its writers to its last read and the last writer.
    let sharers = |node: usize, ordinal: usize| {
        nodes.iter().enumerate().filter_map(move |(sharer, planned)| {
            planned
                .share
                .filter(|(_, primary)| primary.node as usize == node && primary.result as usize == ordinal)
                .map(|_| sharer)
        })
    };
    for (node, row) in results.iter().enumerate() {
        for (ordinal, result) in row.iter().enumerate() {
            let Some(spec) = result else { continue };
            // A stored result lives in its port's tensor, a shared one in
            // the storage of the result it shares (below).
            if stored(node, ordinal).is_some() || nodes[node].share.is_some_and(|(shared, _)| shared as usize == ordinal) {
                continue;
            }
            let exported = exports
                .iter()
                .any(|item| item.node as usize == node && item.result as usize == ordinal);
            let writers = std::iter::once(node).chain(sharers(node, ordinal));
            let (first, last) = writers.fold((node, node), |(first, last), writer| (first.min(writer), last.max(writer)));
            intervals.push(Interval {
                key: StorageKey::Result(node, ordinal),
                start: first * 2 + 1,
                end: last_use[node][ordinal].max(last * 2 + 1),
                bytes: spec.byte_len.max(1),
                exported,
            });
        }
    }
    // Sync scratch lives in the reserved range at the workspace's start,
    // each node's buffers disjoint from every other node's; the rest of the
    // workspace begins after the range (`cursor` below). Graphs of one
    // family share an arena, so every graph that places anything reserves
    // the range; a graph that places nothing needs no workspace at all.
    let mut sync_cursor = 0u64;
    let mut node_scratch = nodes
        .iter()
        .map(|node| vec![0u64; node.scratch.len()])
        .collect::<Vec<_>>();
    for (node, planned) in nodes.iter().enumerate() {
        for (ordinal, need) in planned.scratch.iter().enumerate() {
            if need.sync {
                let offset = sync_cursor.next_multiple_of(BUFFER_ALIGNMENT);
                sync_cursor = offset + need.bytes;
                assert!(
                    sync_cursor <= SYNC_SCRATCH_BYTES,
                    "sync scratch of one graph exceeds the reserved {SYNC_SCRATCH_BYTES} bytes"
                );
                node_scratch[node][ordinal] = offset;
                continue;
            }
            intervals.push(Interval {
                key: StorageKey::NodeScratch(node, ordinal),
                start: node * 2,
                end: node * 2 + 1,
                bytes: need.bytes,
                exported: false,
            });
        }
    }
    intervals.sort_by_key(|interval| interval.start);
    let scratch = intervals
        .iter()
        .filter(|interval| !interval.exported)
        .collect::<Vec<_>>();
    let (scratch_offsets, placed_bytes) = place_by_liveness(&scratch);
    // The placed buffers follow the reserved sync range, which a graph
    // reserves whenever it places anything.
    let reserved = if sync_cursor == 0 && placed_bytes == 0 {
        0
    } else {
        SYNC_SCRATCH_BYTES
    };
    let scratch_bytes = reserved + placed_bytes;
    let scratch_floor_bytes = reserved + liveness_floor(&scratch);
    let mut scratch_offsets = scratch_offsets.into_iter().map(|offset| reserved + offset);
    let mut output_bytes = 0u64;
    let mut placements = results
        .iter()
        .map(|row| vec![Placement::Scalar; row.len()])
        .collect::<Vec<_>>();
    for interval in &intervals {
        let placement = if interval.exported {
            let offset = align_up(output_bytes, BUFFER_ALIGNMENT);
            output_bytes = offset
                .checked_add(interval.bytes)
                .expect("native graph output bytes overflow");
            Placement::Export(offset)
        } else {
            Placement::Scratch(
                scratch_offsets
                    .next()
                    .expect("every scratch interval is placed"),
            )
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
    for store in stores {
        placements[store.result.node as usize][store.result.result as usize] =
            Placement::Port(store.port);
    }
    for (node, planned) in nodes.iter().enumerate() {
        if let Some((ordinal, primary)) = planned.share {
            placements[node][ordinal as usize] = placements[primary.node as usize][primary.result as usize];
        }
    }
    StoragePlan {
        port_placements,
        placements,
        node_scratch,
        scratch_bytes,
        scratch_floor_bytes,
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
    scratch_floor_bytes: u64,
    output_bytes: u64,
    upload_bytes: u64,
    /// What recording one of its runs takes ([`recorded_census`]).
    recorded: GraphCensus,
    /// Whether it is launched ([`NativeGraphDraft::seal_launched`]).
    launched: bool,
    /// What its graphs are keyed by beside its runs' storage (`ReplayKeys`).
    shape: PlanShape,
}

/// A plan in the shared allocation its holders keep it in: the plan and the
/// allocation's two reference counts.
const SHARED_PLAN_BYTES: usize =
    std::mem::size_of::<NativeGraphPlan>() + 2 * std::mem::size_of::<usize>();

impl NativeGraphPlan {
    /// The host heap this plan holds: itself in its shared allocation and
    /// every vector it owns, at the vector's capacity. Its kernels and its
    /// device are shared with the prepared programs and are theirs; the
    /// replays a CUDA or Vulkan device records of a plan's submissions
    /// belong to the device's queue and exist only once the plan has run.
    pub fn host_heap(&self) -> HostHeap {
        let mut heap = HostHeap::default();
        heap.hold(1, SHARED_PLAN_BYTES);
        heap.hold_vec(&self.ports);
        for port in &self.ports {
            heap.hold_vec(&port.extents);
            heap.hold_vec(&port.strides);
        }
        heap.hold_vec(&self.executable.nodes);
        for node in &self.executable.nodes {
            heap.hold_vec(&node.sites);
            heap.hold_vec(&node.representations);
            heap.hold_vec(&node.words);
            heap.hold_vec(&node.launches.geometry);
        }
        heap.hold_vec(&self.executable.external_writes);
        heap.hold_vec(&self.executable.port_uses);
        heap.hold_vec(&self.executable.timelines);
        heap.hold_vec(&self.executable.marks);
        heap.hold_vec(&self.executable.disjoint);
        heap.hold_vec(&self.results);
        for row in &self.results {
            heap.hold_vec(row);
            for spec in row.iter().flatten() {
                heap.hold_vec(&spec.extents);
                heap.hold_vec(&spec.strides);
            }
        }
        heap.hold_vec(&self.exports);
        heap.hold_vec(&self.port_placements);
        heap.hold_vec(&self.placements);
        for row in &self.placements {
            heap.hold_vec(row);
        }
        heap
    }

    /// The bytes of `host_heap`.
    pub fn host_bytes(&self) -> u64 {
        self.host_heap().bytes
    }

    pub fn workspace_bytes(&self) -> u64 {
        self.scratch_bytes
    }
    /// The most workspace bytes live at one step of the plan's placement:
    /// no placement fits in less.
    pub fn workspace_floor_bytes(&self) -> u64 {
        self.scratch_floor_bytes
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
    /// What recording one of its runs adds to its submission's graphs.
    pub fn recorded(&self) -> GraphCensus {
        self.recorded
    }

    /// What its runs' graphs cost serving so far
    /// (`NativeState::plan_graph_replays`), with every plan's of its layout
    /// where those share its graphs (CUDA).
    pub fn graph_replays(&self) -> super::GraphReplays {
        let backend = super::backend_name(&self.device.kind);
        self.device.native.plan_graph_replays(self.shape.keyed(backend))
    }
    /// Runs of the plan with `bindings` as `seismic-measure` times them,
    /// on a slot of their own ([`GraphPoint`]); a plan with timeline nodes
    /// is refused.
    pub fn point(
        self: &Arc<Self>,
        bindings: NativeGraphBindings,
        timing: GraphTiming,
    ) -> Result<GraphPoint, CallError> {
        GraphPoint::new(self, bindings, timing)
    }
    /// The device the plan runs on.
    pub(super) fn device(&self) -> &Arc<DeviceInner> {
        &self.device
    }
    /// Whether a run of the plan waits on or signals a timeline.
    pub(super) fn has_timelines(&self) -> bool {
        !self.executable.timelines.is_empty()
    }
    pub fn bindings(&self) -> NativeGraphBindings {
        NativeGraphBindings {
            identity: self.identity,
            inputs: vec![None; self.ports.len()],
            external: self.ports.iter().map(|port| !port.local).collect(),
            uses: self.executable.port_uses.clone(),
            checked: vec![false; self.ports.len()],
            timelines: vec![None; self.executable.timelines.len()],
        }
    }
    /// Refuse the disjoint pairs of `bindings` that overlap: those whose
    /// both ports are statically bound (`fixed`, checked once in
    /// [`Self::bind_static`]), or else those a run binds a port of (checked
    /// at attach, when every external port is bound).
    fn check_disjoint(&self, bindings: &NativeGraphBindings, fixed: bool) -> Result<(), CallError> {
        for (first, second) in &self.executable.disjoint {
            if (bindings.checked[first.port] && bindings.checked[second.port]) != fixed {
                continue;
            }
            let range = |range: &PortRange| {
                let tensor = bindings.inputs[range.port]
                    .as_ref()
                    .expect("every port of a checked pair is bound");
                let start = tensor.byte_offset() + range.offset;
                (tensor.alias_identity(), start, start + range.bytes)
            };
            let (first_allocation, first_start, first_end) = range(first);
            let (second_allocation, second_start, second_end) = range(second);
            if first_allocation == second_allocation
                && first.bytes != 0
                && second.bytes != 0
                && first_start < second_end
                && second_start < first_end
            {
                let nodes = &self.executable.nodes;
                return Err(CallError::Invocation(InvocationError::IllegalAliasing {
                    first: first.parameter_name(nodes),
                    second: second.parameter_name(nodes),
                }));
            }
        }
        Ok(())
    }

    /// Whether `tensor` may bind external port `port`: of its description,
    /// writable where a node writes it, and host-addressed where a launch
    /// condition reads a word of it (checked here, as the graph's bindings
    /// form, so a queued run reads it as it is).
    fn check_binding(&self, port: usize, tensor: &Arc<TensorInner>) -> Result<(), WorkflowError> {
        if !port_matches(&self.device, &self.ports[port], tensor)
            || (self.executable.external_writes[port] && tensor.allocation().storage().read_only())
        {
            return Err(WorkflowError::NativePortMismatch { port });
        }
        if self.executable.port_uses[port].condition && tensor.allocation().storage().host_address().is_none() {
            return Err(WorkflowError::UnaddressedCondition { parameter: format!("port {port}") });
        }
        Ok(())
    }

    /// Bind the ports that stay fixed across runs, checked once here; a
    /// run's attach checks only the ports set after this. A disjoint
    /// pair of fixed ports that overlaps is refused here, so no run of the
    /// bound plan can alias it.
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
            self.check_binding(ordinal, tensor).map_err(CallError::Workflow)?;
            bindings.checked[ordinal] = true;
        }
        self.check_disjoint(&bindings, true)?;
        Ok(BoundNativeGraphPlan {
            _plan: self.clone(),
            fixed: bindings,
        })
    }
    /// What one run's graphs hold when its layout's graph holds the
    /// launches `form` names (`super::LayoutForms`): on CUDA a graph of each
    /// of its segments of calls between timeline nodes (`graph_segments`)
    /// with a node for each launch the form holds, exactly what
    /// `cuda::instantiate` makes, none of a launched plan, whose runs launch
    /// call by call ([`NativeGraphDraft::seal_launched`]); elsewhere what
    /// recording its run takes at most ([`Self::recorded`]).
    fn run_graphs(&self, form: &[u64]) -> GraphCensus {
        if !self.graphed() {
            return GraphCensus::default();
        }
        match super::backend_name(&self.device.kind) {
            BackendName::Cuda => {
                let nodes = self.executable.nodes.len();
                let mut cuts = self.executable.bounds();
                cuts.extend([0, nodes]);
                cuts.sort_unstable();
                cuts.dedup();
                GraphCensus {
                    graphs: cuts.windows(2).filter(|cut| cut[0] < cut[1]).count() as u64,
                    launches: form.iter().map(|word| u64::from(word.count_ones())).sum::<u64>() + self.held_marks().len() as u64,
                    arguments: 0,
                    largest: 0,
                }
            }
            BackendName::Vulkan | BackendName::Metal | BackendName::Cpu => self.recorded,
        }
    }

    /// The span marks its graphs hold, each the node it precedes, in order
    /// ([`NativeGraphDraft::span_mark`]): those inside a graph, of a plan
    /// whose runs replay graphs. A mark at a run's edge or a timeline node
    /// falls between graphs, and costs a run not sampled nothing.
    fn held_marks(&self) -> Vec<usize> {
        if !self.graphed() {
            return Vec::new();
        }
        let (nodes, bounds) = (self.executable.nodes.len(), self.executable.bounds());
        let held = self.executable.marks.iter().copied().filter(|mark| (1..nodes).contains(mark) && !bounds.contains(mark));
        held.collect()
    }

    /// Whether its runs replay graphs: on CUDA all but a launched plan's,
    /// which launch call by call ([`NativeGraphDraft::seal_launched`]).
    fn graphed(&self) -> bool {
        !(self.launched && super::backend_name(&self.device.kind) == BackendName::Cuda)
    }

    /// A slot's share of its device's graph budget ([`super::SlotGraphs`]):
    /// a launch of the plan's own slot is one run, its layout's graph of
    /// the launches it makes, instantiated ([`instantiate`]).
    fn slot_hold(self: &Arc<Self>) -> Result<Arc<super::SlotGraphs>, TensorError> {
        let form: Arc<[u64]> = self.shape.structure.clone().into();
        let forms = Arc::new(super::LayoutForms::new([(self.shape.layout, form.clone())]));
        super::SlotGraphs::new(&self.device, self.run_graphs(&form), forms, |owned| {
            instantiate(std::slice::from_ref(self), form, 1, owned)
        })
    }

    /// A slot of this plan alone, holding a run's graphs in its device's
    /// graph budget ([`super::SlotGraphs`]). Its upload region is fixed, so
    /// host writes to its inputs wait for the slot's previous submission; a
    /// family slot instead takes the smallest of its idle upload regions
    /// that holds a run's input ([`idle_upload`]).
    pub fn new_slot(self: &Arc<Self>) -> Result<NativeGraphSlot, TensorError> {
        let workspace = if self.scratch_bytes == 0 {
            None
        } else {
            Some(self.device.allocate_zeroed(self.scratch_bytes, BUFFER_ALIGNMENT)?)
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
            workspace_records: workspace
                .as_ref()
                .map(|workspace| self.workspace_record_table(workspace))
                .transpose()?
                .flatten(),
            workspace,
            upload,
            hold: self.slot_hold()?,
        })
    }

    /// The table of the addresses of the records every record site in the
    /// workspace binds (`Site::table`), over `workspace`, the slot's: formed
    /// once with the slot, so no run allocates one, and charged with the
    /// workspace while it lives (`Allocation::hold_table`). None for a plan
    /// without such sites.
    fn workspace_record_table(
        &self,
        workspace: &Arc<Allocation>,
    ) -> Result<Option<Arc<Allocation>>, TensorError> {
        if self.executable.workspace_records == 0 {
            return Ok(None);
        }
        let offsets = self
            .executable
            .nodes
            .iter()
            .flat_map(|node| &node.sites)
            .filter(|site| matches!(site.region, Region::Workspace) && site.records != 0)
            .flat_map(|site| {
                let record = site.bytes / site.records;
                (0..site.records).map(move |index| site.offset + index * record)
            });
        let table = crate::api::tensor::record_table(&self.device, workspace, offsets)?;
        workspace.hold_table(&table);
        Ok(Some(table))
    }

    /// Exported outputs are a separate owner so their lifetime can outlive
    /// scratch-slot reuse without retaining the scratch allocation.
    pub fn new_outputs(self: &Arc<Self>) -> Result<NativeGraphOutputs, TensorError> {
        Ok(self.outputs_from_arena(self.output_arena()?, None))
    }

    /// A run's output arena; none for a plan that exports nothing.
    pub(super) fn output_arena(&self) -> Result<Option<Arc<Allocation>>, TensorError> {
        if self.output_bytes == 0 {
            return Ok(None);
        }
        Ok(Some(self.device.allocate(self.output_bytes, BUFFER_ALIGNMENT)?))
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

    pub(super) fn outputs_from_arena(
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
            parks: Vec::new(),
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

    /// The host heap this binding holds, at its vectors' capacities. The
    /// plan it binds is shared and is its sealer's.
    pub fn host_heap(&self) -> HostHeap {
        let mut heap = HostHeap::default();
        heap.hold_vec(&self.fixed.inputs);
        heap.hold_vec(&self.fixed.external);
        heap.hold_vec(&self.fixed.uses);
        heap.hold_vec(&self.fixed.checked);
        heap.hold_vec(&self.fixed.timelines);
        heap
    }
}

/// The host heap of the bindings of a plan of `ports` ports and
/// `timelines` timeline nodes (`NativeGraphPlan::bindings`).
fn bindings_heap(ports: usize, timelines: usize) -> HostHeap {
    use std::mem::size_of;
    let mut heap = HostHeap::default();
    heap.hold(ports, size_of::<Option<Arc<TensorInner>>>());
    heap.hold(ports, size_of::<bool>());
    heap.hold(ports, size_of::<PortUse>());
    heap.hold(ports, size_of::<bool>());
    heap.hold(timelines, size_of::<Option<TimelineBinding>>());
    heap
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
    /// Its plans by layout, each layout's graph and the most runs of it one
    /// launch makes: a slot instantiates that many graphs of each layout
    /// ([`instantiate`]), so serving replays them and instantiates none.
    layouts: Vec<LayoutGraphs>,
    /// The form of each layout's graph, which its slots' runs replay.
    forms: Arc<super::LayoutForms>,
    /// What one slot holds of its device's graph budget
    /// ([`super::SlotGraphs`]): each layout's graph for each of its runs.
    slot_graphs: GraphCensus,
}

/// A family's plans of one layout, which one graph serves
/// (`super::LayoutForms`), and the most runs of them one launch makes.
struct LayoutGraphs {
    plans: Vec<Arc<NativeGraphPlan>>,
    form: Arc<[u64]>,
    runs: usize,
}

impl NativeGraphFamily {
    /// A family of `plans` whose launch is one run of one of them.
    pub fn new(plans: &[Arc<NativeGraphPlan>]) -> Result<Self, WorkflowError> {
        let launches = plans.iter().map(|plan| vec![plan.clone()]).collect::<Vec<_>>();
        Self::with_launches(plans, &launches)
    }

    /// A family of `plans` whose launches each make the runs of one of
    /// `launches`, by plan, a plan once for each of its runs. Its plans of
    /// one layout share one graph (`super::LayoutForms`): on CUDA a node for
    /// each launch any of them makes, each run enabling its own; elsewhere
    /// their own graphs, recorded at first use, each run of a launch taking
    /// at most one. A slot holds, of each layout, a graph for each run of
    /// it the launch making most of them makes, whatever class each run
    /// is ([`Self::slot_graphs`]); on CUDA it instantiates them as it is
    /// created, so serving replays them, updated where a run binds other
    /// storage, and instantiates none.
    pub fn with_launches(plans: &[Arc<NativeGraphPlan>], launches: &[Vec<Arc<NativeGraphPlan>>]) -> Result<Self, WorkflowError> {
        let first = plans.first().ok_or(WorkflowError::Empty)?;
        assert!(
            plans.iter().all(|plan| !plan.launched),
            "a launched plan's slots hold no graph, so it is never a family's"
        );
        if plans
            .iter()
            .any(|plan| !Arc::ptr_eq(&plan.device, &first.device))
        {
            return Err(WorkflowError::NativeGraphSlotMismatch);
        }
        // Its plans by layout, in order of their first.
        let mut at = std::collections::HashMap::<u64, usize>::new();
        let mut layouts = Vec::<LayoutGraphs>::new();
        for plan in plans {
            let index = *at.entry(plan.shape.layout).or_insert_with(|| {
                layouts.push(LayoutGraphs { plans: Vec::new(), form: Arc::from([]), runs: 0 });
                layouts.len() - 1
            });
            layouts[index].plans.push(plan.clone());
        }
        for graphs in &mut layouts {
            let calls = graphs.plans[0].shape.structure.len();
            assert!(
                graphs.plans.iter().all(|plan| plan.shape.structure.len() == calls),
                "a layout's plans make launches of the same calls"
            );
            graphs.form = (0..calls)
                .map(|call| graphs.plans.iter().fold(0, |made, plan| made | plan.shape.structure[call]))
                .collect();
        }
        // Of each layout, the most runs a launch makes; every plan runs in
        // some launch, else a run of it would find no graph its slot made.
        let members = plans.iter().map(Arc::as_ptr).collect::<std::collections::HashSet<_>>();
        let mut launched = std::collections::HashSet::new();
        for runs in launches {
            let mut counts = std::collections::HashMap::<usize, usize>::new();
            for plan in runs {
                assert!(members.contains(&Arc::as_ptr(plan)), "a family's launches run its plans");
                launched.insert(Arc::as_ptr(plan));
                *counts.entry(at[&plan.shape.layout]).or_default() += 1;
            }
            for (index, count) in counts {
                layouts[index].runs = layouts[index].runs.max(count);
            }
        }
        assert_eq!(launched.len(), members.len(), "every plan of a family runs in one of its launches");
        let slot_graphs = layouts
            .iter()
            .try_fold(GraphCensus::default(), |census, graphs| {
                let graph = graphs
                    .plans
                    .iter()
                    .fold(GraphCensus::default(), |most, plan| most.most(plan.run_graphs(&graphs.form)));
                census.plus(graph.times(graphs.runs as u64)?)
            })
            .expect("a family's graph census fits u64");
        let forms = Arc::new(super::LayoutForms::new(
            layouts.iter().map(|graphs| (graphs.plans[0].shape.layout, graphs.form.clone())),
        ));
        Ok(Self {
            layouts,
            forms,
            slot_graphs,
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

    /// The most input bytes one of the family's runs writes: an upload
    /// region that holds any of its runs' input.
    pub fn upload_bytes(&self) -> u64 {
        self.upload_bytes
    }

    pub fn workspace_bytes(&self) -> u64 {
        self.workspace_bytes
    }

    /// The largest liveness floor of the family's plans: the workspace no
    /// placement of them fits in less than.
    pub fn workspace_floor_bytes(&self) -> u64 {
        self.plans
            .iter()
            .map(|plan| plan.scratch_floor_bytes)
            .max()
            .unwrap_or(0)
    }

    pub fn output_bytes(&self) -> u64 {
        self.output_bytes
    }

    /// What its plans' graphs cost serving so far: graphs recorded and
    /// replays that updated them (`Device::graph_replays`), this family's
    /// alone of what its device counts.
    pub fn graph_replays(&self) -> super::GraphReplays {
        let backend = super::backend_name(&self.device.kind);
        let mut keyed = self.plans.iter().map(|plan| plan.shape.keyed(backend)).collect::<Vec<_>>();
        keyed.sort_unstable();
        keyed.dedup();
        keyed.into_iter().fold(super::GraphReplays::default(), |counts, keyed| {
            counts.plus(self.device.native.plan_graph_replays(keyed))
        })
    }

    /// What one live slot of the family holds of its device's graph
    /// budget: each layout's graph for each run of it one launch makes.
    pub fn slot_graphs(&self) -> GraphCensus {
        self.slot_graphs
    }

    /// A slot's share of its device's graph budget ([`super::SlotGraphs`]),
    /// with each layout's graphs instantiated ([`instantiate`]).
    fn slot_hold(&self) -> Result<Arc<super::SlotGraphs>, TensorError> {
        super::SlotGraphs::new(&self.device, self.slot_graphs, self.forms.clone(), |owned| {
            self.layouts
                .iter()
                .try_for_each(|graphs| instantiate(&graphs.plans, graphs.form.clone(), graphs.runs, owned))
        })
    }

    pub fn new_output_slot(self: &Arc<Self>) -> Result<NativeGraphFamilyOutputSlot, TensorError> {
        self.output_slot_of(|device, bytes| device.allocate(bytes, BUFFER_ALIGNMENT))
    }

    /// A slot of outputs the host reads as they complete (`Contract::Shared`):
    /// a host read of them lands in host-mapped memory, with no copy queued
    /// behind device work a later run has queued.
    pub fn new_host_output_slot(self: &Arc<Self>) -> Result<NativeGraphFamilyOutputSlot, TensorError> {
        self.output_slot_of(|device, bytes| {
            device.allocate_mapped(seismic_topology::Contract::Shared, bytes, BUFFER_ALIGNMENT)
        })
    }

    fn output_slot_of(
        self: &Arc<Self>,
        allocate: impl FnOnce(&Arc<DeviceInner>, u64) -> Result<Arc<Allocation>, seismic_compiler::errors::ExecutionError>,
    ) -> Result<NativeGraphFamilyOutputSlot, TensorError> {
        let arena = match self.output_bytes {
            0 => None,
            bytes => Some(allocate(&self.device, bytes)?),
        };
        Ok(NativeGraphFamilyOutputSlot {
            family: self.clone(),
            arena,
        })
    }

    /// A slot with its own scratch arena and upload regions of `regions`
    /// bytes each, all allocated here: one region per graph run that writes
    /// input the slot's owner keeps in flight at once, holding that run's
    /// input (a run whose plan writes none takes none). A family whose
    /// plans write no input allocates none. The slot holds the family's
    /// share of its device's graph budget ([`Self::launch`]).
    pub fn new_slot(
        self: &Arc<Self>,
        regions: &[u64],
    ) -> Result<NativeGraphFamilySlot, TensorError> {
        let scratch = if self.workspace_bytes == 0 {
            None
        } else {
            Some(
                self.device
                    .allocate_zeroed(self.workspace_bytes, BUFFER_ALIGNMENT)?,
            )
        };
        Ok(NativeGraphFamilySlot {
            family: self.clone(),
            workspace_records: self.workspace_record_tables(scratch.as_ref())?,
            scratch,
            uploads: self.upload_regions(regions)?,
            lent_locals: Vec::new(),
            hold: self.slot_hold()?,
        })
    }

    /// A slot whose workspace is the shared `arena`, with upload regions of
    /// `regions` bytes each of its own ([`Self::new_slot`]). The arena must
    /// hold this family's workspace, and no plan of the family may keep a
    /// value in workspace between runs: a prewritten port is refused, since
    /// another family's graph rewrites the arena before this family's next
    /// run.
    pub fn new_slot_in(
        self: &Arc<Self>,
        arena: &NativeExecutionArena,
        regions: &[u64],
    ) -> Result<NativeGraphFamilySlot, WorkflowError> {
        if !Arc::ptr_eq(&arena.device, &self.device) {
            return Err(WorkflowError::NativeGraphSlotMismatch);
        }
        if self.workspace_bytes > arena.bytes {
            return Err(WorkflowError::NativeArenaTooSmall {
                required: self.workspace_bytes,
                capacity: arena.bytes,
            });
        }
        if self
            .plans
            .iter()
            .any(|plan| plan.ports.iter().any(|port| port.prewritten))
        {
            return Err(WorkflowError::NativeArenaPrewrittenPort);
        }
        let scratch = (self.workspace_bytes != 0)
            .then(|| arena.allocation.clone())
            .flatten();
        Ok(NativeGraphFamilySlot {
            family: self.clone(),
            workspace_records: self
                .workspace_record_tables(scratch.as_ref())
                .map_err(WorkflowError::TensorView)?,
            scratch,
            uploads: self
                .upload_regions(regions)
                .map_err(WorkflowError::TensorView)?,
            lent_locals: Vec::new(),
            hold: self.slot_hold().map_err(WorkflowError::TensorView)?,
        })
    }

    /// Each plan's workspace record table over `scratch`, in plan order.
    fn workspace_record_tables(
        &self,
        scratch: Option<&Arc<Allocation>>,
    ) -> Result<Vec<Option<Arc<Allocation>>>, TensorError> {
        self.plans
            .iter()
            .map(|plan| {
                scratch
                    .map(|scratch| plan.workspace_record_table(scratch))
                    .transpose()
                    .map(Option::flatten)
            })
            .collect()
    }

    fn upload_regions(&self, regions: &[u64]) -> Result<Vec<Arc<Allocation>>, TensorError> {
        if self.upload_bytes == 0 {
            return Ok(Vec::new());
        }
        regions
            .iter()
            .map(|&bytes| Ok(self.device.allocate_upload(bytes, BUFFER_ALIGNMENT)?))
            .collect()
    }
}

/// One device's graph workspace, shared by every family slot created in it.
/// A device executes one submission at a time, in submission order, and a
/// graph's workspace holds only the values of its own run, so graphs bound
/// to one arena never use it at once: the device needs one workspace the
/// size of its largest family's, whatever the families and launches in
/// flight.
#[derive(Clone)]
pub struct NativeExecutionArena {
    device: Arc<DeviceInner>,
    allocation: Option<Arc<Allocation>>,
    bytes: u64,
}

impl NativeExecutionArena {
    pub fn new(device: &Arc<DeviceInner>, bytes: u64) -> Result<Self, TensorError> {
        let allocation = if bytes == 0 {
            None
        } else {
            Some(device.allocate_zeroed(bytes, BUFFER_ALIGNMENT)?)
        };
        Ok(Self {
            device: device.clone(),
            allocation,
            bytes,
        })
    }

    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// What the arena is charged: its bytes, and the record tables graph
    /// slots formed over it while they live
    /// (`NativeGraphPlan::workspace_record_table`).
    pub fn charged_bytes(&self) -> u64 {
        self.allocation.as_ref().map_or(0, |allocation| allocation.charged_bytes())
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
    /// Each family plan's workspace record table over `scratch`, in plan
    /// order ([`NativeGraphPlan::workspace_record_table`]).
    workspace_records: Vec<Option<Arc<Allocation>>>,
    /// Upload regions, fixed when the slot was created: one for each run its
    /// launch writes input for. Activation takes one no run or view of the
    /// launch holds, idle by construction ([`idle_upload`]).
    uploads: Vec<Arc<Allocation>>,
    lent_locals: Vec<Weak<TensorInner>>,
    /// A launch's graphs in the device's graph budget, held by every run
    /// queued from the slot while it lives.
    hold: Arc<super::SlotGraphs>,
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
        let Some(member) = self
            .family
            .plans
            .iter()
            .position(|member| Arc::ptr_eq(member, plan))
        else {
            return Err(WorkflowError::NativeGraphSlotMismatch);
        };
        // A run whose plan writes no input takes no region.
        let upload = match plan.upload_bytes {
            0 => None,
            bytes => Some(idle_upload(&self.uploads, bytes)?),
        };
        let slot = NativeGraphSlot {
            plan: plan.clone(),
            workspace: self.scratch.clone(),
            workspace_records: self.workspace_records[member].clone(),
            upload,
            hold: self.hold.clone(),
        };
        Ok(NativeGraphFamilyActive { slot, _owner: self })
    }

    /// Tensors of `tensors`' representations and extents in one of the
    /// slot's idle upload regions, placed as [`upload_placement`] places
    /// them, each holding its bytes written from the host: input several
    /// runs bind as external ports, written once. The region stays taken
    /// while any of them or a view of one lives, and until the device's last
    /// use of it completes, as a run's own input region does.
    pub fn upload(
        &mut self,
        tensors: &[(RepresentationId, &[u64], &[u8])],
    ) -> Result<Vec<Arc<TensorInner>>, WorkflowError> {
        let layouts = tensors
            .iter()
            .map(|&(representation, extents, _)| (representation, extents))
            .collect::<Vec<_>>();
        let (placed, bytes) = upload_placement(&layouts)?;
        let region = idle_upload(&self.uploads, bytes)?;
        tensors
            .iter()
            .zip(placed)
            .map(|(&(representation, extents, bytes), (offset, canonical))| {
                let tensor = TensorInner::new_view(
                    self.family.device.clone(),
                    region.clone(),
                    offset,
                    canonical.byte_len,
                    representation,
                    extents.to_vec(),
                    canonical.strides,
                );
                tensor.write_idle(bytes).map_err(WorkflowError::TensorView)?;
                Ok(Arc::new(tensor))
            })
            .collect()
    }
}

/// The smallest region of `uploads` (a slot's) that holds `bytes` and that
/// no run or view of the slot's launch has taken, the first such in slot
/// order. The slot is one launch in flight (its family's activations are
/// `FlightDepth::launches`): its lease returns only once its launch's
/// completion was observed, and every submitted run holds its region until
/// that completion drops, so a region no one holds is idle, by construction.
/// The slot holds a region for each run its launch writes input for
/// (`block_upload_regions`), so one is free. Either failing is an invariant
/// violation: it panics, never waits. Of a sequence of runs that some
/// assignment gives each a region of its own holding its input, every run
/// gets one: were the smallest region holding a run's input another run's in
/// that assignment, the two can swap, as that other run's input fits the
/// first run's assigned region, no smaller. The same sequence binds the same
/// regions every time, and replays its graphs.
fn idle_upload(uploads: &[Arc<Allocation>], bytes: u64) -> Result<Arc<Allocation>, WorkflowError> {
    let holding = || uploads.iter().filter(|upload| upload.bytes() >= bytes);
    if holding().next().is_none() {
        return Err(WorkflowError::UploadRegionTooSmall {
            bytes,
            largest: uploads.iter().map(|upload| upload.bytes()).max().unwrap_or(0),
        });
    }
    let region = holding()
        .filter(|upload| Arc::strong_count(upload) == 1)
        .min_by_key(|upload| upload.bytes())
        .expect("a launch's runs take no more upload regions than its slot holds");
    assert!(region.device_idle(), "an upload region no run holds was released by its last run's completion");
    Ok(region.clone())
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

    /// Write an owned input port's bytes into the region the launch took
    /// idle ([`idle_upload`]): never waiting for the device.
    pub fn write_input(&mut self, port: NativePort, bytes: &[u8]) -> Result<(), TensorError> {
        self.slot.write_upload(port, bytes, true)
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
    /// The plan's workspace record table over `workspace`
    /// ([`NativeGraphPlan::workspace_record_table`]), formed with it.
    workspace_records: Option<Arc<Allocation>>,
    /// Host-written inputs of the next run.
    upload: Option<Arc<Allocation>>,
    /// Its runs' graphs in the device's graph budget (its family slot's).
    hold: Arc<super::SlotGraphs>,
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

    /// Write an owned input port's bytes into the slot's fixed upload
    /// region, after the slot's previous submission that reads it: a plan's
    /// own slot, which measurement and tests run one submission at a time.
    pub fn write_input(&mut self, port: NativePort, bytes: &[u8]) -> Result<(), TensorError> {
        self.write_upload(port, bytes, false)
    }

    /// Write an owned input port's bytes into the upload region; `idle`: no
    /// device work can use it ([`TensorInner::write_idle`]).
    fn write_upload(&mut self, port: NativePort, bytes: &[u8], idle: bool) -> Result<(), TensorError> {
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
        let view = self.plan.port_view(ordinal, upload, offset);
        match idle {
            true => view.write_idle(bytes),
            false => view.write_from_host(bytes),
        }
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
            if !bindings.checked[port] {
                self.plan.check_binding(port, tensor).map_err(CallError::Workflow)?;
            }
            // An external binding never names this run's own storage: the
            // plan placed every graph buffer apart from external tensors.
            if owned.contains(&tensor.allocation().identity()) {
                return Err(CallError::Workflow(WorkflowError::NativePortMismatch {
                    port,
                }));
            }
        }
        self.plan.check_disjoint(&bindings, false)?;
        let mut timelines = Vec::with_capacity(bindings.timelines.len());
        let mut parks = Vec::new();
        for (node, (bound, &(position, kind))) in bindings
            .timelines
            .into_iter()
            .zip(&self.plan.executable.timelines)
            .enumerate()
        {
            // A publication binds as a park; only a park waits for an answer.
            let parked = matches!(kind, TimelineNode::Park | TimelineNode::Publish);
            let (timeline, value) = match bound {
                None => return Err(CallError::Workflow(WorkflowError::NativeTimelineUnbound { node })),
                Some(TimelineBinding::Parked(_)) if !parked => {
                    return Err(CallError::Workflow(WorkflowError::NotPark { node }));
                }
                Some(TimelineBinding::Answered) if kind != TimelineNode::Park => {
                    return Err(CallError::Workflow(WorkflowError::NotPark { node }));
                }
                Some(TimelineBinding::Bound(..)) if parked => {
                    return Err(CallError::Workflow(WorkflowError::ParkAtValue { node }));
                }
                Some(TimelineBinding::Answered) => continue,
                Some(TimelineBinding::Parked(timeline)) => {
                    if !Arc::ptr_eq(timeline.device(), &self.plan.device) {
                        return Err(CallError::Workflow(WorkflowError::NativeTimelineMismatch { node }));
                    }
                    if !timeline.parks_in_kernels() {
                        parks.push(PendingPark {
                            node,
                            position,
                            timeline,
                            publishes: kind == TimelineNode::Publish,
                        });
                    }
                    continue;
                }
                Some(TimelineBinding::Bound(timeline, value)) => (timeline, value),
            };
            if !Arc::ptr_eq(timeline.device(), &self.plan.device) {
                return Err(CallError::Workflow(WorkflowError::NativeTimelineMismatch {
                    node,
                }));
            }
            let action = match kind {
                TimelineNode::Wait => {
                    timeline.await_at(value);
                    TimelineAction::Wait
                }
                TimelineNode::Signal => TimelineAction::Signal,
                TimelineNode::Park | TimelineNode::Publish => unreachable!("a park or publication binds no value"),
            };
            timelines.push((
                position,
                TimelineOperation {
                    action,
                    timeline,
                    value,
                },
            ));
        }
        Ok(ReadyNativeGraphRun {
            slot: self,
            externals: bindings.inputs,
            exports,
            timelines,
            parks,
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
    /// The value each park of the queued run took, by its timeline node
    /// ([`Self::park`]).
    parks: Vec<(usize, u64)>,
}

impl NativeGraphOutputs {
    /// The value park or publication node `port` of the queued run took on
    /// its timeline (`NativeGraphBindings::set_park`): the device reaches it
    /// once a park's call waits for the host's answer, or a publication's
    /// call has run, every write before visible to the host; a park's call
    /// goes on once the host answers it (`TimelineInner::answer`), a
    /// publication's next call at once. `None` where the node binds nothing: the
    /// device's kernels park themselves, it was bound answered, or the run
    /// encodes none of its call's launches (their conditions leave them
    /// out), so nothing waits.
    pub fn park(&self, port: NativeTimelinePort) -> Option<u64> {
        if port.workflow != self.plan.identity {
            return None;
        }
        self.parks
            .iter()
            .find(|&&(node, _)| node == port.ordinal as usize)
            .map(|&(_, value)| value)
    }

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
    /// How the nodes bind each port (`Executable::port_uses`).
    uses: Vec<PortUse>,
    /// Ports whose binding was checked by `bind_static`.
    checked: Vec<bool>,
    /// Per timeline node, how the run binds it.
    timelines: Vec<Option<TimelineBinding>>,
}

/// How a run binds a timeline node.
#[derive(Clone)]
enum TimelineBinding {
    /// To a timeline at a value.
    Bound(Arc<TimelineInner>, u64),
    /// A park to a timeline, at the value the run takes when queued
    /// ([`NativeGraphBindings::set_park`]).
    Parked(Arc<TimelineInner>),
    /// A park whose answer the host wrote before the run is submitted
    /// ([`NativeGraphBindings::set_answered`]): it binds no operation.
    Answered,
}

impl NativeGraphBindings {
    /// Bind timeline node `port` to `timeline` at `value` for this run.
    pub fn set_timeline(
        &mut self,
        port: NativeTimelinePort,
        timeline: Arc<TimelineInner>,
        value: u64,
    ) -> Result<(), WorkflowError> {
        if port.workflow != self.identity {
            return Err(WorkflowError::CrossWorkflowResult);
        }
        let slot = self
            .timelines
            .get_mut(port.ordinal as usize)
            .ok_or(WorkflowError::CrossWorkflowResult)?;
        if slot.is_some() {
            return Err(WorkflowError::NativeTimelineAlreadyBound {
                node: port.ordinal as usize,
            });
        }
        *slot = Some(TimelineBinding::Bound(timeline, value));
        Ok(())
    }

    /// Bind park node `port` to `timeline`. Where the device's kernels wait
    /// for the host themselves (`BackendName::parks_in_kernels`) it binds
    /// no operation; elsewhere, when the run is queued and encodes a launch
    /// of the park's call, the park takes the timeline's next park value
    /// `v` (`TimelineInner::next_park`): the work before it signals `v`
    /// and the call waits for `v + 1`, the host's answer. The run's outputs
    /// report the value ([`NativeGraphOutputs::park`]). A publication
    /// (`TimelineNode::Publish`) binds the same way and takes a value as
    /// the call before it launches, which the work before it signals and
    /// nothing waits for; no output reports it. The timeline's values are
    /// the runtime's: no run binds it at a value of its own.
    pub fn set_park(&mut self, port: NativeTimelinePort, timeline: Arc<TimelineInner>) -> Result<(), WorkflowError> {
        if port.workflow != self.identity {
            return Err(WorkflowError::CrossWorkflowResult);
        }
        let slot = self
            .timelines
            .get_mut(port.ordinal as usize)
            .ok_or(WorkflowError::CrossWorkflowResult)?;
        if slot.is_some() {
            return Err(WorkflowError::NativeTimelineAlreadyBound {
                node: port.ordinal as usize,
            });
        }
        *slot = Some(TimelineBinding::Parked(timeline));
        Ok(())
    }

    /// Bind park node `port` as answered before the run is submitted: the
    /// words its call awaits hold their answer already, so the call waits
    /// for nothing and the park binds no operation on any backend. A run of
    /// only such parks is the same work each time, so a measurement point
    /// may run it.
    pub fn set_answered(&mut self, port: NativeTimelinePort) -> Result<(), WorkflowError> {
        if port.workflow != self.identity {
            return Err(WorkflowError::CrossWorkflowResult);
        }
        let slot = self
            .timelines
            .get_mut(port.ordinal as usize)
            .ok_or(WorkflowError::CrossWorkflowResult)?;
        if slot.is_some() {
            return Err(WorkflowError::NativeTimelineAlreadyBound {
                node: port.ordinal as usize,
            });
        }
        *slot = Some(TimelineBinding::Answered);
        Ok(())
    }

    /// Whether every timeline node is a park bound answered
    /// ([`Self::set_answered`]): the run waits on and signals nothing.
    pub(super) fn answered_only(&self) -> bool {
        self.timelines.iter().all(|binding| matches!(binding, Some(TimelineBinding::Answered)))
    }

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
        if self.uses[ordinal].grid && tensor.is_slab_view() {
            return Err(WorkflowError::NativePortSlabGrid { port: ordinal });
        }
        if self.uses[ordinal].direct && tensor.chunked() {
            return Err(WorkflowError::ChunkedBinding {
                parameter: format!("port {ordinal}"),
            });
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
    /// The run's timeline operations, each with the kernel node it precedes.
    timelines: Vec<(usize, TimelineOperation)>,
    /// Its parks and publications bound to a timeline whose device's
    /// kernels do not park, in node order: each takes a value when the run
    /// is queued.
    parks: Vec<PendingPark>,
}

/// A park or a publication a run binds to a timeline
/// ([`NativeGraphBindings::set_park`]).
struct PendingPark {
    /// Its timeline node.
    node: usize,
    /// The kernel node it precedes: the call that parks, or the one after
    /// the call a publication publishes.
    position: usize,
    timeline: Arc<TimelineInner>,
    /// A publication (`TimelineNode::Publish`): of the call before
    /// `position`, signalling its value and waiting for nothing.
    publishes: bool,
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
        let mut sequence = NativeGraphSequence::new(&self.slot.plan.device, NativeQueue::Compute);
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
        self.queue_stamped(sequence, None)
    }

    /// [`Self::queue`], the run sampled for its spans' device time: each of
    /// its plan's marks (`NativeGraphDraft::span_mark`) stamped as the
    /// device passes it, read through the returned spans once the run's
    /// work completes (`spans`). Why the device's backend times no span.
    pub fn queue_sampled(
        self,
        sequence: &mut NativeGraphSequence,
    ) -> Result<(NativeGraphOutputs, NativeSpans), CallError> {
        let spans = NativeSpans::new(&self.slot.plan.device, self.slot.plan.executable.marks.len())?;
        let outputs = self.queue_stamped(sequence, Some(&spans))?;
        Ok((outputs, spans))
    }

    fn queue_stamped(
        self,
        sequence: &mut NativeGraphSequence,
        spans: Option<&NativeSpans>,
    ) -> Result<NativeGraphOutputs, CallError> {
        let plan = self.slot.plan.clone();
        if !Arc::ptr_eq(&plan.device, &sequence.device) {
            return Err(CallError::Workflow(WorkflowError::NativeGraphSlotMismatch));
        }
        // A record port's tensor's records, through the table kept with it.
        let externals = self
            .externals
            .iter()
            .zip(&plan.executable.port_uses)
            .map(|(tensor, uses)| {
                tensor
                    .as_ref()
                    .map(|tensor| {
                        Ok::<_, TensorError>(External {
                            allocation: tensor.allocation().clone(),
                            offset: tensor.byte_offset(),
                            placement: tensor.record_placement(),
                            records: uses.records.then(|| tensor.records()).transpose()?,
                        })
                    })
                    .transpose()
            })
            .collect::<Result<Vec<_>, TensorError>>()
            .map_err(|error| CallError::Workflow(WorkflowError::TensorView(error)))?;
        sequence.access.extend(
            self.externals
                .iter()
                .zip(&plan.executable.external_writes)
                .flat_map(|(tensor, write)| {
                    tensor
                        .iter()
                        .flat_map(|tensor| tensor.bound_allocations())
                        .map(|allocation| (allocation, *write))
                })
                // The slot's workspace (a family's arena, which every graph
                // of the family shares) only where the plan places in it:
                // a run that places nothing there touches none of it, and
                // orders against no other run's use of it.
                .chain(
                    self.slot
                        .workspace
                        .iter()
                        .filter(|_| plan.scratch_bytes > 0)
                        .map(|allocation| (allocation.clone(), true)),
                )
                .chain(
                    self.slot
                        .workspace_records
                        .iter()
                        .filter(|_| plan.scratch_bytes > 0)
                        .map(|allocation| (allocation.clone(), false)),
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
        sequence.binding_leases.extend(
            self.externals
                .iter()
                .flatten()
                .filter(|tensor| tensor.holds_binding_lease())
                .cloned(),
        );
        let slabs = self
            .externals
            .iter()
            .enumerate()
            .filter_map(|(port, tensor)| {
                tensor
                    .as_ref()
                    .and_then(|tensor| tensor.slab_reach())
                    .map(|slab| (port, slab))
            })
            .collect();
        // Where sampled, each of its plan's marks stamped at its node.
        let stamps = spans
            .into_iter()
            .flat_map(|spans| plan.executable.marks.iter().copied().zip(spans.stamps()))
            .collect();
        let mut run = QueuedRun {
            plan,
            externals,
            slabs,
            workspace: self.slot.workspace.clone(),
            workspace_records: self.slot.workspace_records.clone(),
            upload: self.slot.upload.clone(),
            outputs: self.exports.arena.clone(),
            timelines: self.timelines,
            stamps,
            left_out: Vec::new(),
            hold: self.slot.hold.clone(),
        };
        // The run's launches are decided here, once: its encoding and its
        // parks follow the same conditions.
        run.left_out = run.left_out();
        let mut exports = self.exports;
        exports.parks.clear();
        for park in self.parks {
            // A park's call is the node after it; a publication's, the one
            // before (`validate_park`: a publication follows a call).
            let call = park.position - usize::from(park.publishes);
            let encodes = run.plan.executable.nodes[call]
                .launches
                .geometry
                .iter()
                .enumerate()
                .any(|(ordinal, launch)| launch.is_some() && run.encodes(call, ordinal));
            if !encodes {
                continue;
            }
            let value = park.timeline.next_park();
            // Every value a node took is the run's output, a publication's as
            // a park's: its host learns that the device reached the node, its
            // every write before it visible, only by its value.
            exports.parks.push((park.node, value));
            if park.publishes {
                run.timelines.push((
                    park.position,
                    TimelineOperation {
                        action: TimelineAction::Signal,
                        timeline: park.timeline.clone(),
                        value,
                    },
                ));
                continue;
            }
            park.timeline.await_at(value + 1);
            for (action, value) in [(TimelineAction::Signal, value), (TimelineAction::Wait, value + 1)] {
                run.timelines.push((
                    park.position,
                    TimelineOperation {
                        action,
                        timeline: park.timeline.clone(),
                        value,
                    },
                ));
            }
        }
        sequence.runs.push(run);
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
    /// The queue its submission runs on.
    queue: NativeQueue,
    runs: Vec<QueuedRun>,
    /// Every allocation the queued runs touch, with whether they write it.
    access: Vec<(Arc<Allocation>, bool)>,
    /// Keep slab placements and pinned reserved regions fixed until the
    /// submission records its fences.
    binding_leases: Vec<Arc<TensorInner>>,
}

impl NativeGraphSequence {
    /// An empty sequence whose submission runs on `queue`.
    pub fn new(device: &Arc<DeviceInner>, queue: NativeQueue) -> Self {
        Self {
            device: device.clone(),
            queue,
            runs: Vec::new(),
            access: Vec::new(),
            binding_leases: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.runs.is_empty()
    }

    /// Submit every queued run without waiting. The completion reports the
    /// outcome of all of them.
    pub fn submit(self) -> Result<NativeGraphCompletion, CallError> {
        self.submit_observed(false)
    }

    /// [`Self::submit`], observing the work's device interval where
    /// `observed` (a measurement's).
    pub(super) fn submit_observed(self, observed: bool) -> Result<NativeGraphCompletion, CallError> {
        let access = merge_access(self.access.iter().cloned());
        // The work keeps its plans and the timelines it waits on or signals.
        let retained: Arc<dyn std::any::Any + Send + Sync> = Arc::new((
            self.runs
                .iter()
                .map(|run| run.plan.clone())
                .collect::<Vec<_>>(),
            self.runs
                .iter()
                .flat_map(|run| run.timelines.iter())
                .map(|(_, operation)| operation.timeline.clone())
                .collect::<Vec<_>>(),
        ));
        let nodes = self
            .runs
            .iter()
            .enumerate()
            .flat_map(|(run, queued)| {
                (0..queued.plan.executable.nodes.len()).map(move |node| (run, node))
            })
            .collect::<Vec<_>>();
        // A run's operation before its node `n` is at the boundary before
        // that node's dispatch; the runs' dispatches follow one another.
        let mut boundaries = vec![Vec::new(); nodes.len() + 1];
        let mut stamps = vec![Vec::new(); nodes.len() + 1];
        let mut first = 0;
        for run in &self.runs {
            for (position, operation) in &run.timelines {
                boundaries[first + position].push(operation.clone());
            }
            for (position, stamp) in &run.stamps {
                stamps[first + position].push(stamp.clone());
            }
            first += run.plan.executable.nodes.len();
        }
        let list = SequenceList {
            runs: &self.runs,
            nodes,
            boundaries,
            stamps,
        };
        let mut submission = submit(&list, &access, retained, 1, observed, self.queue)?;
        // The runs own the storage their work reaches; it retires with this
        // completion, never with the replay caches the plans' retention
        // reaches, GraphPoint conversion included. The binding leases only
        // fixed placement until the submission recorded its fences: they
        // end here, and a slab freed or a region shrunk from now on is
        // released physically once the work completes (allocation fences).
        submission._retained = Arc::new(GraphRetention { _runs: self.runs });
        drop(self.binding_leases);
        Ok(NativeGraphCompletion { submission })
    }
}

/// The storage a sequence's runs reach, transferred to its completion
/// alone. Replay caches must never retain it.
struct GraphRetention {
    _runs: Vec<QueuedRun>,
}

/// One queued run: its plan's sealed nodes over the storage it binds.
struct QueuedRun {
    plan: Arc<NativeGraphPlan>,
    /// Per port, the bound tensor (`None` for graph locals).
    externals: Vec<Option<External>>,
    /// Each port bound to a slab view, with what it reaches (the
    /// encodings order by it).
    slabs: Vec<(usize, super::hazard::SlabReach)>,
    workspace: Option<Arc<Allocation>>,
    /// The slot's workspace record table (`NativeGraphSlot::workspace_records`).
    workspace_records: Option<Arc<Allocation>>,
    upload: Option<Arc<Allocation>>,
    outputs: Option<Arc<Allocation>>,
    /// Its timeline operations, each with the node it precedes; a park's
    /// last at its node.
    timelines: Vec<(usize, TimelineOperation)>,
    /// Where it is sampled, its plan's marks' stamps, each with the node it
    /// precedes; held until its work completes.
    stamps: Vec<(usize, Stamp)>,
    /// The launches, (node, ordinal), its launch conditions leave out, read
    /// when the run is queued.
    left_out: Vec<(usize, usize)>,
    /// Its slot's graphs in the device's graph budget, held until the run
    /// is recorded and replayed, and the form of its layout's graph there.
    hold: Arc<super::SlotGraphs>,
}

/// The tensor a queued run binds to an external port.
struct External {
    allocation: Arc<Allocation>,
    /// The view's first byte.
    offset: u64,
    /// Where the view's records lie, where it has records: packed in its
    /// chunks on chunked storage (a host port's view; Metal's encoding
    /// orders by them).
    placement: Option<RecordPlacement>,
    /// Where a record port binds the port, the view's records.
    records: Option<Arc<Records>>,
}

impl QueuedRun {
    /// The tensor bound to external port `port`.
    fn external(&self, port: u32) -> &External {
        self.externals[port as usize]
            .as_ref()
            .expect("attach bound every external port")
    }

    /// The records of the tensor bound to `port`, which a record port binds.
    fn records(&self, port: u32) -> &Records {
        self.external(port)
            .records
            .as_ref()
            .expect("a run forms the records of every port a record port binds")
    }

    /// Node `node`; the buffers its kernel binds are appended to `buffers`
    /// (`DispatchList::dispatch`).
    fn dispatch<'s>(&'s self, node: usize, buffers: &mut Vec<(&'s Allocation, u64)>) -> Dispatch<'s> {
        let run = self;
        let node = &self.plan.executable.nodes[node];
        let region = |region: &'s Option<Arc<Allocation>>| {
            &**region
                .as_ref()
                .expect("a planned graph storage region is present")
        };
        node.kernel.bind(node.sites.iter().map(|site| match site.region {
            Region::External(port) if site.records != 0 => {
                let records = run.records(port);
                let entry = site.offset / records.placement.record();
                (
                    &*records.table,
                    records.table_offset + seismic_lang::registry::record_table_entry(entry),
                )
            }
            Region::Workspace if site.records != 0 => (region(&run.workspace_records), site.table),
            Region::External(port) => {
                let external = run.external(port);
                (&*external.allocation, external.offset + site.offset)
            }
            Region::Workspace => (region(&run.workspace), site.offset),
            Region::Upload => (region(&run.upload), site.offset),
            Region::Outputs => (region(&run.outputs), site.offset),
        }), buffers);
        Dispatch {
            kernel: &node.kernel,
            words: &node.words,
            launches: &node.launches.geometry,
            rounds: node.launches.rounds,
            representations: &node.representations,
        }
    }

    /// The launches, (node, ordinal), whose launch conditions' words, as
    /// the host reads them now, leave them out.
    fn left_out(&self) -> Vec<(usize, usize)> {
        let holds = |condition: &LaunchCondition| {
            let (allocation, base) = match condition.word {
                HostWord::Port(port) => {
                    let external = self.external(port);
                    (&*external.allocation, external.offset)
                }
                HostWord::Upload => (&**self.upload.as_ref().expect("a run reading its upload region holds one"), 0),
            };
            super::condition_holds(allocation, base + condition.offset, condition.encoded)
        };
        self.plan
            .executable
            .nodes
            .iter()
            .enumerate()
            .flat_map(|(node, sealed)| {
                sealed
                    .conditions
                    .iter()
                    .enumerate()
                    .filter(|(_, conditions)| !conditions.iter().all(holds))
                    .map(move |(ordinal, _)| (node, ordinal))
            })
            .collect()
    }

    /// Whether launch `ordinal` of node `node` is encoded.
    fn encodes(&self, node: usize, ordinal: usize) -> bool {
        !self.left_out.contains(&(node, ordinal))
    }
}

/// A sequence's runs as the encoder reads them: every node of every run.
struct SequenceList<'a> {
    runs: &'a [QueuedRun],
    /// (run, node) of each dispatch, in order.
    nodes: Vec<(usize, usize)>,
    /// Per dispatch boundary (before dispatch `i`, `nodes.len()` after the
    /// last), the timeline operations there, in run order.
    boundaries: Vec<Vec<TimelineOperation>>,
    /// Per dispatch boundary, the sampled runs' stamps there, in run order.
    stamps: Vec<Vec<Stamp>>,
}

/// Instantiate `count` graphs of each segment of a run of the first of
/// `plans` that replays graphs, for a slot as it is created: on CUDA,
/// graphs of the plans' one layout whose nodes are the launches `form`
/// holds, which those plans make (`cuda::instantiate`), over placeholder
/// storage a replay binds to its run's ([`PlaceholderRun`]). A launched
/// plan replays none ([`NativeGraphPlan::graphed`]). Elsewhere a graph is
/// recorded at its shape's first use. Each graph instantiated is pushed to
/// `owned`, the slot's ([`super::SlotGraphs`]).
fn instantiate(plans: &[Arc<NativeGraphPlan>], form: Arc<[u64]>, count: usize, owned: &mut Vec<(Vec<u64>, u64)>) -> Result<(), TensorError> {
    assert!(
        plans.iter().all(|plan| plan.shape.layout == plans[0].shape.layout),
        "a graph is one layout's"
    );
    let plans = plans.iter().filter(|plan| plan.graphed()).cloned().collect::<Vec<_>>();
    let Some(first) = plans.first() else {
        return Ok(());
    };
    let device = &first.device;
    let crate::backends::OpenedKind::Cuda(opened) = &device.kind else {
        return Ok(());
    };
    let placeholder = device.allocate_zeroed(BUFFER_ALIGNMENT, BUFFER_ALIGNMENT)?;
    let runs = plans
        .iter()
        .map(|plan| PlaceholderRun { plan, placeholder: &placeholder, form: form.clone() })
        .collect::<Vec<_>>();
    // A graph's nodes launch every plan's kernels it was made of.
    let retained: Arc<dyn std::any::Any + Send + Sync> = Arc::new(plans.to_vec());
    super::cuda::instantiate(
        opened.service(),
        &device.native.cuda_replays,
        device.native.graph_budget(),
        &runs,
        count,
        &retained,
        owned,
    )
    .map_err(TensorError::Execution)
}

/// A sealed plan's run over placeholder storage: every buffer it binds at
/// `placeholder`, with its sealed words and geometry and every launch
/// encoded. A slot instantiates its graphs from it
/// (`NativeGraphPlan::instantiate`); the run's addresses and encoding are
/// parameters a replay sets, so its graphs are those of every run of its
/// shape.
struct PlaceholderRun<'a> {
    plan: &'a NativeGraphPlan,
    placeholder: &'a Allocation,
    /// The words of its layout's graph (`super::LayoutForms`).
    form: Arc<[u64]>,
}

impl DispatchList for PlaceholderRun<'_> {
    fn count(&self) -> usize {
        self.plan.executable.nodes.len()
    }

    fn dispatch<'s>(&'s self, index: usize, buffers: &mut Vec<(&'s Allocation, u64)>) -> Dispatch<'s> {
        let node = &self.plan.executable.nodes[index];
        node.kernel.bind(node.sites.iter().map(|_| (self.placeholder, 0)), buffers);
        Dispatch {
            kernel: &node.kernel,
            words: &node.words,
            launches: &node.launches.geometry,
            rounds: node.launches.rounds,
            representations: &node.representations,
        }
    }

    fn encodes(&self, _index: usize, _ordinal: usize) -> bool {
        true
    }

    /// None: a graph's launches keep stream order, whatever they reach.
    fn reaches<'s>(&'s self, _index: usize, _reaches: &mut Vec<Reach<'s>>) {}

    fn runs(&self) -> Option<Vec<super::PlanRun>> {
        let bounds = self.plan.executable.bounds();
        Some(vec![super::PlanRun {
            plan: self.plan.identity,
            layout: self.plan.shape.layout,
            calls: 0..self.count(),
            bounds,
            marks: self.plan.held_marks(),
            launched: !self.plan.graphed(),
            form: self.form.clone(),
        }])
    }
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
        self.runs[run].dispatch(node, buffers)
    }

    fn encodes(&self, index: usize, ordinal: usize) -> bool {
        let (run, node) = self.nodes[index];
        self.runs[run].encodes(node, ordinal)
    }

    fn reaches<'s>(&'s self, index: usize, reaches: &mut Vec<Reach<'s>>) {
        let (run, node) = self.nodes[index];
        let run = &self.runs[run];
        let region = |region: &'s Option<Arc<Allocation>>| {
            &**region
                .as_ref()
                .expect("a planned graph storage region is present")
        };
        for (buffer, site) in run.plan.executable.nodes[node].sites.iter().enumerate() {
            let (allocation, offset, bytes) = match site.region {
                Region::External(port) if site.records != 0 => {
                    let records = run.records(port);
                    let first = site.offset / records.placement.record();
                    records.extend(buffer, first, site.records, reaches);
                    continue;
                }
                // A workspace record site's records, which its table
                // addresses; the entries order nothing (`Records::extend`).
                Region::Workspace if site.records != 0 => {
                    reaches.push(Reach {
                        buffer,
                        allocation: region(&run.workspace),
                        offset: site.offset,
                        bytes: site.bytes,
                        through: Through::Slab,
                    });
                    continue;
                }
                Region::External(port) => {
                    let External { allocation, offset: start, placement, .. } = run.external(port);
                    if let Some((_, slab)) =
                        run.slabs.iter().find(|(slabbed, _)| *slabbed == port as usize)
                    {
                        slab.extend(buffer, allocation, start + site.offset, reaches);
                        continue;
                    }
                    // A host port's view of chunked storage reaches its
                    // records as packed.
                    let (offset, bytes) = match placement {
                        Some(placement) => placement.reach(site.offset, site.bytes),
                        None => (start + site.offset, site.bytes),
                    };
                    (&**allocation, offset, bytes)
                }
                Region::Workspace => (region(&run.workspace), site.offset, site.bytes),
                Region::Upload => (region(&run.upload), site.offset, site.bytes),
                Region::Outputs => (region(&run.outputs), site.offset, site.bytes),
            };
            reaches.push(Reach {
                buffer,
                allocation,
                offset,
                bytes,
                through: Through::Binding,
            });
        }
    }

    fn runs(&self) -> Option<Vec<super::PlanRun>> {
        let mut first = 0;
        Some(
            self.runs
                .iter()
                .map(|run| {
                    let calls = first..first + run.plan.executable.nodes.len();
                    first = calls.end;
                    let bounds = run.plan.executable.bounds();
                    let form = run.hold.form(run.plan.shape.layout);
                    super::PlanRun {
                        plan: run.plan.identity,
                        layout: run.plan.shape.layout,
                        calls,
                        bounds,
                        marks: run.plan.held_marks(),
                        launched: !run.plan.graphed(),
                        form,
                    }
                })
                .collect(),
        )
    }

    fn timelines(&self, boundary: usize) -> &[TimelineOperation] {
        &self.boundaries[boundary]
    }

    fn has_timelines(&self) -> bool {
        self.boundaries.iter().any(|operations| !operations.is_empty())
    }

    fn stamps(&self, boundary: usize) -> &[Stamp] {
        &self.stamps[boundary]
    }
}

/// The outcome of one submitted native graph run.
#[must_use = "a native graph completion reports whether the run succeeded"]
pub struct NativeGraphCompletion {
    submission: NativeSubmission,
}

impl NativeGraphCompletion {
    /// Whether the run ended, so [`Self::wait`] reports its outcome without
    /// blocking: it completed, or faulted.
    pub fn is_complete(&self) -> bool {
        self.submission.is_complete()
    }
    /// Wait for the run and report its outcome.
    pub fn wait(self) -> Result<(), CallError> {
        self.submission.wait()
    }
    /// The submission of the run (every run of a sequence), whose device
    /// time its trace reports ([`super::GraphPoint`]).
    pub(super) fn into_submission(self) -> NativeSubmission {
        self.submission
    }
}

#[cfg(test)]
mod layout_certificate_tests {
    use super::*;

    fn class(
        rows: u64,
    ) -> (
        Vec<PortSpec>,
        Vec<PlannedNode>,
        Vec<Vec<Option<NativeTensorSpec>>>,
    ) {
        let representation = seismic_lang::registry::representation("f32").unwrap();
        let canonical = layout::canonical(representation, &[rows]).unwrap();
        let port = PortSpec {
            representation,
            extents: vec![rows],
            strides: canonical.strides.clone(),
            byte_len: canonical.byte_len,
            local: true,
            owned_input: false,
            prewritten: false,
        };
        let mut args = EncodedWorkflowArgs::new();
        args.push_result_tensor(WorkflowResultRef {
            workflow: 1,
            node: INPUT_NODE,
            result: 0,
        });
        let node = PlannedNode {
            args,
            scratch: vec![ScratchNeed {
                bytes: rows * 4,
                sync: false,
            }],
            condition: None,
            share: None,
        };
        let result = NativeTensorSpec {
            representation,
            extents: vec![rows],
            strides: canonical.strides,
            byte_len: canonical.byte_len,
        };
        (vec![port], vec![node], vec![vec![Some(result)]])
    }

    #[test]
    fn dominating_layout_accepts_smaller_exact_graph_and_rejects_larger_one() {
        let (ports, nodes, results) = class(16);
        let certified =
            make_layout(&ports, &nodes, &results, &[], &[], HostHeap::default(), 0, GraphCensus::default());
        let charge = certified.storage_bytes();
        let (ports, nodes, results) = class(8);
        let exact = certified
            .storage_for(&ports, &nodes, &results, &[], &[])
            .unwrap();
        assert_eq!(exact.scratch_bytes, charge.workspace);
        let (ports, nodes, results) = class(32);
        assert!(matches!(
            certified.storage_for(&ports, &nodes, &results, &[], &[]),
            Err(WorkflowError::NativeGraphLayoutMismatch)
        ));
    }
}

#[cfg(test)]
mod liveness_placement_tests {
    use super::*;

    fn interval(start: usize, end: usize, bytes: u64) -> Interval {
        Interval {
            key: StorageKey::Result(0, 0),
            start,
            end,
            bytes,
            exported: false,
        }
    }

    fn place(intervals: &[Interval]) -> (Vec<u64>, u64) {
        place_by_liveness(&intervals.iter().collect::<Vec<_>>())
    }

    fn floor(intervals: &[Interval]) -> u64 {
        liveness_floor(&intervals.iter().collect::<Vec<_>>())
    }

    fn assert_disjoint(intervals: &[Interval], offsets: &[u64]) {
        for (a, first) in intervals.iter().enumerate() {
            for (b, second) in intervals.iter().enumerate().skip(a + 1) {
                if first.overlaps(second) {
                    let (x, y) = (offsets[a], offsets[b]);
                    assert!(
                        x + first.bytes <= y || y + second.bytes <= x,
                        "live buffers {a} and {b} alias"
                    );
                }
            }
        }
    }

    #[test]
    fn buffers_with_disjoint_lifetimes_share_storage() {
        let intervals = [interval(0, 1, 1024), interval(2, 3, 1024)];
        let (offsets, size) = place(&intervals);
        assert_eq!(offsets, vec![0, 0]);
        assert_eq!(size, 1024);
    }

    #[test]
    fn a_shared_endpoint_is_an_overlap() {
        // A node's inputs end where its results start.
        let intervals = [interval(0, 3, 512), interval(3, 5, 512)];
        let (offsets, size) = place(&intervals);
        assert_disjoint(&intervals, &offsets);
        assert_eq!(size, 1024);
    }

    #[test]
    fn freed_neighbours_serve_one_larger_buffer() {
        // Two small buffers die, then one buffer as large as both together
        // is born. First-fit block reuse cannot combine the two freed blocks;
        // placement by liveness reaches the peak of live bytes.
        let intervals = [
            interval(0, 1, 2048),
            interval(0, 1, 2048),
            interval(2, 3, 4096),
        ];
        let (offsets, size) = place(&intervals);
        assert_disjoint(&intervals, &offsets);
        assert_eq!(size, 4096);
        assert_eq!(floor(&intervals), 4096);
    }

    #[test]
    fn a_long_lived_buffer_does_not_strand_a_large_block() {
        // A small long-lived buffer born while a large one is free must not
        // take the large block's offset ahead of a later large buffer.
        let intervals = [
            interval(0, 1, 8192),
            interval(2, 9, 256),
            interval(3, 4, 8192),
        ];
        let (offsets, size) = place(&intervals);
        assert_disjoint(&intervals, &offsets);
        assert_eq!(size, 8192 + 256);
        assert_eq!(floor(&intervals), 8192 + 256);
    }

    #[test]
    fn placement_is_deterministic() {
        let intervals = [
            interval(0, 4, 1000),
            interval(1, 2, 1000),
            interval(3, 6, 1000),
            interval(5, 7, 3000),
        ];
        assert_eq!(place(&intervals), place(&intervals));
        let (offsets, _) = place(&intervals);
        assert_disjoint(&intervals, &offsets);
        assert!(offsets.iter().all(|offset| offset % BUFFER_ALIGNMENT == 0));
    }
}

#[cfg(test)]
mod upload_region_tests {
    use super::*;

    /// No region holding the input is a slot too small for it; of those
    /// holding it, the smallest no run has taken.
    #[test]
    fn a_launch_takes_the_smallest_free_region_holding_its_input() {
        let device = crate::devices::Devices::discover(&[])
            .unwrap()
            .open_backend(seismic_lang::registry::BackendName::Cpu)
            .unwrap();
        let uploads = [64, 256]
            .map(|bytes| device.allocate_upload(bytes, BUFFER_ALIGNMENT).unwrap())
            .to_vec();
        assert!(matches!(
            idle_upload(&uploads, 512),
            Err(WorkflowError::UploadRegionTooSmall { bytes: 512, largest: 256 })
        ));
        // The free region holding it, the smallest such; a taken one is
        // passed over.
        let taken = idle_upload(&uploads, 100).unwrap();
        assert_eq!(taken.bytes(), 256);
        assert_eq!(idle_upload(&uploads, 10).unwrap().bytes(), 64);
    }

    /// A launch whose runs take more regions than its slot holds breaks the
    /// slot's sizing: it panics, never waits.
    #[test]
    #[should_panic(expected = "take no more upload regions than its slot holds")]
    fn a_run_beyond_the_slots_regions_panics() {
        let device = crate::devices::Devices::discover(&[])
            .unwrap()
            .open_backend(seismic_lang::registry::BackendName::Cpu)
            .unwrap();
        let uploads = vec![device.allocate_upload(64, BUFFER_ALIGNMENT).unwrap()];
        let _taken = idle_upload(&uploads, 10).unwrap();
        let _ = idle_upload(&uploads, 10);
    }
}

#[cfg(all(test, target_os = "macos"))]
mod condition_tests {
    use super::*;
    use seismic_lang::checked::{Encoded, SourceFile, SourceSet, check_source};
    use seismic_lang::registry::BackendName;

    /// `fill`, whose one launch its kernel conditions on `policy[2]`.
    fn fill(device: &Arc<DeviceInner>) -> Arc<NativePreparedAny> {
        let mut module = check_source(SourceSet::new(vec![SourceFile {
            path: "policy.seismic".into(),
            text: "fn fill[N](policy: &tensor[3] i32, x: &mut tensor[N] f32):\n    for i in 0..N:\n        x[i] = 0.0\n\nnative fill for metal from \"fill.metal\":\n    launch fill:\n        threadgroups (1, 1, 1)\n        unless (policy[2])\n        threads_per_threadgroup (1, 1, 1)\n".into(),
        }]))
        .expect("policy source checks");
        let entry = module.entry_named("fill").expect("fill entry");
        module
            .capture_native_asset(entry, BackendName::Metal, "kernel void fill() {}\n".into())
            .expect("asset capture");
        let inner = NativePrepared::prepare(
            device,
            &module,
            entry,
            seismic_lang::entry::ElementBindings::default(),
            super::super::NativeSpecialization::new(),
            None,
        )
        .expect("fill prepares");
        Arc::new(NativePreparedAny { inner })
    }

    fn args(policy: NativePort, x: NativePort) -> EncodedWorkflowArgs {
        let mut args = EncodedWorkflowArgs::new();
        args.push_result_tensor(policy.reference);
        args.push_result_tensor(x.reference);
        args
    }

    /// One mechanism carries a node's launch conditions: its kernel's, of
    /// the word its site binds, and the node's own on every launch, each a
    /// word of a tensor the host writes; a kernel's condition bound to graph
    /// storage the host does not write is refused as the graph forms.
    #[test]
    #[ignore = "requires Metal device"]
    fn a_node_carries_its_kernels_and_its_own_launch_conditions() {
        let device = crate::devices::Devices::discover(&[])
            .expect("device discovery")
            .open_backend(BackendName::Metal)
            .expect("Metal device");
        let kernel = fill(&device);
        let i32s = seismic_lang::registry::dense(seismic_lang::types::DType::I32);
        let f32s = seismic_lang::registry::dense(seismic_lang::types::DType::F32);
        let mut draft = NativeGraphDraft::new(&device);
        let policy = draft.port(i32s, &[3]).unwrap();
        let x = draft.port(f32s, &[4]).unwrap();
        draft
            .enqueue_under(&kernel, args(policy, x), NodeCondition { port: policy, index: 0, encoded: Encoded::WhileSet })
            .unwrap();
        let plan = draft.seal().expect("a graph over host-written words seals");
        let word = HostWord::Port(policy.reference.result);
        assert_eq!(
            plan.executable.nodes[0].conditions,
            [vec![
                LaunchCondition { word, offset: 8, encoded: Encoded::WhileClear },
                LaunchCondition { word, offset: 0, encoded: Encoded::WhileSet },
            ]]
        );
        assert!(plan.executable.port_uses[policy.reference.result as usize].condition);
        assert!(!plan.executable.port_uses[x.reference.result as usize].condition);

        // The kernel's word in graph storage: a local, which no host writes.
        let mut draft = NativeGraphDraft::new(&device);
        let policy = draft.local_for(&kernel, "policy", &[]).unwrap();
        let x = draft.port(f32s, &[4]).unwrap();
        draft.enqueue(&kernel, args(policy, x)).unwrap();
        assert!(matches!(
            draft.seal(),
            Err(CallError::Workflow(WorkflowError::ConditionWord { node: 0 }))
        ));
    }
}
