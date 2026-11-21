//! Checked direct-native graph construction and reusable result storage.
//!
//! A graph is sealed from the generated entry contracts before resident model
//! storage is admitted. Ports are the only late bindings. No consumer supplies
//! a name-to-buffer recipe for intermediate results.
//!
//! Sealing validates every node against its entry contract once and fixes
//! its executable form: argument words, launch geometry, and for every
//! buffer the storage region and offset it lives at. A run supplies only the
//! regions (the slot's workspace and sync region, the run's upload region
//! and output arena)
//! and the tensors bound to external ports; attaching checks those bindings
//! alone, so per-run host work does not revalidate nodes.

#[cfg(any(test, target_os = "macos"))]
use super::hazard::{Reach, Through};
use super::point::{GraphPoint, GraphTiming};
use super::spans::{NativeSpans, Stamp};
use super::timeline::{
    Parks, TimelineAction, TimelineHost, TimelineInner, TimelineNode, TimelineNodeKind, TimelineOperation,
    TimelineRole, ValuedRole,
};
use super::{
    charge_emissions, merge_access, submit, CallLaunches, Dispatch, DispatchList, FormedAt, LaunchGeometry,
    NativeOperandNorm, NativePrepared, NativeQueue, NativeRoute, NativeSubmission, NativeTensorSpec, ScratchNeed,
    ScratchOperand, BUFFER_ALIGNMENT, MINIMUM_SCRATCH_BYTES,
};
use crate::api::device::DeviceInner;
use crate::api::kernel::{
    EncodedScalar, EncodedWorkflowArgs, EncodedWorkflowArgument, NativePreparedAny, PendingWorkflowResults,
    ViewOperation, WorkflowResultRef, WorkflowTensorArgument,
};
use crate::api::tensor::{exclusive_pairs, meet, Reaching, Records, TensorInner};
use crate::api::{CallError, GraphBuffer, LayoutMismatch, TensorError, WorkflowError};
use crate::driver::Allocation;
use crate::layout;
use crate::memory::Charges;
use seismic_compiler::errors::{ExecutionError, InvocationError};
use seismic_compiler::graph_cost::{argument_bytes, GraphCensus, LaunchGraphs, LaunchedGraph};
use seismic_compiler::prepared::{ArgumentValue, DeviceIdentity, TensorDescriptor};
use seismic_lang::checked::NativeImplementation;
use seismic_lang::entry::{CallSchema, ParameterKind, ResultKind};
use seismic_lang::ids::RepresentationId;
use seismic_lang::registry::BackendName;
use std::marker::PhantomData;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};

static NEXT_NATIVE_GRAPH: AtomicU64 = AtomicU64::new(1);
pub const INPUT_NODE: u32 = u32::MAX;
const _: () = assert!(BUFFER_ALIGNMENT == seismic_memory::BUFFER_ALIGNMENT);

/// Host heap a structure holds: the bytes it asked the allocator for and in
/// how many allocations. The allocator's own rounding and bookkeeping of
/// each allocation are not part of `bytes`; `rounding` bounds them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
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
    /// Total launch conditions declared across all launches.
    conditions: usize,
    /// Whether a call parks (`CallSchema::parks`), so a park node precedes
    /// it.
    parks: bool,
}

impl SealedNodeShape {
    pub fn of(schema: &CallSchema, implementation: &NativeImplementation) -> Self {
        let arguments = super::abi::buffer_slots(schema, implementation) - 2;
        let words = super::abi::native_word_count(schema, implementation);
        let conditions = implementation.launches.iter().map(|l| l.conditions.len()).sum();
        Self {
            buffers: sealed_buffers(schema, implementation),
            arguments,
            words,
            launches: implementation.launches.len(),
            conditions,
            parks: schema.parks(),
        }
    }

    fn launch_census(&self, backend: BackendName, active_launches: u64) -> GraphCensus {
        GraphCensus::launch(argument_bytes(backend, self.arguments as u64, 8 * self.words as u64))
            .times(active_launches)
            .expect("a node's launch census fits u64")
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

/// Construct a [`NativePort`] handle for graph `workflow` and port ordinal `result`.
pub fn make_native_port(workflow: u64, result: u32) -> NativePort {
    NativePort {
        reference: WorkflowResultRef {
            workflow,
            node: INPUT_NODE,
            result,
        },
    }
}

/// Construct a [`NativeTimelinePort`] handle for graph `workflow` and timeline ordinal `ordinal`.
pub fn make_timeline_port<R: TimelineRole>(workflow: u64, ordinal: u32) -> NativeTimelinePort<R> {
    NativeTimelinePort {
        workflow,
        ordinal,
        role: PhantomData,
    }
}

/// Add a timeline node of kind `N` after the `nodes` enqueued so far.
fn add_timeline<N: TimelineNodeKind>(
    identity: u64,
    timelines: &mut Vec<(usize, TimelineNode)>,
    nodes: usize,
) -> NativeTimelinePort<N::Role> {
    let ordinal = u32::try_from(timelines.len()).expect("timeline ordinal exhausted");
    timelines.push((nodes, N::NODE));
    NativeTimelinePort {
        workflow: identity,
        ordinal,
        role: PhantomData,
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
fn validate_stores<R: AsRef<[Option<NativeTensorSpec>]>>(
    stores: &[Store],
    ports: &[PortSpec],
    results: &[R],
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

/// The forms of an operand a carry holds ([`OperandCarry`]): its formed
/// F32 rows always, under the NVFP4 operand transform where `nvfp4`, and its
/// Q8 blocks where `quantized`. They are the forms its readers' kernels
/// take, so only a graph of kernels names them
/// (`NativeGraphDraft::carry_out`, `carry_in`): its seal writes them into
/// its producers' emission words and holds its readers to them. A carry's
/// ports, and every byte a graph reserves for it, are the same in each
/// ([`operand_carry_ports`]), so a graph built to be sized names none.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OperandForms {
    pub quantized: bool,
    pub nvfp4: bool,
}

impl OperandForms {
    /// The emission word (`abi::emitted_word`) of a producer's buffer of
    /// these forms: its rows (`rows`) or its Q8 blocks.
    fn word(self, rows: bool) -> u64 {
        emission_word(rows, rows || self.quantized, self.nvfp4)
    }

    /// Whether a reader's buffer of a carry in these forms, its rows
    /// (`rows`) or its Q8 blocks, standing as `at` at its call, reads what
    /// the carry holds: its rows formed under the same transform, by a
    /// reader that also reads the carry's Q8 blocks where it holds them
    /// (`blocks`: whether the reader reads any); its Q8 blocks formed
    /// exactly where the carry holds them.
    fn read_by(self, rows: bool, at: FormedAt, blocks: bool) -> bool {
        match rows {
            true => at.formed && at.nvfp4 == self.nvfp4 && (blocks || !self.quantized),
            false => at.formed == self.quantized,
        }
    }
}

/// The emission word of a producer's operand buffer, its rows (`rows`) or
/// its Q8 blocks, which a consumer reads: 0 where the consumer does not
/// form it (`formed`); Q8 blocks 1; rows bit 0 to emit, bit 1 where they
/// take the NVFP4 operand transform (`nvfp4`).
fn emission_word(rows: bool, formed: bool, nvfp4: bool) -> u64 {
    match (formed, rows) {
        (false, _) => 0,
        (true, false) => 1,
        (true, true) => 1 | (u64::from(nvfp4) << 1),
    }
}

/// An operand carried across graphs: one graph's producer of a tensor
/// writes the tensor's normed operand into the tensors a run binds to these
/// ports (`carry_out`), and another graph's consumers of that tensor read it
/// from them in place of forming it (`carry_in`). Each graph declares its
/// own carry's ports; a run binds a producer's tensors to the next graph's
/// consumer ports ([`operand_carry_ports`] describes them).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OperandCarry {
    /// F32 `[ROWS, WIDTH]`: the formed rows.
    pub rows: NativePort,
    /// U32 words `[q_rows, ceil(WIDTH / Q8_VALUES) * OPERAND_CARRY_BLOCK_WORDS]`:
    /// the rows' Q8 blocks, where `q_rows` is `ROWS` rounded up to the
    /// backend's Q8 row tile ([`operand_carry_row_tile`]).
    pub quantized: NativePort,
    /// The activation element both sides bind
    /// ([`super::OPERAND_ACTIVATION`]), which the rows are rounded to.
    pub activation: RepresentationId,
}

/// The u32 words of one Q8 block (`cpu::CPU_Q8_BYTES`) in a carry's
/// `quantized` port: its blocks are opaque words, as a call's scratch is
/// bytes.
const OPERAND_CARRY_BLOCK_WORDS: u64 = seismic_lang::cpu::CPU_Q8_BYTES / 4;
const _: () = assert!(seismic_lang::cpu::CPU_Q8_BYTES % 4 == 0, "a Q8 block is whole u32 words");

/// The row tile alignment of a backend's carried Q8 operand buffer: CPU
/// tiled projections size their Q8 scratch in whole `CPU_Q8_ROWS` strips,
/// while GPU backends do not tile CPU Q8 blocks.
pub fn operand_carry_row_tile(backend: BackendName) -> u64 {
    match backend {
        BackendName::Cpu => seismic_lang::cpu::CPU_Q8_ROWS,
        BackendName::Metal | BackendName::Cuda | BackendName::Vulkan => 1,
    }
}

/// An extent of an operand carry's ports: concrete (`u64`) in the runtime,
/// or symbolic (`Dim`) in metadata templates.
pub trait CarryExtent: Clone {
    /// `ceil(self / divisor) * factor`.
    fn ceil_div_mul(&self, divisor: u64, factor: u64) -> Self;
}

impl CarryExtent for u64 {
    fn ceil_div_mul(&self, divisor: u64, factor: u64) -> Self {
        self.div_ceil(divisor) * factor
    }
}

/// The representation and extents of each port of a carry of `rows` rows of
/// `width` values on `backend`: its rows, then its Q8 blocks
/// ([`OperandCarry`]). Generic over concrete (`u64`) and symbolic (`Dim`)
/// extents so runtime graphs, carry allocation, validation, and metadata
/// templates all share one owner.
pub fn operand_carry_ports<D: CarryExtent>(
    backend: BackendName,
    rows: D,
    width: D,
) -> [(RepresentationId, [D; 2]); 2] {
    let representation =
        |name| seismic_lang::registry::representation(name).expect("the registry holds every dense dtype");
    let tile = operand_carry_row_tile(backend);
    let quantized_rows = if tile == 1 { rows.clone() } else { rows.ceil_div_mul(tile, tile) };
    let quantized_cols = width.ceil_div_mul(seismic_lang::cpu::CPU_Q8_VALUES, OPERAND_CARRY_BLOCK_WORDS);
    [
        (representation("f32"), [rows, width]),
        (representation("u32"), [quantized_rows, quantized_cols]),
    ]
}

/// A carry a graph declares ([`OperandCarry`]), by port ordinal, in the
/// forms `F` its graph names: a graph of kernels its [`OperandForms`], a
/// graph built to be sized none. What a graph reserves for a carry
/// (`operand_links`, `plan_storage`) is decided over any `F`, so by no form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Carry<F = ()> {
    rows: u32,
    quantized: u32,
    forms: F,
    activation: RepresentationId,
    side: OperandCarrySide,
}

/// One end of an operand carry ([`OperandCarrySpec`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperandCarrySide {
    /// The node producing `result` emits its normed operand into the ports.
    Out { result: WorkflowResultRef },
    /// Each consumer of port `hidden`'s whole tensor whose operand is
    /// normed by port `norm`'s under the epsilon of bits `epsilon` reads
    /// the operand from the ports.
    In { hidden: u32, norm: u32, epsilon: u32 },
}

/// One end of a carry as a graph declares it (`carry_out`, `carry_in`).
enum CarryEnd {
    Out(WorkflowResultRef),
    In { hidden: NativePort, norm: NativePort, epsilon: f32 },
}

/// Record that graph `identity`, of `ports` and the `nodes` enqueued so
/// far, carries `carry` in `forms` at `end`. Its ports are two external
/// ports of the graph, of the shapes a carry takes, that no other carry
/// names; a carried out result is one of its nodes', carried out once; a
/// carried in tensor and norm are its ports.
fn add_carry<F>(
    backend: BackendName,
    identity: u64,
    ports: &[PortSpec],
    nodes: usize,
    carries: &mut Vec<Carry<F>>,
    carry: &OperandCarry,
    forms: F,
    end: CarryEnd,
) -> Result<(), WorkflowError> {
    let own = |port: NativePort| {
        port.reference.workflow == identity
            && port.reference.node == INPUT_NODE
            && (port.reference.result as usize) < ports.len()
    };
    let mismatch = |port: NativePort| WorkflowError::NativePortMismatch { port: port.reference.result as usize };
    if !own(carry.rows) || !own(carry.quantized) {
        return Err(WorkflowError::CrossWorkflowResult);
    }
    let side = match end {
        CarryEnd::Out(result) => {
            if result.workflow != identity || result.node == INPUT_NODE || result.node as usize >= nodes {
                return Err(WorkflowError::CrossWorkflowResult);
            }
            if carries.iter().any(|other| other.side == OperandCarrySide::Out { result }) {
                return Err(WorkflowError::OperandEmitter { node: result.node as usize });
            }
            OperandCarrySide::Out { result }
        }
        CarryEnd::In { hidden, norm, epsilon } => {
            if !own(hidden) || !own(norm) {
                return Err(WorkflowError::CrossWorkflowResult);
            }
            if [hidden, norm].iter().any(|port| *port == carry.rows || *port == carry.quantized) {
                return Err(mismatch(hidden));
            }
            OperandCarrySide::In { hidden: hidden.reference.result, norm: norm.reference.result, epsilon: epsilon.to_bits() }
        }
    };
    let rows = &ports[carry.rows.reference.result as usize];
    let [rows_port, quantized_port] = match rows.extents[..] {
        [count, width] => operand_carry_ports(backend, count, width),
        _ => return Err(mismatch(carry.rows)),
    };
    for (port, (representation, extents)) in [(carry.rows, rows_port), (carry.quantized, quantized_port)] {
        let spec = &ports[port.reference.result as usize];
        let named = carries
            .iter()
            .any(|other| [other.rows, other.quantized].contains(&port.reference.result));
        if spec.local || spec.representation != representation || spec.extents != extents || named {
            return Err(mismatch(port));
        }
    }
    if carry.rows == carry.quantized {
        return Err(mismatch(carry.quantized));
    }
    carries.push(Carry {
        rows: carry.rows.reference.result,
        quantized: carry.quantized.reference.result,
        forms,
        activation: carry.activation,
        side,
    });
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
    /// Operands carried out to and in from other graphs, in the forms
    /// its seal writes and checks.
    carries: Vec<Carry<OperandForms>>,
    layout: Option<Arc<NativeGraphLayout>>,
    /// Timeline nodes, each before the kernel node of its position.
    timelines: Vec<(usize, TimelineNode)>,
    /// Span marks (`span_mark`), each before the kernel node of its
    /// position.
    marks: Vec<usize>,
}

/// A timeline node of a graph (`timeline`), which each run binds to a
/// timeline of role `R` (`NativeGraphBindings::set_timeline`, `set_park`).
pub struct NativeTimelinePort<R: TimelineRole> {
    workflow: u64,
    ordinal: u32,
    role: PhantomData<fn() -> R>,
}

impl<R: TimelineRole> Clone for NativeTimelinePort<R> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<R: TimelineRole> Copy for NativeTimelinePort<R> {}

impl<R: TimelineRole> std::fmt::Debug for NativeTimelinePort<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeTimelinePort")
            .field("workflow", &self.workflow)
            .field("ordinal", &self.ordinal)
            .finish()
    }
}

impl<R: TimelineRole> PartialEq for NativeTimelinePort<R> {
    fn eq(&self, other: &Self) -> bool {
        (self.workflow, self.ordinal) == (other.workflow, other.ordinal)
    }
}

impl<R: TimelineRole> Eq for NativeTimelinePort<R> {}

impl NativeGraphDraft {
    pub fn new(device: &Arc<DeviceInner>) -> Self {
        Self {
            identity: NEXT_NATIVE_GRAPH.fetch_add(1, Ordering::Relaxed),
            device: device.clone(),
            ports: Vec::new(),
            nodes: Vec::new(),
            exports: Vec::new(),
            stores: Vec::new(),
            carries: Vec::new(),
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

    /// A timeline node of kind `N` after every node enqueued so far: a host
    /// gate the device waits at (`Wait`), a signal the host waits for
    /// (`Signal`), a park or a publication. Its port binds only a timeline
    /// of the node's role.
    pub fn timeline<N: TimelineNodeKind>(&mut self, _node: N) -> NativeTimelinePort<N::Role> {
        add_timeline::<N>(self.identity, &mut self.timelines, self.nodes.len())
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

    /// The backend of the device this graph drafts for.
    pub fn backend(&self) -> BackendName {
        super::backend_name(&self.device.kind)
    }

    /// Declare a carry of `rows` rows of `width` values, its rows rounded
    /// to `activation`: two external ports of the shapes
    /// [`operand_carry_ports`] gives, which each run binds.
    pub fn operand_carry(
        &mut self,
        rows: u64,
        width: u64,
        activation: RepresentationId,
    ) -> Result<OperandCarry, TensorError> {
        let [(rows_element, rows_extents), (quantized_element, quantized_extents)] =
            operand_carry_ports(self.backend(), rows, width);
        Ok(OperandCarry {
            rows: self.port(rows_element, &rows_extents)?,
            quantized: self.port(quantized_element, &quantized_extents)?,
            activation,
        })
    }

    /// The node producing `result` emits its normed operand into `carry`'s
    /// ports in `forms`, those the carry's readers in the next graph take
    /// (sealing refuses a node that emits none, `OperandEmitter`).
    pub fn carry_out(
        &mut self,
        result: WorkflowResultRef,
        carry: &OperandCarry,
        forms: OperandForms,
    ) -> Result<(), WorkflowError> {
        add_carry(self.backend(), self.identity, &self.ports, self.nodes.len(), &mut self.carries, carry, forms, CarryEnd::Out(result))
    }

    /// Each consumer of `hidden`'s whole tensor whose operand is normed by
    /// `norm`'s tensor under `epsilon` reads it from `carry`'s ports, which
    /// hold it in `forms`, its formation left out (sealing refuses a carry
    /// no consumer reads, `UnreadOperandCarry`, and one whose forms differ
    /// from a reader's kernel's, `OperandForms`).
    pub fn carry_in(
        &mut self,
        hidden: NativePort,
        carry: &OperandCarry,
        forms: OperandForms,
        norm: NativePort,
        epsilon: f32,
    ) -> Result<(), WorkflowError> {
        let end = CarryEnd::In { hidden, norm, epsilon };
        add_carry(self.backend(), self.identity, &self.ports, self.nodes.len(), &mut self.carries, carry, forms, end)
    }

    /// Seal the graph as a plan whose runs replay graphs: on CUDA graphs
    /// recorded as a run first needs one, once the device's queued work has
    /// drained.
    pub fn seal(self) -> Result<NativeGraphPlan, CallError> {
        self.sealed(false)
    }

    /// Seal the graph as a launched plan: per-use work whose shapes no load
    /// can enumerate ahead (rows a request brings), so no graph of it is
    /// kept. On CUDA, where recording a graph drains the device, its runs
    /// are launched call by call in their submission's stream order, and a
    /// slot of it holds no graph, so it runs with work in flight unhindered;
    /// elsewhere its graphs are recorded at a shape's first use, as every
    /// plan's are.
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
            self.nodes.iter().map(|node| (&node.args, node.condition.map(|condition| condition.port.reference.result))),
            &self.exports,
            &self.stores,
            &self.carries,
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
                args: Arc::new(node.args),
                scratch: shape.scratch.clone(),
                activation: node.kernel.inner.activation(),
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
        let links = operand_links(&planned, &self.carries).map_err(CallError::Workflow)?;
        charge_node_emissions(&mut planned, &links);
        // A consumer whose operands its producers emit (in the graph or
        // through a carry), where each stands for the one it would form,
        // leaves out the launch that forms them, and each of those producers
        // emits: its word for the operand (`abi::emitted_word`) is set from
        // the forms its consumer reads. Every other call emits nothing.
        let mut emitting: Vec<(usize, usize, u64)> = Vec::new();
        for (node, (_, _, shape)) in shaped.iter_mut().enumerate() {
            let scratch = &planned[node].scratch;
            let left_out = formations_left_out(node, scratch, &shape.forms, &links);
            for carried in links.carried.iter().filter(|carried| carried.node == node) {
                let carry = &self.carries[carried.carry];
                match (scratch[carried.scratch].operand, shape.forms[carried.scratch]) {
                    // A carried consumer reads the carry alone: its forms are
                    // the carry's and the launch forming them is left out.
                    (Some(ScratchOperand::Formed { formation, rows, .. }), Some(at)) => {
                        let blocks = links.carried.iter().any(|other| {
                            (other.node, other.carry, other.port) == (node, carried.carry, carry.quantized)
                        });
                        if !carry.forms.read_by(rows, at, blocks) || !left_out.contains(&(formation as usize)) {
                            return Err(CallError::Workflow(WorkflowError::OperandForms { node }));
                        }
                    }
                    (Some(ScratchOperand::Emitted { rows, .. }), None) => {
                        emitting.push((node, carried.scratch, carry.forms.word(rows)));
                    }
                    _ => unreachable!("a carried buffer is an operand buffer"),
                }
            }
            for pair in links.pairs.iter().filter(|pair| pair.consumer == node) {
                let (Some(ScratchOperand::Formed { formation, rows, .. }), Some(at)) =
                    (scratch[pair.formed].operand, shape.forms[pair.formed])
                else {
                    unreachable!("a paired consumer's buffer is a formed operand");
                };
                if left_out.contains(&(formation as usize)) {
                    emitting.push((pair.producer, pair.emitted, emission_word(rows, at.formed, at.nvfp4)));
                }
            }
            for &launch in &left_out {
                shape.launches.geometry[launch] = None;
            }
        }
        for (producer, emitted, word) in emitting {
            let (kernel, _, shape) = &mut shaped[producer];
            shape.words[super::abi::emitted_word(kernel.schema(), kernel.implementation(), emitted)] = word;
        }
        let backend = super::backend_name(&self.device.kind);
        let launches = shaped
            .iter()
            .map(|(kernel, _, shape)| {
                let active = shape.launches.geometry.iter().filter(|g| g.is_some()).count() as u64;
                SealedNodeShape::of(kernel.schema(), kernel.implementation())
                    .launch_census(backend, active)
            })
            .try_fold(GraphCensus::default(), GraphCensus::plus)
            .expect("a graph's launch census fits u64");
        let recorded = recorded_census(backend, shaped.len(), launches, &self.timelines, &self.marks);
        let storage = if let Some(layout) = &self.layout {
            layout
                .storage_for(&self.ports, &planned, &links, &results, &self.exports, &self.stores)
                .map_err(CallError::Workflow)?
        } else {
            plan_storage(&self.ports, &planned, &links, &results, &self.exports, &self.stores)
        };
        let mut executable = Executable {
            nodes: Vec::with_capacity(planned.len()),
            external_writes: vec![false; self.ports.len()],
            port_uses: vec![PortUse::default(); self.ports.len()],
            upload_written: false,
            exclusive: None,
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
            executable.exclude(node, planned)?;
        }
        executable.refuse_plain_overlaps()?;
        let shape = PlanShape::of(identity, self.layout.as_deref(), &executable);
        let (candidate, flat) = SealedTopology::pack(
            self.ports,
            executable,
            results,
            self.exports,
            storage.port_placements,
            storage.placements,
        );
        let topology = if let Some(layout) = &self.layout {
            let mut slot = layout.topology.lock().expect("layout topology lock");
            match &*slot {
                Some(cached) => {
                    assert!(
                        **cached == candidate,
                        "invariant violated: plan sealed into layout {} produced a different static topology",
                        layout.identity
                    );
                    Arc::clone(cached)
                }
                None => {
                    let shared = Arc::new(candidate);
                    *slot = Some(Arc::clone(&shared));
                    shared
                }
            }
        } else {
            Arc::new(candidate)
        };
        self.device.native.register_plan_key(backend, shape.keyed(backend));
        Ok(NativeGraphPlan {
            identity,
            shape,
            device: self.device,
            topology,
            kernels: flat.kernels,
            rounds: flat.rounds,
            site_bytes: flat.site_bytes,
            site_offsets: flat.site_offsets,
            site_records: flat.site_records,
            port_uses: flat.port_uses,
            port_byte_lens: flat.port_byte_lens,
            result_byte_lens: flat.result_byte_lens,
            extents_and_strides: flat.extents_and_strides,
            words: flat.words,
            word_ends: flat.word_ends,
            launches: flat.launches,
            launch_ends: flat.launch_ends,
            conditions: flat.conditions,
            condition_ends: flat.condition_ends,
            workspace_records: flat.workspace_records,
            scratch_bytes: storage.scratch_bytes,
            scratch_floor_bytes: storage.scratch_floor_bytes,
            sync_bytes: storage.sync_bytes,
            output_bytes: storage.output_bytes,
            upload_bytes: storage.upload_bytes,
            recorded,
            launched,
        })
    }
}

/// Per-node metadata recorded in [`NativeGraphMetadataDraft`].
struct MetadataNode {
    planned: PlannedNode,
    /// The name of this node's checked entry, which a mismatch names.
    entry: &'static str,
    /// The lengths of this node's sealed form's vectors.
    shape: SealedNodeShape,
    results: Vec<Option<NativeTensorSpec>>,
    /// The active-launch bitmask at this node's enqueued dimensions
    /// ([`recorded_census`]).
    active_launches: u64,
}

impl MetadataNode {
    fn plan_heap_node(&self) -> PlanHeapNode {
        PlanHeapNode {
            buffers: self.shape.buffers,
            arguments: self.shape.arguments,
            words: self.shape.words,
            launches: self.shape.launches,
            conditions: self.shape.conditions
                + if self.planned.condition.is_some() { self.shape.launches } else { 0 },
        }
    }
}

impl AsRef<PlannedNode> for MetadataNode {
    fn as_ref(&self) -> &PlannedNode {
        &self.planned
    }
}

impl AsMut<PlannedNode> for MetadataNode {
    fn as_mut(&mut self) -> &mut PlannedNode {
        &mut self.planned
    }
}

impl AsRef<[Option<NativeTensorSpec>]> for MetadataNode {
    fn as_ref(&self) -> &[Option<NativeTensorSpec>] {
        &self.results
    }
}

/// Storage-only graph construction from checked entry metadata. It uses the
/// same port identities and interval allocator as a prepared native graph,
/// but owns no device, formed kernel, or executable node.
pub struct NativeGraphMetadataDraft {
    /// The backend whose native declarations it checks, which records it.
    backend: BackendName,
    identity: u64,
    ports: Vec<PortSpec>,
    nodes: Vec<MetadataNode>,
    exports: Vec<WorkflowResultRef>,
    stores: Vec<Store>,
    /// Operands carried out to and in from other graphs
    /// (`NativeGraphDraft::carry_out`, `carry_in`).
    carries: Vec<Carry>,
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
    /// The host heap of the shared static [`SealedTopology`] held once per
    /// layout across every plan sealed into it.
    topology_heap: HostHeap,
    /// The topology's timeline nodes.
    timelines: usize,
    /// What recording one run of a plan sealed into it takes.
    recorded: GraphCensus,
    /// Shared static topology for all plans sealed into this layout.
    topology: Arc<std::sync::Mutex<Option<Arc<SealedTopology>>>>,
}

impl NativeGraphLayout {
    /// Unique identifier of this layout, shared by clones of it and by every
    /// plan sealed into it.
    pub fn identity(&self) -> u64 {
        self.identity
    }

    /// The host heap of each plan sealed into this layout: every flat slice of
    /// a sealed plan has a length the layout's topology fixes, so its classes'
    /// plans hold the same heap. `NativeGraphPlan::host_heap` is what a sealed
    /// plan holds.
    pub fn plan_heap(&self) -> HostHeap {
        self.plan_heap
    }

    /// The host heap of the shared static topology held once across all plans
    /// sealed into this layout (`NativeGraphPlan::topology_heap`).
    pub fn topology_heap(&self) -> HostHeap {
        self.topology_heap
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
            sync: self.storage.sync_bytes,
            output: self.storage.output_bytes,
            upload: self.storage.upload_bytes,
            recorded: self.recorded,
        }
    }

    /// The storage of an exact graph sealed into this layout: refused
    /// unless its topology (its edges and carries among them, so each
    /// carried buffer lies on the same port) is the layout's and every
    /// buffer fits its capacity.
    fn storage_for<N: AsRef<PlannedNode>, R: AsRef<[Option<NativeTensorSpec>]>>(
        &self,
        ports: &[PortSpec],
        nodes: &[N],
        links: &OperandLinks,
        results: &[R],
        exports: &[WorkflowResultRef],
        stores: &[Store],
    ) -> Result<StoragePlan, WorkflowError> {
        let topology = || WorkflowError::NativeGraphLayoutMismatch(LayoutMismatch::Topology);
        let exceeds = |buffer, bytes, capacity| {
            WorkflowError::NativeGraphLayoutMismatch(LayoutMismatch::Exceeds {
                buffer,
                bytes,
                capacity,
            })
        };
        if storage_signature(ports, nodes, links, results, exports, stores) != self.signature
            || ports.len() != self.port_capacity.len()
            || results.len() != self.result_capacity.len()
            || nodes.len() != self.scratch_capacity.len()
        {
            return Err(topology());
        }
        for (port, (spec, &capacity)) in ports.iter().zip(&self.port_capacity).enumerate() {
            if spec.byte_len > capacity {
                let buffer = GraphBuffer::Port {
                    port,
                    extents: spec.extents.clone(),
                };
                return Err(exceeds(buffer, spec.byte_len, capacity));
            }
        }
        for (launch, (row, capacities)) in results.iter().zip(&self.result_capacity).enumerate() {
            let row = row.as_ref();
            if row.len() != capacities.len() {
                return Err(topology());
            }
            for (result, (spec, capacity)) in row.iter().zip(capacities).enumerate() {
                match (spec, *capacity) {
                    (Some(spec), Some(capacity)) if spec.byte_len > capacity => {
                        let buffer = GraphBuffer::Result {
                            launch,
                            result,
                            extents: spec.extents.clone(),
                        };
                        return Err(exceeds(buffer, spec.byte_len, capacity));
                    }
                    (Some(_), Some(_)) | (None, None) => {}
                    (Some(_), None) | (None, Some(_)) => return Err(topology()),
                }
            }
        }
        for (launch, (node, capacities)) in nodes.iter().zip(&self.scratch_capacity).enumerate() {
            let node = node.as_ref();
            if node.scratch.len() != capacities.len() {
                return Err(topology());
            }
            for (scratch, (need, capacity)) in node.scratch.iter().zip(capacities).enumerate() {
                if need.sync != capacity.sync || need.operand != capacity.operand || need.emission != capacity.emission {
                    return Err(topology());
                }
                if need.bytes > capacity.bytes {
                    let buffer = GraphBuffer::Scratch { launch, scratch };
                    return Err(exceeds(buffer, need.bytes, capacity.bytes));
                }
            }
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
    /// Each carried buffer's (node, buffer, port).
    carried: Vec<(usize, usize, u32)>,
}

fn storage_signature<N: AsRef<PlannedNode>, R: AsRef<[Option<NativeTensorSpec>]>>(
    ports: &[PortSpec],
    nodes: &[N],
    links: &OperandLinks,
    results: &[R],
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
                row.as_ref()
                    .iter()
                    .map(|result| result.as_ref().map(|result| result.representation))
                    .collect()
            })
            .collect(),
        edges: nodes
            .iter()
            .map(|node| {
                node.as_ref()
                    .args
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
            .map(|node| {
                let node = node.as_ref();
                (node.condition, node.share.map(|(ordinal, primary)| (ordinal, primary.node, primary.result)))
            })
            .collect(),
        carried: links.carried.iter().map(|carried| (carried.node, carried.scratch, carried.port)).collect(),
    }
}

/// What a graph holds while it runs and once it is recorded: its storage
/// (workspace, sync region, output arena, upload region) and what recording
/// one run of it may take ([`GraphCensus`]). The sync region is each
/// execution slot's own; the workspace may be shared by slots whose runs
/// never overlap.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NativeGraphStorageBytes {
    pub workspace: u64,
    pub sync: u64,
    pub output: u64,
    pub upload: u64,
    pub recorded: GraphCensus,
}

impl NativeGraphStorageBytes {
    /// What one run of either graph holds at most: each term's larger.
    pub fn most(self, other: Self) -> Self {
        Self {
            workspace: self.workspace.max(other.workspace),
            sync: self.sync.max(other.sync),
            output: self.output.max(other.output),
            upload: self.upload.max(other.upload),
            recorded: self.recorded.most(other.recorded),
        }
    }

    /// Whether `self` (an exact class sealed into a certified layout) fits
    /// within `certified`: storage capacities (`workspace`, `sync`, `output`,
    /// `upload`) must match the layout's exactly, while `recorded` may be at
    /// most the layout's because a layout certified across multiple row
    /// classes holds the group's maximum active launch census.
    pub fn within(self, certified: Self) -> bool {
        self.workspace == certified.workspace
            && self.sync == certified.sync
            && self.output == certified.output
            && self.upload == certified.upload
            && self.recorded.graphs <= certified.recorded.graphs
            && self.recorded.launches <= certified.recorded.launches
            && self.recorded.arguments <= certified.recorded.arguments
            && self.recorded.largest <= certified.recorded.largest
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

/// What recording one run of a graph of `shapes`' nodes and the timeline
/// nodes `timelines` takes at most on `backend`: every launch its nodes
/// declare, in a graph of the run's own (a run's calls are a graph of their
/// shape, `ReplayKeys`), and for each timeline node a graph more, where it
/// ends the graph the calls before it are recorded in whether or not a run
/// binds it (`graph_segments`), or on CUDA the nodes of it a graph holds
/// instead (`cuda::timeline_nodes`). Every run of the plan, on any slot or
/// place, replays these graphs.
/// What recording one run of a graph of `(arguments, words, launches)` nodes
/// and `timelines` takes at most on `backend` (exact match with
/// [`SealedNodeShape::of`] + [`recorded_census`]).
pub fn recorded_census_of(
    backend: BackendName,
    nodes: &[(usize, usize, usize)],
    timelines: &[TimelineNode],
) -> GraphCensus {
    let launches = nodes
        .iter()
        .map(|&(arguments, words, launches)| {
            let launch_arguments = argument_bytes(backend, arguments as u64, 8 * words as u64);
            GraphCensus::launch(launch_arguments)
                .times(launches as u64)
                .expect("a node's launch census fits u64")
        })
        .try_fold(GraphCensus::default(), GraphCensus::plus)
        .expect("a graph's launch census fits u64");
    let (graphs, held) = match backend {
        BackendName::Cuda => (0, timelines.iter().map(|&node| super::cuda::timeline_nodes(node)).sum()),
        BackendName::Vulkan | BackendName::Metal | BackendName::Cpu => (timelines.len() as u64, 0),
    };
    GraphCensus { graphs: 1 + graphs, launches: launches.launches + held, ..launches }
}

/// Per-node sealed vector lengths needed to compute a plan's [`HostHeap`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanHeapNode {
    pub buffers: usize,
    pub arguments: usize,
    pub words: usize,
    pub launches: usize,
    pub conditions: usize,
}

/// Element counts of every heap-owning field of [`NativeGraphPlan`].
/// Both [`plan_heap_of`] and [`NativeGraphPlan::host_heap`] derive their
/// [`HostHeap`] from this single owner so no field can be missed or diverge.
struct PlanHeapCounts {
    nodes: usize,
    ports: usize,
    total_sites: usize,
    total_results: usize,
    extents_and_strides: usize,
    total_words: usize,
    total_launches: usize,
    total_conditions: usize,
}

fn plan_heap_from_counts(c: PlanHeapCounts) -> HostHeap {
    use std::mem::size_of;
    let mut heap = HostHeap::default();
    heap.hold(1, SHARED_PLAN_BYTES);
    heap.hold(c.nodes, size_of::<Arc<NativePrepared>>());
    heap.hold(c.nodes, size_of::<u64>());
    heap.hold(c.total_sites, size_of::<u64>());
    heap.hold(c.total_sites, size_of::<u64>());
    heap.hold(c.total_sites, size_of::<SiteRecords>());
    heap.hold(c.ports, size_of::<PortUse>());
    heap.hold(c.ports, size_of::<u64>());
    heap.hold(c.total_results, size_of::<u64>());
    heap.hold(c.extents_and_strides, size_of::<u64>());
    heap.hold(c.total_words, size_of::<u64>());
    heap.hold(c.nodes, size_of::<u32>());
    heap.hold(c.total_launches, size_of::<Option<LaunchGeometry>>());
    heap.hold(c.nodes, size_of::<u32>());
    heap.hold(c.total_conditions, size_of::<LaunchCondition>());
    heap.hold(c.total_launches, size_of::<u32>());
    heap
}

/// The per-plan host heap of a plan sealed from a topology of these lengths
/// (`NativeGraphPlan::host_heap`, single owner shared by
/// `NativeGraphMetadataDraft::plan_heap` and `SizingDraft`).
pub fn plan_heap_of(
    port_ranks: &[usize],
    nodes: &[PlanHeapNode],
    result_ranks: &[&[Option<usize>]],
) -> HostHeap {
    let port_total_ranks: usize = port_ranks.iter().sum();
    let result_total_ranks: usize = result_ranks
        .iter()
        .flat_map(|row| row.iter().flatten())
        .sum();
    plan_heap_from_counts(PlanHeapCounts {
        nodes: nodes.len(),
        ports: port_ranks.len(),
        total_sites: nodes.iter().map(|s| s.buffers).sum(),
        total_results: result_ranks.iter().map(|row| row.len()).sum(),
        extents_and_strides: 2 * (port_total_ranks + result_total_ranks),
        total_words: nodes.iter().map(|s| s.words).sum(),
        total_launches: nodes.iter().map(|s| s.launches).sum(),
        total_conditions: nodes.iter().map(|s| s.conditions).sum(),
    })
}

/// Element counts of every heap-owning field of [`SealedTopology`].
/// Both [`topology_heap_of`] and [`SealedTopology::host_heap`] derive their
/// [`HostHeap`] from this single owner so no field can be missed or diverge.
struct TopologyHeapCounts {
    ports: usize,
    nodes: usize,
    result_rows: usize,
    total_results: usize,
    exports: usize,
    timelines: usize,
    marks: usize,
}

fn topology_heap_from_counts(
    c: TopologyHeapCounts,
    node_sites_and_reps: impl IntoIterator<Item = (usize, usize)>,
) -> HostHeap {
    use std::mem::size_of;
    let mut heap = HostHeap::default();
    heap.hold(1, SHARED_TOPOLOGY_BYTES);
    heap.hold(c.ports, size_of::<PortMeta>());
    heap.hold(c.nodes, size_of::<StaticNode>());
    for (sites, reps) in node_sites_and_reps {
        heap.hold(sites, size_of::<StaticSite>());
        heap.hold(reps, size_of::<&'static str>());
    }
    heap.hold(c.result_rows, size_of::<std::ops::Range<usize>>());
    heap.hold(
        c.total_results,
        size_of::<Option<(RepresentationId, std::ops::Range<usize>)>>(),
    );
    heap.hold(c.exports, size_of::<(u32, u32)>());
    heap.hold(c.ports, size_of::<Option<Placement>>());
    heap.hold(c.total_results, size_of::<Placement>());
    heap.hold(c.ports, size_of::<bool>());
    heap.hold(c.timelines, size_of::<(usize, TimelineNode)>());
    heap.hold(c.marks, size_of::<usize>());
    heap
}

/// The host heap of one shared [`SealedTopology`] allocation
/// (`NativeGraphPlan::topology_heap`, held once per [`NativeGraphLayout`]
/// across every plan sealed into it).
pub fn topology_heap_of(
    port_count: usize,
    nodes: &[PlanHeapNode],
    result_rows: usize,
    total_results: usize,
    exports: usize,
    timelines: usize,
    marks: usize,
) -> HostHeap {
    topology_heap_from_counts(
        TopologyHeapCounts {
            ports: port_count,
            nodes: nodes.len(),
            result_rows,
            total_results,
            exports,
            timelines,
            marks,
        },
        nodes.iter().map(|s| (s.buffers, s.arguments)),
    )
}

/// What recording one run of a graph takes on `backend`, given the summed
/// active-launch census `launches` across its `nodes` nodes:
/// - On CUDA: one graph when `nodes > 0`, with a node for each active
///   launch, each span mark inside `(1..nodes)`, and the nodes of each
///   timeline node inside `(1..nodes)` (`cuda::timeline_nodes`), matching
///   [`NativeGraphPlan::run_graphs`] exactly.
/// - On Vulkan, Metal and CPU: one graph plus one per timeline boundary,
///   holding only the active launches and their argument bytes.
fn recorded_census(
    backend: BackendName,
    nodes: usize,
    launches: GraphCensus,
    timelines: &[(usize, TimelineNode)],
    marks: &[usize],
) -> GraphCensus {
    match backend {
        BackendName::Cuda => {
            let held_timelines = timelines
                .iter()
                .filter(|(position, _)| (1..nodes).contains(position))
                .map(|&(_, node)| super::cuda::timeline_nodes(node))
                .sum::<u64>();
            let held_marks = marks
                .iter()
                .filter(|mark| (1..nodes).contains(mark))
                .count() as u64;
            GraphCensus {
                graphs: u64::from(nodes > 0),
                launches: launches.launches + held_marks + held_timelines,
                arguments: 0,
                largest: 0,
            }
        }
        BackendName::Vulkan | BackendName::Metal | BackendName::Cpu => GraphCensus {
            graphs: 1 + timelines.len() as u64,
            ..launches
        },
    }
}

/// What recording one run of a CUDA graph takes: one graph when `nodes > 0`,
/// with a node for each active launch in `node_launches`, each span mark
/// inside `(1..nodes)`, and the nodes of each timeline node inside `(1..nodes)`
/// (`cuda::timeline_nodes`), matching [`NativeGraphPlan::run_graphs`] exactly.
pub fn cuda_recorded_census(
    nodes: usize,
    node_launches: &[u64],
    timelines: &[(usize, TimelineNode)],
    marks: &[usize],
) -> GraphCensus {
    let launches = GraphCensus {
        launches: node_launches.iter().copied().sum::<u64>(),
        ..GraphCensus::default()
    };
    recorded_census(BackendName::Cuda, nodes, launches, timelines, marks)
}

impl NativeGraphMetadataDraft {
    /// An empty graph of `backend`'s native declarations.
    pub fn new(backend: BackendName) -> Self {
        Self {
            backend,
            identity: NEXT_NATIVE_GRAPH.fetch_add(1, Ordering::Relaxed),
            ports: Vec::new(),
            nodes: Vec::new(),
            exports: Vec::new(),
            stores: Vec::new(),
            carries: Vec::new(),
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
    pub fn timeline<N: TimelineNodeKind>(&mut self, _node: N) -> NativeTimelinePort<N::Role> {
        add_timeline::<N>(self.identity, &mut self.timelines, self.nodes.len())
    }

    /// The host heap of the plan this topology seals to
    /// (`NativeGraphPlan::host_heap`), from the lengths of its vectors: the
    /// plan in its shared allocation, its ports, each node's buffers, words
    /// and launches, the result specs, the exports and the placements.
    pub fn plan_heap(&self) -> HostHeap {
        let port_ranks: Vec<usize> = self
            .ports
            .iter()
            .map(|port| {
                debug_assert_eq!(port.extents.len(), port.strides.len());
                port.extents.len()
            })
            .collect();
        let nodes: Vec<PlanHeapNode> = self.nodes.iter().map(MetadataNode::plan_heap_node).collect();
        let result_rows: Vec<Vec<Option<usize>>> = self
            .nodes
            .iter()
            .map(|node| {
                node.results
                    .iter()
                    .map(|spec| {
                        spec.as_ref().map(|s| {
                            debug_assert_eq!(s.extents.len(), s.strides.len());
                            s.extents.len()
                        })
                    })
                    .collect()
            })
            .collect();
        let result_slices: Vec<&[Option<usize>]> = result_rows.iter().map(Vec::as_slice).collect();
        plan_heap_of(&port_ranks, &nodes, &result_slices)
    }

    /// The host heap of one shared [`SealedTopology`] allocation for this
    /// topology (`NativeGraphPlan::topology_heap`).
    pub fn topology_heap(&self) -> HostHeap {
        let nodes: Vec<PlanHeapNode> = self.nodes.iter().map(MetadataNode::plan_heap_node).collect();
        let total_results: usize = self.nodes.iter().map(|node| node.results.len()).sum();
        topology_heap_of(
            self.ports.len(),
            &nodes,
            self.nodes.len(),
            total_results,
            self.exports.len(),
            self.timelines.len(),
            self.marks.len(),
        )
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

    /// Enqueue node `entry` (its checked entry's name) over `args`, its
    /// parameters and results of these shapes, with `scratch`, its entry's
    /// activation element bound to `activation` ([`super::OPERAND_ACTIVATION`])
    /// where it has one.
    #[allow(clippy::too_many_arguments)]
    pub fn enqueue(
        &mut self,
        entry: &'static str,
        args: EncodedWorkflowArgs,
        parameter_shapes: Vec<Option<(RepresentationId, Vec<u64>)>>,
        result_shapes: Vec<Option<(RepresentationId, Vec<u64>)>>,
        scratch: Vec<ScratchNeed>,
        activation: Option<RepresentationId>,
        shape: SealedNodeShape,
        active_launches: Option<u64>,
    ) -> Result<PendingWorkflowResults, CallError> {
        let node = u32::try_from(self.nodes.len()).expect("native graph node ordinal exhausted");
        validate_node_references(self.identity, self.ports.len(), node, &args)
            .map_err(CallError::Workflow)?;
        validate_metadata_arguments(entry, &args, &parameter_shapes, &self.ports, &self.nodes, None)?;
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
        let default_mask = if shape.launches >= 64 { u64::MAX } else { (1u64 << shape.launches) - 1 };
        self.nodes.push(MetadataNode {
            planned: PlannedNode {
                args: Arc::new(args),
                scratch,
                activation,
                condition: None,
                share: None,
            },
            entry,
            shape,
            results,
            active_launches: active_launches.unwrap_or(default_mask),
        });
        Ok(PendingWorkflowResults::new(self.identity, node, count))
    }

    pub fn export(&mut self, reference: WorkflowResultRef) -> Result<(), WorkflowError> {
        add_export(self.identity, &self.ports, self.nodes.len(), &mut self.exports, reference)
    }

    /// [`Self::enqueue`] under `condition` (`NativeGraphDraft::enqueue_under`):
    /// planning holds the node as any other, its condition kept for the
    /// proof that results may share storage (`Self::share`).
    #[allow(clippy::too_many_arguments)]
    pub fn enqueue_under(
        &mut self,
        entry: &'static str,
        args: EncodedWorkflowArgs,
        parameter_shapes: Vec<Option<(RepresentationId, Vec<u64>)>>,
        result_shapes: Vec<Option<(RepresentationId, Vec<u64>)>>,
        scratch: Vec<ScratchNeed>,
        activation: Option<RepresentationId>,
        shape: SealedNodeShape,
        active_launches: Option<u64>,
        condition: NodeCondition,
    ) -> Result<PendingWorkflowResults, CallError> {
        let port = condition_port(self.identity, &self.ports, self.nodes.len(), condition).map_err(CallError::Workflow)?;
        let pending = self.enqueue(entry, args, parameter_shapes, result_shapes, scratch, activation, shape, active_launches)?;
        self.nodes.last_mut().expect("the node just enqueued").planned.condition =
            Some((port, condition.index, condition.encoded));
        Ok(pending)
    }

    /// `NativeGraphDraft::share`.
    pub fn share(&mut self, other: WorkflowResultRef, primary: WorkflowResultRef) -> Result<(), WorkflowError> {
        let node = (other.workflow == self.identity && primary.workflow == self.identity)
            .then(|| self.nodes.get_mut(other.node as usize))
            .flatten()
            .filter(|node| node.planned.share.is_none())
            .ok_or(WorkflowError::SharedResult { node: other.node as usize })?;
        node.planned.share = Some((other.result, primary));
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

    /// `NativeGraphDraft::carry_out`, in no form: `carry`'s ports are two
    /// of this graph's ports, of [`operand_carry_ports`]' shapes, and what
    /// the graph reserves for them is the same in every form
    /// ([`OperandForms`]).
    pub fn carry_out(&mut self, result: WorkflowResultRef, carry: &OperandCarry) -> Result<(), WorkflowError> {
        add_carry(self.backend, self.identity, &self.ports, self.nodes.len(), &mut self.carries, carry, (), CarryEnd::Out(result))
    }

    /// `NativeGraphDraft::carry_in`, in no form.
    pub fn carry_in(
        &mut self,
        hidden: NativePort,
        carry: &OperandCarry,
        norm: NativePort,
        epsilon: f32,
    ) -> Result<(), WorkflowError> {
        let end = CarryEnd::In { hidden, norm, epsilon };
        add_carry(self.backend, self.identity, &self.ports, self.nodes.len(), &mut self.carries, carry, (), end)
    }

    pub fn seal(mut self) -> Result<NativeGraphStorageBytes, WorkflowError> {
        let links = self.validate_topology()?;
        charge_node_emissions(&mut self.nodes, &links);
        let storage = plan_storage(
            &self.ports,
            &self.nodes,
            &links,
            &self.nodes,
            &self.exports,
            &self.stores,
        );
        Ok(NativeGraphStorageBytes {
            workspace: storage.scratch_bytes,
            sync: storage.sync_bytes,
            output: storage.output_bytes,
            upload: storage.upload_bytes,
            recorded: self.recorded(),
        })
    }

    /// What recording one run of this graph takes when each node makes the
    /// launches in `masks` ([`recorded_census`]).
    fn recorded_with(&self, masks: impl Iterator<Item = u64>) -> GraphCensus {
        let launches = self
            .nodes
            .iter()
            .zip(masks)
            .map(|(node, mask)| node.shape.launch_census(self.backend, u64::from(mask.count_ones())))
            .try_fold(GraphCensus::default(), GraphCensus::plus)
            .expect("a graph's launch census fits u64");
        recorded_census(self.backend, self.nodes.len(), launches, &self.timelines, &self.marks)
    }

    /// What recording one run of this graph takes when each node makes the
    /// launches in `masks` ([`recorded_census`]).
    fn recorded_for(&self, masks: &[u64]) -> GraphCensus {
        debug_assert_eq!(masks.len(), self.nodes.len());
        self.recorded_with(masks.iter().copied())
    }

    /// What recording one run of this graph takes ([`recorded_census`]).
    fn recorded(&self) -> GraphCensus {
        self.recorded_with(self.nodes.iter().map(|node| node.active_launches))
    }

    /// Verify an exact checked metadata graph against the storage certificate
    /// used by executable preparation, without placing it again.
    pub fn seal_with_layout(
        mut self,
        layout: &NativeGraphLayout,
    ) -> Result<NativeGraphStorageBytes, WorkflowError> {
        let links = self.validate_topology()?;
        charge_node_emissions(&mut self.nodes, &links);
        let storage = layout.storage_for(
            &self.ports,
            &self.nodes,
            &links,
            &self.nodes,
            &self.exports,
            &self.stores,
        )?;
        Ok(NativeGraphStorageBytes {
            workspace: storage.scratch_bytes,
            sync: storage.sync_bytes,
            output: storage.output_bytes,
            upload: storage.upload_bytes,
            recorded: self.recorded(),
        })
    }

    /// It as a resource template for other exact shape classes, once its
    /// ordered edges and exports are validated.
    pub fn validated(self) -> Result<ValidatedMetadataDraft, WorkflowError> {
        let links = self.validate_topology()?;
        Ok(ValidatedMetadataDraft {
            links,
            plan_heap: self.plan_heap(),
            topology_heap: self.topology_heap(),
            bound_heap: bindings_heap(self.ports.len(), self.timelines.len()),
            census: self.recorded(),
            draft: self,
        })
    }

    /// Validate a graph's ordered edges, exports and carries: its operand
    /// links, where it has them.
    fn validate_topology(&self) -> Result<OperandLinks, WorkflowError> {
        if self.nodes.is_empty() {
            return Err(WorkflowError::Empty);
        }
        validate_used_ports(
            self.ports.len(),
            self.nodes
                .iter()
                .map(|node| (&*node.planned.args, node.planned.condition.map(|(port, _, _)| port))),
            &self.exports,
            &self.stores,
            &self.carries,
        )?;
        validate_park(&self.timelines, self.nodes.len(), false)?;
        for reference in &self.exports {
            if reference.node != INPUT_NODE && result_spec(&self.nodes, *reference).is_none() {
                return Err(WorkflowError::MissingProducerResult);
            }
        }
        validate_stores(&self.stores, &self.ports, &self.nodes, &self.exports)?;
        validate_shares(&self.nodes, &self.nodes, &self.exports, &self.stores)?;
        operand_links(&self.nodes, &self.carries)
    }
}

/// A metadata draft whose topology is validated
/// ([`NativeGraphMetadataDraft::validated`]): what a resource template
/// certifies every class of, each class checked and placed against the one
/// validation. What its topology alone decides is derived once with it.
pub struct ValidatedMetadataDraft {
    draft: NativeGraphMetadataDraft,
    /// Its operand links, which its topology decides.
    links: OperandLinks,
    plan_heap: HostHeap,
    topology_heap: HostHeap,
    bound_heap: HostHeap,
    /// What recording one run of it takes.
    census: GraphCensus,
}

impl ValidatedMetadataDraft {
    /// Whether this validated topology is valid at other shapes: each port's
    /// extents as `ports` gives them, each node's parameter and result
    /// shapes as `parameters` and `results` give them, and each view at
    /// `facts` (a graph trace's evaluated at another model's facts). Every check enqueueing,
    /// storing and sharing made of a shape is made again at them: each
    /// argument against its parameter, each stored result against its port,
    /// each shared result against its primary. The topology is this one's.
    #[allow(clippy::type_complexity)]
    pub fn validate_at(
        &self,
        facts: &[u64],
        ports: &[Vec<u64>],
        parameters: &[Vec<Option<(RepresentationId, Vec<u64>)>>],
        results: &[Vec<Option<(RepresentationId, Vec<u64>)>>],
    ) -> Result<(), CallError> {
        let topology = || CallError::Workflow(WorkflowError::NativeGraphLayoutMismatch(LayoutMismatch::Topology));
        if ports.len() != self.draft.ports.len()
            || parameters.len() != self.draft.nodes.len()
            || results.len() != self.draft.nodes.len()
        {
            return Err(topology());
        }
        let ports = self
            .draft
            .ports
            .iter()
            .zip(ports)
            .map(|(port, extents)| {
                let layout = layout::canonical(port.representation, extents).map_err(CallError::Execution)?;
                Ok(PortSpec {
                    extents: extents.clone(),
                    strides: layout.strides,
                    byte_len: layout.byte_len,
                    ..port.clone()
                })
            })
            .collect::<Result<Vec<_>, CallError>>()?;
        let results = self
            .draft
            .nodes
            .iter()
            .zip(results)
            .map(|(node, shapes)| {
                if node.results.len() != shapes.len() {
                    return Err(topology());
                }
                node.results
                    .iter()
                    .zip(shapes)
                    .map(|(recorded, shape)| match (recorded, shape) {
                        (Some(recorded), Some((representation, extents))) if recorded.representation == *representation => {
                            let layout = layout::canonical(*representation, extents).map_err(CallError::Execution)?;
                            Ok(Some(NativeTensorSpec {
                                representation: *representation,
                                extents: extents.clone(),
                                strides: layout.strides,
                                byte_len: layout.byte_len,
                            }))
                        }
                        (None, None) => Ok(None),
                        _ => Err(topology()),
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        for (node, parameters) in self.draft.nodes.iter().zip(parameters) {
            validate_metadata_arguments(node.entry, &node.planned.args, parameters, &ports, &results, Some(facts))?;
        }
        validate_stores(&self.draft.stores, &ports, &results, &self.draft.exports).map_err(CallError::Workflow)?;
        validate_shares(&self.draft.nodes, &results, &self.draft.exports, &self.draft.stores).map_err(CallError::Workflow)
    }

    /// Certify a single [`NativeGraphLayout`] as the element-wise maximum of
    /// buffer capacities and active launch counts across a non-empty sequence
    /// of concrete exact-shape drafts sharing one topology.
    ///
    /// Every draft must have the exact same topology (backend, entries,
    /// timelines, marks, scratch structure, and storage signature); any
    /// divergence is rejected as [`LayoutMismatch::Topology`].
    pub fn certify_group(
        drafts: impl IntoIterator<Item = NativeGraphMetadataDraft>,
    ) -> Result<NativeGraphLayout, WorkflowError> {
        let topology = || WorkflowError::NativeGraphLayoutMismatch(LayoutMismatch::Topology);
        let mut iter = drafts.into_iter();
        let first = iter.next().ok_or_else(topology)?;
        let validated = first.validated()?;
        let base_sig = storage_signature(
            &validated.draft.ports,
            &validated.draft.nodes,
            &validated.links,
            &validated.draft.nodes,
            &validated.draft.exports,
            &validated.draft.stores,
        );
        let mut ports: Vec<u64> = validated.draft.ports.iter().map(|p| p.byte_len).collect();
        let mut results: Vec<Vec<u64>> = validated
            .draft
            .nodes
            .iter()
            .map(|node| node.results.iter().map(|r| r.as_ref().map_or(0, |s| s.byte_len)).collect())
            .collect();
        let mut scratch: Vec<Vec<ScratchNeed>> = validated
            .draft
            .nodes
            .iter()
            .map(|node| node.planned.scratch.clone())
            .collect();
        let mut active_launches: Vec<u64> = validated
            .draft
            .nodes
            .iter()
            .map(|node| node.active_launches)
            .collect();

        for draft in iter {
            let links = draft.validate_topology()?;
            let same_entries = draft.nodes.len() == validated.draft.nodes.len()
                && draft
                    .nodes
                    .iter()
                    .zip(&validated.draft.nodes)
                    .all(|(a, b)| a.entry == b.entry);
            if draft.backend != validated.draft.backend
                || !same_entries
                || draft.timelines != validated.draft.timelines
                || draft.marks != validated.draft.marks
                || storage_signature(
                    &draft.ports,
                    &draft.nodes,
                    &links,
                    &draft.nodes,
                    &draft.exports,
                    &draft.stores,
                ) != base_sig
            {
                return Err(topology());
            }
            for (cap, port) in ports.iter_mut().zip(&draft.ports) {
                *cap = (*cap).max(port.byte_len);
            }
            for (row_cap, node) in results.iter_mut().zip(&draft.nodes) {
                for (cap, result) in row_cap.iter_mut().zip(&node.results) {
                    if let Some(spec) = result {
                        *cap = (*cap).max(spec.byte_len);
                    }
                }
            }
            for (node_cap, node) in scratch.iter_mut().zip(&draft.nodes) {
                if node_cap.len() != node.planned.scratch.len() {
                    return Err(topology());
                }
                for (cap, need) in node_cap.iter_mut().zip(&node.planned.scratch) {
                    if cap.sync != need.sync || cap.operand != need.operand {
                        return Err(topology());
                    }
                    cap.bytes = cap.bytes.max(need.bytes);
                }
            }
            for (cap, node) in active_launches.iter_mut().zip(&draft.nodes) {
                *cap |= node.active_launches;
            }
        }
        validated.capacity_layout(&ports, &results, scratch, Some(&active_launches))
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
        active_launches: Option<&[u64]>,
    ) -> Result<NativeGraphLayout, WorkflowError> {
        let (ports, nodes, results) = self.at_capacities(ports, results, scratch)?;
        let recorded = match active_launches {
            Some(masks) => self.draft.recorded_for(masks),
            None => self.census,
        };
        Ok(make_layout(
            &ports,
            &nodes,
            &self.links,
            &results,
            &self.draft.exports,
            &self.draft.stores,
            self.plan_heap,
            self.topology_heap,
            self.draft.timelines.len(),
            recorded,
        ))
    }

    /// What [`Self::capacity_layout`] at the same capacities holds, exactly:
    /// the same storage plan's sizes, without the layout a load seals into
    /// (its offsets, signature and capacities).
    pub fn capacity_storage(
        &self,
        ports: &[u64],
        results: &[Vec<u64>],
        scratch: Vec<Vec<ScratchNeed>>,
        active_launches: Option<&[u64]>,
    ) -> Result<NativeGraphStorageBytes, WorkflowError> {
        let (ports, nodes, results) = self.at_capacities(ports, results, scratch)?;
        let storage = plan_storage(&ports, &nodes, &self.links, &results, &self.draft.exports, &self.draft.stores);
        let recorded = match active_launches {
            Some(masks) => self.draft.recorded_for(masks),
            None => self.census,
        };
        Ok(NativeGraphStorageBytes {
            workspace: storage.scratch_bytes,
            sync: storage.sync_bytes,
            output: storage.output_bytes,
            upload: storage.upload_bytes,
            recorded,
        })
    }

    /// The host heap of the plan this topology seals to
    /// ([`NativeGraphMetadataDraft::plan_heap`]).
    pub fn plan_heap(&self) -> HostHeap {
        self.plan_heap
    }

    /// The host heap of the shared static topology for this template
    /// ([`NativeGraphMetadataDraft::topology_heap`]).
    pub fn topology_heap(&self) -> HostHeap {
        self.topology_heap
    }

    /// The host heap of one static binding of a plan of this topology
    /// ([`NativeGraphLayout::bound_heap`]).
    pub fn bound_heap(&self) -> HostHeap {
        self.bound_heap
    }

    /// This validated topology with every buffer at its supplied byte
    /// capacity, each that serves an emission no consumer reads charged the
    /// minimum.
    #[allow(clippy::type_complexity)]
    fn at_capacities(
        &self,
        ports: &[u64],
        results: &[Vec<u64>],
        scratch: Vec<Vec<ScratchNeed>>,
    ) -> Result<(Vec<PortSpec>, Vec<PlannedNode>, Vec<Vec<Option<NativeTensorSpec>>>), WorkflowError> {
        let shaped = ports.len() == self.draft.ports.len()
            && results.len() == self.draft.nodes.len()
            && results
                .iter()
                .zip(&self.draft.nodes)
                .all(|(capacities, node)| capacities.len() == node.results.len())
            && scratch.len() == self.draft.nodes.len()
            && scratch
                .iter()
                .zip(&self.draft.nodes)
                .all(|(capacities, node)| capacities.len() == node.planned.scratch.len());
        if !shaped {
            return Err(WorkflowError::NativeGraphLayoutMismatch(LayoutMismatch::Topology));
        }
        let ports = self
            .draft
            .ports
            .iter()
            .zip(ports)
            .map(|(port, &byte_len)| PortSpec {
                byte_len,
                ..port.clone()
            })
            .collect::<Vec<_>>();
        let results = self
            .draft
            .nodes
            .iter()
            .zip(results)
            .map(|(node, capacities)| {
                node.results
                    .iter()
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
        let mut nodes = self
            .draft
            .nodes
            .iter()
            .zip(scratch)
            .map(|(node, scratch)| PlannedNode {
                args: Arc::clone(&node.planned.args),
                scratch,
                activation: node.planned.activation,
                condition: node.planned.condition,
                share: node.planned.share,
            })
            .collect::<Vec<_>>();
        charge_node_emissions(&mut nodes, &self.links);
        Ok((ports, nodes, results))
    }
}

#[allow(clippy::too_many_arguments)]
fn make_layout(
    ports: &[PortSpec],
    nodes: &[PlannedNode],
    links: &OperandLinks,
    results: &[Vec<Option<NativeTensorSpec>>],
    exports: &[WorkflowResultRef],
    stores: &[Store],
    plan_heap: HostHeap,
    topology_heap: HostHeap,
    timelines: usize,
    recorded: GraphCensus,
) -> NativeGraphLayout {
    NativeGraphLayout {
        identity: NEXT_NATIVE_GRAPH.fetch_add(1, Ordering::Relaxed),
        plan_heap,
        topology_heap,
        timelines,
        recorded,
        storage: plan_storage(ports, nodes, links, results, exports, stores),
        signature: storage_signature(ports, nodes, links, results, exports, stores),
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
        topology: Arc::new(std::sync::Mutex::new(None)),
    }
}

/// Check each argument of a node against its parameter's shape: at the
/// recording, or, with `facts`, each view's operations at a graph trace's
/// facts (`NativeGraphMetadataDraft::validate_at`).
fn validate_metadata_arguments<R: AsRef<[Option<NativeTensorSpec>]>>(
    entry: &'static str,
    args: &EncodedWorkflowArgs,
    parameter_shapes: &[Option<(RepresentationId, Vec<u64>)>],
    ports: &[PortSpec],
    results: &[R],
    facts: Option<&[u64]>,
) -> Result<(), CallError> {
    let mismatch = |parameter: usize, expected: String, bound: String| {
        CallError::Workflow(WorkflowError::NativeGraphArgumentMismatch { entry, parameter, expected, bound })
    };
    let shape = |shape: &Option<(RepresentationId, Vec<u64>)>| match shape {
        Some((representation, extents)) => format!("a {representation:?} tensor {extents:?}"),
        None => "a scalar".to_owned(),
    };
    if parameter_shapes.len() != args.arguments().len() {
        return Err(mismatch(
            0,
            format!("{} arguments", parameter_shapes.len()),
            format!("{} arguments", args.arguments().len()),
        ));
    }
    for (ordinal, (argument, expected)) in args.arguments().iter().zip(parameter_shapes).enumerate()
    {
        match (argument, expected) {
            (EncodedWorkflowArgument::Tensor(argument), Some((representation, extents))) => {
                let actual = metadata_argument_shape(entry, argument, ordinal, ports, results, facts)?;
                if actual.0 != *representation || actual.1 != *extents {
                    return Err(mismatch(ordinal, shape(expected), shape(&Some(actual))));
                }
            }
            (EncodedWorkflowArgument::Scalar(_), None) => {}
            (EncodedWorkflowArgument::Tensor(_), None) => {
                return Err(mismatch(ordinal, shape(expected), "a tensor".to_owned()));
            }
            (EncodedWorkflowArgument::Scalar(_) | EncodedWorkflowArgument::ScalarResult(_), Some(_)) => {
                return Err(mismatch(ordinal, shape(expected), "a scalar".to_owned()));
            }
            (EncodedWorkflowArgument::ScalarResult(_), None) => {
                return Err(mismatch(ordinal, "a host scalar".to_owned(), "a node's scalar result".to_owned()));
            }
        }
    }
    Ok(())
}

fn metadata_argument_shape<R: AsRef<[Option<NativeTensorSpec>]>>(
    entry: &'static str,
    argument: &WorkflowTensorArgument,
    ordinal: usize,
    ports: &[PortSpec],
    results: &[R],
    facts: Option<&[u64]>,
) -> Result<(RepresentationId, Vec<u64>), CallError> {
    let (reference, operations) = match argument {
        WorkflowTensorArgument::Result(reference) => (*reference, std::borrow::Cow::Borrowed(&[][..])),
        WorkflowTensorArgument::ResultView { result, operations, expressions } => match (facts, expressions) {
            (Some(facts), Some(expressions)) => (
                *result,
                std::borrow::Cow::Owned(
                    expressions
                        .iter()
                        .map(|expression| expression.at(facts))
                        .collect::<Option<Vec<_>>>()
                        .ok_or_else(|| {
                            CallError::Workflow(WorkflowError::NativeGraphArgumentMismatch {
                                entry,
                                parameter: ordinal,
                                expected: "a view defined at the trace's facts".to_owned(),
                                bound: "a view with an undefined extent there".to_owned(),
                            })
                        })?,
                ),
            ),
            _ => (*result, std::borrow::Cow::Borrowed(operations.as_slice())),
        },
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
    for operation in operations.iter() {
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

/// Every port is used: by a node's argument or condition (the port whose
/// word its launches are encoded under), an export, a store or a carry. A
/// carry's ports are its alone: no node, export or store names them.
fn validate_used_ports<'a, F>(
    port_count: usize,
    nodes: impl Iterator<Item = (&'a EncodedWorkflowArgs, Option<u32>)>,
    exports: &[WorkflowResultRef],
    stores: &[Store],
    carries: &[Carry<F>],
) -> Result<(), WorkflowError> {
    let mut used = vec![false; port_count];
    for (node, condition) in nodes {
        if let Some(port) = condition {
            used[port as usize] = true;
        }
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
    for carry in carries {
        for port in [carry.rows, carry.quantized] {
            if used[port as usize] {
                return Err(WorkflowError::NativePortMismatch { port: port as usize });
            }
            used[port as usize] = true;
        }
    }
    if let Some(port) = used.iter().position(|used| !used) {
        return Err(WorkflowError::NativePortUnbound { port });
    }
    Ok(())
}

pub fn argument_reference(argument: &EncodedWorkflowArgument) -> Option<WorkflowResultRef> {
    match argument {
        EncodedWorkflowArgument::Tensor(WorkflowTensorArgument::Result(reference))
        | EncodedWorkflowArgument::ScalarResult(reference) => Some(*reference),
        EncodedWorkflowArgument::Tensor(WorkflowTensorArgument::ResultView { result, .. }) => {
            Some(*result)
        }
        _ => None,
    }
}

fn result_spec<R: AsRef<[Option<NativeTensorSpec>]>>(
    results: &[R],
    reference: WorkflowResultRef,
) -> Option<&NativeTensorSpec> {
    results
        .get(reference.node as usize)?
        .as_ref()
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
                ..
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
    /// Its arguments, which every class a resource template places shares.
    args: Arc<EncodedWorkflowArgs>,
    /// Each of the node's call-private scratch buffers.
    scratch: Vec<ScratchNeed>,
    /// The element its entry binds its activation to
    /// ([`super::OPERAND_ACTIVATION`]), where it has one: a pair's or a
    /// carry's ends bind one.
    activation: Option<RepresentationId>,
    /// Its condition (`NodeCondition`): the external port, the word and what
    /// the node is encoded under.
    condition: Option<Condition>,
    /// Its result that shares an earlier or later node's result storage
    /// (`NativeGraphDraft::share`): its ordinal and that result.
    share: Option<(u32, WorkflowResultRef)>,
}

impl AsRef<PlannedNode> for PlannedNode {
    fn as_ref(&self) -> &PlannedNode {
        self
    }
}

impl AsMut<PlannedNode> for PlannedNode {
    fn as_mut(&mut self) -> &mut PlannedNode {
        self
    }
}

/// Plain descriptor of one graph node for [`resolve_operand_links`].
#[derive(Clone, Copy)]
pub struct OperandNode<'a> {
    pub args: &'a EncodedWorkflowArgs,
    pub scratch: &'a [ScratchNeed],
    pub activation: Option<RepresentationId>,
}

/// Plain descriptor of one graph carry for [`resolve_operand_links`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OperandCarrySpec {
    pub rows: u32,
    pub quantized: u32,
    pub activation: RepresentationId,
    pub side: OperandCarrySide,
}

/// One operand pairing (`ScratchOperand`): node `consumer`'s buffer
/// `formed` holds one form of the operand of a tensor node `producer` writes
/// whole, whose buffer `emitted` holds that form of that tensor's operand,
/// under the same norm. The two are one storage, live from the producer to
/// the consumer: the producer's launches write the operand, and the consumer
/// reads it, or forms it over the producer's where the producer's does not
/// stand for its own (`formations_left_out`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OperandPair {
    pub consumer: usize,
    pub formed: usize,
    pub producer: usize,
    pub emitted: usize,
}

/// Every operand pairing of `nodes`: each consumer's formed operand whose
/// tensor parameter is bound to a whole result of an earlier node that emits
/// that form of that result's operand, under one norm: none on either, or
/// each bound to the same norm weight and epsilon. An emitted operand pairs
/// with its first consumer alone: another consumer's own layout may differ,
/// so it forms its own.
fn operand_pairs(nodes: &[OperandNode<'_>]) -> Vec<OperandPair> {
    let mut pairs: Vec<OperandPair> = Vec::new();
    for (consumer, node) in nodes.iter().enumerate() {
        for (formed, need) in node.scratch.iter().enumerate() {
            let Some(ScratchOperand::Formed { parameter, rows, norm, .. }) = need.operand else { continue };
            // A view of a result is not its whole tensor, whose operand the
            // producer emits.
            let source = match node.args.arguments().get(parameter as usize) {
                Some(EncodedWorkflowArgument::Tensor(WorkflowTensorArgument::Result(source)))
                    if source.node != INPUT_NODE =>
                {
                    *source
                }
                _ => continue,
            };
            let producer = source.node as usize;
            let emitted = nodes[producer].scratch.iter().position(|need| match need.operand {
                Some(ScratchOperand::Emitted { result, rows: emitted_rows, norm: emitted_norm }) => {
                    result == source.result
                        && emitted_rows == rows
                        && same_norm(node.args, norm, nodes[producer].args, emitted_norm)
                }
                _ => false,
            });
            let Some(emitted) = emitted else { continue };
            if pairs.iter().any(|pair| pair.producer == producer && pair.emitted == emitted) {
                continue;
            }
            pairs.push(OperandPair { consumer, formed, producer, emitted });
        }
    }
    pairs
}

/// Whether a consumer's operand norm (`consumer`, of its arguments
/// `formed`) and a producer's (`producer`, of `emitting`) are one: both
/// absent, or both bound to the same norm weight tensor and the same
/// epsilon.
fn same_norm(
    formed: &EncodedWorkflowArgs,
    consumer: Option<NativeOperandNorm>,
    emitting: &EncodedWorkflowArgs,
    producer: Option<NativeOperandNorm>,
) -> bool {
    let argument = |args: &EncodedWorkflowArgs, ordinal: u32| args.arguments().get(ordinal as usize).cloned();
    match (consumer, producer) {
        (None, None) => true,
        (Some(consumer), Some(producer)) => {
            let weights = match (argument(formed, consumer.weight), argument(emitting, producer.weight)) {
                (Some(EncodedWorkflowArgument::Tensor(a)), Some(EncodedWorkflowArgument::Tensor(b))) => match (a, b) {
                    (WorkflowTensorArgument::External(a), WorkflowTensorArgument::External(b)) => Arc::ptr_eq(&a, &b),
                    (WorkflowTensorArgument::Result(a), WorkflowTensorArgument::Result(b)) => a == b,
                    (
                        WorkflowTensorArgument::ResultView { result: a, operations: x, .. },
                        WorkflowTensorArgument::ResultView { result: b, operations: y, .. },
                    ) => a == b && x == y,
                    _ => false,
                },
                _ => false,
            };
            let epsilons = match (argument(formed, consumer.epsilon), argument(emitting, producer.epsilon)) {
                (Some(EncodedWorkflowArgument::Scalar(a)), Some(EncodedWorkflowArgument::Scalar(b))) => a == b,
                _ => false,
            };
            weights && epsilons
        }
        _ => false,
    }
}

/// Where a graph's operand buffers lie beyond their own node: its in-graph
/// pairs (`operand_pairs`) and its carried buffers. Its topology decides it
/// ([`resolve_operand_links`]), every class of a layout alike.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OperandLinks {
    pub pairs: Vec<OperandPair>,
    pub carried: Vec<CarriedBuffer>,
}

/// Node `node`'s operand buffer `scratch`, which lies on port `port` of
/// carry `carry` (its ordinal among the graph's carries): an emitted
/// buffer of a carry out, which the node writes, or a formed one of a carry
/// in, which it reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CarriedBuffer {
    pub node: usize,
    pub scratch: usize,
    pub port: u32,
    pub carry: usize,
}

impl OperandLinks {
    /// Whether node `node`'s buffer `scratch` is a consumer's operand its
    /// producer emits: in the graph or through a carry.
    pub fn linked(&self, node: usize, scratch: usize) -> bool {
        self.pairs.iter().any(|pair| pair.consumer == node && pair.formed == scratch)
            || self.carried.iter().any(|carried| carried.node == node && carried.scratch == scratch)
    }

    /// The carry node `node`'s buffer `scratch` lies on, if any.
    pub fn carried(&self, node: usize, scratch: usize) -> Option<&CarriedBuffer> {
        self.carried.iter().find(|carried| carried.node == node && carried.scratch == scratch)
    }

    /// The buffers node `node` emits an operand into for a consumer: its
    /// paired and carried out ones.
    pub fn emitted(&self, node: usize) -> impl Iterator<Item = usize> + '_ {
        let paired = self.pairs.iter().filter(move |pair| pair.producer == node).map(|pair| pair.emitted);
        let carried = self.carried.iter().filter(move |carried| carried.node == node).map(|carried| carried.scratch);
        paired.chain(carried)
    }

    /// Charge `MINIMUM_SCRATCH_BYTES` to every emission scratch buffer of
    /// `node` whose result is not emitted for any consumer in `self`.
    pub fn charge_node_emissions(&self, node: usize, scratch: &mut [ScratchNeed]) {
        let emitted = self
            .emitted(node)
            .filter_map(|ordinal| match scratch[ordinal].operand {
                Some(ScratchOperand::Emitted { result, .. }) => Some(result),
                _ => None,
            })
            .collect::<Vec<_>>();
        charge_emissions(scratch, |result| emitted.contains(&result));
    }
}

/// Resolve the operand links of plain node descriptors `nodes` and carry
/// descriptors `carries`: its in-graph pairs, and each carry's buffers.
///
/// - A carry out's producer emits both forms of its result's operand under
///   a norm, neither paired in the graph (`OperandEmitter` otherwise).
/// - A carry in's readers are the consumers whose formed operand is of the
///   carried port's whole tensor, under the carried norm port's tensor and
///   epsilon: each forms the rows once and the Q8 blocks at most once
///   (`OperandForms` otherwise); at least one reads it
///   (`UnreadOperandCarry` otherwise).
/// - Every pair's ends bind one activation element, and every carry's ends
///   the carry's (`OperandElement` otherwise).
pub fn resolve_operand_links(
    nodes: &[OperandNode<'_>],
    carries: &[OperandCarrySpec],
) -> Result<OperandLinks, WorkflowError> {
    let pairs = operand_pairs(nodes);
    if let Some(pair) = pairs.iter().find(|pair| nodes[pair.producer].activation != nodes[pair.consumer].activation) {
        return Err(WorkflowError::OperandElement { node: pair.consumer });
    }
    let mut carried = Vec::new();
    for (index, carry) in carries.iter().enumerate() {
        let port = |rows: bool| if rows { carry.rows } else { carry.quantized };
        let buffer = |node: usize, (scratch, rows): (usize, bool)| CarriedBuffer { node, scratch, port: port(rows), carry: index };
        match carry.side {
            OperandCarrySide::Out { result } => {
                let producer = result.node as usize;
                let emitted = |rows: bool| {
                    nodes[producer].scratch.iter().position(|need| {
                        matches!(
                            need.operand,
                            Some(ScratchOperand::Emitted { result: of, rows: form, norm: Some(_) })
                                if of == result.result && form == rows
                        )
                    })
                };
                let (Some(emitted_rows), Some(emitted_blocks)) = (emitted(true), emitted(false)) else {
                    return Err(WorkflowError::OperandEmitter { node: producer });
                };
                if pairs
                    .iter()
                    .any(|pair| pair.producer == producer && [emitted_rows, emitted_blocks].contains(&pair.emitted))
                {
                    return Err(WorkflowError::OperandEmitter { node: producer });
                }
                if nodes[producer].activation != Some(carry.activation) {
                    return Err(WorkflowError::OperandElement { node: producer });
                }
                carried.extend([(emitted_rows, true), (emitted_blocks, false)].map(|form| buffer(producer, form)));
            }
            OperandCarrySide::In { hidden, norm, epsilon } => {
                let mut read = false;
                for (consumer, node) in nodes.iter().enumerate() {
                    let arguments = node.args.arguments();
                    let whole_port = |ordinal: u32, port: u32| {
                        matches!(
                            arguments.get(ordinal as usize),
                            Some(EncodedWorkflowArgument::Tensor(WorkflowTensorArgument::Result(reference)))
                                if reference.node == INPUT_NODE && reference.result == port
                        )
                    };
                    let reading = node
                        .scratch
                        .iter()
                        .enumerate()
                        .filter_map(|(ordinal, need)| match need.operand {
                            Some(ScratchOperand::Formed { parameter, rows, norm: Some(formed), .. })
                                if whole_port(parameter, hidden)
                                    && whole_port(formed.weight, norm)
                                    && matches!(
                                        arguments.get(formed.epsilon as usize),
                                        Some(EncodedWorkflowArgument::Scalar(value))
                                            if *value == EncodedScalar::F32Bits(epsilon)
                                    ) =>
                            {
                                Some((ordinal, rows))
                            }
                            _ => None,
                        })
                        .collect::<Vec<_>>();
                    if reading.is_empty() {
                        continue;
                    }
                    let forms = |rows: bool| reading.iter().filter(|&&(_, form)| form == rows).count();
                    if forms(true) != 1 || forms(false) > 1 {
                        return Err(WorkflowError::OperandForms { node: consumer });
                    }
                    if node.activation != Some(carry.activation) {
                        return Err(WorkflowError::OperandElement { node: consumer });
                    }
                    read = true;
                    carried.extend(reading.into_iter().map(|form| buffer(consumer, form)));
                }
                if !read {
                    return Err(WorkflowError::UnreadOperandCarry { port: carry.rows as usize });
                }
            }
        }
    }
    Ok(OperandLinks { pairs, carried })
}

/// The operand links of `nodes` and `carries`: delegates to
/// [`resolve_operand_links`], ignoring any carry forms `F`.
fn operand_links<F, N: AsRef<PlannedNode>>(nodes: &[N], carries: &[Carry<F>]) -> Result<OperandLinks, WorkflowError> {
    let node_specs = nodes
        .iter()
        .map(|node| {
            let node = node.as_ref();
            OperandNode {
                args: &node.args,
                scratch: &node.scratch,
                activation: node.activation,
            }
        })
        .collect::<Vec<_>>();
    let carry_specs = carries
        .iter()
        .map(|carry| OperandCarrySpec {
            rows: carry.rows,
            quantized: carry.quantized,
            activation: carry.activation,
            side: carry.side,
        })
        .collect::<Vec<_>>();
    resolve_operand_links(&node_specs, &carry_specs)
}

/// Charge each node of `nodes` the minimum for every buffer serving the
/// emission of a result it emits no operand of for a consumer (`links`):
/// unpaired and uncarried producers reserve what a call alone does
/// (`charge_emissions`). Every class of a topology is charged alike, so a
/// layout's capacities bound its classes'.
fn charge_node_emissions<N: AsMut<PlannedNode>>(nodes: &mut [N], links: &OperandLinks) {
    for (node, planned) in nodes.iter_mut().enumerate() {
        links.charge_node_emissions(node, &mut planned.as_mut().scratch);
    }
}

/// The formation launches node `node` leaves out at its call (`forms`):
/// each launch of whose formed operand buffers (`scratch`) some are formed
/// at it, and every one formed is linked to an emitting producer, in the
/// graph or through a carry (`links`), whose operand stands for it
/// (`FormedAt::pairs`). A launch forming nothing linked (a staging
/// buffer's alone) stays.
fn formations_left_out(
    node: usize,
    scratch: &[ScratchNeed],
    forms: &[Option<FormedAt>],
    links: &OperandLinks,
) -> Vec<usize> {
    let formation = |ordinal: usize| match scratch[ordinal].operand {
        Some(ScratchOperand::Formed { formation, .. }) => Some(formation as usize),
        _ => None,
    };
    let mut launches = (0..scratch.len()).filter_map(formation).collect::<Vec<_>>();
    launches.sort_unstable();
    launches.dedup();
    launches.retain(|&launch| {
        let mut formed = (0..scratch.len())
            .filter(|&ordinal| formation(ordinal) == Some(launch))
            .filter_map(|ordinal| forms[ordinal].filter(|at| at.formed).map(|at| (ordinal, at)))
            .peekable();
        formed.peek().is_some() && formed.all(|(ordinal, at)| at.pairs && links.linked(node, ordinal))
    });
    launches
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
/// reads, exports or stores it (its readers read the result it shares), the
/// node does not read the result it shares (the two take the same bytes), and
/// the result it shares is neither stored nor itself shared.
fn validate_shares<N: AsRef<PlannedNode>, R: AsRef<[Option<NativeTensorSpec>]>>(
    nodes: &[N],
    results: &[R],
    exports: &[WorkflowResultRef],
    stores: &[Store],
) -> Result<(), WorkflowError> {
    for (node, planned) in nodes.iter().enumerate() {
        let planned = planned.as_ref();
        let Some((ordinal, primary)) = planned.share else { continue };
        let other = WorkflowResultRef { workflow: primary.workflow, node: node as u32, result: ordinal };
        let spec = |reference: WorkflowResultRef| {
            (reference.node != INPUT_NODE)
                .then(|| result_spec(results, reference))
                .flatten()
        };
        let (Some(own), Some(shared)) = (spec(other), spec(primary)) else {
            return Err(WorkflowError::SharedResult { node });
        };
        let same = own.representation == shared.representation
            && own.extents == shared.extents
            && own.strides == shared.strides
            && own.byte_len == shared.byte_len;
        let primary_node = nodes[primary.node as usize].as_ref();
        let read = nodes.iter().any(|node| {
            node.as_ref()
                .args
                .arguments()
                .iter()
                .any(|argument| argument_reference(argument) == Some(other))
        }) || planned.args.arguments().iter().any(|argument| argument_reference(argument) == Some(primary));
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Region {
    /// The tensor bound to this external port (a port ordinal is a `u32`
    /// result reference); the site offset is added to the tensor's own byte
    /// offset.
    External(u32),
    /// The slot's workspace arena: results, graph locals and node scratch.
    Workspace,
    /// The slot's own sync region: node `sync` scratch, zero whenever a
    /// launch starts.
    Sync,
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
    /// How it binds records, if it is a record port's.
    records: SiteRecords,
    /// The site after it in its node's external chain
    /// ([`ExternalPairs::first`]), where it is in one.
    next_external: Option<u32>,
}

impl Site {
    /// The bytes of its region it reaches.
    fn range(&self) -> std::ops::Range<u64> {
        self.offset..self.offset + self.bytes
    }
}

/// The records a site binds (whole records, checked at seal), through a
/// table of their addresses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SiteRecords {
    /// It binds its view: no record port's.
    None,
    /// An external record port's records from the record at the site's
    /// offset, through the port's table. A site of no records is still one:
    /// it binds its port's table, as the port's storage may be no buffer any
    /// binding reaches (a reservation's chunks).
    Port(u64),
    /// In the workspace, records the slot's workspace record table
    /// (`NativeGraphPlan::workspace_record_table`) addresses from byte
    /// `table`. A workspace record site of no records is none: the
    /// workspace is ordinary storage, which its view binds.
    Workspace { records: NonZeroU64, table: u64 },
}

/// How a graph's nodes use one external port.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct PortUse {
    /// A launch reads its grid from it, so it binds no slab view.
    grid: bool,
    /// A node binds it to a slab port (`&slab`), so it binds a slab view
    /// or no elements.
    slab: bool,
    /// A node binds it as plain elements (neither `&slab` nor `records`,
    /// and not a host port), so it binds no non-empty slab view.
    plain: bool,
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
    /// Where it is in [`Executable::exclusive`]'s chain: its external
    /// pairs that bindings are checked for.
    exclusive: Option<ExternalPairs>,
}

/// A node's exclusive pairs ([`exclusive_pairs`]) of buffers both in external
/// ports' tensors, whose bytes only bindings fix, as
/// [`NativeGraphPlan::check_disjoint`] walks them: its external sites in a
/// chain through [`Site::next_external`], those it writes first, so each
/// one it writes pairs with every site after it, and nothing else does.
/// Held inline in the node and its sites, so a plan's host heap is what its
/// topology fixes ([`NativeGraphMetadataDraft::plan_heap`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ExternalPairs {
    /// The next node of the chain.
    next: Option<u32>,
    /// Its first external site.
    first: u32,
}

impl SealedNode {
    /// The refusal of its buffers `first` and `second` (ABI order), an
    /// exclusive pair ([`exclusive_pairs`]), bound to bytes in common.
    fn aliasing(&self, first: usize, second: usize) -> CallError {
        CallError::Invocation(InvocationError::IllegalAliasing {
            first: self.buffer_name(first),
            second: self.buffer_name(second),
        })
    }

    /// The name of its buffer `buffer` (ABI order): a tensor parameter's
    /// own ([`super::abi::tensor_buffer_name`]), a tensor result's ordinal
    /// among the entry's results, a scratch buffer's own.
    fn buffer_name(&self, buffer: usize) -> String {
        let (schema, implementation) = (self.kernel.schema(), self.kernel.implementation());
        let parameters = schema
            .parameters()
            .iter()
            .filter(|parameter| matches!(parameter.kind, ParameterKind::Tensor { .. }))
            .count();
        let scratch = self.sites.len() - implementation.scratch.len();
        if buffer < parameters {
            return super::abi::tensor_buffer_name(schema, buffer).to_owned();
        }
        if buffer >= scratch {
            return format!("scratch {}", implementation.scratch[buffer - scratch].name);
        }
        let (ordinal, _) = schema
            .results()
            .iter()
            .enumerate()
            .filter(|(_, result)| matches!(result.kind, ResultKind::Tensor { .. }))
            .nth(buffer - parameters)
            .expect("a result buffer is a tensor result's");
        format!("result {ordinal}")
    }
}

/// The external port of `node`'s site `buffer`, a link of its exclusive
/// chain ([`ExternalPairs`]), and the site.
fn external_site(node: &SealedNode, buffer: u32) -> (usize, &Site) {
    let site = &node.sites[buffer as usize];
    let Region::External(port) = site.region else {
        unreachable!("an external chain links external sites");
    };
    (port as usize, site)
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
    /// The last node two of whose exclusive buffers lie in external ports'
    /// tensors, the first of the chain through [`ExternalPairs::next`] that
    /// bindings are checked along
    /// ([`NativeGraphPlan::check_disjoint`]): the nodes a run's bindings can
    /// alias, and only those.
    exclusive: Option<u32>,
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
        let record_ports = kernel.records(&shape.launches);
        let mut sites = Vec::with_capacity(buffers);
        let mut representations = Vec::with_capacity(buffers);
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
                    None => (Region::External(reference.result), 0),
                }
            } else {
                storage.placements[reference.node as usize][reference.result as usize].site()
            };
            let write = access.writer().is_some();
            match region {
                Region::External(port) => {
                    self.external_writes[port as usize] |= write;
                    self.port_uses[port as usize].grid |= kernel.reads_grid(ordinal);
                    self.port_uses[port as usize].slab |= kernel.addresses_slab(ordinal);
                    // A slab port binds its port's address table from its
                    // first entry (`QueuedRun::dispatch`), so it takes the
                    // port's view from its first byte: a view of it from
                    // another byte has no table the binding could address.
                    // So a slab view with elements is bound only where a
                    // site's offset is both its table's and its view's (the
                    // hazards' and `NativeGraphPlan::check_disjoint`'s); an
                    // empty one reaches nothing from any byte.
                    if kernel.addresses_slab(ordinal) && descriptor.byte_offset != 0 && descriptor.byte_len > 0 {
                        return Err(WorkflowError::PlainSlabBinding {
                            parameter: parameter.name.clone(),
                        });
                    }
                }
                // Graph storage is never a slab view: a slab port takes it
                // only where it holds no elements.
                Region::Upload | Region::Workspace | Region::Sync | Region::Outputs
                    if kernel.addresses_slab(ordinal) && descriptor.byte_len > 0 =>
                {
                    return Err(WorkflowError::PlainSlabBinding {
                        parameter: parameter.name.clone(),
                    });
                }
                Region::Upload => self.upload_written |= write,
                Region::Workspace | Region::Sync | Region::Outputs => {}
            }
            let mut site = Site {
                region,
                offset: base + descriptor.byte_offset,
                bytes: descriptor.byte_len,
                records: SiteRecords::None,
                next_external: None,
            };
            let (extents, strides) = (&descriptor.extents, &descriptor.strides);
            let (record, bound) = (record_ports[sites.len()], kernel.bound[sites.len()]);
            let slab = kernel.addresses_slab(ordinal);
            if !self.use_site(&mut site, record, slab, bound, descriptor.representation, extents, strides, ports) {
                return Err(WorkflowError::RecordPortBinding {
                    parameter: parameter.name.clone(),
                });
            }
            sites.push(site);
            representations
                .push(seismic_lang::registry::representation_info(descriptor.representation).name);
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
                records: SiteRecords::None,
                next_external: None,
            };
            let record = record_ports[sites.len()];
            if !self.use_site(&mut site, record, false, true, spec.representation, &spec.extents, &spec.strides, ports) {
                return Err(WorkflowError::RecordPortBinding {
                    parameter: format!("result {ordinal}"),
                });
            }
            sites.push(site);
            representations
                .push(seismic_lang::registry::representation_info(spec.representation).name);
        }
        // Scratch at the graph's charges (`charge_node_emissions`): in the
        // workspace, or a carried operand buffer in its port's tensor, which
        // covers it, written by the producer and read by the consumer.
        for (placement, need) in storage.node_scratch[node].iter().zip(&planned.scratch) {
            let (region, offset) = placement.site();
            if let Region::External(port) = region {
                if need.bytes > ports[port as usize].byte_len {
                    return Err(WorkflowError::NativePortMismatch { port: port as usize });
                }
                // Its bytes are elements, so the port binds no slab view
                // with elements (whose binding is its address table).
                self.port_uses[port as usize].direct = true;
                self.port_uses[port as usize].plain = true;
                if let Some(ScratchOperand::Emitted { .. }) = need.operand {
                    self.external_writes[port as usize] = true;
                }
            }
            sites.push(Site {
                region,
                offset,
                bytes: need.bytes,
                records: SiteRecords::None,
                next_external: None,
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
            Region::Workspace | Region::Sync | Region::Outputs => None,
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
                        let (word, offset) = matches!(site.records, SiteRecords::None)
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
            exclusive: None,
        });
        Ok(())
    }

    /// Hold node `node`'s buffers to the no-alias rule
    /// ([`exclusive_pairs`]). Two of an exclusive pair in graph storage
    /// meet only where they bind one value (its `planned` arguments name
    /// it twice), which is refused; the planner places distinct values a
    /// node binds apart. Pairs both in external ports' tensors, whose bytes
    /// only bindings fix, are chained ([`ExternalPairs`]) for
    /// [`NativeGraphPlan::check_disjoint`], unless each is two ranges of
    /// one port that cannot meet: a port's view reaches its bytes one to
    /// one, a slab view's slots distinct slabs. (Two of one port's that do
    /// meet are refused once every node is sealed where the port binds a
    /// plain view, [`Self::refuse_plain_overlaps`].) A buffer in graph
    /// storage never meets an external tensor: a run's attach refuses a
    /// port bound to its own storage.
    fn exclude(&mut self, node: usize, planned: &PlannedNode) -> Result<(), CallError> {
        let sealed = &self.nodes[node];
        // The value each tensor parameter's buffer binds; a result's or a
        // scratch buffer's is its own.
        let values: Vec<Option<WorkflowResultRef>> = sealed
            .kernel
            .schema()
            .parameters()
            .iter()
            .zip(planned.args.arguments())
            .filter(|(parameter, _)| matches!(parameter.kind, ParameterKind::Tensor { .. }))
            .map(|(_, argument)| argument_reference(argument))
            .collect();
        let value = |buffer: usize| values.get(buffer).copied().flatten();
        let writes = |buffer: usize| sealed.kernel.writes(buffer);
        let mut chained = false;
        for (first, second) in exclusive_pairs(sealed.sites.len(), writes) {
            let (first_site, second_site) = (&sealed.sites[first], &sealed.sites[second]);
            let meets = meet(&first_site.range(), &second_site.range());
            match (first_site.region, second_site.region) {
                (Region::External(port), Region::External(other)) => chained |= port != other || meets,
                (Region::External(_), _) | (_, Region::External(_)) => {}
                (region, other) if region == other && meets => {
                    if value(first).is_some() && value(first) == value(second) {
                        return Err(sealed.aliasing(first, second));
                    }
                    panic!(
                        "graph storage planned buffers {first} and {second} of node {node}, distinct values, in common bytes"
                    );
                }
                _ => {}
            }
        }
        if !chained {
            return Ok(());
        }
        let external = |buffer: &usize| matches!(sealed.sites[*buffer].region, Region::External(_));
        let (written, read): (Vec<usize>, Vec<usize>) =
            (0..sealed.sites.len()).filter(external).partition(|&buffer| writes(buffer));
        let chain: Vec<u32> = written
            .into_iter()
            .chain(read)
            .map(|buffer| u32::try_from(buffer).expect("native node buffer ordinal exhausted"))
            .collect();
        let sealed = &mut self.nodes[node];
        for link in chain.windows(2) {
            sealed.sites[link[0] as usize].next_external = Some(link[1]);
        }
        let ordinal = u32::try_from(node).expect("native graph node ordinal exhausted");
        sealed.exclusive = Some(ExternalPairs { next: self.exclusive.replace(ordinal), first: chain[0] });
        Ok(())
    }

    /// The pairs of the exclusive chain ([`ExternalPairs`]): each node of
    /// it with each external site it writes, which lead its chain, and
    /// every site after that one, as buffers (ABI order), the written first.
    fn external_pairs(&self) -> impl Iterator<Item = (&SealedNode, u32, u32)> + '_ {
        let pairs = |node: &SealedNode| node.exclusive.expect("a node of the exclusive chain holds its external pairs");
        std::iter::successors(self.exclusive, move |&ordinal| pairs(&self.nodes[ordinal as usize]).next).flat_map(
            move |ordinal| {
                let node = &self.nodes[ordinal as usize];
                let after = move |buffer: u32| {
                    std::iter::successors(node.sites[buffer as usize].next_external, move |&next| {
                        node.sites[next as usize].next_external
                    })
                };
                let first = pairs(node).first;
                std::iter::once(first)
                    .chain(after(first))
                    .take_while(move |&buffer| node.kernel.writes(buffer as usize))
                    .flat_map(move |written| after(written).map(move |second| (node, written, second)))
            },
        )
    }

    /// Refuse, once every node is sealed, two ranges of one port in a
    /// node's exclusive pair ([`Self::external_pairs`]) that meet where the
    /// port binds a plain view (`PortUse::plain`), which reaches its bytes
    /// one to one: they meet under every binding. Only two of a port that
    /// may bind a slab view, whose free slots reach nothing, stay a
    /// question for each run's bindings
    /// ([`NativeGraphPlan::check_disjoint`]).
    fn refuse_plain_overlaps(&self) -> Result<(), CallError> {
        for (node, first, second) in self.external_pairs() {
            let ((port, first_site), (other, second_site)) = (external_site(node, first), external_site(node, second));
            if port == other && self.port_uses[port].plain && meet(&first_site.range(), &second_site.range()) {
                return Err(node.aliasing(first.min(second) as usize, first.max(second) as usize));
            }
        }
        Ok(())
    }

    /// Record how `site`, a view of `extents` and `strides`, uses its
    /// storage, and whether it may bind there. A record port's site
    /// (`record`) binds only whole records of a canonical view (of its
    /// records' bytes): of an external port's tensor from a record boundary,
    /// through the table formed as it is bound
    /// ([`TensorInner::forming_records`]), or in the workspace, through the
    /// table each slot forms with its workspace
    /// ([`NativeGraphPlan::workspace_record_table`]). One in a run's upload
    /// region or output arena is refused: its table would be formed per run.
    #[allow(clippy::too_many_arguments)]
    fn use_site(
        &mut self,
        site: &mut Site,
        record: bool,
        slab: bool,
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
                site.records = SiteRecords::Port(records);
                whole
            }
            (Region::Workspace, true) => {
                let whole = record_bytes(representation, extents).is_some() && canonical();
                if let Some(records) = NonZeroU64::new(records) {
                    site.records = SiteRecords::Workspace {
                        records,
                        table: seismic_lang::registry::record_table_entry(self.workspace_records),
                    };
                    self.workspace_records += records.get();
                }
                whole
            }
            // A host port is ordered by, never bound: it binds any storage.
            (Region::External(port), false) => {
                self.port_uses[port as usize].direct |= bound;
                self.port_uses[port as usize].plain |= bound && !slab;
                true
            }
            (_, record) => !record,
        }
    }
}

/// Where a graph buffer is placed in storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoragePlacement {
    Scratch(u64),
    /// Arrival counters in the slot's sync region.
    Sync(u64),
    Export(u64),
    /// A host-written input in the per-submission upload region.
    Upload(u64),
    /// A stored result or a carried operand buffer: the tensor bound to
    /// this external port.
    Port(u32),
    Scalar,
}

type Placement = StoragePlacement;

impl StoragePlacement {
    fn site(self) -> (Region, u64) {
        match self {
            Self::Scratch(offset) => (Region::Workspace, offset),
            Self::Sync(offset) => (Region::Sync, offset),
            Self::Export(offset) => (Region::Outputs, offset),
            Self::Upload(offset) => (Region::Upload, offset),
            Self::Port(port) => (Region::External(port), 0),
            Self::Scalar => unreachable!("a scalar result is never a graph buffer"),
        }
    }
}

/// Plain descriptor of one graph port for [`plan_storage_sizes`] and
/// [`plan_storage_with_links`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StoragePort {
    pub byte_len: u64,
    pub local: bool,
    pub owned_input: bool,
    pub prewritten: bool,
}

/// Plain descriptor of one graph node for [`plan_storage_sizes`] and
/// [`plan_storage_with_links`].
#[derive(Clone, Copy)]
pub struct StorageNode<'a> {
    pub args: &'a EncodedWorkflowArgs,
    pub results: &'a [Option<u64>],
    pub scratch: &'a [ScratchNeed],
    pub activation: Option<RepresentationId>,
    pub share: Option<(u32, WorkflowResultRef)>,
}

/// Region byte sizes for one slice of a [`CompiledStoragePlan`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StorageSizes {
    pub workspace: u64,
    pub workspace_floor: u64,
    pub sync: u64,
    pub output: u64,
    pub upload: u64,
}

/// Placed graph storage returned by [`plan_storage_sizes`] and
/// [`plan_storage_with_links`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedStorage {
    pub port_placements: Vec<Option<StoragePlacement>>,
    pub placements: Vec<Vec<StoragePlacement>>,
    pub node_scratch: Vec<Vec<StoragePlacement>>,
    pub workspace: u64,
    pub workspace_floor: u64,
    pub sync: u64,
    pub output: u64,
    pub upload: u64,
}

#[cfg(test)]
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
    /// Per node, each scratch buffer's: in the workspace, or a carried
    /// operand buffer's port.
    node_scratch: Vec<Vec<Placement>>,
    scratch_bytes: u64,
    /// The most workspace bytes live at one step: no placement fits in less.
    scratch_floor_bytes: u64,
    /// The sync region's bytes: every node's `sync` scratch, each disjoint
    /// from every other's.
    sync_bytes: u64,
    output_bytes: u64,
    upload_bytes: u64,
}

#[cfg(test)]
struct Interval {
    key: StorageKey,
    start: usize,
    end: usize,
    bytes: u64,
    exported: bool,
}

#[cfg(test)]
impl Interval {
    /// Whether the two buffers are live at a common step. Endpoints count:
    /// a node's inputs end and its results start at the same step, so they
    /// never alias.
    fn overlaps(&self, other: &Interval) -> bool {
        self.start <= other.end && other.start <= self.end
    }

    /// Whether `other` lives only while this one does.
    fn contains(&self, other: &Interval) -> bool {
        self.start <= other.start && other.end <= self.end
    }

    /// Whether the two overlap and neither contains the other: they cannot
    /// lie on one end of a two-ended stack.
    fn crosses(&self, other: &Interval) -> bool {
        self.overlaps(other) && !self.contains(other) && !other.contains(self)
    }

    /// Whether it is a node's scratch: call-private, which lives at its node
    /// alone, or a paired operand, which lives from its producer to its
    /// consumer (`operand_pairs`). Either lies between the ends.
    fn scratch(&self) -> bool {
        matches!(self.key, StorageKey::NodeScratch(..))
    }
}

/// How a workspace or output interval obtains its byte length from a slice's
/// port, result, and scratch byte sizes.
#[derive(Clone, Copy, Debug)]
enum BufferSource {
    Port(usize),
    Result(usize, usize),
    Scratch(usize, usize),
    PairedScratch {
        producer: usize,
        emitted: usize,
        consumer: usize,
        formed: usize,
    },
}

impl BufferSource {
    fn bytes(
        self,
        port_byte: impl Fn(usize) -> u64,
        result_byte: impl Fn(usize, usize) -> Option<u64>,
        scratch_byte: impl Fn(usize, usize) -> u64,
    ) -> u64 {
        match self {
            Self::Port(ordinal) => port_byte(ordinal).max(1),
            Self::Result(node, ordinal) => result_byte(node, ordinal)
                .expect("tensor result has byte length")
                .max(1),
            Self::Scratch(node, ordinal) => scratch_byte(node, ordinal),
            Self::PairedScratch {
                producer,
                emitted,
                consumer,
                formed,
            } => scratch_byte(producer, emitted).max(scratch_byte(consumer, formed)),
        }
    }
}

struct StructuralInterval {
    source: BufferSource,
    start: usize,
    end: usize,
    exported: bool,
}

impl StructuralInterval {
    fn scratch(&self) -> bool {
        matches!(
            self.source,
            BufferSource::Scratch(..) | BufferSource::PairedScratch { .. }
        )
    }
}

/// Lifetime-only storage classification shared by [`CompiledStoragePlan`] and
/// [`plan_storage_with_links`]. Every decision here depends strictly on graph
/// topology, ports, exports, stores, shares, and operand links — never on
/// slice byte lengths.
struct StorageTopology {
    upload_ports: Vec<usize>,
    sync_scratch: Vec<(usize, usize)>,
    carried_scratch: Vec<(usize, usize, u32)>,
    intervals: Vec<StructuralInterval>,
}

impl StorageTopology {
    fn of(
        ports: &[StoragePort],
        nodes: &[StorageNode<'_>],
        links: &OperandLinks,
        exports: &[WorkflowResultRef],
        stores: &[(WorkflowResultRef, u32)],
    ) -> Self {
        let stored = |node: usize, ordinal: usize| {
            stores
                .iter()
                .find(|&&(result, _)| {
                    result.node as usize == node && result.result as usize == ordinal
                })
                .map(|&(_, port)| port as usize)
        };
        let mut last_use = nodes
            .iter()
            .enumerate()
            .map(|(node, spec)| vec![node * 2 + 1; spec.results.len()])
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
                        last_use[reference.node as usize][reference.result as usize] =
                            consumer * 2 + 1;
                    }
                }
            }
        }
        let mut upload_ports = Vec::new();
        let mut intervals = Vec::new();
        for (ordinal, port) in ports.iter().enumerate() {
            if !port.local {
                continue;
            }
            if port.owned_input {
                upload_ports.push(ordinal);
                continue;
            }
            let exported = exports
                .iter()
                .any(|item| item.node == INPUT_NODE && item.result as usize == ordinal);
            intervals.push(StructuralInterval {
                source: BufferSource::Port(ordinal),
                start: if port.prewritten {
                    0
                } else {
                    port_first[ordinal].unwrap_or(0)
                },
                end: port_last[ordinal],
                exported,
            });
        }
        // A result another shares (`NativeGraphDraft::share`) lives from the
        // first of its writers to its last read and the last writer.
        let sharers = |node: usize, ordinal: usize| {
            nodes.iter().enumerate().filter_map(move |(sharer, planned)| {
                planned
                    .share
                    .filter(|(_, primary)| {
                        primary.node as usize == node && primary.result as usize == ordinal
                    })
                    .map(|_| sharer)
            })
        };
        for (node, spec) in nodes.iter().enumerate() {
            for (ordinal, &bytes) in spec.results.iter().enumerate() {
                if bytes.is_none() {
                    continue;
                }
                // A stored result lives in its port's tensor, a shared one in
                // the storage of the result it shares (below).
                if stored(node, ordinal).is_some()
                    || spec
                        .share
                        .is_some_and(|(shared, _)| shared as usize == ordinal)
                {
                    continue;
                }
                let exported = exports
                    .iter()
                    .any(|item| item.node as usize == node && item.result as usize == ordinal);
                let writers = std::iter::once(node).chain(sharers(node, ordinal));
                let (first, last) = writers.fold((node, node), |(first, last), writer| {
                    (first.min(writer), last.max(writer))
                });
                intervals.push(StructuralInterval {
                    source: BufferSource::Result(node, ordinal),
                    start: first * 2 + 1,
                    end: last_use[node][ordinal].max(last * 2 + 1),
                    exported,
                });
            }
        }
        let mut sync_scratch = Vec::new();
        let mut carried_scratch = Vec::new();
        let pairs = &links.pairs;
        for (node, planned) in nodes.iter().enumerate() {
            for (ordinal, need) in planned.scratch.iter().enumerate() {
                if let Some(carried) = links.carried(node, ordinal) {
                    carried_scratch.push((node, ordinal, carried.port));
                    continue;
                }
                if pairs
                    .iter()
                    .any(|pair| pair.consumer == node && pair.formed == ordinal)
                {
                    continue;
                }
                if let Some(pair) = pairs
                    .iter()
                    .find(|pair| pair.producer == node && pair.emitted == ordinal)
                {
                    intervals.push(StructuralInterval {
                        source: BufferSource::PairedScratch {
                            producer: node,
                            emitted: ordinal,
                            consumer: pair.consumer,
                            formed: pair.formed,
                        },
                        start: node * 2,
                        end: pair.consumer * 2 + 1,
                        exported: false,
                    });
                    continue;
                }
                if need.sync {
                    sync_scratch.push((node, ordinal));
                    continue;
                }
                intervals.push(StructuralInterval {
                    source: BufferSource::Scratch(node, ordinal),
                    start: node * 2,
                    end: node * 2 + 1,
                    exported: false,
                });
            }
        }
        intervals.sort_by_key(|interval| interval.start);
        Self {
            upload_ports,
            sync_scratch,
            carried_scratch,
            intervals,
        }
    }
}

/// Precompiled lifetime analysis and R_b placement structure for one graph
/// topology, reusable across slices with varying tensor and scratch byte
/// lengths ([`Self::storage_bytes`]).
#[derive(Clone, Debug)]
pub struct CompiledStoragePlan {
    upload_ports: Vec<usize>,
    sync_scratch: Vec<(usize, usize)>,
    output_sources: Vec<BufferSource>,
    workspace_sources: Vec<BufferSource>,
    placement: seismic_memory::placement::CompiledPlacement,
}

impl CompiledStoragePlan {
    /// Precompile lifetime analysis, region assignments, and R_b colouring
    /// once from the graph's structural topology.
    pub fn of(
        ports: &[StoragePort],
        nodes: &[StorageNode<'_>],
        links: &OperandLinks,
        exports: &[WorkflowResultRef],
        stores: &[(WorkflowResultRef, u32)],
    ) -> Self {
        let topology = StorageTopology::of(ports, nodes, links, exports, stores);
        let mut output_sources = Vec::new();
        let mut workspace_sources = Vec::new();
        let mut spans = Vec::new();
        let mut scratch = Vec::new();
        for interval in topology.intervals {
            if interval.exported {
                output_sources.push(interval.source);
            } else {
                scratch.push(interval.scratch());
                spans.push(seismic_memory::placement::Span {
                    start: interval.start,
                    end: interval.end,
                });
                workspace_sources.push(interval.source);
            }
        }
        let placement = seismic_memory::placement::CompiledPlacement::of(&spans, &scratch);
        Self {
            upload_ports: topology.upload_ports,
            sync_scratch: topology.sync_scratch,
            output_sources,
            workspace_sources,
            placement,
        }
    }

    /// Compute exact region byte sizes for one slice's byte lengths,
    /// bit-for-bit identical to [`plan_storage_with_links`].
    pub fn storage_bytes(
        &self,
        port_bytes: &[u64],
        result_bytes: &[Vec<Option<u64>>],
        scratch_needs: &[Vec<ScratchNeed>],
    ) -> StorageSizes {
        let mut upload = 0u64;
        for &ordinal in &self.upload_ports {
            let offset = align_up(upload, BUFFER_ALIGNMENT);
            upload = offset
                .checked_add(port_bytes[ordinal].max(1))
                .expect("native graph upload bytes overflow");
        }

        let mut sync = 0u64;
        for &(node, ordinal) in &self.sync_scratch {
            let offset = sync.next_multiple_of(BUFFER_ALIGNMENT);
            sync = offset + scratch_needs[node][ordinal].bytes.max(MINIMUM_SCRATCH_BYTES);
        }

        let port_byte = |ordinal: usize| port_bytes[ordinal];
        let result_byte = |node: usize, ordinal: usize| result_bytes[node][ordinal];
        let scratch_byte = |node: usize, ordinal: usize| scratch_needs[node][ordinal].bytes;

        let mut output = 0u64;
        for &source in &self.output_sources {
            let offset = align_up(output, BUFFER_ALIGNMENT);
            output = offset
                .checked_add(source.bytes(port_byte, result_byte, scratch_byte))
                .expect("native graph output bytes overflow");
        }

        let raw_bytes: Vec<u64> = self
            .workspace_sources
            .iter()
            .map(|&source| source.bytes(port_byte, result_byte, scratch_byte))
            .collect();
        let padded_bytes: Vec<u64> = raw_bytes
            .iter()
            .map(|&bytes| seismic_memory::placement::padded(bytes))
            .collect();
        let workspace = self.placement.arena_bytes(&padded_bytes);
        let workspace_floor = self.placement.liveness_floor(&raw_bytes);

        StorageSizes {
            workspace,
            workspace_floor,
            sync,
            output,
            upload,
        }
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
#[cfg(test)]
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

/// Place `intervals` in one workspace arena by R_b
/// ([`seismic_memory::placement::place`]).
#[cfg(test)]
fn place(intervals: &[&Interval]) -> seismic_memory::placement::Placed {
    let buffers = intervals
        .iter()
        .map(|interval| seismic_memory::placement::Buffer {
            span: seismic_memory::placement::Span { start: interval.start, end: interval.end },
            bytes: interval.bytes,
            scratch: interval.scratch(),
        })
        .collect::<Vec<_>>();
    seismic_memory::placement::place(&buffers)
}

/// Place graph storage from plain descriptors with pre-resolved operand
/// `links` (whose `charge_node_emissions` has already been applied to each
/// node's `scratch`). Single owner of interval construction and R_b placement
/// shared by `NativeGraph` and direct graph sizing.
pub fn plan_storage_with_links(
    ports: &[StoragePort],
    nodes: &[StorageNode<'_>],
    links: &OperandLinks,
    exports: &[WorkflowResultRef],
    stores: &[(WorkflowResultRef, u32)],
) -> PlannedStorage {
    let topology = StorageTopology::of(ports, nodes, links, exports, stores);

    let mut upload_bytes = 0u64;
    let mut port_placements = vec![None; ports.len()];
    for &ordinal in &topology.upload_ports {
        let offset = align_up(upload_bytes, BUFFER_ALIGNMENT);
        upload_bytes = offset
            .checked_add(ports[ordinal].byte_len.max(1))
            .expect("native graph upload bytes overflow");
        port_placements[ordinal] = Some(StoragePlacement::Upload(offset));
    }

    // Sync scratch lives in the slot's sync region, each node's buffers
    // disjoint from every other node's, so the region is as large as they
    // need together.
    let mut sync_bytes = 0u64;
    let mut node_scratch = nodes
        .iter()
        .map(|node| vec![None; node.scratch.len()])
        .collect::<Vec<_>>();
    for &(node, ordinal, port) in &topology.carried_scratch {
        node_scratch[node][ordinal] = Some(StoragePlacement::Port(port));
    }
    for &(node, ordinal) in &topology.sync_scratch {
        // At least the minimum, so a region a buffer lies in exists.
        let offset = sync_bytes.next_multiple_of(BUFFER_ALIGNMENT);
        sync_bytes = offset + nodes[node].scratch[ordinal].bytes.max(MINIMUM_SCRATCH_BYTES);
        node_scratch[node][ordinal] = Some(StoragePlacement::Sync(offset));
    }

    let port_byte = |ordinal: usize| ports[ordinal].byte_len;
    let result_byte = |node: usize, ordinal: usize| nodes[node].results[ordinal];
    let scratch_byte = |node: usize, ordinal: usize| nodes[node].scratch[ordinal].bytes;

    let mut workspace_buffers = Vec::new();
    let mut workspace_spans = Vec::new();
    let mut workspace_raw = Vec::new();
    for interval in &topology.intervals {
        if !interval.exported {
            let bytes = interval.source.bytes(port_byte, result_byte, scratch_byte);
            let span = seismic_memory::placement::Span {
                start: interval.start,
                end: interval.end,
            };
            workspace_spans.push(span);
            workspace_raw.push(bytes);
            workspace_buffers.push(seismic_memory::placement::Buffer {
                span,
                bytes,
                scratch: interval.scratch(),
            });
        }
    }
    let seismic_memory::placement::Placed {
        offsets: scratch_offsets,
        bytes: scratch_bytes,
    } = seismic_memory::placement::place(&workspace_buffers);
    let scratch_floor_bytes =
        seismic_memory::placement::liveness_floor(&workspace_spans, &workspace_raw);

    let mut scratch_offsets = scratch_offsets.into_iter();
    let mut output_bytes = 0u64;
    let mut placements = nodes
        .iter()
        .map(|spec| vec![StoragePlacement::Scalar; spec.results.len()])
        .collect::<Vec<_>>();
    for interval in &topology.intervals {
        let placement = if interval.exported {
            let bytes = interval.source.bytes(port_byte, result_byte, scratch_byte);
            let offset = align_up(output_bytes, BUFFER_ALIGNMENT);
            output_bytes = offset
                .checked_add(bytes)
                .expect("native graph output bytes overflow");
            StoragePlacement::Export(offset)
        } else {
            StoragePlacement::Scratch(
                scratch_offsets
                    .next()
                    .expect("every scratch interval is placed"),
            )
        };
        match (interval.source, placement) {
            (BufferSource::Port(ordinal), _) => port_placements[ordinal] = Some(placement),
            (BufferSource::Result(node, ordinal), _) => placements[node][ordinal] = placement,
            (
                BufferSource::Scratch(node, ordinal)
                | BufferSource::PairedScratch {
                    producer: node,
                    emitted: ordinal,
                    ..
                },
                placement @ StoragePlacement::Scratch(_),
            ) => {
                node_scratch[node][ordinal] = Some(placement);
            }
            (BufferSource::Scratch(..) | BufferSource::PairedScratch { .. }, _) => {
                unreachable!("node scratch is never exported")
            }
        }
    }
    for pair in &links.pairs {
        node_scratch[pair.consumer][pair.formed] = node_scratch[pair.producer][pair.emitted];
    }
    let node_scratch = node_scratch
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|placement| placement.expect("every scratch buffer is placed"))
                .collect()
        })
        .collect();
    for &(result, port) in stores {
        placements[result.node as usize][result.result as usize] = StoragePlacement::Port(port);
    }
    for (node, planned) in nodes.iter().enumerate() {
        if let Some((ordinal, primary)) = planned.share {
            placements[node][ordinal as usize] =
                placements[primary.node as usize][primary.result as usize];
        }
    }
    PlannedStorage {
        port_placements,
        placements,
        node_scratch,
        workspace: scratch_bytes,
        workspace_floor: scratch_floor_bytes,
        sync: sync_bytes,
        output: output_bytes,
        upload: upload_bytes,
    }
}

/// Resolve operand links, charge unemitted scratch buffers, and place graph
/// storage from plain descriptors in one step.
pub fn plan_storage_sizes(
    ports: &[StoragePort],
    nodes: &[StorageNode<'_>],
    carries: &[OperandCarrySpec],
    exports: &[WorkflowResultRef],
    stores: &[(WorkflowResultRef, u32)],
) -> Result<PlannedStorage, WorkflowError> {
    let operand_nodes: Vec<OperandNode<'_>> = nodes
        .iter()
        .map(|node| OperandNode {
            args: node.args,
            scratch: node.scratch,
            activation: node.activation,
        })
        .collect();
    let links = resolve_operand_links(&operand_nodes, carries)?;
    let mut charged_scratch: Vec<Vec<ScratchNeed>> =
        nodes.iter().map(|node| node.scratch.to_vec()).collect();
    for (node, scratch) in charged_scratch.iter_mut().enumerate() {
        links.charge_node_emissions(node, scratch);
    }
    let charged_nodes: Vec<StorageNode<'_>> = nodes
        .iter()
        .zip(&charged_scratch)
        .map(|(node, scratch)| StorageNode {
            scratch,
            ..*node
        })
        .collect();
    Ok(plan_storage_with_links(ports, &charged_nodes, &links, exports, stores))
}

/// Place graph storage. Results, graph locals and node scratch share one
/// arena by lifetime (a node's scratch is live across its inputs and
/// results, so it never aliases them), placed by [`place`]; node `sync`
/// scratch, which must be zero when a launch starts, gets the slot's own
/// sync region, each buffer disjoint from every other; exports get their
/// own arena; inputs the host writes go to a
/// per-submission upload region so a later host write cannot race an
/// earlier submission still reading them. Every buffer starts at a
/// [`BUFFER_ALIGNMENT`] offset of its region.
fn plan_storage<N: AsRef<PlannedNode>, R: AsRef<[Option<NativeTensorSpec>]>>(
    ports: &[PortSpec],
    nodes: &[N],
    links: &OperandLinks,
    results: &[R],
    exports: &[WorkflowResultRef],
    stores: &[Store],
) -> StoragePlan {
    let storage_ports: Vec<StoragePort> = ports
        .iter()
        .map(|port| StoragePort {
            byte_len: port.byte_len,
            local: port.local,
            owned_input: port.owned_input,
            prewritten: port.prewritten,
        })
        .collect();
    let result_bytes: Vec<Vec<Option<u64>>> = results
        .iter()
        .map(|row| row.as_ref().iter().map(|spec| spec.as_ref().map(|s| s.byte_len)).collect())
        .collect();
    let storage_nodes: Vec<StorageNode<'_>> = nodes
        .iter()
        .zip(&result_bytes)
        .map(|(node, results)| {
            let node = node.as_ref();
            StorageNode {
                args: &node.args,
                results,
                scratch: &node.scratch,
                activation: node.activation,
                share: node.share,
            }
        })
        .collect();
    let storage_stores: Vec<(WorkflowResultRef, u32)> = stores
        .iter()
        .map(|store| (store.result, store.port))
        .collect();
    let planned = plan_storage_with_links(
        &storage_ports,
        &storage_nodes,
        links,
        exports,
        &storage_stores,
    );
    StoragePlan {
        port_placements: planned.port_placements,
        placements: planned.placements,
        node_scratch: planned.node_scratch,
        scratch_bytes: planned.workspace,
        scratch_floor_bytes: planned.workspace_floor,
        sync_bytes: planned.sync,
        output_bytes: planned.output,
        upload_bytes: planned.upload,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct StaticSite {
    region: Region,
    next_external: Option<u32>,
}

#[derive(PartialEq, Eq)]
struct StaticNode {
    sites: Box<[StaticSite]>,
    representations: Box<[&'static str]>,
    exclusive: Option<ExternalPairs>,
    site_base: usize,
}

#[derive(Clone, PartialEq, Eq)]
struct PortMeta {
    representation: RepresentationId,
    local: bool,
    owned_input: bool,
    prewritten: bool,
    extents: std::ops::Range<usize>,
}

#[derive(PartialEq, Eq)]
struct ResultMeta {
    nodes: Box<[std::ops::Range<usize>]>,
    entries: Box<[Option<(RepresentationId, std::ops::Range<usize>)>]>,
}

/// Static structure shared via `Arc` across every `NativeGraphPlan` sealed
/// into the same `NativeGraphLayout`.
#[derive(PartialEq, Eq)]
struct SealedTopology {
    ports: Box<[PortMeta]>,
    nodes: Box<[StaticNode]>,
    results: ResultMeta,
    exports: Box<[(u32, u32)]>,
    port_placements: Box<[Option<Placement>]>,
    placements: Box<[Placement]>,
    external_writes: Box<[bool]>,
    upload_written: bool,
    exclusive: Option<u32>,
    timelines: Box<[(usize, TimelineNode)]>,
    marks: Box<[usize]>,
    total_ranks: usize,
}

const SHARED_TOPOLOGY_BYTES: usize =
    std::mem::size_of::<SealedTopology>() + 2 * std::mem::size_of::<usize>();

struct FlatPlanData {
    kernels: Box<[Arc<NativePrepared>]>,
    rounds: Box<[u64]>,
    site_bytes: Box<[u64]>,
    site_offsets: Box<[u64]>,
    site_records: Box<[SiteRecords]>,
    port_uses: Box<[PortUse]>,
    port_byte_lens: Box<[u64]>,
    result_byte_lens: Box<[u64]>,
    extents_and_strides: Box<[u64]>,
    words: Box<[u64]>,
    word_ends: Box<[u32]>,
    launches: Box<[Option<LaunchGeometry>]>,
    launch_ends: Box<[u32]>,
    conditions: Box<[LaunchCondition]>,
    condition_ends: Box<[u32]>,
    workspace_records: u64,
}

impl SealedTopology {
    fn pack(
        ports: Vec<PortSpec>,
        executable: Executable,
        results: Vec<Vec<Option<NativeTensorSpec>>>,
        exports: Vec<WorkflowResultRef>,
        port_placements: Vec<Option<Placement>>,
        placements: Vec<Vec<Placement>>,
    ) -> (Self, FlatPlanData) {
        let mut extents = Vec::new();
        let mut strides = Vec::new();
        let mut port_meta = Vec::with_capacity(ports.len());
        let mut port_byte_lens = Vec::with_capacity(ports.len());
        for port in ports {
            let start = extents.len();
            extents.extend_from_slice(&port.extents);
            strides.extend_from_slice(&port.strides);
            let end = extents.len();
            port_byte_lens.push(port.byte_len);
            port_meta.push(PortMeta {
                representation: port.representation,
                local: port.local,
                owned_input: port.owned_input,
                prewritten: port.prewritten,
                extents: start..end,
            });
        }

        let mut result_nodes = Vec::with_capacity(results.len());
        let mut result_entries = Vec::new();
        let mut result_byte_lens = Vec::new();
        for row in results {
            let row_start = result_entries.len();
            for spec in row {
                match spec {
                    Some(spec) => {
                        let start = extents.len();
                        extents.extend_from_slice(&spec.extents);
                        strides.extend_from_slice(&spec.strides);
                        let end = extents.len();
                        result_byte_lens.push(spec.byte_len);
                        result_entries.push(Some((spec.representation, start..end)));
                    }
                    None => {
                        result_byte_lens.push(0);
                        result_entries.push(None);
                    }
                }
            }
            result_nodes.push(row_start..result_entries.len());
        }
        let total_ranks = extents.len();
        extents.extend_from_slice(&strides);

        let mut kernels = Vec::with_capacity(executable.nodes.len());
        let mut rounds = Vec::with_capacity(executable.nodes.len());
        let mut site_bytes = Vec::new();
        let mut site_offsets = Vec::new();
        let mut site_records = Vec::new();
        let mut words = Vec::new();
        let mut word_ends = Vec::with_capacity(executable.nodes.len());
        let mut launches = Vec::new();
        let mut launch_ends = Vec::with_capacity(executable.nodes.len());
        let mut conditions = Vec::new();
        let mut condition_ends = Vec::new();
        let mut static_nodes = Vec::with_capacity(executable.nodes.len());
        for node in executable.nodes {
            kernels.push(node.kernel);
            rounds.push(node.launches.rounds);
            let site_base = site_bytes.len();
            let mut static_sites = Vec::with_capacity(node.sites.len());
            for site in node.sites {
                site_bytes.push(site.bytes);
                site_offsets.push(site.offset);
                site_records.push(site.records);
                static_sites.push(StaticSite {
                    region: site.region,
                    next_external: site.next_external,
                });
            }
            words.extend_from_slice(&node.words);
            word_ends.push(u32::try_from(words.len()).expect("plan words fit u32"));
            launches.extend(node.launches.geometry);
            launch_ends.push(u32::try_from(launches.len()).expect("plan launches fit u32"));
            for launch_conditions in node.conditions {
                conditions.extend(launch_conditions);
                condition_ends.push(u32::try_from(conditions.len()).expect("plan conditions fit u32"));
            }
            static_nodes.push(StaticNode {
                sites: static_sites.into_boxed_slice(),
                representations: node.representations.into_boxed_slice(),
                exclusive: node.exclusive,
                site_base,
            });
        }

        let mut flat_placements = Vec::with_capacity(result_entries.len());
        for row in placements {
            flat_placements.extend(row);
        }
        debug_assert_eq!(flat_placements.len(), result_entries.len());

        let topology = Self {
            ports: port_meta.into_boxed_slice(),
            nodes: static_nodes.into_boxed_slice(),
            results: ResultMeta {
                nodes: result_nodes.into_boxed_slice(),
                entries: result_entries.into_boxed_slice(),
            },
            exports: exports
                .into_iter()
                .map(|reference| (reference.node, reference.result))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            port_placements: port_placements.into_boxed_slice(),
            placements: flat_placements.into_boxed_slice(),
            external_writes: executable.external_writes.into_boxed_slice(),
            upload_written: executable.upload_written,
            exclusive: executable.exclusive,
            timelines: executable.timelines.into_boxed_slice(),
            marks: executable.marks.into_boxed_slice(),
            total_ranks,
        };
        let flat = FlatPlanData {
            kernels: kernels.into_boxed_slice(),
            rounds: rounds.into_boxed_slice(),
            site_bytes: site_bytes.into_boxed_slice(),
            site_offsets: site_offsets.into_boxed_slice(),
            site_records: site_records.into_boxed_slice(),
            port_uses: executable.port_uses.into_boxed_slice(),
            port_byte_lens: port_byte_lens.into_boxed_slice(),
            result_byte_lens: result_byte_lens.into_boxed_slice(),
            extents_and_strides: extents.into_boxed_slice(),
            words: words.into_boxed_slice(),
            word_ends: word_ends.into_boxed_slice(),
            launches: launches.into_boxed_slice(),
            launch_ends: launch_ends.into_boxed_slice(),
            conditions: conditions.into_boxed_slice(),
            condition_ends: condition_ends.into_boxed_slice(),
            workspace_records: executable.workspace_records,
        };
        (topology, flat)
    }

    fn bounds(&self) -> impl Iterator<Item = usize> + '_ {
        self.timelines.iter().map(|&(position, _)| position)
    }

    fn host_heap(&self) -> HostHeap {
        topology_heap_from_counts(
            TopologyHeapCounts {
                ports: self.ports.len(),
                nodes: self.nodes.len(),
                result_rows: self.results.nodes.len(),
                total_results: self.results.entries.len(),
                exports: self.exports.len(),
                timelines: self.timelines.len(),
                marks: self.marks.len(),
            },
            self.nodes
                .iter()
                .map(|node| (node.sites.len(), node.representations.len())),
        )
    }
}

pub struct NativeGraphPlan {
    identity: u64,
    device: Arc<DeviceInner>,
    topology: Arc<SealedTopology>,
    kernels: Box<[Arc<NativePrepared>]>,
    rounds: Box<[u64]>,
    site_bytes: Box<[u64]>,
    site_offsets: Box<[u64]>,
    site_records: Box<[SiteRecords]>,
    port_uses: Box<[PortUse]>,
    port_byte_lens: Box<[u64]>,
    result_byte_lens: Box<[u64]>,
    extents_and_strides: Box<[u64]>,
    words: Box<[u64]>,
    word_ends: Box<[u32]>,
    launches: Box<[Option<LaunchGeometry>]>,
    launch_ends: Box<[u32]>,
    conditions: Box<[LaunchCondition]>,
    condition_ends: Box<[u32]>,
    workspace_records: u64,
    scratch_bytes: u64,
    scratch_floor_bytes: u64,
    sync_bytes: u64,
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

impl Drop for NativeGraphPlan {
    fn drop(&mut self) {
        let backend = super::backend_name(&self.device.kind);
        self.device.native.unregister_plan_key(backend, self.shape.keyed(backend));
    }
}

impl NativeGraphPlan {
    /// The host heap this plan holds: itself in its shared allocation and
    /// its flat per-class slices. Its static topology is shared across all
    /// plans of one `NativeGraphLayout`.
    pub fn host_heap(&self) -> HostHeap {
        plan_heap_from_counts(PlanHeapCounts {
            nodes: self.kernels.len(),
            ports: self.port_byte_lens.len(),
            total_sites: self.site_bytes.len(),
            total_results: self.result_byte_lens.len(),
            extents_and_strides: self.extents_and_strides.len(),
            total_words: self.words.len(),
            total_launches: self.launches.len(),
            total_conditions: self.conditions.len(),
        })
    }

    fn node_words(&self, node: usize) -> &[u64] {
        let start = if node == 0 { 0 } else { self.word_ends[node - 1] as usize };
        let end = self.word_ends[node] as usize;
        &self.words[start..end]
    }

    fn node_launch_range(&self, node: usize) -> std::ops::Range<usize> {
        let start = if node == 0 { 0 } else { self.launch_ends[node - 1] as usize };
        let end = self.launch_ends[node] as usize;
        start..end
    }

    fn node_launches(&self, node: usize) -> &[Option<LaunchGeometry>] {
        &self.launches[self.node_launch_range(node)]
    }

    fn launch_conditions(&self, launch: usize) -> &[LaunchCondition] {
        let start = if launch == 0 { 0 } else { self.condition_ends[launch - 1] as usize };
        let end = self.condition_ends[launch] as usize;
        &self.conditions[start..end]
    }

    fn extents(&self, range: &std::ops::Range<usize>) -> &[u64] {
        &self.extents_and_strides[range.clone()]
    }

    fn strides(&self, range: &std::ops::Range<usize>) -> &[u64] {
        let off = self.topology.total_ranks;
        &self.extents_and_strides[range.start + off..range.end + off]
    }

    fn buffer_name(&self, node_idx: usize, buffer: usize) -> String {
        let kernel = &self.kernels[node_idx];
        let (schema, implementation) = (kernel.schema(), kernel.implementation());
        let parameters = schema
            .parameters()
            .iter()
            .filter(|parameter| matches!(parameter.kind, ParameterKind::Tensor { .. }))
            .count();
        let scratch = self.topology.nodes[node_idx].sites.len() - implementation.scratch.len();
        if buffer < parameters {
            return super::abi::tensor_buffer_name(schema, buffer).to_owned();
        }
        if buffer >= scratch {
            return format!("scratch {}", implementation.scratch[buffer - scratch].name);
        }
        let (ordinal, _) = schema
            .results()
            .iter()
            .enumerate()
            .filter(|(_, result)| matches!(result.kind, ResultKind::Tensor { .. }))
            .nth(buffer - parameters)
            .expect("a result buffer is a tensor result's");
        format!("result {ordinal}")
    }

    /// Whether `self` and `other` share the same immutable [`SealedTopology`]
    /// allocation (true for every plan sealed from one [`NativeGraphLayout`]).
    pub fn shares_topology(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.topology, &other.topology)
    }

    /// The host heap of the shared immutable [`SealedTopology`] referenced by
    /// this plan (held once per [`NativeGraphLayout`], or once per standalone
    /// plan).
    pub fn topology_heap(&self) -> HostHeap {
        self.topology.host_heap()
    }

    /// Address identity of the shared [`SealedTopology`] allocation, equal
    /// across every plan that shares it (`Self::shares_topology`).
    pub fn topology_identity(&self) -> usize {
        Arc::as_ptr(&self.topology) as usize
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
    /// The bytes of a slot's own sync region a run of the plan uses.
    pub fn sync_bytes(&self) -> u64 {
        self.sync_bytes
    }
    pub fn output_bytes(&self) -> u64 {
        self.output_bytes
    }
    pub fn slot_storage_bytes(&self) -> u64 {
        self.scratch_bytes + self.sync_bytes + self.output_bytes
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
    /// on a slot of their own ([`GraphPoint`]) allocated through `charges`;
    /// a plan with timeline nodes is refused.
    pub fn point(
        self: &Arc<Self>,
        charges: &Charges,
        bindings: NativeGraphBindings,
        timing: GraphTiming,
    ) -> Result<GraphPoint, CallError> {
        GraphPoint::new(self, charges, bindings, timing)
    }
    /// The device the plan runs on.
    pub(super) fn device(&self) -> &Arc<DeviceInner> {
        &self.device
    }
    /// Whether a run of the plan waits on or signals a timeline.
    pub(super) fn has_timelines(&self) -> bool {
        !self.topology.timelines.is_empty()
    }
    pub fn bindings(self: &Arc<Self>) -> NativeGraphBindings {
        NativeGraphBindings {
            plan: self.clone(),
            inputs: vec![None; self.topology.ports.len()],
            fixed: vec![false; self.topology.ports.len()],
            timelines: vec![None; self.topology.timelines.len()],
        }
    }
    /// Refuse bindings under which two buffers of one node, an exclusive
    /// pair ([`exclusive_pairs`]) in external ports' tensors, reach a byte
    /// in common ([`Reaching::meets`]): those of ports both statically bound
    /// (`fixed`, checked once in [`Self::bind_static`]), or else those of a
    /// port a run binds (checked at attach, when every external port is
    /// bound). Only the pairs chained at seal can
    /// ([`Executable::exclude`], which holds the rest of each node's buffers
    /// to the rule), and only those are walked, each in O(1), a bound slab
    /// view's reach read at most once.
    fn check_disjoint(&self, bindings: &NativeGraphBindings, fixed: bool) -> Result<(), CallError> {
        if self.topology.exclusive.is_none() {
            return Ok(());
        }
        let reaching: Vec<Option<Reaching<'_>>> = bindings
            .inputs
            .iter()
            .map(|bound| bound.as_ref().map(|bound| Reaching::new(&bound.tensor)))
            .collect();
        let tensor = |port: usize| reaching[port].as_ref().expect("every external port of a checked pair is bound");
        let mut cursor = self.topology.exclusive;
        while let Some(node_ordinal) = cursor {
            let node_idx = node_ordinal as usize;
            let node = &self.topology.nodes[node_idx];
            let pairs = node.exclusive.expect("a node of the exclusive chain holds its external pairs");
            let kernel = &self.kernels[node_idx];
            let after = |buffer: u32| {
                std::iter::successors(node.sites[buffer as usize].next_external, move |&next| {
                    node.sites[next as usize].next_external
                })
            };
            let first = pairs.first;
            for written in std::iter::once(first)
                .chain(after(first))
                .take_while(|&buffer| kernel.writes(buffer as usize))
            {
                let first_site = &node.sites[written as usize];
                let Region::External(first_port) = first_site.region else {
                    unreachable!("an external chain links external sites");
                };
                let first_port = first_port as usize;
                let first_idx = node.site_base + written as usize;
                let first_bytes = self.site_bytes[first_idx];
                let first_offset = self.site_offsets[first_idx];
                let first_range = first_offset..first_offset + first_bytes;
                for second in after(written) {
                    let second_site = &node.sites[second as usize];
                    let Region::External(second_port) = second_site.region else {
                        unreachable!("an external chain links external sites");
                    };
                    let second_port = second_port as usize;
                    let second_idx = node.site_base + second as usize;
                    let second_bytes = self.site_bytes[second_idx];
                    let second_offset = self.site_offsets[second_idx];
                    let second_range = second_offset..second_offset + second_bytes;
                    if (first_port == second_port && !meet(&first_range, &second_range))
                        || (bindings.fixed[first_port] && bindings.fixed[second_port]) != fixed
                    {
                        continue;
                    }
                    if tensor(first_port).meets(first_range.clone(), tensor(second_port), second_range) {
                        let (lo, hi) = (written.min(second) as usize, written.max(second) as usize);
                        return Err(CallError::Invocation(InvocationError::IllegalAliasing {
                            first: self.buffer_name(node_idx, lo),
                            second: self.buffer_name(node_idx, hi),
                        }));
                    }
                }
            }
            cursor = pairs.next;
        }
        Ok(())
    }

    /// Whether `tensor` may bind external port `port`: of its description,
    /// writable where a node writes it, and host-addressed where a launch
    /// condition reads a word of it (checked here, as the graph's bindings
    /// form, so a queued run reads it as it is).
    fn check_binding(&self, port: usize, tensor: &Arc<TensorInner>) -> Result<(), WorkflowError> {
        let meta = &self.topology.ports[port];
        let extents = self.extents(&meta.extents);
        let strides = self.strides(&meta.extents);
        let byte_len = self.port_byte_lens[port];
        let matches = Arc::ptr_eq(tensor.device(), &self.device)
            && tensor.representation() == meta.representation
            && tensor.extents() == extents
            && tensor.strides() == strides
            && tensor.byte_len() == byte_len;
        if !matches || (self.topology.external_writes[port] && tensor.allocation().storage().read_only()) {
            return Err(WorkflowError::NativePortBinding {
                port,
                expected: format!(
                    "{:?} extents {:?} strides {:?} {} bytes{}",
                    meta.representation,
                    extents,
                    strides,
                    byte_len,
                    if self.topology.external_writes[port] { ", written" } else { "" },
                ),
                bound: format!(
                    "{:?} extents {:?} strides {:?} {} bytes{}{}",
                    tensor.representation(),
                    tensor.extents(),
                    tensor.strides(),
                    tensor.byte_len(),
                    if tensor.allocation().storage().read_only() { ", read-only" } else { "" },
                    if Arc::ptr_eq(tensor.device(), &self.device) { "" } else { ", another device's" },
                ),
            });
        }
        if self.port_uses[port].condition && tensor.allocation().storage().host_address().is_none() {
            return Err(WorkflowError::UnaddressedCondition { parameter: format!("port {port}") });
        }
        Ok(())
    }

    /// Bind the ports that stay fixed across runs, each checked and its
    /// records formed through `charges` as it is bound
    /// ([`NativeGraphBindings::set`]). An exclusive pair of buffers on
    /// fixed ports whose bytes meet is refused here
    /// (`Self::check_disjoint`), so no run of the bound plan can alias it.
    pub fn bind_static(
        self: &Arc<Self>,
        charges: &Charges,
        fixed: &[(NativePort, Arc<TensorInner>)],
    ) -> Result<BoundNativeGraphPlan, CallError> {
        let mut bindings = self.bindings();
        for (port, tensor) in fixed {
            bindings
                .set(charges, *port, tensor.clone())
                .map_err(CallError::Workflow)?;
            bindings.fixed[port.reference.result as usize] = true;
        }
        self.check_disjoint(&bindings, true)?;
        Ok(BoundNativeGraphPlan {
            _plan: self.clone(),
            fixed: bindings,
        })
    }
    /// What one run's graphs hold when its layout's graph holds the
    /// launches `form` names (`super::LayoutForms`): on CUDA one graph
    /// (`graph_segments`) with a node for each launch the form holds, each
    /// mark it holds and the nodes of each timeline node inside it
    /// (`cuda::timeline_nodes`), exactly what `cuda::record_run` makes,
    /// none of a launched plan, whose runs launch call by call
    /// ([`NativeGraphDraft::seal_launched`]); elsewhere its recorded
    /// active-launch census ([`Self::recorded`]).
    fn run_graphs(&self, form: &[u64]) -> GraphCensus {
        if !self.graphed() {
            return GraphCensus::default();
        }
        match super::backend_name(&self.device.kind) {
            BackendName::Cuda => {
                let launches = GraphCensus {
                    launches: form.iter().map(|word| u64::from(word.count_ones())).sum::<u64>(),
                    ..GraphCensus::default()
                };
                recorded_census(
                    BackendName::Cuda,
                    self.topology.nodes.len(),
                    launches,
                    &self.topology.timelines,
                    &self.topology.marks,
                )
            }
            BackendName::Vulkan | BackendName::Metal | BackendName::Cpu => self.recorded,
        }
    }

    /// The span marks its graphs hold, each the node it precedes, in order
    /// ([`NativeGraphDraft::span_mark`]): those inside a graph, of a plan
    /// whose runs replay graphs. A mark at a run's edge, or at a timeline
    /// node where graphs divide there (all but CUDA's, `graph_segments`),
    /// falls between graphs, and costs a run not sampled nothing.
    fn held_marks(&self) -> Vec<usize> {
        if !self.graphed() {
            return Vec::new();
        }
        let nodes = self.topology.nodes.len();
        let divides = |mark: &usize| !self.holds_timelines() && self.topology.bounds().any(|bound| bound == *mark);
        let held = self.topology.marks.iter().copied().filter(|mark| (1..nodes).contains(mark) && !divides(mark));
        held.collect()
    }

    /// Whether its runs replay graphs: on CUDA all but a launched plan's,
    /// which launch call by call ([`NativeGraphDraft::seal_launched`]).
    fn graphed(&self) -> bool {
        !(self.launched && super::backend_name(&self.device.kind) == BackendName::Cuda)
    }

    /// Whether its run's graph holds the timeline nodes inside it, as nodes
    /// of its own (`cuda::timeline_nodes`), rather than dividing there
    /// (`graph_segments`): a graphed plan's on CUDA.
    pub(super) fn holds_timelines(&self) -> bool {
        self.graphed() && super::backend_name(&self.device.kind) == BackendName::Cuda
    }

    /// A slot's share of its device's graph budget ([`super::SlotGraphs`]):
    /// a launch of the plan's own slot is one run, of its layout's graph of
    /// the launches it makes, a family of its own.
    fn slot_hold(self: &Arc<Self>) -> Result<Arc<super::SlotGraphs>, TensorError> {
        let form: Arc<[u64]> = self.shape.structure.clone().into();
        let census = self.run_graphs(&form);
        let layout = super::LayoutGraph { form, plans: vec![self.clone()] };
        let layouts = super::LayoutForms::new([(self.shape.layout, layout)]);
        let graphs = LaunchGraphs::of([LaunchedGraph { graph: census, pinned_runs: 0, runs: 1 }], Vec::new())
            .expect("a plan's graph census fits u64");
        super::FamilyGraphs::new(&self.device, layouts, graphs).slot().map_err(TensorError::Execution)
    }

    /// A slot of this plan alone, holding a run's graphs in its device's
    /// graph budget ([`super::SlotGraphs`]). Its upload region is fixed, so
    /// host writes to its inputs wait for the slot's previous submission; a
    /// family slot instead takes the smallest of its idle upload regions
    /// that holds a run's input ([`idle_upload`]).
    ///
    /// Its storage is allocated through `charges`.
    ///
    /// # Panics
    /// Where `charges` are of another device than the plan's.
    pub fn new_slot(self: &Arc<Self>, charges: &Charges) -> Result<NativeGraphSlot, TensorError> {
        charges.of_device(&self.device);
        let workspace = zeroed_region(charges, self.scratch_bytes)?;
        let sync = zeroed_region(charges, self.sync_bytes)?;
        let upload = if self.upload_bytes == 0 {
            None
        } else {
            Some(charges.allocate_mapped(seismic_topology::Contract::Upload, self.upload_bytes, BUFFER_ALIGNMENT)?)
        };
        Ok(NativeGraphSlot {
            plan: self.clone(),
            workspace_records: workspace
                .as_ref()
                .map(|workspace| self.workspace_record_table(charges, workspace))
                .transpose()?
                .flatten(),
            workspace,
            sync,
            upload,
            hold: self.slot_hold()?,
        })
    }

    /// The table of the addresses of the records every record site in the
    /// workspace binds (`Site::table`), over `workspace`, the slot's: formed
    /// once with the slot, so no run allocates one, and charged with the
    /// workspace while it lives (`Allocation::hold_table`), through
    /// `charges`. None for a plan without such sites.
    fn workspace_record_table(
        &self,
        charges: &Charges,
        workspace: &Arc<Allocation>,
    ) -> Result<Option<Arc<Allocation>>, TensorError> {
        if self.workspace_records == 0 {
            return Ok(None);
        }
        let offsets = self
            .topology
            .nodes
            .iter()
            .flat_map(|node| {
                node.sites.iter().enumerate().map(move |(i, _)| {
                    let idx = node.site_base + i;
                    (self.site_offsets[idx], self.site_bytes[idx], self.site_records[idx])
                })
            })
            .filter_map(|(offset, bytes, records)| match records {
                SiteRecords::Workspace { records, .. } => Some((offset, bytes, records.get())),
                SiteRecords::None | SiteRecords::Port(_) => None,
            })
            .flat_map(|(offset, bytes, records)| {
                let record = bytes / records;
                (0..records).map(move |index| offset + index * record)
            });
        let table = crate::api::tensor::record_table(charges, workspace, offsets)?;
        workspace.hold_table(&table);
        Ok(Some(table))
    }

    /// Exported outputs are a separate owner so their lifetime can outlive
    /// scratch-slot reuse without retaining the scratch allocation. Their
    /// storage is allocated through `charges`.
    ///
    /// # Panics
    /// Where `charges` are of another device than the plan's.
    pub fn new_outputs(self: &Arc<Self>, charges: &Charges) -> Result<NativeGraphOutputs, TensorError> {
        Ok(self.outputs_from_arena(self.output_arena(charges)?, None))
    }

    /// A run's output arena, through `charges`; none for a plan that
    /// exports nothing.
    pub(super) fn output_arena(&self, charges: &Charges) -> Result<Option<Arc<Allocation>>, TensorError> {
        charges.of_device(&self.device);
        if self.output_bytes == 0 {
            return Ok(None);
        }
        Ok(Some(charges.allocate(self.output_bytes, BUFFER_ALIGNMENT)?))
    }

    fn port_view(
        &self,
        ordinal: usize,
        allocation: &Arc<Allocation>,
        offset: u64,
    ) -> Arc<TensorInner> {
        let port = &self.topology.ports[ordinal];
        Arc::new(TensorInner::new_view(
            self.device.clone(),
            allocation.clone(),
            offset,
            self.port_byte_lens[ordinal],
            port.representation,
            self.extents(&port.extents).to_vec(),
            self.strides(&port.extents).to_vec(),
        ))
    }

    pub(super) fn outputs_from_arena(
        self: &Arc<Self>,
        arena: Option<Arc<Allocation>>,
        family: Option<Arc<NativeGraphFamily>>,
    ) -> NativeGraphOutputs {
        let has_exports = self
            .topology
            .port_placements
            .iter()
            .any(|placement| matches!(placement, Some(Placement::Export(_))))
            || self
                .topology
                .placements
                .iter()
                .zip(&self.topology.results.entries)
                .any(|(placement, entry)| entry.is_some() && matches!(placement, Placement::Export(_)));
        let (locals, outputs) = if has_exports {
            let export_arena = arena.as_ref().expect("planned output arena absent");
            let locals = (0..self.topology.ports.len())
                .map(|ordinal| {
                    let Some(Placement::Export(offset)) = self.topology.port_placements[ordinal] else {
                        return None;
                    };
                    Some(self.port_view(ordinal, export_arena, offset))
                })
                .collect();
            let outputs = self
                .topology
                .results
                .nodes
                .iter()
                .map(|row_range| {
                    row_range
                        .clone()
                        .map(|flat_idx| {
                            let (representation, ext_range) =
                                self.topology.results.entries[flat_idx].as_ref()?;
                            let Placement::Export(offset) = self.topology.placements[flat_idx] else {
                                return None;
                            };
                            Some(Arc::new(TensorInner::new_view(
                                self.device.clone(),
                                export_arena.clone(),
                                offset,
                                self.result_byte_lens[flat_idx],
                                *representation,
                                self.extents(ext_range).to_vec(),
                                self.strides(ext_range).to_vec(),
                            )))
                        })
                        .collect()
                })
                .collect();
            (locals, outputs)
        } else {
            let locals = vec![None; self.topology.ports.len()];
            let outputs = self
                .topology
                .results
                .nodes
                .iter()
                .map(|row_range| vec![None; row_range.len()])
                .collect();
            (locals, outputs)
        };
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
        heap.hold_vec(&self.fixed.fixed);
        heap.hold_vec(&self.fixed.timelines);
        heap
    }
}

/// The host heap of the bindings of a plan of `ports` ports and
/// `timelines` timeline nodes (`NativeGraphPlan::bindings`).
pub fn bindings_heap(ports: usize, timelines: usize) -> HostHeap {
    use std::mem::size_of;
    let mut heap = HostHeap::default();
    heap.hold(ports, size_of::<Option<BoundPort>>());
    heap.hold(ports, size_of::<bool>());
    heap.hold(timelines, size_of::<Option<TimelineBinding>>());
    heap
}

/// A finite family of sealed exact-shape graphs sharing one reusable scratch
/// arena per execution slot. Activating a class borrows the family slot
/// exclusively; two classes cannot use the same arena concurrently.
pub struct NativeGraphFamily {
    plans: Vec<Arc<NativeGraphPlan>>,
    device: Arc<DeviceInner>,
    workspace_bytes: u64,
    /// The largest sync region of its plans: each slot's own.
    sync_bytes: u64,
    output_bytes: u64,
    upload_bytes: u64,
    /// Its layouts' graphs and its share of its device's graph budget,
    /// which its live slots hold ([`super::FamilyGraphs`]).
    graphs: Arc<super::FamilyGraphs>,
    /// Its layouts.
    layouts: usize,
    /// Each layout its decode launches run, with the most runs of it any
    /// one of them makes, graphed only ([`Self::record_decode`]).
    decode_layouts: Vec<(u64, usize)>,
    /// What each launch's runs replay, by launch, in the order given:
    /// each run's layout's graph (`GraphFamilyCensus`).
    launch_graphs: Vec<GraphCensus>,
}

/// What a family's graphs are made of, for a load's report: its layouts
/// and what each launch's runs replay.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphFamilyCensus {
    pub layouts: usize,
    pub launches: Vec<GraphCensus>,
}

impl NativeGraphFamily {
    /// A family of `plans` whose launch is one run of one of them, the
    /// first `decodes` those a decode step may make, kept live for each
    /// launch in flight ([`Self::record_decode`]).
    pub fn new(plans: &[Arc<NativeGraphPlan>], decodes: usize) -> Result<Self, WorkflowError> {
        let launches = plans.iter().map(|plan| vec![plan.clone()]).collect::<Vec<_>>();
        Self::with_launches(plans, &launches, decodes, &[])
    }

    /// A family of `plans` whose launches each make the runs of one of
    /// `launches`, by plan, a plan once for each of its runs, the first
    /// `decodes` the launches a decode step may make, whose graphs are kept
    /// live for each launch in flight ([`Self::record_decode`]) and which
    /// every launch in flight beside another may be. Its plans of one
    /// layout share one graph (`super::LayoutForms`): on CUDA a node for
    /// each launch any of them makes, each run enabling its own; elsewhere
    /// their own graphs. Every graph is recorded as a run first needs one,
    /// within what its live slots hold of the device's graph budget
    /// ([`Self::graphs`]). A chain of `k + 2` launches, all in one
    /// submission, further runs `chains[k]`, each plan its count of runs,
    /// beyond each launch's own runs: on a backend where one graph serves one run of its shape a
    /// submission (`BackendName::graphs_update_in_flight`), each a graph of
    /// its own; elsewhere the launches' graphs serve them.
    pub fn with_launches(
        plans: &[Arc<NativeGraphPlan>],
        launches: &[Vec<Arc<NativeGraphPlan>>],
        decodes: usize,
        chains: &[Vec<(Arc<NativeGraphPlan>, u64)>],
    ) -> Result<Self, WorkflowError> {
        let first = plans.first().ok_or(WorkflowError::Empty)?;
        assert!(decodes <= launches.len(), "a family's decode launches are among its launches");
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
        let mut layouts = Vec::<super::LayoutGraph>::new();
        for plan in plans {
            let index = *at.entry(plan.shape.layout).or_insert_with(|| {
                layouts.push(super::LayoutGraph { plans: Vec::new(), form: Arc::from([]) });
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
            // On CUDA a graph holds its run's timeline nodes, so they are
            // part of the form one graph of the layout serves.
            assert!(
                super::backend_name(&first.device.kind) != BackendName::Cuda
                    || graphs.plans.iter().all(|plan| plan.topology.timelines == graphs.plans[0].topology.timelines),
                "a layout's plans hold the same timeline nodes"
            );
            graphs.form = (0..calls)
                .map(|call| graphs.plans.iter().fold(0, |made, plan| made | plan.shape.structure[call]))
                .collect();
        }
        // Every plan runs in some launch, else the graphs its runs record
        // would lie outside what the family holds.
        let members = plans.iter().map(Arc::as_ptr).collect::<std::collections::HashSet<_>>();
        let launched = launches.iter().flatten().map(Arc::as_ptr).collect::<std::collections::HashSet<_>>();
        assert!(launched.is_subset(&members), "a family's launches run its plans");
        assert_eq!(launched.len(), members.len(), "every plan of a family runs in one of its launches");
        let graph_of = |plan: &Arc<NativeGraphPlan>| plan.run_graphs(&layouts[at[&plan.shape.layout]].form);
        let launch_graphs = launches
            .iter()
            .map(|runs| runs.iter().try_fold(GraphCensus::default(), |census, plan| census.plus(graph_of(plan))))
            .collect::<Option<Vec<_>>>()
            .expect("a family's graph census fits u64");
        // Of each layout, its largest plan's graph (every plan of a layout
        // runs in some launch), and the most runs of it any one decode
        // launch, and any one launch, makes.
        let mut graphed = vec![LaunchedGraph { graph: GraphCensus::default(), pinned_runs: 0, runs: 0 }; layouts.len()];
        for (index, runs) in launches.iter().enumerate() {
            let mut counts = vec![0u64; layouts.len()];
            for plan in runs.iter().filter(|plan| plan.graphed()) {
                let layout = at[&plan.shape.layout];
                counts[layout] += 1;
                graphed[layout].graph = graphed[layout].graph.most(plan.run_graphs(&layouts[layout].form));
            }
            for (graphed, count) in graphed.iter_mut().zip(counts) {
                graphed.runs = graphed.runs.max(count);
                if index < decodes {
                    graphed.pinned_runs = graphed.pinned_runs.max(count);
                }
            }
        }
        let chained = match super::backend_name(&first.device.kind).graphs_update_in_flight() {
            true => Vec::new(),
            false => chains
                .iter()
                .map(|runs| runs.iter().try_fold(GraphCensus::default(), |census, (plan, count)| census.plus(graph_of(plan).times(*count)?)))
                .collect::<Option<Vec<_>>>()
                .expect("a family's graph census fits u64"),
        };
        let decode_layouts = layouts
            .iter()
            .zip(&graphed)
            .filter(|(_, graphed)| graphed.pinned_runs > 0)
            .map(|(graphs, graphed)| (graphs.plans[0].shape.layout, graphed.pinned_runs as usize))
            .collect::<Vec<_>>();
        let graphs = LaunchGraphs::of(graphed, chained).expect("a family's graph census fits u64");
        let count = layouts.len();
        let layouts = super::LayoutForms::new(layouts.into_iter().map(|graph| (graph.plans[0].shape.layout, graph)));
        Ok(Self {
            graphs: super::FamilyGraphs::new(&first.device, layouts, graphs),
            layouts: count,
            decode_layouts,
            launch_graphs,
            plans: plans.to_vec(),
            device: first.device.clone(),
            workspace_bytes: plans
                .iter()
                .map(|plan| plan.scratch_bytes)
                .max()
                .unwrap_or(0),
            sync_bytes: plans.iter().map(|plan| plan.sync_bytes).max().unwrap_or(0),
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

    /// The bytes of each slot's own sync region, which every run of the
    /// slot uses and no other slot's does.
    pub fn sync_bytes(&self) -> u64 {
        self.sync_bytes
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

    /// What it pins for each launch in flight and the union of its
    /// launches' graphs, which its live slots, each a launch in flight,
    /// hold of its device's graph budget
    /// (`LaunchGraphs::held`).
    pub fn graphs(&self) -> LaunchGraphs {
        self.graphs.launches()
    }

    /// Whether [`Self::record_decode`] records its decode launches' graphs
    /// ahead of their first use: CUDA's only. Elsewhere (Vulkan, Metal) a
    /// graph is recorded at its shape's first use, a bank shrink's step
    /// included, as any new shape's.
    pub fn records_decode_ahead(&self) -> bool {
        self.cuda().is_some()
    }

    /// The CUDA device its graphs are recorded on, none elsewhere.
    fn cuda(&self) -> Option<&Arc<crate::backends::CudaOpened>> {
        match &self.device.kind {
            crate::backends::OpenedKind::Cuda(opened) => Some(opened),
            _ => None,
        }
    }

    /// Record, ahead of them, every graph its decode launches in flight
    /// replay that is not kept: on CUDA each layout's graphs for the most
    /// runs of it any of its decode launches makes, for each live slot (a
    /// launch in flight, as `LaunchGraphs::held` holds them), once the
    /// device's queued work has drained where one is to be recorded, and
    /// kept while those slots live, whatever other graphs a submission
    /// releases (`graph_replays::Pinned`), so a step of it records none, a
    /// bank shrink's step included; elsewhere a
    /// graph is recorded at its shape's first use with no wait, and this
    /// records nothing. Recording allocates nothing of the device's ledger
    /// (its word nodes name the device's placeholder word), and memory the
    /// driver refuses it is a capacity refusal.
    pub fn record_decode(&self) -> Result<(), TensorError> {
        let Some(opened) = self.cuda() else {
            return Ok(());
        };
        // The drain and the recordings under one guard, the submission
        // order lock, so nothing is queued between them (`cuda::record_run`).
        let _order = self.device.native.order.lock().expect("native submission order lock is never poisoned");
        let replays = &self.device.native.cuda_replays;
        for &(layout, count) in &self.decode_layouts {
            let graph = self.graphs.layout(layout);
            let key = over_placeholders(&graph, |_, runs| super::cuda::run_key(runs)).map_err(TensorError::Execution)?;
            let retained: Arc<dyn std::any::Any + Send + Sync> = Arc::new(graph.plans.clone());
            replays
                .ensure(&key, count, self.graphs.flights(), self.device.native.graph_budget(), &retained, || opened.service().drain(), || {
                    let (recorded, graph) = record_layout(&graph)?;
                    assert_eq!(recorded, key, "a layout's graph is of its key");
                    Ok(graph)
                })
                .map_err(TensorError::Execution)?;
        }
        Ok(())
    }

    /// What its graphs are made of ([`GraphFamilyCensus`]).
    pub fn census(&self) -> GraphFamilyCensus {
        GraphFamilyCensus { layouts: self.layouts, launches: self.launch_graphs.clone() }
    }

    /// A slot's share of its device's graph budget ([`super::SlotGraphs`]).
    fn slot_hold(&self) -> Result<Arc<super::SlotGraphs>, TensorError> {
        self.graphs.slot().map_err(TensorError::Execution)
    }

    /// A slot of outputs in device storage, allocated through `charges`.
    pub fn new_output_slot(self: &Arc<Self>, charges: &Charges) -> Result<NativeGraphFamilyOutputSlot, TensorError> {
        self.output_slot_of(charges, |charges, bytes| charges.allocate(bytes, BUFFER_ALIGNMENT))
    }

    /// A slot of outputs the host reads as they complete (`Contract::Shared`):
    /// a host read of them lands in host-mapped memory, with no copy queued
    /// behind device work a later run has queued. Allocated through
    /// `charges`.
    pub fn new_host_output_slot(self: &Arc<Self>, charges: &Charges) -> Result<NativeGraphFamilyOutputSlot, TensorError> {
        self.output_slot_of(charges, |charges, bytes| {
            charges.allocate_mapped(seismic_topology::Contract::Shared, bytes, BUFFER_ALIGNMENT)
        })
    }

    /// # Panics
    /// Where `charges` are of another device than the family's.
    fn output_slot_of(
        self: &Arc<Self>,
        charges: &Charges,
        allocate: impl FnOnce(&Charges, u64) -> Result<Arc<Allocation>, seismic_compiler::errors::ExecutionError>,
    ) -> Result<NativeGraphFamilyOutputSlot, TensorError> {
        charges.of_device(&self.device);
        let arena = match self.output_bytes {
            0 => None,
            bytes => Some(allocate(charges, bytes)?),
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
    /// share of its device's graph budget ([`Self::launch`]). Its storage
    /// is allocated through `charges`.
    ///
    /// # Panics
    /// Where `charges` are of another device than the family's.
    pub fn new_slot(
        self: &Arc<Self>,
        charges: &Charges,
        regions: &[u64],
    ) -> Result<NativeGraphFamilySlot, TensorError> {
        charges.of_device(&self.device);
        let scratch = zeroed_region(charges, self.workspace_bytes)?;
        Ok(NativeGraphFamilySlot {
            family: self.clone(),
            workspace_records: self.workspace_record_tables(charges, scratch.as_ref())?,
            scratch,
            sync: zeroed_region(charges, self.sync_bytes)?,
            uploads: self.upload_regions(charges, regions)?,
            lent_locals: Vec::new(),
            hold: self.slot_hold()?,
        })
    }

    /// A slot whose workspace is the shared `arena`, with a sync region and
    /// upload regions of `regions` bytes each of its own ([`Self::new_slot`]).
    /// The arena must hold this family's workspace, and no plan of the
    /// family may keep a value in workspace between runs: a prewritten port
    /// is refused, since another family's graph rewrites the arena before
    /// this family's next run. What the slot allocates of its own is
    /// allocated through `charges`.
    ///
    /// # Panics
    /// Where `charges` are of another device than the family's.
    pub fn new_slot_in(
        self: &Arc<Self>,
        charges: &Charges,
        arena: &NativeExecutionArena,
        regions: &[u64],
    ) -> Result<NativeGraphFamilySlot, WorkflowError> {
        charges.of_device(&self.device);
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
            .any(|plan| plan.topology.ports.iter().any(|port| port.prewritten))
        {
            return Err(WorkflowError::NativeArenaPrewrittenPort);
        }
        let scratch = (self.workspace_bytes != 0)
            .then(|| arena.allocation.clone())
            .flatten();
        Ok(NativeGraphFamilySlot {
            family: self.clone(),
            workspace_records: self
                .workspace_record_tables(charges, scratch.as_ref())
                .map_err(WorkflowError::TensorView)?,
            scratch,
            sync: zeroed_region(charges, self.sync_bytes).map_err(WorkflowError::TensorView)?,
            uploads: self
                .upload_regions(charges, regions)
                .map_err(WorkflowError::TensorView)?,
            lent_locals: Vec::new(),
            hold: self.slot_hold().map_err(WorkflowError::TensorView)?,
        })
    }

    /// Each plan's workspace record table over `scratch`, in plan order.
    fn workspace_record_tables(
        &self,
        charges: &Charges,
        scratch: Option<&Arc<Allocation>>,
    ) -> Result<Vec<Option<Arc<Allocation>>>, TensorError> {
        self.plans
            .iter()
            .map(|plan| {
                scratch
                    .map(|scratch| plan.workspace_record_table(charges, scratch))
                    .transpose()
                    .map(Option::flatten)
            })
            .collect()
    }

    fn upload_regions(&self, charges: &Charges, regions: &[u64]) -> Result<Vec<Arc<Allocation>>, TensorError> {
        if self.upload_bytes == 0 {
            return Ok(Vec::new());
        }
        regions
            .iter()
            .map(|&bytes| Ok(charges.allocate_mapped(seismic_topology::Contract::Upload, bytes, BUFFER_ALIGNMENT)?))
            .collect()
    }
}

/// One device's graph workspace, shared by every family slot created in it.
/// A device executes one submission at a time, in submission order, and a
/// graph's workspace holds only the values of its own run, so graphs bound
/// to one arena never use it at once: the device needs one workspace the
/// size of its largest family's, whatever the families and launches in
/// flight. Sync scratch, which must be zero when a launch starts, is never
/// in it: each slot holds its own sync region.
#[derive(Clone)]
pub struct NativeExecutionArena {
    device: Arc<DeviceInner>,
    allocation: Option<Arc<Allocation>>,
    bytes: u64,
}

impl NativeExecutionArena {
    /// An arena of `bytes`, allocated through `charges`.
    pub fn new(charges: &Charges, bytes: u64) -> Result<Self, TensorError> {
        let allocation = zeroed_region(charges, bytes)?;
        Ok(Self {
            device: charges.device().clone(),
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

/// A zeroed region of `bytes` through `charges`; none where it holds
/// nothing.
fn zeroed_region(charges: &Charges, bytes: u64) -> Result<Option<Arc<Allocation>>, TensorError> {
    Ok(match bytes {
        0 => None,
        bytes => Some(charges.allocate_zeroed(bytes, BUFFER_ALIGNMENT)?),
    })
}

/// Release `allocation` where its holder alone held it: the bytes it was
/// charged, its record tables included; none where another holder shares it,
/// which releases it with its own.
fn released(allocation: Arc<Allocation>) -> u64 {
    Arc::try_unwrap(allocation).map_or(0, |allocation| allocation.charged_bytes())
}

pub struct NativeGraphFamilyOutputSlot {
    family: Arc<NativeGraphFamily>,
    arena: Option<Arc<Allocation>>,
}

impl NativeGraphFamilyOutputSlot {
    /// Release the slot's arena: the bytes the device is no longer
    /// charged, those of an arena it alone held ([`released`]).
    pub fn release(self) -> u64 {
        self.arena.map_or(0, released)
    }

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
    /// Its own sync region ([`NativeGraphFamily::sync_bytes`]).
    sync: Option<Arc<Allocation>>,
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
    /// Release the slot's storage, its record tables and uploads before the
    /// scratch they address, then its graph hold: the bytes the device is no
    /// longer charged, those of each allocation it alone held
    /// ([`released`]) and what its family's graph hold shrank by
    /// ([`super::SlotGraphs::release`]).
    pub fn release(self) -> u64 {
        let Self { scratch, sync, workspace_records, uploads, hold, .. } = self;
        let charged: u64 =
            workspace_records.into_iter().flatten().chain(uploads).chain(scratch).chain(sync).map(released).sum();
        charged + super::SlotGraphs::release(hold)
    }

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
            sync: self.sync.clone(),
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
    /// Node `sync` scratch: the slot's own, zero whenever a launch starts.
    sync: Option<Arc<Allocation>>,
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
        if !self.plan.topology.ports.get(ordinal)?.prewritten {
            return None;
        }
        let Some(Placement::Scratch(offset)) = self.plan.topology.port_placements[ordinal] else {
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
                .topology
                .ports
                .get(ordinal)
                .is_some_and(|spec| spec.owned_input);
        let Some(Placement::Upload(offset)) =
            owned.then(|| self.plan.topology.port_placements[ordinal]).flatten()
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
    /// validated at seal, and each binding against its port as it was set
    /// ([`NativeGraphBindings::set`]).
    pub fn attach(
        &mut self,
        bindings: NativeGraphBindings,
        exports: NativeGraphOutputs,
    ) -> Result<ReadyNativeGraphRun<'_>, CallError> {
        if bindings.plan.identity != self.plan.identity || exports.plan.identity != self.plan.identity {
            return Err(CallError::Workflow(WorkflowError::NativeGraphSlotMismatch));
        }
        if exports.submitted {
            return Err(CallError::Workflow(
                WorkflowError::NativeOutputLeaseConsumed,
            ));
        }
        let owned = [&self.workspace, &self.sync, &self.upload, &exports.arena]
            .into_iter()
            .flatten()
            .map(|allocation| allocation.identity())
            .collect::<Vec<_>>();
        for (port, spec) in self.plan.topology.ports.iter().enumerate() {
            if spec.local {
                continue;
            }
            let tensor = &bindings.inputs[port].as_ref().ok_or(CallError::Workflow(
                WorkflowError::NativePortUnbound { port },
            ))?.tensor;
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
            .zip(&self.plan.topology.timelines)
            .enumerate()
        {
            // A port binds only a timeline of its node's role
            // (`NativeTimelinePort`): a publication binds as a park, and
            // only a park waits for an answer.
            let (timeline, value, action) = match (kind, bound) {
                (_, None) => return Err(CallError::Workflow(WorkflowError::NativeTimelineUnbound { node })),
                (TimelineNode::Publish, Some(TimelineBinding::Answered)) => {
                    return Err(CallError::Workflow(WorkflowError::NotPark { node }));
                }
                (TimelineNode::Park, Some(TimelineBinding::Answered)) => continue,
                (TimelineNode::Park | TimelineNode::Publish, Some(TimelineBinding::Parked(timeline))) => {
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
                (TimelineNode::Wait, Some(TimelineBinding::Bound(timeline, value))) => {
                    (timeline, value, TimelineAction::Wait)
                }
                (TimelineNode::Signal, Some(TimelineBinding::Bound(timeline, value))) => {
                    (timeline, value, TimelineAction::Signal)
                }
                (_, Some(_)) => unreachable!("a timeline port binds only its node's role"),
            };
            if !Arc::ptr_eq(timeline.device(), &self.plan.device) {
                return Err(CallError::Workflow(WorkflowError::NativeTimelineMismatch {
                    node,
                }));
            }
            if action == TimelineAction::Wait {
                timeline.await_at(value);
            }
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
    pub fn park(&self, port: NativeTimelinePort<Parks>) -> Option<u64> {
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
        if reference.workflow != self.plan.identity
            || !self
                .plan
                .topology
                .exports
                .contains(&(reference.node, reference.result))
        {
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
            || !self
                .plan
                .topology
                .exports
                .contains(&(reference.node, reference.result))
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
    /// The plan whose run it binds: each binding is checked against its
    /// port, and bound as the plan's nodes use it
    /// (`Executable::port_uses`).
    plan: Arc<NativeGraphPlan>,
    inputs: Vec<Option<BoundPort>>,
    /// Ports bound by `NativeGraphPlan::bind_static`.
    fixed: Vec<bool>,
    /// Per timeline node, how the run binds it.
    timelines: Vec<Option<TimelineBinding>>,
}

/// A tensor bound to an external port, with its records where a record
/// port binds it ([`TensorInner::forming_records`]): formed as it is bound,
/// through the binder's charges, so a run only reads them and allocates
/// nothing.
#[derive(Clone)]
struct BoundPort {
    tensor: Arc<TensorInner>,
    records: Option<Arc<Records>>,
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
    /// Bind timeline node `port` to `timeline`, of the node's role, at
    /// `value` for this run.
    pub fn set_timeline<R: ValuedRole>(
        &mut self,
        port: NativeTimelinePort<R>,
        timeline: &TimelineHost<R>,
        value: u64,
    ) -> Result<(), WorkflowError> {
        let timeline = timeline.inner().clone();
        if port.workflow != self.plan.identity {
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
    pub fn set_park(&mut self, port: NativeTimelinePort<Parks>, timeline: &TimelineHost<Parks>) -> Result<(), WorkflowError> {
        let timeline = timeline.inner().clone();
        if port.workflow != self.plan.identity {
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
    pub fn set_answered(&mut self, port: NativeTimelinePort<Parks>) -> Result<(), WorkflowError> {
        if port.workflow != self.plan.identity {
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

    /// Attach an exported tensor from a reserved, unsubmitted result lease
    /// ([`Self::set`]). The destination port descriptor and aliases are
    /// checked by `attach`.
    pub fn set_reserved_export(
        &mut self,
        charges: &Charges,
        port: NativePort,
        outputs: &NativeGraphOutputs,
        reference: WorkflowResultRef,
    ) -> Result<(), WorkflowError> {
        let tensor = outputs.reserved_export(reference)?;
        self.set(charges, port, tensor)
    }

    /// Bind external port `port` to `tensor`, checked against the port
    /// first ([`NativeGraphPlan::check_binding`]). Where a record port binds
    /// it, its records are formed then: the table its storage does not hold
    /// yet through `charges`, the binder's, and held with the storage from
    /// then on, so a later binding of it finds the table and a run allocates
    /// nothing. A refused binding forms nothing.
    ///
    /// # Panics
    /// Where a table is formed through `charges` of another device, or
    /// through charges that allocate nothing (`Charges::none`).
    pub fn set(&mut self, charges: &Charges, port: NativePort, tensor: Arc<TensorInner>) -> Result<(), WorkflowError> {
        if port.reference.workflow != self.plan.identity || port.reference.node != INPUT_NODE {
            return Err(WorkflowError::CrossWorkflowResult);
        }
        let ordinal = port.reference.result as usize;
        if self.plan.topology.ports.get(ordinal).is_none_or(|port| port.local) {
            return Err(WorkflowError::NativePortMismatch { port: ordinal });
        }
        let uses = &self.plan.port_uses[ordinal];
        let value = self
            .inputs
            .get_mut(ordinal)
            .ok_or(WorkflowError::CrossWorkflowResult)?;
        if value.is_some() {
            return Err(WorkflowError::NativePortAlreadyBound { port: ordinal });
        }
        if uses.grid && tensor.is_slab_view() {
            return Err(WorkflowError::NativePortSlabGrid { port: ordinal });
        }
        if uses.slab && !tensor.binds_slab_port() {
            return Err(WorkflowError::PlainSlabBinding {
                parameter: format!("port {ordinal}"),
            });
        }
        if uses.plain && tensor.is_slab_view() && tensor.byte_len() > 0 {
            return Err(WorkflowError::SlabPlainBinding {
                parameter: format!("port {ordinal}"),
            });
        }
        self.plan.check_binding(ordinal, &tensor)?;
        let records = match uses.records {
            true => Some(tensor.forming_records(charges).map_err(WorkflowError::TensorView)?),
            false => None,
        };
        *value = Some(BoundPort { tensor, records });
        Ok(())
    }
}

pub struct ReadyNativeGraphRun<'a> {
    slot: &'a mut NativeGraphSlot,
    /// The tensor of every external port (`None` for graph locals).
    externals: Vec<Option<BoundPort>>,
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
        let spans = NativeSpans::new(&self.slot.plan.device, self.slot.plan.topology.marks.len())?;
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
        // A record port's tensor's records were formed as it was bound
        // (`NativeGraphBindings::set`): a run allocates no table.
        let externals = self
            .externals
            .iter()
            .map(|bound| {
                bound.as_ref().map(|bound| External {
                    allocation: bound.tensor.allocation().clone(),
                    offset: bound.tensor.byte_offset(),
                    records: bound.records.clone(),
                })
            })
            .collect::<Vec<_>>();
        sequence.access.extend(
            self.externals
                .iter()
                .zip(&plan.topology.external_writes)
                .flat_map(|(bound, write)| {
                    bound
                        .iter()
                        .flat_map(|bound| bound.tensor.bound_allocations())
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
                // The slot's own sync region, where the plan places in it.
                .chain(
                    self.slot
                        .sync
                        .iter()
                        .filter(|_| plan.sync_bytes > 0)
                        .map(|allocation| (allocation.clone(), true)),
                )
                .chain(
                    self.slot
                        .upload
                        .iter()
                        .map(|allocation| (allocation.clone(), plan.topology.upload_written)),
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
                .map(|bound| &bound.tensor)
                .filter(|tensor| tensor.holds_binding_lease())
                .cloned(),
        );
        #[cfg(any(test, target_os = "macos"))]
        let slabs = self
            .externals
            .iter()
            .enumerate()
            .filter_map(|(port, bound)| {
                bound
                    .as_ref()
                    .and_then(|bound| bound.tensor.slab_reach())
                    .map(|slab| (port, slab))
            })
            .collect();
        // Where sampled, each of its plan's marks stamped at its node.
        let stamps = spans
            .into_iter()
            .flat_map(|spans| plan.topology.marks.iter().copied().zip(spans.stamps()))
            .collect();
        let mut run = QueuedRun {
            plan,
            externals,
            #[cfg(any(test, target_os = "macos"))]
            slabs,
            workspace: self.slot.workspace.clone(),
            workspace_records: self.slot.workspace_records.clone(),
            sync: self.slot.sync.clone(),
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
            let encodes = run.plan.node_launches(call)
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
                (0..queued.plan.topology.nodes.len()).map(move |node| (run, node))
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
            first += run.plan.topology.nodes.len();
        }
        #[cfg(any(test, target_os = "macos"))]
        let access_summary = SequenceAccessSummary::of(&self.runs, &nodes);
        let list = SequenceList {
            runs: &self.runs,
            nodes,
            boundaries,
            stamps,
            #[cfg(any(test, target_os = "macos"))]
            access_summary,
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
    #[cfg(any(test, target_os = "macos"))]
    slabs: Vec<(usize, Arc<crate::api::tensor::SlabReach>)>,
    workspace: Option<Arc<Allocation>>,
    /// The slot's workspace record table (`NativeGraphSlot::workspace_records`).
    workspace_records: Option<Arc<Allocation>>,
    sync: Option<Arc<Allocation>>,
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

    /// What device work reaches through the slab view bound to `port`,
    /// where a slab view is bound there.
    #[cfg(any(test, target_os = "macos"))]
    fn slab(&self, port: u32) -> Option<&crate::api::tensor::SlabReach> {
        self.slabs
            .iter()
            .find(|(slabbed, _)| *slabbed == port as usize)
            .map(|(_, slab)| &**slab)
    }

    /// The records of the tensor bound to `port`, which a record port binds.
    fn records(&self, port: u32) -> &Records {
        self.external(port)
            .records
            .as_ref()
            .expect("binding a record port forms its records")
    }

    /// Node `node`; the buffers its kernel binds are appended to `buffers`
    /// (`DispatchList::dispatch`).
    fn dispatch<'s>(&'s self, node_idx: usize, buffers: &mut Vec<(&'s Allocation, u64)>) -> Dispatch<'s> {
        let run = self;
        let node = &self.plan.topology.nodes[node_idx];
        let kernel = &self.plan.kernels[node_idx];
        let region = |region: &'s Option<Arc<Allocation>>| {
            &**region
                .as_ref()
                .expect("a planned graph storage region is present")
        };
        kernel.bind(node.sites.iter().enumerate().map(|(i, site)| {
            let idx = node.site_base + i;
            let offset = self.plan.site_offsets[idx];
            let records = self.plan.site_records[idx];
            match (site.region, records) {
                (Region::External(port), SiteRecords::Port(_)) => {
                    let records = run.records(port);
                    let entry = offset / records.record;
                    (
                        &*records.table,
                        records.table_offset + seismic_lang::registry::record_table_entry(entry),
                    )
                }
                (Region::Workspace, SiteRecords::Workspace { table, .. }) => (region(&run.workspace_records), table),
                (Region::External(port), SiteRecords::None) => {
                    let external = run.external(port);
                    (&*external.allocation, external.offset + offset)
                }
                (Region::Workspace, SiteRecords::None) => (region(&run.workspace), offset),
                (Region::Sync, SiteRecords::None) => (region(&run.sync), offset),
                (Region::Upload, SiteRecords::None) => (region(&run.upload), offset),
                (Region::Outputs, SiteRecords::None) => (region(&run.outputs), offset),
                (region, records) => unreachable!("a {records:?} record site in {region:?}"),
            }
        }), buffers);
        Dispatch {
            kernel,
            words: self.plan.node_words(node_idx),
            launches: self.plan.node_launches(node_idx),
            rounds: self.plan.rounds[node_idx],
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
        (0..self.plan.topology.nodes.len())
            .flat_map(|node| {
                self.plan
                    .node_launch_range(node)
                    .enumerate()
                    .filter(|&(_, launch)| !self.plan.launch_conditions(launch).iter().all(holds))
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
    /// Per-sequence access summary for Metal hazard scheduling:
    /// - `written_groups`: access groups written by at least one dispatch.
    ///   Strictly read-only sites are omitted from `reaches`.
    /// - `plain_groups`: access groups touched by any non-slab binding.
    /// A slab site whose pool access group (`table`) is not in `plain_groups`
    /// is summarized in O(1) via `SlabReach::extend_logical` (emitting exact
    /// physical bytes for any leading lent slabs and one logical range for all
    /// pool slabs).
    #[cfg(any(test, target_os = "macos"))]
    access_summary: SequenceAccessSummary,
}

#[cfg(any(test, target_os = "macos"))]
struct SequenceAccessSummary {
    written_groups: std::collections::HashSet<usize>,
    plain_groups: std::collections::HashSet<usize>,
}

#[cfg(any(test, target_os = "macos"))]
impl SequenceAccessSummary {
    fn of(runs: &[QueuedRun], nodes: &[(usize, usize)]) -> Self {
        let mut written_groups = std::collections::HashSet::new();
        let mut plain_groups = std::collections::HashSet::new();
        for &(run_idx, node_idx) in nodes {
            let run = &runs[run_idx];
            let static_node = &run.plan.topology.nodes[node_idx];
            let kernel = &run.plan.kernels[node_idx];
            if let Some(scalars) = &kernel.scalars {
                let g = scalars.slots.access_identity();
                written_groups.insert(g);
                plain_groups.insert(g);
            }
            for (buffer, site) in static_node.sites.iter().enumerate() {
                let writes = kernel.writes(buffer);
                let idx = static_node.site_base + buffer;
                match (site.region, run.plan.site_records[idx]) {
                    (Region::External(port), SiteRecords::Port(_)) => {
                        let records = run.records(port);
                        if let Some(slab) = run.slab(port) {
                            if writes {
                                written_groups.insert(records.table.access_identity());
                                written_groups.extend(slab.lender_groups.iter().copied());
                            }
                        } else if let Some((storage, _)) = &records.storage {
                            let g = storage.access_identity();
                            plain_groups.insert(g);
                            if writes {
                                written_groups.insert(g);
                            }
                        }
                    }
                    (Region::External(port), SiteRecords::None) => {
                        let g = run.external(port).allocation.access_identity();
                        if let Some(slab) = run.slab(port) {
                            if writes {
                                written_groups.insert(g);
                                written_groups.extend(slab.lender_groups.iter().copied());
                            }
                        } else {
                            plain_groups.insert(g);
                            if writes {
                                written_groups.insert(g);
                            }
                        }
                    }
                    (Region::Workspace, _) => {
                        if let Some(allocation) = &run.workspace {
                            let g = allocation.access_identity();
                            plain_groups.insert(g);
                            if writes {
                                written_groups.insert(g);
                            }
                        }
                    }
                    (Region::Sync, _) => {
                        if let Some(allocation) = &run.sync {
                            let g = allocation.access_identity();
                            plain_groups.insert(g);
                            if writes {
                                written_groups.insert(g);
                            }
                        }
                    }
                    (Region::Upload, _) => {
                        if let Some(allocation) = &run.upload {
                            let g = allocation.access_identity();
                            plain_groups.insert(g);
                            if writes {
                                written_groups.insert(g);
                            }
                        }
                    }
                    (Region::Outputs, SiteRecords::None) => {
                        if let Some(allocation) = &run.outputs {
                            let g = allocation.access_identity();
                            plain_groups.insert(g);
                            if writes {
                                written_groups.insert(g);
                            }
                        }
                    }
                    (region, records) => unreachable!("a {records:?} record site in {region:?}"),
                }
            }
        }
        Self {
            written_groups,
            plain_groups,
        }
    }

    fn slab_written(&self, table_group: usize, slab: &crate::api::tensor::SlabReach) -> bool {
        self.written_groups.contains(&table_group)
            || slab
                .lender_groups
                .iter()
                .any(|group| self.written_groups.contains(group))
    }

    fn slab_is_logical(&self, table_group: usize) -> bool {
        !self.plain_groups.contains(&table_group)
    }
}

/// Record the CUDA graph of a run of `layout`, whose nodes are the
/// launches its form holds, which its plans make, over placeholder storage
/// a replay binds to its run's ([`PlaceholderRun`], `cuda::record_run`):
/// as a run first needs one, with no device work queued
/// (`graph_replays::Replays::record`). Returns its shape's key with it.
pub(super) fn record_layout(layout: &Arc<super::LayoutGraph>) -> Result<(Vec<u64>, seismic_cuda::direct::DirectGraph), ExecutionError> {
    over_placeholders(layout, |device, runs| super::cuda::record_run(device, runs))?
}

/// `f` of a CUDA device and a run of each of `layout`'s plans over one
/// placeholder allocation ([`PlaceholderRun`]).
fn over_placeholders<R>(
    layout: &Arc<super::LayoutGraph>,
    f: impl FnOnce(&seismic_cuda::Device, &[PlaceholderRun<'_>]) -> R,
) -> Result<R, ExecutionError> {
    let plans = &layout.plans;
    assert!(
        plans.iter().all(|plan| plan.shape.layout == plans[0].shape.layout && plan.graphed()),
        "a graph is one layout's, of plans that replay graphs"
    );
    let device = &plans[0].device;
    let crate::backends::OpenedKind::Cuda(opened) = &device.kind else {
        unreachable!("only CUDA records a layout's graph");
    };
    // The device's own: recording is caused by no holder's allocation, and
    // runs on whatever thread first submits the layout.
    let placeholder = Charges::recording(device).allocate_zeroed(BUFFER_ALIGNMENT, BUFFER_ALIGNMENT)?;
    let runs = plans
        .iter()
        .map(|plan| PlaceholderRun { plan, placeholder: &placeholder, layout })
        .collect::<Vec<_>>();
    Ok(f(opened.service(), &runs))
}

/// A sealed plan's run over placeholder storage: every buffer it binds at
/// `placeholder`, with its sealed words and geometry and every launch
/// encoded. A layout's graph is recorded from it ([`record_layout`]); the
/// run's addresses and encoding are parameters a replay sets, so its
/// graphs are those of every run of its shape.
struct PlaceholderRun<'a> {
    plan: &'a NativeGraphPlan,
    placeholder: &'a Allocation,
    /// Its layout's graph (`super::LayoutForms`).
    layout: &'a Arc<super::LayoutGraph>,
}

impl DispatchList for PlaceholderRun<'_> {
    fn count(&self) -> usize {
        self.plan.topology.nodes.len()
    }

    fn dispatch<'s>(&'s self, index: usize, buffers: &mut Vec<(&'s Allocation, u64)>) -> Dispatch<'s> {
        let node = &self.plan.topology.nodes[index];
        let kernel = &self.plan.kernels[index];
        kernel.bind(node.sites.iter().map(|_| (self.placeholder, 0)), buffers);
        Dispatch {
            kernel,
            words: self.plan.node_words(index),
            launches: self.plan.node_launches(index),
            rounds: self.plan.rounds[index],
            representations: &node.representations,
        }
    }

    fn encodes(&self, _index: usize, _ordinal: usize) -> bool {
        true
    }

    /// None: a graph's launches keep stream order, whatever they reach.
    #[cfg(any(test, target_os = "macos"))]
    fn reaches(&self, _index: usize, _reaches: &mut Vec<Reach>) {}

    fn runs(&self) -> Option<Vec<super::PlanRun>> {
        Some(vec![super::PlanRun {
            plan: self.plan.identity,
            layout: self.plan.shape.layout,
            calls: 0..self.count(),
            timelines: self.plan.topology.timelines.to_vec(),
            marks: self.plan.held_marks(),
            launched: !self.plan.graphed(),
            holds_timelines: self.plan.holds_timelines(),
            graph: self.layout.clone(),
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

    #[cfg(any(test, target_os = "macos"))]
    fn reaches(&self, index: usize, reaches: &mut Vec<Reach>) {
        let (run, node) = self.nodes[index];
        let run = &self.runs[run];
        let region = |region: &Option<Arc<Allocation>>| {
            region
                .as_ref()
                .expect("a planned graph storage region is present")
                .identity()
        };
        let extend_slab = |buffer: usize,
                           slab: &crate::api::tensor::SlabReach,
                           table: &Allocation,
                           table_offset: u64,
                           offset: u64,
                           bytes: u64,
                           reaches: &mut Vec<Reach>| {
            let group = table.access_identity();
            if !self.access_summary.slab_written(group, slab) {
                return;
            }
            if self.access_summary.slab_is_logical(group) {
                slab.extend_logical(buffer, table, table_offset, offset, bytes, reaches);
            } else {
                slab.extend(buffer, table, table_offset, offset, bytes, reaches);
            }
        };
        let static_node = &run.plan.topology.nodes[node];
        for (buffer, site) in static_node.sites.iter().enumerate() {
            let idx = static_node.site_base + buffer;
            let site_bytes = run.plan.site_bytes[idx];
            let site_offset = run.plan.site_offsets[idx];
            let site_records = run.plan.site_records[idx];
            let (allocation, offset, bytes) = match (site.region, site_records) {
                (Region::External(port), SiteRecords::Port(count)) => {
                    let records = run.records(port);
                    match run.slab(port) {
                        Some(slab) => extend_slab(
                            buffer,
                            slab,
                            &records.table,
                            records.table_offset,
                            site_offset,
                            count * records.record,
                            reaches,
                        ),
                        None => {
                            if records.storage.as_ref().is_none_or(|(storage, _)| {
                                self.access_summary
                                    .written_groups
                                    .contains(&storage.access_identity())
                            }) {
                                records.extend(buffer, site_offset / records.record, count, reaches);
                            }
                        }
                    }
                    continue;
                }
                // A workspace record site's records, which its table
                // addresses; the entries order nothing (`Records::extend`).
                (Region::Workspace, SiteRecords::Workspace { .. }) => {
                    reaches.push(Reach {
                        buffer,
                        allocation: region(&run.workspace),
                        offset: site_offset,
                        bytes: site_bytes,
                        through: Through::Slab,
                    });
                    continue;
                }
                (Region::External(port), SiteRecords::None) => {
                    let External { allocation, offset: start, .. } = run.external(port);
                    if let Some(slab) = run.slab(port) {
                        extend_slab(buffer, slab, allocation, *start, site_offset, site_bytes, reaches);
                        continue;
                    }
                    if !self
                        .access_summary
                        .written_groups
                        .contains(&allocation.access_identity())
                    {
                        continue;
                    }
                    (allocation.identity(), start + site_offset, site_bytes)
                }
                (Region::Workspace, SiteRecords::None) => (region(&run.workspace), site_offset, site_bytes),
                (Region::Sync, SiteRecords::None) => (region(&run.sync), site_offset, site_bytes),
                (Region::Upload, SiteRecords::None) => (region(&run.upload), site_offset, site_bytes),
                (Region::Outputs, SiteRecords::None) => (region(&run.outputs), site_offset, site_bytes),
                (region, records) => unreachable!("a {records:?} record site in {region:?}"),
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
                    let calls = first..first + run.plan.topology.nodes.len();
                    first = calls.end;
                    let graph = run.hold.layout(run.plan.shape.layout);
                    super::PlanRun {
                        plan: run.plan.identity,
                        layout: run.plan.shape.layout,
                        calls,
                        timelines: run.plan.topology.timelines.to_vec(),
                        marks: run.plan.held_marks(),
                        launched: !run.plan.graphed(),
                        holds_timelines: run.plan.holds_timelines(),
                        graph,
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
            args: Arc::new(args),
            scratch: vec![ScratchNeed {
                bytes: rows * 4,
                sync: false,
                operand: None,
                emission: None,
            }],
            activation: None,
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
        let links = operand_links::<(), _>(&nodes, &[]).unwrap();
        let certified =
            make_layout(&ports, &nodes, &links, &results, &[], &[], HostHeap::default(), HostHeap::default(), 0, GraphCensus::default());
        let charge = certified.storage_bytes();
        let (ports, nodes, results) = class(8);
        let exact = certified
            .storage_for(&ports, &nodes, &links, &results, &[], &[])
            .unwrap();
        assert_eq!(exact.scratch_bytes, charge.workspace);
        let (ports, nodes, results) = class(32);
        assert_eq!(
            certified.storage_for(&ports, &nodes, &links, &results, &[], &[]).err(),
            Some(WorkflowError::NativeGraphLayoutMismatch(LayoutMismatch::Exceeds {
                buffer: GraphBuffer::Port {
                    port: 0,
                    extents: vec![32],
                },
                bytes: 128,
                capacity: 64,
            }))
        );
    }

    #[test]
    fn non_cuda_recorded_census_counts_only_active_launches() {
        let f32s = seismic_lang::registry::representation("f32").unwrap();
        let shape = SealedNodeShape {
            buffers: 2,
            arguments: 2,
            words: 4,
            launches: 4,
            conditions: 0,
            parks: false,
        };
        let build = |backend: BackendName, active_mask: Option<u64>| {
            let mut draft = NativeGraphMetadataDraft::new(backend);
            let input = draft.port(f32s, &[4], false, false).unwrap();
            let mut args = EncodedWorkflowArgs::new();
            args.push_result_tensor(input.reference);
            let mut pending = draft
                .enqueue(
                    "routed",
                    args,
                    vec![Some((f32s, vec![4]))],
                    vec![Some((f32s, vec![4]))],
                    Vec::new(),
                    None,
                    shape.clone(),
                    active_mask,
                )
                .unwrap();
            draft.export(pending.take()).unwrap();
            draft
        };

        // Metal: 1 active launch out of 4 declared -> launches == 1, not 4.
        let metal_single = build(BackendName::Metal, Some(0b0010)).seal().unwrap();
        assert_eq!(
            metal_single.recorded,
            GraphCensus { graphs: 1, launches: 1, arguments: 0, largest: 0 }
        );

        // Metal certify_group: union of 0b0001 and 0b0100 -> 2 active launches, not 4.
        let metal_group = ValidatedMetadataDraft::certify_group([
            build(BackendName::Metal, Some(0b0001)),
            build(BackendName::Metal, Some(0b0100)),
        ])
        .unwrap();
        assert_eq!(
            metal_group.storage_bytes().recorded,
            GraphCensus { graphs: 1, launches: 2, arguments: 0, largest: 0 }
        );

        // Vulkan: 2 active launches out of 4 declared -> 2 launches and 2x argument_bytes.
        let arg_bytes = argument_bytes(BackendName::Vulkan, 2, 8 * 4);
        let vulkan = build(BackendName::Vulkan, Some(0b1010)).seal().unwrap();
        assert_eq!(
            vulkan.recorded,
            GraphCensus {
                graphs: 1,
                launches: 2,
                arguments: 2 * arg_bytes,
                largest: arg_bytes,
            }
        );

        // Zero active launches (all branches inactive): 0 launches, 0 arguments, 0 largest.
        let none_active = build(BackendName::Vulkan, Some(0)).seal().unwrap();
        assert_eq!(
            none_active.recorded,
            GraphCensus { graphs: 1, launches: 0, arguments: 0, largest: 0 }
        );

        // CPU: None mask falls back to all declared launches (4).
        let cpu_all = build(BackendName::Cpu, None).seal().unwrap();
        assert_eq!(
            cpu_all.recorded,
            GraphCensus { graphs: 1, launches: 4, arguments: 0, largest: 0 }
        );
    }
}

#[cfg(test)]
mod operand_pairing_tests {
    use super::*;

    const WORKFLOW: u64 = 1;

    fn spec(bytes: u64) -> NativeTensorSpec {
        let representation = seismic_lang::registry::representation("f32").unwrap();
        let canonical = layout::canonical(representation, &[bytes / 4]).unwrap();
        NativeTensorSpec {
            representation,
            extents: vec![bytes / 4],
            strides: canonical.strides,
            byte_len: canonical.byte_len,
        }
    }

    fn port(representation: &str, extents: &[u64], local: bool) -> PortSpec {
        let representation = seismic_lang::registry::representation(representation).unwrap();
        let canonical = layout::canonical(representation, extents).unwrap();
        PortSpec {
            representation,
            extents: extents.to_vec(),
            strides: canonical.strides,
            byte_len: canonical.byte_len,
            local,
            owned_input: false,
            prewritten: false,
        }
    }

    fn bf16() -> Option<RepresentationId> {
        seismic_lang::registry::representation("bf16")
    }

    /// A node reading `source` (a port's or a node result's tensor) with
    /// `scratch`, its activation bf16.
    fn node(source: (u32, u32), scratch: Vec<ScratchNeed>) -> PlannedNode {
        let mut args = EncodedWorkflowArgs::new();
        args.push_result_tensor(WorkflowResultRef { workflow: WORKFLOW, node: source.0, result: source.1 });
        PlannedNode { args: Arc::new(args), scratch, activation: bf16(), condition: None, share: None }
    }

    /// A node reading `source`, normed by port `norm`'s tensor under
    /// `epsilon` (its arguments 1 and 2), with `scratch`.
    fn normed(source: (u32, u32), norm: u32, epsilon: f32, scratch: Vec<ScratchNeed>) -> PlannedNode {
        let mut args = EncodedWorkflowArgs::new();
        args.push_result_tensor(WorkflowResultRef { workflow: WORKFLOW, node: source.0, result: source.1 });
        args.push_result_tensor(WorkflowResultRef { workflow: WORKFLOW, node: INPUT_NODE, result: norm });
        args.push_scalar(EncodedScalar::F32Bits(epsilon.to_bits()));
        PlannedNode { args: Arc::new(args), scratch, activation: bf16(), condition: None, share: None }
    }

    const NORM: Option<NativeOperandNorm> = Some(NativeOperandNorm { weight: 1, epsilon: 2 });

    fn need(bytes: u64, operand: Option<ScratchOperand>) -> ScratchNeed {
        ScratchNeed { bytes, sync: false, operand, emission: None }
    }

    /// A buffer serving the emission of result 0.
    fn emitting(bytes: u64, operand: Option<ScratchOperand>, sync: bool) -> ScratchNeed {
        ScratchNeed { bytes, sync, operand, emission: Some(0) }
    }

    fn formed(rows: bool, formation: u32) -> Option<ScratchOperand> {
        Some(ScratchOperand::Formed { parameter: 0, formation, rows, norm: NORM })
    }

    fn emitted(rows: bool) -> Option<ScratchOperand> {
        Some(ScratchOperand::Emitted { result: 0, rows, norm: NORM })
    }

    fn at(pairs: bool, formed: bool) -> Option<FormedAt> {
        Some(FormedAt { pairs, formed, nvfp4: false })
    }

    /// A producer of node 0 emitting both forms of its result's operand
    /// under port 1's norm, with its arrivals.
    fn producer() -> PlannedNode {
        normed(
            (INPUT_NODE, 0),
            1,
            1e-6,
            vec![
                emitting(4096, emitted(true), false),
                emitting(320, emitted(false), false),
                emitting(16, None, true),
            ],
        )
    }

    /// A consumer of `source` forming both forms in launch 0 under port 1's
    /// norm, beside a staging buffer of its own.
    fn consumer(source: (u32, u32)) -> PlannedNode {
        normed(source, 1, 1e-6, vec![need(4096, formed(true, 0)), need(320, formed(false, 0)), need(64, None)])
    }

    fn carry(side: OperandCarrySide, rows: u32) -> Carry {
        Carry { rows, quantized: rows + 1, forms: (), activation: bf16().unwrap(), side }
    }

    fn carry_in(rows: u32) -> Carry {
        carry(OperandCarrySide::In { hidden: 0, norm: 1, epsilon: 1e-6f32.to_bits() }, rows)
    }

    fn carry_out(rows: u32) -> Carry {
        carry(OperandCarrySide::Out { result: WorkflowResultRef { workflow: WORKFLOW, node: 0, result: 0 } }, rows)
    }

    /// The storage of a chain of nodes, each with the `sync` buffers of
    /// `bytes`.
    fn sync_chain(layers: &[Vec<u64>]) -> StoragePlan {
        let nodes = layers
            .iter()
            .enumerate()
            .map(|(at, buffers)| {
                let source = if at == 0 { (INPUT_NODE, 0) } else { (at as u32 - 1, 0) };
                let needs = buffers.iter().map(|&bytes| ScratchNeed { bytes, sync: true, operand: None, emission: None });
                node(source, needs.collect())
            })
            .collect::<Vec<_>>();
        let links = operand_links::<(), _>(&nodes, &[]).unwrap();
        let ports = [port("f32", &[64], true)];
        let results = vec![vec![Some(spec(1024))]; nodes.len()];
        plan_storage(&ports, &nodes, &links, &results, &[], &[])
    }

    /// A graph of more `sync` buffers than any fixed reservation held forms,
    /// its sync region sized by what its nodes place, every buffer at its
    /// own aligned offset: 300 one-counter nodes (beyond the 256 aligned
    /// buffers 64 KiB once held), and a deep, wide MoE decode graph (61
    /// layers at H = 7168, each a routed output's per-row arrivals of
    /// H * 4 bytes, a landing launch's counters of a 64-entry list and two
    /// one-word arrivals: about 1.8 MiB).
    #[test]
    fn a_graphs_sync_region_holds_every_nodes_counters() {
        let counters = sync_chain(&vec![vec![4]; 300]);
        assert_eq!(counters.sync_bytes, 299 * BUFFER_ALIGNMENT + 4);
        for (node, row) in counters.node_scratch.iter().enumerate() {
            assert_eq!(row[0], Placement::Sync(node as u64 * BUFFER_ALIGNMENT), "node {node}");
        }

        let layer = vec![7168 * 4, 4 * (4 + 64), 4, 4];
        let moe = sync_chain(&vec![layer.clone(); 61]);
        let per_layer = layer.iter().map(|bytes| bytes.next_multiple_of(BUFFER_ALIGNMENT)).sum::<u64>();
        assert_eq!(moe.sync_bytes, 61 * per_layer - (BUFFER_ALIGNMENT - 4));
        let mut placed = moe
            .node_scratch
            .iter()
            .flatten()
            .zip(std::iter::repeat(&layer).flatten())
            .map(|(placement, &bytes)| match *placement {
                Placement::Sync(offset) => offset..offset + bytes,
                other => panic!("a sync buffer lies at {other:?}"),
            })
            .collect::<Vec<_>>();
        placed.sort_by_key(|range| range.start);
        assert!(placed.windows(2).all(|pair| pair[0].end <= pair[1].start), "sync buffers are disjoint");
        assert_eq!(placed.last().unwrap().end, moe.sync_bytes);
    }

    /// A producer that emits its result's operand, a node between, and two
    /// consumers of that result's operand: the first consumer's buffer is
    /// the producer's, live across the node between, and its formation is
    /// left out where its operand stands for its own; the second forms its
    /// own.
    #[test]
    fn a_consumers_operand_lies_where_its_producer_emits_it() {
        let input = (INPUT_NODE, 0);
        let emitted = Some(ScratchOperand::Emitted { result: 0, rows: false, norm: None });
        let formed = Some(ScratchOperand::Formed { parameter: 0, formation: 0, rows: false, norm: None });
        let nodes = vec![
            node(input, vec![need(272, emitted), ScratchNeed { bytes: 4, sync: true, operand: None, emission: None }]),
            node(input, vec![need(4096, None)]),
            node((0, 0), vec![need(2176, formed), need(512, None)]),
            node((0, 0), vec![need(8192, formed)]),
        ];
        let links = operand_links::<(), _>(&nodes, &[]).unwrap();
        assert_eq!(links.pairs, [OperandPair { consumer: 2, formed: 0, producer: 0, emitted: 0 }]);
        assert!(!links.linked(3, 0), "a second consumer forms its own");
        assert_eq!(formations_left_out(2, &nodes[2].scratch, &[at(true, true), None], &links), [0]);
        assert!(formations_left_out(2, &nodes[2].scratch, &[at(false, true), None], &links).is_empty());
        assert!(formations_left_out(3, &nodes[3].scratch, &[at(true, true)], &links).is_empty());

        let ports = [port("f32", &[64], true)];
        let results = vec![vec![Some(spec(1024))], vec![Some(spec(1024))], vec![Some(spec(1024))], vec![Some(spec(1024))]];
        let storage = plan_storage(&ports, &nodes, &links, &results, &[], &[]);
        let scratch = |node: usize, ordinal: usize| match storage.node_scratch[node][ordinal] {
            Placement::Scratch(offset) => offset,
            other => panic!("node {node}'s buffer {ordinal} lies at {other:?}"),
        };
        let at = scratch(0, 0);
        assert_eq!(scratch(2, 0), at, "the consumer reads the producer's operand");
        // The second consumer's larger buffer is placed only across its own
        // step, never widening the paired buffer live across node 1.
        assert!(
            storage.scratch_bytes < 8192 + 4096 + 1024,
            "the paired buffer is sized to its paired consumer alone: {}",
            storage.scratch_bytes,
        );
        assert_eq!(storage.node_scratch[0][1], Placement::Sync(0), "sync scratch lies in the sync region");
        assert_eq!(storage.sync_bytes, 4, "the sync region is what its nodes place");
        // The paired operand, as large as the consumer's, lives across the
        // node between: nothing else live then aliases it.
        let between = scratch(1, 0);
        assert!(between + 4096 <= at || at + 2176 <= between, "the node between aliases the operand");
        let Placement::Scratch(result) = storage.placements[0][0] else { panic!("the producer's result is placed") };
        assert!(result + 1024 <= at || at + 2176 <= result, "the producer's result aliases its operand");
    }

    /// A formation is left out where every buffer it forms at the call is
    /// linked and stands for its own, whatever it does not form: its Q8
    /// blocks where it takes no INT8 dot need no producer. Where nothing it
    /// forms is linked, or nothing is formed, it stays.
    #[test]
    fn a_formation_is_left_out_on_the_forms_it_writes() {
        let nodes = vec![producer(), consumer((0, 0))];
        let links = operand_links::<(), _>(&nodes, &[]).unwrap();
        assert_eq!(links.pairs.len(), 2, "both forms pair");
        let scratch = &nodes[1].scratch;
        let left_out = |forms: [Option<FormedAt>; 2]| formations_left_out(1, scratch, &[forms[0], forms[1], None], &links);
        assert_eq!(left_out([at(true, true), at(true, true)]), [0]);
        assert_eq!(left_out([at(true, true), at(false, false)]), [0], "Q8 blocks it does not form");
        assert!(left_out([at(true, true), at(false, true)]).is_empty(), "Q8 blocks it forms that do not pair");
        assert!(left_out([at(true, false), at(true, false)]).is_empty(), "nothing formed");

        // Rows alone pair: a formation that also forms Q8 blocks stays.
        let mut unpaired = consumer((0, 0));
        unpaired.scratch[1].operand = Some(ScratchOperand::Formed { parameter: 0, formation: 0, rows: false, norm: None });
        let nodes = vec![producer(), unpaired];
        let links = operand_links::<(), _>(&nodes, &[]).unwrap();
        let scratch = &nodes[1].scratch;
        assert!(formations_left_out(1, scratch, &[at(true, true), at(true, true), None], &links).is_empty());
        assert_eq!(formations_left_out(1, scratch, &[at(true, true), at(true, false), None], &links), [0]);
    }

    /// Every buffer serving an emission no consumer reads, its operand
    /// buffers and its arrivals alike, is charged the minimum, as a call
    /// alone charges it; a paired or carried emission keeps its bytes.
    #[test]
    fn unread_emissions_reserve_the_minimum() {
        let minimum = |nodes: &[PlannedNode]| nodes[0].scratch.iter().map(|need| need.bytes).collect::<Vec<_>>();
        let mut alone = vec![producer(), node((0, 0), vec![need(64, None)])];
        let links = operand_links::<(), _>(&alone, &[]).unwrap();
        charge_node_emissions(&mut alone, &links);
        assert_eq!(minimum(&alone), [1, 1, 1]);

        let mut paired = vec![producer(), consumer((0, 0))];
        let links = operand_links::<(), _>(&paired, &[]).unwrap();
        charge_node_emissions(&mut paired, &links);
        assert_eq!(minimum(&paired), [4096, 320, 16]);

        let mut carried = vec![producer()];
        let links = operand_links(&carried, &[carry_out(2)]).unwrap();
        charge_node_emissions(&mut carried, &links);
        assert_eq!(minimum(&carried), [4096, 320, 16]);

        let mut standalone = producer().scratch;
        charge_emissions(&mut standalone, |_| false);
        assert_eq!(standalone.iter().map(|need| need.bytes).collect::<Vec<_>>(), [1, 1, 1]);
    }

    /// A carried operand buffer lies on its carry's port, on either side,
    /// in no workspace; a layout certifies the carry with its topology.
    #[test]
    fn carried_operands_lie_on_their_ports() {
        let [(rows, rows_extents), (blocks, blocks_extents)] = operand_carry_ports(BackendName::Cpu, 1, 256);
        let name = |id| seismic_lang::registry::representation_info(id).name;
        let ports = [
            port("f32", &[1, 256], false),
            port("f32", &[256], false),
            port(name(rows), &rows_extents, false),
            port(name(blocks), &blocks_extents, false),
        ];
        let results = vec![vec![Some(spec(1024))]];

        let producing = vec![producer()];
        let links = operand_links(&producing, &[carry_out(2)]).unwrap();
        let storage = plan_storage(&ports, &producing, &links, &results, &[], &[]);
        assert_eq!(storage.node_scratch[0][0], Placement::Port(2));
        assert_eq!(storage.node_scratch[0][1], Placement::Port(3));
        assert!(matches!(storage.node_scratch[0][2], Placement::Sync(_)), "its arrivals stay in the slot's sync region");

        let consuming = vec![consumer((INPUT_NODE, 0))];
        let links = operand_links(&consuming, &[carry_in(2)]).unwrap();
        let storage = plan_storage(&ports, &consuming, &links, &results, &[], &[]);
        assert_eq!(storage.node_scratch[0][0], Placement::Port(2));
        assert_eq!(storage.node_scratch[0][1], Placement::Port(3));
        assert!(matches!(storage.node_scratch[0][2], Placement::Scratch(_)), "its staging stays in the workspace");
        let uncarried = operand_links::<(), _>(&consuming, &[]).unwrap();
        let formed = plan_storage(&ports, &consuming, &uncarried, &results, &[], &[]);
        assert!(storage.scratch_bytes < formed.scratch_bytes, "carried buffers take no workspace");

        let certified = make_layout(&ports, &consuming, &links, &results, &[], &[], HostHeap::default(), HostHeap::default(), 0, GraphCensus::default());
        assert!(certified.storage_for(&ports, &consuming, &links, &results, &[], &[]).is_ok());
        assert_eq!(
            certified.storage_for(&ports, &consuming, &uncarried, &results, &[], &[]).err(),
            Some(WorkflowError::NativeGraphLayoutMismatch(LayoutMismatch::Topology)),
            "a layout of a carry certifies no graph without it"
        );
    }

    /// The quantized port of an operand carry rounds its row extent up to
    /// `CPU_Q8_ROWS` on CPU (covering the tiled projection's Q8 scratch for
    /// `rows = 1` and other non-multiples) and stays unpadded on GPU backends.
    #[test]
    fn operand_carry_quantized_port_rounds_rows_per_backend() {
        use seismic_lang::cpu::{CPU_Q8_BYTES, CPU_Q8_ROWS, CPU_Q8_VALUES};
        let name = |id| seismic_lang::registry::representation_info(id).name;
        let native_port = |ordinal: u32| NativePort {
            reference: WorkflowResultRef { workflow: WORKFLOW, node: INPUT_NODE, result: ordinal },
        };
        let backends = [
            (BackendName::Cpu, CPU_Q8_ROWS),
            (BackendName::Metal, 1),
            (BackendName::Cuda, 1),
            (BackendName::Vulkan, 1),
        ];
        for (backend, tile) in backends {
            assert_eq!(operand_carry_row_tile(backend), tile);
            for rows in [1u64, 2, 7, 8, 9, 15, 16] {
                for width in [1u64, 256, 257, 2048] {
                    let [(rows_repr, rows_extents), (q_repr, q_extents)] = operand_carry_ports(backend, rows, width);
                    let expected_q_rows = rows.next_multiple_of(tile);
                    let expected_blocks = width.div_ceil(CPU_Q8_VALUES);
                    assert_eq!(rows_extents, [rows, width]);
                    assert_eq!(q_extents, [expected_q_rows, expected_blocks * OPERAND_CARRY_BLOCK_WORDS]);
                    let q_port = port(name(q_repr), &q_extents, false);
                    let expected_bytes = expected_q_rows * expected_blocks * CPU_Q8_BYTES;
                    assert_eq!(q_port.byte_len, expected_bytes);

                    let ports = [
                        port("f32", &[rows, width], false),
                        port("f32", &[width], false),
                        port(name(rows_repr), &rows_extents, false),
                        q_port,
                    ];
                    let carry = OperandCarry {
                        rows: native_port(2),
                        quantized: native_port(3),
                        activation: bf16().unwrap(),
                    };
                    let mut carries = Vec::new();
                    add_carry(
                        backend,
                        WORKFLOW,
                        &ports,
                        0,
                        &mut carries,
                        &carry,
                        (),
                        CarryEnd::In { hidden: native_port(0), norm: native_port(1), epsilon: 1e-6 },
                    )
                    .unwrap();

                    if rows % CPU_Q8_ROWS != 0 {
                        let wrong_rows = if backend == BackendName::Cpu {
                            rows
                        } else {
                            rows.next_multiple_of(CPU_Q8_ROWS)
                        };
                        let mut mismatched = ports;
                        mismatched[3] = port(name(q_repr), &[wrong_rows, expected_blocks * OPERAND_CARRY_BLOCK_WORDS], false);
                        let mut carries = Vec::new();
                        assert_eq!(
                            add_carry(
                                backend,
                                WORKFLOW,
                                &mismatched,
                                0,
                                &mut carries,
                                &carry,
                                (),
                                CarryEnd::In { hidden: native_port(0), norm: native_port(1), epsilon: 1e-6 },
                            )
                            .err(),
                            Some(WorkflowError::NativePortMismatch { port: 3 }),
                            "mismatched quantized rows={wrong_rows} for backend={backend:?} rows={rows} must be rejected"
                        );
                    }
                }
            }
        }
    }

    /// What a carry or pair does not hold is refused: a carry nothing
    /// reads, a carry out of a producer emitting no normed operand, and
    /// ends of another activation element.
    #[test]
    fn carries_and_pairs_refuse_what_they_cannot_hold() {
        // Another norm or epsilon reads no carry.
        let other_epsilon = vec![normed((INPUT_NODE, 0), 1, 1e-5, consumer((INPUT_NODE, 0)).scratch)];
        assert_eq!(
            operand_links(&other_epsilon, &[carry_in(2)]).err(),
            Some(WorkflowError::UnreadOperandCarry { port: 2 })
        );
        let raw = vec![node((INPUT_NODE, 0), vec![need(64, None)])];
        assert_eq!(operand_links(&raw, &[carry_out(2)]).err(), Some(WorkflowError::OperandEmitter { node: 0 }));
        // A producer emitting rows alone lacks a form a carry holds.
        let mut rows_alone = producer();
        rows_alone.scratch.remove(1);
        assert_eq!(
            operand_links(&[rows_alone], &[carry_out(2)]).err(),
            Some(WorkflowError::OperandEmitter { node: 0 })
        );
        // A consumer forming rows alone reads a carry's rows.
        let mut rows_reader = consumer((INPUT_NODE, 0));
        rows_reader.scratch.remove(1);
        assert_eq!(operand_links(&[rows_reader], &[carry_in(2)]).unwrap().carried.len(), 1);

        let f32 = seismic_lang::registry::representation("f32");
        let mut other = consumer((INPUT_NODE, 0));
        other.activation = f32;
        assert_eq!(
            operand_links(&[other], &[carry_in(2)]).err(),
            Some(WorkflowError::OperandElement { node: 0 })
        );
        let mut other = consumer((0, 0));
        other.activation = f32;
        assert_eq!(operand_links::<(), _>(&[producer(), other], &[]).err(), Some(WorkflowError::OperandElement { node: 1 }));
    }

    /// A reader's kernel reads a carry only in the forms the carry holds:
    /// its rows formed under the carry's transform, its Q8 blocks formed
    /// exactly where the carry holds them, and never rows alone of a carry
    /// of Q8 blocks.
    #[test]
    fn a_carry_is_read_in_the_forms_it_holds() {
        let at = |formed, nvfp4| FormedAt { pairs: true, formed, nvfp4 };
        let rows_only = OperandForms { quantized: false, nvfp4: false };
        let quantized = OperandForms { quantized: true, nvfp4: false };
        let transformed = OperandForms { quantized: false, nvfp4: true };
        for blocks in [false, true] {
            assert!(rows_only.read_by(true, at(true, false), blocks));
            assert!(!rows_only.read_by(true, at(false, false), blocks), "its rows are always formed");
            assert!(!rows_only.read_by(true, at(true, true), blocks));
            assert!(transformed.read_by(true, at(true, true), blocks));
            assert!(!transformed.read_by(true, at(true, false), blocks));
        }
        assert!(quantized.read_by(true, at(true, false), true));
        assert!(!quantized.read_by(true, at(true, false), false), "a reader of rows alone leaves its blocks unread");
        assert!(quantized.read_by(false, at(true, false), true));
        assert!(!quantized.read_by(false, at(false, false), true));
        assert!(rows_only.read_by(false, at(false, false), true));
        assert!(!rows_only.read_by(false, at(true, false), true));
    }

    /// A producer's operand formed under a norm stands for a consumer's only
    /// under the same one: both absent, or both bound to the same norm
    /// weight and the same epsilon.
    #[test]
    fn operands_pair_only_under_one_norm() {
        let args = |norm_port: u32, epsilon: f32| {
            let mut args = EncodedWorkflowArgs::new();
            args.push_result_tensor(WorkflowResultRef { workflow: WORKFLOW, node: INPUT_NODE, result: norm_port });
            args.push_scalar(crate::api::kernel::EncodedScalar::F32Bits(epsilon.to_bits()));
            args
        };
        let norm = Some(NativeOperandNorm { weight: 0, epsilon: 1 });
        assert!(same_norm(&args(0, 1e-6), norm, &args(0, 1e-6), norm));
        assert!(!same_norm(&args(0, 1e-6), norm, &args(1, 1e-6), norm), "another norm weight");
        assert!(!same_norm(&args(0, 1e-6), norm, &args(0, 1e-5), norm), "another epsilon");
        assert!(!same_norm(&args(0, 1e-6), norm, &args(0, 1e-6), None), "a raw operand");
        assert!(same_norm(&args(0, 1e-6), None, &args(1, 1e-5), None));
    }
}

#[cfg(test)]
mod liveness_placement_tests {
    use super::*;

    struct OraclePlaced {
        offsets: Vec<u64>,
        bytes: u64,
        guards: usize,
    }

    fn oracle_colour_ends(intervals: &[&Interval], order: &[usize]) -> (Vec<Option<u8>>, Vec<usize>) {
        let mut parent = (0..intervals.len()).collect::<Vec<_>>();
        let mut parity = vec![0u8; intervals.len()];
        let find = |parent: &[usize], parity: &[u8], mut index: usize| {
            let mut relative = 0u8;
            while parent[index] != index {
                relative ^= parity[index];
                index = parent[index];
            }
            (index, relative)
        };
        let mut coloured = Vec::with_capacity(order.len());
        let mut lifted = Vec::new();
        for &index in order {
            let mut required: Vec<(usize, u8)> = Vec::new();
            let mut odd = false;
            for &other in &coloured {
                if !intervals[index].crosses(intervals[other]) {
                    continue;
                }
                let (root, relative) = find(&parent, &parity, other);
                let want = relative ^ 1;
                match required.iter().find(|(found, _)| *found == root) {
                    Some(&(_, wanted)) if wanted != want => {
                        odd = true;
                        break;
                    }
                    Some(_) => {}
                    None => required.push((root, want)),
                }
            }
            if odd {
                lifted.push(index);
                continue;
            }
            for (root, want) in required {
                parent[root] = index;
                parity[root] = want;
            }
            coloured.push(index);
        }
        let mut ends = vec![None; intervals.len()];
        for index in coloured {
            ends[index] = Some(find(&parent, &parity, index).1);
        }
        (ends, lifted)
    }

    fn aligned_floor_end(intervals: &[&Interval], steps: &[usize]) -> u64 {
        steps
            .iter()
            .map(|&step| {
                let live = intervals.iter().filter(|interval| interval.start <= step && step <= interval.end);
                let (bytes, pad) = live.fold((0u64, 0u64), |(bytes, pad), interval| {
                    let aligned = align_up(interval.bytes, BUFFER_ALIGNMENT);
                    (bytes + aligned, pad.max(aligned - interval.bytes))
                });
                bytes - pad
            })
            .max()
            .unwrap_or(0)
    }

    /// Oracle copy of `main`'s old `place` (`first_fit` + largest-first odd-cycle pass).
    fn oracle_place(intervals: &[&Interval]) -> OraclePlaced {
        fn first_fit(intervals: &[&Interval]) -> OraclePlaced {
            let mut order = (0..intervals.len()).collect::<Vec<_>>();
            order.sort_by_key(|&index| std::cmp::Reverse(intervals[index].bytes));
            let mut guards = order.len().saturating_sub(1);
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
                    guards += 1;
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
            OraclePlaced { offsets, bytes: size, guards }
        }
        let bytes = |index: usize| align_up(intervals[index].bytes, BUFFER_ALIGNMENT);
        let birth = |&index: &usize| (intervals[index].start, std::cmp::Reverse(intervals[index].end), index);
        let mut held = (0..intervals.len()).filter(|&index| !intervals[index].scratch()).collect::<Vec<_>>();
        held.sort_by_key(birth);
        let (mut ends, mut lifted) = oracle_colour_ends(intervals, &held);
        let odd = !lifted.is_empty();
        let mut guards = 0usize;
        let largest_first = |order: &mut Vec<usize>, guards: &mut usize| {
            order.sort_by_key(|index| (std::cmp::Reverse(intervals[*index].bytes), birth(index)));
            *guards += order.len().saturating_sub(1);
        };
        if odd {
            largest_first(&mut held, &mut guards);
            (ends, lifted) = oracle_colour_ends(intervals, &held);
        }
        lifted.extend((0..intervals.len()).filter(|&index| intervals[index].scratch()));
        match odd {
            true => largest_first(&mut lifted, &mut guards),
            false => lifted.sort_by_key(birth),
        }
        let mut steps = intervals.iter().map(|interval| interval.start).collect::<Vec<_>>();
        steps.sort_unstable();
        steps.dedup();
        let lives = |index: usize, step: usize| intervals[index].start <= step && step <= intervals[index].end;
        let mut held_at = [vec![0u64; steps.len()], vec![0u64; steps.len()]];
        for (index, end) in ends.iter().enumerate() {
            let Some(end) = end else { continue };
            for (at, &step) in steps.iter().enumerate() {
                if lives(index, step) {
                    held_at[*end as usize][at] += bytes(index);
                }
            }
        }
        let mut lifted_at = Vec::with_capacity(lifted.len());
        for &index in &lifted {
            let life = (0..steps.len()).filter(|&at| lives(index, steps[at])).collect::<Vec<_>>();
            let on = |side: usize| {
                let offset = life.iter().map(|&at| held_at[side][at]).max().unwrap_or(0);
                let most = life.iter().map(|&at| offset + bytes(index) + held_at[1 - side][at]).max().unwrap_or(0);
                (most, offset)
            };
            let (side, offset) = match odd {
                false => (0, on(0).1),
                true => {
                    guards += 1;
                    let [bottom, top] = [on(0), on(1)];
                    match top.0 < bottom.0 {
                        true => (1, top.1),
                        false => (0, bottom.1),
                    }
                }
            };
            for &at in &life {
                held_at[side][at] = offset + bytes(index);
            }
            lifted_at.push((index, side, offset));
        }
        let arena = (0..steps.len()).map(|at| held_at[0][at] + held_at[1][at]).max().unwrap_or(0);
        let mut offsets = vec![0u64; intervals.len()];
        for (index, end) in ends.iter().enumerate() {
            let Some(end) = *end else { continue };
            let enclosing = (0..intervals.len())
                .filter(|&other| {
                    other != index
                        && ends[other] == Some(end)
                        && intervals[other].contains(intervals[index])
                        && (!intervals[index].contains(intervals[other]) || other < index)
                })
                .map(bytes)
                .sum::<u64>();
            offsets[index] = match end {
                0 => enclosing,
                _ => arena - bytes(index) - enclosing,
            };
        }
        for (index, side, offset) in lifted_at {
            offsets[index] = match side {
                0 => offset,
                _ => arena - offset - bytes(index),
            };
        }
        let end = intervals.iter().zip(&offsets).map(|(interval, offset)| offset + interval.bytes).max().unwrap_or(0);
        guards += 1;
        let placed = OraclePlaced { offsets, bytes: end, guards };
        if end <= aligned_floor_end(intervals, &steps) {
            return placed;
        }
        let first_fit = first_fit(intervals);
        let guards = placed.guards + first_fit.guards + 1;
        match first_fit.bytes < placed.bytes {
            true => OraclePlaced { guards, ..first_fit },
            false => OraclePlaced { guards, ..placed },
        }
    }

    fn catalog_arenas() -> Vec<(u64, Vec<Interval>)> {
        use std::io::Read;
        let mut text = String::new();
        flate2::read::GzDecoder::new(&include_bytes!("fixtures/arenas.txt.gz")[..])
            .read_to_string(&mut text)
            .unwrap();
        text.lines()
            .enumerate()
            .map(|(row, line)| {
                let mut fields = line.split(' ');
                let first_fit = fields.next().unwrap().parse::<u64>().unwrap();
                let intervals = fields
                    .enumerate()
                    .map(|(ordinal, field)| {
                        let parts = field.split(':').collect::<Vec<_>>();
                        let number = |at: usize| parts[at].parse::<u64>().unwrap();
                        Interval {
                            key: match parts[0] {
                                "P" => StorageKey::Port(ordinal),
                                "R" => StorageKey::Result(ordinal, 0),
                                "S" => StorageKey::NodeScratch(ordinal, 0),
                                kind => panic!("arena {row}: unknown buffer kind {kind}"),
                            },
                            start: number(1) as usize,
                            end: number(2) as usize,
                            bytes: number(3),
                            exported: false,
                        }
                    })
                    .collect::<Vec<_>>();
                (first_fit, intervals)
            })
            .collect()
    }

    fn interval(start: usize, end: usize, bytes: u64) -> Interval {
        Interval {
            key: StorageKey::Result(0, 0),
            start,
            end,
            bytes,
            exported: false,
        }
    }

    fn placed(intervals: &[Interval]) -> (Vec<u64>, u64) {
        let placed = place(&intervals.iter().collect::<Vec<_>>());
        (placed.offsets, placed.bytes)
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
    fn a_chain_alternates_between_the_ends_with_one_guard() {
        // Each buffer read by the next node alone: a chain, which takes the
        // most two adjacent links hold.
        let intervals = [interval(1, 3, 256), interval(3, 5, 512), interval(5, 7, 768), interval(7, 9, 256)];
        let refs = intervals.iter().collect::<Vec<_>>();
        let placed = place(&refs);
        assert_disjoint(&intervals, &placed.offsets);
        assert_eq!(placed.bytes, 512 + 768);
        let oracle = oracle_place(&refs);
        assert_eq!(oracle.guards, 1);
    }

    #[test]
    fn an_odd_cycle_of_crossings_reaches_the_floor() {
        // Three buffers each crossing the other two, all live at step 5.
        let intervals = [interval(1, 5, 1024), interval(3, 7, 256), interval(5, 9, 512)];
        let refs = intervals.iter().collect::<Vec<_>>();
        let placed = place(&refs);
        assert_disjoint(&intervals, &placed.offsets);
        assert_eq!(placed.bytes, floor(&intervals));
        let oracle = oracle_place(&refs);
        assert!(oracle.guards > 1);
    }

    /// The 8,373 distinct arenas the served catalog's outline tests place on
    /// every backend, each line first fit's bytes for the arena and then its
    /// buffers (`kind:start:end:bytes`, kind `P` port, `R` result, `S` node
    /// scratch). Every arena places disjointly under the oracle, never above
    /// first fit, and under no more guards than the most any shape of the
    /// catalog holds.
    #[test]
    fn no_catalog_arena_places_above_first_fit() {
        const MOST_GUARDS: usize = 4_238;
        for (row, (first_fit, intervals)) in catalog_arenas().into_iter().enumerate() {
            let placed = oracle_place(&intervals.iter().collect::<Vec<_>>());
            assert_disjoint(&intervals, &placed.offsets);
            assert!(
                placed.bytes <= first_fit,
                "arena {row}: {} bytes above first fit's {first_fit}",
                placed.bytes
            );
            assert!(placed.guards <= MOST_GUARDS, "arena {row}: {} guards", placed.guards);
        }
    }

    /// R_b (`seismic_memory::placement::place`) places every catalog arena
    /// disjointly and never exceeds `main`'s aligned placement end, and
    /// `CompiledPlacement` matches `place` and `liveness_floor` bit-for-bit on
    /// every arena.
    #[test]
    fn rb_never_exceeds_main_on_arena_fixture() {
        for (row, (_first_fit, intervals)) in catalog_arenas().into_iter().enumerate() {
            let refs = intervals.iter().collect::<Vec<_>>();
            let rb = place(&refs);
            let main_placed = oracle_place(&refs);
            assert_disjoint(&intervals, &rb.offsets);
            let main_aligned = align_up(main_placed.bytes, BUFFER_ALIGNMENT);
            assert!(
                rb.bytes <= main_aligned,
                "arena {row}: R_b {} bytes above main's aligned {main_aligned}",
                rb.bytes
            );

            let spans = intervals
                .iter()
                .map(|interval| seismic_memory::placement::Span {
                    start: interval.start,
                    end: interval.end,
                })
                .collect::<Vec<_>>();
            let scratch = intervals
                .iter()
                .map(Interval::scratch)
                .collect::<Vec<_>>();
            let raw_bytes = intervals.iter().map(|interval| interval.bytes).collect::<Vec<_>>();
            let padded_bytes = raw_bytes
                .iter()
                .map(|&bytes| seismic_memory::placement::padded(bytes))
                .collect::<Vec<_>>();
            let compiled = seismic_memory::placement::CompiledPlacement::of(&spans, &scratch);
            assert_eq!(
                compiled.arena_bytes(&padded_bytes),
                rb.bytes,
                "arena {row}: CompiledPlacement::arena_bytes mismatch"
            );
            assert_eq!(
                compiled.liveness_floor(&raw_bytes),
                liveness_floor(&refs),
                "arena {row}: CompiledPlacement::liveness_floor mismatch"
            );
        }
    }

    #[test]
    fn compiled_storage_plan_matches_plan_storage_with_links() {
        const WORKFLOW: u64 = 42;
        let port_ref = |ordinal: u32| WorkflowResultRef {
            workflow: WORKFLOW,
            node: INPUT_NODE,
            result: ordinal,
        };
        let node_ref = |node: u32, result: u32| WorkflowResultRef {
            workflow: WORKFLOW,
            node,
            result,
        };

        // Ports:
        // 0: external (non-local)
        // 1: local owned_input (upload)
        // 2: local prewritten workspace
        // 3: local exported
        // 4: external store target
        // 5: external carry target
        let make_ports = |sizes: [u64; 6]| {
            vec![
                StoragePort { byte_len: sizes[0], local: false, owned_input: false, prewritten: false },
                StoragePort { byte_len: sizes[1], local: true, owned_input: true, prewritten: false },
                StoragePort { byte_len: sizes[2], local: true, owned_input: false, prewritten: true },
                StoragePort { byte_len: sizes[3], local: true, owned_input: false, prewritten: false },
                StoragePort { byte_len: sizes[4], local: false, owned_input: false, prewritten: false },
                StoragePort { byte_len: sizes[5], local: false, owned_input: false, prewritten: false },
            ]
        };

        // Node 0: consumes port 1 & 2, produces tensor result 0 & scalar result 1,
        //         has unpaired scratch 0, sync scratch 1, paired producer scratch 2.
        let mut args0 = EncodedWorkflowArgs::new();
        args0.push_result_tensor(port_ref(1));
        args0.push_result_tensor(port_ref(2));

        // Node 1: consumes node 0 result 0 & port 3, produces tensor result 0 (stored to port 4)
        //         and tensor result 1 (workspace), has carried scratch 0.
        let mut args1 = EncodedWorkflowArgs::new();
        args1.push_result_tensor(node_ref(0, 0));
        args1.push_result_tensor(port_ref(3));

        // Node 2: consumes node 0 result 0 & node 1 result 1, produces tensor result 0 (exported)
        //         and shares result 1 with node 0 result 0; has paired consumer scratch 0 & sync scratch 1.
        let mut args2 = EncodedWorkflowArgs::new();
        args2.push_result_tensor(node_ref(0, 0));
        args2.push_result_tensor(node_ref(1, 1));

        let links = OperandLinks {
            pairs: vec![OperandPair {
                consumer: 2,
                formed: 0,
                producer: 0,
                emitted: 2,
            }],
            carried: vec![CarriedBuffer {
                node: 1,
                scratch: 0,
                port: 5,
                carry: 0,
            }],
        };
        let exports = vec![port_ref(3), node_ref(2, 0)];
        let stores = vec![(node_ref(1, 0), 4u32)];

        let need = |bytes: u64, sync: bool| ScratchNeed {
            bytes,
            sync,
            operand: None,
            emission: None,
        };

        let slices: &[([u64; 6], Vec<Vec<Option<u64>>>, Vec<Vec<ScratchNeed>>)] = &[
            (
                [1024, 512, 2048, 300, 4096, 1024],
                vec![
                    vec![Some(4096), None],
                    vec![Some(4096), Some(1536)],
                    vec![Some(8192), Some(4096)],
                ],
                vec![
                    vec![need(768, false), need(64, true), need(1024, false)],
                    vec![need(2048, false)],
                    vec![need(3072, false), need(512, true)],
                ],
            ),
            // Edge case: zero-byte ports, results, and scratch mixed with non-zero.
            (
                [0, 0, 0, 0, 0, 0],
                vec![
                    vec![Some(0), None],
                    vec![Some(0), Some(128)],
                    vec![Some(0), Some(0)],
                ],
                vec![
                    vec![need(0, false), need(0, true), need(512, false)],
                    vec![need(0, false)],
                    vec![need(0, false), need(0, true)],
                ],
            ),
            // Larger slice with producer scratch > consumer scratch.
            (
                [16384, 777, 32768, 999, 8192, 4096],
                vec![
                    vec![Some(65536), None],
                    vec![Some(8192), Some(12288)],
                    vec![Some(10000), Some(65536)],
                ],
                vec![
                    vec![need(4096, false), need(256, true), need(16384, false)],
                    vec![need(4096, false)],
                    vec![need(2048, false), need(1024, true)],
                ],
            ),
        ];

        // Compile once from slice 0.
        let init_ports = make_ports(slices[0].0);
        let init_nodes = [
            StorageNode {
                args: &args0,
                results: &slices[0].1[0],
                scratch: &slices[0].2[0],
                activation: None,
                share: None,
            },
            StorageNode {
                args: &args1,
                results: &slices[0].1[1],
                scratch: &slices[0].2[1],
                activation: None,
                share: None,
            },
            StorageNode {
                args: &args2,
                results: &slices[0].1[2],
                scratch: &slices[0].2[2],
                activation: None,
                share: Some((1, node_ref(0, 0))),
            },
        ];
        let compiled = CompiledStoragePlan::of(&init_ports, &init_nodes, &links, &exports, &stores);

        for (idx, (port_bytes, result_bytes, scratch_needs)) in slices.iter().enumerate() {
            let ports = make_ports(*port_bytes);
            let nodes = [
                StorageNode {
                    args: &args0,
                    results: &result_bytes[0],
                    scratch: &scratch_needs[0],
                    activation: None,
                    share: None,
                },
                StorageNode {
                    args: &args1,
                    results: &result_bytes[1],
                    scratch: &scratch_needs[1],
                    activation: None,
                    share: None,
                },
                StorageNode {
                    args: &args2,
                    results: &result_bytes[2],
                    scratch: &scratch_needs[2],
                    activation: None,
                    share: Some((1, node_ref(0, 0))),
                },
            ];
            let planned = plan_storage_with_links(&ports, &nodes, &links, &exports, &stores);
            let sizes = compiled.storage_bytes(port_bytes, result_bytes, scratch_needs);
            assert_eq!(sizes.workspace, planned.workspace, "slice {idx}: workspace");
            assert_eq!(sizes.workspace_floor, planned.workspace_floor, "slice {idx}: workspace_floor");
            assert_eq!(sizes.sync, planned.sync, "slice {idx}: sync");
            assert_eq!(sizes.output, planned.output, "slice {idx}: output");
            assert_eq!(sizes.upload, planned.upload, "slice {idx}: upload");
        }
    }

    #[test]
    fn buffers_with_disjoint_lifetimes_share_storage() {
        let intervals = [interval(0, 1, 1024), interval(2, 3, 1024)];
        let (offsets, size) = placed(&intervals);
        assert_eq!(offsets, vec![0, 0]);
        assert_eq!(size, 1024);
    }

    #[test]
    fn a_shared_endpoint_is_an_overlap() {
        // A node's inputs end where its results start.
        let intervals = [interval(0, 3, 512), interval(3, 5, 512)];
        let (offsets, size) = placed(&intervals);
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
        let (offsets, size) = placed(&intervals);
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
        let (offsets, size) = placed(&intervals);
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
        assert_eq!(placed(&intervals), placed(&intervals));
        let (offsets, _) = placed(&intervals);
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
            .map(|bytes| crate::memory::Charges::tally(&device).allocate_mapped(seismic_topology::Contract::Upload, bytes, BUFFER_ALIGNMENT).unwrap())
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
        let uploads = vec![crate::memory::Charges::tally(&device).allocate_mapped(seismic_topology::Contract::Upload, 64, BUFFER_ALIGNMENT).unwrap()];
        let _taken = idle_upload(&uploads, 10).unwrap();
        let _ = idle_upload(&uploads, 10);
    }
}

#[cfg(all(test, target_os = "macos"))]
mod condition_tests {
    use super::*;
    use seismic_lang::checked::{Encoded, NativeImplementation, SourceFile, SourceSet, check_source_with};
    use seismic_lang::registry::BackendName;

    /// `fill`, whose one launch its kernel conditions on `policy[2]`.
    fn fill(device: &Arc<DeviceInner>) -> Arc<NativePreparedAny> {
        let module = check_source_with(
            SourceSet::new(vec![SourceFile {
                path: "policy.seismic".into(),
                text: "fn fill[N](policy: &tensor[3] i32, x: &mut tensor[N] f32):\n    for i in 0..N:\n        x[i] = 0.0\n\nnative fill for metal from \"fill.metal\":\n    launch fill:\n        threadgroups (1, 1, 1)\n        unless (policy[2])\n        threads_per_threadgroup (1, 1, 1)\n".into(),
            }]),
            &mut |_entry: &str, _native: &NativeImplementation| {
                Ok::<_, std::convert::Infallible>(Some("kernel void fill() {}\n".into()))
            },
        )
        .expect("policy source checks");
        let entry = module.entry_named("fill").expect("fill entry");
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
            plan.node_launch_range(0)
                .map(|launch| plan.launch_conditions(launch).to_vec())
                .collect::<Vec<_>>(),
            vec![vec![
                LaunchCondition { word, offset: 8, encoded: Encoded::WhileClear },
                LaunchCondition { word, offset: 0, encoded: Encoded::WhileSet },
            ]]
        );
        assert!(plan.port_uses[policy.reference.result as usize].condition);
        assert!(!plan.port_uses[x.reference.result as usize].condition);

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
