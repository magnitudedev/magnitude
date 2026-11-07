//! The explicitly selected direct-native route
//! (`docs/seismic/language/functions-and-capabilities.md`).
//!
//! A checked entry's authored implementation for the opened device's backend
//! is formed under one [`NativeSpecialization`] and executed without a
//! compiler plan. Metal, CUDA and Vulkan sources receive the generated
//! prefix of [`abi`]; CPU implementations are Rust compiled into the binary
//! and reached through [`cpu`].
//!
//! Submission is asynchronous. Launches of one submission share one command
//! buffer of concurrent encoders, each beginning where [`hazard`] finds a
//! dependency (Metal), one stream (CUDA), one command buffer on the device's
//! queue (Vulkan) or the worker pool (CPU); the device queue orders
//! submissions, and allocation fences order host access after them.

pub(crate) mod abi;
mod admission;
mod batch;
mod budget;
mod comparison;
#[cfg(feature = "coverage")]
pub mod coverage;
pub(crate) use batch::{NativeTensorBatch, NativeTensorBatchCompletion};

/// The engine's kernel sources, for the tests that check them: the tree
/// beside this crate's directory, as cargo names it to the running test.
#[cfg(test)]
pub(crate) fn engine_kernels() -> std::path::PathBuf {
    let manifest = std::env::var_os("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR");
    std::path::Path::new(&manifest).join("../../engine/kernels/kernels")
}

#[cfg(test)]
mod bundle_identity_tests;
pub mod cpu;
mod cuda;
pub mod graph;
mod graph_replays;
// The order a submission's launches need: Metal encodes them concurrently
// by level.
pub(crate) mod hazard;
#[cfg(test)]
mod handoff_tests;
mod observation;
pub mod plan;
pub(crate) mod point;
pub mod replay;
pub mod sampling;
pub mod search;
pub mod spans;
pub mod timeline;
mod timing;
pub mod trace;
pub mod transfer;
pub mod tune;
mod validation;
#[cfg(not(target_os = "macos"))]
pub(crate) mod vulkan;

use crate::api::device::DeviceInner;
use crate::api::kernel::{DecodedResults, DecodedValue, EncodedArgs, EncodedOutputs, PrepareError};
use crate::api::tensor::Records;
use crate::api::tensor::TensorInner;
use crate::api::{CallError, OutputError};
use crate::artifacts::ArtifactKey;
use crate::backends::{CpuOpened, CudaOpened, OpenedKind};
use crate::completion::{Committed, DeviceCompletion, IssuedFailure};
pub(crate) use crate::completion::RouteSubmission;
use crate::driver::{collect_native_access, typed_buffer, write_zeros, Allocation};
pub(crate) use crate::formation::FormationClass;
use crate::formation::{Compile, Formed, Keyed, ProgramFormer};
pub use budget::{Footprint, Linear, Observed, Quote};
pub use graph_replays::GraphReplays;
pub use point::{GraphPending, GraphPoint, GraphTiming, KernelPoint, Point, PointPending};
pub use spans::NativeSpans;
use seismic_compiler::errors::{ExecutionError, InvocationError, PreparationError};
use seismic_compiler::graph_cost::{GraphCensus, LaunchGraphs};
use seismic_compiler::prepared::{
    validate_invocation, ArgumentValue, DeviceIdentity, InvocationContract,
};
use seismic_lang::checked::{
    CheckedModule, NativeCondition, NativeImplementation, NativeNatExpr, NativeSpecialization,
    OnDevice,
};
use seismic_lang::entry::{
    CallSchema, CheckedProgramSubject, ElementBindings, LogicalEntry, ParameterKind, ResultClass,
    ResultKind, TensorAccess,
};
use seismic_lang::expr::compiled::{CompiledNat, InvocationValues};
use seismic_lang::expr::{SymbolId, SymbolValue};
use seismic_lang::ids::{EntryId, RepresentationId};
use seismic_lang::registry::BackendName;
use seismic_native_target::{ProgramEntry, ProgramSource, Toolchain};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::{Arc, Mutex};

pub use cpu::{CpuInvocation, CpuKernelFn, CpuLaunchVariants, CpuNativeKernels, CpuVariant};

#[cfg(target_os = "macos")]
use crate::backends::MetalOpened;

#[derive(Clone)]
pub(crate) enum NativeResult {
    Tensor {
        representation: RepresentationId,
        axes: Vec<CompiledNat>,
    },
    Scalar(seismic_lang::types::DType),
    Index,
    Range,
}

impl NativeResult {
    fn class(&self) -> ResultClass {
        match self {
            Self::Tensor { .. } => ResultClass::Tensor,
            Self::Scalar(_) => ResultClass::Scalar,
            Self::Index => ResultClass::Index,
            Self::Range => ResultClass::Range,
        }
    }
}

#[derive(Clone)]
pub(crate) struct NativeTensorSpec {
    pub(crate) representation: RepresentationId,
    pub(crate) extents: Vec<u64>,
    pub(crate) strides: Vec<u64>,
    pub(crate) byte_len: u64,
}

/// Evaluated geometry of one launch of one call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LaunchGeometry {
    /// The groups on each axis; of a launch with `grid`, the most it reads
    /// (on CUDA, the blocks of its grid-stride loop).
    pub(crate) groups: [u64; 3],
    /// The ABI buffer holding the grid the launch dispatches, written by an
    /// earlier launch (`NativeLaunch::grid`).
    pub(crate) grid: Option<usize>,
    /// The ABI buffer holding a resident launch's barrier
    /// (`NativeLaunch::resident`), whose give-up word the host reads once
    /// the work completes ([`NativeSubmission::wait`]).
    pub(crate) barrier: Option<usize>,
    pub(crate) threads: [u64; 3],
    pub(crate) shared_bytes: u64,
}

impl LaunchGeometry {
    /// Whether the grid is empty: the launch does no device work. A grid
    /// read at dispatch is empty when its bound is.
    pub(crate) fn is_empty(&self) -> bool {
        self.groups.contains(&0) || self.threads.contains(&0)
    }
}

/// The launches of one call.
pub(crate) struct CallLaunches {
    /// Geometry by declaration ordinal; `None` for a launch whose `when`
    /// condition does not hold (not encoded).
    pub(crate) geometry: Vec<Option<LaunchGeometry>>,
    /// How often the implementation's repeat block is dispatched; 1 without
    /// one.
    pub(crate) rounds: u64,
}

/// Alignment of the base of every buffer the route places: standalone
/// scratch, and every graph result, local, host-written input, export and
/// scratch buffer, on every device the launch contract's
/// (`seismic_native_cpu::tensor::BUFFER_ALIGNMENT`). Device allocations are
/// at least this aligned, so kernels may rely on it for vector access to any
/// placed buffer, as they do for a standalone call's.
pub(crate) const BUFFER_ALIGNMENT: u64 = seismic_native_cpu::tensor::BUFFER_ALIGNMENT as u64;

/// Per-device state of native submission, on every native queue.
#[derive(Default)]
pub(crate) struct NativeState {
    /// Held from encoding a submission to recording its allocation fences.
    order: Mutex<()>,
    /// CUDA graphs of sealed-plan submissions (CUDA devices only).
    cuda_replays: cuda::CudaReplays,
    /// Recorded Vulkan graphs of sealed-plan submissions (Vulkan devices
    /// only).
    #[cfg(not(target_os = "macos"))]
    vulkan_replays: vulkan::VulkanReplays,
    /// Schedules of sealed-plan submissions (Metal devices only).
    #[cfg(target_os = "macos")]
    metal_schedules: graph_replays::Replays<hazard::Schedule>,
    /// What the graphs the device keeps may be made of.
    graph_budget: Mutex<GraphBudget>,
}

/// The device's graph budget: the sum of every live [`GraphHold`]'s census,
/// how many are live, and the ledger's charge of what that census costs on
/// the device (`DeviceInner::graph_cost`), in device and upload storage:
/// one charge each, resized as holds come and go, so it is always the cost
/// of the whole census, which a sum of the holds' own costs would not be (a
/// driver's pool grows in granules, argument blocks fill chunks).
#[derive(Default)]
struct GraphBudget {
    census: GraphCensus,
    holds: usize,
    device: Option<crate::memory::MemoryCharge>,
    upload: Option<crate::memory::MemoryCharge>,
}

impl GraphBudget {
    /// Charge what `census` costs on `device`, the budget's census becoming
    /// it: the growth reserved whole, or refused with nothing changed. The
    /// bytes the device is no longer charged, none for a growth.
    fn charge(&mut self, device: &DeviceInner, census: GraphCensus) -> Result<u64, ExecutionError> {
        let charged = |budget: &Self| {
            [&budget.device, &budget.upload].into_iter().flatten().map(crate::memory::MemoryCharge::bytes).sum::<u64>()
        };
        let before = charged(self);
        let bytes = device.graph_cost.bytes(census).expect("a device's graph budget's memory fits u64");
        let memory = &device.memory;
        let growth = |charge: &Option<crate::memory::MemoryCharge>, bytes: u64| {
            bytes.saturating_sub(charge.as_ref().map_or(0, crate::memory::MemoryCharge::bytes))
        };
        let reserve = |placement: &Arc<crate::memory::Placement>, bytes: u64| {
            (bytes != 0).then(|| crate::driver::reserve(placement, bytes)).transpose()
        };
        let device_growth = reserve(memory.device(), growth(&self.device, bytes.device))?;
        let upload_growth = reserve(memory.upload(), growth(&self.upload, bytes.upload))?;
        for (charge, growth, bytes) in [
            (&mut self.device, device_growth, bytes.device),
            (&mut self.upload, upload_growth, bytes.upload),
        ] {
            match (charge.as_mut(), growth) {
                (Some(held), Some(mut growth)) => {
                    let extra = growth.remaining();
                    growth.extend(held, extra);
                }
                (None, Some(mut growth)) => *charge = Some(growth.take(bytes)),
                (Some(held), None) => held.shrink(bytes),
                (None, None) => {}
            }
        }
        self.census = census;
        Ok(before.saturating_sub(charged(self)))
    }
}

impl NativeState {
    /// What the device's graphs cost its submissions so far: graphs
    /// recorded (a submission during which one was paid its recording, not
    /// only its launches) and replays that updated their graph's parameters.
    /// A Metal schedule is host data no device holds, derived without a
    /// recording, and is not counted.
    pub(crate) fn graph_replays(&self) -> graph_replays::GraphReplays {
        let counts = self.cuda_replays.counts();
        #[cfg(not(target_os = "macos"))]
        let counts = counts.plus(self.vulkan_replays.counts());
        counts
    }

    /// What the graphs of plan `plan`'s runs cost so far
    /// ([`Self::graph_replays`]).
    pub(crate) fn plan_graph_replays(&self, plan: u64) -> graph_replays::GraphReplays {
        let counts = self.cuda_replays.plan_counts(plan);
        #[cfg(not(target_os = "macos"))]
        let counts = counts.plus(self.vulkan_replays.plan_counts(plan));
        counts
    }

    /// What the graphs the device keeps may be made of: what every live
    /// slot's runs may record.
    pub(crate) fn graph_budget(&self) -> GraphCensus {
        self.graph_budget.lock().expect("graph budget lock is never poisoned").census
    }

    /// Release graphs of every kind beyond `budget`, the device's graph
    /// budget as a hold of it shrank ([`graph_replays::Replays::fit_to`]).
    fn fit_graphs(&self, budget: GraphCensus) {
        self.cuda_replays.fit_to(budget);
        #[cfg(not(target_os = "macos"))]
        self.vulkan_replays.fit_to(budget);
        #[cfg(target_os = "macos")]
        self.metal_schedules.fit_to(budget);
    }

    /// Drop every recorded graph, of every kind, while no hold of the graph
    /// budget is live, checked under each kind's lock.
    fn clear_unheld_graphs(&self) {
        let unheld = || self.graph_budget.lock().expect("graph budget lock is never poisoned").holds == 0;
        self.cuda_replays.clear_unheld(unheld);
        #[cfg(not(target_os = "macos"))]
        self.vulkan_replays.clear_unheld(unheld);
        #[cfg(target_os = "macos")]
        self.metal_schedules.clear_unheld(unheld);
    }
}

/// A share of a device's graph budget: graphs of `census` the device may
/// keep recorded, held until dropped ([`SlotGraphs`]). The device keeps no
/// more graphs than its holds' census bounds.
pub struct GraphHold {
    device: Arc<DeviceInner>,
    census: GraphCensus,
}

impl GraphHold {
    /// A hold of `census`, its memory charged to the device's ledger with
    /// the rest of the budget's; refused, holding nothing, where the ledger
    /// has no room for it.
    fn new(device: &Arc<DeviceInner>, census: GraphCensus) -> Result<Self, ExecutionError> {
        let mut budget = device.native.graph_budget.lock().expect("graph budget lock is never poisoned");
        let grown = budget.census.plus(census).expect("a device's graph budget fits u64");
        budget.charge(device, grown)?;
        budget.holds += 1;
        Ok(Self { device: device.clone(), census })
    }
}

impl GraphHold {
    /// Hold `census` instead: a growth reserved whole, or refused with
    /// nothing changed; a shrink releases the graphs beyond the smaller
    /// budget, the least recently replayed first, so the graphs never hold
    /// more than their owners do. The bytes the device is no longer
    /// charged, none for a growth.
    fn resize(&mut self, census: GraphCensus) -> Result<u64, ExecutionError> {
        let shrinks = census.within(self.census);
        let (budget, freed) = {
            let mut budget = self.device.native.graph_budget.lock().expect("graph budget lock is never poisoned");
            let resized = budget
                .census
                .minus(self.census)
                .and_then(|rest| rest.plus(census))
                .expect("a hold's census is part of its device's budget");
            let freed = budget.charge(&self.device, resized)?;
            (budget.census, freed)
        };
        self.census = census;
        if shrinks {
            self.device.native.fit_graphs(budget);
        }
        Ok(freed)
    }

    /// Release the hold as its drop does: the bytes the device is no longer
    /// charged.
    fn release(self) -> u64 {
        let hold = std::mem::ManuallyDrop::new(self);
        // SAFETY: `hold` is never used or dropped again; its device is read
        // out of it once, and dropped once `unhold` returns.
        let device = unsafe { std::ptr::read(&hold.device) };
        Self::unhold(&device, hold.census)
    }

    /// The budget shrinks by `census`, a hold's. Once no hold is live the
    /// device keeps no graph: only a live hold's runs replay one, and each
    /// graph keeps its plans, which hold the device (a device whose last
    /// hold dropped would otherwise never be freed). A submission that
    /// records or fits graphs holds its runs' holds throughout, so this
    /// never runs beneath one. The bytes the device is no longer charged.
    fn unhold(device: &Arc<DeviceInner>, census: GraphCensus) -> u64 {
        let (unheld, shrunk, freed) = {
            let mut budget = device.native.graph_budget.lock().expect("graph budget lock is never poisoned");
            let shrunk = budget.census.minus(census).expect("a hold's census is part of its device's budget");
            let freed = budget
                .charge(device, shrunk)
                .expect("a smaller census reserves nothing, so its charge is never refused");
            budget.holds -= 1;
            (budget.holds == 0, shrunk, freed)
        };
        match unheld {
            true => device.native.clear_unheld_graphs(),
            // The graphs never hold more than the holds left do.
            false => device.native.fit_graphs(shrunk),
        }
        freed
    }
}

/// The CUDA graph of one layout a family runs: a word per call of its
/// plans, a bit set for each launch of the call any plan of the layout
/// makes (`cuda::launch_structure`'s words, or'ed), each a node of the
/// layout's graph, and those plans, over which a run that finds no graph
/// of its shape records one (`graph::record_layout`). One graph serves
/// every class, block, place and encoding sealed into the layout, each run
/// enabling the nodes of the launches it makes (`cuda::issue`), so a family
/// holds graphs by layout, never by class. Empty words on every other
/// backend, whose graphs are their plans'.
pub(crate) struct LayoutGraph {
    pub(crate) form: Arc<[u64]>,
    pub(crate) plans: Vec<Arc<graph::NativeGraphPlan>>,
}

/// A family's [`LayoutGraph`]s, by layout.
pub(crate) struct LayoutForms {
    forms: std::collections::HashMap<u64, Arc<LayoutGraph>>,
}

impl LayoutForms {
    pub(crate) fn new(forms: impl IntoIterator<Item = (u64, LayoutGraph)>) -> Self {
        Self { forms: forms.into_iter().map(|(layout, graph)| (layout, Arc::new(graph))).collect() }
    }

    /// `layout`'s graph.
    pub(crate) fn of(&self, layout: u64) -> Arc<LayoutGraph> {
        self.forms
            .get(&layout)
            .cloned()
            .expect("a family runs only plans of layouts its graphs were made for")
    }
}

/// A family's share of its device's graph budget and the layouts its runs
/// replay ([`LayoutForms`]): what its live slots' launches in flight can
/// need at once ([`Self::census`]), held from its first live slot to its
/// last. Graphs are recorded as a run first needs one, never as a slot is
/// made (`cuda::issue`).
pub struct FamilyGraphs {
    device: Arc<DeviceInner>,
    layouts: LayoutForms,
    /// What its decode launch replays, which each launch in flight beside
    /// another may hold, and the union of every launch's, which any one may
    /// (`LaunchGraphs::held`).
    launches: LaunchGraphs,
    /// Its live slots, and the hold of their census, none while no slot
    /// lives.
    held: Mutex<(u64, Option<GraphHold>)>,
    /// Its live slots, as the hold counts them: no more than its census
    /// holds, read without the hold's lock by a release of graphs
    /// (`graph_replays::Pinned`), which runs beneath it.
    flights: Arc<std::sync::atomic::AtomicUsize>,
}

impl FamilyGraphs {
    /// `layout`'s graph ([`LayoutForms::of`]).
    pub(crate) fn layout(&self, layout: u64) -> Arc<LayoutGraph> {
        self.layouts.of(layout)
    }

    pub(crate) fn new(device: &Arc<DeviceInner>, layouts: LayoutForms, launches: LaunchGraphs) -> Arc<Self> {
        Arc::new(Self { device: device.clone(), layouts, launches, held: Mutex::new((0, None)), flights: Arc::default() })
    }

    /// What its decode launch and the union of its launches replay.
    pub(crate) fn launches(&self) -> LaunchGraphs {
        self.launches.clone()
    }

    /// Its live slots, each a launch in flight, as its hold counts them.
    pub(crate) fn flights(&self) -> &Arc<std::sync::atomic::AtomicUsize> {
        &self.flights
    }

    /// What the family's graphs hold at most with `slots` live slots, each
    /// a launch in flight ([`LaunchGraphs::held`]).
    fn census(&self, slots: u64) -> GraphCensus {
        self.launches.held(slots).expect("a family's graph census fits u64")
    }

    /// A live slot's share: the family's hold grown to one more slot's
    /// census, or refused, holding nothing more, where the ledger has no
    /// room for it.
    pub(crate) fn slot(self: &Arc<Self>) -> Result<Arc<SlotGraphs>, ExecutionError> {
        let mut held = self.held.lock().expect("family graph hold lock is never poisoned");
        let census = self.census(held.0 + 1);
        match &mut held.1 {
            Some(hold) => drop(hold.resize(census)?),
            None => held.1 = Some(GraphHold::new(&self.device, census)?),
        }
        held.0 += 1;
        // Counted once the hold covers it.
        self.flights.store(held.0 as usize, std::sync::atomic::Ordering::Release);
        Ok(Arc::new(SlotGraphs { family: self.clone() }))
    }
}

/// A live slot of a family ([`FamilyGraphs::slot`]), held by every run
/// queued from it until replayed, so a submission of its runs always fits
/// ([`graph_replays::Replays::record`]).
pub struct SlotGraphs {
    family: Arc<FamilyGraphs>,
}

impl SlotGraphs {
    /// `layout`'s graph ([`LayoutForms::of`]).
    pub(crate) fn layout(&self, layout: u64) -> Arc<LayoutGraph> {
        self.family.layouts.of(layout)
    }
}

impl SlotGraphs {
    /// Release `graphs` where the slot alone held them, as their drop
    /// does: the bytes the device is no longer charged; none where a queued
    /// run shares them, which releases them with its own.
    pub(crate) fn release(graphs: Arc<Self>) -> u64 {
        Arc::try_unwrap(graphs).map_or(0, |graphs| {
            let graphs = std::mem::ManuallyDrop::new(graphs);
            // SAFETY: `graphs` is never used or dropped again; its family is
            // read out of it once, and dropped once `unhold` returns.
            let family = unsafe { std::ptr::read(&graphs.family) };
            Self::unhold(&family)
        })
    }

    /// `family`'s hold shrinks to one slot fewer, and goes with its last:
    /// the bytes the device is no longer charged.
    fn unhold(family: &FamilyGraphs) -> u64 {
        let mut held = family.held.lock().expect("family graph hold lock is never poisoned");
        held.0 -= 1;
        // Uncounted before the hold shrinks and releases graphs beyond it.
        family.flights.store(held.0 as usize, std::sync::atomic::Ordering::Release);
        let census = family.census(held.0);
        let hold = held.1.take().expect("a family holds its graphs while a slot lives");
        match held.0 {
            0 => hold.release(),
            _ => {
                let hold = held.1.insert(hold);
                hold.resize(census).expect("a smaller census reserves nothing, so its charge is never refused")
            }
        }
    }
}

impl Drop for SlotGraphs {
    fn drop(&mut self) {
        Self::unhold(&self.family);
    }
}

impl Drop for GraphHold {
    fn drop(&mut self) {
        Self::unhold(&self.device, self.census);
    }
}

/// Bytes charged to a scratch buffer that is empty or inactive: it keeps
/// its ABI slot.
const MINIMUM_SCRATCH_BYTES: u64 = 1;

/// Bytes at the start of every non-empty scratch arena (a standalone call's
/// and every graph workspace that places anything) reserved for `sync`
/// scratch; a graph that places nothing needs no workspace. Nothing else is ever placed
/// there, arenas are zeroed when allocated, and every call restores its sync
/// buffers to zero before it ends, so the range is zero whenever a launch
/// starts, whichever call or graph used it last.
pub(crate) const SYNC_SCRATCH_BYTES: u64 = 64 * 1024;

/// One call-private scratch buffer: its bytes, and whether it holds arrival
/// counters (`sync`), which live in the reserved range of their arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScratchNeed {
    pub bytes: u64,
    pub sync: bool,
}

/// Place one standalone call's scratch buffers: `sync` buffers in the
/// reserved range, the rest after it. Returns the offsets in declaration
/// order and the end of the placed range.
fn place_scratch(needs: &[ScratchNeed]) -> (Vec<u64>, u64) {
    let mut sync_end = 0u64;
    let mut end = SYNC_SCRATCH_BYTES;
    let offsets = needs
        .iter()
        .map(|need| {
            let cursor = if need.sync { &mut sync_end } else { &mut end };
            let offset = cursor.next_multiple_of(BUFFER_ALIGNMENT);
            *cursor = offset + need.bytes;
            offset
        })
        .collect();
    assert!(
        sync_end <= SYNC_SCRATCH_BYTES,
        "sync scratch of one call exceeds the reserved {SYNC_SCRATCH_BYTES} bytes"
    );
    (offsets, end)
}

#[derive(Clone)]
enum NativeRoute {
    Cpu(cpu::CpuRoute),
    #[cfg(target_os = "macos")]
    Metal {
        opened: Arc<MetalOpened>,
        launches: Launches<Vec<seismic_metal::DirectPipeline>>,
    },
    Cuda {
        opened: Arc<CudaOpened>,
        launches: Launches<seismic_cuda::direct::DirectModule>,
    },
    #[cfg(not(target_os = "macos"))]
    Vulkan {
        opened: Arc<crate::backends::VulkanOpened>,
        launches: Launches<seismic_vulkan::formation::DirectModule>,
    },
}

impl NativeRoute {
    /// The route as serving holds it: what its launches run and nothing
    /// formed beside them. A Metal launch keeps its own pipeline
    /// ([`Launches::alone`]); a CUDA or Vulkan module is one loaded image
    /// of every entry it was compiled with and is held whole.
    fn served(&self) -> Self {
        match self {
            #[cfg(target_os = "macos")]
            Self::Metal { opened, launches } => Self::Metal {
                opened: opened.clone(),
                launches: launches.alone(),
            },
            other => other.clone(),
        }
    }
}

/// An implementation's launches in compiles formed on its device, by
/// launch ordinal: each launch's compile, its entry in it and the key of
/// its source's text. A whole-entry source is one compile with one entry
/// per launch; a launch-scoped source is one text per launch, whose
/// compile may hold more variants than the launch uses. The implementation
/// holds the compiles its launches run and no other of their texts. A
/// launch no call within the specialization's ranges activates is not
/// formed (`None`).
struct Launches<P> {
    launches: Vec<Option<(Arc<Compile<P>>, usize, ArtifactKey)>>,
}

impl<P> Clone for Launches<P> {
    fn clone(&self) -> Self {
        Self {
            launches: self.launches.clone(),
        }
    }
}

impl<P> Launches<P> {
    /// The program and entry of active launch `ordinal`. A launch is active
    /// only for a call within the specialization's ranges
    /// ([`NativePrepared::launches`] refuses any other), and every launch
    /// such a call can activate is formed.
    fn launch(&self, ordinal: usize) -> (&P, usize) {
        let (program, entry, _) = self
            .formed(ordinal)
            .expect("an active launch is formed: its call is within the specialization's ranges");
        (program, entry)
    }

    /// The program, entry ordinal and entry of launch `ordinal`, where it
    /// is formed.
    fn formed(&self, ordinal: usize) -> Option<(&P, usize, &ProgramEntry)> {
        self.launches[ordinal]
            .as_ref()
            .map(|(compile, entry, _)| (compile.program(), *entry, &compile.entries()[*entry]))
    }

    /// What each launch's program uses, in ordinal order, as `T`, the
    /// toolchain that formed it, reports it; none for a launch not formed.
    fn programs<T: Toolchain<Program = P>>(&self) -> Vec<Option<seismic_native_target::LaunchProgram>> {
        (0..self.launches.len())
            .map(|ordinal| {
                self.formed(ordinal).map(|(program, entry, at)| seismic_native_target::LaunchProgram {
                    threads: at.group_size.map(|size| size.iter().map(|extent| u64::from(*extent)).product()),
                    statistics: T::statistics(program, entry).to_vec(),
                })
            })
            .collect()
    }

    /// Each launch's program text address and entry, which with the
    /// toolchain identity determine the implementation's code.
    fn identify(&self, digest: &mut Sha256) {
        for launch in &self.launches {
            match launch {
                Some((compile, entry, key)) => {
                    digest.update(key.as_str());
                    digest.update(format!("{:?}", compile.entries()[*entry]));
                }
                None => digest.update(b"unformed"),
            }
        }
    }
}

#[cfg(target_os = "macos")]
impl Launches<Vec<seismic_metal::DirectPipeline>> {
    /// The same launches, each holding its own pipeline and no other of
    /// the compile it was formed in. A tuning compile forms many variants
    /// of a launch at once (up to 64), and a kernel that kept its compile
    /// kept every one of them for as long as it served, where a load from
    /// stored results forms the one it runs. A pipeline is its own object:
    /// holding one keeps nothing of the others.
    fn alone(&self) -> Self {
        Self {
            launches: self
                .launches
                .iter()
                .map(|launch| {
                    launch.as_ref().map(|(compile, entry, key)| {
                        (
                            Arc::new(Compile::of(
                                vec![compile.entries()[*entry].clone()],
                                vec![compile.program()[*entry].clone()],
                            )),
                            0,
                            key.clone(),
                        )
                    })
                })
                .collect(),
        }
    }

    fn pipeline(&self, ordinal: usize) -> &seismic_metal::DirectPipeline {
        let (pipelines, entry) = self.launch(ordinal);
        &pipelines[entry]
    }
}

/// Programs held so that implementations formed while they live share them:
/// a tuning run holds every launch variant it may assemble, and assembles
/// each from them ([`NativePrepared::assemble_held`]).
pub(crate) enum HeldPrograms {
    /// A backend without launch-scoped programs holds none.
    Nothing,
    /// A formed kernel's programs, held apart from the kernel.
    Route(#[allow(dead_code)] Box<dyn Send + Sync>),
    /// Each launch source formed with all its variants, by launch ordinal.
    #[cfg(target_os = "macos")]
    Metal(Vec<(usize, Formed<Vec<seismic_metal::DirectPipeline>>)>),
    Cuda(Vec<(usize, Formed<seismic_cuda::direct::DirectModule>)>),
}

/// The launches of `specialization` placed at the entries `held` formed:
/// each launch it reaches runs its variant's entry, which a held source of
/// that launch has. Nothing renders or forms.
fn held_launches<P>(
    implementation: &NativeImplementation,
    specialization: &NativeSpecialization,
    dialect: abi::Dialect,
    held: &[&[(usize, Formed<P>)]],
) -> Launches<P> {
    let launches = (0..implementation.launches.len())
        .map(|ordinal| {
            if !implementation.reachable(specialization, ordinal) {
                return None;
            }
            let entry = bounded(
                dialect,
                template_instance(
                    &implementation.launches[ordinal].kernel,
                    &plan::code_values(implementation, specialization, ordinal),
                ),
                implementation.static_group_size(specialization, ordinal),
            );
            let placed = held
                .iter()
                .flat_map(|sources| sources.iter())
                .filter(|(launch, _)| *launch == ordinal)
                .find_map(|(_, formed)| {
                    let (compile, position) = formed.holding(&entry)?;
                    Some((compile.clone(), position, formed.key().clone()))
                })
                .expect("a tuning run assembles only the launch variants it holds");
            Some(placed)
        })
        .collect();
    Launches { launches }
}

/// What was formed, for tuning records and measurement attribution: the
/// backend, entry, element bindings, specialization, rendered-source digest
/// and toolchain.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NativeArtifactIdentity(pub String);

/// One authored native implementation formed for one device, element
/// bindings and specialization.
pub struct NativePrepared {
    name: String,
    device: DeviceIdentity,
    public_device: Arc<DeviceInner>,
    /// The entry's call schema and parameter shapes: all a formed
    /// implementation reads of its logical entry once formed, so the
    /// entry's lowered semantics (its arena and program) are not kept.
    parameters: seismic_lang::entry::ParameterShapes,
    handoff: NativeHandoffKey,
    invocation: InvocationContract,
    implementation: NativeImplementation,
    specialization: NativeSpecialization,
    /// Static dimensions: name, invocation symbol and fixed value.
    statics: Vec<(String, SymbolId, u64)>,
    /// Tensor parameters (by schema ordinal) whose static extents render
    /// constant canonical strides (S10), with those strides.
    canonical_parameters: Vec<(usize, Vec<u64>)>,
    /// Whether a launch writes each buffer and whether its host rewrites it
    /// once it sees the launch, in ABI order ([`abi::buffer_writes`],
    /// [`abi::host_writes`]), which Metal's encoding orders by ([`hazard`]).
    buffer_writes: Vec<bool>,
    host_writes: Vec<bool>,
    /// Whether each buffer, in ABI order, is bound in the kernel's argument
    /// table ([`abi::bound`]): a host port is ordered by, never bound.
    bound: Vec<bool>,
    /// Per launch, in ordinal order, whether each buffer, in ABI order, is
    /// one of its record ports, bound as the table of its view's records
    /// ([`abi::records`]); the launches a call runs agree ([`Self::records`]).
    records: Vec<Vec<bool>>,
    /// Per launch, in ordinal order, the launch conditions the host writes
    /// the words of (`NativeLaunch::conditions`): each ABI
    /// buffer, the word of it and the value the launch is encoded under, a
    /// conjunction. The call or graph that binds the buffers resolves each
    /// to its host-written word ([`condition_holds`]).
    conditions: Vec<Vec<(usize, u64, seismic_lang::checked::Encoded)>>,
    results: Vec<NativeResult>,
    /// Scalar-result slots, of an entry with scalar results. Graph nodes
    /// publish no scalars, so only standalone calls write them.
    scalars: Option<ScalarSlots>,
    artifact: NativeArtifactIdentity,
    route: NativeRoute,
}

/// The device words an entry's scalar results are written to
/// ([`abi::scalar_word_count`]): held only by an entry with scalar results.
pub(crate) struct ScalarSlots {
    slots: Arc<Allocation>,
    words: usize,
    /// Standalone calls share the slots: it serializes their placement,
    /// reset, execution and read.
    calls: Mutex<()>,
}

impl ScalarSlots {
    /// The slots of `words` scalar-result words on `device`; none for none.
    fn allocate(device: &Arc<DeviceInner>, words: usize) -> Result<Option<Self>, ExecutionError> {
        match words {
            0 => Ok(None),
            words => Ok(Some(Self {
                slots: device.allocate(Self::bytes_of(words), ResultClass::WORD_BYTES)?,
                words,
                calls: Mutex::new(()),
            })),
        }
    }

    fn bytes_of(words: usize) -> u64 {
        words as u64 * ResultClass::WORD_BYTES
    }

    fn bytes(&self) -> u64 {
        Self::bytes_of(self.words)
    }
}

/// Formed resources awaiting serving preparation. This lives on the device,
/// so it must not own a prepared kernel's strong reference back to that device.
pub(crate) struct NativeHandoff {
    route: NativeRoute,
    scalars: Option<ScalarSlots>,
    artifact: NativeArtifactIdentity,
}

/// Exact formation request within one device. Stable entry labels identify
/// the unbound declaration, so they cannot distinguish element bindings.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct NativeHandoffKey {
    subject: Arc<CheckedProgramSubject>,
    entry: usize,
    specialization: NativeSpecialization,
    asset: Option<Arc<str>>,
    cpu_digest: Option<&'static str>,
}

impl NativeHandoffKey {
    fn new(
        module: &CheckedModule,
        entry: EntryId,
        logical: &LogicalEntry,
        specialization: &NativeSpecialization,
        backend: BackendName,
        cpu: Option<&'static CpuNativeKernels>,
    ) -> Self {
        Self {
            subject: logical.program().subject().clone(),
            entry: module
                .entries()
                .iter()
                .position(|candidate| candidate.id == entry)
                .expect("entry belongs to its module"),
            specialization: specialization.clone(),
            asset: module.native_asset(entry, backend).map(|asset| Arc::from(asset.text)),
            cpu_digest: if backend == BackendName::Cpu {
                cpu.map(|kernels| kernels.digest)
            } else {
                None
            },
        }
    }
}

/// The pipeline geometry of every launch of a Vulkan implementation: its
/// workgroup size and group-memory view lengths. Both read only static
/// dimensions and parameters (the checker's Vulkan rule), so every pipeline
/// is formed, and checked against the device, at preparation.
#[cfg(not(target_os = "macos"))]
fn vulkan_geometry(
    opened: &crate::backends::VulkanOpened,
    name: &str,
    implementation: &NativeImplementation,
    specialization: &NativeSpecialization,
) -> Result<Vec<([u32; 3], Vec<u32>)>, PrepareError> {
    let limits = opened.service().facts().limits;
    let properties = *opened.service().properties();
    let narrow = |value: u64, what: &str| {
        u32::try_from(value)
            .map_err(|_| preparation(format!("`{name}`: {what} {value} exceeds the Vulkan ABI")))
    };
    implementation
        .launches
        .iter()
        .enumerate()
        .map(|(ordinal, launch)| {
            let kernel = &launch.kernel;
            let evaluate = |expression: &NativeNatExpr| {
                expression
                    .evaluate(&|dimension| specialization.static_value(dimension), &|parameter| {
                        specialization.launch_param(ordinal, parameter).or_else(|| specialization.param(parameter))
                    })
                    .map_err(|error| preparation(format!("`{name}` launch `{kernel}`: {error}")))
            };
            let threads = [
                evaluate(&launch.group_extent[0])?,
                evaluate(&launch.group_extent[1])?,
                evaluate(&launch.group_extent[2])?,
            ];
            let invocations = threads.iter().try_fold(1u64, |product, value| product.checked_mul(*value));
            if threads.contains(&0)
                || invocations.is_none_or(|invocations| invocations > properties.max_group_threads.get())
                || threads.iter().zip(limits.max_group_size).any(|(threads, limit)| *threads > limit)
            {
                return Err(preparation(format!(
                    "launch `{kernel}` requests {threads:?} threads per workgroup; the device allows {:?} and {} in total",
                    limits.max_group_size, properties.max_group_threads.get()
                )));
            }
            let shared_bytes = evaluate(&launch.shared_bytes)?;
            let footprint = abi::vulkan::shared_footprint(shared_bytes);
            if footprint > properties.shared_bytes_per_group.get() {
                return Err(preparation(format!(
                    "launch `{kernel}` needs {footprint} shared bytes; the device allows {}",
                    properties.shared_bytes_per_group.get()
                )));
            }
            Ok((
                [
                    narrow(threads[0], "a workgroup extent")?,
                    narrow(threads[1], "a workgroup extent")?,
                    narrow(threads[2], "a workgroup extent")?,
                ],
                abi::vulkan::shared_view_lengths(shared_bytes)
                    .into_iter()
                    .map(|length| narrow(length, "a shared view length"))
                    .collect::<Result<_, _>>()?,
            ))
        })
        .collect()
}

/// `implementation` as `device` offers it, bound on it: on a CPU device
/// with the Seismic-owned participant count of each launch and the tier
/// appended, their domains exact for the device (its pool size, the tiers
/// below its detected one), and every `device.NAME` it reads at the
/// device's value ([`NativeImplementation::at_device`]). Idempotent. The
/// one binder of an implementation on a device: every other path binds
/// through it ([`offered`]).
pub fn on_device(
    device: &DeviceInner,
    mut implementation: NativeImplementation,
) -> OnDevice {
    use seismic_lang::checked::{NativeParameter, NativeParameterRole};
    if let OpenedKind::Cpu(opened) = &device.kind {
        if implementation
            .params
            .iter()
            .all(|parameter| parameter.role == NativeParameterRole::Declared)
        {
            let lower = seismic_native_cpu::Tier::detected()
                .map(|detected| {
                    detected
                        .at_or_below()
                        .filter(|tier| *tier != detected)
                        .map(|tier| tier.name())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            implementation.params.extend(NativeParameter::cpu_parameters(
                implementation.launches.len(),
                opened.executor().workers(),
                &lower,
            ));
        }
    }
    implementation.at_device(device.properties())
}

/// `implementation`'s declared defaults as `device` offers them at
/// `bindings`. On a CPU whose tier forms quantized dots from vector integer
/// products (AVX2, AVX-512 or VNNI, `Tier::quantized_dots`, a fact of the
/// device), a projection over a packed weight the quantized dot takes
/// (`form(W, INT8)`) defaults to its `INT8` form: it quantizes the staged
/// rows to Q8 blocks and runs the representation's integer dot, and the F32
/// form is the challenger; where `where` refuses it, the walk from the
/// defaults takes the F32 form. Elsewhere (x86-64-v2, NEON, dense weights,
/// GPUs) the declaration's defaults stand.
pub(crate) fn device_defaults(
    device: &DeviceInner,
    mut implementation: NativeImplementation,
    bindings: &ElementBindings,
) -> NativeImplementation {
    use seismic_lang::checked::{NativeDefault, NativeNatExpr, NativeParameterRole};
    let OpenedKind::Cpu(_) = &device.kind else {
        return implementation;
    };
    if !seismic_native_cpu::Tier::detected().is_some_and(|tier| tier.quantized_dots() > 0) {
        return implementation;
    }
    let packed = bindings.iter().any(|(_, representation)| {
        matches!(
            seismic_lang::registry::representation_info(representation).kind,
            seismic_lang::registry::RepresentationKind::PackedRows(_)
        ) && seismic_lang::forms::cpu_int8_of(representation)
    });
    if !packed {
        return implementation;
    }
    for parameter in &mut implementation.params {
        if parameter.role == NativeParameterRole::Declared
            && parameter.form
            && parameter.name == "INT8"
            && parameter.values.contains(&1)
        {
            parameter.default = NativeDefault::Value(NativeNatExpr::Constant(1));
        }
    }
    implementation
}

/// `entry`'s implementation at `bindings` ([`CheckedModule::native_implementation_at`])
/// as `device` offers it ([`device_defaults`], [`on_device`]): the
/// declaration whose configurations preparation forms at these bindings,
/// bound on the device. `None` when the backend has no implementation.
pub fn offered(
    device: &DeviceInner,
    module: &CheckedModule,
    entry: EntryId,
    bindings: &ElementBindings,
) -> Option<OnDevice> {
    module
        .native_implementation_at(entry, backend_name(&device.kind), bindings)
        .cloned()
        .map(|implementation| on_device(device, device_defaults(device, implementation, bindings)))
}

/// Check a choice against the implementation as it exists on this device
/// at `bindings` ([`offered`]), as preparation takes it
/// ([`NativeImplementation::prepared`]): CPU worker and ISA parameters are
/// Seismic-owned additions to the checked declaration, which a cached
/// choice carries and a caller's may leave at their defaults.
pub fn specialization_valid(
    device: &Arc<DeviceInner>,
    module: &CheckedModule,
    entry: EntryId,
    bindings: &ElementBindings,
    specialization: &NativeSpecialization,
) -> bool {
    offered(device, module, entry, bindings)
        .is_some_and(|implementation| implementation.prepared(specialization.clone()).is_ok())
}

/// The default specialization of `entry` at `bindings` and `statics` as
/// `device` offers the implementation ([`offered`]): on a CPU device it
/// also values the Seismic-owned worker counts and tier, which a stored
/// choice carries and the bare declaration's default lacks. `None` when the
/// backend has no implementation or nothing is admissible at `statics`.
pub fn default_specialization(
    device: &Arc<DeviceInner>,
    module: &CheckedModule,
    entry: EntryId,
    bindings: &ElementBindings,
    statics: &NativeSpecialization,
) -> Option<NativeSpecialization> {
    offered(device, module, entry, bindings)
        .and_then(|implementation| implementation.default_specialization(statics).ok())
}

/// The CPU route of `implementation` (as [`on_device`] offers it) for a
/// validated `specialization`: the tier it runs at, one function per launch
/// for that tier and the bound dense elements, the participants of each
/// launch and the declared parameter values.
fn cpu_route(
    opened: &Arc<CpuOpened>,
    name: &str,
    kernels: &CpuNativeKernels,
    implementation: &NativeImplementation,
    specialization: &NativeSpecialization,
    bindings: &ElementBindings,
    domain: &seismic_lang::checked::ElementDomain,
) -> Result<cpu::CpuRoute, PrepareError> {
    use seismic_lang::checked::NativeParameterRole;
    use seismic_native_cpu::Tier;
    let mut tier = Tier::detected().ok_or_else(|| {
        preparation(format!(
            "`{name}`: the host has no CPU instruction-set tier"
        ))
    })?;
    let mut workers = vec![0usize; implementation.launches.len()];
    let mut params = Vec::new();
    for parameter in &implementation.params {
        let value = specialization
            .param(&parameter.name)
            .expect("validated specialization values every parameter");
        match parameter.role {
            NativeParameterRole::Declared => params.push(value),
            NativeParameterRole::Workers { launch } => workers[launch as usize] = value as usize,
            NativeParameterRole::Tier if value == 0 => {}
            NativeParameterRole::Tier => {
                let named = NativeParameterRole::TIERS[value as usize - 1];
                tier = Tier::all()
                    .iter()
                    .copied()
                    .find(|tier| tier.name() == named)
                    .expect("the device offers only its own tiers");
            }
        }
    }
    // Every weight is bound in a form the tier's components read where it
    // lies (a CPU row layout, or a device's).
    for (element, representation) in bindings.iter() {
        if !domain.read_as_weight(element) {
            continue;
        }
        if seismic_native_cpu::WeightOperand::of(tier, representation).is_none() {
            let bound = seismic_lang::registry::representation_info(representation).name;
            return Err(preparation(format!(
                "`{name}`: weight `{element}` is bound to `{bound}`, which no `{}` component reads",
                tier.name()
            )));
        }
    }
    let elements = kernels
        .elements
        .iter()
        .map(|parameter| {
            let representation = bindings
                .get(parameter)
                .expect("admitted bindings bind every element parameter");
            seismic_lang::registry::representation_info(representation).name
        })
        .collect::<Vec<_>>();
    let launches = (0..implementation.launches.len())
        .map(|launch| {
            kernels.function(launch, tier, &elements).ok_or_else(|| {
                preparation(format!(
                    "`{name}` launch {launch} has no CPU form for tier `{}` and elements {elements:?}",
                    tier.name()
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(cpu::CpuRoute {
        opened: opened.clone(),
        tier,
        launches,
        workers,
        params,
    })
}

fn entry_info(module: &CheckedModule, entry: EntryId) -> &seismic_lang::checked::EntryInfo {
    module
        .entries()
        .iter()
        .find(|candidate| candidate.id == entry)
        .expect("entry belongs to its module")
}

fn entry_name(module: &CheckedModule, entry: EntryId) -> String {
    entry_info(module, entry).name.clone()
}

fn preparation(message: impl Into<String>) -> PrepareError {
    PrepareError::Preparation(PreparationError::NativeSpecialization(message.into()))
}

fn compilation(error: seismic_native_target::NativeCompilationError) -> PrepareError {
    PrepareError::Preparation(PreparationError::NativeCompilation(error))
}

/// `entry` with the group size its launch runs, on Metal: the pipeline is
/// formed to admit that many threads, so its register allocation never lowers
/// the limit below the declared size.
fn bounded(
    dialect: abi::Dialect,
    mut entry: ProgramEntry,
    group_size: Option<[u64; 3]>,
) -> ProgramEntry {
    if matches!(dialect, abi::Dialect::Metal(_)) {
        entry.group_size = group_size.and_then(|size| {
            Some([
                u32::try_from(size[0]).ok()?,
                u32::try_from(size[1]).ok()?,
                u32::try_from(size[2]).ok()?,
            ])
        });
    }
    entry
}

/// The entry of one code variant of `kernel`: the kernel itself, or its
/// template instance for the variant's code values.
fn template_instance(kernel: &str, values: &[u64]) -> ProgramEntry {
    ProgramEntry::named(if values.is_empty() {
        kernel.to_owned()
    } else {
        format!(
            "{kernel}<{}>",
            values
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        )
    })
}

/// The dialect of an opened Metal device's native sources: MSL with its
/// feature macros.
#[cfg(target_os = "macos")]
pub(crate) fn metal_dialect(opened: &crate::backends::MetalOpened) -> abi::Dialect {
    abi::Dialect::Metal(abi::MetalFeatures {
        tensor_ops: opened.device_description().facts().properties().tensor_ops.holds(),
    })
}

/// The dialect of an opened CUDA device's native sources: CUDA C++ with its
/// fact macros.
pub(crate) fn cuda_dialect(opened: &crate::backends::CudaOpened) -> abi::Dialect {
    abi::Dialect::Cuda(abi::CudaFeatures {
        multiprocessors: opened.device_description().facts().properties.units.get(),
    })
}

/// One launch's program with `variants` as its entries. Every variant of a
/// launch shares its source, so tuning forms all of them in one compile and
/// preparation's single variant is served from that program. Variants that
/// differ only in a group size a backend does not form for share an entry.
fn launch_program(
    dialect: abi::Dialect,
    logical: &LogicalEntry,
    bindings: &ElementBindings,
    implementation: &NativeImplementation,
    specialization: &NativeSpecialization,
    asset: seismic_lang::checked::NativeAsset<'_>,
    ordinal: usize,
    variants: &[plan::LaunchVariant],
) -> ProgramSource {
    let kernel = &implementation.launches[ordinal].kernel;
    let mut entries = Vec::<ProgramEntry>::new();
    for variant in variants {
        let entry = bounded(
            dialect,
            template_instance(kernel, &variant.code),
            variant.group_size,
        );
        if !entries.contains(&entry) {
            entries.push(entry);
        }
    }
    ProgramSource {
        text: abi::render_launch_source(
            dialect,
            logical,
            bindings,
            implementation,
            specialization,
            asset,
            ordinal,
        ),
        entries,
    }
}

/// The programs of a Metal or CUDA implementation at `specialization`: one
/// per launch for launch-scoped parameters, holding `variants[launch]` as its
/// entries, else the entry's whole source with one entry per launch.
pub(crate) fn implementation_programs(
    dialect: abi::Dialect,
    logical: &LogicalEntry,
    bindings: &ElementBindings,
    implementation: &NativeImplementation,
    specialization: &NativeSpecialization,
    asset: seismic_lang::checked::NativeAsset<'_>,
    variants: &[Vec<plan::LaunchVariant>],
) -> Vec<ProgramSource> {
    if implementation.launch_scoped() {
        return (0..implementation.launches.len())
            .map(|ordinal| {
                launch_program(
                    dialect,
                    logical,
                    bindings,
                    implementation,
                    specialization,
                    asset,
                    ordinal,
                    &variants[ordinal],
                )
            })
            .collect();
    }
    let entries = implementation
        .launches
        .iter()
        .enumerate()
        .map(|(ordinal, launch)| {
            bounded(
                dialect,
                ProgramEntry::named(launch.kernel.as_str()),
                implementation.static_group_size(specialization, ordinal),
            )
        })
        .collect();
    vec![ProgramSource {
        text: abi::render_source(
            dialect,
            logical,
            bindings,
            implementation,
            specialization,
            asset,
        ),
        entries,
    }]
}

/// The program sources a device's toolchain forms `implementation` at
/// `specialization` from, and by launch the source and the entry of it the
/// launch runs: each launch's single variant, the specialization's code
/// values and static group size. A launch no call within the
/// specialization's ranges activates has no entry and no source of its own,
/// so nothing of it is formed: a backend forms a pipeline or function per
/// entry, and on some drivers that is most of the compile. The CPU backend
/// renders nothing and is never asked.
fn rendered_programs(
    device: &Arc<DeviceInner>,
    module: &CheckedModule,
    entry: EntryId,
    logical: &LogicalEntry,
    bindings: &ElementBindings,
    implementation: &NativeImplementation,
    specialization: &NativeSpecialization,
) -> Result<(Vec<ProgramSource>, Vec<Option<(usize, ProgramEntry)>>), PrepareError> {
    let reachable = (0..implementation.launches.len())
        .map(|ordinal| implementation.reachable(specialization, ordinal))
        .collect::<Vec<_>>();
    // A whole-entry source keeps the entries of its reachable launches, and
    // each reachable launch runs its own entry of it.
    let whole = |mut source: ProgramSource| {
        let requests = source
            .entries
            .iter()
            .zip(&reachable)
            .map(|(entry, &reachable)| reachable.then(|| (0, entry.clone())))
            .collect::<Vec<_>>();
        source.entries = requests
            .iter()
            .flatten()
            .map(|(_, entry)| entry.clone())
            .collect();
        (vec![source], requests)
    };
    let asset = |backend| {
        module.native_asset(entry, backend).ok_or_else(|| {
            preparation(format!(
                "native asset of `{}` is absent from the module",
                entry_name(module, entry)
            ))
        })
    };
    let variants = (0..implementation.launches.len())
        .map(|ordinal| {
            vec![plan::LaunchVariant {
                code: plan::code_values(implementation, specialization, ordinal),
                group_size: implementation.static_group_size(specialization, ordinal),
            }]
        })
        .collect::<Vec<_>>();
    let programs = |dialect: abi::Dialect, asset: seismic_lang::checked::NativeAsset<'_>| {
        let sources = implementation_programs(
            dialect,
            logical,
            bindings,
            implementation,
            specialization,
            asset,
            &variants,
        );
        if implementation.launch_scoped() {
            // One source per launch: the reachable launches' sources, each
            // launch at its own.
            let mut kept = Vec::new();
            let requests = sources
                .into_iter()
                .zip(&reachable)
                .map(|(source, &reachable)| {
                    reachable.then(|| {
                        let request = (kept.len(), source.entries[0].clone());
                        kept.push(source);
                        request
                    })
                })
                .collect();
            (kept, requests)
        } else {
            whole(sources.into_iter().next().expect(
                "`implementation_programs` renders a whole-entry implementation as one source",
            ))
        }
    };
    match &device.kind {
        OpenedKind::Cpu(_) => unreachable!("the CPU backend renders no programs"),
        #[cfg(target_os = "macos")]
        OpenedKind::Metal(opened) => {
            Ok(programs(metal_dialect(opened), asset(BackendName::Metal)?))
        }
        OpenedKind::Cuda(opened) => Ok(programs(cuda_dialect(opened), asset(BackendName::Cuda)?)),
        #[cfg(not(target_os = "macos"))]
        OpenedKind::Vulkan(opened) => {
            let geometry = vulkan_geometry(
                opened,
                &entry_name(module, entry),
                implementation,
                specialization,
            )?;
            // Each entry's constants are its workgroup size, then its
            // group-memory views: the pipeline's specialization.
            let source = ProgramSource {
                text: abi::render_source(
                    abi::Dialect::Vulkan(opened.features()),
                    logical,
                    bindings,
                    implementation,
                    specialization,
                    asset(BackendName::Vulkan)?,
                ),
                entries: implementation
                    .launches
                    .iter()
                    .zip(&geometry)
                    .map(|(launch, (threads, views))| ProgramEntry {
                        symbol: launch.kernel.clone(),
                        group_size: Some(*threads),
                        constants: views.clone(),
                    })
                    .collect(),
            };
            Ok(whole(source))
        }
    }
}

/// The programs a formation forms, rendered: what its
/// class is read from ([`ProgramFormer::class`]). Rendering depends only on
/// what is formed, so it is kept; the class is read again whenever it is
/// asked, because what is live on the device and kept by the store changes
/// with every unit's work, not only with the work of the unit asking. Units
/// whose launches render the same text share its live compiles: a class
/// kept from when another unit held such a compile read as kept after that
/// unit had given it up (a GB10 load, 2026-10-12, priced compiles that took
/// 0.45 s at nothing). Programs that cannot be rendered are classed as
/// compiled with every entry they name: forming them fails as it starts,
/// and nothing was learned of what it would have taken.
#[derive(Clone)]
pub(crate) enum Rendered {
    Sources(Vec<Keyed>),
    Unrendered(usize),
}

impl Rendered {
    /// What forming the programs on `device` does now.
    pub(crate) fn class(&self, device: &DeviceInner) -> (FormationClass, usize) {
        match self {
            Self::Sources(sources) => device.programs.class(sources),
            Self::Unrendered(entries) => (FormationClass::Compiled, *entries),
        }
    }

}

/// The programs `sources` of a launch-scoped implementation are formed
/// from, one per launch with all its variants; none on a backend without
/// launch-scoped programs.
fn launch_programs(
    device: &Arc<DeviceInner>,
    module: &CheckedModule,
    entry: EntryId,
    bindings: &ElementBindings,
    statics: &NativeSpecialization,
    implementation: &NativeImplementation,
    sources: &[plan::LaunchSource],
) -> Result<Option<Vec<ProgramSource>>, PrepareError> {
    let backend = backend_name(&device.kind);
    let dialect = match &device.kind {
        #[cfg(target_os = "macos")]
        OpenedKind::Metal(opened) => metal_dialect(opened),
        OpenedKind::Cuda(opened) => cuda_dialect(opened),
        _ => return Ok(None),
    };
    let logical = module
        .entry(entry, bindings)
        .map_err(PrepareError::Source)?;
    let asset = module.native_asset(entry, backend).ok_or_else(|| {
        preparation(format!(
            "native asset of `{}` is absent from the module",
            entry_name(module, entry)
        ))
    })?;
    Ok(Some(
        sources
            .iter()
            .map(|source| {
                launch_program(
                    dialect,
                    &logical,
                    bindings,
                    implementation,
                    statics,
                    asset,
                    source.ordinal,
                    &source.variants,
                )
            })
            .collect(),
    ))
}

/// Form `sources` and place each launch at its entry: `requests` names, by
/// launch ordinal, the source and entry it runs (none for a launch that is
/// not formed).
fn form_launches<T: Toolchain>(
    former: &ProgramFormer<T>,
    sources: &[ProgramSource],
    requests: &[Option<(usize, ProgramEntry)>],
) -> Result<Launches<T::Program>, PrepareError> {
    let formed = former.form(sources).map_err(compilation)?;
    let launches = requests
        .iter()
        .map(|request| {
            request.as_ref().map(|(source, entry)| {
                let (compile, position) = formed[*source].entry(entry);
                (compile.clone(), position, formed[*source].key().clone())
            })
        })
        .collect();
    Ok(Launches { launches })
}

pub(crate) fn backend_name(kind: &OpenedKind) -> BackendName {
    match kind {
        OpenedKind::Cpu(_) => BackendName::Cpu,
        #[cfg(target_os = "macos")]
        OpenedKind::Metal(_) => BackendName::Metal,
        OpenedKind::Cuda(_) => BackendName::Cuda,
        #[cfg(not(target_os = "macos"))]
        OpenedKind::Vulkan(_) => BackendName::Vulkan,
    }
}

impl NativePrepared {
    /// What each launch's program uses, in ordinal order: its workgroup's
    /// threads and what its backend's compiler reports
    /// ([`Toolchain::statistics`]: CUDA's function attributes, Vulkan's
    /// pipeline executable statistics); none for a launch not formed, and
    /// nothing of the CPU backend, which forms no programs.
    pub fn launch_programs(&self) -> Vec<Option<seismic_native_target::LaunchProgram>> {
        match &self.route {
            NativeRoute::Cpu(_) => Vec::new(),
            #[cfg(target_os = "macos")]
            NativeRoute::Metal { launches, .. } => launches.programs::<seismic_metal::toolchain::MetalToolchain>(),
            NativeRoute::Cuda { launches, .. } => launches.programs::<seismic_cuda::toolchain::CudaToolchain>(),
            #[cfg(not(target_os = "macos"))]
            NativeRoute::Vulkan { launches, .. } => launches.programs::<seismic_vulkan::toolchain::VulkanToolchain>(),
        }
    }

    /// Hand this kernel's formed programs, chosen by tuning, to the serving
    /// preparation of its specialization on its device: serving gets its own
    /// kernel over the same programs, with fresh call state, and forms
    /// nothing.
    pub(crate) fn hand_off(&self) -> Result<(), ExecutionError> {
        self.hand_off_replacing(None)
    }

    /// Hand this kernel off as its tuning unit's choice in place of
    /// `previous`, the unit's choice until now. A unit's handoff is its one
    /// current published choice: under one hold of the device's lock the
    /// previous handoff is withdrawn with its device allocation and this
    /// one takes its place, so no handoff that serving will never take is
    /// left on the device.
    pub(crate) fn hand_off_replacing(
        &self,
        previous: Option<&NativePrepared>,
    ) -> Result<(), ExecutionError> {
        let scalars = ScalarSlots::allocate(&self.public_device, self.scalar_words())?;
        let served = NativeHandoff {
            scalars,
            artifact: self.artifact.clone(),
            route: self.route.served(),
        };
        let mut chosen = self
            .public_device
            .chosen
            .lock()
            .expect("chosen-kernel lock is never poisoned");
        if let Some(previous) = previous {
            chosen.remove(&previous.handoff);
        }
        chosen.insert(self.handoff.clone(), served);
        Ok(())
    }

    /// This kernel's formed programs, held apart from the kernel: while
    /// they are held, forming its specialization again forms nothing, and
    /// none of the kernel's call state (its scalar slots and scratch) is
    /// kept with them.
    pub(crate) fn programs(&self) -> HeldPrograms {
        HeldPrograms::Route(Box::new(self.route.clone()))
    }

    /// Form each launch in `sources` with all its code variants, one
    /// program per launch, and hold the programs, so that each
    /// specialization a tuning run assembles from them while they are held
    /// forms nothing. Backends without launch-scoped programs hold none.
    pub(crate) fn hold_launch_variants(
        device: &Arc<DeviceInner>,
        module: &CheckedModule,
        entry: EntryId,
        bindings: &ElementBindings,
        statics: &NativeSpecialization,
        implementation: &NativeImplementation,
        sources: &[plan::LaunchSource],
    ) -> Result<HeldPrograms, PrepareError> {
        let Some(programs) = launch_programs(
            device,
            module,
            entry,
            bindings,
            statics,
            implementation,
            sources,
        )?
        else {
            return Ok(HeldPrograms::Nothing);
        };
        let ordinals = || sources.iter().map(|source| source.ordinal);
        Ok(match &device.programs {
            #[cfg(target_os = "macos")]
            crate::backends::DevicePrograms::Metal(former) => HeldPrograms::Metal(
                ordinals()
                    .zip(former.form(&programs).map_err(compilation)?)
                    .collect(),
            ),
            crate::backends::DevicePrograms::Cuda(former) => HeldPrograms::Cuda(
                ordinals()
                    .zip(former.form(&programs).map_err(compilation)?)
                    .collect(),
            ),
            _ => unreachable!("only Metal and CUDA form launch-scoped programs"),
        })
    }

    /// The programs [`NativePrepared::hold_launch_variants`] of `sources`
    /// forms, what its class is read from ([`Rendered`]). A formation that
    /// cannot be rendered names every variant of its launches.
    pub(crate) fn launch_variants_rendered(
        device: &Arc<DeviceInner>,
        module: &CheckedModule,
        entry: EntryId,
        bindings: &ElementBindings,
        statics: &NativeSpecialization,
        implementation: &NativeImplementation,
        sources: &[plan::LaunchSource],
    ) -> Rendered {
        match launch_programs(
            device,
            module,
            entry,
            bindings,
            statics,
            implementation,
            sources,
        ) {
            Ok(programs) => Rendered::Sources(device.programs.keyed(programs.unwrap_or_default())),
            Err(_) => Rendered::Unrendered(
                sources
                    .iter()
                    .map(|source| source.variants.len())
                    .max()
                    .unwrap_or(0),
            ),
        }
    }

    /// The programs [`NativePrepared::prepare_implementation`] of
    /// `specialization` forms, what its class is read from ([`Rendered`]):
    /// none on the CPU backend, which compiles nothing at run time.
    pub(crate) fn implementation_rendered(
        device: &Arc<DeviceInner>,
        module: &CheckedModule,
        entry: EntryId,
        logical: &LogicalEntry,
        bindings: &ElementBindings,
        specialization: &NativeSpecialization,
        implementation: &NativeImplementation,
    ) -> Rendered {
        if matches!(device.kind, OpenedKind::Cpu(_)) {
            return Rendered::Sources(Vec::new());
        }
        let implementation = on_device(device, implementation.clone());
        let specialization = implementation.with_owned_defaults(specialization.clone());
        match rendered_programs(
            device,
            module,
            entry,
            logical,
            bindings,
            &implementation,
            &specialization,
        ) {
            Ok((sources, _)) => Rendered::Sources(device.programs.keyed(sources)),
            Err(_) => Rendered::Unrendered(implementation.launches.len()),
        }
    }

    /// Form the entry's native implementation for `device`'s backend.
    /// `cpu` carries the compiled CPU launch functions when the build
    /// generated them.
    pub(crate) fn prepare(
        device: &Arc<DeviceInner>,
        module: &CheckedModule,
        entry: EntryId,
        bindings: ElementBindings,
        specialization: NativeSpecialization,
        cpu: Option<&'static CpuNativeKernels>,
    ) -> Result<Arc<Self>, PrepareError> {
        let backend = backend_name(&device.kind);
        let implementation = module
            .native_implementation(entry, backend)
            .cloned()
            .ok_or_else(|| {
                preparation(format!(
                    "`{}` has no native implementation for `{}`",
                    entry_name(module, entry),
                    backend.as_str()
                ))
            })?;
        let logical = Arc::new(
            module
                .entry(entry, &bindings)
                .map_err(PrepareError::Source)?,
        );
        Self::prepare_implementation(
            device,
            module,
            entry,
            &logical,
            bindings,
            specialization,
            cpu,
            implementation,
        )
    }

    /// Form `implementation` of the entry, which may differ from the
    /// module's declaration in its parameter domains only (a tuning
    /// survey's widened domains).
    /// `logical` is the entry at `bindings`, built once by the caller for
    /// every configuration it forms.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare_implementation(
        device: &Arc<DeviceInner>,
        module: &CheckedModule,
        entry: EntryId,
        logical: &Arc<LogicalEntry>,
        bindings: ElementBindings,
        specialization: NativeSpecialization,
        cpu: Option<&'static CpuNativeKernels>,
        implementation: NativeImplementation,
    ) -> Result<Arc<Self>, PrepareError> {
        Self::prepare_formed(
            device,
            module,
            entry,
            logical,
            bindings,
            specialization,
            cpu,
            implementation,
            None,
        )
    }

    /// [`NativePrepared::prepare_implementation`] of a launch-scoped
    /// specialization a tuning run assembles from the launch variants it
    /// holds (`held`, [`NativePrepared::hold_launch_variants`]): its
    /// launches are placed at the held entries, and no source is rendered
    /// or formed.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn assemble_held(
        device: &Arc<DeviceInner>,
        module: &CheckedModule,
        entry: EntryId,
        logical: &Arc<LogicalEntry>,
        bindings: ElementBindings,
        specialization: NativeSpecialization,
        cpu: Option<&'static CpuNativeKernels>,
        implementation: NativeImplementation,
        held: &[&HeldPrograms],
    ) -> Result<Arc<Self>, PrepareError> {
        Self::prepare_formed(
            device,
            module,
            entry,
            logical,
            bindings,
            specialization,
            cpu,
            implementation,
            Some(held),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_formed(
        device: &Arc<DeviceInner>,
        module: &CheckedModule,
        entry: EntryId,
        logical: &Arc<LogicalEntry>,
        bindings: ElementBindings,
        specialization: NativeSpecialization,
        cpu: Option<&'static CpuNativeKernels>,
        implementation: NativeImplementation,
        held: Option<&[&HeldPrograms]>,
    ) -> Result<Arc<Self>, PrepareError> {
        let backend = backend_name(&device.kind);
        let name = entry_name(module, entry);
        let implementation = on_device(device, implementation)
            .at_elements(&bindings)
            .ok_or_else(|| {
                preparation(format!(
                    "`{name}` has no `{}` implementation at these element bindings",
                    backend.as_str()
                ))
            })?;
        let scoped = implementation.launch_scoped();
        if scoped && !matches!(backend, BackendName::Metal | BackendName::Cuda) {
            return Err(preparation(format!(
                "`{name}` has launch-scoped parameters but per-launch native formation is not available for `{}`",
                backend.as_str()
            )));
        }
        let specialization = implementation
            .prepared(specialization)
            .map_err(|error| preparation(format!("`{name}`: {error}")))?;
        // A range holds a call-time dimension of the entry; the declaration
        // alone does not know them.
        let dimensions = &entry_info(module, entry).dimensions;
        if let Some(ranged) = specialization
            .ranges()
            .keys()
            .find(|ranged| !dimensions.contains(ranged))
        {
            return Err(preparation(format!(
                "`{name}`: `{ranged}` is not a dimension of the entry and takes no range"
            )));
        }
        let handoff = NativeHandoffKey::new(module, entry, logical, &specialization, backend, cpu);
        // A specialization tuning already formed and chose is served as
        // formed: it is never formed twice.
        let handed_off = device
            .chosen
            .lock()
            .expect("chosen-kernel lock is never poisoned")
            .remove(&handoff);
        let schema = logical.schema();
        let statics = implementation
            .statics
            .iter()
            .map(|dimension| {
                let symbol = schema
                    .dimensions()
                    .iter()
                    .find(|candidate| &candidate.name == dimension)
                    .expect("checked static names an entry dimension")
                    .symbol;
                let value = specialization
                    .static_value(dimension)
                    .expect("validated specialization fixes every static dimension");
                (dimension.clone(), symbol, value)
            })
            .collect::<Vec<_>>();
        let invocation = InvocationContract::compile_entry(&logical);
        let results = schema
            .results()
            .iter()
            .map(|result| match &result.kind {
                ResultKind::Tensor {
                    representation,
                    axes,
                } => NativeResult::Tensor {
                    representation: *representation,
                    axes: axes
                        .iter()
                        .map(|axis| logical.arena().compile_nat(*axis))
                        .collect(),
                },
                ResultKind::Scalar(dtype) => NativeResult::Scalar(*dtype),
                ResultKind::Index { .. } => NativeResult::Index,
                ResultKind::Range { .. } => NativeResult::Range,
            })
            .collect::<Vec<_>>();
        let words = abi::native_word_count(schema, &implementation);
        let scalar_words = abi::scalar_word_count(schema);
        let rendered = || {
            rendered_programs(
                device,
                module,
                entry,
                logical,
                &bindings,
                &implementation,
                &specialization,
            )
        };
        let form = || -> Result<_, PrepareError> {
            let mut digest = Sha256::new();
            let (route, toolchain) = match &device.kind {
                OpenedKind::Cpu(opened) => {
                    let kernels = cpu.ok_or_else(|| {
                        preparation(format!(
                            "`{name}` has no compiled CPU native functions in this build"
                        ))
                    })?;
                    let route = cpu_route(
                        opened,
                        &name,
                        kernels,
                        &implementation,
                        &specialization,
                        &bindings,
                        &entry_info(module, entry).element_domain,
                    )?;
                    digest.update(kernels.digest.as_bytes());
                    let toolchain = format!(
                        "cpu;{};{};{}",
                        std::env::consts::ARCH,
                        route.tier.name(),
                        seismic_native_cpu::VERSION
                    );
                    (NativeRoute::Cpu(route), toolchain)
                }
                #[cfg(target_os = "macos")]
                OpenedKind::Metal(opened) => {
                    let slots = abi::buffer_slots(schema, &implementation);
                    if slots > seismic_metal::DIRECT_BUFFER_SLOTS {
                        return Err(PrepareError::Preparation(
                            PreparationError::NativeBufferSlots {
                                entry: name.clone(),
                                slots,
                                limit: seismic_metal::DIRECT_BUFFER_SLOTS,
                            },
                        ));
                    }
                    if words * 8 > seismic_metal::DIRECT_WORD_BYTES_LIMIT {
                        return Err(preparation(format!(
                            "`{name}` needs {} argument bytes, beyond Metal's setBytes limit",
                            words * 8
                        )));
                    }
                    let former = device.programs.metal();
                    let launches = match held {
                        Some(held) => held_launches(
                            &implementation,
                            &specialization,
                            metal_dialect(opened),
                            &held
                                .iter()
                                .filter_map(|held| match held {
                                    HeldPrograms::Metal(sources) => Some(sources.as_slice()),
                                    _ => None,
                                })
                                .collect::<Vec<_>>(),
                        ),
                        None => {
                            let (sources, requests) = rendered()?;
                            form_launches(former, &sources, &requests)?
                        }
                    };
                    launches.identify(&mut digest);
                    (
                        NativeRoute::Metal {
                            opened: opened.clone(),
                            launches,
                        },
                        former.toolchain().identity().material.clone(),
                    )
                }
                OpenedKind::Cuda(opened) => {
                    let former = device.programs.cuda();
                    let launches = match held {
                        Some(held) => held_launches(
                            &implementation,
                            &specialization,
                            cuda_dialect(opened),
                            &held
                                .iter()
                                .filter_map(|held| match held {
                                    HeldPrograms::Cuda(sources) => Some(sources.as_slice()),
                                    _ => None,
                                })
                                .collect::<Vec<_>>(),
                        ),
                        None => {
                            let (sources, requests) = rendered()?;
                            form_launches(former, &sources, &requests)?
                        }
                    };
                    launches.identify(&mut digest);
                    // Measurements depend on the driver the CUBIN runs under.
                    let toolchain = format!(
                        "{};driver {}",
                        former.toolchain().identity().material,
                        opened.device_description().facts().driver_api.0
                    );
                    (
                        NativeRoute::Cuda {
                            opened: opened.clone(),
                            launches,
                        },
                        toolchain,
                    )
                }
                #[cfg(not(target_os = "macos"))]
                OpenedKind::Vulkan(opened) => {
                    let (sources, requests) = rendered()?;
                    let former = device.programs.vulkan();
                    let launches = form_launches(former, &sources, &requests)?;
                    launches.identify(&mut digest);
                    (
                        NativeRoute::Vulkan {
                            opened: opened.clone(),
                            launches,
                        },
                        former.toolchain().identity().material.clone(),
                    )
                }
            };
            let bindings_text = bindings
                .iter()
                .map(|(parameter, representation)| {
                    format!(
                        "{parameter}={}",
                        seismic_lang::registry::representation_info(representation).name
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            let artifact = NativeArtifactIdentity(format!(
                "{};{name};{bindings_text};{specialization:?};{};{toolchain}",
                backend.as_str(),
                crate::telemetry::hex(&digest.finalize())
            ));
            Ok((route, artifact))
        };
        // S10: tensor parameters whose extents are static render constant
        // canonical strides, so binding them requires those strides.
        let canonical_parameters = abi::static_geometry(&logical, &specialization)
            .parameters
            .into_iter()
            .enumerate()
            .filter_map(|(ordinal, fixed)| {
                fixed
                    .and_then(|fixed| fixed.strides)
                    .map(|strides| (ordinal, strides))
            })
            .collect();
        let (route, artifact, scalars) = if let Some(formed) = handed_off {
            (formed.route, formed.artifact, formed.scalars)
        } else {
            let (route, artifact) = form()?;
            let scalars = ScalarSlots::allocate(device, scalar_words).map_err(|error| {
                PrepareError::Preparation(PreparationError::NativeScalarSlotAllocation(
                    error.to_string(),
                ))
            })?;
            (route, artifact, scalars)
        };
        let buffer_writes = abi::buffer_writes(logical.schema(), &implementation);
        let host_writes = abi::host_writes(logical.schema(), &implementation);
        let bound = abi::bound(logical.schema(), &implementation);
        let records = (0..implementation.launches.len())
            .map(|launch| abi::records(logical.schema(), &implementation, launch))
            .collect();
        let conditions = implementation
            .launches
            .iter()
            .map(|launch| {
                launch
                    .conditions
                    .iter()
                    .map(|condition| {
                        let word = abi::tensor_buffer(logical.schema(), &condition.word)
                            .expect("a checked launch condition is a tensor parameter");
                        (word, condition.index, condition.encoded)
                    })
                    .collect()
            })
            .collect();
        Ok(Arc::new(Self {
            name,
            device: device.kind.identity(),
            public_device: device.clone(),
            parameters: logical.parameter_shapes(),
            handoff,
            invocation,
            implementation: implementation.into_implementation(),
            specialization,
            statics,
            canonical_parameters,
            buffer_writes,
            host_writes,
            bound,
            records,
            conditions,
            results,
            scalars,
            artifact,
            route,
        }))
    }

    pub(crate) fn implementation(&self) -> &NativeImplementation {
        &self.implementation
    }

    /// Whether a launch reads its grid from tensor parameter `ordinal`
    /// (`NativeLaunch::grid`): one binding no slab view may take.
    pub(crate) fn reads_grid(&self, ordinal: usize) -> bool {
        let name = &self.schema().parameters()[ordinal].name;
        self.implementation.launches.iter().any(|launch| {
            matches!(&launch.grid, Some(seismic_lang::checked::LaunchGrid::Parameter(grid)) if grid == name)
        })
    }

    /// The buffers the kernel's argument table binds: every one but its
    /// host ports.
    pub(crate) fn bound_buffers(&self) -> usize {
        self.bound.iter().filter(|&&bound| bound).count()
    }

    /// `buffers`, one per buffer in ABI order, as the kernel's argument
    /// table binds them (without its host ports), appended to `into`.
    pub(crate) fn bind<T>(&self, buffers: impl IntoIterator<Item = T>, into: &mut Vec<T>) {
        into.extend(
            buffers
                .into_iter()
                .zip(&self.bound)
                .filter_map(|(buffer, bound)| bound.then_some(buffer)),
        );
    }

    /// The host-written launch conditions launch `ordinal` declares
    /// (`NativeLaunch::conditions`): each ABI buffer, its
    /// word and the value the launch is encoded under.
    pub(crate) fn conditions(&self, ordinal: usize) -> &[(usize, u64, seismic_lang::checked::Encoded)] {
        &self.conditions[ordinal]
    }

    /// The ABI buffers a launch condition reads a word of.
    pub(crate) fn conditioned(&self) -> impl Iterator<Item = usize> + '_ {
        self.conditions.iter().flatten().map(|&(buffer, _, _)| buffer)
    }

    /// The position in the kernel's argument table of ABI buffer `buffer`,
    /// a bound one.
    fn argument(&self, buffer: usize) -> usize {
        self.bound[..buffer].iter().filter(|bound| **bound).count()
    }
    pub(crate) fn specialization(&self) -> &NativeSpecialization {
        &self.specialization
    }
    pub(crate) fn artifact(&self) -> &NativeArtifactIdentity {
        &self.artifact
    }
    pub(crate) fn schema(&self) -> &CallSchema {
        self.parameters.schema()
    }
    /// Device storage this prepared implementation holds for its calls: its
    /// scalar-result slots, none without scalar results. A standalone call's
    /// scratch is its own, released when the call ends.
    pub(crate) fn scalar_result_bytes(&self) -> u64 {
        self.scalars.as_ref().map_or(0, ScalarSlots::bytes)
    }

    /// The words of its scalar results.
    fn scalar_words(&self) -> usize {
        self.scalars.as_ref().map_or(0, |scalars| scalars.words)
    }

    /// Its scalar-result slots, as a launch binds them: none for an entry
    /// without scalar results.
    pub(crate) fn scalar_slots(&self) -> Option<&Arc<Allocation>> {
        self.scalars.as_ref().map(|scalars| &scalars.slots)
    }
    pub(crate) fn result_count(&self) -> u32 {
        u32::try_from(self.results.len()).expect("native result ordinal space exhausted")
    }

    /// Validate arguments against the checked contract and the fixed static
    /// dimensions.
    pub(crate) fn validate(
        &self,
        arguments: &[ArgumentValue],
    ) -> Result<InvocationValues, CallError> {
        let values = validate_invocation(&self.invocation, self.device, arguments)
            .map_err(CallError::Invocation)?;
        for (dimension, symbol, expected) in &self.statics {
            match values.get(*symbol) {
                Some(SymbolValue::Nat(actual)) if actual == (*expected).into() => {}
                Some(SymbolValue::Nat(actual)) => {
                    return Err(CallError::Invocation(InvocationError::StaticDimension {
                        dimension: dimension.clone(),
                        expected: *expected,
                        actual,
                    }));
                }
                _ => panic!("validated invocation omitted a native static dimension"),
            }
        }
        for (ordinal, strides) in &self.canonical_parameters {
            let ArgumentValue::Tensor(tensor) = &arguments[*ordinal] else {
                panic!("validated invocation bound a non-tensor to a tensor parameter")
            };
            if &tensor.strides != strides {
                return Err(CallError::Invocation(
                    InvocationError::NoncanonicalStaticTensor {
                        parameter: self.schema().parameters()[*ordinal].name.clone(),
                    },
                ));
            }
        }
        Ok(values)
    }

    fn dimension(&self, values: &InvocationValues, name: &str) -> Option<u64> {
        let symbol = self
            .schema()
            .dimensions()
            .iter()
            .find(|dimension| dimension.name == name)?
            .symbol;
        match values.get(symbol) {
            Some(SymbolValue::Nat(value)) => u64::try_from(value).ok(),
            _ => None,
        }
    }

    fn evaluation_error(&self, error: seismic_lang::checked::NativeEvalError) -> CallError {
        CallError::Execution(ExecutionError::SubmissionFailed(format!(
            "native expression of `{}`: {error}",
            self.name
        )))
    }

    fn evaluate(
        &self,
        expression: &NativeNatExpr,
        values: &InvocationValues,
        launch: Option<usize>,
    ) -> Result<u64, CallError> {
        self.evaluate_at(expression, &|name| self.dimension(values, name), launch)
    }

    /// `expression` where `dimension` gives the entry's dimensions.
    fn evaluate_at(
        &self,
        expression: &NativeNatExpr,
        dimension: &dyn Fn(&str) -> Option<u64>,
        launch: Option<usize>,
    ) -> Result<u64, CallError> {
        expression
            .evaluate(&|name| dimension(name), &|name| {
                launch
                    .and_then(|ordinal| self.specialization.launch_param(ordinal, name))
                    .or_else(|| self.specialization.param(name))
            })
            .map_err(|error| self.evaluation_error(error))
    }

    /// Whether a launch or scratch buffer with this `when` condition is
    /// active for one invocation.
    fn active(
        &self,
        when: &Option<NativeCondition>,
        values: &InvocationValues,
        launch: Option<usize>,
    ) -> Result<bool, CallError> {
        self.active_at(when, &|name| self.dimension(values, name), launch)
    }

    /// Whether each buffer, in ABI order, is a record port of a call that
    /// runs `launches`: the record ports of its active launches, which
    /// agree, as each buffer binds once for all of them; with none active,
    /// the first launch's.
    ///
    /// # Panics
    /// Where launches active together list different record ports: the
    /// native's declaration is wrong.
    pub(crate) fn records(&self, launches: &CallLaunches) -> &[bool] {
        let mut active = launches
            .geometry
            .iter()
            .zip(&self.records)
            .filter_map(|(geometry, records)| geometry.as_ref().map(|_| records));
        let records = active.next().unwrap_or(&self.records[0]);
        assert!(
            active.all(|other| other == records),
            "launches of a native that run together list different record ports"
        );
        records
    }

    /// [`Self::active`] where `dimension` gives the entry's dimensions.
    fn active_at(
        &self,
        when: &Option<NativeCondition>,
        dimension: &dyn Fn(&str) -> Option<u64>,
        launch: Option<usize>,
    ) -> Result<bool, CallError> {
        match when {
            None => Ok(true),
            Some(condition) => condition
                .holds(&|name| dimension(name), &|name| {
                    launch
                        .and_then(|ordinal| self.specialization.launch_param(ordinal, name))
                        .or_else(|| self.specialization.param(name))
                })
                .map_err(|error| self.evaluation_error(error)),
        }
    }

    /// Geometry of every launch for one invocation, checked against the
    /// formed functions and device limits. An inactive launch is `None`:
    /// its geometry is neither evaluated nor checked.
    pub(crate) fn launches(&self, values: &InvocationValues) -> Result<CallLaunches, CallError> {
        self.launches_at(&|name| self.dimension(values, name))
    }

    /// Check the launches of a call whose dimensions are `dimensions`
    /// against the formed functions and the device limits, as
    /// [`Self::shape`] checks a call's from its arguments. A call the kernel
    /// serves at a shape no argument set at hand takes (more history
    /// segments than a tuning point's) is checked here before the kernel is
    /// chosen for it.
    pub(crate) fn admits(&self, dimensions: &BTreeMap<String, u64>) -> Result<(), CallError> {
        self.launches_at(&|name| dimensions.get(name).copied())
            .map(|_| ())
    }

    fn launches_at(
        &self,
        dimension: &dyn Fn(&str) -> Option<u64>,
    ) -> Result<CallLaunches, CallError> {
        // Only the launches a call within the ranges can activate are
        // formed, so a call outside them is refused before any launch is.
        for (name, &(low, high)) in self.specialization.ranges() {
            let actual = dimension(name).ok_or_else(|| {
                self.evaluation_error(seismic_lang::checked::NativeEvalError::Unbound(
                    name.clone(),
                ))
            })?;
            if !(low..=high).contains(&actual) {
                return Err(CallError::Invocation(InvocationError::OutsideRange {
                    dimension: name.clone(),
                    low,
                    high,
                    actual,
                }));
            }
        }
        let mut geometry = Vec::with_capacity(self.implementation.launches.len());
        for (ordinal, launch) in self.implementation.launches.iter().enumerate() {
            if !self.active_at(&launch.when, dimension, Some(ordinal))? {
                geometry.push(None);
                continue;
            }
            let axes = |expressions: &[NativeNatExpr; 3]| -> Result<[u64; 3], CallError> {
                Ok([
                    self.evaluate_at(&expressions[0], dimension, Some(ordinal))?,
                    self.evaluate_at(&expressions[1], dimension, Some(ordinal))?,
                    self.evaluate_at(&expressions[2], dimension, Some(ordinal))?,
                ])
            };
            let mut groups = axes(&launch.groups)?;
            let threads = axes(&launch.group_extent)?;
            let shared_bytes = self.evaluate_at(&launch.shared_bytes, dimension, Some(ordinal))?;
            let limit =
                |message: String| CallError::Execution(ExecutionError::SubmissionFailed(message));
            let participants = threads
                .iter()
                .try_fold(1u64, |product, value| product.checked_mul(*value))
                .ok_or_else(|| {
                    limit(format!("launch `{}` thread count overflows", launch.kernel))
                })?;
            match &self.route {
                NativeRoute::Cpu(_) => {}
                #[cfg(target_os = "macos")]
                NativeRoute::Metal { opened, launches } => {
                    let pipeline = launches.pipeline(ordinal);
                    if participants > pipeline.max_threads_per_threadgroup() {
                        return Err(limit(format!(
                            "launch `{}` requests {participants} threads per threadgroup; the pipeline allows {}",
                            launch.kernel,
                            pipeline.max_threads_per_threadgroup()
                        )));
                    }
                    let available = opened.device_description().limits().max_workgroup_bytes;
                    let group_bytes = shared_bytes + pipeline.static_threadgroup_bytes();
                    if group_bytes > available {
                        return Err(limit(format!(
                            "launch `{}` needs {group_bytes} threadgroup bytes; the device allows {available}",
                            launch.kernel,
                        )));
                    }
                    // `RESIDENT`: half of the threadgroups the cores hold at
                    // once, each core as many as the pipeline's threads and
                    // the threadgroup memory allow. The memory term takes
                    // the per-threadgroup limit as a core's capacity, which
                    // holds at least because a core fits one maximal
                    // threadgroup; the half covers the rest. On an M4 Pro a
                    // grid of the full count was seen co-resident and one
                    // more was not; the half leaves the margin no query
                    // states, and the barrier's bounded wait reports a grid
                    // that still was not.
                    if launch.resident.is_some() {
                        let per_core = (pipeline.max_threads_per_threadgroup() / participants.max(1))
                            .min(available / group_bytes.max(1));
                        let cores = opened.device_description().facts().properties().units.get();
                        groups = [groups[0].min((cores * per_core / 2).max(1)), 1, 1];
                    }
                }
                NativeRoute::Cuda { opened, launches } => {
                    let (module, function) = launches.launch(ordinal);
                    if participants > module.max_threads_per_block(function) {
                        return Err(limit(format!(
                            "launch `{}` requests {participants} threads per block; the function allows {}",
                            launch.kernel,
                            module.max_threads_per_block(function)
                        )));
                    }
                    let available = opened.device_description().limits().max_workgroup_bytes;
                    if shared_bytes + module.static_shared_bytes(function) > available {
                        return Err(limit(format!(
                            "launch `{}` needs {} shared bytes; the device allows {available}",
                            launch.kernel,
                            shared_bytes + module.static_shared_bytes(function)
                        )));
                    }
                    // A written grid's blocks run as a grid-stride loop
                    // (`abi`): as many blocks as the device holds at once,
                    // never more than the grid's bound.
                    if launch.grid.is_some() {
                        let resident = module
                            .active_blocks(opened.service(), function, participants, shared_bytes)
                            .map_err(CallError::Execution)?
                            * opened.device_description().facts().properties.units.get();
                        if resident == 0 {
                            return Err(limit(format!(
                                "launch `{}`: no multiprocessor holds one of its blocks",
                                launch.kernel
                            )));
                        }
                        let bound = groups.iter().product::<u64>();
                        groups = [bound.min(resident), 1, 1];
                    }
                }
                // The workgroup size and group memory are the pipeline's,
                // checked at preparation; only the grid varies per call.
                #[cfg(not(target_os = "macos"))]
                NativeRoute::Vulkan { opened, .. } => {
                    let available = opened.service().facts().limits.max_group_count;
                    if let Some(axis) = (0..3).find(|axis| groups[*axis] > available[*axis]) {
                        return Err(limit(format!(
                            "launch `{}` requests {} workgroups on axis {axis}; the device allows {}",
                            launch.kernel, groups[axis], available[axis]
                        )));
                    }
                }
            }
            let grid = launch
                .grid
                .as_ref()
                .map(|grid| abi::grid_buffer(self.schema(), &self.implementation, grid));
            let barrier = launch
                .resident
                .as_ref()
                .map(|resident| abi::scratch_buffer(self.schema(), &self.implementation, &resident.barrier));
            geometry.push(Some(LaunchGeometry {
                groups,
                grid,
                barrier,
                threads,
                shared_bytes,
            }));
        }
        let rounds = match &self.implementation.repeat {
            None => 1,
            Some(repeat) => self.evaluate_at(&repeat.count, dimension, None)?,
        };
        Ok(CallLaunches { geometry, rounds })
    }

    /// Every scratch buffer of one invocation. An inactive buffer is charged
    /// the minimum without evaluating its size.
    pub(crate) fn scratch_needs(
        &self,
        values: &InvocationValues,
    ) -> Result<Vec<ScratchNeed>, CallError> {
        self.implementation
            .scratch
            .iter()
            .map(|scratch| {
                let bytes = if self.active(&scratch.when, values, None)? {
                    self.evaluate(&scratch.bytes, values, None)?
                        .max(MINIMUM_SCRATCH_BYTES)
                } else {
                    MINIMUM_SCRATCH_BYTES
                };
                Ok(ScratchNeed {
                    bytes,
                    sync: scratch.sync,
                })
            })
            .collect()
    }

    fn result_extents(
        &self,
        values: &InvocationValues,
    ) -> Result<Vec<Option<(RepresentationId, Vec<u64>)>>, CallError> {
        self.results
            .iter()
            .map(|result| match result {
                NativeResult::Tensor {
                    representation,
                    axes,
                } => Ok(Some((
                    *representation,
                    axes.iter()
                        .map(|axis| evaluate_compiled(axis, values))
                        .collect::<Result<Vec<_>, _>>()?,
                ))),
                NativeResult::Scalar(_) | NativeResult::Index | NativeResult::Range => Ok(None),
            })
            .collect()
    }

    /// The parts of one call its arguments' geometry and scalar values fix,
    /// after full contract validation: result specs, scratch sizes, argument
    /// words and launch geometry. A standalone call evaluates it per call; a
    /// sealed graph evaluates it once per node, at seal.
    pub(crate) fn shape(&self, arguments: &[ArgumentValue]) -> Result<CallShape, CallError> {
        let values = self.validate(arguments)?;
        let launches = self.launches(&values)?;
        let results = self
            .result_extents(&values)?
            .into_iter()
            .map(|result| {
                result
                    .map(|(representation, extents)| {
                        let layout = crate::layout::canonical(representation, &extents)
                            .map_err(CallError::Execution)?;
                        Ok(NativeTensorSpec {
                            representation,
                            extents,
                            strides: layout.strides,
                            byte_len: layout.byte_len,
                        })
                    })
                    .transpose()
            })
            .collect::<Result<Vec<_>, CallError>>()?;
        let words = native_words(
            self.schema(),
            &self.implementation,
            &self.specialization,
            arguments,
            &results,
            &values,
        )?;
        Ok(CallShape {
            scratch: self.scratch_needs(&values)?,
            results,
            words,
            launches,
        })
    }

    pub(crate) fn tensor_parameter_spec(
        &self,
        name: &str,
        dimensions: &[(&str, u64)],
    ) -> Result<NativeTensorSpec, CallError> {
        let failure =
            |message: String| CallError::Execution(ExecutionError::SubmissionFailed(message));
        let shape = self
            .parameters
            .tensor_parameter_shape(name, dimensions)
            .map_err(|error| failure(error.to_string()))?;
        let layout = crate::layout::canonical(shape.representation, &shape.extents)
            .map_err(CallError::Execution)?;
        Ok(NativeTensorSpec {
            representation: shape.representation,
            extents: shape.extents,
            strides: layout.strides,
            byte_len: layout.byte_len,
        })
    }

    /// Graph nodes cannot publish scalar results: a scalar crosses a host
    /// boundary.
    pub(crate) fn validate_graph_node(&self, device: DeviceIdentity) -> Result<(), CallError> {
        if self.device != device {
            return Err(CallError::Workflow(
                crate::api::WorkflowError::NativeGraphSlotMismatch,
            ));
        }
        if self
            .results
            .iter()
            .any(|result| !matches!(result, NativeResult::Tensor { .. }))
        {
            return Err(CallError::Workflow(
                crate::api::WorkflowError::HostBoundaryRequired,
            ));
        }
        Ok(())
    }

    pub(crate) fn call(self: &Arc<Self>, args: EncodedArgs) -> Result<DecodedResults, CallError> {
        self.call_with(args, None, Parking::Unanswered, || {})
    }

    /// A standalone call whose every awaited word already holds its answer
    /// (the host wrote it before the call), as a graph's park bound
    /// answered is (`NativeGraphBindings::set_answered`): it waits for
    /// nothing, so every backend runs it, wherever its parks are realized.
    pub(crate) fn call_answered(self: &Arc<Self>, args: EncodedArgs) -> Result<DecodedResults, CallError> {
        self.call_with(args, None, Parking::Answered, || {})
    }

    pub(crate) fn call_into(
        self: &Arc<Self>,
        args: EncodedArgs,
        outputs: EncodedOutputs,
    ) -> Result<DecodedResults, CallError> {
        self.call_with(args, Some(outputs), Parking::Unanswered, || {})
    }

    pub(crate) fn call_with_commit(
        self: &Arc<Self>,
        args: EncodedArgs,
        commit: impl FnOnce(),
    ) -> Result<DecodedResults, CallError> {
        self.call_with(args, None, Parking::Unanswered, commit)
    }

    /// Validate a standalone call, place its results and its scratch (in
    /// `arena`, which the calls placed together share), and fix its
    /// executable form.
    fn prepare_call(
        &self,
        arena: &mut ScratchArena,
        args: EncodedArgs,
        outputs: Option<EncodedOutputs>,
        parking: Parking,
    ) -> Result<NativeBoundCall, CallError> {
        validate_standalone_parking(self.schema(), self.public_device.info().backend(), parking)?;
        for (parameter, tensor) in self.schema().parameters().iter().zip(args.tensors()) {
            if let (ParameterKind::Tensor { access, .. }, Some(tensor)) = (&parameter.kind, tensor)
            {
                if matches!(access, TensorAccess::Owned | TensorAccess::Mutable)
                    && tensor.allocation().storage().read_only()
                {
                    return Err(CallError::Execution(ExecutionError::SubmissionFailed(
                        format!(
                            "native parameter `{}` cannot write a mapped tensor",
                            parameter.name
                        ),
                    )));
                }
            }
        }
        let shape = self.shape(&args.values())?;
        let specs = shape.results.iter().flatten().collect::<Vec<_>>();
        let supplied = outputs.map(EncodedOutputs::into_tensors);
        if let Some(outputs) = &supplied {
            if outputs.len() != specs.len() {
                return Err(CallError::Output(OutputError::Count {
                    expected: specs.len(),
                    actual: outputs.len(),
                }));
            }
        }
        let mut results = Vec::with_capacity(specs.len());
        for (index, spec) in specs.into_iter().enumerate() {
            let tensor = match &supplied {
                Some(outputs) => {
                    let tensor = outputs[index].clone();
                    validate_output(
                        index,
                        &tensor,
                        self.device,
                        spec.representation,
                        &spec.extents,
                        &args,
                        &results,
                    )?;
                    tensor
                }
                None => Arc::new(
                    TensorInner::zeros(&self.public_device, spec.representation, &spec.extents)
                        .map_err(tensor_error)?,
                ),
            };
            results.push(tensor);
        }
        let (scratch, scratch_end) = if shape.scratch.is_empty() {
            (Vec::new(), 0)
        } else {
            place_scratch(&shape.scratch)
        };
        let arena = arena.holding(&self.public_device, scratch_end)?;

        let schema = self.schema();
        let mut buffers = Vec::new();
        let mut slabs = Vec::new();
        let mut records = Vec::new();
        let mut representations = Vec::new();
        let mut access = collect_native_access(schema, &args);
        // Tensor parameters by schema ordinal, then results by ordinal.
        let parameters = schema
            .parameters()
            .iter()
            .enumerate()
            .filter(|(_, parameter)| matches!(parameter.kind, ParameterKind::Tensor { .. }))
            .map(|(ordinal, _)| {
                let tensor = args
                    .tensor(ordinal)
                    .expect("validated native tensor argument disappeared");
                (Ok(ordinal), tensor)
            });
        let produced = results.iter().enumerate().map(|(ordinal, tensor)| (Err(ordinal), tensor));
        let record_ports = self.records(&shape.launches);
        for (port, tensor) in parameters.chain(produced) {
            let name = || match port {
                Ok(ordinal) => schema.parameters()[ordinal].name.clone(),
                Err(ordinal) => format!("result {ordinal}"),
            };
            if port.is_ok_and(|ordinal| self.reads_grid(ordinal)) && tensor.is_slab_view() {
                return Err(CallError::Invocation(InvocationError::SlabGrid { parameter: name() }));
            }
            let buffer = buffers.len();
            if record_ports[buffer] {
                let bound = tensor.records().map_err(tensor_error)?;
                buffers.push((
                    bound.table.clone(),
                    bound.table_offset,
                    seismic_lang::registry::record_table_entry(bound.count),
                ));
                records.push((buffer, bound));
            } else if !tensor.binds_directly() && self.bound[buffer] {
                // Chunked storage with no one buffer: device work binds it
                // only through a record port. A host port, never bound,
                // takes it, ordered by its packed span.
                return Err(CallError::Workflow(crate::api::WorkflowError::ChunkedBinding {
                    parameter: name(),
                }));
            } else {
                if let Some(slab) = tensor.slab_reach() {
                    slabs.push((buffer, slab));
                }
                buffers.push((
                    tensor.allocation().clone(),
                    tensor.byte_offset(),
                    tensor.reached_bytes(),
                ));
            }
            representations.push(representation_name(tensor.representation()));
        }
        // Each launch condition's word is one the host reads as the call is
        // encoded: refused here unless the host addresses its storage.
        if let Some(buffer) = self.conditioned().find(|&buffer| buffers[buffer].0.storage().host_address().is_none()) {
            let parameter = abi::tensor_buffer_name(schema, buffer).to_owned();
            return Err(CallError::Workflow(crate::api::WorkflowError::UnaddressedCondition { parameter }));
        }
        for tensor in &results {
            access.push((tensor.allocation().clone(), true));
        }
        if let Some(arena) = arena {
            for (offset, need) in scratch.into_iter().zip(&shape.scratch) {
                buffers.push((arena.clone(), offset, need.bytes));
                representations.push("bytes");
            }
            access.push((arena, true));
        }
        access.extend(self.scalar_slots().map(|slots| (slots.clone(), true)));
        // The kernel's arguments: its bound buffers'.
        let mut bound = Vec::with_capacity(representations.len());
        self.bind(representations, &mut bound);
        Ok(NativeBoundCall {
            results,
            buffers,
            slabs,
            records,
            representations: bound,
            words: shape.words,
            launches: shape.launches,
            access,
            binding_leases: args
                .tensors()
                .flatten()
                .filter(|tensor| tensor.holds_binding_lease())
                .cloned()
                .collect(),
        })
    }

    fn call_with(
        self: &Arc<Self>,
        args: EncodedArgs,
        outputs: Option<EncodedOutputs>,
        parking: Parking,
        commit: impl FnOnce(),
    ) -> Result<DecodedResults, CallError> {
        let _slots = self.scalars.as_ref().map(|scalars| {
            scalars
                .calls
                .lock()
                .expect("native scalar-result slots lock poisoned")
        });
        // A standalone call's scratch is its own, allocated and zeroed per
        // call, deliberately: an arena kept on the kernel between calls held
        // 17.1 MB per Qwen3.6 35B load that no plan term counted
        // (`standalone_scratch_is_the_calls_own`), and no hot path makes
        // standalone calls. Batches and tuning points share one arena.
        let call = self.prepare_call(&mut ScratchArena::default(), args, outputs, parking)?;
        self.reset_scalar_results()?;
        commit();
        let calls = [call];
        StandaloneCalls {
            kernel: self,
            calls: &calls,
        }
        .submit(1, false)?
        .wait()?;
        self.read_call_results(&calls[0])
    }

    fn reset_scalar_results(&self) -> Result<(), CallError> {
        if let Some(scalars) = &self.scalars {
            let _host = scalars.slots.acquire(true).map_err(CallError::Execution)?;
            write_zeros(scalars.slots.storage(), scalars.bytes()).map_err(CallError::Execution)?;
        }
        Ok(())
    }

    fn read_call_results(&self, call: &NativeBoundCall) -> Result<DecodedResults, CallError> {
        let mut scalar_bytes = Vec::new();
        if let Some(scalars) = &self.scalars {
            scalar_bytes.resize(scalars.bytes() as usize, 0);
            scalars
                .slots
                .acquire(false)
                .map_err(CallError::Execution)?
                .read(&scalars.slots, 0, &mut scalar_bytes)
                .map_err(CallError::Execution)?;
        }
        let mut offset = 0usize;
        let mut tensors = call.results.iter().cloned();
        let mut decoded = Vec::with_capacity(self.results.len());
        for result in &self.results {
            match result {
                NativeResult::Tensor { .. } => decoded.push(DecodedValue::Tensor(
                    tensors.next().expect("native result tensor count changed"),
                )),
                NativeResult::Scalar(dtype) => decoded.push(DecodedValue::Scalar(scalar_value(
                    *dtype,
                    read_word(&scalar_bytes, offset),
                ))),
                NativeResult::Index => decoded.push(DecodedValue::Scalar(ArgumentValue::Index(
                    read_word(&scalar_bytes, offset).into(),
                ))),
                NativeResult::Range => decoded.push(DecodedValue::Scalar(ArgumentValue::Range {
                    start: read_word(&scalar_bytes, offset).into(),
                    end: read_word(&scalar_bytes, offset + 1).into(),
                })),
            }
            offset += result.class().scalar_words();
        }
        Ok(DecodedResults::new(decoded))
    }
}

/// The scratch of standalone calls placed together (one call, a batch, a
/// tuning point): one arena, grown to the largest call's scratch and reused
/// by every later one (the device queue orders its users). Each placed call
/// keeps the arena it was placed in, so the storage lives as long as the
/// calls do, never as long as their kernel.
#[derive(Default)]
pub(crate) struct ScratchArena {
    scratch: Option<Arc<Allocation>>,
}

impl ScratchArena {
    /// The arena, holding at least `bytes`; `None` when the call has no
    /// scratch.
    fn holding(
        &mut self,
        device: &Arc<DeviceInner>,
        bytes: u64,
    ) -> Result<Option<Arc<Allocation>>, CallError> {
        if bytes == 0 {
            return Ok(None);
        }
        if let Some(arena) = self.scratch.as_ref().filter(|arena| arena.bytes() >= bytes) {
            return Ok(Some(arena.clone()));
        }
        let arena = device
            .allocate_zeroed(bytes, BUFFER_ALIGNMENT)
            .map_err(CallError::Execution)?;
        self.scratch = Some(arena.clone());
        Ok(Some(arena))
    }
}

/// What a call's argument geometry and scalar values fix
/// ([`NativePrepared::shape`]).
pub(crate) struct CallShape {
    /// Per result ordinal; `None` for a scalar result.
    pub(crate) results: Vec<Option<NativeTensorSpec>>,
    /// Each scratch buffer, in declaration order.
    pub(crate) scratch: Vec<ScratchNeed>,
    pub(crate) words: Vec<u64>,
    pub(crate) launches: CallLaunches,
}

/// Argument words as device argument ABIs read them: each word's
/// little-endian bytes, which on the little-endian hosts the runtime builds
/// for are the words' own memory. One owner serves CPU (native words) and
/// device (bytes) encoders alike.
pub(crate) fn word_bytes(words: &[u64]) -> &[u8] {
    const _: () = assert!(cfg!(target_endian = "little"), "argument words are little-endian in memory");
    // SAFETY: `u64` has no padding and every byte is initialized; `u8` has
    // alignment 1; the length covers exactly the words' bytes.
    unsafe { std::slice::from_raw_parts(words.as_ptr().cast::<u8>(), std::mem::size_of_val(words)) }
}

/// Whether a standalone call's awaited words may still be waiting for their
/// answers (`Unanswered`), or already hold them (`Answered`,
/// [`NativePrepared::call_answered`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Parking {
    Unanswered,
    Answered,
}

/// Standalone calls and points have no timeline binding that can answer a
/// park: an awaited entry whose answers may still come is called standalone
/// only where the device's kernels park themselves; one already answered
/// waits for nothing, on every backend. A graph binds its parks explicitly.
fn validate_standalone_parking(schema: &CallSchema, backend: BackendName, parking: Parking) -> Result<(), CallError> {
    if schema.parks() && parking == Parking::Unanswered && !backend.parks_in_kernels() {
        return Err(CallError::Workflow(crate::api::WorkflowError::ParkRequiresGraph));
    }
    Ok(())
}

fn representation_name(representation: RepresentationId) -> &'static str {
    seismic_lang::registry::representation_info(representation).name
}

/// One validated standalone call ready for submission: its arguments,
/// results, buffers, argument words and launch geometry are fixed.
struct NativeBoundCall {
    /// The result views, read back by a standalone call once it completes.
    /// Their storage is also in `buffers`.
    results: Vec<Arc<TensorInner>>,
    /// Buffers in ABI order with their byte offsets and the bytes the call
    /// reaches there.
    buffers: Vec<(Arc<Allocation>, u64, u64)>,
    /// Each buffer bound to a slab view, by ABI ordinal, with what it
    /// reaches (the encodings order by it).
    slabs: Vec<(usize, hazard::SlabReach)>,
    /// Each buffer bound to a record port, by ABI ordinal, with the records
    /// its table reaches.
    records: Vec<(usize, Arc<Records>)>,
    /// The bound buffers' representations, in argument order.
    representations: Vec<&'static str>,
    words: Vec<u64>,
    launches: CallLaunches,
    /// Every allocation the call touches, with whether it writes it.
    access: Vec<(Arc<Allocation>, bool)>,
    /// Keep slab placements and pinned reserved regions fixed while the
    /// call holds them: a standalone call until it is dropped, past its
    /// completion; a batch's call only until its submission records its
    /// fences ([`Self::release_views`]), which order any release after the
    /// work.
    binding_leases: Vec<Arc<TensorInner>>,
}

impl NativeBoundCall {
    /// Whether launch `ordinal` of `kernel`, this call's, is encoded: each
    /// of its launch conditions' words, of the buffers the call binds, holds
    /// ([`condition_holds`]).
    fn encodes(&self, kernel: &NativePrepared, ordinal: usize) -> bool {
        kernel.conditions(ordinal).iter().all(|&(buffer, index, encoded)| {
            let (allocation, offset, _) = &self.buffers[buffer];
            condition_holds(allocation, offset + 4 * index, encoded)
        })
    }

    /// The submission has recorded its fences: the call's argument and
    /// result views end, and with them any slab or region lease they hold,
    /// so placement may change from now on. The call keeps its storage
    /// (`buffers`, `access`), whose fences defer any physical release to
    /// completion. For a call whose results nobody reads back (a batch's:
    /// its caller owns the output tensors).
    fn release_views(&mut self) {
        self.binding_leases = Vec::new();
        self.results = Vec::new();
    }
}

impl NativeBoundCall {
    /// What the call's launches reach, into `reaches`.
    fn reaches<'s>(&'s self, reaches: &mut Vec<hazard::Reach<'s>>) {
        for (buffer, (allocation, offset, bytes)) in self.buffers.iter().enumerate() {
            if let Some((_, records)) = self.records.iter().find(|(bound, _)| *bound == buffer) {
                records.extend(buffer, 0, records.count, reaches);
                continue;
            }
            match self.slabs.iter().find(|(slabbed, _)| *slabbed == buffer) {
                Some((_, slab)) => slab.extend(buffer, allocation, *offset, reaches),
                None => reaches.push(hazard::Reach {
                    buffer,
                    allocation,
                    offset: *offset,
                    bytes: *bytes,
                    through: hazard::Through::Binding,
                }),
            }
        }
    }
}

/// Standalone calls of one prepared implementation, in submission order.
struct StandaloneCalls<'k> {
    kernel: &'k Arc<NativePrepared>,
    calls: &'k [NativeBoundCall],
}

impl StandaloneCalls<'_> {
    fn submit(&self, repetitions: usize, observed: bool) -> Result<NativeSubmission, CallError> {
        let access = merge_access(
            self.calls
                .iter()
                .flat_map(|call| call.access.iter().cloned()),
        );
        submit(self, &access, self.kernel.clone(), repetitions, observed, NativeQueue::Compute)
    }
}

impl DispatchList for StandaloneCalls<'_> {
    fn count(&self) -> usize {
        self.calls.len()
    }

    fn dispatch<'s>(
        &'s self,
        index: usize,
        buffers: &mut Vec<(&'s Allocation, u64)>,
    ) -> Dispatch<'s> {
        let call = &self.calls[index];
        self.kernel.bind(
            call.buffers
                .iter()
                .map(|(allocation, offset, _)| (&**allocation, *offset)),
            buffers,
        );
        Dispatch {
            kernel: self.kernel,
            words: &call.words,
            launches: &call.launches.geometry,
            rounds: call.launches.rounds,
            representations: &call.representations,
        }
    }

    fn reaches<'s>(&'s self, index: usize, reaches: &mut Vec<hazard::Reach<'s>>) {
        self.calls[index].reaches(reaches);
    }

    fn encodes(&self, index: usize, ordinal: usize) -> bool {
        self.calls[index].encodes(self.kernel, ordinal)
    }

    fn runs(&self) -> Option<Vec<PlanRun>> {
        None
    }

    /// Standalone calls run as a served step runs one kernel's calls: a
    /// tuning point's rotation cycles a model's layers, which the step
    /// runs each after the one before.
    fn chained(&self) -> bool {
        true
    }
}

/// One call as the encoder consumes it: everything but its buffers is
/// fixed.
pub(crate) struct Dispatch<'a> {
    pub(crate) kernel: &'a NativePrepared,
    pub(crate) words: &'a [u64],
    /// By declaration ordinal; `None` for an inactive launch.
    pub(crate) launches: &'a [Option<LaunchGeometry>],
    /// How often the kernel's repeat block is dispatched.
    pub(crate) rounds: u64,
    /// Registry name of each bound buffer's representation, in argument
    /// order.
    pub(crate) representations: &'a [&'static str],
}

/// Whether a launch condition holds now: the `i32` word at byte `offset` of
/// `allocation`, which the host writes between submissions, at the value
/// the launch is encoded under. The call or graph that binds the word
/// refused storage the host cannot address as it formed.
pub(crate) fn condition_holds(allocation: &Allocation, offset: u64, encoded: seismic_lang::checked::Encoded) -> bool {
    let base = allocation
        .storage()
        .host_address()
        .expect("a launch condition's word is host-addressed: its call or graph checked it as it formed");
    // SAFETY: a live, aligned i32 of the bound view (the check bounds the
    // word by the view's extent); the host touches shared words only
    // through atomics.
    let word = unsafe { std::sync::atomic::AtomicU32::from_ptr(base.as_ptr().add(offset as usize).cast()) };
    encoded.encodes(word.load(std::sync::atomic::Ordering::Acquire))
}

/// The calls of one submission, in order. Every call belongs to one device.
pub(crate) trait DispatchList {
    fn count(&self) -> usize;
    /// Call `index`; the buffers its kernel binds (every buffer but a host
    /// port, [`abi::bound`]), in ABI order with byte offsets, are appended to
    /// `buffers`.
    fn dispatch<'s>(
        &'s self,
        index: usize,
        buffers: &mut Vec<(&'s Allocation, u64)>,
    ) -> Dispatch<'s>;
    /// What call `index`'s launches reach through all its buffers, host
    /// ports included (each by its ABI ordinal), appended to `reaches`.
    /// Metal's concurrent encoding orders by them ([`hazard`]).
    fn reaches<'s>(&'s self, index: usize, reaches: &mut Vec<hazard::Reach<'s>>);
    /// The sealed plans' runs that make up the list, in order, each with
    /// its calls: a run's segments between timeline boundaries are replayed
    /// as graphs of their shape ([`ReplayKeys`]). `None` for lists launched
    /// call by call.
    fn runs(&self) -> Option<Vec<PlanRun>>;
    /// Whether launch `ordinal` of call `index` is encoded in this
    /// submission: every launch condition it carries holds
    /// ([`condition_holds`]).
    fn encodes(&self, index: usize, ordinal: usize) -> bool;
    /// Per call, whether each of its launches, by ordinal, is encoded in
    /// this submission ([`Self::encodes`]).
    fn encoded(&self) -> Vec<Vec<bool>> {
        let mut buffers = Vec::new();
        (0..self.count())
            .map(|index| {
                buffers.clear();
                let launches = self.dispatch(index, &mut buffers).launches.len();
                (0..launches).map(|ordinal| self.encodes(index, ordinal)).collect()
            })
            .collect()
    }
    /// The timeline operations at `boundary`, before call `boundary`
    /// (`count()` after the last), in order.
    fn timelines(&self, _boundary: usize) -> &[timeline::TimelineOperation] {
        &[]
    }
    /// Whether any boundary has timeline operations.
    fn has_timelines(&self) -> bool {
        false
    }
    /// Whether each call runs after the one before it, whatever they reach:
    /// a served step runs one kernel's calls (a model's layers) each after
    /// its predecessor, so calls timed for it run so too ([`hazard`]).
    fn chained(&self) -> bool {
        false
    }
    /// The stamps of sampled runs' marks at `boundary` (`spans`), in order.
    fn stamps(&self, _boundary: usize) -> &[spans::Stamp] {
        &[]
    }
}

/// One entry per access group with merged write access, in group order:
/// the order device permits are taken in, so concurrent submitters cannot
/// deadlock. Slab regions and their address table share one group.
pub(crate) fn merge_access(
    access: impl IntoIterator<Item = (Arc<Allocation>, bool)>,
) -> Vec<(Arc<Allocation>, bool)> {
    let mut merged: BTreeMap<usize, (Arc<Allocation>, bool)> = BTreeMap::new();
    for (allocation, write) in access {
        merged
            .entry(allocation.access_identity())
            .and_modify(|(_, current)| *current |= write)
            .or_insert((allocation, write));
    }
    merged.into_values().collect()
}

/// Submitted native work. It retains the formed functions it executes;
/// allocation fences keep every touched storage alive and order host access
/// until the work completes. Opaque outside the runtime: its seconds are
/// read through [`seismic_measure::Pending`].
pub struct NativeSubmission {
    route: Arc<RouteSubmission>,
    /// Each host gate the work waits at: its timeline and value.
    gates: Vec<(Arc<timeline::TimelineInner>, u64)>,
    /// Each resident launch's barrier, whose give-up fails the work.
    barriers: Vec<ResidentBarrier>,
    _retained: Arc<dyn std::any::Any + Send + Sync>,
}

impl Drop for NativeSubmission {
    /// A binding lease can be the last lease without being the allocation's
    /// last owner: the work retires before its leases and formed functions
    /// are released, measured runs included.
    fn drop(&mut self) {
        self.route.retire();
    }
}

impl NativeSubmission {
    /// Whether the work ended, so [`Self::wait`] reports its outcome
    /// without blocking: it completed, or faulted.
    pub(crate) fn is_complete(&self) -> bool {
        match self.route.completed() {
            Ok(complete) => complete,
            Err(_fault) => true,
        }
    }

    /// Wait for completion and report the work's outcome.
    pub(crate) fn wait(&self) -> Result<(), CallError> {
        self.route.finish()?;
        // Work behind a gate its timeline's release opened read what the
        // host never put in place (`timeline`).
        if let Some(error) = self
            .gates
            .iter()
            .find_map(|(timeline, value)| timeline.released_unsignalled(*value))
        {
            return Err(CallError::Execution(error));
        }
        // A give-up is terminal: its word stays set, so every later launch
        // over the barrier gives up at once, and the device is lost to the
        // load, not a call to retry.
        if self.barriers.iter().any(ResidentBarrier::gave_up) {
            return Err(CallError::Execution(ExecutionError::DeviceLost(
                "a resident launch's threadgroups were not all on the device at once: its grid barrier gave up".into(),
            )));
        }
        Ok(())
    }

    /// Device execution time of the completed submission.
    pub(crate) fn device_seconds(&self) -> Result<f64, CallError> {
        self.wait()?;
        match &*self.route {
            RouteSubmission::Committed(committed) => committed.device_seconds().map_err(CallError::Execution),
            RouteSubmission::Failed(failure) => Err(failure.outcome()),
        }
    }
}

/// The device queue a native submission runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeQueue {
    /// The device's compute queue.
    Compute,
    /// Fill work: moves into device memory and their waits, which overlap
    /// compute where the backend runs a fill queue beside it
    /// ([`BackendName::fills_beside_compute`]), and are compute work
    /// elsewhere.
    ///
    /// [`BackendName::fills_beside_compute`]: seismic_lang::registry::BackendName::fills_beside_compute
    Fill,
}

impl NativeQueue {
    pub(crate) const ALL: [Self; 2] = [Self::Compute, Self::Fill];

    /// The queue `backend` runs this queue's work on.
    fn on(self, backend: seismic_lang::registry::BackendName) -> Self {
        match self {
            Self::Fill if backend.fills_beside_compute() => Self::Fill,
            Self::Compute | Self::Fill => Self::Compute,
        }
    }
}

/// Encode `list` `repetitions` times as one unit of device work on `queue`
/// and commit it. `access` names every allocation the work touches exactly
/// once ([`merge_access`]); `retained` keeps the formed functions alive with
/// the submission. Before any of it runs, the device waits for the work of
/// the other native queue it must follow by what it touches
/// ([`Allocation::queue_order`]): order across queues follows from the
/// allocations alone, as within one.
pub(crate) fn submit(
    list: &impl DispatchList,
    access: &[(Arc<Allocation>, bool)],
    retained: Arc<dyn std::any::Any + Send + Sync>,
    repetitions: usize,
    observed: bool,
    queue: NativeQueue,
) -> Result<NativeSubmission, CallError> {
    if list.count() == 0 {
        return Err(CallError::Workflow(crate::api::WorkflowError::Empty));
    }
    // Owns retirement until the submission takes it. Declared before the
    // submission's permits and order lock so that unwinding releases them
    // before it waits.
    let mut issued = IssuedWork(None);
    // Host access is excluded while the work is encoded.
    let permits = access
        .iter()
        .map(|(allocation, write)| allocation.acquire_for_device(*write))
        .collect::<Vec<_>>();
    let first = list.dispatch(0, &mut Vec::new()).kernel;
    // Commit and fence recording in one order: allocation fences rely on
    // being recorded in the queue's execution order.
    let _order = first
        .public_device
        .native
        .order
        .lock()
        .expect("native submission order lock is never poisoned");
    let queue = queue.on(first.public_device.info().backend());
    // Each fence once: work touching several allocations another queue's
    // submission used waits for it once.
    let mut after: Vec<Arc<dyn DeviceCompletion>> = Vec::new();
    for (allocation, write) in access {
        for fence in allocation.queue_order(queue, *write).map_err(CallError::Execution)? {
            if !after.iter().any(|known| Arc::ptr_eq(known, &fence)) {
                after.push(fence);
            }
        }
    }
    let trace = first.public_device.active_trace();
    let encode_start = trace::host_seconds();
    let labels = trace.as_ref().map(|_| launch_labels(list, repetitions));
    let timed = trace
        .as_deref()
        .filter(|sink| sink.detail() == trace::TraceDetail::Launches)
        .map(|sink| (sink, labels.as_ref().map_or(0, Vec::len)));
    // Work committed before an issue failure runs on as this submission:
    // its caller receives it, and can answer a park it reached, before the
    // failure is reported by its outcome.
    let work = Encoding {
        list,
        repetitions,
        timed,
        observed: observed || trace.is_some(),
        retained: &retained,
        queue,
        after: &after,
    };
    let route = Arc::new(
        encode(first, &work)
            .map(RouteSubmission::Committed)
            .or_else(EncodeFailure::completion)?,
    );
    let route = issued.0.insert(route);
    // Fences are published before any trace allocation that could unwind:
    // later host access and the storage's release order after the work.
    let completion: Arc<dyn DeviceCompletion> = route.clone();
    for (allocation, write) in access {
        allocation.record_device_use(crate::driver::DeviceQueue::Native(queue), completion.clone(), *write);
    }
    #[cfg(test)]
    if FAIL_AFTER_PUBLICATION.replace(false) {
        panic!("injected failure after fence publication");
    }
    if let (Some(sink), Some(labels)) = (trace, labels) {
        sink.record(route.clone(), labels, encode_start, trace::host_seconds());
    }
    drop(permits);
    let gates = (0..=list.count())
        .flat_map(|boundary| list.timelines(boundary))
        .filter(|operation| operation.action == timeline::TimelineAction::Wait)
        .map(|operation| (operation.timeline.clone(), operation.value))
        .collect();
    Ok(NativeSubmission {
        route: issued.take(),
        gates,
        barriers: resident_barriers(list, access),
        _retained: retained,
    })
}

/// Each resident launch's barrier of `list` (`NativeLaunch::resident`), once:
/// the host address of its give-up word, with the allocation of `access`
/// that keeps it alive.
fn resident_barriers(list: &impl DispatchList, access: &[(Arc<Allocation>, bool)]) -> Vec<ResidentBarrier> {
    let mut barriers: Vec<ResidentBarrier> = Vec::new();
    let mut bound = Vec::new();
    for index in 0..list.count() {
        bound.clear();
        let dispatch = list.dispatch(index, &mut bound);
        for barrier in dispatch.launches.iter().flatten().filter_map(|launch| launch.barrier) {
            let (allocation, offset) = bound[dispatch.kernel.argument(barrier)];
            let base = allocation
                .storage()
                .host_address()
                .expect("a resident launch's barrier is host-addressed scratch: only Metal runs resident launches");
            // SAFETY: the barrier's checked bytes lie inside the bound view.
            let give_up = unsafe { base.as_ptr().add((offset + seismic_lang::registry::RESIDENT_GIVE_UP_BYTE) as usize) } as usize;
            if barriers.iter().any(|known| known.give_up == give_up) {
                continue;
            }
            let (held, _) = access
                .iter()
                .find(|(held, _)| allocation.shares_access_with(held))
                .expect("a submission's access names every allocation its launches bind");
            barriers.push(ResidentBarrier { give_up, _held: held.clone() });
        }
    }
    barriers
}

/// A resident launch's barrier, read once its submission completes.
struct ResidentBarrier {
    /// The host address of its give-up word.
    give_up: usize,
    /// Keeps the word's storage alive.
    _held: Arc<Allocation>,
}

impl ResidentBarrier {
    /// Whether a threadgroup of a launch over this barrier stopped waiting
    /// for the others: the device did not hold its grid at once, so the
    /// launch's results are not its kernel's. The word stays set, the one
    /// word of `sync` scratch no launch restores: a host reset races the
    /// work queued after it, and a grid sized at half of what the cores hold
    /// that still was not co-resident is a broken premise, so it is
    /// terminal (`ExecutionError::DeviceLost`).
    fn gave_up(&self) -> bool {
        // SAFETY: the address is a word of storage `_held` keeps alive; the
        // device's writes are visible once the work completed.
        unsafe { std::sync::atomic::AtomicU32::from_ptr(self.give_up as *mut u32) }
            .load(std::sync::atomic::Ordering::Acquire)
            != 0
    }
}

/// Issued work's retirement, until its [`NativeSubmission`] takes it: on
/// unwind it waits for the work, so nothing the work uses is released
/// while it runs.
struct IssuedWork(Option<Arc<RouteSubmission>>);

impl IssuedWork {
    fn take(mut self) -> Arc<RouteSubmission> {
        self.0.take().expect("issued work has a route")
    }
}

impl Drop for IssuedWork {
    fn drop(&mut self) {
        if let Some(route) = &self.0 {
            route.retire();
        }
    }
}

#[cfg(test)]
thread_local! {
    static FAIL_AFTER_PUBLICATION: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Evaluated physical work of every launch, in encode order. Recording it
/// requires no per-launch timestamps and does not change device dispatch.
fn launch_labels(list: &impl DispatchList, repetitions: usize) -> Vec<trace::TracedLaunch> {
    let mut labels = Vec::new();
    for _ in 0..repetitions {
        for index in 0..list.count() {
            let dispatch = list.dispatch(index, &mut Vec::new());
            let mut occurrences = vec![0; dispatch.launches.len()];
            for launch in dispatch
                .kernel
                .implementation
                .dispatch_order(dispatch.rounds)
            {
                labels.push(trace::TracedLaunch {
                    entry: dispatch.kernel.name.clone(),
                    launch,
                    occurrence: occurrences[launch],
                    geometry: dispatch.launches[launch].map(|geometry| {
                        trace::TracedLaunchGeometry {
                            workgroups: geometry.groups,
                            threads_per_workgroup: geometry.threads,
                            shared_bytes: geometry.shared_bytes,
                        }
                    }),
                    device: None,
                });
                occurrences[launch] += 1;
            }
        }
    }
    labels
}

/// One submission's work as its backend encodes it: `list` `repetitions`
/// times on `queue`, after the work of the other native queue `after` names,
/// which only a backend with a fill queue beside compute has. `timed` (a
/// launch-detail trace and the launch count) encodes every launch in its own
/// timed unit; `observed` says whether its device time is read; `retained`
/// keeps the formed functions alive with the submission.
struct Encoding<'a, L> {
    list: &'a L,
    repetitions: usize,
    timed: Option<(&'a trace::TraceSink, usize)>,
    observed: bool,
    retained: &'a Arc<dyn std::any::Any + Send + Sync>,
    queue: NativeQueue,
    after: &'a [Arc<dyn DeviceCompletion>],
}

/// Encode and commit `work`.
fn encode(first: &NativePrepared, work: &Encoding<'_, impl DispatchList>) -> Result<Committed, EncodeFailure> {
    let Encoding { list, repetitions, timed, observed, retained, queue, after } = *work;
    #[cfg(target_os = "macos")]
    let mut buffers = Vec::new();
    assert!(
        matches!(first.route, NativeRoute::Cuda { .. }) || (queue == NativeQueue::Compute && after.is_empty()),
        "only a backend with a fill queue beside compute runs native work on another queue"
    );
    match &first.route {
        NativeRoute::Cpu(route) => {
            let (outcome, interval) = cpu::run(&route.opened, list, repetitions);
            Ok(Committed::Cpu { outcome, interval })
        }
        #[cfg(target_os = "macos")]
        NativeRoute::Metal { opened, .. } => {
            type Metal = seismic_metal::Metal;
            type Executor = seismic_metal::MetalExecutor;
            let mut batch = match timed {
                Some((sink, _)) => {
                    seismic_metal::DirectBatch::timed(opened.service(), sink.metal_timestamps())
                }
                None if observed => seismic_metal::DirectBatch::observed(opened.service()),
                None => seismic_metal::DirectBatch::new(opened.service()),
            }
            .map_err(CallError::Execution)
            .map_err(EncodeFailure::uncommitted)?;
            let mut typed = Vec::new();
            let timelines = |batch: &mut seismic_metal::DirectBatch, boundary: usize| {
                for operation in list.timelines(boundary) {
                    let timeline::TimelineKind::Metal(timeline) = operation.timeline.kind() else {
                        unreachable!("a run binds timelines of its own device");
                    };
                    match operation.action {
                        timeline::TimelineAction::Wait => batch
                            .wait_event(&timeline.gate, operation.value)
                            .map_err(CallError::Execution)?,
                        timeline::TimelineAction::Signal => {
                            batch.signal_event(&timeline.reached, operation.value)
                        }
                    }
                }
                Ok::<(), CallError>(())
            };
            // Encode launch `ordinal` of dispatch `index`.
            let mut launch = |batch: &mut seismic_metal::DirectBatch, index: usize, ordinal: usize| {
                buffers.clear();
                let dispatch = list.dispatch(index, &mut buffers);
                let NativeRoute::Metal {
                    launches: pipelines,
                    ..
                } = &dispatch.kernel.route
                else {
                    unreachable!("one device has one native route");
                };
                let Some(launch) = &dispatch.launches[ordinal] else {
                    batch.skip();
                    return Ok(());
                };
                // A launch its conditions leave out of this run: the cached
                // order of the others stands.
                if !list.encodes(index, ordinal) {
                    batch.skip();
                    return Ok(());
                }
                typed.clear();
                typed.extend(buffers.iter().map(|(allocation, offset)| {
                    (crate::backends::metal_binding(allocation), *offset)
                }));
                // A grid's binding is its three words: a `&tensor[3] i32`
                // binds with canonical strides and never as a slab
                // view (`NativePrepared::reads_grid`).
                let grid = launch
                    .grid
                    .map(|buffer| typed[dispatch.kernel.argument(buffer)]);
                batch
                    .encode(&seismic_metal::DirectLaunch {
                        pipeline: pipelines.pipeline(ordinal),
                        buffers: &typed,
                        words: word_bytes(dispatch.words),
                        scalar_results: dispatch
                            .kernel
                            .scalar_slots()
                            .map(|slots| (typed_buffer::<Metal, Executor>(slots), 0)),
                        threadgroups: launch.groups,
                        grid,
                        threads_per_threadgroup: launch.threads,
                        threadgroup_bytes: launch.shared_bytes,
                    })
                    .map_err(CallError::Execution)
            };
            let mut encode_all = |batch: &mut seismic_metal::DirectBatch| {
                if timed.is_some() {
                    // A timed batch times each launch in its own encoder, in
                    // submission order: its labels are in that order.
                    for _ in 0..repetitions {
                        for index in 0..list.count() {
                            timelines(batch, index)?;
                            let dispatch = list.dispatch(index, &mut Vec::new());
                            for ordinal in dispatch
                                .kernel
                                .implementation
                                .dispatch_order(dispatch.rounds)
                            {
                                launch(batch, index, ordinal)?;
                            }
                        }
                        timelines(batch, list.count())?;
                    }
                } else {
                    let mut scheduled = |schedule: &hazard::Schedule| {
                        for step in &schedule.0 {
                            match *step {
                                hazard::Scheduled::Timelines(boundary) => {
                                    timelines(batch, boundary)?
                                }
                                hazard::Scheduled::Level(launches) => batch.group(launches),
                                hazard::Scheduled::Launch { index, ordinal } => {
                                    launch(batch, index, ordinal)?
                                }
                            }
                        }
                        Ok(())
                    };
                    let derive = || hazard::Schedule(hazard::schedule(list, repetitions));
                    match list.runs() {
                        Some(runs) => {
                            // A schedule holds no parameter: its shape is
                            // its plans, its repetitions and how their
                            // reaches meet, and it binds nothing.
                            let mut key = runs.iter().map(|run| run.plan).collect::<Vec<_>>();
                            key.push(repetitions as u64);
                            key.extend(hazard::relative_placement(list, 0..list.count()));
                            let shaped = graph_replays::Shaped { key, bound: Vec::new() };
                            let schedules = &first.public_device.native.metal_schedules;
                            // Deriving a schedule is host work alone: it
                            // waits for nothing.
                            let budget = first.public_device.native.graph_budget();
                            let taken = schedules.record(std::slice::from_ref(&shaped), budget, retained, || Ok(()), |_| Ok(derive()))?;
                            schedules.replay(&shaped, taken[0], |_, _| unreachable!("a schedule binds no parameter"), |schedule| scheduled(schedule))?
                        }
                        None => scheduled(&derive())?,
                    }
                }
                Ok::<(), CallError>(())
            };
            match encode_all(&mut batch) {
                Ok(()) => Ok(Committed::Metal(batch.commit())),
                Err(error) => Err(EncodeFailure {
                    error,
                    committed: batch.abandon().map(Committed::Metal),
                }),
            }
        }
        NativeRoute::Cuda { opened, .. } => cuda::encode(
            opened.service(),
            &first.public_device.native.cuda_replays,
            first.public_device.native.graph_budget(),
            work,
        ),
        #[cfg(not(target_os = "macos"))]
        NativeRoute::Vulkan { opened, .. } => vulkan::encode(
            opened.service(),
            &first.public_device.native.vulkan_replays,
            first.public_device.native.graph_budget(),
            work,
        ),
    }
}

/// Encoding failed: the error, and the device work committed before it,
/// which runs on the storage the work touches ([`Self::completion`]).
pub(super) struct EncodeFailure {
    pub(super) error: CallError,
    pub(super) committed: Option<Committed>,
}

impl EncodeFailure {
    /// The submission's outcome: work committed before the failure is
    /// issued work whose completion reports the failure; only entirely
    /// unissued work is refused at once.
    fn completion(self) -> Result<RouteSubmission, CallError> {
        match self.committed {
            Some(submitted) => Ok(RouteSubmission::Failed(Box::new(IssuedFailure {
                submitted,
                issue: self.error,
            }))),
            None => Err(self.error),
        }
    }

    /// A failure before any device work was committed.
    fn uncommitted(error: CallError) -> Self {
        Self {
            error,
            committed: None,
        }
    }
}

fn validate_output(
    result: usize,
    tensor: &Arc<TensorInner>,
    device: DeviceIdentity,
    representation: RepresentationId,
    extents: &[u64],
    args: &EncodedArgs,
    prior_outputs: &[Arc<TensorInner>],
) -> Result<(), CallError> {
    if tensor.allocation().storage().read_only() {
        return Err(CallError::Execution(ExecutionError::SubmissionFailed(
            "native result cannot write a mapped tensor".into(),
        )));
    }
    let descriptor = tensor.descriptor();
    if descriptor.device != device {
        return Err(CallError::Output(OutputError::WrongDevice { result }));
    }
    if descriptor.representation != representation {
        return Err(CallError::Output(OutputError::WrongRepresentation {
            result,
        }));
    }
    if descriptor.extents.len() != extents.len() {
        return Err(CallError::Output(OutputError::ShapeMismatch {
            result,
            axis: descriptor.extents.len().min(extents.len()),
        }));
    }
    for (axis, (actual, expected)) in descriptor.extents.iter().zip(extents).enumerate() {
        if actual != expected {
            return Err(CallError::Output(OutputError::ShapeMismatch {
                result,
                axis,
            }));
        }
    }
    let layout = crate::layout::canonical(representation, extents).map_err(CallError::Execution)?;
    if descriptor.strides != layout.strides || descriptor.byte_len != layout.byte_len {
        return Err(CallError::Output(OutputError::NoncanonicalLayout {
            result,
        }));
    }
    let overlaps = |other: &Arc<TensorInner>| {
        let other = other.descriptor();
        descriptor.allocation == other.allocation
            && descriptor.byte_offset < other.byte_offset.saturating_add(other.byte_len)
            && other.byte_offset < descriptor.byte_offset.saturating_add(descriptor.byte_len)
    };
    if args.tensors().flatten().any(|other| overlaps(other))
        || prior_outputs.iter().any(|other| overlaps(other))
    {
        return Err(CallError::Output(OutputError::IllegalAliasing { result }));
    }
    Ok(())
}

fn tensor_error(error: crate::api::TensorError) -> CallError {
    match error {
        crate::api::TensorError::Execution(error) => CallError::Execution(error),
        other => CallError::Execution(ExecutionError::AllocationFailed(other.to_string())),
    }
}

fn evaluate_compiled(
    expression: &CompiledNat,
    values: &InvocationValues,
) -> Result<u64, CallError> {
    expression.evaluate_u64(values).map_err(|error| {
        CallError::Execution(ExecutionError::SubmissionFailed(format!(
            "native ABI expression failed after invocation validation: {error:?}"
        )))
    })
}

fn native_words(
    schema: &CallSchema,
    implementation: &NativeImplementation,
    specialization: &NativeSpecialization,
    arguments: &[ArgumentValue],
    results: &[Option<NativeTensorSpec>],
    values: &InvocationValues,
) -> Result<Vec<u64>, CallError> {
    let runtime = abi::runtime_parameters(implementation);
    let mut words = Vec::with_capacity(abi::word_count(schema) + runtime.len());
    for dimension in schema.dimensions() {
        match values.get(dimension.symbol) {
            Some(SymbolValue::Nat(value)) => words.push(word(SymbolValue::Nat(value))?),
            _ => panic!("validated invocation omitted a native ABI dimension"),
        }
    }
    for (parameter, argument) in schema.parameters().iter().zip(arguments) {
        match (&parameter.kind, argument) {
            (ParameterKind::Tensor { .. }, ArgumentValue::Tensor(tensor)) => {
                words.extend_from_slice(&tensor.extents);
                words.extend_from_slice(&tensor.strides);
            }
            (ParameterKind::Tensor { .. }, _) => {
                panic!("validated native tensor argument is not a tensor")
            }
            (ParameterKind::Scalar { symbol, .. } | ParameterKind::Index { symbol, .. }, _) => {
                words.push(word(
                    values.get(*symbol).expect("validated scalar disappeared"),
                )?);
            }
            (ParameterKind::Range { start, end, .. }, _) => {
                words.push(word(
                    values
                        .get(*start)
                        .expect("validated range start disappeared"),
                )?);
                words.push(word(
                    values.get(*end).expect("validated range end disappeared"),
                )?);
            }
        }
    }
    for spec in results.iter().flatten() {
        words.extend_from_slice(&spec.extents);
        words.extend_from_slice(&spec.strides);
    }
    for address in runtime {
        words.push(address.value(specialization));
    }
    Ok(words)
}

fn word(value: SymbolValue) -> Result<u64, CallError> {
    value.try_word64().map_err(|error| {
        CallError::Execution(ExecutionError::ConstructionContradiction(format!(
            "native ABI quantity does not fit its word: {error:?}"
        )))
    })
}

fn scalar_value(dtype: seismic_lang::types::DType, value: u64) -> ArgumentValue {
    match dtype {
        seismic_lang::types::DType::F32 => ArgumentValue::F32(f32::from_bits(value as u32)),
        seismic_lang::types::DType::F16 => ArgumentValue::F16(value as u16),
        seismic_lang::types::DType::BF16 => ArgumentValue::BF16(value as u16),
        seismic_lang::types::DType::I32 => ArgumentValue::I32(value as u32 as i32),
        seismic_lang::types::DType::U32 => ArgumentValue::U32(value as u32),
        seismic_lang::types::DType::U64 => ArgumentValue::U64(value),
        seismic_lang::types::DType::Bool => ArgumentValue::Bool(value != 0),
    }
}

fn read_word(bytes: &[u8], word: usize) -> u64 {
    let start = word * ResultClass::WORD_BYTES as usize;
    u64::from_le_bytes(
        bytes[start..start + ResultClass::WORD_BYTES as usize]
            .try_into()
            .expect("native scalar word"),
    )
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use seismic_lang::checked::{check_source_with, SourceFile, SourceSet};

    /// A module built without seismic-build (which rejects this at build
    /// time) reaches preparation, which rejects it before compiling.
    #[test]
    #[ignore = "requires Metal device"]
    fn metal_preparation_rejects_implementations_beyond_the_buffer_table() {
        let parameters = (0..29)
            .map(|index| format!("x{index}: &tensor[N] f32"))
            .collect::<Vec<_>>()
            .join(", ");
        let module = check_source_with(
            SourceSet::new(vec![SourceFile {
                path: "wide.seismic".into(),
                text: format!("fn wide[N]({parameters}) -> tensor[N] f32:\n    let mut output = tensor[N] f32\n    for i in 0..N:\n        output[i] = x0[i]\n    return output\n\nnative wide for metal from \"wide.metal\":\n    launch wide:\n        threadgroups (1, 1, 1)\n        threads_per_threadgroup (1, 1, 1)\n"),
            }]),
            &mut |_entry: &str, _native: &NativeImplementation| {
                Ok::<_, std::convert::Infallible>(Some("kernel void wide() {}\n".into()))
            },
        )
        .expect("wide source checks");
        let entry = module.entry_named("wide").expect("wide entry");
        let device = crate::devices::Devices::discover(&[])
            .expect("device discovery")
            .open_backend(BackendName::Metal)
            .expect("Metal device");
        match NativePrepared::prepare(
            &device,
            &module,
            entry,
            ElementBindings::default(),
            NativeSpecialization::new(),
            None,
        ) {
            Err(PrepareError::Preparation(PreparationError::NativeBufferSlots {
                entry,
                slots,
                limit,
            })) => {
                assert_eq!(entry, "wide");
                assert_eq!(slots, 32);
                assert_eq!(limit, 31);
            }
            Err(other) => panic!("unexpected preparation error: {other:?}"),
            Ok(_) => panic!("32 Metal buffer slots were prepared"),
        }
    }

    /// A resident launch's grid is `RESIDENT`: half of what the cores hold
    /// of its pipeline at once, never more than its work bound, and the
    /// launch names its barrier's ABI buffer.
    #[test]
    #[ignore = "requires Metal device"]
    fn a_resident_launch_dispatches_at_most_half_of_what_the_cores_hold() {
        let module = check_source_with(
            SourceSet::new(vec![SourceFile {
                path: "spread.seismic".into(),
                text: "fn spread[N](x: &mut tensor[N] f32):\n    for i in 0..N:\n        x[i] = 0.0\n\nnative spread for metal from \"spread.metal\":\n    scratch barrier bytes (16) sync\n    launch spread resident (barrier):\n        threadgroups (N, 1, 1)\n        threads_per_threadgroup (256, 1, 1)\n".into(),
            }]),
            &mut |_entry: &str, _native: &NativeImplementation| {
                Ok::<_, std::convert::Infallible>(Some("kernel void spread() {}\n".into()))
            },
        )
        .expect("spread source checks");
        let entry = module.entry_named("spread").expect("spread entry");
        let device = crate::devices::Devices::discover(&[])
            .expect("device discovery")
            .open_backend(BackendName::Metal)
            .expect("Metal device");
        let prepared = NativePrepared::prepare(
            &device,
            &module,
            entry,
            ElementBindings::default(),
            NativeSpecialization::new(),
            None,
        )
        .expect("spread prepares");
        let NativeRoute::Metal { opened, launches } = &prepared.route else {
            unreachable!("a Metal device's route");
        };
        let cores = opened.device_description().facts().properties().units.get();
        let per_core = launches.pipeline(0).max_threads_per_threadgroup() / 256;
        let resident = (cores * per_core / 2).max(1);
        for (work, expected) in [(1, 1), (resident, resident), (resident + 1, resident), (1 << 20, resident)] {
            let geometry = prepared
                .launches_at(&|name| (name == "N").then_some(work))
                .expect("spread's launches")
                .geometry[0]
                .expect("the launch is active");
            assert_eq!(geometry.groups, [expected, 1, 1], "work {work}, {cores} cores");
            assert_eq!(geometry.barrier, Some(1), "the barrier follows x");
        }
    }

    /// A launch condition reads its own word of its parameter: of a launch
    /// under `unless (policy[2])` and `if (policy[0])`, word 2 and word 0
    /// of the bound view decide alone, at the view's offset (one rule,
    /// `condition_holds`, every run-encoding backend reads).
    #[test]
    #[ignore = "requires Metal device"]
    fn an_indexed_launch_condition_reads_its_word() {
        let module = check_source_with(
            SourceSet::new(vec![SourceFile {
                path: "policy.seismic".into(),
                text: "const WHOLE = 2\n\nfn fill[N](policy: &tensor[3] i32, x: &mut tensor[N] f32):\n    for i in 0..N:\n        x[i] = 0.0\n\nnative fill for metal from \"fill.metal\":\n    launch fill:\n        threadgroups (1, 1, 1)\n        unless (policy[WHOLE])\n        if (policy[0])\n        threads_per_threadgroup (1, 1, 1)\n".into(),
            }]),
            &mut |_entry: &str, _native: &NativeImplementation| {
                Ok::<_, std::convert::Infallible>(Some("kernel void fill() {}\n".into()))
            },
        )
        .expect("policy source checks");
        let entry = module.entry_named("fill").expect("fill entry");
        let device = crate::devices::Devices::discover(&[])
            .expect("device discovery")
            .open_backend(BackendName::Metal)
            .expect("Metal device");
        let prepared = NativePrepared::prepare(
            &device,
            &module,
            entry,
            ElementBindings::default(),
            NativeSpecialization::new(),
            None,
        )
        .expect("fill prepares");
        // A view of three words past a leading word, so the offset counts.
        let allocation = device.allocate_mapped_zeroed(seismic_topology::Contract::Words, 16, 16).expect("shared words");
        let base = allocation.storage().host_address().expect("host-addressed words");
        let store = |word: usize, value: u32| unsafe {
            std::sync::atomic::AtomicU32::from_ptr(base.as_ptr().add(4 + 4 * word).cast())
                .store(value, std::sync::atomic::Ordering::Release)
        };
        // The launch's conditions read `policy`, buffer 0, bound at byte 4.
        let encodes = || {
            prepared.conditions(0).iter().all(|&(buffer, index, encoded)| {
                assert_eq!(buffer, 0, "the conditions read policy");
                condition_holds(&allocation, 4 + 4 * index, encoded)
            })
        };
        for (set, cleared, encoded) in [(1, 0, true), (1, 1, false), (0, 0, false), (0, 1, false)] {
            store(0, set);
            store(2, cleared);
            // The other word never decides.
            store(1, 7);
            assert_eq!(encodes(), encoded, "policy[0] {set}, policy[2] {cleared}");
        }
    }
}

/// One sealed plan's run in a submission ([`DispatchList::runs`]): the
/// plan, what its graphs are keyed by (`graph::PlanShape`), the run's calls,
/// its timeline nodes, each where it sits from the run's first call (where
/// graphs hold none, each ends a graph whether or not the run binds it,
/// [`graph_segments`]), where the span marks its graphs hold sit, whether
/// its plan is launched (`NativeGraphDraft::seal_launched`), and whether
/// its graph holds its timeline nodes.
#[derive(Clone)]
pub(crate) struct PlanRun {
    pub(crate) plan: u64,
    pub(crate) layout: u64,
    pub(crate) calls: Range<usize>,
    pub(crate) timelines: Vec<(usize, timeline::TimelineNode)>,
    /// The span marks inside its graphs, each the call it precedes, in
    /// order (`NativeGraphPlan::held_marks`).
    pub(crate) marks: Vec<usize>,
    pub(crate) launched: bool,
    /// Whether its graph holds the timeline nodes inside it, as nodes of
    /// its own (`NativeGraphPlan::holds_timelines`).
    pub(crate) holds_timelines: bool,
    /// Its layout's graph ([`LayoutGraph`]): its words, a word per call of
    /// the run, and the plans it is recorded from.
    pub(crate) graph: Arc<LayoutGraph>,
}

/// The calls of `list` a graph each, every range with the boundary that
/// ends it: those between timeline operations and, where the list is made
/// of `runs`, between runs and at each run's timeline nodes its graphs do
/// not hold (a graph is one run's, divided where its plan may bind a
/// timeline, so a run binding none there divides alike), from a leading
/// empty range for the operations before the first call to the range
/// ending at the list's end. A boundary that divides only runs or unbound
/// nodes has no timeline operations. A run whose graph holds the timeline
/// nodes inside it ([`PlanRun::holds_timelines`]) is not divided at them.
/// A stamp divides only where no graph holds
/// its mark ([`PlanRun::marks`]): at a run's edges and in lists launched one
/// by one, where it falls between launches on the stream and a graph is not
/// cut.
pub(super) fn graph_segments(list: &impl DispatchList, runs: Option<&[PlanRun]>) -> Vec<(Range<usize>, usize)> {
    let runs = runs.unwrap_or_default();
    let starts = runs
        .iter()
        .flat_map(|run| {
            let bounds = run.timelines.iter().filter(|_| !run.holds_timelines);
            std::iter::once(run.calls.start).chain(bounds.map(|&(position, _)| run.calls.start + position))
        })
        .collect::<std::collections::BTreeSet<_>>();
    let mut segments = Vec::new();
    let mut first = 0;
    for boundary in 0..=list.count() {
        let divides = boundary > first && starts.contains(&boundary);
        let inside = || runs.iter().find(|run| run.inside(boundary));
        let held = || inside().is_some_and(|run| run.holds(boundary));
        let stamped = !list.stamps(boundary).is_empty() && !held();
        let timelines = !list.timelines(boundary).is_empty()
            && !inside().is_some_and(|run| run.holds_timelines);
        if boundary == list.count() || timelines || stamped || divides {
            segments.push((first..boundary, boundary));
            first = boundary;
        }
    }
    segments
}

impl PlanRun {
    /// Whether `boundary` of the list lies inside the run, between two of
    /// its calls.
    fn inside(&self, boundary: usize) -> bool {
        self.calls.start < boundary && boundary < self.calls.end
    }

    /// Whether a graph of the run holds its marks at `boundary` of the list.
    fn holds(&self, boundary: usize) -> bool {
        self.inside(boundary) && self.marks.binary_search(&(boundary - self.calls.start)).is_ok()
    }

    /// The run's timeline nodes its graph of `calls` holds, each the call of
    /// the list it precedes, with what it does, in order: those strictly
    /// inside the graph, of a run whose graph holds them.
    pub(super) fn held_timelines<'a>(&'a self, calls: &Range<usize>) -> impl Iterator<Item = (usize, timeline::TimelineNode)> + 'a {
        let inside = calls.start + 1..calls.end;
        let start = self.calls.start;
        self.timelines
            .iter()
            .filter(move |_| self.holds_timelines)
            .map(move |&(position, node)| (start + position, node))
            .filter(move |(at, _)| inside.contains(at))
    }
}

/// The shapes a sealed-plan submission's graphs are recorded and replayed
/// under ([`graph_replays::Replays`]): a graph is one run's calls between
/// timeline nodes ([`graph_segments`]), and its key is what fixes its form
/// wherever it runs (`graph::PlanShape`): on CUDA the run's layout, the
/// calls' place in the run and the words the backend adds (`form`: the
/// launches its layout's graph holds of each call, [`LayoutForms`]),
/// every class, block, place and encoding of the layout replaying the
/// graph; elsewhere the run's plan with the launches the calls encode. No
/// key holds an address: a run on any place or slot replays its shape's
/// graph, updated to what it binds.
pub(super) struct ReplayKeys<'a> {
    pub(super) runs: Vec<PlanRun>,
    /// The launches each call encodes, where they are part of the shape
    /// (none on CUDA, which enables a graph's launches by parameter).
    pub(super) encoded: Option<&'a [Vec<bool>]>,
}

impl ReplayKeys<'_> {
    /// The graph of the segment of `calls`, which lies in one run, with the
    /// backend's words `form` and the storage it binds, `bound`.
    pub(super) fn shaped(&self, calls: &Range<usize>, form: impl IntoIterator<Item = u64>, bound: Vec<u64>) -> graph_replays::Shaped {
        graph_replays::Shaped { key: self.key(calls, form), bound }
    }

    /// The plan of the run `calls` lie in.
    pub(super) fn plan(&self, calls: &Range<usize>) -> u64 {
        self.run(calls).plan
    }

    /// Whether the run `calls` lie in is of a launched plan
    /// (`NativeGraphDraft::seal_launched`).
    pub(super) fn launched(&self, calls: &Range<usize>) -> bool {
        self.run(calls).launched
    }

    /// The words of the graph of `calls`, which lie in one run, a word per
    /// call ([`LayoutForms`]).
    pub(super) fn form(&self, calls: &Range<usize>) -> &[u64] {
        let run = self.run(calls);
        &run.graph.form[calls.start - run.calls.start..calls.end - run.calls.start]
    }

    /// The marks the graph of `calls` holds, each the call of the list it
    /// precedes, in order: those strictly inside it.
    pub(super) fn marks(&self, calls: &Range<usize>) -> impl Iterator<Item = usize> + '_ {
        let run = self.run(calls);
        let inside = calls.start + 1..calls.end;
        run.marks.iter().map(|mark| run.calls.start + mark).filter(move |mark| inside.contains(mark))
    }

    /// The stamps of a sampled run at the marks the graph of `calls` holds,
    /// in order; none where the run isn't sampled.
    pub(super) fn stamps<'l>(&self, list: &'l impl DispatchList, calls: &Range<usize>) -> Vec<&'l spans::Stamp> {
        let mut marks = self.marks(calls).collect::<Vec<_>>();
        marks.dedup();
        marks.into_iter().flat_map(|mark| list.stamps(mark)).collect()
    }

    /// The run `calls` lie in.
    fn run(&self, calls: &Range<usize>) -> &PlanRun {
        let at = self.runs.partition_point(|run| run.calls.end <= calls.start);
        &self.runs[at]
    }

    /// The key of the segment of `calls`, which lies in one run, with the
    /// backend's words `form`.
    fn key(&self, calls: &Range<usize>, form: impl IntoIterator<Item = u64>) -> Vec<u64> {
        let run = self.run(calls);
        assert!(
            run.calls.start <= calls.start && calls.end <= run.calls.end,
            "a graph's calls lie in one run"
        );
        let shape = match self.encoded {
            None => run.layout,
            Some(_) => run.plan,
        };
        let mut key = vec![shape, (calls.start - run.calls.start) as u64, (calls.end - run.calls.start) as u64];
        if let Some(encoded) = self.encoded {
            key.extend(encoded_words(encoded, calls.clone()));
        }
        key.extend(form);
        key
    }
}

/// The encoded launches of `calls` as words of a replay key, one bit a
/// launch: two submissions of one plan encoding different launches replay
/// different graphs.
pub(super) fn encoded_words(encoded: &[Vec<bool>], calls: Range<usize>) -> Vec<u64> {
    let mut words = Vec::new();
    for (bit, launch) in encoded[calls].iter().flatten().enumerate() {
        if bit % 64 == 0 {
            words.push(0);
        }
        if *launch {
            *words.last_mut().expect("a word per 64 launches") |= 1 << (bit % 64);
        }
    }
    words
}

#[cfg(test)]
mod issue_lifecycle_tests {
    use super::*;
    use seismic_lang::checked::{check_source, SourceFile, SourceSet};

    fn cpu_noop() -> Arc<NativePrepared> {
        use seismic_native_cpu::Tier;
        unsafe fn unused(_: &CpuInvocation<'_>, _: [u64; 3], _: &mut [u8]) {}
        const fn variant(tier: Tier) -> CpuVariant {
            CpuVariant { tier, elements: &[], function: unused }
        }
        static KERNELS: CpuNativeKernels = CpuNativeKernels {
            digest: "issue-lifecycle-unwind",
            elements: &[],
            launches: &[CpuLaunchVariants {
                kernel: "entry",
                variants: &[
                    variant(Tier::X86V2), variant(Tier::X86V3), variant(Tier::X86V4),
                    variant(Tier::X86V4Vnni), variant(Tier::Neon),
                ],
            }],
        };
        let module = check_source(SourceSet::new(vec![SourceFile {
            path: "unwind.seismic".into(),
            text: "fn entry(x: &mut tensor[1] f32):\n    return\n\nnative entry for cpu from \"entry.rs\":\n    launch entry:\n        threadgroups (1, 1, 1)\n        threads_per_threadgroup (1, 1, 1)\n".into(),
        }])).unwrap();
        let device = crate::devices::Devices::discover(&[]).unwrap().open_backend(BackendName::Cpu).unwrap();
        NativePrepared::prepare(&device, &module, module.entry_named("entry").unwrap(),
            ElementBindings::new(), NativeSpecialization::new(), Some(&KERNELS)).unwrap()
    }

    #[test]
    fn issued_failure_returns_its_completion_and_reports_issue_and_retirement() {
        let issue = CallError::Workflow(crate::api::WorkflowError::Empty);
        assert!(matches!(EncodeFailure::uncommitted(issue.clone()).completion(), Err(error) if error == issue));
        for outcome in [Ok(()), Err(ExecutionError::SubmissionFailed("retire failed".into()))] {
            let route = EncodeFailure {
                error: issue.clone(),
                committed: Some(Committed::Cpu { outcome: outcome.clone(), interval: (1., 2.) }),
            }
            .completion()
            .ok()
            .expect("issued work returns its completion");
            assert_eq!(route.completed(), outcome.clone().map(|()| true));
            let expected = match outcome {
                Ok(()) => issue.clone(),
                Err(fence) => CallError::Unfenced { issue: Box::new(issue.clone()), fence },
            };
            assert_eq!(route.finish(), Err(expected));
        }
    }

    /// A failure after the work is issued and fenced unwinds through its
    /// retirement, releasing the submission's permits and order lock: the
    /// storage it touched stays ordered after the work and usable.
    #[test]
    fn unwinding_after_issue_retires_the_work_and_releases_submission_locks() {
        let kernel = cpu_noop();
        let device = kernel.public_device.clone();
        let tensor = Arc::new(TensorInner::zeros(&device, seismic_lang::registry::dense(
            seismic_lang::types::DType::F32), &[1]).unwrap());
        let mut args = EncodedArgs::new();
        args.push_tensor(tensor.clone());
        let call = kernel.prepare_call(&mut ScratchArena::default(), args, None, Parking::Unanswered).unwrap();
        let calls = [call];
        FAIL_AFTER_PUBLICATION.set(true);
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            StandaloneCalls { kernel: &kernel, calls: &calls }.submit(1, false).map(drop)
        }));
        assert!(outcome.is_err(), "the injected failure unwinds");
        assert!(!FAIL_AFTER_PUBLICATION.get());
        assert!(!matches!(device.native.order.try_lock(), Err(std::sync::TryLockError::WouldBlock)),
            "the order lock is released");
        assert!(tensor.allocation().try_acquire(true).is_some(), "the device permit is released");
    }

    #[test]
    fn standalone_awaited_calls_require_kernel_parking() {
        for borrow in ["&", "&await ", "&await mut "] {
            let module = check_source(SourceSet::new(vec![SourceFile {
                path: "standalone-park.seismic".into(),
                text: format!("fn entry(words: {borrow}tensor[1] i32):\n    return\n"),
            }])).unwrap();
            let entry = module.entry(module.entry_named("entry").unwrap(),
                &seismic_lang::entry::ElementBindings::default()).unwrap();
            for backend in [BackendName::Cpu, BackendName::Metal, BackendName::Cuda, BackendName::Vulkan] {
                let result = validate_standalone_parking(entry.schema(), backend, Parking::Unanswered);
                if borrow.contains("await") && !backend.parks_in_kernels() {
                    assert_eq!(result, Err(CallError::Workflow(crate::api::WorkflowError::ParkRequiresGraph)));
                } else {
                    result.unwrap();
                }
                // Answered, it waits for nothing: every backend calls it.
                validate_standalone_parking(entry.schema(), backend, Parking::Answered).unwrap();
            }
        }
    }
}
