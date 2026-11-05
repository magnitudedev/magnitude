//! Ordered target execution through sealed Seismic native graphs. A step
//! queues one run per graph without waiting; the device queue orders them,
//! and the returned submission owns the launch until completion is observed.
//! A chain of steps interleaves their runs block by block, as a wavefront
//! at a scratch-served layer: each step's back there, then its next block,
//! so the next layer's requests are published while the layer's later
//! backs wait.

use super::{
    Branded, Chain, DeviceSubmission, TargetProgram,
    graph::readout::{
        BoundTargetReadoutGraphs, ReadoutClass, ReadoutKind, certifies, shapes, write_selection,
    },
    native_target_graph::{
        BankRun, BoundTargetGraphs, EntryTokens, GroupBuffers, LaunchControl, LaunchControlLayout, TokenRow,
    },
};
use crate::device_table::{SelectedRead, SubmittedRead};
use crate::host_tables::HostTableError;
use crate::stored_table::{NoticeStep, StepRows, StoredTable};
use crate::operators::block::{BlockControlPorts, BlockStatePorts};
use crate::experts::StepRoute;
use crate::operators::routed::split::{RunStage, SplitPorts, SplitValues};
use crate::{
    ConditioningRef, ConditioningSlice, DeviceError, GraphOutputTensor, InvariantError,
    NativeGraphOutputLease, NativeGraphWorkspaceLease, SubmitError, TargetGraphOutputLease,
    TargetGraphWorkspaceLease, TargetLaunchCore, TargetLaunchWorkspace, TargetTokens,
    ValidatedTargetLaunch, completion::CompletionWaiter, native::AttestedState,
};
use magnitude_batching::{Demand, TargetBatchUpload};
use magnitude_family_contracts::Decoder;
use magnitude_kernels::{conditioning_overlay, feature_copy};
use magnitude_state::LayerRef;
use seismic::{
    Device, Element, NativeGraphCompletion, NativeGraphOutputs, NativeGraphPlan,
    NativeGraphSequence, NativePort, NativeQueue, Tensor,
};
use std::{cell::RefCell, collections::BTreeMap, rc::Rc, sync::Arc, time::Instant};

fn invalid(detail: impl Into<String>) -> SubmitError {
    SubmitError::Invariant(InvariantError {
        context: "native target program",
        detail: detail.into(),
    })
}

fn device(error: impl ToString) -> SubmitError {
    SubmitError::Device(DeviceError::Execution(error.to_string()))
}

/// Host tokens as entry input rows: the `sample_rows` result layout
/// (token, status), status 0.
fn token_rows(tokens: &[i32]) -> Vec<u8> {
    tokens
        .iter()
        .flat_map(|token| [*token, 0])
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// A launch's encoded visible spans, `segments` a row, with each row's
/// padded to `padded` spans by empty ones.
fn padded_spans(visible: &[u8], segments: u64, padded: u64) -> Vec<u8> {
    let (row, padded) = (segments as usize * 8, padded as usize * 8);
    visible
        .chunks_exact(row)
        .flat_map(|spans| {
            spans
                .iter()
                .copied()
                .chain(std::iter::repeat_n(0, padded - row))
        })
        .collect()
}

/// Whether a selected read scores in a launch whose rows read the block-key
/// history `blocks` over `segments` spans a row: whether some row has more
/// than `selected` complete blocks, the block rows it sees and its own when
/// it completes one (`indexer_select`). A launch where none does reads every
/// row's whole history through the class that does not score.
fn selection_scores(blocks: &HistoryControls, segments: u64, selected: u64) -> bool {
    let word = |bytes: &[u8], at: usize| {
        i64::from(i32::from_le_bytes([
            bytes[at],
            bytes[at + 1],
            bytes[at + 2],
            bytes[at + 3],
        ]))
    };
    let row = (segments as usize * 8).max(1);
    blocks
        .visible
        .chunks_exact(row)
        .zip(blocks.destinations.chunks_exact(4))
        .any(|(spans, destination)| {
            let seen = spans
                .chunks_exact(8)
                .map(|span| word(span, 4) - word(span, 0))
                .sum::<i64>();
            let own = i64::from(word(destination, 0) >= 0);
            seen + own > selected as i64
        })
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// Step `sequence`'s words (`routed_step`'s `sequence`): the u64 sequence
/// as its pair, STEP_LOW and STEP_HIGH.
fn step_bytes(step: u64) -> Vec<u8> {
    use magnitude_kernels::constants::{STEP_HIGH, STEP_LOW, STEP_WORDS};
    let mut words = [0u32; STEP_WORDS as usize];
    words[STEP_LOW as usize] = step as u32;
    words[STEP_HIGH as usize] = (step >> 32) as u32;
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

/// Observe every run, even after a failure, so none is still executing when
/// the caller regains the storage the runs use. The first failure is reported.
fn wait_all(completions: Vec<NativeGraphCompletion>) -> Result<(), SubmitError> {
    completions.into_iter().fold(Ok(()), |outcome, completion| {
        outcome.and(completion.wait().map_err(device))
    })
}

/// Submits a step's runs in sequences of doubling length (1, 2, 4, … runs):
/// the device starts on the first run as soon as it is queued, and each
/// sequence is queued while the device executes the previous ones, so a step
/// of N runs has about log2(N) submission boundaries instead of N. Every
/// run replays one of the graphs its family's slots hold
/// (`seismic::NativeGraphFamily::launch`), so a sequence of any length and
/// mix of places and classes records nothing once each shape has run.
struct StepSubmitter {
    device: Device,
    sequence: NativeGraphSequence,
    queued: usize,
    limit: usize,
    completions: Vec<NativeGraphCompletion>,
    /// The stored-table reads of the runs since the last submission, the
    /// first `queued_reads` of them the queued runs', then of the submitted
    /// runs.
    pending_reads: Vec<StoredRead>,
    queued_reads: usize,
    reads: Vec<StoredRead>,
    /// The landed reads of the runs since the last submission, the first
    /// `queued_landed` of them the queued runs', then of the submitted
    /// runs.
    pending_landed: Vec<SelectedRead>,
    queued_landed: usize,
    landed: Vec<SubmittedRead>,
}

/// A run's read of a stored table: the table and the step it queued there.
#[derive(Clone)]
pub struct StoredRead {
    table: Arc<StoredTable>,
    value: u64,
}

impl StoredRead {
    /// A read of `table`'s step `value`.
    pub fn new(table: Arc<StoredTable>, value: u64) -> Self {
        Self { table, value }
    }
}

impl StepSubmitter {
    fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
            sequence: device.native_sequence(NativeQueue::Compute),
            queued: 0,
            limit: 1,
            completions: Vec::new(),
            pending_reads: Vec::new(),
            queued_reads: 0,
            reads: Vec::new(),
            pending_landed: Vec::new(),
            queued_landed: 0,
            landed: Vec::new(),
        }
    }

    /// The next run queued reads a stored table at its step `value`.
    fn reads(&mut self, table: Arc<StoredTable>, value: u64) {
        self.pending_reads.push(StoredRead { table, value });
    }

    /// The next run queued makes the landed read `read`.
    fn lands(&mut self, read: SelectedRead) {
        self.pending_landed.push(read);
    }

    /// After a failure, once the submitted runs completed: the stored-table
    /// steps and the landed reads (dropped unsubmitted) of runs never
    /// submitted read nothing.
    fn abandon_unsubmitted(self) {
        for read in self.pending_reads {
            let _ = read.table.abandon(read.value);
        }
    }

    /// Queue `ready`, `register` given its outputs before anything queued
    /// is submitted: a run's parks are registered before the device can
    /// reach them (`crate::experts::Parks`).
    fn queue(
        &mut self,
        ready: seismic::ReadyNativeGraphRun<'_>,
        register: impl FnOnce(&NativeGraphOutputs),
    ) -> Result<NativeGraphOutputs, SubmitError> {
        let outputs = ready.queue(&mut self.sequence).map_err(device)?;
        register(&outputs);
        (self.queued_reads, self.queued_landed) = (self.pending_reads.len(), self.pending_landed.len());
        self.queued += 1;
        if self.queued == self.limit {
            self.flush()?;
            self.limit *= 2;
        }
        Ok(outputs)
    }

    /// Queue `ready`, a fill, `register` given its outputs before anything
    /// it is queued with is submitted, once what is queued is submitted, so
    /// no park of the fill holds work queued before it. Where fills run
    /// beside compute it is submitted alone on the fill queue
    /// (`seismic::NativeQueue::Fill`): the device orders it after the
    /// submitted work it must follow by what it touches, and later work
    /// after it likewise, so its moves and parks hold only what reads what
    /// it writes. Where they share the compute queue it begins the next
    /// compute sequence, as any run.
    fn fill(
        &mut self,
        ready: seismic::ReadyNativeGraphRun<'_>,
        register: impl FnOnce(&NativeGraphOutputs),
    ) -> Result<(), SubmitError> {
        self.flush()?;
        if !self.device.backend().fills_beside_compute() {
            return self.queue(ready, register).map(drop);
        }
        let mut sequence = self.device.native_sequence(NativeQueue::Fill);
        let outputs = ready.queue(&mut sequence).map_err(device)?;
        register(&outputs);
        self.completions.push(sequence.submit().map_err(device)?);
        Ok(())
    }

    /// Submit what is queued, with the reads its runs make; those of the
    /// run about to be queued stay pending.
    fn flush(&mut self) -> Result<(), SubmitError> {
        self.queued = 0;
        let sequence = std::mem::replace(&mut self.sequence, self.device.native_sequence(NativeQueue::Compute));
        if !sequence.is_empty() {
            self.completions.push(sequence.submit().map_err(device)?);
            self.reads.extend(self.pending_reads.drain(..self.queued_reads));
            self.landed
                .extend(self.pending_landed.drain(..self.queued_landed).map(SelectedRead::submitted));
        }
        (self.queued_reads, self.queued_landed) = (0, 0);
        Ok(())
    }
}

/// Diagnostic output, chosen once when the program is bound.
#[derive(Clone, Copy)]
struct TargetTrace {
    batch: bool,
    blocks: bool,
}

impl TargetTrace {
    fn from_environment() -> Self {
        Self {
            batch: std::env::var_os("MAGNITUDE_TRACE_TARGET").is_some(),
            blocks: std::env::var_os("MAGNITUDE_TRACE_TARGET_BLOCKS").is_some(),
        }
    }
}

/// A sealed single-range conditioning overlay for one row count, or the
/// boundary capture's copy of one row. The source and the destination
/// hidden rows are late bindings, so a plan serves every placement of that
/// many rows. A step makes its slot with earlier steps in flight, and an
/// overlay's rows are a request's, which no load enumerates, so the plan is
/// launched (`seismic::NativeGraph::seal_launched`): its slot holds no
/// graph, and its one launch runs in its place in the step's submission.
#[derive(Clone)]
struct OverlayGraph {
    plan: NativeGraphPlan,
    source: NativePort,
    destination: NativePort,
}

#[derive(Clone)]
pub struct NativeTargetProgram {
    device: Device,
    state: AttestedState,
    geometry: Decoder,
    graphs: Rc<BoundTargetGraphs>,
    readout_graphs: Rc<BoundTargetReadoutGraphs>,
    /// Overlay plans by row count, sealed on first use.
    overlays: Rc<RefCell<BTreeMap<u64, OverlayGraph>>>,
    boundary_copy: Rc<RefCell<Option<OverlayGraph>>>,
    waiter: CompletionWaiter,
    trace: TargetTrace,
}

/// The launch whose rows a step runs: its advances' state and conditioning.
struct RowState<'a>(&'a TargetLaunchCore);

pub struct TargetReadoutGraphResult {
    pub features: GraphOutputTensor,
    pub logits: Option<GraphOutputTensor>,
    pub selected: Option<GraphOutputTensor>,
    /// Indices into the feature output rows, in projected logits order.
    pub projected_output_rows: Vec<usize>,
}

/// Host instants at which a step's first and last command buffers were
/// committed to the device queue.
#[derive(Clone, Copy, Debug)]
pub struct CommitSpan {
    pub first: Instant,
    pub last: Instant,
}

/// A target step's readout and the commit span of its device work.
pub struct TargetOutput {
    pub boundary: Option<crate::FeatureRef>,
    pub readout: Option<TargetReadoutGraphResult>,
    pub commits: CommitSpan,
    /// The step's landed reads of the table rows the device selected its
    /// tokens for (`DeviceTable::read_selected`); none of a step of the
    /// host's tokens or of a table held whole.
    pub landed_reads: Vec<SubmittedRead>,
    /// The step's reads of stored tables (`StoredTable`).
    pub stored_reads: Vec<StoredRead>,
    /// The step of a streamed load's expert bank, until it completes.
    pub(crate) expert_step: Option<crate::experts::ExpertStep>,
}

/// A target step's output once the step completed (or will never be used,
/// an orphan's): its reads and its expert bank's step completed
/// ([`TargetOutput::complete`]).
pub(crate) struct CompletedTargetOutput {
    pub boundary: Option<crate::FeatureRef>,
    pub readout: Option<TargetReadoutGraphResult>,
    pub commits: CommitSpan,
    /// Per batch slot, why the weights its rows named could not be read:
    /// that slot's outputs are not to be used.
    pub failures: Vec<Option<crate::WeightReadError>>,
    /// How the step's expert bank's step ended.
    pub(crate) experts: ExpertStepEnd,
}

/// How a step's expert bank's step ended.
#[derive(Clone, Debug)]
pub(crate) enum ExpertStepEnd {
    /// It ran; the experts it could not read fail their slots
    /// ([`CompletedTargetOutput::failures`]). The load's miss record, once
    /// a new one is due.
    Ran(Option<std::sync::Arc<crate::assessment::MissRecord>>),
    /// A late beat of the load's pulse interrupted it: the step's outputs
    /// are not to be used, and it is re-run from its accepted state.
    Interrupted(String),
}

impl TargetOutput {
    /// The step of `batch` completed, or will never be used (an orphan's):
    /// its landed reads bring the rows they landed into slots of their own
    /// (`SubmittedRead::complete`), its expert bank's step completes (its
    /// misses `recorded` where the load records them), and its stored
    /// tables forget its failures. Each slot fails where a stored table's
    /// rows it named, or the experts its rows named, could not be read, and
    /// every slot where a landed read could not land its rows (the step then
    /// read other rows). A watcher that panicked serves no step again: the
    /// load fails.
    pub(crate) fn complete(
        mut self,
        batch: &magnitude_batching::ValidatedTargetBatch,
        recorded: Option<crate::experts::Recording>,
    ) -> Result<CompletedTargetOutput, crate::InvariantError> {
        let landing = std::mem::take(&mut self.landed_reads)
            .into_iter()
            .map(SubmittedRead::complete)
            .fold(None, |failure, outcome| {
                failure.or(outcome.err().map(|detail| crate::WeightReadError { detail }))
            });
        let step = self.expert_step.take().map(|step| match recorded {
            Some(recording) => step.recorded(recording),
            None => step,
        });
        let (experts, unread) = match step.map(crate::experts::ExpertStep::complete) {
            None => (ExpertStepEnd::Ran(None), None),
            Some(Ok(record)) => (ExpertStepEnd::Ran(record), None),
            Some(Err(
                crate::experts::StepFailure::Read { rows, reason } | crate::experts::StepFailure::Site { rows, reason },
            )) => (ExpertStepEnd::Ran(None), Some((rows, crate::WeightReadError { detail: reason }))),
            Some(Err(crate::experts::StepFailure::Interrupted(reason))) => (ExpertStepEnd::Interrupted(reason), None),
            Some(Err(failure @ crate::experts::StepFailure::Watcher(_))) => {
                return Err(crate::InvariantError { context: "expert bank", detail: failure.to_string() });
            }
        };
        let failures = (0..batch.actual_slots())
            .map(|slot| {
                let rows = batch
                    .slot(slot)
                    .ok_or_else(|| crate::InvariantError {
                        context: "target output",
                        detail: "a completed target slot has no batch segment".into(),
                    })?
                    .span();
                let table = self.stored_reads.iter().find_map(|read| {
                    read.table
                        .failure(read.value)
                        .filter(|failure| failure.fails(slot))
                        .map(|failure| crate::WeightReadError { detail: failure.error })
                });
                Ok(table
                    .or_else(|| landing.clone())
                    .or_else(|| unread.as_ref().and_then(|(failed, error)| failed.fails(rows).then(|| error.clone()))))
            })
            .collect::<Result<Vec<_>, crate::InvariantError>>()?;
        for read in &self.stored_reads {
            read.table.retire(read.value);
        }
        let Self { boundary, readout, commits, .. } = self;
        Ok(CompletedTargetOutput { boundary, readout, commits, failures, experts })
    }
}

/// Runs of one step queued on the device, with the outputs they will fill.
struct DeviceStep {
    readout: Option<TargetReadoutGraphResult>,
    commits: CommitSpan,
    completions: Vec<NativeGraphCompletion>,
    landed_reads: Vec<SubmittedRead>,
    stored_reads: Vec<StoredRead>,
    expert_step: Option<crate::experts::ExpertStep>,
}

/// One launch of a chain once its readout is queued: its readout, its
/// commit span, and where its command sequences end in the chain's
/// (`StepSubmitter`). A launch's begin where the one before it ends.
struct QueuedRuns {
    readout: Option<TargetReadoutGraphResult>,
    commits: CommitSpan,
    sequences: usize,
}

/// The rows of the launch-row buffers a launch of a chain owns: the draft
/// input rows its blocks write and its readout reads, the per-layer rows and
/// the kept token rows its entry writes and its blocks read. A chain runs
/// each block over every launch before the next block, so its `k`th launch
/// takes region `k` (`PreparedTargetGraphs::region_start`) of its group's
/// buffers (`BoundTargetGraphs::group`), never rows another launch of the
/// chain writes or reads.
#[derive(Clone)]
struct ChainRegion {
    /// The launch's place in its chain, whose words of the expert bank's
    /// placed parts it binds (`BankPart::placed`).
    place: usize,
    /// The region's first row.
    start: u64,
    /// The buffers of the launch's group.
    group: Rc<GroupBuffers>,
}

/// A chain's output slots: each launch's, which its entry writes, and the
/// chain's spare (`TargetProgram::submit`). A graph of a run reads only its
/// run's latest output, so between graphs each run holds one slot and one
/// is free: the next graph writes it, and the slot its run held is the free
/// one once that graph is queued (the device queue orders its rewrite after
/// every read).
struct OutputRotation<'l> {
    slots: Vec<&'l mut TargetGraphOutputLease>,
    free: usize,
}

impl<'l> OutputRotation<'l> {
    /// The rotation of a chain's launches' `slots`, in chain order, and its
    /// `spare`.
    fn new(mut slots: Vec<&'l mut TargetGraphOutputLease>, spare: &'l mut TargetGraphOutputLease) -> Self {
        let free = slots.len();
        slots.push(spare);
        Self { slots, free }
    }

    /// Activate the slot of the launch at `place` for its entry's `plan`.
    fn entry(&mut self, place: usize, plan: &NativeGraphPlan) -> Result<(usize, NativeGraphOutputs), SubmitError> {
        let outputs = self.slots[place].activate(plan).map_err(SubmitError::Invariant)?;
        Ok((place, outputs))
    }

    /// Activate the free slot for `plan`.
    fn activate(&mut self, plan: &NativeGraphPlan) -> Result<(usize, NativeGraphOutputs), SubmitError> {
        let outputs = self.slots[self.free].activate(plan).map_err(SubmitError::Invariant)?;
        Ok((self.free, outputs))
    }

    /// Free slot `at`, whose outputs every reader queued has read.
    fn recycle(&mut self, (at, outputs): (usize, NativeGraphOutputs)) -> Result<(), SubmitError> {
        self.slots[at].recycle(outputs).map_err(SubmitError::Invariant)?;
        self.free = at;
        Ok(())
    }
}

impl ChainRegion {
    /// Whether the launch is alone in its chain, in the floor's one region:
    /// only then may it use the per-layer token-row slots, which lie in
    /// every row of it.
    fn alone(&self) -> bool {
        self.group.places() == 1
    }

    /// The region's leading `rows` of `buffer`.
    fn rows(&self, buffer: &Tensor, rows: u64) -> Result<Tensor, SubmitError> {
        let end = self
            .start
            .checked_add(rows)
            .filter(|&end| end <= buffer.extents()[0])
            .ok_or_else(|| invalid("a chained launch's rows exceed its chain region"))?;
        buffer.slice_leading(self.start, end).map_err(device)
    }
}

/// What a step's run is threaded with from its launch: its batch, its view
/// of the state of the launch's rows, the graph workspace its entry and
/// blocks run in, and its readout's workspace and output.
struct RunLaunch<'l> {
    batch: TargetBatchUpload<'l>,
    state: RowState<'l>,
    graph_workspace: &'l mut TargetGraphWorkspaceLease,
    readout: (&'l mut NativeGraphWorkspaceLease, &'l mut Option<NativeGraphOutputLease>),
}

/// The class of a block a run takes at its rows: its history segment class,
/// the tiles it lists with their list's bytes, and whether it scores.
struct BlockRun {
    rows: u64,
    segments: u64,
    listed: Option<(u64, Vec<u8>)>,
    scored: bool,
}

/// One step's queued work between its blocks: its launch, what the next
/// block reads and the outputs still in use (`begin_run`, `run_block`,
/// `finish_run`).
struct TargetRun<'l> {
    launch: RunLaunch<'l>,
    /// The output slot of the chain's rotation the run's latest graph wrote
    /// (`OutputRotation`), which holds `hidden`.
    output: (usize, NativeGraphOutputs),
    /// The block whose front the run queued last and its class, whose back
    /// its merge set queues next (`run_sets`).
    back: Option<(usize, BlockRun)>,
    /// The scratch-served streamed block whose front the run queued and
    /// whose fill is not yet queued (`run_sets`).
    unfilled: Option<usize>,
    residual: Option<Tensor>,
    hidden: Tensor,
    per_layer_rows: Option<Tensor>,
    region: ChainRegion,
    controls: GraphControls,
    window: RowWindow,
    segments: u64,
    slots: u64,
    /// The entry's selection notice, of a step reading selected tokens in a
    /// model with a stored table.
    notice: Option<Arc<NoticeStep>>,
    /// The recurrent component the next recurrent block binds first.
    recurrent_component: usize,
    /// Blocks whose outputs the step reads; the last one's output feeds the
    /// readout.
    executed: usize,
    first_commit: Instant,
    /// The step of its load's expert bank it runs, of a streamed load.
    bank: Option<BankRun>,
}

impl RowState<'_> {
    fn slots(&self) -> usize {
        self.0.advances().len()
    }
    fn tokens(&self) -> &TargetTokens {
        self.0.tokens()
    }
    fn preceding(&self) -> &[Vec<u64>] {
        self.0.preceding()
    }
    /// The state store's recurrent arenas, one per component. Every slot's
    /// advance belongs to the same store; the batch's bank columns select
    /// the rows each slot reads and publishes.
    fn recurrent_arenas(&self) -> Result<&[Tensor], SubmitError> {
        self.0
            .advances()
            .first()
            .map(|advance| advance.bindings().recurrent)
            .ok_or_else(|| invalid("recurrent batch has no state advance"))
    }
    /// `layer`'s history read (`HistoryReads::read`, the index of its
    /// controls) and planes in the codec's plane-descriptor order, as the
    /// attention graph's state ports take them. A Shared layer binds its
    /// source layer's planes and its source domain's shared read.
    fn history(&self, layer: LayerRef) -> Result<(usize, Vec<&Tensor>), SubmitError> {
        let store = self.0.store();
        let missing = || invalid("attention layer has no history in its store");
        let source = store.history_source(layer).ok_or_else(missing)?;
        let read = store.history_reads().read(layer).ok_or_else(missing)?;
        let planes = self
            .0
            .advances()
            .first()
            .map(|advance| advance.bindings().history)
            .ok_or_else(|| invalid("attention batch has no state advance"))?;
        Ok((
            read,
            planes
                .iter()
                .filter(|plane| plane.domain == source.domain && plane.layer == source.layer)
                .map(|plane| &plane.buffer)
                .collect(),
        ))
    }
    /// Target block `block`'s indexer histories: its raw keys' and its
    /// block keys' reads, each with its planes in descriptor order.
    fn indexer(&self, block: u32) -> Result<[(usize, Vec<&Tensor>); 2], SubmitError> {
        let store = self.0.store();
        let (window, blocks) = store
            .indexer_domains(block)
            .ok_or_else(|| invalid("a selected attention layer has no indexer history"))?;
        let planes = self
            .0
            .advances()
            .first()
            .map(|advance| advance.bindings().history)
            .ok_or_else(|| invalid("attention batch has no state advance"))?;
        let layer = LayerRef::Indexer(block);
        let of = |domain: magnitude_state::HistoryDomainId| {
            (
                domain.0,
                planes
                    .iter()
                    .filter(|plane| plane.domain == domain && plane.layer == layer)
                    .map(|plane| &plane.buffer)
                    .collect(),
            )
        };
        Ok([of(window), of(blocks)])
    }
    fn conditioning(&self, slot: usize) -> Option<&ConditioningRef> {
        self.0.conditioning().get(slot).and_then(Option::as_ref)
    }
    fn conditioning_slices(&self, slot: usize) -> &[ConditioningSlice] {
        self.0
            .conditioning_slices()
            .get(slot)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }
}

impl NativeTargetProgram {
    /// What the graphs of the overlay plans sealed so far cost serving:
    /// plans of no graph family (`seismic::NativeGraphPlan::graph_replays`).
    pub(crate) fn overlay_graph_replays(&self) -> seismic::GraphReplays {
        let overlays = self.overlays.borrow();
        let copy = self.boundary_copy.borrow();
        overlays
            .values()
            .chain(copy.iter())
            .fold(seismic::GraphReplays::default(), |counts, overlay| counts.plus(overlay.plan.graph_replays()))
    }

    /// Visit the graph constants the program binds.
    pub(crate) fn visit_constants(&self, visit: &mut dyn FnMut(&seismic::Tensor)) {
        self.graphs.visit_constants(visit);
    }

    pub(crate) fn constant_bytes(&self) -> Result<u64, &'static str> {
        self.graphs.constant_bytes()
    }

    pub(crate) fn group_bytes(&self, places: usize) -> Result<u64, String> {
        self.graphs.group_bytes(places)
    }

    pub(crate) fn hold_group(&self, places: usize) -> Result<(), crate::resources::AllocationError> {
        self.graphs.hold_group(places)
    }

    pub(crate) fn new(
        device: Device,
        state: AttestedState,
        geometry: Decoder,
        graphs: BoundTargetGraphs,
        readout_graphs: BoundTargetReadoutGraphs,
    ) -> Result<Self, SubmitError> {
        let waiter = CompletionWaiter::spawn()
            .map_err(|error| SubmitError::Device(DeviceError::Execution(error)))?;
        Ok(Self {
            device,
            state,
            geometry,
            graphs: Rc::new(graphs),
            readout_graphs: Rc::new(readout_graphs),
            overlays: Rc::new(RefCell::new(BTreeMap::new())),
            boundary_copy: Rc::default(),
            waiter,
            trace: TargetTrace::from_environment(),
        })
    }

    /// The readout over a run's final hidden rows, from batch row
    /// `window.first` on (`RowWindow`), in its launch's region.
    fn graph_readout(
        &self,
        run: &mut TargetRun<'_>,
        submitter: &mut StepSubmitter,
    ) -> Result<Option<TargetReadoutGraphResult>, SubmitError> {
        let (workspace, output) = &mut run.launch.readout;
        let mut output = output
            .take()
            .ok_or_else(|| invalid("readout output was not reserved before submit"))?;
        let workspace = &mut **workspace;
        let (batch, region, window, hidden, residual) = (
            &run.launch.batch,
            &run.region,
            run.window,
            &run.hidden,
            run.residual.as_ref(),
        );
        let actual_outputs = batch.out_rows.len();
        if actual_outputs == 0 {
            return Ok(None);
        }
        let output_class = magnitude_batching::row_class(actual_outputs)
            .ok_or_else(|| invalid("readout output rows have no class"))?;
        let actual_selected = batch.select_rows.len();
        let selected_class = if actual_selected == 0 {
            0
        } else {
            magnitude_batching::row_class(actual_selected)
                .ok_or_else(|| invalid("readout selected rows have no class"))?
        };
        // Projected outputs, the selected ones first and in selection order:
        // shaping and sampling read the leading selected logits rows.
        let mut projected_output_rows = Vec::new();
        // Whether a selected row also reads its logits on the host.
        let mut selection_reads_logits = false;
        for &output_index in batch.select_rows {
            projected_output_rows.push(
                usize::try_from(output_index)
                    .map_err(|_| invalid("negative selection output row"))?,
            );
        }
        for (output_index, &row) in batch.out_rows.iter().enumerate() {
            let row = usize::try_from(row).map_err(|_| invalid("negative readout row"))?;
            let demand = batch
                .demand
                .get(row)
                .and_then(|bits| Demand::from_bits(*bits))
                .ok_or_else(|| invalid("readout demand is absent or invalid"))?;
            let selected = projected_output_rows[..actual_selected].contains(&output_index);
            selection_reads_logits |= selected && demand.contains(Demand::LOGITS);
            if demand.computes_logits() && !selected {
                projected_output_rows.push(output_index);
            } else if selected && !demand.computes_logits() {
                return Err(invalid("selection row has no projected logits"));
            }
        }
        let actual_projected = projected_output_rows.len();
        if actual_projected > self.readout_graphs.prepared.max_projected_rows() {
            return Err(invalid(format!(
                "readout requests {actual_projected} logits rows; admitted maximum is {}",
                self.readout_graphs.prepared.max_projected_rows()
            )));
        }
        let projected_class = if actual_projected == 0 {
            0
        } else {
            magnitude_batching::row_class(actual_projected)
                .ok_or_else(|| invalid("readout projected rows have no class"))?
        };
        // A selection the certified readout serves: every selected row
        // samples or takes its largest admitted logit at a temperature at
        // most, none reads its logits, and nothing else projects.
        let certified = actual_projected == actual_selected
            && self
                .readout_graphs
                .prepared
                .certifies(projected_class as u64)
            && !selection_reads_logits
            && batch.shaping[..actual_selected].iter().all(certifies);
        let kind = if actual_selected > 0 {
            ReadoutKind::Selection {
                shaped: batch.shaping[..actual_selected].iter().any(shapes),
                certified,
            }
        } else if actual_projected > 0 {
            ReadoutKind::Logits
        } else {
            ReadoutKind::Features
        };
        let class = ReadoutClass {
            rows: window.rows,
            outputs: output_class as u64,
            projected: projected_class as u64,
            selected: selected_class as u64,
            kind,
        };
        let (graph, bound) = self.readout_graphs.class(class).map_err(invalid)?;
        let mut active = workspace.slot_mut().activate(&graph.plan).map_err(device)?;
        let tap_rows = graph
            .taps
            .as_ref()
            .map(|_| self.tap_rows(region, class.rows))
            .transpose()?;
        let mut bindings = bound.bindings();
        if let Some(port) = &graph.residual {
            bindings.set(port, residual.ok_or_else(|| invalid("residual conditioning was not exported"))?).map_err(device)?;
        }
        if let Some(rows) = &graph.final_rows {
            bindings.set(&rows.hidden, hidden).map_err(device)?;
        }
        if let (Some(ports), Some(tap_rows)) = (&graph.taps, &tap_rows) {
            bindings.set(&ports.taps, tap_rows).map_err(device)?;
        }
        let mut out_rows = vec![0_i32; output_class];
        for (index, &row) in batch.out_rows.iter().enumerate() {
            out_rows[index] = window.row(row)?;
        }
        active
            .write_input(&graph.out_rows, &i32_bytes(&out_rows))
            .map_err(device)?;
        if let Some(port) = &graph.logit_rows {
            // The head gathers the hidden rows of the projected outputs.
            let mut rows = vec![0_i32; projected_class];
            for (index, &output_index) in projected_output_rows.iter().enumerate() {
                rows[index] = window.row(
                    *batch
                        .out_rows
                        .get(output_index)
                        .ok_or_else(|| invalid("projected output row exceeds output rows"))?,
                )?;
            }
            active
                .write_input(port, &i32_bytes(&rows))
                .map_err(device)?;
        }
        if let Some(ports) = &graph.selection {
            write_selection(batch, &mut active, ports, selected_class, batch.mask_words)?;
        }
        let ready = active
            .attach(
                bindings,
                output
                    .activate(&graph.plan)
                    .map_err(SubmitError::Invariant)?,
            )
            .map_err(device)?;
        let outputs = submitter.queue(ready, |_| {})?;
        let owner = output.publish(outputs);
        let features = owner
            .tensor(&graph.features)
            .ok_or_else(|| invalid("readout features were not exported"))?;
        let logits = graph
            .logits
            .as_ref()
            .map(|edge| {
                owner
                    .tensor(edge)
                    .ok_or_else(|| invalid("readout logits were not exported"))
            })
            .transpose()?;
        let selected = graph
            .selected
            .as_ref()
            .map(|edge| {
                owner
                    .tensor(edge)
                    .ok_or_else(|| invalid("readout selection was not exported"))
            })
            .transpose()?;
        Ok(Some(TargetReadoutGraphResult {
            features,
            logits,
            selected,
            projected_output_rows,
        }))
    }

    /// The leading `rows` draft input rows of a tapped target in `region`.
    fn tap_rows(&self, region: &ChainRegion, rows: u64) -> Result<Tensor, SubmitError> {
        region.rows(
            region
                .group
                .taps
                .as_ref()
                .ok_or_else(|| invalid("a tapped graph has no bound draft input rows"))?,
            rows,
        )
    }

    fn boundary_copy(&self) -> Result<OverlayGraph, SubmitError> {
        if let Some(graph) = self.boundary_copy.borrow().as_ref() {
            return Ok(graph.clone());
        }
        let (element, width) = self.readout_graphs.prepared.residual_width.map_or((Element::dense(crate::planning::activation_dtype(self.geometry.activation_dtype)), self.geometry.hidden), |width| (Element::f32(), width));
        let mut graph = self.device.native_graph();
        let source = graph
            .port(element, &[1, width])
            .map_err(device)?;
        let mut destination = graph
            .port(element, &[1, width])
            .map_err(device)?;
        graph
            .enqueue(
                &self.state.feature_copy,
                feature_copy::WorkflowArgs {
                    input: source.tensor().into(),
                    out: destination.tensor_mut().into(),
                },
            )
            .map_err(device)?;
        let plan = graph.seal_launched().map_err(device)?;
        if plan.workspace_bytes() != 0 || plan.output_bytes() != 0 || plan.upload_bytes() != 0 {
            return Err(invalid(
                "boundary capture unexpectedly requires graph storage",
            ));
        }
        let graph = OverlayGraph {
            plan,
            source,
            destination,
        };
        *self.boundary_copy.borrow_mut() = Some(graph.clone());
        Ok(graph)
    }

    fn overlay(&self, rows: u64) -> Result<OverlayGraph, SubmitError> {
        if let Some(overlay) = self.overlays.borrow().get(&rows) {
            return Ok(overlay.clone());
        }
        let kernel = self
            .state
            .conditioning
            .as_ref()
            .ok_or_else(|| invalid("conditioning program slot is absent"))?;
        let mut graph = self.device.native_graph();
        let source = graph
            .port(Element::f32(), &[rows, self.geometry.hidden])
            .map_err(device)?;
        let mut destination = graph
            .port(Element::f32(), &[rows, self.geometry.hidden])
            .map_err(device)?;
        graph
            .enqueue(
                kernel,
                conditioning_overlay::WorkflowArgs {
                    input: source.tensor().into(),
                    out: destination.tensor_mut().into(),
                },
            )
            .map_err(device)?;
        let plan = graph.seal_launched().map_err(device)?;
        if plan.workspace_bytes() != 0 || plan.output_bytes() != 0 || plan.upload_bytes() != 0 {
            return Err(invalid(
                "conditioning overlay unexpectedly requires graph storage",
            ));
        }
        let overlay = OverlayGraph {
            plan,
            source,
            destination,
        };
        self.overlays.borrow_mut().insert(rows, overlay.clone());
        Ok(overlay)
    }

    /// Queue a per-layer model's per-layer entry over a beginning run's
    /// embedded and overlaid hidden rows, and return the per-layer rows its
    /// blocks read, in the launch's region. Each batch row gathers its
    /// token's table row, a row of `media_rows` the entry's media row.
    fn queue_per_layer_entry(
        &self,
        run: &mut TargetRun<'_>,
        media_rows: &[std::ops::Range<usize>],
        submitter: &mut StepSubmitter,
    ) -> Result<Option<Tensor>, SubmitError> {
        let (batch, region, rows, hidden) = (&run.launch.batch, &run.region, run.window.rows, &run.hidden);
        let Some(entry) = self.graphs.per_layer_entry(rows) else {
            return Ok(None);
        };
        let (graph, bound, per_layer) = entry.map_err(invalid)?;
        let TargetTokens::Host = run.launch.state.tokens() else {
            return Err(invalid(
                "a per-layer entry gathers table rows by host tokens",
            ));
        };
        let media_row = self
            .geometry
            .entry
            .per_layer
            .as_ref()
            .ok_or_else(|| invalid("a per-layer entry graph without its geometry"))?
            .media_row;
        let media_row =
            i64::try_from(media_row).map_err(|_| invalid("per-layer media row exceeds i64"))?;
        let table_rows = batch
            .tokens
            .iter()
            .enumerate()
            .map(|(row, token)| {
                if media_rows.iter().any(|range| range.contains(&row)) {
                    media_row
                } else {
                    i64::from(*token)
                }
            })
            .collect::<Vec<_>>();
        let per_layer_rows = region
            .group
            .per_layer_rows
            .as_ref()
            .ok_or_else(|| invalid("a per-layer entry without its per-layer rows"))?;
        let mut inputs = region.rows(per_layer_rows, rows)?;
        // The token whose per-layer row this step writes into a token
        // row's slot, and the slot.
        let mut written = None;
        // The token rows' slots are the floor's per-layer rows after the
        // first, the last row the first slot, so only a launch alone in its
        // chain reads or writes one. Its entry writes the leading rows: alone,
        // its region is the floor's, from row 0, so its rows after the first
        // overwrite the slots of the last `rows - 1` rows but the last.
        if region.alone() {
            let last = per_layer_rows.extents()[0] - 1;
            per_layer
                .token_rows
                .overwrite((last + 1 - rows) as usize..last as usize);
            if let ([token], true) = (batch.tokens, media_rows.is_empty()) {
                let slot = |slot: usize| {
                    let row = last - slot as u64;
                    per_layer_rows.slice_leading(row, row + 1).map_err(device)
                };
                let token = u64::try_from(*token)
                    .map_err(|_| invalid(format!("token {token} has no per-layer row")))?;
                match per_layer.token_rows.find(token).map_err(invalid)? {
                    TokenRow::Held(held) => return Ok(Some(slot(held)?)),
                    TokenRow::Free(free) => {
                        inputs = slot(free)?;
                        written = Some((token, free));
                    }
                    TokenRow::None => {}
                }
            }
        }
        let mut active = run
            .launch
            .graph_workspace
            .slot_mut()
            .activate(&graph.plan)
            .map_err(device)?;
        per_layer
            .table
            .gather(&table_rows, |gathered| {
                active
                    .write_input(&graph.ports.gathered, gathered)
                    .map_err(device)
            })
            .map_err(|error: HostTableError| invalid(error.to_string()))??;
        let mut bindings = bound.bindings();
        bindings.set(&graph.ports.hidden, hidden).map_err(device)?;
        bindings.set(&graph.ports.rows, &inputs).map_err(device)?;
        let ready = active
            .attach(bindings, graph.plan.new_outputs().map_err(device)?)
            .map_err(device)?;
        submitter.queue(ready, |_| {})?;
        if let Some((token, slot)) = written {
            per_layer.token_rows.hold(token, slot);
        }
        Ok(Some(inputs))
    }

    /// The rows block `index`'s stored table is read at in a step: its row
    /// rule over each slot's tokens and the tokens its sequence holds before
    /// them (`NGramHash::rows`), the rule the device evaluates
    /// (`ngram_rows`). Of host tokens they are known now; of selected
    /// tokens, the table's filler learns them from the step's notice.
    fn stored_step_rows(
        &self,
        index: usize,
        batch: &TargetBatchUpload<'_>,
        state: &RowState<'_>,
        notice: Option<&Arc<NoticeStep>>,
    ) -> Result<StepRows, SubmitError> {
        let hash = crate::operators::row_rules(&self.geometry)
            .into_iter()
            .find_map(|(block, hash)| (block == index).then(|| hash.clone()))
            .ok_or_else(|| invalid("a stored table's block has no row rule"))?;
        let preceding = state.preceding();
        if preceding.len() != state.slots() {
            return Err(invalid("the step's preceding tokens differ from its slots"));
        }
        match state.tokens() {
            TargetTokens::Host => {
                let mut rows = Vec::new();
                for (slot, preceding) in preceding.iter().enumerate() {
                    let [start, end] = batch.segments[slot];
                    let tokens = usize::try_from(start)
                        .ok()
                        .zip(usize::try_from(end).ok())
                        .and_then(|(start, end)| batch.tokens.get(start..end))
                        .ok_or_else(|| invalid("a slot's segment lies outside the batch"))?
                        .iter()
                        .map(|token| u64::try_from(*token))
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|_| invalid("a host token is negative"))?;
                    rows.push(hash.rows(preceding, &tokens));
                }
                Ok(StepRows::Known(rows))
            }
            TargetTokens::Selected(_) => {
                let notice = notice
                    .cloned()
                    .ok_or_else(|| invalid("selected tokens without their notice"))?;
                let preceding = preceding.to_vec();
                Ok(StepRows::Selected(Box::new(move || {
                    // A slot's one row is its selection row; a failed
                    // selection's slot names none, as its step is never
                    // used.
                    let selected = notice.read()?;
                    Ok(preceding
                        .iter()
                        .zip(selected)
                        .map(|(preceding, [token, status])| match (u64::try_from(token), status) {
                            (Ok(token), 0) => hash.rows(preceding, &[token]),
                            _ => Vec::new(),
                        })
                        .collect())
                })))
            }
        }
    }

    /// The leading `rows` in `region` of the token rows a model whose
    /// blocks read the step's token rows keeps
    /// (`GroupBuffers::token_rows`).
    fn kept_token_rows(&self, region: &ChainRegion, rows: u64) -> Result<Tensor, SubmitError> {
        region.rows(
            region
                .group
                .token_rows
                .as_ref()
                .ok_or_else(|| invalid("the model keeps no token rows"))?,
            rows,
        )
    }

    /// Conditioning rows that replace embedding rows, as (source, first
    /// batch row, end batch row). Most steps have none.
    fn conditioning_sources(
        &self,
        state: &RowState<'_>,
        batch: &TargetBatchUpload<'_>,
    ) -> Result<Vec<(Tensor, usize, usize)>, SubmitError> {
        let mut sources = Vec::new();
        for slot in 0..state.slots() {
            let conditioning = state.conditioning(slot);
            let slices = state.conditioning_slices(slot);
            if conditioning.is_none() && slices.is_empty() {
                continue;
            }
            let [start, end] = batch.segments[slot];
            let start = usize::try_from(start)
                .map_err(|_| invalid("conditioning segment start is negative"))?;
            let end = usize::try_from(end)
                .map_err(|_| invalid("conditioning segment end is negative"))?;
            if let Some(conditioning) = conditioning {
                if end - start != conditioning.allocation().rows() {
                    return Err(invalid("conditioning rows differ from validated segment"));
                }
                for range in conditioning.allocation().ranges() {
                    let source = conditioning
                        .allocation()
                        .tensor()
                        .map_err(device)?
                        .slice_leading(range.destination.start as u64, range.destination.end as u64)
                        .map_err(device)?;
                    sources.push((
                        source,
                        start + range.destination.start,
                        start + range.destination.end,
                    ));
                }
            }
            for slice in slices {
                let source_end = slice
                    .source
                    .start
                    .checked_add(slice.source.count)
                    .ok_or_else(|| invalid("conditioning source span overflows"))?;
                let destination_start = start
                    .checked_add(slice.destination)
                    .ok_or_else(|| invalid("conditioning destination overflows"))?;
                let destination_end = destination_start
                    .checked_add(slice.source.count)
                    .ok_or_else(|| invalid("conditioning destination span overflows"))?;
                if destination_end > end {
                    return Err(invalid("conditioning slice exceeds validated slot"));
                }
                let source = slice
                    .source
                    .features
                    .allocation()
                    .tensor()
                    .map_err(device)?
                    .slice_leading(slice.source.start as u64, source_end as u64)
                    .map_err(device)?;
                sources.push((source, destination_start, destination_end));
            }
        }
        Ok(sources)
    }

    /// Queue every run of a chain (`TargetProgram::submit`), one step per
    /// launch: each launch with its queued step. On failure, runs already
    /// queued are observed before the caller regains the storage they use,
    /// and the expert steps complete: the miss record their completion made
    /// due goes with the failure, and the launches with it.
    fn queue_chain(
        &self,
        chain: Chain<ValidatedTargetLaunch>,
        spare: &mut TargetGraphOutputLease,
    ) -> Result<Chain<(ValidatedTargetLaunch, DeviceStep)>, super::ChainRefusal> {
        let refused = |error, chain| super::ChainRefusal { error, chain, misses: None };
        // A streamed load's steps, in chain order, one group: the watcher
        // serves each from before its first run is queued until it
        // completes. Scratch-served where a prompt group is not resident.
        let route = |launch: &ValidatedTargetLaunch| {
            match crate::operators::routed::fused_graph::decodes(launch.batch().upload().class.rows() as u64) {
                true => StepRoute::Decode,
                false => StepRoute::Prompt,
            }
        };
        let prompt = chain.iter().any(|launch| route(launch) == StepRoute::Prompt);
        // Per block, the steps of the merge set its group ends with
        // (`Shared::begin`): the whole group but at a streamed layer of a
        // group that is not resident (`ExpertBank::resident`).
        let mut sets = vec![chain.len(); self.geometry.blocks.len()];
        let mut scratch = false;
        // Each launch with its expert step, of a streamed load: a decode
        // launch's first row is at its position (validated nonnegative).
        let steps = match &self.graphs.bank {
            Some(bank) => {
                let begun = bank.begin(chain, |launch| route(launch), || {
                    let resident = bank.resident()?;
                    scratch = prompt && !resident;
                    Ok(scratch)
                });
                let (steps, merges) = begun.map_err(|(error, chain)| refused(invalid(error), chain))?;
                let steps = steps.map(|(launch, step)| {
                    let upload = launch.batch().upload();
                    let step = match route(&launch) {
                        StepRoute::Decode => step.decoding(upload.actual_rows, upload.coordinates[0][0] as u64),
                        StepRoute::Prompt => step,
                    };
                    (launch, Some((bank.clone(), step)))
                });
                // A streamed layer is a block's routed sublayer or a parallel
                // sublayer's routed branch.
                for (scope, size) in merges {
                    use magnitude_family_contracts::WeightScope;
                    let (WeightScope::TargetSublayer(at) | WeightScope::TargetBranch { sublayer: at, .. }) = scope else {
                        let error = invalid(format!("an expert bank layer of {scope:?} outside the target's blocks"));
                        return Err(refused(error, steps.map(|(launch, _)| launch)));
                    };
                    sets[at.block as usize] = size;
                }
                steps
            }
            None => chain.map(|launch| (launch, None)),
        };
        let (mut submitter, queued) = steps.scoped(
            StepSubmitter::new(&self.device),
            |steps, submitter| self.queue_runs(steps, spare, &sets, scratch, submitter),
        );
        match queued {
            Ok(runs) => {
                // Each launch takes its command sequences, from where the
                // launch before it ended.
                let mut completions = submitter.completions.into_iter();
                let mut taken = 0;
                let mut steps = runs.map(|((launch, banked), runs)| {
                    let step = DeviceStep {
                        readout: runs.readout,
                        commits: runs.commits,
                        completions: completions.by_ref().take(runs.sequences - taken).collect(),
                        stored_reads: Vec::new(),
                        landed_reads: Vec::new(),
                        expert_step: banked.map(|(_, step)| step),
                    };
                    taken = runs.sequences;
                    (launch, step)
                });
                // Every launch's reads are queued before the first ends, and
                // a chain of several launches is one request that commits as
                // one: its last step, which completes once every read's gate
                // opened, takes them all.
                let (_, last) = steps.last_mut();
                last.stored_reads = submitter.reads;
                last.landed_reads = submitter.landed;
                Ok(steps)
            }
            Err((error, steps)) => {
                // The failure is reported; outcomes of runs submitted before
                // it matter only for draining the device. Runs still queued
                // are dropped unsubmitted, and their stored-table steps
                // read nothing.
                let _ = wait_all(std::mem::take(&mut submitter.completions));
                submitter.abandon_unsubmitted();
                // The device is past the chain's runs: each step is refused,
                // none fed, as no run of the chain's is known to have written
                // its records. A watcher that panicked fails the load. A miss
                // record due now goes with the submission that failed.
                let (mut watcher, mut misses) = (None, None);
                let (chain, expert_steps) = steps.unzip();
                for outcome in expert_steps.into_iter().flatten().map(|(_, step)| step.refuse()) {
                    match outcome {
                        Ok(record) => misses = record.or(misses),
                        Err(failure) => watcher = watcher.or(Some(failure)),
                    }
                }
                let error = match watcher {
                    Some(failure) => invalid(format!("{error}; {failure}")),
                    None => error,
                };
                Err(super::ChainRefusal { error, chain, misses })
            }
        }
    }

    /// Queue a chain's runs: every launch's entry, then each block over
    /// every launch in chain order, then every launch's readout, each
    /// ending its own command sequences. Each launch is exactly the step it
    /// would be alone, in its own region of the launch-row buffers
    /// (`ChainRegion`); every run follows the runs it reads, as launch
    /// `k + 1` reads launch `k`'s history rows and successor bank at the
    /// block it runs.
    fn queue_runs<'b>(
        &self,
        steps: Branded<'b, &'b mut (ValidatedTargetLaunch, Option<(Arc<crate::experts::ExpertBank>, crate::experts::ExpertStep)>)>,
        spare: &mut TargetGraphOutputLease,
        sets: &[usize],
        scratch: bool,
        submitter: &mut StepSubmitter,
    ) -> Result<Branded<'b, QueuedRuns>, SubmitError> {
        let group = self.graphs.group(steps.len()).map_err(invalid)?;
        // Every step of the group holds its group's record.
        let reads = steps
            .iter()
            .next()
            .and_then(|(_, banked)| banked.as_ref())
            .and_then(|(bank, step)| Some((bank.clone(), step.group_reads()?)));
        let reads = reads.as_ref().map(|(bank, record)| (&**bank, *record));
        let (launches, outputs) = steps
            .map(|(launch, banked)| {
                let bank = banked.as_ref().map(|(bank, step)| BankRun { bank: bank.clone(), sequence: step.sequence() });
                let (core, graph_workspace, graph_output, workspace, output) = launch.submission_parts_mut();
                ((core, graph_workspace, (workspace, output), bank), graph_output)
            })
            .unzip();
        let (first, rest) = outputs.into_chain().split_first();
        let mut rotation =
            OutputRotation::new(std::iter::once(first).chain(rest).collect(), spare);
        let mut place = 0;
        let mut runs = launches.try_map(
            |(core, graph_workspace, readout, bank)| {
                // A later launch's entry runs before the chain's readouts,
                // so it cannot read a selection the chain makes.
                if place > 0 && matches!(core.tokens(), TargetTokens::Selected(_)) {
                    return Err(invalid("a chained launch reads selected tokens"));
                }
                let region = ChainRegion {
                    place,
                    start: self.graphs.prepared.region_start(place),
                    group: group.clone(),
                };
                place += 1;
                let launch = RunLaunch {
                    batch: core.batch().upload(),
                    state: RowState(core),
                    graph_workspace,
                    readout,
                };
                self.begin_run(launch, region, bank, &mut rotation, submitter)
            },
        )?;
        // A wavefront over merge sets: at each block, the backs of the block
        // before by its merge sets, in chain order, each set's runs then
        // taking the block; so a step's next front follows its own back and
        // precedes the next set's backs, every front of a block precedes its
        // first set's pass, and its later sets' passes overlap the next
        // block's reads its earlier fronts requested.
        for index in 0..=self.geometry.blocks.len() {
            self.run_sets(&mut runs, index, sets, scratch, reads, &mut rotation, submitter)?;
        }
        runs.try_map(|run| {
            let (readout, commits) = self.finish_run(run, &mut rotation, submitter)?;
            submitter.flush()?;
            Ok(QueuedRuns {
                readout,
                commits,
                sequences: submitter.completions.len(),
            })
        })
    }

    /// Queue a step's entry: its embedding, conditioning overlays and
    /// per-layer inputs. The returned run then takes its blocks in order
    /// (`run_block`) and its readout (`finish_run`).
    fn begin_run<'l>(
        &self,
        launch: RunLaunch<'l>,
        region: ChainRegion,
        bank: Option<BankRun>,
        rotation: &mut OutputRotation<'_>,
        submitter: &mut StepSubmitter,
    ) -> Result<TargetRun<'l>, SubmitError> {
        let RunLaunch {
            batch: upload,
            state: rows_state,
            graph_workspace,
            readout,
        } = launch;
        let (batch, state) = (&upload, &rows_state);
        let rows =
            u64::try_from(batch.class.rows()).map_err(|_| invalid("row class exceeds u64"))?;
        let tokens = state.tokens();
        let source = match tokens {
            TargetTokens::Host => EntryTokens::Uploaded,
            TargetTokens::Selected(_) => EntryTokens::Selected,
        };
        let (entry, entry_bound) = self.graphs.entry(rows, source).map_err(invalid)?;
        // A step reading selected tokens in a model with a stored table
        // copies them for the host first (`EntryNotice`).
        let mut notice = None;
        let entry_outputs = {
            let mut active = graph_workspace
                .slot_mut()
                .activate(&entry.plan)
                .map_err(device)?;
            let mut bindings = entry_bound.bindings();
            match tokens {
                TargetTokens::Host => {
                    let rows = self
                        .graphs
                        .embedding
                        .indices(batch.tokens)
                        .map_err(invalid)?;
                    active
                        .write_input(&entry.tokens, &token_rows(&rows))
                        .map_err(device)?;
                    // A model whose blocks read the step's token rows keeps
                    // the host's tokens in the bound token rows.
                    if let Some(kept) = &entry.kept_tokens {
                        active
                            .write_input(&kept.uploaded, &token_rows(batch.tokens))
                            .map_err(device)?;
                        bindings
                            .set(
                                &kept.rows,
                                &self.kept_token_rows(&region, batch.tokens.len() as u64)?,
                            )
                            .map_err(device)?;
                    }
                }
                TargetTokens::Selected(selected) => bindings
                    .set(&entry.tokens, selected.tensor())
                    .map_err(device)?,
            }
            // Selected tokens are copied for the host, which learns from
            // them the stored tables' rows the step names.
            if let Some(ports) = &entry.notice {
                let step = self
                    .graphs
                    .notice
                    .as_ref()
                    .ok_or_else(|| invalid("an entry notice without a selection notice"))?
                    .next(rows)
                    .map_err(invalid)?;
                bindings.set(&ports.tokens, &step.tokens).map_err(device)?;
                bindings
                    .set_timeline(ports.signal, &step.timeline, step.value)
                    .map_err(device)?;
                notice = Some(Arc::new(step));
            }
            // Selected tokens of a table with landing slots: a landed read.
            if let Some(ports) = &entry.landed {
                let read = ports
                    .queue(&self.graphs.embedding, &mut bindings)
                    .map_err(invalid)?;
                submitter.lands(read);
            }
            let (at, outputs) = rotation.entry(region.place, &entry.plan)?;
            let ready = active.attach(bindings, outputs).map_err(device)?;
            (at, submitter.queue(ready, |_| {})?)
        };
        // The device embeds while the host prepares the remaining runs.
        let first_commit = Instant::now();
        let segments = u64::try_from(batch.class.segments())
            .map_err(|_| invalid("segment class exceeds u64"))?;
        let slots = u64::try_from(batch.actual_slots)
            .map_err(|_| invalid("request slot class exceeds u64"))?;
        let controls = GraphControls::new(
            batch,
            self.graphs.prepared.controls(),
            graph_workspace.slot_mut(),
        )?;
        let hidden = entry_outputs
            .1
            .exported(&entry.hidden)
            .ok_or_else(|| invalid("target embedding hidden was not exported"))?;
        let conditioning = self.conditioning_sources(state, batch)?;
        let media_rows = conditioning
            .iter()
            .map(|(_, start, end)| *start..*end)
            .collect::<Vec<_>>();
        for (source, start, end) in conditioning {
            let overlay = self.overlay((end - start) as u64)?;
            let destination = hidden
                .slice_leading(start as u64, end as u64)
                .map_err(device)?;
            let mut bindings = overlay.plan.bindings();
            bindings.set(&overlay.source, &source).map_err(device)?;
            bindings
                .set(&overlay.destination, &destination)
                .map_err(device)?;
            let mut slot = overlay.plan.new_slot().map_err(device)?;
            let ready = slot
                .attach(bindings, overlay.plan.new_outputs().map_err(device)?)
                .map_err(device)?;
            submitter.queue(ready, |_| {})?;
        }
        if self.trace.batch {
            eprintln!(
                "target batch actual_rows={} class_rows={} actual_slots={} outputs={} selections={}",
                batch.actual_rows,
                batch.class.rows(),
                batch.actual_slots,
                batch.out_rows.len(),
                batch.select_rows.len()
            );
            for (domain, history) in controls.histories.iter().enumerate() {
                let spans = history
                    .visible
                    .chunks_exact(8)
                    .map(|span| {
                        let bound = |at: usize| {
                            i32::from_le_bytes([span[at], span[at + 1], span[at + 2], span[at + 3]])
                        };
                        (bound(0), bound(4))
                    })
                    .collect::<Vec<_>>();
                // Every row of a decode launch; a prefill launch's first
                // and last.
                let per_row = spans.chunks((segments as usize).max(1)).collect::<Vec<_>>();
                let shown = match per_row.len() {
                    0..=16 => per_row,
                    rows => vec![per_row[0], per_row[rows - 1]],
                };
                eprintln!(
                    "target batch history domain={domain} segments={segments} spans={shown:?}"
                );
            }
        }
        let mut run = TargetRun {
            launch: RunLaunch {
                batch: upload,
                state: rows_state,
                graph_workspace,
                readout,
            },
            output: entry_outputs,
            back: None,
            unfilled: None,
            residual: None,
            hidden,
            per_layer_rows: None,
            region,
            controls,
            window: RowWindow::whole(rows),
            segments,
            slots,
            notice,
            recurrent_component: 0,
            executed: self.geometry.blocks.len(),
            first_commit,
            bank,
        };
        run.per_layer_rows = self.queue_per_layer_entry(&mut run, &media_rows, submitter)?;
        Ok(run)
    }

    /// Queue block `index` of a step's run: the whole block, or a split
    /// block's front, whose back `run_sets` queues once the chain's later
    /// runs queued their fronts there. Blocks are taken in
    /// order; a run whose stateless tail reads no rows stops before it.
    fn run_block(
        &self,
        run: &mut TargetRun<'_>,
        index: usize,
        rotation: &mut OutputRotation<'_>,
        submitter: &mut StepSubmitter,
    ) -> Result<(), SubmitError> {
        if index >= run.executed {
            return Ok(());
        }
        let (batch, state) = (&run.launch.batch, &run.launch.state);
        let segments = run.segments;
        if index == self.graphs.prepared.stateless_tail() {
            // From here no block publishes state: run the rest over the
            // rows the readout consumes, or not at all without any.
            let rows = run.window.rows;
            match RowWindow::consumed(batch.out_rows, rows)? {
                None => {
                    run.executed = index;
                    return Ok(());
                }
                Some(tail) if tail.rows < rows => {
                    // The streamed layers from here name their requests'
                    // rows within the window.
                    if let Some(BankRun { bank, sequence }) = &run.bank {
                        bank.window(*sequence, index, run.window.first + tail.first).map_err(invalid)?;
                    }
                    run.hidden = tail.slice(&run.hidden)?;
                    run.per_layer_rows = run
                        .per_layer_rows
                        .as_ref()
                        .map(|inputs| tail.slice(inputs))
                        .transpose()?;
                    run.controls.window(rows, tail)?;
                    run.window = tail;
                }
                Some(_) => {}
            }
        }
        let controls = &run.controls;
        let rows = run.window.rows;
        // A block with listing classes lists the history row tiles the
        // launch's rows see in the smallest class that holds them; a
        // launch that sees more than the largest (many requests with long
        // histories) takes the class that lists none.
        let listed = match self.graphs.prepared.listed_tiles(index, rows) {
            [] => None,
            classes => {
                let (read, _) = state.history(LayerRef::Target(index as u32))?;
                let visible = &controls
                    .histories
                    .get(read)
                    .ok_or_else(|| invalid("the batch lacks a listing block's history read"))?
                    .visible;
                classes.iter().find_map(|&tiles| {
                    crate::operators::attention::graph::history_tile_bytes(
                        visible,
                        usize::try_from(tiles).expect("a tile count fits in usize"),
                    )
                    .map(|bytes| (tiles, bytes))
                })
            }
        };
        // A listing class may be sealed for more segments than the launch
        // has: its rows' spans are then padded with empty ones.
        let block_segments = match &listed {
            Some(_) => self.graphs.prepared.listed_segments(index, segments),
            None => segments,
        };
        let scored = match self.graphs.prepared.selection(index) {
            None => false,
            Some(selection) => {
                let [_, (block_read, _)] = state.indexer(index as u32)?;
                let blocks = controls
                    .histories
                    .get(block_read)
                    .ok_or_else(|| invalid("the batch lacks an indexer's history read"))?;
                selection_scores(blocks, segments, selection.blocks)
            }
        };
        let class = BlockRun {
            rows,
            segments: block_segments,
            listed,
            scored,
        };
        let stages = self.graphs.prepared.stages(index, rows);
        self.run_stage(run, index, &class, stages[0], rotation, submitter)?;
        run.back = (stages.len() > 1).then_some((index, class));
        Ok(())
    }

    /// Queue the backs the runs' fronts at the block before `index` left
    /// (`run_block`), by merge set, each set's runs then taking block
    /// `index` (none past the last). The group's runs end with a set of
    /// `sets[block]` runs and begin with single ones. A set's backs follow
    /// its pass (`run_pass`); its group's `reads` record takes the pass's
    /// reads.
    ///
    /// In a `scratch`-served group at a streamed layer, the pass reads the
    /// scratch tier its runs' fills (`run_fill`) wrote, which every
    /// streamed layer shares, so a fill follows its run's front, the layer
    /// before's last pass, and the passes of its own layer's merge sets
    /// before its own (whose steps' records its seeding keeps: the
    /// watcher's `Scratch` seeding), and precedes its set's pass. Where
    /// fills run beside compute (`seismic::BackendName::fills_beside_compute`)
    /// each is queued as soon as those allow, so it moves beside the backs
    /// and fronts after it, which are submitted ahead of any later pass;
    /// where they share the compute queue, right
    /// before its set's pass, which leaves its request the most host time.
    /// The device orders each by what it touches.
    #[allow(clippy::too_many_arguments)]
    fn run_sets(
        &self,
        runs: &mut Branded<'_, TargetRun<'_>>,
        index: usize,
        sets: &[usize],
        scratch: bool,
        reads: Option<(&crate::experts::ExpertBank, usize)>,
        rotation: &mut OutputRotation<'_>,
        submitter: &mut StepSubmitter,
    ) -> Result<(), SubmitError> {
        let count = runs.len();
        let block = runs.iter_mut().find_map(|run| run.back.as_ref().map(|&(at, _)| at));
        let merged = block.map_or(1, |at| sets[at].clamp(1, count));
        // A pass covers at most the runs whose rows its graph holds
        // (`group_pass_rows`): a longer merge set is several passes.
        let most = self.graphs.passes().launches();
        let filled = |at: usize| scratch && self.graphs.prepared.streams(at);
        let early = self.device.backend().fills_beside_compute();
        // The runs of block `index`'s first merge set, whose fills follow
        // no pass of their own layer: every run where the layer merges
        // whole, else its first, a set of one.
        let first = |place: usize| index < self.geometry.blocks.len() && (sets[index].clamp(1, count) == count || place == 0);
        let mut start = 0;
        while start < count {
            let set = start..if start + merged < count { start + 1 } else { count.min(start + most) };
            if let Some(at) = block {
                for (_, run) in runs.iter_mut().enumerate().filter(|(place, run)| set.contains(place) && run.unfilled == Some(at)) {
                    self.run_fill(run, at, submitter)?;
                }
                self.run_pass(runs, at, set.clone(), reads, submitter)?;
            }
            // No pass of the block before follows in this call: its last
            // has been queued.
            let passed = early && (block.is_none() || set.end == count);
            if passed {
                for (_, run) in runs.iter_mut().enumerate().filter(|(place, run)| first(*place) && run.unfilled == Some(index)) {
                    self.run_fill(run, index, submitter)?;
                }
            }
            for (place, run) in runs.iter_mut().enumerate().filter(|(place, _)| set.contains(place)) {
                if let Some((at, class)) = run.back.take() {
                    self.run_stage(run, at, &class, RunStage::Combine, rotation, submitter)?;
                }
                if index < self.geometry.blocks.len() {
                    self.run_block(run, index, rotation, submitter)?;
                    if filled(index) && run.back.as_ref().is_some_and(|&(at, _)| at == index) {
                        run.unfilled = Some(index);
                        if passed && first(place) {
                            self.run_fill(run, index, submitter)?;
                        }
                    }
                }
            }
            // The device orders a submission by all it touches: fronts left
            // queued would wait, with the next pass behind them, for the
            // fill that pass reads, and their requests with them.
            if early && index < self.geometry.blocks.len() && filled(index) {
                submitter.flush()?;
            }
            start = set.end;
        }
        Ok(())
    }

    /// Queue a scratch-served run's fill at block `index` (`OnceStage::Fill`,
    /// `StepSubmitter::fill`): where fills run beside compute no park of it
    /// holds compute, and where they share its queue a park holds only what
    /// follows it. It runs in its run's workspace, the graphs' shared arena,
    /// which it touches, and so orders against, only where its graph places
    /// something there.
    fn run_fill(&self, run: &mut TargetRun<'_>, index: usize, submitter: &mut StepSubmitter) -> Result<(), SubmitError> {
        run.unfilled = None;
        let (graph, bound) = self.graphs.fill(index).map_err(invalid)?;
        let mut bindings = bound.bindings();
        if let Some(bank) = &run.bank {
            bank.bind(graph, index, run.region.place, &mut bindings).map_err(invalid)?;
        }
        let mut active = run.launch.graph_workspace.slot_mut().activate(&graph.plan).map_err(device)?;
        let ready = active
            .attach(bindings, graph.plan.new_outputs().map_err(device)?)
            .map_err(device)?;
        let mut registered = Ok(());
        submitter.fill(ready, |outputs| {
            registered = run.bank.as_ref().map_or(Ok(()), |bank| bank.register_parks(graph, index, outputs));
        })?;
        registered.map_err(invalid)
    }

    /// Queue block `index`'s pass over merge set `set` of the chain's runs,
    /// once every front there is queued: one grouping and expert pass over
    /// the crossed rows from the set's first run's that has a back there
    /// through its last's (a stopped run's region between them is computed
    /// and never read), into the merged buffers its runs' backs combine
    /// from, and its reads into its group's `reads` record. It runs in that
    /// last run's workspace.
    fn run_pass(
        &self,
        runs: &mut Branded<'_, TargetRun<'_>>,
        index: usize,
        set: std::ops::Range<usize>,
        reads: Option<(&crate::experts::ExpertBank, usize)>,
        submitter: &mut StepSubmitter,
    ) -> Result<(), SubmitError> {
        let launches = runs.len();
        let mut backs = runs
            .iter_mut()
            .enumerate()
            .filter_map(|(place, run)| {
                let (_, class) = run.back.as_ref().filter(|(at, _)| set.contains(&place) && *at == index)?;
                Some((run.region.start, class.rows, run))
            });
        let Some(head) = backs.next() else {
            return Ok(());
        };
        let first = head.0;
        let (start, last_rows, run) = backs.last().unwrap_or(head);
        let rows = last_rows + start - first;
        let (graph, bound, pass_rows) = self.graphs.pass(index, launches).map_err(invalid)?;
        let into = &graph.ports;
        let group = run.region.group.clone();
        let SplitValues { crossing, merged } = group.split(index).map_err(invalid)?;
        // The pass's graph holds `pass_rows` crossed rows: the group's rows
        // from the set's first, or its last ones where fewer follow it, the
        // set's rows from `first` of them.
        let chain_rows = crossing.routes.extents()[0];
        let view = first.min(chain_rows - pass_rows);
        let first = first - view;
        let rows_of = |buffer: &seismic::Tensor| buffer.slice_leading(view, view + pass_rows).map_err(device);
        let mut bindings = bound.bindings();
        for (port, buffer) in into.crossing.iter().zip(crossing.pass().iter()) {
            bindings.set(port, &rows_of(buffer)?).map_err(device)?;
        }
        // A streamed layer's pass writes its set's reads into its group's
        // record at the set's first place (`BankPart::GroupReads`).
        if let Some((bank, record)) = reads {
            let at = bank.group_reads_at(record, set.start);
            for (port, words) in super::native_target_graph::placed_banks(bank, graph, index, at).map_err(invalid)? {
                bindings.set(port, &words).map_err(device)?;
            }
        }
        bindings
            .set(&into.experts, &merged.experts.slice_leading(0, into.blocks).map_err(device)?)
            .map_err(device)?;
        bindings.set(&into.inverse, &rows_of(&merged.inverse)?).map_err(device)?;
        for (port, entry) in [(&into.rows, rows), (&into.first, first)] {
            bindings
                .set(port, &merged.table.slice_leading(entry, entry + 1).map_err(device)?)
                .map_err(device)?;
        }
        let mut active = run
            .launch
            .graph_workspace
            .slot_mut()
            .activate(&graph.plan)
            .map_err(device)?;
        let ready = active
            .attach(bindings, graph.plan.new_outputs().map_err(device)?)
            .map_err(device)?;
        submitter.queue(ready, |_| {})?;
        Ok(())
    }

    /// Queue the graph of `stage` of block `index` of a step's run, of
    /// `class`, into the chain's free output slot, freeing the run's last.
    fn run_stage(
        &self,
        run: &mut TargetRun<'_>,
        index: usize,
        class: &BlockRun,
        stage: RunStage,
        rotation: &mut OutputRotation<'_>,
        submitter: &mut StepSubmitter,
    ) -> Result<(), SubmitError> {
        let RunLaunch {
            batch,
            state,
            graph_workspace,
            ..
        } = &mut run.launch;
        let (batch, state) = (&*batch, &*state);
        let (segments, slots, bank) = (run.segments, run.slots, run.bank.as_ref());
        let controls = &run.controls;
        let BlockRun {
            rows,
            segments: block_segments,
            listed: ref listed_tiles,
            scored,
        } = *class;
        {
            let block_started = self.trace.blocks.then(Instant::now);
            let (graph, bound) = self
                .graphs
                .block(
                    rows,
                    block_segments,
                    slots,
                    listed_tiles.as_ref().map_or(0, |(tiles, _)| *tiles),
                    scored,
                    index,
                    stage,
                )
                .map_err(invalid)?;
            let mut active = graph_workspace
                .slot_mut()
                .activate(&graph.plan)
                .map_err(device)?;
            let tap_rows = graph
                .taps
                .as_ref()
                .map(|_| self.tap_rows(&run.region, rows))
                .transpose()?;
            let mut bindings = bound.bindings();
            let hidden = graph.hidden.as_ref().ok_or_else(|| invalid("a run's graph reads no rows"))?;
            bindings.set(hidden, &run.hidden).map_err(device)?;
            // A split block binds its chain place's words of the bank's
            // placed parts and its rows of the crossing buffers; a back after
            // the group's pass the chain's experts' rows and its rows'
            // positions in them.
            if let Some(bank) = bank {
                bank.bind(graph, index, run.region.place, &mut bindings).map_err(invalid)?;
            }
            if let Some(ports) = &graph.ports {
                let SplitValues { crossing, merged } = run.region.group.split(index).map_err(invalid)?;
                match ports {
                    SplitPorts::Front(ports) => {
                        for (port, buffer) in ports.iter().zip(crossing.iter()) {
                            bindings.set(port, &run.region.rows(buffer, rows)?).map_err(device)?;
                        }
                    }
                    SplitPorts::Back(ports) => {
                        for (port, buffer) in ports.crossing.iter().zip(crossing.back().iter()) {
                            bindings.set(port, &run.region.rows(buffer, rows)?).map_err(device)?;
                        }
                        bindings.set(&ports.experts, &merged.experts).map_err(device)?;
                        bindings
                            .set(&ports.inverse, &run.region.rows(&merged.inverse, rows)?)
                            .map_err(device)?;
                    }
                }
            }
            // A graph that sets the run's step words writes its sequence.
            if let (Some(port), Some(bank)) = (&graph.sequence, bank) {
                active.write_input(port, &step_bytes(bank.sequence)).map_err(device)?;
            }
            match (&graph.per_layer, &run.per_layer_rows) {
                (None, _) => {}
                (Some(port), Some(inputs)) => bindings.set(port, inputs).map_err(device)?,
                (Some(_), None) => {
                    return Err(invalid("a per-layer block graph has no per-layer rows"));
                }
            }
            // An n-gram input reads the step's token rows: the host's, as
            // the entry kept them, or the previous step's selection.
            if let Some(ports) = &graph.ngram {
                match state.tokens() {
                    TargetTokens::Host => bindings
                        .set(&ports.tokens, &self.kept_token_rows(&run.region, rows)?)
                        .map_err(device)?,
                    TargetTokens::Selected(selected) => bindings
                        .set(&ports.tokens, selected.tensor())
                        .map_err(device)?,
                }
                // Its table's rows: written by the host before the step's
                // gate opens, from the step's tokens and the ones before.
                let table = self
                    .graphs
                    .stored
                    .get(&index)
                    .ok_or_else(|| invalid("an n-gram block has no stored table"))?;
                let rows = self.stored_step_rows(index, batch, state, run.notice.as_ref())?;
                let step = table.queue(rows).map_err(invalid)?;
                submitter.reads(table.clone(), step.value);
                bindings
                    .set_timeline(ports.table.gate, &step.gate, step.value)
                    .map_err(device)?;
                bindings
                    .set_timeline(ports.table.read, &step.read, step.value)
                    .map_err(device)?;
            }
            if let (Some(ports), Some(tap_rows)) = (&graph.taps, &tap_rows) {
                let taps = self.graphs.prepared.block_taps(index);
                bindings.set(&ports.taps, tap_rows).map_err(device)?;
                let [entry, middle, output] = stage.taps();
                for (port, tap) in [
                    (&ports.entry, taps.entry.filter(|_| entry)),
                    (&ports.middle, taps.middle.filter(|_| middle)),
                    (&ports.output, taps.output.filter(|_| output)),
                ] {
                    match (port, tap) {
                        (Some(port), Some(tap)) => active
                            .write_input(port, &i32_bytes(&[tap as i32]))
                            .map_err(device)?,
                        (None, None) => {}
                        _ => return Err(invalid("a block's tap ports differ from its taps")),
                    }
                }
            }
            // A graph that mixes binds the mixer's state and controls.
            match &graph.mixer {
                None => {}
                Some((
                    BlockStatePorts::Attention(ports, indexer_planes),
                    BlockControlPorts::Attention {
                        coordinates,
                        visible,
                        fresh,
                        destinations,
                        history_tiles,
                        indexer,
                    },
                )) => {
                    // A selected read's indexer: its planes, and its rows'
                    // spans and destinations in its two histories, written
                    // as the block's own read is.
                    match (indexer_planes, indexer) {
                        (Some(planes), Some(tables)) => {
                            let [(window_read, window), (block_read, blocks)] =
                                state.indexer(index as u32)?;
                            let [window_keys, window_coordinates] = window.as_slice() else {
                                return Err(invalid("an indexer's raw-key planes are not two"));
                            };
                            let [block_keys] = blocks.as_slice() else {
                                return Err(invalid("an indexer's block-key planes are not one"));
                            };
                            for (port, plane) in [
                                (&planes.window_keys, window_keys),
                                (&planes.window_coordinates, window_coordinates),
                                (&planes.block_keys, block_keys),
                            ] {
                                bindings.set(port, plane).map_err(device)?;
                            }
                            for (read, visible, destinations) in [
                                (
                                    window_read,
                                    &tables.window_visible,
                                    &tables.window_destinations,
                                ),
                                (block_read, &tables.block_visible, &tables.block_destinations),
                            ] {
                                bindings
                                    .set(visible, controls.bound(LaunchControl::Visible { read })?)
                                    .map_err(device)?;
                                bindings
                                    .set(
                                        destinations,
                                        controls.bound(LaunchControl::Destinations { read })?,
                                    )
                                    .map_err(device)?;
                            }
                        }
                        (None, None) => {}
                        _ => return Err(invalid("an indexer's planes and tables disagree")),
                    }
                    let (read, history) = state.history(LayerRef::Target(index as u32))?;
                    if history.len() != ports.len() {
                        return Err(invalid(
                            "the layer's history planes disagree with its attention entry",
                        ));
                    }
                    for (port, plane) in ports.iter().zip(history) {
                        bindings.set(port, plane).map_err(device)?;
                    }
                    // A listing class sealed for more segments than the
                    // launch has reads its spans padded.
                    let spans = match block_segments == segments {
                        true => LaunchControl::Visible { read },
                        false => LaunchControl::Padded { read },
                    };
                    for (port, control) in [
                        (coordinates, LaunchControl::Coordinates),
                        (visible, spans),
                        (fresh, LaunchControl::Fresh { read }),
                        (destinations, LaunchControl::Destinations { read }),
                    ] {
                        bindings.set(port, controls.bound(control)?).map_err(device)?;
                    }
                    match (history_tiles, &listed_tiles) {
                        (Some(tiles), Some((_, bytes))) => {
                            active.write_input(&tiles.port, bytes).map_err(device)?
                        }
                        (None, None) => {}
                        _ => {
                            return Err(invalid(
                                "the block's graph class and its history tile list disagree",
                            ));
                        }
                    }
                }
                Some((BlockStatePorts::Recurrent(ports), BlockControlPorts::Recurrent(recurrent))) => {
                    let arenas = state.recurrent_arenas()?;
                    for (offset, port) in ports.iter().enumerate() {
                        bindings
                            .set(
                                port,
                                arenas
                                    .get(run.recurrent_component + offset)
                                    .ok_or_else(|| invalid("recurrent bank arena is absent"))?,
                            )
                            .map_err(device)?;
                    }
                    for (port, control) in [
                        (&recurrent.segments, LaunchControl::Segments),
                        (&recurrent.stop, LaunchControl::Stop),
                        (&recurrent.previous_bank, LaunchControl::PreviousBank),
                        (&recurrent.previous_tape, LaunchControl::PreviousTape),
                        (&recurrent.following_bank, LaunchControl::FollowingBank),
                    ] {
                        bindings.set(port, controls.bound(control)?).map_err(device)?;
                    }
                }
                _ => return Err(invalid("sealed graph block state and controls disagree")),
            }
            let (next, outputs) = rotation.activate(&graph.plan)?;
            let ready = active.attach(bindings, outputs).map_err(device)?;
            let mut registered = Ok(());
            let outputs = submitter.queue(ready, |outputs| {
                registered = bank.map_or(Ok(()), |bank| bank.register_parks(graph, index, outputs));
            })?;
            registered.map_err(invalid)?;
            if let Some((BlockStatePorts::Recurrent(ports), _)) = &graph.mixer {
                run.recurrent_component += ports.len();
            }
            run.residual = graph.residual.as_ref().map(|edge| outputs.exported(edge).ok_or_else(|| invalid("target residual was not exported"))).transpose()?;
            run.hidden = outputs
                .exported(&graph.output)
                .ok_or_else(|| invalid("target block output was not exported"))?;
            rotation.recycle(std::mem::replace(&mut run.output, (next, outputs)))?;
            if let Some(started) = block_started {
                let kind = match &graph.mixer {
                    Some((BlockStatePorts::Attention(..), _)) => "attention",
                    Some((BlockStatePorts::Recurrent(_), _)) => "recurrent",
                    None => "back",
                };
                eprintln!(
                    "target block {index} {kind} rows {rows} lists {} history tiles {:.3}s",
                    listed_tiles.as_ref().map_or(0, |(tiles, _)| *tiles),
                    started.elapsed().as_secs_f64()
                );
            }
        }
        Ok(())
    }

    /// Queue a step's readout and boundary capture once every block is
    /// queued, and return its readout and commit span.
    fn finish_run(
        &self,
        mut run: TargetRun<'_>,
        rotation: &mut OutputRotation<'_>,
        submitter: &mut StepSubmitter,
    ) -> Result<(Option<TargetReadoutGraphResult>, CommitSpan), SubmitError> {
        let result = self.graph_readout(&mut run, submitter)?;
        let TargetRun {
            launch: RunLaunch { batch, state, .. },
            output,
            residual,
            hidden,
            first_commit,
            ..
        } = run;
        if let Some(boundary) = state.0.boundary() {
            let features = &result
                .as_ref()
                .ok_or_else(|| invalid("boundary capture has no features"))?
                .features;
            let row = batch.actual_rows - 1;
            let source = features
                .tensor()
                .slice_leading(row as u64, row as u64 + 1)
                .map_err(device)?;
            let destination = boundary.allocation().tensor().map_err(device)?;
            let copy = self.boundary_copy()?;
            let mut bindings = copy.plan.bindings();
            bindings.set(&copy.source, &source).map_err(device)?;
            bindings
                .set(&copy.destination, destination)
                .map_err(device)?;
            let mut slot = copy.plan.new_slot().map_err(device)?;
            let ready = slot
                .attach(bindings, copy.plan.new_outputs().map_err(device)?)
                .map_err(device)?;
            submitter.queue(ready, |_| {})?;
        }
        let last_commit = Instant::now();
        // Every consumer of the final hidden rows is queued, so its output
        // slot returns to the rotation; the device queue orders any reuse.
        drop(hidden);
        drop(residual);
        rotation.recycle(output)?;
        Ok((
            result,
            CommitSpan {
                first: first_commit,
                last: last_commit,
            },
        ))
    }
}

/// The batch rows a step's blocks run over: `rows` rows (a row class) from
/// batch row `first`. Every block before the stateless tail runs the whole
/// batch; the tail runs the smallest window that holds the rows the readout
/// consumes. A row's computation in a block reads only its own hidden,
/// per-layer and control rows and the history, so a window's rows compute
/// what they would in the whole batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RowWindow {
    first: u64,
    rows: u64,
}

impl RowWindow {
    fn whole(rows: u64) -> Self {
        Self { first: 0, rows }
    }

    /// The window of the row class covering `out_rows` within a batch of
    /// `rows` rows, placed at their first row unless that would pass the
    /// batch's end. `None` when no row is consumed.
    fn consumed(out_rows: &[i32], rows: u64) -> Result<Option<Self>, SubmitError> {
        let row = |row: &i32| {
            u64::try_from(*row)
                .ok()
                .filter(|row| *row < rows)
                .ok_or_else(|| invalid("readout row lies outside the batch"))
        };
        let rows_read = out_rows.iter().map(row).collect::<Result<Vec<_>, _>>()?;
        let (Some(&first), Some(&last)) = (rows_read.iter().min(), rows_read.iter().max()) else {
            return Ok(None);
        };
        let class = magnitude_batching::row_class((last - first + 1) as usize)
            .ok_or_else(|| invalid("consumed rows have no row class"))? as u64;
        Ok(Some(Self {
            first: first.min(rows - class),
            rows: class,
        }))
    }

    /// The window's rows of a tensor over the batch rows.
    fn slice(self, rows: &Tensor) -> Result<Tensor, SubmitError> {
        rows.slice_leading(self.first, self.first + self.rows)
            .map_err(device)
    }

    /// Batch row `row` as a row of the window.
    fn row(self, row: i32) -> Result<i32, SubmitError> {
        u64::try_from(row)
            .ok()
            .and_then(|row| row.checked_sub(self.first))
            .filter(|row| *row < self.rows)
            .map(|row| row as i32)
            .ok_or_else(|| invalid("readout row lies outside the step's row window"))
    }

    /// The window's rows of a per-row host table over `batch_rows` rows.
    fn rows_of(self, table: &[u8], batch_rows: u64) -> Vec<u8> {
        let stride = table.len() / batch_rows as usize;
        table[self.first as usize * stride..(self.first + self.rows) as usize * stride].to_vec()
    }
}

/// A launch's per-step controls: what the host reads of them to class its
/// blocks, and their one upload, which every block graph of the launch
/// binds (`ControlSource::Bound`).
struct GraphControls {
    /// One control set per history read, in read order: an attention block
    /// reads its own read's.
    histories: Vec<HistoryControls>,
    /// Each control's view of the launch's upload, in upload order
    /// (`LaunchControlLayout::controls`).
    launch: Vec<(LaunchControl, Tensor)>,
}

/// What the host reads of one history read's controls: its rows' visible
/// spans and destinations (`history_tile_bytes`, `selection_scores`).
struct HistoryControls {
    visible: Vec<u8>,
    destinations: Vec<u8>,
}

impl GraphControls {
    /// Keep the per-row controls of `window`'s rows, of controls over
    /// `batch_rows` rows. The per-slot recurrent controls stay whole: only
    /// attention blocks run a window (`stateless_tail`).
    fn window(&mut self, batch_rows: u64, window: RowWindow) -> Result<(), SubmitError> {
        for history in &mut self.histories {
            history.visible = window.rows_of(&history.visible, batch_rows);
            history.destinations = window.rows_of(&history.destinations, batch_rows);
        }
        for (control, view) in &mut self.launch {
            if control.per_row() {
                *view = window.slice(view)?;
            }
        }
        Ok(())
    }

    /// The launch's view of `control`.
    fn bound(&self, control: LaunchControl) -> Result<&Tensor, SubmitError> {
        self.launch
            .iter()
            .find(|(of, _)| *of == control)
            .map(|(_, view)| view)
            .ok_or_else(|| invalid(format!("the launch's upload has no {control:?}")))
    }

    /// The controls of `batch`, uploaded once into an idle region of the
    /// launch's graph workspace `slot` as `layout` places them.
    fn new(
        batch: &TargetBatchUpload<'_>,
        layout: &LaunchControlLayout,
        slot: &mut seismic::NativeGraphFamilySlot,
    ) -> Result<Self, SubmitError> {
        let rows = batch.class.rows();
        let segments = batch.class.segments();
        let slots = batch
            .segments
            .len()
            .checked_sub(1)
            .ok_or_else(|| invalid("batch has no terminal segment row"))?;
        if !slots.is_power_of_two()
            || batch.actual_slots > slots
            || batch.actual_rows > rows
            || batch.tokens.len() != rows
            || batch.coordinates.len() != rows
            || batch.histories.iter().any(|history| {
                history.visible.len() != rows
                    || history.fresh.len() != rows
                    || history.destinations.len() != rows
                    || history
                        .visible
                        .iter()
                        .any(|ranges| ranges.len() != segments)
            })
        {
            return Err(invalid(
                "validated batch does not match exact graph row, segment, and slot classes",
            ));
        }
        let histories = batch
            .histories
            .iter()
            .map(|history| HistoryControls {
                visible: history
                    .visible
                    .iter()
                    .flatten()
                    .flatten()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
                destinations: i32_bytes(&history.destinations),
            })
            .collect::<Vec<_>>();
        let pairs = |pairs: &[[i32; 2]]| i32_bytes(pairs.as_flattened());
        let (rows, segments, slots) = (rows as u64, segments as u64, batch.actual_slots as u64);
        let controls = layout.controls(rows, segments, slots);
        let read = |read: usize| {
            batch
                .histories
                .get(read)
                .zip(histories.get(read))
                .ok_or_else(|| invalid("the batch lacks a history read its layout places"))
        };
        let bytes = controls
            .iter()
            .map(|(control, extents)| {
                Ok(match *control {
                    LaunchControl::Coordinates => i32_bytes(batch.coordinates.as_flattened()),
                    LaunchControl::Visible { read: at } => read(at)?.1.visible.clone(),
                    LaunchControl::Padded { read: at } => {
                        padded_spans(&read(at)?.1.visible, segments, extents[1])
                    }
                    LaunchControl::Fresh { read: at } => pairs(&read(at)?.0.fresh),
                    LaunchControl::Destinations { read: at } => read(at)?.1.destinations.clone(),
                    LaunchControl::Segments => pairs(&batch.segments[..=batch.actual_slots]),
                    LaunchControl::Stop => i32_bytes(&batch.stop[..batch.actual_slots]),
                    LaunchControl::PreviousBank => i32_bytes(&batch.bank[..batch.actual_slots]),
                    LaunchControl::PreviousTape => {
                        i32_bytes(&batch.previous_tape[..batch.actual_slots])
                    }
                    LaunchControl::FollowingBank => {
                        i32_bytes(&batch.following_bank[..batch.actual_slots])
                    }
                })
            })
            .collect::<Result<Vec<_>, SubmitError>>()?;
        let tensors = controls
            .iter()
            .zip(&bytes)
            .map(|((_, extents), bytes)| (Element::i32(), extents.as_slice(), bytes.as_slice()))
            .collect::<Vec<_>>();
        let uploaded = slot.upload(&tensors).map_err(device)?;
        let launch = controls
            .into_iter()
            .map(|(control, _)| control)
            .zip(uploaded)
            .collect();
        Ok(Self { histories, launch })
    }
}

impl TargetProgram for NativeTargetProgram {
    type Submission = DeviceSubmission<TargetLaunchCore, TargetLaunchWorkspace, TargetOutput>;

    /// The device queue completes command sequences in order, so each
    /// launch ends its own sequences after its readout and completes with
    /// them.
    fn submit(
        &mut self,
        chain: Chain<ValidatedTargetLaunch>,
        mut spare: TargetGraphOutputLease,
    ) -> Result<Chain<Self::Submission>, super::ChainRefusal> {
        Ok(self.queue_chain(chain, &mut spare)?.map(|(launch, queued)| {
            let (core, workspace) = launch.into_submission_parts();
            let boundary = core.boundary().cloned();
            DeviceSubmission::new(
                self.waiter.completion(queued.completions),
                core,
                workspace,
                TargetOutput {
                    boundary,
                    readout: queued.readout,
                    commits: queued.commits,
                    landed_reads: queued.landed_reads,
                    stored_reads: queued.stored_reads,
                    expert_step: queued.expert_step,
                },
            )
        }))
    }
}

#[cfg(test)]
mod row_window_tests {
    use super::RowWindow;

    #[test]
    fn an_unread_step_runs_no_tail() {
        assert_eq!(RowWindow::consumed(&[], 512).unwrap(), None);
    }

    #[test]
    fn the_window_is_the_row_class_covering_the_consumed_rows() {
        // A prompt chunk's last row.
        let last = RowWindow::consumed(&[511], 512).unwrap().unwrap();
        assert_eq!(
            last,
            RowWindow {
                first: 511,
                rows: 1
            }
        );
        assert_eq!(last.row(511).unwrap(), 0);
        assert!(last.row(510).is_err());
        // Rows spanning 11 rows take the 16-row class, kept inside the batch.
        let spread = RowWindow::consumed(&[500, 510, 505], 512).unwrap().unwrap();
        assert_eq!(
            spread,
            RowWindow {
                first: 496,
                rows: 16
            }
        );
        assert_eq!(spread.row(510).unwrap(), 14);
        // Rows across the batch keep the whole batch.
        assert_eq!(
            RowWindow::consumed(&[0, 511], 512).unwrap().unwrap(),
            RowWindow::whole(512)
        );
        assert!(RowWindow::consumed(&[512], 512).is_err());
    }

    #[test]
    fn a_window_takes_its_rows_of_a_host_table() {
        let table = (0..8u8).flat_map(|row| [row, row]).collect::<Vec<_>>();
        let window = RowWindow { first: 4, rows: 2 };
        assert_eq!(window.rows_of(&table, 8), vec![4, 4, 5, 5]);
    }
}
