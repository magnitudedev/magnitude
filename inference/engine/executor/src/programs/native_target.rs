//! Ordered target execution through sealed Seismic native graphs. A step
//! queues one run per graph without waiting; the device queue orders them,
//! and the returned submission owns the launch until completion is observed.

use super::{
    DeviceSubmission, TargetProgram,
    graph::readout::{
        BoundTargetReadoutGraphs, ReadoutClass, ReadoutKind, certifies, shapes, write_selection,
    },
    native_target_graph::{BoundTargetGraphs, EntryTokens, TokenRow},
};
use crate::device_table::{SelectedRead, SubmittedRead};
use crate::host_tables::HostTableError;
use crate::stored_table::{NoticeStep, StepRows, StoredTable};
use crate::operators::block::{BlockControlPorts, BlockStatePorts};
use crate::{
    ConditioningRef, ConditioningSlice, DeviceError, GraphOutputTensor, InvariantError,
    LaunchRows, LayerMajor, NativeGraphOutputLease, NativeGraphWorkspaceLease, SubmitError,
    TargetGraphOutputLease, TargetGraphWorkspaceLease, TargetLaunchCore, TargetLaunchWorkspace, TargetTokens,
    ValidatedTargetLaunch, completion::CompletionWaiter, native::AttestedState,
};
use magnitude_batching::{Demand, TargetBatchUpload};
use magnitude_family_contracts::Decoder;
use magnitude_kernels::{
    conditioning_overlay, feature_copy, readout_features_rows, readout_head_rows,
    readout_scores_rows, readout_selected_rows,
};
use magnitude_state::LayerRef;
use seismic::{
    Device, Element, NativeGraphCompletion, NativeGraphOutputs, NativeGraphPlan,
    NativeGraphSequence, NativeKernel, NativePort, Tensor, WorkflowTensor,
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
/// of N runs has about log2(N) submission boundaries instead of N.
struct StepSubmitter {
    device: Device,
    sequence: NativeGraphSequence,
    queued: usize,
    limit: usize,
    completions: Vec<NativeGraphCompletion>,
    /// The stored-table reads of the runs queued since the last
    /// submission, then of the submitted runs.
    pending_reads: Vec<StoredRead>,
    reads: Vec<StoredRead>,
    /// The landed reads of the runs queued since the last submission, then
    /// of the submitted runs.
    pending_landed: Vec<SelectedRead>,
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
            sequence: device.native_sequence(),
            queued: 0,
            limit: 1,
            completions: Vec::new(),
            pending_reads: Vec::new(),
            reads: Vec::new(),
            pending_landed: Vec::new(),
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

    fn queue(
        &mut self,
        ready: seismic::ReadyNativeGraphRun<'_>,
    ) -> Result<NativeGraphOutputs, SubmitError> {
        let outputs = ready.queue(&mut self.sequence).map_err(device)?;
        self.queued += 1;
        if self.queued == self.limit {
            self.flush()?;
            self.limit *= 2;
        }
        Ok(outputs)
    }

    /// Ready an upload region of `slot` for its next activation. A step
    /// queueing more runs than the slot has regions (a layer-major step)
    /// submits what it queued and awaits its oldest submissions until one
    /// is idle; a step of at most as many runs never waits.
    fn await_upload(&mut self, slot: &seismic::NativeGraphFamilySlot) -> Result<(), SubmitError> {
        if slot.upload_idle() {
            return Ok(());
        }
        self.flush()?;
        while !slot.upload_idle() {
            if self.completions.is_empty() {
                return Err(invalid("no submitted run frees an upload region"));
            }
            self.completions.remove(0).wait().map_err(device)?;
        }
        Ok(())
    }

    /// Submit what is queued.
    fn flush(&mut self) -> Result<(), SubmitError> {
        self.queued = 0;
        let sequence = std::mem::replace(&mut self.sequence, self.device.native_sequence());
        if !sequence.is_empty() {
            self.completions.push(sequence.submit().map_err(device)?);
            self.reads.append(&mut self.pending_reads);
            self.landed
                .extend(self.pending_landed.drain(..).map(SelectedRead::submitted));
        }
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

/// A sealed single-range conditioning overlay for one row count. The source
/// and the destination hidden rows are late bindings, so a plan serves every
/// placement of that many rows.
#[derive(Clone)]
struct OverlayGraph {
    plan: NativeGraphPlan,
    source: NativePort,
    destination: NativePort,
}

/// What a step's readout rows read their values through, over the
/// decoder's final rows and the exit norm.
#[derive(Clone)]
pub(crate) struct ReadoutValueWeights {
    /// The exit norm, or an empty F32 tensor for an exit that normalizes
    /// nothing.
    pub norm: Tensor,
    /// Logits of chosen tokens, when the plan binds them
    /// (`ReadoutBinding::logits`).
    pub logits: Option<SelectedLogitsWeights>,
    /// The model's readout head, when it has one.
    pub head: Option<HeadValueWeights>,
    /// The final normalization of retained rows (`readout_features_rows`),
    /// which `AttestedState::feature_copy` copies into their stores.
    pub features: NativeKernel<readout_features_rows::Entry>,
    /// The scores of retained rows (`readout_scores_rows`), when the plan
    /// binds them (`ReadoutBinding::scores`).
    pub scores: Option<NativeKernel<readout_scores_rows::Entry>>,
}

/// The selected-row projection of chosen logits (`readout_selected_rows`,
/// bound by `ReadoutBinding::logits`) and the packed output projection it
/// reads.
#[derive(Clone)]
pub(crate) struct SelectedLogitsWeights {
    pub selected: NativeKernel<readout_selected_rows::Entry>,
    pub weight: Tensor,
}

/// The element a readout head is stored in, which its values are published
/// in: the load admits a head stored F32 or BF16 only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadElement {
    F32,
    Bf16,
}

impl HeadElement {
    /// The element of a resident head `weight`.
    pub fn of(weight: &Tensor) -> Result<Self, String> {
        match weight.element() {
            element if element == Element::f32() => Ok(Self::F32),
            element if element == Element::bf16() => Ok(Self::Bf16),
            element => Err(format!("a readout head is resident as {}", element.name())),
        }
    }

    /// A head value as a projection in this element publishes it: an F32
    /// head its F32 product, a BF16 head (a BF16 linear layer) its product
    /// rounded to BF16, to nearest even. A NaN stays a NaN: its truncated
    /// payload made quiet, as BF16 conversion does, since rounding its bits
    /// could carry into the sign.
    pub fn publish(self, value: f32) -> f32 {
        match self {
            Self::F32 => value,
            Self::Bf16 => {
                let bits = value.to_bits();
                if value.is_nan() {
                    return f32::from_bits((bits | 0x0040_0000) & 0xffff_0000);
                }
                f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000)
            }
        }
    }
}

/// The model's readout head (`readout_head_rows`, without a second-level
/// scale or softcap) and its weight, dense in the element it is stored in.
#[derive(Clone)]
pub(crate) struct HeadValueWeights {
    pub kernel: NativeKernel<readout_head_rows::Entry>,
    pub weight: Tensor,
    pub element: HeadElement,
    /// The empty F32 tensor the entry's absent scale port binds.
    pub unscaled: Tensor,
}

/// The selected-vocabulary projection of `outputs` final rows of a
/// `rows`-row window onto `columns` vocabulary rows, sealed on first use.
/// Every input is a port bound per run, so the plan holds no storage but
/// the projection's scratch.
#[derive(Clone)]
struct ValuesGraph {
    plan: NativeGraphPlan,
    hidden: NativePort,
    norm: NativePort,
    weight: NativePort,
    out_rows: NativePort,
    selected: NativePort,
    values: WorkflowTensor,
}

/// The readout head over `outputs` final rows of a `rows`-row window,
/// sealed on first use; every input is a port bound per run.
#[derive(Clone)]
struct HeadValuesGraph {
    plan: NativeGraphPlan,
    hidden: NativePort,
    norm: NativePort,
    weight: NativePort,
    out_rows: NativePort,
    scale: NativePort,
    values: WorkflowTensor,
}

/// The final normalization of `rows` final rows of a `window`-row window
/// into a store's rows, sealed on first use; every input is a port bound
/// per run.
#[derive(Clone)]
struct RetainGraph {
    plan: NativeGraphPlan,
    hidden: NativePort,
    norm: NativePort,
    out_rows: NativePort,
    store: NativePort,
}

/// The scores of one query row against `keys` key rows, `keys` a power of
/// two: every key run is computed as consecutive runs of these classes
/// ([`power_of_two_runs`]), each sealed on first use. Both inputs are ports
/// bound per run.
#[derive(Clone)]
struct ScoresGraph {
    plan: NativeGraphPlan,
    query: NativePort,
    keys: NativePort,
    scores: WorkflowTensor,
}

/// `rows` as consecutive runs whose lengths are powers of two, longest
/// first: at most one run of each class, so a graph sealed once per class
/// computes any run exactly.
fn power_of_two_runs(rows: std::ops::Range<u64>) -> impl Iterator<Item = std::ops::Range<u64>> {
    let mut start = rows.start;
    std::iter::from_fn(move || {
        let left = rows.end.checked_sub(start).filter(|left| *left > 0)?;
        let run = start..start + (1 << left.ilog2());
        start = run.end;
        Some(run)
    })
}

/// A group of a step's logits readouts, read by one projection onto the
/// union of their tokens: its `[outputs, columns]` F32 values, both classes
/// powers of two, stay within the load's logits budget, the planned logits
/// of one launch (`NativeTargetProgram::logits_budget`). A readout row is
/// a group alone whatever its tokens.
#[derive(Default)]
struct LogitsGroup {
    rows: u64,
    columns: std::collections::BTreeSet<i32>,
}

impl LogitsGroup {
    /// Whether a row reading `tokens` joins the group within `budget`
    /// values.
    fn admits(&self, tokens: &[i32], budget: u64) -> bool {
        let columns = self.columns.iter().chain(tokens).collect::<std::collections::BTreeSet<_>>();
        (self.rows + 1).next_power_of_two() * (columns.len() as u64).next_power_of_two() <= budget
    }

    fn add(&mut self, tokens: &[i32]) {
        self.rows += 1;
        self.columns.extend(tokens);
    }

    /// The device bytes its projection allocates: its output rows and
    /// columns, uploaded as i32, and its F32 values.
    fn bytes(&self) -> u64 {
        let outputs = self.rows.next_power_of_two();
        let columns = (self.columns.len() as u64).next_power_of_two();
        4 * (outputs + columns + outputs * columns)
    }
}

/// One group of a step's logits readouts: the projected `[outputs, columns]`
/// logits of its readout rows onto the ascending union of their tokens.
pub struct ReadoutValueGroup {
    pub values: Tensor,
    /// The union, ascending: column `c` holds token `columns[c]`'s logits.
    pub columns: Vec<i32>,
    /// The group's logits readouts in output order: each one's index in the
    /// batch's readouts (`TargetBatchUpload::readouts`) and the tokens it
    /// reads, in its readout's order.
    pub rows: Vec<(usize, Arc<[i32]>)>,
}

/// The step's readout head values: `[outputs, K]` F32 products of the
/// head's rows, in output order, computed from a weight stored in
/// `element`.
pub struct HeadReadoutValues {
    pub values: Tensor,
    pub element: HeadElement,
    /// Each output's index in the batch's readouts.
    pub readouts: Vec<usize>,
}

/// One scores readout of a step: its index in the batch's readouts and its
/// `[1, keys]` F32 scores per run of key rows, in key order.
pub struct ScoresReadoutValues {
    pub readout: usize,
    pub runs: Vec<Tensor>,
}

/// A step's readout values, by kind.
#[derive(Default)]
pub struct StepReadoutValues {
    pub logits: Vec<ReadoutValueGroup>,
    pub head: Option<HeadReadoutValues>,
    pub scores: Vec<ScoresReadoutValues>,
}

#[derive(Clone)]
pub struct NativeTargetProgram {
    device: Device,
    state: AttestedState,
    geometry: Decoder,
    graphs: Rc<BoundTargetGraphs>,
    readout_graphs: Rc<BoundTargetReadoutGraphs>,
    readout_values: ReadoutValueWeights,
    /// Logits readout plans by (window rows, output rows, columns) classes,
    /// sealed on first use.
    value_graphs: Rc<RefCell<BTreeMap<(u64, u64, u64), ValuesGraph>>>,
    /// Readout head plans by (window rows, output rows) classes, sealed on
    /// first use.
    head_graphs: Rc<RefCell<BTreeMap<(u64, u64), HeadValuesGraph>>>,
    /// Retention plans by (window rows, retained rows) classes, the
    /// retained rows a power of two, sealed on first use.
    retain_graphs: Rc<RefCell<BTreeMap<(u64, u64), RetainGraph>>>,
    /// Scores plans by key rows class, sealed on first use.
    scores_graphs: Rc<RefCell<BTreeMap<u64, ScoresGraph>>>,
    /// The most values one group of logits readouts projects onto: the
    /// logits of the load's largest projected rows class over its
    /// vocabulary class, the logits memory it plans for one launch.
    logits_budget: u64,
    /// Row copy plans by row count and width, sealed on first use.
    overlays: Rc<RefCell<BTreeMap<(u64, u64), OverlayGraph>>>,
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
    /// The values of the step's readouts, per batch of its rows
    /// (`LaunchRows::batches`).
    pub readout_values: Vec<StepReadoutValues>,
    pub commits: CommitSpan,
    /// The step's landed reads of the table rows the device selected its
    /// tokens for (`DeviceTable::read_selected`); none of a step of the
    /// host's tokens or of a table held whole.
    pub landed_reads: Vec<SubmittedRead>,
    /// The step's reads of stored tables (`StoredTable`).
    pub stored_reads: Vec<StoredRead>,
}

impl TargetOutput {
    /// The step completed: its landed reads bring the rows they landed into
    /// slots of their own (`SubmittedRead::complete`). Why one could not
    /// land its rows: every slot of the step then read other rows, and its
    /// outputs are not to be used.
    pub fn complete_landed_reads(&mut self) -> Option<crate::TableReadError> {
        std::mem::take(&mut self.landed_reads)
            .into_iter()
            .map(SubmittedRead::complete)
            .fold(None, |failure, outcome| {
                failure.or(outcome.err().map(|detail| crate::TableReadError { detail }))
            })
    }

    /// Why a stored table's rows batch slot `slot` named could not be read,
    /// once the step completed: that slot's outputs are not to be used.
    pub fn table_read_failure(&self, slot: usize) -> Option<crate::TableReadError> {
        self.stored_reads.iter().find_map(|read| {
            read.table
                .failure(read.value)
                .filter(|failure| failure.fails(slot))
                .map(|failure| crate::TableReadError {
                    detail: failure.error,
                })
        })
    }

    /// The step's outcome was taken: its stored tables forget its failures.
    pub fn retire_stored_reads(&self) {
        for read in &self.stored_reads {
            read.table.retire(read.value);
        }
    }
}

/// Runs of one step queued on the device, with the outputs they will fill.
struct QueuedStep {
    readout: Option<TargetReadoutGraphResult>,
    readout_values: Vec<StepReadoutValues>,
    commits: CommitSpan,
    completions: Vec<NativeGraphCompletion>,
    landed_reads: Vec<SubmittedRead>,
    stored_reads: Vec<StoredRead>,
}

impl RowState<'_> {
    fn slots(&self) -> usize {
        self.0.advances().len()
    }
    fn stores(&self, slot: usize) -> &[crate::FeatureRef] {
        self.0.stores(slot)
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
    /// `layer`'s history read (`StateStore::history_read`, the index of its
    /// controls) and planes in the codec's plane-descriptor order, as the
    /// attention graph's state ports take them. A Shared layer binds its
    /// source layer's planes and its source domain's shared read.
    fn history(&self, layer: LayerRef) -> Result<(usize, Vec<&Tensor>), SubmitError> {
        let store = self.0.store();
        let missing = || invalid("attention layer has no history in its store");
        let source = store.history_source(layer).ok_or_else(missing)?;
        let read = store.history_read(layer).ok_or_else(missing)?;
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
    /// Visit the graph constants the program binds.
    pub(crate) fn visit_constants(&self, visit: &mut dyn FnMut(&seismic::Tensor)) {
        self.graphs.visit_constants(visit);
    }

    pub(crate) fn constant_bytes(&self) -> Result<u64, &'static str> {
        self.graphs.constant_bytes()
    }

    pub(crate) fn new(
        device: Device,
        state: AttestedState,
        geometry: Decoder,
        graphs: BoundTargetGraphs,
        readout_graphs: BoundTargetReadoutGraphs,
        readout_values: ReadoutValueWeights,
    ) -> Result<Self, SubmitError> {
        let waiter = CompletionWaiter::spawn()
            .map_err(|error| SubmitError::Device(DeviceError::Execution(error)))?;
        let logits_budget = (readout_graphs.prepared.max_projected_rows().max(1) as u64)
            .next_power_of_two()
            * geometry.vocabulary.next_power_of_two();
        Ok(Self {
            device,
            state,
            geometry,
            graphs: Rc::new(graphs),
            readout_graphs: Rc::new(readout_graphs),
            readout_values,
            value_graphs: Rc::default(),
            head_graphs: Rc::default(),
            retain_graphs: Rc::default(),
            scores_graphs: Rc::default(),
            logits_budget,
            overlays: Rc::new(RefCell::new(BTreeMap::new())),
            boundary_copy: Rc::default(),
            waiter,
            trace: TargetTrace::from_environment(),
        })
    }

    /// The readout over `hidden`, the step's final rows from batch row
    /// `window.first` on (`RowWindow`).
    fn graph_readout(
        &self,
        batch: &TargetBatchUpload<'_>,
        window: RowWindow,
        hidden: &Tensor,
        residual: Option<&Tensor>,
        workspace: &mut NativeGraphWorkspaceLease,
        mut output: NativeGraphOutputLease,
        submitter: &mut StepSubmitter,
    ) -> Result<Option<TargetReadoutGraphResult>, SubmitError> {
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
            .map(|_| self.tap_rows(class.rows))
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
        let outputs = submitter.queue(ready)?;
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

    /// `PreparedTargetGraphs::layer_major_rows`.
    pub(crate) fn layer_major_rows(&self) -> usize {
        self.graphs.prepared.layer_major_rows() as usize
    }

    /// The readouts the load computes (`ReadoutValueWeights`).
    pub(crate) fn readouts(&self) -> crate::ReadoutSupport {
        crate::ReadoutSupport {
            logits: self.readout_values.logits.is_some(),
            head: self.readout_values.head.is_some(),
            scores: self.readout_values.scores.is_some(),
        }
    }

    fn values_graph(&self, rows: u64, outputs: u64, columns: u64) -> Result<ValuesGraph, SubmitError> {
        if let Some(graph) = self.value_graphs.borrow().get(&(rows, outputs, columns)) {
            return Ok(graph.clone());
        }
        let norm_weight = &self.readout_values.norm;
        let weights = self
            .readout_values
            .logits
            .as_ref()
            .ok_or_else(|| invalid("the load reads no logits at chosen rows"))?;
        let form = crate::operators::InputForm::exit(&self.geometry);
        let mut graph = self.device.native_graph();
        let hidden = graph
            .port(Element::f32(), &[rows, form.parts * self.geometry.hidden])
            .map_err(device)?;
        let norm = graph
            .port(norm_weight.element(), norm_weight.extents())
            .map_err(device)?;
        let weight = graph
            .port(weights.weight.element(), weights.weight.extents())
            .map_err(device)?;
        let out_rows = graph.port(Element::i32(), &[outputs]).map_err(device)?;
        let selected = graph.port(Element::i32(), &[columns]).map_err(device)?;
        let values = graph
            .enqueue(
                &weights.selected,
                readout_selected_rows::WorkflowArgs {
                    hidden: hidden.tensor().into(),
                    norm: norm.tensor().into(),
                    weight: weight.tensor().into(),
                    out_rows: out_rows.tensor().into(),
                    selected: selected.tensor().into(),
                    epsilon: super::graph::readout::readout_epsilon(&self.geometry)
                        .map_err(invalid)?,
                    softcap: super::graph::readout::readout_softcap(&self.geometry),
                },
            )
            .map_err(device)?
            .value;
        graph.export(&values).map_err(device)?;
        let plan = graph.seal().map_err(device)?;
        if plan.upload_bytes() != 0 {
            return Err(invalid("readout values unexpectedly upload host inputs"));
        }
        let graph = ValuesGraph {
            plan,
            hidden,
            norm,
            weight,
            out_rows,
            selected,
            values,
        };
        self.value_graphs
            .borrow_mut()
            .insert((rows, outputs, columns), graph.clone());
        Ok(graph)
    }

    fn head_graph(&self, rows: u64, outputs: u64) -> Result<HeadValuesGraph, SubmitError> {
        if let Some(graph) = self.head_graphs.borrow().get(&(rows, outputs)) {
            return Ok(graph.clone());
        }
        let norm_weight = &self.readout_values.norm;
        let weights = self
            .readout_values
            .head
            .as_ref()
            .ok_or_else(|| invalid("the model has no readout head"))?;
        let form = crate::operators::InputForm::exit(&self.geometry);
        let mut graph = self.device.native_graph();
        let hidden = graph
            .port(Element::f32(), &[rows, form.parts * self.geometry.hidden])
            .map_err(device)?;
        let norm = graph
            .port(norm_weight.element(), norm_weight.extents())
            .map_err(device)?;
        let weight = graph
            .port(weights.weight.element(), weights.weight.extents())
            .map_err(device)?;
        let out_rows = graph.port(Element::i32(), &[outputs]).map_err(device)?;
        let scale = graph
            .port(weights.unscaled.element(), weights.unscaled.extents())
            .map_err(device)?;
        let values = graph
            .enqueue(
                &weights.kernel,
                readout_head_rows::WorkflowArgs {
                    hidden: hidden.tensor().into(),
                    norm: norm.tensor().into(),
                    weight: weight.tensor().into(),
                    out_rows: out_rows.tensor().into(),
                    epsilon: super::graph::readout::readout_epsilon(&self.geometry)
                        .map_err(invalid)?,
                    // A head's values are its products: no softcap.
                    softcap: 0.0,
                    weight_scale: scale.tensor().into(),
                },
            )
            .map_err(device)?
            .value;
        graph.export(&values).map_err(device)?;
        let plan = graph.seal().map_err(device)?;
        if plan.upload_bytes() != 0 {
            return Err(invalid("readout head values unexpectedly upload host inputs"));
        }
        let graph = HeadValuesGraph {
            plan,
            hidden,
            norm,
            weight,
            out_rows,
            scale,
            values,
        };
        self.head_graphs
            .borrow_mut()
            .insert((rows, outputs), graph.clone());
        Ok(graph)
    }

    fn retain_graph(&self, rows: u64, retained: u64) -> Result<RetainGraph, SubmitError> {
        if let Some(graph) = self.retain_graphs.borrow().get(&(rows, retained)) {
            return Ok(graph.clone());
        }
        let norm_weight = &self.readout_values.norm;
        let form = crate::operators::InputForm::exit(&self.geometry);
        let activation =
            Element::dense(crate::planning::activation_dtype(self.geometry.activation_dtype));
        let mut graph = self.device.native_graph();
        let hidden = graph
            .port(Element::f32(), &[rows, form.parts * self.geometry.hidden])
            .map_err(device)?;
        let norm = graph
            .port(norm_weight.element(), norm_weight.extents())
            .map_err(device)?;
        let out_rows = graph.port(Element::i32(), &[retained]).map_err(device)?;
        let mut store = graph
            .port(activation, &[retained, self.geometry.hidden])
            .map_err(device)?;
        let features = graph
            .enqueue(
                &self.readout_values.features,
                readout_features_rows::WorkflowArgs {
                    // The entry reads its rows' parts through a
                    // `[rows, parts, hidden]` view.
                    hidden: (&hidden
                        .tensor()
                        .reshape(&[rows, form.parts, self.geometry.hidden]))
                        .into(),
                    norm: norm.tensor().into(),
                    out_rows: out_rows.tensor().into(),
                    epsilon: super::graph::readout::readout_epsilon(&self.geometry)
                        .map_err(invalid)?,
                },
            )
            .map_err(device)?
            .value;
        graph
            .enqueue(
                &self.state.feature_copy,
                feature_copy::WorkflowArgs {
                    input: (&features).into(),
                    out: store.tensor_mut().into(),
                },
            )
            .map_err(device)?;
        let plan = graph.seal().map_err(device)?;
        if plan.upload_bytes() != 0 {
            return Err(invalid("retained rows unexpectedly upload host inputs"));
        }
        let graph = RetainGraph {
            plan,
            hidden,
            norm,
            out_rows,
            store,
        };
        self.retain_graphs
            .borrow_mut()
            .insert((rows, retained), graph.clone());
        Ok(graph)
    }

    /// Queue the retention of `rows` (batch rows, in order) into consecutive
    /// rows of `store` from `first`.
    fn queue_retention(
        &self,
        window: RowWindow,
        hidden: &Tensor,
        rows: &[i32],
        store: &crate::FeatureRef,
        first: u64,
        submitter: &mut StepSubmitter,
    ) -> Result<(), SubmitError> {
        let store = store.allocation().tensor().map_err(device)?;
        for run in power_of_two_runs(0..rows.len() as u64) {
            let count = run.end - run.start;
            let graph = self.retain_graph(window.rows, count)?;
            let out_rows = rows[run.start as usize..run.end as usize]
                .iter()
                .map(|row| window.row(*row))
                .collect::<Result<Vec<_>, _>>()?;
            let out_rows =
                Tensor::from_host(&self.device, Element::i32(), &[count], &i32_bytes(&out_rows))
                    .map_err(device)?;
            let destination = store
                .slice_leading(first + run.start, first + run.end)
                .map_err(device)?;
            let mut bindings = graph.plan.bindings();
            bindings.set(&graph.hidden, hidden).map_err(device)?;
            bindings.set(&graph.norm, &self.readout_values.norm).map_err(device)?;
            bindings.set(&graph.out_rows, &out_rows).map_err(device)?;
            bindings.set(&graph.store, &destination).map_err(device)?;
            let mut slot = graph.plan.new_slot().map_err(device)?;
            let ready = slot
                .attach(bindings, graph.plan.new_outputs().map_err(device)?)
                .map_err(device)?;
            submitter.queue(ready)?;
        }
        Ok(())
    }

    /// The scores graph of `keys` key rows, a power of two.
    fn scores_graph(&self, keys: u64) -> Result<ScoresGraph, SubmitError> {
        if let Some(graph) = self.scores_graphs.borrow().get(&keys) {
            return Ok(graph.clone());
        }
        let kernel = self
            .readout_values
            .scores
            .as_ref()
            .ok_or_else(|| invalid("scores readouts reached a load that reads none"))?;
        let activation =
            Element::dense(crate::planning::activation_dtype(self.geometry.activation_dtype));
        let mut graph = self.device.native_graph();
        let query = graph
            .port(activation, &[1, self.geometry.hidden])
            .map_err(device)?;
        let keys_port = graph
            .port(activation, &[keys, self.geometry.hidden])
            .map_err(device)?;
        let scores = graph
            .enqueue(
                kernel,
                readout_scores_rows::WorkflowArgs {
                    query: query.tensor().into(),
                    keys: keys_port.tensor().into(),
                    scale: (self.geometry.hidden as f32).sqrt().recip(),
                },
            )
            .map_err(device)?
            .value;
        graph.export(&scores).map_err(device)?;
        let plan = graph.seal().map_err(device)?;
        if plan.upload_bytes() != 0 {
            return Err(invalid("scores unexpectedly upload host inputs"));
        }
        let graph = ScoresGraph {
            plan,
            query,
            keys: keys_port,
            scores,
        };
        self.scores_graphs.borrow_mut().insert(keys, graph.clone());
        Ok(graph)
    }

    /// Queue the scores of one query row against one run of key rows, both
    /// retained, as consecutive runs of the load's scores classes: their
    /// `[1, keys]` scores in key order.
    fn queue_scores(
        &self,
        query: &Tensor,
        keys: &Tensor,
        submitter: &mut StepSubmitter,
    ) -> Result<Vec<Tensor>, SubmitError> {
        let rows = keys
            .extents()
            .first()
            .copied()
            .ok_or_else(|| invalid("scores keys are not rows"))?;
        power_of_two_runs(0..rows)
            .map(|run| {
                let graph = self.scores_graph(run.end - run.start)?;
                let keys = keys.slice_leading(run.start, run.end).map_err(device)?;
                let mut bindings = graph.plan.bindings();
                bindings.set(&graph.query, query).map_err(device)?;
                bindings.set(&graph.keys, &keys).map_err(device)?;
                let mut slot = graph.plan.new_slot().map_err(device)?;
                let ready = slot
                    .attach(bindings, graph.plan.new_outputs().map_err(device)?)
                    .map_err(device)?;
                submitter
                    .queue(ready)?
                    .exported(&graph.scores)
                    .ok_or_else(|| invalid("scores were not exported"))
            })
            .collect()
    }

    /// The device bytes the readout values of a launch allocate beside its
    /// graphs' leases: what `queue_readout_values` creates for `slots`, each
    /// slot's row readouts in batch order. A launch reserves them from the
    /// heap before it is submitted.
    pub(crate) fn readout_values_bytes(
        &self,
        slots: &[Vec<magnitude_batching::RowReadout>],
    ) -> Result<u64, SubmitError> {
        use magnitude_batching::RowReadout;
        let activation =
            Element::dense(crate::planning::activation_dtype(self.geometry.activation_dtype));
        let features = |rows: u64| {
            activation
                .canonical_byte_len(&[rows, self.geometry.hidden])
                .map_err(|error| invalid(error.to_string()))
        };
        let mut groups = Vec::<LogitsGroup>::new();
        let mut heads = 0u64;
        let mut bytes = 0u64;
        for readouts in slots {
            let mut retained = BTreeMap::<u32, u64>::new();
            for readout in readouts {
                match readout {
                    RowReadout::Logits(tokens) => {
                        if !groups
                            .last()
                            .is_some_and(|group| group.admits(tokens, self.logits_budget))
                        {
                            groups.push(LogitsGroup::default());
                        }
                        groups.last_mut().expect("a group was opened").add(tokens);
                    }
                    RowReadout::Head => heads += 1,
                    RowReadout::Retain { store, .. } => *retained.entry(*store).or_default() += 1,
                    RowReadout::Scores { keys, .. } => {
                        self.readout_values
                            .scores
                            .as_ref()
                            .ok_or_else(|| invalid("scores readouts reached a load that reads none"))?;
                        for (_, rows) in keys {
                            for run in power_of_two_runs(u64::from(rows.start)..u64::from(rows.end)) {
                                bytes += 4 * (run.end - run.start);
                            }
                        }
                    }
                }
            }
            for rows in retained.into_values() {
                for run in power_of_two_runs(0..rows) {
                    let count = run.end - run.start;
                    bytes += 4 * count + features(count)?;
                }
            }
        }
        if !groups.is_empty() {
            self.readout_values
                .logits
                .as_ref()
                .ok_or_else(|| invalid("logits readouts reached a load that reads no logits"))?;
            for group in &groups {
                bytes += group.bytes();
            }
        }
        if heads != 0 {
            let head = self
                .readout_values
                .head
                .as_ref()
                .ok_or_else(|| invalid("head readouts reached a load with no readout head"))?;
            let outputs = heads.next_power_of_two();
            let values = head.weight.extents().first().copied().unwrap_or(0);
            bytes += 4 * (outputs + outputs * values);
        }
        Ok(bytes)
    }

    /// Queue the values of the step's readouts over `hidden`, the final rows
    /// of `window`: first the retention of the rows its sequences retain,
    /// one run per store, then its logits readouts in order in groups whose
    /// values fit the load's logits budget (`LogitsGroup`), each group's
    /// logits onto the union of its tokens; one readout head
    /// projection of every head readout's row; and each scores readout's
    /// scores per run of key rows, read from the stores.
    fn queue_readout_values(
        &self,
        batch: &TargetBatchUpload<'_>,
        state: &RowState<'_>,
        window: RowWindow,
        hidden: &Tensor,
        submitter: &mut StepSubmitter,
    ) -> Result<StepReadoutValues, SubmitError> {
        let mut groups = Vec::<(Vec<(i32, usize, Arc<[i32]>)>, LogitsGroup)>::new();
        let mut heads = Vec::<(i32, usize)>::new();
        // Per (slot, store), its rows retained this step: the first store
        // row and the batch rows, consecutive in the store.
        let mut retained = BTreeMap::<(usize, u32), (u32, Vec<i32>)>::new();
        let mut scores = Vec::new();
        for (index, (&output, readout)) in batch.readout_rows.iter().zip(batch.readouts).enumerate() {
            let output = usize::try_from(output).map_err(|_| invalid("negative readout output"))?;
            let row = *batch
                .out_rows
                .get(output)
                .ok_or_else(|| invalid("readout output exceeds output rows"))?;
            let slot = *usize::try_from(row)
                .ok()
                .and_then(|row| batch.row_slots.get(row))
                .ok_or_else(|| invalid("readout row exceeds the batch"))?;
            match readout {
                magnitude_batching::RowReadout::Logits(tokens) => {
                    if !groups
                        .last()
                        .is_some_and(|(_, group)| group.admits(tokens, self.logits_budget))
                    {
                        groups.push(Default::default());
                    }
                    let (rows, group) = groups.last_mut().expect("a group was opened");
                    rows.push((row, index, tokens.clone()));
                    group.add(tokens);
                }
                magnitude_batching::RowReadout::Head => heads.push((row, index)),
                magnitude_batching::RowReadout::Retain { store, index } => {
                    let slot = usize::try_from(slot).map_err(|_| invalid("negative slot"))?;
                    let (first, rows) = retained.entry((slot, *store)).or_insert((*index, Vec::new()));
                    if *first + rows.len() as u32 != *index {
                        return Err(invalid("a step's retained rows are not consecutive in their store"));
                    }
                    rows.push(row);
                }
                magnitude_batching::RowReadout::Scores { query, keys } => {
                    let slot = usize::try_from(slot).map_err(|_| invalid("negative slot"))?;
                    scores.push((index, slot, *query, keys.clone()));
                }
            }
        }
        let mut values = StepReadoutValues::default();
        for ((slot, store), (first, rows)) in &retained {
            let store = state
                .stores(*slot)
                .get(*store as usize)
                .ok_or_else(|| invalid("a retained row names no store of its slot"))?;
            self.queue_retention(window, hidden, rows, store, u64::from(*first), submitter)?;
        }
        for (readout, slot, (query_store, query_row), keys) in scores {
            let stores = state.stores(slot);
            let store = |store: u32| {
                stores
                    .get(store as usize)
                    .ok_or_else(|| invalid("a scores readout names no store of its slot"))?
                    .allocation()
                    .tensor()
                    .cloned()
                    .map_err(device)
            };
            let query = store(query_store)?
                .slice_leading(u64::from(query_row), u64::from(query_row) + 1)
                .map_err(device)?;
            let runs = keys
                .iter()
                .map(|(key_store, rows)| {
                    let keys = store(*key_store)?
                        .slice_leading(u64::from(rows.start), u64::from(rows.end))
                        .map_err(device)?;
                    self.queue_scores(&query, &keys, submitter)
                })
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .flatten()
                .collect();
            values.scores.push(ScoresReadoutValues { readout, runs });
        }
        for (rows, LogitsGroup { columns, .. }) in groups {
            let weights = self
                .readout_values
                .logits
                .as_ref()
                .ok_or_else(|| invalid("logits readouts reached a load that reads no logits"))?;
            let columns = columns.into_iter().collect::<Vec<_>>();
            let column_class = (columns.len() as u64).next_power_of_two();
            let (outputs, out_rows) = self.output_rows(window, rows.iter().map(|row| row.0))?;
            let graph = self.values_graph(window.rows, outputs, column_class)?;
            let mut selected = vec![0_i32; column_class as usize];
            selected[..columns.len()].copy_from_slice(&columns);
            let selected = Tensor::from_host(
                &self.device,
                Element::i32(),
                &[column_class],
                &i32_bytes(&selected),
            )
            .map_err(device)?;
            let mut bindings = graph.plan.bindings();
            bindings.set(&graph.hidden, hidden).map_err(device)?;
            bindings.set(&graph.norm, &self.readout_values.norm).map_err(device)?;
            bindings.set(&graph.weight, &weights.weight).map_err(device)?;
            bindings.set(&graph.out_rows, &out_rows).map_err(device)?;
            bindings.set(&graph.selected, &selected).map_err(device)?;
            let mut slot = graph.plan.new_slot().map_err(device)?;
            let ready = slot
                .attach(bindings, graph.plan.new_outputs().map_err(device)?)
                .map_err(device)?;
            let outputs = submitter.queue(ready)?;
            values.logits.push(ReadoutValueGroup {
                values: outputs
                    .exported(&graph.values)
                    .ok_or_else(|| invalid("readout values were not exported"))?,
                columns,
                rows: rows
                    .into_iter()
                    .map(|(_, index, tokens)| (index, tokens))
                    .collect(),
            });
        }
        if !heads.is_empty() {
            let weights = self
                .readout_values
                .head
                .as_ref()
                .ok_or_else(|| invalid("head readouts reached a model with no readout head"))?;
            let (outputs, out_rows) = self.output_rows(window, heads.iter().map(|head| head.0))?;
            let graph = self.head_graph(window.rows, outputs)?;
            let mut bindings = graph.plan.bindings();
            bindings.set(&graph.hidden, hidden).map_err(device)?;
            bindings.set(&graph.norm, &self.readout_values.norm).map_err(device)?;
            bindings.set(&graph.weight, &weights.weight).map_err(device)?;
            bindings.set(&graph.out_rows, &out_rows).map_err(device)?;
            bindings.set(&graph.scale, &weights.unscaled).map_err(device)?;
            let mut slot = graph.plan.new_slot().map_err(device)?;
            let ready = slot
                .attach(bindings, graph.plan.new_outputs().map_err(device)?)
                .map_err(device)?;
            let outputs = submitter.queue(ready)?;
            values.head = Some(HeadReadoutValues {
                values: outputs
                    .exported(&graph.values)
                    .ok_or_else(|| invalid("readout head values were not exported"))?,
                element: weights.element,
                readouts: heads.into_iter().map(|(_, index)| index).collect(),
            });
        }
        Ok(values)
    }

    /// The power-of-two class of `rows`' count, and those batch rows as
    /// `window`'s output rows on the device, the class's padding reading the
    /// window's first row.
    fn output_rows(
        &self,
        window: RowWindow,
        rows: impl ExactSizeIterator<Item = i32>,
    ) -> Result<(u64, Tensor), SubmitError> {
        let outputs = (rows.len() as u64).next_power_of_two();
        let mut out_rows = vec![0_i32; outputs as usize];
        for (index, row) in rows.enumerate() {
            out_rows[index] = window.row(row)?;
        }
        let out_rows = Tensor::from_host(&self.device, Element::i32(), &[outputs], &i32_bytes(&out_rows))
            .map_err(device)?;
        Ok((outputs, out_rows))
    }

    /// The leading `rows` draft input rows of a tapped target.
    fn tap_rows(&self, rows: u64) -> Result<Tensor, SubmitError> {
        self.graphs
            .taps
            .as_ref()
            .ok_or_else(|| invalid("a tapped graph has no bound draft input rows"))?
            .slice_leading(0, rows)
            .map_err(device)
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
        let plan = graph.seal().map_err(device)?;
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

    /// A copy of `rows` F32 rows of `width` columns: conditioning rows into
    /// embedded rows, or a layer-major chunk's rows into its held rows.
    fn overlay(&self, rows: u64, width: u64) -> Result<OverlayGraph, SubmitError> {
        if let Some(overlay) = self.overlays.borrow().get(&(rows, width)) {
            return Ok(overlay.clone());
        }
        let kernel = self
            .state
            .conditioning
            .as_ref()
            .ok_or_else(|| invalid("conditioning program slot is absent"))?;
        let mut graph = self.device.native_graph();
        let source = graph
            .port(Element::f32(), &[rows, width])
            .map_err(device)?;
        let mut destination = graph
            .port(Element::f32(), &[rows, width])
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
        let plan = graph.seal().map_err(device)?;
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
        self.overlays.borrow_mut().insert((rows, width), overlay.clone());
        Ok(overlay)
    }

    /// Queue a per-layer model's per-layer entry over the step's embedded
    /// and overlaid `hidden` rows, and return the per-layer rows its blocks
    /// read. Each batch row gathers its token's table row, a media row the
    /// entry's media row.
    fn queue_per_layer_entry(
        &self,
        batch: &TargetBatchUpload<'_>,
        tokens: &TargetTokens,
        rows: u64,
        hidden: &Tensor,
        media_rows: &[std::ops::Range<usize>],
        graph_workspace: &mut TargetGraphWorkspaceLease,
        submitter: &mut StepSubmitter,
    ) -> Result<Option<Tensor>, SubmitError> {
        let Some(entry) = self.graphs.per_layer_entry(rows) else {
            return Ok(None);
        };
        let (graph, bound, per_layer) = entry.map_err(invalid)?;
        let TargetTokens::Host = tokens else {
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
        let mut inputs = per_layer.rows.slice_leading(0, rows).map_err(device)?;
        // The token whose per-layer row this step writes into a token
        // row's slot, and the slot.
        let mut written = None;
        {
            // The token rows' slots are the per-layer rows after the first,
            // the last row the first slot. This step's entry writes its
            // leading rows.
            let last = per_layer.rows.extents()[0] - 1;
            per_layer
                .token_rows
                .overwrite((last + 1 - rows) as usize..last as usize);
            if let ([token], true) = (batch.tokens, media_rows.is_empty()) {
                let slot = |slot: usize| {
                    let row = last - slot as u64;
                    per_layer.rows.slice_leading(row, row + 1).map_err(device)
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
        let mut active = graph_workspace
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
        submitter.queue(ready)?;
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

    /// The leading `rows` of the token rows a model whose blocks read the
    /// step's token rows keeps (`BoundTargetGraphs::token_rows`).
    fn kept_token_rows(&self, rows: u64) -> Result<Tensor, SubmitError> {
        self.graphs
            .token_rows
            .as_ref()
            .ok_or_else(|| invalid("the model keeps no token rows"))?
            .slice_leading(0, rows)
            .map_err(device)
    }

    /// Conditioning rows that replace embedding rows, as (source, first
    /// batch row, end batch row), of a batch whose slots compute their
    /// advances' rows from `first`: 0 for whole advances, a layer-major
    /// chunk's first row for its one slot. Most steps have none.
    fn conditioning_sources(
        &self,
        state: &RowState<'_>,
        batch: &TargetBatchUpload<'_>,
        first: usize,
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
            let slices = ConditioningSlice::within(slices, &(first..first + (end - start)));
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
            for slice in &slices {
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

    /// Queue every run of one step. On failure, runs already queued are
    /// observed before the caller regains the storage they use.
    fn queue_rows(
        &self,
        rows: &LaunchRows,
        state: &RowState<'_>,
        readout: (&mut NativeGraphWorkspaceLease, NativeGraphOutputLease),
        graph_workspace: &mut TargetGraphWorkspaceLease,
        graph_outputs: &mut [TargetGraphOutputLease; 2],
    ) -> Result<QueuedStep, SubmitError> {
        let mut submitter = StepSubmitter::new(&self.device);
        let queued = match rows {
            LaunchRows::Batch(batch) => self
                .queue_batch(
                    &batch.upload(),
                    state,
                    readout,
                    graph_workspace,
                    graph_outputs,
                    &mut submitter,
                )
                .map(|(readout, values, commits)| (readout, vec![values], commits)),
            LaunchRows::LayerMajor(layered) => self
                .queue_layer_major(layered, state, graph_workspace, graph_outputs, &mut submitter)
                .map(|(values, commits)| (None, values, commits)),
        }
        .and_then(|step| submitter.flush().map(|()| step));
        match queued {
            Ok((readout, readout_values, commits)) => Ok(QueuedStep {
                readout,
                readout_values,
                commits,
                completions: submitter.completions,
                landed_reads: submitter.landed,
                stored_reads: submitter.reads,
            }),
            Err(error) => {
                // The failure is reported; outcomes of runs submitted before
                // it matter only for draining the device. Runs still queued
                // are dropped unsubmitted, and their stored-table steps
                // read nothing.
                let _ = wait_all(std::mem::take(&mut submitter.completions));
                submitter.abandon_unsubmitted();
                Err(error)
            }
        }
    }

    /// Queue a step's entry over `batch`'s `rows` (a row class), whose
    /// slots compute their advances' rows from `first`
    /// (`conditioning_sources`), into `output`: its embedded rows,
    /// conditioning overlaid, and a per-layer model's per-layer rows.
    fn queue_entry(
        &self,
        batch: &TargetBatchUpload<'_>,
        first: usize,
        state: &RowState<'_>,
        rows: u64,
        graph_workspace: &mut TargetGraphWorkspaceLease,
        output: &mut TargetGraphOutputLease,
        submitter: &mut StepSubmitter,
    ) -> Result<Entered, SubmitError> {
        let tokens = state.tokens();
        let source = match tokens {
            TargetTokens::Host => EntryTokens::Uploaded,
            TargetTokens::Selected(_) => EntryTokens::Selected,
        };
        let (entry, entry_bound) = self.graphs.entry(rows, source).map_err(invalid)?;
        // A step reading selected tokens in a model with a stored table
        // copies them for the host first (`EntryNotice`).
        let mut notice = None;
        submitter.await_upload(graph_workspace.slot_mut())?;
        let outputs = {
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
                            .set(&kept.rows, &self.kept_token_rows(batch.tokens.len() as u64)?)
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
            let ready = active
                .attach(
                    bindings,
                    output.activate(&entry.plan).map_err(SubmitError::Invariant)?,
                )
                .map_err(device)?;
            submitter.queue(ready)?
        };
        let queued = Instant::now();
        let hidden = outputs
            .exported(&entry.hidden)
            .ok_or_else(|| invalid("target embedding hidden was not exported"))?;
        let conditioning = self.conditioning_sources(state, batch, first)?;
        let media_rows = conditioning
            .iter()
            .map(|(_, start, end)| *start..*end)
            .collect::<Vec<_>>();
        for (source, start, end) in conditioning {
            let overlay = self.overlay((end - start) as u64, self.geometry.hidden)?;
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
            submitter.queue(ready)?;
        }
        let per_layer = self.queue_per_layer_entry(
            batch,
            tokens,
            rows,
            &hidden,
            &media_rows,
            graph_workspace,
            submitter,
        )?;
        Ok(Entered {
            outputs,
            queued,
            hidden,
            per_layer,
            notice,
        })
    }

    /// Queue block `index` over `hidden`, writing its outputs to `output`.
    fn queue_block(
        &self,
        index: usize,
        step: &BlockInputs<'_>,
        hidden: &Tensor,
        graph_workspace: &mut TargetGraphWorkspaceLease,
        output: &mut TargetGraphOutputLease,
        submitter: &mut StepSubmitter,
    ) -> Result<Blocked, SubmitError> {
        let BlockInputs {
            batch,
            state,
            controls,
            rows,
            segments,
            slots,
            per_layer,
            notice,
        } = *step;
        let block_started = self.trace.blocks.then(Instant::now);
        // An attention block's history controls: its read's.
        let history_controls = match state.history(LayerRef::Target(index as u32)) {
            Ok((read, _)) => Some(
                controls
                    .histories
                    .get(read)
                    .ok_or_else(|| invalid("the batch lacks the layer's history read"))?,
            ),
            Err(_) => None,
        };
        // A block with listing classes lists the history row tiles the
        // launch's rows see in the smallest class that holds them; a
        // launch that sees more than the largest (many requests with
        // long histories) takes the class that lists none.
        let listed_tiles = self
            .graphs
            .prepared
            .listed_tiles(index, rows)
            .iter()
            .find_map(|&tiles| {
                let visible = &history_controls.as_ref()?.visible;
                crate::operators::attention::graph::history_tile_bytes(
                    visible,
                    usize::try_from(tiles).ok()?,
                )
                .map(|bytes| (tiles, bytes))
            });
        // A listing class may be sealed for more segments than the
        // launch has: its rows' spans are then padded with empty ones.
        let block_segments = match &listed_tiles {
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
        let (graph, bound) = self
            .graphs
            .block(
                rows,
                block_segments,
                slots,
                listed_tiles.as_ref().map_or(0, |(tiles, _)| *tiles),
                scored,
                false,
                index,
            )
            .map_err(invalid)?;
        submitter.await_upload(graph_workspace.slot_mut())?;
        let mut active = graph_workspace
            .slot_mut()
            .activate(&graph.plan)
            .map_err(device)?;
        let tap_rows = graph
            .taps
            .as_ref()
            .map(|_| self.tap_rows(rows))
            .transpose()?;
        let mut bindings = bound.bindings();
        bindings.set(&graph.hidden, hidden).map_err(device)?;
        match (&graph.per_layer, per_layer) {
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
                    .set(&ports.tokens, &self.kept_token_rows(rows)?)
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
            let rows = self.stored_step_rows(index, batch, state, notice)?;
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
            for (port, tap) in [
                (&ports.entry, taps.entry),
                (&ports.middle, taps.middle),
                (&ports.output, taps.output),
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
        match (&graph.state, &graph.controls) {
            (
                BlockStatePorts::Attention(ports, indexer_planes),
                BlockControlPorts::Attention {
                    coordinates,
                    visible,
                    fresh,
                    destinations,
                    history_tiles,
                    indexer,
                },
            ) => {
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
                            let history = controls.histories.get(read).ok_or_else(|| {
                                invalid("the batch lacks an indexer's history read")
                            })?;
                            active
                                .write_input(visible, &history.visible)
                                .map_err(device)?;
                            active
                                .write_input(destinations, &history.destinations)
                                .map_err(device)?;
                        }
                    }
                    (None, None) => {}
                    _ => return Err(invalid("an indexer's planes and tables disagree")),
                }
                let (_, history) = state.history(LayerRef::Target(index as u32))?;
                if history.len() != ports.len() {
                    return Err(invalid(
                        "the layer's history planes disagree with its attention entry",
                    ));
                }
                for (port, plane) in ports.iter().zip(history) {
                    bindings.set(port, plane).map_err(device)?;
                }
                let history_controls = history_controls
                    .as_ref()
                    .ok_or_else(|| invalid("the batch lacks the layer's history read"))?;
                active
                    .write_input(coordinates, &controls.coordinates)
                    .map_err(device)?;
                match block_segments == segments {
                    true => active.write_input(visible, &history_controls.visible),
                    false => active.write_input(
                        visible,
                        &padded_spans(&history_controls.visible, segments, block_segments),
                    ),
                }
                .map_err(device)?;
                active
                    .write_input(fresh, &history_controls.fresh)
                    .map_err(device)?;
                active
                    .write_input(destinations, &history_controls.destinations)
                    .map_err(device)?;
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
            (BlockStatePorts::Recurrent(ports), BlockControlPorts::Recurrent(recurrent)) => {
                let arenas = state.recurrent_arenas()?;
                let first = crate::operators::bank_component_index(&self.geometry.blocks, index)
                    .map_err(|error| invalid(error.to_string()))?;
                for (offset, port) in ports.iter().enumerate() {
                    bindings
                        .set(
                            port,
                            arenas
                                .get(first + offset)
                                .ok_or_else(|| invalid("recurrent bank arena is absent"))?,
                        )
                        .map_err(device)?;
                }
                active
                    .write_input(&recurrent.segments, &controls.segments)
                    .map_err(device)?;
                active
                    .write_input(&recurrent.stop, &controls.stop)
                    .map_err(device)?;
                active
                    .write_input(&recurrent.previous_bank, &controls.previous_bank)
                    .map_err(device)?;
                active
                    .write_input(&recurrent.previous_tape, &controls.previous_tape)
                    .map_err(device)?;
                active
                    .write_input(&recurrent.following_bank, &controls.following_bank)
                    .map_err(device)?;
            }
            _ => return Err(invalid("sealed graph block state and controls disagree")),
        }
        let ready = active
            .attach(
                bindings,
                output.activate(&graph.plan).map_err(SubmitError::Invariant)?,
            )
            .map_err(device)?;
        let outputs = submitter.queue(ready)?;
        let residual = graph
            .residual
            .as_ref()
            .map(|edge| {
                outputs
                    .exported(edge)
                    .ok_or_else(|| invalid("target residual was not exported"))
            })
            .transpose()?;
        let hidden = graph
            .output
            .as_ref()
            .and_then(|edge| outputs.exported(edge))
            .ok_or_else(|| invalid("target block output was not exported"))?;
        if let Some(started) = block_started {
            let kind = match &graph.state {
                BlockStatePorts::Attention(..) => "attention",
                BlockStatePorts::Recurrent(_) => "recurrent",
            };
            eprintln!(
                "target block {index} {kind} rows {rows} lists {} history tiles {:.3}s",
                listed_tiles.as_ref().map_or(0, |(tiles, _)| *tiles),
                started.elapsed().as_secs_f64()
            );
        }
        Ok(Blocked {
            outputs,
            hidden,
            residual,
        })
    }

    /// Queue one batch's step: its entry, every block, and its readout.
    fn queue_batch(
        &self,
        batch: &TargetBatchUpload<'_>,
        state: &RowState<'_>,
        readout: (&mut NativeGraphWorkspaceLease, NativeGraphOutputLease),
        graph_workspace: &mut TargetGraphWorkspaceLease,
        graph_outputs: &mut [TargetGraphOutputLease; 2],
        submitter: &mut StepSubmitter,
    ) -> Result<(Option<TargetReadoutGraphResult>, StepReadoutValues, CommitSpan), SubmitError> {
        let rows =
            u64::try_from(batch.class.rows()).map_err(|_| invalid("row class exceeds u64"))?;
        let mut residuals = Residuals::new(graph_outputs);
        let entered =
            self.queue_entry(batch, 0, state, rows, graph_workspace, residuals.lease(), submitter)?;
        let Entered {
            outputs,
            queued: first_commit,
            mut hidden,
            mut per_layer,
            notice,
        } = entered;
        residuals.wrote(outputs)?;
        // The device embeds while the host prepares the remaining runs.
        let segments = u64::try_from(batch.class.segments())
            .map_err(|_| invalid("segment class exceeds u64"))?;
        let slots = u64::try_from(batch.actual_slots)
            .map_err(|_| invalid("request slot class exceeds u64"))?;
        let mut controls = GraphControls::new(batch)?;
        let mut window = RowWindow::whole(rows);
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
        // Blocks whose outputs the step reads; the last one's output feeds
        // the readout.
        let mut residual = None;
        for index in 0..self.geometry.blocks.len() {
            if index == self.graphs.prepared.stateless_tail() {
                // From here no block publishes state: run the rest over the
                // rows the readout consumes, or not at all without any.
                match RowWindow::consumed(batch.out_rows, rows)? {
                    None => break,
                    Some(tail) if tail.rows < rows => {
                        hidden = tail.slice(&hidden)?;
                        per_layer = per_layer.map(|inputs| tail.slice(&inputs)).transpose()?;
                        controls = controls.window(rows, tail);
                        window = tail;
                    }
                    Some(_) => {}
                }
            }
            let step = BlockInputs {
                batch,
                state,
                controls: &controls,
                rows: window.rows,
                segments,
                slots,
                per_layer: per_layer.as_ref(),
                notice: notice.as_ref(),
            };
            let block = self.queue_block(
                index,
                &step,
                &hidden,
                graph_workspace,
                residuals.lease(),
                submitter,
            )?;
            residual = block.residual;
            hidden = block.hidden;
            residuals.wrote(block.outputs)?;
        }
        // Only rows reading more than their readout values read the graph
        // readout.
        let read = batch.out_rows.iter().try_fold(false, |read, &row| {
            let demand = usize::try_from(row)
                .ok()
                .and_then(|row| batch.demand.get(row))
                .and_then(|bits| Demand::from_bits(*bits))
                .ok_or_else(|| invalid("readout demand is absent or invalid"))?;
            Ok::<_, SubmitError>(read || demand != demand & Demand::READOUT)
        })?;
        let (readout_workspace, readout_output) = readout;
        let result = if read {
            self.graph_readout(
                batch,
                window,
                &hidden,
                residual.as_ref(),
                readout_workspace,
                readout_output,
                submitter,
            )?
        } else {
            None
        };
        let readout_values = if batch.readout_rows.is_empty() {
            StepReadoutValues::default()
        } else {
            self.queue_readout_values(batch, state, window, &hidden, submitter)?
        };
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
            submitter.queue(ready)?;
        }
        let last_commit = Instant::now();
        // Every consumer of the final hidden rows is queued, so its output
        // slot returns to the lease; the device queue orders any reuse.
        drop(hidden);
        drop(residual);
        residuals.finish()?;
        Ok((
            result,
            readout_values,
            CommitSpan {
                first: first_commit,
                last: last_commit,
            },
        ))
    }

    /// Queue a publishing pass of attention block `index` over `hidden`
    /// (`queue_layer_major`): only the keys and values its rows append at
    /// their destinations in the block's history, outputting nothing.
    fn queue_keys_values(
        &self,
        index: usize,
        step: &BlockInputs<'_>,
        hidden: &Tensor,
        graph_workspace: &mut TargetGraphWorkspaceLease,
        output: &mut TargetGraphOutputLease,
        submitter: &mut StepSubmitter,
    ) -> Result<NativeGraphOutputs, SubmitError> {
        let BlockInputs {
            state,
            controls,
            rows,
            ..
        } = *step;
        let (read, history) = state.history(LayerRef::Target(index as u32))?;
        let history_controls = controls
            .histories
            .get(read)
            .ok_or_else(|| invalid("the batch lacks the layer's history read"))?;
        let (graph, bound) = self
            .graphs
            .block(rows, 1, 1, 0, false, true, index)
            .map_err(invalid)?;
        let (
            BlockStatePorts::Attention(ports, None),
            BlockControlPorts::Append {
                coordinates,
                destinations,
            },
        ) = (&graph.state, &graph.controls)
        else {
            return Err(invalid("a keys-and-values graph appends to one history"));
        };
        if history.len() != ports.len() {
            return Err(invalid(
                "the layer's history planes disagree with its append entry",
            ));
        }
        submitter.await_upload(graph_workspace.slot_mut())?;
        let mut active = graph_workspace
            .slot_mut()
            .activate(&graph.plan)
            .map_err(device)?;
        let mut bindings = bound.bindings();
        bindings.set(&graph.hidden, hidden).map_err(device)?;
        for (port, plane) in ports.iter().zip(history) {
            bindings.set(port, plane).map_err(device)?;
        }
        active
            .write_input(coordinates, &controls.coordinates)
            .map_err(device)?;
        active
            .write_input(destinations, &history_controls.destinations)
            .map_err(device)?;
        let ready = active
            .attach(
                bindings,
                output
                    .activate(&graph.plan)
                    .map_err(SubmitError::Invariant)?,
            )
            .map_err(device)?;
        submitter.queue(ready)
    }

    /// Queue a layer-major step (`LayerMajor`): for each stretch of blocks
    /// before a barrier, the last after one, every chunk in turn runs the
    /// stretch from its entry or its held rows, in its own class; before a
    /// barrier it holds its rows and publishes to the barrier's history, so
    /// the next stretch's barrier reads every chunk's rows. Each chunk reads
    /// its readout values after the last stretch.
    fn queue_layer_major(
        &self,
        layered: &LayerMajor,
        state: &RowState<'_>,
        graph_workspace: &mut TargetGraphWorkspaceLease,
        graph_outputs: &mut [TargetGraphOutputLease; 2],
        submitter: &mut StepSubmitter,
    ) -> Result<(Vec<StepReadoutValues>, CommitSpan), SubmitError> {
        let uploads = layered
            .chunks
            .iter()
            .map(|chunk| chunk.batch.upload())
            .collect::<Vec<_>>();
        let controls = uploads
            .iter()
            .map(GraphControls::new)
            .collect::<Result<Vec<_>, _>>()?;
        let class_rows = |upload: &TargetBatchUpload<'_>| {
            u64::try_from(upload.class.rows()).map_err(|_| invalid("row class exceeds u64"))
        };
        // Held rows are laid out a chunk per widest (first) chunk's class
        // rows.
        let stride = class_rows(&uploads[0])?;
        let segments = u64::try_from(uploads[0].class.segments())
            .map_err(|_| invalid("segment class exceeds u64"))?;
        let chunks = uploads.len() as u64;
        let barriers = layered
            .barriers
            .as_ref()
            .map_or(&[][..], |barriers| barriers.blocks.as_slice());
        // Chunk `chunk`'s `rows` held rows of the barrier `block`'s input
        // width.
        let held = |chunk: usize, rows: u64, block: usize| -> Result<Tensor, SubmitError> {
            let width =
                crate::programs::native_target_graph::block_input_width(&self.geometry, block)
                    .map_err(invalid)?;
            let barriers = layered
                .barriers
                .as_ref()
                .ok_or_else(|| invalid("a layer-major step without barriers holds no rows"))?;
            let first = chunk as u64 * stride;
            barriers
                .held
                .slice_leading(0, chunks * stride * width)
                .and_then(|flat| flat.reshape(&[chunks * stride, width]))
                .and_then(|held| held.slice_leading(first, first + rows))
                .map_err(device)
        };
        let blocks = self.geometry.blocks.len();
        let stretches = std::iter::once(0)
            .chain(barriers.iter().copied())
            .zip(barriers.iter().copied().chain(std::iter::once(blocks)))
            .collect::<Vec<_>>();
        let mut first_commit = None;
        let mut values = Vec::new();
        for (stretch, &(start, end)) in stretches.iter().enumerate() {
            let last = stretch + 1 == stretches.len();
            for (chunk, (batch, chunk_controls)) in uploads.iter().zip(&controls).enumerate() {
                let rows = class_rows(batch)?;
                let mut residuals = Residuals::new(graph_outputs);
                let mut hidden = if stretch == 0 {
                    let entered = self.queue_entry(
                        batch,
                        layered.chunks[chunk].rows.start,
                        state,
                        rows,
                        graph_workspace,
                        residuals.lease(),
                        submitter,
                    )?;
                    first_commit.get_or_insert(entered.queued);
                    residuals.wrote(entered.outputs)?;
                    entered.hidden
                } else {
                    held(chunk, rows, start)?
                };
                let mut controls = std::borrow::Cow::Borrowed(chunk_controls);
                let mut window = RowWindow::whole(rows);
                for index in start..end {
                    if index == self.graphs.prepared.stateless_tail() {
                        match RowWindow::consumed(batch.out_rows, rows)? {
                            None => break,
                            Some(tail) if tail.rows < rows => {
                                hidden = tail.slice(&hidden)?;
                                controls =
                                    std::borrow::Cow::Owned(controls.into_owned().window(rows, tail));
                                window = tail;
                            }
                            Some(_) => {}
                        }
                    }
                    let step = BlockInputs {
                        batch,
                        state,
                        controls: &controls,
                        rows: window.rows,
                        segments,
                        slots: 1,
                        per_layer: None,
                        notice: None,
                    };
                    let block = self.queue_block(
                        index,
                        &step,
                        &hidden,
                        graph_workspace,
                        residuals.lease(),
                        submitter,
                    )?;
                    let Blocked {
                        outputs,
                        hidden: rows_out,
                        residual,
                    } = block;
                    drop(residual);
                    hidden = rows_out;
                    residuals.wrote(outputs)?;
                }
                if last {
                    values.push(if batch.readout_rows.is_empty() {
                        StepReadoutValues::default()
                    } else {
                        self.queue_readout_values(batch, state, window, &hidden, submitter)?
                    });
                    drop(hidden);
                    residuals.finish()?;
                    continue;
                }
                // Hold the stretch's rows, then publish them to the
                // barrier's history.
                let destination = held(chunk, rows, end)?;
                let width = destination.extents()[1];
                let copy = self.overlay(rows, width)?;
                let mut bindings = copy.plan.bindings();
                bindings.set(&copy.source, &hidden).map_err(device)?;
                bindings.set(&copy.destination, &destination).map_err(device)?;
                let mut slot = copy.plan.new_slot().map_err(device)?;
                let ready = slot
                    .attach(bindings, copy.plan.new_outputs().map_err(device)?)
                    .map_err(device)?;
                submitter.queue(ready)?;
                drop(hidden);
                residuals.finish()?;
                let mut residuals = Residuals::new(graph_outputs);
                let step = BlockInputs {
                    batch,
                    state,
                    controls: chunk_controls,
                    rows,
                    segments,
                    slots: 1,
                    per_layer: None,
                    notice: None,
                };
                let outputs = self.queue_keys_values(
                    end,
                    &step,
                    &destination,
                    graph_workspace,
                    residuals.lease(),
                    submitter,
                )?;
                residuals.wrote(outputs)?;
                residuals.finish()?;
            }
        }
        Ok((
            values,
            CommitSpan {
                first: first_commit.ok_or_else(|| invalid("a layer-major step queued no entry"))?,
                last: Instant::now(),
            },
        ))
    }
}

/// A step's entry: its outputs, when it was queued, its embedded rows, a
/// per-layer model's per-layer rows, and a selected-token step's notice.
struct Entered {
    outputs: NativeGraphOutputs,
    queued: Instant,
    hidden: Tensor,
    per_layer: Option<Tensor>,
    notice: Option<Arc<NoticeStep>>,
}

/// What every block of a pass over one batch reads beside its hidden rows:
/// `rows` rows (a row class) of controls, `segments` history spans a row
/// and `slots` request slots.
#[derive(Clone, Copy)]
struct BlockInputs<'a> {
    batch: &'a TargetBatchUpload<'a>,
    state: &'a RowState<'a>,
    controls: &'a GraphControls,
    rows: u64,
    segments: u64,
    slots: u64,
    per_layer: Option<&'a Tensor>,
    notice: Option<&'a Arc<NoticeStep>>,
}

/// A queued block's outputs and their exported rows.
struct Blocked {
    outputs: NativeGraphOutputs,
    hidden: Tensor,
    residual: Option<Tensor>,
}

/// The residual pair a pass of blocks alternates: each block reads the rows
/// the run before it wrote to one lease and writes the other's. An output
/// returns to its lease once every view of it is released and the run
/// reading it is queued; the device queue orders its reuse.
struct Residuals<'a> {
    leases: &'a mut [TargetGraphOutputLease; 2],
    pending: [Option<NativeGraphOutputs>; 2],
    next: usize,
}

impl<'a> Residuals<'a> {
    fn new(leases: &'a mut [TargetGraphOutputLease; 2]) -> Self {
        Self {
            leases,
            pending: [None, None],
            next: 0,
        }
    }

    /// The lease the next run writes.
    fn lease(&mut self) -> &mut TargetGraphOutputLease {
        &mut self.leases[self.next]
    }

    /// The next run, queued, wrote `outputs`: the rows it read return.
    fn wrote(&mut self, outputs: NativeGraphOutputs) -> Result<(), SubmitError> {
        let read = 1 - self.next;
        if let Some(previous) = self.pending[read].take() {
            self.leases[read]
                .recycle(previous)
                .map_err(SubmitError::Invariant)?;
        }
        self.pending[self.next] = Some(outputs);
        self.next = read;
        Ok(())
    }

    /// Every reader of the pass's outputs is queued: they return.
    fn finish(mut self) -> Result<(), SubmitError> {
        for (lease, pending) in self.leases.iter_mut().zip(&mut self.pending) {
            if let Some(outputs) = pending.take() {
                lease.recycle(outputs).map_err(SubmitError::Invariant)?;
            }
        }
        Ok(())
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

#[derive(Clone)]
struct GraphControls {
    coordinates: Vec<u8>,
    /// One control set per history domain, in domain order: an attention
    /// block reads its own domain's.
    histories: Vec<HistoryControls>,
    segments: Vec<u8>,
    /// Rows after which each slot publishes its recurrent state; later rows
    /// are recorded on the successor bank's tape.
    stop: Vec<u8>,
    /// Each slot's read version: bank, and the tape rows that complete it.
    previous_bank: Vec<u8>,
    previous_tape: Vec<u8>,
    following_bank: Vec<u8>,
}

/// One history domain's attention controls.
#[derive(Clone)]
struct HistoryControls {
    visible: Vec<u8>,
    fresh: Vec<u8>,
    destinations: Vec<u8>,
}

impl GraphControls {
    /// The per-row controls of `window`'s rows, of controls over
    /// `batch_rows` rows. The per-slot recurrent controls stay whole: only
    /// attention blocks run a window (`stateless_tail`).
    fn window(self, batch_rows: u64, window: RowWindow) -> Self {
        Self {
            coordinates: window.rows_of(&self.coordinates, batch_rows),
            histories: self
                .histories
                .iter()
                .map(|history| HistoryControls {
                    visible: window.rows_of(&history.visible, batch_rows),
                    fresh: window.rows_of(&history.fresh, batch_rows),
                    destinations: window.rows_of(&history.destinations, batch_rows),
                })
                .collect(),
            ..self
        }
    }

    fn new(batch: &TargetBatchUpload<'_>) -> Result<Self, SubmitError> {
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
        Ok(Self {
            coordinates: batch
                .coordinates
                .iter()
                .flatten()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
            histories: batch
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
                    fresh: history
                        .fresh
                        .iter()
                        .flatten()
                        .flat_map(|value| value.to_le_bytes())
                        .collect(),
                    destinations: i32_bytes(&history.destinations),
                })
                .collect(),
            segments: batch.segments[..=batch.actual_slots]
                .iter()
                .flatten()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
            stop: i32_bytes(&batch.stop[..batch.actual_slots]),
            previous_bank: i32_bytes(&batch.bank[..batch.actual_slots]),
            previous_tape: i32_bytes(&batch.previous_tape[..batch.actual_slots]),
            following_bank: i32_bytes(&batch.following_bank[..batch.actual_slots]),
        })
    }
}

impl TargetProgram for NativeTargetProgram {
    type Submission = DeviceSubmission<TargetLaunchCore, TargetLaunchWorkspace, TargetOutput>;

    fn submit(
        &mut self,
        mut launch: ValidatedTargetLaunch,
    ) -> Result<Self::Submission, (SubmitError, ValidatedTargetLaunch)> {
        let queued = {
            let (core, graph_workspace, graph_outputs, readout_workspace, readout_output) =
                launch.submission_parts_mut();
            self.queue_rows(
                core.rows(),
                &RowState(core),
                (
                    readout_workspace,
                    readout_output
                        .take()
                        .expect("readout output was reserved before submit"),
                ),
                graph_workspace,
                graph_outputs,
            )
        };
        let queued = match queued {
            Ok(queued) => queued,
            Err(error) => return Err((error, launch)),
        };
        let (core, workspace) = launch.into_submission_parts();
        let boundary = core.boundary().cloned();
        Ok(DeviceSubmission::new(
            self.waiter.completion(queued.completions),
            core,
            workspace,
            TargetOutput {
                boundary,
                readout: queued.readout,
                readout_values: queued.readout_values,
                commits: queued.commits,
                landed_reads: queued.landed_reads,
                stored_reads: queued.stored_reads,
            },
        ))
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

#[cfg(test)]
mod head_element_tests {
    use super::HeadElement;

    #[test]
    fn a_bf16_head_rounds_to_nearest_even_and_keeps_nan() {
        assert_eq!(HeadElement::Bf16.publish(f32::from_bits(0x3f80_8000)).to_bits(), 0x3f80_0000);
        assert_eq!(HeadElement::Bf16.publish(f32::from_bits(0x3f81_8000)).to_bits(), 0x3f82_0000);
        // A NaN whose payload rounding would carry through the exponent
        // into the sign.
        for bits in [0x7fff_ffff_u32, 0xffff_ffff, 0x7f80_0001] {
            let published = HeadElement::Bf16.publish(f32::from_bits(bits));
            assert!(published.is_nan());
            assert_eq!(published.to_bits() >> 31, bits >> 31);
            assert_eq!(published.to_bits() & 0xffff, 0);
        }
        assert_eq!(HeadElement::Bf16.publish(f32::INFINITY), f32::INFINITY);
    }
}
